"""
Chat service — per-case AI conversation.

Each case has a persistent chat thread stored in the Message table.
The AI model receives:
  - A system prompt anchored to the case's latest complete analysis result
    (so the model already knows the problem type, severity, tools, steps)
  - The full message history for the case (user + assistant turns)
  - The new user message

This means the user never has to re-explain the problem — the model starts
with full repair context and can answer follow-up questions like
"what if I don't have a pipe wrench?" or "is this safe to do at night?".

Message storage:
  All messages (both user turns and assistant replies) are persisted to the
  Message table.  The full history is loaded on every call so the model
  maintains multi-turn context — the same way a training sequence uses all
  previous tokens to predict the next one.

Model:
  Uses CHAT_MODEL from environment (default: llama3.1:8b).
  Not a structured-output call — we want a conversational reply,
  so format="json" is NOT used here.

Graceful degradation:
  If no complete analysis exists for the case, the system prompt falls back
  to a generic home repair assistant persona.  The model can still help,
  it just won't have specific context about this case's images.
"""

import logging
import os
from typing import Optional

import ollama
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Analysis, Message, Session as RepairSession

log = logging.getLogger(__name__)

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
CHAT_MODEL       = os.getenv("CHAT_MODEL",       "llama3.1:8b")
CHAT_NUM_CTX     = int(os.getenv("CHAT_NUM_CTX",     "2048"))
CHAT_NUM_PREDICT = int(os.getenv("CHAT_NUM_PREDICT", "768"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

# The scope instruction appended to every system prompt — keeps the assistant
# from drifting into general-purpose chat unrelated to the repair case.
_REPAIR_SCOPE = (
    "Your role is strictly limited to helping with this repair/maintenance case. "
    "You may discuss: diagnosis, repair steps, tools, materials, costs, safety, "
    "difficulty, and when to call a professional. "
    "If the user asks something unrelated to repair or maintenance, politely "
    "acknowledge it and redirect them back to their repair question."
)

_ollama = ollama.AsyncClient(host=OLLAMA_BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

async def get_messages(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> list[Message]:
    """Return all messages for a case, oldest first."""
    await _require_owned_case(db, case_id, user_id)

    result = await db.execute(
        select(Message)
        .where(Message.session_id == case_id)
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def send_message(
    db: AsyncSession,
    case_id: int,
    user_id: int,
    content: str,
) -> Message:
    """
    Persist the user's message, call the AI, persist the reply, return the reply.

    Steps:
      1. Verify case ownership.
      2. Load the latest complete analysis (for system prompt context).
      3. Load the full message history.
      4. Persist the user's new message.
      5. Build the Ollama messages list (system + history + new user message).
      6. Call CHAT_MODEL.
      7. Persist and return the assistant reply.
    """
    await _require_owned_case(db, case_id, user_id)

    # Guard: chat requires at least one completed analysis.
    # This ensures the assistant always has repair context to work from,
    # and prevents the chat from being used as a general-purpose assistant
    # before the case has been properly diagnosed.
    analysis_exists = await db.execute(
        select(Analysis)
        .where(Analysis.session_id == case_id, Analysis.status == "complete")
        .limit(1)
    )
    if analysis_exists.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Please run analysis on this case before starting the chat.",
        )

    # 1. Build system prompt from the latest complete analysis
    system_prompt = await _build_system_prompt(db, case_id)

    # 2. Load message history
    history_result = await db.execute(
        select(Message)
        .where(Message.session_id == case_id)
        .order_by(Message.created_at.asc())
    )
    history = list(history_result.scalars().all())

    # 3. Persist user message before calling the model
    user_msg = Message(
        session_id=case_id,
        role="user",
        content=content,
        message_type="chat",
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # 4. Build Ollama messages list
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        ollama_messages.append({"role": msg.role, "content": msg.content})
    ollama_messages.append({"role": "user", "content": content})

    # 5. Call the chat model
    try:
        response = await _ollama.chat(
            model=CHAT_MODEL,
            messages=ollama_messages,
            options={
                "temperature":  0.6,
                "num_predict":  CHAT_NUM_PREDICT,
                "num_ctx":      CHAT_NUM_CTX,
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        reply_content = response.message.content.strip()
    except Exception as exc:
        log.error("Chat model call failed for case %d: %s", case_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI model unavailable: {type(exc).__name__}: {exc}",
        )

    # 6. Persist assistant reply
    assistant_msg = Message(
        session_id=case_id,
        role="assistant",
        content=reply_content,
        message_type="chat",
    )
    db.add(assistant_msg)
    await db.commit()
    await db.refresh(assistant_msg)

    return assistant_msg


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _require_owned_case(
    db: AsyncSession, case_id: int, user_id: int
) -> RepairSession:
    """Return the case if it belongs to this user, or raise 404."""
    result = await db.execute(
        select(RepairSession).where(
            RepairSession.id == case_id,
            RepairSession.user_id == user_id,
        )
    )
    case = result.scalar_one_or_none()
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )
    return case


async def _build_system_prompt(db: AsyncSession, case_id: int) -> str:
    """
    Build a case-aware system prompt from the latest complete analysis.

    If a complete analysis exists, the model receives full structured context:
      - What the problem is (problem_type, trade_category, severity)
      - What tools and materials the user has
      - What the repair steps are

    If no complete analysis exists yet, falls back to a generic assistant
    persona that can still answer general home repair questions.
    """
    result = await db.execute(
        select(Analysis)
        .where(
            Analysis.session_id == case_id,
            Analysis.status == "complete",
        )
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    analysis = result.scalar_one_or_none()

    if analysis is None or not analysis.structured_result:
        # Fallback — analysis exists (checked above) but structured_result is empty.
        # This can happen when the Stage 1 vision fallback ran.
        return (
            "You are Fixly, a repair assistant. "
            + _REPAIR_SCOPE + " "
            "The automated image analysis for this case was inconclusive. "
            "Ask the user to describe the problem in detail and guide them "
            "through diagnosis and repair steps based on their description. "
            "If a repair involves significant risk (electrical mains, structural, gas), "
            "always recommend consulting a qualified professional first."
        )

    sr = analysis.structured_result
    s1 = sr.get("stage_1", {})
    s2 = sr.get("stage_2", {})
    s3 = sr.get("stage_3", {})

    # Summarise the structured result into a readable context block
    steps_summary = ""
    for step in s3.get("steps", []):
        steps_summary += f"\n  Step {step.get('step_number')}: {step.get('title')}"

    # Build tools line — include optional tools if present
    tools_parts = s2.get("tools", [])
    optional_tools = s2.get("optional_tools", [])
    tools_line = ", ".join(tools_parts)
    if optional_tools:
        tools_line += f" | optional: {', '.join(optional_tools)}"

    materials_parts = s2.get("materials", [])
    optional_materials = s2.get("optional_materials", [])
    materials_line = ", ".join(materials_parts)
    if optional_materials:
        materials_line += f" | optional: {', '.join(optional_materials)}"

    # Include the user's original question if they provided one
    original_question = (analysis.text_context or "").strip()
    user_question_line = (
        f"User's original question: {original_question}\n"
        if original_question else ""
    )

    context = (
        f"{user_question_line}"
        f"Problem     : {s1.get('problem_type', 'Unknown')}\n"
        f"Trade       : {s1.get('trade_category', 'Unknown')}\n"
        f"Severity    : {s1.get('severity', 'Unknown')}\n"
        f"Tools       : {tools_line}\n"
        f"Materials   : {materials_line}\n"
        f"Difficulty  : {s3.get('difficulty_level', 'Unknown')}\n"
        f"Est. time   : {s2.get('estimated_time_minutes', '?')} minutes\n"
        f"Repair steps:{steps_summary}\n"
        f"Safety notes: {'; '.join(s3.get('safety_notes', []))}\n"
        f"When to call a pro: {s3.get('when_to_call_professional', '')}"
    )

    # Check if Stage 1 used the fallback (image analysis failed)
    s1_fallback = analysis.stage_outputs.get("stage_1_fallback", False)
    fallback_note = (
        "\nNote: the image could not be automatically analysed. "
        "The repair plan was generated from the user's text description only. "
        "Ask clarifying questions if the problem is unclear.\n"
        if s1_fallback else ""
    )

    severity = s1.get("severity", "medium")
    high_risk_note = ""
    if severity == "high":
        high_risk_note = (
            "\nThis case has HIGH severity. Prioritise safety — "
            "if there is any doubt about whether the repair is safe for a DIYer, "
            "clearly recommend calling a qualified professional.\n"
        )

    return (
        "You are Fixly, the repair assistant for this specific case.\n"
        + _REPAIR_SCOPE + "\n"
        + fallback_note
        + high_risk_note
        + "\nRepair case context:\n"
        + context
        + "\n\nAnswer follow-up questions accurately using the case context above. "
        "If the user had an original question (shown above), make sure your answers "
        "address it directly — do not ignore it or treat the chat as starting from scratch. "
        "Reference specific steps, tools, or materials by name when relevant. "
        "If the user describes new or worsening symptoms, acknowledge this and "
        "adjust your guidance — do not ignore signs that the situation has changed. "
        "If a repair involves electrical mains, gas, structural load, or significant "
        "water damage, explicitly recommend consulting a qualified tradesperson."
    )
