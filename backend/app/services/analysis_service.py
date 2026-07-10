"""
Three-stage AI analysis pipeline, all models served locally via Ollama.

Architecture
────────────
1. create_analysis()
   Called by the route handler. Validates preconditions, creates an Analysis
   DB row (status="pending"), then launches _run_pipeline() as a non-blocking
   asyncio background task. Returns immediately so the HTTP response is fast.

2. _run_pipeline()
   Runs independently of the request lifecycle with its own DB session.
   Executes stages 1→2→3 in sequence, writing progress after each stage so
   the frontend can poll and show labelled progress.

3. _stage_1/2/3_*()
   One async function per pipeline stage. Each calls its Ollama model,
   validates the JSON response against a Pydantic schema, and retries up to
   PIPELINE_RETRY_COUNT times. Graceful fallback for Stage 1: if vision
   fails all retries the pipeline continues from the user's text description.

Model assignments
─────────────────
  Stage 1  Qwen2.5-VL:3b       vision    image classification + observations
  Stage 2  Mistral:7b-instruct  text      tools (essential/optional), materials, gear
  Stage 3  Llama 3.1:8b         text      step-by-step repair guide, safety notes

Stage 2 output separates tools and materials into:
  - essential (must-have to complete the repair)
  - optional  (helpful but not required, shown as "Nice to have" in the UI)

Why asyncio.create_task instead of BackgroundTasks?
────────────────────────────────────────────────────
FastAPI's BackgroundTasks closes the request session when the request ends.
The pipeline runs for up to ~2 minutes and needs multiple DB writes (one per
stage). create_task() keeps the coroutine alive with its own session via
AsyncSessionLocal.

JSONB mutation tracking
───────────────────────
SQLAlchemy does not detect in-place dict mutations on JSONB columns.
Always reassign with the spread pattern:
    analysis.stage_outputs = {**analysis.stage_outputs, "key": value}
expire_on_commit=False in database.py ensures the object stays valid across
multiple commits within the same background task session.

num_ctx for Stage 1
───────────────────
Qwen2.5-VL encodes each 512px image as ~300-500 internal tokens.
System prompt (~200 tok) + image (~500 tok) + JSON response (~300 tok) = ~1000 min.
The old default of 512 caused mid-JSON truncation on every call.
STAGE_1_NUM_CTX=4096 gives headroom for multiple images and retries.
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import ollama
from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Analysis, Session as RepairSession, Upload
from app.services.clarification_reasoning import assess_clarification, finalize_context
from app.schemas.analysis import (
    AnalysisResult,
    AssessmentOutput,
    CareOutput,
    ClarificationOutput,
    OutOfScopeOutput,
    RepairStep,
    Stage1Output,
    Stage2Output,
    Stage3Output,
)

from app.utils.json_utils import (
    _clean_json,
    _heal_truncated_json,
    _normalise_dashes_deep,
    _scrub_stage3_prose,
)
from app.services.safety_routing_service import (
    _BASE_CONFIDENCE_FLOOR,
    _apply_decision_to_stage3,
    _apply_safety_net,
    _decide_actions,
    _determine_output_mode,
)
from app.utils.sanitization_utils import (
    _filter_answered_questions,
    _sanitize_stage_1,
)

log = logging.getLogger(__name__)

# ── Ollama configuration ───────────────────────────────────────────────────────
# All values are read from environment variables so you can switch models
# without touching code.  Defaults match the agreed model assignments.
import os

OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
STAGE_1_MODEL        = os.getenv("STAGE_1_MODEL",        "qwen2.5vl:7b")
STAGE_2_MODEL        = os.getenv("STAGE_2_MODEL",        "mistral:7b-instruct")
STAGE_3_MODEL        = os.getenv("STAGE_3_MODEL",        "llama3.1:8b")
PIPELINE_RETRY_COUNT = int(os.getenv("PIPELINE_RETRY_COUNT", "1"))

# Per-stage context window and output budgets, read from .env so they can be
# tuned per-machine without code changes.
#
# STAGE_1_NUM_CTX must be large enough to hold:
#   system prompt (~200 tok) + image tokens (~500-700 per image) + JSON response (~300 tok)
# 512 (the old default) caused every Stage 1 call to truncate mid-JSON.
STAGE_1_NUM_CTX     = int(os.getenv("STAGE_1_NUM_CTX",     "4096"))
STAGE_1_NUM_PREDICT = int(os.getenv("STAGE_1_NUM_PREDICT", "400"))
STAGE_2_NUM_CTX     = int(os.getenv("STAGE_2_NUM_CTX",     "2048"))
STAGE_2_NUM_PREDICT = int(os.getenv("STAGE_2_NUM_PREDICT", "512"))
STAGE_3_NUM_CTX     = int(os.getenv("STAGE_3_NUM_CTX",     "2048"))
STAGE_3_NUM_PREDICT = int(os.getenv("STAGE_3_NUM_PREDICT", "1024"))
CHAT_NUM_CTX        = int(os.getenv("CHAT_NUM_CTX",        "2048"))
CHAT_NUM_PREDICT    = int(os.getenv("CHAT_NUM_PREDICT",    "768"))

# How long Ollama keeps a model in RAM after its last use.
# "5m" keeps models warm for retries and follow-up chat calls.
# Set to "0" on RAM-constrained machines to unload immediately.
OLLAMA_KEEP_ALIVE   = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

# ── Feature flag: repeated-question filter ────────────────────────────────────
# When True, clarifying_questions that repeat a topic already present in the
# re-analysis text_context are dropped before _determine_output_mode runs.
# This prevents the pipeline from looping in clarification mode after the user
# has already answered a question Stage 1 is asking again.
#
# Disable instantly by setting FILTER_REPEATED_CLARIFICATION_QUESTIONS=false
# in backend/.env and restarting the backend.  No code change required.
#
# Default: true  — the filter is conservative (requires 2+ content-word overlap)
# and the alternative (leaving it off) is a demonstrably broken loop for burn-
# mark cases.  Set to false to revert to pre-filter behaviour.
_FILTER_REPEATED_CLARIFICATION_Q: bool = (
    os.getenv("FILTER_REPEATED_CLARIFICATION_QUESTIONS", "true")
    .strip().lower() == "true"
)


# Maximum number of clarification rounds before the pipeline falls back to
# verify_first.  Prevents infinite Q/A loops when Stage 1 keeps returning
# clarifying_questions despite receiving user answers.
# Default: 2  (most genuine ambiguities resolve in one or two exchanges).
_MAX_CLARIFICATION_ROUNDS: int = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))

# Upload directory, resolved the same way as uploads_service.py so paths match.
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"

# One shared async Ollama client, created once at module load, reused for all
# pipeline calls.  AsyncClient is safe to share across coroutines.
_ollama = ollama.AsyncClient(host=OLLAMA_BASE_URL)


def _log_gpu_vram(label: str) -> None:
    """Log total GPU VRAM state using pynvml (sees ALL processes, not just PyTorch)."""
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name   = pynvml.nvmlDeviceGetName(handle)
        total  = info.total / 1024 ** 3
        used   = info.used  / 1024 ** 3
        free   = info.free  / 1024 ** 3
        log.info(
            "VRAM [%s] gpu=%s total=%.1fGB used=%.1fGB free=%.1fGB",
            label, name, total, used, free,
        )
        pynvml.nvmlShutdown()
    except Exception as exc:
        log.info("VRAM [%s] pynvml unavailable: %s", label, exc)


async def _unload_ollama_models() -> None:
    """Unload all Ollama models from GPU VRAM before video inference.

    On a single GPU, Ollama's chat/image models (llama3.1, qwen2.5vl) compete
    directly with the video transformer model for VRAM. Sending a generate
    request with keep_alive=0 tells Ollama to evict any loaded model immediately,
    freeing VRAM for the video pipeline. Fails silently — a failed unload is not
    fatal, it just means video inference has less headroom.
    """
    _log_gpu_vram("before_ollama_unload")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            ps = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            if ps.status_code != 200:
                log.warning("video_pipeline: /api/ps returned %d", ps.status_code)
                return
            loaded = [m.get("name", "") for m in (ps.json().get("models") or [])]
            if not loaded:
                log.info("video_pipeline: no ollama models loaded in VRAM")
                return
            log.info("video_pipeline: unloading ollama models from VRAM: %s", loaded)
            for model_name in loaded:
                try:
                    await client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={"model": model_name, "keep_alive": 0},
                        timeout=10.0,
                    )
                    log.info("video_pipeline: unloaded %s", model_name)
                except Exception as exc:
                    log.warning("video_pipeline: failed to unload %s: %s", model_name, exc)
        import asyncio as _asyncio
        await _asyncio.sleep(2.0)   # give Ollama time to release GPU memory
    except Exception as exc:
        log.warning("video_pipeline: ollama unload failed (non-fatal): %s", exc)
    _log_gpu_vram("after_ollama_unload")


# ── Pipeline stage label constants ────────────────────────────────────────────
# Written to Analysis.pipeline_stage after each step.
# The frontend maps these to human-readable progress labels:
#   _S1_RUNNING  → "Reviewing your uploaded images…"
#   _S1_DONE     → "Problem identified"
#   _S2_RUNNING  → "Gathering tools and materials…"
#   _S2_DONE     → "Resources listed"
#   _S3_RUNNING  → "Preparing repair steps…"
#   _S3_DONE     → "Steps ready"
#   "complete"   → render the full result

_S1_RUNNING = "stage_1_running"
_S1_DONE    = "stage_1_complete"
_S2_RUNNING = "stage_2_running"
_S2_DONE    = "stage_2_complete"
_S3_RUNNING = "stage_3_running"
_S3_DONE    = "stage_3_complete"
_SA_RUNNING = "stage_a_running"   # assessment-first mode
_SA_DONE    = "stage_a_complete"
_SC_RUNNING = "stage_c_running"   # care mode
_SC_DONE    = "stage_c_complete"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

async def _build_manual_reanalysis_context(
    db: AsyncSession,
    case_id: int,
    caller_text: str | None,
) -> str | None:
    """
    Build cumulative Q/A text_context for a manual re-analyse triggered from
    the clarification textarea.

    The frontend only sends the textarea value.  This function reconstructs the
    full context that auto re-analysis would have built:
      - prior analysis.text_context  (earlier clarification rounds)
      - Q/A pairs from THIS round only (answers after the most-recent opener)
      - optional "Additional note" from the textarea

    To avoid duplicating earlier rounds, answer collection is anchored to the
    MOST RECENT clarification_opener — exactly like the auto re-analysis path in
    chat_service._check_auto_reanalyse.  Earlier rounds already live in the prior
    analysis.text_context, so re-collecting all answers across the case would
    double-count them.

    Returns caller_text unchanged when there are no clarification-answer messages
    for the current round (fresh analysis, or textarea-only re-analyse).
    """
    from app.models import Message  # local to avoid circular import risk

    # Fetch the latest completed clarification analysis for this case
    analysis_r = await db.execute(
        select(Analysis)
        .where(
            Analysis.session_id == case_id,
            Analysis.status == "complete",
        )
        .order_by(Analysis.id.desc())
    )
    latest_analysis = analysis_r.scalars().first()
    if latest_analysis is None:
        return caller_text  # no prior analysis — nothing to accumulate

    prior_result = (latest_analysis.structured_result or {})
    if prior_result.get("output_mode") != "clarification":
        return caller_text  # last analysis wasn't clarification — pass through

    # Anchor on the MOST RECENT clarification opener so we only gather answers
    # belonging to THIS round.  Earlier rounds are already in prior text_context.
    opener_r = await db.execute(
        select(Message)
        .where(
            Message.session_id == case_id,
            Message.message_type == "clarification_opener",
        )
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    opener = opener_r.scalar_one_or_none()

    # Collect this round's clarification-answer messages, oldest-first.
    # If an opener exists, scope to messages after it; otherwise (defensive)
    # fall back to all clarification-answer messages on the case.
    msg_q = (
        select(Message)
        .where(
            Message.session_id == case_id,
            Message.role == "user",
        )
        .order_by(Message.created_at.asc())
    )
    if opener is not None:
        msg_q = msg_q.where(Message.created_at > opener.created_at)
    msgs_r = await db.execute(msg_q)
    all_user_msgs = list(msgs_r.scalars().all())
    answer_msgs = [m for m in all_user_msgs if (m.msg_metadata or {}).get("clarification_answer")]

    if not answer_msgs:
        return caller_text  # no current-round answers — textarea-only or none

    # Delegate coverage, content-aware question/answer mapping, and cumulative
    # de-duplication to the shared clarification reasoning unit (the same unit the
    # automatic path will adopt). The DB fetch above stays here; the unit is pure.
    clarif = prior_result.get("clarification") or {}
    questions = list((clarif.get("clarifying_questions") or []))
    optional = list((clarif.get("optional_questions") or []))
    answer_texts = [m.content for m in answer_msgs]
    prior_ctx = (latest_analysis.text_context or "").strip()

    assessment = assess_clarification(
        required_questions=questions,
        answer_messages=answer_texts,
        prior_context=prior_ctx or None,
        optional_questions=optional,
    )
    # Final assembly is shared with the automatic path (finalize_context) so both
    # paths produce identical context for identical inputs. Manual re-analysis is a
    # deliberate override, so it passes the textarea note and still proceeds even
    # when coverage is incomplete; unmapped answers are preserved, never dropped.
    text_context = finalize_context(assessment, answer_texts, note=caller_text)

    log.info(
        "manual_reanalysis_context_built case=%d round_answers=%d mapped=%d "
        "ready=%s opener=%s prior_ctx=%s note=%s",
        case_id, len(answer_msgs), len(assessment.resolved_pairs),
        assessment.ready, "yes" if opener else "no",
        "yes" if prior_ctx else "no", "yes" if (caller_text or "").strip() else "no",
    )
    return text_context


async def create_analysis(
    db: AsyncSession,
    case_id: int,
    user_id: int,
    text_context: str | None = None,
) -> Analysis:
    """
    Validate preconditions, create an Analysis row, launch the background pipeline.

    Returns the Analysis row immediately (status="pending").
    The caller should tell the client to poll GET /analysis for progress.

    Preconditions enforced:
      - case exists and belongs to user
      - at least one photo has been uploaded
      - no pipeline is currently running for this case (prevents duplicate runs)
    """
    # 1. Verify ownership, raises 404 if case not found or not owned
    await _require_owned_case(db, case_id, user_id)

    # 2. Refuse if no photos exist, the vision stage needs at least one image
    uploads_result = await db.execute(
        select(Upload).where(Upload.session_id == case_id)
    )
    uploads = list(uploads_result.scalars().all())
    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Upload at least one photo before requesting analysis.",
        )

    # 3. Prevent duplicate concurrent runs
    running = await db.execute(
        select(Analysis).where(
            Analysis.session_id == case_id,
            Analysis.status == "running",
        )
    )
    if running.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An analysis is already running for this case. "
                   "Wait for it to complete, then re-trigger if needed.",
        )

    # 4a. On manual re-analyse from the clarification textarea the frontend
    #     sends only the textarea value — not the prior Q/A conversation.
    #     Reconstruct the full cumulative context so Stage 1 sees all answers.
    #     Skip when the caller already passed a pre-built Q/A block (auto
    #     re-analysis path, which starts with "User answers").
    _is_prebuilt_qa = bool(
        text_context
        and (
            text_context.strip().startswith("User answers")
            or "Q1:" in text_context
        )
    )
    if not _is_prebuilt_qa:
        text_context = await _build_manual_reanalysis_context(db, case_id, text_context)

    # 4. Create the Analysis row (status="pending", no results yet)
    analysis = Analysis(
        session_id=case_id,
        status="pending",
        stage_outputs={},   # always initialise to empty dict, not None
        pipeline_stage=None,
        text_context=text_context,
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    # 4a. Route video uploads through the dedicated video pipeline.
    #     Image cases pass through to _run_pipeline unchanged.
    has_video = any(getattr(u, "source_type", "image") == "video" for u in uploads)
    if has_video:
        video_upload = next(
            u for u in uploads if getattr(u, "source_type", "image") == "video"
        )
        asyncio.create_task(
            _run_video_pipeline(
                analysis.id, case_id, video_upload.storage_path, text_context
            )
        )
        return analysis

    # 5. Extract just the storage paths, do not pass ORM objects across session
    #    boundaries.  The request session closes when this function returns; the
    #    background task creates its own session and only needs these strings to
    #    locate the image files on disk.
    upload_paths = [u.storage_path for u in uploads]

    # 6. Launch pipeline, non-blocking, runs concurrently with other requests
    asyncio.create_task(_run_pipeline(analysis.id, case_id, upload_paths, text_context))

    return analysis


async def get_latest_analysis(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> Optional[Analysis]:
    """
    Return the most recent analysis for a case, or None if none exists.

    The frontend calls this endpoint on a polling interval while status == "running".
    The pipeline_stage field updates after each stage so granular progress is visible.
    """
    await _require_owned_case(db, case_id, user_id)

    result = await db.execute(
        select(Analysis)
        .where(Analysis.session_id == case_id)
        .options(selectinload(Analysis.session))   # needed for Analysis.hazard_ack property
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


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


def _load_images_as_base64(storage_paths: list[str]) -> list[str]:
    """
    Read image files from UPLOAD_DIR, resize to max 1024 px, compress to JPEG,
    and return as base64-encoded strings for the Ollama vision API.

    Resizing before encoding reduces the base64 payload sent to the vision
    model by 60-80% for typical phone photos, cutting Stage 1 inference time
    noticeably with no meaningful loss in diagnostic accuracy, home repair
    problems are large, visible features that don't need full-resolution images.

    Compression pipeline per image:
      1. Open with Pillow.
      2. Convert non-RGB modes (RGBA, P, etc.) to RGB so JPEG save works.
      3. thumbnail(1024, 1024), preserves aspect ratio, in-place.
      4. Save to BytesIO as JPEG quality=82.
      5. base64-encode the compressed bytes.

    Falls back to raw bytes if Pillow fails for any reason (e.g. unsupported
    format), so the pipeline still runs even on exotic image types.
    """
    import io
    try:
        from PIL import Image as _PILImage
        _pil_available = True
    except ImportError:
        _pil_available = False
        log.warning("Pillow not installed, images will be sent at full resolution. "
                    "Run: pip install Pillow")

    # 512px is sufficient for home repair diagnosis, the damage features are large
    # and visible at low resolution.  Smaller images mean fewer ViT patch tokens,
    # which is the primary inference-time bottleneck for Qwen2.5-VL on CPU.
    # Changing this to 768 or 1024 improves fine-detail accuracy at the cost of
    # ~2x longer Stage 1 inference time.
    _MAX_DIM = int(os.getenv("STAGE_1_IMAGE_MAX_DIM", "512"))

    result = []
    for path in storage_paths:
        full = UPLOAD_DIR / path
        if not full.exists():
            log.warning("Image not found on disk: %s", full)
            continue
        try:
            raw_bytes = full.read_bytes()

            if _pil_available:
                try:
                    img = _PILImage.open(io.BytesIO(raw_bytes))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    # thumbnail is in-place and preserves aspect ratio
                    if max(img.width, img.height) > _MAX_DIM:
                        img.thumbnail((_MAX_DIM, _MAX_DIM), _PILImage.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=82, optimize=True)
                    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                    log.debug(
                        "Image %s: %dx%d → JPEG %d KB",
                        path, img.width, img.height, len(buf.getvalue()) // 1024,
                    )
                    result.append(encoded)
                    continue
                except Exception as pil_exc:
                    log.warning("Pillow resize failed for %s (%s), using raw bytes",
                                path, pil_exc)

            # Fallback: send raw bytes without resize
            result.append(base64.b64encode(raw_bytes).decode("utf-8"))

        except OSError as exc:
            log.warning("Could not read image %s: %s", full, exc)
    return result


async def _call_stage(
    model: str,
    messages: list[dict],
    schema_class,
    stage_name: str,
    extra_options: dict | None = None,
) -> object:
    """
    Call an Ollama model and parse + validate the JSON response.

    This is the single point where all Ollama HTTP calls happen.
    Centralising here keeps the stage functions focused on prompt construction
    and keeps retry logic, JSON cleaning, and error handling in one place.

    Flow per attempt:
      1. Call Ollama with format="json" (JSON output mode)
      2. Strip any markdown fences from the response
      3. Parse with json.loads()
      4. Validate by constructing the Pydantic schema_class
      5. On JSONDecodeError or ValidationError: wait and retry
      6. On connection/timeout error: raise immediately (no point retrying)

    The retry pattern mirrors the idea from training: if the output doesn't
    satisfy the schema constraint, the system asks the model again rather than
    propagating bad data to the next stage.

    Parameters
    ──────────
    model        : Ollama model name (e.g. "mistral:latest")
    messages     : list of chat message dicts (role + content, optionally images)
    schema_class : Pydantic model class, used to validate and construct the result
    stage_name   : human-readable label for log messages
    """
    last_error = None
    last_raw   = ""
    # Two independent retry budgets:
    #   - validation_attempts: bad JSON / schema mismatch (bounded by PIPELINE_RETRY_COUNT).
    #   - transient_retries: the Ollama model runner crashing ("model runner has
    #     unexpectedly stopped" / status 500) or a dropped connection. This happens
    #     intermittently under memory pressure when the heavy Stage-1 7B vision model
    #     is swapped out for the next stage's model. The fix is simply to wait a few
    #     seconds for Ollama to respawn the runner / free VRAM, then retry, so one
    #     hiccup never fails the whole analysis (important for a live demo).
    validation_attempts    = 0
    transient_retries_left = 3

    while True:
        try:
            options = {
                "temperature": 0.1,   # deterministic → schema-compliant
                "num_predict": 512,   # per-stage default; overridden via extra_options
            }
            if extra_options:
                options.update(extra_options)

            response = await _ollama.chat(
                model=model,
                messages=messages,
                format="json",
                options=options,
                keep_alive=OLLAMA_KEEP_ALIVE,   # keep model warm for retries + follow-up
            )
            raw      = response.message.content
            last_raw = raw
            log.debug("%s raw (%d chars): %s…", stage_name, len(raw), raw[:120])

            cleaned = _clean_json(raw)
            data    = _normalise_dashes_deep(json.loads(cleaned))
            return schema_class(**data)

        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning(
                "%s validation attempt %d/%d, parse error: %s | raw[:80]=%r",
                stage_name, validation_attempts + 1, PIPELINE_RETRY_COUNT + 1,
                exc, last_raw[:80],
            )

            # Before retrying, attempt to heal a truncated response.
            # If healing succeeds we return immediately without using another
            # model call, cheaper and faster than a full retry.
            if last_raw:
                try:
                    healed = _heal_truncated_json(_clean_json(last_raw))
                    data   = _normalise_dashes_deep(json.loads(healed))
                    result = schema_class(**data)
                    log.info("%s, healed truncated JSON successfully", stage_name)
                    return result
                except Exception:
                    pass    # healing failed, proceed to retry

            validation_attempts += 1
            if validation_attempts <= PIPELINE_RETRY_COUNT:
                await asyncio.sleep(1.0)
                continue
            raise ValueError(
                f"{stage_name} failed after {validation_attempts} attempts. "
                f"Last error: {type(last_error).__name__}: {last_error}"
            ) from exc

        except Exception as exc:
            # Is this a transient Ollama/connection failure worth retrying?
            msg = str(exc).lower()
            transient = any(s in msg for s in (
                "unexpectedly stopped", "resource limit", "status code: 500",
                "status code: 502", "status code: 503", "connection",
                "timed out", "timeout", "eof occurred", "disconnect",
                "remoteprotocol",
            ))
            if transient and transient_retries_left > 0:
                transient_retries_left -= 1
                last_error = exc
                log.warning(
                    "%s, transient Ollama error, backing off 4s then retrying "
                    "(%d retr%s left): %s",
                    stage_name, transient_retries_left,
                    "y" if transient_retries_left == 1 else "ies", exc,
                )
                await asyncio.sleep(4.0)   # let Ollama respawn the runner / free VRAM
                continue
            raise RuntimeError(
                f"Ollama call failed for {stage_name} (model={model}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def _run_video_pipeline(
    analysis_id: int,
    case_id: int,
    video_storage_path: str,
    text_context: str | None = None,
) -> None:
    """
    Background task: run video Stage 1 analysis then delegate to _run_pipeline.

    Sequence
    ─────────
    1. Open a short-lived DB session; mark analysis running / S1_RUNNING so
       the frontend sees progress immediately (inference can take 30-90 s).
    2. Close the session before inference so the DB connection is not held open
       during the long blocking call.
    3. Call analyse_video() via the video_analysis_service (runs in a thread pool).
    4. Apply conservative video safety adjustments to the VideoStage1Output.
    5. Delegate to _run_pipeline() with the precomputed Stage 1 result.
       All routing (assessment_first / clarification / verify_first / diy / care /
       out_of_scope), Stage 2, Stage 3, and Stage A run identically to image cases.

    All failure paths write analysis.status = "failed" and exit cleanly — no
    analysis is ever left stuck in S1_RUNNING or partially completed states.
    """
    from app.services.video_analysis_service import (  # lazy — avoids startup load
        analyse_video,
        VideoAnalysisError,
        _apply_video_safety_adjustments,
        VIDEO_MODEL_LOCAL,
        VIDEO_MODEL_CLOUD,
        VIDEO_INFERENCE_MODE,
    )

    video_path = UPLOAD_DIR / video_storage_path

    # ── 1. Mark running so the frontend shows progress immediately ────────────
    async with AsyncSessionLocal() as db:
        analysis = await db.get(Analysis, analysis_id)
        if analysis is None:
            return  # row deleted between creation and task start

        case_row = await db.get(RepairSession, case_id)
        analysis.status = "running"
        if case_row is not None:
            case_row.status = "running"
        analysis.pipeline_stage = _S1_RUNNING
        await db.commit()
    # Session closed — DB connection released for the duration of inference.

    # ── 1b. Unload Ollama models from GPU before video inference ─────────────
    # Ollama keeps chat/image models in VRAM for up to 5 min. On a single GPU
    # they compete directly with the video transformer model. Unloading them
    # first ensures the full VRAM budget is available for video inference.
    await _unload_ollama_models()

    # ── 2. Video Stage 1 inference ────────────────────────────────────────────
    _t_s1 = time.monotonic()
    try:
        s1 = await analyse_video(video_path, description=text_context)

    except VideoAnalysisError as ve:
        log.error(
            "video_pipeline_s1_failed analysis_id=%d reason=%s", analysis_id, ve
        )
        async with AsyncSessionLocal() as db:
            analysis = await db.get(Analysis, analysis_id)
            case_row = await db.get(RepairSession, case_id)
            if analysis is not None:
                analysis.status = "failed"
                analysis.validation_issues = f"Video analysis failed: {ve}"
                analysis.pipeline_stage = None
                if case_row is not None:
                    case_row.status = "failed"
                await db.commit()
        return

    except Exception as exc:
        log.error(
            "video_pipeline_s1_unexpected analysis_id=%d error=%s",
            analysis_id, exc, exc_info=True,
        )
        async with AsyncSessionLocal() as db:
            analysis = await db.get(Analysis, analysis_id)
            case_row = await db.get(RepairSession, case_id)
            if analysis is not None:
                analysis.status = "failed"
                analysis.validation_issues = (
                    f"Video analysis failed: {type(exc).__name__}: {exc}"
                )
                analysis.pipeline_stage = None
                if case_row is not None:
                    case_row.status = "failed"
                await db.commit()
        return

    _ms_s1 = int((time.monotonic() - _t_s1) * 1000)

    # ── 3. Video-specific safety adjustments ──────────────────────────────────
    _adj_notes = _apply_video_safety_adjustments(s1)
    if _adj_notes:
        log.info(
            "video_safety_adjustments analysis_id=%d: %s",
            analysis_id, "; ".join(_adj_notes),
        )
    else:
        log.info(
            "video_safety_adjustments analysis_id=%d: none fired", analysis_id
        )

    # Determine the active model name for display in analysis.model
    _s1_model = (
        VIDEO_MODEL_CLOUD if VIDEO_INFERENCE_MODE == "cloud" else VIDEO_MODEL_LOCAL
    )

    log.info(
        "video_pipeline_s1_complete analysis_id=%d ms_s1=%d model=%s "
        "confidence=%d risk=%s diy=%s motion=%s brief_hazard=%s",
        analysis_id, _ms_s1, _s1_model,
        s1.confidence, s1.risk_level, s1.diy_suitability,
        s1.motion_detected, s1.brief_hazard_observed,
    )

    # ── 4. Delegate to the existing pipeline with precomputed Stage 1 ─────────
    # _run_pipeline will skip its own Stage 1 image call and proceed directly
    # to _sanitize_stage_1 → _apply_safety_net → _decide_actions →
    # _determine_output_mode → the appropriate output mode branch.
    await _run_pipeline(
        analysis_id,
        case_id,
        [],               # upload_paths unused when _precomputed_s1 is provided
        text_context,
        _precomputed_s1=s1,
        _precomputed_ms_s1=_ms_s1,
        _precomputed_s1_model=_s1_model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR  (unchanged from Session 1 except provider/model fields)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_pipeline(
    analysis_id: int,
    case_id: int,
    upload_paths: list[str],
    text_context: str | None = None,
    _precomputed_s1: "Stage1Output | None" = None,
    _precomputed_ms_s1: int = 0,
    _precomputed_s1_model: str = "",
) -> None:
    """
    Execute the 3-stage analysis pipeline as a background task.

    Each stage:
      1. Updates pipeline_stage in the DB (→ frontend can show labelled progress)
      2. Calls the stage function (real Ollama call in Session 2)
      3. The stage function validates the output against the Pydantic schema
      4. Writes the result to stage_outputs in the DB
      5. Updates pipeline_stage to "{stage}_complete" before moving on

    If any stage raises an unhandled exception:
      - status is set to "failed"
      - validation_issues stores the error string
      - the pipeline stops, it does not try to continue with partial results

    This function creates its own DB session.  The analysis_id integer is the
    only thing that crosses the session boundary safely.
    """
    start = time.monotonic()

    async with AsyncSessionLocal() as db:
        case_row = None   # initialised here so the except block can always reference it
        try:
            # Load the analysis row and its parent case in this new session
            analysis = await db.get(Analysis, analysis_id)
            if analysis is None:
                return  # row was deleted between creation and pipeline start

            # Load case_row early so we can mirror analysis status throughout.
            # This keeps Session.status in sync so the My Cases status filter works.
            case_row = await db.get(RepairSession, case_id)

            # Mark running, mirror to case so the Cases list shows "Analysing"
            analysis.status = "running"
            if case_row is not None:
                case_row.status = "running"
            await db.commit()

            # Labels for model/provider fields — differ between image and video paths
            _s1_label    = _precomputed_s1_model or STAGE_1_MODEL
            _s1_provider = "huggingface" if _precomputed_s1 is not None else "ollama"

            # ── STAGE 1, Image / Problem Analysis ───────────────────────────
            analysis.pipeline_stage = _S1_RUNNING
            await db.commit()

            _s1_fallback = False
            if _precomputed_s1 is not None:
                # Video path: Stage 1 was completed by _run_video_pipeline.
                # Video-specific safety adjustments were already applied there.
                s1 = _precomputed_s1
                _ms_s1 = _precomputed_ms_s1
                log.info(
                    "Stage 1 precomputed (video path) in %.1fs model=%s",
                    _ms_s1 / 1000, _s1_label,
                )
            else:
                # Image path: call the vision model via Ollama (unchanged).
                _t_s1 = time.monotonic()
                try:
                    s1: Stage1Output = await _stage_1_analyse(upload_paths, text_context)
                except RuntimeError as _s1_exc:
                    # RuntimeError = Ollama connection/OOM/server error.
                    # These are NOT parse failures, retrying immediately will not help
                    # and continuing with fake Stage1Output produces garbage downstream.
                    # Fail fast with a clear message so the user knows to retry.
                    log.error(
                        "Stage 1 runtime failure (OOM or Ollama error), failing pipeline. Error: %s",
                        _s1_exc,
                    )
                    raise  # re-raise → pipeline catches it in the outer except and marks failed
                except ValueError as _s1_exc:
                    # ValueError = all retries exhausted on parse/validation errors.
                    # Only fall back to text context if the user provided one, that gives
                    # Stages 2+3 something real to work from rather than a generic placeholder.
                    if text_context and text_context.strip():
                        log.warning(
                            "Stage 1 parse failure, falling back to user text context. Error: %s",
                            _s1_exc,
                        )
                        s1 = Stage1Output(
                            problem_type=text_context.strip(),
                            trade_category="other",
                            severity="medium",
                            observations=[
                                "Image could not be automatically analysed, using your description instead.",
                                "For a better diagnosis, try uploading a clearer or wider photo.",
                            ],
                        )
                        _s1_fallback = True
                    else:
                        log.error(
                            "Stage 1 failed and no text context provided, failing pipeline. Error: %s",
                            _s1_exc,
                        )
                        raise  # no text context to fall back to, fail the pipeline
                _ms_s1 = int((time.monotonic() - _t_s1) * 1000)
                log.info("Stage 1 in %.1fs%s", _ms_s1 / 1000,
                         " [fallback]" if _s1_fallback else "")

            # ── Stage 1 reliability pass (Priority 5) ─────────────────────────
            # De-dupe echoey observations/evidence and reconcile self-reported
            # confidence with the model's own uncertainty BEFORE it can influence
            # the gate or propagate into Stages 2 and 3. Grounded in real cases
            # #54 (triple-duplicated observation) and #56/#57 (overconfidence).
            _s1_fixes = _sanitize_stage_1(s1, text_context=text_context)
            if _s1_fixes:
                log.info("Stage 1 reliability pass: %s", "; ".join(_s1_fixes))

            # ── Repeated-question filter (feature-flagged) ────────────────────
            # Stage 1 sometimes re-asks a question whose topic was already covered
            # in a previous clarification round.  When the flag is on, drop those
            # repeats before _determine_output_mode sees the list so the pipeline
            # can advance rather than loop.
            #
            # Guards: flag on AND text_context present AND Stage 1 has questions.
            # A missing or non-Q/A text_context causes _filter_answered_questions
            # to return the list unchanged, so first-pass analyses are unaffected.
            #
            # Disable: set FILTER_REPEATED_CLARIFICATION_QUESTIONS=false in .env
            # and restart the backend.
            if _FILTER_REPEATED_CLARIFICATION_Q and text_context and s1.clarifying_questions:
                _kept_qs, _dropped_qs = _filter_answered_questions(
                    s1.clarifying_questions, text_context
                )
                if _dropped_qs:
                    log.info(
                        "filter_answered_questions: dropped %d repeated question(s): %s",
                        len(_dropped_qs),
                        "; ".join(q[:70] for q in _dropped_qs),
                    )
                    s1.clarifying_questions = _kept_qs

            # ── Thin one-way safety net ───────────────────────────────────────
            # For a few inherently dangerous trades, raise the model's risk signals
            # to a cautious floor if it under-called them. Never lowers a signal.
            # Runs BEFORE the decision + the gate so everything downstream sees the
            # floored signals. The model's reasoning still drives everything else.
            _net_notes = _apply_safety_net(s1)
            if _net_notes:
                log.info("Safety net raised signals (%s): %s", s1.trade_category, "; ".join(_net_notes))

            # Derive the pipeline reactions purely from the (model-decided, possibly
            # net-raised) signals, no category logic in here.
            _decision = _decide_actions(s1)

            analysis.stage_outputs = {
                **analysis.stage_outputs,
                "stage_1": s1.model_dump(),
                **({"stage_1_fallback": True} if _s1_fallback else {}),
                **({"stage_1_sanitised": _s1_fixes} if _s1_fixes else {}),
                "risk_signals": {
                    "risk_level": s1.risk_level,
                    "hidden_damage_risk": s1.hidden_damage_risk,
                    "diy_suitability": s1.diy_suitability,
                    "professional_assessment_recommended": s1.professional_assessment_recommended,
                    "safety_concerns": s1.safety_concerns,
                    "assessment_first": _decision["assessment_first"],
                    "verify_first": _decision["verify_first"],
                    **({"safety_net_raised": _net_notes} if _net_notes else {}),
                },
            }
            analysis.pipeline_stage = _S1_DONE
            await db.commit()

            # ── Clarify gate (model-led, generic floor) ───────────────────────
            # Ask the user for more when the MODEL is not confident enough or it
            # asked to clarify. The floor is a single generic value for every
            # trade, the model, not a category table, decides danger (handled by
            # the assessment-first reaction after Stage 3). 75 is used because the
            # vision model tends to emit 60-70 even on featureless/dark images.
            #
            # Video path uses a lower floor (65) because the local Qwen2.5-VL
            # model consistently rates clear repair videos at 65-75. The video
            # safety adjustment rules in video_analysis_service.py already clear
            # needs_clarification and clarification_request when the case is
            # low-risk, so the gate still protects against genuinely uncertain clips.
            _confidence_floor = (
                max(_BASE_CONFIDENCE_FLOOR - 10, 65)
                if _precomputed_s1 is not None
                else _BASE_CONFIDENCE_FLOOR
            )
            # High-risk cases must route to assessment_first, not needs_clarification.
            # A clarification question ("Is the boiler still on?") does not change
            # the safety recommendation ("call a professional").
            #
            # Video path: risk_level alone drives escalation — the local Qwen2.5-VL
            # model over-sets professional_assessment_recommended for medium-risk tasks
            # (drain blockages, dripping taps, peeling wallpaper) because the system
            # prompt says "if in doubt, recommend professional assessment."  Trusting
            # par=True at medium risk on the video path causes routine maintenance to
            # force assessment_first.  risk_level is a more reliable signal because
            # the model only rates high/critical when it observes a specific danger.
            # The post-processing rule in video_analysis_service.py also corrects par
            # for plumbing/decorating at medium risk with no serious hazard evidence.
            #
            # Image path keeps the combined gate (par OR high/critical) because the
            # image model does not show the same systematic over-escalation pattern.
            _is_video_path = _precomputed_s1 is not None
            _force_assessment_first = (
                getattr(s1, "risk_level", "") in ("high", "critical")
                if _is_video_path
                else (
                    getattr(s1, "professional_assessment_recommended", False) is True
                    or getattr(s1, "risk_level", "") in ("high", "critical")
                )
            )
            # Video path: nc=True does not fire the early gate on its own.
            # The video model sets needs_clarification=True when it has specific
            # diagnostic questions (duration, extent, damage type) even when the
            # video quality is adequate and Stage 1 is complete.  Those questions
            # belong in Stage C (clarification mode) so the chat opener, Q&A flow,
            # and auto-reanalysis work exactly as they do for image cases.
            # The early gate is reserved for genuinely low-confidence video (conf
            # below floor), where asking the user to re-upload or describe more is
            # the only meaningful action.
            # Image path: unchanged — nc=True often signals poor image quality.
            _nc_triggers_early_gate = (
                False if _is_video_path else s1.needs_clarification
            )
            if not _s1_fallback and not _force_assessment_first and (
                _nc_triggers_early_gate or s1.confidence < _confidence_floor
            ):
                log.info(
                    "Stage 1 confidence=%d (floor=%d, trade=%s) needs_clarification=%s, "
                    "stopping pipeline, asking user for clarification.",
                    s1.confidence, _confidence_floor, s1.trade_category, s1.needs_clarification,
                )
                # Prefer the model's own clarifying questions, then its clarification
                # request, then a sensible generic fallback.
                if s1.clarifying_questions:
                    clarification_msg = (
                        "I need a bit more to be sure before advising. "
                        + " ".join(q.strip().rstrip("?") + "?" for q in s1.clarifying_questions[:3])
                    )
                elif s1.clarification_request.strip():
                    clarification_msg = s1.clarification_request.strip()
                else:
                    clarification_msg = (
                        "The photo is not clear enough to identify a specific home repair problem. "
                        "Please upload a closer, well-lit photo of the damaged area, "
                        "or describe the issue in the text field above."
                    )
                analysis.status = "needs_clarification"
                analysis.pipeline_stage = "needs_clarification"
                analysis.validation_issues = clarification_msg
                if case_row is not None:
                    case_row.status = "needs_clarification"
                await db.commit()
                return  # stop, do not run Stage 2 or 3

            # Auto-title: overwrite the placeholder "Diagnosing…" with the AI result
            if case_row is not None:
                case_row.title    = s1.problem_type
                case_row.category = s1.trade_category
                await db.commit()

            # ── Output-mode branch ────────────────────────────────────────────
            # _determine_output_mode selects the pipeline path from Stage 1 signals.
            # The current implementation branches on assessment_first (Stage A) vs
            # the full DIY path (Stage 2 + Stage 3). Future steps add branches for
            # clarification (step 4) and care (step 3).
            _mode = _determine_output_mode(s1, text_context=text_context)
            log.info("Output mode: %s (trade=%s risk=%s diy=%s prof=%s hazards=%d q=%d)",
                     _mode, s1.trade_category, s1.risk_level, s1.diy_suitability,
                     s1.professional_assessment_recommended, len(s1.safety_concerns),
                     len(s1.clarifying_questions))
            analysis.stage_outputs = {
                **analysis.stage_outputs,
                "output_mode": _mode,
            }
            await db.commit()

            _ms_total = 0   # initialised here so finalise block always has it

            if _mode == "assessment_first":
                # ── STAGE A — Assessment / escalation output ──────────────────
                # Skip Stage 2 and Stage 3 entirely. The model's role is now
                # stabilisation, safety, and professional escalation — not repair.
                analysis.pipeline_stage = _SA_RUNNING
                await db.commit()

                _t_sa = time.monotonic()
                stage_a: AssessmentOutput = await _stage_A_assessment(s1)
                _ms_sa = int((time.monotonic() - _t_sa) * 1000)
                log.info("Stage A (assessment) complete in %.1fs", _ms_sa / 1000)

                analysis.stage_outputs = {
                    **analysis.stage_outputs,
                    "stage_a": stage_a.model_dump(),
                }
                analysis.pipeline_stage = _SA_DONE
                await db.commit()

                # ── Finalise assessment-first ──────────────────────────────────
                combined = AnalysisResult(
                    output_mode="assessment_first",
                    stage_1=s1,
                    assessment=stage_a,
                    # stage_2 and stage_3 intentionally omitted (None)
                )
                _ms_total = int((time.monotonic() - start) * 1000)
                analysis.structured_result  = combined.model_dump()
                analysis.status             = "complete"
                analysis.pipeline_stage     = "complete"
                if case_row is not None:
                    case_row.status = "complete"
                    case_row.clarification_round = 0
                analysis.processing_time_ms = _ms_total
                analysis.provider           = _s1_provider
                analysis.model              = f"{_s1_label} | {STAGE_3_MODEL}(A)"
                analysis.validation_issues  = (
                    f'{{"stage_1_ms":{_ms_s1},"stage_a_ms":{_ms_sa},"total_ms":{_ms_total}}}'
                )
                log.info("Pipeline complete (assessment_first), S1:%ds SA:%ds total:%ds",
                         _ms_s1 // 1000, _ms_sa // 1000, _ms_total // 1000)
                await db.commit()

            elif _mode == "out_of_scope":
                # ── Out-of-scope — active emergency OR not a repair image ──────
                # Stage 1's scope decision short-circuited here. Run NO further
                # stages (no Stage A/2/3): no repair plan, tools, materials, steps,
                # or trade-professional referral. Build a deterministic guidance
                # object from the scope value — no extra model call.
                _scope = (s1.scope or "not_repair")
                _is_emergency = (_scope == "emergency")
                if _is_emergency:
                    _oos = OutOfScopeOutput(
                        scope=_scope,
                        is_emergency=True,
                        headline="This looks like an active emergency",
                        guidance=(
                            "This appears to be an active emergency. Move away from danger, "
                            "evacuate if necessary, and contact emergency services. Do not "
                            "attempt a repair while the hazard is active."
                        ),
                    )
                else:
                    _oos = OutOfScopeOutput(
                        scope=_scope,
                        is_emergency=False,
                        headline="No repair issue identified",
                        guidance=(
                            "Knitra could not identify an in-scope household repair issue from "
                            "this image. Please upload a clear image of the damaged area or "
                            "repair concern."
                        ),
                    )
                combined = AnalysisResult(
                    output_mode="out_of_scope",
                    stage_1=s1,
                    out_of_scope=_oos,
                    # stage_2 / stage_3 / assessment intentionally omitted (None)
                )
                _ms_total = int((time.monotonic() - start) * 1000)
                analysis.structured_result  = combined.model_dump()
                analysis.status             = "complete"
                analysis.pipeline_stage     = "complete"
                if case_row is not None:
                    case_row.status = "complete"
                    case_row.clarification_round = 0
                analysis.processing_time_ms = _ms_total
                analysis.provider           = _s1_provider
                analysis.model              = f"{_s1_label} (out_of_scope/{_scope})"
                analysis.validation_issues  = (
                    f'{{"stage_1_ms":{_ms_s1},"total_ms":{_ms_total},'
                    f'"mode":"out_of_scope","scope":"{_scope}"}}'
                )
                log.info("Pipeline complete (out_of_scope), scope=%s S1:%ds total:%ds",
                         _scope, _ms_s1 // 1000, _ms_total // 1000)
                await db.commit()

            elif _mode == "clarification":
                # ── Clarification mode — surface questions, skip Stage 2 + 3 ──
                # Increment clarification_round.  If the limit is reached, fall
                # back to verify_first so the loop can't run forever.
                _current_round = (case_row.clarification_round if case_row is not None else 0)
                _new_round     = _current_round + 1
                if case_row is not None:
                    case_row.clarification_round = _new_round

                if _new_round > _MAX_CLARIFICATION_ROUNDS:
                    # ── Bounded clarification fallback → verify_first ──────────
                    log.warning(
                        "clarification_round=%d > MAX=%d — forcing verify_first (case=%d)",
                        _new_round, _MAX_CLARIFICATION_ROUNDS, case_id,
                    )
                    # Use clarification mode with no questions, flagged inconclusive
                    # so the frontend shows a distinct "could not reach a confident
                    # assessment" banner instead of a normal needs-answers card.
                    # Guidance goes into what_is_unclear (a rendered field); the
                    # frontend does NOT render why_it_matters, so it must not be the
                    # only place the guidance lives.
                    _vf_clarif = ClarificationOutput(
                        what_is_known=(
                            f"{s1.problem_type}: " + "; ".join(s1.observations[:3])
                            if s1.observations else s1.problem_type
                        ),
                        what_is_unclear=[
                            "Knitra could not reach a confident assessment from the "
                            "available photos and answers.",
                            "No more clarification questions are being asked right now.",
                            "Please provide clearer, well-lit photos from another angle, "
                            "or have the issue checked in person before starting work.",
                        ],
                        clarifying_questions=[],
                        optional_questions=[],
                        why_it_matters=(
                            "Please provide clearer photos or verify the issue in person "
                            "before starting work."
                        ),
                        inconclusive=True,
                    )
                    combined = AnalysisResult(
                        output_mode="clarification",
                        stage_1=s1,
                        clarification=_vf_clarif,
                    )
                    _ms_total = int((time.monotonic() - start) * 1000)
                    analysis.structured_result  = combined.model_dump()
                    analysis.status             = "complete"
                    analysis.pipeline_stage     = "complete"
                    if case_row is not None:
                        # status stays needs_info so the UI stays in clarif mode;
                        # round is reset so future user action starts clean.
                        case_row.status = "needs_info"
                        case_row.clarification_round = 0
                    analysis.processing_time_ms = _ms_total
                    analysis.provider           = _s1_provider
                    analysis.model              = f"{_s1_label} (verify_first_fallback)"
                    analysis.validation_issues  = (
                        f'{{"stage_1_ms":{_ms_s1},"total_ms":{_ms_total},'
                        f'"mode":"clarification","reason":"max_clarification_rounds"}}'
                    )
                    log.info("Pipeline complete (verify_first/fallback), round=%d", _new_round)
                    await db.commit()

                else:
                    # ── Normal clarification path ─────────────────────────────
                    # Build ClarificationOutput DETERMINISTICALLY from Stage 1 data.
                    # No extra model call — Stage 1 already identified the unknowns.
                    _unc_lines = [l.strip() for l in (s1.uncertainty_notes or "").split(".")
                                  if l.strip() and len(l.strip()) > 10]
                    _what_unclear = _unc_lines[:4] if _unc_lines else (
                        ["The exact repair method cannot be confirmed from the photo alone."]
                    )
                    _diy_hint = {
                        "conditional": (
                            "Once these questions are answered, this should be a manageable DIY repair."
                        ),
                        "suitable": (
                            "This is likely DIY-suitable once the details above are clarified."
                        ),
                    }.get(s1.diy_suitability, "")
                    _plural = len(s1.clarifying_questions) > 1
                    _why = (
                        f"{'These answers' if _plural else 'This answer'} will affect the repair "
                        f"method, the materials or parts to buy, and whether the job is safe to "
                        f"do yourself. {_diy_hint}"
                    ).strip()
                    clarif = ClarificationOutput(
                        what_is_known=(
                            f"{s1.problem_type}: " + "; ".join(s1.observations[:3])
                            if s1.observations else s1.problem_type
                        ),
                        what_is_unclear=_what_unclear,
                        clarifying_questions=list(s1.clarifying_questions[:3]),
                        optional_questions=list((s1.optional_questions or [])[:1]),
                        why_it_matters=_why,
                    )
                    combined = AnalysisResult(
                        output_mode="clarification",
                        stage_1=s1,
                        clarification=clarif,
                    )
                    _ms_total = int((time.monotonic() - start) * 1000)
                    analysis.structured_result  = combined.model_dump()
                    analysis.status             = "complete"
                    analysis.pipeline_stage     = "complete"
                    if case_row is not None:
                        case_row.status = "needs_info"
                    analysis.processing_time_ms = _ms_total
                    analysis.provider           = _s1_provider
                    analysis.model              = f"{_s1_label} (clarification)"
                    analysis.validation_issues  = (
                        f'{{"stage_1_ms":{_ms_s1},"total_ms":{_ms_total},'
                        f'"mode":"clarification","round":{_new_round}}}'
                    )
                    log.info(
                        "Pipeline complete (clarification), S1:%ds total:%ds questions=%d round=%d",
                        _ms_s1 // 1000, _ms_total // 1000,
                        len(s1.clarifying_questions), _new_round,
                    )
                    await db.commit()

            elif _mode == "care":
                # ── STAGE C — Care-mode output ────────────────────────────────
                # Skip Stage 2 and Stage 3. The model gives maintenance/care
                # advice for the object, not a repair execution plan.
                analysis.pipeline_stage = _SC_RUNNING
                await db.commit()

                _t_sc = time.monotonic()
                stage_c: CareOutput = await _stage_C_care(s1)
                _ms_sc = int((time.monotonic() - _t_sc) * 1000)
                log.info("Stage C (care) complete in %.1fs", _ms_sc / 1000)

                analysis.stage_outputs = {
                    **analysis.stage_outputs,
                    "stage_c": stage_c.model_dump(),
                }
                analysis.pipeline_stage = _SC_DONE
                await db.commit()

                # ── Finalise care ─────────────────────────────────────────────
                combined = AnalysisResult(
                    output_mode="care",
                    stage_1=s1,
                    care=stage_c,
                )
                _ms_total = int((time.monotonic() - start) * 1000)
                analysis.structured_result  = combined.model_dump()
                analysis.status             = "complete"
                analysis.pipeline_stage     = "complete"
                if case_row is not None:
                    case_row.status = "complete"
                    case_row.clarification_round = 0
                analysis.processing_time_ms = _ms_total
                analysis.provider           = _s1_provider
                analysis.model              = f"{_s1_label} | {STAGE_3_MODEL}(C)"
                analysis.validation_issues  = (
                    f'{{"stage_1_ms":{_ms_s1},"stage_c_ms":{_ms_sc},"total_ms":{_ms_total}}}'
                )
                log.info("Pipeline complete (care), S1:%ds SC:%ds total:%ds",
                         _ms_s1 // 1000, _ms_sc // 1000, _ms_total // 1000)
                await db.commit()

            else:
                # ── Full DIY pipeline (diy / verify_first; clarification in Step 4)
                # ── STAGE 2, Tools, Materials, Protective Gear ────────────────
                analysis.pipeline_stage = _S2_RUNNING
                await db.commit()

                _t_s2 = time.monotonic()
                s2: Stage2Output = await _stage_2_resources(s1)
                _ms_s2 = int((time.monotonic() - _t_s2) * 1000)
                log.info("Stage 2 complete in %.1fs", _ms_s2 / 1000)

                analysis.stage_outputs = {**analysis.stage_outputs, "stage_2": s2.model_dump()}
                analysis.pipeline_stage = _S2_DONE
                await db.commit()

                # ── STAGE 3, Step-by-step Repair Instructions ─────────────────
                analysis.pipeline_stage = _S3_RUNNING
                await db.commit()

                _t_s3 = time.monotonic()
                s3: Stage3Output = await _stage_3_steps(s1, s2)
                _ms_s3 = int((time.monotonic() - _t_s3) * 1000)
                log.info("Stage 3 complete in %.1fs", _ms_s3 / 1000)

                # Strip leaked tier tags from Stage 3 prose (deterministic backstop).
                _scrubbed = _scrub_stage3_prose(s3)
                if _scrubbed:
                    log.info("Stage 3 prose: scrubbed %d leaked tier tag string(s)", _scrubbed)

                # Consistency backstop: enforce Stage 1 decision on Stage 3 output.
                _decision_notes = _apply_decision_to_stage3(s3, _decision, s1)
                if _decision_notes:
                    log.info("Stage 3 decision enforcement: %s", "; ".join(_decision_notes))

                analysis.stage_outputs = {
                    **analysis.stage_outputs,
                    "stage_3": s3.model_dump(),
                    **({"stage_3_decision": _decision_notes} if _decision_notes else {}),
                }
                analysis.pipeline_stage = _S3_DONE
                await db.commit()

                # ── Finalise DIY path ──────────────────────────────────────────
                combined = AnalysisResult(
                    output_mode=_mode,
                    stage_1=s1, stage_2=s2, stage_3=s3,
                )
                _ms_total = int((time.monotonic() - start) * 1000)
                analysis.structured_result  = combined.model_dump()
                analysis.status             = "complete"
                analysis.pipeline_stage     = "complete"
                if case_row is not None:
                    case_row.status = "complete"
                    case_row.clarification_round = 0
                analysis.processing_time_ms = _ms_total
                analysis.provider           = _s1_provider
                analysis.model              = f"{_s1_label} | {STAGE_2_MODEL} | {STAGE_3_MODEL}"
                analysis.validation_issues  = (
                    f'{{"stage_1_ms":{_ms_s1},"stage_2_ms":{_ms_s2},"stage_3_ms":{_ms_s3},'
                    f'"total_ms":{_ms_total}}}'
                )
                log.info("Pipeline complete (%s), S1:%ds S2:%ds S3:%ds total:%ds",
                         _mode, _ms_s1 // 1000, _ms_s2 // 1000,
                         _ms_s3 // 1000, _ms_total // 1000)
                await db.commit()

        except Exception as exc:
            # Catch all unhandled exceptions, never let the background task
            # crash silently.  Write the error to the DB so the frontend can
            # surface it and the user can decide whether to re-trigger.
            log.error("Pipeline failed for analysis_id=%d: %s", analysis_id, exc, exc_info=True)
            try:
                analysis.status            = "failed"
                analysis.validation_issues = f"{type(exc).__name__}: {exc}"
                if case_row is not None:
                    case_row.status = "failed"
                await db.commit()
            except Exception:
                pass  # if even the error write fails, there is nothing more we can do


# ─────────────────────────────────────────────────────────────────────────────
# STAGE FUNCTIONS  (real Ollama calls, replaces Session 1 stubs)
#
# ─── DESIGN PRINCIPLE ─────────────────────────────────────────────────────────
# Each function:
#   - has an identical signature to the Session 1 stub
#   - constructs a system prompt that defines the model's role
#   - constructs a user prompt that includes context from previous stages
#   - explicitly describes the expected JSON schema in the prompt
#     (belt-and-suspenders alongside format="json")
#   - calls _call_stage() which handles retry, parsing, and validation
#   - returns a validated Pydantic object, never raw text
#
# The pipeline orchestrator (_run_pipeline) never changes regardless of whether
# the stage function calls a local model, a cloud API, or a DB lookup.
# ─────────────────────────────────────────────────────────────────────────────


async def _stage_1_analyse(
    upload_paths: list[str],
    text_context: str | None = None,
) -> Stage1Output:
    """
    Stage 1: image / problem analysis using Qwen2.5-VL (vision model).

    REAL, reads each uploaded image from disk, encodes as base64, and sends
    to Qwen2.5-VL via Ollama.  The model examines the images and classifies
    the home repair problem into a structured Stage1Output.

    Image handling:
      Images are base64-encoded and passed in the Ollama message's 'images'
      field.  Qwen2.5-VL can process multiple images in a single call, so all
      uploads for the case are sent together for a holistic diagnosis.

    trade_category constraint:
      The value must be from a fixed vocabulary so it can drive downstream
      structured queries (e.g. nearby services geo-lookup) without another
      LLM call.  The prompt enforces this, the model must pick one label
      from the allowed set, not invent its own.
    """
    images_b64 = _load_images_as_base64(upload_paths)
    if not images_b64:
        raise ValueError(
            "No readable image files found.  Cannot run Stage 1 without images."
        )

    system_prompt = (
        "You are an expert home repair diagnostician. "
        "Your job is to examine photos submitted by homeowners and identify what is damaged, "
        "broken, worn, or malfunctioning.\n\n"
        "IMPORTANT rules:\n"
        "- Describe ONLY what is LITERALLY VISIBLE. Do not invent problems or mechanisms.\n"
        "- observations and visual_evidence must be directly visible facts (e.g. 'a hole through\n"
        "  the cabinet door panel', 'water spraying from the joint'). Do NOT state that a part is\n"
        "  detached, loose, warped, missing, rotten, internally cracked, or 'not aligned' UNLESS\n"
        "  that exact condition is plainly visible. If you are INFERRING a likely cause, failure\n"
        "  mode, or hidden damage, do NOT put it in observations, put it in uncertainty_notes.\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  scope               : decide first, one of: repair, emergency, not_repair, unclear.\n"
        "                        emergency = active fire/flames/smoke, active flooding, collapse in\n"
        "                        progress, or a person in danger. not_repair = not a household repair\n"
        "                        (food, pet, person, scenery, screenshot). unclear = cannot tell.\n"
        "                        repair = an ordinary household repair. When scope is not repair, set\n"
        "                        trade_category to general/other and do not invent a repair fault.\n"
        "  problem_type        : string, name the specific damage you can actually SEE, matching\n"
        "                        your main observation (e.g. 'Hole through cabinet door panel',\n"
        "                        'Leaking pipe joint', 'Cracked wall plaster'). Do not guess a\n"
        "                        mechanism you cannot see (e.g. do not say 'Loose cabinet door' for\n"
        "                        a holed but still-mounted door).\n"
        "  trade_category      : string, EXACTLY one of these 11 values (use the exact string):\n"
        "                        plumbing, electrical, appliance, carpentry/joinery,\n"
        "                        glazing, sealing/weatherproofing, plastering/decorating,\n"
        "                        stone/masonry, structural, hvac, general/other\n"
        "\n"
        "                        Use these rules to pick the correct value:\n"
        "                        WINDOW-SPECIFIC:\n"
        "                          cracked glass, broken pane, condensation between panes (failed\n"
        "                            double-glazing unit) → glazing\n"
        "                          wooden frame rot/damage, sash alignment, window latch/handle/hinge,\n"
        "                            stiff or misaligned opening sash → carpentry/joinery\n"
        "                          draught, failed silicone bead, water ingress around the frame,\n"
        "                            gaps between frame and wall → sealing/weatherproofing\n"
        "                          crack in the wall or lintel above/beside window opening → structural\n"
        "                        OTHER:\n"
        "                          pipes, taps, drains, radiator leaks → plumbing\n"
        "                          wiring, sockets, switches, consumer units → electrical\n"
        "                          washing machine, dishwasher, boiler body, oven → appliance\n"
        "                          cracked/missing plaster, peeling paint, damp patches on walls → plastering/decorating\n"
        "                          chipped/cracked/stained natural stone (marble, granite, slate,\n"
        "                            limestone, travertine), cracked or loose ceramic/porcelain\n"
        "                            tiles, broken grout, damaged brickwork or stone fireplace,\n"
        "                            worktops/hearths/floor or wall tiles in stone or ceramic →\n"
        "                            stone/masonry  (NOT plastering, NOT general/other)\n"
        "                          load-bearing walls, foundations, lintels, masonry cracks → structural\n"
        "                          boiler controls, central heating, air conditioning, ventilation → hvac\n"
        "                          anything else → general/other\n"
        "  severity            : string, exactly one of: low, medium, high\n"
        "  observations        : array of strings, 3 to 5 specific visual observations\n"
        "                        describing exactly what you see in the image\n"
        "  confidence          : integer 0-100, your confidence that you can clearly identify\n"
        "                        a specific home repair problem from this image and any user text.\n"
        "                        STRICT RULES for confidence:\n"
        "                        90-100 = damage is crystal clear in a sharp, well-lit, close-up photo.\n"
        "                        76-89  = image is usable but not ideal (slightly blurry, far away, etc).\n"
        "                        0-75   = ANYTHING that is: mostly dark or black, mostly one solid colour,\n"
        "                                 random noise or static, a non-home-repair subject (food, person,\n"
        "                                 sky, text document, screenshot), too blurry to identify any object,\n"
        "                                 or where you are guessing rather than seeing clear damage.\n"
        "                        If you are not certain what is shown, you MUST use 0-75.\n"
        "  needs_clarification : boolean, true if confidence is 75 or below OR the image does not\n"
        "                        show a recognisable home repair problem; false otherwise.\n"
        "  clarification_request: string, if needs_clarification is true, write ONE specific sentence\n"
        "                        telling the user exactly what better photo or information would help\n"
        "                        (e.g. 'Please upload a closer, well-lit photo of the damaged pipe joint').\n"
        "                        If needs_clarification is false, use an empty string.\n"
        "  confidence_reason   : string, one to two sentences explaining exactly why you gave this\n"
        "                        confidence score. Be specific about what made the image clear or unclear.\n"
        "                        Example: 'Sharp close-up clearly shows a diagonal crack through painted\n"
        "                        plaster with visible dust below; damage extent and location are unambiguous.'\n"
        "  visual_evidence     : array of strings, list every distinct object or feature you can clearly\n"
        "                        see in the image, not just the damaged part. Be specific and factual.\n"
        "                        Example: ['White painted wall', 'Diagonal hairline crack approx 20cm',\n"
        "                        'Plaster dust on skirting board below crack', 'Painted skirting board']\n"
        "                        Do NOT include items you cannot actually see.\n"
        "  uncertainty_notes   : string, what you CANNOT confirm from the photo alone. For ANY\n"
        "                        damage you MUST fill this in: typically the material type, whether\n"
        "                        there is hidden/internal damage, the depth/extent, and whether a\n"
        "                        repair or a full replacement is the right fix. Use an empty string\n"
        "                        \"\" ONLY if the problem and its fix are completely unambiguous.\n"
        "                        Example: 'Cannot confirm if the door panel is solid timber or\n"
        "                        hollow/MDF, how deep the damage goes, or whether it can be patched\n"
        "                        or needs replacing.'\n\n"
        "RISK ASSESSMENT, reason about these for a careful NON-EXPERT DIY homeowner (basic hand\n"
        "tools, no trade qualifications). IMPORTANT MINDSET: most ordinary household repairs ARE\n"
        "suitable for a careful DIY user, filling a hole, replacing a panel or door, tightening or\n"
        "swapping a fitting, sealing, sanding, painting. Do NOT treat a repair as risky just because\n"
        "the visible DAMAGE looks bad. Severity (how bad the damage is) is SEPARATE from risk (how\n"
        "dangerous the REPAIR is for a DIYer). A big hole in a cabinet door is high severity but LOW\n"
        "risk. Judge each field from THIS scene, not from the trade name:\n"
        "  risk_level          : one of low, medium, high, critical, how dangerous it is for a DIY\n"
        "                        user to DO this repair. Drive it from real stored/again energy or\n"
        "                        consequence: mains electricity, gas, water under pressure, structural\n"
        "                        loads, working at height, vehicle/airbag/fuel systems, heavy or\n"
        "                        razor-sharp parts. low = ordinary hand-tool work (wood, filler,\n"
        "                        sealant, a panel). high = a real chance of serious injury. critical =\n"
        "                        a mistake could kill or disable a safety system (live mains, gas,\n"
        "                        load-bearing failure). If NONE of those energy/consequence factors is\n"
        "                        present, risk_level is low even if the damage looks severe.\n"
        "  hidden_damage_risk  : one of low, medium, high, how likely important damage is NOT visible\n"
        "                        (internal/structural/behind surfaces). A fully-visible surface defect\n"
        "                        is low; suspected concealed/internal damage is high.\n"
        "  diy_suitability     : one of suitable, conditional, not_recommended.\n"
        "                        'conditional' = the repair is DIY-ok BUT the user must resolve a\n"
        "                        specific unknown before buying parts or starting work. You MUST use\n"
        "                        'conditional' (NOT 'suitable') when uncertainty_notes mentions any\n"
        "                        of these unknowns:\n"
        "                          - material type (copper vs plastic, plaster vs drywall vs concrete,\n"
        "                            solid timber vs hollow MDF, stone type, paint type)\n"
        "                          - fitting or connector type (compression vs push-fit vs threaded)\n"
        "                          - pipe or cable size / diameter\n"
        "                          - substrate or base material\n"
        "                        If uncertainty_notes mentions any of the above, diy_suitability MUST\n"
        "                        be 'conditional' (not 'suitable') and clarifying_questions MUST\n"
        "                        contain at least one specific question to resolve it.\n"
        "                        'suitable' = everything needed is clear from the photo.\n"
        "                        'not_recommended' = real hazard or professional expertise needed.\n"
        "  professional_assessment_recommended : boolean, true ONLY when an in-person professional\n"
        "                        should inspect BEFORE any repair, because the photo cannot establish\n"
        "                        safety or the correct fix AND a real hazard is present. False for\n"
        "                        ordinary visible DIY repairs.\n"
        "  safety_concerns     : array naming hazards that are VISIBLE in the image (a clearly\n"
        "                        visible danger such as fire or standing water counts). Do NOT claim\n"
        "                        exposed mains, exposed conductors, or an electrical fault unless\n"
        "                        bare/charred wiring is actually visible; a hazard you only infer goes\n"
        "                        in uncertainty_notes, not here. CONSISTENCY RULE: if risk_level is\n"
        "                        high/critical, OR diy_suitability is not_recommended, OR\n"
        "                        professional_assessment_recommended is true, name the visible hazard\n"
        "                        plainly rather than inventing a mechanism. If you cannot name any\n"
        "                        real visible hazard, the risk is LOW and the job is suitable, [].\n"
        "  clarifying_questions: Work through these three checks before deciding what to ask:\n"
        "      CHECK 1: What am I NOT sure about that would change my DIAGNOSIS of what is broken?\n"
        "      CHECK 2: What would change whether this is safe for DIY or needs a professional?\n"
        "      CHECK 3: What would change WHICH method, tool, or material the user should buy?\n"
        "      COMPREHENSIVENESS RULE: Ask ALL questions you need at once. If you need to know\n"
        "      cause AND size AND surface condition, put all three here. Do not hold back to\n"
        "      keep the list short — a user looping through multiple re-analyses because you\n"
        "      asked one question at a time has a far worse experience than being asked three\n"
        "      questions upfront. Think ahead: what is the complete set of unknowns that would\n"
        "      change your routing decision?\n"
        "      CONTEXT RULE: If the user message contains previous answers (look for lines like\n"
        "      'Q1: ... A1: ...'), read them carefully. Only ask about uncertainties that those\n"
        "      answers did NOT resolve. Do not repeat or rephrase already-answered questions.\n"
        "      For each unresolved check that produces a 'yes', write one specific, answerable\n"
        "      question. Maximum 3 questions, ordered by importance. Plain English.\n"
        "      GOOD questions (change the recommendation):\n"
        "        'Is the pipe copper, plastic, or steel?' - changes the fitting and method.\n"
        "        'Is the wall plaster, drywall, or concrete?' - changes the filler and primer.\n"
        "        'Is the area above the ceiling still wet?' - changes safety and method.\n"
        "        'What caused the burn mark — heat, chemical, or friction?' - changes method.\n"
        "        'Approximately how wide is the damaged area?' - changes repair vs replace.\n"
        "      BAD questions (never ask — they do not change the recommendation):\n"
        "        'Can you describe the damage more?', 'How old is the fitting?',\n"
        "        'What brand is it?', 'Can you provide more context?'\n"
        "      EMPTY ONLY when all three checks produce 'nothing important is unclear'.\n"
        "      For most real damage photos, at least one check will produce a useful question.\n"
        "  optional_questions: At most 1. A photo in better lighting, a wider shot, or one minor\n"
        "      detail that would improve advice quality but is NOT needed to make a routing\n"
        "      decision. Only include something here if it is genuinely optional — i.e., you\n"
        "      could give safe, specific advice without it. Leave [] if nothing is optional.\n"
        "  EXAMPLES (calibration):\n"
        "  Hole in a cabinet door -> risk_level low, diy_suitability suitable,\n"
        "  clarifying_questions [] (nothing unclear), optional_questions [].\n"
        "  Leaking pipe joint, fitting type unknown -> diy_suitability conditional,\n"
        "  clarifying_questions ['Is the pipe copper, plastic, or steel?',\n"
        "  'Is the fitting a compression, push-fit, or threaded type?'],\n"
        "  optional_questions [].\n"
        "  Burn mark on floor, cause/size/depth all unknown -> diy_suitability conditional,\n"
        "  clarifying_questions ['What caused the burn mark (e.g. heat, cigarette, chemical)?',\n"
        "  'Approximately how wide is the mark?',\n"
        "  'Does the surface feel soft, cracked, raised, or deeply charred?'],\n"
        "  optional_questions ['Can you upload a closer photo in natural light?'].\n"
        "  Visibly exposed/charred wiring -> scope repair, risk_level critical, not_recommended,\n"
        "  safety_concerns about exposed conductors (visible), professional_assessment_recommended true.\n"
        "  Active fire or flames (e.g. a pan on fire) -> scope emergency, trade_category general/other,\n"
        "  safety_concerns name the visible fire; do not infer exposed mains unless wiring is visible.\n"
        "  Not a repair (food, pet, person, scenery, screenshot) -> scope not_repair, general/other.\n\n"
        "Return ONLY the JSON object.  No explanation, no markdown, no extra text."
    )

    user_prompt = "Examine the attached image(s) carefully. What physical object or fixture is shown, and what is damaged or needs repair?"
    if text_context:
        # Detect a structured Q/A block produced by _check_auto_reanalyse
        # (format: "User answers to clarification questions:\nQ1: ...\nA1: ...").
        # Use a stronger "do not re-ask" framing for structured blocks so Stage 1
        # treats these as resolved topics rather than just context.
        # Plain user descriptions keep the original "described the problem as" framing.
        _has_qa_structure = "Q1:" in text_context or text_context.strip().startswith("User answers")
        if _has_qa_structure:
            user_prompt = (
                "PREVIOUSLY RESOLVED — DO NOT RE-ASK:\n"
                "The following questions were asked in a previous analysis round and "
                "have been answered by the user. These topics are now known. "
                "Do not include any of these in clarifying_questions — not even rephrased.\n\n"
                f"{text_context}\n\n"
                "Only raise a new clarifying_question if it covers a DIFFERENT unknown "
                "that would materially change the repair method, safety assessment, or "
                "tool/material choice, and that is NOT already answered above.\n\n"
            ) + user_prompt
        else:
            user_prompt = (
                f"The user has described the problem as: {text_context}\n\n"
            ) + user_prompt
    user_prompt += " Return your diagnosis as a JSON object matching the schema described."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
            "images": images_b64,       # base64-encoded image bytes, Ollama vision API
        },
    ]

    # CRITICAL: num_ctx must cover system prompt (~350 tok now, with new fields)
    # + image tokens (~500-700 per image for Qwen2.5-VL) + JSON response (~600 tok).
    # num_predict is set to STAGE_1_NUM_PREDICT (default 800) to accommodate
    # confidence/clarification fields + visual_evidence list + confidence_reason + uncertainty_notes.
    return await _call_stage(
        model=STAGE_1_MODEL,
        messages=messages,
        schema_class=Stage1Output,
        stage_name="stage_1 (Qwen2.5-VL)",
        extra_options={
            "num_ctx":     STAGE_1_NUM_CTX,     # 4096
            "num_predict": STAGE_1_NUM_PREDICT,  # bumped to 600 in .env
            "temperature": 0.0,                 # greedy → reduce run-to-run variance
        },
    )


# ── Stage 2 per-trade profiles ──────────────────────────────────────────────
# Injected into the Stage 2 prompt based on Stage 1's trade_category so the model
# reasons within ONE trade's vocabulary instead of a generic cross-trade brand
# menu. Each block states where tools/materials legitimately live, the
# repair-vs-replace cue, and a hard "never recommend" list that stops cross-trade
# leakage (e.g. plaster bonding agent showing up on a cabinet-door job).
_STAGE_2_TRADE_PROFILES = {
    "carpentry/joinery": (
        "TRADE FOCUS, carpentry/joinery:\n"
        "IF the problem involves a HINGE (cabinet door hinge, door hinge, concealed/European\n"
        "hinge, butt hinge):\n"
        "- Most hinge problems are ADJUSTMENT, not replacement. Concealed (European-style)\n"
        "  hinges have three screws: depth (in/out), side (left/right), and height (up/down).\n"
        "  A misaligned door almost always needs screw adjustment before anything else.\n"
        "- Likely essential items: cross-head screwdriver (Pozi or Phillips, size 2 — NOT an\n"
        "  Allen key; concealed hinges use screwdrivers exclusively). Small flat-head screwdriver\n"
        "  only if side-adjust screws have a flat slot. Hinge replacement only if the hinge body\n"
        "  is cracked, bent, or the clip mechanism is broken.\n"
        "- [Verify First] items: replacement hinge (only if hinge is physically damaged or\n"
        "  screws are stripped past adjustment). Wood-fill product only if screw holes are\n"
        "  stripped (toothpick-and-PVA or proper wood filler plug).\n"
        "- NEVER list for hinge work: Allen keys, lubricant of any kind (WD-40, oil, grease —\n"
        "  not as a fix, not as maintenance, not as prevention), drills unless the hinge plate\n"
        "  itself must be moved to a new position, plaster fillers, wall paint.\n"
        "OTHERWISE (cabinet door damage, timber repair, etc.):\n"
        "- Decide repair vs replace FIRST. A large hole through THIN, HOLLOW or MDF/\n"
        "  particleboard is REPLACE (matching door/panel). Only SOLID timber with a SMALL\n"
        "  hole is a sensible patch (two-part wood filler + matching finish).\n"
        "- NEVER recommend: plaster fillers, wall emulsion, plumbing or electrical parts."
    ),
    "plumbing": (
        "TRADE FOCUS, plumbing:\n"
        "- First action is almost always: isolate the water supply (stopcock/isolation valve)\n"
        "  and release the pressure before anything else.\n"
        "- Commit to ONE joint/fitting type from the evidence: compression, push-fit, solvent-\n"
        "  weld, threaded (thread-seal tape/compound), or soldered (often a pro job). Listing\n"
        "  more than one sealing/jointing method as required is contradictory, do NOT do it.\n"
        "- Legitimate items: the one chosen fitting/coupler/seal, towels/bucket, pipe cleaner\n"
        "  or deburr tool. NEVER recommend wall fillers, paint, or wood products.\n"
        "- If the fitting type is not clear from the photo, mark the specific parts\n"
        "  [Verify First] and tell the user to identify the fitting and pipe size before buying.\n"
        "- For an outdoor/buried supply pipe, note it may be the homeowner's responsibility past\n"
        "  the boundary stopcock and that buried services/pressure raise the risk."
    ),
    "electrical": (
        "TRADE FOCUS, electrical:\n"
        "- Default to STOP and ESCALATE. For most homeowners the only safe [Confirmed] action\n"
        "  is: isolate the circuit at the consumer unit and call a registered electrician.\n"
        "- Keep items minimal (isolation; a socket/voltage tester at most, [Verify First]).\n"
        "- NEVER recommend buying parts for DIY mains (230V) install or imply DIY rewiring."
    ),
    "plastering/decorating": (
        "TRADE FOCUS, plastering/decorating:\n"
        "- Match filler grade to the damage: fine filler for hairline cracks, deeper/flexible\n"
        "  filler for wider or movement cracks; bonding agent only on smooth/previously-sealed\n"
        "  surfaces. Generic type first, brand only as an example.\n"
        "- Legitimate items: filler, abrasive, primer/mist coat, matching paint.\n"
        "- NEVER recommend wood, plumbing, or electrical parts."
    ),
    "glazing": (
        "TRADE FOCUS, glazing:\n"
        "- Cracked/broken glass and failed sealed (misted) units are normally REPLACE, usually\n"
        "  measured and fitted by a glazier. Lead with measured replacement glass/unit as\n"
        "  [Likely] plus a [Verify First] step to measure the pane and confirm single vs double\n"
        "  glazing. NEVER recommend fillers or paint."
    ),
    "sealing/weatherproofing": (
        "TRADE FOCUS, sealing/weatherproofing:\n"
        "- Pick ONE correct sealant for the location (frame/exterior sealant, or sanitary\n"
        "  silicone for wet areas) plus removal of the old bead. One sealant type only.\n"
        "- NEVER recommend structural, wood, or plumbing parts unless the evidence shows them."
    ),
    "stone/masonry": (
        "TRADE FOCUS, stone/masonry (natural stone like marble/granite/slate/limestone, and\n"
        "ceramic/porcelain tiling):\n"
        "- A crack/chip in stone is STABILISE-AND-FILL, NEVER chisel-and-demolish. Do NOT list\n"
        "  hammers, chisels or breakers for crack repair, that is the wrong method. Only consider\n"
        "  a chisel if the stone is genuinely broken into separate pieces that need to be lifted\n"
        "  for a full replacement (a different job).\n"
        "- The PRIMARY repair material is a knife-grade two-part stone/marble epoxy (or\n"
        "  clear-setting cyanoacrylate for very fine hairlines), [Confirmed] or [Likely], NEVER\n"
        "  [Do Not Buy Yet]. For colour-matching, knife-grade epoxy can be tinted with stone-dust\n"
        "  or pigment of the same stone type, [Likely].\n"
        "- Other essentials: acetone or IPA for degreasing the crack (acidic cleaners ETCH stone,\n"
        "  do NOT recommend vinegar/lemon/general bathroom cleaner). Low-tack painter's tape to\n"
        "  mask the crack edges. Clean lint-free cloth. Razor blade or stone scraper for trimming\n"
        "  proud epoxy once cured. Wet/dry sandpaper progressing 400 -> 800 -> 1500 -> 3000 for\n"
        "  polishing back to gloss on polished marble/granite (omit for honed/matte finishes).\n"
        "  Tin-oxide marble polish or a diamond polishing pad to restore the original sheen.\n"
        "- NEVER recommend wood filler, plaster filler, builder's caulk, decorator's caulk, or\n"
        "  general-purpose silicone for a stone crack, they discolour, shrink and cannot be\n"
        "  polished. NEVER recommend grout for cracked stone (grout is for joints between tiles).\n"
        "- For cracked grout between TILES (not the tile itself): grout removal tool/saw plus\n"
        "  matching grout, NOT epoxy."
    ),
    "structural": (
        "TRADE FOCUS, structural:\n"
        "- Default to ESCALATE: recommend assessment by a structural engineer or builder before\n"
        "  any work or purchase. Keep any DIY materials minimal and [Verify First]."
    ),
    "general/other": (
        "TRADE FOCUS, general household repair:\n"
        "- Read the problem_type carefully and tailor resources to THAT SPECIFIC REPAIR.\n"
        "  Do NOT fall back to generic lists. If it involves hinges: see the carpentry/joinery\n"
        "  hinge guidance above. If it involves flat-pack furniture: Allen key and replacement\n"
        "  cam/bolt/dowel. If it involves a sticky drawer: runner lubricant or runner replacement.\n"
        "- List only what the specific observed problem requires. No items 'just in case'.\n"
        "- NEVER mix parts from different repair categories (e.g. no plumbing parts for a\n"
        "  furniture repair, no electrical parts for a cabinet job)."
    ),
}

_STAGE_2_TRADE_GENERIC = (
    "TRADE FOCUS, {tc}:\n"
    "- Recommend only items the observations and visual evidence directly support, described\n"
    "  generically. If the correct approach cannot be determined from the photo, mark the key\n"
    "  items [Verify First] and state the check the user must do. Escalate to a qualified\n"
    "  professional where safety or compliance is involved."
)


def _stage_2_trade_profile(trade_category: str, problem_type: str = "") -> str:
    """Return the per-trade guidance block for the Stage 2 prompt (gated by trade)."""
    tc = (trade_category or "general/other").strip().lower()
    pt = (problem_type or "").lower()
    if tc == "general/other" and any(w in pt for w in ("hinge", "cabinet door", "cupboard door", "wardrobe door")):
        tc = "carpentry/joinery"
    return _STAGE_2_TRADE_PROFILES.get(tc, _STAGE_2_TRADE_GENERIC.format(tc=tc))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE A — Assessment-first output (replaces Stage 2 + 3 when the model has
# determined that professional in-person assessment is needed before any repair)
#
# Design principle: the model's role is now stabilisation, safety, and
# escalation — not repair execution. The prompt teaches it to reason about:
#   (1) what the user must do RIGHT NOW to make things safe
#   (2) what they must NOT do (worsen the danger or the damage)
#   (3) why a professional is needed (specific to THIS scene)
#   (4) what information to gather before the professional arrives
#   (5) who to call and how urgently
#
# Nothing in this output describes how to perform the repair.
# ─────────────────────────────────────────────────────────────────────────────

async def _stage_A_assessment(stage_1: Stage1Output) -> AssessmentOutput:
    """
    Stage A: assessment-first output using Llama 3.1:8b (text model).

    Called INSTEAD of Stage 2 and Stage 3 when output_mode == 'assessment_first'.
    Receives the Stage 1 diagnosis as context and produces AssessmentOutput:
    stabilisation actions, what NOT to do, risk explanation, information to
    gather, and professional guidance.

    Never produces repair steps, tool lists, or material lists.
    """
    _vis     = "; ".join(stage_1.visual_evidence[:6]) if stage_1.visual_evidence else "none recorded"
    _unc     = stage_1.uncertainty_notes.strip() if stage_1.uncertainty_notes else "none"
    _hazards = "; ".join(stage_1.safety_concerns) if stage_1.safety_concerns else "not specified"
    context  = (
        f"Problem                : {stage_1.problem_type}\n"
        f"Trade                  : {stage_1.trade_category}\n"
        f"Severity               : {stage_1.severity}\n"
        f"Diagnosis confidence   : {stage_1.confidence}%\n"
        f"Risk level             : {stage_1.risk_level}\n"
        f"Hidden damage risk     : {stage_1.hidden_damage_risk}\n"
        f"Named hazards          : {_hazards}\n"
        f"Visually confirmed     : {_vis}\n"
        f"Cannot confirm         : {_unc}\n"
        f"Observations           :\n"
        + "\n".join(f"  - {obs}" for obs in stage_1.observations)
    )

    system_prompt = (
        "You are a safety-first remote assessor advising a homeowner who has sent you a photo.\n"
        "The diagnosis above indicates this situation requires in-person professional assessment\n"
        "before any repair work is attempted.\n\n"
        "YOUR ROLE HAS CHANGED. You are NOT providing repair guidance. You are providing:\n"
        "  1. Stabilisation: what the user must do right now to make things safe\n"
        "  2. Caution: what they must NOT do under any circumstances\n"
        "  3. Explanation: why a professional must assess it (specific to this scene)\n"
        "  4. Preparation: what information/evidence to gather for the professional\n"
        "  5. Escalation: who to call and how urgently\n\n"
        "RULES:\n"
        "- Do NOT describe how to perform the repair or any part of it.\n"
        "- Do NOT suggest DIY tools, materials, or methods.\n"
        "- Do NOT give speculative repair estimates.\n"
        "- Be specific to THIS scene. Use the diagnosis context to name the exact hazards.\n"
        "- Immediate actions must be doable right now with no special tools.\n"
        "  Examples: 'Turn off the water at the stopcock', 'Switch off the circuit at\n"
        "  the consumer unit and leave it off', 'Do not use the room until assessed',\n"
        "  'Place a bucket under the leak'.\n"
        "- do_not_do must name specific actions that could worsen the danger for THIS case.\n"
        "  Examples: 'Do not attempt to open the consumer unit', 'Do not remove ceiling\n"
        "  panels', 'Do not apply weight to the affected area'.\n"
        "- why_assessment_needed must explain the specific risk in plain English.\n"
        "  Do NOT use generic phrases like 'this is dangerous'. Say WHY it is dangerous\n"
        "  for this specific scene (e.g. 'water near electrical wiring above the ceiling\n"
        "  creates a risk of electrocution even if the immediate area looks dry').\n"
        "- information_needed is what a professional would want to know or see when they\n"
        "  arrive. Be practical: 'photograph the source of the leak', 'note when it\n"
        "  started', 'check if the circuit breaker has tripped'.\n"
        "- professional_urgency must be exactly one of: 'immediately', 'within 24 hours',\n"
        "  'within the week', 'when convenient'. Use 'immediately' only for active\n"
        "  danger (live exposed wiring, active structural failure, gas smell).\n\n"
        "Return a single JSON object with exactly these fields:\n"
        "  immediate_actions    : array of 2 to 5 plain strings, each one specific action\n"
        "  do_not_do            : array of 2 to 4 plain strings, each one specific prohibition\n"
        "  why_assessment_needed: string, 2 to 3 sentences, specific to this scene\n"
        "  information_needed   : array of 2 to 4 plain strings, evidence to gather\n"
        "  professional_type    : string, exact trade to call (e.g. 'CIPHE-registered plumber',\n"
        "                         'NICEIC-registered electrician', 'structural engineer')\n"
        "  professional_urgency : string, exactly one of: immediately | within 24 hours |\n"
        "                         within the week | when convenient\n\n"
        "Return ONLY the JSON object. No explanation, no markdown, no extra text."
    )

    user_prompt = (
        f"Based on this assessment:\n\n{context}\n\n"
        "Provide stabilisation guidance and professional escalation. "
        "Do not provide repair instructions."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    return await _call_stage(
        model=STAGE_3_MODEL,   # reuse Llama 3.1:8b — same quality, avoids extra model load
        messages=messages,
        schema_class=AssessmentOutput,
        stage_name="stage_A (assessment)",
        extra_options={
            "num_ctx":     STAGE_3_NUM_CTX,
            "num_predict": 600,       # assessment output is shorter than a full repair plan
            "temperature": 0.0,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE C — Care-mode output (replaces Stage 2 + 3 when the model determines
# this is a low-risk maintenance or care task, not a repair execution task)
#
# Design principle: the model advises on care and maintenance of the specific
# object and material. No safety notes, no protective gear, no professional
# guidance, no repair execution steps. The tone is practical and friendly,
# like advice from a product-care specialist, not a tradesperson.
# ─────────────────────────────────────────────────────────────────────────────

async def _stage_C_care(stage_1: Stage1Output) -> CareOutput:
    """
    Stage C: care-mode output using Llama 3.1:8b (text model).

    Called INSTEAD of Stage 2 and Stage 3 when output_mode == 'care'.
    Receives the Stage 1 diagnosis as context and produces CareOutput:
    a plain-English condition summary, ordered care recommendations,
    things to avoid, and what the result should look like.

    Never produces repair steps, tool lists, safety warnings, or professional
    guidance — this is care and maintenance advice only.
    """
    _vis = "; ".join(stage_1.visual_evidence[:5]) if stage_1.visual_evidence else "none recorded"
    _obs = "\n".join(f"  - {o}" for o in stage_1.observations)
    context = (
        f"Object / problem type  : {stage_1.problem_type}\n"
        f"Visually confirmed     : {_vis}\n"
        f"Observations           :\n{_obs}"
    )

    system_prompt = (
        "You are a practical care and maintenance advisor. The user has sent you a photo of "
        "an object that needs care or maintenance, not a structural repair.\n\n"
        "YOUR ROLE: give specific, friendly, practical advice on how to care for and restore "
        "this object based on what was observed. Think like a product-care specialist who "
        "knows the material, not a builder or tradesperson.\n\n"
        "RULES:\n"
        "- Be specific to the ACTUAL object and material shown. Name the material if visible.\n"
        "- Recommendations must be ordered: preparation first, then cleaning, then treatment,\n"
        "  then protection/finishing. Skip steps that do not apply.\n"
        "- Do NOT include safety warnings, protective gear, professional guidance, or repair\n"
        "  steps. This is maintenance and care advice only.\n"
        "- Do NOT include dust sheets, work gloves, or other trade-level precautions.\n"
        "- Use plain, friendly language. 'Wipe gently with a damp cloth' not 'Apply a\n"
        "  cleaning agent to the substrate using circular motions'.\n"
        "- what_to_avoid must be specific to the material (e.g. 'Do not use bleach on\n"
        "  leather, it dries and cracks the surface'). Generic warnings are not useful.\n"
        "- expected_outcome is one sentence describing what the item looks/feels like after\n"
        "  following the advice. Be realistic and specific.\n\n"
        "Return a single JSON object with exactly these fields:\n"
        "  condition_summary  : string, 1-2 sentences describing what the photo shows in\n"
        "                       plain English (e.g. 'The leather upper shows surface\n"
        "                       discolouration and light scuff marks; the sole shows wear\n"
        "                       but the structure looks intact.')\n"
        "  recommendations    : array of 2 to 6 strings, specific ordered care actions.\n"
        "                       Each string is one action: what to use, how to apply it,\n"
        "                       what done-right looks like. Keep each under 30 words.\n"
        "  what_to_avoid      : array of 0 to 3 strings, material-specific pitfalls. Empty\n"
        "                       array if nothing specific applies. Do NOT list generic\n"
        "                       cautions like 'be careful' or 'test a small area first' -\n"
        "                       only include if there is a real material-specific risk.\n"
        "  expected_outcome   : string, one sentence on what the result should look like.\n\n"
        "Return ONLY the JSON object. No explanation, no markdown, no extra text."
    )

    user_prompt = (
        f"Based on this observation:\n\n{context}\n\n"
        "Provide specific care and maintenance advice for this object."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    return await _call_stage(
        model=STAGE_3_MODEL,   # Llama 3.1:8b — good at practical advice
        messages=messages,
        schema_class=CareOutput,
        stage_name="stage_C (care)",
        extra_options={
            "num_ctx":     STAGE_3_NUM_CTX,
            "num_predict": 500,   # care output is compact
            "temperature": 0.1,   # slight warmth for friendly tone, still deterministic
        },
    )


async def _stage_2_resources(stage_1: Stage1Output) -> Stage2Output:
    """
    Stage 2: tools / materials / protective gear using Mistral (text model).

    REAL, receives Stage 1's structured output as context and asks Mistral
    to list what the user needs before starting the repair.

    Context passed forward:
      The stage_1 object is serialised to a readable text block so the model
      understands the full diagnosis, not just the problem type, but also
      the severity and what was visually observed.

    Hybrid-design note:
      For well-known (trade_category, problem_type) combinations, tools and
      materials are largely deterministic.  A future optimisation is to serve
      Stage 2 from a structured lookup table (DB query or rules engine) rather
      than spending LLM tokens on a call that always gives the same answer
      for "plumbing / leaking joint".

      To make that swap: replace the body of this function with a DB query.
      The function signature, return type, and the pipeline orchestrator are
      completely unchanged.  This is the hybrid architecture design point.
    """
    # Serialise Stage 1 output into a readable context block for the prompt.
    # Using field-by-field text rather than raw JSON makes it easier for the
    # model to understand what was diagnosed.
    # Confidence / uncertainty context from Stage 1.  Stage 2 uses this to decide
    # how firmly to recommend specific parts/sizes versus flag them for verification:
    # a low confidence or a populated "uncertain" list means the exact fitting, size,
    # or material type is an inference from the photo, not a confirmed fact.
    _vis = "; ".join(stage_1.visual_evidence[:8]) if stage_1.visual_evidence else "none recorded"
    _unc = stage_1.uncertainty_notes.strip() if stage_1.uncertainty_notes else "none recorded"
    _reason = stage_1.confidence_reason.strip() if stage_1.confidence_reason else "not given"
    # Video-path extra fields (safe on image path — VideoStage1Output is a superset)
    _key_moments = getattr(stage_1, "key_moments", []) or []
    _can_confirm = getattr(stage_1, "what_can_be_confirmed", []) or []
    _cannot_confirm = getattr(stage_1, "what_cannot_be_confirmed", []) or []
    _key_moments_block = (
        "\nKey moments (video)    :\n" + "\n".join(f"  - {m}" for m in _key_moments)
        if _key_moments else ""
    )
    _confirm_block = (
        "\nCan confirm            : " + "; ".join(_can_confirm)
        if _can_confirm else ""
    )
    _cannot_block = (
        "\nCannot confirm         : " + "; ".join(_cannot_confirm)
        if _cannot_confirm else ""
    )
    context = (
        f"Problem type           : {stage_1.problem_type}\n"
        f"Trade                  : {stage_1.trade_category}\n"
        f"Severity               : {stage_1.severity}\n"
        f"Diagnosis confidence   : {stage_1.confidence}% ({_reason})\n"
        f"Visually confirmed     : {_vis}\n"
        f"NOT confirmed/uncertain: {_unc}\n"
        + _confirm_block
        + _cannot_block
        + f"\nObservations  :\n"
        + "\n".join(f"  - {obs}" for obs in stage_1.observations)
        + _key_moments_block
    )

    _trade_block = _stage_2_trade_profile(stage_1.trade_category, stage_1.problem_type or "")
    system_prompt = (
        "You are an experienced UK tradesperson preparing a homeowner for a repair.\n"
        "You are working from a vision-model diagnosis. You CANNOT see the image or video yourself,\n"
        "so treat the diagnosis as evidence. Sound like a careful, experienced tradesperson — not\n"
        "a generic safety manual.\n\n"
        "ROOT CAUSE INFERENCE — do this FIRST before listing resources:\n"
        "Read 'Problem type', 'Observations', 'Visually confirmed', and 'Key moments' (if given).\n"
        "Do NOT simply repeat what is visible. Instead, reason about WHAT IS CAUSING the symptoms.\n"
        "Ask: given everything observed, what is the single most likely underlying fault?\n"
        "  Wrong: 'The hinge appears to be malfunctioning.'\n"
        "  Right: 'The door is misaligned, most likely because the hinge adjustment screws have\n"
        "          loosened over time — the video shows the user turning adjustment screws, which\n"
        "          is consistent with hinge-plate drift rather than structural damage.'\n"
        "List the top 3 probable causes ranked by confidence percentage in 'likely_causes'.\n"
        "Write 'repair_confidence_reasoning' as ONE paragraph: cite specific observations or\n"
        "key moments that led you to the primary cause. Mention what would change your diagnosis.\n\n"
        "TIER EVERY TOOL AND MATERIAL. Begin each string with exactly one tag:\n"
        "  [Confirmed]      the observations / visual evidence clearly show this is needed.\n"
        "  [Likely]         the standard fix for THIS exact damage in THIS trade. The PRIMARY\n"
        "                   repair material (the filler, the replacement part, the sealant that\n"
        "                   actually does the fix) belongs here at minimum, never lower.\n"
        "  [Verify First]   depends on a check the user must do first (size, thread/fitting type,\n"
        "                   panel material, screw head) before buying.\n"
        "  [Do Not Buy Yet] only needed in a contingency IF a specific hypothesis turns out true.\n"
        "                   This is a CONTINGENCY tier, NEVER use it for the primary repair item.\n"
        "Pick the tier from the context's 'Diagnosis confidence', 'Visually confirmed' and 'NOT\n"
        "confirmed/uncertain'. Lower confidence means fewer [Confirmed] items and more [Verify\n"
        "First], it never demotes the primary fix to [Do Not Buy Yet].\n"
        "CRITICAL: the single most important material or item to FIX the problem MUST be tagged\n"
        "[Confirmed] or [Likely]. If you find yourself tagging the main repair item as [Verify\n"
        "First] or [Do Not Buy Yet], your tiers are inverted — fix them before returning.\n\n"
        "EVIDENCE RULE: every tool or material must follow from a specific observation or\n"
        "visual-evidence entry. If you cannot point to what makes it necessary, drop it.\n"
        "Do NOT add items just because they are common for the trade.\n\n"
        "ONE METHOD ONLY: choose the single most likely repair method. NEVER list mutually\n"
        "exclusive methods together. Put any alternative as a SINGLE [Verify First] line.\n\n"
        "REPAIR vs REPLACE: decide before listing materials. State the deciding factor.\n\n"
        "BRANDS: describe items generically first. One brand in parentheses as example only.\n\n"
        f"{_trade_block}\n\n"
        "Format each tool/material entry as: '[Tier] generic item (spec if evidence supports it),\n"
        "short purpose'. Separate ESSENTIAL from OPTIONAL.\n\n"
        "Return a single JSON object with exactly these fields:\n"
        "  likely_causes          : array of 3 strings, probable root causes ranked by confidence,\n"
        "                           e.g. 'Hinge adjustment screws loose over time (80%)'\n"
        "  repair_confidence_reasoning : string, one paragraph citing specific evidence for the\n"
        "                           primary cause and what would change the diagnosis\n"
        "  tools                  : array of strings, ESSENTIAL tools, each tier-tagged\n"
        "  optional_tools         : array of strings, OPTIONAL tools, each tier-tagged\n"
        "  materials              : array of strings, ESSENTIAL materials, each tier-tagged\n"
        "  optional_materials     : array of strings, OPTIONAL materials, each tier-tagged\n"
        "  protective_gear        : array of strings, safety gear with one-line reason ONLY when\n"
        "                           a genuine hazard exists; leave empty if none apply\n"
        "  estimated_time_minutes : integer, realistic total including any drying/curing\n\n"
        "Return ONLY the JSON object. No explanation, no markdown, no extra text."
    )

    user_prompt = (
        f"Based on this repair diagnosis:\n\n{context}\n\n"
        "List the tools, materials, protective gear, and estimated time needed."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    return await _call_stage(
        model=STAGE_2_MODEL,
        messages=messages,
        schema_class=Stage2Output,
        stage_name="stage_2 (Mistral)",
        extra_options={
            "num_ctx":     STAGE_2_NUM_CTX,
            "num_predict": STAGE_2_NUM_PREDICT,
            "temperature": 0.0,   # deterministic recommendations
        },
    )


# ── Stage 3 per-trade profiles ──────────────────────────────────────────────
# Injected into the Stage 3 prompt based on trade_category so the repair steps
# and hazards are category-specific. This replaces the old hardcoded plaster-wall
# guidance (mist coat / feathering / "pink to white") that used to leak into every
# job. Each block gives the correct step shape for the trade plus its REAL hazards
# so the risk reasoning is grounded, not generic.
_STAGE_3_TRADE_PROFILES = {
    "carpentry/joinery": (
        "TRADE FOCUS, carpentry/joinery:\n"
        "IF the problem involves a HINGE (cabinet door hinge, concealed/European hinge, butt hinge):\n"
        "- The standard concealed (European-style) hinge has THREE independent adjustments:\n"
        "  1. DEPTH screw (also called 'in/out'): moves the door towards or away from the frame.\n"
        "     Turn clockwise to pull the door in; anti-clockwise to push it out.\n"
        "  2. SIDE screw (also called 'lateral' or 'left/right'): slides the door left or right.\n"
        "     This is usually a sliding cam or a side screw on the mounting plate.\n"
        "  3. HEIGHT screw: raises or lowers the door. Usually on the mounting plate.\n"
        "- Step sequence for a misaligned concealed hinge:\n"
        "  a. Open the door fully and locate the hinge arms (usually 2 or 3 hinges per door).\n"
        "  b. Identify which type of misalignment exists: door proud/recessed (depth), left/right\n"
        "     gap uneven (lateral), or top/bottom gap uneven (height).\n"
        "  c. Adjust the relevant screw by half-turns. Close the door after each half-turn to\n"
        "     assess change. Repeat until alignment is correct.\n"
        "  d. Check ALL hinges — if one is adjusted, the others may need matching adjustment.\n"
        "  e. Check for stripped screw holes: if a screw turns freely without gripping, the hole\n"
        "     is stripped. Fix with toothpick-and-PVA or a proper wood-thread repair insert.\n"
        "  f. If the hinge arm is bent, cracked, or the clip mechanism is broken: replace the hinge.\n"
        "- Real hazards FOR HINGE WORK: stripped holes cause the hinge plate to pull free\n"
        "  suddenly — test screw grip before fully tightening. Hinges on heavy doors can\n"
        "  spring shut if unsupported.\n"
        "ELSE (replacement door/panel, timber repair):\n"
        "- Replacement: remove old door, measure, obtain matching panel, fit and align.\n"
        "- Wood finishing: sand -> prime -> paint/finish. No plaster-wall language.\n"
        "- Real hazards: sharp broken MDF/timber edges; heavy door coming free; fine MDF dust."
    ),
    "plumbing": (
        "TRADE FOCUS, plumbing:\n"
        "- First step is ALWAYS isolate the water at the correct valve and release the pressure;\n"
        "  the last step is a slow re-pressurise and leak check. For an outdoor/buried supply the\n"
        "  stopcock is usually at the boundary/meter, NOT under the kitchen sink.\n"
        "- Commit to the ONE fitting method in the resource list. For any [Verify First] fitting,\n"
        "  add a step to identify the pipe material/size and fitting type before buying or cutting.\n"
        "- Real hazards: water under pressure; water near electrics; buried/unmarked services;\n"
        "  contamination of drinking water; hidden rot/damp from a long-running leak."
    ),
    "electrical": (
        "TRADE FOCUS, electrical:\n"
        "- For 230V work the workflow is: isolate at the consumer unit, prove dead, and STOP -\n"
        "  most socket/wiring repairs are notifiable and belong to a registered electrician.\n"
        "- Real hazards: electric shock/electrocution; arc flash burns; fire from loose or burnt\n"
        "  terminals. Treat exposed conductors as live until proven dead."
    ),
    "plastering/decorating": (
        "TRADE FOCUS, plastering/decorating:\n"
        "- Plaster/filler language IS correct here: fill in thin coats, leave each to dry, sand\n"
        "  between coats, feather the edges, and mist-coat before the topcoat.\n"
        "- Real hazards: fine dust (mask + ventilation); working at height if above head height;\n"
        "  skin/eye irritation from filler, bonding agent, or primer."
    ),
    "glazing": (
        "TRADE FOCUS, glazing:\n"
        "- Broken glass / failed sealed units are measured and replaced (often by a glazier).\n"
        "  Steps centre on safe removal of broken glass and fitting the measured replacement.\n"
        "- Real hazards: deep glass lacerations; falling glass shards; cuts from old beads/points."
    ),
    "sealing/weatherproofing": (
        "TRADE FOCUS, sealing/weatherproofing:\n"
        "- Steps: strip the old sealant fully, clean and dry the surface, apply ONE correct\n"
        "  sealant in a continuous bead, tool it, then allow a full cure before exposure.\n"
        "- Real hazards: knife cuts stripping old sealant; working at height; solvent fumes."
    ),
    "stone/masonry": (
        "TRADE FOCUS, stone/masonry (natural stone like marble/granite/slate/limestone, and tile):\n"
        "- The repair is STABILISE-AND-FILL, never demolish. Do NOT write hammer/chisel steps for a\n"
        "  crack repair, and do NOT mention removing 'loose or damaged marble', that is wrong; the\n"
        "  crack stays in place, the epoxy bridges it.\n"
        "- Correct sequence: 1) dust the crack out (vacuum + soft brush, no water yet). 2) Degrease\n"
        "  with acetone or IPA on a lint-free cloth, leave fully dry. 3) Mask each side of the\n"
        "  crack with low-tack tape, leaving a 1-2 mm gap along the crack line, this stops epoxy\n"
        "  staining the surrounding polish. 4) Mix knife-grade two-part stone/marble epoxy on a\n"
        "  scrap, tint with stone-dust/pigment to match. 5) Press epoxy firmly into the crack with\n"
        "  a small palette knife or razor blade, leaving it slightly PROUD of the surface. 6) Let\n"
        "  cure to the manufacturer's full cure time (typically 30-60 min hands-off, 24 h hard\n"
        "  cure). 7) Trim flush with a razor blade held flat to the stone. 8) Wet-sand polished\n"
        "  stone only, 400 -> 800 -> 1500 -> 3000 grit, rinsing between grits; SKIP sanding on\n"
        "  honed/matte stone or you will create a shiny patch. 9) Polish back to original sheen\n"
        "  with tin-oxide marble polish or a diamond polishing pad on a damp cloth.\n"
        "- NEVER use plaster-wall language (skim, mist coat, feathering, 'pink to white').\n"
        "- For acidic CLEANING products, warn explicitly: no vinegar, lemon juice, bleach or\n"
        "  general bathroom cleaner on marble/limestone, they etch the polish permanently.\n"
        "- Real hazards: epoxy resin skin/eye irritation (gloves + ventilation); solvent fumes\n"
        "  from acetone (open a window, no naked flames); sharp razor blade; if the stone is a\n"
        "  worktop or hearth used near heat, mention the epoxy's heat tolerance limit.\n"
        "- For cracked GROUT between tiles (not the tile itself): rake the old grout out with a\n"
        "  grout saw, vacuum, then apply matching grout, smooth with a finger or grout float, wipe\n"
        "  haze off after the initial set."
    ),
    "structural": (
        "TRADE FOCUS, structural:\n"
        "- Do NOT write a DIY repair sequence. The workflow is: make the area safe, avoid loading\n"
        "  it, and get a structural engineer/builder to assess before any work.\n"
        "- Real hazards: collapse; concealed load paths; falling debris."
    ),
    "general/other": (
        "TRADE FOCUS, general household repair:\n"
        "- Anchor every step in the specific 'Likely root causes' from Stage 2. Do not write\n"
        "  generic household steps — write steps for the specific fault identified.\n"
        "- IF hinge-related: apply the full hinge adjustment sequence above (concealed hinge\n"
        "  depth/side/height screws, check for stripped holes, replace only if physically broken).\n"
        "- IF furniture assembly/flat-pack: steps follow the specific joint failure (loose cam,\n"
        "  missing dowel, snapped connector). Do not use carpentry finishing language.\n"
        "- Real hazards: only those genuinely present in the specific repair — stripped screw\n"
        "  holes (sudden plate pull-free), heavy doors/panels (finger trap), small fixings\n"
        "  (swallowing risk if children present). No generic platitudes."
    ),
}

_STAGE_3_TRADE_GENERIC = (
    "TRADE FOCUS, {tc}:\n"
    "- Keep steps specific to what was actually observed; do not borrow language or methods from\n"
    "  other trades. Add a verification step wherever the resource list says [Verify First].\n"
    "- Real hazards: identify the genuine hazards for THIS specific repair (mechanical, electrical,\n"
    "  water, height, dust, sharp edges), no generic platitudes."
)


def _stage_3_trade_profile(trade_category: str, problem_type: str = "") -> str:
    """Return the per-trade step/hazard guidance block for the Stage 3 prompt."""
    tc = (trade_category or "general/other").strip().lower()
    pt = (problem_type or "").lower()
    if tc == "general/other" and any(w in pt for w in ("hinge", "cabinet door", "cupboard door", "wardrobe door")):
        tc = "carpentry/joinery"
    return _STAGE_3_TRADE_PROFILES.get(tc, _STAGE_3_TRADE_GENERIC.format(tc=tc))


async def _stage_3_steps(
    stage_1: Stage1Output,
    stage_2: Stage2Output,
) -> Stage3Output:
    """
    Stage 3: step-by-step repair instructions using Llama 3.1 (text model).

    REAL, receives the full context from both Stage 1 (diagnosis) and Stage 2
    (resources) and generates a complete, ordered repair procedure.

    Context passed forward:
      Both stage_1 and stage_2 outputs are included so the model knows:
        - exactly what the problem is and how severe it is
        - which tools and materials the user already has available
        - what protective gear the user will be wearing
      This allows the steps to reference specific tools and materials by name.

    Stage 3 is the most generative stage, it produces a list of RepairStep
    objects, each with a step number, title, description, and an optional
    per-step safety warning.  Llama 3.1 handles this structured generation
    task well with explicit field descriptions in the prompt.
    """
    # Build a rich context block from both previous stages. Confidence, visual
    # evidence and uncertainty are now carried through from Stage 1 so Stage 3 can
    # hedge / add verification when the diagnosis is not certain. Tool/material
    # lists keep their [tier] tags so Stage 3 can tell confirmed from verify-first.
    _vis = "; ".join(stage_1.visual_evidence[:8]) if stage_1.visual_evidence else "none recorded"
    _unc = stage_1.uncertainty_notes.strip() if stage_1.uncertainty_notes else "none recorded"
    _reason = stage_1.confidence_reason.strip() if stage_1.confidence_reason else "not given"
    _opt_tools = ", ".join(stage_2.optional_tools) if stage_2.optional_tools else "none"
    _opt_mats = ", ".join(stage_2.optional_materials) if stage_2.optional_materials else "none"
    _concerns = "; ".join(stage_1.safety_concerns) if stage_1.safety_concerns else "none named"
    # Video-path extra fields
    _key_moments_s3 = getattr(stage_1, "key_moments", []) or []
    _key_moments_s3_block = (
        "\nKey moments (video)    :\n" + "\n".join(f"  - {m}" for m in _key_moments_s3)
        if _key_moments_s3 else ""
    )
    _likely_causes = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(stage_2.likely_causes)) \
        if stage_2.likely_causes else "  (not available)"
    _rcr = stage_2.repair_confidence_reasoning or "Not provided."
    context = (
        f"Problem                : {stage_1.problem_type}\n"
        f"Trade                  : {stage_1.trade_category}\n"
        f"Severity               : {stage_1.severity}\n"
        f"Diagnosis confidence   : {stage_1.confidence}% ({_reason})\n"
        f"Risk level (model)     : {stage_1.risk_level}\n"
        f"Hidden-damage risk     : {stage_1.hidden_damage_risk}\n"
        f"DIY suitability (model): {stage_1.diy_suitability}\n"
        f"Pro assessment advised : {stage_1.professional_assessment_recommended}\n"
        f"Safety concerns (model): {_concerns}\n"
        f"Visually confirmed     : {_vis}\n"
        f"NOT confirmed/uncertain: {_unc}\n"
        f"Observations           :\n"
        + "\n".join(f"  - {obs}" for obs in stage_1.observations)
        + _key_moments_s3_block
        + f"\n\nLikely root causes (Stage 2 diagnosis):\n{_likely_causes}\n"
        f"Repair confidence reasoning: {_rcr}\n"
        + f"\nTools (tier-tagged)     : {', '.join(stage_2.tools)}\n"
        f"Optional tools          : {_opt_tools}\n"
        f"Materials (tier-tagged) : {', '.join(stage_2.materials)}\n"
        f"Optional materials      : {_opt_mats}\n"
        f"Protective gear         : {', '.join(stage_2.protective_gear) if stage_2.protective_gear else 'none required'}\n"
        f"Estimated time          : {stage_2.estimated_time_minutes} minutes"
    )

    _trade_block = _stage_3_trade_profile(stage_1.trade_category, stage_1.problem_type or "")
    system_prompt = (
        "You are an experienced UK tradesperson writing repair instructions for a homeowner.\n"
        "Be direct, specific, practical and honest. UK context throughout.\n\n"
        "USE THE EVIDENCE YOU ARE GIVEN:\n"
        "- The resource list is TIER-TAGGED. Treat [Confirmed] items as part of the plan. Treat\n"
        "  [Likely] items as probable but say 'likely'. Treat [Verify First] and [Do Not Buy Yet]\n"
        "  as CONDITIONAL, do NOT write steps that assume them as fact; instead add a step to\n"
        "  check/measure first, then say what to do depending on the result.\n"
        "- Follow the repair-vs-replace decision implied by the tiers: if a REPLACEMENT part is\n"
        "  [Confirmed]/[Likely], the steps are remove -> obtain the matching part -> fit and align,\n"
        "  NOT patch-and-fill a missing section. Only describe filling/patching when a patch\n"
        "  material is the confirmed approach.\n"
        "- Read 'Diagnosis confidence' and 'NOT confirmed/uncertain'. When confidence is lower or\n"
        "  key facts are unconfirmed, add explicit verify/expose/measure steps BEFORE any cutting,\n"
        "  drilling or removal, prefer reversible actions first, and hedge the wording.\n\n"
        "STEP WRITING RULES:\n"
        "1. ONE action per step, never combine two actions into one step.\n"
        "2. Begin with a clear PREP step suited to the trade (open the cabinet fully, clear the\n"
        "   work area, lay a dust sheet, isolate the water/circuit). Keep it brief.\n"
        "3. Where a material must DRY or CURE, make that a separate step with a realistic wait.\n"
        "4. Reference tools/materials from the resource list by their PLAIN NAME. NEVER write\n"
        "   the tier prefix ('[Confirmed]', '[Likely]', '[Verify First]', '[Do Not Buy Yet]')\n"
        "   inside any step. For a [Verify First] item, write the check into the step itself.\n"
        "5. EVERY STEP MUST ANSWER THREE QUESTIONS — all three, in order:\n"
        "   - WHAT: the exact physical action and which tool ('Turn the depth-adjust screw\n"
        "     clockwise with a Pozi-2 screwdriver')\n"
        "   - WHY: the mechanical reason this step is needed ('to move the door inward so it\n"
        "     sits flush with the cabinet frame')\n"
        "   - RESULT: what correct completion looks like ('the door should close without\n"
        "     catching and sit level with the adjacent panel')\n"
        "   Bad example: 'Adjust the hinge screws.'\n"
        "   Good example: 'Turn the depth-adjust screw (the one set back furthest on the hinge\n"
        "   arm) clockwise by half a turn with a Pozi-2 screwdriver to pull the door inward\n"
        "   towards the frame. Close the door — it should now sit flush rather than proud.'\n"
        "6. Step titles are short action phrases, max 8 words. No 'check', 'examine', 'inspect'\n"
        "   in the title unless the step is purely diagnostic.\n"
        "7. 6 to 9 steps. Do not pad with generic steps like 'clean the area', 'gather tools'.\n"
        "8. Use direct imperatives ('press', 'turn', 'tighten', 'slide'). No vague verbs alone.\n"
        "9. STAY ON THE SPECIFIC PROBLEM. Root your steps in the 'Likely root causes' from\n"
        "   Stage 2. Do NOT write steps about things the camera saw that are unrelated to the\n"
        "   identified fault. A hinge-alignment job has no 'wipe down the cabinet exterior'\n"
        "   step; a marble crack has no 'inspect the wooden frame' step.\n\n"
        f"{_trade_block}\n\n"
        "RISK & SAFETY — include ONLY what is genuinely relevant to THIS repair:\n"
        "safety_notes must be an array of 1 to 4 PLAIN STRINGS (never objects/dicts).\n"
        "ONLY include a note when there is a real, specific hazard for this job. It is correct\n"
        "to return 1 or 2 notes if only 1 or 2 genuine hazards exist. Do NOT pad to reach a\n"
        "minimum count with generic warnings.\n"
        "DO NOT include any of the following unless the evidence specifically supports them:\n"
        "  - 'Sharp edges' or 'potential injury' (generic)\n"
        "  - 'Proceed with caution' or 'be careful' (meaningless)\n"
        "  - 'Use the correct tool' or 'wear gloves' without a specific reason\n"
        "  - 'Check what the photo/video cannot show' or 'person's expertise uncertain' (not a hazard)\n"
        "  - Any note about uncertainty, unknown skill level, or 'may be worse than it looks'\n"
        "Each note that IS included must follow this shape:\n"
        "  what is dangerous -> why -> the likely consequence -> the precaution.\n"
        "Example of a GOOD note: 'If the hinge mounting screws are stripped, the hinge plate\n"
        "can pull free unexpectedly — stop adjusting and fill the holes before re-tightening\n"
        "or the door could drop and trap fingers.'\n"
        "Example of a BAD note (do NOT write): 'Always use the correct tools for the job.'\n"
        "Each per-step 'warning' covers a step-specific hazard or quality check, or is null.\n\n"
        "DIY SUITABILITY:\n"
        "FIRST honour the diagnosis's own risk assessment in the context above: if 'DIY suitability\n"
        "(model)' is not_recommended, or 'Pro assessment advised' is True, or 'Risk level (model)' is\n"
        "high/critical, then diy_recommended MUST be false, set professional_type accordingly, and\n"
        "frame the steps as make-safe-and-get-assessed rather than a full DIY repair. If 'Hidden-\n"
        "damage risk' is high, add explicit expose/verify steps before anything irreversible.\n"
        "Set diy_recommended=false (and professional_type to the exact trade) if the repair involves:\n"
        "  - mains electrical (230V), consumer units, or fuse boxes -> 'NICEIC-registered electrician'\n"
        "  - gas pipes, boilers or gas appliances -> 'Gas Safe engineer'\n"
        "  - structural load-bearing walls, beams or foundations -> 'structural engineer'\n"
        "  - significant water near electrics, or buried mains supply work -> 'CIPHE-registered plumber'\n"
        "  - asbestos (pre-2000 textured coatings, tiles, pipe lagging) -> licensed asbestos removal\n"
        "ALSO lean towards 'get it assessed' when severity is HIGH and diagnosis confidence is LOW,\n"
        "even outside those categories, say so honestly. If diy_recommended is true, professional_type\n"
        "is null.\n"
        "when_to_call_professional is ALWAYS required: give the specific stop/escalate signs FOR THIS\n"
        "repair and trade, the dangers that mean the user should down tools and call a pro (for\n"
        "example: the underlying structure/carcass is cracked, the damage is bigger than it first\n"
        "looked, or a hazard specific to this trade appears). Do NOT copy signs from other trades\n"
        "(e.g. do not mention water/gas/electrics unless THIS repair actually involves them). If\n"
        "diy_recommended is false, explain why.\n\n"
        "Return a single JSON object with exactly these fields:\n"
        "  steps                      : array of step objects, each containing:\n"
        "    step_number              : integer starting at 1\n"
        "    title                    : short action phrase, max 8 words\n"
        "    description              : practical instruction following the rules above\n"
        "    warning                  : specific safety or quality warning string, or null\n"
        "  safety_notes               : array of 1 to 4 plain reasoned strings for GENUINE\n"
        "                               hazards only (hazard->why->consequence->precaution),\n"
        "                               NOT objects; omit generic warnings entirely\n"
        "  difficulty_level           : exactly one of: beginner, intermediate, advanced\n"
        "  when_to_call_professional  : string, ALWAYS required, never null (stop/escalate signs,\n"
        "                               or why a professional is needed)\n"
        "  diy_recommended            : boolean\n"
        "  professional_type          : string or null\n\n"
        "Return ONLY the JSON object.  No explanation, no markdown, no extra text."
    )

    user_prompt = (
        f"Based on this repair scenario:\n\n{context}\n\n"
        "Write the complete step-by-step repair instructions."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    return await _call_stage(
        model=STAGE_3_MODEL,
        messages=messages,
        schema_class=Stage3Output,
        stage_name="stage_3 (Llama 3.1)",
        extra_options={
            "num_ctx":     STAGE_3_NUM_CTX,
            "num_predict": STAGE_3_NUM_PREDICT,
            "temperature": 0.0,   # deterministic steps + risk
        },
    )
