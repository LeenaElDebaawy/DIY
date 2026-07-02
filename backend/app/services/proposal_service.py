"""
Advisory fix-proposal service — admin-only, write-nothing.

Calls local Ollama to generate a structured, human-reviewable proposal
for addressing a recurring quality pattern detected by the QA pipeline.

Gated by JUDGE_ENABLED=true (same flag as the judge — needs the same
Ollama model to be available). Writes nothing to disk or database.
Every response includes a safety_note asserting advisory-only status.
"""
import json
import logging
import os
import re

from ollama import AsyncClient

log = logging.getLogger(__name__)

PROPOSAL_MODEL   = os.getenv("JUDGE_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
PROPOSAL_ENABLED = os.getenv("JUDGE_ENABLED", "false").lower() == "true"

_SAFETY_NOTE = (
    "This is an advisory QA proposal only. "
    "It does not automatically change the repair pipeline. "
    "All suggestions require human review and validation before implementation."
)

_REQUIRED_FIELDS = (
    "title", "problem_observed", "evidence",
    "suggested_change", "pipeline_area",
    "expected_impact", "risks_or_cautions",
)


def _build_prompt(ctx: dict) -> str:
    signal = ctx.get("signal", "")
    signal_explain = {
        "both":  "BOTH the automated judge AND real user feedback flag this — treat as high confidence",
        "judge": "The automated judge flags this pattern",
        "user":  "Real user feedback flags this pattern",
    }.get(signal, "Quality signals flag this pattern")

    trigger_readable = ", ".join(
        t.replace("_", " ") for t in (ctx.get("trigger_issues") or [])
    )
    case_ids = ", ".join(
        f"#{i}" for i in (ctx.get("example_case_ids") or [])[:5]
    ) or "none recorded"

    return f"""You are a senior QA advisor for Knitra, an AI home repair assistant for UK homeowners.
Knitra's pipeline: Stage 1 extracts problem type, observations and safety concerns from photos/video.
Stage 2 routes to an output mode (diy / verify_first / assessment_first / active_emergency / etc.).
Stage 3 generates the repair guidance, assessment, or emergency response shown to the user.

A recurring quality pattern has been detected that a human developer needs to review and address.

PATTERN DETAILS
---------------
Suggestion: {ctx.get("suggestion", "")}
Pipeline area affected: {ctx.get("pipeline_area", "")}
Signal source: {signal.upper()} — {signal_explain}
Trigger issues: {trigger_readable}
Total occurrences: {ctx.get("total_occurrences", 0)}
Example case IDs: {case_ids}

Write a specific, actionable advisory fix proposal for a developer to review.
This will NOT be applied automatically. It is for human review only.
Do not write code. Do not modify any files. Be specific to UK home repair context.

Respond with valid JSON only — no text before or after. Use exactly this structure:
{{
  "title": "Short title (max 10 words) describing what this proposal addresses",
  "problem_observed": "One concise paragraph: what the QA data shows is going wrong in the pipeline",
  "evidence": "Specific evidence: which trigger issues fire, what signals confirm the pattern, example cases",
  "suggested_change": "Concrete developer instruction: which prompt or logic to update and what to add or change. Name the pipeline stage. Quote example wording if helpful. UK context where relevant.",
  "pipeline_area": "Exact part: Stage 1 prompt / Stage 2 routing logic / Stage 3 DIY prompt / Stage 3 assessment prompt / Stage 3 emergency prompt / etc.",
  "expected_impact": "What improvement is expected if this change is made and how to verify it via the judge or user feedback",
  "risks_or_cautions": "What to test carefully after making this change; edge cases to watch; other modes that might be affected",
  "safety_note": "{_SAFETY_NOTE}"
}}"""


async def generate_fix_proposal(ctx: dict) -> dict:
    """Call local Ollama to generate an advisory proposal. Reads nothing, writes nothing."""
    if not PROPOSAL_ENABLED:
        return {
            "error": "proposal_disabled",
            "message": (
                f"AI proposals require JUDGE_ENABLED=true in your local .env "
                f"and {PROPOSAL_MODEL} running in Ollama."
            ),
            "ai_generated": False,
        }

    try:
        client = AsyncClient(host=OLLAMA_BASE_URL)
        resp = await client.chat(
            model=PROPOSAL_MODEL,
            messages=[{"role": "user", "content": _build_prompt(ctx)}],
            options={"temperature": 0.3, "num_predict": 900, "num_gpu": 0},
            keep_alive=0,
        )
        raw = (resp.message.content or "").strip()
    except Exception as exc:
        log.warning("fix_proposal: Ollama call failed: %s", exc)
        return {
            "error": "ollama_unavailable",
            "message": f"Could not reach Ollama ({PROPOSAL_MODEL}): {exc!s}",
            "ai_generated": False,
        }

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        log.warning("fix_proposal: no JSON block in response. raw=%r", raw[:300])
        return {
            "error": "parse_failed",
            "message": "Model response did not contain valid JSON. Try again.",
            "ai_generated": False,
        }

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.warning("fix_proposal: JSON parse error: %s", exc)
        return {
            "error": "parse_failed",
            "message": f"Could not parse model JSON: {exc!s}",
            "ai_generated": False,
        }

    for field in _REQUIRED_FIELDS:
        if field not in data:
            data[field] = "—"
    data["safety_note"] = _SAFETY_NOTE
    data["ai_generated"] = True
    return data
