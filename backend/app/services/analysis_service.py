"""
Three-stage AI analysis pipeline — all models served locally via Ollama.

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
  - optional  (helpful but not required — shown as "Nice to have" in the UI)

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
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Analysis, Session as RepairSession, Upload
from app.schemas.analysis import (
    AnalysisResult,
    RepairStep,
    Stage1Output,
    Stage2Output,
    Stage3Output,
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

# Per-stage context window and output budgets — read from .env so they can be
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

# Upload directory — resolved the same way as uploads_service.py so paths match.
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"

# One shared async Ollama client — created once at module load, reused for all
# pipeline calls.  AsyncClient is safe to share across coroutines.
_ollama = ollama.AsyncClient(host=OLLAMA_BASE_URL)

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


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (unchanged from Session 1)
# ─────────────────────────────────────────────────────────────────────────────

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
    # 1. Verify ownership — raises 404 if case not found or not owned
    await _require_owned_case(db, case_id, user_id)

    # 2. Refuse if no photos exist — the vision stage needs at least one image
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

    # 5. Extract just the storage paths — do not pass ORM objects across session
    #    boundaries.  The request session closes when this function returns; the
    #    background task creates its own session and only needs these strings to
    #    locate the image files on disk.
    upload_paths = [u.storage_path for u in uploads]

    # 6. Launch pipeline — non-blocking, runs concurrently with other requests
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
    model by 60–80% for typical phone photos, cutting Stage 1 inference time
    noticeably with no meaningful loss in diagnostic accuracy — home repair
    problems are large, visible features that don't need full-resolution images.

    Compression pipeline per image:
      1. Open with Pillow.
      2. Convert non-RGB modes (RGBA, P, etc.) to RGB so JPEG save works.
      3. thumbnail(1024, 1024) — preserves aspect ratio, in-place.
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
        log.warning("Pillow not installed — images will be sent at full resolution. "
                    "Run: pip install Pillow")

    # 512px is sufficient for home repair diagnosis — the damage features are large
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
                    log.warning("Pillow resize failed for %s (%s) — using raw bytes",
                                path, pil_exc)

            # Fallback: send raw bytes without resize
            result.append(base64.b64encode(raw_bytes).decode("utf-8"))

        except OSError as exc:
            log.warning("Could not read image %s: %s", full, exc)
    return result


def _clean_json(text: str) -> str:
    """
    Strip markdown fences and extract the first JSON object from the response.
    Local models sometimes wrap JSON in ```json ... ``` or add prose before/after.
    """
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find the outermost {...} block in case the model added prose around the JSON
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]

    return text


def _heal_truncated_json(text: str) -> str:
    """
    Attempt to repair JSON that was cut off mid-generation (truncated by num_predict
    or num_ctx limits).  Closes any unclosed strings, arrays, and objects.

    This is a best-effort recovery — it will not always produce semantically
    correct JSON, but it often lets the Pydantic validator extract partial
    data rather than failing completely.
    """
    text = text.rstrip()
    if not text:
        return text

    # If we're inside an unclosed string, close it
    # Heuristic: count unescaped double-quotes; if odd, we're mid-string
    in_string   = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        text += '"'     # close the open string

    # Close open arrays and objects in reverse order of opening
    stack = []
    in_string   = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            continue
        if ch == '"' and in_string:
            in_string = False
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    # Append the closing tokens in reverse
    text += "".join(reversed(stack))
    return text


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
    schema_class : Pydantic model class — used to validate and construct the result
    stage_name   : human-readable label for log messages
    """
    last_error = None
    last_raw   = ""

    for attempt in range(PIPELINE_RETRY_COUNT + 1):
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
            data    = json.loads(cleaned)
            return schema_class(**data)

        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning(
                "%s attempt %d/%d — parse error: %s | raw[:80]=%r",
                stage_name, attempt + 1, PIPELINE_RETRY_COUNT + 1,
                exc, last_raw[:80],
            )

            # Before retrying, attempt to heal a truncated response.
            # If healing succeeds we return immediately without using another
            # model call — cheaper and faster than a full retry.
            if last_raw:
                try:
                    healed = _heal_truncated_json(_clean_json(last_raw))
                    data   = json.loads(healed)
                    result = schema_class(**data)
                    log.info("%s — healed truncated JSON successfully", stage_name)
                    return result
                except Exception:
                    pass    # healing failed — proceed to retry

            if attempt < PIPELINE_RETRY_COUNT:
                await asyncio.sleep(1.0)

        except Exception as exc:
            raise RuntimeError(
                f"Ollama call failed for {stage_name} (model={model}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    raise ValueError(
        f"{stage_name} failed after {PIPELINE_RETRY_COUNT + 1} attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR  (unchanged from Session 1 except provider/model fields)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_pipeline(
    analysis_id: int,
    case_id: int,
    upload_paths: list[str],
    text_context: str | None = None,
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
      - the pipeline stops — it does not try to continue with partial results

    This function creates its own DB session.  The analysis_id integer is the
    only thing that crosses the session boundary safely.
    """
    start = time.monotonic()

    async with AsyncSessionLocal() as db:
        try:
            # Load the analysis row in this new session
            analysis = await db.get(Analysis, analysis_id)
            if analysis is None:
                return  # row was deleted between creation and pipeline start

            # Mark running
            analysis.status = "running"
            await db.commit()

            # ── STAGE 1 — Image / Problem Analysis ───────────────────────────
            analysis.pipeline_stage = _S1_RUNNING
            await db.commit()

            _t_s1 = time.monotonic()
            _s1_fallback = False
            try:
                s1: Stage1Output = await _stage_1_analyse(upload_paths, text_context)
            except (ValueError, RuntimeError) as _s1_exc:
                # Stage 1 failed all retries. Rather than aborting the whole
                # pipeline, create a minimal Stage1Output from the user's text
                # context (if any) so Stages 2 and 3 can still run and produce
                # a useful repair guide. The frontend will show a warning.
                log.warning(
                    "Stage 1 failed — using graceful fallback. Error: %s", _s1_exc
                )
                _problem = (
                    text_context.strip()
                    if text_context and text_context.strip()
                    else "Repair issue (could not analyse image automatically)"
                )
                s1 = Stage1Output(
                    problem_type=_problem,
                    trade_category="other",
                    severity="medium",
                    observations=[
                        "Image could not be automatically analysed.",
                        "Please describe the problem in more detail using the chat.",
                    ],
                )
                _s1_fallback = True
            _ms_s1 = int((time.monotonic() - _t_s1) * 1000)
            log.info("Stage 1 in %.1fs%s", _ms_s1 / 1000,
                     " [fallback]" if _s1_fallback else "")

            analysis.stage_outputs = {
                **analysis.stage_outputs,
                "stage_1": s1.model_dump(),
                **({"stage_1_fallback": True} if _s1_fallback else {}),
            }
            analysis.pipeline_stage = _S1_DONE
            await db.commit()

            # Auto-title: overwrite the placeholder "Diagnosing…" with the AI result
            case_row = await db.get(RepairSession, case_id)
            if case_row is not None:
                case_row.title    = s1.problem_type
                case_row.category = s1.trade_category
                await db.commit()

            # ── STAGE 2 — Tools, Materials, Protective Gear ───────────────────
            analysis.pipeline_stage = _S2_RUNNING
            await db.commit()

            _t_s2 = time.monotonic()
            s2: Stage2Output = await _stage_2_resources(s1)
            _ms_s2 = int((time.monotonic() - _t_s2) * 1000)
            log.info("Stage 2 complete in %.1fs", _ms_s2 / 1000)

            analysis.stage_outputs = {**analysis.stage_outputs, "stage_2": s2.model_dump()}
            analysis.pipeline_stage = _S2_DONE
            await db.commit()

            # ── STAGE 3 — Step-by-step Repair Instructions ────────────────────
            analysis.pipeline_stage = _S3_RUNNING
            await db.commit()

            _t_s3 = time.monotonic()
            s3: Stage3Output = await _stage_3_steps(s1, s2)
            _ms_s3 = int((time.monotonic() - _t_s3) * 1000)
            log.info("Stage 3 complete in %.1fs", _ms_s3 / 1000)

            analysis.stage_outputs = {**analysis.stage_outputs, "stage_3": s3.model_dump()}
            analysis.pipeline_stage = _S3_DONE
            await db.commit()

            # ── FINALISE ──────────────────────────────────────────────────────
            combined = AnalysisResult(stage_1=s1, stage_2=s2, stage_3=s3)
            _ms_total = int((time.monotonic() - start) * 1000)

            analysis.structured_result  = combined.model_dump()
            analysis.status             = "complete"
            analysis.pipeline_stage     = "complete"
            analysis.processing_time_ms = _ms_total
            analysis.provider           = "ollama"
            analysis.model              = f"{STAGE_1_MODEL} | {STAGE_2_MODEL} | {STAGE_3_MODEL}"
            # Store per-stage timing in validation_issues as a JSON note (non-breaking)
            analysis.validation_issues  = (
                f'{{"stage_1_ms":{_ms_s1},"stage_2_ms":{_ms_s2},"stage_3_ms":{_ms_s3},'
                f'"total_ms":{_ms_total}}}'
            )
            log.info(
                "Pipeline complete — S1:%ds S2:%ds S3:%ds total:%ds",
                _ms_s1 // 1000, _ms_s2 // 1000, _ms_s3 // 1000, _ms_total // 1000,
            )
            await db.commit()

        except Exception as exc:
            # Catch all unhandled exceptions — never let the background task
            # crash silently.  Write the error to the DB so the frontend can
            # surface it and the user can decide whether to re-trigger.
            log.error("Pipeline failed for analysis_id=%d: %s", analysis_id, exc, exc_info=True)
            try:
                analysis.status           = "failed"
                analysis.validation_issues = f"{type(exc).__name__}: {exc}"
                await db.commit()
            except Exception:
                pass  # if even the error write fails, there is nothing more we can do


# ─────────────────────────────────────────────────────────────────────────────
# STAGE FUNCTIONS  (real Ollama calls — replaces Session 1 stubs)
#
# ─── DESIGN PRINCIPLE ─────────────────────────────────────────────────────────
# Each function:
#   - has an identical signature to the Session 1 stub
#   - constructs a system prompt that defines the model's role
#   - constructs a user prompt that includes context from previous stages
#   - explicitly describes the expected JSON schema in the prompt
#     (belt-and-suspenders alongside format="json")
#   - calls _call_stage() which handles retry, parsing, and validation
#   - returns a validated Pydantic object — never raw text
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

    REAL — reads each uploaded image from disk, encodes as base64, and sends
    to Qwen2.5-VL via Ollama.  The model examines the images and classifies
    the home repair problem into a structured Stage1Output.

    Image handling:
      Images are base64-encoded and passed in the Ollama message's 'images'
      field.  Qwen2.5-VL can process multiple images in a single call, so all
      uploads for the case are sent together for a holistic diagnosis.

    trade_category constraint:
      The value must be from a fixed vocabulary so it can drive downstream
      structured queries (e.g. nearby services geo-lookup) without another
      LLM call.  The prompt enforces this — the model must pick one label
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
        "- Only describe what is LITERALLY VISIBLE in the image. Do not invent problems.\n"
        "- If an image shows a door, lock, handle, hinge, or door hardware — classify it as carpentry.\n"
        "- If an image shows a pipe, tap, drain, or water fitting — classify it as plumbing.\n"
        "- If an image shows wiring, sockets, switches, or circuit breakers — classify it as electrical.\n"
        "- If an image shows a wall, ceiling, or floor — classify it as structural or painting.\n"
        "- For the observations, list only what you can concretely see in the photo.\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  problem_type        : string — a short description of the specific item that needs repair\n"
        "                        (e.g. 'Loose door handle', 'Leaking pipe joint', 'Cracked wall plaster',\n"
        "                        'Broken door lock', 'Worn door hinge')\n"
        "  trade_category      : string — exactly one of:\n"
        "                        plumbing, electrical, carpentry, appliance, structural,\n"
        "                        painting, roofing, other\n"
        "  severity            : string — exactly one of: low, medium, high\n"
        "  observations        : array of strings — 3 to 5 specific visual observations\n"
        "                        describing exactly what you see in the image\n\n"
        "Return ONLY the JSON object.  No explanation, no markdown, no extra text."
    )

    user_prompt = "Examine the attached image(s) carefully. What physical object or fixture is shown, and what is damaged or needs repair?"
    if text_context:
        user_prompt = (
            f"The user has described the problem as: {text_context}\n\n"
            + user_prompt
        )
    user_prompt += " Return your diagnosis as a JSON object matching the schema described."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
            "images": images_b64,       # base64-encoded image bytes — Ollama vision API
        },
    ]

    # CRITICAL: num_ctx must cover system prompt (~200 tok) + image tokens
    # (~500-700 per image for Qwen2.5-VL) + JSON response (~300 tok).
    # The old value of 512 caused the model to truncate mid-JSON on every call.
    # num_predict=400 is sufficient for a 4-field JSON response; keeping it
    # small reduces inference time without risking output truncation.
    return await _call_stage(
        model=STAGE_1_MODEL,
        messages=messages,
        schema_class=Stage1Output,
        stage_name="stage_1 (Qwen2.5-VL)",
        extra_options={
            "num_ctx":     STAGE_1_NUM_CTX,     # 4096 — the real fix
            "num_predict": STAGE_1_NUM_PREDICT,  # 400  — short JSON output
        },
    )


async def _stage_2_resources(stage_1: Stage1Output) -> Stage2Output:
    """
    Stage 2: tools / materials / protective gear using Mistral (text model).

    REAL — receives Stage 1's structured output as context and asks Mistral
    to list what the user needs before starting the repair.

    Context passed forward:
      The stage_1 object is serialised to a readable text block so the model
      understands the full diagnosis — not just the problem type, but also
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
    context = (
        f"Problem type  : {stage_1.problem_type}\n"
        f"Trade         : {stage_1.trade_category}\n"
        f"Severity      : {stage_1.severity}\n"
        f"Observations  :\n"
        + "\n".join(f"  - {obs}" for obs in stage_1.observations)
    )

    system_prompt = (
        "You are a home repair planning expert. "
        "Given a diagnosis of a home repair problem, your job is to list exactly "
        "what the user needs before they can start the repair.\n\n"
        "Separate tools and materials into ESSENTIAL (must-have to complete the job) "
        "and OPTIONAL (helpful but the job can be done without them).\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  tools                   : array of strings — ESSENTIAL tools you cannot do without\n"
        "                            (e.g. ['Adjustable spanner', 'Pipe cutter'])\n"
        "  optional_tools          : array of strings — OPTIONAL tools that make the job easier\n"
        "                            (e.g. ['Pipe vice', 'Deburring tool'])\n"
        "  materials               : array of strings — ESSENTIAL consumable materials\n"
        "                            (e.g. ['PTFE tape', 'Replacement connector'])\n"
        "  optional_materials      : array of strings — OPTIONAL materials that improve the result\n"
        "                            (e.g. ['Thread sealant compound', 'Pipe lagging'])\n"
        "  protective_gear         : array of strings — safety equipment\n"
        "                            (e.g. ['Rubber gloves', 'Safety glasses'])\n"
        "  estimated_time_minutes  : integer — realistic time for a competent DIYer\n\n"
        "Return ONLY the JSON object.  No explanation, no markdown, no extra text."
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
        },
    )


async def _stage_3_steps(
    stage_1: Stage1Output,
    stage_2: Stage2Output,
) -> Stage3Output:
    """
    Stage 3: step-by-step repair instructions using Llama 3.1 (text model).

    REAL — receives the full context from both Stage 1 (diagnosis) and Stage 2
    (resources) and generates a complete, ordered repair procedure.

    Context passed forward:
      Both stage_1 and stage_2 outputs are included so the model knows:
        - exactly what the problem is and how severe it is
        - which tools and materials the user already has available
        - what protective gear the user will be wearing
      This allows the steps to reference specific tools and materials by name.

    Stage 3 is the most generative stage — it produces a list of RepairStep
    objects, each with a step number, title, description, and an optional
    per-step safety warning.  Llama 3.1 handles this structured generation
    task well with explicit field descriptions in the prompt.
    """
    # Build a rich context block from both previous stages
    context = (
        f"Problem       : {stage_1.problem_type}\n"
        f"Trade         : {stage_1.trade_category}\n"
        f"Severity      : {stage_1.severity}\n"
        f"Observations  :\n"
        + "\n".join(f"  - {obs}" for obs in stage_1.observations)
        + f"\n\nTools available    : {', '.join(stage_2.tools)}\n"
        f"Materials           : {', '.join(stage_2.materials)}\n"
        f"Protective gear     : {', '.join(stage_2.protective_gear)}\n"
        f"Estimated time      : {stage_2.estimated_time_minutes} minutes"
    )

    system_prompt = (
        "You are a home repair instruction writer. "
        "Given a repair diagnosis and a list of available tools and materials, "
        "your job is to write clear, safe, step-by-step repair instructions "
        "for a competent DIYer.\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  steps                      : array of step objects, each containing:\n"
        "    step_number              : integer starting at 1\n"
        "    title                    : short action title (e.g. 'Shut off water supply')\n"
        "    description              : full instruction paragraph\n"
        "    warning                  : safety warning string, or null if not needed\n"
        "  safety_notes               : array of 3–5 global safety notes\n"
        "                              that apply throughout the entire repair\n"
        "  difficulty_level           : exactly one of: beginner, intermediate, advanced\n"
        "  when_to_call_professional  : string — clear condition under which the\n"
        "                              user should stop and call a qualified tradesperson\n\n"
        "Make instructions specific and actionable.  Reference the tools and materials "
        "by name.  Add a warning field only when there is a genuine safety risk.\n\n"
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
        },
    )
