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
        case_row = None   # initialised here so the except block can always reference it
        try:
            # Load the analysis row and its parent case in this new session
            analysis = await db.get(Analysis, analysis_id)
            if analysis is None:
                return  # row was deleted between creation and pipeline start

            # Load case_row early so we can mirror analysis status throughout.
            # This keeps Session.status in sync so the My Cases status filter works.
            case_row = await db.get(RepairSession, case_id)

            # Mark running — mirror to case so the Cases list shows "Analysing"
            analysis.status = "running"
            if case_row is not None:
                case_row.status = "running"
            await db.commit()

            # ── STAGE 1 — Image / Problem Analysis ───────────────────────────
            analysis.pipeline_stage = _S1_RUNNING
            await db.commit()

            _t_s1 = time.monotonic()
            _s1_fallback = False
            try:
                s1: Stage1Output = await _stage_1_analyse(upload_paths, text_context)
            except RuntimeError as _s1_exc:
                # RuntimeError = Ollama connection/OOM/server error.
                # These are NOT parse failures — retrying immediately will not help
                # and continuing with fake Stage1Output produces garbage downstream.
                # Fail fast with a clear message so the user knows to retry.
                log.error(
                    "Stage 1 runtime failure (OOM or Ollama error) — failing pipeline. Error: %s",
                    _s1_exc,
                )
                raise  # re-raise → pipeline catches it in the outer except and marks failed
            except ValueError as _s1_exc:
                # ValueError = all retries exhausted on parse/validation errors.
                # Only fall back to text context if the user provided one — that gives
                # Stages 2+3 something real to work from rather than a generic placeholder.
                if text_context and text_context.strip():
                    log.warning(
                        "Stage 1 parse failure — falling back to user text context. Error: %s",
                        _s1_exc,
                    )
                    s1 = Stage1Output(
                        problem_type=text_context.strip(),
                        trade_category="other",
                        severity="medium",
                        observations=[
                            "Image could not be automatically analysed — using your description instead.",
                            "For a better diagnosis, try uploading a clearer or wider photo.",
                        ],
                    )
                    _s1_fallback = True
                else:
                    log.error(
                        "Stage 1 failed and no text context provided — failing pipeline. Error: %s",
                        _s1_exc,
                    )
                    raise  # no text context to fall back to — fail the pipeline
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

            # ── Confidence gate ───────────────────────────────────────────────
            # If Stage 1 is not confident enough to diagnose the problem, stop
            # the pipeline and ask the user for a clearer photo or more context
            # instead of generating unreliable repair steps.
            # Threshold is 75 (not 50) because Qwen2.5-VL:3b tends to hallucinate
            # confidence values of 60-70 even for featureless/dark images.
            # 75 catches those while still allowing genuine 80+ readings through.
            _LOW_CONFIDENCE = 75
            if not _s1_fallback and (s1.needs_clarification or s1.confidence < _LOW_CONFIDENCE):
                log.info(
                    "Stage 1 confidence=%d needs_clarification=%s — stopping pipeline, "
                    "asking user for clarification.",
                    s1.confidence, s1.needs_clarification,
                )
                clarification_msg = (
                    s1.clarification_request.strip()
                    if s1.clarification_request.strip()
                    else (
                        "The photo is not clear enough to identify a specific home repair problem. "
                        "Please upload a closer, well-lit photo of the damaged area, "
                        "or describe the issue in the text field above."
                    )
                )
                analysis.status = "needs_clarification"
                analysis.pipeline_stage = "needs_clarification"
                analysis.validation_issues = clarification_msg
                if case_row is not None:
                    case_row.status = "needs_clarification"
                await db.commit()
                return  # stop — do not run Stage 2 or 3

            # Auto-title: overwrite the placeholder "Diagnosing…" with the AI result
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
            if case_row is not None:
                case_row.status = "complete"
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
                analysis.status            = "failed"
                analysis.validation_issues = f"{type(exc).__name__}: {exc}"
                if case_row is not None:
                    case_row.status = "failed"
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
        "- For the observations and visual_evidence, list only what you can concretely see.\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  problem_type        : string — a short description of the specific item that needs repair\n"
        "                        (e.g. 'Loose door handle', 'Leaking pipe joint', 'Cracked wall plaster',\n"
        "                        'Failed window seal', 'Draughty window frame')\n"
        "  trade_category      : string — EXACTLY one of these 10 values (use the exact string):\n"
        "                        plumbing, electrical, appliance, carpentry/joinery,\n"
        "                        glazing, sealing/weatherproofing, plastering/decorating,\n"
        "                        structural, hvac, general/other\n"
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
        "                          load-bearing walls, foundations, lintels, masonry cracks → structural\n"
        "                          boiler controls, central heating, air conditioning, ventilation → hvac\n"
        "                          anything else → general/other\n"
        "  severity            : string — exactly one of: low, medium, high\n"
        "  observations        : array of strings — 3 to 5 specific visual observations\n"
        "                        describing exactly what you see in the image\n"
        "  confidence          : integer 0-100 — your confidence that you can clearly identify\n"
        "                        a specific home repair problem from this image and any user text.\n"
        "                        STRICT RULES for confidence:\n"
        "                        90-100 = damage is crystal clear in a sharp, well-lit, close-up photo.\n"
        "                        76-89  = image is usable but not ideal (slightly blurry, far away, etc).\n"
        "                        0-75   = ANYTHING that is: mostly dark or black, mostly one solid colour,\n"
        "                                 random noise or static, a non-home-repair subject (food, person,\n"
        "                                 sky, text document, screenshot), too blurry to identify any object,\n"
        "                                 or where you are guessing rather than seeing clear damage.\n"
        "                        If you are not certain what is shown, you MUST use 0-75.\n"
        "  needs_clarification : boolean — true if confidence is 75 or below OR the image does not\n"
        "                        show a recognisable home repair problem; false otherwise.\n"
        "  clarification_request: string — if needs_clarification is true, write ONE specific sentence\n"
        "                        telling the user exactly what better photo or information would help\n"
        "                        (e.g. 'Please upload a closer, well-lit photo of the damaged pipe joint').\n"
        "                        If needs_clarification is false, use an empty string.\n"
        "  confidence_reason   : string — one to two sentences explaining exactly why you gave this\n"
        "                        confidence score. Be specific about what made the image clear or unclear.\n"
        "                        Example: 'Sharp close-up clearly shows a diagonal crack through painted\n"
        "                        plaster with visible dust below; damage extent and location are unambiguous.'\n"
        "  visual_evidence     : array of strings — list every distinct object or feature you can clearly\n"
        "                        see in the image, not just the damaged part. Be specific and factual.\n"
        "                        Example: ['White painted wall', 'Diagonal hairline crack approx 20cm',\n"
        "                        'Plaster dust on skirting board below crack', 'Painted skirting board']\n"
        "                        Do NOT include items you cannot actually see.\n"
        "  uncertainty_notes   : string — describe ONLY what you cannot confirm from the image alone\n"
        "                        (e.g. depth of crack, hidden damage behind surface, age of problem).\n"
        "                        If you are confident about everything relevant, use an empty string \"\".\n"
        "                        Example: 'Cannot determine whether crack penetrates full plaster depth\n"
        "                        or is surface-only; no sign of damp but cannot rule it out.'\n\n"
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
        "You are an experienced UK DIY advisor helping a homeowner prepare for a repair.\n"
        "Your job is to list exactly what they need before starting — tools, materials, and safety gear.\n\n"
        "RULES:\n"
        "1. Recommend SPECIFIC UK products and brands where relevant. Examples by trade:\n"
        "   Plasterwork/filling: Polyfilla Ready Mixed (small cracks up to 5mm), Toupret TX110 Fibre (cracks\n"
        "     5-20mm or on damp/movement-prone surfaces), Toupret Interior Filler (general patching),\n"
        "     Blue Grit bonding agent (before skim coat on smooth surfaces), Gyproc EasiFill (for large areas).\n"
        "   Adhesives/sealants: UniBond No More Nails (general bonding), Gorilla Grab Adhesive,\n"
        "     CT1 Sealant (superior waterproof multipurpose sealant for wet areas),\n"
        "     Dow Corning 785 or UniBond Silicone (bathroom/kitchen silicone), Everbuild Flash Band.\n"
        "   Paints/primers: Dulux or Johnstone's Trade matt emulsion (interior walls), Zinsser Bulls Eye 1-2-3\n"
        "     (stain-blocking primer), Johnstone's Exterior Gloss (exterior woodwork), Dulux Weathershield.\n"
        "   Plumbing: PTFE tape (standard 12mm thread seal), Fernox F1 or Sentinel X100 (heating system\n"
        "     inhibitor if relevant), push-fit fittings (e.g. John Guest or Speedfit), solvent weld fittings.\n"
        "   Wood/timber: Ronseal Exterior Wood Filler (two-part, for outdoor wood), Ronseal Interior Wood\n"
        "     Filler (ready mixed, for indoor), Ronseal Hardener + Filler (for soft/rotten timber),\n"
        "     Osmo Polyx Oil or Ronseal Hardwood Decking Oil (for hardwood), Sadolin Classic (softwood).\n"
        "   Fixings: Rawlplug or Fischer wall plugs (specify size: red=6mm, brown=7mm, yellow=5mm),\n"
        "     Resin anchor (Hilti HIT-RE 500 or Fischer FIS EV) for structural fixings.\n"
        "2. Size and spec every tool entry. Example: 'No.2 cross-head (Pozidriv) screwdriver — for the\n"
        "   M5 pan-head screws securing the bracket'. Never list a bare tool name.\n"
        "3. For materials, include the pack/tube size and mixing ratio where relevant.\n"
        "   Example: 'Polyfilla Ready Mixed (500g tub — apply with filling knife, no mixing required,\n"
        "   dries white in 2-4 hours depending on depth)'.\n"
        "4. Separate ESSENTIAL items (job cannot be done without them) from OPTIONAL (improve the result\n"
        "   or speed up the work but not strictly required).\n"
        "5. Include realistic estimated_time_minutes — this must account for drying/curing wait times\n"
        "   where relevant (e.g. a two-coat filler repair is not 30 min, it is 3-4 hours including drying).\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  tools                   : array of strings — ESSENTIAL tools, each with spec and purpose\n"
        "  optional_tools          : array of strings — OPTIONAL tools, each with spec and purpose\n"
        "  materials               : array of strings — ESSENTIAL consumable materials, each with brand,\n"
        "                            pack size, and brief usage note\n"
        "  optional_materials      : array of strings — OPTIONAL materials, each with brand and usage note\n"
        "  protective_gear         : array of strings — safety equipment with one-line reason\n"
        "  estimated_time_minutes  : integer — total realistic time including any drying/curing waits\n\n"
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
        "You are an experienced UK DIY advisor writing repair instructions for a homeowner.\n"
        "Write like a knowledgeable tradesperson giving practical, honest advice — not like a textbook.\n"
        "Be direct, specific, and realistic. UK context throughout.\n\n"
        "STEP WRITING RULES — read every rule before writing:\n"
        "1. ONE action per step. Never combine two actions (e.g. 'remove the screw AND clean the area')\n"
        "   into one step. Split them into two separate steps.\n"
        "2. Always begin with a clear PREP step (e.g. 'Lay dust sheets', 'Turn off the water supply',\n"
        "   'Turn off the circuit breaker') before any work step.\n"
        "3. Where a material must DRY or CURE, make that a separate step with a realistic wait time.\n"
        "   Example: 'Leave the first coat of Polyfilla to dry fully — this takes 2 to 4 hours\n"
        "   depending on depth and room temperature. The filler changes from pink to white when dry.'\n"
        "4. Reference the tools and materials from the provided list by their exact names.\n"
        "5. Each step description must say: (a) which tool or material, (b) the exact action,\n"
        "   (c) what done correctly looks or feels like, (d) one warning sign that something is wrong.\n"
        "6. For plaster/filler work: always mention feathering the edges, sanding between coats,\n"
        "   and applying mist coat before painting. Mention primer where needed.\n"
        "7. For any wet-area work (bathroom, kitchen): mention CT1 or silicone over standard filler,\n"
        "   and allow 24 hours cure before exposing to water.\n"
        "8. Use realistic language: 'press firmly', 'sand with light circular strokes', 'the surface\n"
        "   should feel smooth under your fingertip', 'no ridges or proud edges'. Avoid 'ensure' and\n"
        "   'make sure' — use direct imperatives instead ('press', 'sand', 'wipe').\n"
        "9. Step titles must be short action phrases: 'Lay dust sheets', 'Apply first filler coat',\n"
        "   'Sand the dried filler'. Never use step titles longer than 8 words.\n"
        "10. Include 6 to 10 steps for typical repairs. Do not pad with unnecessary steps.\n\n"
        "SAFETY NOTES RULES:\n"
        "safety_notes must be an array of plain strings. Each string is one complete sentence.\n"
        "Do NOT return objects or dicts inside safety_notes. CORRECT: [\"Wear safety glasses when sanding.\"]\n"
        "WRONG: [{\"note\": \"Wear safety glasses when sanding.\"}]\n"
        "Each note must be specific to the trade category and this problem — no generic platitudes.\n\n"
        "RISK ASSESSMENT:\n"
        "Set diy_recommended to false if the repair involves:\n"
        "  - mains electrical wiring (230V), consumer units, or fuse boxes\n"
        "  - gas pipes, boilers, or gas appliances (Gas Safe registration required by law)\n"
        "  - structural load-bearing walls, beams, or foundations\n"
        "  - significant water damage with risk of electrical exposure\n"
        "  - asbestos (pre-2000 buildings, textured coatings, floor tiles, pipe lagging)\n"
        "If diy_recommended is false, set professional_type to the exact trade: 'Gas Safe engineer',\n"
        "'NICEIC-registered electrician', 'structural engineer', 'CIPHE-registered plumber', etc.\n"
        "If diy_recommended is true, set professional_type to null.\n\n"
        "You must return a single JSON object with exactly these fields:\n"
        "  steps                      : array of step objects, each containing:\n"
        "    step_number              : integer starting at 1\n"
        "    title                    : short action phrase, max 8 words\n"
        "    description              : practical instruction paragraph following the rules above\n"
        "    warning                  : specific safety or quality warning string, or null\n"
        "  safety_notes               : array of 3 to 5 plain strings — NOT objects, plain strings only\n"
        "  difficulty_level           : exactly one of: beginner, intermediate, advanced\n"
        "  when_to_call_professional  : string — ALWAYS required, never null. If diy_recommended is true,\n"
        "                              write the specific warning signs that mean the user should stop\n"
        "                              and call a professional. If diy_recommended is false, explain why.\n"
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
        },
    )
