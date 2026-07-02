"""
Video analysis service — Phase 4B implementation.

Architecture
─────────────────────────────────────────────────────────────────────────────
VideoInferenceClient (abstract)
├── LocalVideoInferenceClient   — HuggingFace transformers, Qwen2.5-VL locally
└── CloudVideoInferenceClient   — Phase 5: vLLM / remote endpoint

analyse_video()  is the single entry point called by _run_video_pipeline()
(analysis_service.py).  It returns a VideoStage1Output, which IS a Stage1Output,
so _run_pipeline() routes, escalates, and generates Stage 2/3 with no changes.

Model singleton
─────────────────────────────────────────────────────────────────────────────
The local model is loaded ONCE per process, on the first video inference call.
An asyncio.Lock prevents double-loading under concurrent requests.
Subsequent calls reuse the in-memory model (~30s warm vs ~1012s cold).

Windows DLL note
─────────────────────────────────────────────────────────────────────────────
On Windows, torch must be imported BEFORE decord or c10.dll initialisation
fails.  All sync helpers import torch as their very first statement.

Video reader
─────────────────────────────────────────────────────────────────────────────
Frame extraction uses PyAV (av package) via _extract_frames_pyav().
decord crashes on Windows with forrtl error 200 (Intel MKL abort).
torchvision.io.read_video was removed in torchvision >= 0.20.
PyAV wraps libavcodec with no Fortran dependency and works on all platforms.

Error handling
─────────────────────────────────────────────────────────────────────────────
All failure paths raise VideoAnalysisError with a human-readable message.
_run_video_pipeline() catches this and marks analysis.status = "failed".

Config (read from backend/.env)
─────────────────────────────────────────────────────────────────────────────
  VIDEO_ANALYSIS_ENABLED    true | false  (default: false)
  VIDEO_INFERENCE_MODE      local | cloud (default: local)
  VIDEO_MODEL_LOCAL         HF model ID for local inference
  VIDEO_MODEL_CLOUD         HF model ID for cloud inference
  VIDEO_INFERENCE_ENDPOINT  URL of the remote vLLM endpoint (cloud mode)
  VIDEO_INFERENCE_API_KEY   Bearer token for the remote endpoint (cloud mode)
  VIDEO_NFRAMES             Frames sampled per video (default: 12)
  VIDEO_MAX_NEW_TOKENS      Token budget for model response (default: 2048)
  VIDEO_USE_4BIT            4-bit NF4 quantisation (default: true)
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.schemas.analysis import VideoStage1Output

log = logging.getLogger(__name__)

# ── Environment configuration ──────────────────────────────────────────────────
VIDEO_ANALYSIS_ENABLED   = os.getenv("VIDEO_ANALYSIS_ENABLED",   "false").lower() == "true"
VIDEO_INFERENCE_MODE     = os.getenv("VIDEO_INFERENCE_MODE",     "local")
VIDEO_MODEL_LOCAL        = os.getenv("VIDEO_MODEL_LOCAL",        "Qwen/Qwen2.5-VL-7B-Instruct")
VIDEO_MODEL_CLOUD        = os.getenv("VIDEO_MODEL_CLOUD",        "Qwen/Qwen2.5-VL-72B-Instruct")
VIDEO_INFERENCE_ENDPOINT = os.getenv("VIDEO_INFERENCE_ENDPOINT", "")
VIDEO_INFERENCE_API_KEY  = os.getenv("VIDEO_INFERENCE_API_KEY",  "")
VIDEO_NFRAMES            = int(os.getenv("VIDEO_NFRAMES",        "12"))
VIDEO_MAX_NEW_TOKENS     = int(os.getenv("VIDEO_MAX_NEW_TOKENS", "2048"))
VIDEO_USE_4BIT           = os.getenv("VIDEO_USE_4BIT",           "true").lower() == "true"


# ── Exceptions ─────────────────────────────────────────────────────────────────

class VideoAnalysisError(Exception):
    """Raised when video analysis cannot be completed.

    Caught by _run_video_pipeline() and converted to a user-visible failure
    message stored on the Analysis row — same pattern as other pipeline errors.
    """


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert home repair diagnostician working for Knitra, a UK household \
repair guidance application. You will be shown a video of a domestic repair \
issue filmed by the homeowner.

Your job is to analyse the video and produce a structured diagnostic assessment \
that will be used to route the homeowner to appropriate repair guidance or \
professional help.

IMPORTANT rules:
1. You MUST output ONLY a valid JSON object. No prose, no markdown, no backticks.
2. Be HONEST about uncertainty. If you cannot confirm something, say so explicitly.
3. Pay close attention to TEMPORAL information — what changes, what flickers, \
what is intermittent. This is why video evidence is being collected.
4. Never overstate confidence. A confident wrong answer is worse than an honest \
'I cannot confirm this without a closer look'.
5. Safety is always the primary concern. If in doubt, recommend professional \
assessment.
6. Set professional_assessment_recommended to true only when the evidence \
supports it: risk_level is high or critical, OR a specific hazard is visible \
that a competent DIYer cannot safely manage — live electrical exposure or water \
within 1 metre of a live fitting, a gas supply or boiler fault, an active water \
leak under pressure (burst pipe, leaking water heater), sewage contamination, \
active flooding, or structural instability (sagging ceiling, visible load-bearing \
damage). Do NOT set it to true purely out of general caution for medium-risk \
maintenance where no such hazard is observable. A blocked drain, dripping tap, \
peeling paint or wallpaper without damp evidence, or grout and tile repair are \
tasks a competent homeowner can assess safely — route them through the risk and \
hazard evidence, not through precautionary instinct.
"""

# Double braces {{ }} are literal braces in the JSON schema (format() escaping).
# {user_note} is the only real format variable.
_USER_PROMPT_TEMPLATE = """\
Analyse this household repair video carefully.

Return ONLY a JSON object with EXACTLY these fields. No extra text, no markdown.

{{
  "scope": "<repair | emergency | not_repair | unclear>",
  "problem_type": "<short human-readable description, e.g. 'Leaking pipe joint under kitchen sink'>",
  "trade_category": "<EXACTLY one of: plumbing | electrical | appliance | carpentry/joinery | glazing | sealing/weatherproofing | plastering/decorating | stone/masonry | structural | hvac | general/other — use stone/masonry for tile, grout, and stone surface work; use plumbing only for water supply, drainage, and pipe fittings>",
  "severity": "<low | medium | high>",
  "risk_level": "<low | medium | high | critical>",
  "confidence": <integer 0 to 100>,
  "confidence_reason": "<1-2 sentences explaining this confidence level honestly>",
  "observations": ["<observation 1>", "<observation 2>"],
  "visual_evidence": ["<object or feature 1>", "<object or feature 2>"],
  "what_can_be_confirmed": ["<confirmed finding 1>"],
  "what_cannot_be_confirmed": ["<unknown 1>"],
  "missing_information": ["<specific evidence that would improve the diagnosis>"],
  "uncertainty_notes": "<overall summary of limitations>",
  "needs_clarification": <true | false>,
  "clarification_request": "<user-facing message if needs_clarification is true; empty string if false>",
  "hidden_damage_risk": "<low | medium | high>",
  "diy_suitability": "<suitable | conditional | not_recommended>",
  "professional_assessment_recommended": <true | false>,
  "safety_concerns": ["<specific hazard 1>"],
  "safety_precautions": ["<protective action 1>"],
  "clarifying_questions": ["<required question 1>"],
  "optional_questions": ["<helpful non-blocking question>"],
  "motion_detected": <true | false>,
  "brief_hazard_observed": <true | false>,
  "brief_hazards": ["<transient hazard description with timing>"],
  "key_moments": ["<approx 0:05 — description>"],
  "temporal_confidence": <float 0 to 100>,
  "video_quality_notes": "<limitations due to video quality; empty string if quality was adequate>",
  "recommended_next_action": "<single most important action for the homeowner>",
  "input_modality": "video"
}}{user_note}
"""


def _build_user_note_suffix(description: str) -> str:
    if not description or not description.strip():
        return ""
    return f'\n\nAdditional context from the homeowner: "{description.strip()}"'


# ── Model singleton ────────────────────────────────────────────────────────────
# Loaded once per process on the first video inference call.
# asyncio.Lock prevents double-loading if two video requests arrive simultaneously.

_model = None
_processor = None
_model_name_loaded: Optional[str] = None
_model_lock = asyncio.Lock()


# ── Sync helpers (all run via asyncio.to_thread) ──────────────────────────────

def _load_model_and_processor_sync(model_name: str, use_4bit: bool) -> tuple:
    """Load Qwen2.5-VL model + processor. Blocking — call via asyncio.to_thread.

    torch is imported FIRST to avoid the Windows c10.dll/decord DLL conflict.
    bitsandbytes 4-bit NF4 reduces VRAM from ~15 GB (7B fp16) to ~3 GB.
    """
    # Windows DLL ordering: torch must be initialised before decord loads.
    import torch  # noqa: PLC0415 — intentional late import for DLL ordering

    from transformers import (  # noqa: PLC0415
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    log.info("video_model_loading model=%s use_4bit=%s", model_name, use_4bit)
    t0 = time.monotonic()

    model_kwargs: dict = {}

    if use_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
            log.info("video_model_loading 4-bit NF4 config applied")
        except Exception as exc:
            log.warning("video_model_loading 4-bit config failed (%s), falling back to fp16", exc)
            use_4bit = False

    if not use_4bit:
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
    except Exception as exc:
        raise VideoAnalysisError(
            f"Failed to load video model '{model_name}': {type(exc).__name__}: {exc}"
        ) from exc

    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as exc:
        raise VideoAnalysisError(
            f"Failed to load video processor for '{model_name}': {type(exc).__name__}: {exc}"
        ) from exc

    load_ms = int((time.monotonic() - t0) * 1000)

    # Log VRAM usage after load for diagnostics
    if torch.cuda.is_available():
        vram_used  = torch.cuda.memory_allocated(0) / 1024 ** 3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        log.info(
            "video_model_loaded model=%s load_ms=%d vram_used=%.2fGB vram_total=%.1fGB",
            model_name, load_ms, vram_used, vram_total,
        )
    else:
        log.info("video_model_loaded model=%s load_ms=%d device=cpu", model_name, load_ms)

    return model, processor


def _extract_frames_pyav(video_path: Path, nframes: int) -> list:
    """Extract nframes evenly-spaced frames from a video as PIL Images using PyAV.

    Avoids both decord (forrtl crash on Windows) and torchvision.io.read_video
    (removed in torchvision >= 0.20). PyAV wraps libavcodec directly with no
    Fortran/MKL dependency.
    """
    import av
    import numpy as np

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        # Estimate total frames from container metadata.
        # stream.frames is 0 for many MP4s — fall back to duration × fps.
        total: int = stream.frames or 0
        if total <= 0:
            fps = float(stream.average_rate or stream.base_rate or 30)
            if container.duration:
                total = max(1, int(container.duration / 1_000_000 * fps))
            else:
                total = 300  # safe fallback for unknown duration

        # Frame indices to extract (0-based, evenly spaced)
        if nframes >= total:
            target = set(range(total))
        else:
            target = set(int(i) for i in np.linspace(0, total - 1, nframes, dtype=int))

        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in target:
                frames.append(frame.to_image())
            if len(frames) >= nframes:
                break

    if not frames:
        raise VideoAnalysisError(
            f"PyAV decoded 0 frames from '{video_path.name}'"
        )

    return frames


def _run_inference_sync(
    model,
    processor,
    video_path: Path,
    prompt_text: str,
    nframes: int,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Run Qwen2.5-VL on a video and return (raw_output, nframes_used).

    Blocking — call via asyncio.to_thread.
    torch must already be imported (guaranteed by _load_model_and_processor_sync).
    """
    import torch  # noqa: PLC0415 — re-import is a no-op; confirms DLL order
    from qwen_vl_utils import process_vision_info  # noqa: PLC0415

    # Pre-extract frames with PyAV and pass as a list of PIL Images.
    # qwen-vl-utils fetch_video() accepts list|tuple directly (calls fetch_image
    # per frame), bypassing all video-reader backends entirely.
    # This avoids: decord forrtl crash on Windows, torchvision.io.read_video
    # removal in torchvision >= 0.20.
    try:
        frames = _extract_frames_pyav(video_path, nframes)
    except VideoAnalysisError:
        raise
    except Exception as exc:
        raise VideoAnalysisError(
            f"Frame extraction failed for '{video_path.name}': {type(exc).__name__}: {exc}"
        ) from exc

    # Pass frames as "type": "video" with a PIL list.
    # fetch_video() takes the else branch for list input (no video reader called),
    # so FORCE_QWENVL_VIDEO_READER=decord guards the string-path case only.
    # This is important: video mode applies a lower per-frame pixel budget
    # (VIDEO_FRAME_MAX_PIXELS) than image mode (IMAGE_MAX_PIXELS), which
    # prevents the CUDA OOM that occurs when 8 large images are passed separately.
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "sample_fps": 1.0,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception as exc:
        raise VideoAnalysisError(
            f"process_vision_info failed for '{video_path.name}': {type(exc).__name__}: {exc}"
        ) from exc

    actual_nframes = len(frames)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to the model's first device (may differ with device_map="auto")
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    inputs = inputs.to(input_device)
    n_input_tokens = inputs["input_ids"].shape[1]

    log.info(
        "video_inference_start nframes=%d input_tokens=%d max_new_tokens=%d",
        actual_nframes, n_input_tokens, max_new_tokens,
    )

    t0 = time.monotonic()
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy decoding — deterministic output
            )
    except Exception as exc:
        # Free GPU memory before re-raising so a retry has a better chance.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg or "accelerator" in type(exc).__name__.lower():
            raise VideoAnalysisError(
                f"GPU out of memory during video inference. "
                f"Try reducing VIDEO_NFRAMES (currently {nframes}) or "
                f"enabling VIDEO_USE_4BIT=true. Error: {exc}"
            ) from exc
        raise VideoAnalysisError(
            f"Inference runtime error: {type(exc).__name__}: {exc}"
        ) from exc

    infer_ms = int((time.monotonic() - t0) * 1000)

    # Decode only the newly generated tokens (strip the input prefix)
    generated_only = [out[n_input_tokens:] for out in generated_ids]
    raw_output = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        torch.cuda.empty_cache()   # free allocator fragments for the next request
        log.info(
            "video_inference_complete infer_ms=%d output_chars=%d peak_vram=%.2fGB",
            infer_ms, len(raw_output), vram_peak,
        )
    else:
        log.info(
            "video_inference_complete infer_ms=%d output_chars=%d",
            infer_ms, len(raw_output),
        )

    return raw_output, actual_nframes


def _extract_json_from_raw(raw: str) -> dict:
    """Extract and parse the first JSON object from the model's raw output.

    Handles common model quirks: markdown fences, leading/trailing prose.
    Raises VideoAnalysisError on failure so the caller can mark analysis failed.
    """
    s = raw.strip()

    # Strip markdown fences
    if s.startswith("```"):
        lines = s.splitlines()
        end_fence = next(
            (i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "```"),
            None,
        )
        s = "\n".join(lines[1:end_fence]) if end_fence else "\n".join(lines[1:])

    start = s.find("{")
    end   = s.rfind("}") + 1
    if start == -1 or end <= start:
        raise VideoAnalysisError(
            f"Model output contains no JSON object. "
            f"First 200 chars: {raw[:200]!r}"
        )

    json_str = s[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise VideoAnalysisError(
            f"JSON parse error in model output: {exc}. "
            f"First 200 chars of extracted JSON: {json_str[:200]!r}"
        ) from exc


# ── Async model accessor ───────────────────────────────────────────────────────

async def _get_local_model(model_name: str) -> tuple:
    """Return the cached (model, processor) pair, loading on first call.

    The asyncio.Lock ensures only one load runs concurrently even if multiple
    video requests arrive before the first load completes.
    """
    global _model, _processor, _model_name_loaded

    async with _model_lock:
        if _model is None or _model_name_loaded != model_name:
            log.info("video_model_singleton_load starting model=%s", model_name)
            try:
                _model, _processor = await asyncio.to_thread(
                    _load_model_and_processor_sync, model_name, VIDEO_USE_4BIT
                )
                _model_name_loaded = model_name
                log.info("video_model_singleton_load complete model=%s", model_name)
            except VideoAnalysisError:
                raise
            except Exception as exc:
                raise VideoAnalysisError(
                    f"Model load failed for '{model_name}': {type(exc).__name__}: {exc}"
                ) from exc

    return _model, _processor


# ── Inference clients ──────────────────────────────────────────────────────────

class VideoInferenceClient(ABC):
    """Abstract base for all video inference backends."""

    @abstractmethod
    async def infer(self, video_path: Path, prompt: str) -> tuple[str, int]:
        """Run inference on *video_path*.

        Returns (raw_model_output, nframes_used).
        Raises VideoAnalysisError on any failure.
        """


class LocalVideoInferenceClient(VideoInferenceClient):
    """Phase 4B: Qwen2.5-VL via HuggingFace transformers, running locally."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def infer(self, video_path: Path, prompt: str) -> tuple[str, int]:
        model, processor = await _get_local_model(self.model_name)

        prompt_text = _USER_PROMPT_TEMPLATE.format(
            user_note=_build_user_note_suffix(prompt)
        )

        try:
            raw, nframes_used = await asyncio.to_thread(
                _run_inference_sync,
                model,
                processor,
                video_path,
                prompt_text,
                VIDEO_NFRAMES,
                VIDEO_MAX_NEW_TOKENS,
            )
        except VideoAnalysisError:
            raise
        except Exception as exc:
            raise VideoAnalysisError(
                f"Unexpected error during local video inference: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return raw, nframes_used


class CloudVideoInferenceClient(VideoInferenceClient):
    """Phase 5: remote vLLM endpoint running Qwen2.5-VL-72B.

    Not implemented yet. Raises VideoAnalysisError until Phase 5 is ready.
    """

    def __init__(self, model_name: str, endpoint: str, api_key: str) -> None:
        self.model_name = model_name
        self.endpoint   = endpoint
        self.api_key    = api_key

    async def infer(self, video_path: Path, prompt: str) -> tuple[str, int]:
        raise VideoAnalysisError(
            f"Cloud video model '{self.model_name}' is not configured yet. "
            f"Endpoint: '{self.endpoint or '(not set)'}'. "
            "Phase 5 (cloud inference) has not been implemented. "
            "Set VIDEO_INFERENCE_MODE=local to use the local model instead."
        )


# ── Client factory ─────────────────────────────────────────────────────────────

def _get_client() -> VideoInferenceClient:
    if VIDEO_INFERENCE_MODE == "cloud":
        return CloudVideoInferenceClient(
            model_name=VIDEO_MODEL_CLOUD,
            endpoint=VIDEO_INFERENCE_ENDPOINT,
            api_key=VIDEO_INFERENCE_API_KEY,
        )
    return LocalVideoInferenceClient(model_name=VIDEO_MODEL_LOCAL)


# ── Video-specific safety adjustments ──────────────────────────────────────────

# Generic mechanical cautions present in virtually any hands-on repair.
# Matched case-insensitively as substrings of individual brief_hazards entries.
# If ALL entries in brief_hazards match one of these phrases, they are stripped
# by Rule 0 before any escalation logic runs.
_BOILERPLATE_HAZARD_PHRASES: frozenset[str] = frozenset({
    "sharp edge", "sharp edges",
    "sharp object", "sharp objects",   # generic kitchen/bathroom debris caution
    "if left unattended",              # consequence-of-inaction, not an observed hazard
    "if improper tool", "if improper tools",  # technique advice, not observed in video
    "moving part", "moving parts",
    "pinch risk", "pinch hazard",
    "hand tool", "tool injury",
    "small cut", "minor cut",
    "splinter",
    "wear gloves", "safety gloves", "gloves recommended", "use gloves",
    "eye protection", "safety glasses", "wear goggles",
    "minor injury", "potential injury",
    "exercise caution", "general caution", "standard caution",
    "be careful", "take care",
})

# Real hazard keywords — if any appear anywhere in the model's combined textual
# output, the boilerplate filter is bypassed and all escalation is preserved.
_SERIOUS_HAZARD_KEYWORDS: frozenset[str] = frozenset({
    # Electrical
    "electric", "live wire", "live cable",
    "outlet", "socket", "voltage", "circuit", "fuse", "breaker",
    "shock", "electrocution", "spark", "arcing",
    # Fire / heat
    "fire", "flame", "flammable", "smoke", "smoulder", "smoldering",
    "burn", "burning",
    # Gas / chemical
    "gas leak", "gas pipe", " gas ", "carbon monoxide", "fumes",
    "asbestos", "toxic", "chemical hazard",
    # Water / structural
    "flooding", "active leak", "sewage", "mould", "mold",
    "structural", "load bearing", "load-bearing",
    "collaps", "subsid", "ceiling collapse",
    # Height / weight
    "fall from", "scaffold", "working at height",
    "crush", "heavy object",
})

# Trade categories where diy_suitability "conditional" may safely be upgraded
# to "suitable" after all brief_hazards prove to be boilerplate.
_LOW_RISK_TRADES: frozenset[str] = frozenset({
    "carpentry/joinery", "carpentry", "joinery",
    "general/other", "general", "other",
    "cleaning", "cosmetic", "cleaning/cosmetic",
    "painting", "decorating", "painting/decorating",
    "tiling", "flooring",
    "furniture", "household maintenance",
})

# Keywords that distinguish hazardous motion (active leak, sparks, smoke, structural
# movement) from normal repair activity (person using a hand tool, adjusting a hinge).
# Used by _motion_indicates_hazard() — only escalate when these appear in the model's
# own descriptions, not just because something is moving in the frame.
_HAZARD_MOTION_KEYWORDS: frozenset[str] = frozenset({
    # Water / liquid
    "leak", "leaking", "leaks", "flood", "flooding", "seep", "seeping",
    "drip", "dripping", "pooling", "spreading water", "running water",
    # Fire / heat
    "smoke", "smoking", "smoulder", "smouldering", "fire", "flame", "burning",
    # Electrical
    "spark", "sparks", "sparking", "flicker", "flickering", "arcing",
    # Structural movement
    "collaps", "unstable", "crumbling", "falling", "subsiding", "shifting",
    "crack widening", "widening crack",
    # Active worsening
    "worsening", "spreading damage", "active damage", "damage progressing",
    "rapidly",
})


def _motion_indicates_hazard(s1: VideoStage1Output) -> bool:
    """Return True only when motion signals an active fault, not normal repair activity.

    Checks the model's own textual output (observations, problem_type, brief_hazards,
    key_moments, uncertainty_notes) for vocabulary that suggests a live, changing, or
    worsening condition — as opposed to a person performing a routine repair action.
    """
    parts: list[str] = []
    if s1.problem_type:
        parts.append(s1.problem_type)
    parts.extend(s1.observations or [])
    parts.extend(s1.brief_hazards or [])
    parts.extend(s1.key_moments or [])
    if s1.uncertainty_notes:
        parts.append(s1.uncertainty_notes)
    combined = " ".join(parts).casefold()
    return any(kw in combined for kw in _HAZARD_MOTION_KEYWORDS)


def _apply_video_safety_adjustments(s1: VideoStage1Output) -> list[str]:
    """Apply conservative safety adjustments to VideoStage1Output before routing.

    These complement _apply_safety_net() (trade-based escalation for images).
    Video-specific signals — motion, transient hazards, temporal confidence,
    quality — may require escalation that the base image routing does not cover.

    Mutates s1 in place. Returns human-readable notes (logged for transparency).
    Never lowers any signal; only raises them.
    """
    notes: list[str] = []

    # -1. Scope consistency fix
    #
    # The video model occasionally contradicts itself: it identifies a specific
    # repair problem (non-empty problem_type + a valid repair trade_category) but
    # simultaneously sets scope="not_repair" or scope="unclear".  The _determine_
    # output_mode scope gate (highest priority) would short-circuit to out_of_scope
    # before any of the downstream rules or routing logic can run.
    #
    # We override scope to "repair" only when:
    #   - scope is "not_repair" or "unclear" (NOT "emergency" — active hazards stay)
    #   - problem_type is non-empty (the model did identify something)
    #   - trade_category is non-empty (the model assigned a repair trade)
    #
    # This covers the case where the model says "Cabinet door hinge malfunction"
    # (trade=general/other) yet also outputs scope="not_repair" — the repair
    # identification is concrete enough to override the contradictory scope label.
    # Genuine non-repair videos typically produce empty or very vague problem_type
    # with no trade category, so they are not affected.
    _scope = s1.scope or "repair"
    if _scope in ("not_repair", "unclear") and s1.problem_type and s1.trade_category:
        s1.scope = "repair"
        notes.append(
            f"scope_fix: scope {_scope!r} -> 'repair' "
            f"(problem_type={s1.problem_type!r}, trade={s1.trade_category!r})"
        )

    # If the model outputs scope="emergency" but the content doesn't describe an
    # actual emergency (fire, flooding, collapse, gas, smoke), it's a 3B model
    # error — the same contradiction fix applied to not_repair/unclear above.
    # Real emergencies contain at least one of these keywords in problem_type or
    # observations; a hinge or similar non-hazardous repair will not.
    _EMERGENCY_KEYWORDS = {
        "fire", "flame", "smoke", "flood", "flooding", "collapse",
        "gas leak", "gas", "explosion", "electrocution", "electric shock",
        "sparks", "arcing", "carbon monoxide",
    }
    if _scope == "emergency" and s1.problem_type and s1.trade_category:
        _content = f"{s1.problem_type} {' '.join(getattr(s1, 'observations', None) or [])}".lower()
        if not any(kw in _content for kw in _EMERGENCY_KEYWORDS):
            s1.scope = "repair"
            notes.append(
                f"scope_fix: scope 'emergency' -> 'repair' "
                f"(no emergency keywords found in problem_type/observations; "
                f"problem_type={s1.problem_type!r})"
            )

    # -0. Decorative / aesthetic task → not_repair
    #
    # The video model sometimes classifies purely decorative activities (mural
    # painting, artistic wall decoration) as scope="repair" with
    # trade_category="plastering/decorating".  These are not repair jobs and
    # should route out_of_scope so Stage 2/3 never run.
    #
    # Fires ONLY when:
    #   - A decorative-signal word appears in problem_type or observations
    #   - No damage/repair-signal word is present anywhere in the same text
    #   - trade_category is plastering/decorating or general/other
    #     (trades that legitimately overlap with decoration)
    #
    # Painting IS still repair when damage signals are present: peeling paint,
    # damp stain, mould, cracked plaster, water damage, flaking, blistering, rot.
    _DECORATIVE_SIGNALS: frozenset[str] = frozenset({
        "mural", "wall art", "artwork", "wall artwork", "painting mural",
        "decorative painting", "aesthetic paint", "decorating",
    })
    _DECORATION_DAMAGE_SIGNALS: frozenset[str] = frozenset({
        "peel", "damp", "mould", "mold", "crack", "water damage", "water-damage",
        "stain", "flak", "blister", "rot", "repair", "damage", "remediat",
        "patch", "fill", "plaster crack", "plaster repair",
    })
    _pt_lower  = (s1.problem_type or "").lower()
    _obs_lower = " ".join(s1.observations or []).lower()
    _decor_combined = _pt_lower + " " + _obs_lower
    _has_decorative = any(sig in _decor_combined for sig in _DECORATIVE_SIGNALS)
    _has_damage     = any(sig in _decor_combined for sig in _DECORATION_DAMAGE_SIGNALS)
    _decor_trade    = (s1.trade_category or "").lower() in (
        "plastering/decorating", "general/other", ""
    )
    if _has_decorative and not _has_damage and _decor_trade:
        s1.scope = "not_repair"
        notes.append(
            f"decorative_task: scope -> 'not_repair' "
            f"(problem_type={s1.problem_type!r}, no damage signals detected)"
        )

    # 0. Boilerplate generic hazard filter
    #
    # The video model frequently marks standard mechanical cautions as "hazards"
    # (e.g. "sharp edges", "moving parts", "pinch risk").  These phrases are
    # true of virtually any hands-on repair and are PPE/common-sense reminders,
    # not signals that a professional is required.
    #
    # This rule strips those generic entries from brief_hazards BEFORE Rule 1
    # evaluates brief_hazard_observed, so that only real hazards (electrical,
    # fire, gas, structural collapse, flooding, etc.) trigger escalation.
    #
    # Safety guarantee: if ANY keyword from _SERIOUS_HAZARD_KEYWORDS appears
    # anywhere in the model's own text output (problem_type, observations,
    # brief_hazards, safety_concerns, key_moments, uncertainty_notes), the
    # filter is entirely skipped and every escalation flag is left as-is.
    _combined_text = " ".join(filter(None, [
        s1.problem_type or "",
        " ".join(s1.observations or []),
        " ".join(s1.brief_hazards or []),
        " ".join(s1.safety_concerns or []),
        " ".join(s1.key_moments or []),
        s1.uncertainty_notes or "",
    ])).casefold()
    _has_serious_hazard = any(kw in _combined_text for kw in _SERIOUS_HAZARD_KEYWORDS)

    if not _has_serious_hazard and (
        s1.brief_hazards or s1.brief_hazard_observed or s1.safety_concerns
    ):
        # Partition brief_hazards into real vs. boilerplate
        _real_hazards: list[str] = []
        _stripped_hazards: list[str] = []
        for h in (s1.brief_hazards or []):
            h_lower = h.casefold()
            if any(phrase in h_lower for phrase in _BOILERPLATE_HAZARD_PHRASES):
                _stripped_hazards.append(h)
            else:
                _real_hazards.append(h)

        # Partition safety_concerns into real vs. boilerplate.
        # The video model often mirrors the same boilerplate text into both
        # brief_hazards and safety_concerns — both must be cleaned.
        _real_concerns: list[str] = []
        _stripped_concerns: list[str] = []
        for sc in (s1.safety_concerns or []):
            sc_lower = sc.casefold()
            if any(phrase in sc_lower for phrase in _BOILERPLATE_HAZARD_PHRASES):
                _stripped_concerns.append(sc)
            else:
                _real_concerns.append(sc)

        _any_stripped = bool(_stripped_hazards or _stripped_concerns)
        _no_real_hazards = not _real_hazards and not _real_concerns

        if _any_stripped:
            if _stripped_hazards:
                s1.brief_hazards = _real_hazards
                notes.append(
                    f"boilerplate_filter: stripped {len(_stripped_hazards)} generic "
                    f"brief_hazard(s): " + "; ".join(repr(h) for h in _stripped_hazards)
                )
            if _stripped_concerns:
                s1.safety_concerns = _real_concerns
                notes.append(
                    f"boilerplate_filter: stripped {len(_stripped_concerns)} generic "
                    f"safety_concern(s): " + "; ".join(repr(sc) for sc in _stripped_concerns)
                )

            if _no_real_hazards:
                s1.brief_hazard_observed = False
                notes.append("boilerplate_filter: brief_hazard_observed -> False")

                # Clear professional_assessment_recommended — no real hazard
                # remains to justify it.
                if s1.professional_assessment_recommended:
                    s1.professional_assessment_recommended = False
                    notes.append(
                        "boilerplate_filter: professional_assessment_recommended -> False "
                        "(no real hazards or safety_concerns remain)"
                    )

                # Upgrade diy_suitability from "conditional" to "suitable" for
                # low-risk trades.  This unblocks Rule 5's demotion logic below.
                trade = (s1.trade_category or "").casefold()
                _is_low_risk_trade = any(t in trade for t in _LOW_RISK_TRADES)
                if (
                    (s1.diy_suitability or "conditional") == "conditional"
                    and (s1.confidence or 0) >= 65
                    and _is_low_risk_trade
                ):
                    s1.diy_suitability = "suitable"
                    notes.append(
                        f"boilerplate_filter: diy_suitability conditional->suitable "
                        f"(trade={s1.trade_category!r}, conf={s1.confidence}, "
                        f"no real hazards)"
                    )

    # 1. Transient hazard observed — force professional assessment
    if s1.brief_hazard_observed or s1.brief_hazards:
        if not s1.professional_assessment_recommended:
            s1.professional_assessment_recommended = True
            notes.append("brief_hazard_observed -> professional_assessment_recommended=True")
        # If safety_concerns is empty, populate from brief_hazards so the
        # assessment-first consistency guard (_decide_actions) fires correctly.
        if not s1.safety_concerns and s1.brief_hazards:
            s1.safety_concerns = [
                f"Transient hazard observed in video: {h}"
                for h in s1.brief_hazards[:2]
            ]
            notes.append(
                f"added {len(s1.safety_concerns)} safety_concern(s) from brief_hazards"
            )

    # 1R. Medium-risk plumbing / plastering-decorating with no serious hazard evidence
    #
    # The video model over-sets professional_assessment_recommended in these two
    # trades because the system prompt says "if in doubt, recommend professional
    # assessment."  This triggers for medium-risk maintenance where no unmanageable
    # hazard is actually present: drain blockages, dripping taps, peeling wallpaper.
    #
    # This rule is evidence-based, NOT problem-name-based.  It fires only when ALL
    # of the following hold simultaneously:
    #
    #   1. trade_category is plumbing or plastering/decorating
    #      (the two trades where systematic over-escalation is observed)
    #   2. risk_level == "medium"
    #      (high/critical always warrant professional escalation — untouched)
    #   3. _has_serious_hazard is False
    #      (computed above from the model's own text; covers electrical, gas,
    #      flooding, sewage, structural collapse, fire, mould, asbestos, etc.)
    #   4. No real brief_hazards remain after Rule 0 boilerplate stripping
    #      (a non-boilerplate specific hazard in the model's output keeps par=True)
    #
    # The routing gate in analysis_service.py (_force_assessment_first) is the
    # PRIMARY fix for stored analyses; this rule corrects the stored par signal
    # so future routing and display are consistent with the evidence.
    _ROUTINE_ESCALATION_TRADES: frozenset[str] = frozenset({
        "plumbing",
        "plastering/decorating",
    })
    if (
        (s1.trade_category or "").lower() in _ROUTINE_ESCALATION_TRADES
        and (s1.risk_level or "medium") == "medium"
        and not _has_serious_hazard
        and not s1.brief_hazards         # real entries remaining after Rule 0
        and s1.professional_assessment_recommended
    ):
        s1.professional_assessment_recommended = False
        notes.append(
            f"par_downgrade: professional_assessment_recommended -> False "
            f"(trade={s1.trade_category!r}, risk=medium, no_serious_hazard, "
            f"no_real_brief_hazards after boilerplate filter)"
        )

    # 2. Active hazard in motion — only escalate when motion signals a live fault.
    #
    # motion_detected=True is structurally true in any video of an ongoing repair
    # (someone adjusting a hinge or holding a tile is, by definition, moving).
    # Escalating on motion alone would block every active-repair video from reaching
    # DIY — the opposite of the intended behaviour.
    #
    # We only escalate when the model's own language describes an active, changing,
    # or worsening condition (leak, smoke, sparks, structural movement, etc.).
    # brief_hazard_observed already handles transient hazards in Rule 1 above.
    if s1.motion_detected and (s1.diy_suitability or "conditional") == "suitable":
        if _motion_indicates_hazard(s1):
            s1.diy_suitability = "conditional"
            notes.append(
                "motion_detected + hazard_keywords -> diy_suitability: suitable->conditional"
            )
            if not s1.clarifying_questions:
                s1.clarifying_questions = [
                    "The video shows what appears to be an active or changing condition. "
                    "Is this still ongoing? Is it safe to approach for a closer inspection?"
                ]
                notes.append(
                    "hazard_motion_detected -> added clarifying question about active state"
                )
        else:
            notes.append(
                "motion_detected but no hazard keywords — normal repair activity, "
                "diy_suitability unchanged"
            )

    # 3. Low temporal confidence — request clarification
    tc = s1.temporal_confidence if s1.temporal_confidence is not None else 75.0
    if tc < 60.0 and not s1.clarifying_questions:
        if s1.confidence > 75:
            old_conf = s1.confidence
            s1.confidence = 75
            notes.append(
                f"low temporal_confidence ({tc:.0f}) -> confidence capped {old_conf}->75"
            )
        s1.clarifying_questions = [
            "The video analysis had limited confidence in what was happening. "
            "Could you provide a clearer, well-lit video, or describe the issue "
            "in more detail?"
        ]
        notes.append(f"low temporal_confidence ({tc:.0f}) -> added clarifying question")

    # 4. Video quality issues — surface as clarification if nothing else fires
    if s1.video_quality_notes and s1.video_quality_notes.strip() and not s1.clarifying_questions:
        s1.needs_clarification = True
        s1.clarifying_questions = [
            f"The video quality limited the analysis: {s1.video_quality_notes.strip()} "
            "Could you upload a clearer video or describe the issue in more detail?"
        ]
        notes.append("video_quality_notes present -> added clarifying question")

    # 5. Detail-question demotion for low-risk video cases.
    #
    # When the model says DIY is suitable, confidence is adequate, no hazards are
    # present and no professional assessment is flagged, any remaining
    # clarifying_questions are detail-gathering rather than safety-blocking.
    # Examples: "What type of hinge?", "What tile material?", "What brand?"
    #
    # These questions can still improve repair instruction specificity, so they are
    # moved to optional_questions — where Stage 2/3 prompts can use them — rather
    # than being discarded. Crucially, they no longer block routing to DIY.
    #
    # This rule does NOT fire if ANY of the safety guards above elevated the case:
    #   - brief_hazard_observed / brief_hazards       (Rule 1 fired)
    #   - diy_suitability != "suitable"               (Rule 2 fired, or model said so)
    #   - confidence < 65                             (model genuinely unsure)
    #   - safety_concerns present                     (hazards named by model)
    #   - professional_assessment_recommended          (model said: call a pro)
    #
    # The check uses the CURRENT state of s1 (after all rules above), so if Rule 1
    # or 2 escalated anything, the conditions will no longer hold and demotion is
    # correctly skipped.
    _low_risk = (
        (s1.diy_suitability or "conditional") == "suitable"
        and (s1.confidence or 0) >= 65
        and not s1.brief_hazard_observed
        and not s1.brief_hazards
        and not s1.safety_concerns
        and not s1.professional_assessment_recommended
    )
    if _low_risk:
        demoted_something = False
        if s1.clarifying_questions:
            _demoted = s1.clarifying_questions[:]
            s1.clarifying_questions = []
            s1.optional_questions = list(s1.optional_questions or []) + _demoted
            notes.append(
                f"low_risk_video: demoted {len(_demoted)} clarifying_question(s) to "
                f"optional_questions (suitable + conf={s1.confidence} + no_hazards + "
                f"no_professional_flag)"
            )
            demoted_something = True
        # Clear needs_clarification and clarification_request so the pipeline's
        # confidence/clarification gate (analysis_service._BASE_CONFIDENCE_FLOOR)
        # does not re-trigger clarification using the now-demoted text.
        if s1.needs_clarification:
            s1.needs_clarification = False
            notes.append("low_risk_video: needs_clarification -> False")
            demoted_something = True
        if s1.clarification_request:
            s1.clarification_request = ""
            notes.append("low_risk_video: clarification_request cleared")
            demoted_something = True
        if not demoted_something:
            notes.append(
                f"low_risk_video: all clear (suitable + conf={s1.confidence} + "
                f"no_hazards + no_questions)"
            )

    return notes


# ── Public API ─────────────────────────────────────────────────────────────────

async def analyse_video(
    video_path: Path,
    *,
    description: Optional[str] = None,
) -> VideoStage1Output:
    """Analyse a video file and return a VideoStage1Output.

    This is the single entry point called by _run_video_pipeline() in
    analysis_service.py.  It returns a VideoStage1Output, which IS a Stage1Output,
    so the existing routing pipeline requires no changes.

    Parameters
    ----------
    video_path :
        Absolute path to the uploaded video file on disk.
    description :
        Optional free-text context from the homeowner (used as model prompt context).

    Raises
    ------
    VideoAnalysisError
        On any failure: flag disabled, file missing, model load error, OOM,
        inference error, invalid JSON, or schema validation failure.
        _run_video_pipeline() catches this and marks the analysis as failed.
    """
    if not VIDEO_ANALYSIS_ENABLED:
        raise VideoAnalysisError(
            "Video analysis is not enabled. "
            "Set VIDEO_ANALYSIS_ENABLED=true in backend/.env and restart the backend."
        )

    if not video_path.exists():
        raise VideoAnalysisError(f"Video file not found on disk: {video_path}")

    client = _get_client()
    active_model = VIDEO_MODEL_CLOUD if VIDEO_INFERENCE_MODE == "cloud" else VIDEO_MODEL_LOCAL

    log.info(
        "analyse_video_start mode=%s model=%s path=%s description=%s",
        VIDEO_INFERENCE_MODE,
        active_model,
        video_path.name,
        "yes" if (description or "").strip() else "no",
    )

    # Run inference
    try:
        raw, nframes_used = await client.infer(video_path, prompt=description or "")
    except VideoAnalysisError:
        raise  # already has a clear message
    except Exception as exc:
        raise VideoAnalysisError(
            f"Video inference failed unexpectedly: {type(exc).__name__}: {exc}"
        ) from exc

    log.info(
        "analyse_video_raw_output nframes_used=%d output_chars=%d",
        nframes_used, len(raw),
    )

    # Extract JSON — if the model produces no JSON at all (common for unrelated
    # or unintelligible video content), treat as not_repair rather than failing.
    try:
        data = _extract_json_from_raw(raw)
    except VideoAnalysisError as exc:
        if "contains no JSON object" in str(exc) or "JSON parse error" in str(exc):
            log.warning(
                "video_stage1: no JSON from model for %s — treating as not_repair. "
                "raw[:80]=%r",
                video_path.name, raw[:80],
            )
            return VideoStage1Output(
                scope="not_repair",
                problem_type="Unrelated or unrecognisable video content",
                trade_category="general/other",
                severity="low",
                observations=[],
                risk_level="low",
                diy_suitability="not_recommended",
                professional_assessment_recommended=False,
                needs_clarification=False,
                confidence=90,
                input_modality="video",
                motion_detected=False,
                brief_hazard_observed=False,
                temporal_confidence=75.0,
            )
        raise
    except Exception as exc:
        raise VideoAnalysisError(
            f"JSON extraction failed: {type(exc).__name__}: {exc}"
        ) from exc

    # Validate against schema
    from pydantic import ValidationError  # noqa: PLC0415
    try:
        result = VideoStage1Output.model_validate(data)
    except ValidationError as exc:
        field_errors = "; ".join(
            f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
            for e in exc.errors()[:5]
        )
        raise VideoAnalysisError(
            f"VideoStage1Output validation failed ({exc.error_count()} error(s)): "
            f"{field_errors}"
        ) from exc
    except Exception as exc:
        raise VideoAnalysisError(
            f"Schema validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    log.info(
        "analyse_video_complete nframes=%d confidence=%d mode=%s risk=%s diy=%s "
        "prof_rec=%s motion=%s brief_hazard=%s temporal_conf=%.0f",
        nframes_used,
        result.confidence,
        result.scope,
        result.risk_level,
        result.diy_suitability,
        result.professional_assessment_recommended,
        result.motion_detected,
        result.brief_hazard_observed,
        result.temporal_confidence if result.temporal_confidence is not None else 0.0,
    )

    return result
