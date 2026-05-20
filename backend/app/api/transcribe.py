"""
Voice transcription endpoint.

POST /api/v1/transcribe
  Accepts a multipart audio file (webm/ogg/wav from MediaRecorder).
  Returns {"text": "..."} using faster-whisper tiny model.
  The av (PyAV) library bundled with faster-whisper decodes any format
  that libavcodec supports — no separate system ffmpeg required.

The WhisperModel is loaded once and cached as a module-level singleton.
First call downloads ~75 MB of model weights from HuggingFace.
Subsequent calls are instant (model already in memory).
"""

import asyncio
import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.deps import get_current_user
from app.models import User

log = logging.getLogger(__name__)
router = APIRouter(prefix="/transcribe", tags=["transcribe"])

# ── Lazy singleton ────────────────────────────────────────────────────────────
_model = None
_model_lock = asyncio.Lock()


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:          # double-checked locking
            return _model
        try:
            from faster_whisper import WhisperModel
            log.info("Loading Whisper tiny model (first use — downloading if needed)…")
            # int8 quantisation: fast, low memory, good accuracy for short clips
            _model = WhisperModel("tiny", device="cpu", compute_type="int8")
            log.info("Whisper model ready.")
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="faster-whisper is not installed. Run: pip install faster-whisper",
            )
    return _model


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("")
async def transcribe_audio(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Transcribe a voice recording to text.

    The client sends audio/webm (from MediaRecorder) as multipart form data.
    Returns {"text": "<transcription>"} or {"text": ""} if nothing was heard.
    """
    data = await audio.read()

    # Reject suspiciously small blobs (user clicked stop immediately)
    if len(data) < 2_000:
        return {"text": ""}

    # Determine file extension from content-type for a correct temp-file suffix
    ct = (audio.content_type or "").lower()
    if "wav" in ct:
        suffix = ".wav"
    elif "mp4" in ct or "m4a" in ct:
        suffix = ".mp4"
    elif "ogg" in ct:
        suffix = ".ogg"
    else:
        suffix = ".webm"   # Chrome / Edge default

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        model = await _get_model()

        # Transcription is CPU-bound — run in thread pool so the async loop stays free
        loop = asyncio.get_running_loop()

        def _transcribe():
            segments, _ = model.transcribe(
                tmp_path,
                language="en",
                beam_size=1,            # fastest setting
                vad_filter=True,        # skip silent sections
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return " ".join(s.text for s in segments).strip()

        text = await loop.run_in_executor(None, _transcribe)
        log.info("Transcribed %d bytes → %d chars", len(data), len(text))
        return {"text": text}

    except Exception as exc:
        log.error("Transcription failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
