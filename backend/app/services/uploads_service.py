import io
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile, status
from PIL import Image as PILImage   # Pillow, reads image headers, not full pixel data
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import Session model aliased as RepairSession so the name does not clash
# with SQLAlchemy's AsyncSession which is also called "Session" in some imports.
from app.models import Session as RepairSession, Upload

# Where uploaded files are stored on disk.
# Resolves to  backend/uploads/  regardless of where uvicorn is launched from.
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Image constraints (unchanged) ─────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB

# ── Video constraints ──────────────────────────────────────────────────────────
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm"}
MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_VIDEO_DURATION_SECONDS = 120.0   # 2 minutes

# Combined allowlist — used by the MIME gate in save_upload
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES

# Legacy alias kept for any existing callers / tests that reference it directly
MAX_BYTES = MAX_IMAGE_BYTES


# ── Image helpers (unchanged) ─────────────────────────────────────────────────

def _extract_dimensions(content: bytes) -> tuple[int | None, int | None]:
    """Return (width, height) by reading only the image file header.

    PIL's Image.open() is lazy; for JPEG and PNG it reads just enough bytes
    to determine dimensions without decompressing the full image.
    Returns (None, None) if the format is unrecognised or the header is corrupt.

    The dimensions are stored on the Upload row so the AI pipeline can pass
    image size metadata to the vision model without re-reading files from disk.
    """
    try:
        with PILImage.open(io.BytesIO(content)) as img:
            return img.size   # (width, height)
    except Exception:
        return None, None


# ── Video helpers ─────────────────────────────────────────────────────────────

def _extract_video_metadata(video_path: Path) -> Optional[float]:
    """Return video duration in seconds from an on-disk file.

    Reads the container header only — does not decode any frames.
    Returns None if the file is unreadable or the container reports no duration.

    PyAV (`av`) is imported inside the function so the image-only code path
    never pays the import cost when video support is not needed.
    """
    try:
        import av  # noqa: PLC0415
        with av.open(str(video_path)) as container:
            if container.duration and container.duration > 0:
                return container.duration / 1_000_000  # microseconds → seconds
            # Some containers store duration on the stream rather than container
            for stream in container.streams.video:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
        return None
    except Exception:
        return None


def _extract_video_thumbnails(video_path: Path, storage_stem: str) -> list[str]:
    """Extract 3 JPEG display frames at 10 %, 50 %, 90 % of video duration.

    Frames are resized to fit within 640×480 and saved as JPEG (quality 82)
    alongside the original video in UPLOAD_DIR, using the naming convention
    {storage_stem}_thumb_{1|2|3}.jpg.

    Returns a list of filenames (not full paths); may be shorter than 3 if
    seeking or decoding fails for a given timestamp.  All failures are silenced:
    thumbnails are best-effort display aids, not diagnostic data.
    """
    filenames: list[str] = []
    try:
        import av  # noqa: PLC0415
        with av.open(str(video_path)) as container:
            duration_us = container.duration
            if not duration_us or duration_us <= 0:
                return filenames

            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream is None:
                return filenames

            for i, fraction in enumerate((0.10, 0.50, 0.90)):
                seek_us = int(duration_us * fraction)
                try:
                    container.seek(seek_us)
                    frame = None
                    for packet in container.demux(video_stream):
                        for f in packet.decode():
                            frame = f
                            break
                        if frame is not None:
                            break

                    if frame is None:
                        continue

                    img = frame.to_image()       # PIL Image
                    img.thumbnail((640, 480))    # in-place resize, preserves aspect ratio

                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=82)

                    filename = f"{storage_stem}_thumb_{i + 1}.jpg"
                    (UPLOAD_DIR / filename).write_bytes(buf.getvalue())
                    filenames.append(filename)
                except Exception:
                    continue   # skip this seek point; attempt the next one

    except Exception:
        pass   # return whatever thumbnails were successfully extracted

    return filenames


# ── Case ownership guard (unchanged) ─────────────────────────────────────────

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


# ── Main entry points ─────────────────────────────────────────────────────────

async def save_upload(
    db: AsyncSession,
    case_id: int,
    user_id: int,
    file: UploadFile,
) -> Upload:
    """Validate, save to disk, and insert an Upload row.

    Image uploads follow the original code path completely unchanged.
    Video uploads additionally extract duration and generate display thumbnails.
    Both paths converge at the Upload row construction below.
    """
    await _require_owned_case(db, case_id, user_id)

    content_type = file.content_type or ""
    is_video = content_type in ALLOWED_VIDEO_TYPES
    is_image = content_type in ALLOWED_IMAGE_TYPES

    if not is_image and not is_video:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"File type '{content_type}' is not allowed. "
                "Use JPEG, PNG, GIF or WebP for images, "
                "or MP4, MOV or WebM for video."
            ),
        )

    content = await file.read()

    max_bytes = MAX_VIDEO_BYTES if is_video else MAX_IMAGE_BYTES
    size_label = "100 MB" if is_video else "10 MB"
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File is too large. Maximum size is {size_label}.",
        )

    # Write to disk first so video helpers can open the file by path
    suffix = Path(file.filename or "").suffix.lower() or (".mp4" if is_video else ".bin")
    filename = f"{uuid.uuid4()}{suffix}"
    dest = UPLOAD_DIR / filename
    dest.write_bytes(content)

    # ── Image path (original, unchanged) ──────────────────────────────────────
    width: Optional[int] = None
    height: Optional[int] = None
    source_type = "image"
    duration_seconds: Optional[float] = None
    thumbnail_paths: Optional[list] = None

    if is_image:
        width, height = _extract_dimensions(content)

    # ── Video path (additive) ─────────────────────────────────────────────────
    if is_video:
        source_type = "video"
        duration_seconds = _extract_video_metadata(dest)

        if (
            duration_seconds is not None
            and duration_seconds > MAX_VIDEO_DURATION_SECONDS
        ):
            dest.unlink(missing_ok=True)   # clean up the written file
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Video is too long. Maximum duration is "
                    f"{int(MAX_VIDEO_DURATION_SECONDS)} seconds "
                    f"({int(MAX_VIDEO_DURATION_SECONDS) // 60} minutes)."
                ),
            )

        storage_stem = Path(filename).stem
        extracted = _extract_video_thumbnails(dest, storage_stem)
        thumbnail_paths = extracted if extracted else None

    upload = Upload(
        session_id=case_id,
        original_filename=file.filename or filename,
        mime_type=content_type,
        file_size_bytes=len(content),
        storage_path=filename,
        source_type=source_type,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
        thumbnail_paths=thumbnail_paths,
    )
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    return upload


async def list_uploads(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> list[Upload]:
    """Return all uploads for a case owned by this user."""
    await _require_owned_case(db, case_id, user_id)
    result = await db.execute(
        select(Upload)
        .where(Upload.session_id == case_id)
        .order_by(Upload.uploaded_at.asc())
    )
    return list(result.scalars().all())
