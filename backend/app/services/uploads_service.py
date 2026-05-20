import io
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image as PILImage   # Pillow — reads image headers, not full pixel data
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import Session model aliased as RepairSession so the name does not clash
# with SQLAlchemy's AsyncSession which is also called "Session" in some imports.
from app.models import Session as RepairSession, Upload

# Where uploaded files are stored on disk.
# Resolves to  backend/uploads/  regardless of where uvicorn is launched from.
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def _extract_dimensions(content: bytes) -> tuple[int | None, int | None]:
    """
    Return (width, height) by reading only the image file header.

    PIL's Image.open() is lazy — for JPEG and PNG it reads just enough bytes
    to determine the dimensions without decompressing the full image.
    Returns (None, None) if the format is unrecognised or the header is corrupt.

    The dimensions are stored on the Upload row so the AI pipeline can pass
    image size metadata to the vision model without re-reading files from disk.
    """
    try:
        with PILImage.open(io.BytesIO(content)) as img:
            return img.size   # (width, height)
    except Exception:
        return None, None


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


async def save_upload(
    db: AsyncSession,
    case_id: int,
    user_id: int,
    file: UploadFile,
) -> Upload:
    """Validate, save to disk, and insert an Upload row."""
    await _require_owned_case(db, case_id, user_id)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File type '{file.content_type}' is not allowed. Use JPEG, PNG, GIF or WebP.",
        )

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is too large. Maximum size is 10 MB.",
        )

    # Use a UUID filename to avoid collisions if two users upload "photo.jpg"
    suffix = Path(file.filename).suffix.lower() or ".bin"
    filename = f"{uuid.uuid4()}{suffix}"
    dest = UPLOAD_DIR / filename
    dest.write_bytes(content)

    width, height = _extract_dimensions(content)

    upload = Upload(
        session_id=case_id,
        original_filename=file.filename,
        mime_type=file.content_type,
        file_size_bytes=len(content),
        storage_path=filename,
        width=width,
        height=height,
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
