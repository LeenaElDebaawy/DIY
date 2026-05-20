from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# The ORM model is named Session (a repair session).
# We alias it to RepairSession here to avoid confusion with
# SQLAlchemy's own AsyncSession class, which is also in scope.
from app.models import Session as RepairSession, Upload


async def create_case(
    db: AsyncSession,
    user_id: int,
    title: str | None = "Diagnosing…",
    category: str | None = None,
) -> RepairSession:
    """Insert a new repair case and return the persisted row."""
    case = RepairSession(
        user_id=user_id,
        title=title,
        category=category,
        status="open",          # "open" is the starting state for a new case
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)      # populates server-side defaults (created_at, updated_at)
    return case


async def list_cases(
    db: AsyncSession,
    user_id: int,
) -> list[RepairSession]:
    """Return all cases owned by the given user, newest first."""
    result = await db.execute(
        select(RepairSession)
        .where(RepairSession.user_id == user_id)
        .order_by(RepairSession.created_at.desc())
    )
    return list(result.scalars().all())


async def get_case(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> RepairSession | None:
    """Return a single case if it exists and belongs to the user."""
    result = await db.execute(
        select(RepairSession).where(
            RepairSession.id == case_id,
            RepairSession.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def patch_case(
    db: AsyncSession,
    case_id: int,
    user_id: int,
    title: str | None = None,
    category: str | None = None,
    status: str | None = None,
) -> RepairSession | None:
    """
    Partial update — only fields that are not None are written.

    Used by:
      - the pipeline (auto-title from Stage 1 problem_type)
      - the PATCH /cases/{case_id} API endpoint (manual edits)
    """
    case = await get_case(db, case_id, user_id)
    if case is None:
        return None
    if title    is not None: case.title    = title
    if category is not None: case.category = category
    if status   is not None: case.status   = status
    await db.commit()
    await db.refresh(case)
    return case


async def delete_case(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> bool:
    """
    Permanently delete a case and all related data.

    Steps:
      1. Load the case and its uploads (selectinload — avoids lazy SQL in async context).
      2. Delete each upload's physical file from disk.
      3. Delete the ORM object — SQLAlchemy's cascade="all, delete-orphan" on
         Session.uploads / .messages / .analyses removes the child DB rows automatically.

    Returns True if the case was found and deleted, False if it was not found
    or belongs to another user (same response to avoid leaking existence).
    """
    result = await db.execute(
        select(RepairSession)
        .where(
            RepairSession.id == case_id,
            RepairSession.user_id == user_id,
        )
        .options(selectinload(RepairSession.uploads))   # pre-load so we can read storage_path
    )
    case = result.scalar_one_or_none()
    if case is None:
        return False

    # backend/uploads/ — resolved relative to this file so it is correct
    # regardless of where uvicorn is launched from.
    upload_dir = Path(__file__).resolve().parent.parent.parent / "uploads"

    for upload in case.uploads:
        file_path = upload_dir / upload.storage_path
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass    # log in production; silently skip here

    await db.delete(case)
    await db.commit()
    return True
