import datetime
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# The ORM model is named Session (a repair session).
# We alias it to RepairSession here to avoid confusion with
# SQLAlchemy's own AsyncSession class, which is also in scope.
from app.models import Feedback, Session as RepairSession, Upload


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
    """
    Return all cases owned by the given user, newest first.

    Analyses are selectinload-ed so that Session.confidence (a Python
    property that reads structured_result JSONB) is accessible without
    triggering lazy-load errors in the async context.
    """
    result = await db.execute(
        select(RepairSession)
        .where(RepairSession.user_id == user_id)
        .options(selectinload(RepairSession.analyses))
        .order_by(RepairSession.created_at.desc())
    )
    return list(result.scalars().all())


async def get_case(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> RepairSession | None:
    """Return a single case if it exists and belongs to the user.

    Analyses are selectinload-ed so that Session.output_mode and
    Session.confidence (Python properties that read structured_result JSONB)
    are accessible without triggering lazy-load errors in the async context.
    """
    result = await db.execute(
        select(RepairSession)
        .where(
            RepairSession.id == case_id,
            RepairSession.user_id == user_id,
        )
        .options(selectinload(RepairSession.analyses))
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
    Partial update, only fields that are not None are written.

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


async def acknowledge_hazard(
    db: AsyncSession,
    case_id: int,
    user_id: int,
) -> RepairSession | None:
    """
    Permanently record that the user has acknowledged the hazard notice for
    this case.

    Rules:
    - Returns None if the case does not exist or belongs to another user.
    - Returns 400 (raised here as ValueError) if the case's output_mode is
      not 'assessment_first' — only hazardous cases may be acknowledged.
    - Idempotent: if hazard_ack is already True the row is not modified and
      the existing timestamp is preserved.

    The output_mode is read from the latest completed Analysis.structured_result.
    If no completed analysis exists yet the call is rejected with ValueError
    ("no completed analysis") so the route handler can return 409.

    P5 integration point: when hazard_events is implemented, emit a
    HAZARD_ACK event inside this function after the commit.
    """
    from app.models import Analysis  # local import avoids circular dependency

    case = await get_case(db, case_id, user_id)
    if case is None:
        return None

    # Idempotent: already acknowledged — return as-is without touching timestamp.
    if case.hazard_ack:
        return case

    # Verify the case is assessment_first before acknowledging.
    analysis_result = await db.execute(
        select(Analysis)
        .where(Analysis.session_id == case_id, Analysis.status == "complete")
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    analysis = analysis_result.scalar_one_or_none()
    if analysis is None:
        raise ValueError("no_completed_analysis")
    output_mode = (analysis.structured_result or {}).get("output_mode", "diy")
    if output_mode != "assessment_first":
        raise ValueError(f"wrong_output_mode:{output_mode}")

    case.hazard_ack = True
    case.hazard_ack_at = datetime.datetime.now(datetime.timezone.utc)
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
      1. Load the case and its uploads (selectinload, avoids lazy SQL in async context).
      2. Delete each upload's physical file from disk.
      3. Delete the ORM object, SQLAlchemy's cascade="all, delete-orphan" on
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

    # Safety-evidence guard: refuse to delete a case that carries safety feedback,
    # so a reported safety incident's originating analysis is preserved for review.
    _safety_n = (await db.execute(
        select(func.count()).select_from(Feedback).where(
            Feedback.session_id == case_id, Feedback.is_safety.is_(True)
        )
    )).scalar_one()
    if _safety_n > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This case has safety feedback and cannot be deleted while under review.",
        )

    # backend/uploads/, resolved relative to this file so it is correct
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
