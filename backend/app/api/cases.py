from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.cases import CaseCreate, CasePatch, CaseResponse
from app.services.cases_service import (
    acknowledge_hazard,
    create_case,
    delete_case,
    get_case,
    list_cases,
    patch_case,
)

# Mounted under /api/v1 in router.py, so full paths are:
#   POST   /api/v1/cases
#   GET    /api/v1/cases
#   GET    /api/v1/cases/{case_id}
#   PATCH  /api/v1/cases/{case_id}
#   DELETE /api/v1/cases/{case_id}
router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_new_case(
    body: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Open a new repair case for the authenticated user."""
    return await create_case(db, current_user.id, body.title, body.category)


@router.get("", response_model=list[CaseResponse])
async def list_my_cases(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all cases belonging to the authenticated user, newest first."""
    return await list_cases(db, current_user.id)


@router.get("/{case_id}", response_model=CaseResponse)
async def get_my_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a single case.  404 if it does not exist or belongs to another user."""
    case = await get_case(db, case_id, current_user.id)
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )
    return case


@router.patch("/{case_id}", response_model=CaseResponse)
async def patch_my_case(
    case_id: int,
    body: CasePatch,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Partial update, only non-null fields are written."""
    case = await patch_case(
        db, case_id, current_user.id,
        title=body.title, category=body.category, status=body.status,
    )
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")
    return case


@router.patch(
    "/{case_id}/hazard-ack",
    response_model=CaseResponse,
    summary="Acknowledge the hazard notice for an assessment-first case",
)
async def acknowledge_hazard_notice(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Record that the user has read and acknowledged the hazardous-case safety
    notice.

    Rules:
    - 404 if the case does not exist or belongs to another user.
    - 400 if the case's output_mode is not 'assessment_first'.
    - 409 if no completed analysis exists yet for the case.
    - Idempotent: calling twice returns 200 with the existing timestamp unchanged.

    On success returns the updated CaseResponse with hazard_ack=true and
    hazard_ack_at set to the UTC timestamp of the first acknowledgement.
    """
    try:
        case = await acknowledge_hazard(db, case_id, current_user.id)
    except ValueError as exc:
        msg = str(exc)
        if msg == "no_completed_analysis":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No completed analysis found for this case. "
                       "Run the analysis first.",
            )
        # wrong_output_mode:diy / clarification / care / verify_first
        mode = msg.split(":", 1)[-1] if ":" in msg else msg
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hazard acknowledgement is only valid for assessment_first "
                   f"cases (this case is '{mode}').",
        )
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )
    return case


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete a case and all its uploads, messages, and analyses.

    Ownership is enforced, a user can only delete their own cases.
    Returns 204 No Content on success, 404 if the case is not found or
    belongs to another user (same response to avoid leaking existence).
    """
    deleted = await delete_case(db, case_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )
