from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.cases import CaseCreate, CasePatch, CaseResponse
from app.services.cases_service import create_case, delete_case, get_case, list_cases, patch_case

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
    """Partial update — only non-null fields are written."""
    case = await patch_case(
        db, case_id, current_user.id,
        title=body.title, category=body.category, status=body.status,
    )
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")
    return case


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete a case and all its uploads, messages, and analyses.

    Ownership is enforced — a user can only delete their own cases.
    Returns 204 No Content on success, 404 if the case is not found or
    belongs to another user (same response to avoid leaking existence).
    """
    deleted = await delete_case(db, case_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )
