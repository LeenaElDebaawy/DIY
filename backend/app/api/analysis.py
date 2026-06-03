"""
Analysis API routes.

POST /api/v1/cases/{case_id}/analyse
    Trigger the 3-stage AI analysis pipeline for a case.
    Returns 202 Accepted immediately with status="pending".
    The pipeline runs as a background task.

GET  /api/v1/cases/{case_id}/analysis
    Return the latest analysis for a case.
    Poll this endpoint while status == "running".
    pipeline_stage updates after each stage completes so the frontend
    can show granular progress labels.

HTTP status choices:
    202 Accepted , used for the trigger because the work is not done yet.
                    This is the correct REST status for "started, go poll".
    200 OK       , used for the GET because it returns the current state.
    404          , no analysis exists yet for this case.
    409 Conflict , a pipeline is already running (duplicate trigger attempt).
    422          , no photos uploaded yet.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User
from app.schemas.analysis import AnalyseRequest, AnalysisResponse
from app.services.analysis_service import create_analysis, get_latest_analysis

# Mounted under /api/v1 in router.py.
# prefix="/cases/{case_id}" is a sub-resource of the cases router.
router = APIRouter(prefix="/cases/{case_id}", tags=["analysis"])


@router.post(
    "/analyse",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger the 3-stage AI analysis pipeline",
)
async def trigger_analysis(
    case_id: int,
    body: AnalyseRequest = AnalyseRequest(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start the analysis pipeline for this case.

    The optional body may contain `text_context`, a free-text note from the
    user (e.g. "The pipe makes a dripping sound and the joint is wet").
    This is injected into the Stage 1 vision prompt so the model has two
    evidence streams: the images and the user's own description.

    The pipeline runs in the background.  This endpoint returns immediately
    with status="pending".  Poll GET /analysis to track progress.

    The three stages run in sequence:
      1. Image / problem classification (vision model)
      2. Tools, materials, protective gear determination
      3. Step-by-step repair instruction generation

    Each stage's output feeds into the next as context.
    """
    return await create_analysis(db, case_id, current_user.id, text_context=body.text_context)


@router.get(
    "/analysis",
    response_model=AnalysisResponse,
    summary="Get the latest analysis result (or current progress)",
)
async def get_analysis(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the most recent analysis for a case.

    Use the status and pipeline_stage fields to decide what to show:
      status="pending"  → pipeline is queued, show spinner
      status="running"  → pipeline is active, show stage label from pipeline_stage
      status="complete" → render structured_result
      status="failed"   → show validation_issues, offer retry button

    Returns 404 if no analysis has been triggered for this case yet.
    """
    analysis = await get_latest_analysis(db, case_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis found for this case. Trigger one with POST /analyse.",
        )
    return analysis
