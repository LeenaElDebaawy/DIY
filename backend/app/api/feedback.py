"""
Feedback API routes.

User-facing:
  POST /api/v1/cases/{case_id}/feedback     submit feedback on a completed analysis

Admin-only (require_admin → 403 for non-admins):
  GET   /api/v1/admin/feedback/overview      summary counts (count-first)
  GET   /api/v1/admin/feedback/priority      weighted-count priority board
  GET   /api/v1/admin/feedback/modes         per-mode breakdown
  GET   /api/v1/admin/feedback/comments      recent comments
  GET   /api/v1/admin/feedback/safety        safety incident queue
  PATCH /api/v1/admin/feedback/{id}          review a safety incident
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, require_admin
from app.database import get_db
from app.models import User
from app.schemas.feedback import FeedbackCreate, FeedbackResponse, IncidentUpdate
from app.services import feedback_service as svc

# Two routers mounted under /api/v1: one user-facing, one admin-only.
user_router = APIRouter(prefix="/cases/{case_id}/feedback", tags=["feedback"])
admin_router = APIRouter(prefix="/admin/feedback", tags=["admin-feedback"])


@user_router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    case_id: int,
    body: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback on a completed analysis of a case the user owns."""
    return await svc.create_feedback(db, case_id, current_user.id, body)


@admin_router.get("/overview")
async def overview(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    return await svc.get_overview(db)


@admin_router.get("/priority")
async def priority(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    return await svc.get_priority_board(db)


@admin_router.get("/modes")
async def modes(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    return await svc.get_mode_breakdown(db)


@admin_router.get("/comments", response_model=list[FeedbackResponse])
async def comments(
    limit: int = 10,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    return await svc.list_recent_comments(db, limit=limit)


@admin_router.get("/safety", response_model=list[FeedbackResponse])
async def safety(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    return await svc.list_safety_incidents(db)


@admin_router.patch("/{feedback_id}", response_model=FeedbackResponse)
async def review_incident(
    feedback_id: int,
    body: IncidentUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    return await svc.update_incident(db, feedback_id, body)
