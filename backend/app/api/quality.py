"""Quality dashboard API — admin-only, read-only."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import require_admin
from app.database import get_db
from app.models import User
from app.services import quality_service as svc
from app.services import proposal_service as proposal_svc

quality_router = APIRouter(prefix="/admin/quality", tags=["admin-quality"])


@quality_router.get("/summary")
async def quality_summary(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Full QA summary for the admin quality dashboard.

    Returns aggregated metrics, per-case alignment data, recurring issues,
    and advisory improvement suggestions derived from judge evaluations and
    user feedback. Read-only — no mutations.
    """
    return await svc.get_summary(db)


class FixProposalRequest(BaseModel):
    suggestion: str
    signal: str
    trigger_issues: list[str]
    total_occurrences: int
    pipeline_area: str
    example_case_ids: list[int]


@quality_router.post("/fix-proposal")
async def generate_fix_proposal(
    body: FixProposalRequest,
    _: User = Depends(require_admin),
) -> dict:
    """
    Generate an advisory AI fix proposal for a detected quality pattern.
    Uses local Ollama (same model as the judge). Admin-only. Writes nothing.
    """
    return await proposal_svc.generate_fix_proposal(body.model_dump())
