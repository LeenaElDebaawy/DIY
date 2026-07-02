"""
Judge API routes (admin-only).

case_router (mounted under /api/v1):
  POST /api/v1/cases/{case_id}/judge   — run judge on latest completed analysis
  GET  /api/v1/cases/{case_id}/judge   — fetch stored judge result

admin_router (mounted under /api/v1/admin/judge):
  GET  /api/v1/admin/judge/summary     — aggregate issue tags across all judged analyses

All endpoints require the require_admin dependency (403 for non-admins).
Judge failure never propagates as a 500 — a 503 is returned when JUDGE_ENABLED=false,
and a 422 is returned when the judge model fails to produce usable output.
"""
import os
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import attributes

from app.core.deps import require_admin
from app.database import get_db
from app.models import Analysis, Session, User
from app.services.judge_service import JUDGE_ENABLED, run_judge

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:7b")

case_router  = APIRouter(prefix="/cases/{case_id}", tags=["judge"])
admin_router = APIRouter(prefix="/admin/judge",     tags=["admin-judge"])


async def _get_latest_complete_analysis(db: AsyncSession, case_id: int) -> Analysis | None:
    result = await db.execute(
        select(Analysis)
        .where(Analysis.session_id == case_id, Analysis.status == "complete")
        .order_by(desc(Analysis.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


@case_router.post("/judge", summary="Run judge on latest completed analysis (admin only)")
async def run_case_judge(
    case_id: int,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    if not JUDGE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Judge is disabled. Set JUDGE_ENABLED=true in backend/.env and restart.",
        )

    analysis = await _get_latest_complete_analysis(db, case_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed analysis found for this case.",
        )

    structured_result = analysis.structured_result or {}
    judge_result = await run_judge(analysis.id, structured_result)

    if judge_result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Judge model failed to produce a usable evaluation. Check backend logs.",
        )

    # Persist in stage_outputs["judge"] (additive — never overwrites other keys)
    stage_outputs = dict(analysis.stage_outputs or {})
    stage_outputs["judge"] = judge_result
    analysis.stage_outputs = stage_outputs
    attributes.flag_modified(analysis, "stage_outputs")
    await db.commit()

    return judge_result


@case_router.get("/judge", summary="Get stored judge result for case (admin only)")
async def get_case_judge(
    case_id: int,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    analysis = await _get_latest_complete_analysis(db, case_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed analysis found for this case.",
        )

    judge = (analysis.stage_outputs or {}).get("judge")
    if judge is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No judge evaluation yet. Run POST /judge first.",
        )
    return judge


@admin_router.get("/summary", summary="Recurring issue summary across all judged analyses (admin only)")
async def get_judge_summary(
    limit: int = 50,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the most frequent issue tags across all judged analyses,
    overall score distribution, verdict counts, and per-tag suggested fixes.
    """
    result = await db.execute(
        select(Analysis)
        .where(Analysis.status == "complete")
        .order_by(desc(Analysis.created_at))
        .limit(limit)
    )
    analyses = result.scalars().all()

    judged = [
        (a.stage_outputs or {}).get("judge")
        for a in analyses
        if (a.stage_outputs or {}).get("judge")
    ]

    if not judged:
        return {
            "total_judged": 0,
            "verdict_counts": {},
            "top_issue_tags": [],
            "score_distribution": {},
            "suggested_fixes": [],
        }

    verdict_counts: Counter = Counter()
    tag_counts: Counter     = Counter()
    scores: list[int]       = []

    for j in judged:
        verdict_counts[j.get("final_verdict", "unknown")] += 1
        for tag in (j.get("issue_tags") or []):
            tag_counts[tag] += 1
        score = j.get("overall_score")
        if score is not None:
            scores.append(int(score))

    # Score buckets
    def _bucket(s: int) -> str:
        if s >= 85:   return "excellent (85-100)"
        if s >= 70:   return "good (70-84)"
        if s >= 50:   return "needs_review (50-69)"
        return "poor (<50)"

    score_dist: Counter = Counter(_bucket(s) for s in scores)

    top_tags = tag_counts.most_common(12)

    from app.services.judge_service import _PROPOSED_FIXES
    suggested_fixes = [
        {
            "tag":          tag,
            "count":        count,
            "proposed_fix": _PROPOSED_FIXES.get(tag, ""),
        }
        for tag, count in top_tags
    ]

    return {
        "total_judged":       len(judged),
        "verdict_counts":     dict(verdict_counts),
        "top_issue_tags":     [{"tag": t, "count": c} for t, c in top_tags],
        "score_distribution": dict(score_dist),
        "suggested_fixes":    suggested_fixes,
        "judge_model":        JUDGE_MODEL,
    }
