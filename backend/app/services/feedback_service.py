"""
Feedback service (Phase 1).

Creation (user-facing) + read aggregates and incident review (admin-facing).
All analytics are COUNT-FIRST and carry their own sample size — no charts here,
the dashboard renders raw counts so it stays honest at low N.
"""
from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Analysis, Feedback, Session as RepairSession
from app.schemas.feedback import FeedbackCreate, IncidentUpdate

# Severity weights for the priority score. Safety is handled in its own lane and
# is NOT scored here. Higher weight = a worse failure for the same frequency.
_ISSUE_WEIGHTS = {
    "wrong_professional_recommendation": 3,
    "wrong_diy_recommendation":          3,
    "scope_incorrect":                   2,
    "wrong_severity":                    2,
    "wrong_trade":                       2,
    "clarification_not_useful":          1,
    "steps_unclear":                     1,
    "care_not_useful":                   1,
    "other":                             1,
}


async def _require_owned_case(db: AsyncSession, case_id: int, user_id: int) -> RepairSession:
    case = (await db.execute(
        select(RepairSession).where(
            RepairSession.id == case_id,
            RepairSession.user_id == user_id,
        )
    )).scalar_one_or_none()
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")
    return case


# ── Creation ─────────────────────────────────────────────────────────────────
async def create_feedback(
    db: AsyncSession, case_id: int, user_id: int, body: FeedbackCreate
) -> Feedback:
    """Validate ownership, snapshot the analysis mode, auto-flag safety, insert."""
    await _require_owned_case(db, case_id, user_id)

    analysis = (await db.execute(
        select(Analysis).where(
            Analysis.id == body.analysis_id,
            Analysis.session_id == case_id,
        )
    )).scalar_one_or_none()
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found for this case.",
        )

    sr = analysis.structured_result or {}
    output_mode = sr.get("output_mode")
    is_emergency = bool((sr.get("out_of_scope") or {}).get("is_emergency"))

    # Auto safety-flag: hazardous assessment, active emergency, or a safety issue.
    is_safety = (
        output_mode == "assessment_first"
        or (output_mode == "out_of_scope" and is_emergency)
        or body.safety_category is not None
    )

    fb = Feedback(
        session_id=case_id,
        analysis_id=body.analysis_id,
        user_id=user_id,
        rating=body.rating,
        issue_category=body.issue_category,
        comment=body.comment,
        output_mode=output_mode,
        is_safety=is_safety,
        safety_category=body.safety_category,
        status="new",
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return fb


# ── Admin reads ──────────────────────────────────────────────────────────────
async def get_overview(db: AsyncSession) -> dict:
    total_analyses = (await db.execute(
        select(func.count()).select_from(Analysis).where(Analysis.status == "complete")
    )).scalar_one()
    total_feedback = (await db.execute(select(func.count()).select_from(Feedback))).scalar_one()

    rating_rows = (await db.execute(
        select(Feedback.rating, func.count()).group_by(Feedback.rating)
    )).all()
    ratings = {r: c for r, c in rating_rows}
    helpful = ratings.get("helpful", 0)
    partial = ratings.get("partial", 0)
    not_helpful = ratings.get("not", 0)

    safety_total = (await db.execute(
        select(func.count()).select_from(Feedback).where(Feedback.is_safety.is_(True))
    )).scalar_one()
    safety_open = (await db.execute(
        select(func.count()).select_from(Feedback).where(
            Feedback.is_safety.is_(True), Feedback.status != "resolved"
        )
    )).scalar_one()

    response_rate = (total_feedback / total_analyses) if total_analyses else 0.0
    helpful_rate = (helpful / total_feedback) if total_feedback else 0.0

    return {
        "total_analyses": total_analyses,
        "total_feedback": total_feedback,
        "response_rate": round(response_rate, 3),
        "helpful": helpful,
        "partial": partial,
        "not_helpful": not_helpful,
        "helpful_rate": round(helpful_rate, 3),
        "safety_total": safety_total,
        "safety_open": safety_open,
    }


async def get_priority_board(db: AsyncSession) -> list[dict]:
    """Weighted-count ranking of non-safety issues from partial/negative feedback."""
    rows = (await db.execute(
        select(Feedback.issue_category, func.count())
        .where(
            Feedback.rating.in_(["partial", "not"]),
            Feedback.issue_category.is_not(None),
            Feedback.is_safety.is_(False),
        )
        .group_by(Feedback.issue_category)
    )).all()
    board = []
    for category, count in rows:
        weight = _ISSUE_WEIGHTS.get(category, 1)
        board.append({
            "issue_category": category,
            "count": count,
            "weight": weight,
            "priority_score": count * weight,
        })
    board.sort(key=lambda x: (-x["priority_score"], -x["count"]))
    return board


async def get_mode_breakdown(db: AsyncSession) -> list[dict]:
    rows = (await db.execute(
        select(Feedback.output_mode, Feedback.rating, func.count())
        .group_by(Feedback.output_mode, Feedback.rating)
    )).all()
    modes: dict[str, dict] = {}
    for mode, rating, count in rows:
        m = modes.setdefault(mode or "unknown",
                             {"output_mode": mode or "unknown", "helpful": 0,
                              "partial": 0, "not": 0, "total": 0})
        key = rating if rating in ("helpful", "partial", "not") else "not"
        m[key] += count
        m["total"] += count
    return sorted(modes.values(), key=lambda x: -x["total"])


async def list_recent_comments(db: AsyncSession, limit: int = 10) -> list[Feedback]:
    return list((await db.execute(
        select(Feedback)
        .where(Feedback.comment.is_not(None))
        .order_by(Feedback.created_at.desc())
        .limit(limit)
    )).scalars().all())


async def list_safety_incidents(db: AsyncSession) -> list[Feedback]:
    # New first, then by recency, so the review queue surfaces untriaged items.
    return list((await db.execute(
        select(Feedback)
        .where(Feedback.is_safety.is_(True))
        .order_by(Feedback.status.asc(), Feedback.created_at.desc())
    )).scalars().all())


async def update_incident(db: AsyncSession, feedback_id: int, body: IncidentUpdate) -> Feedback:
    fb = await db.get(Feedback, feedback_id)
    if fb is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found.")
    if body.status is not None:
        fb.status = body.status
    if body.resolution is not None:
        fb.resolution = body.resolution
    if body.reviewer_note is not None:
        fb.reviewer_note = body.reviewer_note
    await db.commit()
    await db.refresh(fb)
    return fb


async def case_has_safety_feedback(db: AsyncSession, case_id: int) -> bool:
    """Used by the case-delete guard to preserve safety evidence."""
    n = (await db.execute(
        select(func.count()).select_from(Feedback).where(
            Feedback.session_id == case_id, Feedback.is_safety.is_(True)
        )
    )).scalar_one()
    return n > 0
