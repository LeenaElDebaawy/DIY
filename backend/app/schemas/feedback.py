"""
Pydantic schemas for the feedback / data-collection mechanism (Phase 1).

The issue-category and safety-category vocabularies are defined here as the
single source of truth, shared conceptually with the frontend widget and the
admin dashboard so collection and analytics stay aligned.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ── Controlled vocabularies (single source of truth) ─────────────────────────
# Non-safety issue categories (the second question when rating is partial/not).
ISSUE_CATEGORIES = {
    "wrong_trade",            # wrong trade/category
    "wrong_severity",         # wrong severity/risk
    "wrong_diy_recommendation",
    "wrong_professional_recommendation",
    "clarification_not_useful",
    "scope_incorrect",
    "steps_unclear",          # diy/verify_first specific
    "care_not_useful",        # care specific
    "other",
}

# Safety issue categories — selecting any of these auto-flags the feedback as
# safety-relevant (in addition to mode-based auto-flagging in the service).
SAFETY_CATEGORIES = {
    "missed_emergency",
    "incorrect_emergency",
    "missed_hazard",
    "unsafe_recommendation",
    "escalation_failure",
}


class FeedbackCreate(BaseModel):
    """Body for POST /cases/{case_id}/feedback."""
    analysis_id: int
    rating: Literal["helpful", "partial", "not"]
    issue_category: Optional[str] = None
    safety_category: Optional[str] = None
    comment: Optional[str] = None

    @field_validator("issue_category")
    @classmethod
    def _check_issue(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        return v if v in ISSUE_CATEGORIES else "other"

    @field_validator("safety_category")
    @classmethod
    def _check_safety(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        return v if v in SAFETY_CATEGORIES else None

    @field_validator("comment")
    @classmethod
    def _trim_comment(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v[:2000] if v else None


class FeedbackResponse(BaseModel):
    """A feedback row as returned to the admin dashboard."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: Optional[int]
    analysis_id: Optional[int]
    user_id: Optional[int]
    rating: str
    issue_category: Optional[str]
    comment: Optional[str]
    output_mode: Optional[str]
    is_safety: bool
    safety_category: Optional[str]
    status: str
    resolution: Optional[str]
    reviewer_note: Optional[str]
    created_at: object


class IncidentUpdate(BaseModel):
    """Body for PATCH /admin/feedback/{id} — safety incident review actions."""
    status: Optional[Literal["new", "reviewing", "resolved"]] = None
    resolution: Optional[Literal["fixed", "accepted_limitation", "dismissed"]] = None
    reviewer_note: Optional[str] = None

    @field_validator("reviewer_note")
    @classmethod
    def _trim_note(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v[:2000] if v else None
