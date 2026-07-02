"""
Pydantic schemas for the feedback / data-collection mechanism.

The issue-category, safety-category, and reason-chip vocabularies are defined
here as the single source of truth so collection and analytics stay aligned.
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

# ── Reason chips — multi-select shown for "partial" / "not" ratings ──────────
# Each value is stored as-is in reason_chips (JSONB list).
# Label strings are duplicated in the frontend widget and the admin display —
# keep them in sync if values ever change.
REASON_CHIPS = {
    "wrong_diagnosis",               # wrong trade / misidentified problem
    "too_cautious",                  # over-cautious routing, unnecessary referral
    "not_cautious_enough",           # missed hazard, under-cautious routing
    "not_detailed_enough",           # vague steps, missing detail
    "wrong_tools_materials",         # wrong or missing tools/materials list
    "steps_unclear",                 # unclear step ordering or wording
    "wrong_professional_recommendation",  # wrong professional type / unnecessary
    "advice_didnt_match_upload",     # poor grounding, hallucinated advice
    "repeated_question",             # unnecessary clarification
    "advice_felt_unsafe",            # user perceived the advice as unsafe
    "other",
}

# Intended mapping to judge issue tags (for future QA dashboard alignment):
#   wrong_diagnosis              → wrong_trade_category, poor_grounding
#   too_cautious                 → over_cautious_routing, useful_but_overcautious
#   not_cautious_enough          → under_cautious_routing, missing_safety_warning
#   not_detailed_enough          → vague_repair_steps, vague_tools_or_materials
#   wrong_tools_materials        → vague_tools_or_materials
#   steps_unclear                → vague_repair_steps, unclear_user_language
#   wrong_professional_recommendation → wrong_professional_type, unsupported_professional_escalation
#   advice_didnt_match_upload    → poor_grounding, unsupported_hazard_claim, hallucinated_hazard
#   repeated_question            → unnecessary_clarification
#   advice_felt_unsafe           → under_cautious_routing, missing_safety_warning

# Derived issue_category from chips (first matching chip wins).
_CHIP_TO_ISSUE: dict[str, str] = {
    "wrong_diagnosis":                    "wrong_trade",
    "too_cautious":                       "wrong_diy_recommendation",
    "not_cautious_enough":                "wrong_diy_recommendation",
    "not_detailed_enough":                "steps_unclear",
    "wrong_tools_materials":              "steps_unclear",
    "steps_unclear":                      "steps_unclear",
    "wrong_professional_recommendation":  "wrong_professional_recommendation",
    "advice_didnt_match_upload":          "wrong_trade",
    "repeated_question":                  "clarification_not_useful",
    "advice_felt_unsafe":                 "other",
    "other":                              "other",
}

# Chips that also imply a safety concern.
_CHIP_TO_SAFETY: dict[str, str] = {
    "not_cautious_enough": "missed_hazard",
    "advice_felt_unsafe":  "unsafe_recommendation",
}


class FeedbackCreate(BaseModel):
    """Body for POST /cases/{case_id}/feedback."""
    analysis_id: int
    rating: Literal["helpful", "partial", "not"]
    issue_category: Optional[str] = None
    safety_category: Optional[str] = None
    comment: Optional[str] = None
    reason_chips: Optional[list[str]] = None

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

    @field_validator("reason_chips")
    @classmethod
    def _check_chips(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if not v:
            return None
        cleaned = [c for c in v if isinstance(c, str) and c in REASON_CHIPS]
        return cleaned[:11] or None  # max 11 (one per defined chip)


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
    reason_chips: Optional[list[str]] = None
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
