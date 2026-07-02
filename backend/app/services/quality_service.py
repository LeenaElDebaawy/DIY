"""
Quality dashboard service — admin-only, read-only.

Joins completed analyses, judge evaluations (stage_outputs["judge"]), and user
feedback in a single async pass and returns a comprehensive summary dict for
the /admin/quality dashboard.

No mutations. No changes to judge scoring or pipeline logic.
"""
from collections import Counter, defaultdict
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Analysis, Feedback, Upload

# ── Human-readable labels ─────────────────────────────────────────────────────

_TAG_LABELS: dict[str, str] = {
    "over_cautious_routing":                           "Over-cautious routing",
    "under_cautious_routing":                          "Under-cautious routing",
    "missing_safety_warning":                          "Missing safety warning",
    "hallucinated_hazard":                             "Hallucinated hazard",
    "vague_repair_steps":                              "Vague repair steps",
    "vague_tools_or_materials":                        "Vague tools/materials",
    "poor_grounding":                                  "Poor grounding",
    "unnecessary_clarification":                       "Unnecessary clarification",
    "missing_clarification":                           "Missing clarification",
    "unclear_user_language":                           "Unclear language",
    "wrong_trade_category":                            "Wrong trade category",
    "weak_step_ordering":                              "Weak step ordering",
    "emergency_wording_could_be_stronger":             "Emergency wording too weak",
    "emergency_repair_steps_inappropriately_expected": "Repair steps wrongly expected",
    "unsupported_professional_escalation":             "Unsupported escalation",
    "unsupported_hazard_claim":                        "Unsupported hazard claim",
    "wrong_professional_type":                         "Wrong professional type",
    "over_specific_urgency":                           "Urgency over-specified",
    "useful_but_overcautious":                         "Useful but overcautious",
    "evidence_uncertainty_not_explained":              "Uncertainty not explained",
    "correct_but_unclear_reasoning":                   "Correct but unclear reasoning",
    "hallucinate_hazard":                              "Hallucinated hazard",
}

_CHIP_LABELS: dict[str, str] = {
    "wrong_diagnosis":                   "Wrong diagnosis",
    "too_cautious":                      "Too cautious",
    "not_cautious_enough":               "Not cautious enough",
    "not_detailed_enough":               "Not detailed enough",
    "wrong_tools_materials":             "Wrong tools/materials",
    "steps_unclear":                     "Steps unclear",
    "wrong_professional_recommendation": "Wrong professional recommendation",
    "advice_didnt_match_upload":         "Advice didn't match upload",
    "repeated_question":                 "Repeated a question I answered",
    "advice_felt_unsafe":                "Advice felt unsafe",
    "other":                             "Other",
}

# ── Alignment ─────────────────────────────────────────────────────────────────

_SAFETY_RISK_TAGS = frozenset({
    "hallucinated_hazard", "unsupported_hazard_claim",
    "missing_safety_warning", "under_cautious_routing",
})


def _compute_alignment(verdict: Optional[str], rating: Optional[str]) -> str:
    if verdict == "unsafe" and rating == "helpful":
        return "safety_override"
    if verdict in ("needs_review", "unsafe") and rating in ("partial", "not"):
        return "confirmed_issue"
    if verdict == "pass" and rating in ("not", "partial"):
        return "judge_calibration_issue"
    if verdict == "pass" and rating == "helpful":
        return "likely_good"
    if rating in ("partial", "not") and not verdict:
        return "needs_judge"
    if verdict and not rating:
        return "monitor"
    return "no_signal"


def _compute_priority(
    verdict: Optional[str],
    rating: Optional[str],
    alignment: str,
    tags: list[str],
) -> str:
    if verdict == "unsafe":
        return "high"
    if alignment == "safety_override":
        return "high"
    if alignment == "judge_calibration_issue" and rating == "not":
        return "high"
    if any(t in _SAFETY_RISK_TAGS for t in tags):
        return "high"
    if alignment == "confirmed_issue":
        return "medium"
    if rating in ("partial", "not"):
        return "medium"
    if verdict == "needs_review":
        return "medium"
    return "low"


_SUGGESTED_ACTION: dict[str, Optional[str]] = {
    "safety_override":          "Review urgently — user said Helpful but judge flagged Unsafe",
    "confirmed_issue":          "Both judge and user confirm issue — investigate this case",
    "judge_calibration_issue":  "User said No but judge passed — check judge calibration",
    "needs_judge":              "Run judge to evaluate this case",
    "monitor":                  "Monitor — judge flagged but no user complaint yet",
    "likely_good":              None,
    "no_signal":                None,
}


def _priority_reason(
    verdict: Optional[str],
    rating: Optional[str],
    alignment: str,
    tags: list[str],
) -> str:
    if verdict == "unsafe":
        return "Judge marked unsafe"
    if alignment == "safety_override":
        return "User said Helpful but judge flagged Unsafe"
    if alignment == "judge_calibration_issue" and rating == "not":
        return "User said No but judge passed"
    if "hallucinated_hazard" in tags or "hallucinate_hazard" in tags:
        return "Hallucinated hazard tag"
    if "missing_safety_warning" in tags:
        return "Missing safety warning tag"
    if "unsupported_hazard_claim" in tags:
        return "Unsupported hazard claim tag"
    if "under_cautious_routing" in tags:
        return "Under-cautious routing tag"
    if alignment == "confirmed_issue":
        return "User and judge both flagged an issue"
    if rating in ("partial", "not"):
        return "Negative user feedback"
    if verdict == "needs_review":
        return "Judge needs review"
    if alignment == "monitor":
        return "Monitoring only"
    return "No judge or feedback signal yet"

# ── Improvement suggestion rules ──────────────────────────────────────────────

_IMPROVEMENT_RULES: list[dict] = [
    {
        "judge_tags": {"wrong_professional_type", "unsupported_professional_escalation"},
        "user_chips": {"wrong_professional_recommendation"},
        "suggestion": "Strengthen professional-type selection in Stage 3 assessment prompt.",
        "pipeline":   "Stage 3 / assessment prompt",
    },
    {
        "judge_tags": {"vague_tools_or_materials"},
        "user_chips": {"wrong_tools_materials", "not_detailed_enough"},
        "suggestion": "Add tool specificity guidance (brand, type, size) to Stage 3 DIY prompt.",
        "pipeline":   "Stage 3 / DIY prompt — tools section",
    },
    {
        "judge_tags": {"poor_grounding", "unsupported_hazard_claim", "hallucinated_hazard"},
        "user_chips": {"advice_didnt_match_upload", "wrong_diagnosis"},
        "suggestion": "Improve Stage 1→Stage 3 evidence linking; enforce claim grounding.",
        "pipeline":   "Stage 1 observations → Stage 3 reasoning chain",
    },
    {
        "judge_tags": {"vague_repair_steps", "weak_step_ordering"},
        "user_chips": {"steps_unclear", "not_detailed_enough"},
        "suggestion": "Add step-ordering and completeness rules to Stage 3 DIY prompt.",
        "pipeline":   "Stage 3 / DIY prompt — steps section",
    },
    {
        "judge_tags": {"unnecessary_clarification"},
        "user_chips": {"repeated_question"},
        "suggestion": "Improve clarification deduplication — avoid re-asking context already in the upload.",
        "pipeline":   "Clarification prompt / context extraction",
    },
    {
        "judge_tags": {"over_cautious_routing", "useful_but_overcautious"},
        "user_chips": {"too_cautious"},
        "suggestion": "Review routing thresholds — may be over-escalating medium-risk cases.",
        "pipeline":   "Routing logic / Stage 1 risk_level mapping",
    },
    {
        "judge_tags": {"under_cautious_routing", "missing_safety_warning"},
        "user_chips": {"not_cautious_enough", "advice_felt_unsafe"},
        "suggestion": "Strengthen safety escalation rules for borderline risk_level cases.",
        "pipeline":   "Routing logic / safety escalation thresholds",
    },
    {
        "judge_tags": {"emergency_wording_could_be_stronger"},
        "user_chips": set(),
        "suggestion": "Add UK emergency number (999) explicitly to active-emergency output template.",
        "pipeline":   "Active emergency output template",
    },
]


def _build_suggestions(
    tag_counts: Counter,
    chip_counts: Counter,
    tag_case_ids: dict[str, list[int]],
    chip_case_ids: dict[str, list[int]],
) -> list[dict]:
    suggestions = []
    for rule in _IMPROVEMENT_RULES:
        judge_hits = [(t, tag_counts[t]) for t in rule["judge_tags"] if tag_counts[t] > 0]
        user_hits  = [(c, chip_counts[c]) for c in rule["user_chips"] if chip_counts[c] > 0]
        if not judge_hits and not user_hits:
            continue
        total_j = sum(n for _, n in judge_hits)
        total_u = sum(n for _, n in user_hits)
        signal  = "both" if (judge_hits and user_hits) else ("judge" if judge_hits else "user")
        ex_ids: set[int] = set()
        for t, _ in judge_hits:
            ex_ids.update(tag_case_ids.get(t, [])[:3])
        for c, _ in user_hits:
            ex_ids.update(chip_case_ids.get(c, [])[:3])
        suggestions.append({
            "trigger_issues":    [t for t, _ in judge_hits] + [c for c, _ in user_hits],
            "total_occurrences": total_j + total_u,
            "judge_occurrences": total_j,
            "user_occurrences":  total_u,
            "signal":            signal,
            "suggestion":        rule["suggestion"],
            "pipeline_area":     rule["pipeline"],
            "example_case_ids":  sorted(ex_ids)[:5],
        })
    suggestions.sort(key=lambda s: -s["total_occurrences"])
    return suggestions


# ── Main aggregation ──────────────────────────────────────────────────────────

async def get_summary(db: AsyncSession) -> dict:
    """
    Single-pass join of analyses, judge results, and feedback.
    Returns everything the dashboard needs in one response.
    Capped at the 200 most-recent completed sessions for demo scale.
    """
    # One completed analysis per session — the latest one
    subq = (
        select(Analysis.session_id, func.max(Analysis.id).label("max_id"))
        .where(Analysis.status == "complete")
        .where(Analysis.session_id.is_not(None))
        .group_by(Analysis.session_id)
        .subquery()
    )
    analyses = list((await db.execute(
        select(Analysis)
        .join(subq, Analysis.id == subq.c.max_id)
        .order_by(Analysis.id.desc())
        .limit(200)
    )).scalars().all())

    if not analyses:
        return _empty_summary()

    session_ids = [a.session_id for a in analyses]

    # Most-recent feedback per session
    all_fb = list((await db.execute(
        select(Feedback)
        .where(Feedback.session_id.in_(session_ids))
        .order_by(Feedback.created_at.desc())
    )).scalars().all())
    fb_by_session: dict[int, Feedback] = {}
    for fb in all_fb:
        if fb.session_id not in fb_by_session:
            fb_by_session[fb.session_id] = fb

    # Source type — video wins if any upload for this session is video
    upload_rows = list((await db.execute(
        select(Upload.session_id, Upload.source_type)
        .where(Upload.session_id.in_(session_ids))
    )).all())
    src_by_session: dict[int, str] = {}
    for sid, stype in upload_rows:
        if stype == "video" or sid not in src_by_session:
            src_by_session[sid] = stype

    # ── Build per-case rows ───────────────────────────────────────────────────
    cases: list[dict] = []
    tag_counter:  Counter = Counter()
    chip_counter: Counter = Counter()
    tag_case_ids:  dict[str, list[int]] = defaultdict(list)
    chip_case_ids: dict[str, list[int]] = defaultdict(list)

    for a in analyses:
        sr    = a.structured_result or {}
        s1    = sr.get("stage_1") or {}
        judge = (a.stage_outputs or {}).get("judge") or {}
        fb    = fb_by_session.get(a.session_id)

        verdict   = judge.get("final_verdict") if judge else None
        score     = judge.get("overall_score")  if judge else None
        tags: list[str] = judge.get("issue_tags") or [] if judge else []
        j_summary = (judge.get("qa_summary") or "")[:140] if judge else None

        rating  = fb.rating               if fb else None
        chips   = (fb.reason_chips or []) if fb else []
        comment = ((fb.comment or "")[:120]) if fb else None

        align = _compute_alignment(verdict, rating)
        prio  = _compute_priority(verdict, rating, align, tags)
        case_id = a.session_id

        for t in tags:
            tag_counter[t] += 1
            if len(tag_case_ids[t]) < 5:
                tag_case_ids[t].append(case_id)
        for c in chips:
            chip_counter[c] += 1
            if len(chip_case_ids[c]) < 5:
                chip_case_ids[c].append(case_id)

        cases.append({
            "case_id":          case_id,
            "analysis_id":      a.id,
            "problem_type":     s1.get("problem_type"),
            "output_mode":      sr.get("output_mode"),
            "rubric_mode":      (judge.get("rubric_mode") if judge else None),
            "trade_category":   s1.get("trade_category"),
            "source_type":      src_by_session.get(case_id),
            "confidence":       s1.get("confidence"),
            "judge_verdict":    verdict,
            "judge_score":      int(score) if score is not None else None,
            "judge_tags":       tags,
            "judge_summary":    j_summary,
            "fb_rating":        rating,
            "fb_chips":         chips,
            "fb_comment":       comment,
            "alignment":        align,
            "priority":         prio,
            "needs_attention":  (
                prio in ("high", "medium")
                and align not in ("no_signal", "likely_good")
            ),
            "suggested_action": _SUGGESTED_ACTION.get(align),
            "priority_reason":  _priority_reason(verdict, rating, align, tags),
            # Timestamps — all optional; old rows without data return None
            "created_at":       a.created_at.isoformat() if a.created_at else None,
            "completed_at":     None,  # Analysis model has no updated_at
            "judged_at":        (judge.get("judged_at") if judge else None),
            "fb_created_at":    (fb.created_at.isoformat() if fb and fb.created_at else None),
            "case_url":         f"/cases/{case_id}",
        })

    # ── Aggregates ────────────────────────────────────────────────────────────
    total_judged   = sum(1 for c in cases if c["judge_verdict"])
    total_feedback = sum(1 for c in cases if c["fb_rating"])
    scores = [c["judge_score"] for c in cases if c["judge_score"] is not None]
    avg_score = round(sum(scores) / len(scores)) if scores else None

    needs_attention = sorted(
        [c for c in cases if c["needs_attention"]],
        key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(c["priority"], 3),
    )

    top_judge_tags = [
        {
            "value": t,
            "label": _TAG_LABELS.get(t, t.replace("_", " ")),
            "count": n,
            "example_case_ids": tag_case_ids[t],
        }
        for t, n in tag_counter.most_common(15)
    ]
    top_user_chips = [
        {
            "value": c,
            "label": _CHIP_LABELS.get(c, c.replace("_", " ")),
            "count": n,
            "example_case_ids": chip_case_ids[c],
        }
        for c, n in chip_counter.most_common(11)
    ]

    # Combined recurring-issue view (judge tags + user chips merged by key)
    all_keys = set(tag_counter) | set(chip_counter)
    recurring: list[dict] = []
    for key in sorted(all_keys, key=lambda k: -(tag_counter[k] + chip_counter[k])):
        j_n = tag_counter[key]
        u_n = chip_counter[key]
        if j_n + u_n == 0:
            continue
        signal = "both" if (j_n > 0 and u_n > 0) else ("judge" if j_n > 0 else "user")
        label  = _TAG_LABELS.get(key) or _CHIP_LABELS.get(key) or key.replace("_", " ")
        ex_ids = list(dict.fromkeys(
            tag_case_ids.get(key, []) + chip_case_ids.get(key, [])
        ))[:5]
        recurring.append({
            "issue":            key,
            "label":            label,
            "judge_count":      j_n,
            "user_count":       u_n,
            "total":            j_n + u_n,
            "signal":           signal,
            "example_case_ids": ex_ids,
        })

    return {
        "cases":            cases,
        "needs_attention":  needs_attention[:30],
        "totals": {
            "total_cases":     len(cases),
            "total_judged":    total_judged,
            "total_feedback":  total_feedback,
            "avg_judge_score": avg_score,
            "needs_attention": len(needs_attention),
        },
        "verdict_counts":          dict(Counter(c["judge_verdict"] for c in cases if c["judge_verdict"])),
        "rating_counts":           dict(Counter(c["fb_rating"] for c in cases if c["fb_rating"])),
        "alignment_counts":        dict(Counter(c["alignment"] for c in cases)),
        "priority_counts":         dict(Counter(c["priority"] for c in cases)),
        "top_judge_tags":          top_judge_tags,
        "top_user_chips":          top_user_chips,
        "recurring_issues":        recurring[:20],
        "improvement_suggestions": _build_suggestions(
            tag_counter, chip_counter, tag_case_ids, chip_case_ids
        ),
    }


def _empty_summary() -> dict:
    return {
        "cases": [], "needs_attention": [],
        "totals": {
            "total_cases": 0, "total_judged": 0, "total_feedback": 0,
            "avg_judge_score": None, "needs_attention": 0,
        },
        "verdict_counts": {}, "rating_counts": {}, "alignment_counts": {},
        "priority_counts": {}, "top_judge_tags": [], "top_user_chips": [],
        "recurring_issues": [], "improvement_suggestions": [],
    }
