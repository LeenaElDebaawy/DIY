import logging
import os

from app.schemas.analysis import Stage1Output, Stage3Output

log = logging.getLogger(__name__)

# When true: if the care-mode hazard gate fires on a re-analysis (text_context
# present) and Stage 1 was already satisfied (no clarifying_questions), trust
# Stage 1's judgment and route directly to diy instead of asking another
# clarifying question.  Stage 1 read the user's context and produced no
# unknowns — the gate firing is a false positive in that situation.
#
# Disable instantly: set TRUST_STAGE1_ON_REANALYSIS=false in .env and restart
# the backend only.  Default: false (safe off; enable explicitly for demo).
_TRUST_STAGE1_ON_REANALYSIS: bool = (
    os.getenv("TRUST_STAGE1_ON_REANALYSIS", "false")
    .strip().lower() == "true"
)

# ─────────────────────────────────────────────────────────────────────────────
# AI-LED RISK REACTIONS + THIN ONE-WAY SAFETY NET
#
# Design contract (model-led, not rule-led):
#   • The MODEL decides risk: it fills risk_level, hidden_damage_risk,
#     diy_suitability, professional_assessment_recommended, safety_concerns and
#     clarifying_questions by reasoning about THIS scene (see the Stage 1 prompt).
#   • The PIPELINE only REACTS to those model signals:
#       - ask           when the model is not confident enough / wants clarifying
#       - assessment    when the model says professional assessment is needed or
#                       the job is not DIY-suitable (and it can name a hazard)
#       - verify-first  when the model flags high risk or hidden damage
#   • CATEGORY is a thin one-way SAFETY NET only. For a small set of inherently
#     dangerous trades it may RAISE the model's signals to a cautious floor when
#     the model has clearly under-called them (a known failure mode of a local
#     vision model). It can NEVER lower a signal and is never the primary driver.
#
# This generalises: a crash-damaged vehicle has no matching trade_category, yet a
# model that reads it as high risk / not DIY-suitable / hidden-damage will still
# become assessment-first, because the reaction keys on the SIGNALS, not the name.
# ─────────────────────────────────────────────────────────────────────────────

# Inherently dangerous trades where a local vision model sometimes UNDER-rates the
# risk. The net only ever escalates and supplies a concrete professional + hazard.
_SAFETY_NET_TRADES = {
    "electrical": {
        "professional": "NICEIC-registered electrician",
        "hazard": "electric shock from mains electricity",
    },
    "hvac": {
        "professional": "Gas Safe registered engineer",
        "hazard": "gas or heating-system hazard",
    },
    "structural": {
        "professional": "structural engineer",
        "hazard": "possible structural failure",
    },
}

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_DIY_ORDER = {"suitable": 0, "conditional": 1, "not_recommended": 2}

_BASE_CONFIDENCE_FLOOR = 75   # generic ask-gate; same for every trade (model-led)

# ── Care-mode hazard gate ─────────────────────────────────────────────────────
# Words that are incompatible with genuine care/maintenance subjects (shoes,
# jewellery, fabric, furniture finishes). If any appear in problem_type or
# observations when care mode would otherwise fire, the model has most likely
# misclassified a hazardous scene as a safe maintenance task.
#
# Designed to catch the demonstrated failure: an electrical sparking image
# classified as "Sparks from welding or cutting", trade=general/other, risk=low
# → routed to care → produced cleaning advice for what could be live electrics.
#
# When triggered: route to clarification (ask what the situation is) rather than
# assessment_first (which requires a named hazard the model did not provide).
# This is a one-way safety net only — it never lowers risk for a genuine care case.
_CARE_HAZARD_WORDS: frozenset[str] = frozenset({
    "spark", "sparks", "electric", "electrical", "electricity",
    "smoke", "smoking", "burning", "burn", "fire", "flame",
    "mains", "voltage", "live wire", "live conductor",
    "fuse", "arc flash", "short circuit", "short-circuit",
    "gas leak", "welding", "blowtorch", "soldering iron",
})


def _apply_safety_net(s1: Stage1Output) -> list[str]:
    """
    Thin one-way safety net (guardrail, NOT decision-maker).

    For a few inherently dangerous trades, RAISE the model's own risk signals to a
    cautious floor when it under-called them. Never lowers any signal. Mutates s1
    and returns human-readable notes (logged + stored for transparency).
    """
    tc = (s1.trade_category or "").strip().lower()
    net = _SAFETY_NET_TRADES.get(tc)
    if not net:
        return []   # not a safety-net trade → the model's signals stand unchanged

    notes: list[str] = []
    if _RISK_ORDER.get(s1.risk_level, 1) < _RISK_ORDER["high"]:
        notes.append(f"risk_level {s1.risk_level}->high")
        s1.risk_level = "high"
    if _DIY_ORDER.get(s1.diy_suitability, 1) < _DIY_ORDER["not_recommended"]:
        notes.append(f"diy_suitability {s1.diy_suitability}->not_recommended")
        s1.diy_suitability = "not_recommended"
    if not s1.professional_assessment_recommended:
        s1.professional_assessment_recommended = True
        notes.append("professional_assessment_recommended->True")
    if not s1.safety_concerns:
        s1.safety_concerns = [net["hazard"]]
        notes.append("added default safety_concern")
    return notes


def _decide_actions(s1: Stage1Output) -> dict:
    """
    Derive the pipeline's reactions PURELY from the (model-decided, possibly
    net-raised) Stage 1 signals. No category logic here, the safety net has
    already done its one-way raise upstream.

    Returns a dict describing what the pipeline should do, with a 'professional'
    hint taken from the safety-net trade when available.
    """
    diy = (s1.diy_suitability or "conditional").strip().lower()
    has_hazard = bool(s1.safety_concerns)

    # Assessment-first when the model says a professional should look first OR the
    # job is not DIY-suitable. CONSISTENCY GUARD: require a named hazard so we do
    # not act on an over-call the model could not justify (e.g. "not_recommended"
    # with empty safety_concerns). The safety net always supplies a hazard, so
    # genuinely dangerous trades are unaffected by this guard.
    assessment_first = (
        (s1.professional_assessment_recommended or diy == "not_recommended")
        and has_hazard
    )

    # Verify-first (lighter touch): the model flags real danger or that important
    # damage may be hidden, but the job can still be DIY with checks first.
    verify_first = (not assessment_first) and (
        s1.hidden_damage_risk == "high"
        or s1.risk_level in ("high", "critical")
    )

    pro_hint = _SAFETY_NET_TRADES.get(
        (s1.trade_category or "").strip().lower(), {}
    ).get("professional")

    return {
        "assessment_first": assessment_first,
        "verify_first": verify_first,
        "professional_hint": pro_hint,
    }


def _determine_output_mode(s1: Stage1Output, text_context: str | None = None) -> str:
    """
    Determine which pipeline output mode this Stage 1 result calls for.

    Priority order (highest wins):

    assessment_first
        The model (or safety net) determined that professional in-person
        assessment is needed AND named at least one concrete hazard.
        Consistency guard: a named hazard is required so we do not act on a
        vague over-call ('not_recommended' with no justification).
        Stages 2 and 3 will NOT run. A focused AssessmentOutput is produced
        instead: immediate actions, what NOT to do, why, professional guidance.

    clarification
        Meaningful unknowns exist that would materially change the diagnosis,
        risk assessment, DIY suitability, repair method, tool or material
        choice, or professional assessment decision.
        Triggered by EITHER:
          a) The model generated clarifying_questions (most direct signal), OR
          b) The model set diy_suitability='conditional' (explicitly "only do
             it after checking X") — that IS a meaningful unknown by definition.
        NOT triggered by information that is merely missing but would not change
        the recommendation (e.g. the brand of a product, the age of the item).
        Stages 2 and 3 will NOT run. A ClarificationOutput is produced.

    care
        A low-risk, low-severity subject that is maintenance or care rather
        than a repair execution task (e.g. shoe care, cleaning, preserving).
        Identified by: severity=low + risk=low + diy=suitable + no hazards +
        trade=general/other. A short care-recommendations output is produced,
        without the safety/professional/step apparatus that implies danger.

    verify_first
        The job is plausibly DIY but hidden damage risk is high, meaning the
        true extent cannot be established from the photo alone. Stages 2 and 3
        run, but verification steps come first and the output carries a clear
        "check before you commit" framing.

    diy
        Default for clear, straightforward, visible repairs where the model is
        confident about the diagnosis, method, and suitability. Full pipeline.

    This function is PURELY DETERMINISTIC given a Stage1Output. It has no side
    effects and no model calls. All five test cases must return the expected mode
    before any other pipeline stage runs.
    """
    # ── 0. Scope gate (highest priority) ──────────────────────────────────────
    # Decided by the model BEFORE any repair routing. Anything that is not an
    # in-scope repair (active emergency, a non-repair image, or too ambiguous to
    # tell) short-circuits to the terminal out_of_scope mode — no Stage 2/3/A, no
    # repair plan, no trade-professional referral. This is the deterministic
    # enforcement of the model-led scope decision: it removes the pressure that
    # forced an emergency (e.g. a stovetop fire) into a fabricated repair trade.
    # "unclear" is routed here rather than to clarification because clarification
    # is built around answering repair-specific questions about an in-scope
    # repair; an image that may not be a repair at all has no such questions, so
    # the safe minimal response is "we couldn't identify a repair, send a clearer
    # repair image" — the same out_of_scope guidance, worded for ambiguity.
    if (s1.scope or "repair") != "repair":
        return "out_of_scope"

    diy = (s1.diy_suitability or "conditional").strip().lower()
    has_hazard = bool(s1.safety_concerns)

    # ── 1. Assessment-first ───────────────────────────────────────────────────
    # diy_suitability == "not_recommended" is a HARD escalation: the model has said
    # a competent DIYer should NOT attempt this, which is fundamentally incompatible
    # with producing a DIY plan. Escalate even when no concrete hazard was named —
    # the safe failure direction is over-caution, never a contradictory full plan.
    # (Stage A produces a coherent assessment from the wider context when
    # safety_concerns is empty, so no hazard synthesis is required here.)
    if diy == "not_recommended":
        return "assessment_first"
    # professional_assessment_recommended is a direct model signal that the job
    # exceeds safe DIY scope. Escalate unconditionally — a homeowner should never
    # receive a repair plan for something the model flagged as needing a professional.
    if s1.professional_assessment_recommended:
        return "assessment_first"

    # ── 2. Clarification ──────────────────────────────────────────────────────
    # Model generated questions (most direct signal) OR said "conditional"
    # (which by definition means "there is an unknown that must be resolved
    # before you can safely proceed"). Either is a meaningful unknown.
    if s1.clarifying_questions or diy == "conditional":
        return "clarification"

    # ── 3. Verify-first ───────────────────────────────────────────────────────
    # The repair may be DIY, but either important damage may be hidden from the
    # photo (hidden_damage_risk=high) OR the model rated the job high/critical risk.
    # Either warrants verification before committing. Checked AFTER assessment-first
    # and clarification so a stronger escalation always wins. Including risk_level
    # here aligns the authoritative router with _decide_actions, which already used
    # it, closing a divergence that let high-risk cases fall through to diy.
    if s1.hidden_damage_risk == "high" or s1.risk_level in ("high", "critical"):
        return "verify_first"

    # ── 4. Care ───────────────────────────────────────────────────────────────
    # Low-risk maintenance subject: no hazards, low severity, DIY-suitable,
    # and classified as general/other (prevents real trades being mislabelled).
    # Checked after verify_first so hidden=high cannot silently produce a
    # care output with no verification warning.
    if (
        s1.severity == "low"
        and s1.risk_level == "low"
        and diy == "suitable"
        and not has_hazard
        and s1.trade_category == "general/other"
    ):
        # ── Care-mode hazard gate ─────────────────────────────────────────
        # Before returning care, verify the language is consistent with a
        # safe maintenance subject. If problem_type or observations contain
        # hazard-indicator words (sparks, fire, electric, smoke, mains…) the
        # model has likely misclassified a dangerous scene. Route to
        # clarification instead — ask the user what is happening rather than
        # confidently giving cleaning/polishing advice for a potential hazard.
        _problem_lower = (s1.problem_type or "").casefold()
        _obs_lower = " ".join(s1.observations or []).casefold()
        _combined = _problem_lower + " " + _obs_lower
        if any(word in _combined for word in _CARE_HAZARD_WORDS):
            log.info(
                "care_mode_hazard_gate_triggered",
                extra={
                    "problem_type": s1.problem_type,
                    "matched_in": "problem_type or observations",
                    "trade": s1.trade_category,
                },
            )
            # ── Re-analysis bypass (feature-flagged) ──────────────────────────
            # If text_context is present (re-analysis) and Stage 1 returned no
            # questions, Stage 1 read the user's context and was satisfied.
            # Routing to clarification here would be a false positive — trust
            # Stage 1 and go directly to diy so Stages 2 and 3 run.
            if _TRUST_STAGE1_ON_REANALYSIS and text_context and not s1.clarifying_questions:
                log.info(
                    "care_mode_hazard_gate_bypassed: text_context present and "
                    "Stage 1 satisfied (no questions) — routing to diy."
                )
                return "diy"
            # Inject a clarification question so the pipeline has something to
            # surface to the user (block 2 in _determine_output_mode fires next).
            #
            # The question is tailored to the type of hazard word that triggered
            # the gate:
            #   Electrical words (spark, mains, live wire…) → ask whether this is
            #     an electrical component, because that is the dangerous ambiguity.
            #   Fire/heat/burn words without electrical context → ask what caused
            #     the mark and what material is affected, because the answer
            #     determines whether the repair is a simple surface treatment or
            #     something that requires specialist attention.
            #
            # Using the electrical question for a surface burn mark (e.g. candle
            # mark on wooden flooring) is unanswerable and traps the case in an
            # infinite clarification loop.  Separating the two question types
            # breaks that loop while preserving the safety gate for both hazard
            # classes.
            if not s1.clarifying_questions:
                _electrical_trigger_words: frozenset[str] = frozenset({
                    "spark", "sparks", "electric", "electrical", "electricity",
                    "mains", "voltage", "live wire", "live conductor",
                    "fuse", "arc flash", "short circuit", "short-circuit",
                })
                _has_electrical_trigger = bool(
                    {w for w in _electrical_trigger_words if w in _combined}
                )
                if _has_electrical_trigger:
                    # Electrical / arcing context — the original question is
                    # correct: confirm what the scene actually is before giving
                    # any advice.
                    s1.clarifying_questions = [
                        "Can you describe what is happening in more detail? "
                        "For example, is this an electrical component, metalwork, "
                        "or something else? A clearer description will help identify "
                        "the safest next step."
                    ]
                else:
                    # Burn/heat/smoke without electrical indicators (e.g. a scorch
                    # mark on flooring, a heat stain on a worktop).  Ask for cause
                    # and surface type — the two unknowns that actually determine
                    # the repair method and material choice.
                    s1.clarifying_questions = [
                        "What caused this mark or damage — for example, was it a "
                        "candle, heat, a chemical, or friction? "
                        "And what material is the affected surface "
                        "(for example, wood, carpet, stone, or plastic)?"
                    ]
            return "clarification"
        return "care"

    # ── 5. Default ────────────────────────────────────────────────────────────
    return "diy"


def _apply_decision_to_stage3(
    s3: Stage3Output,
    decision: dict,
    s1: Stage1Output,
) -> list[str]:
    """
    Enforce the model-derived decision on the final Stage 3 output. This is a
    deterministic CONSISTENCY backstop, Stage 3 is also given the signals in its
    prompt, but a local model may forget to escalate, so we make the outcome match
    the assessment the model already produced in Stage 1.

    Mutates s3 in place. Returns notes on what changed.
    """
    notes: list[str] = []
    concerns = "; ".join(s1.safety_concerns) if s1.safety_concerns else ""

    if decision["assessment_first"]:
        pro = (s3.professional_type or "").strip() or decision.get("professional_hint") \
            or "a suitably qualified professional"
        if s3.diy_recommended:
            s3.diy_recommended = False
            notes.append("forced diy_recommended=False (model: assessment-first)")
        if not (s3.professional_type or "").strip():
            s3.professional_type = pro
            notes.append(f"set professional_type='{pro}'")
        lead = (
            "Do not attempt this as a DIY repair yet. "
            + (f"The main hazard here is {concerns}. " if concerns else "")
            + f"Make the area safe, avoid using or disturbing it, and get {pro} to "
            "assess it in person before any work."
        )
        if lead not in s3.safety_notes:
            s3.safety_notes = [lead] + list(s3.safety_notes)
            notes.append("prepended assessment-first escalation note")
        if not (s3.when_to_call_professional or "").strip():
            s3.when_to_call_professional = lead
        return notes

    if decision["verify_first"]:
        _PLACEHOLDER_TERMS: frozenset[str] = frozenset({
            "unknown issues", "unknown issue", "unknown", "none", "n/a",
            "not applicable", "not known", "unclear", "unspecified", "",
        })
        _cannot_raw = getattr(s1, "what_cannot_be_confirmed", None) or []
        if isinstance(_cannot_raw, list):
            _cannot = [
                c for c in _cannot_raw
                if c.strip().lower() not in _PLACEHOLDER_TERMS and len(c.strip()) > 8
            ]
        else:
            _cannot = []
        if _cannot:
            _parts = [c.rstrip(".") for c in _cannot]
            unknowns = "; ".join(_parts) + "."
        else:
            unknowns = (s1.uncertainty_notes or "").strip() or "the full extent of any damage"
        verify = (
            f"Before any irreversible step, check: {unknowns} "
            "Stop and consult a professional if the problem turns out to be worse than it appears."
        )
        if verify not in s3.safety_notes:
            s3.safety_notes = [verify] + list(s3.safety_notes)
            notes.append("prepended verify-first note (model: high risk / hidden damage)")
        if s3.diy_recommended and not (s3.when_to_call_professional or "").strip():
            s3.when_to_call_professional = verify
            notes.append("set when_to_call_professional from verify-first requirement")

    return notes
