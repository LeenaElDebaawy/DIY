"""
Pydantic schemas for the 3-stage analysis pipeline.

Design note, structured prediction over free-text generation
─────────────────────────────────────────────────────────────
Each stage schema acts as the output contract for one AI call.
The model is instructed to return JSON that must satisfy this schema
(the same way a classification head must output a valid probability
distribution, the output space is constrained, not free).

If the model returns JSON that fails Pydantic validation, the pipeline
retries that stage (up to retry_count) rather than propagating bad data
to the next stage.  This mirrors loss-function feedback in training -
the system self-corrects before advancing.

Stage chain:
  Stage1Output ──► Stage2Output ──► Stage3Output
                (fed as context)  (fed as context)

The final AnalysisResult is the union of all three, stored in
Analysis.structured_result as a JSONB dict.
"""

import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER REQUEST
# ─────────────────────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    """
    Optional body for POST /cases/{case_id}/analyse.

    text_context carries the user's typed note from the quick-start form.
    The Stage 1 vision model receives both the image(s) and this text so
    it has two evidence streams for its classification.
    """

    text_context: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 , Image / Problem Analysis
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Output(BaseModel):
    """
    What the vision model sees and classifies from the uploaded images.

    trade_category is intentionally a clean normalised label from the fixed
    vocabulary below.  This label is reused later for:
      - routing nearby-services geo-queries (no extra LLM call needed)
      - structured tool/material lookups in a future hybrid approach
      - frontend filtering in My Cases

    Allowed trade_category values:
      plumbing              - pipes, taps, drains, radiator leaks, water fittings
      electrical            - wiring, sockets, switches, consumer units
      appliance             - washing machine, dishwasher, boiler, oven
      carpentry/joinery     - wooden frames, doors, sashes, hinges, latches, handles
      glazing               - cracked glass, failed double-glazing (condensation between panes)
      sealing/weatherproofing - draughts, failed silicone seals, water ingress around frames
      plastering/decorating - cracked/missing plaster, peeling paint, damp patches on walls
      structural            - load-bearing walls, foundations, lintels, masonry cracks, roofing
      hvac                  - boiler controls, central heating, air conditioning, ventilation
      general/other         - anything that does not fit the above

    confidence is a self-rated integer 0-100.  Values at or below _LOW_CONFIDENCE
    cause the pipeline to stop and ask the user for a clearer photo or description
    instead of generating potentially wrong repair steps.
    """
    problem_type: str           # short human-readable description, e.g. "Leaking pipe joint"
    trade_category: str         # normalised trade label, see docstring
    severity: str               # "low" | "medium" | "high"
    observations: list[str]     # bullet-point list of what was visually identified
    confidence: int = 75        # 0-100; below LOW_CONFIDENCE_THRESHOLD → clarification gate
    needs_clarification: bool = False  # true when the image/text is not enough to diagnose
    clarification_request: str = ""    # user-facing message when needs_clarification is true
    confidence_reason: str = ""        # one-two sentence explanation of why this score was given
    visual_evidence: list[str] = []    # every distinct object/feature clearly seen in the image
    uncertainty_notes: str = ""        # what could not be confirmed; empty string if everything is clear

    # ── Model-reasoned risk signals (AI-led design) ──────────────────────────
    # These are the PRIMARY drivers of how cautious the pipeline becomes. The
    # model, not a category lookup, decides them by reasoning about what it sees.
    # The pipeline reacts to THESE signals (ask / assessment-first / verify-first),
    # so the behaviour generalises to any subject, including ones with no matching
    # trade_category (e.g. a crash-damaged vehicle). Category is only a thin
    # one-way safety net layered on top, it may raise these but never lower them.
    # Defaults are deliberately cautious-neutral (medium / conditional) so a model
    # that omits a field never reads as "definitely safe".
    risk_level: str = "medium"                 # low | medium | high | critical, danger if mishandled or acted on from a photo alone
    hidden_damage_risk: str = "medium"         # low | medium | high, chance important damage is NOT visible in the photo
    diy_suitability: str = "conditional"       # suitable | conditional | not_recommended, for a competent DIY user (not a pro)
    professional_assessment_recommended: bool = False  # true, an in-person professional should assess before action
    safety_concerns: list[str] = []            # specific hazards seen (e.g. "electrocution", "structural collapse", "airbag/fuel")
    clarifying_questions: list[str] = []       # REQUIRED questions — answers block auto-reanalysis
    optional_questions:   list[str] = []       # helpful extras — do not block auto-reanalysis (max 1)

    @field_validator("risk_level", mode="before")
    @classmethod
    def _normalise_risk_level(cls, v: object) -> str:
        """Map to low/medium/high/critical; unknown/empty -> cautious 'medium'."""
        raw = str(v).strip().lower() if v is not None else ""
        if raw in ("low", "medium", "high", "critical"):
            return raw
        # tolerate a few synonyms the model may emit
        if raw in ("severe", "very high", "extreme", "danger", "dangerous"):
            return "critical"
        if raw in ("moderate", "med", "mid"):
            return "medium"
        if raw in ("none", "minimal", "very low"):
            return "low"
        return "medium"

    @field_validator("hidden_damage_risk", mode="before")
    @classmethod
    def _normalise_hidden_damage_risk(cls, v: object) -> str:
        """Map to low/medium/high; unknown/empty -> cautious 'medium'."""
        raw = str(v).strip().lower() if v is not None else ""
        if raw in ("low", "medium", "high"):
            return raw
        if raw in ("critical", "severe", "very high", "extreme"):
            return "high"
        if raw in ("moderate", "med", "mid"):
            return "medium"
        if raw in ("none", "minimal", "very low"):
            return "low"
        return "medium"

    @field_validator("diy_suitability", mode="before")
    @classmethod
    def _normalise_diy_suitability(cls, v: object) -> str:
        """Map to suitable/conditional/not_recommended; unknown -> cautious 'conditional'."""
        raw = str(v).strip().lower().replace("-", "_").replace(" ", "_") if v is not None else ""
        if raw in ("suitable", "conditional", "not_recommended"):
            return raw
        if raw in ("yes", "diy", "diy_suitable", "easy", "recommended"):
            return "suitable"
        if raw in ("no", "not_diy", "not_suitable", "unsuitable", "professional", "pro_only"):
            return "not_recommended"
        return "conditional"

    @field_validator("professional_assessment_recommended", mode="before")
    @classmethod
    def _coerce_professional_assessment(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y", "recommended")
        return bool(v)

    @field_validator("safety_concerns", "clarifying_questions", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: object) -> list[str]:
        """Accept a list of strings, a list of dicts (extract a string value), or a
        single string; always return list[str]. Empty/None -> []."""
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, str) and val.strip():
                        result.append(val.strip())
                        break
        return result

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> int:
        """Accept string digits or floats; clamp to 0-100."""
        try:
            val = int(float(str(v)))
        except (TypeError, ValueError):
            return 75
        return max(0, min(100, val))

    @field_validator("needs_clarification", mode="before")
    @classmethod
    def _coerce_needs_clarification(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @field_validator("trade_category", mode="before")
    @classmethod
    def _normalise_trade_category(cls, v: object) -> str:
        """
        Normalise model output to the current 10-category vocabulary.

        Old analyses stored in JSONB may carry legacy labels such as
        "carpentry", "painting", "roofing", or "other".  New analyses
        from models that ignore the prompt may return minor variants.
        This validator maps all of them to the canonical label so the
        rest of the pipeline and the frontend can rely on a stable set.
        """
        _CANONICAL = {
            # Current vocabulary (pass through unchanged)
            "plumbing":                 "plumbing",
            "electrical":               "electrical",
            "appliance":                "appliance",
            "carpentry/joinery":        "carpentry/joinery",
            "glazing":                  "glazing",
            "sealing/weatherproofing":  "sealing/weatherproofing",
            "plastering/decorating":    "plastering/decorating",
            "stone/masonry":            "stone/masonry",
            "structural":               "structural",
            "hvac":                     "hvac",
            "general/other":            "general/other",
            # stone/masonry synonyms (so a model that says just "marble" or "tile"
            # still routes to the correct trade profile)
            "stone":                    "stone/masonry",
            "masonry":                  "stone/masonry",
            "marble":                   "stone/masonry",
            "granite":                  "stone/masonry",
            "tile":                     "stone/masonry",
            "tiling":                   "stone/masonry",
            "tiles":                    "stone/masonry",
            "stonework":                "stone/masonry",
            "brickwork":                "stone/masonry",
            # Legacy labels from the old 8-category prompt
            "carpentry":                "carpentry/joinery",
            "joinery":                  "carpentry/joinery",
            "painting":                 "plastering/decorating",
            "decorating":               "plastering/decorating",
            "plastering":               "plastering/decorating",
            "roofing":                  "structural",
            "general":                  "general/other",
            "other":                    "general/other",
            # Plausible model variants
            "sealing":                  "sealing/weatherproofing",
            "weatherproofing":          "sealing/weatherproofing",
            "heating":                  "hvac",
            "ventilation":              "hvac",
            "air conditioning":         "hvac",
            "mechanical":               "hvac",
            "glass":                    "glazing",
            "windows":                  "glazing",
        }
        raw = str(v).strip().lower() if v else ""
        return _CANONICAL.get(raw, "general/other")

    @field_validator("clarification_request", mode="before")
    @classmethod
    def _coerce_clarification_request(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v)

    @field_validator("confidence_reason", mode="before")
    @classmethod
    def _coerce_confidence_reason(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v)

    @field_validator("visual_evidence", mode="before")
    @classmethod
    def _coerce_visual_evidence(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, str) and val.strip():
                        result.append(val.strip())
                        break
        return result

    @field_validator("uncertainty_notes", mode="before")
    @classmethod
    def _coerce_uncertainty_notes(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 , Tools, Materials, Protective Gear
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Output(BaseModel):
    """
    What the user will need before starting the repair.

    Hybrid note: in a production system, tools and materials for common
    trade_category + problem_type combinations could be served from a
    structured lookup table rather than an LLM call.  The pipeline is
    designed so _stage_2_resources() can be swapped for a DB/rules query
    without changing the Stage 1 or Stage 3 functions.
    """
    tools: list[str]                # ESSENTIAL tools, must-have to complete the repair
    optional_tools: list[str] = [] # OPTIONAL tools, helpful but not required
    materials: list[str]            # ESSENTIAL consumables, must-have
    optional_materials: list[str] = []  # OPTIONAL materials, improve the result but not required
    protective_gear: list[str]      # safety equipment, e.g. "Rubber gloves"
    estimated_time_minutes: int     # realistic estimate for a competent DIYer


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 , Repair Steps
# ─────────────────────────────────────────────────────────────────────────────

class RepairStep(BaseModel):
    """One step in the repair sequence."""
    step_number: int
    title: str                      # short action title, e.g. "Shut off the water supply"
    description: str                # full instruction paragraph
    warning: Optional[str] = None   # per-step safety note, only when relevant


class Stage3Output(BaseModel):
    """
    The ordered repair procedure, generated from Stage 1 + Stage 2 context.

    difficulty_level uses a fixed vocabulary so the frontend can render it
    with a consistent badge rather than parsing free text.

    diy_recommended and professional_type provide structured risk assessment:
      - diy_recommended: False means the repair is not safe for a DIYer to attempt
      - professional_type: the specific trade to call (e.g. "licensed electrician")
      - when_to_call_professional: plain-English explanation of the risk conditions

    All three fields are always returned. For safe DIY repairs, diy_recommended
    is True and professional_type is null.

    Defaults ensure old stored JSONB analyses without these fields do not crash.
    """
    steps: list[RepairStep]
    safety_notes: list[str]              # global notes that apply throughout the repair
    difficulty_level: str                # "beginner" | "intermediate" | "advanced"
    when_to_call_professional: str = ""  # plain-English condition under which DIY should stop; empty string if fully DIY safe
    diy_recommended: bool = True         # False if repair is unsafe for a DIYer
    professional_type: Optional[str] = None  # e.g. "licensed electrician", "gas engineer"

    @field_validator("safety_notes", mode="before")
    @classmethod
    def _coerce_safety_notes(cls, v: object) -> list[str]:
        """
        Llama 3.1 sometimes returns safety_notes as a list of dicts
        e.g. [{"note": "Wear gloves."}] instead of ["Wear gloves."].
        This validator extracts the string value from any dict entry so
        Pydantic can proceed without a ValidationError.
        Entries that are already strings pass through unchanged.
        """
        if not isinstance(v, list):
            return v  # let Pydantic raise the type error normally
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # grab the first string value found in the dict
                for val in item.values():
                    if isinstance(val, str) and val.strip():
                        result.append(val.strip())
                        break
        return result

    @field_validator("when_to_call_professional", mode="before")
    @classmethod
    def _coerce_when_to_call(cls, v: object) -> str:
        """Return empty string instead of None so the field stays a str."""
        if v is None:
            return ""
        return v


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED RESULT
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ASSESSMENT-FIRST OUTPUT  (produced instead of Stage 2 + Stage 3 when the
# model determines that professional in-person assessment is needed before any
# repair work.  No tools, no materials, no repair steps — only stabilisation,
# safety, escalation, and information gathering.)
# ─────────────────────────────────────────────────────────────────────────────

class AssessmentOutput(BaseModel):
    """
    Produced by Stage A (_stage_A_assessment) when output_mode == 'assessment_first'.

    Contains ONLY what a careful assessor communicates when a professional
    must inspect before any repair is attempted:
      - what the user should do RIGHT NOW to make the situation safe
      - what they must NOT do (actions that could worsen danger or damage)
      - why professional assessment is required (plain English, specific)
      - what information/evidence to gather before the professional arrives
      - who to call and how urgently

    Never contains repair steps, tool lists, material lists, or DIY guidance.
    Defaults ensure old analyses that pre-date this field still load cleanly.
    """
    immediate_actions: list[str] = []   # e.g. "Isolate the water at the stopcock"
    do_not_do: list[str] = []           # e.g. "Do not attempt to remove ceiling panels"
    why_assessment_needed: str = ""     # specific risk explanation for this scene
    information_needed: list[str] = []  # e.g. "Photograph the leak source from above"
    professional_type: str = ""         # e.g. "CIPHE-registered plumber"
    professional_urgency: str = ""      # "immediately" | "within 24 hours" | "when convenient"


# ─────────────────────────────────────────────────────────────────────────────
# CARE OUTPUT  (produced instead of Stage 2 + Stage 3 when the model determines
# this is a low-risk maintenance/care task rather than a repair execution task.
# Examples: worn leather shoe, tarnished jewellery, stained fabric, dull wood
# finish. No safety notes, no professional guidance, no repair steps — only
# practical, material-specific care advice.)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CLARIFICATION OUTPUT  (produced instead of Stage 2 + Stage 3 when Stage 1
# identifies meaningful unknowns that would materially change the diagnosis,
# DIY suitability, or repair method. Surfaces targeted questions to the user
# rather than guessing at an answer with insufficient evidence.)
# ─────────────────────────────────────────────────────────────────────────────

class ClarificationOutput(BaseModel):
    """
    Produced when output_mode == 'clarification'.

    Derived DETERMINISTICALLY from Stage 1 data — no extra model call.
    Contains what the model knows, what is unclear, the questions to resolve
    the uncertainty, and why the answers change the advice.

    Defaults ensure old analyses that pre-date this field still load cleanly.
    """
    what_is_known: str = ""              # plain summary of what was diagnosed
    what_is_unclear: list[str] = []      # specific unknowns that change the advice
    clarifying_questions: list[str] = [] # REQUIRED questions — block auto-reanalysis until answered
    optional_questions:   list[str] = [] # helpful extras — shown separately, do not block auto-reanalysis
    why_it_matters: str = ""             # why the answers change the recommendation
    # True ONLY for the bounded-clarification fallback: the pipeline hit
    # MAX_CLARIFICATION_ROUNDS and is no longer asking questions.  The frontend
    # renders a distinct "could not reach a confident assessment" banner instead
    # of a normal needs-answers card.  Default False keeps old analyses valid.
    inconclusive: bool = False


class CareOutput(BaseModel):
    """
    Produced by Stage C (_stage_C_care) when output_mode == 'care'.

    Contains practical, material-specific care and maintenance advice for the
    object shown. Structured as:
      - condition_summary: plain-English assessment of what was seen
      - recommendations: ordered list of specific care actions (2-6 items)
      - what_to_avoid: things that would damage or worsen the item (0-3 items)
      - expected_outcome: what the item should look/feel like after following advice

    Never contains repair steps, tool lists, protective gear requirements,
    safety warnings, or professional guidance — those imply risk that is absent
    from care-mode subjects.
    Defaults ensure old analyses that pre-date this field still load cleanly.
    """
    condition_summary: str = ""         # e.g. "Light surface discolouration on worn leather upper"
    recommendations: list[str] = []     # specific ordered care actions
    what_to_avoid: list[str] = []       # material-specific pitfalls
    expected_outcome: str = ""          # what good looks like after following the advice


class AnalysisResult(BaseModel):
    """
    The output of the analysis pipeline.

    output_mode drives both what was generated and how the frontend renders it:
      diy            - full Stage 1 + Stage 2 + Stage 3 DIY plan
      verify_first   - full plan but verification steps are first
      clarification  - Stage 1 only + clarifying questions (Step 4)
      assessment_first - Stage 1 + AssessmentOutput, NO Stage 2/Stage 3
      care           - Stage 1 + care recommendations (Step 3)

    stage_2 and stage_3 are None for assessment_first and clarification modes.
    assessment is None unless output_mode == 'assessment_first'.

    Backward compatibility: old analyses stored in JSONB have no output_mode
    field. The frontend defaults to 'diy' rendering when the field is absent,
    which matches the behaviour before this field was introduced.
    """
    output_mode: str = "diy"           # see docstring above
    stage_1: Stage1Output
    stage_2: Optional[Stage2Output] = None
    stage_3: Optional[Stage3Output] = None
    assessment: Optional[AssessmentOutput] = None
    care: Optional[CareOutput] = None
    clarification: Optional[ClarificationOutput] = None


# ─────────────────────────────────────────────────────────────────────────────
# API RESPONSE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """
    Shape returned by GET /api/v1/cases/{case_id}/analysis.

    The frontend polls this endpoint while status == "running".
    pipeline_stage updates after each stage completes so the UI can show
    granular progress ("Identifying problem…", "Listing tools…", "Writing steps…").
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int                     # the case this analysis belongs to

    # Lifecycle state
    status: str                         # "pending" | "running" | "complete" | "failed"
    pipeline_stage: Optional[str]       # "stage_1_running" | "stage_1_complete" | ...

    # Results
    structured_result: Optional[dict]   # AnalysisResult as dict, populated on "complete"
    stage_outputs: Optional[dict]       # intermediate per-stage outputs, for debugging

    # Diagnostics
    validation_issues: Optional[str]    # populated on "failed"
    processing_time_ms: Optional[int]
    provider: Optional[str]             # which AI provider was used ("stub", "openai", ...)
    model: Optional[str]                # which model was used

    created_at: datetime.datetime
    # The user's clarification answers passed to this re-analysis run.
    # None for first-pass analyses.  Present when the case went through at least
    # one clarification round — used by the frontend to show the "Updated after
    # clarification" badge and the collapsible Q/A history section.
    text_context: Optional[str] = None
    # Hazardous-case acknowledgement, delegated from the owning session via the
    # Analysis.hazard_ack property.  False until the user clicks "I Understand"
    # on the assessment_first acknowledgement card.
    hazard_ack: bool = False
