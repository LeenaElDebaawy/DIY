"""
Pydantic schemas for the 3-stage analysis pipeline.

Design note — structured prediction over free-text generation
─────────────────────────────────────────────────────────────
Each stage schema acts as the output contract for one AI call.
The model is instructed to return JSON that must satisfy this schema
(the same way a classification head must output a valid probability
distribution — the output space is constrained, not free).

If the model returns JSON that fails Pydantic validation, the pipeline
retries that stage (up to retry_count) rather than propagating bad data
to the next stage.  This mirrors loss-function feedback in training —
the system self-corrects before advancing.

Stage chain:
  Stage1Output ──► Stage2Output ──► Stage3Output
                (fed as context)  (fed as context)

The final AnalysisResult is the union of all three, stored in
Analysis.structured_result as a JSONB dict.
"""

import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


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
# STAGE 1  — Image / Problem Analysis
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Output(BaseModel):
    """
    What the vision model sees and classifies from the uploaded images.

    trade_category is intentionally a clean normalised label
    ("plumbing", "electrical", "carpentry", "appliance", "structural", "other").
    This label is reused later for:
      - routing nearby-services geo-queries (no extra LLM call needed)
      - structured tool/material lookups in a future hybrid approach
    """
    problem_type: str           # short human-readable description, e.g. "Leaking pipe joint"
    trade_category: str         # normalised trade label — see docstring
    severity: str               # "low" | "medium" | "high"
    observations: list[str]     # bullet-point list of what was visually identified


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2  — Tools, Materials, Protective Gear
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
    tools: list[str]                # ESSENTIAL tools — must-have to complete the repair
    optional_tools: list[str] = [] # OPTIONAL tools — helpful but not required
    materials: list[str]            # ESSENTIAL consumables — must-have
    optional_materials: list[str] = []  # OPTIONAL materials — improve the result but not required
    protective_gear: list[str]      # safety equipment, e.g. "Rubber gloves"
    estimated_time_minutes: int     # realistic estimate for a competent DIYer


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3  — Repair Steps
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
    """
    steps: list[RepairStep]
    safety_notes: list[str]         # global notes that apply throughout the repair
    difficulty_level: str           # "beginner" | "intermediate" | "advanced"
    when_to_call_professional: str  # condition under which DIY should stop


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED RESULT
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """
    The union of all three stage outputs.
    Stored in Analysis.structured_result as a JSONB dict when the pipeline completes.
    The frontend reads this one field to render the full diagnosis.
    """
    stage_1: Stage1Output
    stage_2: Stage2Output
    stage_3: Stage3Output


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
    structured_result: Optional[dict]   # AnalysisResult as dict — populated on "complete"
    stage_outputs: Optional[dict]       # intermediate per-stage outputs — for debugging

    # Diagnostics
    validation_issues: Optional[str]    # populated on "failed"
    processing_time_ms: Optional[int]
    provider: Optional[str]             # which AI provider was used ("stub", "openai", ...)
    model: Optional[str]                # which model was used

    created_at: datetime.datetime
