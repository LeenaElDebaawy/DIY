import re

from app.schemas.analysis import Stage1Output

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 RELIABILITY PASS  (Priority 5)
#
# Qwen2.5-VL:3b has three recurring failure modes seen in the real saved cases:
#   • echoey duplicate observations   (case #54: "Soil and debris on the ground
#                                       around the pipe" listed three times)
#   • overconfidence on a weak read   (case #56: confidence 90 on a hallucinated
#                                       "cabinet"; case #57: confidence 80 while
#                                       missing a structural hole)
#   • near-duplicate evidence that pads the list without adding information
#
# This deterministic post-processor cleans the structured output BEFORE it can
# propagate into Stages 2 and 3.  It never invents content, it only de-dupes,
# trims, and reconciles the self-reported confidence with the model's own
# uncertainty.  Returns a list of human-readable notes describing what it changed
# (logged + appended to validation_issues for transparency).
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that, when they appear in uncertainty_notes, mean the model is NOT
# actually sure about the core diagnosis, so a 90% confidence is inconsistent.
_UNSURE_MARKERS = (
    "cannot determine", "can't determine", "cannot confirm", "can't confirm",
    "cannot tell", "can't tell", "unable to determine", "unable to confirm",
    "not clear whether", "unclear whether", "cannot identify", "hard to tell",
    "could be", "may be either", "multiple possible", "not certain",
)

# When the model admits it cannot establish the CAUSE/root of the problem,
# confidence is capped here so the UI never shows near-certainty on a guess.
_UNCERTAINTY_CONF_CAP = 84

# ── Stopwords for the repeated-question filter ────────────────────────────────
# Words that carry no diagnostic meaning and must be ignored when comparing
# a new clarifying_question against questions already present in text_context.
# Kept deliberately small: only words that are (a) high-frequency and (b) would
# never distinguish two genuinely different questions.
_Q_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "if", "it", "its", "this",
    "that", "or", "and", "but", "not", "no", "nor",
    "what", "which", "how", "when", "where", "who", "why",
    "you", "your", "me", "my", "we", "our", "they", "their",
    "please", "tell", "describe", "explain",
    "eg", "example", "such", "like", "any", "some",
    "approximately", "about", "around", "roughly",
})

# ── Extended clarification backfill keyword groups ────────────────────────────
# Used by block 6 of _sanitize_stage_1 to catch Qwen's specific failure mode:
# the model fills uncertainty_notes with real method-changing unknowns but still
# returns diy_suitability="suitable" and clarifying_questions=[].
#
# Each entry is (trade_categories, uncertainty_keywords, targeted_questions).
# A group fires when trade_category is in the tuple AND any keyword appears as
# a substring in uncertainty_notes (casefold). Only method-changing unknowns
# qualify — things the user genuinely cannot proceed without knowing.
# Trade-specific tuples prevent false positives on cabinet holes, shoes, etc.
_CLARIF_EXTENDED_GROUPS: list[tuple[tuple[str, ...], tuple[str, ...], list[str]]] = [
    (
        # Plumbing: pipe material and fitting type change the entire repair method.
        # Without knowing these the user cannot buy the correct coupler or seal.
        ("plumbing",),
        ("pipe", "fitting", "connector", "compression", "push-fit", "push fit",
         "threaded", "solvent weld", "solvent-weld", "copper", "plastic",
         "steel", "joint type", "fitting type", "pipe material", "pipe type",
         "material of the pipe", "corrosion"),
        [
            "What material is the pipe — copper, plastic, or steel?",
            "What type of fitting is currently installed — compression, push-fit, or threaded?",
        ],
    ),
    (
        # Plastering/decorating: substrate and moisture source are blocking.
        # Filler choice, primer type, and whether it is safe to start at all depend on these.
        ("plastering/decorating",),
        ("substrate", "damp", "moisture", "water ingress", "plasterboard",
         "drywall", "concrete", "moisture source", "source of damp",
         "cause of the damp", "wet", "saturated"),
        [
            "What is the wall material — plaster, plasterboard/drywall, or bare concrete?",
            "Is there any sign of dampness or a moisture source behind the surface?",
        ],
    ),
]

# ── Trade-specific uncertainty defaults for step 2b ───────────────────────────
# When the model leaves uncertainty_notes empty on a damage case, step 2b injects
# one of these. Trade-specific defaults contain the keywords that block 6 uses,
# so the two mechanisms work together: 2b ensures the field is non-empty and
# meaningful; block 6 can then detect method-changing unknowns from the text.
# The generic fallback covers all other trades.
_UNCERTAINTY_DEFAULTS: dict[str, str] = {
    "plumbing": (
        "Cannot confirm the pipe material (copper, plastic, or steel), "
        "the fitting type (compression, push-fit, or threaded), or the full "
        "extent of any corrosion or internal damage from the photo alone."
    ),
    "plastering/decorating": (
        "Cannot confirm the wall substrate (plaster, plasterboard/drywall, or bare "
        "concrete), whether there is any damp or moisture source behind the surface, "
        "or the full depth/extent of the damage from the photo alone."
    ),
}
_UNCERTAINTY_DEFAULT_GENERIC = (
    "Cannot confirm from the photo alone: the exact material, any hidden or internal "
    "damage, the full depth/extent, or whether a repair or a full replacement is the "
    "appropriate fix."
)


def _dedupe_strings(items: list[str], jaccard_threshold: float = 0.7) -> list[str]:
    """
    Order-preserving de-duplication that also collapses near-duplicates.

    Two strings are treated as duplicates when either:
      • they are identical after casefold + whitespace normalisation, or
      • their word sets overlap by >= jaccard_threshold (catches the
        "Soil and debris around the pipe" / "Soil and debris on the ground
        around the pipe" variants that exact-match dedup would keep).
    """
    out: list[str] = []
    seen_token_sets: list[set[str]] = []
    for raw in items:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        norm = " ".join(text.casefold().split())
        tokens = set(norm.split())
        is_dup = False
        for prev in seen_token_sets:
            if not tokens or not prev:
                continue
            inter = len(tokens & prev)
            union = len(tokens | prev)
            if union and (inter / union) >= jaccard_threshold:
                is_dup = True
                break
        if is_dup:
            continue
        out.append(text)
        seen_token_sets.append(tokens)
    return out


def _filter_answered_questions(
    questions: list[str],
    text_context: str,
) -> tuple[list[str], list[str]]:
    """Filter out clarifying questions whose topic was already covered in text_context.

    Called only on re-analysis runs (text_context is the Q/A block built by
    _check_auto_reanalyse).  Stage 1 sometimes re-asks a question whose topic
    was already in the previous round, in violation of its own CONTEXT RULE.
    This deterministic filter removes those repeats before _determine_output_mode
    sees the list, preventing an unnecessary extra clarification cycle.

    Algorithm
    ---------
    1. Parse previous question texts from text_context (lines starting "Q<n>: ").
       If no Q/A structure is found (plain text_context), return unchanged.
    2. For each new question: tokenise (lowercase, strip punctuation, drop
       stopwords and single/two-character words).
    3. If a new question shares >= 2 content tokens with ANY previous question,
       it covers the same topic and is dropped.  Otherwise it is kept.

    The >= 2 threshold is the key safety guard:
      - A repeated cause question shares 3-5 words with the original  -> dropped.
      - A new depth/size/condition question shares 0-1 words           -> kept.
      - A safety/risk question uses different vocabulary entirely       -> kept.

    Returns
    -------
    (kept, dropped) — both are lists of question strings.
    The caller reassigns s1.clarifying_questions = kept and logs dropped.
    """
    # Step 1 — extract previous question texts from the structured Q/A block.
    # Format produced by _check_auto_reanalyse:
    #   "User answers to clarification questions:\nQ1: <q>\nA1: <a>\nQ2: ..."
    prev_qs: list[str] = re.findall(
        r"Q\d+:\s*(.+?)(?=\nA\d+:|\Z)", text_context, re.DOTALL
    )
    prev_qs = [q.strip() for q in prev_qs if q.strip()]

    # No Q/A structure in text_context (plain description, first-pass fallback,
    # or a manually typed note) — nothing to compare against, return unchanged.
    if not prev_qs:
        return questions, []

    def _tokens(text: str) -> frozenset[str]:
        """Lowercase content words, length > 2, not in stopword list."""
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return frozenset(w for w in words if len(w) > 2 and w not in _Q_STOPWORDS)

    prev_token_sets: list[frozenset[str]] = [_tokens(q) for q in prev_qs]

    kept: list[str] = []
    dropped: list[str] = []
    for new_q in questions:
        new_tok = _tokens(new_q)
        is_repeat = any(
            len(new_tok & prev_tok) >= 2
            for prev_tok in prev_token_sets
        )
        (dropped if is_repeat else kept).append(new_q)

    return kept, dropped


def _sanitize_stage_1(s1: Stage1Output, text_context: str | None = None) -> list[str]:
    """
    Clean a validated Stage1Output in place. Returns notes on what changed.

    Safe to call on any Stage1Output, including the text-only fallback object.

    text_context is passed on re-analysis runs so that extended backfill block 6
    can detect Q/A-structured input and skip re-injecting questions the user has
    already answered.
    """
    fixes: list[str] = []

    # 1. De-dupe observations (exact + near-duplicate).
    if s1.observations:
        deduped = _dedupe_strings(s1.observations)
        if len(deduped) != len(s1.observations):
            fixes.append(
                f"removed {len(s1.observations) - len(deduped)} duplicate observation(s)"
            )
            s1.observations = deduped

    # 2. De-dupe visual_evidence the same way.
    if s1.visual_evidence:
        deduped_ev = _dedupe_strings(s1.visual_evidence)
        if len(deduped_ev) != len(s1.visual_evidence):
            fixes.append(
                f"removed {len(s1.visual_evidence) - len(deduped_ev)} duplicate evidence item(s)"
            )
            s1.visual_evidence = deduped_ev

    # 2b. For damage cases the model often returns an EMPTY uncertainty_notes even
    #     though material, hidden damage, extent and repair-vs-replace are genuinely
    #     unknown from a photo. Inject a trade-specific default so every downstream
    #     stage receives an honest "what we cannot confirm" signal.
    #     Trade-specific text ensures block 6 can detect method-changing unknowns
    #     (e.g. pipe material for plumbing, substrate for decorating) even when the
    #     model left uncertainty_notes completely empty.
    if not (s1.uncertainty_notes or "").strip() and (s1.severity or "").strip().lower() in ("medium", "high"):
        tc_2b = (s1.trade_category or "").strip().lower()
        s1.uncertainty_notes = _UNCERTAINTY_DEFAULTS.get(tc_2b, _UNCERTAINTY_DEFAULT_GENERIC)
        fixes.append(
            f"added {'trade-specific' if tc_2b in _UNCERTAINTY_DEFAULTS else 'generic'} "
            f"uncertainty note (model left it empty on a {s1.severity} damage case)"
        )

    # 3. Confidence ↔ uncertainty consistency. If the model both claims high
    #    confidence AND admits in its own words that it cannot establish the
    #    core problem/cause, cap the score so the UI does not read as certain.
    unc = (s1.uncertainty_notes or "").casefold()
    if unc and s1.confidence > _UNCERTAINTY_CONF_CAP:
        if any(marker in unc for marker in _UNSURE_MARKERS):
            old = s1.confidence
            s1.confidence = _UNCERTAINTY_CONF_CAP
            fixes.append(
                f"capped confidence {old}->{_UNCERTAINTY_CONF_CAP} "
                f"(uncertainty notes admit an unconfirmed detail)"
            )

    # 4. Very low diversity after dedup = the model echoed itself rather than
    #    truly reading the image → it is not as sure as it claims.
    if not s1.needs_clarification and len(s1.observations) <= 1 and s1.confidence > 80:
        old = s1.confidence
        s1.confidence = 80
        fixes.append(
            f"capped confidence {old}->80 (only one distinct observation extracted)"
        )

    # 5. Clarifying-question backfill (deterministic consistency guard).
    #    The model said 'conditional' (meaningful unknown must be resolved before
    #    proceeding) but produced no questions. This is internally inconsistent:
    #    if it's conditional, something specific needs checking — the model just
    #    forgot to generate the question.  Synthesise up to one generic question
    #    from uncertainty_notes so the pipeline can trigger clarification mode.
    #    Only fires when there is real uncertainty text to draw from.
    if (
        s1.diy_suitability == "conditional"
        and not s1.clarifying_questions
        and (s1.uncertainty_notes or "").strip()
    ):
        unc = s1.uncertainty_notes.strip()
        # Convert "Cannot confirm X" → "To proceed safely, what is X?" format
        # This is a deterministic fallback — the Step 4 prompt fix should make it
        # rare by training the model to always pair conditional+questions.
        backfill_q = (
            f"To give you the right advice, I need to know a bit more. "
            f"Specifically: {unc.rstrip('.')}"
            f" — can you check or upload a closer photo?"
        )
        s1.clarifying_questions = [backfill_q]
        fixes.append("backfilled 1 clarifying_question from uncertainty_notes (conditional but empty)")

    # 6. Extended backfill: "suitable" + method-changing uncertainty contradiction.
    #
    #    Qwen's specific failure mode: the model populates uncertainty_notes with
    #    real unknowns (pipe material, fitting type, substrate, damp source) but
    #    still defaults to diy_suitability="suitable" with clarifying_questions=[].
    #    "Suitable" is internally inconsistent when those unknowns directly change
    #    the repair method, parts, or safety assessment.
    #
    #    Narrow triggers (ALL must be true before any keyword check):
    #      - diy_suitability is "suitable"   (block 5 already handles "conditional")
    #      - clarifying_questions is empty    (nothing generated by model or block 5)
    #      - uncertainty_notes is non-empty   (real text to draw from)
    #      - professional_assessment_recommended is False  (assessment-first cases stay put)
    #      - safety_concerns is empty         (hazard cases stay on their path)
    #      - NOT a re-analysis with Q/A-structured text_context — on re-analysis the
    #        model already saw the answers and deliberately left questions empty.
    #        Re-injecting the original questions here would loop the case back into
    #        clarification mode even though the user already responded.  The filter
    #        (FILTER_REPEATED_CLARIFICATION_QUESTIONS) is a second safety net, but
    #        it cannot reset diy_suitability after the backfill changes it to
    #        "conditional", so blocking the backfill entirely is the right fix.
    #    Then: trade + uncertainty keywords must match a known method-changing pattern.
    _is_reanalysis_qa = bool(
        text_context
        and (
            "Q1:" in text_context
            or text_context.strip().startswith("User answers")
        )
    )
    if (
        s1.diy_suitability == "suitable"
        and not s1.clarifying_questions
        and (s1.uncertainty_notes or "").strip()
        and not s1.professional_assessment_recommended
        and not s1.safety_concerns
        and not _is_reanalysis_qa
    ):
        unc_lower = (s1.uncertainty_notes or "").casefold()
        tc = (s1.trade_category or "").strip().lower()
        for trade_tuple, unc_keywords, questions in _CLARIF_EXTENDED_GROUPS:
            if tc in trade_tuple and any(kw in unc_lower for kw in unc_keywords):
                s1.diy_suitability = "conditional"
                s1.clarifying_questions = list(questions)
                fixes.append(
                    f"extended backfill: suitable->conditional, added "
                    f"{len(questions)} question(s) "
                    f"(trade={tc}, uncertainty contains method-changing unknown)"
                )
                break  # first matching group wins; never fire two groups

    return fixes
