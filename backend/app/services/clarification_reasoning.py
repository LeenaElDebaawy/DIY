"""
Clarification reasoning unit — Phase 2 closeout, Step 1.

PURE, ISOLATED reasoning component. It is intentionally NOT wired into the
automatic re-analysis path, the manual re-analysis path, or the frontend. It
performs no database access, no model calls, and has no side effects. Importing
this module changes no application behaviour.

Given the clarification questions for the current round and the user's answer
messages (in the order they were sent), it determines:

  - which required question each answer actually addresses (by content, not by
    position),
  - which required questions remain open,
  - whether every required question is covered (readiness),
  - the resolved question -> answer mapping (for accurate history),
  - a cumulative, de-duplicated context string suitable for re-analysis.

Coverage is decided from the CONTENT of the answers against the meaning of each
question, using a deterministic content-overlap strategy: an answer addresses a
question when it shares at least one distinctive term with that question. This
naturally supports answering questions out of order and answering several
questions in one message.

The component FAILS SAFE. When an answer cannot be confidently associated with a
question (irrelevant, ambiguous, or a bare yes/no with no distinctive content),
the question stays open and the round is reported as NOT ready — the unit prefers
to ask again rather than proceed on misunderstood input. Bare yes/no answers
with no distinctive terms are a known conservative case, resolvable later by a
stronger (e.g. model-assisted) coverage strategy behind this same interface.

The coverage strategy is deliberately the only "intelligence" here and is kept
behind a small private surface so it can be upgraded without changing callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Terms with no diagnostic content. Removed before matching so an answer is
# associated with a question only through meaningful, distinctive words.
_STOPWORDS: frozenset[str] = frozenset({
    "is", "are", "am", "was", "were", "be", "been", "being",
    "the", "a", "an", "of", "or", "and", "any", "to", "in", "on", "at",
    "for", "with", "this", "that", "these", "those", "there", "here",
    "it", "its", "do", "does", "did", "done", "has", "have", "had",
    "you", "your", "yours", "i", "we", "they", "them",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "if", "then", "else", "so", "but", "as", "by", "from", "about",
    "another", "other", "some", "no", "not", "yes", "maybe", "sure",
    "currently", "installed", "used", "made", "kind", "type", "please",
    "into", "onto", "over", "under", "behind", "near", "around",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "more", "most", "very", "just", "also", "still", "now", "actually",
})

_MIN_TERM_LEN = 3
_CONTEXT_HEADER = "User answers to clarification questions:"


def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split into word tokens."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def _salient_terms(text: str) -> set[str]:
    """The distinctive content terms of a question or answer."""
    return {
        t for t in _normalise(text)
        if t not in _STOPWORDS and len(t) >= _MIN_TERM_LEN
    }


def _addresses(answer: str, question_terms: set[str]) -> bool:
    """True if the answer shares at least one distinctive term with the question."""
    if not question_terms:
        return False
    return bool(_salient_terms(answer) & question_terms)


# ── Yes/No refinement (applies ONLY to a single remaining open question) ───────
_YES_NO_LEAD: frozenset[str] = frozenset({
    "is", "are", "am", "was", "were", "do", "does", "did",
    "has", "have", "had", "can", "could", "will", "would", "should",
})
_AFFIRM: frozenset[str] = frozenset({
    "yes", "yeah", "yep", "yup", "correct", "right", "true", "affirmative",
    "ok", "okay",
})
_NEGATE: frozenset[str] = frozenset({
    "no", "nope", "nah", "none", "negative", "false",
})
# Words allowed to surround a yes/no without disqualifying it as a clean answer.
_YESNO_FILLER: frozenset[str] = frozenset({
    "really", "definitely", "absolutely", "certainly", "indeed", "sure",
    "probably", "think", "all", "isn", "doesn", "don", "didn",
})


def _is_yes_no_question(question: str) -> bool:
    """True only if the question is genuinely answerable by affirmation/negation.

    Conservative: a question that leads with a yes/no auxiliary is still excluded
    when it enumerates choices (e.g. "Is the wall plasterboard/drywall or another
    material?"), because a bare "yes"/"no" carries no information there. Slash-
    separated options and the words "another/other/either" mark such choices.
    """
    toks = _normalise(question)
    if not toks or toks[0] not in _YES_NO_LEAD:
        return False
    if "/" in question:  # slash-enumerated options -> a choice, not a yes/no
        return False
    if {"another", "other", "either"} & set(toks):
        return False
    return True


def _clean_yes_no(message: str) -> str | None:
    """Return 'affirm' or 'negate' only for a clean yes/no message with no
    competing content (e.g. 'No', 'Yes it is', 'definitely yes'). Returns None
    for anything ambiguous ('no idea', 'maybe', 'yes but the wall is wet')."""
    toks = _normalise(message)
    if not toks:
        return None
    has_aff = any(t in _AFFIRM for t in toks)
    has_neg = any(t in _NEGATE for t in toks)
    if has_aff == has_neg:  # neither present, or both present -> not clean
        return None
    leftover = [
        t for t in toks
        if t not in _AFFIRM and t not in _NEGATE
        and t not in _YESNO_FILLER and t not in _STOPWORDS
    ]
    if leftover:
        return None
    return "affirm" if has_aff else "negate"


@dataclass(frozen=True)
class QuestionCoverage:
    """Coverage state for a single required question."""
    question: str
    answered: bool
    answer: str | None  # the resolved (most recent) addressing answer, or None


@dataclass(frozen=True)
class ClarificationAssessment:
    """The result of reasoning over one clarification round."""
    coverage: list[QuestionCoverage]        # one entry per required question, in order
    ready: bool                             # True only when every required question is answered
    open_questions: list[str]               # required questions still unanswered
    resolved_pairs: list[tuple[str, str]]   # (question, answer) for answered questions
    assembled_context: str                  # cumulative, de-duplicated re-analysis context


def _assemble_context(
    resolved_pairs: list[tuple[str, str]],
    prior_context: str | None,
) -> str:
    """Build the cumulative context: prior rounds carried forward once, then this
    round's resolved pairs, skipping any pair already present in the prior text."""
    prior = (prior_context or "").strip()
    lines: list[str] = []
    n = 0
    for question, answer in resolved_pairs:
        # De-duplicate: do not re-emit a pair already present in the prior context.
        if prior and question in prior and answer in prior:
            continue
        n += 1
        lines.append(f"Q{n}: {question}")
        lines.append(f"A{n}: {answer}")

    parts: list[str] = []
    if prior:
        parts.append(prior)
    if lines:
        parts.append(_CONTEXT_HEADER + "\n" + "\n".join(lines))
    return "\n\n".join(parts)


def assess_clarification(
    required_questions: list[str],
    answer_messages: list[str],
    prior_context: str | None = None,
    optional_questions: list[str] | None = None,
) -> ClarificationAssessment:
    """Reason over one clarification round. Pure: no I/O, no model calls.

    Parameters
    ----------
    required_questions
        The round's required questions, in the order they were asked.
    answer_messages
        The user's answer messages for this round, in chronological order. A
        single message may address several questions.
    prior_context
        Assembled context carried forward from earlier rounds, or None for the
        first round. Earlier rounds are preserved and never duplicated.
    optional_questions
        Helpful extras that are captured if answered but never block readiness.

    Returns
    -------
    ClarificationAssessment
        Coverage per required question, readiness, open questions, the resolved
        question->answer mapping, and the de-duplicated cumulative context.
    """
    required = [q for q in (required_questions or []) if q and q.strip()]
    optional = [q for q in (optional_questions or []) if q and q.strip()]
    answers = [m for m in (answer_messages or []) if m and m.strip()]

    coverage: list[QuestionCoverage] = []
    resolved_pairs: list[tuple[str, str]] = []
    open_questions: list[str] = []

    for question in required:
        terms = _salient_terms(question)
        # The most recent addressing answer wins, so a later correction supersedes
        # an earlier, contradicting answer.
        resolved: str | None = None
        for message in answers:
            if _addresses(message, terms):
                resolved = message.strip()
        if resolved is not None:
            coverage.append(QuestionCoverage(question, True, resolved))
            resolved_pairs.append((question, resolved))
        else:
            coverage.append(QuestionCoverage(question, False, None))
            open_questions.append(question)

    # Yes/No refinement: ONLY when exactly one required question remains open and
    # it is yes/no-answerable, a clean affirmation/negation closes it. With two or
    # more open, a bare "yes"/"no" is ambiguous and is intentionally NOT applied.
    if len(open_questions) == 1 and _is_yes_no_question(open_questions[0]):
        only_open = open_questions[0]
        yes_no_answer: str | None = None
        for message in answers:
            if _clean_yes_no(message) is not None:
                yes_no_answer = message.strip()  # most recent clean yes/no wins
        if yes_no_answer is not None:
            coverage = [
                QuestionCoverage(c.question, True, yes_no_answer)
                if (c.question == only_open and not c.answered) else c
                for c in coverage
            ]
            open_questions = [q for q in open_questions if q != only_open]
            resolved_pairs.append((only_open, yes_no_answer))

    # Optional questions never affect readiness, but are carried into context if
    # the user happened to answer them.
    for question in optional:
        terms = _salient_terms(question)
        resolved = None
        for message in answers:
            if _addresses(message, terms):
                resolved = message.strip()
        if resolved is not None:
            resolved_pairs.append((question, resolved))

    ready = len(open_questions) == 0
    assembled_context = _assemble_context(resolved_pairs, prior_context)

    return ClarificationAssessment(
        coverage=coverage,
        ready=ready,
        open_questions=open_questions,
        resolved_pairs=resolved_pairs,
        assembled_context=assembled_context,
    )


def finalize_context(
    assessment: ClarificationAssessment,
    answer_messages: list[str],
    note: str | None = None,
) -> str:
    """Produce the final re-analysis context from an assessment.

    Shared by the automatic and manual paths so both build identical context for
    identical inputs. Starts from the assessment's de-duplicated assembled context,
    then preserves any answer the unit could not confidently map (as free-text, not
    a false question pairing) and appends an optional user note. A manual override
    must never silently discard user input; the automatic path uses note=None.
    """
    text = assessment.assembled_context
    mapped = {answer for _, answer in assessment.resolved_pairs}
    unmapped = [
        m.strip() for m in (answer_messages or [])
        if m and m.strip() and m.strip() not in mapped
    ]
    addenda: list[str] = []
    if unmapped:
        addenda.append("Additional details from the user: " + " ".join(unmapped))
    if note and note.strip():
        addenda.append(f"Additional note from user: {note.strip()}")
    if addenda:
        text = (text + "\n\n" + "\n".join(addenda)).strip() if text else "\n".join(addenda)
    return text


def build_reprompt(
    resolved_pairs: list[tuple[str, str]],
    open_questions: list[str],
) -> str:
    """Deterministic clarification re-prompt, driven entirely by the unit's output.

    Acknowledges what has already been answered and asks ONLY the remaining open
    question(s). Never depends on free-form model generation.
    """
    lines: list[str] = []
    if resolved_pairs:
        noted = "; ".join(answer for _, answer in resolved_pairs)
        lines.append(f"Thanks — I've noted: {noted}.")
    else:
        lines.append("Thanks for that.")
    if open_questions:
        lines.append("")
        lines.append("Before I can update the analysis, I still need to confirm:")
        for question in open_questions:
            lines.append(f"• {question}")
    return "\n".join(lines)
