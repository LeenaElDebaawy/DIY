"""
Step 3 acceptance tests — automatic clarification path + yes/no refinement.

Pure/deterministic. No Postgres, no models. The automatic evaluator and the
manual builder are driven through a lightweight fake async DB session.

Runnable standalone:  python tests/test_auto_clarification_step3.py
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clarification_reasoning import (
    assess_clarification,
    build_reprompt,
    finalize_context,
)
from app.services.chat_service import _evaluate_clarification, _INTRAROUND_ANSWER_LIMIT
from app.services.analysis_service import _build_manual_reanalysis_context

Q_DAMP = "Is there any sign of dampness or a moisture source?"
Q_WALL = "Is the wall plasterboard/drywall or another material?"
ROUND1 = (
    "User answers to clarification questions:\n"
    "Q1: What material is the pipe?\n"
    "A1: It is copper"
)


# ── Fake async DB session ────────────────────────────────────────────────────
class _Scalars:
    def __init__(self, items):
        self._items = items

    def first(self):
        return self._items[0] if self._items else None

    def all(self):
        return list(self._items)


class _Result:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)

    def scalar_one_or_none(self):
        return self._items[0] if self._items else None


class _FakeDB:
    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    async def execute(self, _query):
        r = self._results[self._i]
        self._i += 1
        return r


def _analysis(questions, prior=None, mode="clarification"):
    return SimpleNamespace(
        id=2,
        text_context=prior,
        structured_result={
            "output_mode": mode,
            "clarification": {
                "clarifying_questions": list(questions),
                "optional_questions": [],
            },
        },
    )


def _msg(content):
    return SimpleNamespace(
        content=content, role="user", message_type="chat",
        msg_metadata={"clarification_answer": True},
    )


def _opener():
    return SimpleNamespace(created_at="t", message_type="clarification_opener")


def _evaluate(analysis, messages, opener=True, sentinel=False, prior=None):
    """Drive the automatic evaluator through the fake DB (analysis, opener,
    sentinel, answers — in that query order)."""
    results = [
        _Result([analysis]),
        _Result([_opener()] if opener else []),
        _Result([SimpleNamespace()] if sentinel else []),
        _Result(messages),
    ]
    return asyncio.run(_evaluate_clarification(_FakeDB(results), 999))


# ── 3b — Yes/No refinement ───────────────────────────────────────────────────
def test_single_open_yes_no_closed_by_no():
    a = assess_clarification([Q_DAMP], ["No"])
    assert a.ready is True
    assert a.coverage[0].answered is True and a.coverage[0].answer == "No"


def test_single_open_yes_no_closed_by_yes():
    a = assess_clarification([Q_DAMP], ["Yes"])
    assert a.ready is True


def test_two_open_yes_no_is_ambiguous():
    a = assess_clarification([Q_DAMP, Q_WALL], ["No"])
    assert a.ready is False  # bare yes/no must NOT close coverage with 2 open


def test_choice_question_not_closed_by_yes():
    # Q_WALL leads with "Is" but enumerates options -> not a true yes/no question.
    a = assess_clarification([Q_WALL], ["yes"])
    assert a.ready is False


def test_no_idea_is_not_a_clean_negation():
    a = assess_clarification([Q_DAMP], ["no idea"])
    assert a.ready is False


def test_content_then_yesno_completes_round():
    # drywall maps Q_WALL by content; "No" closes the single remaining Q_DAMP.
    a = assess_clarification([Q_DAMP, Q_WALL], ["It is drywall", "No"])
    assert a.ready is True
    assert len(a.resolved_pairs) == 2


# ── Parity: automatic vs manual produce identical context ────────────────────
def test_auto_manual_context_parity():
    analysis = _analysis([Q_DAMP, Q_WALL], prior=ROUND1)
    msgs = [_msg("It is drywall and there is no sign of dampness")]
    decision = _evaluate(analysis, msgs, prior=ROUND1)
    manual_db = _FakeDB([_Result([analysis]), _Result([_opener()]), _Result(msgs)])
    manual_ctx = asyncio.run(_build_manual_reanalysis_context(manual_db, 999, None))
    assert decision.context == manual_ctx


# ── 3a — Automatic path decision logic ───────────────────────────────────────
def test_two_questions_one_answered_not_ready_reprompts_open_only():
    decision = _evaluate(_analysis([Q_DAMP, Q_WALL]), [_msg("It is drywall")])
    assert decision.eligible is True
    assert decision.ready is False
    assert decision.attempts == 1
    # Re-prompt acknowledges the answered one and asks ONLY the open one.
    assert "It is drywall" in decision.reprompt
    assert Q_DAMP in decision.reprompt
    assert Q_WALL not in decision.reprompt


def test_one_message_answers_both_ready():
    decision = _evaluate(
        _analysis([Q_DAMP, Q_WALL]),
        [_msg("It is drywall and there is no sign of dampness")],
    )
    assert decision.eligible is True and decision.ready is True


def test_answers_across_multiple_messages_ready_when_complete():
    decision = _evaluate(
        _analysis([Q_DAMP, Q_WALL]),
        [_msg("It is drywall"), _msg("There is no sign of dampness")],
    )
    assert decision.ready is True
    assert decision.attempts == 2


def test_out_of_order_maps_correctly_in_context():
    decision = _evaluate(_analysis([Q_DAMP, Q_WALL]), [_msg("It is drywall")])
    # The answer is paired with the wall question in the assembled context.
    assert f"Q1: {Q_WALL}\nA1: It is drywall" in decision.context


def test_already_triggered_is_ineligible():
    decision = _evaluate(_analysis([Q_DAMP]), [_msg("No")], sentinel=True)
    assert decision.eligible is False


def test_non_clarification_is_ineligible():
    decision = _evaluate(_analysis([Q_DAMP], mode="diy"), [_msg("No")])
    assert decision.eligible is False


def test_no_opener_is_ineligible():
    decision = _evaluate(_analysis([Q_DAMP]), [_msg("No")], opener=False)
    assert decision.eligible is False


def test_intra_round_override_conditions_met():
    # Three unmappable answers: still not ready, but attempts reaches the limit so
    # stream_message's intra-round safeguard would fire; context preserves input.
    msgs = [_msg("hmm"), _msg("not certain"), _msg("dunno really")]
    decision = _evaluate(_analysis([Q_DAMP, Q_WALL]), msgs)
    assert decision.eligible is True
    assert decision.ready is False
    assert decision.attempts >= _INTRAROUND_ANSWER_LIMIT
    assert "Additional details from the user:" in decision.context


def test_no_silence_invariant():
    # Every eligible not-ready evaluation yields a non-empty deterministic re-prompt.
    decision = _evaluate(_analysis([Q_DAMP, Q_WALL]), [_msg("What does drywall mean?")])
    assert decision.eligible is True and decision.ready is False
    assert decision.reprompt.strip() != ""


_TESTS = [
    ("3b-1 single open yes/no closed by No",     test_single_open_yes_no_closed_by_no),
    ("3b-2 single open yes/no closed by Yes",    test_single_open_yes_no_closed_by_yes),
    ("3b-3 two open + No is ambiguous",          test_two_open_yes_no_is_ambiguous),
    ("3b-4 choice question not closed by yes",   test_choice_question_not_closed_by_yes),
    ("3b-5 'no idea' is not a clean negation",   test_no_idea_is_not_a_clean_negation),
    ("3b-6 content then yes/no completes round", test_content_then_yesno_completes_round),
    ("parity auto == manual context",            test_auto_manual_context_parity),
    ("3a-1 two Qs one answered -> reprompt open", test_two_questions_one_answered_not_ready_reprompts_open_only),
    ("3a-2 one message answers both -> ready",   test_one_message_answers_both_ready),
    ("3a-3 multi-message -> ready when complete", test_answers_across_multiple_messages_ready_when_complete),
    ("3a-4 out-of-order maps correctly",         test_out_of_order_maps_correctly_in_context),
    ("3a-5 already triggered -> ineligible",     test_already_triggered_is_ineligible),
    ("3a-6 non-clarification -> ineligible",     test_non_clarification_is_ineligible),
    ("3a-7 no opener -> ineligible",             test_no_opener_is_ineligible),
    ("3a-8 intra-round override conditions",     test_intra_round_override_conditions_met),
    ("3a-9 no-silence invariant",                test_no_silence_invariant),
]


def _run() -> bool:
    print("=" * 64)
    print("AUTOMATIC CLARIFICATION PATH — STEP 3 ACCEPTANCE TESTS")
    print("=" * 64)
    passed = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {name}: {exc!r}")
    print(f"\n  {passed}/{len(_TESTS)} passed")
    print("=" * 64)
    return passed == len(_TESTS)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
