"""
Deterministic unit tests for the clarification reasoning unit (Phase 2, Step 1).

No database, no models, no network — pure reasoning over plain strings. Runnable
two ways:
    python tests/test_clarification_reasoning.py     # standalone, prints PASS/FAIL
    pytest tests/test_clarification_reasoning.py      # discovered as test_* funcs
"""
import os
import sys

# Make 'app' importable when run as a standalone script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clarification_reasoning import assess_clarification

Q_DAMP = "Is there any sign of dampness or a moisture source?"
Q_WALL = "Is the wall plasterboard/drywall or another material?"


def test_out_of_order_maps_to_correct_question():
    r = assess_clarification([Q_DAMP, Q_WALL], ["It is drywall"])
    cov = {c.question: c for c in r.coverage}
    assert cov[Q_WALL].answered is True
    assert cov[Q_WALL].answer == "It is drywall"
    assert cov[Q_DAMP].answered is False
    assert r.ready is False
    assert r.open_questions == [Q_DAMP]


def test_one_message_answers_two_questions():
    r = assess_clarification(
        [Q_DAMP, Q_WALL],
        ["It is drywall and there is no sign of dampness"],
    )
    assert r.ready is True
    assert len(r.resolved_pairs) == 2
    assert r.open_questions == []


def test_partial_answer_leaves_other_open():
    r = assess_clarification([Q_DAMP, Q_WALL], ["There is no sign of dampness anywhere"])
    assert r.ready is False
    assert r.open_questions == [Q_WALL]


def test_irrelevant_answer_maps_to_nothing_and_not_ready():
    r = assess_clarification([Q_DAMP, Q_WALL], ["What time is it?"])
    assert r.ready is False
    assert r.resolved_pairs == []
    assert all(c.answered is False for c in r.coverage)


def test_contradictory_answers_use_last():
    a1 = "There is no dampness in the area"
    a2 = "Actually dampness is present behind the plaster"
    r = assess_clarification([Q_DAMP], [a1, a2])
    cov = {c.question: c for c in r.coverage}
    assert cov[Q_DAMP].answered is True
    assert cov[Q_DAMP].answer == a2


def test_prior_round_context_appears_once():
    prior = (
        "User answers to clarification questions:\n"
        "Q1: What material is the pipe?\n"
        "A1: It is copper"
    )
    q_access = "What is the access like around the joint?"
    r = assess_clarification(
        [q_access],
        ["There is good access from above"],
        prior_context=prior,
    )
    assert r.assembled_context.count(prior) == 1
    assert r.assembled_context.count("A1: It is copper") == 1


def test_ambiguous_input_fails_safe_not_ready():
    r = assess_clarification([Q_DAMP, Q_WALL], ["maybe", "I'm not sure"])
    assert r.ready is False
    assert r.resolved_pairs == []


_TESTS = [
    ("1 out-of-order maps to correct question", test_out_of_order_maps_to_correct_question),
    ("2 one message answers two questions",     test_one_message_answers_two_questions),
    ("3 partial answer leaves other open",      test_partial_answer_leaves_other_open),
    ("4 irrelevant maps to nothing, not ready", test_irrelevant_answer_maps_to_nothing_and_not_ready),
    ("5 contradictory uses last answer",        test_contradictory_answers_use_last),
    ("6 prior-round context appears once",      test_prior_round_context_appears_once),
    ("7 ambiguous fails safe (not ready)",      test_ambiguous_input_fails_safe_not_ready),
]


def _run() -> bool:
    print("=" * 64)
    print("CLARIFICATION REASONING — STEP 1 UNIT TESTS")
    print("=" * 64)
    passed = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {name}")
            print(f"         {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error
            print(f"  [ERROR] {name}: {exc!r}")
    print(f"\n  {passed}/{len(_TESTS)} passed")
    print("=" * 64)
    return passed == len(_TESTS)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
