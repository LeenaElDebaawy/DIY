"""
Acceptance tests for Step 2 — the manual re-analysis context builder now
delegates to the shared clarification reasoning unit.

These drive the REAL `_build_manual_reanalysis_context` through a lightweight
fake async DB session (a test double), so no Postgres is required. The fake
returns canned results in the fixed order the function queries them:
    1) latest analysis, 2) most-recent opener, 3) this round's user messages.

Runnable standalone:  python tests/test_manual_reanalysis_context.py
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Returns queued results in call order; ignores the query itself."""
    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    async def execute(self, _query):
        r = self._results[self._i]
        self._i += 1
        return r


def _analysis(questions, optional=None, prior=ROUND1, mode="clarification"):
    return SimpleNamespace(
        id=2,
        text_context=prior,
        structured_result={
            "output_mode": mode,
            "clarification": {
                "clarifying_questions": list(questions),
                "optional_questions": list(optional or []),
            },
        },
    )


def _msg(content):
    return SimpleNamespace(
        content=content,
        role="user",
        message_type="chat",
        msg_metadata={"clarification_answer": True},
    )


def _opener():
    return SimpleNamespace(created_at="t", message_type="clarification_opener")


def _build(analysis, opener, messages, caller_text):
    db = _FakeDB([_Result([analysis]), _Result([opener]), _Result(messages)])
    return asyncio.run(_build_manual_reanalysis_context(db, 999, caller_text))


# ── Tests ────────────────────────────────────────────────────────────────────
def test_out_of_order_maps_to_correct_question():
    ctx = _build(_analysis([Q_DAMP, Q_WALL]), _opener(), [_msg("It is drywall")], None)
    # Answer is paired with the wall question, not the dampness question.
    assert f"Q1: {Q_WALL}\nA1: It is drywall" in ctx
    assert "A1: It is drywall" in ctx and f"{Q_DAMP}\nA1: It is drywall" not in ctx


def test_multi_round_no_duplication():
    ctx = _build(_analysis([Q_DAMP, Q_WALL]), _opener(), [_msg("It is drywall")], None)
    # Round-one content carried forward exactly once.
    assert ctx.count("A1: It is copper") == 1
    assert ctx.count(ROUND1) == 1


def test_single_message_answers_both():
    ctx = _build(
        _analysis([Q_DAMP, Q_WALL]),
        _opener(),
        [_msg("It is drywall and there is no sign of dampness")],
        None,
    )
    assert Q_DAMP in ctx and Q_WALL in ctx
    assert "no sign of dampness" in ctx


def test_textarea_note_is_appended():
    ctx = _build(_analysis([Q_DAMP, Q_WALL]), _opener(), [_msg("It is drywall")], "please recheck")
    assert "Additional note from user: please recheck" in ctx


def test_unmapped_answer_is_preserved_not_dropped():
    # A user answer that maps to no question must still reach the model.
    ctx = _build(_analysis([Q_DAMP, Q_WALL]), _opener(), [_msg("not really certain about that")], None)
    assert "not really certain about that" in ctx
    # …but it is not falsely attached to a question as a Q/A pair.
    assert "A1: not really certain about that" not in ctx


def test_no_answers_passthrough_returns_caller_text():
    ctx = _build(_analysis([Q_DAMP, Q_WALL]), _opener(), [], "just a typed note")
    assert ctx == "just a typed note"


def test_non_clarification_passthrough():
    a = _analysis([Q_DAMP], mode="diy")
    db = _FakeDB([_Result([a])])  # only the analysis fetch happens
    ctx = asyncio.run(_build_manual_reanalysis_context(db, 999, "carry me through"))
    assert ctx == "carry me through"


_TESTS = [
    ("1 out-of-order maps to correct question", test_out_of_order_maps_to_correct_question),
    ("2 multi-round: no duplication",           test_multi_round_no_duplication),
    ("3 single message answers both",           test_single_message_answers_both),
    ("4 textarea note appended",                test_textarea_note_is_appended),
    ("5 unmapped answer preserved, not paired", test_unmapped_answer_is_preserved_not_dropped),
    ("6 no answers -> caller_text passthrough", test_no_answers_passthrough_returns_caller_text),
    ("7 non-clarification -> passthrough",      test_non_clarification_passthrough),
]


def _run() -> bool:
    print("=" * 64)
    print("MANUAL RE-ANALYSIS CONTEXT — STEP 2 ACCEPTANCE TESTS")
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
