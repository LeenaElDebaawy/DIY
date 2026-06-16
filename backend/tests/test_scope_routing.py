"""
Tests for the lean hybrid scope-first fix.

Pure/deterministic: exercises Stage1Output schema defaults + _determine_output_mode
with crafted scope values. No Ollama, no DB.

Runnable standalone:  python tests/test_scope_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.analysis import Stage1Output, OutOfScopeOutput
from app.services.analysis_service import _determine_output_mode


def _s1(**over):
    base = dict(
        problem_type="Test", trade_category="general/other", severity="low",
        observations=["something"], confidence=85,
        risk_level="low", hidden_damage_risk="low", diy_suitability="suitable",
        professional_assessment_recommended=False, safety_concerns=[],
        clarifying_questions=[], optional_questions=[],
    )
    base.update(over)
    return Stage1Output(**base)


# ── Schema default ────────────────────────────────────────────────────────────
def test_scope_defaults_to_repair_when_missing():
    # Old analyses (and any output that omits scope) must default to "repair".
    s1 = _s1()  # no scope passed
    assert s1.scope == "repair"


def test_scope_validator_normalises_synonyms():
    assert _s1(scope="active-emergency").scope == "emergency"
    assert _s1(scope="non_repair").scope == "not_repair"
    assert _s1(scope="ambiguous").scope == "unclear"
    assert _s1(scope="totally-unknown-value").scope == "repair"  # unknown -> default


# ── Routing: scope short-circuits ─────────────────────────────────────────────
def test_emergency_routes_out_of_scope():
    # Fire-style signals: even with critical/not_recommended, scope wins.
    s1 = _s1(scope="emergency", risk_level="critical", severity="high",
             diy_suitability="not_recommended", trade_category="general/other")
    assert _determine_output_mode(s1) == "out_of_scope"


def test_not_repair_routes_out_of_scope():
    s1 = _s1(scope="not_repair", confidence=30)
    assert _determine_output_mode(s1) == "out_of_scope"


def test_unclear_routes_out_of_scope():
    # Safest minimal: unclear is treated as out_of_scope (no repair pipeline).
    s1 = _s1(scope="unclear", confidence=40)
    assert _determine_output_mode(s1) == "out_of_scope"


def test_repair_scope_normal_diy_unchanged():
    s1 = _s1(scope="repair", diy_suitability="suitable", risk_level="low",
             severity="medium", trade_category="plumbing")
    assert _determine_output_mode(s1) == "diy"


def test_repair_scope_not_recommended_still_assessment_first():
    # The earlier professional-required fix must still hold under scope=repair.
    s1 = _s1(scope="repair", diy_suitability="not_recommended", risk_level="high",
             trade_category="glazing", safety_concerns=[])
    assert _determine_output_mode(s1) == "assessment_first"


def test_scope_default_repair_preserves_all_existing_routes():
    # Without scope, the five existing modes must route exactly as before.
    assert _determine_output_mode(_s1(diy_suitability="suitable", risk_level="low",
                                      severity="medium", trade_category="plumbing")) == "diy"
    assert _determine_output_mode(_s1(diy_suitability="conditional")) == "clarification"
    assert _determine_output_mode(_s1(diy_suitability="suitable", risk_level="high")) == "verify_first"
    assert _determine_output_mode(_s1(diy_suitability="not_recommended",
                                      safety_concerns=["x"])) == "assessment_first"
    assert _determine_output_mode(_s1(problem_type="Scuffed shoe", trade_category="general/other",
                                      severity="low", risk_level="low",
                                      diy_suitability="suitable")) == "care"


# ── Out-of-scope output shape ─────────────────────────────────────────────────
def test_out_of_scope_output_emergency_shape():
    o = OutOfScopeOutput(scope="emergency", is_emergency=True,
                         headline="h", guidance="g")
    d = o.model_dump()
    assert d["is_emergency"] is True and d["scope"] == "emergency"


def test_out_of_scope_output_defaults():
    o = OutOfScopeOutput()
    assert o.scope == "not_repair" and o.is_emergency is False


_TESTS = [
    ("1 scope defaults to repair when missing",          test_scope_defaults_to_repair_when_missing),
    ("2 scope validator normalises synonyms",            test_scope_validator_normalises_synonyms),
    ("3 emergency -> out_of_scope",                       test_emergency_routes_out_of_scope),
    ("4 not_repair -> out_of_scope",                      test_not_repair_routes_out_of_scope),
    ("5 unclear -> out_of_scope (safest minimal)",       test_unclear_routes_out_of_scope),
    ("6 repair + normal -> diy unchanged",               test_repair_scope_normal_diy_unchanged),
    ("7 repair + not_recommended -> assessment_first",   test_repair_scope_not_recommended_still_assessment_first),
    ("8 scope default preserves five modes",             test_scope_default_repair_preserves_all_existing_routes),
    ("9 out_of_scope emergency output shape",            test_out_of_scope_output_emergency_shape),
    ("10 out_of_scope output defaults",                  test_out_of_scope_output_defaults),
]


def _run() -> bool:
    print("=" * 64)
    print("SCOPE-FIRST HYBRID — ROUTING & SCHEMA TESTS")
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
