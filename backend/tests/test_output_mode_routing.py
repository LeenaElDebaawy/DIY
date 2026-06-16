"""
Routing tests for the "Professional required vs DIY steps" fix.

Pure/deterministic: exercises _determine_output_mode directly with crafted
Stage1Output objects. No Ollama, no DB.

Runnable standalone:  python tests/test_output_mode_routing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.analysis import Stage1Output
from app.services.analysis_service import _determine_output_mode


def _s1(**over):
    """Stage1Output with safe, DIY-leaning defaults; override per test."""
    base = dict(
        problem_type="Test", trade_category="general/other", severity="low",
        observations=["something"], confidence=85,
        risk_level="low", hidden_damage_risk="low", diy_suitability="suitable",
        professional_assessment_recommended=False, safety_concerns=[],
        clarifying_questions=[], optional_questions=[],
    )
    base.update(over)
    return Stage1Output(**base)


def test_not_recommended_empty_hazards_routes_assessment_first():
    # Case 167 scenario: not_recommended + high risk + NO named hazard.
    s1 = _s1(diy_suitability="not_recommended", risk_level="high",
             severity="high", trade_category="glazing", safety_concerns=[])
    assert _determine_output_mode(s1) == "assessment_first"


def test_not_recommended_with_hazard_routes_assessment_first():
    s1 = _s1(diy_suitability="not_recommended", risk_level="high",
             safety_concerns=["flying glass shards"])
    assert _determine_output_mode(s1) == "assessment_first"


def test_high_risk_suitable_low_hidden_routes_verify_first():
    s1 = _s1(diy_suitability="suitable", risk_level="high", hidden_damage_risk="low",
             trade_category="plumbing")
    assert _determine_output_mode(s1) == "verify_first"


def test_critical_risk_routes_at_least_verify_first():
    s1 = _s1(diy_suitability="suitable", risk_level="critical", hidden_damage_risk="low")
    assert _determine_output_mode(s1) == "verify_first"


def test_conditional_still_clarification_even_if_high_risk():
    # Stronger escalation (clarification) wins over verify_first.
    s1 = _s1(diy_suitability="conditional", risk_level="high")
    assert _determine_output_mode(s1) == "clarification"


def test_normal_low_risk_suitable_routes_diy():
    s1 = _s1(diy_suitability="suitable", risk_level="low", severity="medium",
             hidden_damage_risk="low", trade_category="plumbing")
    assert _determine_output_mode(s1) == "diy"


def test_clarifying_questions_route_clarification():
    s1 = _s1(diy_suitability="suitable", clarifying_questions=["What material is it?"])
    assert _determine_output_mode(s1) == "clarification"


def test_care_mode_still_reachable():
    # low/low/suitable/general-other/no hazard, benign wording -> care
    s1 = _s1(problem_type="Scuffed leather shoe", trade_category="general/other",
             severity="low", risk_level="low", hidden_damage_risk="low",
             diy_suitability="suitable", safety_concerns=[])
    assert _determine_output_mode(s1) == "care"


def test_hidden_damage_high_routes_verify_first():
    s1 = _s1(diy_suitability="suitable", risk_level="low", hidden_damage_risk="high")
    assert _determine_output_mode(s1) == "verify_first"


def test_invariant_not_recommended_never_diy_verify_care():
    # Sweep risk/hidden/severity/trade combinations: not_recommended must always
    # produce assessment_first.
    bad = []
    for risk in ("low", "medium", "high", "critical"):
        for hidden in ("low", "medium", "high"):
            for sev in ("low", "medium", "high"):
                for trade in ("glazing", "plumbing", "general/other"):
                    s1 = _s1(diy_suitability="not_recommended", risk_level=risk,
                             hidden_damage_risk=hidden, severity=sev,
                             trade_category=trade, safety_concerns=[])
                    m = _determine_output_mode(s1)
                    if m != "assessment_first":
                        bad.append((risk, hidden, sev, trade, m))
    assert not bad, f"not_recommended leaked to non-assessment modes: {bad[:5]}"


_TESTS = [
    ("1 not_recommended + empty hazards -> assessment_first", test_not_recommended_empty_hazards_routes_assessment_first),
    ("2 not_recommended + hazard -> assessment_first",        test_not_recommended_with_hazard_routes_assessment_first),
    ("3 high risk + suitable + low hidden -> verify_first",   test_high_risk_suitable_low_hidden_routes_verify_first),
    ("3b critical risk -> verify_first",                      test_critical_risk_routes_at_least_verify_first),
    ("3c conditional+high risk -> clarification (stronger)",  test_conditional_still_clarification_even_if_high_risk),
    ("4 low-risk suitable -> diy",                            test_normal_low_risk_suitable_routes_diy),
    ("5 clarifying questions -> clarification",               test_clarifying_questions_route_clarification),
    ("6 care still reachable",                                test_care_mode_still_reachable),
    ("7 hidden=high -> verify_first",                         test_hidden_damage_high_routes_verify_first),
    ("8 invariant: not_recommended is ALWAYS assessment_first", test_invariant_not_recommended_never_diy_verify_care),
]


def _run() -> bool:
    print("=" * 64)
    print("OUTPUT-MODE ROUTING — PROFESSIONAL-REQUIRED FIX TESTS")
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
