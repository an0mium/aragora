"""Tests for DIC-25 adversarial world-state stress-test (#6219).

No network, queue, or database access.
"""

from __future__ import annotations

import pytest

from aragora.epistemic.stress_test import (
    FragilityReport,
    StressPerturbation,
    StressTestResult,
    _probe_unit,
    _recommended_action,
    run_stress_test,
    stress_test_enabled,
)


def _p(pid: str = "p1", impact: float = 0.4, units: list[str] | None = None) -> StressPerturbation:
    return StressPerturbation(
        pid, "cve_drop", "Synthetic.", impact, affected_proof_unit_ids=units or []
    )


# --- Flag gate ---


def test_disabled_by_default() -> None:
    with pytest.raises(RuntimeError, match="ARAGORA_STRESS_TEST_ENABLED"):
        run_stress_test([], {})


def test_enabled_via_kwarg() -> None:
    assert run_stress_test([], {}, enabled=True).perturbations_tested == 0


def test_env_var_truthiness(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv("ARAGORA_STRESS_TEST_ENABLED", val)
        assert stress_test_enabled() is True
    monkeypatch.setenv("ARAGORA_STRESS_TEST_ENABLED", "0")
    assert stress_test_enabled() is False


# --- _probe_unit ---


def test_probe_reduces_integrity() -> None:
    rep = _probe_unit("u1", 0.8, _p(impact=0.3, units=["u1"]))
    assert rep.stressed_integrity == pytest.approx(0.5, abs=1e-4)
    assert rep.fragility_delta == pytest.approx(0.3, abs=1e-4)


def test_probe_floors_at_zero() -> None:
    rep = _probe_unit("u1", 0.6, _p(impact=2.0, units=["u1"]))
    assert rep.stressed_integrity == 0.0
    assert rep.fragility_delta == pytest.approx(0.6, abs=1e-4)


def test_probe_out_of_scope_unchanged() -> None:
    rep = _probe_unit("u1", 0.75, _p(impact=0.9, units=["other"]))
    assert rep.fragility_delta == 0.0 and rep.reason == "not in perturbation scope"


def test_probe_empty_scope_hits_all() -> None:
    assert _probe_unit("any", 0.9, _p(impact=0.35, units=[])).fragility_delta == pytest.approx(
        0.35, abs=1e-4
    )


# --- _recommended_action ---


def test_recommended_actions() -> None:
    assert _recommended_action(0.1, 0.25) == "fail_closed"  # stressed < 0.3
    assert _recommended_action(0.45, 0.55) == "repair_required"
    assert _recommended_action(0.25, 0.7) == "monitor"
    assert _recommended_action(0.05, 0.9) == "pass"


# --- run_stress_test ---


def test_identifies_most_fragile_unit() -> None:
    result = run_stress_test(
        [_p("p1", 0.6, ["u1"]), _p("p2", 0.15, ["u2"])],
        {"u1": 0.9, "u2": 0.8},
        enabled=True,
    )
    assert result.most_fragile_unit_id == "u1"
    assert result.max_fragility_delta == pytest.approx(0.6, abs=1e-4)


def test_empty_run() -> None:
    r = run_stress_test([], {}, enabled=True)
    assert r.max_fragility_delta == 0.0 and r.most_fragile_unit_id == ""


def test_high_fragility_filter() -> None:
    reports = [
        FragilityReport("u1", "p1", 0.9, 0.4, 0.5, "t", "repair_required"),
        FragilityReport("u2", "p1", 0.8, 0.75, 0.05, "t", "pass"),
    ]
    result = StressTestResult(
        1, 2, reports=reports, most_fragile_unit_id="u1", max_fragility_delta=0.5
    )
    assert [r.proof_unit_id for r in result.high_fragility_units] == ["u1"]


# --- Serialisation ---


def test_serialisation_round_trips() -> None:
    result = run_stress_test([_p(impact=0.4, units=["u1"])], {"u1": 0.9}, enabled=True)
    rd = result.to_dict()
    assert rd["max_fragility_delta"] > 0 and len(rd["reports"]) == 1
    pd = result.reports[0].to_dict()
    assert {"proof_unit_id", "fragility_delta", "recommended_action"} <= pd.keys()
    sp = StressPerturbation("p1", "corpus_revision", "d", 0.25, [], ["u3"])
    sd = sp.to_dict()
    assert sd["kind"] == "corpus_revision" and "u3" in sd["affected_proof_unit_ids"]
