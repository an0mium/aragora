"""Tests for DIC-25 adversarial world-state stress-test (#6219).

All tests run without network, queue, or database access.
Flag gate is exercised via ``enabled`` kwarg to avoid env-var pollution.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perturb(
    pid: str = "p-001",
    kind: str = "cve_drop",
    impact: float = 0.4,
    affected_units: list[str] | None = None,
    affected_claims: list[str] | None = None,
    source: str = "CVE-2026-0001",
) -> StressPerturbation:
    return StressPerturbation(
        perturbation_id=pid,
        kind=kind,  # type: ignore[arg-type]
        description="Synthetic perturbation for testing.",
        simulated_impact=impact,
        affected_proof_unit_ids=affected_units or [],
        affected_claim_ids=affected_claims or [],
        source=source,
    )


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------


def test_stress_test_disabled_by_default() -> None:
    """Gate must be off unless the env var or kwarg enables it."""
    with pytest.raises(RuntimeError, match="ARAGORA_STRESS_TEST_ENABLED"):
        run_stress_test([], {})


def test_stress_test_enabled_via_kwarg() -> None:
    result = run_stress_test([], {}, enabled=True)
    assert result.perturbations_tested == 0
    assert result.proof_units_probed == 0


def test_stress_test_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("ARAGORA_STRESS_TEST_ENABLED", truthy)
        assert stress_test_enabled() is True
    monkeypatch.setenv("ARAGORA_STRESS_TEST_ENABLED", "0")
    assert stress_test_enabled() is False


def test_stress_test_override_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_STRESS_TEST_ENABLED", "0")
    # explicit True wins over env-var False
    result = run_stress_test([], {}, enabled=True)
    assert isinstance(result, StressTestResult)


# ---------------------------------------------------------------------------
# _probe_unit
# ---------------------------------------------------------------------------


def test_probe_unit_reduces_integrity_when_in_scope() -> None:
    p = _perturb(impact=0.3, affected_units=["u1"])
    report = _probe_unit("u1", 0.8, p)
    assert report.stressed_integrity == pytest.approx(0.5, abs=1e-4)
    assert report.fragility_delta == pytest.approx(0.3, abs=1e-4)


def test_probe_unit_floors_at_zero() -> None:
    p = _perturb(impact=2.0, affected_units=["u1"])
    report = _probe_unit("u1", 0.6, p)
    assert report.stressed_integrity == 0.0
    assert report.fragility_delta == pytest.approx(0.6, abs=1e-4)


def test_probe_unit_out_of_scope_unchanged() -> None:
    p = _perturb(impact=0.9, affected_units=["other"])
    report = _probe_unit("u1", 0.75, p)
    assert report.fragility_delta == 0.0
    assert report.stressed_integrity == pytest.approx(0.75, abs=1e-4)
    assert report.reason == "not in perturbation scope"


def test_probe_unit_empty_scope_applies_to_all() -> None:
    """Empty affected_proof_unit_ids means every unit is in scope."""
    p = _perturb(impact=0.35, affected_units=[])
    report = _probe_unit("any-unit", 0.9, p)
    assert report.fragility_delta == pytest.approx(0.35, abs=1e-4)


# ---------------------------------------------------------------------------
# _recommended_action
# ---------------------------------------------------------------------------


def test_action_fail_closed_on_very_low_stressed_integrity() -> None:
    # stressed below 0.3 threshold → fail_closed regardless of delta
    assert _recommended_action(0.1, 0.25) == "fail_closed"


def test_action_repair_required_on_high_delta() -> None:
    assert _recommended_action(0.45, 0.55) == "repair_required"


def test_action_monitor_on_moderate_delta() -> None:
    assert _recommended_action(0.25, 0.7) == "monitor"


def test_action_pass_on_low_delta() -> None:
    assert _recommended_action(0.05, 0.9) == "pass"


# ---------------------------------------------------------------------------
# run_stress_test aggregate
# ---------------------------------------------------------------------------


def test_run_stress_test_identifies_most_fragile_unit() -> None:
    perturbations = [
        _perturb(pid="p1", impact=0.6, affected_units=["u1"]),
        _perturb(pid="p2", impact=0.15, affected_units=["u2"]),
    ]
    result = run_stress_test(perturbations, {"u1": 0.9, "u2": 0.8}, enabled=True)
    assert result.perturbations_tested == 2
    assert result.proof_units_probed == 2
    assert result.most_fragile_unit_id == "u1"
    assert result.max_fragility_delta == pytest.approx(0.6, abs=1e-4)


def test_run_stress_test_report_count() -> None:
    """Result must have one report per perturbation × proof-unit pair."""
    perturbations = [_perturb(pid="p1"), _perturb(pid="p2")]
    integrities = {"u1": 0.9, "u2": 0.7, "u3": 0.5}
    result = run_stress_test(perturbations, integrities, enabled=True)
    assert len(result.reports) == 2 * 3


def test_run_stress_test_empty_returns_zero_delta() -> None:
    result = run_stress_test([], {}, enabled=True)
    assert result.max_fragility_delta == 0.0
    assert result.most_fragile_unit_id == ""


def test_high_fragility_filter() -> None:
    reports = [
        FragilityReport(
            proof_unit_id="u1",
            perturbation_id="p1",
            baseline_integrity=0.9,
            stressed_integrity=0.4,
            fragility_delta=0.5,
            reason="test",
            recommended_action="repair_required",
        ),
        FragilityReport(
            proof_unit_id="u2",
            perturbation_id="p1",
            baseline_integrity=0.8,
            stressed_integrity=0.75,
            fragility_delta=0.05,
            reason="test",
            recommended_action="pass",
        ),
    ]
    result = StressTestResult(
        perturbations_tested=1,
        proof_units_probed=2,
        reports=reports,
        most_fragile_unit_id="u1",
        max_fragility_delta=0.5,
    )
    high = result.high_fragility_units
    assert len(high) == 1
    assert high[0].proof_unit_id == "u1"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_perturbation_to_dict_round_trips() -> None:
    p = _perturb(kind="corpus_revision", impact=0.25, affected_units=["u3"])
    d = p.to_dict()
    assert d["kind"] == "corpus_revision"
    assert d["simulated_impact"] == 0.25
    assert "u3" in d["affected_proof_unit_ids"]


def test_fragility_report_to_dict() -> None:
    p = _perturb(impact=0.5, affected_units=["u1"])
    report = _probe_unit("u1", 0.85, p)
    d = report.to_dict()
    assert set(d) >= {
        "proof_unit_id",
        "perturbation_id",
        "baseline_integrity",
        "stressed_integrity",
        "fragility_delta",
        "reason",
        "recommended_action",
    }


def test_stress_test_result_to_dict() -> None:
    result = run_stress_test(
        [_perturb(impact=0.4, affected_units=["u1"])],
        {"u1": 0.9},
        enabled=True,
    )
    d = result.to_dict()
    assert d["perturbations_tested"] == 1
    assert d["proof_units_probed"] == 1
    assert len(d["reports"]) == 1
    assert d["max_fragility_delta"] > 0
