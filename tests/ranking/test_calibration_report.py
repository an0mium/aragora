"""Tests for the auditable calibration report builder (issue #8229, ODR-5).

Covers:
- Honest absence: agents with no calibration data get explicit
  ``{"status": "absent", "reason": ...}`` — never invented numbers.
- Sample-size disclosure: every numeric block carries ``sample_size``.
- Read-only aggregation over the existing calibration stores.
- ODR provenance builder: emitted only when calibration data exists.
"""

from __future__ import annotations

import json

import pytest

from aragora.ranking.calibration_report import (
    CALIBRATION_REPORT_ENDPOINT_TEMPLATE,
    build_calibration_report,
    build_odr_calibration_provenance,
)
from aragora.ranking.elo import EloSystem


@pytest.fixture()
def elo(tmp_path) -> EloSystem:
    system = EloSystem(db_path=tmp_path / "elo.db")
    # Class-level TTL caches are shared across instances; isolate the test.
    EloSystem._rating_cache.clear()
    return system


@pytest.fixture()
def elo_with_data(elo: EloSystem) -> EloSystem:
    engine = elo._domain_calibration_engine
    engine.record_prediction("claude", "security", 0.8, correct=True)
    engine.record_prediction("claude", "security", 0.7, correct=False)
    engine.record_prediction("claude", "performance", 0.9, correct=True)
    return elo


class TestAbsenceContract:
    def test_unknown_agent_is_absent(self, elo: EloSystem) -> None:
        report = build_calibration_report("ghost-agent", elo_system=elo)
        assert report["status"] == "absent"
        assert report["sample_size"] == 0
        assert "no calibration data" in report["reason"]
        assert report["agent"] == "ghost-agent"

    def test_absent_report_has_no_invented_figures(self, elo: EloSystem) -> None:
        report = build_calibration_report("ghost-agent", elo_system=elo)
        for fabricated in ("overall", "domains", "calibration_curve", "expected_calibration_error"):
            assert fabricated not in report

    def test_rated_agent_without_calibration_is_absent(self, elo: EloSystem) -> None:
        # An agent can have an ELO rating row but zero resolved predictions.
        rating = elo.get_rating("rated-only")
        elo._save_rating(rating)
        EloSystem._rating_cache.clear()
        report = build_calibration_report("rated-only", elo_system=elo)
        assert report["status"] == "absent"
        assert report["sample_size"] == 0

    def test_absent_report_is_json_serializable(self, elo: EloSystem) -> None:
        report = build_calibration_report("ghost-agent", elo_system=elo)
        assert json.loads(json.dumps(report)) == report


class TestReportContent:
    def test_status_ok_with_data(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        assert report["status"] == "ok"
        assert report["agent"] == "claude"
        assert report["endpoint"] == "/api/v1/agents/claude/calibration-report"

    def test_sample_size_disclosed_on_every_block(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        assert report["sample_size"] == 3
        assert report["overall"]["sample_size"] == 3
        assert report["domains"]["sample_size"] == 3
        assert report["calibration_curve"]["sample_size"] == 3
        assert report["expected_calibration_error"]["sample_size"] == 3
        for bucket in report["calibration_curve"]["buckets"]:
            assert "sample_size" in bucket
        for domain_stats in report["domains"]["by_domain"].values():
            assert "sample_size" in domain_stats

    def test_per_domain_breakdown(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        by_domain = report["domains"]["by_domain"]
        assert by_domain["security"]["predictions"] == 2
        assert by_domain["security"]["correct"] == 1
        assert by_domain["security"]["accuracy"] == pytest.approx(0.5)
        assert by_domain["performance"]["predictions"] == 1
        assert by_domain["performance"]["accuracy"] == pytest.approx(1.0)

    def test_overall_accuracy_and_brier_from_existing_store(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        overall = report["overall"]
        assert overall["predictions"] == 3
        assert overall["correct"] == 2
        assert overall["accuracy"] == pytest.approx(2 / 3)
        # Brier: ((0.8-1)^2 + (0.7-0)^2 + (0.9-1)^2) / 3
        expected_brier = (0.04 + 0.49 + 0.01) / 3
        assert overall["brier_score"] == pytest.approx(expected_brier)

    def test_minimum_sample_disclosure(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        overall = report["overall"]
        assert "meets_minimum_sample" in overall
        assert "minimum_sample_threshold" in overall

    def test_domain_filter(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data, domain="security")
        assert report["domain_filter"] == "security"
        assert set(report["domains"]["by_domain"]) == {"security"}

    def test_data_window_present(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        window = report["data_window"]
        # Domain predictions are timestamped via domain_calibration.updated_at.
        assert window["domain_calibration_last_updated_at"] is not None
        # No tournament predictions recorded -> explicit absent, not fabricated.
        assert window["tournament_predictions"]["status"] == "absent"

    def test_sources_disclosed(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        assert any("calibration" in s for s in report["sources"])

    def test_report_is_json_serializable(self, elo_with_data: EloSystem) -> None:
        report = build_calibration_report("claude", elo_system=elo_with_data)
        assert json.loads(json.dumps(report))["status"] == "ok"

    def test_tournament_prediction_window(self, elo_with_data: EloSystem) -> None:
        engine = elo_with_data._calibration_engine
        engine.record_winner_prediction("t-1", "claude", "claude", 0.9)
        engine.resolve_tournament("t-1", "claude")
        EloSystem._rating_cache.clear()
        report = build_calibration_report("claude", elo_system=elo_with_data)
        window = report["data_window"]["tournament_predictions"]
        assert window["sample_size"] == 1
        assert window["first_recorded_at"] is not None


class TestOdrProvenance:
    def test_no_agents_returns_none(self, elo: EloSystem) -> None:
        assert build_odr_calibration_provenance([], elo_system=elo) is None

    def test_no_calibration_data_returns_none(self, elo: EloSystem) -> None:
        assert build_odr_calibration_provenance(["ghost"], elo_system=elo) is None

    def test_provenance_only_for_agents_with_data(self, elo_with_data: EloSystem) -> None:
        prov = build_odr_calibration_provenance(["claude", "ghost-agent"], elo_system=elo_with_data)
        assert prov is not None
        assert prov["type"] == "aragora.calibration_report"
        assert prov["endpoint_template"] == CALIBRATION_REPORT_ENDPOINT_TEMPLATE
        agents = prov["agents"]
        assert [row["agent"] for row in agents] == ["claude"]
        assert agents[0]["sample_size"] == 3
        assert agents[0]["report_ref"] == "/api/v1/agents/claude/calibration-report"

    def test_invalid_agent_names_skipped(self, elo_with_data: EloSystem) -> None:
        prov = build_odr_calibration_provenance(
            ["claude", "", "   ", "../../etc/passwd"], elo_system=elo_with_data
        )
        assert prov is not None
        assert [row["agent"] for row in prov["agents"]] == ["claude"]
