"""Tests for aragora.debate.ascot_fragility — ASCoT Fragility Analysis."""

from __future__ import annotations

import pytest
import numpy as np

from aragora.debate.ascot_fragility import (
    ASCoTFragilityAnalyzer,
    FragilityConfig,
    FragilityScore,
    calculate_fragility,
    create_fragility_analyzer,
)


# ---------------------------------------------------------------------------
# FragilityScore dataclass
# ---------------------------------------------------------------------------


class TestFragilityScore:
    def test_fields(self):
        s = FragilityScore(
            round_number=3,
            total_rounds=10,
            base_fragility=0.5,
            dependency_depth=2,
            error_risk=0.1,
            scrutiny_level="MEDIUM",
            combined_fragility=0.4,
        )
        assert s.round_number == 3
        assert s.total_rounds == 10
        assert s.base_fragility == 0.5
        assert s.dependency_depth == 2
        assert s.error_risk == pytest.approx(0.1)
        assert s.scrutiny_level == "MEDIUM"
        assert s.combined_fragility == pytest.approx(0.4)

    def test_default_combined_fragility(self):
        s = FragilityScore(
            round_number=1,
            total_rounds=5,
            base_fragility=0.2,
            dependency_depth=0,
            error_risk=0.0,
            scrutiny_level="LOW",
        )
        assert s.combined_fragility == 0.0


# ---------------------------------------------------------------------------
# FragilityConfig
# ---------------------------------------------------------------------------


class TestFragilityConfig:
    def test_defaults(self):
        c = FragilityConfig()
        assert c.lambda_factor == 2.0
        assert c.base_error_rate == 0.05
        assert c.critical_threshold == 0.8
        assert c.high_threshold == 0.6
        assert c.medium_threshold == 0.3
        assert c.dependency_weight == pytest.approx(0.4)
        assert c.position_weight == pytest.approx(0.6)

    def test_custom(self):
        c = FragilityConfig(lambda_factor=3.0, base_error_rate=0.1)
        assert c.lambda_factor == 3.0
        assert c.base_error_rate == 0.1


# ---------------------------------------------------------------------------
# ASCoTFragilityAnalyzer — calculate_round_fragility
# ---------------------------------------------------------------------------


class TestCalculateRoundFragility:
    def setup_method(self):
        self.analyzer = ASCoTFragilityAnalyzer()

    def test_first_round_low_fragility(self):
        f = self.analyzer.calculate_round_fragility(1, 10)
        assert f.base_fragility < 0.3
        assert f.scrutiny_level in ("LOW", "MEDIUM")

    def test_last_round_high_fragility(self):
        f = self.analyzer.calculate_round_fragility(10, 10)
        assert f.base_fragility > 0.7
        assert f.scrutiny_level in ("HIGH", "CRITICAL")

    def test_fragility_increases_with_round(self):
        f1 = self.analyzer.calculate_round_fragility(2, 10)
        f5 = self.analyzer.calculate_round_fragility(5, 10)
        f9 = self.analyzer.calculate_round_fragility(9, 10)
        assert f1.combined_fragility < f5.combined_fragility < f9.combined_fragility

    def test_explicit_dependencies(self):
        f_no_deps = self.analyzer.calculate_round_fragility(5, 10, dependencies=[])
        f_many_deps = self.analyzer.calculate_round_fragility(5, 10, dependencies=[1, 2, 3, 4])
        assert f_many_deps.error_risk > f_no_deps.error_risk
        assert f_many_deps.combined_fragility >= f_no_deps.combined_fragility

    def test_no_explicit_dependencies_uses_prior_rounds(self):
        f = self.analyzer.calculate_round_fragility(5, 10, dependencies=None)
        assert f.dependency_depth == 4  # round 5 depends on 1-4

    def test_zero_total_rounds_handled(self):
        f = self.analyzer.calculate_round_fragility(1, 0)
        assert f.total_rounds == 1  # Clamped to 1
        assert f.base_fragility >= 0

    def test_zero_round_number_handled(self):
        f = self.analyzer.calculate_round_fragility(0, 10)
        assert f.round_number == 1  # Clamped to 1

    def test_stores_in_history(self):
        self.analyzer.calculate_round_fragility(1, 5)
        self.analyzer.calculate_round_fragility(2, 5)
        assert len(self.analyzer._fragility_history) == 2

    def test_custom_config(self):
        config = FragilityConfig(lambda_factor=5.0, base_error_rate=0.2)
        analyzer = ASCoTFragilityAnalyzer(config)
        f = analyzer.calculate_round_fragility(5, 10)
        # Higher lambda → higher base fragility
        default_f = self.analyzer.calculate_round_fragility(5, 10)
        assert f.base_fragility > default_f.base_fragility


# ---------------------------------------------------------------------------
# _determine_scrutiny_level
# ---------------------------------------------------------------------------


class TestDetermineScrutinyLevel:
    def setup_method(self):
        self.analyzer = ASCoTFragilityAnalyzer()

    def test_low(self):
        assert self.analyzer._determine_scrutiny_level(0.1) == "LOW"

    def test_medium(self):
        assert self.analyzer._determine_scrutiny_level(0.4) == "MEDIUM"

    def test_high(self):
        assert self.analyzer._determine_scrutiny_level(0.7) == "HIGH"

    def test_critical(self):
        assert self.analyzer._determine_scrutiny_level(0.9) == "CRITICAL"

    def test_boundary_medium(self):
        assert self.analyzer._determine_scrutiny_level(0.3) == "MEDIUM"

    def test_boundary_high(self):
        assert self.analyzer._determine_scrutiny_level(0.6) == "HIGH"

    def test_boundary_critical(self):
        assert self.analyzer._determine_scrutiny_level(0.8) == "CRITICAL"


# ---------------------------------------------------------------------------
# get_verification_intensity
# ---------------------------------------------------------------------------


class TestGetVerificationIntensity:
    def setup_method(self):
        self.analyzer = ASCoTFragilityAnalyzer()

    def test_low_scrutiny(self):
        f = FragilityScore(
            round_number=1,
            total_rounds=10,
            base_fragility=0.1,
            dependency_depth=0,
            error_risk=0.0,
            scrutiny_level="LOW",
            combined_fragility=0.1,
        )
        config = self.analyzer.get_verification_intensity(f)
        assert config["formal_verification"] is False
        assert config["evidence_check"] is False
        assert config["fragility_score"] == pytest.approx(0.1)
        assert config["round_number"] == 1

    def test_critical_scrutiny(self):
        f = FragilityScore(
            round_number=10,
            total_rounds=10,
            base_fragility=0.9,
            dependency_depth=9,
            error_risk=0.4,
            scrutiny_level="CRITICAL",
            combined_fragility=0.9,
        )
        config = self.analyzer.get_verification_intensity(f)
        assert config["formal_verification"] is True
        assert config["require_multi_agent_agreement"] is True
        assert config["critique_weight_boost"] == 2.0
        assert config["timeout_seconds"] == 180

    def test_returns_copy(self):
        f = FragilityScore(
            round_number=1,
            total_rounds=10,
            base_fragility=0.1,
            dependency_depth=0,
            error_risk=0.0,
            scrutiny_level="LOW",
            combined_fragility=0.1,
        )
        config1 = self.analyzer.get_verification_intensity(f)
        config2 = self.analyzer.get_verification_intensity(f)
        config1["extra_key"] = True
        assert "extra_key" not in config2

    def test_unknown_scrutiny_falls_back_to_medium(self):
        f = FragilityScore(
            round_number=1,
            total_rounds=10,
            base_fragility=0.1,
            dependency_depth=0,
            error_risk=0.0,
            scrutiny_level="UNKNOWN",
            combined_fragility=0.1,
        )
        config = self.analyzer.get_verification_intensity(f)
        # Falls back to MEDIUM
        assert config["evidence_check"] is True
        assert config["formal_verification"] is False


# ---------------------------------------------------------------------------
# is_in_fragile_zone
# ---------------------------------------------------------------------------


class TestIsInFragileZone:
    def setup_method(self):
        self.analyzer = ASCoTFragilityAnalyzer()

    def test_early_round_not_fragile(self):
        assert self.analyzer.is_in_fragile_zone(1, 10) is False

    def test_late_round_fragile(self):
        assert self.analyzer.is_in_fragile_zone(10, 10) is True

    def test_custom_threshold(self):
        assert self.analyzer.is_in_fragile_zone(5, 10, threshold=0.3) is True


# ---------------------------------------------------------------------------
# get_fragility_for_stability_gate
# ---------------------------------------------------------------------------


class TestGetFragilityForStabilityGate:
    def setup_method(self):
        self.analyzer = ASCoTFragilityAnalyzer()

    def test_returns_float(self):
        score = self.analyzer.get_fragility_for_stability_gate(5, 10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_increases_with_round(self):
        s1 = self.analyzer.get_fragility_for_stability_gate(1, 10)
        s9 = self.analyzer.get_fragility_for_stability_gate(9, 10)
        assert s9 > s1


# ---------------------------------------------------------------------------
# reset / get_metrics
# ---------------------------------------------------------------------------


class TestResetAndMetrics:
    def setup_method(self):
        self.analyzer = ASCoTFragilityAnalyzer()

    def test_empty_metrics(self):
        m = self.analyzer.get_metrics()
        assert m["total_rounds_analyzed"] == 0
        assert m["avg_fragility"] == 0.0
        assert m["max_fragility"] == 0.0
        assert m["critical_rounds"] == 0

    def test_metrics_after_analysis(self):
        for i in range(1, 11):
            self.analyzer.calculate_round_fragility(i, 10)
        m = self.analyzer.get_metrics()
        assert m["total_rounds_analyzed"] == 10
        assert m["avg_fragility"] > 0
        assert m["max_fragility"] > m["avg_fragility"]
        assert m["min_fragility"] < m["avg_fragility"]

    def test_reset_clears_history(self):
        self.analyzer.calculate_round_fragility(1, 5)
        self.analyzer.reset()
        assert len(self.analyzer._fragility_history) == 0
        m = self.analyzer.get_metrics()
        assert m["total_rounds_analyzed"] == 0


# ---------------------------------------------------------------------------
# calculate_fragility (convenience function)
# ---------------------------------------------------------------------------


class TestCalculateFragilityFunction:
    def test_basic(self):
        score = calculate_fragility(5, 10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_increases_with_round(self):
        s1 = calculate_fragility(1, 10)
        s9 = calculate_fragility(9, 10)
        assert s9 > s1

    def test_zero_total_rounds(self):
        assert calculate_fragility(1, 0) == 0.0

    def test_custom_lambda(self):
        low_lambda = calculate_fragility(5, 10, lambda_factor=1.0)
        high_lambda = calculate_fragility(5, 10, lambda_factor=5.0)
        assert high_lambda > low_lambda


# ---------------------------------------------------------------------------
# create_fragility_analyzer (factory)
# ---------------------------------------------------------------------------


class TestCreateFragilityAnalyzer:
    def test_default(self):
        analyzer = create_fragility_analyzer()
        assert isinstance(analyzer, ASCoTFragilityAnalyzer)
        assert analyzer.config.lambda_factor == 2.0

    def test_custom(self):
        analyzer = create_fragility_analyzer(lambda_factor=3.0, critical_threshold=0.9)
        assert analyzer.config.lambda_factor == 3.0
        assert analyzer.config.critical_threshold == 0.9

    def test_kwargs_passthrough(self):
        analyzer = create_fragility_analyzer(base_error_rate=0.1)
        assert analyzer.config.base_error_rate == 0.1
