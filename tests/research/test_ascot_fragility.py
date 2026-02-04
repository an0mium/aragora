"""Tests for ASCoTFragilityAnalyzer."""

import pytest
from aragora.debate.ascot_fragility import (
    ASCoTFragilityAnalyzer,
    FragilityConfig,
    FragilityScore,
    calculate_fragility,
    create_fragility_analyzer,
)


class TestASCoTFragilityAnalyzer:
    """Test suite for ASCoTFragilityAnalyzer."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        analyzer = ASCoTFragilityAnalyzer()
        assert analyzer.config.lambda_factor == 2.0
        assert analyzer.config.critical_threshold == 0.8
        assert analyzer.config.base_error_rate == 0.05

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = FragilityConfig(
            lambda_factor=3.0,
            critical_threshold=0.9,
            base_error_rate=0.1,
        )
        analyzer = ASCoTFragilityAnalyzer(config)
        assert analyzer.config.lambda_factor == 3.0
        assert analyzer.config.critical_threshold == 0.9

    def test_early_round_low_fragility(self) -> None:
        """Test that early rounds have low fragility."""
        analyzer = ASCoTFragilityAnalyzer()

        fragility = analyzer.calculate_round_fragility(
            round_number=1,
            total_rounds=10,
        )

        assert fragility.base_fragility < 0.3
        assert fragility.scrutiny_level in ["LOW", "MEDIUM"]
        assert fragility.combined_fragility < 0.4

    def test_late_round_high_fragility(self) -> None:
        """Test that late rounds have high fragility."""
        analyzer = ASCoTFragilityAnalyzer()

        fragility = analyzer.calculate_round_fragility(
            round_number=9,
            total_rounds=10,
        )

        assert fragility.base_fragility > 0.7
        assert fragility.scrutiny_level in ["HIGH", "CRITICAL"]
        assert fragility.combined_fragility > 0.6

    def test_final_round_high_scrutiny(self) -> None:
        """Test that final round has high scrutiny."""
        analyzer = ASCoTFragilityAnalyzer()

        fragility = analyzer.calculate_round_fragility(
            round_number=10,
            total_rounds=10,
        )

        assert fragility.base_fragility > 0.8
        # Final round should be HIGH or CRITICAL depending on error risk
        assert fragility.scrutiny_level in ["HIGH", "CRITICAL"]

    def test_dependency_increases_error_risk(self) -> None:
        """Test that more dependencies increase error risk."""
        analyzer = ASCoTFragilityAnalyzer()

        # Same round position, different dependencies
        fragility_few_deps = analyzer.calculate_round_fragility(
            round_number=5,
            total_rounds=10,
            dependencies=[1],
        )

        fragility_many_deps = analyzer.calculate_round_fragility(
            round_number=5,
            total_rounds=10,
            dependencies=[1, 2, 3, 4],
        )

        assert fragility_many_deps.error_risk > fragility_few_deps.error_risk
        assert fragility_many_deps.dependency_depth > fragility_few_deps.dependency_depth

    def test_scrutiny_level_thresholds(self) -> None:
        """Test that scrutiny levels match thresholds."""
        config = FragilityConfig(
            critical_threshold=0.8,
            high_threshold=0.6,
            medium_threshold=0.3,
        )
        analyzer = ASCoTFragilityAnalyzer(config)

        # Test across different positions
        results = []
        for round_num in range(1, 11):
            fragility = analyzer.calculate_round_fragility(
                round_number=round_num,
                total_rounds=10,
            )
            results.append((round_num, fragility.scrutiny_level, fragility.combined_fragility))

        # Early rounds should be LOW or MEDIUM
        early_scrutiny = [r[1] for r in results[:3]]
        assert all(s in ["LOW", "MEDIUM"] for s in early_scrutiny)

        # Late rounds should be HIGH or CRITICAL
        late_scrutiny = [r[1] for r in results[-2:]]
        assert all(s in ["HIGH", "CRITICAL"] for s in late_scrutiny)

    def test_verification_intensity_low(self) -> None:
        """Test verification config for LOW scrutiny."""
        analyzer = ASCoTFragilityAnalyzer()

        fragility = FragilityScore(
            round_number=1,
            total_rounds=10,
            base_fragility=0.1,
            dependency_depth=0,
            error_risk=0.0,
            scrutiny_level="LOW",
            combined_fragility=0.1,
        )

        config = analyzer.get_verification_intensity(fragility)

        assert config["formal_verification"] is False
        assert config["evidence_check"] is False
        assert config["critique_weight_boost"] == 1.0

    def test_verification_intensity_critical(self) -> None:
        """Test verification config for CRITICAL scrutiny."""
        analyzer = ASCoTFragilityAnalyzer()

        fragility = FragilityScore(
            round_number=10,
            total_rounds=10,
            base_fragility=0.9,
            dependency_depth=9,
            error_risk=0.4,
            scrutiny_level="CRITICAL",
            combined_fragility=0.85,
        )

        config = analyzer.get_verification_intensity(fragility)

        assert config["formal_verification"] is True
        assert config["evidence_check"] is True
        assert config["critique_weight_boost"] == 2.0
        assert config["require_multi_agent_agreement"] is True

    def test_is_in_fragile_zone(self) -> None:
        """Test fragile zone detection."""
        analyzer = ASCoTFragilityAnalyzer()

        # Early round not in fragile zone
        assert not analyzer.is_in_fragile_zone(
            round_number=2,
            total_rounds=10,
            threshold=0.6,
        )

        # Late round in fragile zone
        assert analyzer.is_in_fragile_zone(
            round_number=9,
            total_rounds=10,
            threshold=0.6,
        )

    def test_get_fragility_for_stability_gate(self) -> None:
        """Test integration point with stability detector."""
        analyzer = ASCoTFragilityAnalyzer()

        fragility = analyzer.get_fragility_for_stability_gate(
            round_number=7,
            total_rounds=10,
        )

        assert isinstance(fragility, float)
        assert 0.0 <= fragility <= 1.0

    def test_reset_clears_history(self) -> None:
        """Test that reset clears internal state."""
        analyzer = ASCoTFragilityAnalyzer()

        # Generate some history
        for round_num in range(1, 6):
            analyzer.calculate_round_fragility(round_num, 10)

        metrics = analyzer.get_metrics()
        assert metrics["total_rounds_analyzed"] == 5

        analyzer.reset()

        metrics = analyzer.get_metrics()
        assert metrics["total_rounds_analyzed"] == 0

    def test_get_metrics(self) -> None:
        """Test metrics retrieval."""
        analyzer = ASCoTFragilityAnalyzer()

        # Analyze several rounds
        for round_num in range(1, 11):
            analyzer.calculate_round_fragility(round_num, 10)

        metrics = analyzer.get_metrics()

        assert metrics["total_rounds_analyzed"] == 10
        assert 0.0 < metrics["avg_fragility"] < 1.0
        assert metrics["max_fragility"] > metrics["min_fragility"]
        # At least some rounds should have high scrutiny
        assert metrics["high_rounds"] > 0 or metrics["critical_rounds"] > 0

    def test_handles_zero_total_rounds(self) -> None:
        """Test handling of edge case with zero total rounds."""
        analyzer = ASCoTFragilityAnalyzer()

        # Should not crash
        fragility = analyzer.calculate_round_fragility(
            round_number=1,
            total_rounds=0,
        )

        assert isinstance(fragility.combined_fragility, float)

    def test_handles_negative_round(self) -> None:
        """Test handling of edge case with negative round number."""
        analyzer = ASCoTFragilityAnalyzer()

        # Should not crash
        fragility = analyzer.calculate_round_fragility(
            round_number=-1,
            total_rounds=10,
        )

        assert isinstance(fragility.combined_fragility, float)


class TestCalculateFragility:
    """Test the convenience calculate_fragility function."""

    def test_basic_calculation(self) -> None:
        """Test basic fragility calculation."""
        fragility = calculate_fragility(
            round_number=5,
            total_rounds=10,
        )

        assert 0.0 < fragility < 1.0

    def test_early_vs_late(self) -> None:
        """Test that early rounds have lower fragility than late."""
        early = calculate_fragility(2, 10)
        late = calculate_fragility(9, 10)

        assert early < late

    def test_lambda_factor_effect(self) -> None:
        """Test that higher lambda increases fragility curve steepness."""
        fragility_low_lambda = calculate_fragility(5, 10, lambda_factor=1.0)
        fragility_high_lambda = calculate_fragility(5, 10, lambda_factor=4.0)

        # Higher lambda = steeper curve = higher fragility at mid-point
        assert fragility_high_lambda > fragility_low_lambda


class TestCreateFragilityAnalyzer:
    """Test the factory function."""

    def test_creates_analyzer_with_defaults(self) -> None:
        """Test factory creates analyzer with defaults."""
        analyzer = create_fragility_analyzer()

        assert isinstance(analyzer, ASCoTFragilityAnalyzer)
        assert analyzer.config.lambda_factor == 2.0

    def test_creates_analyzer_with_custom_config(self) -> None:
        """Test factory accepts custom configuration."""
        analyzer = create_fragility_analyzer(
            lambda_factor=3.0,
            critical_threshold=0.9,
        )

        assert analyzer.config.lambda_factor == 3.0
        assert analyzer.config.critical_threshold == 0.9
