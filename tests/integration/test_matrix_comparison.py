"""
Tests for matrix debate comparison and conclusion extraction.

Phase 8: Debate Integration Test Gaps - Matrix comparison tests.

Tests:
- test_universal_conclusions_all_consensus - Universal detection
- test_universal_conclusions_partial_consensus - Empty when partial
- test_no_universal_when_divergent - Divergent scenarios
- test_conditional_conclusions_extraction - Parameter mapping
- test_conditional_confidence_tracking - Confidence per scenario
- test_baseline_scenario_handling - is_baseline flag behavior
"""

from __future__ import annotations

import pytest


# ============================================================================
# Matrix Comparison Logic (mirroring handler implementation)
# ============================================================================


class MatrixComparisonLogic:
    """Matrix comparison logic extracted for testing."""

    def find_universal_conclusions(self, results: list[dict]) -> list[str]:
        """Find conclusions that are consistent across all scenarios."""
        if not results:
            return []

        # Simple heuristic: if all scenarios reached consensus, that's universal
        consensus_results = [r for r in results if r.get("consensus_reached")]
        if len(consensus_results) == len(results):
            return ["All scenarios reached consensus"]

        return []

    def find_conditional_conclusions(self, results: list[dict]) -> list[dict]:
        """Find conclusions that depend on specific scenarios."""
        conditional = []
        for r in results:
            if r.get("final_answer"):
                conditional.append(
                    {
                        "condition": f"When {r['scenario_name']}",
                        "parameters": r.get("parameters", {}),
                        "conclusion": r["final_answer"],
                        "confidence": r.get("confidence", 0),
                    }
                )
        return conditional

    def build_comparison_matrix(self, results: list[dict]) -> dict:
        """Build a comparison matrix of scenarios."""
        return {
            "scenarios": [r["scenario_name"] for r in results],
            "consensus_rate": sum(1 for r in results if r.get("consensus_reached"))
            / max(len(results), 1),
            "avg_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
            "avg_rounds": sum(r.get("rounds_used", 0) for r in results) / max(len(results), 1),
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def comparison_logic():
    """Create a matrix comparison logic instance."""
    return MatrixComparisonLogic()


@pytest.fixture
def all_consensus_results():
    """Results where all scenarios reached consensus."""
    return [
        {
            "scenario_name": "Scenario A",
            "consensus_reached": True,
            "final_answer": "Answer A",
            "confidence": 0.9,
            "parameters": {"param1": "value1"},
            "rounds_used": 3,
        },
        {
            "scenario_name": "Scenario B",
            "consensus_reached": True,
            "final_answer": "Answer B",
            "confidence": 0.85,
            "parameters": {"param1": "value2"},
            "rounds_used": 4,
        },
        {
            "scenario_name": "Scenario C",
            "consensus_reached": True,
            "final_answer": "Answer C",
            "confidence": 0.95,
            "parameters": {"param1": "value3"},
            "rounds_used": 2,
        },
    ]


@pytest.fixture
def partial_consensus_results():
    """Results where only some scenarios reached consensus."""
    return [
        {
            "scenario_name": "Scenario A",
            "consensus_reached": True,
            "final_answer": "Answer A",
            "confidence": 0.9,
            "parameters": {},
            "rounds_used": 3,
        },
        {
            "scenario_name": "Scenario B",
            "consensus_reached": False,
            "final_answer": "Inconclusive",
            "confidence": 0.4,
            "parameters": {},
            "rounds_used": 5,
        },
        {
            "scenario_name": "Scenario C",
            "consensus_reached": True,
            "final_answer": "Answer C",
            "confidence": 0.8,
            "parameters": {},
            "rounds_used": 4,
        },
    ]


@pytest.fixture
def divergent_results():
    """Results where scenarios have completely divergent outcomes."""
    return [
        {
            "scenario_name": "Scenario A",
            "consensus_reached": False,
            "final_answer": None,
            "confidence": 0.3,
            "parameters": {},
            "rounds_used": 5,
        },
        {
            "scenario_name": "Scenario B",
            "consensus_reached": False,
            "final_answer": None,
            "confidence": 0.2,
            "parameters": {},
            "rounds_used": 5,
        },
    ]


# ============================================================================
# Test: Universal Conclusions - All Consensus
# ============================================================================


class TestUniversalConclusionsAllConsensus:
    """Test universal conclusion detection when all scenarios agree."""

    def test_universal_conclusions_all_consensus(self, comparison_logic, all_consensus_results):
        """Test that universal conclusions are detected when all scenarios reach consensus."""
        universal = comparison_logic.find_universal_conclusions(all_consensus_results)

        assert len(universal) == 1
        assert "All scenarios reached consensus" in universal[0]

    def test_universal_conclusions_empty_results(self, comparison_logic):
        """Test that empty results return no universal conclusions."""
        universal = comparison_logic.find_universal_conclusions([])
        assert universal == []

    def test_universal_conclusions_single_scenario(self, comparison_logic):
        """Test universal conclusions with single scenario."""
        results = [
            {
                "scenario_name": "Only Scenario",
                "consensus_reached": True,
                "final_answer": "The Answer",
            }
        ]

        universal = comparison_logic.find_universal_conclusions(results)
        assert len(universal) == 1


# ============================================================================
# Test: Universal Conclusions - Partial Consensus
# ============================================================================


class TestUniversalConclusionsPartialConsensus:
    """Test universal conclusion handling with partial consensus."""

    def test_universal_conclusions_partial_consensus(
        self, comparison_logic, partial_consensus_results
    ):
        """Test that partial consensus returns empty universal conclusions."""
        universal = comparison_logic.find_universal_conclusions(partial_consensus_results)

        # When not all scenarios reach consensus, no universal conclusions
        assert universal == []

    def test_partial_consensus_ratio(self, comparison_logic, partial_consensus_results):
        """Test that partial consensus doesn't count as universal."""
        # 2 out of 3 reached consensus, but not all
        matrix = comparison_logic.build_comparison_matrix(partial_consensus_results)

        assert matrix["consensus_rate"] == pytest.approx(2 / 3, rel=0.01)
        # But no universal conclusions
        universal = comparison_logic.find_universal_conclusions(partial_consensus_results)
        assert universal == []


# ============================================================================
# Test: No Universal When Divergent
# ============================================================================


class TestNoUniversalWhenDivergent:
    """Test that divergent scenarios produce no universal conclusions."""

    def test_no_universal_when_divergent(self, comparison_logic, divergent_results):
        """Test that completely divergent scenarios have no universal conclusions."""
        universal = comparison_logic.find_universal_conclusions(divergent_results)
        assert universal == []

    def test_divergent_low_consensus_rate(self, comparison_logic, divergent_results):
        """Test that divergent results have 0 consensus rate."""
        matrix = comparison_logic.build_comparison_matrix(divergent_results)
        assert matrix["consensus_rate"] == 0.0

    def test_divergent_still_tracks_confidence(self, comparison_logic, divergent_results):
        """Test that divergent results still track average confidence."""
        matrix = comparison_logic.build_comparison_matrix(divergent_results)

        # Average of 0.3 and 0.2
        expected_avg = (0.3 + 0.2) / 2
        assert matrix["avg_confidence"] == pytest.approx(expected_avg, rel=0.01)


# ============================================================================
# Test: Conditional Conclusions Extraction
# ============================================================================


class TestConditionalConclusionsExtraction:
    """Test conditional conclusion extraction."""

    def test_conditional_conclusions_extraction(self, comparison_logic, all_consensus_results):
        """Test that conditional conclusions are extracted correctly."""
        conditional = comparison_logic.find_conditional_conclusions(all_consensus_results)

        assert len(conditional) == 3

        # Check structure of first conditional
        first = conditional[0]
        assert "condition" in first
        assert "parameters" in first
        assert "conclusion" in first
        assert "confidence" in first

    def test_conditional_includes_scenario_name(self, comparison_logic, all_consensus_results):
        """Test that condition includes scenario name."""
        conditional = comparison_logic.find_conditional_conclusions(all_consensus_results)

        conditions = [c["condition"] for c in conditional]
        assert "When Scenario A" in conditions
        assert "When Scenario B" in conditions
        assert "When Scenario C" in conditions

    def test_conditional_preserves_parameters(self, comparison_logic):
        """Test that conditional conclusions preserve parameters."""
        results = [
            {
                "scenario_name": "High Budget",
                "consensus_reached": True,
                "final_answer": "Invest aggressively",
                "parameters": {"budget": 100000, "risk_tolerance": "high"},
                "confidence": 0.85,
            },
            {
                "scenario_name": "Low Budget",
                "consensus_reached": True,
                "final_answer": "Save conservatively",
                "parameters": {"budget": 1000, "risk_tolerance": "low"},
                "confidence": 0.9,
            },
        ]

        conditional = comparison_logic.find_conditional_conclusions(results)

        high_budget = next(c for c in conditional if "High Budget" in c["condition"])
        assert high_budget["parameters"]["budget"] == 100000
        assert high_budget["parameters"]["risk_tolerance"] == "high"

    def test_conditional_skips_no_final_answer(self, comparison_logic):
        """Test that results without final_answer are skipped."""
        results = [
            {
                "scenario_name": "Successful",
                "consensus_reached": True,
                "final_answer": "Got an answer",
                "confidence": 0.9,
            },
            {
                "scenario_name": "Failed",
                "consensus_reached": False,
                "final_answer": None,  # No answer
                "confidence": 0.0,
            },
        ]

        conditional = comparison_logic.find_conditional_conclusions(results)

        # Only one conditional (the successful one)
        assert len(conditional) == 1
        assert "Successful" in conditional[0]["condition"]


# ============================================================================
# Test: Conditional Confidence Tracking
# ============================================================================


class TestConditionalConfidenceTracking:
    """Test confidence tracking in conditional conclusions."""

    def test_conditional_confidence_tracking(self, comparison_logic, all_consensus_results):
        """Test that confidence is tracked per conditional conclusion."""
        conditional = comparison_logic.find_conditional_conclusions(all_consensus_results)

        confidences = [c["confidence"] for c in conditional]

        # Should have confidences from fixture: 0.9, 0.85, 0.95
        assert 0.9 in confidences
        assert 0.85 in confidences
        assert 0.95 in confidences

    def test_confidence_defaults_to_zero(self, comparison_logic):
        """Test that missing confidence defaults to 0."""
        results = [
            {
                "scenario_name": "No Confidence",
                "consensus_reached": True,
                "final_answer": "Some answer",
                # confidence not specified
            }
        ]

        conditional = comparison_logic.find_conditional_conclusions(results)
        assert conditional[0]["confidence"] == 0

    def test_average_confidence_calculation(self, comparison_logic, all_consensus_results):
        """Test average confidence calculation in matrix."""
        matrix = comparison_logic.build_comparison_matrix(all_consensus_results)

        # Average of 0.9, 0.85, 0.95
        expected_avg = (0.9 + 0.85 + 0.95) / 3
        assert matrix["avg_confidence"] == pytest.approx(expected_avg, rel=0.01)


# ============================================================================
# Test: Baseline Scenario Handling
# ============================================================================


class TestBaselineScenarioHandling:
    """Test handling of baseline scenarios."""

    def test_baseline_scenario_handling(self, comparison_logic):
        """Test that baseline scenarios are included in comparison."""
        results = [
            {
                "scenario_name": "Baseline",
                "is_baseline": True,
                "consensus_reached": True,
                "final_answer": "Baseline answer",
                "parameters": {},
                "confidence": 0.8,
                "rounds_used": 3,
            },
            {
                "scenario_name": "Variation A",
                "is_baseline": False,
                "consensus_reached": True,
                "final_answer": "Variation A answer",
                "parameters": {"modified": True},
                "confidence": 0.75,
                "rounds_used": 4,
            },
        ]

        # Baseline should be included in all calculations
        universal = comparison_logic.find_universal_conclusions(results)
        assert len(universal) == 1  # All reached consensus

        conditional = comparison_logic.find_conditional_conclusions(results)
        assert len(conditional) == 2  # Both have answers

        matrix = comparison_logic.build_comparison_matrix(results)
        assert "Baseline" in matrix["scenarios"]
        assert "Variation A" in matrix["scenarios"]

    def test_baseline_affects_consensus_rate(self, comparison_logic):
        """Test that baseline consensus affects overall rate."""
        results = [
            {
                "scenario_name": "Baseline",
                "is_baseline": True,
                "consensus_reached": False,  # Baseline didn't reach consensus
                "final_answer": None,
                "confidence": 0.3,
                "rounds_used": 5,
            },
            {
                "scenario_name": "Variation",
                "is_baseline": False,
                "consensus_reached": True,
                "final_answer": "Answer",
                "confidence": 0.9,
                "rounds_used": 2,
            },
        ]

        # Not all reached consensus, so no universal
        universal = comparison_logic.find_universal_conclusions(results)
        assert universal == []

        matrix = comparison_logic.build_comparison_matrix(results)
        assert matrix["consensus_rate"] == 0.5  # 1 out of 2


# ============================================================================
# Test: Comparison Matrix Building
# ============================================================================


class TestComparisonMatrixBuilding:
    """Test comparison matrix construction."""

    def test_matrix_contains_all_scenarios(self, comparison_logic, all_consensus_results):
        """Test that matrix includes all scenario names."""
        matrix = comparison_logic.build_comparison_matrix(all_consensus_results)

        assert "Scenario A" in matrix["scenarios"]
        assert "Scenario B" in matrix["scenarios"]
        assert "Scenario C" in matrix["scenarios"]

    def test_matrix_average_rounds(self, comparison_logic, all_consensus_results):
        """Test average rounds calculation."""
        matrix = comparison_logic.build_comparison_matrix(all_consensus_results)

        # Average of 3, 4, 2
        expected_avg = (3 + 4 + 2) / 3
        assert matrix["avg_rounds"] == pytest.approx(expected_avg, rel=0.01)

    def test_matrix_empty_results_no_division_by_zero(self, comparison_logic):
        """Test that empty results don't cause division by zero."""
        matrix = comparison_logic.build_comparison_matrix([])

        assert matrix["scenarios"] == []
        assert matrix["consensus_rate"] == 0.0
        assert matrix["avg_confidence"] == 0.0
        assert matrix["avg_rounds"] == 0.0


__all__ = [
    "TestUniversalConclusionsAllConsensus",
    "TestUniversalConclusionsPartialConsensus",
    "TestNoUniversalWhenDivergent",
    "TestConditionalConclusionsExtraction",
    "TestConditionalConfidenceTracking",
    "TestBaselineScenarioHandling",
    "TestComparisonMatrixBuilding",
]
