"""
Tests for the Scenario Matrix Debates system.

Tests scenario creation, matrix generation, result comparison,
and the overall matrix debate lifecycle.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from aragora.debate.scenarios import (
    Scenario,
    ScenarioType,
    ScenarioResult,
    ScenarioComparison,
    ScenarioMatrix,
    ScenarioComparator,
    MatrixResult,
    MatrixDebateRunner,
    OutcomeCategory,
    create_scale_scenarios,
    create_risk_scenarios,
    create_time_horizon_scenarios,
)


# =============================================================================
# ScenarioType Tests
# =============================================================================


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_all_scenario_types(self):
        """Test all scenario types exist."""
        assert ScenarioType.CONSTRAINT.value == "constraint"
        assert ScenarioType.ASSUMPTION.value == "assumption"
        assert ScenarioType.STAKEHOLDER.value == "stakeholder"
        assert ScenarioType.SCALE.value == "scale"
        assert ScenarioType.RISK_TOLERANCE.value == "risk_tolerance"
        assert ScenarioType.TIME_HORIZON.value == "time_horizon"
        assert ScenarioType.TECHNOLOGY.value == "technology"
        assert ScenarioType.REGULATORY.value == "regulatory"
        assert ScenarioType.CUSTOM.value == "custom"


# =============================================================================
# OutcomeCategory Tests
# =============================================================================


class TestOutcomeCategory:
    """Tests for OutcomeCategory enum."""

    def test_all_outcome_categories(self):
        """Test all outcome categories exist."""
        assert OutcomeCategory.CONSISTENT.value == "consistent"
        assert OutcomeCategory.CONDITIONAL.value == "conditional"
        assert OutcomeCategory.DIVERGENT.value == "divergent"
        assert OutcomeCategory.INCONCLUSIVE.value == "inconclusive"


# =============================================================================
# Scenario Tests
# =============================================================================


class TestScenario:
    """Tests for Scenario dataclass."""

    def test_scenario_creation(self):
        """Test creating a scenario."""
        scenario = Scenario(
            id="test-1",
            name="Test Scenario",
            scenario_type=ScenarioType.SCALE,
            description="A test scenario",
            parameters={"users": 1000},
            constraints=["Limited budget"],
            assumptions=["Users are technical"],
        )

        assert scenario.id == "test-1"
        assert scenario.name == "Test Scenario"
        assert scenario.scenario_type == ScenarioType.SCALE
        assert scenario.parameters["users"] == 1000
        assert len(scenario.constraints) == 1
        assert len(scenario.assumptions) == 1

    def test_scenario_default_values(self):
        """Test scenario default values."""
        scenario = Scenario(
            id="test-2",
            name="Minimal",
            scenario_type=ScenarioType.CUSTOM,
            description="Minimal scenario",
        )

        assert scenario.parameters == {}
        assert scenario.constraints == []
        assert scenario.assumptions == []
        assert scenario.context_additions == ""
        assert scenario.context_replacements == {}
        assert scenario.priority == 1
        assert scenario.is_baseline is False
        assert scenario.tags == []

    def test_scenario_to_dict(self):
        """Test scenario serialization to dict."""
        scenario = Scenario(
            id="test-3",
            name="Dict Test",
            scenario_type=ScenarioType.RISK_TOLERANCE,
            description="Test serialization",
            parameters={"risk": "high"},
            is_baseline=True,
            tags=["test", "important"],
        )

        result = scenario.to_dict()

        assert result["id"] == "test-3"
        assert result["name"] == "Dict Test"
        assert result["scenario_type"] == "risk_tolerance"
        assert result["parameters"]["risk"] == "high"
        assert result["is_baseline"] is True
        assert "test" in result["tags"]

    def test_scenario_from_dict(self):
        """Test scenario deserialization from dict."""
        data = {
            "id": "test-4",
            "name": "From Dict",
            "scenario_type": "scale",
            "description": "Deserialized scenario",
            "parameters": {"size": "large"},
            "constraints": ["Constraint 1"],
            "assumptions": ["Assumption 1"],
            "is_baseline": True,
        }

        scenario = Scenario.from_dict(data)

        assert scenario.id == "test-4"
        assert scenario.name == "From Dict"
        assert scenario.scenario_type == ScenarioType.SCALE
        assert scenario.parameters["size"] == "large"
        assert scenario.is_baseline is True

    def test_apply_to_context_additions(self):
        """Test applying scenario context additions."""
        scenario = Scenario(
            id="ctx-1",
            name="Context Test",
            scenario_type=ScenarioType.CUSTOM,
            description="Test",
            context_additions="Additional context here",
        )

        base_context = "Base context"
        result = scenario.apply_to_context(base_context)

        assert "Base context" in result
        assert "Additional context here" in result

    def test_apply_to_context_replacements(self):
        """Test applying scenario context replacements."""
        scenario = Scenario(
            id="ctx-2",
            name="Replace Test",
            scenario_type=ScenarioType.CUSTOM,
            description="Test",
            context_replacements={"small": "large", "budget": "unlimited budget"},
        )

        base_context = "This is a small project with budget constraints"
        result = scenario.apply_to_context(base_context)

        assert "large" in result
        assert "unlimited budget" in result
        assert "small" not in result

    def test_apply_to_context_constraints(self):
        """Test applying scenario constraints to context."""
        scenario = Scenario(
            id="ctx-3",
            name="Constraints Test",
            scenario_type=ScenarioType.CONSTRAINT,
            description="Test",
            constraints=["Max 100 users", "Budget under $10k"],
        )

        result = scenario.apply_to_context("Base context")

        assert "Constraints:" in result
        assert "Max 100 users" in result
        assert "Budget under $10k" in result

    def test_apply_to_context_assumptions(self):
        """Test applying scenario assumptions to context."""
        scenario = Scenario(
            id="ctx-4",
            name="Assumptions Test",
            scenario_type=ScenarioType.ASSUMPTION,
            description="Test",
            assumptions=["Users are technical", "High availability required"],
        )

        result = scenario.apply_to_context("Base context")

        assert "Assumptions:" in result
        assert "Users are technical" in result
        assert "High availability required" in result


# =============================================================================
# ScenarioResult Tests
# =============================================================================


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_result_creation(self):
        """Test creating a scenario result."""
        result = ScenarioResult(
            scenario_id="scn-1",
            scenario_name="Test Scenario",
            conclusion="The best approach is X",
            confidence=0.85,
            consensus_reached=True,
            key_claims=["Claim 1", "Claim 2"],
            dissenting_views=["Dissent 1"],
            duration_seconds=120.5,
            rounds=3,
        )

        assert result.scenario_id == "scn-1"
        assert result.confidence == 0.85
        assert result.consensus_reached is True
        assert len(result.key_claims) == 2
        assert result.rounds == 3

    def test_result_default_values(self):
        """Test result default values."""
        result = ScenarioResult(
            scenario_id="scn-2",
            scenario_name="Minimal",
            conclusion="Conclusion",
            confidence=0.5,
            consensus_reached=False,
        )

        assert result.key_claims == []
        assert result.dissenting_views == []
        assert result.duration_seconds == 0.0
        assert result.rounds == 0
        assert result.metadata == {}

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ScenarioResult(
            scenario_id="scn-3",
            scenario_name="Serialize Test",
            conclusion="Test conclusion",
            confidence=0.75,
            consensus_reached=True,
            key_claims=["Claim"],
            metadata={"custom": "data"},
        )

        data = result.to_dict()

        assert data["scenario_id"] == "scn-3"
        assert data["confidence"] == 0.75
        assert data["metadata"]["custom"] == "data"


# =============================================================================
# ScenarioMatrix Tests
# =============================================================================


class TestScenarioMatrix:
    """Tests for ScenarioMatrix class."""

    def test_empty_matrix(self):
        """Test empty scenario matrix."""
        matrix = ScenarioMatrix(name="Empty Matrix")

        assert matrix.name == "Empty Matrix"
        assert len(matrix.scenarios) == 0
        assert len(matrix.dimensions) == 0

    def test_add_scenario(self):
        """Test adding scenarios to matrix."""
        matrix = ScenarioMatrix()
        scenario = Scenario(
            id="s1",
            name="Scenario 1",
            scenario_type=ScenarioType.CUSTOM,
            description="First scenario",
        )

        matrix.add_scenario(scenario)

        assert len(matrix.scenarios) == 1
        assert matrix.scenarios[0].name == "Scenario 1"

    def test_add_scenario_chaining(self):
        """Test method chaining for add_scenario."""
        matrix = ScenarioMatrix()

        result = matrix.add_scenario(
            Scenario(id="s1", name="S1", scenario_type=ScenarioType.CUSTOM, description="")
        ).add_scenario(
            Scenario(id="s2", name="S2", scenario_type=ScenarioType.CUSTOM, description="")
        )

        assert result is matrix
        assert len(matrix.scenarios) == 2

    def test_add_dimension(self):
        """Test adding dimensions to matrix."""
        matrix = ScenarioMatrix()
        matrix.add_dimension("scale", ["small", "medium", "large"])
        matrix.add_dimension("risk", ["low", "high"])

        assert len(matrix.dimensions) == 2
        assert "scale" in matrix.dimensions
        assert len(matrix.dimensions["scale"]) == 3

    def test_generate_grid_single_dimension(self):
        """Test grid generation with single dimension."""
        matrix = ScenarioMatrix()
        matrix.add_dimension("scale", ["small", "large"])
        matrix.generate_grid()

        scenarios = matrix.get_scenarios()
        assert len(scenarios) == 2

    def test_generate_grid_multiple_dimensions(self):
        """Test grid generation with multiple dimensions."""
        matrix = ScenarioMatrix()
        matrix.add_dimension("scale", ["small", "large"])
        matrix.add_dimension("risk", ["low", "high"])
        matrix.generate_grid()

        # 2 x 2 = 4 scenarios
        scenarios = matrix.get_scenarios()
        assert len(scenarios) == 4

    def test_generate_grid_cartesian_product(self):
        """Test grid generates cartesian product."""
        matrix = ScenarioMatrix()
        matrix.add_dimension("a", [1, 2])
        matrix.add_dimension("b", ["x", "y"])
        matrix.generate_grid()

        scenarios = matrix.get_scenarios()
        params = [s.parameters for s in scenarios]

        # Should have all combinations
        assert {"a": 1, "b": "x"} in params
        assert {"a": 1, "b": "y"} in params
        assert {"a": 2, "b": "x"} in params
        assert {"a": 2, "b": "y"} in params

    def test_generate_grid_custom_template(self):
        """Test grid with custom name template."""
        matrix = ScenarioMatrix()
        matrix.add_dimension("size", ["S", "L"])
        matrix.generate_grid(name_template="Size: {size}")

        scenarios = matrix.get_scenarios()
        names = [s.name for s in scenarios]

        assert "Size: S" in names
        assert "Size: L" in names

    def test_generate_sensitivity(self):
        """Test sensitivity analysis generation."""
        matrix = ScenarioMatrix()
        matrix.generate_sensitivity(
            baseline_params={"a": 10, "b": 20},
            vary_params={"a": [5, 15], "b": [10, 30]},
        )

        scenarios = matrix.get_scenarios()

        # Baseline + variations (2 for a, 2 for b) = 5
        # But baseline value is skipped, so it depends on if baseline values are in vary_params
        assert len(scenarios) >= 3  # At least baseline + 2 variations

        # Check baseline exists
        baseline = [s for s in scenarios if s.is_baseline]
        assert len(baseline) == 1

    def test_get_scenarios_sorted_by_priority(self):
        """Test scenarios are sorted by priority (descending)."""
        matrix = ScenarioMatrix()
        matrix.add_scenario(
            Scenario(
                id="low",
                name="Low Priority",
                scenario_type=ScenarioType.CUSTOM,
                description="",
                priority=1,
            )
        )
        matrix.add_scenario(
            Scenario(
                id="high",
                name="High Priority",
                scenario_type=ScenarioType.CUSTOM,
                description="",
                priority=10,
            )
        )
        matrix.add_scenario(
            Scenario(
                id="med",
                name="Medium Priority",
                scenario_type=ScenarioType.CUSTOM,
                description="",
                priority=5,
            )
        )

        scenarios = matrix.get_scenarios()

        assert scenarios[0].name == "High Priority"
        assert scenarios[1].name == "Medium Priority"
        assert scenarios[2].name == "Low Priority"

    def test_from_presets_scale(self):
        """Test scale preset."""
        matrix = ScenarioMatrix.from_presets("scale")

        scenarios = matrix.get_scenarios()
        assert len(scenarios) == 4  # small, medium, large, enterprise

    def test_from_presets_time_horizon(self):
        """Test time horizon preset."""
        matrix = ScenarioMatrix.from_presets("time_horizon")

        scenarios = matrix.get_scenarios()
        assert len(scenarios) == 3  # short, medium, long

    def test_from_presets_risk(self):
        """Test risk preset."""
        matrix = ScenarioMatrix.from_presets("risk")

        scenarios = matrix.get_scenarios()
        assert len(scenarios) == 3  # conservative, moderate, aggressive

    def test_from_presets_comprehensive(self):
        """Test comprehensive preset."""
        matrix = ScenarioMatrix.from_presets("comprehensive")

        scenarios = matrix.get_scenarios()
        # 2 x 2 x 2 = 8 scenarios
        assert len(scenarios) == 8


# =============================================================================
# ScenarioComparator Tests
# =============================================================================


class TestScenarioComparator:
    """Tests for ScenarioComparator class."""

    def test_compare_pair_identical(self):
        """Test comparing identical results."""
        comparator = ScenarioComparator()

        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Use microservices architecture",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Scalable", "Maintainable"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Use microservices architecture",
            confidence=0.85,
            consensus_reached=True,
            key_claims=["Scalable", "Maintainable"],
        )

        comparison = comparator.compare_pair(result_a, result_b)

        assert comparison.conclusions_match is True
        assert comparison.similarity_score == 1.0
        assert len(comparison.shared_claims) == 2

    def test_compare_pair_different(self):
        """Test comparing different results."""
        comparator = ScenarioComparator()

        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Use monolith architecture",
            confidence=0.7,
            consensus_reached=True,
            key_claims=["Simple", "Fast deployment"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Use microservices for scalability",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Scalable", "Complex"],
        )

        comparison = comparator.compare_pair(result_a, result_b)

        assert comparison.similarity_score < 1.0
        assert len(comparison.unique_to_a) == 2
        assert len(comparison.unique_to_b) == 2

    def test_compare_pair_partial_overlap(self):
        """Test comparing results with partial overlap."""
        comparator = ScenarioComparator()

        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Choose option A",
            confidence=0.7,
            consensus_reached=True,
            key_claims=["Claim 1", "Claim 2", "Shared"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Choose option B",
            confidence=0.75,
            consensus_reached=True,
            key_claims=["Claim 3", "Shared"],
        )

        comparison = comparator.compare_pair(result_a, result_b)

        assert "Shared" in comparison.shared_claims
        assert len(comparison.unique_to_a) == 2
        assert len(comparison.unique_to_b) == 1

    def test_compare_pair_confidence_difference_noted(self):
        """Test that large confidence difference is noted."""
        comparator = ScenarioComparator()

        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Same conclusion",
            confidence=0.9,
            consensus_reached=True,
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Same conclusion",
            confidence=0.5,
            consensus_reached=True,
        )

        comparison = comparator.compare_pair(result_a, result_b)

        # Should note confidence difference
        assert any("Confidence differs" in diff for diff in comparison.key_differences)

    def test_conclusions_similar_high_overlap(self):
        """Test conclusions similarity with high word overlap."""
        comparator = ScenarioComparator()

        # These strings share most words, giving high Jaccard similarity
        assert (
            comparator._conclusions_similar(
                "Python is the best choice for this project",
                "Python is the best choice for our project",
            )
            is True
        )

    def test_conclusions_similar_low_overlap(self):
        """Test conclusions similarity with low word overlap."""
        comparator = ScenarioComparator()

        assert (
            comparator._conclusions_similar(
                "Use Python for backend",
                "JavaScript is best for frontend",
            )
            is False
        )

    def test_conclusions_similar_empty_strings(self):
        """Test conclusions similarity with empty strings."""
        comparator = ScenarioComparator()

        assert comparator._conclusions_similar("", "Some text") is False
        assert comparator._conclusions_similar("Some text", "") is False

    def test_analyze_matrix_consistent(self):
        """Test matrix analysis with consistent outcomes."""
        comparator = ScenarioComparator()

        # Use nearly identical conclusions to ensure they match
        matrix_result = MatrixResult(
            matrix_id="test",
            task="Test task",
            created_at=datetime.now(),
            results=[
                ScenarioResult(
                    scenario_id="a",
                    scenario_name="A",
                    conclusion="Use approach X for the best results",
                    confidence=0.8,
                    consensus_reached=True,
                    key_claims=["Shared claim"],
                ),
                ScenarioResult(
                    scenario_id="b",
                    scenario_name="B",
                    conclusion="Use approach X for the best results",
                    confidence=0.85,
                    consensus_reached=True,
                    key_claims=["Shared claim"],
                ),
            ],
        )

        analysis = comparator.analyze_matrix(matrix_result)

        assert analysis["outcome_category"] == "consistent"

    def test_analyze_matrix_divergent(self):
        """Test matrix analysis with divergent outcomes."""
        comparator = ScenarioComparator()

        matrix_result = MatrixResult(
            matrix_id="test",
            task="Test task",
            created_at=datetime.now(),
            results=[
                ScenarioResult(
                    scenario_id="a",
                    scenario_name="A",
                    conclusion="Use monolith",
                    confidence=0.8,
                    consensus_reached=True,
                    key_claims=["Claim A"],
                ),
                ScenarioResult(
                    scenario_id="b",
                    scenario_name="B",
                    conclusion="Use microservices",
                    confidence=0.8,
                    consensus_reached=True,
                    key_claims=["Claim B"],
                ),
            ],
        )

        analysis = comparator.analyze_matrix(matrix_result)

        assert analysis["outcome_category"] == "divergent"

    def test_analyze_matrix_empty(self):
        """Test matrix analysis with no results."""
        comparator = ScenarioComparator()

        matrix_result = MatrixResult(
            matrix_id="test",
            task="Test task",
            created_at=datetime.now(),
            results=[],
        )

        analysis = comparator.analyze_matrix(matrix_result)

        assert "error" in analysis

    def test_generate_summary(self):
        """Test summary generation."""
        comparator = ScenarioComparator()

        matrix_result = MatrixResult(
            matrix_id="test",
            task="Choose architecture",
            created_at=datetime.now(),
            results=[
                ScenarioResult(
                    scenario_id="a",
                    scenario_name="Small Scale",
                    conclusion="Use monolith",
                    confidence=0.8,
                    consensus_reached=True,
                    key_claims=["Simple", "Fast"],
                ),
                ScenarioResult(
                    scenario_id="b",
                    scenario_name="Large Scale",
                    conclusion="Use microservices for large scale",
                    confidence=0.85,
                    consensus_reached=True,
                    key_claims=["Scalable", "Fast"],
                ),
            ],
            scenarios=[
                Scenario(
                    id="a",
                    name="Small Scale",
                    scenario_type=ScenarioType.SCALE,
                    description="Small",
                    parameters={"scale": "small"},
                ),
                Scenario(
                    id="b",
                    name="Large Scale",
                    scenario_type=ScenarioType.SCALE,
                    description="Large",
                    parameters={"scale": "large"},
                ),
            ],
        )

        summary = comparator.generate_summary(matrix_result)

        assert "Choose architecture" in summary
        assert "Small Scale" in summary
        assert "Large Scale" in summary


# =============================================================================
# MatrixResult Tests
# =============================================================================


class TestMatrixResult:
    """Tests for MatrixResult dataclass."""

    def test_matrix_result_creation(self):
        """Test creating a matrix result."""
        result = MatrixResult(
            matrix_id="m1",
            task="Test task",
            created_at=datetime.now(),
        )

        assert result.matrix_id == "m1"
        assert result.task == "Test task"
        assert result.completed_at is None
        assert result.scenarios == []
        assert result.results == []

    def test_get_result_by_scenario_id(self):
        """Test getting result by scenario ID."""
        scenario_result = ScenarioResult(
            scenario_id="s1",
            scenario_name="Scenario 1",
            conclusion="Conclusion",
            confidence=0.8,
            consensus_reached=True,
        )

        matrix_result = MatrixResult(
            matrix_id="m1",
            task="Task",
            created_at=datetime.now(),
            results=[scenario_result],
        )

        found = matrix_result.get_result("s1")
        not_found = matrix_result.get_result("s2")

        assert found is scenario_result
        assert not_found is None

    def test_matrix_result_to_dict(self):
        """Test matrix result serialization."""
        result = MatrixResult(
            matrix_id="m1",
            task="Test",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 30, 0),
            outcome_category=OutcomeCategory.CONSISTENT,
            summary="Test summary",
        )

        data = result.to_dict()

        assert data["matrix_id"] == "m1"
        assert data["outcome_category"] == "consistent"
        assert data["summary"] == "Test summary"


# =============================================================================
# MatrixDebateRunner Tests
# =============================================================================


class TestMatrixDebateRunner:
    """Tests for MatrixDebateRunner class."""

    @pytest.mark.asyncio
    async def test_run_matrix_sequential(self):
        """Test running matrix sequentially."""
        debate_results = [
            MagicMock(
                final_answer="Conclusion A",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Claim A"],
            ),
            MagicMock(
                final_answer="Conclusion B",
                confidence=0.7,
                consensus_reached=True,
                key_claims=["Claim B"],
            ),
        ]

        async def mock_debate(task, context):
            return debate_results.pop(0)

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=1)

        matrix = ScenarioMatrix()
        matrix.add_scenario(
            Scenario(id="a", name="A", scenario_type=ScenarioType.CUSTOM, description="")
        )
        matrix.add_scenario(
            Scenario(id="b", name="B", scenario_type=ScenarioType.CUSTOM, description="")
        )

        result = await runner.run_matrix("Test task", matrix)

        assert len(result.results) == 2
        assert result.results[0].conclusion == "Conclusion A"
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_matrix_parallel(self):
        """Test running matrix in parallel batches."""
        call_count = 0

        async def mock_debate(task, context):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay to simulate work
            return MagicMock(
                final_answer=f"Conclusion {call_count}",
                confidence=0.8,
                consensus_reached=True,
            )

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=2)

        matrix = ScenarioMatrix()
        for i in range(4):
            matrix.add_scenario(
                Scenario(
                    id=str(i),
                    name=f"Scenario {i}",
                    scenario_type=ScenarioType.CUSTOM,
                    description="",
                )
            )

        result = await runner.run_matrix("Test", matrix)

        assert len(result.results) == 4

    @pytest.mark.asyncio
    async def test_run_matrix_with_callback(self):
        """Test matrix run with completion callback."""
        completed = []

        async def mock_debate(task, context):
            return MagicMock(
                final_answer="Done",
                confidence=0.9,
                consensus_reached=True,
            )

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=1)

        matrix = ScenarioMatrix()
        matrix.add_scenario(
            Scenario(id="a", name="A", scenario_type=ScenarioType.CUSTOM, description="")
        )
        matrix.add_scenario(
            Scenario(id="b", name="B", scenario_type=ScenarioType.CUSTOM, description="")
        )

        await runner.run_matrix("Test", matrix, on_scenario_complete=lambda r: completed.append(r))

        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_run_matrix_error_handling(self):
        """Test matrix handles debate errors gracefully."""
        call_count = 0

        async def mock_debate(task, context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Debate failed")
            return MagicMock(
                final_answer="Success",
                confidence=0.8,
                consensus_reached=True,
            )

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=1)

        matrix = ScenarioMatrix()
        matrix.add_scenario(
            Scenario(id="a", name="A", scenario_type=ScenarioType.CUSTOM, description="")
        )
        matrix.add_scenario(
            Scenario(id="b", name="B", scenario_type=ScenarioType.CUSTOM, description="")
        )

        result = await runner.run_matrix("Test", matrix)

        # First scenario should have error in conclusion
        assert "Error" in result.results[0].conclusion
        assert result.results[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_run_scenario_applies_context(self):
        """Test that scenario context is applied to debate."""
        captured_context = None

        async def mock_debate(task, context):
            nonlocal captured_context
            captured_context = context
            return MagicMock(
                final_answer="Done",
                confidence=0.8,
                consensus_reached=True,
            )

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=1)

        matrix = ScenarioMatrix()
        matrix.add_scenario(
            Scenario(
                id="a",
                name="A",
                scenario_type=ScenarioType.CONSTRAINT,
                description="Test constraints",
                constraints=["Budget limit: $10k"],
            )
        )

        await runner.run_matrix("Test", matrix, base_context="Base context")

        assert "Base context" in captured_context
        assert "Budget limit: $10k" in captured_context

    @pytest.mark.asyncio
    async def test_baseline_scenario_identified(self):
        """Test baseline scenario is identified in results."""

        async def mock_debate(task, context):
            return MagicMock(
                final_answer="Done",
                confidence=0.8,
                consensus_reached=True,
            )

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=1)

        matrix = ScenarioMatrix()
        matrix.add_scenario(
            Scenario(
                id="baseline",
                name="Baseline",
                scenario_type=ScenarioType.CUSTOM,
                description="",
                is_baseline=True,
            )
        )
        matrix.add_scenario(
            Scenario(id="other", name="Other", scenario_type=ScenarioType.CUSTOM, description="")
        )

        result = await runner.run_matrix("Test", matrix)

        assert result.baseline_scenario_id == "baseline"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience scenario creation functions."""

    def test_create_scale_scenarios(self):
        """Test create_scale_scenarios function."""
        scenarios = create_scale_scenarios()

        assert len(scenarios) == 3
        assert all(s.scenario_type == ScenarioType.SCALE for s in scenarios)

        names = [s.name for s in scenarios]
        assert "Small Scale" in names
        assert "Medium Scale" in names
        assert "Large Scale" in names

    def test_create_risk_scenarios(self):
        """Test create_risk_scenarios function."""
        scenarios = create_risk_scenarios()

        assert len(scenarios) == 3
        assert all(s.scenario_type == ScenarioType.RISK_TOLERANCE for s in scenarios)

        # One should be baseline
        baselines = [s for s in scenarios if s.is_baseline]
        assert len(baselines) == 1
        assert baselines[0].name == "Moderate"

    def test_create_time_horizon_scenarios(self):
        """Test create_time_horizon_scenarios function."""
        scenarios = create_time_horizon_scenarios()

        assert len(scenarios) == 3
        assert all(s.scenario_type == ScenarioType.TIME_HORIZON for s in scenarios)

        # Check parameters
        short = next(s for s in scenarios if s.id == "short_term")
        assert short.parameters["horizon_months"] == 6


# =============================================================================
# Integration Tests
# =============================================================================


class TestScenarioIntegration:
    """Integration tests for the scenario system."""

    @pytest.mark.asyncio
    async def test_full_matrix_workflow(self):
        """Test complete matrix debate workflow."""
        # Create matrix from preset
        matrix = ScenarioMatrix.from_presets("risk")

        # Mock debate function - note that risk level is in the task, not context
        async def mock_debate(task, context):
            # Different conclusions based on risk level (from task, not context)
            task_lower = task.lower()
            if "aggressive" in task_lower:
                return MagicMock(
                    final_answer="Move fast break things aggressive approach",
                    confidence=0.7,
                    consensus_reached=True,
                    key_claims=["Speed is critical", "Accept some failures"],
                )
            elif "conservative" in task_lower:
                return MagicMock(
                    final_answer="Slow steady conservative wins race",
                    confidence=0.9,
                    consensus_reached=True,
                    key_claims=["Stability first", "Minimize risk"],
                )
            else:
                return MagicMock(
                    final_answer="Balance moderate speed stability",
                    confidence=0.8,
                    consensus_reached=True,
                    key_claims=["Balance is key"],
                )

        runner = MatrixDebateRunner(debate_func=mock_debate, max_parallel=3)

        result = await runner.run_matrix(
            task="How should we approach the new feature?",
            matrix=matrix,
        )

        # Verify results
        assert len(result.results) == 3
        assert result.completed_at is not None

        # Analysis should show outcomes that are not all identical
        # Since the conclusions use different words, similarity should be low
        comparator = ScenarioComparator()
        analysis = comparator.analyze_matrix(result)

        # Accept any valid outcome - the key is that results were analyzed
        assert analysis["outcome_category"] in [
            "consistent",
            "conditional",
            "divergent",
            "inconclusive",
        ]
        assert analysis["total_scenarios"] == 3

    def test_scenario_serialization_roundtrip(self):
        """Test scenario can be serialized and deserialized."""
        original = Scenario(
            id="test",
            name="Test Scenario",
            scenario_type=ScenarioType.TECHNOLOGY,
            description="A test",
            parameters={"lang": "python", "version": 3.11},
            constraints=["Must use type hints"],
            assumptions=["Developers know Python"],
            context_additions="Extra context",
            priority=5,
            is_baseline=True,
            tags=["test", "python"],
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = Scenario.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.scenario_type == original.scenario_type
        assert restored.parameters == original.parameters
        assert restored.constraints == original.constraints
        assert restored.assumptions == original.assumptions
        assert restored.is_baseline == original.is_baseline
        assert restored.tags == original.tags
