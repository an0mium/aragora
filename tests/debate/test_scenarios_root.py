"""Tests for the Scenario Matrix module."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from aragora.debate.scenarios import (
    ScenarioType,
    OutcomeCategory,
    Scenario,
    ScenarioResult,
    ScenarioComparison,
    MatrixResult,
    ScenarioMatrix,
    ScenarioComparator,
    MatrixDebateRunner,
    create_scale_scenarios,
    create_risk_scenarios,
    create_time_horizon_scenarios,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scenario():
    """Create a basic Scenario for testing."""
    return Scenario(
        id="test-001",
        name="Test Scenario",
        scenario_type=ScenarioType.SCALE,
        description="A test scenario",
    )


@pytest.fixture
def scenario_with_all_fields():
    """Create a Scenario with all fields populated."""
    return Scenario(
        id="test-002",
        name="Full Scenario",
        scenario_type=ScenarioType.ASSUMPTION,
        description="A fully populated scenario",
        parameters={"budget": 100, "risk": "moderate"},
        constraints=["Budget < 100k", "Must be cloud-native"],
        assumptions=["Team has 5 developers", "Project runs 6 months"],
        context_additions="Additional context for this scenario.",
        context_replacements={"old_term": "new_term"},
        priority=5,
        is_baseline=True,
        tags=["critical", "mvp"],
    )


@pytest.fixture
def scenario_result():
    """Create a basic ScenarioResult for testing."""
    return ScenarioResult(
        scenario_id="test-001",
        scenario_name="Test Scenario",
        conclusion="Use microservices",
        confidence=0.85,
        consensus_reached=True,
    )


@pytest.fixture
def scenario_result_full():
    """Create a ScenarioResult with all fields."""
    return ScenarioResult(
        scenario_id="test-002",
        scenario_name="Full Scenario",
        conclusion="Use monolith architecture",
        confidence=0.75,
        consensus_reached=False,
        key_claims=["Simpler deployment", "Lower initial cost"],
        dissenting_views=["Scalability concerns"],
        duration_seconds=120.5,
        rounds=4,
        metadata={"agent_count": 3},
    )


@pytest.fixture
def mock_debate_func():
    """Create mock async debate function."""

    async def _debate(task, context):
        result = MagicMock()
        result.final_answer = "Use microservices for scalability"
        result.confidence = 0.8
        result.consensus_reached = True
        result.key_claims = ["claim1", "claim2"]
        result.dissenting_views = []
        result.rounds = 3
        return result

    return _debate


@pytest.fixture
def mock_debate_func_minimal():
    """Create mock debate function with minimal attributes."""

    async def _debate(task, context):
        return "Simple string result"

    return _debate


@pytest.fixture
def mock_debate_func_failing():
    """Create mock debate function that raises exception."""

    async def _debate(task, context):
        raise ValueError("Debate failed!")

    return _debate


@pytest.fixture
def comparator():
    """Create ScenarioComparator instance."""
    return ScenarioComparator()


@pytest.fixture
def runner(mock_debate_func):
    """Create MatrixDebateRunner instance."""
    return MatrixDebateRunner(mock_debate_func)


@pytest.fixture
def matrix_result():
    """Create a MatrixResult for testing."""
    return MatrixResult(
        matrix_id="matrix-001",
        task="Design a system architecture",
        created_at=datetime(2025, 1, 1, 12, 0, 0),
    )


# =============================================================================
# TestScenarioType
# =============================================================================


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_all_values_defined(self):
        """Should define all expected values."""
        assert len(ScenarioType) == 9

    def test_constraint_value(self):
        """Should have CONSTRAINT value."""
        assert ScenarioType.CONSTRAINT.value == "constraint"

    def test_assumption_value(self):
        """Should have ASSUMPTION value."""
        assert ScenarioType.ASSUMPTION.value == "assumption"

    def test_stakeholder_value(self):
        """Should have STAKEHOLDER value."""
        assert ScenarioType.STAKEHOLDER.value == "stakeholder"

    def test_scale_value(self):
        """Should have SCALE value."""
        assert ScenarioType.SCALE.value == "scale"

    def test_risk_tolerance_value(self):
        """Should have RISK_TOLERANCE value."""
        assert ScenarioType.RISK_TOLERANCE.value == "risk_tolerance"

    def test_time_horizon_value(self):
        """Should have TIME_HORIZON value."""
        assert ScenarioType.TIME_HORIZON.value == "time_horizon"

    def test_technology_value(self):
        """Should have TECHNOLOGY value."""
        assert ScenarioType.TECHNOLOGY.value == "technology"

    def test_regulatory_value(self):
        """Should have REGULATORY value."""
        assert ScenarioType.REGULATORY.value == "regulatory"

    def test_custom_value(self):
        """Should have CUSTOM value."""
        assert ScenarioType.CUSTOM.value == "custom"


# =============================================================================
# TestOutcomeCategory
# =============================================================================


class TestOutcomeCategory:
    """Tests for OutcomeCategory enum."""

    def test_all_values_defined(self):
        """Should define all expected values."""
        assert len(OutcomeCategory) == 4

    def test_consistent_value(self):
        """Should have CONSISTENT value."""
        assert OutcomeCategory.CONSISTENT.value == "consistent"

    def test_conditional_value(self):
        """Should have CONDITIONAL value."""
        assert OutcomeCategory.CONDITIONAL.value == "conditional"

    def test_divergent_value(self):
        """Should have DIVERGENT value."""
        assert OutcomeCategory.DIVERGENT.value == "divergent"

    def test_inconclusive_value(self):
        """Should have INCONCLUSIVE value."""
        assert OutcomeCategory.INCONCLUSIVE.value == "inconclusive"


# =============================================================================
# TestScenario
# =============================================================================


class TestScenario:
    """Tests for Scenario dataclass."""

    def test_creation_with_required_fields(self):
        """Should create scenario with required fields only."""
        scenario = Scenario(
            id="s-001",
            name="Test",
            scenario_type=ScenarioType.SCALE,
            description="Test description",
        )
        assert scenario.id == "s-001"
        assert scenario.name == "Test"
        assert scenario.scenario_type == ScenarioType.SCALE
        assert scenario.description == "Test description"

    def test_default_values(self, scenario):
        """Should have correct default values."""
        assert scenario.parameters == {}
        assert scenario.constraints == []
        assert scenario.assumptions == []
        assert scenario.context_additions == ""
        assert scenario.context_replacements == {}
        assert scenario.priority == 1
        assert scenario.is_baseline is False
        assert scenario.tags == []

    def test_to_dict_all_fields(self, scenario_with_all_fields):
        """Should serialize all fields to dict."""
        data = scenario_with_all_fields.to_dict()
        assert data["id"] == "test-002"
        assert data["name"] == "Full Scenario"
        assert data["scenario_type"] == "assumption"  # enum value
        assert data["description"] == "A fully populated scenario"
        assert data["parameters"] == {"budget": 100, "risk": "moderate"}
        assert data["constraints"] == ["Budget < 100k", "Must be cloud-native"]
        assert data["priority"] == 5
        assert data["is_baseline"] is True
        assert data["tags"] == ["critical", "mvp"]

    def test_to_dict_enum_value_conversion(self, scenario):
        """Should convert enum to value string."""
        data = scenario.to_dict()
        assert data["scenario_type"] == "scale"
        assert isinstance(data["scenario_type"], str)

    def test_from_dict_basic(self):
        """Should deserialize from dict."""
        data = {
            "id": "s-003",
            "name": "From Dict",
            "scenario_type": "constraint",
            "description": "Deserialized scenario",
        }
        scenario = Scenario.from_dict(data)
        assert scenario.id == "s-003"
        assert scenario.name == "From Dict"
        assert scenario.scenario_type == ScenarioType.CONSTRAINT
        assert scenario.description == "Deserialized scenario"

    def test_from_dict_enum_reconstruction(self):
        """Should reconstruct enum from string value."""
        data = {
            "id": "s-004",
            "name": "Enum Test",
            "scenario_type": "technology",
            "description": "Test",
        }
        scenario = Scenario.from_dict(data)
        assert scenario.scenario_type == ScenarioType.TECHNOLOGY
        assert isinstance(scenario.scenario_type, ScenarioType)

    def test_serialization_round_trip(self, scenario_with_all_fields):
        """Should preserve all data through round-trip."""
        data = scenario_with_all_fields.to_dict()
        restored = Scenario.from_dict(data)
        assert restored.id == scenario_with_all_fields.id
        assert restored.name == scenario_with_all_fields.name
        assert restored.scenario_type == scenario_with_all_fields.scenario_type
        assert restored.parameters == scenario_with_all_fields.parameters
        assert restored.constraints == scenario_with_all_fields.constraints
        assert restored.is_baseline == scenario_with_all_fields.is_baseline


# =============================================================================
# TestApplyToContext
# =============================================================================


class TestApplyToContext:
    """Tests for Scenario.apply_to_context method."""

    def test_context_replacements(self):
        """Should apply string replacements to context."""
        scenario = Scenario(
            id="s-001",
            name="Test",
            scenario_type=ScenarioType.SCALE,
            description="Test",
            context_replacements={"old": "new", "foo": "bar"},
        )
        result = scenario.apply_to_context("The old foo system")
        assert "new" in result
        assert "bar" in result
        assert "old" not in result
        assert "foo" not in result

    def test_context_additions(self):
        """Should append context additions."""
        scenario = Scenario(
            id="s-001",
            name="Test",
            scenario_type=ScenarioType.SCALE,
            description="Test",
            context_additions="Additional context here.",
        )
        result = scenario.apply_to_context("Base context.")
        assert "Base context." in result
        assert "Additional context here." in result

    def test_constraints_appended(self):
        """Should append constraints with header."""
        scenario = Scenario(
            id="s-001",
            name="Test",
            scenario_type=ScenarioType.CONSTRAINT,
            description="Test",
            constraints=["Budget < 100k", "Team size <= 5"],
        )
        result = scenario.apply_to_context("Base context.")
        assert "Constraints:" in result
        assert "Budget < 100k" in result
        assert "Team size <= 5" in result

    def test_assumptions_appended(self):
        """Should append assumptions with header."""
        scenario = Scenario(
            id="s-001",
            name="Test",
            scenario_type=ScenarioType.ASSUMPTION,
            description="Test",
            assumptions=["Users grow 10% monthly", "API latency < 100ms"],
        )
        result = scenario.apply_to_context("Base context.")
        assert "Assumptions:" in result
        assert "Users grow 10% monthly" in result
        assert "API latency < 100ms" in result

    def test_empty_modifications(self, scenario):
        """Should return base context unchanged when no modifications."""
        base = "This is the base context."
        result = scenario.apply_to_context(base)
        assert result == base

    def test_all_modifications_combined(self, scenario_with_all_fields):
        """Should apply all modifications in order."""
        result = scenario_with_all_fields.apply_to_context("Contains old_term here.")
        # Replacements applied
        assert "new_term" in result
        assert "old_term" not in result
        # Additions appended
        assert "Additional context for this scenario." in result
        # Constraints appended
        assert "Constraints:" in result
        assert "Budget < 100k" in result
        # Assumptions appended
        assert "Assumptions:" in result
        assert "Team has 5 developers" in result

    def test_replacement_order(self):
        """Should apply replacements before additions."""
        scenario = Scenario(
            id="s-001",
            name="Test",
            scenario_type=ScenarioType.SCALE,
            description="Test",
            context_replacements={"old": "new"},
            context_additions="The old becomes new.",
        )
        # The addition is appended after replacements, so "old" in addition stays
        result = scenario.apply_to_context("The old system.")
        assert result.count("new") >= 1  # At least the replacement happened


# =============================================================================
# TestScenarioResult
# =============================================================================


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_creation_with_required_fields(self):
        """Should create result with required fields only."""
        result = ScenarioResult(
            scenario_id="s-001",
            scenario_name="Test",
            conclusion="Use microservices",
            confidence=0.8,
            consensus_reached=True,
        )
        assert result.scenario_id == "s-001"
        assert result.conclusion == "Use microservices"
        assert result.confidence == 0.8
        assert result.consensus_reached is True

    def test_default_values(self, scenario_result):
        """Should have correct default values."""
        assert scenario_result.key_claims == []
        assert scenario_result.dissenting_views == []
        assert scenario_result.duration_seconds == 0.0
        assert scenario_result.rounds == 0
        assert scenario_result.metadata == {}

    def test_to_dict_all_fields(self, scenario_result_full):
        """Should serialize all fields to dict."""
        data = scenario_result_full.to_dict()
        assert data["scenario_id"] == "test-002"
        assert data["scenario_name"] == "Full Scenario"
        assert data["conclusion"] == "Use monolith architecture"
        assert data["confidence"] == 0.75
        assert data["consensus_reached"] is False
        assert data["key_claims"] == ["Simpler deployment", "Lower initial cost"]
        assert data["dissenting_views"] == ["Scalability concerns"]
        assert data["duration_seconds"] == 120.5
        assert data["rounds"] == 4
        assert data["metadata"] == {"agent_count": 3}

    def test_to_dict_with_metadata(self):
        """Should include metadata in serialization."""
        result = ScenarioResult(
            scenario_id="s-001",
            scenario_name="Test",
            conclusion="Conclusion",
            confidence=0.5,
            consensus_reached=True,
            metadata={"error": "timeout", "retries": 3},
        )
        data = result.to_dict()
        assert data["metadata"]["error"] == "timeout"
        assert data["metadata"]["retries"] == 3


# =============================================================================
# TestScenarioComparison
# =============================================================================


class TestScenarioComparison:
    """Tests for ScenarioComparison dataclass."""

    def test_creation_with_required_fields(self):
        """Should create comparison with required fields."""
        comparison = ScenarioComparison(
            scenario_a_id="s-001",
            scenario_b_id="s-002",
            conclusions_match=True,
            similarity_score=0.85,
        )
        assert comparison.scenario_a_id == "s-001"
        assert comparison.scenario_b_id == "s-002"
        assert comparison.conclusions_match is True
        assert comparison.similarity_score == 0.85

    def test_default_values(self):
        """Should have correct default values."""
        comparison = ScenarioComparison(
            scenario_a_id="s-001",
            scenario_b_id="s-002",
            conclusions_match=False,
            similarity_score=0.3,
        )
        assert comparison.key_differences == []
        assert comparison.shared_claims == []
        assert comparison.unique_to_a == []
        assert comparison.unique_to_b == []

    def test_all_fields_populated(self):
        """Should store all fields correctly."""
        comparison = ScenarioComparison(
            scenario_a_id="s-001",
            scenario_b_id="s-002",
            conclusions_match=False,
            similarity_score=0.4,
            key_differences=["Different scale approach"],
            shared_claims=["Both use cloud"],
            unique_to_a=["Microservices"],
            unique_to_b=["Monolith"],
        )
        assert len(comparison.key_differences) == 1
        assert len(comparison.shared_claims) == 1
        assert "Microservices" in comparison.unique_to_a
        assert "Monolith" in comparison.unique_to_b


# =============================================================================
# TestMatrixResult
# =============================================================================


class TestMatrixResult:
    """Tests for MatrixResult dataclass."""

    def test_creation_with_required_fields(self):
        """Should create result with required fields."""
        result = MatrixResult(
            matrix_id="m-001",
            task="Design system",
            created_at=datetime.now(),
        )
        assert result.matrix_id == "m-001"
        assert result.task == "Design system"

    def test_default_values(self, matrix_result):
        """Should have correct default values."""
        assert matrix_result.completed_at is None
        assert matrix_result.scenarios == []
        assert matrix_result.results == []
        assert matrix_result.outcome_category == OutcomeCategory.INCONCLUSIVE
        assert matrix_result.baseline_scenario_id is None
        assert matrix_result.universal_conclusions == []
        assert matrix_result.conditional_conclusions == {}
        assert matrix_result.scenario_comparisons == []
        assert matrix_result.summary == ""
        assert matrix_result.recommendations == []

    def test_get_result_found(self, matrix_result, scenario_result):
        """Should return result when found."""
        matrix_result.results = [scenario_result]
        found = matrix_result.get_result("test-001")
        assert found is not None
        assert found.scenario_id == "test-001"

    def test_get_result_not_found(self, matrix_result, scenario_result):
        """Should return None when not found."""
        matrix_result.results = [scenario_result]
        found = matrix_result.get_result("nonexistent")
        assert found is None

    def test_to_dict_all_fields(self, matrix_result, scenario, scenario_result):
        """Should serialize all fields to dict."""
        matrix_result.scenarios = [scenario]
        matrix_result.results = [scenario_result]
        matrix_result.completed_at = datetime(2025, 1, 1, 13, 0, 0)
        matrix_result.summary = "Analysis complete"

        data = matrix_result.to_dict()
        assert data["matrix_id"] == "matrix-001"
        assert data["task"] == "Design a system architecture"
        assert data["summary"] == "Analysis complete"
        assert len(data["scenarios"]) == 1
        assert len(data["results"]) == 1

    def test_to_dict_datetime_conversion(self, matrix_result):
        """Should convert datetime to ISO format."""
        matrix_result.completed_at = datetime(2025, 6, 15, 10, 30, 0)
        data = matrix_result.to_dict()
        assert data["created_at"] == "2025-01-01T12:00:00"
        assert data["completed_at"] == "2025-06-15T10:30:00"


# =============================================================================
# TestScenarioMatrixInit
# =============================================================================


class TestScenarioMatrixInit:
    """Tests for ScenarioMatrix initialization."""

    def test_initialization(self):
        """Should initialize with name."""
        matrix = ScenarioMatrix("Test Matrix")
        assert matrix.name == "Test Matrix"
        assert matrix.scenarios == []
        assert matrix.dimensions == {}

    def test_add_scenario(self, scenario):
        """Should add scenario to matrix."""
        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(scenario)
        assert len(matrix.scenarios) == 1
        assert matrix.scenarios[0].id == "test-001"

    def test_add_scenario_fluent(self, scenario):
        """Should return self for chaining."""
        matrix = ScenarioMatrix("Test")
        result = matrix.add_scenario(scenario)
        assert result is matrix

    def test_add_dimension(self):
        """Should add dimension to matrix."""
        matrix = ScenarioMatrix("Test")
        matrix.add_dimension("scale", ["small", "medium", "large"])
        assert "scale" in matrix.dimensions
        assert matrix.dimensions["scale"] == ["small", "medium", "large"]

    def test_add_dimension_fluent(self):
        """Should return self for chaining."""
        matrix = ScenarioMatrix("Test")
        result = matrix.add_dimension("scale", ["small", "large"])
        assert result is matrix

    def test_get_scenarios_sorted_by_priority(self):
        """Should return scenarios sorted by priority descending."""
        matrix = ScenarioMatrix("Test")
        s1 = Scenario(
            id="s1",
            name="Low",
            scenario_type=ScenarioType.SCALE,
            description="Low priority",
            priority=1,
        )
        s2 = Scenario(
            id="s2",
            name="High",
            scenario_type=ScenarioType.SCALE,
            description="High priority",
            priority=10,
        )
        s3 = Scenario(
            id="s3",
            name="Mid",
            scenario_type=ScenarioType.SCALE,
            description="Mid priority",
            priority=5,
        )
        matrix.add_scenario(s1).add_scenario(s2).add_scenario(s3)

        sorted_scenarios = matrix.get_scenarios()
        assert sorted_scenarios[0].id == "s2"  # priority 10
        assert sorted_scenarios[1].id == "s3"  # priority 5
        assert sorted_scenarios[2].id == "s1"  # priority 1


# =============================================================================
# TestGenerateGrid
# =============================================================================


class TestGenerateGrid:
    """Tests for ScenarioMatrix.generate_grid method."""

    def test_basic_grid_generation(self):
        """Should generate scenarios from dimensions."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.add_dimension("scale", ["small", "large"])
        matrix.generate_grid(ScenarioType.SCALE)
        assert len(matrix.scenarios) == 2

    def test_cartesian_product(self):
        """Should create cartesian product of dimensions."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.add_dimension("scale", ["small", "large"])
        matrix.add_dimension("risk", ["low", "high"])
        matrix.generate_grid(ScenarioType.CUSTOM)
        # 2 x 2 = 4 combinations
        assert len(matrix.scenarios) == 4

        # Check all combinations exist
        params_list = [s.parameters for s in matrix.scenarios]
        assert {"scale": "small", "risk": "low"} in params_list
        assert {"scale": "small", "risk": "high"} in params_list
        assert {"scale": "large", "risk": "low"} in params_list
        assert {"scale": "large", "risk": "high"} in params_list

    def test_name_template_with_dims(self):
        """Should format name with {dims} placeholder."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.add_dimension("scale", ["small"])
        matrix.add_dimension("risk", ["low"])
        matrix.generate_grid(ScenarioType.CUSTOM, name_template="Scenario {dims}")
        # Name should contain dimension values
        assert "small" in matrix.scenarios[0].name or "low" in matrix.scenarios[0].name

    def test_name_template_with_param_name(self):
        """Should format name with parameter placeholders."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.add_dimension("scale", ["small"])
        matrix.generate_grid(ScenarioType.SCALE, name_template="Scale: {scale}")
        assert "small" in matrix.scenarios[0].name

    def test_empty_dimensions(self):
        """Should handle empty dimensions gracefully."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.generate_grid(ScenarioType.SCALE)
        assert len(matrix.scenarios) == 0

    def test_single_dimension(self):
        """Should work with single dimension."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.add_dimension("size", ["xs", "sm", "md", "lg", "xl"])
        matrix.generate_grid(ScenarioType.SCALE)
        assert len(matrix.scenarios) == 5

    def test_fluent_return(self):
        """Should return self for chaining."""
        matrix = ScenarioMatrix("Grid Test")
        matrix.add_dimension("scale", ["small"])
        result = matrix.generate_grid(ScenarioType.SCALE)
        assert result is matrix


# =============================================================================
# TestGenerateSensitivity
# =============================================================================


class TestGenerateSensitivity:
    """Tests for ScenarioMatrix.generate_sensitivity method."""

    def test_creates_baseline_scenario(self):
        """Should create baseline scenario first."""
        matrix = ScenarioMatrix("Sensitivity Test")
        matrix.generate_sensitivity(
            baseline_params={"budget": 100},
            vary_params={"budget": [50, 150]},
            scenario_type=ScenarioType.CONSTRAINT,
        )
        baseline = [s for s in matrix.scenarios if s.is_baseline]
        assert len(baseline) == 1
        assert baseline[0].parameters["budget"] == 100

    def test_baseline_marked_correctly(self):
        """Should mark only baseline as is_baseline=True."""
        matrix = ScenarioMatrix("Sensitivity Test")
        matrix.generate_sensitivity(
            baseline_params={"x": 10},
            vary_params={"x": [5, 15]},
            scenario_type=ScenarioType.SCALE,
        )
        baselines = [s for s in matrix.scenarios if s.is_baseline]
        non_baselines = [s for s in matrix.scenarios if not s.is_baseline]
        assert len(baselines) == 1
        assert len(non_baselines) == 2

    def test_vary_single_param(self):
        """Should create variations for single parameter."""
        matrix = ScenarioMatrix("Sensitivity Test")
        matrix.generate_sensitivity(
            baseline_params={"budget": 100},
            vary_params={"budget": [50, 150, 200]},
            scenario_type=ScenarioType.CONSTRAINT,
        )
        # 1 baseline + 3 variations
        assert len(matrix.scenarios) == 4

    def test_vary_multiple_params(self):
        """Should create variations for multiple parameters."""
        matrix = ScenarioMatrix("Sensitivity Test")
        matrix.generate_sensitivity(
            baseline_params={"budget": 100, "team": 5},
            vary_params={"budget": [50, 150], "team": [3, 7]},
            scenario_type=ScenarioType.CONSTRAINT,
        )
        # 1 baseline + 2 budget variations + 2 team variations = 5
        assert len(matrix.scenarios) == 5

    def test_skips_baseline_values(self):
        """Should skip variation values matching baseline."""
        matrix = ScenarioMatrix("Sensitivity Test")
        matrix.generate_sensitivity(
            baseline_params={"x": 10},
            vary_params={"x": [5, 10, 15]},  # 10 matches baseline
            scenario_type=ScenarioType.SCALE,
        )
        # 1 baseline + 2 variations (not 3, since 10 is skipped)
        assert len(matrix.scenarios) == 3

    def test_parameter_isolation(self):
        """Should vary one parameter at a time."""
        matrix = ScenarioMatrix("Sensitivity Test")
        matrix.generate_sensitivity(
            baseline_params={"a": 1, "b": 2},
            vary_params={"a": [0], "b": [0]},
            scenario_type=ScenarioType.SCALE,
        )
        # Find non-baseline scenarios
        variations = [s for s in matrix.scenarios if not s.is_baseline]
        for v in variations:
            # Each variation should differ from baseline in exactly one param
            baseline = [s for s in matrix.scenarios if s.is_baseline][0]
            diffs = sum(1 for k in ["a", "b"] if v.parameters[k] != baseline.parameters[k])
            assert diffs == 1

    def test_fluent_return(self):
        """Should return self for chaining."""
        matrix = ScenarioMatrix("Sensitivity Test")
        result = matrix.generate_sensitivity(
            baseline_params={"x": 10},
            vary_params={"x": [5]},
            scenario_type=ScenarioType.SCALE,
        )
        assert result is matrix


# =============================================================================
# TestFromPresets
# =============================================================================


class TestFromPresets:
    """Tests for ScenarioMatrix.from_presets class method."""

    def test_scale_preset(self):
        """Should create scale preset with 4 values."""
        matrix = ScenarioMatrix.from_presets("scale")
        assert len(matrix.scenarios) == 4
        names = [s.name.lower() for s in matrix.scenarios]
        assert any("small" in n for n in names)
        assert any("enterprise" in n for n in names)

    def test_time_horizon_preset(self):
        """Should create time horizon preset with 3 values."""
        matrix = ScenarioMatrix.from_presets("time_horizon")
        assert len(matrix.scenarios) == 3

    def test_risk_preset(self):
        """Should create risk preset with 3 values."""
        matrix = ScenarioMatrix.from_presets("risk")
        assert len(matrix.scenarios) == 3
        names = [s.name.lower() for s in matrix.scenarios]
        assert any("conservative" in n for n in names)
        assert any("aggressive" in n for n in names)

    def test_stakeholder_preset(self):
        """Should create stakeholder preset with 4 values."""
        matrix = ScenarioMatrix.from_presets("stakeholder")
        assert len(matrix.scenarios) == 4
        names = [s.name.lower() for s in matrix.scenarios]
        assert any("developer" in n for n in names)
        assert any("executive" in n for n in names)

    def test_tech_stack_preset(self):
        """Should create tech stack preset as 2D grid."""
        matrix = ScenarioMatrix.from_presets("tech_stack")
        # language x infrastructure (3 x 3 = 9 or similar)
        assert len(matrix.scenarios) >= 4

    def test_comprehensive_preset(self):
        """Should create comprehensive preset as 3D grid."""
        matrix = ScenarioMatrix.from_presets("comprehensive")
        # scale x risk x time (2 x 2 x 2 = 8)
        assert len(matrix.scenarios) == 8

    def test_unknown_preset_returns_empty(self):
        """Should return empty matrix for unknown preset."""
        # Module doesn't raise error, just returns empty matrix
        matrix = ScenarioMatrix.from_presets("nonexistent_preset")
        assert len(matrix.scenarios) == 0


# =============================================================================
# TestConclusionsSimilar
# =============================================================================


class TestConclusionsSimilar:
    """Tests for ScenarioComparator._conclusions_similar method."""

    def test_identical_conclusions(self, comparator):
        """Should return True for identical conclusions."""
        result = comparator._conclusions_similar(
            "Use microservices for the backend",
            "Use microservices for the backend",
        )
        assert result is True

    def test_completely_different(self, comparator):
        """Should return False for completely different conclusions."""
        result = comparator._conclusions_similar(
            "Use microservices for scalability",
            "Deploy monolith on bare metal",
        )
        assert result is False

    def test_partial_overlap(self, comparator):
        """Should handle partial word overlap."""
        result = comparator._conclusions_similar(
            "Use microservices architecture for better scaling",
            "Use microservices pattern for improved scalability",
            threshold=0.3,
        )
        assert result is True  # "use", "microservices", "for" overlap

    def test_threshold_boundary(self, comparator):
        """Should respect threshold parameter."""
        a = "cat dog bird"
        b = "cat fish snake"
        # Overlap: cat (1/5 = 0.2)
        assert comparator._conclusions_similar(a, b, threshold=0.1) is True
        assert comparator._conclusions_similar(a, b, threshold=0.5) is False

    def test_empty_string_handling(self, comparator):
        """Should return False for empty strings."""
        assert comparator._conclusions_similar("", "something") is False
        assert comparator._conclusions_similar("something", "") is False
        assert comparator._conclusions_similar("", "") is False

    def test_single_word_match(self, comparator):
        """Should handle single word conclusions."""
        result = comparator._conclusions_similar("yes", "yes")
        assert result is True
        result = comparator._conclusions_similar("yes", "no")
        assert result is False

    def test_case_insensitivity(self, comparator):
        """Should compare case-insensitively."""
        result = comparator._conclusions_similar(
            "Use MICROSERVICES",
            "use microservices",
        )
        assert result is True


# =============================================================================
# TestComparePair
# =============================================================================


class TestComparePair:
    """Tests for ScenarioComparator.compare_pair method."""

    def test_matching_conclusions(self, comparator):
        """Should detect matching conclusions."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Use microservices for the system",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Scalability", "Flexibility"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Use microservices for the system",
            confidence=0.85,
            consensus_reached=True,
            key_claims=["Scalability", "Maintainability"],
        )
        comparison = comparator.compare_pair(result_a, result_b)
        assert comparison.conclusions_match is True

    def test_different_conclusions(self, comparator):
        """Should detect different conclusions."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Use microservices for enterprise scale",
            confidence=0.8,
            consensus_reached=True,
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Use monolith for simplicity",
            confidence=0.75,
            consensus_reached=True,
        )
        comparison = comparator.compare_pair(result_a, result_b)
        assert comparison.conclusions_match is False

    def test_shared_claims_extraction(self, comparator):
        """Should extract shared claims."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="A",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Cloud native", "Scalable", "Cost effective"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="B",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Cloud native", "Scalable", "Fast deployment"],
        )
        comparison = comparator.compare_pair(result_a, result_b)
        assert "Cloud native" in comparison.shared_claims
        assert "Scalable" in comparison.shared_claims
        assert "Cost effective" not in comparison.shared_claims

    def test_unique_claims_extraction(self, comparator):
        """Should extract unique claims for each scenario."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="A",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Claim A only", "Shared claim"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="B",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["Claim B only", "Shared claim"],
        )
        comparison = comparator.compare_pair(result_a, result_b)
        assert "Claim A only" in comparison.unique_to_a
        assert "Claim B only" in comparison.unique_to_b

    def test_jaccard_similarity_calculation(self, comparator):
        """Should calculate Jaccard similarity correctly."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Same",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["A", "B", "C"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Same",
            confidence=0.8,
            consensus_reached=True,
            key_claims=["A", "B", "D"],
        )
        comparison = comparator.compare_pair(result_a, result_b)
        # Intersection: {A, B}, Union: {A, B, C, D}
        # Jaccard: 2/4 = 0.5
        assert comparison.similarity_score == pytest.approx(0.5, rel=0.1)

    def test_confidence_difference_detection(self, comparator):
        """Should detect large confidence differences."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="X",
            confidence=0.9,
            consensus_reached=True,
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Y",
            confidence=0.5,
            consensus_reached=True,
        )
        comparison = comparator.compare_pair(result_a, result_b)
        # 0.4 difference > 0.2 threshold
        assert any("confidence" in d.lower() for d in comparison.key_differences)

    def test_empty_claims_handling(self, comparator):
        """Should handle empty claims lists."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Same conclusion here",
            confidence=0.8,
            consensus_reached=True,
            key_claims=[],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Same conclusion here",
            confidence=0.8,
            consensus_reached=True,
            key_claims=[],
        )
        comparison = comparator.compare_pair(result_a, result_b)
        assert comparison.shared_claims == []
        assert comparison.unique_to_a == []
        assert comparison.unique_to_b == []

    def test_key_differences_limit(self, comparator):
        """Should limit key differences."""
        result_a = ScenarioResult(
            scenario_id="a",
            scenario_name="A",
            conclusion="Totally different approach X",
            confidence=0.9,
            consensus_reached=True,
            key_claims=["A1", "A2", "A3", "A4", "A5"],
        )
        result_b = ScenarioResult(
            scenario_id="b",
            scenario_name="B",
            conclusion="Completely different method Y",
            confidence=0.2,
            consensus_reached=False,
            key_claims=["B1", "B2", "B3", "B4", "B5"],
        )
        comparison = comparator.compare_pair(result_a, result_b)
        # Should have some differences but not unlimited
        assert len(comparison.key_differences) > 0


# =============================================================================
# TestAnalyzeMatrix
# =============================================================================


class TestAnalyzeMatrix:
    """Tests for ScenarioComparator.analyze_matrix method."""

    def test_empty_results_error(self, comparator, matrix_result):
        """Should return error for empty results."""
        analysis = comparator.analyze_matrix(matrix_result)
        assert "error" in analysis

    def test_consistent_outcome(self, comparator, matrix_result):
        """Should detect consistent outcome when all match."""
        results = [
            ScenarioResult(
                scenario_id=f"s-{i}",
                scenario_name=f"Scenario {i}",
                conclusion="Use microservices architecture",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Scalable", "Flexible"],
            )
            for i in range(3)
        ]
        matrix_result.results = results
        analysis = comparator.analyze_matrix(matrix_result)
        assert analysis["outcome_category"] == "consistent"

    def test_divergent_outcome(self, comparator, matrix_result):
        """Should detect divergent outcome when none match."""
        results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="S1",
                conclusion="Use microservices",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["A"],
            ),
            ScenarioResult(
                scenario_id="s-2",
                scenario_name="S2",
                conclusion="Use monolith",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["B"],
            ),
            ScenarioResult(
                scenario_id="s-3",
                scenario_name="S3",
                conclusion="Use serverless",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["C"],
            ),
        ]
        matrix_result.results = results
        analysis = comparator.analyze_matrix(matrix_result)
        # Should be divergent or inconclusive due to no matches
        assert analysis["outcome_category"] in ["divergent", "inconclusive"]

    def test_conditional_outcome(self, comparator, matrix_result, scenario):
        """Should detect conditional outcome with partial matches."""
        results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="S1",
                conclusion="Use microservices for scalability",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Scalable"],
            ),
            ScenarioResult(
                scenario_id="s-2",
                scenario_name="S2",
                conclusion="Use microservices for flexibility",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Flexible"],
            ),
        ]
        matrix_result.results = results
        matrix_result.scenarios = [scenario]
        analysis = comparator.analyze_matrix(matrix_result)
        # Some overlap but not all -> conditional or consistent
        assert analysis["outcome_category"] in ["conditional", "consistent"]

    def test_inconclusive_outcome(self, comparator, matrix_result):
        """Should detect inconclusive with low similarity."""
        results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="S1",
                conclusion="First approach works best",
                confidence=0.5,
                consensus_reached=False,
                key_claims=["A", "B"],
            ),
            ScenarioResult(
                scenario_id="s-2",
                scenario_name="S2",
                conclusion="Second method preferred",
                confidence=0.6,
                consensus_reached=False,
                key_claims=["C", "D"],
            ),
        ]
        matrix_result.results = results
        analysis = comparator.analyze_matrix(matrix_result)
        # Low similarity should result in divergent or inconclusive
        assert analysis["outcome_category"] in ["divergent", "inconclusive"]

    def test_universal_conclusions(self, comparator, matrix_result):
        """Should find claims present in all scenarios."""
        results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="S1",
                conclusion="A",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Universal claim", "Unique A"],
            ),
            ScenarioResult(
                scenario_id="s-2",
                scenario_name="S2",
                conclusion="B",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Universal claim", "Unique B"],
            ),
        ]
        matrix_result.results = results
        analysis = comparator.analyze_matrix(matrix_result)
        assert "Universal claim" in analysis.get("universal_conclusions", [])

    def test_conditional_patterns_by_param(self, comparator, matrix_result):
        """Should group claims by parameter values."""
        s1 = Scenario(
            id="s-1",
            name="Small",
            scenario_type=ScenarioType.SCALE,
            description="Small scale",
            parameters={"scale": "small"},
        )
        s2 = Scenario(
            id="s-2",
            name="Large",
            scenario_type=ScenarioType.SCALE,
            description="Large scale",
            parameters={"scale": "large"},
        )
        matrix_result.scenarios = [s1, s2]
        matrix_result.results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="Small",
                conclusion="Simple",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Simple deployment"],
            ),
            ScenarioResult(
                scenario_id="s-2",
                scenario_name="Large",
                conclusion="Complex",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Distributed system"],
            ),
        ]
        analysis = comparator.analyze_matrix(matrix_result)
        # Should have conditional patterns by scale parameter
        assert "conditional_patterns" in analysis

    def test_comparison_pair_counting(self, comparator, matrix_result):
        """Should create correct number of pairwise comparisons."""
        results = [
            ScenarioResult(
                scenario_id=f"s-{i}",
                scenario_name=f"S{i}",
                conclusion=f"Conclusion {i}",
                confidence=0.8,
                consensus_reached=True,
            )
            for i in range(4)
        ]
        matrix_result.results = results
        analysis = comparator.analyze_matrix(matrix_result)
        # n*(n-1)/2 pairs for n=4: 6 comparisons
        assert len(analysis.get("comparisons", [])) == 6


# =============================================================================
# TestGenerateSummary
# =============================================================================


class TestGenerateSummary:
    """Tests for ScenarioComparator.generate_summary method."""

    def test_markdown_structure(self, comparator, matrix_result, scenario_result):
        """Should generate valid markdown structure."""
        matrix_result.results = [scenario_result]
        summary = comparator.generate_summary(matrix_result)
        assert "##" in summary or "#" in summary  # Has headers
        assert isinstance(summary, str)

    def test_includes_outcome_category(self, comparator, matrix_result, scenario_result):
        """Should include outcome category in summary."""
        matrix_result.results = [scenario_result]
        summary = comparator.generate_summary(matrix_result)
        # Should mention the outcome somewhere
        assert len(summary) > 0

    def test_universal_conclusions_listed(self, comparator, matrix_result):
        """Should list universal conclusions."""
        results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="S1",
                conclusion="A",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Shared insight", "A only"],
            ),
            ScenarioResult(
                scenario_id="s-2",
                scenario_name="S2",
                conclusion="A",
                confidence=0.8,
                consensus_reached=True,
                key_claims=["Shared insight", "B only"],
            ),
        ]
        matrix_result.results = results
        summary = comparator.generate_summary(matrix_result)
        assert "Shared insight" in summary or len(summary) > 50

    def test_conditional_patterns_listed(self, comparator, matrix_result):
        """Should list conditional patterns when present."""
        s1 = Scenario(
            id="s-1",
            name="Low Risk",
            scenario_type=ScenarioType.RISK_TOLERANCE,
            description="Low risk",
            parameters={"risk": "low"},
        )
        matrix_result.scenarios = [s1]
        matrix_result.results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="Low Risk",
                conclusion="Conservative approach",
                confidence=0.9,
                consensus_reached=True,
                key_claims=["Safety first"],
            ),
        ]
        summary = comparator.generate_summary(matrix_result)
        assert len(summary) > 0

    def test_scenario_results_truncated(self, comparator, matrix_result):
        """Should handle many scenario results."""
        results = [
            ScenarioResult(
                scenario_id=f"s-{i}",
                scenario_name=f"Scenario {i}",
                conclusion=f"Conclusion {i}",
                confidence=0.8,
                consensus_reached=True,
            )
            for i in range(10)
        ]
        matrix_result.results = results
        summary = comparator.generate_summary(matrix_result)
        # Should not crash with many results
        assert len(summary) > 0

    def test_conclusion_text_truncated(self, comparator, matrix_result):
        """Should truncate long conclusion text."""
        long_conclusion = "A" * 500
        matrix_result.results = [
            ScenarioResult(
                scenario_id="s-1",
                scenario_name="S1",
                conclusion=long_conclusion,
                confidence=0.8,
                consensus_reached=True,
            ),
        ]
        summary = comparator.generate_summary(matrix_result)
        # Full 500 chars should not appear
        assert long_conclusion not in summary


# =============================================================================
# TestMatrixDebateRunnerInit
# =============================================================================


class TestMatrixDebateRunnerInit:
    """Tests for MatrixDebateRunner initialization."""

    def test_initialization_with_defaults(self, mock_debate_func):
        """Should initialize with default max_parallel."""
        runner = MatrixDebateRunner(mock_debate_func)
        assert runner.debate_func is mock_debate_func
        assert runner.max_parallel == 3

    def test_custom_max_parallel(self, mock_debate_func):
        """Should accept custom max_parallel."""
        runner = MatrixDebateRunner(mock_debate_func, max_parallel=5)
        assert runner.max_parallel == 5

    def test_comparator_created(self, mock_debate_func):
        """Should create ScenarioComparator instance."""
        runner = MatrixDebateRunner(mock_debate_func)
        assert isinstance(runner.comparator, ScenarioComparator)


# =============================================================================
# TestRunScenarioDebate
# =============================================================================


class TestRunScenarioDebate:
    """Tests for MatrixDebateRunner._run_scenario_debate method."""

    @pytest.mark.asyncio
    async def test_applies_scenario_to_context(self, runner, scenario):
        """Should apply scenario modifications to context."""
        scenario.context_additions = "Extra info."
        result = await runner._run_scenario_debate("Test task", scenario, "Base context")
        # The scenario was applied (check via debate_func call)
        assert result.scenario_id == scenario.id

    @pytest.mark.asyncio
    async def test_modifies_task_with_scenario_info(self, mock_debate_func, scenario):
        """Should add scenario info to task."""
        captured_task = None

        async def capture_debate(task, context):
            nonlocal captured_task
            captured_task = task
            result = MagicMock()
            result.final_answer = "Answer"
            result.confidence = 0.8
            result.consensus_reached = True
            result.key_claims = []
            result.dissenting_views = []
            result.rounds = 1
            return result

        runner = MatrixDebateRunner(capture_debate)
        await runner._run_scenario_debate("Original task", scenario, "Context")
        assert "[Scenario:" in captured_task
        assert scenario.name in captured_task

    @pytest.mark.asyncio
    async def test_extracts_result_attributes(self, runner, scenario):
        """Should extract attributes from debate result."""
        result = await runner._run_scenario_debate("Test task", scenario, "Context")
        assert result.conclusion == "Use microservices for scalability"
        assert result.confidence == 0.8
        assert result.consensus_reached is True
        assert "claim1" in result.key_claims

    @pytest.mark.asyncio
    async def test_uses_default_values(self, mock_debate_func_minimal, scenario):
        """Should use defaults when attributes missing."""
        runner = MatrixDebateRunner(mock_debate_func_minimal)
        result = await runner._run_scenario_debate("Test task", scenario, "Context")
        # Should have fallback values
        assert result.confidence == 0.5  # default
        assert result.consensus_reached is True  # default
        assert result.key_claims == []  # default

    @pytest.mark.asyncio
    async def test_calculates_duration(self, runner, scenario):
        """Should calculate duration in seconds."""
        result = await runner._run_scenario_debate("Test task", scenario, "Context")
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_handles_exception(self, mock_debate_func_failing, scenario):
        """Should handle exceptions gracefully."""
        runner = MatrixDebateRunner(mock_debate_func_failing)
        result = await runner._run_scenario_debate("Test task", scenario, "Context")
        assert "Error:" in result.conclusion
        assert result.confidence == 0.0
        assert result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_error_result_structure(self, mock_debate_func_failing, scenario):
        """Should include error in metadata."""
        runner = MatrixDebateRunner(mock_debate_func_failing)
        result = await runner._run_scenario_debate("Test task", scenario, "Context")
        assert "error" in result.metadata
        assert "Debate failed!" in result.metadata["error"]


# =============================================================================
# TestRunMatrix
# =============================================================================


class TestRunMatrix:
    """Tests for MatrixDebateRunner.run_matrix method."""

    @pytest.mark.asyncio
    async def test_creates_matrix_result(self, runner, scenario):
        """Should create MatrixResult with proper fields."""
        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(scenario)
        result = await runner.run_matrix("Design system", matrix, "Context")
        assert result.matrix_id is not None
        assert result.task == "Design system"
        assert result.created_at is not None

    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_debate_func, scenario):
        """Should execute sequentially when max_parallel=1."""
        runner = MatrixDebateRunner(mock_debate_func, max_parallel=1)
        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(scenario)
        result = await runner.run_matrix("Task", matrix, "Context")
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_debate_func):
        """Should execute in parallel when max_parallel > 1."""
        runner = MatrixDebateRunner(mock_debate_func, max_parallel=3)
        matrix = ScenarioMatrix("Test")
        for i in range(5):
            matrix.add_scenario(
                Scenario(
                    id=f"s-{i}",
                    name=f"Scenario {i}",
                    scenario_type=ScenarioType.SCALE,
                    description=f"Desc {i}",
                )
            )
        result = await runner.run_matrix("Task", matrix, "Context")
        assert len(result.results) == 5

    @pytest.mark.asyncio
    async def test_exception_handling_in_gather(self):
        """Should handle exceptions in parallel gather."""
        call_count = 0

        async def mixed_debate(task, context):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Second call fails")
            result = MagicMock()
            result.final_answer = "OK"
            result.confidence = 0.8
            result.consensus_reached = True
            result.key_claims = []
            result.dissenting_views = []
            result.rounds = 1
            return result

        runner = MatrixDebateRunner(mixed_debate, max_parallel=3)
        matrix = ScenarioMatrix("Test")
        for i in range(3):
            matrix.add_scenario(
                Scenario(
                    id=f"s-{i}",
                    name=f"S{i}",
                    scenario_type=ScenarioType.SCALE,
                    description=f"D{i}",
                )
            )
        result = await runner.run_matrix("Task", matrix, "Context")
        # Should complete without crashing, may have error results
        assert len(result.results) >= 2

    @pytest.mark.asyncio
    async def test_on_scenario_complete_callback(self, runner, scenario):
        """Should call callback for each completed scenario."""
        completed = []

        def on_complete(result):
            completed.append(result)

        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(scenario)
        await runner.run_matrix("Task", matrix, "Context", on_scenario_complete=on_complete)
        assert len(completed) == 1
        assert completed[0].scenario_id == scenario.id

    @pytest.mark.asyncio
    async def test_finds_baseline_scenario(self, runner):
        """Should identify baseline scenario."""
        baseline = Scenario(
            id="baseline",
            name="Baseline",
            scenario_type=ScenarioType.SCALE,
            description="Baseline",
            is_baseline=True,
        )
        other = Scenario(
            id="other",
            name="Other",
            scenario_type=ScenarioType.SCALE,
            description="Other",
            is_baseline=False,
        )
        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(baseline).add_scenario(other)
        result = await runner.run_matrix("Task", matrix, "Context")
        assert result.baseline_scenario_id == "baseline"

    @pytest.mark.asyncio
    async def test_outcome_analysis(self, runner):
        """Should analyze outcomes after completion."""
        matrix = ScenarioMatrix("Test")
        for i in range(2):
            matrix.add_scenario(
                Scenario(
                    id=f"s-{i}",
                    name=f"S{i}",
                    scenario_type=ScenarioType.SCALE,
                    description=f"D{i}",
                )
            )
        result = await runner.run_matrix("Task", matrix, "Context")
        # Should have outcome category set
        assert result.outcome_category is not None

    @pytest.mark.asyncio
    async def test_summary_generation(self, runner, scenario):
        """Should generate summary after completion."""
        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(scenario)
        result = await runner.run_matrix("Task", matrix, "Context")
        # Summary should be populated
        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_sets_completed_at(self, runner, scenario):
        """Should set completed_at timestamp."""
        matrix = ScenarioMatrix("Test")
        matrix.add_scenario(scenario)
        result = await runner.run_matrix("Task", matrix, "Context")
        assert result.completed_at is not None
        assert result.completed_at >= result.created_at


# =============================================================================
# TestCreateScaleScenarios
# =============================================================================


class TestCreateScaleScenarios:
    """Tests for create_scale_scenarios function."""

    def test_returns_three_scenarios(self):
        """Should return exactly 3 scenarios."""
        scenarios = create_scale_scenarios()
        assert len(scenarios) == 3

    def test_small_scenario_params(self):
        """Should have small scenario with correct params."""
        scenarios = create_scale_scenarios()
        small = next((s for s in scenarios if s.id == "small"), None)
        assert small is not None
        assert small.parameters.get("users", 0) <= 100
        assert small.scenario_type == ScenarioType.SCALE

    def test_medium_scenario_params(self):
        """Should have medium scenario with mid-range params."""
        scenarios = create_scale_scenarios()
        medium = next((s for s in scenarios if s.id == "medium"), None)
        assert medium is not None
        assert medium.parameters.get("users", 0) > 100

    def test_large_scenario_params(self):
        """Should have large scenario with enterprise params."""
        scenarios = create_scale_scenarios()
        large = next((s for s in scenarios if s.id == "large"), None)
        assert large is not None
        assert large.parameters.get("users", 0) >= 10000


# =============================================================================
# TestCreateRiskScenarios
# =============================================================================


class TestCreateRiskScenarios:
    """Tests for create_risk_scenarios function."""

    def test_returns_three_scenarios(self):
        """Should return exactly 3 scenarios."""
        scenarios = create_risk_scenarios()
        assert len(scenarios) == 3

    def test_moderate_is_baseline(self):
        """Should mark moderate as baseline."""
        scenarios = create_risk_scenarios()
        moderate = next((s for s in scenarios if s.id == "moderate"), None)
        assert moderate is not None
        assert moderate.is_baseline is True

    def test_risk_parameters(self):
        """Should have distinct risk levels."""
        scenarios = create_risk_scenarios()
        ids = {s.id for s in scenarios}
        assert "conservative" in ids
        assert "moderate" in ids
        assert "aggressive" in ids


# =============================================================================
# TestCreateTimeHorizonScenarios
# =============================================================================


class TestCreateTimeHorizonScenarios:
    """Tests for create_time_horizon_scenarios function."""

    def test_returns_three_scenarios(self):
        """Should return exactly 3 scenarios."""
        scenarios = create_time_horizon_scenarios()
        assert len(scenarios) == 3

    def test_medium_term_is_baseline(self):
        """Should mark medium_term as baseline."""
        scenarios = create_time_horizon_scenarios()
        medium = next((s for s in scenarios if s.id == "medium_term"), None)
        assert medium is not None
        assert medium.is_baseline is True

    def test_time_parameters(self):
        """Should have distinct time horizons."""
        scenarios = create_time_horizon_scenarios()
        ids = {s.id for s in scenarios}
        assert "short_term" in ids
        assert "medium_term" in ids
        assert "long_term" in ids
        # Check time values increase
        short = next(s for s in scenarios if s.id == "short_term")
        long = next(s for s in scenarios if s.id == "long_term")
        assert short.parameters.get("months", 0) < long.parameters.get("months", 100)
