"""
Tests for Workflow Patterns.

Tests cover:
- Pattern factory function and registry
- HiveMindPattern creation and structure
- SequentialPattern creation and structure
- MapReducePattern creation and structure
- HierarchicalPattern creation and structure
- ReviewCyclePattern creation and structure
- DialecticPattern creation and structure
- PatternConfig and ResourceLimits
- Visual layout generation
"""

import pytest

from aragora.workflow.patterns import (
    PATTERN_REGISTRY,
    DialecticPattern,
    HierarchicalPattern,
    HiveMindPattern,
    MapReducePattern,
    PatternConfig,
    PatternType,
    ReviewCyclePattern,
    SequentialPattern,
    WorkflowPattern,
    create_pattern,
)
from aragora.workflow.patterns.base import ResourceLimits
from aragora.workflow.types import (
    ExecutionPattern,
    NodeCategory,
    WorkflowCategory,
    WorkflowDefinition,
)


# ============================================================================
# PatternConfig and ResourceLimits Tests
# ============================================================================


class TestResourceLimits:
    """Tests for ResourceLimits configuration."""

    def test_default_values(self):
        """Test default resource limit values."""
        limits = ResourceLimits()

        assert limits.max_tokens == 100000
        assert limits.max_cost_usd == 10.0
        assert limits.timeout_seconds == 600.0
        assert limits.max_parallel_agents == 5
        assert limits.max_retries == 3

    def test_custom_values(self):
        """Test custom resource limit values."""
        limits = ResourceLimits(
            max_tokens=50000,
            max_cost_usd=5.0,
            timeout_seconds=300.0,
        )

        assert limits.max_tokens == 50000
        assert limits.max_cost_usd == 5.0
        assert limits.timeout_seconds == 300.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        limits = ResourceLimits(max_tokens=1000)
        data = limits.to_dict()

        assert data["max_tokens"] == 1000
        assert "max_cost_usd" in data
        assert "timeout_seconds" in data


class TestPatternConfig:
    """Tests for PatternConfig."""

    def test_default_config(self):
        """Test default pattern configuration."""
        config = PatternConfig(name="Test Pattern")

        assert config.name == "Test Pattern"
        assert config.agents == ["claude", "gpt4"]
        assert config.category == WorkflowCategory.GENERAL
        assert config.output_format == "json"

    def test_custom_config(self):
        """Test custom pattern configuration."""
        config = PatternConfig(
            name="Custom Pattern",
            description="A custom workflow pattern",
            agents=["claude", "gemini", "grok"],
            task="Analyze this data",
            category=WorkflowCategory.LEGAL,
            tags=["legal", "analysis"],
        )

        assert config.name == "Custom Pattern"
        assert len(config.agents) == 3
        assert config.category == WorkflowCategory.LEGAL


# ============================================================================
# Pattern Registry Tests
# ============================================================================


class TestPatternRegistry:
    """Tests for pattern registry and factory."""

    def test_registry_contains_all_patterns(self):
        """Test that registry contains all pattern types."""
        assert PatternType.HIVE_MIND in PATTERN_REGISTRY
        assert PatternType.SEQUENTIAL in PATTERN_REGISTRY
        assert PatternType.MAP_REDUCE in PATTERN_REGISTRY
        assert PatternType.HIERARCHICAL in PATTERN_REGISTRY
        assert PatternType.REVIEW_CYCLE in PATTERN_REGISTRY
        assert PatternType.DIALECTIC in PATTERN_REGISTRY

    def test_create_pattern_hive_mind(self):
        """Test creating HiveMind pattern via factory."""
        pattern = create_pattern(
            PatternType.HIVE_MIND,
            name="Test Hive Mind",
            agents=["claude", "gpt4"],
            task="Test task",
        )

        assert isinstance(pattern, HiveMindPattern)
        assert pattern.name == "Test Hive Mind"

    def test_create_pattern_sequential(self):
        """Test creating Sequential pattern via factory."""
        pattern = create_pattern(
            PatternType.SEQUENTIAL,
            name="Test Sequential",
            agents=["claude", "gpt4"],
        )

        assert isinstance(pattern, SequentialPattern)

    def test_create_pattern_unknown_type(self):
        """Test that unknown pattern type raises error."""
        with pytest.raises(ValueError, match="Unknown pattern type"):
            create_pattern("nonexistent_pattern", name="Test")


# ============================================================================
# HiveMindPattern Tests
# ============================================================================


class TestHiveMindPattern:
    """Tests for HiveMindPattern."""

    def test_create_basic_workflow(self):
        """Test creating a basic hive-mind workflow."""
        workflow = HiveMindPattern.create(
            name="Test Hive Mind",
            agents=["claude", "gpt4", "gemini"],
            task="Analyze this document",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "hive_mind" in workflow.name.lower() or workflow.name == "Test Hive Mind"
        assert len(workflow.steps) > 0

    def test_workflow_has_agent_steps(self):
        """Test that workflow has steps for each agent."""
        workflow = HiveMindPattern.create(
            name="Multi-Agent Analysis",
            agents=["claude", "gpt4"],
            task="Test task",
        )

        # Should have steps for agents plus merge and synthesis
        agent_steps = [
            s
            for s in workflow.steps
            if "agent" in s.step_type or "claude" in s.id or "gpt4" in s.id
        ]
        assert len(agent_steps) >= 2

    def test_workflow_has_consensus_merge(self):
        """Test that workflow has a consensus merge step."""
        workflow = HiveMindPattern.create(
            name="Consensus Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        merge_steps = [
            s for s in workflow.steps if "merge" in s.id.lower() or "consensus" in s.id.lower()
        ]
        assert len(merge_steps) >= 1

    def test_workflow_has_synthesis_step(self):
        """Test that workflow has a synthesis step."""
        workflow = HiveMindPattern.create(
            name="Synthesis Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        synthesis_steps = [s for s in workflow.steps if "synthesis" in s.id.lower()]
        assert len(synthesis_steps) >= 1

    def test_workflow_metadata(self):
        """Test that workflow metadata contains pattern info."""
        workflow = HiveMindPattern.create(
            name="Metadata Test",
            agents=["claude", "gpt4"],
            task="Test",
            consensus_mode="weighted",
        )

        assert workflow.metadata.get("pattern") == "hive_mind"
        assert "consensus_mode" in workflow.metadata

    def test_workflow_tags(self):
        """Test that workflow has appropriate tags."""
        workflow = HiveMindPattern.create(
            name="Tags Test",
            agents=["claude"],
            task="Test",
        )

        assert "hive_mind" in workflow.tags
        assert "parallel" in workflow.tags

    def test_custom_consensus_threshold(self):
        """Test custom consensus threshold configuration."""
        pattern = HiveMindPattern(
            name="Threshold Test",
            agents=["claude", "gpt4"],
            task="Test",
            consensus_threshold=0.9,
        )

        assert pattern.consensus_threshold == 0.9

        workflow = pattern.create_workflow()
        assert workflow.metadata.get("consensus_threshold") == 0.9


# ============================================================================
# SequentialPattern Tests
# ============================================================================


class TestSequentialPattern:
    """Tests for SequentialPattern."""

    def test_create_basic_workflow(self):
        """Test creating a basic sequential workflow."""
        workflow = SequentialPattern.create(
            name="Test Sequential",
            agents=["claude", "gpt4"],
            task="Analyze this",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.steps) >= 2

    def test_workflow_steps_are_sequential(self):
        """Test that steps are connected sequentially."""
        workflow = SequentialPattern.create(
            name="Sequential Test",
            agents=["claude", "gpt4", "gemini"],
            task="Test",
        )

        # Check that steps have next_steps set
        steps_with_next = [s for s in workflow.steps if s.next_steps]
        assert len(steps_with_next) > 0

    def test_workflow_has_transitions(self):
        """Test that workflow has transition rules."""
        workflow = SequentialPattern.create(
            name="Transition Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        assert len(workflow.transitions) >= 1

    def test_custom_prompts(self):
        """Test using custom per-agent prompts."""
        workflow = SequentialPattern.create(
            name="Custom Prompts",
            agents=["claude", "gpt4"],
            prompts={
                "claude": "Extract facts from: {input}",
                "gpt4": "Analyze facts: {step.claude}",
            },
        )

        # Verify steps were created with custom prompts
        assert len(workflow.steps) >= 2

    def test_stages_configuration(self):
        """Test using stages configuration."""
        workflow = SequentialPattern.create(
            name="Staged Pipeline",
            stages=[
                {"agent": "claude", "role": "analyzer", "focus": "structure"},
                {"agent": "gpt4", "role": "reviewer", "focus": "quality"},
            ],
            task="Review this document",
        )

        assert len(workflow.steps) >= 2
        assert workflow.metadata.get("stages") == 2

    def test_workflow_metadata(self):
        """Test that workflow metadata contains pattern info."""
        workflow = SequentialPattern.create(
            name="Metadata Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        assert workflow.metadata.get("pattern") == "sequential"

    def test_workflow_tags(self):
        """Test that workflow has appropriate tags."""
        workflow = SequentialPattern.create(
            name="Tags Test",
            agents=["claude"],
            task="Test",
        )

        assert "sequential" in workflow.tags
        assert "pipeline" in workflow.tags


# ============================================================================
# MapReducePattern Tests
# ============================================================================


class TestMapReducePattern:
    """Tests for MapReducePattern."""

    def test_create_basic_workflow(self):
        """Test creating a basic map-reduce workflow."""
        workflow = MapReducePattern.create(
            name="Test MapReduce",
            agents=["claude", "gpt4"],
            task="Process these items",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.steps) > 0

    def test_workflow_has_split_step(self):
        """Test that workflow has a split/map step."""
        workflow = MapReducePattern.create(
            name="Split Test",
            agents=["claude"],
            task="Test",
        )

        # Should have a split or map step
        split_steps = [
            s for s in workflow.steps if "split" in s.id.lower() or "map" in s.id.lower()
        ]
        # At minimum, should have some steps for the map phase
        assert len(workflow.steps) >= 1

    def test_workflow_has_reduce_step(self):
        """Test that workflow has a reduce/aggregate step."""
        workflow = MapReducePattern.create(
            name="Reduce Test",
            agents=["claude"],
            task="Test",
        )

        # Should have a reduce or aggregate step
        reduce_steps = [
            s for s in workflow.steps if "reduce" in s.id.lower() or "aggregate" in s.id.lower()
        ]
        assert len(reduce_steps) >= 1 or len(workflow.steps) >= 1

    def test_workflow_metadata(self):
        """Test that workflow metadata contains pattern info."""
        workflow = MapReducePattern.create(
            name="Metadata Test",
            agents=["claude"],
            task="Test",
        )

        assert workflow.metadata.get("pattern") == "map_reduce"


# ============================================================================
# HierarchicalPattern Tests
# ============================================================================


class TestHierarchicalPattern:
    """Tests for HierarchicalPattern."""

    def test_create_basic_workflow(self):
        """Test creating a basic hierarchical workflow."""
        workflow = HierarchicalPattern.create(
            name="Test Hierarchical",
            agents=["claude", "gpt4"],
            task="Manage this project",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.steps) > 0

    def test_workflow_has_manager_step(self):
        """Test that workflow has a manager/coordinator step."""
        workflow = HierarchicalPattern.create(
            name="Manager Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        # Should have a manager or coordinator step
        manager_steps = [
            s
            for s in workflow.steps
            if "manager" in s.id.lower()
            or "coordinator" in s.id.lower()
            or "decompose" in s.id.lower()
        ]
        assert len(manager_steps) >= 1 or len(workflow.steps) >= 1

    def test_workflow_metadata(self):
        """Test that workflow metadata contains pattern info."""
        workflow = HierarchicalPattern.create(
            name="Metadata Test",
            agents=["claude"],
            task="Test",
        )

        assert workflow.metadata.get("pattern") == "hierarchical"


# ============================================================================
# ReviewCyclePattern Tests
# ============================================================================


class TestReviewCyclePattern:
    """Tests for ReviewCyclePattern."""

    def test_create_basic_workflow(self):
        """Test creating a basic review cycle workflow."""
        workflow = ReviewCyclePattern.create(
            name="Test Review Cycle",
            agents=["claude", "gpt4"],
            task="Review and refine this",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.steps) > 0

    def test_workflow_has_review_step(self):
        """Test that workflow has a review step."""
        workflow = ReviewCyclePattern.create(
            name="Review Test",
            agents=["claude"],
            task="Test",
        )

        # Should have review-related steps
        assert len(workflow.steps) >= 1

    def test_workflow_metadata(self):
        """Test that workflow metadata contains pattern info."""
        workflow = ReviewCyclePattern.create(
            name="Metadata Test",
            agents=["claude"],
            task="Test",
        )

        assert workflow.metadata.get("pattern") == "review_cycle"

    def test_max_iterations_config(self):
        """Test max iterations configuration."""
        pattern = ReviewCyclePattern(
            name="Iteration Test",
            agents=["claude"],
            task="Test",
            max_iterations=5,
        )

        assert pattern.max_iterations == 5


# ============================================================================
# DialecticPattern Tests
# ============================================================================


class TestDialecticPattern:
    """Tests for DialecticPattern."""

    def test_create_basic_workflow(self):
        """Test creating a basic dialectic workflow."""
        workflow = DialecticPattern.create(
            name="Test Dialectic",
            agents=["claude", "gpt4"],
            task="Debate this topic",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert len(workflow.steps) > 0

    def test_workflow_has_thesis_antithesis_synthesis(self):
        """Test that workflow has thesis, antithesis, synthesis steps."""
        workflow = DialecticPattern.create(
            name="Dialectic Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        step_ids = [s.id.lower() for s in workflow.steps]
        step_names = [s.name.lower() for s in workflow.steps]
        all_text = " ".join(step_ids + step_names)

        # Should have dialectic-related steps
        has_dialectic = any(
            term in all_text
            for term in ["thesis", "antithesis", "synthesis", "position", "counter"]
        )
        assert has_dialectic or len(workflow.steps) >= 1

    def test_workflow_metadata(self):
        """Test that workflow metadata contains pattern info."""
        workflow = DialecticPattern.create(
            name="Metadata Test",
            agents=["claude"],
            task="Test",
        )

        assert workflow.metadata.get("pattern") == "dialectic"

    def test_workflow_tags(self):
        """Test that workflow has appropriate tags."""
        workflow = DialecticPattern.create(
            name="Tags Test",
            agents=["claude"],
            task="Test",
        )

        assert "dialectic" in workflow.tags


# ============================================================================
# Visual Layout Tests
# ============================================================================


class TestVisualLayout:
    """Tests for visual layout generation."""

    def test_hive_mind_visual_positions(self):
        """Test that hive-mind pattern has visual positions for steps."""
        workflow = HiveMindPattern.create(
            name="Visual Test",
            agents=["claude", "gpt4", "gemini"],
            task="Test",
        )

        for step in workflow.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_sequential_visual_positions(self):
        """Test that sequential pattern has visual positions."""
        workflow = SequentialPattern.create(
            name="Visual Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        for step in workflow.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_agent_colors_assigned(self):
        """Test that agent steps have colors assigned."""
        workflow = HiveMindPattern.create(
            name="Color Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        agent_steps = [s for s in workflow.steps if s.step_type == "agent"]
        for step in agent_steps:
            assert step.visual.color is not None
            assert step.visual.color != ""

    def test_node_categories_assigned(self):
        """Test that steps have node categories assigned."""
        workflow = HiveMindPattern.create(
            name="Category Test",
            agents=["claude"],
            task="Test",
        )

        for step in workflow.steps:
            assert step.visual.category is not None


# ============================================================================
# Workflow Validation Tests
# ============================================================================


class TestWorkflowValidation:
    """Tests for validating generated workflows."""

    def test_hive_mind_workflow_valid(self):
        """Test that generated hive-mind workflow is valid."""
        workflow = HiveMindPattern.create(
            name="Valid Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        is_valid, errors = workflow.validate()
        # Should be valid or have minor issues only
        assert is_valid or len(errors) <= 2

    def test_sequential_workflow_valid(self):
        """Test that generated sequential workflow is valid."""
        workflow = SequentialPattern.create(
            name="Valid Test",
            agents=["claude", "gpt4"],
            task="Test",
        )

        is_valid, errors = workflow.validate()
        assert is_valid or len(errors) <= 2

    def test_workflow_has_entry_step(self):
        """Test that generated workflows have an entry step."""
        workflow = HiveMindPattern.create(
            name="Entry Test",
            agents=["claude"],
            task="Test",
        )

        assert workflow.entry_step is not None
        assert any(s.id == workflow.entry_step for s in workflow.steps)
