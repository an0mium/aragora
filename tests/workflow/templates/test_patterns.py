"""
Tests for Pattern-based Workflow Templates.

Tests the HiveMind, MapReduce, and ReviewCycle pattern templates.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.workflow.templates.patterns import (
    HIVE_MIND_TEMPLATE,
    MAP_REDUCE_TEMPLATE,
    REVIEW_CYCLE_TEMPLATE,
    PATTERN_TEMPLATES,
    create_hive_mind_workflow,
    create_map_reduce_workflow,
    create_review_cycle_workflow,
    get_pattern_template,
    list_pattern_templates,
)


# ============================================================================
# Template Definition Tests
# ============================================================================


class TestHiveMindTemplate:
    """Tests for Hive Mind template definition."""

    def test_template_structure(self):
        """Test Hive Mind template has required fields."""
        assert HIVE_MIND_TEMPLATE["id"] == "pattern/hive-mind"
        assert HIVE_MIND_TEMPLATE["name"] == "Hive Mind Analysis"
        assert "description" in HIVE_MIND_TEMPLATE
        assert HIVE_MIND_TEMPLATE["pattern"] == "hive_mind"
        assert HIVE_MIND_TEMPLATE["version"] == "1.0.0"

    def test_template_config(self):
        """Test Hive Mind template configuration."""
        config = HIVE_MIND_TEMPLATE["config"]

        assert "agents" in config
        assert "consensus_mode" in config
        assert "consensus_threshold" in config
        assert config["consensus_threshold"] == 0.7
        assert config["include_dissent"] is True

    def test_template_inputs(self):
        """Test Hive Mind template inputs."""
        inputs = HIVE_MIND_TEMPLATE["inputs"]

        assert "task" in inputs
        assert inputs["task"]["required"] is True
        assert "context" in inputs
        assert "data" in inputs

    def test_template_outputs(self):
        """Test Hive Mind template outputs."""
        outputs = HIVE_MIND_TEMPLATE["outputs"]

        assert "perspectives" in outputs
        assert "synthesis" in outputs
        assert "confidence" in outputs
        assert "dissent" in outputs

    def test_template_tags(self):
        """Test Hive Mind template tags."""
        tags = HIVE_MIND_TEMPLATE["tags"]

        assert "parallel" in tags
        assert "consensus" in tags
        assert "multi-agent" in tags


class TestMapReduceTemplate:
    """Tests for MapReduce template definition."""

    def test_template_structure(self):
        """Test MapReduce template has required fields."""
        assert MAP_REDUCE_TEMPLATE["id"] == "pattern/map-reduce"
        assert MAP_REDUCE_TEMPLATE["name"] == "MapReduce Processing"
        assert MAP_REDUCE_TEMPLATE["pattern"] == "map_reduce"

    def test_template_config(self):
        """Test MapReduce template configuration."""
        config = MAP_REDUCE_TEMPLATE["config"]

        assert config["split_strategy"] == "chunks"
        assert config["chunk_size"] == 4000
        assert "map_agent" in config
        assert "reduce_agent" in config
        assert config["parallel_limit"] == 5

    def test_template_inputs(self):
        """Test MapReduce template inputs."""
        inputs = MAP_REDUCE_TEMPLATE["inputs"]

        assert "input" in inputs
        assert inputs["input"]["required"] is True
        assert "task" in inputs
        assert "split_strategy" in inputs
        assert "chunk_size" in inputs

    def test_template_outputs(self):
        """Test MapReduce template outputs."""
        outputs = MAP_REDUCE_TEMPLATE["outputs"]

        assert "chunks" in outputs
        assert "map_results" in outputs
        assert "aggregated" in outputs
        assert "statistics" in outputs


class TestReviewCycleTemplate:
    """Tests for Review Cycle template definition."""

    def test_template_structure(self):
        """Test Review Cycle template has required fields."""
        assert REVIEW_CYCLE_TEMPLATE["id"] == "pattern/review-cycle"
        assert REVIEW_CYCLE_TEMPLATE["name"] == "Iterative Review Cycle"
        assert REVIEW_CYCLE_TEMPLATE["pattern"] == "review_cycle"

    def test_template_config(self):
        """Test Review Cycle template configuration."""
        config = REVIEW_CYCLE_TEMPLATE["config"]

        assert "draft_agent" in config
        assert "review_agent" in config
        assert config["max_iterations"] == 3
        assert config["convergence_threshold"] == 0.85
        assert "review_criteria" in config

    def test_template_inputs(self):
        """Test Review Cycle template inputs."""
        inputs = REVIEW_CYCLE_TEMPLATE["inputs"]

        assert "task" in inputs
        assert inputs["task"]["required"] is True
        assert "max_iterations" in inputs
        assert "threshold" in inputs

    def test_template_outputs(self):
        """Test Review Cycle template outputs."""
        outputs = REVIEW_CYCLE_TEMPLATE["outputs"]

        assert "final_output" in outputs
        assert "iterations" in outputs
        assert "final_score" in outputs
        assert "review_history" in outputs


# ============================================================================
# Pattern Templates Registry Tests
# ============================================================================


class TestPatternTemplatesRegistry:
    """Tests for pattern templates registry."""

    def test_all_patterns_in_registry(self):
        """Test all pattern templates are in registry."""
        assert "pattern/hive-mind" in PATTERN_TEMPLATES
        assert "pattern/map-reduce" in PATTERN_TEMPLATES
        assert "pattern/review-cycle" in PATTERN_TEMPLATES

    def test_registry_count(self):
        """Test registry has expected number of templates."""
        assert len(PATTERN_TEMPLATES) == 3

    def test_get_pattern_template(self):
        """Test getting pattern template by ID."""
        template = get_pattern_template("pattern/hive-mind")

        assert template is not None
        assert template["name"] == "Hive Mind Analysis"

    def test_get_pattern_template_not_found(self):
        """Test getting nonexistent template returns None."""
        template = get_pattern_template("nonexistent")

        assert template is None

    def test_list_pattern_templates(self):
        """Test listing all pattern templates."""
        templates = list_pattern_templates()

        assert len(templates) == 3
        assert all("id" in t for t in templates)
        assert all("name" in t for t in templates)
        assert all("pattern" in t for t in templates)

    def test_list_includes_category(self):
        """Test listed templates include category."""
        templates = list_pattern_templates()

        for t in templates:
            assert t["category"] == "pattern"


# ============================================================================
# Workflow Factory Tests
# ============================================================================


class TestCreateHiveMindWorkflow:
    """Tests for create_hive_mind_workflow factory."""

    def test_create_default_workflow(self):
        """Test creating workflow with defaults."""
        workflow = create_hive_mind_workflow()

        assert workflow is not None
        assert workflow.name == "Hive Mind Analysis"
        assert hasattr(workflow, "steps")
        assert len(workflow.steps) > 0

    def test_create_workflow_with_custom_name(self):
        """Test creating workflow with custom name."""
        workflow = create_hive_mind_workflow(name="Custom Analysis")

        assert workflow.name == "Custom Analysis"

    def test_create_workflow_with_custom_agents(self):
        """Test creating workflow with custom agents."""
        workflow = create_hive_mind_workflow(
            agents=["claude", "mistral"],
        )

        assert workflow is not None
        # Workflow should be configured for the specified agents

    def test_create_workflow_with_task(self):
        """Test creating workflow with task."""
        workflow = create_hive_mind_workflow(
            task="Analyze this document",
        )

        assert workflow is not None

    def test_create_workflow_with_consensus_options(self):
        """Test creating workflow with consensus options."""
        workflow = create_hive_mind_workflow(
            consensus_mode="weighted",
            consensus_threshold=0.8,
            include_dissent=False,
        )

        assert workflow is not None


class TestCreateMapReduceWorkflow:
    """Tests for create_map_reduce_workflow factory."""

    def test_create_default_workflow(self):
        """Test creating workflow with defaults."""
        workflow = create_map_reduce_workflow()

        assert workflow is not None
        assert workflow.name == "MapReduce Processing"
        assert len(workflow.steps) > 0

    def test_create_workflow_with_split_strategy(self):
        """Test creating workflow with split strategy."""
        workflow = create_map_reduce_workflow(
            split_strategy="lines",
            chunk_size=1000,
        )

        assert workflow is not None

    def test_create_workflow_with_agents(self):
        """Test creating workflow with custom agents."""
        workflow = create_map_reduce_workflow(
            map_agent="gpt4",
            reduce_agent="claude",
        )

        assert workflow is not None

    def test_create_workflow_with_prompts(self):
        """Test creating workflow with custom prompts."""
        workflow = create_map_reduce_workflow(
            map_prompt="Analyze this chunk: {chunk}",
            reduce_prompt="Aggregate these results: {results}",
        )

        assert workflow is not None

    def test_create_workflow_with_parallel_limit(self):
        """Test creating workflow with parallel limit."""
        workflow = create_map_reduce_workflow(
            parallel_limit=10,
        )

        assert workflow is not None


class TestCreateReviewCycleWorkflow:
    """Tests for create_review_cycle_workflow factory."""

    def test_create_default_workflow(self):
        """Test creating workflow with defaults."""
        workflow = create_review_cycle_workflow()

        assert workflow is not None
        assert workflow.name == "Iterative Review Cycle"
        assert len(workflow.steps) > 0

    def test_create_workflow_with_agents(self):
        """Test creating workflow with custom agents."""
        workflow = create_review_cycle_workflow(
            draft_agent="gpt4",
            review_agent="claude",
        )

        assert workflow is not None

    def test_create_workflow_with_iterations(self):
        """Test creating workflow with iteration settings."""
        workflow = create_review_cycle_workflow(
            max_iterations=5,
            convergence_threshold=0.9,
        )

        assert workflow is not None

    def test_create_workflow_with_criteria(self):
        """Test creating workflow with review criteria."""
        workflow = create_review_cycle_workflow(
            review_criteria=["correctness", "clarity", "efficiency"],
        )

        assert workflow is not None

    def test_create_workflow_with_task(self):
        """Test creating workflow with task."""
        workflow = create_review_cycle_workflow(
            task="Implement a sorting algorithm",
        )

        assert workflow is not None


# ============================================================================
# Workflow Definition Tests
# ============================================================================


class TestWorkflowDefinitions:
    """Tests for generated workflow definitions."""

    def test_hive_mind_has_entry_step(self):
        """Test Hive Mind workflow has entry step."""
        workflow = create_hive_mind_workflow()

        assert workflow.entry_step is not None
        assert workflow.entry_step != ""

    def test_map_reduce_has_entry_step(self):
        """Test MapReduce workflow has entry step."""
        workflow = create_map_reduce_workflow()

        assert workflow.entry_step is not None

    def test_review_cycle_has_entry_step(self):
        """Test Review Cycle workflow has entry step."""
        workflow = create_review_cycle_workflow()

        assert workflow.entry_step is not None

    def test_workflow_has_id(self):
        """Test generated workflow has ID."""
        workflow = create_hive_mind_workflow()

        assert workflow.id is not None
        assert len(workflow.id) > 0

    def test_workflow_steps_have_ids(self):
        """Test all workflow steps have IDs."""
        workflow = create_hive_mind_workflow()

        for step in workflow.steps:
            assert step.id is not None
            assert len(step.id) > 0

    def test_workflow_steps_have_types(self):
        """Test all workflow steps have types."""
        workflow = create_map_reduce_workflow()

        for step in workflow.steps:
            assert step.step_type is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestPatternIntegration:
    """Integration tests for patterns."""

    def test_template_matches_factory_output(self):
        """Test template config is reflected in factory output."""
        template = HIVE_MIND_TEMPLATE
        workflow = create_hive_mind_workflow()

        # Workflow should reflect template settings
        assert workflow.name == template["name"]

    def test_all_patterns_create_valid_workflows(self):
        """Test all patterns create valid workflows."""
        factories = [
            create_hive_mind_workflow,
            create_map_reduce_workflow,
            create_review_cycle_workflow,
        ]

        for factory in factories:
            workflow = factory()
            assert workflow is not None
            assert len(workflow.steps) > 0
            assert workflow.entry_step is not None

    def test_patterns_are_reusable(self):
        """Test patterns can create multiple workflows."""
        workflows = [create_hive_mind_workflow(name=f"Workflow {i}") for i in range(3)]

        # Each workflow should be independent
        ids = [w.id for w in workflows]
        assert len(set(ids)) == 3  # All unique IDs


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_agents_list(self):
        """Test creating workflow with empty agents defaults to standard."""
        workflow = create_hive_mind_workflow(agents=[])

        # Should use default agents or handle gracefully
        assert workflow is not None

    def test_single_agent(self):
        """Test creating hive mind with single agent."""
        workflow = create_hive_mind_workflow(agents=["claude"])

        assert workflow is not None

    def test_very_low_threshold(self):
        """Test creating workflow with very low threshold."""
        workflow = create_review_cycle_workflow(
            convergence_threshold=0.1,
        )

        assert workflow is not None

    def test_very_high_threshold(self):
        """Test creating workflow with very high threshold."""
        workflow = create_review_cycle_workflow(
            convergence_threshold=0.99,
        )

        assert workflow is not None

    def test_large_chunk_size(self):
        """Test creating workflow with large chunk size."""
        workflow = create_map_reduce_workflow(
            chunk_size=100000,
        )

        assert workflow is not None

    def test_many_iterations(self):
        """Test creating workflow with many iterations."""
        workflow = create_review_cycle_workflow(
            max_iterations=20,
        )

        assert workflow is not None
