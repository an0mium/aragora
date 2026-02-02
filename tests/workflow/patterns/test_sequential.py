"""
Tests for Sequential Workflow Pattern.

Tests cover:
- SequentialPattern initialization and defaults
- Workflow generation (ID, name, steps, entry step)
- Stage configuration from agents and explicit stages
- Per-agent prompts and fallback to task
- pass_full_context behavior
- Single agent edge case (no output step)
- Visual metadata (positions, categories, colors)
- Transition rules and next_steps linkage
- Workflow tags and metadata
- Factory classmethod
- _build_context_prompt helper
"""

import pytest


# ============================================================================
# SequentialPattern Initialization Tests
# ============================================================================


class TestSequentialPatternInit:
    """Tests for SequentialPattern initialization."""

    def test_default_init(self):
        """Test default initialization with minimal parameters."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(name="Test Sequential")
        assert pattern.name == "Test Sequential"
        assert pattern.agents == ["claude", "gpt4"]
        assert pattern.task == ""
        assert pattern.prompts == {}
        assert pattern.stages is None
        assert pattern.pass_full_context is True
        assert pattern.timeout_per_step == 120.0

    def test_custom_agents(self):
        """Test initialization with custom agents list."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(
            name="Custom",
            agents=["gemini", "mistral", "grok"],
        )
        assert pattern.agents == ["gemini", "mistral", "grok"]

    def test_custom_task(self):
        """Test initialization with a task string."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(name="Test", task="Analyze security issues")
        assert pattern.task == "Analyze security issues"

    def test_custom_prompts(self):
        """Test initialization with per-agent prompts."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        prompts = {
            "claude": "Extract key facts from: {input}",
            "gpt4": "Analyze the extracted facts: {step.claude}",
        }
        pattern = SequentialPattern(name="Test", prompts=prompts)
        assert pattern.prompts == prompts

    def test_prompts_default_to_empty_dict(self):
        """Test that prompts default to empty dict when None."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(name="Test", prompts=None)
        assert pattern.prompts == {}

    def test_custom_stages(self):
        """Test initialization with explicit stages."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        stages = [
            {"agent": "claude", "role": "reviewer", "focus": "security"},
            {"agent": "gpt4", "role": "synthesizer", "focus": "summary"},
        ]
        pattern = SequentialPattern(name="Test", stages=stages)
        assert pattern.stages == stages

    def test_pass_full_context_false(self):
        """Test initialization with pass_full_context disabled."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(name="Test", pass_full_context=False)
        assert pattern.pass_full_context is False

    def test_custom_timeout(self):
        """Test initialization with custom timeout_per_step."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(name="Test", timeout_per_step=60.0)
        assert pattern.timeout_per_step == 60.0

    def test_pattern_type(self):
        """Test that pattern_type is SEQUENTIAL."""
        from aragora.workflow.patterns.sequential import SequentialPattern
        from aragora.workflow.patterns.base import PatternType

        assert SequentialPattern.pattern_type == PatternType.SEQUENTIAL

    def test_kwargs_stored_in_config(self):
        """Test that extra kwargs are stored in self.config."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        pattern = SequentialPattern(
            name="Test",
            tags=["custom_tag"],
            category="code",
        )
        assert pattern.config.get("tags") == ["custom_tag"]
        assert pattern.config.get("category") == "code"


# ============================================================================
# Workflow Generation Tests
# ============================================================================


class TestSequentialWorkflowGeneration:
    """Tests for sequential workflow generation."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with seq_."""
        wf = self._create_workflow()
        assert wf.id.startswith("seq_")

    def test_workflow_id_unique(self):
        """Test that successive workflow IDs are unique."""
        wf1 = self._create_workflow()
        wf2 = self._create_workflow()
        assert wf1.id != wf2.id

    def test_workflow_name(self):
        """Test workflow name is set correctly."""
        wf = self._create_workflow(name="Code Review Pipeline")
        assert wf.name == "Code Review Pipeline"

    def test_workflow_description(self):
        """Test workflow description includes stage count."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert "3" in wf.description
        assert "stages" in wf.description.lower()

    def test_step_count_with_default_agents(self):
        """Test workflow has agents + output steps (2 agents + 1 output = 3)."""
        wf = self._create_workflow()
        assert len(wf.steps) == 3  # 2 agent steps + 1 output step

    def test_step_count_with_three_agents(self):
        """Test workflow with 3 agents has 4 steps."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert len(wf.steps) == 4  # 3 agent steps + 1 output step

    def test_entry_step_is_first_stage(self):
        """Test that entry step is the first agent stage."""
        wf = self._create_workflow()
        assert wf.entry_step == "stage_0_claude"

    def test_entry_step_with_custom_agents(self):
        """Test entry step with custom agent order."""
        wf = self._create_workflow(agents=["gemini", "mistral"])
        assert wf.entry_step == "stage_0_gemini"

    def test_workflow_category_default(self):
        """Test workflow category defaults to GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL

    def test_workflow_category_custom(self):
        """Test custom workflow category via kwargs."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow(category=WorkflowCategory.CODE)
        assert wf.category == WorkflowCategory.CODE


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestSequentialStepConfig:
    """Tests for step configuration details."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_agent_step_ids(self):
        """Test that agent steps have correct IDs based on role."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "stage_0_claude" in step_ids
        assert "stage_1_gpt4" in step_ids

    def test_agent_steps_are_agent_type(self):
        """Test that agent steps have step_type 'agent'."""
        wf = self._create_workflow()
        for step in wf.steps:
            if step.id.startswith("stage_"):
                assert step.step_type == "agent"

    def test_agent_step_config_has_agent_type(self):
        """Test that agent step config includes agent_type."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert step.config["agent_type"] == "claude"

    def test_agent_step_config_has_prompt_template(self):
        """Test that agent step config includes prompt_template."""
        wf = self._create_workflow(task="Analyze this code")
        step = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert "prompt_template" in step.config

    def test_agent_step_timeout(self):
        """Test that agent steps use configured timeout."""
        wf = self._create_workflow(timeout_per_step=90.0)
        step = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert step.timeout_seconds == 90.0

    def test_step_names_include_role(self):
        """Test that step names include the role in title case."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert "Claude" in step.name
        assert "Stage 1" in step.name

    def test_output_step_exists(self):
        """Test that output aggregation step is created for multi-agent."""
        wf = self._create_workflow()
        output_step = next((s for s in wf.steps if s.id == "output"), None)
        assert output_step is not None

    def test_output_step_is_task_type(self):
        """Test that output step has step_type 'task'."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "output")
        assert step.step_type == "task"

    def test_output_step_has_transform_config(self):
        """Test that output step has transform task_type."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "output")
        assert step.config["task_type"] == "transform"
        assert "output_format" in step.config

    def test_output_step_name(self):
        """Test output step name is 'Final Output'."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "output")
        assert step.name == "Final Output"


# ============================================================================
# Stages Configuration Tests
# ============================================================================


class TestSequentialStagesConfig:
    """Tests for explicit stages configuration."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_stages_override_agents(self):
        """Test that stages config overrides agent list for step creation."""
        stages = [
            {"agent": "claude", "role": "security_reviewer", "focus": "security"},
            {"agent": "gpt4", "role": "performance_reviewer", "focus": "performance"},
        ]
        wf = self._create_workflow(stages=stages)
        step_ids = [s.id for s in wf.steps if s.id.startswith("stage_")]
        assert "stage_0_security_reviewer" in step_ids
        assert "stage_1_performance_reviewer" in step_ids

    def test_stages_use_agent_type(self):
        """Test that stages correctly assign agent_type from config."""
        stages = [
            {"agent": "gemini", "role": "analyst"},
            {"agent": "mistral", "role": "writer"},
        ]
        wf = self._create_workflow(stages=stages)
        step0 = next(s for s in wf.steps if s.id == "stage_0_analyst")
        step1 = next(s for s in wf.steps if s.id == "stage_1_writer")
        assert step0.config["agent_type"] == "gemini"
        assert step1.config["agent_type"] == "mistral"

    def test_stages_apply_focus(self):
        """Test that stages with focus have it in config."""
        stages = [
            {"agent": "claude", "role": "reviewer", "focus": "security"},
        ]
        wf = self._create_workflow(stages=stages)
        step = next(s for s in wf.steps if s.id == "stage_0_reviewer")
        assert step.config.get("focus") == "security"

    def test_stages_without_focus(self):
        """Test that stages without focus do not add focus to config."""
        stages = [
            {"agent": "claude", "role": "reviewer"},
        ]
        wf = self._create_workflow(stages=stages)
        step = next(s for s in wf.steps if s.id == "stage_0_reviewer")
        assert "focus" not in step.config

    def test_stages_use_custom_prompt(self):
        """Test that stages with prompt use it instead of task."""
        stages = [
            {"agent": "claude", "role": "analyst", "prompt": "Custom stage prompt"},
        ]
        wf = self._create_workflow(stages=stages, task="Default task")
        step = next(s for s in wf.steps if s.id == "stage_0_analyst")
        # Since it's the first stage, no context prefix should be added
        assert "Custom stage prompt" in step.config["prompt_template"]

    def test_stages_fallback_to_task(self):
        """Test that stages without prompt fall back to task."""
        stages = [
            {"agent": "claude", "role": "analyst"},
        ]
        wf = self._create_workflow(stages=stages, task="Default task")
        step = next(s for s in wf.steps if s.id == "stage_0_analyst")
        assert "Default task" in step.config["prompt_template"]

    def test_stages_step_names_use_role(self):
        """Test that step names use stage role in title case."""
        stages = [
            {"agent": "claude", "role": "security_reviewer"},
        ]
        wf = self._create_workflow(stages=stages)
        step = next(s for s in wf.steps if s.id == "stage_0_security_reviewer")
        assert "Security Reviewer" in step.name

    def test_three_stages_creates_four_steps(self):
        """Test that 3 stages creates 3 agent steps + 1 output = 4 steps."""
        stages = [
            {"agent": "claude", "role": "a"},
            {"agent": "gpt4", "role": "b"},
            {"agent": "gemini", "role": "c"},
        ]
        wf = self._create_workflow(stages=stages)
        assert len(wf.steps) == 4


# ============================================================================
# Prompt Configuration Tests
# ============================================================================


class TestSequentialPrompts:
    """Tests for per-agent prompts and task fallback."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_per_agent_prompts_used(self):
        """Test that per-agent prompts are assigned correctly."""
        prompts = {
            "claude": "Extract facts from: {input}",
            "gpt4": "Analyze facts: {step.claude}",
        }
        wf = self._create_workflow(prompts=prompts, pass_full_context=False)
        step0 = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert step0.config["prompt_template"] == "Extract facts from: {input}"

    def test_missing_agent_prompt_falls_back_to_task(self):
        """Test that agents without prompts use the task string."""
        prompts = {"claude": "Claude-specific prompt"}
        wf = self._create_workflow(
            prompts=prompts,
            task="Fallback task",
            pass_full_context=False,
        )
        step1 = next(s for s in wf.steps if s.id == "stage_1_gpt4")
        assert step1.config["prompt_template"] == "Fallback task"

    def test_empty_prompts_uses_task(self):
        """Test that empty prompts dict causes all agents to use task."""
        wf = self._create_workflow(
            prompts={},
            task="Shared task",
            pass_full_context=False,
        )
        step0 = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert step0.config["prompt_template"] == "Shared task"


# ============================================================================
# Context Passing Tests
# ============================================================================


class TestSequentialContextPassing:
    """Tests for pass_full_context behavior."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_first_step_has_no_context_prefix(self):
        """Test that the first step never has context prefix."""
        wf = self._create_workflow(task="Analyze code", pass_full_context=True)
        step0 = next(s for s in wf.steps if s.id == "stage_0_claude")
        assert "Previous analysis" not in step0.config["prompt_template"]

    def test_second_step_has_context_prefix_when_enabled(self):
        """Test that second step has context prefix when pass_full_context=True."""
        wf = self._create_workflow(task="Analyze code", pass_full_context=True)
        step1 = next(s for s in wf.steps if s.id == "stage_1_gpt4")
        assert "Previous analysis" in step1.config["prompt_template"]
        assert "stage_0_claude" in step1.config["prompt_template"]

    def test_second_step_no_context_prefix_when_disabled(self):
        """Test that second step has no context prefix when pass_full_context=False."""
        wf = self._create_workflow(task="Analyze code", pass_full_context=False)
        step1 = next(s for s in wf.steps if s.id == "stage_1_gpt4")
        assert "Previous analysis" not in step1.config["prompt_template"]

    def test_context_prefix_references_previous_step(self):
        """Test that context prefix references the correct previous step ID."""
        wf = self._create_workflow(
            agents=["claude", "gpt4", "gemini"],
            task="Review",
            pass_full_context=True,
        )
        step2 = next(s for s in wf.steps if s.id == "stage_2_gemini")
        # Should reference stage_1_gpt4, not stage_0_claude
        assert "stage_1_gpt4" in step2.config["prompt_template"]

    def test_context_with_focus_includes_focus_area(self):
        """Test that context prompt includes focus area when specified."""
        stages = [
            {"agent": "claude", "role": "analyst"},
            {"agent": "gpt4", "role": "reviewer", "focus": "performance"},
        ]
        wf = self._create_workflow(
            stages=stages,
            task="Review code",
            pass_full_context=True,
        )
        step1 = next(s for s in wf.steps if s.id == "stage_1_reviewer")
        assert "performance" in step1.config["prompt_template"]


# ============================================================================
# Single Agent Edge Case Tests
# ============================================================================


class TestSequentialSingleAgent:
    """Tests for single-agent edge case."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_single_agent_no_output_step(self):
        """Test that single agent workflow has no output step."""
        wf = self._create_workflow(agents=["claude"])
        assert len(wf.steps) == 1
        assert wf.steps[0].id == "stage_0_claude"

    def test_single_agent_no_transitions(self):
        """Test that single agent workflow has no transitions."""
        wf = self._create_workflow(agents=["claude"])
        assert len(wf.transitions) == 0

    def test_single_agent_entry_step(self):
        """Test entry step for single agent workflow."""
        wf = self._create_workflow(agents=["claude"])
        assert wf.entry_step == "stage_0_claude"

    def test_single_stage_no_output_step(self):
        """Test that single stage config also produces no output step."""
        stages = [{"agent": "gemini", "role": "solo"}]
        wf = self._create_workflow(stages=stages)
        assert len(wf.steps) == 1
        assert wf.steps[0].id == "stage_0_solo"


# ============================================================================
# Transition Tests
# ============================================================================


class TestSequentialTransitions:
    """Tests for transition rules between steps."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_transition_count_two_agents(self):
        """Test transition count for 2 agents (2: stage0->stage1, stage1->output)."""
        wf = self._create_workflow()
        assert len(wf.transitions) == 2

    def test_transition_count_three_agents(self):
        """Test transition count for 3 agents (3 transitions)."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert len(wf.transitions) == 3

    def test_linear_transition_flow(self):
        """Test that transitions form a linear chain."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["stage_0_claude"] == "stage_1_gpt4"
        assert flow["stage_1_gpt4"] == "stage_2_gemini"
        assert flow["stage_2_gemini"] == "output"

    def test_transitions_have_ids(self):
        """Test that all transitions have IDs starting with tr_."""
        wf = self._create_workflow()
        for t in wf.transitions:
            assert t.id.startswith("tr_")

    def test_transition_default_condition(self):
        """Test that transitions have default condition 'True'."""
        wf = self._create_workflow()
        for t in wf.transitions:
            assert t.condition == "True"

    def test_next_steps_linkage(self):
        """Test that next_steps are set on each agent step."""
        wf = self._create_workflow()
        step0 = next(s for s in wf.steps if s.id == "stage_0_claude")
        step1 = next(s for s in wf.steps if s.id == "stage_1_gpt4")
        assert step0.next_steps == ["stage_1_gpt4"]
        assert step1.next_steps == ["output"]

    def test_last_agent_next_steps_point_to_output(self):
        """Test that the last agent step has next_steps pointing to output."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        step2 = next(s for s in wf.steps if s.id == "stage_2_gemini")
        assert step2.next_steps == ["output"]

    def test_output_transform_references_last_stage(self):
        """Test that output step transform references last agent step."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        output_step = next(s for s in wf.steps if s.id == "output")
        assert "stage_2_gemini" in output_step.config["transform"]


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestSequentialVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        wf = self._create_workflow()
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_agent_steps_have_agent_category(self):
        """Test that agent steps have AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        for step in wf.steps:
            if step.id.startswith("stage_"):
                assert step.visual.category == NodeCategory.AGENT

    def test_output_step_has_task_category(self):
        """Test that output step has TASK category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        output_step = next(s for s in wf.steps if s.id == "output")
        assert output_step.visual.category == NodeCategory.TASK

    def test_positions_spaced_by_250(self):
        """Test that step positions are spaced 250px apart horizontally."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        positions = [(s.id, s.visual.position.x) for s in wf.steps]
        # Steps should be at x=100, 350, 600, 850
        stage_positions = [(sid, x) for sid, x in positions if sid.startswith("stage_")]
        stage_positions.sort(key=lambda p: p[1])
        assert stage_positions[0][1] == 100
        assert stage_positions[1][1] == 350
        assert stage_positions[2][1] == 600

    def test_output_step_position_after_last_stage(self):
        """Test that output step is positioned after the last stage."""
        wf = self._create_workflow(agents=["claude", "gpt4"])
        output_step = next(s for s in wf.steps if s.id == "output")
        # 2 agents => output at start_x + 2 * spacing = 100 + 2*250 = 600
        assert output_step.visual.position.x == 600

    def test_all_steps_same_y_position(self):
        """Test that all steps have the same y position."""
        wf = self._create_workflow()
        y_positions = {s.visual.position.y for s in wf.steps}
        assert len(y_positions) == 1
        assert 200 in y_positions

    def test_agent_step_color_matches_agent_type(self):
        """Test that agent step colors match their agent type."""
        wf = self._create_workflow()
        step_claude = next(s for s in wf.steps if s.id == "stage_0_claude")
        # Claude color is #7c3aed (purple)
        assert step_claude.visual.color == "#7c3aed"

    def test_transitions_have_visual_edge_data(self):
        """Test that transitions have visual edge data."""
        from aragora.workflow.types import EdgeType

        wf = self._create_workflow()
        for t in wf.transitions:
            assert t.visual is not None
            assert t.visual.edge_type == EdgeType.DATA_FLOW


# ============================================================================
# Tags and Metadata Tests
# ============================================================================


class TestSequentialTagsAndMetadata:
    """Tests for workflow tags and metadata."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        pattern = SequentialPattern(**defaults)
        return pattern.create_workflow()

    def test_default_tags(self):
        """Test that default tags include sequential and pipeline."""
        wf = self._create_workflow()
        assert "sequential" in wf.tags
        assert "pipeline" in wf.tags

    def test_custom_tags_appended(self):
        """Test that custom tags are appended to default tags."""
        wf = self._create_workflow(tags=["security", "code_review"])
        assert "sequential" in wf.tags
        assert "pipeline" in wf.tags
        assert "security" in wf.tags
        assert "code_review" in wf.tags

    def test_metadata_pattern(self):
        """Test that metadata includes pattern='sequential'."""
        wf = self._create_workflow()
        assert wf.metadata["pattern"] == "sequential"

    def test_metadata_stages_count(self):
        """Test that metadata includes correct stages count."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert wf.metadata["stages"] == 3

    def test_metadata_stages_count_with_stages_config(self):
        """Test stages count matches explicit stages config."""
        stages = [
            {"agent": "claude", "role": "a"},
            {"agent": "gpt4", "role": "b"},
            {"agent": "gemini", "role": "c"},
            {"agent": "mistral", "role": "d"},
        ]
        wf = self._create_workflow(stages=stages)
        assert wf.metadata["stages"] == 4

    def test_metadata_pass_full_context_true(self):
        """Test that metadata includes pass_full_context when True."""
        wf = self._create_workflow(pass_full_context=True)
        assert wf.metadata["pass_full_context"] is True

    def test_metadata_pass_full_context_false(self):
        """Test that metadata includes pass_full_context when False."""
        wf = self._create_workflow(pass_full_context=False)
        assert wf.metadata["pass_full_context"] is False


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestSequentialFactory:
    """Tests for factory classmethod."""

    def test_create_classmethod(self):
        """Test SequentialPattern.create factory method returns WorkflowDefinition."""
        from aragora.workflow.patterns.sequential import SequentialPattern
        from aragora.workflow.types import WorkflowDefinition

        wf = SequentialPattern.create(
            name="Factory Test",
            agents=["claude", "gpt4"],
            task="Analyze code",
        )
        assert isinstance(wf, WorkflowDefinition)

    def test_create_classmethod_with_stages(self):
        """Test create factory with stages configuration."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        stages = [
            {"agent": "claude", "role": "reviewer", "focus": "security"},
            {"agent": "gpt4", "role": "synthesizer"},
        ]
        wf = SequentialPattern.create(
            name="Staged Pipeline",
            stages=stages,
            task="Review this code",
        )
        assert wf.metadata["pattern"] == "sequential"
        assert wf.metadata["stages"] == 2
        assert wf.id.startswith("seq_")

    def test_create_classmethod_with_all_options(self):
        """Test create factory with all configuration options."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        wf = SequentialPattern.create(
            name="Full Config",
            agents=["gemini", "mistral", "grok"],
            task="Comprehensive analysis",
            prompts={
                "gemini": "Extract: {input}",
                "mistral": "Analyze: {step.gemini}",
            },
            pass_full_context=False,
            timeout_per_step=60.0,
            tags=["custom"],
        )
        assert wf.name == "Full Config"
        assert len(wf.steps) == 4  # 3 agents + 1 output
        assert "custom" in wf.tags
        assert wf.metadata["pass_full_context"] is False


# ============================================================================
# _build_context_prompt Tests
# ============================================================================


class TestBuildContextPrompt:
    """Tests for _build_context_prompt helper method."""

    def _get_pattern(self, **kwargs):
        """Helper to create a SequentialPattern instance."""
        from aragora.workflow.patterns.sequential import SequentialPattern

        defaults = {"name": "Test Sequential"}
        defaults.update(kwargs)
        return SequentialPattern(**defaults)

    def test_includes_previous_step_reference(self):
        """Test that context prompt includes step output reference."""
        pattern = self._get_pattern()
        result = pattern._build_context_prompt("Analyze", "stage_0_claude", "")
        assert "{step.stage_0_claude}" in result

    def test_includes_previous_analysis_label(self):
        """Test that context prompt includes 'Previous analysis' label."""
        pattern = self._get_pattern()
        result = pattern._build_context_prompt("Analyze", "stage_0_claude", "")
        assert "Previous analysis" in result

    def test_original_prompt_preserved(self):
        """Test that the original prompt text is preserved."""
        pattern = self._get_pattern()
        result = pattern._build_context_prompt("Analyze the security issues", "stage_0_claude", "")
        assert "Analyze the security issues" in result

    def test_focus_included_when_provided(self):
        """Test that focus area is included when specified."""
        pattern = self._get_pattern()
        result = pattern._build_context_prompt("Review code", "stage_0_claude", "security")
        assert "security" in result
        assert "focus area" in result.lower()

    def test_focus_not_included_when_empty(self):
        """Test that focus section is omitted when focus is empty."""
        pattern = self._get_pattern()
        result = pattern._build_context_prompt("Review code", "stage_0_claude", "")
        assert "focus area" not in result.lower()

    def test_context_prompt_with_different_step_ids(self):
        """Test context prompt works with various step IDs."""
        pattern = self._get_pattern()
        result = pattern._build_context_prompt(
            "Synthesize", "stage_2_performance_reviewer", "performance"
        )
        assert "{step.stage_2_performance_reviewer}" in result
        assert "performance" in result
