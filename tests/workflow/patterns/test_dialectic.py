"""
Tests for Dialectic Workflow Pattern.

Tests cover:
- DialecticPattern initialization and defaults
- Agent assignment (thesis, antithesis, synthesis)
- Thesis stance configuration (supportive, critical, neutral)
- Meta-analysis toggle
- Workflow generation with all steps
- Step configurations, types, and system prompts
- Transition rules between steps
- Visual metadata for workflow builder
- Prompt building for thesis, antithesis, synthesis
- Workflow metadata and tags
- Factory classmethod
- Custom prompts and agent assignments
"""

import pytest


# ============================================================================
# DialecticPattern Initialization Tests
# ============================================================================


class TestDialecticPatternInit:
    """Tests for DialecticPattern initialization."""

    def test_default_init(self):
        """Test default initialization with minimal parameters."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(name="Test Dialectic")
        assert pattern.name == "Test Dialectic"
        assert pattern.task == ""
        assert pattern.thesis_agent == "claude"
        assert pattern.antithesis_agent == "gpt4"
        assert pattern.synthesis_agent == "claude"
        assert pattern.thesis_prompt == ""
        assert pattern.antithesis_prompt == ""
        assert pattern.synthesis_prompt == ""
        assert pattern.thesis_stance == "supportive"
        assert pattern.include_meta_analysis is True
        assert pattern.timeout_per_step == 120.0

    def test_pattern_type(self):
        """Test that pattern_type is DIALECTIC."""
        from aragora.workflow.patterns.dialectic import DialecticPattern
        from aragora.workflow.patterns.base import PatternType

        assert DialecticPattern.pattern_type == PatternType.DIALECTIC

    def test_custom_task(self):
        """Test initialization with a custom task."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Ethics Debate",
            task="Should AI have legal personhood?",
        )
        assert pattern.task == "Should AI have legal personhood?"

    def test_custom_agents_list(self):
        """Test that agents list determines default agent assignments."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Custom Agents",
            agents=["gemini", "mistral", "grok"],
        )
        assert pattern.thesis_agent == "gemini"
        assert pattern.antithesis_agent == "mistral"
        assert pattern.synthesis_agent == "grok"

    def test_explicit_thesis_agent(self):
        """Test explicit thesis agent overrides agents list."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Explicit",
            agents=["gpt4", "gemini", "mistral"],
            thesis_agent="claude",
        )
        assert pattern.thesis_agent == "claude"

    def test_explicit_antithesis_agent(self):
        """Test explicit antithesis agent overrides agents list."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Explicit",
            agents=["claude", "gemini", "mistral"],
            antithesis_agent="grok",
        )
        assert pattern.antithesis_agent == "grok"

    def test_explicit_synthesis_agent(self):
        """Test explicit synthesis agent overrides agents list."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Explicit",
            agents=["claude", "gpt4", "gemini"],
            synthesis_agent="deepseek",
        )
        assert pattern.synthesis_agent == "deepseek"

    def test_agents_list_with_single_agent(self):
        """Test agents list with only one agent falls back correctly."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Single",
            agents=["claude"],
        )
        assert pattern.thesis_agent == "claude"
        assert pattern.antithesis_agent == "claude"
        assert pattern.synthesis_agent == "claude"

    def test_agents_list_with_two_agents(self):
        """Test agents list with two agents assigns synthesis to first."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Two",
            agents=["gemini", "mistral"],
        )
        assert pattern.thesis_agent == "gemini"
        assert pattern.antithesis_agent == "mistral"
        assert pattern.synthesis_agent == "gemini"

    def test_custom_prompts(self):
        """Test custom thesis, antithesis, and synthesis prompts."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(
            name="Custom Prompts",
            thesis_prompt="Argue for: {task}",
            antithesis_prompt="Argue against: {task}",
            synthesis_prompt="Reconcile: {task}",
        )
        assert pattern.thesis_prompt == "Argue for: {task}"
        assert pattern.antithesis_prompt == "Argue against: {task}"
        assert pattern.synthesis_prompt == "Reconcile: {task}"

    def test_thesis_stance_configuration(self):
        """Test thesis_stance parameter for all valid stances."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        for stance in ["supportive", "critical", "neutral"]:
            pattern = DialecticPattern(name="Test", thesis_stance=stance)
            assert pattern.thesis_stance == stance

    def test_meta_analysis_disabled(self):
        """Test disabling meta-analysis step."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(name="No Meta", include_meta_analysis=False)
        assert pattern.include_meta_analysis is False

    def test_custom_timeout(self):
        """Test custom timeout per step."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        pattern = DialecticPattern(name="Fast", timeout_per_step=30.0)
        assert pattern.timeout_per_step == 30.0


# ============================================================================
# Workflow Generation Tests
# ============================================================================


class TestDialecticWorkflowGeneration:
    """Tests for dialectic workflow generation."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        defaults = {"name": "Test Dialectic"}
        defaults.update(kwargs)
        pattern = DialecticPattern(**defaults)
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with dial_."""
        wf = self._create_workflow()
        assert wf.id.startswith("dial_")

    def test_workflow_name(self):
        """Test workflow name is set correctly."""
        wf = self._create_workflow(name="AI Ethics Analysis")
        assert wf.name == "AI Ethics Analysis"

    def test_workflow_description(self):
        """Test workflow description mentions dialectic pattern."""
        wf = self._create_workflow()
        assert "dialectic" in wf.description.lower()

    def test_workflow_has_core_steps(self):
        """Test that all required core steps are present."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "thesis" in step_ids
        assert "antithesis" in step_ids
        assert "synthesis" in step_ids

    def test_step_count_with_meta_analysis(self):
        """Test workflow has 4 steps when meta-analysis is enabled."""
        wf = self._create_workflow(include_meta_analysis=True)
        assert len(wf.steps) == 4
        step_ids = [s.id for s in wf.steps]
        assert "meta_analysis" in step_ids

    def test_step_count_without_meta_analysis(self):
        """Test workflow has 3 steps when meta-analysis is disabled."""
        wf = self._create_workflow(include_meta_analysis=False)
        assert len(wf.steps) == 3
        step_ids = [s.id for s in wf.steps]
        assert "meta_analysis" not in step_ids

    def test_entry_step_is_thesis(self):
        """Test that entry step is thesis."""
        wf = self._create_workflow()
        assert wf.entry_step == "thesis"

    def test_transition_count_with_meta_analysis(self):
        """Test workflow has 3 transitions when meta-analysis is enabled."""
        wf = self._create_workflow(include_meta_analysis=True)
        assert len(wf.transitions) == 3

    def test_transition_count_without_meta_analysis(self):
        """Test workflow has 2 transitions when meta-analysis is disabled."""
        wf = self._create_workflow(include_meta_analysis=False)
        assert len(wf.transitions) == 2

    def test_transition_flow_core(self):
        """Test the core transition flow: thesis -> antithesis -> synthesis."""
        wf = self._create_workflow(include_meta_analysis=False)
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["thesis"] == "antithesis"
        assert flow["antithesis"] == "synthesis"

    def test_transition_flow_with_meta(self):
        """Test full transition flow including meta-analysis."""
        wf = self._create_workflow(include_meta_analysis=True)
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["thesis"] == "antithesis"
        assert flow["antithesis"] == "synthesis"
        assert flow["synthesis"] == "meta_analysis"

    def test_workflow_tags(self):
        """Test workflow tags include dialectic-related tags."""
        wf = self._create_workflow()
        assert "dialectic" in wf.tags
        assert "debate" in wf.tags
        assert "synthesis" in wf.tags

    def test_custom_tags_included(self):
        """Test custom tags are merged with default tags."""
        wf = self._create_workflow(tags=["policy", "ethics"])
        assert "dialectic" in wf.tags
        assert "debate" in wf.tags
        assert "policy" in wf.tags
        assert "ethics" in wf.tags

    def test_workflow_metadata(self):
        """Test workflow metadata includes pattern information."""
        wf = self._create_workflow(
            thesis_agent="gemini",
            antithesis_agent="mistral",
            synthesis_agent="grok",
            thesis_stance="critical",
        )
        assert wf.metadata["pattern"] == "dialectic"
        assert wf.metadata["thesis_agent"] == "gemini"
        assert wf.metadata["antithesis_agent"] == "mistral"
        assert wf.metadata["synthesis_agent"] == "grok"
        assert wf.metadata["thesis_stance"] == "critical"

    def test_workflow_category_defaults_to_general(self):
        """Test workflow category defaults to GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL

    def test_custom_workflow_category(self):
        """Test custom workflow category."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow(category=WorkflowCategory.LEGAL)
        assert wf.category == WorkflowCategory.LEGAL


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestDialecticStepConfig:
    """Tests for step configuration details."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        defaults = {"name": "Test Dialectic"}
        defaults.update(kwargs)
        pattern = DialecticPattern(**defaults)
        return pattern.create_workflow()

    def _get_step(self, wf, step_id):
        """Helper to get a step by ID."""
        return next(s for s in wf.steps if s.id == step_id)

    def test_thesis_step_type(self):
        """Test thesis step is an agent step."""
        wf = self._create_workflow()
        step = self._get_step(wf, "thesis")
        assert step.step_type == "agent"

    def test_antithesis_step_type(self):
        """Test antithesis step is an agent step."""
        wf = self._create_workflow()
        step = self._get_step(wf, "antithesis")
        assert step.step_type == "agent"

    def test_synthesis_step_type(self):
        """Test synthesis step is an agent step."""
        wf = self._create_workflow()
        step = self._get_step(wf, "synthesis")
        assert step.step_type == "agent"

    def test_meta_analysis_step_type(self):
        """Test meta-analysis step is a task step."""
        wf = self._create_workflow(include_meta_analysis=True)
        step = self._get_step(wf, "meta_analysis")
        assert step.step_type == "task"

    def test_thesis_uses_thesis_agent(self):
        """Test thesis step uses the thesis agent."""
        wf = self._create_workflow(thesis_agent="gemini")
        step = self._get_step(wf, "thesis")
        assert step.config["agent_type"] == "gemini"

    def test_antithesis_uses_antithesis_agent(self):
        """Test antithesis step uses the antithesis agent."""
        wf = self._create_workflow(antithesis_agent="mistral")
        step = self._get_step(wf, "antithesis")
        assert step.config["agent_type"] == "mistral"

    def test_synthesis_uses_synthesis_agent(self):
        """Test synthesis step uses the synthesis agent."""
        wf = self._create_workflow(synthesis_agent="grok")
        step = self._get_step(wf, "synthesis")
        assert step.config["agent_type"] == "grok"

    def test_thesis_has_system_prompt(self):
        """Test thesis step has a system prompt configured."""
        wf = self._create_workflow()
        step = self._get_step(wf, "thesis")
        assert "system_prompt" in step.config
        assert len(step.config["system_prompt"]) > 0

    def test_thesis_system_prompt_supportive(self):
        """Test thesis system prompt for supportive stance."""
        wf = self._create_workflow(thesis_stance="supportive")
        step = self._get_step(wf, "thesis")
        assert "advocate" in step.config["system_prompt"].lower()

    def test_thesis_system_prompt_critical(self):
        """Test thesis system prompt for critical stance."""
        wf = self._create_workflow(thesis_stance="critical")
        step = self._get_step(wf, "thesis")
        assert "critic" in step.config["system_prompt"].lower()

    def test_thesis_system_prompt_neutral(self):
        """Test thesis system prompt for neutral stance."""
        wf = self._create_workflow(thesis_stance="neutral")
        step = self._get_step(wf, "thesis")
        assert "analyst" in step.config["system_prompt"].lower()

    def test_thesis_system_prompt_unknown_stance_falls_back(self):
        """Test unknown thesis stance falls back to supportive."""
        wf = self._create_workflow(thesis_stance="unknown_stance")
        step = self._get_step(wf, "thesis")
        assert "advocate" in step.config["system_prompt"].lower()

    def test_antithesis_has_system_prompt(self):
        """Test antithesis step has a critical-thinking system prompt."""
        wf = self._create_workflow()
        step = self._get_step(wf, "antithesis")
        assert "system_prompt" in step.config
        assert "critical" in step.config["system_prompt"].lower()

    def test_synthesis_has_system_prompt(self):
        """Test synthesis step has a synthesizer system prompt."""
        wf = self._create_workflow()
        step = self._get_step(wf, "synthesis")
        assert "system_prompt" in step.config
        assert (
            "synthesize" in step.config["system_prompt"].lower()
            or "integrat" in step.config["system_prompt"].lower()
        )

    def test_meta_analysis_config(self):
        """Test meta-analysis step has transform and output_format config."""
        wf = self._create_workflow(include_meta_analysis=True)
        step = self._get_step(wf, "meta_analysis")
        assert "transform" in step.config
        assert step.config["output_format"] == "json"

    def test_meta_analysis_transform_references_agents(self):
        """Test meta-analysis transform includes agent names."""
        wf = self._create_workflow(
            thesis_agent="claude",
            antithesis_agent="gpt4",
            synthesis_agent="gemini",
            include_meta_analysis=True,
        )
        step = self._get_step(wf, "meta_analysis")
        transform = step.config["transform"]
        assert "claude" in transform
        assert "gpt4" in transform
        assert "gemini" in transform

    def test_step_timeout(self):
        """Test steps use the configured timeout."""
        wf = self._create_workflow(timeout_per_step=45.0)
        for step_id in ["thesis", "antithesis", "synthesis"]:
            step = self._get_step(wf, step_id)
            assert step.timeout_seconds == 45.0


# ============================================================================
# Next Steps / Transition Tests
# ============================================================================


class TestDialecticNextSteps:
    """Tests for next_steps configuration on steps."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        defaults = {"name": "Test Dialectic"}
        defaults.update(kwargs)
        pattern = DialecticPattern(**defaults)
        return pattern.create_workflow()

    def _get_step(self, wf, step_id):
        """Helper to get a step by ID."""
        return next(s for s in wf.steps if s.id == step_id)

    def test_thesis_next_steps(self):
        """Test thesis step transitions to antithesis."""
        wf = self._create_workflow()
        step = self._get_step(wf, "thesis")
        assert step.next_steps == ["antithesis"]

    def test_antithesis_next_steps(self):
        """Test antithesis step transitions to synthesis."""
        wf = self._create_workflow()
        step = self._get_step(wf, "antithesis")
        assert step.next_steps == ["synthesis"]

    def test_synthesis_next_steps_with_meta(self):
        """Test synthesis transitions to meta_analysis when enabled."""
        wf = self._create_workflow(include_meta_analysis=True)
        step = self._get_step(wf, "synthesis")
        assert step.next_steps == ["meta_analysis"]

    def test_synthesis_no_next_steps_without_meta(self):
        """Test synthesis has no next steps when meta-analysis is disabled."""
        wf = self._create_workflow(include_meta_analysis=False)
        step = self._get_step(wf, "synthesis")
        assert step.next_steps == []


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestDialecticVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        defaults = {"name": "Test Dialectic"}
        defaults.update(kwargs)
        pattern = DialecticPattern(**defaults)
        return pattern.create_workflow()

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        wf = self._create_workflow()
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_thesis_step_category(self):
        """Test thesis step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "thesis")
        assert step.visual.category == NodeCategory.AGENT

    def test_antithesis_step_category(self):
        """Test antithesis step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "antithesis")
        assert step.visual.category == NodeCategory.AGENT

    def test_synthesis_step_category(self):
        """Test synthesis step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.visual.category == NodeCategory.AGENT

    def test_meta_analysis_step_category(self):
        """Test meta-analysis step has TASK category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow(include_meta_analysis=True)
        step = next(s for s in wf.steps if s.id == "meta_analysis")
        assert step.visual.category == NodeCategory.TASK

    def test_steps_have_distinct_x_positions(self):
        """Test that steps are laid out horizontally with distinct x positions."""
        wf = self._create_workflow(include_meta_analysis=True)
        x_positions = [s.visual.position.x for s in wf.steps]
        # All x positions should be unique (steps laid out left to right)
        assert len(set(x_positions)) == len(x_positions)

    def test_steps_share_y_position(self):
        """Test that core steps share the same y position for horizontal layout."""
        wf = self._create_workflow(include_meta_analysis=True)
        y_positions = [s.visual.position.y for s in wf.steps]
        assert len(set(y_positions)) == 1

    def test_transitions_have_visual(self):
        """Test that transitions have visual edge data."""
        from aragora.workflow.types import EdgeType

        wf = self._create_workflow()
        for transition in wf.transitions:
            assert transition.visual is not None
            assert transition.visual.edge_type == EdgeType.DATA_FLOW


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestDialecticPromptBuilding:
    """Tests for prompt building methods."""

    def _create_pattern(self, **kwargs):
        """Helper to create a DialecticPattern instance."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        defaults = {"name": "Test Dialectic"}
        defaults.update(kwargs)
        return DialecticPattern(**defaults)

    def test_build_thesis_prompt_default(self):
        """Test default thesis prompt is generated when none provided."""
        pattern = self._create_pattern()
        prompt = pattern._build_thesis_prompt()
        assert "{task}" in prompt
        assert "thesis" in prompt.lower()

    def test_build_thesis_prompt_supportive_stance(self):
        """Test thesis prompt includes supportive instruction."""
        pattern = self._create_pattern(thesis_stance="supportive")
        prompt = pattern._build_thesis_prompt()
        assert "favor" in prompt.lower() or "in favor" in prompt.lower()

    def test_build_thesis_prompt_critical_stance(self):
        """Test thesis prompt includes critical instruction."""
        pattern = self._create_pattern(thesis_stance="critical")
        prompt = pattern._build_thesis_prompt()
        assert "concern" in prompt.lower() or "problem" in prompt.lower()

    def test_build_thesis_prompt_neutral_stance(self):
        """Test thesis prompt includes neutral instruction."""
        pattern = self._create_pattern(thesis_stance="neutral")
        prompt = pattern._build_thesis_prompt()
        assert "balanced" in prompt.lower()

    def test_build_antithesis_prompt_default(self):
        """Test default antithesis prompt is generated."""
        pattern = self._create_pattern()
        prompt = pattern._build_antithesis_prompt()
        assert "{task}" in prompt
        assert "antithesis" in prompt.lower() or "counter" in prompt.lower()

    def test_build_antithesis_prompt_references_thesis(self):
        """Test antithesis prompt references the thesis output."""
        pattern = self._create_pattern()
        prompt = pattern._build_antithesis_prompt()
        assert "{step.thesis}" in prompt

    def test_build_synthesis_prompt_default(self):
        """Test default synthesis prompt is generated."""
        pattern = self._create_pattern()
        prompt = pattern._build_synthesis_prompt()
        assert "{task}" in prompt
        assert "synthe" in prompt.lower()

    def test_build_synthesis_prompt_references_both(self):
        """Test synthesis prompt references both thesis and antithesis outputs."""
        pattern = self._create_pattern()
        prompt = pattern._build_synthesis_prompt()
        assert "{step.thesis}" in prompt
        assert "{step.antithesis}" in prompt

    def test_custom_thesis_prompt_used_in_workflow(self):
        """Test that custom thesis prompt is used in workflow generation."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        custom = "Custom thesis prompt for {task}"
        pattern = DialecticPattern(name="Test", thesis_prompt=custom)
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "thesis")
        assert step.config["prompt_template"] == custom

    def test_custom_antithesis_prompt_used_in_workflow(self):
        """Test that custom antithesis prompt is used in workflow generation."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        custom = "Custom antithesis prompt for {task}"
        pattern = DialecticPattern(name="Test", antithesis_prompt=custom)
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "antithesis")
        assert step.config["prompt_template"] == custom

    def test_custom_synthesis_prompt_used_in_workflow(self):
        """Test that custom synthesis prompt is used in workflow generation."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        custom = "Custom synthesis prompt for {task}"
        pattern = DialecticPattern(name="Test", synthesis_prompt=custom)
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.config["prompt_template"] == custom

    def test_build_meta_transform(self):
        """Test meta-analysis transform expression structure."""
        pattern = self._create_pattern(
            thesis_agent="claude",
            antithesis_agent="gpt4",
            synthesis_agent="gemini",
        )
        transform = pattern._build_meta_transform()
        assert "dialectic_complete" in transform
        assert "thesis_summary" in transform
        assert "antithesis_summary" in transform
        assert "synthesis_summary" in transform
        assert "claude" in transform
        assert "gpt4" in transform
        assert "gemini" in transform


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestDialecticFactory:
    """Tests for factory methods."""

    def test_create_classmethod(self):
        """Test DialecticPattern.create factory method."""
        from aragora.workflow.patterns.dialectic import DialecticPattern
        from aragora.workflow.types import WorkflowDefinition

        wf = DialecticPattern.create(
            name="Factory Test",
            thesis_agent="claude",
            antithesis_agent="gpt4",
            synthesis_agent="gemini",
            thesis_stance="critical",
        )
        assert isinstance(wf, WorkflowDefinition)
        assert wf.metadata["pattern"] == "dialectic"
        assert wf.metadata["thesis_agent"] == "claude"
        assert wf.metadata["antithesis_agent"] == "gpt4"
        assert wf.metadata["synthesis_agent"] == "gemini"
        assert wf.metadata["thesis_stance"] == "critical"

    def test_create_returns_valid_workflow(self):
        """Test that factory method returns a fully valid workflow."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        wf = DialecticPattern.create(
            name="Validity Check",
            task="Evaluate climate policy",
        )
        assert wf.entry_step == "thesis"
        assert len(wf.steps) >= 3
        assert len(wf.transitions) >= 2
        assert "dialectic" in wf.tags

    def test_create_with_meta_disabled(self):
        """Test factory method with meta-analysis disabled."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        wf = DialecticPattern.create(
            name="No Meta",
            include_meta_analysis=False,
        )
        step_ids = [s.id for s in wf.steps]
        assert "meta_analysis" not in step_ids
        assert len(wf.steps) == 3

    def test_create_with_custom_category(self):
        """Test factory method with custom category."""
        from aragora.workflow.patterns.dialectic import DialecticPattern
        from aragora.workflow.types import WorkflowCategory

        wf = DialecticPattern.create(
            name="Legal Dialectic",
            category=WorkflowCategory.LEGAL,
        )
        assert wf.category == WorkflowCategory.LEGAL


# ============================================================================
# Step Name Tests
# ============================================================================


class TestDialecticStepNames:
    """Tests for step names and display labels."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.dialectic import DialecticPattern

        defaults = {"name": "Test Dialectic"}
        defaults.update(kwargs)
        pattern = DialecticPattern(**defaults)
        return pattern.create_workflow()

    def test_thesis_step_name(self):
        """Test thesis step has correct display name."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "thesis")
        assert step.name == "Thesis"

    def test_antithesis_step_name(self):
        """Test antithesis step has correct display name."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "antithesis")
        assert step.name == "Antithesis"

    def test_synthesis_step_name(self):
        """Test synthesis step has correct display name."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.name == "Synthesis"

    def test_meta_analysis_step_name(self):
        """Test meta-analysis step has correct display name."""
        wf = self._create_workflow(include_meta_analysis=True)
        step = next(s for s in wf.steps if s.id == "meta_analysis")
        assert step.name == "Meta-Analysis"
