"""
Tests for HiveMind Workflow Pattern.

Tests cover:
- HiveMindPattern initialization and defaults
- Custom agent and consensus configuration
- Workflow generation with all steps (agent, merge, synthesis)
- Step configurations, types, and handlers
- Transition rules between parallel agents, merge, and synthesis
- Visual metadata for workflow builder
- next_steps configuration on each step
- Prompt building for synthesis step
- Workflow metadata and tags
- Factory classmethod
- Various agent counts and layouts
"""

import pytest
from unittest.mock import MagicMock


# ============================================================================
# HiveMindPattern Initialization Tests
# ============================================================================


class TestHiveMindPatternInit:
    """Tests for HiveMindPattern initialization."""

    def test_default_init(self):
        """Test default initialization with minimal parameters."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test Hive Mind")
        assert pattern.name == "Test Hive Mind"
        assert pattern.agents == ["claude", "gpt4"]
        assert pattern.task == ""
        assert pattern.consensus_mode == "synthesis"
        assert pattern.consensus_threshold == 0.7
        assert pattern.include_dissent is True
        assert pattern.timeout_per_agent == 120.0

    def test_custom_agents(self):
        """Test initialization with custom agents list."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(
            name="Custom",
            agents=["gemini", "mistral", "grok"],
        )
        assert pattern.agents == ["gemini", "mistral", "grok"]

    def test_custom_task(self):
        """Test initialization with a custom task."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(
            name="Task Test",
            task="Analyze this contract for risks",
        )
        assert pattern.task == "Analyze this contract for risks"

    def test_custom_consensus_mode(self):
        """Test initialization with custom consensus mode."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        for mode in ["weighted", "majority", "synthesis"]:
            pattern = HiveMindPattern(name="Test", consensus_mode=mode)
            assert pattern.consensus_mode == mode

    def test_custom_consensus_threshold(self):
        """Test initialization with custom consensus threshold."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test", consensus_threshold=0.9)
        assert pattern.consensus_threshold == 0.9

    def test_include_dissent_false(self):
        """Test initialization with include_dissent disabled."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test", include_dissent=False)
        assert pattern.include_dissent is False

    def test_custom_timeout_per_agent(self):
        """Test initialization with custom timeout per agent."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test", timeout_per_agent=60.0)
        assert pattern.timeout_per_agent == 60.0

    def test_pattern_type(self):
        """Test that pattern_type is HIVE_MIND."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern
        from aragora.workflow.patterns.base import PatternType

        assert HiveMindPattern.pattern_type == PatternType.HIVE_MIND

    def test_kwargs_passed_to_config(self):
        """Test that extra kwargs are stored in config."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(
            name="Test",
            tags=["custom_tag"],
        )
        assert pattern.config["tags"] == ["custom_tag"]

    def test_single_agent(self):
        """Test initialization with a single agent."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Single", agents=["claude"])
        assert pattern.agents == ["claude"]

    def test_many_agents(self):
        """Test initialization with many agents."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        agents = ["claude", "gpt4", "gemini", "mistral", "grok", "deepseek"]
        pattern = HiveMindPattern(name="Many", agents=agents)
        assert pattern.agents == agents
        assert len(pattern.agents) == 6


# ============================================================================
# Workflow Generation Tests
# ============================================================================


class TestHiveMindWorkflowGeneration:
    """Tests for HiveMind workflow generation."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        defaults = {"name": "Test Hive Mind"}
        defaults.update(kwargs)
        pattern = HiveMindPattern(**defaults)
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with hm_."""
        wf = self._create_workflow()
        assert wf.id.startswith("hm_")

    def test_workflow_name(self):
        """Test workflow name is set correctly."""
        wf = self._create_workflow(name="Risk Analysis")
        assert wf.name == "Risk Analysis"

    def test_step_count_default_agents(self):
        """Test workflow has correct step count with default 2 agents."""
        wf = self._create_workflow()
        # 2 agents + 1 merge + 1 synthesis = 4 steps
        assert len(wf.steps) == 4

    def test_step_count_three_agents(self):
        """Test workflow has correct step count with 3 agents."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        # 3 agents + 1 merge + 1 synthesis = 5 steps
        assert len(wf.steps) == 5

    def test_step_count_single_agent(self):
        """Test workflow has correct step count with 1 agent."""
        wf = self._create_workflow(agents=["claude"])
        # 1 agent + 1 merge + 1 synthesis = 3 steps
        assert len(wf.steps) == 3

    def test_step_count_five_agents(self):
        """Test workflow has correct step count with 5 agents."""
        agents = ["claude", "gpt4", "gemini", "mistral", "grok"]
        wf = self._create_workflow(agents=agents)
        # 5 agents + 1 merge + 1 synthesis = 7 steps
        assert len(wf.steps) == 7

    def test_agent_steps_created(self):
        """Test that all agent steps are created with correct IDs."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        step_ids = [s.id for s in wf.steps]
        assert "agent_claude_0" in step_ids
        assert "agent_gpt4_1" in step_ids
        assert "agent_gemini_2" in step_ids

    def test_merge_step_exists(self):
        """Test that consensus_merge step exists."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "consensus_merge" in step_ids

    def test_synthesis_step_exists(self):
        """Test that synthesis step exists."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "synthesis" in step_ids

    def test_entry_step_is_first_agent(self):
        """Test that entry step is the first agent step."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert wf.entry_step == "agent_claude_0"

    def test_entry_step_with_different_agents(self):
        """Test entry step uses first agent from custom list."""
        wf = self._create_workflow(agents=["gemini", "mistral"])
        assert wf.entry_step == "agent_gemini_0"

    def test_workflow_description(self):
        """Test workflow description includes agent count."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert "3" in wf.description
        assert "Hive Mind" in wf.description


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestHiveMindStepConfig:
    """Tests for step configuration details."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        defaults = {"name": "Test Hive Mind"}
        defaults.update(kwargs)
        pattern = HiveMindPattern(**defaults)
        return pattern.create_workflow()

    def test_agent_step_type(self):
        """Test agent steps have 'agent' step_type."""
        wf = self._create_workflow(agents=["claude", "gpt4"])
        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        for step in agent_steps:
            assert step.step_type == "agent"

    def test_agent_step_config_has_agent_type(self):
        """Test agent step config includes agent_type."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        step = next(s for s in wf.steps if s.id == "agent_gemini_2")
        assert step.config["agent_type"] == "gemini"

    def test_agent_step_config_has_prompt(self):
        """Test agent step config includes prompt from task."""
        wf = self._create_workflow(
            agents=["claude"],
            task="Analyze this document",
        )
        step = next(s for s in wf.steps if s.id == "agent_claude_0")
        assert step.config["prompt_template"] == "Analyze this document"

    def test_merge_step_type(self):
        """Test merge step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        assert step.step_type == "task"

    def test_merge_step_task_type(self):
        """Test merge step has aggregate task type."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        assert step.config["task_type"] == "aggregate"

    def test_merge_step_mode(self):
        """Test merge step has merge mode."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        assert step.config["mode"] == "merge"

    def test_merge_step_inputs_list(self):
        """Test merge step inputs list matches agent step IDs."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        expected_inputs = ["agent_claude_0", "agent_gpt4_1", "agent_gemini_2"]
        assert step.config["inputs"] == expected_inputs

    def test_synthesis_step_type(self):
        """Test synthesis step is an agent step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.step_type == "agent"

    def test_synthesis_step_uses_first_agent(self):
        """Test synthesis step uses the first agent from the list."""
        wf = self._create_workflow(agents=["gemini", "claude", "gpt4"])
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.config["agent_type"] == "gemini"

    def test_synthesis_step_has_system_prompt(self):
        """Test synthesis step has a system_prompt about synthesizing."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert "system_prompt" in step.config
        assert "synthesiz" in step.config["system_prompt"].lower()

    def test_agent_step_timeout(self):
        """Test agent steps use the configured timeout."""
        wf = self._create_workflow(
            agents=["claude"],
            timeout_per_agent=90.0,
        )
        step = next(s for s in wf.steps if s.id == "agent_claude_0")
        assert step.timeout_seconds == 90.0


# ============================================================================
# Transition Tests
# ============================================================================


class TestHiveMindTransitions:
    """Tests for transition rules between steps."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        defaults = {"name": "Test Hive Mind"}
        defaults.update(kwargs)
        pattern = HiveMindPattern(**defaults)
        return pattern.create_workflow()

    def test_transition_count_default(self):
        """Test transition count with default 2 agents: 2 agent->merge + 1 merge->synthesis = 3."""
        wf = self._create_workflow()
        assert len(wf.transitions) == 3

    def test_transition_count_three_agents(self):
        """Test transition count with 3 agents: 3 + 1 = 4."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert len(wf.transitions) == 4

    def test_transition_count_single_agent(self):
        """Test transition count with 1 agent: 1 + 1 = 2."""
        wf = self._create_workflow(agents=["claude"])
        assert len(wf.transitions) == 2

    def test_transition_count_five_agents(self):
        """Test transition count with 5 agents: 5 + 1 = 6."""
        agents = ["claude", "gpt4", "gemini", "mistral", "grok"]
        wf = self._create_workflow(agents=agents)
        assert len(wf.transitions) == 6

    def test_agent_transitions_to_merge(self):
        """Test each agent step has a transition to consensus_merge."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        agent_to_merge = [t for t in wf.transitions if t.to_step == "consensus_merge"]
        assert len(agent_to_merge) == 3
        from_steps = {t.from_step for t in agent_to_merge}
        assert from_steps == {"agent_claude_0", "agent_gpt4_1", "agent_gemini_2"}

    def test_merge_to_synthesis_transition(self):
        """Test merge step transitions to synthesis."""
        wf = self._create_workflow()
        merge_transitions = [t for t in wf.transitions if t.from_step == "consensus_merge"]
        assert len(merge_transitions) == 1
        assert merge_transitions[0].to_step == "synthesis"

    def test_no_transition_from_synthesis(self):
        """Test synthesis step has no outgoing transitions (terminal)."""
        wf = self._create_workflow()
        synthesis_transitions = [t for t in wf.transitions if t.from_step == "synthesis"]
        assert len(synthesis_transitions) == 0

    def test_transitions_have_ids(self):
        """Test all transitions have unique IDs."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        ids = [t.id for t in wf.transitions]
        assert len(ids) == len(set(ids))
        for tid in ids:
            assert tid.startswith("tr_")


# ============================================================================
# next_steps Configuration Tests
# ============================================================================


class TestHiveMindNextSteps:
    """Tests for next_steps configuration on steps."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        defaults = {"name": "Test Hive Mind"}
        defaults.update(kwargs)
        pattern = HiveMindPattern(**defaults)
        return pattern.create_workflow()

    def test_agent_next_steps_point_to_merge(self):
        """Test each agent step's next_steps is ['consensus_merge']."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        for step in wf.steps:
            if step.id.startswith("agent_"):
                assert step.next_steps == ["consensus_merge"], (
                    f"Step {step.id} next_steps should be ['consensus_merge'], "
                    f"got {step.next_steps}"
                )

    def test_merge_next_steps_point_to_synthesis(self):
        """Test merge step's next_steps is ['synthesis']."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        assert step.next_steps == ["synthesis"]

    def test_synthesis_has_no_next_steps(self):
        """Test synthesis step has empty next_steps (terminal)."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.next_steps == []


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestHiveMindVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        defaults = {"name": "Test Hive Mind"}
        defaults.update(kwargs)
        pattern = HiveMindPattern(**defaults)
        return pattern.create_workflow()

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_agent_steps_category(self):
        """Test agent steps have AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow(agents=["claude", "gpt4"])
        for step in wf.steps:
            if step.id.startswith("agent_"):
                assert step.visual.category == NodeCategory.AGENT

    def test_merge_step_category(self):
        """Test merge step has TASK category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        assert step.visual.category == NodeCategory.TASK

    def test_synthesis_step_category(self):
        """Test synthesis step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.visual.category == NodeCategory.AGENT

    def test_agent_positions_stacked_vertically(self):
        """Test agent steps are positioned with increasing Y coordinates."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        y_positions = [s.visual.position.y for s in agent_steps]
        # Agent steps should be stacked vertically (increasing Y)
        assert y_positions == sorted(y_positions)
        # Adjacent agents should be spaced apart
        for i in range(len(y_positions) - 1):
            assert y_positions[i + 1] > y_positions[i]

    def test_agent_positions_same_x(self):
        """Test all agent steps share the same X position."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        x_positions = {s.visual.position.x for s in agent_steps}
        assert len(x_positions) == 1

    def test_merge_position_centered_vertically(self):
        """Test merge step is centered vertically relative to agents."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        merge_step = next(s for s in wf.steps if s.id == "consensus_merge")

        y_positions = [s.visual.position.y for s in agent_steps]
        min_y = min(y_positions)
        max_y = max(y_positions)
        expected_center = (min_y + max_y) / 2
        assert merge_step.visual.position.y == expected_center

    def test_merge_position_right_of_agents(self):
        """Test merge step is positioned to the right of agent steps."""
        wf = self._create_workflow(agents=["claude", "gpt4"])
        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        merge_step = next(s for s in wf.steps if s.id == "consensus_merge")

        agent_x = agent_steps[0].visual.position.x
        assert merge_step.visual.position.x > agent_x

    def test_synthesis_position_right_of_merge(self):
        """Test synthesis step is positioned to the right of merge step."""
        wf = self._create_workflow()
        merge_step = next(s for s in wf.steps if s.id == "consensus_merge")
        synthesis_step = next(s for s in wf.steps if s.id == "synthesis")

        assert synthesis_step.visual.position.x > merge_step.visual.position.x

    def test_transitions_have_visual(self):
        """Test that transitions have visual edge data."""
        from aragora.workflow.types import EdgeType

        wf = self._create_workflow()
        for transition in wf.transitions:
            assert transition.visual is not None
            assert transition.visual.edge_type == EdgeType.DATA_FLOW


# ============================================================================
# Tags and Metadata Tests
# ============================================================================


class TestHiveMindTagsAndMetadata:
    """Tests for workflow tags and metadata."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        defaults = {"name": "Test Hive Mind"}
        defaults.update(kwargs)
        pattern = HiveMindPattern(**defaults)
        return pattern.create_workflow()

    def test_default_tags(self):
        """Test default tags include hive_mind, parallel, consensus."""
        wf = self._create_workflow()
        assert "hive_mind" in wf.tags
        assert "parallel" in wf.tags
        assert "consensus" in wf.tags

    def test_custom_tags_included(self):
        """Test custom tags are appended to default tags."""
        wf = self._create_workflow(tags=["risk_analysis", "legal"])
        assert "hive_mind" in wf.tags
        assert "parallel" in wf.tags
        assert "consensus" in wf.tags
        assert "risk_analysis" in wf.tags
        assert "legal" in wf.tags

    def test_metadata_pattern(self):
        """Test metadata includes pattern name."""
        wf = self._create_workflow()
        assert wf.metadata["pattern"] == "hive_mind"

    def test_metadata_agents(self):
        """Test metadata includes agents list."""
        wf = self._create_workflow(agents=["claude", "gpt4", "gemini"])
        assert wf.metadata["agents"] == ["claude", "gpt4", "gemini"]

    def test_metadata_consensus_mode(self):
        """Test metadata includes consensus_mode."""
        wf = self._create_workflow(consensus_mode="majority")
        assert wf.metadata["consensus_mode"] == "majority"

    def test_metadata_consensus_threshold(self):
        """Test metadata includes consensus_threshold."""
        wf = self._create_workflow(consensus_threshold=0.85)
        assert wf.metadata["consensus_threshold"] == 0.85

    def test_workflow_category_default(self):
        """Test workflow category defaults to GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL

    def test_workflow_category_custom(self):
        """Test custom workflow category."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow(category=WorkflowCategory.LEGAL)
        assert wf.category == WorkflowCategory.LEGAL


# ============================================================================
# Synthesis Prompt Tests
# ============================================================================


class TestHiveMindSynthesisPrompt:
    """Tests for synthesis prompt building."""

    def test_synthesis_prompt_contains_task_placeholder(self):
        """Test synthesis prompt includes {task} placeholder."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test", task="Review contract")
        prompt = pattern._build_synthesis_prompt()
        assert "{task}" in prompt

    def test_synthesis_prompt_contains_merge_reference(self):
        """Test synthesis prompt includes {step.consensus_merge} placeholder."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test")
        prompt = pattern._build_synthesis_prompt()
        assert "{step.consensus_merge}" in prompt

    def test_synthesis_prompt_used_in_step(self):
        """Test synthesis step uses the built prompt."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        pattern = HiveMindPattern(name="Test", task="Analyze risks")
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "synthesis")
        prompt = step.config["prompt_template"]
        assert "{task}" in prompt
        assert "{step.consensus_merge}" in prompt


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestHiveMindFactory:
    """Tests for factory methods."""

    def test_create_classmethod(self):
        """Test HiveMindPattern.create factory method."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern
        from aragora.workflow.types import WorkflowDefinition

        wf = HiveMindPattern.create(
            name="Factory Test",
            agents=["claude", "gpt4", "gemini"],
            consensus_mode="majority",
            consensus_threshold=0.8,
        )
        assert isinstance(wf, WorkflowDefinition)
        assert wf.metadata["pattern"] == "hive_mind"
        assert wf.metadata["agents"] == ["claude", "gpt4", "gemini"]
        assert wf.metadata["consensus_mode"] == "majority"
        assert wf.metadata["consensus_threshold"] == 0.8

    def test_create_returns_correct_step_count(self):
        """Test factory creates workflow with correct step count."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(
            name="Factory Count",
            agents=["claude", "gpt4"],
        )
        # 2 agents + merge + synthesis = 4
        assert len(wf.steps) == 4

    def test_create_with_task(self):
        """Test factory passes task through correctly."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(
            name="Factory Task",
            task="Evaluate proposal",
        )
        # Agent steps should have the task as prompt
        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        for step in agent_steps:
            assert step.config["prompt_template"] == "Evaluate proposal"


# ============================================================================
# Integration / End-to-End Scenario Tests
# ============================================================================


class TestHiveMindIntegration:
    """Integration tests validating complete workflow structures."""

    def test_three_agent_workflow_structure(self):
        """Test complete structure with 3 agents: 5 steps, 4 transitions."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(
            name="Three Agent Analysis",
            agents=["claude", "gpt4", "gemini"],
            task="Identify contract risks",
            consensus_mode="synthesis",
            consensus_threshold=0.7,
        )

        # 3 agents + merge + synthesis = 5 steps
        assert len(wf.steps) == 5

        # 3 agent->merge + 1 merge->synthesis = 4 transitions
        assert len(wf.transitions) == 4

        # Entry is first agent
        assert wf.entry_step == "agent_claude_0"

        # Tags
        assert "hive_mind" in wf.tags
        assert "parallel" in wf.tags
        assert "consensus" in wf.tags

    def test_workflow_steps_are_ordered(self):
        """Test steps are in order: agents, merge, synthesis."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(
            name="Order Test",
            agents=["claude", "gpt4"],
        )

        step_ids = [s.id for s in wf.steps]
        # Agents first, then merge, then synthesis
        assert step_ids.index("agent_claude_0") < step_ids.index("consensus_merge")
        assert step_ids.index("agent_gpt4_1") < step_ids.index("consensus_merge")
        assert step_ids.index("consensus_merge") < step_ids.index("synthesis")

    def test_all_agent_names_are_titled(self):
        """Test each agent step has a title-cased name."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(
            name="Name Test",
            agents=["claude", "gpt4", "gemini"],
        )

        agent_steps = [s for s in wf.steps if s.id.startswith("agent_")]
        for step in agent_steps:
            assert "Analysis" in step.name

    def test_merge_step_name(self):
        """Test merge step has correct name."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(name="Name Test")
        step = next(s for s in wf.steps if s.id == "consensus_merge")
        assert step.name == "Consensus Merge"

    def test_synthesis_step_name(self):
        """Test synthesis step has correct name."""
        from aragora.workflow.patterns.hive_mind import HiveMindPattern

        wf = HiveMindPattern.create(name="Name Test")
        step = next(s for s in wf.steps if s.id == "synthesis")
        assert step.name == "Synthesis"
