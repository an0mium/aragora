"""
Tests for Hierarchical Workflow Pattern.

Tests cover:
- HierarchicalPattern initialization and defaults
- Manager and worker agent configuration
- Workflow generation with all steps
- Step configurations and handlers
- Transition rules between steps
- Visual metadata for workflow builder
- Task decomposition prompt building
- Review prompt building
- Workflow metadata and tags
- Factory classmethod
- Handler registration for hierarchical_parse_subtasks and hierarchical_dispatch
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# HierarchicalPattern Initialization Tests
# ============================================================================


class TestHierarchicalPatternInit:
    """Tests for HierarchicalPattern initialization."""

    def test_default_init(self):
        """Test default initialization with minimal parameters."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(name="Test Hierarchical")
        assert pattern.name == "Test Hierarchical"
        # Base class sets self.agents to default ["claude", "gpt4"]
        assert pattern.agents == ["claude", "gpt4"]
        assert pattern.task == ""
        assert pattern.manager_agent == "claude"
        # worker_agents uses local `agents` param which is None, so falls through to default
        assert pattern.worker_agents == ["gpt4", "gemini"]
        assert pattern.max_subtasks == 4
        assert pattern.delegation_prompt == ""
        assert pattern.review_prompt == ""
        assert pattern.timeout_per_worker == 120.0

    def test_init_with_explicit_agents(self):
        """Test initialization with explicit agents list."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        # When agents is explicitly passed, worker_agents uses that list
        pattern = HierarchicalPattern(name="Test", agents=["claude", "gpt4"])
        assert pattern.agents == ["claude", "gpt4"]
        assert pattern.manager_agent == "claude"
        assert pattern.worker_agents == ["claude", "gpt4"]

    def test_custom_agents(self):
        """Test initialization with custom agents list."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(
            name="Custom",
            agents=["gpt4", "gemini", "mistral"],
        )
        assert pattern.agents == ["gpt4", "gemini", "mistral"]
        # Manager defaults to first agent in list
        assert pattern.manager_agent == "gpt4"
        # Workers default to agents list
        assert pattern.worker_agents == ["gpt4", "gemini", "mistral"]

    def test_explicit_manager_agent(self):
        """Test explicit manager agent configuration."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(
            name="Explicit Manager",
            agents=["gpt4", "gemini"],
            manager_agent="claude",
        )
        assert pattern.manager_agent == "claude"
        assert pattern.worker_agents == ["gpt4", "gemini"]

    def test_explicit_worker_agents(self):
        """Test explicit worker agents configuration."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(
            name="Explicit Workers",
            agents=["claude"],
            worker_agents=["gpt4", "gemini", "mistral"],
        )
        assert pattern.manager_agent == "claude"
        assert pattern.worker_agents == ["gpt4", "gemini", "mistral"]

    def test_max_subtasks_configuration(self):
        """Test max_subtasks parameter."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(
            name="Test",
            max_subtasks=8,
        )
        assert pattern.max_subtasks == 8

    def test_custom_prompts(self):
        """Test custom delegation and review prompts."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(
            name="Custom Prompts",
            delegation_prompt="Custom delegation: {task}",
            review_prompt="Custom review: {results}",
        )
        assert pattern.delegation_prompt == "Custom delegation: {task}"
        assert pattern.review_prompt == "Custom review: {results}"

    def test_custom_timeout(self):
        """Test custom timeout per worker."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(
            name="Test",
            timeout_per_worker=60.0,
        )
        assert pattern.timeout_per_worker == 60.0

    def test_pattern_type(self):
        """Test that pattern_type is HIERARCHICAL."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern
        from aragora.workflow.patterns.base import PatternType

        assert HierarchicalPattern.pattern_type == PatternType.HIERARCHICAL


# ============================================================================
# Workflow Generation Tests
# ============================================================================


class TestHierarchicalWorkflowGeneration:
    """Tests for hierarchical workflow generation."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        defaults = {"name": "Test Hierarchical"}
        defaults.update(kwargs)
        pattern = HierarchicalPattern(**defaults)
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with hier_."""
        wf = self._create_workflow()
        assert wf.id.startswith("hier_")

    def test_workflow_name(self):
        """Test workflow name is set correctly."""
        wf = self._create_workflow(name="Research Project")
        assert wf.name == "Research Project"

    def test_workflow_has_all_steps(self):
        """Test that all required steps are present."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "decompose" in step_ids
        assert "parse_subtasks" in step_ids
        assert "dispatch_workers" in step_ids
        assert "review" in step_ids

    def test_step_count(self):
        """Test workflow has exactly 4 steps."""
        wf = self._create_workflow()
        assert len(wf.steps) == 4

    def test_entry_step_is_decompose(self):
        """Test that entry step is decompose."""
        wf = self._create_workflow()
        assert wf.entry_step == "decompose"

    def test_transition_count(self):
        """Test workflow has correct number of transitions."""
        wf = self._create_workflow()
        # decompose -> parse_subtasks -> dispatch_workers -> review = 3 transitions
        assert len(wf.transitions) == 3

    def test_transition_flow(self):
        """Test the transition flow order."""
        wf = self._create_workflow()
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["decompose"] == "parse_subtasks"
        assert flow["parse_subtasks"] == "dispatch_workers"
        assert flow["dispatch_workers"] == "review"

    def test_workflow_description(self):
        """Test workflow description includes manager and worker info."""
        wf = self._create_workflow(
            manager_agent="claude",
            worker_agents=["gpt4", "gemini"],
        )
        assert "claude" in wf.description
        assert "2 workers" in wf.description

    def test_workflow_tags(self):
        """Test workflow tags."""
        wf = self._create_workflow()
        assert "hierarchical" in wf.tags
        assert "delegation" in wf.tags
        assert "manager-worker" in wf.tags

    def test_workflow_metadata(self):
        """Test workflow metadata includes pattern information."""
        wf = self._create_workflow(
            manager_agent="claude",
            worker_agents=["gpt4", "gemini"],
            max_subtasks=6,
        )
        assert wf.metadata["pattern"] == "hierarchical"
        assert wf.metadata["manager_agent"] == "claude"
        assert wf.metadata["worker_agents"] == ["gpt4", "gemini"]
        assert wf.metadata["max_subtasks"] == 6

    def test_workflow_category(self):
        """Test workflow category defaults to GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL

    def test_custom_workflow_category(self):
        """Test custom workflow category."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow(category=WorkflowCategory.CODE)
        assert wf.category == WorkflowCategory.CODE

    def test_custom_tags(self):
        """Test custom tags are included."""
        wf = self._create_workflow(tags=["research", "multi-agent"])
        assert "hierarchical" in wf.tags
        assert "research" in wf.tags
        assert "multi-agent" in wf.tags


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestHierarchicalStepConfig:
    """Tests for step configuration details."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        defaults = {"name": "Test Hierarchical"}
        defaults.update(kwargs)
        pattern = HierarchicalPattern(**defaults)
        return pattern.create_workflow()

    def test_decompose_step_type(self):
        """Test decompose step is an agent step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "decompose")
        assert step.step_type == "agent"

    def test_decompose_step_uses_manager_agent(self):
        """Test decompose step uses manager agent."""
        wf = self._create_workflow(manager_agent="gpt4")
        step = next(s for s in wf.steps if s.id == "decompose")
        assert step.config["agent_type"] == "gpt4"

    def test_decompose_step_has_system_prompt(self):
        """Test decompose step has system prompt for project manager."""
        wf = self._create_workflow(max_subtasks=5)
        step = next(s for s in wf.steps if s.id == "decompose")
        assert "system_prompt" in step.config
        assert "project manager" in step.config["system_prompt"].lower()
        assert "5" in step.config["system_prompt"]

    def test_parse_subtasks_step_type(self):
        """Test parse_subtasks step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "parse_subtasks")
        assert step.step_type == "task"

    def test_parse_subtasks_handler(self):
        """Test parse_subtasks step uses correct handler."""
        wf = self._create_workflow(max_subtasks=3)
        step = next(s for s in wf.steps if s.id == "parse_subtasks")
        assert step.config["handler"] == "hierarchical_parse_subtasks"
        assert step.config["args"]["max_subtasks"] == 3

    def test_dispatch_workers_step_type(self):
        """Test dispatch_workers step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "dispatch_workers")
        assert step.step_type == "task"

    def test_dispatch_workers_handler(self):
        """Test dispatch_workers step uses correct handler."""
        wf = self._create_workflow(
            worker_agents=["gpt4", "gemini"],
            timeout_per_worker=90.0,
        )
        step = next(s for s in wf.steps if s.id == "dispatch_workers")
        assert step.config["handler"] == "hierarchical_dispatch"
        assert step.config["args"]["worker_agents"] == ["gpt4", "gemini"]
        assert step.config["args"]["timeout"] == 90.0

    def test_review_step_type(self):
        """Test review step is an agent step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert step.step_type == "agent"

    def test_review_step_uses_manager_agent(self):
        """Test review step uses manager agent."""
        wf = self._create_workflow(manager_agent="gemini")
        step = next(s for s in wf.steps if s.id == "review")
        assert step.config["agent_type"] == "gemini"

    def test_review_step_has_system_prompt(self):
        """Test review step has system prompt for reviewing work."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert "system_prompt" in step.config
        assert "review" in step.config["system_prompt"].lower()


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestHierarchicalVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def _create_workflow(self):
        """Helper to create workflow."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(name="Test Hierarchical")
        return pattern.create_workflow()

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        wf = self._create_workflow()
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_decompose_step_category(self):
        """Test decompose step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "decompose")
        assert step.visual.category == NodeCategory.AGENT

    def test_parse_subtasks_step_category(self):
        """Test parse_subtasks step has CONTROL category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "parse_subtasks")
        assert step.visual.category == NodeCategory.CONTROL

    def test_dispatch_workers_step_category(self):
        """Test dispatch_workers step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "dispatch_workers")
        assert step.visual.category == NodeCategory.AGENT

    def test_review_step_category(self):
        """Test review step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert step.visual.category == NodeCategory.AGENT

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


class TestHierarchicalPromptBuilding:
    """Tests for prompt building methods."""

    def test_build_decompose_prompt_includes_max_subtasks(self):
        """Test decompose prompt includes max_subtasks."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(name="Test", max_subtasks=5)
        prompt = pattern._build_decompose_prompt()
        assert "5" in prompt
        assert "{task}" in prompt
        assert "JSON" in prompt

    def test_build_review_prompt_format(self):
        """Test review prompt has expected format."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(name="Test")
        prompt = pattern._build_review_prompt()
        assert "{task}" in prompt
        assert "{step.dispatch_workers}" in prompt
        assert "integrate" in prompt.lower()

    def test_custom_delegation_prompt_used(self):
        """Test that custom delegation prompt is used when provided."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        custom_prompt = "My custom delegation prompt: {task}"
        pattern = HierarchicalPattern(name="Test", delegation_prompt=custom_prompt)
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "decompose")
        assert step.config["prompt_template"] == custom_prompt

    def test_custom_review_prompt_used(self):
        """Test that custom review prompt is used when provided."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        custom_prompt = "My custom review prompt: {results}"
        pattern = HierarchicalPattern(name="Test", review_prompt=custom_prompt)
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert step.config["prompt_template"] == custom_prompt


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestHierarchicalFactory:
    """Tests for factory methods."""

    def test_create_classmethod(self):
        """Test HierarchicalPattern.create factory method."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern
        from aragora.workflow.types import WorkflowDefinition

        wf = HierarchicalPattern.create(
            name="Factory Test",
            manager_agent="claude",
            worker_agents=["gpt4", "gemini"],
            max_subtasks=3,
        )
        assert isinstance(wf, WorkflowDefinition)
        assert wf.metadata["manager_agent"] == "claude"
        assert wf.metadata["worker_agents"] == ["gpt4", "gemini"]
        assert wf.metadata["max_subtasks"] == 3

    def test_create_with_task(self):
        """Test factory with task parameter."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        wf = HierarchicalPattern.create(
            name="Research Task",
            task="Research the impact of AI on healthcare",
        )
        step = next(s for s in wf.steps if s.id == "decompose")
        # The task should be embedded in the prompt template placeholder
        assert "{task}" in step.config["prompt_template"]


# ============================================================================
# Handler Tests
# ============================================================================


class TestHierarchicalParseSubtasksHandler:
    """Tests for hierarchical_parse_subtasks handler."""

    @pytest.mark.asyncio
    async def test_parse_subtasks_from_json(self):
        """Test parsing subtasks from JSON response."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_parse_subtasks")
        assert handler is not None

        context = MagicMock()
        context.step_outputs = {
            "decompose": {
                "response": """Here are the subtasks:
                [
                    {"title": "Task 1", "description": "Do first thing", "focus": "area1"},
                    {"title": "Task 2", "description": "Do second thing", "focus": "area2"}
                ]
                """
            }
        }

        result = await handler(context, max_subtasks=4)
        assert "subtasks" in result
        assert len(result["subtasks"]) == 2
        assert result["subtasks"][0]["title"] == "Task 1"
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_parse_subtasks_limits_count(self):
        """Test that subtasks are limited to max_subtasks."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_parse_subtasks")

        context = MagicMock()
        context.step_outputs = {
            "decompose": {
                "response": """[
                    {"title": "Task 1", "description": "1"},
                    {"title": "Task 2", "description": "2"},
                    {"title": "Task 3", "description": "3"},
                    {"title": "Task 4", "description": "4"},
                    {"title": "Task 5", "description": "5"}
                ]"""
            }
        }

        result = await handler(context, max_subtasks=3)
        assert len(result["subtasks"]) == 3
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_parse_subtasks_no_json_array_found(self):
        """Test when no JSON array is found in response."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_parse_subtasks")

        context = MagicMock()
        context.step_outputs = {
            "decompose": {"response": "This is not valid JSON, just plain text response"}
        }

        result = await handler(context, max_subtasks=4)
        # When no JSON array is found, subtasks remains empty
        assert result["subtasks"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_parse_subtasks_empty_response(self):
        """Test handling of empty response."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_parse_subtasks")

        context = MagicMock()
        context.step_outputs = {"decompose": {"response": ""}}

        result = await handler(context, max_subtasks=4)
        # Empty response returns no subtasks
        assert result["subtasks"] == []
        assert result["count"] == 0


class TestHierarchicalDispatchHandler:
    """Tests for hierarchical_dispatch handler."""

    @pytest.mark.asyncio
    async def test_dispatch_returns_empty_on_no_subtasks(self):
        """Test dispatch returns error when no subtasks."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_dispatch")

        context = MagicMock()
        context.step_outputs = {"parse_subtasks": {"subtasks": []}}
        context.inputs = {}

        result = await handler(context, worker_agents=["claude"])
        assert result["results"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_dispatch_round_robin_assignment(self):
        """Test that subtasks are assigned round-robin to workers."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_dispatch")

        context = MagicMock()
        context.step_outputs = {
            "parse_subtasks": {
                "subtasks": [
                    {"title": "Task 1", "description": "Desc 1"},
                    {"title": "Task 2", "description": "Desc 2"},
                    {"title": "Task 3", "description": "Desc 3"},
                ]
            }
        }
        context.inputs = {"task": "Main task"}

        # Mock create_agent - it's imported inside the handler from aragora.agents
        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.generate.return_value = "Result"
            mock_create.return_value = mock_agent

            result = await handler(
                context,
                worker_agents=["agent_a", "agent_b"],
                timeout=30.0,
            )

            # With 3 tasks and 2 agents, assignment should be:
            # Task 0 -> agent_a (0 % 2 = 0)
            # Task 1 -> agent_b (1 % 2 = 1)
            # Task 2 -> agent_a (2 % 2 = 0)
            assert mock_create.call_count == 3
            agent_calls = [call[0][0] for call in mock_create.call_args_list]
            assert agent_calls == ["agent_a", "agent_b", "agent_a"]

    @pytest.mark.asyncio
    async def test_dispatch_handles_agent_errors(self):
        """Test that dispatch handles agent errors gracefully."""
        import aragora.workflow.patterns.hierarchical  # noqa: F401 - ensure handlers registered
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_dispatch")

        context = MagicMock()
        context.step_outputs = {
            "parse_subtasks": {
                "subtasks": [
                    {"title": "Task 1", "description": "Desc 1"},
                ]
            }
        }
        context.inputs = {"task": "Main task"}

        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.generate.side_effect = RuntimeError("API Error")
            mock_create.return_value = mock_agent

            result = await handler(context, worker_agents=["claude"], timeout=30.0)

            assert len(result["failed"]) == 1
            assert result["failed"][0]["success"] is False
            assert result["failed"][0]["error"] == "Subtask processing failed"

    @pytest.mark.asyncio
    async def test_dispatch_formats_results(self):
        """Test that dispatch formats successful results."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("hierarchical_dispatch")

        context = MagicMock()
        context.step_outputs = {
            "parse_subtasks": {
                "subtasks": [
                    {"title": "Research Topic", "description": "Desc"},
                ]
            }
        }
        context.inputs = {"task": "Main task"}

        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.generate.return_value = "This is the research result."
            mock_create.return_value = mock_agent

            result = await handler(context, worker_agents=["claude"], timeout=30.0)

            assert "formatted" in result
            assert "Research Topic" in result["formatted"]
            assert "This is the research result." in result["formatted"]


# ============================================================================
# Next Steps Tests
# ============================================================================


class TestHierarchicalNextSteps:
    """Tests for next_steps configuration on steps."""

    def _create_workflow(self):
        """Helper to create workflow."""
        from aragora.workflow.patterns.hierarchical import HierarchicalPattern

        pattern = HierarchicalPattern(name="Test Hierarchical")
        return pattern.create_workflow()

    def test_decompose_next_steps(self):
        """Test decompose step next_steps."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "decompose")
        assert step.next_steps == ["parse_subtasks"]

    def test_parse_subtasks_next_steps(self):
        """Test parse_subtasks step next_steps."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "parse_subtasks")
        assert step.next_steps == ["dispatch_workers"]

    def test_dispatch_workers_next_steps(self):
        """Test dispatch_workers step next_steps."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "dispatch_workers")
        assert step.next_steps == ["review"]

    def test_review_has_no_next_steps(self):
        """Test review step has no next_steps (terminal)."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert step.next_steps == []
