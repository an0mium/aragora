"""
Tests for ReviewCycle Workflow Pattern.

Tests cover:
- ReviewCyclePattern initialization and defaults
- Draft and review agent configuration
- Review criteria and prompt customization
- Workflow generation with all steps (draft, review, check, output)
- Step configurations and handler setup
- Transition rules including conditional convergence/refinement
- Conditional transition priorities and conditions
- Visual metadata for workflow builder
- Prompt building for draft and review steps
- Workflow metadata and tags
- Factory classmethod
- review_cycle_check handler: score extraction, iteration tracking,
  convergence logic, feedback extraction
"""

import pytest
from unittest.mock import MagicMock


# ============================================================================
# ReviewCyclePattern Initialization Tests
# ============================================================================


class TestReviewCyclePatternInit:
    """Tests for ReviewCyclePattern initialization."""

    def test_default_init(self):
        """Test default initialization with minimal parameters."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test Review")
        assert pattern.name == "Test Review"
        assert pattern.agents == ["claude", "gpt4"]
        assert pattern.task == ""
        assert pattern.draft_agent == "claude"
        assert pattern.review_agent == "gpt4"
        assert pattern.max_iterations == 3
        assert pattern.convergence_threshold == 0.85
        assert pattern.review_criteria == ["quality", "completeness", "accuracy"]
        assert pattern.draft_prompt == ""
        assert pattern.review_prompt == ""
        assert pattern.timeout_per_step == 120.0

    def test_custom_agents(self):
        """Test initialization with custom agents list."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(
            name="Custom",
            agents=["gemini", "mistral", "grok"],
        )
        assert pattern.agents == ["gemini", "mistral", "grok"]
        # draft_agent defaults to first agent
        assert pattern.draft_agent == "gemini"
        # review_agent defaults to second agent
        assert pattern.review_agent == "mistral"

    def test_single_agent_list(self):
        """Test initialization with single agent list falls back for review."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Single", agents=["gemini"])
        assert pattern.draft_agent == "gemini"
        assert pattern.review_agent == "gpt4"

    def test_explicit_draft_agent(self):
        """Test explicit draft_agent overrides agents list."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(
            name="Explicit Draft",
            agents=["gpt4", "mistral"],
            draft_agent="claude",
        )
        assert pattern.draft_agent == "claude"

    def test_explicit_review_agent(self):
        """Test explicit review_agent overrides agents list."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(
            name="Explicit Review",
            agents=["claude", "gemini"],
            review_agent="gpt4",
        )
        assert pattern.review_agent == "gpt4"

    def test_custom_max_iterations(self):
        """Test custom max_iterations parameter."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", max_iterations=5)
        assert pattern.max_iterations == 5

    def test_custom_convergence_threshold(self):
        """Test custom convergence_threshold parameter."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", convergence_threshold=0.95)
        assert pattern.convergence_threshold == 0.95

    def test_custom_review_criteria(self):
        """Test custom review_criteria parameter."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        criteria = ["correctness", "efficiency", "readability"]
        pattern = ReviewCyclePattern(name="Test", review_criteria=criteria)
        assert pattern.review_criteria == criteria

    def test_custom_draft_prompt(self):
        """Test custom draft_prompt parameter."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", draft_prompt="Write: {task}")
        assert pattern.draft_prompt == "Write: {task}"

    def test_custom_review_prompt(self):
        """Test custom review_prompt parameter."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", review_prompt="Review: {step.draft}")
        assert pattern.review_prompt == "Review: {step.draft}"

    def test_custom_timeout_per_step(self):
        """Test custom timeout_per_step parameter."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", timeout_per_step=60.0)
        assert pattern.timeout_per_step == 60.0

    def test_pattern_type(self):
        """Test that pattern_type is REVIEW_CYCLE."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern
        from aragora.workflow.patterns.base import PatternType

        assert ReviewCyclePattern.pattern_type == PatternType.REVIEW_CYCLE

    def test_no_agents_defaults(self):
        """Test default agents when none provided."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test")
        assert pattern.draft_agent == "claude"
        assert pattern.review_agent == "gpt4"

    def test_task_parameter(self):
        """Test task parameter is stored."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", task="Implement rate limiter")
        assert pattern.task == "Implement rate limiter"


# ============================================================================
# Workflow Generation Tests
# ============================================================================


class TestReviewCycleWorkflowGeneration:
    """Tests for ReviewCycle workflow generation."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        defaults = {"name": "Test Review Cycle"}
        defaults.update(kwargs)
        pattern = ReviewCyclePattern(**defaults)
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with rc_."""
        wf = self._create_workflow()
        assert wf.id.startswith("rc_")

    def test_workflow_name(self):
        """Test workflow name is set correctly."""
        wf = self._create_workflow(name="Code Review")
        assert wf.name == "Code Review"

    def test_step_count(self):
        """Test workflow has exactly 4 steps."""
        wf = self._create_workflow()
        assert len(wf.steps) == 4

    def test_workflow_has_all_steps(self):
        """Test that all required steps are present."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "draft" in step_ids
        assert "review" in step_ids
        assert "check_convergence" in step_ids
        assert "output" in step_ids

    def test_entry_step_is_draft(self):
        """Test that entry step is draft."""
        wf = self._create_workflow()
        assert wf.entry_step == "draft"

    def test_transition_count(self):
        """Test workflow has 4 transitions (2 normal + 2 conditional)."""
        wf = self._create_workflow()
        assert len(wf.transitions) == 4

    def test_transition_draft_to_review(self):
        """Test transition from draft to review exists."""
        wf = self._create_workflow()
        draft_transitions = [t for t in wf.transitions if t.from_step == "draft"]
        assert len(draft_transitions) == 1
        assert draft_transitions[0].to_step == "review"

    def test_transition_review_to_check(self):
        """Test transition from review to check_convergence exists."""
        wf = self._create_workflow()
        review_transitions = [t for t in wf.transitions if t.from_step == "review"]
        assert len(review_transitions) == 1
        assert review_transitions[0].to_step == "check_convergence"

    def test_conditional_transition_converged(self):
        """Test conditional transition from check to output (converged)."""
        wf = self._create_workflow()
        converged_tr = next(
            (t for t in wf.transitions if t.id == "tr_converged"),
            None,
        )
        assert converged_tr is not None
        assert converged_tr.from_step == "check_convergence"
        assert converged_tr.to_step == "output"
        assert "converged" in converged_tr.condition
        assert "True" in converged_tr.condition

    def test_conditional_transition_refine(self):
        """Test conditional transition from check to draft (not converged)."""
        wf = self._create_workflow()
        refine_tr = next(
            (t for t in wf.transitions if t.id == "tr_refine"),
            None,
        )
        assert refine_tr is not None
        assert refine_tr.from_step == "check_convergence"
        assert refine_tr.to_step == "draft"
        assert "converged" in refine_tr.condition
        assert "False" in refine_tr.condition

    def test_converged_transition_priority(self):
        """Test converged transition has priority 10."""
        wf = self._create_workflow()
        converged_tr = next(t for t in wf.transitions if t.id == "tr_converged")
        assert converged_tr.priority == 10

    def test_refine_transition_priority(self):
        """Test refine transition has priority 5."""
        wf = self._create_workflow()
        refine_tr = next(t for t in wf.transitions if t.id == "tr_refine")
        assert refine_tr.priority == 5

    def test_workflow_description_format(self):
        """Test workflow description includes max_iterations and threshold."""
        wf = self._create_workflow(max_iterations=5, convergence_threshold=0.9)
        assert "5" in wf.description
        assert "0.9" in wf.description

    def test_workflow_tags(self):
        """Test workflow has expected tags."""
        wf = self._create_workflow()
        assert "review_cycle" in wf.tags
        assert "iterative" in wf.tags
        assert "refinement" in wf.tags

    def test_custom_tags_appended(self):
        """Test custom tags are appended to default tags."""
        wf = self._create_workflow(tags=["code", "quality"])
        assert "review_cycle" in wf.tags
        assert "code" in wf.tags
        assert "quality" in wf.tags

    def test_workflow_metadata(self):
        """Test workflow metadata includes all pattern information."""
        wf = self._create_workflow(
            draft_agent="gemini",
            review_agent="mistral",
            max_iterations=5,
            convergence_threshold=0.9,
            review_criteria=["correctness", "style"],
        )
        assert wf.metadata["pattern"] == "review_cycle"
        assert wf.metadata["draft_agent"] == "gemini"
        assert wf.metadata["review_agent"] == "mistral"
        assert wf.metadata["max_iterations"] == 5
        assert wf.metadata["convergence_threshold"] == 0.9
        assert wf.metadata["review_criteria"] == ["correctness", "style"]

    def test_workflow_category_default(self):
        """Test workflow category defaults to GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL

    def test_workflow_custom_category(self):
        """Test custom workflow category."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow(category=WorkflowCategory.CODE)
        assert wf.category == WorkflowCategory.CODE


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestReviewCycleStepConfig:
    """Tests for step configuration details."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        defaults = {"name": "Test Review Cycle"}
        defaults.update(kwargs)
        pattern = ReviewCyclePattern(**defaults)
        return pattern.create_workflow()

    def test_draft_step_type(self):
        """Test draft step is an agent step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "draft")
        assert step.step_type == "agent"

    def test_draft_step_uses_draft_agent(self):
        """Test draft step uses the configured draft agent."""
        wf = self._create_workflow(draft_agent="gemini")
        step = next(s for s in wf.steps if s.id == "draft")
        assert step.config["agent_type"] == "gemini"

    def test_draft_step_has_prompt_template(self):
        """Test draft step has a prompt template."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "draft")
        assert "prompt_template" in step.config
        assert len(step.config["prompt_template"]) > 0

    def test_draft_step_custom_prompt(self):
        """Test draft step uses custom prompt when provided."""
        wf = self._create_workflow(draft_prompt="Custom draft: {task}")
        step = next(s for s in wf.steps if s.id == "draft")
        assert step.config["prompt_template"] == "Custom draft: {task}"

    def test_draft_step_next_steps(self):
        """Test draft step next_steps includes review."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "draft")
        assert "review" in step.next_steps

    def test_review_step_type(self):
        """Test review step is an agent step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert step.step_type == "agent"

    def test_review_step_uses_review_agent(self):
        """Test review step uses the configured review agent."""
        wf = self._create_workflow(review_agent="mistral")
        step = next(s for s in wf.steps if s.id == "review")
        assert step.config["agent_type"] == "mistral"

    def test_review_step_has_system_prompt(self):
        """Test review step has system_prompt with criteria."""
        wf = self._create_workflow(review_criteria=["correctness", "efficiency"])
        step = next(s for s in wf.steps if s.id == "review")
        assert "system_prompt" in step.config
        assert "correctness" in step.config["system_prompt"]
        assert "efficiency" in step.config["system_prompt"]

    def test_review_step_system_prompt_scoring(self):
        """Test review step system prompt mentions scoring."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert "0.0 to 1.0" in step.config["system_prompt"]

    def test_review_step_has_prompt_template(self):
        """Test review step has a prompt template."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert "prompt_template" in step.config
        assert len(step.config["prompt_template"]) > 0

    def test_review_step_custom_prompt(self):
        """Test review step uses custom prompt when provided."""
        wf = self._create_workflow(review_prompt="Custom review: {step.draft}")
        step = next(s for s in wf.steps if s.id == "review")
        assert step.config["prompt_template"] == "Custom review: {step.draft}"

    def test_review_step_next_steps(self):
        """Test review step next_steps includes check_convergence."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert "check_convergence" in step.next_steps

    def test_check_convergence_step_type(self):
        """Test check_convergence step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "check_convergence")
        assert step.step_type == "task"

    def test_check_convergence_task_type(self):
        """Test check_convergence step is function type."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "check_convergence")
        assert step.config["task_type"] == "function"

    def test_check_convergence_handler(self):
        """Test check_convergence step uses review_cycle_check handler."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "check_convergence")
        assert step.config["handler"] == "review_cycle_check"

    def test_check_convergence_threshold_in_args(self):
        """Test check_convergence step has threshold in args."""
        wf = self._create_workflow(convergence_threshold=0.9)
        step = next(s for s in wf.steps if s.id == "check_convergence")
        assert step.config["args"]["threshold"] == 0.9

    def test_check_convergence_max_iterations_in_args(self):
        """Test check_convergence step has max_iterations in args."""
        wf = self._create_workflow(max_iterations=5)
        step = next(s for s in wf.steps if s.id == "check_convergence")
        assert step.config["args"]["max_iterations"] == 5

    def test_output_step_type(self):
        """Test output step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "output")
        assert step.step_type == "task"

    def test_output_step_task_type(self):
        """Test output step is transform type."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "output")
        assert step.config["task_type"] == "transform"


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestReviewCycleVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def _create_workflow(self):
        """Helper to create workflow."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test Review Cycle")
        return pattern.create_workflow()

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        wf = self._create_workflow()
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_draft_step_agent_category(self):
        """Test draft step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "draft")
        assert step.visual.category == NodeCategory.AGENT

    def test_review_step_agent_category(self):
        """Test review step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "review")
        assert step.visual.category == NodeCategory.AGENT

    def test_check_step_control_category(self):
        """Test check_convergence step has CONTROL category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "check_convergence")
        assert step.visual.category == NodeCategory.CONTROL

    def test_conditional_transitions_have_visual(self):
        """Test that conditional transitions have CONDITIONAL edge type."""
        from aragora.workflow.types import EdgeType

        wf = self._create_workflow()
        converged_tr = next(t for t in wf.transitions if t.id == "tr_converged")
        assert converged_tr.visual is not None
        assert converged_tr.visual.edge_type == EdgeType.CONDITIONAL

        refine_tr = next(t for t in wf.transitions if t.id == "tr_refine")
        assert refine_tr.visual is not None
        assert refine_tr.visual.edge_type == EdgeType.CONDITIONAL

    def test_converged_visual_label(self):
        """Test converged transition has correct visual label."""
        wf = self._create_workflow()
        converged_tr = next(t for t in wf.transitions if t.id == "tr_converged")
        assert converged_tr.visual.label == "Converged"

    def test_refine_visual_label(self):
        """Test refine transition has correct visual label."""
        wf = self._create_workflow()
        refine_tr = next(t for t in wf.transitions if t.id == "tr_refine")
        assert refine_tr.visual.label == "Needs Work"


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestReviewCyclePromptBuilding:
    """Tests for prompt building methods."""

    def test_build_draft_prompt_contains_task(self):
        """Test default draft prompt contains {task} placeholder."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test", task="Write a REST API")
        prompt = pattern._build_draft_prompt()
        assert "{task}" in prompt

    def test_build_draft_prompt_contains_feedback_section(self):
        """Test default draft prompt contains {feedback_section} placeholder."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test")
        prompt = pattern._build_draft_prompt()
        assert "{feedback_section}" in prompt

    def test_build_review_prompt_contains_criteria(self):
        """Test review prompt includes all review criteria."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        criteria = ["correctness", "efficiency", "readability"]
        pattern = ReviewCyclePattern(name="Test", review_criteria=criteria)
        prompt = pattern._build_review_prompt()
        for criterion in criteria:
            assert criterion in prompt

    def test_build_review_prompt_score_format(self):
        """Test review prompt includes SCORE format instructions."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test")
        prompt = pattern._build_review_prompt()
        assert "SCORE" in prompt

    def test_build_review_prompt_feedback_format(self):
        """Test review prompt includes FEEDBACK format instructions."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test")
        prompt = pattern._build_review_prompt()
        assert "FEEDBACK" in prompt

    def test_build_review_prompt_suggestions_format(self):
        """Test review prompt includes SUGGESTIONS format instructions."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        pattern = ReviewCyclePattern(name="Test")
        prompt = pattern._build_review_prompt()
        assert "SUGGESTIONS" in prompt


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestReviewCycleFactory:
    """Tests for factory methods."""

    def test_create_classmethod(self):
        """Test ReviewCyclePattern.create factory method."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern
        from aragora.workflow.types import WorkflowDefinition

        wf = ReviewCyclePattern.create(
            name="Factory Test",
            draft_agent="claude",
            review_agent="gpt4",
            max_iterations=4,
            convergence_threshold=0.9,
        )
        assert isinstance(wf, WorkflowDefinition)
        assert wf.metadata["draft_agent"] == "claude"
        assert wf.metadata["review_agent"] == "gpt4"
        assert wf.metadata["max_iterations"] == 4
        assert wf.metadata["convergence_threshold"] == 0.9

    def test_create_classmethod_with_criteria(self):
        """Test factory method preserves review criteria."""
        from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

        wf = ReviewCyclePattern.create(
            name="Factory Criteria",
            review_criteria=["security", "performance"],
        )
        assert wf.metadata["review_criteria"] == ["security", "performance"]


# ============================================================================
# Handler Tests - review_cycle_check
# ============================================================================


class TestReviewCycleCheckHandler:
    """Tests for review_cycle_check handler."""

    @pytest.mark.asyncio
    async def test_handler_is_registered(self):
        """Test that review_cycle_check handler is registered."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")
        assert handler is not None

    @pytest.mark.asyncio
    async def test_extracts_score_from_response(self):
        """Test extracting SCORE from response text."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.9\nFEEDBACK:\nGood work."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=3)
        assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_converges_when_score_meets_threshold(self):
        """Test convergence when score >= threshold."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.9\nFEEDBACK:\nExcellent."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["converged"] is True
        assert "0.9" in result["reason"]
        assert "threshold" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_does_not_converge_below_threshold(self):
        """Test no convergence when score < threshold and iterations remain."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.5\nFEEDBACK:\nNeeds work."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["converged"] is False
        assert "continuing" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_converges_at_max_iterations(self):
        """Test convergence when max_iterations is reached."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.5\nFEEDBACK:\nStill needs work."}}
        # Simulate we are already at iteration max_iterations - 1
        context.state = {"review_iteration": 2}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=3)
        assert result["converged"] is True
        assert result["iteration"] == 3
        assert "max iterations" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_tracks_iteration_count(self):
        """Test iteration counter increments correctly."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.7\nFEEDBACK:\nOk."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_tracks_iteration_from_existing_state(self):
        """Test iteration counter increments from existing state value."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.7\nFEEDBACK:\nOk."}}
        context.state = {"review_iteration": 1}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["iteration"] == 2

    @pytest.mark.asyncio
    async def test_handles_missing_score(self):
        """Test handling when SCORE is not in response."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "No score provided here."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["score"] == 0.0
        assert result["converged"] is False

    @pytest.mark.asyncio
    async def test_handles_empty_review_response(self):
        """Test handling when review output is empty."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_extracts_feedback(self):
        """Test extracting FEEDBACK section from response."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {
            "review": {
                "response": "SCORE: 0.7\nFEEDBACK:\nNeeds better error handling.\nSUGGESTIONS:\nAdd try/except blocks."
            }
        }
        context.state = {}
        set_calls = {}
        context.set_state = MagicMock(side_effect=lambda k, v: set_calls.__setitem__(k, v))

        await handler(context, threshold=0.85, max_iterations=5)
        assert "last_feedback" in set_calls
        assert "error handling" in set_calls["last_feedback"]

    @pytest.mark.asyncio
    async def test_extracts_suggestions(self):
        """Test extracting SUGGESTIONS section from response."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {
            "review": {"response": "SCORE: 0.7\nFEEDBACK:\nOk.\nSUGGESTIONS:\nAdd type hints."}
        }
        context.state = {}
        set_calls = {}
        context.set_state = MagicMock(side_effect=lambda k, v: set_calls.__setitem__(k, v))

        await handler(context, threshold=0.85, max_iterations=5)
        assert "last_suggestions" in set_calls
        assert "type hints" in set_calls["last_suggestions"]

    @pytest.mark.asyncio
    async def test_stores_score_in_state(self):
        """Test that score is stored in context state."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.75\nFEEDBACK:\nGood."}}
        context.state = {}
        set_calls = {}
        context.set_state = MagicMock(side_effect=lambda k, v: set_calls.__setitem__(k, v))

        await handler(context, threshold=0.85, max_iterations=5)
        assert set_calls["last_score"] == 0.75

    @pytest.mark.asyncio
    async def test_result_includes_threshold_and_max(self):
        """Test result includes threshold and max_iterations values."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.8\nFEEDBACK:\nGood."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.9, max_iterations=4)
        assert result["threshold"] == 0.9
        assert result["max_iterations"] == 4

    @pytest.mark.asyncio
    async def test_score_exactly_at_threshold(self):
        """Test convergence when score is exactly equal to threshold."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.85\nFEEDBACK:\nMeets threshold."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["converged"] is True

    @pytest.mark.asyncio
    async def test_handles_invalid_score_value(self):
        """Test handling when SCORE has an invalid (non-numeric) value."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: abc\nFEEDBACK:\nBad format."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        # The regex won't match "abc" since it looks for [\d.]+,
        # so score defaults to 0.0
        result = await handler(context, threshold=0.85, max_iterations=5)
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_reason_on_continuing(self):
        """Test reason string when continuing refinement."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.6\nFEEDBACK:\nKeep going."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert "0.60" in result["reason"] or "0.6" in result["reason"]
        assert "continuing" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_reason_on_score_convergence(self):
        """Test reason string when converging by score."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("review_cycle_check")

        context = MagicMock()
        context.step_outputs = {"review": {"response": "SCORE: 0.95\nFEEDBACK:\nGreat."}}
        context.state = {}
        context.set_state = MagicMock(side_effect=lambda k, v: context.state.__setitem__(k, v))

        result = await handler(context, threshold=0.85, max_iterations=5)
        assert ">=" in result["reason"]
        assert "threshold" in result["reason"].lower()
