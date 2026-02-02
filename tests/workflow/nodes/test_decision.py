"""
Tests for Workflow Decision and Switch Nodes.

Tests cover:
- DecisionStep initialization
- DecisionStep first_match evaluation mode
- DecisionStep all evaluation mode
- DecisionStep default branch handling
- DecisionStep condition evaluation errors
- DecisionStep AI fallback path
- DecisionStep context summarization
- DecisionStep _evaluate_condition with builtins and helpers
- DecisionStep step output namespace injection (hyphen replacement)
- SwitchStep initialization
- SwitchStep exact case matching
- SwitchStep default case fallback
- SwitchStep value expression evaluation
- SwitchStep DictWrapper attribute access
- SwitchStep None value handling
- SwitchStep SafeEvalError handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.nodes.decision import DecisionStep, SwitchStep
from aragora.workflow.step import WorkflowContext


def _make_context(inputs=None, state=None, step_outputs=None, current_step_config=None):
    """Create a WorkflowContext for testing."""
    return WorkflowContext(
        workflow_id="wf_test",
        definition_id="def_test",
        inputs=inputs or {},
        state=state or {},
        step_outputs=step_outputs or {},
        current_step_config=current_step_config or {},
    )


# ============================================================================
# DecisionStep Initialization Tests
# ============================================================================


class TestDecisionStepInit:
    """Tests for DecisionStep initialization."""

    def test_basic_init(self):
        """Test basic DecisionStep initialization with name and config."""
        step = DecisionStep(
            name="Risk Router",
            config={
                "conditions": [{"name": "high", "expression": "True", "next_step": "review"}],
                "default_branch": "auto_approve",
            },
        )
        assert step.name == "Risk Router"
        assert step.config["default_branch"] == "auto_approve"
        assert len(step.config["conditions"]) == 1

    def test_init_no_config(self):
        """Test DecisionStep with no config."""
        step = DecisionStep(name="Empty Decision")
        assert step.config == {}

    def test_init_empty_config(self):
        """Test DecisionStep with empty config dict."""
        step = DecisionStep(name="Empty Config", config={})
        assert step.config == {}


# ============================================================================
# DecisionStep First-Match Mode Tests
# ============================================================================


class TestDecisionStepFirstMatch:
    """Tests for DecisionStep with first_match evaluation mode."""

    @pytest.mark.asyncio
    async def test_first_match_stops_at_first_true(self):
        """Test that first_match mode stops evaluating after the first true condition."""
        step = DecisionStep(
            name="First Match",
            config={
                "conditions": [
                    {"name": "cond_a", "expression": "True", "next_step": "step_a"},
                    {"name": "cond_b", "expression": "True", "next_step": "step_b"},
                ],
                "evaluation_mode": "first_match",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "step_a"
        assert result["decision_name"] == "cond_a"
        assert len(result["matched_branches"]) == 1
        # Only one evaluation should have been performed
        assert len(result["evaluations"]) == 1

    @pytest.mark.asyncio
    async def test_first_match_skips_false_conditions(self):
        """Test that first_match skips false conditions and returns the first true."""
        step = DecisionStep(
            name="Skip False",
            config={
                "conditions": [
                    {"name": "false_one", "expression": "False", "next_step": "nope"},
                    {"name": "false_two", "expression": "False", "next_step": "nope2"},
                    {"name": "true_one", "expression": "True", "next_step": "yes"},
                ],
                "evaluation_mode": "first_match",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "yes"
        assert result["decision_name"] == "true_one"
        assert len(result["matched_branches"]) == 1
        # All three conditions were evaluated up to the match
        assert len(result["evaluations"]) == 3

    @pytest.mark.asyncio
    async def test_first_match_is_default_mode(self):
        """Test that first_match is the default evaluation mode."""
        step = DecisionStep(
            name="Default Mode",
            config={
                "conditions": [
                    {"name": "a", "expression": "True", "next_step": "step_a"},
                    {"name": "b", "expression": "True", "next_step": "step_b"},
                ],
                # No evaluation_mode specified
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        # Should behave as first_match: only one evaluation performed
        assert result["decision"] == "step_a"
        assert len(result["evaluations"]) == 1

    @pytest.mark.asyncio
    async def test_first_match_with_input_expression(self):
        """Test first_match with expressions referencing inputs."""
        step = DecisionStep(
            name="Input Check",
            config={
                "conditions": [
                    {
                        "name": "high_score",
                        "expression": "inputs['score'] > 90",
                        "next_step": "excellent",
                    },
                    {
                        "name": "medium_score",
                        "expression": "inputs['score'] > 50",
                        "next_step": "good",
                    },
                ],
                "evaluation_mode": "first_match",
            },
        )
        ctx = _make_context(inputs={"score": 75})
        result = await step.execute(ctx)

        assert result["decision"] == "good"
        assert result["decision_name"] == "medium_score"


# ============================================================================
# DecisionStep All-Evaluation Mode Tests
# ============================================================================


class TestDecisionStepAllMode:
    """Tests for DecisionStep with 'all' evaluation mode."""

    @pytest.mark.asyncio
    async def test_all_mode_evaluates_all_conditions(self):
        """Test that 'all' mode evaluates every condition."""
        step = DecisionStep(
            name="All Mode",
            config={
                "conditions": [
                    {"name": "cond_a", "expression": "True", "next_step": "step_a"},
                    {"name": "cond_b", "expression": "False", "next_step": "step_b"},
                    {"name": "cond_c", "expression": "True", "next_step": "step_c"},
                ],
                "evaluation_mode": "all",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        # All three conditions evaluated
        assert len(result["evaluations"]) == 3
        # Two matched
        assert len(result["matched_branches"]) == 2
        # Decision uses the first matched branch
        assert result["decision"] == "step_a"
        assert result["decision_name"] == "cond_a"

    @pytest.mark.asyncio
    async def test_all_mode_no_match_uses_default(self):
        """Test that 'all' mode falls back to default when none match."""
        step = DecisionStep(
            name="All No Match",
            config={
                "conditions": [
                    {"name": "a", "expression": "False", "next_step": "step_a"},
                    {"name": "b", "expression": "False", "next_step": "step_b"},
                ],
                "evaluation_mode": "all",
                "default_branch": "fallback",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "fallback"
        assert result["decision_name"] == "default"
        assert len(result["evaluations"]) == 2
        assert len(result["matched_branches"]) == 0

    @pytest.mark.asyncio
    async def test_all_mode_collects_all_matched_branches(self):
        """Test that 'all' mode collects all matching branches."""
        step = DecisionStep(
            name="Collect All",
            config={
                "conditions": [
                    {"name": "x", "expression": "True", "next_step": "s_x"},
                    {"name": "y", "expression": "True", "next_step": "s_y"},
                    {"name": "z", "expression": "True", "next_step": "s_z"},
                ],
                "evaluation_mode": "all",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert len(result["matched_branches"]) == 3
        names = [b["name"] for b in result["matched_branches"]]
        assert names == ["x", "y", "z"]


# ============================================================================
# DecisionStep Default Branch Tests
# ============================================================================


class TestDecisionStepDefaultBranch:
    """Tests for DecisionStep default branch handling."""

    @pytest.mark.asyncio
    async def test_default_branch_when_no_conditions_match(self):
        """Test default branch is used when no conditions match."""
        step = DecisionStep(
            name="Default Test",
            config={
                "conditions": [
                    {"name": "never", "expression": "False", "next_step": "nope"},
                ],
                "default_branch": "default_step",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "default_step"
        assert result["decision_name"] == "default"
        assert result["next_step"] == "default_step"

    @pytest.mark.asyncio
    async def test_empty_default_branch(self):
        """Test empty string default branch when not specified."""
        step = DecisionStep(
            name="No Default",
            config={
                "conditions": [
                    {"name": "never", "expression": "False", "next_step": "nope"},
                ],
                # No default_branch specified
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == ""
        assert result["decision_name"] == "default"

    @pytest.mark.asyncio
    async def test_empty_conditions_list_uses_default(self):
        """Test that empty conditions list falls through to default."""
        step = DecisionStep(
            name="No Conditions",
            config={
                "conditions": [],
                "default_branch": "only_choice",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "only_choice"
        assert result["decision_name"] == "default"
        assert result["evaluations"] == []
        assert result["matched_branches"] == []

    @pytest.mark.asyncio
    async def test_no_conditions_key_uses_default(self):
        """Test that missing conditions key falls through to default."""
        step = DecisionStep(
            name="Missing Conditions",
            config={
                "default_branch": "fallback",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "fallback"


# ============================================================================
# DecisionStep Condition Evaluation Error Tests
# ============================================================================


class TestDecisionStepErrors:
    """Tests for DecisionStep error handling during condition evaluation."""

    @pytest.mark.asyncio
    async def test_invalid_expression_records_error(self):
        """Test that an invalid expression records an error and continues."""
        step = DecisionStep(
            name="Error Handling",
            config={
                "conditions": [
                    {
                        "name": "bad",
                        "expression": "undefined_variable > 0",
                        "next_step": "bad_step",
                    },
                    {"name": "good", "expression": "True", "next_step": "good_step"},
                ],
                "evaluation_mode": "first_match",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        # The bad condition should have been evaluated with an error
        assert result["evaluations"][0]["result"] is False
        assert "error" in result["evaluations"][0]
        # The good condition should still have been reached
        assert result["decision"] == "good_step"

    @pytest.mark.asyncio
    async def test_syntax_error_expression(self):
        """Test that a syntax error in expression is handled gracefully."""
        step = DecisionStep(
            name="Syntax Error",
            config={
                "conditions": [
                    {
                        "name": "broken",
                        "expression": "if True then yes",
                        "next_step": "broken_step",
                    },
                ],
                "default_branch": "safe_step",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["evaluations"][0]["result"] is False
        assert "error" in result["evaluations"][0]
        assert result["decision"] == "safe_step"

    @pytest.mark.asyncio
    async def test_multiple_errors_all_recorded(self):
        """Test that multiple condition errors are all recorded."""
        step = DecisionStep(
            name="Multiple Errors",
            config={
                "conditions": [
                    {"name": "err1", "expression": "bad_var_1", "next_step": "s1"},
                    {"name": "err2", "expression": "bad_var_2", "next_step": "s2"},
                ],
                "evaluation_mode": "all",
                "default_branch": "safe",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert len(result["evaluations"]) == 2
        assert all("error" in e for e in result["evaluations"])
        assert result["decision"] == "safe"

    @pytest.mark.asyncio
    async def test_unnamed_condition_uses_unnamed_default(self):
        """Test that condition without name uses 'unnamed'."""
        step = DecisionStep(
            name="Unnamed",
            config={
                "conditions": [
                    {"expression": "True", "next_step": "target"},
                ],
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["evaluations"][0]["name"] == "unnamed"

    @pytest.mark.asyncio
    async def test_condition_without_next_step(self):
        """Test that condition without next_step defaults to empty string."""
        step = DecisionStep(
            name="No Next",
            config={
                "conditions": [
                    {"name": "no_target", "expression": "True"},
                ],
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == ""
        assert result["matched_branches"][0]["next_step"] == ""


# ============================================================================
# DecisionStep Expression Evaluation Tests
# ============================================================================


class TestDecisionStepExpressions:
    """Tests for _evaluate_condition with various expressions."""

    def test_simple_comparison(self):
        """Test simple numeric comparison."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"score": 85})
        assert step._evaluate_condition("inputs['score'] > 80", ctx) is True
        assert step._evaluate_condition("inputs['score'] > 90", ctx) is False

    def test_boolean_logic(self):
        """Test boolean and/or logic."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"x": 5, "y": 10})
        assert step._evaluate_condition("inputs['x'] > 0 and inputs['y'] > 0", ctx) is True
        assert step._evaluate_condition("inputs['x'] > 100 or inputs['y'] > 5", ctx) is True
        assert step._evaluate_condition("inputs['x'] > 100 and inputs['y'] > 100", ctx) is False

    def test_builtin_len(self):
        """Test len() builtin in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"items": [1, 2, 3]})
        assert step._evaluate_condition("len(inputs['items']) == 3", ctx) is True
        assert step._evaluate_condition("len(inputs['items']) > 5", ctx) is False

    def test_builtin_str(self):
        """Test str() builtin in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"num": 42})
        assert step._evaluate_condition("str(inputs['num']) == '42'", ctx) is True

    def test_builtin_int_float(self):
        """Test int() and float() builtins in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"val": "10"})
        assert step._evaluate_condition("int(inputs['val']) > 5", ctx) is True
        assert step._evaluate_condition("float(inputs['val']) == 10.0", ctx) is True

    def test_builtin_abs(self):
        """Test abs() builtin in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"val": -5})
        assert step._evaluate_condition("abs(inputs['val']) == 5", ctx) is True

    def test_builtin_min_max(self):
        """Test min() and max() builtins in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"vals": [3, 1, 4, 1, 5]})
        assert step._evaluate_condition("min(inputs['vals']) == 1", ctx) is True
        assert step._evaluate_condition("max(inputs['vals']) == 5", ctx) is True

    def test_builtin_sum(self):
        """Test sum() builtin in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"nums": [1, 2, 3, 4]})
        assert step._evaluate_condition("sum(inputs['nums']) == 10", ctx) is True

    def test_builtin_all_any(self):
        """Test all() and any() builtins in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"flags": [True, True, True], "mixed": [True, False]})
        assert step._evaluate_condition("all(inputs['flags'])", ctx) is True
        assert step._evaluate_condition("all(inputs['mixed'])", ctx) is False
        assert step._evaluate_condition("any(inputs['mixed'])", ctx) is True

    def test_builtin_bool(self):
        """Test bool() builtin in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"val": 0})
        assert step._evaluate_condition("bool(inputs['val'])", ctx) is False

    def test_helper_contains(self):
        """Test contains() helper function."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"text": "hello world"})
        assert step._evaluate_condition("contains(inputs['text'], 'world')", ctx) is True
        assert step._evaluate_condition("contains(inputs['text'], 'xyz')", ctx) is False

    def test_helper_contains_with_none(self):
        """Test contains() helper with None value."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"val": None})
        # contains checks `if a` first, so None should return False
        assert step._evaluate_condition("contains(inputs['val'], 'x')", ctx) is False

    def test_helper_startswith(self):
        """Test startswith() helper function."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"text": "hello world"})
        assert step._evaluate_condition("startswith(inputs['text'], 'hello')", ctx) is True
        assert step._evaluate_condition("startswith(inputs['text'], 'world')", ctx) is False

    def test_helper_startswith_non_string(self):
        """Test startswith() helper with non-string value."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"val": 123})
        assert step._evaluate_condition("startswith(inputs['val'], '1')", ctx) is False

    def test_helper_endswith(self):
        """Test endswith() helper function."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"text": "hello world"})
        assert step._evaluate_condition("endswith(inputs['text'], 'world')", ctx) is True
        assert step._evaluate_condition("endswith(inputs['text'], 'hello')", ctx) is False

    def test_helper_endswith_non_string(self):
        """Test endswith() helper with non-string value."""
        step = DecisionStep(name="Test")
        ctx = _make_context(inputs={"val": [1, 2]})
        assert step._evaluate_condition("endswith(inputs['val'], 'x')", ctx) is False

    def test_outputs_alias(self):
        """Test that 'outputs' and 'step' are aliases for step_outputs."""
        step = DecisionStep(name="Test")
        ctx = _make_context(step_outputs={"analysis": {"risk": 0.9}})
        assert step._evaluate_condition("outputs['analysis']['risk'] > 0.5", ctx) is True
        assert step._evaluate_condition("step['analysis']['risk'] > 0.5", ctx) is True

    def test_state_access(self):
        """Test accessing state in expressions."""
        step = DecisionStep(name="Test")
        ctx = _make_context(state={"retry_count": 3})
        assert step._evaluate_condition("state['retry_count'] >= 3", ctx) is True

    def test_step_output_as_direct_variable(self):
        """Test that step outputs are available as direct namespace variables."""
        step = DecisionStep(name="Test")
        ctx = _make_context(step_outputs={"risk_check": {"score": 0.95}})
        assert step._evaluate_condition("risk_check['score'] > 0.9", ctx) is True

    def test_step_output_hyphen_replaced_with_underscore(self):
        """Test that hyphens in step IDs are replaced with underscores."""
        step = DecisionStep(name="Test")
        ctx = _make_context(step_outputs={"risk-check-v2": {"score": 0.95}})
        assert step._evaluate_condition("risk_check_v2['score'] > 0.9", ctx) is True

    def test_step_output_dot_replaced_with_underscore(self):
        """Test that dots in step IDs are replaced with underscores."""
        step = DecisionStep(name="Test")
        ctx = _make_context(step_outputs={"module.check": {"ok": True}})
        assert step._evaluate_condition("module_check['ok'] == True", ctx) is True

    def test_evaluate_condition_raises_on_unsafe(self):
        """Test that _evaluate_condition raises SafeEvalError on unsafe expressions."""
        from aragora.workflow.safe_eval import SafeEvalError

        step = DecisionStep(name="Test")
        ctx = _make_context()
        with pytest.raises(SafeEvalError):
            step._evaluate_condition("__import__('os')", ctx)


# ============================================================================
# DecisionStep AI Fallback Tests
# ============================================================================


class TestDecisionStepAIFallback:
    """Tests for DecisionStep AI fallback path."""

    @pytest.mark.asyncio
    async def test_ai_fallback_triggered_when_no_match(self):
        """Test AI fallback is triggered when no conditions match."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="high_risk seems most appropriate")

        step = DecisionStep(
            name="AI Fallback",
            config={
                "conditions": [
                    {"name": "high_risk", "expression": "False", "next_step": "review"},
                    {"name": "low_risk", "expression": "False", "next_step": "approve"},
                ],
                "default_branch": "manual",
                "ai_fallback": True,
            },
        )
        ctx = _make_context(inputs={"data": "test"})

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            result = await step.execute(ctx)

        assert result["decision"] == "review"
        assert result["decision_name"] == "high_risk"
        # AI fallback evaluation recorded
        ai_eval = [e for e in result["evaluations"] if e["name"] == "ai_fallback"]
        assert len(ai_eval) == 1
        assert ai_eval[0]["result"] is True
        assert "ai_reasoning" in ai_eval[0]

    @pytest.mark.asyncio
    async def test_ai_fallback_with_custom_prompt(self):
        """Test AI fallback uses custom prompt when provided."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="low_risk")

        step = DecisionStep(
            name="AI Custom Prompt",
            config={
                "conditions": [
                    {"name": "low_risk", "expression": "False", "next_step": "approve"},
                ],
                "default_branch": "manual",
                "ai_fallback": True,
                "ai_prompt": "Custom decision prompt: decide wisely.",
            },
        )
        ctx = _make_context()

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            result = await step.execute(ctx)

        mock_agent.generate.assert_called_once_with("Custom decision prompt: decide wisely.")
        assert result["decision"] == "approve"

    @pytest.mark.asyncio
    async def test_ai_fallback_no_matching_branch_uses_default(self):
        """Test AI fallback returns default when AI response matches no branch."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="I am not sure what to pick")

        step = DecisionStep(
            name="AI No Match",
            config={
                "conditions": [
                    {"name": "specific", "expression": "False", "next_step": "target"},
                ],
                "default_branch": "default_target",
                "ai_fallback": True,
            },
        )
        ctx = _make_context()

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            result = await step.execute(ctx)

        assert result["decision"] == "default_target"
        assert result["decision_name"] == "default"

    @pytest.mark.asyncio
    async def test_ai_fallback_exception_uses_default(self):
        """Test AI fallback handles exceptions and falls back to default."""
        step = DecisionStep(
            name="AI Error",
            config={
                "conditions": [
                    {"name": "cond", "expression": "False", "next_step": "target"},
                ],
                "default_branch": "safe_default",
                "ai_fallback": True,
            },
        )
        ctx = _make_context()

        with patch(
            "aragora.agents.create_agent",
            side_effect=RuntimeError("API unavailable"),
        ):
            result = await step.execute(ctx)

        assert result["decision"] == "safe_default"

    @pytest.mark.asyncio
    async def test_ai_fallback_not_triggered_when_match_exists(self):
        """Test that AI fallback is not triggered when a condition matches."""
        step = DecisionStep(
            name="Match Exists",
            config={
                "conditions": [
                    {"name": "match", "expression": "True", "next_step": "matched"},
                ],
                "default_branch": "default",
                "ai_fallback": True,
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "matched"
        # No AI evaluation recorded
        ai_evals = [e for e in result["evaluations"] if e["name"] == "ai_fallback"]
        assert len(ai_evals) == 0

    @pytest.mark.asyncio
    async def test_ai_fallback_disabled_uses_default_directly(self):
        """Test that when ai_fallback is False, default is used directly."""
        step = DecisionStep(
            name="No AI",
            config={
                "conditions": [
                    {"name": "miss", "expression": "False", "next_step": "nope"},
                ],
                "default_branch": "direct_default",
                "ai_fallback": False,
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["decision"] == "direct_default"
        assert result["decision_name"] == "default"


# ============================================================================
# DecisionStep Context Summarization Tests
# ============================================================================


class TestDecisionStepSummarize:
    """Tests for DecisionStep._summarize_context."""

    def test_summarize_with_inputs_and_outputs(self):
        """Test context summarization includes inputs and step outputs."""
        step = DecisionStep(name="Summarize")
        ctx = _make_context(
            inputs={"topic": "AI safety", "urgency": "high"},
            step_outputs={"analysis": {"risk": 0.8, "confidence": 0.9}},
        )
        summary = step._summarize_context(ctx)

        assert "Inputs:" in summary
        assert "topic" in summary
        assert "AI safety" in summary
        assert "Step Outputs:" in summary
        assert "analysis" in summary

    def test_summarize_empty_context(self):
        """Test context summarization with empty context."""
        step = DecisionStep(name="Summarize")
        ctx = _make_context()
        summary = step._summarize_context(ctx)

        assert summary == ""

    def test_summarize_truncates_long_values(self):
        """Test that summarization truncates values exceeding 200 chars."""
        step = DecisionStep(name="Summarize")
        long_value = "x" * 500
        ctx = _make_context(inputs={"long_key": long_value})
        summary = step._summarize_context(ctx)

        # The value should be truncated to 200 chars
        assert len(summary) < 500

    def test_summarize_limits_items(self):
        """Test that summarization limits to 5 items."""
        step = DecisionStep(name="Summarize")
        inputs = {f"key_{i}": f"value_{i}" for i in range(10)}
        ctx = _make_context(inputs=inputs)
        summary = step._summarize_context(ctx)

        # Should only include first 5
        assert "key_0" in summary
        assert "key_4" in summary
        assert "key_5" not in summary

    def test_summarize_dict_step_output(self):
        """Test summarization of dict-type step outputs."""
        step = DecisionStep(name="Summarize")
        ctx = _make_context(
            step_outputs={
                "analysis": {
                    "field1": "value1",
                    "field2": "value2",
                    "field3": "value3",
                    "field4": "value4",
                }
            },
        )
        summary = step._summarize_context(ctx)

        assert "Step Outputs:" in summary
        assert "analysis" in summary

    def test_summarize_non_dict_step_output(self):
        """Test summarization of non-dict step outputs."""
        step = DecisionStep(name="Summarize")
        ctx = _make_context(step_outputs={"simple": "just a string"})
        summary = step._summarize_context(ctx)

        assert "just a string" in summary


# ============================================================================
# DecisionStep Config Merge Tests
# ============================================================================


class TestDecisionStepConfigMerge:
    """Tests for DecisionStep config merging with current_step_config."""

    @pytest.mark.asyncio
    async def test_current_step_config_overrides(self):
        """Test that current_step_config overrides step config."""
        step = DecisionStep(
            name="Override",
            config={
                "conditions": [
                    {"name": "a", "expression": "True", "next_step": "step_a"},
                ],
                "default_branch": "original_default",
            },
        )
        ctx = _make_context(current_step_config={"default_branch": "overridden_default"})
        # The conditions should still match, so default is not used here
        result = await step.execute(ctx)
        assert result["decision"] == "step_a"

    @pytest.mark.asyncio
    async def test_current_step_config_provides_conditions(self):
        """Test conditions provided via current_step_config."""
        step = DecisionStep(name="Dynamic", config={})
        ctx = _make_context(
            current_step_config={
                "conditions": [
                    {"name": "dynamic", "expression": "True", "next_step": "dynamic_step"},
                ],
            }
        )
        result = await step.execute(ctx)
        assert result["decision"] == "dynamic_step"


# ============================================================================
# DecisionStep Output Structure Tests
# ============================================================================


class TestDecisionStepOutput:
    """Tests for DecisionStep output structure."""

    @pytest.mark.asyncio
    async def test_output_contains_required_keys(self):
        """Test that output contains all required keys."""
        step = DecisionStep(
            name="Output",
            config={
                "conditions": [
                    {"name": "cond", "expression": "True", "next_step": "target"},
                ],
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert "decision" in result
        assert "decision_name" in result
        assert "matched_branches" in result
        assert "evaluations" in result
        assert "next_step" in result

    @pytest.mark.asyncio
    async def test_next_step_equals_decision(self):
        """Test that next_step mirrors decision for engine compatibility."""
        step = DecisionStep(
            name="Mirror",
            config={
                "conditions": [
                    {"name": "x", "expression": "True", "next_step": "the_step"},
                ],
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["next_step"] == result["decision"]
        assert result["next_step"] == "the_step"

    @pytest.mark.asyncio
    async def test_evaluation_includes_expression(self):
        """Test that evaluations include the original expression."""
        step = DecisionStep(
            name="Eval Detail",
            config={
                "conditions": [
                    {
                        "name": "check",
                        "expression": "inputs['x'] > 10",
                        "next_step": "target",
                    },
                ],
            },
        )
        ctx = _make_context(inputs={"x": 5})
        result = await step.execute(ctx)

        assert result["evaluations"][0]["expression"] == "inputs['x'] > 10"
        assert result["evaluations"][0]["result"] is False


# ============================================================================
# SwitchStep Initialization Tests
# ============================================================================


class TestSwitchStepInit:
    """Tests for SwitchStep initialization."""

    def test_basic_init(self):
        """Test basic SwitchStep initialization."""
        step = SwitchStep(
            name="Category Router",
            config={
                "value": "inputs.category",
                "cases": {"legal": "legal_review", "tech": "tech_review"},
                "default": "general_review",
            },
        )
        assert step.name == "Category Router"
        assert step.config["default"] == "general_review"

    def test_init_no_config(self):
        """Test SwitchStep with no config."""
        step = SwitchStep(name="Empty Switch")
        assert step.config == {}

    def test_init_empty_cases(self):
        """Test SwitchStep with empty cases dict."""
        step = SwitchStep(
            name="No Cases",
            config={"value": "inputs.x", "cases": {}, "default": "fallback"},
        )
        assert step.config["cases"] == {}


# ============================================================================
# SwitchStep Execution Tests
# ============================================================================


class TestSwitchStepExecution:
    """Tests for SwitchStep execution."""

    @pytest.mark.asyncio
    async def test_exact_case_match(self):
        """Test exact string case matching."""
        step = SwitchStep(
            name="Switch",
            config={
                "value": "inputs['category']",
                "cases": {
                    "legal": "legal_review",
                    "tech": "tech_review",
                    "finance": "finance_review",
                },
                "default": "general_review",
            },
        )
        ctx = _make_context(inputs={"category": "tech"})
        result = await step.execute(ctx)

        assert result["value"] == "tech"
        assert result["matched_case"] == "tech"
        assert result["next_step"] == "tech_review"

    @pytest.mark.asyncio
    async def test_default_case(self):
        """Test default case when no match found."""
        step = SwitchStep(
            name="Switch Default",
            config={
                "value": "inputs['category']",
                "cases": {"legal": "legal_review"},
                "default": "general_review",
            },
        )
        ctx = _make_context(inputs={"category": "unknown"})
        result = await step.execute(ctx)

        assert result["value"] == "unknown"
        assert result["matched_case"] == "default"
        assert result["next_step"] == "general_review"

    @pytest.mark.asyncio
    async def test_numeric_value_converted_to_string(self):
        """Test that numeric values are converted to string for case matching."""
        step = SwitchStep(
            name="Numeric Switch",
            config={
                "value": "inputs['priority']",
                "cases": {"1": "urgent", "2": "normal", "3": "low"},
                "default": "unknown_priority",
            },
        )
        ctx = _make_context(inputs={"priority": 1})
        result = await step.execute(ctx)

        assert result["value"] == "1"
        assert result["next_step"] == "urgent"

    @pytest.mark.asyncio
    async def test_boolean_value_converted_to_string(self):
        """Test that boolean values are converted to string for case matching."""
        step = SwitchStep(
            name="Bool Switch",
            config={
                "value": "inputs['active']",
                "cases": {"True": "active_path", "False": "inactive_path"},
                "default": "unknown",
            },
        )
        ctx = _make_context(inputs={"active": True})
        result = await step.execute(ctx)

        assert result["value"] == "True"
        assert result["next_step"] == "active_path"

    @pytest.mark.asyncio
    async def test_none_value_handling(self):
        """Test handling when value expression evaluates to None."""
        step = SwitchStep(
            name="None Switch",
            config={
                "value": "inputs.get('missing')",
                "cases": {"": "empty_path", "something": "other"},
                "default": "default_path",
            },
        )
        ctx = _make_context(inputs={})
        result = await step.execute(ctx)

        # inputs.get('missing') returns None, which becomes "" in value_str
        assert result["value"] == ""
        assert result["next_step"] == "empty_path"

    @pytest.mark.asyncio
    async def test_key_error_propagates(self):
        """Test that KeyError from missing dict key propagates (not caught as SafeEvalError)."""
        step = SwitchStep(
            name="Key Error",
            config={
                "value": "inputs['missing']",
                "cases": {"x": "path_x"},
                "default": "default_path",
            },
        )
        ctx = _make_context(inputs={})
        # KeyError from subscript is not a SafeEvalError, so it propagates
        with pytest.raises(KeyError):
            await step.execute(ctx)

    @pytest.mark.asyncio
    async def test_safe_eval_error_sets_value_none(self):
        """Test that SafeEvalError causes value to be None."""
        step = SwitchStep(
            name="Eval Error",
            config={
                "value": "completely_undefined_var",
                "cases": {"test": "path"},
                "default": "error_default",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["value"] == ""
        assert result["next_step"] == "error_default"

    @pytest.mark.asyncio
    async def test_dict_wrapper_attribute_access(self):
        """Test that DictWrapper enables attribute-style access to inputs."""
        step = SwitchStep(
            name="Attr Access",
            config={
                "value": "inputs.document_type",
                "cases": {
                    "contract": "contract_review",
                    "invoice": "invoice_processing",
                },
                "default": "general",
            },
        )
        ctx = _make_context(inputs={"document_type": "contract"})
        result = await step.execute(ctx)

        assert result["value"] == "contract"
        assert result["next_step"] == "contract_review"

    @pytest.mark.asyncio
    async def test_outputs_attribute_access(self):
        """Test DictWrapper attribute access on step outputs with subscript for nested."""
        step = SwitchStep(
            name="Output Attr",
            config={
                "value": "outputs.classifier['label']",
                "cases": {"spam": "reject", "ham": "accept"},
                "default": "review",
            },
        )
        ctx = _make_context(step_outputs={"classifier": {"label": "spam"}})
        result = await step.execute(ctx)

        assert result["value"] == "spam"
        assert result["next_step"] == "reject"

    @pytest.mark.asyncio
    async def test_nested_attribute_access_fails_on_plain_dict(self):
        """Test that nested attribute access fails when inner dict is not DictWrapper."""
        step = SwitchStep(
            name="Nested Fail",
            config={
                "value": "outputs.classifier.label",
                "cases": {"spam": "reject"},
                "default": "fallback",
            },
        )
        ctx = _make_context(step_outputs={"classifier": {"label": "spam"}})
        result = await step.execute(ctx)

        # DictWrapper only wraps top-level; inner dict is plain, so .label fails
        assert result["value"] == ""
        assert result["next_step"] == "fallback"

    @pytest.mark.asyncio
    async def test_state_attribute_access(self):
        """Test DictWrapper attribute access on state."""
        step = SwitchStep(
            name="State Attr",
            config={
                "value": "state.mode",
                "cases": {"fast": "quick_path", "thorough": "deep_path"},
                "default": "normal_path",
            },
        )
        ctx = _make_context(state={"mode": "thorough"})
        result = await step.execute(ctx)

        assert result["value"] == "thorough"
        assert result["next_step"] == "deep_path"

    @pytest.mark.asyncio
    async def test_empty_value_expression(self):
        """Test with empty value expression."""
        step = SwitchStep(
            name="Empty Expr",
            config={
                "value": "",
                "cases": {"test": "path"},
                "default": "empty_default",
            },
        )
        ctx = _make_context()
        result = await step.execute(ctx)

        # Empty expression should cause SafeEvalError
        assert result["next_step"] == "empty_default"

    @pytest.mark.asyncio
    async def test_empty_cases_dict(self):
        """Test with empty cases dict."""
        step = SwitchStep(
            name="No Cases",
            config={
                "value": "inputs['x']",
                "cases": {},
                "default": "only_default",
            },
        )
        ctx = _make_context(inputs={"x": "anything"})
        result = await step.execute(ctx)

        assert result["next_step"] == "only_default"
        assert result["matched_case"] == "default"

    @pytest.mark.asyncio
    async def test_no_default_and_no_match(self):
        """Test with no default and no matching case."""
        step = SwitchStep(
            name="No Default",
            config={
                "value": "inputs['x']",
                "cases": {"a": "path_a"},
            },
        )
        ctx = _make_context(inputs={"x": "b"})
        result = await step.execute(ctx)

        assert result["next_step"] == ""

    @pytest.mark.asyncio
    async def test_config_merge_with_current_step_config(self):
        """Test that current_step_config merges with step config."""
        step = SwitchStep(
            name="Merge",
            config={
                "value": "inputs['x']",
                "cases": {"a": "path_a"},
                "default": "original",
            },
        )
        ctx = _make_context(
            inputs={"x": "b"},
            current_step_config={"default": "overridden"},
        )
        result = await step.execute(ctx)

        assert result["next_step"] == "overridden"

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        """Test with a more complex value expression."""
        step = SwitchStep(
            name="Complex",
            config={
                "value": "str(inputs['score'] > 80)",
                "cases": {"True": "pass", "False": "fail"},
                "default": "error",
            },
        )
        ctx = _make_context(inputs={"score": 95})
        result = await step.execute(ctx)

        assert result["value"] == "True"
        assert result["next_step"] == "pass"


# ============================================================================
# SwitchStep Output Structure Tests
# ============================================================================


class TestSwitchStepOutput:
    """Tests for SwitchStep output structure."""

    @pytest.mark.asyncio
    async def test_output_contains_required_keys(self):
        """Test that output contains all required keys."""
        step = SwitchStep(
            name="Output",
            config={
                "value": "inputs['x']",
                "cases": {"a": "path_a"},
                "default": "default",
            },
        )
        ctx = _make_context(inputs={"x": "a"})
        result = await step.execute(ctx)

        assert "value" in result
        assert "matched_case" in result
        assert "next_step" in result

    @pytest.mark.asyncio
    async def test_matched_case_reflects_actual_match(self):
        """Test that matched_case shows 'default' when no case matches."""
        step = SwitchStep(
            name="Match Info",
            config={
                "value": "inputs['x']",
                "cases": {"a": "path_a"},
                "default": "fallback",
            },
        )

        # Matching case
        ctx_match = _make_context(inputs={"x": "a"})
        result_match = await step.execute(ctx_match)
        assert result_match["matched_case"] == "a"

        # Non-matching case
        ctx_miss = _make_context(inputs={"x": "z"})
        result_miss = await step.execute(ctx_miss)
        assert result_miss["matched_case"] == "default"
