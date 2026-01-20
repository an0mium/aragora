"""
Tests for RLM iterative refinement features (Prime Intellect alignment).

Tests:
- RLMResult ready/iteration fields
- REPLState refinement tracking
- FINAL primitive with ready parameter
- query_with_refinement method
- Refinement metrics
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rlm.types import RLMResult, RLMContext, RLMConfig, AbstractionLevel


class TestRLMResultRefinement:
    """Test RLMResult refinement fields."""

    def test_result_has_ready_field(self):
        """Test that RLMResult has ready field defaulting to True."""
        result = RLMResult(answer="test")
        assert hasattr(result, "ready")
        assert result.ready is True

    def test_result_has_iteration_field(self):
        """Test that RLMResult has iteration field defaulting to 0."""
        result = RLMResult(answer="test")
        assert hasattr(result, "iteration")
        assert result.iteration == 0

    def test_result_has_refinement_history(self):
        """Test that RLMResult has refinement_history field."""
        result = RLMResult(answer="test")
        assert hasattr(result, "refinement_history")
        assert result.refinement_history == []

    def test_result_with_ready_false(self):
        """Test creating result with ready=False."""
        result = RLMResult(answer="partial", ready=False, iteration=1)
        assert result.ready is False
        assert result.iteration == 1

    def test_result_with_refinement_history(self):
        """Test result with refinement history."""
        history = ["first attempt", "second attempt"]
        result = RLMResult(
            answer="final answer",
            ready=True,
            iteration=2,
            refinement_history=history,
        )
        assert result.refinement_history == history
        assert len(result.refinement_history) == 2


class TestREPLStateRefinement:
    """Test REPLState refinement tracking."""

    def test_repl_state_has_ready(self):
        """Test that REPLState has ready field."""
        from aragora.rlm.repl import REPLState

        state = REPLState()
        assert hasattr(state, "ready")
        assert state.ready is True

    def test_repl_state_has_iteration(self):
        """Test that REPLState has iteration field."""
        from aragora.rlm.repl import REPLState

        state = REPLState()
        assert hasattr(state, "iteration")
        assert state.iteration == 0

    def test_repl_state_has_feedback(self):
        """Test that REPLState has feedback field."""
        from aragora.rlm.repl import REPLState

        state = REPLState()
        assert hasattr(state, "feedback")
        assert state.feedback is None


class TestFINALPrimitive:
    """Test FINAL primitive with ready parameter."""

    def test_final_defaults_to_ready(self):
        """Test FINAL() defaults to ready=True."""
        from aragora.rlm.repl import RLMEnvironment, REPLState

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env._final("test answer")
        assert env.state.final_answer == "test answer"
        assert env.state.ready is True

    def test_final_with_ready_false(self):
        """Test FINAL() with ready=False."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env._final("partial answer", ready=False)
        assert env.state.final_answer == "partial answer"
        assert env.state.ready is False

    def test_final_var_with_ready(self):
        """Test FINAL_VAR() with ready parameter."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env.state.namespace["my_answer"] = "computed answer"
        env._final_var("my_answer", ready=True)
        assert env.state.final_var == "my_answer"
        assert env.state.ready is True


class TestSetReadyPrimitive:
    """Test SET_READY primitive."""

    def test_set_ready_true(self):
        """Test SET_READY(True)."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env.state.ready = False
        env._set_ready(True)
        assert env.state.ready is True

    def test_set_ready_false(self):
        """Test SET_READY(False)."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env._set_ready(False)
        assert env.state.ready is False


class TestFeedbackPrimitive:
    """Test FEEDBACK primitive."""

    def test_feedback_returns_none_initially(self):
        """Test FEEDBACK() returns None on first iteration."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        feedback = env._get_feedback()
        assert feedback is None

    def test_feedback_returns_set_value(self):
        """Test FEEDBACK() returns value set via set_iteration_context."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env.set_iteration_context(1, "Please improve your answer")
        feedback = env._get_feedback()
        assert feedback == "Please improve your answer"


class TestSetIterationContext:
    """Test set_iteration_context method."""

    def test_sets_iteration_and_feedback(self):
        """Test that set_iteration_context sets iteration and feedback."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env.set_iteration_context(2, "Try harder")

        assert env.state.iteration == 2
        assert env.state.feedback == "Try harder"
        assert env.state.ready is True  # Reset to True
        assert env.state.final_answer is None  # Cleared

    def test_clears_previous_final_answer(self):
        """Test that set_iteration_context clears previous final answer."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env._final("previous answer", ready=False)
        assert env.state.final_answer == "previous answer"

        env.set_iteration_context(1, "feedback")
        assert env.state.final_answer is None
        assert env.state.final_var is None


class TestGetResultWithRefinement:
    """Test get_result includes refinement fields."""

    def test_get_result_includes_ready(self):
        """Test that get_result includes ready field."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env._final("answer", ready=False)
        result = env.get_result()

        assert result.ready is False

    def test_get_result_includes_iteration(self):
        """Test that get_result includes iteration field."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        env.set_iteration_context(3, None)
        env._final("answer")
        result = env.get_result()

        assert result.iteration == 3


class TestMaxOutputChars:
    """Test MAX_OUTPUT_CHARS configuration."""

    def test_max_output_chars_is_8192(self):
        """Test MAX_OUTPUT_CHARS is 8192 (Prime Intellect alignment)."""
        from aragora.rlm.repl import RLMEnvironment

        assert RLMEnvironment.MAX_OUTPUT_CHARS == 8192


class TestREPLNamespaceHasNewPrimitives:
    """Test REPL namespace includes new primitives."""

    def test_namespace_has_set_ready(self):
        """Test namespace has SET_READY primitive."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        assert "SET_READY" in env.state.namespace

    def test_namespace_has_feedback(self):
        """Test namespace has FEEDBACK primitive."""
        from aragora.rlm.repl import RLMEnvironment

        context = RLMContext(
            original_content="test content",
            original_tokens=10,
        )
        config = RLMConfig()
        env = RLMEnvironment(config, context)

        assert "FEEDBACK" in env.state.namespace
