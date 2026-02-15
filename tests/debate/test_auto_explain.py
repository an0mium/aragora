"""Tests for auto-explain default behavior (Task 14B).

Validates that:
- auto_explain defaults to True in ArenaExtensions
- auto_explain defaults to True in ExtensionsConfig
- auto_explain defaults to True in PostDebateConfig
- explanation_summary appears in status update
- Can be disabled with auto_explain=False
- Explanation failure is non-critical
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAutoExplainDefaults:
    """Tests for auto_explain defaulting to True."""

    def test_arena_extensions_auto_explain_default_true(self):
        """ArenaExtensions should default auto_explain to True."""
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions()
        assert ext.auto_explain is True

    def test_extensions_config_auto_explain_default_true(self):
        """ExtensionsConfig should default auto_explain to True."""
        from aragora.debate.extensions import ExtensionsConfig

        config = ExtensionsConfig()
        assert config.auto_explain is True

    def test_post_debate_config_auto_explain_default_true(self):
        """PostDebateConfig should default auto_explain to True."""
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        config = PostDebateConfig()
        assert config.auto_explain is True

    def test_can_disable_auto_explain(self):
        """auto_explain can be explicitly set to False."""
        from aragora.debate.extensions import ArenaExtensions, ExtensionsConfig
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        ext = ArenaExtensions(auto_explain=False)
        assert ext.auto_explain is False

        config = ExtensionsConfig(auto_explain=False)
        assert config.auto_explain is False

        pdc = PostDebateConfig(auto_explain=False)
        assert pdc.auto_explain is False

    def test_explanation_generated_by_default(self):
        """With auto_explain=True (default), _auto_generate_explanation should run."""
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions()
        assert ext.has_explanation is True

    def test_explanation_failure_is_non_critical(self):
        """Explanation generation failure should not raise."""
        from aragora.debate.extensions import ArenaExtensions

        ext = ArenaExtensions(auto_explain=True)

        # Create mock context and result
        ctx = MagicMock()
        ctx.query = "Test question"
        ctx.debate_id = "test-debate-1"
        ctx.environment = MagicMock()
        ctx.environment.task = "Test question"

        result = MagicMock()
        result.final_answer = "Test answer"
        result.consensus = None
        result.messages = []

        # Patch ExplanationBuilder to raise (imported locally inside the method)
        with patch(
            "aragora.explainability.builder.ExplanationBuilder",
            side_effect=RuntimeError("builder failed"),
        ):
            # Should not raise
            ext._auto_generate_explanation(ctx, result)


class TestExplanationSummaryInStatus:
    """Tests for explanation_summary in status update dict."""

    def test_explanation_summary_included_in_status(self):
        """Status update should include explanation_summary when available."""
        from aragora.server.debate_controller import DebateController

        controller = DebateController(
            factory=MagicMock(),
            emitter=MagicMock(),
            storage=MagicMock(),
        )

        result = MagicMock()
        result.final_answer = "Answer here"
        result.consensus_reached = True
        result.confidence = 0.8
        result.status = "completed"
        result.agent_failures = []
        result.participants = ["a", "b"]
        result.grounded_verdict = None
        result.total_cost_usd = 0.0
        result.per_agent_cost = {}

        # With explanation on result
        result.explanation = MagicMock()
        result.explanation.summary = "Key reasons for this decision..."

        config = MagicMock()
        config.question = "Test question"
        config.agents_str = "a,b"
        config.rounds = 3
        config.metadata = {}
        config.debate_format = "full"
        config.debate_id = "test-1"

        with patch(
            "aragora.server.debate_controller.run_async"
        ) as mock_run_async, patch(
            "aragora.server.debate_controller.update_debate_status"
        ) as mock_update, patch(
            "aragora.server.debate_controller.create_arena_hooks"
        ), patch(
            "aragora.server.debate_controller.wrap_agent_for_streaming"
        ), patch(
            "aragora.storage.receipt_store.get_receipt_store"
        ):
            mock_run_async.return_value = result
            controller.factory.create_arena.return_value = MagicMock()
            controller.factory.create_arena.return_value.protocol = MagicMock()
            controller.factory.create_arena.return_value.protocol.timeout_seconds = 0

            controller._run_debate(config, "test-debate-1")

            # Find the 'completed' status update
            completed_calls = [
                c
                for c in mock_update.call_args_list
                if len(c.args) >= 2 and c.args[1] == "completed"
            ]
            assert len(completed_calls) >= 1
            first_completed = completed_calls[0]
            result_data = first_completed.kwargs.get("result", {}) or (
                first_completed.args[2] if len(first_completed.args) > 2 else {}
            )
            assert "explanation_summary" in result_data
