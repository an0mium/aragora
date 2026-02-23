"""Tests for cost fields in debate results (Task 14C).

Validates that:
- total_cost_usd and per_agent_cost appear in status update
- Zero cost handled correctly
- per_agent_cost dict format correct
- Missing cost fields default gracefully
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def controller():
    """Create a DebateController with mocked dependencies."""
    from aragora.server.debate_controller import DebateController

    return DebateController(
        factory=MagicMock(),
        emitter=MagicMock(),
        storage=MagicMock(),
    )


def _make_result(**overrides):
    """Create a mock debate result with sensible defaults."""
    result = MagicMock()
    result.final_answer = "The answer is 42."
    result.consensus_reached = True
    result.confidence = 0.9
    result.status = "completed"
    result.agent_failures = []
    result.participants = ["claude", "gpt4"]
    result.grounded_verdict = None
    result.total_cost_usd = overrides.get("total_cost_usd", 0.15)
    result.per_agent_cost = overrides.get("per_agent_cost", {"claude": 0.08, "gpt4": 0.07})
    result.explanation = None
    result.messages = []
    result.winner = None
    return result


def _make_config():
    """Create a mock debate config."""
    config = MagicMock()
    config.question = "Test question"
    config.agents_str = "claude,gpt4"
    config.rounds = 3
    config.metadata = {}
    config.debate_format = "full"
    config.debate_id = "test-cost-1"
    return config


class TestCostInDebateResults:
    """Tests for cost fields in the debate status update."""

    def test_cost_fields_in_status_update(self, controller):
        """total_cost_usd and per_agent_cost should appear in status update."""
        result = _make_result(
            total_cost_usd=0.15,
            per_agent_cost={"claude": 0.08, "gpt4": 0.07},
        )
        config = _make_config()

        with (
            patch("aragora.server.debate_controller.run_async", return_value=result),
            patch("aragora.server.debate_controller.update_debate_status") as mock_update,
            patch("aragora.server.debate_controller.create_arena_hooks"),
            patch("aragora.server.debate_controller.wrap_agent_for_streaming"),
            patch("aragora.storage.receipt_store.get_receipt_store"),
        ):
            controller.factory.create_arena.return_value = MagicMock()
            controller.factory.create_arena.return_value.protocol = MagicMock()
            controller.factory.create_arena.return_value.protocol.timeout_seconds = 0

            controller._run_debate(config, "test-cost-1")

            # Find the 'completed' status update
            completed_calls = [
                c
                for c in mock_update.call_args_list
                if len(c.args) >= 2 and c.args[1] == "completed"
            ]
            assert len(completed_calls) >= 1
            result_data = completed_calls[0].kwargs.get("result", {})
            assert result_data["total_cost_usd"] == 0.15
            assert result_data["per_agent_cost"] == {"claude": 0.08, "gpt4": 0.07}

    def test_zero_cost_handled(self, controller):
        """Zero cost should be included correctly, not omitted."""
        result = _make_result(total_cost_usd=0.0, per_agent_cost={})
        config = _make_config()

        with (
            patch("aragora.server.debate_controller.run_async", return_value=result),
            patch("aragora.server.debate_controller.update_debate_status") as mock_update,
            patch("aragora.server.debate_controller.create_arena_hooks"),
            patch("aragora.server.debate_controller.wrap_agent_for_streaming"),
            patch("aragora.storage.receipt_store.get_receipt_store"),
        ):
            controller.factory.create_arena.return_value = MagicMock()
            controller.factory.create_arena.return_value.protocol = MagicMock()
            controller.factory.create_arena.return_value.protocol.timeout_seconds = 0

            controller._run_debate(config, "test-zero-cost")

            completed_calls = [
                c
                for c in mock_update.call_args_list
                if len(c.args) >= 2 and c.args[1] == "completed"
            ]
            assert len(completed_calls) >= 1
            result_data = completed_calls[0].kwargs.get("result", {})
            assert result_data["total_cost_usd"] == 0.0
            assert result_data["per_agent_cost"] == {}

    def test_per_agent_cost_dict_format(self, controller):
        """per_agent_cost should be a dict mapping agent name to cost."""
        result = _make_result(per_agent_cost={"claude": 0.08, "gpt4": 0.07, "mistral": 0.03})
        config = _make_config()

        with (
            patch("aragora.server.debate_controller.run_async", return_value=result),
            patch("aragora.server.debate_controller.update_debate_status") as mock_update,
            patch("aragora.server.debate_controller.create_arena_hooks"),
            patch("aragora.server.debate_controller.wrap_agent_for_streaming"),
            patch("aragora.storage.receipt_store.get_receipt_store"),
        ):
            controller.factory.create_arena.return_value = MagicMock()
            controller.factory.create_arena.return_value.protocol = MagicMock()
            controller.factory.create_arena.return_value.protocol.timeout_seconds = 0

            controller._run_debate(config, "test-per-agent")

            completed_calls = [
                c
                for c in mock_update.call_args_list
                if len(c.args) >= 2 and c.args[1] == "completed"
            ]
            result_data = completed_calls[0].kwargs.get("result", {})
            pac = result_data["per_agent_cost"]
            assert isinstance(pac, dict)
            assert len(pac) == 3
            assert pac["claude"] == 0.08

    def test_missing_cost_fields_default_gracefully(self, controller):
        """Missing cost attributes on result should default to 0.0 and {}."""
        result = MagicMock(spec=[])  # No attributes at all
        result.final_answer = "Answer"
        result.consensus_reached = True
        result.confidence = 0.8
        result.status = "completed"
        result.agent_failures = []
        result.participants = ["a"]
        result.grounded_verdict = None
        result.messages = []
        result.winner = None
        # Deliberately do NOT set total_cost_usd or per_agent_cost
        # Use a side_effect to raise AttributeError for missing attrs
        del result.total_cost_usd
        del result.per_agent_cost
        del result.explanation

        config = _make_config()

        with (
            patch("aragora.server.debate_controller.run_async", return_value=result),
            patch("aragora.server.debate_controller.update_debate_status") as mock_update,
            patch("aragora.server.debate_controller.create_arena_hooks"),
            patch("aragora.server.debate_controller.wrap_agent_for_streaming"),
            patch("aragora.storage.receipt_store.get_receipt_store"),
        ):
            controller.factory.create_arena.return_value = MagicMock()
            controller.factory.create_arena.return_value.protocol = MagicMock()
            controller.factory.create_arena.return_value.protocol.timeout_seconds = 0

            controller._run_debate(config, "test-missing")

            completed_calls = [
                c
                for c in mock_update.call_args_list
                if len(c.args) >= 2 and c.args[1] == "completed"
            ]
            assert len(completed_calls) >= 1
            result_data = completed_calls[0].kwargs.get("result", {})
            # getattr with default should provide 0.0 and {}
            assert result_data["total_cost_usd"] == 0.0
            assert result_data["per_agent_cost"] == {}
