"""
Tests for Cost Preview in Debate Creation Flow (5B).

Tests cover:
- Cost estimation included in deliberation responses
- dry_run mode returns estimate only without execution
- Cost estimate accuracy with different agent/round counts
- OrchestrationRequest dry_run field parsing
- Sync and async deliberation both include cost
- Error handling for cost estimation failures
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.orchestration.models import OrchestrationRequest
from aragora.server.handlers.orchestration.handler import OrchestrationHandler
from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create an OrchestrationHandler with empty context."""
    return OrchestrationHandler({})


@pytest.fixture
def base_request_data():
    """Base request data for deliberation."""
    return {
        "question": "Should we migrate to microservices?",
        "agents": ["anthropic-api", "openai-api", "gemini"],
        "max_rounds": 3,
    }


def _parse_response(result) -> dict[str, Any]:
    """Parse HandlerResult body as JSON."""
    if isinstance(result.body, bytes):
        return json.loads(result.body.decode("utf-8"))
    return json.loads(result.body)


# ===========================================================================
# OrchestrationRequest dry_run field
# ===========================================================================


class TestOrchestrationRequestDryRun:
    """Test dry_run field on OrchestrationRequest."""

    def test_default_dry_run_is_false(self):
        req = OrchestrationRequest.from_dict({"question": "test"})
        assert req.dry_run is False

    def test_dry_run_true_from_dict(self):
        req = OrchestrationRequest.from_dict({"question": "test", "dry_run": True})
        assert req.dry_run is True

    def test_dry_run_false_from_dict(self):
        req = OrchestrationRequest.from_dict({"question": "test", "dry_run": False})
        assert req.dry_run is False

    def test_dry_run_preserved_in_request(self, base_request_data):
        base_request_data["dry_run"] = True
        req = OrchestrationRequest.from_dict(base_request_data)
        assert req.dry_run is True
        assert req.question == "Should we migrate to microservices?"


# ===========================================================================
# estimate_debate_cost unit tests
# ===========================================================================


class TestEstimateDebateCost:
    """Test the estimate_debate_cost function directly."""

    def test_default_estimate(self):
        result = estimate_debate_cost()
        assert "total_estimated_cost_usd" in result
        assert "breakdown_by_model" in result
        assert "assumptions" in result
        assert result["num_agents"] == 3
        assert result["num_rounds"] == 9

    def test_custom_agents_and_rounds(self):
        result = estimate_debate_cost(num_agents=2, num_rounds=3)
        assert result["num_agents"] == 2
        assert result["num_rounds"] == 3
        assert len(result["breakdown_by_model"]) == 2

    def test_cost_increases_with_agents(self):
        cost_2 = estimate_debate_cost(num_agents=2, num_rounds=3)
        cost_4 = estimate_debate_cost(num_agents=4, num_rounds=3)
        assert cost_4["total_estimated_cost_usd"] > cost_2["total_estimated_cost_usd"]

    def test_cost_increases_with_rounds(self):
        cost_3 = estimate_debate_cost(num_agents=3, num_rounds=3)
        cost_9 = estimate_debate_cost(num_agents=3, num_rounds=9)
        assert cost_9["total_estimated_cost_usd"] > cost_3["total_estimated_cost_usd"]

    def test_breakdown_has_required_fields(self):
        result = estimate_debate_cost(num_agents=1, num_rounds=1)
        entry = result["breakdown_by_model"][0]
        assert "model" in entry
        assert "provider" in entry
        assert "estimated_input_tokens" in entry
        assert "estimated_output_tokens" in entry
        assert "subtotal_usd" in entry


# ===========================================================================
# Dry-run mode tests (via handler)
# ===========================================================================


class TestDryRunMode:
    """Test dry_run mode returns cost estimate without executing."""

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    async def test_dry_run_returns_estimate_only(
        self, mock_auth, mock_perm, handler, base_request_data
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")
        base_request_data["dry_run"] = True

        result = handler._handle_deliberate(
            base_request_data, MagicMock(), MagicMock(), sync=False
        )
        body = _parse_response(result)

        assert body["dry_run"] is True
        assert "estimated_cost" in body
        assert "total_estimated_cost_usd" in body["estimated_cost"]
        assert body["agents"] == ["anthropic-api", "openai-api", "gemini"]
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    async def test_dry_run_does_not_execute_debate(
        self, mock_auth, mock_perm, handler, base_request_data
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")
        base_request_data["dry_run"] = True

        with patch.object(handler, "_execute_deliberation") as mock_exec:
            result = handler._handle_deliberate(
                base_request_data, MagicMock(), MagicMock(), sync=True
            )
            mock_exec.assert_not_called()
            body = _parse_response(result)
            assert body["dry_run"] is True

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    async def test_dry_run_includes_request_id(
        self, mock_auth, mock_perm, handler, base_request_data
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")
        base_request_data["dry_run"] = True

        result = handler._handle_deliberate(
            base_request_data, MagicMock(), MagicMock(), sync=False
        )
        body = _parse_response(result)
        assert "request_id" in body

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    async def test_dry_run_max_rounds_in_response(
        self, mock_auth, mock_perm, handler, base_request_data
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")
        base_request_data["dry_run"] = True
        base_request_data["max_rounds"] = 5

        result = handler._handle_deliberate(
            base_request_data, MagicMock(), MagicMock(), sync=False
        )
        body = _parse_response(result)
        assert body["max_rounds"] == 5


# ===========================================================================
# Cost estimate in non-dry-run responses
# ===========================================================================


class TestCostInDeliberationResponse:
    """Test that cost estimate is included in normal deliberation responses."""

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    @patch("aragora.server.handlers.orchestration.handler.asyncio")
    async def test_async_response_includes_estimated_cost(
        self, mock_asyncio, mock_auth, mock_perm, handler, base_request_data
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")
        mock_asyncio.create_task = MagicMock()

        result = handler._handle_deliberate(
            base_request_data, MagicMock(), MagicMock(), sync=False
        )
        body = _parse_response(result)

        assert "estimated_cost_usd" in body
        assert isinstance(body["estimated_cost_usd"], float)
        assert result.status_code == 202

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    @patch("aragora.server.handlers.orchestration.handler.run_async")
    async def test_sync_response_includes_estimated_cost(
        self, mock_run_async, mock_auth, mock_perm, handler, base_request_data
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")

        # Create a mock OrchestrationResult
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "request_id": "test-123",
            "success": True,
            "consensus_reached": True,
        }
        mock_run_async.return_value = mock_result

        result = handler._handle_deliberate(
            base_request_data, MagicMock(), MagicMock(), sync=True
        )
        body = _parse_response(result)

        assert "estimated_cost_usd" in body
        assert isinstance(body["estimated_cost_usd"], float)

    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.check_permission"
    )
    @patch(
        "aragora.server.handlers.orchestration.handler.OrchestrationHandler.get_auth_context",
        new_callable=AsyncMock,
    )
    @patch("aragora.server.handlers.orchestration.handler.asyncio")
    async def test_cost_scales_with_agent_count(
        self, mock_asyncio, mock_auth, mock_perm, handler
    ):
        mock_auth.return_value = MagicMock(user_id="test-user")
        mock_asyncio.create_task = MagicMock()

        # 2 agents
        data_2 = {"question": "test", "agents": ["a", "b"], "max_rounds": 3}
        result_2 = handler._handle_deliberate(data_2, MagicMock(), MagicMock(), sync=False)
        cost_2 = _parse_response(result_2)["estimated_cost_usd"]

        # 4 agents
        data_4 = {"question": "test", "agents": ["a", "b", "c", "d"], "max_rounds": 3}
        result_4 = handler._handle_deliberate(data_4, MagicMock(), MagicMock(), sync=False)
        cost_4 = _parse_response(result_4)["estimated_cost_usd"]

        assert cost_4 > cost_2


# ===========================================================================
# Error handling
# ===========================================================================


class TestCostPreviewErrors:
    """Test error handling in cost preview."""

    def test_empty_question_returns_error(self, handler):
        data = {"question": "", "dry_run": True}
        result = handler._handle_deliberate(data, MagicMock(), MagicMock(), sync=False)
        assert result.status_code == 400

    def test_estimate_cost_with_single_agent(self):
        result = estimate_debate_cost(num_agents=1, num_rounds=1)
        assert result["total_estimated_cost_usd"] > 0
        assert len(result["breakdown_by_model"]) == 1

    def test_estimate_cost_with_max_agents(self):
        result = estimate_debate_cost(num_agents=8, num_rounds=12)
        assert result["total_estimated_cost_usd"] > 0
        assert len(result["breakdown_by_model"]) == 8
