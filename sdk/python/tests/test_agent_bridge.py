"""Tests for agent-bridge namespace route mappings."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.agent_bridge import AgentBridgeAPI, AsyncAgentBridgeAPI


def test_agent_bridge_list_runs_uses_cursor_contract() -> None:
    with patch.object(AragoraClient, "request") as mock_request:
        mock_request.return_value = {"schema_version": 1, "runs": []}
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
        agent_bridge = AgentBridgeAPI(client)

        agent_bridge.list_runs(limit=10, cursor="run:abc")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/agent-bridge/runs",
            params={"limit": 10, "cursor": "run:abc"},
        )
        client.close()


def test_agent_bridge_start_and_dispatch_routes() -> None:
    with patch.object(AragoraClient, "request") as mock_request:
        mock_request.return_value = {"schema_version": 1, "run_id": "bridge-1"}
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
        agent_bridge = AgentBridgeAPI(client)

        agent_bridge.start_run(
            task="Coordinate review",
            actors=[{"role": "implementer", "harness": "codex"}],
            run_id="bridge-1",
            next_actor="implementer",
            repair_budget_per_turn=2,
        )
        agent_bridge.dispatch_turn("bridge-1", role="implementer", prompt="Proceed")
        agent_bridge.auto_step("bridge-1", context_turns=3)

        assert mock_request.call_args_list[0].args == ("POST", "/api/v1/agent-bridge/runs")
        assert mock_request.call_args_list[0].kwargs["json"] == {
            "task": "Coordinate review",
            "actors": [{"role": "implementer", "harness": "codex"}],
            "run_id": "bridge-1",
            "next_actor": "implementer",
            "repair_budget_per_turn": 2,
        }
        assert mock_request.call_args_list[1].args == (
            "POST",
            "/api/v1/agent-bridge/runs/bridge-1/dispatch",
        )
        assert mock_request.call_args_list[1].kwargs["json"] == {
            "role": "implementer",
            "prompt": "Proceed",
        }
        assert mock_request.call_args_list[2].args == (
            "POST",
            "/api/v1/agent-bridge/runs/bridge-1/auto-step",
        )
        assert mock_request.call_args_list[2].kwargs["json"] == {"context_turns": 3}
        client.close()


@pytest.mark.asyncio
async def test_async_agent_bridge_list_runs_uses_cursor_contract() -> None:
    with patch.object(AragoraAsyncClient, "request") as mock_request:
        mock_request.return_value = {"schema_version": 1, "runs": []}
        async with AragoraAsyncClient(
            base_url="https://api.aragora.ai", api_key="test-key"
        ) as client:
            agent_bridge = AsyncAgentBridgeAPI(client)

            await agent_bridge.list_runs(limit=20, cursor="run:def")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agent-bridge/runs",
                params={"limit": 20, "cursor": "run:def"},
            )


@pytest.mark.asyncio
async def test_async_agent_bridge_start_and_dispatch_routes() -> None:
    with patch.object(AragoraAsyncClient, "request") as mock_request:
        mock_request.return_value = {"schema_version": 1, "run_id": "bridge-1"}
        async with AragoraAsyncClient(
            base_url="https://api.aragora.ai", api_key="test-key"
        ) as client:
            agent_bridge = AsyncAgentBridgeAPI(client)

            await agent_bridge.start_run(
                task="Coordinate review",
                actors=[{"role": "implementer", "harness": "codex"}],
            )
            await agent_bridge.dispatch_turn("bridge-1", role="implementer", prompt="Proceed")
            await agent_bridge.auto_step("bridge-1", prompt="Continue")

            assert mock_request.call_args_list[0].args == (
                "POST",
                "/api/v1/agent-bridge/runs",
            )
            assert mock_request.call_args_list[0].kwargs["json"] == {
                "task": "Coordinate review",
                "actors": [{"role": "implementer", "harness": "codex"}],
            }
            assert mock_request.call_args_list[1].args == (
                "POST",
                "/api/v1/agent-bridge/runs/bridge-1/dispatch",
            )
            assert mock_request.call_args_list[2].args == (
                "POST",
                "/api/v1/agent-bridge/runs/bridge-1/auto-step",
            )
            assert mock_request.call_args_list[2].kwargs["json"] == {"prompt": "Continue"}
