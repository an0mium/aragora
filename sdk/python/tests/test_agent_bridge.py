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
