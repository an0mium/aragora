from __future__ import annotations

from unittest.mock import call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.mcp import AsyncMCPAPI
from aragora_sdk.namespaces.settlements import AsyncSettlementAPI


def test_sync_mcp_and_settlement_routes() -> None:
    with patch.object(AragoraClient, "request") as mock_request:
        mock_request.return_value = {"ok": True}
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

        client.mcp.list_tools(category="debate")
        client.mcp.get_tool("run_debate")
        client.settlements.list(debate_id="deb_123", domain="ops", limit=5)
        client.settlements.get_history(limit=7)
        client.settlements.get_summary()
        client.settlements.get("set_123")
        client.settlements.settle(
            "set_123",
            outcome="correct",
            evidence="verified",
            settled_by="codex",
        )
        client.settlements.settle_batch(
            [{"settlement_id": "set_123", "outcome": "incorrect", "evidence": "counterexample"}],
            settled_by="reviewer",
        )
        client.settlements.get_agent_accuracy("agent/demo")

        expected_calls = [
            call("GET", "/api/v1/mcp/tools", params={"category": "debate"}),
            call("GET", "/api/v1/mcp/tools/run_debate"),
            call(
                "GET",
                "/api/v1/settlements",
                params={"debate_id": "deb_123", "domain": "ops", "limit": 5},
            ),
            call("GET", "/api/v1/settlements/history", params={"limit": 7}),
            call("GET", "/api/v1/settlements/summary"),
            call("GET", "/api/v1/settlements/set_123"),
            call(
                "POST",
                "/api/v1/settlements/set_123/settle",
                json={
                    "outcome": "correct",
                    "evidence": "verified",
                    "settled_by": "codex",
                },
            ),
            call(
                "POST",
                "/api/v1/settlements/batch",
                json={
                    "settlements": [
                        {
                            "settlement_id": "set_123",
                            "outcome": "incorrect",
                            "evidence": "counterexample",
                        }
                    ],
                    "settled_by": "reviewer",
                },
            ),
            call("GET", "/api/v1/settlements/agent/agent/demo/accuracy"),
        ]
        mock_request.assert_has_calls(expected_calls)
        client.close()


@pytest.mark.asyncio
async def test_async_mcp_and_settlement_routes() -> None:
    with patch.object(AragoraAsyncClient, "request") as mock_request:
        mock_request.return_value = {"ok": True}
        async with AragoraAsyncClient(
            base_url="https://api.aragora.ai", api_key="test-key"
        ) as client:
            mcp = AsyncMCPAPI(client)
            settlements = AsyncSettlementAPI(client)

            await mcp.list_tools(category="debate")
            await mcp.get_tool("run_debate")
            await settlements.list(debate_id="deb_123", domain="ops", limit=5)
            await settlements.get_history(limit=7)
            await settlements.get_summary()
            await settlements.get("set_123")
            await settlements.settle(
                "set_123",
                outcome="partial",
                evidence="mixed",
                settled_by="codex",
            )
            await settlements.settle_batch(
                [{"settlement_id": "set_123", "outcome": "correct"}],
                settled_by="reviewer",
            )
            await settlements.get_agent_accuracy("agent/demo")

            expected_calls = [
                call("GET", "/api/v1/mcp/tools", params={"category": "debate"}),
                call("GET", "/api/v1/mcp/tools/run_debate"),
                call(
                    "GET",
                    "/api/v1/settlements",
                    params={"debate_id": "deb_123", "domain": "ops", "limit": 5},
                ),
                call("GET", "/api/v1/settlements/history", params={"limit": 7}),
                call("GET", "/api/v1/settlements/summary"),
                call("GET", "/api/v1/settlements/set_123"),
                call(
                    "POST",
                    "/api/v1/settlements/set_123/settle",
                    json={
                        "outcome": "partial",
                        "evidence": "mixed",
                        "settled_by": "codex",
                    },
                ),
                call(
                    "POST",
                    "/api/v1/settlements/batch",
                    json={
                        "settlements": [{"settlement_id": "set_123", "outcome": "correct"}],
                        "settled_by": "reviewer",
                    },
                ),
                call("GET", "/api/v1/settlements/agent/agent/demo/accuracy"),
            ]
            mock_request.assert_has_calls(expected_calls)
