"""Tests for DecisionsAPI request/response mapping."""

from __future__ import annotations

from typing import Any

import pytest

from aragora_client.decisions import (
    DecisionResult,
    DecisionsAPI,
    DecisionStatus,
    DecisionType,
)


class _DummyClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.last_path: str | None = None
        self.last_data: dict[str, Any] | None = None

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        self.last_path = path
        self.last_data = data
        return self.response


@pytest.mark.asyncio
async def test_create_payload_and_result_mapping() -> None:
    response = {
        "request_id": "req-123",
        "status": "completed",
        "result": {
            "decision_type": "workflow",
            "answer": "ok",
            "confidence": 0.81,
        },
    }
    client = _DummyClient(response)
    api = DecisionsAPI(client)

    result = await api.create(
        "Ship it?",
        decision_type=DecisionType.WORKFLOW,
        agents=["alpha", "beta"],
        max_rounds=3,
        consensus_threshold=0.6,
        workflow_id="wf-1",
        gauntlet_checks=["injection"],
        metadata={"team": "platform"},
        async_mode=False,
    )

    assert client.last_path == "/api/v1/decisions"
    assert client.last_data is not None
    assert client.last_data["content"] == "Ship it?"
    assert client.last_data["decision_type"] == "workflow"
    assert client.last_data["context"]["metadata"]["team"] == "platform"
    assert client.last_data["config"]["agents"] == ["alpha", "beta"]
    assert client.last_data["config"]["rounds"] == 3
    assert client.last_data["config"]["consensus_threshold"] == 0.6
    assert client.last_data["config"]["workflow_id"] == "wf-1"
    assert client.last_data["config"]["attack_categories"] == ["injection"]
    assert client.last_data["config"]["async"] is False

    assert result.request_id == "req-123"
    assert result.decision_type is DecisionType.WORKFLOW
    assert result.status is DecisionStatus.COMPLETED
    assert result.answer == "ok"
    assert result.task == "Ship it?"


def test_from_dict_handles_unknown_values() -> None:
    result = DecisionResult.from_dict(
        {
            "request_id": "req-unknown",
            "status": "mystery",
            "decision_type": "unknown",
        }
    )
    assert result.status is DecisionStatus.UNKNOWN
    assert result.decision_type is DecisionType.AUTO
