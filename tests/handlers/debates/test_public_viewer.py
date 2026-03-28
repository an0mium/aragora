"""Tests for the public debate viewer handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.debates.public_viewer import (
    PublicDebateViewerHandler,
    _reset_public_viewer_rate_limits,
)


def _make_http_handler(client_ip: str = "10.0.0.1") -> MagicMock:
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    return handler


def _parse_body(result) -> dict:
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture(autouse=True)
def reset_rate_limits():
    _reset_public_viewer_rate_limits()
    yield
    _reset_public_viewer_rate_limits()


def test_public_viewer_reads_shared_debate_from_primary_storage():
    storage = MagicMock()
    storage.is_public.return_value = True
    storage.get_debate.return_value = {
        "id": "debate-123",
        "task": "Should we ship the feature?",
        "status": "concluded",
        "agents": ["analyst", "critic"],
        "messages": [
            {
                "agent": "analyst",
                "role": "proposer",
                "content": "Ship the feature behind a flag.",
                "round": 1,
            },
            {
                "agent": "critic",
                "role": "critic",
                "content": "Do not ship without a rollback plan.",
                "round": 1,
            },
        ],
        "votes": [
            {
                "agent": "analyst",
                "choice": "ship-behind-flag",
                "confidence": 0.91,
                "reasoning": "The rollback path is cheap.",
            }
        ],
        "consensus_proof": {
            "reached": True,
            "confidence": 0.84,
            "final_answer": "Ship behind a feature flag with a rollback plan.",
        },
    }

    handler = PublicDebateViewerHandler(ctx={"storage": storage})
    result = handler.handle(
        "/api/v1/debates/public/debate-123",
        {},
        _make_http_handler(),
    )

    assert result is not None
    assert result.status_code == 200
    body = _parse_body(result)
    assert body["id"] == "debate-123"
    assert body["topic"] == "Should we ship the feature?"
    assert body["participants"] == ["analyst", "critic"]
    assert body["proposals"] == {"analyst": "Ship the feature behind a flag."}
    assert body["final_answer"] == "Ship behind a feature flag with a rollback plan."
    assert body["messages"][1]["content"] == "Do not ship without a rollback plan."
    assert body["is_public"] is True


def test_public_viewer_rejects_private_primary_storage_debate():
    storage = MagicMock()
    storage.is_public.return_value = False
    storage.get_debate.return_value = {
        "id": "debate-456",
        "task": "Private debate",
    }

    handler = PublicDebateViewerHandler(ctx={"storage": storage})
    result = handler.handle(
        "/api/v1/debates/public/debate-456",
        {},
        _make_http_handler(),
    )

    assert result is not None
    assert result.status_code == 404
