"""Tests for the public debate viewer handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debates.public_viewer import (
    PublicDebateViewerHandler,
    _reset_public_viewer_rate_limits,
)


def _body(result) -> dict:
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture(autouse=True)
def reset_rate_limits():
    _reset_public_viewer_rate_limits()
    yield
    _reset_public_viewer_rate_limits()


def _make_http_handler() -> MagicMock:
    handler = MagicMock()
    handler.client_address = ("10.0.0.1", 12345)
    return handler


class TestPublicViewerRouting:
    def test_accepts_standard_debate_ids(self):
        handler = PublicDebateViewerHandler()
        assert handler.can_handle("/api/v1/debates/public/debate-123")

    def test_accepts_uuid_debate_ids(self):
        handler = PublicDebateViewerHandler()
        assert handler.can_handle("/api/v1/debates/public/550e8400-e29b-41d4-a716-446655440000")


class TestPublicViewerStorageFallback:
    @patch("aragora.server.handlers.debates.public_viewer._get_debate_result", return_value=None)
    def test_loads_public_debate_from_primary_storage(self, _mock_store_lookup):
        storage = MagicMock()
        storage.is_public.return_value = True
        storage.get_debate.return_value = {
            "id": "debate-123",
            "question": "Should we publish the debate permalink?",
            "status": "completed",
            "messages": [
                {
                    "agent": "analyst",
                    "role": "proposal",
                    "content": "Yes, once the public page works.",
                    "round": 1,
                }
            ],
            "result": {
                "consensus_reached": True,
                "confidence": 0.91,
                "final_answer": "Publish it after enabling anonymous access.",
                "participants": ["analyst", "critic"],
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
        body = _body(result)
        assert body["id"] == "debate-123"
        storage.is_public.assert_called_once_with("debate-123")
        storage.get_debate.assert_called_once_with("debate-123")

    @patch("aragora.server.handlers.debates.public_viewer._get_debate_result", return_value=None)
    def test_returns_404_for_private_primary_storage_debate(self, _mock_store_lookup):
        storage = MagicMock()
        storage.is_public.return_value = False
        storage.get_debate.return_value = {
            "id": "debate-123",
            "question": "Should we expose this debate?",
        }
        handler = PublicDebateViewerHandler(ctx={"storage": storage})

        result = handler.handle(
            "/api/v1/debates/public/debate-123",
            {},
            _make_http_handler(),
        )

        assert result is not None
        assert result.status_code == 404

    def test_rejects_invalid_debate_ids(self):
        handler = PublicDebateViewerHandler()

        result = handler.handle(
            "/api/v1/debates/public/../../etc/passwd",
            {},
            _make_http_handler(),
        )

        assert result is not None
        assert result.status_code == 400
