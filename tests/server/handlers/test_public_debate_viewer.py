"""Tests for the public debate viewer handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debates.public_viewer import (
    PublicDebateViewerHandler,
    _reset_public_viewer_rate_limits,
)
from aragora.server.handlers.debates.share import _reset_share_state, set_public_spectate


def _make_http_handler(client_ip: str = "10.0.0.1") -> MagicMock:
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    return handler


@pytest.fixture(autouse=True)
def reset_public_state():
    _reset_public_viewer_rate_limits()
    _reset_share_state()
    yield
    _reset_public_viewer_rate_limits()
    _reset_share_state()


def test_reads_public_debate_from_primary_storage_for_generic_ids():
    playground_store = MagicMock()
    playground_store.get.return_value = None

    storage = MagicMock()
    storage.get.return_value = {
        "id": "debate-123",
        "task": "Should archived debates be link-shareable?",
        "status": "completed",
    }
    storage.is_public.return_value = True

    with patch("aragora.storage.debate_store.get_debate_store", return_value=playground_store):
        with patch("aragora.server.storage.get_debates_db", return_value=storage):
            result = PublicDebateViewerHandler().handle(
                "/api/v1/debates/public/debate-123",
                {},
                _make_http_handler(),
            )

    assert result is not None
    assert result.status_code == 200
    body = json.loads(result.body.decode("utf-8"))
    assert body["id"] == "debate-123"
    assert body["task"] == "Should archived debates be link-shareable?"
    assert body["visibility"] == "public"
    assert body["share_url"] == "/debate/debate-123"


def test_reads_immediately_shared_debate_from_primary_storage():
    playground_store = MagicMock()
    playground_store.get.return_value = None

    storage = MagicMock()
    storage.get.return_value = {
        "id": "debate-124",
        "task": "Should the share link work before the next persistence cycle?",
        "status": "completed",
        "visibility": "private",
    }
    storage.is_public.return_value = False
    set_public_spectate("debate-124", True)

    with patch("aragora.storage.debate_store.get_debate_store", return_value=playground_store):
        with patch("aragora.server.storage.get_debates_db", return_value=storage):
            result = PublicDebateViewerHandler().handle(
                "/api/v1/debates/public/debate-124",
                {},
                _make_http_handler(),
            )

    assert result is not None
    assert result.status_code == 200
    body = json.loads(result.body.decode("utf-8"))
    assert body["id"] == "debate-124"
    assert body["visibility"] == "public"
    assert body["share_url"] == "/debate/debate-124"
