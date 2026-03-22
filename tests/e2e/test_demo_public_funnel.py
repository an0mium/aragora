"""Integration tests for the truthful public demo path.

Covers:
  POST /api/v1/playground/debate with source="demo"
  GET  /api/v1/debates/public/{id}
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_http_handler(body: dict, client_ip: str = "10.0.0.1") -> MagicMock:
    raw = json.dumps(body).encode()
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = MagicMock()
    handler.headers.get = lambda key, default="": {
        "Content-Length": str(len(raw)),
        "Content-Type": "application/json",
    }.get(key, default)
    handler.rfile = MagicMock()
    handler.rfile.read = MagicMock(return_value=raw)
    return handler


@pytest.fixture()
def _shared_debate_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    import aragora.storage.debate_store as mod

    monkeypatch.setattr(mod, "_store", None)
    return mod


@pytest.fixture()
def playground_handler(_shared_debate_store):
    from aragora.server.handlers.playground import PlaygroundHandler, _reset_rate_limits

    _reset_rate_limits()
    yield PlaygroundHandler()
    _reset_rate_limits()


@pytest.fixture()
def public_viewer_handler(_shared_debate_store):
    from aragora.server.handlers.debates.public_viewer import (
        PublicDebateViewerHandler,
        _reset_public_viewer_rate_limits,
    )

    _reset_public_viewer_rate_limits()
    yield PublicDebateViewerHandler()
    _reset_public_viewer_rate_limits()


def test_demo_source_produces_shareable_result(playground_handler):
    handler = _make_http_handler(
        {
            "topic": "Should we adopt AI code review as a mandatory CI step?",
            "question": "Should we adopt AI code review as a mandatory CI step?",
            "source": "demo",
        }
    )

    result = playground_handler.handle_post("/api/v1/playground/debate", {}, handler)

    assert result is not None
    assert result.status_code == 200

    body = json.loads(result.body.decode("utf-8"))
    assert body["source"] == "demo"
    assert body["share_token"] == body["id"]
    assert body["share_url"] == f"/debate/{body['id']}"


def test_demo_source_is_publicly_viewable(playground_handler, public_viewer_handler):
    handler = _make_http_handler(
        {
            "topic": "Should we standardize AI code review for sensitive services?",
            "question": "Should we standardize AI code review for sensitive services?",
            "source": "demo",
        }
    )

    created = playground_handler.handle_post("/api/v1/playground/debate", {}, handler)
    assert created is not None
    assert created.status_code == 200

    body = json.loads(created.body.decode("utf-8"))
    debate_id = body["id"]

    viewer_result = public_viewer_handler.handle(
        f"/api/v1/debates/public/{debate_id}",
        {},
        MagicMock(client_address=("10.0.0.2", 23456)),
    )

    assert viewer_result is not None
    assert viewer_result.status_code == 200

    viewer_body = json.loads(viewer_result.body.decode("utf-8"))
    assert viewer_body["id"] == debate_id
    assert viewer_body["source"] == "demo"
