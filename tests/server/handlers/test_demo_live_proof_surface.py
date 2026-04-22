from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch

import pytest

from aragora.server.handlers.base import error_response
from aragora.server.handlers.playground import (
    PlaygroundHandler,
    _reset_oracle_sessions,
    _reset_rate_limits,
)
from aragora.storage.debate_store import get_debate_store, normalize_cache_key


class _MockHeaders:
    def __init__(self, raw_len: int):
        self._data = {
            "Content-Type": "application/json",
            "Content-Length": str(raw_len),
        }

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)


def _make_http_handler(body: dict[str, Any], client_ip: str = "10.0.0.1"):
    raw = json.dumps(body).encode()
    handler = type("MockHandler", (), {})()
    handler.client_address = (client_ip, 12345)
    handler.headers = _MockHeaders(len(raw))
    handler.rfile = io.BytesIO(raw)
    return handler


def _parse_result(result) -> tuple[dict[str, Any], int]:
    assert result is not None
    return json.loads(result.body), result.status_code


@pytest.fixture(autouse=True)
def _clean_rate_limits():
    _reset_rate_limits()
    _reset_oracle_sessions()
    yield
    _reset_rate_limits()
    _reset_oracle_sessions()


@pytest.fixture()
def handler(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    import aragora.storage.debate_store as debate_store_mod

    monkeypatch.setattr(debate_store_mod, "_store", None)
    return PlaygroundHandler({})


def _live_result(debate_id: str) -> dict[str, Any]:
    return {
        "id": debate_id,
        "topic": "Should we require AI code review in CI?",
        "status": "completed",
        "rounds_used": 1,
        "consensus_reached": True,
        "confidence": 0.74,
        "verdict": "needs_review",
        "duration_seconds": 2.4,
        "participants": ["claude", "gpt"],
        "proposals": {
            "claude": "Treat AI review as a structured advisory layer.",
            "gpt": "Gate only high-risk paths while false-positive rates stabilize.",
        },
        "critiques": [],
        "votes": [],
        "dissenting_views": [],
        "final_answer": "Adopt tiered enforcement and measure it.",
        "is_live": True,
        "receipt_hash": "abc123",
    }


@patch("aragora.storage.debate_store.DebateResultStore.get_by_cache_key", return_value=None)
def test_demo_source_requires_live_proof_when_backend_cannot_deliver(
    _mock_cache,
    handler,
):
    request = _make_http_handler(
        {
            "topic": "Should we require AI code review in CI?",
            "question": "Should we require AI code review in CI?",
            "source": "demo",
        }
    )

    with (
        patch("aragora.server.handlers.playground._try_oracle_tentacles", return_value=None),
        patch.object(
            handler,
            "_run_live_debate",
            return_value=error_response("Live playground unavailable", 503),
        ),
    ):
        result = handler.handle_post("/api/v1/playground/debate", {}, request)

    body, status = _parse_result(result)
    assert status == 503
    assert body["code"] == "live_demo_unavailable"
    assert body["show_recorded_sample"] is True
    assert body["is_live"] is False


def test_demo_source_skips_cached_results_without_live_provenance(handler):
    request = _make_http_handler(
        {
            "topic": "Should we require AI code review in CI?",
            "question": "Should we require AI code review in CI?",
            "source": "demo",
        }
    )
    cached_mock = {
        "id": "cached-fallback",
        "topic": "Should we require AI code review in CI?",
        "status": "completed",
        "participants": ["analyst", "critic"],
        "proposals": {"analyst": "Mock answer"},
        "final_answer": "Mock answer",
    }

    with (
        patch(
            "aragora.storage.debate_store.DebateResultStore.get_by_cache_key",
            return_value=cached_mock,
        ),
        patch(
            "aragora.server.handlers.playground._try_oracle_tentacles",
            return_value=_live_result("fresh-live-result"),
        ) as mock_tentacles,
    ):
        result = handler.handle_post("/api/v1/playground/debate", {}, request)

    body, status = _parse_result(result)
    assert status == 200
    assert body["id"] == "fresh-live-result"
    assert body["is_live"] is True
    assert body.get("cached") is not True
    mock_tentacles.assert_called_once()


def test_demo_source_can_replay_cached_live_results(handler):
    request = _make_http_handler(
        {
            "topic": "Should we require AI code review in CI?",
            "question": "Should we require AI code review in CI?",
            "source": "demo",
        }
    )
    cached_live = _live_result("cached-live-result")

    with (
        patch(
            "aragora.storage.debate_store.DebateResultStore.get_by_cache_key",
            return_value=cached_live,
        ),
        patch("aragora.server.handlers.playground._try_oracle_tentacles") as mock_tentacles,
    ):
        result = handler.handle_post("/api/v1/playground/debate", {}, request)

    body, status = _parse_result(result)
    assert status == 200
    assert body["id"] == "cached-live-result"
    assert body["is_live"] is True
    assert body["cached"] is True
    mock_tentacles.assert_not_called()


def test_demo_source_can_replay_cached_live_results_without_live_agents(handler):
    request = _make_http_handler(
        {
            "topic": "Should we require AI code review in CI?",
            "question": "Should we require AI code review in CI?",
            "source": "demo",
        }
    )
    cached_live = _live_result("cached-live-result")
    store = get_debate_store()
    store.save(cached_live["id"], cached_live["topic"], cached_live)

    model_ids = ["anthropic/claude-sonnet-4", "openai/gpt-4o", "google/gemini-2.0-flash-001"]
    cache_key = normalize_cache_key(cached_live["topic"], model_ids, 2)
    store.save_cache_index(
        cache_key=cache_key,
        debate_id=cached_live["id"],
        topic_normalized=cached_live["topic"].strip().lower(),
        model_ids="|".join(sorted(model_ids)),
        rounds=2,
    )

    with (
        patch("aragora.server.handlers.playground._get_available_live_agents", return_value=[]),
        patch("aragora.server.handlers.playground._try_oracle_tentacles") as mock_tentacles,
    ):
        result = handler.handle_post("/api/v1/playground/debate", {}, request)

    body, status = _parse_result(result)
    assert status == 200
    assert body["id"] == "cached-live-result"
    assert body["is_live"] is True
    assert body["cached"] is True
    mock_tentacles.assert_not_called()


@patch("aragora.storage.debate_store.DebateResultStore.get_by_cache_key", return_value=None)
def test_try_source_keeps_shareable_beta_fallbacks(
    _mock_cache,
    handler,
):
    request = _make_http_handler(
        {
            "topic": "Should we require AI code review in CI?",
            "question": "Should we require AI code review in CI?",
            "source": "try",
        }
    )

    with (
        patch("aragora.server.handlers.playground._try_oracle_tentacles", return_value=None),
        patch.object(
            handler,
            "_run_live_debate",
            return_value=error_response("Live playground unavailable", 503),
        ),
    ):
        result = handler.handle_post("/api/v1/playground/debate", {}, request)

    body, status = _parse_result(result)
    assert status == 200
    assert body["source"] == "try"
    assert body["is_live"] is False
    assert body["mock_fallback"] is True
    assert body["share_token"] == body["id"]
    assert body["share_url"] == f"/debate/{body['id']}"
