from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from aragora.server.handlers.base import error_response
from aragora.server.handlers.playground import (
    PlaygroundHandler,
    _reset_oracle_sessions,
    _reset_rate_limits,
    _try_oracle_response,
    start_playground_debate,
)
from aragora.spectate.stream import SpectatorStream
from aragora.spectate.ws_bridge import get_spectate_bridge, reset_spectate_bridge


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


@pytest.fixture()
def spectate_bridge():
    reset_spectate_bridge()
    bridge = get_spectate_bridge()
    bridge.start()
    try:
        yield bridge
    finally:
        bridge.stop()
        reset_spectate_bridge()


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


def test_demo_source_replay_emits_spectate_task_and_agents(handler, spectate_bridge):
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
    assert body["cached"] is True
    mock_tentacles.assert_not_called()

    events = spectate_bridge.get_recent_events(3)
    assert [event.event_type for event in events] == ["debate_start", "proposal", "consensus"]
    assert all(event.debate_id == cached_live["id"] for event in events)
    for event in events:
        assert event.data["task"] == cached_live["topic"]
        assert event.data["agents"] == cached_live["participants"]


def test_oracle_response_emits_spectate_task_and_agents(spectate_bridge):
    question = "Should we require AI code review in CI?"

    with (
        patch("aragora.server.handlers.playground._append_session_turn"),
        patch(
            "aragora.server.handlers.playground._call_llm",
            return_value="Direct answer from oracle.",
        ),
    ):
        result = _try_oracle_response(
            "consult",
            question,
            client_debate_id="oracle-spectate-test",
        )

    assert result is not None
    events = spectate_bridge.get_recent_events(3)
    assert [event.event_type for event in events] == ["debate_start", "proposal", "consensus"]
    assert all(event.debate_id == "oracle-spectate-test" for event in events)
    for event in events:
        assert event.data["task"] == question
        assert event.data["agents"] == ["oracle"]


def test_live_debate_binds_spectate_task_and_agents(spectate_bridge):
    question = "Should we require AI code review in CI?"

    class _FakeArena:
        async def run(self):
            SpectatorStream(enabled=True, output=io.StringIO(), format="plain").emit(
                "debate_start",
                agent="anthropic-api",
                details="Live debate underway",
            )
            return SimpleNamespace(
                status="completed",
                rounds_used=1,
                consensus_reached=True,
                confidence=0.82,
                verdict=SimpleNamespace(value="approved"),
                duration_seconds=1.2,
                participants=["anthropic-api", "openai-api"],
                proposals={
                    "anthropic-api": "A" * 90,
                    "openai-api": "B" * 90,
                },
                critiques=[],
                votes=[],
                dissenting_views=[],
                final_answer="C" * 90,
            )

    class _FakeFactory:
        def create_arena(self, _config):
            return _FakeArena()

    with (
        patch(
            "aragora.server.handlers.playground._get_available_live_agents",
            return_value=["anthropic-api", "openai-api"],
        ),
        patch("aragora.server.debate_factory.DebateFactory", return_value=_FakeFactory()),
        patch("aragora.server.debate_factory.DebateConfig", side_effect=lambda **kwargs: kwargs),
    ):
        result = start_playground_debate(
            question=question,
            agent_count=2,
            max_rounds=1,
            timeout=5,
            debate_id="playground-live-test",
        )

    assert result["participants"] == ["anthropic-api", "openai-api"]
    event = spectate_bridge.get_recent_events(1)[0]
    assert event.debate_id == "playground-live-test"
    assert event.data["task"] == question
    assert event.data["agents"] == ["anthropic-api", "openai-api"]


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
