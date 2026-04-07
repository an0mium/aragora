"""Tests: landing debate response must include share_token (Task B TDD)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import json_response
from aragora.server.handlers.playground import (
    PlaygroundHandler,
    _persist_playground_debate,
    _reset_oracle_sessions,
    _reset_rate_limits,
)
from aragora.storage.debate_store import get_debate_store


@pytest.fixture(autouse=True)
def _clean_rate_limits():
    _reset_rate_limits()
    _reset_oracle_sessions()
    yield
    _reset_rate_limits()
    _reset_oracle_sessions()


@pytest.fixture(autouse=True)
def _mock_live_debate():
    """Patch _run_live_debate so tests don't need real API keys."""

    def _side_effect(self, topic, rounds, agent_count):
        participants = [f"agent-{i}" for i in range(agent_count)]
        return json_response(
            {
                "id": f"playground_{uuid.uuid4().hex[:8]}",
                "topic": topic,
                "status": "completed",
                "rounds_used": 1,
                "consensus_reached": True,
                "confidence": 0.82,
                "verdict": "consensus",
                "duration_seconds": 1.5,
                "participants": participants,
                "proposals": {p: f"Position from {p}" for p in participants},
                "critiques": [],
                "votes": [],
                "dissenting_views": [],
                "final_answer": "Consensus reached.",
                "is_live": True,
                "receipt_preview": {},
                "upgrade_cta": {},
            }
        )

    with patch.object(PlaygroundHandler, "_run_live_debate", _side_effect):
        yield


@pytest.fixture()
def handler(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    import aragora.storage.debate_store as mod

    monkeypatch.setattr(mod, "_store", None)
    return PlaygroundHandler()


class TestShareTokenInResponse:
    def test_persist_and_respond_includes_share_token(self, handler):
        """_persist_and_respond must inject share_token alongside share_url."""
        debate_data = {
            "id": "abcdef1234567890",
            "topic": "Should we use Python 3.11?",
            "verdict": "approved",
        }
        original = json_response(debate_data)

        result = handler._persist_and_respond(original, "Should we use Python 3.11?", "landing")

        body = json.loads(result.body.decode("utf-8"))
        assert "share_token" in body, f"Missing share_token in response keys: {list(body.keys())}"
        assert body["share_token"] == "abcdef1234567890"

    def test_persist_and_respond_share_url_uses_debate_id(self, handler):
        """share_url must reference the debate ID so the public viewer can find it."""
        debate_data = {
            "id": "abcdef1234567890",
            "topic": "Test topic",
            "verdict": "approved",
        }
        original = json_response(debate_data)

        result = handler._persist_and_respond(original, "Test topic", "playground")

        body = json.loads(result.body.decode("utf-8"))
        assert "share_url" in body
        assert "abcdef1234567890" in body["share_url"]

    def test_no_share_token_when_no_id(self, handler):
        """When debate has no id, share_token should not be injected."""
        debate_data = {"topic": "No ID debate", "verdict": "approved"}
        original = json_response(debate_data)

        result = handler._persist_and_respond(original, "No ID debate", "mock")

        body = json.loads(result.body.decode("utf-8"))
        assert "share_token" not in body

    def test_post_debate_response_includes_share_token(self, handler):
        """POST /api/v1/playground/debate response must include share_token."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("10.0.0.1", 12345)
        payload = json.dumps(
            {"topic": "Should we standardize on GraphQL?", "source": "landing"}
        ).encode()
        mock_handler.headers = MagicMock()
        mock_handler.headers.get = lambda k, d="": {
            "Content-Length": str(len(payload)),
            "Content-Type": "application/json",
        }.get(k, d)
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read = MagicMock(return_value=payload)

        result = handler.handle_post("/api/v1/playground/debate", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "share_token" in body, (
            f"Missing share_token in POST /debate response. Keys: {list(body.keys())}"
        )

    def test_persist_and_respond_does_not_overwrite_existing_share_id(self, handler):
        """Colliding public debate IDs must mint a new share instead of replacing the old one."""
        original = {
            "id": "abcdef1234567890",
            "topic": "Original topic",
            "final_answer": "Original answer",
            "verdict": "approved",
        }
        attacker = {
            "id": "abcdef1234567890",
            "topic": "Attacker topic",
            "final_answer": "Overwrite me",
            "verdict": "approved",
        }

        handler._persist_and_respond(
            json_response(original),
            original["topic"],
            "landing",
        )

        result = handler._persist_and_respond(
            json_response(attacker),
            attacker["topic"],
            "landing",
        )
        body = json.loads(result.body.decode("utf-8"))
        store = get_debate_store()

        assert body["id"] != "abcdef1234567890"
        assert body["share_token"] == body["id"]
        assert store.get("abcdef1234567890")["final_answer"] == "Original answer"
        assert store.get(body["id"])["final_answer"] == "Overwrite me"

    def test_persist_playground_debate_does_not_reuse_existing_public_id(self, handler):
        """Repository-backed playground persistence must also preserve existing public shares."""
        existing = {
            "id": "fedcba0987654321",
            "topic": "Existing shared debate",
            "final_answer": "Keep this result",
            "verdict": "approved",
            "participants": [],
            "proposals": {},
        }
        _persist_playground_debate(existing)

        colliding = {
            "id": "fedcba0987654321",
            "topic": "Collision attempt",
            "final_answer": "New result",
            "verdict": "approved",
            "participants": [],
            "proposals": {},
        }
        _persist_playground_debate(colliding)
        store = get_debate_store()

        assert colliding["id"] != "fedcba0987654321"
        assert store.get("fedcba0987654321")["final_answer"] == "Keep this result"
        assert store.get(colliding["id"])["final_answer"] == "New result"
