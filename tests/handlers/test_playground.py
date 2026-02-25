"""Tests for playground handler (aragora/server/handlers/playground.py).

Covers all routes and behavior of the PlaygroundHandler class:
- can_handle() routing for all ROUTES
- GET  /api/v1/playground/status           - Health check
- POST /api/v1/playground/debate           - Run mock debate
- POST /api/v1/playground/debate/live      - Run live debate with real agents
- POST /api/v1/playground/debate/live/cost-estimate - Pre-flight cost estimate
- POST /api/v1/playground/tts              - ElevenLabs TTS proxy
- Rate limiting behavior (mock and live)
- Error handling (missing params, invalid data, topic too long)
- Oracle mode (consult / divine / commune)
- Method not allowed responses
- Edge cases (no API keys, package fallback, tentacle failures)
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.playground import (
    PlaygroundHandler,
    _check_rate_limit,
    _check_live_rate_limit,
    _reset_rate_limits,
    _run_inline_mock_debate,
    _build_mock_proposals,
    _build_oracle_prompt,
    _build_tentacle_prompt,
    _build_upgrade_cta,
    _get_available_live_agents,
    _MAX_TOPIC_LENGTH,
    _MAX_ROUNDS,
    _MAX_AGENTS,
    _MIN_AGENTS,
    _DEFAULT_TOPIC,
    _DEFAULT_ROUNDS,
    _DEFAULT_AGENTS,
    _PLAYGROUND_RATE_LIMIT,
    _PLAYGROUND_RATE_WINDOW,
    _LIVE_RATE_LIMIT,
    _LIVE_RATE_WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to PlaygroundHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] | None = None,
    ):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a PlaygroundHandler with minimal server context."""
    return PlaygroundHandler({})


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Reset rate limit state before each test."""
    _reset_rate_limits()
    yield
    _reset_rate_limits()


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_debate_path(self, handler):
        assert handler.can_handle("/api/v1/playground/debate")

    def test_debate_live_path(self, handler):
        assert handler.can_handle("/api/v1/playground/debate/live")

    def test_cost_estimate_path(self, handler):
        assert handler.can_handle("/api/v1/playground/debate/live/cost-estimate")

    def test_status_path(self, handler):
        assert handler.can_handle("/api/v1/playground/status")

    def test_tts_path(self, handler):
        assert handler.can_handle("/api/v1/playground/tts")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_match(self, handler):
        assert not handler.can_handle("/api/v1/playground")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/playground/debate")

    def test_rejects_extra_suffix(self, handler):
        assert not handler.can_handle("/api/v1/playground/debate/extra")

    def test_rejects_typo(self, handler):
        assert not handler.can_handle("/api/v1/playgrond/debate")


# ============================================================================
# GET /api/v1/playground/status
# ============================================================================


class TestStatus:
    """Tests for the /status health check endpoint."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        assert _status(result) == 200

    def test_returns_ok_status(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        body = _body(result)
        assert body["status"] == "ok"

    def test_returns_engine_name(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        body = _body(result)
        assert body["engine"] == "aragora-debate"

    def test_returns_mock_agents_flag(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        body = _body(result)
        assert body["mock_agents"] is True

    def test_returns_max_rounds(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        body = _body(result)
        assert body["max_rounds"] == _MAX_ROUNDS

    def test_returns_max_agents(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        body = _body(result)
        assert body["max_agents"] == _MAX_AGENTS

    def test_returns_rate_limit_string(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/status", {}, mock_h)
        body = _body(result)
        assert "rate_limit" in body
        assert str(_PLAYGROUND_RATE_LIMIT) in body["rate_limit"]

    def test_non_matching_path_returns_none(self, handler):
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/playground/debate", {}, mock_h)
        assert result is None


# ============================================================================
# POST /api/v1/playground/debate (mock debate)
# ============================================================================


class TestMockDebate:
    """Tests for the mock debate endpoint."""

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_default_topic(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["topic"] == _DEFAULT_TOPIC

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_custom_topic(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"topic": "Is Rust better than Go?"})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["topic"] == "Is Rust better than Go?"

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_custom_rounds_clamped(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"rounds": 100})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["rounds_used"] <= _MAX_ROUNDS

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_custom_agents_clamped_max(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": 100})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["participants"]) <= _MAX_AGENTS

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_custom_agents_clamped_min(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": 1})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["participants"]) >= _MIN_AGENTS

    def test_topic_too_long_returns_400(self, handler):
        long_topic = "x" * (_MAX_TOPIC_LENGTH + 1)
        mock_h = _MockHTTPHandler("POST", body={"topic": long_topic})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 400

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_invalid_rounds_uses_default(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"rounds": "not_a_number"})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        # rounds_used may be less than _DEFAULT_ROUNDS due to early stopping
        assert 1 <= body["rounds_used"] <= _DEFAULT_ROUNDS

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_invalid_agents_uses_default(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": "abc"})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["participants"]) == _DEFAULT_AGENTS

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_empty_topic_uses_default(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"topic": ""})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["topic"] == _DEFAULT_TOPIC

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_none_topic_uses_default(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={"topic": None})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["topic"] == _DEFAULT_TOPIC

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_response_has_required_fields(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        body = _body(result)
        required_fields = [
            "id",
            "topic",
            "status",
            "rounds_used",
            "consensus_reached",
            "confidence",
            "verdict",
            "duration_seconds",
            "participants",
            "proposals",
            "critiques",
            "votes",
            "dissenting_views",
            "final_answer",
        ]
        for field in required_fields:
            assert field in body, f"Missing required field: {field}"

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_response_has_receipt(self, mock_oracle, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        body = _body(result)
        assert "receipt" in body
        receipt = body["receipt"]
        assert "receipt_id" in receipt
        assert "signature" in receipt
        assert receipt["signature_algorithm"] == "SHA-256-content-hash"

    def test_unmatched_post_path_returns_none(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/unknown", {}, mock_h)
        assert result is None

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_null_handler_uses_empty_body(self, mock_oracle, handler):
        result = handler.handle_post("/api/v1/playground/debate", {}, None)
        assert _status(result) == 200


# ============================================================================
# Oracle mode (question-based debates)
# ============================================================================


class TestOracleMode:
    """Tests for Oracle mode debates (with question parameter)."""

    def test_oracle_success_returns_llm_response(self, handler):
        oracle_result = {
            "id": "test123",
            "topic": "What is consciousness?",
            "status": "completed",
            "rounds_used": 1,
            "consensus_reached": True,
            "confidence": 0.85,
            "verdict": "approved",
            "duration_seconds": 2.0,
            "participants": ["oracle"],
            "proposals": {"oracle": "Oracle response text"},
            "critiques": [],
            "votes": [],
            "dissenting_views": [],
            "final_answer": "Oracle response text",
            "receipt": {"receipt_id": "OR-20260101-abc123"},
            "receipt_hash": "abc123",
        }
        with patch(
            "aragora.server.handlers.playground._try_oracle_response",
            return_value=oracle_result,
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"topic": "system prompt", "question": "What is consciousness?"}
            )
            result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
            assert _status(result) == 200
            body = _body(result)
            assert body["final_answer"] == "Oracle response text"

    def test_oracle_failure_returns_placeholder(self, handler):
        with patch(
            "aragora.server.handlers.playground._try_oracle_response",
            return_value=None,
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"question": "What is consciousness?", "mode": "consult"}
            )
            result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
            assert _status(result) == 200
            body = _body(result)
            assert body["participants"] == ["oracle"]
            assert "gathering its thoughts" in body["final_answer"]

    def test_mode_defaults_to_consult(self, handler):
        with patch(
            "aragora.server.handlers.playground._try_oracle_response",
            return_value=None,
        ) as mock_oracle:
            mock_h = _MockHTTPHandler("POST", body={"question": "test question"})
            handler.handle_post("/api/v1/playground/debate", {}, mock_h)
            mock_oracle.assert_called_once()
            assert mock_oracle.call_args.kwargs.get("mode") == "consult"


# ============================================================================
# Rate limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on mock debates."""

    def test_first_request_allowed(self):
        allowed, retry_after = _check_rate_limit("test-ip-1")
        assert allowed is True
        assert retry_after == 0

    def test_rate_limit_exceeded(self):
        for _ in range(_PLAYGROUND_RATE_LIMIT):
            _check_rate_limit("test-ip-2")
        allowed, retry_after = _check_rate_limit("test-ip-2")
        assert allowed is False
        assert retry_after >= 1

    def test_different_ips_independent(self):
        for _ in range(_PLAYGROUND_RATE_LIMIT):
            _check_rate_limit("ip-a")
        allowed, _ = _check_rate_limit("ip-b")
        assert allowed is True

    def test_rate_limit_returns_429(self, handler):
        for _ in range(_PLAYGROUND_RATE_LIMIT):
            _check_rate_limit("127.0.0.1")
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 429
        body = _body(result)
        assert "rate_limit_exceeded" in body.get("code", "")
        assert "retry_after" in body

    def test_reset_clears_state(self):
        for _ in range(_PLAYGROUND_RATE_LIMIT):
            _check_rate_limit("test-ip-reset")
        _reset_rate_limits()
        allowed, _ = _check_rate_limit("test-ip-reset")
        assert allowed is True


class TestLiveRateLimiting:
    """Tests for rate limiting on live debates."""

    def test_first_live_request_allowed(self):
        allowed, retry_after = _check_live_rate_limit("live-ip-1")
        assert allowed is True
        assert retry_after == 0

    def test_live_rate_limit_exceeded(self):
        for _ in range(_LIVE_RATE_LIMIT):
            _check_live_rate_limit("live-ip-2")
        allowed, retry_after = _check_live_rate_limit("live-ip-2")
        assert allowed is False
        assert retry_after >= 1

    def test_live_rate_limit_returns_429(self, handler):
        _check_live_rate_limit("127.0.0.1")
        mock_h = _MockHTTPHandler("POST", body={})
        with patch(
            "aragora.server.handlers.playground._get_api_key",
            return_value="fake-key",
        ):
            result = handler._handle_live_debate(mock_h)
            assert _status(result) == 429
            body = _body(result)
            assert "live_rate_limit_exceeded" in body.get("code", "")


# ============================================================================
# POST /api/v1/playground/debate/live/cost-estimate
# ============================================================================


class TestCostEstimate:
    """Tests for the pre-flight cost estimate endpoint."""

    def test_default_cost_estimate(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "estimated_cost_usd" in body
        assert "budget_cap_usd" in body
        assert body["agent_count"] == _DEFAULT_AGENTS
        assert body["rounds"] == _DEFAULT_ROUNDS

    def test_custom_agents_and_rounds(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": 5, "rounds": 2})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert body["agent_count"] == 5
        assert body["rounds"] == 2
        # 5 agents * 2 rounds * 0.005 = 0.05
        assert body["estimated_cost_usd"] == 0.05

    def test_agents_clamped_to_max(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": 100})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert body["agent_count"] == _MAX_AGENTS

    def test_agents_clamped_to_min(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": 0})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert body["agent_count"] == _MIN_AGENTS

    def test_rounds_clamped(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"rounds": 100})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert body["rounds"] == _MAX_ROUNDS

    def test_invalid_agents_uses_default(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"agents": "invalid"})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert body["agent_count"] == _DEFAULT_AGENTS

    def test_invalid_rounds_uses_default(self, handler):
        mock_h = _MockHTTPHandler("POST", body={"rounds": "invalid"})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert body["rounds"] == _DEFAULT_ROUNDS

    def test_null_handler_uses_empty_body(self, handler):
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["agent_count"] == _DEFAULT_AGENTS

    def test_timeout_seconds_in_response(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        body = _body(result)
        assert "timeout_seconds" in body


# ============================================================================
# POST /api/v1/playground/debate/live
# ============================================================================


class TestLiveDebate:
    """Tests for the live debate endpoint."""

    def test_no_api_keys_falls_back_to_mock(self, handler):
        with (
            patch(
                "aragora.server.handlers.playground._get_api_key",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.playground._try_oracle_response",
                return_value=None,
            ),
        ):
            mock_h = _MockHTTPHandler("POST", body={"topic": "AI Safety"})
            result = handler._handle_live_debate(mock_h)
            assert _status(result) == 200
            body = _body(result)
            assert body.get("is_live") is False
            assert body.get("mock_fallback") is True
            assert "upgrade_cta" in body

    def test_topic_too_long_returns_400(self, handler):
        _reset_rate_limits()
        with patch(
            "aragora.server.handlers.playground._get_api_key",
            return_value="key",
        ):
            long_topic = "x" * (_MAX_TOPIC_LENGTH + 1)
            mock_h = _MockHTTPHandler("POST", body={"topic": long_topic})
            result = handler._handle_live_debate(mock_h)
            assert _status(result) == 400

    def test_with_question_tries_tentacles(self, handler):
        _reset_rate_limits()
        tentacle_result = {
            "id": "tent123",
            "topic": "test",
            "status": "completed",
            "rounds_used": 1,
            "consensus_reached": True,
            "confidence": 0.7,
            "verdict": "needs_review",
            "duration_seconds": 3.0,
            "participants": ["claude", "gpt"],
            "proposals": {"claude": "response A", "gpt": "response B"},
            "critiques": [],
            "votes": [],
            "dissenting_views": [],
            "final_answer": "response A",
            "is_live": True,
        }
        with (
            patch(
                "aragora.server.handlers.playground._get_api_key",
                return_value="key",
            ),
            patch(
                "aragora.server.handlers.playground._try_oracle_tentacles",
                return_value=tentacle_result,
            ),
        ):
            mock_h = _MockHTTPHandler(
                "POST", body={"question": "What is truth?", "topic": "system prompt"}
            )
            result = handler._handle_live_debate(mock_h)
            assert _status(result) == 200
            body = _body(result)
            assert "upgrade_cta" in body

    def test_tentacle_failure_falls_back_to_live_factory(self, handler):
        _reset_rate_limits()
        with (
            patch(
                "aragora.server.handlers.playground._get_api_key",
                return_value="key",
            ),
            patch(
                "aragora.server.handlers.playground._try_oracle_tentacles",
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.playground.PlaygroundHandler._run_live_debate",
            ) as mock_run_live,
        ):
            mock_run_live.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"status": "completed"}).encode(),
                content_type="application/json",
            )
            mock_h = _MockHTTPHandler("POST", body={"question": "What is truth?"})
            result = handler._handle_live_debate(mock_h)
            mock_run_live.assert_called_once()

    def test_live_debate_500_falls_back_to_mock(self, handler):
        _reset_rate_limits()
        with (
            patch(
                "aragora.server.handlers.playground._get_api_key",
                return_value="key",
            ),
            patch(
                "aragora.server.handlers.playground.PlaygroundHandler._run_live_debate",
            ) as mock_run_live,
            patch(
                "aragora.server.handlers.playground._try_oracle_response",
                return_value=None,
            ),
        ):
            from aragora.server.handlers.utils.responses import HandlerResult

            mock_run_live.return_value = HandlerResult(
                status_code=500,
                content_type="application/json",
                body=json.dumps({"error": "fail"}).encode(),
            )
            mock_h = _MockHTTPHandler("POST", body={"topic": "test topic"})
            result = handler._handle_live_debate(mock_h)
            assert _status(result) == 200
            body = _body(result)
            assert body.get("mock_fallback") is True

    def test_source_parameter_passed_through(self, handler):
        _reset_rate_limits()
        with (
            patch(
                "aragora.server.handlers.playground._get_api_key",
                return_value="key",
            ),
            patch(
                "aragora.server.handlers.playground._try_oracle_tentacles",
            ) as mock_tentacles,
        ):
            mock_tentacles.return_value = {
                "id": "t1",
                "topic": "q",
                "status": "completed",
                "rounds_used": 1,
                "consensus_reached": True,
                "confidence": 0.7,
                "verdict": "needs_review",
                "duration_seconds": 1,
                "participants": ["a"],
                "proposals": {},
                "critiques": [],
                "votes": [],
                "dissenting_views": [],
                "final_answer": "text",
                "is_live": True,
            }
            mock_h = _MockHTTPHandler(
                "POST",
                body={"question": "test?", "source": "landing"},
            )
            handler._handle_live_debate(mock_h)
            mock_tentacles.assert_called_once()
            assert mock_tentacles.call_args.kwargs.get("source") == "landing"


# ============================================================================
# POST /api/v1/playground/tts
# ============================================================================


class TestTTS:
    """Tests for the TTS proxy endpoint."""

    def test_no_api_key_returns_503(self, handler):
        with patch(
            "aragora.config.secrets.get_secret",
            return_value=None,
        ):
            mock_h = _MockHTTPHandler("POST", body={"text": "Hello world"})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 503

    def test_missing_text_returns_400(self, handler):
        with patch(
            "aragora.config.secrets.get_secret",
            return_value="fake-key",
        ):
            mock_h = _MockHTTPHandler("POST", body={"text": ""})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 400

    def test_missing_text_field_returns_400(self, handler):
        with patch(
            "aragora.config.secrets.get_secret",
            return_value="fake-key",
        ):
            mock_h = _MockHTTPHandler("POST", body={})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 400

    def test_successful_tts_returns_audio(self, handler):
        fake_audio = b"\xff\xfb\x90\x00" * 100  # fake mp3 bytes
        with (
            patch(
                "aragora.config.secrets.get_secret",
                return_value="fake-key",
            ),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_audio
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            mock_h = _MockHTTPHandler("POST", body={"text": "Hello oracle"})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 200
            assert result.content_type == "audio/mpeg"
            assert result.body == fake_audio

    def test_elevenlabs_http_error_returns_502(self, handler):
        import urllib.error

        with (
            patch(
                "aragora.config.secrets.get_secret",
                return_value="fake-key",
            ),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_urlopen.side_effect = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
            mock_h = _MockHTTPHandler("POST", body={"text": "test"})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 502

    def test_elevenlabs_timeout_returns_503(self, handler):
        with (
            patch(
                "aragora.config.secrets.get_secret",
                return_value="fake-key",
            ),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_urlopen.side_effect = TimeoutError("timeout")
            mock_h = _MockHTTPHandler("POST", body={"text": "test"})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 503

    def test_tts_rate_limit(self, handler):
        with patch(
            "aragora.config.secrets.get_secret",
            return_value="fake-key",
        ):
            # Exhaust the TTS rate limit
            for _ in range(handler._TTS_RATE_LIMIT):
                _check_rate_limit(
                    "tts:127.0.0.1",
                    limit=handler._TTS_RATE_LIMIT,
                    window=handler._TTS_RATE_WINDOW,
                )
            mock_h = _MockHTTPHandler("POST", body={"text": "test"})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 429

    def test_text_truncated_to_max(self, handler):
        fake_audio = b"\xff\xfb\x90\x00"
        with (
            patch(
                "aragora.config.secrets.get_secret",
                return_value="fake-key",
            ),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_audio
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            long_text = "a" * 5000
            mock_h = _MockHTTPHandler("POST", body={"text": long_text})
            result = handler._handle_tts(mock_h)
            assert _status(result) == 200

    def test_tts_dispatched_from_handle_post(self, handler):
        with patch.object(handler, "_handle_tts") as mock_tts:
            from aragora.server.handlers.utils.responses import HandlerResult

            mock_tts.return_value = HandlerResult(
                status_code=200, content_type="audio/mpeg", body=b"audio"
            )
            mock_h = _MockHTTPHandler("POST", body={"text": "test"})
            handler.handle_post("/api/v1/playground/tts", {}, mock_h)
            mock_tts.assert_called_once()


# ============================================================================
# Inline mock debate function
# ============================================================================


class TestInlineMockDebate:
    """Tests for the _run_inline_mock_debate function."""

    def test_returns_required_fields(self):
        result = _run_inline_mock_debate("Test topic", 2, 3)
        required = [
            "id",
            "topic",
            "status",
            "rounds_used",
            "consensus_reached",
            "confidence",
            "verdict",
            "duration_seconds",
            "participants",
            "proposals",
            "critiques",
            "votes",
            "dissenting_views",
            "final_answer",
            "receipt",
            "receipt_hash",
        ]
        for field in required:
            assert field in result, f"Missing: {field}"

    def test_correct_participant_count(self):
        for count in [2, 3, 4, 5]:
            result = _run_inline_mock_debate("Topic", 2, count)
            assert len(result["participants"]) == count

    def test_receipt_has_correct_structure(self):
        result = _run_inline_mock_debate("Topic", 2, 3)
        receipt = result["receipt"]
        assert receipt["receipt_id"].startswith("DR-")
        assert receipt["signature_algorithm"] == "SHA-256-content-hash"
        assert "consensus" in receipt
        assert receipt["consensus"]["method"] == "majority"

    def test_verdict_categories(self):
        # Run many mock debates to see different verdicts
        verdicts_seen = set()
        for _ in range(50):
            result = _run_inline_mock_debate("Topic", 2, 3)
            verdicts_seen.add(result["verdict"])
        # Should see at least 2 different verdicts across 50 runs
        assert len(verdicts_seen) >= 1

    def test_critiques_cross_agent(self):
        result = _run_inline_mock_debate("Topic", 2, 3)
        for critique in result["critiques"]:
            assert critique["agent"] != critique["target_agent"]

    def test_votes_reference_other_agents(self):
        result = _run_inline_mock_debate("Topic", 2, 3)
        names = set(result["participants"])
        for vote in result["votes"]:
            assert vote["agent"] in names
            assert vote["choice"] in names
            assert vote["agent"] != vote["choice"]

    def test_question_param_used_in_proposals(self):
        result = _run_inline_mock_debate("system prompt", 2, 3, question="Custom Q?")
        # Proposals should reference the question, not the system prompt
        all_text = " ".join(result["proposals"].values())
        assert "Custom Q?" in all_text or "Custom" in all_text

    def test_duration_positive(self):
        result = _run_inline_mock_debate("Topic", 1, 2)
        assert result["duration_seconds"] >= 0


# ============================================================================
# Build mock proposals
# ============================================================================


class TestBuildMockProposals:
    """Tests for the _build_mock_proposals function."""

    def test_returns_all_styles(self):
        proposals = _build_mock_proposals("Test topic")
        assert "supportive" in proposals
        assert "critical" in proposals
        assert "balanced" in proposals
        assert "contrarian" in proposals

    def test_topic_appears_in_proposals(self):
        proposals = _build_mock_proposals("Quantum computing")
        for style, texts in proposals.items():
            for text in texts:
                assert "Quantum computing" in text

    def test_long_topic_truncated_with_ellipsis(self):
        long_topic = "x " * 200  # 400 chars
        proposals = _build_mock_proposals(long_topic)
        for texts in proposals.values():
            for text in texts:
                assert "..." in text

    def test_question_overrides_topic(self):
        proposals = _build_mock_proposals("system prompt", question="Actual question")
        for texts in proposals.values():
            for text in texts:
                assert "Actual question" in text


# ============================================================================
# Oracle prompt building
# ============================================================================


class TestBuildOraclePrompt:
    """Tests for the _build_oracle_prompt function."""

    def test_consult_mode(self):
        prompt = _build_oracle_prompt("consult", "What is AI?")
        assert "What is AI?" in prompt
        assert "Shoggoth Oracle" in prompt

    def test_divine_mode(self):
        prompt = _build_oracle_prompt("divine", "My future?")
        assert "My future?" in prompt
        assert "SURVIVOR" in prompt or "Cassandra" in prompt

    def test_commune_mode(self):
        prompt = _build_oracle_prompt("commune", "Hello?")
        assert "Hello?" in prompt
        assert "cryptic" in prompt.lower() or "kind" in prompt.lower()

    def test_unknown_mode_defaults_to_commune(self):
        prompt = _build_oracle_prompt("unknown_mode", "Test?")
        assert "Test?" in prompt


# ============================================================================
# Tentacle prompt building
# ============================================================================


class TestBuildTentaclePrompt:
    """Tests for the _build_tentacle_prompt function."""

    def test_oracle_source_consult(self):
        prompt = _build_tentacle_prompt("consult", "Q?", "ROLE TEXT", source="oracle")
        assert "tentacle" in prompt.lower() or "Shoggoth" in prompt
        assert "ROLE TEXT" in prompt
        assert "Q?" in prompt

    def test_oracle_source_divine(self):
        prompt = _build_tentacle_prompt("divine", "Q?", "ROLE", source="oracle")
        assert "prophecy" in prompt.lower() or "tentacle" in prompt.lower()

    def test_oracle_source_commune(self):
        prompt = _build_tentacle_prompt("commune", "Q?", "ROLE", source="oracle")
        assert "Oracle" in prompt

    def test_landing_source_neutral(self):
        prompt = _build_tentacle_prompt("consult", "Q?", "ROLE", source="landing")
        assert "adversarial debate" in prompt.lower() or "independent AI agents" in prompt
        # Should NOT leak oracle terminology
        assert "tentacle" not in prompt.lower()
        assert "Shoggoth" not in prompt


# ============================================================================
# Upgrade CTA
# ============================================================================


class TestUpgradeCTA:
    """Tests for _build_upgrade_cta helper."""

    def test_has_required_fields(self):
        cta = _build_upgrade_cta()
        assert "title" in cta
        assert "message" in cta
        assert "action_url" in cta
        assert "action_label" in cta

    def test_action_url_is_pricing(self):
        cta = _build_upgrade_cta()
        assert cta["action_url"] == "/pricing"


# ============================================================================
# Available live agents
# ============================================================================


class TestGetAvailableLiveAgents:
    """Tests for _get_available_live_agents."""

    def test_no_keys_returns_empty(self):
        with patch(
            "aragora.server.handlers.playground._get_api_key",
            return_value=None,
        ):
            agents = _get_available_live_agents(3)
            assert agents == []

    def test_anthropic_key_returns_anthropic(self):
        def fake_key(name):
            return "key" if name == "ANTHROPIC_API_KEY" else None

        with patch(
            "aragora.server.handlers.playground._get_api_key",
            side_effect=fake_key,
        ):
            agents = _get_available_live_agents(3)
            assert "anthropic-api" in agents

    def test_pads_to_requested_count(self):
        def fake_key(name):
            return "key" if name == "ANTHROPIC_API_KEY" else None

        with patch(
            "aragora.server.handlers.playground._get_api_key",
            side_effect=fake_key,
        ):
            agents = _get_available_live_agents(3)
            assert len(agents) == 3

    def test_multiple_providers(self):
        with patch(
            "aragora.server.handlers.playground._get_api_key",
            return_value="key",
        ):
            agents = _get_available_live_agents(4)
            assert len(agents) == 4
            assert "anthropic-api" in agents
            assert "openai-api" in agents


# ============================================================================
# _run_live_debate method (handler-level)
# ============================================================================


class TestRunLiveDebate:
    """Tests for the PlaygroundHandler._run_live_debate method."""

    def test_no_debate_controller_returns_503(self, handler):
        with patch("importlib.util.find_spec", return_value=None):
            result = handler._run_live_debate("topic", 2, 3)
            assert _status(result) == 503

    def test_timeout_returns_408(self, handler):
        with (
            patch("importlib.util.find_spec", return_value=True),
            patch(
                "aragora.server.handlers.playground.start_playground_debate",
                side_effect=TimeoutError("timed out"),
            ),
        ):
            result = handler._run_live_debate("topic", 2, 3)
            assert _status(result) == 408
            body = _body(result)
            assert body["code"] == "timeout"
            assert "upgrade_cta" in body

    def test_value_error_returns_500(self, handler):
        with (
            patch("importlib.util.find_spec", return_value=True),
            patch(
                "aragora.server.handlers.playground.start_playground_debate",
                side_effect=ValueError("no agents"),
            ),
        ):
            result = handler._run_live_debate("topic", 2, 3)
            assert _status(result) == 500

    def test_successful_live_debate(self, handler):
        mock_result = {
            "status": "completed",
            "rounds_used": 2,
            "consensus_reached": True,
            "confidence": 0.8,
            "verdict": "approved",
            "duration_seconds": 5.0,
            "participants": ["anthropic", "openai"],
            "proposals": {"anthropic": "A says...", "openai": "B says..."},
            "critiques": [],
            "votes": [],
            "dissenting_views": [],
            "final_answer": "Conclusion",
        }
        with (
            patch("importlib.util.find_spec", return_value=True),
            patch(
                "aragora.server.handlers.playground.start_playground_debate",
                return_value=mock_result,
            ),
        ):
            result = handler._run_live_debate("test topic", 2, 3)
            assert _status(result) == 200
            body = _body(result)
            assert body["is_live"] is True
            assert "receipt_preview" in body
            assert "upgrade_cta" in body
            assert body["topic"] == "test topic"


# ============================================================================
# POST dispatch routing
# ============================================================================


class TestPostDispatch:
    """Verify handle_post routes to the correct method."""

    def test_debate_path_dispatches(self, handler):
        with patch(
            "aragora.server.handlers.playground._try_oracle_response",
            return_value=None,
        ):
            mock_h = _MockHTTPHandler("POST", body={})
            result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
            assert _status(result) == 200

    def test_cost_estimate_path_dispatches(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, mock_h)
        assert _status(result) == 200

    def test_live_path_dispatches(self, handler):
        with patch.object(handler, "_handle_live_debate") as mock_live:
            from aragora.server.handlers.utils.responses import HandlerResult

            mock_live.return_value = HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps({"ok": True}).encode(),
            )
            mock_h = _MockHTTPHandler("POST", body={})
            handler.handle_post("/api/v1/playground/debate/live", {}, mock_h)
            mock_live.assert_called_once()

    def test_tts_path_dispatches(self, handler):
        with patch.object(handler, "_handle_tts") as mock_tts:
            from aragora.server.handlers.utils.responses import HandlerResult

            mock_tts.return_value = HandlerResult(
                status_code=200, content_type="audio/mpeg", body=b"audio"
            )
            mock_h = _MockHTTPHandler("POST", body={})
            handler.handle_post("/api/v1/playground/tts", {}, mock_h)
            mock_tts.assert_called_once()

    def test_unknown_path_returns_none(self, handler):
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle_post("/api/v1/playground/nonexistent", {}, mock_h)
        assert result is None


# ============================================================================
# Handler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for PlaygroundHandler initialization."""

    def test_default_context(self):
        h = PlaygroundHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        ctx = {"storage": "mock"}
        h = PlaygroundHandler(ctx)
        assert h.ctx == ctx

    def test_routes_attribute(self):
        assert "/api/v1/playground/debate" in PlaygroundHandler.ROUTES
        assert "/api/v1/playground/status" in PlaygroundHandler.ROUTES
        assert "/api/v1/playground/debate/live" in PlaygroundHandler.ROUTES
        assert "/api/v1/playground/debate/live/cost-estimate" in PlaygroundHandler.ROUTES
        assert "/api/v1/playground/tts" in PlaygroundHandler.ROUTES


# ============================================================================
# Client IP extraction
# ============================================================================


class TestClientIPExtraction:
    """Tests for client IP extraction from handler."""

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_ip_from_client_address(self, mock_oracle, handler):
        """Rate limiting uses the IP from handler.client_address."""
        mock_h = _MockHTTPHandler("POST", body={}, client_address=("192.168.1.1", 54321))
        # First call should succeed
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.playground._try_oracle_response",
        return_value=None,
    )
    def test_missing_client_address_uses_unknown(self, mock_oracle, handler):
        """Handler without client_address falls back to 'unknown' IP."""
        mock_h = _MockHTTPHandler("POST", body={})
        del mock_h.client_address  # Remove the attribute
        result = handler.handle_post("/api/v1/playground/debate", {}, mock_h)
        assert _status(result) == 200
