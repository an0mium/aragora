"""Tests for the live playground debate endpoint.

Covers:
- Live debate returns real result with is_live flag
- Rate limit enforces 1/10min per IP
- Budget cap / timeout abort
- Cost-estimate endpoint
- upgrade_cta present in response
- Mock fallback when no API keys available
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.playground import (
    PlaygroundHandler,
    _check_live_rate_limit,
    _reset_rate_limits,
    _build_upgrade_cta,
    start_playground_debate,
    _LIVE_TIMEOUT,
)


def _parse(result: HandlerResult) -> dict:
    return json.loads(result.body.decode("utf-8"))


def _make_handler(client_ip: str = "10.0.0.1") -> MagicMock:
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {"Content-Length": "0"}
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b""
    return handler


def _make_handler_with_body(body: dict, client_ip: str = "10.0.0.1") -> MagicMock:
    raw = json.dumps(body).encode()
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = raw
    return handler


@pytest.fixture(autouse=True)
def reset_limits():
    _reset_rate_limits()
    yield
    _reset_rate_limits()


# ============================================================================
# Cost estimate endpoint
# ============================================================================


class TestCostEstimate:
    def test_cost_estimate_returns_estimate(self):
        pg = PlaygroundHandler()
        handler = _make_handler_with_body({"agents": 3, "rounds": 2})
        result = pg.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, handler)
        assert result is not None
        data = _parse(result)
        assert "estimated_cost_usd" in data
        assert "budget_cap_usd" in data
        assert data["budget_cap_usd"] == 0.05
        assert data["agent_count"] == 3
        assert data["rounds"] == 2
        assert data["timeout_seconds"] == _LIVE_TIMEOUT

    def test_cost_estimate_clamps_agents(self):
        pg = PlaygroundHandler()
        handler = _make_handler_with_body({"agents": 100, "rounds": 1})
        result = pg.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, handler)
        data = _parse(result)
        assert data["agent_count"] == 5  # max

    def test_cost_estimate_defaults(self):
        pg = PlaygroundHandler()
        handler = _make_handler()
        result = pg.handle_post("/api/v1/playground/debate/live/cost-estimate", {}, handler)
        data = _parse(result)
        assert data["agent_count"] == 3  # default
        assert data["rounds"] == 2  # default


# ============================================================================
# Live rate limiting
# ============================================================================


class TestLiveRateLimit:
    def test_first_request_allowed(self):
        allowed, retry = _check_live_rate_limit("1.2.3.4")
        assert allowed is True
        assert retry == 0

    def test_second_request_blocked(self):
        _check_live_rate_limit("1.2.3.4")
        allowed, retry = _check_live_rate_limit("1.2.3.4")
        assert allowed is False
        assert retry > 0

    def test_different_ips_independent(self):
        _check_live_rate_limit("1.2.3.4")
        allowed, _ = _check_live_rate_limit("5.6.7.8")
        assert allowed is True

    def test_rate_limit_returns_429(self):
        pg = PlaygroundHandler()
        ip = "10.0.0.99"
        # First request needs API keys to not hit the mock fallback path first
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch(
                "aragora.server.handlers.playground.start_playground_debate",
                return_value={"status": "completed", "final_answer": "test"},
            ):
                handler1 = _make_handler_with_body({"topic": "test"}, client_ip=ip)
                pg.handle_post("/api/v1/playground/debate/live", {}, handler1)

            handler2 = _make_handler_with_body({"topic": "test"}, client_ip=ip)
            result = pg.handle_post("/api/v1/playground/debate/live", {}, handler2)
            assert result is not None
            assert result.status_code == 429
            data = _parse(result)
            assert data["code"] == "live_rate_limit_exceeded"


# ============================================================================
# Live debate with real agents (mocked)
# ============================================================================


class TestLiveDebate:
    def test_live_debate_returns_is_live(self):
        pg = PlaygroundHandler()
        mock_result = {
            "status": "completed",
            "rounds_used": 2,
            "consensus_reached": True,
            "confidence": 0.85,
            "verdict": None,
            "duration_seconds": 12.5,
            "participants": ["anthropic", "openai"],
            "proposals": [],
            "critiques": [],
            "votes": [],
            "dissenting_views": [],
            "final_answer": "Test answer",
        }
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch(
                "aragora.server.handlers.playground.start_playground_debate",
                return_value=mock_result,
            ):
                handler = _make_handler_with_body({"topic": "Test question"})
                result = pg.handle_post("/api/v1/playground/debate/live", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = _parse(result)
        assert data["is_live"] is True
        assert data["final_answer"] == "Test answer"
        assert data["consensus_reached"] is True

    def test_live_debate_includes_receipt_preview(self):
        pg = PlaygroundHandler()
        mock_result = {
            "status": "completed",
            "consensus_reached": False,
            "confidence": 0.5,
            "participants": ["anthropic"],
            "final_answer": "Answer",
        }
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with patch(
                "aragora.server.handlers.playground.start_playground_debate",
                return_value=mock_result,
            ):
                handler = _make_handler_with_body({"topic": "Test"})
                result = pg.handle_post("/api/v1/playground/debate/live", {}, handler)

        data = _parse(result)
        assert "receipt_preview" in data
        assert data["receipt_preview"]["consensus_reached"] is False
        assert "note" in data["receipt_preview"]

    def test_live_debate_includes_upgrade_cta(self):
        pg = PlaygroundHandler()
        mock_result = {"status": "completed", "final_answer": "x"}
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch(
                "aragora.server.handlers.playground.start_playground_debate",
                return_value=mock_result,
            ):
                handler = _make_handler_with_body({"topic": "Test"})
                result = pg.handle_post("/api/v1/playground/debate/live", {}, handler)

        data = _parse(result)
        assert "upgrade_cta" in data
        cta = data["upgrade_cta"]
        assert "title" in cta
        assert "action_url" in cta

    def test_timeout_returns_408(self):
        pg = PlaygroundHandler()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            with patch(
                "aragora.server.handlers.playground.start_playground_debate",
                side_effect=TimeoutError("timed out"),
            ):
                handler = _make_handler_with_body({"topic": "Test"})
                result = pg.handle_post("/api/v1/playground/debate/live", {}, handler)

        assert result is not None
        assert result.status_code == 408
        data = _parse(result)
        assert data["code"] == "timeout"


# ============================================================================
# Mock fallback
# ============================================================================


class TestMockFallback:
    def test_fallback_when_no_api_keys(self):
        pg = PlaygroundHandler()
        env = {
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        }
        with patch.dict("os.environ", env, clear=False):
            # Mock the aragora_debate imports used by _run_debate
            mock_result = MagicMock()
            mock_result.id = "mock-123"
            mock_result.task = "Test"
            mock_result.status = "completed"
            mock_result.rounds_used = 2
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.verdict = None
            mock_result.duration_seconds = 1.0
            mock_result.participants = ["analyst", "critic"]
            mock_result.proposals = []
            mock_result.critiques = []
            mock_result.votes = []
            mock_result.dissenting_views = []
            mock_result.final_answer = "Mock answer"
            mock_result.receipt = None

            with patch(
                "aragora.server.handlers.playground.PlaygroundHandler._run_debate",
                return_value=json.loads(json.dumps({
                    "id": "mock-123",
                    "topic": "Test",
                    "status": "completed",
                    "final_answer": "Mock answer",
                }).encode()).encode() if False else None,
            ):
                # Actually let _run_debate produce a HandlerResult
                from aragora.server.handlers.base import json_response as _jr

                mock_hr = _jr({
                    "id": "mock-123",
                    "topic": "Test",
                    "status": "completed",
                    "final_answer": "Mock answer",
                    "consensus_reached": True,
                    "confidence": 0.9,
                })
                with patch.object(pg, "_run_debate", return_value=mock_hr):
                    handler = _make_handler_with_body({"topic": "Test"})
                    result = pg.handle_post("/api/v1/playground/debate/live", {}, handler)

        assert result is not None
        data = _parse(result)
        assert data["is_live"] is False
        assert data["mock_fallback"] is True
        assert "mock_fallback_reason" in data
        assert "upgrade_cta" in data


# ============================================================================
# upgrade_cta helper
# ============================================================================


class TestUpgradeCta:
    def test_build_upgrade_cta_structure(self):
        cta = _build_upgrade_cta()
        assert isinstance(cta, dict)
        assert "title" in cta
        assert "message" in cta
        assert "action_url" in cta
        assert "action_label" in cta


# ============================================================================
# can_handle routing
# ============================================================================


class TestRouting:
    def test_can_handle_live(self):
        pg = PlaygroundHandler()
        assert pg.can_handle("/api/v1/playground/debate/live") is True

    def test_can_handle_cost_estimate(self):
        pg = PlaygroundHandler()
        assert pg.can_handle("/api/v1/playground/debate/live/cost-estimate") is True

    def test_can_handle_mock(self):
        pg = PlaygroundHandler()
        assert pg.can_handle("/api/v1/playground/debate") is True

    def test_cannot_handle_unknown(self):
        pg = PlaygroundHandler()
        assert pg.can_handle("/api/v1/playground/unknown") is False
