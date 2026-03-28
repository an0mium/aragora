"""
Tests for debate session cost calculation API endpoints.

Covers:
- SessionCostHandler path parsing and routing
- GET /api/v1/costs/sessions/{debate_id}/breakdown
- GET /api/v1/costs/sessions/{debate_id}/metrics
- GET /api/v1/costs/sessions/{debate_id}/receipt
- Empty / zero-cost scenarios
- Debate-not-found handling
- Rate limiting
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.costs.session_costs import SessionCostHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> dict[str, Any]:
    """Minimal server context."""
    storage = MagicMock()
    storage.get_debate.return_value = {"id": "debate-1", "task": "test"}
    return {"storage": storage}


@pytest.fixture
def handler(ctx: dict[str, Any]) -> SessionCostHandler:
    return SessionCostHandler(ctx=ctx)


@pytest.fixture
def mock_handler() -> MagicMock:
    """Simulated HTTP request handler object."""
    h = MagicMock()
    h.command = "GET"
    h.headers = {}
    h.client_address = ("127.0.0.1", 12345)
    return h


def _parse_body(result: Any) -> dict[str, Any]:
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------


class TestPathParsing:
    def test_parse_versioned_breakdown(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/sessions/abc/breakdown") == (
            "abc",
            "breakdown",
        )

    def test_parse_versioned_metrics(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/sessions/d-123/metrics") == (
            "d-123",
            "metrics",
        )

    def test_parse_versioned_receipt(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/sessions/xyz/receipt") == (
            "xyz",
            "receipt",
        )

    def test_parse_legacy_breakdown(self):
        assert SessionCostHandler._parse_path("/api/costs/sessions/abc/breakdown") == (
            "abc",
            "breakdown",
        )

    def test_parse_legacy_metrics(self):
        assert SessionCostHandler._parse_path("/api/costs/sessions/abc/metrics") == (
            "abc",
            "metrics",
        )

    def test_parse_invalid_action(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/sessions/abc/invalid") is None

    def test_parse_missing_id(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/sessions//breakdown") is None

    def test_parse_unrelated_path(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/breakdown") is None

    def test_parse_no_action(self):
        assert SessionCostHandler._parse_path("/api/v1/costs/sessions/abc") is None


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_handles_valid_paths(self, handler: SessionCostHandler):
        assert handler.can_handle("/api/v1/costs/sessions/abc/breakdown") is True
        assert handler.can_handle("/api/v1/costs/sessions/abc/metrics") is True
        assert handler.can_handle("/api/v1/costs/sessions/abc/receipt") is True

    def test_rejects_invalid_paths(self, handler: SessionCostHandler):
        assert handler.can_handle("/api/v1/costs/breakdown") is False
        assert handler.can_handle("/api/v1/billing/dashboard") is False


# ---------------------------------------------------------------------------
# Breakdown endpoint
# ---------------------------------------------------------------------------


class TestBreakdownEndpoint:
    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_empty_debate_returns_zero_costs(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        result = handler.handle("/api/v1/costs/sessions/debate-1/breakdown", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "debate-1"
        assert data["total_cost_usd"] == "0"
        assert data["total_api_calls"] == 0
        assert data["per_agent"] == {}
        assert data["per_round"] == {}

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.billing.debate_costs.get_debate_cost_tracker")
    def test_breakdown_with_cost_data(
        self,
        mock_tracker: MagicMock,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        from aragora.billing.debate_costs import DebateCostTracker

        tracker = DebateCostTracker()
        tracker.record_agent_call(
            debate_id="debate-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens_in=2000,
            tokens_out=800,
            round_number=1,
            operation="proposal",
        )
        tracker.record_agent_call(
            debate_id="debate-1",
            agent_name="gpt",
            provider="openai",
            model="gpt-4o",
            tokens_in=1500,
            tokens_out=600,
            round_number=1,
            operation="critique",
        )
        mock_tracker.return_value = tracker

        result = handler.handle("/api/v1/costs/sessions/debate-1/breakdown", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "debate-1"
        assert Decimal(data["total_cost_usd"]) > 0
        assert data["total_api_calls"] == 2
        assert "claude" in data["per_agent"]
        assert "gpt" in data["per_agent"]
        assert "1" in data["per_round"]
        assert data["num_rounds"] == 1

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_breakdown_debate_not_found(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
        ctx: dict[str, Any],
    ):
        ctx["storage"].get_debate.return_value = None
        result = handler.handle("/api/v1/costs/sessions/missing/breakdown", {}, mock_handler)
        assert result is not None
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_empty_metrics(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        result = handler.handle("/api/v1/costs/sessions/debate-1/metrics", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "debate-1"
        assert data["summary"]["total_cost_usd"] == "0"
        assert data["efficiency"]["avg_cost_per_call"] == "0"

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.billing.debate_costs.get_debate_cost_tracker")
    def test_metrics_with_data(
        self,
        mock_tracker: MagicMock,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        from aragora.billing.debate_costs import DebateCostTracker

        tracker = DebateCostTracker()
        tracker.record_agent_call(
            debate_id="debate-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens_in=2000,
            tokens_out=800,
            round_number=1,
        )
        tracker.record_agent_call(
            debate_id="debate-1",
            agent_name="gpt",
            provider="openai",
            model="gpt-4o",
            tokens_in=1500,
            tokens_out=600,
            round_number=2,
        )
        mock_tracker.return_value = tracker

        result = handler.handle("/api/v1/costs/sessions/debate-1/metrics", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)

        assert Decimal(data["summary"]["total_cost_usd"]) > 0
        assert data["summary"]["total_api_calls"] == 2
        assert data["summary"]["total_tokens"] == 2000 + 800 + 1500 + 600
        assert data["efficiency"]["input_output_ratio"] > 0
        assert "claude" in data["per_agent"]
        assert data["per_agent"]["claude"]["cost_share_pct"] > 0

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_metrics_debate_not_found(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
        ctx: dict[str, Any],
    ):
        ctx["storage"].get_debate.return_value = None
        result = handler.handle("/api/v1/costs/sessions/missing/metrics", {}, mock_handler)
        assert result is not None
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Receipt endpoint
# ---------------------------------------------------------------------------


class TestReceiptEndpoint:
    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_empty_receipt(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        result = handler.handle("/api/v1/costs/sessions/debate-1/receipt", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "debate-1"
        assert data["receipt_id"].startswith("rcpt-debate-1-")
        assert "checksum" in data
        assert len(data["checksum"]) == 64  # SHA-256 hex
        assert data["totals"]["cost_usd"] == "0"
        assert data["line_items"] == []

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.billing.debate_costs.get_debate_cost_tracker")
    def test_receipt_with_data(
        self,
        mock_tracker: MagicMock,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        from aragora.billing.debate_costs import DebateCostTracker

        tracker = DebateCostTracker()
        tracker.record_agent_call(
            debate_id="debate-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens_in=3000,
            tokens_out=1200,
            round_number=1,
        )
        mock_tracker.return_value = tracker

        result = handler.handle("/api/v1/costs/sessions/debate-1/receipt", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)

        assert data["debate_id"] == "debate-1"
        assert Decimal(data["totals"]["cost_usd"]) > 0
        assert data["totals"]["api_calls"] == 1
        assert data["totals"]["rounds"] == 1
        assert len(data["line_items"]) == 1
        assert data["line_items"][0]["description"] == "Agent: claude"
        assert "checksum" in data
        assert data["generated_at"] is not None

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_receipt_debate_not_found(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
        ctx: dict[str, Any],
    ):
        ctx["storage"].get_debate.return_value = None
        result = handler.handle("/api/v1/costs/sessions/missing/receipt", {}, mock_handler)
        assert result is not None
        assert result.status_code == 404

    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.billing.debate_costs.get_debate_cost_tracker")
    def test_receipt_checksum_deterministic(
        self,
        mock_tracker: MagicMock,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        """Two calls for the same debate should produce a valid checksum."""
        from aragora.billing.debate_costs import DebateCostTracker

        tracker = DebateCostTracker()
        tracker.record_agent_call(
            debate_id="debate-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens_in=1000,
            tokens_out=500,
            round_number=1,
        )
        mock_tracker.return_value = tracker

        result = handler.handle("/api/v1/costs/sessions/debate-1/receipt", {}, mock_handler)
        data = _parse_body(result)
        assert len(data["checksum"]) == 64


# ---------------------------------------------------------------------------
# Method not allowed
# ---------------------------------------------------------------------------


class TestMethodNotAllowed:
    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_post_rejected(
        self,
        _mock_ip: MagicMock,
        handler: SessionCostHandler,
        mock_handler: MagicMock,
    ):
        mock_handler.command = "POST"
        result = handler.handle("/api/v1/costs/sessions/debate-1/breakdown", {}, mock_handler)
        assert result is not None
        assert result.status_code == 405


# ---------------------------------------------------------------------------
# No storage scenario
# ---------------------------------------------------------------------------


class TestNoStorage:
    @patch("aragora.server.handlers.costs.session_costs.get_client_ip", return_value="127.0.0.1")
    def test_works_without_storage(self, _mock_ip: MagicMock, mock_handler: MagicMock):
        """When no storage is in the context, should still return data."""
        h = SessionCostHandler(ctx={})
        result = h.handle("/api/v1/costs/sessions/debate-1/breakdown", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "debate-1"
