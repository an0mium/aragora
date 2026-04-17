"""
Tests for debate session cost calculation API endpoints.

Covers:
- GET /api/v1/costs/debates/{debate_id} - cost summary for a debate
- GET /api/v1/costs/debates/{debate_id}/line-items - line-item breakdown
- GET /api/v1/costs/debates/{debate_id}/performance - performance metrics
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.costs import CostHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    return CostHandler()


@pytest.fixture
def mock_request():
    request = MagicMock(spec=web.Request)
    request.query = {}
    request.match_info = {}
    return request


def _make_usage(
    debate_id="debate_123",
    agent_name="claude",
    agent_id="agent_1",
    provider="anthropic",
    model="claude-opus-4",
    tokens_in=1000,
    tokens_out=500,
    cost_usd="0.025",
    latency_ms=350.0,
    operation="debate_round",
    usage_id="usage_1",
    timestamp=None,
):
    """Build a mock TokenUsage object."""
    usage = MagicMock()
    usage.id = usage_id
    usage.debate_id = debate_id
    usage.agent_name = agent_name
    usage.agent_id = agent_id
    usage.provider = provider
    usage.model = model
    usage.tokens_in = tokens_in
    usage.tokens_out = tokens_out
    usage.tokens_cached = 0
    usage.cost_usd = Decimal(cost_usd)
    usage.latency_ms = latency_ms
    usage.operation = operation
    usage.timestamp = timestamp or datetime.now(timezone.utc)
    usage.metadata = {}
    return usage


def _make_tracker(usage_items=None, debate_cost=None, budget_status=None):
    """Build a mock CostTracker with an async context manager buffer lock."""
    tracker = MagicMock()

    # Create a real asyncio.Lock for _buffer_lock
    tracker._buffer_lock = asyncio.Lock()
    tracker._usage_buffer = usage_items or []

    # get_debate_cost is async
    default_cost = debate_cost or {
        "debate_id": "debate_123",
        "total_cost_usd": "0.075",
        "total_tokens_in": 3000,
        "total_tokens_out": 1500,
        "api_calls": 3,
    }
    tracker.get_debate_cost = AsyncMock(return_value=default_cost)

    # check_debate_budget is sync
    default_budget = budget_status or {
        "within_budget": True,
        "current_cost": "0.075",
        "limit": None,
        "remaining": None,
        "message": "No budget limit set",
    }
    tracker.check_debate_budget = MagicMock(return_value=default_budget)

    return tracker


# ===========================================================================
# Debate cost summary tests
# ===========================================================================


class TestDebateCosts:
    """Tests for GET /api/v1/costs/debates/{debate_id}."""

    @pytest.mark.asyncio
    async def test_get_debate_costs_success(self, handler, mock_request):
        """Returns cost summary for a debate with agent and model breakdowns."""
        now = datetime.now(timezone.utc)
        usages = [
            _make_usage(
                usage_id="u1",
                agent_name="claude",
                model="claude-opus-4",
                cost_usd="0.025",
                tokens_in=1000,
                tokens_out=500,
                latency_ms=300.0,
                timestamp=now,
            ),
            _make_usage(
                usage_id="u2",
                agent_name="gpt",
                agent_id="agent_2",
                provider="openai",
                model="gpt-4o",
                cost_usd="0.015",
                tokens_in=800,
                tokens_out=400,
                latency_ms=250.0,
                timestamp=now,
            ),
            _make_usage(
                usage_id="u3",
                agent_name="claude",
                model="claude-opus-4",
                cost_usd="0.035",
                tokens_in=1200,
                tokens_out=600,
                latency_ms=400.0,
                timestamp=now,
            ),
        ]
        tracker = _make_tracker(usage_items=usages)

        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_costs(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]
        assert data["debate_id"] == "debate_123"
        assert data["api_calls"] == 3
        assert len(data["by_agent"]) == 2
        assert len(data["by_model"]) == 2
        # claude should be first (higher cost)
        assert data["by_agent"][0]["agent"] == "claude"
        assert data["by_agent"][0]["calls"] == 2

    @pytest.mark.asyncio
    async def test_get_debate_costs_no_tracker(self, handler, mock_request):
        """Returns 503 when tracker is unavailable."""
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=None):
            response = await handler.handle_get_debate_costs(mock_request)

        assert response.status == 503

    @pytest.mark.asyncio
    async def test_get_debate_costs_missing_id(self, handler, mock_request):
        """Returns 400 when debate_id is empty."""
        mock_request.match_info = {"debate_id": ""}

        with patch(
            "aragora.server.handlers.costs.models._get_cost_tracker",
            return_value=_make_tracker(),
        ):
            response = await handler.handle_get_debate_costs(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_get_debate_costs_no_usage_data(self, handler, mock_request):
        """Returns zeros when no usage data exists for debate."""
        tracker = _make_tracker(
            usage_items=[],
            debate_cost={
                "debate_id": "debate_empty",
                "total_cost_usd": "0",
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "api_calls": 0,
            },
        )
        mock_request.match_info = {"debate_id": "debate_empty"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_costs(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]
        assert data["by_agent"] == []
        assert data["by_model"] == []
        assert data["avg_latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_get_debate_costs_budget_included(self, handler, mock_request):
        """Budget status is included in the response."""
        budget = {
            "within_budget": True,
            "current_cost": "0.025",
            "limit": "1.00",
            "remaining": "0.975",
            "message": "Within budget",
        }
        tracker = _make_tracker(
            usage_items=[_make_usage()],
            budget_status=budget,
        )
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_costs(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["data"]["budget"]["within_budget"] is True


# ===========================================================================
# Line-items tests
# ===========================================================================


class TestDebateLineItems:
    """Tests for GET /api/v1/costs/debates/{debate_id}/line-items."""

    @pytest.mark.asyncio
    async def test_line_items_success(self, handler, mock_request):
        """Returns individual API call line items."""
        now = datetime.now(timezone.utc)
        usages = [
            _make_usage(
                usage_id=f"u{i}",
                cost_usd=f"0.0{i + 1}0",
                tokens_in=1000 * (i + 1),
                tokens_out=500 * (i + 1),
                timestamp=now + timedelta(seconds=i),
            )
            for i in range(3)
        ]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}
        mock_request.query = {}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]
        assert data["debate_id"] == "debate_123"
        assert data["total_count"] == 3
        assert data["returned_count"] == 3
        assert len(data["line_items"]) == 3
        # Each line item should have required fields
        item = data["line_items"][0]
        assert "id" in item
        assert "provider" in item
        assert "model" in item
        assert "tokens_in" in item
        assert "tokens_out" in item
        assert "cost_usd" in item
        assert "latency_ms" in item

    @pytest.mark.asyncio
    async def test_line_items_sort_by_cost(self, handler, mock_request):
        """Line items can be sorted by cost."""
        usages = [
            _make_usage(usage_id="u1", cost_usd="0.010"),
            _make_usage(usage_id="u2", cost_usd="0.050"),
            _make_usage(usage_id="u3", cost_usd="0.030"),
        ]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}
        mock_request.query = {"sort_by": "cost", "order": "desc"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        items = body["data"]["line_items"]
        assert items[0]["cost_usd"] >= items[1]["cost_usd"] >= items[2]["cost_usd"]

    @pytest.mark.asyncio
    async def test_line_items_pagination(self, handler, mock_request):
        """Line items support limit/offset pagination."""
        usages = [_make_usage(usage_id=f"u{i}") for i in range(10)]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}
        mock_request.query = {"limit": "3", "offset": "2"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]
        assert data["total_count"] == 10
        assert data["returned_count"] == 3
        assert data["offset"] == 2
        assert data["limit"] == 3

    @pytest.mark.asyncio
    async def test_line_items_invalid_sort(self, handler, mock_request):
        """Returns 400 for invalid sort_by parameter."""
        mock_request.match_info = {"debate_id": "debate_123"}
        mock_request.query = {"sort_by": "invalid"}
        tracker = _make_tracker()

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_line_items_invalid_order(self, handler, mock_request):
        """Returns 400 for invalid order parameter."""
        mock_request.match_info = {"debate_id": "debate_123"}
        mock_request.query = {"order": "invalid"}
        tracker = _make_tracker()

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_line_items_no_tracker(self, handler, mock_request):
        """Returns 503 when tracker is unavailable."""
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=None):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 503

    @pytest.mark.asyncio
    async def test_line_items_empty(self, handler, mock_request):
        """Returns empty list when debate has no usage."""
        tracker = _make_tracker(usage_items=[])
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["data"]["line_items"] == []
        assert body["data"]["total_count"] == 0

    @pytest.mark.asyncio
    async def test_line_items_filters_by_debate(self, handler, mock_request):
        """Only returns items matching the requested debate_id."""
        usages = [
            _make_usage(usage_id="u1", debate_id="debate_123"),
            _make_usage(usage_id="u2", debate_id="other_debate"),
            _make_usage(usage_id="u3", debate_id="debate_123"),
        ]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_line_items(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["data"]["total_count"] == 2


# ===========================================================================
# Performance tests
# ===========================================================================


class TestDebatePerformance:
    """Tests for GET /api/v1/costs/debates/{debate_id}/performance."""

    @pytest.mark.asyncio
    async def test_performance_success(self, handler, mock_request):
        """Returns performance metrics for a debate."""
        base = datetime.now(timezone.utc)
        usages = [
            _make_usage(
                usage_id="u1",
                cost_usd="0.025",
                latency_ms=300.0,
                tokens_in=1000,
                tokens_out=500,
                operation="proposal",
                timestamp=base,
            ),
            _make_usage(
                usage_id="u2",
                cost_usd="0.015",
                latency_ms=200.0,
                tokens_in=800,
                tokens_out=400,
                operation="critique",
                timestamp=base + timedelta(seconds=30),
            ),
            _make_usage(
                usage_id="u3",
                cost_usd="0.035",
                latency_ms=500.0,
                tokens_in=1200,
                tokens_out=600,
                operation="proposal",
                timestamp=base + timedelta(seconds=60),
            ),
        ]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_performance(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]

        assert data["debate_id"] == "debate_123"
        assert data["api_calls"] == 3
        assert data["total_tokens"] == (1000 + 500 + 800 + 400 + 1200 + 600)

        # Latency metrics
        lat = data["latency"]
        assert lat["min_ms"] == 200.0
        assert lat["max_ms"] == 500.0
        assert lat["avg_ms"] == pytest.approx(333.33, rel=0.01)

        # Cost efficiency
        eff = data["cost_efficiency"]
        assert eff["cost_per_call_usd"] == pytest.approx(0.025, rel=0.01)
        assert eff["avg_tokens_per_call"] > 0

        # Per-operation breakdown
        assert len(data["by_operation"]) == 2
        ops = {op["operation"]: op for op in data["by_operation"]}
        assert ops["proposal"]["calls"] == 2
        assert ops["critique"]["calls"] == 1

        # Time range
        assert data["time_range"]["start"] is not None
        assert data["time_range"]["end"] is not None

        # Duration
        assert data["duration_seconds"] == pytest.approx(60.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_performance_no_data(self, handler, mock_request):
        """Returns message when no usage data exists."""
        tracker = _make_tracker(usage_items=[])
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_performance(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]
        assert data["api_calls"] == 0
        assert "message" in data

    @pytest.mark.asyncio
    async def test_performance_no_tracker(self, handler, mock_request):
        """Returns 503 when tracker is unavailable."""
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=None):
            response = await handler.handle_get_debate_performance(mock_request)

        assert response.status == 503

    @pytest.mark.asyncio
    async def test_performance_missing_debate_id(self, handler, mock_request):
        """Returns 400 when debate_id is empty."""
        mock_request.match_info = {"debate_id": ""}

        with patch(
            "aragora.server.handlers.costs.models._get_cost_tracker",
            return_value=_make_tracker(),
        ):
            response = await handler.handle_get_debate_performance(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_performance_throughput(self, handler, mock_request):
        """Throughput metrics are computed correctly."""
        base = datetime.now(timezone.utc)
        # 6 calls over 2 minutes = 3 calls/min
        usages = [
            _make_usage(
                usage_id=f"u{i}",
                tokens_in=100,
                tokens_out=50,
                latency_ms=100.0,
                timestamp=base + timedelta(seconds=i * 20),
            )
            for i in range(6)
        ]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_performance(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        tp = body["data"]["throughput"]
        # 100 seconds duration, 6 calls = 3.6 calls/min
        assert tp["calls_per_minute"] > 0
        assert tp["tokens_per_minute"] > 0

    @pytest.mark.asyncio
    async def test_performance_single_call(self, handler, mock_request):
        """Performance works with a single API call (duration=0)."""
        usages = [_make_usage(usage_id="u1")]
        tracker = _make_tracker(usage_items=usages)
        mock_request.match_info = {"debate_id": "debate_123"}

        with patch("aragora.server.handlers.costs.models._get_cost_tracker", return_value=tracker):
            response = await handler.handle_get_debate_performance(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        data = body["data"]
        assert data["api_calls"] == 1
        assert data["duration_seconds"] == 0
        assert data["throughput"]["calls_per_minute"] == 0


# ===========================================================================
# Route registration tests
# ===========================================================================


class TestDebateCostRoutes:
    """Tests for route registration of debate cost endpoints."""

    def test_routes_registered(self):
        """All debate cost routes are in the ROUTES list."""
        routes = CostHandler.ROUTES
        assert "/api/v1/costs/debates/{debate_id}" in routes
        assert "/api/v1/costs/debates/{debate_id}/line-items" in routes
        assert "/api/v1/costs/debates/{debate_id}/performance" in routes
        assert "/api/costs/debates/{debate_id}" in routes
        assert "/api/costs/debates/{debate_id}/line-items" in routes
        assert "/api/costs/debates/{debate_id}/performance" in routes

    def test_can_handle_debate_paths(self):
        """CostHandler.can_handle returns True for debate cost paths."""
        handler = CostHandler()
        assert handler.can_handle("/api/v1/costs/debates/abc123")
        assert handler.can_handle("/api/v1/costs/debates/abc123/line-items")
        assert handler.can_handle("/api/v1/costs/debates/abc123/performance")
        assert handler.can_handle("/api/costs/debates/abc123")

    def test_resolve_registry_target(self):
        """Dynamic route resolution maps debate paths to correct handlers."""
        resolved = CostHandler._resolve_registry_target("/api/v1/costs/debates/abc123", "GET")
        assert resolved is not None
        handler_name, params = resolved
        assert handler_name == "handle_get_debate_costs"
        assert params["debate_id"] == "abc123"

        resolved = CostHandler._resolve_registry_target(
            "/api/v1/costs/debates/abc123/line-items", "GET"
        )
        assert resolved is not None
        handler_name, params = resolved
        assert handler_name == "handle_get_debate_line_items"
        assert params["debate_id"] == "abc123"

        resolved = CostHandler._resolve_registry_target(
            "/api/v1/costs/debates/abc123/performance", "GET"
        )
        assert resolved is not None
        handler_name, params = resolved
        assert handler_name == "handle_get_debate_performance"
        assert params["debate_id"] == "abc123"
