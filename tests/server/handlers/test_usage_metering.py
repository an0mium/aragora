"""
Tests for Usage Metering API Handler.

Tests coverage for:
- GET /api/v1/billing/usage - Usage summary
- GET /api/v1/billing/usage/breakdown - Detailed breakdown
- GET /api/v1/billing/limits - Usage limits
- GET /api/v1/billing/usage/export - CSV export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.server.handlers.conftest import parse_handler_response


@dataclass
class MockUsageSummary:
    """Mock usage summary response."""

    period_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    period_end: datetime = field(
        default_factory=lambda: datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
    )
    period_type: str = "month"
    input_tokens: int = 500000
    output_tokens: int = 250000
    total_tokens: int = 750000
    cost: Decimal = field(default_factory=lambda: Decimal("12.50"))
    debates: int = 45
    api_calls: int = 1500

    def to_dict(self):
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_type": self.period_type,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
                "cost": str(self.cost),
            },
            "counts": {
                "debates": self.debates,
                "api_calls": self.api_calls,
            },
        }


@dataclass
class MockUsageBreakdown:
    """Mock usage breakdown response."""

    period_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    period_end: datetime = field(
        default_factory=lambda: datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
    )
    total_cost: Decimal = field(default_factory=lambda: Decimal("125.50"))
    total_tokens: int = 5000000
    total_debates: int = 150
    total_api_calls: int = 5000
    by_model: list = field(default_factory=list)
    by_provider: list = field(default_factory=list)
    by_day: list = field(default_factory=list)
    by_user: list = field(default_factory=list)

    def __post_init__(self):
        if not self.by_model:
            self.by_model = [
                {
                    "model": "claude-3.5-sonnet",
                    "input_tokens": 2000000,
                    "output_tokens": 1000000,
                    "total_tokens": 3000000,
                    "cost": "75.00",
                    "requests": 1000,
                },
            ]
        if not self.by_provider:
            self.by_provider = [
                {
                    "provider": "anthropic",
                    "total_tokens": 3000000,
                    "cost": "75.00",
                    "requests": 1000,
                },
            ]
        if not self.by_day:
            self.by_day = [
                {
                    "day": "2025-01-15",
                    "total_tokens": 100000,
                    "cost": "2.50",
                    "debates": 5,
                    "api_calls": 100,
                },
            ]

    def to_dict(self):
        return {
            "totals": {
                "cost": str(self.total_cost),
                "tokens": self.total_tokens,
                "debates": self.total_debates,
                "api_calls": self.total_api_calls,
            },
            "by_model": self.by_model,
            "by_provider": self.by_provider,
            "by_day": self.by_day,
            "by_user": self.by_user,
        }


@dataclass
class MockUsageLimits:
    """Mock usage limits response."""

    tier: str = "enterprise_plus"
    token_limit: int = 999999999
    debate_limit: int = 999999
    api_call_limit: int = 999999
    tokens_used: int = 750000
    debates_used: int = 45
    api_calls_used: int = 1500

    def to_dict(self):
        return {
            "tier": self.tier,
            "limits": {
                "tokens": self.token_limit,
                "debates": self.debate_limit,
                "api_calls": self.api_call_limit,
            },
            "used": {
                "tokens": self.tokens_used,
                "debates": self.debates_used,
                "api_calls": self.api_calls_used,
            },
            "percent": {
                "tokens": self.tokens_used / self.token_limit,
                "debates": self.debates_used / self.debate_limit,
                "api_calls": self.api_calls_used / self.api_call_limit,
            },
            "exceeded": {
                "tokens": self.tokens_used > self.token_limit,
                "debates": self.debates_used > self.debate_limit,
                "api_calls": self.api_calls_used > self.api_call_limit,
            },
        }


class TestUsageMeteringHandler:
    """Tests for UsageMeteringHandler."""

    @pytest.fixture
    def mock_usage_meter(self):
        """Create mock usage meter."""
        meter = MagicMock()
        meter.get_usage_summary = AsyncMock(return_value=MockUsageSummary())
        meter.get_usage_breakdown = AsyncMock(return_value=MockUsageBreakdown())
        meter.get_usage_limits = AsyncMock(return_value=MockUsageLimits())
        return meter

    @pytest.fixture
    def metering_handler(self, mock_server_context):
        """Create UsageMeteringHandler with mocked context."""
        from aragora.server.handlers.usage_metering import UsageMeteringHandler

        return UsageMeteringHandler(server_context=mock_server_context)

    def test_can_handle_valid_routes(self, metering_handler):
        """Handler recognizes valid routes."""
        assert metering_handler.can_handle("/api/v1/billing/usage") is True
        assert metering_handler.can_handle("/api/v1/billing/usage/breakdown") is True
        assert metering_handler.can_handle("/api/v1/billing/limits") is True
        assert metering_handler.can_handle("/api/v1/billing/usage/summary") is True
        assert metering_handler.can_handle("/api/v1/billing/usage/export") is True

    def test_can_handle_invalid_routes(self, metering_handler):
        """Handler rejects invalid routes."""
        assert metering_handler.can_handle("/api/v1/billing/invoices") is False
        assert metering_handler.can_handle("/api/v1/debates") is False
        assert metering_handler.can_handle("/billing/usage") is False

    def test_get_org_tier_returns_free_for_none(self, metering_handler):
        """_get_org_tier returns 'free' when org is None."""
        assert metering_handler._get_org_tier(None) == "free"

    def test_get_org_tier_handles_string_tier(self, metering_handler):
        """_get_org_tier handles string tier values."""
        org = MagicMock()
        org.tier = "professional"
        assert metering_handler._get_org_tier(org) == "professional"


class TestUsageMeteringRouting:
    """Tests for request routing."""

    @pytest.fixture
    def metering_handler(self, mock_server_context):
        """Create handler with mocked context."""
        from aragora.server.handlers.usage_metering import UsageMeteringHandler

        return UsageMeteringHandler(server_context=mock_server_context)

    @pytest.fixture
    def mock_http(self, mock_http_handler):
        """Create mock HTTP handler."""
        return mock_http_handler(method="GET")

    def test_handle_routes_to_usage(self, metering_handler, mock_http):
        """handle() routes /api/v1/billing/usage to _get_usage."""
        with patch.object(
            metering_handler, "_get_usage", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch("aragora.server.handlers.usage_metering._usage_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                metering_handler.handle("/api/v1/billing/usage", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_breakdown(self, metering_handler, mock_http):
        """handle() routes /api/v1/billing/usage/breakdown to _get_usage_breakdown."""
        with patch.object(
            metering_handler, "_get_usage_breakdown", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch("aragora.server.handlers.usage_metering._usage_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                metering_handler.handle("/api/v1/billing/usage/breakdown", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_routes_to_limits(self, metering_handler, mock_http):
        """handle() routes /api/v1/billing/limits to _get_limits."""
        with patch.object(
            metering_handler, "_get_limits", return_value=MagicMock(status_code=200)
        ) as mock_get:
            with patch("aragora.server.handlers.usage_metering._usage_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True
                metering_handler.handle("/api/v1/billing/limits", {}, mock_http, "GET")
                mock_get.assert_called_once()

    def test_handle_rate_limit_exceeded(self, metering_handler, mock_http):
        """handle() returns 429 when rate limit exceeded."""
        with patch("aragora.server.handlers.usage_metering._usage_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = metering_handler.handle("/api/v1/billing/usage", {}, mock_http, "GET")
            assert result.status_code == 429

    def test_handle_method_not_allowed(self, metering_handler):
        """handle() returns 405 for unsupported method."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"
        mock_http.client_address = ("127.0.0.1", 12345)
        mock_http.headers = {}

        with patch("aragora.server.handlers.usage_metering._usage_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = metering_handler.handle("/api/v1/billing/usage", {}, mock_http, "DELETE")
            assert result.status_code == 405
