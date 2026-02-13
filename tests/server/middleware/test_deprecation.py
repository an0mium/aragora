"""
Tests for aragora.server.middleware.deprecation - V1 API Sunset Deprecation Middleware.

Tests cover:
- V1 path detection (is_v1_request)
- Deprecation header generation (get_v1_deprecation_headers)
- Header injection (inject_v1_deprecation_headers)
- V1UsageTracker statistics and logging
- Middleware enable/disable configuration
- aiohttp middleware (v1_sunset_middleware)
- BaseHTTPRequestHandler support (add_v1_headers_to_handler)
- RFC 8594 Sunset/Deprecation header compliance
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import aiohttp  # noqa: F401


# =============================================================================
# Test V1 Path Detection
# =============================================================================


class TestIsV1Request:
    """Tests for is_v1_request function."""

    def test_v1_api_path_detected(self):
        """Should detect /api/v1/ paths."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v1/debates") is True
        assert is_v1_request("/api/v1/agents") is True
        assert is_v1_request("/api/v1/users/123") is True

    def test_v1_api_root_detected(self):
        """Should detect /api/v1 root path."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v1") is True
        assert is_v1_request("/api/v1/") is True

    def test_v2_api_not_detected(self):
        """Should not detect /api/v2/ paths."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v2/debates") is False
        assert is_v1_request("/api/v2/agents") is False

    def test_non_api_paths_not_detected(self):
        """Should not detect non-API paths."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/health") is False
        assert is_v1_request("/api/debates") is False
        assert is_v1_request("/static/v1/file.js") is False

    def test_partial_v1_not_detected(self):
        """Should not detect partial v1 matches."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v10/test") is False
        assert is_v1_request("/api/v1test") is False
        assert is_v1_request("/apiv1/test") is False

    def test_v1_prefix_only(self):
        """Should only match paths starting with /api/v1."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/other/api/v1/test") is False
        assert is_v1_request("/v1/api/test") is False


# =============================================================================
# Test Deprecation Header Generation
# =============================================================================


class TestGetV1DeprecationHeaders:
    """Tests for get_v1_deprecation_headers function."""

    def test_returns_sunset_header(self):
        """Should return Sunset header with HTTP-date format."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "Sunset" in headers
        # Should be HTTP-date format (RFC 7231)
        assert "2026" in headers["Sunset"]
        assert "GMT" in headers["Sunset"]

    def test_returns_deprecation_header(self):
        """Should return Deprecation header with timestamp."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "Deprecation" in headers
        # RFC 8594 format: @<timestamp>
        assert headers["Deprecation"].startswith("@")

    def test_returns_link_header(self):
        """Should return Link header with migration docs."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "Link" in headers
        assert 'rel="sunset"' in headers["Link"]
        assert "docs.aragora.ai" in headers["Link"]

    def test_returns_x_api_version(self):
        """Should return X-API-Version header."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "X-API-Version" in headers
        assert headers["X-API-Version"] == "v1"

    def test_returns_x_api_version_warning(self):
        """Should return X-API-Version-Warning header."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "X-API-Version-Warning" in headers
        assert "deprecated" in headers["X-API-Version-Warning"].lower()
        assert "v2" in headers["X-API-Version-Warning"]

    def test_returns_x_api_sunset(self):
        """Should return X-API-Sunset header with ISO date."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "X-API-Sunset" in headers
        assert "2026-06-01" in headers["X-API-Sunset"]

    def test_returns_x_deprecation_level(self):
        """Should return X-Deprecation-Level header."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        assert "X-Deprecation-Level" in headers
        assert headers["X-Deprecation-Level"] in ("warning", "critical", "sunset")

    def test_link_header_includes_v2_equivalent(self):
        """Should include v2 equivalent path in Link header when path provided."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers(path="/api/v1/debates/123")

        assert "Link" in headers
        assert 'rel="successor-version"' in headers["Link"]
        assert "/api/v2/debates/123" in headers["Link"]


class TestDeprecationLevel:
    """Tests for deprecation level calculation."""

    def test_deprecation_level_warning(self):
        """Should return 'warning' when far from sunset."""
        from aragora.server.versioning.constants import deprecation_level, days_until_v1_sunset

        # If more than 30 days until sunset
        if days_until_v1_sunset() > 30:
            assert deprecation_level() == "warning"

    def test_deprecation_level_critical(self):
        """Should return 'critical' within 30 days of sunset."""
        from aragora.server.versioning.constants import deprecation_level, days_until_v1_sunset

        # If within 30 days
        if 0 < days_until_v1_sunset() < 30:
            assert deprecation_level() == "critical"

    def test_deprecation_level_sunset(self):
        """Should return 'sunset' after sunset date."""
        from aragora.server.versioning.constants import deprecation_level, is_v1_sunset

        # If past sunset
        if is_v1_sunset():
            assert deprecation_level() == "sunset"


# =============================================================================
# Test Header Injection
# =============================================================================


class TestInjectV1DeprecationHeaders:
    """Tests for inject_v1_deprecation_headers function."""

    def test_injects_headers_into_dict(self):
        """Should inject all deprecation headers into existing dict."""
        from aragora.server.middleware.deprecation import inject_v1_deprecation_headers

        response_headers = {"Content-Type": "application/json"}
        result = inject_v1_deprecation_headers(response_headers)

        assert result is response_headers  # Same dict
        assert "Sunset" in result
        assert "Deprecation" in result
        assert "Link" in result
        assert "Content-Type" in result  # Preserved

    def test_injects_headers_with_path(self):
        """Should inject headers with v2 equivalent link when path provided."""
        from aragora.server.middleware.deprecation import inject_v1_deprecation_headers

        response_headers = {}
        inject_v1_deprecation_headers(response_headers, path="/api/v1/debates")

        assert "/api/v2/debates" in response_headers["Link"]

    def test_modifies_dict_in_place(self):
        """Should modify the dict in place."""
        from aragora.server.middleware.deprecation import inject_v1_deprecation_headers

        response_headers = {}
        inject_v1_deprecation_headers(response_headers)

        # Original dict should be modified
        assert "Sunset" in response_headers


# =============================================================================
# Test V1UsageTracker
# =============================================================================


class TestV1UsageTracker:
    """Tests for V1UsageTracker class."""

    def test_tracker_initializes_with_zero_counts(self):
        """Should initialize with zero counts."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()

        assert tracker.total_requests == 0
        assert tracker.last_request_time == 0.0

    def test_record_increments_total(self):
        """Should increment total request count."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker.record("/api/v1/debates", "GET")
        tracker.record("/api/v1/agents", "POST")

        assert tracker.total_requests == 2

    def test_record_tracks_by_path(self):
        """Should track requests by path."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker.record("/api/v1/debates", "GET")
        tracker.record("/api/v1/debates", "GET")
        tracker.record("/api/v1/agents", "GET")

        assert tracker.requests_by_path["/api/v1/debates"] == 2
        assert tracker.requests_by_path["/api/v1/agents"] == 1

    def test_record_tracks_by_method(self):
        """Should track requests by HTTP method."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker.record("/api/v1/debates", "GET")
        tracker.record("/api/v1/debates", "POST")
        tracker.record("/api/v1/debates", "get")  # Lowercase

        assert tracker.requests_by_method["GET"] == 2
        assert tracker.requests_by_method["POST"] == 1

    def test_record_updates_last_request_time(self):
        """Should update last request time."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        before = time.time()
        tracker.record("/api/v1/test", "GET")
        after = time.time()

        assert before <= tracker.last_request_time <= after

    def test_get_stats_returns_complete_info(self):
        """Should return complete statistics."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker.record("/api/v1/debates", "GET")
        tracker.record("/api/v1/debates", "POST")

        stats = tracker.get_stats()

        assert stats["total_v1_requests"] == 2
        assert len(stats["top_endpoints"]) > 0
        assert "GET" in stats["requests_by_method"]
        assert "days_until_sunset" in stats
        assert "deprecation_level" in stats
        assert "sunset_date" in stats

    def test_get_stats_top_endpoints_sorted(self):
        """Should return top endpoints sorted by count."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        for _ in range(10):
            tracker.record("/api/v1/popular", "GET")
        for _ in range(5):
            tracker.record("/api/v1/medium", "GET")
        for _ in range(1):
            tracker.record("/api/v1/rare", "GET")

        stats = tracker.get_stats()
        endpoints = stats["top_endpoints"]

        assert endpoints[0]["path"] == "/api/v1/popular"
        assert endpoints[0]["count"] == 10


class TestGlobalUsageTracker:
    """Tests for global usage tracker functions."""

    def test_get_v1_usage_tracker(self):
        """Should return the global tracker instance."""
        from aragora.server.middleware.deprecation import (
            V1UsageTracker,
            get_v1_usage_tracker,
        )

        tracker = get_v1_usage_tracker()
        assert isinstance(tracker, V1UsageTracker)

    def test_reset_v1_usage_tracker(self):
        """Should reset the global tracker."""
        from aragora.server.middleware.deprecation import (
            get_v1_usage_tracker,
            reset_v1_usage_tracker,
        )

        # Record some data
        tracker = get_v1_usage_tracker()
        tracker.record("/api/v1/test", "GET")

        # Reset
        reset_v1_usage_tracker()

        # Should be fresh
        new_tracker = get_v1_usage_tracker()
        assert new_tracker.total_requests == 0


# =============================================================================
# Test Middleware Configuration
# =============================================================================


class TestMiddlewareConfiguration:
    """Tests for middleware enable/disable configuration."""

    def test_middleware_enabled_by_default(self):
        """Should be enabled by default."""
        from aragora.server.middleware.deprecation import is_deprecation_middleware_enabled

        with patch.dict(os.environ, {}, clear=True):
            assert is_deprecation_middleware_enabled() is True

    def test_middleware_disabled_via_env(self):
        """Should be disabled when env var is set."""
        from aragora.server.middleware.deprecation import is_deprecation_middleware_enabled

        with patch.dict(os.environ, {"ARAGORA_DISABLE_V1_DEPRECATION": "true"}, clear=True):
            assert is_deprecation_middleware_enabled() is False

    def test_middleware_disabled_various_values(self):
        """Should recognize various truthy values for disabling."""
        from aragora.server.middleware.deprecation import is_deprecation_middleware_enabled

        for value in ["true", "TRUE", "1", "yes"]:
            with patch.dict(os.environ, {"ARAGORA_DISABLE_V1_DEPRECATION": value}, clear=True):
                assert is_deprecation_middleware_enabled() is False

    def test_middleware_enabled_with_false_values(self):
        """Should be enabled when env var is false."""
        from aragora.server.middleware.deprecation import is_deprecation_middleware_enabled

        for value in ["false", "FALSE", "0", "no"]:
            with patch.dict(os.environ, {"ARAGORA_DISABLE_V1_DEPRECATION": value}, clear=True):
                assert is_deprecation_middleware_enabled() is True


# =============================================================================
# Test BaseHTTPRequestHandler Support
# =============================================================================


class TestAddV1HeadersToHandler:
    """Tests for add_v1_headers_to_handler function."""

    def test_adds_headers_for_v1_path(self):
        """Should add headers for v1 API paths."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock()
        handler.send_header = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            add_v1_headers_to_handler(handler, "/api/v1/debates")

        # Should have called send_header multiple times
        assert handler.send_header.call_count > 0

        # Check that key headers were sent
        header_names = [call[0][0] for call in handler.send_header.call_args_list]
        assert "Sunset" in header_names
        assert "Deprecation" in header_names

    def test_skips_non_v1_path(self):
        """Should not add headers for non-v1 paths."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock()
        handler.send_header = MagicMock()

        add_v1_headers_to_handler(handler, "/api/v2/debates")

        handler.send_header.assert_not_called()

    def test_skips_when_disabled(self):
        """Should not add headers when middleware is disabled."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock()
        handler.send_header = MagicMock()

        with patch.dict(os.environ, {"ARAGORA_DISABLE_V1_DEPRECATION": "true"}, clear=True):
            add_v1_headers_to_handler(handler, "/api/v1/debates")

        handler.send_header.assert_not_called()

    def test_handles_handler_without_send_header(self):
        """Should handle handler without send_header method."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock(spec=[])  # No send_header

        # Should not raise
        with patch.dict(os.environ, {}, clear=True):
            add_v1_headers_to_handler(handler, "/api/v1/debates")


# =============================================================================
# Test aiohttp Middleware
# =============================================================================


class TestAiohttpMiddleware:
    """Tests for aiohttp v1_sunset_middleware."""

    @pytest.mark.asyncio
    async def test_middleware_skips_non_v1(self):
        """Should pass through non-v1 requests without modification."""
        from aragora.server.middleware.deprecation import v1_sunset_middleware

        mock_request = MagicMock()
        mock_request.path = "/api/v2/debates"

        mock_response = MagicMock()
        mock_handler = AsyncMock(return_value=mock_response)

        result = await v1_sunset_middleware(mock_request, mock_handler)

        assert result == mock_response
        mock_handler.assert_awaited_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_middleware_adds_headers_for_v1(self):
        """Should add deprecation headers for v1 requests."""
        from aragora.server.middleware.deprecation import (
            reset_v1_usage_tracker,
            v1_sunset_middleware,
        )

        reset_v1_usage_tracker()

        mock_request = MagicMock()
        mock_request.path = "/api/v1/debates"
        mock_request.method = "GET"

        mock_headers = MagicMock()
        mock_headers.get = lambda k, d="": d
        mock_request.headers = mock_headers

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_handler = AsyncMock(return_value=mock_response)

        with patch.dict(os.environ, {}, clear=True):
            result = await v1_sunset_middleware(mock_request, mock_handler)

        assert "Sunset" in result.headers
        assert "Deprecation" in result.headers

    @pytest.mark.asyncio
    async def test_middleware_tracks_usage(self):
        """Should track v1 usage."""
        from aragora.server.middleware.deprecation import (
            get_v1_usage_tracker,
            reset_v1_usage_tracker,
            v1_sunset_middleware,
        )

        reset_v1_usage_tracker()

        mock_request = MagicMock()
        mock_request.path = "/api/v1/test"
        mock_request.method = "POST"

        mock_headers = MagicMock()
        mock_headers.get = lambda k, d="": d
        mock_request.headers = mock_headers

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_handler = AsyncMock(return_value=mock_response)

        with patch.dict(os.environ, {}, clear=True):
            await v1_sunset_middleware(mock_request, mock_handler)

        tracker = get_v1_usage_tracker()
        assert tracker.total_requests >= 1

    @pytest.mark.asyncio
    async def test_middleware_skips_when_disabled(self):
        """Should skip when disabled via env var."""
        from aragora.server.middleware.deprecation import v1_sunset_middleware

        mock_request = MagicMock()
        mock_request.path = "/api/v1/debates"

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_handler = AsyncMock(return_value=mock_response)

        with patch.dict(os.environ, {"ARAGORA_DISABLE_V1_DEPRECATION": "true"}, clear=True):
            result = await v1_sunset_middleware(mock_request, mock_handler)

        # Headers should not be added when disabled
        assert "Sunset" not in result.headers


class TestCreateV1SunsetMiddleware:
    """Tests for create_v1_sunset_middleware factory."""

    def test_returns_middleware_function(self):
        """Should return the middleware function."""
        from aragora.server.middleware.deprecation import (
            create_v1_sunset_middleware,
            v1_sunset_middleware,
        )

        middleware = create_v1_sunset_middleware()
        assert middleware is v1_sunset_middleware


# =============================================================================
# Test RFC 8594 Compliance
# =============================================================================


class TestRFC8594Compliance:
    """Tests for RFC 8594 Sunset/Deprecation header compliance."""

    def test_sunset_header_http_date_format(self):
        """Sunset header should be in HTTP-date format (RFC 7231)."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        # HTTP-date format: "Day, DD Mon YYYY HH:MM:SS GMT"
        sunset = headers["Sunset"]
        assert "GMT" in sunset
        # Should have day name
        assert any(day in sunset for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    def test_deprecation_header_timestamp_format(self):
        """Deprecation header should have @<timestamp> format."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        deprecation = headers["Deprecation"]
        assert deprecation.startswith("@")
        # Should be a valid timestamp
        timestamp = int(deprecation[1:])
        assert timestamp > 0

    def test_link_header_rel_sunset(self):
        """Link header should have rel="sunset" relation."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        link = headers["Link"]
        assert 'rel="sunset"' in link


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_core_exports_accessible(self):
        """Core exports should be accessible."""
        from aragora.server.middleware.deprecation import (
            V1UsageTracker,
            add_v1_headers_to_handler,
            get_v1_deprecation_headers,
            get_v1_usage_tracker,
            inject_v1_deprecation_headers,
            is_deprecation_middleware_enabled,
            is_v1_request,
            reset_v1_usage_tracker,
        )

        assert get_v1_deprecation_headers is not None
        assert inject_v1_deprecation_headers is not None
        assert is_v1_request is not None
        assert V1UsageTracker is not None
        assert get_v1_usage_tracker is not None
        assert reset_v1_usage_tracker is not None
        assert is_deprecation_middleware_enabled is not None
        assert add_v1_headers_to_handler is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_is_v1_request_empty_path(self):
        """Should handle empty path."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("") is False

    def test_is_v1_request_root_path(self):
        """Should handle root path."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/") is False

    def test_tracker_client_info_truncation(self):
        """Should truncate long client info."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        long_client_info = "x" * 200

        # Should not raise
        tracker.record("/api/v1/test", "GET", client_info=long_client_info)

    def test_inject_headers_empty_dict(self):
        """Should work with empty headers dict."""
        from aragora.server.middleware.deprecation import inject_v1_deprecation_headers

        headers = {}
        inject_v1_deprecation_headers(headers)

        assert "Sunset" in headers

    def test_get_stats_empty_tracker(self):
        """Should return valid stats for empty tracker."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        stats = tracker.get_stats()

        assert stats["total_v1_requests"] == 0
        assert stats["top_endpoints"] == []

    def test_handler_without_headers_attribute(self):
        """Should handle handler without headers attribute."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        class MinimalHandler:
            def send_header(self, name, value):
                pass

        handler = MinimalHandler()

        with patch.dict(os.environ, {}, clear=True):
            # Should not raise
            add_v1_headers_to_handler(handler, "/api/v1/test")

    def test_deprecation_level_with_future_sunset(self):
        """Should return warning level when sunset is far in future."""
        from aragora.server.versioning.constants import (
            days_until_v1_sunset,
        )

        # The sunset date is 2026-06-01
        # Test is running before that date
        days = days_until_v1_sunset()
        assert days >= 0  # Should not be negative before sunset

    def test_get_headers_with_query_string_path(self):
        """Should handle path with query string."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers(path="/api/v1/debates?page=1")

        # Should still generate valid headers
        assert "Sunset" in headers
        assert "Link" in headers


# =============================================================================
# Test Versioning Constants Integration
# =============================================================================


class TestVersioningConstants:
    """Tests for versioning constants integration."""

    def test_constants_imported(self):
        """Should import constants from versioning module."""
        from aragora.server.versioning.constants import (
            CURRENT_API_VERSION,
            MIGRATION_DOCS_URL,
            V1_DEPRECATION_TIMESTAMP,
            V1_SUNSET_HTTP_DATE,
            V1_SUNSET_ISO,
        )

        assert CURRENT_API_VERSION == "v2"
        assert V1_SUNSET_ISO == "2026-06-01"
        assert "GMT" in V1_SUNSET_HTTP_DATE
        assert V1_DEPRECATION_TIMESTAMP > 0
        assert "docs.aragora.ai" in MIGRATION_DOCS_URL

    def test_headers_use_constants(self):
        """Headers should use constants from versioning module."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers
        from aragora.server.versioning.constants import V1_SUNSET_HTTP_DATE, V1_SUNSET_ISO

        headers = get_v1_deprecation_headers()

        assert headers["Sunset"] == V1_SUNSET_HTTP_DATE
        assert headers["X-API-Sunset"] == V1_SUNSET_ISO


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestUsageTrackerSummaryLogging:
    """Tests for V1UsageTracker periodic summary logging."""

    def test_summary_logged_every_n_requests(self):
        """Should log summary every _log_interval requests."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker._log_interval = 10

        with patch("aragora.server.middleware.deprecation.logger") as mock_logger:
            for i in range(10):
                tracker.record(f"/api/v1/test{i}", "GET")

            # Should have logged 10 warnings (one per record) + 1 info (summary)
            assert mock_logger.info.call_count == 1

    def test_summary_not_logged_before_interval(self):
        """Should not log summary before reaching interval."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker._log_interval = 100

        with patch("aragora.server.middleware.deprecation.logger") as mock_logger:
            for i in range(5):
                tracker.record("/api/v1/test", "GET")

            assert mock_logger.info.call_count == 0

    def test_summary_top_paths_limited(self):
        """Summary should show top 10 paths."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker._log_interval = 50

        with patch("aragora.server.middleware.deprecation.logger"):
            for i in range(50):
                tracker.record(f"/api/v1/endpoint{i}", "GET")

        # get_stats returns top 20
        stats = tracker.get_stats()
        assert len(stats["top_endpoints"]) <= 20

    def test_record_with_client_info(self):
        """Should accept and log client info."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()

        with patch("aragora.server.middleware.deprecation.logger") as mock_logger:
            tracker.record("/api/v1/test", "GET", client_info="my-client-app")

            call_args = mock_logger.warning.call_args
            assert "my-client-app" in str(call_args)


class TestGetV1DeprecationHeadersExtended:
    """Extended tests for header generation edge cases."""

    def test_headers_without_path_no_successor_version(self):
        """Without path, Link should not include successor-version."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()
        assert 'rel="successor-version"' not in headers["Link"]

    def test_headers_with_non_v1_path_no_successor(self):
        """With path not starting with /api/v1/, no successor link."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers(path="/api/v2/debates")
        assert 'rel="successor-version"' not in headers["Link"]

    def test_headers_v2_equivalent_for_nested_path(self):
        """Should compute correct v2 equivalent for nested paths."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers(path="/api/v1/debates/123/rounds/4")
        assert "/api/v2/debates/123/rounds/4" in headers["Link"]

    def test_all_required_headers_present(self):
        """All 7 required deprecation headers should be present."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        headers = get_v1_deprecation_headers()

        required = {
            "Sunset",
            "Deprecation",
            "Link",
            "X-API-Version",
            "X-API-Version-Warning",
            "X-API-Sunset",
            "X-Deprecation-Level",
        }
        assert required.issubset(headers.keys())

    def test_critical_level_urgent_warning(self):
        """Should produce urgent warning at critical level."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        with patch(
            "aragora.server.middleware.deprecation.deprecation_level", return_value="critical"
        ):
            with patch(
                "aragora.server.middleware.deprecation.days_until_v1_sunset", return_value=15
            ):
                headers = get_v1_deprecation_headers()

                assert "URGENT" in headers["X-API-Version-Warning"]
                assert "15 days" in headers["X-API-Version-Warning"]

    def test_sunset_level_warning_message(self):
        """Should produce sunset warning at sunset level."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        with patch(
            "aragora.server.middleware.deprecation.deprecation_level", return_value="sunset"
        ):
            with patch(
                "aragora.server.middleware.deprecation.days_until_v1_sunset", return_value=-5
            ):
                headers = get_v1_deprecation_headers()

                assert "passed its sunset date" in headers["X-API-Version-Warning"]

    def test_warning_level_standard_message(self):
        """Should produce standard warning at warning level."""
        from aragora.server.middleware.deprecation import get_v1_deprecation_headers

        with patch(
            "aragora.server.middleware.deprecation.deprecation_level", return_value="warning"
        ):
            with patch(
                "aragora.server.middleware.deprecation.days_until_v1_sunset", return_value=120
            ):
                headers = get_v1_deprecation_headers()

                assert "deprecated" in headers["X-API-Version-Warning"].lower()
                assert "URGENT" not in headers["X-API-Version-Warning"]


class TestIsV1RequestExtended:
    """Extended tests for v1 path detection."""

    def test_v1_with_trailing_slash(self):
        """Should detect /api/v1/ with trailing slash."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v1/") is True

    def test_v1_with_deep_nesting(self):
        """Should detect deeply nested v1 paths."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v1/debates/123/rounds/4/votes") is True

    def test_v1_case_sensitive(self):
        """Path matching should be case-sensitive."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/API/V1/debates") is False
        assert is_v1_request("/Api/V1/debates") is False

    def test_v1_with_special_characters(self):
        """Should handle special characters in path."""
        from aragora.server.middleware.deprecation import is_v1_request

        assert is_v1_request("/api/v1/debates?q=test&page=1") is True


class TestInjectV1DeprecationHeadersExtended:
    """Extended tests for header injection."""

    def test_inject_overwrites_existing_headers(self):
        """Should overwrite existing deprecation headers."""
        from aragora.server.middleware.deprecation import inject_v1_deprecation_headers

        headers = {"Sunset": "old-value", "X-API-Version": "old"}
        inject_v1_deprecation_headers(headers)

        assert headers["Sunset"] != "old-value"
        assert headers["X-API-Version"] == "v1"

    def test_inject_preserves_non_deprecation_headers(self):
        """Should not remove non-deprecation headers."""
        from aragora.server.middleware.deprecation import inject_v1_deprecation_headers

        headers = {
            "Content-Type": "application/json",
            "X-Custom-Header": "custom",
        }
        inject_v1_deprecation_headers(headers)

        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom-Header"] == "custom"
        assert "Sunset" in headers


class TestUsageTrackerStats:
    """Tests for V1UsageTracker.get_stats edge cases."""

    def test_get_stats_includes_all_fields(self):
        """get_stats should include all expected fields."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        stats = tracker.get_stats()

        required_fields = {
            "total_v1_requests",
            "top_endpoints",
            "requests_by_method",
            "tracking_since",
            "last_request",
            "days_until_sunset",
            "deprecation_level",
            "sunset_date",
        }
        assert required_fields.issubset(stats.keys())

    def test_get_stats_tracking_since_set_at_init(self):
        """tracking_since should be set at initialization time."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        before = time.time()
        tracker = V1UsageTracker()
        after = time.time()

        stats = tracker.get_stats()
        assert before <= stats["tracking_since"] <= after

    def test_get_stats_last_request_zero_initially(self):
        """last_request should be 0 before any requests."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        stats = tracker.get_stats()
        assert stats["last_request"] == 0.0

    def test_get_stats_methods_are_regular_dict(self):
        """requests_by_method should be a regular dict."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        tracker.record("/api/v1/test", "GET")
        tracker.record("/api/v1/test", "POST")

        stats = tracker.get_stats()
        assert isinstance(stats["requests_by_method"], dict)
        assert stats["requests_by_method"]["GET"] == 1
        assert stats["requests_by_method"]["POST"] == 1

    def test_top_endpoints_limited_to_20(self):
        """Top endpoints should be limited to 20."""
        from aragora.server.middleware.deprecation import V1UsageTracker

        tracker = V1UsageTracker()
        for i in range(30):
            tracker.record(f"/api/v1/endpoint{i}", "GET")

        stats = tracker.get_stats()
        assert len(stats["top_endpoints"]) == 20


class TestAddV1HeadersToHandlerExtended:
    """Extended tests for BaseHTTPRequestHandler support."""

    def test_adds_all_seven_headers(self):
        """Should add all 7 deprecation headers."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock()
        handler.send_header = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            add_v1_headers_to_handler(handler, "/api/v1/debates")

        assert handler.send_header.call_count == 7

        header_names = [call[0][0] for call in handler.send_header.call_args_list]
        assert "Sunset" in header_names
        assert "Deprecation" in header_names
        assert "Link" in header_names
        assert "X-API-Version" in header_names
        assert "X-API-Version-Warning" in header_names
        assert "X-API-Sunset" in header_names
        assert "X-Deprecation-Level" in header_names

    def test_skips_root_path(self):
        """Should not add headers for root path."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock()
        add_v1_headers_to_handler(handler, "/")
        handler.send_header.assert_not_called()

    def test_skips_health_endpoint(self):
        """Should not add headers for health endpoint."""
        from aragora.server.middleware.deprecation import add_v1_headers_to_handler

        handler = MagicMock()
        add_v1_headers_to_handler(handler, "/healthz")
        handler.send_header.assert_not_called()


class TestV1PathRegex:
    """Tests for the V1 path regex pattern."""

    def test_regex_pattern_anchored_to_start(self):
        """Regex should be anchored to the start of the string."""
        from aragora.server.middleware.deprecation import _V1_PATH_RE

        assert _V1_PATH_RE.pattern.startswith("^")

    def test_regex_requires_api_v1(self):
        """Regex should require /api/v1."""
        from aragora.server.middleware.deprecation import _V1_PATH_RE

        assert _V1_PATH_RE.match("/api/v1/") is not None
        assert _V1_PATH_RE.match("/api/v1") is not None
        assert _V1_PATH_RE.match("/api/v2/") is None
        assert _V1_PATH_RE.match("/api/") is None
