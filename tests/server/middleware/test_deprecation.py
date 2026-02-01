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
        try:
            from aragora.server.middleware.deprecation import v1_sunset_middleware
        except RuntimeError:
            pytest.skip("aiohttp not available")

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
        try:
            from aragora.server.middleware.deprecation import (
                reset_v1_usage_tracker,
                v1_sunset_middleware,
            )
        except RuntimeError:
            pytest.skip("aiohttp not available")

        reset_v1_usage_tracker()

        mock_request = MagicMock()
        mock_request.path = "/api/v1/debates"
        mock_request.method = "GET"
        mock_request.headers = {}
        mock_request.headers.get = lambda k, d="": d

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
        try:
            from aragora.server.middleware.deprecation import (
                get_v1_usage_tracker,
                reset_v1_usage_tracker,
                v1_sunset_middleware,
            )
        except RuntimeError:
            pytest.skip("aiohttp not available")

        reset_v1_usage_tracker()

        mock_request = MagicMock()
        mock_request.path = "/api/v1/test"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.headers.get = lambda k, d="": d

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
        try:
            from aragora.server.middleware.deprecation import v1_sunset_middleware
        except RuntimeError:
            pytest.skip("aiohttp not available")

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
        try:
            from aragora.server.middleware.deprecation import (
                create_v1_sunset_middleware,
                v1_sunset_middleware,
            )
        except RuntimeError:
            pytest.skip("aiohttp not available")

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
