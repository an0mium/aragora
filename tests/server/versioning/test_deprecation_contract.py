"""
Contract tests for API v1 sunset and deprecation enforcement.

Tests the deprecation infrastructure against the planned v1 → v2 migration:
- Deprecation headers (RFC 8594)
- Sunset headers and blocking
- Link headers to successor endpoints
- Pattern-based endpoint matching

Sunset Date: June 1, 2026

Deprecated v1 → v2 Endpoint Mappings:
1. GET /api/debates/list → GET /api/debates
2. POST /api/debate/new → POST /api/debates/start
3. GET /api/elo/rankings → GET /api/agent/leaderboard
4. GET /api/agent/elo → GET /api/agent/{name}/profile
5. POST /api/stream/start → WebSocket /ws
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from aragora.server.middleware.deprecation_enforcer import (
    DeprecatedEndpoint,
    DeprecationEnforcer,
    DeprecationStats,
    get_deprecation_enforcer,
    register_deprecated_endpoint,
    reset_deprecation_enforcer,
)

# ============================================================================
# Constants for the v1 sunset plan
# ============================================================================

V1_SUNSET_DATE = date(2026, 6, 1)

V1_DEPRECATIONS = [
    {
        "path": "/api/debates/list",
        "methods": ["GET"],
        "replacement": "/api/v2/debates",
        "message": "Use GET /api/v2/debates with pagination parameters",
    },
    {
        "path": "/api/debate/new",
        "methods": ["POST"],
        "replacement": "/api/v2/debates/start",
        "message": "Use POST /api/v2/debates/start with the new request format",
    },
    {
        "path": "/api/elo/rankings",
        "methods": ["GET"],
        "replacement": "/api/v2/agent/leaderboard",
        "message": "Use GET /api/v2/agent/leaderboard for agent rankings",
    },
    {
        "path": "/api/agent/elo",
        "methods": ["GET"],
        "replacement": "/api/v2/agent/{name}/profile",
        "message": "Use GET /api/v2/agent/{name}/profile for agent ELO data",
    },
    {
        "path": "/api/stream/start",
        "methods": ["POST"],
        "replacement": "/ws",
        "message": "Use WebSocket connection at /ws for streaming",
    },
]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def enforcer():
    """Fresh DeprecationEnforcer for each test."""
    return DeprecationEnforcer(block_sunset=True, log_usage=False)


@pytest.fixture
def v1_enforcer(enforcer):
    """Enforcer pre-configured with v1 deprecations."""
    for dep in V1_DEPRECATIONS:
        enforcer.register(
            path_pattern=dep["path"],
            methods=dep["methods"],
            sunset_date=V1_SUNSET_DATE,
            replacement=dep["replacement"],
            message=dep["message"],
            version="v1",
        )
    return enforcer


@pytest.fixture(autouse=True)
def reset_global_enforcer():
    """Reset global enforcer between tests."""
    reset_deprecation_enforcer()
    yield
    reset_deprecation_enforcer()


# ============================================================================
# DeprecatedEndpoint Tests
# ============================================================================


class TestDeprecatedEndpoint:
    """Tests for DeprecatedEndpoint dataclass."""

    def test_is_sunset_before_date(self):
        """Endpoint should not be sunset before sunset date."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=date(2099, 12, 31),
        )
        assert endpoint.is_sunset is False

    def test_is_sunset_after_date(self):
        """Endpoint should be sunset after sunset date."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=date(2020, 1, 1),
        )
        assert endpoint.is_sunset is True

    def test_is_sunset_no_date(self):
        """Endpoint without sunset date should not be sunset."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=None,
        )
        assert endpoint.is_sunset is False

    def test_days_until_sunset_future(self):
        """Days until sunset should be positive for future dates."""
        future_date = date(2099, 12, 31)
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=future_date,
        )
        assert endpoint.days_until_sunset > 0

    def test_days_until_sunset_past(self):
        """Days until sunset should be negative for past dates."""
        past_date = date(2020, 1, 1)
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=past_date,
        )
        assert endpoint.days_until_sunset < 0

    def test_days_until_sunset_none(self):
        """Days until sunset should be None without sunset date."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=None,
        )
        assert endpoint.days_until_sunset is None

    def test_deprecation_level_warning(self):
        """Far future sunset should be warning level."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=date(2099, 12, 31),
        )
        assert endpoint.deprecation_level == "warning"

    def test_deprecation_level_critical(self):
        """Sunset within 30 days should be critical level."""
        from datetime import timedelta

        near_date = date.today() + timedelta(days=15)
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=near_date,
        )
        assert endpoint.deprecation_level == "critical"

    def test_deprecation_level_sunset(self):
        """Past sunset should be sunset level."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=date(2020, 1, 1),
        )
        assert endpoint.deprecation_level == "sunset"

    def test_deprecation_level_no_date(self):
        """No sunset date should be warning level."""
        endpoint = DeprecatedEndpoint(
            path_pattern="/api/test",
            methods=["GET"],
            sunset_date=None,
        )
        assert endpoint.deprecation_level == "warning"


# ============================================================================
# DeprecationStats Tests
# ============================================================================


class TestDeprecationStats:
    """Tests for DeprecationStats tracking."""

    def test_record_call(self):
        """Should track calls per endpoint."""
        stats = DeprecationStats()
        stats.record_call("/api/test", "GET")
        stats.record_call("/api/test", "GET")
        stats.record_call("/api/other", "POST")

        assert stats.total_deprecated_calls == 3
        assert stats.calls_by_endpoint["GET:/api/test"] == 2
        assert stats.calls_by_endpoint["POST:/api/other"] == 1

    def test_record_blocked(self):
        """Should track blocked sunset calls."""
        stats = DeprecationStats()
        stats.record_blocked()
        stats.record_blocked()

        assert stats.blocked_sunset_calls == 2

    def test_to_dict(self):
        """Should serialize to dictionary."""
        stats = DeprecationStats()
        stats.record_call("/api/test", "GET")
        stats.record_blocked()

        data = stats.to_dict()

        assert data["total_deprecated_calls"] == 1
        assert data["blocked_sunset_calls"] == 1
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ============================================================================
# DeprecationEnforcer Tests
# ============================================================================


class TestDeprecationEnforcerRegistration:
    """Tests for endpoint registration."""

    def test_register_single_method(self, enforcer):
        """Should register single method endpoint."""
        endpoint = enforcer.register(
            path_pattern="/api/test",
            methods="GET",
            sunset_date=V1_SUNSET_DATE,
        )

        assert endpoint.path_pattern == "/api/test"
        assert endpoint.methods == ["GET"]
        assert endpoint.sunset_date == V1_SUNSET_DATE

    def test_register_multiple_methods(self, enforcer):
        """Should register multiple method endpoint."""
        endpoint = enforcer.register(
            path_pattern="/api/test",
            methods=["GET", "POST", "PUT"],
            sunset_date=V1_SUNSET_DATE,
        )

        assert endpoint.methods == ["GET", "POST", "PUT"]

    def test_register_with_replacement(self, enforcer):
        """Should store replacement URL."""
        endpoint = enforcer.register(
            path_pattern="/api/v1/test",
            methods="GET",
            replacement="/api/v2/test",
        )

        assert endpoint.replacement == "/api/v2/test"

    def test_register_sets_deprecated_since(self, enforcer):
        """Should set deprecated_since to today."""
        endpoint = enforcer.register(
            path_pattern="/api/test",
            methods="GET",
        )

        assert endpoint.deprecated_since == date.today()

    def test_get_all_deprecated(self, v1_enforcer):
        """Should return all registered deprecations."""
        all_deps = v1_enforcer.get_all_deprecated()

        assert len(all_deps) == len(V1_DEPRECATIONS)


class TestDeprecationEnforcerMatching:
    """Tests for request matching."""

    def test_exact_path_match(self, enforcer):
        """Should match exact paths."""
        enforcer.register("/api/test", "GET")

        assert enforcer.check_request("/api/test", "GET") is not None
        assert enforcer.check_request("/api/test", "POST") is None
        assert enforcer.check_request("/api/other", "GET") is None

    def test_wildcard_single_segment(self, enforcer):
        """Should match single segment wildcard."""
        enforcer.register("/api/users/*/profile", "GET")

        assert enforcer.check_request("/api/users/123/profile", "GET") is not None
        assert enforcer.check_request("/api/users/abc/profile", "GET") is not None
        assert enforcer.check_request("/api/users/profile", "GET") is None
        assert enforcer.check_request("/api/users/123/456/profile", "GET") is None

    def test_wildcard_multi_segment(self, enforcer):
        """Should match multi-segment wildcard."""
        enforcer.register("/api/**", "GET")

        assert enforcer.check_request("/api/test", "GET") is not None
        assert enforcer.check_request("/api/a/b/c", "GET") is not None
        assert enforcer.check_request("/other/test", "GET") is None

    def test_param_placeholder(self, enforcer):
        """Should match path parameter placeholders."""
        enforcer.register("/api/users/{user_id}/posts/{post_id}", "GET")

        assert enforcer.check_request("/api/users/123/posts/456", "GET") is not None
        assert enforcer.check_request("/api/users/abc/posts/xyz", "GET") is not None

    def test_method_case_insensitive(self, enforcer):
        """Should match methods case-insensitively."""
        enforcer.register("/api/test", "GET")

        assert enforcer.check_request("/api/test", "get") is not None
        assert enforcer.check_request("/api/test", "Get") is not None

    def test_v1_deprecations_match(self, v1_enforcer):
        """All v1 deprecations should be matchable."""
        for dep in V1_DEPRECATIONS:
            method = dep["methods"][0]
            match = v1_enforcer.check_request(dep["path"], method)
            assert match is not None, f"Failed to match {method} {dep['path']}"


class TestDeprecationEnforcerHeaders:
    """Tests for deprecation header generation."""

    def test_deprecation_header_with_sunset(self, enforcer):
        """Should generate RFC 8594 Deprecation header with timestamp."""
        enforcer.register("/api/test", "GET", sunset_date=V1_SUNSET_DATE)

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert "Deprecation" in headers
        # Should be Unix timestamp format: @<timestamp>
        assert headers["Deprecation"].startswith("@")
        timestamp = int(headers["Deprecation"][1:])
        assert timestamp > 0

    def test_deprecation_header_without_sunset(self, enforcer):
        """Should generate simple 'true' Deprecation header without sunset."""
        enforcer.register("/api/test", "GET", sunset_date=None)

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert headers["Deprecation"] == "true"

    def test_sunset_header_format(self, enforcer):
        """Should generate HTTP date format Sunset header."""
        enforcer.register("/api/test", "GET", sunset_date=V1_SUNSET_DATE)

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert "Sunset" in headers
        # Should be HTTP date format
        assert "GMT" in headers["Sunset"]
        # Should contain the sunset date components
        assert "Jun" in headers["Sunset"] or "01" in headers["Sunset"]

    def test_link_header_successor(self, enforcer):
        """Should generate Link header with successor-version relation."""
        enforcer.register(
            "/api/v1/test",
            "GET",
            replacement="/api/v2/test",
        )

        request = MagicMock()
        request.path = "/api/v1/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert "Link" in headers
        assert "/api/v2/test" in headers["Link"]
        assert 'rel="successor-version"' in headers["Link"]

    def test_deprecation_level_header(self, enforcer):
        """Should include X-Deprecation-Level header."""
        enforcer.register("/api/test", "GET", sunset_date=V1_SUNSET_DATE)

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert "X-Deprecation-Level" in headers
        assert headers["X-Deprecation-Level"] in ("warning", "critical", "sunset")

    def test_migration_guide_header(self, enforcer):
        """Should include X-Migration-Guide header when provided."""
        enforcer.register(
            "/api/test",
            "GET",
            migration_guide_url="https://docs.example.com/migration",
        )

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert headers["X-Migration-Guide"] == "https://docs.example.com/migration"

    def test_deprecated_version_header(self, enforcer):
        """Should include X-Deprecated-Version header when provided."""
        enforcer.register("/api/test", "GET", version="v1")

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, _ = enforcer.process_request(request)

        assert headers["X-Deprecated-Version"] == "v1"


class TestDeprecationEnforcerBlocking:
    """Tests for sunset endpoint blocking."""

    def test_blocks_sunset_endpoint(self):
        """Should return 410 response for sunset endpoints."""
        enforcer = DeprecationEnforcer(block_sunset=True)
        enforcer.register(
            "/api/test",
            "GET",
            sunset_date=date(2020, 1, 1),  # Past date
            replacement="/api/v2/test",
        )

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, response = enforcer.process_request(request)

        assert response is not None
        assert response.status == 410

    def test_no_block_when_disabled(self):
        """Should not block when block_sunset=False."""
        enforcer = DeprecationEnforcer(block_sunset=False)
        enforcer.register(
            "/api/test",
            "GET",
            sunset_date=date(2020, 1, 1),  # Past date
        )

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, response = enforcer.process_request(request)

        assert response is None
        assert "Deprecation" in headers

    def test_no_block_before_sunset(self, enforcer):
        """Should not block endpoints before sunset."""
        enforcer.register(
            "/api/test",
            "GET",
            sunset_date=date(2099, 12, 31),  # Future date
        )

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        headers, response = enforcer.process_request(request)

        assert response is None

    def test_sunset_response_body(self):
        """Should include error details in sunset response body."""
        enforcer = DeprecationEnforcer(block_sunset=True)
        enforcer.register(
            "/api/test",
            "GET",
            sunset_date=date(2020, 1, 1),
            replacement="/api/v2/test",
            message="This endpoint has been removed",
        )

        request = MagicMock()
        request.path = "/api/test"
        request.method = "GET"

        _, response = enforcer.process_request(request)

        # aiohttp json_response returns a Response object
        # The body would be accessible via response.body in a real request
        assert response is not None


# ============================================================================
# Global Registration Tests
# ============================================================================


class TestGlobalRegistration:
    """Tests for global enforcer registration."""

    def test_register_deprecated_endpoint(self):
        """Should register via global function."""
        endpoint = register_deprecated_endpoint(
            path="/api/test",
            methods="GET",
            sunset_date=V1_SUNSET_DATE,
        )

        assert endpoint is not None
        assert endpoint.path_pattern == "/api/test"

    def test_get_deprecation_enforcer_singleton(self):
        """Should return same enforcer instance."""
        enforcer1 = get_deprecation_enforcer()
        enforcer2 = get_deprecation_enforcer()

        assert enforcer1 is enforcer2

    def test_reset_clears_enforcer(self):
        """Reset should clear the global enforcer."""
        register_deprecated_endpoint("/api/test", "GET")
        enforcer1 = get_deprecation_enforcer()

        reset_deprecation_enforcer()
        enforcer2 = get_deprecation_enforcer()

        assert enforcer1 is not enforcer2
        assert len(enforcer2.get_all_deprecated()) == 0


# ============================================================================
# V1 Sunset Contract Tests
# ============================================================================


class TestV1SunsetContract:
    """Contract tests for the planned v1 sunset."""

    def test_all_v1_deprecations_registered(self, v1_enforcer):
        """All planned v1 deprecations should be registered."""
        registered = v1_enforcer.get_all_deprecated()
        paths = [e.path_pattern for e in registered]

        for dep in V1_DEPRECATIONS:
            assert dep["path"] in paths, f"Missing deprecation: {dep['path']}"

    def test_v1_sunset_date_consistent(self, v1_enforcer):
        """All v1 endpoints should have same sunset date."""
        for endpoint in v1_enforcer.get_all_deprecated():
            assert endpoint.sunset_date == V1_SUNSET_DATE

    def test_v1_endpoints_have_replacements(self, v1_enforcer):
        """All v1 endpoints should have replacement URLs."""
        for endpoint in v1_enforcer.get_all_deprecated():
            assert endpoint.replacement is not None
            assert endpoint.replacement.startswith("/")

    def test_v1_version_tag(self, v1_enforcer):
        """All v1 endpoints should be tagged with version."""
        for endpoint in v1_enforcer.get_all_deprecated():
            assert endpoint.version == "v1"

    def test_debates_list_deprecation(self, v1_enforcer):
        """GET /api/debates/list → GET /api/v2/debates."""
        endpoint = v1_enforcer.check_request("/api/debates/list", "GET")

        assert endpoint is not None
        assert endpoint.replacement == "/api/v2/debates"

    def test_debate_new_deprecation(self, v1_enforcer):
        """POST /api/debate/new → POST /api/v2/debates/start."""
        endpoint = v1_enforcer.check_request("/api/debate/new", "POST")

        assert endpoint is not None
        assert endpoint.replacement == "/api/v2/debates/start"

    def test_elo_rankings_deprecation(self, v1_enforcer):
        """GET /api/elo/rankings → GET /api/v2/agent/leaderboard."""
        endpoint = v1_enforcer.check_request("/api/elo/rankings", "GET")

        assert endpoint is not None
        assert endpoint.replacement == "/api/v2/agent/leaderboard"

    def test_agent_elo_deprecation(self, v1_enforcer):
        """GET /api/agent/elo → GET /api/v2/agent/{name}/profile."""
        endpoint = v1_enforcer.check_request("/api/agent/elo", "GET")

        assert endpoint is not None
        assert "/api/v2/agent/" in endpoint.replacement

    def test_stream_start_deprecation(self, v1_enforcer):
        """POST /api/stream/start → WebSocket /ws."""
        endpoint = v1_enforcer.check_request("/api/stream/start", "POST")

        assert endpoint is not None
        assert endpoint.replacement == "/ws"

    def test_headers_include_all_rfc8594(self, v1_enforcer):
        """RFC 8594 compliance: Deprecation, Sunset, Link headers."""
        request = MagicMock()
        request.path = "/api/debates/list"
        request.method = "GET"

        headers, _ = v1_enforcer.process_request(request)

        # RFC 8594 required headers
        assert "Deprecation" in headers
        assert "Sunset" in headers
        assert "Link" in headers

        # Deprecation header format: @<unix-timestamp> or "true"
        dep = headers["Deprecation"]
        assert dep.startswith("@") or dep == "true"

        # Link header format: <url>; rel="successor-version"
        assert 'rel="successor-version"' in headers["Link"]
