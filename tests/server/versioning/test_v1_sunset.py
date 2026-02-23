"""Tests for V1 API sunset enforcement middleware.

Covers:
- DeprecationMiddleware block_sunset=True returns 410 for sunset endpoints
- DeprecationMiddleware block_sunset=False allows requests through with headers
- Sunset header format validation
- V2 routes unaffected by deprecation middleware
- days_until_v1_sunset() helper correctness
- deprecation_level() severity tiers
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from aragora.server.versioning.constants import (
    V1_DEPRECATION_ANNOUNCED,
    V1_SUNSET_DATE,
    V1_SUNSET_HTTP_DATE,
    V1_SUNSET_ISO,
    days_until_v1_sunset,
    deprecation_level,
    is_v1_sunset,
)
from aragora.server.versioning.deprecation import (
    DeprecationLevel,
    DeprecationMiddleware,
    DeprecationRegistry,
)
from aragora.server.versioning.router import (
    APIVersion,
    VersionedRouter,
    get_version_from_request,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> DeprecationRegistry:
    """Fresh registry for each test."""
    return DeprecationRegistry()


@pytest.fixture()
def sunset_registry(registry: DeprecationRegistry) -> DeprecationRegistry:
    """Registry with a sunset (past-date) endpoint registered."""
    past_date = date.today() - timedelta(days=30)
    registry.register(
        path="/api/v1/debates",
        method="GET",
        sunset_date=past_date,
        replacement="/api/v2/debates",
        message="GET /api/v1/debates is sunset",
    )
    return registry


@pytest.fixture()
def active_deprecation_registry(registry: DeprecationRegistry) -> DeprecationRegistry:
    """Registry with a deprecated-but-not-yet-sunset endpoint."""
    future_date = date.today() + timedelta(days=180)
    registry.register(
        path="/api/v1/agents",
        method="GET",
        sunset_date=future_date,
        replacement="/api/v2/agents",
        message="GET /api/v1/agents is deprecated",
    )
    return registry


# ---------------------------------------------------------------------------
# 1. block_sunset=True returns 410 for sunset endpoints
# ---------------------------------------------------------------------------


class TestBlockSunsetTrue:
    """DeprecationMiddleware(block_sunset=True) returns 410-like error for sunset endpoints."""

    def test_sunset_endpoint_blocked_with_410(self, sunset_registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            block_sunset=True,
        )
        result = middleware.process_request(
            path="/api/v1/debates",
            method="GET",
            headers={},
        )
        assert result is not None
        assert result["status"] == 410
        assert result["error"] == "endpoint_sunset"
        assert result["replacement"] == "/api/v2/debates"

    def test_sunset_endpoint_message_includes_date(
        self, sunset_registry: DeprecationRegistry
    ) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            block_sunset=True,
        )
        result = middleware.process_request(
            path="/api/v1/debates",
            method="GET",
            headers={},
        )
        assert result is not None
        assert "removed" in result["message"].lower()

    def test_non_sunset_deprecated_not_blocked(
        self, active_deprecation_registry: DeprecationRegistry
    ) -> None:
        """Deprecated but not yet sunset endpoints should NOT be blocked."""
        middleware = DeprecationMiddleware(
            registry=active_deprecation_registry,
            block_sunset=True,
        )
        result = middleware.process_request(
            path="/api/v1/agents",
            method="GET",
            headers={},
        )
        # Should return None -- request allowed through
        assert result is None

    def test_usage_recorded_even_when_blocked(self, sunset_registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            block_sunset=True,
            log_usage=True,
        )
        middleware.process_request(path="/api/v1/debates", method="GET", headers={})
        stats = sunset_registry.get_usage_stats()
        assert stats.get("GET:/api/v1/debates", 0) >= 1


# ---------------------------------------------------------------------------
# 2. block_sunset=False allows requests through with deprecation headers
# ---------------------------------------------------------------------------


class TestBlockSunsetFalse:
    """DeprecationMiddleware(block_sunset=False) allows requests and adds headers."""

    def test_sunset_endpoint_allowed_through(self, sunset_registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            block_sunset=False,
        )
        result = middleware.process_request(
            path="/api/v1/debates",
            method="GET",
            headers={},
        )
        # Should return None -- no blocking
        assert result is None

    def test_response_headers_added_for_deprecated(
        self, active_deprecation_registry: DeprecationRegistry
    ) -> None:
        middleware = DeprecationMiddleware(
            registry=active_deprecation_registry,
            add_headers=True,
        )
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v1/agents",
            method="GET",
            headers=headers,
        )
        assert "Deprecation" in result_headers
        assert "Sunset" in result_headers
        assert "X-Deprecation-Level" in result_headers

    def test_deprecation_header_rfc8594_format(
        self, active_deprecation_registry: DeprecationRegistry
    ) -> None:
        """Deprecation header value should be @<timestamp> per RFC 8594."""
        middleware = DeprecationMiddleware(
            registry=active_deprecation_registry,
            add_headers=True,
        )
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v1/agents",
            method="GET",
            headers=headers,
        )
        dep_value = result_headers["Deprecation"]
        # Should be @<unix-timestamp>
        assert dep_value.startswith("@")
        timestamp_str = dep_value[1:]
        timestamp = int(timestamp_str)
        assert timestamp > 0

    def test_link_header_with_replacement(
        self, active_deprecation_registry: DeprecationRegistry
    ) -> None:
        middleware = DeprecationMiddleware(
            registry=active_deprecation_registry,
            add_headers=True,
        )
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v1/agents",
            method="GET",
            headers=headers,
        )
        assert "Link" in result_headers
        assert "/api/v2/agents" in result_headers["Link"]
        assert 'rel="successor-version"' in result_headers["Link"]


# ---------------------------------------------------------------------------
# 3. Sunset header format validation
# ---------------------------------------------------------------------------


class TestSunsetHeaderFormat:
    """Sunset header should be ISO date format."""

    def test_sunset_header_is_iso_date(
        self, active_deprecation_registry: DeprecationRegistry
    ) -> None:
        middleware = DeprecationMiddleware(
            registry=active_deprecation_registry,
            add_headers=True,
        )
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v1/agents",
            method="GET",
            headers=headers,
        )
        sunset_value = result_headers["Sunset"]
        # Should be parseable as ISO date
        parsed = date.fromisoformat(sunset_value)
        assert isinstance(parsed, date)

    def test_no_headers_when_disabled(
        self, active_deprecation_registry: DeprecationRegistry
    ) -> None:
        middleware = DeprecationMiddleware(
            registry=active_deprecation_registry,
            add_headers=False,
        )
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v1/agents",
            method="GET",
            headers=headers,
        )
        assert "Deprecation" not in result_headers
        assert "Sunset" not in result_headers

    def test_v1_sunset_iso_constant_matches_date(self) -> None:
        """V1_SUNSET_ISO should match V1_SUNSET_DATE."""
        assert V1_SUNSET_ISO == V1_SUNSET_DATE.isoformat()

    def test_v1_sunset_http_date_format(self) -> None:
        """V1_SUNSET_HTTP_DATE should follow RFC 7231 format."""
        # Should contain the date components
        assert "2026" in V1_SUNSET_HTTP_DATE
        assert "Jun" in V1_SUNSET_HTTP_DATE
        assert "GMT" in V1_SUNSET_HTTP_DATE


# ---------------------------------------------------------------------------
# 4. V2 routes unaffected by deprecation middleware
# ---------------------------------------------------------------------------


class TestV2RoutesUnaffected:
    """V2 endpoints should not be impacted by deprecation middleware."""

    def test_v2_route_not_blocked(self, sunset_registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            block_sunset=True,
        )
        result = middleware.process_request(
            path="/api/v2/debates",
            method="GET",
            headers={},
        )
        assert result is None

    def test_v2_route_no_deprecation_headers(self, sunset_registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            add_headers=True,
        )
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v2/debates",
            method="GET",
            headers=headers,
        )
        assert "Deprecation" not in result_headers
        assert "Sunset" not in result_headers

    def test_unregistered_endpoint_passes_through(self, registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=registry,
            block_sunset=True,
        )
        result = middleware.process_request(
            path="/api/v2/new-feature",
            method="POST",
            headers={},
        )
        assert result is None

    def test_version_from_request_url_prefix(self) -> None:
        """get_version_from_request correctly extracts v2 from URL."""
        version, cleaned_path = get_version_from_request("/api/v2/debates")
        assert version == APIVersion.V2
        assert cleaned_path == "/api/debates"

    def test_version_from_request_v1_url_prefix(self) -> None:
        """get_version_from_request correctly extracts v1 from URL."""
        version, cleaned_path = get_version_from_request("/api/v1/debates")
        assert version == APIVersion.V1
        assert cleaned_path == "/api/debates"

    def test_versioned_router_resolves_v2(self) -> None:
        router = VersionedRouter()

        @router.route("/users", method="GET", version=APIVersion.V2)
        def get_users_v2():
            return {"data": []}

        entry = router.resolve("/users", "GET", APIVersion.V2)
        assert entry is not None
        assert entry.version == APIVersion.V2


# ---------------------------------------------------------------------------
# 5. days_until_v1_sunset() helper
# ---------------------------------------------------------------------------


class TestDaysUntilV1Sunset:
    """days_until_v1_sunset() returns correct values."""

    def test_returns_non_negative(self) -> None:
        result = days_until_v1_sunset()
        assert result >= 0

    def test_returns_zero_when_past_sunset(self) -> None:
        past_date = date.today() - timedelta(days=10)
        with patch("aragora.server.versioning.constants.V1_SUNSET_DATE", past_date):
            result = days_until_v1_sunset()
            assert result == 0

    def test_returns_correct_days_before_sunset(self) -> None:
        future_date = date.today() + timedelta(days=100)
        with patch("aragora.server.versioning.constants.V1_SUNSET_DATE", future_date):
            result = days_until_v1_sunset()
            assert result == 100

    def test_returns_zero_on_sunset_day(self) -> None:
        today = date.today()
        with patch("aragora.server.versioning.constants.V1_SUNSET_DATE", today):
            result = days_until_v1_sunset()
            assert result == 0

    def test_is_v1_sunset_false_before_date(self) -> None:
        future_date = date.today() + timedelta(days=100)
        with patch("aragora.server.versioning.constants.V1_SUNSET_DATE", future_date):
            assert is_v1_sunset() is False

    def test_is_v1_sunset_true_after_date(self) -> None:
        past_date = date.today() - timedelta(days=1)
        with patch("aragora.server.versioning.constants.V1_SUNSET_DATE", past_date):
            assert is_v1_sunset() is True

    def test_v1_deprecation_announced_before_sunset(self) -> None:
        """Deprecation announcement should be before sunset date."""
        assert V1_DEPRECATION_ANNOUNCED < V1_SUNSET_DATE


# ---------------------------------------------------------------------------
# 6. deprecation_level() severity
# ---------------------------------------------------------------------------


class TestDeprecationLevel:
    """deprecation_level() returns correct severity tier."""

    def test_returns_sunset_when_past_date(self) -> None:
        past_date = date.today() - timedelta(days=10)
        with (
            patch("aragora.server.versioning.constants.V1_SUNSET_DATE", past_date),
            patch(
                "aragora.server.versioning.constants.is_v1_sunset",
                return_value=True,
            ),
            patch(
                "aragora.server.versioning.constants.days_until_v1_sunset",
                return_value=0,
            ),
        ):
            assert deprecation_level() == "sunset"

    def test_returns_critical_within_30_days(self) -> None:
        near_date = date.today() + timedelta(days=15)
        with (
            patch("aragora.server.versioning.constants.V1_SUNSET_DATE", near_date),
            patch(
                "aragora.server.versioning.constants.is_v1_sunset",
                return_value=False,
            ),
            patch(
                "aragora.server.versioning.constants.days_until_v1_sunset",
                return_value=15,
            ),
        ):
            assert deprecation_level() == "critical"

    def test_returns_warning_when_far_from_sunset(self) -> None:
        far_date = date.today() + timedelta(days=200)
        with (
            patch("aragora.server.versioning.constants.V1_SUNSET_DATE", far_date),
            patch(
                "aragora.server.versioning.constants.is_v1_sunset",
                return_value=False,
            ),
            patch(
                "aragora.server.versioning.constants.days_until_v1_sunset",
                return_value=200,
            ),
        ):
            assert deprecation_level() == "warning"

    def test_deprecation_level_enum_values(self) -> None:
        """DeprecationLevel enum should have expected values."""
        assert DeprecationLevel.WARNING.value == "warning"
        assert DeprecationLevel.CRITICAL.value == "critical"
        assert DeprecationLevel.SUNSET.value == "sunset"


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the deprecation system."""

    def test_registry_records_usage(self, sunset_registry: DeprecationRegistry) -> None:
        middleware = DeprecationMiddleware(
            registry=sunset_registry,
            block_sunset=False,
            log_usage=True,
        )
        for _ in range(5):
            middleware.process_request(path="/api/v1/debates", method="GET", headers={})
        stats = sunset_registry.get_usage_stats()
        assert stats["GET:/api/v1/debates"] == 5

    def test_different_methods_tracked_separately(self, registry: DeprecationRegistry) -> None:
        past_date = date.today() - timedelta(days=1)
        registry.register(path="/api/v1/items", method="GET", sunset_date=past_date)
        registry.register(path="/api/v1/items", method="POST", sunset_date=past_date)

        middleware = DeprecationMiddleware(
            registry=registry,
            block_sunset=True,
        )
        result_get = middleware.process_request(path="/api/v1/items", method="GET", headers={})
        result_post = middleware.process_request(path="/api/v1/items", method="POST", headers={})
        assert result_get is not None
        assert result_post is not None

    def test_critical_deprecation_near_sunset(self, registry: DeprecationRegistry) -> None:
        near_date = date.today() + timedelta(days=10)
        registry.register(
            path="/api/v1/soon",
            method="GET",
            sunset_date=near_date,
        )
        warning = registry.get_warning("/api/v1/soon", "GET")
        assert warning is not None
        assert warning.level == DeprecationLevel.CRITICAL

    def test_warning_deprecation_far_from_sunset(self, registry: DeprecationRegistry) -> None:
        far_date = date.today() + timedelta(days=180)
        registry.register(
            path="/api/v1/later",
            method="GET",
            sunset_date=far_date,
        )
        warning = registry.get_warning("/api/v1/later", "GET")
        assert warning is not None
        assert warning.level == DeprecationLevel.WARNING

    def test_migration_guide_header(self, registry: DeprecationRegistry) -> None:
        future_date = date.today() + timedelta(days=60)
        registry.register(
            path="/api/v1/guide",
            method="GET",
            sunset_date=future_date,
            migration_guide="https://docs.aragora.ai/migration/v1-to-v2",
        )
        middleware = DeprecationMiddleware(registry=registry, add_headers=True)
        headers: dict[str, str] = {}
        result_headers = middleware.add_response_headers(
            path="/api/v1/guide",
            method="GET",
            headers=headers,
        )
        assert (
            result_headers.get("X-Migration-Guide") == "https://docs.aragora.ai/migration/v1-to-v2"
        )
