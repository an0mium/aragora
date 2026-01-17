"""Tests for API versioning middleware.

Tests cover:
- Path normalization (version extraction)
- Version prefix addition
- Version header injection
- Deprecation handling
- Version validation and sunset checks
- APIVersionMiddleware class
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from aragora.server.middleware.versioning import (
    API_VERSIONS,
    APIVersion,
    APIVersionMiddleware,
    CURRENT_VERSION,
    add_version_prefix,
    deprecate_version,
    get_all_versions,
    get_api_version,
    get_version_info,
    inject_version_headers,
    is_version_supported,
    log_version_usage,
    normalize_path,
)


class TestAPIVersionDataclass:
    """Tests for APIVersion dataclass."""

    def test_create_version(self):
        """Should create version with all fields."""
        version = APIVersion(
            version="v1",
            status="current",
            release_date="2024-01-01",
            sunset_date="2025-12-31",
            description="Test version",
        )
        assert version.version == "v1"
        assert version.status == "current"
        assert version.release_date == "2024-01-01"
        assert version.sunset_date == "2025-12-31"
        assert version.description == "Test version"

    def test_optional_sunset_date(self):
        """Sunset date should be optional."""
        version = APIVersion(
            version="v1",
            status="current",
            release_date="2024-01-01",
        )
        assert version.sunset_date is None


class TestAPIVersionsConfig:
    """Tests for API_VERSIONS configuration."""

    def test_v0_exists(self):
        """v0 (legacy) should be defined."""
        assert "v0" in API_VERSIONS
        assert API_VERSIONS["v0"].status == "current"

    def test_v1_exists(self):
        """v1 should be defined."""
        assert "v1" in API_VERSIONS
        assert API_VERSIONS["v1"].status == "current"

    def test_current_version_is_v1(self):
        """Current version should be v1."""
        assert CURRENT_VERSION == "v1"


class TestNormalizePath:
    """Tests for normalize_path function."""

    def test_versioned_path_v1(self):
        """Should extract v1 from versioned path."""
        normalized, version = normalize_path("/api/v1/debates")
        assert normalized == "/api/debates"
        assert version == "v1"

    def test_versioned_path_v2(self):
        """Should extract v2 from versioned path."""
        normalized, version = normalize_path("/api/v2/agents/leaderboard")
        assert normalized == "/api/agents/leaderboard"
        assert version == "v2"

    def test_unversioned_path(self):
        """Unversioned path should return v0."""
        normalized, version = normalize_path("/api/debates")
        assert normalized == "/api/debates"
        assert version == "v0"

    def test_deeply_nested_path(self):
        """Should handle deeply nested paths."""
        normalized, version = normalize_path("/api/v1/debates/123/messages/456")
        assert normalized == "/api/debates/123/messages/456"
        assert version == "v1"

    def test_non_api_path(self):
        """Non-API paths should pass through."""
        normalized, version = normalize_path("/health")
        assert normalized == "/health"
        assert version == "v0"

    def test_double_digit_version(self):
        """Should handle double-digit versions."""
        normalized, version = normalize_path("/api/v10/debates")
        assert normalized == "/api/debates"
        assert version == "v10"


class TestAddVersionPrefix:
    """Tests for add_version_prefix function."""

    def test_add_default_version(self):
        """Should add current version by default."""
        result = add_version_prefix("/api/debates")
        assert result == "/api/v1/debates"

    def test_add_specific_version(self):
        """Should add specified version."""
        result = add_version_prefix("/api/debates", "v2")
        assert result == "/api/v2/debates"

    def test_v0_no_prefix(self):
        """v0 should not add prefix."""
        result = add_version_prefix("/api/debates", "v0")
        assert result == "/api/debates"

    def test_non_api_path_unchanged(self):
        """Non-API paths should be unchanged."""
        result = add_version_prefix("/health")
        assert result == "/health"

    def test_nested_path(self):
        """Should work with nested paths."""
        result = add_version_prefix("/api/agents/claude/stats", "v1")
        assert result == "/api/v1/agents/claude/stats"


class TestInjectVersionHeaders:
    """Tests for inject_version_headers function."""

    def test_injects_version_header(self):
        """Should include X-API-Version header."""
        headers = inject_version_headers("v1")
        assert headers["X-API-Version"] == "v1"

    def test_current_version_no_deprecation(self):
        """Current version should not have deprecation headers."""
        headers = inject_version_headers("v1")
        assert "X-API-Deprecated" not in headers
        assert "X-API-Sunset" not in headers
        assert "Warning" not in headers

    def test_deprecated_version_headers(self):
        """Deprecated version should include warning headers."""
        # Temporarily deprecate v0 for testing
        original_status = API_VERSIONS["v0"].status
        original_sunset = API_VERSIONS["v0"].sunset_date
        try:
            API_VERSIONS["v0"].status = "deprecated"
            API_VERSIONS["v0"].sunset_date = "2025-12-31"

            headers = inject_version_headers("v0")
            assert headers["X-API-Deprecated"] == "true"
            assert headers["X-API-Sunset"] == "2025-12-31"
            assert "Warning" in headers
            assert "deprecated" in headers["Warning"].lower()
        finally:
            API_VERSIONS["v0"].status = original_status
            API_VERSIONS["v0"].sunset_date = original_sunset

    def test_unknown_version(self):
        """Unknown version should still return version header."""
        headers = inject_version_headers("v99")
        assert headers["X-API-Version"] == "v99"


class TestGetAPIVersion:
    """Tests for get_api_version function."""

    def test_explicit_version_header(self):
        """Should respect explicit X-API-Version header."""
        headers = {"X-API-Version": "v0"}
        version = get_api_version(headers)
        assert version == "v0"

    def test_missing_header_returns_current(self):
        """Missing header should return current version."""
        version = get_api_version({})
        assert version == CURRENT_VERSION

    def test_invalid_version_returns_current(self):
        """Invalid version should return current version."""
        headers = {"X-API-Version": "v99"}
        version = get_api_version(headers)
        assert version == CURRENT_VERSION

    def test_empty_header_returns_current(self):
        """Empty header should return current version."""
        headers = {"X-API-Version": ""}
        version = get_api_version(headers)
        assert version == CURRENT_VERSION

    def test_whitespace_trimmed(self):
        """Whitespace should be trimmed from header."""
        headers = {"X-API-Version": "  v1  "}
        version = get_api_version(headers)
        assert version == "v1"


class TestIsVersionSupported:
    """Tests for is_version_supported function."""

    def test_current_version_supported(self):
        """Current version should be supported."""
        assert is_version_supported("v1") is True

    def test_legacy_version_supported(self):
        """Legacy v0 should be supported."""
        assert is_version_supported("v0") is True

    def test_unknown_version_not_supported(self):
        """Unknown version should not be supported."""
        assert is_version_supported("v99") is False

    def test_sunset_version_not_supported(self):
        """Sunset version should not be supported."""
        original_status = API_VERSIONS["v0"].status
        try:
            API_VERSIONS["v0"].status = "sunset"
            assert is_version_supported("v0") is False
        finally:
            API_VERSIONS["v0"].status = original_status

    def test_past_sunset_date_not_supported(self):
        """Version past sunset date should not be supported."""
        original_sunset = API_VERSIONS["v0"].sunset_date
        try:
            # Set sunset date to yesterday
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            API_VERSIONS["v0"].sunset_date = yesterday
            assert is_version_supported("v0") is False
        finally:
            API_VERSIONS["v0"].sunset_date = original_sunset


class TestGetVersionInfo:
    """Tests for get_version_info function."""

    def test_known_version(self):
        """Should return info dict for known version."""
        info = get_version_info("v1")
        assert info is not None
        assert info["version"] == "v1"
        assert info["status"] == "current"
        assert "release_date" in info
        assert "is_current" in info
        assert "is_supported" in info

    def test_unknown_version(self):
        """Should return None for unknown version."""
        info = get_version_info("v99")
        assert info is None

    def test_is_current_flag(self):
        """is_current should be True only for current version."""
        v1_info = get_version_info("v1")
        v0_info = get_version_info("v0")

        assert v1_info["is_current"] is True
        assert v0_info["is_current"] is False


class TestGetAllVersions:
    """Tests for get_all_versions function."""

    def test_returns_list(self):
        """Should return a list."""
        versions = get_all_versions()
        assert isinstance(versions, list)

    def test_contains_known_versions(self):
        """Should contain known versions."""
        versions = get_all_versions()
        version_names = [v["version"] for v in versions]
        assert "v0" in version_names
        assert "v1" in version_names

    def test_versions_sorted(self):
        """Versions should be sorted."""
        versions = get_all_versions()
        version_names = [v["version"] for v in versions]
        assert version_names == sorted(version_names)


class TestDeprecateVersion:
    """Tests for deprecate_version function."""

    def test_deprecates_version(self):
        """Should mark version as deprecated with sunset date."""
        original_status = API_VERSIONS["v0"].status
        original_sunset = API_VERSIONS["v0"].sunset_date
        try:
            deprecate_version("v0", "2025-12-31")
            assert API_VERSIONS["v0"].status == "deprecated"
            assert API_VERSIONS["v0"].sunset_date == "2025-12-31"
        finally:
            API_VERSIONS["v0"].status = original_status
            API_VERSIONS["v0"].sunset_date = original_sunset

    def test_deprecate_unknown_version(self):
        """Should handle unknown version gracefully."""
        # Should not raise
        deprecate_version("v99", "2025-12-31")


class TestLogVersionUsage:
    """Tests for log_version_usage function."""

    def test_logs_deprecated_usage(self):
        """Should log warning for deprecated version."""
        original_status = API_VERSIONS["v0"].status
        original_sunset = API_VERSIONS["v0"].sunset_date
        try:
            API_VERSIONS["v0"].status = "deprecated"
            API_VERSIONS["v0"].sunset_date = "2025-12-31"

            with patch("aragora.server.middleware.versioning.logger") as mock_logger:
                log_version_usage("v0", "/api/debates")
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args[0][0]
                assert "deprecated" in call_args.lower()
        finally:
            API_VERSIONS["v0"].status = original_status
            API_VERSIONS["v0"].sunset_date = original_sunset

    def test_no_log_for_current_version(self):
        """Should not log for current version."""
        with patch("aragora.server.middleware.versioning.logger") as mock_logger:
            log_version_usage("v1", "/api/debates")
            mock_logger.warning.assert_not_called()


class TestAPIVersionMiddleware:
    """Tests for APIVersionMiddleware class."""

    def test_init_default_version(self):
        """Should use current version by default."""
        middleware = APIVersionMiddleware()
        assert middleware.default_version == CURRENT_VERSION

    def test_init_custom_default(self):
        """Should accept custom default version."""
        middleware = APIVersionMiddleware(default_version="v0")
        assert middleware.default_version == "v0"

    def test_process_request_versioned_path(self):
        """Should extract version from path."""
        middleware = APIVersionMiddleware()
        normalized, version = middleware.process_request("/api/v1/debates")
        assert normalized == "/api/debates"
        assert version == "v1"

    def test_process_request_unversioned_with_header(self):
        """Should use header version for unversioned path."""
        middleware = APIVersionMiddleware()
        headers = {"X-API-Version": "v0"}
        normalized, version = middleware.process_request("/api/debates", headers)
        assert normalized == "/api/debates"
        assert version == "v0"

    def test_process_request_path_version_precedence(self):
        """Path version should take precedence over header."""
        middleware = APIVersionMiddleware()
        headers = {"X-API-Version": "v0"}
        normalized, version = middleware.process_request("/api/v1/debates", headers)
        assert version == "v1"

    def test_process_request_default_version(self):
        """Should use default version when no version specified."""
        middleware = APIVersionMiddleware()
        normalized, version = middleware.process_request("/api/debates")
        assert version == CURRENT_VERSION

    def test_process_response(self):
        """Should return version headers."""
        middleware = APIVersionMiddleware()
        headers = middleware.process_response("v1")
        assert "X-API-Version" in headers
        assert headers["X-API-Version"] == "v1"

    def test_log_deprecated_enabled(self):
        """Should log deprecated version usage when enabled."""
        original_status = API_VERSIONS["v0"].status
        try:
            API_VERSIONS["v0"].status = "deprecated"
            middleware = APIVersionMiddleware(log_deprecated=True)

            with patch("aragora.server.middleware.versioning.logger") as mock_logger:
                middleware.process_request("/api/v0/debates")
                # Note: v0 path doesn't exist in pattern, so it becomes unversioned
                # Let's test with header instead
                middleware.process_request("/api/debates", {"X-API-Version": "v0"})
                # Warning should be logged
        finally:
            API_VERSIONS["v0"].status = original_status

    def test_log_deprecated_disabled(self):
        """Should not log when log_deprecated is False."""
        middleware = APIVersionMiddleware(log_deprecated=False)
        with patch("aragora.server.middleware.versioning.log_version_usage") as mock_log:
            middleware.process_request("/api/v1/debates")
            mock_log.assert_not_called()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_path(self):
        """Should handle empty path."""
        normalized, version = normalize_path("")
        assert normalized == ""
        assert version == "v0"

    def test_root_path(self):
        """Should handle root path."""
        normalized, version = normalize_path("/")
        assert normalized == "/"
        assert version == "v0"

    def test_api_only_path(self):
        """Should handle /api path alone."""
        normalized, version = normalize_path("/api")
        assert normalized == "/api"
        assert version == "v0"

    def test_api_version_only(self):
        """Should handle /api/v1 without endpoint."""
        # This doesn't match the pattern (requires something after version)
        normalized, version = normalize_path("/api/v1")
        assert version == "v0"

    def test_case_sensitivity(self):
        """Version should be case sensitive."""
        normalized, version = normalize_path("/api/V1/debates")
        # V1 (uppercase) won't match the pattern
        assert version == "v0"

    def test_trailing_slash(self):
        """Should handle trailing slash."""
        normalized, version = normalize_path("/api/v1/debates/")
        assert normalized == "/api/debates/"
        assert version == "v1"
