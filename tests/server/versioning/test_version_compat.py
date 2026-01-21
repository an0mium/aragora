"""Tests for API versioning compatibility layer."""

import pytest

from aragora.server.versioning import (
    APIVersion,
    extract_version,
    is_legacy_path,
    is_versioned_path,
    normalize_path_version,
    strip_version_prefix,
    version_response_headers,
)


class TestVersionExtraction:
    """Tests for extract_version function."""

    def test_versioned_v1_path(self):
        """V1 paths should extract correctly."""
        version, is_legacy = extract_version("/api/v1/debates")
        assert version == APIVersion.V1
        assert is_legacy is False

    def test_versioned_v2_path(self):
        """V2 paths should extract correctly."""
        version, is_legacy = extract_version("/api/v2/debates")
        assert version == APIVersion.V2
        assert is_legacy is False

    def test_legacy_path_defaults_to_v1(self):
        """Legacy paths should default to V1 with is_legacy=True."""
        version, is_legacy = extract_version("/api/debates")
        assert version == APIVersion.V1
        assert is_legacy is True

    def test_non_api_path(self):
        """Non-API paths should return current version."""
        version, is_legacy = extract_version("/healthz")
        assert version == APIVersion.V2  # Current version
        assert is_legacy is False

    def test_header_version_override(self):
        """X-API-Version header should override path version."""
        headers = {"X-API-Version": "v2"}
        version, is_legacy = extract_version("/api/debates", headers)
        assert version == APIVersion.V2
        assert is_legacy is False


class TestPathChecks:
    """Tests for path checking functions."""

    def test_is_versioned_path(self):
        """Should correctly identify versioned paths."""
        assert is_versioned_path("/api/v1/debates") is True
        assert is_versioned_path("/api/v2/debates") is True
        assert is_versioned_path("/api/debates") is False
        assert is_versioned_path("/healthz") is False

    def test_is_legacy_path(self):
        """Should correctly identify legacy paths."""
        assert is_legacy_path("/api/debates") is True
        assert is_legacy_path("/api/v1/debates") is False
        assert is_legacy_path("/healthz") is False


class TestPathNormalization:
    """Tests for path normalization functions."""

    def test_strip_version_prefix(self):
        """Should strip version prefix correctly."""
        assert strip_version_prefix("/api/v1/debates") == "/api/debates"
        assert strip_version_prefix("/api/v2/agents/list") == "/api/agents/list"
        assert strip_version_prefix("/api/debates") == "/api/debates"

    def test_normalize_path_version(self):
        """Should add current version to legacy paths."""
        result = normalize_path_version("/api/debates")
        assert result == "/api/v2/debates"  # v2 is current

    def test_normalize_versioned_path_unchanged(self):
        """Already versioned paths should be unchanged."""
        result = normalize_path_version("/api/v1/debates")
        assert result == "/api/v1/debates"


class TestResponseHeaders:
    """Tests for response header generation."""

    def test_legacy_headers(self):
        """Legacy requests should include migration instructions."""
        headers = version_response_headers(APIVersion.V1, is_legacy=True)

        assert headers["X-API-Version"] == "v1"
        assert headers["X-API-Legacy"] == "true"
        assert "v2" in headers["X-API-Migration"]

    def test_deprecated_version_headers(self):
        """Deprecated versions should include sunset headers."""
        headers = version_response_headers(APIVersion.V1, is_legacy=False)

        assert headers["X-API-Deprecated"] == "true"
        assert "Sunset" in headers
        assert headers["Deprecation"] == "true"

    def test_current_version_headers(self):
        """Current version should not have deprecation headers."""
        headers = version_response_headers(APIVersion.V2, is_legacy=False)

        assert headers["X-API-Version"] == "v2"
        assert "X-API-Deprecated" not in headers
        assert "Sunset" not in headers

    def test_supported_versions_header(self):
        """Should list all supported versions."""
        headers = version_response_headers(APIVersion.V1)

        assert "X-API-Supported-Versions" in headers
        assert "v1" in headers["X-API-Supported-Versions"]
        assert "v2" in headers["X-API-Supported-Versions"]
