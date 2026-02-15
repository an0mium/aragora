"""Tests for feature flag administration endpoints.

Tests:
- GET /api/v1/admin/feature-flags - list all flags
- GET /api/v1/admin/feature-flags/:name - get single flag with usage
- PUT /api/v1/admin/feature-flags/:name - toggle/set flag value
- Category and status filtering
- Error cases (not found, invalid body, invalid filter values)
"""

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.config.feature_flags import (
    FlagCategory,
    FlagDefinition,
    FlagStatus,
    FlagUsage,
    FeatureFlagRegistry,
    RegistryStats,
    get_flag_registry,
    reset_flag_registry,
)
from aragora.server.handlers.admin.feature_flags import FeatureFlagAdminHandler


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset flag registry and clean up env vars before and after each test."""
    reset_flag_registry()
    # Snapshot env vars with ARAGORA_ prefix that are test-related
    test_env_keys = {
        "ARAGORA_TEST_FLAG_ALPHA",
        "ARAGORA_TEST_FLAG_ACTIVE",
        "ARAGORA_TEST_INT_FLAG",
        "ARAGORA_TEST_DEPRECATED_FLAG",
    }
    old_values = {k: os.environ.get(k) for k in test_env_keys}
    yield
    reset_flag_registry()
    # Restore env vars
    for k, v in old_values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def handler():
    """Create a FeatureFlagAdminHandler instance."""
    return FeatureFlagAdminHandler(ctx={})


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    h = MagicMock()
    h.headers = {"Content-Type": "application/json"}
    return h


@pytest.fixture
def registry():
    """Get a fresh flag registry with test flags registered."""
    reg = get_flag_registry()
    reg.register(
        "test_flag_alpha",
        bool,
        False,
        "An alpha test flag",
        FlagCategory.EXPERIMENTAL,
        FlagStatus.ALPHA,
    )
    reg.register(
        "test_flag_active",
        bool,
        True,
        "An active test flag",
        FlagCategory.CORE,
        FlagStatus.ACTIVE,
    )
    reg.register(
        "test_int_flag",
        int,
        42,
        "An integer flag",
        FlagCategory.PERFORMANCE,
        FlagStatus.ACTIVE,
    )
    reg.register(
        "test_deprecated_flag",
        bool,
        False,
        "A deprecated flag",
        FlagCategory.DEPRECATED,
        FlagStatus.DEPRECATED,
        deprecated_since="1.0.0",
        removed_in="2.0.0",
        replacement="test_flag_active",
    )
    return reg


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_feature_flags_path(self, handler):
        result = handler.can_handle("/api/v1/admin/feature-flags")
        assert result is True

    def test_handles_specific_flag_path(self, handler):
        result = handler.can_handle("/api/v1/admin/feature-flags/test_flag")
        assert result is True

    def test_does_not_handle_other_path(self, handler):
        result = handler.can_handle("/api/v1/admin/users")
        assert result is False

    def test_handles_without_version_prefix(self, handler):
        result = handler.can_handle("/api/admin/feature-flags")
        assert result is True


class TestListFlags:
    """Tests for GET /api/v1/admin/feature-flags."""

    def test_list_all_flags(self, handler, mock_handler, registry):
        result = handler.handle("/api/v1/admin/feature-flags", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "flags" in body
        assert "total" in body
        assert "stats" in body
        assert body["total"] > 0

    def test_list_flags_includes_builtin(self, handler, mock_handler, registry):
        """Built-in flags from registry should appear in listing."""
        result = handler.handle("/api/v1/admin/feature-flags", {}, mock_handler)
        body = parse_body(result)
        names = [f["name"] for f in body["flags"]]
        assert "test_flag_active" in names
        assert "test_flag_alpha" in names

    def test_flag_fields_present(self, handler, mock_handler, registry):
        result = handler.handle("/api/v1/admin/feature-flags", {}, mock_handler)
        body = parse_body(result)
        flag = next(f for f in body["flags"] if f["name"] == "test_flag_active")
        assert flag["value"] is True
        assert flag["default"] is True
        assert flag["type"] == "bool"
        assert flag["category"] == "core"
        assert flag["status"] == "active"
        assert flag["description"] == "An active test flag"
        assert "env_var" in flag

    def test_filter_by_category(self, handler, mock_handler, registry):
        result = handler.handle(
            "/api/v1/admin/feature-flags",
            {"category": "experimental"},
            mock_handler,
        )
        body = parse_body(result)
        for flag in body["flags"]:
            assert flag["category"] == "experimental"

    def test_filter_by_status(self, handler, mock_handler, registry):
        result = handler.handle(
            "/api/v1/admin/feature-flags",
            {"status": "alpha"},
            mock_handler,
        )
        body = parse_body(result)
        for flag in body["flags"]:
            assert flag["status"] == "alpha"

    def test_invalid_category_returns_400(self, handler, mock_handler, registry):
        result = handler.handle(
            "/api/v1/admin/feature-flags",
            {"category": "nonexistent"},
            mock_handler,
        )
        assert result.status_code == 400
        body = parse_body(result)
        assert "Invalid category" in body.get("error", "")

    def test_invalid_status_returns_400(self, handler, mock_handler, registry):
        result = handler.handle(
            "/api/v1/admin/feature-flags",
            {"status": "nonexistent"},
            mock_handler,
        )
        assert result.status_code == 400
        body = parse_body(result)
        assert "Invalid status" in body.get("error", "")

    def test_stats_in_response(self, handler, mock_handler, registry):
        result = handler.handle("/api/v1/admin/feature-flags", {}, mock_handler)
        body = parse_body(result)
        stats = body["stats"]
        assert "total_flags" in stats
        assert "active_flags" in stats
        assert "deprecated_flags" in stats
        assert "flags_by_category" in stats

    def test_empty_filter_returns_empty(self, handler, mock_handler, registry):
        """Filtering by a status with no matching flags returns empty."""
        result = handler.handle(
            "/api/v1/admin/feature-flags",
            {"status": "removed"},
            mock_handler,
        )
        body = parse_body(result)
        assert body["total"] == 0
        assert body["flags"] == []


class TestGetFlag:
    """Tests for GET /api/v1/admin/feature-flags/:name."""

    def test_get_existing_flag(self, handler, mock_handler, registry):
        result = handler.handle("/api/v1/admin/feature-flags/test_flag_active", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["name"] == "test_flag_active"
        assert body["value"] is True
        assert body["type"] == "bool"

    def test_get_flag_not_found(self, handler, mock_handler, registry):
        result = handler.handle("/api/v1/admin/feature-flags/nonexistent_flag", {}, mock_handler)
        assert result.status_code == 404
        body = parse_body(result)
        assert "not found" in body.get("error", "").lower()

    def test_get_flag_includes_usage(self, handler, mock_handler, registry):
        # Access the flag to generate usage data
        registry.get_value("test_flag_active")
        registry.get_value("test_flag_active")

        result = handler.handle("/api/v1/admin/feature-flags/test_flag_active", {}, mock_handler)
        body = parse_body(result)
        assert "usage" in body
        # access_count includes calls from _list (get_value) + 2 explicit
        assert body["usage"]["access_count"] >= 2

    def test_get_deprecated_flag_includes_metadata(self, handler, mock_handler, registry):
        result = handler.handle(
            "/api/v1/admin/feature-flags/test_deprecated_flag", {}, mock_handler
        )
        body = parse_body(result)
        assert body["deprecated_since"] == "1.0.0"
        assert body["removed_in"] == "2.0.0"
        assert body["replacement"] == "test_flag_active"

    def test_get_int_flag(self, handler, mock_handler, registry):
        result = handler.handle("/api/v1/admin/feature-flags/test_int_flag", {}, mock_handler)
        body = parse_body(result)
        assert body["value"] == 42
        assert body["type"] == "int"

    def test_get_flag_empty_name(self, handler, mock_handler, registry):
        """Trailing slash without flag name returns 400."""
        result = handler.handle("/api/v1/admin/feature-flags/", {}, mock_handler)
        assert result.status_code == 400


class TestSetFlag:
    """Tests for PUT /api/v1/admin/feature-flags/:name."""

    def _make_put_handler(self, body: dict) -> MagicMock:
        """Create a mock handler with JSON body for PUT requests."""
        h = MagicMock()
        raw = json.dumps(body).encode("utf-8")
        h.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        }
        h.rfile = MagicMock()
        h.rfile.read.return_value = raw
        return h

    def test_set_bool_flag(self, handler, registry):
        mock_h = self._make_put_handler({"value": True})
        result = handler.handle_put("/api/v1/admin/feature-flags/test_flag_alpha", {}, mock_h)
        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["name"] == "test_flag_alpha"
        assert body["value"] is True
        assert body["updated"] is True

    def test_set_int_flag(self, handler, registry):
        mock_h = self._make_put_handler({"value": 100})
        result = handler.handle_put("/api/v1/admin/feature-flags/test_int_flag", {}, mock_h)
        assert result.status_code == 200
        body = parse_body(result)
        assert body["value"] == 100

    def test_set_flag_type_mismatch(self, handler, registry):
        mock_h = self._make_put_handler({"value": "not_a_bool"})
        result = handler.handle_put("/api/v1/admin/feature-flags/test_flag_active", {}, mock_h)
        assert result.status_code == 400
        body = parse_body(result)
        assert "expects bool" in body.get("error", "")

    def test_set_flag_not_found(self, handler, registry):
        mock_h = self._make_put_handler({"value": True})
        result = handler.handle_put("/api/v1/admin/feature-flags/nonexistent", {}, mock_h)
        assert result.status_code == 404

    def test_set_flag_missing_value(self, handler, registry):
        mock_h = self._make_put_handler({"other": "data"})
        result = handler.handle_put("/api/v1/admin/feature-flags/test_flag_active", {}, mock_h)
        assert result.status_code == 400
        body = parse_body(result)
        assert "value" in body.get("error", "").lower()

    def test_set_flag_invalid_json(self, handler, registry):
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "application/json", "Content-Length": "0"}
        mock_h.rfile = MagicMock()
        mock_h.rfile.read.return_value = b""
        result = handler.handle_put("/api/v1/admin/feature-flags/test_flag_active", {}, mock_h)
        assert result.status_code == 400

    def test_set_flag_sets_env_var(self, handler, registry):
        mock_h = self._make_put_handler({"value": True})
        flag = registry.get_definition("test_flag_alpha")
        env_var = flag.env_var

        # Clean up env
        old_val = os.environ.pop(env_var, None)
        try:
            handler.handle_put("/api/v1/admin/feature-flags/test_flag_alpha", {}, mock_h)
            assert os.environ.get(env_var) == "True"
        finally:
            if old_val is not None:
                os.environ[env_var] = old_val
            else:
                os.environ.pop(env_var, None)

    def test_set_flag_response_includes_previous_default(self, handler, registry):
        mock_h = self._make_put_handler({"value": True})
        result = handler.handle_put("/api/v1/admin/feature-flags/test_flag_alpha", {}, mock_h)
        body = parse_body(result)
        assert body["previous_default"] is False

    def test_put_empty_flag_name(self, handler, registry):
        mock_h = self._make_put_handler({"value": True})
        result = handler.handle_put("/api/v1/admin/feature-flags/", {}, mock_h)
        assert result.status_code == 400


class TestUnavailable:
    """Tests for when feature flag system is not available."""

    def test_list_unavailable(self, handler, mock_handler):
        with patch("aragora.server.handlers.admin.feature_flags.FLAGS_AVAILABLE", False):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_handler)
            assert result.status_code == 503

    def test_get_unavailable(self, handler, mock_handler):
        with patch("aragora.server.handlers.admin.feature_flags.FLAGS_AVAILABLE", False):
            result = handler.handle("/api/v1/admin/feature-flags/test", {}, mock_handler)
            assert result.status_code == 503

    def test_put_unavailable(self, handler, mock_handler):
        with patch("aragora.server.handlers.admin.feature_flags.FLAGS_AVAILABLE", False):
            result = handler.handle_put("/api/v1/admin/feature-flags/test", {}, mock_handler)
            assert result.status_code == 503


class TestUnhandledRoutes:
    """Tests for paths not handled by this handler."""

    def test_unrelated_path_returns_none(self, handler, mock_handler):
        result = handler.handle("/api/v1/admin/users", {}, mock_handler)
        assert result is None

    def test_put_unrelated_path_returns_none(self, handler, mock_handler):
        result = handler.handle_put("/api/v1/admin/users", {}, mock_handler)
        assert result is None
