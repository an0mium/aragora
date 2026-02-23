"""Tests for feature_flags handler (aragora/server/handlers/feature_flags.py).

Covers all routes and behavior of the FeatureFlagsHandler class:
- can_handle() routing for versioned and unversioned paths
- GET /api/v1/feature-flags (list all flags)
- GET /api/v1/feature-flags/:name (get specific flag)
- Category filtering via query param
- Method not allowed (non-GET)
- FLAGS_AVAILABLE=False (503)
- Flag not found (404)
- Empty flag name (400)
- Invalid category (400)
- RBAC permission checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.feature_flags import FeatureFlagsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for FeatureFlagsHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock flag system enums and dataclasses
# ---------------------------------------------------------------------------


class MockFlagCategory(str, Enum):
    CORE = "core"
    KNOWLEDGE = "knowledge"
    PERFORMANCE = "performance"
    EXPERIMENTAL = "experimental"
    DEBUG = "debug"


class MockFlagStatus(str, Enum):
    ACTIVE = "active"
    BETA = "beta"
    DEPRECATED = "deprecated"


@dataclass
class MockFlagDefinition:
    name: str
    flag_type: type
    default: Any
    description: str
    category: MockFlagCategory
    status: MockFlagStatus
    env_var: str | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a FeatureFlagsHandler instance."""
    return FeatureFlagsHandler(server_context={})


@pytest.fixture
def http_get():
    """Create a mock GET HTTP handler."""
    return MockHTTPHandler(method="GET")


@pytest.fixture
def http_post():
    """Create a mock POST HTTP handler."""
    return MockHTTPHandler(method="POST")


@pytest.fixture
def http_put():
    """Create a mock PUT HTTP handler."""
    return MockHTTPHandler(method="PUT")


@pytest.fixture
def http_delete():
    """Create a mock DELETE HTTP handler."""
    return MockHTTPHandler(method="DELETE")


def _make_flag(
    name: str = "test_flag",
    default: Any = True,
    description: str = "A test flag",
    category: MockFlagCategory = MockFlagCategory.CORE,
    status: MockFlagStatus = MockFlagStatus.ACTIVE,
) -> MockFlagDefinition:
    return MockFlagDefinition(
        name=name,
        flag_type=bool,
        default=default,
        description=description,
        category=category,
        status=status,
    )


def _mock_registry(
    flags: list[MockFlagDefinition] | None = None,
    value_map: dict[str, Any] | None = None,
    definition_map: dict[str, MockFlagDefinition] | None = None,
):
    """Build a mock registry that responds to get_all_flags, get_value, get_definition."""
    registry = MagicMock()
    flags = flags or []
    value_map = value_map or {}
    definition_map = definition_map or {}

    registry.get_all_flags.return_value = flags

    def _get_value(name, default=None):
        return value_map.get(name, default)

    registry.get_value.side_effect = _get_value

    def _get_definition(name):
        return definition_map.get(name)

    registry.get_definition.side_effect = _get_definition

    return registry


# ---------------------------------------------------------------------------
# Tests: can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test route matching via can_handle()."""

    def test_versioned_list_path(self, handler):
        assert handler.can_handle("/api/v1/feature-flags") is True

    def test_versioned_single_flag_path(self, handler):
        assert handler.can_handle("/api/v1/feature-flags/my_flag") is True

    def test_versioned_nested_flag_path(self, handler):
        assert handler.can_handle("/api/v1/feature-flags/some/nested") is True

    def test_unversioned_list_path(self, handler):
        assert handler.can_handle("/api/feature-flags") is True

    def test_unversioned_single_flag_path(self, handler):
        assert handler.can_handle("/api/feature-flags/my_flag") is True

    def test_v2_versioned_list_path(self, handler):
        assert handler.can_handle("/api/v2/feature-flags") is True

    def test_v2_versioned_single_flag_path(self, handler):
        assert handler.can_handle("/api/v2/feature-flags/enable_tls") is True

    def test_unrelated_path_returns_false(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_prefix_returns_false(self, handler):
        assert handler.can_handle("/api/v1/feature-flag") is False

    def test_root_path_returns_false(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path_returns_false(self, handler):
        assert handler.can_handle("") is False

    def test_feature_flags_settings_returns_false(self, handler):
        assert handler.can_handle("/api/v1/feature-flags-settings") is False

    def test_similar_prefix_no_slash_returns_false(self, handler):
        """feature-flagsXYZ should not match since there's no slash after prefix."""
        assert handler.can_handle("/api/v1/feature-flagsxyz") is False


# ---------------------------------------------------------------------------
# Tests: method not allowed
# ---------------------------------------------------------------------------


class TestMethodNotAllowed:
    """Test that non-GET methods return 405."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_post_returns_405(self, handler, http_post):
        result = handler.handle("/api/v1/feature-flags", {}, http_post)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_put_returns_405(self, handler, http_put):
        result = handler.handle("/api/v1/feature-flags", {}, http_put)
        assert _status(result) == 405

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_delete_returns_405(self, handler, http_delete):
        result = handler.handle("/api/v1/feature-flags", {}, http_delete)
        assert _status(result) == 405

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_post_on_specific_flag_returns_405(self, handler, http_post):
        result = handler.handle("/api/v1/feature-flags/my_flag", {}, http_post)
        assert _status(result) == 405


# ---------------------------------------------------------------------------
# Tests: FLAGS_AVAILABLE = False (503)
# ---------------------------------------------------------------------------


class TestFlagsUnavailable:
    """When the feature flag module cannot be imported, return 503."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", False)
    def test_list_returns_503_when_unavailable(self, handler, http_get):
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", False)
    def test_get_specific_returns_503_when_unavailable(self, handler, http_get):
        result = handler.handle("/api/v1/feature-flags/some_flag", {}, http_get)
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# Tests: list flags (GET /api/v1/feature-flags)
# ---------------------------------------------------------------------------


class TestListFlags:
    """Test listing all feature flags."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_empty_registry(self, mock_get_reg, handler, http_get):
        mock_get_reg.return_value = _mock_registry(flags=[])
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["flags"] == []
        assert body["total"] == 0

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_single_flag(self, mock_get_reg, handler, http_get):
        flag = _make_flag("enable_tls", True, "Enable TLS")
        registry = _mock_registry(flags=[flag], value_map={"enable_tls": True})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["flags"][0]["name"] == "enable_tls"
        assert body["flags"][0]["value"] is True
        assert body["flags"][0]["description"] == "Enable TLS"
        assert body["flags"][0]["category"] == "core"
        assert body["flags"][0]["status"] == "active"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_multiple_flags(self, mock_get_reg, handler, http_get):
        flags = [
            _make_flag("flag_a", True, "Flag A"),
            _make_flag("flag_b", False, "Flag B", category=MockFlagCategory.EXPERIMENTAL),
            _make_flag("flag_c", 42, "Flag C", status=MockFlagStatus.BETA),
        ]
        value_map = {"flag_a": True, "flag_b": False, "flag_c": 42}
        registry = _mock_registry(flags=flags, value_map=value_map)
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        names = [f["name"] for f in body["flags"]]
        assert "flag_a" in names
        assert "flag_b" in names
        assert "flag_c" in names

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_flag_value_falls_back_to_default(self, mock_get_reg, handler, http_get):
        flag = _make_flag("test_flag", default="fallback_val", description="Test")
        # value_map does NOT have the key, so get_value will return the default passed in
        registry = _mock_registry(flags=[flag], value_map={})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        # The handler calls registry.get_value(flag.name, flag.default)
        # Our mock side_effect returns default when name not in value_map
        assert body["flags"][0]["value"] == "fallback_val"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_flags_includes_category_and_status(self, mock_get_reg, handler, http_get):
        flag = _make_flag(
            "exp_flag",
            False,
            "Experimental",
            category=MockFlagCategory.EXPERIMENTAL,
            status=MockFlagStatus.BETA,
        )
        registry = _mock_registry(flags=[flag], value_map={"exp_flag": False})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        body = _body(result)
        assert body["flags"][0]["category"] == "experimental"
        assert body["flags"][0]["status"] == "beta"


# ---------------------------------------------------------------------------
# Tests: category filtering
# ---------------------------------------------------------------------------


class TestCategoryFilter:
    """Test category query parameter filtering."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.FlagCategory", MockFlagCategory)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_filter_by_valid_category(self, mock_get_reg, handler, http_get):
        flag = _make_flag("perf_flag", True, "Performance flag", category=MockFlagCategory.PERFORMANCE)
        registry = _mock_registry(flags=[flag], value_map={"perf_flag": True})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {"category": "performance"}, http_get)
        assert _status(result) == 200
        # Registry's get_all_flags was called with category=performance
        registry.get_all_flags.assert_called_once_with(category=MockFlagCategory.PERFORMANCE)

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.FlagCategory", MockFlagCategory)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_filter_by_invalid_category_returns_400(self, mock_get_reg, handler, http_get):
        result = handler.handle("/api/v1/feature-flags", {"category": "nonexistent"}, http_get)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid category" in body.get("error", "")

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.FlagCategory", MockFlagCategory)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_invalid_category_lists_valid_options(self, mock_get_reg, handler, http_get):
        result = handler.handle("/api/v1/feature-flags", {"category": "bad"}, http_get)
        assert _status(result) == 400
        body = _body(result)
        error_msg = body.get("error", "")
        # Check valid categories are listed
        assert "core" in error_msg
        assert "knowledge" in error_msg
        assert "performance" in error_msg

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.FlagCategory", MockFlagCategory)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_no_category_param_returns_all(self, mock_get_reg, handler, http_get):
        flags = [
            _make_flag("a", True, "A"),
            _make_flag("b", False, "B", category=MockFlagCategory.KNOWLEDGE),
        ]
        registry = _mock_registry(flags=flags, value_map={"a": True, "b": False})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert _status(result) == 200
        registry.get_all_flags.assert_called_once_with(category=None)

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.FlagCategory", MockFlagCategory)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_empty_category_string_returns_all(self, mock_get_reg, handler, http_get):
        """An empty string for category should be treated as no filter."""
        registry = _mock_registry(flags=[])
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {"category": ""}, http_get)
        assert _status(result) == 200
        registry.get_all_flags.assert_called_once_with(category=None)


# ---------------------------------------------------------------------------
# Tests: get specific flag (GET /api/v1/feature-flags/:name)
# ---------------------------------------------------------------------------


class TestGetFlag:
    """Test getting a specific flag by name."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_existing_flag(self, mock_get_reg, handler, http_get):
        flag = _make_flag("my_flag", True, "My flag description")
        registry = _mock_registry(
            definition_map={"my_flag": flag},
            value_map={"my_flag": True},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/my_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "my_flag"
        assert body["value"] is True
        assert body["description"] == "My flag description"
        assert body["category"] == "core"
        assert body["status"] == "active"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_nonexistent_flag_returns_404(self, mock_get_reg, handler, http_get):
        registry = _mock_registry(definition_map={})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/missing_flag", {}, http_get)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()
        assert "missing_flag" in body.get("error", "")

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_flag_with_beta_status(self, mock_get_reg, handler, http_get):
        flag = _make_flag("beta_flag", False, "Beta feature", status=MockFlagStatus.BETA)
        registry = _mock_registry(
            definition_map={"beta_flag": flag},
            value_map={"beta_flag": False},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/beta_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "beta"
        assert body["value"] is False

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_flag_with_deprecated_status(self, mock_get_reg, handler, http_get):
        flag = _make_flag(
            "old_flag", True, "Deprecated flag", status=MockFlagStatus.DEPRECATED
        )
        registry = _mock_registry(
            definition_map={"old_flag": flag},
            value_map={"old_flag": True},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/old_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "deprecated"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_flag_with_experimental_category(self, mock_get_reg, handler, http_get):
        flag = _make_flag(
            "exp_flag",
            "some_value",
            "Experimental",
            category=MockFlagCategory.EXPERIMENTAL,
        )
        registry = _mock_registry(
            definition_map={"exp_flag": flag},
            value_map={"exp_flag": "some_value"},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/exp_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["category"] == "experimental"
        assert body["value"] == "some_value"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_flag_value_uses_default_from_definition(self, mock_get_reg, handler, http_get):
        """When get_value returns the default, we get the definition default."""
        flag = _make_flag("default_flag", default=99, description="Uses default")
        registry = _mock_registry(
            definition_map={"default_flag": flag},
            value_map={},  # not in map, so returns the default arg
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/default_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] == 99

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_flag_integer_value(self, mock_get_reg, handler, http_get):
        flag = _make_flag("max_retries", default=3, description="Max retries")
        registry = _mock_registry(
            definition_map={"max_retries": flag},
            value_map={"max_retries": 5},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/max_retries", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] == 5

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_get_flag_string_value(self, mock_get_reg, handler, http_get):
        flag = _make_flag("log_level", default="info", description="Log level")
        registry = _mock_registry(
            definition_map={"log_level": flag},
            value_map={"log_level": "debug"},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/log_level", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] == "debug"


# ---------------------------------------------------------------------------
# Tests: empty flag name
# ---------------------------------------------------------------------------


class TestEmptyFlagName:
    """Trailing slash with no flag name should return 400."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_trailing_slash_returns_400(self, handler, http_get):
        result = handler.handle("/api/v1/feature-flags/", {}, http_get)
        assert _status(result) == 400
        body = _body(result)
        assert "required" in body.get("error", "").lower()


# ---------------------------------------------------------------------------
# Tests: unmatched sub-paths (returns None)
# ---------------------------------------------------------------------------


class TestUnmatchedPaths:
    """Test that paths not matching either list or detail return None."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_completely_unrelated_path_returns_none(self, handler, http_get):
        result = handler.handle("/api/v1/debates", {}, http_get)
        # Even though can_handle returns False, handle() should return None for non-matching
        assert result is None or _status(result) == 405

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_non_feature_flags_api_path(self, handler, http_get):
        # Calling handle on a path that doesn't match /api/feature-flags* after stripping
        result = handler.handle("/api/v1/other-endpoint", {}, http_get)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: ROUTES attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test the ROUTES class attribute."""

    def test_routes_contains_list_route(self, handler):
        assert "/api/v1/feature-flags" in handler.ROUTES

    def test_routes_contains_wildcard_route(self, handler):
        assert "/api/v1/feature-flags/*" in handler.ROUTES

    def test_routes_length(self, handler):
        assert len(handler.ROUTES) == 2


# ---------------------------------------------------------------------------
# Tests: flag data serialization
# ---------------------------------------------------------------------------


class TestFlagSerialization:
    """Test that flag data is correctly serialized in responses."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_response_has_flags_and_total_keys(self, mock_get_reg, handler, http_get):
        registry = _mock_registry(flags=[])
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        body = _body(result)
        assert "flags" in body
        assert "total" in body

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_item_has_all_expected_fields(self, mock_get_reg, handler, http_get):
        flag = _make_flag("full_flag", True, "Full description")
        registry = _mock_registry(flags=[flag], value_map={"full_flag": True})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        body = _body(result)
        item = body["flags"][0]
        assert set(item.keys()) == {"name", "value", "description", "category", "status"}

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_detail_response_has_all_expected_fields(self, mock_get_reg, handler, http_get):
        flag = _make_flag("detail_flag", True, "Detail description")
        registry = _mock_registry(
            definition_map={"detail_flag": flag},
            value_map={"detail_flag": True},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/detail_flag", {}, http_get)
        body = _body(result)
        assert set(body.keys()) == {"name", "value", "description", "category", "status"}

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_total_matches_flags_length(self, mock_get_reg, handler, http_get):
        flags = [_make_flag(f"f{i}", True, f"Flag {i}") for i in range(5)]
        value_map = {f"f{i}": True for i in range(5)}
        registry = _mock_registry(flags=flags, value_map=value_map)
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        body = _body(result)
        assert body["total"] == 5
        assert body["total"] == len(body["flags"])

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_flag_none_value(self, mock_get_reg, handler, http_get):
        """A flag with None value should serialize correctly."""
        flag = _make_flag("nullable_flag", None, "Nullable flag")
        registry = _mock_registry(
            definition_map={"nullable_flag": flag},
            value_map={"nullable_flag": None},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/nullable_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] is None

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_flag_list_value(self, mock_get_reg, handler, http_get):
        """A flag whose value is a list should be JSON-serializable."""
        flag = _make_flag("list_flag", [], "A list flag")
        registry = _mock_registry(
            definition_map={"list_flag": flag},
            value_map={"list_flag": ["a", "b"]},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/list_flag", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests: response content type
# ---------------------------------------------------------------------------


class TestResponseContentType:
    """All responses should be JSON."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_response_is_json(self, mock_get_reg, handler, http_get):
        mock_get_reg.return_value = _mock_registry(flags=[])
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert result.content_type == "application/json"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_detail_response_is_json(self, mock_get_reg, handler, http_get):
        flag = _make_flag("json_flag", True, "JSON")
        registry = _mock_registry(
            definition_map={"json_flag": flag},
            value_map={"json_flag": True},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/json_flag", {}, http_get)
        assert result.content_type == "application/json"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    def test_error_response_is_json(self, handler, http_post):
        result = handler.handle("/api/v1/feature-flags", {}, http_post)
        assert result.content_type == "application/json"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", False)
    def test_503_response_is_json(self, handler, http_get):
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        assert result.content_type == "application/json"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case scenarios."""

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_flag_name_with_special_characters(self, mock_get_reg, handler, http_get):
        """Flag names with dots, hyphens, underscores should work."""
        flag = _make_flag("enable.tls-v1_3", True, "TLS flag")
        registry = _mock_registry(
            definition_map={"enable.tls-v1_3": flag},
            value_map={"enable.tls-v1_3": True},
        )
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags/enable.tls-v1_3", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "enable.tls-v1_3"

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_registry_get_all_flags_called_for_list(self, mock_get_reg, handler, http_get):
        registry = _mock_registry(flags=[])
        mock_get_reg.return_value = registry
        handler.handle("/api/v1/feature-flags", {}, http_get)
        registry.get_all_flags.assert_called_once()

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_registry_get_definition_called_for_detail(self, mock_get_reg, handler, http_get):
        flag = _make_flag("check_call", True, "Verify call")
        registry = _mock_registry(
            definition_map={"check_call": flag},
            value_map={"check_call": True},
        )
        mock_get_reg.return_value = registry
        handler.handle("/api/v1/feature-flags/check_call", {}, http_get)
        registry.get_definition.assert_called_once_with("check_call")

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_registry_get_value_called_with_correct_default(self, mock_get_reg, handler, http_get):
        flag = _make_flag("defaulted_flag", default=42, description="Defaulted")
        registry = _mock_registry(
            definition_map={"defaulted_flag": flag},
            value_map={"defaulted_flag": 42},
        )
        mock_get_reg.return_value = registry
        handler.handle("/api/v1/feature-flags/defaulted_flag", {}, http_get)
        registry.get_value.assert_called_once_with("defaulted_flag", 42)

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_list_flags_calls_get_value_per_flag(self, mock_get_reg, handler, http_get):
        flags = [_make_flag(f"f{i}", True, f"F{i}") for i in range(3)]
        value_map = {f"f{i}": True for i in range(3)}
        registry = _mock_registry(flags=flags, value_map=value_map)
        mock_get_reg.return_value = registry
        handler.handle("/api/v1/feature-flags", {}, http_get)
        assert registry.get_value.call_count == 3

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_method_not_allowed_checked_before_flags_available(self, mock_get_reg, handler, http_post):
        """Method check happens first, before flags availability check."""
        result = handler.handle("/api/v1/feature-flags", {}, http_post)
        assert _status(result) == 405
        # get_flag_registry should not be called for wrong method
        mock_get_reg.assert_not_called()

    def test_handler_is_instance_of_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler
        assert isinstance(handler, BaseHandler)

    @patch("aragora.server.handlers.feature_flags.FLAGS_AVAILABLE", True)
    @patch("aragora.server.handlers.feature_flags.get_flag_registry")
    def test_flag_with_debug_category(self, mock_get_reg, handler, http_get):
        flag = _make_flag("debug_mode", False, "Debug mode", category=MockFlagCategory.DEBUG)
        registry = _mock_registry(flags=[flag], value_map={"debug_mode": False})
        mock_get_reg.return_value = registry
        result = handler.handle("/api/v1/feature-flags", {}, http_get)
        body = _body(result)
        assert body["flags"][0]["category"] == "debug"
