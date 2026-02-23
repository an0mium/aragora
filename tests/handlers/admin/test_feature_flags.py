"""Tests for FeatureFlagAdminHandler in aragora/server/handlers/admin/feature_flags.py.

Comprehensive coverage of all endpoints:
- GET /api/v1/admin/feature-flags           (_list_flags)
- GET /api/v1/admin/feature-flags/:name     (_get_flag)
- PUT /api/v1/admin/feature-flags/:name     (_set_flag via handle_put)

Also covers:
- can_handle() path matching
- Version prefix stripping (/api/v1/... -> /api/...)
- FLAGS_AVAILABLE=False fallback (503)
- RBAC decorator pass-through (auto-auth fixture)
- rate_limit decorator pass-through
- Category/status query filtering
- Flag type validation on PUT
- Missing/invalid JSON body handling
- Deprecated/alpha flag metadata (deprecated_since, removed_in, replacement)
- Usage stats in GET single flag
- Environment variable override on PUT
- Edge cases: empty name, unknown flags, trailing slashes
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch
from collections import defaultdict

import pytest

from aragora.server.handlers.admin.feature_flags import FeatureFlagAdminHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


def _make_http_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    h = MagicMock()
    h.command = "GET"
    h.client_address = ("127.0.0.1", 12345)
    if body is not None:
        body_bytes = json.dumps(body).encode()
        h.rfile.read.return_value = body_bytes
        h.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
    else:
        h.rfile.read.return_value = b"{}"
        h.headers = {"Content-Type": "application/json", "Content-Length": "0"}
    return h


# ===========================================================================
# Mock feature flag types (to avoid coupling tests to exact builtin flags)
# ===========================================================================


class MockFlagCategory(str, Enum):
    CORE = "core"
    KNOWLEDGE = "knowledge"
    PERFORMANCE = "performance"
    EXPERIMENTAL = "experimental"
    DEBUG = "debug"


class MockFlagStatus(str, Enum):
    ACTIVE = "active"
    BETA = "beta"
    ALPHA = "alpha"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass
class MockFlagDefinition:
    name: str
    flag_type: type
    default: Any
    description: str
    category: MockFlagCategory = MockFlagCategory.CORE
    status: MockFlagStatus = MockFlagStatus.ACTIVE
    env_var: str | None = None
    deprecated_since: str | None = None
    removed_in: str | None = None
    replacement: str | None = None


@dataclass
class MockFlagUsage:
    name: str
    access_count: int = 0
    last_accessed: float | None = None
    access_locations: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class MockRegistryStats:
    total_flags: int = 0
    active_flags: int = 0
    deprecated_flags: int = 0
    flags_by_category: dict[str, int] = field(default_factory=dict)
    total_accesses: int = 0
    unknown_accesses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_flags": self.total_flags,
            "active_flags": self.active_flags,
            "deprecated_flags": self.deprecated_flags,
            "flags_by_category": self.flags_by_category,
            "total_accesses": self.total_accesses,
            "unknown_accesses": self.unknown_accesses,
        }


class MockFlagRegistry:
    """Mock registry that provides controllable flag data for tests."""

    def __init__(self, flags: list[MockFlagDefinition] | None = None):
        self._flags = {f.name: f for f in (flags or [])}
        self._usage: dict[str, MockFlagUsage] = {}
        self._values: dict[str, Any] = {}

    def get_all_flags(
        self,
        category: Any | None = None,
        status: Any | None = None,
    ) -> list[MockFlagDefinition]:
        flags = list(self._flags.values())
        if category:
            flags = [f for f in flags if f.category.value == category.value]
        if status:
            flags = [f for f in flags if f.status.value == status.value]
        return sorted(flags, key=lambda f: (f.category.value, f.name))

    def get_value(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def get_definition(self, name: str) -> MockFlagDefinition | None:
        return self._flags.get(name)

    def get_usage(self, name: str) -> MockFlagUsage | None:
        return self._usage.get(name)

    def get_stats(self) -> MockRegistryStats:
        return MockRegistryStats(
            total_flags=len(self._flags),
            active_flags=sum(
                1 for f in self._flags.values() if f.status == MockFlagStatus.ACTIVE
            ),
            deprecated_flags=sum(
                1 for f in self._flags.values() if f.status == MockFlagStatus.DEPRECATED
            ),
            flags_by_category={
                cat.value: sum(1 for f in self._flags.values() if f.category == cat)
                for cat in MockFlagCategory
                if any(f.category == cat for f in self._flags.values())
            },
            total_accesses=sum(u.access_count for u in self._usage.values()),
            unknown_accesses=0,
        )


# ===========================================================================
# Sample flags for testing
# ===========================================================================


def _sample_flags() -> list[MockFlagDefinition]:
    return [
        MockFlagDefinition(
            name="enable_knowledge_retrieval",
            flag_type=bool,
            default=True,
            description="Query Knowledge Mound before debates",
            category=MockFlagCategory.KNOWLEDGE,
            status=MockFlagStatus.ACTIVE,
            env_var="ARAGORA_ENABLE_KNOWLEDGE_RETRIEVAL",
        ),
        MockFlagDefinition(
            name="enable_checkpointing",
            flag_type=bool,
            default=False,
            description="Auto-create CheckpointManager",
            category=MockFlagCategory.CORE,
            status=MockFlagStatus.ACTIVE,
            env_var="ARAGORA_ENABLE_CHECKPOINTING",
        ),
        MockFlagDefinition(
            name="enable_prompt_evolution",
            flag_type=bool,
            default=False,
            description="Auto-create PromptEvolver for adaptive prompts",
            category=MockFlagCategory.EXPERIMENTAL,
            status=MockFlagStatus.ALPHA,
            env_var="ARAGORA_ENABLE_PROMPT_EVOLUTION",
        ),
        MockFlagDefinition(
            name="max_agent_retries",
            flag_type=int,
            default=3,
            description="Max retries per agent call",
            category=MockFlagCategory.PERFORMANCE,
            status=MockFlagStatus.ACTIVE,
            env_var="ARAGORA_MAX_AGENT_RETRIES",
        ),
        MockFlagDefinition(
            name="old_flag",
            flag_type=bool,
            default=False,
            description="A deprecated flag",
            category=MockFlagCategory.CORE,
            status=MockFlagStatus.DEPRECATED,
            env_var="ARAGORA_OLD_FLAG",
            deprecated_since="2.0.0",
            removed_in="3.0.0",
            replacement="new_flag",
        ),
    ]


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """A FeatureFlagAdminHandler with empty context."""
    return FeatureFlagAdminHandler(ctx={})


@pytest.fixture
def mock_http():
    """Minimal mock HTTP handler."""
    return _make_http_handler()


@pytest.fixture
def mock_registry():
    """A mock flag registry with sample data."""
    reg = MockFlagRegistry(_sample_flags())
    reg._values = {
        "enable_knowledge_retrieval": True,
        "enable_checkpointing": False,
        "enable_prompt_evolution": False,
        "max_agent_retries": 3,
        "old_flag": False,
    }
    reg._usage = {
        "enable_knowledge_retrieval": MockFlagUsage(
            name="enable_knowledge_retrieval",
            access_count=42,
            last_accessed=time.time(),
            access_locations=defaultdict(int, {"orchestrator.py": 30, "bridge.py": 12}),
        ),
        "max_agent_retries": MockFlagUsage(
            name="max_agent_retries",
            access_count=5,
            last_accessed=time.time(),
            access_locations=defaultdict(int, {"runner.py": 5}),
        ),
    }
    return reg


# ===========================================================================
# Patch helper: patches get_flag_registry + FlagCategory + FlagStatus
# ===========================================================================


def _patch_flags_available(registry):
    """Context manager that patches the handler module to use mock registry."""
    return patch.multiple(
        "aragora.server.handlers.admin.feature_flags",
        FLAGS_AVAILABLE=True,
        get_flag_registry=lambda: registry,
        FlagCategory=MockFlagCategory,
        FlagStatus=MockFlagStatus,
    )


def _patch_flags_unavailable():
    """Context manager that marks the feature flag system as unavailable."""
    return patch.object(
        __import__(
            "aragora.server.handlers.admin.feature_flags",
            fromlist=["FLAGS_AVAILABLE"],
        ),
        "FLAGS_AVAILABLE",
        False,
    )


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle path matching."""

    def test_can_handle_list_path(self, handler):
        assert handler.can_handle("/api/v1/admin/feature-flags") is True

    def test_can_handle_list_path_no_version(self, handler):
        assert handler.can_handle("/api/admin/feature-flags") is True

    def test_can_handle_single_flag_path(self, handler):
        assert handler.can_handle("/api/v1/admin/feature-flags/enable_something") is True

    def test_can_handle_single_flag_path_no_version(self, handler):
        assert handler.can_handle("/api/admin/feature-flags/some_flag") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/admin/users") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/admin/feature") is False

    def test_cannot_handle_root(self, handler):
        assert handler.can_handle("/") is False

    def test_cannot_handle_empty(self, handler):
        assert handler.can_handle("") is False

    def test_can_handle_nested_flag_name(self, handler):
        """Flag names with slashes in the wildcard portion."""
        assert handler.can_handle("/api/v1/admin/feature-flags/a/b") is True


# ===========================================================================
# Tests: ROUTES class attribute
# ===========================================================================


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_contains_list_path(self):
        assert "/api/v1/admin/feature-flags" in FeatureFlagAdminHandler.ROUTES

    def test_routes_contains_wildcard_path(self):
        assert "/api/v1/admin/feature-flags/*" in FeatureFlagAdminHandler.ROUTES

    def test_routes_length(self):
        assert len(FeatureFlagAdminHandler.ROUTES) == 2


# ===========================================================================
# Tests: Constructor
# ===========================================================================


class TestConstructor:
    """Tests for handler initialization."""

    def test_default_ctx_is_empty_dict(self):
        h = FeatureFlagAdminHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        h = FeatureFlagAdminHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_none_ctx_becomes_empty_dict(self):
        h = FeatureFlagAdminHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Tests: FLAGS_AVAILABLE = False (503 Service Unavailable)
# ===========================================================================


class TestFlagsUnavailable:
    """Tests when feature flag system is not available (import failed)."""

    def test_handle_get_list_returns_503(self, handler, mock_http):
        with _patch_flags_unavailable():
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    def test_handle_get_single_returns_503(self, handler, mock_http):
        with _patch_flags_unavailable():
            result = handler.handle("/api/v1/admin/feature-flags/some_flag", {}, mock_http)
        assert _status(result) == 503

    def test_handle_put_returns_503(self, handler):
        h = _make_http_handler(body={"value": True})
        with _patch_flags_unavailable():
            result = handler.handle_put("/api/v1/admin/feature-flags/some_flag", {}, h)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]


# ===========================================================================
# Tests: GET /api/v1/admin/feature-flags (list flags)
# ===========================================================================


class TestListFlags:
    """Tests for listing all feature flags."""

    def test_list_all_flags(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "flags" in body
        assert "total" in body
        assert "stats" in body
        assert body["total"] == 5

    def test_list_flags_structure(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        body = _body(result)
        flag = next(f for f in body["flags"] if f["name"] == "enable_knowledge_retrieval")
        assert flag["value"] is True
        assert flag["default"] is True
        assert flag["type"] == "bool"
        assert flag["description"] == "Query Knowledge Mound before debates"
        assert flag["category"] == "knowledge"
        assert flag["status"] == "active"
        assert flag["env_var"] == "ARAGORA_ENABLE_KNOWLEDGE_RETRIEVAL"

    def test_list_flags_includes_int_type(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        body = _body(result)
        flag = next(f for f in body["flags"] if f["name"] == "max_agent_retries")
        assert flag["type"] == "int"
        assert flag["default"] == 3

    def test_list_flags_stats(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        body = _body(result)
        stats = body["stats"]
        assert stats["total_flags"] == 5
        assert stats["active_flags"] == 3
        assert stats["deprecated_flags"] == 1

    def test_list_flags_with_version_prefix(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        assert _status(result) == 200

    def test_list_flags_without_version_prefix(self, handler, mock_http, mock_registry):
        """Path without /v1/ is also handled correctly."""
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/admin/feature-flags", {}, mock_http)
        assert _status(result) == 200

    def test_list_flags_empty_registry(self, handler, mock_http):
        empty_reg = MockFlagRegistry([])
        with _patch_flags_available(empty_reg):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        body = _body(result)
        assert body["total"] == 0
        assert body["flags"] == []

    def test_list_flags_filter_by_category(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "knowledge"},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["flags"][0]["name"] == "enable_knowledge_retrieval"

    def test_list_flags_filter_by_status(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"status": "deprecated"},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["flags"][0]["name"] == "old_flag"

    def test_list_flags_filter_by_category_and_status(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "core", "status": "active"},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["flags"][0]["name"] == "enable_checkpointing"

    def test_list_flags_invalid_category(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "nonexistent_category"},
                mock_http,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid category" in body["error"]
        assert "Valid" in body["error"]

    def test_list_flags_invalid_status(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"status": "nonexistent_status"},
                mock_http,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid status" in body["error"]
        assert "Valid" in body["error"]

    def test_list_flags_filter_no_matches(self, handler, mock_http, mock_registry):
        """Category filter that returns no flags."""
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "debug"},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["flags"] == []

    def test_list_flags_filter_alpha_status(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"status": "alpha"},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["flags"][0]["name"] == "enable_prompt_evolution"


# ===========================================================================
# Tests: GET /api/v1/admin/feature-flags/:name (single flag)
# ===========================================================================


class TestGetSingleFlag:
    """Tests for getting a specific feature flag."""

    def test_get_existing_flag(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/enable_knowledge_retrieval", {}, mock_http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "enable_knowledge_retrieval"
        assert body["value"] is True
        assert body["default"] is True
        assert body["type"] == "bool"
        assert body["category"] == "knowledge"
        assert body["status"] == "active"

    def test_get_flag_with_usage(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/enable_knowledge_retrieval", {}, mock_http
            )
        body = _body(result)
        assert "usage" in body
        assert body["usage"]["access_count"] == 42
        assert body["usage"]["last_accessed"] is not None
        assert "orchestrator.py" in body["usage"]["access_locations"]
        assert body["usage"]["access_locations"]["orchestrator.py"] == 30

    def test_get_flag_without_usage(self, handler, mock_http, mock_registry):
        """Flag that has no usage data."""
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, mock_http
            )
        body = _body(result)
        assert "usage" not in body

    def test_get_deprecated_flag_has_metadata(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/feature-flags/old_flag", {}, mock_http)
        body = _body(result)
        assert body["status"] == "deprecated"
        assert body["deprecated_since"] == "2.0.0"
        assert body["removed_in"] == "3.0.0"
        assert body["replacement"] == "new_flag"

    def test_get_active_flag_no_deprecation_fields(self, handler, mock_http, mock_registry):
        """Active flags should NOT include deprecated_since, removed_in, replacement."""
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, mock_http
            )
        body = _body(result)
        assert "deprecated_since" not in body
        assert "removed_in" not in body
        assert "replacement" not in body

    def test_get_nonexistent_flag(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/nonexistent_flag", {}, mock_http
            )
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_get_flag_without_version_prefix(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/admin/feature-flags/enable_checkpointing", {}, mock_http
            )
        assert _status(result) == 200
        assert _body(result)["name"] == "enable_checkpointing"

    def test_get_int_flag(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/max_agent_retries", {}, mock_http
            )
        body = _body(result)
        assert body["type"] == "int"
        assert body["default"] == 3
        assert body["value"] == 3

    def test_get_flag_int_flag_with_usage(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/max_agent_retries", {}, mock_http
            )
        body = _body(result)
        assert body["usage"]["access_count"] == 5
        assert "runner.py" in body["usage"]["access_locations"]

    def test_get_flag_empty_name_returns_400(self, handler, mock_http, mock_registry):
        """Path /api/admin/feature-flags/ with trailing slash has empty name."""
        with _patch_flags_available(mock_registry):
            result = handler.handle(
                "/api/v1/admin/feature-flags/", {}, mock_http
            )
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()


# ===========================================================================
# Tests: PUT /api/v1/admin/feature-flags/:name (set flag)
# ===========================================================================


class TestSetFlag:
    """Tests for setting a feature flag value."""

    def test_set_bool_flag(self, handler, mock_registry):
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "enable_checkpointing"
        assert body["value"] is True
        assert body["previous_default"] is False
        assert body["updated"] is True

    def test_set_int_flag(self, handler, mock_registry):
        h = _make_http_handler(body={"value": 5})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/max_agent_retries", {}, h
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] == 5
        assert body["previous_default"] == 3

    def test_set_flag_updates_env_var(self, handler, mock_registry):
        """PUT should set the environment variable for the flag."""
        h = _make_http_handler(body={"value": True})
        env_key = "ARAGORA_ENABLE_CHECKPOINTING"
        # Remove the env var if it exists
        os.environ.pop(env_key, None)
        try:
            with _patch_flags_available(mock_registry):
                result = handler.handle_put(
                    "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
                )
            assert _status(result) == 200
            assert os.environ.get(env_key) == "True"
        finally:
            os.environ.pop(env_key, None)

    def test_set_flag_nonexistent(self, handler, mock_registry):
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/nonexistent_flag", {}, h
            )
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_set_flag_invalid_json_body(self, handler, mock_registry):
        """Malformed JSON body."""
        h = MagicMock()
        h.command = "PUT"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {
            "Content-Type": "application/json",
            "Content-Length": "11",
        }
        h.rfile.read.return_value = b"not-json!!!"
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        assert "json" in _body(result)["error"].lower()

    def test_set_flag_missing_value_field(self, handler, mock_registry):
        h = _make_http_handler(body={"not_value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        assert "'value'" in _body(result)["error"]

    def test_set_flag_wrong_type_str_for_bool(self, handler, mock_registry):
        h = _make_http_handler(body={"value": "true"})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        body = _body(result)
        assert "bool" in body["error"]
        assert "str" in body["error"]

    def test_set_flag_wrong_type_str_for_int(self, handler, mock_registry):
        h = _make_http_handler(body={"value": "five"})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/max_agent_retries", {}, h
            )
        assert _status(result) == 400
        body = _body(result)
        assert "int" in body["error"]
        assert "str" in body["error"]

    def test_set_flag_wrong_type_int_for_bool(self, handler, mock_registry):
        """In JSON, 1 is int, not bool."""
        h = _make_http_handler(body={"value": 1})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        body = _body(result)
        assert "bool" in body["error"]
        assert "int" in body["error"]

    def test_set_flag_empty_name_returns_400(self, handler, mock_registry):
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/", {}, h
            )
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_set_flag_without_version_prefix(self, handler, mock_registry):
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 200

    def test_set_flag_false_value(self, handler, mock_registry):
        """Setting a bool flag to False should work."""
        h = _make_http_handler(body={"value": False})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_knowledge_retrieval", {}, h
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] is False
        assert body["updated"] is True

    def test_set_flag_zero_int(self, handler, mock_registry):
        """Setting int flag to 0 is valid."""
        h = _make_http_handler(body={"value": 0})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/max_agent_retries", {}, h
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["value"] == 0

    def test_set_flag_no_env_var(self, handler):
        """Flag with env_var=None should not set any environment variable."""
        flag_no_env = MockFlagDefinition(
            name="no_env_flag",
            flag_type=bool,
            default=False,
            description="A flag with no env var",
            env_var=None,
        )
        reg = MockFlagRegistry([flag_no_env])
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(reg):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/no_env_flag", {}, h
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["updated"] is True

    def test_set_flag_empty_body(self, handler, mock_registry):
        """Empty body ({}) should return 400 for missing 'value'."""
        h = _make_http_handler(body={})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        assert "'value'" in _body(result)["error"]

    def test_set_flag_null_value(self, handler, mock_registry):
        """Setting value to null (None) should fail type validation for bool."""
        h = _make_http_handler(body={"value": None})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        body = _body(result)
        assert "bool" in body["error"]

    def test_set_flag_extra_fields_ignored(self, handler, mock_registry):
        """Extra fields in body should be ignored."""
        h = _make_http_handler(body={"value": True, "extra": "ignored", "reason": "admin request"})
        env_key = "ARAGORA_ENABLE_CHECKPOINTING"
        os.environ.pop(env_key, None)
        try:
            with _patch_flags_available(mock_registry):
                result = handler.handle_put(
                    "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
                )
            assert _status(result) == 200
            assert _body(result)["updated"] is True
        finally:
            os.environ.pop(env_key, None)


# ===========================================================================
# Tests: PUT on unrecognized path (returns None)
# ===========================================================================


class TestPutUnrecognizedPath:
    """Tests for PUT on paths that don't match feature flag patterns."""

    def test_put_unrelated_path_returns_none(self, handler, mock_registry):
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put("/api/v1/admin/users/123", {}, h)
        assert result is None

    def test_put_base_path_without_name(self, handler, mock_registry):
        """PUT to /api/admin/feature-flags (no trailing slash, no name) should return None."""
        h = _make_http_handler(body={"value": True})
        with _patch_flags_available(mock_registry):
            result = handler.handle_put("/api/v1/admin/feature-flags", {}, h)
        assert result is None


# ===========================================================================
# Tests: GET on unrecognized path (returns None)
# ===========================================================================


class TestGetUnrecognizedPath:
    """Tests for GET on paths that don't match feature flag patterns."""

    def test_get_unrelated_path_returns_none(self, handler, mock_http, mock_registry):
        with _patch_flags_available(mock_registry):
            result = handler.handle("/api/v1/admin/other", {}, mock_http)
        assert result is None


# ===========================================================================
# Tests: Integration with real FlagCategory / FlagStatus enums
# ===========================================================================


class TestIntegrationWithRealEnums:
    """Tests that verify the handler works with the actual feature_flags module."""

    def test_list_with_real_registry(self, handler, mock_http):
        """Use the real get_flag_registry to verify integration."""
        from aragora.config.feature_flags import (
            FlagCategory as RealFlagCategory,
            FlagStatus as RealFlagStatus,
            get_flag_registry,
            reset_flag_registry,
        )

        # Use a fresh registry
        reset_flag_registry()
        try:
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["total"] > 0
            assert isinstance(body["flags"], list)
            # Verify flag structure
            flag = body["flags"][0]
            assert "name" in flag
            assert "value" in flag
            assert "type" in flag
            assert "category" in flag
            assert "status" in flag
        finally:
            reset_flag_registry()

    def test_get_single_with_real_registry(self, handler, mock_http):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        try:
            result = handler.handle(
                "/api/v1/admin/feature-flags/enable_knowledge_retrieval", {}, mock_http
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["name"] == "enable_knowledge_retrieval"
            assert body["type"] == "bool"
        finally:
            reset_flag_registry()

    def test_invalid_category_with_real_enums(self, handler, mock_http):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        try:
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "totally_fake"},
                mock_http,
            )
            assert _status(result) == 400
            assert "Invalid category" in _body(result)["error"]
        finally:
            reset_flag_registry()

    def test_invalid_status_with_real_enums(self, handler, mock_http):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        try:
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"status": "totally_fake"},
                mock_http,
            )
            assert _status(result) == 400
            assert "Invalid status" in _body(result)["error"]
        finally:
            reset_flag_registry()

    def test_filter_by_real_category(self, handler, mock_http):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        try:
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "knowledge"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            for flag in body["flags"]:
                assert flag["category"] == "knowledge"
        finally:
            reset_flag_registry()

    def test_filter_by_real_status(self, handler, mock_http):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        try:
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"status": "active"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            for flag in body["flags"]:
                assert flag["status"] == "active"
        finally:
            reset_flag_registry()

    def test_put_with_real_registry(self, handler):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        env_key = "ARAGORA_ENABLE_CHECKPOINTING"
        os.environ.pop(env_key, None)
        try:
            h = _make_http_handler(body={"value": True})
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["updated"] is True
            assert body["value"] is True
            assert os.environ.get(env_key) == "True"
        finally:
            os.environ.pop(env_key, None)
            reset_flag_registry()

    def test_put_type_mismatch_with_real_registry(self, handler):
        from aragora.config.feature_flags import reset_flag_registry

        reset_flag_registry()
        try:
            h = _make_http_handler(body={"value": "not_a_bool"})
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
            assert _status(result) == 400
            assert "bool" in _body(result)["error"]
        finally:
            reset_flag_registry()


# ===========================================================================
# Tests: Edge cases
# ===========================================================================


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_flag_name_with_special_chars(self, handler, mock_http):
        """Flag name containing hyphens and dots."""
        flag = MockFlagDefinition(
            name="my-flag.v2",
            flag_type=bool,
            default=True,
            description="Special chars",
            env_var="MY_FLAG_V2",
        )
        reg = MockFlagRegistry([flag])
        reg._values["my-flag.v2"] = True
        with _patch_flags_available(reg):
            result = handler.handle(
                "/api/v1/admin/feature-flags/my-flag.v2", {}, mock_http
            )
        assert _status(result) == 200
        assert _body(result)["name"] == "my-flag.v2"

    def test_multiple_flags_same_category(self, handler, mock_http):
        """Multiple flags in the same category are all returned."""
        flags = [
            MockFlagDefinition(
                name=f"flag_{i}", flag_type=bool, default=False,
                description=f"Flag {i}", category=MockFlagCategory.CORE,
            )
            for i in range(5)
        ]
        reg = MockFlagRegistry(flags)
        with _patch_flags_available(reg):
            result = handler.handle(
                "/api/v1/admin/feature-flags",
                {"category": "core"},
                mock_http,
            )
        assert _status(result) == 200
        assert _body(result)["total"] == 5

    def test_flag_with_none_default(self, handler, mock_http):
        """Flag whose default is None."""
        flag = MockFlagDefinition(
            name="nullable_flag",
            flag_type=str,
            default=None,
            description="A nullable flag",
            env_var="NULLABLE_FLAG",
        )
        reg = MockFlagRegistry([flag])
        reg._values["nullable_flag"] = None
        with _patch_flags_available(reg):
            result = handler.handle(
                "/api/v1/admin/feature-flags/nullable_flag", {}, mock_http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["default"] is None
        assert body["value"] is None

    def test_flag_value_differs_from_default(self, handler, mock_http):
        """Flag value differs from default (e.g., env override in effect)."""
        flag = MockFlagDefinition(
            name="overridden_flag",
            flag_type=bool,
            default=False,
            description="Overridden via env",
            env_var="OVERRIDDEN",
        )
        reg = MockFlagRegistry([flag])
        reg._values["overridden_flag"] = True
        with _patch_flags_available(reg):
            result = handler.handle(
                "/api/v1/admin/feature-flags/overridden_flag", {}, mock_http
            )
        body = _body(result)
        assert body["value"] is True
        assert body["default"] is False

    def test_list_flags_sorted_by_category_and_name(self, handler, mock_http):
        """Flags should be sorted by category then name."""
        flags = [
            MockFlagDefinition(
                name="z_flag", flag_type=bool, default=False,
                description="Z", category=MockFlagCategory.CORE,
            ),
            MockFlagDefinition(
                name="a_flag", flag_type=bool, default=False,
                description="A", category=MockFlagCategory.CORE,
            ),
            MockFlagDefinition(
                name="m_flag", flag_type=bool, default=False,
                description="M", category=MockFlagCategory.EXPERIMENTAL,
            ),
        ]
        reg = MockFlagRegistry(flags)
        with _patch_flags_available(reg):
            result = handler.handle("/api/v1/admin/feature-flags", {}, mock_http)
        body = _body(result)
        names = [f["name"] for f in body["flags"]]
        # core comes before experimental alphabetically
        assert names == ["a_flag", "z_flag", "m_flag"]

    def test_set_flag_content_length_zero(self, handler, mock_registry):
        """Content-Length: 0 means read_json_body returns {} (no body)."""
        h = MagicMock()
        h.command = "PUT"
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"Content-Type": "application/json", "Content-Length": "0"}
        h.rfile.read.return_value = b""
        with _patch_flags_available(mock_registry):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/enable_checkpointing", {}, h
            )
        assert _status(result) == 400
        # read_json_body returns {} for Content-Length: 0, so "value" is missing
        assert "'value'" in _body(result)["error"]

    def test_concurrent_get_and_set_different_flags(self, handler, mock_registry):
        """Verify GET and PUT can target different flags independently."""
        mock_http = _make_http_handler()
        with _patch_flags_available(mock_registry):
            get_result = handler.handle(
                "/api/v1/admin/feature-flags/enable_knowledge_retrieval", {}, mock_http
            )
            put_h = _make_http_handler(body={"value": 10})
            put_result = handler.handle_put(
                "/api/v1/admin/feature-flags/max_agent_retries", {}, put_h
            )
        assert _status(get_result) == 200
        assert _body(get_result)["name"] == "enable_knowledge_retrieval"
        assert _status(put_result) == 200
        assert _body(put_result)["name"] == "max_agent_retries"


# ===========================================================================
# Tests: str-typed flags
# ===========================================================================


class TestStringFlags:
    """Tests for string-typed feature flags."""

    def test_set_string_flag(self, handler):
        flag = MockFlagDefinition(
            name="log_level",
            flag_type=str,
            default="INFO",
            description="Logging level",
            env_var="ARAGORA_LOG_LEVEL",
        )
        reg = MockFlagRegistry([flag])
        h = _make_http_handler(body={"value": "DEBUG"})
        env_key = "ARAGORA_LOG_LEVEL"
        os.environ.pop(env_key, None)
        try:
            with _patch_flags_available(reg):
                result = handler.handle_put(
                    "/api/v1/admin/feature-flags/log_level", {}, h
                )
            assert _status(result) == 200
            body = _body(result)
            assert body["value"] == "DEBUG"
            assert body["previous_default"] == "INFO"
            assert os.environ.get(env_key) == "DEBUG"
        finally:
            os.environ.pop(env_key, None)

    def test_set_string_flag_wrong_type(self, handler):
        flag = MockFlagDefinition(
            name="log_level",
            flag_type=str,
            default="INFO",
            description="Logging level",
        )
        reg = MockFlagRegistry([flag])
        h = _make_http_handler(body={"value": 42})
        with _patch_flags_available(reg):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/log_level", {}, h
            )
        assert _status(result) == 400
        assert "str" in _body(result)["error"]
        assert "int" in _body(result)["error"]

    def test_get_string_flag(self, handler, mock_http):
        flag = MockFlagDefinition(
            name="log_level",
            flag_type=str,
            default="INFO",
            description="Logging level",
            env_var="ARAGORA_LOG_LEVEL",
        )
        reg = MockFlagRegistry([flag])
        reg._values["log_level"] = "WARNING"
        with _patch_flags_available(reg):
            result = handler.handle(
                "/api/v1/admin/feature-flags/log_level", {}, mock_http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["type"] == "str"
        assert body["value"] == "WARNING"


# ===========================================================================
# Tests: float-typed flags
# ===========================================================================


class TestFloatFlags:
    """Tests for float-typed feature flags."""

    def test_set_float_flag(self, handler):
        flag = MockFlagDefinition(
            name="threshold",
            flag_type=float,
            default=0.5,
            description="Score threshold",
            env_var="ARAGORA_THRESHOLD",
        )
        reg = MockFlagRegistry([flag])
        h = _make_http_handler(body={"value": 0.75})
        env_key = "ARAGORA_THRESHOLD"
        os.environ.pop(env_key, None)
        try:
            with _patch_flags_available(reg):
                result = handler.handle_put(
                    "/api/v1/admin/feature-flags/threshold", {}, h
                )
            assert _status(result) == 200
            assert _body(result)["value"] == 0.75
            assert os.environ.get(env_key) == "0.75"
        finally:
            os.environ.pop(env_key, None)

    def test_set_float_flag_with_int_value(self, handler):
        """In JSON, 1 is an int, not float. This should fail type validation.
        Note: In Python, isinstance(True, int) is True but isinstance(1, float) is False.
        """
        flag = MockFlagDefinition(
            name="threshold",
            flag_type=float,
            default=0.5,
            description="Score threshold",
        )
        reg = MockFlagRegistry([flag])
        h = _make_http_handler(body={"value": 1})
        with _patch_flags_available(reg):
            result = handler.handle_put(
                "/api/v1/admin/feature-flags/threshold", {}, h
            )
        assert _status(result) == 400
        assert "float" in _body(result)["error"]
