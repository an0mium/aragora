"""
Comprehensive tests for KMAdapterStatusHandler.

Tests the Knowledge Mound adapter status HTTP handler endpoint:
- GET /api/v1/knowledge/adapters - List all KM adapters with status, priority,
  circuit breaker state, and sync metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge.adapters import KMAdapterStatusHandler


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# =============================================================================
# Mock data classes
# =============================================================================


@dataclass
class MockAdapterSpec:
    """Mock for AdapterSpec from the factory."""

    name: str
    priority: int = 10
    enabled_by_default: bool = True
    required_deps: list = None
    forward_method: str = "sync_to_km"
    reverse_method: str | None = "sync_from_km"

    def __post_init__(self):
        if self.required_deps is None:
            self.required_deps = []


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{}"
        self.headers = {"Content-Length": "2"}
        self.client_address = ("127.0.0.1", 12345)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture
def handler():
    """Create a KMAdapterStatusHandler with empty context."""
    return KMAdapterStatusHandler()


@pytest.fixture
def handler_with_ctx():
    """Factory for creating a KMAdapterStatusHandler with a given context."""
    def _make(ctx: dict[str, Any] | None = None):
        return KMAdapterStatusHandler(ctx=ctx)
    return _make


def _make_specs(*entries: tuple[str, dict]) -> dict[str, MockAdapterSpec]:
    """Build a dict of MockAdapterSpec from (name, kwargs) tuples."""
    specs = {}
    for name, kwargs in entries:
        specs[name] = MockAdapterSpec(name=name, **kwargs)
    return specs


SAMPLE_SPECS = _make_specs(
    ("continuum", {"priority": 100, "required_deps": ["continuum_memory"], "forward_method": "store", "reverse_method": "sync_validations_to_continuum"}),
    ("consensus", {"priority": 90, "required_deps": ["consensus_store"], "forward_method": "sync_to_km", "reverse_method": "sync_from_km"}),
    ("elo", {"priority": 50, "required_deps": ["elo_system"], "forward_method": "sync_to_km", "reverse_method": None, "enabled_by_default": False}),
)


# =============================================================================
# can_handle tests
# =============================================================================


class TestCanHandle:
    """Tests for the can_handle routing method."""

    def test_can_handle_versioned_path(self, handler):
        """Handler accepts the versioned path /api/v1/knowledge/adapters."""
        assert handler.can_handle("/api/v1/knowledge/adapters") is True

    def test_can_handle_unversioned_path(self, handler):
        """Handler accepts the stripped path /api/knowledge/adapters."""
        assert handler.can_handle("/api/knowledge/adapters") is True

    def test_can_handle_v2_versioned_path(self, handler):
        """Handler accepts /api/v2/knowledge/adapters."""
        assert handler.can_handle("/api/v2/knowledge/adapters") is True

    def test_cannot_handle_different_path(self, handler):
        """Handler rejects unrelated paths."""
        assert handler.can_handle("/api/v1/knowledge/mound") is False

    def test_cannot_handle_sub_path(self, handler):
        """Handler rejects sub-paths of the adapters endpoint."""
        assert handler.can_handle("/api/v1/knowledge/adapters/continuum") is False

    def test_cannot_handle_root_path(self, handler):
        """Handler rejects root path."""
        assert handler.can_handle("/") is False

    def test_cannot_handle_empty_path(self, handler):
        """Handler rejects empty string."""
        assert handler.can_handle("") is False

    def test_cannot_handle_partial_match(self, handler):
        """Handler rejects paths that partially match."""
        assert handler.can_handle("/api/v1/knowledge/adapter") is False

    def test_cannot_handle_knowledge_root(self, handler):
        """Handler rejects the knowledge root path."""
        assert handler.can_handle("/api/v1/knowledge") is False


# =============================================================================
# handle() routing tests
# =============================================================================


class TestHandleRouting:
    """Tests for the handle() dispatch method."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_handle_returns_result_for_valid_path(self, handler, mock_handler):
        """handle() returns a HandlerResult for the adapters path."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert result is not None
        assert _status(result) == 200

    def test_handle_returns_none_for_unrecognized_path(self, handler, mock_handler):
        """handle() returns None when path does not match."""
        result = handler.handle("/api/v1/something/else", {}, mock_handler)
        assert result is None

    def test_handle_returns_none_for_sub_path(self, handler, mock_handler):
        """handle() returns None for sub-paths of the adapters endpoint."""
        result = handler.handle("/api/v1/knowledge/adapters/detail", {}, mock_handler)
        assert result is None

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_handle_strips_version_prefix(self, handler, mock_handler):
        """handle() correctly strips version prefix for v2 paths."""
        result = handler.handle("/api/v2/knowledge/adapters", {}, mock_handler)
        assert result is not None
        assert _status(result) == 200


# =============================================================================
# _list_adapters - KM unavailable
# =============================================================================


class TestListAdaptersKMUnavailable:
    """Tests for when Knowledge Mound system is not available."""

    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", False)
    def test_returns_503_when_km_unavailable(self, handler, mock_handler):
        """Returns 503 Service Unavailable when KM is not loaded."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", False)
    def test_503_body_structure(self, handler, mock_handler):
        """503 response has a proper error structure."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert "error" in body


# =============================================================================
# _list_adapters - basic listing
# =============================================================================


class TestListAdaptersBasic:
    """Tests for basic adapter listing without a coordinator."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_lists_all_adapters(self, handler, mock_handler):
        """Returns all registered adapters."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["total"] == 3
        assert len(body["adapters"]) == 3

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_adapters_sorted_by_name(self, handler, mock_handler):
        """Adapters are returned sorted alphabetically by name."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        names = [a["name"] for a in body["adapters"]]
        assert names == sorted(names)

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_adapter_fields_present(self, handler, mock_handler):
        """Each adapter entry includes all expected fields."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        for adapter in body["adapters"]:
            assert "name" in adapter
            assert "priority" in adapter
            assert "enabled_by_default" in adapter
            assert "required_deps" in adapter
            assert "forward_method" in adapter
            assert "reverse_method" in adapter
            assert "status" in adapter

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_adapter_priority_values(self, handler, mock_handler):
        """Adapter priorities match the spec values."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["priority"] == 100
        assert by_name["consensus"]["priority"] == 90
        assert by_name["elo"]["priority"] == 50

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_adapter_enabled_by_default(self, handler, mock_handler):
        """enabled_by_default is correctly reported."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["enabled_by_default"] is True
        assert by_name["elo"]["enabled_by_default"] is False

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_adapter_methods(self, handler, mock_handler):
        """forward_method and reverse_method are correctly reported."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["forward_method"] == "store"
        assert by_name["continuum"]["reverse_method"] == "sync_validations_to_continuum"
        assert by_name["elo"]["reverse_method"] is None

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_adapter_required_deps(self, handler, mock_handler):
        """required_deps lists are correctly reported."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["required_deps"] == ["continuum_memory"]
        assert by_name["elo"]["required_deps"] == ["elo_system"]

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_registered_status_without_coordinator(self, handler, mock_handler):
        """All adapters have status 'registered' when no coordinator is present."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        for adapter in body["adapters"]:
            assert adapter["status"] == "registered"

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_available_false_without_ctx(self, handler, mock_handler):
        """coordinator_available is False when no context is set."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["coordinator_available"] is False


# =============================================================================
# _list_adapters - empty specs
# =============================================================================


class TestListAdaptersEmpty:
    """Tests for when ADAPTER_SPECS is empty."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", {})
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_empty_adapter_list(self, handler, mock_handler):
        """Returns empty list when no adapters are registered."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["adapters"] == []
        assert body["total"] == 0

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", {})
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_available_true_with_coordinator(self, handler_with_ctx, mock_handler):
        """coordinator_available is True when coordinator exists in ctx."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {"adapters": {}}
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["coordinator_available"] is True


# =============================================================================
# _list_adapters - with coordinator (live status)
# =============================================================================


class TestListAdaptersWithCoordinator:
    """Tests for adapter listing when a coordinator provides live status."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_active_status_from_coordinator(self, handler_with_ctx, mock_handler):
        """Adapters with live status get status 'active'."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {
            "adapters": {
                "continuum": {
                    "enabled": True,
                    "has_reverse": True,
                    "forward_errors": 0,
                    "reverse_errors": 2,
                    "last_forward_sync": "2026-02-23T09:00:00Z",
                    "last_reverse_sync": "2026-02-23T08:30:00Z",
                },
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["status"] == "active"
        # Other adapters without live status remain "registered"
        assert by_name["consensus"]["status"] == "registered"
        assert by_name["elo"]["status"] == "registered"

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_enabled_field(self, handler_with_ctx, mock_handler):
        """Live 'enabled' overrides enabled_by_default."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {
            "adapters": {
                "elo": {"enabled": True},  # elo has enabled_by_default=False
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["elo"]["enabled"] is True

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_has_reverse_field(self, handler_with_ctx, mock_handler):
        """Live 'has_reverse' is reported from coordinator."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {
            "adapters": {
                "elo": {"has_reverse": False},
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["elo"]["has_reverse"] is False

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_forward_errors(self, handler_with_ctx, mock_handler):
        """Live forward_errors count is reported."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {
            "adapters": {
                "consensus": {"forward_errors": 5, "reverse_errors": 0},
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["consensus"]["forward_errors"] == 5
        assert by_name["consensus"]["reverse_errors"] == 0

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_sync_timestamps(self, handler_with_ctx, mock_handler):
        """Live last_forward_sync and last_reverse_sync are reported."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {
            "adapters": {
                "continuum": {
                    "last_forward_sync": "2026-02-23T10:00:00Z",
                    "last_reverse_sync": "2026-02-23T09:00:00Z",
                },
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["last_forward_sync"] == "2026-02-23T10:00:00Z"
        assert by_name["continuum"]["last_reverse_sync"] == "2026-02-23T09:00:00Z"

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_empty_dict_is_falsy_so_registered(self, handler_with_ctx, mock_handler):
        """Empty live dict is falsy, so adapter stays 'registered' with no extra fields."""
        coordinator = MagicMock()
        # Empty dict is falsy in Python, so `if live:` branch is skipped
        coordinator.get_status.return_value = {
            "adapters": {
                "continuum": {},
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        cont = by_name["continuum"]
        assert cont["status"] == "registered"
        # No live-specific fields should be present
        assert "enabled" not in cont
        assert "has_reverse" not in cont
        assert "forward_errors" not in cont

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_defaults_for_missing_fields(self, handler_with_ctx, mock_handler):
        """Missing fields in live status fall back to spec defaults."""
        coordinator = MagicMock()
        # Non-empty dict (truthy) but missing most fields
        coordinator.get_status.return_value = {
            "adapters": {
                "continuum": {"enabled": True},  # Minimal non-empty entry
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        cont = by_name["continuum"]
        assert cont["status"] == "active"
        assert cont["enabled"] is True
        # has_reverse falls back to spec.reverse_method is not None
        assert cont["has_reverse"] is True
        assert cont["forward_errors"] == 0
        assert cont["reverse_errors"] == 0
        assert cont["last_forward_sync"] is None
        assert cont["last_reverse_sync"] is None

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_live_nonempty_dict_means_active(self, handler_with_ctx, mock_handler):
        """A non-empty live dict gives status 'active'."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {
            "adapters": {
                "continuum": {"enabled": True},
            }
        }
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        by_name = {a["name"]: a for a in body["adapters"]}
        assert by_name["continuum"]["status"] == "active"

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_available_true(self, handler_with_ctx, mock_handler):
        """coordinator_available is True when coordinator is in ctx."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {"adapters": {}}
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["coordinator_available"] is True


# =============================================================================
# Coordinator error handling
# =============================================================================


class TestCoordinatorErrors:
    """Tests for graceful degradation when coordinator has issues."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_get_status_raises_attribute_error(self, handler_with_ctx, mock_handler):
        """Gracefully handles AttributeError from coordinator.get_status()."""
        coordinator = MagicMock()
        coordinator.get_status.side_effect = AttributeError("no status")
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        # All adapters should still be listed but as "registered"
        for adapter in body["adapters"]:
            assert adapter["status"] == "registered"

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_get_status_raises_type_error(self, handler_with_ctx, mock_handler):
        """Gracefully handles TypeError from coordinator.get_status()."""
        coordinator = MagicMock()
        coordinator.get_status.side_effect = TypeError("bad type")
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_get_status_raises_value_error(self, handler_with_ctx, mock_handler):
        """Gracefully handles ValueError from coordinator.get_status()."""
        coordinator = MagicMock()
        coordinator.get_status.side_effect = ValueError("bad value")
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_get_status_raises_runtime_error(self, handler_with_ctx, mock_handler):
        """Gracefully handles RuntimeError from coordinator.get_status()."""
        coordinator = MagicMock()
        coordinator.get_status.side_effect = RuntimeError("fail")
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["coordinator_available"] is True  # coordinator is present even though get_status failed

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_without_get_status_method(self, handler_with_ctx, mock_handler):
        """Handles a coordinator that has no get_status method."""
        coordinator = object()  # no get_status attribute
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        for adapter in body["adapters"]:
            assert adapter["status"] == "registered"

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_returns_no_adapters_key(self, handler_with_ctx, mock_handler):
        """Handles coordinator returning a status dict without 'adapters' key."""
        coordinator = MagicMock()
        coordinator.get_status.return_value = {"version": "1.0"}
        h = handler_with_ctx({"km_coordinator": coordinator})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        for adapter in body["adapters"]:
            assert adapter["status"] == "registered"


# =============================================================================
# Initialization tests
# =============================================================================


class TestInitialization:
    """Tests for handler initialization."""

    def test_default_context_is_empty_dict(self):
        """Handler defaults to empty context when none provided."""
        h = KMAdapterStatusHandler()
        assert h.ctx == {}

    def test_context_is_stored(self):
        """Handler stores the provided context."""
        ctx = {"km_coordinator": MagicMock()}
        h = KMAdapterStatusHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_context_becomes_empty_dict(self):
        """Passing ctx=None results in empty dict."""
        h = KMAdapterStatusHandler(ctx=None)
        assert h.ctx == {}

    def test_routes_constant(self):
        """ROUTES class attribute contains the expected endpoint."""
        assert "/api/v1/knowledge/adapters" in KMAdapterStatusHandler.ROUTES

    def test_routes_length(self):
        """ROUTES has exactly one entry."""
        assert len(KMAdapterStatusHandler.ROUTES) == 1


# =============================================================================
# Single adapter tests
# =============================================================================


class TestSingleAdapter:
    """Tests with a single adapter to verify structure precisely."""

    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_single_adapter_full_structure(self, handler, mock_handler):
        """Verify complete structure of a single adapter entry."""
        single_spec = _make_specs(
            ("test_adapter", {
                "priority": 42,
                "enabled_by_default": True,
                "required_deps": ["dep_a", "dep_b"],
                "forward_method": "push",
                "reverse_method": "pull",
            }),
        )
        with patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", single_spec):
            result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["total"] == 1
        adapter = body["adapters"][0]
        assert adapter == {
            "name": "test_adapter",
            "priority": 42,
            "enabled_by_default": True,
            "required_deps": ["dep_a", "dep_b"],
            "forward_method": "push",
            "reverse_method": "pull",
            "status": "registered",
        }

    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_single_adapter_no_reverse(self, handler, mock_handler):
        """Adapter with reverse_method=None is properly serialized."""
        spec = _make_specs(
            ("one_way", {
                "priority": 5,
                "reverse_method": None,
            }),
        )
        with patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", spec):
            result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["adapters"][0]["reverse_method"] is None


# =============================================================================
# Context edge cases
# =============================================================================


class TestContextEdgeCases:
    """Tests for various context configurations."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", {})
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_coordinator_is_none_in_ctx(self, mock_handler):
        """Coordinator explicitly set to None in ctx."""
        h = KMAdapterStatusHandler(ctx={"km_coordinator": None})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["coordinator_available"] is False

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", {})
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_empty_ctx_dict(self, mock_handler):
        """Empty context dict means no coordinator."""
        h = KMAdapterStatusHandler(ctx={})
        result = h.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["coordinator_available"] is False


# =============================================================================
# Response format tests
# =============================================================================


class TestResponseFormat:
    """Tests for response formatting details."""

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_response_is_json(self, handler, mock_handler):
        """Response body is valid JSON."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        # Should not throw
        data = json.loads(result.body)
        assert isinstance(data, dict)

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_response_has_content_type_json(self, handler, mock_handler):
        """Response content_type is application/json."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert "json" in result.content_type

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_response_top_level_keys(self, handler, mock_handler):
        """Response has the expected top-level keys."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert set(body.keys()) == {"adapters", "total", "coordinator_available"}

    @patch("aragora.server.handlers.knowledge.adapters.ADAPTER_SPECS", SAMPLE_SPECS)
    @patch("aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", True)
    def test_total_matches_adapter_count(self, handler, mock_handler):
        """total field always matches the length of adapters list."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = _body(result)
        assert body["total"] == len(body["adapters"])
