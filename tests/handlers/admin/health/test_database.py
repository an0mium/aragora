"""Comprehensive tests for database health check handler.

Tests all public functions in
aragora/server/handlers/admin/health/database.py:

  TestHealthHandlerProtocol         - _HealthHandlerProtocol runtime check
  TestDatabaseSchemaHealth          - database_schema_health() function
  TestDatabaseSchemaHealthErrors    - Error handling for schema health
  TestDatabaseSchemaHealthEdgeCases - Edge cases for schema health
  TestDatabaseStoresHealth          - database_stores_health() function
  TestDatabaseStoresHealthSummary   - Summary field calculations
  TestDatabaseStoresHealthEdgeCases - Edge cases for stores health
  TestIntegration                   - Cross-function integration tests

45+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.database import (
    _HealthHandlerProtocol,
    database_schema_health,
    database_stores_health,
)

# The _UTILS prefix makes patching targets readable.  database_stores_health()
# does ``from .database_utils import handle_store_check_errors, ...`` *inside*
# the function body, so we must patch the symbols at their *source* module.
_UTILS = "aragora.server.handlers.admin.health.database_utils"


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


class FakeHandler:
    """Concrete implementation of _HealthHandlerProtocol for testing."""

    def __init__(
        self,
        ctx: dict[str, Any] | None = None,
        nomic_dir: Path | None = None,
    ):
        self.ctx = ctx or {}
        self._nomic_dir = nomic_dir
        self._storage = self.ctx.get("storage")
        self._elo_system = self.ctx.get("elo_system")

    def get_storage(self) -> Any:
        return self._storage

    def get_elo_system(self) -> Any:
        return self._elo_system

    def get_nomic_dir(self) -> Path | None:
        return self._nomic_dir


def _make_handler(
    ctx: dict[str, Any] | None = None,
    nomic_dir: Path | None = None,
) -> FakeHandler:
    """Create a FakeHandler with the given context and optional nomic dir."""
    return FakeHandler(ctx=ctx, nomic_dir=nomic_dir)


def _patch_validator(health_fn):
    """Return a context manager that patches the persistence.validator module."""
    return patch.dict(
        "sys.modules",
        {"aragora.persistence.validator": MagicMock(get_database_health=health_fn)},
    )


def _patch_all_stores(handle_fn):
    """Return a context manager that patches handle_store_check_errors in database_utils."""
    return patch(f"{_UTILS}.handle_store_check_errors", side_effect=handle_fn)


def _patch_handle_rv(result_tuple):
    """Return a context manager that patches handle_store_check_errors with a fixed return value."""
    return patch(f"{_UTILS}.handle_store_check_errors", return_value=result_tuple)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Default handler with empty context."""
    return _make_handler()


# ===========================================================================
# TestHealthHandlerProtocol
# ===========================================================================


class TestHealthHandlerProtocol:
    """Tests for the _HealthHandlerProtocol runtime checkable protocol."""

    def test_fake_handler_satisfies_protocol(self):
        """FakeHandler implements all required protocol methods."""
        h = _make_handler()
        assert isinstance(h, _HealthHandlerProtocol)

    def test_plain_object_does_not_satisfy_protocol(self):
        """A plain object does not satisfy the protocol."""
        assert not isinstance(object(), _HealthHandlerProtocol)

    def test_missing_get_storage_fails_protocol(self):
        """An object missing get_storage() fails the protocol check."""

        class Incomplete:
            ctx: dict[str, Any] = {}

            def get_elo_system(self) -> Any:
                return None

            def get_nomic_dir(self) -> Any:
                return None

        assert not isinstance(Incomplete(), _HealthHandlerProtocol)

    def test_missing_ctx_fails_protocol(self):
        """An object missing ctx attribute fails the protocol check."""

        class NoCtx:
            def get_storage(self) -> Any:
                return None

            def get_elo_system(self) -> Any:
                return None

            def get_nomic_dir(self) -> Any:
                return None

        obj = NoCtx()
        # Python 3.12+ runtime_checkable checks data attributes too.
        assert not isinstance(obj, _HealthHandlerProtocol)

    def test_dict_does_not_satisfy_protocol(self):
        """A plain dict does not satisfy the protocol."""
        assert not isinstance({}, _HealthHandlerProtocol)

    def test_mock_with_all_attrs_satisfies_protocol(self):
        """A MagicMock with correct spec satisfies the protocol."""
        mock = MagicMock(spec=FakeHandler)
        mock.ctx = {}
        assert isinstance(mock, _HealthHandlerProtocol)


# ===========================================================================
# TestDatabaseSchemaHealth
# ===========================================================================


class TestDatabaseSchemaHealth:
    """Tests for database_schema_health() -- happy paths."""

    def test_healthy_status_returns_200(self, handler):
        """When validator reports healthy, return 200."""
        mock_health = {"status": "healthy", "databases": {"core.db": "ok"}}
        with _patch_validator(lambda: mock_health):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "healthy"
        assert _status(result) == 200

    def test_unhealthy_status_returns_503(self, handler):
        """When validator reports unhealthy, return 503."""
        mock_health = {"status": "unhealthy", "missing_tables": ["debates"]}
        with _patch_validator(lambda: mock_health):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "unhealthy"
        assert _status(result) == 503

    def test_degraded_status_returns_503(self, handler):
        """When validator reports degraded, return 503."""
        mock_health = {"status": "degraded", "warnings": ["missing index"]}
        with _patch_validator(lambda: mock_health):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "degraded"
        assert _status(result) == 503

    def test_healthy_response_preserves_full_dict(self, handler):
        """The full health dict from the validator is returned verbatim."""
        mock_health = {
            "status": "healthy",
            "databases": {"core.db": "ok", "memory.db": "ok"},
            "table_count": 42,
        }
        with _patch_validator(lambda: mock_health):
            result = database_schema_health(handler)
        body = _body(result)
        assert body["databases"] == {"core.db": "ok", "memory.db": "ok"}
        assert body["table_count"] == 42

    def test_response_content_type_is_json(self, handler):
        """Response content type should be application/json."""
        with _patch_validator(lambda: {"status": "healthy"}):
            result = database_schema_health(handler)
        assert "json" in result.content_type

    def test_returns_handler_result_instance(self, handler):
        """Return type is HandlerResult."""
        from aragora.server.handlers.utils.responses import HandlerResult

        with _patch_validator(lambda: {"status": "healthy"}):
            result = database_schema_health(handler)
        assert isinstance(result, HandlerResult)


# ===========================================================================
# TestDatabaseSchemaHealthErrors
# ===========================================================================


class TestDatabaseSchemaHealthErrors:
    """Tests for database_schema_health() -- error branches."""

    def test_import_error_returns_503_unavailable(self, handler):
        """When persistence.validator is not importable, return 503."""
        with patch.dict("sys.modules", {"aragora.persistence.validator": None}):
            result = database_schema_health(handler)
        body = _body(result)
        assert body["status"] == "unavailable"
        assert "not available" in body["error"]
        assert _status(result) == 503

    def test_import_error_content_type_is_json(self, handler):
        """Error from ImportError still returns JSON content type."""
        with patch.dict("sys.modules", {"aragora.persistence.validator": None}):
            result = database_schema_health(handler)
        assert "json" in result.content_type

    def test_key_error_returns_500(self, handler):
        """KeyError from validator returns 500 error."""

        def bad():
            raise KeyError("status")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "error"
        assert _status(result) == 500

    def test_value_error_returns_500(self, handler):
        """ValueError from validator returns 500 error."""

        def bad():
            raise ValueError("corrupt")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "error"
        assert _status(result) == 500

    def test_os_error_returns_500(self, handler):
        """OSError from validator returns 500 error."""

        def bad():
            raise OSError("disk failure")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "error"
        assert _status(result) == 500

    def test_type_error_returns_500(self, handler):
        """TypeError from validator returns 500 error."""

        def bad():
            raise TypeError("unexpected None")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "error"
        assert _status(result) == 500

    def test_attribute_error_returns_500(self, handler):
        """AttributeError from validator returns 500 error."""

        def bad():
            raise AttributeError("no attr")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        assert _body(result)["status"] == "error"
        assert _status(result) == 500

    def test_runtime_error_returns_500(self, handler):
        """RuntimeError from validator returns 500 error."""

        def bad():
            raise RuntimeError("db locked")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        body = _body(result)
        assert body["status"] == "error"
        assert "failed" in body["error"].lower()
        assert _status(result) == 500

    def test_error_response_has_error_field(self, handler):
        """All 500 responses include an 'error' field."""

        def bad():
            raise OSError("boom")

        with _patch_validator(bad):
            result = database_schema_health(handler)
        body = _body(result)
        assert "error" in body
        assert body["error"] == "Database health check failed"


# ===========================================================================
# TestDatabaseSchemaHealthEdgeCases
# ===========================================================================


class TestDatabaseSchemaHealthEdgeCases:
    """Edge case and boundary tests for database_schema_health."""

    def test_health_dict_with_extra_fields(self, handler):
        """Extra fields in health dict are preserved."""
        mock_health = {"status": "healthy", "extra": "value", "nested": {"a": 1}}
        with _patch_validator(lambda: mock_health):
            result = database_schema_health(handler)
        body = _body(result)
        assert body["extra"] == "value"
        assert body["nested"] == {"a": 1}

    def test_empty_string_status_returns_503(self, handler):
        """Empty string status is not 'healthy', so returns 503."""
        with _patch_validator(lambda: {"status": ""}):
            result = database_schema_health(handler)
        assert _status(result) == 503

    def test_none_status_returns_503(self, handler):
        """None status is not 'healthy', so returns 503."""
        with _patch_validator(lambda: {"status": None}):
            result = database_schema_health(handler)
        assert _status(result) == 503

    def test_status_case_sensitive(self, handler):
        """'Healthy' (capitalized) is not the same as 'healthy'."""
        with _patch_validator(lambda: {"status": "Healthy"}):
            result = database_schema_health(handler)
        assert _status(result) == 503


# ===========================================================================
# TestDatabaseStoresHealth
# ===========================================================================


class TestDatabaseStoresHealth:
    """Tests for database_stores_health() -- happy paths and routing."""

    def test_all_stores_healthy_returns_healthy(self, handler):
        """When all stores are healthy, overall status is 'healthy'."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        assert _body(result)["status"] == "healthy"
        assert _status(result) == 200

    def test_one_unhealthy_returns_degraded(self, handler):
        """When one store is unhealthy, overall status is 'degraded'."""
        healthy = {"healthy": True, "status": "connected"}
        unhealthy = {"healthy": False, "error": "Connection refused"}

        def mock_handle(name, fn):
            if name == "elo_system":
                return unhealthy, False
            return healthy, True

        with _patch_all_stores(mock_handle):
            result = database_stores_health(handler)
        assert _body(result)["status"] == "degraded"

    def test_all_unhealthy_returns_degraded(self, handler):
        """When all stores are unhealthy, overall status is 'degraded'."""
        unhealthy = {"healthy": False, "error": "down"}
        with _patch_handle_rv((unhealthy, False)):
            result = database_stores_health(handler)
        assert _body(result)["status"] == "degraded"

    def test_response_includes_all_11_stores(self, handler):
        """Response contains entries for all 11 store checks."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        stores = _body(result)["stores"]
        expected = {
            "debate_storage",
            "elo_system",
            "insight_store",
            "flip_detector",
            "user_store",
            "consensus_memory",
            "agent_metadata",
            "integration_store",
            "gmail_token_store",
            "sync_store",
            "decision_result_store",
        }
        assert set(stores.keys()) == expected

    def test_store_check_order_is_deterministic(self, handler):
        """Store names are passed in the documented order."""
        captured = []

        def capture(name, fn):
            captured.append(name)
            return {"healthy": True, "status": "connected"}, True

        with _patch_all_stores(capture):
            database_stores_health(handler)

        assert captured == [
            "debate_storage",
            "elo_system",
            "insight_store",
            "flip_detector",
            "user_store",
            "consensus_memory",
            "agent_metadata",
            "integration_store",
            "gmail_token_store",
            "sync_store",
            "decision_result_store",
        ]

    def test_handle_store_check_errors_called_11_times(self, handler):
        """handle_store_check_errors is called once per store."""
        with _patch_handle_rv(({"healthy": True, "status": "connected"}, True)) as m:
            database_stores_health(handler)
        assert m.call_count == 11

    def test_stores_dict_preserves_individual_results(self, handler):
        """Each store's result dict is preserved verbatim in the response."""
        debate_result = {"healthy": True, "status": "connected", "type": "SQLiteStorage"}

        def mock_handle(name, fn):
            if name == "debate_storage":
                return debate_result, True
            return {"healthy": True, "status": "connected"}, True

        with _patch_all_stores(mock_handle):
            result = database_stores_health(handler)
        assert _body(result)["stores"]["debate_storage"] == debate_result

    def test_status_code_is_always_200(self, handler):
        """database_stores_health returns 200 even when degraded."""
        unhealthy = {"healthy": False, "error": "down"}
        with _patch_handle_rv((unhealthy, False)):
            result = database_stores_health(handler)
        assert _status(result) == 200

    def test_content_type_is_json(self, handler):
        """Response content type should be application/json."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        assert "json" in result.content_type

    def test_returns_handler_result_instance(self, handler):
        """Return type is HandlerResult."""
        from aragora.server.handlers.utils.responses import HandlerResult

        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        assert isinstance(result, HandlerResult)

    def test_elapsed_ms_present_and_nonnegative(self, handler):
        """Response includes elapsed_ms with a non-negative value."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        body = _body(result)
        assert "elapsed_ms" in body
        assert body["elapsed_ms"] >= 0

    def test_elapsed_ms_reflects_wall_clock(self, handler):
        """elapsed_ms tracks actual wall-clock time."""
        healthy = {"healthy": True, "status": "connected"}
        times = iter([100.0, 100.050])

        with _patch_handle_rv((healthy, True)), patch(
            "aragora.server.handlers.admin.health.database.time.time",
            side_effect=lambda: next(times, 100.050),
        ):
            result = database_stores_health(handler)
        assert _body(result)["elapsed_ms"] == 50.0


# ===========================================================================
# TestDatabaseStoresHealthSummary
# ===========================================================================


class TestDatabaseStoresHealthSummary:
    """Tests for the summary field in database_stores_health response."""

    def test_total_count(self, handler):
        """Summary total matches the number of stores."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        assert _body(result)["summary"]["total"] == 11

    def test_all_healthy_count(self, handler):
        """Healthy count is 11 when all stores are healthy."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        assert _body(result)["summary"]["healthy"] == 11

    def test_some_unhealthy_count(self, handler):
        """Healthy count correctly excludes unhealthy stores."""
        healthy = {"healthy": True, "status": "connected"}
        unhealthy = {"healthy": False, "error": "down"}

        def mock_handle(name, fn):
            if name in ("insight_store", "consensus_memory"):
                return unhealthy, False
            return healthy, True

        with _patch_all_stores(mock_handle):
            result = database_stores_health(handler)
        summary = _body(result)["summary"]
        assert summary["healthy"] == 9
        assert summary["total"] == 11

    def test_connected_count(self, handler):
        """Connected count tracks stores with status='connected'."""
        connected = {"healthy": True, "status": "connected"}
        not_init = {"healthy": True, "status": "not_initialized"}

        def mock_handle(name, fn):
            if name in ("flip_detector", "sync_store", "gmail_token_store"):
                return not_init, True
            return connected, True

        with _patch_all_stores(mock_handle):
            result = database_stores_health(handler)
        assert _body(result)["summary"]["connected"] == 8

    def test_not_initialized_count(self, handler):
        """not_initialized count tracks stores with that status."""
        connected = {"healthy": True, "status": "connected"}
        not_init = {"healthy": True, "status": "not_initialized"}

        def mock_handle(name, fn):
            if name in ("flip_detector", "sync_store"):
                return not_init, True
            return connected, True

        with _patch_all_stores(mock_handle):
            result = database_stores_health(handler)
        assert _body(result)["summary"]["not_initialized"] == 2

    def test_mixed_statuses(self, handler):
        """Summary is correct with a mix of connected, not_initialized, and errors."""
        results_map = {
            "debate_storage": ({"healthy": True, "status": "connected"}, True),
            "elo_system": ({"healthy": True, "status": "connected"}, True),
            "insight_store": ({"healthy": True, "status": "not_initialized"}, True),
            "flip_detector": ({"healthy": False, "error": "timeout"}, False),
            "user_store": ({"healthy": True, "status": "connected"}, True),
            "consensus_memory": ({"healthy": True, "status": "not_initialized"}, True),
            "agent_metadata": ({"healthy": True, "status": "connected"}, True),
            "integration_store": ({"healthy": True, "status": "not_initialized"}, True),
            "gmail_token_store": ({"healthy": True, "status": "not_initialized"}, True),
            "sync_store": ({"healthy": False, "error": "unavailable"}, False),
            "decision_result_store": ({"healthy": True, "status": "connected"}, True),
        }

        with _patch_all_stores(lambda name, fn: results_map[name]):
            result = database_stores_health(handler)
        body = _body(result)
        assert body["status"] == "degraded"
        assert body["summary"]["total"] == 11
        assert body["summary"]["healthy"] == 9
        assert body["summary"]["connected"] == 5
        assert body["summary"]["not_initialized"] == 4


# ===========================================================================
# TestDatabaseStoresHealthEdgeCases
# ===========================================================================


class TestDatabaseStoresHealthEdgeCases:
    """Edge case tests for database_stores_health."""

    def test_missing_healthy_key_defaults_to_false_in_summary(self, handler):
        """Store result missing 'healthy' key counts as not healthy in summary."""
        no_healthy = {"status": "connected"}
        with _patch_handle_rv((no_healthy, True)):
            result = database_stores_health(handler)
        body = _body(result)
        # .get("healthy", False) -> False when key missing
        assert body["summary"]["healthy"] == 0
        # But overall status uses is_healthy from handle_store_check_errors
        assert body["status"] == "healthy"

    def test_empty_dict_result(self, handler):
        """Store returning empty dict is handled gracefully."""
        with _patch_handle_rv(({}, True)):
            result = database_stores_health(handler)
        summary = _body(result)["summary"]
        assert summary["healthy"] == 0
        assert summary["connected"] == 0
        assert summary["not_initialized"] == 0

    def test_elapsed_ms_under_one_second(self, handler):
        """Elapsed time should be well under 1 second for mocked checks."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        assert _body(result)["elapsed_ms"] < 1000

    def test_store_result_with_extra_metadata(self, handler):
        """Extra metadata in store results is preserved."""
        rich = {
            "healthy": True,
            "status": "connected",
            "type": "PostgresStore",
            "version": "14.2",
            "connections_active": 3,
        }
        with _patch_handle_rv((rich, True)):
            result = database_stores_health(handler)
        store = _body(result)["stores"]["debate_storage"]
        assert store["type"] == "PostgresStore"
        assert store["version"] == "14.2"
        assert store["connections_active"] == 3


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    """Integration tests combining both functions."""

    def test_schema_health_result_is_valid_json(self, handler):
        """database_schema_health produces parseable JSON body."""
        with _patch_validator(lambda: {"status": "healthy"}):
            result = database_schema_health(handler)
        parsed = json.loads(result.body)
        assert isinstance(parsed, dict)

    def test_stores_health_result_is_valid_json(self, handler):
        """database_stores_health produces parseable JSON body."""
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            result = database_stores_health(handler)
        parsed = json.loads(result.body)
        assert "stores" in parsed
        assert "summary" in parsed

    def test_schema_and_stores_can_both_be_called(self, handler):
        """Both functions can be called on the same handler."""
        with _patch_validator(lambda: {"status": "healthy"}):
            schema_result = database_schema_health(handler)
        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            stores_result = database_stores_health(handler)
        assert _status(schema_result) == 200
        assert _status(stores_result) == 200

    def test_schema_error_does_not_affect_stores(self, handler):
        """Schema health failure does not impact stores health."""
        with patch.dict("sys.modules", {"aragora.persistence.validator": None}):
            schema_result = database_schema_health(handler)
        assert _status(schema_result) == 503

        healthy = {"healthy": True, "status": "connected"}
        with _patch_handle_rv((healthy, True)):
            stores_result = database_stores_health(handler)
        assert _status(stores_result) == 200
        assert _body(stores_result)["status"] == "healthy"

    def test_different_handler_contexts(self):
        """Different handler configs both produce valid results."""
        handler1 = _make_handler(ctx={"storage": MagicMock()})
        handler2 = _make_handler(ctx={})
        healthy = {"healthy": True, "status": "connected"}

        with _patch_handle_rv((healthy, True)):
            result1 = database_stores_health(handler1)
            result2 = database_stores_health(handler2)
        assert _status(result1) == 200
        assert _status(result2) == 200
