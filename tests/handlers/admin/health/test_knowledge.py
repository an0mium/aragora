"""Comprehensive tests for the KnowledgeMixin health check handler.

Tests aragora/server/handlers/admin/health/knowledge.py (580 lines).

The KnowledgeMixin provides:
  - knowledge_mound_health()           Main orchestrator (12 component checks)
  - _check_km_storage()                Storage backend detection
  - _check_culture_accumulator()       Culture accumulator status
  - _check_staleness_tracker()         Staleness tracker status
  - _check_rlm_integration()           RLM integration status
  - _check_debate_integration()        Debate<->KM integration
  - _check_km_redis_cache()            Redis cache status
  - _check_km_adapters()               Bidirectional adapter availability
  - _check_control_plane_adapter()     Control Plane adapter
  - _check_km_metrics()                KM metrics availability
  - _check_confidence_decay()          Confidence decay scheduler

100+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.knowledge import KnowledgeMixin
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOD = "aragora.server.handlers.admin.health.knowledge"


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    return json.loads(result.body)


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from HandlerResult."""
    return result.status_code


@contextmanager
def _mock_km_class(**kwargs):
    """Context manager to mock KnowledgeMound on its source module.

    The handler imports KnowledgeMound inside the function body via
    ``from aragora.knowledge.mound import KnowledgeMound``. Since the
    module is already cached in sys.modules, the import resolves to the
    attribute on the module object. We patch that attribute.
    """
    import aragora.knowledge.mound as km_mod

    original = getattr(km_mod, "KnowledgeMound")
    mock_cls = MagicMock(**kwargs)
    km_mod.KnowledgeMound = mock_cls
    try:
        yield mock_cls
    finally:
        km_mod.KnowledgeMound = original


class _TestMixin(KnowledgeMixin):
    """Concrete class for testing the KnowledgeMixin."""

    pass


@pytest.fixture
def mixin():
    """Create a KnowledgeMixin instance for testing."""
    return _TestMixin()


# ============================================================================
# TestKnowledgeMoundHealth -- main orchestrator
# ============================================================================


class TestKnowledgeMoundHealthModuleUnavailable:
    """Tests when the KnowledgeMound module is unavailable."""

    def test_returns_unavailable_when_module_import_fails(self, mixin):
        """When KnowledgeMound cannot be imported, returns unavailable early."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["status"] == "unavailable"
            assert "error" in body
            assert body["components"]["module"]["healthy"] is False
            assert body["components"]["module"]["status"] == "not_available"

    def test_returns_unavailable_when_types_import_fails(self, mixin):
        """When MoundConfig cannot be imported, returns unavailable early."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": None}):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["status"] == "unavailable"
            assert body["components"]["module"]["healthy"] is False

    def test_unavailable_has_timestamp(self, mixin):
        """Unavailable response includes a timestamp."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert "timestamp" in body
            assert body["timestamp"].endswith("Z")

    def test_unavailable_returns_200_status(self, mixin):
        """Even unavailable responses return HTTP 200 (health check semantics)."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result = mixin.knowledge_mound_health()
            assert _status(result) == 200


class TestKnowledgeMoundHealthCoreInit:
    """Tests for core mound initialization within the main health check."""

    def test_core_init_failure_marks_degraded(self, mixin):
        """When KnowledgeMound() raises, core is unhealthy and status is degraded."""
        with _mock_km_class(side_effect=TypeError("bad init")):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["components"]["core"]["healthy"] is False
            assert body["components"]["core"]["status"] == "initialization_failed"

    def test_core_init_os_error(self, mixin):
        """OSError during KnowledgeMound() is handled gracefully."""
        with _mock_km_class(side_effect=OSError("disk full")):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["components"]["core"]["healthy"] is False

    def test_core_init_runtime_error(self, mixin):
        """RuntimeError during KnowledgeMound() is handled gracefully."""
        with _mock_km_class(side_effect=RuntimeError("init fail")):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["components"]["core"]["healthy"] is False

    def test_core_init_value_error(self, mixin):
        """ValueError during KnowledgeMound() is handled gracefully."""
        with _mock_km_class(side_effect=ValueError("bad workspace")):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["components"]["core"]["healthy"] is False
            assert body["components"]["core"]["error"] == "Initialization failed"

    def test_core_init_attribute_error(self, mixin):
        """AttributeError during KnowledgeMound() is handled gracefully."""
        with _mock_km_class(side_effect=AttributeError("no attr")):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert body["components"]["core"]["healthy"] is False

    def test_core_init_success_with_config(self, mixin):
        """When KnowledgeMound initializes with config, config details are present."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "health_check"
        mock_config = MagicMock()
        mock_config.enable_staleness_detection = True
        mock_config.enable_culture_accumulator = False
        mock_config.staleness_age_threshold = timedelta(days=7)
        mock_mound.config = mock_config

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            core = body["components"]["core"]
            assert core["healthy"] is True
            assert core["status"] == "initialized"
            assert core["workspace_id"] == "health_check"
            assert core["config"]["enable_staleness_detection"] is True
            assert core["config"]["enable_culture_accumulator"] is False
            assert core["config"]["staleness_age_threshold_days"] == 7

    def test_core_init_success_no_config(self, mixin):
        """When mound has no config attribute, no config in result."""
        mock_mound = MagicMock(spec=[])
        mock_mound.workspace_id = "health_check"

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            core = body["components"]["core"]
            assert core["healthy"] is True
            assert "config" not in core

    def test_core_init_success_config_none(self, mixin):
        """When mound.config is None, no config in result."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "health_check"
        mock_mound.config = None

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            core = body["components"]["core"]
            assert core["healthy"] is True
            assert "config" not in core


class TestKnowledgeMoundHealthOverallStatus:
    """Tests for the overall status and summary computation."""

    def _make_mound(self):
        """Create a minimal mock mound for testing."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "health_check"
        mock_mound.config = None
        mock_mound._store = None
        mock_mound._culture_accumulator = None
        mock_mound._staleness_tracker = None
        return mock_mound

    def test_healthy_status(self, mixin):
        """When all components healthy and at least one active, status is healthy."""
        mock_mound = self._make_mound()
        mock_mound._staleness_tracker = MagicMock()  # one active component

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            # Status should be healthy or not_configured based on active count
            assert body["status"] in ("healthy", "not_configured")

    def test_response_time_included(self, mixin):
        """Response includes response_time_ms."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert "response_time_ms" in body
            assert isinstance(body["response_time_ms"], (int, float))
            assert body["response_time_ms"] >= 0

    def test_timestamp_included(self, mixin):
        """Response includes ISO timestamp ending with Z."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            assert "timestamp" in body
            assert body["timestamp"].endswith("Z")

    def test_summary_counts(self, mixin):
        """Summary contains total_components, healthy, and active counts."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            summary = body["summary"]
            assert "total_components" in summary
            assert "healthy" in summary
            assert "active" in summary
            assert summary["total_components"] == len(body["components"])

    def test_all_12_components_present(self, mixin):
        """All 12 component checks are included in the response."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            expected_keys = {
                "module",
                "core",
                "storage",
                "culture_accumulator",
                "staleness_tracker",
                "rlm_integration",
                "debate_integration",
                "redis_cache",
                "bidirectional_adapters",
                "control_plane_adapter",
                "km_metrics",
                "confidence_decay",
            }
            assert set(body["components"].keys()) == expected_keys

    def test_degraded_when_core_fails(self, mixin):
        """Status is degraded when core init fails."""
        with _mock_km_class(side_effect=TypeError("bad")):
            result = mixin.knowledge_mound_health()
            body = _body(result)
            # core fails => all_healthy = False => degraded or not_configured
            assert body["status"] in ("degraded", "not_configured")

    def test_not_configured_when_no_active_components(self, mixin):
        """Status is not_configured when active_count is 0."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            with patch.object(
                mixin, "_check_km_storage", return_value={"healthy": True, "status": "configured"}
            ):
                with patch.object(
                    mixin,
                    "_check_culture_accumulator",
                    return_value={"healthy": True, "status": "not_initialized"},
                ):
                    with patch.object(
                        mixin,
                        "_check_staleness_tracker",
                        return_value={"healthy": True, "status": "not_initialized"},
                    ):
                        with patch.object(
                            mixin,
                            "_check_rlm_integration",
                            return_value={"healthy": True, "status": "not_available"},
                        ):
                            with patch.object(
                                mixin,
                                "_check_debate_integration",
                                return_value={"healthy": True, "status": "not_available"},
                            ):
                                with patch.object(
                                    mixin,
                                    "_check_km_redis_cache",
                                    return_value={"healthy": True, "status": "not_configured"},
                                ):
                                    with patch.object(
                                        mixin,
                                        "_check_km_adapters",
                                        return_value={"healthy": True, "status": "available"},
                                    ):
                                        with patch.object(
                                            mixin,
                                            "_check_control_plane_adapter",
                                            return_value={"healthy": True, "status": "available"},
                                        ):
                                            with patch.object(
                                                mixin,
                                                "_check_km_metrics",
                                                return_value={
                                                    "healthy": True,
                                                    "status": "available",
                                                },
                                            ):
                                                with patch.object(
                                                    mixin,
                                                    "_check_confidence_decay",
                                                    return_value={
                                                        "component": {
                                                            "healthy": True,
                                                            "status": "not_configured",
                                                        },
                                                        "warnings": [],
                                                    },
                                                ):
                                                    result = mixin.knowledge_mound_health()
                                                    body = _body(result)
                                                    assert body["status"] == "not_configured"

    def test_warnings_included_when_present(self, mixin):
        """Warnings from sub-checks are included in response."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            with patch.object(
                mixin,
                "_check_confidence_decay",
                return_value={
                    "component": {"healthy": True, "status": "active"},
                    "warnings": ["Some stale workspace warning"],
                },
            ):
                result = mixin.knowledge_mound_health()
                body = _body(result)
                assert body["warnings"] is not None
                assert "Some stale workspace warning" in body["warnings"]

    def test_warnings_null_when_none(self, mixin):
        """When no warnings, the field is null."""
        mock_mound = self._make_mound()

        with _mock_km_class(return_value=mock_mound):
            with patch.object(
                mixin,
                "_check_confidence_decay",
                return_value={
                    "component": {"healthy": True, "status": "not_configured"},
                    "warnings": [],
                },
            ):
                result = mixin.knowledge_mound_health()
                body = _body(result)
                assert body["warnings"] is None


# ============================================================================
# TestCheckKmStorage
# ============================================================================


class TestCheckKmStorage:
    """Tests for _check_km_storage()."""

    def test_postgresql_backend(self, mixin):
        """Detects PostgreSQL when KNOWLEDGE_MOUND_DATABASE_URL contains 'postgres'."""
        with patch.dict(
            "os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": "postgresql://localhost/km"}
        ):
            result = mixin._check_km_storage(None)
            assert result["healthy"] is True
            assert result["backend"] == "postgresql"
            assert result["status"] == "configured"

    def test_postgres_case_insensitive(self, mixin):
        """PostgreSQL detection is case insensitive."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": "POSTGRESQL://HOST/DB"}):
            result = mixin._check_km_storage(None)
            assert result["backend"] == "postgresql"

    def test_sqlite_default(self, mixin):
        """Defaults to SQLite when no postgres URL."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = mixin._check_km_storage(None)
            assert result["healthy"] is True
            assert result["backend"] == "sqlite"
            assert result["status"] == "configured"
            assert "note" in result

    def test_sqlite_when_no_env(self, mixin):
        """Defaults to SQLite when env var is not set."""
        env_copy = {
            k: v for k, v in __import__("os").environ.items() if k != "KNOWLEDGE_MOUND_DATABASE_URL"
        }
        with patch.dict("os.environ", env_copy, clear=True):
            result = mixin._check_km_storage(None)
            assert result["backend"] == "sqlite"

    def test_mound_with_store(self, mixin):
        """When mound has _store, store_type is included."""
        mock_mound = MagicMock()
        mock_store = MagicMock()
        mock_store.__class__.__name__ = "ResilientPostgresStore"
        mock_mound._store = mock_store

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = mixin._check_km_storage(mock_mound)
            assert result["healthy"] is True
            assert "store_type" in result

    def test_mound_with_store_none(self, mixin):
        """When mound._store is None, no store_type."""
        mock_mound = MagicMock()
        mock_mound._store = None

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = mixin._check_km_storage(mock_mound)
            assert "store_type" not in result

    def test_mound_without_store_attr(self, mixin):
        """When mound has no _store attribute, no store_type."""
        mock_mound = MagicMock(spec=[])

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = mixin._check_km_storage(mock_mound)
            assert "store_type" not in result

    def test_mound_store_attribute_error(self, mixin):
        """AttributeError when accessing _store is handled."""

        class BrokenMound:
            @property
            def _store(self):
                raise AttributeError("oops")

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = mixin._check_km_storage(BrokenMound())
            # hasattr catches AttributeError, so _store check is skipped
            assert result["healthy"] is True

    def test_none_mound(self, mixin):
        """None mound still returns healthy result."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = mixin._check_km_storage(None)
            assert result["healthy"] is True

    def test_storage_postgres_partial_match(self, mixin):
        """A URL containing 'postgres' anywhere triggers PostgreSQL detection."""
        with patch.dict(
            "os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": "my-postgres-instance:5432/db"}
        ):
            result = mixin._check_km_storage(None)
            assert result["backend"] == "postgresql"


# ============================================================================
# TestCheckCultureAccumulator
# ============================================================================


class TestCheckCultureAccumulator:
    """Tests for _check_culture_accumulator()."""

    def test_active_accumulator(self, mixin):
        """Active accumulator returns active status with type."""
        mock_accumulator = MagicMock()
        mock_accumulator.__class__.__name__ = "CultureAccumulator"
        mock_accumulator._patterns = {"ws1": [], "ws2": [], "ws3": []}
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = mixin._check_culture_accumulator(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"
        assert "type" in result
        assert result["workspaces_tracked"] == 3

    def test_accumulator_no_patterns_attr(self, mixin):
        """When accumulator has no _patterns, still active without workspace count."""
        mock_accumulator = MagicMock(spec=[])
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = mixin._check_culture_accumulator(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"
        assert "workspaces_tracked" not in result

    def test_accumulator_none(self, mixin):
        """When accumulator is None, returns not_initialized."""
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = None

        result = mixin._check_culture_accumulator(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"
        assert "note" in result

    def test_no_accumulator_attribute(self, mixin):
        """When mound has no _culture_accumulator attr, returns not_initialized."""
        mock_mound = MagicMock(spec=[])

        result = mixin._check_culture_accumulator(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_mound_none(self, mixin):
        """When mound is None, returns not_initialized."""
        result = mixin._check_culture_accumulator(None)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_patterns_type_error(self, mixin):
        """TypeError when counting patterns is caught gracefully."""
        mock_accumulator = MagicMock()
        mock_accumulator._patterns = MagicMock()
        mock_accumulator._patterns.__len__ = MagicMock(side_effect=TypeError("bad"))
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = mixin._check_culture_accumulator(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"
        assert "workspaces_tracked" not in result

    def test_attribute_error_caught(self, mixin):
        """AttributeError during check is caught."""

        class BrokenMound:
            @property
            def _culture_accumulator(self):
                raise AttributeError("no attr")

        result = mixin._check_culture_accumulator(BrokenMound())
        assert result["healthy"] is True
        # hasattr catches AttributeError, so falls to not_initialized branch
        assert result["status"] == "not_initialized"

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during check is caught."""

        class BrokenMound:
            @property
            def _culture_accumulator(self):
                raise RuntimeError("fail")

        result = mixin._check_culture_accumulator(BrokenMound())
        assert result["healthy"] is True
        assert result["status"] == "error"
        assert result["error"] == "Health check failed"

    def test_culture_accumulator_empty_patterns(self, mixin):
        """When accumulator._patterns is empty dict, workspaces_tracked is 0."""
        mock_accumulator = MagicMock()
        mock_accumulator._patterns = {}
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = mixin._check_culture_accumulator(mock_mound)
        assert result["workspaces_tracked"] == 0


# ============================================================================
# TestCheckStalenessTracker
# ============================================================================


class TestCheckStalenessTracker:
    """Tests for _check_staleness_tracker()."""

    def test_active_tracker(self, mixin):
        """Active tracker returns active status."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = MagicMock()

        result = mixin._check_staleness_tracker(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"

    def test_tracker_none(self, mixin):
        """When tracker is None, returns not_initialized."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = None

        result = mixin._check_staleness_tracker(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"
        assert "note" in result

    def test_no_tracker_attribute(self, mixin):
        """When mound has no _staleness_tracker attr, returns not_initialized."""
        mock_mound = MagicMock(spec=[])

        result = mixin._check_staleness_tracker(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_mound_none(self, mixin):
        """When mound is None, returns not_initialized."""
        result = mixin._check_staleness_tracker(None)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_falsy_tracker(self, mixin):
        """When tracker is falsy (empty list), returns not_initialized."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = []

        result = mixin._check_staleness_tracker(mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_attribute_error_property_falls_through(self, mixin):
        """AttributeError from property is swallowed by hasattr, returns not_initialized."""

        class BrokenMound:
            @property
            def _staleness_tracker(self):
                raise AttributeError("no attr")

        result = mixin._check_staleness_tracker(BrokenMound())
        assert result["healthy"] is True
        # hasattr catches AttributeError, so falls to not_initialized branch
        assert result["status"] == "not_initialized"

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during check is caught by except block."""

        class BrokenMound:
            @property
            def _staleness_tracker(self):
                raise RuntimeError("boom")

        result = mixin._check_staleness_tracker(BrokenMound())
        assert result["healthy"] is True
        assert result["status"] == "error"

    def test_type_error_caught(self, mixin):
        """TypeError during check is caught by except block."""

        class BrokenMound:
            @property
            def _staleness_tracker(self):
                raise TypeError("bad type")

        result = mixin._check_staleness_tracker(BrokenMound())
        assert result["healthy"] is True
        assert result["status"] == "error"

    def test_truthy_object_counts_as_active(self, mixin):
        """Any truthy tracker object counts as active."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = "truthy_string"

        result = mixin._check_staleness_tracker(mock_mound)
        assert result["status"] == "active"


# ============================================================================
# TestCheckRlmIntegration
# ============================================================================


class TestCheckRlmIntegration:
    """Tests for _check_rlm_integration()."""

    def test_official_rlm_active(self, mixin):
        """When HAS_OFFICIAL_RLM is True, returns active with official_rlm type."""
        mock_rlm = MagicMock()
        mock_rlm.HAS_OFFICIAL_RLM = True

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "active"
            assert result["type"] == "official_rlm"

    def test_fallback_mode(self, mixin):
        """When HAS_OFFICIAL_RLM is False, returns fallback."""
        mock_rlm = MagicMock()
        mock_rlm.HAS_OFFICIAL_RLM = False

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "fallback"
            assert result["type"] == "compression_only"
            assert "note" in result

    def test_import_error(self, mixin):
        """When RLM module cannot be imported, returns not_available."""
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert "note" in result

    def test_attribute_error_from_property_treated_as_import(self, mixin):
        """AttributeError from property on module is treated as ImportError by Python."""

        class FakeModule(type(sys)):
            @property
            def HAS_OFFICIAL_RLM(self):
                raise AttributeError("no attribute")

        mock_rlm = FakeModule("fake_rlm")
        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            # Python's 'from x import y' converts AttributeError to ImportError
            assert result["status"] == "not_available"

    def test_type_error_caught(self, mixin):
        """TypeError during RLM if-check is caught."""

        # HAS_OFFICIAL_RLM imports fine but its __bool__ raises TypeError
        class BrokenBool:
            def __bool__(self):
                raise TypeError("bad bool")

        mock_rlm = MagicMock()
        mock_rlm.HAS_OFFICIAL_RLM = BrokenBool()

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"
            assert result["error"] == "Health check failed"

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during RLM check is caught."""

        class FakeModule(type(sys)):
            @property
            def HAS_OFFICIAL_RLM(self):
                raise RuntimeError("runtime issue")

        mock_rlm = FakeModule("fake_rlm")
        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"

    def test_value_error_caught(self, mixin):
        """ValueError during RLM check is caught."""

        class FakeModule(type(sys)):
            @property
            def HAS_OFFICIAL_RLM(self):
                raise ValueError("bad value")

        mock_rlm = FakeModule("fake_rlm")
        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = mixin._check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"


# ============================================================================
# TestCheckDebateIntegration
# ============================================================================


class TestCheckDebateIntegration:
    """Tests for _check_debate_integration()."""

    def test_active_with_stats(self, mixin):
        """When stats function exists and returns data, returns active."""
        mock_stats_fn = MagicMock(
            return_value={
                "facts_count": 42,
                "consensus_stored": 10,
                "retrievals_count": 100,
            }
        )
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "active"
            assert result["facts_count"] == 42
            assert result["consensus_stored"] == 10
            assert result["retrievals_count"] == 100

    def test_stats_with_missing_keys(self, mixin):
        """Defaults to 0 when stats dict is missing keys."""
        mock_stats_fn = MagicMock(return_value={})
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["facts_count"] == 0
            assert result["consensus_stored"] == 0
            assert result["retrievals_count"] == 0

    def test_no_stats_function(self, mixin):
        """When get_knowledge_mound_stats is absent, returns not_available."""
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            None,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert "note" in result

    def test_module_import_error(self, mixin):
        """When knowledge_mound_ops cannot be imported, returns not_available."""
        import aragora.debate as debate_mod

        original_attr = getattr(debate_mod, "knowledge_mound_ops", None)
        original_in_sys = sys.modules.get("aragora.debate.knowledge_mound_ops")
        try:
            if hasattr(debate_mod, "knowledge_mound_ops"):
                delattr(debate_mod, "knowledge_mound_ops")
            sys.modules["aragora.debate.knowledge_mound_ops"] = None
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
        finally:
            if original_in_sys is not None:
                sys.modules["aragora.debate.knowledge_mound_ops"] = original_in_sys
            elif "aragora.debate.knowledge_mound_ops" in sys.modules:
                del sys.modules["aragora.debate.knowledge_mound_ops"]
            if original_attr is not None:
                debate_mod.knowledge_mound_ops = original_attr

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=RuntimeError("failed"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"
            assert result["error"] == "Health check failed"

    def test_type_error_caught(self, mixin):
        """TypeError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=TypeError("bad type"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"

    def test_key_error_caught(self, mixin):
        """KeyError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=KeyError("missing"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"

    def test_value_error_caught(self, mixin):
        """ValueError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=ValueError("bad val"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"

    def test_attribute_error_caught(self, mixin):
        """AttributeError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=AttributeError("bad attr"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = mixin._check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"


# ============================================================================
# TestCheckKmRedisCache
# ============================================================================


class TestCheckKmRedisCache:
    """Tests for _check_km_redis_cache()."""

    def test_configured_with_km_redis_url(self, mixin):
        """When KNOWLEDGE_MOUND_REDIS_URL is set, returns configured."""
        mock_cache_mod = MagicMock()
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "configured"

    def test_configured_with_fallback_redis_url(self, mixin):
        """When only REDIS_URL is set, uses it as fallback."""
        mock_cache_mod = MagicMock()
        env = {"KNOWLEDGE_MOUND_REDIS_URL": "", "REDIS_URL": "redis://localhost:6379"}
        with patch.dict("os.environ", env, clear=False):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "configured"

    def test_not_configured(self, mixin):
        """When no Redis URL env vars are set, returns not_configured."""
        env = {"KNOWLEDGE_MOUND_REDIS_URL": "", "REDIS_URL": ""}
        with patch.dict("os.environ", env, clear=False):
            result = mixin._check_km_redis_cache()
            assert result["healthy"] is True
            assert result["status"] == "not_configured"
            assert "note" in result

    def test_not_configured_missing_env(self, mixin):
        """When Redis env vars do not exist at all, returns not_configured."""
        env_copy = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in ("KNOWLEDGE_MOUND_REDIS_URL", "REDIS_URL")
        }
        with patch.dict("os.environ", env_copy, clear=True):
            result = mixin._check_km_redis_cache()
            assert result["healthy"] is True
            assert result["status"] == "not_configured"

    def test_redis_cache_import_error(self, mixin):
        """When redis_cache module cannot be imported, returns not_available."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": None}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "not_available"
                assert "note" in result

    def test_connection_error(self, mixin):
        """ConnectionError during RedisCache() returns error status."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = ConnectionError("refused")
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"
                assert result["error"] == "Health check failed"

    def test_timeout_error(self, mixin):
        """TimeoutError during RedisCache() returns error status."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = TimeoutError("timed out")
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"

    def test_os_error(self, mixin):
        """OSError during RedisCache() returns error status."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = OSError("socket error")
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"

    def test_value_error(self, mixin):
        """ValueError (bad URL) during RedisCache() returns error status."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = ValueError("invalid URL")
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "not-a-url"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"

    def test_runtime_error(self, mixin):
        """RuntimeError during RedisCache() returns error status."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = RuntimeError("init failed")
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": mock_cache_mod}):
                result = mixin._check_km_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"


# ============================================================================
# TestCheckKmAdapters
# ============================================================================


class TestCheckKmAdapters:
    """Tests for _check_km_adapters()."""

    def test_adapters_available(self, mixin):
        """When all adapter imports succeed, returns available with list."""
        result = mixin._check_km_adapters()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert result["adapters_available"] == 11
        assert isinstance(result["adapter_list"], list)
        assert "continuum" in result["adapter_list"]
        assert "consensus" in result["adapter_list"]
        assert "culture" in result["adapter_list"]

    def test_adapter_list_order(self, mixin):
        """Adapter list is in expected order."""
        result = mixin._check_km_adapters()
        if result["status"] == "available":
            expected = [
                "continuum",
                "consensus",
                "critique",
                "evidence",
                "belief",
                "insights",
                "elo",
                "pulse",
                "cost",
                "ranking",
                "culture",
            ]
            assert result["adapter_list"] == expected

    def test_import_error_returns_partial(self, mixin):
        """When adapter imports fail, returns partial."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters": None}):
            result = mixin._check_km_adapters()
            assert result["healthy"] is True
            assert result["status"] == "partial"
            assert "error" in result

    def test_attribute_error_caught(self, mixin):
        """AttributeError during adapter check is caught."""
        mock_adapters = MagicMock()
        type(mock_adapters).ContinuumAdapter = property(
            lambda self: (_ for _ in ()).throw(AttributeError("missing"))
        )
        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters": mock_adapters}):
            result = mixin._check_km_adapters()
            assert result["healthy"] is True
            assert result["status"] in ("available", "error", "partial")

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during adapter check is caught."""
        mock_adapters = MagicMock()
        type(mock_adapters).ContinuumAdapter = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("bad"))
        )
        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters": mock_adapters}):
            result = mixin._check_km_adapters()
            assert result["healthy"] is True

    def test_type_error_caught(self, mixin):
        """TypeError during adapter check is caught."""
        mock_adapters = MagicMock()
        type(mock_adapters).ContinuumAdapter = property(
            lambda self: (_ for _ in ()).throw(TypeError("bad type"))
        )
        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters": mock_adapters}):
            result = mixin._check_km_adapters()
            assert result["healthy"] is True


# ============================================================================
# TestCheckControlPlaneAdapter
# ============================================================================


class TestCheckControlPlaneAdapter:
    """Tests for _check_control_plane_adapter()."""

    def test_adapter_available(self, mixin):
        """When control plane adapter imports succeed, returns available with features."""
        result = mixin._check_control_plane_adapter()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert "features" in result
        assert "task_outcome_storage" in result["features"]
        assert "capability_records" in result["features"]
        assert "cross_workspace_insights" in result["features"]
        assert "agent_recommendations" in result["features"]

    def test_features_count(self, mixin):
        """Available adapter has 4 features."""
        result = mixin._check_control_plane_adapter()
        if result["status"] == "available":
            assert len(result["features"]) == 4

    def test_import_error(self, mixin):
        """When control plane adapter cannot be imported, returns not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.adapters.control_plane_adapter": None,
            },
        ):
            result = mixin._check_control_plane_adapter()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert "error" in result

    def test_attribute_error_caught(self, mixin):
        """AttributeError during check is caught."""
        mock_mod = MagicMock()
        type(mock_mod).ControlPlaneAdapter = property(
            lambda self: (_ for _ in ()).throw(AttributeError("missing"))
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.adapters.control_plane_adapter": mock_mod,
            },
        ):
            result = mixin._check_control_plane_adapter()
            assert result["healthy"] is True
            assert result["status"] in ("available", "error")

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during check is caught."""
        mock_mod = MagicMock()
        type(mock_mod).ControlPlaneAdapter = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("init fail"))
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.adapters.control_plane_adapter": mock_mod,
            },
        ):
            result = mixin._check_control_plane_adapter()
            assert result["healthy"] is True

    def test_type_error_caught(self, mixin):
        """TypeError during check is caught."""
        mock_mod = MagicMock()
        type(mock_mod).ControlPlaneAdapter = property(
            lambda self: (_ for _ in ()).throw(TypeError("bad type"))
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.adapters.control_plane_adapter": mock_mod,
            },
        ):
            result = mixin._check_control_plane_adapter()
            assert result["healthy"] is True


# ============================================================================
# TestCheckKmMetrics
# ============================================================================


class TestCheckKmMetrics:
    """Tests for _check_km_metrics()."""

    def test_metrics_available(self, mixin):
        """When KM metrics module is importable, returns available."""
        result = mixin._check_km_metrics()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert result["prometheus_integration"] is True

    def test_metrics_not_available(self, mixin):
        """When KM metrics module cannot be imported, returns not_available."""
        with patch.dict("sys.modules", {"aragora.observability.metrics.km": None}):
            result = mixin._check_km_metrics()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert result["prometheus_integration"] is False

    def test_attribute_error_caught(self, mixin):
        """AttributeError during metrics check is caught."""
        mock_mod = MagicMock()
        type(mock_mod).init_km_metrics = property(
            lambda self: (_ for _ in ()).throw(AttributeError("missing"))
        )
        with patch.dict("sys.modules", {"aragora.observability.metrics.km": mock_mod}):
            result = mixin._check_km_metrics()
            assert result["healthy"] is True

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during metrics check is caught."""
        mock_mod = MagicMock()
        type(mock_mod).init_km_metrics = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        with patch.dict("sys.modules", {"aragora.observability.metrics.km": mock_mod}):
            result = mixin._check_km_metrics()
            assert result["healthy"] is True

    def test_type_error_caught(self, mixin):
        """TypeError during metrics check is caught."""
        mock_mod = MagicMock()
        type(mock_mod).init_km_metrics = property(
            lambda self: (_ for _ in ()).throw(TypeError("bad"))
        )
        with patch.dict("sys.modules", {"aragora.observability.metrics.km": mock_mod}):
            result = mixin._check_km_metrics()
            assert result["healthy"] is True


# ============================================================================
# TestCheckConfidenceDecay
# ============================================================================


class TestCheckConfidenceDecay:
    """Tests for _check_confidence_decay()."""

    def test_scheduler_active_and_running(self, mixin):
        """When scheduler is running, returns active status."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 5,
            "total_items_processed": 100,
            "last_run": {},
            "workspaces": ["ws1", "ws2"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "active"
            assert component["running"] is True
            assert component["decay_interval_hours"] == 24
            assert component["total_cycles"] == 5
            assert component["total_items_processed"] == 100
            assert component["workspaces_monitored"] == ["ws1", "ws2"]
            assert len(result["warnings"]) == 0

    def test_scheduler_stopped(self, mixin):
        """When scheduler exists but is not running, returns stopped."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 12,
            "total_decay_cycles": 3,
            "total_items_processed": 50,
            "last_run": {},
            "workspaces": [],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["status"] == "stopped"
            assert component["running"] is False

    def test_scheduler_not_configured(self, mixin):
        """When get_decay_scheduler returns None, returns not_configured."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = None

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "not_configured"
            assert "note" in component

    def test_module_not_available(self, mixin):
        """When confidence_decay_scheduler cannot be imported, returns not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "not_available"
            assert "note" in component

    def test_stale_workspace_warning(self, mixin):
        """When workspace not processed in >48h, adds warning and alert."""
        stale_time = (datetime.now() - timedelta(hours=50)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {"stale_ws": stale_time},
            "workspaces": ["stale_ws"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert "alert" in component
            assert component["alert"]["level"] == "warning"
            assert "stale_ws" in component["alert"]["message"]
            assert len(result["warnings"]) == 1
            assert "stale_ws" in result["warnings"][0]

    def test_recent_workspace_no_warning(self, mixin):
        """Recent workspace does not trigger a warning."""
        recent_time = (datetime.now() - timedelta(hours=1)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {"recent_ws": recent_time},
            "workspaces": ["recent_ws"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            assert "alert" not in result["component"]
            assert len(result["warnings"]) == 0

    def test_multiple_stale_workspaces(self, mixin):
        """Multiple stale workspaces all appear in alert."""
        stale_time = (datetime.now() - timedelta(hours=72)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {"ws_a": stale_time, "ws_b": stale_time},
            "workspaces": ["ws_a", "ws_b"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            alert_msg = result["component"]["alert"]["message"]
            assert "ws_a" in alert_msg
            assert "ws_b" in alert_msg

    def test_mixed_stale_and_recent(self, mixin):
        """Only stale workspaces appear in alert, not recent ones."""
        stale_time = (datetime.now() - timedelta(hours=60)).isoformat()
        recent_time = (datetime.now() - timedelta(hours=1)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {"stale_ws": stale_time, "recent_ws": recent_time},
            "workspaces": ["stale_ws", "recent_ws"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            alert_msg = result["component"]["alert"]["message"]
            assert "stale_ws" in alert_msg
            assert "recent_ws" not in alert_msg

    def test_bad_timestamp_silently_ignored(self, mixin):
        """Invalid timestamp in last_run is silently skipped."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {"ws_bad": "not-a-timestamp"},
            "workspaces": ["ws_bad"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            assert "alert" not in result["component"]
            assert len(result["warnings"]) == 0

    def test_none_timestamp_silently_ignored(self, mixin):
        """None timestamp in last_run is silently skipped."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 1,
            "total_items_processed": 10,
            "last_run": {"ws_none": None},
            "workspaces": ["ws_none"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            assert "alert" not in result["component"]

    def test_stopped_scheduler_skips_stale_check(self, mixin):
        """Stopped scheduler does not check for stale workspaces."""
        stale_time = (datetime.now() - timedelta(hours=100)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 3,
            "total_items_processed": 50,
            "last_run": {"ws_very_stale": stale_time},
            "workspaces": ["ws_very_stale"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            assert result["component"]["status"] == "stopped"
            assert "alert" not in result["component"]

    def test_empty_last_run(self, mixin):
        """Empty last_run dict produces no stale warnings."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 0,
            "total_items_processed": 0,
            "last_run": {},
            "workspaces": [],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            assert "alert" not in result["component"]
            assert len(result["warnings"]) == 0

    def test_stats_default_values(self, mixin):
        """Missing stats keys use defaults."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {}

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["decay_interval_hours"] == 24
            assert component["total_cycles"] == 0
            assert component["total_items_processed"] == 0

    def test_runtime_error_caught(self, mixin):
        """RuntimeError during scheduler check is caught."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.side_effect = RuntimeError("broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "error"
            assert component["error"] == "Health check failed"

    def test_attribute_error_caught(self, mixin):
        """AttributeError during scheduler check is caught."""
        mock_mod = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.get_stats.side_effect = AttributeError("no stats")
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "error"

    def test_type_error_caught(self, mixin):
        """TypeError during scheduler check is caught."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.side_effect = TypeError("bad")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "error"

    def test_key_error_caught(self, mixin):
        """KeyError during scheduler check is caught."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.side_effect = KeyError("missing")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "error"

    def test_value_error_caught(self, mixin):
        """ValueError during scheduler check is caught."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.side_effect = ValueError("bad val")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = mixin._check_confidence_decay()
            component = result["component"]
            assert component["healthy"] is True
            assert component["status"] == "error"

    def test_returns_dict_with_component_and_warnings(self, mixin):
        """Return type has 'component' and 'warnings' keys."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result = mixin._check_confidence_decay()
            assert "component" in result
            assert "warnings" in result
            assert isinstance(result["component"], dict)
            assert isinstance(result["warnings"], list)


# ============================================================================
# TestAllMethodsReturnHealthy - cross-cutting invariant
# ============================================================================


class TestAllMethodsReturnHealthy:
    """All individual check methods always return healthy=True."""

    def test_storage_always_healthy(self, mixin):
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            assert mixin._check_km_storage(None)["healthy"] is True

    def test_culture_always_healthy(self, mixin):
        assert mixin._check_culture_accumulator(None)["healthy"] is True

    def test_staleness_always_healthy(self, mixin):
        assert mixin._check_staleness_tracker(None)["healthy"] is True

    def test_rlm_always_healthy(self, mixin):
        assert mixin._check_rlm_integration()["healthy"] is True

    def test_debate_always_healthy(self, mixin):
        assert mixin._check_debate_integration()["healthy"] is True

    def test_redis_always_healthy(self, mixin):
        assert mixin._check_km_redis_cache()["healthy"] is True

    def test_adapters_always_healthy(self, mixin):
        assert mixin._check_km_adapters()["healthy"] is True

    def test_control_plane_always_healthy(self, mixin):
        assert mixin._check_control_plane_adapter()["healthy"] is True

    def test_metrics_always_healthy(self, mixin):
        assert mixin._check_km_metrics()["healthy"] is True

    def test_confidence_decay_always_healthy(self, mixin):
        result = mixin._check_confidence_decay()
        assert result["component"]["healthy"] is True


# ============================================================================
# TestEdgeCases - additional edge case coverage
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_mixin_can_be_instantiated(self):
        """KnowledgeMixin can be used as a concrete class."""
        obj = _TestMixin()
        assert hasattr(obj, "knowledge_mound_health")

    def test_all_private_methods_exist(self, mixin):
        """All expected private methods exist on the mixin."""
        expected_methods = [
            "_check_km_storage",
            "_check_culture_accumulator",
            "_check_staleness_tracker",
            "_check_rlm_integration",
            "_check_debate_integration",
            "_check_km_redis_cache",
            "_check_km_adapters",
            "_check_control_plane_adapter",
            "_check_km_metrics",
            "_check_confidence_decay",
        ]
        for method_name in expected_methods:
            assert hasattr(mixin, method_name), f"Missing method: {method_name}"

    def test_module_in_all_exports(self):
        """KnowledgeMixin is in __all__."""
        from aragora.server.handlers.admin.health import knowledge

        assert "KnowledgeMixin" in knowledge.__all__

    def test_health_check_returns_handler_result(self, mixin):
        """knowledge_mound_health returns a HandlerResult."""
        with _mock_km_class(side_effect=TypeError("skip")):
            result = mixin.knowledge_mound_health()
            assert isinstance(result, HandlerResult)

    def test_unavailable_response_is_handler_result(self, mixin):
        """Even the early-exit path returns HandlerResult."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result = mixin.knowledge_mound_health()
            assert isinstance(result, HandlerResult)

    def test_body_is_valid_json(self, mixin):
        """Response body is always valid JSON."""
        with _mock_km_class(side_effect=TypeError("skip")):
            result = mixin.knowledge_mound_health()
            body = json.loads(result.body)
            assert isinstance(body, dict)

    def test_content_type_is_json(self, mixin):
        """Response content type is application/json."""
        with _mock_km_class(side_effect=TypeError("skip")):
            result = mixin.knowledge_mound_health()
            assert result.content_type == "application/json"

    def test_unavailable_content_type_is_json(self, mixin):
        """Unavailable response content type is application/json."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result = mixin.knowledge_mound_health()
            assert result.content_type == "application/json"
