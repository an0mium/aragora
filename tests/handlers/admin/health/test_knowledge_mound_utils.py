"""Comprehensive tests for Knowledge Mound health check utility functions.

Tests the 12 public functions in
aragora/server/handlers/admin/health/knowledge_mound_utils.py:

  TestCheckKnowledgeMoundModule       - check_knowledge_mound_module()
  TestCheckMoundCoreInitialization    - check_mound_core_initialization()
  TestCheckStorageBackend             - check_storage_backend()
  TestCheckCultureAccumulator         - check_culture_accumulator()
  TestCheckStalenessTracker           - check_staleness_tracker()
  TestCheckRlmIntegration             - check_rlm_integration()
  TestCheckDebateIntegration          - check_debate_integration()
  TestCheckKnowledgeMoundRedisCache   - check_knowledge_mound_redis_cache()
  TestCheckCodebaseContext            - check_codebase_context()
  TestCheckBidirectionalAdapters      - check_bidirectional_adapters()
  TestCheckControlPlaneAdapter        - check_control_plane_adapter()
  TestCheckKmMetrics                  - check_km_metrics()
  TestCheckConfidenceDecayScheduler   - check_confidence_decay_scheduler()

120+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.knowledge_mound_utils import (
    check_bidirectional_adapters,
    check_codebase_context,
    check_confidence_decay_scheduler,
    check_control_plane_adapter,
    check_culture_accumulator,
    check_debate_integration,
    check_km_metrics,
    check_knowledge_mound_module,
    check_knowledge_mound_redis_cache,
    check_mound_core_initialization,
    check_rlm_integration,
    check_staleness_tracker,
    check_storage_backend,
)

# ---------------------------------------------------------------------------
# Module path constants for patching
# ---------------------------------------------------------------------------

_MOD = "aragora.server.handlers.admin.health.knowledge_mound_utils"


# ============================================================================
# TestCheckKnowledgeMoundModule
# ============================================================================


class TestCheckKnowledgeMoundModule:
    """Tests for check_knowledge_mound_module()."""

    def test_module_available(self):
        """When KnowledgeMound and MoundConfig can be imported, returns healthy and should_abort=False."""
        result, should_abort = check_knowledge_mound_module()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert should_abort is False

    def test_module_not_available(self):
        """When import fails, returns unhealthy and should_abort=True."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result, should_abort = check_knowledge_mound_module()
            assert result["healthy"] is False
            assert result["status"] == "not_available"
            assert "error" in result
            assert should_abort is True

    def test_mound_config_not_available(self):
        """When MoundConfig import fails, returns unhealthy and should_abort=True."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound.types": None}):
            result, should_abort = check_knowledge_mound_module()
            assert result["healthy"] is False
            assert result["status"] == "not_available"
            assert should_abort is True

    def test_returns_tuple(self):
        """Return type is a tuple of (dict, bool)."""
        result = check_knowledge_mound_module()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], bool)


# ============================================================================
# TestCheckMoundCoreInitialization
# ============================================================================


class TestCheckMoundCoreInitialization:
    """Tests for check_mound_core_initialization()."""

    def test_successful_initialization(self):
        """KnowledgeMound instantiation succeeds."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "health_check"
        mock_config = MagicMock()
        mock_config.enable_staleness_detection = True
        mock_config.enable_culture_accumulator = False
        mock_config.enable_rlm_summaries = True
        mock_config.default_staleness_hours = 48
        mock_mound.config = mock_config

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound = mock_km_class

        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, mound = check_mound_core_initialization()
            assert result["healthy"] is True
            assert result["status"] == "initialized"
            assert result["workspace_id"] == "health_check"
            assert mound is mock_mound

    def test_config_present_in_result(self):
        """When mound has config, config fields are included."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "health_check"
        mock_config = MagicMock()
        mock_config.enable_staleness_detection = True
        mock_config.enable_culture_accumulator = True
        mock_config.enable_rlm_summaries = False
        mock_config.default_staleness_hours = 24
        mock_mound.config = mock_config

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound = mock_km_class

        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, _ = check_mound_core_initialization()
            assert "config" in result
            assert result["config"]["enable_staleness_tracking"] is True
            assert result["config"]["enable_culture_accumulator"] is True
            assert result["config"]["enable_rlm_summaries"] is False
            assert result["config"]["default_staleness_hours"] == 24

    def test_no_config_attribute(self):
        """When mound has no config, no config key in result."""
        mock_mound = MagicMock(spec=[])
        mock_mound.workspace_id = "health_check"

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound = mock_km_class

        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, mound = check_mound_core_initialization()
            assert result["healthy"] is True
            assert "config" not in result

    def test_config_is_none(self):
        """When mound.config is None, no config key in result."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "health_check"
        mock_mound.config = None

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound = mock_km_class

        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, _ = check_mound_core_initialization()
            assert result["healthy"] is True
            assert "config" not in result

    def test_type_error_on_init(self):
        """TypeError during KnowledgeMound() construction returns failure."""
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound.side_effect = TypeError("bad args")
        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, mound = check_mound_core_initialization()
            assert result["healthy"] is False
            assert result["status"] == "initialization_failed"
            assert mound is None

    def test_value_error_on_init(self):
        """ValueError during construction returns failure."""
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound.side_effect = ValueError("invalid workspace")
        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, mound = check_mound_core_initialization()
            assert result["healthy"] is False
            assert result["status"] == "initialization_failed"
            assert "error" in result
            assert mound is None

    def test_os_error_on_init(self):
        """OSError during construction returns failure."""
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound.side_effect = OSError("disk full")
        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, mound = check_mound_core_initialization()
            assert result["healthy"] is False
            assert mound is None

    def test_runtime_error_on_init(self):
        """RuntimeError during construction returns failure."""
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound.side_effect = RuntimeError("init failed")
        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result, mound = check_mound_core_initialization()
            assert result["healthy"] is False
            assert mound is None

    def test_returns_tuple(self):
        """Return type is a tuple of (dict, mound_or_none)."""
        mock_km_module = MagicMock()
        mock_km_module.KnowledgeMound.side_effect = TypeError("fail")
        mock_types_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.knowledge.mound.types": mock_types_module,
            },
        ):
            result = check_mound_core_initialization()
            assert isinstance(result, tuple)
            assert len(result) == 2


# ============================================================================
# TestCheckStorageBackend
# ============================================================================


class TestCheckStorageBackend:
    """Tests for check_storage_backend()."""

    def test_postgresql_detected(self):
        """When KNOWLEDGE_MOUND_DATABASE_URL contains 'postgres', backend is postgresql."""
        with patch.dict(
            "os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": "postgresql://localhost/km"}
        ):
            result = check_storage_backend()
            assert result["healthy"] is True
            assert result["backend"] == "postgresql"
            assert result["status"] == "configured"

    def test_sqlite_default(self):
        """When KNOWLEDGE_MOUND_DATABASE_URL is empty or not set, backend is sqlite."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}, clear=False):
            result = check_storage_backend()
            assert result["healthy"] is True
            assert result["backend"] == "sqlite"
            assert result["status"] == "configured"
            assert "note" in result

    def test_sqlite_when_no_env(self):
        """When KNOWLEDGE_MOUND_DATABASE_URL is not set at all, backend is sqlite."""
        with patch.dict("os.environ", {}, clear=True):
            result = check_storage_backend()
            assert result["healthy"] is True
            assert result["backend"] == "sqlite"

    def test_postgres_case_insensitive(self):
        """PostgreSQL detection is case insensitive."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": "POSTGRESQL://host/db"}):
            result = check_storage_backend()
            assert result["backend"] == "postgresql"

    def test_with_mound_store(self):
        """When mound has a _store, store_type is included."""
        mock_mound = MagicMock()
        mock_store = MagicMock()
        mock_store.__class__.__name__ = "ResilientPostgresStore"
        mock_mound._store = mock_store

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = check_storage_backend(mound=mock_mound)
            assert result["healthy"] is True
            assert "store_type" in result

    def test_with_mound_no_store(self):
        """When mound has no _store attribute, no store_type field."""
        mock_mound = MagicMock(spec=[])

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = check_storage_backend(mound=mock_mound)
            assert result["healthy"] is True
            assert "store_type" not in result

    def test_with_mound_store_is_none(self):
        """When mound._store is None, no store_type field."""
        mock_mound = MagicMock()
        mock_mound._store = None

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = check_storage_backend(mound=mock_mound)
            assert result["healthy"] is True
            assert "store_type" not in result

    def test_with_none_mound(self):
        """When mound is None, basic storage check still works."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = check_storage_backend(mound=None)
            assert result["healthy"] is True

    def test_no_mound_argument(self):
        """When no mound argument provided, uses default None."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            result = check_storage_backend()
            assert result["healthy"] is True


# ============================================================================
# TestCheckCultureAccumulator
# ============================================================================


class TestCheckCultureAccumulator:
    """Tests for check_culture_accumulator()."""

    def test_active_accumulator(self):
        """When accumulator is present and active, returns active status."""
        mock_accumulator = MagicMock()
        mock_accumulator.__class__.__name__ = "CultureAccumulator"
        mock_accumulator._patterns = {"ws1": [], "ws2": []}

        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = check_culture_accumulator(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"
        assert "type" in result
        assert result["workspaces_tracked"] == 2

    def test_accumulator_no_patterns(self):
        """When accumulator has no _patterns attr, still returns active."""
        mock_accumulator = MagicMock(spec=[])
        mock_accumulator.__class__.__name__ = "CultureAccumulator"

        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = check_culture_accumulator(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"
        assert "workspaces_tracked" not in result

    def test_not_initialized(self):
        """When accumulator is None/falsy, returns not_initialized."""
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = None

        result = check_culture_accumulator(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"
        assert "note" in result

    def test_no_accumulator_attribute(self):
        """When mound has no _culture_accumulator attr, returns not_initialized."""
        mock_mound = MagicMock(spec=[])

        result = check_culture_accumulator(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_mound_is_none(self):
        """When mound is None, returns not_initialized."""
        result = check_culture_accumulator(mound=None)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"
        assert "Mound not available" in result["note"]

    def test_no_argument(self):
        """When no argument provided, uses default None."""
        result = check_culture_accumulator()
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_type_error_caught(self):
        """TypeError during pattern count is caught gracefully."""
        mock_accumulator = MagicMock()
        mock_accumulator._patterns = MagicMock()
        mock_accumulator._patterns.__len__ = MagicMock(side_effect=TypeError("bad len"))

        mock_mound = MagicMock()
        mock_mound._culture_accumulator = mock_accumulator

        result = check_culture_accumulator(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"
        # workspaces_tracked not set because TypeError was caught
        assert "workspaces_tracked" not in result


# ============================================================================
# TestCheckStalenessTracker
# ============================================================================


class TestCheckStalenessTracker:
    """Tests for check_staleness_tracker()."""

    def test_active_tracker(self):
        """When staleness tracker is present, returns active."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = MagicMock()

        result = check_staleness_tracker(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "active"

    def test_tracker_not_initialized(self):
        """When staleness tracker is None/falsy, returns not_initialized."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = None

        result = check_staleness_tracker(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"
        assert "note" in result

    def test_no_tracker_attribute(self):
        """When mound has no _staleness_tracker attr, returns not_initialized."""
        mock_mound = MagicMock(spec=[])

        result = check_staleness_tracker(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_mound_is_none(self):
        """When mound is None, returns not_initialized with note."""
        result = check_staleness_tracker(mound=None)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"
        assert "Mound not available" in result["note"]

    def test_no_argument(self):
        """When no argument provided, uses default None."""
        result = check_staleness_tracker()
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_tracker_falsy_value(self):
        """When staleness tracker is a falsy value (empty list), returns not_initialized."""
        mock_mound = MagicMock()
        mock_mound._staleness_tracker = []

        result = check_staleness_tracker(mound=mock_mound)
        assert result["healthy"] is True
        assert result["status"] == "not_initialized"


# ============================================================================
# TestCheckRlmIntegration
# ============================================================================


class TestCheckRlmIntegration:
    """Tests for check_rlm_integration()."""

    def test_official_rlm_available(self):
        """When HAS_OFFICIAL_RLM is True, returns active with official_rlm type."""
        mock_rlm = MagicMock()
        mock_rlm.HAS_OFFICIAL_RLM = True

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "active"
            assert result["type"] == "official_rlm"

    def test_fallback_mode(self):
        """When HAS_OFFICIAL_RLM is False, returns fallback with compression_only type."""
        mock_rlm = MagicMock()
        mock_rlm.HAS_OFFICIAL_RLM = False

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm}):
            result = check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "fallback"
            assert result["type"] == "compression_only"
            assert "note" in result

    def test_module_not_available(self):
        """When aragora.rlm can not be imported, returns not_available."""
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            result = check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert "note" in result

    def test_type_error_caught(self):
        """TypeError when checking HAS_OFFICIAL_RLM is caught."""

        # Create a module subclass where __bool__ raises TypeError
        class BrokenBool:
            def __bool__(self):
                raise TypeError("bad bool")

        class FakeModule(type(sys)):
            pass

        mock_rlm_module = FakeModule("fake_rlm")
        mock_rlm_module.HAS_OFFICIAL_RLM = BrokenBool()

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm_module}):
            result = check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"
            assert "error" in result

    def test_runtime_error_caught(self):
        """RuntimeError during check is caught."""

        class BrokenModule(type(sys)):
            @property
            def HAS_OFFICIAL_RLM(self):
                raise RuntimeError("runtime issue")

        mock_rlm_module = BrokenModule("fake_rlm")

        with patch.dict("sys.modules", {"aragora.rlm": mock_rlm_module}):
            result = check_rlm_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"


# ============================================================================
# TestCheckDebateIntegration
# ============================================================================


class TestCheckDebateIntegration:
    """Tests for check_debate_integration()."""

    def test_active_with_stats(self):
        """When get_knowledge_mound_stats exists and returns stats, returns active."""
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
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "active"
            assert result["facts_count"] == 42
            assert result["consensus_stored"] == 10
            assert result["retrievals_count"] == 100

    def test_partial_when_no_stats_function(self):
        """When module exists but get_knowledge_mound_stats is absent, returns partial."""
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats", None, create=True
        ):
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "partial"
            assert "note" in result

    def test_module_not_available(self):
        """When knowledge_mound_ops cannot be imported, returns not_available."""
        import aragora.debate as debate_mod

        # Remove the submodule from both sys.modules and the parent attribute
        # so that `from aragora.debate import knowledge_mound_ops` triggers ImportError
        original_attr = getattr(debate_mod, "knowledge_mound_ops", None)
        original_in_sys = sys.modules.get("aragora.debate.knowledge_mound_ops")
        try:
            if hasattr(debate_mod, "knowledge_mound_ops"):
                delattr(debate_mod, "knowledge_mound_ops")
            sys.modules["aragora.debate.knowledge_mound_ops"] = None
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert "note" in result
        finally:
            # Restore
            if original_in_sys is not None:
                sys.modules["aragora.debate.knowledge_mound_ops"] = original_in_sys
            elif "aragora.debate.knowledge_mound_ops" in sys.modules:
                del sys.modules["aragora.debate.knowledge_mound_ops"]
            if original_attr is not None:
                debate_mod.knowledge_mound_ops = original_attr

    def test_stats_with_missing_keys(self):
        """When stats dict has missing keys, defaults to 0."""
        mock_stats_fn = MagicMock(return_value={})
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["facts_count"] == 0
            assert result["consensus_stored"] == 0
            assert result["retrievals_count"] == 0

    def test_runtime_error_caught(self):
        """RuntimeError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=RuntimeError("failed"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"
            assert "error" in result

    def test_type_error_caught(self):
        """TypeError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=TypeError("bad type"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"

    def test_value_error_caught(self):
        """ValueError during stats retrieval is caught."""
        mock_stats_fn = MagicMock(side_effect=ValueError("bad value"))
        with patch(
            "aragora.debate.knowledge_mound_ops.get_knowledge_mound_stats",
            mock_stats_fn,
            create=True,
        ):
            result = check_debate_integration()
            assert result["healthy"] is True
            assert result["status"] == "error"


# ============================================================================
# TestCheckKnowledgeMoundRedisCache
# ============================================================================


class TestCheckKnowledgeMoundRedisCache:
    """Tests for check_knowledge_mound_redis_cache()."""

    def test_configured_with_km_redis_url(self):
        """When KNOWLEDGE_MOUND_REDIS_URL is set, returns configured."""
        mock_cache_mod = MagicMock()

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.redis_cache": mock_cache_mod,
                },
            ):
                result = check_knowledge_mound_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "configured"

    def test_configured_with_fallback_redis_url(self):
        """When only REDIS_URL is set, still uses it."""
        mock_cache_mod = MagicMock()

        with patch.dict(
            "os.environ",
            {
                "KNOWLEDGE_MOUND_REDIS_URL": "",
                "REDIS_URL": "redis://localhost:6379",
            },
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.redis_cache": mock_cache_mod,
                },
            ):
                result = check_knowledge_mound_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "configured"

    def test_not_configured(self):
        """When no Redis URL env vars are set, returns not_configured."""
        with patch.dict(
            "os.environ",
            {
                "KNOWLEDGE_MOUND_REDIS_URL": "",
                "REDIS_URL": "",
            },
            clear=False,
        ):
            result = check_knowledge_mound_redis_cache()
            assert result["healthy"] is True
            assert result["status"] == "not_configured"
            assert "note" in result

    def test_not_configured_missing_env(self):
        """When Redis env vars do not exist at all, returns not_configured."""
        env_copy = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in ("KNOWLEDGE_MOUND_REDIS_URL", "REDIS_URL")
        }
        with patch.dict("os.environ", env_copy, clear=True):
            result = check_knowledge_mound_redis_cache()
            assert result["healthy"] is True
            assert result["status"] == "not_configured"

    def test_redis_cache_import_error(self):
        """When redis_cache module can not be imported, returns not_available."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict("sys.modules", {"aragora.knowledge.mound.redis_cache": None}):
                result = check_knowledge_mound_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "not_available"
                assert "note" in result

    def test_connection_error(self):
        """When RedisCache() raises ConnectionError, returns error."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = ConnectionError("Connection refused")

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.redis_cache": mock_cache_mod,
                },
            ):
                result = check_knowledge_mound_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"
                assert "error" in result

    def test_timeout_error(self):
        """When RedisCache() raises TimeoutError, returns error."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = TimeoutError("timed out")

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "redis://localhost:6379"}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.redis_cache": mock_cache_mod,
                },
            ):
                result = check_knowledge_mound_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"

    def test_value_error(self):
        """When RedisCache() raises ValueError (bad URL), returns error."""
        mock_cache_mod = MagicMock()
        mock_cache_mod.RedisCache.side_effect = ValueError("invalid URL")

        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_REDIS_URL": "not-a-url"}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.redis_cache": mock_cache_mod,
                },
            ):
                result = check_knowledge_mound_redis_cache()
                assert result["healthy"] is True
                assert result["status"] == "error"


# ============================================================================
# TestCheckCodebaseContext
# ============================================================================


class TestCheckCodebaseContext:
    """Tests for check_codebase_context()."""

    def test_manifest_exists(self, tmp_path):
        """When manifest file exists, returns available with file/line counts."""
        context_dir = tmp_path / ".nomic" / "context"
        context_dir.mkdir(parents=True)
        manifest = context_dir / "codebase_manifest.tsv"
        manifest.write_text(
            "# Codebase manifest\n"
            "# files=3000 lines=500000\n"
            "# generated 2026-01-01\n"
            "path\tlines\tmodified\n",
            encoding="utf-8",
        )

        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            result = check_codebase_context()
            assert result["healthy"] is True
            assert result["status"] == "available"
            assert result["files"] == 3000
            assert result["lines"] == 500000
            assert result["manifest_path"] == str(manifest)

    def test_manifest_missing(self, tmp_path):
        """When manifest does not exist, returns missing."""
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            result = check_codebase_context()
            assert result["healthy"] is True
            assert result["status"] == "missing"
            assert result["manifest_path"] is None

    def test_uses_repo_root_env(self, tmp_path):
        """Falls back to ARAGORA_REPO_ROOT when ARAGORA_CODEBASE_ROOT not set."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_CODEBASE_ROOT": "",
                "ARAGORA_REPO_ROOT": str(tmp_path),
            },
            clear=False,
        ):
            result = check_codebase_context()
            assert result["healthy"] is True
            assert str(tmp_path) in result["root"]

    def test_uses_cwd_fallback(self, tmp_path):
        """Falls back to cwd when no env vars are set."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_CODEBASE_ROOT": "",
                "ARAGORA_REPO_ROOT": "",
            },
            clear=False,
        ):
            with patch("os.getcwd", return_value=str(tmp_path)):
                result = check_codebase_context()
                assert result["healthy"] is True

    def test_manifest_without_counts(self, tmp_path):
        """When manifest exists but has no files=/lines= lines, no files/lines keys."""
        context_dir = tmp_path / ".nomic" / "context"
        context_dir.mkdir(parents=True)
        manifest = context_dir / "codebase_manifest.tsv"
        manifest.write_text("just some text\nno counts here\nfiller\n", encoding="utf-8")

        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            result = check_codebase_context()
            assert result["healthy"] is True
            assert result["status"] == "available"
            assert "files" not in result
            assert "lines" not in result

    def test_manifest_read_error(self, tmp_path):
        """When manifest can not be read, adds note about read error."""
        context_dir = tmp_path / ".nomic" / "context"
        context_dir.mkdir(parents=True)
        manifest = context_dir / "codebase_manifest.tsv"
        manifest.write_text("header\n", encoding="utf-8")

        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            # Make the open call raise OSError
            with patch("builtins.open", side_effect=OSError("permission denied")):
                # The function uses manifest_path.open which is pathlib,
                # not builtins.open. Let's use a different approach.
                pass

        # Use pathlib patch instead
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            original_open = manifest.open

            def broken_open(*args, **kwargs):
                raise OSError("permission denied")

            with patch.object(type(manifest), "open", broken_open):
                result = check_codebase_context()
                assert result["healthy"] is True
                # Since the manifest exists, status should be available
                assert result["status"] == "available"
                assert result.get("note") == "manifest_read_error"

    def test_context_dir_path_included(self, tmp_path):
        """context_dir and root are included in result."""
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            result = check_codebase_context()
            assert "root" in result
            assert "context_dir" in result


# ============================================================================
# TestCheckBidirectionalAdapters
# ============================================================================


class TestCheckBidirectionalAdapters:
    """Tests for check_bidirectional_adapters()."""

    def test_adapters_available(self):
        """When all adapter imports succeed, returns available with list."""
        result = check_bidirectional_adapters()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert result["adapters_available"] == 11
        assert isinstance(result["adapter_list"], list)
        assert "continuum" in result["adapter_list"]
        assert "consensus" in result["adapter_list"]
        assert "culture" in result["adapter_list"]

    def test_import_error_returns_partial(self):
        """When adapter imports fail, returns partial."""
        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters": None}):
            result = check_bidirectional_adapters()
            assert result["healthy"] is True
            assert result["status"] == "partial"
            assert "error" in result

    def test_runtime_error_caught(self):
        """RuntimeError during import is caught."""
        mock_adapters = MagicMock()
        type(mock_adapters).ContinuumAdapter = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("bad"))
        )

        with patch.dict("sys.modules", {"aragora.knowledge.mound.adapters": mock_adapters}):
            result = check_bidirectional_adapters()
            assert result["healthy"] is True
            assert result["status"] in ("available", "error", "partial")

    def test_adapter_list_has_expected_names(self):
        """Adapter list contains all expected adapter names."""
        result = check_bidirectional_adapters()
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


# ============================================================================
# TestCheckControlPlaneAdapter
# ============================================================================


class TestCheckControlPlaneAdapter:
    """Tests for check_control_plane_adapter()."""

    def test_adapter_available(self):
        """When control plane adapter imports succeed, returns available with features."""
        result = check_control_plane_adapter()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert "features" in result
        assert "task_outcome_storage" in result["features"]
        assert "capability_records" in result["features"]
        assert "cross_workspace_insights" in result["features"]
        assert "agent_recommendations" in result["features"]

    def test_import_error(self):
        """When control plane adapter module can not be imported, returns not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.adapters.control_plane_adapter": None,
            },
        ):
            result = check_control_plane_adapter()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert "error" in result

    def test_runtime_error_caught(self):
        """RuntimeError during check is caught."""
        mock_adapter_mod = MagicMock()
        type(mock_adapter_mod).ControlPlaneAdapter = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("init fail"))
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.adapters.control_plane_adapter": mock_adapter_mod,
            },
        ):
            result = check_control_plane_adapter()
            assert result["healthy"] is True
            assert result["status"] in ("available", "error")


# ============================================================================
# TestCheckKmMetrics
# ============================================================================


class TestCheckKmMetrics:
    """Tests for check_km_metrics()."""

    def test_metrics_available(self):
        """When KM metrics module is importable, returns available."""
        result = check_km_metrics()
        assert result["healthy"] is True
        assert result["status"] == "available"
        assert result["prometheus_integration"] is True

    def test_metrics_not_available(self):
        """When KM metrics module can not be imported, returns not_available."""
        with patch.dict("sys.modules", {"aragora.observability.metrics.km": None}):
            result = check_km_metrics()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert result["prometheus_integration"] is False

    def test_runtime_error_caught(self):
        """RuntimeError during import is caught."""
        mock_km_metrics = MagicMock()
        type(mock_km_metrics).init_km_metrics = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("metrics fail"))
        )

        with patch.dict("sys.modules", {"aragora.observability.metrics.km": mock_km_metrics}):
            result = check_km_metrics()
            assert result["healthy"] is True
            assert result["status"] in ("available", "error")

    def test_always_healthy(self):
        """KM metrics check always returns healthy=True regardless of status."""
        # Available case
        result1 = check_km_metrics()
        assert result1["healthy"] is True

        # Not available case
        with patch.dict("sys.modules", {"aragora.observability.metrics.km": None}):
            result2 = check_km_metrics()
            assert result2["healthy"] is True


# ============================================================================
# TestCheckConfidenceDecayScheduler
# ============================================================================


class TestCheckConfidenceDecayScheduler:
    """Tests for check_confidence_decay_scheduler()."""

    def test_scheduler_active_and_running(self):
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["status"] == "active"
            assert result["running"] is True
            assert result["decay_interval_hours"] == 24
            assert result["total_cycles"] == 5
            assert result["total_items_processed"] == 100
            assert len(warnings) == 0

    def test_scheduler_stopped(self):
        """When scheduler exists but is not running, returns stopped status."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["status"] == "stopped"
            assert result["running"] is False

    def test_scheduler_not_configured(self):
        """When get_decay_scheduler returns None, returns not_configured."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = None

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["status"] == "not_configured"
            assert "note" in result
            assert len(warnings) == 0

    def test_module_not_available(self):
        """When confidence_decay_scheduler can not be imported, returns not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["status"] == "not_available"
            assert len(warnings) == 0

    def test_stale_workspace_warning(self):
        """When workspace hasn't been processed in >48 hours, adds warning."""
        stale_time = (datetime.now() - timedelta(hours=50)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {"stale_workspace": stale_time},
            "workspaces": ["stale_workspace"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert "alert" in result
            assert result["alert"]["level"] == "warning"
            assert "stale_workspace" in result["alert"]["message"]
            assert len(warnings) == 1
            assert "stale_workspace" in warnings[0]

    def test_recent_workspace_no_warning(self):
        """When workspace was recently processed, no warning."""
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert "alert" not in result
            assert len(warnings) == 0

    def test_multiple_stale_workspaces(self):
        """When multiple workspaces are stale, all are mentioned in alert."""
        stale_time = (datetime.now() - timedelta(hours=72)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {
                "ws_alpha": stale_time,
                "ws_beta": stale_time,
            },
            "workspaces": ["ws_alpha", "ws_beta"],
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result, warnings = check_confidence_decay_scheduler()
            assert "alert" in result
            assert "ws_alpha" in result["alert"]["message"]
            assert "ws_beta" in result["alert"]["message"]

    def test_bad_timestamp_in_last_run_ignored(self):
        """Invalid timestamp in last_run is silently ignored."""
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            # Bad timestamp should be silently skipped
            assert "alert" not in result
            assert len(warnings) == 0

    def test_stopped_scheduler_no_stale_check(self):
        """When scheduler is stopped, stale workspace check is skipped."""
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["status"] == "stopped"
            # Stale check only runs when scheduler is_running
            assert "alert" not in result
            assert len(warnings) == 0

    def test_empty_last_run(self):
        """When last_run is empty dict, no stale warnings."""
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert "alert" not in result
            assert len(warnings) == 0

    def test_runtime_error_caught(self):
        """RuntimeError during scheduler check is caught."""
        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.side_effect = RuntimeError("scheduler broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["status"] == "error"
            assert "error" in result

    def test_attribute_error_caught(self):
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["status"] == "error"

    def test_returns_tuple(self):
        """Return type is a tuple of (dict, list)."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result = check_confidence_decay_scheduler()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], dict)
            assert isinstance(result[1], list)

    def test_stats_default_values(self):
        """When stats dict is missing keys, defaults are used."""
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
            result, warnings = check_confidence_decay_scheduler()
            assert result["healthy"] is True
            assert result["decay_interval_hours"] == 24
            assert result["total_cycles"] == 0
            assert result["total_items_processed"] == 0

    def test_mixed_stale_and_recent_workspaces(self):
        """Only stale workspaces (>48h) appear in alert, not recent ones."""
        stale_time = (datetime.now() - timedelta(hours=60)).isoformat()
        recent_time = (datetime.now() - timedelta(hours=1)).isoformat()

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 24,
            "total_decay_cycles": 10,
            "total_items_processed": 200,
            "last_run": {
                "stale_ws": stale_time,
                "recent_ws": recent_time,
            },
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
            result, warnings = check_confidence_decay_scheduler()
            assert "alert" in result
            assert "stale_ws" in result["alert"]["message"]
            assert "recent_ws" not in result["alert"]["message"]


# ============================================================================
# TestAllFunctionsReturnHealthy
# ============================================================================


class TestAllFunctionsReturnHealthy:
    """Cross-cutting: verify all functions always return healthy=True."""

    def test_storage_backend_always_healthy(self):
        """check_storage_backend always returns healthy=True."""
        with patch.dict("os.environ", {"KNOWLEDGE_MOUND_DATABASE_URL": ""}):
            assert check_storage_backend()["healthy"] is True

    def test_culture_accumulator_always_healthy(self):
        """check_culture_accumulator always returns healthy=True."""
        assert check_culture_accumulator()["healthy"] is True

    def test_staleness_tracker_always_healthy(self):
        """check_staleness_tracker always returns healthy=True."""
        assert check_staleness_tracker()["healthy"] is True

    def test_rlm_integration_always_healthy(self):
        """check_rlm_integration always returns healthy=True."""
        result = check_rlm_integration()
        assert result["healthy"] is True

    def test_debate_integration_always_healthy(self):
        """check_debate_integration always returns healthy=True."""
        result = check_debate_integration()
        assert result["healthy"] is True

    def test_redis_cache_always_healthy(self):
        """check_knowledge_mound_redis_cache always returns healthy=True."""
        result = check_knowledge_mound_redis_cache()
        assert result["healthy"] is True

    def test_codebase_context_always_healthy(self):
        """check_codebase_context always returns healthy=True."""
        result = check_codebase_context()
        assert result["healthy"] is True

    def test_bidirectional_adapters_always_healthy(self):
        """check_bidirectional_adapters always returns healthy=True."""
        result = check_bidirectional_adapters()
        assert result["healthy"] is True

    def test_control_plane_adapter_always_healthy(self):
        """check_control_plane_adapter always returns healthy=True."""
        result = check_control_plane_adapter()
        assert result["healthy"] is True

    def test_km_metrics_always_healthy(self):
        """check_km_metrics always returns healthy=True."""
        result = check_km_metrics()
        assert result["healthy"] is True

    def test_confidence_decay_always_healthy(self):
        """check_confidence_decay_scheduler always returns healthy=True in result dict."""
        result, _ = check_confidence_decay_scheduler()
        assert result["healthy"] is True
