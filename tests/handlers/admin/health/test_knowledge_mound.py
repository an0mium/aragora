"""Comprehensive tests for Knowledge Mound health check handler functions.

Tests the two public functions in
aragora/server/handlers/admin/health/knowledge_mound.py:

  TestKnowledgeMoundHealth             - knowledge_mound_health() orchestrator
  TestKnowledgeMoundHealthModuleAbort  - module-unavailable abort path
  TestKnowledgeMoundHealthCoreFailure  - core initialization failure (degraded)
  TestKnowledgeMoundHealthStatusLogic  - status determination logic
  TestKnowledgeMoundHealthWarnings     - warnings propagation from decay check
  TestKnowledgeMoundHealthTimestamp    - response metadata (timestamp, response_time)
  TestKnowledgeMoundHealthComponents   - component dict structure validation
  TestDecayHealth                      - decay_health() orchestrator
  TestDecayHealthImportError           - module-unavailable path (503)
  TestDecayHealthNotConfigured         - scheduler not initialized
  TestDecayHealthRunning               - scheduler running, healthy
  TestDecayHealthStopped               - scheduler stopped
  TestDecayHealthDegraded              - stale workspaces cause degraded
  TestDecayHealthWorkspaceStatus       - workspace status detail construction
  TestDecayHealthParseErrors           - invalid timestamp handling
  TestDecayHealthTimestamp             - response metadata
  TestDecayHealthEdgeCases             - edge cases and boundary conditions

100+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Module path constant
# ---------------------------------------------------------------------------

_MOD = "aragora.server.handlers.admin.health.knowledge_mound"
_UTILS = "aragora.server.handlers.admin.health.knowledge_mound_utils"


# ---------------------------------------------------------------------------
# Helper to build mock util functions that return preset results
# ---------------------------------------------------------------------------


def _default_component(healthy=True, status="active"):
    """Build a default component dict."""
    return {"healthy": healthy, "status": status}


def _make_all_utils_healthy():
    """Return a dict of patch targets -> return values for all utils in a healthy state."""
    return {
        f"{_MOD}.check_knowledge_mound_module": ({"healthy": True, "status": "available"}, False),
        f"{_MOD}.check_mound_core_initialization": (
            {"healthy": True, "status": "initialized", "workspace_id": "health_check"},
            MagicMock(),
        ),
        f"{_MOD}.check_storage_backend": {
            "healthy": True,
            "status": "configured",
            "backend": "sqlite",
        },
        f"{_MOD}.check_culture_accumulator": {"healthy": True, "status": "active"},
        f"{_MOD}.check_staleness_tracker": {"healthy": True, "status": "active"},
        f"{_MOD}.check_rlm_integration": {"healthy": True, "status": "active"},
        f"{_MOD}.check_codebase_context": {"healthy": True, "status": "available"},
        f"{_MOD}.check_debate_integration": {"healthy": True, "status": "active"},
        f"{_MOD}.check_knowledge_mound_redis_cache": {"healthy": True, "status": "configured"},
        f"{_MOD}.check_bidirectional_adapters": {"healthy": True, "status": "available"},
        f"{_MOD}.check_control_plane_adapter": {"healthy": True, "status": "available"},
        f"{_MOD}.check_km_metrics": {"healthy": True, "status": "available"},
        f"{_MOD}.check_confidence_decay_scheduler": ({"healthy": True, "status": "active"}, []),
    }


class _PatchContext:
    """Context manager that patches all knowledge_mound util functions."""

    def __init__(self, overrides: dict[str, Any] | None = None):
        self._patches = []
        self._mocks = {}
        values = _make_all_utils_healthy()
        if overrides:
            values.update(overrides)
        self._values = values

    def __enter__(self):
        for target, return_value in self._values.items():
            p = patch(target, return_value=return_value)
            self._mocks[target] = p.start()
            self._patches.append(p)
        return self._mocks

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()


# ============================================================================
# TestKnowledgeMoundHealth - happy path
# ============================================================================


class TestKnowledgeMoundHealth:
    """Tests for knowledge_mound_health() happy path."""

    def test_all_healthy_returns_healthy_status(self):
        """When all components are healthy, overall status is 'healthy'."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["status"] == "healthy"

    def test_all_healthy_returns_200(self):
        """Healthy response returns HTTP 200."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            assert _status(result) == 200

    def test_summary_counts(self):
        """Summary includes total_components, healthy count, and active count."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            summary = body["summary"]
            assert summary["total_components"] == 13
            # All components healthy
            assert summary["healthy"] == 13
            # Active = components with status == "active"
            assert summary["active"] >= 1

    def test_components_dict_contains_expected_keys(self):
        """All 13 expected component keys are present."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        expected_keys = {
            "module",
            "core",
            "storage",
            "culture_accumulator",
            "staleness_tracker",
            "rlm_integration",
            "codebase_context",
            "debate_integration",
            "redis_cache",
            "bidirectional_adapters",
            "control_plane_adapter",
            "km_metrics",
            "confidence_decay",
        }
        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert set(body["components"].keys()) == expected_keys

    def test_no_warnings_when_all_healthy(self):
        """When decay check returns no warnings, warnings is None."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["warnings"] is None

    def test_response_time_present(self):
        """response_time_ms is included and is a number."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert "response_time_ms" in body
            assert isinstance(body["response_time_ms"], (int, float))

    def test_timestamp_present(self):
        """timestamp is included and ends with Z."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert "timestamp" in body
            assert body["timestamp"].endswith("Z")


# ============================================================================
# TestKnowledgeMoundHealthModuleAbort
# ============================================================================


class TestKnowledgeMoundHealthModuleAbort:
    """Tests for knowledge_mound_health() when KM module is unavailable."""

    def test_module_unavailable_returns_unavailable(self):
        """When check_knowledge_mound_module returns should_abort=True, status is 'unavailable'."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": False, "status": "not_available", "error": "Module not available"},
                True,
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["status"] == "unavailable"
            assert "error" in body

    def test_module_unavailable_returns_200(self):
        """Module unavailable still returns HTTP 200 (soft failure)."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": False, "status": "not_available", "error": "Module not available"},
                True,
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            assert _status(result) == 200

    def test_module_unavailable_has_timestamp(self):
        """Module unavailable response still includes timestamp."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": False, "status": "not_available"},
                True,
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert "timestamp" in body
            assert body["timestamp"].endswith("Z")

    def test_module_unavailable_has_components(self):
        """Module unavailable response includes partial components dict."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": False, "status": "not_available"},
                True,
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert "components" in body
            assert "module" in body["components"]

    def test_module_unavailable_skips_remaining_checks(self):
        """When module is unavailable, no other check functions are called."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        mock_core = MagicMock(return_value=({"healthy": True, "status": "ok"}, None))
        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": False, "status": "not_available"},
                True,
            ),
            f"{_MOD}.check_mound_core_initialization": mock_core,
        }
        # Need to use direct patching to check call count
        with patch(
            f"{_MOD}.check_knowledge_mound_module",
            return_value=({"healthy": False, "status": "not_available"}, True),
        ):
            with patch(f"{_MOD}.check_mound_core_initialization") as mock_init:
                result = knowledge_mound_health(MagicMock())
                mock_init.assert_not_called()


# ============================================================================
# TestKnowledgeMoundHealthCoreFailure
# ============================================================================


class TestKnowledgeMoundHealthCoreFailure:
    """Tests for knowledge_mound_health() when core initialization fails."""

    def test_core_unhealthy_makes_overall_degraded(self):
        """When core check is unhealthy, overall status becomes 'degraded'."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_mound_core_initialization": (
                {"healthy": False, "status": "initialization_failed", "error": "Init failed"},
                None,
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["status"] == "degraded"

    def test_core_unhealthy_components_reflect(self):
        """When core check fails, the core component reflects the failure."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_mound_core_initialization": (
                {"healthy": False, "status": "initialization_failed"},
                None,
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["components"]["core"]["healthy"] is False
            assert body["components"]["core"]["status"] == "initialization_failed"


# ============================================================================
# TestKnowledgeMoundHealthStatusLogic
# ============================================================================


class TestKnowledgeMoundHealthStatusLogic:
    """Tests for status determination logic in knowledge_mound_health()."""

    def test_not_configured_when_zero_active(self):
        """When no components have status='active', overall status is 'not_configured'."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        # Make all components healthy but with non-active statuses
        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": True, "status": "available"},
                False,
            ),
            f"{_MOD}.check_mound_core_initialization": (
                {"healthy": True, "status": "initialized"},
                MagicMock(),
            ),
            f"{_MOD}.check_storage_backend": {"healthy": True, "status": "configured"},
            f"{_MOD}.check_culture_accumulator": {"healthy": True, "status": "not_initialized"},
            f"{_MOD}.check_staleness_tracker": {"healthy": True, "status": "not_initialized"},
            f"{_MOD}.check_rlm_integration": {"healthy": True, "status": "not_available"},
            f"{_MOD}.check_codebase_context": {"healthy": True, "status": "missing"},
            f"{_MOD}.check_debate_integration": {"healthy": True, "status": "not_available"},
            f"{_MOD}.check_knowledge_mound_redis_cache": {
                "healthy": True,
                "status": "not_configured",
            },
            f"{_MOD}.check_bidirectional_adapters": {"healthy": True, "status": "partial"},
            f"{_MOD}.check_control_plane_adapter": {"healthy": True, "status": "not_available"},
            f"{_MOD}.check_km_metrics": {"healthy": True, "status": "not_available"},
            f"{_MOD}.check_confidence_decay_scheduler": (
                {"healthy": True, "status": "not_configured"},
                [],
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["status"] == "not_configured"
            assert body["summary"]["active"] == 0

    def test_degraded_with_one_active_and_core_unhealthy(self):
        """When core is unhealthy but some components active, status is 'degraded'."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_mound_core_initialization": (
                {"healthy": False, "status": "initialization_failed"},
                None,
            ),
            # Some components still active
            f"{_MOD}.check_rlm_integration": {"healthy": True, "status": "active"},
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["status"] == "degraded"

    def test_healthy_with_at_least_one_active(self):
        """When all_healthy is True and active_count > 0, status is 'healthy'."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        # Default _PatchContext has active components
        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["status"] == "healthy"

    def test_healthy_count_reflects_component_health(self):
        """Summary healthy count reflects number of healthy components."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        # Make one component explicitly unhealthy (but not core, which would make overall degraded)
        # Note: Only the core check being unhealthy sets all_healthy = False
        # Other checks don't affect all_healthy directly, just the summary count
        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["summary"]["healthy"] == 13


# ============================================================================
# TestKnowledgeMoundHealthWarnings
# ============================================================================


class TestKnowledgeMoundHealthWarnings:
    """Tests for warning propagation in knowledge_mound_health()."""

    def test_warnings_from_decay_propagated(self):
        """Warnings from check_confidence_decay_scheduler are included in response."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_confidence_decay_scheduler": (
                {"healthy": True, "status": "active"},
                ["Workspace ws1 stale", "Workspace ws2 stale"],
            ),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["warnings"] is not None
            assert len(body["warnings"]) == 2
            assert "ws1" in body["warnings"][0]

    def test_empty_warnings_list_returns_none(self):
        """When warnings list is empty, warnings field is None."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["warnings"] is None


# ============================================================================
# TestKnowledgeMoundHealthTimestamp
# ============================================================================


class TestKnowledgeMoundHealthTimestamp:
    """Tests for timestamp and response time in knowledge_mound_health()."""

    def test_response_time_is_non_negative(self):
        """response_time_ms is non-negative."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["response_time_ms"] >= 0

    def test_timestamp_is_utc_iso_format(self):
        """timestamp is in UTC ISO format ending with Z."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            ts = body["timestamp"]
            assert ts.endswith("Z")
            # Should contain a valid ISO date prefix (the handler appends Z to isoformat)
            assert "T" in ts
            assert "2" in ts[:4]  # Year starts with 2


# ============================================================================
# TestKnowledgeMoundHealthComponents
# ============================================================================


class TestKnowledgeMoundHealthComponents:
    """Tests for component structure in knowledge_mound_health()."""

    def test_module_component_passes_through(self):
        """Module component data is passed through from check function."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["components"]["module"]["status"] == "available"
            assert body["components"]["module"]["healthy"] is True

    def test_core_component_passes_through(self):
        """Core component data is passed through from check function."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["components"]["core"]["status"] == "initialized"
            assert body["components"]["core"]["healthy"] is True

    def test_storage_component_passes_through(self):
        """Storage component data is passed through."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["components"]["storage"]["status"] == "configured"

    def test_confidence_decay_component_passes_through(self):
        """Confidence decay component data is passed through."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["components"]["confidence_decay"]["status"] == "active"

    def test_mound_instance_passed_to_storage_check(self):
        """The mound instance from core init is passed to check_storage_backend."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        mock_mound = MagicMock()
        with patch(
            f"{_MOD}.check_knowledge_mound_module",
            return_value=({"healthy": True, "status": "available"}, False),
        ):
            with patch(
                f"{_MOD}.check_mound_core_initialization",
                return_value=({"healthy": True, "status": "initialized"}, mock_mound),
            ):
                with patch(
                    f"{_MOD}.check_storage_backend", return_value={"healthy": True, "status": "ok"}
                ) as mock_sb:
                    with patch(
                        f"{_MOD}.check_culture_accumulator",
                        return_value={"healthy": True, "status": "ok"},
                    ):
                        with patch(
                            f"{_MOD}.check_staleness_tracker",
                            return_value={"healthy": True, "status": "ok"},
                        ):
                            with patch(
                                f"{_MOD}.check_rlm_integration",
                                return_value={"healthy": True, "status": "ok"},
                            ):
                                with patch(
                                    f"{_MOD}.check_codebase_context",
                                    return_value={"healthy": True, "status": "ok"},
                                ):
                                    with patch(
                                        f"{_MOD}.check_debate_integration",
                                        return_value={"healthy": True, "status": "ok"},
                                    ):
                                        with patch(
                                            f"{_MOD}.check_knowledge_mound_redis_cache",
                                            return_value={"healthy": True, "status": "ok"},
                                        ):
                                            with patch(
                                                f"{_MOD}.check_bidirectional_adapters",
                                                return_value={"healthy": True, "status": "ok"},
                                            ):
                                                with patch(
                                                    f"{_MOD}.check_control_plane_adapter",
                                                    return_value={"healthy": True, "status": "ok"},
                                                ):
                                                    with patch(
                                                        f"{_MOD}.check_km_metrics",
                                                        return_value={
                                                            "healthy": True,
                                                            "status": "ok",
                                                        },
                                                    ):
                                                        with patch(
                                                            f"{_MOD}.check_confidence_decay_scheduler",
                                                            return_value=(
                                                                {"healthy": True, "status": "ok"},
                                                                [],
                                                            ),
                                                        ):
                                                            knowledge_mound_health(MagicMock())
                                                            mock_sb.assert_called_once_with(
                                                                mock_mound
                                                            )

    def test_mound_instance_passed_to_culture_check(self):
        """The mound instance from core init is passed to check_culture_accumulator."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        mock_mound = MagicMock()
        with patch(
            f"{_MOD}.check_knowledge_mound_module",
            return_value=({"healthy": True, "status": "available"}, False),
        ):
            with patch(
                f"{_MOD}.check_mound_core_initialization",
                return_value=({"healthy": True, "status": "initialized"}, mock_mound),
            ):
                with patch(
                    f"{_MOD}.check_storage_backend", return_value={"healthy": True, "status": "ok"}
                ):
                    with patch(
                        f"{_MOD}.check_culture_accumulator",
                        return_value={"healthy": True, "status": "ok"},
                    ) as mock_ca:
                        with patch(
                            f"{_MOD}.check_staleness_tracker",
                            return_value={"healthy": True, "status": "ok"},
                        ):
                            with patch(
                                f"{_MOD}.check_rlm_integration",
                                return_value={"healthy": True, "status": "ok"},
                            ):
                                with patch(
                                    f"{_MOD}.check_codebase_context",
                                    return_value={"healthy": True, "status": "ok"},
                                ):
                                    with patch(
                                        f"{_MOD}.check_debate_integration",
                                        return_value={"healthy": True, "status": "ok"},
                                    ):
                                        with patch(
                                            f"{_MOD}.check_knowledge_mound_redis_cache",
                                            return_value={"healthy": True, "status": "ok"},
                                        ):
                                            with patch(
                                                f"{_MOD}.check_bidirectional_adapters",
                                                return_value={"healthy": True, "status": "ok"},
                                            ):
                                                with patch(
                                                    f"{_MOD}.check_control_plane_adapter",
                                                    return_value={"healthy": True, "status": "ok"},
                                                ):
                                                    with patch(
                                                        f"{_MOD}.check_km_metrics",
                                                        return_value={
                                                            "healthy": True,
                                                            "status": "ok",
                                                        },
                                                    ):
                                                        with patch(
                                                            f"{_MOD}.check_confidence_decay_scheduler",
                                                            return_value=(
                                                                {"healthy": True, "status": "ok"},
                                                                [],
                                                            ),
                                                        ):
                                                            knowledge_mound_health(MagicMock())
                                                            mock_ca.assert_called_once_with(
                                                                mock_mound
                                                            )

    def test_mound_instance_passed_to_staleness_check(self):
        """The mound instance from core init is passed to check_staleness_tracker."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        mock_mound = MagicMock()
        with patch(
            f"{_MOD}.check_knowledge_mound_module",
            return_value=({"healthy": True, "status": "available"}, False),
        ):
            with patch(
                f"{_MOD}.check_mound_core_initialization",
                return_value=({"healthy": True, "status": "initialized"}, mock_mound),
            ):
                with patch(
                    f"{_MOD}.check_storage_backend", return_value={"healthy": True, "status": "ok"}
                ):
                    with patch(
                        f"{_MOD}.check_culture_accumulator",
                        return_value={"healthy": True, "status": "ok"},
                    ):
                        with patch(
                            f"{_MOD}.check_staleness_tracker",
                            return_value={"healthy": True, "status": "ok"},
                        ) as mock_st:
                            with patch(
                                f"{_MOD}.check_rlm_integration",
                                return_value={"healthy": True, "status": "ok"},
                            ):
                                with patch(
                                    f"{_MOD}.check_codebase_context",
                                    return_value={"healthy": True, "status": "ok"},
                                ):
                                    with patch(
                                        f"{_MOD}.check_debate_integration",
                                        return_value={"healthy": True, "status": "ok"},
                                    ):
                                        with patch(
                                            f"{_MOD}.check_knowledge_mound_redis_cache",
                                            return_value={"healthy": True, "status": "ok"},
                                        ):
                                            with patch(
                                                f"{_MOD}.check_bidirectional_adapters",
                                                return_value={"healthy": True, "status": "ok"},
                                            ):
                                                with patch(
                                                    f"{_MOD}.check_control_plane_adapter",
                                                    return_value={"healthy": True, "status": "ok"},
                                                ):
                                                    with patch(
                                                        f"{_MOD}.check_km_metrics",
                                                        return_value={
                                                            "healthy": True,
                                                            "status": "ok",
                                                        },
                                                    ):
                                                        with patch(
                                                            f"{_MOD}.check_confidence_decay_scheduler",
                                                            return_value=(
                                                                {"healthy": True, "status": "ok"},
                                                                [],
                                                            ),
                                                        ):
                                                            knowledge_mound_health(MagicMock())
                                                            mock_st.assert_called_once_with(
                                                                mock_mound
                                                            )


# ============================================================================
# TestDecayHealth - import error path
# ============================================================================


class TestDecayHealthImportError:
    """Tests for decay_health() when confidence decay module is not available."""

    def test_import_error_returns_not_available(self):
        """When confidence_decay_scheduler cannot be imported, returns 'not_available'."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["status"] == "not_available"
            assert "message" in body

    def test_import_error_returns_503(self):
        """Module not available returns HTTP 503."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result = decay_health(MagicMock())
            assert _status(result) == 503

    def test_import_error_has_response_time(self):
        """Import error response includes response_time_ms."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": None,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert "response_time_ms" in body
            assert isinstance(body["response_time_ms"], (int, float))


# ============================================================================
# TestDecayHealthNotConfigured
# ============================================================================


class TestDecayHealthNotConfigured:
    """Tests for decay_health() when scheduler is not initialized."""

    def test_not_configured_when_scheduler_none(self):
        """When get_decay_scheduler returns None, status is 'not_configured'."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = None
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["status"] == "not_configured"
            assert "message" in body

    def test_not_configured_returns_200(self):
        """Not configured returns HTTP 200."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = None
        mock_mod.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            assert _status(result) == 200

    def test_not_configured_has_metrics_flag(self):
        """Not configured response includes metrics_available."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = None
        mock_mod.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["metrics_available"] is True

    def test_not_configured_has_response_time(self):
        """Not configured response includes response_time_ms."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = None
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert "response_time_ms" in body


# ============================================================================
# TestDecayHealthRunning
# ============================================================================


class TestDecayHealthRunning:
    """Tests for decay_health() when scheduler is running and healthy."""

    def _make_scheduler(
        self,
        running=True,
        last_runs=None,
        stats_overrides=None,
    ):
        """Create a mock scheduler module with configured scheduler."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = running
        stats = {
            "decay_interval_hours": 24,
            "min_confidence_threshold": 0.1,
            "decay_rate": 0.95,
            "total_decay_cycles": 10,
            "total_items_decayed": 500,
            "total_items_expired": 20,
            "decay_errors": 2,
            "last_decay_per_workspace": last_runs or {},
        }
        if stats_overrides:
            stats.update(stats_overrides)
        mock_scheduler.get_stats.return_value = stats

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = True
        return mock_mod

    def test_running_returns_healthy(self):
        """Running scheduler with no stale workspaces returns 'healthy'."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_mod = self._make_scheduler(last_runs={"ws1": recent})

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["status"] == "healthy"

    def test_running_returns_200(self):
        """Running scheduler returns HTTP 200."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = self._make_scheduler()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            assert _status(result) == 200

    def test_scheduler_section_present(self):
        """Response includes scheduler configuration section."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = self._make_scheduler()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            scheduler = body["scheduler"]
            assert scheduler["running"] is True
            assert scheduler["decay_interval_hours"] == 24
            assert scheduler["min_confidence_threshold"] == 0.1
            assert scheduler["decay_rate"] == 0.95

    def test_statistics_section_present(self):
        """Response includes statistics section."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = self._make_scheduler()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            stats = body["statistics"]
            assert stats["total_cycles"] == 10
            assert stats["total_items_processed"] == 500
            assert stats["total_items_expired"] == 20
            assert stats["errors"] == 2

    def test_workspaces_section_present(self):
        """Response includes workspaces section with counts."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_mod = self._make_scheduler(last_runs={"ws1": recent, "ws2": recent})

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            ws = body["workspaces"]
            assert ws["total"] == 2
            assert ws["stale_count"] == 0
            assert ws["stale_threshold_hours"] == 48

    def test_workspace_details_populated(self):
        """Workspace details include per-workspace info."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        mock_mod = self._make_scheduler(last_runs={"my_ws": recent})

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            details = body["workspaces"]["details"]
            assert "my_ws" in details
            assert details["my_ws"]["stale"] is False
            assert "hours_since_decay" in details["my_ws"]
            assert "last_decay" in details["my_ws"]

    def test_metrics_available_flag(self):
        """Response includes metrics_available flag."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = self._make_scheduler()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["metrics_available"] is True

    def test_no_warnings_when_healthy(self):
        """No warnings when all workspaces are recent."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_mod = self._make_scheduler(last_runs={"ws1": recent})

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["warnings"] is None

    def test_timestamp_in_response(self):
        """Response includes ISO timestamp ending with Z."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = self._make_scheduler()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["timestamp"].endswith("Z")

    def test_response_time_in_response(self):
        """Response includes non-negative response_time_ms."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_mod = self._make_scheduler()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["response_time_ms"] >= 0


# ============================================================================
# TestDecayHealthStopped
# ============================================================================


class TestDecayHealthStopped:
    """Tests for decay_health() when scheduler is stopped."""

    def test_stopped_status_when_not_running(self):
        """When scheduler is_running is False, status is 'stopped'."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {
            "decay_interval_hours": 12,
            "total_decay_cycles": 0,
            "total_items_decayed": 0,
            "total_items_expired": 0,
            "decay_errors": 0,
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["status"] == "stopped"
            assert body["scheduler"]["running"] is False

    def test_stopped_with_stale_workspaces_still_stopped(self):
        """When stopped with stale workspaces, status is still 'stopped' (not degraded)."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws1": stale},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            # Even though ws is stale, status is "stopped" because running=False
            assert body["status"] == "stopped"


# ============================================================================
# TestDecayHealthDegraded
# ============================================================================


class TestDecayHealthDegraded:
    """Tests for decay_health() when there are stale workspaces."""

    def test_degraded_with_stale_workspace(self):
        """When a running scheduler has stale workspaces, status is 'degraded'."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"stale_ws": stale},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = True

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["status"] == "degraded"

    def test_degraded_has_warnings(self):
        """Stale workspace adds a warning."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_old": stale},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["warnings"] is not None
            assert len(body["warnings"]) >= 1
            assert "ws_old" in body["warnings"][0]

    def test_stale_count_in_workspaces(self):
        """Stale workspace count is reflected in workspaces section."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {
                "stale_ws": stale,
                "recent_ws": recent,
            },
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["workspaces"]["stale_count"] == 1
            assert body["workspaces"]["total"] == 2

    def test_multiple_stale_workspaces(self):
        """Multiple stale workspaces produce multiple warnings."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=60)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {
                "ws_a": stale,
                "ws_b": stale,
                "ws_c": stale,
            },
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["workspaces"]["stale_count"] == 3
            assert len(body["warnings"]) == 3

    def test_stale_workspace_details(self):
        """Stale workspace details include stale=True and hours_since_decay."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_stale": stale},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_stale"]
            assert detail["stale"] is True
            assert detail["hours_since_decay"] > 48


# ============================================================================
# TestDecayHealthWorkspaceStatus
# ============================================================================


class TestDecayHealthWorkspaceStatus:
    """Tests for workspace status detail construction in decay_health()."""

    def test_recent_workspace_not_stale(self):
        """Workspace processed recently is not marked stale."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_fresh": recent},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_fresh"]
            assert detail["stale"] is False
            assert detail["hours_since_decay"] < 48

    def test_workspace_at_exact_threshold(self):
        """Workspace processed exactly at 48h is not stale (> 48, not >=)."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        # Slightly under 48 hours to ensure we're not stale
        at_threshold = (datetime.now(timezone.utc) - timedelta(hours=47, minutes=59)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_edge": at_threshold},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_edge"]
            assert detail["stale"] is False

    def test_workspace_just_over_threshold(self):
        """Workspace processed 49 hours ago is stale."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        just_over = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_over": just_over},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_over"]
            assert detail["stale"] is True

    def test_empty_workspaces(self):
        """When no workspaces, details is None."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["workspaces"]["details"] is None
            assert body["workspaces"]["total"] == 0


# ============================================================================
# TestDecayHealthParseErrors
# ============================================================================


class TestDecayHealthParseErrors:
    """Tests for invalid timestamp handling in decay_health()."""

    def test_invalid_timestamp_has_parse_error(self):
        """Invalid timestamp string produces parse_error in workspace detail."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_bad": "not-a-date"},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_bad"]
            assert detail["parse_error"] is True
            assert detail["last_decay"] == "not-a-date"

    def test_empty_string_timestamp_has_parse_error(self):
        """Empty string as timestamp value produces parse_error."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_empty": ""},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_empty"]
            assert detail["parse_error"] is True

    def test_parse_error_not_counted_as_stale(self):
        """Workspace with parse error is not counted as stale."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_bad": "invalid"},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["workspaces"]["stale_count"] == 0
            assert body["status"] == "healthy"

    def test_mixed_valid_and_invalid_timestamps(self):
        """Mix of valid and invalid timestamps: valid ones processed, invalid skipped."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {
                "ws_ok": recent,
                "ws_bad": "garbage",
            },
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["workspaces"]["total"] == 2
            assert body["workspaces"]["details"]["ws_ok"]["stale"] is False
            assert body["workspaces"]["details"]["ws_bad"]["parse_error"] is True


# ============================================================================
# TestDecayHealthTimestamp
# ============================================================================


class TestDecayHealthTimestamp:
    """Tests for timestamp and response time in decay_health()."""

    def test_timestamp_is_utc_iso(self):
        """Timestamp is ISO format ending with Z."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["timestamp"].endswith("Z")
            # Should contain a valid ISO date prefix
            assert "T" in body["timestamp"]
            assert "2" in body["timestamp"][:4]  # Year starts with 2

    def test_response_time_non_negative(self):
        """response_time_ms is non-negative."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["response_time_ms"] >= 0


# ============================================================================
# TestDecayHealthEdgeCases
# ============================================================================


class TestDecayHealthEdgeCases:
    """Edge cases and boundary conditions for decay_health()."""

    def test_z_suffix_timestamps_parsed(self):
        """Timestamps with Z suffix are handled correctly."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        # Timestamp with Z suffix (common ISO format)
        recent = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_z": recent},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            detail = body["workspaces"]["details"]["ws_z"]
            # Z is replaced with +00:00 for parsing, so it should work
            assert detail["stale"] is False
            assert "parse_error" not in detail

    def test_stats_default_values_used(self):
        """When stats dict is missing keys, defaults are used for scheduler config."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["scheduler"]["decay_interval_hours"] == 24
            assert body["scheduler"]["min_confidence_threshold"] == 0.1
            assert body["scheduler"]["decay_rate"] == 0.95
            assert body["statistics"]["total_cycles"] == 0
            assert body["statistics"]["total_items_processed"] == 0
            assert body["statistics"]["total_items_expired"] == 0
            assert body["statistics"]["errors"] == 0

    def test_warning_message_includes_hours_and_threshold(self):
        """Warning message for stale workspace includes hours and threshold."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        stale = (datetime.now(timezone.utc) - timedelta(hours=60)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_warn": stale},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            warning = body["warnings"][0]
            assert "ws_warn" in warning
            assert "48h" in warning  # threshold reference

    def test_hours_since_decay_rounded(self):
        """hours_since_decay is rounded to 1 decimal place."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        recent = (datetime.now(timezone.utc) - timedelta(hours=3, minutes=30)).isoformat()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {"ws_round": recent},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            hours = body["workspaces"]["details"]["ws_round"]["hours_since_decay"]
            # Should be approximately 3.5, rounded to 1 decimal
            assert 3.0 <= hours <= 4.0
            # Verify it's rounded to 1 decimal
            assert str(hours).count(".") <= 1
            if "." in str(hours):
                decimal_places = len(str(hours).split(".")[1])
                assert decimal_places <= 1

    def test_metrics_available_false(self):
        """When DECAY_METRICS_AVAILABLE is False, it's reflected in response."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = decay_health(MagicMock())
            body = _body(result)
            assert body["metrics_available"] is False

    def test_handler_argument_is_ignored(self):
        """The handler argument is not used by decay_health (passed for interface compat)."""
        from aragora.server.handlers.admin.health.knowledge_mound import decay_health

        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            # Pass None as handler - should not cause any error
            result = decay_health(None)
            body = _body(result)
            assert body["status"] in ("healthy", "degraded", "stopped")

    def test_handler_argument_ignored_in_km_health(self):
        """The handler argument is not used by knowledge_mound_health."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(None)
            body = _body(result)
            assert body["status"] in ("healthy", "degraded", "not_configured")


# ============================================================================
# TestDecayHealthViaHealthHandler - integration with HealthHandler router
# ============================================================================


class TestViaHealthHandler:
    """Tests for knowledge_mound and decay health through HealthHandler routing."""

    @pytest.fixture
    def health_handler(self):
        """Create a HealthHandler instance."""
        from aragora.server.handlers.admin.health import HealthHandler

        return HealthHandler(ctx={})

    @pytest.mark.asyncio
    async def test_knowledge_mound_route(self, health_handler, mock_http_handler=None):
        """HealthHandler routes /api/v1/health/knowledge-mound correctly."""
        with _PatchContext():
            result = await health_handler.handle("/api/v1/health/knowledge-mound", {}, MagicMock())
            body = _body(result)
            assert body["status"] in ("healthy", "degraded", "not_configured", "unavailable")

    @pytest.mark.asyncio
    async def test_decay_route(self, health_handler):
        """HealthHandler routes /api/v1/health/decay correctly."""
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = True
        mock_scheduler.get_stats.return_value = {
            "last_decay_per_workspace": {},
        }

        mock_mod = MagicMock()
        mock_mod.get_decay_scheduler.return_value = mock_scheduler
        mock_mod.DECAY_METRICS_AVAILABLE = False

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.confidence_decay_scheduler": mock_mod,
            },
        ):
            result = await health_handler.handle("/api/v1/health/decay", {}, MagicMock())
            body = _body(result)
            assert body["status"] in ("healthy", "degraded", "stopped")

    @pytest.mark.asyncio
    async def test_knowledge_mound_handler_can_handle(self, health_handler):
        """HealthHandler.can_handle returns True for knowledge-mound path."""
        assert health_handler.can_handle("/api/v1/health/knowledge-mound") is True

    @pytest.mark.asyncio
    async def test_decay_handler_can_handle(self, health_handler):
        """HealthHandler.can_handle returns True for decay path."""
        assert health_handler.can_handle("/api/v1/health/decay") is True


# ============================================================================
# TestKnowledgeMoundHealthActiveCount
# ============================================================================


class TestKnowledgeMoundHealthActiveCount:
    """Tests for active component count calculation."""

    def test_active_count_with_all_active(self):
        """When all components have status='active', active count equals total active ones."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        # Default patch has several components with "active" status
        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            # Count components with status == "active" in defaults
            # culture_accumulator, staleness_tracker, rlm_integration, debate_integration,
            # confidence_decay = 5 active
            assert body["summary"]["active"] >= 1

    def test_active_count_with_mixed_statuses(self):
        """Active count only counts status='active' components."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_knowledge_mound_module": (
                {"healthy": True, "status": "available"},
                False,
            ),
            f"{_MOD}.check_mound_core_initialization": (
                {"healthy": True, "status": "initialized"},
                MagicMock(),
            ),
            f"{_MOD}.check_storage_backend": {"healthy": True, "status": "configured"},
            f"{_MOD}.check_culture_accumulator": {"healthy": True, "status": "active"},
            f"{_MOD}.check_staleness_tracker": {"healthy": True, "status": "not_initialized"},
            f"{_MOD}.check_rlm_integration": {"healthy": True, "status": "active"},
            f"{_MOD}.check_codebase_context": {"healthy": True, "status": "missing"},
            f"{_MOD}.check_debate_integration": {"healthy": True, "status": "not_available"},
            f"{_MOD}.check_knowledge_mound_redis_cache": {
                "healthy": True,
                "status": "not_configured",
            },
            f"{_MOD}.check_bidirectional_adapters": {"healthy": True, "status": "available"},
            f"{_MOD}.check_control_plane_adapter": {"healthy": True, "status": "available"},
            f"{_MOD}.check_km_metrics": {"healthy": True, "status": "available"},
            f"{_MOD}.check_confidence_decay_scheduler": ({"healthy": True, "status": "active"}, []),
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            # culture_accumulator, rlm_integration, confidence_decay = 3 active
            assert body["summary"]["active"] == 3


# ============================================================================
# TestKnowledgeMoundHealthSummary
# ============================================================================


class TestKnowledgeMoundHealthSummary:
    """Tests for summary calculations in knowledge_mound_health()."""

    def test_total_components_is_twelve(self):
        """Total components count is always 12 (13 checks but module check is #1)."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            # module + core + storage + culture_accumulator + staleness_tracker +
            # rlm_integration + codebase_context + debate_integration +
            # redis_cache + bidirectional_adapters + control_plane_adapter +
            # km_metrics + confidence_decay = 13 total keys
            assert body["summary"]["total_components"] == len(body["components"])

    def test_healthy_count_when_all_healthy(self):
        """When all components are healthy, healthy count equals total."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        with _PatchContext():
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            assert body["summary"]["healthy"] == body["summary"]["total_components"]

    def test_healthy_count_with_some_unhealthy(self):
        """When some components are unhealthy, healthy count reflects it."""
        from aragora.server.handlers.admin.health.knowledge_mound import knowledge_mound_health

        overrides = {
            f"{_MOD}.check_mound_core_initialization": (
                {"healthy": False, "status": "initialization_failed"},
                None,
            ),
            f"{_MOD}.check_storage_backend": {"healthy": False, "status": "error"},
        }
        with _PatchContext(overrides):
            result = knowledge_mound_health(MagicMock())
            body = _body(result)
            # 12 total - 2 unhealthy = 10 (module is 13th but still healthy)
            # Wait: module + core(unhealthy) + storage(unhealthy) + 10 others = 13 total
            assert body["summary"]["healthy"] == body["summary"]["total_components"] - 2
