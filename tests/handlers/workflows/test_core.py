"""Tests for workflow handler core utilities (aragora/server/handlers/workflows/core.py).

Covers all public functions, classes, constants, and edge cases:
- _step_result_to_dict: StepResult serialization with datetime, enums, None fields
- _get_store: persistent store retrieval with override, fallback, and error paths
- _call_store_method: sync and async store method dispatching
- _get_engine: lazy engine initialization with override, fallback, caching
- _TemplateStore: in-memory template storage
- Module-level constants: RBAC_AVAILABLE, METRICS_AVAILABLE
- Fallback stubs: record_rbac_check, track_handler when metrics unavailable
- __all__ exports
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

from aragora.server.handlers.workflows.core import (
    _step_result_to_dict,
    _get_store,
    _call_store_method,
    _get_engine,
    _store,
    _TemplateStore,
    _UnauthenticatedSentinel,
    RBAC_AVAILABLE,
    METRICS_AVAILABLE,
    logger,
    __all__ as CORE_ALL,
)
from aragora.workflow.types import StepResult, StepStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step_result(
    step_id: str = "step-1",
    step_name: str = "analysis",
    status: StepStatus = StepStatus.COMPLETED,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    duration_ms: float = 123.4,
    output: Any = None,
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
    retry_count: int = 0,
) -> StepResult:
    return StepResult(
        step_id=step_id,
        step_name=step_name,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        output=output,
        error=error,
        metrics=metrics or {},
        retry_count=retry_count,
    )


# ===========================================================================
# _step_result_to_dict
# ===========================================================================


class TestStepResultToDict:
    """Tests for _step_result_to_dict serialization helper."""

    def test_basic_completed_step(self):
        """Completed step serializes with correct fields."""
        sr = _make_step_result()
        d = _step_result_to_dict(sr)
        assert d["step_id"] == "step-1"
        assert d["step_name"] == "analysis"
        assert d["status"] == "completed"
        assert d["duration_ms"] == 123.4
        assert d["retry_count"] == 0
        assert d["output"] is None
        assert d["error"] is None
        assert d["metrics"] == {}

    def test_datetime_iso_formatting(self):
        """Datetime fields are converted to ISO format strings."""
        now = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        later = datetime(2025, 6, 15, 12, 30, 5, tzinfo=timezone.utc)
        sr = _make_step_result(started_at=now, completed_at=later)
        d = _step_result_to_dict(sr)
        assert d["started_at"] == "2025-06-15T12:30:00+00:00"
        assert d["completed_at"] == "2025-06-15T12:30:05+00:00"

    def test_none_datetimes(self):
        """None datetimes serialize to None, not crash."""
        sr = _make_step_result(started_at=None, completed_at=None)
        d = _step_result_to_dict(sr)
        assert d["started_at"] is None
        assert d["completed_at"] is None

    def test_status_enum_value(self):
        """StepStatus enum uses .value for serialization."""
        for status in StepStatus:
            sr = _make_step_result(status=status)
            d = _step_result_to_dict(sr)
            assert d["status"] == status.value

    def test_status_without_value_attribute(self):
        """Status without .value falls back to str()."""
        sr = _make_step_result()
        # Replace status with a plain string (no .value attribute)
        sr.status = "custom_status"  # type: ignore[assignment]
        d = _step_result_to_dict(sr)
        assert d["status"] == "custom_status"

    def test_output_dict(self):
        """Output containing a dict is preserved."""
        sr = _make_step_result(output={"key": "value", "count": 42})
        d = _step_result_to_dict(sr)
        assert d["output"] == {"key": "value", "count": 42}

    def test_output_list(self):
        """Output containing a list is preserved."""
        sr = _make_step_result(output=[1, 2, 3])
        d = _step_result_to_dict(sr)
        assert d["output"] == [1, 2, 3]

    def test_output_string(self):
        """Output containing a string is preserved."""
        sr = _make_step_result(output="some text result")
        d = _step_result_to_dict(sr)
        assert d["output"] == "some text result"

    def test_error_message(self):
        """Error string is preserved in the dict."""
        sr = _make_step_result(error="timeout exceeded")
        d = _step_result_to_dict(sr)
        assert d["error"] == "timeout exceeded"

    def test_metrics_dict(self):
        """Metrics dictionary is preserved."""
        m = {"latency_ms": 50, "tokens": 1500}
        sr = _make_step_result(metrics=m)
        d = _step_result_to_dict(sr)
        assert d["metrics"] == m

    def test_retry_count_nonzero(self):
        """Non-zero retry count is captured."""
        sr = _make_step_result(retry_count=3)
        d = _step_result_to_dict(sr)
        assert d["retry_count"] == 3

    def test_failed_step(self):
        """Failed step serializes status and error correctly."""
        sr = _make_step_result(
            status=StepStatus.FAILED,
            error="assertion error in step",
            duration_ms=99.0,
        )
        d = _step_result_to_dict(sr)
        assert d["status"] == "failed"
        assert d["error"] == "assertion error in step"
        assert d["duration_ms"] == 99.0

    def test_pending_step(self):
        """Pending step with no timestamps or output."""
        sr = _make_step_result(
            status=StepStatus.PENDING,
            duration_ms=0.0,
        )
        d = _step_result_to_dict(sr)
        assert d["status"] == "pending"
        assert d["started_at"] is None
        assert d["completed_at"] is None
        assert d["duration_ms"] == 0.0

    def test_all_keys_present(self):
        """Result dict always contains the full set of expected keys."""
        sr = _make_step_result()
        d = _step_result_to_dict(sr)
        expected_keys = {
            "step_id",
            "step_name",
            "status",
            "started_at",
            "completed_at",
            "duration_ms",
            "output",
            "error",
            "metrics",
            "retry_count",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# _get_store
# ===========================================================================


class TestGetStore:
    """Tests for _get_store persistent store retrieval."""

    @patch("aragora.server.handlers.workflows.core.get_workflow_store")
    def test_default_returns_workflow_store(self, mock_gws):
        """Without override, returns result of get_workflow_store()."""
        sentinel = MagicMock(name="default_store")
        mock_gws.return_value = sentinel
        # Ensure no override is registered on the package
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        old = getattr(pkg, "_get_store", None) if pkg else None
        try:
            if pkg and hasattr(pkg, "_get_store"):
                delattr(pkg, "_get_store")
            result = _get_store()
            assert result is sentinel
        finally:
            if pkg and old is not None:
                pkg._get_store = old

    @patch("aragora.server.handlers.workflows.core.get_workflow_store")
    def test_override_on_package(self, mock_gws):
        """When package has _get_store override, uses it instead."""
        override_store = MagicMock(name="override_store")
        override_fn = MagicMock(return_value=override_store)

        pkg = sys.modules.get("aragora.server.handlers.workflows")
        old = getattr(pkg, "_get_store", None) if pkg else None
        try:
            if pkg is not None:
                pkg._get_store = override_fn
            result = _get_store()
            assert result is override_store
            override_fn.assert_called_once()
        finally:
            if pkg is not None:
                if old is not None:
                    pkg._get_store = old
                elif hasattr(pkg, "_get_store"):
                    delattr(pkg, "_get_store")

    @patch("aragora.server.handlers.workflows.core.get_workflow_store")
    def test_override_attribute_error_falls_through(self, mock_gws):
        """AttributeError from override lookup falls through to default."""
        fallback_store = MagicMock(name="fallback")
        mock_gws.return_value = fallback_store

        pkg = sys.modules.get("aragora.server.handlers.workflows")
        old = getattr(pkg, "_get_store", None) if pkg else None
        try:
            # Temporarily remove the package from sys.modules
            # so the getattr raises no exception but returns None
            if pkg and hasattr(pkg, "_get_store"):
                delattr(pkg, "_get_store")
            result = _get_store()
            assert result is fallback_store
        finally:
            if pkg and old is not None:
                pkg._get_store = old

    @patch("aragora.server.handlers.workflows.core.get_workflow_store")
    def test_package_not_in_sys_modules(self, mock_gws):
        """When workflows package not in sys.modules, falls back to default."""
        sentinel = MagicMock(name="default")
        mock_gws.return_value = sentinel
        key = "aragora.server.handlers.workflows"
        old_module = sys.modules.pop(key, None)
        try:
            result = _get_store()
            assert result is sentinel
        finally:
            if old_module is not None:
                sys.modules[key] = old_module


# ===========================================================================
# _call_store_method
# ===========================================================================


class TestCallStoreMethod:
    """Tests for _call_store_method sync/async dispatcher."""

    def test_sync_result_returned_directly(self):
        """Non-coroutine values are returned as-is."""
        assert _call_store_method(42) == 42
        assert _call_store_method("hello") == "hello"
        assert _call_store_method(None) is None
        assert _call_store_method([1, 2]) == [1, 2]

    def test_dict_returned_directly(self):
        """Dict value returned without awaiting."""
        d = {"workflows": [], "total": 0}
        assert _call_store_method(d) == d

    @patch("aragora.server.handlers.workflows.core._run_async")
    def test_coroutine_dispatched_via_run_async(self, mock_run):
        """Coroutine objects are dispatched to _run_async."""

        async def my_coro():
            return "async_result"

        coro = my_coro()
        mock_run.return_value = "async_result"
        result = _call_store_method(coro)
        mock_run.assert_called_once()
        assert result == "async_result"

    def test_mock_non_coroutine_not_dispatched(self):
        """MagicMock (non-coroutine) is returned directly."""
        m = MagicMock()
        result = _call_store_method(m)
        assert result is m

    def test_false_value_returned(self):
        """Falsy values (False, 0, empty string) returned correctly."""
        assert _call_store_method(False) is False
        assert _call_store_method(0) == 0
        assert _call_store_method("") == ""


# ===========================================================================
# _get_engine
# ===========================================================================


class TestGetEngine:
    """Tests for _get_engine lazy initialization."""

    def test_returns_workflow_engine(self):
        """Returns a WorkflowEngine instance."""
        from aragora.workflow.engine import WorkflowEngine

        engine = _get_engine()
        assert isinstance(engine, WorkflowEngine)

    def test_returns_same_instance_on_subsequent_calls(self):
        """Lazy singleton: subsequent calls return the same object."""
        e1 = _get_engine()
        e2 = _get_engine()
        assert e1 is e2

    @patch("aragora.server.handlers.workflows.core.WorkflowEngine")
    def test_creates_engine_when_none(self, mock_cls):
        """Creates new WorkflowEngine when _engine is None."""
        import aragora.server.handlers.workflows.core as core_mod

        old_engine = core_mod._engine
        try:
            core_mod._engine = None
            # Remove any package override
            pkg = sys.modules.get("aragora.server.handlers.workflows")
            old_pkg_engine = getattr(pkg, "_engine", None) if pkg else None
            if pkg and hasattr(pkg, "_engine"):
                # Ensure override won't interfere
                pkg._engine = None
            mock_instance = MagicMock(name="new_engine")
            mock_cls.return_value = mock_instance
            result = _get_engine()
            assert result is mock_instance
            mock_cls.assert_called_once()
        finally:
            core_mod._engine = old_engine
            if pkg and old_pkg_engine is not None:
                pkg._engine = old_pkg_engine

    def test_override_on_package(self):
        """When package has _engine override, uses it."""
        override_engine = MagicMock(name="override_engine")
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        import aragora.server.handlers.workflows.core as core_mod

        old_engine = core_mod._engine
        old_pkg_engine = getattr(pkg, "_engine", None) if pkg else None
        try:
            if pkg is not None:
                pkg._engine = override_engine
            # Make sure the override is different from core._engine
            core_mod._engine = MagicMock(name="core_engine")
            result = _get_engine()
            assert result is override_engine
        finally:
            core_mod._engine = old_engine
            if pkg is not None:
                if old_pkg_engine is not None:
                    pkg._engine = old_pkg_engine
                elif hasattr(pkg, "_engine"):
                    delattr(pkg, "_engine")

    def test_package_not_in_sys_modules_still_works(self):
        """When package missing from sys.modules, engine still created."""
        import aragora.server.handlers.workflows.core as core_mod

        key = "aragora.server.handlers.workflows"
        old_engine = core_mod._engine
        old_module = sys.modules.pop(key, None)
        try:
            core_mod._engine = None
            engine = _get_engine()
            from aragora.workflow.engine import WorkflowEngine

            assert isinstance(engine, WorkflowEngine)
        finally:
            if old_module is not None:
                sys.modules[key] = old_module
            core_mod._engine = old_engine


# ===========================================================================
# _TemplateStore
# ===========================================================================


class TestTemplateStore:
    """Tests for the in-memory _TemplateStore class."""

    def test_initial_state_empty(self):
        """Newly created store has no templates."""
        store = _TemplateStore()
        assert store.templates == {}

    def test_add_template(self):
        """Can add a template to the store."""
        store = _TemplateStore()
        mock_wf = MagicMock(name="template_wf")
        store.templates["tpl-1"] = mock_wf
        assert "tpl-1" in store.templates
        assert store.templates["tpl-1"] is mock_wf

    def test_overwrite_template(self):
        """Overwriting a template replaces the previous value."""
        store = _TemplateStore()
        old = MagicMock(name="old")
        new = MagicMock(name="new")
        store.templates["tpl-1"] = old
        store.templates["tpl-1"] = new
        assert store.templates["tpl-1"] is new

    def test_delete_template(self):
        """Can delete a template from the store."""
        store = _TemplateStore()
        store.templates["tpl-1"] = MagicMock()
        del store.templates["tpl-1"]
        assert "tpl-1" not in store.templates

    def test_multiple_templates(self):
        """Store handles multiple templates correctly."""
        store = _TemplateStore()
        for i in range(5):
            store.templates[f"tpl-{i}"] = MagicMock(name=f"wf-{i}")
        assert len(store.templates) == 5
        assert all(f"tpl-{i}" in store.templates for i in range(5))

    def test_module_level_store_is_template_store(self):
        """Module-level _store is an instance of _TemplateStore."""
        assert isinstance(_store, _TemplateStore)

    def test_templates_dict_type(self):
        """templates attribute is a plain dict."""
        store = _TemplateStore()
        assert isinstance(store.templates, dict)


# ===========================================================================
# Module-level constants and flags
# ===========================================================================


class TestModuleConstants:
    """Tests for module-level constants and availability flags."""

    def test_rbac_available_is_bool(self):
        """RBAC_AVAILABLE is a boolean."""
        assert isinstance(RBAC_AVAILABLE, bool)

    def test_metrics_available_is_bool(self):
        """METRICS_AVAILABLE is a boolean."""
        assert isinstance(METRICS_AVAILABLE, bool)

    def test_logger_exists(self):
        """Module logger is properly configured."""
        import logging

        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        """Logger has the correct module name."""
        assert logger.name == "aragora.server.handlers.workflows.core"

    def test_unauthenticated_sentinel_type(self):
        """_UnauthenticatedSentinel is a Literal type annotation."""
        # It's a typing construct, not a runtime value with instances
        assert _UnauthenticatedSentinel is not None

    def test_all_exports_exist(self):
        """Every name in __all__ is actually importable from the module."""
        import aragora.server.handlers.workflows.core as core_mod

        for name in CORE_ALL:
            assert hasattr(core_mod, name), f"{name} listed in __all__ but not found"

    def test_all_contains_key_items(self):
        """__all__ contains the most important exports."""
        required = {
            "logger",
            "_step_result_to_dict",
            "_get_store",
            "_call_store_method",
            "_get_engine",
            "_store",
            "RBAC_AVAILABLE",
            "METRICS_AVAILABLE",
            "WorkflowDefinition",
            "WorkflowCategory",
            "StepDefinition",
            "StepResult",
            "TransitionRule",
            "PersistentWorkflowStore",
            "_UnauthenticatedSentinel",
            "record_rbac_check",
            "track_handler",
            "audit_data",
            "_run_async",
        }
        assert required.issubset(set(CORE_ALL))


# ===========================================================================
# Metrics fallback stubs
# ===========================================================================


class TestMetricsFallbacks:
    """Tests for fallback stubs when metrics module is unavailable."""

    def test_record_rbac_check_callable(self):
        """record_rbac_check is callable regardless of metrics availability."""
        from aragora.server.handlers.workflows.core import record_rbac_check

        # Should not raise -- signature is (permission: str, granted: bool)
        record_rbac_check("workflows:read", True)

    def test_track_handler_as_decorator(self):
        """track_handler works as a decorator regardless of metrics availability."""
        from aragora.server.handlers.workflows.core import track_handler

        @track_handler("test_handler")
        def my_handler():
            return "result"

        assert my_handler() == "result"

    def test_record_rbac_check_with_kwargs(self):
        """record_rbac_check accepts keyword arguments."""
        from aragora.server.handlers.workflows.core import record_rbac_check

        # When metrics available: (permission, granted); when not: (*args, **kwargs)
        if METRICS_AVAILABLE:
            record_rbac_check(permission="workflows:write", granted=False)
        else:
            record_rbac_check(resource="x", action="y", allowed=False)

    def test_track_handler_preserves_function(self):
        """track_handler returns the original function when metrics unavailable."""
        from aragora.server.handlers.workflows.core import track_handler

        def original():
            return 42

        decorated = track_handler("label")(original)
        # If metrics unavailable, should be identity decorator
        # If available, should still be callable
        assert decorated() == 42


# ===========================================================================
# Re-exports
# ===========================================================================


class TestReExports:
    """Tests that re-exported types are accessible from the core module."""

    def test_workflow_definition_class(self):
        """WorkflowDefinition is re-exported from workflow.types."""
        from aragora.server.handlers.workflows.core import WorkflowDefinition
        from aragora.workflow.types import WorkflowDefinition as OrigWD

        assert WorkflowDefinition is OrigWD

    def test_workflow_category_enum(self):
        """WorkflowCategory is re-exported from workflow.types."""
        from aragora.server.handlers.workflows.core import WorkflowCategory
        from aragora.workflow.types import WorkflowCategory as OrigWC

        assert WorkflowCategory is OrigWC

    def test_step_definition_class(self):
        """StepDefinition is re-exported."""
        from aragora.server.handlers.workflows.core import StepDefinition
        from aragora.workflow.types import StepDefinition as OrigSD

        assert StepDefinition is OrigSD

    def test_step_result_class(self):
        """StepResult is re-exported."""
        from aragora.server.handlers.workflows.core import StepResult as CoreSR

        assert CoreSR is StepResult

    def test_transition_rule_class(self):
        """TransitionRule is re-exported."""
        from aragora.server.handlers.workflows.core import TransitionRule
        from aragora.workflow.types import TransitionRule as OrigTR

        assert TransitionRule is OrigTR

    def test_persistent_workflow_store_class(self):
        """PersistentWorkflowStore is re-exported."""
        from aragora.server.handlers.workflows.core import PersistentWorkflowStore
        from aragora.workflow.persistent_store import PersistentWorkflowStore as OrigPWS

        assert PersistentWorkflowStore is OrigPWS

    def test_audit_data_function(self):
        """audit_data is re-exported from aragora.audit.unified."""
        from aragora.server.handlers.workflows.core import audit_data
        from aragora.audit.unified import audit_data as OrigAD

        assert audit_data is OrigAD

    def test_run_async_function(self):
        """_run_async is re-exported from http_utils."""
        from aragora.server.handlers.workflows.core import _run_async

        assert callable(_run_async)
