"""
Tests for aragora.server.handlers.admin.nomic_admin - Nomic Admin Loop Control Endpoints.

Comprehensive tests covering:
- NomicAdminMixin._get_nomic_dir() with various context values
- GET /api/v1/admin/nomic/status - all branches
- GET /api/v1/admin/nomic/circuit-breakers - all branches
- POST /api/v1/admin/nomic/reset - phase validation, state file I/O, metrics, audit
- POST /api/v1/admin/nomic/pause - with/without reason, state recording
- POST /api/v1/admin/nomic/resume - paused/not-paused, target phase, cleanup
- POST /api/v1/admin/nomic/circuit-breakers/reset - reset, metrics, module unavailable
- RBAC/auth gate tests for every endpoint
- Error handling (file I/O errors, corrupt JSON, import failures)
- Integration flows (pause+resume, reset during pause, status during pause)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.nomic_admin import (
    NomicAdminMixin,
    PERM_ADMIN_NOMIC_WRITE,
    PERM_ADMIN_SYSTEM_WRITE,
)
from aragora.server.handlers.utils.responses import HandlerResult, error_response


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


# ===========================================================================
# Mock classes
# ===========================================================================


class MockAuthContext:
    """Minimal mock auth context for the mixin tests."""

    def __init__(self, user_id: str = "admin-001", org_id: str = "org-001"):
        self.user_id = user_id
        self.org_id = org_id


class MockHTTPHandler:
    """Mock HTTP handler providing request_body and standard fields."""

    def __init__(
        self,
        body: bytes = b"{}",
        path: str = "/api/v1/admin/nomic/status",
        method: str = "GET",
    ):
        self.request_body = body
        self.path = path
        self.command = method
        self.headers = {"Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)


class MockRegistry:
    """Mock CircuitBreakerRegistry with controllable state."""

    def __init__(
        self,
        open_circuits: list[str] | None = None,
        breaker_count: int = 0,
    ):
        self._open_circuits = open_circuits or []
        self._breakers: dict[str, Any] = {f"b{i}": True for i in range(breaker_count)}

    def all_open(self) -> list[str]:
        return list(self._open_circuits)

    def to_dict(self) -> dict:
        return {"breakers": {n: {"open": True} for n in self._open_circuits}}

    def reset_all(self) -> None:
        self._open_circuits.clear()


class TestableHandler(NomicAdminMixin):
    """Concrete class wiring the mixin to controllable auth stubs."""

    def __init__(
        self,
        ctx: dict[str, Any] | None = None,
        admin_result: tuple[MockAuthContext | None, HandlerResult | None] | None = None,
        rbac_result: HandlerResult | None = None,
    ):
        self.ctx = ctx or {"nomic_dir": ".nomic"}
        self._admin_result = admin_result or (MockAuthContext(), None)
        self._rbac_result = rbac_result

    def _require_admin(self, handler: Any) -> tuple[MockAuthContext | None, HandlerResult | None]:
        return self._admin_result

    def _check_rbac_permission(
        self, auth_ctx: Any, permission: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        return self._rbac_result


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler(tmp_path):
    """Handler with a tmp_path-backed nomic dir."""
    nomic_dir = tmp_path / "nomic"
    nomic_dir.mkdir()
    return TestableHandler(ctx={"nomic_dir": str(nomic_dir)})


@pytest.fixture
def http(request):
    """Factory for MockHTTPHandler; call http(body=...) or just http()."""

    def _make(body: bytes = b"{}", path: str = "/api/v1/admin/nomic/status", method: str = "GET"):
        return MockHTTPHandler(body=body, path=path, method=method)

    return _make


# ===========================================================================
# Permission Constants
# ===========================================================================


class TestPermissionConstants:
    def test_nomic_write_value(self):
        assert PERM_ADMIN_NOMIC_WRITE == "admin:nomic:write"

    def test_system_write_value(self):
        assert PERM_ADMIN_SYSTEM_WRITE == "admin:system:write"


# ===========================================================================
# _get_nomic_dir
# ===========================================================================


class TestGetNomicDir:
    def test_from_context(self):
        h = TestableHandler(ctx={"nomic_dir": "/opt/nomic"})
        assert h._get_nomic_dir() == "/opt/nomic"

    def test_default_when_missing(self):
        h = TestableHandler(ctx={})
        assert h._get_nomic_dir() == ".nomic"

    def test_default_when_none(self):
        h = TestableHandler(ctx={"nomic_dir": None})
        assert h._get_nomic_dir() == ".nomic"

    def test_default_when_empty_string(self):
        h = TestableHandler(ctx={"nomic_dir": ""})
        assert h._get_nomic_dir() == ".nomic"

    def test_path_object_converted_to_string(self, tmp_path):
        h = TestableHandler(ctx={"nomic_dir": tmp_path / "nomic"})
        assert isinstance(h._get_nomic_dir(), str)


# ===========================================================================
# GET /api/v1/admin/nomic/status
# ===========================================================================


class TestGetNomicStatus:
    """Tests for _get_nomic_status."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._get_nomic_status(http())
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._get_nomic_status(http())
        assert _status(result) == 403

    def test_no_state_file(self, handler, http, tmp_path):
        # nomic dir exists but no state file
        result = handler._get_nomic_status(http())
        assert _status(result) == 200
        data = _body(result)
        assert data["running"] is False
        assert data["current_phase"] is None
        assert data["cycle_id"] is None
        assert data["state_machine"] is None

    def test_with_valid_state_file(self, handler, http, tmp_path):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = {"running": True, "phase": "design", "cycle_id": "c-42"}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = handler._get_nomic_status(http())
        data = _body(result)
        assert data["running"] is True
        assert data["current_phase"] == "design"
        assert data["cycle_id"] == "c-42"
        assert data["state_machine"] == state

    def test_corrupted_state_file(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text("{bad json")

        result = handler._get_nomic_status(http())
        data = _body(result)
        assert _status(result) == 200
        assert any("Failed to read state" in e for e in data["errors"])

    def test_metrics_available(self, handler, http):
        mock_summary = {"total_cycles": 5}
        mock_stuck = {"stuck": False}
        with (
            patch("aragora.nomic.metrics.get_nomic_metrics_summary", return_value=mock_summary),
            patch("aragora.nomic.metrics.check_stuck_phases", return_value=mock_stuck),
        ):
            result = handler._get_nomic_status(http())

        data = _body(result)
        assert data["metrics"] == mock_summary
        assert data["stuck_detection"] == mock_stuck

    def test_metrics_import_error(self, handler, http):
        with patch.dict("sys.modules", {"aragora.nomic.metrics": None}):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert _status(result) == 200
        assert any("Metrics module not available" in e for e in data["errors"])

    def test_metrics_runtime_error(self, handler, http):
        with patch(
            "aragora.nomic.metrics.get_nomic_metrics_summary",
            side_effect=ValueError("bad"),
        ):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert any("Failed to get metrics" in e for e in data["errors"])

    def test_circuit_breakers_included(self, handler, http):
        registry = MockRegistry(open_circuits=["verify"])
        with patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert data["circuit_breakers"]["open"] == ["verify"]

    def test_circuit_breakers_import_error(self, handler, http):
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=ImportError("no module"),
        ):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert any("Failed to get circuit breakers" in e for e in data["errors"])

    def test_checkpoints_available(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "checkpoints").mkdir()
        cps = [{"id": "cp1"}, {"id": "cp2"}]
        with patch("aragora.nomic.checkpoints.list_checkpoints", return_value=cps):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert data["last_checkpoint"] == {"id": "cp1"}

    def test_checkpoints_empty(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "checkpoints").mkdir()
        with patch("aragora.nomic.checkpoints.list_checkpoints", return_value=[]):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert data["last_checkpoint"] is None

    def test_checkpoints_dir_missing(self, handler, http):
        result = handler._get_nomic_status(http())
        data = _body(result)
        # No error for missing checkpoint dir -- it just stays None
        assert data["last_checkpoint"] is None

    def test_checkpoints_import_error(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "checkpoints").mkdir()
        with patch(
            "aragora.nomic.checkpoints.list_checkpoints",
            side_effect=ImportError("no"),
        ):
            result = handler._get_nomic_status(http())
        data = _body(result)
        assert any("Failed to list checkpoints" in e for e in data["errors"])


# ===========================================================================
# GET /api/v1/admin/nomic/circuit-breakers
# ===========================================================================


class TestGetCircuitBreakers:
    def test_requires_admin(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._get_nomic_circuit_breakers(http())
        assert _status(result) == 401

    def test_requires_rbac(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._get_nomic_circuit_breakers(http())
        assert _status(result) == 403

    def test_success_with_open_circuits(self, handler, http):
        registry = MockRegistry(open_circuits=["a", "b"], breaker_count=3)
        with patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry):
            result = handler._get_nomic_circuit_breakers(http())
        assert _status(result) == 200
        data = _body(result)
        assert data["open_circuits"] == ["a", "b"]
        assert data["total_count"] == 3

    def test_success_no_open_circuits(self, handler, http):
        registry = MockRegistry(open_circuits=[], breaker_count=2)
        with patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry):
            result = handler._get_nomic_circuit_breakers(http())
        data = _body(result)
        assert data["open_circuits"] == []
        assert data["total_count"] == 2

    def test_module_unavailable(self, handler, http):
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=ImportError("no module"),
        ):
            result = handler._get_nomic_circuit_breakers(http())
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    def test_runtime_error_returns_500(self, handler, http):
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=ValueError("unexpected"),
        ):
            result = handler._get_nomic_circuit_breakers(http())
        assert _status(result) == 500

    def test_type_error_returns_500(self, handler, http):
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=TypeError("bad type"),
        ):
            result = handler._get_nomic_circuit_breakers(http())
        assert _status(result) == 500


# ===========================================================================
# POST /api/v1/admin/nomic/reset
# ===========================================================================


class TestResetNomicPhase:
    def test_requires_admin(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._reset_nomic_phase(http(body=b"{}"))
        assert _status(result) == 401

    def test_requires_rbac(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._reset_nomic_phase(http(body=b"{}"))
        assert _status(result) == 403

    def test_invalid_json_body(self, handler, http):
        result = handler._reset_nomic_phase(http(body=b"not json"))
        assert _status(result) == 400
        assert "Invalid JSON body" in _body(result)["error"]

    def test_invalid_target_phase(self, handler, http):
        result = handler._reset_nomic_phase(
            http(body=json.dumps({"target_phase": "nope"}).encode())
        )
        assert _status(result) == 400
        assert "Invalid target phase" in _body(result)["error"]

    @pytest.mark.parametrize(
        "phase", ["idle", "context", "debate", "design", "implement", "verify", "commit"]
    )
    def test_all_valid_phases(self, handler, http, phase):
        h = http(body=json.dumps({"target_phase": phase}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(h)
        assert _status(result) == 200
        assert _body(result)["new_phase"] == phase

    def test_defaults_to_idle(self, handler, http):
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(http(body=b""))
        assert _body(result)["new_phase"] == "idle"

    def test_case_insensitive(self, handler, http):
        h = http(body=json.dumps({"target_phase": "DEBATE"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(h)
        assert _body(result)["new_phase"] == "debate"

    def test_idle_sets_running_false(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(h)
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["running"] is False

    def test_non_idle_sets_running_true(self, handler, http):
        h = http(body=json.dumps({"target_phase": "verify"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(h)
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["running"] is True

    def test_preserves_errors_by_default(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(json.dumps({"phase": "x", "errors": ["e1"]}))
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._reset_nomic_phase(h)
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["errors"] == ["e1"]

    def test_clear_errors(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "x", "errors": ["e1", "e2"]})
        )
        h = http(body=json.dumps({"target_phase": "idle", "clear_errors": True}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._reset_nomic_phase(h)
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["errors"] == []

    def test_records_previous_phase_and_cycle_id(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "debate", "cycle_id": "c-99"})
        )
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(h)
        data = _body(result)
        assert data["previous_phase"] == "debate"
        assert data["cycle_id"] == "c-99"

    def test_creates_nomic_dir_if_missing(self, tmp_path, http):
        nomic_dir = tmp_path / "fresh_nomic"
        h = TestableHandler(ctx={"nomic_dir": str(nomic_dir)})
        req = http(body=json.dumps({"target_phase": "context"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = h._reset_nomic_phase(req)
        assert _status(result) == 200
        assert nomic_dir.exists()

    def test_tracks_metrics(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(json.dumps({"phase": "debate"}))
        h = http(body=json.dumps({"target_phase": "design"}).encode())
        with (
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
            patch("aragora.nomic.metrics.track_phase_transition") as mock_track,
        ):
            handler._reset_nomic_phase(h)
            mock_track.assert_called_once()
            kw = mock_track.call_args[1]
            assert kw["from_phase"] == "debate"
            assert kw["to_phase"] == "design"

    def test_metrics_import_error_still_succeeds(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with (
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
            patch.dict("sys.modules", {"aragora.nomic.metrics": None}),
        ):
            result = handler._reset_nomic_phase(h)
        assert _status(result) == 200

    def test_audit_called(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle", "reason": "test"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin") as mock_audit:
            handler._reset_nomic_phase(h)
            mock_audit.assert_called_once()
            kw = mock_audit.call_args[1]
            assert kw["action"] == "reset_nomic_phase"
            assert kw["admin_id"] == "admin-001"
            assert kw["reason"] == "test"

    def test_state_write_failure(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("builtins.open", side_effect=PermissionError("nope")):
            result = handler._reset_nomic_phase(h)
        assert _status(result) == 500

    def test_corrupted_existing_state_still_resets(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text("bad{json")
        h = http(body=json.dumps({"target_phase": "context"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(h)
        # Should still succeed with empty current_state
        assert _status(result) == 200

    def test_custom_reason_in_state(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle", "reason": "Hotfix"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._reset_nomic_phase(h)
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["reset_reason"] == "Hotfix"

    def test_default_reason(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._reset_nomic_phase(h)
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["reset_reason"] == "Admin manual reset"

    def test_reset_by_user_id_stored(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._reset_nomic_phase(h)
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["reset_by"] == "admin-001"

    def test_last_update_populated(self, handler, http):
        h = http(body=json.dumps({"target_phase": "idle"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._reset_nomic_phase(h)
        nomic_dir = Path(handler.ctx["nomic_dir"])
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert "last_update" in state
        assert state["last_update"].endswith("Z")


# ===========================================================================
# POST /api/v1/admin/nomic/pause
# ===========================================================================


class TestPauseNomic:
    def test_requires_admin(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._pause_nomic(http())
        assert _status(result) == 401

    def test_requires_rbac(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._pause_nomic(http())
        assert _status(result) == 403

    def test_success_with_reason(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "debate", "running": True, "cycle_id": "c1"})
        )
        h = http(body=json.dumps({"reason": "Maintenance window"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._pause_nomic(h)
        data = _body(result)
        assert data["success"] is True
        assert data["status"] == "paused"
        assert data["previous_phase"] == "debate"
        assert data["reason"] == "Maintenance window"

    def test_default_reason(self, handler, http):
        h = http(body=b"{}")
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._pause_nomic(h)
        assert _body(result)["reason"] == "Admin requested pause"

    def test_invalid_json_uses_default_reason(self, handler, http):
        h = http(body=b"not json {")
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._pause_nomic(h)
        assert _status(result) == 200
        assert _body(result)["reason"] == "Admin requested pause"

    def test_state_file_written(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "implement", "cycle_id": "c2"})
        )
        h = http(body=json.dumps({"reason": "Testing"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._pause_nomic(h)
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["phase"] == "paused"
        assert state["running"] is False
        assert state["previous_phase"] == "implement"
        assert state["paused_by"] == "admin-001"
        assert state["pause_reason"] == "Testing"
        assert "paused_at" in state
        assert "last_update" in state

    def test_creates_directory(self, tmp_path, http):
        nomic_dir = tmp_path / "new_nomic"
        h = TestableHandler(ctx={"nomic_dir": str(nomic_dir)})
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = h._pause_nomic(http(body=b"{}"))
        assert _status(result) == 200
        assert nomic_dir.exists()

    def test_pause_from_no_state_file(self, handler, http):
        h = http(body=b"{}")
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._pause_nomic(h)
        assert _status(result) == 200
        data = _body(result)
        assert data["previous_phase"] is None

    def test_write_error_returns_500(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(json.dumps({"phase": "debate"}))
        sf = nomic_dir / "nomic_state.json"
        sf.chmod(0o444)
        try:
            with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
                result = handler._pause_nomic(http(body=b"{}"))
            assert _status(result) == 500
        finally:
            sf.chmod(0o644)

    def test_audit_called(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "debate", "cycle_id": "c5"})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin") as mock_audit:
            handler._pause_nomic(http(body=json.dumps({"reason": "R"}).encode()))
            mock_audit.assert_called_once()
            kw = mock_audit.call_args[1]
            assert kw["action"] == "pause_nomic"
            assert kw["target_id"] == "c5"
            assert kw["reason"] == "R"

    def test_preserves_existing_state_fields(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "design", "cycle_id": "c7", "extra_field": "kept"})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._pause_nomic(http(body=b"{}"))
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        # Spread operator preserves extra fields
        assert state["cycle_id"] == "c7"
        assert state.get("extra_field") == "kept"


# ===========================================================================
# POST /api/v1/admin/nomic/resume
# ===========================================================================


class TestResumeNomic:
    def test_requires_admin(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._resume_nomic(http())
        assert _status(result) == 401

    def test_requires_rbac(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._resume_nomic(http())
        assert _status(result) == 403

    def test_not_paused_returns_400(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "debate", "running": True})
        )
        result = handler._resume_nomic(http())
        assert _status(result) == 400
        assert "not currently paused" in _body(result)["error"]

    def test_no_state_file_returns_400(self, handler, http):
        result = handler._resume_nomic(http())
        assert _status(result) == 400

    def test_success_resumes_to_previous_phase(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "paused", "previous_phase": "design", "cycle_id": "c1"})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._resume_nomic(http())
        data = _body(result)
        assert data["success"] is True
        assert data["status"] == "resumed"
        assert data["phase"] == "design"

    def test_target_phase_overrides_previous(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "paused", "previous_phase": "design"})
        )
        h = http(body=json.dumps({"target_phase": "implement"}).encode())
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._resume_nomic(h)
        assert _body(result)["phase"] == "implement"

    def test_defaults_to_context_when_no_previous(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(json.dumps({"phase": "paused"}))
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._resume_nomic(http())
        assert _body(result)["phase"] == "context"

    def test_clears_pause_fields(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps(
                {
                    "phase": "paused",
                    "previous_phase": "context",
                    "paused_at": "2024-01-01T00:00:00Z",
                    "paused_by": "admin-001",
                    "pause_reason": "test",
                }
            )
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._resume_nomic(http())
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert "paused_at" not in state
        assert "paused_by" not in state
        assert "pause_reason" not in state
        assert state["running"] is True
        assert "resumed_at" in state
        assert "resumed_by" in state

    def test_write_error_returns_500(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(json.dumps({"phase": "paused"}))
        original_open = open

        def mock_open_fn(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if "w" in mode:
                raise PermissionError("denied")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_fn):
            result = handler._resume_nomic(http())
        assert _status(result) == 500

    def test_invalid_json_body_uses_empty_data(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "paused", "previous_phase": "debate"})
        )
        h = http(body=b"bad json {")
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._resume_nomic(h)
        assert _status(result) == 200
        assert _body(result)["phase"] == "debate"

    def test_audit_called(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "paused", "previous_phase": "verify", "cycle_id": "c9"})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin") as mock_audit:
            handler._resume_nomic(http())
            mock_audit.assert_called_once()
            kw = mock_audit.call_args[1]
            assert kw["action"] == "resume_nomic"
            assert kw["resume_phase"] == "verify"

    def test_resumed_by_user_id(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "paused", "previous_phase": "context"})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._resume_nomic(http())
        assert _body(result)["resumed_by"] == "admin-001"

    def test_corrupted_state_file_returns_400(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text("bad{json")
        result = handler._resume_nomic(http())
        # Corrupted state means empty dict => phase != "paused" => 400
        assert _status(result) == 400


# ===========================================================================
# POST /api/v1/admin/nomic/circuit-breakers/reset
# ===========================================================================


class TestResetCircuitBreakers:
    def test_requires_admin(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._reset_nomic_circuit_breakers(http())
        assert _status(result) == 401

    def test_requires_system_write_rbac(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._reset_nomic_circuit_breakers(http())
        assert _status(result) == 403

    def test_success_with_open_breakers(self, handler, http):
        registry = MockRegistry(open_circuits=["cb1", "cb2"])
        with (
            patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry),
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
        ):
            result = handler._reset_nomic_circuit_breakers(http())
        data = _body(result)
        assert data["success"] is True
        assert data["previously_open"] == ["cb1", "cb2"]
        assert "reset" in data["message"].lower()

    def test_success_no_open_breakers(self, handler, http):
        registry = MockRegistry(open_circuits=[])
        with (
            patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry),
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
        ):
            result = handler._reset_nomic_circuit_breakers(http())
        data = _body(result)
        assert data["success"] is True
        assert data["previously_open"] == []

    def test_module_unavailable(self, handler, http):
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=ImportError("no module"),
        ):
            result = handler._reset_nomic_circuit_breakers(http())
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    def test_runtime_error_returns_500(self, handler, http):
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=RuntimeError("boom"),
        ):
            result = handler._reset_nomic_circuit_breakers(http())
        assert _status(result) == 500

    def test_updates_metrics(self, handler, http):
        registry = MockRegistry(open_circuits=["cb1"])
        with (
            patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry),
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
            patch("aragora.nomic.metrics.update_circuit_breaker_count") as mock_update,
        ):
            handler._reset_nomic_circuit_breakers(http())
            mock_update.assert_called_once_with(0)

    def test_metrics_import_error_still_succeeds(self, handler, http):
        registry = MockRegistry(open_circuits=[])
        with (
            patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry),
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
            patch(
                "aragora.nomic.metrics.update_circuit_breaker_count",
                side_effect=ImportError("no metrics"),
            ),
        ):
            result = handler._reset_nomic_circuit_breakers(http())
        assert _status(result) == 200

    def test_metrics_runtime_error_still_succeeds(self, handler, http):
        registry = MockRegistry(open_circuits=[])
        with (
            patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry),
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin"),
            patch(
                "aragora.nomic.metrics.update_circuit_breaker_count",
                side_effect=ValueError("bad"),
            ),
        ):
            result = handler._reset_nomic_circuit_breakers(http())
        assert _status(result) == 200

    def test_audit_called(self, handler, http):
        registry = MockRegistry(open_circuits=["x"])
        with (
            patch("aragora.nomic.recovery.CircuitBreakerRegistry", return_value=registry),
            patch("aragora.server.handlers.admin.nomic_admin.audit_admin") as mock_audit,
        ):
            handler._reset_nomic_circuit_breakers(http())
            mock_audit.assert_called_once()
            kw = mock_audit.call_args[1]
            assert kw["action"] == "reset_circuit_breakers"
            assert kw["previously_open"] == ["x"]


# ===========================================================================
# Integration flows
# ===========================================================================


class TestIntegrationFlows:
    def test_pause_then_resume(self, handler, http):
        """Full pause -> resume flow preserves phase."""
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "debate", "running": True, "cycle_id": "c1"})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            # Pause
            pause_r = handler._pause_nomic(
                http(body=json.dumps({"reason": "integration"}).encode())
            )
            assert _status(pause_r) == 200
            state = json.loads((nomic_dir / "nomic_state.json").read_text())
            assert state["phase"] == "paused"

            # Resume
            resume_r = handler._resume_nomic(http(body=b"{}"))
            assert _status(resume_r) == 200
            state = json.loads((nomic_dir / "nomic_state.json").read_text())
            assert state["phase"] == "debate"
            assert state["running"] is True
            assert "paused_at" not in state

    def test_reset_while_paused(self, handler, http):
        """Reset to a new phase while paused."""
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "paused", "previous_phase": "implement", "running": False})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = handler._reset_nomic_phase(
                http(body=json.dumps({"target_phase": "context"}).encode())
            )
        data = _body(result)
        assert data["previous_phase"] == "paused"
        assert data["new_phase"] == "context"
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["running"] is True

    def test_status_while_paused(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps(
                {
                    "phase": "paused",
                    "previous_phase": "verify",
                    "running": False,
                    "paused_by": "admin-001",
                }
            )
        )
        result = handler._get_nomic_status(http())
        data = _body(result)
        assert data["running"] is False
        assert data["current_phase"] == "paused"
        assert data["state_machine"]["previous_phase"] == "verify"

    def test_double_pause_overwrites(self, handler, http):
        nomic_dir = Path(handler.ctx["nomic_dir"])
        (nomic_dir / "nomic_state.json").write_text(
            json.dumps({"phase": "debate", "running": True})
        )
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            handler._pause_nomic(http(body=json.dumps({"reason": "first"}).encode()))
            handler._pause_nomic(http(body=json.dumps({"reason": "second"}).encode()))
        state = json.loads((nomic_dir / "nomic_state.json").read_text())
        assert state["pause_reason"] == "second"
        assert state["phase"] == "paused"

    def test_reset_then_pause_then_resume(self, handler, http):
        """Complex flow: reset to design, pause, resume."""
        nomic_dir = Path(handler.ctx["nomic_dir"])
        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            # Reset to design
            handler._reset_nomic_phase(http(body=json.dumps({"target_phase": "design"}).encode()))
            state = json.loads((nomic_dir / "nomic_state.json").read_text())
            assert state["phase"] == "design"

            # Pause
            handler._pause_nomic(http(body=b"{}"))
            state = json.loads((nomic_dir / "nomic_state.json").read_text())
            assert state["phase"] == "paused"
            assert state["previous_phase"] == "design"

            # Resume
            handler._resume_nomic(http(body=b"{}"))
            state = json.loads((nomic_dir / "nomic_state.json").read_text())
            assert state["phase"] == "design"
            assert state["running"] is True


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from aragora.server.handlers.admin.nomic_admin import __all__

        assert "NomicAdminMixin" in __all__
        assert "PERM_ADMIN_NOMIC_WRITE" in __all__
        assert "PERM_ADMIN_SYSTEM_WRITE" in __all__

    def test_mixin_is_importable(self):
        assert NomicAdminMixin is not None

    def test_mixin_has_expected_methods(self):
        assert hasattr(NomicAdminMixin, "_get_nomic_status")
        assert hasattr(NomicAdminMixin, "_get_nomic_circuit_breakers")
        assert hasattr(NomicAdminMixin, "_reset_nomic_phase")
        assert hasattr(NomicAdminMixin, "_pause_nomic")
        assert hasattr(NomicAdminMixin, "_resume_nomic")
        assert hasattr(NomicAdminMixin, "_reset_nomic_circuit_breakers")
        assert hasattr(NomicAdminMixin, "_get_nomic_dir")
