"""
Tests for aragora.server.handlers.admin.nomic_admin - Nomic Admin Loop Control Endpoints.

Tests cover:
- NomicAdminMixin initialization and method delegation
- Get nomic status endpoint (state machine, metrics, circuit breakers)
- Get circuit breakers endpoint
- Reset nomic phase endpoint (validation, file operations, audit)
- Pause nomic loop endpoint
- Resume nomic loop endpoint
- Reset circuit breakers endpoint
- RBAC permission checks for all endpoints
- Error handling and edge cases
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aragora.server.handlers.admin.nomic_admin import (
    NomicAdminMixin,
    PERM_ADMIN_NOMIC_WRITE,
    PERM_ADMIN_SYSTEM_WRITE,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helper Functions
# ===========================================================================


def get_response_data(result: HandlerResult) -> dict:
    """Extract JSON data from HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockAuthContext:
    """Mock authentication context for testing."""

    def __init__(
        self,
        user_id: str = "admin-001",
        is_authenticated: bool = True,
        org_id: str = "org-001",
    ):
        self.user_id = user_id
        self.is_authenticated = is_authenticated
        self.org_id = org_id


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
    ):
        self.headers = headers or {}
        self._body = body
        self.request_body = body
        self.path = path
        self.command = method
        self.client_address = ("127.0.0.1", 12345)


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, name: str = "test-breaker", is_open: bool = False):
        self.name = name
        self._is_open = is_open

    def is_open(self) -> bool:
        return self._is_open


class MockCircuitBreakerRegistry:
    """Mock circuit breaker registry for testing."""

    def __init__(self, open_circuits: list[str] | None = None):
        self._breakers: dict[str, MockCircuitBreaker] = {}
        self._open_circuits = open_circuits or []

    def all_open(self) -> list[str]:
        return self._open_circuits

    def to_dict(self) -> dict:
        return {
            "breakers": {
                name: {"open": name in self._open_circuits} for name in self._open_circuits
            }
        }

    def reset_all(self) -> None:
        self._open_circuits = []


class TestableNomicAdminHandler(NomicAdminMixin):
    """Testable class that uses NomicAdminMixin with mock dependencies."""

    def __init__(
        self,
        ctx: dict[str, Any] | None = None,
        require_admin_result: tuple[MockAuthContext | None, HandlerResult | None] | None = None,
        check_rbac_result: HandlerResult | None = None,
    ):
        self.ctx = ctx or {"nomic_dir": ".nomic"}
        self._require_admin_result = require_admin_result or (MockAuthContext(), None)
        self._check_rbac_result = check_rbac_result

    def _require_admin(self, handler: Any) -> tuple[MockAuthContext | None, HandlerResult | None]:
        return self._require_admin_result

    def _check_rbac_permission(
        self, auth_ctx: Any, permission: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        return self._check_rbac_result


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_server_context() -> dict:
    """Create mock server context."""
    return {
        "nomic_dir": ".nomic",
    }


@pytest.fixture
def nomic_handler(mock_server_context) -> TestableNomicAdminHandler:
    """Create TestableNomicAdminHandler instance."""
    return TestableNomicAdminHandler(ctx=mock_server_context)


@pytest.fixture
def mock_http_handler() -> MockHandler:
    """Create mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json"},
        path="/api/v1/admin/nomic/status",
        method="GET",
    )


@pytest.fixture
def mock_auth_context() -> MockAuthContext:
    """Create mock auth context."""
    return MockAuthContext()


# ===========================================================================
# Permission Constants Tests
# ===========================================================================


class TestPermissionConstants:
    """Tests for permission constants."""

    def test_admin_nomic_write_permission(self):
        """Test PERM_ADMIN_NOMIC_WRITE is correctly defined."""
        assert PERM_ADMIN_NOMIC_WRITE == "admin:nomic:write"

    def test_admin_system_write_permission(self):
        """Test PERM_ADMIN_SYSTEM_WRITE is correctly defined."""
        assert PERM_ADMIN_SYSTEM_WRITE == "admin:system:write"


# ===========================================================================
# Get Nomic Dir Tests
# ===========================================================================


class TestGetNomicDir:
    """Tests for _get_nomic_dir method."""

    def test_get_nomic_dir_from_context(self, nomic_handler):
        """Test getting nomic dir from context."""
        nomic_handler.ctx["nomic_dir"] = "/custom/nomic"
        result = nomic_handler._get_nomic_dir()
        assert result == "/custom/nomic"

    def test_get_nomic_dir_default(self):
        """Test default nomic dir when not in context."""
        handler = TestableNomicAdminHandler(ctx={})
        result = handler._get_nomic_dir()
        assert result == ".nomic"

    def test_get_nomic_dir_none_in_context(self):
        """Test default when context has None."""
        handler = TestableNomicAdminHandler(ctx={"nomic_dir": None})
        result = handler._get_nomic_dir()
        assert result == ".nomic"


# ===========================================================================
# Get Nomic Status Tests
# ===========================================================================


class TestGetNomicStatus:
    """Tests for _get_nomic_status endpoint."""

    def test_get_status_requires_admin(self, mock_http_handler):
        """Test get status requires admin auth."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Unauthorized", 401)
        handler = TestableNomicAdminHandler(
            require_admin_result=(None, err),
        )
        result = handler._get_nomic_status(mock_http_handler)

        assert result.status_code == 401

    def test_get_status_requires_rbac_permission(self, mock_http_handler):
        """Test get status requires RBAC permission."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Permission denied", 403)
        handler = TestableNomicAdminHandler(check_rbac_result=err)

        result = handler._get_nomic_status(mock_http_handler)

        assert result.status_code == 403

    def test_get_status_no_state_file(self, nomic_handler, mock_http_handler, tmp_path):
        """Test get status when state file doesn't exist."""
        nomic_handler.ctx["nomic_dir"] = str(tmp_path / "nomic")

        result = nomic_handler._get_nomic_status(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["running"] is False
        assert data["current_phase"] is None
        assert data["cycle_id"] is None

    def test_get_status_with_state_file(self, nomic_handler, mock_http_handler, tmp_path):
        """Test get status when state file exists."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_data = {
            "running": True,
            "phase": "debate",
            "cycle_id": "cycle-001",
        }
        state_file.write_text(json.dumps(state_data))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        result = nomic_handler._get_nomic_status(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["running"] is True
        assert data["current_phase"] == "debate"
        assert data["cycle_id"] == "cycle-001"
        assert data["state_machine"] == state_data

    def test_get_status_with_corrupted_state_file(self, nomic_handler, mock_http_handler, tmp_path):
        """Test get status when state file is corrupted."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text("not valid json {{{")
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        result = nomic_handler._get_nomic_status(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert len(data["errors"]) > 0
        assert "Failed to read state" in data["errors"][0]

    def test_get_status_metrics_import_error(self, nomic_handler, mock_http_handler, tmp_path):
        """Test get status handles metrics import error."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch.dict("sys.modules", {"aragora.nomic.metrics": None}):
            result = nomic_handler._get_nomic_status(mock_http_handler)

            assert result.status_code == 200
            data = get_response_data(result)
            # Errors may or may not include import error depending on implementation

    def test_get_status_with_checkpoints(self, nomic_handler, mock_http_handler, tmp_path):
        """Test get status includes checkpoints when available."""
        nomic_dir = tmp_path / "nomic"
        checkpoint_dir = nomic_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        mock_checkpoints = [{"id": "cp-001", "phase": "debate"}]
        with patch("aragora.nomic.checkpoints.list_checkpoints", return_value=mock_checkpoints):
            result = nomic_handler._get_nomic_status(mock_http_handler)

            assert result.status_code == 200
            data = get_response_data(result)
            assert data["last_checkpoint"] == mock_checkpoints[0]

    def test_get_status_with_circuit_breakers(self, nomic_handler, mock_http_handler, tmp_path):
        """Test get status includes circuit breaker info."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        mock_registry = MockCircuitBreakerRegistry(open_circuits=["phase-verify"])
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            return_value=mock_registry,
        ):
            result = nomic_handler._get_nomic_status(mock_http_handler)

            assert result.status_code == 200
            data = get_response_data(result)
            assert data["circuit_breakers"]["open"] == ["phase-verify"]


# ===========================================================================
# Get Circuit Breakers Tests
# ===========================================================================


class TestGetNomicCircuitBreakers:
    """Tests for _get_nomic_circuit_breakers endpoint."""

    def test_get_circuit_breakers_requires_admin(self, mock_http_handler):
        """Test get circuit breakers requires admin auth."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Unauthorized", 401)
        handler = TestableNomicAdminHandler(
            require_admin_result=(None, err),
        )
        result = handler._get_nomic_circuit_breakers(mock_http_handler)

        assert result.status_code == 401

    def test_get_circuit_breakers_requires_rbac_permission(self, mock_http_handler):
        """Test get circuit breakers requires RBAC permission."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Permission denied", 403)
        handler = TestableNomicAdminHandler(check_rbac_result=err)

        result = handler._get_nomic_circuit_breakers(mock_http_handler)

        assert result.status_code == 403

    def test_get_circuit_breakers_success(self, nomic_handler, mock_http_handler):
        """Test get circuit breakers returns data."""
        mock_registry = MockCircuitBreakerRegistry(open_circuits=["phase-1", "phase-2"])
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            return_value=mock_registry,
        ):
            result = nomic_handler._get_nomic_circuit_breakers(mock_http_handler)

            assert result.status_code == 200
            data = get_response_data(result)
            assert "circuit_breakers" in data
            assert data["open_circuits"] == ["phase-1", "phase-2"]
            assert data["total_count"] == 0  # _breakers is empty dict in mock

    def test_get_circuit_breakers_module_unavailable(self, nomic_handler, mock_http_handler):
        """Test get circuit breakers when recovery module unavailable."""
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=ImportError("Module not found"),
        ):
            result = nomic_handler._get_nomic_circuit_breakers(mock_http_handler)

            assert result.status_code == 503
            data = get_response_data(result)
            assert "not available" in data["error"]

    def test_get_circuit_breakers_handles_exception(self, nomic_handler, mock_http_handler):
        """Test get circuit breakers handles unexpected exceptions."""
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = nomic_handler._get_nomic_circuit_breakers(mock_http_handler)

            assert result.status_code == 500
            data = get_response_data(result)
            assert data["error"]  # Sanitized error message present


# ===========================================================================
# Reset Nomic Phase Tests
# ===========================================================================


class TestResetNomicPhase:
    """Tests for _reset_nomic_phase endpoint."""

    def test_reset_phase_requires_admin(self, mock_http_handler):
        """Test reset phase requires admin auth."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Unauthorized", 401)
        handler = TestableNomicAdminHandler(
            require_admin_result=(None, err),
        )
        result = handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 401

    def test_reset_phase_requires_rbac_permission(self, mock_http_handler):
        """Test reset phase requires RBAC permission."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Permission denied", 403)
        handler = TestableNomicAdminHandler(check_rbac_result=err)

        result = handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 403

    def test_reset_phase_invalid_json_body(self, nomic_handler, mock_http_handler):
        """Test reset phase with invalid JSON body."""
        mock_http_handler.request_body = b"not valid json"

        result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 400
        data = get_response_data(result)
        assert "Invalid JSON body" in data["error"]

    def test_reset_phase_invalid_target_phase(self, nomic_handler, mock_http_handler):
        """Test reset phase with invalid target phase."""
        mock_http_handler.request_body = json.dumps({"target_phase": "invalid_phase"}).encode()

        result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 400
        data = get_response_data(result)
        assert "Invalid target phase" in data["error"]

    def test_reset_phase_to_idle(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase to idle."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps(
            {"target_phase": "idle", "reason": "Test reset"}
        ).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["success"] is True
        assert data["new_phase"] == "idle"

        # Verify state file was written
        state_file = nomic_dir / "nomic_state.json"
        state_data = json.loads(state_file.read_text())
        assert state_data["phase"] == "idle"
        assert state_data["running"] is False

    def test_reset_phase_to_debate(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase to debate."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "debate"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["new_phase"] == "debate"

        state_file = nomic_dir / "nomic_state.json"
        state_data = json.loads(state_file.read_text())
        assert state_data["running"] is True  # Non-idle phases are running

    @pytest.mark.parametrize(
        "valid_phase",
        ["idle", "context", "debate", "design", "implement", "verify", "commit"],
    )
    def test_reset_phase_all_valid_phases(
        self, nomic_handler, mock_http_handler, tmp_path, valid_phase
    ):
        """Test reset phase accepts all valid phase values."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": valid_phase}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["new_phase"] == valid_phase

    def test_reset_phase_clears_errors(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase clears errors when requested."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debug", "errors": ["error1", "error2"]}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps(
            {"target_phase": "idle", "clear_errors": True}
        ).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        state_data = json.loads(state_file.read_text())
        assert state_data["errors"] == []

    def test_reset_phase_preserves_errors(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase preserves errors by default."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debug", "errors": ["error1", "error2"]}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "idle"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        state_data = json.loads(state_file.read_text())
        assert state_data["errors"] == ["error1", "error2"]

    def test_reset_phase_records_previous_phase(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase records previous phase."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debate", "cycle_id": "cycle-001"}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "idle"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["previous_phase"] == "debate"
        assert data["cycle_id"] == "cycle-001"

    def test_reset_phase_creates_directory(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase creates nomic directory if needed."""
        nomic_dir = tmp_path / "new_nomic"
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "context"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        assert nomic_dir.exists()

    def test_reset_phase_tracks_metrics(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase tracks metrics."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debate"}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "design"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            with patch("aragora.nomic.metrics.track_phase_transition") as mock_track:
                result = nomic_handler._reset_nomic_phase(mock_http_handler)

                assert result.status_code == 200
                mock_track.assert_called_once()
                call_kwargs = mock_track.call_args[1]
                assert call_kwargs["from_phase"] == "debate"
                assert call_kwargs["to_phase"] == "design"

    def test_reset_phase_empty_body(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase with empty body defaults to idle."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = b""

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["new_phase"] == "idle"

    def test_reset_phase_case_insensitive(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase is case insensitive."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "DEBATE"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["new_phase"] == "debate"


# ===========================================================================
# Pause Nomic Tests
# ===========================================================================


class TestPauseNomic:
    """Tests for _pause_nomic endpoint."""

    def test_pause_requires_admin(self, mock_http_handler):
        """Test pause requires admin auth."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Unauthorized", 401)
        handler = TestableNomicAdminHandler(
            require_admin_result=(None, err),
        )
        result = handler._pause_nomic(mock_http_handler)

        assert result.status_code == 401

    def test_pause_requires_rbac_permission(self, mock_http_handler):
        """Test pause requires RBAC permission."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Permission denied", 403)
        handler = TestableNomicAdminHandler(check_rbac_result=err)

        result = handler._pause_nomic(mock_http_handler)

        assert result.status_code == 403

    def test_pause_nomic_success(self, nomic_handler, mock_http_handler, tmp_path):
        """Test pause nomic successfully."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps({"phase": "debate", "running": True, "cycle_id": "cycle-001"})
        )
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"reason": "Maintenance"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._pause_nomic(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["success"] is True
        assert data["status"] == "paused"
        assert data["previous_phase"] == "debate"
        assert data["reason"] == "Maintenance"

    def test_pause_nomic_records_state(self, nomic_handler, mock_http_handler, tmp_path):
        """Test pause nomic records state correctly."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "implement"}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"reason": "Test"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._pause_nomic(mock_http_handler)

        assert result.status_code == 200

        state_data = json.loads(state_file.read_text())
        assert state_data["phase"] == "paused"
        assert state_data["running"] is False
        assert state_data["previous_phase"] == "implement"
        assert state_data["paused_by"] == "admin-001"
        assert state_data["pause_reason"] == "Test"
        assert "paused_at" in state_data

    def test_pause_nomic_default_reason(self, nomic_handler, mock_http_handler, tmp_path):
        """Test pause nomic uses default reason."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = b"{}"

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._pause_nomic(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["reason"] == "Admin requested pause"

    def test_pause_nomic_creates_directory(self, nomic_handler, mock_http_handler, tmp_path):
        """Test pause nomic creates directory if needed."""
        nomic_dir = tmp_path / "new_nomic"
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._pause_nomic(mock_http_handler)

        assert result.status_code == 200
        assert nomic_dir.exists()

    def test_pause_nomic_invalid_json_body(self, nomic_handler, mock_http_handler, tmp_path):
        """Test pause nomic handles invalid JSON gracefully."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = b"invalid json {"

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._pause_nomic(mock_http_handler)

        # Should still succeed with default reason
        assert result.status_code == 200
        data = get_response_data(result)
        assert data["reason"] == "Admin requested pause"


# ===========================================================================
# Resume Nomic Tests
# ===========================================================================


class TestResumeNomic:
    """Tests for _resume_nomic endpoint."""

    def test_resume_requires_admin(self, mock_http_handler):
        """Test resume requires admin auth."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Unauthorized", 401)
        handler = TestableNomicAdminHandler(
            require_admin_result=(None, err),
        )
        result = handler._resume_nomic(mock_http_handler)

        assert result.status_code == 401

    def test_resume_requires_rbac_permission(self, mock_http_handler):
        """Test resume requires RBAC permission."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Permission denied", 403)
        handler = TestableNomicAdminHandler(check_rbac_result=err)

        result = handler._resume_nomic(mock_http_handler)

        assert result.status_code == 403

    def test_resume_nomic_not_paused(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume nomic fails when not paused."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debate", "running": True}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 400
        data = get_response_data(result)
        assert "not currently paused" in data["error"]

    def test_resume_nomic_success(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume nomic successfully."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "phase": "paused",
                    "running": False,
                    "previous_phase": "design",
                    "paused_by": "admin-001",
                    "cycle_id": "cycle-001",
                }
            )
        )
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["success"] is True
        assert data["status"] == "resumed"
        assert data["phase"] == "design"  # Resumes to previous phase

    def test_resume_nomic_with_target_phase(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume nomic with custom target phase."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "paused", "previous_phase": "design"}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "implement"}).encode()

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["phase"] == "implement"

    def test_resume_nomic_clears_pause_fields(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume nomic clears pause-related fields."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "phase": "paused",
                    "previous_phase": "context",
                    "paused_at": "2024-01-01T00:00:00Z",
                    "paused_by": "admin-001",
                    "pause_reason": "Test",
                }
            )
        )
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 200

        state_data = json.loads(state_file.read_text())
        assert "paused_at" not in state_data
        assert "paused_by" not in state_data
        assert "pause_reason" not in state_data
        assert state_data["running"] is True
        assert "resumed_by" in state_data
        assert "resumed_at" in state_data

    def test_resume_nomic_default_to_context(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume nomic defaults to context when no previous phase."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "paused"}))  # No previous_phase
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["phase"] == "context"

    def test_resume_nomic_no_state_file(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume nomic when state file doesn't exist."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 400
        data = get_response_data(result)
        assert "not currently paused" in data["error"]


# ===========================================================================
# Reset Circuit Breakers Tests
# ===========================================================================


class TestResetNomicCircuitBreakers:
    """Tests for _reset_nomic_circuit_breakers endpoint."""

    def test_reset_circuit_breakers_requires_admin(self, mock_http_handler):
        """Test reset circuit breakers requires admin auth."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Unauthorized", 401)
        handler = TestableNomicAdminHandler(
            require_admin_result=(None, err),
        )
        result = handler._reset_nomic_circuit_breakers(mock_http_handler)

        assert result.status_code == 401

    def test_reset_circuit_breakers_requires_system_write(self, mock_http_handler):
        """Test reset circuit breakers requires admin:system:write permission."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Permission denied", 403)
        handler = TestableNomicAdminHandler(check_rbac_result=err)

        result = handler._reset_nomic_circuit_breakers(mock_http_handler)

        assert result.status_code == 403

    def test_reset_circuit_breakers_success(self, nomic_handler, mock_http_handler):
        """Test reset circuit breakers successfully."""
        mock_registry = MockCircuitBreakerRegistry(open_circuits=["phase-1", "phase-2"])
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            return_value=mock_registry,
        ):
            with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
                result = nomic_handler._reset_nomic_circuit_breakers(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["success"] is True
        assert data["previously_open"] == ["phase-1", "phase-2"]
        assert "All circuit breakers have been reset" in data["message"]

    def test_reset_circuit_breakers_module_unavailable(self, nomic_handler, mock_http_handler):
        """Test reset circuit breakers when recovery module unavailable."""
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=ImportError("Module not found"),
        ):
            result = nomic_handler._reset_nomic_circuit_breakers(mock_http_handler)

            assert result.status_code == 503
            data = get_response_data(result)
            assert "not available" in data["error"]

    def test_reset_circuit_breakers_handles_exception(self, nomic_handler, mock_http_handler):
        """Test reset circuit breakers handles unexpected exceptions."""
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = nomic_handler._reset_nomic_circuit_breakers(mock_http_handler)

            assert result.status_code == 500
            data = get_response_data(result)
            assert data["error"]  # Sanitized error message present

    def test_reset_circuit_breakers_updates_metrics(self, nomic_handler, mock_http_handler):
        """Test reset circuit breakers updates metrics."""
        mock_registry = MockCircuitBreakerRegistry(open_circuits=["phase-1"])
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            return_value=mock_registry,
        ):
            with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
                with patch("aragora.nomic.metrics.update_circuit_breaker_count") as mock_update:
                    result = nomic_handler._reset_nomic_circuit_breakers(mock_http_handler)

                    assert result.status_code == 200
                    mock_update.assert_called_once_with(0)

    def test_reset_circuit_breakers_metrics_import_error(self, nomic_handler, mock_http_handler):
        """Test reset circuit breakers handles metrics import error gracefully."""
        mock_registry = MockCircuitBreakerRegistry(open_circuits=[])
        with patch(
            "aragora.nomic.recovery.CircuitBreakerRegistry",
            return_value=mock_registry,
        ):
            with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
                with patch(
                    "aragora.nomic.metrics.update_circuit_breaker_count",
                    side_effect=ImportError("No metrics"),
                ):
                    result = nomic_handler._reset_nomic_circuit_breakers(mock_http_handler)

                    # Should still succeed
                    assert result.status_code == 200


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestNomicAdminIntegration:
    """Integration tests for nomic admin flow."""

    def test_pause_and_resume_flow(self, nomic_handler, mock_http_handler, tmp_path):
        """Test complete pause and resume flow."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps({"phase": "debate", "running": True, "cycle_id": "cycle-001"})
        )
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            # Step 1: Pause
            mock_http_handler.request_body = json.dumps({"reason": "Integration test"}).encode()
            pause_result = nomic_handler._pause_nomic(mock_http_handler)
            assert pause_result.status_code == 200

            # Verify paused state
            state_data = json.loads(state_file.read_text())
            assert state_data["phase"] == "paused"
            assert state_data["previous_phase"] == "debate"

            # Step 2: Resume
            mock_http_handler.request_body = b"{}"
            resume_result = nomic_handler._resume_nomic(mock_http_handler)
            assert resume_result.status_code == 200

            # Verify resumed state
            state_data = json.loads(state_file.read_text())
            assert state_data["phase"] == "debate"
            assert state_data["running"] is True
            assert "paused_at" not in state_data

    def test_reset_during_pause(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resetting phase while paused."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps({"phase": "paused", "previous_phase": "implement", "running": False})
        )
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
            # Reset to context while paused
            mock_http_handler.request_body = json.dumps({"target_phase": "context"}).encode()
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

            assert result.status_code == 200
            data = get_response_data(result)
            assert data["previous_phase"] == "paused"
            assert data["new_phase"] == "context"

            state_data = json.loads(state_file.read_text())
            assert state_data["running"] is True

    def test_status_during_pause(self, nomic_handler, mock_http_handler, tmp_path):
        """Test getting status while paused."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "phase": "paused",
                    "previous_phase": "verify",
                    "running": False,
                    "paused_at": "2024-01-01T00:00:00Z",
                    "paused_by": "admin-001",
                    "pause_reason": "Testing",
                }
            )
        )
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        result = nomic_handler._get_nomic_status(mock_http_handler)

        assert result.status_code == 200
        data = get_response_data(result)
        assert data["running"] is False
        assert data["current_phase"] == "paused"
        assert data["state_machine"]["previous_phase"] == "verify"


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_file_write_error_reset(self, nomic_handler, mock_http_handler, tmp_path):
        """Test reset phase handles file write errors."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)
        mock_http_handler.request_body = json.dumps({"target_phase": "idle"}).encode()

        with patch("builtins.open", side_effect=PermissionError("No write permission")):
            result = nomic_handler._reset_nomic_phase(mock_http_handler)

        assert result.status_code == 500
        data = get_response_data(result)
        assert data["error"]  # Sanitized error message present

    def test_file_write_error_pause(self, nomic_handler, mock_http_handler, tmp_path):
        """Test pause handles file write errors."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        # Create existing state file so the read succeeds
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "debate"}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        # Make the state file read-only to trigger write error
        state_file.chmod(0o444)

        try:
            with patch("aragora.server.handlers.admin.nomic_admin.audit_admin"):
                result = nomic_handler._pause_nomic(mock_http_handler)

            assert result.status_code == 500
            data = get_response_data(result)
            assert data["error"]  # Sanitized error message present
        finally:
            # Restore permissions for cleanup
            state_file.chmod(0o644)

    def test_file_write_error_resume(self, nomic_handler, mock_http_handler, tmp_path):
        """Test resume handles file write errors."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        state_file = nomic_dir / "nomic_state.json"
        state_file.write_text(json.dumps({"phase": "paused"}))
        nomic_handler.ctx["nomic_dir"] = str(nomic_dir)

        # Create a mock that allows read but fails on write
        original_open = open
        call_count = [0]

        def mock_open_side_effect(path, *args, **kwargs):
            call_count[0] += 1
            mode = args[0] if args else kwargs.get("mode", "r")
            if "w" in mode:
                raise PermissionError("No write permission")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_side_effect):
            result = nomic_handler._resume_nomic(mock_http_handler)

        assert result.status_code == 500
        data = get_response_data(result)
        assert data["error"]  # Sanitized error message present


# ===========================================================================
# Module Exports Test
# ===========================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_nomic_admin_mixin(self):
        """Test NomicAdminMixin is exported."""
        from aragora.server.handlers.admin.nomic_admin import NomicAdminMixin

        assert NomicAdminMixin is not None

    def test_exports_permission_constants(self):
        """Test permission constants are exported."""
        from aragora.server.handlers.admin.nomic_admin import (
            PERM_ADMIN_NOMIC_WRITE,
            PERM_ADMIN_SYSTEM_WRITE,
        )

        assert PERM_ADMIN_NOMIC_WRITE is not None
        assert PERM_ADMIN_SYSTEM_WRITE is not None


__all__ = [
    "TestPermissionConstants",
    "TestGetNomicDir",
    "TestGetNomicStatus",
    "TestGetNomicCircuitBreakers",
    "TestResetNomicPhase",
    "TestPauseNomic",
    "TestResumeNomic",
    "TestResetNomicCircuitBreakers",
    "TestNomicAdminIntegration",
    "TestErrorHandling",
    "TestModuleExports",
]
