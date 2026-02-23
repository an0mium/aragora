"""Tests for workspace_utils module.

Tests the shared utility functions for the workspace handler package:
- WorkspaceCircuitBreaker re-export
- _get_workspace_circuit_breaker: per-subsystem circuit breaker creation/retrieval
- get_workspace_circuit_breaker_status: status of all workspace circuit breakers
- _validate_workspace_id: workspace ID validation
- _validate_policy_id: policy ID validation
- _validate_user_id: user ID validation
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_workspace_circuit_breakers():
    """Reset the module-level circuit breaker registry between tests."""
    from aragora.server.handlers.workspace import workspace_utils

    with workspace_utils._workspace_circuit_breaker_lock:
        workspace_utils._workspace_circuit_breakers.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_circuit_breakers():
    """Ensure a clean circuit breaker registry for each test."""
    _reset_workspace_circuit_breakers()
    yield
    _reset_workspace_circuit_breakers()


# ===========================================================================
# _get_workspace_circuit_breaker tests
# ===========================================================================


class TestGetWorkspaceCircuitBreaker:
    """Tests for _get_workspace_circuit_breaker."""

    def test_creates_new_circuit_breaker_for_unknown_subsystem(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        cb = _get_workspace_circuit_breaker("storage")
        assert cb is not None
        assert cb.name == "workspace"

    def test_returns_same_instance_on_repeated_calls(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        cb1 = _get_workspace_circuit_breaker("storage")
        cb2 = _get_workspace_circuit_breaker("storage")
        assert cb1 is cb2

    def test_returns_different_instances_for_different_subsystems(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        cb_a = _get_workspace_circuit_breaker("storage")
        cb_b = _get_workspace_circuit_breaker("auth")
        assert cb_a is not cb_b

    def test_circuit_breaker_has_correct_failure_threshold(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        cb = _get_workspace_circuit_breaker("storage")
        assert cb.failure_threshold == 3

    def test_circuit_breaker_has_correct_half_open_max_calls(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        cb = _get_workspace_circuit_breaker("storage")
        assert cb.half_open_max_calls == 2

    def test_circuit_breaker_starts_closed(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        cb = _get_workspace_circuit_breaker("storage")
        assert cb.state == "closed"

    def test_thread_safety_concurrent_creation(self):
        """Verify thread-safe creation under concurrent access."""
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
        )

        results = []
        errors = []

        def worker():
            try:
                cb = _get_workspace_circuit_breaker("concurrent-sub")
                results.append(cb)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get the exact same instance
        assert all(r is results[0] for r in results)

    def test_multiple_subsystems_registered(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
            _workspace_circuit_breakers,
        )

        _get_workspace_circuit_breaker("sub-a")
        _get_workspace_circuit_breaker("sub-b")
        _get_workspace_circuit_breaker("sub-c")
        assert len(_workspace_circuit_breakers) == 3


# ===========================================================================
# get_workspace_circuit_breaker_status tests
# ===========================================================================


class TestGetWorkspaceCircuitBreakerStatus:
    """Tests for get_workspace_circuit_breaker_status."""

    def test_empty_when_no_breakers_created(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            get_workspace_circuit_breaker_status,
        )

        status = get_workspace_circuit_breaker_status()
        assert status == {}

    def test_returns_status_for_single_breaker(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
            get_workspace_circuit_breaker_status,
        )

        _get_workspace_circuit_breaker("storage")
        status = get_workspace_circuit_breaker_status()
        assert "storage" in status
        assert status["storage"]["state"] == "closed"
        assert status["storage"]["failure_count"] == 0

    def test_returns_status_for_multiple_breakers(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
            get_workspace_circuit_breaker_status,
        )

        _get_workspace_circuit_breaker("storage")
        _get_workspace_circuit_breaker("auth")
        status = get_workspace_circuit_breaker_status()
        assert len(status) == 2
        assert "storage" in status
        assert "auth" in status

    def test_reflects_failure_state(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
            get_workspace_circuit_breaker_status,
        )

        cb = _get_workspace_circuit_breaker("fragile")
        # Record enough failures to trip the breaker (threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        status = get_workspace_circuit_breaker_status()
        assert status["fragile"]["state"] == "open"
        assert status["fragile"]["failure_count"] == 3

    def test_status_includes_all_expected_keys(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
            get_workspace_circuit_breaker_status,
        )

        _get_workspace_circuit_breaker("metrics")
        status = get_workspace_circuit_breaker_status()
        keys = set(status["metrics"].keys())
        expected = {"state", "failure_count", "success_count", "failure_threshold",
                    "cooldown_seconds", "last_failure_time"}
        assert expected.issubset(keys)

    def test_status_after_success_resets_failure_count(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _get_workspace_circuit_breaker,
            get_workspace_circuit_breaker_status,
        )

        cb = _get_workspace_circuit_breaker("reset-test")
        cb.record_failure()
        cb.record_success()
        status = get_workspace_circuit_breaker_status()
        assert status["reset-test"]["failure_count"] == 0


# ===========================================================================
# WorkspaceCircuitBreaker re-export tests
# ===========================================================================


class TestWorkspaceCircuitBreakerExport:
    """Verify WorkspaceCircuitBreaker is properly re-exported."""

    def test_workspace_circuit_breaker_is_simple_circuit_breaker(self):
        from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker
        from aragora.server.handlers.workspace.workspace_utils import (
            WorkspaceCircuitBreaker,
        )

        assert WorkspaceCircuitBreaker is SimpleCircuitBreaker

    def test_workspace_circuit_breaker_instantiable(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            WorkspaceCircuitBreaker,
        )

        cb = WorkspaceCircuitBreaker("test-cb")
        assert cb.state == "closed"
        assert cb.can_proceed() is True


# ===========================================================================
# _validate_workspace_id tests
# ===========================================================================


class TestValidateWorkspaceId:
    """Tests for _validate_workspace_id."""

    def test_valid_simple_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("workspace-123")
        assert is_valid is True
        assert err is None

    def test_valid_underscore_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("my_workspace")
        assert is_valid is True
        assert err is None

    def test_valid_alphanumeric_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("abc123XYZ")
        assert is_valid is True
        assert err is None

    def test_empty_string_is_invalid(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("")
        assert is_valid is False
        assert err == "workspace_id is required"

    def test_path_traversal_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("../../etc/passwd")
        assert is_valid is False
        assert err is not None

    def test_spaces_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("invalid workspace")
        assert is_valid is False
        assert err is not None

    def test_special_characters_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("ws@#$%")
        assert is_valid is False
        assert err is not None

    def test_too_long_id_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        # SAFE_ID_PATTERN allows 1-64 characters
        long_id = "a" * 65
        is_valid, err = _validate_workspace_id(long_id)
        assert is_valid is False
        assert err is not None

    def test_max_length_id_accepted(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        # Exactly 64 characters should be valid
        max_id = "a" * 64
        is_valid, err = _validate_workspace_id(max_id)
        assert is_valid is True
        assert err is None

    def test_single_char_id_accepted(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("a")
        assert is_valid is True
        assert err is None

    def test_slash_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_workspace_id,
        )

        is_valid, err = _validate_workspace_id("ws/subdir")
        assert is_valid is False
        assert err is not None


# ===========================================================================
# _validate_policy_id tests
# ===========================================================================


class TestValidatePolicyId:
    """Tests for _validate_policy_id."""

    def test_valid_policy_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        is_valid, err = _validate_policy_id("policy-001")
        assert is_valid is True
        assert err is None

    def test_empty_string_is_invalid(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        is_valid, err = _validate_policy_id("")
        assert is_valid is False
        assert err == "policy_id is required"

    def test_path_traversal_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        is_valid, err = _validate_policy_id("../../../secret")
        assert is_valid is False
        assert err is not None

    def test_special_characters_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        is_valid, err = _validate_policy_id("pol!@#$")
        assert is_valid is False
        assert err is not None

    def test_too_long_policy_id_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        long_id = "p" * 65
        is_valid, err = _validate_policy_id(long_id)
        assert is_valid is False
        assert err is not None

    def test_max_length_policy_id_accepted(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        max_id = "p" * 64
        is_valid, err = _validate_policy_id(max_id)
        assert is_valid is True
        assert err is None

    def test_underscore_policy_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        is_valid, err = _validate_policy_id("retention_policy_99")
        assert is_valid is True
        assert err is None

    def test_numeric_only_policy_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_policy_id,
        )

        is_valid, err = _validate_policy_id("123456")
        assert is_valid is True
        assert err is None


# ===========================================================================
# _validate_user_id tests
# ===========================================================================


class TestValidateUserId:
    """Tests for _validate_user_id."""

    def test_valid_user_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("user-001")
        assert is_valid is True
        assert err is None

    def test_empty_string_is_invalid(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("")
        assert is_valid is False
        assert err == "user_id is required"

    def test_path_traversal_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("../../admin")
        assert is_valid is False
        assert err is not None

    def test_spaces_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("user name")
        assert is_valid is False
        assert err is not None

    def test_special_characters_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("user<script>")
        assert is_valid is False
        assert err is not None

    def test_too_long_user_id_rejected(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        long_id = "u" * 65
        is_valid, err = _validate_user_id(long_id)
        assert is_valid is False
        assert err is not None

    def test_max_length_user_id_accepted(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        max_id = "u" * 64
        is_valid, err = _validate_user_id(max_id)
        assert is_valid is True
        assert err is None

    def test_underscore_user_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("test_user_42")
        assert is_valid is True
        assert err is None

    def test_hyphen_user_id(self):
        from aragora.server.handlers.workspace.workspace_utils import (
            _validate_user_id,
        )

        is_valid, err = _validate_user_id("user-abc-xyz")
        assert is_valid is True
        assert err is None


# ===========================================================================
# __all__ export tests
# ===========================================================================


class TestModuleExports:
    """Verify __all__ contains the expected public API."""

    def test_all_exports(self):
        from aragora.server.handlers.workspace import workspace_utils

        expected = {
            "WorkspaceCircuitBreaker",
            "_get_workspace_circuit_breaker",
            "get_workspace_circuit_breaker_status",
            "_validate_workspace_id",
            "_validate_policy_id",
            "_validate_user_id",
        }
        assert set(workspace_utils.__all__) == expected


# ===========================================================================
# validate_path_segment delegation tests
# ===========================================================================


class TestValidationDelegation:
    """Verify that validation functions properly delegate to validate_path_segment."""

    def test_workspace_id_delegates_to_validate_path_segment(self):
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(True, None),
        ) as mock_vps:
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_workspace_id,
            )

            result = _validate_workspace_id("ws-1")
            mock_vps.assert_called_once_with("ws-1", "workspace_id")
            assert result == (True, None)

    def test_policy_id_delegates_to_validate_path_segment(self):
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(True, None),
        ) as mock_vps:
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_policy_id,
            )

            result = _validate_policy_id("pol-1")
            mock_vps.assert_called_once_with("pol-1", "policy_id")
            assert result == (True, None)

    def test_user_id_delegates_to_validate_path_segment(self):
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(True, None),
        ) as mock_vps:
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_user_id,
            )

            result = _validate_user_id("usr-1")
            mock_vps.assert_called_once_with("usr-1", "user_id")
            assert result == (True, None)

    def test_workspace_id_returns_fallback_error_when_validate_returns_none_err(self):
        """When validate_path_segment returns (False, None), the fallback message is used."""
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(False, None),
        ):
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_workspace_id,
            )

            is_valid, err = _validate_workspace_id("bad")
            assert is_valid is False
            assert "Invalid workspace_id format" in err

    def test_policy_id_returns_fallback_error_when_validate_returns_none_err(self):
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(False, None),
        ):
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_policy_id,
            )

            is_valid, err = _validate_policy_id("bad")
            assert is_valid is False
            assert "Invalid policy_id format" in err

    def test_user_id_returns_fallback_error_when_validate_returns_none_err(self):
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(False, None),
        ):
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_user_id,
            )

            is_valid, err = _validate_user_id("bad")
            assert is_valid is False
            assert "Invalid user_id format" in err

    def test_workspace_id_returns_upstream_error_when_provided(self):
        """When validate_path_segment returns a specific error, it is passed through."""
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
            return_value=(False, "custom error from upstream"),
        ):
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_workspace_id,
            )

            is_valid, err = _validate_workspace_id("bad")
            assert is_valid is False
            assert err == "custom error from upstream"

    def test_empty_ids_short_circuit_before_delegation(self):
        """Empty strings return early without calling validate_path_segment."""
        with patch(
            "aragora.server.handlers.workspace.workspace_utils.validate_path_segment",
        ) as mock_vps:
            from aragora.server.handlers.workspace.workspace_utils import (
                _validate_workspace_id,
                _validate_policy_id,
                _validate_user_id,
            )

            _validate_workspace_id("")
            _validate_policy_id("")
            _validate_user_id("")
            mock_vps.assert_not_called()
