"""Tests for BaseHandler.require_user — the auth-narrowing helper.

``require_user`` wraps the legacy ``require_auth_or_error`` /
``require_permission_or_error`` tuple-unpacking pattern so that mypy can
narrow the returned ``UserAuthContext`` without needing call-site asserts.

These tests lock in:
1. The happy paths (authenticated-only, authenticated + permission).
2. The error paths (401 on missing auth, 403 on missing permission).
3. That the existing tuple-returning helpers remain unchanged so callers
   outside the migrated hotspot keep working.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import BaseHandler, HandlerResult, error_response


def _make_handler() -> BaseHandler:
    """Return a BaseHandler with a minimal stub context."""
    return BaseHandler(
        {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": MagicMock(),
        }
    )


# =============================================================================
# require_user — authentication only
# =============================================================================


class TestRequireUserAuthOnly:
    """``require_user(handler)`` without a permission argument."""

    def test_authenticated_returns_narrow_user(self) -> None:
        handler = _make_handler()
        request = MagicMock()

        expected_user = MagicMock(user_id="u1", org_id="o1")
        with patch.object(
            handler,
            "require_auth_or_error",
            return_value=(expected_user, None),
        ) as spy:
            result = handler.require_user(request)

        assert result is expected_user
        assert not isinstance(result, HandlerResult)
        spy.assert_called_once_with(request)

    def test_unauthenticated_returns_401_handler_result(self) -> None:
        handler = _make_handler()
        request = MagicMock()
        err = error_response("Authentication required", 401)

        with patch.object(
            handler,
            "require_auth_or_error",
            return_value=(None, err),
        ):
            result = handler.require_user(request)

        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    def test_permission_arg_not_used_when_none(self) -> None:
        """When permission=None, the permission helper is never invoked."""
        handler = _make_handler()
        request = MagicMock()

        with (
            patch.object(
                handler,
                "require_auth_or_error",
                return_value=(MagicMock(user_id="u1"), None),
            ) as auth_spy,
            patch.object(handler, "require_permission_or_error") as perm_spy,
        ):
            handler.require_user(request)

        auth_spy.assert_called_once_with(request)
        perm_spy.assert_not_called()


# =============================================================================
# require_user — with a permission argument
# =============================================================================


class TestRequireUserPermission:
    """``require_user(handler, permission=...)`` routes through the permission helper."""

    def test_granted_permission_returns_narrow_user(self) -> None:
        handler = _make_handler()
        request = MagicMock()
        expected_user = MagicMock(user_id="u1", org_id="o1")

        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(expected_user, None),
        ) as spy:
            result = handler.require_user(request, permission="write")

        assert result is expected_user
        assert not isinstance(result, HandlerResult)
        spy.assert_called_once_with(request, "write")

    def test_denied_permission_returns_403(self) -> None:
        handler = _make_handler()
        request = MagicMock()
        err = error_response("Permission denied", 403)

        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, err),
        ):
            result = handler.require_user(request, permission="write")

        assert isinstance(result, HandlerResult)
        assert result.status_code == 403

    def test_unauthenticated_returns_401_via_permission_path(self) -> None:
        handler = _make_handler()
        request = MagicMock()
        err = error_response("Authentication required", 401)

        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, err),
        ):
            result = handler.require_user(request, permission="read")

        assert isinstance(result, HandlerResult)
        assert result.status_code == 401


# =============================================================================
# Defensive behaviour
# =============================================================================


class TestRequireUserDefensive:
    """Invariant: the helpers never simultaneously return (None, None)."""

    def test_both_none_raises_runtime_error(self) -> None:
        """If the underlying helper violates its contract, fail loudly."""
        handler = _make_handler()
        request = MagicMock()

        with patch.object(
            handler,
            "require_auth_or_error",
            return_value=(None, None),
        ):
            with pytest.raises(RuntimeError, match="authenticated user missing"):
                handler.require_user(request)


# =============================================================================
# Non-regression: existing tuple-returning helpers are preserved
# =============================================================================


class TestExistingHelpersUnchanged:
    """``require_user`` is additive; the tuple-returning helpers must still exist."""

    def test_require_auth_or_error_still_exists(self) -> None:
        """Existing callers that unpack the tuple keep working."""
        handler = _make_handler()
        assert callable(getattr(handler, "require_auth_or_error", None))
        assert callable(getattr(handler, "require_permission_or_error", None))

    def test_require_user_added_to_base_handler(self) -> None:
        """`require_user` is exposed on BaseHandler (and therefore all subclasses)."""
        from aragora.server.handlers.base import BaseHandler

        assert callable(getattr(BaseHandler, "require_user", None))
