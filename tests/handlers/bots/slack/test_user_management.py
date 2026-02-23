"""
Tests for Slack User and Team Management.

Covers all functions in aragora.server.handlers.bots.slack.user_management:
- get_org_from_team(team_id)
  - Successful org lookup from workspace store
  - Workspace found but no org_id
  - Workspace not found (returns None)
  - ImportError fallback (returns None)
  - AttributeError fallback (returns None)
  - RuntimeError fallback (returns None)
- get_user_roles_from_slack(team_id, user_id)
  - Successful role lookup
  - Returns set from list
  - Empty roles returns default {"user"}
  - None roles returns default {"user"}
  - ImportError fallback returns {"user"}
  - AttributeError fallback returns {"user"}
  - RuntimeError fallback returns {"user"}
- check_workspace_authorized(team_id)
  - Valid, authorized workspace
  - Invalid team_id format
  - Workspace not found (unauthorized)
  - Workspace revoked
  - ImportError in dev mode (allowed)
  - ImportError in production (denied)
  - Various ARAGORA_ENV values
- build_auth_context_from_slack(team_id, user_id, channel_id)
  - RBAC available, builds context with org + roles
  - RBAC not available (returns None)
  - AuthorizationContext is None (returns None)
  - TypeError during construction (returns None)
  - channel_id forwarded as workspace_id (uses team_id)
  - user_id prefixed with "slack:"
- check_user_permission(team_id, user_id, permission_key, channel_id)
  - RBAC not available + fail open
  - RBAC not available + fail closed (503)
  - Workspace not authorized (403)
  - Permission granted (returns None)
  - Permission denied (403 + audit)
  - Context build fails + dev mode (allowed)
  - Context build fails + production (500)
  - RBAC check raises exception + dev mode
  - RBAC check raises exception + production (500)
  - Various exception types in RBAC check
- check_user_permission_or_admin(team_id, user_id, permission_key, channel_id)
  - RBAC not available + fail open
  - RBAC not available + fail closed (503)
  - Admin permission granted (bypasses specific check)
  - Admin denied, specific permission granted
  - Admin denied, specific permission denied
  - Context build fails (returns None)
  - Admin check raises exception (falls back to specific check)
"""

from __future__ import annotations

import json
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
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Lazy import fixture (so conftest auto-auth patches run first)
# ---------------------------------------------------------------------------


@pytest.fixture
def um():
    """Import the user_management module lazily."""
    import aragora.server.handlers.bots.slack.user_management as mod

    return mod


@pytest.fixture(autouse=True)
def _patch_rbac_defaults(monkeypatch):
    """Default: RBAC available, permission granted, fail-open, audit mocked."""
    mock_decision = MagicMock()
    mock_decision.allowed = True
    mock_decision.reason = "granted"

    # Patch on the constants module (source of truth)
    monkeypatch.setattr("aragora.server.handlers.bots.slack.constants.RBAC_AVAILABLE", True)
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.constants.check_permission",
        MagicMock(return_value=mock_decision),
    )
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.constants.AuthorizationContext",
        MagicMock,
    )

    # Patch on the user_management module (which imports from constants)
    monkeypatch.setattr("aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", True)
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.user_management.check_permission",
        MagicMock(return_value=mock_decision),
    )
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
        MagicMock,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
        lambda: False,
    )
    monkeypatch.setattr(
        "aragora.server.handlers.bots.slack.user_management.audit_data",
        MagicMock(),
    )


# ===========================================================================
# get_org_from_team
# ===========================================================================


class TestGetOrgFromTeam:
    """Tests for get_org_from_team()."""

    def test_returns_org_id_when_workspace_found(self, um, monkeypatch):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": "org-123"}
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_slack_workspace_store",
            lambda: mock_store,
            raising=False,
        )
        # We need to patch the import inside the function
        with patch(
            "aragora.server.handlers.bots.slack.user_management.get_slack_workspace_store",
            create=True,
        ) as mock_getter:
            # Actually we need to patch the inner import
            pass

        # The function does `from aragora.storage... import get_slack_workspace_store`
        # We need to patch it at the source
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": "org-123"}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_org_from_team("T12345ABC")
        assert result == "org-123"

    def test_returns_none_when_workspace_not_found(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = None
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_org_from_team("T12345ABC")
        assert result is None

    def test_returns_none_when_no_org_id_key(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"name": "workspace"}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_org_from_team("T12345ABC")
        assert result is None

    def test_returns_none_on_import_error(self, um):
        """ImportError on workspace store import -> returns None."""
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = um.get_org_from_team("T12345ABC")
        assert result is None

    def test_returns_none_on_attribute_error(self, um):
        """Module exists but get_slack_workspace_store missing."""
        mock_mod = MagicMock(spec=[])  # No attributes
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_mod}):
            result = um.get_org_from_team("T12345ABC")
        assert result is None

    def test_returns_none_on_runtime_error(self, um):
        """get_slack_workspace_store raises RuntimeError."""
        mock_mod = MagicMock()
        mock_mod.get_slack_workspace_store.side_effect = RuntimeError("db down")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_mod}):
            result = um.get_org_from_team("T12345ABC")
        assert result is None

    def test_returns_none_when_org_id_is_none(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": None}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_org_from_team("T12345ABC")
        assert result is None

    def test_passes_team_id_to_store(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": "org-x"}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            um.get_org_from_team("TABC123")
        mock_store.get_workspace.assert_called_once_with("TABC123")


# ===========================================================================
# get_user_roles_from_slack
# ===========================================================================


class TestGetUserRolesFromSlack:
    """Tests for get_user_roles_from_slack()."""

    def test_returns_roles_from_store(self, um):
        mock_store = MagicMock()
        mock_store.get_user_roles.return_value = ["admin", "moderator"]
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"admin", "moderator"}

    def test_returns_set_type(self, um):
        mock_store = MagicMock()
        mock_store.get_user_roles.return_value = ["admin"]
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert isinstance(result, set)

    def test_default_user_role_when_no_roles(self, um):
        mock_store = MagicMock()
        mock_store.get_user_roles.return_value = None
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"user"}

    def test_default_user_role_when_empty_list(self, um):
        mock_store = MagicMock()
        mock_store.get_user_roles.return_value = []
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"user"}

    def test_default_user_role_on_import_error(self, um):
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"user"}

    def test_default_user_role_on_attribute_error(self, um):
        mock_mod = MagicMock(spec=[])
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_mod}):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"user"}

    def test_default_user_role_on_runtime_error(self, um):
        mock_mod = MagicMock()
        mock_mod.get_slack_workspace_store.side_effect = RuntimeError("down")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_mod}):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"user"}

    def test_passes_team_and_user_to_store(self, um):
        mock_store = MagicMock()
        mock_store.get_user_roles.return_value = ["user"]
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            um.get_user_roles_from_slack("TABC", "UXYZ")
        mock_store.get_user_roles.assert_called_once_with("TABC", "UXYZ")

    def test_deduplicates_roles(self, um):
        mock_store = MagicMock()
        mock_store.get_user_roles.return_value = ["admin", "admin", "user"]
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.get_user_roles_from_slack("T12345", "U12345")
        assert result == {"admin", "user"}


# ===========================================================================
# check_workspace_authorized
# ===========================================================================


class TestCheckWorkspaceAuthorized:
    """Tests for check_workspace_authorized()."""

    def test_authorized_workspace(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"name": "test-ws", "revoked": False}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True
        assert error is None

    def test_invalid_team_id_format(self, um):
        """Team IDs must start with T followed by alphanumeric chars."""
        authorized, error = um.check_workspace_authorized("invalid")
        assert authorized is False
        assert error is not None

    def test_empty_team_id(self, um):
        authorized, error = um.check_workspace_authorized("")
        assert authorized is False
        assert error is not None

    def test_workspace_not_found(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = None
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is False
        assert "not authorized" in error.lower()

    def test_workspace_revoked(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"name": "test-ws", "revoked": True}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is False
        assert "revoked" in error.lower()

    def test_workspace_not_revoked_explicitly_false(self, um):
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"name": "test-ws", "revoked": False}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True

    def test_workspace_without_revoked_key(self, um):
        """Workspace dict without 'revoked' key should still be authorized."""
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"name": "test-ws"}
        mock_get_store = MagicMock(return_value=mock_store)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        # dict.get("revoked") returns None -> falsy -> authorized
        assert authorized is True
        assert error is None

    def test_import_error_dev_mode_allows(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "development")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True
        assert error is None

    def test_import_error_test_mode_allows(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "test")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True

    def test_import_error_local_mode_allows(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "local")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True

    def test_import_error_dev_short_allows(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "dev")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True

    def test_import_error_production_denies(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is False
        assert "failed" in error.lower()

    def test_import_error_default_env_denies(self, um, monkeypatch):
        """Default ARAGORA_ENV is 'production', so denies on import error."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is False

    def test_runtime_error_production_denies(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        mock_mod = MagicMock()
        mock_mod.get_slack_workspace_store.side_effect = RuntimeError("db down")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_mod}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is False

    def test_runtime_error_dev_mode_allows(self, um, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "development")
        mock_mod = MagicMock()
        mock_mod.get_slack_workspace_store.side_effect = RuntimeError("db down")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_mod}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True

    def test_team_id_too_long(self, um):
        long_id = "T" + "A" * 200
        authorized, error = um.check_workspace_authorized(long_id)
        assert authorized is False

    def test_case_insensitive_env_check(self, um, monkeypatch):
        """ARAGORA_ENV comparison is case-insensitive."""
        monkeypatch.setenv("ARAGORA_ENV", "DEVELOPMENT")
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            authorized, error = um.check_workspace_authorized("T12345ABC")
        assert authorized is True


# ===========================================================================
# build_auth_context_from_slack
# ===========================================================================


class TestBuildAuthContextFromSlack:
    """Tests for build_auth_context_from_slack()."""

    def test_returns_none_when_rbac_unavailable(self, um, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", False
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is None

    def test_returns_none_when_auth_context_class_is_none(self, um, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", True
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext", None
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is None

    def test_builds_context_with_slack_prefix_user_id(self, um, monkeypatch):
        """user_id should be prefixed with 'slack:'."""
        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: "org-abc",
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"admin"},
        )
        um.build_auth_context_from_slack("T12345", "U67890")
        mock_ctx_cls.assert_called_once()
        call_kwargs = mock_ctx_cls.call_args[1]
        assert call_kwargs["user_id"] == "slack:U67890"

    def test_builds_context_with_org_id(self, um, monkeypatch):
        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: "org-abc",
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        um.build_auth_context_from_slack("T12345", "U12345")
        call_kwargs = mock_ctx_cls.call_args[1]
        assert call_kwargs["org_id"] == "org-abc"

    def test_builds_context_with_roles(self, um, monkeypatch):
        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"admin", "moderator"},
        )
        um.build_auth_context_from_slack("T12345", "U12345")
        call_kwargs = mock_ctx_cls.call_args[1]
        assert call_kwargs["roles"] == {"admin", "moderator"}

    def test_workspace_id_is_team_id(self, um, monkeypatch):
        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        um.build_auth_context_from_slack("TABC123", "U12345")
        call_kwargs = mock_ctx_cls.call_args[1]
        assert call_kwargs["workspace_id"] == "TABC123"

    def test_channel_id_does_not_affect_workspace_id(self, um, monkeypatch):
        """channel_id is accepted but workspace_id always uses team_id."""
        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        um.build_auth_context_from_slack("T12345", "U12345", channel_id="C99999")
        call_kwargs = mock_ctx_cls.call_args[1]
        assert call_kwargs["workspace_id"] == "T12345"

    def test_returns_none_on_type_error(self, um, monkeypatch):
        mock_ctx_cls = MagicMock(side_effect=TypeError("bad args"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is None

    def test_returns_none_on_value_error(self, um, monkeypatch):
        mock_ctx_cls = MagicMock(side_effect=ValueError("invalid"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is None

    def test_returns_none_on_key_error(self, um, monkeypatch):
        mock_ctx_cls = MagicMock(side_effect=KeyError("missing"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is None

    def test_returns_none_on_attribute_error(self, um, monkeypatch):
        mock_ctx_cls = MagicMock(side_effect=AttributeError("missing"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is None

    def test_returns_context_object(self, um, monkeypatch):
        """Verify the function returns whatever AuthorizationContext() produces."""
        sentinel = object()
        mock_ctx_cls = MagicMock(return_value=sentinel)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is sentinel

    def test_no_channel_id_default(self, um, monkeypatch):
        """channel_id defaults to None and function still works."""
        mock_ctx_cls = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            mock_ctx_cls,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_org_from_team",
            lambda t: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.get_user_roles_from_slack",
            lambda t, u: {"user"},
        )
        result = um.build_auth_context_from_slack("T12345", "U12345")
        assert result is not None


# ===========================================================================
# check_user_permission
# ===========================================================================


class TestCheckUserPermission:
    """Tests for check_user_permission()."""

    def test_rbac_unavailable_fail_open_allows(self, um, monkeypatch):
        """When RBAC not available and fail_closed=False, returns None (allow)."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
            lambda: False,
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_rbac_unavailable_fail_closed_returns_503(self, um, monkeypatch):
        """When RBAC not available and fail_closed=True, returns 503."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
            lambda: True,
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 503
        assert "unavailable" in _body(result).get("error", "").lower()

    def test_check_permission_none_fail_open(self, um, monkeypatch):
        """check_permission is None -> treated as RBAC unavailable."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", True
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
            lambda: False,
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_workspace_not_authorized_returns_403(self, um, monkeypatch):
        """Unauthorized workspace returns 403."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (False, "Workspace not authorized"),
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 403

    def test_workspace_authorized_error_message_forwarded(self, um, monkeypatch):
        """Error message from check_workspace_authorized is forwarded."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (False, "Custom error message"),
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert "Custom error message" in _body(result).get("error", "")

    def test_workspace_authorized_none_error_fallback(self, um, monkeypatch):
        """When workspace error is None, fallback message is used."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (False, None),
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 403
        assert "not authorized" in _body(result).get("error", "").lower()

    def test_permission_granted_returns_none(self, um, monkeypatch):
        """Permission granted -> returns None (no error)."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        mock_ctx = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: mock_ctx,
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_permission_denied_returns_403(self, um, monkeypatch):
        """Permission denied -> returns 403."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "Insufficient permissions"
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        mock_ctx = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: mock_ctx,
        )
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 403
        assert "denied" in _body(result).get("error", "").lower()

    def test_permission_denied_calls_audit_data(self, um, monkeypatch):
        """Permission denied triggers audit_data call."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "No permission"
        mock_check = MagicMock(return_value=mock_decision)
        mock_audit = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.audit_data",
            mock_audit,
        )
        um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        mock_audit.assert_called_once()
        kwargs = mock_audit.call_args[1]
        assert kwargs["user_id"] == "slack:U12345"
        assert kwargs["resource_type"] == "slack_permission"
        assert kwargs["resource_id"] == "slack.commands.execute"
        assert kwargs["action"] == "denied"
        assert kwargs["platform"] == "slack"
        assert kwargs["team_id"] == "T12345ABC"

    def test_permission_granted_does_not_audit(self, um, monkeypatch):
        """Permission granted does not trigger audit_data."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)
        mock_audit = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.audit_data",
            mock_audit,
        )
        um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        mock_audit.assert_not_called()

    def test_context_none_dev_mode_allows(self, um, monkeypatch):
        """Context build returns None in dev mode -> allows."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: None,
        )
        monkeypatch.setenv("ARAGORA_ENV", "development")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_context_none_test_mode_allows(self, um, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: None,
        )
        monkeypatch.setenv("ARAGORA_ENV", "test")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_context_none_production_returns_500(self, um, monkeypatch):
        """Context build returns None in production -> 500."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: None,
        )
        monkeypatch.setenv("ARAGORA_ENV", "production")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 500

    def test_rbac_check_type_error_dev_allows(self, um, monkeypatch):
        mock_check = MagicMock(side_effect=TypeError("bad"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setenv("ARAGORA_ENV", "development")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_rbac_check_runtime_error_production_returns_500(self, um, monkeypatch):
        mock_check = MagicMock(side_effect=RuntimeError("RBAC down"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setenv("ARAGORA_ENV", "production")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 500

    def test_rbac_check_value_error_dev_allows(self, um, monkeypatch):
        mock_check = MagicMock(side_effect=ValueError("invalid"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setenv("ARAGORA_ENV", "dev")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_rbac_check_key_error_production_returns_500(self, um, monkeypatch):
        mock_check = MagicMock(side_effect=KeyError("missing"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setenv("ARAGORA_ENV", "production")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 500

    def test_rbac_check_attribute_error_local_allows(self, um, monkeypatch):
        mock_check = MagicMock(side_effect=AttributeError("missing"))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setenv("ARAGORA_ENV", "local")
        result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_channel_id_forwarded(self, um, monkeypatch):
        """channel_id parameter is forwarded to build_auth_context_from_slack."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        build_mock = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            build_mock,
        )
        um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute", "C99999")
        build_mock.assert_called_once_with("T12345ABC", "U12345", "C99999")

    def test_permission_key_forwarded_to_check_permission(self, um, monkeypatch):
        """Permission key is forwarded to check_permission."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        ctx = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: ctx,
        )
        um.check_user_permission("T12345ABC", "U12345", "slack.debates.create")
        mock_check.assert_called_once_with(ctx, "slack.debates.create")

    def test_invalid_team_id_returns_403(self, um, monkeypatch):
        """Invalid team ID format causes workspace check to fail -> 403."""
        # The actual workspace check validates team_id format first
        result = um.check_user_permission("invalid", "U12345", "slack.commands.execute")
        assert _status(result) == 403


# ===========================================================================
# check_user_permission_or_admin
# ===========================================================================


class TestCheckUserPermissionOrAdmin:
    """Tests for check_user_permission_or_admin()."""

    def test_rbac_unavailable_fail_open_allows(self, um, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
            lambda: False,
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_rbac_unavailable_fail_closed_returns_503(self, um, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
            lambda: True,
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 503

    def test_admin_permission_granted_bypasses_specific_check(self, um, monkeypatch):
        """Admin permission -> returns None immediately without checking specific perm."""
        admin_decision = MagicMock()
        admin_decision.allowed = True

        call_count = {"n": 0}

        def mock_check(ctx, perm):
            call_count["n"] += 1
            return admin_decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None
        # Only called once (admin check), not twice
        assert call_count["n"] == 1

    def test_admin_denied_falls_through_to_specific_permission(self, um, monkeypatch):
        """Admin denied -> falls through to check_user_permission."""
        admin_decision = MagicMock()
        admin_decision.allowed = False

        specific_decision = MagicMock()
        specific_decision.allowed = True

        calls = []

        def mock_check(ctx, perm):
            calls.append(perm)
            if perm == "slack.admin":
                return admin_decision
            return specific_decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None
        assert "slack.admin" in calls

    def test_both_denied_returns_403(self, um, monkeypatch):
        """Admin denied + specific denied -> 403."""
        denied_decision = MagicMock()
        denied_decision.allowed = False
        denied_decision.reason = "Not allowed"
        mock_check = MagicMock(return_value=denied_decision)
        mock_audit = MagicMock()

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.audit_data",
            mock_audit,
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 403

    def test_context_build_fails_returns_none(self, um, monkeypatch):
        """When context build fails, returns None (permissive)."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: None,
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_admin_check_exception_falls_back_to_specific(self, um, monkeypatch):
        """Admin check raises -> falls back to check_user_permission."""
        call_count = {"n": 0}

        def mock_check(ctx, perm):
            call_count["n"] += 1
            if perm == "slack.admin":
                raise RuntimeError("admin check failed")
            decision = MagicMock()
            decision.allowed = True
            return decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_admin_check_type_error_falls_back(self, um, monkeypatch):
        """TypeError in admin check -> falls back to check_user_permission."""
        first_call = [True]

        def mock_check(ctx, perm):
            if first_call[0]:
                first_call[0] = False
                raise TypeError("bad admin check")
            decision = MagicMock()
            decision.allowed = True
            return decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_admin_check_value_error_falls_back(self, um, monkeypatch):
        first_call = [True]

        def mock_check(ctx, perm):
            if first_call[0]:
                first_call[0] = False
                raise ValueError("bad value")
            decision = MagicMock()
            decision.allowed = True
            return decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_admin_check_key_error_falls_back(self, um, monkeypatch):
        first_call = [True]

        def mock_check(ctx, perm):
            if first_call[0]:
                first_call[0] = False
                raise KeyError("no key")
            decision = MagicMock()
            decision.allowed = True
            return decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_admin_check_attribute_error_falls_back(self, um, monkeypatch):
        first_call = [True]

        def mock_check(ctx, perm):
            if first_call[0]:
                first_call[0] = False
                raise AttributeError("missing attr")
            decision = MagicMock()
            decision.allowed = True
            return decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None

    def test_channel_id_forwarded_to_build_context(self, um, monkeypatch):
        """channel_id is forwarded to build_auth_context_from_slack."""
        admin_decision = MagicMock()
        admin_decision.allowed = True
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            MagicMock(return_value=admin_decision),
        )
        build_mock = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            build_mock,
        )
        um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute", "C12345")
        build_mock.assert_called_once_with("T12345ABC", "U12345", "C12345")

    def test_check_permission_none_fail_closed(self, um, monkeypatch):
        """check_permission is None with fail_closed -> 503."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.RBAC_AVAILABLE", True
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission", None
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.rbac_fail_closed",
            lambda: True,
        )
        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert _status(result) == 503

    def test_admin_checks_perm_slack_admin(self, um, monkeypatch):
        """First check should be for PERM_SLACK_ADMIN ('slack.admin')."""
        perms_checked = []

        def mock_check(ctx, perm):
            perms_checked.append(perm)
            decision = MagicMock()
            decision.allowed = True
            return decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert perms_checked[0] == "slack.admin"


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self, um):
        for name in um.__all__:
            assert hasattr(um, name), f"Missing export: {name}"

    def test_get_org_from_team_exported(self, um):
        assert "get_org_from_team" in um.__all__

    def test_get_user_roles_from_slack_exported(self, um):
        assert "get_user_roles_from_slack" in um.__all__

    def test_check_workspace_authorized_exported(self, um):
        assert "check_workspace_authorized" in um.__all__

    def test_build_auth_context_from_slack_exported(self, um):
        assert "build_auth_context_from_slack" in um.__all__

    def test_check_user_permission_exported(self, um):
        assert "check_user_permission" in um.__all__

    def test_check_user_permission_or_admin_exported(self, um):
        assert "check_user_permission_or_admin" in um.__all__

    def test_exports_count(self, um):
        assert len(um.__all__) == 6


# ===========================================================================
# Integration-style tests: full flow through check_user_permission
# ===========================================================================


class TestPermissionFlowIntegration:
    """Integration-style tests that exercise multiple functions together."""

    def test_full_flow_permission_granted(self, um, monkeypatch):
        """Full path: workspace authorized -> context built -> permission granted."""
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": "org-1", "revoked": False}
        mock_store.get_user_roles.return_value = ["admin"]
        mock_get_store = MagicMock(return_value=mock_store)

        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )

        from aragora.rbac.models import AuthorizationContext as RealAuthCtx

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            RealAuthCtx,
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")

        assert result is None
        # Verify check_permission was called with correct context
        mock_check.assert_called_once()
        ctx_arg = mock_check.call_args[0][0]
        assert ctx_arg.user_id == "slack:U12345"
        assert ctx_arg.org_id == "org-1"
        assert ctx_arg.workspace_id == "T12345ABC"
        assert "admin" in ctx_arg.roles

    def test_full_flow_permission_denied(self, um, monkeypatch):
        """Full path: workspace authorized -> context built -> permission denied."""
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": "org-1", "revoked": False}
        mock_store.get_user_roles.return_value = ["user"]
        mock_get_store = MagicMock(return_value=mock_store)

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "No permission"
        mock_check = MagicMock(return_value=mock_decision)
        mock_audit = MagicMock()

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.audit_data",
            mock_audit,
        )

        from aragora.rbac.models import AuthorizationContext as RealAuthCtx

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.AuthorizationContext",
            RealAuthCtx,
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")

        assert _status(result) == 403
        mock_audit.assert_called_once()

    def test_full_flow_workspace_revoked(self, um, monkeypatch):
        """Full path: workspace revoked -> 403 without permission check."""
        mock_store = MagicMock()
        mock_store.get_workspace.return_value = {"org_id": "org-1", "revoked": True}
        mock_get_store = MagicMock(return_value=mock_store)

        mock_check = MagicMock()

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.slack_workspace_store": MagicMock(
                    get_slack_workspace_store=mock_get_store
                )
            },
        ):
            result = um.check_user_permission("T12345ABC", "U12345", "slack.commands.execute")

        assert _status(result) == 403
        # check_permission should not have been called since workspace is revoked
        mock_check.assert_not_called()

    def test_admin_bypass_full_flow(self, um, monkeypatch):
        """Admin user bypasses specific permission check."""
        admin_decision = MagicMock()
        admin_decision.allowed = True

        perms_checked = []

        def mock_check(ctx, perm):
            perms_checked.append(perm)
            return admin_decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )

        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.commands.execute")
        assert result is None
        # Only admin check, specific permission not checked
        assert perms_checked == ["slack.admin"]

    def test_non_admin_falls_to_specific_permission_flow(self, um, monkeypatch):
        """Non-admin user goes through full specific permission check."""
        admin_decision = MagicMock()
        admin_decision.allowed = False

        specific_decision = MagicMock()
        specific_decision.allowed = True

        def mock_check(ctx, perm):
            if perm == "slack.admin":
                return admin_decision
            return specific_decision

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.build_auth_context_from_slack",
            lambda t, u, channel_id=None: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.user_management.check_workspace_authorized",
            lambda t: (True, None),
        )

        result = um.check_user_permission_or_admin("T12345ABC", "U12345", "slack.debates.create")
        assert result is None
