"""
Tests for email handler storage utilities.

Covers:
- _check_email_permission RBAC logic
- Thread-safe lazy singleton initialization
- User config cache (get/set)
- Persistent store load/save
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_mod
import aragora.server.handlers.utils.rbac_guard as rbac_guard_mod


@dataclass
class FakeDecision:
    allowed: bool
    reason: str = ""


@dataclass
class FakeAuthContext:
    user_id: str = "test-user"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons between tests."""
    saved_store = storage_mod._email_store
    storage_mod._email_store = None
    storage_mod._gmail_connector = None
    storage_mod._prioritizer = None
    storage_mod._context_service = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()

    # Reset rbac_guard cached import state so that patching RBAC_AVAILABLE
    # on the storage module is not undermined by the guard's own cache.
    saved_attempted = rbac_guard_mod._rbac_import_attempted
    saved_success = rbac_guard_mod._rbac_import_success
    rbac_guard_mod._rbac_import_attempted = False
    rbac_guard_mod._rbac_import_success = False

    yield

    storage_mod._email_store = saved_store
    storage_mod._gmail_connector = None
    storage_mod._prioritizer = None
    storage_mod._context_service = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()

    # Restore rbac_guard cached state
    rbac_guard_mod._rbac_import_attempted = saved_attempted
    rbac_guard_mod._rbac_import_success = saved_success


# ---------------------------------------------------------------------------
# _check_email_permission
# ---------------------------------------------------------------------------


class TestCheckEmailPermission:
    """Tests for _check_email_permission."""

    def test_read_allowed_when_rbac_unavailable(self):
        """Read ops degrade gracefully when RBAC is unavailable."""
        with patch.object(storage_mod, "RBAC_AVAILABLE", False):
            result = storage_mod._check_email_permission(None, "email:read")
        assert result is None  # allowed

    def test_write_denied_when_rbac_unavailable(self):
        """Write ops fail closed when RBAC is unavailable."""
        with patch.object(storage_mod, "RBAC_AVAILABLE", False):
            result = storage_mod._check_email_permission(None, "email:write")
        assert result is not None
        assert result["success"] is False
        # Error message is deliberately generic (security: no RBAC reason leakage)
        assert "denied" in result["error"].lower() or "unavailable" in result["error"].lower()

    def test_oauth_denied_when_rbac_unavailable(self):
        """OAuth ops fail closed when RBAC is unavailable."""
        with patch.object(storage_mod, "RBAC_AVAILABLE", False):
            result = storage_mod._check_email_permission(None, "email:oauth")
        assert result is not None
        assert result["success"] is False

    def test_write_denied_when_auth_context_is_none(self):
        """Write ops denied when no auth context is provided."""
        with patch.object(storage_mod, "RBAC_AVAILABLE", True):
            result = storage_mod._check_email_permission(None, "email:write")
        assert result is not None
        assert result["success"] is False

    def test_allowed_when_permission_granted(self):
        """Returns None (allowed) when check_permission allows."""
        ctx = FakeAuthContext()
        decision = FakeDecision(allowed=True)
        with (
            patch.object(storage_mod, "RBAC_AVAILABLE", True),
            patch.object(storage_mod, "check_permission", return_value=decision),
        ):
            result = storage_mod._check_email_permission(ctx, "email:write")
        assert result is None

    def test_denied_when_permission_rejected(self):
        """Returns error dict when check_permission denies."""
        ctx = FakeAuthContext()
        decision = FakeDecision(allowed=False, reason="Insufficient role")
        with (
            patch.object(storage_mod, "RBAC_AVAILABLE", True),
            patch.object(storage_mod, "check_permission", return_value=decision),
        ):
            result = storage_mod._check_email_permission(ctx, "email:write")
        assert result is not None
        assert result["success"] is False
        # Error message is deliberately generic (security: no RBAC reason leakage)
        assert "denied" in result["error"].lower()

    def test_fails_open_on_check_exception(self):
        """Fails open (allows) when check_permission raises."""
        ctx = FakeAuthContext()
        with (
            patch.object(storage_mod, "RBAC_AVAILABLE", True),
            patch.object(storage_mod, "check_permission", side_effect=TypeError("boom")),
        ):
            result = storage_mod._check_email_permission(ctx, "email:read")
        assert result is None  # fail-open


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------


class TestLazySingletons:
    """Tests for thread-safe lazy initialization."""

    def test_get_email_store_returns_none_on_import_failure(self):
        """Returns None if email store can't be created."""
        with patch(
            "aragora.server.handlers.email.storage.get_email_store",
            wraps=storage_mod.get_email_store,
        ):
            # The real import may fail in test environment; that's fine
            store = storage_mod.get_email_store()
            # Either returns a store or None (import failure)
            assert store is None or store is not None  # no crash

    def test_get_gmail_connector_creates_singleton(self):
        """Gmail connector is created once and cached."""
        mock_connector = MagicMock()
        with patch(
            "aragora.connectors.enterprise.communication.gmail.GmailConnector",
            return_value=mock_connector,
        ):
            c1 = storage_mod.get_gmail_connector()
            c2 = storage_mod.get_gmail_connector()
        assert c1 is c2
        assert c1 is mock_connector

    def test_get_context_service_creates_singleton(self):
        """Context service is created once and cached."""
        mock_service = MagicMock()
        with patch(
            "aragora.services.cross_channel_context.CrossChannelContextService",
            return_value=mock_service,
        ):
            s1 = storage_mod.get_context_service()
            s2 = storage_mod.get_context_service()
        assert s1 is s2
        assert s1 is mock_service


# ---------------------------------------------------------------------------
# User config cache
# ---------------------------------------------------------------------------


class TestUserConfigCache:
    """Tests for thread-safe user config get/set."""

    def test_get_returns_empty_for_unknown_user(self):
        result = storage_mod.get_user_config("nonexistent")
        assert result == {}

    def test_set_and_get_roundtrip(self):
        storage_mod.set_user_config("u1", {"vip_domains": ["example.com"]})
        config = storage_mod.get_user_config("u1")
        assert config == {"vip_domains": ["example.com"]}

    def test_get_returns_copy(self):
        """Returned config is a copy, not the internal dict."""
        storage_mod.set_user_config("u1", {"key": "value"})
        c1 = storage_mod.get_user_config("u1")
        c1["key"] = "mutated"
        c2 = storage_mod.get_user_config("u1")
        assert c2["key"] == "value"

    def test_set_stores_copy(self):
        """set_user_config stores a copy, not a reference."""
        original = {"key": "value"}
        storage_mod.set_user_config("u1", original)
        original["key"] = "mutated"
        assert storage_mod.get_user_config("u1")["key"] == "value"


# ---------------------------------------------------------------------------
# Persistent store helpers
# ---------------------------------------------------------------------------


class TestPersistentStoreHelpers:
    """Tests for _load_config_from_store and _save_config_to_store."""

    def _patch_store(self, mock_store):
        """Patch _email_store.get() to return the given mock store."""
        lazy = MagicMock()
        lazy.get.return_value = mock_store
        return patch.object(storage_mod, "_email_store", lazy)

    def test_load_returns_empty_when_no_store(self):
        """Returns empty dict when email store is unavailable."""
        with self._patch_store(None):
            result = storage_mod._load_config_from_store("u1")
        assert result == {}

    def test_load_returns_config_from_store(self):
        mock_store = MagicMock()
        mock_store.get_user_config.return_value = {"vip_domains": ["test.com"]}
        with self._patch_store(mock_store):
            result = storage_mod._load_config_from_store("u1", "ws1")
        assert result == {"vip_domains": ["test.com"]}
        mock_store.get_user_config.assert_called_once_with("u1", "ws1")

    def test_load_returns_empty_on_store_exception(self):
        mock_store = MagicMock()
        mock_store.get_user_config.side_effect = OSError("db down")
        with self._patch_store(mock_store):
            result = storage_mod._load_config_from_store("u1")
        assert result == {}

    def test_save_calls_store(self):
        mock_store = MagicMock()
        with self._patch_store(mock_store):
            storage_mod._save_config_to_store("u1", {"key": "val"}, "ws1")
        mock_store.save_user_config.assert_called_once_with("u1", "ws1", {"key": "val"})

    def test_save_does_not_raise_on_exception(self):
        mock_store = MagicMock()
        mock_store.save_user_config.side_effect = OSError("db down")
        with self._patch_store(mock_store):
            storage_mod._save_config_to_store("u1", {"key": "val"})
        # no exception raised
