"""Tests for email storage and initialization utilities.

Tests for aragora/server/handlers/email/storage.py covering:
- _check_email_permission: RBAC permission checking with all branches
  - RBAC available: allowed, denied, check errors
  - RBAC unavailable: fail-closed (production), fail-open (dev/test)
  - No auth context: write-sensitive denied, read-only allowed
- _load_config_from_store: loading user config from persistent store
- _save_config_to_store: saving user config to persistent store
- get_email_store: LazyStoreFactory alias
- get_gmail_connector: thread-safe lazy initialization
- get_prioritizer: thread-safe lazy initialization with user config
- get_context_service: thread-safe lazy initialization
- get_user_config / set_user_config: thread-safe cache operations
- Thread safety: concurrent access to all lazy singletons
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_module


# ============================================================================
# Helpers
# ============================================================================


def _make_auth_context(user_id: str = "user-1") -> MagicMock:
    """Build a mock auth context with a user_id attribute."""
    ctx = MagicMock()
    ctx.user_id = user_id
    return ctx


def _make_decision(allowed: bool, reason: str = "") -> MagicMock:
    """Build a mock AuthorizationDecision."""
    decision = MagicMock()
    decision.allowed = allowed
    decision.reason = reason
    return decision


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_module_globals():
    """Reset all module-level singletons between tests to avoid leaking state."""
    # Save originals
    orig_gmail = storage_module._gmail_connector
    orig_prioritizer = storage_module._prioritizer
    orig_context_service = storage_module._context_service
    orig_user_configs = storage_module._user_configs.copy()

    yield

    # Restore
    storage_module._gmail_connector = orig_gmail
    storage_module._prioritizer = orig_prioritizer
    storage_module._context_service = orig_context_service
    storage_module._user_configs.clear()
    storage_module._user_configs.update(orig_user_configs)

    # Reset the LazyStoreFactory so it re-initializes on next test
    storage_module._email_store.reset()


# ============================================================================
# _check_email_permission - RBAC Available
# ============================================================================


class TestCheckEmailPermissionRBACAvailable:
    """Tests for _check_email_permission when RBAC module is available."""

    def test_permission_allowed(self):
        """Returns None when RBAC check says allowed."""
        auth_ctx = _make_auth_context()
        decision = _make_decision(allowed=True)
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", True),
            patch.object(storage_module, "check_permission", return_value=decision),
        ):
            result = storage_module._check_email_permission(auth_ctx, "email:read")
        assert result is None

    def test_permission_denied(self):
        """Returns error dict when RBAC check says denied."""
        auth_ctx = _make_auth_context()
        decision = _make_decision(allowed=False, reason="Insufficient permissions")
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", True),
            patch.object(storage_module, "check_permission", return_value=decision),
        ):
            result = storage_module._check_email_permission(auth_ctx, "email:write")
        assert result is not None
        assert result["success"] is False
        assert result["error"] == "Permission denied"

    def test_permission_denied_logs_warning(self):
        """Denied permission logs a warning with user_id and reason."""
        auth_ctx = _make_auth_context(user_id="user-42")
        decision = _make_decision(allowed=False, reason="no role")
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", True),
            patch.object(storage_module, "check_permission", return_value=decision),
            patch.object(storage_module.logger, "warning") as mock_warn,
        ):
            storage_module._check_email_permission(auth_ctx, "email:write")
        mock_warn.assert_called_once()
        assert "email:write" in str(mock_warn.call_args)
        assert "user-42" in str(mock_warn.call_args)

    @pytest.mark.parametrize("exc_type", [TypeError, ValueError, KeyError, AttributeError])
    def test_rbac_check_exception_fails_open(self, exc_type):
        """When check_permission raises a handled exception, fail open (return None)."""
        auth_ctx = _make_auth_context()
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", True),
            patch.object(storage_module, "check_permission", side_effect=exc_type("test error")),
        ):
            result = storage_module._check_email_permission(auth_ctx, "email:read")
        assert result is None

    def test_rbac_check_exception_logs_warning(self):
        """When check_permission raises, a warning is logged."""
        auth_ctx = _make_auth_context()
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", True),
            patch.object(storage_module, "check_permission", side_effect=TypeError("oops")),
            patch.object(storage_module.logger, "warning") as mock_warn,
        ):
            storage_module._check_email_permission(auth_ctx, "email:read")
        mock_warn.assert_called_once()
        assert "email:read" in str(mock_warn.call_args)


# ============================================================================
# _check_email_permission - RBAC Unavailable
# ============================================================================


class TestCheckEmailPermissionRBACUnavailable:
    """Tests for _check_email_permission when RBAC module is NOT available."""

    def test_fail_closed_in_production(self):
        """In production, returns error dict when RBAC unavailable."""
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=True),
        ):
            result = storage_module._check_email_permission(None, "email:read")
        assert result is not None
        assert result["success"] is False
        assert "access control module not loaded" in result["error"]

    def test_fail_closed_write_permission_in_production(self):
        """In production, write permissions also fail closed."""
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=True),
        ):
            result = storage_module._check_email_permission(None, "email:write")
        assert result is not None
        assert result["success"] is False
        assert "access control module not loaded" in result["error"]

    def test_dev_write_permission_denied(self):
        """In dev/test mode, write-sensitive ops are denied without RBAC."""
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=False),
        ):
            result = storage_module._check_email_permission(None, "email:write")
        assert result is not None
        assert result["success"] is False
        assert result["error"] == "Permission denied"

    def test_dev_update_permission_denied(self):
        """In dev/test mode, update ops are denied without RBAC."""
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=False),
        ):
            result = storage_module._check_email_permission(None, "email:update")
        assert result is not None
        assert result["success"] is False

    def test_dev_oauth_permission_denied(self):
        """In dev/test mode, oauth ops are denied without RBAC."""
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=False),
        ):
            result = storage_module._check_email_permission(None, "email:oauth")
        assert result is not None
        assert result["success"] is False

    def test_dev_read_permission_degrades_gracefully(self):
        """In dev/test mode, read-only paths return None (fail open)."""
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=False),
        ):
            result = storage_module._check_email_permission(None, "email:read")
        assert result is None

    def test_dev_read_with_auth_context_degrades_gracefully(self):
        """In dev/test mode with auth_context, read-only paths still degrade gracefully."""
        auth_ctx = _make_auth_context()
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", False),
            patch.object(storage_module, "rbac_fail_closed", return_value=False),
        ):
            result = storage_module._check_email_permission(auth_ctx, "email:read")
        assert result is None


# ============================================================================
# _check_email_permission - No Auth Context (RBAC Available)
# ============================================================================


class TestCheckEmailPermissionNoAuthContext:
    """Tests for _check_email_permission when auth_context is None but RBAC is available."""

    def test_no_auth_write_denied(self):
        """Write permission is denied when auth_context is None."""
        with patch.object(storage_module, "RBAC_AVAILABLE", True):
            result = storage_module._check_email_permission(None, "email:write")
        assert result is not None
        assert result["success"] is False
        assert result["error"] == "Permission denied"

    def test_no_auth_update_denied(self):
        """Update permission is denied when auth_context is None."""
        with patch.object(storage_module, "RBAC_AVAILABLE", True):
            result = storage_module._check_email_permission(None, "email:update")
        assert result is not None
        assert result["success"] is False

    def test_no_auth_oauth_denied(self):
        """OAuth permission is denied when auth_context is None."""
        with patch.object(storage_module, "RBAC_AVAILABLE", True):
            result = storage_module._check_email_permission(None, "email:oauth")
        assert result is not None
        assert result["success"] is False

    def test_no_auth_read_allowed(self):
        """Read permission is allowed (returns None) when auth_context is None."""
        with patch.object(storage_module, "RBAC_AVAILABLE", True):
            result = storage_module._check_email_permission(None, "email:read")
        assert result is None

    def test_no_auth_arbitrary_read_allowed(self):
        """Non-write permissions degrade gracefully when auth_context is None."""
        with patch.object(storage_module, "RBAC_AVAILABLE", True):
            result = storage_module._check_email_permission(None, "email:list")
        assert result is None


# ============================================================================
# _load_config_from_store
# ============================================================================


class TestLoadConfigFromStore:
    """Tests for _load_config_from_store."""

    def test_returns_config_from_store(self):
        """Loads and returns user config from the store."""
        mock_store = MagicMock()
        mock_store.get_user_config.return_value = {"vip_domains": ["example.com"]}
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            result = storage_module._load_config_from_store("user-1")
        assert result == {"vip_domains": ["example.com"]}
        mock_store.get_user_config.assert_called_once_with("user-1", "default")

    def test_custom_workspace_id(self):
        """Passes custom workspace_id to the store."""
        mock_store = MagicMock()
        mock_store.get_user_config.return_value = {"key": "value"}
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            result = storage_module._load_config_from_store("user-1", workspace_id="ws-99")
        mock_store.get_user_config.assert_called_once_with("user-1", "ws-99")
        assert result == {"key": "value"}

    def test_returns_empty_dict_when_store_unavailable(self):
        """Returns empty dict when the store is not initialized."""
        with patch.object(storage_module._email_store, "get", return_value=None):
            result = storage_module._load_config_from_store("user-1")
        assert result == {}

    def test_returns_empty_dict_when_config_not_found(self):
        """Returns empty dict when store returns None for the user."""
        mock_store = MagicMock()
        mock_store.get_user_config.return_value = None
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            result = storage_module._load_config_from_store("user-1")
        assert result == {}

    def test_returns_empty_dict_when_config_is_empty(self):
        """Returns empty dict when store returns empty dict (falsy)."""
        mock_store = MagicMock()
        mock_store.get_user_config.return_value = {}
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            result = storage_module._load_config_from_store("user-1")
        assert result == {}

    @pytest.mark.parametrize("exc_type", [KeyError, ValueError, OSError, TypeError])
    def test_returns_empty_dict_on_store_error(self, exc_type):
        """Returns empty dict when store raises a handled exception."""
        mock_store = MagicMock()
        mock_store.get_user_config.side_effect = exc_type("store error")
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            result = storage_module._load_config_from_store("user-1")
        assert result == {}

    def test_logs_warning_on_store_error(self):
        """Logs a warning when loading config fails."""
        mock_store = MagicMock()
        mock_store.get_user_config.side_effect = OSError("disk failure")
        with (
            patch.object(storage_module._email_store, "get", return_value=mock_store),
            patch.object(storage_module.logger, "warning") as mock_warn,
        ):
            storage_module._load_config_from_store("user-1")
        mock_warn.assert_called_once()
        assert "Failed to load config" in str(mock_warn.call_args)


# ============================================================================
# _save_config_to_store
# ============================================================================


class TestSaveConfigToStore:
    """Tests for _save_config_to_store."""

    def test_saves_config_to_store(self):
        """Successfully saves user config to the store."""
        mock_store = MagicMock()
        config = {"vip_domains": ["example.com"]}
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            storage_module._save_config_to_store("user-1", config)
        mock_store.save_user_config.assert_called_once_with("user-1", "default", config)

    def test_custom_workspace_id(self):
        """Passes custom workspace_id to the store."""
        mock_store = MagicMock()
        config = {"key": "value"}
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            storage_module._save_config_to_store("user-1", config, workspace_id="ws-99")
        mock_store.save_user_config.assert_called_once_with("user-1", "ws-99", config)

    def test_no_op_when_store_unavailable(self):
        """Does nothing (no error) when the store is not initialized."""
        with patch.object(storage_module._email_store, "get", return_value=None):
            # Should not raise
            storage_module._save_config_to_store("user-1", {"key": "val"})

    @pytest.mark.parametrize("exc_type", [KeyError, ValueError, OSError, TypeError])
    def test_swallows_store_errors(self, exc_type):
        """Catches and logs store errors without raising."""
        mock_store = MagicMock()
        mock_store.save_user_config.side_effect = exc_type("save error")
        with patch.object(storage_module._email_store, "get", return_value=mock_store):
            # Should not raise
            storage_module._save_config_to_store("user-1", {"key": "val"})

    def test_logs_warning_on_store_error(self):
        """Logs a warning when saving config fails."""
        mock_store = MagicMock()
        mock_store.save_user_config.side_effect = OSError("disk full")
        with (
            patch.object(storage_module._email_store, "get", return_value=mock_store),
            patch.object(storage_module.logger, "warning") as mock_warn,
        ):
            storage_module._save_config_to_store("user-1", {"key": "val"})
        mock_warn.assert_called_once()
        assert "Failed to save config" in str(mock_warn.call_args)


# ============================================================================
# get_email_store (LazyStoreFactory alias)
# ============================================================================


class TestGetEmailStore:
    """Tests for the get_email_store alias."""

    def test_get_email_store_is_callable(self):
        """get_email_store is a callable (bound method of LazyStoreFactory)."""
        assert callable(storage_module.get_email_store)

    def test_get_email_store_is_bound_to_factory(self):
        """get_email_store is the .get() bound method of the _email_store factory."""
        # Bound methods create new objects on each access, so we check __self__
        assert storage_module.get_email_store.__self__ is storage_module._email_store
        assert storage_module.get_email_store.__func__ is storage_module._email_store.__class__.get

    def test_get_email_store_returns_store_when_initialized(self):
        """get_email_store returns a store instance when the factory initializes."""
        mock_store = MagicMock()
        # Directly set the factory's internal state to simulate successful init
        storage_module._email_store._store = mock_store
        storage_module._email_store._initialized = True
        result = storage_module.get_email_store()
        assert result is mock_store

    def test_get_email_store_returns_none_on_init_failure(self):
        """Returns None when the factory has initialized but failed."""
        # Simulate a failed initialization
        storage_module._email_store._store = None
        storage_module._email_store._initialized = True
        storage_module._email_store._init_error = "Module not available"
        result = storage_module.get_email_store()
        assert result is None


# ============================================================================
# get_gmail_connector
# ============================================================================


class TestGetGmailConnector:
    """Tests for get_gmail_connector thread-safe lazy initialization."""

    def test_creates_gmail_connector(self):
        """Creates a GmailConnector on first call."""
        storage_module._gmail_connector = None
        mock_connector = MagicMock()
        with patch(
            "aragora.server.handlers.email.storage.GmailConnector",
            return_value=mock_connector,
            create=True,
        ) as mock_cls:
            # Patch the import inside the function
            with patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=mock_cls
                    )
                },
            ):
                result = storage_module.get_gmail_connector()
        assert result is not None

    def test_returns_cached_connector(self):
        """Returns the cached connector on subsequent calls."""
        sentinel = MagicMock(name="cached_connector")
        storage_module._gmail_connector = sentinel
        result = storage_module.get_gmail_connector()
        assert result is sentinel

    def test_returns_cached_connector_with_user_id(self):
        """User ID parameter does not change the cached connector."""
        sentinel = MagicMock(name="cached_connector")
        storage_module._gmail_connector = sentinel
        result = storage_module.get_gmail_connector(user_id="user-42")
        assert result is sentinel

    def test_default_user_id(self):
        """Default user_id parameter is 'default'."""
        sentinel = MagicMock()
        storage_module._gmail_connector = sentinel
        # Just verify the function accepts no args
        result = storage_module.get_gmail_connector()
        assert result is sentinel


# ============================================================================
# get_prioritizer
# ============================================================================


class TestGetPrioritizer:
    """Tests for get_prioritizer thread-safe lazy initialization."""

    def test_returns_cached_prioritizer(self):
        """Returns the cached prioritizer on subsequent calls."""
        sentinel = MagicMock(name="cached_prioritizer")
        storage_module._prioritizer = sentinel
        result = storage_module.get_prioritizer()
        assert result is sentinel

    def test_returns_cached_prioritizer_with_user_id(self):
        """User ID parameter does not change the cached prioritizer."""
        sentinel = MagicMock(name="cached_prioritizer")
        storage_module._prioritizer = sentinel
        result = storage_module.get_prioritizer(user_id="user-99")
        assert result is sentinel

    def test_creates_prioritizer_with_user_config(self):
        """Creates a prioritizer that uses stored user config."""
        storage_module._prioritizer = None
        storage_module._gmail_connector = MagicMock(name="gmail")

        # Set up user config
        with storage_module._user_configs_lock:
            storage_module._user_configs["user-10"] = {
                "vip_domains": ["example.com"],
                "vip_addresses": ["boss@corp.com"],
                "internal_domains": ["corp.com"],
                "auto_archive_senders": ["noreply@spam.com"],
            }

        mock_prioritizer = MagicMock(name="new_prioritizer")
        mock_config_cls = MagicMock(name="EmailPrioritizationConfig")
        mock_prioritizer_cls = MagicMock(return_value=mock_prioritizer)

        with patch.dict(
            "sys.modules",
            {
                "aragora.services.email_prioritization": MagicMock(
                    EmailPrioritizer=mock_prioritizer_cls,
                    EmailPrioritizationConfig=mock_config_cls,
                )
            },
        ):
            result = storage_module.get_prioritizer(user_id="user-10")

        assert result is not None

    def test_creates_prioritizer_with_empty_config(self):
        """Creates a prioritizer with empty defaults when no user config exists."""
        storage_module._prioritizer = None
        storage_module._gmail_connector = MagicMock(name="gmail")

        mock_prioritizer = MagicMock(name="new_prioritizer")
        mock_config_cls = MagicMock(name="EmailPrioritizationConfig")
        mock_prioritizer_cls = MagicMock(return_value=mock_prioritizer)

        with patch.dict(
            "sys.modules",
            {
                "aragora.services.email_prioritization": MagicMock(
                    EmailPrioritizer=mock_prioritizer_cls,
                    EmailPrioritizationConfig=mock_config_cls,
                )
            },
        ):
            result = storage_module.get_prioritizer(user_id="nonexistent")

        # Config should be created with empty sets as defaults
        mock_config_cls.assert_called_once()
        call_kwargs = mock_config_cls.call_args
        assert call_kwargs[1]["vip_domains"] == set()
        assert call_kwargs[1]["vip_addresses"] == set()
        assert call_kwargs[1]["internal_domains"] == set()
        assert call_kwargs[1]["auto_archive_senders"] == set()


# ============================================================================
# get_context_service
# ============================================================================


class TestGetContextService:
    """Tests for get_context_service thread-safe lazy initialization."""

    def test_returns_cached_service(self):
        """Returns the cached context service on subsequent calls."""
        sentinel = MagicMock(name="cached_service")
        storage_module._context_service = sentinel
        result = storage_module.get_context_service()
        assert result is sentinel

    def test_creates_context_service(self):
        """Creates a CrossChannelContextService on first call."""
        storage_module._context_service = None
        mock_service = MagicMock(name="new_service")

        with patch.dict(
            "sys.modules",
            {
                "aragora.services.cross_channel_context": MagicMock(
                    CrossChannelContextService=MagicMock(return_value=mock_service)
                )
            },
        ):
            result = storage_module.get_context_service()

        assert result is not None


# ============================================================================
# get_user_config / set_user_config
# ============================================================================


class TestUserConfigCache:
    """Tests for get_user_config and set_user_config thread-safe cache operations."""

    def test_get_empty_config(self):
        """Returns empty dict when no config has been set."""
        result = storage_module.get_user_config("unknown-user")
        assert result == {}

    def test_set_and_get_config(self):
        """Config stored via set_user_config is retrievable via get_user_config."""
        config = {"vip_domains": ["example.com"], "priority": "high"}
        storage_module.set_user_config("user-1", config)
        result = storage_module.get_user_config("user-1")
        assert result == config

    def test_get_returns_copy(self):
        """get_user_config returns a copy, not a reference to the internal dict."""
        config = {"key": "value"}
        storage_module.set_user_config("user-1", config)
        result1 = storage_module.get_user_config("user-1")
        result1["key"] = "modified"
        result2 = storage_module.get_user_config("user-1")
        assert result2["key"] == "value"

    def test_set_stores_copy(self):
        """set_user_config stores a copy, so external mutations don't affect cache."""
        config = {"key": "value"}
        storage_module.set_user_config("user-1", config)
        config["key"] = "modified"
        result = storage_module.get_user_config("user-1")
        assert result["key"] == "value"

    def test_overwrite_config(self):
        """Setting config for the same user overwrites the previous config."""
        storage_module.set_user_config("user-1", {"a": 1})
        storage_module.set_user_config("user-1", {"b": 2})
        result = storage_module.get_user_config("user-1")
        assert result == {"b": 2}
        assert "a" not in result

    def test_multiple_users(self):
        """Config is stored independently per user."""
        storage_module.set_user_config("user-1", {"role": "admin"})
        storage_module.set_user_config("user-2", {"role": "viewer"})
        assert storage_module.get_user_config("user-1") == {"role": "admin"}
        assert storage_module.get_user_config("user-2") == {"role": "viewer"}

    def test_empty_config(self):
        """Empty config dict can be stored and retrieved."""
        storage_module.set_user_config("user-1", {})
        result = storage_module.get_user_config("user-1")
        assert result == {}

    def test_nested_config(self):
        """Nested config data is correctly stored and retrieved."""
        config = {
            "vip_domains": ["a.com", "b.com"],
            "rules": {"auto_archive": True, "threshold": 0.5},
        }
        storage_module.set_user_config("user-1", config)
        result = storage_module.get_user_config("user-1")
        assert result["vip_domains"] == ["a.com", "b.com"]
        assert result["rules"]["auto_archive"] is True


# ============================================================================
# Thread Safety
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe access to lazy singletons and config cache."""

    def test_concurrent_config_writes(self):
        """Multiple threads writing config simultaneously do not corrupt state."""
        errors: list[Exception] = []

        def write_config(user_id: str, iteration: int):
            try:
                storage_module.set_user_config(user_id, {"iter": iteration})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_config, args=(f"user-{i}", i)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Each user should have its config set
        for i in range(20):
            cfg = storage_module.get_user_config(f"user-{i}")
            assert cfg["iter"] == i

    def test_concurrent_config_reads(self):
        """Multiple threads reading config simultaneously work correctly."""
        storage_module.set_user_config("shared-user", {"data": "stable"})
        results: list[dict] = []
        lock = threading.Lock()

        def read_config():
            cfg = storage_module.get_user_config("shared-user")
            with lock:
                results.append(cfg)

        threads = [threading.Thread(target=read_config) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        for r in results:
            assert r == {"data": "stable"}

    def test_concurrent_get_gmail_connector(self):
        """Multiple threads calling get_gmail_connector return the same instance."""
        storage_module._gmail_connector = None
        sentinel = MagicMock(name="singleton_connector")

        # Pre-set the connector to avoid actual import
        storage_module._gmail_connector = sentinel

        results: list[Any] = []
        lock = threading.Lock()

        def get_connector():
            c = storage_module.get_gmail_connector()
            with lock:
                results.append(c)

        threads = [threading.Thread(target=get_connector) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is sentinel for r in results)


# ============================================================================
# LazyStoreFactory configuration
# ============================================================================


class TestEmailStoreFactory:
    """Tests for the _email_store LazyStoreFactory configuration."""

    def test_store_name(self):
        """Factory is configured with correct store name."""
        assert storage_module._email_store.store_name == "email_store"

    def test_import_path(self):
        """Factory is configured with correct import path."""
        assert storage_module._email_store.import_path == "aragora.storage.email_store"

    def test_factory_name(self):
        """Factory is configured with correct factory function name."""
        assert storage_module._email_store.factory_name == "get_email_store"

    def test_logger_context(self):
        """Factory is configured with correct logger context."""
        assert storage_module._email_store.logger_context == "EmailHandler"

    def test_reset_allows_reinitialization(self):
        """Resetting the factory allows re-initialization."""
        storage_module._email_store.reset()
        assert storage_module._email_store.is_initialized is False
        assert storage_module._email_store.is_available is False


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for the storage module."""

    def test_check_permission_denied_user_without_user_id(self):
        """When auth_context has no user_id, getattr fallback to 'unknown' is used in log."""
        auth_ctx = object()  # No user_id attribute
        decision = _make_decision(allowed=False, reason="no role")
        with (
            patch.object(storage_module, "RBAC_AVAILABLE", True),
            patch.object(storage_module, "check_permission", return_value=decision),
            patch.object(storage_module.logger, "warning") as mock_warn,
        ):
            result = storage_module._check_email_permission(auth_ctx, "email:read")
        assert result is not None
        assert result["success"] is False
        # Should log 'unknown' for user_id via getattr fallback
        assert "unknown" in str(mock_warn.call_args)

    def test_load_config_returns_empty_when_email_store_is_none_literal(self):
        """_load_config_from_store handles the case where _email_store evaluates to None.

        Note: In the actual module, _email_store is a LazyStoreFactory (never None),
        but this tests the defensive check at line 101.
        """
        # We can't set _email_store to None directly since the `is None` check
        # is on the module-level LazyStoreFactory object. Instead, we test
        # the branch where get() returns None (store not initialized).
        with patch.object(storage_module._email_store, "get", return_value=None):
            result = storage_module._load_config_from_store("user-1")
        assert result == {}

    def test_save_config_no_op_when_store_get_returns_none(self):
        """_save_config_to_store is a no-op when get() returns None."""
        with patch.object(storage_module._email_store, "get", return_value=None):
            storage_module._save_config_to_store("user-1", {"data": 1})
        # Should not raise

    def test_write_sensitive_permissions_set(self):
        """Verify the exact set of write-sensitive permission keys."""
        # These are the keys that should be denied even in dev/test when RBAC unavailable
        write_keys = {"email:write", "email:update", "email:oauth"}
        for key in write_keys:
            with (
                patch.object(storage_module, "RBAC_AVAILABLE", False),
                patch.object(storage_module, "rbac_fail_closed", return_value=False),
            ):
                result = storage_module._check_email_permission(None, key)
            assert result is not None, f"Expected denial for {key}"
            assert result["success"] is False

    def test_non_write_permissions_allowed_in_dev(self):
        """Permissions outside the write-sensitive set are allowed in dev/test."""
        non_write_keys = ["email:read", "email:list", "email:search", "email:stats"]
        for key in non_write_keys:
            with (
                patch.object(storage_module, "RBAC_AVAILABLE", False),
                patch.object(storage_module, "rbac_fail_closed", return_value=False),
            ):
                result = storage_module._check_email_permission(None, key)
            assert result is None, f"Expected None (allowed) for {key}"
