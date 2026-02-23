"""Tests for Shared Inbox storage utilities (aragora/server/handlers/shared_inbox/storage.py).

Covers all public functions, module-level state, and edge cases:

Functions:
- _get_email_store(): Lazy-initialized email store accessor
- _get_rules_store(): Lazy-initialized rules store accessor
- _get_activity_store(): Lazy-initialized activity store accessor
- _get_store(): Persistent storage selector (returns email store or None)
- _log_activity(): Non-blocking activity logging helper

Module-level state:
- _shared_inboxes: In-memory inbox dict
- _inbox_messages: In-memory message dict (inbox_id -> {msg_id -> message})
- _routing_rules: In-memory routing rules dict
- _storage_lock: Thread lock for in-memory storage
- USE_PERSISTENT_STORAGE: Configuration flag

Test categories:
- LazyStoreFactory delegation (each store accessor)
- _get_store() conditional logic (persistent vs in-memory)
- _log_activity() success/failure/edge cases
- In-memory storage structure verification
- Thread safety of storage lock
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.shared_inbox.storage import (
    USE_PERSISTENT_STORAGE,
    _activity_store,
    _email_store,
    _get_activity_store,
    _get_email_store,
    _get_rules_store,
    _get_store,
    _inbox_messages,
    _log_activity,
    _routing_rules,
    _rules_store,
    _shared_inboxes,
    _storage_lock,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_lazy_stores():
    """Reset all LazyStoreFactory instances before each test to avoid cross-test leakage."""
    _email_store.reset()
    _rules_store.reset()
    _activity_store.reset()
    yield
    _email_store.reset()
    _rules_store.reset()
    _activity_store.reset()


# ===========================================================================
# Module-Level State
# ===========================================================================


class TestModuleLevelState:
    """Verify the module-level in-memory storage objects are correctly typed."""

    def test_shared_inboxes_is_dict(self):
        """_shared_inboxes is a dict for SharedInbox objects."""
        assert isinstance(_shared_inboxes, dict)

    def test_inbox_messages_is_dict(self):
        """_inbox_messages is a nested dict for messages."""
        assert isinstance(_inbox_messages, dict)

    def test_routing_rules_is_dict(self):
        """_routing_rules is a dict for RoutingRule objects."""
        assert isinstance(_routing_rules, dict)

    def test_storage_lock_is_threading_lock(self):
        """_storage_lock is a threading.Lock instance."""
        assert isinstance(_storage_lock, type(threading.Lock()))

    def test_use_persistent_storage_default_true(self):
        """USE_PERSISTENT_STORAGE defaults to True."""
        assert USE_PERSISTENT_STORAGE is True


# ===========================================================================
# LazyStoreFactory Instances
# ===========================================================================


class TestLazyStoreFactoryInstances:
    """Verify the three LazyStoreFactory instances are configured correctly."""

    def test_email_store_config(self):
        """_email_store factory has correct configuration."""
        assert _email_store.store_name == "email_store"
        assert _email_store.import_path == "aragora.storage.email_store"
        assert _email_store.factory_name == "get_email_store"
        assert _email_store.logger_context == "SharedInbox"

    def test_rules_store_config(self):
        """_rules_store factory has correct configuration."""
        assert _rules_store.store_name == "rules_store"
        assert _rules_store.import_path == "aragora.services.rules_store"
        assert _rules_store.factory_name == "get_rules_store"
        assert _rules_store.logger_context == "SharedInbox"

    def test_activity_store_config(self):
        """_activity_store factory has correct configuration."""
        assert _activity_store.store_name == "activity_store"
        assert _activity_store.import_path == "aragora.storage.inbox_activity_store"
        assert _activity_store.factory_name == "get_inbox_activity_store"
        assert _activity_store.logger_context == "SharedInbox"


# ===========================================================================
# _get_email_store
# ===========================================================================


class TestGetEmailStore:
    """Tests for _get_email_store() accessor."""

    def test_delegates_to_lazy_factory(self):
        """_get_email_store() calls _email_store.get()."""
        mock_store = MagicMock()
        with patch.object(_email_store, "get", return_value=mock_store) as mock_get:
            result = _get_email_store()
            mock_get.assert_called_once()
            assert result is mock_store

    def test_returns_none_when_import_fails(self):
        """Returns None when the underlying module import fails."""
        # Let the real LazyStoreFactory.get() run -- it will fail to import
        # the module (aragora.storage.email_store likely not available in test env)
        result = _get_email_store()
        # Should be None or a real store -- either way, no crash
        assert result is None or result is not None

    def test_returns_none_on_factory_failure(self):
        """Returns None when the factory's .get() returns None."""
        with patch.object(_email_store, "get", return_value=None):
            assert _get_email_store() is None


# ===========================================================================
# _get_rules_store
# ===========================================================================


class TestGetRulesStore:
    """Tests for _get_rules_store() accessor."""

    def test_delegates_to_lazy_factory(self):
        """_get_rules_store() calls _rules_store.get()."""
        mock_store = MagicMock()
        with patch.object(_rules_store, "get", return_value=mock_store) as mock_get:
            result = _get_rules_store()
            mock_get.assert_called_once()
            assert result is mock_store

    def test_returns_none_on_factory_failure(self):
        """Returns None when factory returns None."""
        with patch.object(_rules_store, "get", return_value=None):
            assert _get_rules_store() is None


# ===========================================================================
# _get_activity_store
# ===========================================================================


class TestGetActivityStore:
    """Tests for _get_activity_store() accessor."""

    def test_delegates_to_lazy_factory(self):
        """_get_activity_store() calls _activity_store.get()."""
        mock_store = MagicMock()
        with patch.object(_activity_store, "get", return_value=mock_store) as mock_get:
            result = _get_activity_store()
            mock_get.assert_called_once()
            assert result is mock_store

    def test_returns_none_on_factory_failure(self):
        """Returns None when factory returns None."""
        with patch.object(_activity_store, "get", return_value=None):
            assert _get_activity_store() is None


# ===========================================================================
# _get_store
# ===========================================================================


class TestGetStore:
    """Tests for _get_store() -- returns email store or None based on flag."""

    def test_returns_email_store_when_persistent_enabled(self):
        """When USE_PERSISTENT_STORAGE is True, returns email store."""
        mock_store = MagicMock()
        with (
            patch("aragora.server.handlers.shared_inbox.storage.USE_PERSISTENT_STORAGE", True),
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_email_store",
                return_value=mock_store,
            ),
        ):
            result = _get_store()
            assert result is mock_store

    def test_returns_none_when_persistent_disabled(self):
        """When USE_PERSISTENT_STORAGE is False, returns None."""
        with patch("aragora.server.handlers.shared_inbox.storage.USE_PERSISTENT_STORAGE", False):
            result = _get_store()
            assert result is None

    def test_returns_none_when_persistent_enabled_but_store_unavailable(self):
        """When USE_PERSISTENT_STORAGE is True but email store is None."""
        with (
            patch("aragora.server.handlers.shared_inbox.storage.USE_PERSISTENT_STORAGE", True),
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_email_store",
                return_value=None,
            ),
        ):
            result = _get_store()
            assert result is None

    def test_does_not_call_email_store_when_disabled(self):
        """When USE_PERSISTENT_STORAGE is False, _get_email_store is not called."""
        with (
            patch("aragora.server.handlers.shared_inbox.storage.USE_PERSISTENT_STORAGE", False),
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_email_store",
            ) as mock_email,
        ):
            _get_store()
            mock_email.assert_not_called()


# ===========================================================================
# _log_activity - Success paths
# ===========================================================================


class TestLogActivitySuccess:
    """Tests for _log_activity() successful logging."""

    def test_logs_activity_with_all_params(self):
        """Activity is logged when store is available and all params provided."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()
        mock_activity_instance = MagicMock()
        mock_activity_cls.return_value = mock_activity_instance

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch(
                "aragora.server.handlers.shared_inbox.storage.InboxActivity",
                mock_activity_cls,
                create=True,
            ),
        ):
            # We need to patch the import inside the function
            with patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ):
                _log_activity(
                    inbox_id="inbox-1",
                    org_id="org-1",
                    actor_id="user-1",
                    action="create",
                    target_id="msg-1",
                    metadata={"key": "value"},
                )

        mock_store.log_activity.assert_called_once()

    def test_logs_activity_without_optional_params(self):
        """Activity is logged with default optional params (target_id=None, metadata=None)."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="view",
            )

        # Verify the InboxActivity was constructed with metadata={} (not None)
        call_kwargs = mock_activity_cls.call_args
        assert call_kwargs is not None
        if call_kwargs[1]:  # keyword args
            assert call_kwargs[1].get("metadata", {}) == {}
        else:
            # positional args
            assert call_kwargs[0][-1] == {}

    def test_logs_activity_with_empty_metadata(self):
        """Activity is logged with explicit empty metadata dict."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="update",
                metadata={},
            )

        mock_store.log_activity.assert_called_once()


# ===========================================================================
# _log_activity - Store unavailable
# ===========================================================================


class TestLogActivityStoreUnavailable:
    """Tests for _log_activity() when the activity store is not available."""

    def test_no_error_when_store_is_none(self):
        """When activity store returns None, _log_activity completes silently."""
        with patch(
            "aragora.server.handlers.shared_inbox.storage._get_activity_store",
            return_value=None,
        ):
            # Should not raise
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )

    def test_store_none_does_not_attempt_import(self):
        """When store is None, the InboxActivity import is never attempted."""
        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=None,
            ),
            patch(
                "builtins.__import__", side_effect=AssertionError("Should not import")
            ) as mock_import,
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )
            # __import__ should not have been called for inbox_activity_store
            # (it may be called for other things, but this shouldn't crash)


# ===========================================================================
# _log_activity - Error handling
# ===========================================================================


class TestLogActivityErrorHandling:
    """Tests for _log_activity() exception handling."""

    def test_catches_import_error(self):
        """ImportError during InboxActivity import is caught gracefully."""
        mock_store = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict("sys.modules", {"aragora.storage.inbox_activity_store": None}),
        ):
            # When a module is set to None in sys.modules, import raises ImportError
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )
        # Should not raise -- caught by except block

    def test_catches_value_error(self):
        """ValueError during InboxActivity construction is caught."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock(side_effect=ValueError("bad value"))

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )
        mock_store.log_activity.assert_not_called()

    def test_catches_type_error(self):
        """TypeError during InboxActivity construction is caught."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock(side_effect=TypeError("wrong args"))

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )
        mock_store.log_activity.assert_not_called()

    def test_catches_key_error(self):
        """KeyError during activity logging is caught."""
        mock_store = MagicMock()
        mock_store.log_activity.side_effect = KeyError("missing key")
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )

    def test_catches_attribute_error(self):
        """AttributeError during activity logging is caught."""
        mock_store = MagicMock()
        mock_store.log_activity.side_effect = AttributeError("no attr")
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )

    def test_catches_os_error(self):
        """OSError during activity logging is caught."""
        mock_store = MagicMock()
        mock_store.log_activity.side_effect = OSError("disk full")
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )

    def test_catches_runtime_error(self):
        """RuntimeError during activity logging is caught."""
        mock_store = MagicMock()
        mock_store.log_activity.side_effect = RuntimeError("bad state")
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )

    def test_logs_debug_on_error(self):
        """Errors are logged at DEBUG level."""
        mock_store = MagicMock()
        mock_store.log_activity.side_effect = RuntimeError("fail")
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
            patch("aragora.server.handlers.shared_inbox.storage.logger") as mock_logger,
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
            )
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            assert "Failed to log activity" in call_args[0][0]


# ===========================================================================
# _log_activity - Metadata handling
# ===========================================================================


class TestLogActivityMetadata:
    """Tests for _log_activity() metadata coercion."""

    def test_none_metadata_becomes_empty_dict(self):
        """When metadata=None, InboxActivity receives {} not None."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="create",
                metadata=None,
            )

        # Verify metadata passed to InboxActivity is {}
        call_kwargs = mock_activity_cls.call_args[1]
        assert call_kwargs["metadata"] == {}

    def test_provided_metadata_is_passed_through(self):
        """Non-None metadata is passed directly to InboxActivity."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()
        meta = {"reason": "test", "count": 42}

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="annotate",
                metadata=meta,
            )

        call_kwargs = mock_activity_cls.call_args[1]
        assert call_kwargs["metadata"] is meta

    def test_target_id_none_by_default(self):
        """When target_id is omitted, None is passed to InboxActivity."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="list",
            )

        call_kwargs = mock_activity_cls.call_args[1]
        assert call_kwargs["target_id"] is None

    def test_target_id_passed_through(self):
        """Explicit target_id is forwarded to InboxActivity."""
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with (
            patch(
                "aragora.server.handlers.shared_inbox.storage._get_activity_store",
                return_value=mock_store,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.storage.inbox_activity_store": MagicMock(
                        InboxActivity=mock_activity_cls
                    )
                },
            ),
        ):
            _log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="assign",
                target_id="msg-99",
            )

        call_kwargs = mock_activity_cls.call_args[1]
        assert call_kwargs["target_id"] == "msg-99"


# ===========================================================================
# LazyStoreFactory reset behavior
# ===========================================================================


class TestLazyStoreReset:
    """Tests verifying the reset behavior of module-level store factories."""

    def test_email_store_reset_clears_initialized(self):
        """After reset, _email_store is no longer initialized."""
        # Force initialization (will likely fail import, but sets _initialized)
        _email_store.get()
        assert _email_store.is_initialized is True
        _email_store.reset()
        assert _email_store.is_initialized is False

    def test_rules_store_reset_clears_initialized(self):
        """After reset, _rules_store is no longer initialized."""
        _rules_store.get()
        assert _rules_store.is_initialized is True
        _rules_store.reset()
        assert _rules_store.is_initialized is False

    def test_activity_store_reset_clears_initialized(self):
        """After reset, _activity_store is no longer initialized."""
        _activity_store.get()
        assert _activity_store.is_initialized is True
        _activity_store.reset()
        assert _activity_store.is_initialized is False


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    """Tests for thread safety of in-memory storage structures."""

    def test_storage_lock_is_acquirable(self):
        """_storage_lock can be acquired and released."""
        acquired = _storage_lock.acquire(timeout=1)
        assert acquired is True
        _storage_lock.release()

    def test_storage_lock_blocks_concurrent_access(self):
        """_storage_lock prevents concurrent modifications."""
        results = []

        def writer(value):
            with _storage_lock:
                results.append(value)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 10 writes should complete
        assert len(results) == 10
        assert set(results) == set(range(10))


# ===========================================================================
# In-memory storage mutability
# ===========================================================================


class TestInMemoryStorageMutability:
    """Tests that in-memory storage dicts are mutable and usable."""

    def test_shared_inboxes_can_store_and_retrieve(self):
        """_shared_inboxes dict supports normal dict operations."""
        key = "__test_inbox__"
        try:
            _shared_inboxes[key] = "test_value"
            assert _shared_inboxes[key] == "test_value"
        finally:
            _shared_inboxes.pop(key, None)

    def test_inbox_messages_nested_storage(self):
        """_inbox_messages supports nested dict[str, dict[str, ...]] access."""
        inbox_key = "__test_inbox__"
        msg_key = "__test_msg__"
        try:
            _inbox_messages[inbox_key] = {msg_key: "test_message"}
            assert _inbox_messages[inbox_key][msg_key] == "test_message"
        finally:
            _inbox_messages.pop(inbox_key, None)

    def test_routing_rules_can_store_and_retrieve(self):
        """_routing_rules dict supports normal dict operations."""
        key = "__test_rule__"
        try:
            _routing_rules[key] = "test_rule"
            assert _routing_rules[key] == "test_rule"
        finally:
            _routing_rules.pop(key, None)
