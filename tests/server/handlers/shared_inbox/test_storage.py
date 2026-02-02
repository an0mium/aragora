"""Tests for shared inbox storage utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.shared_inbox import storage as storage_mod
from aragora.server.handlers.shared_inbox.models import (
    RoutingRule,
    SharedInbox,
    SharedInboxMessage,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_stores():
    """Reset LazyStoreFactory instances and in-memory storage between tests."""
    storage_mod._email_store.reset()
    storage_mod._rules_store.reset()
    storage_mod._activity_store.reset()
    storage_mod._shared_inboxes.clear()
    storage_mod._inbox_messages.clear()
    storage_mod._routing_rules.clear()
    yield
    storage_mod._email_store.reset()
    storage_mod._rules_store.reset()
    storage_mod._activity_store.reset()
    storage_mod._shared_inboxes.clear()
    storage_mod._inbox_messages.clear()
    storage_mod._routing_rules.clear()


# =============================================================================
# LazyStoreFactory configuration
# =============================================================================


class TestStoreFactoryConfiguration:
    """Verify LazyStoreFactory instances are configured correctly."""

    def test_email_store_config(self):
        assert storage_mod._email_store.store_name == "email_store"
        assert storage_mod._email_store.import_path == "aragora.storage.email_store"
        assert storage_mod._email_store.factory_name == "get_email_store"
        assert storage_mod._email_store.logger_context == "SharedInbox"

    def test_rules_store_config(self):
        assert storage_mod._rules_store.store_name == "rules_store"
        assert storage_mod._rules_store.import_path == "aragora.services.rules_store"
        assert storage_mod._rules_store.factory_name == "get_rules_store"
        assert storage_mod._rules_store.logger_context == "SharedInbox"

    def test_activity_store_config(self):
        assert storage_mod._activity_store.store_name == "activity_store"
        assert storage_mod._activity_store.import_path == "aragora.storage.inbox_activity_store"
        assert storage_mod._activity_store.factory_name == "get_inbox_activity_store"
        assert storage_mod._activity_store.logger_context == "SharedInbox"


# =============================================================================
# Store accessor helpers
# =============================================================================


class TestStoreAccessors:
    """Test _get_email_store, _get_rules_store, _get_activity_store."""

    def test_get_email_store_delegates(self):
        mock_store = MagicMock()
        with patch.object(storage_mod._email_store, "get", return_value=mock_store):
            result = storage_mod._get_email_store()
            assert result is mock_store

    def test_get_rules_store_delegates(self):
        mock_store = MagicMock()
        with patch.object(storage_mod._rules_store, "get", return_value=mock_store):
            result = storage_mod._get_rules_store()
            assert result is mock_store

    def test_get_activity_store_delegates(self):
        mock_store = MagicMock()
        with patch.object(storage_mod._activity_store, "get", return_value=mock_store):
            result = storage_mod._get_activity_store()
            assert result is mock_store


# =============================================================================
# _log_activity
# =============================================================================


class TestLogActivity:
    """Tests for _log_activity helper."""

    def test_logs_when_store_available(self):
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with patch.object(storage_mod, "_get_activity_store", return_value=mock_store):
            with patch(
                "aragora.storage.inbox_activity_store.InboxActivity",
                mock_activity_cls,
            ):
                storage_mod._log_activity(
                    inbox_id="inbox-1",
                    org_id="org-1",
                    actor_id="user-1",
                    action="assign",
                    target_id="msg-1",
                    metadata={"reason": "test"},
                )

                mock_activity_cls.assert_called_once_with(
                    inbox_id="inbox-1",
                    org_id="org-1",
                    actor_id="user-1",
                    action="assign",
                    target_id="msg-1",
                    metadata={"reason": "test"},
                )
                mock_store.log_activity.assert_called_once()

    def test_skips_when_store_none(self):
        with patch.object(storage_mod, "_get_activity_store", return_value=None):
            # Should not raise
            storage_mod._log_activity(
                inbox_id="inbox-1",
                org_id="org-1",
                actor_id="user-1",
                action="assign",
            )

    def test_handles_exception_gracefully(self):
        mock_store = MagicMock()
        mock_store.log_activity.side_effect = RuntimeError("db error")

        with patch.object(storage_mod, "_get_activity_store", return_value=mock_store):
            with patch(
                "aragora.storage.inbox_activity_store.InboxActivity",
                MagicMock(),
            ):
                # Should not raise
                storage_mod._log_activity(
                    inbox_id="inbox-1",
                    org_id="org-1",
                    actor_id="user-1",
                    action="assign",
                )

    def test_default_metadata(self):
        mock_store = MagicMock()
        mock_activity_cls = MagicMock()

        with patch.object(storage_mod, "_get_activity_store", return_value=mock_store):
            with patch(
                "aragora.storage.inbox_activity_store.InboxActivity",
                mock_activity_cls,
            ):
                storage_mod._log_activity(
                    inbox_id="inbox-1",
                    org_id="org-1",
                    actor_id="user-1",
                    action="resolve",
                )

                # metadata defaults to {}
                call_kwargs = mock_activity_cls.call_args.kwargs
                assert call_kwargs["metadata"] == {}


# =============================================================================
# _get_store
# =============================================================================


class TestGetStore:
    """Tests for _get_store persistent storage accessor."""

    def test_returns_none_when_persistent_disabled(self):
        original = storage_mod.USE_PERSISTENT_STORAGE
        try:
            storage_mod.USE_PERSISTENT_STORAGE = False
            result = storage_mod._get_store()
            assert result is None
        finally:
            storage_mod.USE_PERSISTENT_STORAGE = original

    def test_returns_email_store_when_persistent_enabled(self):
        mock_store = MagicMock()
        original = storage_mod.USE_PERSISTENT_STORAGE
        try:
            storage_mod.USE_PERSISTENT_STORAGE = True
            with patch.object(storage_mod, "_get_email_store", return_value=mock_store):
                result = storage_mod._get_store()
                assert result is mock_store
        finally:
            storage_mod.USE_PERSISTENT_STORAGE = original


# =============================================================================
# In-memory storage
# =============================================================================


class TestInMemoryStorage:
    """Verify in-memory storage containers exist and are usable."""

    def test_shared_inboxes_dict(self):
        assert isinstance(storage_mod._shared_inboxes, dict)
        storage_mod._shared_inboxes["inbox-1"] = MagicMock(spec=SharedInbox)
        assert "inbox-1" in storage_mod._shared_inboxes

    def test_inbox_messages_dict(self):
        assert isinstance(storage_mod._inbox_messages, dict)
        storage_mod._inbox_messages["inbox-1"] = {"msg-1": MagicMock(spec=SharedInboxMessage)}
        assert "msg-1" in storage_mod._inbox_messages["inbox-1"]

    def test_routing_rules_dict(self):
        assert isinstance(storage_mod._routing_rules, dict)
        storage_mod._routing_rules["rule-1"] = MagicMock(spec=RoutingRule)
        assert "rule-1" in storage_mod._routing_rules

    def test_storage_lock_exists(self):
        assert hasattr(storage_mod._storage_lock, "acquire")
        assert hasattr(storage_mod._storage_lock, "release")
