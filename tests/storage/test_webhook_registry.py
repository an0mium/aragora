"""
Tests for Webhook Registration Storage.

Tests cover:
- Register new webhook endpoint
- Update existing webhook
- Deactivate webhook
- List webhooks with filtering
- Delivery tracking updates
- Secret generation for webhook signing
- Multi-tenant isolation (tenant A cannot see tenant B's webhooks)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.storage.webhook_registry import (
    WebhookConfig,
    SQLiteWebhookRegistry,
    get_webhook_registry,
    set_webhook_registry,
    reset_webhook_registry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def webhook_registry(tmp_path):
    """Create a SQLite webhook registry in a temporary directory."""
    db_path = tmp_path / "webhook_registry_test.db"
    registry = SQLiteWebhookRegistry(db_path=db_path)
    yield registry
    registry.close()


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global webhook registry after each test."""
    yield
    reset_webhook_registry()


# =============================================================================
# Test: WebhookConfig dataclass
# =============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_to_dict_excludes_secret_by_default(self):
        """to_dict should exclude secret by default."""
        config = WebhookConfig(
            id="wh_123",
            url="https://example.com/webhook",
            events=["payment.success"],
            secret="secret_abc123",
        )
        result = config.to_dict()

        assert "secret" not in result
        assert result["id"] == "wh_123"
        assert result["url"] == "https://example.com/webhook"

    def test_to_dict_includes_secret_when_requested(self):
        """to_dict should include secret when include_secret=True."""
        config = WebhookConfig(
            id="wh_123",
            url="https://example.com/webhook",
            events=["payment.success"],
            secret="secret_abc123",
        )
        result = config.to_dict(include_secret=True)

        assert result["secret"] == "secret_abc123"

    def test_matches_event_when_active(self):
        """matches_event should return True for matching events."""
        config = WebhookConfig(
            id="wh_123",
            url="https://example.com/webhook",
            events=["payment.success", "payment.failed"],
            secret="secret",
            active=True,
        )

        assert config.matches_event("payment.success") is True
        assert config.matches_event("payment.failed") is True
        assert config.matches_event("payment.refunded") is False

    def test_matches_event_wildcard(self):
        """matches_event should match all events with * wildcard."""
        config = WebhookConfig(
            id="wh_123",
            url="https://example.com/webhook",
            events=["*"],
            secret="secret",
            active=True,
        )

        assert config.matches_event("payment.success") is True
        assert config.matches_event("any.event.type") is True

    def test_matches_event_returns_false_when_inactive(self):
        """matches_event should return False when webhook is inactive."""
        config = WebhookConfig(
            id="wh_123",
            url="https://example.com/webhook",
            events=["payment.success"],
            secret="secret",
            active=False,
        )

        assert config.matches_event("payment.success") is False


# =============================================================================
# Test: Register new webhook endpoint
# =============================================================================


class TestRegisterWebhook:
    """Tests for registering new webhook endpoints."""

    def test_register_creates_webhook(self, webhook_registry):
        """Should create a new webhook registration."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["payment.success"],
        )

        assert webhook.id is not None
        assert webhook.url == "https://example.com/webhook"
        assert webhook.events == ["payment.success"]
        assert webhook.active is True

    def test_register_generates_unique_id(self, webhook_registry):
        """Each registration should have a unique ID."""
        webhook1 = webhook_registry.register(
            url="https://example.com/webhook1",
            events=["event1"],
        )
        webhook2 = webhook_registry.register(
            url="https://example.com/webhook2",
            events=["event2"],
        )

        assert webhook1.id != webhook2.id

    def test_register_with_name_and_description(self, webhook_registry):
        """Should store name and description."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["payment.success"],
            name="Payment Notifications",
            description="Receives all payment-related events",
        )

        assert webhook.name == "Payment Notifications"
        assert webhook.description == "Receives all payment-related events"

    def test_register_with_user_id(self, webhook_registry):
        """Should store user_id for ownership."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
            user_id="user_123",
        )

        assert webhook.user_id == "user_123"

    def test_register_with_workspace_id(self, webhook_registry):
        """Should store workspace_id for organization."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
            workspace_id="ws_456",
        )

        assert webhook.workspace_id == "ws_456"

    def test_register_sets_timestamps(self, webhook_registry):
        """Should set created_at and updated_at timestamps."""
        before = time.time()
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )
        after = time.time()

        assert before <= webhook.created_at <= after
        assert before <= webhook.updated_at <= after

    def test_register_persists_to_database(self, webhook_registry):
        """Registered webhook should be retrievable."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["payment.success"],
        )

        retrieved = webhook_registry.get(webhook.id)
        assert retrieved is not None
        assert retrieved.url == webhook.url
        assert retrieved.events == webhook.events


# =============================================================================
# Test: Update existing webhook
# =============================================================================


class TestUpdateWebhook:
    """Tests for updating existing webhooks."""

    def test_update_url(self, webhook_registry):
        """Should update webhook URL."""
        webhook = webhook_registry.register(
            url="https://old-url.com/webhook",
            events=["*"],
        )

        updated = webhook_registry.update(webhook.id, url="https://new-url.com/webhook")

        assert updated is not None
        assert updated.url == "https://new-url.com/webhook"

    def test_update_events(self, webhook_registry):
        """Should update webhook events."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["event1"],
        )

        updated = webhook_registry.update(webhook.id, events=["event1", "event2", "event3"])

        assert updated is not None
        assert updated.events == ["event1", "event2", "event3"]

    def test_update_name(self, webhook_registry):
        """Should update webhook name."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
            name="Old Name",
        )

        updated = webhook_registry.update(webhook.id, name="New Name")

        assert updated is not None
        assert updated.name == "New Name"

    def test_update_description(self, webhook_registry):
        """Should update webhook description."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )

        updated = webhook_registry.update(webhook.id, description="Updated description")

        assert updated is not None
        assert updated.description == "Updated description"

    def test_update_updates_timestamp(self, webhook_registry):
        """Update should update the updated_at timestamp."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )
        original_updated_at = webhook.updated_at

        time.sleep(0.01)
        updated = webhook_registry.update(webhook.id, name="Updated")

        assert updated is not None
        assert updated.updated_at > original_updated_at

    def test_update_nonexistent_returns_none(self, webhook_registry):
        """Updating non-existent webhook should return None."""
        result = webhook_registry.update("nonexistent_id", url="https://new.com")
        assert result is None

    def test_update_multiple_fields(self, webhook_registry):
        """Should update multiple fields at once."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )

        updated = webhook_registry.update(
            webhook.id,
            url="https://new.com/webhook",
            events=["new_event"],
            name="New Name",
        )

        assert updated is not None
        assert updated.url == "https://new.com/webhook"
        assert updated.events == ["new_event"]
        assert updated.name == "New Name"


# =============================================================================
# Test: Deactivate webhook
# =============================================================================


class TestDeactivateWebhook:
    """Tests for deactivating webhooks."""

    def test_deactivate_via_update(self, webhook_registry):
        """Should deactivate webhook via update."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )
        assert webhook.active is True

        updated = webhook_registry.update(webhook.id, active=False)

        assert updated is not None
        assert updated.active is False

    def test_deactivated_webhook_not_in_active_only_list(self, webhook_registry):
        """Deactivated webhooks should not appear in active_only listings."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )
        webhook_registry.update(webhook.id, active=False)

        active_webhooks = webhook_registry.list(active_only=True)
        webhook_ids = [w.id for w in active_webhooks]

        assert webhook.id not in webhook_ids

    def test_reactivate_webhook(self, webhook_registry):
        """Should be able to reactivate a deactivated webhook."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )
        webhook_registry.update(webhook.id, active=False)
        webhook_registry.update(webhook.id, active=True)

        retrieved = webhook_registry.get(webhook.id)
        assert retrieved is not None
        assert retrieved.active is True

    def test_delete_removes_webhook(self, webhook_registry):
        """Delete should remove webhook entirely."""
        webhook = webhook_registry.register(
            url="https://example.com/webhook",
            events=["*"],
        )

        deleted = webhook_registry.delete(webhook.id)
        assert deleted is True

        retrieved = webhook_registry.get(webhook.id)
        assert retrieved is None

    def test_delete_nonexistent_returns_false(self, webhook_registry):
        """Deleting non-existent webhook should return False."""
        deleted = webhook_registry.delete("nonexistent_id")
        assert deleted is False


# =============================================================================
# Test: List webhooks with filtering
# =============================================================================


class TestListWebhooks:
    """Tests for listing webhooks with filtering."""

    def test_list_all_webhooks(self, webhook_registry):
        """Should list all webhooks."""
        webhook_registry.register(url="https://a.com", events=["*"])
        webhook_registry.register(url="https://b.com", events=["*"])
        webhook_registry.register(url="https://c.com", events=["*"])

        webhooks = webhook_registry.list()
        assert len(webhooks) == 3

    def test_list_filter_by_user_id(self, webhook_registry):
        """Should filter by user_id."""
        webhook_registry.register(url="https://a.com", events=["*"], user_id="user_1")
        webhook_registry.register(url="https://b.com", events=["*"], user_id="user_1")
        webhook_registry.register(url="https://c.com", events=["*"], user_id="user_2")

        user1_webhooks = webhook_registry.list(user_id="user_1")
        assert len(user1_webhooks) == 2
        assert all(w.user_id == "user_1" for w in user1_webhooks)

    def test_list_filter_by_workspace_id(self, webhook_registry):
        """Should filter by workspace_id."""
        webhook_registry.register(url="https://a.com", events=["*"], workspace_id="ws_1")
        webhook_registry.register(url="https://b.com", events=["*"], workspace_id="ws_2")
        webhook_registry.register(url="https://c.com", events=["*"], workspace_id="ws_1")

        ws1_webhooks = webhook_registry.list(workspace_id="ws_1")
        assert len(ws1_webhooks) == 2
        assert all(w.workspace_id == "ws_1" for w in ws1_webhooks)

    def test_list_filter_active_only(self, webhook_registry):
        """Should filter to active webhooks only."""
        wh1 = webhook_registry.register(url="https://a.com", events=["*"])
        wh2 = webhook_registry.register(url="https://b.com", events=["*"])
        webhook_registry.update(wh2.id, active=False)

        active = webhook_registry.list(active_only=True)
        assert len(active) == 1
        assert active[0].id == wh1.id

    def test_list_combined_filters(self, webhook_registry):
        """Should combine multiple filters."""
        webhook_registry.register(
            url="https://a.com", events=["*"], user_id="u1", workspace_id="ws1"
        )
        webhook_registry.register(
            url="https://b.com", events=["*"], user_id="u1", workspace_id="ws2"
        )
        webhook_registry.register(
            url="https://c.com", events=["*"], user_id="u2", workspace_id="ws1"
        )

        result = webhook_registry.list(user_id="u1", workspace_id="ws1")
        assert len(result) == 1
        assert result[0].url == "https://a.com"

    def test_list_returns_newest_first(self, webhook_registry):
        """List should return webhooks ordered by created_at descending."""
        wh1 = webhook_registry.register(url="https://first.com", events=["*"])
        time.sleep(0.01)
        wh2 = webhook_registry.register(url="https://second.com", events=["*"])
        time.sleep(0.01)
        wh3 = webhook_registry.register(url="https://third.com", events=["*"])

        webhooks = webhook_registry.list()
        assert webhooks[0].id == wh3.id
        assert webhooks[1].id == wh2.id
        assert webhooks[2].id == wh1.id

    def test_list_all_alias(self, webhook_registry):
        """list_all should be an alias for list."""
        webhook_registry.register(url="https://a.com", events=["*"], user_id="u1")
        webhook_registry.register(url="https://b.com", events=["*"], user_id="u2")

        result = webhook_registry.list_all(user_id="u1")
        assert len(result) == 1


# =============================================================================
# Test: Delivery tracking updates
# =============================================================================


class TestDeliveryTracking:
    """Tests for delivery tracking updates."""

    def test_record_successful_delivery(self, webhook_registry):
        """Should record successful delivery."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])

        webhook_registry.record_delivery(webhook.id, status_code=200, success=True)

        updated = webhook_registry.get(webhook.id)
        assert updated is not None
        assert updated.last_delivery_status == 200
        assert updated.last_delivery_at is not None
        assert updated.delivery_count == 1
        assert updated.failure_count == 0

    def test_record_failed_delivery(self, webhook_registry):
        """Should record failed delivery."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])

        webhook_registry.record_delivery(webhook.id, status_code=500, success=False)

        updated = webhook_registry.get(webhook.id)
        assert updated is not None
        assert updated.last_delivery_status == 500
        assert updated.delivery_count == 1
        assert updated.failure_count == 1

    def test_delivery_count_increments(self, webhook_registry):
        """Delivery count should increment with each delivery."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])

        webhook_registry.record_delivery(webhook.id, status_code=200, success=True)
        webhook_registry.record_delivery(webhook.id, status_code=200, success=True)
        webhook_registry.record_delivery(webhook.id, status_code=200, success=True)

        updated = webhook_registry.get(webhook.id)
        assert updated is not None
        assert updated.delivery_count == 3

    def test_failure_count_increments_only_on_failure(self, webhook_registry):
        """Failure count should only increment on failures."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])

        webhook_registry.record_delivery(webhook.id, status_code=200, success=True)
        webhook_registry.record_delivery(webhook.id, status_code=500, success=False)
        webhook_registry.record_delivery(webhook.id, status_code=200, success=True)
        webhook_registry.record_delivery(webhook.id, status_code=503, success=False)

        updated = webhook_registry.get(webhook.id)
        assert updated is not None
        assert updated.delivery_count == 4
        assert updated.failure_count == 2


# =============================================================================
# Test: Secret generation for webhook signing
# =============================================================================


class TestSecretGeneration:
    """Tests for webhook secret generation."""

    def test_register_generates_secret(self, webhook_registry):
        """Registration should generate a secret."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])

        assert webhook.secret is not None
        assert len(webhook.secret) > 0

    def test_secrets_are_unique(self, webhook_registry):
        """Each webhook should have a unique secret."""
        wh1 = webhook_registry.register(url="https://a.com", events=["*"])
        wh2 = webhook_registry.register(url="https://b.com", events=["*"])
        wh3 = webhook_registry.register(url="https://c.com", events=["*"])

        secrets = {wh1.secret, wh2.secret, wh3.secret}
        assert len(secrets) == 3  # All unique

    def test_secret_is_cryptographically_random(self, webhook_registry):
        """Secrets should be cryptographically random (32 bytes = 43 chars in base64)."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])

        # secrets.token_urlsafe(32) produces ~43 characters
        assert len(webhook.secret) >= 40

    def test_secret_persisted_to_database(self, webhook_registry):
        """Secret should be persisted and retrievable."""
        webhook = webhook_registry.register(url="https://example.com", events=["*"])
        original_secret = webhook.secret

        retrieved = webhook_registry.get(webhook.id)
        assert retrieved is not None
        assert retrieved.secret == original_secret


# =============================================================================
# Test: Multi-tenant isolation
# =============================================================================


class TestMultiTenantIsolation:
    """Tests for multi-tenant isolation."""

    def test_tenant_a_cannot_list_tenant_b_webhooks(self, webhook_registry):
        """Tenant A should not see Tenant B's webhooks in listings."""
        # Create webhooks for different tenants
        webhook_registry.register(
            url="https://tenant-a.com",
            events=["*"],
            workspace_id="tenant_a",
        )
        webhook_registry.register(
            url="https://tenant-b.com",
            events=["*"],
            workspace_id="tenant_b",
        )

        # Tenant A lists their webhooks
        tenant_a_webhooks = webhook_registry.list(workspace_id="tenant_a")

        # Should only see their own webhooks
        assert len(tenant_a_webhooks) == 1
        assert tenant_a_webhooks[0].workspace_id == "tenant_a"
        assert all(w.workspace_id == "tenant_a" for w in tenant_a_webhooks)

    def test_user_isolation_within_workspace(self, webhook_registry):
        """Users within a workspace should have isolated webhooks."""
        webhook_registry.register(
            url="https://user1.com",
            events=["*"],
            user_id="user_1",
            workspace_id="shared_ws",
        )
        webhook_registry.register(
            url="https://user2.com",
            events=["*"],
            user_id="user_2",
            workspace_id="shared_ws",
        )

        # User 1 lists their webhooks
        user1_webhooks = webhook_registry.list(user_id="user_1", workspace_id="shared_ws")

        assert len(user1_webhooks) == 1
        assert user1_webhooks[0].user_id == "user_1"

    def test_get_for_event_respects_workspace(self, webhook_registry):
        """get_for_event should respect workspace isolation."""
        webhook_registry.register(
            url="https://tenant-a.com",
            events=["payment.success"],
            workspace_id="tenant_a",
        )
        webhook_registry.register(
            url="https://tenant-b.com",
            events=["payment.success"],
            workspace_id="tenant_b",
        )

        # Get webhooks for tenant_a
        webhooks = webhook_registry.get_for_event(
            "payment.success",
            workspace_id="tenant_a",
        )

        assert len(webhooks) == 1
        assert webhooks[0].workspace_id == "tenant_a"

    def test_get_for_event_respects_user(self, webhook_registry):
        """get_for_event should respect user isolation."""
        webhook_registry.register(
            url="https://user1.com",
            events=["order.created"],
            user_id="user_1",
        )
        webhook_registry.register(
            url="https://user2.com",
            events=["order.created"],
            user_id="user_2",
        )

        webhooks = webhook_registry.get_for_event("order.created", user_id="user_1")

        assert len(webhooks) == 1
        assert webhooks[0].user_id == "user_1"


# =============================================================================
# Test: get_for_event functionality
# =============================================================================


class TestGetForEvent:
    """Tests for get_for_event method."""

    def test_returns_matching_webhooks(self, webhook_registry):
        """Should return webhooks that match the event type."""
        wh1 = webhook_registry.register(url="https://a.com", events=["payment.success"])
        wh2 = webhook_registry.register(url="https://b.com", events=["payment.failed"])
        wh3 = webhook_registry.register(url="https://c.com", events=["*"])

        webhooks = webhook_registry.get_for_event("payment.success")

        webhook_ids = [w.id for w in webhooks]
        assert wh1.id in webhook_ids
        assert wh3.id in webhook_ids  # Wildcard matches
        assert wh2.id not in webhook_ids

    def test_excludes_inactive_webhooks(self, webhook_registry):
        """Should exclude inactive webhooks."""
        wh = webhook_registry.register(url="https://a.com", events=["payment.success"])
        webhook_registry.update(wh.id, active=False)

        webhooks = webhook_registry.get_for_event("payment.success")
        assert len(webhooks) == 0

    def test_returns_empty_for_no_matches(self, webhook_registry):
        """Should return empty list if no webhooks match."""
        webhook_registry.register(url="https://a.com", events=["payment.success"])

        webhooks = webhook_registry.get_for_event("order.created")
        assert len(webhooks) == 0


# =============================================================================
# Test: Global registry management
# =============================================================================


class TestGlobalRegistryManagement:
    """Tests for global webhook registry management functions."""

    def test_set_webhook_registry_uses_custom(self):
        """set_webhook_registry should set a custom registry."""
        with patch("aragora.storage.webhook_registry._webhook_registry", None):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                custom_registry = SQLiteWebhookRegistry(Path(tmp_dir) / "custom.db")
                set_webhook_registry(custom_registry)

                retrieved = get_webhook_registry()
                assert retrieved is custom_registry

                custom_registry.close()

    def test_reset_webhook_registry_clears_global(self):
        """reset_webhook_registry should clear the global registry."""
        reset_webhook_registry()
        # After reset, next call should create new registry


# =============================================================================
# Test: Database schema and initialization
# =============================================================================


class TestDatabaseSchema:
    """Tests for database schema and initialization."""

    def test_creates_database_file(self, tmp_path):
        """Should create database file."""
        db_path = tmp_path / "new_registry.db"
        assert not db_path.exists()

        registry = SQLiteWebhookRegistry(db_path=db_path)
        assert db_path.exists()

        registry.close()

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories."""
        db_path = tmp_path / "nested" / "path" / "registry.db"

        registry = SQLiteWebhookRegistry(db_path=db_path)
        assert db_path.exists()

        registry.close()

    def test_schema_version_tracked(self, tmp_path):
        """Should track schema version."""
        db_path = tmp_path / "versioned.db"
        registry = SQLiteWebhookRegistry(db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT version FROM _schema_versions WHERE module='webhook_registry'"
        )
        version = cursor.fetchone()
        assert version is not None
        assert version[0] == 1

        conn.close()
        registry.close()

    def test_indexes_created(self, tmp_path):
        """Should create database indexes."""
        db_path = tmp_path / "indexed.db"
        registry = SQLiteWebhookRegistry(db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_webhook%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_webhook_user" in indexes
        assert "idx_webhook_workspace" in indexes
        assert "idx_webhook_active" in indexes

        conn.close()
        registry.close()


# =============================================================================
# Test: Concurrent access
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access safety."""

    def test_concurrent_registrations(self, webhook_registry):
        """Concurrent registrations should be thread-safe."""
        errors = []
        webhook_ids = []
        lock = threading.Lock()

        def register_webhook(n):
            try:
                for i in range(10):
                    wh = webhook_registry.register(
                        url=f"https://thread{n}.com/{i}",
                        events=["*"],
                    )
                    with lock:
                        webhook_ids.append(wh.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_webhook, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(webhook_ids) == 50  # 5 threads * 10 webhooks

    def test_concurrent_reads_and_writes(self, webhook_registry):
        """Concurrent reads and writes should be thread-safe."""
        wh = webhook_registry.register(url="https://test.com", events=["*"])
        errors = []

        def read_operations():
            try:
                for _ in range(50):
                    webhook_registry.get(wh.id)
                    webhook_registry.list()
            except Exception as e:
                errors.append(e)

        def write_operations():
            try:
                for i in range(50):
                    webhook_registry.record_delivery(wh.id, 200, True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=read_operations),
            threading.Thread(target=write_operations),
            threading.Thread(target=read_operations),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test: Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_events_list(self, webhook_registry):
        """Should handle empty events list."""
        webhook = webhook_registry.register(url="https://example.com", events=[])

        assert webhook.events == []
        # Should not match any events
        assert webhook.matches_event("any.event") is False

    def test_many_events(self, webhook_registry):
        """Should handle many event types."""
        events = [f"event.type.{i}" for i in range(100)]
        webhook = webhook_registry.register(url="https://example.com", events=events)

        assert len(webhook.events) == 100
        assert webhook.matches_event("event.type.50") is True

    def test_special_characters_in_url(self, webhook_registry):
        """Should handle special characters in URL."""
        url = "https://example.com/webhook?key=value&other=123"
        webhook = webhook_registry.register(url=url, events=["*"])

        retrieved = webhook_registry.get(webhook.id)
        assert retrieved is not None
        assert retrieved.url == url

    def test_unicode_in_name(self, webhook_registry):
        """Should handle unicode in name."""
        webhook = webhook_registry.register(
            url="https://example.com",
            events=["*"],
            name="Webhook \u4e2d\u6587",
        )

        retrieved = webhook_registry.get(webhook.id)
        assert retrieved is not None
        assert retrieved.name == "Webhook \u4e2d\u6587"

    def test_get_nonexistent_returns_none(self, webhook_registry):
        """Getting non-existent webhook should return None."""
        result = webhook_registry.get("nonexistent_id")
        assert result is None

    def test_close_multiple_times(self, tmp_path):
        """Closing multiple times should not raise."""
        registry = SQLiteWebhookRegistry(tmp_path / "close_test.db")
        registry.close()
        registry.close()  # Should not raise
