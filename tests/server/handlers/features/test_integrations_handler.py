"""
Tests for IntegrationsHandler.

Verifies the handler correctly uses the persistent IntegrationStore.
"""

import json
import pytest
import time
from unittest.mock import MagicMock

from aragora.server.handlers.features.integrations import (
    IntegrationsHandler,
    IntegrationStatus,
    VALID_INTEGRATION_TYPES,
)
from aragora.storage.integration_store import (
    InMemoryIntegrationStore,
    IntegrationConfig,
    set_integration_store,
    reset_integration_store,
)


def parse_result(result):
    """Parse HandlerResult into dict with status and body."""
    return {
        "status": result.status_code,
        "body": json.loads(result.body.decode("utf-8")) if result.body else {},
    }


@pytest.fixture(autouse=True)
def setup_memory_store():
    """Use in-memory store for all tests."""
    store = InMemoryIntegrationStore()
    set_integration_store(store)
    yield store
    reset_integration_store()


@pytest.fixture
def mock_server_context():
    """Create a mock server context."""
    ctx = MagicMock()
    ctx.state_manager = MagicMock()
    ctx.elo_tracker = MagicMock()
    return ctx


@pytest.fixture
def handler(mock_server_context):
    """Create handler instance with mock context."""
    return IntegrationsHandler(mock_server_context)


class TestGetStatus:
    """Tests for get_status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_empty(self, handler):
        """Should return not_configured for all types when empty."""
        result = parse_result(await handler.get_status("user1"))
        assert result["status"] == 200
        integrations = result["body"]["integrations"]
        assert len(integrations) == len(VALID_INTEGRATION_TYPES)
        for integration in integrations:
            assert integration["status"] == "not_configured"
            assert integration["enabled"] is False

    @pytest.mark.asyncio
    async def test_get_status_with_configured(self, handler, setup_memory_store):
        """Should return configured status for existing integrations."""
        # Pre-configure an integration
        config = IntegrationConfig(
            type="slack",
            enabled=True,
            user_id="user1",
            last_activity=time.time(),
        )
        await setup_memory_store.save(config)

        result = parse_result(await handler.get_status("user1"))
        integrations = {i["type"]: i for i in result["body"]["integrations"]}
        assert integrations["slack"]["status"] == "connected"
        assert integrations["slack"]["enabled"] is True


class TestGetIntegration:
    """Tests for get_integration endpoint."""

    @pytest.mark.asyncio
    async def test_get_integration_not_found(self, handler):
        """Should return 404 for unconfigured integration."""
        result = parse_result(await handler.get_integration("slack", "user1"))
        assert result["status"] == 404
        assert "not configured" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_get_integration_invalid_type(self, handler):
        """Should return 400 for invalid type."""
        result = parse_result(await handler.get_integration("invalid_type", "user1"))
        assert result["status"] == 400
        assert "invalid" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_get_integration_success(self, handler, setup_memory_store):
        """Should return integration config."""
        config = IntegrationConfig(
            type="slack",
            user_id="user1",
            settings={"channel": "#test"},
        )
        await setup_memory_store.save(config)

        result = parse_result(await handler.get_integration("slack", "user1"))
        assert result["status"] == 200
        assert result["body"]["integration"]["type"] == "slack"
        assert result["body"]["integration"]["settings"]["channel"] == "#test"


class TestConfigureIntegration:
    """Tests for configure_integration endpoint."""

    @pytest.mark.asyncio
    async def test_configure_new_integration(self, handler, setup_memory_store):
        """Should create new integration."""
        data = {
            "enabled": True,
            "webhook_url": "https://hooks.slack.com/test",
            "channel": "#debates",
            "notify_on_consensus": True,
        }
        result = parse_result(await handler.configure_integration("slack", data, "user1"))
        assert result["status"] == 201
        assert result["body"]["integration"]["type"] == "slack"
        assert result["body"]["integration"]["enabled"] is True

        # Verify persisted
        saved = await setup_memory_store.get("slack", "user1")
        assert saved is not None
        assert saved.settings["channel"] == "#debates"

    @pytest.mark.asyncio
    async def test_configure_update_existing(self, handler, setup_memory_store):
        """Should update existing integration."""
        # First create
        config = IntegrationConfig(
            type="slack",
            user_id="user1",
            settings={"channel": "#old"},
        )
        await setup_memory_store.save(config)

        # Then update
        data = {"channel": "#new", "enabled": False}
        result = parse_result(await handler.configure_integration("slack", data, "user1"))
        assert result["status"] == 200
        assert result["body"]["integration"]["enabled"] is False

        # Verify updated
        saved = await setup_memory_store.get("slack", "user1")
        assert saved.settings["channel"] == "#new"
        assert saved.enabled is False

    @pytest.mark.asyncio
    async def test_configure_invalid_type(self, handler):
        """Should reject invalid type."""
        result = parse_result(await handler.configure_integration("invalid", {}, "user1"))
        assert result["status"] == 400


class TestUpdateIntegration:
    """Tests for update_integration (PATCH) endpoint."""

    @pytest.mark.asyncio
    async def test_update_not_found(self, handler):
        """Should return 404 for unconfigured integration."""
        result = parse_result(await handler.update_integration("slack", {"enabled": False}, "user1"))
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_update_enable_disable(self, handler, setup_memory_store):
        """Should toggle enabled status."""
        config = IntegrationConfig(type="slack", user_id="user1", enabled=True)
        await setup_memory_store.save(config)

        result = parse_result(await handler.update_integration("slack", {"enabled": False}, "user1"))
        assert result["status"] == 200
        assert result["body"]["integration"]["enabled"] is False

        # Verify persisted
        saved = await setup_memory_store.get("slack", "user1")
        assert saved.enabled is False

    @pytest.mark.asyncio
    async def test_update_notification_settings(self, handler, setup_memory_store):
        """Should update notification settings."""
        config = IntegrationConfig(
            type="slack",
            user_id="user1",
            notify_on_error=False,
        )
        await setup_memory_store.save(config)

        result = parse_result(await handler.update_integration(
            "slack", {"notify_on_error": True}, "user1"
        ))
        assert result["status"] == 200

        saved = await setup_memory_store.get("slack", "user1")
        assert saved.notify_on_error is True


class TestDeleteIntegration:
    """Tests for delete_integration endpoint."""

    @pytest.mark.asyncio
    async def test_delete_not_found(self, handler):
        """Should return 404 for unconfigured integration."""
        result = parse_result(await handler.delete_integration("slack", "user1"))
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_delete_success(self, handler, setup_memory_store):
        """Should delete integration."""
        config = IntegrationConfig(type="slack", user_id="user1")
        await setup_memory_store.save(config)

        result = parse_result(await handler.delete_integration("slack", "user1"))
        assert result["status"] == 200
        assert "deleted" in result["body"]["message"].lower()

        # Verify deleted
        saved = await setup_memory_store.get("slack", "user1")
        assert saved is None

    @pytest.mark.asyncio
    async def test_delete_invalid_type(self, handler):
        """Should reject invalid type."""
        result = parse_result(await handler.delete_integration("invalid", "user1"))
        assert result["status"] == 400


class TestTestIntegration:
    """Tests for test_integration endpoint."""

    @pytest.mark.asyncio
    async def test_test_not_configured(self, handler):
        """Should return 404 for unconfigured integration."""
        result = parse_result(await handler.test_integration("slack", "user1"))
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_test_email_valid_config(self, handler, setup_memory_store):
        """Should test email integration (validation-based)."""
        config = IntegrationConfig(
            type="email",
            user_id="user1",
            settings={"provider": "smtp", "smtp_host": "smtp.example.com"},
        )
        await setup_memory_store.save(config)

        result = parse_result(await handler.test_integration("email", "user1"))
        assert result["status"] == 200
        # Email validation is config-based, should succeed
        assert result["body"]["success"] is True

    @pytest.mark.asyncio
    async def test_test_email_invalid_config(self, handler, setup_memory_store):
        """Should fail for invalid email config."""
        config = IntegrationConfig(
            type="email",
            user_id="user1",
            settings={"provider": "smtp"},  # Missing smtp_host
        )
        await setup_memory_store.save(config)

        result = parse_result(await handler.test_integration("email", "user1"))
        assert result["status"] == 200
        assert result["body"]["success"] is False

        # Should increment error count
        saved = await setup_memory_store.get("email", "user1")
        assert saved.errors_24h == 1


class TestIntegrationStatus:
    """Tests for IntegrationStatus dataclass."""

    def test_to_dict(self):
        """Should serialize to dict."""
        status = IntegrationStatus(
            type="slack",
            enabled=True,
            status="connected",
            messages_sent=10,
            errors=2,
            last_activity="2024-01-01T00:00:00Z",
        )
        result = status.to_dict()
        assert result["type"] == "slack"
        assert result["enabled"] is True
        assert result["messages_sent"] == 10


class TestMultiTenant:
    """Tests for multi-tenant isolation."""

    @pytest.mark.asyncio
    async def test_user_isolation(self, handler, setup_memory_store):
        """Different users should have isolated configurations."""
        # User 1 config
        await handler.configure_integration(
            "slack", {"channel": "#user1"}, "user1"
        )
        # User 2 config
        await handler.configure_integration(
            "slack", {"channel": "#user2"}, "user2"
        )

        # Verify isolation
        result1 = parse_result(await handler.get_integration("slack", "user1"))
        result2 = parse_result(await handler.get_integration("slack", "user2"))

        assert result1["body"]["integration"]["settings"]["channel"] == "#user1"
        assert result2["body"]["integration"]["settings"]["channel"] == "#user2"

    @pytest.mark.asyncio
    async def test_delete_isolation(self, handler, setup_memory_store):
        """Deleting one user's config shouldn't affect another."""
        await handler.configure_integration("slack", {}, "user1")
        await handler.configure_integration("slack", {}, "user2")

        await handler.delete_integration("slack", "user1")

        # User 1 deleted
        result1 = parse_result(await handler.get_integration("slack", "user1"))
        assert result1["status"] == 404

        # User 2 still exists
        result2 = parse_result(await handler.get_integration("slack", "user2"))
        assert result2["status"] == 200
