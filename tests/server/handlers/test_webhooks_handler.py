"""
Tests for aragora.server.handlers.webhooks - Webhook management API.

Tests cover:
- WebhookConfig dataclass
- WebhookStore CRUD operations
- Signature generation and verification
- WebhookHandler endpoints (register, list, get, delete, update, test)
- Event validation
- User ownership filtering
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockHandler:
    """Mock HTTP handler for testing."""

    headers: dict[str, str]
    body: bytes = b""
    rfile: Any = None
    client_address: tuple[str, int] = ("127.0.0.1", 12345)

    def __post_init__(self):
        import io

        self.rfile = io.BytesIO(self.body)


@pytest.fixture
def webhook_store():
    """Create a fresh webhook store."""
    from aragora.storage.webhook_config_store import InMemoryWebhookConfigStore

    return InMemoryWebhookConfigStore()


@pytest.fixture
def sample_webhook(webhook_store):
    """Create a sample webhook in the store."""
    return webhook_store.register(
        url="https://example.com/webhook",
        events=["debate_start", "debate_end"],
        name="Test Webhook",
        description="A test webhook",
        user_id="user-123",
    )


@pytest.fixture
def server_context():
    """Create mock server context."""
    from aragora.storage.webhook_config_store import InMemoryWebhookConfigStore

    return {"webhook_store": InMemoryWebhookConfigStore()}


@pytest.fixture
def webhook_handler(server_context):
    """Create webhook handler instance."""
    from aragora.server.handlers.webhooks import WebhookHandler

    return WebhookHandler(server_context)


# ===========================================================================
# Test WebhookConfig
# ===========================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_default_values(self):
        """WebhookConfig should have sensible defaults."""
        from aragora.server.handlers.webhooks import WebhookConfig

        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start"],
            secret="test-secret",
        )

        assert config.active is True
        assert config.delivery_count == 0
        assert config.failure_count == 0
        assert config.last_delivery_at is None

    def test_to_dict_excludes_secret(self):
        """to_dict should exclude secret by default."""
        from aragora.server.handlers.webhooks import WebhookConfig

        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start"],
            secret="super-secret",
        )

        result = config.to_dict()
        assert "secret" not in result

    def test_to_dict_includes_secret_when_requested(self):
        """to_dict should include secret when explicitly requested."""
        from aragora.server.handlers.webhooks import WebhookConfig

        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start"],
            secret="super-secret",
        )

        result = config.to_dict(include_secret=True)
        assert result["secret"] == "super-secret"

    def test_matches_event_when_active(self):
        """matches_event should return True for subscribed events."""
        from aragora.server.handlers.webhooks import WebhookConfig

        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start", "debate_end"],
            secret="secret",
            active=True,
        )

        assert config.matches_event("debate_start") is True
        assert config.matches_event("debate_end") is True
        assert config.matches_event("vote") is False

    def test_matches_event_when_inactive(self):
        """matches_event should return False when webhook is inactive."""
        from aragora.server.handlers.webhooks import WebhookConfig

        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start"],
            secret="secret",
            active=False,
        )

        assert config.matches_event("debate_start") is False

    def test_matches_event_wildcard(self):
        """matches_event should accept any valid event with '*' subscription."""
        from aragora.server.handlers.webhooks import WebhookConfig, WEBHOOK_EVENTS

        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["*"],
            secret="secret",
            active=True,
        )

        # Should match all valid webhook events
        for event in WEBHOOK_EVENTS:
            assert config.matches_event(event) is True

        # Should not match invalid events
        assert config.matches_event("invalid_event") is False


# ===========================================================================
# Test WebhookStore
# ===========================================================================


class TestWebhookStore:
    """Tests for WebhookStore CRUD operations."""

    def test_register_creates_webhook(self, webhook_store):
        """register should create a new webhook with generated ID and secret."""
        webhook = webhook_store.register(
            url="https://example.com/hook",
            events=["debate_start"],
            name="My Webhook",
        )

        assert webhook.id is not None
        assert len(webhook.id) > 0
        assert webhook.secret is not None
        assert len(webhook.secret) > 20  # Token should be sufficiently long
        assert webhook.url == "https://example.com/hook"
        assert webhook.events == ["debate_start"]
        assert webhook.name == "My Webhook"

    def test_get_returns_webhook(self, webhook_store, sample_webhook):
        """get should return webhook by ID."""
        result = webhook_store.get(sample_webhook.id)
        assert result is not None
        assert result.id == sample_webhook.id

    def test_get_returns_none_for_invalid_id(self, webhook_store):
        """get should return None for non-existent ID."""
        result = webhook_store.get("non-existent-id")
        assert result is None

    def test_list_returns_all_webhooks(self, webhook_store):
        """list should return all webhooks."""
        webhook_store.register(url="https://a.com", events=["debate_start"])
        webhook_store.register(url="https://b.com", events=["debate_end"])
        webhook_store.register(url="https://c.com", events=["vote"])

        result = webhook_store.list()
        assert len(result) == 3

    def test_list_filters_by_user_id(self, webhook_store):
        """list should filter by user_id."""
        webhook_store.register(url="https://a.com", events=["debate_start"], user_id="user-1")
        webhook_store.register(url="https://b.com", events=["debate_end"], user_id="user-1")
        webhook_store.register(url="https://c.com", events=["vote"], user_id="user-2")

        result = webhook_store.list(user_id="user-1")
        assert len(result) == 2

    def test_list_filters_active_only(self, webhook_store):
        """list should filter by active status."""
        w1 = webhook_store.register(url="https://a.com", events=["debate_start"])
        webhook_store.register(url="https://b.com", events=["debate_end"])

        # Deactivate one webhook
        webhook_store.update(w1.id, active=False)

        result = webhook_store.list(active_only=True)
        assert len(result) == 1

    def test_delete_removes_webhook(self, webhook_store, sample_webhook):
        """delete should remove webhook."""
        result = webhook_store.delete(sample_webhook.id)
        assert result is True
        assert webhook_store.get(sample_webhook.id) is None

    def test_delete_returns_false_for_invalid_id(self, webhook_store):
        """delete should return False for non-existent ID."""
        result = webhook_store.delete("non-existent-id")
        assert result is False

    def test_update_modifies_webhook(self, webhook_store, sample_webhook):
        """update should modify webhook fields."""
        updated = webhook_store.update(
            webhook_id=sample_webhook.id,
            url="https://new-url.com",
            events=["vote"],
            active=False,
            name="New Name",
        )

        assert updated is not None
        assert updated.url == "https://new-url.com"
        assert updated.events == ["vote"]
        assert updated.active is False
        assert updated.name == "New Name"

    def test_update_returns_none_for_invalid_id(self, webhook_store):
        """update should return None for non-existent ID."""
        result = webhook_store.update("non-existent-id", url="https://new.com")
        assert result is None

    def test_record_delivery_updates_stats(self, webhook_store, sample_webhook):
        """record_delivery should update delivery statistics."""
        webhook_store.record_delivery(sample_webhook.id, status_code=200, success=True)

        webhook = webhook_store.get(sample_webhook.id)
        assert webhook.delivery_count == 1
        assert webhook.failure_count == 0
        assert webhook.last_delivery_status == 200
        assert webhook.last_delivery_at is not None

    def test_record_delivery_tracks_failures(self, webhook_store, sample_webhook):
        """record_delivery should increment failure count on failure."""
        webhook_store.record_delivery(sample_webhook.id, status_code=500, success=False)

        webhook = webhook_store.get(sample_webhook.id)
        assert webhook.delivery_count == 1
        assert webhook.failure_count == 1

    def test_get_for_event_returns_matching_webhooks(self, webhook_store):
        """get_for_event should return webhooks subscribed to the event."""
        webhook_store.register(url="https://a.com", events=["debate_start"])
        webhook_store.register(url="https://b.com", events=["debate_end"])
        webhook_store.register(url="https://c.com", events=["debate_start", "vote"])

        result = webhook_store.get_for_event("debate_start")
        assert len(result) == 2


# ===========================================================================
# Test Signature Utilities
# ===========================================================================


class TestSignatureUtilities:
    """Tests for HMAC signature generation and verification."""

    def test_generate_signature_format(self):
        """generate_signature should return sha256= prefixed hex string."""
        from aragora.server.handlers.webhooks import generate_signature

        signature = generate_signature('{"test": "data"}', "secret-key")

        assert signature.startswith("sha256=")
        assert len(signature) == 7 + 64  # "sha256=" + 64 hex chars

    def test_generate_signature_deterministic(self):
        """generate_signature should be deterministic."""
        from aragora.server.handlers.webhooks import generate_signature

        sig1 = generate_signature("test payload", "secret")
        sig2 = generate_signature("test payload", "secret")

        assert sig1 == sig2

    def test_generate_signature_different_for_different_secrets(self):
        """generate_signature should differ for different secrets."""
        from aragora.server.handlers.webhooks import generate_signature

        sig1 = generate_signature("test payload", "secret1")
        sig2 = generate_signature("test payload", "secret2")

        assert sig1 != sig2

    def test_verify_signature_valid(self):
        """verify_signature should return True for valid signature."""
        from aragora.server.handlers.webhooks import generate_signature, verify_signature

        payload = '{"event": "test"}'
        secret = "webhook-secret"
        signature = generate_signature(payload, secret)

        assert verify_signature(payload, signature, secret) is True

    def test_verify_signature_invalid(self):
        """verify_signature should return False for invalid signature."""
        from aragora.server.handlers.webhooks import verify_signature

        payload = '{"event": "test"}'
        secret = "webhook-secret"
        invalid_signature = "sha256=invalid"

        assert verify_signature(payload, invalid_signature, secret) is False

    def test_verify_signature_tampered_payload(self):
        """verify_signature should return False for tampered payload."""
        from aragora.server.handlers.webhooks import generate_signature, verify_signature

        original_payload = '{"event": "test"}'
        tampered_payload = '{"event": "tampered"}'
        secret = "webhook-secret"
        signature = generate_signature(original_payload, secret)

        assert verify_signature(tampered_payload, signature, secret) is False


# ===========================================================================
# Test WebhookHandler Endpoints
# ===========================================================================


class TestWebhookHandlerListEvents:
    """Tests for GET /api/webhooks/events endpoint."""

    @pytest.mark.asyncio
    async def test_list_events_returns_all_event_types(self, webhook_handler):
        """Should return all available webhook event types."""
        from aragora.server.handlers.webhooks import WEBHOOK_EVENTS

        handler = MockHandler(headers={})
        result = await webhook_handler.handle("/api/v1/webhooks/events", {}, handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "events" in body
        assert len(body["events"]) == len(WEBHOOK_EVENTS)
        assert "description" in body


class TestWebhookHandlerRegister:
    """Tests for POST /api/webhooks endpoint."""

    @pytest.mark.asyncio
    async def test_register_webhook_success(self, webhook_handler):
        """Should successfully register a webhook."""
        body = json.dumps(
            {
                "url": "https://example.com/webhook",
                "events": ["debate_start", "debate_end"],
                "name": "Test Webhook",
            }
        ).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = await webhook_handler.handle_post("/api/v1/webhooks", {}, handler)

        assert result is not None
        assert result.status_code == 201

        body = json.loads(result.body)
        assert "webhook" in body
        assert body["webhook"]["url"] == "https://example.com/webhook"
        # Secret should be included on creation
        assert "secret" in body["webhook"]

    @pytest.mark.asyncio
    async def test_register_webhook_missing_url(self, webhook_handler):
        """Should reject registration without URL."""
        body = json.dumps({"events": ["debate_start"]}).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = await webhook_handler.handle_post("/api/v1/webhooks", {}, handler)

        assert result.status_code == 400
        assert b"URL is required" in result.body

    @pytest.mark.asyncio
    async def test_register_webhook_invalid_url(self, webhook_handler):
        """Should reject registration with invalid URL."""
        body = json.dumps(
            {
                "url": "not-a-valid-url",
                "events": ["debate_start"],
            }
        ).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = await webhook_handler.handle_post("/api/v1/webhooks", {}, handler)

        assert result.status_code == 400
        assert b"http" in result.body.lower()

    @pytest.mark.asyncio
    async def test_register_webhook_missing_events(self, webhook_handler):
        """Should reject registration without events."""
        body = json.dumps({"url": "https://example.com"}).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = await webhook_handler.handle_post("/api/v1/webhooks", {}, handler)

        assert result.status_code == 400
        assert b"event" in result.body.lower()

    @pytest.mark.asyncio
    async def test_register_webhook_invalid_events(self, webhook_handler):
        """Should reject registration with invalid event types."""
        body = json.dumps(
            {
                "url": "https://example.com",
                "events": ["invalid_event_type"],
            }
        ).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = await webhook_handler.handle_post("/api/v1/webhooks", {}, handler)

        assert result.status_code == 400
        assert b"Invalid event" in result.body


class TestWebhookHandlerList:
    """Tests for GET /api/webhooks endpoint."""

    @pytest.mark.asyncio
    async def test_list_webhooks_empty(self, webhook_handler):
        """Should return empty list when no webhooks registered."""
        handler = MockHandler(headers={})

        result = await webhook_handler.handle("/api/v1/webhooks", {}, handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["webhooks"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_webhooks_returns_registered(self, webhook_handler, server_context):
        """Should return registered webhooks."""
        # Register some webhooks (with user_id matching mock auth context)
        store = server_context["webhook_store"]
        store.register(url="https://a.com", events=["debate_start"], user_id="test-user-001")
        store.register(url="https://b.com", events=["debate_end"], user_id="test-user-001")

        handler = MockHandler(headers={})
        result = await webhook_handler.handle("/api/v1/webhooks", {}, handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["webhooks"]) == 2

    @pytest.mark.asyncio
    async def test_list_webhooks_excludes_secrets(self, webhook_handler, server_context):
        """Should not include secrets in list response."""
        store = server_context["webhook_store"]
        store.register(url="https://a.com", events=["debate_start"], user_id="test-user-001")

        handler = MockHandler(headers={})
        result = await webhook_handler.handle("/api/v1/webhooks", {}, handler)

        body = json.loads(result.body)
        for webhook in body["webhooks"]:
            assert "secret" not in webhook


class TestWebhookHandlerGet:
    """Tests for GET /api/webhooks/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_webhook_success(self, webhook_handler, server_context):
        """Should return specific webhook by ID."""
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com", events=["debate_start"], user_id="test-user-001"
        )

        handler = MockHandler(headers={})
        result = await webhook_handler.handle(f"/api/v1/webhooks/{webhook.id}", {}, handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["webhook"]["id"] == webhook.id

    @pytest.mark.asyncio
    async def test_get_webhook_not_found(self, webhook_handler):
        """Should return 404 for non-existent webhook."""
        handler = MockHandler(headers={})
        result = await webhook_handler.handle("/api/v1/webhooks/non-existent-id", {}, handler)

        assert result.status_code == 404


class TestWebhookHandlerDelete:
    """Tests for DELETE /api/webhooks/:id endpoint."""

    @pytest.mark.asyncio
    async def test_delete_webhook_success(self, webhook_handler, server_context):
        """Should delete webhook."""
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com", events=["debate_start"], user_id="test-user-001"
        )

        handler = MockHandler(headers={})
        result = await webhook_handler.handle_delete(f"/api/v1/webhooks/{webhook.id}", {}, handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deleted"] is True

        # Verify webhook is actually deleted
        assert store.get(webhook.id) is None

    @pytest.mark.asyncio
    async def test_delete_webhook_not_found(self, webhook_handler):
        """Should return 404 when deleting non-existent webhook."""
        handler = MockHandler(headers={})
        result = await webhook_handler.handle_delete(
            "/api/v1/webhooks/non-existent-id", {}, handler
        )

        assert result.status_code == 404


class TestWebhookHandlerUpdate:
    """Tests for PATCH /api/webhooks/:id endpoint."""

    def test_update_webhook_success(self, webhook_handler, server_context):
        """Should update webhook fields."""
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://old.com", events=["debate_start"], user_id="test-user-001"
        )

        body = json.dumps({"url": "https://new.com", "active": False}).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = webhook_handler.handle_patch(f"/api/v1/webhooks/{webhook.id}", {}, handler)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["webhook"]["url"] == "https://new.com"
        assert response_body["webhook"]["active"] is False

    def test_update_webhook_not_found(self, webhook_handler):
        """Should return 404 when updating non-existent webhook."""
        body = json.dumps({"url": "https://new.com"}).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = webhook_handler.handle_patch("/api/v1/webhooks/non-existent-id", {}, handler)
        assert result.status_code == 404

    def test_update_webhook_invalid_events(self, webhook_handler, server_context):
        """Should reject invalid event types in update."""
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com", events=["debate_start"], user_id="test-user-001"
        )

        body = json.dumps({"events": ["invalid_event"]}).encode()
        handler = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        result = webhook_handler.handle_patch(f"/api/v1/webhooks/{webhook.id}", {}, handler)
        assert result.status_code == 400


class TestWebhookHandlerTest:
    """Tests for POST /api/webhooks/:id/test endpoint."""

    @pytest.mark.asyncio
    async def test_test_webhook_not_found(self, webhook_handler):
        """Should return 404 when testing non-existent webhook."""
        handler = MockHandler(headers={})
        result = await webhook_handler.handle_post(
            "/api/v1/webhooks/non-existent-id/test", {}, handler
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_test_webhook_success(self, webhook_handler, server_context):
        """Should send test event to webhook."""
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com/hook", events=["debate_start"], user_id="test-user-001"
        )

        handler = MockHandler(headers={})

        # Mock the dispatch_webhook function (imported inside the method)
        with patch("aragora.events.dispatcher.dispatch_webhook") as mock_dispatch:
            mock_dispatch.return_value = (True, 200, None)

            result = await webhook_handler.handle_post(
                f"/api/v1/webhooks/{webhook.id}/test", {}, handler
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            mock_dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_webhook_delivery_failure(self, webhook_handler, server_context):
        """Should report delivery failure."""
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com/hook", events=["debate_start"], user_id="test-user-001"
        )

        handler = MockHandler(headers={})

        with patch("aragora.events.dispatcher.dispatch_webhook") as mock_dispatch:
            mock_dispatch.return_value = (False, 500, "Connection refused")

            result = await webhook_handler.handle_post(
                f"/api/v1/webhooks/{webhook.id}/test", {}, handler
            )

            assert result.status_code == 502
            body = json.loads(result.body)
            assert body["success"] is False
            assert "error" in body


# ===========================================================================
# Test get_webhook_store Singleton
# ===========================================================================


class TestGetWebhookStore:
    """Tests for get_webhook_store function."""

    def test_returns_singleton(self):
        """get_webhook_store should return the same instance."""
        import aragora.server.handlers.webhooks as webhooks_module

        # Reset the global store
        webhooks_module._webhook_store = None

        store1 = webhooks_module.get_webhook_store()
        store2 = webhooks_module.get_webhook_store()

        assert store1 is store2

        # Clean up
        webhooks_module._webhook_store = None


# ===========================================================================
# RBAC Tests
# ===========================================================================


@dataclass
class MockPermissionDecision:
    """Mock RBAC permission decision."""

    allowed: bool = True
    reason: str = "Allowed by test"


@dataclass
class MockUser:
    """Mock user for RBAC testing."""

    user_id: str = "user-123"
    role: str = "admin"
    org_id: str = "org-123"


@dataclass
class MockAuthorizationContext:
    """Minimal auth context for RBAC testing."""

    user_id: str
    roles: set[str]
    org_id: str | None = None


def mock_check_permission_allowed(*args, **kwargs):
    """Mock check_permission that always allows."""
    return MockPermissionDecision(allowed=True)


def mock_check_permission_denied(*args, **kwargs):
    """Mock check_permission that always denies."""
    return MockPermissionDecision(allowed=False, reason="Permission denied by test")


class TestWebhookRBAC:
    """Tests for RBAC permission checks in WebhookHandler."""

    @pytest.fixture
    def handler(self, server_context):
        from aragora.server.handlers.webhooks import WebhookHandler

        return WebhookHandler(server_context)

    def test_rbac_helper_methods_exist(self, handler):
        """Handler should have RBAC helper methods."""
        assert hasattr(handler, "_check_rbac_permission")
        assert hasattr(handler, "_get_auth_context")

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", False)
    def test_permission_check_without_rbac(self, handler):
        """Permission check should pass when RBAC not available."""
        mock_http = MockHandler(headers={})

        result = handler._check_rbac_permission(mock_http, "webhooks.create")
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.webhooks.AuthorizationContext", MockAuthorizationContext)
    @patch("aragora.server.handlers.webhooks.check_permission", mock_check_permission_allowed)
    def test_permission_check_allowed(self, handler):
        """Permission check should pass when RBAC allows."""
        mock_http = MockHandler(headers={})

        with patch.object(handler, "get_current_user", return_value=MockUser()):
            result = handler._check_rbac_permission(mock_http, "webhooks.create")
            assert result is None  # None means allowed

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.webhooks.AuthorizationContext", MockAuthorizationContext)
    @patch("aragora.server.handlers.webhooks.check_permission", mock_check_permission_denied)
    def test_permission_check_denied(self, handler):
        """Permission check should return error when RBAC denies."""
        mock_http = MockHandler(headers={})

        with patch.object(handler, "get_current_user", return_value=MockUser(role="viewer")):
            result = handler._check_rbac_permission(mock_http, "webhooks.create")
            assert result is not None
            assert result.status_code == 403

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.webhooks.AuthorizationContext", MockAuthorizationContext)
    @patch("aragora.server.handlers.webhooks.check_permission", mock_check_permission_denied)
    def test_register_webhook_rbac_denied(self, handler):
        """Register webhook should deny when RBAC denies."""
        body = json.dumps(
            {
                "url": "https://example.com/webhook",
                "events": ["debate_start"],
            }
        ).encode()
        mock_http = MockHandler(
            headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
            body=body,
        )

        with patch.object(handler, "get_current_user", return_value=MockUser()):
            result = handler._handle_register_webhook(
                {"url": "https://example.com", "events": ["debate_start"]}, mock_http
            )
            assert result.status_code == 403

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.webhooks.AuthorizationContext", MockAuthorizationContext)
    @patch("aragora.server.handlers.webhooks.check_permission", mock_check_permission_denied)
    def test_delete_webhook_rbac_denied(self, handler, server_context):
        """Delete webhook should deny when RBAC denies."""
        # First create a webhook so it exists
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com", events=["debate_start"], user_id="test-user-001"
        )
        mock_http = MockHandler(headers={})

        with patch.object(handler, "get_current_user", return_value=MockUser()):
            result = handler._handle_delete_webhook(webhook.id, mock_http)
            assert result.status_code == 403

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.webhooks.AuthorizationContext", MockAuthorizationContext)
    @patch("aragora.server.handlers.webhooks.check_permission", mock_check_permission_denied)
    def test_update_webhook_rbac_denied(self, handler, server_context):
        """Update webhook should deny when RBAC denies."""
        # First create a webhook so it exists
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com", events=["debate_start"], user_id="test-user-001"
        )
        mock_http = MockHandler(headers={})

        with patch.object(handler, "get_current_user", return_value=MockUser()):
            result = handler._handle_update_webhook(webhook.id, {}, mock_http)
            assert result.status_code == 403

    @patch("aragora.server.handlers.webhooks.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.webhooks.AuthorizationContext", MockAuthorizationContext)
    @patch("aragora.server.handlers.webhooks.check_permission", mock_check_permission_denied)
    def test_test_webhook_rbac_denied(self, handler, server_context):
        """Test webhook should deny when RBAC denies."""
        # First create a webhook so it exists
        store = server_context["webhook_store"]
        webhook = store.register(
            url="https://example.com", events=["debate_start"], user_id="test-user-001"
        )
        mock_http = MockHandler(headers={})

        with patch.object(handler, "get_current_user", return_value=MockUser()):
            result = handler._handle_test_webhook(webhook.id, mock_http)
            assert result.status_code == 403
