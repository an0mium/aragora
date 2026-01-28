"""
Tests for Gauntlet Webhooks.

Tests the webhook module that provides:
- Webhook configuration and registration
- Async webhook delivery with retry
- HMAC signature verification
- Circuit breaker for failing endpoints
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def webhook_manager():
    """Create a WebhookManager instance for testing."""
    from aragora.gauntlet.webhooks import WebhookManager

    manager = WebhookManager()
    return manager


@pytest.fixture
def sample_config():
    """Create a sample webhook config."""
    from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

    return WebhookConfig(
        url="https://example.com/webhook",
        secret="test-secret",
        events=[WebhookEventType.GAUNTLET_COMPLETED],
        enabled=True,
    )


# ===========================================================================
# Tests: WebhookEventType Enum
# ===========================================================================


class TestWebhookEventType:
    """Tests for WebhookEventType enum."""

    def test_event_types_exist(self):
        """Test all event types are defined."""
        from aragora.gauntlet.webhooks import WebhookEventType

        assert WebhookEventType.GAUNTLET_STARTED.value == "gauntlet.started"
        assert WebhookEventType.GAUNTLET_PROGRESS.value == "gauntlet.progress"
        assert WebhookEventType.GAUNTLET_COMPLETED.value == "gauntlet.completed"
        assert WebhookEventType.GAUNTLET_FAILED.value == "gauntlet.failed"
        assert WebhookEventType.FINDING_CRITICAL.value == "finding.critical"


# ===========================================================================
# Tests: WebhookConfig Dataclass
# ===========================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_creation_valid(self):
        """Test WebhookConfig creation with valid URL."""
        from aragora.gauntlet.webhooks import WebhookConfig

        config = WebhookConfig(url="https://example.com/webhook")

        assert config.url == "https://example.com/webhook"
        assert config.enabled is True
        assert config.max_retries == 3

    def test_creation_with_all_params(self):
        """Test WebhookConfig creation with all parameters."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(
            url="https://example.com/webhook",
            secret="my-secret",
            events=[WebhookEventType.GAUNTLET_COMPLETED],
            enabled=False,
            timeout_seconds=60.0,
            max_retries=5,
            retry_delay_seconds=2.0,
            retry_backoff_multiplier=3.0,
            headers={"X-Custom": "value"},
        )

        assert config.secret == "my-secret"
        assert len(config.events) == 1
        assert config.enabled is False
        assert config.timeout_seconds == 60.0

    def test_creation_invalid_url_empty(self):
        """Test WebhookConfig rejects empty URL."""
        from aragora.gauntlet.webhooks import WebhookConfig

        with pytest.raises(ValueError, match="URL is required"):
            WebhookConfig(url="")

    def test_creation_invalid_url_scheme(self):
        """Test WebhookConfig rejects non-HTTP URL."""
        from aragora.gauntlet.webhooks import WebhookConfig

        with pytest.raises(ValueError, match="HTTP or HTTPS"):
            WebhookConfig(url="ftp://example.com")

    def test_creation_localhost_blocked_by_default(self, monkeypatch):
        """Test WebhookConfig blocks localhost by default."""
        from aragora.gauntlet.webhooks import WebhookConfig

        monkeypatch.delenv("ARAGORA_WEBHOOK_ALLOW_LOCALHOST", raising=False)

        with pytest.raises(ValueError, match="Localhost webhooks disabled"):
            WebhookConfig(url="http://localhost:8080/webhook")

    def test_creation_localhost_allowed_with_env(self, monkeypatch):
        """Test WebhookConfig allows localhost with env var."""
        from aragora.gauntlet.webhooks import WebhookConfig

        monkeypatch.setenv("ARAGORA_WEBHOOK_ALLOW_LOCALHOST", "true")

        config = WebhookConfig(url="http://localhost:8080/webhook")

        assert config.url == "http://localhost:8080/webhook"

    def test_default_events_all(self):
        """Test WebhookConfig defaults to all events."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(url="https://example.com/webhook")

        assert len(config.events) == len(WebhookEventType)


# ===========================================================================
# Tests: WebhookPayload Dataclass
# ===========================================================================


class TestWebhookPayload:
    """Tests for WebhookPayload dataclass."""

    def test_creation(self):
        """Test WebhookPayload creation."""
        from aragora.gauntlet.webhooks import WebhookPayload, WebhookEventType

        payload = WebhookPayload(
            event_type=WebhookEventType.GAUNTLET_COMPLETED,
            timestamp="2024-01-15T10:30:00Z",
            gauntlet_id="gauntlet-123",
            data={"verdict": "pass", "confidence": 0.95},
        )

        assert payload.event_type == WebhookEventType.GAUNTLET_COMPLETED
        assert payload.gauntlet_id == "gauntlet-123"

    def test_to_dict(self):
        """Test WebhookPayload.to_dict."""
        from aragora.gauntlet.webhooks import WebhookPayload, WebhookEventType

        payload = WebhookPayload(
            event_type=WebhookEventType.GAUNTLET_STARTED,
            timestamp="2024-01-15T10:30:00Z",
            gauntlet_id="gauntlet-123",
            data={"agents": ["claude"]},
        )

        result = payload.to_dict()

        assert result["event"] == "gauntlet.started"
        assert result["gauntlet_id"] == "gauntlet-123"
        assert result["data"]["agents"] == ["claude"]

    def test_to_json(self):
        """Test WebhookPayload.to_json."""
        from aragora.gauntlet.webhooks import WebhookPayload, WebhookEventType

        payload = WebhookPayload(
            event_type=WebhookEventType.GAUNTLET_COMPLETED,
            timestamp="2024-01-15T10:30:00Z",
            gauntlet_id="gauntlet-123",
            data={"verdict": "pass"},
        )

        json_str = payload.to_json()
        parsed = json.loads(json_str)

        assert parsed["event"] == "gauntlet.completed"


# ===========================================================================
# Tests: WebhookDeliveryResult Dataclass
# ===========================================================================


class TestWebhookDeliveryResult:
    """Tests for WebhookDeliveryResult dataclass."""

    def test_creation_success(self):
        """Test WebhookDeliveryResult for successful delivery."""
        from aragora.gauntlet.webhooks import WebhookDeliveryResult

        result = WebhookDeliveryResult(
            success=True,
            status_code=200,
            response_body='{"ok": true}',
            attempts=1,
            duration_ms=150.0,
        )

        assert result.success is True
        assert result.status_code == 200

    def test_creation_failure(self):
        """Test WebhookDeliveryResult for failed delivery."""
        from aragora.gauntlet.webhooks import WebhookDeliveryResult

        result = WebhookDeliveryResult(
            success=False,
            status_code=500,
            error="Internal Server Error",
            attempts=3,
            duration_ms=5000.0,
        )

        assert result.success is False
        assert result.attempts == 3


# ===========================================================================
# Tests: WebhookManager Registration
# ===========================================================================


class TestWebhookManagerRegistration:
    """Tests for WebhookManager registration methods."""

    def test_register_webhook(self, webhook_manager, sample_config):
        """Test registering a webhook."""
        webhook_manager.register("test-webhook", sample_config)

        webhooks = webhook_manager.list_webhooks()

        assert len(webhooks) == 1
        assert webhooks[0]["name"] == "test-webhook"
        assert webhooks[0]["url"] == "https://example.com/webhook"

    def test_register_multiple_webhooks(self, webhook_manager):
        """Test registering multiple webhooks."""
        from aragora.gauntlet.webhooks import WebhookConfig

        webhook_manager.register("webhook-1", WebhookConfig(url="https://example1.com/hook"))
        webhook_manager.register("webhook-2", WebhookConfig(url="https://example2.com/hook"))

        webhooks = webhook_manager.list_webhooks()

        assert len(webhooks) == 2

    def test_unregister_webhook(self, webhook_manager, sample_config):
        """Test unregistering a webhook."""
        webhook_manager.register("test-webhook", sample_config)

        result = webhook_manager.unregister("test-webhook")

        assert result is True
        assert webhook_manager.list_webhooks() == []

    def test_unregister_nonexistent(self, webhook_manager):
        """Test unregistering a non-existent webhook."""
        result = webhook_manager.unregister("nonexistent")

        assert result is False

    def test_list_webhooks_empty(self, webhook_manager):
        """Test listing webhooks when none registered."""
        webhooks = webhook_manager.list_webhooks()

        assert webhooks == []


# ===========================================================================
# Tests: WebhookManager Emit Methods
# ===========================================================================


class TestWebhookManagerEmit:
    """Tests for WebhookManager emit methods."""

    @pytest.mark.asyncio
    async def test_emit_queues_event(self, webhook_manager, sample_config):
        """Test emit adds event to queue."""
        from aragora.gauntlet.webhooks import WebhookEventType

        webhook_manager.register("test", sample_config)

        await webhook_manager.emit(
            WebhookEventType.GAUNTLET_COMPLETED,
            "gauntlet-123",
            {"verdict": "pass"},
        )

        assert not webhook_manager._delivery_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_skips_disabled_webhook(self, webhook_manager):
        """Test emit skips disabled webhooks."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(
            url="https://example.com/hook",
            enabled=False,
        )
        webhook_manager.register("disabled", config)

        await webhook_manager.emit(
            WebhookEventType.GAUNTLET_COMPLETED,
            "gauntlet-123",
            {},
        )

        assert webhook_manager._delivery_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_skips_unsubscribed_event(self, webhook_manager):
        """Test emit skips webhooks not subscribed to event."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(
            url="https://example.com/hook",
            events=[WebhookEventType.GAUNTLET_STARTED],  # Only started
        )
        webhook_manager.register("started-only", config)

        await webhook_manager.emit(
            WebhookEventType.GAUNTLET_COMPLETED,  # Completed event
            "gauntlet-123",
            {},
        )

        assert webhook_manager._delivery_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_gauntlet_started(self, webhook_manager):
        """Test emit_gauntlet_started helper."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(
            url="https://example.com/hook",
            events=[WebhookEventType.GAUNTLET_STARTED],
        )
        webhook_manager.register("test", config)

        await webhook_manager.emit_gauntlet_started(
            gauntlet_id="gauntlet-123",
            input_type="spec",
            input_summary="Test spec",
            agents=["claude", "gpt-4"],
        )

        assert not webhook_manager._delivery_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_gauntlet_completed(self, webhook_manager, sample_config):
        """Test emit_gauntlet_completed helper."""
        webhook_manager.register("test", sample_config)

        await webhook_manager.emit_gauntlet_completed(
            gauntlet_id="gauntlet-123",
            verdict="pass",
            confidence=0.95,
            total_findings=5,
            critical_count=0,
            high_count=1,
            robustness_score=0.8,
            duration_seconds=45.5,
        )

        assert not webhook_manager._delivery_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_gauntlet_failed(self, webhook_manager):
        """Test emit_gauntlet_failed helper."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(
            url="https://example.com/hook",
            events=[WebhookEventType.GAUNTLET_FAILED],
        )
        webhook_manager.register("test", config)

        await webhook_manager.emit_gauntlet_failed(
            gauntlet_id="gauntlet-123",
            error="Connection timeout",
        )

        assert not webhook_manager._delivery_queue.empty()

    @pytest.mark.asyncio
    async def test_emit_critical_finding(self, webhook_manager):
        """Test emit_critical_finding helper."""
        from aragora.gauntlet.webhooks import WebhookConfig, WebhookEventType

        config = WebhookConfig(
            url="https://example.com/hook",
            events=[WebhookEventType.FINDING_CRITICAL],
        )
        webhook_manager.register("test", config)

        await webhook_manager.emit_critical_finding(
            gauntlet_id="gauntlet-123",
            finding_id="finding-1",
            title="SQL Injection",
            category="security",
            description="Potential SQL injection vulnerability",
        )

        assert not webhook_manager._delivery_queue.empty()


# ===========================================================================
# Tests: WebhookManager Circuit Breaker
# ===========================================================================


class TestWebhookManagerCircuitBreaker:
    """Tests for WebhookManager circuit breaker."""

    def test_circuit_closed_initially(self, webhook_manager):
        """Test circuit is closed initially."""
        is_open = webhook_manager._is_circuit_open("https://example.com/hook")

        assert is_open is False

    def test_circuit_opens_after_failures(self, webhook_manager):
        """Test circuit opens after threshold failures."""
        url = "https://example.com/hook"

        # Record failures up to threshold
        for _ in range(webhook_manager._circuit_breaker_threshold):
            webhook_manager._record_failure(url)

        is_open = webhook_manager._is_circuit_open(url)

        assert is_open is True

    def test_circuit_stays_closed_below_threshold(self, webhook_manager):
        """Test circuit stays closed below threshold."""
        url = "https://example.com/hook"

        # Record failures below threshold
        for _ in range(webhook_manager._circuit_breaker_threshold - 1):
            webhook_manager._record_failure(url)

        is_open = webhook_manager._is_circuit_open(url)

        assert is_open is False

    def test_success_resets_circuit(self, webhook_manager):
        """Test successful delivery resets circuit."""
        url = "https://example.com/hook"

        # Open circuit
        for _ in range(webhook_manager._circuit_breaker_threshold):
            webhook_manager._record_failure(url)

        # Record success
        webhook_manager._record_success(url)

        is_open = webhook_manager._is_circuit_open(url)

        assert is_open is False


# ===========================================================================
# Tests: WebhookManager Start/Stop
# ===========================================================================


class TestWebhookManagerLifecycle:
    """Tests for WebhookManager lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_creates_worker(self, webhook_manager):
        """Test start creates delivery worker."""
        await webhook_manager.start()

        assert webhook_manager._running is True
        assert webhook_manager._worker_task is not None

        await webhook_manager.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, webhook_manager):
        """Test start is idempotent."""
        await webhook_manager.start()
        task1 = webhook_manager._worker_task

        await webhook_manager.start()
        task2 = webhook_manager._worker_task

        assert task1 is task2

        await webhook_manager.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_worker(self, webhook_manager):
        """Test stop cancels worker task."""
        await webhook_manager.start()
        await webhook_manager.stop()

        assert webhook_manager._running is False


# ===========================================================================
# Tests: Module Helper Functions
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_get_webhook_manager_singleton(self):
        """Test get_webhook_manager returns singleton."""
        from aragora.gauntlet import webhooks as webhooks_module

        # Reset singleton
        webhooks_module._webhook_manager = None

        manager1 = webhooks_module.get_webhook_manager()
        manager2 = webhooks_module.get_webhook_manager()

        assert manager1 is manager2

        # Cleanup
        webhooks_module._webhook_manager = None

    @pytest.mark.asyncio
    async def test_notify_gauntlet_completed(self):
        """Test notify_gauntlet_completed convenience function."""
        from aragora.gauntlet import webhooks as webhooks_module

        # Reset singleton
        webhooks_module._webhook_manager = None

        # Should not raise even with no webhooks configured
        await webhooks_module.notify_gauntlet_completed(
            gauntlet_id="gauntlet-123",
            verdict="pass",
            confidence=0.95,
            total_findings=5,
        )

        # Cleanup
        webhooks_module._webhook_manager = None
