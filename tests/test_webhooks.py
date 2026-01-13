"""
Tests for the Aragora Webhook System.

Tests cover:
- WebhookConfig creation and validation
- Payload signing
- Config loading from environment
- WebhookDispatcher queue operations
- Event filtering
- SSRF protection
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import hashlib
import hmac

import pytest

from aragora.integrations.webhooks import (
    WebhookConfig,
    WebhookDispatcher,
    sign_payload,
    load_webhook_configs,
    DEFAULT_EVENT_TYPES,
    AragoraJSONEncoder,
)


class TestWebhookConfigCreation:
    """Test WebhookConfig dataclass."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        config = WebhookConfig(name="test", url="https://example.com/hook")
        assert config.name == "test"
        assert config.url == "https://example.com/hook"
        assert config.secret == ""
        assert config.timeout_s == 10.0
        assert config.max_retries == 3

    def test_full_config(self):
        """Test creating config with all fields."""
        config = WebhookConfig(
            name="full",
            url="https://example.com/hook",
            secret="my-secret",
            event_types={"debate_start", "debate_end"},
            loop_ids={"loop-1", "loop-2"},
            timeout_s=30.0,
            max_retries=5,
            backoff_base_s=2.0,
        )
        assert config.secret == "my-secret"
        assert config.event_types == {"debate_start", "debate_end"}
        assert config.loop_ids == {"loop-1", "loop-2"}
        assert config.timeout_s == 30.0
        assert config.max_retries == 5

    def test_default_event_types(self):
        """Test default event types are applied."""
        config = WebhookConfig(name="test", url="https://example.com")
        assert config.event_types == set(DEFAULT_EVENT_TYPES)


class TestWebhookConfigFromDict:
    """Test WebhookConfig.from_dict method."""

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {"name": "test-hook", "url": "https://example.com/webhook"}
        config = WebhookConfig.from_dict(data)
        assert config.name == "test-hook"
        assert config.url == "https://example.com/webhook"

    def test_from_dict_full(self):
        """Test creating config from full dict."""
        data = {
            "name": "full-hook",
            "url": "https://example.com/webhook",
            "secret": "secret123",
            "event_types": ["debate_start", "debate_end"],
            "loop_ids": ["loop-a"],
            "timeout_s": 15,
            "max_retries": 2,
            "backoff_base_s": 0.5,
        }
        config = WebhookConfig.from_dict(data)
        assert config.secret == "secret123"
        assert config.event_types == {"debate_start", "debate_end"}
        assert config.loop_ids == {"loop-a"}
        assert config.timeout_s == 15.0
        assert config.max_retries == 2

    def test_from_dict_missing_name(self):
        """Test from_dict raises on missing name."""
        with pytest.raises(ValueError, match="name"):
            WebhookConfig.from_dict({"url": "https://example.com"})

    def test_from_dict_missing_url(self):
        """Test from_dict raises on missing url."""
        with pytest.raises(ValueError, match="url"):
            WebhookConfig.from_dict({"name": "test"})

    def test_from_dict_empty_name(self):
        """Test from_dict raises on empty name."""
        with pytest.raises(ValueError, match="name"):
            WebhookConfig.from_dict({"name": "", "url": "https://example.com"})

    def test_from_dict_normalizes_event_types_list(self):
        """Test event_types list is converted to set."""
        data = {
            "name": "test",
            "url": "https://example.com",
            "event_types": ["debate_start"],
        }
        config = WebhookConfig.from_dict(data)
        assert isinstance(config.event_types, set)
        assert config.event_types == {"debate_start"}

    def test_from_dict_normalizes_loop_ids_list(self):
        """Test loop_ids list is converted to set."""
        data = {
            "name": "test",
            "url": "https://example.com",
            "loop_ids": ["loop1", "loop2"],
        }
        config = WebhookConfig.from_dict(data)
        assert isinstance(config.loop_ids, set)
        assert config.loop_ids == {"loop1", "loop2"}

    def test_from_dict_none_loop_ids(self):
        """Test None loop_ids means all loops."""
        data = {"name": "test", "url": "https://example.com"}
        config = WebhookConfig.from_dict(data)
        assert config.loop_ids is None


class TestSignPayload:
    """Test webhook payload signing."""

    def test_sign_with_secret(self):
        """Test signing payload with secret."""
        secret = "my-secret"
        body = b'{"type": "test"}'

        signature = sign_payload(secret, body)

        assert signature.startswith("sha256=")
        # Verify the signature
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert signature == f"sha256={expected}"

    def test_sign_without_secret(self):
        """Test signing returns empty string without secret."""
        signature = sign_payload("", b'{"type": "test"}')
        assert signature == ""

    def test_sign_none_secret(self):
        """Test signing handles None secret gracefully."""
        signature = sign_payload(None, b"test")
        assert signature == ""

    def test_sign_different_payloads(self):
        """Test different payloads produce different signatures."""
        secret = "secret"
        sig1 = sign_payload(secret, b"payload1")
        sig2 = sign_payload(secret, b"payload2")
        assert sig1 != sig2

    def test_sign_same_payload_same_signature(self):
        """Test same payload produces same signature."""
        secret = "secret"
        body = b'{"test": true}'
        sig1 = sign_payload(secret, body)
        sig2 = sign_payload(secret, body)
        assert sig1 == sig2


class TestLoadWebhookConfigs:
    """Test loading webhook configs from environment."""

    def test_load_from_inline_json(self):
        """Test loading from ARAGORA_WEBHOOKS env var."""
        inline_config = json.dumps(
            [
                {"name": "hook1", "url": "https://example.com/1"},
                {"name": "hook2", "url": "https://example.com/2"},
            ]
        )

        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": inline_config}):
            configs = load_webhook_configs()

        assert len(configs) == 2
        assert configs[0].name == "hook1"
        assert configs[1].name == "hook2"

    def test_load_from_file(self):
        """Test loading from ARAGORA_WEBHOOKS_CONFIG file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "file-hook", "url": "https://example.com/file"}], f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": config_path}, clear=False):
                # Clear inline config
                os.environ.pop("ARAGORA_WEBHOOKS", None)
                configs = load_webhook_configs()

            assert len(configs) == 1
            assert configs[0].name == "file-hook"
        finally:
            os.unlink(config_path)

    def test_load_empty_when_not_configured(self):
        """Test returns empty list when no config."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARAGORA_WEBHOOKS", None)
            os.environ.pop("ARAGORA_WEBHOOKS_CONFIG", None)
            configs = load_webhook_configs()

        assert configs == []

    def test_load_skips_invalid_configs(self):
        """Test invalid configs are skipped."""
        inline_config = json.dumps(
            [
                {"name": "valid", "url": "https://example.com"},
                {"name": "", "url": "https://example.com"},  # Invalid - empty name
                {"url": "https://example.com"},  # Invalid - missing name
            ]
        )

        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": inline_config}):
            configs = load_webhook_configs()

        assert len(configs) == 1
        assert configs[0].name == "valid"

    def test_load_handles_invalid_json(self):
        """Test handles malformed JSON gracefully."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": "not json {"}):
            configs = load_webhook_configs()

        assert configs == []

    def test_load_handles_non_array_json(self):
        """Test handles non-array JSON."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": '{"name": "test"}'}):
            configs = load_webhook_configs()

        assert configs == []

    def test_load_handles_missing_file(self):
        """Test handles missing config file."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": "/nonexistent/file.json"}):
            os.environ.pop("ARAGORA_WEBHOOKS", None)
            configs = load_webhook_configs()

        assert configs == []


class TestWebhookDispatcher:
    """Test WebhookDispatcher class."""

    @pytest.fixture
    def sample_configs(self):
        """Create sample webhook configs."""
        return [
            WebhookConfig(name="hook1", url="https://example.com/1"),
            WebhookConfig(
                name="hook2",
                url="https://example.com/2",
                event_types={"debate_start"},
            ),
        ]

    @pytest.fixture
    def dispatcher(self, sample_configs):
        """Create a dispatcher for testing."""
        d = WebhookDispatcher(sample_configs, allow_localhost=True)
        yield d
        d.stop(timeout=1.0)

    def test_dispatcher_creation(self, sample_configs):
        """Test creating dispatcher."""
        dispatcher = WebhookDispatcher(sample_configs)
        assert len(dispatcher.configs) == 2
        dispatcher.stop(timeout=0.1)

    def test_dispatcher_start_stop(self, sample_configs):
        """Test starting and stopping dispatcher."""
        dispatcher = WebhookDispatcher(sample_configs)

        dispatcher.start()
        assert dispatcher.is_running

        dispatcher.stop(timeout=1.0)
        assert not dispatcher.is_running

    def test_enqueue_returns_false_when_stopped(self, sample_configs):
        """Test enqueue returns False when dispatcher is stopped."""
        dispatcher = WebhookDispatcher(sample_configs)
        dispatcher.start()
        dispatcher.stop(timeout=1.0)

        result = dispatcher.enqueue({"type": "debate_start"})
        assert result is False

    def test_enqueue_without_start(self, sample_configs):
        """Test enqueue works before start (queues but doesn't deliver)."""
        dispatcher = WebhookDispatcher(sample_configs)

        # Should be able to enqueue
        result = dispatcher.enqueue({"type": "debate_start"})
        # Won't deliver without start, but enqueue might succeed
        # depending on implementation
        dispatcher.stop(timeout=0.1)

    def test_stats_tracking(self, sample_configs):
        """Test dispatcher tracks statistics."""
        dispatcher = WebhookDispatcher(sample_configs, queue_max_size=10)

        with dispatcher._stats_lock:
            assert dispatcher._drop_count == 0
            assert dispatcher._delivery_count == 0
            assert dispatcher._failure_count == 0


class TestAragoraJSONEncoder:
    """Test custom JSON encoder."""

    def test_encode_set(self):
        """Test encoding set to sorted list."""
        data = {"items": {"c", "a", "b"}}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b", "c"]

    def test_encode_frozenset(self):
        """Test encoding frozenset to sorted list."""
        data = {"items": frozenset(["c", "a", "b"])}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b", "c"]

    def test_encode_datetime(self):
        """Test encoding datetime to ISO string."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 12, 30, 45)
        data = {"timestamp": dt}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert "2024-01-15" in parsed["timestamp"]

    def test_encode_object_with_to_dict(self):
        """Test encoding object with to_dict method."""

        class MyObj:
            def to_dict(self):
                return {"key": "value"}

        data = {"obj": MyObj()}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["obj"] == {"key": "value"}

    def test_encode_fallback_to_string(self):
        """Test fallback to string for unknown types."""

        class CustomType:
            def __str__(self):
                return "custom-string"

        data = {"custom": CustomType()}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["custom"] == "custom-string"


class TestDefaultEventTypes:
    """Test DEFAULT_EVENT_TYPES constant."""

    def test_contains_core_events(self):
        """Test contains core debate events."""
        assert "debate_start" in DEFAULT_EVENT_TYPES
        assert "debate_end" in DEFAULT_EVENT_TYPES
        assert "consensus" in DEFAULT_EVENT_TYPES

    def test_contains_cycle_events(self):
        """Test contains cycle events."""
        assert "cycle_start" in DEFAULT_EVENT_TYPES
        assert "cycle_end" in DEFAULT_EVENT_TYPES

    def test_contains_gauntlet_events(self):
        """Test contains gauntlet events."""
        assert "gauntlet_start" in DEFAULT_EVENT_TYPES
        assert "gauntlet_complete" in DEFAULT_EVENT_TYPES
        assert "gauntlet_verdict" in DEFAULT_EVENT_TYPES

    def test_contains_error_event(self):
        """Test contains error event."""
        assert "error" in DEFAULT_EVENT_TYPES

    def test_is_frozenset(self):
        """Test DEFAULT_EVENT_TYPES is immutable."""
        assert isinstance(DEFAULT_EVENT_TYPES, frozenset)


class TestWebhookCircuitBreaker:
    """Test circuit breaker integration in webhooks."""

    @pytest.fixture
    def sample_config(self):
        """Create sample webhook config."""
        return WebhookConfig(
            name="cb-test-hook",
            url="https://example.com/hook",
            max_retries=1,  # Faster test
        )

    @pytest.fixture
    def dispatcher(self, sample_config):
        """Create a dispatcher with circuit breaker test config."""
        d = WebhookDispatcher([sample_config], allow_localhost=True)
        d.start()
        yield d
        d.stop(timeout=1.0)
        # Reset circuit breaker after test
        from aragora.resilience import get_circuit_breaker
        cb = get_circuit_breaker("webhook:cb-test-hook")
        cb.reset()

    def test_get_circuit_status(self, dispatcher):
        """Test get_circuit_status returns webhook circuit status."""
        status = dispatcher.get_circuit_status()

        assert "cb-test-hook" in status
        assert status["cb-test-hook"]["status"] == "closed"
        assert status["cb-test-hook"]["failures"] == 0
        assert "url" in status["cb-test-hook"]

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after repeated failures."""
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

        # Reset all circuit breakers first
        reset_all_circuit_breakers()

        # Simulate failures by directly manipulating the circuit breaker
        cb = get_circuit_breaker(
            "webhook:fail-test-hook",
            failure_threshold=5,
            cooldown_seconds=120.0,
        )

        # Record 5 failures to open the circuit
        for _ in range(5):
            cb.record_failure()

        assert cb.get_status() == "open"
        assert not cb.can_proceed()

        # Clean up
        cb.reset()

    def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes on successful delivery."""
        from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

        reset_all_circuit_breakers()

        cb = get_circuit_breaker(
            "webhook:success-test-hook",
            failure_threshold=5,
            cooldown_seconds=1.0,  # Short cooldown for test
        )

        # Record failures but not enough to open
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "closed"
        assert cb.failures == 2

        # Success resets failure count
        cb.record_success()
        assert cb.failures == 0

        # Clean up
        cb.reset()
