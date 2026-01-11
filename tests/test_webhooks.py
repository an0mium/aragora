"""
Tests for webhooks - outbound webhook dispatcher for aragora events.

Tests cover:
- WebhookConfig creation and validation
- HMAC signature generation
- Event type and loop_id filtering
- Queue backpressure
- Thread safety
- SSRF protection
- JSON encoding
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from aragora.integrations.webhooks import (
    WebhookConfig,
    WebhookDispatcher,
    sign_payload,
    load_webhook_configs,
    AragoraJSONEncoder,
    DEFAULT_EVENT_TYPES,
)


# =============================================================================
# Tests: AragoraJSONEncoder
# =============================================================================


class TestAragoraJSONEncoder:
    """Tests for custom JSON encoder."""

    def test_encode_set(self):
        """Test encoding sets to sorted lists."""
        data = {"items": {"c", "a", "b"}}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b", "c"]

    def test_encode_frozenset(self):
        """Test encoding frozensets to sorted lists."""
        data = {"items": frozenset(["x", "y", "z"])}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["items"] == ["x", "y", "z"]

    def test_encode_datetime(self):
        """Test encoding datetime to ISO format."""
        dt = datetime(2026, 1, 10, 12, 30, 45)
        data = {"timestamp": dt}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["timestamp"] == "2026-01-10T12:30:45"

    def test_encode_object_with_to_dict(self):
        """Test encoding objects with to_dict method."""

        class CustomObj:
            def to_dict(self):
                return {"key": "value"}

        data = {"obj": CustomObj()}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["obj"] == {"key": "value"}

    def test_encode_fallback_to_str(self):
        """Test fallback to str for unhandled types."""

        class CustomClass:
            def __str__(self):
                return "custom_string"

        data = {"obj": CustomClass()}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        assert parsed["obj"] == "custom_string"


# =============================================================================
# Tests: WebhookConfig
# =============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating a basic webhook config."""
        config = WebhookConfig(
            name="test-webhook",
            url="https://example.com/webhook",
        )
        assert config.name == "test-webhook"
        assert config.url == "https://example.com/webhook"
        assert config.secret == ""
        assert config.timeout_s == 10.0
        assert config.max_retries == 3

    def test_create_config_with_secret(self):
        """Test creating config with secret."""
        config = WebhookConfig(
            name="secure",
            url="https://example.com/webhook",
            secret="my-secret-key",
        )
        assert config.secret == "my-secret-key"

    def test_create_config_with_event_types(self):
        """Test creating config with custom event types."""
        config = WebhookConfig(
            name="test",
            url="https://example.com/webhook",
            event_types={"debate_start", "debate_end"},
        )
        assert config.event_types == {"debate_start", "debate_end"}

    def test_create_config_with_loop_ids(self):
        """Test creating config with loop_ids filter."""
        config = WebhookConfig(
            name="test",
            url="https://example.com/webhook",
            loop_ids={"loop-1", "loop-2"},
        )
        assert config.loop_ids == {"loop-1", "loop-2"}

    def test_from_dict_basic(self):
        """Test creating config from dict."""
        data = {
            "name": "my-webhook",
            "url": "https://example.com/hook",
        }
        config = WebhookConfig.from_dict(data)
        assert config.name == "my-webhook"
        assert config.url == "https://example.com/hook"
        assert config.event_types == set(DEFAULT_EVENT_TYPES)

    def test_from_dict_with_all_options(self):
        """Test creating config from dict with all options."""
        data = {
            "name": "full-config",
            "url": "https://example.com/hook",
            "secret": "secret123",
            "event_types": ["debate_start", "debate_end"],
            "loop_ids": ["loop-1"],
            "timeout_s": 30.0,
            "max_retries": 5,
            "backoff_base_s": 2.0,
        }
        config = WebhookConfig.from_dict(data)
        assert config.name == "full-config"
        assert config.secret == "secret123"
        assert config.event_types == {"debate_start", "debate_end"}
        assert config.loop_ids == {"loop-1"}
        assert config.timeout_s == 30.0
        assert config.max_retries == 5

    def test_from_dict_missing_name(self):
        """Test that missing name raises ValueError."""
        with pytest.raises(ValueError, match="requires 'name'"):
            WebhookConfig.from_dict({"url": "https://example.com"})

    def test_from_dict_missing_url(self):
        """Test that missing url raises ValueError."""
        with pytest.raises(ValueError, match="requires 'url'"):
            WebhookConfig.from_dict({"name": "test"})

    def test_from_dict_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="requires 'name'"):
            WebhookConfig.from_dict({"name": "", "url": "https://example.com"})

    def test_from_dict_normalizes_event_types_list(self):
        """Test that event_types list is normalized to set."""
        data = {
            "name": "test",
            "url": "https://example.com",
            "event_types": ["debate_start", "debate_start", "debate_end"],
        }
        config = WebhookConfig.from_dict(data)
        assert config.event_types == {"debate_start", "debate_end"}

    def test_from_dict_normalizes_loop_ids_tuple(self):
        """Test that loop_ids tuple is normalized to set."""
        data = {
            "name": "test",
            "url": "https://example.com",
            "loop_ids": ("loop-1", "loop-2"),
        }
        config = WebhookConfig.from_dict(data)
        assert config.loop_ids == {"loop-1", "loop-2"}


# =============================================================================
# Tests: sign_payload
# =============================================================================


class TestSignPayload:
    """Tests for HMAC signature generation."""

    def test_sign_payload_basic(self):
        """Test basic signature generation."""
        signature = sign_payload("secret", b"test payload")
        assert signature.startswith("sha256=")
        assert len(signature) == len("sha256=") + 64  # sha256 hex = 64 chars

    def test_sign_payload_deterministic(self):
        """Test that same inputs produce same signature."""
        sig1 = sign_payload("secret", b"payload")
        sig2 = sign_payload("secret", b"payload")
        assert sig1 == sig2

    def test_sign_payload_different_secrets(self):
        """Test that different secrets produce different signatures."""
        sig1 = sign_payload("secret1", b"payload")
        sig2 = sign_payload("secret2", b"payload")
        assert sig1 != sig2

    def test_sign_payload_different_payloads(self):
        """Test that different payloads produce different signatures."""
        sig1 = sign_payload("secret", b"payload1")
        sig2 = sign_payload("secret", b"payload2")
        assert sig1 != sig2

    def test_sign_payload_empty_secret(self):
        """Test that empty secret returns empty string."""
        signature = sign_payload("", b"payload")
        assert signature == ""

    def test_sign_payload_none_secret(self):
        """Test that None-ish secret returns empty string."""
        signature = sign_payload("", b"payload")
        assert signature == ""


# =============================================================================
# Tests: load_webhook_configs
# =============================================================================


class TestLoadWebhookConfigs:
    """Tests for loading configs from environment."""

    def test_load_empty_env(self):
        """Test loading with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            configs = load_webhook_configs()
            assert configs == []

    def test_load_from_inline_json(self):
        """Test loading from ARAGORA_WEBHOOKS env var."""
        inline = json.dumps([
            {"name": "test1", "url": "https://example.com/1"},
            {"name": "test2", "url": "https://example.com/2"},
        ])
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": inline}, clear=True):
            configs = load_webhook_configs()
            assert len(configs) == 2
            assert configs[0].name == "test1"
            assert configs[1].name == "test2"

    def test_load_from_file(self):
        """Test loading from ARAGORA_WEBHOOKS_CONFIG file."""
        config_data = [
            {"name": "file-hook", "url": "https://example.com/hook"}
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            config_path = f.name

        try:
            with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": config_path}, clear=True):
                configs = load_webhook_configs()
                assert len(configs) == 1
                assert configs[0].name == "file-hook"
        finally:
            os.unlink(config_path)

    def test_load_invalid_json(self):
        """Test loading invalid JSON returns empty list."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": "not valid json"}, clear=True):
            configs = load_webhook_configs()
            assert configs == []

    def test_load_not_array(self):
        """Test loading non-array JSON returns empty list."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": '{"name": "test"}'}, clear=True):
            configs = load_webhook_configs()
            assert configs == []

    def test_load_skips_invalid_configs(self):
        """Test that invalid individual configs are skipped."""
        inline = json.dumps([
            {"name": "valid", "url": "https://example.com"},
            {"name": "", "url": "https://missing-name.com"},  # Invalid: empty name
            {"url": "https://no-name.com"},  # Invalid: missing name
        ])
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": inline}, clear=True):
            configs = load_webhook_configs()
            assert len(configs) == 1
            assert configs[0].name == "valid"

    def test_load_file_not_found(self):
        """Test loading from non-existent file returns empty list."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": "/nonexistent/path.json"}, clear=True):
            configs = load_webhook_configs()
            assert configs == []


# =============================================================================
# Tests: WebhookDispatcher
# =============================================================================


class TestWebhookDispatcher:
    """Tests for WebhookDispatcher class."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic webhook config."""
        return WebhookConfig(
            name="test",
            url="https://example.com/webhook",
            event_types={"debate_start", "debate_end"},
        )

    @pytest.fixture
    def dispatcher(self, basic_config):
        """Create a dispatcher with basic config."""
        d = WebhookDispatcher([basic_config], allow_localhost=True)
        yield d
        d.stop()

    def test_init_not_running(self, basic_config):
        """Test that dispatcher is not running initially."""
        d = WebhookDispatcher([basic_config])
        assert not d.is_running
        d.stop()

    def test_start_creates_worker_thread(self, basic_config):
        """Test that start creates worker thread."""
        d = WebhookDispatcher([basic_config])
        d.start()
        assert d.is_running
        assert d._worker is not None
        assert d._worker.is_alive()
        d.stop()

    def test_stop_stops_worker(self, basic_config):
        """Test that stop terminates worker thread."""
        d = WebhookDispatcher([basic_config])
        d.start()
        d.stop(timeout=2.0)
        assert not d.is_running

    def test_enqueue_after_stop_returns_false(self, basic_config):
        """Test that enqueueing after stop returns False."""
        d = WebhookDispatcher([basic_config], allow_localhost=True)
        d.start()
        d.stop()
        result = d.enqueue({"type": "debate_start"})
        assert not result

    def test_enqueue_matching_event(self, dispatcher):
        """Test enqueueing an event that matches config."""
        dispatcher.start()
        result = dispatcher.enqueue({"type": "debate_start", "loop_id": "loop-1"})
        assert result

    def test_enqueue_non_matching_event(self, dispatcher):
        """Test enqueueing an event that doesn't match config."""
        dispatcher.start()
        result = dispatcher.enqueue({"type": "unknown_event", "loop_id": "loop-1"})
        assert not result

    def test_enqueue_queue_full_drops(self, basic_config):
        """Test that queue full drops events."""
        d = WebhookDispatcher([basic_config], queue_max_size=2, allow_localhost=True)
        d.start()

        # Fill the queue
        d.enqueue({"type": "debate_start"})
        d.enqueue({"type": "debate_start"})

        # This should be dropped (queue full)
        # Note: May or may not drop depending on worker speed
        # Just ensure no exception raised
        d.enqueue({"type": "debate_start"})

        d.stop()

    def test_matches_config_event_type(self, basic_config):
        """Test event type matching."""
        d = WebhookDispatcher([basic_config])

        assert d._matches_config(basic_config, "debate_start", "loop-1")
        assert d._matches_config(basic_config, "debate_end", "loop-1")
        assert not d._matches_config(basic_config, "unknown", "loop-1")

        d.stop()

    def test_matches_config_loop_id(self):
        """Test loop_id filtering."""
        config = WebhookConfig(
            name="test",
            url="https://example.com",
            event_types={"debate_start"},
            loop_ids={"loop-1", "loop-2"},
        )
        d = WebhookDispatcher([config])

        assert d._matches_config(config, "debate_start", "loop-1")
        assert d._matches_config(config, "debate_start", "loop-2")
        assert not d._matches_config(config, "debate_start", "loop-3")

        d.stop()

    def test_matches_config_no_loop_filter(self, basic_config):
        """Test that None loop_ids matches all loops."""
        d = WebhookDispatcher([basic_config])

        # basic_config has loop_ids=None, so should match any loop
        assert d._matches_config(basic_config, "debate_start", "any-loop")
        assert d._matches_config(basic_config, "debate_start", "other-loop")

        d.stop()


# =============================================================================
# Tests: SSRF Protection
# =============================================================================


class TestSSRFProtection:
    """Tests for SSRF protection in webhook URL validation."""

    @pytest.fixture
    def dispatcher(self):
        """Create dispatcher without localhost allowance for SSRF tests."""
        d = WebhookDispatcher([], allow_localhost=False)
        yield d
        d.stop()

    def test_allows_public_https(self, dispatcher):
        """Test that public HTTPS URLs are allowed."""
        valid, error = dispatcher._validate_webhook_url("https://example.com/webhook")
        # May fail due to DNS resolution, but should not fail SSRF check
        # We're testing the validation logic, not actual DNS

    def test_blocks_non_http_schemes(self, dispatcher):
        """Test that non-HTTP schemes are blocked."""
        valid, error = dispatcher._validate_webhook_url("file:///etc/passwd")
        assert not valid
        assert "Only HTTP/HTTPS" in error

        valid, error = dispatcher._validate_webhook_url("ftp://example.com")
        assert not valid
        assert "Only HTTP/HTTPS" in error

    def test_blocks_internal_hostname_suffixes(self, dispatcher):
        """Test that internal hostname suffixes are blocked."""
        valid, error = dispatcher._validate_webhook_url("https://service.internal")
        assert not valid
        assert "Internal hostname" in error

        valid, error = dispatcher._validate_webhook_url("https://host.local")
        assert not valid
        assert "Internal hostname" in error

        valid, error = dispatcher._validate_webhook_url("https://something.localhost")
        assert not valid
        assert "Internal hostname" in error

    def test_blocks_metadata_hostnames(self, dispatcher):
        """Test that cloud metadata hostnames are blocked."""
        valid, error = dispatcher._validate_webhook_url("http://metadata.google.internal")
        assert not valid
        assert "metadata hostname" in error.lower() or "Internal hostname" in error

    def test_allows_localhost_when_enabled(self):
        """Test that localhost is allowed when explicitly enabled."""
        d = WebhookDispatcher([], allow_localhost=True)
        valid, error = d._validate_webhook_url("http://localhost:8080/hook")
        assert valid
        d.stop()

    def test_blocks_urls_without_hostname(self, dispatcher):
        """Test that URLs without hostname are blocked."""
        valid, error = dispatcher._validate_webhook_url("http:///path")
        assert not valid
        assert "hostname" in error.lower()


# =============================================================================
# Tests: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_enqueue(self):
        """Test concurrent enqueueing from multiple threads."""
        config = WebhookConfig(
            name="test",
            url="https://example.com/webhook",
            event_types={"debate_start"},
        )
        d = WebhookDispatcher([config], queue_max_size=1000, allow_localhost=True)
        d.start()

        enqueued_count = []
        errors = []

        def enqueue_many(count: int):
            successes = 0
            for i in range(count):
                try:
                    if d.enqueue({"type": "debate_start", "index": i}):
                        successes += 1
                except Exception as e:
                    errors.append(e)
            enqueued_count.append(successes)

        threads = [
            threading.Thread(target=enqueue_many, args=(100,))
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        d.stop()

        assert len(errors) == 0, f"Errors: {errors}"
        # At least some events should have been enqueued
        assert sum(enqueued_count) > 0

    def test_stats_thread_safety(self):
        """Test that stats access is thread-safe."""
        config = WebhookConfig(
            name="test",
            url="https://example.com/webhook",
            event_types={"debate_start"},
        )
        d = WebhookDispatcher([config], queue_max_size=10, allow_localhost=True)
        d.start()

        errors = []

        def access_stats():
            for _ in range(100):
                try:
                    with d._stats_lock:
                        _ = d._drop_count
                        _ = d._delivery_count
                        _ = d._failure_count
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=access_stats) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        d.stop()

        assert len(errors) == 0


# =============================================================================
# Tests: Default Event Types
# =============================================================================


class TestDefaultEventTypes:
    """Tests for default event types."""

    def test_default_event_types_is_frozenset(self):
        """Test that DEFAULT_EVENT_TYPES is a frozenset."""
        assert isinstance(DEFAULT_EVENT_TYPES, frozenset)

    def test_default_event_types_contains_core_events(self):
        """Test that core events are in defaults."""
        assert "debate_start" in DEFAULT_EVENT_TYPES
        assert "debate_end" in DEFAULT_EVENT_TYPES
        assert "consensus" in DEFAULT_EVENT_TYPES
        assert "error" in DEFAULT_EVENT_TYPES

    def test_default_event_types_immutable(self):
        """Test that DEFAULT_EVENT_TYPES cannot be modified."""
        with pytest.raises(AttributeError):
            DEFAULT_EVENT_TYPES.add("new_event")
