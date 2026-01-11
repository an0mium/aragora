"""Tests for outbound webhook dispatcher."""

import hashlib
import hmac
import http.server
import json
import os
import queue
import threading
import time
import unittest
from datetime import datetime
from unittest.mock import patch

from aragora.integrations.webhooks import (
    AragoraJSONEncoder,
    WebhookConfig,
    WebhookDispatcher,
    sign_payload,
    load_webhook_configs,
    DEFAULT_EVENT_TYPES,
    shutdown_dispatcher,
)
from aragora.server.handlers.base import clear_cache


class TestAragoraJSONEncoder(unittest.TestCase):
    """Tests for custom JSON encoder."""

    def test_encodes_set_as_sorted_list(self):
        data = {"agents": {"charlie", "alice", "bob"}}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["agents"], ["alice", "bob", "charlie"])

    def test_encodes_frozenset_as_sorted_list(self):
        data = {"tags": frozenset(["z", "a", "m"])}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["tags"], ["a", "m", "z"])

    def test_encodes_datetime_as_iso(self):
        dt = datetime(2024, 1, 15, 12, 30, 45)
        data = {"timestamp": dt}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["timestamp"], "2024-01-15T12:30:45")

    def test_encodes_object_with_to_dict(self):
        class CustomObj:
            def to_dict(self):
                return {"custom": "value", "id": 42}

        data = {"obj": CustomObj()}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["obj"], {"custom": "value", "id": 42})

    def test_fallback_to_str_for_unknown_types(self):
        class UnserializableObj:
            def __str__(self):
                return "UnserializableObj<123>"

        data = {"obj": UnserializableObj()}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["obj"], "UnserializableObj<123>")

    def test_encodes_nested_structures(self):
        data = {
            "type": "debate_start",
            "data": {
                "participants": {"agent_a", "agent_b"},
                "started_at": datetime(2024, 1, 1, 0, 0, 0),
            }
        }
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["data"]["participants"], ["agent_a", "agent_b"])
        self.assertEqual(parsed["data"]["started_at"], "2024-01-01T00:00:00")


class TestSignPayload(unittest.TestCase):
    def test_generates_correct_signature(self):
        secret = "test-secret-key"
        body = b'{"type": "debate_start", "data": {}}'
        sig = sign_payload(secret, body)

        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        self.assertEqual(sig, f"sha256={expected}")

    def test_empty_secret_returns_empty_string(self):
        self.assertEqual(sign_payload("", b"any data"), "")

    def test_signature_changes_with_different_payloads(self):
        secret = "key"
        sig1 = sign_payload(secret, b"payload1")
        sig2 = sign_payload(secret, b"payload2")
        self.assertNotEqual(sig1, sig2)


class TestWebhookConfig(unittest.TestCase):
    def test_from_dict_minimal(self):
        cfg = WebhookConfig.from_dict({"name": "test", "url": "http://example.com"})
        self.assertEqual(cfg.name, "test")
        self.assertEqual(cfg.url, "http://example.com")
        self.assertEqual(cfg.event_types, set(DEFAULT_EVENT_TYPES))
        self.assertIsNone(cfg.loop_ids)

    def test_from_dict_full(self):
        cfg = WebhookConfig.from_dict({
            "name": "slack",
            "url": "https://hooks.slack.com/xxx",
            "secret": "my-secret",
            "event_types": ["debate_start", "consensus"],
            "loop_ids": ["loop-1", "loop-2"],
            "timeout_s": 5.0,
            "max_retries": 2,
        })
        self.assertEqual(cfg.secret, "my-secret")
        self.assertEqual(cfg.event_types, {"debate_start", "consensus"})
        self.assertEqual(cfg.loop_ids, {"loop-1", "loop-2"})
        self.assertEqual(cfg.timeout_s, 5.0)
        self.assertEqual(cfg.max_retries, 2)

    def test_from_dict_missing_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            WebhookConfig.from_dict({"url": "http://example.com"})
        self.assertIn("name", str(ctx.exception))

    def test_from_dict_missing_url_raises(self):
        with self.assertRaises(ValueError) as ctx:
            WebhookConfig.from_dict({"name": "test"})
        self.assertIn("url", str(ctx.exception))

    def test_from_dict_does_not_mutate_input(self):
        original = {"name": "test", "url": "http://x", "event_types": ["a", "b"]}
        original_copy = original.copy()
        WebhookConfig.from_dict(original)
        self.assertEqual(original, original_copy)

    def test_from_dict_invalid_event_types_uses_default(self):
        cfg = WebhookConfig.from_dict({
            "name": "test",
            "url": "http://x",
            "event_types": "not-a-list"  # Invalid type
        })
        self.assertEqual(cfg.event_types, set(DEFAULT_EVENT_TYPES))

    def test_from_dict_event_types_already_set(self):
        """event_types that's already a set should pass through."""
        cfg = WebhookConfig.from_dict({
            "name": "test",
            "url": "http://x",
            "event_types": {"debate_start", "consensus"}  # Already a set
        })
        self.assertEqual(cfg.event_types, {"debate_start", "consensus"})

    def test_from_dict_loop_ids_already_set(self):
        """loop_ids that's already a set should pass through."""
        cfg = WebhookConfig.from_dict({
            "name": "test",
            "url": "http://x",
            "loop_ids": {"loop-1", "loop-2"}  # Already a set
        })
        self.assertEqual(cfg.loop_ids, {"loop-1", "loop-2"})

    def test_from_dict_invalid_loop_ids_type_uses_none(self):
        """Invalid loop_ids type should default to None (all loops)."""
        cfg = WebhookConfig.from_dict({
            "name": "test",
            "url": "http://x",
            "loop_ids": "not-a-list"  # Invalid type
        })
        self.assertIsNone(cfg.loop_ids)


class TestLoadWebhookConfigs(unittest.TestCase):
    def test_loads_from_env_inline(self):
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": '[{"name": "test", "url": "http://x"}]'}, clear=False):
            configs = load_webhook_configs()
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].name, "test")

    def test_returns_empty_on_invalid_json(self):
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": "not valid json"}, clear=False):
            configs = load_webhook_configs()
            self.assertEqual(configs, [])

    def test_skips_invalid_configs_in_array(self):
        with patch.dict(os.environ, {
            "ARAGORA_WEBHOOKS": '[{"name": "good", "url": "http://x"}, {"bad": "config"}]'
        }, clear=False):
            configs = load_webhook_configs()
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].name, "good")

    def test_returns_empty_when_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            configs = load_webhook_configs()
            self.assertEqual(configs, [])

    def test_returns_empty_when_inline_not_array(self):
        """ARAGORA_WEBHOOKS must be a JSON array."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS": '{"name": "test", "url": "http://x"}'}, clear=False):
            configs = load_webhook_configs()
            self.assertEqual(configs, [])

    def test_loads_from_config_file(self):
        """Should load configs from ARAGORA_WEBHOOKS_CONFIG file path."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"name": "file-test", "url": "http://file-test.com"}], f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": config_path}, clear=True):
                configs = load_webhook_configs()
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0].name, "file-test")
        finally:
            os.unlink(config_path)

    def test_returns_empty_when_config_file_not_found(self):
        """Should return empty list if config file doesn't exist."""
        with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": "/nonexistent/path.json"}, clear=True):
            configs = load_webhook_configs()
            self.assertEqual(configs, [])

    def test_returns_empty_when_config_file_not_array(self):
        """Config file must contain a JSON array."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"name": "test", "url": "http://x"}, f)  # Object, not array
            config_path = f.name

        try:
            with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": config_path}, clear=True):
                configs = load_webhook_configs()
                self.assertEqual(configs, [])
        finally:
            os.unlink(config_path)

    def test_skips_invalid_configs_in_file(self):
        """Should skip invalid individual configs in file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"name": "good", "url": "http://good.com"},
                {"invalid": "config"},  # Missing required fields
            ], f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": config_path}, clear=True):
                configs = load_webhook_configs()
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0].name, "good")
        finally:
            os.unlink(config_path)

    def test_returns_empty_on_config_file_parse_error(self):
        """Should return empty list if config file contains invalid JSON."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {]")
            config_path = f.name

        try:
            with patch.dict(os.environ, {"ARAGORA_WEBHOOKS_CONFIG": config_path}, clear=True):
                configs = load_webhook_configs()
                self.assertEqual(configs, [])
        finally:
            os.unlink(config_path)


class TestEventFiltering(unittest.TestCase):
    def test_matches_by_event_type(self):
        cfg = WebhookConfig(name="t", url="http://x", event_types={"debate_start", "consensus"})
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)

        self.assertTrue(dispatcher._matches_config(cfg, "debate_start", ""))
        self.assertTrue(dispatcher._matches_config(cfg, "consensus", ""))
        self.assertFalse(dispatcher._matches_config(cfg, "agent_message", ""))

    def test_matches_by_loop_id(self):
        cfg = WebhookConfig(name="t", url="http://x", loop_ids={"loop-abc"})
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)

        self.assertTrue(dispatcher._matches_config(cfg, "debate_start", "loop-abc"))
        self.assertFalse(dispatcher._matches_config(cfg, "debate_start", "loop-xyz"))

    def test_none_loop_ids_matches_all(self):
        cfg = WebhookConfig(name="t", url="http://x", loop_ids=None)
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)

        self.assertTrue(dispatcher._matches_config(cfg, "debate_start", "any-loop"))


class TestQueueBehavior(unittest.TestCase):
    def test_enqueue_returns_false_when_no_matching_config(self):
        cfg = WebhookConfig(name="t", url="http://x", event_types={"consensus"})
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)

        result = dispatcher.enqueue({"type": "agent_message"})
        self.assertFalse(result)

    def test_queue_overflow_drops_events(self):
        cfg = WebhookConfig(name="t", url="http://x")
        dispatcher = WebhookDispatcher([cfg], queue_max_size=3, allow_localhost=True)
        dispatcher.start()  # Must start before enqueue works

        # Fill queue
        for i in range(3):
            self.assertTrue(dispatcher.enqueue({"type": "debate_start", "i": i}))

        # Next should drop
        self.assertFalse(dispatcher.enqueue({"type": "debate_start", "i": 3}))
        self.assertEqual(dispatcher.stats["dropped"], 1)

    def test_queue_size_from_env(self):
        with patch.dict(os.environ, {"ARAGORA_WEBHOOK_QUEUE_SIZE": "50"}, clear=False):
            cfg = WebhookConfig(name="t", url="http://x")
            dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
            self.assertEqual(dispatcher._queue.maxsize, 50)


class TestWebhookDelivery(unittest.TestCase):
    """Integration tests with real HTTP server."""

    def setUp(self):
        self.received = []
        self.port = 19876
        self.server = None
        self.response_code = 200

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def _start_server(self, status_code=200):
        self.response_code = status_code
        received = self.received
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                received.append({
                    "body": json.loads(body),
                    "headers": dict(self.headers),
                })
                self.send_response(parent.response_code)
                self.end_headers()

            def log_message(self, *args):
                pass

        class ReusableHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        self.server = ReusableHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.1)

    def test_delivers_event_with_correct_headers(self):
        self._start_server(200)

        cfg = WebhookConfig(
            name="test-hook",
            url=f"http://127.0.0.1:{self.port}/webhook",
            secret="my-secret",
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({
            "type": "debate_start",
            "loop_id": "test-loop",
            "timestamp": 1234567890,
        })

        time.sleep(0.5)
        dispatcher.stop()

        self.assertEqual(len(self.received), 1)
        req = self.received[0]

        self.assertEqual(req["body"]["type"], "debate_start")
        self.assertEqual(req["body"]["loop_id"], "test-loop")
        self.assertEqual(req["headers"]["X-Aragora-Event-Type"], "debate_start")
        self.assertEqual(req["headers"]["X-Aragora-Loop-Id"], "test-loop")
        self.assertTrue(req["headers"]["X-Aragora-Signature"].startswith("sha256="))

    def test_handles_set_in_payload(self):
        self._start_server(200)

        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({
            "type": "debate_start",
            "data": {"agents": {"alice", "bob"}},
        })

        time.sleep(0.5)
        dispatcher.stop()

        self.assertEqual(len(self.received), 1)
        # Should be a list, not a string repr of set
        agents = self.received[0]["body"]["data"]["agents"]
        self.assertIsInstance(agents, list)
        self.assertEqual(sorted(agents), ["alice", "bob"])

    def test_stats_tracking(self):
        self._start_server(200)

        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        dispatcher.enqueue({"type": "consensus"})
        time.sleep(0.5)

        stats = dispatcher.stats
        self.assertEqual(stats["delivered"], 2)
        self.assertEqual(stats["failed"], 0)

        dispatcher.stop()


class TestRetryLogic(unittest.TestCase):
    """Tests for webhook retry behavior."""

    def setUp(self):
        # Clear any global state from previous tests
        shutdown_dispatcher()
        clear_cache()
        self.received = []
        self.port = 19877
        self.server = None
        self.response_codes = []  # Queue of response codes to return
        self.retry_after = None

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        # Clean up global state
        shutdown_dispatcher()
        clear_cache()

    def _start_server(self, status_codes):
        """Start server that returns different status codes for each request."""
        self.response_codes = list(status_codes)
        received = self.received
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                received.append({
                    "body": json.loads(body),
                    "headers": dict(self.headers),
                    "time": time.time(),
                })
                # Pop next status code or use 200 as default
                code = parent.response_codes.pop(0) if parent.response_codes else 200
                self.send_response(code)
                if parent.retry_after and code == 429:
                    self.send_header("Retry-After", str(parent.retry_after))
                self.end_headers()

            def log_message(self, *args):
                pass

        class ReusableHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        self.server = ReusableHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.1)

    def test_retries_on_5xx(self):
        """Should retry with exponential backoff on 5xx responses."""
        self._start_server([500, 500, 200])  # Fail twice, then succeed

        cfg = WebhookConfig(
            name="test",
            url=f"http://127.0.0.1:{self.port}/x",
            max_retries=3,
            backoff_base_s=0.05,  # Fast for testing
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(1.0)  # Wait for retries
        dispatcher.stop()

        # Should have tried 3 times total
        self.assertEqual(len(self.received), 3)
        self.assertEqual(dispatcher.stats["delivered"], 1)
        self.assertEqual(dispatcher.stats["failed"], 0)

    def test_retries_on_429(self):
        """Should retry on 429 Too Many Requests."""
        self._start_server([429, 200])

        cfg = WebhookConfig(
            name="test",
            url=f"http://127.0.0.1:{self.port}/x",
            max_retries=3,
            backoff_base_s=0.05,
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(0.5)
        dispatcher.stop()

        self.assertEqual(len(self.received), 2)
        self.assertEqual(dispatcher.stats["delivered"], 1)

    def test_respects_retry_after_header(self):
        """Should use Retry-After header when present on 429."""
        self.retry_after = 1  # 1 second
        self._start_server([429, 200])

        cfg = WebhookConfig(
            name="test",
            url=f"http://127.0.0.1:{self.port}/x",
            max_retries=3,
            backoff_base_s=0.01,  # Would be very fast without Retry-After
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(1.5)  # Wait for Retry-After delay
        dispatcher.stop()

        self.assertEqual(len(self.received), 2)
        # Check that there was at least ~1 second between requests
        if len(self.received) >= 2:
            elapsed = self.received[1]["time"] - self.received[0]["time"]
            self.assertGreaterEqual(elapsed, 0.9)  # Allow some tolerance

    def test_no_retry_on_4xx(self):
        """Should NOT retry on 4xx errors (except 429)."""
        self._start_server([400])

        cfg = WebhookConfig(
            name="test",
            url=f"http://127.0.0.1:{self.port}/x",
            max_retries=3,
            backoff_base_s=0.05,
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(0.3)
        dispatcher.stop()

        # Should only try once - 4xx is permanent failure
        self.assertEqual(len(self.received), 1)
        self.assertEqual(dispatcher.stats["failed"], 1)
        self.assertEqual(dispatcher.stats["delivered"], 0)

    def test_max_retries_exhausted(self):
        """Should fail after max_retries exhausted."""
        self._start_server([500, 500, 500])  # All failures

        cfg = WebhookConfig(
            name="test",
            url=f"http://127.0.0.1:{self.port}/x",
            max_retries=3,
            backoff_base_s=0.02,
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(0.5)
        dispatcher.stop()

        # Should have tried exactly max_retries times
        self.assertEqual(len(self.received), 3)
        self.assertEqual(dispatcher.stats["failed"], 1)
        self.assertEqual(dispatcher.stats["delivered"], 0)

    def test_network_error_triggers_retry(self):
        """Should retry on network errors (connection refused, timeout)."""
        # Don't start server - connection will be refused
        cfg = WebhookConfig(
            name="test",
            url="http://127.0.0.1:19999/x",  # Nothing listening here
            max_retries=2,
            backoff_base_s=0.02,
            timeout_s=0.1,
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(0.5)
        dispatcher.stop()

        # Should have tried max_retries times and then failed
        self.assertEqual(dispatcher.stats["failed"], 1)
        self.assertEqual(dispatcher.stats["delivered"], 0)


class TestTimeoutBehavior(unittest.TestCase):
    """Tests for webhook timeout handling."""

    def setUp(self):
        self.port = 19878
        self.server = None
        self.delay = 0

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def _start_slow_server(self, delay_seconds):
        """Start server that delays before responding."""
        self.delay = delay_seconds
        parent = self

        class SlowHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(parent.delay)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        class ReusableHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        self.server = ReusableHTTPServer(("127.0.0.1", self.port), SlowHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.1)

    def test_timeout_is_respected(self):
        """Should timeout if server takes too long."""
        self._start_slow_server(2.0)  # 2 second delay

        cfg = WebhookConfig(
            name="test",
            url=f"http://127.0.0.1:{self.port}/x",
            timeout_s=0.2,  # 200ms timeout
            max_retries=1,  # No retries
        )
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(0.5)  # Should timeout before 2 seconds
        dispatcher.stop()

        # Should have failed due to timeout
        self.assertEqual(dispatcher.stats["failed"], 1)

    def test_custom_timeout_per_config(self):
        """Each webhook can have different timeout."""
        # This tests that WebhookConfig.timeout_s is used
        cfg = WebhookConfig(
            name="test",
            url="http://127.0.0.1:19999/x",
            timeout_s=0.05,  # Very short timeout
        )
        self.assertEqual(cfg.timeout_s, 0.05)


class TestConcurrency(unittest.TestCase):
    """Tests for thread safety and concurrent operations."""

    def setUp(self):
        self.port = 19879
        self.server = None
        self.received = []
        self.lock = threading.Lock()

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def _start_server(self):
        received = self.received
        lock = self.lock

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                with lock:
                    received.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        class ReusableHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        self.server = ReusableHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.1)

    def test_concurrent_enqueue(self):
        """Multiple threads can enqueue simultaneously."""
        self._start_server()

        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        num_threads = 10
        events_per_thread = 5

        def enqueue_events(thread_id):
            for i in range(events_per_thread):
                dispatcher.enqueue({
                    "type": "debate_start",
                    "thread": thread_id,
                    "event": i,
                })

        threads = [
            threading.Thread(target=enqueue_events, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for all to be processed
        time.sleep(1.0)
        dispatcher.stop()

        # All events should be delivered
        total_expected = num_threads * events_per_thread
        self.assertEqual(len(self.received), total_expected)

    def test_stats_thread_safe(self):
        """Stats should be accurate under concurrent load."""
        self._start_server()

        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        num_events = 50

        for i in range(num_events):
            dispatcher.enqueue({"type": "debate_start", "i": i})

        time.sleep(1.0)
        dispatcher.stop()

        # Stats should match actual deliveries
        self.assertEqual(dispatcher.stats["delivered"], num_events)
        self.assertEqual(len(self.received), num_events)


class TestLifecycle(unittest.TestCase):
    """Tests for dispatcher lifecycle (start/stop)."""

    def setUp(self):
        self.port = 19880
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def _start_server(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        class ReusableHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        self.server = ReusableHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.1)

    def test_stop_sets_running_false(self):
        """Stop should prevent further enqueueing."""
        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        self.assertTrue(dispatcher.is_running)
        dispatcher.stop()
        self.assertFalse(dispatcher.is_running)

        # Enqueue after stop should return False
        result = dispatcher.enqueue({"type": "debate_start"})
        self.assertFalse(result)

    def test_graceful_shutdown_logs_stats(self):
        """Stop should log final statistics."""
        self._start_server()

        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        time.sleep(0.3)

        # Stop should complete without error
        dispatcher.stop(timeout=2.0)
        self.assertFalse(dispatcher.is_running)

    def test_start_is_idempotent(self):
        """Calling start() twice should not create duplicate workers."""
        cfg = WebhookConfig(name="t", url=f"http://127.0.0.1:{self.port}/x")
        dispatcher = WebhookDispatcher([cfg], allow_localhost=True)

        dispatcher.start()
        worker1 = dispatcher._worker

        dispatcher.start()  # Second call
        worker2 = dispatcher._worker

        # Should be the same worker
        self.assertIs(worker1, worker2)
        dispatcher.stop()


if __name__ == "__main__":
    unittest.main()