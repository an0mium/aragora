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
)


class TestAragoraJSONEncoder(unittest.TestCase):
    """Tests for custom JSON encoder."""

    def test_encodes_set_as_sorted_list(self):
        data = {"agents": {"charlie", "alice", "bob"}}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["agents"], ["alice", "bob", "charlie"])

    def test_encodes_datetime_as_iso(self):
        dt = datetime(2024, 1, 15, 12, 30, 45)
        data = {"timestamp": dt}
        result = json.dumps(data, cls=AragoraJSONEncoder)
        parsed = json.loads(result)
        self.assertEqual(parsed["timestamp"], "2024-01-15T12:30:45")

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


class TestEventFiltering(unittest.TestCase):
    def test_matches_by_event_type(self):
        cfg = WebhookConfig(name="t", url="http://x", event_types={"debate_start", "consensus"})
        dispatcher = WebhookDispatcher([cfg])

        self.assertTrue(dispatcher._matches_config(cfg, "debate_start", ""))
        self.assertTrue(dispatcher._matches_config(cfg, "consensus", ""))
        self.assertFalse(dispatcher._matches_config(cfg, "agent_message", ""))

    def test_matches_by_loop_id(self):
        cfg = WebhookConfig(name="t", url="http://x", loop_ids={"loop-abc"})
        dispatcher = WebhookDispatcher([cfg])

        self.assertTrue(dispatcher._matches_config(cfg, "debate_start", "loop-abc"))
        self.assertFalse(dispatcher._matches_config(cfg, "debate_start", "loop-xyz"))

    def test_none_loop_ids_matches_all(self):
        cfg = WebhookConfig(name="t", url="http://x", loop_ids=None)
        dispatcher = WebhookDispatcher([cfg])

        self.assertTrue(dispatcher._matches_config(cfg, "debate_start", "any-loop"))


class TestQueueBehavior(unittest.TestCase):
    def test_enqueue_returns_false_when_no_matching_config(self):
        cfg = WebhookConfig(name="t", url="http://x", event_types={"consensus"})
        dispatcher = WebhookDispatcher([cfg])

        result = dispatcher.enqueue({"type": "agent_message"})
        self.assertFalse(result)

    def test_queue_overflow_drops_events(self):
        cfg = WebhookConfig(name="t", url="http://x")
        dispatcher = WebhookDispatcher([cfg], queue_max_size=3)
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
            dispatcher = WebhookDispatcher([cfg])
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
        dispatcher = WebhookDispatcher([cfg])
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
        dispatcher = WebhookDispatcher([cfg])
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
        dispatcher = WebhookDispatcher([cfg])
        dispatcher.start()

        dispatcher.enqueue({"type": "debate_start"})
        dispatcher.enqueue({"type": "consensus"})
        time.sleep(0.5)

        stats = dispatcher.stats
        self.assertEqual(stats["delivered"], 2)
        self.assertEqual(stats["failed"], 0)

        dispatcher.stop()


if __name__ == "__main__":
    unittest.main()