"""Tests for streaming connector handler.

Covers all routes and behavior of StreamingConnectorHandler:
    GET  /api/streaming/connectors                  - List all connectors
    GET  /api/streaming/connectors/{type}           - Connector details
    GET  /api/streaming/connectors/{type}/health    - Health check
    GET  /api/streaming/connectors/{type}/config    - Get config
    PUT  /api/streaming/connectors/{type}/config    - Update config
    POST /api/streaming/connectors/{type}/connect   - Connect to broker
    POST /api/streaming/connectors/{type}/disconnect - Disconnect from broker
    POST /api/streaming/connectors/{type}/test      - Test connectivity

Also tests: can_handle routing, invalid connector types, state transitions,
connect/disconnect idempotency, error handling, and edge cases.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.streaming.handler import (
    CONNECTOR_TYPES,
    StreamingConnectorHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    if hasattr(result, "to_dict"):
        d = result.to_dict()
        return d.get("body", d)
    if isinstance(result, dict):
        return result.get("body", result)
    try:
        body, _s, _ = result
        return body if isinstance(body, dict) else {}
    except (TypeError, ValueError):
        return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    try:
        _, status, _ = result
        return status
    except (TypeError, ValueError):
        return 200


class MockHTTPHandler:
    """Mock HTTP handler used by BaseHandler methods."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a StreamingConnectorHandler instance."""
    return StreamingConnectorHandler(server_context={})


@pytest.fixture
def http():
    """Create a default mock HTTP handler (no body)."""
    return MockHTTPHandler()


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_root_list(self, handler):
        assert handler.can_handle("/api/streaming/connectors") is True

    def test_handles_root_list_trailing_slash(self, handler):
        assert handler.can_handle("/api/streaming/connectors/") is True

    def test_handles_connector_type(self, handler):
        assert handler.can_handle("/api/streaming/connectors/kafka") is True

    def test_handles_health(self, handler):
        assert handler.can_handle("/api/streaming/connectors/kafka/health") is True

    def test_handles_config(self, handler):
        assert handler.can_handle("/api/streaming/connectors/rabbitmq/config") is True

    def test_handles_connect(self, handler):
        assert handler.can_handle("/api/streaming/connectors/snssqs/connect") is True

    def test_handles_disconnect(self, handler):
        assert handler.can_handle("/api/streaming/connectors/kafka/disconnect") is True

    def test_handles_test(self, handler):
        assert handler.can_handle("/api/streaming/connectors/kafka/test") is True

    def test_rejects_non_streaming_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/streaming") is False

    def test_rejects_close_prefix(self, handler):
        assert handler.can_handle("/api/streaming/connect") is False


# ===========================================================================
# GET /api/streaming/connectors — List all connectors
# ===========================================================================


class TestListConnectors:
    """Test GET /api/streaming/connectors."""

    def test_list_all_connectors(self, handler, http):
        result = handler.handle("/api/streaming/connectors", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert isinstance(body, list)
        # All three connector types should be present
        types_in_response = {c["type"] for c in body}
        assert types_in_response == CONNECTOR_TYPES

    def test_list_connectors_trailing_slash(self, handler, http):
        result = handler.handle("/api/streaming/connectors/", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert isinstance(body, list)
        assert len(body) == 3

    def test_list_connectors_default_status(self, handler, http):
        result = handler.handle("/api/streaming/connectors", {}, http)
        body = _body(result)
        for connector in body:
            assert connector["status"] == "disconnected"

    def test_list_connectors_have_health(self, handler, http):
        result = handler.handle("/api/streaming/connectors", {}, http)
        body = _body(result)
        for connector in body:
            health = connector["health"]
            assert health["healthy"] is False  # All disconnected initially
            assert health["error"] == "Not connected"

    def test_list_connectors_have_config(self, handler, http):
        result = handler.handle("/api/streaming/connectors", {}, http)
        body = _body(result)
        for connector in body:
            assert "config" in connector
            assert isinstance(connector["config"], dict)

    def test_list_connectors_have_timestamps(self, handler, http):
        result = handler.handle("/api/streaming/connectors", {}, http)
        body = _body(result)
        for connector in body:
            assert "created_at" in connector
            assert "updated_at" in connector


# ===========================================================================
# GET /api/streaming/connectors/{type} — Connector detail
# ===========================================================================


class TestConnectorDetail:
    """Test GET /api/streaming/connectors/{type}."""

    @pytest.mark.parametrize("connector_type", sorted(CONNECTOR_TYPES))
    def test_detail_valid_types(self, handler, http, connector_type):
        result = handler.handle(f"/api/streaming/connectors/{connector_type}", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["type"] == connector_type
        assert body["status"] == "disconnected"
        assert "health" in body
        assert "config" in body

    def test_detail_invalid_type(self, handler, http):
        result = handler.handle("/api/streaming/connectors/invalid_type", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body.get("error", "").lower() or "invalid" in json.dumps(body).lower()

    def test_detail_unknown_type(self, handler, http):
        result = handler.handle("/api/streaming/connectors/redis", {}, http)
        assert _status(result) == 400


# ===========================================================================
# GET /api/streaming/connectors/{type}/health — Health check
# ===========================================================================


class TestHealthCheck:
    """Test GET /api/streaming/connectors/{type}/health."""

    @pytest.mark.parametrize("connector_type", sorted(CONNECTOR_TYPES))
    def test_health_disconnected(self, handler, http, connector_type):
        result = handler.handle(f"/api/streaming/connectors/{connector_type}/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["healthy"] is False
        assert body["latency_ms"] == 0
        assert body["error"] == "Not connected"
        assert body["circuit_breaker_state"] == "closed"

    def test_health_connected(self, handler, http):
        # Manually set kafka to connected
        handler._statuses["kafka"] = "connected"
        result = handler.handle("/api/streaming/connectors/kafka/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["healthy"] is True
        assert body["latency_ms"] == 15
        assert body["error"] is None

    def test_health_invalid_type(self, handler, http):
        result = handler.handle("/api/streaming/connectors/invalid/health", {}, http)
        assert _status(result) == 400

    def test_health_includes_message_counts(self, handler, http):
        result = handler.handle("/api/streaming/connectors/kafka/health", {}, http)
        body = _body(result)
        assert "messages_processed" in body
        assert "messages_failed" in body
        assert body["messages_processed"] == 0
        assert body["messages_failed"] == 0

    def test_health_includes_last_message_at(self, handler, http):
        result = handler.handle("/api/streaming/connectors/kafka/health", {}, http)
        body = _body(result)
        assert "last_message_at" in body
        assert body["last_message_at"] is None


# ===========================================================================
# GET /api/streaming/connectors/{type}/config — Get config
# ===========================================================================


class TestGetConfig:
    """Test GET /api/streaming/connectors/{type}/config."""

    def test_get_kafka_config(self, handler, http):
        result = handler.handle("/api/streaming/connectors/kafka/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["bootstrap_servers"] == "localhost:9092"
        assert body["topics"] == ["aragora-events"]
        assert body["group_id"] == "aragora-consumer"
        assert body["security_protocol"] == "PLAINTEXT"
        assert body["enable_circuit_breaker"] is True
        assert body["enable_dlq"] is True

    def test_get_rabbitmq_config(self, handler, http):
        result = handler.handle("/api/streaming/connectors/rabbitmq/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["queue"] == "aragora-events"
        assert body["exchange_type"] == "direct"
        assert body["durable"] is True
        assert body["prefetch_count"] == 10

    def test_get_snssqs_config(self, handler, http):
        result = handler.handle("/api/streaming/connectors/snssqs/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["region"] == "us-east-1"
        assert body["max_messages"] == 10
        assert body["wait_time_seconds"] == 20
        assert body["enable_circuit_breaker"] is True

    def test_get_config_invalid_type(self, handler, http):
        result = handler.handle("/api/streaming/connectors/nats/config", {}, http)
        assert _status(result) == 400


# ===========================================================================
# PUT /api/streaming/connectors/{type}/config — Update config
# ===========================================================================


class TestUpdateConfig:
    """Test PUT /api/streaming/connectors/{type}/config."""

    def test_update_kafka_config(self, handler, http):
        update_body = {"bootstrap_servers": "broker1:9092,broker2:9092"}
        result = handler.handle_put(
            "/api/streaming/connectors/kafka/config",
            {},
            http,
            body=update_body,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "updated"
        assert body["config"]["bootstrap_servers"] == "broker1:9092,broker2:9092"
        # Original fields still present (merge)
        assert body["config"]["topics"] == ["aragora-events"]

    def test_update_rabbitmq_config(self, handler, http):
        update_body = {"url": "amqp://user:pass@broker:5672"}
        result = handler.handle_put(
            "/api/streaming/connectors/rabbitmq/config",
            {},
            http,
            body=update_body,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["url"] == "amqp://user:pass@broker:5672"
        assert body["config"]["queue"] == "aragora-events"  # Unchanged

    def test_update_snssqs_config(self, handler, http):
        update_body = {
            "region": "eu-west-1",
            "queue_url": "https://sqs.eu-west-1.amazonaws.com/123/queue",
        }
        result = handler.handle_put(
            "/api/streaming/connectors/snssqs/config",
            {},
            http,
            body=update_body,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["region"] == "eu-west-1"
        assert body["config"]["queue_url"] == "https://sqs.eu-west-1.amazonaws.com/123/queue"

    def test_update_config_invalid_type(self, handler, http):
        result = handler.handle_put(
            "/api/streaming/connectors/invalid/config",
            {},
            http,
            body={"key": "value"},
        )
        assert _status(result) == 400

    def test_update_config_empty_body(self, handler, http):
        result = handler.handle_put(
            "/api/streaming/connectors/kafka/config",
            {},
            http,
            body={},
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "updated"
        # Config should be unchanged
        assert body["config"]["bootstrap_servers"] == "localhost:9092"

    def test_update_config_none_body(self, handler, http):
        result = handler.handle_put(
            "/api/streaming/connectors/kafka/config",
            {},
            http,
            body=None,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "updated"

    def test_update_config_persists(self, handler, http):
        """Config updates are visible in subsequent GET."""
        handler.handle_put(
            "/api/streaming/connectors/kafka/config",
            {},
            http,
            body={"bootstrap_servers": "new-broker:9092"},
        )
        result = handler.handle("/api/streaming/connectors/kafka/config", {}, http)
        body = _body(result)
        assert body["bootstrap_servers"] == "new-broker:9092"

    def test_update_config_non_matching_path(self, handler, http):
        result = handler.handle_put("/api/other/path", {}, http, body={"key": "val"})
        assert result is None

    def test_update_config_missing_action(self, handler, http):
        """PUT to /api/streaming/connectors/kafka (no /config) should return None."""
        result = handler.handle_put(
            "/api/streaming/connectors/kafka", {}, http, body={"key": "val"}
        )
        assert result is None

    def test_update_config_wrong_action(self, handler, http):
        """PUT to /api/streaming/connectors/kafka/health should return None."""
        result = handler.handle_put("/api/streaming/connectors/kafka/health", {}, http, body={})
        assert result is None


# ===========================================================================
# POST /api/streaming/connectors/{type}/connect — Connect
# ===========================================================================


class TestConnect:
    """Test POST /api/streaming/connectors/{type}/connect."""

    @pytest.mark.parametrize("connector_type", sorted(CONNECTOR_TYPES))
    def test_connect_success(self, handler, http, connector_type):
        """Connecting succeeds when _try_connect returns True."""
        with patch.object(handler, "_try_connect", return_value=True):
            result = handler.handle_post(
                f"/api/streaming/connectors/{connector_type}/connect", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "connected"
        assert connector_type in body["message"]
        # Status should be updated
        assert handler._statuses[connector_type] == "connected"

    def test_connect_kafka_demo_mode(self, handler, http):
        """Kafka connect falls through ImportError to demo mode."""
        result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "connected"

    def test_connect_rabbitmq_demo_mode(self, handler, http):
        """RabbitMQ connect succeeds (module may exist or ImportError -> demo)."""
        result = handler.handle_post("/api/streaming/connectors/rabbitmq/connect", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "connected"

    def test_connect_already_connected(self, handler, http):
        handler._statuses["kafka"] = "connected"
        result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "already_connected"

    def test_connect_invalid_type(self, handler, http):
        result = handler.handle_post("/api/streaming/connectors/invalid/connect", {}, http)
        assert _status(result) == 400

    def test_connect_failure(self, handler, http):
        """Connection attempt that returns False should yield 500."""
        with patch.object(handler, "_try_connect", return_value=False):
            result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 500
        assert handler._statuses["kafka"] == "error"

    def test_connect_exception_sets_error_status(self, handler, http):
        """An exception during connect should set status to 'error'."""
        with patch.object(handler, "_try_connect", side_effect=ConnectionError("refused")):
            result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 500
        assert handler._statuses["kafka"] == "error"

    def test_connect_timeout_error(self, handler, http):
        with patch.object(handler, "_try_connect", side_effect=TimeoutError("timed out")):
            result = handler.handle_post("/api/streaming/connectors/rabbitmq/connect", {}, http)
        assert _status(result) == 500
        assert handler._statuses["rabbitmq"] == "error"

    def test_connect_import_error(self, handler, http):
        with patch.object(handler, "_try_connect", side_effect=ImportError("no module")):
            result = handler.handle_post("/api/streaming/connectors/snssqs/connect", {}, http)
        assert _status(result) == 500
        assert handler._statuses["snssqs"] == "error"

    def test_connect_runtime_error(self, handler, http):
        with patch.object(handler, "_try_connect", side_effect=RuntimeError("runtime fail")):
            result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 500

    def test_connect_value_error(self, handler, http):
        with patch.object(handler, "_try_connect", side_effect=ValueError("bad value")):
            result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 500

    def test_connect_os_error(self, handler, http):
        with patch.object(handler, "_try_connect", side_effect=OSError("os fail")):
            result = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(result) == 500


# ===========================================================================
# POST /api/streaming/connectors/{type}/disconnect — Disconnect
# ===========================================================================


class TestDisconnect:
    """Test POST /api/streaming/connectors/{type}/disconnect."""

    def test_disconnect_connected(self, handler, http):
        handler._statuses["kafka"] = "connected"
        result = handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "disconnected"
        assert handler._statuses["kafka"] == "disconnected"

    def test_disconnect_already_disconnected(self, handler, http):
        result = handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "already_disconnected"

    def test_disconnect_invalid_type(self, handler, http):
        result = handler.handle_post("/api/streaming/connectors/invalid/disconnect", {}, http)
        assert _status(result) == 400

    def test_disconnect_from_error_state(self, handler, http):
        """Disconnect from error state should succeed."""
        handler._statuses["kafka"] = "error"
        result = handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "disconnected"

    def test_disconnect_cleans_up_connector(self, handler, http):
        """Disconnect should remove stored connector."""
        mock_connector = MagicMock()
        handler._connectors["kafka"] = mock_connector
        handler._statuses["kafka"] = "connected"
        result = handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        assert _status(result) == 200
        assert "kafka" not in handler._connectors
        mock_connector.close.assert_called_once()

    def test_disconnect_connector_without_close(self, handler, http):
        """Disconnect a connector that has no close method."""
        handler._connectors["rabbitmq"] = "plain-string-connector"
        handler._statuses["rabbitmq"] = "connected"
        result = handler.handle_post("/api/streaming/connectors/rabbitmq/disconnect", {}, http)
        assert _status(result) == 200
        assert "rabbitmq" not in handler._connectors

    def test_disconnect_exception(self, handler, http):
        handler._statuses["kafka"] = "connected"
        with patch.object(handler, "_try_disconnect", side_effect=ConnectionError("err")):
            result = handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        assert _status(result) == 500

    def test_disconnect_timeout_error(self, handler, http):
        handler._statuses["rabbitmq"] = "connected"
        with patch.object(handler, "_try_disconnect", side_effect=TimeoutError("timeout")):
            result = handler.handle_post("/api/streaming/connectors/rabbitmq/disconnect", {}, http)
        assert _status(result) == 500

    @pytest.mark.parametrize("connector_type", sorted(CONNECTOR_TYPES))
    def test_disconnect_all_types(self, handler, http, connector_type):
        handler._statuses[connector_type] = "connected"
        result = handler.handle_post(
            f"/api/streaming/connectors/{connector_type}/disconnect", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "disconnected"


# ===========================================================================
# POST /api/streaming/connectors/{type}/test — Test connectivity
# ===========================================================================


class TestConnectivity:
    """Test POST /api/streaming/connectors/{type}/test."""

    def test_test_kafka_missing_servers(self, handler, http):
        """Kafka test with empty bootstrap_servers should fail."""
        handler._configs["kafka"]["bootstrap_servers"] = ""
        result = handler.handle_post("/api/streaming/connectors/kafka/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False
        assert "not configured" in body["message"].lower()

    def test_test_kafka_with_servers_import_error(self, handler, http):
        """Kafka test should succeed in demo mode on ImportError."""
        with patch.dict("sys.modules", {"kafka": None}):
            result = handler.handle_post("/api/streaming/connectors/kafka/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "demo" in body["message"].lower() or "not installed" in body["message"].lower()

    def test_test_rabbitmq_missing_url(self, handler, http):
        handler._configs["rabbitmq"]["url"] = ""
        result = handler.handle_post("/api/streaming/connectors/rabbitmq/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False
        assert "not configured" in body["message"].lower()

    def test_test_snssqs_missing_queue_url(self, handler, http):
        handler._configs["snssqs"]["queue_url"] = ""
        result = handler.handle_post("/api/streaming/connectors/snssqs/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False
        assert "not configured" in body["message"].lower()

    def test_test_invalid_type(self, handler, http):
        result = handler.handle_post("/api/streaming/connectors/invalid/test", {}, http)
        assert _status(result) == 400

    def test_test_connection_exception_returns_false(self, handler, http):
        """Exception during test should return success: False."""
        with patch.object(handler, "_test_connection", side_effect=ConnectionError("down")):
            result = handler.handle_post("/api/streaming/connectors/kafka/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False
        assert "failed" in body["message"].lower()

    def test_test_connection_timeout_error(self, handler, http):
        with patch.object(handler, "_test_connection", side_effect=TimeoutError("slow")):
            result = handler.handle_post("/api/streaming/connectors/rabbitmq/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False

    def test_test_connection_import_error(self, handler, http):
        with patch.object(handler, "_test_connection", side_effect=ImportError("missing")):
            result = handler.handle_post("/api/streaming/connectors/snssqs/test", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is False


# ===========================================================================
# POST routing — non-matching paths
# ===========================================================================


class TestPostRouting:
    """Test POST routing edge cases."""

    def test_post_non_matching_path(self, handler, http):
        result = handler.handle_post("/api/other/path", {}, http)
        assert result is None

    def test_post_unknown_action(self, handler, http):
        result = handler.handle_post("/api/streaming/connectors/kafka/unknown_action", {}, http)
        assert result is None

    def test_post_no_action(self, handler, http):
        """POST to /api/streaming/connectors/kafka (no action) returns None."""
        result = handler.handle_post("/api/streaming/connectors/kafka", {}, http)
        assert result is None

    def test_post_too_many_segments(self, handler, http):
        """POST with more than 2 path segments returns None."""
        result = handler.handle_post("/api/streaming/connectors/kafka/connect/extra", {}, http)
        assert result is None


# ===========================================================================
# GET routing — non-matching and edge cases
# ===========================================================================


class TestGetRouting:
    """Test GET routing edge cases."""

    def test_get_non_matching_path(self, handler, http):
        result = handler.handle("/api/other/path", {}, http)
        assert result is None

    def test_get_unknown_sub_path(self, handler, http):
        """Unknown two-segment sub-path returns None."""
        result = handler.handle("/api/streaming/connectors/kafka/unknown", {}, http)
        assert result is None

    def test_get_three_segment_path(self, handler, http):
        """Three segments under type returns None."""
        result = handler.handle("/api/streaming/connectors/kafka/health/extra", {}, http)
        assert result is None


# ===========================================================================
# Connector type validation
# ===========================================================================


class TestConnectorTypeValidation:
    """Test connector type validation across endpoints."""

    @pytest.mark.parametrize("endpoint", ["health", "config"])
    def test_get_endpoints_reject_invalid_type(self, handler, http, endpoint):
        result = handler.handle(f"/api/streaming/connectors/pulsar/{endpoint}", {}, http)
        assert _status(result) == 400

    @pytest.mark.parametrize("action", ["connect", "disconnect", "test"])
    def test_post_endpoints_reject_invalid_type(self, handler, http, action):
        result = handler.handle_post(f"/api/streaming/connectors/pulsar/{action}", {}, http)
        assert _status(result) == 400

    def test_validate_type_returns_none_for_valid(self):
        for t in CONNECTOR_TYPES:
            assert StreamingConnectorHandler._validate_type(t) is None

    def test_validate_type_returns_error_for_invalid(self):
        result = StreamingConnectorHandler._validate_type("nats")
        assert result is not None
        assert _status(result) == 400

    def test_error_message_lists_valid_types(self):
        result = StreamingConnectorHandler._validate_type("bogus")
        body = _body(result)
        error_text = json.dumps(body).lower()
        assert "kafka" in error_text
        assert "rabbitmq" in error_text
        assert "snssqs" in error_text


# ===========================================================================
# Internal connect implementations
# ===========================================================================


class TestConnectKafka:
    """Test _connect_kafka internal method."""

    def test_connect_kafka_import_error(self, handler):
        """ImportError in kafka connect falls back to demo mode (returns True)."""
        with patch.dict("sys.modules", {"aragora.connectors.enterprise.streaming.kafka": None}):
            # The import inside _connect_kafka will raise ImportError
            # but the handler catches it and returns True (demo mode)
            result = handler._connect_kafka(handler._configs["kafka"])
            assert result is True

    def test_connect_kafka_connection_error(self, handler):
        with patch.object(handler, "_connect_kafka", return_value=False):
            result = handler._connect_kafka(handler._configs["kafka"])
            assert result is False


class TestConnectRabbitMQ:
    """Test _connect_rabbitmq internal method."""

    def test_connect_rabbitmq_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.connectors.enterprise.streaming.rabbitmq": None}):
            result = handler._connect_rabbitmq(handler._configs["rabbitmq"])
            assert result is True


class TestConnectSNSSQS:
    """Test _connect_snssqs internal method."""

    def test_connect_snssqs_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.connectors.enterprise.streaming.snssqs": None}):
            result = handler._connect_snssqs(handler._configs["snssqs"])
            assert result is True


# ===========================================================================
# Internal test-connection implementations
# ===========================================================================


class TestTestKafka:
    """Test _test_kafka internal method."""

    def test_test_kafka_no_servers(self, handler):
        success, msg = handler._test_kafka({"bootstrap_servers": ""})
        assert success is False
        assert "not configured" in msg.lower()

    def test_test_kafka_import_error(self, handler):
        """When kafka package is not installed, returns demo mode success."""
        with patch.dict("sys.modules", {"kafka": None}):
            success, msg = handler._test_kafka({"bootstrap_servers": "broker:9092"})
            assert success is True
            assert "not installed" in msg.lower() or "demo" in msg.lower()


class TestTestRabbitMQ:
    """Test _test_rabbitmq internal method."""

    def test_test_rabbitmq_no_url(self, handler):
        success, msg = handler._test_rabbitmq({"url": ""})
        assert success is False
        assert "not configured" in msg.lower()

    def test_test_rabbitmq_import_error(self, handler):
        with patch.dict("sys.modules", {"pika": None}):
            success, msg = handler._test_rabbitmq({"url": "amqp://localhost"})
            assert success is True
            assert "not installed" in msg.lower() or "demo" in msg.lower()


class TestTestSNSSQS:
    """Test _test_snssqs internal method."""

    def test_test_snssqs_no_queue_url(self, handler):
        success, msg = handler._test_snssqs({"queue_url": ""})
        assert success is False
        assert "not configured" in msg.lower()

    def test_test_snssqs_import_error(self, handler):
        with patch.dict("sys.modules", {"boto3": None}):
            success, msg = handler._test_snssqs(
                {"queue_url": "https://sqs.us-east-1.amazonaws.com/123/q"}
            )
            assert success is True
            assert "not installed" in msg.lower() or "demo" in msg.lower()


# ===========================================================================
# State transition tests
# ===========================================================================


class TestStateTransitions:
    """Test connect/disconnect state machine behavior."""

    def test_connect_then_disconnect(self, handler, http):
        # Connect
        r1 = handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        assert _status(r1) == 200
        assert handler._statuses["kafka"] == "connected"

        # Disconnect
        r2 = handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        assert _status(r2) == 200
        assert handler._statuses["kafka"] == "disconnected"

    def test_connect_changes_health(self, handler, http):
        """After connect, health should report healthy."""
        handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        result = handler.handle("/api/streaming/connectors/kafka/health", {}, http)
        body = _body(result)
        assert body["healthy"] is True

    def test_disconnect_changes_health(self, handler, http):
        """After connect then disconnect, health should report unhealthy."""
        handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)
        handler.handle_post("/api/streaming/connectors/kafka/disconnect", {}, http)
        result = handler.handle("/api/streaming/connectors/kafka/health", {}, http)
        body = _body(result)
        assert body["healthy"] is False

    def test_error_state_shows_in_list(self, handler, http):
        handler._statuses["kafka"] = "error"
        result = handler.handle("/api/streaming/connectors", {}, http)
        body = _body(result)
        kafka_entry = next(c for c in body if c["type"] == "kafka")
        assert kafka_entry["status"] == "error"

    def test_connecting_state_intermediate(self, handler, http):
        """During connect, status transitions through 'connecting'."""
        states_seen = []

        original_try_connect = handler._try_connect

        def spy_try_connect(connector_type):
            states_seen.append(handler._statuses[connector_type])
            return original_try_connect(connector_type)

        with patch.object(handler, "_try_connect", side_effect=spy_try_connect):
            handler.handle_post("/api/streaming/connectors/kafka/connect", {}, http)

        assert "connecting" in states_seen


# ===========================================================================
# Default config structure tests
# ===========================================================================


class TestDefaultConfigs:
    """Test default configurations have expected structure."""

    def test_kafka_config_keys(self, handler):
        config = handler._configs["kafka"]
        expected_keys = {
            "bootstrap_servers",
            "topics",
            "group_id",
            "security_protocol",
            "sasl_mechanism",
            "sasl_username",
            "sasl_password",
            "ssl_cafile",
            "ssl_certfile",
            "ssl_keyfile",
            "auto_offset_reset",
            "enable_auto_commit",
            "auto_commit_interval_ms",
            "max_poll_records",
            "session_timeout_ms",
            "heartbeat_interval_ms",
            "schema_registry_url",
            "batch_size",
            "poll_timeout_seconds",
            "enable_circuit_breaker",
            "enable_dlq",
            "enable_graceful_shutdown",
        }
        assert expected_keys.issubset(set(config.keys()))

    def test_rabbitmq_config_keys(self, handler):
        config = handler._configs["rabbitmq"]
        expected_keys = {
            "url",
            "queue",
            "exchange",
            "exchange_type",
            "routing_key",
            "durable",
            "auto_delete",
            "exclusive",
            "prefetch_count",
            "dead_letter_exchange",
            "dead_letter_routing_key",
            "message_ttl",
            "ssl",
            "ssl_cafile",
            "ssl_certfile",
            "ssl_keyfile",
            "batch_size",
            "auto_ack",
            "requeue_on_error",
            "enable_circuit_breaker",
            "enable_dlq",
            "enable_graceful_shutdown",
        }
        assert expected_keys.issubset(set(config.keys()))

    def test_snssqs_config_keys(self, handler):
        config = handler._configs["snssqs"]
        expected_keys = {
            "region",
            "queue_url",
            "topic_arn",
            "max_messages",
            "wait_time_seconds",
            "visibility_timeout_seconds",
            "dead_letter_queue_url",
            "enable_circuit_breaker",
            "enable_idempotency",
        }
        assert expected_keys.issubset(set(config.keys()))

    def test_all_connector_types_have_configs(self, handler):
        for ct in CONNECTOR_TYPES:
            assert ct in handler._configs
            assert isinstance(handler._configs[ct], dict)
            assert len(handler._configs[ct]) > 0

    def test_all_connector_types_have_statuses(self, handler):
        for ct in CONNECTOR_TYPES:
            assert ct in handler._statuses
            assert handler._statuses[ct] == "disconnected"


# ===========================================================================
# Constructor tests
# ===========================================================================


class TestInit:
    """Test handler initialization."""

    def test_default_context(self):
        h = StreamingConnectorHandler()
        assert h._configs is not None
        assert h._statuses is not None
        assert h._connectors == {}

    def test_custom_context(self):
        ctx = {"key": "value"}
        h = StreamingConnectorHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_initial_statuses(self):
        h = StreamingConnectorHandler()
        for ct in CONNECTOR_TYPES:
            assert h._statuses[ct] == "disconnected"

    def test_connector_types_constant(self):
        assert CONNECTOR_TYPES == {"kafka", "rabbitmq", "snssqs"}


# ===========================================================================
# _try_connect dispatcher tests
# ===========================================================================


class TestTryConnect:
    """Test _try_connect method dispatching."""

    def test_try_connect_kafka(self, handler):
        with patch.object(handler, "_connect_kafka", return_value=True) as m:
            result = handler._try_connect("kafka")
            assert result is True
            m.assert_called_once_with(handler._configs["kafka"])

    def test_try_connect_rabbitmq(self, handler):
        with patch.object(handler, "_connect_rabbitmq", return_value=True) as m:
            result = handler._try_connect("rabbitmq")
            assert result is True
            m.assert_called_once_with(handler._configs["rabbitmq"])

    def test_try_connect_snssqs(self, handler):
        with patch.object(handler, "_connect_snssqs", return_value=True) as m:
            result = handler._try_connect("snssqs")
            assert result is True
            m.assert_called_once_with(handler._configs["snssqs"])

    def test_try_connect_unknown_type(self, handler):
        result = handler._try_connect("unknown")
        assert result is False


# ===========================================================================
# _test_connection dispatcher tests
# ===========================================================================


class TestTestConnection:
    """Test _test_connection method dispatching."""

    def test_test_connection_kafka(self, handler):
        with patch.object(handler, "_test_kafka", return_value=(True, "ok")) as m:
            success, msg = handler._test_connection("kafka")
            assert success is True
            assert msg == "ok"
            m.assert_called_once()

    def test_test_connection_rabbitmq(self, handler):
        with patch.object(handler, "_test_rabbitmq", return_value=(True, "ok")) as m:
            success, msg = handler._test_connection("rabbitmq")
            assert success is True
            m.assert_called_once()

    def test_test_connection_snssqs(self, handler):
        with patch.object(handler, "_test_snssqs", return_value=(True, "ok")) as m:
            success, msg = handler._test_connection("snssqs")
            assert success is True
            m.assert_called_once()

    def test_test_connection_unknown_type(self, handler):
        success, msg = handler._test_connection("unknown")
        assert success is False
        assert "unknown" in msg.lower()

    def test_test_connection_exception_caught(self, handler):
        with patch.object(handler, "_test_kafka", side_effect=ConnectionError("fail")):
            success, msg = handler._test_connection("kafka")
            assert success is False
            assert "failed" in msg.lower()

    def test_test_connection_value_error_caught(self, handler):
        with patch.object(handler, "_test_rabbitmq", side_effect=ValueError("bad")):
            success, msg = handler._test_connection("rabbitmq")
            assert success is False

    def test_test_connection_runtime_error_caught(self, handler):
        with patch.object(handler, "_test_snssqs", side_effect=RuntimeError("crash")):
            success, msg = handler._test_connection("snssqs")
            assert success is False


# ===========================================================================
# _try_disconnect tests
# ===========================================================================


class TestTryDisconnect:
    """Test _try_disconnect internal method."""

    def test_disconnect_with_close(self, handler):
        mock_conn = MagicMock()
        handler._connectors["kafka"] = mock_conn
        handler._try_disconnect("kafka")
        mock_conn.close.assert_called_once()
        assert "kafka" not in handler._connectors

    def test_disconnect_without_close(self, handler):
        # Object without close method
        handler._connectors["rabbitmq"] = 42
        handler._try_disconnect("rabbitmq")
        assert "rabbitmq" not in handler._connectors

    def test_disconnect_no_connector(self, handler):
        # Should not raise
        handler._try_disconnect("snssqs")
        assert "snssqs" not in handler._connectors


# ===========================================================================
# Health status helper tests
# ===========================================================================


class TestGetHealthStatus:
    """Test _get_health_status internal method."""

    def test_disconnected_health(self, handler):
        health = handler._get_health_status("kafka")
        assert health["healthy"] is False
        assert health["latency_ms"] == 0
        assert health["error"] == "Not connected"
        assert health["circuit_breaker_state"] == "closed"
        assert health["messages_processed"] == 0
        assert health["messages_failed"] == 0
        assert health["last_message_at"] is None

    def test_connected_health(self, handler):
        handler._statuses["kafka"] = "connected"
        health = handler._get_health_status("kafka")
        assert health["healthy"] is True
        assert health["latency_ms"] == 15
        assert health["error"] is None

    def test_error_state_health(self, handler):
        handler._statuses["kafka"] = "error"
        health = handler._get_health_status("kafka")
        assert health["healthy"] is False
        assert health["error"] == "Not connected"

    def test_unknown_connector_health(self, handler):
        """Unknown connector type returns disconnected health."""
        health = handler._get_health_status("unknown")
        assert health["healthy"] is False
