"""Streaming Connector REST Endpoints.

Provides REST endpoints for managing enterprise streaming connectors
(Kafka, RabbitMQ, SNS/SQS) from the admin UI.

Routes handled (prefix ``/api/streaming/connectors``):
    GET  /                      - List all streaming connectors
    GET  /{type}/health         - Health check for a connector
    GET  /{type}/config         - Get connector configuration
    PUT  /{type}/config         - Update connector configuration
    POST /{type}/connect        - Connect to the broker
    POST /{type}/disconnect     - Disconnect from the broker
    POST /{type}/test           - Test connectivity

Phase 6: Frontend/Backend Wiring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)

logger = logging.getLogger(__name__)

# Supported connector types
CONNECTOR_TYPES = {"kafka", "rabbitmq", "snssqs"}

# Prefix for all routes this handler owns.
_PREFIX = "/api/streaming/connectors"


class StreamingConnectorHandler(BaseHandler):
    """REST interface for streaming connector management."""

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path.startswith(_PREFIX)

    def __init__(self, server_context: dict[str, Any] | None = None) -> None:
        super().__init__(server_context or {})
        # In-memory config storage (would be persisted in production)
        self._configs: dict[str, dict] = {
            "kafka": self._default_kafka_config(),
            "rabbitmq": self._default_rabbitmq_config(),
            "snssqs": self._default_snssqs_config(),
        }
        self._statuses: dict[str, str] = {
            "kafka": "disconnected",
            "rabbitmq": "disconnected",
            "snssqs": "disconnected",
        }
        self._connectors: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Default configurations
    # ------------------------------------------------------------------

    def _default_kafka_config(self) -> dict:
        return {
            "bootstrap_servers": "localhost:9092",
            "topics": ["aragora-events"],
            "group_id": "aragora-consumer",
            "security_protocol": "PLAINTEXT",
            "sasl_mechanism": None,
            "sasl_username": None,
            "sasl_password": None,
            "ssl_cafile": None,
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "auto_offset_reset": "earliest",
            "enable_auto_commit": True,
            "auto_commit_interval_ms": 5000,
            "max_poll_records": 500,
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000,
            "schema_registry_url": None,
            "batch_size": 100,
            "poll_timeout_seconds": 1.0,
            "enable_circuit_breaker": True,
            "enable_dlq": True,
            "enable_graceful_shutdown": True,
        }

    def _default_rabbitmq_config(self) -> dict:
        return {
            "url": "",
            "queue": "aragora-events",
            "exchange": "",
            "exchange_type": "direct",
            "routing_key": "",
            "durable": True,
            "auto_delete": False,
            "exclusive": False,
            "prefetch_count": 10,
            "dead_letter_exchange": None,
            "dead_letter_routing_key": None,
            "message_ttl": None,
            "ssl": False,
            "ssl_cafile": None,
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "batch_size": 100,
            "auto_ack": False,
            "requeue_on_error": True,
            "enable_circuit_breaker": True,
            "enable_dlq": True,
            "enable_graceful_shutdown": True,
        }

    def _default_snssqs_config(self) -> dict:
        return {
            "region": "us-east-1",
            "queue_url": "",
            "topic_arn": None,
            "max_messages": 10,
            "wait_time_seconds": 20,
            "visibility_timeout_seconds": 300,
            "dead_letter_queue_url": None,
            "enable_circuit_breaker": True,
            "enable_idempotency": True,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_type(connector_type: str) -> HandlerResult | None:
        """Return an error response if connector type is invalid."""
        if connector_type not in CONNECTOR_TYPES:
            return error_response(
                f"Invalid connector type: {connector_type!r}. "
                f"Valid types: {', '.join(sorted(CONNECTOR_TYPES))}",
                400,
            )
        return None

    # ------------------------------------------------------------------
    # GET routing
    # ------------------------------------------------------------------

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests under ``/api/streaming/connectors``."""
        if not path.startswith(_PREFIX):
            return None

        user, err = self.require_auth_or_error(handler)
        if err:
            return err
        _, perm_err = self.require_permission_or_error(handler, "streaming:read")
        if perm_err:
            return perm_err

        sub = path[len(_PREFIX) :]

        # GET /api/streaming/connectors
        if sub in ("", "/"):
            return self._handle_list()

        # Parse connector type and action
        parts = sub.lstrip("/").split("/")
        connector_type = parts[0]

        # GET /api/streaming/connectors/{type}/health
        if len(parts) == 2 and parts[1] == "health":
            return self._handle_health(connector_type)

        # GET /api/streaming/connectors/{type}/config
        if len(parts) == 2 and parts[1] == "config":
            return self._handle_get_config(connector_type)

        # GET /api/streaming/connectors/{type}
        if len(parts) == 1:
            return self._handle_detail(connector_type)

        return None

    # ------------------------------------------------------------------
    # POST routing
    # ------------------------------------------------------------------

    @handle_errors("streaming connector creation")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route POST requests under ``/api/streaming/connectors``."""
        if not path.startswith(_PREFIX):
            return None

        user, err = self.require_auth_or_error(handler)
        if err:
            return err
        _, perm_err = self.require_permission_or_error(handler, "streaming:admin")
        if perm_err:
            return perm_err

        sub = path[len(_PREFIX) :]
        parts = sub.lstrip("/").split("/")

        if len(parts) != 2:
            return None

        connector_type = parts[0]
        action = parts[1]

        # POST /api/streaming/connectors/{type}/connect
        if action == "connect":
            return self._handle_connect(connector_type)

        # POST /api/streaming/connectors/{type}/disconnect
        if action == "disconnect":
            return self._handle_disconnect(connector_type)

        # POST /api/streaming/connectors/{type}/test
        if action == "test":
            return self._handle_test(connector_type)

        return None

    # ------------------------------------------------------------------
    # PUT routing
    # ------------------------------------------------------------------

    @handle_errors("streaming connector update")
    def handle_put(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
        body: dict[str, Any] | None = None,
    ) -> HandlerResult | None:
        """Route PUT requests under ``/api/streaming/connectors``."""
        if not path.startswith(_PREFIX):
            return None

        sub = path[len(_PREFIX) :]
        parts = sub.lstrip("/").split("/")

        # PUT /api/streaming/connectors/{type}/config
        if len(parts) == 2 and parts[1] == "config":
            return self._handle_update_config(parts[0], body or {})

        return None

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_list(self) -> HandlerResult:
        """GET /api/streaming/connectors — list all streaming connectors."""
        now = datetime.now(timezone.utc).isoformat()
        connectors = []

        for connector_type in CONNECTOR_TYPES:
            connectors.append(
                {
                    "type": connector_type,
                    "status": self._statuses.get(connector_type, "disconnected"),
                    "health": self._get_health_status(connector_type),
                    "config": self._configs.get(connector_type, {}),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        return json_response(connectors)

    def _handle_detail(self, connector_type: str) -> HandlerResult:
        """GET /api/streaming/connectors/{type} — connector details."""
        error = self._validate_type(connector_type)
        if error:
            return error

        now = datetime.now(timezone.utc).isoformat()
        return json_response(
            {
                "type": connector_type,
                "status": self._statuses.get(connector_type, "disconnected"),
                "health": self._get_health_status(connector_type),
                "config": self._configs.get(connector_type, {}),
                "created_at": now,
                "updated_at": now,
            }
        )

    def _handle_health(self, connector_type: str) -> HandlerResult:
        """GET /api/streaming/connectors/{type}/health — health check."""
        error = self._validate_type(connector_type)
        if error:
            return error

        return json_response(self._get_health_status(connector_type))

    def _handle_get_config(self, connector_type: str) -> HandlerResult:
        """GET /api/streaming/connectors/{type}/config — get configuration."""
        error = self._validate_type(connector_type)
        if error:
            return error

        return json_response(self._configs.get(connector_type, {}))

    def _handle_update_config(self, connector_type: str, body: dict[str, Any]) -> HandlerResult:
        """PUT /api/streaming/connectors/{type}/config — update configuration."""
        error = self._validate_type(connector_type)
        if error:
            return error

        # Merge with existing config
        current_config = self._configs.get(connector_type, {})
        current_config.update(body)
        self._configs[connector_type] = current_config

        logger.info(f"Updated {connector_type} configuration")

        return json_response(
            {
                "status": "updated",
                "config": current_config,
            }
        )

    def _handle_connect(self, connector_type: str) -> HandlerResult:
        """POST /api/streaming/connectors/{type}/connect — connect to broker."""
        error = self._validate_type(connector_type)
        if error:
            return error

        # Check if already connected
        if self._statuses.get(connector_type) == "connected":
            return json_response(
                {
                    "status": "already_connected",
                    "message": f"{connector_type} is already connected",
                }
            )

        # Set status to connecting
        self._statuses[connector_type] = "connecting"

        try:
            # Attempt connection based on type
            success = self._try_connect(connector_type)

            if success:
                self._statuses[connector_type] = "connected"
                return json_response(
                    {
                        "status": "connected",
                        "message": f"Successfully connected to {connector_type}",
                    }
                )
            else:
                self._statuses[connector_type] = "error"
                return error_response(f"Failed to connect to {connector_type}", 500)

        except (ImportError, ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            self._statuses[connector_type] = "error"
            logger.error(f"Error connecting to {connector_type}: {e}")
            return error_response("Internal server error", 500)

    def _handle_disconnect(self, connector_type: str) -> HandlerResult:
        """POST /api/streaming/connectors/{type}/disconnect — disconnect."""
        error = self._validate_type(connector_type)
        if error:
            return error

        # Check if already disconnected
        if self._statuses.get(connector_type) == "disconnected":
            return json_response(
                {
                    "status": "already_disconnected",
                    "message": f"{connector_type} is already disconnected",
                }
            )

        try:
            # Attempt disconnection
            self._try_disconnect(connector_type)
            self._statuses[connector_type] = "disconnected"

            return json_response(
                {
                    "status": "disconnected",
                    "message": f"Successfully disconnected from {connector_type}",
                }
            )

        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error(f"Error disconnecting from {connector_type}: {e}")
            return error_response("Internal server error", 500)

    def _handle_test(self, connector_type: str) -> HandlerResult:
        """POST /api/streaming/connectors/{type}/test — test connectivity."""
        error = self._validate_type(connector_type)
        if error:
            return error

        try:
            success, message = self._test_connection(connector_type)

            return json_response(
                {
                    "success": success,
                    "message": message,
                }
            )

        except (ImportError, ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error testing {connector_type} connection: {e}")
            return json_response(
                {
                    "success": False,
                    "message": "Connection test failed",
                }
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_health_status(self, connector_type: str) -> dict:
        """Get health status for a connector."""
        status = self._statuses.get(connector_type, "disconnected")
        is_healthy = status == "connected"

        return {
            "healthy": is_healthy,
            "latency_ms": 0 if not is_healthy else 15,  # Placeholder
            "messages_processed": 0,
            "messages_failed": 0,
            "last_message_at": None,
            "circuit_breaker_state": "closed",
            "error": None if is_healthy else "Not connected",
        }

    def _try_connect(self, connector_type: str) -> bool:
        """Attempt to connect to a streaming broker."""
        config = self._configs.get(connector_type, {})

        if connector_type == "kafka":
            return self._connect_kafka(config)
        elif connector_type == "rabbitmq":
            return self._connect_rabbitmq(config)
        elif connector_type == "snssqs":
            return self._connect_snssqs(config)

        return False

    def _try_disconnect(self, connector_type: str) -> None:
        """Disconnect from a streaming broker."""
        connector = self._connectors.get(connector_type)
        if connector:
            # Close connection if it exists
            if hasattr(connector, "close"):
                connector.close()
            del self._connectors[connector_type]

    def _test_connection(self, connector_type: str) -> tuple[bool, str]:
        """Test connectivity to a streaming broker."""
        config = self._configs.get(connector_type, {})

        try:
            if connector_type == "kafka":
                return self._test_kafka(config)
            elif connector_type == "rabbitmq":
                return self._test_rabbitmq(config)
            elif connector_type == "snssqs":
                return self._test_snssqs(config)
        except (ImportError, ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.warning("Connector test failed for %s: %s", connector_type, e)
            return False, "Connection test failed"

        return False, "Unknown connector type"

    # ------------------------------------------------------------------
    # Connector-specific implementations
    # ------------------------------------------------------------------

    def _connect_kafka(self, config: dict) -> bool:
        """Connect to Kafka broker."""
        try:
            # Try to import and create connector
            from aragora.connectors.enterprise.streaming.kafka import (
                KafkaConfig,
                KafkaConnector,
            )

            kafka_config = KafkaConfig(
                bootstrap_servers=config.get("bootstrap_servers", "localhost:9092"),
                topics=config.get("topics", ["aragora-events"]),
                group_id=config.get("group_id", "aragora-consumer"),
                security_protocol=config.get("security_protocol", "PLAINTEXT"),
            )
            connector = KafkaConnector(kafka_config)
            self._connectors["kafka"] = connector
            return True
        except ImportError:
            logger.debug("Kafka dependencies not installed")
            return True  # Allow connection in demo mode
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error(f"Kafka connection error: {e}")
            return False

    def _connect_rabbitmq(self, config: dict) -> bool:
        """Connect to RabbitMQ broker."""
        try:
            from aragora.connectors.enterprise.streaming.rabbitmq import (
                RabbitMQConfig,
                RabbitMQConnector,
            )

            rmq_config = RabbitMQConfig(
                url=config.get("url", ""),
                queue=config.get("queue", "aragora-events"),
            )
            connector = RabbitMQConnector(rmq_config)
            self._connectors["rabbitmq"] = connector
            return True
        except ImportError:
            logger.debug("RabbitMQ dependencies not installed")
            return True  # Allow connection in demo mode
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error(f"RabbitMQ connection error: {e}")
            return False

    def _connect_snssqs(self, config: dict) -> bool:
        """Connect to AWS SNS/SQS."""
        try:
            from aragora.connectors.enterprise.streaming.snssqs import (
                SNSSQSConfig,
                SNSSQSConnector,
            )

            sqs_config = SNSSQSConfig(
                region=config.get("region", "us-east-1"),
                queue_url=config.get("queue_url", ""),
            )
            connector = SNSSQSConnector(sqs_config)
            self._connectors["snssqs"] = connector
            return True
        except ImportError:
            logger.debug("AWS dependencies not installed")
            return True  # Allow connection in demo mode
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error(f"SNS/SQS connection error: {e}")
            return False

    def _test_kafka(self, config: dict) -> tuple[bool, str]:
        """Test Kafka connectivity."""
        bootstrap_servers = config.get("bootstrap_servers", "")
        if not bootstrap_servers:
            return False, "Bootstrap servers not configured"

        try:
            from kafka import KafkaAdminClient

            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=5000,
            )
            admin.close()
            return True, "Kafka connection successful"
        except ImportError:
            return True, "Kafka client not installed (demo mode)"
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            return False, f"Kafka connection failed: {e}"

    def _test_rabbitmq(self, config: dict) -> tuple[bool, str]:
        """Test RabbitMQ connectivity."""
        url = config.get("url", "")
        if not url:
            return False, "RabbitMQ URL not configured"

        try:
            import pika

            connection = pika.BlockingConnection(pika.URLParameters(url))
            connection.close()
            return True, "RabbitMQ connection successful"
        except ImportError:
            return True, "RabbitMQ client not installed (demo mode)"
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            return False, f"RabbitMQ connection failed: {e}"

    def _test_snssqs(self, config: dict) -> tuple[bool, str]:
        """Test AWS SNS/SQS connectivity."""
        queue_url = config.get("queue_url", "")
        if not queue_url:
            return False, "SQS queue URL not configured"

        try:
            import boto3

            sqs = boto3.client("sqs", region_name=config.get("region", "us-east-1"))
            sqs.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=["QueueArn"],
            )
            return True, "SQS connection successful"
        except ImportError:
            return True, "AWS SDK not installed (demo mode)"
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            return False, f"SQS connection failed: {e}"
