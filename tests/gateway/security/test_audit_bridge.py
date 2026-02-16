"""Tests for audit bridge."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.gateway.security.audit_bridge import (
    AuditBridge,
    AuditEvent,
    AuditEventType,
)
from aragora.gateway.external_agents.base import (
    AgentCapability,
    IsolationLevel,
    ExternalAgentTask,
    ExternalAgentResult,
)


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_event_type_values(self):
        """Test event type values."""
        assert AuditEventType.EXECUTION_START.value == "external_agent.execution.start"
        assert AuditEventType.EXECUTION_COMPLETE.value == "external_agent.execution.complete"
        assert AuditEventType.POLICY_DECISION.value == "external_agent.policy.decision"
        assert AuditEventType.CREDENTIAL_ACCESS.value == "external_agent.credential.access"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_default_event(self):
        """Test event with default values."""
        event = AuditEvent()
        assert event.event_id is not None
        assert event.event_type == AuditEventType.EXECUTION_START
        assert event.timestamp is not None
        assert event.severity == "info"

    def test_custom_event(self):
        """Test event with custom values."""
        event = AuditEvent(
            event_type=AuditEventType.EXECUTION_FAILED,
            tenant_id="tenant-123",
            user_id="user-456",
            agent_name="openclaw",
            task_id="task-789",
            severity="error",
            details={"error": "timeout"},
        )
        assert event.event_type == AuditEventType.EXECUTION_FAILED
        assert event.tenant_id == "tenant-123"
        assert event.user_id == "user-456"
        assert event.agent_name == "openclaw"
        assert event.severity == "error"

    def test_to_dict(self):
        """Test event serialization."""
        event = AuditEvent(
            event_type=AuditEventType.EXECUTION_START,
            tenant_id="tenant-123",
            agent_name="openclaw",
        )
        data = event.to_dict()

        assert data["event_type"] == "external_agent.execution.start"
        assert data["tenant_id"] == "tenant-123"
        assert data["agent_name"] == "openclaw"
        assert "timestamp" in data


class TestAuditBridge:
    """Tests for AuditBridge."""

    def test_default_bridge(self):
        """Test bridge with default configuration."""
        bridge = AuditBridge()
        assert bridge._enable_signing is True
        assert bridge._signing_key is not None

    def test_bridge_with_custom_key(self):
        """Test bridge with custom signing key."""
        key = b"custom-signing-key-32-bytes-long"
        bridge = AuditBridge(signing_key=key)
        assert bridge._signing_key == key

    def test_bridge_disable_signing(self):
        """Test bridge with signing disabled."""
        bridge = AuditBridge(enable_signing=False)
        assert bridge._enable_signing is False

    def test_sign_event(self):
        """Test event signing."""
        bridge = AuditBridge()
        event = AuditEvent(
            event_type=AuditEventType.EXECUTION_START,
            tenant_id="tenant-123",
        )
        signature = bridge._sign_event(event)

        assert signature is not None
        assert len(signature) == 64  # SHA-256 hex digest

    def test_sign_event_deterministic(self):
        """Test that signing is deterministic."""
        bridge = AuditBridge(signing_key=b"test-key")
        event = AuditEvent(
            event_id="fixed-id",
            event_type=AuditEventType.EXECUTION_START,
            tenant_id="tenant-123",
        )

        sig1 = bridge._sign_event(event)
        sig2 = bridge._sign_event(event)

        assert sig1 == sig2

    def test_sign_event_different_for_different_events(self):
        """Test that different events have different signatures."""
        bridge = AuditBridge(signing_key=b"test-key")
        event1 = AuditEvent(
            event_type=AuditEventType.EXECUTION_START,
            tenant_id="tenant-123",
        )
        event2 = AuditEvent(
            event_type=AuditEventType.EXECUTION_START,
            tenant_id="tenant-456",
        )

        sig1 = bridge._sign_event(event1)
        sig2 = bridge._sign_event(event2)

        assert sig1 != sig2

    @pytest.mark.asyncio
    async def test_log_execution_start(self):
        """Test logging execution start."""
        bridge = AuditBridge()
        task = ExternalAgentTask(
            prompt="test prompt",
            required_capabilities=[AgentCapability.WEB_SEARCH],
            tenant_id="tenant-123",
            user_id="user-456",
        )

        event_id = await bridge.log_execution_start(
            adapter_name="openclaw",
            task=task,
        )

        assert event_id is not None
        assert len(bridge._event_buffer) == 1
        assert bridge._event_buffer[0].event_type == AuditEventType.EXECUTION_START
        assert bridge._event_buffer[0].agent_name == "openclaw"

    @pytest.mark.asyncio
    async def test_log_execution_complete_success(self):
        """Test logging successful execution completion."""
        bridge = AuditBridge()
        result = ExternalAgentResult(
            task_id="task-123",
            success=True,
            output="result",
            agent_name="openclaw",
            agent_version="1.0.0",
            execution_time_ms=150.0,
            capabilities_used=[AgentCapability.WEB_SEARCH],
            was_sandboxed=True,
            isolation_level=IsolationLevel.CONTAINER,
        )

        event_id = await bridge.log_execution_complete(
            result=result,
            tenant_id="tenant-123",
        )

        assert event_id is not None
        assert len(bridge._event_buffer) == 1
        event = bridge._event_buffer[0]
        assert event.event_type == AuditEventType.EXECUTION_COMPLETE
        assert event.severity == "info"

    @pytest.mark.asyncio
    async def test_log_execution_complete_failure(self):
        """Test logging failed execution completion."""
        bridge = AuditBridge()
        result = ExternalAgentResult(
            task_id="task-123",
            success=False,
            error="Timeout exceeded",
            agent_name="openclaw",
            capabilities_used=[],
            was_sandboxed=True,
            isolation_level=IsolationLevel.CONTAINER,
        )

        event_id = await bridge.log_execution_complete(
            result=result,
            tenant_id="tenant-123",
        )

        event = bridge._event_buffer[0]
        assert event.event_type == AuditEventType.EXECUTION_FAILED
        assert event.severity == "warning"

    @pytest.mark.asyncio
    async def test_log_credential_access(self):
        """Test logging credential access."""
        bridge = AuditBridge()

        event_id = await bridge.log_credential_access(
            agent_name="openclaw",
            tenant_id="tenant-123",
            credentials_accessed=["OPENAI_API_KEY", "AWS_SECRET_KEY"],
        )

        assert event_id is not None
        event = bridge._event_buffer[0]
        assert event.event_type == AuditEventType.CREDENTIAL_ACCESS
        assert event.details["credential_count"] == 2

    @pytest.mark.asyncio
    async def test_log_output_redaction(self):
        """Test logging output redaction."""
        bridge = AuditBridge()

        event_id = await bridge.log_output_redaction(
            agent_name="openclaw",
            task_id="task-123",
            tenant_id="tenant-123",
            redaction_count=5,
            redacted_types={"api_key": 3, "ssn": 2},
        )

        event = bridge._event_buffer[0]
        assert event.event_type == AuditEventType.OUTPUT_REDACTED
        assert event.details["redaction_count"] == 5

    @pytest.mark.asyncio
    async def test_log_capability_usage_allowed(self):
        """Test logging allowed capability usage."""
        bridge = AuditBridge()

        event_id = await bridge.log_capability_usage(
            agent_name="openclaw",
            task_id="task-123",
            tenant_id="tenant-123",
            capability="web_search",
            allowed=True,
        )

        event = bridge._event_buffer[0]
        assert event.event_type == AuditEventType.CAPABILITY_USED
        assert event.severity == "info"

    @pytest.mark.asyncio
    async def test_log_capability_usage_blocked(self):
        """Test logging blocked capability usage."""
        bridge = AuditBridge()

        event_id = await bridge.log_capability_usage(
            agent_name="openclaw",
            task_id="task-123",
            tenant_id="tenant-123",
            capability="shell_access",
            allowed=False,
        )

        event = bridge._event_buffer[0]
        assert event.event_type == AuditEventType.CAPABILITY_BLOCKED
        assert event.severity == "warning"

    @pytest.mark.asyncio
    async def test_buffer_flush(self):
        """Test buffer flush when full."""
        mock_storage = AsyncMock()
        bridge = AuditBridge(storage_backend=mock_storage)
        bridge._buffer_size = 3  # Small buffer for testing

        task = ExternalAgentTask(prompt="test")

        # Log events to fill buffer
        for _ in range(3):
            await bridge.log_execution_start(
                adapter_name="openclaw",
                task=task,
            )

        # Buffer should have been flushed
        mock_storage.store_events_batch.assert_called()

    @pytest.mark.asyncio
    async def test_close_flushes_buffer(self):
        """Test that close flushes remaining events."""
        bridge = AuditBridge()
        task = ExternalAgentTask(prompt="test")

        await bridge.log_execution_start(
            adapter_name="openclaw",
            task=task,
        )

        assert len(bridge._event_buffer) == 1

        await bridge.close()

        assert len(bridge._event_buffer) == 0

    @pytest.mark.asyncio
    async def test_storage_backend_integration(self):
        """Test integration with storage backend."""
        mock_storage = AsyncMock()
        bridge = AuditBridge(storage_backend=mock_storage)

        task = ExternalAgentTask(prompt="test")
        await bridge.log_execution_start(
            adapter_name="openclaw",
            task=task,
        )

        mock_storage.store_event.assert_called_once()
        call_args = mock_storage.store_event.call_args[0][0]
        assert call_args["event_type"] == "external_agent.execution.start"

    @pytest.mark.asyncio
    async def test_storage_error_handling(self):
        """Test that storage errors are handled gracefully."""
        mock_storage = AsyncMock()
        mock_storage.store_event.side_effect = RuntimeError("Storage error")
        bridge = AuditBridge(storage_backend=mock_storage)

        task = ExternalAgentTask(prompt="test")

        # Should not raise exception
        event_id = await bridge.log_execution_start(
            adapter_name="openclaw",
            task=task,
        )

        assert event_id is not None  # Event still created
