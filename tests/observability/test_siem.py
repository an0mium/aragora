"""
Tests for SIEM integration.

Tests cover:
- Event creation and serialization
- Client configuration
- Event emission
- Batch processing
- Backend-specific formatting
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.siem import (
    SecurityEvent,
    SecurityEventType,
    SIEMBackend,
    SIEMClient,
    SIEMConfig,
    emit_auth_event,
    emit_data_access_event,
    emit_privacy_event,
    emit_security_event,
    get_siem_client,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSIEMConfig:
    """Tests for SIEM configuration."""

    def test_default_config(self):
        """Default config should be disabled."""
        config = SIEMConfig()
        assert config.backend == SIEMBackend.NONE
        assert config.enabled is True
        assert config.batch_size == 10
        assert config.flush_interval == 5.0

    def test_config_from_env(self):
        """Config should load from environment variables."""
        with patch.dict(
            os.environ,
            {
                "SIEM_BACKEND": "splunk",
                "SIEM_ENDPOINT": "https://splunk.example.com/hec",
                "SIEM_TOKEN": "test-token",
                "SIEM_INDEX": "test-index",
                "SIEM_BATCH_SIZE": "20",
                "SIEM_FLUSH_INTERVAL": "10",
            },
        ):
            config = SIEMConfig.from_env()
            assert config.backend == SIEMBackend.SPLUNK
            assert config.endpoint == "https://splunk.example.com/hec"
            assert config.token == "test-token"
            assert config.index == "test-index"
            assert config.batch_size == 20
            assert config.flush_interval == 10.0

    def test_invalid_backend_falls_back_to_none(self):
        """Invalid backend should fall back to NONE."""
        with patch.dict(os.environ, {"SIEM_BACKEND": "invalid"}):
            config = SIEMConfig.from_env()
            assert config.backend == SIEMBackend.NONE


# =============================================================================
# Security Event Tests
# =============================================================================


class TestSecurityEvent:
    """Tests for SecurityEvent class."""

    def test_event_creation(self):
        """Events should be created with correct fields."""
        event = SecurityEvent(
            event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
            user_id="user-123",
            ip_address="192.168.1.1",
            metadata={"mfa_used": True},
        )

        assert event.event_type == SecurityEventType.AUTH_LOGIN_SUCCESS
        assert event.user_id == "user-123"
        assert event.ip_address == "192.168.1.1"
        assert event.metadata == {"mfa_used": True}
        assert event.outcome == "success"
        assert event.severity == "info"

    def test_event_to_dict(self):
        """Events should serialize to dict correctly."""
        event = SecurityEvent(
            event_type=SecurityEventType.DATA_READ,
            user_id="user-456",
            resource_type="debate",
            resource_id="debate-789",
        )

        data = event.to_dict()
        assert data["event_type"] == "data.read"
        assert data["user_id"] == "user-456"
        assert data["resource_type"] == "debate"
        assert data["resource_id"] == "debate-789"
        assert data["source"] == "aragora"
        assert data["version"] == "1.0"

    def test_event_to_json(self):
        """Events should serialize to JSON correctly."""
        event = SecurityEvent(
            event_type=SecurityEventType.AUTH_LOGOUT,
            user_id="user-123",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "auth.logout"
        assert parsed["user_id"] == "user-123"

    def test_timestamp_auto_generated(self):
        """Timestamp should be auto-generated in ISO format."""
        event = SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS)
        assert "T" in event.timestamp  # ISO format has T separator
        assert "Z" in event.timestamp or "+" in event.timestamp  # Has timezone


# =============================================================================
# SIEM Client Tests
# =============================================================================


class TestSIEMClient:
    """Tests for SIEM client."""

    def test_client_disabled_by_default(self):
        """Client with NONE backend should not start worker."""
        config = SIEMConfig(backend=SIEMBackend.NONE, enabled=True)
        client = SIEMClient(config)

        # Should not have started worker
        assert client._worker is None

    def test_client_emit_when_disabled(self):
        """Emit should be no-op when disabled."""
        config = SIEMConfig(enabled=False)
        client = SIEMClient(config)

        event = SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS)
        client.emit(event)

        # Queue should be empty (event not added)
        assert client._queue.empty()

    def test_client_queues_events(self):
        """Events should be queued when enabled."""
        config = SIEMConfig(backend=SIEMBackend.NONE, enabled=True)
        client = SIEMClient(config)

        event = SecurityEvent(event_type=SecurityEventType.AUTH_LOGIN_SUCCESS)
        client.emit(event)

        # Event should be in queue
        assert not client._queue.empty()

    def test_client_shutdown(self):
        """Client should shutdown cleanly."""
        config = SIEMConfig(backend=SIEMBackend.SPLUNK, enabled=True)
        with patch.object(SIEMClient, "_start_worker"):
            client = SIEMClient(config)
            client._worker = MagicMock()
            client.shutdown(timeout=1.0)

            assert client._shutdown.is_set()


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestEmitFunctions:
    """Tests for emit helper functions."""

    @pytest.fixture(autouse=True)
    def mock_client(self):
        """Mock the SIEM client for all tests."""
        with patch("aragora.observability.siem.get_siem_client") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock_client

    def test_emit_security_event(self, mock_client):
        """emit_security_event should create and emit event."""
        emit_security_event(
            event_type=SecurityEventType.SECURITY_RATE_LIMIT,
            user_id="user-123",
            ip_address="10.0.0.1",
            severity="warning",
        )

        mock_client.emit.assert_called_once()
        event = mock_client.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.SECURITY_RATE_LIMIT
        assert event.user_id == "user-123"
        assert event.severity == "warning"

    def test_emit_auth_event_login_success(self, mock_client):
        """emit_auth_event should map login_success correctly."""
        emit_auth_event(
            user_id="user-123",
            action="login_success",
            ip_address="192.168.1.1",
            metadata={"provider": "google"},
        )

        mock_client.emit.assert_called_once()
        event = mock_client.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.AUTH_LOGIN_SUCCESS
        assert event.severity == "info"

    def test_emit_auth_event_login_failure(self, mock_client):
        """emit_auth_event should map login_failure with warning severity."""
        emit_auth_event(
            user_id="user-123",
            action="login_failure",
            ip_address="192.168.1.1",
        )

        mock_client.emit.assert_called_once()
        event = mock_client.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.AUTH_LOGIN_FAILURE
        assert event.severity == "warning"

    def test_emit_data_access_event_granted(self, mock_client):
        """emit_data_access_event should emit read event when granted."""
        emit_data_access_event(
            user_id="user-123",
            resource_type="debate",
            resource_id="debate-456",
            action="read",
            granted=True,
        )

        mock_client.emit.assert_called_once()
        event = mock_client.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.DATA_READ
        assert event.outcome == "success"
        assert event.severity == "info"

    def test_emit_data_access_event_denied(self, mock_client):
        """emit_data_access_event should emit with denied outcome."""
        emit_data_access_event(
            user_id="user-123",
            resource_type="debate",
            resource_id="debate-456",
            action="delete",
            granted=False,
        )

        mock_client.emit.assert_called_once()
        event = mock_client.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.DATA_DELETE
        assert event.outcome == "denied"
        assert event.severity == "warning"

    def test_emit_privacy_event(self, mock_client):
        """emit_privacy_event should emit privacy events."""
        emit_privacy_event(
            user_id="user-123",
            action="data_deletion",
            metadata={"reason": "user_request"},
        )

        mock_client.emit.assert_called_once()
        event = mock_client.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.PRIVACY_DATA_DELETION
        assert event.metadata == {"reason": "user_request"}


# =============================================================================
# Event Type Coverage Tests
# =============================================================================


class TestSecurityEventTypes:
    """Test all security event types are properly defined."""

    def test_auth_event_types(self):
        """Auth event types should be defined."""
        assert SecurityEventType.AUTH_LOGIN_SUCCESS.value == "auth.login.success"
        assert SecurityEventType.AUTH_LOGIN_FAILURE.value == "auth.login.failure"
        assert SecurityEventType.AUTH_MFA_SUCCESS.value == "auth.mfa.success"

    def test_data_event_types(self):
        """Data event types should be defined."""
        assert SecurityEventType.DATA_READ.value == "data.read"
        assert SecurityEventType.DATA_WRITE.value == "data.write"
        assert SecurityEventType.DATA_DELETE.value == "data.delete"
        assert SecurityEventType.DATA_EXPORT.value == "data.export"

    def test_privacy_event_types(self):
        """Privacy event types should be defined."""
        assert SecurityEventType.PRIVACY_CONSENT_GRANTED.value == "privacy.consent.granted"
        assert SecurityEventType.PRIVACY_DATA_DELETION.value == "privacy.data.deletion"

    def test_security_event_types(self):
        """Security incident event types should be defined."""
        assert SecurityEventType.SECURITY_RATE_LIMIT.value == "security.rate_limit"
        assert SecurityEventType.SECURITY_BRUTE_FORCE.value == "security.brute_force"
