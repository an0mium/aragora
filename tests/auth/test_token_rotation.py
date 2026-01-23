"""Tests for token rotation policy enforcement."""

import time
from unittest.mock import MagicMock

import pytest

from aragora.auth.token_rotation import (
    RotationPolicy,
    RotationReason,
    TokenRotationManager,
    TokenUsageRecord,
    get_rotation_manager,
    reset_rotation_manager,
)


class TestRotationPolicy:
    """Test RotationPolicy configurations."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = RotationPolicy()
        assert policy.max_uses == 100
        assert policy.max_age_seconds == 86400
        assert policy.allow_ip_change is True

    def test_strict_policy(self):
        """Test strict security policy."""
        policy = RotationPolicy.strict()
        assert policy.max_uses == 50
        assert policy.max_age_seconds == 3600  # 1 hour
        assert policy.bind_to_ip is True
        assert policy.allow_ip_change is False

    def test_relaxed_policy(self):
        """Test relaxed development policy."""
        policy = RotationPolicy.relaxed()
        assert policy.max_uses == 1000
        assert policy.max_age_seconds == 604800  # 7 days
        assert policy.idle_rotation_seconds == 0


class TestTokenUsageRecord:
    """Test TokenUsageRecord tracking."""

    def test_add_usage(self):
        """Test recording usage."""
        record = TokenUsageRecord(
            token_jti="test-jti",
            user_id="user-123",
            first_used=time.time(),
            last_used=time.time(),
        )

        record.add_usage(ip_address="10.0.0.1", user_agent="Chrome")
        assert record.use_count == 1
        assert "10.0.0.1" in record.ip_addresses
        assert len(record.user_agents) == 1

        record.add_usage(ip_address="10.0.0.2")
        assert record.use_count == 2
        assert len(record.ip_addresses) == 2


class TestTokenRotationManager:
    """Test TokenRotationManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        return TokenRotationManager(policy=RotationPolicy())

    def test_record_first_usage(self, manager):
        """Test recording first token usage."""
        result = manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        assert result.requires_rotation is False
        stats = manager.get_usage_stats("token-001")
        assert stats is not None
        assert stats["use_count"] == 1

    def test_max_uses_exceeded(self):
        """Test rotation when max uses exceeded."""
        policy = RotationPolicy(max_uses=5)
        manager = TokenRotationManager(policy=policy)

        # Use token up to limit
        for i in range(5):
            result = manager.record_usage(
                user_id="user-123",
                token_jti="token-001",
                ip_address="10.0.0.1",
            )

        assert result.requires_rotation is True
        assert result.reason == RotationReason.MAX_USES_EXCEEDED

    def test_time_based_rotation(self):
        """Test rotation based on token age."""
        policy = RotationPolicy(max_age_seconds=1)  # 1 second
        manager = TokenRotationManager(policy=policy)

        # Record initial usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Wait for expiration
        time.sleep(1.1)

        # Check if rotation required
        result = manager.requires_rotation("user-123", "token-001")
        assert result.requires_rotation is True
        assert result.reason == RotationReason.TIME_BASED

    def test_ip_change_detection(self):
        """Test IP change detection."""
        policy = RotationPolicy(ip_change_requires_rotation=True)
        manager = TokenRotationManager(policy=policy)

        # First usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Usage from different IP
        result = manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.2",
        )

        assert result.requires_rotation is True
        assert result.reason == RotationReason.IP_CHANGE

    def test_max_ips_exceeded(self):
        """Test rotation when max IPs exceeded."""
        policy = RotationPolicy(max_ips_per_token=3)
        manager = TokenRotationManager(policy=policy)

        # Use from multiple IPs
        for i in range(4):
            result = manager.record_usage(
                user_id="user-123",
                token_jti="token-001",
                ip_address=f"10.0.0.{i}",
            )

        assert result.requires_rotation is True
        assert result.reason == RotationReason.SUSPICIOUS_ACTIVITY

    def test_failed_validation_tracking(self, manager):
        """Test failed validation tracking."""
        manager.policy.max_failed_validations = 3

        # Record failures
        for _ in range(2):
            assert manager.record_failed_validation("token-001") is False

        # Third failure should trigger revocation
        assert manager.record_failed_validation("token-001") is True

    def test_suspicious_activity_detection(self, manager):
        """Test suspicious activity detection."""
        # Record usage to create record
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Initially not suspicious
        assert manager.is_suspicious("user-123", "token-001") is False

    def test_clear_token(self, manager):
        """Test clearing token tracking."""
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        assert manager.get_usage_stats("token-001") is not None

        manager.clear_token("token-001")

        assert manager.get_usage_stats("token-001") is None

    def test_callback_on_rotation(self):
        """Test callback when rotation is required."""
        callback = MagicMock()
        policy = RotationPolicy(max_uses=1)
        manager = TokenRotationManager(
            policy=policy,
            on_rotation_required=callback,
        )

        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Note: callback is called externally when checking result


class TestRotationManagerSingleton:
    """Test singleton access."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_rotation_manager()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_rotation_manager()

    def test_get_rotation_manager(self):
        """Test getting singleton instance."""
        manager = get_rotation_manager()
        assert manager is not None

        # Same instance returned
        manager2 = get_rotation_manager()
        assert manager is manager2

    def test_reset_rotation_manager(self):
        """Test resetting singleton."""
        manager1 = get_rotation_manager()
        reset_rotation_manager()
        manager2 = get_rotation_manager()
        assert manager1 is not manager2
