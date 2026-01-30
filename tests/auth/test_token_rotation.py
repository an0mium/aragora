"""Tests for token rotation policy enforcement."""

import logging
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.auth.token_rotation import (
    RotationCheckResult,
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

    def test_add_usage_with_location(self):
        """Test recording usage with location tracking."""
        record = TokenUsageRecord(
            token_jti="test-jti",
            user_id="user-123",
            first_used=time.time(),
            last_used=time.time(),
        )

        record.add_usage(ip_address="10.0.0.1", location="US-NY")
        assert "US-NY" in record.locations
        assert record.use_count == 1

        record.add_usage(ip_address="10.0.0.1", location="US-CA")
        assert "US-CA" in record.locations
        assert len(record.locations) == 2

    def test_user_agent_hashing(self):
        """Test that user agents are hashed for privacy."""
        record = TokenUsageRecord(
            token_jti="test-jti",
            user_id="user-123",
            first_used=time.time(),
            last_used=time.time(),
        )

        record.add_usage(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

        # User agent should be hashed (16 char hex string)
        assert len(record.user_agents) == 1
        ua_hash = list(record.user_agents)[0]
        assert len(ua_hash) == 16
        # Verify it's hexadecimal
        int(ua_hash, 16)


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
        for _ in range(5):
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


class TestRotationScheduling:
    """Test token rotation scheduling based on policies."""

    def test_rotation_scheduled_by_use_count_approaching_limit(self):
        """Test that recommendations are given when approaching use limit."""
        policy = RotationPolicy(max_uses=10)
        manager = TokenRotationManager(policy=policy)

        # Use token 8 times (80% of limit)
        for _ in range(8):
            result = manager.record_usage(
                user_id="user-123",
                token_jti="token-001",
                ip_address="10.0.0.1",
            )

        # Should not require rotation yet but recommend it
        assert result.requires_rotation is False
        assert len(result.recommendations) > 0
        assert any("consider rotating" in r.lower() for r in result.recommendations)

    def test_rotation_scheduled_by_age_approaching_limit(self):
        """Test that recommendations are given when approaching age limit."""
        policy = RotationPolicy(max_age_seconds=10)  # 10 seconds
        manager = TokenRotationManager(policy=policy)

        # Record initial usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Wait for 80% of age limit
        time.sleep(8.1)

        result = manager.requires_rotation("user-123", "token-001")

        # Should not require rotation yet but recommend it
        assert result.requires_rotation is False
        assert len(result.recommendations) > 0
        assert any("approaching max" in r.lower() for r in result.recommendations)


class TestGracePeriodHandling:
    """Test idle/grace period handling for token rotation."""

    def test_idle_rotation_trigger(self):
        """Test that idle tokens trigger rotation after grace period."""
        policy = RotationPolicy(idle_rotation_seconds=1)  # 1 second grace
        manager = TokenRotationManager(policy=policy)

        # Record initial usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Wait for idle period
        time.sleep(1.1)

        result = manager.requires_rotation("user-123", "token-001")
        assert result.requires_rotation is True
        assert result.reason == RotationReason.TIME_BASED
        assert "idle" in result.details.lower()

    def test_activity_resets_idle_timer(self):
        """Test that activity resets the idle timer."""
        policy = RotationPolicy(idle_rotation_seconds=2, max_age_seconds=0)
        manager = TokenRotationManager(policy=policy)

        # Record initial usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Wait 1 second
        time.sleep(1)

        # Record another usage (resets idle timer)
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Wait another 1 second (only 1s since last use)
        time.sleep(1)

        result = manager.requires_rotation("user-123", "token-001")
        # Should not require rotation since we used it 1s ago (threshold is 2s)
        assert result.requires_rotation is False

    def test_disabled_idle_rotation(self):
        """Test that idle rotation can be disabled."""
        policy = RotationPolicy(idle_rotation_seconds=0, max_age_seconds=0)
        manager = TokenRotationManager(policy=policy)

        # Record initial usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Wait some time
        time.sleep(0.5)

        result = manager.requires_rotation("user-123", "token-001")
        assert result.requires_rotation is False


class TestOldTokenInvalidation:
    """Test old token invalidation after rotation."""

    def test_clear_token_removes_all_tracking_data(self):
        """Test that clear_token removes all tracking data for a token."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Record usage and failed validation
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )
        manager.record_failed_validation("token-001")

        # Verify data exists
        assert manager.get_usage_stats("token-001") is not None
        assert manager._failed_validations.get("token-001") is not None

        # Clear token
        manager.clear_token("token-001")

        # Verify all data removed
        assert manager.get_usage_stats("token-001") is None
        assert manager._failed_validations.get("token-001") is None
        assert "token-001" not in manager._recent_uses

    def test_clear_nonexistent_token(self):
        """Test that clearing a nonexistent token does not raise."""
        manager = TokenRotationManager(policy=RotationPolicy())
        # Should not raise
        manager.clear_token("nonexistent-token")


class TestConcurrentRotationHandling:
    """Test concurrent access to token rotation manager."""

    def test_thread_safe_record_usage(self):
        """Test that record_usage is thread-safe under concurrent access."""
        manager = TokenRotationManager(policy=RotationPolicy(max_uses=1000))
        errors = []
        results = []

        def worker(worker_id: int):
            try:
                for i in range(50):
                    result = manager.record_usage(
                        user_id=f"user-{worker_id}",
                        token_jti=f"token-{worker_id}",
                        ip_address=f"10.0.{worker_id}.{i % 256}",
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 500

        # Verify each user's token was tracked
        for i in range(10):
            stats = manager.get_usage_stats(f"token-{i}")
            assert stats is not None
            assert stats["use_count"] == 50

    def test_thread_safe_failed_validation(self):
        """Test that failed validation tracking is thread-safe."""
        manager = TokenRotationManager(policy=RotationPolicy(max_failed_validations=100))
        errors = []

        def worker(worker_id: int):
            try:
                for _ in range(20):
                    manager.record_failed_validation("shared-token")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Total should be 100 (5 threads * 20 attempts)
        assert manager._failed_validations.get("shared-token") == 100

    def test_concurrent_clear_and_record(self):
        """Test concurrent clear and record operations."""
        manager = TokenRotationManager(policy=RotationPolicy())
        errors = []

        def record_worker():
            try:
                for _ in range(50):
                    manager.record_usage(
                        user_id="user-123",
                        token_jti="token-001",
                        ip_address="10.0.0.1",
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def clear_worker():
            try:
                for _ in range(50):
                    manager.clear_token("token-001")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_worker),
            threading.Thread(target=clear_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestErrorHandling:
    """Test error handling in token rotation."""

    def test_requires_rotation_unknown_token(self):
        """Test requires_rotation for unknown token."""
        manager = TokenRotationManager(policy=RotationPolicy())

        result = manager.requires_rotation("user-123", "unknown-token")

        assert result.requires_rotation is False
        assert "no usage record" in result.details.lower()

    def test_get_usage_stats_unknown_token(self):
        """Test get_usage_stats for unknown token."""
        manager = TokenRotationManager(policy=RotationPolicy())

        stats = manager.get_usage_stats("unknown-token")

        assert stats is None

    def test_is_suspicious_unknown_token(self):
        """Test is_suspicious for unknown token."""
        manager = TokenRotationManager(policy=RotationPolicy())

        result = manager.is_suspicious("user-123", "unknown-token")

        assert result is False

    def test_suspicious_callback_on_excessive_failures(self):
        """Test that suspicious activity callback is invoked on excessive failures."""
        suspicious_callback = MagicMock()
        manager = TokenRotationManager(
            policy=RotationPolicy(max_failed_validations=3),
            on_suspicious_activity=suspicious_callback,
        )

        # Record usage first to create the token record
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Trigger excessive failures
        for _ in range(3):
            manager.record_failed_validation("token-001")

        suspicious_callback.assert_called_once()
        call_args = suspicious_callback.call_args[0]
        assert call_args[0] == "user-123"
        assert call_args[1] == "token-001"
        assert "excessive_failed_validations" in call_args[2]


class TestMetricsAndLogging:
    """Test metrics and logging for rotation events."""

    def test_cleanup_logs_removed_entries(self, caplog):
        """Test that cleanup logs the number of removed entries."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Directly add a stale record
        stale_time = time.time() - (86400 * 8)  # 8 days ago
        manager._usage["stale-token"] = TokenUsageRecord(
            token_jti="stale-token",
            user_id="user-stale",
            first_used=stale_time,
            last_used=stale_time,
        )

        # Force cleanup by setting last cleanup far in the past
        manager._last_cleanup = time.time() - 400

        with caplog.at_level(logging.DEBUG):
            # Trigger cleanup via record_usage
            manager.record_usage(
                user_id="user-123",
                token_jti="token-001",
                ip_address="10.0.0.1",
            )

        # Check that cleanup removed the stale record
        assert "stale-token" not in manager._usage

    def test_usage_stats_contains_all_fields(self):
        """Test that usage stats contain all expected fields."""
        manager = TokenRotationManager(policy=RotationPolicy())

        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
            user_agent="Chrome/100",
        )

        stats = manager.get_usage_stats("token-001")

        assert stats is not None
        assert "token_jti" in stats
        assert "user_id" in stats
        assert "first_used" in stats
        assert "last_used" in stats
        assert "use_count" in stats
        assert "unique_ips" in stats
        assert "unique_user_agents" in stats
        assert "suspicious_flags" in stats
        assert "age_seconds" in stats

        # Token JTI should be truncated for privacy
        assert stats["token_jti"] == "token-00..."

    def test_rotation_result_to_dict(self):
        """Test RotationCheckResult serialization."""
        result = RotationCheckResult(
            requires_rotation=True,
            reason=RotationReason.MAX_USES_EXCEEDED,
            details="Token used 100 times",
            is_suspicious=False,
            recommendations=["Consider using shorter-lived tokens"],
        )

        result_dict = result.to_dict()

        assert result_dict["requires_rotation"] is True
        assert result_dict["reason"] == "max_uses_exceeded"
        assert result_dict["details"] == "Token used 100 times"
        assert result_dict["is_suspicious"] is False
        assert len(result_dict["recommendations"]) == 1

    def test_rotation_result_to_dict_no_reason(self):
        """Test RotationCheckResult serialization without reason."""
        result = RotationCheckResult(
            requires_rotation=False,
            reason=None,
            details="",
        )

        result_dict = result.to_dict()

        assert result_dict["requires_rotation"] is False
        assert result_dict["reason"] is None


class TestIPSecurityPolicies:
    """Test IP-based security policies."""

    def test_ip_change_not_allowed(self):
        """Test that IP change triggers rotation when not allowed."""
        policy = RotationPolicy(allow_ip_change=False)
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
        assert result.reason == RotationReason.SUSPICIOUS_ACTIVITY
        assert result.is_suspicious is True

    def test_excessive_ip_diversity_detection(self):
        """Test detection of excessive IP diversity as suspicious."""
        policy = RotationPolicy(max_ips_per_token=3)
        manager = TokenRotationManager(policy=policy)

        # Use from multiple IPs
        for i in range(4):
            manager.record_usage(
                user_id="user-123",
                token_jti="token-001",
                ip_address=f"10.0.0.{i}",
            )

        # Check if suspicious
        is_suspicious = manager.is_suspicious("user-123", "token-001")
        assert is_suspicious is True

    def test_rapid_use_detection(self):
        """Test detection of rapid token use as suspicious."""
        policy = RotationPolicy(rapid_use_threshold=5)
        manager = TokenRotationManager(policy=policy)

        # Rapid fire usage in same second
        for _ in range(10):
            manager.record_usage(
                user_id="user-123",
                token_jti="token-001",
                ip_address="10.0.0.1",
            )

        # Check if suspicious
        is_suspicious = manager.is_suspicious("user-123", "token-001")
        assert is_suspicious is True


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_rotation_manager()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_rotation_manager()

    def test_strict_policy_from_env(self):
        """Test strict policy from environment variable."""
        with patch.dict(os.environ, {"ARAGORA_TOKEN_ROTATION_POLICY": "strict"}):
            manager = get_rotation_manager()
            assert manager.policy.max_uses == 50
            assert manager.policy.max_age_seconds == 3600

    def test_relaxed_policy_from_env(self):
        """Test relaxed policy from environment variable."""
        reset_rotation_manager()
        with patch.dict(os.environ, {"ARAGORA_TOKEN_ROTATION_POLICY": "relaxed"}):
            manager = get_rotation_manager()
            assert manager.policy.max_uses == 1000
            assert manager.policy.max_age_seconds == 604800

    def test_standard_policy_default(self):
        """Test standard policy as default."""
        with patch.dict(os.environ, {"ARAGORA_TOKEN_ROTATION_POLICY": "standard"}):
            manager = get_rotation_manager()
            assert manager.policy.max_uses == 100
            assert manager.policy.max_age_seconds == 86400


class TestSuspiciousActivityFlags:
    """Test suspicious activity flag management."""

    def test_suspicious_flags_trigger_rotation(self):
        """Test that suspicious flags trigger rotation."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Record usage to create record
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Manually add suspicious flag
        manager._usage["token-001"].suspicious_flags.append("manual_flag")

        result = manager.requires_rotation("user-123", "token-001")

        assert result.requires_rotation is True
        assert result.reason == RotationReason.SUSPICIOUS_ACTIVITY
        assert result.is_suspicious is True
        assert "manual_flag" in result.details

    def test_suspicious_flags_returned_in_stats(self):
        """Test that suspicious flags are included in stats."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Record usage
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Add suspicious flag
        manager._usage["token-001"].suspicious_flags.append("test_flag")

        stats = manager.get_usage_stats("token-001")

        assert "test_flag" in stats["suspicious_flags"]


class TestCleanupBehavior:
    """Test automatic cleanup of stale records."""

    def test_cleanup_removes_old_records(self):
        """Test that cleanup removes records older than 7 days."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Add old record (8 days ago)
        old_time = time.time() - (86400 * 8)
        manager._usage["old-token"] = TokenUsageRecord(
            token_jti="old-token",
            user_id="old-user",
            first_used=old_time,
            last_used=old_time,
        )
        manager._failed_validations["old-token"] = 5
        manager._recent_uses["old-token"] = [old_time]

        # Force cleanup
        manager._last_cleanup = time.time() - 400
        manager._maybe_cleanup()

        # Old record should be removed
        assert "old-token" not in manager._usage
        assert "old-token" not in manager._failed_validations
        assert "old-token" not in manager._recent_uses

    def test_cleanup_preserves_recent_records(self):
        """Test that cleanup preserves recent records."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Add recent record
        manager.record_usage(
            user_id="user-123",
            token_jti="recent-token",
            ip_address="10.0.0.1",
        )

        # Force cleanup
        manager._last_cleanup = time.time() - 400
        manager._maybe_cleanup()

        # Recent record should be preserved
        assert "recent-token" in manager._usage

    def test_cleanup_interval_respected(self):
        """Test that cleanup only runs at configured intervals."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Add old record
        old_time = time.time() - (86400 * 8)
        manager._usage["old-token"] = TokenUsageRecord(
            token_jti="old-token",
            user_id="old-user",
            first_used=old_time,
            last_used=old_time,
        )

        # Don't force cleanup (last cleanup was recent)
        manager._last_cleanup = time.time()
        manager._maybe_cleanup()

        # Old record should still exist
        assert "old-token" in manager._usage
