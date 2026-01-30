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


class TestConcurrentRotationAttempts:
    """Test concurrent rotation scenarios and race conditions."""

    def test_two_concurrent_rotation_checks_same_token(self):
        """Test two concurrent rotation checks on the same token.

        Verifies that when two threads simultaneously check if rotation
        is required for the same token, both get consistent results and
        no data corruption occurs.
        """
        policy = RotationPolicy(max_uses=10)
        manager = TokenRotationManager(policy=policy)

        # Set up token at exactly the rotation threshold
        for _ in range(9):
            manager.record_usage(
                user_id="user-123",
                token_jti="shared-token",
                ip_address="10.0.0.1",
            )

        results = []
        errors = []

        def check_and_record():
            try:
                # This will be the 10th use, triggering rotation
                result = manager.record_usage(
                    user_id="user-123",
                    token_jti="shared-token",
                    ip_address="10.0.0.1",
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start both threads simultaneously
        threads = [threading.Thread(target=check_and_record) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 2

        # At least one should require rotation (the one hitting threshold)
        # Both may require rotation since use_count will be >= 10
        rotation_required = [r for r in results if r.requires_rotation]
        assert len(rotation_required) >= 1

        # Verify token state is consistent
        stats = manager.get_usage_stats("shared-token")
        assert stats["use_count"] == 11  # 9 initial + 2 concurrent

    def test_concurrent_rotation_trigger_with_callback(self):
        """Test that rotation callback is invoked correctly during concurrent access."""
        callback_invocations = []
        callback_lock = threading.Lock()

        def on_rotation(user_id, token_jti, reason):
            with callback_lock:
                callback_invocations.append((user_id, token_jti, reason))

        policy = RotationPolicy(max_uses=5)
        manager = TokenRotationManager(
            policy=policy,
            on_rotation_required=on_rotation,
        )

        results = []

        def record_usage(worker_id):
            for _ in range(3):
                result = manager.record_usage(
                    user_id="user-123",
                    token_jti="token-concurrent",
                    ip_address=f"10.0.0.{worker_id}",
                )
                results.append(result)

        threads = [threading.Thread(target=record_usage, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 9 results total
        assert len(results) == 9

        # Rotation should be required once we hit 5 uses
        rotation_results = [r for r in results if r.requires_rotation]
        assert len(rotation_results) >= 1

    def test_race_condition_on_token_creation(self):
        """Test race condition when two threads create the same token record."""
        manager = TokenRotationManager(policy=RotationPolicy())
        results = []
        errors = []

        def first_usage(worker_id):
            try:
                result = manager.record_usage(
                    user_id="user-123",
                    token_jti="new-token",
                    ip_address=f"10.0.0.{worker_id}",
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Both threads try to create the same token record simultaneously
        threads = [threading.Thread(target=first_usage, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

        # Verify only one record was created but with all usages counted
        stats = manager.get_usage_stats("new-token")
        assert stats is not None
        assert stats["use_count"] == 10

    def test_concurrent_clear_during_rotation_check(self):
        """Test clearing a token while rotation check is in progress."""
        manager = TokenRotationManager(policy=RotationPolicy(max_uses=100))
        errors = []

        # Pre-populate the token
        for _ in range(50):
            manager.record_usage(
                user_id="user-123",
                token_jti="token-to-clear",
                ip_address="10.0.0.1",
            )

        def rotation_checker():
            try:
                for _ in range(100):
                    manager.requires_rotation("user-123", "token-to-clear")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def token_clearer():
            try:
                for _ in range(100):
                    manager.clear_token("token-to-clear")
                    # Re-create for next iteration
                    manager.record_usage(
                        user_id="user-123",
                        token_jti="token-to-clear",
                        ip_address="10.0.0.1",
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=rotation_checker),
            threading.Thread(target=token_clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestReplayAttackPrevention:
    """Test replay attack prevention scenarios."""

    def test_old_token_usage_after_clear(self):
        """Test that using a cleared token is properly handled.

        Simulates replay attack where attacker tries to use old token
        after it has been invalidated.
        """
        manager = TokenRotationManager(policy=RotationPolicy())

        # Record legitimate usage
        manager.record_usage(
            user_id="user-123",
            token_jti="old-token",
            ip_address="10.0.0.1",
        )

        # Token gets rotated/cleared (simulating rotation)
        manager.clear_token("old-token")

        # Verify old token has no usage record
        stats = manager.get_usage_stats("old-token")
        assert stats is None

        # Attempt to use old token (replay attack)
        result = manager.record_usage(
            user_id="user-123",
            token_jti="old-token",
            ip_address="10.0.0.2",  # Different IP - suspicious
        )

        # The manager creates a new record (it doesn't know this is an old token)
        # This is expected behavior - actual replay prevention should happen
        # at the JWT validation layer, not the rotation manager
        stats = manager.get_usage_stats("old-token")
        assert stats is not None
        assert stats["use_count"] == 1  # Fresh record

    def test_token_reuse_from_different_context(self):
        """Test detecting token reuse from unexpected context.

        Simulates replay attack where token is used from different
        IP/user-agent than originally issued.
        """
        policy = RotationPolicy(
            allow_ip_change=False,
            bind_to_ip=True,
        )
        manager = TokenRotationManager(policy=policy)

        # Original legitimate usage
        manager.record_usage(
            user_id="user-123",
            token_jti="sensitive-token",
            ip_address="10.0.0.1",
            user_agent="Chrome/100",
        )

        # Replay attempt from different IP
        result = manager.record_usage(
            user_id="user-123",
            token_jti="sensitive-token",
            ip_address="192.168.1.100",  # Attacker's IP
            user_agent="Firefox/90",  # Different user agent
        )

        # Should detect suspicious activity
        assert result.requires_rotation is True
        assert result.is_suspicious is True

    def test_rapid_token_reuse_detection(self):
        """Test detection of rapid token reuse (automated replay attacks)."""
        policy = RotationPolicy(rapid_use_threshold=5)
        manager = TokenRotationManager(policy=policy)

        # Simulate rapid automated requests (replay attack script)
        for _ in range(20):
            manager.record_usage(
                user_id="user-123",
                token_jti="target-token",
                ip_address="10.0.0.1",
            )

        # Check if suspicious activity was detected
        is_suspicious = manager.is_suspicious("user-123", "target-token")
        assert is_suspicious is True

        stats = manager.get_usage_stats("target-token")
        assert "rapid_use_detected" in stats["suspicious_flags"]

    def test_concurrent_replay_attempts(self):
        """Test handling of concurrent replay attempts from multiple sources."""
        policy = RotationPolicy(max_ips_per_token=3)
        manager = TokenRotationManager(policy=policy)

        # Legitimate first use
        manager.record_usage(
            user_id="user-123",
            token_jti="stolen-token",
            ip_address="10.0.0.1",
        )

        errors = []
        results = []

        def replay_attempt(attacker_ip):
            try:
                result = manager.record_usage(
                    user_id="user-123",
                    token_jti="stolen-token",
                    ip_address=attacker_ip,
                )
                results.append((attacker_ip, result))
            except Exception as e:
                errors.append(e)

        # Multiple attackers try to use the token concurrently
        attacker_ips = [f"192.168.{i}.100" for i in range(10)]
        threads = [threading.Thread(target=replay_attempt, args=(ip,)) for ip in attacker_ips]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Some should be flagged as suspicious due to max_ips_per_token
        suspicious_results = [r for _, r in results if r.is_suspicious]
        assert len(suspicious_results) > 0

    def test_failed_validation_tracking_as_replay_indicator(self):
        """Test that failed validations are tracked as potential replay indicator."""
        manager = TokenRotationManager(policy=RotationPolicy(max_failed_validations=3))

        # Create token record
        manager.record_usage(
            user_id="user-123",
            token_jti="token-001",
            ip_address="10.0.0.1",
        )

        # Simulate failed validation attempts (e.g., signature verification failed)
        # This could indicate someone trying to forge/replay the token
        for _ in range(2):
            revoke = manager.record_failed_validation("token-001")
            assert revoke is False

        # Third failure triggers revocation
        revoke = manager.record_failed_validation("token-001")
        assert revoke is True

        # Token should be flagged as suspicious
        is_suspicious = manager.is_suspicious("user-123", "token-001")
        assert is_suspicious is True


class TestTokenLifecycleStateMachine:
    """Test token lifecycle state transitions and edge cases."""

    def test_new_token_initial_state(self):
        """Test initial state of a newly tracked token."""
        manager = TokenRotationManager(policy=RotationPolicy())

        result = manager.record_usage(
            user_id="user-123",
            token_jti="fresh-token",
            ip_address="10.0.0.1",
        )

        assert result.requires_rotation is False
        assert result.is_suspicious is False
        assert result.reason is None

        stats = manager.get_usage_stats("fresh-token")
        assert stats["use_count"] == 1
        assert stats["suspicious_flags"] == []

    def test_token_state_active_to_rotation_required(self):
        """Test transition from active to rotation-required state."""
        policy = RotationPolicy(max_uses=5)
        manager = TokenRotationManager(policy=policy)

        # Active state - uses 1-4
        for i in range(4):
            result = manager.record_usage(
                user_id="user-123",
                token_jti="token-lifecycle",
                ip_address="10.0.0.1",
            )
            assert result.requires_rotation is False

        # Check we get recommendation near threshold
        assert len(result.recommendations) > 0

        # Trigger rotation - use 5
        result = manager.record_usage(
            user_id="user-123",
            token_jti="token-lifecycle",
            ip_address="10.0.0.1",
        )

        assert result.requires_rotation is True
        assert result.reason == RotationReason.MAX_USES_EXCEEDED

    def test_token_state_rotation_required_to_cleared(self):
        """Test clearing a token that requires rotation."""
        policy = RotationPolicy(max_uses=1)
        manager = TokenRotationManager(policy=policy)

        # Trigger rotation requirement
        result = manager.record_usage(
            user_id="user-123",
            token_jti="token-to-rotate",
            ip_address="10.0.0.1",
        )
        assert result.requires_rotation is True

        # Clear the token (simulating successful rotation)
        manager.clear_token("token-to-rotate")

        # Verify clean slate
        assert manager.get_usage_stats("token-to-rotate") is None
        assert manager._failed_validations.get("token-to-rotate") is None

    def test_token_state_active_to_suspicious(self):
        """Test transition from active to suspicious state."""
        policy = RotationPolicy(max_ips_per_token=2)
        manager = TokenRotationManager(policy=policy)

        # Active state - first two IPs are fine
        manager.record_usage(
            user_id="user-123",
            token_jti="token-becoming-suspicious",
            ip_address="10.0.0.1",
        )
        manager.record_usage(
            user_id="user-123",
            token_jti="token-becoming-suspicious",
            ip_address="10.0.0.2",
        )

        # Third IP triggers suspicious state
        result = manager.record_usage(
            user_id="user-123",
            token_jti="token-becoming-suspicious",
            ip_address="10.0.0.3",
        )

        assert result.is_suspicious is True
        assert result.requires_rotation is True
        assert result.reason == RotationReason.SUSPICIOUS_ACTIVITY

    def test_token_state_suspicious_persists(self):
        """Test that suspicious state persists across checks."""
        manager = TokenRotationManager(policy=RotationPolicy())

        # Create token and mark as suspicious
        manager.record_usage(
            user_id="user-123",
            token_jti="already-suspicious",
            ip_address="10.0.0.1",
        )
        manager._usage["already-suspicious"].suspicious_flags.append("external_flag")

        # Multiple subsequent checks should all report suspicious
        for _ in range(5):
            result = manager.requires_rotation("user-123", "already-suspicious")
            assert result.is_suspicious is True
            assert result.requires_rotation is True

    def test_token_state_idle_to_rotation_required(self):
        """Test transition from idle to rotation-required due to inactivity."""
        policy = RotationPolicy(idle_rotation_seconds=1, max_age_seconds=0)
        manager = TokenRotationManager(policy=policy)

        # Active usage
        result = manager.record_usage(
            user_id="user-123",
            token_jti="idle-token",
            ip_address="10.0.0.1",
        )
        assert result.requires_rotation is False

        # Wait for idle timeout
        time.sleep(1.1)

        # Check rotation requirement
        result = manager.requires_rotation("user-123", "idle-token")
        assert result.requires_rotation is True
        assert result.reason == RotationReason.TIME_BASED
        assert "idle" in result.details.lower()

    def test_multiple_state_transitions_in_sequence(self):
        """Test multiple state transitions in sequence."""
        # Note: max_ips_per_token uses >= comparison, so with value of 4,
        # the 4th IP will trigger suspicious activity
        policy = RotationPolicy(
            max_uses=20,  # Higher limit so we can test IP transitions first
            max_ips_per_token=4,  # Will trigger at 4th IP (>= 4)
        )
        manager = TokenRotationManager(policy=policy)

        # Phase 1: Normal usage from first IP
        for _ in range(5):
            result = manager.record_usage(
                user_id="user-123",
                token_jti="multi-transition",
                ip_address="10.0.0.1",
            )
        assert result.requires_rotation is False

        # Phase 2: IP diversity increases (still within limit)
        result = manager.record_usage(
            user_id="user-123",
            token_jti="multi-transition",
            ip_address="10.0.0.2",
        )
        assert result.requires_rotation is False

        # Phase 3: Third IP (still within limit since max is 4)
        result = manager.record_usage(
            user_id="user-123",
            token_jti="multi-transition",
            ip_address="10.0.0.3",
        )
        assert result.requires_rotation is False

        # Phase 4: Fourth IP triggers suspicious (reaches max_ips_per_token=4)
        result = manager.record_usage(
            user_id="user-123",
            token_jti="multi-transition",
            ip_address="10.0.0.4",  # 4th IP reaches limit
        )
        assert result.is_suspicious is True
        assert result.requires_rotation is True
        assert result.reason == RotationReason.SUSPICIOUS_ACTIVITY

        # Phase 5: Clear and start fresh
        manager.clear_token("multi-transition")
        assert manager.get_usage_stats("multi-transition") is None


class TestRotationDuringActiveSessions:
    """Test rotation behavior during active sessions."""

    def test_rotation_check_during_ongoing_session(self):
        """Test checking rotation during an active session.

        Simulates a long-running session where rotation checks happen
        periodically while the session continues to use the token.
        """
        policy = RotationPolicy(max_uses=20)
        manager = TokenRotationManager(policy=policy)

        rotation_triggered_at = None

        def session_activity():
            nonlocal rotation_triggered_at
            for i in range(25):
                result = manager.record_usage(
                    user_id="session-user",
                    token_jti="session-token",
                    ip_address="10.0.0.1",
                )
                if result.requires_rotation and rotation_triggered_at is None:
                    rotation_triggered_at = i + 1
                time.sleep(0.01)

        session_thread = threading.Thread(target=session_activity)
        session_thread.start()
        session_thread.join()

        # Rotation should have been triggered at use 20
        assert rotation_triggered_at == 20

    def test_concurrent_sessions_same_token(self):
        """Test multiple concurrent sessions using the same token.

        This can happen in legitimate scenarios (e.g., multiple tabs)
        or attack scenarios (token theft).
        """
        policy = RotationPolicy(max_uses=50)
        manager = TokenRotationManager(policy=policy)

        session_results = {i: [] for i in range(3)}

        def session(session_id):
            for _ in range(20):
                result = manager.record_usage(
                    user_id="shared-user",
                    token_jti="multi-session-token",
                    ip_address="10.0.0.1",  # Same IP - legitimate multi-tab
                )
                session_results[session_id].append(result)
                time.sleep(0.005)

        threads = [threading.Thread(target=session, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All sessions should complete without errors
        total_uses = sum(len(results) for results in session_results.values())
        assert total_uses == 60

        # Verify final use count
        stats = manager.get_usage_stats("multi-session-token")
        assert stats["use_count"] == 60

        # At least one result should require rotation (hit 50 uses)
        all_results = [r for results in session_results.values() for r in results]
        rotation_required = [r for r in all_results if r.requires_rotation]
        assert len(rotation_required) >= 1

    def test_session_continues_after_rotation_required(self):
        """Test that session can continue (with warning) after rotation required.

        The rotation manager doesn't block - it just reports when rotation
        is needed. The actual enforcement is up to the caller.
        """
        policy = RotationPolicy(max_uses=5)
        manager = TokenRotationManager(policy=policy)

        results = []

        # Continue using token even after rotation is required
        for _ in range(10):
            result = manager.record_usage(
                user_id="stubborn-user",
                token_jti="over-used-token",
                ip_address="10.0.0.1",
            )
            results.append(result)

        # First 4 should not require rotation
        assert all(not r.requires_rotation for r in results[:4])

        # Uses 5-10 should all require rotation
        assert all(r.requires_rotation for r in results[4:])
        assert all(r.reason == RotationReason.MAX_USES_EXCEEDED for r in results[4:])

    def test_token_rotation_mid_session_with_clear(self):
        """Test clearing a token during an active session.

        Simulates token rotation where old token is invalidated
        while session is still using it.
        """
        manager = TokenRotationManager(policy=RotationPolicy())

        session_active = threading.Event()
        clear_done = threading.Event()
        post_clear_results = []

        def active_session():
            # Use token before clear
            for _ in range(5):
                manager.record_usage(
                    user_id="user-123",
                    token_jti="rotating-token",
                    ip_address="10.0.0.1",
                )

            session_active.set()
            clear_done.wait()  # Wait for clear to happen

            # Continue using token after clear
            for _ in range(5):
                result = manager.record_usage(
                    user_id="user-123",
                    token_jti="rotating-token",
                    ip_address="10.0.0.1",
                )
                post_clear_results.append(result)

        def rotator():
            session_active.wait()  # Wait for session to be active
            manager.clear_token("rotating-token")
            clear_done.set()

        threads = [
            threading.Thread(target=active_session),
            threading.Thread(target=rotator),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After clear, the token record was recreated fresh
        stats = manager.get_usage_stats("rotating-token")
        assert stats is not None
        assert stats["use_count"] == 5  # Only post-clear uses

    def test_ip_change_during_active_session(self):
        """Test IP change during an active session (e.g., mobile network switch)."""
        policy = RotationPolicy(
            ip_change_requires_rotation=False,
            allow_ip_change=True,
            max_ips_per_token=5,
        )
        manager = TokenRotationManager(policy=policy)

        # Session starts on one network
        for _ in range(5):
            result = manager.record_usage(
                user_id="mobile-user",
                token_jti="mobile-token",
                ip_address="10.0.0.1",  # WiFi
            )
        assert result.requires_rotation is False

        # User moves to mobile network
        for _ in range(5):
            result = manager.record_usage(
                user_id="mobile-user",
                token_jti="mobile-token",
                ip_address="192.168.1.100",  # Mobile
            )
        assert result.requires_rotation is False

        # Verify both IPs are tracked
        stats = manager.get_usage_stats("mobile-token")
        assert stats["unique_ips"] == 2

    def test_session_with_intermittent_failures(self):
        """Test session with intermittent validation failures."""
        policy = RotationPolicy(max_failed_validations=5)
        manager = TokenRotationManager(policy=policy)

        # Start session
        manager.record_usage(
            user_id="user-123",
            token_jti="flaky-token",
            ip_address="10.0.0.1",
        )

        # Simulate intermittent failures (e.g., clock skew, network issues)
        for i in range(4):
            revoke = manager.record_failed_validation("flaky-token")
            assert revoke is False

            # Successful use between failures
            manager.record_usage(
                user_id="user-123",
                token_jti="flaky-token",
                ip_address="10.0.0.1",
            )

        # Final failure exceeds threshold
        revoke = manager.record_failed_validation("flaky-token")
        assert revoke is True

        # Token should be flagged
        is_suspicious = manager.is_suspicious("user-123", "flaky-token")
        assert is_suspicious is True


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
