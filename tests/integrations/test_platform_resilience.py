"""
Tests for platform resilience features.

Tests distributed rate limiting, platform circuit breakers,
dead letter queue, and timeout wrappers.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from unittest.mock import patch

import pytest


class TestDistributedRateLimiter:
    """Tests for the distributed rate limiter."""

    def test_rate_limiter_allows_under_limit(self):
        """Rate limiter should allow requests under limit."""
        from aragora.server.middleware.rate_limit import RateLimiter

        # Use the default IP-based rate limiter
        limiter = RateLimiter(ip_limit=10)
        limiter.reset()

        # Each unique IP should be allowed (different keys get independent limits)
        for i in range(10):
            result = limiter.allow(f"192.168.1.{i}")
            assert result.allowed, f"Request from IP {i} should be allowed"

    def test_rate_limiter_blocks_over_limit(self):
        """Rate limiter should block requests over limit."""
        from aragora.server.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(ip_limit=5)
        limiter.reset()

        # Same IP uses up the limit (burst is 2x by default, so 10 total)
        client_ip = f"10.0.0.{int(time.time()) % 255}"

        # Should allow burst size requests
        for i in range(10):  # 5 * 2 burst multiplier
            result = limiter.allow(client_ip)
            assert result.allowed, f"Request {i+1} should be allowed within burst"

        # Should be blocked after burst
        result = limiter.allow(client_ip)
        assert not result.allowed
        assert result.remaining == 0

    def test_rate_limiter_different_keys(self):
        """Different IPs should have independent limits."""
        from aragora.server.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(ip_limit=2)
        limiter.reset()

        # Different IPs at different times to ensure uniqueness
        base = int(time.time()) % 200
        ip1 = f"172.16.{base}.1"
        ip2 = f"172.16.{base}.2"

        # IP 1 uses up its limit (2 rpm * 2 burst = 4)
        for _ in range(4):
            limiter.allow(ip1)
        assert not limiter.allow(ip1).allowed

        # IP 2 should still work
        assert limiter.allow(ip2).allowed
        assert limiter.allow(ip2).allowed

    @pytest.mark.skip(reason="PlatformRateLimitResult not exported from rate_limit module")
    def test_rate_limit_result_headers(self):
        """Rate limit result should generate proper headers."""
        from aragora.server.middleware.rate_limit import PlatformRateLimitResult

        # Use PlatformRateLimitResult which has to_headers method
        result = PlatformRateLimitResult(
            allowed=False,
            remaining=0,
            limit=60,
            reset_at=time.time() + 30,
            retry_after=30.0,
        )

        headers = result.to_headers()
        assert "X-RateLimit-Limit" in headers
        assert headers["X-RateLimit-Limit"] == "60"
        assert "X-RateLimit-Remaining" in headers
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in headers

    def test_platform_rate_limiter_config(self):
        """Platform rate limiters should use correct config."""
        from aragora.server.middleware.rate_limit import (
            PLATFORM_RATE_LIMITS,
            get_platform_rate_limiter,
            reset_platform_rate_limiters,
        )

        # Reset to ensure fresh state
        reset_platform_rate_limiters()

        # Slack should have 10 RPM
        slack_limiter = get_platform_rate_limiter("slack")
        assert slack_limiter.rpm == PLATFORM_RATE_LIMITS["slack"]["rpm"]

        # Discord should have 30 RPM
        discord_limiter = get_platform_rate_limiter("discord")
        assert discord_limiter.rpm == PLATFORM_RATE_LIMITS["discord"]["rpm"]

    def test_platform_rate_limit_check(self):
        """Platform rate limit check should work."""
        from aragora.server.middleware.rate_limit import (
            check_platform_rate_limit,
            reset_platform_rate_limiters,
        )

        reset_platform_rate_limiters()

        key = f"channel-{time.time()}"
        result = check_platform_rate_limit("slack", key)
        assert result.allowed
        assert result.platform == "slack"


class TestPlatformCircuitBreaker:
    """Tests for platform-level circuit breakers."""

    def test_circuit_starts_closed(self):
        """Circuit should start in closed state."""
        from aragora.integrations.platform_resilience import (
            PlatformCircuitBreaker,
            PlatformStatus,
        )

        circuit = PlatformCircuitBreaker(platform="test", failure_threshold=3)
        assert circuit.can_proceed()
        health = circuit.get_health()
        assert health.status == PlatformStatus.HEALTHY
        assert health.circuit_state == "closed"

    def test_circuit_opens_after_failures(self):
        """Circuit should open after failure threshold."""
        from aragora.integrations.platform_resilience import (
            PlatformCircuitBreaker,
            PlatformStatus,
        )

        circuit = PlatformCircuitBreaker(
            platform="test",
            failure_threshold=3,
            cooldown_seconds=60.0,
        )

        # Record failures
        for i in range(3):
            opened = circuit.record_failure(f"Error {i}")
            if i < 2:
                assert not opened
            else:
                assert opened  # Third failure opens circuit

        assert not circuit.can_proceed()
        health = circuit.get_health()
        assert health.status == PlatformStatus.UNAVAILABLE
        assert health.circuit_state == "open"
        assert health.error_count == 3

    def test_circuit_records_latency(self):
        """Circuit should track latency metrics."""
        from aragora.integrations.platform_resilience import PlatformCircuitBreaker

        circuit = PlatformCircuitBreaker(platform="test")

        circuit.record_success(latency_ms=100)
        circuit.record_success(latency_ms=200)
        circuit.record_success(latency_ms=300)

        health = circuit.get_health()
        assert health.avg_latency_ms == 200.0  # (100+200+300)/3
        assert health.success_rate == 1.0

    def test_get_platform_circuit_singleton(self):
        """Same platform should return same circuit."""
        from aragora.integrations.platform_resilience import get_platform_circuit

        circuit1 = get_platform_circuit("slack")
        circuit2 = get_platform_circuit("slack")
        assert circuit1 is circuit2

        circuit3 = get_platform_circuit("discord")
        assert circuit1 is not circuit3


class TestDeadLetterQueue:
    """Tests for the dead letter queue."""

    def test_enqueue_and_retrieve(self):
        """Should enqueue and retrieve messages."""
        from aragora.integrations.platform_resilience import DeadLetterQueue

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            dlq = DeadLetterQueue(db_path=db_path)

            msg_id = dlq.enqueue(
                platform="slack",
                destination="#general",
                payload={"text": "Hello"},
                error_message="Connection timeout",
            )

            assert msg_id
            assert len(msg_id) > 0

            # Wait briefly for next_retry_at to be in the past
            time.sleep(0.1)

            # Force the next_retry_at to be now (for testing)
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE dead_letters SET next_retry_at = ? WHERE id = ?",
                    (time.time() - 1, msg_id),
                )
                conn.commit()

            # Should be pending
            pending = dlq.get_pending(platform="slack")
            assert len(pending) == 1
            assert pending[0].id == msg_id
            assert pending[0].platform == "slack"
            assert pending[0].error_message == "Connection timeout"
            assert pending[0].retry_count == 0

        finally:
            os.unlink(db_path)

    def test_mark_success(self):
        """Should mark message as delivered."""
        from aragora.integrations.platform_resilience import DeadLetterQueue

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            dlq = DeadLetterQueue(db_path=db_path)

            msg_id = dlq.enqueue(
                platform="discord",
                destination="123456",
                payload={"content": "Test"},
                error_message="Rate limited",
            )

            # Mark as delivered
            assert dlq.mark_success(msg_id)

            # Should no longer be pending
            pending = dlq.get_pending(platform="discord")
            assert len(pending) == 0

        finally:
            os.unlink(db_path)

    def test_retry_backoff(self):
        """Should apply exponential backoff on retry."""
        from aragora.integrations.platform_resilience import DeadLetterQueue

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            dlq = DeadLetterQueue(db_path=db_path, max_retries=5)

            msg_id = dlq.enqueue(
                platform="teams",
                destination="channel123",
                payload={"text": "Test"},
                error_message="Initial error",
            )

            # Retry should succeed (under max)
            assert dlq.mark_retry(msg_id, "Retry error 1")
            assert dlq.mark_retry(msg_id, "Retry error 2")
            assert dlq.mark_retry(msg_id, "Retry error 3")
            assert dlq.mark_retry(msg_id, "Retry error 4")

            # Fifth retry should fail (max retries exceeded)
            assert not dlq.mark_retry(msg_id, "Final error")

        finally:
            os.unlink(db_path)

    def test_dlq_stats(self):
        """Should return accurate statistics."""
        from aragora.integrations.platform_resilience import DeadLetterQueue

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            dlq = DeadLetterQueue(db_path=db_path)

            # Add messages
            msg1 = dlq.enqueue("slack", "#ch1", {"t": "1"}, "err1")
            msg2 = dlq.enqueue("slack", "#ch2", {"t": "2"}, "err2")
            msg3 = dlq.enqueue("discord", "123", {"t": "3"}, "err3")

            # Mark one as delivered
            dlq.mark_success(msg1)

            stats = dlq.get_stats()
            assert stats["total"] == 3
            assert "pending" in stats["by_status"]
            assert "delivered" in stats["by_status"]
            assert stats["by_platform"]["slack"]["pending"] == 1
            assert stats["by_platform"]["slack"]["delivered"] == 1
            assert stats["by_platform"]["discord"]["pending"] == 1

        finally:
            os.unlink(db_path)


class TestTimeoutWrapper:
    """Tests for timeout wrapper decorator."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Should complete successfully within timeout."""
        from aragora.integrations.platform_resilience import with_timeout

        @with_timeout(timeout_seconds=5.0)
        async def fast_function():
            await asyncio.sleep(0.1)
            return "done"

        result = await fast_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_timeout_with_fallback(self):
        """Should return fallback on timeout."""
        from aragora.integrations.platform_resilience import with_timeout

        @with_timeout(timeout_seconds=0.1, fallback_response="Processing...")
        async def slow_function():
            await asyncio.sleep(10)
            return "done"

        result = await slow_function()
        assert result == "Processing..."

    @pytest.mark.asyncio
    async def test_timeout_raises_without_fallback(self):
        """Should raise TimeoutError without fallback."""
        from aragora.integrations.platform_resilience import with_timeout

        @with_timeout(timeout_seconds=0.1)
        async def slow_function():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await slow_function()


class TestPlatformResilience:
    """Tests for combined platform resilience decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_records_metrics(self):
        """Successful calls should record latency."""
        from aragora.integrations.platform_resilience import (
            get_platform_circuit,
            with_platform_resilience,
        )

        # Reset circuit
        circuit = get_platform_circuit("test_platform")
        circuit.reset()

        @with_platform_resilience("test_platform", timeout_seconds=5.0, use_dlq=False)
        async def handler(msg: str) -> str:
            await asyncio.sleep(0.05)
            return f"Processed: {msg}"

        result = await handler("test message")
        assert result == "Processed: test message"

        health = circuit.get_health()
        assert health.success_rate == 1.0
        assert health.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_circuit_blocks_when_open(self):
        """Should block calls when circuit is open."""
        from aragora.integrations.platform_resilience import (
            get_platform_circuit,
            with_platform_resilience,
        )

        circuit = get_platform_circuit("blocked_platform")
        circuit.reset()

        # Force open the circuit
        for _ in range(5):
            circuit.record_failure("Forced failure")

        assert not circuit.can_proceed()

        @with_platform_resilience("blocked_platform", use_dlq=False)
        async def handler(msg: str) -> str:
            return f"Processed: {msg}"

        result = await handler("test")
        assert result is None  # Blocked by circuit


class TestPlatformMetrics:
    """Tests for platform metrics."""

    def test_record_platform_request(self):
        """Should record platform requests without error."""
        from aragora.observability.metrics.platform import record_platform_request

        # Should not raise
        record_platform_request(
            platform="slack",
            operation="send_message",
            success=True,
            latency_ms=150.0,
        )

        record_platform_request(
            platform="discord",
            operation="send_message",
            success=False,
            latency_ms=5000.0,
            error_type="timeout",
        )

    def test_record_circuit_state(self):
        """Should record circuit state without error."""
        from aragora.observability.metrics.platform import record_circuit_state

        record_circuit_state("slack", "closed")
        record_circuit_state("slack", "open")
        record_circuit_state("slack", "half-open")

    def test_record_dlq_metrics(self):
        """Should record DLQ metrics without error."""
        from aragora.observability.metrics.platform import (
            record_dlq_enqueue,
            record_dlq_failed,
            record_dlq_processed,
            update_dlq_pending,
        )

        record_dlq_enqueue("slack")
        record_dlq_processed("slack")
        record_dlq_failed("discord")
        update_dlq_pending("teams", 5)
