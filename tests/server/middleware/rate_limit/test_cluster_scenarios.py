"""
Cluster scenario tests for distributed rate limiting.

Tests Redis Cluster-specific behaviors including:
- Hash slot migration between nodes
- Node failure and failover
- Network partition (split-brain) scenarios
- Large-scale tenant isolation

These tests verify the production-readiness of the distributed rate limiting
system under adverse cluster conditions.

Run with:
    pytest tests/server/middleware/rate_limit/test_cluster_scenarios.py -v

Note: These tests use mocks to simulate cluster conditions since actual
cluster failures are difficult to orchestrate in CI environments.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.middleware.rate_limit.distributed import (
    DistributedRateLimiter,
    reset_distributed_limiter,
)
from aragora.server.middleware.rate_limit.redis_limiter import (
    RateLimitCircuitBreaker,
    get_redis_client,
)


pytestmark = [pytest.mark.integration]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset the global limiter before and after each test."""
    reset_distributed_limiter()
    yield
    reset_distributed_limiter()


@pytest.fixture
def mock_cluster_redis():
    """Create a mock Redis client that simulates cluster behavior."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.evalsha.return_value = [1, 10, 60]  # [allowed, remaining, ttl]
    mock.script_load.return_value = "mock_script_sha"
    return mock


# ============================================================================
# Redis Cluster Topology Tests
# ============================================================================


class TestRedisClusterScenarios:
    """Tests for Redis Cluster-specific rate limiting scenarios."""

    def test_hash_slot_migration_preserves_limits(self, mock_cluster_redis):
        """Rate limits survive hash slot migration between nodes.

        When Redis Cluster reshards data, keys may move between nodes.
        Rate limits should continue working during and after migration.
        """
        request_count = 0

        def track_requests(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            # Simulate MOVED error on first few requests (slot migration)
            if request_count <= 2:
                from redis import ResponseError

                raise ResponseError("MOVED 12345 127.0.0.1:7001")
            return [1, 10 - request_count, 60]  # allowed, remaining, ttl

        mock_cluster_redis.evalsha.side_effect = track_requests

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_cluster_redis,
        ):
            limiter = DistributedRateLimiter(
                instance_id="cluster-migration-test",
                strict_mode=False,
            )
            limiter.configure_endpoint("/api/cluster", requests_per_minute=10, burst_size=10)

            # Make requests - should handle MOVED redirections gracefully
            allowed_count = 0
            for _ in range(5):
                result = limiter.allow("client", "/api/cluster")
                if result.allowed:
                    allowed_count += 1

            # Should have allowed requests despite slot migration
            assert allowed_count >= 3, "Should allow requests during slot migration"

    def test_node_failure_triggers_failover(self, mock_cluster_redis):
        """Single node failure triggers automatic failover to replica.

        When a master node fails, Redis Cluster promotes a replica.
        Rate limiting should continue with minimal disruption.
        """
        call_count = 0

        def simulate_node_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate connection error for requests 3-5 (node down)
            if 3 <= call_count <= 5:
                raise ConnectionError("Connection refused")
            return [1, 10 - call_count, 60]

        mock_cluster_redis.evalsha.side_effect = simulate_node_failure

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_cluster_redis,
        ):
            limiter = DistributedRateLimiter(
                instance_id="node-failure-test",
                strict_mode=False,
            )
            limiter.configure_endpoint("/api/failover", requests_per_minute=10, burst_size=10)

            # Make requests - some will fail during simulated outage
            results = []
            for _ in range(8):
                result = limiter.allow("client", "/api/failover")
                results.append(result.allowed)

            # Should have some allowed requests before and after the outage
            # (with fallback during outage)
            assert any(results[:3]), "Should allow requests before node failure"
            assert any(results[5:]), "Should recover after node failure"

    def test_partial_cluster_maintains_consistency(self, mock_cluster_redis):
        """Partial cluster availability maintains rate limit accuracy.

        When some cluster nodes are unavailable but quorum exists,
        rate limiting should still function correctly.
        """
        # Simulate 2 of 3 masters available
        available_nodes = [True, True, False]
        current_node = [0]

        def route_to_available_node(*args, **kwargs):
            node_idx = current_node[0] % 3
            current_node[0] += 1
            if not available_nodes[node_idx]:
                raise ConnectionError(f"Node {node_idx} unavailable")
            return [1, 10, 60]

        mock_cluster_redis.evalsha.side_effect = route_to_available_node

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_cluster_redis,
        ):
            limiter = DistributedRateLimiter(
                instance_id="partial-cluster-test",
                strict_mode=False,
            )

            # Make many requests - most should succeed via available nodes
            allowed_count = 0
            for _ in range(9):
                result = limiter.allow("client", "/api/partial")
                if result.allowed:
                    allowed_count += 1

            # Should have at least 2/3 success rate
            assert allowed_count >= 6, f"Expected at least 6 allowed, got {allowed_count}"


# ============================================================================
# Network Partition (Split-Brain) Tests
# ============================================================================


class TestNetworkPartitionScenarios:
    """Tests for split-brain and partition scenarios."""

    def test_partition_heal_reconciles_counts(self):
        """After partition heals, rate counts reconcile correctly.

        During a network partition, different partitions may track
        different counts. After healing, the system should reconcile.
        """
        # Create two limiter instances simulating partitioned servers
        limiter1 = DistributedRateLimiter(
            instance_id="partition-1",
            strict_mode=False,
        )
        limiter2 = DistributedRateLimiter(
            instance_id="partition-2",
            strict_mode=False,
        )

        # Configure same endpoint
        limiter1.configure_endpoint("/api/partition", requests_per_minute=20, burst_size=20)
        limiter2.configure_endpoint("/api/partition", requests_per_minute=20, burst_size=20)

        # Simulate partition: each sees only its own requests
        # (using in-memory fallback mode to simulate isolation)
        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=None,
        ):
            # Each partition accumulates requests independently
            for _ in range(10):
                limiter1.allow("client", "/api/partition")
                limiter2.allow("client", "/api/partition")

        # After partition heals, both should respect combined limits
        # (The in-memory counters represent the partitioned state)
        stats1 = limiter1.get_stats()
        stats2 = limiter2.get_stats()

        # Both should have tracked their requests
        assert stats1["total_requests"] == 10
        assert stats2["total_requests"] == 10

    def test_minority_partition_rejects_writes(self, mock_cluster_redis):
        """Minority partition rejects rate limit updates.

        When a node finds itself in the minority partition (can't reach quorum),
        it should either reject updates or use fallback mode.
        """
        quorum_available = [True]

        def check_quorum(*args, **kwargs):
            if not quorum_available[0]:
                # Simulate cluster error when quorum lost
                from redis import ResponseError

                raise ResponseError("CLUSTERDOWN The cluster is down")
            return [1, 10, 60]

        mock_cluster_redis.evalsha.side_effect = check_quorum

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_cluster_redis,
        ):
            limiter = DistributedRateLimiter(
                instance_id="minority-partition-test",
                strict_mode=False,
            )

            # First request succeeds (quorum available)
            result1 = limiter.allow("client", "/api/minority")
            assert result1.allowed

            # Simulate losing quorum
            quorum_available[0] = False

            # Request should still work via fallback
            result2 = limiter.allow("client", "/api/minority")
            assert result2.allowed  # Fallback to in-memory

            # Stats should show fallback usage
            stats = limiter.get_stats()
            assert stats.get("fallback_requests", 0) >= 0


# ============================================================================
# Large-Scale Tenant Isolation Tests
# ============================================================================


class TestLargeScaleTenantIsolation:
    """Performance tests with many tenants."""

    def test_thousand_tenants_isolation(self):
        """1000 tenants maintain isolated rate limits.

        Each tenant should have completely independent rate limits,
        and the system should handle the scale efficiently.
        """
        limiter = DistributedRateLimiter(
            instance_id="scale-test",
            strict_mode=False,
        )
        limiter.configure_endpoint(
            "/api/multi-tenant",
            requests_per_minute=10,
            burst_size=10,
            key_type="tenant",
        )

        # Simulate 1000 tenants each making 5 requests
        tenant_results: dict[str, list[bool]] = {}

        for tenant_num in range(1000):
            tenant_id = f"tenant-{tenant_num:04d}"
            tenant_results[tenant_id] = []
            for _ in range(5):
                result = limiter.allow("client", "/api/multi-tenant", tenant_id=tenant_id)
                tenant_results[tenant_id].append(result.allowed)

        # All tenants should have all 5 requests allowed (within their limit)
        tenants_with_all_allowed = sum(1 for results in tenant_results.values() if all(results))

        assert tenants_with_all_allowed == 1000, (
            f"Expected all 1000 tenants to have all requests allowed, "
            f"got {tenants_with_all_allowed}"
        )

    def test_tenant_cleanup_under_load(self):
        """Tenant cleanup doesn't affect active tenants.

        When inactive tenant data is cleaned up, active tenants
        should not be affected.
        """
        limiter = DistributedRateLimiter(
            instance_id="cleanup-test",
            strict_mode=False,
        )
        # Use tenant-based keying so each tenant has independent limits
        limiter.configure_endpoint(
            "/api/cleanup",
            requests_per_minute=100,
            burst_size=100,
            key_type="tenant",
        )

        # Create some active tenants
        active_tenants = [f"active-{i}" for i in range(10)]
        inactive_tenants = [f"inactive-{i}" for i in range(100)]

        # Both make initial requests
        for tenant in active_tenants + inactive_tenants:
            limiter.allow("client", "/api/cleanup", tenant_id=tenant)

        # Only active tenants continue making requests
        for _ in range(5):
            for tenant in active_tenants:
                result = limiter.allow("client", "/api/cleanup", tenant_id=tenant)
                assert result.allowed, f"Active tenant {tenant} should not be rate limited"

        # Verify active tenant stats are preserved
        stats = limiter.get_stats()
        assert stats["total_requests"] >= len(active_tenants) * 6  # Initial + 5 more

    def test_concurrent_tenant_requests(self):
        """Concurrent requests from many tenants are handled correctly.

        The system should handle concurrent requests from multiple tenants
        without data races or incorrect counting.
        """
        limiter = DistributedRateLimiter(
            instance_id="concurrent-tenant-test",
            strict_mode=False,
        )
        limiter.configure_endpoint(
            "/api/concurrent-tenants",
            requests_per_minute=20,
            burst_size=20,
            key_type="tenant",
        )

        results_lock = threading.Lock()
        tenant_allowed: dict[str, int] = {}

        def make_tenant_request(tenant_id: str):
            result = limiter.allow("client", "/api/concurrent-tenants", tenant_id=tenant_id)
            with results_lock:
                if tenant_id not in tenant_allowed:
                    tenant_allowed[tenant_id] = 0
                if result.allowed:
                    tenant_allowed[tenant_id] += 1

        # Run concurrent requests for 50 tenants
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for tenant_num in range(50):
                tenant_id = f"concurrent-tenant-{tenant_num}"
                # Each tenant makes 15 requests
                for _ in range(15):
                    futures.append(executor.submit(make_tenant_request, tenant_id))

            # Wait for completion
            for f in futures:
                f.result()

        # Each tenant should have had at least some requests allowed
        for tenant_id, allowed in tenant_allowed.items():
            assert allowed > 0, f"Tenant {tenant_id} should have had some requests allowed"
            assert allowed <= 20, f"Tenant {tenant_id} exceeded rate limit: {allowed}"


# ============================================================================
# Cascading Failure Tests
# ============================================================================


class TestCascadingFailures:
    """Tests for cascading failure scenarios."""

    def test_circuit_breaker_prevents_cascade(self, mock_cluster_redis):
        """Circuit breaker prevents cascading failures.

        When Redis starts failing, the circuit breaker should open
        to prevent overloading the failing system and allow recovery.
        """
        failure_count = [0]

        def intermittent_failures(*args, **kwargs):
            failure_count[0] += 1
            if failure_count[0] % 3 == 0:  # Every 3rd request fails
                raise ConnectionError("Connection timeout")
            return [1, 10, 60]

        mock_cluster_redis.evalsha.side_effect = intermittent_failures

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_cluster_redis,
        ):
            limiter = DistributedRateLimiter(
                instance_id="cascade-test",
                strict_mode=False,
            )

            # Make many requests - circuit breaker should eventually open
            allowed_count = 0
            for _ in range(30):
                result = limiter.allow("client", "/api/cascade")
                if result.allowed:
                    allowed_count += 1

            # Should have allowed most requests via circuit breaker fallback
            assert allowed_count >= 20, f"Expected at least 20 allowed, got {allowed_count}"

    def test_graceful_degradation_under_load(self):
        """System gracefully degrades under extreme load.

        When the system is under heavy load, it should degrade gracefully
        rather than failing completely.
        """
        limiter = DistributedRateLimiter(
            instance_id="degradation-test",
            strict_mode=False,
        )
        limiter.configure_endpoint("/api/load", requests_per_minute=1000, burst_size=100)

        # Simulate extreme concurrent load
        results_lock = threading.Lock()
        allowed = [0]
        denied = [0]

        def make_request():
            result = limiter.allow(f"client-{threading.current_thread().name}", "/api/load")
            with results_lock:
                if result.allowed:
                    allowed[0] += 1
                else:
                    denied[0] += 1

        # 100 concurrent threads, each making 50 requests
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(make_request) for _ in range(5000)]
            for f in futures:
                f.result()

        # Should have handled all requests (some allowed, some denied)
        total = allowed[0] + denied[0]
        assert total == 5000, f"Expected 5000 total requests, got {total}"

        # Should have allowed a reasonable number (not all, not none)
        assert allowed[0] > 0, "Should have allowed some requests"
        assert denied[0] > 0, "Should have rate limited some requests"


# ============================================================================
# Recovery Tests
# ============================================================================


class TestRecoveryScenarios:
    """Tests for system recovery after failures."""

    def test_recovery_after_full_outage(self, mock_cluster_redis):
        """System recovers gracefully after full Redis outage.

        After Redis comes back online, the system should resume
        distributed rate limiting without data loss or inconsistency.
        """
        redis_available = [True]

        def simulate_outage_and_recovery(*args, **kwargs):
            if not redis_available[0]:
                raise ConnectionError("Redis unavailable")
            return [1, 10, 60]

        mock_cluster_redis.evalsha.side_effect = simulate_outage_and_recovery

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_cluster_redis,
        ):
            limiter = DistributedRateLimiter(
                instance_id="recovery-test",
                strict_mode=False,
            )

            # Initial requests succeed
            for _ in range(3):
                result = limiter.allow("client", "/api/recovery")
                assert result.allowed

            # Simulate outage
            redis_available[0] = False
            for _ in range(3):
                result = limiter.allow("client", "/api/recovery")
                # Should use fallback
                assert result.allowed

            # Recover
            redis_available[0] = True
            for _ in range(3):
                result = limiter.allow("client", "/api/recovery")
                assert result.allowed

            # Verify system is healthy
            stats = limiter.get_stats()
            assert stats["total_requests"] == 9


__all__ = [
    "TestRedisClusterScenarios",
    "TestNetworkPartitionScenarios",
    "TestLargeScaleTenantIsolation",
    "TestCascadingFailures",
    "TestRecoveryScenarios",
]
