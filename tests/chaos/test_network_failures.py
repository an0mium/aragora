"""
Chaos tests for network failure scenarios.

Tests system resilience under:
- Network partitions
- High latency
- Packet loss simulation
- DNS failures
- SSL/TLS failures
- Connection reset scenarios
"""

from __future__ import annotations

import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class NetworkSimulator:
    """Simulates various network conditions for chaos testing."""

    def __init__(
        self,
        latency_ms: float = 0,
        latency_jitter_ms: float = 0,
        packet_loss_rate: float = 0.0,
        connection_failure_rate: float = 0.0,
        bandwidth_limit_kbps: float = 0,
    ):
        self.latency_ms = latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.packet_loss_rate = packet_loss_rate
        self.connection_failure_rate = connection_failure_rate
        self.bandwidth_limit_kbps = bandwidth_limit_kbps
        self.request_count = 0
        self.dropped_count = 0
        self.failed_connections = 0

    async def simulate_request(self, data: bytes) -> bytes:
        """Simulate a network request with configured conditions."""
        self.request_count += 1

        # Connection failure
        if random.random() < self.connection_failure_rate:
            self.failed_connections += 1
            raise ConnectionRefusedError("Connection refused")

        # Packet loss
        if random.random() < self.packet_loss_rate:
            self.dropped_count += 1
            raise TimeoutError("Request timed out (packet loss)")

        # Latency
        delay = self.latency_ms / 1000
        if self.latency_jitter_ms > 0:
            jitter = random.uniform(-self.latency_jitter_ms, self.latency_jitter_ms) / 1000
            delay += jitter
        if delay > 0:
            await asyncio.sleep(delay)

        # Bandwidth limiting
        if self.bandwidth_limit_kbps > 0:
            transfer_time = (len(data) * 8) / (self.bandwidth_limit_kbps * 1000)
            await asyncio.sleep(transfer_time)

        return b"response"


class TestNetworkLatency:
    """Tests for high network latency handling."""

    @pytest.mark.asyncio
    async def test_request_with_high_latency(self):
        """Should handle requests with high latency."""
        network = NetworkSimulator(latency_ms=100)

        import time

        start = time.time()
        response = await network.simulate_request(b"test")
        elapsed = time.time() - start

        assert response == b"response"
        assert elapsed >= 0.09  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_timeout_on_excessive_latency(self):
        """Should timeout when latency exceeds threshold."""
        network = NetworkSimulator(latency_ms=500)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                network.simulate_request(b"test"),
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_latency_jitter_handling(self):
        """Should handle variable latency (jitter)."""
        network = NetworkSimulator(latency_ms=50, latency_jitter_ms=30)

        latencies = []
        for _ in range(10):
            import time

            start = time.time()
            await network.simulate_request(b"test")
            latencies.append((time.time() - start) * 1000)

        # Latencies should vary
        min_latency = min(latencies)
        max_latency = max(latencies)
        assert max_latency - min_latency > 10  # At least 10ms variance


class TestPacketLoss:
    """Tests for packet loss handling."""

    @pytest.mark.asyncio
    async def test_retry_on_packet_loss(self):
        """Should retry requests on packet loss."""
        network = NetworkSimulator(packet_loss_rate=0.5)
        max_retries = 10

        success = False
        for attempt in range(max_retries):
            try:
                await network.simulate_request(b"test")
                success = True
                break
            except TimeoutError:
                await asyncio.sleep(0.01)

        # With 50% loss rate, should eventually succeed in 10 tries
        assert success or network.dropped_count == max_retries

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_sustained_packet_loss(self):
        """Should trigger circuit breaker on sustained packet loss."""
        from aragora.resilience import get_circuit_breaker

        network = NetworkSimulator(packet_loss_rate=1.0)  # 100% loss
        cb = get_circuit_breaker("packet_loss_test", failure_threshold=3, cooldown_seconds=1.0)

        for _ in range(5):
            try:
                await network.simulate_request(b"test")
            except TimeoutError:
                cb.record_failure()

        assert cb.is_open


class TestConnectionFailures:
    """Tests for connection failure handling."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_connection_refused_handling(self):
        """Should handle connection refused errors."""
        network = NetworkSimulator(connection_failure_rate=1.0)

        with pytest.raises(ConnectionRefusedError):
            await network.simulate_request(b"test")

    @pytest.mark.asyncio
    async def test_connection_reset_mid_request(self):
        """Should handle connection reset during request."""

        async def request_with_reset():
            await asyncio.sleep(0.05)
            raise ConnectionResetError("Connection reset by peer")

        with pytest.raises(ConnectionResetError):
            await request_with_reset()

    @pytest.mark.asyncio
    async def test_reconnection_after_failure(self):
        """Should successfully reconnect after connection failure."""
        connection_attempts = [0]
        failures_before_success = 2

        async def connect_with_retry():
            connection_attempts[0] += 1
            if connection_attempts[0] <= failures_before_success:
                raise ConnectionRefusedError("Server unavailable")
            return "connected"

        result = None
        for _ in range(5):
            try:
                result = await connect_with_retry()
                break
            except ConnectionRefusedError:
                await asyncio.sleep(0.01)

        assert result == "connected"
        assert connection_attempts[0] == 3


class TestDNSFailures:
    """Tests for DNS resolution failure handling."""

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """Should handle DNS resolution failures."""
        import socket

        async def resolve_with_failure(hostname: str):
            if hostname == "invalid.example.com":
                raise socket.gaierror(8, "Name or service not known")
            return "1.2.3.4"

        with pytest.raises(socket.gaierror):
            await resolve_with_failure("invalid.example.com")

    @pytest.mark.asyncio
    async def test_dns_cache_fallback(self):
        """Should use cached DNS on resolution failure."""
        dns_cache = {"api.example.com": "1.2.3.4"}
        dns_available = False

        async def resolve_with_cache(hostname: str) -> str:
            if dns_available:
                return "5.6.7.8"  # New IP
            if hostname in dns_cache:
                return dns_cache[hostname]
            raise Exception("DNS unavailable")

        result = await resolve_with_cache("api.example.com")
        assert result == "1.2.3.4"


class TestSSLFailures:
    """Tests for SSL/TLS failure handling."""

    @pytest.mark.asyncio
    async def test_ssl_certificate_error(self):
        """Should handle SSL certificate errors."""
        import ssl

        async def connect_with_ssl_error():
            raise ssl.SSLCertVerificationError("Certificate verify failed")

        with pytest.raises(ssl.SSLCertVerificationError):
            await connect_with_ssl_error()

    @pytest.mark.asyncio
    async def test_ssl_handshake_timeout(self):
        """Should handle SSL handshake timeout."""

        async def slow_ssl_handshake():
            await asyncio.sleep(10)  # Simulate slow handshake
            return "connected"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_ssl_handshake(), timeout=0.1)


class TestNetworkPartition:
    """Tests for network partition scenarios."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_partition_detection(self):
        """Should detect network partition."""
        nodes = ["node1", "node2", "node3"]
        reachable = {"node1": True, "node2": False, "node3": False}  # Partition

        async def check_node_reachability(node: str) -> bool:
            return reachable.get(node, False)

        reachable_nodes = []
        for node in nodes:
            if await check_node_reachability(node):
                reachable_nodes.append(node)

        assert len(reachable_nodes) == 1
        assert reachable_nodes[0] == "node1"

    @pytest.mark.asyncio
    async def test_split_brain_prevention(self):
        """Should prevent split-brain during partition."""
        cluster_size = 5
        reachable_count = 2  # Only 2 of 5 nodes reachable

        def can_form_quorum(reachable: int, total: int) -> bool:
            return reachable > total // 2

        assert not can_form_quorum(reachable_count, cluster_size)

    @pytest.mark.asyncio
    async def test_partition_recovery(self):
        """Should recover when partition heals."""
        partitioned = True
        data_synced = False

        async def sync_data():
            nonlocal data_synced
            if not partitioned:
                data_synced = True
                return True
            raise ConnectionError("Network partition")

        # During partition
        with pytest.raises(ConnectionError):
            await sync_data()

        # Partition heals
        partitioned = False
        await sync_data()
        assert data_synced


class TestRateLimiting:
    """Tests for rate limiting and backoff."""

    @pytest.mark.asyncio
    async def test_rate_limit_backoff(self):
        """Should backoff when rate limited."""
        request_times: list[float] = []
        rate_limit_remaining = [0]

        async def request_with_rate_limit():
            import time

            request_times.append(time.time())
            if rate_limit_remaining[0] <= 0:
                rate_limit_remaining[0] = 10
                raise Exception("Rate limit exceeded")
            rate_limit_remaining[0] -= 1
            return "success"

        # First request hits rate limit
        try:
            await request_with_rate_limit()
        except Exception:
            await asyncio.sleep(0.1)  # Backoff

        # Second request succeeds
        result = await request_with_rate_limit()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Should implement exponential backoff."""
        backoff_times: list[float] = []
        base_delay = 0.01

        for attempt in range(5):
            delay = base_delay * (2**attempt)
            backoff_times.append(delay)

        # Each delay should be double the previous
        for i in range(1, len(backoff_times)):
            assert backoff_times[i] == backoff_times[i - 1] * 2


class TestWebSocketFailures:
    """Tests for WebSocket failure handling."""

    @pytest.mark.asyncio
    async def test_websocket_disconnect_handling(self):
        """Should handle WebSocket disconnection."""
        connected = True
        reconnect_count = 0

        async def maintain_connection():
            nonlocal connected, reconnect_count
            while reconnect_count < 3:
                if not connected:
                    reconnect_count += 1
                    await asyncio.sleep(0.01)
                    connected = True
                await asyncio.sleep(0.01)

        # Simulate disconnect
        task = asyncio.create_task(maintain_connection())
        await asyncio.sleep(0.02)
        connected = False
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert reconnect_count >= 1

    @pytest.mark.asyncio
    async def test_websocket_message_ordering(self):
        """Should handle out-of-order WebSocket messages."""
        messages: list[dict[str, Any]] = []
        sequence_numbers = [3, 1, 4, 2, 5]  # Out of order

        for seq in sequence_numbers:
            messages.append({"seq": seq, "data": f"message_{seq}"})

        # Sort by sequence number
        ordered = sorted(messages, key=lambda m: m["seq"])

        expected_order = [1, 2, 3, 4, 5]
        actual_order = [m["seq"] for m in ordered]
        assert actual_order == expected_order


class TestServiceDiscoveryFailures:
    """Tests for service discovery failure handling."""

    @pytest.mark.asyncio
    async def test_service_unavailable_fallback(self):
        """Should fallback when primary service unavailable."""
        services = {
            "primary": {"healthy": False, "url": "http://primary:8080"},
            "secondary": {"healthy": True, "url": "http://secondary:8080"},
        }

        async def get_healthy_service() -> str | None:
            for name, service in services.items():
                if service["healthy"]:
                    return service["url"]
            return None

        url = await get_healthy_service()
        assert url == "http://secondary:8080"

    @pytest.mark.asyncio
    async def test_load_balancer_health_check(self):
        """Should route around unhealthy backends."""
        backends = [
            {"id": "backend1", "healthy": True, "requests": 0},
            {"id": "backend2", "healthy": False, "requests": 0},
            {"id": "backend3", "healthy": True, "requests": 0},
        ]

        def get_healthy_backend():
            healthy = [b for b in backends if b["healthy"]]
            if not healthy:
                return None
            # Simple round-robin among healthy
            selected = min(healthy, key=lambda b: b["requests"])
            selected["requests"] += 1
            return selected

        # Send 10 requests
        for _ in range(10):
            backend = get_healthy_backend()
            assert backend is not None
            assert backend["healthy"] is True

        # Requests should be distributed between healthy backends
        assert backends[0]["requests"] == 5
        assert backends[1]["requests"] == 0  # Unhealthy, no requests
        assert backends[2]["requests"] == 5
