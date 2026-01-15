"""
WebSocket Load Testing Suite for Aragora.

Run with:
    pytest tests/load/websocket_load.py -v --asyncio-mode=auto

For stress testing:
    pytest tests/load/websocket_load.py -v -k stress --asyncio-mode=auto

Environment Variables:
    ARAGORA_WS_URL: WebSocket URL (default: ws://localhost:8080/ws)
    ARAGORA_CONCURRENT_WS: Concurrent connections (default: 50)
    ARAGORA_WS_DURATION: Test duration in seconds (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

# Configuration
WS_URL = os.environ.get("ARAGORA_WS_URL", "ws://localhost:8080/ws")
CONCURRENT_WS = int(os.environ.get("ARAGORA_CONCURRENT_WS", "50"))
TEST_DURATION = int(os.environ.get("ARAGORA_WS_DURATION", "30"))


@dataclass
class WebSocketMetrics:
    """Metrics collected during WebSocket load test."""

    connections_attempted: int = 0
    connections_successful: int = 0
    connections_failed: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_received: int = 0
    connection_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def avg_connection_time_ms(self) -> float:
        if not self.connection_times:
            return 0.0
        return sum(self.connection_times) / len(self.connection_times) * 1000

    @property
    def p95_connection_time_ms(self) -> float:
        if not self.connection_times:
            return 0.0
        sorted_times = sorted(self.connection_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] * 1000

    @property
    def messages_per_second(self) -> float:
        if self.duration == 0:
            return 0.0
        return self.messages_received / self.duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connections_attempted": self.connections_attempted,
            "connections_successful": self.connections_successful,
            "connections_failed": self.connections_failed,
            "success_rate": (
                self.connections_successful / self.connections_attempted * 100
                if self.connections_attempted > 0
                else 0
            ),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_received": self.bytes_received,
            "avg_connection_time_ms": round(self.avg_connection_time_ms, 2),
            "p95_connection_time_ms": round(self.p95_connection_time_ms, 2),
            "messages_per_second": round(self.messages_per_second, 2),
            "duration_seconds": round(self.duration, 2),
            "error_count": len(self.errors),
        }


def random_string(length: int = 10) -> str:
    """Generate random string."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


class WebSocketClient:
    """Async WebSocket client for load testing."""

    def __init__(self, url: str, metrics: WebSocketMetrics, client_id: int):
        self.url = url
        self.metrics = metrics
        self.client_id = client_id
        self.ws: Optional[Any] = None
        self.connected = False

    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            import websockets

            self.metrics.connections_attempted += 1
            start_time = time.time()

            self.ws = await asyncio.wait_for(
                websockets.connect(self.url, ping_interval=20, ping_timeout=10),
                timeout=10.0,
            )

            connection_time = time.time() - start_time
            self.metrics.connection_times.append(connection_time)
            self.metrics.connections_successful += 1
            self.connected = True
            return True

        except ImportError:
            self.metrics.errors.append("websockets package not installed")
            return False
        except asyncio.TimeoutError:
            self.metrics.connections_failed += 1
            self.metrics.errors.append(f"Client {self.client_id}: Connection timeout")
            return False
        except Exception as e:
            self.metrics.connections_failed += 1
            self.metrics.errors.append(f"Client {self.client_id}: {str(e)[:100]}")
            return False

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to server."""
        if not self.ws or not self.connected:
            return

        try:
            await self.ws.send(json.dumps(message))
            self.metrics.messages_sent += 1
        except Exception as e:
            self.metrics.errors.append(f"Send error: {str(e)[:50]}")

    async def receive_messages(self, duration: float) -> None:
        """Receive messages for specified duration."""
        if not self.ws or not self.connected:
            return

        end_time = time.time() + duration

        try:
            while time.time() < end_time:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                    self.metrics.messages_received += 1
                    self.metrics.bytes_received += len(message)
                except asyncio.TimeoutError:
                    continue
        except Exception:
            pass

    async def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.connected = False


async def run_websocket_load_test(
    url: str,
    concurrent: int,
    duration: float,
    send_messages: bool = True,
) -> WebSocketMetrics:
    """
    Run WebSocket load test.

    Args:
        url: WebSocket URL to connect to
        concurrent: Number of concurrent connections
        duration: Test duration in seconds
        send_messages: Whether to send messages during test

    Returns:
        WebSocketMetrics with test results
    """
    metrics = WebSocketMetrics()
    metrics.start_time = time.time()

    clients: List[WebSocketClient] = []

    # Create clients
    for i in range(concurrent):
        clients.append(WebSocketClient(url, metrics, i))

    # Connect all clients
    connect_tasks = [client.connect() for client in clients]
    await asyncio.gather(*connect_tasks)

    print(f"Connected {metrics.connections_successful}/{metrics.connections_attempted} clients")

    if metrics.connections_successful == 0:
        metrics.end_time = time.time()
        return metrics

    # Send initial messages if requested
    if send_messages:
        for client in clients:
            if client.connected:
                await client.send_message(
                    {
                        "type": "subscribe",
                        "channel": "debates",
                        "client_id": f"load_test_{client.client_id}",
                    }
                )

    # Receive messages for duration
    receive_tasks = [client.receive_messages(duration) for client in clients if client.connected]
    await asyncio.gather(*receive_tasks)

    # Close all connections
    close_tasks = [client.close() for client in clients]
    await asyncio.gather(*close_tasks)

    metrics.end_time = time.time()
    return metrics


# =============================================================================
# Pytest Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_single_websocket_connection():
    """Test single WebSocket connection."""
    metrics = WebSocketMetrics()
    client = WebSocketClient(WS_URL, metrics, 0)

    connected = await client.connect()

    if not connected:
        pytest.skip(f"Could not connect to {WS_URL}: {metrics.errors}")

    assert client.connected
    await client.close()


@pytest.mark.asyncio
async def test_concurrent_websocket_connections():
    """Test multiple concurrent WebSocket connections."""
    metrics = await run_websocket_load_test(
        url=WS_URL,
        concurrent=10,
        duration=5.0,
        send_messages=True,
    )

    print(f"\nResults: {json.dumps(metrics.to_dict(), indent=2)}")

    if metrics.connections_attempted == 0:
        pytest.skip("No connections attempted")

    # Expect at least 50% success rate
    success_rate = metrics.connections_successful / metrics.connections_attempted
    assert success_rate >= 0.5, f"Success rate too low: {success_rate:.0%}"


@pytest.mark.asyncio
async def test_websocket_message_throughput():
    """Test WebSocket message throughput."""
    metrics = await run_websocket_load_test(
        url=WS_URL,
        concurrent=20,
        duration=10.0,
        send_messages=True,
    )

    print(f"\nThroughput: {metrics.messages_per_second:.1f} msg/s")
    print(f"Bytes received: {metrics.bytes_received}")

    if metrics.connections_successful == 0:
        pytest.skip("No connections established")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_websocket_stress():
    """Stress test with many concurrent connections."""
    metrics = await run_websocket_load_test(
        url=WS_URL,
        concurrent=CONCURRENT_WS,
        duration=TEST_DURATION,
        send_messages=True,
    )

    print(f"\n{'=' * 60}")
    print("WebSocket Stress Test Results")
    print("=" * 60)
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)

    if metrics.connections_successful == 0:
        pytest.skip("No connections established")

    # Assertions
    success_rate = metrics.connections_successful / metrics.connections_attempted
    assert success_rate >= 0.7, f"Success rate below 70%: {success_rate:.0%}"


@pytest.mark.asyncio
async def test_websocket_connection_churn():
    """Test rapid connect/disconnect cycles."""
    metrics = WebSocketMetrics()
    metrics.start_time = time.time()

    cycles = 20

    for i in range(cycles):
        client = WebSocketClient(WS_URL, metrics, i)
        connected = await client.connect()

        if connected:
            # Send a message
            await client.send_message({"type": "ping"})
            await asyncio.sleep(0.1)

        await client.close()

    metrics.end_time = time.time()

    print(f"\nChurn test: {metrics.connections_successful}/{cycles} successful")
    print(f"Avg connection time: {metrics.avg_connection_time_ms:.1f}ms")

    if metrics.connections_attempted == 0:
        pytest.skip("No connections attempted")


@pytest.mark.asyncio
async def test_debate_websocket_stream():
    """Test debate streaming WebSocket connection."""
    metrics = WebSocketMetrics()
    client = WebSocketClient(f"{WS_URL}/stream", metrics, 0)

    connected = await client.connect()
    if not connected:
        pytest.skip(f"Could not connect to stream endpoint: {metrics.errors}")

    # Subscribe to a debate
    await client.send_message(
        {
            "type": "subscribe",
            "debate_id": f"test_debate_{random_string(8)}",
        }
    )

    # Listen for a short time
    await client.receive_messages(2.0)
    await client.close()

    print(f"Received {metrics.messages_received} messages")


if __name__ == "__main__":
    # Run standalone for quick testing
    import sys

    async def main() -> None:
        print(f"Running WebSocket load test against {WS_URL}")
        print(f"Concurrent connections: {CONCURRENT_WS}")
        print(f"Duration: {TEST_DURATION}s")
        print()

        metrics = await run_websocket_load_test(
            url=WS_URL,
            concurrent=CONCURRENT_WS,
            duration=TEST_DURATION,
            send_messages=True,
        )

        print("\nResults:")
        print(json.dumps(metrics.to_dict(), indent=2))

        if metrics.errors[:5]:
            print("\nSample errors:")
            for error in metrics.errors[:5]:
                print(f"  - {error}")

    asyncio.run(main())
