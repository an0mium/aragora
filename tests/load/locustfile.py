"""
Locust load testing suite for Aragora.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8080

For headless mode:
    locust -f tests/load/locustfile.py --host=http://localhost:8080 \
        --headless -u 100 -r 10 --run-time 5m

Scenarios:
    - HealthCheckUser: Lightweight health check probes
    - APIBrowsingUser: Typical API browsing patterns
    - DebateUser: Users creating and monitoring debates
    - HeavyLoadUser: Stress testing with concurrent debates

Environment Variables:
    ARAGORA_API_TOKEN: Authentication token (optional)
    ARAGORA_TEST_AGENT: Agent to use for debate tests (default: echo)
"""

import json
import os
import random
import string
import time
from typing import Any, Dict, Optional

from locust import HttpUser, TaskSet, between, task, events
from locust.runners import MasterRunner


# Configuration
API_TOKEN = os.environ.get("ARAGORA_API_TOKEN", "")
TEST_AGENT = os.environ.get("ARAGORA_TEST_AGENT", "echo")


def random_string(length: int = 10) -> str:
    """Generate a random string for test data."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers if token is configured."""
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


# ==============================================================================
# Health Check User - Lightweight probes
# ==============================================================================


class HealthCheckTasks(TaskSet):
    """Tasks for health check probing."""

    @task(10)
    def health_check(self) -> None:
        """Basic health check endpoint."""
        self.client.get("/api/health", name="GET /api/health")

    @task(5)
    def readiness_check(self) -> None:
        """Kubernetes readiness probe."""
        self.client.get("/readyz", name="GET /readyz")

    @task(5)
    def liveness_check(self) -> None:
        """Kubernetes liveness probe."""
        self.client.get("/healthz", name="GET /healthz")

    @task(2)
    def detailed_health(self) -> None:
        """Detailed health check."""
        self.client.get("/api/health/detailed", name="GET /api/health/detailed")


class HealthCheckUser(HttpUser):
    """User that only performs health checks - for baseline load."""

    tasks = [HealthCheckTasks]
    wait_time = between(0.5, 2)
    weight = 3  # 3x more common than other users


# ==============================================================================
# API Browsing User - Typical browsing patterns
# ==============================================================================


class APIBrowsingTasks(TaskSet):
    """Tasks simulating typical API browsing behavior."""

    def on_start(self) -> None:
        """Called when user starts - fetch initial data."""
        self.debate_ids: list[str] = []
        self._fetch_debates()

    def _fetch_debates(self) -> None:
        """Fetch list of debates to browse."""
        with self.client.get(
            "/api/debates",
            name="GET /api/debates",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    debates = data.get("debates", data) if isinstance(data, dict) else data
                    if isinstance(debates, list):
                        self.debate_ids = [d.get("id") for d in debates[:10] if d.get("id")]
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 401:
                response.success()  # Auth required is ok
            else:
                response.failure(f"Status {response.status_code}")

    @task(10)
    def list_debates(self) -> None:
        """List debates with pagination."""
        limit = random.choice([10, 20, 50])
        offset = random.randint(0, 100)
        self.client.get(
            f"/api/debates?limit={limit}&offset={offset}",
            name="GET /api/debates?limit=N",
            headers=get_auth_headers(),
        )

    @task(8)
    def get_debate_detail(self) -> None:
        """Get details of a specific debate."""
        if self.debate_ids:
            debate_id = random.choice(self.debate_ids)
            self.client.get(
                f"/api/debates/{debate_id}",
                name="GET /api/debates/:id",
                headers=get_auth_headers(),
            )

    @task(5)
    def get_leaderboard(self) -> None:
        """Fetch agent leaderboard."""
        self.client.get(
            "/api/leaderboard",
            name="GET /api/leaderboard",
            headers=get_auth_headers(),
        )

    @task(3)
    def get_system_info(self) -> None:
        """Fetch system information."""
        self.client.get(
            "/api/system/info",
            name="GET /api/system/info",
            headers=get_auth_headers(),
        )

    @task(2)
    def get_slo_status(self) -> None:
        """Fetch SLO status."""
        self.client.get(
            "/api/slo/status",
            name="GET /api/slo/status",
            headers=get_auth_headers(),
        )

    @task(3)
    def search_debates(self) -> None:
        """Search for debates."""
        query = random.choice(["AI", "climate", "technology", "ethics", "future"])
        self.client.get(
            f"/api/debates/search?q={query}",
            name="GET /api/debates/search",
            headers=get_auth_headers(),
        )


class APIBrowsingUser(HttpUser):
    """User simulating typical API browsing patterns."""

    tasks = [APIBrowsingTasks]
    wait_time = between(1, 5)
    weight = 5


# ==============================================================================
# Debate User - Creates and monitors debates
# ==============================================================================


class DebateTasks(TaskSet):
    """Tasks for users creating and monitoring debates."""

    def on_start(self) -> None:
        """Initialize user state."""
        self.active_debate_id: Optional[str] = None
        self.completed_debates: list[str] = []

    @task(2)
    def create_debate(self) -> None:
        """Create a new debate."""
        topics = [
            "Should AI systems be required to explain their decisions?",
            "Is remote work better for productivity?",
            "Should social media platforms be regulated?",
            "Is nuclear energy the solution to climate change?",
            "Should autonomous vehicles be allowed on public roads?",
            f"Test topic {random_string(8)}",
        ]

        payload = {
            "task": random.choice(topics),
            "agents": [TEST_AGENT, TEST_AGENT],
            "rounds": random.choice([2, 3, 5]),
            "protocol": {"consensus": "majority"},
        }

        with self.client.post(
            "/api/debate",
            json=payload,
            name="POST /api/debate",
            headers=get_auth_headers(),
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201, 202):
                try:
                    data = response.json()
                    self.active_debate_id = data.get("debate_id") or data.get("id")
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")
            elif response.status_code == 429:
                response.success()  # Rate limited is expected under load
            elif response.status_code == 401:
                response.success()  # Auth required
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def check_debate_status(self) -> None:
        """Check status of active debate."""
        if self.active_debate_id:
            with self.client.get(
                f"/api/debates/{self.active_debate_id}",
                name="GET /api/debates/:id (polling)",
                headers=get_auth_headers(),
                catch_response=True,
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        status = data.get("status")
                        if status in ("completed", "consensus", "no_consensus"):
                            self.completed_debates.append(self.active_debate_id)
                            self.active_debate_id = None
                        response.success()
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON")
                elif response.status_code == 404:
                    self.active_debate_id = None
                    response.success()  # Debate may have been cleaned up
                else:
                    response.failure(f"Status {response.status_code}")

    @task(3)
    def get_debate_messages(self) -> None:
        """Get messages from a completed debate."""
        if self.completed_debates:
            debate_id = random.choice(self.completed_debates)
            self.client.get(
                f"/api/debates/{debate_id}/messages",
                name="GET /api/debates/:id/messages",
                headers=get_auth_headers(),
            )


class DebateUser(HttpUser):
    """User that creates and monitors debates."""

    tasks = [DebateTasks]
    wait_time = between(2, 10)
    weight = 2


# ==============================================================================
# Heavy Load User - Stress testing
# ==============================================================================


class HeavyLoadTasks(TaskSet):
    """Tasks for stress testing with heavy concurrent load."""

    @task(5)
    def rapid_health_checks(self) -> None:
        """Rapid fire health checks."""
        for _ in range(5):
            self.client.get("/api/health", name="GET /api/health (burst)")
            time.sleep(0.1)

    @task(3)
    def bulk_debate_list(self) -> None:
        """Request large debate lists."""
        self.client.get(
            "/api/debates?limit=100",
            name="GET /api/debates?limit=100",
            headers=get_auth_headers(),
        )

    @task(2)
    def concurrent_debate_creation(self) -> None:
        """Attempt to create debates rapidly."""
        payload = {
            "task": f"Stress test debate {random_string(12)}",
            "agents": [TEST_AGENT, TEST_AGENT],
            "rounds": 1,
        }

        with self.client.post(
            "/api/debate",
            json=payload,
            name="POST /api/debate (stress)",
            headers=get_auth_headers(),
            catch_response=True,
        ) as response:
            # Accept 429 as success for rate limiting validation
            if response.status_code in (200, 201, 202, 429, 401):
                response.success()

    @task(1)
    def deep_health_check(self) -> None:
        """Deep health check under load."""
        self.client.get(
            "/api/health/deep",
            name="GET /api/health/deep",
            headers=get_auth_headers(),
        )


class HeavyLoadUser(HttpUser):
    """User for stress testing scenarios."""

    tasks = [HeavyLoadTasks]
    wait_time = between(0.1, 1)
    weight = 1


# ==============================================================================
# WebSocket User - Real-time connections
# ==============================================================================


class WebSocketTasks(TaskSet):
    """Tasks for WebSocket connection testing.

    Note: Locust doesn't natively support WebSockets well.
    This simulates the HTTP handshake and fallback patterns.
    For true WebSocket load testing, consider using a specialized tool.
    """

    @task(1)
    def websocket_upgrade_attempt(self) -> None:
        """Attempt WebSocket upgrade (will fail but tests the upgrade path)."""
        headers = {
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
            "Sec-WebSocket-Version": "13",
        }

        # This will typically fail with 400/426, which is expected
        with self.client.get(
            "/ws",
            headers=headers,
            name="GET /ws (upgrade attempt)",
            catch_response=True,
        ) as response:
            # Accept any response as this is just testing the upgrade path
            response.success()


class WebSocketUser(HttpUser):
    """User simulating WebSocket connection patterns."""

    tasks = [WebSocketTasks]
    wait_time = between(5, 15)
    weight = 1


# ==============================================================================
# Event Hooks for Metrics
# ==============================================================================


@events.init.add_listener
def on_locust_init(environment, **kwargs) -> None:
    """Initialize custom metrics on Locust startup."""
    if isinstance(environment.runner, MasterRunner):
        print("Locust master node starting")


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response: Any,
    context: Any,
    exception: Any,
    **kwargs,
) -> None:
    """Track additional metrics per request."""
    # Log slow requests
    if response_time > 5000:  # 5 seconds
        print(f"SLOW REQUEST: {name} took {response_time:.0f}ms")

    # Track rate limiting
    if response and hasattr(response, "status_code") and response.status_code == 429:
        print(f"RATE LIMITED: {name}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Called when load test starts."""
    print("=" * 60)
    print("Aragora Load Test Starting")
    print(f"Target: {environment.host}")
    print(f"Auth configured: {bool(API_TOKEN)}")
    print(f"Test agent: {TEST_AGENT}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Called when load test stops."""
    print("=" * 60)
    print("Aragora Load Test Complete")
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Avg response time: {stats.total.avg_response_time:.0f}ms")
    print(f"p99 response time: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print("=" * 60)
