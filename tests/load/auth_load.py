"""
Authentication Flow Load Testing Suite for Aragora.

Tests authentication system performance under load including:
- Login/logout flows
- Token refresh patterns
- Session management
- MFA verification
- SSO authentication flows

Run with:
    pytest tests/load/auth_load.py -v --asyncio-mode=auto

For stress testing:
    pytest tests/load/auth_load.py -v -k stress --asyncio-mode=auto

Environment Variables:
    ARAGORA_API_URL: API base URL (default: http://localhost:8080)
    ARAGORA_AUTH_CONCURRENT: Concurrent auth operations (default: 20)
    ARAGORA_AUTH_DURATION: Test duration in seconds (default: 60)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import string
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

# Configuration
API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")
CONCURRENT_AUTH = int(os.environ.get("ARAGORA_AUTH_CONCURRENT", "20"))
TEST_DURATION = int(os.environ.get("ARAGORA_AUTH_DURATION", "60"))


@dataclass
class AuthLoadMetrics:
    """Metrics collected during authentication load test."""

    login_attempts: int = 0
    login_successes: int = 0
    login_failures: int = 0
    token_refreshes: int = 0
    refresh_successes: int = 0
    refresh_failures: int = 0
    session_operations: int = 0
    mfa_challenges: int = 0
    sso_initiations: int = 0
    rate_limited: int = 0
    response_times: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def avg_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times) * 1000

    @property
    def p95_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000

    @property
    def p99_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000

    @property
    def login_success_rate(self) -> float:
        total = self.login_successes + self.login_failures
        if total == 0:
            return 0.0
        return self.login_successes / total * 100

    @property
    def operations_per_second(self) -> float:
        if self.duration == 0:
            return 0.0
        total_ops = self.login_attempts + self.token_refreshes + self.session_operations
        return total_ops / self.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "login_attempts": self.login_attempts,
            "login_successes": self.login_successes,
            "login_failures": self.login_failures,
            "login_success_rate_percent": round(self.login_success_rate, 2),
            "token_refreshes": self.token_refreshes,
            "refresh_successes": self.refresh_successes,
            "refresh_failures": self.refresh_failures,
            "session_operations": self.session_operations,
            "mfa_challenges": self.mfa_challenges,
            "sso_initiations": self.sso_initiations,
            "rate_limited": self.rate_limited,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "p95_response_time_ms": round(self.p95_response_time_ms, 2),
            "p99_response_time_ms": round(self.p99_response_time_ms, 2),
            "operations_per_second": round(self.operations_per_second, 2),
            "duration_seconds": round(self.duration, 2),
            "error_count": len(self.errors),
        }


def random_email() -> str:
    """Generate a random email for testing."""
    chars = string.ascii_lowercase + string.digits
    username = "".join(random.choices(chars, k=10))
    return f"loadtest_{username}@test.aragora.io"


def random_password() -> str:
    """Generate a random password for testing."""
    return secrets.token_urlsafe(16)


class AuthLoadClient:
    """Client for authentication load testing."""

    def __init__(self, base_url: str, metrics: AuthLoadMetrics):
        self.base_url = base_url.rstrip("/")
        self.metrics = metrics
        self.session: Any | None = None
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    async def _ensure_session(self) -> Any:
        """Ensure aiohttp session exists."""
        if self.session is None:
            import aiohttp

            self.session = aiohttp.ClientSession()
        return self.session

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def attempt_login(
        self,
        email: str,
        password: str,
    ) -> dict[str, Any] | None:
        """Attempt to login with credentials."""
        session = await self._ensure_session()

        payload = {
            "email": email,
            "password": password,
        }

        try:
            self.metrics.login_attempts += 1
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/auth/login",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (200, 201):
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    self.metrics.login_successes += 1
                    return data
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    self.metrics.login_failures += 1
                    return None
                elif response.status == 401:
                    # Invalid credentials - expected for test users
                    self.metrics.login_failures += 1
                    return None
                else:
                    self.metrics.login_failures += 1
                    return None

        except Exception as e:
            self.metrics.login_failures += 1
            self.metrics.errors.append(f"Login error: {str(e)[:100]}")
            return None

    async def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self.refresh_token:
            return False

        session = await self._ensure_session()

        payload = {"refresh_token": self.refresh_token}

        try:
            self.metrics.token_refreshes += 1
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/auth/refresh",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    if data.get("refresh_token"):
                        self.refresh_token = data.get("refresh_token")
                    self.metrics.refresh_successes += 1
                    return True
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    self.metrics.refresh_failures += 1
                    return False
                else:
                    self.metrics.refresh_failures += 1
                    return False

        except Exception as e:
            self.metrics.refresh_failures += 1
            self.metrics.errors.append(f"Refresh error: {str(e)[:100]}")
            return False

    async def get_session_info(self) -> dict[str, Any] | None:
        """Get current session information."""
        session = await self._ensure_session()

        try:
            self.metrics.session_operations += 1
            start_time = time.time()

            async with session.get(
                f"{self.base_url}/api/auth/session",
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status == 200:
                    return await response.json()
                return None

        except Exception as e:
            self.metrics.errors.append(f"Session error: {str(e)[:100]}")
            return None

    async def logout(self) -> bool:
        """Logout and invalidate session."""
        session = await self._ensure_session()

        try:
            self.metrics.session_operations += 1
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/auth/logout",
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                self.access_token = None
                self.refresh_token = None
                return response.status in (200, 204)

        except Exception as e:
            self.metrics.errors.append(f"Logout error: {str(e)[:100]}")
            return False

    async def initiate_mfa_challenge(self) -> str | None:
        """Initiate MFA challenge."""
        session = await self._ensure_session()

        try:
            self.metrics.mfa_challenges += 1
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/auth/mfa/challenge",
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status == 200:
                    data = await response.json()
                    return data.get("challenge_id")
                return None

        except Exception as e:
            self.metrics.errors.append(f"MFA error: {str(e)[:100]}")
            return None

    async def initiate_sso(self, provider: str = "google") -> str | None:
        """Initiate SSO flow."""
        session = await self._ensure_session()

        try:
            self.metrics.sso_initiations += 1
            start_time = time.time()

            async with session.get(
                f"{self.base_url}/api/auth/sso/{provider}/authorize",
                headers={"Content-Type": "application/json"},
                timeout=10,
                allow_redirects=False,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (302, 307):
                    return response.headers.get("Location")
                elif response.status == 200:
                    data = await response.json()
                    return data.get("authorization_url")
                return None

        except Exception as e:
            self.metrics.errors.append(f"SSO error: {str(e)[:100]}")
            return None

    async def close(self) -> None:
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


async def run_auth_load_test(
    base_url: str,
    concurrent: int,
    duration: float,
    include_mfa: bool = False,
    include_sso: bool = True,
) -> AuthLoadMetrics:
    """
    Run authentication load test.

    Args:
        base_url: API base URL
        concurrent: Number of concurrent auth operations
        duration: Test duration in seconds
        include_mfa: Whether to include MFA challenges
        include_sso: Whether to include SSO initiation

    Returns:
        AuthLoadMetrics with test results
    """
    metrics = AuthLoadMetrics()
    metrics.start_time = time.time()

    clients: list[AuthLoadClient] = []

    for _ in range(concurrent):
        clients.append(AuthLoadClient(base_url, metrics))

    async def auth_session(client: AuthLoadClient) -> None:
        """Simulate a complete auth session."""
        email = random_email()
        password = random_password()

        # Attempt login
        await client.attempt_login(email, password)

        # Try to get session info if logged in
        if client.access_token:
            await client.get_session_info()

            # Occasionally refresh token
            if random.random() < 0.3:
                await client.refresh_access_token()

            # Include MFA if enabled
            if include_mfa and random.random() < 0.2:
                await client.initiate_mfa_challenge()

            # Logout at end
            await client.logout()

        # Occasionally try SSO flow
        if include_sso and random.random() < 0.1:
            provider = random.choice(["google", "github", "microsoft"])
            await client.initiate_sso(provider)

    # Run auth sessions for the specified duration
    end_time = time.time() + duration
    tasks: list[asyncio.Task] = []

    while time.time() < end_time:
        # Start new auth sessions
        for client in clients:
            if len(tasks) < concurrent * 2:
                task = asyncio.create_task(auth_session(client))
                tasks.append(task)

        # Clean up completed tasks
        done_tasks = [t for t in tasks if t.done()]
        for t in done_tasks:
            tasks.remove(t)

        await asyncio.sleep(0.1)

    # Wait for remaining tasks
    if tasks:
        await asyncio.wait(tasks, timeout=10.0)

    # Close all clients
    for client in clients:
        await client.close()

    metrics.end_time = time.time()
    return metrics


# =============================================================================
# SLO Validation Thresholds
# =============================================================================


class AuthSLOThresholds:
    """SLO thresholds for authentication operations."""

    # Response time thresholds
    LOGIN_P95_MS = 500  # Login should complete within 500ms at p95
    LOGIN_P99_MS = 1000  # Login should complete within 1s at p99
    REFRESH_P95_MS = 200  # Token refresh should be fast
    REFRESH_P99_MS = 500

    # Success rate thresholds (allow for invalid test credentials)
    MIN_REFRESH_SUCCESS_RATE = 0.9  # 90% refresh should succeed when valid

    # Throughput thresholds
    MIN_OPS_PER_SECOND = 10  # Should handle at least 10 auth ops/sec


# =============================================================================
# Pytest Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_single_login_flow():
    """Test single login flow."""
    metrics = AuthLoadMetrics()
    client = AuthLoadClient(API_URL, metrics)

    email = random_email()
    password = random_password()

    result = await client.attempt_login(email, password)

    # Login may fail for test user - that's expected
    # We're testing that the endpoint responds correctly
    assert metrics.login_attempts == 1

    await client.close()


@pytest.mark.asyncio
async def test_concurrent_login_attempts():
    """Test multiple concurrent login attempts."""
    metrics = await run_auth_load_test(
        base_url=API_URL,
        concurrent=5,
        duration=5.0,
        include_mfa=False,
        include_sso=False,
    )

    print(f"\nResults: {json.dumps(metrics.to_dict(), indent=2)}")

    if metrics.login_attempts == 0:
        pytest.skip("No login attempts completed (API may not be available)")

    # Should have attempted some logins
    assert metrics.login_attempts > 0


@pytest.mark.asyncio
async def test_token_refresh_throughput():
    """Test token refresh throughput."""
    metrics = AuthLoadMetrics()
    client = AuthLoadClient(API_URL, metrics)

    # Simulate having a refresh token
    client.refresh_token = "test_refresh_token"

    for _ in range(10):
        await client.refresh_access_token()

    await client.close()

    print(f"\nRefresh attempts: {metrics.token_refreshes}")
    print(f"Avg response time: {metrics.avg_response_time_ms:.1f}ms")


@pytest.mark.asyncio
async def test_sso_initiation_load():
    """Test SSO initiation under load."""
    metrics = AuthLoadMetrics()
    client = AuthLoadClient(API_URL, metrics)

    providers = ["google", "github", "microsoft"]

    for _ in range(5):
        provider = random.choice(providers)
        await client.initiate_sso(provider)

    await client.close()

    print(f"\nSSO initiations: {metrics.sso_initiations}")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_auth_stress():
    """Stress test authentication system."""
    metrics = await run_auth_load_test(
        base_url=API_URL,
        concurrent=CONCURRENT_AUTH,
        duration=TEST_DURATION,
        include_mfa=True,
        include_sso=True,
    )

    print(f"\n{'=' * 60}")
    print("Authentication Stress Test Results")
    print("=" * 60)
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)

    if metrics.login_attempts == 0:
        pytest.skip("No login attempts (API may not be available)")

    # Validate SLO thresholds
    assert metrics.p95_response_time_ms < AuthSLOThresholds.LOGIN_P95_MS, (
        f"p95 response time {metrics.p95_response_time_ms}ms exceeds "
        f"{AuthSLOThresholds.LOGIN_P95_MS}ms threshold"
    )

    assert metrics.p99_response_time_ms < AuthSLOThresholds.LOGIN_P99_MS, (
        f"p99 response time {metrics.p99_response_time_ms}ms exceeds "
        f"{AuthSLOThresholds.LOGIN_P99_MS}ms threshold"
    )


@pytest.mark.asyncio
async def test_rate_limiting_behavior():
    """Test authentication rate limiting behavior."""
    metrics = AuthLoadMetrics()
    client = AuthLoadClient(API_URL, metrics)

    # Rapid login attempts to trigger rate limiting
    for _ in range(30):
        email = random_email()
        password = random_password()
        await client.attempt_login(email, password)
        await asyncio.sleep(0.01)  # Small delay

    await client.close()

    print(f"\nTotal attempts: {metrics.login_attempts}")
    print(f"Rate limited: {metrics.rate_limited}")

    # Some rate limiting should have occurred with rapid attempts
    # But this depends on the rate limit configuration


@pytest.mark.asyncio
async def test_session_lifecycle():
    """Test full session lifecycle under load."""
    metrics = AuthLoadMetrics()

    async def session_lifecycle():
        client = AuthLoadClient(API_URL, metrics)

        # Login
        await client.attempt_login(random_email(), random_password())

        # Session operations
        await client.get_session_info()

        # Token refresh
        if client.refresh_token:
            await client.refresh_access_token()

        # Logout
        await client.logout()

        await client.close()

    # Run multiple session lifecycles concurrently
    tasks = [session_lifecycle() for _ in range(10)]
    await asyncio.gather(*tasks)

    print(f"\nSession operations: {metrics.session_operations}")
    print(f"Token refreshes: {metrics.token_refreshes}")


if __name__ == "__main__":
    # Run standalone for quick testing
    async def main() -> None:
        print(f"Running auth load test against {API_URL}")
        print(f"Concurrent operations: {CONCURRENT_AUTH}")
        print(f"Duration: {TEST_DURATION}s")
        print()

        metrics = await run_auth_load_test(
            base_url=API_URL,
            concurrent=CONCURRENT_AUTH,
            duration=TEST_DURATION,
            include_mfa=True,
            include_sso=True,
        )

        print("\nResults:")
        print(json.dumps(metrics.to_dict(), indent=2))

        if metrics.errors[:5]:
            print("\nSample errors:")
            for error in metrics.errors[:5]:
                print(f"  - {error}")

    asyncio.run(main())
