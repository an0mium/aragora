"""
Gauntlet Load Testing Suite for Aragora.

Tests the adversarial stress-testing system under load.

Run with:
    pytest tests/load/gauntlet_load.py -v --asyncio-mode=auto

For full stress test:
    pytest tests/load/gauntlet_load.py -v -k stress --asyncio-mode=auto -s

Environment Variables:
    ARAGORA_API_URL: API base URL (default: http://localhost:8080)
    ARAGORA_API_TOKEN: Authentication token (optional)
    ARAGORA_GAUNTLET_CONCURRENT: Concurrent runs (default: 5)
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
API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")
API_TOKEN = os.environ.get("ARAGORA_API_TOKEN", "")
CONCURRENT_RUNS = int(os.environ.get("ARAGORA_GAUNTLET_CONCURRENT", "5"))


@dataclass
class GauntletMetrics:
    """Metrics for Gauntlet load testing."""

    runs_started: int = 0
    runs_completed: int = 0
    runs_failed: int = 0
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    api_requests: int = 0
    api_errors: int = 0
    response_times: List[float] = field(default_factory=list)
    verdicts: Dict[str, int] = field(default_factory=dict)
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
    def success_rate(self) -> float:
        total = self.runs_completed + self.runs_failed
        if total == 0:
            return 0.0
        return self.runs_completed / total * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs_started": self.runs_started,
            "runs_completed": self.runs_completed,
            "runs_failed": self.runs_failed,
            "success_rate_percent": round(self.success_rate, 1),
            "total_findings": self.total_findings,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "api_requests": self.api_requests,
            "api_errors": self.api_errors,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "verdicts": self.verdicts,
            "duration_seconds": round(self.duration, 2),
        }


def random_string(length: int = 10) -> str:
    """Generate random string."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def get_sample_specs() -> List[str]:
    """Get sample specifications for testing."""
    return [
        """
# User Authentication System

## Overview
Implement a secure user authentication system.

## Requirements
1. Users can register with email and password
2. Passwords must be at least 8 characters
3. Support password reset via email
4. Implement session management with JWT tokens

## Security
- Hash passwords using bcrypt
- Rate limit login attempts
- Implement CSRF protection
""",
        """
# API Rate Limiting

## Overview
Design a rate limiting system for the API.

## Requirements
1. Limit requests per IP address
2. Support tiered limits for authenticated users
3. Return appropriate 429 responses
4. Track usage metrics

## Implementation
- Use sliding window algorithm
- Store counters in Redis
- Support burst allowance
""",
        """
# Data Export Feature

## Overview
Allow users to export their data in multiple formats.

## Requirements
1. Support JSON, CSV, and PDF exports
2. Queue large exports for background processing
3. Notify users when exports are ready
4. Expire download links after 24 hours

## Privacy
- Only export user's own data
- Log all export requests
- Sanitize sensitive fields
""",
        """
# Webhook System

## Overview
Implement a webhook notification system.

## Requirements
1. Users can register webhook URLs
2. Support multiple event types
3. Implement retry logic for failures
4. Sign payloads with HMAC

## Reliability
- Queue webhook deliveries
- Track delivery status
- Support manual re-delivery
""",
        """
# Search Functionality

## Overview
Implement full-text search across content.

## Requirements
1. Support keyword search
2. Implement filters and facets
3. Rank results by relevance
4. Support pagination

## Performance
- Use Elasticsearch or similar
- Cache popular queries
- Implement query suggestions
""",
    ]


class GauntletClient:
    """Client for Gauntlet API testing."""

    def __init__(self, base_url: str, metrics: GauntletMetrics, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.metrics = metrics
        self.token = token
        self.session: Optional[Any] = None

    async def _ensure_session(self) -> Any:
        """Ensure aiohttp session exists."""
        if self.session is None:
            import aiohttp

            self.session = aiohttp.ClientSession()
        return self.session

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def start_gauntlet(
        self,
        spec: str,
        profile: str = "quick",
        input_type: str = "spec",
    ) -> Optional[str]:
        """Start a Gauntlet run."""
        session = await self._ensure_session()

        payload = {
            "input_content": spec,
            "input_type": input_type,
            "profile": profile,
        }

        try:
            start_time = time.time()
            self.metrics.api_requests += 1

            async with session.post(
                f"{self.base_url}/api/gauntlet/run",
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (200, 201, 202):
                    data = await response.json()
                    self.metrics.runs_started += 1
                    return data.get("id") or data.get("gauntlet_id")
                elif response.status == 429:
                    # Rate limited - not a failure
                    return None
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception as e:
            self.metrics.api_errors += 1
            return None

    async def get_gauntlet_status(self, gauntlet_id: str) -> Optional[Dict[str, Any]]:
        """Get Gauntlet run status."""
        session = await self._ensure_session()

        try:
            self.metrics.api_requests += 1

            async with session.get(
                f"{self.base_url}/api/gauntlet/{gauntlet_id}",
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception:
            self.metrics.api_errors += 1
            return None

    async def wait_for_completion(
        self,
        gauntlet_id: str,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        """Wait for Gauntlet run to complete."""
        end_time = time.time() + timeout

        while time.time() < end_time:
            status = await self.get_gauntlet_status(gauntlet_id)

            if status is None:
                await asyncio.sleep(poll_interval)
                continue

            run_status = status.get("status", "")

            if run_status in ("completed", "complete"):
                self.metrics.runs_completed += 1

                # Extract findings
                findings = status.get("findings", [])
                self.metrics.total_findings += len(findings)

                for finding in findings:
                    severity = finding.get("severity", "").upper()
                    if severity == "CRITICAL":
                        self.metrics.critical_findings += 1
                    elif severity == "HIGH":
                        self.metrics.high_findings += 1

                # Track verdict
                verdict = status.get("verdict", {})
                verdict_value = verdict.get("verdict", "UNKNOWN")
                self.metrics.verdicts[verdict_value] = (
                    self.metrics.verdicts.get(verdict_value, 0) + 1
                )

                return status

            elif run_status in ("failed", "error"):
                self.metrics.runs_failed += 1
                return status

            await asyncio.sleep(poll_interval)

        # Timeout
        self.metrics.runs_failed += 1
        return None

    async def close(self) -> None:
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


async def run_gauntlet_load_test(
    base_url: str,
    token: str,
    concurrent: int,
    wait_for_completion: bool = True,
) -> GauntletMetrics:
    """
    Run Gauntlet load test.

    Args:
        base_url: API base URL
        token: Authentication token
        concurrent: Number of concurrent runs
        wait_for_completion: Whether to wait for runs to complete

    Returns:
        GauntletMetrics with test results
    """
    metrics = GauntletMetrics()
    metrics.start_time = time.time()

    client = GauntletClient(base_url, metrics, token)
    specs = get_sample_specs()

    async def run_single_gauntlet(spec: str) -> None:
        """Run a single Gauntlet test."""
        gauntlet_id = await client.start_gauntlet(
            spec=spec,
            profile="quick",
            input_type="spec",
        )

        if gauntlet_id and wait_for_completion:
            await client.wait_for_completion(gauntlet_id, timeout=180.0)

    # Run concurrent tests
    tasks = [run_single_gauntlet(random.choice(specs)) for _ in range(concurrent)]

    await asyncio.gather(*tasks, return_exceptions=True)
    await client.close()

    metrics.end_time = time.time()
    return metrics


# =============================================================================
# Pytest Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_single_gauntlet_run():
    """Test single Gauntlet run."""
    metrics = GauntletMetrics()
    client = GauntletClient(API_URL, metrics, API_TOKEN)

    specs = get_sample_specs()
    gauntlet_id = await client.start_gauntlet(
        spec=specs[0],
        profile="quick",
        input_type="spec",
    )

    if gauntlet_id is None:
        await client.close()
        pytest.skip("Could not start Gauntlet run (API may not be available)")

    print(f"Started Gauntlet: {gauntlet_id}")

    result = await client.wait_for_completion(gauntlet_id, timeout=120.0)
    await client.close()

    assert result is not None, "Gauntlet run timed out"
    print(f"Verdict: {result.get('verdict')}")
    print(f"Findings: {len(result.get('findings', []))}")


@pytest.mark.asyncio
async def test_concurrent_gauntlet_runs():
    """Test multiple concurrent Gauntlet runs."""
    metrics = await run_gauntlet_load_test(
        base_url=API_URL,
        token=API_TOKEN,
        concurrent=3,
        wait_for_completion=False,  # Don't wait to speed up test
    )

    print(f"\nResults: {json.dumps(metrics.to_dict(), indent=2)}")

    if metrics.runs_started == 0:
        pytest.skip("No runs started (API may not be available)")

    # Should be able to start at least some runs
    assert metrics.runs_started >= 1


@pytest.mark.asyncio
async def test_gauntlet_api_throughput():
    """Test Gauntlet API response times."""
    metrics = GauntletMetrics()
    client = GauntletClient(API_URL, metrics, API_TOKEN)

    # Make several quick requests
    specs = get_sample_specs()

    for _ in range(5):
        await client.start_gauntlet(
            spec=random.choice(specs),
            profile="quick",
            input_type="spec",
        )

    await client.close()

    print(f"API requests: {metrics.api_requests}")
    print(f"Avg response time: {metrics.avg_response_time_ms:.1f}ms")

    if metrics.api_requests == 0:
        pytest.skip("No API requests completed")

    # Response times should be reasonable (under 5 seconds)
    assert metrics.avg_response_time_ms < 5000, "API responses too slow"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_gauntlet_stress():
    """Stress test Gauntlet with many concurrent runs."""
    metrics = await run_gauntlet_load_test(
        base_url=API_URL,
        token=API_TOKEN,
        concurrent=CONCURRENT_RUNS,
        wait_for_completion=True,
    )

    print(f"\n{'=' * 60}")
    print("Gauntlet Stress Test Results")
    print("=" * 60)
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)

    if metrics.runs_started == 0:
        pytest.skip("No runs started")

    # At least 50% of runs should succeed
    assert metrics.success_rate >= 50, f"Success rate below 50%: {metrics.success_rate}%"


@pytest.mark.asyncio
async def test_gauntlet_different_input_types():
    """Test Gauntlet with different input types."""
    metrics = GauntletMetrics()
    client = GauntletClient(API_URL, metrics, API_TOKEN)

    test_inputs = [
        ("spec", "# Feature Spec\n\nImplement user login."),
        ("architecture", "# Architecture\n\nMicroservices design."),
        ("policy", "# Policy\n\nData retention policy."),
    ]

    for input_type, content in test_inputs:
        gauntlet_id = await client.start_gauntlet(
            spec=content,
            profile="quick",
            input_type=input_type,
        )
        if gauntlet_id:
            print(f"Started {input_type}: {gauntlet_id}")

    await client.close()

    print(f"\nStarted {metrics.runs_started} runs")


@pytest.mark.asyncio
async def test_gauntlet_export_endpoint():
    """Test Gauntlet export endpoint under load."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        headers = {"Content-Type": "application/json"}
        if API_TOKEN:
            headers["Authorization"] = f"Bearer {API_TOKEN}"

        # Get list of recent results
        async with session.get(
            f"{API_URL}/api/gauntlet/results?limit=5",
            headers=headers,
            timeout=10,
        ) as response:
            if response.status != 200:
                pytest.skip("Could not fetch Gauntlet results")

            data = await response.json()
            results = data.get("results", [])

            if not results:
                pytest.skip("No Gauntlet results available")

            # Test export for first result
            gauntlet_id = results[0].get("id")

            async with session.get(
                f"{API_URL}/api/gauntlet/{gauntlet_id}/export?format=json",
                headers=headers,
                timeout=10,
            ) as export_response:
                assert export_response.status in (
                    200,
                    404,
                ), f"Export failed: {export_response.status}"

                if export_response.status == 200:
                    export_data = await export_response.json()
                    print(f"Export keys: {list(export_data.keys())}")


if __name__ == "__main__":
    # Run standalone
    import sys

    async def main() -> None:
        print(f"Running Gauntlet load test against {API_URL}")
        print(f"Concurrent runs: {CONCURRENT_RUNS}")
        print()

        metrics = await run_gauntlet_load_test(
            base_url=API_URL,
            token=API_TOKEN,
            concurrent=CONCURRENT_RUNS,
            wait_for_completion=True,
        )

        print("\nResults:")
        print(json.dumps(metrics.to_dict(), indent=2))

    asyncio.run(main())
