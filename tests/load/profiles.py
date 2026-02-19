"""
Load Test Configuration Profiles for Aragora.

Provides standardized load test profiles for different testing scenarios:
- Smoke: Quick validation with minimal load
- Light: Normal operational load
- Medium: Peak traffic simulation
- Heavy: Stress testing
- Spike: Sudden traffic burst
- Soak: Extended duration testing

Usage:
    from tests.load.profiles import get_profile, LoadProfile, SLOThresholds

    profile = get_profile("medium")
    print(f"Users: {profile.users}, Duration: {profile.duration_seconds}s")

Environment Variables:
    ARAGORA_LOAD_PROFILE: Override default profile (default: light)
    ARAGORA_BASE_URL: API base URL (default: http://localhost:8080)
    ARAGORA_API_TOKEN: Authentication token (optional)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProfileType(str, Enum):
    """Load test profile types."""

    SMOKE = "smoke"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    SPIKE = "spike"
    SOAK = "soak"


@dataclass
class RampingStage:
    """Configuration for a ramping stage."""

    duration_seconds: int
    target_users: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "target_users": self.target_users,
        }


@dataclass
class SLOThresholds:
    """SLO thresholds for load testing validation."""

    # Response time thresholds (milliseconds)
    http_p50_ms: int = 200
    http_p95_ms: int = 500
    http_p99_ms: int = 1000

    # Debate-specific thresholds
    debate_create_p95_ms: int = 2000
    debate_poll_p95_ms: int = 500

    # Search thresholds
    search_p95_ms: int = 1000
    search_p99_ms: int = 2000

    # Auth thresholds
    auth_p95_ms: int = 500
    auth_p99_ms: int = 1000

    # Pipeline thresholds (Idea-to-Execution)
    pipeline_create_p95_ms: int = 3000
    pipeline_status_p95_ms: int = 500
    pipeline_graph_p95_ms: int = 1000

    # Error rate thresholds (as decimals, e.g., 0.01 = 1%)
    max_error_rate: float = 0.01
    max_timeout_rate: float = 0.005

    # Throughput thresholds (requests per second)
    min_throughput_rps: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "http_p50_ms": self.http_p50_ms,
            "http_p95_ms": self.http_p95_ms,
            "http_p99_ms": self.http_p99_ms,
            "debate_create_p95_ms": self.debate_create_p95_ms,
            "debate_poll_p95_ms": self.debate_poll_p95_ms,
            "search_p95_ms": self.search_p95_ms,
            "search_p99_ms": self.search_p99_ms,
            "auth_p95_ms": self.auth_p95_ms,
            "auth_p99_ms": self.auth_p99_ms,
            "max_error_rate": self.max_error_rate,
            "max_timeout_rate": self.max_timeout_rate,
            "min_throughput_rps": self.min_throughput_rps,
            "pipeline_create_p95_ms": self.pipeline_create_p95_ms,
            "pipeline_status_p95_ms": self.pipeline_status_p95_ms,
            "pipeline_graph_p95_ms": self.pipeline_graph_p95_ms,
        }


@dataclass
class LoadProfile:
    """Configuration for a load test profile."""

    name: str
    description: str

    # User/connection configuration
    users: int
    spawn_rate: float  # Users per second

    # Duration configuration
    duration_seconds: int
    warmup_seconds: int = 30
    cooldown_seconds: int = 30

    # Ramping stages (optional, for complex load patterns)
    stages: list[RampingStage] = field(default_factory=list)

    # Feature flags
    include_websockets: bool = True
    include_debates: bool = True
    include_auth: bool = True
    include_knowledge: bool = True
    include_gauntlet: bool = False  # Expensive, opt-in

    # Concurrency limits
    max_concurrent_debates: int = 10
    max_concurrent_websockets: int = 50
    max_concurrent_searches: int = 20

    # SLO thresholds
    slo_thresholds: SLOThresholds = field(default_factory=SLOThresholds)

    # Wait times (seconds between requests for a user)
    min_wait: float = 1.0
    max_wait: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "users": self.users,
            "spawn_rate": self.spawn_rate,
            "duration_seconds": self.duration_seconds,
            "warmup_seconds": self.warmup_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "stages": [s.to_dict() for s in self.stages],
            "include_websockets": self.include_websockets,
            "include_debates": self.include_debates,
            "include_auth": self.include_auth,
            "include_knowledge": self.include_knowledge,
            "include_gauntlet": self.include_gauntlet,
            "max_concurrent_debates": self.max_concurrent_debates,
            "max_concurrent_websockets": self.max_concurrent_websockets,
            "max_concurrent_searches": self.max_concurrent_searches,
            "slo_thresholds": self.slo_thresholds.to_dict(),
            "min_wait": self.min_wait,
            "max_wait": self.max_wait,
        }

    def to_locust_options(self) -> dict[str, Any]:
        """Convert to Locust command-line options format."""
        options = {
            "users": self.users,
            "spawn_rate": self.spawn_rate,
            "run_time": f"{self.duration_seconds}s",
        }
        return options

    def to_k6_options(self) -> dict[str, Any]:
        """Convert to k6 options format."""
        if self.stages:
            return {
                "scenarios": {
                    self.name: {
                        "executor": "ramping-vus",
                        "startVUs": 0,
                        "stages": [
                            {
                                "duration": f"{s.duration_seconds}s",
                                "target": s.target_users,
                            }
                            for s in self.stages
                        ],
                    }
                },
                "thresholds": {
                    "http_req_duration": [
                        f"p(95)<{self.slo_thresholds.http_p95_ms}",
                        f"p(99)<{self.slo_thresholds.http_p99_ms}",
                    ],
                    "http_req_failed": [f"rate<{self.slo_thresholds.max_error_rate}"],
                },
            }
        else:
            return {
                "vus": self.users,
                "duration": f"{self.duration_seconds}s",
                "thresholds": {
                    "http_req_duration": [
                        f"p(95)<{self.slo_thresholds.http_p95_ms}",
                        f"p(99)<{self.slo_thresholds.http_p99_ms}",
                    ],
                    "http_req_failed": [f"rate<{self.slo_thresholds.max_error_rate}"],
                },
            }


# =============================================================================
# Predefined Profiles
# =============================================================================


SMOKE_PROFILE = LoadProfile(
    name="smoke",
    description="Quick smoke test with minimal load for CI/CD validation",
    users=5,
    spawn_rate=1.0,
    duration_seconds=60,
    warmup_seconds=10,
    cooldown_seconds=10,
    include_websockets=True,
    include_debates=True,
    include_auth=True,
    include_knowledge=True,
    include_gauntlet=False,
    max_concurrent_debates=3,
    max_concurrent_websockets=10,
    max_concurrent_searches=5,
    min_wait=0.5,
    max_wait=2.0,
    slo_thresholds=SLOThresholds(
        http_p95_ms=1000,  # More relaxed for smoke tests
        http_p99_ms=2000,
        max_error_rate=0.05,  # Allow higher error rate in smoke
    ),
)


LIGHT_PROFILE = LoadProfile(
    name="light",
    description="Normal operational load simulating typical daily traffic",
    users=20,
    spawn_rate=2.0,
    duration_seconds=300,  # 5 minutes
    warmup_seconds=30,
    cooldown_seconds=30,
    include_websockets=True,
    include_debates=True,
    include_auth=True,
    include_knowledge=True,
    include_gauntlet=False,
    max_concurrent_debates=5,
    max_concurrent_websockets=30,
    max_concurrent_searches=10,
    min_wait=1.0,
    max_wait=5.0,
    slo_thresholds=SLOThresholds(
        http_p95_ms=500,
        http_p99_ms=1000,
        max_error_rate=0.01,
    ),
)


MEDIUM_PROFILE = LoadProfile(
    name="medium",
    description="Peak traffic simulation for capacity planning",
    users=50,
    spawn_rate=5.0,
    duration_seconds=600,  # 10 minutes
    warmup_seconds=60,
    cooldown_seconds=60,
    stages=[
        RampingStage(60, 10),  # Ramp to 10
        RampingStage(120, 30),  # Ramp to 30
        RampingStage(180, 50),  # Ramp to 50 (peak)
        RampingStage(180, 50),  # Stay at peak
        RampingStage(60, 20),  # Ramp down
    ],
    include_websockets=True,
    include_debates=True,
    include_auth=True,
    include_knowledge=True,
    include_gauntlet=True,
    max_concurrent_debates=10,
    max_concurrent_websockets=50,
    max_concurrent_searches=20,
    min_wait=0.5,
    max_wait=3.0,
    slo_thresholds=SLOThresholds(
        http_p95_ms=500,
        http_p99_ms=1500,
        max_error_rate=0.02,
        min_throughput_rps=20.0,
    ),
)


HEAVY_PROFILE = LoadProfile(
    name="heavy",
    description="Stress testing to find system limits",
    users=100,
    spawn_rate=10.0,
    duration_seconds=900,  # 15 minutes
    warmup_seconds=120,
    cooldown_seconds=120,
    stages=[
        RampingStage(60, 20),  # Warm up
        RampingStage(120, 50),  # Ramp to 50
        RampingStage(180, 100),  # Ramp to 100 (stress)
        RampingStage(300, 100),  # Sustain stress
        RampingStage(120, 50),  # Ramp down
        RampingStage(60, 0),  # Cool down
    ],
    include_websockets=True,
    include_debates=True,
    include_auth=True,
    include_knowledge=True,
    include_gauntlet=True,
    max_concurrent_debates=20,
    max_concurrent_websockets=100,
    max_concurrent_searches=50,
    min_wait=0.1,
    max_wait=1.0,
    slo_thresholds=SLOThresholds(
        http_p95_ms=1000,  # More relaxed under heavy load
        http_p99_ms=3000,
        max_error_rate=0.05,  # Allow higher error rate under stress
        min_throughput_rps=50.0,
    ),
)


SPIKE_PROFILE = LoadProfile(
    name="spike",
    description="Sudden traffic burst to test auto-scaling and resilience",
    users=200,
    spawn_rate=50.0,  # Very fast spawn
    duration_seconds=300,  # 5 minutes
    warmup_seconds=30,
    cooldown_seconds=60,
    stages=[
        RampingStage(30, 10),  # Baseline
        RampingStage(10, 200),  # Spike up!
        RampingStage(60, 200),  # Sustain spike
        RampingStage(10, 50),  # Sharp drop
        RampingStage(60, 50),  # Recover
        RampingStage(30, 10),  # Return to baseline
        RampingStage(10, 200),  # Second spike
        RampingStage(60, 200),  # Sustain
        RampingStage(30, 0),  # Cool down
    ],
    include_websockets=True,
    include_debates=True,
    include_auth=True,
    include_knowledge=True,
    include_gauntlet=False,  # Skip expensive ops during spike
    max_concurrent_debates=30,
    max_concurrent_websockets=200,
    max_concurrent_searches=100,
    min_wait=0.05,
    max_wait=0.5,
    slo_thresholds=SLOThresholds(
        http_p95_ms=2000,  # Accept degradation during spike
        http_p99_ms=5000,
        max_error_rate=0.10,  # Allow higher errors during spike
        max_timeout_rate=0.02,
    ),
)


SOAK_PROFILE = LoadProfile(
    name="soak",
    description="Extended duration testing for memory leaks and stability",
    users=30,
    spawn_rate=2.0,
    duration_seconds=3600,  # 1 hour
    warmup_seconds=120,
    cooldown_seconds=120,
    include_websockets=True,
    include_debates=True,
    include_auth=True,
    include_knowledge=True,
    include_gauntlet=True,
    max_concurrent_debates=10,
    max_concurrent_websockets=50,
    max_concurrent_searches=20,
    min_wait=1.0,
    max_wait=5.0,
    slo_thresholds=SLOThresholds(
        http_p95_ms=500,
        http_p99_ms=1000,
        max_error_rate=0.01,
        min_throughput_rps=15.0,
    ),
)


# Profile registry
PROFILES: dict[str, LoadProfile] = {
    ProfileType.SMOKE.value: SMOKE_PROFILE,
    ProfileType.LIGHT.value: LIGHT_PROFILE,
    ProfileType.MEDIUM.value: MEDIUM_PROFILE,
    ProfileType.HEAVY.value: HEAVY_PROFILE,
    ProfileType.SPIKE.value: SPIKE_PROFILE,
    ProfileType.SOAK.value: SOAK_PROFILE,
}


def get_profile(name: str | None = None) -> LoadProfile:
    """
    Get a load test profile by name.

    Args:
        name: Profile name (smoke, light, medium, heavy, spike, soak)
              If None, uses ARAGORA_LOAD_PROFILE env var or defaults to "light"

    Returns:
        LoadProfile configuration

    Raises:
        ValueError: If profile name is not recognized
    """
    if name is None:
        name = os.environ.get("ARAGORA_LOAD_PROFILE", "light")

    name = name.lower()

    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")

    return PROFILES[name]


def list_profiles() -> list[dict[str, Any]]:
    """List all available profiles with their descriptions."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "users": p.users,
            "duration_seconds": p.duration_seconds,
        }
        for p in PROFILES.values()
    ]


# =============================================================================
# Environment Configuration
# =============================================================================


@dataclass
class LoadTestEnvironment:
    """Environment configuration for load tests."""

    base_url: str
    api_token: str
    profile: LoadProfile
    workspace_id: str | None = None
    tenant_id: str | None = None

    @classmethod
    def from_env(cls, profile_name: str | None = None) -> LoadTestEnvironment:
        """Create environment configuration from environment variables."""
        return cls(
            base_url=os.environ.get("ARAGORA_BASE_URL", "http://localhost:8080"),
            api_token=os.environ.get("ARAGORA_API_TOKEN", ""),
            profile=get_profile(profile_name),
            workspace_id=os.environ.get("ARAGORA_WORKSPACE_ID"),
            tenant_id=os.environ.get("ARAGORA_TENANT_ID"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "has_token": bool(self.api_token),
            "profile": self.profile.to_dict(),
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
        }


# =============================================================================
# CLI Support
# =============================================================================


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Aragora Load Test Profiles")
    parser.add_argument(
        "action",
        choices=["list", "show", "env"],
        help="Action to perform",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default="light",
        help="Profile name for 'show' action",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "locust", "k6"],
        default="json",
        help="Output format",
    )

    args = parser.parse_args()

    if args.action == "list":
        profiles = list_profiles()
        print("\nAvailable Load Test Profiles:")
        print("=" * 60)
        for p in profiles:
            print(f"\n  {p['name'].upper()}")
            print(f"    {p['description']}")
            print(f"    Users: {p['users']}, Duration: {p['duration_seconds']}s")

    elif args.action == "show":
        profile = get_profile(args.profile)

        if args.format == "json":
            print(json.dumps(profile.to_dict(), indent=2))
        elif args.format == "locust":
            opts = profile.to_locust_options()
            print(
                f"locust --users {opts['users']} "
                f"--spawn-rate {opts['spawn_rate']} "
                f"--run-time {opts['run_time']}"
            )
        elif args.format == "k6":
            print(json.dumps(profile.to_k6_options(), indent=2))

    elif args.action == "env":
        env = LoadTestEnvironment.from_env(args.profile)
        print(json.dumps(env.to_dict(), indent=2))
