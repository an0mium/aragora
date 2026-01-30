#!/usr/bin/env python3
"""
08_error_handling.py - Resilient error handling patterns.

This example demonstrates:
- Retry patterns with exponential backoff
- Fallback agents when primary fails
- Circuit breaker for failing services

Usage:
    python 08_error_handling.py --dry-run
    python 08_error_handling.py --scenario retry
    python 08_error_handling.py --scenario circuit-breaker
"""

import argparse
import asyncio
from aragora_sdk import ArenaClient, DebateConfig, Agent
from aragora_sdk.resilience import (
    RetryPolicy,
    CircuitBreaker,
    CircuitState,
    FallbackAgent,
)
from aragora_sdk.errors import RateLimitError, ServiceUnavailableError, AgentError


# -----------------------------------------------------------------------------
# Pattern 1: Retry with exponential backoff
# -----------------------------------------------------------------------------


async def retry_pattern_example(dry_run: bool = False) -> dict:
    """Demonstrate retry pattern with exponential backoff."""

    print("\n=== Retry Pattern ===")

    # Configure retry policy
    retry_policy = RetryPolicy(
        max_retries=3,
        initial_delay=1.0,  # Start with 1 second delay
        max_delay=30.0,  # Cap at 30 seconds
        exponential_base=2.0,  # Double delay each retry
        jitter=0.1,  # Add 10% random jitter
        # Only retry these error types
        retryable_errors=(RateLimitError, ServiceUnavailableError),
    )

    if dry_run:
        print(f"[DRY RUN] Retry policy: max_retries={retry_policy.max_retries}")
        print(f"[DRY RUN] Backoff: {retry_policy.initial_delay}s -> {retry_policy.max_delay}s")
        print("[DRY RUN] Would retry on: RateLimitError, ServiceUnavailableError")
        return {"pattern": "retry", "status": "dry_run"}

    client = ArenaClient(retry_policy=retry_policy)

    # The client will automatically retry failed requests
    try:
        result = await client.run_debate(
            DebateConfig(
                topic="Retry pattern test",
                agents=[Agent(name="claude", model="claude-sonnet-4-20250514")],
                rounds=1,
            )
        )
        return {"pattern": "retry", "status": "success", "result": result.to_dict()}
    except Exception as e:
        return {"pattern": "retry", "status": "failed_after_retries", "error": str(e)}


# -----------------------------------------------------------------------------
# Pattern 2: Fallback agents
# -----------------------------------------------------------------------------


async def fallback_agent_example(dry_run: bool = False) -> dict:
    """Demonstrate fallback agents when primary fails."""

    print("\n=== Fallback Agents ===")

    # Define agents with fallback chain
    primary_agent = Agent(
        name="claude",
        model="claude-sonnet-4-20250514",
        fallback=FallbackAgent(
            agent=Agent(name="gpt_fallback", model="gpt-4o"),
            # Use fallback on these errors
            trigger_on=(RateLimitError, AgentError),
            fallback=FallbackAgent(
                # Second-level fallback (OpenRouter)
                agent=Agent(name="openrouter_fallback", model="anthropic/claude-3-haiku"),
                trigger_on=(RateLimitError, AgentError),
            ),
        ),
    )

    if dry_run:
        print("[DRY RUN] Primary: claude-sonnet-4-20250514")
        print("[DRY RUN] Fallback 1: gpt-4o")
        print("[DRY RUN] Fallback 2: anthropic/claude-3-haiku (via OpenRouter)")
        return {"pattern": "fallback", "status": "dry_run"}

    client = ArenaClient()

    result = await client.run_debate(
        DebateConfig(
            topic="Fallback agents test",
            agents=[primary_agent],
            rounds=1,
        )
    )

    # Check which agent actually responded
    actual_agent = result.contributions[0].agent if result.contributions else "unknown"
    used_fallback = actual_agent != "claude"

    return {
        "pattern": "fallback",
        "status": "success",
        "used_fallback": used_fallback,
        "actual_agent": actual_agent,
    }


# -----------------------------------------------------------------------------
# Pattern 3: Circuit breaker
# -----------------------------------------------------------------------------


async def circuit_breaker_example(dry_run: bool = False) -> dict:
    """Demonstrate circuit breaker for failing services."""

    print("\n=== Circuit Breaker ===")

    # Configure circuit breaker
    circuit = CircuitBreaker(
        failure_threshold=5,  # Open after 5 failures
        recovery_timeout=60.0,  # Try to recover after 60 seconds
        half_open_max_calls=3,  # Allow 3 test calls when half-open
    )

    if dry_run:
        print(f"[DRY RUN] Circuit state: {circuit.state.value}")
        print(f"[DRY RUN] Failure threshold: {circuit.failure_threshold}")
        print(f"[DRY RUN] Recovery timeout: {circuit.recovery_timeout}s")

        # Simulate circuit states
        print("\n[DRY RUN] Circuit state transitions:")
        print("  CLOSED -> (5 failures) -> OPEN")
        print("  OPEN -> (60s timeout) -> HALF_OPEN")
        print("  HALF_OPEN -> (success) -> CLOSED")
        print("  HALF_OPEN -> (failure) -> OPEN")
        return {"pattern": "circuit_breaker", "status": "dry_run"}

    client = ArenaClient(circuit_breaker=circuit)

    # Check circuit state before making calls
    if circuit.state == CircuitState.OPEN:
        print("Circuit is OPEN - calls will fail fast")
        return {"pattern": "circuit_breaker", "status": "circuit_open"}

    try:
        result = await client.run_debate(
            DebateConfig(
                topic="Circuit breaker test",
                agents=[Agent(name="claude", model="claude-sonnet-4-20250514")],
                rounds=1,
            )
        )

        # Record success
        circuit.record_success()
        return {
            "pattern": "circuit_breaker",
            "status": "success",
            "circuit_state": circuit.state.value,
        }

    except Exception as e:
        # Record failure
        circuit.record_failure()
        return {
            "pattern": "circuit_breaker",
            "status": "failed",
            "circuit_state": circuit.state.value,
            "error": str(e),
        }


async def run_error_handling_example(scenario: str, dry_run: bool = False) -> dict:
    """Run the specified error handling scenario."""

    if scenario == "retry":
        return await retry_pattern_example(dry_run)
    elif scenario == "fallback":
        return await fallback_agent_example(dry_run)
    elif scenario == "circuit-breaker":
        return await circuit_breaker_example(dry_run)
    else:
        # Run all scenarios
        results = {}
        for s in ["retry", "fallback", "circuit-breaker"]:
            results[s] = await run_error_handling_example(s, dry_run)
        return results


def main():
    parser = argparse.ArgumentParser(description="Error handling patterns demo")
    parser.add_argument(
        "--scenario",
        choices=["retry", "fallback", "circuit-breaker", "all"],
        default="all",
        help="Scenario to demonstrate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_error_handling_example(args.scenario, args.dry_run))
    return result


if __name__ == "__main__":
    main()
