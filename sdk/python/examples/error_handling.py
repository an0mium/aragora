"""
Error Handling Example

Demonstrates comprehensive error handling patterns with the Aragora SDK.
Shows all exception types and retry strategies.

Usage:
    python examples/error_handling.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from aragora_sdk import AragoraClient
from aragora_sdk.exceptions import (
    AragoraError,
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)

T = TypeVar("T")


# =============================================================================
# Retry Strategies
# =============================================================================


def retry_with_exponential_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_errors: tuple[type[Exception], ...] = (
        RateLimitError,
        ServerError,
        TimeoutError,
        ConnectionError,
    ),
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry (should take no arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        retryable_errors: Exception types that should trigger a retry

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_errors as e:
            last_exception = e
            if attempt == max_retries:
                print(f"  All {max_retries} retries exhausted")
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2**attempt), max_delay)

            # Use retry_after if available (rate limit errors)
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = e.retry_after

            print(f"  Attempt {attempt + 1} failed: {e}")
            print(f"  Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise last_exception or RuntimeError("Unexpected retry exit")


# =============================================================================
# Error Handling Patterns
# =============================================================================


def handle_authentication_error(client: AragoraClient) -> None:
    """Demonstrate handling authentication errors."""
    print("\n=== Authentication Error Handling ===")
    print("Simulating request with invalid credentials...")

    # Create a client with invalid API key
    bad_client = AragoraClient(
        base_url=client._base_url,
        api_key="invalid",
    )

    try:
        bad_client.debates.list()
    except AuthenticationError as e:
        print("Caught AuthenticationError:")
        print(f"  Message: {e.message}")
        print(f"  Status code: {e.status_code}")
        print(f"  Error code: {e.error_code}")
        print(f"  Trace ID: {e.trace_id}")
        print("\nRecovery: Check your ARAGORA_API_KEY environment variable")


def handle_authorization_error(client: AragoraClient) -> None:
    """Demonstrate handling authorization errors."""
    print("\n=== Authorization Error Handling ===")
    print("Simulating access to restricted resource...")

    try:
        # Attempt to access admin-only endpoint
        client.admin.list_all_users()  # type: ignore
    except AuthorizationError as e:
        print("Caught AuthorizationError:")
        print(f"  Message: {e.message}")
        print(f"  Status code: {e.status_code}")
        print(f"  Error code: {e.error_code}")
        print("\nRecovery: Contact your admin to request access")
    except AragoraError as e:
        # Endpoint might not exist or different error
        print(f"Caught AragoraError: {e.message}")


def handle_not_found_error(client: AragoraClient) -> None:
    """Demonstrate handling not found errors."""
    print("\n=== Not Found Error Handling ===")
    print("Simulating request for non-existent resource...")

    try:
        client.debates.get("nonexistent-debate-id-12345")
    except NotFoundError as e:
        print("Caught NotFoundError:")
        print(f"  Message: {e.message}")
        print(f"  Status code: {e.status_code}")
        print(f"  Trace ID: {e.trace_id}")
        print("\nRecovery: Verify the resource ID and try again")


def handle_validation_error(client: AragoraClient) -> None:
    """Demonstrate handling validation errors."""
    print("\n=== Validation Error Handling ===")
    print("Simulating request with invalid parameters...")

    try:
        # Pass invalid parameters
        client.debates.create(
            task="",  # Empty task (invalid)
            agents=[],  # No agents (invalid)
            rounds=-1,  # Negative rounds (invalid)
        )
    except ValidationError as e:
        print("Caught ValidationError:")
        print(f"  Message: {e.message}")
        print(f"  Status code: {e.status_code}")
        print(f"  Error code: {e.error_code}")
        if e.errors:
            print("  Validation errors:")
            for err in e.errors:
                field = err.get("field", "unknown")
                msg = err.get("message", err)
                print(f"    - {field}: {msg}")
        print("\nRecovery: Check request parameters against API documentation")


def handle_rate_limit_error(client: AragoraClient) -> None:
    """Demonstrate handling rate limit errors with retry."""
    print("\n=== Rate Limit Error Handling ===")
    print("When rate limited, the SDK provides retry_after information.")

    # Example of how rate limit errors are handled
    print("\nExample rate limit response:")
    example_error = RateLimitError(
        message="Rate limit exceeded",
        retry_after=30,
        error_code="RATE_LIMITED",
        trace_id="trace-abc123",
    )
    print(f"  Error: {example_error}")
    print(f"  Retry after: {example_error.retry_after}s")

    print("\nUse retry_with_exponential_backoff() for automatic retry:")
    print("  result = retry_with_exponential_backoff(lambda: client.debates.list())")


def handle_server_error(client: AragoraClient) -> None:
    """Demonstrate handling server errors."""
    print("\n=== Server Error Handling ===")
    print("Server errors (5xx) should be retried with backoff.")

    # Example of how server errors work
    example_error = ServerError(
        message="Internal server error",
        error_code="INTERNAL_ERROR",
        trace_id="trace-xyz789",
    )
    print("\nExample server error:")
    print(f"  Message: {example_error.message}")
    print(f"  Error code: {example_error.error_code}")
    print(f"  Trace ID: {example_error.trace_id}")
    print("\nRecovery: Retry with exponential backoff, contact support if persistent")


def handle_timeout_error(client: AragoraClient) -> None:
    """Demonstrate handling timeout errors."""
    print("\n=== Timeout Error Handling ===")
    print("Timeouts can occur during long-running operations.")

    # Example timeout
    example_error = TimeoutError(
        message="Request timed out after 30s",
        trace_id="trace-timeout-123",
    )
    print("\nExample timeout error:")
    print(f"  Message: {example_error.message}")
    print(f"  Trace ID: {example_error.trace_id}")

    print("\nRecovery options:")
    print("  1. Increase timeout: client = AragoraClient(timeout=60.0)")
    print("  2. Retry the request with exponential backoff")
    print("  3. For long operations, use async streaming instead")


def handle_connection_error(client: AragoraClient) -> None:
    """Demonstrate handling connection errors."""
    print("\n=== Connection Error Handling ===")
    print("Connection errors occur when the server is unreachable.")

    # Create client with invalid URL
    bad_client = AragoraClient(
        base_url="https://invalid-host.aragora.ai",
        api_key="test-key",
    )

    try:
        bad_client.debates.list()
    except ConnectionError as e:
        print("Caught ConnectionError:")
        print(f"  Message: {e.message}")
        print("\nRecovery: Check network connection and ARAGORA_API_URL")
    except AragoraError as e:
        # Might be a different error type depending on httpx behavior
        print(f"Caught AragoraError: {e.message}")


# =============================================================================
# Robust Client Pattern
# =============================================================================


class RobustAragoraClient:
    """Wrapper providing automatic retry and error handling."""

    def __init__(
        self,
        client: AragoraClient,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self._client = client
        self._max_retries = max_retries
        self._base_delay = base_delay

    def create_debate_with_retry(self, **kwargs: Any) -> dict[str, Any]:
        """Create a debate with automatic retry on transient errors."""
        return retry_with_exponential_backoff(
            func=lambda: self._client.debates.create(**kwargs),
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )

    def get_debate_with_fallback(
        self, debate_id: str, fallback: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get a debate, returning fallback on not found."""
        try:
            return self._client.debates.get(debate_id)
        except NotFoundError:
            if fallback is not None:
                return fallback
            raise


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run error handling demonstrations."""
    print("Aragora SDK Error Handling Examples")
    print("=" * 50)

    # Create client
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY", "demo-key"),
    )

    # Run demonstrations
    handle_authentication_error(client)
    handle_authorization_error(client)
    handle_not_found_error(client)
    handle_validation_error(client)
    handle_rate_limit_error(client)
    handle_server_error(client)
    handle_timeout_error(client)
    handle_connection_error(client)

    # Show robust client pattern
    print("\n" + "=" * 50)
    print("=== Robust Client Pattern ===")
    print("\nWrap your client for automatic error handling:")
    print("""
    robust = RobustAragoraClient(client, max_retries=3)
    debate = robust.create_debate_with_retry(
        task="What should we decide?",
        agents=["claude", "gpt-4"],
    )
    """)

    print("\n" + "=" * 50)
    print("All error handling examples complete!")
    print("\nException Hierarchy:")
    print("  AragoraError (base)")
    print("  ├── AuthenticationError (401)")
    print("  ├── AuthorizationError (403)")
    print("  ├── NotFoundError (404)")
    print("  ├── RateLimitError (429)")
    print("  ├── ValidationError (400)")
    print("  ├── ServerError (5xx)")
    print("  ├── TimeoutError")
    print("  └── ConnectionError")


if __name__ == "__main__":
    main()
