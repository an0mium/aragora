"""
Slack messaging utilities.

Response helpers and async message posting for Slack Web API and response URLs.

Includes circuit breaker pattern for resilience when Slack APIs are degraded.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import (
    HandlerResult,
    SLACK_BOT_TOKEN,
    _validate_slack_url,
    json_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker for Slack API
# =============================================================================


class SlackCircuitBreaker:
    """Circuit breaker for Slack API calls.

    Prevents cascading failures when Slack APIs are unavailable or degraded.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit tripped, requests fail fast without calling Slack
    - HALF_OPEN: Testing if Slack is back, limited requests allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            cooldown_seconds: Time to wait before testing if Slack is back
            half_open_max_calls: Number of test calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Slack circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_proceed(self) -> bool:
        """Check if a call can proceed.

        Returns:
            True if call is allowed, False if circuit is open
        """
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Slack circuit breaker closed after successful recovery")
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._state = self.OPEN
                self._success_count = 0
                logger.warning("Slack circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Slack circuit breaker opened after {self._failure_count} failures"
                    )

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state (for testing)."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Global circuit breaker instance for Slack API calls
_slack_circuit_breaker: SlackCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def get_slack_circuit_breaker() -> SlackCircuitBreaker:
    """Get or create the Slack circuit breaker singleton."""
    global _slack_circuit_breaker
    with _circuit_breaker_lock:
        if _slack_circuit_breaker is None:
            _slack_circuit_breaker = SlackCircuitBreaker()
        return _slack_circuit_breaker


def reset_slack_circuit_breaker() -> None:
    """Reset the circuit breaker (for testing)."""
    global _slack_circuit_breaker
    with _circuit_breaker_lock:
        if _slack_circuit_breaker is not None:
            _slack_circuit_breaker.reset()


class MessagingMixin:
    """Mixin providing Slack message posting and response formatting."""

    def _slack_response(
        self,
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a simple Slack response."""
        return json_response(
            {
                "response_type": response_type,
                "text": text,
            }
        )

    def _slack_blocks_response(
        self,
        blocks: list[dict[str, Any]],
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a Slack response with blocks."""
        return json_response(
            {
                "response_type": response_type,
                "text": text,
                "blocks": blocks,
            }
        )

    async def _post_to_response_url(self, url: str, payload: dict[str, Any]) -> None:
        """POST a message to Slack's response_url.

        Includes:
        - SSRF protection by validating the URL is a Slack endpoint
        - Circuit breaker pattern for resilience
        """
        # Validate URL to prevent SSRF attacks
        if not _validate_slack_url(url):
            logger.warning(f"Invalid Slack response_url blocked (SSRF protection): {url[:50]}")
            return

        # Check circuit breaker before making the call
        circuit_breaker = get_slack_circuit_breaker()
        if not circuit_breaker.can_proceed():
            logger.warning("Slack circuit breaker OPEN - skipping response_url POST")
            return

        from aragora.server.http_client_pool import get_http_pool

        try:
            pool = get_http_pool()
            async with pool.get_session("slack") as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if response.status_code != 200:
                    text = response.text
                    logger.warning(
                        f"Slack response_url POST failed: {response.status_code} - {text[:100]}"
                    )
                    # Record failure for non-2xx responses
                    if response.status_code >= 500:
                        circuit_breaker.record_failure()
                else:
                    circuit_breaker.record_success()
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error posting to Slack response_url: {e}")
            circuit_breaker.record_failure()
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.exception(f"Unexpected error posting to Slack response_url: {e}")
            circuit_breaker.record_failure()

    async def _post_message_async(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Post a message to Slack using the Web API.

        Includes circuit breaker pattern for resilience when Slack APIs are degraded.

        Args:
            channel: Channel ID to post to
            text: Message text
            thread_ts: Optional thread timestamp to reply to
            blocks: Optional Block Kit blocks for rich formatting

        Returns:
            Message timestamp (ts) if successful, None otherwise
        """
        from aragora.server.http_client_pool import get_http_pool

        if not SLACK_BOT_TOKEN:
            logger.warning("Cannot post message: SLACK_BOT_TOKEN not configured")
            return None

        # Check circuit breaker before making the call
        circuit_breaker = get_slack_circuit_breaker()
        if not circuit_breaker.can_proceed():
            logger.warning("Slack circuit breaker OPEN - skipping Web API POST")
            return None

        try:
            payload: dict[str, Any] = {
                "channel": channel,
                "text": text,
            }
            if thread_ts:
                payload["thread_ts"] = thread_ts
            if blocks:
                payload["blocks"] = blocks

            pool = get_http_pool()
            async with pool.get_session("slack") as client:
                response = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                result = response.json()
                if not result.get("ok"):
                    error = result.get("error", "unknown")
                    logger.warning(f"Slack API error: {error}")
                    # Some errors indicate Slack issues (rate_limited, service_unavailable)
                    if error in ("rate_limited", "service_unavailable", "fatal_error"):
                        circuit_breaker.record_failure()
                    return None
                # Success - record for circuit breaker
                circuit_breaker.record_success()
                # Return message timestamp for thread tracking
                return result.get("ts")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error posting Slack message: {e}")
            circuit_breaker.record_failure()
            return None
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.exception(f"Unexpected error posting Slack message: {e}")
            circuit_breaker.record_failure()
            return None

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get the Slack circuit breaker status for monitoring."""
        return get_slack_circuit_breaker().get_status()
