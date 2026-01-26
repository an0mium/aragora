"""
Error Recovery Strategies for Connectors.

Provides automated recovery patterns for connector errors:
- Automatic retry with backoff
- Circuit breaker integration
- OAuth token refresh on auth errors
- Fallback to alternative endpoints
- Error escalation

Usage:
    from aragora.connectors.recovery import (
        RecoveryStrategy,
        with_recovery,
        create_recovery_chain,
    )

    # Use decorator for automatic recovery
    @with_recovery(max_retries=3, enable_token_refresh=True)
    async def fetch_data():
        return await connector.search("query")

    # Or use recovery chain directly
    strategy = create_recovery_chain(
        connector=my_connector,
        enable_circuit_breaker=True,
        enable_token_refresh=True,
    )
    result = await strategy.execute(fetch_data)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

if TYPE_CHECKING:
    from aragora.resilience import CircuitBreaker

from aragora.connectors.exceptions import (
    ConnectorAuthError,
    ConnectorError,
    ConnectorRateLimitError,
    classify_exception,
    get_retry_delay,
    is_retryable_error,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryAction(Enum):
    """Actions that can be taken to recover from errors."""

    RETRY = "retry"  # Retry the operation
    WAIT = "wait"  # Wait before retry
    REFRESH_TOKEN = "refresh_token"  # Refresh auth token
    OPEN_CIRCUIT = "open_circuit"  # Open circuit breaker
    FALLBACK = "fallback"  # Use fallback endpoint/method
    ESCALATE = "escalate"  # Escalate to human/alerting
    FAIL = "fail"  # Give up and fail


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    action: RecoveryAction
    success: bool
    error: Optional[Exception] = None
    attempts: int = 0
    total_wait_seconds: float = 0.0
    message: str = ""


@dataclass
class RecoveryConfig:
    """Configuration for recovery strategies."""

    # Retry settings
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter_factor: float = 0.3

    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_failures: int = 5
    circuit_breaker_cooldown: float = 60.0

    # Token refresh
    enable_token_refresh: bool = True
    token_refresh_timeout: float = 30.0

    # Fallback
    fallback_endpoints: List[str] = field(default_factory=list)

    # Escalation
    escalation_threshold: int = 3  # Consecutive failures before escalation
    escalation_callback: Optional[Callable[[Exception, int], None]] = None


class RecoveryStrategy:
    """
    Coordinates recovery actions for connector errors.

    Provides a unified approach to handling failures with:
    - Automatic retry with exponential backoff
    - Circuit breaker integration
    - OAuth token refresh
    - Fallback support
    - Error escalation
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        connector_name: str = "unknown",
    ):
        """
        Initialize recovery strategy.

        Args:
            config: Recovery configuration
            connector_name: Name for logging and metrics
        """
        self.config = config or RecoveryConfig()
        self.connector_name = connector_name

        # State tracking
        self._consecutive_failures = 0
        self._total_retries = 0
        self._last_error_time: Optional[float] = None

        # Circuit breaker (lazy initialized)
        self._circuit_breaker: Optional["CircuitBreaker"] = None

        # Token refresh callback
        self._refresh_token_callback: Optional[Callable[[], Any]] = None

    def set_token_refresh_callback(
        self,
        callback: Callable[[], Any],
    ) -> None:
        """Set callback for refreshing auth tokens."""
        self._refresh_token_callback = callback

    def _get_circuit_breaker(self):
        """Get or create circuit breaker."""
        if self._circuit_breaker is None and self.config.enable_circuit_breaker:
            from aragora.resilience import get_circuit_breaker

            self._circuit_breaker = get_circuit_breaker(
                name=f"recovery_{self.connector_name}",
                failure_threshold=self.config.circuit_breaker_failures,
                cooldown_seconds=self.config.circuit_breaker_cooldown,
            )
        return self._circuit_breaker

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay with jitter."""
        import random

        delay = min(
            self.config.base_delay_seconds * (2**attempt),
            self.config.max_delay_seconds,
        )
        jitter = delay * self.config.jitter_factor * random.uniform(-1, 1)
        return max(0.1, delay + jitter)

    def _determine_recovery_action(
        self,
        error: Exception,
        attempt: int,
    ) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""
        # Check circuit breaker first
        cb = self._get_circuit_breaker()
        if cb is not None and not cb.can_proceed():
            return RecoveryAction.OPEN_CIRCUIT

        # Classify the error if needed
        if not isinstance(error, ConnectorError):
            error = classify_exception(error, self.connector_name)

        # Auth errors -> try token refresh
        if isinstance(error, ConnectorAuthError) and self.config.enable_token_refresh:
            if self._refresh_token_callback is not None:
                return RecoveryAction.REFRESH_TOKEN
            return RecoveryAction.FAIL

        # Rate limits -> wait
        if isinstance(error, ConnectorRateLimitError):
            return RecoveryAction.WAIT

        # Retryable errors -> retry
        if is_retryable_error(error) and attempt < self.config.max_retries:
            return RecoveryAction.RETRY

        # Exhausted retries with fallback available
        if attempt >= self.config.max_retries and self.config.fallback_endpoints:
            return RecoveryAction.FALLBACK

        # Check escalation threshold
        if self._consecutive_failures >= self.config.escalation_threshold:
            return RecoveryAction.ESCALATE

        return RecoveryAction.FAIL

    async def execute(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an operation with automatic recovery.

        Args:
            operation: Async callable to execute
            operation_name: Name for logging

        Returns:
            Result from successful operation

        Raises:
            ConnectorError: If all recovery attempts fail
        """
        attempt = 0
        total_wait = 0.0
        last_error: Optional[Exception] = None

        while True:
            # Check circuit breaker
            cb = self._get_circuit_breaker()
            if cb is not None and not cb.can_proceed():
                logger.warning(f"[{self.connector_name}] Circuit open for {operation_name}")
                raise ConnectorError(
                    f"Circuit breaker open for {operation_name}",
                    connector_name=self.connector_name,
                    is_retryable=False,
                )

            try:
                # Execute operation
                result = await operation()

                # Success - reset failure counters
                self._consecutive_failures = 0
                if cb is not None:
                    cb.record_success()

                return result

            except Exception as e:
                last_error = e
                self._consecutive_failures += 1
                self._last_error_time = time.time()

                if cb is not None:
                    cb.record_failure()

                # Determine recovery action
                action = self._determine_recovery_action(e, attempt)
                logger.info(
                    f"[{self.connector_name}] {operation_name} failed "
                    f"(attempt {attempt + 1}), action: {action.value}"
                )

                if action == RecoveryAction.RETRY:
                    delay = self._calculate_backoff(attempt)
                    logger.info(f"[{self.connector_name}] Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    total_wait += delay
                    attempt += 1
                    continue

                elif action == RecoveryAction.WAIT:
                    delay = get_retry_delay(e, default=60.0)
                    logger.info(f"[{self.connector_name}] Rate limited, waiting {delay:.1f}s")
                    await asyncio.sleep(delay)
                    total_wait += delay
                    # Don't count as retry attempt for rate limits
                    continue

                elif action == RecoveryAction.REFRESH_TOKEN:
                    logger.info(f"[{self.connector_name}] Refreshing auth token")
                    try:
                        if asyncio.iscoroutinefunction(self._refresh_token_callback):
                            await self._refresh_token_callback()
                        else:
                            self._refresh_token_callback()
                        # Retry after token refresh
                        attempt += 1
                        continue
                    except Exception as refresh_error:
                        logger.error(
                            f"[{self.connector_name}] Token refresh failed: {refresh_error}"
                        )
                        raise classify_exception(e, self.connector_name) from e

                elif action == RecoveryAction.ESCALATE:
                    logger.warning(
                        f"[{self.connector_name}] Escalating after "
                        f"{self._consecutive_failures} consecutive failures"
                    )
                    if self.config.escalation_callback:
                        try:
                            self.config.escalation_callback(e, self._consecutive_failures)
                        except Exception as cb_error:
                            logger.error(f"Escalation callback failed: {cb_error}")
                    # Fall through to fail

                elif action == RecoveryAction.OPEN_CIRCUIT:
                    raise ConnectorError(
                        "Circuit breaker open",
                        connector_name=self.connector_name,
                        is_retryable=False,
                    )

                # Give up
                classified = classify_exception(e, self.connector_name)
                logger.error(
                    f"[{self.connector_name}] {operation_name} failed after "
                    f"{attempt + 1} attempts: {classified}"
                )
                raise classified from e

        # Should never reach here
        if last_error:
            raise classify_exception(last_error, self.connector_name)
        raise ConnectorError("Unknown error", connector_name=self.connector_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "connector_name": self.connector_name,
            "consecutive_failures": self._consecutive_failures,
            "total_retries": self._total_retries,
            "last_error_time": self._last_error_time,
            "circuit_breaker": (
                self._circuit_breaker.get_status() if self._circuit_breaker else None
            ),
        }


def with_recovery(
    max_retries: int = 3,
    enable_token_refresh: bool = False,
    enable_circuit_breaker: bool = True,
    connector_name: Optional[str] = None,
) -> Callable:
    """
    Decorator to add automatic recovery to async functions.

    Args:
        max_retries: Maximum retry attempts
        enable_token_refresh: Whether to attempt token refresh on auth errors
        enable_circuit_breaker: Whether to use circuit breaker
        connector_name: Name for logging (auto-detected if not provided)

    Returns:
        Decorated function with recovery

    Example:
        @with_recovery(max_retries=3)
        async def fetch_data():
            return await client.get(url)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Determine connector name
        name = connector_name or func.__name__

        # Create strategy
        config = RecoveryConfig(
            max_retries=max_retries,
            enable_token_refresh=enable_token_refresh,
            enable_circuit_breaker=enable_circuit_breaker,
        )
        strategy = RecoveryStrategy(config=config, connector_name=name)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async def operation():
                return await func(*args, **kwargs)

            return await strategy.execute(operation, func.__name__)

        # Attach strategy for configuration
        wrapper._recovery_strategy = strategy  # type: ignore
        return wrapper

    return decorator


def create_recovery_chain(
    connector: Any,
    config: Optional[RecoveryConfig] = None,
) -> RecoveryStrategy:
    """
    Create a recovery strategy for a connector.

    Automatically configures token refresh if the connector
    has a refresh method.

    Args:
        connector: Connector instance
        config: Recovery configuration

    Returns:
        Configured RecoveryStrategy

    Example:
        strategy = create_recovery_chain(my_connector)
        result = await strategy.execute(
            lambda: connector.search("query")
        )
    """
    config = config or RecoveryConfig()
    connector_name = getattr(connector, "name", type(connector).__name__)

    strategy = RecoveryStrategy(
        config=config,
        connector_name=connector_name,
    )

    # Set up token refresh if available
    if config.enable_token_refresh:
        refresh_method = getattr(connector, "_get_access_token", None)
        if refresh_method is None:
            refresh_method = getattr(connector, "refresh_token", None)
        if refresh_method is not None:
            strategy.set_token_refresh_callback(refresh_method)

    return strategy


__all__ = [
    "RecoveryAction",
    "RecoveryResult",
    "RecoveryConfig",
    "RecoveryStrategy",
    "with_recovery",
    "create_recovery_chain",
]
