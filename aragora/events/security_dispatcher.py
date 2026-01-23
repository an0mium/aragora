"""
Security Event Dispatcher.

Listens for security events and triggers multi-agent debates when severity
thresholds are met. Provides configurable thresholds and integrates with
the Arena debate orchestrator.

Flow:
    Scanner detects critical issue → SecurityEvent emitted →
    SecurityDispatcher receives → Arena.run_security_debate() →
    Multi-agent deliberation → ConsensusProof with remediation

Usage:
    from aragora.events.security_dispatcher import (
        SecurityDispatcher,
        get_security_dispatcher,
    )

    # Get global dispatcher
    dispatcher = get_security_dispatcher()

    # Configure thresholds
    dispatcher.set_severity_threshold(SecuritySeverity.HIGH)

    # Start listening (usually done at application startup)
    await dispatcher.start()

    # Events are automatically dispatched to debates when severity >= threshold
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from aragora.events.security_events import (
    SecurityEvent,
    SecurityEventEmitter,
    SecurityEventType,
    SecuritySeverity,
    get_security_emitter,
)

logger = logging.getLogger(__name__)

# Type alias for debate trigger callback
DebateTriggerCallback = Callable[[SecurityEvent], Coroutine[Any, Any, Optional[str]]]


@dataclass
class DispatcherConfig:
    """Configuration for the security event dispatcher."""

    # Severity threshold for triggering debates
    min_severity: SecuritySeverity = SecuritySeverity.HIGH

    # Minimum number of critical findings to trigger debate
    critical_finding_threshold: int = 1

    # Minimum number of high-severity findings to trigger debate
    high_finding_threshold: int = 3

    # Event types that should always trigger a debate
    always_trigger_types: Set[SecurityEventType] = field(
        default_factory=lambda: {
            SecurityEventType.CRITICAL_CVE,
            SecurityEventType.CRITICAL_VULNERABILITY,
            SecurityEventType.CRITICAL_SECRET,
            SecurityEventType.SAST_CRITICAL,
            SecurityEventType.THREAT_DETECTED,
        }
    )

    # Cooldown period (seconds) between debates for same repository
    repository_cooldown_seconds: int = 300

    # Maximum concurrent debates
    max_concurrent_debates: int = 5

    # Confidence threshold for debate consensus
    debate_confidence_threshold: float = 0.7

    # Timeout for debates (seconds)
    debate_timeout_seconds: int = 300

    # Whether to auto-start the dispatcher
    auto_start: bool = False


@dataclass
class DispatcherStats:
    """Statistics for the security dispatcher."""

    events_received: int = 0
    events_filtered: int = 0
    debates_triggered: int = 0
    debates_completed: int = 0
    debates_failed: int = 0
    last_event_time: Optional[datetime] = None
    last_debate_time: Optional[datetime] = None


class SecurityDispatcher:
    """
    Dispatches security events to multi-agent debates.

    Listens for security events and triggers debates when severity thresholds
    are met. Manages cooldowns, concurrent debate limits, and provides
    statistics on dispatch activity.

    Features:
    - Configurable severity thresholds
    - Repository-based cooldown to prevent debate flooding
    - Concurrent debate limits
    - Event type filtering
    - Statistics and monitoring

    Example:
        dispatcher = SecurityDispatcher()
        await dispatcher.start()

        # Events from SecurityEventEmitter will automatically trigger debates
        # when severity >= configured threshold
    """

    def __init__(
        self,
        config: Optional[DispatcherConfig] = None,
        emitter: Optional[SecurityEventEmitter] = None,
    ):
        """
        Initialize the security dispatcher.

        Args:
            config: Dispatcher configuration
            emitter: Optional SecurityEventEmitter to subscribe to
        """
        self.config = config or DispatcherConfig()
        self._emitter = emitter
        self._running = False
        self._pending_debates: Dict[str, asyncio.Task] = {}
        self._repository_cooldowns: Dict[str, datetime] = {}
        self._stats = DispatcherStats()
        self._custom_trigger_callback: Optional[DebateTriggerCallback] = None

    async def start(self) -> None:
        """
        Start the dispatcher and subscribe to security events.

        Should be called at application startup.
        """
        if self._running:
            logger.warning("SecurityDispatcher is already running")
            return

        # Get or create emitter
        if self._emitter is None:
            self._emitter = get_security_emitter()

        # Subscribe to all security events
        self._emitter.subscribe_all(self._handle_event)
        self._running = True
        logger.info(f"SecurityDispatcher started (min_severity={self.config.min_severity.value})")

    async def stop(self) -> None:
        """
        Stop the dispatcher and cancel pending debates.

        Should be called at application shutdown.
        """
        self._running = False

        # Cancel pending debates
        for debate_id, task in list(self._pending_debates.items()):
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled pending debate {debate_id}")

        self._pending_debates.clear()
        logger.info("SecurityDispatcher stopped")

    def set_severity_threshold(self, severity: SecuritySeverity) -> None:
        """
        Set the minimum severity threshold for triggering debates.

        Args:
            severity: Minimum severity level
        """
        self.config.min_severity = severity
        logger.info(f"Security dispatch threshold set to {severity.value}")

    def set_custom_trigger(self, callback: DebateTriggerCallback) -> None:
        """
        Set a custom callback for triggering debates.

        This allows overriding the default Arena.run_security_debate call.

        Args:
            callback: Async function that takes a SecurityEvent and returns
                     the debate ID or None
        """
        self._custom_trigger_callback = callback

    async def _handle_event(self, event: SecurityEvent) -> None:
        """
        Handle an incoming security event.

        Determines if the event should trigger a debate based on:
        - Severity threshold
        - Event type
        - Finding counts
        - Repository cooldowns
        - Concurrent debate limits

        Args:
            event: The security event to handle
        """
        self._stats.events_received += 1
        self._stats.last_event_time = datetime.now(timezone.utc)

        # Check if dispatcher is running
        if not self._running:
            return

        # Check if event should trigger a debate
        if not self._should_trigger_debate(event):
            self._stats.events_filtered += 1
            return

        # Check repository cooldown
        if not self._check_cooldown(event.repository):
            logger.debug(f"Debate skipped for {event.repository}: cooldown active")
            self._stats.events_filtered += 1
            return

        # Check concurrent debate limit
        if len(self._pending_debates) >= self.config.max_concurrent_debates:
            logger.warning(
                f"Max concurrent debates ({self.config.max_concurrent_debates}) reached, "
                f"skipping event {event.id}"
            )
            self._stats.events_filtered += 1
            return

        # Trigger the debate
        await self._trigger_debate(event)

    def _should_trigger_debate(self, event: SecurityEvent) -> bool:
        """
        Determine if an event should trigger a debate.

        Args:
            event: The security event to evaluate

        Returns:
            True if the event should trigger a debate
        """
        # Always trigger for specific event types
        if event.event_type in self.config.always_trigger_types:
            return True

        # Check severity threshold
        severity_order = {
            SecuritySeverity.CRITICAL: 0,
            SecuritySeverity.HIGH: 1,
            SecuritySeverity.MEDIUM: 2,
            SecuritySeverity.LOW: 3,
            SecuritySeverity.INFO: 4,
        }
        event_severity = severity_order.get(event.severity, 4)
        threshold_severity = severity_order.get(self.config.min_severity, 1)

        if event_severity > threshold_severity:
            return False

        # Check finding counts
        if event.critical_count >= self.config.critical_finding_threshold:
            return True

        if event.high_count >= self.config.high_finding_threshold:
            return True

        # Check if event is critical severity
        if event.is_critical:
            return True

        return False

    def _check_cooldown(self, repository: Optional[str]) -> bool:
        """
        Check if a repository is in cooldown period.

        Args:
            repository: Repository identifier

        Returns:
            True if debate can proceed (not in cooldown)
        """
        if not repository:
            return True

        last_debate = self._repository_cooldowns.get(repository)
        if not last_debate:
            return True

        elapsed = (datetime.now(timezone.utc) - last_debate).total_seconds()
        return elapsed >= self.config.repository_cooldown_seconds

    def _set_cooldown(self, repository: Optional[str]) -> None:
        """Set cooldown for a repository."""
        if repository:
            self._repository_cooldowns[repository] = datetime.now(timezone.utc)

    async def _trigger_debate(self, event: SecurityEvent) -> None:
        """
        Trigger a multi-agent debate for the security event.

        Args:
            event: The security event to debate
        """
        self._stats.debates_triggered += 1
        self._stats.last_debate_time = datetime.now(timezone.utc)

        # Set cooldown for repository
        self._set_cooldown(event.repository)

        # Create debate task
        task = asyncio.create_task(self._run_debate(event))
        self._pending_debates[event.id] = task

        # Clean up task when done
        task.add_done_callback(lambda t: self._pending_debates.pop(event.id, None))

        logger.info(
            f"[security_dispatcher] Triggered debate for event {event.id} "
            f"(type={event.event_type.value}, severity={event.severity.value}, "
            f"findings={len(event.findings)})"
        )

    async def _run_debate(self, event: SecurityEvent) -> Optional[str]:
        """
        Run the actual debate for a security event.

        Args:
            event: The security event to debate

        Returns:
            The debate ID if successful, None otherwise
        """
        try:
            # Use custom callback if provided
            if self._custom_trigger_callback:
                debate_id = await self._custom_trigger_callback(event)
            else:
                # Use Arena.run_security_debate
                from aragora.debate.orchestrator import Arena

                result = await Arena.run_security_debate(
                    event=event,
                    confidence_threshold=self.config.debate_confidence_threshold,
                    timeout_seconds=self.config.debate_timeout_seconds,
                )
                debate_id = result.debate_id

            self._stats.debates_completed += 1
            return debate_id

        except asyncio.CancelledError:
            logger.info(f"Debate for event {event.id} was cancelled")
            return None

        except Exception as e:
            logger.exception(f"Debate for event {event.id} failed: {e}")
            self._stats.debates_failed += 1
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dispatcher statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "events_received": self._stats.events_received,
            "events_filtered": self._stats.events_filtered,
            "debates_triggered": self._stats.debates_triggered,
            "debates_completed": self._stats.debates_completed,
            "debates_failed": self._stats.debates_failed,
            "debates_pending": len(self._pending_debates),
            "last_event_time": (
                self._stats.last_event_time.isoformat() if self._stats.last_event_time else None
            ),
            "last_debate_time": (
                self._stats.last_debate_time.isoformat() if self._stats.last_debate_time else None
            ),
            "config": {
                "min_severity": self.config.min_severity.value,
                "max_concurrent_debates": self.config.max_concurrent_debates,
                "repository_cooldown_seconds": self.config.repository_cooldown_seconds,
            },
        }

    def get_pending_debates(self) -> List[str]:
        """Get list of pending debate event IDs."""
        return list(self._pending_debates.keys())

    @property
    def is_running(self) -> bool:
        """Check if the dispatcher is running."""
        return self._running


# =============================================================================
# Global Dispatcher Instance
# =============================================================================

_dispatcher: Optional[SecurityDispatcher] = None


def get_security_dispatcher() -> SecurityDispatcher:
    """
    Get the global security dispatcher instance.

    Returns:
        The global SecurityDispatcher instance
    """
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = SecurityDispatcher()
    return _dispatcher


def set_security_dispatcher(dispatcher: SecurityDispatcher) -> None:
    """
    Set the global security dispatcher instance.

    Args:
        dispatcher: The SecurityDispatcher instance to use
    """
    global _dispatcher
    _dispatcher = dispatcher


async def start_security_dispatcher(
    config: Optional[DispatcherConfig] = None,
) -> SecurityDispatcher:
    """
    Initialize and start the global security dispatcher.

    Convenience function for application startup.

    Args:
        config: Optional dispatcher configuration

    Returns:
        The started SecurityDispatcher instance
    """
    global _dispatcher

    if _dispatcher is None:
        _dispatcher = SecurityDispatcher(config=config)

    await _dispatcher.start()
    return _dispatcher


async def stop_security_dispatcher() -> None:
    """
    Stop the global security dispatcher.

    Convenience function for application shutdown.
    """
    global _dispatcher

    if _dispatcher is not None:
        await _dispatcher.stop()


__all__ = [
    # Main class
    "SecurityDispatcher",
    "DispatcherConfig",
    "DispatcherStats",
    # Global instance functions
    "get_security_dispatcher",
    "set_security_dispatcher",
    "start_security_dispatcher",
    "stop_security_dispatcher",
    # Type alias
    "DebateTriggerCallback",
]
