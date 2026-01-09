"""
Transparent Immune System - Real-time health broadcasting.

Turns system failures into audience entertainment by:
- Broadcasting health events via WebSocket
- Progressive timeout escalation with transparency
- Structured failure events as API
- Making debugging visible and participatory

Inspired by nomic loop debate synthesis on resilience.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"          # All systems nominal
    DEGRADED = "degraded"        # Some issues but functioning
    STRESSED = "stressed"        # Performance impacted
    CRITICAL = "critical"        # Major failures occurring
    RECOVERING = "recovering"    # Coming back from failure


class AgentStatus(Enum):
    """Individual agent status."""

    IDLE = "idle"
    THINKING = "thinking"
    RESPONDING = "responding"
    TIMEOUT = "timeout"
    FAILED = "failed"
    RECOVERED = "recovered"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class HealthEvent:
    """A health event to broadcast."""

    timestamp: float
    event_type: str
    status: str
    component: str
    message: str
    details: dict
    audience_message: Optional[str] = None  # Human-friendly message

    def to_dict(self) -> dict:
        return asdict(self)

    def to_broadcast(self) -> dict:
        """Format for WebSocket broadcast."""
        return {
            "type": "health_event",
            "data": self.to_dict(),
        }


@dataclass
class AgentHealthState:
    """Health state for a single agent."""

    name: str
    status: AgentStatus = AgentStatus.IDLE
    last_response_time: float = 0.0
    consecutive_failures: int = 0
    total_timeouts: int = 0
    avg_response_ms: float = 0.0
    circuit_open: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_response_time": self.last_response_time,
            "consecutive_failures": self.consecutive_failures,
            "total_timeouts": self.total_timeouts,
            "avg_response_ms": round(self.avg_response_ms, 2),
            "circuit_open": self.circuit_open,
        }


class TransparentImmuneSystem:
    """
    Real-time health monitoring and broadcasting system.

    Provides transparency into system health by broadcasting events
    to connected WebSocket clients. Turns failures into entertainment
    rather than hiding them.

    Usage:
        immune = TransparentImmuneSystem()
        immune.set_broadcast_callback(websocket_broadcast)

        # In debate loop
        immune.agent_started("claude")
        immune.agent_responding("claude", 1500)  # 1.5s so far
        immune.agent_timeout("claude", 90.0)
        immune.agent_recovered("claude", "fallback_response")
    """

    # Progressive timeout thresholds for escalating transparency
    TIMEOUT_STAGES = [
        (5, "Agent is thinking deeply..."),
        (15, "Taking longer than usual - complex analysis in progress"),
        (30, "Extended processing - we're working on it!"),
        (60, "This is taking a while - agent may need help"),
        (90, "Critical delay - considering alternatives"),
    ]

    def __init__(self):
        """Initialize the immune system."""
        self.agent_states: dict[str, AgentHealthState] = {}
        self.system_status = HealthStatus.HEALTHY
        self.event_history: list[HealthEvent] = []
        self.broadcast_callback: Optional[Callable] = None
        self.start_time = time.time()

        # Metrics
        self.total_events = 0
        self.total_failures = 0
        self.total_recoveries = 0

    def set_broadcast_callback(self, callback: Callable[[dict], Any]) -> None:
        """
        Set the callback for broadcasting events.

        Args:
            callback: Function that takes a dict and broadcasts it
                      (typically sends to WebSocket clients)
        """
        self.broadcast_callback = callback

    def _get_agent_state(self, agent_name: str) -> AgentHealthState:
        """Get or create agent state."""
        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = AgentHealthState(name=agent_name)
        return self.agent_states[agent_name]

    def _broadcast(self, event: HealthEvent) -> None:
        """Broadcast an event to listeners."""
        self.event_history.append(event)
        self.total_events += 1

        # Limit history size
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]

        if self.broadcast_callback:
            try:
                self.broadcast_callback(event.to_broadcast())
            except Exception as e:
                logger.error(f"immune_broadcast_failed error={e}")

        logger.debug(f"immune_event type={event.event_type} component={event.component}")

    def _update_system_status(self) -> None:
        """Update overall system status based on agent states."""
        if not self.agent_states:
            self.system_status = HealthStatus.HEALTHY
            return

        failed_count = sum(
            1 for s in self.agent_states.values()
            if s.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT, AgentStatus.CIRCUIT_OPEN)
        )
        total_agents = len(self.agent_states)

        if failed_count == 0:
            self.system_status = HealthStatus.HEALTHY
        elif failed_count == 1:
            self.system_status = HealthStatus.DEGRADED
        elif failed_count < total_agents / 2:
            self.system_status = HealthStatus.STRESSED
        else:
            self.system_status = HealthStatus.CRITICAL

    # Agent lifecycle events

    def agent_started(self, agent_name: str, task: str = "") -> None:
        """Called when an agent starts processing."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.THINKING
        state.last_response_time = time.time()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="agent_started",
            status=AgentStatus.THINKING.value,
            component=agent_name,
            message=f"Agent {agent_name} started processing",
            details={"task": task[:100] if task else ""},
            audience_message=f"{agent_name} is thinking...",
        ))

    def agent_progress(self, agent_name: str, elapsed_seconds: float) -> None:
        """
        Called periodically to report agent progress.

        Uses progressive escalation to keep audience informed.
        """
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.RESPONDING

        # Find appropriate message for elapsed time
        audience_msg = "Processing..."
        for threshold, msg in self.TIMEOUT_STAGES:
            if elapsed_seconds >= threshold:
                audience_msg = msg

        # Only broadcast at stage transitions (every 5-15 seconds)
        for threshold, _ in self.TIMEOUT_STAGES:
            if abs(elapsed_seconds - threshold) < 1.0:  # Within 1 second of threshold
                self._broadcast(HealthEvent(
                    timestamp=time.time(),
                    event_type="agent_progress",
                    status=AgentStatus.RESPONDING.value,
                    component=agent_name,
                    message=f"Agent {agent_name} still working ({elapsed_seconds:.0f}s)",
                    details={"elapsed_seconds": elapsed_seconds},
                    audience_message=audience_msg,
                ))
                break

    def agent_completed(
        self,
        agent_name: str,
        response_ms: float,
        success: bool = True,
    ) -> None:
        """Called when an agent completes successfully."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.IDLE
        state.consecutive_failures = 0

        # Update rolling average
        alpha = 0.3  # Smoothing factor
        state.avg_response_ms = (
            alpha * response_ms + (1 - alpha) * state.avg_response_ms
        )

        self._update_system_status()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="agent_completed",
            status=AgentStatus.IDLE.value,
            component=agent_name,
            message=f"Agent {agent_name} completed in {response_ms:.0f}ms",
            details={
                "response_ms": response_ms,
                "success": success,
            },
            audience_message=f"{agent_name} responded!",
        ))

    def agent_timeout(
        self,
        agent_name: str,
        timeout_seconds: float,
        context: Optional[dict] = None,
    ) -> None:
        """Called when an agent times out."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.TIMEOUT
        state.consecutive_failures += 1
        state.total_timeouts += 1
        self.total_failures += 1

        self._update_system_status()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="agent_timeout",
            status=AgentStatus.TIMEOUT.value,
            component=agent_name,
            message=f"Agent {agent_name} timed out after {timeout_seconds}s",
            details={
                "timeout_seconds": timeout_seconds,
                "consecutive_failures": state.consecutive_failures,
                "total_timeouts": state.total_timeouts,
                "context": context or {},
            },
            audience_message=f"{agent_name} is having trouble - we're finding alternatives!",
        ))

    def agent_failed(
        self,
        agent_name: str,
        error: str,
        recoverable: bool = True,
    ) -> None:
        """Called when an agent fails with an error."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.FAILED
        state.consecutive_failures += 1
        self.total_failures += 1

        self._update_system_status()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="agent_failed",
            status=AgentStatus.FAILED.value,
            component=agent_name,
            message=f"Agent {agent_name} encountered an error",
            details={
                "error": error[:200],
                "recoverable": recoverable,
                "consecutive_failures": state.consecutive_failures,
            },
            audience_message=f"{agent_name} hit a snag - working on recovery!",
        ))

    def agent_recovered(
        self,
        agent_name: str,
        recovery_method: str,
        details: Optional[dict] = None,
    ) -> None:
        """Called when an agent recovers from failure."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.RECOVERED
        self.total_recoveries += 1

        self._update_system_status()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="agent_recovered",
            status=AgentStatus.RECOVERED.value,
            component=agent_name,
            message=f"Agent {agent_name} recovered via {recovery_method}",
            details={
                "recovery_method": recovery_method,
                **(details or {}),
            },
            audience_message=f"{agent_name} is back in action!",
        ))

    def circuit_opened(self, agent_name: str, reason: str) -> None:
        """Called when circuit breaker opens for an agent."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.CIRCUIT_OPEN
        state.circuit_open = True

        self._update_system_status()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="circuit_opened",
            status=AgentStatus.CIRCUIT_OPEN.value,
            component=agent_name,
            message=f"Circuit breaker opened for {agent_name}",
            details={"reason": reason},
            audience_message=f"{agent_name} is taking a break to cool down",
        ))

    def circuit_closed(self, agent_name: str) -> None:
        """Called when circuit breaker closes for an agent."""
        state = self._get_agent_state(agent_name)
        state.status = AgentStatus.IDLE
        state.circuit_open = False
        state.consecutive_failures = 0

        self._update_system_status()

        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type="circuit_closed",
            status=AgentStatus.IDLE.value,
            component=agent_name,
            message=f"Circuit breaker closed for {agent_name}",
            details={},
            audience_message=f"{agent_name} is ready to rejoin!",
        ))

    # System-level events

    def system_event(
        self,
        event_type: str,
        message: str,
        details: Optional[dict] = None,
        audience_message: Optional[str] = None,
    ) -> None:
        """Broadcast a general system event."""
        self._broadcast(HealthEvent(
            timestamp=time.time(),
            event_type=event_type,
            status=self.system_status.value,
            component="system",
            message=message,
            details=details or {},
            audience_message=audience_message,
        ))

    def get_system_health(self) -> dict:
        """Get current system health summary."""
        return {
            "status": self.system_status.value,
            "uptime_seconds": time.time() - self.start_time,
            "total_events": self.total_events,
            "total_failures": self.total_failures,
            "total_recoveries": self.total_recoveries,
            "recovery_rate": self.total_recoveries / max(self.total_failures, 1),
            "agents": {
                name: state.to_dict()
                for name, state in self.agent_states.items()
            },
        }

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get recent health events."""
        return [e.to_dict() for e in self.event_history[-limit:]]


# Global instance for easy access
_immune_system: Optional[TransparentImmuneSystem] = None


def get_immune_system() -> TransparentImmuneSystem:
    """Get the global immune system instance."""
    global _immune_system
    if _immune_system is None:
        _immune_system = TransparentImmuneSystem()
    return _immune_system


def reset_immune_system() -> None:
    """Reset the global immune system (for testing)."""
    global _immune_system
    _immune_system = None
