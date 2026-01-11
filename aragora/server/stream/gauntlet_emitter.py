"""
Gauntlet Stream Emitter - Real-time event streaming for Gauntlet stress-tests.

Provides event emission during Gauntlet execution for WebSocket visualization.
Integrates with the existing debate streaming infrastructure.

Usage:
    emitter = GauntletStreamEmitter(broadcast_fn)
    orchestrator = GauntletOrchestrator(agents, event_emitter=emitter)
    result = await orchestrator.run(config)

Events emitted:
- gauntlet_start: Stress-test initiated
- gauntlet_phase: Phase transition (redteam, probe, audit, verify)
- gauntlet_agent_active: Agent became active
- gauntlet_attack: Red-team attack executed
- gauntlet_finding: New finding discovered
- gauntlet_probe: Capability probe result
- gauntlet_verification: Formal verification result
- gauntlet_risk: Risk assessment update
- gauntlet_progress: Progress percentage
- gauntlet_verdict: Final verdict
- gauntlet_complete: Stress-test completed
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from .events import StreamEventType, StreamEvent

if TYPE_CHECKING:
    from aragora.gauntlet import GauntletConfig, GauntletResult, Finding

logger = logging.getLogger(__name__)


@dataclass
class GauntletPhase:
    """Gauntlet execution phase."""
    INIT = "init"
    RISK_ASSESSMENT = "risk_assessment"
    REDTEAM = "redteam"
    PROBING = "probing"
    DEEP_AUDIT = "deep_audit"
    VERIFICATION = "verification"
    AGGREGATION = "aggregation"
    VERDICT = "verdict"
    COMPLETE = "complete"


class GauntletStreamEmitter:
    """
    Emits real-time events during Gauntlet execution.

    Integrates with the WebSocket streaming infrastructure to provide
    live updates to connected clients.
    """

    def __init__(
        self,
        broadcast_fn: Optional[Callable[[StreamEvent], None]] = None,
        gauntlet_id: Optional[str] = None,
    ):
        """
        Initialize the emitter.

        Args:
            broadcast_fn: Function to broadcast events to WebSocket clients.
                         If None, events are logged but not broadcast.
            gauntlet_id: Optional Gauntlet ID for event correlation.
        """
        self.broadcast_fn = broadcast_fn
        self.gauntlet_id = gauntlet_id or ""
        self._seq = 0
        self._start_time: Optional[float] = None
        self._phase = GauntletPhase.INIT
        self._finding_count = 0
        self._attack_count = 0
        self._probe_count = 0

    def _next_seq(self) -> int:
        """Get next sequence number."""
        self._seq += 1
        return self._seq

    def _emit(self, event_type: StreamEventType, data: dict, agent: str = "") -> None:
        """Emit an event."""
        event = StreamEvent(
            type=event_type,
            data=data,
            timestamp=time.time(),
            agent=agent,
            loop_id=self.gauntlet_id,
            seq=self._next_seq(),
        )

        if self.broadcast_fn:
            try:
                self.broadcast_fn(event)
            except Exception as e:
                logger.debug(f"Failed to broadcast event: {e}")

        logger.debug(f"Gauntlet event: {event_type.value} - {data.get('message', '')}")

    def set_gauntlet_id(self, gauntlet_id: str) -> None:
        """Set the Gauntlet ID for event correlation."""
        self.gauntlet_id = gauntlet_id

    # Lifecycle events

    def emit_start(
        self,
        gauntlet_id: str,
        input_type: str,
        input_summary: str,
        agents: list[str],
        config_summary: dict,
    ) -> None:
        """Emit gauntlet_start event."""
        self.gauntlet_id = gauntlet_id
        self._start_time = time.time()
        self._phase = GauntletPhase.INIT

        self._emit(
            StreamEventType.GAUNTLET_START,
            {
                "gauntlet_id": gauntlet_id,
                "input_type": input_type,
                "input_summary": input_summary[:500],
                "agents": agents,
                "config": config_summary,
                "message": f"Gauntlet stress-test started with {len(agents)} agents",
            },
        )

    def emit_complete(
        self,
        gauntlet_id: str,
        verdict: str,
        confidence: float,
        findings_count: int,
        duration_seconds: float,
    ) -> None:
        """Emit gauntlet_complete event."""
        self._phase = GauntletPhase.COMPLETE

        self._emit(
            StreamEventType.GAUNTLET_COMPLETE,
            {
                "gauntlet_id": gauntlet_id,
                "verdict": verdict,
                "confidence": confidence,
                "findings_count": findings_count,
                "attacks_run": self._attack_count,
                "probes_run": self._probe_count,
                "duration_seconds": duration_seconds,
                "message": f"Gauntlet complete: {verdict} ({confidence:.0%} confidence)",
            },
        )

    # Phase events

    def emit_phase(self, phase: str, message: str = "") -> None:
        """Emit gauntlet_phase event."""
        self._phase = phase
        elapsed = time.time() - self._start_time if self._start_time else 0

        self._emit(
            StreamEventType.GAUNTLET_PHASE,
            {
                "gauntlet_id": self.gauntlet_id,
                "phase": phase,
                "elapsed_seconds": elapsed,
                "message": message or f"Entering phase: {phase}",
            },
        )

    def emit_progress(self, progress: float, phase: str = "", message: str = "") -> None:
        """Emit gauntlet_progress event."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        self._emit(
            StreamEventType.GAUNTLET_PROGRESS,
            {
                "gauntlet_id": self.gauntlet_id,
                "progress": progress,  # 0.0 - 1.0
                "phase": phase or self._phase,
                "elapsed_seconds": elapsed,
                "findings_count": self._finding_count,
                "attacks_run": self._attack_count,
                "probes_run": self._probe_count,
                "message": message or f"Progress: {progress:.0%}",
            },
        )

    # Agent events

    def emit_agent_active(self, agent: str, role: str) -> None:
        """Emit gauntlet_agent_active event."""
        self._emit(
            StreamEventType.GAUNTLET_AGENT_ACTIVE,
            {
                "gauntlet_id": self.gauntlet_id,
                "agent": agent,
                "role": role,
                "message": f"Agent {agent} active as {role}",
            },
            agent=agent,
        )

    # Attack events

    def emit_attack(
        self,
        attack_type: str,
        agent: str,
        target_summary: str,
        success: bool,
        severity: Optional[float] = None,
    ) -> None:
        """Emit gauntlet_attack event."""
        self._attack_count += 1

        self._emit(
            StreamEventType.GAUNTLET_ATTACK,
            {
                "gauntlet_id": self.gauntlet_id,
                "attack_type": attack_type,
                "agent": agent,
                "target_summary": target_summary[:200],
                "success": success,
                "severity": severity,
                "attack_number": self._attack_count,
                "message": f"Attack: {attack_type} by {agent}",
            },
            agent=agent,
        )

    # Finding events

    def emit_finding(
        self,
        finding_id: str,
        severity: str,
        category: str,
        title: str,
        description: str,
        source: str,
    ) -> None:
        """Emit gauntlet_finding event."""
        self._finding_count += 1

        self._emit(
            StreamEventType.GAUNTLET_FINDING,
            {
                "gauntlet_id": self.gauntlet_id,
                "finding_id": finding_id,
                "severity": severity,
                "category": category,
                "title": title,
                "description": description[:300],
                "source": source,
                "finding_number": self._finding_count,
                "message": f"[{severity}] {title}",
            },
        )

    # Probe events

    def emit_probe(
        self,
        probe_type: str,
        agent: str,
        vulnerability_found: bool,
        severity: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Emit gauntlet_probe event."""
        self._probe_count += 1

        self._emit(
            StreamEventType.GAUNTLET_PROBE,
            {
                "gauntlet_id": self.gauntlet_id,
                "probe_type": probe_type,
                "agent": agent,
                "vulnerability_found": vulnerability_found,
                "severity": severity,
                "description": description[:200] if description else None,
                "probe_number": self._probe_count,
                "message": f"Probe: {probe_type} - {'vulnerable' if vulnerability_found else 'passed'}",
            },
            agent=agent,
        )

    # Verification events

    def emit_verification(
        self,
        claim: str,
        verified: bool,
        method: str,
        proof_hash: Optional[str] = None,
    ) -> None:
        """Emit gauntlet_verification event."""
        self._emit(
            StreamEventType.GAUNTLET_VERIFICATION,
            {
                "gauntlet_id": self.gauntlet_id,
                "claim": claim[:200],
                "verified": verified,
                "method": method,
                "proof_hash": proof_hash,
                "message": f"Verification: {'proved' if verified else 'failed'} via {method}",
            },
        )

    # Risk events

    def emit_risk(
        self,
        risk_type: str,
        level: str,
        description: str,
        confidence: float,
    ) -> None:
        """Emit gauntlet_risk event."""
        self._emit(
            StreamEventType.GAUNTLET_RISK,
            {
                "gauntlet_id": self.gauntlet_id,
                "risk_type": risk_type,
                "level": level,
                "description": description[:200],
                "confidence": confidence,
                "message": f"Risk: {level} - {risk_type}",
            },
        )

    # Verdict event

    def emit_verdict(
        self,
        verdict: str,
        confidence: float,
        risk_score: float,
        robustness_score: float,
        critical_count: int,
        high_count: int,
        medium_count: int,
        low_count: int,
    ) -> None:
        """Emit gauntlet_verdict event."""
        self._phase = GauntletPhase.VERDICT

        self._emit(
            StreamEventType.GAUNTLET_VERDICT,
            {
                "gauntlet_id": self.gauntlet_id,
                "verdict": verdict,
                "confidence": confidence,
                "risk_score": risk_score,
                "robustness_score": robustness_score,
                "findings": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "low": low_count,
                    "total": critical_count + high_count + medium_count + low_count,
                },
                "message": f"Verdict: {verdict} ({confidence:.0%} confidence)",
            },
        )


def create_gauntlet_emitter(
    broadcast_fn: Optional[Callable[[StreamEvent], None]] = None,
) -> GauntletStreamEmitter:
    """
    Create a GauntletStreamEmitter.

    Args:
        broadcast_fn: Function to broadcast events to WebSocket clients.

    Returns:
        Configured GauntletStreamEmitter instance.
    """
    return GauntletStreamEmitter(broadcast_fn=broadcast_fn)
