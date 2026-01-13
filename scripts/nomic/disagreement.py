"""
Disagreement handling for nomic loop debates.

Analyzes disagreement patterns from debates and determines appropriate
actions: rejection, forking, escalation, or proceeding with caution.

Heavy3-inspired: Makes disagreement data actionable rather than just logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DisagreementActions:
    """Actions recommended based on disagreement analysis."""

    should_reject: bool = False
    should_fork: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    fork_topic: Optional[str] = None
    escalate_to: Optional[str] = None


@dataclass
class CriticalDisagreement:
    """Record of a critical disagreement for later review."""

    phase: str
    cycle: int
    unanimous_critiques: list[str]
    agreement_score: float
    actions_taken: dict
    timestamp: str


class DisagreementHandler:
    """
    Handles disagreement patterns to influence debate decisions.

    Analyzes DisagreementReport from debates and determines:
    - Whether to reject proposals (unanimous issues)
    - Whether to fork debates (low agreement)
    - Whether to escalate to cross-examination
    - Whether to proceed with caution

    Usage:
        handler = DisagreementHandler(log_fn=loop._log, stream_emit=loop._stream_emit)

        result = await arena.run()
        if result.disagreement_report:
            actions = handler.handle(
                report=result.disagreement_report,
                phase_name="design",
                cycle=3,
            )
            if actions.should_reject:
                # Handle rejection
            elif actions.should_fork:
                # Fork the debate
    """

    # Thresholds
    UNANIMOUS_REJECT_THRESHOLD = 3  # >= 3 unanimous issues = reject
    FORK_AGREEMENT_THRESHOLD = 0.4  # < 40% agreement = fork
    ESCALATE_SPLIT_THRESHOLD = 3  # >= 3 split opinions = escalate

    def __init__(
        self,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit: Optional[Callable] = None,
    ):
        """
        Initialize disagreement handler.

        Args:
            log_fn: Optional logging function
            stream_emit: Optional stream event emitter
        """
        self._log = log_fn or (lambda msg: logger.info(msg))
        self._stream_emit = stream_emit or (lambda *args, **kwargs: None)

        # Track patterns across cycles
        self._agent_disagreement_patterns: dict[str, list] = {}
        self._critical_disagreements: list[CriticalDisagreement] = []

    def handle(
        self,
        report: Any,  # DisagreementReport
        phase_name: str,
        cycle: int = 0,
    ) -> DisagreementActions:
        """
        Handle disagreement patterns to influence decisions.

        Args:
            report: DisagreementReport from the debate
            phase_name: Current phase ("debate", "design", "implement")
            cycle: Current cycle number

        Returns:
            DisagreementActions with recommendations
        """
        actions = DisagreementActions()
        critical_warning = False

        # ACTION 1: Auto-reject on unanimous critiques
        if len(report.unanimous_critiques) >= self.UNANIMOUS_REJECT_THRESHOLD:
            self._log(
                f"    [disagreement] REJECT: {len(report.unanimous_critiques)} "
                "unanimous issues - proposal blocked"
            )
            actions.should_reject = True
            actions.rejection_reasons = report.unanimous_critiques[:5]
            critical_warning = True

        # ACTION 2: Fork trigger for low agreement
        if report.agreement_score < self.FORK_AGREEMENT_THRESHOLD and not actions.should_reject:
            self._log(
                f"    [disagreement] FORK: Low agreement ({report.agreement_score:.0%}) "
                "- exploring alternatives"
            )
            actions.should_fork = True

            # Create fork topic from main disagreement
            if report.split_opinions:
                first_split = (
                    report.split_opinions[0]
                    if isinstance(report.split_opinions, list)
                    else str(report.split_opinions)
                )
                actions.fork_topic = f"Resolve disagreement: {first_split[:200]}"
            critical_warning = True

        # ACTION 3: Escalate split opinions
        if len(report.split_opinions) >= self.ESCALATE_SPLIT_THRESHOLD:
            self._log(
                f"    [disagreement] ESCALATE: {len(report.split_opinions)} "
                "split opinions detected"
            )
            actions.escalate_to = "cross_examination"
            for opinion in report.split_opinions[:3]:
                self._log(f"      Split: {str(opinion)[:100]}")

        # Warn on low agreement
        if report.agreement_score < self.FORK_AGREEMENT_THRESHOLD and not actions.should_reject:
            self._log(
                f"    [disagreement] WARNING: Low agreement ({report.agreement_score:.0%}) "
                "- consider revising proposal"
            )
            critical_warning = True

        # High-stakes phase logging
        if phase_name in ("design", "implement") and (
            len(report.unanimous_critiques) >= 2 or report.agreement_score < 0.5
        ):
            self._log(
                f"    [disagreement] ATTENTION: High-stakes phase '{phase_name}' "
                "has significant disagreement"
            )
            self._critical_disagreements.append(
                CriticalDisagreement(
                    phase=phase_name,
                    cycle=cycle,
                    unanimous_critiques=report.unanimous_critiques,
                    agreement_score=report.agreement_score,
                    actions_taken={
                        "should_reject": actions.should_reject,
                        "should_fork": actions.should_fork,
                        "escalate_to": actions.escalate_to,
                    },
                    timestamp=datetime.now().isoformat(),
                )
            )

        # Log risk areas
        if len(report.risk_areas) >= 2:
            self._log(f"    [disagreement] {len(report.risk_areas)} RISK AREAS to monitor:")
            for risk in report.risk_areas[:3]:
                self._log(f"      - {risk[:100]}")

        # Stream critical warnings
        if critical_warning:
            action_str = ""
            if actions.should_reject:
                action_str = " [REJECTED]"
            elif actions.should_fork:
                action_str = " [FORKING]"

            self._stream_emit(
                "on_log_message",
                f"Disagreement alert in {phase_name}: "
                f"{len(report.unanimous_critiques)} unanimous issues, "
                f"{report.agreement_score:.0%} agreement{action_str}",
                level="warning",
                phase=phase_name,
            )

        return actions

    def get_critical_disagreements(self) -> list[CriticalDisagreement]:
        """Get all recorded critical disagreements."""
        return self._critical_disagreements.copy()

    def clear_history(self) -> None:
        """Clear disagreement history (e.g., at cycle start)."""
        self._agent_disagreement_patterns.clear()


__all__ = [
    "DisagreementActions",
    "CriticalDisagreement",
    "DisagreementHandler",
]
