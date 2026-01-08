"""
Red Team ELO Integration.

Adjusts agent ELO ratings based on security vulnerability assessments.
Extracted from EloSystem to separate security concerns from competitive ranking.

Usage:
    integrator = RedTeamIntegrator(elo_system)
    elo_change = integrator.record_result("claude",
        robustness_score=0.85,
        successful_attacks=2,
        total_attacks=10,
    )
    summary = integrator.get_vulnerability_summary("claude")
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from aragora.config import ELO_K_FACTOR

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)

K_FACTOR = ELO_K_FACTOR


@dataclass
class RedTeamResult:
    """Result of a red team assessment."""

    agent_name: str
    robustness_score: float
    successful_attacks: int
    total_attacks: int
    critical_vulnerabilities: int
    session_id: Optional[str]
    elo_change: float


@dataclass
class VulnerabilitySummary:
    """Summary of an agent's red team history."""

    redteam_sessions: int
    total_elo_impact: float
    last_session: Optional[str]


class RedTeamIntegrator:
    """
    Integrates red team results with ELO rankings.

    Adjusts agent ELO ratings based on vulnerability assessments:
    - Robust agents (robustness >= 0.8) get a small boost (+5 to +10)
    - Neutral range (0.5-0.8) has no effect
    - Vulnerable agents (robustness < 0.5) get penalized (-5 to -30)

    Usage:
        integrator = RedTeamIntegrator(elo_system)

        # Record a red team result
        elo_change = integrator.record_result("claude",
            robustness_score=0.85,
            successful_attacks=2,
            total_attacks=10,
            critical_vulnerabilities=0,
            session_id="rt_20240115",
        )

        # Get vulnerability history
        summary = integrator.get_vulnerability_summary("claude")
    """

    def __init__(self, elo_system: "EloSystem"):
        """
        Initialize the red team integrator.

        Args:
            elo_system: EloSystem instance for rating updates
        """
        self.elo_system = elo_system

    def record_result(
        self,
        agent_name: str,
        robustness_score: float,
        successful_attacks: int,
        total_attacks: int,
        critical_vulnerabilities: int = 0,
        session_id: Optional[str] = None,
    ) -> float:
        """
        Record red team results and adjust ELO based on vulnerability.

        The robustness score (0-1) affects ELO:
        - robustness >= 0.8: Small ELO boost (+5 to +10)
        - robustness 0.5-0.8: No change
        - robustness < 0.5: ELO penalty (-5 to -20 based on critical vulns)

        Args:
            agent_name: Agent that was red-teamed
            robustness_score: Overall robustness (0-1, higher is better)
            successful_attacks: Number of attacks that succeeded
            total_attacks: Total attacks attempted
            critical_vulnerabilities: Count of critical severity issues
            session_id: Optional red team session ID

        Returns:
            ELO change applied
        """
        rating = self.elo_system.get_rating(agent_name)
        elo_change = 0.0

        # Calculate vulnerability rate
        vulnerability_rate = successful_attacks / total_attacks if total_attacks > 0 else 0

        # Robust agents get a boost
        if robustness_score >= 0.8:
            elo_change = K_FACTOR * 0.3 * robustness_score  # +5 to +10

        # Vulnerable agents get penalized
        elif robustness_score < 0.5:
            # Base penalty from vulnerability rate
            base_penalty = K_FACTOR * 0.5 * vulnerability_rate  # Up to -16

            # Additional penalty for critical vulnerabilities
            critical_penalty = critical_vulnerabilities * 2  # -2 per critical

            elo_change = -(base_penalty + critical_penalty)
            elo_change = max(elo_change, -30)  # Cap at -30

        # Apply the change
        if elo_change != 0:
            rating.elo += elo_change
            rating.updated_at = datetime.now().isoformat()
            self.elo_system._save_rating(rating)
            self.elo_system._record_elo_history(
                agent_name,
                rating.elo,
                f"redteam_{session_id}" if session_id else "redteam",
            )

        return elo_change

    def get_vulnerability_summary(self, agent_name: str) -> VulnerabilitySummary:
        """
        Get summary of agent's red team history from ELO adjustments.

        Args:
            agent_name: Agent to query

        Returns:
            VulnerabilitySummary with session count, total impact, and last session
        """
        history = self.elo_system.get_elo_history(agent_name, limit=100)

        redteam_sessions = 0
        total_impact = 0.0
        last_session: Optional[str] = None

        prev_elo: Optional[float] = None
        for timestamp, elo in reversed(history):
            # Note: timestamp here is actually the debate_id from elo_history
            # Red team entries have "redteam" in their debate_id
            if "redteam" in str(timestamp):
                redteam_sessions += 1
                if prev_elo is not None:
                    total_impact += elo - prev_elo
                if last_session is None:
                    last_session = timestamp
            prev_elo = elo

        return VulnerabilitySummary(
            redteam_sessions=redteam_sessions,
            total_elo_impact=round(total_impact, 1),
            last_session=last_session,
        )

    def calculate_elo_adjustment(
        self,
        robustness_score: float,
        vulnerability_rate: float,
        critical_vulnerabilities: int = 0,
    ) -> float:
        """
        Calculate ELO adjustment without applying it.

        Useful for previewing the impact of a red team result.

        Args:
            robustness_score: Overall robustness (0-1)
            vulnerability_rate: Successful attacks / total attacks
            critical_vulnerabilities: Number of critical issues

        Returns:
            Projected ELO change
        """
        if robustness_score >= 0.8:
            return K_FACTOR * 0.3 * robustness_score

        if robustness_score < 0.5:
            base_penalty = K_FACTOR * 0.5 * vulnerability_rate
            critical_penalty = critical_vulnerabilities * 2
            return max(-(base_penalty + critical_penalty), -30)

        return 0.0
