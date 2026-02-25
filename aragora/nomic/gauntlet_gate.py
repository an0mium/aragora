"""
Gauntlet Approval Gate for the Nomic Loop.

Runs a lightweight Gauntlet benchmark as a verification gate during
the Nomic Loop's verify phase. If the Gauntlet produces CRITICAL or HIGH
severity findings, the approval is blocked.

This gate is opt-in (disabled by default) and designed to run in a
lightweight mode -- skipping the full scenario matrix and using minimal
attack/probe rounds -- to avoid slowing down the self-improvement cycle.

Usage in autonomous orchestrator:
    orchestrator = AutonomousOrchestrator(
        enable_gauntlet_gate=True,  # existing flag
    )

Usage standalone:
    gate = GauntletApprovalGate()
    result = await gate.evaluate(
        content="The design spec or implementation diff",
        context="Additional context about the change",
    )
    if result.blocked:
        print(f"Blocked: {result.reason}")
        for f in result.blocking_findings:
            print(f"  - [{f.severity}] {f.title}")
"""

from __future__ import annotations

__all__ = [
    "GauntletApprovalGate",
    "GauntletGateConfig",
    "GauntletGateResult",
]

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GauntletGateConfig:
    """Configuration for the Gauntlet approval gate.

    Attributes:
        enabled: Whether the gate is active (default False).
        max_critical: Maximum number of CRITICAL findings before blocking.
            Default 0 means any critical finding blocks.
        max_high: Maximum number of HIGH findings before blocking.
            Default 0 means any high finding blocks.
        attack_rounds: Number of red-team attack rounds (1 for lightweight).
        probes_per_category: Number of probes per category (1 for lightweight).
        run_scenario_matrix: Whether to run the scenario matrix (False for lightweight).
        timeout_seconds: Maximum time for the Gauntlet run.
        agents: Agent names to use for the Gauntlet. Empty list uses defaults.
    """

    enabled: bool = False
    max_critical: int = 0
    max_high: int = 0
    attack_rounds: int = 1
    probes_per_category: int = 1
    run_scenario_matrix: bool = False
    timeout_seconds: int = 120
    agents: list[str] = field(default_factory=lambda: ["anthropic-api", "openai-api"])


@dataclass
class BlockingFinding:
    """A finding that contributed to blocking approval."""

    severity: str
    title: str
    description: str
    category: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "source": self.source,
        }


@dataclass
class GauntletGateResult:
    """Result of a Gauntlet approval gate evaluation.

    Attributes:
        blocked: Whether the gate blocked the approval.
        reason: Human-readable reason for the decision.
        critical_count: Number of CRITICAL findings.
        high_count: Number of HIGH findings.
        total_findings: Total number of findings across all severities.
        blocking_findings: List of findings that caused the block.
        gauntlet_id: ID of the Gauntlet run for traceability.
        duration_seconds: How long the Gauntlet run took.
        skipped: True if the gate was skipped (e.g., disabled or import error).
        error: Error message if the gate failed to run.
    """

    blocked: bool = False
    reason: str = ""
    critical_count: int = 0
    high_count: int = 0
    total_findings: int = 0
    blocking_findings: list[BlockingFinding] = field(default_factory=list)
    gauntlet_id: str = ""
    duration_seconds: float = 0.0
    skipped: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Convenience: True when the gate did not block."""
        return not self.blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "passed": self.passed,
            "reason": self.reason,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "total_findings": self.total_findings,
            "blocking_findings": [f.to_dict() for f in self.blocking_findings],
            "gauntlet_id": self.gauntlet_id,
            "duration_seconds": self.duration_seconds,
            "skipped": self.skipped,
            "error": self.error,
        }


class GauntletApprovalGate:
    """Runs a lightweight Gauntlet benchmark as a verification gate.

    The gate creates a GauntletRunner with a minimal configuration
    (few attack rounds, minimal probes, no scenario matrix) and evaluates
    the findings against configurable thresholds.

    If the number of CRITICAL or HIGH severity findings exceeds the
    configured thresholds, the gate blocks the approval.
    """

    def __init__(self, config: GauntletGateConfig | None = None):
        self.config = config or GauntletGateConfig()

    async def evaluate(
        self,
        content: str,
        context: str = "",
    ) -> GauntletGateResult:
        """Run the Gauntlet and evaluate findings against thresholds.

        Args:
            content: The content to validate (design spec, implementation diff,
                or description of changes).
            context: Additional context for the validation.

        Returns:
            GauntletGateResult indicating whether the gate passed or blocked.
        """
        if not self.config.enabled:
            return GauntletGateResult(
                blocked=False,
                reason="Gauntlet gate disabled",
                skipped=True,
            )

        try:
            from aragora.gauntlet.config import (
                AttackCategory,
                GauntletConfig,
                ProbeCategory,
            )
            from aragora.gauntlet.runner import GauntletRunner
            from aragora.gauntlet.result import SeverityLevel
        except ImportError as e:
            logger.debug("Gauntlet gate skipped: gauntlet module unavailable: %s", e)
            return GauntletGateResult(
                blocked=False,
                reason="Gauntlet module not available",
                skipped=True,
                error=str(e),
            )

        # Build lightweight Gauntlet configuration
        gauntlet_config = GauntletConfig(
            name="Nomic Loop Approval Gate",
            description="Lightweight adversarial validation for self-improvement gate",
            attack_categories=[
                AttackCategory.SECURITY,
                AttackCategory.LOGIC,
            ],
            attack_rounds=self.config.attack_rounds,
            attacks_per_category=2,
            probe_categories=[
                ProbeCategory.CONTRADICTION,
                ProbeCategory.HALLUCINATION,
            ],
            probes_per_category=self.config.probes_per_category,
            run_scenario_matrix=self.config.run_scenario_matrix,
            enable_scenario_analysis=self.config.run_scenario_matrix,
            agents=self.config.agents,
            max_agents=2,
            critical_threshold=self.config.max_critical,
            high_threshold=self.config.max_high,
            timeout_seconds=self.config.timeout_seconds,
        )

        runner = GauntletRunner(config=gauntlet_config)

        try:
            gauntlet_result = await runner.run(
                input_content=content,
                context=context,
            )
        except (RuntimeError, ValueError, TimeoutError, OSError) as e:
            logger.warning("Gauntlet gate run failed: %s", e)
            return GauntletGateResult(
                blocked=False,
                reason=f"Gauntlet run failed (non-blocking): {type(e).__name__}",
                skipped=True,
                error=str(e),
            )

        # Count findings by severity
        critical_count = gauntlet_result.risk_summary.critical
        high_count = gauntlet_result.risk_summary.high
        total_findings = len(gauntlet_result.vulnerabilities)

        # Collect blocking findings
        blocking_findings: list[BlockingFinding] = []
        for vuln in gauntlet_result.vulnerabilities:
            if vuln.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
                blocking_findings.append(
                    BlockingFinding(
                        severity=vuln.severity.value,
                        title=vuln.title,
                        description=vuln.description[:300],
                        category=vuln.category,
                        source=vuln.source,
                    )
                )

        # Evaluate thresholds
        blocked = False
        reasons: list[str] = []

        if critical_count > self.config.max_critical:
            blocked = True
            reasons.append(
                f"{critical_count} CRITICAL findings (threshold: {self.config.max_critical})"
            )

        if high_count > self.config.max_high:
            blocked = True
            reasons.append(f"{high_count} HIGH findings (threshold: {self.config.max_high})")

        if blocked:
            reason = "Gauntlet gate BLOCKED: " + "; ".join(reasons)
        else:
            reason = (
                f"Gauntlet gate passed ({total_findings} total findings, "
                f"{critical_count} critical, {high_count} high)"
            )

        logger.info(
            "gauntlet_gate_result blocked=%s critical=%d high=%d total=%d",
            blocked,
            critical_count,
            high_count,
            total_findings,
        )

        return GauntletGateResult(
            blocked=blocked,
            reason=reason,
            critical_count=critical_count,
            high_count=high_count,
            total_findings=total_findings,
            blocking_findings=blocking_findings,
            gauntlet_id=gauntlet_result.gauntlet_id,
            duration_seconds=gauntlet_result.duration_seconds,
        )
