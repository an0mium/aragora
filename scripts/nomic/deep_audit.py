"""
Deep audit functionality for nomic loop.

Provides intensive multi-round review for critical topics and protected
file changes. Implements Heavy3-inspired cross-examination patterns.

Audit types:
- Design audit: Strategic review of architectural proposals
- Protected file audit: Intensive review of changes to core files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Keywords that trigger deep audit
CRITICAL_KEYWORDS = [
    "architecture",
    "security",
    "authentication",
    "authorization",
    "database",
    "migration",
    "breaking change",
    "api contract",
    "consensus",
    "voting",
    "protocol",
    "protected file",
]

STRATEGY_KEYWORDS = [
    "strategy",
    "design pattern",
    "refactor",
    "restructure",
    "system design",
    "infrastructure",
    "scale",
    "performance",
]


@dataclass
class AuditResult:
    """Result of a deep audit."""

    approved: bool
    confidence: float = 0.0
    unanimous_issues: list[str] | None = None
    split_opinions: list[str] | None = None
    risk_areas: list[str] | None = None
    message: str = ""

    def __post_init__(self):
        self.unanimous_issues = self.unanimous_issues or []
        self.split_opinions = self.split_opinions or []
        self.risk_areas = self.risk_areas or []


class DeepAuditRunner:
    """
    Runs intensive multi-round audits for critical topics.

    Uses Deep Audit Mode with cross-examination for:
    - Strategic design proposals
    - Protected file changes

    Usage:
        runner = DeepAuditRunner(
            agents=[gemini, claude, codex, grok, deepseek],
            run_deep_audit_fn=run_deep_audit,
            audit_config=CODE_ARCHITECTURE_AUDIT,
            log_fn=loop._log,
        )

        # Check if topic needs audit
        if runner.should_use_audit(topic, phase="design"):
            result = await runner.audit_design(improvement, design_context)
            if not result.approved:
                # Handle rejection
    """

    def __init__(
        self,
        agents: list[Any] | None = None,
        run_deep_audit_fn: Optional[Callable] = None,
        audit_config: Any = None,
        log_fn: Optional[Callable[[str], None]] = None,
        deep_audit_available: bool = False,
    ):
        """
        Initialize deep audit runner.

        Args:
            agents: List of agents to use for audits
            run_deep_audit_fn: The run_deep_audit async function
            audit_config: CODE_ARCHITECTURE_AUDIT or similar config
            log_fn: Optional logging function
            deep_audit_available: Whether deep audit is available
        """
        self.agents = agents or []
        self._run_deep_audit = run_deep_audit_fn
        self._audit_config = audit_config
        self._log = log_fn or (lambda msg: logger.info(msg))
        self._available = deep_audit_available and run_deep_audit_fn is not None

    @property
    def is_available(self) -> bool:
        """Whether deep audit is available."""
        return self._available

    def should_use_audit(self, topic: str, phase: str = "design") -> tuple[bool, str]:
        """
        Determine if a topic warrants deep audit mode.

        Args:
            topic: The topic/improvement description
            phase: Current phase ("design", "implement", etc.)

        Returns:
            (should_use: bool, reason: str)
        """
        if not self._available:
            return False, "Deep audit not available"

        topic_lower = topic.lower()

        # Check for critical topics
        for keyword in CRITICAL_KEYWORDS:
            if keyword in topic_lower:
                return True, f"Critical topic detected: {keyword}"

        # Check for strategy topics in design phase
        if phase == "design":
            for keyword in STRATEGY_KEYWORDS:
                if keyword in topic_lower:
                    return True, f"Strategy topic detected: {keyword}"

        # Check topic length/complexity
        if len(topic) > 500:
            return True, "Complex topic (length > 500 chars)"

        return False, "Standard topic, normal debate sufficient"

    async def audit_design(
        self,
        improvement: str,
        design_context: str = "",
    ) -> AuditResult:
        """
        Run deep audit for design phase of critical topics.

        Uses strategic design review with cross-examination.

        Args:
            improvement: The proposed improvement
            design_context: Additional context for the design

        Returns:
            AuditResult with verdict details
        """
        if not self._available:
            return AuditResult(
                approved=True,
                message="Deep audit not available, skipping",
            )

        # Type narrowing: _available implies _run_deep_audit is not None
        assert self._run_deep_audit is not None

        self._log("    [deep-audit] Running strategic design audit (5-round)")

        try:
            verdict = await self._run_deep_audit(
                task=f"""STRATEGIC DESIGN REVIEW

## Proposed Improvement
{improvement[:8000]}

{design_context}

## Your Task
1. Evaluate the architectural soundness of this proposal
2. Identify potential risks and unintended consequences
3. Check for conflicts with existing systems
4. Assess complexity vs. value tradeoff
5. Propose refinements or alternatives if needed
6. Flag any concerns that need unanimous agreement before proceeding

Cross-examine each other's reasoning. Be thorough.""",
                agents=self.agents,
                config=self._audit_config,
            )

            approved = len(verdict.unanimous_issues) == 0

            self._log(f"    [deep-audit] Design confidence: {verdict.confidence:.0%}")
            if verdict.unanimous_issues:
                self._log(f"    [deep-audit] Blocking issues: {len(verdict.unanimous_issues)}")
                for issue in verdict.unanimous_issues[:3]:
                    self._log(f"      - {issue[:150]}...")

            return AuditResult(
                approved=approved,
                confidence=verdict.confidence,
                unanimous_issues=verdict.unanimous_issues,
                split_opinions=verdict.split_opinions,
                risk_areas=verdict.risk_areas,
                message="Design audit complete",
            )

        except Exception as e:
            self._log(f"    [deep-audit] Design audit failed: {e}")
            return AuditResult(
                approved=True,
                message=f"Audit failed with error: {e}",
            )

    async def audit_protected_files(
        self,
        diff: str,
        touched_files: list[str],
    ) -> AuditResult:
        """
        Run deep audit for changes to protected files.

        Heavy3-inspired 5-round intensive review with cross-examination.

        Args:
            diff: Git diff of the changes
            touched_files: List of protected file paths being modified

        Returns:
            AuditResult with verdict details
        """
        if not self._available:
            self._log("    [deep-audit] Not available, falling back to regular review")
            return AuditResult(
                approved=True,
                message="Deep audit not available",
            )

        # Type narrowing: _available implies _run_deep_audit is not None
        assert self._run_deep_audit is not None

        self._log(
            f"    [deep-audit] Starting intensive review for protected files: {touched_files}"
        )
        self._log(
            "    [deep-audit] Running 5-round CODE_ARCHITECTURE_AUDIT with cross-examination..."
        )

        try:
            verdict = await self._run_deep_audit(
                task=f"""CRITICAL: Review changes to protected files.

These files are essential to aragora's functionality and must be reviewed with maximum scrutiny.

## Protected Files Being Modified
{", ".join(touched_files)}

## Changes (git diff)
```
{diff[:15000]}
```

## Your Task
1. Analyze each change for correctness and safety
2. Identify any breaking changes or regressions
3. Check for security vulnerabilities
4. Verify backward compatibility is preserved
5. Flag any unanimous issues that must be addressed before merge

Be rigorous. These files are protected for a reason.""",
                agents=self.agents,
                config=self._audit_config,
            )

            # Log verdict summary
            self._log(f"    [deep-audit] Confidence: {verdict.confidence:.0%}")
            self._log(f"    [deep-audit] Unanimous issues: {len(verdict.unanimous_issues)}")
            self._log(f"    [deep-audit] Split opinions: {len(verdict.split_opinions)}")
            self._log(f"    [deep-audit] Risk areas: {len(verdict.risk_areas)}")

            # If there are unanimous issues, reject
            if verdict.unanimous_issues:
                self._log("    [deep-audit] REJECTED - Unanimous issues found:")
                for issue in verdict.unanimous_issues[:5]:
                    self._log(f"      - {issue[:200]}")
                return AuditResult(
                    approved=False,
                    confidence=verdict.confidence,
                    unanimous_issues=verdict.unanimous_issues,
                    split_opinions=verdict.split_opinions,
                    risk_areas=verdict.risk_areas,
                    message="\n".join(verdict.unanimous_issues),
                )

            # Low confidence warning
            if verdict.confidence < 0.5 and len(verdict.split_opinions) > 2:
                self._log("    [deep-audit] WARNING - Low confidence, proceed with caution")

            self._log("    [deep-audit] APPROVED - No unanimous blocking issues")
            return AuditResult(
                approved=True,
                confidence=verdict.confidence,
                unanimous_issues=[],
                split_opinions=verdict.split_opinions,
                risk_areas=verdict.risk_areas,
                message="No blocking issues found",
            )

        except Exception as e:
            self._log(f"    [deep-audit] ERROR: {e}")
            self._log("    [deep-audit] Falling back to regular review due to error")
            return AuditResult(
                approved=True,
                message=f"Audit failed with error: {e}",
            )


__all__ = [
    "AuditResult",
    "DeepAuditRunner",
    "CRITICAL_KEYWORDS",
    "STRATEGY_KEYWORDS",
]
