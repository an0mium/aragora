"""
Risk Register - Identify and track risks from debate outcomes.

Analyzes debate traces for:
- Unresolved critiques (potential risks)
- Low-confidence claims
- Dissenting views
- Unverified dependencies

Produces a structured risk register for project management.
"""

__all__ = [
    "Risk",
    "RiskAnalyzer",
    "RiskCategory",
    "RiskLevel",
    "RiskRegister",
    "generate_risk_register",
]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.export.artifact import DebateArtifact


class RiskLevel(Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Categories of risk."""

    TECHNICAL = "technical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    COMPATIBILITY = "compatibility"
    UNKNOWN = "unknown"


@dataclass
class Risk:
    """A single identified risk."""

    id: str
    title: str
    description: str
    level: RiskLevel
    category: RiskCategory
    source: str  # Where this risk was identified (agent, critique, etc.)

    # Impact and likelihood (for risk matrix)
    impact: float = 0.5  # 0-1
    likelihood: float = 0.5  # 0-1

    # Mitigation
    mitigation: str = ""
    mitigation_status: str = "proposed"  # proposed, in_progress, implemented, accepted

    # Traceability
    related_critique_ids: list[str] = field(default_factory=list)
    related_claim_ids: list[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def risk_score(self) -> float:
        """Calculate risk score (impact * likelihood)."""
        return self.impact * self.likelihood

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "level": self.level.value,
            "category": self.category.value,
            "source": self.source,
            "impact": self.impact,
            "likelihood": self.likelihood,
            "risk_score": self.risk_score,
            "mitigation": self.mitigation,
            "mitigation_status": self.mitigation_status,
            "related_critique_ids": self.related_critique_ids,
            "related_claim_ids": self.related_claim_ids,
            "created_at": self.created_at,
        }


@dataclass
class RiskRegister:
    """
    Collection of risks identified from a debate.

    Provides a structured view of all potential issues
    that need to be addressed before implementation.
    """

    debate_id: str
    risks: list[Risk] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Thresholds (configurable)
    low_support_threshold: float = 0.5
    critical_support_threshold: float = 0.7

    def add_risk(self, risk: Risk) -> None:
        """Add a risk to the register."""
        self.risks.append(risk)

    def get_by_level(self, level: RiskLevel) -> list[Risk]:
        """Get risks by severity level."""
        return [r for r in self.risks if r.level == level]

    def get_by_category(self, category: RiskCategory) -> list[Risk]:
        """Get risks by category."""
        return [r for r in self.risks if r.category == category]

    def get_unmitigated(self) -> list[Risk]:
        """Get risks without implemented mitigations."""
        return [r for r in self.risks if r.mitigation_status != "implemented"]

    def get_critical_risks(self) -> list[Risk]:
        """Get high and critical severity risks."""
        return [r for r in self.risks if r.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]

    @property
    def summary(self) -> dict:
        """Generate summary statistics."""
        return {
            "total_risks": len(self.risks),
            "critical": len(self.get_by_level(RiskLevel.CRITICAL)),
            "high": len(self.get_by_level(RiskLevel.HIGH)),
            "medium": len(self.get_by_level(RiskLevel.MEDIUM)),
            "low": len(self.get_by_level(RiskLevel.LOW)),
            "unmitigated": len(self.get_unmitigated()),
            "avg_risk_score": (
                sum(r.risk_score for r in self.risks) / len(self.risks) if self.risks else 0
            ),
        }

    def to_markdown(self) -> str:
        """Generate markdown representation."""
        summary = self.summary

        risks_by_level = ""
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            level_risks = self.get_by_level(level)
            if level_risks:
                risks_by_level += f"\n### {level.value.upper()} ({len(level_risks)})\n\n"
                for risk in level_risks:
                    status_emoji = {
                        "proposed": "?",
                        "in_progress": "...",
                        "implemented": "v",
                        "accepted": "!",
                    }.get(risk.mitigation_status, "?")

                    risks_by_level += f"""
#### [{status_emoji}] {risk.title}

- **Category:** {risk.category.value}
- **Source:** {risk.source}
- **Risk Score:** {risk.risk_score:.2f} (Impact: {risk.impact:.1f}, Likelihood: {risk.likelihood:.1f})

{risk.description}

**Mitigation:** {risk.mitigation or "*Not defined*"}

---
"""

        return f"""# Risk Register

**Debate ID:** {self.debate_id}
**Generated:** {self.created_at[:10]}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Risks | {summary["total_risks"]} |
| Critical | {summary["critical"]} |
| High | {summary["high"]} |
| Medium | {summary["medium"]} |
| Low | {summary["low"]} |
| Unmitigated | {summary["unmitigated"]} |
| Avg Risk Score | {summary["avg_risk_score"]:.2f} |

---

## Risks by Severity

{risks_by_level}

---

*Generated by aragora v0.8.0*
"""

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "risks": [r.to_dict() for r in self.risks],
            "summary": self.summary,
            "created_at": self.created_at,
            "thresholds": {
                "low_support": self.low_support_threshold,
                "critical_support": self.critical_support_threshold,
            },
        }


class RiskAnalyzer:
    """
    Analyzes debate artifacts to identify risks.

    Examines:
    - Unresolved critiques
    - Low-confidence claims
    - Verification failures
    - Dissenting opinions
    """

    def __init__(self, artifact: "DebateArtifact") -> None:
        self.artifact: DebateArtifact = artifact

    def analyze(self) -> RiskRegister:
        """Perform full risk analysis."""
        register = RiskRegister(debate_id=self.artifact.debate_id)

        # Analyze critiques for unresolved issues
        self._analyze_critiques(register)

        # Analyze verification results
        self._analyze_verifications(register)

        # Analyze consensus for low confidence
        self._analyze_consensus(register)

        return register

    def _analyze_critiques(self, register: RiskRegister) -> None:
        """Extract risks from debate critiques."""
        if not self.artifact.trace_data:
            return

        events = self.artifact.trace_data.get("events", [])
        critique_events = [e for e in events if e.get("event_type") == "agent_critique"]

        for i, event in enumerate(critique_events):
            content = event.get("content", {})
            severity = content.get("severity", 0.5)
            issues = content.get("issues", [])

            if severity >= 0.6:  # High severity critiques become risks
                for issue in issues[:2]:  # Top 2 issues per critique
                    risk = Risk(
                        id=f"critique-{i}",
                        title=issue[:60],
                        description=issue,
                        level=RiskLevel.HIGH if severity >= 0.8 else RiskLevel.MEDIUM,
                        category=self._categorize_issue(issue),
                        source=event.get("agent", "unknown"),
                        impact=severity,
                        likelihood=0.7,
                        mitigation=", ".join(content.get("suggestions", [])[:2]),
                    )
                    register.add_risk(risk)

    def _analyze_verifications(self, register: RiskRegister) -> None:
        """Extract risks from failed verifications."""
        for v in self.artifact.verification_results:
            if v.status in ["refuted", "timeout"]:
                risk = Risk(
                    id=f"verification-{v.claim_id}",
                    title=f"Verification {v.status}: {v.claim_text[:40]}...",
                    description=f"Claim could not be verified: {v.claim_text}",
                    level=RiskLevel.HIGH if v.status == "refuted" else RiskLevel.MEDIUM,
                    category=RiskCategory.TECHNICAL,
                    source=f"formal verification ({v.method})",
                    impact=0.8 if v.status == "refuted" else 0.5,
                    likelihood=0.9,
                    related_claim_ids=[v.claim_id],
                )
                register.add_risk(risk)

    def _analyze_consensus(self, register: RiskRegister) -> None:
        """Extract risks from low consensus confidence."""
        consensus = self.artifact.consensus_proof
        if not consensus:
            return

        if consensus.confidence < 0.7:
            risk = Risk(
                id="consensus-low-confidence",
                title="Low consensus confidence",
                description=f"Debate reached only {consensus.confidence:.0%} confidence. "
                f"Implementation may face challenges or require revision.",
                level=RiskLevel.MEDIUM if consensus.confidence >= 0.5 else RiskLevel.HIGH,
                category=RiskCategory.UNKNOWN,
                source="consensus analysis",
                impact=0.6,
                likelihood=1.0 - consensus.confidence,
            )
            register.add_risk(risk)

        if not consensus.reached:
            risk = Risk(
                id="consensus-not-reached",
                title="No consensus reached",
                description="Agents did not reach consensus. Decision may be contested.",
                level=RiskLevel.HIGH,
                category=RiskCategory.UNKNOWN,
                source="consensus analysis",
                impact=0.8,
                likelihood=0.7,
            )
            register.add_risk(risk)

    def _categorize_issue(self, issue: str) -> RiskCategory:
        """Categorize an issue based on keywords."""
        issue_lower = issue.lower()

        if any(k in issue_lower for k in ["security", "auth", "permission", "vulnerable"]):
            return RiskCategory.SECURITY
        if any(k in issue_lower for k in ["performance", "slow", "latency", "speed"]):
            return RiskCategory.PERFORMANCE
        if any(k in issue_lower for k in ["scale", "load", "capacity", "throughput"]):
            return RiskCategory.SCALABILITY
        if any(k in issue_lower for k in ["maintain", "complex", "readab", "test"]):
            return RiskCategory.MAINTAINABILITY
        if any(k in issue_lower for k in ["compat", "version", "depend", "integrat"]):
            return RiskCategory.COMPATIBILITY

        return RiskCategory.TECHNICAL


def generate_risk_register(artifact) -> RiskRegister:
    """Convenience function to generate risk register from artifact."""
    analyzer = RiskAnalyzer(artifact)
    return analyzer.analyze()
