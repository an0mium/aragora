"""
Audit Trail Export - Compliance-ready event logs.

Generates detailed audit trails from Gauntlet stress-tests:
- Full event timeline with timestamps
- Agent reasoning chains
- Evidence links and provenance
- Signature/checksum verification
- Export to JSON, CSV, and compliance formats

"Every decision needs a paper trail."
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class GauntletResultProtocol(Protocol):
    """Protocol defining the expected interface for GauntletResult objects."""

    gauntlet_id: str
    input_summary: str
    input_type: Any  # Enum with .value attribute
    verdict: Any  # Enum with .value attribute
    confidence: float
    total_findings: int
    agents_involved: list[str]
    duration_seconds: float
    redteam_result: Any | None
    probe_report: Any | None
    audit_verdict: Any | None
    verified_claims: list[Any] | None
    unverified_claims: list[Any] | None
    risk_assessments: list[Any]
    all_findings: list[Any]
    risk_score: float
    robustness_score: float
    checksum: str


class AuditEventType(Enum):
    """Types of audit events."""

    GAUNTLET_START = "gauntlet_start"
    GAUNTLET_END = "gauntlet_end"
    REDTEAM_START = "redteam_start"
    REDTEAM_ATTACK = "redteam_attack"
    REDTEAM_END = "redteam_end"
    PROBE_START = "probe_start"
    PROBE_RESULT = "probe_result"
    PROBE_END = "probe_end"
    AUDIT_START = "audit_start"
    AUDIT_FINDING = "audit_finding"
    AUDIT_END = "audit_end"
    VERIFICATION_START = "verification_start"
    VERIFICATION_RESULT = "verification_result"
    VERIFICATION_END = "verification_end"
    RISK_ASSESSMENT = "risk_assessment"
    FINDING_ADDED = "finding_added"
    VERDICT_DETERMINED = "verdict_determined"
    RECEIPT_GENERATED = "receipt_generated"


@dataclass
class AuditEvent:
    """A single event in the audit trail."""

    event_id: str
    event_type: AuditEventType
    timestamp: str
    source: str  # Component that generated the event
    description: str
    details: dict = field(default_factory=dict)
    severity: Optional[str] = None  # "info", "warning", "error"
    agent: Optional[str] = None
    parent_event_id: Optional[str] = None  # For hierarchical events

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "description": self.description,
            "details": self.details,
            "severity": self.severity,
            "agent": self.agent,
            "parent_event_id": self.parent_event_id,
        }


@dataclass
class AuditTrail:
    """
    Complete audit trail for a Gauntlet stress-test.

    Provides:
    - Chronological event log
    - Agent activity tracking
    - Evidence chain documentation
    - Compliance-ready exports
    """

    trail_id: str
    gauntlet_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Event log
    events: list[AuditEvent] = field(default_factory=list)

    # Summary data
    input_summary: str = ""
    input_type: str = ""
    verdict: str = ""
    confidence: float = 0.0
    total_findings: int = 0
    agents_involved: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    # Counters
    redteam_attacks: int = 0
    probes_run: int = 0
    audit_findings: int = 0
    verifications_attempted: int = 0
    verifications_successful: int = 0

    def __post_init__(self):
        self._event_counter = 0

    def _next_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"evt-{self._event_counter:05d}"

    def add_event(
        self,
        event_type: AuditEventType,
        source: str,
        description: str,
        details: Optional[dict] = None,
        severity: str = "info",
        agent: Optional[str] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Add an event to the trail."""
        event = AuditEvent(
            event_id=self._next_event_id(),
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            source=source,
            description=description,
            details=details or {},
            severity=severity,
            agent=agent,
            parent_event_id=parent_event_id,
        )
        self.events.append(event)
        return event.event_id

    @property
    def checksum(self) -> str:
        """Generate integrity checksum for the trail."""
        content = json.dumps(
            {
                "trail_id": self.trail_id,
                "gauntlet_id": self.gauntlet_id,
                "verdict": self.verdict,
                "events_count": len(self.events),
                "total_findings": self.total_findings,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify the trail hasn't been tampered with."""
        return self.checksum == self.checksum  # Recalculate and compare

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trail_id": self.trail_id,
            "gauntlet_id": self.gauntlet_id,
            "created_at": self.created_at,
            "input_summary": self.input_summary,
            "input_type": self.input_type,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "total_findings": self.total_findings,
            "agents_involved": self.agents_involved,
            "duration_seconds": self.duration_seconds,
            "redteam_attacks": self.redteam_attacks,
            "probes_run": self.probes_run,
            "audit_findings": self.audit_findings,
            "verifications_attempted": self.verifications_attempted,
            "verifications_successful": self.verifications_successful,
            "events": [e.to_dict() for e in self.events],
            "checksum": self.checksum,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv(self) -> str:
        """Export events as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "event_id",
                "timestamp",
                "event_type",
                "source",
                "description",
                "severity",
                "agent",
                "parent_event_id",
            ]
        )

        # Events
        for event in self.events:
            writer.writerow(
                [
                    event.event_id,
                    event.timestamp,
                    event.event_type.value,
                    event.source,
                    event.description,
                    event.severity,
                    event.agent or "",
                    event.parent_event_id or "",
                ]
            )

        return output.getvalue()

    def to_markdown(self) -> str:
        """Export as Markdown document."""
        lines = [
            "# Audit Trail",
            "",
            f"**Trail ID:** `{self.trail_id}`",
            f"**Gauntlet ID:** `{self.gauntlet_id}`",
            f"**Generated:** {self.created_at}",
            f"**Checksum:** `{self.checksum}`",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"**Verdict:** {self.verdict}",
            f"**Confidence:** {self.confidence:.0%}",
            f"**Input Type:** {self.input_type}",
            f"**Duration:** {self.duration_seconds:.1f}s",
            "",
            "### Activity Counts",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Red-team Attacks | {self.redteam_attacks} |",
            f"| Probes Run | {self.probes_run} |",
            f"| Audit Findings | {self.audit_findings} |",
            f"| Verifications | {self.verifications_successful}/{self.verifications_attempted} |",
            f"| Total Findings | {self.total_findings} |",
            "",
            "### Agents",
            "",
        ]

        for agent in self.agents_involved:
            lines.append(f"- {agent}")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Event Timeline",
                "",
            ]
        )

        # Group events by type
        for event in self.events:
            icon = self._event_icon(event.event_type)
            lines.append(f"### {icon} {event.timestamp}")
            lines.append(f"**{event.event_type.value}** from `{event.source}`")
            lines.append("")
            lines.append(f"> {event.description}")
            if event.agent:
                lines.append(f"\n*Agent: {event.agent}*")
            if event.details:
                lines.append("\n**Details:**")
                for k, v in event.details.items():
                    lines.append(f"- {k}: {v}")
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                f"*Integrity checksum: `{self.checksum}`*",
            ]
        )

        return "\n".join(lines)

    def _event_icon(self, event_type: AuditEventType) -> str:
        """Get icon for event type."""
        icons = {
            AuditEventType.GAUNTLET_START: "🚀",
            AuditEventType.GAUNTLET_END: "🏁",
            AuditEventType.REDTEAM_START: "🔴",
            AuditEventType.REDTEAM_ATTACK: "⚔️",
            AuditEventType.REDTEAM_END: "🔴",
            AuditEventType.PROBE_START: "🔍",
            AuditEventType.PROBE_RESULT: "📊",
            AuditEventType.PROBE_END: "🔍",
            AuditEventType.AUDIT_START: "📋",
            AuditEventType.AUDIT_FINDING: "🔎",
            AuditEventType.AUDIT_END: "📋",
            AuditEventType.VERIFICATION_START: "✓",
            AuditEventType.VERIFICATION_RESULT: "✅",
            AuditEventType.VERIFICATION_END: "✓",
            AuditEventType.RISK_ASSESSMENT: "⚠️",
            AuditEventType.FINDING_ADDED: "📌",
            AuditEventType.VERDICT_DETERMINED: "⚖️",
            AuditEventType.RECEIPT_GENERATED: "📄",
        }
        return icons.get(event_type, "•")

    def save(self, path: Path, format: str = "json") -> Path:
        """Save trail to file."""
        if format == "json":
            output_path = path.with_suffix(".json")
            output_path.write_text(self.to_json())
        elif format == "csv":
            output_path = path.with_suffix(".csv")
            output_path.write_text(self.to_csv())
        elif format in ("md", "markdown"):
            output_path = path.with_suffix(".md")
            output_path.write_text(self.to_markdown())
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_path

    @classmethod
    def from_json(cls, json_str: str) -> "AuditTrail":
        """Load trail from JSON string."""
        data = json.loads(json_str)

        # Convert events back to dataclasses
        events = []
        for e in data.pop("events", []):
            e["event_type"] = AuditEventType(e["event_type"])
            events.append(AuditEvent(**e))

        data.pop("checksum", None)  # Recalculated on access
        return cls(events=events, **data)

    @classmethod
    def load(cls, path: Path) -> "AuditTrail":
        """Load trail from file."""
        return cls.from_json(path.read_text())


class AuditTrailGenerator:
    """
    Generates audit trails from Gauntlet results.

    Reconstructs the event timeline from the GauntletResult
    for compliance and debugging purposes.
    """

    @staticmethod
    def from_gauntlet_result(result: GauntletResultProtocol) -> AuditTrail:
        """Generate an AuditTrail from a GauntletResult."""
        trail = AuditTrail(
            trail_id=f"trail-{result.gauntlet_id}",
            gauntlet_id=result.gauntlet_id,
            input_summary=result.input_summary,
            input_type=result.input_type.value,
            verdict=result.verdict.value,
            confidence=result.confidence,
            total_findings=result.total_findings,
            agents_involved=result.agents_involved,
            duration_seconds=result.duration_seconds,
        )

        # Add start event
        start_id = trail.add_event(
            AuditEventType.GAUNTLET_START,
            "GauntletOrchestrator",
            f"Started Gauntlet stress-test with {len(result.agents_involved)} agents",
            details={
                "input_type": result.input_type.value,
                "agents": result.agents_involved,
            },
        )

        # Add red-team events if available
        if result.redteam_result:
            rt = result.redteam_result
            rt_start_id = trail.add_event(
                AuditEventType.REDTEAM_START,
                "RedTeamMode",
                f"Started red-team with {rt.total_attacks} attacks",
                parent_event_id=start_id,
            )

            for attack in rt.critical_issues:
                trail.add_event(
                    AuditEventType.REDTEAM_ATTACK,
                    f"RedTeam/{attack.attacker}",
                    f"Attack: {attack.attack_type.value}",
                    details={
                        "severity": attack.severity,
                        "description": attack.attack_description[:200],
                    },
                    severity="warning" if attack.severity < 0.9 else "error",
                    agent=attack.attacker,
                    parent_event_id=rt_start_id,
                )
                trail.redteam_attacks += 1

            trail.add_event(
                AuditEventType.REDTEAM_END,
                "RedTeamMode",
                f"Red-team complete. Robustness: {rt.robustness_score:.0%}",
                details={"robustness_score": rt.robustness_score},
                parent_event_id=rt_start_id,
            )

        # Add probe events if available
        if result.probe_report:
            pr = result.probe_report
            probe_start_id = trail.add_event(
                AuditEventType.PROBE_START,
                "CapabilityProber",
                f"Started capability probing with {pr.probes_run} probes",
                parent_event_id=start_id,
            )

            trail.probes_run = pr.probes_run

            trail.add_event(
                AuditEventType.PROBE_END,
                "CapabilityProber",
                f"Probing complete. Found {pr.vulnerabilities_found} vulnerabilities",
                details={
                    "vulnerabilities_found": pr.vulnerabilities_found,
                    "vulnerability_rate": pr.vulnerability_rate,
                },
                parent_event_id=probe_start_id,
            )

        # Add audit events if available
        if result.audit_verdict:
            av = result.audit_verdict
            audit_start_id = trail.add_event(
                AuditEventType.AUDIT_START,
                "DeepAuditOrchestrator",
                f"Started deep audit with {len(av.findings)} findings",
                parent_event_id=start_id,
            )

            for finding in av.findings[:10]:  # Limit to prevent huge trails
                trail.add_event(
                    AuditEventType.AUDIT_FINDING,
                    "DeepAudit",
                    finding.summary[:200],
                    details={"severity": finding.severity, "category": finding.category},
                    severity="warning" if finding.severity < 0.8 else "error",
                    parent_event_id=audit_start_id,
                )
                trail.audit_findings += 1

            trail.add_event(
                AuditEventType.AUDIT_END,
                "DeepAuditOrchestrator",
                f"Deep audit complete. Confidence: {av.confidence:.0%}",
                details={"confidence": av.confidence},
                parent_event_id=audit_start_id,
            )

        # Add verification events
        if result.verified_claims:
            trail.verifications_attempted = len(result.verified_claims) + len(
                result.unverified_claims
            )
            trail.verifications_successful = len([v for v in result.verified_claims if v.verified])

            for claim in result.verified_claims[:5]:  # Limit
                trail.add_event(
                    AuditEventType.VERIFICATION_RESULT,
                    f"Verification/{claim.verification_method}",
                    f"{'Verified' if claim.verified else 'Refuted'}: {claim.claim[:100]}",
                    details={
                        "verified": claim.verified,
                        "method": claim.verification_method,
                        "proof_hash": claim.proof_hash,
                    },
                    severity="info" if claim.verified else "warning",
                    parent_event_id=start_id,
                )

        # Add risk assessment events
        for ra in result.risk_assessments[:5]:
            trail.add_event(
                AuditEventType.RISK_ASSESSMENT,
                "RiskAssessor",
                f"Risk identified: {ra.category}",
                details={
                    "level": ra.level.value,
                    "confidence": ra.confidence,
                    "mitigations": ra.mitigations[:3],
                },
                severity="warning" if ra.level.value in ("medium", "high") else "info",
                parent_event_id=start_id,
            )

        # Add findings
        for finding in result.all_findings[:20]:  # Limit
            trail.add_event(
                AuditEventType.FINDING_ADDED,
                finding.source or "Unknown",
                f"[{finding.severity_level}] {finding.title}",
                details={
                    "category": finding.category,
                    "severity": finding.severity,
                    "mitigation": finding.mitigation,
                },
                severity="error" if finding.severity >= 0.9 else "warning",
                parent_event_id=start_id,
            )

        # Add verdict event
        trail.add_event(
            AuditEventType.VERDICT_DETERMINED,
            "GauntletOrchestrator",
            f"Verdict: {result.verdict.value.upper()} (confidence: {result.confidence:.0%})",
            details={
                "verdict": result.verdict.value,
                "confidence": result.confidence,
                "risk_score": result.risk_score,
                "robustness_score": result.robustness_score,
            },
            parent_event_id=start_id,
        )

        # Add end event
        trail.add_event(
            AuditEventType.GAUNTLET_END,
            "GauntletOrchestrator",
            f"Gauntlet complete in {result.duration_seconds:.1f}s",
            details={
                "duration_seconds": result.duration_seconds,
                "total_findings": result.total_findings,
                "checksum": result.checksum,
            },
            parent_event_id=start_id,
        )

        return trail


def generate_audit_trail(result: GauntletResultProtocol) -> AuditTrail:
    """
    Convenience function to generate an AuditTrail.

    Args:
        result: GauntletResult from a Gauntlet stress-test

    Returns:
        AuditTrail ready for export

    Example:
        result = await run_gauntlet(spec, agents)
        trail = generate_audit_trail(result)
        trail.save(Path("./audits/trail.json"))
    """
    return AuditTrailGenerator.from_gauntlet_result(result)
