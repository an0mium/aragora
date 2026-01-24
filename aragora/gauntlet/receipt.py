"""
Decision Receipt - Audit-ready output format.

Provides a tamper-evident, comprehensive record of a Gauntlet validation
suitable for compliance, audit trails, and decision documentation.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Any, Optional

from .result import GauntletResult


@dataclass
class ProvenanceRecord:
    """A single provenance record in the chain."""

    timestamp: str
    event_type: str  # "attack", "probe", "scenario", "verdict"
    agent: Optional[str] = None
    description: str = ""
    evidence_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "agent": self.agent,
            "description": self.description,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class ConsensusProof:
    """Proof of agent consensus."""

    reached: bool
    confidence: float
    supporting_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)
    method: str = "majority"
    evidence_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "reached": self.reached,
            "confidence": self.confidence,
            "supporting_agents": self.supporting_agents,
            "dissenting_agents": self.dissenting_agents,
            "method": self.method,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class DecisionReceipt:
    """
    Audit-ready receipt for a Gauntlet validation.

    Contains:
    - Input identification and hash
    - Complete findings summary
    - Verdict with reasoning
    - Provenance chain for auditability
    - Content-addressable artifact hash
    """

    # Identification
    receipt_id: str
    gauntlet_id: str
    timestamp: str

    # Input
    input_summary: str
    input_hash: str  # SHA-256 for integrity verification

    # Findings summary
    risk_summary: dict  # Critical/High/Medium/Low counts
    attacks_attempted: int
    attacks_successful: int
    probes_run: int
    vulnerabilities_found: int

    # Verdict
    verdict: str  # "PASS", "CONDITIONAL", "FAIL"
    confidence: float
    robustness_score: float

    # Fields with defaults must come after fields without defaults
    vulnerability_details: list[dict] = field(default_factory=list)
    verdict_reasoning: str = ""

    # Evidence
    dissenting_views: list[str] = field(default_factory=list)
    consensus_proof: Optional[ConsensusProof] = None
    provenance_chain: list[ProvenanceRecord] = field(default_factory=list)

    # Integrity
    artifact_hash: str = ""  # Content-addressable hash of entire receipt
    config_used: dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate artifact hash if not provided."""
        if not self.artifact_hash:
            self.artifact_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate content-addressable hash."""
        content = json.dumps(
            {
                "receipt_id": self.receipt_id,
                "gauntlet_id": self.gauntlet_id,
                "input_hash": self.input_hash,
                "risk_summary": self.risk_summary,
                "verdict": self.verdict,
                "confidence": self.confidence,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt has not been tampered with."""
        expected_hash = self._calculate_hash()
        return expected_hash == self.artifact_hash

    @classmethod
    def from_result(cls, result: GauntletResult) -> "DecisionReceipt":
        """Create receipt from GauntletResult."""
        receipt_id = str(uuid.uuid4())

        # Build provenance chain
        provenance = []

        # Add attack events
        for vuln in result.vulnerabilities:
            if vuln.source == "red_team":
                provenance.append(
                    ProvenanceRecord(
                        timestamp=vuln.created_at,
                        event_type="attack",
                        agent=vuln.agent_name,
                        description=f"[{vuln.severity.value}] {vuln.title[:50]}",
                        evidence_hash=hashlib.sha256(vuln.description.encode()).hexdigest()[:16],
                    )
                )

        # Add probe events
        for vuln in result.vulnerabilities:
            if vuln.source == "capability_probe":
                provenance.append(
                    ProvenanceRecord(
                        timestamp=vuln.created_at,
                        event_type="probe",
                        agent=vuln.agent_name,
                        description=f"[{vuln.category}] {vuln.title[:50]}",
                        evidence_hash=hashlib.sha256(vuln.description.encode()).hexdigest()[:16],
                    )
                )

        # Add verdict event
        provenance.append(
            ProvenanceRecord(
                timestamp=result.completed_at,
                event_type="verdict",
                description=f"Verdict: {result.verdict.value} ({result.confidence:.1%} confidence)",
            )
        )

        # Build consensus proof
        consensus = ConsensusProof(
            reached=result.verdict.value != "fail",
            confidence=result.confidence,
            supporting_agents=result.agents_used,
            method="adversarial_validation",
        )

        return cls(
            receipt_id=receipt_id,
            gauntlet_id=result.gauntlet_id,
            timestamp=result.completed_at,
            input_summary=result.input_summary,
            input_hash=result.input_hash,
            risk_summary=result.risk_summary.to_dict(),
            attacks_attempted=result.attack_summary.total_attacks,
            attacks_successful=result.attack_summary.successful_attacks,
            probes_run=result.probe_summary.probes_run,
            vulnerabilities_found=result.risk_summary.total,
            vulnerability_details=[v.to_dict() for v in result.get_critical_vulnerabilities()],
            verdict=result.verdict.value.upper(),
            confidence=result.confidence,
            robustness_score=result.attack_summary.robustness_score,
            verdict_reasoning=result.verdict_reasoning,
            dissenting_views=result.dissenting_views,
            consensus_proof=consensus,
            provenance_chain=provenance,
            config_used=result.config_used,
        )

    @classmethod
    def from_mode_result(
        cls,
        result: Any,
        input_hash: Optional[str] = None,
    ) -> "DecisionReceipt":
        """Create receipt from aragora.modes.gauntlet.GauntletResult."""
        receipt_id = str(uuid.uuid4())

        findings = list(getattr(result, "all_findings", []))
        critical = len(getattr(result, "critical_findings", []))
        high = len(getattr(result, "high_findings", []))
        medium = len(getattr(result, "medium_findings", []))
        low = len(getattr(result, "low_findings", []))

        redteam = getattr(result, "redteam_result", None)
        probe_report = getattr(result, "probe_report", None)
        audit_verdict = getattr(result, "audit_verdict", None)

        provenance = []
        for finding in findings:
            provenance.append(
                ProvenanceRecord(
                    timestamp=getattr(finding, "timestamp", ""),
                    event_type="finding",
                    agent=None,
                    description=f"[{finding.severity_level}] {finding.title[:50]}",
                    evidence_hash=hashlib.sha256(finding.description.encode()).hexdigest()[:16],
                )
            )

        provenance.append(
            ProvenanceRecord(
                timestamp=getattr(result, "created_at", ""),
                event_type="verdict",
                description=f"Verdict: {result.verdict.value} ({result.confidence:.1%} confidence)",
            )
        )

        dissenting = []
        for dissent in getattr(result, "dissenting_views", []):
            if hasattr(dissent, "agent"):
                reasons = "; ".join(getattr(dissent, "reasons", []) or [])
                summary = f"{dissent.agent}: {reasons}".strip()
                if getattr(dissent, "alternative_view", None):
                    summary = f"{summary} | alt: {dissent.alternative_view}".strip()
                dissenting.append(summary)
            else:
                dissenting.append(str(dissent))

        dissenting_agents = [
            getattr(d, "agent", "")
            for d in getattr(result, "dissenting_views", [])
            if getattr(d, "agent", None)
        ]

        consensus = ConsensusProof(
            reached=bool(getattr(result, "consensus_reached", False)),
            confidence=result.confidence,
            supporting_agents=list(getattr(result, "agents_involved", [])),
            dissenting_agents=dissenting_agents,
            method="gauntlet_consensus",
            evidence_hash=getattr(result, "checksum", ""),
        )

        verdict_reasoning = (
            f"Risk score: {result.risk_score:.0%}, "
            f"Coverage: {result.coverage_score:.0%}, "
            f"Verification: {getattr(result, 'verification_coverage', 0.0):.0%}"
        )
        if audit_verdict and getattr(audit_verdict, "recommendation", None):
            verdict_reasoning = audit_verdict.recommendation[:500]

        severity_details = [
            {
                "id": f.finding_id,
                "category": f.category,
                "severity": f.severity,
                "severity_level": f.severity_level,
                "title": f.title,
                "description": f.description,
                "evidence": f.evidence,
                "mitigation": f.mitigation,
                "source": f.source,
                "verified": f.verified,
                "timestamp": f.timestamp,
            }
            for f in findings
            if f.severity_level in ("CRITICAL", "HIGH")
        ]

        return cls(
            receipt_id=receipt_id,
            gauntlet_id=result.gauntlet_id,
            timestamp=getattr(result, "created_at", ""),
            input_summary=result.input_summary,
            input_hash=input_hash
            or getattr(result, "input_hash", "")
            or getattr(result, "checksum", ""),
            risk_summary={
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
                "total": len(findings),
            },
            attacks_attempted=getattr(redteam, "total_attacks", 0) if redteam else 0,
            attacks_successful=getattr(redteam, "successful_attacks", 0) if redteam else 0,
            probes_run=getattr(probe_report, "probes_run", 0) if probe_report else 0,
            vulnerabilities_found=len(findings),
            verdict=result.verdict.value.upper(),
            confidence=result.confidence,
            robustness_score=result.robustness_score,
            vulnerability_details=severity_details,
            verdict_reasoning=verdict_reasoning,
            dissenting_views=dissenting,
            consensus_proof=consensus,
            provenance_chain=provenance,
        )

    @classmethod
    def from_gauntlet_result(cls, result: Any) -> "DecisionReceipt":
        """Create receipt from aragora.gauntlet.config.GauntletResult.

        This handles the GauntletResult dataclass from config.py which has
        different attributes than the one from result.py.
        """
        receipt_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Build provenance chain from findings
        provenance: list[ProvenanceRecord] = []
        for finding in getattr(result, "findings", []):
            provenance.append(
                ProvenanceRecord(
                    timestamp=timestamp,
                    event_type="finding",
                    description=f"[{finding.severity.value}] {finding.title[:50]}",
                    evidence_hash=hashlib.sha256(finding.description.encode()).hexdigest()[:16],
                )
            )

        # Add verdict event
        verdict_str = "PASS" if result.passed else "FAIL"
        provenance.append(
            ProvenanceRecord(
                timestamp=timestamp,
                event_type="verdict",
                description=f"Verdict: {verdict_str} ({result.confidence:.1%} confidence)",
            )
        )

        # Build consensus proof
        consensus = ConsensusProof(
            reached=result.consensus_reached,
            confidence=result.confidence,
            supporting_agents=result.agents_used,
            method="gauntlet_validation",
        )

        # Build risk summary from severity counts
        severity_counts = result.severity_counts
        risk_summary = {
            "critical": severity_counts.get("critical", 0),
            "high": severity_counts.get("high", 0),
            "medium": severity_counts.get("medium", 0),
            "low": severity_counts.get("low", 0),
            "total": len(result.findings),
        }

        # Build vulnerability details from critical findings
        vulnerability_details = [
            {
                "id": f.id,
                "category": f.category,
                "severity": f.severity.value,
                "title": f.title,
                "description": f.description,
                "recommendations": f.recommendations,
            }
            for f in result.critical_findings
        ]

        # Calculate input hash
        input_hash = hashlib.sha256(result.input_text.encode()).hexdigest()

        return cls(
            receipt_id=receipt_id,
            gauntlet_id=result.id,
            timestamp=timestamp,
            input_summary=result.input_text[:500] if result.input_text else "",
            input_hash=input_hash,
            risk_summary=risk_summary,
            attacks_attempted=result.probes_executed,
            attacks_successful=len(result.findings),
            probes_run=result.probes_executed,
            vulnerabilities_found=len(result.findings),
            verdict=verdict_str,
            confidence=result.confidence,
            robustness_score=result.robustness_score,
            vulnerability_details=vulnerability_details,
            verdict_reasoning=result.verdict_summary,
            consensus_proof=consensus,
            provenance_chain=provenance,
            config_used=result.config.to_dict() if result.config else {},
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "receipt_id": self.receipt_id,
            "gauntlet_id": self.gauntlet_id,
            "timestamp": self.timestamp,
            "input_summary": self.input_summary,
            "input_hash": self.input_hash,
            "risk_summary": self.risk_summary,
            "attacks_attempted": self.attacks_attempted,
            "attacks_successful": self.attacks_successful,
            "probes_run": self.probes_run,
            "vulnerabilities_found": self.vulnerabilities_found,
            "vulnerability_details": self.vulnerability_details,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "robustness_score": self.robustness_score,
            "verdict_reasoning": self.verdict_reasoning,
            "dissenting_views": self.dissenting_views,
            "consensus_proof": self.consensus_proof.to_dict() if self.consensus_proof else None,
            "provenance_chain": [p.to_dict() for p in self.provenance_chain],
            "artifact_hash": self.artifact_hash,
            "config_used": self.config_used,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        verdict_emoji = {
            "PASS": "✓",
            "CONDITIONAL": "~",
            "FAIL": "✗",
        }.get(self.verdict, "?")

        lines = [
            "# Decision Receipt",
            "",
            f"**Receipt ID:** `{self.receipt_id}`",
            f"**Gauntlet ID:** `{self.gauntlet_id}`",
            f"**Generated:** {self.timestamp}",
            "",
            "---",
            "",
            f"## Verdict: [{verdict_emoji}] {self.verdict}",
            "",
            f"**Confidence:** {self.confidence:.1%}",
            f"**Robustness Score:** {self.robustness_score:.1%}",
            "",
            f"> {self.verdict_reasoning}",
            "",
            "---",
            "",
            "## Risk Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| Critical | {self.risk_summary.get('critical', 0)} |",
            f"| High | {self.risk_summary.get('high', 0)} |",
            f"| Medium | {self.risk_summary.get('medium', 0)} |",
            f"| Low | {self.risk_summary.get('low', 0)} |",
            f"| **Total** | **{self.vulnerabilities_found}** |",
            "",
            "---",
            "",
            "## Validation Coverage",
            "",
            f"- **Attacks Attempted:** {self.attacks_attempted}",
            f"- **Attacks Successful:** {self.attacks_successful}",
            f"- **Probes Run:** {self.probes_run}",
            "",
        ]

        if self.vulnerability_details:
            lines.append("---")
            lines.append("")
            lines.append("## Critical Findings")
            lines.append("")
            for vuln in self.vulnerability_details[:5]:
                lines.append(f"### {vuln.get('title', 'Unknown')}")
                lines.append(f"**Severity:** {vuln.get('severity', 'unknown').upper()}")
                lines.append(f"**Category:** {vuln.get('category', 'unknown')}")
                lines.append("")
                lines.append(vuln.get("description", "")[:500])
                if vuln.get("mitigation"):
                    lines.append("")
                    lines.append(f"**Mitigation:** {vuln.get('mitigation')}")
                lines.append("")

        if self.dissenting_views:
            lines.append("---")
            lines.append("")
            lines.append("## Dissenting Views")
            lines.append("")
            for view in self.dissenting_views[:5]:
                lines.append(f"- {view}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Integrity")
        lines.append("")
        lines.append(f"**Input Hash:** `{self.input_hash[:16]}...`")
        lines.append(f"**Artifact Hash:** `{self.artifact_hash[:16]}...`")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Aragora Gauntlet*")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Export as self-contained HTML document."""
        verdict_color = {
            "PASS": "#28a745",
            "CONDITIONAL": "#ffc107",
            "FAIL": "#dc3545",
        }.get(self.verdict, "#6c757d")

        findings_html = ""
        for vuln in self.vulnerability_details[:20]:
            severity = str(vuln.get("severity", "UNKNOWN")).upper()
            severity_color = {
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#28a745",
            }.get(severity, "#6c757d")
            title = escape(str(vuln.get("title", "")))
            description = escape(str(vuln.get("description", "")))
            mitigation = vuln.get("mitigation")
            mitigation_html = ""
            if mitigation:
                mitigation_html = f"<p><em>Mitigation: {escape(str(mitigation))}</em></p>"

            findings_html += (
                f'<div class="finding" style="border-left: 4px solid {severity_color};">'
                f'<strong style="color: {severity_color};">[{severity}]</strong> {title}'
                f"<p>{description}</p>"
                f"{mitigation_html}"
                "</div>"
            )

        risk_summary = self.risk_summary or {}

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Decision Receipt - {escape(self.receipt_id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .verdict {{ font-size: 22px; font-weight: bold; color: {verdict_color}; margin: 20px 0; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
        .scores {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 20px 0; }}
        .score {{ text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px; }}
        .score-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        .score-label {{ font-size: 12px; color: #666; }}
        .section {{ margin: 24px 0; }}
        .finding {{ margin: 10px 0; padding: 12px; background: #f8f9fa; border-left: 4px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .meta {{ font-size: 13px; color: #666; }}
        code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; }}
    </style>
</head>
<body>
    <h1>Decision Receipt</h1>
    <p class="meta">
        <strong>Receipt ID:</strong> <code>{escape(self.receipt_id)}</code><br>
        <strong>Gauntlet ID:</strong> <code>{escape(self.gauntlet_id)}</code><br>
        <strong>Generated:</strong> {escape(self.timestamp)}
    </p>

    <div class="verdict">
        VERDICT: {escape(self.verdict)}
        <div style="font-size: 14px; font-weight: normal; margin-top: 8px;">
            Confidence: {self.confidence:.0%} | Robustness: {self.robustness_score:.0%}
        </div>
        {f'<div style="font-size: 13px; font-weight: normal; margin-top: 8px;">{escape(self.verdict_reasoning)}</div>' if self.verdict_reasoning else ""}
    </div>

    <div class="scores">
        <div class="score">
            <div class="score-value">{self.confidence:.0%}</div>
            <div class="score-label">Confidence</div>
        </div>
        <div class="score">
            <div class="score-value">{self.robustness_score:.0%}</div>
            <div class="score-label">Robustness</div>
        </div>
    </div>

    <div class="section">
        <h2>Risk Summary</h2>
        <table>
            <tr><th>Severity</th><th>Count</th></tr>
            <tr><td>Critical</td><td>{risk_summary.get("critical", 0)}</td></tr>
            <tr><td>High</td><td>{risk_summary.get("high", 0)}</td></tr>
            <tr><td>Medium</td><td>{risk_summary.get("medium", 0)}</td></tr>
            <tr><td>Low</td><td>{risk_summary.get("low", 0)}</td></tr>
            <tr><td><strong>Total</strong></td><td><strong>{self.vulnerabilities_found}</strong></td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Coverage</h2>
        <p class="meta">
            Attacks Attempted: {self.attacks_attempted}<br>
            Attacks Successful: {self.attacks_successful}<br>
            Probes Run: {self.probes_run}
        </p>
    </div>

    <div class="section">
        <h2>Findings</h2>
        {findings_html or '<p class="meta">No findings reported.</p>'}
    </div>

    <div class="section">
        <h2>Integrity</h2>
        <p class="meta">
            Input Hash: <code>{escape(self.input_hash[:32])}...</code><br>
            Artifact Hash: <code>{escape(self.artifact_hash[:32])}...</code>
        </p>
    </div>
</body>
</html>
"""

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_sarif(self) -> dict:
        """Export as SARIF 2.1.0 format.

        SARIF (Static Analysis Results Interchange Format) is the OASIS standard
        for exchanging static analysis results. This enables interoperability with:
        - GitHub Security (code scanning)
        - Azure DevOps
        - VS Code SARIF Viewer
        - SonarQube
        - DefectDojo
        """
        # Map severity to SARIF levels
        sarif_level_map = {
            "CRITICAL": "error",
            "HIGH": "error",
            "MEDIUM": "warning",
            "LOW": "note",
        }

        # Map severity to SARIF security-severity scores (CVSS-like)
        sarif_severity_map = {
            "CRITICAL": "9.0",
            "HIGH": "7.0",
            "MEDIUM": "4.0",
            "LOW": "1.0",
        }

        # Build rules from unique vulnerability categories
        rules: list[dict[str, Any]] = []
        rule_ids: dict[str, int] = {}

        for idx, vuln in enumerate(self.vulnerability_details):
            category = vuln.get("category", "unknown")
            if category not in rule_ids:
                rule_id = f"ARAGORA-{len(rule_ids) + 1:03d}"
                rule_ids[category] = len(rules)
                rules.append(
                    {
                        "id": rule_id,
                        "name": category.replace("_", " ").title(),
                        "shortDescription": {"text": f"Aragora Gauntlet: {category}"},
                        "fullDescription": {"text": f"Security finding in category: {category}"},
                        "helpUri": "https://aragora.ai/docs/gauntlet",
                        "properties": {
                            "security-severity": sarif_severity_map.get(
                                str(vuln.get("severity_level", "MEDIUM")).upper(), "4.0"
                            ),
                            "tags": ["security", "aragora", category],
                        },
                    }
                )

        # Build results from vulnerability details
        results = []
        for vuln in self.vulnerability_details:
            category = vuln.get("category", "unknown")
            severity = str(vuln.get("severity_level", vuln.get("severity", "MEDIUM"))).upper()
            rule_idx = rule_ids.get(category, 0)
            rule_id = rules[rule_idx]["id"] if rule_idx < len(rules) else "ARAGORA-000"

            result = {
                "ruleId": rule_id,
                "ruleIndex": rule_idx,
                "level": sarif_level_map.get(severity, "warning"),
                "message": {"text": vuln.get("description", vuln.get("title", "Finding"))},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": f"input/{self.input_hash[:8]}",
                                "uriBaseId": "GAUNTLET_ROOT",
                            }
                        },
                        "logicalLocations": [
                            {
                                "name": vuln.get("title", "Unknown"),
                                "kind": "finding",
                            }
                        ],
                    }
                ],
                "fingerprints": {
                    "aragora/v1": hashlib.sha256(
                        f"{vuln.get('id', '')}:{vuln.get('title', '')}".encode()
                    ).hexdigest()[:32]
                },
                "properties": {
                    "gauntlet_id": self.gauntlet_id,
                    "receipt_id": self.receipt_id,
                    "category": category,
                    "severity": severity,
                    "verified": vuln.get("verified", False),
                },
            }

            # Add fix suggestions if mitigation is present
            if vuln.get("mitigation"):
                result["fixes"] = [{"description": {"text": vuln.get("mitigation", "")}}]

            results.append(result)

        # Build SARIF document
        sarif = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Aragora Gauntlet",
                            "version": "1.0.0",
                            "informationUri": "https://aragora.ai/gauntlet",
                            "rules": rules,
                            "properties": {
                                "verdict": self.verdict,
                                "confidence": self.confidence,
                                "robustness_score": self.robustness_score,
                            },
                        }
                    },
                    "results": results,
                    "invocations": [
                        {
                            "executionSuccessful": True,
                            "endTimeUtc": self.timestamp,
                            "properties": {
                                "gauntlet_id": self.gauntlet_id,
                                "receipt_id": self.receipt_id,
                                "attacks_attempted": self.attacks_attempted,
                                "attacks_successful": self.attacks_successful,
                                "probes_run": self.probes_run,
                            },
                        }
                    ],
                    "artifacts": [
                        {
                            "location": {
                                "uri": f"input/{self.input_hash[:8]}",
                                "uriBaseId": "GAUNTLET_ROOT",
                            },
                            "hashes": {
                                "sha-256": self.input_hash,
                            },
                            "length": -1,
                            "properties": {
                                "summary": self.input_summary[:200],
                            },
                        }
                    ],
                    "properties": {
                        "risk_summary": self.risk_summary,
                        "artifact_hash": self.artifact_hash,
                        "consensus_proof": (
                            self.consensus_proof.to_dict() if self.consensus_proof else None
                        ),
                    },
                }
            ],
        }

        return sarif

    def to_sarif_json(self, indent: int = 2) -> str:
        """Export as SARIF JSON string."""
        return json.dumps(self.to_sarif(), indent=indent)

    def to_pdf(self) -> bytes:
        """Export as PDF document.

        Requires weasyprint to be installed: pip install weasyprint

        Returns:
            PDF content as bytes

        Raises:
            ImportError: If weasyprint is not installed
        """
        try:
            from weasyprint import HTML
        except ImportError as e:
            raise ImportError(
                "weasyprint is required for PDF export. " "Install with: pip install weasyprint"
            ) from e

        html_content = self.to_html()
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes

    def to_csv(self) -> str:
        """Export findings as CSV format.

        Returns:
            CSV content with vulnerability details
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Finding ID",
                "Category",
                "Severity",
                "Title",
                "Description",
                "Mitigation",
                "Verified",
                "Source",
            ]
        )

        # Data rows
        for vuln in self.vulnerability_details:
            writer.writerow(
                [
                    vuln.get("id", ""),
                    vuln.get("category", ""),
                    vuln.get("severity_level", vuln.get("severity", "")),
                    vuln.get("title", ""),
                    vuln.get("description", "")[:500],
                    vuln.get("mitigation", ""),
                    vuln.get("verified", False),
                    vuln.get("source", ""),
                ]
            )

        return output.getvalue()
