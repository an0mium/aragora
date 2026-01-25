"""
Decision Receipt - Audit-ready compliance artifacts.

Generates structured receipts from Gauntlet stress-tests:
- Verdict with confidence and risk level
- Findings with severity and mitigations
- Dissenting views and unresolved tensions
- Verified claims with proof hashes
- Complete audit trail for compliance

"Every high-stakes decision deserves a paper trail."
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core_types import DebateResult
    from aragora.export.audit_trail import AuditTrail
    from aragora.modes.gauntlet import GauntletResult  # Full orchestrator result


@dataclass
class ReceiptFinding:
    """Simplified finding for receipt export."""

    id: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    category: str
    title: str
    description: str
    mitigation: Optional[str] = None
    source: str = ""
    verified: bool = False


@dataclass
class ReceiptDissent:
    """Simplified dissent record for receipt export."""

    agent: str
    type: str
    severity: float
    reasons: list[str]
    alternative: Optional[str] = None


@dataclass
class ReceiptVerification:
    """Verification result for receipt export."""

    claim: str
    verified: bool
    method: str
    proof_hash: Optional[str] = None


@dataclass
class DecisionReceipt:
    """
    Audit-ready decision receipt from a Gauntlet stress-test.

    This is the primary compliance artifact - a self-contained record
    of the validation process that can be stored, audited, and referenced.

    Attributes:
        receipt_id: Unique identifier for this receipt
        timestamp: When the decision was made
        input_summary: Brief description of what was tested
        verdict: Final recommendation (APPROVED, REJECTED, etc.)
        confidence: 0-1 confidence in the verdict
        risk_level: Overall risk classification

        findings: All findings from the stress-test
        mitigations: Recommended mitigations
        dissenting_views: Agents who disagreed
        unresolved_tensions: Issues not fully resolved

        verified_claims: Claims that were formally verified
        unverified_claims: Claims that could not be verified

        agents_involved: Which agents participated
        rounds_completed: How many rounds of analysis
        duration_seconds: Total analysis time
        checksum: Integrity hash for tamper detection
    """

    # Identifiers
    receipt_id: str
    gauntlet_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Input context
    input_summary: str = ""
    input_type: str = "spec"

    # Core verdict
    verdict: str = (
        "NEEDS_REVIEW"  # "APPROVED", "APPROVED_WITH_CONDITIONS", "NEEDS_REVIEW", "REJECTED"
    )
    confidence: float = 0.0
    risk_level: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_score: float = 0.0

    # Scores
    robustness_score: float = 0.0
    coverage_score: float = 0.0
    verification_coverage: float = 0.0

    # Findings
    findings: list[ReceiptFinding] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Mitigations
    mitigations: list[str] = field(default_factory=list)

    # Dissent & tensions
    dissenting_views: list[ReceiptDissent] = field(default_factory=list)
    unresolved_tensions: list[str] = field(default_factory=list)

    # Verification
    verified_claims: list[ReceiptVerification] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)

    # Audit metadata
    agents_involved: list[str] = field(default_factory=list)
    rounds_completed: int = 0
    duration_seconds: float = 0.0

    # Cross-reference to audit trail (bidirectional link)
    audit_trail_id: Optional[str] = None

    # Integrity
    checksum: str = ""

    # Cost/usage data (optional, populated if cost tracking enabled)
    cost_usd: float = 0.0
    tokens_used: int = 0
    budget_limit_usd: Optional[float] = None

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute integrity checksum."""
        content = json.dumps(
            {
                "receipt_id": self.receipt_id,
                "verdict": self.verdict,
                "confidence": self.confidence,
                "findings_count": len(self.findings),
                "critical_count": self.critical_count,
                "timestamp": self.timestamp,
                "audit_trail_id": self.audit_trail_id,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify the receipt hasn't been tampered with."""
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "receipt_id": self.receipt_id,
            "gauntlet_id": self.gauntlet_id,
            "timestamp": self.timestamp,
            "input_summary": self.input_summary,
            "input_type": self.input_type,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "robustness_score": self.robustness_score,
            "coverage_score": self.coverage_score,
            "verification_coverage": self.verification_coverage,
            "findings": [asdict(f) for f in self.findings],
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "mitigations": self.mitigations,
            "dissenting_views": [asdict(d) for d in self.dissenting_views],
            "unresolved_tensions": self.unresolved_tensions,
            "verified_claims": [asdict(v) for v in self.verified_claims],
            "unverified_claims": self.unverified_claims,
            "agents_involved": self.agents_involved,
            "rounds_completed": self.rounds_completed,
            "duration_seconds": self.duration_seconds,
            "audit_trail_id": self.audit_trail_id,
            "checksum": self.checksum,
            "cost_usd": self.cost_usd,
            "tokens_used": self.tokens_used,
            "budget_limit_usd": self.budget_limit_usd,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Export as Markdown document."""
        lines = [
            "# Decision Receipt",
            "",
            f"**Receipt ID:** `{self.receipt_id}`",
            f"**Gauntlet ID:** `{self.gauntlet_id}`",
            f"**Generated:** {self.timestamp}",
            f"**Checksum:** `{self.checksum}`",
            "",
            "---",
            "",
            "## Verdict",
            "",
            f"### {self.verdict}",
            "",
            f"**Confidence:** {self.confidence:.0%}",
            f"**Risk Level:** {self.risk_level}",
            f"**Risk Score:** {self.risk_score:.0%}",
            "",
            "### Scores",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Robustness | {self.robustness_score:.0%} |",
            f"| Coverage | {self.coverage_score:.0%} |",
            f"| Verification | {self.verification_coverage:.0%} |",
            "",
            "---",
            "",
            "## Input",
            "",
            f"**Type:** {self.input_type}",
            "",
            "```",
            self.input_summary[:1000] + ("..." if len(self.input_summary) > 1000 else ""),
            "```",
            "",
            "---",
            "",
            "## Findings Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| Critical | {self.critical_count} |",
            f"| High | {self.high_count} |",
            f"| Medium | {self.medium_count} |",
            f"| Low | {self.low_count} |",
            f"| **Total** | **{len(self.findings)}** |",
            "",
        ]

        # Critical findings
        critical = [f for f in self.findings if f.severity == "CRITICAL"]
        if critical:
            lines.extend(
                [
                    "### Critical Issues",
                    "",
                ]
            )
            for f in critical:
                lines.extend(
                    [
                        f"#### {f.title}",
                        "",
                        f.description,
                        "",
                        f"**Source:** {f.source}",
                    ]
                )
                if f.mitigation:
                    lines.append(f"**Mitigation:** {f.mitigation}")
                if f.verified:
                    lines.append("**Status:** Formally verified")
                lines.append("")

        # High findings
        high = [f for f in self.findings if f.severity == "HIGH"]
        if high:
            lines.extend(
                [
                    "### High-Severity Issues",
                    "",
                ]
            )
            for f in high:
                lines.extend(
                    [
                        f"- **{f.title}**: {f.description[:200]}{'...' if len(f.description) > 200 else ''}",
                    ]
                )
            lines.append("")

        # Mitigations
        if self.mitigations:
            lines.extend(
                [
                    "---",
                    "",
                    "## Recommended Mitigations",
                    "",
                ]
            )
            for m in self.mitigations:
                lines.append(f"- {m}")
            lines.append("")

        # Dissenting views
        if self.dissenting_views:
            lines.extend(
                [
                    "---",
                    "",
                    "## Dissenting Views",
                    "",
                ]
            )
            for d in self.dissenting_views:
                lines.extend(
                    [
                        f"### {d.agent}",
                        f"**Type:** {d.type}",
                        f"**Severity:** {d.severity:.0%}",
                        "",
                        "**Reasons:**",
                    ]
                )
                for r in d.reasons:
                    lines.append(f"- {r}")
                if d.alternative:
                    lines.append(f"\n**Alternative view:** {d.alternative}")
                lines.append("")

        # Unresolved tensions
        if self.unresolved_tensions:
            lines.extend(
                [
                    "---",
                    "",
                    "## Unresolved Tensions",
                    "",
                ]
            )
            for t in self.unresolved_tensions:
                lines.append(f"- {t}")
            lines.append("")

        # Verification results
        if self.verified_claims or self.unverified_claims:
            lines.extend(
                [
                    "---",
                    "",
                    "## Verification Results",
                    "",
                    f"**Coverage:** {self.verification_coverage:.0%}",
                    "",
                ]
            )

            if self.verified_claims:
                lines.append("### Verified Claims")
                lines.append("")
                for v in self.verified_claims:
                    status = "VERIFIED" if v.verified else "REFUTED"
                    lines.append(
                        f"- [{status}] {v.claim[:100]}{'...' if len(v.claim) > 100 else ''}"
                    )
                    if v.proof_hash:
                        lines.append(f"  - Proof: `{v.proof_hash}`")
                lines.append("")

            if self.unverified_claims:
                lines.append("### Unverified Claims")
                lines.append("")
                for c in self.unverified_claims[:10]:
                    lines.append(f"- {c[:100]}{'...' if len(c) > 100 else ''}")
                if len(self.unverified_claims) > 10:
                    lines.append(f"- ... and {len(self.unverified_claims) - 10} more")
                lines.append("")

        # Audit trail
        lines.extend(
            [
                "---",
                "",
                "## Audit Trail",
                "",
                f"**Agents:** {', '.join(self.agents_involved)}",
                f"**Rounds:** {self.rounds_completed}",
                f"**Duration:** {self.duration_seconds:.1f}s",
                "",
                "---",
                "",
                "*This receipt was generated by Aragora Gauntlet.*",
                f"*Integrity checksum: `{self.checksum}`*",
            ]
        )

        return "\n".join(lines)

    def to_html(self) -> str:
        """Export as self-contained HTML document."""
        verdict_color = {
            "APPROVED": "#28a745",
            "APPROVED_WITH_CONDITIONS": "#ffc107",
            "NEEDS_REVIEW": "#fd7e14",
            "REJECTED": "#dc3545",
        }.get(self.verdict, "#6c757d")

        findings_html = ""
        for f in self.findings:
            severity_color = {
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#28a745",
            }.get(f.severity, "#6c757d")

            findings_html += f"""
            <div class="finding" style="border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; background: #f8f9fa;">
                <strong style="color: {severity_color};">[{f.severity}]</strong> {f.title}
                <p>{f.description}</p>
                {f"<p><em>Mitigation: {f.mitigation}</em></p>" if f.mitigation else ""}
            </div>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Decision Receipt - {self.receipt_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .verdict {{ font-size: 24px; font-weight: bold; color: {verdict_color}; margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .scores {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .score {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .score-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .score-label {{ font-size: 14px; color: #666; }}
        .section {{ margin: 30px 0; }}
        .finding {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .checksum {{ font-family: monospace; font-size: 12px; color: #666; }}
        .meta {{ font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <h1>Decision Receipt</h1>
    <p class="meta">
        <strong>Receipt ID:</strong> <code>{self.receipt_id}</code><br>
        <strong>Generated:</strong> {self.timestamp}<br>
        <strong>Input Type:</strong> {self.input_type}
    </p>

    <div class="verdict">
        VERDICT: {self.verdict}
        <div style="font-size: 16px; font-weight: normal; margin-top: 10px;">
            Confidence: {self.confidence:.0%} | Risk Level: {self.risk_level}
        </div>
    </div>

    <div class="scores">
        <div class="score">
            <div class="score-value">{self.robustness_score:.0%}</div>
            <div class="score-label">Robustness</div>
        </div>
        <div class="score">
            <div class="score-value">{self.coverage_score:.0%}</div>
            <div class="score-label">Coverage</div>
        </div>
        <div class="score">
            <div class="score-value">{self.verification_coverage:.0%}</div>
            <div class="score-label">Verification</div>
        </div>
    </div>

    <div class="section">
        <h2>Findings Summary</h2>
        <table>
            <tr><th>Severity</th><th>Count</th></tr>
            <tr><td style="color: #dc3545;">Critical</td><td>{self.critical_count}</td></tr>
            <tr><td style="color: #fd7e14;">High</td><td>{self.high_count}</td></tr>
            <tr><td style="color: #ffc107;">Medium</td><td>{self.medium_count}</td></tr>
            <tr><td style="color: #28a745;">Low</td><td>{self.low_count}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>All Findings</h2>
        {findings_html if findings_html else "<p>No findings.</p>"}
    </div>

    <div class="section">
        <h2>Audit Trail</h2>
        <p><strong>Agents:</strong> {", ".join(self.agents_involved)}</p>
        <p><strong>Duration:</strong> {self.duration_seconds:.1f}s</p>
        <p><strong>Rounds:</strong> {self.rounds_completed}</p>
    </div>

    <hr>
    <p class="checksum">
        Integrity Checksum: <code>{self.checksum}</code><br>
        Generated by Aragora Gauntlet
    </p>
</body>
</html>"""

    def to_pdf(self) -> bytes:
        """Export as PDF document.

        Requires weasyprint to be installed.

        Returns:
            PDF bytes

        Raises:
            ImportError: If weasyprint is not installed
        """
        from weasyprint import HTML  # type: ignore[import-untyped]

        html_content = self.to_html()
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes

    def to_csv(self) -> str:
        """Export findings as CSV format.

        Returns:
            CSV string with findings data
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Receipt ID",
                "Timestamp",
                "Verdict",
                "Confidence",
                "Risk Level",
                "Finding ID",
                "Severity",
                "Title",
                "Description",
                "Mitigation",
                "Category",
            ]
        )

        # Write one row per finding
        for f in self.findings:
            writer.writerow(
                [
                    self.receipt_id,
                    self.timestamp,
                    self.verdict,
                    f"{self.confidence:.2f}",
                    self.risk_level,
                    f.id,
                    f.severity,
                    f.title,
                    f.description,
                    f.mitigation or "",
                    f.category,
                ]
            )

        # If no findings, write a summary row
        if not self.findings:
            writer.writerow(
                [
                    self.receipt_id,
                    self.timestamp,
                    self.verdict,
                    f"{self.confidence:.2f}",
                    self.risk_level,
                    "",
                    "",
                    "No findings",
                    "",
                    "",
                    "",
                ]
            )

        return output.getvalue()

    def save(self, path: Path, format: str = "json") -> Path:
        """
        Save receipt to file.

        Args:
            path: Output path (extension will be adjusted if needed)
            format: Output format ("json", "md", "html")

        Returns:
            Path to saved file
        """
        if format == "json":
            output_path = path.with_suffix(".json")
            output_path.write_text(self.to_json())
        elif format == "md" or format == "markdown":
            output_path = path.with_suffix(".md")
            output_path.write_text(self.to_markdown())
        elif format == "html":
            output_path = path.with_suffix(".html")
            output_path.write_text(self.to_html())
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_path

    @classmethod
    def from_json(cls, json_str: str) -> "DecisionReceipt":
        """Load receipt from JSON string."""
        data = json.loads(json_str)

        # Convert nested dicts back to dataclasses
        findings = [ReceiptFinding(**f) for f in data.pop("findings", [])]
        dissenting_views = [ReceiptDissent(**d) for d in data.pop("dissenting_views", [])]
        verified_claims = [ReceiptVerification(**v) for v in data.pop("verified_claims", [])]

        return cls(
            findings=findings,
            dissenting_views=dissenting_views,
            verified_claims=verified_claims,
            **data,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionReceipt":
        """Load receipt from dictionary.

        Args:
            data: Dictionary representation of the receipt

        Returns:
            DecisionReceipt instance
        """
        # Make a copy to avoid mutating the input
        data = dict(data)

        # Convert nested dicts back to dataclasses
        findings = [ReceiptFinding(**f) for f in data.pop("findings", [])]
        dissenting_views = [ReceiptDissent(**d) for d in data.pop("dissenting_views", [])]
        verified_claims = [ReceiptVerification(**v) for v in data.pop("verified_claims", [])]

        return cls(
            findings=findings,
            dissenting_views=dissenting_views,
            verified_claims=verified_claims,
            **data,
        )

    @classmethod
    def load(cls, path: Path) -> "DecisionReceipt":
        """Load receipt from file."""
        return cls.from_json(path.read_text())

    @classmethod
    def from_debate_result(
        cls,
        result: "DebateResult",
        include_cost: bool = True,
        cost_data: Optional[dict[str, Any]] = None,
    ) -> "DecisionReceipt":
        """
        Generate a DecisionReceipt from a standard DebateResult.

        Unlike from_gauntlet_result which uses full Gauntlet stress-test data,
        this creates a receipt from a regular debate for audit purposes.

        Args:
            result: The DebateResult from a completed debate
            include_cost: Whether to include cost data in the receipt
            cost_data: Optional dict with cost_usd, tokens_used, budget_limit_usd

        Returns:
            DecisionReceipt suitable for audit trail

        Example:
            result = await arena.run()
            receipt = DecisionReceipt.from_debate_result(result)
            receipt.save(Path("./receipts/debate.json"), format="json")
        """
        from datetime import timezone

        receipt_id = f"rcpt_{uuid.uuid4().hex[:12]}"

        # Map confidence to verdict
        if result.confidence >= 0.9:
            verdict = "APPROVED"
        elif result.confidence >= 0.7:
            verdict = "APPROVED_WITH_CONDITIONS"
        elif result.confidence >= 0.5:
            verdict = "NEEDS_REVIEW"
        else:
            verdict = "REJECTED"

        # Map confidence to risk (inverse relationship)
        risk_score = 1.0 - result.confidence
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        elif risk_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Extract cost data from result or provided cost_data
        cost_usd = 0.0
        tokens_used = 0
        budget_limit_usd = None

        if include_cost:
            if cost_data:
                cost_usd = cost_data.get("cost_usd", 0.0)
                tokens_used = cost_data.get("tokens_used", 0)
                budget_limit_usd = cost_data.get("budget_limit_usd")
            elif hasattr(result, "total_cost_usd"):
                cost_usd = result.total_cost_usd
                tokens_used = result.total_tokens
                budget_limit_usd = result.budget_limit_usd

        # Convert critiques to findings (high severity by default)
        findings = []
        for critique in result.critiques:
            for idx, issue in enumerate(critique.issues):
                findings.append(
                    ReceiptFinding(
                        id=f"crit_{critique.agent}_{idx}",
                        severity="HIGH"
                        if critique.severity >= 7
                        else "MEDIUM"
                        if critique.severity >= 4
                        else "LOW",
                        category="critique",
                        title=f"Critique from {critique.agent}",
                        description=issue,
                        mitigation=critique.suggestions[idx]
                        if idx < len(critique.suggestions)
                        else None,
                        source=critique.agent,
                        verified=False,
                    )
                )

        # Count findings by severity
        critical_count = len([f for f in findings if f.severity == "CRITICAL"])
        high_count = len([f for f in findings if f.severity == "HIGH"])
        medium_count = len([f for f in findings if f.severity == "MEDIUM"])
        low_count = len([f for f in findings if f.severity == "LOW"])

        # Convert dissenting views
        dissents = []
        for view in result.dissenting_views:
            dissents.append(
                ReceiptDissent(
                    agent="unknown",  # Dissenting views are strings in DebateResult
                    type="dissent",
                    severity=0.5,
                    reasons=[view],
                    alternative=None,
                )
            )

        return cls(
            receipt_id=receipt_id,
            gauntlet_id=result.debate_id or result.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=result.task[:500] if result.task else "",
            input_type="debate",
            verdict=verdict,
            confidence=result.confidence,
            risk_level=risk_level,
            risk_score=risk_score,
            robustness_score=result.confidence,  # Use confidence as proxy
            coverage_score=1.0 if result.consensus_reached else 0.5,
            verification_coverage=0.0,  # No formal verification in regular debates
            findings=findings,
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            mitigations=[],  # No structured mitigations in regular debates
            dissenting_views=dissents,
            unresolved_tensions=[],
            verified_claims=[],
            unverified_claims=[],
            agents_involved=result.participants,
            rounds_completed=result.rounds_completed,
            duration_seconds=result.duration_seconds,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            budget_limit_usd=budget_limit_usd,
        )


class DecisionReceiptGenerator:
    """
    Generates Decision Receipts from Gauntlet results.

    Transforms the detailed GauntletResult into a clean,
    audit-ready DecisionReceipt.
    """

    @staticmethod
    def from_gauntlet_result(result: "GauntletResult") -> DecisionReceipt:
        """
        Generate a DecisionReceipt from a GauntletResult.

        Args:
            result: The GauntletResult to convert

        Returns:
            A DecisionReceipt ready for export
        """

        # Convert findings
        findings = []
        for f in result.all_findings:
            findings.append(
                ReceiptFinding(
                    id=f.finding_id,
                    severity=f.severity_level,
                    category=f.category,
                    title=f.title,
                    description=f.description,
                    mitigation=f.mitigation,
                    source=f.source,
                    verified=f.verified,
                )
            )

        # Convert dissenting views
        dissents = []
        for d in result.dissenting_views:
            dissents.append(
                ReceiptDissent(
                    agent=d.agent,
                    type=d.dissent_type,
                    severity=d.severity,
                    reasons=d.reasons,
                    alternative=d.alternative_view,
                )
            )

        # Convert verified claims
        verified = []
        for v in result.verified_claims:
            verified.append(
                ReceiptVerification(
                    claim=v.claim,
                    verified=v.verified,
                    method=v.verification_method,
                    proof_hash=v.proof_hash,
                )
            )

        # Extract mitigations from findings
        mitigations = list(set(f.mitigation for f in result.all_findings if f.mitigation))

        # Determine risk level from score
        if result.risk_score >= 0.8:
            risk_level = "CRITICAL"
        elif result.risk_score >= 0.6:
            risk_level = "HIGH"
        elif result.risk_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Determine rounds from sub-results
        rounds = 0
        if result.audit_verdict:
            rounds = len(result.audit_verdict.findings)
        if result.redteam_result:
            rounds = max(rounds, len(result.redteam_result.rounds))

        return DecisionReceipt(
            receipt_id=str(uuid.uuid4()),
            gauntlet_id=result.gauntlet_id,
            timestamp=result.created_at,
            input_summary=result.input_summary,
            input_type=result.input_type.value,
            verdict=result.verdict.value.upper(),
            confidence=result.confidence,
            risk_level=risk_level,
            risk_score=result.risk_score,
            robustness_score=result.robustness_score,
            coverage_score=result.coverage_score,
            verification_coverage=result.verification_coverage,
            findings=findings,
            critical_count=len(result.critical_findings),
            high_count=len(result.high_findings),
            medium_count=len(result.medium_findings),
            low_count=len(result.low_findings),
            mitigations=mitigations,
            dissenting_views=dissents,
            unresolved_tensions=[t.description for t in result.unresolved_tensions],
            verified_claims=verified,
            unverified_claims=result.unverified_claims,
            agents_involved=result.agents_involved,
            rounds_completed=rounds,
            duration_seconds=result.duration_seconds,
        )


def generate_decision_receipt(result: "GauntletResult") -> DecisionReceipt:
    """
    Convenience function to generate a DecisionReceipt.

    Args:
        result: GauntletResult from a Gauntlet stress-test

    Returns:
        DecisionReceipt ready for export

    Example:
        result = await run_gauntlet(spec, agents)
        receipt = generate_decision_receipt(result)
        receipt.save(Path("./receipts/decision.html"), format="html")
    """
    return DecisionReceiptGenerator.from_gauntlet_result(result)


def link_receipt_to_trail(
    receipt: DecisionReceipt,
    trail: "AuditTrail",
) -> tuple[DecisionReceipt, "AuditTrail"]:
    """
    Link a DecisionReceipt and AuditTrail bidirectionally.

    Creates cross-references between the receipt and trail for:
    - Compliance auditing (trace from receipt to full event log)
    - Evidence chain verification (trace from events to final decision)

    After linking, both checksums are recomputed to include the link.

    Args:
        receipt: The DecisionReceipt to link
        trail: The AuditTrail to link

    Returns:
        Tuple of (updated_receipt, updated_trail)

    Example:
        receipt = generate_decision_receipt(result)
        trail = generate_audit_trail(result)
        receipt, trail = link_receipt_to_trail(receipt, trail)
        # Now receipt.audit_trail_id == trail.trail_id
        # And trail.receipt_id == receipt.receipt_id
    """
    # Establish bidirectional links
    receipt.audit_trail_id = trail.trail_id
    trail.receipt_id = receipt.receipt_id

    # Recompute checksums to include the links
    receipt.checksum = receipt._compute_checksum()
    # Note: trail.checksum is a property, automatically recomputed

    return receipt, trail
