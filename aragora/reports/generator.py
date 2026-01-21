"""
Audit Report Generator.

Generates professional audit reports from completed audit sessions.

Supports multiple output formats:
- PDF: Executive summaries and detailed findings
- Markdown: Documentation-ready format
- JSON: Machine-readable for integrations
- HTML: Web-viewable reports
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.audit.document_auditor import AuditFinding, AuditSession

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report output formats."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class ReportTemplate(str, Enum):
    """Report template styles."""

    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_FINDINGS = "detailed_findings"
    COMPLIANCE_ATTESTATION = "compliance_attestation"
    SECURITY_ASSESSMENT = "security_assessment"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Output settings
    output_dir: str = "."
    filename_prefix: str = "audit_report"

    # Content settings
    include_executive_summary: bool = True
    include_findings_detail: bool = True
    include_charts: bool = True
    include_recommendations: bool = True
    include_appendix: bool = False

    # Filtering
    min_severity: str = "low"  # Minimum severity to include
    include_resolved: bool = False
    include_false_positives: bool = False

    # Branding
    company_name: str = "Aragora"
    logo_path: Optional[str] = None

    # Metadata
    author: str = ""
    reviewer: str = ""


@dataclass
class ReportSection:
    """A section of the report."""

    title: str
    content: str
    order: int = 0
    subsections: list["ReportSection"] = field(default_factory=list)


@dataclass
class GeneratedReport:
    """Result of report generation."""

    format: ReportFormat
    content: bytes
    filename: str
    size_bytes: int
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    session_id: str = ""
    findings_count: int = 0
    pages: int = 0

    def save(self, path: Optional[Path] = None) -> Path:
        """Save report to file."""
        if path is None:
            path = Path(self.filename)
        path.write_bytes(self.content)
        return path


class AuditReportGenerator:
    """
    Generates audit reports from completed sessions.

    Supports multiple templates and output formats.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    async def generate(
        self,
        session: "AuditSession",
        format: ReportFormat = ReportFormat.MARKDOWN,
        template: ReportTemplate = ReportTemplate.DETAILED_FINDINGS,
    ) -> GeneratedReport:
        """
        Generate a report from an audit session.

        Args:
            session: Completed audit session
            format: Output format
            template: Report template style

        Returns:
            Generated report
        """
        # Filter findings
        findings = self._filter_findings(session.findings)

        # Build report sections
        sections = self._build_sections(session, findings, template)

        # Render to format
        if format == ReportFormat.MARKDOWN:
            content = self._render_markdown(sections, session, findings)
        elif format == ReportFormat.JSON:
            content = self._render_json(session, findings)
        elif format == ReportFormat.HTML:
            content = self._render_html(sections, session, findings)
        elif format == ReportFormat.PDF:
            content = await self._render_pdf(sections, session, findings)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Generate filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        extension = {
            ReportFormat.MARKDOWN: "md",
            ReportFormat.JSON: "json",
            ReportFormat.HTML: "html",
            ReportFormat.PDF: "pdf",
        }[format]
        filename = f"{self.config.filename_prefix}_{session.id[:8]}_{timestamp}.{extension}"

        return GeneratedReport(
            format=format,
            content=content,
            filename=filename,
            size_bytes=len(content),
            session_id=session.id,
            findings_count=len(findings),
        )

    def _filter_findings(self, findings: list["AuditFinding"]) -> list["AuditFinding"]:
        """Filter findings based on config."""
        from aragora.audit.document_auditor import FindingSeverity, FindingStatus

        severity_order = ["critical", "high", "medium", "low", "info"]
        min_idx = severity_order.index(self.config.min_severity)
        allowed_severities = {FindingSeverity(s) for s in severity_order[: min_idx + 1]}

        filtered = []
        for f in findings:
            # Check severity
            if f.severity not in allowed_severities:
                continue

            # Check status
            if f.status == FindingStatus.RESOLVED and not self.config.include_resolved:
                continue
            if f.status == FindingStatus.FALSE_POSITIVE and not self.config.include_false_positives:
                continue

            filtered.append(f)

        return filtered

    def _build_sections(
        self,
        session: "AuditSession",
        findings: list["AuditFinding"],
        template: ReportTemplate,
    ) -> list[ReportSection]:
        """Build report sections based on template."""
        sections = []

        if self.config.include_executive_summary:
            sections.append(self._build_executive_summary(session, findings))

        if template == ReportTemplate.EXECUTIVE_SUMMARY:
            # Just summary, no detailed findings
            sections.append(self._build_summary_stats(session, findings))

        elif template == ReportTemplate.DETAILED_FINDINGS:
            sections.append(self._build_findings_by_severity(findings))
            if self.config.include_recommendations:
                sections.append(self._build_recommendations(findings))

        elif template == ReportTemplate.COMPLIANCE_ATTESTATION:
            sections.append(self._build_compliance_summary(session, findings))
            sections.append(self._build_attestation_section(session))

        elif template == ReportTemplate.SECURITY_ASSESSMENT:
            sections.append(self._build_security_summary(findings))
            sections.append(self._build_vulnerability_details(findings))
            sections.append(self._build_remediation_roadmap(findings))

        if self.config.include_appendix:
            sections.append(self._build_appendix(session, findings))

        return sections

    def _build_executive_summary(
        self,
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build executive summary section."""
        severity_counts: dict[str, int] = {}
        for f in findings:
            sev = f.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        critical = severity_counts.get("critical", 0)
        high = severity_counts.get("high", 0)

        risk_level = "Low"
        if critical > 0:
            risk_level = "Critical"
        elif high > 2:
            risk_level = "High"
        elif high > 0:
            risk_level = "Medium"

        content = f"""This audit examined {len(session.document_ids)} documents and identified {len(findings)} findings.

**Overall Risk Level: {risk_level}**

**Summary:**
- Critical findings: {critical}
- High severity: {high}
- Medium severity: {severity_counts.get('medium', 0)}
- Low severity: {severity_counts.get('low', 0)}
- Informational: {severity_counts.get('info', 0)}

**Audit Duration:** {session.duration_seconds:.1f} seconds
**Completion:** {session.completed_at.strftime('%Y-%m-%d %H:%M UTC') if session.completed_at else 'N/A'}
"""

        return ReportSection(title="Executive Summary", content=content, order=1)

    def _build_summary_stats(
        self,
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build summary statistics section."""
        by_type: dict[str, int] = {}
        by_doc: dict[str, int] = {}

        for f in findings:
            t = f.audit_type.value
            by_type[t] = by_type.get(t, 0) + 1

            d = f.document_id
            by_doc[d] = by_doc.get(d, 0) + 1

        type_lines = [f"- {t}: {c}" for t, c in sorted(by_type.items())]
        doc_lines = [
            f"- {d[:20]}...: {c}" for d, c in sorted(by_doc.items(), key=lambda x: -x[1])[:5]
        ]

        content = f"""**Findings by Type:**
{chr(10).join(type_lines)}

**Top Documents by Findings:**
{chr(10).join(doc_lines)}
"""

        return ReportSection(title="Summary Statistics", content=content, order=2)

    def _build_findings_by_severity(
        self,
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build detailed findings section grouped by severity."""
        severity_order = ["critical", "high", "medium", "low", "info"]
        grouped: dict[str, list["AuditFinding"]] = {s: [] for s in severity_order}

        for f in findings:
            grouped[f.severity.value].append(f)

        subsections = []
        for sev in severity_order:
            if not grouped[sev]:
                continue

            finding_texts = []
            for i, f in enumerate(grouped[sev], 1):
                finding_texts.append(
                    f"""
**{sev.upper()}-{i}: {f.title}**
- Document: `{f.document_id}`
- Category: {f.category}
- Confidence: {f.confidence:.0%}
- Description: {f.description}
- Evidence: {f.evidence_text[:200]}{'...' if len(f.evidence_text) > 200 else ''}
- Recommendation: {f.recommendation or 'N/A'}
"""
                )

            subsections.append(
                ReportSection(
                    title=f"{sev.title()} Severity ({len(grouped[sev])})",
                    content="\n".join(finding_texts),
                    order=severity_order.index(sev),
                )
            )

        return ReportSection(
            title="Detailed Findings",
            content="",
            order=3,
            subsections=subsections,
        )

    def _build_recommendations(
        self,
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build recommendations section."""
        # Group recommendations by category
        by_category: dict[str, list[str]] = {}
        for f in findings:
            if f.recommendation:
                cat = f.category or "General"
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(f.recommendation)

        rec_lines = []
        for cat, recs in sorted(by_category.items()):
            rec_lines.append(f"\n**{cat}:**")
            for r in set(recs):  # Deduplicate
                rec_lines.append(f"- {r}")

        content = "\n".join(rec_lines) if rec_lines else "No specific recommendations."

        return ReportSection(title="Recommendations", content=content, order=4)

    def _build_compliance_summary(
        self,
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build compliance summary for attestation template."""
        from aragora.audit.document_auditor import AuditType

        compliance_findings = [f for f in findings if f.audit_type == AuditType.COMPLIANCE]
        critical = sum(1 for f in compliance_findings if f.severity.value == "critical")

        status = "Non-Compliant" if critical > 0 else "Compliant with Observations"
        if len(compliance_findings) == 0:
            status = "Compliant"

        content = f"""**Compliance Status: {status}**

Total Compliance Findings: {len(compliance_findings)}
Critical Issues: {critical}
Documents Reviewed: {len(session.document_ids)}

This assessment covers security controls, data handling practices, and regulatory requirements.
"""

        return ReportSection(title="Compliance Summary", content=content, order=2)

    def _build_attestation_section(
        self,
        session: "AuditSession",
    ) -> ReportSection:
        """Build attestation signature section."""
        content = f"""
---

**Attestation**

I hereby attest that this audit was conducted in accordance with established procedures and that the findings represent an accurate assessment of the documents reviewed.

Audit Session ID: `{session.id}`
Audit Date: {session.completed_at.strftime('%Y-%m-%d') if session.completed_at else 'N/A'}

Prepared by: {self.config.author or '________________'}
Reviewed by: {self.config.reviewer or '________________'}

Signature: ________________ Date: ________________
"""

        return ReportSection(title="Attestation", content=content, order=10)

    def _build_security_summary(
        self,
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build security assessment summary."""
        from aragora.audit.document_auditor import AuditType

        sec_findings = [f for f in findings if f.audit_type == AuditType.SECURITY]

        vuln_types: dict[str, int] = {}
        for f in sec_findings:
            cat = f.category or "Unknown"
            vuln_types[cat] = vuln_types.get(cat, 0) + 1

        type_lines = [f"- {t}: {c}" for t, c in sorted(vuln_types.items(), key=lambda x: -x[1])]

        content = f"""**Security Assessment Results**

Total Security Findings: {len(sec_findings)}

**Vulnerability Types:**
{chr(10).join(type_lines) if type_lines else 'No security vulnerabilities detected.'}
"""

        return ReportSection(title="Security Summary", content=content, order=2)

    def _build_vulnerability_details(
        self,
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build vulnerability details for security template."""
        from aragora.audit.document_auditor import AuditType

        sec_findings = [f for f in findings if f.audit_type == AuditType.SECURITY]

        details = []
        for i, f in enumerate(sec_findings, 1):
            details.append(
                f"""
### {i}. {f.title}
- **Severity:** {f.severity.value.upper()}
- **Category:** {f.category}
- **Location:** {f.evidence_location or f.document_id}
- **Description:** {f.description}
- **Impact:** {f.affected_scope or 'Unknown'}
- **Remediation:** {f.recommendation or 'Review and address accordingly.'}
"""
            )

        content = "\n".join(details) if details else "No vulnerabilities to report."

        return ReportSection(title="Vulnerability Details", content=content, order=3)

    def _build_remediation_roadmap(
        self,
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build remediation roadmap section."""
        from aragora.audit.document_auditor import AuditType

        sec_findings = [f for f in findings if f.audit_type == AuditType.SECURITY]

        # Group by priority
        immediate = [f for f in sec_findings if f.severity.value in ("critical", "high")]
        short_term = [f for f in sec_findings if f.severity.value == "medium"]
        long_term = [f for f in sec_findings if f.severity.value in ("low", "info")]

        content = f"""**Remediation Priorities:**

**Immediate Action Required ({len(immediate)} items):**
{chr(10).join(f'- {f.title}' for f in immediate) or 'None'}

**Short-term (30 days) ({len(short_term)} items):**
{chr(10).join(f'- {f.title}' for f in short_term) or 'None'}

**Long-term ({len(long_term)} items):**
{chr(10).join(f'- {f.title}' for f in long_term) or 'None'}
"""

        return ReportSection(title="Remediation Roadmap", content=content, order=4)

    def _build_appendix(
        self,
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> ReportSection:
        """Build appendix with raw data."""
        content = f"""**Audit Configuration:**
- Model: {session.model}
- Audit Types: {', '.join(t.value for t in session.audit_types)}
- Documents: {len(session.document_ids)}
- Chunks Processed: {session.processed_chunks}/{session.total_chunks}

**Document IDs:**
{chr(10).join(f'- {d}' for d in session.document_ids)}

**Errors:**
{chr(10).join(f'- {e}' for e in session.errors) or 'None'}
"""

        return ReportSection(title="Appendix", content=content, order=99)

    def _render_markdown(
        self,
        sections: list[ReportSection],
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> bytes:
        """Render report to Markdown format."""
        lines = [
            f"# Audit Report: {session.name or session.id}",
            f"\n*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            f"*Session: {session.id}*\n",
        ]

        for section in sorted(sections, key=lambda s: s.order):
            lines.append(f"\n## {section.title}\n")
            lines.append(section.content)

            for sub in sorted(section.subsections, key=lambda s: s.order):
                lines.append(f"\n### {sub.title}\n")
                lines.append(sub.content)

        return "\n".join(lines).encode("utf-8")

    def _render_json(
        self,
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> bytes:
        """Render report to JSON format."""
        report_data = {
            "report": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "Aragora Audit Report Generator",
                "version": "1.0",
            },
            "session": session.to_dict(),
            "findings": [f.to_dict() for f in findings],
            "summary": {
                "total_findings": len(findings),
                "by_severity": session.findings_by_severity,
                "documents_audited": len(session.document_ids),
                "duration_seconds": session.duration_seconds,
            },
        }

        return json.dumps(report_data, indent=2).encode("utf-8")

    def _render_html(
        self,
        sections: list[ReportSection],
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> bytes:
        """Render report to HTML format."""
        # Build HTML content
        section_html = []
        for section in sorted(sections, key=lambda s: s.order):
            sub_html = ""
            for sub in sorted(section.subsections, key=lambda s: s.order):
                sub_html += f"""
                <div class="subsection">
                    <h3>{sub.title}</h3>
                    <div class="content">{self._md_to_html(sub.content)}</div>
                </div>
                """

            section_html.append(
                f"""
            <section class="report-section">
                <h2>{section.title}</h2>
                <div class="content">{self._md_to_html(section.content)}</div>
                {sub_html}
            </section>
            """
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit Report: {session.name or session.id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 2px solid #4a4a6a;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2a2a4a;
            margin-top: 30px;
        }}
        h3 {{
            color: #3a3a5a;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .report-section {{
            margin-bottom: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .subsection {{
            margin-left: 20px;
            padding: 10px;
            border-left: 3px solid #ddd;
        }}
        pre {{
            background: #272822;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            font-family: 'Fira Code', monospace;
        }}
        .severity-critical {{ color: #dc3545; font-weight: bold; }}
        .severity-high {{ color: #fd7e14; font-weight: bold; }}
        .severity-medium {{ color: #ffc107; }}
        .severity-low {{ color: #28a745; }}
    </style>
</head>
<body>
    <h1>Audit Report: {session.name or session.id}</h1>
    <div class="meta">
        <p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
        <p>Session ID: {session.id}</p>
    </div>
    {"".join(section_html)}
</body>
</html>"""

        return html.encode("utf-8")

    def _md_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        import re

        html = markdown

        # Bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # Code blocks
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Lists
        lines = html.split("\n")
        in_list = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("- "):
                if not in_list:
                    new_lines.append("<ul>")
                    in_list = True
                new_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    new_lines.append("</ul>")
                    in_list = False
                new_lines.append(f"<p>{line}</p>" if line.strip() else "")
        if in_list:
            new_lines.append("</ul>")

        return "\n".join(new_lines)

    async def _render_pdf(
        self,
        sections: list[ReportSection],
        session: "AuditSession",
        findings: list["AuditFinding"],
    ) -> bytes:
        """Render report to PDF format."""
        # Try to use a PDF library if available
        try:
            from weasyprint import HTML

            html_content = self._render_html(sections, session, findings)
            pdf_bytes = HTML(string=html_content.decode("utf-8")).write_pdf()
            return pdf_bytes

        except ImportError:
            # Fall back to returning HTML with PDF header
            logger.warning("weasyprint not installed, returning HTML for PDF request")
            logger.info("Install weasyprint for proper PDF: pip install weasyprint")
            return self._render_html(sections, session, findings)


__all__ = [
    "AuditReportGenerator",
    "ReportConfig",
    "ReportFormat",
    "ReportTemplate",
    "GeneratedReport",
]
