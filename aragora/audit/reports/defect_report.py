"""
Defect Report Generator.

Creates structured reports from audit findings in multiple formats:
- JSON: Machine-readable format for API consumption
- Markdown: Human-readable format for documentation
- HTML: Rich format for web display
- CSV: Spreadsheet-compatible format

Usage:
    from aragora.audit.reports.defect_report import DefectReport, generate_report

    findings = await auditor.get_findings(session_id)
    report = DefectReport(findings)

    # Generate in different formats
    json_output = report.to_json()
    markdown_output = report.to_markdown()
    html_output = report.to_html()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Available report output formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Output format
    format: ReportFormat = ReportFormat.MARKDOWN

    # Include sections
    include_summary: bool = True
    include_severity_breakdown: bool = True
    include_category_breakdown: bool = True
    include_findings_detail: bool = True
    include_remediation: bool = True
    include_evidence: bool = True

    # Filtering
    min_severity: Optional[str] = None  # Filter findings below this severity
    audit_types: Optional[list[str]] = None  # Filter to specific audit types
    max_findings: int = 0  # 0 = unlimited

    # Grouping
    group_by: str = "severity"  # severity, category, document, audit_type

    # Style
    title: str = "Document Audit Report"
    company_name: str = ""
    logo_url: str = ""


@dataclass
class SeverityStats:
    """Statistics by severity level."""

    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    info: int = 0

    @property
    def total(self) -> int:
        return self.critical + self.high + self.medium + self.low + self.info

    def to_dict(self) -> dict[str, int]:
        return {
            "critical": self.critical,
            "high": self.high,
            "medium": self.medium,
            "low": self.low,
            "info": self.info,
            "total": self.total,
        }


@dataclass
class CategoryStats:
    """Statistics by audit category."""

    counts: dict[str, int] = field(default_factory=dict)

    def add(self, category: str):
        self.counts[category] = self.counts.get(category, 0) + 1

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items(), key=lambda x: x[1], reverse=True))


@dataclass
class DocumentStats:
    """Statistics by document."""

    findings_per_doc: dict[str, int] = field(default_factory=dict)
    severity_per_doc: dict[str, SeverityStats] = field(default_factory=dict)

    def add_finding(self, document_id: str, severity: str):
        # Overall count
        self.findings_per_doc[document_id] = self.findings_per_doc.get(document_id, 0) + 1

        # Severity breakdown
        if document_id not in self.severity_per_doc:
            self.severity_per_doc[document_id] = SeverityStats()

        stats = self.severity_per_doc[document_id]
        severity_lower = severity.lower()
        if severity_lower == "critical":
            stats.critical += 1
        elif severity_lower == "high":
            stats.high += 1
        elif severity_lower == "medium":
            stats.medium += 1
        elif severity_lower == "low":
            stats.low += 1
        else:
            stats.info += 1


class DefectReport:
    """
    Generates comprehensive defect reports from audit findings.

    Supports multiple output formats and customizable sections.
    """

    def __init__(
        self,
        findings: list[Any],
        config: Optional[ReportConfig] = None,
        session_id: str = "",
        audit_start: Optional[datetime] = None,
        audit_end: Optional[datetime] = None,
    ):
        """
        Initialize defect report.

        Args:
            findings: List of AuditFinding objects
            config: Report configuration
            session_id: Audit session identifier
            audit_start: When the audit started
            audit_end: When the audit completed
        """
        self.config = config or ReportConfig()
        self.session_id = session_id
        self.audit_start = audit_start or datetime.now()
        self.audit_end = audit_end or datetime.now()

        # Filter findings based on config
        self.findings = self._filter_findings(findings)

        # Compute statistics
        self.severity_stats = self._compute_severity_stats()
        self.category_stats = self._compute_category_stats()
        self.document_stats = self._compute_document_stats()

    def _filter_findings(self, findings: list[Any]) -> list[Any]:
        """Filter findings based on configuration."""
        filtered = findings

        # Filter by severity
        if self.config.min_severity:
            severity_order = ["critical", "high", "medium", "low", "info"]
            min_idx = severity_order.index(self.config.min_severity.lower())
            filtered = [
                f
                for f in filtered
                if severity_order.index(self._get_severity(f).lower()) <= min_idx
            ]

        # Filter by audit type
        if self.config.audit_types:
            type_set = set(t.lower() for t in self.config.audit_types)
            filtered = [f for f in filtered if self._get_audit_type(f).lower() in type_set]

        # Limit count
        if self.config.max_findings > 0:
            filtered = filtered[: self.config.max_findings]

        return filtered

    def _get_severity(self, finding: Any) -> str:
        """Extract severity from finding (handles different formats)."""
        if hasattr(finding, "severity"):
            sev = finding.severity
            return sev.value if hasattr(sev, "value") else str(sev)
        if isinstance(finding, dict):
            return str(finding.get("severity", "medium"))
        return "medium"

    def _get_audit_type(self, finding: Any) -> str:
        """Extract audit type from finding."""
        if hasattr(finding, "audit_type"):
            at = finding.audit_type
            return at.value if hasattr(at, "value") else str(at)
        if isinstance(finding, dict):
            return str(finding.get("audit_type", "quality"))
        return "quality"

    def _get_category(self, finding: Any) -> str:
        """Extract category from finding."""
        if hasattr(finding, "category"):
            return finding.category or "uncategorized"
        if isinstance(finding, dict):
            return str(finding.get("category", "uncategorized"))
        return "uncategorized"

    def _get_document_id(self, finding: Any) -> str:
        """Extract document ID from finding."""
        if hasattr(finding, "document_id"):
            return finding.document_id or "unknown"
        if isinstance(finding, dict):
            return str(finding.get("document_id", "unknown"))
        return "unknown"

    def _compute_severity_stats(self) -> SeverityStats:
        """Compute severity breakdown statistics."""
        stats = SeverityStats()
        for finding in self.findings:
            severity = self._get_severity(finding).lower()
            if severity == "critical":
                stats.critical += 1
            elif severity == "high":
                stats.high += 1
            elif severity == "medium":
                stats.medium += 1
            elif severity == "low":
                stats.low += 1
            else:
                stats.info += 1
        return stats

    def _compute_category_stats(self) -> CategoryStats:
        """Compute category breakdown statistics."""
        stats = CategoryStats()
        for finding in self.findings:
            category = self._get_category(finding)
            stats.add(category)
        return stats

    def _compute_document_stats(self) -> DocumentStats:
        """Compute per-document statistics."""
        stats = DocumentStats()
        for finding in self.findings:
            doc_id = self._get_document_id(finding)
            severity = self._get_severity(finding)
            stats.add_finding(doc_id, severity)
        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "meta": {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "audit_start": self.audit_start.isoformat(),
                "audit_end": self.audit_end.isoformat(),
                "total_findings": len(self.findings),
            },
            "summary": {
                "severity": self.severity_stats.to_dict(),
                "categories": self.category_stats.to_dict(),
                "documents": {
                    doc_id: {
                        "total": count,
                        "severity": self.document_stats.severity_per_doc.get(
                            doc_id, SeverityStats()
                        ).to_dict(),
                    }
                    for doc_id, count in self.document_stats.findings_per_doc.items()
                },
            },
            "findings": [self._finding_to_dict(f) for f in self.findings],
        }

    def _finding_to_dict(self, finding: Any) -> dict[str, Any]:
        """Convert a finding to dictionary."""
        if hasattr(finding, "to_dict"):
            result: dict[str, Any] = finding.to_dict()
            return result
        if isinstance(finding, dict):
            return finding
        # Fallback for dataclass-like objects
        return {
            "id": getattr(finding, "id", ""),
            "title": getattr(finding, "title", ""),
            "description": getattr(finding, "description", ""),
            "severity": self._get_severity(finding),
            "audit_type": self._get_audit_type(finding),
            "category": self._get_category(finding),
            "document_id": self._get_document_id(finding),
            "evidence_text": getattr(finding, "evidence_text", ""),
            "evidence_location": getattr(finding, "evidence_location", ""),
            "remediation": getattr(finding, "remediation", ""),
            "confidence": getattr(finding, "confidence", 0.8),
        }

    def to_json(self, indent: int = 2) -> str:
        """Generate JSON report."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = []

        # Title
        lines.append(f"# {self.config.title}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Session ID:** {self.session_id}")
        lines.append("")

        # Summary
        if self.config.include_summary:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(f"Total findings: **{len(self.findings)}**")
            lines.append("")

            if self.config.include_severity_breakdown:
                lines.append("### Severity Breakdown")
                lines.append("")
                lines.append("| Severity | Count |")
                lines.append("|----------|-------|")
                if self.severity_stats.critical > 0:
                    lines.append(f"| Critical | {self.severity_stats.critical} |")
                if self.severity_stats.high > 0:
                    lines.append(f"| High | {self.severity_stats.high} |")
                if self.severity_stats.medium > 0:
                    lines.append(f"| Medium | {self.severity_stats.medium} |")
                if self.severity_stats.low > 0:
                    lines.append(f"| Low | {self.severity_stats.low} |")
                if self.severity_stats.info > 0:
                    lines.append(f"| Info | {self.severity_stats.info} |")
                lines.append("")

            if self.config.include_category_breakdown:
                lines.append("### Category Breakdown")
                lines.append("")
                lines.append("| Category | Count |")
                lines.append("|----------|-------|")
                for category, count in self.category_stats.to_dict().items():
                    lines.append(f"| {category} | {count} |")
                lines.append("")

        # Detailed Findings
        if self.config.include_findings_detail:
            lines.append("## Detailed Findings")
            lines.append("")

            # Group findings
            grouped = self._group_findings()

            for group_name, group_findings in grouped.items():
                lines.append(f"### {group_name}")
                lines.append("")

                for finding in group_findings:
                    f_dict = self._finding_to_dict(finding)
                    severity = f_dict.get("severity", "medium")
                    title = f_dict.get("title", "Untitled Finding")

                    # Severity badge
                    severity_badge = self._severity_badge(severity)
                    lines.append(f"#### {severity_badge} {title}")
                    lines.append("")

                    if f_dict.get("description"):
                        lines.append(f_dict["description"])
                        lines.append("")

                    if self.config.include_evidence and f_dict.get("evidence_text"):
                        lines.append("**Evidence:**")
                        lines.append(f"> {f_dict['evidence_text'][:500]}")
                        if f_dict.get("evidence_location"):
                            lines.append(f"> *Location: {f_dict['evidence_location']}*")
                        lines.append("")

                    if self.config.include_remediation and f_dict.get("remediation"):
                        lines.append("**Remediation:**")
                        lines.append(f_dict["remediation"])
                        lines.append("")

                    lines.append("---")
                    lines.append("")

        return "\n".join(lines)

    def _severity_badge(self, severity: str) -> str:
        """Generate a severity indicator."""
        badges = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "medium": "[MEDIUM]",
            "low": "[LOW]",
            "info": "[INFO]",
        }
        return badges.get(severity.lower(), "[UNKNOWN]")

    def _group_findings(self) -> dict[str, list[Any]]:
        """Group findings by configured criteria."""
        grouped: dict[str, list[Any]] = {}

        for finding in self.findings:
            if self.config.group_by == "severity":
                key = self._get_severity(finding).upper()
            elif self.config.group_by == "category":
                key = self._get_category(finding)
            elif self.config.group_by == "document":
                key = self._get_document_id(finding)
            elif self.config.group_by == "audit_type":
                key = self._get_audit_type(finding).upper()
            else:
                key = "All Findings"

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(finding)

        # Sort groups by severity priority
        if self.config.group_by == "severity":
            order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
            grouped = {k: grouped[k] for k in order if k in grouped}

        return grouped

    def to_html(self) -> str:
        """Generate HTML report."""
        # Convert markdown to basic HTML
        md = self.to_markdown()

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.config.title}</title>",
            "<style>",
            "body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #1a1a1a; }",
            "h2 { color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; }",
            "h3 { color: #444; }",
            "h4 { color: #555; }",
            "table { border-collapse: collapse; width: 100%; margin: 15px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f5f5f5; }",
            "blockquote { border-left: 4px solid #ddd; margin: 10px 0; padding-left: 15px; color: #666; }",
            ".critical { color: #d32f2f; font-weight: bold; }",
            ".high { color: #f57c00; font-weight: bold; }",
            ".medium { color: #fbc02d; }",
            ".low { color: #388e3c; }",
            ".info { color: #1976d2; }",
            "hr { border: none; border-top: 1px solid #eee; margin: 20px 0; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        # Simple markdown to HTML conversion
        import re

        html_content = md
        # Headers
        html_content = re.sub(r"^#### (.*)$", r"<h4>\1</h4>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^### (.*)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^## (.*)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^# (.*)$", r"<h1>\1</h1>", html_content, flags=re.MULTILINE)
        # Bold
        html_content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html_content)
        # Severity badges
        html_content = html_content.replace(
            "[CRITICAL]", '<span class="critical">[CRITICAL]</span>'
        )
        html_content = html_content.replace("[HIGH]", '<span class="high">[HIGH]</span>')
        html_content = html_content.replace("[MEDIUM]", '<span class="medium">[MEDIUM]</span>')
        html_content = html_content.replace("[LOW]", '<span class="low">[LOW]</span>')
        html_content = html_content.replace("[INFO]", '<span class="info">[INFO]</span>')
        # Blockquotes
        html_content = re.sub(
            r"^> (.*)$", r"<blockquote>\1</blockquote>", html_content, flags=re.MULTILINE
        )
        # Horizontal rules
        html_content = html_content.replace("---", "<hr>")
        # Tables (basic)
        html_content = re.sub(
            r"\|([^|]+)\|([^|]+)\|", r"<tr><td>\1</td><td>\2</td></tr>", html_content
        )
        html_content = re.sub(r"\|-+\|-+\|", "", html_content)  # Remove separator rows
        # Paragraphs - convert double newlines to paragraph breaks
        html_content = re.sub(r"\n\n+", r"</p>\n<p>", html_content)
        # Wrap in paragraph if needed
        if not html_content.strip().startswith("<"):
            html_content = "<p>" + html_content + "</p>"

        html_parts.append(html_content)
        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def to_csv(self) -> str:
        """Generate CSV report."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "ID",
                "Title",
                "Severity",
                "Category",
                "Audit Type",
                "Document ID",
                "Confidence",
                "Description",
                "Evidence",
                "Location",
                "Remediation",
            ]
        )

        # Data rows
        for finding in self.findings:
            f_dict = self._finding_to_dict(finding)
            writer.writerow(
                [
                    f_dict.get("id", ""),
                    f_dict.get("title", ""),
                    f_dict.get("severity", ""),
                    f_dict.get("category", ""),
                    f_dict.get("audit_type", ""),
                    f_dict.get("document_id", ""),
                    f_dict.get("confidence", ""),
                    f_dict.get("description", ""),
                    f_dict.get("evidence_text", "")[:200],  # Truncate for CSV
                    f_dict.get("evidence_location", ""),
                    f_dict.get("remediation", ""),
                ]
            )

        return output.getvalue()

    def generate(self, format: Optional[ReportFormat] = None) -> str:
        """
        Generate report in specified format.

        Args:
            format: Output format (uses config.format if not specified)

        Returns:
            Report string in requested format
        """
        fmt = format or self.config.format

        if fmt == ReportFormat.JSON:
            return self.to_json()
        elif fmt == ReportFormat.MARKDOWN:
            return self.to_markdown()
        elif fmt == ReportFormat.HTML:
            return self.to_html()
        elif fmt == ReportFormat.CSV:
            return self.to_csv()
        else:
            return self.to_markdown()


def generate_report(
    findings: list[Any],
    format: ReportFormat = ReportFormat.MARKDOWN,
    session_id: str = "",
    **config_kwargs,
) -> str:
    """
    Convenience function to generate a report.

    Args:
        findings: List of audit findings
        format: Output format
        session_id: Audit session ID
        **config_kwargs: Additional ReportConfig options

    Returns:
        Report string in requested format
    """
    config = ReportConfig(format=format, **config_kwargs)
    report = DefectReport(findings, config=config, session_id=session_id)
    return report.generate()


__all__ = [
    "DefectReport",
    "ReportConfig",
    "ReportFormat",
    "SeverityStats",
    "CategoryStats",
    "DocumentStats",
    "generate_report",
]
