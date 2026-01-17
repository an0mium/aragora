"""
Tests for defect report generation.
"""

import pytest
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from aragora.audit.reports.defect_report import (
    DefectReport,
    ReportConfig,
    ReportFormat,
    SeverityStats,
    CategoryStats,
    DocumentStats,
    generate_report,
)


# Mock Finding class for testing
class MockSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MockAuditType(str, Enum):
    SECURITY = "security"
    COMPLIANCE = "compliance"
    CONSISTENCY = "consistency"
    QUALITY = "quality"


@dataclass
class MockFinding:
    """Mock audit finding for testing."""

    id: str = "finding-1"
    title: str = "Test Finding"
    description: str = "This is a test finding."
    severity: MockSeverity = MockSeverity.MEDIUM
    audit_type: MockAuditType = MockAuditType.QUALITY
    category: str = "documentation"
    document_id: str = "doc-1"
    evidence_text: str = "Evidence snippet here"
    evidence_location: str = "Page 5"
    remediation: str = "Fix the issue"
    confidence: float = 0.85

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "audit_type": self.audit_type.value,
            "category": self.category,
            "document_id": self.document_id,
            "evidence_text": self.evidence_text,
            "evidence_location": self.evidence_location,
            "remediation": self.remediation,
            "confidence": self.confidence,
        }


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReportConfig()
        assert config.format == ReportFormat.MARKDOWN
        assert config.include_summary is True
        assert config.include_severity_breakdown is True
        assert config.group_by == "severity"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            format=ReportFormat.JSON,
            title="Security Audit Report",
            min_severity="high",
            max_findings=50,
        )
        assert config.format == ReportFormat.JSON
        assert config.title == "Security Audit Report"
        assert config.min_severity == "high"
        assert config.max_findings == 50


class TestSeverityStats:
    """Tests for SeverityStats dataclass."""

    def test_empty_stats(self):
        """Test empty statistics."""
        stats = SeverityStats()
        assert stats.total == 0

    def test_counting(self):
        """Test severity counting."""
        stats = SeverityStats(critical=2, high=5, medium=10, low=3, info=1)
        assert stats.total == 21

    def test_to_dict(self):
        """Test serialization."""
        stats = SeverityStats(critical=1, high=2)
        data = stats.to_dict()
        assert data["critical"] == 1
        assert data["high"] == 2
        assert data["total"] == 3


class TestCategoryStats:
    """Tests for CategoryStats dataclass."""

    def test_add_categories(self):
        """Test adding categories."""
        stats = CategoryStats()
        stats.add("security")
        stats.add("security")
        stats.add("compliance")

        assert stats.counts["security"] == 2
        assert stats.counts["compliance"] == 1

    def test_to_dict_sorted(self):
        """Test that to_dict returns sorted categories."""
        stats = CategoryStats()
        stats.add("low-count")
        stats.add("high-count")
        stats.add("high-count")
        stats.add("high-count")

        data = stats.to_dict()
        keys = list(data.keys())
        # Should be sorted by count descending
        assert keys[0] == "high-count"


class TestDocumentStats:
    """Tests for DocumentStats dataclass."""

    def test_add_findings(self):
        """Test adding findings per document."""
        stats = DocumentStats()
        stats.add_finding("doc-1", "critical")
        stats.add_finding("doc-1", "high")
        stats.add_finding("doc-2", "medium")

        assert stats.findings_per_doc["doc-1"] == 2
        assert stats.findings_per_doc["doc-2"] == 1
        assert stats.severity_per_doc["doc-1"].critical == 1
        assert stats.severity_per_doc["doc-1"].high == 1


class TestDefectReport:
    """Tests for DefectReport class."""

    @pytest.fixture
    def sample_findings(self):
        """Create sample findings for testing."""
        return [
            MockFinding(
                id="f1",
                title="SQL Injection Vulnerability",
                severity=MockSeverity.CRITICAL,
                audit_type=MockAuditType.SECURITY,
                category="injection",
                document_id="doc-1",
            ),
            MockFinding(
                id="f2",
                title="Missing PII Handling",
                severity=MockSeverity.HIGH,
                audit_type=MockAuditType.COMPLIANCE,
                category="gdpr",
                document_id="doc-2",
            ),
            MockFinding(
                id="f3",
                title="Outdated Reference",
                severity=MockSeverity.MEDIUM,
                audit_type=MockAuditType.CONSISTENCY,
                category="references",
                document_id="doc-1",
            ),
            MockFinding(
                id="f4",
                title="Missing Documentation",
                severity=MockSeverity.LOW,
                audit_type=MockAuditType.QUALITY,
                category="documentation",
                document_id="doc-2",
            ),
        ]

    def test_report_initialization(self, sample_findings):
        """Test report initialization."""
        report = DefectReport(sample_findings, session_id="test-session")
        assert len(report.findings) == 4
        assert report.session_id == "test-session"

    def test_severity_stats_computed(self, sample_findings):
        """Test that severity stats are computed correctly."""
        report = DefectReport(sample_findings)
        assert report.severity_stats.critical == 1
        assert report.severity_stats.high == 1
        assert report.severity_stats.medium == 1
        assert report.severity_stats.low == 1
        assert report.severity_stats.total == 4

    def test_category_stats_computed(self, sample_findings):
        """Test that category stats are computed correctly."""
        report = DefectReport(sample_findings)
        assert "injection" in report.category_stats.counts
        assert "gdpr" in report.category_stats.counts

    def test_document_stats_computed(self, sample_findings):
        """Test that document stats are computed correctly."""
        report = DefectReport(sample_findings)
        assert report.document_stats.findings_per_doc["doc-1"] == 2
        assert report.document_stats.findings_per_doc["doc-2"] == 2

    def test_filter_by_severity(self, sample_findings):
        """Test filtering findings by minimum severity."""
        config = ReportConfig(min_severity="high")
        report = DefectReport(sample_findings, config=config)
        # Should only include critical and high
        assert len(report.findings) == 2

    def test_filter_by_audit_type(self, sample_findings):
        """Test filtering by audit type."""
        config = ReportConfig(audit_types=["security", "compliance"])
        report = DefectReport(sample_findings, config=config)
        assert len(report.findings) == 2

    def test_max_findings_limit(self, sample_findings):
        """Test limiting number of findings."""
        config = ReportConfig(max_findings=2)
        report = DefectReport(sample_findings, config=config)
        assert len(report.findings) == 2

    def test_to_dict(self, sample_findings):
        """Test conversion to dictionary."""
        report = DefectReport(sample_findings, session_id="test-123")
        data = report.to_dict()

        assert "meta" in data
        assert data["meta"]["session_id"] == "test-123"
        assert data["meta"]["total_findings"] == 4
        assert "summary" in data
        assert "findings" in data
        assert len(data["findings"]) == 4

    def test_to_json(self, sample_findings):
        """Test JSON output."""
        report = DefectReport(sample_findings)
        json_output = report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_output)
        assert "meta" in parsed
        assert "findings" in parsed

    def test_to_markdown(self, sample_findings):
        """Test Markdown output."""
        config = ReportConfig(title="Security Audit Report")
        report = DefectReport(sample_findings, config=config)
        md = report.to_markdown()

        assert "# Security Audit Report" in md
        assert "## Executive Summary" in md
        assert "Total findings: **4**" in md
        assert "[CRITICAL]" in md
        assert "SQL Injection Vulnerability" in md

    def test_markdown_includes_severity_breakdown(self, sample_findings):
        """Test that Markdown includes severity breakdown."""
        config = ReportConfig(include_severity_breakdown=True)
        report = DefectReport(sample_findings, config=config)
        md = report.to_markdown()

        assert "### Severity Breakdown" in md
        assert "| Critical |" in md

    def test_markdown_includes_evidence(self, sample_findings):
        """Test that Markdown includes evidence when configured."""
        config = ReportConfig(include_evidence=True)
        report = DefectReport(sample_findings, config=config)
        md = report.to_markdown()

        assert "**Evidence:**" in md

    def test_to_html(self, sample_findings):
        """Test HTML output."""
        report = DefectReport(sample_findings)
        html = report.to_html()

        assert "<!DOCTYPE html>" in html
        assert "<style>" in html
        assert "Document Audit Report" in html
        assert 'class="critical"' in html

    def test_to_csv(self, sample_findings):
        """Test CSV output."""
        report = DefectReport(sample_findings)
        csv = report.to_csv()

        lines = csv.strip().split("\n")
        # Header + 4 findings
        assert len(lines) == 5
        assert "ID,Title,Severity" in lines[0]
        assert "f1" in lines[1]

    def test_group_by_severity(self, sample_findings):
        """Test grouping by severity."""
        config = ReportConfig(group_by="severity")
        report = DefectReport(sample_findings, config=config)
        md = report.to_markdown()

        # Critical should come before High
        critical_pos = md.find("### CRITICAL")
        high_pos = md.find("### HIGH")
        assert critical_pos < high_pos

    def test_group_by_category(self, sample_findings):
        """Test grouping by category."""
        config = ReportConfig(group_by="category")
        report = DefectReport(sample_findings, config=config)
        grouped = report._group_findings()

        assert "injection" in grouped
        assert "gdpr" in grouped

    def test_group_by_document(self, sample_findings):
        """Test grouping by document."""
        config = ReportConfig(group_by="document")
        report = DefectReport(sample_findings, config=config)
        grouped = report._group_findings()

        assert "doc-1" in grouped
        assert "doc-2" in grouped

    def test_generate_with_format(self, sample_findings):
        """Test generate method with different formats."""
        report = DefectReport(sample_findings)

        json_out = report.generate(ReportFormat.JSON)
        md_out = report.generate(ReportFormat.MARKDOWN)
        html_out = report.generate(ReportFormat.HTML)
        csv_out = report.generate(ReportFormat.CSV)

        assert json_out.startswith("{")
        assert md_out.startswith("#")
        assert html_out.startswith("<!DOCTYPE")
        assert "ID,Title" in csv_out


class TestGenerateReportFunction:
    """Tests for convenience generate_report function."""

    def test_generate_report_simple(self):
        """Test simple report generation."""
        findings = [
            MockFinding(id="f1", title="Test Issue"),
        ]
        output = generate_report(findings, format=ReportFormat.MARKDOWN)
        assert "Test Issue" in output

    def test_generate_report_with_session_id(self):
        """Test report generation with session ID."""
        findings = [MockFinding()]
        output = generate_report(
            findings,
            format=ReportFormat.MARKDOWN,
            session_id="session-123",
        )
        assert "session-123" in output

    def test_generate_report_with_config_kwargs(self):
        """Test report generation with config kwargs."""
        findings = [
            MockFinding(severity=MockSeverity.LOW),
            MockFinding(severity=MockSeverity.CRITICAL),
        ]
        output = generate_report(
            findings,
            format=ReportFormat.MARKDOWN,
            min_severity="high",
        )
        # Should only include critical
        assert "[CRITICAL]" in output
        assert "[LOW]" not in output


class TestDictFindings:
    """Tests with dictionary-based findings."""

    def test_dict_findings(self):
        """Test report with dictionary findings."""
        findings = [
            {
                "id": "d1",
                "title": "Dict Finding",
                "severity": "high",
                "audit_type": "security",
                "category": "test",
                "document_id": "doc-1",
            }
        ]
        report = DefectReport(findings)
        md = report.to_markdown()

        assert "Dict Finding" in md
        assert "[HIGH]" in md

    def test_mixed_findings(self):
        """Test report with mixed finding types."""
        findings = [
            MockFinding(id="f1", title="Object Finding"),
            {"id": "f2", "title": "Dict Finding", "severity": "medium"},
        ]
        report = DefectReport(findings)
        assert len(report.findings) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_findings(self):
        """Test report with no findings."""
        report = DefectReport([])
        assert report.severity_stats.total == 0

        md = report.to_markdown()
        assert "Total findings: **0**" in md

    def test_findings_without_optional_fields(self):
        """Test findings missing optional fields."""
        findings = [{"id": "f1", "title": "Minimal Finding"}]
        report = DefectReport(findings)
        md = report.to_markdown()

        assert "Minimal Finding" in md

    def test_very_long_evidence(self):
        """Test truncation of long evidence in CSV."""
        findings = [
            MockFinding(
                evidence_text="A" * 1000,  # Very long evidence
            )
        ]
        report = DefectReport(findings)
        csv = report.to_csv()

        # Evidence should be truncated
        lines = csv.split("\n")
        assert len(lines[1]) < 1500  # Should be truncated
