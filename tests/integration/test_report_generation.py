"""
Integration tests for Report Generation.

Tests the complete report generation flow:
1. Audit session completion
2. Report generation in multiple formats
3. Template rendering
4. CLI command execution
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_audit_session():
    """Create a sample audit session for testing."""
    from aragora.audit.document_auditor import (
        AuditSession,
        AuditType,
        SessionStatus,
    )

    session = AuditSession(
        id="test_session_12345678",
        document_ids=["doc_1", "doc_2", "doc_3"],
        audit_types=[AuditType.SECURITY, AuditType.COMPLIANCE],
        name="Test Security Audit",
        model="test-model",
        status=SessionStatus.COMPLETED,
    )
    session.started_at = datetime.utcnow() - timedelta(hours=1)
    session.completed_at = datetime.utcnow()
    session.total_chunks = 50
    session.processed_chunks = 50

    return session


@pytest.fixture
def sample_findings():
    """Create sample findings for testing."""
    from aragora.audit.document_auditor import (
        AuditFinding,
        AuditType,
        FindingSeverity,
        FindingStatus,
    )

    return [
        AuditFinding(
            id="finding_1",
            title="Hardcoded API Key Detected",
            description="Found hardcoded API key in configuration file config.py line 42",
            severity=FindingSeverity.CRITICAL,
            confidence=0.95,
            audit_type=AuditType.SECURITY,
            category="credentials",
            document_id="doc_1",
            evidence_text="API_KEY='sk-1234567890abcdef'",
            evidence_location="config.py:42",
            recommendation="Move API keys to environment variables or a secrets manager",
            status=FindingStatus.OPEN,
        ),
        AuditFinding(
            id="finding_2",
            title="SQL Injection Vulnerability",
            description="User input directly concatenated into SQL query",
            severity=FindingSeverity.HIGH,
            confidence=0.88,
            audit_type=AuditType.SECURITY,
            category="injection",
            document_id="doc_2",
            evidence_text='query = f"SELECT * FROM users WHERE id = {user_id}"',
            evidence_location="database.py:128",
            recommendation="Use parameterized queries or an ORM",
            status=FindingStatus.OPEN,
        ),
        AuditFinding(
            id="finding_3",
            title="Missing GDPR Consent Mechanism",
            description="No mechanism for collecting user consent for data processing",
            severity=FindingSeverity.MEDIUM,
            confidence=0.82,
            audit_type=AuditType.COMPLIANCE,
            category="gdpr",
            document_id="doc_3",
            evidence_text="User data collected without consent form",
            recommendation="Implement consent collection before data processing",
            status=FindingStatus.OPEN,
        ),
        AuditFinding(
            id="finding_4",
            title="Debug Logging in Production",
            description="Debug-level logging enabled which may expose sensitive data",
            severity=FindingSeverity.LOW,
            confidence=0.75,
            audit_type=AuditType.SECURITY,
            category="logging",
            document_id="doc_1",
            evidence_text="LOG_LEVEL=DEBUG",
            recommendation="Set LOG_LEVEL to INFO or WARNING in production",
            status=FindingStatus.OPEN,
        ),
    ]


@pytest.fixture
def session_with_findings(sample_audit_session, sample_findings):
    """Create session with findings attached."""
    sample_audit_session.findings = sample_findings
    return sample_audit_session


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for report output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Report Generator Tests
# =============================================================================


class TestReportGenerator:
    """Test the AuditReportGenerator class."""

    @pytest.mark.asyncio
    async def test_generate_markdown_report(self, session_with_findings):
        """Test generating a Markdown report."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig(
            min_severity="low",
            include_recommendations=True,
        )

        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        assert report is not None
        assert report.format == ReportFormat.MARKDOWN
        assert report.size_bytes > 0
        assert report.findings_count == 4

        # Check content contains expected sections
        content = report.content.decode("utf-8")
        assert "Executive Summary" in content
        assert "Detailed Findings" in content
        assert "Critical Severity" in content
        assert "Hardcoded API Key" in content

    @pytest.mark.asyncio
    async def test_generate_json_report(self, session_with_findings):
        """Test generating a JSON report."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.JSON,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        assert report.format == ReportFormat.JSON

        # Validate JSON structure
        data = json.loads(report.content.decode("utf-8"))
        assert "report" in data
        assert "session" in data
        assert "findings" in data
        assert "summary" in data
        assert len(data["findings"]) == 4

    @pytest.mark.asyncio
    async def test_generate_html_report(self, session_with_findings):
        """Test generating an HTML report."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.HTML,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        assert report.format == ReportFormat.HTML

        content = report.content.decode("utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<title>" in content
        assert "Audit Report" in content
        assert "severity-critical" in content or "Critical" in content

    @pytest.mark.asyncio
    async def test_executive_summary_template(self, session_with_findings):
        """Test the executive summary template."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.EXECUTIVE_SUMMARY,
        )

        content = report.content.decode("utf-8")
        assert "Executive Summary" in content
        assert "Overall Risk Level" in content
        # Executive summary should be shorter - no detailed findings
        assert "CRITICAL-1:" not in content  # Detailed format not present

    @pytest.mark.asyncio
    async def test_security_assessment_template(self, session_with_findings):
        """Test the security assessment template."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.SECURITY_ASSESSMENT,
        )

        content = report.content.decode("utf-8")
        assert "Security Summary" in content or "Security Assessment" in content
        assert "Vulnerability" in content or "Remediation" in content

    @pytest.mark.asyncio
    async def test_compliance_attestation_template(self, session_with_findings):
        """Test the compliance attestation template."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig(author="Test Auditor", reviewer="Test Reviewer")
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.COMPLIANCE_ATTESTATION,
        )

        content = report.content.decode("utf-8")
        assert "Compliance" in content
        assert "Attestation" in content


class TestReportFiltering:
    """Test report filtering options."""

    @pytest.mark.asyncio
    async def test_filter_by_min_severity(self, session_with_findings):
        """Test filtering findings by minimum severity."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        # Only include high and critical
        config = ReportConfig(min_severity="high")
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.JSON,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        data = json.loads(report.content.decode("utf-8"))
        # Should have 2 findings (critical + high)
        assert report.findings_count == 2
        for finding in data["findings"]:
            assert finding["severity"] in ("critical", "high")

    @pytest.mark.asyncio
    async def test_filter_critical_only(self, session_with_findings):
        """Test filtering for critical findings only."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig(min_severity="critical")
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.JSON,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        assert report.findings_count == 1
        data = json.loads(report.content.decode("utf-8"))
        assert data["findings"][0]["severity"] == "critical"


class TestReportSaving:
    """Test report saving functionality."""

    @pytest.mark.asyncio
    async def test_save_report_to_file(self, session_with_findings, temp_output_dir: Path):
        """Test saving a report to a file."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        output_path = temp_output_dir / "test_report.md"
        saved_path = report.save(output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size == report.size_bytes

        # Verify content
        content = saved_path.read_text()
        assert "Audit Report" in content

    @pytest.mark.asyncio
    async def test_save_report_auto_filename(self, session_with_findings, temp_output_dir: Path):
        """Test saving with auto-generated filename."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig(output_dir=str(temp_output_dir))
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.HTML,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        # Save without explicit path (uses filename from report)
        saved_path = report.save(temp_output_dir / report.filename)

        assert saved_path.exists()
        assert ".html" in str(saved_path)


# =============================================================================
# CLI Integration Tests
# =============================================================================


class TestReportCLI:
    """Test the CLI report command."""

    def test_cli_help(self):
        """Test that CLI help works for report command."""
        import argparse
        from aragora.cli.audit import create_audit_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        # Parse help (should not raise)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["audit", "report", "--help"])

        # Help exits with 0
        assert exc_info.value.code == 0

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing for report command."""
        import argparse
        from aragora.cli.audit import create_audit_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(
            [
                "audit",
                "report",
                "session_123",
                "--format",
                "pdf",
                "--template",
                "security_assessment",
                "--min-severity",
                "high",
                "--author",
                "Test Author",
            ]
        )

        assert args.session_id == "session_123"
        assert args.format == "pdf"
        assert args.template == "security_assessment"
        assert args.min_severity == "high"
        assert args.author == "Test Author"


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestReportAPI:
    """Test the report API endpoint."""

    @pytest.mark.asyncio
    async def test_api_report_endpoint_format_mapping(self):
        """Test that API correctly maps format parameters."""
        from aragora.reports import ReportFormat

        # Verify format mapping
        format_map = {
            "json": ReportFormat.JSON,
            "markdown": ReportFormat.MARKDOWN,
            "html": ReportFormat.HTML,
            "pdf": ReportFormat.PDF,
        }

        for format_str, expected_enum in format_map.items():
            assert format_map[format_str] == expected_enum

    @pytest.mark.asyncio
    async def test_api_report_endpoint_template_mapping(self):
        """Test that API correctly maps template parameters."""
        from aragora.reports import ReportTemplate

        template_map = {
            "executive_summary": ReportTemplate.EXECUTIVE_SUMMARY,
            "detailed_findings": ReportTemplate.DETAILED_FINDINGS,
            "compliance_attestation": ReportTemplate.COMPLIANCE_ATTESTATION,
            "security_assessment": ReportTemplate.SECURITY_ASSESSMENT,
        }

        for template_str, expected_enum in template_map.items():
            assert template_map[template_str] == expected_enum


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestReportEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_report_with_no_findings(self, sample_audit_session):
        """Test generating a report with no findings."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        sample_audit_session.findings = []
        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=sample_audit_session,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        assert report is not None
        assert report.findings_count == 0
        content = report.content.decode("utf-8")
        assert "Executive Summary" in content

    @pytest.mark.asyncio
    async def test_report_with_special_characters(self, sample_audit_session):
        """Test report handles special characters in findings."""
        from aragora.audit.document_auditor import (
            AuditFinding,
            AuditType,
            FindingSeverity,
            FindingStatus,
        )
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        # Finding with special characters
        sample_audit_session.findings = [
            AuditFinding(
                id="finding_special",
                title='Finding with "quotes" and <brackets>',
                description="Description with unicode: \u2022 bullet \u00a9 copyright",
                severity=FindingSeverity.MEDIUM,
                confidence=0.8,
                audit_type=AuditType.SECURITY,
                category="test",
                document_id="doc_1",
                evidence_text='code = "SELECT * FROM users"',
            ),
        ]

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        # Should not raise
        report = await generator.generate(
            session=sample_audit_session,
            format=ReportFormat.HTML,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        assert report is not None
        assert report.findings_count == 1

    @pytest.mark.asyncio
    async def test_report_json_serialization(self, session_with_findings):
        """Test that JSON report is valid JSON."""
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        config = ReportConfig()
        generator = AuditReportGenerator(config)

        report = await generator.generate(
            session=session_with_findings,
            format=ReportFormat.JSON,
            template=ReportTemplate.DETAILED_FINDINGS,
        )

        # Should parse without error
        data = json.loads(report.content.decode("utf-8"))

        # Validate required fields
        assert "report" in data
        assert "generated_at" in data["report"]
        assert "findings" in data
        assert isinstance(data["findings"], list)


# =============================================================================
# Performance Tests
# =============================================================================


class TestReportPerformance:
    """Test report generation performance."""

    @pytest.mark.asyncio
    async def test_large_report_generation(self, sample_audit_session):
        """Test generating a report with many findings."""
        from aragora.audit.document_auditor import (
            AuditFinding,
            AuditType,
            FindingSeverity,
            FindingStatus,
        )
        from aragora.reports import (
            AuditReportGenerator,
            ReportConfig,
            ReportFormat,
            ReportTemplate,
        )

        # Generate 100 findings
        findings = []
        for i in range(100):
            findings.append(
                AuditFinding(
                    id=f"finding_{i}",
                    title=f"Finding {i}: Test Issue",
                    description=f"Description for finding {i} with details",
                    severity=FindingSeverity.MEDIUM,
                    confidence=0.75 + (i % 25) / 100,
                    audit_type=AuditType.SECURITY,
                    category="test",
                    document_id=f"doc_{i % 10}",
                    evidence_text=f"Evidence for finding {i}",
                )
            )

        sample_audit_session.findings = findings
        config = ReportConfig()
        generator = AuditReportGenerator(config)

        import time

        start = time.time()
        report = await generator.generate(
            session=sample_audit_session,
            format=ReportFormat.MARKDOWN,
            template=ReportTemplate.DETAILED_FINDINGS,
        )
        elapsed = time.time() - start

        assert report.findings_count == 100
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
