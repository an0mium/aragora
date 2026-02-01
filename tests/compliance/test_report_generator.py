"""
Tests for Compliance Report Generator.

Tests cover:
- Report generation from debate results
- Template rendering
- Data aggregation and sections
- Export formats (JSON, Markdown)
- Compliance status calculation
- Framework-specific sections (SOC2, GDPR, HIPAA, ISO27001)
- Attestation building
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.compliance.report_generator import (
    ComplianceFramework,
    ComplianceReport,
    ComplianceReportGenerator,
    ReportFormat,
    ReportSection,
    generate_gdpr_report,
    generate_soc2_report,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockDebateResult:
    """Mock DebateResult for testing."""

    task: str = "Test debate task"
    consensus_reached: bool = True
    confidence: float = 0.85
    rounds_used: int = 3
    winner: str | None = "proposal_a"
    final_answer: str | None = "This is the final answer from the debate"
    agents: list[str] = field(default_factory=lambda: ["claude", "gpt4", "gemini"])
    history: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def mock_debate_result():
    """Create a mock debate result for testing."""
    return MockDebateResult(
        task="Evaluate the risk assessment for Q4 2024",
        consensus_reached=True,
        confidence=0.92,
        rounds_used=4,
        winner="claude",
        final_answer="Based on the analysis, the risk level is moderate with mitigation strategies recommended.",
        agents=["claude", "gpt4", "gemini", "llama"],
        history=[
            {"agent": "claude", "content": "Initial proposal...", "round": 1},
            {"agent": "gpt4", "content": "Counter proposal...", "round": 1},
            {"agent": "gemini", "content": "Alternative view...", "round": 2},
        ],
    )


@pytest.fixture
def generator():
    """Create a ComplianceReportGenerator for testing."""
    return ComplianceReportGenerator(organization="TestOrg Inc")


# =============================================================================
# ComplianceFramework Enum Tests
# =============================================================================


class TestComplianceFramework:
    """Tests for ComplianceFramework enum."""

    def test_all_frameworks_exist(self):
        """Test that all expected frameworks are defined."""
        assert ComplianceFramework.SOC2.value == "soc2"
        assert ComplianceFramework.GDPR.value == "gdpr"
        assert ComplianceFramework.HIPAA.value == "hipaa"
        assert ComplianceFramework.ISO27001.value == "iso27001"
        assert ComplianceFramework.CUSTOM.value == "custom"
        assert ComplianceFramework.GENERAL.value == "general"

    def test_framework_count(self):
        """Test there are exactly 6 frameworks."""
        assert len(ComplianceFramework) == 6


# =============================================================================
# ReportFormat Enum Tests
# =============================================================================


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_all_formats_exist(self):
        """Test that all expected formats are defined."""
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.PDF.value == "pdf"

    def test_format_count(self):
        """Test there are exactly 4 formats."""
        assert len(ReportFormat) == 4


# =============================================================================
# ReportSection Dataclass Tests
# =============================================================================


class TestReportSection:
    """Tests for ReportSection dataclass."""

    def test_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            title="Executive Summary",
            content="This is the summary content.",
            data={"key": "value"},
        )
        assert section.title == "Executive Summary"
        assert section.content == "This is the summary content."
        assert section.data == {"key": "value"}

    def test_section_default_values(self):
        """Test section creation with default values."""
        section = ReportSection(
            title="Test Section",
            content="Content here",
        )
        assert section.data == {}
        assert section.subsections == []
        assert section.metadata == {}

    def test_section_with_subsections(self):
        """Test section with nested subsections."""
        subsection = ReportSection(
            title="Subsection 1",
            content="Subsection content",
        )
        section = ReportSection(
            title="Main Section",
            content="Main content",
            subsections=[subsection],
        )
        assert len(section.subsections) == 1
        assert section.subsections[0].title == "Subsection 1"


# =============================================================================
# ComplianceReport Dataclass Tests
# =============================================================================


class TestComplianceReport:
    """Tests for ComplianceReport dataclass."""

    def test_report_creation(self, mock_debate_result, generator):
        """Test creating a compliance report."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="debate-123",
            framework=ComplianceFramework.GENERAL,
        )
        assert report.debate_id == "debate-123"
        assert report.framework == ComplianceFramework.GENERAL
        assert report.report_id.startswith("CR-")
        assert len(report.sections) > 0

    def test_report_to_dict(self, mock_debate_result, generator):
        """Test converting report to dictionary."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="debate-456",
        )
        report_dict = report.to_dict()

        assert report_dict["debate_id"] == "debate-456"
        assert report_dict["framework"] == "general"
        assert "sections" in report_dict
        assert "attestation" in report_dict
        assert "summary" in report_dict

    def test_report_to_dict_sections_format(self, mock_debate_result, generator):
        """Test section serialization in to_dict."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="debate-789",
        )
        report_dict = report.to_dict()

        # Check section structure
        for section in report_dict["sections"]:
            assert "title" in section
            assert "content" in section
            assert "data" in section
            assert "subsections" in section
            assert "metadata" in section


# =============================================================================
# ComplianceReportGenerator Tests
# =============================================================================


class TestComplianceReportGenerator:
    """Tests for ComplianceReportGenerator class."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = ComplianceReportGenerator(organization="TestOrg")
        assert generator.organization == "TestOrg"
        assert generator.templates == {}

    def test_generator_initialization_with_templates(self):
        """Test generator initialization with custom templates."""
        templates = {"custom_section": "Template content here"}
        generator = ComplianceReportGenerator(
            organization="TestOrg",
            templates=templates,
        )
        assert generator.templates == templates

    def test_generator_default_organization(self):
        """Test generator uses default organization."""
        generator = ComplianceReportGenerator()
        assert generator.organization == "Aragora"

    def test_generate_basic_report(self, mock_debate_result, generator):
        """Test generating a basic report."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-debate-001",
        )

        assert isinstance(report, ComplianceReport)
        assert report.debate_id == "test-debate-001"
        assert "TestOrg Inc" in report.generated_by

    def test_generate_report_with_all_options(self, mock_debate_result, generator):
        """Test generating report with all optional parameters."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-debate-002",
            framework=ComplianceFramework.SOC2,
            include_evidence=True,
            include_chain=True,
            include_full_transcript=True,
            requester="user@example.com",
            additional_context={"department": "Legal"},
        )

        assert report.framework == ComplianceFramework.SOC2
        assert report.metadata["include_evidence"] is True
        assert report.metadata["include_chain"] is True
        assert report.metadata["include_transcript"] is True
        assert report.metadata["additional_context"]["department"] == "Legal"
        assert report.attestation["requester"] == "user@example.com"

    def test_generate_report_without_evidence(self, mock_debate_result, generator):
        """Test generating report without evidence section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-debate-003",
            include_evidence=False,
        )

        section_titles = [s.title for s in report.sections]
        assert "Evidence Citations" not in section_titles

    def test_generate_report_without_chain(self, mock_debate_result, generator):
        """Test generating report without provenance chain section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-debate-004",
            include_chain=False,
        )

        section_titles = [s.title for s in report.sections]
        assert "Provenance Chain" not in section_titles


# =============================================================================
# Section Building Tests
# =============================================================================


class TestSectionBuilding:
    """Tests for individual section building methods."""

    def test_executive_summary_section(self, mock_debate_result, generator):
        """Test executive summary section building."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-exec-001",
        )

        exec_summary = next(s for s in report.sections if s.title == "Executive Summary")
        assert exec_summary is not None
        assert "REACHED" in exec_summary.content
        assert mock_debate_result.task in exec_summary.content
        assert exec_summary.data["consensus_reached"] is True

    def test_decision_overview_section(self, mock_debate_result, generator):
        """Test decision overview section building."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-decision-001",
        )

        decision_section = next(s for s in report.sections if s.title == "Decision Overview")
        assert decision_section is not None
        assert mock_debate_result.winner in decision_section.content
        assert decision_section.data["winner"] == mock_debate_result.winner

    def test_participants_section(self, mock_debate_result, generator):
        """Test participants section building."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-participants-001",
        )

        participants_section = next(s for s in report.sections if s.title == "Participants")
        assert participants_section is not None
        assert participants_section.data["count"] == len(mock_debate_result.agents)
        for agent in mock_debate_result.agents:
            assert agent in participants_section.content

    def test_process_section(self, mock_debate_result, generator):
        """Test process details section building."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-process-001",
        )

        process_section = next(s for s in report.sections if s.title == "Process Details")
        assert process_section is not None
        assert str(mock_debate_result.rounds_used) in process_section.content
        assert process_section.data["rounds"] == mock_debate_result.rounds_used

    def test_transcript_section_with_history(self, mock_debate_result, generator):
        """Test transcript section with debate history."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-transcript-001",
            include_full_transcript=True,
        )

        transcript_section = next(s for s in report.sections if s.title == "Full Transcript")
        assert transcript_section is not None
        assert transcript_section.data["entry_count"] == len(mock_debate_result.history)

    def test_consensus_not_reached(self, generator):
        """Test report when consensus is not reached."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.45,
            winner=None,
            final_answer=None,
        )
        report = generator.generate(
            debate_result=result,
            debate_id="test-no-consensus",
        )

        exec_summary = next(s for s in report.sections if s.title == "Executive Summary")
        assert "NOT REACHED" in exec_summary.content


# =============================================================================
# Framework-Specific Section Tests
# =============================================================================


class TestFrameworkSections:
    """Tests for framework-specific report sections."""

    def test_soc2_section(self, mock_debate_result, generator):
        """Test SOC2 compliance section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-soc2",
            framework=ComplianceFramework.SOC2,
        )

        soc2_section = next(
            (s for s in report.sections if s.title == "SOC2 Compliance"),
            None,
        )
        assert soc2_section is not None
        assert "Trust Service Criteria" in soc2_section.content
        assert soc2_section.data["framework"] == "SOC2"
        assert soc2_section.data["compliant"] is True

    def test_gdpr_section(self, mock_debate_result, generator):
        """Test GDPR compliance section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-gdpr",
            framework=ComplianceFramework.GDPR,
        )

        gdpr_section = next(
            (s for s in report.sections if s.title == "GDPR Compliance"),
            None,
        )
        assert gdpr_section is not None
        assert "Article" in gdpr_section.content
        assert gdpr_section.data["framework"] == "GDPR"

    def test_hipaa_section(self, mock_debate_result, generator):
        """Test HIPAA compliance section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-hipaa",
            framework=ComplianceFramework.HIPAA,
        )

        hipaa_section = next(
            (s for s in report.sections if s.title == "HIPAA Compliance"),
            None,
        )
        assert hipaa_section is not None
        assert "Safeguard" in hipaa_section.content
        assert hipaa_section.data["framework"] == "HIPAA"
        assert "phi_processed" in hipaa_section.data

    def test_iso27001_section(self, mock_debate_result, generator):
        """Test ISO 27001 compliance section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-iso27001",
            framework=ComplianceFramework.ISO27001,
        )

        iso_section = next(
            (s for s in report.sections if s.title == "ISO 27001 Compliance"),
            None,
        )
        assert iso_section is not None
        assert "Control" in iso_section.content
        assert iso_section.data["framework"] == "ISO27001"

    def test_general_framework_no_specific_section(self, mock_debate_result, generator):
        """Test GENERAL framework adds no specific compliance section."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-general",
            framework=ComplianceFramework.GENERAL,
        )

        # Should not have any framework-specific sections
        framework_sections = [
            s
            for s in report.sections
            if s.title
            in ["SOC2 Compliance", "GDPR Compliance", "HIPAA Compliance", "ISO 27001 Compliance"]
        ]
        assert len(framework_sections) == 0


# =============================================================================
# Attestation Tests
# =============================================================================


class TestAttestation:
    """Tests for attestation building."""

    def test_attestation_structure(self, mock_debate_result, generator):
        """Test attestation has correct structure."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-attestation-001",
            requester="auditor@company.com",
        )

        attestation = report.attestation
        assert "timestamp" in attestation
        assert "hash" in attestation
        assert "organization" in attestation
        assert "framework" in attestation
        assert "requester" in attestation
        assert "statement" in attestation

    def test_attestation_hash_format(self, mock_debate_result, generator):
        """Test attestation hash is valid SHA-256 format."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-attestation-002",
        )

        # SHA-256 hex is 64 characters
        assert len(report.attestation["hash"]) == 64
        assert all(c in "0123456789abcdef" for c in report.attestation["hash"])

    def test_attestation_organization(self, mock_debate_result, generator):
        """Test attestation includes organization."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-attestation-003",
        )

        assert report.attestation["organization"] == "TestOrg Inc"


# =============================================================================
# Export Format Tests
# =============================================================================


class TestExportFormats:
    """Tests for report export functionality."""

    def test_export_json(self, mock_debate_result, generator):
        """Test JSON export format."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-export-json",
        )

        json_output = generator.export_json(report)

        # Should be valid JSON
        parsed = json.loads(json_output)
        assert parsed["debate_id"] == "test-export-json"
        assert "sections" in parsed
        assert "attestation" in parsed

    def test_export_json_formatting(self, mock_debate_result, generator):
        """Test JSON export is properly formatted with indentation."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-json-format",
        )

        json_output = generator.export_json(report)

        # Indented JSON should have newlines
        assert "\n" in json_output
        # Should have consistent indentation
        assert "  " in json_output

    def test_export_markdown(self, mock_debate_result, generator):
        """Test Markdown export format."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-export-md",
        )

        md_output = generator.export_markdown(report)

        # Check markdown structure
        assert f"# Compliance Report: {report.report_id}" in md_output
        assert "**Debate ID:**" in md_output
        assert "## Summary" in md_output
        assert "## Attestation" in md_output
        assert "---" in md_output

    def test_export_markdown_sections(self, mock_debate_result, generator):
        """Test Markdown export includes all sections."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-md-sections",
        )

        md_output = generator.export_markdown(report)

        # Check section headers
        assert "## Executive Summary" in md_output
        assert "## Decision Overview" in md_output
        assert "## Participants" in md_output
        assert "## Process Details" in md_output


# =============================================================================
# Report ID Generation Tests
# =============================================================================


class TestReportIdGeneration:
    """Tests for report ID generation."""

    def test_report_id_format(self, mock_debate_result, generator):
        """Test report ID follows expected format."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-id-format",
        )

        # Format: CR-XXXXXXXXXXXX (CR- prefix + 12 hex chars)
        assert report.report_id.startswith("CR-")
        assert len(report.report_id) == 15  # CR- (3) + 12 chars

    def test_report_id_uniqueness(self, mock_debate_result, generator):
        """Test that generated report IDs are unique."""
        report1 = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-unique-1",
        )
        report2 = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-unique-2",
        )

        assert report1.report_id != report2.report_id


# =============================================================================
# Summary Building Tests
# =============================================================================


class TestSummaryBuilding:
    """Tests for report summary building."""

    def test_summary_content(self, mock_debate_result, generator):
        """Test summary contains key information."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-summary",
        )

        summary = report.summary
        assert "consensus reached" in summary.lower()
        assert str(mock_debate_result.rounds_used) in summary
        assert "audit trail" in summary.lower()

    def test_summary_no_consensus(self, generator):
        """Test summary when consensus not reached."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.40,
        )
        report = generator.generate(
            debate_result=result,
            debate_id="test-no-consensus-summary",
        )

        assert "not reached" in report.summary.lower()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_soc2_report(self, mock_debate_result):
        """Test SOC2 report convenience function."""
        report = generate_soc2_report(
            debate_result=mock_debate_result,
            debate_id="test-soc2-convenience",
            organization="ConvenienceOrg",
        )

        assert report.framework == ComplianceFramework.SOC2
        assert "ConvenienceOrg" in report.generated_by

    def test_generate_gdpr_report(self, mock_debate_result):
        """Test GDPR report convenience function."""
        report = generate_gdpr_report(
            debate_result=mock_debate_result,
            debate_id="test-gdpr-convenience",
        )

        assert report.framework == ComplianceFramework.GDPR

    def test_convenience_functions_default_org(self, mock_debate_result):
        """Test convenience functions use default organization."""
        report = generate_soc2_report(
            debate_result=mock_debate_result,
            debate_id="test-default-org",
        )

        assert "Aragora" in report.generated_by


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_agents_list(self, generator):
        """Test handling of empty agents list."""
        result = MockDebateResult(agents=[])
        report = generator.generate(
            debate_result=result,
            debate_id="test-empty-agents",
        )

        participants = next(s for s in report.sections if s.title == "Participants")
        assert participants.data["count"] == 0

    def test_none_winner(self, generator):
        """Test handling of None winner."""
        result = MockDebateResult(winner=None)
        report = generator.generate(
            debate_result=result,
            debate_id="test-none-winner",
        )

        decision = next(s for s in report.sections if s.title == "Decision Overview")
        assert "No clear winner" in decision.content

    def test_none_final_answer(self, generator):
        """Test handling of None final answer."""
        result = MockDebateResult(final_answer=None)
        report = generator.generate(
            debate_result=result,
            debate_id="test-none-answer",
        )

        decision = next(s for s in report.sections if s.title == "Decision Overview")
        assert "No final answer recorded" in decision.content

    def test_long_final_answer_truncation(self, generator):
        """Test long final answer is truncated in decision overview."""
        long_answer = "x" * 2000
        result = MockDebateResult(final_answer=long_answer)
        report = generator.generate(
            debate_result=result,
            debate_id="test-long-answer",
        )

        decision = next(s for s in report.sections if s.title == "Decision Overview")
        assert "..." in decision.content
        assert len(decision.content) < 1500  # Should be truncated

    def test_missing_confidence_attribute(self, generator):
        """Test handling when confidence attribute is missing."""
        # Create a minimal mock without confidence
        result = MagicMock()
        result.task = "Test task"
        result.consensus_reached = True
        result.rounds_used = 2
        result.winner = "agent1"
        result.final_answer = "Answer"
        del result.confidence  # Simulate missing attribute

        # Should use getattr default
        report = generator.generate(
            debate_result=result,
            debate_id="test-missing-confidence",
        )

        exec_summary = next(s for s in report.sections if s.title == "Executive Summary")
        assert "0%" in exec_summary.content  # Default confidence of 0.0

    def test_empty_history(self, generator):
        """Test handling of empty debate history."""
        result = MockDebateResult(history=[])
        report = generator.generate(
            debate_result=result,
            debate_id="test-empty-history",
            include_full_transcript=True,
        )

        transcript = next(s for s in report.sections if s.title == "Full Transcript")
        assert transcript.data["entry_count"] == 0
        assert "No transcript available" in transcript.content

    def test_custom_framework_no_specific_section(self, mock_debate_result, generator):
        """Test CUSTOM framework adds no predefined sections."""
        report = generator.generate(
            debate_result=mock_debate_result,
            debate_id="test-custom",
            framework=ComplianceFramework.CUSTOM,
        )

        # CUSTOM should not add framework-specific sections
        framework_sections = [
            s
            for s in report.sections
            if s.title
            in ["SOC2 Compliance", "GDPR Compliance", "HIPAA Compliance", "ISO 27001 Compliance"]
        ]
        assert len(framework_sections) == 0
