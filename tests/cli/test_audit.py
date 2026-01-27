"""
Tests for aragora.cli.audit module.

Tests audit CLI commands: presets, preset, types, create, start, status,
findings, export, report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.audit import (
    audit_cli,
    audit_findings,
    audit_status,
    create_audit,
    create_audit_parser,
    export_audit,
    generate_report,
    list_presets,
    list_types,
    show_preset,
    start_audit,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockPreset:
    """Mock audit preset."""

    name: str = "Legal Due Diligence"
    description: str = "Preset for legal document audits"
    audit_types: list = field(default_factory=lambda: ["compliance", "quality"])
    custom_rules: list = field(default_factory=list)
    consensus_threshold: float = 0.8
    agents: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)


@dataclass
class MockAuditType:
    """Mock audit type."""

    id: str = "compliance"
    display_name: str = "Compliance Check"
    description: str = "Checks document compliance"
    version: str = "1.0.0"
    capabilities: dict = field(
        default_factory=lambda: {
            "supports_chunk_analysis": True,
            "supports_cross_document": False,
            "requires_llm": True,
        }
    )


@dataclass
class MockSession:
    """Mock audit session."""

    id: str = "session-123"
    status: MagicMock = field(default_factory=lambda: MagicMock(value="completed"))
    progress: float = 1.0
    findings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": self.progress,
        }


@dataclass
class MockFinding:
    """Mock audit finding."""

    severity: MagicMock = field(default_factory=lambda: MagicMock(value="high"))
    title: str = "Missing compliance clause"
    description: str = "Document lacks required clause"
    document_id: str = "doc-1"

    def to_dict(self):
        return {
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "document_id": self.document_id,
        }


@dataclass
class MockAuditResult:
    """Mock audit result."""

    findings: list = field(default_factory=list)


@dataclass
class MockReport:
    """Mock report."""

    filename: str = "audit-report.md"
    format: MagicMock = field(default_factory=lambda: MagicMock(value="markdown"))
    size_bytes: int = 5000
    findings_count: int = 10

    def save(self, path):
        pass


# ===========================================================================
# Tests: create_audit_parser
# ===========================================================================


class TestCreateAuditParser:
    """Tests for create_audit_parser function."""

    def test_creates_audit_subparser(self):
        """Test that audit parser is created with subcommands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        # Parse presets command
        args = parser.parse_args(["audit", "presets"])
        assert args.audit_command == "presets"

    def test_preset_command_options(self):
        """Test preset command with options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(["audit", "preset", "Legal Due Diligence", "--format", "json"])
        assert args.name == "Legal Due Diligence"
        assert args.format == "json"

    def test_create_command_options(self):
        """Test create command with all options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(
            [
                "audit",
                "create",
                "doc1,doc2,doc3",
                "--types",
                "security,compliance",
                "--preset",
                "Legal Due Diligence",
                "--name",
                "Test Audit",
                "--model",
                "gpt-4",
            ]
        )
        assert args.documents == "doc1,doc2,doc3"
        assert args.types == "security,compliance"
        assert args.preset == "Legal Due Diligence"
        assert args.name == "Test Audit"
        assert args.model == "gpt-4"

    def test_findings_command_options(self):
        """Test findings command with filters."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(
            ["audit", "findings", "session-123", "--severity", "high", "--format", "json"]
        )
        assert args.session_id == "session-123"
        assert args.severity == "high"
        assert args.format == "json"

    def test_report_command_options(self):
        """Test report command with all options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(
            [
                "audit",
                "report",
                "session-123",
                "--format",
                "pdf",
                "--template",
                "executive_summary",
                "--output",
                "report.pdf",
                "--min-severity",
                "medium",
                "--include-resolved",
                "--author",
                "Test Author",
                "--company",
                "Test Co",
            ]
        )
        assert args.session_id == "session-123"
        assert args.format == "pdf"
        assert args.template == "executive_summary"
        assert args.output == "report.pdf"
        assert args.min_severity == "medium"
        assert args.include_resolved is True
        assert args.author == "Test Author"
        assert args.company == "Test Co"


# ===========================================================================
# Tests: audit_cli
# ===========================================================================


class TestAuditCli:
    """Tests for audit_cli function."""

    @pytest.fixture
    def base_args(self):
        """Create base args namespace."""
        return argparse.Namespace()

    def test_unknown_command_returns_error(self, base_args, capsys):
        """Test unknown command returns error."""
        base_args.audit_command = "unknown"

        result = audit_cli(base_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown audit command" in captured.out


# ===========================================================================
# Tests: list_presets
# ===========================================================================


class TestListPresets:
    """Tests for list_presets function."""

    @pytest.fixture
    def presets_args(self):
        """Create presets args."""
        args = argparse.Namespace()
        args.format = "text"
        return args

    @pytest.mark.asyncio
    async def test_list_presets_text_output(self, presets_args, capsys):
        """Test listing presets with text output."""
        mock_registry = MagicMock()
        mock_registry.list_presets.return_value = [MockPreset()]

        with patch("aragora.audit.registry.audit_registry", mock_registry):
            result = await list_presets(presets_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Legal Due Diligence" in captured.out
        assert "compliance" in captured.out

    @pytest.mark.asyncio
    async def test_list_presets_json_output(self, presets_args, capsys):
        """Test listing presets with JSON output."""
        presets_args.format = "json"

        mock_registry = MagicMock()
        mock_registry.list_presets.return_value = [MockPreset()]

        with patch("aragora.audit.registry.audit_registry", mock_registry):
            result = await list_presets(presets_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output) == 1
        assert output[0]["name"] == "Legal Due Diligence"

    @pytest.mark.asyncio
    async def test_list_presets_empty(self, presets_args, capsys):
        """Test listing when no presets available."""
        mock_registry = MagicMock()
        mock_registry.list_presets.return_value = []

        with patch("aragora.audit.registry.audit_registry", mock_registry):
            result = await list_presets(presets_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No presets available" in captured.out


# ===========================================================================
# Tests: show_preset
# ===========================================================================


class TestShowPreset:
    """Tests for show_preset function."""

    @pytest.fixture
    def preset_args(self):
        """Create preset args."""
        args = argparse.Namespace()
        args.name = "Legal Due Diligence"
        args.format = "text"
        return args

    @pytest.mark.asyncio
    async def test_show_preset_success(self, preset_args, capsys):
        """Test showing preset details."""
        mock_preset = MockPreset(
            custom_rules=[{"severity": "high", "title": "Test Rule", "category": "legal"}]
        )
        mock_registry = MagicMock()
        mock_registry.get_preset.return_value = mock_preset

        with patch("aragora.audit.registry.audit_registry", mock_registry):
            result = await show_preset(preset_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Legal Due Diligence" in captured.out
        assert "Custom Rules" in captured.out

    @pytest.mark.asyncio
    async def test_show_preset_json_output(self, preset_args, capsys):
        """Test showing preset with JSON output."""
        preset_args.format = "json"

        mock_preset = MockPreset()
        mock_registry = MagicMock()
        mock_registry.get_preset.return_value = mock_preset

        with patch("aragora.audit.registry.audit_registry", mock_registry):
            result = await show_preset(preset_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["name"] == "Legal Due Diligence"

    @pytest.mark.asyncio
    async def test_show_preset_not_found(self, preset_args, capsys):
        """Test showing non-existent preset."""
        preset_args.name = "NonExistent"

        mock_registry = MagicMock()
        mock_registry.get_preset.return_value = None
        mock_registry.list_presets.return_value = [MockPreset()]

        with patch("aragora.audit.registry.audit_registry", mock_registry):
            result = await show_preset(preset_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Preset not found" in captured.out


# ===========================================================================
# Tests: list_types
# ===========================================================================


class TestListTypes:
    """Tests for list_types function."""

    @pytest.fixture
    def types_args(self):
        """Create types args."""
        args = argparse.Namespace()
        args.format = "text"
        return args

    @pytest.mark.asyncio
    async def test_list_types_text_output(self, types_args, capsys):
        """Test listing audit types with text output."""
        mock_registry = MagicMock()
        mock_registry.list_audit_types.return_value = [MockAuditType()]

        with patch.dict("sys.modules", {"aragora.audit.registry": MagicMock()}):
            with patch("aragora.cli.audit.audit_registry", mock_registry):
                result = await list_types(types_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "compliance" in captured.out
        assert "Compliance Check" in captured.out

    @pytest.mark.asyncio
    async def test_list_types_json_output(self, types_args, capsys):
        """Test listing audit types with JSON output."""
        types_args.format = "json"

        mock_registry = MagicMock()
        mock_registry.list_audit_types.return_value = [MockAuditType()]

        with patch.dict("sys.modules", {"aragora.audit.registry": MagicMock()}):
            with patch("aragora.cli.audit.audit_registry", mock_registry):
                result = await list_types(types_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output) == 1
        assert output[0]["id"] == "compliance"


# ===========================================================================
# Tests: create_audit
# ===========================================================================


class TestCreateAudit:
    """Tests for create_audit function."""

    @pytest.fixture
    def create_args(self):
        """Create args for create command."""
        args = argparse.Namespace()
        args.documents = "doc1,doc2"
        args.types = "security,compliance"
        args.preset = None
        args.name = None
        args.model = "gemini-1.5-flash"
        return args

    @pytest.mark.asyncio
    async def test_create_audit_success(self, create_args, capsys):
        """Test creating audit session."""
        mock_session = MockSession()
        mock_auditor = MagicMock()
        mock_auditor.create_session = AsyncMock(return_value=mock_session)

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            with patch.dict("sys.modules", {"aragora.audit.registry": MagicMock()}):
                result = await create_audit(create_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Session created" in captured.out
        assert "session-123" in captured.out

    @pytest.mark.asyncio
    async def test_create_audit_with_preset(self, create_args, capsys):
        """Test creating audit with preset."""
        create_args.preset = "Legal Due Diligence"
        create_args.types = None

        mock_session = MockSession()
        mock_auditor = MagicMock()
        mock_auditor.create_session = AsyncMock(return_value=mock_session)

        mock_registry = MagicMock()
        mock_registry.get_preset.return_value = MockPreset()

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            with patch("aragora.cli.audit.audit_registry", mock_registry):
                result = await create_audit(create_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Using preset" in captured.out

    @pytest.mark.asyncio
    async def test_create_audit_preset_not_found(self, create_args, capsys):
        """Test creating audit with non-existent preset."""
        create_args.preset = "NonExistent"
        create_args.types = None

        mock_registry = MagicMock()
        mock_registry.get_preset.return_value = None

        with patch.dict("sys.modules", {"aragora.audit.registry": MagicMock()}):
            with patch("aragora.cli.audit.audit_registry", mock_registry):
                result = await create_audit(create_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Preset not found" in captured.out


# ===========================================================================
# Tests: start_audit
# ===========================================================================


class TestStartAudit:
    """Tests for start_audit function."""

    @pytest.fixture
    def start_args(self):
        """Create args for start command."""
        args = argparse.Namespace()
        args.session_id = "session-123"
        return args

    @pytest.mark.asyncio
    async def test_start_audit_success(self, start_args, capsys):
        """Test starting audit session."""
        mock_result = MockAuditResult(findings=[MockFinding()])
        mock_auditor = MagicMock()
        mock_auditor.run_audit = AsyncMock(return_value=mock_result)

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await start_audit(start_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Completed" in captured.out
        assert "1 findings" in captured.out


# ===========================================================================
# Tests: audit_status
# ===========================================================================


class TestAuditStatus:
    """Tests for audit_status function."""

    @pytest.fixture
    def status_args(self):
        """Create args for status command."""
        args = argparse.Namespace()
        args.session_id = "session-123"
        return args

    @pytest.mark.asyncio
    async def test_audit_status_success(self, status_args, capsys):
        """Test getting audit status."""
        mock_session = MockSession()
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = mock_session

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await audit_status(status_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "session-123" in captured.out
        assert "100%" in captured.out

    @pytest.mark.asyncio
    async def test_audit_status_not_found(self, status_args, capsys):
        """Test getting status for non-existent session."""
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await audit_status(status_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ===========================================================================
# Tests: audit_findings
# ===========================================================================


class TestAuditFindings:
    """Tests for audit_findings function."""

    @pytest.fixture
    def findings_args(self):
        """Create args for findings command."""
        args = argparse.Namespace()
        args.session_id = "session-123"
        args.severity = None
        args.format = "text"
        return args

    @pytest.mark.asyncio
    async def test_audit_findings_text(self, findings_args, capsys):
        """Test getting findings with text output."""
        mock_auditor = MagicMock()
        mock_auditor.get_findings.return_value = [MockFinding()]

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor
        mock_audit_module.FindingSeverity = MagicMock()

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await audit_findings(findings_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Missing compliance clause" in captured.out

    @pytest.mark.asyncio
    async def test_audit_findings_json(self, findings_args, capsys):
        """Test getting findings with JSON output."""
        findings_args.format = "json"

        mock_auditor = MagicMock()
        mock_auditor.get_findings.return_value = [MockFinding()]

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor
        mock_audit_module.FindingSeverity = MagicMock()

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await audit_findings(findings_args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output) == 1
        assert output[0]["title"] == "Missing compliance clause"


# ===========================================================================
# Tests: export_audit
# ===========================================================================


class TestExportAudit:
    """Tests for export_audit function."""

    @pytest.fixture
    def export_args(self, tmp_path):
        """Create args for export command."""
        args = argparse.Namespace()
        args.session_id = "session-123"
        args.output = str(tmp_path / "export.json")
        return args

    @pytest.mark.asyncio
    async def test_export_audit_success(self, export_args, capsys):
        """Test exporting audit data."""
        mock_session = MockSession(findings=[MockFinding()])
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = mock_session

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await export_audit(export_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Exported to" in captured.out

        # Verify file was created
        output_path = Path(export_args.output)
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_audit_not_found(self, export_args, capsys):
        """Test exporting non-existent session."""
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            result = await export_audit(export_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ===========================================================================
# Tests: generate_report
# ===========================================================================


class TestGenerateReport:
    """Tests for generate_report function."""

    @pytest.fixture
    def report_args(self, tmp_path):
        """Create args for report command."""
        args = argparse.Namespace()
        args.session_id = "session-123"
        args.format = "markdown"
        args.template = "detailed_findings"
        args.output = str(tmp_path / "report.md")
        args.min_severity = "low"
        args.include_resolved = False
        args.author = "Test Author"
        args.company = "Test Co"
        return args

    @pytest.mark.asyncio
    async def test_generate_report_success(self, report_args, capsys):
        """Test generating report."""
        mock_session = MockSession()
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = mock_session

        mock_report = MockReport()
        mock_generator = MagicMock()
        mock_generator.generate = AsyncMock(return_value=mock_report)

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        mock_reports_module = MagicMock()
        mock_reports_module.AuditReportGenerator.return_value = mock_generator
        mock_reports_module.ReportFormat = MagicMock()
        mock_reports_module.ReportFormat.MARKDOWN = "markdown"
        mock_reports_module.ReportTemplate = MagicMock()
        mock_reports_module.ReportTemplate.DETAILED_FINDINGS = "detailed_findings"
        mock_reports_module.ReportConfig = MagicMock()

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            with patch.dict("sys.modules", {"aragora.reports": mock_reports_module}):
                result = await generate_report(report_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Report generated successfully" in captured.out

    @pytest.mark.asyncio
    async def test_generate_report_session_not_found(self, report_args, capsys):
        """Test generating report for non-existent session."""
        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None

        mock_audit_module = MagicMock()
        mock_audit_module.get_document_auditor.return_value = mock_auditor

        with patch.dict("sys.modules", {"aragora.audit": mock_audit_module}):
            with patch.dict("sys.modules", {"aragora.reports": MagicMock()}):
                result = await generate_report(report_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out
