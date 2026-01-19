"""Tests for CLI audit command - document compliance and audit."""

import argparse
import json
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


@pytest.fixture
def mock_args():
    """Create mock args object."""
    args = MagicMock()
    args.format = "text"
    return args


@pytest.fixture
def mock_preset():
    """Create mock audit preset."""
    preset = MagicMock()
    preset.name = "Legal Due Diligence"
    preset.description = "Comprehensive legal review"
    preset.audit_types = ["security", "compliance"]
    preset.custom_rules = [{"severity": "high", "title": "PII Detection", "category": "privacy"}]
    preset.consensus_threshold = 0.8
    preset.agents = ["claude", "gpt-4"]
    preset.parameters = {"depth": "thorough"}
    return preset


@pytest.fixture
def mock_audit_type():
    """Create mock audit type."""
    audit_type = MagicMock()
    audit_type.id = "security"
    audit_type.display_name = "Security Audit"
    audit_type.description = "Check for security vulnerabilities"
    audit_type.version = "1.0"
    audit_type.capabilities = {
        "supports_chunk_analysis": True,
        "supports_cross_document": True,
        "requires_llm": True,
    }
    return audit_type


@pytest.fixture
def mock_session():
    """Create mock audit session."""
    session = MagicMock()
    session.id = "session-123"
    session.status = MagicMock(value="completed")
    session.progress = 1.0
    session.findings = []
    session.to_dict.return_value = {"id": "session-123", "status": "completed"}
    return session


@pytest.fixture
def mock_finding():
    """Create mock audit finding."""
    finding = MagicMock()
    finding.severity = MagicMock(value="high")
    finding.title = "Sensitive data exposed"
    finding.to_dict.return_value = {
        "severity": "high",
        "title": "Sensitive data exposed",
    }
    return finding


class TestCreateAuditParser:
    """Tests for create_audit_parser function."""

    def test_creates_parser(self):
        """Parser is created with correct subcommands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        create_audit_parser(subparsers)

        # Parse an audit command
        args = parser.parse_args(["audit", "presets"])
        assert args.audit_command == "presets"

    def test_presets_command(self):
        """Presets subcommand is registered."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(["audit", "presets", "--format", "json"])
        assert args.audit_command == "presets"
        assert args.format == "json"

    def test_preset_command(self):
        """Preset detail subcommand is registered."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(["audit", "preset", "Legal Due Diligence"])
        assert args.audit_command == "preset"
        assert args.name == "Legal Due Diligence"

    def test_create_command(self):
        """Create subcommand is registered."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(
            ["audit", "create", "doc1,doc2", "--types", "security,compliance", "--preset", "Legal"]
        )
        assert args.audit_command == "create"
        assert args.documents == "doc1,doc2"
        assert args.types == "security,compliance"
        assert args.preset == "Legal"

    def test_findings_command(self):
        """Findings subcommand is registered."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_audit_parser(subparsers)

        args = parser.parse_args(["audit", "findings", "session-123", "--severity", "high"])
        assert args.audit_command == "findings"
        assert args.session_id == "session-123"
        assert args.severity == "high"

    def test_report_command(self):
        """Report subcommand is registered."""
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
            ]
        )
        assert args.audit_command == "report"
        assert args.format == "pdf"
        assert args.template == "executive_summary"


class TestAuditCli:
    """Tests for audit_cli dispatch function."""

    @patch("aragora.cli.audit.list_presets", new_callable=AsyncMock)
    def test_dispatches_presets(self, mock_list_presets, mock_args):
        """Dispatches to list_presets."""
        mock_list_presets.return_value = 0
        mock_args.audit_command = "presets"

        result = audit_cli(mock_args)

        assert result == 0
        mock_list_presets.assert_called_once()

    @patch("aragora.cli.audit.show_preset", new_callable=AsyncMock)
    def test_dispatches_preset(self, mock_show_preset, mock_args):
        """Dispatches to show_preset."""
        mock_show_preset.return_value = 0
        mock_args.audit_command = "preset"

        result = audit_cli(mock_args)

        assert result == 0
        mock_show_preset.assert_called_once()

    @patch("aragora.cli.audit.create_audit", new_callable=AsyncMock)
    def test_dispatches_create(self, mock_create_audit, mock_args):
        """Dispatches to create_audit."""
        mock_create_audit.return_value = 0
        mock_args.audit_command = "create"

        result = audit_cli(mock_args)

        assert result == 0
        mock_create_audit.assert_called_once()

    def test_unknown_command_returns_1(self, mock_args, capsys):
        """Unknown command returns 1."""
        mock_args.audit_command = "unknown"

        result = audit_cli(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown audit command" in captured.out


class TestListPresets:
    """Tests for list_presets function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_lists_presets_text(self, mock_registry, mock_preset, mock_args, capsys):
        """List presets in text format."""
        mock_registry.list_presets.return_value = [mock_preset]

        result = await list_presets(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Legal Due Diligence" in captured.out
        assert "Comprehensive legal review" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_lists_presets_json(self, mock_registry, mock_preset, mock_args, capsys):
        """List presets in JSON format."""
        mock_args.format = "json"
        mock_registry.list_presets.return_value = [mock_preset]

        result = await list_presets(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "Legal Due Diligence"

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_handles_no_presets(self, mock_registry, mock_args, capsys):
        """Handle no presets gracefully."""
        mock_registry.list_presets.return_value = []

        result = await list_presets(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No presets available" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_handles_error(self, mock_registry, mock_args, capsys):
        """Handle error gracefully."""
        mock_registry.auto_discover.side_effect = Exception("Registry error")

        result = await list_presets(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestShowPreset:
    """Tests for show_preset function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_shows_preset_text(self, mock_registry, mock_preset, mock_args, capsys):
        """Show preset in text format."""
        mock_args.name = "Legal Due Diligence"
        mock_registry.get_preset.return_value = mock_preset

        result = await show_preset(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Legal Due Diligence" in captured.out
        assert "Comprehensive legal review" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_shows_preset_json(self, mock_registry, mock_preset, mock_args, capsys):
        """Show preset in JSON format."""
        mock_args.name = "Legal Due Diligence"
        mock_args.format = "json"
        mock_registry.get_preset.return_value = mock_preset

        result = await show_preset(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["name"] == "Legal Due Diligence"

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_preset_not_found(self, mock_registry, mock_args, capsys):
        """Handle preset not found."""
        mock_args.name = "Nonexistent"
        mock_registry.get_preset.return_value = None
        mock_registry.list_presets.return_value = []

        result = await show_preset(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Preset not found" in captured.out


class TestListTypes:
    """Tests for list_types function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_lists_types_text(self, mock_registry, mock_audit_type, mock_args, capsys):
        """List audit types in text format."""
        mock_registry.list_audit_types.return_value = [mock_audit_type]

        result = await list_types(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "security" in captured.out
        assert "Security Audit" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.registry.audit_registry")
    async def test_lists_types_json(self, mock_registry, mock_audit_type, mock_args, capsys):
        """List audit types in JSON format."""
        mock_args.format = "json"
        mock_registry.list_audit_types.return_value = [mock_audit_type]

        result = await list_types(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["id"] == "security"


class TestCreateAudit:
    """Tests for create_audit function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    async def test_creates_audit_session(self, mock_get_auditor, mock_session, mock_args, capsys):
        """Create an audit session."""
        mock_args.documents = "doc1,doc2"
        mock_args.types = "security,compliance"
        mock_args.preset = None
        mock_args.name = "Test Audit"
        mock_args.model = "gemini-1.5-flash"

        mock_auditor = MagicMock()
        mock_auditor.create_session = AsyncMock(return_value=mock_session)
        mock_get_auditor.return_value = mock_auditor

        result = await create_audit(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "session-123" in captured.out
        mock_auditor.create_session.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    @patch("aragora.audit.registry.audit_registry")
    async def test_creates_audit_with_preset(
        self, mock_registry, mock_get_auditor, mock_preset, mock_session, mock_args, capsys
    ):
        """Create audit session with preset."""
        mock_args.documents = "doc1"
        mock_args.types = None
        mock_args.preset = "Legal Due Diligence"
        mock_args.name = None
        mock_args.model = "gemini-1.5-flash"

        mock_registry.get_preset.return_value = mock_preset
        mock_auditor = MagicMock()
        mock_auditor.create_session = AsyncMock(return_value=mock_session)
        mock_get_auditor.return_value = mock_auditor

        result = await create_audit(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Using preset" in captured.out


class TestAuditStatus:
    """Tests for audit_status function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    async def test_shows_status(self, mock_get_auditor, mock_session, mock_args, capsys):
        """Show audit session status."""
        mock_args.session_id = "session-123"

        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = mock_session
        mock_get_auditor.return_value = mock_auditor

        result = await audit_status(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "session-123" in captured.out
        assert "100%" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    async def test_session_not_found(self, mock_get_auditor, mock_args, capsys):
        """Handle session not found."""
        mock_args.session_id = "nonexistent"

        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None
        mock_get_auditor.return_value = mock_auditor

        result = await audit_status(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Session not found" in captured.out


class TestAuditFindings:
    """Tests for audit_findings function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    @patch("aragora.audit.FindingSeverity")
    async def test_lists_findings_text(
        self, mock_severity_enum, mock_get_auditor, mock_finding, mock_args, capsys
    ):
        """List findings in text format."""
        mock_args.session_id = "session-123"
        mock_args.severity = None

        mock_auditor = MagicMock()
        mock_auditor.get_findings.return_value = [mock_finding]
        mock_get_auditor.return_value = mock_auditor

        result = await audit_findings(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        assert "[high]" in captured.out
        assert "Sensitive data exposed" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    @patch("aragora.audit.FindingSeverity")
    async def test_lists_findings_json(
        self, mock_severity_enum, mock_get_auditor, mock_finding, mock_args, capsys
    ):
        """List findings in JSON format."""
        mock_args.session_id = "session-123"
        mock_args.severity = None
        mock_args.format = "json"

        mock_auditor = MagicMock()
        mock_auditor.get_findings.return_value = [mock_finding]
        mock_get_auditor.return_value = mock_auditor

        result = await audit_findings(mock_args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["severity"] == "high"


class TestExportAudit:
    """Tests for export_audit function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    async def test_exports_audit(self, mock_get_auditor, mock_session, mock_args, tmp_path, capsys):
        """Export audit to JSON file."""
        output_file = tmp_path / "export.json"
        mock_args.session_id = "session-123"
        mock_args.output = str(output_file)

        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = mock_session
        mock_get_auditor.return_value = mock_auditor

        result = await export_audit(mock_args)

        assert result == 0
        assert output_file.exists()
        captured = capsys.readouterr()
        assert "Exported to" in captured.out

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    async def test_export_session_not_found(self, mock_get_auditor, mock_args, capsys):
        """Handle session not found during export."""
        mock_args.session_id = "nonexistent"
        mock_args.output = "/tmp/export.json"

        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None
        mock_get_auditor.return_value = mock_auditor

        result = await export_audit(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Session not found" in captured.out


class TestGenerateReport:
    """Tests for generate_report function."""

    @pytest.mark.asyncio
    @patch("aragora.audit.get_document_auditor")
    async def test_session_not_found(self, mock_get_auditor, mock_args, capsys):
        """Handle session not found during report generation."""
        mock_args.session_id = "nonexistent"
        mock_args.format = "markdown"
        mock_args.template = "detailed_findings"
        mock_args.output = None
        mock_args.min_severity = "low"
        mock_args.include_resolved = False
        mock_args.author = None
        mock_args.company = None

        mock_auditor = MagicMock()
        mock_auditor.get_session.return_value = None
        mock_get_auditor.return_value = mock_auditor

        result = await generate_report(mock_args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Session not found" in captured.out
