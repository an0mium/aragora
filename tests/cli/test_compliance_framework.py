"""Tests for compliance framework CLI commands.

Tests the compliance framework CLI integration:
- aragora compliance status
- aragora compliance report
- aragora compliance check
"""

from __future__ import annotations

import argparse
import json
from io import StringIO
from unittest.mock import patch

import pytest

from aragora.cli.commands.compliance import (
    _cmd_check,
    _cmd_report,
    _cmd_status,
    _format_check_result_text,
    add_compliance_parser,
    cmd_compliance,
)


# ---------------------------------------------------------------------------
# Parser registration tests
# ---------------------------------------------------------------------------


class TestComplianceParserRegistration:
    """Verify that compliance subcommands are registered in the CLI parser."""

    def test_compliance_parser_registered(self):
        """The compliance parser registers with subparsers."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_compliance_parser(subparsers)

        # Should not raise
        args = parser.parse_args(["compliance", "status"])
        assert args.compliance_command == "status"

    def test_status_subcommand_parses(self):
        """'compliance status' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_compliance_parser(subparsers)

        args = parser.parse_args(["compliance", "status", "--json"])
        assert args.compliance_command == "status"
        assert args.json is True

    def test_status_with_vertical_filter(self):
        """'compliance status --vertical healthcare' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_compliance_parser(subparsers)

        args = parser.parse_args(["compliance", "status", "--vertical", "healthcare"])
        assert args.vertical == "healthcare"

    def test_report_subcommand_parses(self):
        """'compliance report' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_compliance_parser(subparsers)

        args = parser.parse_args(["compliance", "report", "myfile.py"])
        assert args.compliance_command == "report"
        assert args.content_file == "myfile.py"

    def test_check_subcommand_parses(self):
        """'compliance check' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_compliance_parser(subparsers)

        args = parser.parse_args(["compliance", "check", "plaintext password storage", "--json"])
        assert args.compliance_command == "check"
        assert any("plaintext" in c for c in args.content)
        assert args.json is True

    def test_check_with_frameworks_and_severity(self):
        """'compliance check --frameworks hipaa --min-severity high' parses."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_compliance_parser(subparsers)

        args = parser.parse_args(
            [
                "compliance",
                "check",
                "test content",
                "--frameworks",
                "hipaa,gdpr",
                "--min-severity",
                "high",
            ]
        )
        assert args.frameworks == "hipaa,gdpr"
        assert args.min_severity == "high"


# ---------------------------------------------------------------------------
# Status command tests
# ---------------------------------------------------------------------------


class TestComplianceStatus:
    """Tests for 'aragora compliance status'."""

    def test_status_lists_frameworks(self, capsys):
        """Status command lists all available frameworks."""
        args = argparse.Namespace(
            compliance_command="status",
            json=False,
            vertical=None,
        )
        _cmd_status(args)
        output = capsys.readouterr().out

        assert "Compliance Frameworks:" in output
        assert "HIPAA" in output
        assert "GDPR" in output
        assert "OWASP" in output
        assert "SOX" in output

    def test_status_json_output(self, capsys):
        """Status command outputs valid JSON when --json is set."""
        args = argparse.Namespace(
            compliance_command="status",
            json=True,
            vertical=None,
        )
        _cmd_status(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "frameworks" in data
        assert "total" in data
        assert data["total"] > 0
        # Check structure
        fw = data["frameworks"][0]
        assert "id" in fw
        assert "name" in fw
        assert "rule_count" in fw

    def test_status_filters_by_vertical(self, capsys):
        """Status command filters by vertical."""
        args = argparse.Namespace(
            compliance_command="status",
            json=True,
            vertical="healthcare",
        )
        _cmd_status(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert data["vertical_filter"] == "healthcare"
        # All returned frameworks should be applicable to healthcare
        for fw in data["frameworks"]:
            assert "healthcare" in fw["applicable_verticals"]

    def test_status_no_match_vertical(self, capsys):
        """Status with non-matching vertical shows no frameworks."""
        args = argparse.Namespace(
            compliance_command="status",
            json=False,
            vertical="nonexistent_vertical",
        )
        _cmd_status(args)
        output = capsys.readouterr().out
        assert "No frameworks found" in output


# ---------------------------------------------------------------------------
# Check command tests
# ---------------------------------------------------------------------------


class TestComplianceCheck:
    """Tests for 'aragora compliance check'."""

    def test_check_finds_issues_in_content(self, capsys):
        """Check command detects compliance issues in content."""
        args = argparse.Namespace(
            compliance_command="check",
            content=["plaintext password in patient database with no authentication"],
            content_file=None,
            frameworks=None,
            min_severity="low",
            json=False,
        )
        _cmd_check(args)
        output = capsys.readouterr().out

        assert "COMPLIANCE CHECK RESULT" in output
        assert "Issues found:" in output

    def test_check_json_output(self, capsys):
        """Check command outputs valid JSON when --json is set."""
        args = argparse.Namespace(
            compliance_command="check",
            content=["plaintext password in patient database"],
            content_file=None,
            frameworks=None,
            min_severity="low",
            json=True,
        )
        _cmd_check(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "compliant" in data
        assert "score" in data
        assert "issues" in data
        assert "frameworks_checked" in data

    def test_check_specific_framework(self, capsys):
        """Check command respects --frameworks filter."""
        args = argparse.Namespace(
            compliance_command="check",
            content=["patient name and medical record exposed"],
            content_file=None,
            frameworks="hipaa",
            min_severity="low",
            json=True,
        )
        _cmd_check(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "hipaa" in data["frameworks_checked"]
        # Should only check HIPAA
        assert len(data["frameworks_checked"]) == 1

    def test_check_min_severity_filter(self, capsys):
        """Check command filters by minimum severity."""
        args = argparse.Namespace(
            compliance_command="check",
            content=["shared credentials for patient database with no audit trail"],
            content_file=None,
            frameworks="hipaa",
            min_severity="critical",
            json=True,
        )
        _cmd_check(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        # Only critical issues should be reported
        for issue in data.get("issues", []):
            assert issue["severity"] == "critical"

    def test_check_clean_content(self, capsys):
        """Check command shows compliant for clean content."""
        args = argparse.Namespace(
            compliance_command="check",
            content=["The weather is nice today"],
            content_file=None,
            frameworks=None,
            min_severity="low",
            json=True,
        )
        _cmd_check(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert data["compliant"] is True
        assert data["score"] == 1.0

    def test_check_empty_content_exits(self, capsys):
        """Check command exits with error when no content is provided."""
        args = argparse.Namespace(
            compliance_command="check",
            content=[],
            content_file=None,
            frameworks=None,
            min_severity="low",
            json=False,
        )
        with patch("sys.stdin", StringIO("")):
            with pytest.raises(SystemExit):
                _cmd_check(args)

    def test_check_from_file(self, capsys, tmp_path):
        """Check command reads content from file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("plaintext password = 'secret123'")

        args = argparse.Namespace(
            compliance_command="check",
            content=[],
            content_file=str(test_file),
            frameworks="owasp",
            min_severity="low",
            json=True,
        )
        _cmd_check(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "owasp" in data["frameworks_checked"]


# ---------------------------------------------------------------------------
# Report command tests
# ---------------------------------------------------------------------------


class TestComplianceReport:
    """Tests for 'aragora compliance report'."""

    def test_report_from_file(self, capsys, tmp_path):
        """Report command reads content and generates text report."""
        test_file = tmp_path / "plan.txt"
        test_file.write_text("This system stores patient names in plaintext without encryption.")

        args = argparse.Namespace(
            compliance_command="report",
            content_file=str(test_file),
            frameworks=None,
            output_format="text",
            output=None,
        )
        _cmd_report(args)
        output = capsys.readouterr().out

        assert "COMPLIANCE CHECK RESULT" in output

    def test_report_json_output(self, capsys, tmp_path):
        """Report command outputs valid JSON."""
        test_file = tmp_path / "plan.txt"
        test_file.write_text("The weather is nice today.")

        args = argparse.Namespace(
            compliance_command="report",
            content_file=str(test_file),
            frameworks=None,
            output_format="json",
            output=None,
        )
        _cmd_report(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "compliant" in data
        assert "score" in data

    def test_report_write_to_file(self, tmp_path):
        """Report command writes output to file."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("Clean content with no issues")
        output_file = tmp_path / "report.json"

        args = argparse.Namespace(
            compliance_command="report",
            content_file=str(test_file),
            frameworks=None,
            output_format="json",
            output=str(output_file),
        )
        _cmd_report(args)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "compliant" in data


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


class TestComplianceDispatcher:
    """Tests for the cmd_compliance dispatcher."""

    def test_dispatches_to_status(self, capsys):
        """Dispatcher routes 'status' to _cmd_status."""
        args = argparse.Namespace(
            compliance_command="status",
            json=False,
            vertical=None,
        )
        cmd_compliance(args)
        output = capsys.readouterr().out
        assert "Compliance Frameworks:" in output

    def test_dispatches_to_check(self, capsys):
        """Dispatcher routes 'check' to _cmd_check."""
        args = argparse.Namespace(
            compliance_command="check",
            content=["safe content"],
            content_file=None,
            frameworks=None,
            min_severity="low",
            json=True,
        )
        cmd_compliance(args)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "compliant" in data

    def test_no_subcommand_shows_help(self, capsys):
        """Dispatcher shows help when no subcommand is given."""
        args = argparse.Namespace(compliance_command=None)
        with pytest.raises(SystemExit):
            cmd_compliance(args)
        output = capsys.readouterr().out
        assert "status" in output
        assert "report" in output
        assert "check" in output


# ---------------------------------------------------------------------------
# Format helper tests
# ---------------------------------------------------------------------------


class TestFormatCheckResult:
    """Tests for the text formatting helper."""

    def test_format_compliant_result(self):
        """Formatting a compliant result shows COMPLIANT status."""
        from aragora.compliance.framework import ComplianceCheckResult

        result = ComplianceCheckResult(
            compliant=True,
            issues=[],
            frameworks_checked=["hipaa"],
            score=1.0,
        )
        text = _format_check_result_text(result)
        assert "COMPLIANT" in text
        assert "100%" in text
        assert "No issues found" in text

    def test_format_noncompliant_result(self):
        """Formatting a non-compliant result shows issues."""
        from aragora.compliance.framework import (
            ComplianceCheckResult,
            ComplianceIssue,
            ComplianceSeverity,
        )

        result = ComplianceCheckResult(
            compliant=False,
            issues=[
                ComplianceIssue(
                    framework="hipaa",
                    rule_id="hipaa-phi-exposure",
                    severity=ComplianceSeverity.CRITICAL,
                    description="PHI exposure detected",
                    recommendation="Encrypt PHI",
                    matched_text="patient name",
                    line_number=5,
                ),
            ],
            frameworks_checked=["hipaa"],
            score=0.7,
        )
        text = _format_check_result_text(result)
        assert "NON-COMPLIANT" in text
        assert "[CRITICAL]" in text
        assert "hipaa/hipaa-phi-exposure" in text
        assert "PHI exposure detected" in text
        assert "Line: 5" in text
        assert "Encrypt PHI" in text
