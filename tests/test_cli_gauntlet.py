"""Tests for Gauntlet CLI command."""

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aragora.cli.gauntlet import cmd_gauntlet, create_gauntlet_parser, parse_agents


class TestParseAgents:
    """Tests for parse_agents function."""

    def test_parse_single_agent(self):
        """Parse single agent."""
        result = parse_agents("codex")
        assert result == [("codex", None)]

    def test_parse_multiple_agents(self):
        """Parse comma-separated agents."""
        result = parse_agents("codex,claude,openai")
        assert result == [("codex", None), ("claude", None), ("openai", None)]

    def test_parse_agent_with_role(self):
        """Parse agent with role."""
        result = parse_agents("claude:critic")
        assert result == [("claude", "critic")]

    def test_parse_mixed_agents_and_roles(self):
        """Parse mix of agents with and without roles."""
        result = parse_agents("codex,claude:critic,openai:synthesizer")
        assert result == [
            ("codex", None),
            ("claude", "critic"),
            ("openai", "synthesizer"),
        ]

    def test_parse_strips_whitespace(self):
        """Whitespace around agents is stripped."""
        result = parse_agents(" codex , claude ")
        assert result == [("codex", None), ("claude", None)]

    def test_parse_empty_string(self):
        """Empty string returns single empty tuple."""
        result = parse_agents("")
        assert result == [("", None)]

    def test_parse_agent_with_multiple_colons(self):
        """Multiple colons - first is separator, rest in role."""
        result = parse_agents("claude:critic:verbose")
        assert result == [("claude", "critic:verbose")]

    def test_parse_agent_role_with_special_chars(self):
        """Role can contain special characters."""
        result = parse_agents("claude:my-role_v2")
        assert result == [("claude", "my-role_v2")]


class TestCreateGauntletParser:
    """Tests for create_gauntlet_parser function."""

    @pytest.fixture
    def parser(self):
        """Create main parser with gauntlet subcommand."""
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers()
        create_gauntlet_parser(subparsers)
        return main_parser

    def test_parser_creates_subcommand(self, parser):
        """Parser creates 'gauntlet' subcommand."""
        args = parser.parse_args(["gauntlet", "input.md"])
        assert hasattr(args, "input")

    def test_parser_input_positional_arg(self, parser):
        """Input path is required positional argument."""
        args = parser.parse_args(["gauntlet", "path/to/spec.md"])
        assert args.input == "path/to/spec.md"

    def test_parser_input_type_choices(self, parser):
        """--input-type accepts valid choices."""
        for choice in ["spec", "architecture", "policy", "code", "strategy", "contract"]:
            args = parser.parse_args(["gauntlet", "in.md", "--input-type", choice])
            assert args.input_type == choice

    def test_parser_input_type_default(self, parser):
        """--input-type defaults to 'spec'."""
        args = parser.parse_args(["gauntlet", "in.md"])
        assert args.input_type == "spec"

    def test_parser_agents_default(self, parser):
        """--agents defaults to 'anthropic-api,openai-api'."""
        args = parser.parse_args(["gauntlet", "in.md"])
        assert args.agents == "anthropic-api,openai-api"

    def test_parser_profile_choices(self, parser):
        """--profile accepts valid choices."""
        for choice in [
            "default",
            "quick",
            "thorough",
            "code",
            "policy",
            "gdpr",
            "hipaa",
            "ai_act",
            "security",
            "sox",
        ]:
            args = parser.parse_args(["gauntlet", "in.md", "--profile", choice])
            assert args.profile == choice

    def test_parser_rounds_type(self, parser):
        """--rounds is integer type."""
        args = parser.parse_args(["gauntlet", "in.md", "--rounds", "5"])
        assert args.rounds == 5
        assert isinstance(args.rounds, int)

    def test_parser_timeout_type(self, parser):
        """--timeout is integer type."""
        args = parser.parse_args(["gauntlet", "in.md", "--timeout", "300"])
        assert args.timeout == 300
        assert isinstance(args.timeout, int)

    def test_parser_format_choices(self, parser):
        """--format accepts json, md, html."""
        for fmt in ["json", "md", "html"]:
            args = parser.parse_args(["gauntlet", "in.md", "--format", fmt])
            assert args.format == fmt

    def test_parser_boolean_flags(self, parser):
        """Boolean flags work correctly."""
        args = parser.parse_args(
            ["gauntlet", "in.md", "--verify", "--no-redteam", "--no-probing", "--no-audit"]
        )
        assert args.verify is True
        assert args.no_redteam is True
        assert args.no_probing is True
        assert args.no_audit is True

    def test_parser_sets_func_default(self, parser):
        """Parser sets func=cmd_gauntlet as default."""
        args = parser.parse_args(["gauntlet", "in.md"])
        assert args.func == cmd_gauntlet


class TestCmdGauntletInputHandling:
    """Tests for input file handling."""

    def test_missing_input_file_prints_error(self, capsys):
        """Non-existent input file prints error."""
        args = argparse.Namespace(
            input="/nonexistent/path/spec.md",
            input_type="spec",
            agents="anthropic-api",
            profile="default",
        )
        cmd_gauntlet(args)
        captured = capsys.readouterr()
        assert "Error: Input file not found" in captured.out

    def test_missing_input_file_shows_suggestions(self, capsys):
        """Error message includes troubleshooting suggestions."""
        args = argparse.Namespace(
            input="/nonexistent/spec.md",
            input_type="spec",
            agents="anthropic-api",
            profile="default",
        )
        cmd_gauntlet(args)
        captured = capsys.readouterr()
        assert "Please check" in captured.out
        assert "file path is correct" in captured.out

    def test_valid_input_file_reads_content(self, tmp_path, capsys):
        """Valid input file content is read and displayed."""
        input_file = tmp_path / "spec.md"
        input_file.write_text("# Test Specification\n\nThis is a test.")

        args = argparse.Namespace(
            input=str(input_file),
            input_type="spec",
            agents="anthropic-api",
            profile="default",
            rounds=None,
            timeout=None,
            output=None,
            format=None,
            verify=False,
            no_redteam=False,
            no_probing=False,
            no_audit=False,
        )

        with patch("aragora.agents.base.create_agent", side_effect=Exception("No agent")):
            cmd_gauntlet(args)

        captured = capsys.readouterr()
        assert "Input:" in captured.out
        assert "chars" in captured.out

    def test_input_type_displayed(self, tmp_path, capsys):
        """Input type is displayed in output."""
        input_file = tmp_path / "spec.md"
        input_file.write_text("# Test")

        args = argparse.Namespace(
            input=str(input_file),
            input_type="architecture",
            agents="anthropic-api",
            profile="default",
            rounds=None,
            timeout=None,
            output=None,
            format=None,
            verify=False,
            no_redteam=False,
            no_probing=False,
            no_audit=False,
        )

        with patch("aragora.agents.base.create_agent", side_effect=Exception("No agent")):
            cmd_gauntlet(args)

        captured = capsys.readouterr()
        assert "Type: architecture" in captured.out


class TestCmdGauntletAgentCreation:
    """Tests for agent creation and failure handling."""

    @pytest.fixture
    def mock_args(self, tmp_path):
        """Create mock args."""
        input_file = tmp_path / "spec.md"
        input_file.write_text("# Test")
        return argparse.Namespace(
            input=str(input_file),
            input_type="spec",
            agents="anthropic-api",
            profile="default",
            rounds=None,
            timeout=None,
            output=None,
            format=None,
            verify=False,
            no_redteam=False,
            no_probing=False,
            no_audit=False,
        )

    def test_all_agents_fail_prints_detailed_error(self, mock_args, capsys):
        """All agents failing prints detailed error."""
        with patch("aragora.agents.base.create_agent", side_effect=Exception("API key missing")):
            cmd_gauntlet(mock_args)

        captured = capsys.readouterr()
        assert "Error: No agents could be created" in captured.out
        assert "Failed agents:" in captured.out
        assert "To fix:" in captured.out

    def test_api_agent_failure_shows_env_var_hint(self, mock_args, capsys):
        """API agent failure shows environment variable hint."""
        mock_args.agents = "anthropic-api"
        with patch("aragora.agents.base.create_agent", side_effect=Exception("No key")):
            cmd_gauntlet(mock_args)

        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.out

    def test_openai_agent_failure_shows_env_var_hint(self, mock_args, capsys):
        """OpenAI agent failure shows environment variable hint."""
        mock_args.agents = "openai-api"
        with patch("aragora.agents.base.create_agent", side_effect=Exception("No key")):
            cmd_gauntlet(mock_args)

        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.out

    def test_gemini_agent_failure_shows_env_var_hint(self, mock_args, capsys):
        """Gemini agent failure shows environment variable hint."""
        mock_args.agents = "gemini"
        with patch("aragora.agents.base.create_agent", side_effect=Exception("No key")):
            cmd_gauntlet(mock_args)

        captured = capsys.readouterr()
        assert "GEMINI_API_KEY" in captured.out

    def test_no_agents_returns_early(self, mock_args, capsys):
        """Zero successful agents returns early."""
        with patch("aragora.agents.base.create_agent", side_effect=Exception("Failed")):
            cmd_gauntlet(mock_args)

        captured = capsys.readouterr()
        assert "No agents could be created" in captured.out
        # Should not see "Running stress-test"
        assert "Running stress-test" not in captured.out
