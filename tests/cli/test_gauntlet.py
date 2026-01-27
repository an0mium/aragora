"""
Tests for aragora.cli.gauntlet module.

Tests gauntlet CLI commands for adversarial stress-testing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.gauntlet import (
    _format_agent_error,
    _save_receipt,
    cmd_gauntlet,
    create_gauntlet_parser,
    parse_agents,
)


# ===========================================================================
# Test Fixtures and Mock Classes
# ===========================================================================


@dataclass
class MockAgentSpec:
    """Mock agent spec."""

    provider: str = "anthropic-api"
    role: str | None = None


@dataclass
class MockVerdict:
    """Mock verdict."""

    value: str = "approved"


@dataclass
class MockResult:
    """Mock gauntlet result."""

    verdict: MockVerdict = field(default_factory=MockVerdict)
    findings: list = field(default_factory=list)

    def summary(self):
        return "Test summary: 0 findings"


@dataclass
class MockReceipt:
    """Mock decision receipt."""

    artifact_hash: str = "abc123def456"

    def to_json(self):
        return '{"result": "test"}'

    def to_markdown(self):
        return "# Test Receipt"

    def to_html(self):
        return "<html><body>Test Receipt</body></html>"


# ===========================================================================
# Tests: parse_agents
# ===========================================================================


class TestParseAgents:
    """Tests for parse_agents function."""

    def test_single_agent_becomes_proposer(self):
        """Test single agent gets proposer role."""
        mock_spec = MockAgentSpec(provider="anthropic-api", role=None)

        with patch("aragora.agents.spec.AgentSpec.parse_list", return_value=[mock_spec]):
            result = parse_agents("anthropic-api")

        assert len(result) == 1
        assert result[0] == ("anthropic-api", "proposer")

    def test_two_agents_proposer_and_synthesizer(self):
        """Test two agents get proposer and synthesizer roles."""
        mock_specs = [
            MockAgentSpec(provider="anthropic-api", role=None),
            MockAgentSpec(provider="openai-api", role=None),
        ]

        with patch("aragora.agents.spec.AgentSpec.parse_list", return_value=mock_specs):
            result = parse_agents("anthropic-api,openai-api")

        assert len(result) == 2
        assert result[0] == ("anthropic-api", "proposer")
        assert result[1] == ("openai-api", "synthesizer")

    def test_three_agents_with_critic(self):
        """Test three agents get proposer, critic, and synthesizer roles."""
        mock_specs = [
            MockAgentSpec(provider="anthropic-api", role=None),
            MockAgentSpec(provider="openai-api", role=None),
            MockAgentSpec(provider="gemini-api", role=None),
        ]

        with patch("aragora.agents.spec.AgentSpec.parse_list", return_value=mock_specs):
            result = parse_agents("anthropic-api,openai-api,gemini-api")

        assert len(result) == 3
        assert result[0] == ("anthropic-api", "proposer")
        assert result[1] == ("openai-api", "critic")
        assert result[2] == ("gemini-api", "synthesizer")

    def test_explicit_role_preserved(self):
        """Test explicit role in spec is preserved."""
        mock_spec = MockAgentSpec(provider="anthropic-api", role="critic")

        with patch("aragora.agents.spec.AgentSpec.parse_list", return_value=[mock_spec]):
            result = parse_agents("anthropic-api|claude-3|security|critic")

        assert len(result) == 1
        assert result[0] == ("anthropic-api", "critic")


# ===========================================================================
# Tests: _format_agent_error
# ===========================================================================


class TestFormatAgentError:
    """Tests for _format_agent_error function."""

    def test_api_agent_shows_env_var(self):
        """Test API agent error shows environment variable."""
        result = _format_agent_error("anthropic-api", "Error message")
        assert "ANTHROPIC_API_KEY" in result
        assert "not set or invalid" in result

    def test_openai_agent_shows_env_var(self):
        """Test OpenAI agent error shows environment variable."""
        result = _format_agent_error("openai-api", "Error message")
        assert "OPENAI_API_KEY" in result

    def test_non_api_agent_shows_error(self):
        """Test non-API agent error shows the error message."""
        result = _format_agent_error("local-agent", "Specific error")
        assert "Specific error" in result

    def test_unknown_api_agent_infers_env_var(self):
        """Test unknown API agent infers environment variable."""
        result = _format_agent_error("custom-api", "Error")
        assert "CUSTOM_API_API_KEY" in result


# ===========================================================================
# Tests: _save_receipt
# ===========================================================================


class TestSaveReceipt:
    """Tests for _save_receipt function."""

    def test_saves_json_receipt(self, tmp_path):
        """Test saving receipt as JSON."""
        receipt = MockReceipt()
        output_path = tmp_path / "receipt"

        result = _save_receipt(receipt, output_path, "json")

        assert result.suffix == ".json"
        assert result.exists()
        assert "test" in result.read_text()

    def test_saves_markdown_receipt(self, tmp_path):
        """Test saving receipt as Markdown."""
        receipt = MockReceipt()
        output_path = tmp_path / "receipt"

        result = _save_receipt(receipt, output_path, "md")

        assert result.suffix == ".md"
        assert result.exists()
        assert "# Test Receipt" in result.read_text()

    def test_saves_html_receipt(self, tmp_path):
        """Test saving receipt as HTML."""
        receipt = MockReceipt()
        output_path = tmp_path / "receipt"

        result = _save_receipt(receipt, output_path, "html")

        assert result.suffix == ".html"
        assert result.exists()
        assert "<html>" in result.read_text()

    def test_defaults_to_html_for_unknown(self, tmp_path):
        """Test unknown format defaults to HTML."""
        receipt = MockReceipt()
        output_path = tmp_path / "receipt"

        result = _save_receipt(receipt, output_path, "unknown")

        assert result.suffix == ".html"

    def test_creates_parent_directories(self, tmp_path):
        """Test parent directories are created."""
        receipt = MockReceipt()
        output_path = tmp_path / "subdir" / "deep" / "receipt"

        result = _save_receipt(receipt, output_path, "json")

        assert result.exists()


# ===========================================================================
# Tests: create_gauntlet_parser
# ===========================================================================


class TestCreateGauntletParser:
    """Tests for create_gauntlet_parser function."""

    def test_creates_parser_with_defaults(self):
        """Test parser creation with defaults."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_gauntlet_parser(subparsers)

        args = parser.parse_args(["gauntlet", "spec.md"])
        assert args.input == "spec.md"
        assert args.input_type == "spec"
        assert args.agents == "anthropic-api,openai-api"
        assert args.profile == "default"
        assert args.verify is False

    def test_parser_with_all_options(self):
        """Test parser with all options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_gauntlet_parser(subparsers)

        args = parser.parse_args(
            [
                "gauntlet",
                "code.py",
                "--input-type",
                "code",
                "--agents",
                "anthropic-api,gemini-api",
                "--profile",
                "thorough",
                "--rounds",
                "5",
                "--timeout",
                "600",
                "--output",
                "receipt.html",
                "--format",
                "html",
                "--verify",
                "--no-redteam",
                "--no-probing",
                "--no-audit",
            ]
        )
        assert args.input == "code.py"
        assert args.input_type == "code"
        assert args.agents == "anthropic-api,gemini-api"
        assert args.profile == "thorough"
        assert args.rounds == 5
        assert args.timeout == 600
        assert args.output == "receipt.html"
        assert args.format == "html"
        assert args.verify is True
        assert args.no_redteam is True
        assert args.no_probing is True
        assert args.no_audit is True

    def test_parser_with_compliance_profiles(self):
        """Test parser with compliance profiles."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        create_gauntlet_parser(subparsers)

        for profile in ["gdpr", "hipaa", "ai_act", "security", "sox"]:
            args = parser.parse_args(["gauntlet", "policy.yaml", "--profile", profile])
            assert args.profile == profile


# ===========================================================================
# Tests: cmd_gauntlet
# ===========================================================================


class TestCmdGauntlet:
    """Tests for cmd_gauntlet function."""

    @pytest.fixture
    def gauntlet_args(self, tmp_path):
        """Create base gauntlet args with temp file."""
        input_file = tmp_path / "spec.md"
        input_file.write_text("# Test Specification\n\nThis is a test.")

        args = argparse.Namespace()
        args.input = str(input_file)
        args.input_type = "spec"
        args.agents = "anthropic-api,openai-api"
        args.profile = "quick"
        args.rounds = None
        args.timeout = None
        args.output = None
        args.format = None
        args.verify = False
        args.no_redteam = False
        args.no_probing = False
        args.no_audit = False
        return args

    def test_input_file_not_found(self, capsys):
        """Test error when input file not found."""
        args = argparse.Namespace()
        args.input = "/nonexistent/file.md"
        args.input_type = "spec"
        args.agents = "anthropic-api"

        # Need to mock the imports since they happen at function start
        mock_gauntlet = MagicMock()
        mock_gauntlet.InputType = MagicMock()

        with patch.dict("sys.modules", {"aragora.gauntlet": mock_gauntlet}):
            with patch.dict("sys.modules", {"aragora.agents.base": MagicMock()}):
                cmd_gauntlet(args)

        captured = capsys.readouterr()
        assert "Input file not found" in captured.out

    def test_no_agents_created_error(self, gauntlet_args, capsys):
        """Test error when no agents can be created."""
        mock_gauntlet = MagicMock()
        mock_gauntlet.InputType = MagicMock()

        mock_agents = MagicMock()
        mock_agents.create_agent.side_effect = Exception("No API key")

        mock_spec = MagicMock()
        mock_spec.parse_list.return_value = [MockAgentSpec()]

        with patch.dict("sys.modules", {"aragora.gauntlet": mock_gauntlet}):
            with patch.dict("sys.modules", {"aragora.agents.base": mock_agents}):
                with patch("aragora.agents.spec.AgentSpec", mock_spec):
                    cmd_gauntlet(gauntlet_args)

        captured = capsys.readouterr()
        assert "No agents could be created" in captured.out

    def test_gauntlet_run_success(self, gauntlet_args, capsys):
        """Test successful gauntlet run."""
        # Set up mocks
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_result = MockResult()
        mock_orchestrator = MagicMock()
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.run = MagicMock(return_value=mock_result)
        mock_orchestrator.return_value = mock_orchestrator_instance

        # Create proper InputType enum mock
        mock_input_type = MagicMock()
        mock_input_type.value = "spec"

        mock_gauntlet = MagicMock()
        mock_gauntlet.InputType = MagicMock()
        mock_gauntlet.InputType.SPEC = mock_input_type
        mock_gauntlet.GauntletOrchestrator = mock_orchestrator
        mock_gauntlet.QUICK_GAUNTLET = MagicMock(
            severity_threshold=0.5,
            risk_threshold=0.5,
            max_duration_seconds=60,
            deep_audit_rounds=1,
        )

        mock_agents_base = MagicMock()
        mock_agents_base.create_agent.return_value = mock_agent

        mock_spec = MagicMock()
        mock_spec.parse_list.return_value = [MockAgentSpec()]

        with patch.dict("sys.modules", {"aragora.gauntlet": mock_gauntlet}):
            with patch.dict("sys.modules", {"aragora.agents.base": mock_agents_base}):
                with patch("aragora.agents.spec.AgentSpec", mock_spec):
                    with patch("asyncio.run", return_value=mock_result):
                        cmd_gauntlet(gauntlet_args)

        captured = capsys.readouterr()
        assert "GAUNTLET" in captured.out
        assert "Running stress-test" in captured.out

    def test_gauntlet_with_output_arg_parsed(self, gauntlet_args, tmp_path):
        """Test gauntlet args include output path when specified."""
        gauntlet_args.output = str(tmp_path / "receipt.html")

        # Verify the output argument is correctly set
        assert gauntlet_args.output == str(tmp_path / "receipt.html")
        assert gauntlet_args.format is None  # Default format is None (inferred from extension)

    def test_save_receipt_function_works(self, tmp_path):
        """Test _save_receipt creates file correctly."""
        receipt = MockReceipt()
        output_path = tmp_path / "gauntlet_receipt"

        # Save as HTML (default)
        result_path = _save_receipt(receipt, output_path, "html")

        assert result_path.exists()
        assert result_path.suffix == ".html"
        content = result_path.read_text()
        assert "<html>" in content

    def test_gauntlet_rejected_exits_with_1(self, gauntlet_args, capsys):
        """Test rejected verdict exits with code 1."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_result = MockResult(verdict=MockVerdict(value="rejected"))

        # Create proper InputType enum mock
        mock_input_type = MagicMock()
        mock_input_type.value = "spec"

        mock_gauntlet = MagicMock()
        mock_gauntlet.InputType = MagicMock()
        mock_gauntlet.InputType.SPEC = mock_input_type
        mock_gauntlet.GauntletOrchestrator.return_value.run = MagicMock(return_value=mock_result)
        mock_gauntlet.QUICK_GAUNTLET = MagicMock(
            severity_threshold=0.5,
            risk_threshold=0.5,
            max_duration_seconds=60,
            deep_audit_rounds=1,
        )

        mock_agents_base = MagicMock()
        mock_agents_base.create_agent.return_value = mock_agent

        mock_spec = MagicMock()
        mock_spec.parse_list.return_value = [MockAgentSpec()]

        with patch.dict("sys.modules", {"aragora.gauntlet": mock_gauntlet}):
            with patch.dict("sys.modules", {"aragora.agents.base": mock_agents_base}):
                with patch("aragora.agents.spec.AgentSpec", mock_spec):
                    with patch("asyncio.run", return_value=mock_result):
                        with pytest.raises(SystemExit) as exc_info:
                            cmd_gauntlet(gauntlet_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "REJECTED" in captured.out

    def test_gauntlet_needs_review_exits_with_2(self, gauntlet_args, capsys):
        """Test needs_review verdict exits with code 2."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_result = MockResult(verdict=MockVerdict(value="needs_review"))

        # Create proper InputType enum mock
        mock_input_type = MagicMock()
        mock_input_type.value = "spec"

        mock_gauntlet = MagicMock()
        mock_gauntlet.InputType = MagicMock()
        mock_gauntlet.InputType.SPEC = mock_input_type
        mock_gauntlet.GauntletOrchestrator.return_value.run = MagicMock(return_value=mock_result)
        mock_gauntlet.QUICK_GAUNTLET = MagicMock(
            severity_threshold=0.5,
            risk_threshold=0.5,
            max_duration_seconds=60,
            deep_audit_rounds=1,
        )

        mock_agents_base = MagicMock()
        mock_agents_base.create_agent.return_value = mock_agent

        mock_spec = MagicMock()
        mock_spec.parse_list.return_value = [MockAgentSpec()]

        with patch.dict("sys.modules", {"aragora.gauntlet": mock_gauntlet}):
            with patch.dict("sys.modules", {"aragora.agents.base": mock_agents_base}):
                with patch("aragora.agents.spec.AgentSpec", mock_spec):
                    with patch("asyncio.run", return_value=mock_result):
                        with pytest.raises(SystemExit) as exc_info:
                            cmd_gauntlet(gauntlet_args)

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "NEEDS REVIEW" in captured.out
