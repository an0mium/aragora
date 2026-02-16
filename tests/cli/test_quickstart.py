"""Tests for the CLI quickstart command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.commands.quickstart import (
    _detect_agents,
    _get_question,
    _load_dotenv,
    _save_receipt,
    add_quickstart_parser,
    cmd_quickstart,
)
from aragora.cli.receipt_formatter import receipt_to_html, receipt_to_markdown


# =============================================================================
# Parser registration
# =============================================================================


class TestQuickstartParser:
    def test_parser_registered(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        args = parser.parse_args(["quickstart", "--demo"])
        assert hasattr(args, "func")
        assert args.demo is True

    def test_parser_question_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        args = parser.parse_args(["quickstart", "-q", "Test question"])
        assert args.question == "Test question"

    def test_parser_output_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        args = parser.parse_args(["quickstart", "-o", "out.json"])
        assert args.output == "out.json"

    def test_parser_rounds_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        args = parser.parse_args(["quickstart", "--rounds", "5"])
        assert args.rounds == 5

    def test_parser_format_choices(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        for fmt in ["json", "md", "html"]:
            args = parser.parse_args(["quickstart", "--format", fmt])
            assert args.format == fmt


# =============================================================================
# Agent detection
# =============================================================================


class TestDetectAgents:
    def test_no_keys(self, monkeypatch):
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "MISTRAL_API_KEY",
            "XAI_API_KEY",
            "GROK_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)
        assert _detect_agents() == []

    def test_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.delenv("GROK_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        agents = _detect_agents()
        assert len(agents) == 1
        assert agents[0][0] == "anthropic-api"

    def test_multiple_keys(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test2")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.delenv("GROK_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        agents = _detect_agents()
        assert len(agents) == 2
        providers = [a[0] for a in agents]
        assert "anthropic-api" in providers
        assert "openai-api" in providers


# =============================================================================
# .env loading
# =============================================================================


class TestLoadDotenv:
    def test_load_from_cwd(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_QUICKSTART_VAR=hello\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_QUICKSTART_VAR", raising=False)
        result = _load_dotenv()
        assert result is True
        assert os.environ.get("TEST_QUICKSTART_VAR") == "hello"
        # Cleanup
        monkeypatch.delenv("TEST_QUICKSTART_VAR", raising=False)

    def test_no_env_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _load_dotenv()
        assert result is False

    def test_skips_comments_and_blanks(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nVALID_KEY=value\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VALID_KEY", raising=False)
        _load_dotenv()
        assert os.environ.get("VALID_KEY") == "value"
        monkeypatch.delenv("VALID_KEY", raising=False)


# =============================================================================
# Question getting
# =============================================================================


class TestGetQuestion:
    def test_from_args(self):
        args = argparse.Namespace(question="Test Q")
        assert _get_question(args) == "Test Q"

    def test_demo_uses_default_question(self):
        args = argparse.Namespace(question=None, demo=True)
        result = _get_question(args)
        assert result is not None
        assert "microservices" in result.lower() or "monolith" in result.lower()

    def test_interactive_prompt(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "Interactive Q")
        args = argparse.Namespace(question=None, demo=False)
        assert _get_question(args) == "Interactive Q"

    def test_empty_input_returns_none(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        args = argparse.Namespace(question=None, demo=False)
        assert _get_question(args) is None


# =============================================================================
# Receipt formatting
# =============================================================================


class TestReceiptFormatting:
    SAMPLE = {
        "question": "Migrate to K8s?",
        "verdict": "Yes",
        "confidence": 0.85,
        "rounds": 2,
        "agents": ["claude", "gpt4"],
        "summary": "Proceed with migration.",
        "dissent": [{"agent": "gpt4", "reason": "Timeline risk"}],
    }

    def test_markdown(self):
        md = receipt_to_markdown(self.SAMPLE)
        assert "# Decision Receipt" in md
        assert "Migrate to K8s?" in md
        assert "85%" in md
        assert "Timeline risk" in md

    def test_html(self):
        html = receipt_to_html(self.SAMPLE)
        assert "<!DOCTYPE html>" in html
        assert "Migrate to K8s?" in html
        assert "85%" in html

    def test_save_json(self, tmp_path):
        path = str(tmp_path / "receipt.json")
        _save_receipt(self.SAMPLE, path, "json")
        loaded = json.loads(Path(path).read_text())
        assert loaded["verdict"] == "Yes"

    def test_save_md(self, tmp_path):
        path = str(tmp_path / "receipt.md")
        _save_receipt(self.SAMPLE, path, "md")
        content = Path(path).read_text()
        assert "# Decision Receipt" in content

    def test_save_html(self, tmp_path):
        path = str(tmp_path / "receipt.html")
        _save_receipt(self.SAMPLE, path, "html")
        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content


# =============================================================================
# Command execution
# =============================================================================


class TestCmdQuickstart:
    def test_demo_mode(self, capsys):
        """Test quickstart runs in demo mode with mock agents."""
        args = argparse.Namespace(
            question="Should we use Rust?",
            demo=True,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )
        # Mock the aragora_debate imports
        with patch(
            "aragora.cli.commands.quickstart._run_demo_debate",
            return_value={
                "question": "Should we use Rust?",
                "verdict": "consensus",
                "confidence": 0.85,
                "rounds": 2,
                "agents": ["analyst", "critic", "synthesizer"],
                "summary": "Use Rust for performance-critical paths.",
                "dissent": [],
                "mode": "demo",
            },
        ):
            cmd_quickstart(args)

        output = capsys.readouterr().out
        assert "QUICKSTART" in output
        assert "consensus" in output.lower()

    def test_no_question_exits(self):
        """Test that missing question causes exit."""
        args = argparse.Namespace(
            question=None,
            demo=False,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                cmd_quickstart(args)

    def test_output_saves_receipt(self, tmp_path, capsys):
        """Test that --output flag saves the receipt."""
        output_path = str(tmp_path / "test_receipt.json")
        args = argparse.Namespace(
            question="Test?",
            demo=True,
            output=output_path,
            format="json",
            rounds=2,
            no_browser=True,
        )
        with patch(
            "aragora.cli.commands.quickstart._run_demo_debate",
            return_value={
                "question": "Test?",
                "verdict": "yes",
                "confidence": 0.9,
                "rounds": 2,
                "agents": ["a", "b"],
                "summary": "Summary",
                "dissent": [],
                "mode": "demo",
            },
        ):
            cmd_quickstart(args)

        assert Path(output_path).exists()
        data = json.loads(Path(output_path).read_text())
        assert data["verdict"] == "yes"
