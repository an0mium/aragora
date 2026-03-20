"""Tests for the CLI quickstart command."""

from __future__ import annotations

import argparse
import builtins
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.cli.commands.quickstart import (
    _build_live_receipt,
    _build_live_team,
    _configure_inline_api_key,
    _detect_agents,
    _get_question,
    _load_dotenv,
    _normalize_provider,
    _open_receipt_in_browser,
    _persist_api_key,
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

    def test_parser_accepts_inline_provider_key_flags(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        args = parser.parse_args(
            ["quickstart", "--provider", "openai", "--api-key", "sk-test", "--save-key"]
        )
        assert args.provider == "openai"
        assert args.api_key == "sk-test"
        assert args.save_key is True


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

    def test_preferred_provider_filters_detected_agents(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test2")
        agents = _detect_agents("openai")
        assert agents == [("openai-api", "gpt-4o")]

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            _detect_agents("not-a-provider")


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


class TestProviderSetup:
    def test_normalize_provider_alias(self):
        assert _normalize_provider("openai-api") == "openai"
        assert _normalize_provider("openrouter") == "openrouter"

    def test_configure_inline_api_key_requires_provider(self):
        with pytest.raises(ValueError):
            _configure_inline_api_key(None, "sk-inline")

    def test_persist_api_key_writes_env_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        saved_path = _persist_api_key("openai", "sk-inline")
        assert saved_path == (tmp_path / ".env").resolve()
        assert saved_path.read_text() == "OPENAI_API_KEY=sk-inline\n"

    def test_persist_api_key_replaces_existing_provider_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=old\nOTHER_KEY=keep\n")
        saved_path = _persist_api_key("openai", "sk-new")
        assert saved_path.read_text() == "OPENAI_API_KEY=sk-new\nOTHER_KEY=keep\n"


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
        saved_path = _save_receipt(self.SAMPLE, path, "json")
        loaded = json.loads(saved_path.read_text())
        assert loaded["verdict"] == "Yes"
        assert saved_path == Path(path).resolve()

    def test_save_md(self, tmp_path):
        path = str(tmp_path / "receipt.md")
        saved_path = _save_receipt(self.SAMPLE, path, "md")
        content = saved_path.read_text()
        assert "# Decision Receipt" in content

    def test_save_html(self, tmp_path):
        path = str(tmp_path / "receipt.html")
        saved_path = _save_receipt(self.SAMPLE, path, "html")
        content = saved_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_html_falls_back_to_json_when_formatter_import_fails(self, tmp_path):
        path = tmp_path / "receipt.html"
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aragora.cli.receipt_formatter":
                raise ImportError("formatter unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            saved_path = _save_receipt(self.SAMPLE, path, "html")

        assert json.loads(saved_path.read_text())["verdict"] == "Yes"

    def test_save_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "receipt.json"
        saved_path = _save_receipt(self.SAMPLE, path, "json")
        assert saved_path.exists()

    def test_open_browser_returns_none_when_formatter_import_fails(self):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aragora.cli.receipt_formatter":
                raise ImportError("formatter unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            assert _open_receipt_in_browser(self.SAMPLE) is None


class TestLiveQuickstartHelpers:
    def test_build_live_team_reuses_single_provider_for_real_debate(self):
        team = _build_live_team(
            [("openai-api", "gpt-4o")],
            provider="openai",
            api_key="sk-inline",
        )
        assert [agent["role"] for agent in team] == ["proposer", "critic", "synthesizer"]
        assert [agent["provider"] for agent in team] == ["openai-api"] * 3
        assert [agent["api_key"] for agent in team] == ["sk-inline"] * 3

    def test_build_live_receipt_surfaces_consensus_dissent_and_receipt(self):
        vote_for = argparse.Namespace(agent="alpha", choice="Ship it", reasoning="Best option")
        vote_against = argparse.Namespace(
            agent="beta",
            choice="Wait",
            reasoning="Timeline risk",
        )
        result = argparse.Namespace(
            debate_id="debate-123",
            participants=["alpha", "beta", "gamma"],
            final_answer="Ship it",
            confidence=0.82,
            consensus_reached=True,
            rounds_used=2,
            dissenting_views=["Timeline risk remains unresolved."],
            proposals={"alpha": "Ship it"},
            votes=[vote_for, vote_against],
        )

        receipt = _build_live_receipt(
            result,
            "Should we ship?",
            2,
            [
                {"name": "alpha", "provider": "openai-api"},
                {"name": "beta", "provider": "openai-api"},
                {"name": "gamma", "provider": "openai-api"},
            ],
        )

        assert receipt["receipt_id"] == "debate-123"
        assert receipt["consensus_proof"]["reached"] is True
        assert receipt["consensus_proof"]["supporting_agents"] == ["alpha"]
        assert receipt["consensus_proof"]["dissenting_agents"] == ["beta"]
        assert receipt["dissent"][0]["reason"] == "Timeline risk remains unresolved."


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

    def test_live_mode_saves_default_receipt_artifact(self, tmp_path, monkeypatch, capsys):
        """Test live quickstart saves a deterministic default artifact."""
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(
            question="Should we ship the CLI quickstart?",
            demo=False,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )
        live_result = {
            "question": "Should we ship the CLI quickstart?",
            "verdict": "approve",
            "confidence": 0.91,
            "rounds": 2,
            "agents": ["openai-api"],
            "summary": "Ship the truthful quickstart flow.",
            "dissent": [],
            "mode": "live",
        }

        with (
            patch(
                "aragora.cli.commands.quickstart._detect_agents",
                return_value=[("openai-api", "gpt-4o")],
            ),
            patch(
                "aragora.cli.commands.quickstart._run_live_debate",
                return_value=live_result,
            ),
        ):
            cmd_quickstart(args)

        artifact_path = tmp_path / ".aragora" / "receipts" / "quickstart-live-receipt.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved["mode"] == "live"

        output = capsys.readouterr().out
        assert "Run mode: live" in output
        assert "Mode:       Live" in output
        assert str(artifact_path.resolve()) in output

    def test_inline_provider_key_can_be_saved_and_run_live(self, tmp_path, monkeypatch, capsys):
        """Test quickstart can take one inline key and still build a real debate team."""
        monkeypatch.chdir(tmp_path)
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

        args = argparse.Namespace(
            question="Should we ship the quickstart slice?",
            demo=False,
            provider="openai",
            api_key="sk-inline",
            save_key=True,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )
        live_result = {
            "question": "Should we ship the quickstart slice?",
            "verdict": "consensus",
            "confidence": 0.93,
            "rounds": 2,
            "agents": [
                "openai-api-proposer",
                "openai-api-critic",
                "openai-api-synthesizer",
            ],
            "summary": "Ship the quickstart slice.",
            "dissent": [],
            "mode": "live",
            "receipt_id": "debate-quickstart-1",
        }

        with patch(
            "aragora.cli.commands.quickstart._run_live_debate",
            return_value=live_result,
        ) as run_live_debate:
            cmd_quickstart(args)

        team = run_live_debate.call_args.args[1]
        assert [agent["role"] for agent in team] == ["proposer", "critic", "synthesizer"]
        assert [agent["provider"] for agent in team] == ["openai-api"] * 3
        assert Path(tmp_path / ".env").read_text() == "OPENAI_API_KEY=sk-inline\n"

        output = capsys.readouterr().out
        assert "Saved OPENAI_API_KEY" in output
        assert "Receipt:    debate-quickstart-1" in output

    def test_no_keys_fall_back_to_demo_and_report_demo_artifact(
        self, tmp_path, monkeypatch, capsys
    ):
        """Test quickstart is explicit when it falls back to demo mode."""
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(
            question="Should we use the fallback?",
            demo=False,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )
        demo_result = {
            "question": "Should we use the fallback?",
            "verdict": "consensus",
            "confidence": 0.84,
            "rounds": 2,
            "agents": ["analyst", "critic", "synthesizer"],
            "summary": "Fallback stayed honest about using mock agents.",
            "dissent": [],
            "mode": "demo",
        }

        with (
            patch(
                "aragora.cli.commands.quickstart._detect_agents",
                return_value=[],
            ),
            patch(
                "aragora.cli.commands.quickstart._run_demo_debate",
                return_value=demo_result,
            ),
        ):
            cmd_quickstart(args)

        artifact_path = tmp_path / ".aragora" / "receipts" / "quickstart-demo-receipt.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved["mode"] == "demo"

        output = capsys.readouterr().out
        assert "Falling back to demo mode" in output
        assert "local mock agents, not live model calls" in output
        assert "Mode:       Demo" in output
        assert str(artifact_path.resolve()) in output
