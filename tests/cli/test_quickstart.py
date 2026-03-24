"""Tests for the CLI quickstart command."""

from __future__ import annotations

import argparse
import asyncio
import builtins
import json
import os
import ssl
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.base import create_agent as create_real_agent
from aragora.cli.commands.quickstart import (
    _build_live_receipt,
    _build_live_team,
    _can_reach_provider_tls,
    _configure_inline_api_key,
    _detect_agents,
    _filter_reachable_live_agents,
    _get_question,
    _load_dotenv,
    _normalize_provider,
    _open_receipt_in_browser,
    _resolve_rounds,
    _run_demo_debate,
    _run_live_debate,
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

    def test_parser_inline_provider_key_flags(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_quickstart_parser(subparsers)
        args = parser.parse_args(
            [
                "quickstart",
                "--provider",
                "openai",
                "--api-key",
                "sk-inline",
                "--save-key",
            ]
        )
        assert args.provider == "openai"
        assert args.api_key == "sk-inline"
        assert args.save_key is True

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

    def test_preferred_provider_filters_detected_agents(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
        agents = _detect_agents("openai")
        assert agents == [("openai-api", "gpt-4o-mini")]

    def test_invalid_preferred_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            _detect_agents("bogus")


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


class TestResolveRounds:
    def test_demo_defaults_to_two_rounds(self):
        assert _resolve_rounds(None, use_demo=True) == 2

    def test_live_defaults_to_one_round(self):
        assert _resolve_rounds(None, use_demo=False) == 1

    def test_explicit_rounds_win(self):
        assert _resolve_rounds(4, use_demo=False) == 4


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


class TestInlineApiKeys:
    def test_normalize_provider_aliases(self):
        assert _normalize_provider("openai-api") == "openai"
        assert _normalize_provider("xai") == "grok"
        assert _normalize_provider("nope") is None

    def test_configure_inline_api_key_sets_env_without_persisting(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        normalized_provider, saved_key = _configure_inline_api_key(
            "openai",
            "sk-inline",
            save_key=False,
        )
        assert normalized_provider == "openai"
        assert saved_key is None
        assert os.environ["OPENAI_API_KEY"] == "sk-inline"
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_configure_inline_api_key_persists_via_secure_store(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("aragora.cli.api_keys.set_provider_key") as set_provider_key:
            set_provider_key.return_value = MagicMock(
                provider="openai",
                env_var="OPENAI_API_KEY",
                backend="macos-keychain",
                masked_value="sk-i...line",
            )
            normalized_provider, saved_key = _configure_inline_api_key(
                "openai",
                "sk-inline",
                save_key=True,
            )

        assert normalized_provider == "openai"
        assert saved_key == {
            "provider": "openai",
            "env_var": "OPENAI_API_KEY",
            "backend": "macos-keychain",
            "masked_value": "sk-i...line",
        }
        assert os.environ["OPENAI_API_KEY"] == "sk-inline"
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_configure_inline_api_key_requires_provider(self):
        with pytest.raises(ValueError, match="--api-key requires --provider"):
            _configure_inline_api_key(None, "sk-inline", save_key=False)


class TestLiveQuickstartHelpers:
    def test_build_live_team_reuses_single_provider_for_real_debate(self):
        team = _build_live_team(
            [("openai-api", "gpt-4o")],
            provider="openai",
            api_key="sk-inline",
        )
        assert [agent["role"] for agent in team] == ["proposer", "critic", "synthesizer"]
        # Primary provider is openai-api; OpenRouter fallback may be mixed in
        assert team[0]["provider"] == "openai-api"
        assert team[0]["api_key"] == "sk-inline"

    def test_build_live_team_stays_single_provider_even_with_openrouter_available(self):
        team = _build_live_team(
            [
                ("anthropic-api", "claude-sonnet-4-5-20250929"),
                ("openai-api", "gpt-4o"),
                ("gemini", None),
            ]
        )

        providers = [agent["provider"] for agent in team]
        assert providers[0] == "gemini"
        assert providers == ["gemini"] * 3

    def test_build_live_receipt_surfaces_consensus_dissent_and_receipt(self):
        from aragora.gauntlet.receipt_models import DecisionReceipt

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
            metadata={"thinking_traces": {"alpha": "Consider the risk tradeoffs carefully."}},
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
        assert receipt["artifact_hash"]
        assert receipt["consensus_proof"]["reached"] is True
        assert receipt["consensus_proof"]["supporting_agents"] == ["alpha"]
        assert receipt["consensus_proof"]["dissenting_agents"] == ["beta"]
        assert receipt["thinking_traces"] == {"alpha": "Consider the risk tradeoffs carefully."}
        assert receipt["dissent"][0]["reason"] == "Timeline risk remains unresolved."
        assert receipt["receipt"]["artifact_hash"] == receipt["artifact_hash"]
        assert DecisionReceipt.from_dict(receipt).verify_integrity() is True

    def test_build_live_receipt_clamps_confidence_into_unit_interval(self):
        result = argparse.Namespace(
            debate_id="debate-123",
            participants=["alpha"],
            final_answer="Ship it",
            confidence=1.2,
            consensus_reached=True,
            rounds_used=1,
            dissenting_views=[],
            proposals={},
            votes=[],
        )

        receipt = _build_live_receipt(
            result,
            "Should we ship?",
            1,
            [{"name": "alpha", "provider": "openai-api"}],
        )

        assert receipt["confidence"] == 1.0
        assert receipt["receipt"]["confidence"] == 1.0
        assert receipt["consensus_proof"]["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_can_reach_provider_tls_normalizes_wrapped_cert_errors(self):
        with patch(
            "aragora.cli.commands.quickstart.asyncio.open_connection",
            side_effect=ssl.SSLError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate"
            ),
        ):
            ok, detail = await _can_reach_provider_tls("openai-api")

        assert ok is False
        assert detail == "CERTIFICATE_VERIFY_FAILED"

    @pytest.mark.asyncio
    async def test_can_reach_provider_tls_ignores_close_noise_after_handshake(self):
        writer = MagicMock()
        writer.wait_closed = AsyncMock(side_effect=ConnectionResetError("socket closed"))

        with patch(
            "aragora.cli.commands.quickstart.asyncio.open_connection",
            return_value=(MagicMock(), writer),
        ):
            ok, detail = await _can_reach_provider_tls("openai-api")

        assert ok is True
        assert detail is None
        writer.close.assert_called_once()


# =============================================================================
# Command execution
# =============================================================================


class TestCmdQuickstart:
    @pytest.mark.asyncio
    async def test_run_demo_debate_uses_builtin_demo_agents(self):
        result = await _run_demo_debate("Should we ship the fallback fix?", rounds=2)

        assert result["mode"] == "demo"
        assert result["verdict"] == "consensus"
        assert result["confidence"] == 0.85
        assert result["agents"] == ["analyst", "critic", "synthesizer"]
        assert "Demo synthesis for: Should we ship the fallback fix?" in result["summary"]

    @pytest.mark.asyncio
    async def test_run_live_debate_raises_when_arena_returns_none(self):
        """Live quickstart should fail closed when Arena produces no result."""
        mock_agent = MagicMock()
        mock_agent.name = "openai-api"

        with (
            patch(
                "aragora.cli.commands.quickstart._filter_reachable_live_agents",
                return_value=[("openai-api", "gpt-4o")],
            ),
            patch("aragora.agents.base.create_agent", return_value=mock_agent),
            patch("aragora.debate.orchestrator.Arena") as mock_arena_cls,
        ):
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=None)
            mock_arena_cls.return_value = mock_arena

            with pytest.raises(RuntimeError, match="Live debate returned no result"):
                await _run_live_debate(
                    "Should we ship the quickstart path?",
                    [("openai-api", "gpt-4o")],
                    rounds=2,
                )

    @pytest.mark.asyncio
    async def test_run_live_debate_times_out_cleanly(self, monkeypatch):
        """Live quickstart should bound long-running onboarding debates."""
        mock_agent = MagicMock()
        mock_agent.name = "openai-api"

        async def slow_run():
            await asyncio.sleep(0.05)
            return MagicMock()

        with (
            patch(
                "aragora.cli.commands.quickstart._filter_reachable_live_agents",
                return_value=[("openai-api", "gpt-4o")],
            ),
            patch("aragora.agents.base.create_agent", return_value=mock_agent),
            patch("aragora.debate.orchestrator.Arena") as mock_arena_cls,
        ):
            mock_arena = MagicMock()
            mock_arena.run = slow_run
            mock_arena_cls.return_value = mock_arena
            monkeypatch.setattr(
                "aragora.cli.commands.quickstart._LIVE_DEBATE_TIMEOUT_SECONDS",
                0.01,
            )

            with pytest.raises(RuntimeError, match="Live debate timed out"):
                await _run_live_debate(
                    "Should we ship the quickstart path?",
                    [("openai-api", "gpt-4o")],
                    rounds=2,
                )

    @pytest.mark.asyncio
    async def test_run_live_debate_uses_bounded_quickstart_profile(self):
        """Live quickstart should disable heavyweight debate subsystems by default."""
        mock_agent = MagicMock()
        mock_agent.name = "openai-api"
        mock_result = argparse.Namespace()
        mock_insight_store = MagicMock()

        with (
            patch(
                "aragora.cli.commands.quickstart._filter_reachable_live_agents",
                return_value=[("openai-api", "gpt-4o")],
            ),
            patch("aragora.agents.base.create_agent", return_value=mock_agent),
            patch("aragora.insights.store.InsightStore", return_value=mock_insight_store),
            patch(
                "aragora.cli.commands.quickstart._build_live_receipt",
                return_value={"mode": "live", "rounds": 1},
            ),
            patch("aragora.debate.orchestrator.Arena") as mock_arena_cls,
        ):
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=mock_result)
            mock_arena.knowledge_mound = object()
            mock_arena.enable_knowledge_ingestion = False
            mock_arena_cls.return_value = mock_arena

            result = await _run_live_debate(
                "Should we ship the quickstart path?",
                [("openai-api", "gpt-4o")],
                rounds=1,
            )

        protocol = mock_arena_cls.call_args.args[2]
        assert protocol.rounds == 1
        assert protocol.consensus == "majority"
        assert protocol.convergence_detection is False
        assert protocol.vote_grouping is False
        assert protocol.enable_trickster is False
        assert protocol.enable_research is False
        assert protocol.enable_trending_injection is False
        assert protocol.enable_llm_question_classification is False
        assert protocol.enable_llm_synthesis is False

        arena_kwargs = mock_arena_cls.call_args.kwargs
        assert arena_kwargs["insight_store"] is mock_insight_store
        assert arena_kwargs["disable_post_debate_pipeline"] is True
        assert mock_arena.enable_introspection is False

        # Quickstart now uses config objects instead of individual kwargs
        memory_config = arena_kwargs["memory_config"]
        assert memory_config.enable_knowledge_retrieval is True
        assert memory_config.enable_knowledge_ingestion is False
        assert memory_config.auto_create_knowledge_mound is True
        assert memory_config.enable_belief_guidance is False
        assert memory_config.enable_cross_debate_memory is False
        assert memory_config.use_rlm_limiter is False

        knowledge_config = arena_kwargs["knowledge_config"]
        assert knowledge_config.enable_knowledge_retrieval is True
        assert knowledge_config.enable_knowledge_ingestion is False
        assert knowledge_config.enable_belief_guidance is False

        ml_config = arena_kwargs["ml_config"]
        assert ml_config.enable_ml_delegation is False
        assert ml_config.enable_quality_gates is False
        assert ml_config.enable_consensus_estimation is False
        assert result["km_ingested"] is False

    @pytest.mark.asyncio
    async def test_run_live_debate_skips_crux_event_dispatch_in_bounded_profile(self):
        """Bounded quickstart should not emit crux events through the webhook dispatcher."""

        def fake_create_agent(
            agent_type: str,
            *,
            name: str,
            role: str,
            model: str | None = None,
            api_key: str | None = None,
        ):
            del agent_type, model, api_key
            return create_real_agent("demo", name=name, role=role)

        with (
            patch(
                "aragora.cli.commands.quickstart._filter_reachable_live_agents",
                return_value=[("openai-api", "gpt-4o")],
            ),
            patch("aragora.agents.base.create_agent", side_effect=fake_create_agent),
            patch(
                "aragora.events.dispatcher.dispatch_event",
                side_effect=AssertionError("bounded quickstart should not dispatch crux events"),
            ),
        ):
            result = await _run_live_debate(
                "Should we ship the quickstart path?",
                [("openai-api", "gpt-4o")],
                rounds=1,
            )

        assert result["mode"] == "live"
        assert result["rounds"] == 1

    @pytest.mark.asyncio
    async def test_filter_reachable_live_agents_returns_empty_on_tls_failure(self):
        """When all providers fail TLS, return empty list for demo fallback."""
        with patch(
            "aragora.cli.commands.quickstart._can_reach_provider_tls",
            new=AsyncMock(return_value=(False, "CERTIFICATE_VERIFY_FAILED")),
        ):
            result = await _filter_reachable_live_agents(
                [("openai-api", "gpt-4o"), ("gemini", None)]
            )
            assert result == []

    @pytest.mark.asyncio
    async def test_filter_reachable_live_agents_treats_wrapped_tls_detail_as_cert_failure(self):
        with patch(
            "aragora.cli.commands.quickstart._can_reach_provider_tls",
            new=AsyncMock(
                return_value=(
                    False,
                    "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed",
                )
            ),
        ):
            reachable = await _filter_reachable_live_agents([("openai-api", "gpt-4o")])

        assert reachable == []

    @pytest.mark.asyncio
    async def test_filter_reachable_live_agents_keeps_healthy_subset(self):
        """Providers that pass preflight should still be used."""

        async def fake_probe(provider: str) -> tuple[bool, str | None]:
            if provider == "openai-api":
                return True, None
            return False, "connection refused"

        with patch(
            "aragora.cli.commands.quickstart._can_reach_provider_tls",
            side_effect=fake_probe,
        ):
            reachable = await _filter_reachable_live_agents(
                [("openai-api", "gpt-4o"), ("gemini", None)]
            )

        assert reachable == [("openai-api", "gpt-4o")]

    def test_demo_mode(self, capsys):
        """Test quickstart runs in demo mode with mock agents."""
        args = argparse.Namespace(
            question="Should we use Rust?",
            demo=True,
            provider=None,
            api_key=None,
            save_key=False,
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

    def test_live_mode_defaults_to_one_round_when_unspecified(self, capsys):
        args = argparse.Namespace(
            question="Should we use the live default?",
            demo=False,
            provider=None,
            api_key=None,
            save_key=False,
            output=None,
            format="json",
            rounds=None,
            no_browser=True,
        )

        with (
            patch(
                "aragora.cli.commands.quickstart._detect_agents",
                return_value=[("openai-api", "gpt-4o")],
            ),
            patch(
                "aragora.cli.commands.quickstart._run_live_debate",
                return_value={
                    "question": "Should we use the live default?",
                    "verdict": "approve",
                    "confidence": 0.8,
                    "rounds": 1,
                    "agents": ["openai-api"],
                    "summary": "Use the bounded live default.",
                    "dissent": [],
                    "mode": "live",
                },
            ) as run_live_debate,
        ):
            cmd_quickstart(args)

        assert run_live_debate.call_args.args[2] == 1

    def test_demo_mode_defaults_to_two_rounds_when_unspecified(self, capsys):
        args = argparse.Namespace(
            question="Should we use the demo default?",
            demo=True,
            provider=None,
            api_key=None,
            save_key=False,
            output=None,
            format="json",
            rounds=None,
            no_browser=True,
        )

        with patch(
            "aragora.cli.commands.quickstart._run_demo_debate",
            return_value={
                "question": "Should we use the demo default?",
                "verdict": "consensus",
                "confidence": 0.85,
                "rounds": 2,
                "agents": ["analyst", "critic", "synthesizer"],
                "summary": "Keep demo defaults stable.",
                "dissent": [],
                "mode": "demo",
            },
        ) as run_demo_debate:
            cmd_quickstart(args)

        assert run_demo_debate.call_args.args[1] == 2

    def test_no_question_exits(self):
        """Test that missing question causes exit."""
        args = argparse.Namespace(
            question=None,
            demo=False,
            provider=None,
            api_key=None,
            save_key=False,
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
            provider=None,
            api_key=None,
            save_key=False,
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
            provider=None,
            api_key=None,
            save_key=False,
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
        assert "Agents: openai-api" in output
        assert "Mode:       Live" in output
        assert str(artifact_path.resolve()) in output

    def test_inline_provider_key_can_be_saved_and_run_live(self, tmp_path, monkeypatch, capsys):
        """Test quickstart can take one inline key and save it securely."""
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
            "verdict": "PASS",
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
            "artifact_hash": "abc123def4567890",
            "consensus_proof": {"reached": True},
        }

        with (
            patch("aragora.cli.api_keys.set_provider_key") as set_provider_key,
            patch(
                "aragora.cli.commands.quickstart._run_live_debate",
                return_value=live_result,
            ) as run_live_debate,
        ):
            set_provider_key.return_value = MagicMock(
                provider="openai",
                env_var="OPENAI_API_KEY",
                backend="macos-keychain",
                masked_value="sk-i...line",
            )
            cmd_quickstart(args)

        assert os.environ["OPENAI_API_KEY"] == "sk-inline"
        assert run_live_debate.call_args.kwargs["provider"] == "openai"
        assert run_live_debate.call_args.kwargs["api_key"] == "sk-inline"

        output = capsys.readouterr().out
        assert "Saved OPENAI_API_KEY to secure store" in output
        assert "Receipt:    debate-quickstart-1" in output
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_no_keys_fall_back_to_demo_and_report_demo_artifact(
        self, tmp_path, monkeypatch, capsys
    ):
        """Test quickstart is explicit when it falls back to demo mode."""
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(
            question="Should we use the fallback?",
            demo=False,
            provider=None,
            api_key=None,
            save_key=False,
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

    def test_live_mode_falls_back_to_demo_on_tls_failure(self, capsys):
        """TLS/provider failures should fall back to demo mode, not exit."""
        args = argparse.Namespace(
            question="Should we use the live path?",
            demo=False,
            provider=None,
            api_key=None,
            save_key=False,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )

        mock_demo_result = {
            "question": "Should we use the live path?",
            "verdict": "consensus",
            "confidence": 0.85,
            "rounds": 2,
            "agents": ["analyst", "critic", "synthesizer"],
            "summary": "Demo result",
            "dissent": [],
            "mode": "demo",
        }

        with (
            patch(
                "aragora.cli.commands.quickstart._detect_agents",
                return_value=[("gemini", None)],
            ),
            patch(
                "aragora.cli.commands.quickstart._run_live_debate",
                side_effect=RuntimeError("Live debate failed: CERTIFICATE_VERIFY_FAILED"),
            ),
            patch(
                "aragora.cli.commands.quickstart._run_demo_debate",
                return_value=mock_demo_result,
            ),
        ):
            cmd_quickstart(args)

        output = capsys.readouterr().out
        assert "Falling back to demo" in output
        assert "RESULT" in output  # Demo result was displayed

    def test_live_mode_real_demo_fallback_survives_without_legacy_demo_package(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(
            question="Should we keep the real fallback path working?",
            demo=False,
            provider=None,
            api_key=None,
            save_key=False,
            output=None,
            format="json",
            rounds=2,
            no_browser=True,
        )

        with (
            patch(
                "aragora.cli.commands.quickstart._detect_agents",
                return_value=[("gemini", "gemini-3.1-pro")],
            ),
            patch(
                "aragora.cli.commands.quickstart._run_live_debate",
                side_effect=RuntimeError("Live debate timed out after 120s"),
            ),
        ):
            cmd_quickstart(args)

        artifact_path = tmp_path / ".aragora" / "receipts" / "quickstart-demo-receipt.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved["mode"] == "demo"
        assert saved["agents"] == ["analyst", "critic", "synthesizer"]
        assert (
            "Demo synthesis for: Should we keep the real fallback path working?" in saved["summary"]
        )

        output = capsys.readouterr().out
        assert "Live debate failed: Live debate timed out after 120s" in output
        assert "Mode:       Demo" in output
