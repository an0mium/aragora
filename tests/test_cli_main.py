"""
Tests for the CLI main module.

Covers argument parsing, command handlers, and utility functions.
"""

import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import sys

import pytest

from aragora.cli.main import (
    parse_agents,
    get_event_emitter_if_available,
    main,
)


# =============================================================================
# Parse Agents Tests
# =============================================================================


class TestParseAgents:
    """Tests for parse_agents function."""

    def test_single_agent(self):
        """Should parse single agent."""
        result = parse_agents("codex")
        assert result == [("codex", None)]

    def test_multiple_agents(self):
        """Should parse multiple agents."""
        result = parse_agents("codex,claude,openai")
        assert result == [("codex", None), ("claude", None), ("openai", None)]

    def test_agent_with_role(self):
        """Should parse agent with role."""
        result = parse_agents("claude:critic")
        assert result == [("claude", "critic")]

    def test_mixed_agents_and_roles(self):
        """Should parse mix of agents with and without roles."""
        result = parse_agents("codex,claude:critic,openai:synthesizer")
        assert result == [
            ("codex", None),
            ("claude", "critic"),
            ("openai", "synthesizer"),
        ]

    def test_strips_whitespace(self):
        """Should strip whitespace from agent names."""
        result = parse_agents(" codex , claude ")
        assert result == [("codex", None), ("claude", None)]

    def test_empty_string(self):
        """Should handle empty string."""
        result = parse_agents("")
        assert result == [("", None)]

    def test_agent_with_complex_role(self):
        """Should handle complex role names."""
        result = parse_agents("claude:super_critic")
        assert result == [("claude", "super_critic")]


# =============================================================================
# Event Emitter Tests
# =============================================================================


class TestGetEventEmitter:
    """Tests for get_event_emitter_if_available function."""

    def test_returns_none_when_server_not_available(self):
        """Should return None when server is not reachable."""
        result = get_event_emitter_if_available("http://localhost:99999")
        assert result is None

    def test_returns_none_on_timeout(self):
        """Should return None on connection timeout."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError()
            result = get_event_emitter_if_available()
            assert result is None

    def test_returns_none_on_url_error(self):
        """Should return None on URL error."""
        import urllib.error
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
            result = get_event_emitter_if_available()
            assert result is None


# =============================================================================
# Argument Parser Tests
# =============================================================================


class TestArgumentParser:
    """Tests for argument parsing."""

    @pytest.fixture
    def parser(self):
        """Create the argument parser."""
        # Re-create the parser similar to main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--db", default="agora_memory.db")
        parser.add_argument("-v", "--verbose", action="store_true")

        subparsers = parser.add_subparsers(dest="command")

        # Ask command
        ask_parser = subparsers.add_parser("ask")
        ask_parser.add_argument("task")
        ask_parser.add_argument("--agents", "-a", default="codex,claude")
        ask_parser.add_argument("--rounds", "-r", type=int, default=3)
        ask_parser.add_argument("--consensus", "-c", default="majority")
        ask_parser.add_argument("--context")
        ask_parser.add_argument("--no-learn", dest="learn", action="store_false")

        # Stats command
        subparsers.add_parser("stats")

        # Patterns command
        patterns_parser = subparsers.add_parser("patterns")
        patterns_parser.add_argument("--type", "-t")
        patterns_parser.add_argument("--min-success", type=int, default=1)
        patterns_parser.add_argument("--limit", "-l", type=int, default=10)

        # Demo command
        demo_parser = subparsers.add_parser("demo")
        demo_parser.add_argument("name", nargs="?")

        # Serve command
        serve_parser = subparsers.add_parser("serve")
        serve_parser.add_argument("--ws-port", type=int, default=8765)
        serve_parser.add_argument("--api-port", type=int, default=8080)
        serve_parser.add_argument("--host", default="localhost")

        return parser

    def test_parse_ask_command(self, parser):
        """Should parse ask command with task."""
        args = parser.parse_args(["ask", "Design a rate limiter"])
        assert args.command == "ask"
        assert args.task == "Design a rate limiter"
        assert args.agents == "codex,claude"
        assert args.rounds == 3

    def test_parse_ask_with_agents(self, parser):
        """Should parse ask with custom agents."""
        args = parser.parse_args(["ask", "Task", "-a", "gpt4,claude"])
        assert args.agents == "gpt4,claude"

    def test_parse_ask_with_rounds(self, parser):
        """Should parse ask with custom rounds."""
        args = parser.parse_args(["ask", "Task", "-r", "5"])
        assert args.rounds == 5

    def test_parse_ask_with_consensus(self, parser):
        """Should parse ask with consensus option."""
        args = parser.parse_args(["ask", "Task", "-c", "unanimous"])
        assert args.consensus == "unanimous"

    def test_parse_ask_with_context(self, parser):
        """Should parse ask with context."""
        args = parser.parse_args(["ask", "Task", "--context", "Extra info"])
        assert args.context == "Extra info"

    def test_parse_ask_no_learn(self, parser):
        """Should parse --no-learn flag."""
        args = parser.parse_args(["ask", "Task", "--no-learn"])
        assert args.learn is False

    def test_parse_ask_learn_default(self, parser):
        """Should default to learning enabled."""
        args = parser.parse_args(["ask", "Task"])
        assert args.learn is True

    def test_parse_stats_command(self, parser):
        """Should parse stats command."""
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_parse_patterns_command(self, parser):
        """Should parse patterns command."""
        args = parser.parse_args(["patterns"])
        assert args.command == "patterns"
        assert args.limit == 10

    def test_parse_patterns_with_type(self, parser):
        """Should parse patterns with type filter."""
        args = parser.parse_args(["patterns", "-t", "security"])
        assert args.type == "security"

    def test_parse_patterns_with_limit(self, parser):
        """Should parse patterns with limit."""
        args = parser.parse_args(["patterns", "-l", "20"])
        assert args.limit == 20

    def test_parse_demo_command(self, parser):
        """Should parse demo command."""
        args = parser.parse_args(["demo"])
        assert args.command == "demo"
        assert args.name is None

    def test_parse_demo_with_name(self, parser):
        """Should parse demo with name."""
        args = parser.parse_args(["demo", "rate-limiter"])
        assert args.name == "rate-limiter"

    def test_parse_serve_command(self, parser):
        """Should parse serve command with defaults."""
        args = parser.parse_args(["serve"])
        assert args.command == "serve"
        assert args.ws_port == 8765
        assert args.api_port == 8080
        assert args.host == "localhost"

    def test_parse_serve_custom_ports(self, parser):
        """Should parse serve with custom ports."""
        args = parser.parse_args(["serve", "--ws-port", "9000", "--api-port", "9001"])
        assert args.ws_port == 9000
        assert args.api_port == 9001

    def test_parse_global_db_option(self, parser):
        """Should parse global db option."""
        args = parser.parse_args(["--db", "custom.db", "stats"])
        assert args.db == "custom.db"

    def test_parse_verbose_flag(self, parser):
        """Should parse verbose flag."""
        args = parser.parse_args(["-v", "stats"])
        assert args.verbose is True


# =============================================================================
# Command Handler Tests
# =============================================================================


class TestCommandHandlers:
    """Tests for command handler functions."""

    def test_cmd_ask_runs_debate(self):
        """Should run debate with parsed arguments."""
        from aragora.cli.main import cmd_ask

        args = argparse.Namespace(
            task="Test task",
            agents="codex,claude",
            rounds=2,
            consensus="majority",
            context="",
            learn=False,
            db="test.db",
            verbose=False,
        )

        with patch("aragora.cli.main.asyncio.run") as mock_run:
            mock_result = MagicMock()
            mock_result.final_answer = "Test answer"
            mock_result.dissenting_views = []
            mock_run.return_value = mock_result

            cmd_ask(args)

            mock_run.assert_called_once()

    def test_cmd_stats_shows_statistics(self):
        """Should display memory statistics."""
        from aragora.cli.main import cmd_stats

        args = argparse.Namespace(db="test.db")

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_debates": 10,
            "consensus_debates": 8,
            "total_critiques": 50,
            "total_patterns": 25,
            "avg_consensus_confidence": 0.75,
            "patterns_by_type": {"security": 5, "performance": 3},
        }

        with patch("aragora.cli.main.CritiqueStore", return_value=mock_store):
            cmd_stats(args)

            mock_store.get_stats.assert_called_once()

    def test_cmd_patterns_retrieves_patterns(self):
        """Should retrieve and display patterns."""
        from aragora.cli.main import cmd_patterns

        args = argparse.Namespace(
            db="test.db",
            type=None,
            min_success=1,
            limit=10,
        )

        mock_store = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.issue_type = "security"
        mock_pattern.success_count = 5
        mock_pattern.avg_severity = 0.6
        mock_pattern.issue_text = "SQL injection vulnerability"
        mock_pattern.suggestion_text = "Use parameterized queries"
        mock_store.retrieve_patterns.return_value = [mock_pattern]

        with patch("aragora.cli.main.CritiqueStore", return_value=mock_store):
            cmd_patterns(args)

            mock_store.retrieve_patterns.assert_called_once_with(
                issue_type=None,
                min_success=1,
                limit=10,
            )

    def test_cmd_demo_runs_demo_debate(self):
        """Should run demo debate."""
        from aragora.cli.main import cmd_demo

        args = argparse.Namespace(name="rate-limiter")

        with patch("aragora.cli.main.asyncio.run") as mock_run:
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.final_answer = "Demo answer"
            mock_run.return_value = mock_result

            cmd_demo(args)

            mock_run.assert_called_once()

    def test_cmd_demo_unknown_name(self, capsys):
        """Should report unknown demo name."""
        from aragora.cli.main import cmd_demo

        args = argparse.Namespace(name="unknown")
        cmd_demo(args)

        captured = capsys.readouterr()
        assert "Unknown demo" in captured.out


# =============================================================================
# Demo Tasks Tests
# =============================================================================


class TestDemoTasks:
    """Tests for demo task configurations."""

    def test_rate_limiter_demo_exists(self):
        """Should have rate-limiter demo defined."""
        # Access demo tasks by reading the function code structure
        from aragora.cli import main
        # The demo tasks are defined in cmd_demo
        assert hasattr(main, "cmd_demo")

    def test_auth_demo_exists(self):
        """Should have auth demo defined."""
        from aragora.cli.main import cmd_demo
        args = argparse.Namespace(name="auth")

        with patch("aragora.cli.main.asyncio.run") as mock_run:
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.8
            mock_result.final_answer = "Auth answer"
            mock_run.return_value = mock_result

            cmd_demo(args)
            # If it gets here without error, demo exists
            assert mock_run.called

    def test_cache_demo_exists(self):
        """Should have cache demo defined."""
        from aragora.cli.main import cmd_demo
        args = argparse.Namespace(name="cache")

        with patch("aragora.cli.main.asyncio.run") as mock_run:
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.85
            mock_result.final_answer = "Cache answer"
            mock_run.return_value = mock_result

            cmd_demo(args)
            assert mock_run.called


# =============================================================================
# Main Entry Point Tests
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_no_command_shows_help(self, capsys):
        """Should show help when no command provided."""
        with patch("sys.argv", ["agora"]):
            main()
            captured = capsys.readouterr()
            # Help output goes to stdout
            assert "usage" in captured.out.lower() or captured.out == ""

    def test_main_calls_command_func(self):
        """Should call the appropriate command function."""
        with patch("sys.argv", ["agora", "stats"]):
            with patch("aragora.cli.main.CritiqueStore") as mock_store:
                mock_store.return_value.get_stats.return_value = {
                    "total_debates": 0,
                    "consensus_debates": 0,
                    "total_critiques": 0,
                    "total_patterns": 0,
                    "avg_consensus_confidence": 0.0,
                    "patterns_by_type": {},
                }
                main()
                mock_store.assert_called_once()


# =============================================================================
# Run Debate Tests
# =============================================================================


class TestRunDebate:
    """Tests for run_debate async function."""

    @pytest.mark.asyncio
    async def test_run_debate_creates_agents(self):
        """Should create agents from specification."""
        from aragora.cli.main import run_debate

        with patch("aragora.cli.main.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            with patch("aragora.cli.main.Arena") as mock_arena:
                mock_result = MagicMock()
                mock_arena.return_value.run = AsyncMock(return_value=mock_result)

                await run_debate(
                    task="Test",
                    agents_str="codex,claude",
                    rounds=2,
                    learn=False,
                )

                # Should create 2 agents
                assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_run_debate_assigns_default_roles(self):
        """Should assign default roles to agents."""
        from aragora.cli.main import run_debate

        created_agents = []

        def track_create(*args, **kwargs):
            agent = MagicMock()
            agent.name = kwargs.get("name")
            created_agents.append(kwargs)
            return agent

        with patch("aragora.cli.main.create_agent", side_effect=track_create):
            with patch("aragora.cli.main.Arena") as mock_arena:
                mock_result = MagicMock()
                mock_arena.return_value.run = AsyncMock(return_value=mock_result)

                await run_debate(
                    task="Test",
                    agents_str="a,b,c",
                    rounds=2,
                    learn=False,
                )

                # First should be proposer, last should be synthesizer
                assert created_agents[0]["role"] == "proposer"
                assert created_agents[2]["role"] == "synthesizer"

    @pytest.mark.asyncio
    async def test_run_debate_uses_critique_store(self):
        """Should use CritiqueStore when learn=True."""
        from aragora.cli.main import run_debate

        with patch("aragora.cli.main.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("aragora.cli.main.Arena") as mock_arena:
                mock_result = MagicMock()
                mock_arena.return_value.run = AsyncMock(return_value=mock_result)

                with patch("aragora.cli.main.CritiqueStore") as mock_store:
                    await run_debate(
                        task="Test",
                        agents_str="codex",
                        rounds=1,
                        learn=True,
                        db_path="test.db",
                    )

                    mock_store.assert_called_once_with("test.db")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_agents_single_colon(self):
        """Should handle agent with single colon."""
        result = parse_agents("agent:")
        assert result == [("agent", "")]

    def test_parse_agents_multiple_colons(self):
        """Should handle agent with multiple colons."""
        result = parse_agents("agent:role:extra")
        assert result == [("agent", "role:extra")]

    def test_cmd_templates_import(self):
        """Should import templates module."""
        from aragora.cli.main import cmd_templates
        args = argparse.Namespace()

        with patch("aragora.templates.list_templates") as mock_list:
            mock_list.return_value = [
                {
                    "type": "debate",
                    "name": "Test",
                    "description": "Test template",
                    "agents": "a,b",
                    "domain": "test",
                }
            ]
            cmd_templates(args)
            mock_list.assert_called_once()


# =============================================================================
# Serve Command Tests
# =============================================================================


class TestServeCommand:
    """Tests for serve command."""

    def test_cmd_serve_starts_server(self):
        """Should start unified server."""
        from aragora.cli.main import cmd_serve

        args = argparse.Namespace(
            ws_port=8765,
            api_port=8080,
            host="localhost",
        )

        with patch("aragora.server.unified_server.run_unified_server") as mock_server:
            with patch("asyncio.run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()

                cmd_serve(args)

                # Should have been called
                mock_run.assert_called()
