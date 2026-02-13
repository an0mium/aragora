"""
Tests for the CLI demo command.

Validates the demo command runs a self-contained adversarial debate
without requiring API keys, produces visible output, and supports
all flags (--list, --topic, --server).
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.demo import (
    DEMO_TASKS,
    _AGENT_CONFIGS,
    _DEFAULT_DEMO,
    _print_banner,
    _print_result,
    _wrap,
    list_demos,
    main,
    run_demo,
)


# ---------------------------------------------------------------------------
# DEMO_TASKS configuration
# ---------------------------------------------------------------------------


class TestDemoTasks:
    """Tests for DEMO_TASKS configuration."""

    def test_demo_tasks_not_empty(self):
        assert len(DEMO_TASKS) > 0

    def test_all_demos_have_required_fields(self):
        for name, demo in DEMO_TASKS.items():
            assert "topic" in demo, f"Demo '{name}' missing 'topic'"
            assert "description" in demo, f"Demo '{name}' missing 'description'"

    def test_all_demos_have_string_topics(self):
        for name, demo in DEMO_TASKS.items():
            assert isinstance(demo["topic"], str)
            assert len(demo["topic"]) > 10, f"Demo '{name}' topic too short"

    def test_default_demo_exists(self):
        assert _DEFAULT_DEMO in DEMO_TASKS

    def test_microservices_demo_exists(self):
        assert "microservices" in DEMO_TASKS

    def test_rate_limiter_demo_exists(self):
        assert "rate-limiter" in DEMO_TASKS

    def test_demo_names_are_lowercase(self):
        for name in DEMO_TASKS:
            assert name == name.lower(), f"Demo name '{name}' should be lowercase"
            assert " " not in name, f"Demo name '{name}' should not have spaces"


# ---------------------------------------------------------------------------
# list_demos
# ---------------------------------------------------------------------------


class TestListDemos:
    def test_returns_list(self):
        assert isinstance(list_demos(), list)

    def test_returns_all_demo_names(self):
        assert set(list_demos()) == set(DEMO_TASKS.keys())


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


class TestAgentConfigs:
    def test_has_four_agents(self):
        assert len(_AGENT_CONFIGS) == 4

    def test_each_agent_has_name_and_style(self):
        for name, style in _AGENT_CONFIGS:
            assert isinstance(name, str)
            assert style in ("supportive", "critical", "balanced", "contrarian")

    def test_has_diverse_styles(self):
        styles = {s for _, s in _AGENT_CONFIGS}
        assert len(styles) >= 3, "Agents should have diverse styles"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestWrap:
    def test_short_text_single_line(self):
        assert _wrap("hello world", width=72) == ["hello world"]

    def test_long_text_wraps(self):
        text = "word " * 20
        lines = _wrap(text.strip(), width=30)
        assert len(lines) > 1
        assert all(len(line) <= 35 for line in lines)  # allow slight overflow for words

    def test_empty_string(self):
        assert _wrap("") == [""]


class TestPrintBanner:
    def test_prints_topic_and_agents(self, capsys):
        _print_banner("Test topic", ["Agent1", "Agent2"])
        captured = capsys.readouterr()
        assert "Test topic" in captured.out
        assert "Agent1" in captured.out
        assert "ARAGORA DEMO" in captured.out


class TestPrintResult:
    def test_prints_verdict_and_confidence(self, capsys):
        from aragora_debate.types import DebateResult, DecisionReceipt, Verdict, Consensus, ConsensusMethod

        receipt = DecisionReceipt(
            receipt_id="DR-TEST-abc123",
            question="Test?",
            verdict=Verdict.APPROVED,
            confidence=0.85,
            consensus=Consensus(
                reached=True,
                method=ConsensusMethod.MAJORITY,
                confidence=0.85,
                supporting_agents=["a", "b"],
            ),
            agents=["a", "b"],
            rounds_used=2,
            signature="abc123def456",
            signature_algorithm="SHA-256-content-hash",
        )
        result = DebateResult(
            task="Test?",
            final_answer="We should proceed.",
            confidence=0.85,
            consensus_reached=True,
            verdict=Verdict.APPROVED,
            rounds_used=2,
            participants=["a", "b"],
            proposals={"a": "Proposal A text", "b": "Proposal B text"},
            receipt=receipt,
        )
        _print_result(result, 0.5)
        captured = capsys.readouterr()
        assert "Approved" in captured.out
        assert "85%" in captured.out
        assert "WINNING POSITION" in captured.out
        assert "DECISION RECEIPT" in captured.out
        assert "DR-TEST-abc123" in captured.out
        assert "SUGGESTED NEXT STEPS" in captured.out


# ---------------------------------------------------------------------------
# run_demo
# ---------------------------------------------------------------------------


class TestRunDemo:
    def test_unknown_demo_prints_error(self, capsys):
        result = run_demo("nonexistent_demo_xyz")
        assert result is None
        captured = capsys.readouterr()
        assert "Unknown demo" in captured.out

    def test_unknown_demo_shows_available(self, capsys):
        run_demo("nonexistent")
        captured = capsys.readouterr()
        for demo_name in DEMO_TASKS:
            if demo_name in captured.out:
                return
        pytest.fail("Available demos not shown")

    @patch("aragora.cli.demo._run_demo_debate")
    def test_valid_demo_runs(self, mock_run):
        from aragora_debate.types import DebateResult

        mock_result = DebateResult(task="test", confidence=0.8, consensus_reached=True)
        mock_run.return_value = (mock_result, 0.1)

        with patch("aragora.cli.demo.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = (mock_result, 0.1)
            result = run_demo("microservices")

        # The function was called (either via asyncio.run or directly)
        assert result is not None or mock_run.called or mock_asyncio.run.called


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_list_flag(self, capsys):
        args = argparse.Namespace(
            list_demos=True, server=False, topic=None, name=None,
        )
        main(args)
        captured = capsys.readouterr()
        assert "Available demos:" in captured.out
        assert "microservices" in captured.out

    @patch("aragora.cli.demo._run_server_demo")
    def test_server_flag(self, mock_server):
        args = argparse.Namespace(
            list_demos=False, server=True, topic=None, name=None,
        )
        main(args)
        mock_server.assert_called_once()

    @patch("aragora.cli.demo.asyncio")
    def test_custom_topic(self, mock_asyncio):
        from aragora_debate.types import DebateResult

        mock_result = DebateResult(task="test", confidence=0.8)
        mock_asyncio.run.return_value = (mock_result, 0.1)

        args = argparse.Namespace(
            list_demos=False, server=False,
            topic="Should we use Rust?", name=None,
        )
        main(args)
        mock_asyncio.run.assert_called_once()

    @patch("aragora.cli.demo.run_demo")
    def test_named_demo(self, mock_run_demo):
        args = argparse.Namespace(
            list_demos=False, server=False, topic=None, name="auth",
        )
        main(args)
        mock_run_demo.assert_called_once_with("auth")

    @patch("aragora.cli.demo.run_demo")
    def test_defaults_to_microservices(self, mock_run_demo):
        args = argparse.Namespace(
            list_demos=False, server=False, topic=None, name=None,
        )
        main(args)
        mock_run_demo.assert_called_once_with(_DEFAULT_DEMO)


# ---------------------------------------------------------------------------
# Integration test: full demo runs end-to-end
# ---------------------------------------------------------------------------


class TestDemoIntegration:
    """Run the actual demo and verify it produces output."""

    def test_full_demo_runs_without_error(self, capsys):
        """The demo runs end-to-end with mock agents and produces output."""
        import asyncio
        from aragora.cli.demo import _run_demo_debate

        result, elapsed = asyncio.run(
            _run_demo_debate("Should we adopt microservices?")
        )

        assert result is not None
        assert result.consensus is not None
        assert result.rounds_used == 2
        assert result.receipt is not None
        assert elapsed < 5.0  # Should complete in under 5 seconds

        captured = capsys.readouterr()
        assert "ARAGORA DEMO" in captured.out
        assert "DECISION SUMMARY" in captured.out
        assert "WINNING POSITION" in captured.out
        assert "DECISION RECEIPT" in captured.out

    def test_demo_has_proposals_and_votes(self):
        """The demo produces proposals, critiques, and votes."""
        import asyncio
        from aragora.cli.demo import _run_demo_debate

        result, _ = asyncio.run(
            _run_demo_debate("Design a cache system")
        )

        assert len(result.proposals) == 4  # 4 agents
        assert len(result.votes) > 0
        assert len(result.critiques) > 0
        assert len(result.messages) > 0

    def test_demo_produces_receipt(self):
        """The demo generates a decision receipt with integrity hash."""
        import asyncio
        from aragora.cli.demo import _run_demo_debate

        result, _ = asyncio.run(
            _run_demo_debate("Should we use Kubernetes?")
        )

        assert result.receipt is not None
        assert result.receipt.receipt_id.startswith("DR-")
        assert result.receipt.signature is not None
        assert result.receipt.signature_algorithm == "SHA-256-content-hash"
