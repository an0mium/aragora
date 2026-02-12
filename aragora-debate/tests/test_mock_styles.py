"""Tests for StyledMockAgent style-based behaviour."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate.styled_mock import StyledMockAgent, PROPOSALS
from aragora_debate.types import Critique, Vote


class TestStyledMockAgentStyles:
    """StyledMockAgent produces style-appropriate responses."""

    @pytest.mark.parametrize("style", ["supportive", "critical", "balanced", "contrarian"])
    def test_generate_returns_nonempty(self, style):
        agent = StyledMockAgent("test", style=style)
        result = asyncio.run(agent.generate("Test prompt"))
        assert isinstance(result, str)
        assert len(result) > 20

    @pytest.mark.parametrize("style", ["supportive", "critical", "balanced", "contrarian"])
    def test_generate_from_templates(self, style):
        agent = StyledMockAgent("test", style=style)
        result = asyncio.run(agent.generate("Test prompt"))
        assert result in PROPOSALS[style]

    def test_proposal_override_bypasses_style(self):
        agent = StyledMockAgent("test", style="critical", proposal="Custom answer")
        result = asyncio.run(agent.generate("Anything"))
        assert result == "Custom answer"

    @pytest.mark.parametrize("style", ["supportive", "critical", "balanced", "contrarian"])
    def test_critique_returns_valid_dataclass(self, style):
        agent = StyledMockAgent("test", style=style)
        crit = asyncio.run(
            agent.critique("Some proposal", "Architecture choice", target_agent="alice")
        )
        assert isinstance(crit, Critique)
        assert crit.agent == "test"
        assert crit.target_agent == "alice"
        assert len(crit.issues) > 0
        assert len(crit.suggestions) > 0
        assert 0.0 <= crit.severity <= 10.0

    def test_critical_severity_higher_than_supportive(self):
        critical = StyledMockAgent("c", style="critical")
        supportive = StyledMockAgent("s", style="supportive")
        crits_c = [
            asyncio.run(critical.critique("x", "t", target_agent="a"))
            for _ in range(10)
        ]
        crits_s = [
            asyncio.run(supportive.critique("x", "t", target_agent="a"))
            for _ in range(10)
        ]
        avg_c = sum(c.severity for c in crits_c) / len(crits_c)
        avg_s = sum(c.severity for c in crits_s) / len(crits_s)
        assert avg_c > avg_s

    def test_critique_issues_override(self):
        agent = StyledMockAgent("test", style="balanced", critique_issues=["Custom issue"])
        crit = asyncio.run(agent.critique("prop", "task", target_agent="x"))
        assert crit.issues == ["Custom issue"]

    @pytest.mark.parametrize("style", ["supportive", "critical", "balanced", "contrarian"])
    def test_vote_returns_valid_dataclass(self, style):
        agent = StyledMockAgent("test", style=style)
        proposals = {"alice": "Plan A", "bob": "Plan B"}
        vote = asyncio.run(agent.vote(proposals, "Pick a plan"))
        assert isinstance(vote, Vote)
        assert vote.agent == "test"
        assert vote.choice in proposals
        assert 0.0 <= vote.confidence <= 1.0
        assert len(vote.reasoning) > 0

    def test_supportive_votes_first(self):
        agent = StyledMockAgent("test", style="supportive")
        proposals = {"alice": "Plan A", "bob": "Plan B"}
        vote = asyncio.run(agent.vote(proposals, "Pick"))
        # Supportive picks the first non-self agent
        assert vote.choice == "alice"

    def test_contrarian_votes_last(self):
        agent = StyledMockAgent("test", style="contrarian")
        proposals = {"alice": "Plan A", "bob": "Plan B"}
        vote = asyncio.run(agent.vote(proposals, "Pick"))
        assert vote.choice == "bob"

    def test_vote_for_override(self):
        agent = StyledMockAgent("test", style="contrarian", vote_for="alice")
        proposals = {"alice": "Plan A", "bob": "Plan B"}
        vote = asyncio.run(agent.vote(proposals, "Pick"))
        assert vote.choice == "alice"

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            StyledMockAgent("test", style="invalid")  # type: ignore[arg-type]

    def test_default_style_is_balanced(self):
        agent = StyledMockAgent("test")
        assert agent.style == "balanced"

    def test_agent_repr(self):
        agent = StyledMockAgent("analyst", style="critical")
        r = repr(agent)
        assert "analyst" in r
        assert "mock" in r


class TestStyledMockIntegration:
    """StyledMockAgent works end-to-end through Arena."""

    def test_full_debate_with_styles(self):
        from aragora_debate.arena import Arena
        from aragora_debate.types import DebateConfig

        agents = [
            StyledMockAgent("opt", style="supportive"),
            StyledMockAgent("crit", style="critical"),
            StyledMockAgent("mod", style="balanced"),
        ]
        arena = Arena(
            question="Test question?",
            agents=agents,
            config=DebateConfig(rounds=2),
        )
        result = asyncio.run(arena.run())
        assert result.rounds_used >= 1
        assert len(result.proposals) == 3
        assert len(result.votes) >= 3
        assert result.receipt is not None


class TestMainModule:
    """The __main__ module runs without error."""

    def test_main_function_exists(self):
        from aragora_debate.__main__ import main
        assert callable(main)

    def test_run_demo_completes(self):
        from aragora_debate.__main__ import _run_demo
        asyncio.run(_run_demo("Test question?", 1))

    def test_cli_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "aragora_debate", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
            timeout=10,
        )
        assert result.returncode == 0
        assert "topic" in result.stdout.lower()

    def test_cli_custom_topic(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "aragora_debate",
                "--topic", "Is Python good?",
                "--rounds", "1",
            ],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
            timeout=30,
        )
        assert result.returncode == 0
        assert "Decision Receipt" in result.stdout
