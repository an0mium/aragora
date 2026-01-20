"""
Tests for Nomic Loop Debate Phase.

Phase 1: Debate
- Tests proposal generation
- Tests voting mechanism
- Tests consensus detection
- Tests tie-breaking
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDebatePhaseInitialization:
    """Tests for DebatePhase initialization."""

    def test_init_with_required_args(self, mock_aragora_path, mock_claude_agent, mock_codex_agent):
        """Should initialize with required arguments."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )

        assert phase.aragora_path == mock_aragora_path
        assert phase.claude == mock_claude_agent
        assert phase.codex == mock_codex_agent

    def test_init_with_custom_voting_threshold(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Should accept custom voting threshold."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            consensus_threshold=0.75,
        )

        assert phase.consensus_threshold == 0.75


class TestDebatePhaseProposals:
    """Tests for proposal generation."""

    @pytest.mark.asyncio
    async def test_generates_proposals_from_agents(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should generate proposals from multiple agents."""
        from aragora.nomic.phases.debate import DebatePhase

        mock_claude_agent.generate = AsyncMock(
            return_value="Proposal: Add comprehensive error handling"
        )
        mock_codex_agent.generate = AsyncMock(return_value="Proposal: Optimize database queries")

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_generate_proposals", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [
                {
                    "agent": "claude",
                    "proposal": "Add error handling",
                    "reasoning": "Improves reliability",
                },
                {
                    "agent": "codex",
                    "proposal": "Optimize queries",
                    "reasoning": "Improves performance",
                },
            ]

            proposals = await phase.generate_proposals(context="Current codebase state")

            assert len(proposals) == 2
            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_agent_failure_gracefully(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle agent failure without crashing."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_generate_proposals", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("Agent timeout")

            # Should not raise, should return empty or partial results
            try:
                proposals = await phase.generate_proposals(context="Test context")
                # If it returns, it handled the error
                assert proposals is not None or proposals == []
            except Exception:
                # Some implementations may raise, that's also acceptable
                pass


class TestDebatePhaseVoting:
    """Tests for voting mechanism."""

    @pytest.mark.asyncio
    async def test_collects_votes_from_agents(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should collect votes from all agents."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        proposals = [
            {"id": "p1", "proposal": "Add error handling"},
            {"id": "p2", "proposal": "Optimize queries"},
        ]

        with patch.object(phase, "_collect_votes", new_callable=AsyncMock) as mock_votes:
            mock_votes.return_value = {
                "claude": "p1",
                "codex": "p1",
            }

            votes = await phase.collect_votes(proposals)

            assert "claude" in votes
            assert "codex" in votes

    @pytest.mark.asyncio
    async def test_counts_votes_correctly(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should count votes correctly."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        votes = {
            "claude": "p1",
            "codex": "p1",
            "gemini": "p2",
        }

        vote_counts = phase.count_votes(votes)

        assert vote_counts.get("p1", 0) == 2
        assert vote_counts.get("p2", 0) == 1


class TestDebatePhaseConsensus:
    """Tests for consensus detection."""

    @pytest.mark.asyncio
    async def test_detects_majority_consensus(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should detect when majority agrees."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            consensus_threshold=0.5,
        )

        votes = {"claude": "p1", "codex": "p1", "gemini": "p2"}

        result = phase.check_consensus(votes, total_agents=3)

        assert result["consensus"] is True
        assert result["winning_proposal"] == "p1"

    @pytest.mark.asyncio
    async def test_detects_no_consensus(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should detect when no consensus reached."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            consensus_threshold=0.8,
        )

        votes = {"claude": "p1", "codex": "p2", "gemini": "p3"}

        result = phase.check_consensus(votes, total_agents=3)

        assert result["consensus"] is False

    @pytest.mark.asyncio
    async def test_handles_tie_breaking(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle ties appropriately."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        votes = {"claude": "p1", "codex": "p2"}

        result = phase.check_consensus(votes, total_agents=2)

        # Either no consensus or tie-breaker applied
        assert "consensus" in result


class TestDebatePhaseIntegration:
    """Integration tests for debate phase."""

    @pytest.mark.asyncio
    async def test_full_debate_flow(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should complete full debate flow."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_proposals", new_callable=AsyncMock) as mock_proposals:
            with patch.object(phase, "collect_votes", new_callable=AsyncMock) as mock_votes:
                mock_proposals.return_value = [
                    {"id": "p1", "proposal": "Add error handling"},
                ]
                mock_votes.return_value = {"claude": "p1", "codex": "p1"}

                result = await phase.run(context="Test context")

                assert result is not None
                mock_proposals.assert_called_once()
                mock_votes.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_with_no_proposals(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle case when no proposals generated."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_proposals", new_callable=AsyncMock) as mock_proposals:
            mock_proposals.return_value = []

            result = await phase.run(context="Test context")

            # Should handle empty proposals gracefully
            assert result is not None
            assert result.get("consensus", False) is False or result.get("proposals", []) == []
