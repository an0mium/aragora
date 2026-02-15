"""
Tests for Genesis + ELO Feedback Integration.

Covers:
- fitness_from_elo() conversion function in breeding.py
- EvolutionFeedback ELO integration: elo_system parameter, _compute_elo_fitness_adjustment
- update_genome_fitness() applying ELO-based fitness delta
- PopulationManager.update_fitness() with fitness_delta parameter
- Graceful degradation when ELO system is unavailable
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.genesis.breeding import fitness_from_elo


# ---------------------------------------------------------------------------
# fitness_from_elo() unit tests
# ---------------------------------------------------------------------------


class TestFitnessFromElo:
    """Test the ELO delta -> fitness adjustment conversion."""

    def test_zero_delta_returns_zero(self):
        assert fitness_from_elo(0.0) == 0.0

    def test_positive_delta_returns_positive(self):
        adj = fitness_from_elo(200.0)
        assert adj > 0.0
        assert adj <= 0.2

    def test_negative_delta_returns_negative(self):
        adj = fitness_from_elo(-200.0)
        assert adj < 0.0
        assert adj >= -0.2

    def test_large_positive_approaches_max(self):
        adj = fitness_from_elo(1000.0)
        assert adj > 0.15
        assert adj <= 0.2

    def test_large_negative_approaches_min(self):
        adj = fitness_from_elo(-1000.0)
        assert adj < -0.15
        assert adj >= -0.2

    def test_symmetry(self):
        pos = fitness_from_elo(300.0)
        neg = fitness_from_elo(-300.0)
        assert abs(pos + neg) < 1e-6

    def test_custom_scale(self):
        # Smaller scale means 200 delta maps closer to max
        adj_small = fitness_from_elo(200.0, scale=200.0)
        adj_large = fitness_from_elo(200.0, scale=800.0)
        assert adj_small > adj_large


# ---------------------------------------------------------------------------
# EvolutionFeedback ELO integration tests
# ---------------------------------------------------------------------------


class TestEvolutionFeedbackEloInit:
    """Test EvolutionFeedback constructor with elo_system parameter."""

    def test_default_no_elo_system(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        fb = EvolutionFeedback()
        assert fb.elo_system is None
        assert fb.elo_baseline == 1500.0

    def test_elo_system_stored(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_elo = MagicMock()
        fb = EvolutionFeedback(elo_system=mock_elo, elo_baseline=1600.0)
        assert fb.elo_system is mock_elo
        assert fb.elo_baseline == 1600.0


class TestComputeEloFitnessAdjustment:
    """Test _compute_elo_fitness_adjustment method."""

    def test_no_elo_system_returns_zero(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        fb = EvolutionFeedback(elo_system=None)
        assert fb._compute_elo_fitness_adjustment("claude") == 0.0

    def test_above_baseline_positive(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1700.0
        fb = EvolutionFeedback(elo_system=mock_elo, elo_baseline=1500.0)

        adj = fb._compute_elo_fitness_adjustment("claude", "general")
        assert adj > 0.0
        mock_elo.get_rating.assert_called_once_with("claude", "general")

    def test_below_baseline_negative(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1300.0
        fb = EvolutionFeedback(elo_system=mock_elo, elo_baseline=1500.0)

        adj = fb._compute_elo_fitness_adjustment("gpt4")
        assert adj < 0.0

    def test_at_baseline_returns_zero(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500.0
        fb = EvolutionFeedback(elo_system=mock_elo)

        adj = fb._compute_elo_fitness_adjustment("agent-x")
        assert adj == 0.0

    def test_elo_error_returns_zero(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = RuntimeError("ELO unavailable")
        fb = EvolutionFeedback(elo_system=mock_elo)

        adj = fb._compute_elo_fitness_adjustment("claude")
        assert adj == 0.0

    def test_none_domain_becomes_empty_string(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500.0
        fb = EvolutionFeedback(elo_system=mock_elo)

        fb._compute_elo_fitness_adjustment("claude", domain=None)
        mock_elo.get_rating.assert_called_once_with("claude", "")


# ---------------------------------------------------------------------------
# update_genome_fitness with ELO delta
# ---------------------------------------------------------------------------


def _make_ctx(agents=None, winner="claude", domain="general"):
    """Create a mock DebateContext."""
    ctx = MagicMock()
    ctx.domain = domain
    ctx.debate_id = "d-test-001"

    result = MagicMock()
    result.winner = winner
    result.confidence = 0.9
    result.consensus_reached = True
    result.votes = []
    ctx.result = result

    if agents is None:
        agent_a = MagicMock()
        agent_a.name = "claude"
        agent_a.genome_id = "genome-aaa"

        agent_b = MagicMock()
        agent_b.name = "gpt4"
        agent_b.genome_id = "genome-bbb"

        agents = [agent_a, agent_b]

    ctx.agents = agents
    ctx.choice_mapping = {}
    return ctx


class TestUpdateGenomeFitnessWithElo:
    """Test that update_genome_fitness applies ELO-based adjustments."""

    def test_elo_adjustment_applied(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_pm = MagicMock()
        mock_elo = MagicMock()
        # claude at 1700 (above baseline), gpt4 at 1300 (below)
        mock_elo.get_rating.side_effect = lambda name, domain="": (
            1700.0 if name == "claude" else 1300.0
        )

        fb = EvolutionFeedback(
            population_manager=mock_pm, elo_system=mock_elo
        )
        ctx = _make_ctx()
        fb.update_genome_fitness(ctx)

        # Should have called update_fitness twice per agent:
        # once for consensus/prediction, once for ELO delta
        calls = mock_pm.update_fitness.call_args_list
        assert len(calls) == 4  # 2 agents x 2 calls each

        # First call for claude: consensus_win + prediction
        assert calls[0].args[0] == "genome-aaa"
        assert calls[0].kwargs.get("consensus_win") is True

        # Second call for claude: ELO fitness_delta (positive)
        assert calls[1].args[0] == "genome-aaa"
        assert calls[1].kwargs["fitness_delta"] > 0.0

        # Third call for gpt4: consensus_win + prediction
        assert calls[2].args[0] == "genome-bbb"
        assert calls[2].kwargs.get("consensus_win") is False

        # Fourth call for gpt4: ELO fitness_delta (negative)
        assert calls[3].args[0] == "genome-bbb"
        assert calls[3].kwargs["fitness_delta"] < 0.0

    def test_no_elo_system_skips_adjustment(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_pm = MagicMock()
        fb = EvolutionFeedback(population_manager=mock_pm, elo_system=None)
        ctx = _make_ctx()
        fb.update_genome_fitness(ctx)

        # Only the base update_fitness calls (1 per agent with genome_id)
        calls = mock_pm.update_fitness.call_args_list
        assert len(calls) == 2

    def test_elo_error_does_not_cascade(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_pm = MagicMock()
        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = RuntimeError("db down")

        fb = EvolutionFeedback(
            population_manager=mock_pm, elo_system=mock_elo
        )
        ctx = _make_ctx()
        fb.update_genome_fitness(ctx)

        # Base calls still happen; ELO adjustment returns 0.0, no extra call
        calls = mock_pm.update_fitness.call_args_list
        assert len(calls) == 2  # only base calls

    def test_agents_without_genome_id_skipped(self):
        from aragora.debate.phases.feedback_evolution import EvolutionFeedback

        mock_pm = MagicMock()
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600.0

        agent = MagicMock()
        agent.name = "plain-agent"
        agent.genome_id = None  # No genome

        fb = EvolutionFeedback(
            population_manager=mock_pm, elo_system=mock_elo
        )
        ctx = _make_ctx(agents=[agent])
        fb.update_genome_fitness(ctx)

        mock_pm.update_fitness.assert_not_called()
        mock_elo.get_rating.assert_not_called()


# ---------------------------------------------------------------------------
# PopulationManager.update_fitness with fitness_delta
# ---------------------------------------------------------------------------


class TestPopulationManagerFitnessDelta:
    """Test that PopulationManager.update_fitness handles fitness_delta."""

    def test_fitness_delta_applied(self):
        from unittest.mock import patch

        from aragora.genesis.breeding import PopulationManager

        mock_genome = MagicMock()
        mock_genome.fitness_score = 0.5

        with patch.object(PopulationManager, "__init__", lambda self, **kw: None):
            pm = PopulationManager()
            pm.genome_store = MagicMock()
            pm.genome_store.get.return_value = mock_genome

            pm.update_fitness("genome-xyz", fitness_delta=0.1)

            # fitness_score should be 0.5 + (update_fitness changes) + 0.1
            # But update_fitness() calls genome.update_fitness() first, then applies delta
            mock_genome.update_fitness.assert_called_once()
            # After update_fitness + delta, genome.fitness_score should be clamped
            pm.genome_store.save.assert_called_once_with(mock_genome)

    def test_fitness_delta_clamped_at_one(self):
        from unittest.mock import patch

        from aragora.genesis.breeding import PopulationManager

        mock_genome = MagicMock()
        mock_genome.fitness_score = 0.95

        with patch.object(PopulationManager, "__init__", lambda self, **kw: None):
            pm = PopulationManager()
            pm.genome_store = MagicMock()
            pm.genome_store.get.return_value = mock_genome

            pm.update_fitness("g1", fitness_delta=0.2)

            assert mock_genome.fitness_score == 1.0

    def test_fitness_delta_clamped_at_zero(self):
        from unittest.mock import patch

        from aragora.genesis.breeding import PopulationManager

        mock_genome = MagicMock()
        mock_genome.fitness_score = 0.05

        with patch.object(PopulationManager, "__init__", lambda self, **kw: None):
            pm = PopulationManager()
            pm.genome_store = MagicMock()
            pm.genome_store.get.return_value = mock_genome

            pm.update_fitness("g1", fitness_delta=-0.2)

            assert mock_genome.fitness_score == 0.0

    def test_zero_delta_no_change(self):
        from unittest.mock import patch

        from aragora.genesis.breeding import PopulationManager

        mock_genome = MagicMock()
        mock_genome.fitness_score = 0.6

        with patch.object(PopulationManager, "__init__", lambda self, **kw: None):
            pm = PopulationManager()
            pm.genome_store = MagicMock()
            pm.genome_store.get.return_value = mock_genome

            pm.update_fitness("g1", fitness_delta=0.0)

            # fitness_score should remain at whatever update_fitness set it to
            # (no additional delta applied)
            assert mock_genome.fitness_score == 0.6
