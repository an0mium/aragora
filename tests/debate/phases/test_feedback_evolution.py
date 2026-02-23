"""
Tests for the EvolutionFeedback module.

Tests cover:
- EvolutionFeedback initialization with various parameters
- update_genome_fitness: full flow with win/loss/elo adjustments
- _check_agent_prediction: matching votes, non-matching, missing, edge cases
- _compute_accepted_critiques: accepted/rejected critiques, missing attrs
- _compute_elo_fitness_adjustment: normal, no elo_system, errors
- maybe_evolve_population: threshold, 5-debate trigger, fire-and-forget
- _evolve_async: evolution + event emission
- record_evolution_patterns: confidence threshold, pattern extraction, performance
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from aragora.debate.phases.feedback_evolution import EvolutionFeedback


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockVote:
    """Mock vote."""

    agent: str
    choice: str
    confidence: float = 0.8


@dataclass
class MockCritique:
    """Mock critique."""

    agent: str
    target_agent: str | None = None
    target: str | None = None


@dataclass
class MockMessage:
    """Mock message."""

    agent: str
    content: str
    role: str = "proposer"
    severity: float = 0.5
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


@dataclass
class MockResult:
    """Mock debate result."""

    winner: str | None = "claude"
    confidence: float = 0.85
    consensus_reached: bool = True
    final_answer: str = "The best approach is X"
    votes: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    messages: list = field(default_factory=list)


@dataclass
class MockAgent:
    """Mock agent."""

    name: str
    genome_id: str | None = None
    prompt_version: int | None = None


@dataclass
class MockDebateContext:
    """Mock debate context."""

    result: MockResult | None = field(default_factory=MockResult)
    agents: list = field(default_factory=list)
    debate_id: str = "test-debate-123"
    domain: str = "testing"
    choice_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.agents:
            self.agents = [
                MockAgent("claude", genome_id="genome-aaa"),
                MockAgent("gpt4", genome_id="genome-bbb"),
            ]


@dataclass
class MockPopulation:
    """Mock population returned by population manager."""

    id: str = "pop-001"
    debate_history: list = field(default_factory=list)


@dataclass
class MockEvolvedPopulation:
    """Mock evolved population."""

    generation: int = 2
    genomes: list = field(default_factory=list)
    top_fitness: float = 0.95


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def population_manager():
    mgr = MagicMock()
    mgr.update_fitness = MagicMock()
    mgr.get_or_create_population = MagicMock(return_value=MockPopulation())
    mgr.evolve_population = MagicMock(return_value=MockEvolvedPopulation(genomes=["g1", "g2"]))
    return mgr


@pytest.fixture
def prompt_evolver():
    evolver = MagicMock()
    evolver.extract_winning_patterns = MagicMock(return_value=["pattern1"])
    evolver.store_patterns = MagicMock()
    evolver.update_performance = MagicMock()
    return evolver


@pytest.fixture
def event_emitter():
    return MagicMock()


@pytest.fixture
def elo_system():
    system = MagicMock()
    system.get_rating = MagicMock(return_value=1600.0)
    return system


@pytest.fixture
def feedback(population_manager, prompt_evolver, event_emitter, elo_system):
    return EvolutionFeedback(
        population_manager=population_manager,
        prompt_evolver=prompt_evolver,
        event_emitter=event_emitter,
        elo_system=elo_system,
        loop_id="loop-001",
        auto_evolve=True,
        breeding_threshold=0.8,
        elo_baseline=1500.0,
    )


# ===========================================================================
# Initialization Tests
# ===========================================================================


class TestEvolutionFeedbackInit:
    """Tests for EvolutionFeedback initialization."""

    def test_default_init(self):
        ef = EvolutionFeedback()
        assert ef.population_manager is None
        assert ef.prompt_evolver is None
        assert ef.event_emitter is None
        assert ef.elo_system is None
        assert ef.loop_id is None
        assert ef.auto_evolve is True
        assert ef.breeding_threshold == 0.8
        assert ef.elo_baseline == 1500.0

    def test_custom_init(self, population_manager, elo_system):
        ef = EvolutionFeedback(
            population_manager=population_manager,
            elo_system=elo_system,
            loop_id="custom-loop",
            auto_evolve=False,
            breeding_threshold=0.9,
            elo_baseline=1200.0,
        )
        assert ef.population_manager is population_manager
        assert ef.elo_system is elo_system
        assert ef.loop_id == "custom-loop"
        assert ef.auto_evolve is False
        assert ef.breeding_threshold == 0.9
        assert ef.elo_baseline == 1200.0

    def test_class_constants(self):
        assert EvolutionFeedback.WIN_FITNESS_DELTA == 0.10
        assert EvolutionFeedback.LOSS_FITNESS_DELTA == -0.05


# ===========================================================================
# _check_agent_prediction Tests
# ===========================================================================


class TestCheckAgentPrediction:
    """Tests for _check_agent_prediction."""

    def test_matching_vote(self, feedback):
        """Agent's vote matches the winner."""
        agent = MockAgent("claude")
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="claude", choice="claude")],
            ),
        )
        assert feedback._check_agent_prediction(agent, ctx) is True

    def test_non_matching_vote(self, feedback):
        """Agent voted for a different candidate."""
        agent = MockAgent("gpt4")
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="gpt4", choice="gemini")],
            ),
        )
        assert feedback._check_agent_prediction(agent, ctx) is False

    def test_vote_with_choice_mapping(self, feedback):
        """Vote uses a variant that maps to the winner via choice_mapping."""
        agent = MockAgent("gpt4")
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="gpt4", choice="option_a")],
            ),
            choice_mapping={"option_a": "claude"},
        )
        assert feedback._check_agent_prediction(agent, ctx) is True

    def test_vote_with_choice_mapping_no_match(self, feedback):
        """Vote maps via choice_mapping but does not match winner."""
        agent = MockAgent("gpt4")
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="gpt4", choice="option_b")],
            ),
            choice_mapping={"option_b": "gemini"},
        )
        assert feedback._check_agent_prediction(agent, ctx) is False

    def test_missing_vote_for_agent(self, feedback):
        """Agent has no vote in the results."""
        agent = MockAgent("missing_agent")
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="claude", choice="claude")],
            ),
        )
        assert feedback._check_agent_prediction(agent, ctx) is False

    def test_no_winner(self, feedback):
        """No winner in the result."""
        agent = MockAgent("claude")
        ctx = MockDebateContext(
            result=MockResult(
                winner=None,
                votes=[MockVote(agent="claude", choice="claude")],
            ),
        )
        assert feedback._check_agent_prediction(agent, ctx) is False

    def test_no_votes(self, feedback):
        """Empty votes list."""
        agent = MockAgent("claude")
        ctx = MockDebateContext(
            result=MockResult(winner="claude", votes=[]),
        )
        assert feedback._check_agent_prediction(agent, ctx) is False

    def test_no_result(self, feedback):
        """No result on context."""
        agent = MockAgent("claude")
        ctx = MockDebateContext(result=None)
        assert feedback._check_agent_prediction(agent, ctx) is False

    def test_choice_mapping_passthrough(self, feedback):
        """When choice is not in mapping, it passes through unchanged."""
        agent = MockAgent("claude")
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="claude", choice="claude")],
            ),
            choice_mapping={"other": "different"},
        )
        # "claude" not in mapping, so passes through as "claude", which matches winner
        assert feedback._check_agent_prediction(agent, ctx) is True


# ===========================================================================
# _compute_accepted_critiques Tests
# ===========================================================================


class TestComputeAcceptedCritiques:
    """Tests for _compute_accepted_critiques."""

    def test_critique_targeting_non_winner(self, feedback):
        """Critique targeting a non-winner is accepted."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[MockCritique(agent="gpt4", target_agent="gemini")],
            ),
        )
        accepted = feedback._compute_accepted_critiques(ctx)
        assert "gpt4" in accepted

    def test_critique_targeting_winner(self, feedback):
        """Critique targeting the winner is not accepted."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[MockCritique(agent="gpt4", target_agent="claude")],
            ),
        )
        accepted = feedback._compute_accepted_critiques(ctx)
        assert "gpt4" not in accepted
        assert len(accepted) == 0

    def test_critique_uses_target_fallback(self, feedback):
        """Falls back to `target` attr when `target_agent` is None."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[MockCritique(agent="gpt4", target_agent=None, target="gemini")],
            ),
        )
        accepted = feedback._compute_accepted_critiques(ctx)
        assert "gpt4" in accepted

    def test_critique_target_fallback_is_winner(self, feedback):
        """Falls back to `target` attr and target is the winner."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[MockCritique(agent="gpt4", target_agent=None, target="claude")],
            ),
        )
        accepted = feedback._compute_accepted_critiques(ctx)
        assert len(accepted) == 0

    def test_multiple_critiques_mixed(self, feedback):
        """Mix of accepted and non-accepted critiques."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[
                    MockCritique(agent="gpt4", target_agent="gemini"),  # accepted
                    MockCritique(agent="gemini", target_agent="claude"),  # not accepted
                    MockCritique(agent="mistral", target_agent="gpt4"),  # accepted
                ],
            ),
        )
        accepted = feedback._compute_accepted_critiques(ctx)
        assert accepted == {"gpt4", "mistral"}

    def test_no_result(self, feedback):
        """No result returns empty set."""
        ctx = MockDebateContext(result=None)
        assert feedback._compute_accepted_critiques(ctx) == set()

    def test_no_winner(self, feedback):
        """No winner returns empty set."""
        ctx = MockDebateContext(
            result=MockResult(
                winner=None,
                critiques=[MockCritique(agent="gpt4", target_agent="gemini")],
            ),
        )
        assert feedback._compute_accepted_critiques(ctx) == set()

    def test_no_critiques(self, feedback):
        """No critiques attribute returns empty set."""
        result = MockResult(winner="claude")
        result.critiques = None
        ctx = MockDebateContext(result=result)
        assert feedback._compute_accepted_critiques(ctx) == set()

    def test_empty_critiques(self, feedback):
        """Empty critiques list returns empty set."""
        ctx = MockDebateContext(
            result=MockResult(winner="claude", critiques=[]),
        )
        assert feedback._compute_accepted_critiques(ctx) == set()

    def test_critique_missing_critic_name(self, feedback):
        """Critique with no agent name is skipped."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[MockCritique(agent=None, target_agent="gemini")],
            ),
        )
        assert feedback._compute_accepted_critiques(ctx) == set()

    def test_critique_missing_target(self, feedback):
        """Critique with no target is skipped."""
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                critiques=[MockCritique(agent="gpt4", target_agent=None, target=None)],
            ),
        )
        assert feedback._compute_accepted_critiques(ctx) == set()


# ===========================================================================
# _compute_elo_fitness_adjustment Tests
# ===========================================================================


class TestComputeEloFitnessAdjustment:
    """Tests for _compute_elo_fitness_adjustment."""

    def test_no_elo_system_returns_zero(self):
        """Returns 0.0 when elo_system is not set."""
        ef = EvolutionFeedback(elo_system=None)
        assert ef._compute_elo_fitness_adjustment("claude", "testing") == 0.0

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.05)
    def test_positive_elo_delta(self, mock_fitness, feedback, elo_system):
        """Positive ELO delta produces positive fitness adjustment."""
        elo_system.get_rating.return_value = 1600.0
        result = feedback._compute_elo_fitness_adjustment("claude", "testing")
        assert result == 0.05
        elo_system.get_rating.assert_called_once_with("claude", "testing")
        mock_fitness.assert_called_once_with(100.0)  # 1600 - 1500

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=-0.03)
    def test_negative_elo_delta(self, mock_fitness, feedback, elo_system):
        """Negative ELO delta produces negative fitness adjustment."""
        elo_system.get_rating.return_value = 1400.0
        result = feedback._compute_elo_fitness_adjustment("claude", "testing")
        assert result == -0.03
        mock_fitness.assert_called_once_with(-100.0)  # 1400 - 1500

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_zero_elo_delta(self, mock_fitness, feedback, elo_system):
        """ELO at baseline produces zero adjustment."""
        elo_system.get_rating.return_value = 1500.0
        result = feedback._compute_elo_fitness_adjustment("claude", "testing")
        assert result == 0.0
        mock_fitness.assert_called_once_with(0.0)

    @patch("aragora.genesis.breeding.fitness_from_elo", side_effect=ValueError("bad"))
    def test_error_returns_zero(self, mock_fitness, feedback, elo_system):
        """Returns 0.0 on error from fitness_from_elo."""
        elo_system.get_rating.return_value = 1600.0
        result = feedback._compute_elo_fitness_adjustment("claude", "testing")
        assert result == 0.0

    def test_elo_system_get_rating_error(self, feedback, elo_system):
        """Returns 0.0 when elo_system.get_rating raises."""
        elo_system.get_rating.side_effect = RuntimeError("unavailable")
        result = feedback._compute_elo_fitness_adjustment("claude", "testing")
        assert result == 0.0

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.07)
    def test_domain_none_uses_empty_string(self, mock_fitness, feedback, elo_system):
        """When domain is None, uses empty string for ELO query."""
        elo_system.get_rating.return_value = 1650.0
        feedback._compute_elo_fitness_adjustment("claude", None)
        elo_system.get_rating.assert_called_once_with("claude", "")


# ===========================================================================
# update_genome_fitness Tests
# ===========================================================================


class TestUpdateGenomeFitness:
    """Tests for update_genome_fitness."""

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.02)
    def test_full_flow_winning_agent(self, mock_fitness, feedback, population_manager, elo_system):
        """Winning agent gets rate-based update, win delta, and elo adjustment."""
        elo_system.get_rating.return_value = 1550.0
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="claude", choice="claude")],
                critiques=[],
            ),
            agents=[MockAgent("claude", genome_id="genome-aaa")],
        )
        feedback.update_genome_fitness(ctx)

        # Should call update_fitness 3 times:
        # 1. Rate-based (consensus_win=True, prediction_correct=True, critique_accepted=False)
        # 2. Win fitness_delta = +0.10
        # 3. ELO-based adjustment
        assert population_manager.update_fitness.call_count == 3

        # First call: rate-based
        population_manager.update_fitness.assert_any_call(
            "genome-aaa",
            consensus_win=True,
            critique_accepted=False,
            prediction_correct=True,
        )
        # Second call: win delta
        population_manager.update_fitness.assert_any_call(
            "genome-aaa",
            fitness_delta=0.10,
        )
        # Third call: elo adjustment
        population_manager.update_fitness.assert_any_call(
            "genome-aaa",
            fitness_delta=0.02,
        )

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_full_flow_losing_agent(self, mock_fitness, feedback, population_manager, elo_system):
        """Losing agent gets rate-based update and loss delta."""
        elo_system.get_rating.return_value = 1500.0  # baseline, 0 adjustment
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[MockVote(agent="gpt4", choice="gemini")],
                critiques=[],
            ),
            agents=[MockAgent("gpt4", genome_id="genome-bbb")],
        )
        feedback.update_genome_fitness(ctx)

        # 2 calls: rate-based + loss delta (elo adj is 0.0 so skipped)
        assert population_manager.update_fitness.call_count == 2

        population_manager.update_fitness.assert_any_call(
            "genome-bbb",
            consensus_win=False,
            critique_accepted=False,
            prediction_correct=False,
        )
        population_manager.update_fitness.assert_any_call(
            "genome-bbb",
            fitness_delta=-0.05,
        )

    def test_no_population_manager(self):
        """Returns early when population_manager is None."""
        ef = EvolutionFeedback(population_manager=None)
        ctx = MockDebateContext()
        # Should not raise
        ef.update_genome_fitness(ctx)

    def test_no_result(self, feedback):
        """Returns early when result is None."""
        ctx = MockDebateContext(result=None)
        feedback.update_genome_fitness(ctx)
        feedback.population_manager.update_fitness.assert_not_called()

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_skip_agent_without_genome_id(self, mock_fitness, feedback, population_manager):
        """Agents without genome_id are skipped."""
        ctx = MockDebateContext(
            result=MockResult(winner="claude"),
            agents=[
                MockAgent("claude", genome_id=None),
                MockAgent("gpt4", genome_id="genome-bbb"),
            ],
        )
        feedback.update_genome_fitness(ctx)

        # Only gpt4's genome should be updated
        for c in population_manager.update_fitness.call_args_list:
            assert c[0][0] == "genome-bbb"

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_no_winner_skips_outcome_delta(
        self, mock_fitness, feedback, population_manager, elo_system
    ):
        """When winner is None, outcome_delta call is skipped."""
        elo_system.get_rating.return_value = 1500.0
        ctx = MockDebateContext(
            result=MockResult(winner=None, votes=[]),
            agents=[MockAgent("claude", genome_id="genome-aaa")],
        )
        feedback.update_genome_fitness(ctx)

        # Only rate-based call (no winner → outcome_delta not applied, elo=0 skipped)
        assert population_manager.update_fitness.call_count == 1
        population_manager.update_fitness.assert_called_once_with(
            "genome-aaa",
            consensus_win=False,
            critique_accepted=False,
            prediction_correct=False,
        )

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_accepted_critique_signal(self, mock_fitness, feedback, population_manager, elo_system):
        """Agent with accepted critique gets critique_accepted=True."""
        elo_system.get_rating.return_value = 1500.0
        ctx = MockDebateContext(
            result=MockResult(
                winner="claude",
                votes=[],
                critiques=[MockCritique(agent="gpt4", target_agent="gemini")],
            ),
            agents=[MockAgent("gpt4", genome_id="genome-bbb")],
        )
        feedback.update_genome_fitness(ctx)

        population_manager.update_fitness.assert_any_call(
            "genome-bbb",
            consensus_win=False,
            critique_accepted=True,
            prediction_correct=False,
        )

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_exception_in_agent_loop_is_caught(
        self, mock_fitness, feedback, population_manager, elo_system
    ):
        """Exceptions per agent are caught, other agents still processed."""
        elo_system.get_rating.return_value = 1500.0
        # Make first call raise, second succeed
        call_count = [0]
        original_update = population_manager.update_fitness

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TypeError("test error")
            return original_update(*args, **kwargs)

        population_manager.update_fitness.side_effect = side_effect

        ctx = MockDebateContext(
            result=MockResult(winner="claude"),
            agents=[
                MockAgent("claude", genome_id="genome-aaa"),
                MockAgent("gpt4", genome_id="genome-bbb"),
            ],
        )
        # Should not raise
        feedback.update_genome_fitness(ctx)

    @patch("aragora.genesis.breeding.fitness_from_elo", return_value=0.0)
    def test_multiple_agents_all_processed(
        self, mock_fitness, feedback, population_manager, elo_system
    ):
        """All agents with genome_id are processed."""
        elo_system.get_rating.return_value = 1500.0
        ctx = MockDebateContext(
            result=MockResult(winner="claude", votes=[], critiques=[]),
            agents=[
                MockAgent("claude", genome_id="genome-aaa"),
                MockAgent("gpt4", genome_id="genome-bbb"),
                MockAgent("gemini", genome_id="genome-ccc"),
            ],
        )
        feedback.update_genome_fitness(ctx)

        # Each agent: rate-based + loss delta (winner gets win delta)
        # All get 2 calls each (elo=0 skipped), total = 6
        genome_ids_updated = [c[0][0] for c in population_manager.update_fitness.call_args_list]
        assert "genome-aaa" in genome_ids_updated
        assert "genome-bbb" in genome_ids_updated
        assert "genome-ccc" in genome_ids_updated


# ===========================================================================
# maybe_evolve_population Tests
# ===========================================================================


class TestMaybeEvolvePopulation:
    """Tests for maybe_evolve_population."""

    @pytest.mark.asyncio
    async def test_no_population_manager(self):
        """Returns early when population_manager is None."""
        ef = EvolutionFeedback(population_manager=None, auto_evolve=True)
        ctx = MockDebateContext()
        await ef.maybe_evolve_population(ctx)
        # No error

    @pytest.mark.asyncio
    async def test_auto_evolve_disabled(self, feedback):
        """Returns early when auto_evolve is False."""
        feedback.auto_evolve = False
        ctx = MockDebateContext()
        await feedback.maybe_evolve_population(ctx)
        feedback.population_manager.get_or_create_population.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_result(self, feedback):
        """Returns early when result is None."""
        ctx = MockDebateContext(result=None)
        await feedback.maybe_evolve_population(ctx)
        feedback.population_manager.get_or_create_population.assert_not_called()

    @pytest.mark.asyncio
    async def test_below_confidence_threshold(self, feedback):
        """Returns early when confidence < breeding_threshold."""
        ctx = MockDebateContext(
            result=MockResult(confidence=0.5),
        )
        await feedback.maybe_evolve_population(ctx)
        feedback.population_manager.get_or_create_population.assert_not_called()

    @pytest.mark.asyncio
    async def test_at_confidence_threshold(self, feedback, population_manager):
        """Proceeds when confidence equals breeding_threshold."""
        pop = MockPopulation(debate_history=["d1"])  # non-empty so `or` keeps it
        population_manager.get_or_create_population.return_value = pop
        ctx = MockDebateContext(
            result=MockResult(confidence=0.8),
        )
        await feedback.maybe_evolve_population(ctx)
        population_manager.get_or_create_population.assert_called_once()
        # debate_history now has 2 entries (was 1, appended 1), not a multiple of 5
        assert len(pop.debate_history) == 2

    @pytest.mark.asyncio
    async def test_no_population_returned(self, feedback, population_manager):
        """Returns early when get_or_create_population returns None."""
        population_manager.get_or_create_population.return_value = None
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9),
        )
        await feedback.maybe_evolve_population(ctx)
        population_manager.evolve_population.assert_not_called()

    @pytest.mark.asyncio
    async def test_evolution_triggered_at_5_debates(self, feedback, population_manager):
        """Evolution is triggered when debate_history reaches a multiple of 5."""
        pop = MockPopulation(debate_history=["d1", "d2", "d3", "d4"])
        population_manager.get_or_create_population.return_value = pop
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9),
        )
        await feedback.maybe_evolve_population(ctx)
        # Now 5 debates in history -> trigger evolution
        assert len(pop.debate_history) == 5
        # Give the fire-and-forget task time to run
        await asyncio.sleep(0.05)
        population_manager.evolve_population.assert_called_once_with(pop)

    @pytest.mark.asyncio
    async def test_no_evolution_at_non_multiple_of_5(self, feedback, population_manager):
        """No evolution when debate count is not a multiple of 5."""
        pop = MockPopulation(debate_history=["d1", "d2"])
        population_manager.get_or_create_population.return_value = pop
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9),
        )
        await feedback.maybe_evolve_population(ctx)
        assert len(pop.debate_history) == 3
        await asyncio.sleep(0.05)
        population_manager.evolve_population.assert_not_called()

    @pytest.mark.asyncio
    async def test_evolution_at_10_debates(self, feedback, population_manager):
        """Evolution triggered again at 10 debates."""
        pop = MockPopulation(debate_history=list(range(9)))
        population_manager.get_or_create_population.return_value = pop
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9),
        )
        await feedback.maybe_evolve_population(ctx)
        assert len(pop.debate_history) == 10
        await asyncio.sleep(0.05)
        population_manager.evolve_population.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_in_evolution_caught(self, feedback, population_manager):
        """Exception during evolution check is caught gracefully."""
        population_manager.get_or_create_population.side_effect = RuntimeError("fail")
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9),
        )
        # Should not raise
        await feedback.maybe_evolve_population(ctx)

    @pytest.mark.asyncio
    async def test_population_with_none_debate_history(self, feedback, population_manager):
        """Handles population with debate_history=None."""
        pop = MockPopulation()
        pop.debate_history = None
        population_manager.get_or_create_population.return_value = pop
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9),
        )
        await feedback.maybe_evolve_population(ctx)
        # history was None -> becomes [] -> append -> [debate_id]
        # len=1, not multiple of 5, no evolution


# ===========================================================================
# _evolve_async Tests
# ===========================================================================


class TestEvolveAsync:
    """Tests for _evolve_async."""

    @pytest.mark.asyncio
    async def test_evolve_and_emit_event(self, feedback, population_manager, event_emitter):
        """Evolves population and emits GENESIS_EVOLUTION event."""
        pop = MockPopulation(id="pop-001")
        evolved = MockEvolvedPopulation(generation=3, genomes=["g1", "g2", "g3"], top_fitness=0.92)
        population_manager.evolve_population.return_value = evolved

        await feedback._evolve_async(pop)

        population_manager.evolve_population.assert_called_once_with(pop)
        event_emitter.emit.assert_called_once()
        emitted = event_emitter.emit.call_args[0][0]
        assert emitted.type.value == "genesis_evolution"
        assert emitted.data["generation"] == 3
        assert emitted.data["genome_count"] == 3
        assert emitted.data["population_id"] == "pop-001"
        assert emitted.data["top_fitness"] == 0.92
        assert emitted.loop_id == "loop-001"

    @pytest.mark.asyncio
    async def test_evolve_without_event_emitter(self, population_manager):
        """Evolution works without event emitter."""
        ef = EvolutionFeedback(
            population_manager=population_manager,
            event_emitter=None,
        )
        pop = MockPopulation()
        evolved = MockEvolvedPopulation(generation=2, genomes=["g1"])
        population_manager.evolve_population.return_value = evolved

        await ef._evolve_async(pop)
        population_manager.evolve_population.assert_called_once_with(pop)

    @pytest.mark.asyncio
    async def test_evolve_exception_caught(self, feedback, population_manager):
        """Exception during evolution is caught."""
        population_manager.evolve_population.side_effect = RuntimeError("evolution failed")
        pop = MockPopulation()
        # Should not raise
        await feedback._evolve_async(pop)


# ===========================================================================
# record_evolution_patterns Tests
# ===========================================================================


class TestRecordEvolutionPatterns:
    """Tests for record_evolution_patterns."""

    def test_no_prompt_evolver(self):
        """Returns early when prompt_evolver is None."""
        ef = EvolutionFeedback(prompt_evolver=None)
        ctx = MockDebateContext()
        ef.record_evolution_patterns(ctx)
        # No error

    def test_no_result(self, feedback):
        """Returns early when result is None."""
        ctx = MockDebateContext(result=None)
        feedback.record_evolution_patterns(ctx)
        feedback.prompt_evolver.extract_winning_patterns.assert_not_called()

    def test_low_confidence_skipped(self, feedback):
        """Skipped when confidence < 0.7."""
        ctx = MockDebateContext(
            result=MockResult(confidence=0.69),
        )
        feedback.record_evolution_patterns(ctx)
        feedback.prompt_evolver.extract_winning_patterns.assert_not_called()

    def test_at_confidence_threshold(self, feedback, prompt_evolver):
        """Runs when confidence is exactly 0.7."""
        ctx = MockDebateContext(
            result=MockResult(
                confidence=0.7,
                consensus_reached=True,
                final_answer="answer",
                messages=[],
            ),
            agents=[],
        )
        feedback.record_evolution_patterns(ctx)
        prompt_evolver.extract_winning_patterns.assert_called_once()

    def test_patterns_extracted_and_stored(self, feedback, prompt_evolver):
        """Patterns are extracted and stored."""
        prompt_evolver.extract_winning_patterns.return_value = ["p1", "p2"]
        ctx = MockDebateContext(
            result=MockResult(confidence=0.85, messages=[]),
            agents=[],
        )
        feedback.record_evolution_patterns(ctx)
        prompt_evolver.extract_winning_patterns.assert_called_once()
        prompt_evolver.store_patterns.assert_called_once_with(["p1", "p2"])

    def test_no_patterns_not_stored(self, feedback, prompt_evolver):
        """store_patterns not called when no patterns extracted."""
        prompt_evolver.extract_winning_patterns.return_value = []
        ctx = MockDebateContext(
            result=MockResult(confidence=0.85, messages=[]),
            agents=[],
        )
        feedback.record_evolution_patterns(ctx)
        prompt_evolver.store_patterns.assert_not_called()

    def test_performance_updated_for_agents_with_prompt_version(self, feedback, prompt_evolver):
        """Performance is updated for agents with prompt_version."""
        prompt_evolver.extract_winning_patterns.return_value = ["p1"]
        agents = [
            MockAgent("claude", prompt_version=3),
            MockAgent("gpt4", prompt_version=1),
        ]
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9, messages=[]),
            agents=agents,
        )
        feedback.record_evolution_patterns(ctx)

        assert prompt_evolver.update_performance.call_count == 2
        call_args_list = prompt_evolver.update_performance.call_args_list
        assert call_args_list[0] == call(
            agent_name="claude", version=3, debate_result=call_args_list[0].kwargs["debate_result"]
        )
        assert call_args_list[1] == call(
            agent_name="gpt4", version=1, debate_result=call_args_list[1].kwargs["debate_result"]
        )

    def test_agents_without_prompt_version_skipped(self, feedback, prompt_evolver):
        """Agents without prompt_version are skipped."""
        prompt_evolver.extract_winning_patterns.return_value = ["p1"]
        agents = [
            MockAgent("claude", prompt_version=None),
            MockAgent("gpt4", prompt_version=2),
        ]
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9, messages=[]),
            agents=agents,
        )
        feedback.record_evolution_patterns(ctx)

        assert prompt_evolver.update_performance.call_count == 1
        prompt_evolver.update_performance.assert_called_once()
        assert prompt_evolver.update_performance.call_args.kwargs["agent_name"] == "gpt4"

    def test_critic_messages_become_critique_proxies(self, feedback, prompt_evolver):
        """Messages with role='critic' are converted to CritiqueProxy objects."""
        critic_msg = MockMessage(
            agent="gpt4",
            content="This is flawed",
            role="critic",
            severity=0.8,
            issues=["issue1"],
            suggestions=["fix1"],
        )
        prompt_evolver.extract_winning_patterns.return_value = []
        ctx = MockDebateContext(
            result=MockResult(
                confidence=0.9,
                messages=[critic_msg, MockMessage(agent="claude", content="ok", role="proposer")],
            ),
            agents=[],
        )
        feedback.record_evolution_patterns(ctx)

        # Verify the proxy was created with critic message
        proxy = prompt_evolver.extract_winning_patterns.call_args[0][0][0]
        assert len(proxy.critiques) == 1
        assert proxy.critiques[0].severity == 0.8
        assert proxy.critiques[0].issues == ["issue1"]
        assert proxy.critiques[0].suggestions == ["fix1"]

    def test_debate_result_proxy_attributes(self, feedback, prompt_evolver):
        """DebateResultProxy has correct attributes from context."""
        prompt_evolver.extract_winning_patterns.return_value = []
        ctx = MockDebateContext(
            result=MockResult(
                confidence=0.92,
                consensus_reached=True,
                final_answer="The answer is 42",
                messages=[],
            ),
            debate_id="debate-xyz",
            agents=[],
        )
        feedback.record_evolution_patterns(ctx)

        proxy = prompt_evolver.extract_winning_patterns.call_args[0][0][0]
        assert proxy.id == "debate-xyz"
        assert proxy.consensus_reached is True
        assert proxy.confidence == 0.92
        assert proxy.final_answer == "The answer is 42"
        assert proxy.critiques == []

    def test_exception_caught(self, feedback, prompt_evolver):
        """Exceptions during pattern extraction are caught."""
        prompt_evolver.extract_winning_patterns.side_effect = RuntimeError("fail")
        ctx = MockDebateContext(
            result=MockResult(confidence=0.9, messages=[]),
            agents=[],
        )
        # Should not raise
        feedback.record_evolution_patterns(ctx)

    def test_final_answer_none_becomes_empty_string(self, feedback, prompt_evolver):
        """final_answer=None is converted to empty string in proxy."""
        prompt_evolver.extract_winning_patterns.return_value = []
        ctx = MockDebateContext(
            result=MockResult(
                confidence=0.9,
                final_answer=None,
                messages=[],
            ),
            agents=[],
        )
        feedback.record_evolution_patterns(ctx)
        proxy = prompt_evolver.extract_winning_patterns.call_args[0][0][0]
        assert proxy.final_answer == ""
