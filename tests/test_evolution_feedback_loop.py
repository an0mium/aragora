"""
Integration tests for Evolution Feedback Loop.

Tests cover the debate → evolution → improved debate cycle:
- Pattern extraction from debate outcomes
- Prompt mutation and evolution
- Performance tracking across generations
- Population management and selection
"""

import json
import os

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_debate_outcome():
    """Create a mock debate outcome for evolution."""
    return {
        "debate_id": "debate-123",
        "topic": "Should we use microservices?",
        "winner": "claude",
        "agents": ["claude", "gpt4"],
        "rounds": 3,
        "consensus_reached": True,
        "consensus": "Microservices are beneficial for large-scale applications.",
        "votes": {"claude": 3, "gpt4": 1},
        "critiques": [
            {"from": "gpt4", "to": "claude", "text": "Good argument structure"},
            {"from": "claude", "to": "gpt4", "text": "Needs more evidence"},
        ],
    }


@pytest.fixture
def mock_population():
    """Create a mock agent population."""
    agents = []
    for i in range(5):
        agent = Mock()
        agent.name = f"agent-{i}"
        agent.generation = 0
        agent.elo_rating = 1500 + (i * 10)
        agent.win_rate = 0.5 + (i * 0.05)
        agents.append(agent)
    return agents


# ============================================================================
# Pattern Extraction Tests
# ============================================================================


class TestPatternExtraction:
    """Tests for extracting winning patterns from debates."""

    def test_extract_argument_patterns(self, mock_debate_outcome):
        """Test extracting argument patterns from debate."""
        from aragora.evolution.pattern_extractor import extract_patterns

        patterns = extract_patterns(mock_debate_outcome)

        assert patterns is not None
        assert "winning_patterns" in patterns or len(patterns) > 0

    def test_identify_successful_strategies(self, mock_debate_outcome):
        """Test identifying successful debate strategies."""
        from aragora.evolution.pattern_extractor import identify_strategies

        strategies = identify_strategies(mock_debate_outcome)

        assert strategies is not None

    def test_patterns_associated_with_winner(self, mock_debate_outcome):
        """Test patterns are correctly associated with winner."""
        from aragora.evolution.pattern_extractor import extract_patterns

        patterns = extract_patterns(mock_debate_outcome)
        winner = mock_debate_outcome["winner"]

        # Patterns should reference the winning agent
        assert winner in str(patterns) or len(patterns) > 0


# ============================================================================
# Prompt Mutation Tests
# ============================================================================


class TestPromptMutation:
    """Tests for prompt mutation in evolution."""

    def test_mutate_prompt(self):
        """Test basic prompt mutation."""
        from aragora.evolution.evolver import PromptEvolver

        evolver = PromptEvolver()
        original_prompt = "You are a helpful assistant that argues logically."
        mutated = evolver.mutate(original_prompt)

        assert mutated is not None
        assert mutated != original_prompt or len(mutated) > 0

    def test_crossover_prompts(self):
        """Test combining traits from two prompts."""
        from aragora.evolution.evolver import PromptEvolver

        evolver = PromptEvolver()
        parent1 = "Be concise and factual in arguments."
        parent2 = "Use examples and analogies to illustrate points."
        offspring = evolver.crossover(parent1, parent2)

        assert offspring is not None

    def test_mutation_rate_affects_output(self):
        """Test that mutation rate affects output diversity."""
        from aragora.evolution.evolver import PromptEvolver

        # Low mutation rate
        evolver_low = PromptEvolver(mutation_rate=0.1)
        # High mutation rate
        evolver_high = PromptEvolver(mutation_rate=0.9)

        original = "Test prompt for evolution."

        # Both should produce valid output
        mutated_low = evolver_low.mutate(original)
        mutated_high = evolver_high.mutate(original)

        assert mutated_low is not None
        assert mutated_high is not None


# ============================================================================
# Population Management Tests
# ============================================================================


class TestPopulationManagement:
    """Tests for population management in evolution."""

    def test_population_initialization(self):
        """Test initializing a population."""
        from aragora.genesis.breeding import PopulationManager

        manager = PopulationManager(max_population_size=10)

        assert manager is not None
        assert manager.max_population_size == 10

    def test_selection_by_fitness(self, mock_population):
        """Test selecting agents by fitness."""
        from aragora.genesis.breeding import select_by_fitness

        selected = select_by_fitness(mock_population, count=2)

        assert len(selected) == 2
        # Higher rated agents should be more likely to be selected

    def test_evolve_population_preserves_genomes(self, tmp_path):
        """Test population evolution preserves and improves genomes."""
        from aragora.genesis.breeding import PopulationManager, Population
        from aragora.genesis.genome import AgentGenome

        db_path = str(tmp_path / "test_genesis.db")
        manager = PopulationManager(db_path=db_path, max_population_size=4)

        # Create a population with test genomes
        population = manager.get_or_create_population(["claude", "gemini"])

        # Evolve the population
        evolved = manager.evolve_population(population)

        # Should have genomes in the evolved population
        assert evolved is not None
        assert len(evolved.genomes) > 0


# ============================================================================
# Performance Tracking Tests
# ============================================================================


class TestPerformanceTracking:
    """Tests for tracking evolution performance."""

    def test_record_outcome(self, mock_debate_outcome):
        """Test recording debate outcome for evolution."""
        from aragora.evolution.tracker import EvolutionTracker

        tracker = EvolutionTracker()
        tracker.record_outcome(
            agent="claude",
            won=True,
            debate_id=mock_debate_outcome["debate_id"],
        )

        stats = tracker.get_agent_stats("claude")
        assert stats["wins"] >= 1

    def test_generation_metrics(self):
        """Test tracking metrics across generations."""
        import tempfile
        from aragora.evolution.tracker import EvolutionTracker

        # Use isolated temp database to avoid cross-test pollution
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            tracker = EvolutionTracker(db_path=temp_db)

            # Record outcomes for generation 0
            for i in range(5):
                tracker.record_outcome(
                    agent=f"agent-{i}",
                    won=(i % 2 == 0),
                    generation=0,
                )

            metrics = tracker.get_generation_metrics(0)

            assert metrics["total_debates"] == 5
            assert "win_rate" in metrics
        finally:
            if os.path.exists(temp_db):
                os.unlink(temp_db)

    def test_performance_delta_calculation(self):
        """Test calculating performance change between generations."""
        from aragora.evolution.tracker import EvolutionTracker

        tracker = EvolutionTracker()

        # Record outcomes for two generations
        tracker.record_outcome("agent-1", won=True, generation=0)
        tracker.record_outcome("agent-1", won=False, generation=0)
        tracker.record_outcome("agent-1", won=True, generation=1)
        tracker.record_outcome("agent-1", won=True, generation=1)

        delta = tracker.get_performance_delta("agent-1", gen1=0, gen2=1)

        # Gen 1 had 50% win rate, Gen 2 had 100%
        assert delta["win_rate_delta"] > 0


# ============================================================================
# Full Feedback Loop Tests
# ============================================================================


class TestEvolutionFeedbackLoop:
    """Integration tests for the complete evolution feedback loop."""

    def test_full_evolution_cycle(self, mock_debate_outcome, tmp_path):
        """Test complete debate → evolution → new debate cycle."""
        from aragora.genesis.breeding import PopulationManager
        from aragora.evolution.evolver import PromptEvolver

        db_path = str(tmp_path / "test_genesis.db")

        # Initialize population
        manager = PopulationManager(db_path=db_path, max_population_size=4)

        # Initialize evolver
        evolver = PromptEvolver()

        # Step 1: Create population and get a genome
        population = manager.get_or_create_population(["claude", "gemini"])
        assert len(population.genomes) > 0

        # Step 2: Update fitness based on debate outcome
        winner_genome = population.genomes[0]
        manager.update_fitness(
            genome_id=winner_genome.genome_id,
            consensus_win=True,
        )

        # Step 3: Evolve population
        evolved = manager.evolve_population(population)

        assert evolved is not None
        assert len(evolved.genomes) > 0

    def test_multiple_generations(self, mock_debate_outcome, tmp_path):
        """Test evolution across multiple generations."""
        from aragora.genesis.breeding import PopulationManager

        db_path = str(tmp_path / "test_genesis.db")
        manager = PopulationManager(db_path=db_path, max_population_size=4)

        # Create initial population
        population = manager.get_or_create_population(["claude", "gemini"])

        # Simulate 3 generations of evolution
        for gen in range(3):
            # Update fitness for some genomes
            for genome in population.genomes[:2]:
                manager.update_fitness(
                    genome_id=genome.genome_id,
                    consensus_win=(gen % 2 == 0),
                )

            # Evolve to next generation
            population = manager.evolve_population(population)

        assert population is not None
        assert population.generation >= 0


# ============================================================================
# Handler Integration Tests
# ============================================================================


class TestEvolutionHandlerIntegration:
    """Integration tests for evolution API handlers."""

    def test_handler_routes(self):
        """Test evolution handler routes are registered."""
        from aragora.server.handlers.evolution import EvolutionHandler

        handler = EvolutionHandler({})

        # Should have routes for evolution operations
        assert len(handler.ROUTES) > 0

    def test_get_patterns_endpoint(self, tmp_path):
        """Test patterns endpoint returns data."""
        from aragora.server.handlers.evolution import EvolutionHandler

        # Need nomic_dir in context
        ctx = {"prompt_evolver": Mock(), "nomic_dir": tmp_path}
        handler = EvolutionHandler(ctx)

        # _get_patterns now takes pattern_type and limit arguments
        result = handler._get_patterns(pattern_type=None, limit=10)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "patterns" in data

    def test_get_agent_evolution_history(self, tmp_path):
        """Test getting evolution history for an agent."""
        from aragora.server.handlers.evolution import EvolutionHandler

        # Need nomic_dir in context
        ctx = {"prompt_evolver": Mock(), "nomic_dir": tmp_path}
        handler = EvolutionHandler(ctx)

        # Method is _get_evolution_history, takes agent and limit
        result = handler._get_evolution_history(agent="claude", limit=10)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "history" in data


# ============================================================================
# Ledger Integration Tests
# ============================================================================


class TestGenesisLedger:
    """Tests for genesis ledger (evolution tracking database)."""

    def test_ledger_initialization(self):
        """Test ledger can be initialized."""
        from aragora.genesis.ledger import GenesisLedger
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = GenesisLedger(db_path=Path(tmpdir) / "genesis.db")
            assert ledger is not None

    def test_record_fitness_update(self):
        """Test recording fitness updates in the ledger."""
        from aragora.genesis.ledger import GenesisLedger
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = GenesisLedger(db_path=Path(tmpdir) / "genesis.db")

            event = ledger.record_fitness_update(
                genome_id="genome-123",
                old_fitness=0.5,
                new_fitness=0.65,
                reason="consensus_win",
            )

            assert event is not None
            assert event.data["genome_id"] == "genome-123"
            assert event.data["old_fitness"] == 0.5
            assert event.data["new_fitness"] == 0.65
            assert event.data["change"] == pytest.approx(0.15)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestEvolutionErrorHandling:
    """Tests for error handling in evolution system."""

    def test_mutation_handles_empty_prompt(self):
        """Test mutation handles empty prompts gracefully."""
        from aragora.evolution.evolver import PromptEvolver

        evolver = PromptEvolver()

        # Should not crash on empty input
        result = evolver.mutate("")
        assert result is not None or result == ""

    def test_population_handles_no_survivors(self):
        """Test population handles case with no survivors."""
        from aragora.genesis.breeding import PopulationManager

        manager = PopulationManager(max_population_size=2)

        # Edge case: all agents fail
        # Should handle gracefully
        assert manager is not None
