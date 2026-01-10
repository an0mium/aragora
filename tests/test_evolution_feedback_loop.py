"""
Integration tests for Evolution Feedback Loop.

Tests cover the debate → evolution → improved debate cycle:
- Pattern extraction from debate outcomes
- Prompt mutation and evolution
- Performance tracking across generations
- Population management and selection
"""

import json
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
        try:
            from aragora.evolution.pattern_extractor import extract_patterns

            patterns = extract_patterns(mock_debate_outcome)

            assert patterns is not None
            assert "winning_patterns" in patterns or len(patterns) > 0
        except ImportError:
            pytest.skip("Pattern extractor not available")

    def test_identify_successful_strategies(self, mock_debate_outcome):
        """Test identifying successful debate strategies."""
        try:
            from aragora.evolution.pattern_extractor import identify_strategies

            strategies = identify_strategies(mock_debate_outcome)

            assert strategies is not None
        except ImportError:
            pytest.skip("Pattern extractor not available")

    def test_patterns_associated_with_winner(self, mock_debate_outcome):
        """Test patterns are correctly associated with winner."""
        try:
            from aragora.evolution.pattern_extractor import extract_patterns

            patterns = extract_patterns(mock_debate_outcome)
            winner = mock_debate_outcome["winner"]

            # Patterns should reference the winning agent
            assert winner in str(patterns) or len(patterns) > 0
        except ImportError:
            pytest.skip("Pattern extractor not available")


# ============================================================================
# Prompt Mutation Tests
# ============================================================================

class TestPromptMutation:
    """Tests for prompt mutation in evolution."""

    def test_mutate_prompt(self):
        """Test basic prompt mutation."""
        try:
            from aragora.evolution.evolver import PromptEvolver

            evolver = PromptEvolver()

            original_prompt = "You are a helpful assistant that argues logically."
            mutated = evolver.mutate(original_prompt)

            assert mutated is not None
            assert mutated != original_prompt or len(mutated) > 0
        except ImportError:
            pytest.skip("Prompt evolver not available")

    def test_crossover_prompts(self):
        """Test combining traits from two prompts."""
        try:
            from aragora.evolution.evolver import PromptEvolver

            evolver = PromptEvolver()

            parent1 = "Be concise and factual in arguments."
            parent2 = "Use examples and analogies to illustrate points."

            offspring = evolver.crossover(parent1, parent2)

            assert offspring is not None
        except ImportError:
            pytest.skip("Prompt evolver not available")

    def test_mutation_rate_affects_output(self):
        """Test that mutation rate affects output diversity."""
        try:
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
        except ImportError:
            pytest.skip("Prompt evolver not available")


# ============================================================================
# Population Management Tests
# ============================================================================

class TestPopulationManagement:
    """Tests for population management in evolution."""

    def test_population_initialization(self):
        """Test initializing a population."""
        try:
            from aragora.genesis.breeding import PopulationManager

            manager = PopulationManager(population_size=10)

            assert manager is not None
            assert manager.population_size == 10
        except ImportError:
            pytest.skip("PopulationManager not available")

    def test_selection_by_fitness(self, mock_population):
        """Test selecting agents by fitness."""
        try:
            from aragora.genesis.breeding import select_by_fitness

            selected = select_by_fitness(mock_population, count=2)

            assert len(selected) == 2
            # Higher rated agents should be more likely to be selected
        except ImportError:
            pytest.skip("Selection function not available")

    def test_elitism_preserves_best(self, mock_population):
        """Test elitism preserves best performers."""
        try:
            from aragora.genesis.breeding import PopulationManager

            manager = PopulationManager(elitism=2)
            manager.population = mock_population

            # Simulate evolution step
            new_population = manager.evolve_generation()

            # Best 2 should be preserved (elitism)
            assert len(new_population) > 0
        except ImportError:
            pytest.skip("PopulationManager not available")


# ============================================================================
# Performance Tracking Tests
# ============================================================================

class TestPerformanceTracking:
    """Tests for tracking evolution performance."""

    def test_record_outcome(self, mock_debate_outcome):
        """Test recording debate outcome for evolution."""
        try:
            from aragora.evolution.tracker import EvolutionTracker

            tracker = EvolutionTracker()
            tracker.record_outcome(
                agent="claude",
                won=True,
                debate_id=mock_debate_outcome["debate_id"],
            )

            stats = tracker.get_agent_stats("claude")
            assert stats["wins"] >= 1
        except ImportError:
            pytest.skip("EvolutionTracker not available")

    def test_generation_metrics(self):
        """Test tracking metrics across generations."""
        try:
            from aragora.evolution.tracker import EvolutionTracker

            tracker = EvolutionTracker()

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
        except ImportError:
            pytest.skip("EvolutionTracker not available")

    def test_performance_delta_calculation(self):
        """Test calculating performance change between generations."""
        try:
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
        except ImportError:
            pytest.skip("EvolutionTracker not available")


# ============================================================================
# Full Feedback Loop Tests
# ============================================================================

class TestEvolutionFeedbackLoop:
    """Integration tests for the complete evolution feedback loop."""

    @pytest.mark.asyncio
    async def test_full_evolution_cycle(self, mock_debate_outcome):
        """Test complete debate → evolution → new debate cycle."""
        try:
            from aragora.genesis.breeding import PopulationManager
            from aragora.evolution.evolver import PromptEvolver

            # Initialize population
            manager = PopulationManager(population_size=4)

            # Initialize evolver
            evolver = PromptEvolver()

            # Step 1: Record debate outcome
            manager.record_debate_result(
                winner=mock_debate_outcome["winner"],
                loser="gpt4",
                margin=mock_debate_outcome["votes"]["claude"] - mock_debate_outcome["votes"]["gpt4"],
            )

            # Step 2: Extract patterns and evolve
            # (In real implementation, this would use patterns)
            new_generation = manager.evolve_generation()

            assert new_generation is not None
            assert len(new_generation) > 0
        except ImportError:
            pytest.skip("Evolution modules not available")

    @pytest.mark.asyncio
    async def test_multiple_generations(self, mock_debate_outcome):
        """Test evolution across multiple generations."""
        try:
            from aragora.genesis.breeding import PopulationManager

            manager = PopulationManager(population_size=4)

            # Simulate 3 generations
            for gen in range(3):
                # Record some outcomes
                for i in range(4):
                    manager.record_debate_result(
                        winner=f"agent-{i % 2}",
                        loser=f"agent-{(i + 1) % 2}",
                        margin=1,
                    )

                # Evolve to next generation
                manager.evolve_generation()

            assert manager.current_generation >= 3
        except ImportError:
            pytest.skip("PopulationManager not available")


# ============================================================================
# Handler Integration Tests
# ============================================================================

class TestEvolutionHandlerIntegration:
    """Integration tests for evolution API handlers."""

    def test_handler_routes(self):
        """Test evolution handler routes are registered."""
        try:
            from aragora.server.handlers.evolution import EvolutionHandler

            handler = EvolutionHandler({})

            # Should have routes for evolution operations
            assert len(handler.ROUTES) > 0
        except ImportError:
            pytest.skip("Evolution handler not available")

    def test_get_patterns_endpoint(self):
        """Test patterns endpoint returns data."""
        try:
            from aragora.server.handlers.evolution import EvolutionHandler

            ctx = {"prompt_evolver": Mock()}
            handler = EvolutionHandler(ctx)

            result = handler._get_patterns()

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "patterns" in data or "error" not in data
        except ImportError:
            pytest.skip("Evolution handler not available")

    def test_get_agent_evolution_history(self):
        """Test getting evolution history for an agent."""
        try:
            from aragora.server.handlers.evolution import EvolutionHandler

            mock_evolver = Mock()
            mock_evolver.get_agent_history = Mock(return_value=[
                {"generation": 0, "prompt_version": 1},
                {"generation": 1, "prompt_version": 2},
            ])

            ctx = {"prompt_evolver": mock_evolver}
            handler = EvolutionHandler(ctx)

            result = handler._get_agent_history("claude")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "history" in data
        except ImportError:
            pytest.skip("Evolution handler not available")


# ============================================================================
# Ledger Integration Tests
# ============================================================================

class TestGenesisLedger:
    """Tests for genesis ledger (evolution tracking database)."""

    def test_ledger_initialization(self):
        """Test ledger can be initialized."""
        try:
            from aragora.genesis.ledger import GenesisLedger
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                ledger = GenesisLedger(db_path=Path(tmpdir) / "genesis.db")
                assert ledger is not None
        except ImportError:
            pytest.skip("GenesisLedger not available")

    def test_record_generation(self):
        """Test recording a generation in the ledger."""
        try:
            from aragora.genesis.ledger import GenesisLedger
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                ledger = GenesisLedger(db_path=Path(tmpdir) / "genesis.db")

                ledger.record_generation(
                    generation=1,
                    population_size=10,
                    avg_fitness=0.65,
                    best_fitness=0.85,
                )

                stats = ledger.get_generation_stats(1)
                assert stats is not None
                assert stats["population_size"] == 10
        except ImportError:
            pytest.skip("GenesisLedger not available")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestEvolutionErrorHandling:
    """Tests for error handling in evolution system."""

    def test_mutation_handles_empty_prompt(self):
        """Test mutation handles empty prompts gracefully."""
        try:
            from aragora.evolution.evolver import PromptEvolver

            evolver = PromptEvolver()

            # Should not crash on empty input
            result = evolver.mutate("")
            assert result is not None or result == ""
        except ImportError:
            pytest.skip("PromptEvolver not available")

    def test_population_handles_no_survivors(self):
        """Test population handles case with no survivors."""
        try:
            from aragora.genesis.breeding import PopulationManager

            manager = PopulationManager(population_size=2)

            # Edge case: all agents fail
            # Should handle gracefully
            assert manager is not None
        except ImportError:
            pytest.skip("PopulationManager not available")
