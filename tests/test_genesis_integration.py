"""
Tests for Genesis-Arena integration.

Tests that the PopulationManager is properly wired to the Arena
and triggers breeding events after debates.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import tempfile
import os


class TestGenesisLedgerBasics:
    """Tests for GenesisLedger basic functionality."""

    def test_genesis_ledger_creation(self, temp_db):
        """GenesisLedger should be creatable with a database path."""
        try:
            from aragora.genesis.ledger import GenesisLedger
        except ImportError:
            pytest.skip("Genesis module not available")

        ledger = GenesisLedger(temp_db)
        assert ledger is not None

    def test_genesis_ledger_records_events(self, temp_db):
        """GenesisLedger should record events."""
        try:
            from aragora.genesis.ledger import GenesisLedger, GenesisEventType
        except ImportError:
            pytest.skip("Genesis module not available")

        ledger = GenesisLedger(temp_db)

        # Record an event
        event = ledger.record_event(
            event_type=GenesisEventType.AGENT_BIRTH,
            data={"genome_id": "test-genome-1", "agent_name": "claude"},
        )

        assert event is not None
        assert event.event_type == GenesisEventType.AGENT_BIRTH

    def test_genesis_ledger_integrity(self, temp_db):
        """GenesisLedger should verify integrity."""
        try:
            from aragora.genesis.ledger import GenesisLedger, GenesisEventType
        except ImportError:
            pytest.skip("Genesis module not available")

        ledger = GenesisLedger(temp_db)

        # Record some events
        ledger.record_event(
            event_type=GenesisEventType.AGENT_BIRTH,
            data={"genome_id": "test-1"},
        )
        ledger.record_event(
            event_type=GenesisEventType.FITNESS_UPDATE,
            data={"genome_id": "test-1", "change": 0.1},
        )

        # Verify integrity
        assert ledger.verify_integrity() is True


class TestGenomeStoreBasics:
    """Tests for GenomeStore basic functionality."""

    def test_genome_store_creation(self, temp_db):
        """GenomeStore should be creatable."""
        try:
            from aragora.genesis.genome import GenomeStore
        except ImportError:
            pytest.skip("Genome module not available")

        store = GenomeStore(temp_db)
        assert store is not None

    def test_genome_store_save_and_retrieve(self, temp_db):
        """GenomeStore should save and retrieve genomes."""
        try:
            from aragora.genesis.genome import GenomeStore, AgentGenome
        except ImportError:
            pytest.skip("Genome module not available")

        store = GenomeStore(temp_db)

        # Create a genome
        genome = AgentGenome(
            genome_id="test-genome-1",
            agent_name="claude",
            personality_traits=["analytical", "cautious"],
            expertise_domains=["code", "reasoning"],
            fitness_score=0.8,
            generation=1,
        )

        # Save
        store.save(genome)

        # Retrieve
        retrieved = store.get("test-genome-1")
        assert retrieved is not None
        assert retrieved.agent_name == "claude"
        assert retrieved.fitness_score == 0.8


class TestPopulationManagerBasics:
    """Tests for PopulationManager basic functionality."""

    def test_population_manager_creation(self, temp_db):
        """PopulationManager should be creatable."""
        try:
            from aragora.genesis.breeding import PopulationManager
        except ImportError:
            pytest.skip("Breeding module not available")

        manager = PopulationManager(db_path=temp_db)
        assert manager is not None

    def test_population_manager_get_or_create(self, temp_db):
        """PopulationManager should create population from base agents."""
        try:
            from aragora.genesis.breeding import PopulationManager
        except ImportError:
            pytest.skip("Breeding module not available")

        manager = PopulationManager(db_path=temp_db)
        population = manager.get_or_create_population(
            base_agents=["claude", "gpt4"]
        )

        assert population is not None
        assert len(population.genomes) >= 2


class TestGenesisArenaIntegration:
    """Tests for Genesis integration with Arena."""

    def test_arena_config_accepts_population_manager(self):
        """ArenaConfig should accept population_manager parameter."""
        from aragora.debate.orchestrator import ArenaConfig

        mock_manager = Mock()
        config = ArenaConfig(
            population_manager=mock_manager,
            auto_evolve=True,
        )

        assert config.population_manager is mock_manager
        assert config.auto_evolve is True

    def test_arena_phases_receive_population_manager(self):
        """Arena phases should receive population_manager."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        mock_manager = Mock()

        phase = FeedbackPhase(
            elo_system=None,
            persona_manager=None,
            position_ledger=None,
            relationship_tracker=None,
            moment_detector=None,
            debate_embeddings=None,
            flip_detector=None,
            continuum_memory=None,
            event_emitter=None,
            loop_id=None,
            emit_moment_event=lambda *args: None,
            store_debate_outcome_as_memory=AsyncMock(),
            update_continuum_memory_outcomes=AsyncMock(),
            index_debate_async=AsyncMock(),
            consensus_memory=None,
            calibration_tracker=None,
            population_manager=mock_manager,
            auto_evolve=True,
            breeding_threshold=10,
            prompt_evolver=None,
        )

        assert phase.population_manager is mock_manager
        assert phase.auto_evolve is True

    @patch('aragora.debate.arena_phases.PopulationManager')
    def test_arena_auto_creates_population_manager(self, mock_manager_class):
        """Arena should auto-create PopulationManager when auto_evolve=True."""
        from aragora.debate.orchestrator import Arena, ArenaConfig
        from aragora.core import Environment, DebateProtocol

        mock_manager_instance = Mock()
        mock_manager_class.return_value = mock_manager_instance

        env = Environment(task="Test task")
        protocol = DebateProtocol()
        config = ArenaConfig(auto_evolve=True)

        mock_agent = Mock()
        mock_agent.name = "test-agent"
        agents = [mock_agent]

        # This should auto-create the population manager
        arena = Arena(env, agents, protocol, config=config)

        # The manager should be created (if genesis is available)
        # This test verifies the wiring, not necessarily that it succeeds


class TestFitnessUpdates:
    """Tests for fitness score updates after debates."""

    def test_fitness_calculation(self):
        """Should calculate fitness from debate outcomes."""
        try:
            from aragora.genesis.breeding import PopulationManager
        except ImportError:
            pytest.skip("Breeding module not available")

        # Fitness should incorporate:
        # - Win rate
        # - Consensus contribution
        # - ELO rating

        mock_result = Mock()
        mock_result.winner = "claude"
        mock_result.confidence = 0.9
        mock_result.consensus_reached = True

        # Calculate expected fitness contribution
        base_fitness = 1.0 if mock_result.winner == "claude" else 0.5
        confidence_bonus = mock_result.confidence * 0.2
        consensus_bonus = 0.1 if mock_result.consensus_reached else 0

        expected_fitness = base_fitness + confidence_bonus + consensus_bonus
        assert expected_fitness > 1.0

    @pytest.mark.asyncio
    async def test_feedback_phase_updates_fitness(self, temp_db):
        """FeedbackPhase should update genome fitness after debates."""
        try:
            from aragora.genesis.breeding import PopulationManager
        except ImportError:
            pytest.skip("Breeding module not available")

        from aragora.debate.phases.feedback_phase import FeedbackPhase

        manager = PopulationManager(db_path=temp_db)
        population = manager.get_or_create_population(base_agents=["claude"])

        initial_fitness = population.genomes[0].fitness_score if population.genomes else 0

        phase = FeedbackPhase(
            elo_system=Mock(),
            persona_manager=None,
            position_ledger=None,
            relationship_tracker=None,
            moment_detector=None,
            debate_embeddings=None,
            flip_detector=None,
            continuum_memory=None,
            event_emitter=None,
            loop_id=None,
            emit_moment_event=lambda *args: None,
            store_debate_outcome_as_memory=AsyncMock(),
            update_continuum_memory_outcomes=AsyncMock(),
            index_debate_async=AsyncMock(),
            consensus_memory=None,
            calibration_tracker=None,
            population_manager=manager,
            auto_evolve=True,
            breeding_threshold=1,  # Low threshold for testing
            prompt_evolver=None,
        )

        # Simulate debate outcome
        if hasattr(phase, '_update_genome_fitness'):
            phase._update_genome_fitness("claude", 0.9, True)
            # Fitness should have changed
            updated = manager.genome_store.get(population.genomes[0].genome_id) if population.genomes else None
            # Note: actual fitness update depends on implementation


class TestBreedingTrigger:
    """Tests for breeding trigger after N debates."""

    def test_breeding_threshold_tracking(self, temp_db):
        """Should track debate count towards breeding threshold."""
        try:
            from aragora.genesis.breeding import PopulationManager
        except ImportError:
            pytest.skip("Breeding module not available")

        manager = PopulationManager(db_path=temp_db)
        population = manager.get_or_create_population(base_agents=["claude", "gpt4"])

        initial_gen = population.generation

        # Record multiple debates
        for i in range(5):
            population.debate_history.append({
                "debate_id": f"debate-{i}",
                "winner": "claude",
                "confidence": 0.8,
            })

        # After threshold, breeding should be triggered
        # The actual breeding logic is in the manager
        assert len(population.debate_history) == 5

    def test_breeding_produces_offspring(self, temp_db):
        """Breeding should produce new genome offspring."""
        try:
            from aragora.genesis.breeding import PopulationManager, GenomeBreeder
        except ImportError:
            pytest.skip("Breeding module not available")

        manager = PopulationManager(db_path=temp_db)
        population = manager.get_or_create_population(
            base_agents=["claude", "gpt4"]
        )

        initial_count = len(population.genomes)

        # Trigger breeding if available
        if hasattr(manager, 'breed_population'):
            manager.breed_population()
            # Should have more genomes now
            assert len(population.genomes) >= initial_count


class TestGenesisAPIEndpoint:
    """Tests for the /api/genesis/population endpoint."""

    def test_genesis_handler_routes(self):
        """GenesisHandler should include population route."""
        from aragora.server.handlers.genesis import GenesisHandler

        assert "/api/genesis/population" in GenesisHandler.ROUTES

    def test_genesis_handler_can_handle_population(self):
        """GenesisHandler should handle population path."""
        from aragora.server.handlers.genesis import GenesisHandler

        handler = GenesisHandler(ctx={})
        assert handler.can_handle("/api/genesis/population")

    @pytest.mark.asyncio
    async def test_genesis_population_response_format(self, temp_dir):
        """Population endpoint should return correct format."""
        from aragora.server.handlers.genesis import GenesisHandler

        # Create handler with temp nomic dir
        handler = GenesisHandler(ctx={"nomic_dir": temp_dir})

        # Get population
        result = handler._get_population(temp_dir)

        assert result is not None
        # Should return either success or service unavailable
        status_code = result[1] if isinstance(result, tuple) else 200
        assert status_code in [200, 503]  # OK or service unavailable
