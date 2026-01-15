"""Tests for the DebateFactory class."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from aragora.server.debate_factory import (
    AgentSpec,
    AgentCreationResult,
    DebateConfig,
    DebateFactory,
)
from aragora.config import ALLOWED_AGENT_TYPES, MAX_AGENTS_PER_DEBATE


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""

    def test_valid_agent_type(self):
        """Valid agent types are accepted."""
        spec = AgentSpec(agent_type="anthropic-api")
        assert spec.agent_type == "anthropic-api"

    def test_invalid_agent_type_raises(self):
        """Invalid agent types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid agent type"):
            AgentSpec(agent_type="nonexistent-agent")

    def test_default_name_generated(self):
        """Default name is generated from type and role."""
        spec = AgentSpec(agent_type="anthropic-api", role="critic")
        assert spec.name == "anthropic-api_critic"

    def test_default_name_with_no_role(self):
        """Default name uses 'proposer' when no role specified."""
        spec = AgentSpec(agent_type="anthropic-api")
        assert spec.name == "anthropic-api_proposer"

    def test_custom_name_preserved(self):
        """Custom names are preserved."""
        spec = AgentSpec(agent_type="anthropic-api", name="custom_agent")
        assert spec.name == "custom_agent"


class TestAgentCreationResult:
    """Tests for AgentCreationResult dataclass."""

    def test_empty_result(self):
        """Empty result has zero counts."""
        result = AgentCreationResult()
        assert result.success_count == 0
        assert result.failure_count == 0
        assert not result.has_minimum

    def test_success_count(self):
        """success_count returns agent count."""
        result = AgentCreationResult(agents=["a1", "a2", "a3"])
        assert result.success_count == 3

    def test_failure_count(self):
        """failure_count returns failed count."""
        result = AgentCreationResult(failed=[("a1", "err1"), ("a2", "err2")])
        assert result.failure_count == 2

    def test_has_minimum_with_two(self):
        """has_minimum is True with 2+ agents."""
        result = AgentCreationResult(agents=["a1", "a2"])
        assert result.has_minimum

    def test_has_minimum_with_one(self):
        """has_minimum is False with only 1 agent."""
        result = AgentCreationResult(agents=["a1"])
        assert not result.has_minimum


class TestDebateConfig:
    """Tests for DebateConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = DebateConfig(question="Test question")
        assert config.agents_str == "anthropic-api,openai-api"
        assert config.rounds == 8  # 9-round format (0-8) for web debates
        assert config.consensus == "judge"  # Judge-based for final decisions
        assert config.debate_format == "full"  # Full thorough debate by default

    def test_parse_agent_specs_simple(self):
        """Simple agent string is parsed correctly."""
        config = DebateConfig(question="Q", agents_str="anthropic-api,openai-api")
        specs = config.parse_agent_specs()
        assert len(specs) == 2
        assert specs[0].agent_type == "anthropic-api"
        assert specs[1].agent_type == "openai-api"

    def test_parse_agent_specs_with_roles(self):
        """Agent string with roles is parsed correctly."""
        config = DebateConfig(question="Q", agents_str="anthropic-api:critic,openai-api:proposer")
        specs = config.parse_agent_specs()
        assert specs[0].role == "critic"
        assert specs[1].role == "proposer"

    def test_parse_agent_specs_mixed(self):
        """Mixed agent string (with and without roles) is parsed."""
        config = DebateConfig(question="Q", agents_str="anthropic-api,openai-api:critic")
        specs = config.parse_agent_specs()
        assert specs[0].role is None
        assert specs[1].role == "critic"

    def test_parse_agent_specs_strips_whitespace(self):
        """Whitespace is stripped from agent specs."""
        config = DebateConfig(question="Q", agents_str="  anthropic-api  ,  openai-api  ")
        specs = config.parse_agent_specs()
        assert specs[0].agent_type == "anthropic-api"
        assert specs[1].agent_type == "openai-api"

    def test_parse_agent_specs_too_many_agents(self):
        """Too many agents raises ValueError."""
        agents = ",".join([f"anthropic-api"] * (MAX_AGENTS_PER_DEBATE + 1))
        config = DebateConfig(question="Q", agents_str=agents)
        with pytest.raises(ValueError, match="Too many agents"):
            config.parse_agent_specs()

    def test_parse_agent_specs_too_few_agents(self):
        """Too few agents raises ValueError."""
        config = DebateConfig(question="Q", agents_str="anthropic-api")
        with pytest.raises(ValueError, match="At least 2 agents"):
            config.parse_agent_specs()

    def test_parse_agent_specs_empty_string(self):
        """Empty agents string raises ValueError."""
        config = DebateConfig(question="Q", agents_str="")
        with pytest.raises(ValueError, match="At least 2 agents"):
            config.parse_agent_specs()

    def test_parse_agent_specs_invalid_agent_type(self):
        """Invalid agent type raises ValueError."""
        config = DebateConfig(question="Q", agents_str="anthropic-api,invalid-agent")
        with pytest.raises(ValueError, match="Invalid agent type"):
            config.parse_agent_specs()


class TestDebateFactory:
    """Tests for DebateFactory class."""

    def test_init_with_all_subsystems(self):
        """Factory initializes with all subsystems."""
        elo = Mock()
        persona = Mock()
        embeddings = Mock()
        emitter = Mock()

        factory = DebateFactory(
            elo_system=elo,
            persona_manager=persona,
            debate_embeddings=embeddings,
            stream_emitter=emitter,
        )

        assert factory.elo_system is elo
        assert factory.persona_manager is persona
        assert factory.debate_embeddings is embeddings
        assert factory.stream_emitter is emitter

    def test_init_with_no_subsystems(self):
        """Factory initializes without subsystems."""
        factory = DebateFactory()

        assert factory.elo_system is None
        assert factory.persona_manager is None
        assert factory.stream_emitter is None


class TestDebateFactoryCreateAgents:
    """Tests for DebateFactory.create_agents method."""

    def test_create_agents_success(self):
        """Successfully creates agents from specs."""
        import aragora.server.debate_factory as factory_module

        mock_agent1 = Mock()
        mock_agent2 = Mock()

        with patch.object(factory_module, "create_agent", side_effect=[mock_agent1, mock_agent2]):
            factory = DebateFactory()
            specs = [
                AgentSpec(agent_type="anthropic-api"),
                AgentSpec(agent_type="openai-api"),
            ]

            result = factory.create_agents(specs)

            assert result.success_count == 2
            assert result.failure_count == 0
            assert result.has_minimum
            assert mock_agent1 in result.agents
            assert mock_agent2 in result.agents

    def test_create_agents_partial_failure(self):
        """Records failures but continues creating other agents."""
        import aragora.server.debate_factory as factory_module

        mock_agent = Mock()

        with patch.object(
            factory_module, "create_agent", side_effect=[mock_agent, ValueError("API key missing")]
        ):
            factory = DebateFactory()
            specs = [
                AgentSpec(agent_type="anthropic-api"),
                AgentSpec(agent_type="openai-api"),
            ]

            result = factory.create_agents(specs)

            assert result.success_count == 1
            assert result.failure_count == 1
            assert not result.has_minimum
            assert ("openai-api", "API key missing") in result.failed

    def test_create_agents_checks_api_key(self):
        """Validates API key for API-based agents."""
        import aragora.server.debate_factory as factory_module

        mock_agent = Mock()
        mock_agent.api_key = None  # Missing API key

        with patch.object(factory_module, "create_agent", return_value=mock_agent):
            factory = DebateFactory()
            specs = [AgentSpec(agent_type="anthropic-api")]

            result = factory.create_agents(specs)

            assert result.failure_count == 1
            assert "Missing API key" in result.failed[0][1]

    def test_create_agents_with_stream_wrapper(self):
        """Applies stream wrapper to created agents."""
        import aragora.server.debate_factory as factory_module

        mock_agent = Mock()
        mock_wrapped = Mock()

        wrapper = Mock(return_value=mock_wrapped)
        emitter = Mock()

        with patch.object(factory_module, "create_agent", return_value=mock_agent):
            factory = DebateFactory(stream_emitter=emitter)
            specs = [AgentSpec(agent_type="anthropic-api")]

            result = factory.create_agents(specs, stream_wrapper=wrapper, debate_id="test-123")

            wrapper.assert_called_once_with(mock_agent, emitter, "test-123")
            assert mock_wrapped in result.agents

    def test_create_agents_emits_error_event(self):
        """Emits error events for failed agents."""
        import aragora.server.debate_factory as factory_module

        emitter = Mock()

        with patch.object(
            factory_module, "create_agent", side_effect=ValueError("Creation failed")
        ):
            factory = DebateFactory(stream_emitter=emitter)
            specs = [AgentSpec(agent_type="anthropic-api")]

            factory.create_agents(specs, debate_id="test-123")

            # Verify emit was called (error event emission)
            assert emitter.emit.called


class TestDebateFactoryCreateArena:
    """Tests for DebateFactory.create_arena method."""

    def test_create_arena_success(self):
        """Successfully creates arena with agents."""
        import aragora.server.debate_factory as factory_module

        mock_agent1 = Mock()
        mock_agent2 = Mock()
        mock_env = Mock()
        mock_protocol = Mock()
        mock_arena = Mock()

        with (
            patch.object(factory_module, "create_agent", side_effect=[mock_agent1, mock_agent2]),
            patch("aragora.core.Environment", return_value=mock_env) as env_cls,
            patch(
                "aragora.debate.protocol.DebateProtocol", return_value=mock_protocol
            ) as proto_cls,
            patch("aragora.debate.orchestrator.Arena", return_value=mock_arena) as arena_cls,
        ):

            factory = DebateFactory()
            config = DebateConfig(
                question="Test question",
                agents_str="anthropic-api,openai-api",
                rounds=3,
            )

            arena = factory.create_arena(config)

            env_cls.assert_called_once()
            proto_cls.assert_called_once()
            arena_cls.assert_called_once()

            # Verify arena was returned
            assert arena is mock_arena

    def test_create_arena_insufficient_agents_raises(self):
        """Raises ValueError when not enough agents created."""
        import aragora.server.debate_factory as factory_module

        with patch.object(
            factory_module, "create_agent", side_effect=ValueError("Creation failed")
        ):
            factory = DebateFactory()
            config = DebateConfig(
                question="Test question",
                agents_str="anthropic-api,openai-api",
            )

            with pytest.raises(ValueError, match="Only 0 agents initialized"):
                factory.create_arena(config)

    def test_create_arena_passes_subsystems(self):
        """Passes all subsystems to Arena constructor."""
        import aragora.server.debate_factory as factory_module

        mock_agent = Mock()
        mock_arena = Mock()

        elo = Mock()
        persona = Mock()
        embeddings = Mock()
        emitter = Mock()

        with (
            patch.object(factory_module, "create_agent", return_value=mock_agent),
            patch("aragora.core.Environment"),
            patch("aragora.debate.protocol.DebateProtocol"),
            patch("aragora.debate.orchestrator.Arena", return_value=mock_arena) as arena_cls,
        ):

            factory = DebateFactory(
                elo_system=elo,
                persona_manager=persona,
                debate_embeddings=embeddings,
                stream_emitter=emitter,
            )
            config = DebateConfig(
                question="Test question",
                agents_str="anthropic-api,openai-api",
                debate_id="test-123",
            )

            factory.create_arena(config)

            # Verify subsystems were passed to Arena
            call_kwargs = arena_cls.call_args[1]
            assert call_kwargs["elo_system"] is elo
            assert call_kwargs["persona_manager"] is persona
            assert call_kwargs["debate_embeddings"] is embeddings
            assert call_kwargs["event_emitter"] is emitter
            assert call_kwargs["loop_id"] == "test-123"


class TestDebateFactoryResetCircuitBreakers:
    """Tests for DebateFactory.reset_circuit_breakers method."""

    def test_reset_open_circuits(self):
        """Resets open circuit breakers."""
        arena = Mock()
        arena.circuit_breaker.get_all_status.return_value = {
            "agent1": {"status": "open"},
            "agent2": {"status": "closed"},
        }

        factory = DebateFactory()
        factory.reset_circuit_breakers(arena)

        arena.circuit_breaker.reset.assert_called_once()

    def test_no_reset_when_all_closed(self):
        """Doesn't reset when all circuits are closed."""
        arena = Mock()
        arena.circuit_breaker.get_all_status.return_value = {
            "agent1": {"status": "closed"},
            "agent2": {"status": "closed"},
        }

        factory = DebateFactory()
        factory.reset_circuit_breakers(arena)

        arena.circuit_breaker.reset.assert_not_called()

    def test_handles_empty_status(self):
        """Handles empty circuit breaker status."""
        arena = Mock()
        arena.circuit_breaker.get_all_status.return_value = {}

        factory = DebateFactory()
        factory.reset_circuit_breakers(arena)

        arena.circuit_breaker.reset.assert_not_called()
