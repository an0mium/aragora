"""
Integration tests for debate with personas.

Tests that personas are correctly applied to agents during debate setup,
including system prompts and generation parameters.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestPersonaApplication:
    """Test persona application to agents."""

    def test_apply_persona_to_agent_with_default_persona(self):
        """Test applying a default persona to an agent."""
        from aragora.agents.personas import apply_persona_to_agent

        # Create mock agent
        agent = MagicMock()
        agent.system_prompt = ""
        agent.set_generation_params = MagicMock()

        # Apply philosopher persona
        result = apply_persona_to_agent(agent, "philosopher")

        assert result is True
        assert "philosopher" in agent.system_prompt.lower() or "deep thinker" in agent.system_prompt.lower()
        agent.set_generation_params.assert_called_once()

        # Verify generation params were set
        call_kwargs = agent.set_generation_params.call_args[1]
        assert "temperature" in call_kwargs
        assert "top_p" in call_kwargs

    def test_apply_persona_preserves_existing_prompt(self):
        """Test that applying persona preserves existing system prompt."""
        from aragora.agents.personas import apply_persona_to_agent

        agent = MagicMock()
        agent.system_prompt = "You are a helpful assistant."
        agent.set_generation_params = MagicMock()

        apply_persona_to_agent(agent, "philosopher")

        # Original prompt should still be present
        assert "helpful assistant" in agent.system_prompt
        # Persona prompt should be prepended
        assert agent.system_prompt.startswith("Your role:")

    def test_apply_unknown_persona_returns_false(self):
        """Test that applying unknown persona returns False."""
        from aragora.agents.personas import apply_persona_to_agent

        agent = MagicMock()
        agent.system_prompt = ""

        result = apply_persona_to_agent(agent, "nonexistent_persona_xyz")

        assert result is False

    def test_get_persona_prompt_returns_prompt(self):
        """Test getting persona prompt."""
        from aragora.agents.personas import get_persona_prompt

        prompt = get_persona_prompt("philosopher")

        assert prompt is not None
        assert len(prompt) > 0
        assert "philosophy" in prompt.lower() or "thinker" in prompt.lower()

    def test_get_persona_prompt_unknown_returns_empty(self):
        """Test getting unknown persona prompt returns empty string."""
        from aragora.agents.personas import get_persona_prompt

        prompt = get_persona_prompt("nonexistent_persona_xyz")

        assert prompt == ""


class TestPersonaInDebateFactory:
    """Test persona application in debate factory."""

    @pytest.fixture
    def mock_create_agent(self):
        """Mock agent creation."""
        with patch("aragora.server.debate_factory.create_agent") as mock:
            agent = MagicMock()
            agent.system_prompt = ""
            agent.set_generation_params = MagicMock()
            mock.return_value = agent
            yield mock, agent

    def test_debate_factory_applies_persona(self, mock_create_agent):
        """Test that debate factory applies persona to agents."""
        from aragora.server.debate_factory import DebateFactory
        from aragora.agents.spec import AgentSpec

        mock_create, agent = mock_create_agent

        factory = DebateFactory()
        specs = [
            AgentSpec(provider="anthropic-api", persona="philosopher", role="proposer"),
        ]

        result = factory.create_agents(specs, debate_id="test-debate")

        assert len(result.agents) == 1
        # Persona should have been applied
        assert agent.system_prompt != "" or agent.set_generation_params.called


class TestPersonaParameters:
    """Test persona generation parameters."""

    def test_philosopher_has_higher_temperature(self):
        """Test that philosopher persona has slightly higher temperature for creativity."""
        from aragora.agents.personas import DEFAULT_PERSONAS

        philosopher = DEFAULT_PERSONAS.get("philosopher")
        assert philosopher is not None
        # Philosopher should have higher temp (0.75) than default (0.7)
        assert philosopher.temperature >= 0.7

    def test_security_expert_has_lower_temperature(self):
        """Test that security expert has lower temperature for precision."""
        from aragora.agents.personas import DEFAULT_PERSONAS

        security = DEFAULT_PERSONAS.get("security_engineer")
        if security:
            # Security experts should be more deterministic
            assert security.temperature <= 0.7

    def test_contrarian_has_high_temperature(self):
        """Test that contrarian personas have high temperature for creativity."""
        from aragora.agents.personas import DEFAULT_PERSONAS

        contrarian = DEFAULT_PERSONAS.get("devils_advocate")
        if contrarian:
            # Contrarians should be more creative
            assert contrarian.temperature >= 0.75


class TestPersonaInAgentSpec:
    """Test persona handling in AgentSpec."""

    def test_agentspec_preserves_persona(self):
        """Test that AgentSpec correctly preserves persona field."""
        from aragora.agents.spec import AgentSpec

        spec = AgentSpec(
            provider="anthropic-api",
            persona="philosopher",
            role="proposer",
        )

        assert spec.persona == "philosopher"
        assert spec.role == "proposer"
        assert spec.provider == "anthropic-api"

    def test_agentspec_to_string_includes_persona(self):
        """Test that AgentSpec serialization includes persona."""
        from aragora.agents.spec import AgentSpec

        spec = AgentSpec(
            provider="anthropic-api",
            persona="philosopher",
            role="proposer",
        )

        string = spec.to_string()
        assert "philosopher" in string

    def test_create_team_with_personas(self):
        """Test creating a team with personas."""
        from aragora.agents.spec import AgentSpec

        team = AgentSpec.create_team([
            {"provider": "anthropic-api", "persona": "philosopher"},
            {"provider": "openai-api", "persona": "security_engineer"},
            {"provider": "gemini", "persona": "devils_advocate"},
        ])

        assert len(team) == 3
        assert team[0].persona == "philosopher"
        assert team[0].role == "proposer"  # Default rotation
        assert team[1].persona == "security_engineer"
        assert team[1].role == "critic"
        assert team[2].persona == "devils_advocate"
        assert team[2].role == "synthesizer"


class TestPersonaEvolution:
    """Test persona evolution and learning."""

    def test_persona_manager_creates_persona(self, tmp_path):
        """Test that PersonaManager can create and retrieve personas."""
        from aragora.agents.personas import PersonaManager, EXPERTISE_DOMAINS

        db_path = tmp_path / "test_personas.db"
        manager = PersonaManager(db_path=db_path)

        # Create a custom persona
        persona = manager.create_persona(
            agent_name="test_agent",
            description="A test agent",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.8, "testing": 0.6},
        )

        assert persona.agent_name == "test_agent"
        assert "thorough" in persona.traits
        assert persona.expertise.get("security") == 0.8

        # Retrieve it
        retrieved = manager.get_persona("test_agent")
        assert retrieved is not None
        assert retrieved.agent_name == "test_agent"
        assert "thorough" in retrieved.traits

    def test_apply_persona_with_manager(self, tmp_path):
        """Test applying persona from PersonaManager."""
        from aragora.agents.personas import PersonaManager, apply_persona_to_agent

        db_path = tmp_path / "test_personas.db"
        manager = PersonaManager(db_path=db_path)

        # Create a custom persona
        manager.create_persona(
            agent_name="custom_reviewer",
            description="A meticulous code reviewer",
            traits=["thorough", "direct"],
            expertise={"code_style": 0.9, "testing": 0.7},
        )

        # Apply it to an agent
        agent = MagicMock()
        agent.system_prompt = ""
        agent.set_generation_params = MagicMock()

        result = apply_persona_to_agent(agent, "custom_reviewer", manager=manager)

        assert result is True
        assert "reviewer" in agent.system_prompt.lower() or "thorough" in agent.system_prompt.lower()
