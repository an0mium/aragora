"""
Tests for agent personas with evolving specialization.

Tests cover:
- Persona dataclass properties
- PersonaManager CRUD operations
- Performance recording and expertise evolution
- Trait inference
- Prompt generation
"""

import pytest
import tempfile
from pathlib import Path

from aragora.agents.personas import (
    EXPERTISE_DOMAINS,
    PERSONALITY_TRAITS,
    Persona,
    PersonaManager,
    DEFAULT_PERSONAS,
    get_or_create_persona,
    apply_persona_to_agent,
    get_persona_prompt,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def persona_manager(temp_db):
    """Create a PersonaManager with a temporary database."""
    return PersonaManager(db_path=temp_db)


class TestPersonaDataclass:
    """Tests for the Persona dataclass."""

    def test_create_basic_persona(self):
        """Test creating a basic persona."""
        persona = Persona(
            agent_name="test_agent",
            traits=["thorough", "direct"],
            expertise={"security": 0.8, "testing": 0.6},
        )

        assert persona.agent_name == "test_agent"
        assert "thorough" in persona.traits
        assert persona.expertise["security"] == 0.8

    def test_persona_defaults(self):
        """Test persona default values."""
        persona = Persona(agent_name="minimal")

        assert persona.agent_name == "minimal"
        assert persona.traits == []
        assert persona.expertise == {}
        assert persona.temperature == 0.7
        assert persona.top_p == 1.0  # Default is 1.0

    def test_top_expertise(self):
        """Test top_expertise property returns sorted expertise."""
        persona = Persona(
            agent_name="expert",
            expertise={"security": 0.5, "testing": 0.9, "architecture": 0.7},
        )

        top = persona.top_expertise

        assert len(top) == 3
        # Should be sorted by score descending
        assert top[0][0] == "testing"
        assert top[0][1] == 0.9
        assert top[1][0] == "architecture"
        assert top[2][0] == "security"

    def test_trait_string(self):
        """Test trait_string property."""
        persona = Persona(
            agent_name="traited",
            traits=["thorough", "pragmatic", "direct"],
        )

        trait_str = persona.trait_string

        assert "thorough" in trait_str
        assert "pragmatic" in trait_str
        assert "direct" in trait_str

    def test_to_prompt_context(self):
        """Test generating prompt context from persona."""
        persona = Persona(
            agent_name="prompted",
            traits=["thorough"],
            expertise={"security": 0.9},
        )

        context = persona.to_prompt_context()

        assert isinstance(context, str)
        assert len(context) > 0
        # Should mention the agent's expertise
        assert "security" in context.lower() or "expertise" in context.lower()

    def test_generation_params(self):
        """Test generation_params property."""
        persona = Persona(
            agent_name="configured",
            temperature=0.5,
            top_p=0.8,
        )

        params = persona.generation_params

        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.8


class TestPersonaManager:
    """Tests for the PersonaManager class."""

    def test_create_persona(self, persona_manager):
        """Test creating a new persona."""
        persona = persona_manager.create_persona(
            agent_name="new_agent",
            traits=["thorough", "direct"],
            expertise={"security": 0.5},
        )

        assert persona.agent_name == "new_agent"
        assert "thorough" in persona.traits
        assert persona.expertise.get("security") == 0.5

    def test_get_persona(self, persona_manager):
        """Test retrieving a persona."""
        # Create first
        persona_manager.create_persona(
            agent_name="retrievable",
            traits=["pragmatic"],
        )

        # Then retrieve
        persona = persona_manager.get_persona("retrievable")

        assert persona is not None
        assert persona.agent_name == "retrievable"
        assert "pragmatic" in persona.traits

    def test_get_nonexistent_persona(self, persona_manager):
        """Test getting a persona that doesn't exist."""
        persona = persona_manager.get_persona("nonexistent")

        assert persona is None

    def test_create_persona_with_description(self, persona_manager):
        """Test creating persona with custom description."""
        persona = persona_manager.create_persona(
            agent_name="custom_desc",
            description="A specialized security analyst",
            traits=["thorough", "risk_aware"],
        )

        assert persona.description == "A specialized security analyst"
        assert "thorough" in persona.traits

    def test_get_all_personas_empty(self, persona_manager):
        """Test getting all personas when none exist."""
        personas = persona_manager.get_all_personas()

        assert personas == []

    def test_get_all_personas_multiple(self, persona_manager):
        """Test getting all personas."""
        persona_manager.create_persona(agent_name="agent1")
        persona_manager.create_persona(agent_name="agent2")
        persona_manager.create_persona(agent_name="agent3")

        personas = persona_manager.get_all_personas()

        assert len(personas) == 3
        names = [p.agent_name for p in personas]
        assert "agent1" in names
        assert "agent2" in names
        assert "agent3" in names


class TestPerformanceTracking:
    """Tests for performance recording and expertise evolution."""

    def test_record_performance(self, persona_manager):
        """Test recording performance updates expertise."""
        persona_manager.create_persona(
            agent_name="learner",
            expertise={"security": 0.5},
        )

        # Record a successful performance in security domain
        persona_manager.record_performance(
            agent_name="learner",
            domain="security",
            success=True,
            action="critique",
        )

        # Check expertise is still tracked
        persona = persona_manager.get_persona("learner")
        assert persona is not None
        # Expertise should be tracked
        assert "security" in persona.expertise

    def test_record_performance_new_domain(self, persona_manager):
        """Test recording performance in a new domain."""
        persona_manager.create_persona(agent_name="versatile")

        # Record performance in a domain not yet tracked
        persona_manager.record_performance(
            agent_name="versatile",
            domain="testing",
            success=True,
            action="proposal",
        )

        persona = persona_manager.get_persona("versatile")
        assert persona is not None
        # Should have some expertise in testing now
        assert "testing" in persona.expertise

    def test_get_performance_summary(self, persona_manager):
        """Test getting performance summary."""
        persona_manager.create_persona(agent_name="summarized")
        persona_manager.record_performance(
            agent_name="summarized",
            domain="security",
            success=True,
        )

        summary = persona_manager.get_performance_summary("summarized")

        assert isinstance(summary, dict)


class TestTraitInference:
    """Tests for trait inference."""

    def test_infer_traits(self, persona_manager):
        """Test inferring traits based on performance."""
        persona_manager.create_persona(agent_name="inferable")

        # Record some performance to enable inference
        for i in range(5):
            persona_manager.record_performance(
                agent_name="inferable",
                domain="security",
                success=True,
            )

        traits = persona_manager.infer_traits("inferable")

        assert isinstance(traits, list)


class TestExpertiseDomains:
    """Tests for expertise domain constants."""

    def test_expertise_domains_exist(self):
        """Test that expertise domains are defined."""
        assert len(EXPERTISE_DOMAINS) > 0

    def test_expertise_domains_include_technical(self):
        """Test that technical domains are included."""
        assert "security" in EXPERTISE_DOMAINS
        assert "testing" in EXPERTISE_DOMAINS
        assert "architecture" in EXPERTISE_DOMAINS

    def test_expertise_domains_include_compliance(self):
        """Test that compliance domains are included."""
        assert "hipaa" in EXPERTISE_DOMAINS
        assert "gdpr" in EXPERTISE_DOMAINS
        assert "sox_compliance" in EXPERTISE_DOMAINS


class TestPersonalityTraits:
    """Tests for personality trait constants."""

    def test_personality_traits_exist(self):
        """Test that personality traits are defined."""
        assert len(PERSONALITY_TRAITS) > 0

    def test_personality_traits_include_common(self):
        """Test that common traits are included."""
        assert "thorough" in PERSONALITY_TRAITS
        assert "pragmatic" in PERSONALITY_TRAITS
        assert "direct" in PERSONALITY_TRAITS

    def test_personality_traits_include_compliance(self):
        """Test that compliance-focused traits are included."""
        assert "regulatory" in PERSONALITY_TRAITS
        assert "risk_aware" in PERSONALITY_TRAITS
        assert "audit_minded" in PERSONALITY_TRAITS


class TestDefaultPersonas:
    """Tests for default persona definitions."""

    def test_default_personas_exist(self):
        """Test that default personas are defined."""
        assert len(DEFAULT_PERSONAS) > 0

    def test_default_personas_are_persona_objects(self):
        """Test that default personas are Persona instances."""
        for name, persona in DEFAULT_PERSONAS.items():
            assert isinstance(name, str)
            assert isinstance(persona, Persona)
            assert persona.agent_name == name


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_or_create_persona_creates_new(self, temp_db):
        """Test get_or_create_persona creates new persona."""
        # Use a fresh manager
        manager = PersonaManager(db_path=temp_db)

        # Note: signature is (manager, agent_name)
        persona = get_or_create_persona(manager, "brand_new")

        assert persona is not None
        assert persona.agent_name == "brand_new"

    def test_get_or_create_persona_returns_existing(self, temp_db):
        """Test get_or_create_persona returns existing persona."""
        manager = PersonaManager(db_path=temp_db)

        # Create first
        manager.create_persona(
            agent_name="existing",
            traits=["thorough"],
        )

        # Then get_or_create should return it
        # Note: signature is (manager, agent_name)
        persona = get_or_create_persona(manager, "existing")

        assert persona is not None
        assert "thorough" in persona.traits

    def test_get_persona_prompt(self, temp_db):
        """Test generating a persona prompt."""
        manager = PersonaManager(db_path=temp_db)
        manager.create_persona(
            agent_name="promptable",
            traits=["thorough", "direct"],
            expertise={"security": 0.9},
        )

        prompt = get_persona_prompt("promptable", manager=manager)

        assert isinstance(prompt, str)
        # Should contain persona information
        assert len(prompt) > 0
