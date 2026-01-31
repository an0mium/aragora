"""
Tests for agent personas with evolving specialization.

Tests cover:
- Persona dataclass properties and methods
- PersonaManager CRUD operations
- Performance recording and expertise evolution
- Trait inference from performance patterns
- Prompt generation
- Default persona definitions
- Helper functions (get_or_create_persona, apply_persona_to_agent, get_persona_prompt)
- Edge cases and boundary conditions
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import FrozenInstanceError

from aragora.agents.personas import (
    EXPERTISE_DOMAINS,
    PERSONALITY_TRAITS,
    Persona,
    PersonaManager,
    DEFAULT_PERSONAS,
    PERSONA_SCHEMA_VERSION,
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


# =============================================================================
# Persona Dataclass Tests
# =============================================================================


class TestPersonaDataclass:
    """Tests for the Persona dataclass."""

    def test_create_basic_persona(self):
        """Test creating a basic persona with minimal required fields."""
        persona = Persona(agent_name="test_agent")
        assert persona.agent_name == "test_agent"

    def test_create_persona_with_all_fields(self):
        """Test creating a persona with all fields populated."""
        persona = Persona(
            agent_name="full_agent",
            description="A comprehensive test persona",
            traits=["thorough", "direct", "pragmatic"],
            expertise={"security": 0.9, "testing": 0.7, "architecture": 0.5},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00",
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.1,
        )

        assert persona.agent_name == "full_agent"
        assert persona.description == "A comprehensive test persona"
        assert len(persona.traits) == 3
        assert persona.expertise["security"] == 0.9
        assert persona.temperature == 0.5
        assert persona.top_p == 0.9
        assert persona.frequency_penalty == 0.1

    def test_persona_defaults(self):
        """Test persona default values."""
        persona = Persona(agent_name="minimal")

        assert persona.agent_name == "minimal"
        assert persona.description == ""
        assert persona.traits == []
        assert persona.expertise == {}
        assert persona.temperature == 0.7
        assert persona.top_p == 1.0
        assert persona.frequency_penalty == 0.0

    def test_persona_timestamps_default_to_now(self):
        """Test that created_at and updated_at default to current time."""
        persona = Persona(agent_name="timestamped")

        assert persona.created_at is not None
        assert persona.updated_at is not None
        # ISO format check
        assert "T" in persona.created_at
        assert "T" in persona.updated_at


class TestPersonaTopExpertise:
    """Tests for the top_expertise property."""

    def test_top_expertise_returns_sorted(self):
        """Test top_expertise property returns sorted expertise."""
        persona = Persona(
            agent_name="expert",
            expertise={"security": 0.5, "testing": 0.9, "architecture": 0.7},
        )

        top = persona.top_expertise

        assert len(top) == 3
        assert top[0][0] == "testing"
        assert top[0][1] == 0.9
        assert top[1][0] == "architecture"
        assert top[1][1] == 0.7
        assert top[2][0] == "security"
        assert top[2][1] == 0.5

    def test_top_expertise_limits_to_three(self):
        """Test that top_expertise returns at most 3 items."""
        persona = Persona(
            agent_name="multi_expert",
            expertise={
                "security": 0.9,
                "testing": 0.8,
                "architecture": 0.7,
                "performance": 0.6,
                "api_design": 0.5,
            },
        )

        top = persona.top_expertise

        assert len(top) == 3
        assert top[0][0] == "security"
        assert top[1][0] == "testing"
        assert top[2][0] == "architecture"

    def test_top_expertise_with_empty_expertise(self):
        """Test top_expertise with no expertise defined."""
        persona = Persona(agent_name="novice")

        top = persona.top_expertise

        assert top == []

    def test_top_expertise_with_one_domain(self):
        """Test top_expertise with single expertise domain."""
        persona = Persona(
            agent_name="specialist",
            expertise={"security": 0.95},
        )

        top = persona.top_expertise

        assert len(top) == 1
        assert top[0] == ("security", 0.95)

    def test_top_expertise_with_two_domains(self):
        """Test top_expertise with two expertise domains."""
        persona = Persona(
            agent_name="dual_expert",
            expertise={"security": 0.8, "testing": 0.6},
        )

        top = persona.top_expertise

        assert len(top) == 2
        assert top[0] == ("security", 0.8)
        assert top[1] == ("testing", 0.6)


class TestPersonaTraitString:
    """Tests for the trait_string property."""

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
        assert ", " in trait_str  # Comma-separated

    def test_trait_string_empty_returns_balanced(self):
        """Test trait_string returns 'balanced' when no traits."""
        persona = Persona(agent_name="balanced")

        assert persona.trait_string == "balanced"

    def test_trait_string_single_trait(self):
        """Test trait_string with single trait."""
        persona = Persona(
            agent_name="focused",
            traits=["thorough"],
        )

        assert persona.trait_string == "thorough"


class TestPersonaToPromptContext:
    """Tests for the to_prompt_context method."""

    def test_to_prompt_context(self):
        """Test generating prompt context from persona."""
        persona = Persona(
            agent_name="prompted",
            description="A security-focused code reviewer",
            traits=["thorough"],
            expertise={"security": 0.9},
        )

        context = persona.to_prompt_context()

        assert isinstance(context, str)
        assert len(context) > 0
        assert "security" in context.lower()
        assert "thorough" in context.lower()

    def test_to_prompt_context_with_description(self):
        """Test prompt context includes description."""
        persona = Persona(
            agent_name="described",
            description="A specialized compliance reviewer",
        )

        context = persona.to_prompt_context()

        assert "Your role:" in context
        assert "compliance" in context.lower()

    def test_to_prompt_context_with_traits(self):
        """Test prompt context includes traits."""
        persona = Persona(
            agent_name="trait_agent",
            traits=["pragmatic", "direct"],
        )

        context = persona.to_prompt_context()

        assert "Your approach:" in context
        assert "pragmatic" in context
        assert "direct" in context

    def test_to_prompt_context_with_expertise(self):
        """Test prompt context includes expertise."""
        persona = Persona(
            agent_name="expert_agent",
            expertise={"security": 0.9, "testing": 0.8},
        )

        context = persona.to_prompt_context()

        assert "expertise" in context.lower()
        assert "security" in context.lower()

    def test_to_prompt_context_empty_persona(self):
        """Test prompt context for empty persona returns empty string."""
        persona = Persona(agent_name="empty")

        context = persona.to_prompt_context()

        assert context == ""


class TestPersonaGenerationParams:
    """Tests for the generation_params property."""

    def test_generation_params(self):
        """Test generation_params property."""
        persona = Persona(
            agent_name="configured",
            temperature=0.5,
            top_p=0.8,
            frequency_penalty=0.1,
        )

        params = persona.generation_params

        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.8
        assert params["frequency_penalty"] == 0.1

    def test_generation_params_defaults(self):
        """Test generation_params with default values."""
        persona = Persona(agent_name="default_params")

        params = persona.generation_params

        assert params["temperature"] == 0.7
        assert params["top_p"] == 1.0
        assert params["frequency_penalty"] == 0.0


# =============================================================================
# PersonaManager Tests
# =============================================================================


class TestPersonaManagerInit:
    """Tests for PersonaManager initialization."""

    def test_create_manager_with_temp_db(self, temp_db):
        """Test creating manager with explicit db path."""
        manager = PersonaManager(db_path=temp_db)
        assert manager is not None

    def test_manager_schema_version(self):
        """Test that schema version is defined."""
        assert PERSONA_SCHEMA_VERSION >= 1
        assert PersonaManager.SCHEMA_VERSION == PERSONA_SCHEMA_VERSION


class TestPersonaManagerCreate:
    """Tests for PersonaManager.create_persona."""

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

    def test_create_persona_with_description(self, persona_manager):
        """Test creating persona with custom description."""
        persona = persona_manager.create_persona(
            agent_name="custom_desc",
            description="A specialized security analyst",
            traits=["thorough", "risk_aware"],
        )

        assert persona.description == "A specialized security analyst"
        assert "thorough" in persona.traits

    def test_create_persona_validates_traits(self, persona_manager):
        """Test that invalid traits are filtered out."""
        persona = persona_manager.create_persona(
            agent_name="validated_traits",
            traits=["thorough", "invalid_trait", "direct"],
        )

        assert "thorough" in persona.traits
        assert "direct" in persona.traits
        assert "invalid_trait" not in persona.traits

    def test_create_persona_validates_expertise(self, persona_manager):
        """Test that invalid expertise domains are filtered out."""
        persona = persona_manager.create_persona(
            agent_name="validated_expertise",
            expertise={"security": 0.8, "invalid_domain": 0.5},
        )

        assert "security" in persona.expertise
        assert "invalid_domain" not in persona.expertise

    def test_create_persona_normalizes_expertise_scores(self, persona_manager):
        """Test that expertise scores are normalized to 0-1 range."""
        persona = persona_manager.create_persona(
            agent_name="normalized",
            expertise={"security": 1.5, "testing": -0.5, "architecture": 0.7},
        )

        assert persona.expertise["security"] == 1.0  # Clamped to 1.0
        assert persona.expertise["testing"] == 0.0  # Clamped to 0.0
        assert persona.expertise["architecture"] == 0.7

    def test_create_persona_upsert(self, persona_manager):
        """Test that creating persona with same name updates existing."""
        # Create initial
        persona_manager.create_persona(
            agent_name="updatable",
            description="Original description",
            traits=["thorough"],
        )

        # Update with same name
        updated = persona_manager.create_persona(
            agent_name="updatable",
            description="Updated description",
            traits=["pragmatic"],
        )

        assert updated.description == "Updated description"
        assert "pragmatic" in updated.traits


class TestPersonaManagerGet:
    """Tests for PersonaManager.get_persona."""

    def test_get_persona(self, persona_manager):
        """Test retrieving a persona."""
        persona_manager.create_persona(
            agent_name="retrievable",
            traits=["pragmatic"],
        )

        persona = persona_manager.get_persona("retrievable")

        assert persona is not None
        assert persona.agent_name == "retrievable"
        assert "pragmatic" in persona.traits

    def test_get_nonexistent_persona(self, persona_manager):
        """Test getting a persona that doesn't exist."""
        persona = persona_manager.get_persona("nonexistent")

        assert persona is None

    def test_get_persona_preserves_expertise(self, persona_manager):
        """Test that retrieved persona preserves expertise scores."""
        persona_manager.create_persona(
            agent_name="expert",
            expertise={"security": 0.9, "testing": 0.7},
        )

        persona = persona_manager.get_persona("expert")

        assert persona.expertise["security"] == 0.9
        assert persona.expertise["testing"] == 0.7


class TestPersonaManagerGetAll:
    """Tests for PersonaManager.get_all_personas."""

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

    def test_get_all_personas_are_persona_instances(self, persona_manager):
        """Test that all returned items are Persona instances."""
        persona_manager.create_persona(agent_name="agent1")
        persona_manager.create_persona(agent_name="agent2")

        personas = persona_manager.get_all_personas()

        for p in personas:
            assert isinstance(p, Persona)


# =============================================================================
# Performance Tracking Tests
# =============================================================================


class TestPerformanceTracking:
    """Tests for performance recording and expertise evolution."""

    def test_record_performance(self, persona_manager):
        """Test recording performance updates expertise."""
        persona_manager.create_persona(
            agent_name="learner",
            expertise={"security": 0.5},
        )

        persona_manager.record_performance(
            agent_name="learner",
            domain="security",
            success=True,
            action="critique",
        )

        persona = persona_manager.get_persona("learner")
        assert persona is not None
        assert "security" in persona.expertise

    def test_record_performance_new_domain(self, persona_manager):
        """Test recording performance in a new domain."""
        persona_manager.create_persona(agent_name="versatile")

        persona_manager.record_performance(
            agent_name="versatile",
            domain="testing",
            success=True,
            action="proposal",
        )

        persona = persona_manager.get_persona("versatile")
        assert persona is not None
        assert "testing" in persona.expertise

    def test_record_performance_invalid_domain_ignored(self, persona_manager):
        """Test that invalid domains are silently ignored."""
        persona_manager.create_persona(agent_name="ignored")

        # This should not raise
        persona_manager.record_performance(
            agent_name="ignored",
            domain="invalid_domain",
            success=True,
        )

        # Expertise should not include invalid domain
        persona = persona_manager.get_persona("ignored")
        assert "invalid_domain" not in persona.expertise

    def test_record_performance_with_debate_id(self, persona_manager):
        """Test recording performance with debate ID."""
        persona_manager.create_persona(agent_name="debater")

        persona_manager.record_performance(
            agent_name="debater",
            domain="architecture",
            success=True,
            action="critique",
            debate_id="debate_123",
        )

        persona = persona_manager.get_persona("debater")
        assert "architecture" in persona.expertise

    def test_record_multiple_performances(self, persona_manager):
        """Test recording multiple performances updates expertise correctly."""
        persona_manager.create_persona(agent_name="experienced")

        # Record multiple successes
        for _ in range(10):
            persona_manager.record_performance(
                agent_name="experienced",
                domain="security",
                success=True,
            )

        persona = persona_manager.get_persona("experienced")
        # After many successes, expertise should be high
        assert persona.expertise["security"] > 0.5

    def test_record_performance_failures_lower_score(self, persona_manager):
        """Test that failures lower the expertise score."""
        persona_manager.create_persona(
            agent_name="failing",
            expertise={"security": 0.8},
        )

        # Record multiple failures
        for _ in range(10):
            persona_manager.record_performance(
                agent_name="failing",
                domain="security",
                success=False,
            )

        persona = persona_manager.get_persona("failing")
        # After many failures, expertise should decrease
        assert persona.expertise["security"] < 0.8


class TestPerformanceSummary:
    """Tests for get_performance_summary."""

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
        assert "security" in summary
        assert summary["security"]["total"] == 1
        assert summary["security"]["successes"] == 1
        assert summary["security"]["rate"] == 1.0

    def test_get_performance_summary_multiple_domains(self, persona_manager):
        """Test performance summary with multiple domains."""
        persona_manager.create_persona(agent_name="multi_domain")

        persona_manager.record_performance(
            agent_name="multi_domain", domain="security", success=True
        )
        persona_manager.record_performance(
            agent_name="multi_domain", domain="testing", success=False
        )
        persona_manager.record_performance(
            agent_name="multi_domain", domain="testing", success=True
        )

        summary = persona_manager.get_performance_summary("multi_domain")

        assert "security" in summary
        assert "testing" in summary
        assert summary["security"]["total"] == 1
        assert summary["testing"]["total"] == 2
        assert summary["testing"]["rate"] == 0.5

    def test_get_performance_summary_empty(self, persona_manager):
        """Test performance summary for agent with no history."""
        summary = persona_manager.get_performance_summary("no_history")

        assert summary == {}


# =============================================================================
# Trait Inference Tests
# =============================================================================


class TestTraitInference:
    """Tests for trait inference."""

    def test_infer_traits(self, persona_manager):
        """Test inferring traits based on performance."""
        persona_manager.create_persona(agent_name="inferable")

        for i in range(5):
            persona_manager.record_performance(
                agent_name="inferable",
                domain="security",
                success=True,
            )

        traits = persona_manager.infer_traits("inferable")

        assert isinstance(traits, list)

    def test_infer_traits_thorough_from_many_domains(self, persona_manager):
        """Test that covering many domains infers 'thorough' trait."""
        persona_manager.create_persona(agent_name="thorough_agent")

        domains = ["security", "testing", "architecture", "performance", "api_design"]
        for domain in domains:
            persona_manager.record_performance(
                agent_name="thorough_agent",
                domain=domain,
                success=True,
            )

        traits = persona_manager.infer_traits("thorough_agent")

        assert "thorough" in traits

    def test_infer_traits_pragmatic_from_high_success(self, persona_manager):
        """Test that high success rate infers 'pragmatic' trait."""
        persona_manager.create_persona(agent_name="pragmatic_agent")

        # 80% success rate
        for _ in range(8):
            persona_manager.record_performance(
                agent_name="pragmatic_agent",
                domain="security",
                success=True,
            )
        for _ in range(2):
            persona_manager.record_performance(
                agent_name="pragmatic_agent",
                domain="security",
                success=False,
            )

        traits = persona_manager.infer_traits("pragmatic_agent")

        assert "pragmatic" in traits

    def test_infer_traits_conservative_from_focused_domains(self, persona_manager):
        """Test that focused expertise infers 'conservative' trait."""
        persona_manager.create_persona(agent_name="focused_agent")

        # Many performances in just 2 domains
        for _ in range(10):
            persona_manager.record_performance(
                agent_name="focused_agent",
                domain="security",
                success=True,
            )

        traits = persona_manager.infer_traits("focused_agent")

        assert "conservative" in traits

    def test_infer_traits_no_history(self, persona_manager):
        """Test inferring traits with no performance history."""
        persona_manager.create_persona(agent_name="no_history")

        traits = persona_manager.infer_traits("no_history")

        assert traits == []


# =============================================================================
# Expertise Domains Tests
# =============================================================================


class TestExpertiseDomains:
    """Tests for expertise domain constants."""

    def test_expertise_domains_exist(self):
        """Test that expertise domains are defined."""
        assert len(EXPERTISE_DOMAINS) > 0

    def test_expertise_domains_include_technical(self):
        """Test that technical domains are included."""
        assert "security" in EXPERTISE_DOMAINS
        assert "performance" in EXPERTISE_DOMAINS
        assert "architecture" in EXPERTISE_DOMAINS
        assert "testing" in EXPERTISE_DOMAINS
        assert "error_handling" in EXPERTISE_DOMAINS
        assert "concurrency" in EXPERTISE_DOMAINS
        assert "api_design" in EXPERTISE_DOMAINS
        assert "database" in EXPERTISE_DOMAINS
        assert "frontend" in EXPERTISE_DOMAINS
        assert "devops" in EXPERTISE_DOMAINS
        assert "documentation" in EXPERTISE_DOMAINS
        assert "code_style" in EXPERTISE_DOMAINS

    def test_expertise_domains_include_compliance(self):
        """Test that compliance domains are included."""
        assert "sox_compliance" in EXPERTISE_DOMAINS
        assert "pci_dss" in EXPERTISE_DOMAINS
        assert "hipaa" in EXPERTISE_DOMAINS
        assert "gdpr" in EXPERTISE_DOMAINS
        assert "fda_21_cfr" in EXPERTISE_DOMAINS
        assert "fisma" in EXPERTISE_DOMAINS
        assert "nist_800_53" in EXPERTISE_DOMAINS
        assert "finra" in EXPERTISE_DOMAINS
        assert "audit_trails" in EXPERTISE_DOMAINS
        assert "data_privacy" in EXPERTISE_DOMAINS
        assert "access_control" in EXPERTISE_DOMAINS
        assert "encryption" in EXPERTISE_DOMAINS

    def test_expertise_domains_include_industry_verticals(self):
        """Test that industry vertical domains are included."""
        assert "legal" in EXPERTISE_DOMAINS
        assert "clinical" in EXPERTISE_DOMAINS
        assert "financial" in EXPERTISE_DOMAINS
        assert "academic" in EXPERTISE_DOMAINS

    def test_expertise_domains_include_philosophical(self):
        """Test that philosophical domains are included."""
        assert "philosophy" in EXPERTISE_DOMAINS
        assert "ethics" in EXPERTISE_DOMAINS
        assert "theology" in EXPERTISE_DOMAINS
        assert "humanities" in EXPERTISE_DOMAINS
        assert "sociology" in EXPERTISE_DOMAINS
        assert "psychology" in EXPERTISE_DOMAINS

    def test_expertise_domains_all_strings(self):
        """Test that all domains are strings."""
        for domain in EXPERTISE_DOMAINS:
            assert isinstance(domain, str)
            assert len(domain) > 0


# =============================================================================
# Personality Traits Tests
# =============================================================================


class TestPersonalityTraits:
    """Tests for personality trait constants."""

    def test_personality_traits_exist(self):
        """Test that personality traits are defined."""
        assert len(PERSONALITY_TRAITS) > 0

    def test_personality_traits_include_common(self):
        """Test that common traits are included."""
        assert "thorough" in PERSONALITY_TRAITS
        assert "pragmatic" in PERSONALITY_TRAITS
        assert "innovative" in PERSONALITY_TRAITS
        assert "conservative" in PERSONALITY_TRAITS
        assert "diplomatic" in PERSONALITY_TRAITS
        assert "direct" in PERSONALITY_TRAITS
        assert "collaborative" in PERSONALITY_TRAITS
        assert "contrarian" in PERSONALITY_TRAITS

    def test_personality_traits_include_compliance(self):
        """Test that compliance-focused traits are included."""
        assert "regulatory" in PERSONALITY_TRAITS
        assert "risk_aware" in PERSONALITY_TRAITS
        assert "audit_minded" in PERSONALITY_TRAITS
        assert "procedural" in PERSONALITY_TRAITS

    def test_personality_traits_all_strings(self):
        """Test that all traits are strings."""
        for trait in PERSONALITY_TRAITS:
            assert isinstance(trait, str)
            assert len(trait) > 0


# =============================================================================
# Default Personas Tests
# =============================================================================


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

    def test_default_personas_include_core_agents(self):
        """Test that core agent personas are defined."""
        assert "claude" in DEFAULT_PERSONAS
        assert "codex" in DEFAULT_PERSONAS
        assert "gemini" in DEFAULT_PERSONAS
        assert "grok" in DEFAULT_PERSONAS

    def test_default_personas_include_compliance_agents(self):
        """Test that compliance-focused personas are defined."""
        assert "sox" in DEFAULT_PERSONAS
        assert "pci_dss" in DEFAULT_PERSONAS
        assert "hipaa" in DEFAULT_PERSONAS
        assert "gdpr" in DEFAULT_PERSONAS
        assert "fisma" in DEFAULT_PERSONAS
        assert "finra" in DEFAULT_PERSONAS

    def test_default_personas_include_industry_vertical_agents(self):
        """Test that industry vertical personas are defined."""
        assert "contract_analyst" in DEFAULT_PERSONAS
        assert "clinical_reviewer" in DEFAULT_PERSONAS
        assert "financial_auditor" in DEFAULT_PERSONAS
        assert "peer_reviewer" in DEFAULT_PERSONAS

    def test_claude_persona_properties(self):
        """Test Claude persona has expected properties."""
        claude = DEFAULT_PERSONAS["claude"]

        assert "thorough" in claude.traits
        assert "security" in claude.expertise
        assert claude.temperature < 0.7  # Conservative temperature

    def test_grok_persona_properties(self):
        """Test Grok persona has expected properties."""
        grok = DEFAULT_PERSONAS["grok"]

        assert "contrarian" in grok.traits
        assert grok.temperature > 0.7  # High temperature for creativity

    def test_compliance_personas_have_low_temperature(self):
        """Test that compliance personas have low temperature."""
        compliance_personas = ["sox", "pci_dss", "hipaa", "fisma", "finra"]

        for name in compliance_personas:
            persona = DEFAULT_PERSONAS[name]
            assert persona.temperature <= 0.5, f"{name} should have low temperature"

    def test_all_default_personas_have_valid_traits(self):
        """Test that all default personas have valid traits."""
        for name, persona in DEFAULT_PERSONAS.items():
            for trait in persona.traits:
                # Traits should be known traits (with some exceptions for custom traits)
                assert isinstance(trait, str), f"{name} has invalid trait type"

    def test_all_default_personas_have_normalized_expertise(self):
        """Test that all default personas have normalized expertise scores."""
        for name, persona in DEFAULT_PERSONAS.items():
            for domain, score in persona.expertise.items():
                assert 0.0 <= score <= 1.0, f"{name} has invalid expertise score for {domain}"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetOrCreatePersona:
    """Tests for get_or_create_persona function."""

    def test_get_or_create_persona_creates_new(self, temp_db):
        """Test get_or_create_persona creates new persona."""
        manager = PersonaManager(db_path=temp_db)

        persona = get_or_create_persona(manager, "brand_new")

        assert persona is not None
        assert persona.agent_name == "brand_new"

    def test_get_or_create_persona_returns_existing(self, temp_db):
        """Test get_or_create_persona returns existing persona."""
        manager = PersonaManager(db_path=temp_db)

        manager.create_persona(
            agent_name="existing",
            traits=["thorough"],
        )

        persona = get_or_create_persona(manager, "existing")

        assert persona is not None
        assert "thorough" in persona.traits

    def test_get_or_create_persona_uses_defaults(self, temp_db):
        """Test get_or_create_persona uses default persona when available."""
        manager = PersonaManager(db_path=temp_db)

        # "claude" is a default persona
        persona = get_or_create_persona(manager, "claude_critic")

        assert persona is not None
        # Should inherit from "claude" default
        assert len(persona.traits) > 0 or len(persona.expertise) > 0

    def test_get_or_create_persona_empty_for_unknown(self, temp_db):
        """Test get_or_create_persona creates empty persona for unknown agent."""
        manager = PersonaManager(db_path=temp_db)

        persona = get_or_create_persona(manager, "completely_unknown")

        assert persona is not None
        assert persona.agent_name == "completely_unknown"


class TestApplyPersonaToAgent:
    """Tests for apply_persona_to_agent function."""

    def test_apply_persona_from_defaults(self):
        """Test applying a default persona to an agent."""
        agent = Mock()
        agent.system_prompt = ""

        result = apply_persona_to_agent(agent, "claude")

        assert result is True

    def test_apply_persona_sets_system_prompt(self):
        """Test that apply_persona sets system prompt."""
        agent = Mock()
        agent.system_prompt = ""

        apply_persona_to_agent(agent, "claude")

        # system_prompt should be modified
        assert agent.system_prompt != ""

    def test_apply_persona_sets_generation_params(self):
        """Test that apply_persona sets generation parameters."""
        agent = Mock()
        agent.system_prompt = ""
        agent.temperature = 0.7
        agent.top_p = 1.0
        agent.frequency_penalty = 0.0

        apply_persona_to_agent(agent, "grok")

        # Grok has high temperature
        assert agent.temperature == 0.9

    def test_apply_persona_uses_set_generation_params_method(self):
        """Test that apply_persona uses set_generation_params if available."""
        agent = Mock()
        agent.system_prompt = ""
        agent.set_generation_params = MagicMock()

        apply_persona_to_agent(agent, "claude")

        agent.set_generation_params.assert_called_once()

    def test_apply_persona_not_found(self):
        """Test applying a non-existent persona returns False."""
        agent = Mock()
        agent.system_prompt = ""

        result = apply_persona_to_agent(agent, "nonexistent_persona")

        assert result is False

    def test_apply_persona_with_manager(self, temp_db):
        """Test applying persona with database manager."""
        manager = PersonaManager(db_path=temp_db)
        manager.create_persona(
            agent_name="custom",
            description="A custom persona",
            traits=["thorough"],
        )

        agent = Mock()
        agent.system_prompt = ""

        result = apply_persona_to_agent(agent, "custom", manager=manager)

        assert result is True


class TestGetPersonaPrompt:
    """Tests for get_persona_prompt function."""

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
        assert len(prompt) > 0

    def test_get_persona_prompt_from_defaults(self):
        """Test getting prompt from default persona."""
        prompt = get_persona_prompt("claude")

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_persona_prompt_not_found(self):
        """Test getting prompt for non-existent persona returns empty."""
        prompt = get_persona_prompt("nonexistent")

        assert prompt == ""

    def test_get_persona_prompt_contains_expertise(self, temp_db):
        """Test that prompt contains expertise information."""
        manager = PersonaManager(db_path=temp_db)
        manager.create_persona(
            agent_name="expert",
            expertise={"security": 0.95},
        )

        prompt = get_persona_prompt("expert", manager=manager)

        assert "security" in prompt.lower()

    def test_get_persona_prompt_contains_traits(self, temp_db):
        """Test that prompt contains trait information."""
        manager = PersonaManager(db_path=temp_db)
        manager.create_persona(
            agent_name="traited",
            traits=["thorough", "pragmatic"],
        )

        prompt = get_persona_prompt("traited", manager=manager)

        assert "thorough" in prompt or "pragmatic" in prompt


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_persona_with_empty_name(self):
        """Test creating persona with empty name."""
        persona = Persona(agent_name="")
        assert persona.agent_name == ""

    def test_persona_with_unicode_name(self):
        """Test creating persona with unicode characters in name."""
        persona = Persona(agent_name="agent_")
        assert persona.agent_name == "agent_"

    def test_persona_with_very_long_description(self):
        """Test creating persona with very long description."""
        long_desc = "A" * 10000
        persona = Persona(agent_name="long_desc", description=long_desc)
        assert len(persona.description) == 10000

    def test_persona_manager_concurrent_access(self, temp_db):
        """Test that PersonaManager handles concurrent-like access."""
        manager1 = PersonaManager(db_path=temp_db)
        manager2 = PersonaManager(db_path=temp_db)

        manager1.create_persona(agent_name="concurrent")
        persona = manager2.get_persona("concurrent")

        assert persona is not None

    def test_expertise_score_zero(self, persona_manager):
        """Test handling expertise score of exactly zero."""
        persona = persona_manager.create_persona(
            agent_name="zero_expertise",
            expertise={"security": 0.0},
        )

        assert persona.expertise["security"] == 0.0

    def test_expertise_score_one(self, persona_manager):
        """Test handling expertise score of exactly one."""
        persona = persona_manager.create_persona(
            agent_name="max_expertise",
            expertise={"security": 1.0},
        )

        assert persona.expertise["security"] == 1.0

    def test_many_expertise_domains(self, persona_manager):
        """Test persona with many expertise domains."""
        expertise = {domain: 0.5 for domain in EXPERTISE_DOMAINS[:15]}
        persona = persona_manager.create_persona(
            agent_name="multi_expert",
            expertise=expertise,
        )

        # top_expertise should still return only 3
        assert len(persona.top_expertise) == 3

    def test_all_traits(self, persona_manager):
        """Test persona with all valid traits."""
        persona = persona_manager.create_persona(
            agent_name="all_traits",
            traits=PERSONALITY_TRAITS.copy(),
        )

        assert len(persona.traits) == len(PERSONALITY_TRAITS)


class TestPersonaTemperatureProfiles:
    """Tests for temperature profiles across persona types."""

    def test_conservative_personas_low_temperature(self):
        """Test that conservative personas have lower temperature."""
        conservative = ["sox", "hipaa", "pci_dss", "fisma", "finra"]

        for name in conservative:
            if name in DEFAULT_PERSONAS:
                assert DEFAULT_PERSONAS[name].temperature <= 0.5

    def test_innovative_personas_high_temperature(self):
        """Test that innovative personas have higher temperature."""
        innovative = ["grok", "lateral"]

        for name in innovative:
            if name in DEFAULT_PERSONAS:
                assert DEFAULT_PERSONAS[name].temperature >= 0.8

    def test_balanced_personas_default_temperature(self):
        """Test that balanced personas have default temperature around 0.7."""
        balanced = ["gemini", "deepseek"]

        for name in balanced:
            if name in DEFAULT_PERSONAS:
                assert 0.6 <= DEFAULT_PERSONAS[name].temperature <= 0.75


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """Test that all exported items are importable."""
        from aragora.agents import personas

        expected_exports = [
            "EXPERTISE_DOMAINS",
            "PERSONALITY_TRAITS",
            "Persona",
            "PersonaManager",
            "DEFAULT_PERSONAS",
            "get_or_create_persona",
            "apply_persona_to_agent",
            "get_persona_prompt",
        ]

        for name in expected_exports:
            assert hasattr(personas, name), f"{name} not found in module"

    def test_exports_match_all(self):
        """Test that exports match __all__."""
        from aragora.agents.personas import __all__

        assert "EXPERTISE_DOMAINS" in __all__
        assert "PERSONALITY_TRAITS" in __all__
        assert "Persona" in __all__
        assert "PersonaManager" in __all__
        assert "DEFAULT_PERSONAS" in __all__
        assert "get_or_create_persona" in __all__
        assert "apply_persona_to_agent" in __all__
        assert "get_persona_prompt" in __all__
