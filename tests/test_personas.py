"""Tests for the PersonaManager module."""

import json
import os
import sqlite3
import tempfile

import pytest

from aragora.agents.personas import (
    DEFAULT_PERSONAS,
    EXPERTISE_DOMAINS,
    PERSONALITY_TRAITS,
    Persona,
    PersonaManager,
    get_or_create_persona,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def db_path(temp_dir):
    """Path to temporary database."""
    return os.path.join(temp_dir, "personas.db")


@pytest.fixture
def manager(db_path):
    """Create PersonaManager with temp DB."""
    return PersonaManager(db_path=db_path)


# =============================================================================
# Test Persona Data Class
# =============================================================================


class TestPersona:
    """Tests for Persona dataclass properties and methods."""

    def test_creation_with_defaults(self):
        """Should create persona with default values."""
        persona = Persona(agent_name="test-agent")
        assert persona.agent_name == "test-agent"
        assert persona.description == ""
        assert persona.traits == []
        assert persona.expertise == {}
        assert persona.created_at is not None
        assert persona.updated_at is not None

    def test_creation_with_all_fields(self):
        """Should create persona with all fields populated."""
        persona = Persona(
            agent_name="test-agent",
            description="A test agent",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.8, "testing": 0.6},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00",
        )
        assert persona.agent_name == "test-agent"
        assert persona.description == "A test agent"
        assert persona.traits == ["thorough", "pragmatic"]
        assert persona.expertise == {"security": 0.8, "testing": 0.6}
        assert persona.created_at == "2024-01-01T00:00:00"
        assert persona.updated_at == "2024-01-02T00:00:00"

    def test_top_expertise_sorted(self):
        """Should return expertise sorted by score descending."""
        persona = Persona(
            agent_name="test",
            expertise={"security": 0.5, "testing": 0.8, "architecture": 0.3},
        )
        top = persona.top_expertise
        assert len(top) == 3
        assert top[0] == ("testing", 0.8)
        assert top[1] == ("security", 0.5)
        assert top[2] == ("architecture", 0.3)

    def test_top_expertise_max_three(self):
        """Should return at most 3 expertise areas."""
        persona = Persona(
            agent_name="test",
            expertise={
                "security": 0.9,
                "testing": 0.8,
                "architecture": 0.7,
                "performance": 0.6,
                "database": 0.5,
            },
        )
        top = persona.top_expertise
        assert len(top) == 3
        assert top[0] == ("security", 0.9)
        assert top[1] == ("testing", 0.8)
        assert top[2] == ("architecture", 0.7)

    def test_top_expertise_empty(self):
        """Should return empty list when no expertise."""
        persona = Persona(agent_name="test")
        assert persona.top_expertise == []

    def test_trait_string_with_traits(self):
        """Should return comma-separated traits."""
        persona = Persona(agent_name="test", traits=["thorough", "pragmatic", "direct"])
        assert persona.trait_string == "thorough, pragmatic, direct"

    def test_trait_string_single_trait(self):
        """Should handle single trait."""
        persona = Persona(agent_name="test", traits=["thorough"])
        assert persona.trait_string == "thorough"

    def test_trait_string_empty_returns_balanced(self):
        """Should return 'balanced' when no traits."""
        persona = Persona(agent_name="test")
        assert persona.trait_string == "balanced"

    def test_to_prompt_context_full(self):
        """Should generate full prompt context."""
        persona = Persona(
            agent_name="test",
            description="A security expert",
            traits=["thorough", "conservative"],
            expertise={"security": 0.9, "testing": 0.7},
        )
        context = persona.to_prompt_context()
        assert "Your role: A security expert" in context
        assert "Your approach: thorough, conservative" in context
        assert "Your expertise areas:" in context
        assert "security (90%)" in context
        assert "testing (70%)" in context

    def test_to_prompt_context_empty(self):
        """Should return empty string for empty persona."""
        persona = Persona(agent_name="test")
        assert persona.to_prompt_context() == ""

    def test_to_prompt_context_partial_description_only(self):
        """Should generate context with description only."""
        persona = Persona(agent_name="test", description="A test agent")
        context = persona.to_prompt_context()
        assert context == "Your role: A test agent"

    def test_to_prompt_context_partial_traits_only(self):
        """Should generate context with traits only."""
        persona = Persona(agent_name="test", traits=["thorough"])
        context = persona.to_prompt_context()
        assert context == "Your approach: thorough"


# =============================================================================
# Test PersonaManager Initialization
# =============================================================================


class TestPersonaManagerInit:
    """Tests for PersonaManager initialization."""

    def test_creates_database_file(self, db_path):
        """Should create database file on init."""
        assert not os.path.exists(db_path)
        PersonaManager(db_path=db_path)
        assert os.path.exists(db_path)

    def test_creates_personas_table(self, db_path):
        """Should create personas table."""
        PersonaManager(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='personas'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_performance_history_table(self, db_path):
        """Should create performance_history table."""
        PersonaManager(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='performance_history'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_idempotent_init(self, db_path):
        """Should be safe to init multiple times."""
        manager1 = PersonaManager(db_path=db_path)
        manager1.create_persona("test-agent", description="Test")

        # Init again
        manager2 = PersonaManager(db_path=db_path)
        persona = manager2.get_persona("test-agent")
        assert persona is not None
        assert persona.description == "Test"


# =============================================================================
# Test Persona CRUD Operations
# =============================================================================


class TestPersonaCRUD:
    """Tests for persona create, read, update operations."""

    def test_get_persona_not_found(self, manager):
        """Should return None for non-existent persona."""
        result = manager.get_persona("non-existent")
        assert result is None

    def test_get_persona_found(self, manager):
        """Should return persona when found."""
        manager.create_persona("test-agent", description="Test Description")
        persona = manager.get_persona("test-agent")
        assert persona is not None
        assert persona.agent_name == "test-agent"
        assert persona.description == "Test Description"

    def test_create_persona_basic(self, manager):
        """Should create basic persona."""
        persona = manager.create_persona("test-agent")
        assert persona.agent_name == "test-agent"
        assert persona.description == ""
        assert persona.traits == []
        assert persona.expertise == {}

    def test_create_persona_with_description(self, manager):
        """Should create persona with description."""
        persona = manager.create_persona("test-agent", description="A helpful agent")
        assert persona.description == "A helpful agent"

    def test_create_persona_with_traits(self, manager):
        """Should create persona with valid traits."""
        persona = manager.create_persona("test-agent", traits=["thorough", "pragmatic"])
        assert persona.traits == ["thorough", "pragmatic"]

    def test_create_persona_with_expertise(self, manager):
        """Should create persona with valid expertise."""
        persona = manager.create_persona("test-agent", expertise={"security": 0.8, "testing": 0.6})
        assert persona.expertise == {"security": 0.8, "testing": 0.6}

    def test_create_persona_validates_traits(self, manager):
        """Should filter out invalid traits."""
        persona = manager.create_persona(
            "test-agent", traits=["thorough", "invalid_trait", "pragmatic"]
        )
        assert "thorough" in persona.traits
        assert "pragmatic" in persona.traits
        assert "invalid_trait" not in persona.traits

    def test_create_persona_validates_expertise_domains(self, manager):
        """Should filter out invalid expertise domains."""
        persona = manager.create_persona(
            "test-agent",
            expertise={"security": 0.8, "invalid_domain": 0.5, "testing": 0.6},
        )
        assert "security" in persona.expertise
        assert "testing" in persona.expertise
        assert "invalid_domain" not in persona.expertise

    def test_create_persona_clamps_expertise_values_high(self, manager):
        """Should clamp expertise values above 1.0."""
        persona = manager.create_persona("test-agent", expertise={"security": 1.5})
        assert persona.expertise["security"] == 1.0

    def test_create_persona_clamps_expertise_values_low(self, manager):
        """Should clamp expertise values below 0.0."""
        persona = manager.create_persona("test-agent", expertise={"security": -0.5})
        assert persona.expertise["security"] == 0.0

    def test_create_persona_upsert(self, manager):
        """Should update existing persona."""
        manager.create_persona("test-agent", description="Original")
        manager.create_persona("test-agent", description="Updated")
        persona = manager.get_persona("test-agent")
        assert persona.description == "Updated"

    def test_get_all_personas_empty(self, manager):
        """Should return empty list when no personas."""
        result = manager.get_all_personas()
        assert result == []

    def test_get_all_personas_multiple(self, manager):
        """Should return all personas."""
        manager.create_persona("agent1", description="First")
        manager.create_persona("agent2", description="Second")
        manager.create_persona("agent3", description="Third")

        personas = manager.get_all_personas()
        assert len(personas) == 3
        names = {p.agent_name for p in personas}
        assert names == {"agent1", "agent2", "agent3"}


# =============================================================================
# Test Performance Recording
# =============================================================================


class TestPerformanceRecording:
    """Tests for recording and updating performance."""

    def test_record_performance_success(self, manager, db_path):
        """Should record successful performance event."""
        manager.record_performance("test-agent", "security", success=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT agent_name, domain, success FROM performance_history")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "test-agent"
        assert row[1] == "security"
        assert row[2] == 1

    def test_record_performance_failure(self, manager, db_path):
        """Should record failed performance event."""
        manager.record_performance("test-agent", "testing", success=False)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT success FROM performance_history")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 0

    def test_record_performance_with_action_and_debate_id(self, manager, db_path):
        """Should record action type and debate ID."""
        manager.record_performance(
            "test-agent",
            "architecture",
            success=True,
            action="proposal",
            debate_id="debate-123",
        )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT action, debate_id FROM performance_history")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "proposal"
        assert row[1] == "debate-123"

    def test_record_performance_invalid_domain_ignored(self, manager, db_path):
        """Should ignore invalid domain."""
        manager.record_performance("test-agent", "invalid_domain", success=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM performance_history")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_record_performance_updates_expertise(self, manager):
        """Should update expertise based on performance."""
        # Record some successful events
        for _ in range(5):
            manager.record_performance("test-agent", "security", success=True)

        persona = manager.get_persona("test-agent")
        assert persona is not None
        assert "security" in persona.expertise
        assert persona.expertise["security"] > 0.5  # Success should increase expertise

    def test_expertise_weighted_by_recency(self, manager):
        """Recent events should have more weight than older events."""
        # Record only failures first
        for _ in range(5):
            manager.record_performance("test-agent", "testing", success=False)

        persona_after_failures = manager.get_persona("test-agent")
        expertise_after_failures = persona_after_failures.expertise.get("testing", 0)

        # Now record successes - should increase expertise
        for _ in range(5):
            manager.record_performance("test-agent", "testing", success=True)

        persona_after_successes = manager.get_persona("test-agent")
        expertise_after_successes = persona_after_successes.expertise.get("testing", 0)

        # Recent successes should increase expertise
        assert expertise_after_successes > expertise_after_failures


# =============================================================================
# Test Trait Inference
# =============================================================================


class TestTraitInference:
    """Tests for inferring traits from performance patterns."""

    def test_infer_traits_no_history(self, manager):
        """Should return empty list when no history."""
        traits = manager.infer_traits("test-agent")
        assert traits == []

    def test_infer_thorough_many_domains(self, manager):
        """Should infer 'thorough' when covering many domains."""
        # Record performance in 5+ domains
        domains = ["security", "testing", "architecture", "performance", "database"]
        for domain in domains:
            manager.record_performance("test-agent", domain, success=True)

        traits = manager.infer_traits("test-agent")
        assert "thorough" in traits

    def test_infer_pragmatic_high_success(self, manager):
        """Should infer 'pragmatic' with high success rate."""
        # Record many successes (>70%)
        for _ in range(10):
            manager.record_performance("test-agent", "security", success=True)
        for _ in range(2):
            manager.record_performance("test-agent", "security", success=False)

        traits = manager.infer_traits("test-agent")
        assert "pragmatic" in traits

    def test_infer_conservative_focused(self, manager):
        """Should infer 'conservative' when focused on few domains."""
        # Record 10+ events in only 1-2 domains
        for _ in range(10):
            manager.record_performance("test-agent", "security", success=True)

        traits = manager.infer_traits("test-agent")
        assert "conservative" in traits

    def test_infer_multiple_traits(self, manager):
        """Should infer multiple traits when patterns match."""
        # High success rate in few domains
        for _ in range(15):
            manager.record_performance("test-agent", "security", success=True)

        traits = manager.infer_traits("test-agent")
        assert "pragmatic" in traits  # High success
        assert "conservative" in traits  # Few domains


# =============================================================================
# Test Performance Summary
# =============================================================================


class TestPerformanceSummary:
    """Tests for performance statistics."""

    def test_summary_empty(self, manager):
        """Should return empty dict when no history."""
        summary = manager.get_performance_summary("test-agent")
        assert summary == {}

    def test_summary_single_domain(self, manager):
        """Should summarize single domain performance."""
        for _ in range(3):
            manager.record_performance("test-agent", "security", success=True)
        manager.record_performance("test-agent", "security", success=False)

        summary = manager.get_performance_summary("test-agent")
        assert "security" in summary
        assert summary["security"]["total"] == 4
        assert summary["security"]["successes"] == 3
        assert summary["security"]["rate"] == 0.75

    def test_summary_multiple_domains(self, manager):
        """Should summarize multiple domain performance."""
        manager.record_performance("test-agent", "security", success=True)
        manager.record_performance("test-agent", "testing", success=True)
        manager.record_performance("test-agent", "testing", success=False)

        summary = manager.get_performance_summary("test-agent")
        assert len(summary) == 2
        assert "security" in summary
        assert "testing" in summary

    def test_summary_calculates_rate(self, manager):
        """Should calculate success rate correctly."""
        for _ in range(7):
            manager.record_performance("test-agent", "architecture", success=True)
        for _ in range(3):
            manager.record_performance("test-agent", "architecture", success=False)

        summary = manager.get_performance_summary("test-agent")
        assert summary["architecture"]["rate"] == 0.7


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_expertise_domains_defined(self):
        """Should have 12 expertise domains."""
        assert len(EXPERTISE_DOMAINS) == 12
        assert "security" in EXPERTISE_DOMAINS
        assert "performance" in EXPERTISE_DOMAINS
        assert "architecture" in EXPERTISE_DOMAINS
        assert "testing" in EXPERTISE_DOMAINS

    def test_personality_traits_defined(self):
        """Should have 8 personality traits."""
        assert len(PERSONALITY_TRAITS) == 8
        assert "thorough" in PERSONALITY_TRAITS
        assert "pragmatic" in PERSONALITY_TRAITS
        assert "innovative" in PERSONALITY_TRAITS
        assert "conservative" in PERSONALITY_TRAITS


# =============================================================================
# Test Default Personas
# =============================================================================


class TestDefaultPersonas:
    """Tests for DEFAULT_PERSONAS constant."""

    def test_default_personas_defined(self):
        """Should have 6 default personas."""
        assert len(DEFAULT_PERSONAS) == 6
        assert "claude" in DEFAULT_PERSONAS
        assert "codex" in DEFAULT_PERSONAS
        assert "gemini" in DEFAULT_PERSONAS
        assert "grok" in DEFAULT_PERSONAS
        assert "qwen" in DEFAULT_PERSONAS
        assert "deepseek" in DEFAULT_PERSONAS

    def test_default_claude_persona(self):
        """Should have correct claude persona."""
        claude = DEFAULT_PERSONAS["claude"]
        assert claude.agent_name == "claude"
        assert "thorough" in claude.traits
        assert "diplomatic" in claude.traits
        assert "conservative" in claude.traits
        assert claude.expertise.get("security", 0) == 0.8
        assert claude.expertise.get("error_handling", 0) == 0.7

    def test_default_codex_persona(self):
        """Should have correct codex persona."""
        codex = DEFAULT_PERSONAS["codex"]
        assert codex.agent_name == "codex"
        assert "pragmatic" in codex.traits
        assert "direct" in codex.traits
        assert "innovative" in codex.traits
        assert codex.expertise.get("architecture", 0) == 0.7

    def test_default_personas_have_descriptions(self):
        """All default personas should have descriptions."""
        for name, persona in DEFAULT_PERSONAS.items():
            assert persona.description, f"{name} missing description"

    def test_default_personas_have_valid_traits(self):
        """All default persona traits should be valid."""
        for name, persona in DEFAULT_PERSONAS.items():
            for trait in persona.traits:
                assert trait in PERSONALITY_TRAITS, f"{name} has invalid trait: {trait}"

    def test_default_personas_have_valid_expertise(self):
        """All default persona expertise domains should be valid."""
        for name, persona in DEFAULT_PERSONAS.items():
            for domain in persona.expertise:
                assert domain in EXPERTISE_DOMAINS, f"{name} has invalid domain: {domain}"


# =============================================================================
# Test get_or_create_persona Helper
# =============================================================================


class TestGetOrCreatePersona:
    """Tests for get_or_create_persona helper function."""

    def test_returns_existing_persona(self, manager):
        """Should return existing persona if found."""
        manager.create_persona("test-agent", description="Existing")
        persona = get_or_create_persona(manager, "test-agent")
        assert persona.description == "Existing"

    def test_creates_from_default_exact_match(self, manager):
        """Should create from default persona for exact match."""
        persona = get_or_create_persona(manager, "claude")
        assert persona.agent_name == "claude"
        assert persona.description == DEFAULT_PERSONAS["claude"].description
        assert "thorough" in persona.traits

    def test_creates_from_default_with_suffix(self, manager):
        """Should create from default for suffixed agent name."""
        persona = get_or_create_persona(manager, "claude_critic")
        assert persona.agent_name == "claude_critic"
        assert persona.description == DEFAULT_PERSONAS["claude"].description
        assert "thorough" in persona.traits

    def test_creates_empty_for_unknown(self, manager):
        """Should create empty persona for unknown agent."""
        persona = get_or_create_persona(manager, "unknown-agent")
        assert persona.agent_name == "unknown-agent"
        assert persona.description == ""
        assert persona.traits == []
        assert persona.expertise == {}

    def test_persists_created_persona(self, manager):
        """Should persist newly created persona."""
        get_or_create_persona(manager, "claude")
        # Should now be found
        persona = manager.get_persona("claude")
        assert persona is not None
        assert "thorough" in persona.traits

    def test_uses_default_traits_copy(self, manager):
        """Should copy traits, not reference default."""
        persona = get_or_create_persona(manager, "claude")
        default_traits = DEFAULT_PERSONAS["claude"].traits
        # Modifying persona shouldn't affect default
        assert len(persona.traits) == len(default_traits)
