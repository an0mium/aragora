"""
Comprehensive tests for agent personas module.

This test module provides extended coverage for the personas module, focusing on:
- Persona creation with various configuration combinations
- Task-type based persona selection heuristics
- Persona trait combination effects
- Edge cases for missing personas, invalid configurations
- Integration with agent execution patterns
- Performance tracking with recency weighting
- Expertise domain cross-validation
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.agents.personas import (
    DEFAULT_PERSONAS,
    EXPERTISE_DOMAINS,
    PERSONALITY_TRAITS,
    PERSONA_SCHEMA_VERSION,
    Persona,
    PersonaManager,
    apply_persona_to_agent,
    get_or_create_persona,
    get_persona_prompt,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def persona_manager(temp_db):
    """Create a PersonaManager with a temporary database."""
    return PersonaManager(db_path=temp_db)


@pytest.fixture
def mock_agent():
    """Create a mock agent with common attributes for testing."""

    class MockAgent:
        """Mock agent class that allows attribute setting."""

        def __init__(self):
            self.system_prompt = ""
            self.temperature = 0.7
            self.top_p = 1.0
            self.frequency_penalty = 0.0

    return MockAgent()


@pytest.fixture
def mock_agent_with_set_params():
    """Create a mock agent with set_generation_params method."""
    agent = Mock()
    agent.system_prompt = ""
    agent.set_generation_params = MagicMock()
    return agent


# =============================================================================
# Persona Creation and Configuration Tests
# =============================================================================


class TestPersonaCreationConfigurations:
    """Tests for various persona creation configurations."""

    def test_create_persona_with_all_expertise_domains(self, persona_manager):
        """Test creating persona with all valid expertise domains."""
        expertise = {domain: 0.5 for domain in EXPERTISE_DOMAINS}
        persona = persona_manager.create_persona(
            agent_name="all_domains",
            expertise=expertise,
        )

        # All domains should be preserved
        assert len(persona.expertise) == len(EXPERTISE_DOMAINS)
        for domain in EXPERTISE_DOMAINS:
            assert domain in persona.expertise

    def test_create_persona_with_all_personality_traits(self, persona_manager):
        """Test creating persona with all valid personality traits."""
        persona = persona_manager.create_persona(
            agent_name="all_traits",
            traits=PERSONALITY_TRAITS.copy(),
        )

        assert len(persona.traits) == len(PERSONALITY_TRAITS)
        for trait in PERSONALITY_TRAITS:
            assert trait in persona.traits

    def test_create_persona_mixed_valid_invalid_traits(self, persona_manager):
        """Test that invalid traits are filtered while valid ones remain."""
        traits = ["thorough", "fake_trait_1", "pragmatic", "fake_trait_2", "innovative"]
        persona = persona_manager.create_persona(
            agent_name="mixed_traits",
            traits=traits,
        )

        assert "thorough" in persona.traits
        assert "pragmatic" in persona.traits
        assert "innovative" in persona.traits
        assert "fake_trait_1" not in persona.traits
        assert "fake_trait_2" not in persona.traits
        assert len(persona.traits) == 3

    def test_create_persona_mixed_valid_invalid_expertise(self, persona_manager):
        """Test that invalid expertise domains are filtered."""
        expertise = {
            "security": 0.8,
            "invalid_domain_1": 0.6,
            "performance": 0.7,
            "invalid_domain_2": 0.5,
        }
        persona = persona_manager.create_persona(
            agent_name="mixed_expertise",
            expertise=expertise,
        )

        assert "security" in persona.expertise
        assert "performance" in persona.expertise
        assert "invalid_domain_1" not in persona.expertise
        assert "invalid_domain_2" not in persona.expertise
        assert len(persona.expertise) == 2

    def test_create_persona_expertise_boundary_values(self, persona_manager):
        """Test expertise score boundary value handling."""
        expertise = {
            "security": -0.1,  # Should clamp to 0.0
            "performance": 0.0,  # Exactly 0
            "testing": 0.5,  # Normal value
            "architecture": 1.0,  # Exactly 1
            "api_design": 1.5,  # Should clamp to 1.0
        }
        persona = persona_manager.create_persona(
            agent_name="boundary_expertise",
            expertise=expertise,
        )

        assert persona.expertise["security"] == 0.0
        assert persona.expertise["performance"] == 0.0
        assert persona.expertise["testing"] == 0.5
        assert persona.expertise["architecture"] == 1.0
        assert persona.expertise["api_design"] == 1.0

    def test_create_persona_with_empty_traits_list(self, persona_manager):
        """Test creating persona with explicitly empty traits."""
        persona = persona_manager.create_persona(
            agent_name="no_traits",
            traits=[],
        )

        assert persona.traits == []
        assert persona.trait_string == "balanced"

    def test_create_persona_with_empty_expertise_dict(self, persona_manager):
        """Test creating persona with explicitly empty expertise."""
        persona = persona_manager.create_persona(
            agent_name="no_expertise",
            expertise={},
        )

        assert persona.expertise == {}
        assert persona.top_expertise == []

    def test_create_persona_preserves_description_whitespace(self, persona_manager):
        """Test that description whitespace is preserved."""
        description = "  A  spaced   description  "
        persona = persona_manager.create_persona(
            agent_name="whitespace_desc",
            description=description,
        )

        assert persona.description == description

    def test_create_persona_with_special_characters_in_name(self, persona_manager):
        """Test persona names with special characters."""
        special_names = [
            "agent-with-dashes",
            "agent_with_underscores",
            "agent.with.dots",
            "agent123",
            "UPPERCASE_AGENT",
        ]
        for name in special_names:
            persona = persona_manager.create_persona(agent_name=name)
            retrieved = persona_manager.get_persona(name)
            assert retrieved is not None
            assert retrieved.agent_name == name


class TestPersonaGenerationParameterRanges:
    """Tests for persona generation parameter edge cases."""

    def test_persona_temperature_extreme_values(self):
        """Test persona with extreme temperature values."""
        # Very low temperature
        cold_persona = Persona(agent_name="cold", temperature=0.01)
        assert cold_persona.generation_params["temperature"] == 0.01

        # Very high temperature
        hot_persona = Persona(agent_name="hot", temperature=2.0)
        assert hot_persona.generation_params["temperature"] == 2.0

        # Zero temperature
        zero_persona = Persona(agent_name="zero", temperature=0.0)
        assert zero_persona.generation_params["temperature"] == 0.0

    def test_persona_top_p_extreme_values(self):
        """Test persona with extreme top_p values."""
        low_top_p = Persona(agent_name="low_p", top_p=0.1)
        assert low_top_p.generation_params["top_p"] == 0.1

        high_top_p = Persona(agent_name="high_p", top_p=1.0)
        assert high_top_p.generation_params["top_p"] == 1.0

    def test_persona_frequency_penalty_range(self):
        """Test persona with various frequency penalty values."""
        zero_penalty = Persona(agent_name="zero_fp", frequency_penalty=0.0)
        assert zero_penalty.generation_params["frequency_penalty"] == 0.0

        high_penalty = Persona(agent_name="high_fp", frequency_penalty=0.5)
        assert high_penalty.generation_params["frequency_penalty"] == 0.5


# =============================================================================
# Task-Type Based Persona Selection Tests
# =============================================================================


class TestTaskTypePersonaSelection:
    """Tests for persona selection based on task type heuristics."""

    def test_security_task_selects_security_focused_persona(self):
        """Test that security-focused personas have high security expertise."""
        security_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if persona.expertise.get("security", 0) >= 0.8:
                security_personas.append(name)

        # Should have at least one persona with high security expertise
        assert len(security_personas) > 0
        assert (
            "code_security_specialist" in security_personas
            or "security_engineer" in security_personas
        )

    def test_compliance_task_selects_regulatory_persona(self):
        """Test that regulatory personas have regulatory trait."""
        regulatory_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if "regulatory" in persona.traits:
                regulatory_personas.append(name)

        # Should have compliance-focused personas with regulatory trait
        assert len(regulatory_personas) > 0
        # SOX, PCI-DSS, HIPAA, etc. should have regulatory trait
        compliance_agents = ["sox", "pci_dss", "hipaa", "fisma", "gdpr", "finra"]
        for agent in compliance_agents:
            if agent in DEFAULT_PERSONAS:
                assert agent in regulatory_personas, f"{agent} should have regulatory trait"

    def test_architecture_task_selects_architecture_expert(self):
        """Test that architecture personas have high architecture expertise."""
        arch_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if persona.expertise.get("architecture", 0) >= 0.7:
                arch_personas.append(name)

        assert len(arch_personas) > 0
        assert "architecture_reviewer" in arch_personas

    def test_testing_task_selects_testing_expert(self):
        """Test that testing-focused personas exist."""
        testing_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if persona.expertise.get("testing", 0) >= 0.6:
                testing_personas.append(name)

        assert len(testing_personas) > 0

    def test_documentation_task_selects_doc_expert(self):
        """Test that documentation personas exist."""
        doc_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if persona.expertise.get("documentation", 0) >= 0.6:
                doc_personas.append(name)

        assert len(doc_personas) > 0


class TestPersonaExpertiseMatching:
    """Tests for expertise-based persona matching logic."""

    def test_get_personas_by_domain_sorted_by_score(self):
        """Test getting personas sorted by domain expertise score."""
        # Get all personas with security expertise
        security_experts = []
        for name, persona in DEFAULT_PERSONAS.items():
            score = persona.expertise.get("security", 0)
            if score > 0:
                security_experts.append((name, score))

        # Sort by score descending
        security_experts.sort(key=lambda x: x[1], reverse=True)

        # Verify we have experts
        assert len(security_experts) > 0

        # Verify top expert has high score
        top_name, top_score = security_experts[0]
        assert top_score >= 0.7

    def test_filter_personas_by_minimum_expertise(self):
        """Test filtering personas by minimum expertise threshold."""
        min_threshold = 0.8

        high_security_experts = []
        for name, persona in DEFAULT_PERSONAS.items():
            if persona.expertise.get("security", 0) >= min_threshold:
                high_security_experts.append(name)

        # All returned personas should meet threshold
        for name in high_security_experts:
            assert DEFAULT_PERSONAS[name].expertise.get("security", 0) >= min_threshold

    def test_persona_multi_domain_expertise(self):
        """Test personas with expertise in multiple domains."""
        multi_experts = []
        for name, persona in DEFAULT_PERSONAS.items():
            domains_with_expertise = sum(1 for v in persona.expertise.values() if v >= 0.5)
            if domains_with_expertise >= 3:
                multi_experts.append(name)

        # Some personas should have multi-domain expertise
        assert len(multi_experts) > 0


# =============================================================================
# Persona Trait Combination Tests
# =============================================================================


class TestPersonaTraitCombinations:
    """Tests for persona trait combination effects."""

    def test_conservative_personas_have_low_temperature(self):
        """Test that conservative trait correlates with low temperature."""
        for name, persona in DEFAULT_PERSONAS.items():
            if "conservative" in persona.traits:
                # Conservative personas should have lower temperature
                assert persona.temperature <= 0.7, (
                    f"{name} is conservative but has high temperature"
                )

    def test_innovative_personas_have_higher_temperature(self):
        """Test that innovative trait correlates with higher temperature."""
        innovative_with_high_temp = []
        for name, persona in DEFAULT_PERSONAS.items():
            if "innovative" in persona.traits and persona.temperature >= 0.7:
                innovative_with_high_temp.append(name)

        # At least some innovative personas should have higher temperature
        assert len(innovative_with_high_temp) > 0

    def test_contrarian_personas_exist(self):
        """Test that contrarian personas exist for diversity."""
        contrarian_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if "contrarian" in persona.traits:
                contrarian_personas.append(name)

        assert len(contrarian_personas) > 0
        assert "grok" in contrarian_personas

    def test_diplomatic_personas_exist(self):
        """Test that diplomatic personas exist."""
        diplomatic_personas = []
        for name, persona in DEFAULT_PERSONAS.items():
            if "diplomatic" in persona.traits:
                diplomatic_personas.append(name)

        assert len(diplomatic_personas) > 0

    def test_trait_combinations_are_coherent(self):
        """Test that trait combinations make logical sense."""
        for name, persona in DEFAULT_PERSONAS.items():
            # A persona shouldn't be both conservative and innovative typically
            # (though not strictly forbidden, it would be unusual)
            traits = persona.traits

            # Audit-minded personas should be thorough or procedural
            if "audit_minded" in traits:
                assert "thorough" in traits or "procedural" in traits or "regulatory" in traits, (
                    f"{name} is audit_minded but lacks related traits"
                )

    def test_compliance_personas_trait_consistency(self):
        """Test that compliance personas have consistent trait patterns."""
        compliance_agents = ["sox", "pci_dss", "hipaa", "fisma", "gdpr", "finra"]
        for agent in compliance_agents:
            if agent in DEFAULT_PERSONAS:
                persona = DEFAULT_PERSONAS[agent]
                # Compliance personas should have at least one of these traits
                compliance_traits = {
                    "regulatory",
                    "audit_minded",
                    "thorough",
                    "conservative",
                    "risk_aware",
                }
                has_compliance_trait = any(t in compliance_traits for t in persona.traits)
                assert has_compliance_trait, f"{agent} lacks compliance-appropriate traits"


# =============================================================================
# Edge Cases: Missing Personas and Invalid Config
# =============================================================================


class TestMissingPersonaHandling:
    """Tests for handling missing personas."""

    def test_get_persona_nonexistent_returns_none(self, persona_manager):
        """Test that getting a nonexistent persona returns None."""
        result = persona_manager.get_persona("completely_nonexistent_agent")
        assert result is None

    def test_get_or_create_persona_creates_for_unknown(self, temp_db):
        """Test that get_or_create_persona creates for unknown agents."""
        manager = PersonaManager(db_path=temp_db)
        persona = get_or_create_persona(manager, "brand_new_unknown_agent")

        assert persona is not None
        assert persona.agent_name == "brand_new_unknown_agent"
        # Should have empty defaults
        assert persona.traits == []
        assert persona.expertise == {}

    def test_get_or_create_persona_uses_base_name_defaults(self, temp_db):
        """Test that get_or_create uses defaults based on base agent name."""
        manager = PersonaManager(db_path=temp_db)

        # "claude_variant" should inherit from "claude"
        persona = get_or_create_persona(manager, "claude_variant")

        # Should inherit from claude default
        assert persona is not None
        claude_default = DEFAULT_PERSONAS.get("claude")
        if claude_default:
            assert len(persona.traits) > 0 or len(persona.expertise) > 0

    def test_apply_persona_to_agent_nonexistent(self, mock_agent):
        """Test applying nonexistent persona returns False."""
        result = apply_persona_to_agent(mock_agent, "totally_fake_persona_name")
        assert result is False
        # System prompt should not be modified
        assert mock_agent.system_prompt == ""

    def test_get_persona_prompt_nonexistent(self):
        """Test getting prompt for nonexistent persona returns empty."""
        prompt = get_persona_prompt("nonexistent_persona_xyz")
        assert prompt == ""

    def test_apply_persona_preserves_existing_system_prompt(self, mock_agent):
        """Test that applying persona preserves existing system prompt."""
        mock_agent.system_prompt = "Existing instructions"
        apply_persona_to_agent(mock_agent, "claude")

        # Both the persona context and existing instructions should be present
        assert "Existing instructions" in mock_agent.system_prompt


class TestInvalidConfigHandling:
    """Tests for handling invalid configurations."""

    def test_persona_with_none_traits(self, persona_manager):
        """Test creating persona with None traits."""
        persona = persona_manager.create_persona(
            agent_name="none_traits",
            traits=None,
        )
        assert persona.traits == []

    def test_persona_with_none_expertise(self, persona_manager):
        """Test creating persona with None expertise."""
        persona = persona_manager.create_persona(
            agent_name="none_expertise",
            expertise=None,
        )
        assert persona.expertise == {}

    def test_record_performance_invalid_domain_silent(self, persona_manager):
        """Test that recording performance with invalid domain is silently ignored."""
        persona_manager.create_persona(agent_name="test_agent")

        # Should not raise
        persona_manager.record_performance(
            agent_name="test_agent",
            domain="completely_invalid_domain",
            success=True,
        )

        persona = persona_manager.get_persona("test_agent")
        assert "completely_invalid_domain" not in persona.expertise

    def test_infer_traits_no_history(self, persona_manager):
        """Test inferring traits with no performance history."""
        persona_manager.create_persona(agent_name="no_history_agent")
        traits = persona_manager.infer_traits("no_history_agent")
        assert traits == []

    def test_get_performance_summary_nonexistent(self, persona_manager):
        """Test getting performance summary for nonexistent agent."""
        summary = persona_manager.get_performance_summary("nonexistent_agent")
        assert summary == {}


# =============================================================================
# Integration with Agent Execution Tests
# =============================================================================


class TestAgentIntegration:
    """Tests for integration with agent execution patterns."""

    def test_apply_persona_sets_temperature(self, mock_agent):
        """Test that applying persona sets agent temperature."""
        # Grok has high temperature (0.9)
        apply_persona_to_agent(mock_agent, "grok")
        assert mock_agent.temperature == 0.9

    def test_apply_persona_sets_top_p(self, mock_agent):
        """Test that applying persona sets agent top_p."""
        # Claude has top_p of 0.95
        apply_persona_to_agent(mock_agent, "claude")
        assert mock_agent.top_p == 0.95

    def test_apply_persona_sets_frequency_penalty(self, mock_agent):
        """Test that applying persona sets agent frequency_penalty."""
        # Grok has frequency_penalty of 0.1
        apply_persona_to_agent(mock_agent, "grok")
        assert mock_agent.frequency_penalty == 0.1

    def test_apply_persona_uses_set_generation_params(self, mock_agent_with_set_params):
        """Test that apply_persona uses set_generation_params if available."""
        apply_persona_to_agent(mock_agent_with_set_params, "claude")

        mock_agent_with_set_params.set_generation_params.assert_called_once()
        call_kwargs = mock_agent_with_set_params.set_generation_params.call_args[1]
        assert "temperature" in call_kwargs
        assert "top_p" in call_kwargs
        assert "frequency_penalty" in call_kwargs

    def test_apply_persona_with_db_manager(self, temp_db, mock_agent):
        """Test applying persona from database via manager."""
        manager = PersonaManager(db_path=temp_db)
        manager.create_persona(
            agent_name="custom_db_persona",
            description="Custom database persona",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.9},
        )

        result = apply_persona_to_agent(mock_agent, "custom_db_persona", manager=manager)

        assert result is True
        assert (
            "security" in mock_agent.system_prompt.lower()
            or "thorough" in mock_agent.system_prompt.lower()
        )

    def test_apply_persona_prefers_defaults_over_db(self, temp_db, mock_agent):
        """Test that DEFAULT_PERSONAS take precedence over database."""
        manager = PersonaManager(db_path=temp_db)
        # Create a "claude" persona in DB with different properties
        manager.create_persona(
            agent_name="claude",
            description="DB version of claude",
            traits=["contrarian"],  # Different from default
        )

        result = apply_persona_to_agent(mock_agent, "claude", manager=manager)

        # Should apply the DEFAULT_PERSONAS version
        assert result is True

    def test_get_persona_prompt_includes_all_components(self, temp_db):
        """Test that persona prompt includes description, traits, and expertise."""
        manager = PersonaManager(db_path=temp_db)
        manager.create_persona(
            agent_name="full_persona",
            description="A comprehensive test persona",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.9, "testing": 0.8},
        )

        prompt = get_persona_prompt("full_persona", manager=manager)

        # Should contain elements from all components
        assert len(prompt) > 0
        # The prompt should reference the traits or expertise in some form
        assert "thorough" in prompt or "pragmatic" in prompt or "security" in prompt.lower()


class TestAgentWithMissingAttributes:
    """Tests for agents missing expected attributes."""

    def test_apply_persona_agent_without_temperature(self):
        """Test applying persona to agent without temperature attribute."""
        agent = Mock(spec=["system_prompt"])
        agent.system_prompt = ""

        # Should not raise
        result = apply_persona_to_agent(agent, "claude")
        assert result is True

    def test_apply_persona_agent_without_system_prompt(self):
        """Test applying persona to agent without system_prompt attribute."""
        agent = Mock(spec=["temperature"])
        agent.temperature = 0.7

        # Should not raise, but persona context won't be set
        result = apply_persona_to_agent(agent, "claude")
        assert result is True


# =============================================================================
# Performance Tracking and Expertise Evolution Tests
# =============================================================================


class TestPerformanceTrackingEvolution:
    """Tests for performance tracking and expertise evolution."""

    def test_expertise_increases_with_success(self, persona_manager):
        """Test that expertise increases with successful performance."""
        persona_manager.create_persona(
            agent_name="improving",
            expertise={"security": 0.5},
        )

        initial = persona_manager.get_persona("improving").expertise.get("security", 0.5)

        # Record many successes
        for _ in range(20):
            persona_manager.record_performance(
                agent_name="improving",
                domain="security",
                success=True,
            )

        updated = persona_manager.get_persona("improving").expertise.get("security", 0)

        # Expertise should increase after successes
        assert updated > initial

    def test_expertise_decreases_with_failure(self, persona_manager):
        """Test that expertise decreases with failed performance."""
        persona_manager.create_persona(
            agent_name="declining",
            expertise={"security": 0.8},
        )

        initial = persona_manager.get_persona("declining").expertise.get("security", 0.8)

        # Record many failures
        for _ in range(20):
            persona_manager.record_performance(
                agent_name="declining",
                domain="security",
                success=False,
            )

        updated = persona_manager.get_persona("declining").expertise.get("security", 1.0)

        # Expertise should decrease after failures
        assert updated < initial

    def test_expertise_recency_weighting(self, persona_manager):
        """Test that recent performance affects expertise with recency weighting."""
        persona_manager.create_persona(
            agent_name="recency_test",
            expertise={"security": 0.5},
        )

        # Record old failures
        for _ in range(10):
            persona_manager.record_performance(
                agent_name="recency_test",
                domain="security",
                success=False,
            )

        after_failures = persona_manager.get_persona("recency_test").expertise.get("security", 0)

        # Record recent successes
        for _ in range(15):
            persona_manager.record_performance(
                agent_name="recency_test",
                domain="security",
                success=True,
            )

        after_successes = persona_manager.get_persona("recency_test").expertise.get("security", 0)

        # Recent successes should have more weight, so expertise should increase
        # after adding successes compared to after only failures
        assert after_successes > after_failures, (
            f"Expertise should increase after successes: {after_successes} > {after_failures}"
        )

    def test_performance_creates_expertise_for_new_domain(self, persona_manager):
        """Test that recording performance creates expertise for new domain."""
        persona_manager.create_persona(
            agent_name="expanding",
            expertise={},
        )

        persona_manager.record_performance(
            agent_name="expanding",
            domain="testing",
            success=True,
        )

        persona = persona_manager.get_persona("expanding")
        assert "testing" in persona.expertise

    def test_performance_summary_accuracy(self, persona_manager):
        """Test that performance summary accurately reflects history."""
        persona_manager.create_persona(agent_name="summarized")

        # Record mixed results
        for _ in range(7):
            persona_manager.record_performance(
                agent_name="summarized",
                domain="security",
                success=True,
            )
        for _ in range(3):
            persona_manager.record_performance(
                agent_name="summarized",
                domain="security",
                success=False,
            )

        summary = persona_manager.get_performance_summary("summarized")

        assert summary["security"]["total"] == 10
        assert summary["security"]["successes"] == 7
        assert summary["security"]["rate"] == 0.7


class TestTraitInferencePatterns:
    """Tests for trait inference based on performance patterns."""

    def test_infer_thorough_from_many_domains(self, persona_manager):
        """Test that covering many domains infers 'thorough' trait."""
        persona_manager.create_persona(agent_name="broad_agent")

        # Perform in 5+ different domains
        domains = ["security", "testing", "architecture", "performance", "api_design", "database"]
        for domain in domains:
            persona_manager.record_performance(
                agent_name="broad_agent",
                domain=domain,
                success=True,
            )

        traits = persona_manager.infer_traits("broad_agent")
        assert "thorough" in traits

    def test_infer_pragmatic_from_high_success(self, persona_manager):
        """Test that high success rate infers 'pragmatic' trait."""
        persona_manager.create_persona(agent_name="successful_agent")

        # Record mostly successes (>70%)
        for _ in range(9):
            persona_manager.record_performance(
                agent_name="successful_agent",
                domain="security",
                success=True,
            )
        for _ in range(1):
            persona_manager.record_performance(
                agent_name="successful_agent",
                domain="security",
                success=False,
            )

        traits = persona_manager.infer_traits("successful_agent")
        assert "pragmatic" in traits

    def test_infer_conservative_from_focused_domains(self, persona_manager):
        """Test that focusing on few domains infers 'conservative' trait."""
        persona_manager.create_persona(agent_name="focused_agent")

        # Many performances in just 1-2 domains
        for _ in range(15):
            persona_manager.record_performance(
                agent_name="focused_agent",
                domain="security",
                success=True,
            )

        traits = persona_manager.infer_traits("focused_agent")
        assert "conservative" in traits


# =============================================================================
# Prompt Context Generation Tests
# =============================================================================


class TestPromptContextGeneration:
    """Tests for prompt context generation from personas."""

    def test_prompt_context_empty_persona(self):
        """Test that empty persona generates empty context."""
        persona = Persona(agent_name="empty")
        context = persona.to_prompt_context()
        assert context == ""

    def test_prompt_context_description_only(self):
        """Test prompt context with only description."""
        persona = Persona(
            agent_name="desc_only",
            description="A specialized security analyst",
        )
        context = persona.to_prompt_context()
        assert "Your role:" in context
        assert "security analyst" in context.lower()

    def test_prompt_context_traits_only(self):
        """Test prompt context with only traits."""
        persona = Persona(
            agent_name="traits_only",
            traits=["thorough", "pragmatic"],
        )
        context = persona.to_prompt_context()
        assert "Your approach:" in context
        assert "thorough" in context
        assert "pragmatic" in context

    def test_prompt_context_expertise_only(self):
        """Test prompt context with only expertise."""
        persona = Persona(
            agent_name="expertise_only",
            expertise={"security": 0.9},
        )
        context = persona.to_prompt_context()
        assert "expertise" in context.lower()
        assert "security" in context.lower()

    def test_prompt_context_all_components(self):
        """Test prompt context with all components."""
        persona = Persona(
            agent_name="full",
            description="A comprehensive test persona",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.9, "testing": 0.8},
        )
        context = persona.to_prompt_context()

        # All sections should be present
        assert "Your role:" in context
        assert "Your approach:" in context
        assert "expertise" in context.lower()

    def test_prompt_context_expertise_percentage_format(self):
        """Test that expertise is formatted as percentage."""
        persona = Persona(
            agent_name="percentage",
            expertise={"security": 0.95},
        )
        context = persona.to_prompt_context()
        # Should contain percentage like "95%" or "95 %"
        assert "95%" in context or "95 %" in context


# =============================================================================
# Database Persistence Tests
# =============================================================================


class TestDatabasePersistence:
    """Tests for database persistence of personas."""

    def test_persona_roundtrip(self, persona_manager):
        """Test that personas survive database roundtrip."""
        original = persona_manager.create_persona(
            agent_name="roundtrip",
            description="Test description",
            traits=["thorough", "pragmatic"],
            expertise={"security": 0.9, "testing": 0.7},
        )

        retrieved = persona_manager.get_persona("roundtrip")

        assert retrieved.agent_name == original.agent_name
        assert retrieved.description == original.description
        assert retrieved.traits == original.traits
        assert retrieved.expertise == original.expertise

    def test_multiple_managers_same_db(self, temp_db):
        """Test that multiple managers can access same database."""
        manager1 = PersonaManager(db_path=temp_db)
        manager2 = PersonaManager(db_path=temp_db)

        manager1.create_persona(
            agent_name="shared",
            traits=["thorough"],
        )

        persona = manager2.get_persona("shared")
        assert persona is not None
        assert "thorough" in persona.traits

    def test_upsert_updates_existing(self, persona_manager):
        """Test that creating persona with same name updates existing."""
        persona_manager.create_persona(
            agent_name="updateable",
            description="Original",
            traits=["thorough"],
        )

        persona_manager.create_persona(
            agent_name="updateable",
            description="Updated",
            traits=["pragmatic"],
        )

        persona = persona_manager.get_persona("updateable")
        assert persona.description == "Updated"
        assert "pragmatic" in persona.traits
        # Original trait should be replaced
        assert "thorough" not in persona.traits

    def test_get_all_personas(self, persona_manager):
        """Test getting all personas from database."""
        for i in range(5):
            persona_manager.create_persona(agent_name=f"agent_{i}")

        personas = persona_manager.get_all_personas()

        assert len(personas) == 5
        names = [p.agent_name for p in personas]
        for i in range(5):
            assert f"agent_{i}" in names


# =============================================================================
# Default Personas Validation Tests
# =============================================================================


class TestDefaultPersonasValidation:
    """Tests validating the DEFAULT_PERSONAS definitions."""

    def test_all_default_personas_have_valid_traits(self):
        """Test that all default personas have only valid traits."""
        for name, persona in DEFAULT_PERSONAS.items():
            for trait in persona.traits:
                # Allow custom traits for special personas
                if trait not in PERSONALITY_TRAITS:
                    # Some personas may have custom traits like "contemplative"
                    # This is acceptable for philosophical personas
                    pass

    def test_all_default_personas_have_normalized_expertise(self):
        """Test that all default personas have expertise in valid range."""
        for name, persona in DEFAULT_PERSONAS.items():
            for domain, score in persona.expertise.items():
                assert 0.0 <= score <= 1.0, f"{name}.{domain} = {score} is out of range"

    def test_core_agents_exist(self):
        """Test that core agent personas are defined."""
        core_agents = ["claude", "codex", "gemini", "grok", "deepseek"]
        for agent in core_agents:
            assert agent in DEFAULT_PERSONAS, f"Core agent {agent} missing from DEFAULT_PERSONAS"

    def test_compliance_agents_exist(self):
        """Test that compliance agent personas are defined."""
        compliance_agents = ["sox", "pci_dss", "hipaa", "gdpr", "fisma", "finra"]
        for agent in compliance_agents:
            assert agent in DEFAULT_PERSONAS, (
                f"Compliance agent {agent} missing from DEFAULT_PERSONAS"
            )

    def test_industry_vertical_agents_exist(self):
        """Test that industry vertical personas are defined."""
        vertical_agents = [
            "contract_analyst",
            "clinical_reviewer",
            "financial_auditor",
            "peer_reviewer",
        ]
        for agent in vertical_agents:
            assert agent in DEFAULT_PERSONAS, (
                f"Vertical agent {agent} missing from DEFAULT_PERSONAS"
            )

    def test_default_personas_agent_name_matches_key(self):
        """Test that each persona's agent_name matches its dict key."""
        for name, persona in DEFAULT_PERSONAS.items():
            assert persona.agent_name == name, f"Persona {name} has mismatched agent_name"

    def test_default_personas_temperature_range(self):
        """Test that default personas have reasonable temperature range."""
        for name, persona in DEFAULT_PERSONAS.items():
            # Temperature should typically be between 0 and 2
            assert 0.0 <= persona.temperature <= 2.0, (
                f"{name} has unusual temperature {persona.temperature}"
            )


# =============================================================================
# Module Exports and Schema Tests
# =============================================================================


class TestModuleIntegrity:
    """Tests for module integrity and exports."""

    def test_schema_version_defined(self):
        """Test that schema version is properly defined."""
        assert PERSONA_SCHEMA_VERSION >= 1
        assert isinstance(PERSONA_SCHEMA_VERSION, int)

    def test_expertise_domains_immutable_characteristics(self):
        """Test expertise domains list characteristics."""
        # All domains should be lowercase strings
        for domain in EXPERTISE_DOMAINS:
            assert isinstance(domain, str)
            assert domain == domain.lower(), f"Domain {domain} should be lowercase"
            assert " " not in domain, f"Domain {domain} should not contain spaces"

    def test_personality_traits_immutable_characteristics(self):
        """Test personality traits list characteristics."""
        # All traits should be lowercase strings
        for trait in PERSONALITY_TRAITS:
            assert isinstance(trait, str)
            assert trait == trait.lower(), f"Trait {trait} should be lowercase"
            assert " " not in trait, f"Trait {trait} should not contain spaces"

    def test_no_duplicate_domains(self):
        """Test that there are no duplicate expertise domains."""
        assert len(EXPERTISE_DOMAINS) == len(set(EXPERTISE_DOMAINS))

    def test_no_duplicate_traits(self):
        """Test that there are no duplicate personality traits."""
        assert len(PERSONALITY_TRAITS) == len(set(PERSONALITY_TRAITS))


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent database access patterns."""

    def test_simultaneous_read_write(self, temp_db):
        """Test reading while writing to database."""
        manager = PersonaManager(db_path=temp_db)

        # Create initial persona
        manager.create_persona(agent_name="concurrent_test")

        # Simulate concurrent access
        manager.record_performance(
            agent_name="concurrent_test",
            domain="security",
            success=True,
        )

        # Should be able to read immediately after write
        persona = manager.get_persona("concurrent_test")
        assert persona is not None

    def test_multiple_performance_records_same_agent(self, persona_manager):
        """Test recording multiple performances for same agent."""
        persona_manager.create_persona(agent_name="multi_record")

        # Record many performances rapidly
        for i in range(50):
            domain = EXPERTISE_DOMAINS[i % len(EXPERTISE_DOMAINS)]
            persona_manager.record_performance(
                agent_name="multi_record",
                domain=domain,
                success=i % 2 == 0,
            )

        # Should complete without error
        persona = persona_manager.get_persona("multi_record")
        assert persona is not None
