"""Tests for agent spec parsing in question_classifier.py.

Verifies that agent specs are always in 2-part format (agent_type:persona)
to ensure correct parsing by debate_factory.
"""

import pytest

from aragora.server.question_classifier import (
    PERSONA_TO_AGENT,
    QuestionClassification,
    QuestionClassifier,
    classify_and_assign_agents,
)


class TestPersonaToAgentMapping:
    """Tests for PERSONA_TO_AGENT mapping format."""

    def test_all_mappings_are_simple_agent_types(self):
        """All persona mappings should be simple agent types without colons.

        This ensures get_agent_string() produces 2-part specs that
        debate_factory.parse_agent_specs() can correctly parse.
        """
        for persona, agent_type in PERSONA_TO_AGENT.items():
            assert ":" not in agent_type, (
                f"PERSONA_TO_AGENT['{persona}'] = '{agent_type}' contains a colon. "
                f"Use the registered agent type instead (e.g., 'qwen' not 'openrouter:qwen/model')."
            )

    def test_openrouter_personas_use_registered_types(self):
        """OpenRouter personas should use registered agent types."""
        openrouter_personas = ["qwen", "qwen-max", "yi", "deepseek", "deepseek-r1", "kimi"]

        for persona in openrouter_personas:
            agent_type = PERSONA_TO_AGENT.get(persona)
            assert agent_type is not None, f"Missing mapping for {persona}"
            # Should be a simple registered type, not "openrouter:model/path"
            assert not agent_type.startswith("openrouter:"), (
                f"'{persona}' maps to '{agent_type}'. "
                f"Should use registered type '{persona}' instead."
            )


class TestGetAgentString:
    """Tests for QuestionClassifier.get_agent_string()."""

    def test_always_produces_two_part_specs(self):
        """Agent specs should always be 2-part format: agent_type:persona."""
        classifier = QuestionClassifier()
        classification = QuestionClassification(
            category="technical",
            complexity="moderate",
            recommended_personas=["qwen", "deepseek", "claude", "codex"],
        )

        result = classifier.get_agent_string(classification)

        for spec in result.split(","):
            parts = spec.split(":")
            assert len(parts) == 2, (
                f"Spec '{spec}' should have exactly 2 parts (agent_type:persona), "
                f"but has {len(parts)} parts"
            )

    def test_openrouter_specs_are_two_part(self):
        """OpenRouter agent specs should be 2-part, not 3-part."""
        classifier = QuestionClassifier()
        classification = QuestionClassification(
            category="technical",
            complexity="complex",
            recommended_personas=["qwen", "deepseek", "yi", "kimi"],
        )

        result = classifier.get_agent_string(classification)

        # Should produce specs like "qwen:qwen,deepseek:deepseek"
        # NOT "openrouter:model:qwen,openrouter:model:deepseek"
        for spec in result.split(","):
            parts = spec.split(":")
            assert len(parts) == 2, f"OpenRouter spec '{spec}' should be 2-part format"
            agent_type, persona = parts
            # Agent type should not contain slashes (model paths)
            assert "/" not in agent_type, (
                f"Agent type '{agent_type}' in spec '{spec}' contains '/'. "
                f"Model path should not be in the agent type."
            )

    def test_mixed_personas_produce_valid_specs(self):
        """Mix of OpenRouter and direct API personas should all be 2-part."""
        classifier = QuestionClassifier()
        classification = QuestionClassification(
            category="scientific",
            complexity="complex",
            recommended_personas=["claude", "qwen", "gemini", "deepseek", "grok"],
        )

        result = classifier.get_agent_string(classification)

        specs = result.split(",")
        assert len(specs) >= 2, "Should have at least 2 agents"

        for spec in specs:
            parts = spec.split(":")
            assert len(parts) == 2, f"Spec '{spec}' should be 2-part format"

    def test_avoids_duplicate_agent_types(self):
        """Should avoid duplicate agent types for diversity."""
        classifier = QuestionClassifier()
        # Create classification with personas that map to the same agent type
        classification = QuestionClassification(
            category="ethical",
            complexity="complex",
            recommended_personas=["philosopher", "existentialist", "humanist", "claude"],
        )

        result = classifier.get_agent_string(classification)
        specs = result.split(",")
        agent_types = [spec.split(":")[0] for spec in specs]

        # anthropic-api should not appear more than twice
        # (we allow up to 2 before filtering kicks in)
        anthropic_count = agent_types.count("anthropic-api")
        assert anthropic_count <= 2, (
            f"anthropic-api appears {anthropic_count} times, "
            f"diversity filter should limit duplicates"
        )


class TestClassifyAndAssignAgents:
    """Integration tests for classify_and_assign_agents()."""

    def test_simple_classification_produces_valid_specs(self):
        """Simple (non-LLM) classification should produce valid 2-part specs."""
        agent_string, classification = classify_and_assign_agents(
            "What is the best database for high-throughput writes?",
            use_llm=False,
        )

        assert classification.category in [
            "technical",
            "scientific",
            "general",
            "ethical",
            "financial",
            "healthcare",
            "legal",
            "security",
            "political",
        ]

        for spec in agent_string.split(","):
            parts = spec.split(":")
            assert len(parts) == 2, f"Spec '{spec}' should be 2-part format"

    def test_compliance_question_produces_valid_specs(self):
        """Compliance question should produce valid agent specs."""
        agent_string, classification = classify_and_assign_agents(
            "How do we ensure HIPAA compliance for patient data?",
            use_llm=False,
        )

        for spec in agent_string.split(","):
            parts = spec.split(":")
            assert len(parts) == 2, f"Spec '{spec}' should be 2-part format"


class TestDebateFactoryCompatibility:
    """Tests ensuring compatibility with debate_factory.parse_agent_specs()."""

    def test_specs_parseable_by_debate_factory(self):
        """Generated specs should be parseable by DebateConfig.parse_agent_specs()."""
        from aragora.server.debate_factory import AgentSpec, DebateConfig

        classifier = QuestionClassifier()
        classification = QuestionClassification(
            category="technical",
            complexity="moderate",
            recommended_personas=["qwen", "deepseek", "claude"],
        )

        agent_string = classifier.get_agent_string(classification)

        # Create DebateConfig and parse
        config = DebateConfig(
            question="Test question",
            agents_str=agent_string,
        )

        specs = config.parse_agent_specs()

        assert len(specs) >= 2, "Should have at least 2 agent specs"
        for spec in specs:
            assert isinstance(spec, AgentSpec)
            assert spec.agent_type, f"Agent type should not be empty: {spec}"
            # Role should be the persona, not a model path
            assert "/" not in (spec.role or ""), (
                f"Role '{spec.role}' contains '/', suggesting a model path leaked into role"
            )
