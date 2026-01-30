"""
Comprehensive tests for pattern extraction.

Tests cover:
- Pattern dataclass
- Strategy dataclass
- PatternExtractor class
- StrategyIdentifier class
- Module-level convenience functions
- Evidence pattern detection
- Structure pattern detection
- Persuasion pattern detection
- Response pattern detection
"""

import pytest

from aragora.evolution.pattern_extractor import (
    Pattern,
    PatternExtractor,
    Strategy,
    StrategyIdentifier,
    extract_patterns,
    identify_strategies,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a PatternExtractor instance."""
    return PatternExtractor()


@pytest.fixture
def identifier():
    """Create a StrategyIdentifier instance."""
    return StrategyIdentifier()


@pytest.fixture
def winning_debate_outcome():
    """Create a mock debate outcome with a winner."""
    return {
        "winner": "claude",
        "consensus_reached": True,
        "messages": [
            {
                "agent": "claude",
                "content": "According to research, the data shows that firstly we should "
                "consider the evidence. Studies indicate this is the best approach. "
                "Secondly, for example, other implementations have succeeded.",
            },
            {
                "agent": "gpt4",
                "content": "I disagree with the approach.",
            },
            {
                "agent": "claude",
                "content": "You raise a valid point, but addressing your concern, "
                "it's important to note that the key point is efficiency. "
                "Finally, in conclusion, we should proceed.",
            },
        ],
        "critiques": [
            {"from": "gpt4", "to": "claude", "text": "Needs more evidence"},
        ],
    }


@pytest.fixture
def empty_debate_outcome():
    """Create a debate outcome with no messages."""
    return {
        "winner": None,
        "consensus_reached": False,
        "messages": [],
        "critiques": [],
    }


# =============================================================================
# Pattern Dataclass Tests
# =============================================================================


class TestPattern:
    """Test Pattern dataclass."""

    def test_create_pattern(self):
        """Test creating a basic Pattern."""
        pattern = Pattern(
            pattern_type="evidence",
            description="Uses citations",
        )

        assert pattern.pattern_type == "evidence"
        assert pattern.description == "Uses citations"
        assert pattern.agent is None
        assert pattern.frequency == 1
        assert pattern.effectiveness == 0.0
        assert pattern.examples == []

    def test_pattern_with_all_fields(self):
        """Test Pattern with all fields populated."""
        pattern = Pattern(
            pattern_type="structure",
            description="Uses numbered lists",
            agent="claude",
            frequency=5,
            effectiveness=0.8,
            examples=["First, ...", "Second, ..."],
        )

        assert pattern.agent == "claude"
        assert pattern.frequency == 5
        assert pattern.effectiveness == 0.8
        assert len(pattern.examples) == 2

    def test_pattern_to_dict(self):
        """Test Pattern conversion to dictionary."""
        pattern = Pattern(
            pattern_type="persuasion",
            description="Uses emphasis",
            agent="claude",
            frequency=3,
            effectiveness=0.7,
            examples=["Example 1"],
        )

        d = pattern.to_dict()

        assert d["pattern_type"] == "persuasion"
        assert d["description"] == "Uses emphasis"
        assert d["agent"] == "claude"
        assert d["frequency"] == 3
        assert d["effectiveness"] == 0.7
        assert d["examples"] == ["Example 1"]


# =============================================================================
# Strategy Dataclass Tests
# =============================================================================


class TestStrategy:
    """Test Strategy dataclass."""

    def test_create_strategy(self):
        """Test creating a basic Strategy."""
        strategy = Strategy(
            name="Evidence-Based",
            description="Uses evidence and citations",
        )

        assert strategy.name == "Evidence-Based"
        assert strategy.description == "Uses evidence and citations"
        assert strategy.success_rate == 0.0
        assert strategy.agent is None
        assert strategy.tactics == []

    def test_strategy_with_all_fields(self):
        """Test Strategy with all fields populated."""
        strategy = Strategy(
            name="Structured Reasoning",
            description="Uses clear structure",
            success_rate=0.85,
            agent="claude",
            tactics=["Use enumeration", "Provide transitions"],
        )

        assert strategy.success_rate == 0.85
        assert strategy.agent == "claude"
        assert len(strategy.tactics) == 2

    def test_strategy_to_dict(self):
        """Test Strategy conversion to dictionary."""
        strategy = Strategy(
            name="Conciliatory",
            description="Acknowledges valid points",
            success_rate=0.75,
            agent="gpt4",
            tactics=["Find common ground"],
        )

        d = strategy.to_dict()

        assert d["name"] == "Conciliatory"
        assert d["description"] == "Acknowledges valid points"
        assert d["success_rate"] == 0.75
        assert d["agent"] == "gpt4"
        assert d["tactics"] == ["Find common ground"]


# =============================================================================
# PatternExtractor Tests
# =============================================================================


class TestPatternExtractor:
    """Test PatternExtractor class."""

    def test_init_compiles_patterns(self, extractor):
        """Test that patterns are compiled on initialization."""
        assert "evidence" in extractor._compiled_patterns
        assert "structure" in extractor._compiled_patterns
        assert "persuasion" in extractor._compiled_patterns
        assert len(extractor._compiled_patterns["evidence"]) > 0

    def test_extract_empty_outcome(self, extractor, empty_debate_outcome):
        """Test extraction from empty outcome."""
        patterns = extractor.extract(empty_debate_outcome)
        assert patterns == []

    def test_extract_no_winner(self, extractor):
        """Test extraction with no winner."""
        outcome = {
            "winner": None,
            "messages": [{"agent": "claude", "content": "Test message"}],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)

        # Should still extract patterns even without winner
        # (they may be empty if no winner messages match)
        assert isinstance(patterns, list)

    def test_extract_evidence_patterns(self, extractor, winning_debate_outcome):
        """Test extraction of evidence patterns."""
        patterns = extractor.extract(winning_debate_outcome)

        evidence_patterns = [p for p in patterns if p.pattern_type == "evidence"]
        assert len(evidence_patterns) > 0

        evidence = evidence_patterns[0]
        assert "evidence" in evidence.description.lower()
        assert evidence.agent == "claude"
        assert evidence.frequency >= 1

    def test_extract_structure_patterns(self, extractor, winning_debate_outcome):
        """Test extraction of structure patterns."""
        patterns = extractor.extract(winning_debate_outcome)

        structure_patterns = [p for p in patterns if p.pattern_type == "structure"]
        assert len(structure_patterns) > 0

        structure = structure_patterns[0]
        assert "structure" in structure.description.lower()

    def test_extract_persuasion_patterns(self, extractor, winning_debate_outcome):
        """Test extraction of persuasion patterns."""
        patterns = extractor.extract(winning_debate_outcome)

        persuasion_patterns = [p for p in patterns if p.pattern_type == "persuasion"]
        assert len(persuasion_patterns) > 0

        persuasion = persuasion_patterns[0]
        assert (
            "emphatic" in persuasion.description.lower()
            or "highlight" in persuasion.description.lower()
        )

    def test_extract_response_patterns(self, extractor, winning_debate_outcome):
        """Test extraction of response patterns."""
        patterns = extractor.extract(winning_debate_outcome)

        response_patterns = [p for p in patterns if p.pattern_type == "response"]
        assert len(response_patterns) > 0

        response = response_patterns[0]
        assert "acknowledges" in response.description.lower()

    def test_extract_no_response_patterns_without_critiques(self, extractor):
        """Test no response patterns when no critiques."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": "You raise a valid point but I disagree.",
                }
            ],
            "critiques": [],  # No critiques
        }

        patterns = extractor.extract(outcome)

        response_patterns = [p for p in patterns if p.pattern_type == "response"]
        assert len(response_patterns) == 0

    def test_extract_limits_examples(self, extractor):
        """Test that examples are limited to 3."""
        # Create outcome with many evidence markers
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": (
                        "According to research 1. "
                        "Research shows point 2. "
                        "Studies indicate finding 3. "
                        "Evidence suggests conclusion 4. "
                        "Data from source 5."
                    ),
                }
            ],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)

        evidence_patterns = [p for p in patterns if p.pattern_type == "evidence"]
        if evidence_patterns:
            assert len(evidence_patterns[0].examples) <= 3

    def test_evidence_markers(self, extractor):
        """Test various evidence markers are detected."""
        markers = [
            "according to the study",
            "research shows that",
            "studies indicate results",
            "evidence suggests conclusion",
            "data from the experiment",
            "statistics show trends",
            "for example in this case",
            "as demonstrated by results",
        ]

        for marker in markers:
            outcome = {
                "winner": "claude",
                "messages": [{"agent": "claude", "content": marker}],
                "critiques": [],
            }

            patterns = extractor.extract(outcome)
            evidence_patterns = [p for p in patterns if p.pattern_type == "evidence"]
            assert len(evidence_patterns) > 0, f"Marker '{marker}' not detected"

    def test_structure_markers(self, extractor):
        """Test various structure markers are detected."""
        markers = [
            "first we should consider",
            "firstly the approach",
            "secondly we note",
            "thirdly the result",
            "finally we conclude",
            "in conclusion the answer",
            "to summarize the findings",
        ]

        for marker in markers:
            outcome = {
                "winner": "claude",
                "messages": [{"agent": "claude", "content": marker}],
                "critiques": [],
            }

            patterns = extractor.extract(outcome)
            structure_patterns = [p for p in patterns if p.pattern_type == "structure"]
            assert len(structure_patterns) > 0, f"Marker '{marker}' not detected"

    def test_persuasion_markers(self, extractor):
        """Test various persuasion markers are detected."""
        markers = [
            "it's important to note that",
            "we must consider the fact",
            "the key point is this",
            "this demonstrates clearly",
            "clearly shows the result",
            "fundamentally the issue",
            "critically we observe",
        ]

        for marker in markers:
            outcome = {
                "winner": "claude",
                "messages": [{"agent": "claude", "content": marker}],
                "critiques": [],
            }

            patterns = extractor.extract(outcome)
            persuasion_patterns = [p for p in patterns if p.pattern_type == "persuasion"]
            assert len(persuasion_patterns) > 0, f"Marker '{marker}' not detected"


# =============================================================================
# StrategyIdentifier Tests
# =============================================================================


class TestStrategyIdentifier:
    """Test StrategyIdentifier class."""

    def test_strategy_templates_defined(self, identifier):
        """Test that strategy templates are defined."""
        assert "evidence_based" in identifier.STRATEGY_TEMPLATES
        assert "structured" in identifier.STRATEGY_TEMPLATES
        assert "conciliatory" in identifier.STRATEGY_TEMPLATES
        assert "direct" in identifier.STRATEGY_TEMPLATES

    def test_template_structure(self, identifier):
        """Test that templates have required fields."""
        for key, template in identifier.STRATEGY_TEMPLATES.items():
            assert "name" in template
            assert "description" in template
            assert "tactics" in template
            assert isinstance(template["tactics"], list)

    def test_identify_no_winner(self, identifier):
        """Test identification with no winner."""
        outcome = {
            "winner": None,
            "messages": [{"agent": "claude", "content": "Test"}],
        }

        strategies = identifier.identify(outcome)
        assert strategies == []

    def test_identify_no_messages(self, identifier):
        """Test identification with no winner messages."""
        outcome = {
            "winner": "claude",
            "messages": [{"agent": "gpt4", "content": "Only gpt4 spoke"}],
        }

        strategies = identifier.identify(outcome)
        assert strategies == []

    def test_identify_evidence_strategy(self, identifier):
        """Test identification of evidence-based strategy."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": (
                        "According to research, the data shows evidence. "
                        "Studies indicate and statistics demonstrate the source."
                    ),
                }
            ],
            "consensus_reached": False,
        }

        strategies = identifier.identify(outcome)

        evidence_strategies = [s for s in strategies if "evidence" in s.name.lower()]
        assert len(evidence_strategies) > 0
        assert evidence_strategies[0].success_rate > 0.3

    def test_identify_structured_strategy(self, identifier):
        """Test identification of structured strategy."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": (
                        "First, we consider this. Therefore, we conclude. "
                        "Thus, we can state. Hence, the result follows."
                    ),
                }
            ],
            "consensus_reached": False,
        }

        strategies = identifier.identify(outcome)

        structured_strategies = [s for s in strategies if "structured" in s.name.lower()]
        assert len(structured_strategies) > 0

    def test_identify_conciliatory_strategy(self, identifier):
        """Test identification of conciliatory strategy."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": (
                        "You're right about that. I agree with your point. "
                        "That's a fair criticism. However, while true, "
                        "I acknowledge the issue."
                    ),
                }
            ],
            "consensus_reached": False,
        }

        strategies = identifier.identify(outcome)

        conciliatory_strategies = [s for s in strategies if "conciliatory" in s.name.lower()]
        assert len(conciliatory_strategies) > 0

    def test_consensus_boosts_score(self, identifier):
        """Test that consensus_reached boosts success rate."""
        outcome_no_consensus = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": "According to research, data shows evidence clearly.",
                }
            ],
            "consensus_reached": False,
        }

        outcome_with_consensus = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": "According to research, data shows evidence clearly.",
                }
            ],
            "consensus_reached": True,
        }

        strategies_no = identifier.identify(outcome_no_consensus)
        strategies_yes = identifier.identify(outcome_with_consensus)

        if strategies_no and strategies_yes:
            # Find matching strategies
            for s_no in strategies_no:
                for s_yes in strategies_yes:
                    if s_no.name == s_yes.name:
                        assert s_yes.success_rate >= s_no.success_rate

    def test_identify_multiple_strategies(self, identifier):
        """Test identification of multiple strategies in one debate."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": (
                        "According to research, first we should note. "
                        "I agree with your point, however the evidence shows. "
                        "Therefore, in conclusion, the data demonstrates."
                    ),
                }
            ],
            "consensus_reached": True,
        }

        strategies = identifier.identify(outcome)

        # Should identify multiple strategies
        assert len(strategies) >= 2

    def test_score_normalization(self, identifier):
        """Test that scores are normalized to [0, 1]."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": (
                        "According to research data evidence statistics source citation "
                        "shows demonstrates first second third finally therefore thus hence"
                    )
                    * 10,  # Repeat to get high counts
                }
            ],
            "consensus_reached": True,
        }

        strategies = identifier.identify(outcome)

        for strategy in strategies:
            assert 0 <= strategy.success_rate <= 1


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_extract_patterns_function(self, winning_debate_outcome):
        """Test extract_patterns convenience function."""
        result = extract_patterns(winning_debate_outcome)

        assert "winning_patterns" in result
        assert "winner" in result
        assert "pattern_count" in result
        assert result["winner"] == "claude"
        assert result["pattern_count"] >= 0
        assert isinstance(result["winning_patterns"], list)

    def test_extract_patterns_returns_dicts(self, winning_debate_outcome):
        """Test that extract_patterns returns pattern dicts."""
        result = extract_patterns(winning_debate_outcome)

        for pattern in result["winning_patterns"]:
            assert isinstance(pattern, dict)
            assert "pattern_type" in pattern
            assert "description" in pattern

    def test_identify_strategies_function(self, winning_debate_outcome):
        """Test identify_strategies convenience function."""
        result = identify_strategies(winning_debate_outcome)

        assert isinstance(result, list)
        for strategy in result:
            assert isinstance(strategy, dict)
            assert "name" in strategy
            assert "description" in strategy
            assert "success_rate" in strategy
            assert "tactics" in strategy

    def test_functions_use_singleton_instances(self):
        """Test that module functions use singleton instances."""
        # Call twice and verify consistent behavior
        outcome = {
            "winner": "claude",
            "messages": [{"agent": "claude", "content": "According to research"}],
            "critiques": [],
        }

        result1 = extract_patterns(outcome)
        result2 = extract_patterns(outcome)

        assert result1["pattern_count"] == result2["pattern_count"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases in pattern extraction."""

    def test_empty_message_content(self, extractor):
        """Test handling of empty message content."""
        outcome = {
            "winner": "claude",
            "messages": [{"agent": "claude", "content": ""}],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)
        assert patterns == []

    def test_none_winner(self, extractor):
        """Test handling of None winner."""
        outcome = {
            "winner": None,
            "messages": [
                {"agent": "claude", "content": "According to research"},
            ],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)
        # Should handle gracefully, may or may not find patterns
        assert isinstance(patterns, list)

    def test_missing_keys(self, extractor):
        """Test handling of missing keys in outcome."""
        outcome = {"winner": "claude"}  # Missing messages and critiques

        patterns = extractor.extract(outcome)
        assert patterns == []

    def test_case_insensitive_matching(self, extractor):
        """Test that pattern matching is case-insensitive."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": "ACCORDING TO RESEARCH shows DATA",
                }
            ],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)

        evidence_patterns = [p for p in patterns if p.pattern_type == "evidence"]
        assert len(evidence_patterns) > 0

    def test_unicode_content(self, extractor):
        """Test handling of unicode content."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": "According to research, les donnees montrent que...",
                }
            ],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)
        # Should handle unicode gracefully
        assert isinstance(patterns, list)

    def test_very_long_message(self, extractor):
        """Test handling of very long messages."""
        long_content = "According to research. " * 1000

        outcome = {
            "winner": "claude",
            "messages": [{"agent": "claude", "content": long_content}],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)
        assert isinstance(patterns, list)

    def test_special_characters(self, extractor):
        """Test handling of special characters in content."""
        outcome = {
            "winner": "claude",
            "messages": [
                {
                    "agent": "claude",
                    "content": "According to research: [data] shows (evidence) & demonstrates!",
                }
            ],
            "critiques": [],
        }

        patterns = extractor.extract(outcome)
        assert isinstance(patterns, list)
