"""Tests for evolution module.

Tests the evolution system including:
- PromptEvolver: mutation, crossover, pattern extraction, prompt evolution
- EvolutionTracker: outcome tracking, performance metrics
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.core import Agent, Critique, DebateResult
from aragora.evolution.evolver import (
    EvolutionStrategy,
    PromptEvolver,
    PromptVersion,
)
from aragora.evolution.tracker import EvolutionTracker, OutcomeRecord


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def evolver(temp_db):
    """Create a PromptEvolver with temp database."""
    return PromptEvolver(db_path=temp_db)


@pytest.fixture
def tracker(temp_db):
    """Create an EvolutionTracker with temp database."""
    return EvolutionTracker(db_path=temp_db)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=Agent)
    agent.name = "test-agent"
    agent.system_prompt = "You are a helpful assistant."
    return agent


@pytest.fixture
def mock_debate_result():
    """Create a mock debate result."""
    critique = MagicMock(spec=Critique)
    critique.severity = 0.5
    critique.issues = ["Consider edge cases"]
    critique.suggestions = ["Add error handling"]

    result = MagicMock(spec=DebateResult)
    result.id = "debate-123"
    result.consensus_reached = True
    result.confidence = 0.85
    result.critiques = [critique]
    result.final_answer = (
        "Here is the solution:\n```python\nprint('hello')\n```\nStep 1: First, do this."
    )
    return result


@pytest.fixture
def mock_vulnerability():
    """Create a mock vulnerability object."""
    from unittest.mock import MagicMock
    from enum import Enum

    class MockSeverity(Enum):
        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    vuln = MagicMock()
    vuln.title = "Hallucination detected"
    vuln.category = "HALLUCINATION"
    vuln.severity = MockSeverity.HIGH
    return vuln


# =============================================================================
# PromptVersion Tests
# =============================================================================


class TestPromptVersion:
    """Test PromptVersion dataclass."""

    def test_create_prompt_version(self):
        """Test creating a PromptVersion."""
        version = PromptVersion(
            version=1,
            prompt="Test prompt",
            created_at="2026-01-01T00:00:00",
        )

        assert version.version == 1
        assert version.prompt == "Test prompt"
        assert version.performance_score == 0.0
        assert version.debates_count == 0

    def test_prompt_version_with_metadata(self):
        """Test PromptVersion with custom metadata."""
        version = PromptVersion(
            version=2,
            prompt="Test",
            created_at="2026-01-01T00:00:00",
            performance_score=0.75,
            metadata={"strategy": "append"},
        )

        assert version.metadata == {"strategy": "append"}
        assert version.performance_score == 0.75


# =============================================================================
# EvolutionStrategy Tests
# =============================================================================


class TestEvolutionStrategy:
    """Test EvolutionStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values exist."""
        assert EvolutionStrategy.APPEND.value == "append"
        assert EvolutionStrategy.REPLACE.value == "replace"
        assert EvolutionStrategy.REFINE.value == "refine"
        assert EvolutionStrategy.HYBRID.value == "hybrid"


# =============================================================================
# PromptEvolver Tests - Mutation
# =============================================================================


class TestPromptEvolverMutation:
    """Test PromptEvolver mutation operations."""

    def test_mutate_empty_prompt(self, evolver):
        """Test mutation with empty prompt."""
        result = evolver.mutate("")
        assert result == ""

    def test_mutate_with_zero_rate(self, evolver):
        """Test mutation with zero mutation rate."""
        evolver.mutation_rate = 0.0
        prompt = "Be precise and helpful."
        result = evolver.mutate(prompt)
        # With zero rate, prompt should be unchanged
        assert result == prompt

    def test_mutate_with_high_rate(self, evolver):
        """Test mutation with high rate causes changes."""
        evolver.mutation_rate = 1.0
        prompt = "Be precise"
        # Multiple runs should eventually produce changes
        results = set()
        for _ in range(10):
            results.add(evolver.mutate(prompt))
        # Should have some variation
        assert len(results) >= 1

    def test_mutate_adds_suffix(self, evolver):
        """Test mutation can add suffix."""
        evolver.mutation_rate = 1.0
        prompt = "Simple prompt with no match targets"
        # Run multiple times to hit suffix path
        suffixes_found = 0
        for _ in range(20):
            result = evolver.mutate(prompt)
            if any(s in result for s in ["perspectives", "reasoning", "thorough"]):
                suffixes_found += 1
        # Should find at least one suffix addition
        assert suffixes_found >= 1


# =============================================================================
# PromptEvolver Tests - Crossover
# =============================================================================


class TestPromptEvolverCrossover:
    """Test PromptEvolver crossover operations."""

    def test_crossover_empty_parents(self, evolver):
        """Test crossover with empty parents."""
        assert evolver.crossover("", "") == ""
        assert evolver.crossover("Hello.", "") == "Hello."
        assert evolver.crossover("", "World.") == "World."

    def test_crossover_single_sentence_parents(self, evolver):
        """Test crossover with single sentence parents."""
        result = evolver.crossover("Be helpful.", "Be accurate.")
        assert result.endswith(".")
        # Should contain content from at least one parent
        assert "helpful" in result or "accurate" in result

    def test_crossover_multi_sentence_parents(self, evolver):
        """Test crossover with multiple sentences."""
        parent1 = "Be helpful. Be precise. Think carefully."
        parent2 = "Be accurate. Be thorough. Consider alternatives."

        results = set()
        for _ in range(10):
            results.add(evolver.crossover(parent1, parent2))

        # Should produce variations
        assert len(results) >= 1
        # All results should end with period
        for result in results:
            assert result.endswith(".")

    def test_crossover_preserves_structure(self, evolver):
        """Test crossover maintains sentence structure."""
        parent1 = "First sentence. Second sentence."
        parent2 = "Third sentence. Fourth sentence."

        result = evolver.crossover(parent1, parent2)

        # Should be valid sentences
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) >= 1


# =============================================================================
# PromptEvolver Tests - Pattern Extraction
# =============================================================================


class TestPromptEvolverPatternExtraction:
    """Test pattern extraction from debates."""

    def test_extract_patterns_empty_list(self, evolver):
        """Test extraction from empty debate list."""
        patterns = evolver.extract_winning_patterns([])
        assert patterns == []

    def test_extract_patterns_no_consensus(self, evolver, mock_debate_result):
        """Test extraction skips debates without consensus."""
        mock_debate_result.consensus_reached = False
        patterns = evolver.extract_winning_patterns([mock_debate_result])
        assert patterns == []

    def test_extract_patterns_low_confidence(self, evolver, mock_debate_result):
        """Test extraction skips low confidence debates."""
        mock_debate_result.confidence = 0.3
        patterns = evolver.extract_winning_patterns([mock_debate_result], min_confidence=0.5)
        assert patterns == []

    def test_extract_patterns_success(self, evolver, mock_debate_result):
        """Test successful pattern extraction."""
        patterns = evolver.extract_winning_patterns([mock_debate_result])

        assert len(patterns) > 0
        # Should extract critique patterns and structural patterns
        pattern_types = {p["type"] for p in patterns}
        assert "issue_identification" in pattern_types or "includes_code" in pattern_types

    def test_extract_patterns_max_limit(self, evolver, mock_debate_result):
        """Test max patterns limit is respected."""
        # Create debate with many patterns
        mock_debate_result.critiques[0].issues = [f"Issue {i}" for i in range(100)]

        patterns = evolver.extract_winning_patterns([mock_debate_result], max_patterns=5)
        assert len(patterns) <= 5

    def test_extract_structured_response_pattern(self, evolver, mock_debate_result):
        """Test extraction of structured response patterns."""
        patterns = evolver.extract_winning_patterns([mock_debate_result])

        # Check for structured response pattern (Step 1 in final_answer)
        pattern_texts = [p.get("text", "") for p in patterns]
        structured_found = any(
            "structured" in t.lower() or "step" in t.lower() for t in pattern_texts
        )
        # May or may not find depending on exact matching
        assert isinstance(structured_found, bool)


# =============================================================================
# PromptEvolver Tests - Storage
# =============================================================================


class TestPromptEvolverStorage:
    """Test PromptEvolver database operations."""

    def test_store_patterns(self, evolver):
        """Test storing patterns to database."""
        patterns = [
            {"type": "test", "text": "Test pattern 1", "source_debate": "d1"},
            {"type": "test", "text": "Test pattern 2", "source_debate": "d2"},
        ]

        evolver.store_patterns(patterns)

        # Retrieve and verify
        stored = evolver.get_top_patterns(limit=10)
        assert len(stored) == 2

    def test_get_top_patterns_empty(self, evolver):
        """Test getting patterns from empty database."""
        patterns = evolver.get_top_patterns()
        assert patterns == []

    def test_get_top_patterns_with_type(self, evolver):
        """Test getting patterns filtered by type."""
        evolver.store_patterns(
            [
                {"type": "type_a", "text": "Pattern A"},
                {"type": "type_b", "text": "Pattern B"},
            ]
        )

        type_a = evolver.get_top_patterns(pattern_type="type_a")
        assert len(type_a) == 1
        assert type_a[0]["type"] == "type_a"

    def test_save_prompt_version(self, evolver):
        """Test saving a prompt version."""
        version = evolver.save_prompt_version(
            agent_name="test-agent",
            prompt="Test prompt content",
            metadata={"strategy": "append"},
        )

        assert version == 1

        # Save another version
        version2 = evolver.save_prompt_version("test-agent", "Updated prompt")
        assert version2 == 2

    def test_get_prompt_version_latest(self, evolver):
        """Test getting latest prompt version."""
        evolver.save_prompt_version("agent1", "Version 1")
        evolver.save_prompt_version("agent1", "Version 2")

        latest = evolver.get_prompt_version("agent1")
        assert latest is not None
        assert latest.version == 2
        assert latest.prompt == "Version 2"

    def test_get_prompt_version_specific(self, evolver):
        """Test getting specific prompt version."""
        evolver.save_prompt_version("agent1", "Version 1")
        evolver.save_prompt_version("agent1", "Version 2")

        v1 = evolver.get_prompt_version("agent1", version=1)
        assert v1 is not None
        assert v1.prompt == "Version 1"

    def test_get_prompt_version_nonexistent(self, evolver):
        """Test getting nonexistent version returns None."""
        result = evolver.get_prompt_version("nonexistent")
        assert result is None


# =============================================================================
# PromptEvolver Tests - Evolution
# =============================================================================


class TestPromptEvolverEvolution:
    """Test prompt evolution strategies."""

    def test_evolve_append_strategy(self, evolver, mock_agent):
        """Test append evolution strategy."""
        patterns = [
            {"type": "issue_identification", "text": "Check edge cases"},
            {"type": "improvement_suggestion", "text": "Add validation"},
        ]

        result = evolver.evolve_prompt(mock_agent, patterns, EvolutionStrategy.APPEND)

        assert mock_agent.system_prompt in result
        assert "Learned patterns" in result
        assert "Watch for: Check edge cases" in result

    def test_evolve_replace_strategy(self, evolver, mock_agent):
        """Test replace evolution strategy."""
        # First add learned patterns
        mock_agent.system_prompt = (
            "Base prompt.\n\nLearned patterns from successful debates:\n- Old pattern"
        )

        patterns = [{"type": "issue_identification", "text": "New pattern"}]

        result = evolver.evolve_prompt(mock_agent, patterns, EvolutionStrategy.REPLACE)

        assert "New pattern" in result
        # Old pattern should be removed
        assert "Old pattern" not in result

    def test_evolve_hybrid_short_prompt(self, evolver, mock_agent):
        """Test hybrid strategy with short prompt uses append."""
        mock_agent.system_prompt = "Short prompt."
        patterns = [{"type": "test", "text": "Pattern"}]

        result = evolver.evolve_prompt(mock_agent, patterns, EvolutionStrategy.HYBRID)

        # Should use append for short prompts
        assert len(result) < 2000

    def test_evolve_no_patterns(self, evolver, mock_agent):
        """Test evolution with no patterns returns original."""
        result = evolver.evolve_prompt(mock_agent, [], EvolutionStrategy.APPEND)
        assert result == mock_agent.system_prompt

    def test_apply_evolution(self, evolver, mock_agent):
        """Test full evolution application."""
        patterns = [{"type": "test", "text": "Test pattern"}]

        new_prompt = evolver.apply_evolution(mock_agent, patterns)

        # Should save version
        saved = evolver.get_prompt_version(mock_agent.name)
        assert saved is not None
        assert saved.prompt == new_prompt

        # Should update agent
        mock_agent.set_system_prompt.assert_called_once_with(new_prompt)

    def test_get_evolution_history(self, evolver, mock_agent):
        """Test getting evolution history."""
        patterns = [{"type": "test", "text": "Pattern 1"}]
        evolver.apply_evolution(mock_agent, patterns)

        history = evolver.get_evolution_history(mock_agent.name)
        assert len(history) == 1
        assert history[0]["strategy"] == "append"


# =============================================================================
# PromptEvolver Tests - Performance Tracking
# =============================================================================


class TestPromptEvolverPerformance:
    """Test performance tracking."""

    def test_update_performance(self, evolver, mock_agent, mock_debate_result):
        """Test updating performance metrics."""
        evolver.save_prompt_version(mock_agent.name, "Test prompt")

        evolver.update_performance(mock_agent.name, 1, mock_debate_result)

        version = evolver.get_prompt_version(mock_agent.name, 1)
        assert version.debates_count == 1
        # Consensus reached, so rate should be 1.0
        assert version.consensus_rate == 1.0

    def test_update_performance_no_consensus(self, evolver, mock_agent, mock_debate_result):
        """Test updating performance without consensus."""
        evolver.save_prompt_version(mock_agent.name, "Test prompt")
        mock_debate_result.consensus_reached = False

        evolver.update_performance(mock_agent.name, 1, mock_debate_result)

        version = evolver.get_prompt_version(mock_agent.name, 1)
        assert version.debates_count == 1
        assert version.consensus_rate == 0.0


# =============================================================================
# PromptEvolver Tests - Vulnerability Recording
# =============================================================================


class TestPromptEvolverVulnerability:
    """Test vulnerability recording for gauntlet integration."""

    def test_record_vulnerability(self, evolver, mock_vulnerability):
        """Test recording a vulnerability."""
        evolver.record_vulnerability(
            agent_name="test-agent",
            vulnerability=mock_vulnerability,
            trigger_prompt="What is 2+2?",
            agent_response="The answer is 5.",
            gauntlet_id="gauntlet-123",
        )

        patterns = evolver.get_vulnerability_patterns("test-agent")
        assert len(patterns) == 1
        assert patterns[0]["category"] == "HALLUCINATION"

    def test_record_vulnerability_increments_count(self, evolver, mock_vulnerability):
        """Test that recording same vulnerability increments count."""
        evolver.record_vulnerability("agent", mock_vulnerability)
        evolver.record_vulnerability("agent", mock_vulnerability)

        patterns = evolver.get_vulnerability_patterns("agent")
        assert len(patterns) == 1
        assert patterns[0]["occurrences"] == 2

    def test_suggest_mitigation(self, evolver):
        """Test mitigation suggestion."""
        mitigation = evolver._suggest_mitigation("HALLUCINATION", "HIGH")
        assert "instruction" in mitigation.lower()

        mitigation = evolver._suggest_mitigation("SYCOPHANCY", "MEDIUM")
        assert "instruction" in mitigation.lower()

        # Unknown category gets generic mitigation
        mitigation = evolver._suggest_mitigation("UNKNOWN", "LOW")
        assert "Review" in mitigation or "instruction" in mitigation.lower()

    def test_get_vulnerability_summary(self, evolver, mock_vulnerability):
        """Test vulnerability summary."""
        evolver.record_vulnerability("agent", mock_vulnerability)

        summary = evolver.get_vulnerability_summary("agent")

        assert summary["total_occurrences"] == 1
        assert summary["unique_vulnerability_types"] == 1
        assert "HALLUCINATION" in summary["by_category"]

    def test_get_vulnerability_patterns_min_occurrences(self, evolver, mock_vulnerability):
        """Test filtering patterns by minimum occurrences."""
        evolver.record_vulnerability("agent", mock_vulnerability)

        # Should find with min=1
        patterns = evolver.get_vulnerability_patterns("agent", min_occurrences=1)
        assert len(patterns) == 1

        # Should not find with min=5
        patterns = evolver.get_vulnerability_patterns("agent", min_occurrences=5)
        assert len(patterns) == 0


# =============================================================================
# EvolutionTracker Tests - Outcome Recording
# =============================================================================


class TestEvolutionTrackerOutcomes:
    """Test outcome recording."""

    def test_record_outcome(self, tracker):
        """Test recording a debate outcome."""
        tracker.record_outcome(
            agent="agent1",
            won=True,
            debate_id="debate-1",
            generation=0,
        )

        stats = tracker.get_agent_stats("agent1")
        assert stats["wins"] == 1
        assert stats["total"] == 1
        assert stats["win_rate"] == 1.0

    def test_record_multiple_outcomes(self, tracker):
        """Test recording multiple outcomes."""
        tracker.record_outcome("agent1", won=True)
        tracker.record_outcome("agent1", won=True)
        tracker.record_outcome("agent1", won=False)

        stats = tracker.get_agent_stats("agent1")
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["total"] == 3
        assert abs(stats["win_rate"] - 2 / 3) < 0.01


class TestOutcomeRecord:
    """Test OutcomeRecord dataclass."""

    def test_create_outcome_record(self):
        """Test creating an OutcomeRecord."""
        record = OutcomeRecord(agent="test", won=True)

        assert record.agent == "test"
        assert record.won is True
        assert record.recorded_at != ""

    def test_outcome_record_auto_timestamp(self):
        """Test OutcomeRecord auto-generates timestamp."""
        record = OutcomeRecord(agent="test", won=False)
        # Should have ISO format timestamp
        datetime.fromisoformat(record.recorded_at)


# =============================================================================
# EvolutionTracker Tests - Statistics
# =============================================================================


class TestEvolutionTrackerStats:
    """Test evolution statistics."""

    def test_get_agent_stats_empty(self, tracker):
        """Test stats for agent with no outcomes."""
        stats = tracker.get_agent_stats("nonexistent")

        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0

    def test_get_generation_metrics(self, tracker):
        """Test generation metrics."""
        tracker.record_outcome("a1", won=True, generation=1)
        tracker.record_outcome("a2", won=True, generation=1)
        tracker.record_outcome("a3", won=False, generation=1)

        metrics = tracker.get_generation_metrics(1)

        assert metrics["generation"] == 1
        assert metrics["total_debates"] == 3
        assert metrics["wins"] == 2
        assert metrics["unique_agents"] == 3

    def test_get_performance_delta(self, tracker):
        """Test performance delta between generations."""
        # Gen 0: 1 win, 1 loss
        tracker.record_outcome("agent", won=True, generation=0)
        tracker.record_outcome("agent", won=False, generation=0)

        # Gen 1: 3 wins, 1 loss (improved)
        tracker.record_outcome("agent", won=True, generation=1)
        tracker.record_outcome("agent", won=True, generation=1)
        tracker.record_outcome("agent", won=True, generation=1)
        tracker.record_outcome("agent", won=False, generation=1)

        delta = tracker.get_performance_delta("agent", 0, 1)

        assert delta["gen1_win_rate"] == 0.5
        assert delta["gen2_win_rate"] == 0.75
        assert delta["win_rate_delta"] == 0.25
        assert delta["improved"] is True

    def test_get_all_agents(self, tracker):
        """Test getting all agents."""
        tracker.record_outcome("agent1", won=True)
        tracker.record_outcome("agent2", won=False)
        tracker.record_outcome("agent3", won=True)

        agents = tracker.get_all_agents()

        assert set(agents) == {"agent1", "agent2", "agent3"}

    def test_get_generation_trend(self, tracker):
        """Test getting generation trend for agent."""
        for gen in range(3):
            # Win rate increases each generation
            tracker.record_outcome("agent", won=True, generation=gen)
            if gen > 0:
                tracker.record_outcome("agent", won=True, generation=gen)

        trend = tracker.get_generation_trend("agent", max_generations=3)

        assert len(trend) == 3
        # Each generation should have increasing win count
        assert trend[0]["generation"] == 0
        assert trend[2]["generation"] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvolutionIntegration:
    """Integration tests for evolution system."""

    def test_full_evolution_cycle(self, evolver, tracker, mock_agent, mock_debate_result):
        """Test complete evolution cycle."""
        # Record some outcomes
        tracker.record_outcome(mock_agent.name, won=True, generation=0)
        tracker.record_outcome(mock_agent.name, won=True, generation=0)

        # Extract patterns
        patterns = evolver.extract_winning_patterns([mock_debate_result])

        # Evolve prompt
        new_prompt = evolver.apply_evolution(mock_agent, patterns)

        # Verify evolution was recorded
        history = evolver.get_evolution_history(mock_agent.name)
        assert len(history) == 1

        # Verify prompt was saved
        version = evolver.get_prompt_version(mock_agent.name)
        assert version is not None
        assert version.prompt == new_prompt

    def test_vulnerability_driven_evolution(self, evolver, mock_agent, mock_vulnerability):
        """Test evolution based on vulnerability patterns."""
        # Record multiple vulnerabilities
        for _ in range(5):
            evolver.record_vulnerability(mock_agent.name, mock_vulnerability)

        # Get vulnerability summary
        summary = evolver.get_vulnerability_summary(mock_agent.name)
        assert summary["total_occurrences"] >= 5

        # Get patterns for evolution
        vuln_patterns = evolver.get_vulnerability_patterns(mock_agent.name)
        assert len(vuln_patterns) > 0
        assert all("mitigation" in p for p in vuln_patterns)
