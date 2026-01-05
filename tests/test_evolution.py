"""
Tests for prompt evolution system.

Tests:
- Evolution strategy enumeration
- PromptVersion dataclass
- Pattern extraction from debates
- Prompt version management
- Evolution strategies (append, replace, refine, hybrid)
- Performance tracking
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from aragora.evolution.evolver import (
    EvolutionStrategy,
    PromptVersion,
    PromptEvolver,
)


class TestEvolutionStrategy:
    """Test EvolutionStrategy enumeration."""

    def test_all_strategies_exist(self):
        """All evolution strategies should be defined."""
        assert EvolutionStrategy.APPEND.value == "append"
        assert EvolutionStrategy.REPLACE.value == "replace"
        assert EvolutionStrategy.REFINE.value == "refine"
        assert EvolutionStrategy.HYBRID.value == "hybrid"

    def test_strategies_are_unique(self):
        """Each strategy should have unique value."""
        values = [s.value for s in EvolutionStrategy]
        assert len(values) == len(set(values))


class TestPromptVersion:
    """Test PromptVersion dataclass."""

    def test_create_version(self):
        """Should create prompt version with all fields."""
        version = PromptVersion(
            version=1,
            prompt="You are a helpful assistant.",
            created_at="2024-01-01T00:00:00",
            performance_score=0.85,
            debates_count=10,
            consensus_rate=0.8,
            metadata={"strategy": "append"},
        )

        assert version.version == 1
        assert version.prompt == "You are a helpful assistant."
        assert version.performance_score == 0.85
        assert version.debates_count == 10
        assert version.consensus_rate == 0.8
        assert version.metadata["strategy"] == "append"

    def test_default_values(self):
        """Should have sensible defaults."""
        version = PromptVersion(
            version=1,
            prompt="test",
            created_at="2024-01-01",
        )

        assert version.performance_score == 0.0
        assert version.debates_count == 0
        assert version.consensus_rate == 0.0
        assert version.metadata == {}


class TestPromptEvolver:
    """Test PromptEvolver class."""

    def test_init_creates_db(self):
        """Should create database file on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "evolution.db"
            evolver = PromptEvolver(db_path=str(db_path))

            assert db_path.exists()

    def test_init_default_strategy(self):
        """Should default to APPEND strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "evolution.db"
            evolver = PromptEvolver(db_path=str(db_path))

            assert evolver.strategy == EvolutionStrategy.APPEND

    def test_init_custom_strategy(self):
        """Should accept custom strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "evolution.db"
            evolver = PromptEvolver(
                db_path=str(db_path),
                strategy=EvolutionStrategy.HYBRID,
            )

            assert evolver.strategy == EvolutionStrategy.HYBRID


class TestPatternExtraction:
    """Test pattern extraction from debates."""

    def test_extract_from_empty_list(self):
        """Should return empty list for no debates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            patterns = evolver.extract_winning_patterns([])
            assert patterns == []

    def test_extract_skips_low_confidence(self):
        """Should skip debates below confidence threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            debate = Mock()
            debate.consensus_reached = True
            debate.confidence = 0.3  # Below default 0.6 threshold
            debate.critiques = []
            debate.final_answer = ""

            patterns = evolver.extract_winning_patterns([debate], min_confidence=0.6)
            assert patterns == []

    def test_extract_skips_no_consensus(self):
        """Should skip debates without consensus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            debate = Mock()
            debate.consensus_reached = False
            debate.confidence = 0.9
            debate.critiques = []
            debate.final_answer = ""

            patterns = evolver.extract_winning_patterns([debate])
            assert patterns == []

    def test_extract_critique_patterns(self):
        """Should extract patterns from critiques."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            critique = Mock()
            critique.severity = 0.5  # Addressed (< 0.7)
            critique.issues = ["Missing error handling"]
            critique.suggestions = ["Add try-catch block"]

            debate = Mock()
            debate.id = "d-123"
            debate.consensus_reached = True
            debate.confidence = 0.9
            debate.critiques = [critique]
            debate.final_answer = ""

            patterns = evolver.extract_winning_patterns([debate])

            issue_patterns = [p for p in patterns if p["type"] == "issue_identification"]
            suggestion_patterns = [p for p in patterns if p["type"] == "improvement_suggestion"]

            assert len(issue_patterns) >= 1
            assert "Missing error handling" in issue_patterns[0]["text"]

            assert len(suggestion_patterns) >= 1
            assert "Add try-catch block" in suggestion_patterns[0]["text"]

    def test_extract_code_pattern(self):
        """Should detect code inclusion pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            debate = Mock()
            debate.id = "d-123"
            debate.consensus_reached = True
            debate.confidence = 0.9
            debate.critiques = []
            debate.final_answer = "Here's the solution:\n```python\nprint('hello')\n```"

            patterns = evolver.extract_winning_patterns([debate])

            code_patterns = [p for p in patterns if p["type"] == "includes_code"]
            assert len(code_patterns) >= 1

    def test_extract_structured_response_pattern(self):
        """Should detect structured response pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            debate = Mock()
            debate.id = "d-123"
            debate.consensus_reached = True
            debate.confidence = 0.9
            debate.critiques = []
            debate.final_answer = "Step 1: First, do this\nStep 2: Then do that"

            patterns = evolver.extract_winning_patterns([debate])

            structured_patterns = [p for p in patterns if p["type"] == "structured_response"]
            assert len(structured_patterns) >= 1


class TestPatternStorage:
    """Test pattern storage and retrieval."""

    def test_store_patterns(self):
        """Should store patterns in database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            patterns = [
                {"type": "issue_identification", "text": "Check for null"},
                {"type": "improvement_suggestion", "text": "Use async/await", "source_debate": "d-1"},
            ]

            evolver.store_patterns(patterns)

            # Retrieve to verify
            retrieved = evolver.get_top_patterns(limit=10)
            assert len(retrieved) == 2

    def test_get_top_patterns_by_type(self):
        """Should filter patterns by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            patterns = [
                {"type": "issue_identification", "text": "Issue 1"},
                {"type": "issue_identification", "text": "Issue 2"},
                {"type": "improvement_suggestion", "text": "Suggestion 1"},
            ]

            evolver.store_patterns(patterns)

            issues = evolver.get_top_patterns(pattern_type="issue_identification", limit=10)
            assert len(issues) == 2
            assert all(p["type"] == "issue_identification" for p in issues)

    def test_get_top_patterns_limit(self):
        """Should respect limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            patterns = [{"type": "test", "text": f"Pattern {i}"} for i in range(10)]
            evolver.store_patterns(patterns)

            top_3 = evolver.get_top_patterns(limit=3)
            assert len(top_3) == 3


class TestPromptVersioning:
    """Test prompt version management."""

    def test_save_first_version(self):
        """Should save first version with version=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            version = evolver.save_prompt_version(
                agent_name="test-agent",
                prompt="You are a helpful assistant.",
            )

            assert version == 1

    def test_save_increments_version(self):
        """Should increment version number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            v1 = evolver.save_prompt_version("agent", "Prompt v1")
            v2 = evolver.save_prompt_version("agent", "Prompt v2")
            v3 = evolver.save_prompt_version("agent", "Prompt v3")

            assert v1 == 1
            assert v2 == 2
            assert v3 == 3

    def test_separate_agents_separate_versions(self):
        """Different agents should have separate version numbering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            v1_a = evolver.save_prompt_version("agent-a", "Prompt A")
            v1_b = evolver.save_prompt_version("agent-b", "Prompt B")
            v2_a = evolver.save_prompt_version("agent-a", "Prompt A v2")

            assert v1_a == 1
            assert v1_b == 1  # Separate sequence
            assert v2_a == 2

    def test_get_prompt_version_specific(self):
        """Should retrieve specific version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            evolver.save_prompt_version("agent", "Prompt v1", {"note": "first"})
            evolver.save_prompt_version("agent", "Prompt v2", {"note": "second"})

            v1 = evolver.get_prompt_version("agent", version=1)
            assert v1 is not None
            assert v1.prompt == "Prompt v1"
            assert v1.metadata["note"] == "first"

    def test_get_prompt_version_latest(self):
        """Should retrieve latest version when no version specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            evolver.save_prompt_version("agent", "Prompt v1")
            evolver.save_prompt_version("agent", "Prompt v2")
            evolver.save_prompt_version("agent", "Prompt v3")

            latest = evolver.get_prompt_version("agent")
            assert latest is not None
            assert latest.version == 3
            assert latest.prompt == "Prompt v3"

    def test_get_prompt_version_nonexistent(self):
        """Should return None for non-existent version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            result = evolver.get_prompt_version("nonexistent")
            assert result is None


class TestEvolutionStrategies:
    """Test different evolution strategies."""

    def test_evolve_append(self):
        """Append strategy should add learnings section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.system_prompt = "You are a helpful assistant."

            patterns = [
                {"type": "issue_identification", "text": "Check for edge cases"},
                {"type": "improvement_suggestion", "text": "Use type hints"},
            ]

            new_prompt = evolver.evolve_prompt(
                agent,
                patterns=patterns,
                strategy=EvolutionStrategy.APPEND,
            )

            assert "helpful assistant" in new_prompt
            assert "Learned patterns from successful debates" in new_prompt
            assert "Watch for: Check for edge cases" in new_prompt
            assert "Consider: Use type hints" in new_prompt

    def test_evolve_append_empty_patterns(self):
        """Append with no patterns should return original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.system_prompt = "Original prompt"

            new_prompt = evolver.evolve_prompt(
                agent,
                patterns=[],
                strategy=EvolutionStrategy.APPEND,
            )

            assert new_prompt == "Original prompt"

    def test_evolve_replace_removes_old_learnings(self):
        """Replace strategy should remove old learnings section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.system_prompt = """You are a helpful assistant.

Learned patterns from successful debates:
- Watch for: Old pattern 1
- Consider: Old pattern 2"""

            patterns = [
                {"type": "issue_identification", "text": "New pattern"},
            ]

            new_prompt = evolver.evolve_prompt(
                agent,
                patterns=patterns,
                strategy=EvolutionStrategy.REPLACE,
            )

            assert "Old pattern 1" not in new_prompt
            assert "Old pattern 2" not in new_prompt
            assert "New pattern" in new_prompt

    def test_evolve_hybrid_short_prompt(self):
        """Hybrid with short result should use append."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.system_prompt = "Short prompt"

            patterns = [{"type": "issue_identification", "text": "Test"}]

            new_prompt = evolver.evolve_prompt(
                agent,
                patterns=patterns,
                strategy=EvolutionStrategy.HYBRID,
            )

            # Should use append since result is short
            assert "Learned patterns" in new_prompt

    def test_evolve_uses_default_strategy(self):
        """Should use default strategy when none specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(
                db_path=f"{tmpdir}/test.db",
                strategy=EvolutionStrategy.REPLACE,
            )

            agent = Mock()
            agent.system_prompt = "Original"

            patterns = [{"type": "test", "text": "Pattern"}]

            # evolve_prompt should use evolver.strategy when strategy=None
            new_prompt = evolver.evolve_prompt(agent, patterns=patterns)

            # Result depends on REPLACE strategy behavior
            assert new_prompt is not None


class TestApplyEvolution:
    """Test apply_evolution integration."""

    def test_apply_evolution_saves_version(self):
        """Apply evolution should save new prompt version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.name = "test-agent"
            agent.system_prompt = "Original prompt"

            patterns = [{"type": "test", "text": "New learning"}]

            evolver.apply_evolution(agent, patterns)

            # Should have saved a version
            version = evolver.get_prompt_version("test-agent")
            assert version is not None
            assert version.version == 1

    def test_apply_evolution_updates_agent(self):
        """Apply evolution should update agent's prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.name = "test-agent"
            agent.system_prompt = "Original prompt"

            patterns = [{"type": "issue_identification", "text": "Check types"}]

            evolver.apply_evolution(agent, patterns)

            # Agent.set_system_prompt should have been called
            agent.set_system_prompt.assert_called_once()

    def test_apply_evolution_records_history(self):
        """Apply evolution should record in history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.name = "test-agent"
            agent.system_prompt = "Original"

            evolver.apply_evolution(agent, [{"type": "test", "text": "Pattern"}])

            history = evolver.get_evolution_history("test-agent")
            assert len(history) == 1
            assert history[0]["to_version"] == 1


class TestEvolutionHistory:
    """Test evolution history tracking."""

    def test_get_history_empty(self):
        """Should return empty list for agent with no history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            history = evolver.get_evolution_history("nonexistent")
            assert history == []

    def test_get_history_ordered(self):
        """History should contain all evolutions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.name = "test-agent"
            agent.system_prompt = "Prompt"

            # Apply multiple evolutions
            for i in range(3):
                evolver.apply_evolution(agent, [{"type": "test", "text": f"Pattern {i}"}])

            history = evolver.get_evolution_history("test-agent", limit=10)

            assert len(history) == 3
            # All versions should be present
            versions = {h["to_version"] for h in history}
            assert versions == {1, 2, 3}

    def test_get_history_limit(self):
        """Should respect limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            agent = Mock()
            agent.name = "test-agent"
            agent.system_prompt = "Prompt"

            for i in range(5):
                evolver.apply_evolution(agent, [{"type": "test", "text": f"P{i}"}])

            history = evolver.get_evolution_history("test-agent", limit=2)
            assert len(history) == 2


class TestPerformanceTracking:
    """Test performance metric updates."""

    def test_update_performance_first_debate(self):
        """Should update metrics after first debate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            # Save initial version
            evolver.save_prompt_version("agent", "Prompt v1")

            debate = Mock()
            debate.consensus_reached = True
            debate.confidence = 0.9

            evolver.update_performance("agent", version=1, debate_result=debate)

            version = evolver.get_prompt_version("agent", version=1)
            assert version.debates_count == 1
            assert version.consensus_rate == 1.0  # 1 debate, 1 consensus

    def test_update_performance_multiple_debates(self):
        """Should calculate running average for consensus rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            evolver.save_prompt_version("agent", "Prompt v1")

            # First debate - consensus
            debate1 = Mock()
            debate1.consensus_reached = True
            debate1.confidence = 0.9
            evolver.update_performance("agent", 1, debate1)

            # Second debate - no consensus
            debate2 = Mock()
            debate2.consensus_reached = False
            debate2.confidence = 0.4
            evolver.update_performance("agent", 1, debate2)

            version = evolver.get_prompt_version("agent", version=1)
            assert version.debates_count == 2
            assert version.consensus_rate == 0.5  # 1/2 debates reached consensus

    def test_update_performance_nonexistent_version(self):
        """Should handle non-existent version gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = PromptEvolver(db_path=f"{tmpdir}/test.db")

            debate = Mock()
            debate.consensus_reached = True
            debate.confidence = 0.9

            # Should not raise
            evolver.update_performance("nonexistent", version=99, debate_result=debate)
