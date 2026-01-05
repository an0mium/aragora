"""
Tests for the Prompt Evolution System.

Tests cover:
- EvolutionStrategy enum
- PromptVersion dataclass
- PromptEvolver (pattern extraction, storage, version management)
- Evolution strategies (APPEND, REPLACE, REFINE, HYBRID)
- Performance tracking
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from aragora.evolution.evolver import (
    EvolutionStrategy,
    PromptVersion,
    PromptEvolver,
)
from aragora.core import Agent, DebateResult, Critique


class TestEvolutionStrategyEnum:
    """Tests for EvolutionStrategy enumeration."""

    def test_all_strategy_values_defined(self):
        """Verify all expected strategy values exist."""
        expected = ["append", "replace", "refine", "hybrid"]
        actual = [s.value for s in EvolutionStrategy]
        assert sorted(expected) == sorted(actual)

    def test_strategy_values(self):
        """Test specific strategy values."""
        assert EvolutionStrategy.APPEND.value == "append"
        assert EvolutionStrategy.REPLACE.value == "replace"
        assert EvolutionStrategy.REFINE.value == "refine"
        assert EvolutionStrategy.HYBRID.value == "hybrid"


class TestPromptVersion:
    """Tests for PromptVersion dataclass."""

    def test_prompt_version_creation(self):
        """PromptVersion should be created with defaults."""
        version = PromptVersion(
            version=1,
            prompt="You are a helpful assistant",
            created_at="2026-01-05T00:00:00",
        )
        assert version.version == 1
        assert version.prompt == "You are a helpful assistant"
        assert version.performance_score == 0.0
        assert version.debates_count == 0
        assert version.consensus_rate == 0.0

    def test_prompt_version_with_metrics(self):
        """PromptVersion should store performance metrics."""
        version = PromptVersion(
            version=5,
            prompt="Advanced prompt",
            created_at="2026-01-05T00:00:00",
            performance_score=0.85,
            debates_count=100,
            consensus_rate=0.75,
            metadata={"origin": "evolution"},
        )
        assert version.performance_score == 0.85
        assert version.debates_count == 100
        assert version.consensus_rate == 0.75
        assert version.metadata["origin"] == "evolution"


class TestPromptEvolver:
    """Tests for PromptEvolver class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        """Create a PromptEvolver with temporary database."""
        return PromptEvolver(db_path=temp_db)

    def test_evolver_initialization(self, temp_db):
        """Evolver should initialize database on creation."""
        evolver = PromptEvolver(db_path=temp_db)
        assert evolver.db_path == Path(temp_db)
        assert evolver.strategy == EvolutionStrategy.APPEND

    def test_evolver_custom_strategy(self, temp_db):
        """Evolver should accept custom strategy."""
        evolver = PromptEvolver(db_path=temp_db, strategy=EvolutionStrategy.REPLACE)
        assert evolver.strategy == EvolutionStrategy.REPLACE


class TestPatternExtraction:
    """Tests for pattern extraction from debates."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    @pytest.fixture
    def successful_debate(self):
        """Create a mock successful debate result."""
        critique = Mock(spec=Critique)
        critique.issues = ["Missing error handling"]
        critique.suggestions = ["Add try-except blocks"]
        critique.severity = 0.5  # Low severity = addressed

        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = [critique]
        debate.final_answer = "Here is the solution:\n```python\nprint('hello')\n```"
        return debate

    def test_extract_winning_patterns_from_successful_debate(self, evolver, successful_debate):
        """Should extract patterns from successful debates."""
        patterns = evolver.extract_winning_patterns([successful_debate], min_confidence=0.6)

        assert len(patterns) > 0
        # Should have issue identification patterns
        issue_patterns = [p for p in patterns if p["type"] == "issue_identification"]
        assert len(issue_patterns) > 0
        assert issue_patterns[0]["text"] == "Missing error handling"

    def test_extract_winning_patterns_includes_code(self, evolver, successful_debate):
        """Should detect code inclusion pattern."""
        patterns = evolver.extract_winning_patterns([successful_debate])

        code_patterns = [p for p in patterns if p["type"] == "includes_code"]
        assert len(code_patterns) > 0

    def test_extract_patterns_skips_low_confidence(self, evolver):
        """Should skip debates with low confidence."""
        debate = Mock(spec=DebateResult)
        debate.consensus_reached = True
        debate.confidence = 0.4  # Below threshold
        debate.critiques = []
        debate.final_answer = "Answer"

        patterns = evolver.extract_winning_patterns([debate], min_confidence=0.6)

        assert len(patterns) == 0

    def test_extract_patterns_skips_no_consensus(self, evolver):
        """Should skip debates without consensus."""
        debate = Mock(spec=DebateResult)
        debate.consensus_reached = False
        debate.confidence = 0.9
        debate.critiques = []
        debate.final_answer = "Answer"

        patterns = evolver.extract_winning_patterns([debate])

        assert len(patterns) == 0

    def test_extract_structured_response_pattern(self, evolver):
        """Should detect structured response patterns."""
        debate = Mock(spec=DebateResult)
        debate.id = "debate-2"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = []
        debate.final_answer = "Step 1: First, do this\nStep 2: Then, do that"

        patterns = evolver.extract_winning_patterns([debate])

        structured = [p for p in patterns if p["type"] == "structured_response"]
        assert len(structured) > 0


class TestPatternStorage:
    """Tests for pattern storage and retrieval."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    def test_store_patterns(self, evolver):
        """Should store patterns in database."""
        patterns = [
            {"type": "issue_identification", "text": "Check for edge cases"},
            {"type": "improvement_suggestion", "text": "Add error handling"},
        ]

        evolver.store_patterns(patterns)

        stored = evolver.get_top_patterns(limit=10)
        assert len(stored) == 2

    def test_get_top_patterns_by_type(self, evolver):
        """Should filter patterns by type."""
        patterns = [
            {"type": "issue_identification", "text": "Pattern 1"},
            {"type": "improvement_suggestion", "text": "Pattern 2"},
            {"type": "issue_identification", "text": "Pattern 3"},
        ]

        evolver.store_patterns(patterns)

        issues = evolver.get_top_patterns(pattern_type="issue_identification")
        assert len(issues) == 2
        assert all(p["type"] == "issue_identification" for p in issues)

    def test_get_top_patterns_limit(self, evolver):
        """Should respect limit parameter."""
        patterns = [{"type": "test", "text": f"Pattern {i}"} for i in range(20)]

        evolver.store_patterns(patterns)

        top_5 = evolver.get_top_patterns(limit=5)
        assert len(top_5) == 5


class TestVersionManagement:
    """Tests for prompt version management."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    def test_save_prompt_version(self, evolver):
        """Should save new prompt version."""
        version_num = evolver.save_prompt_version(
            agent_name="claude",
            prompt="You are a helpful assistant",
            metadata={"source": "manual"},
        )

        assert version_num == 1

    def test_save_multiple_versions(self, evolver):
        """Should increment version numbers."""
        v1 = evolver.save_prompt_version("claude", "Prompt v1")
        v2 = evolver.save_prompt_version("claude", "Prompt v2")
        v3 = evolver.save_prompt_version("claude", "Prompt v3")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_get_prompt_version_latest(self, evolver):
        """Should get latest version when version not specified."""
        evolver.save_prompt_version("claude", "Prompt v1")
        evolver.save_prompt_version("claude", "Prompt v2")

        latest = evolver.get_prompt_version("claude")

        assert latest is not None
        assert latest.version == 2
        assert latest.prompt == "Prompt v2"

    def test_get_prompt_version_specific(self, evolver):
        """Should get specific version."""
        evolver.save_prompt_version("claude", "Prompt v1")
        evolver.save_prompt_version("claude", "Prompt v2")

        v1 = evolver.get_prompt_version("claude", version=1)

        assert v1 is not None
        assert v1.version == 1
        assert v1.prompt == "Prompt v1"

    def test_get_prompt_version_nonexistent(self, evolver):
        """Should return None for nonexistent version."""
        version = evolver.get_prompt_version("nonexistent")
        assert version is None


class TestEvolutionStrategies:
    """Tests for different evolution strategies."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock(spec=Agent)
        agent.name = "test-agent"
        agent.system_prompt = "You are a helpful assistant."
        return agent

    def test_evolve_append_strategy(self, evolver, mock_agent):
        """APPEND strategy should add learnings to end of prompt."""
        patterns = [
            {"type": "issue_identification", "text": "Check for edge cases"},
            {"type": "improvement_suggestion", "text": "Add error handling"},
        ]

        new_prompt = evolver.evolve_prompt(
            mock_agent,
            patterns=patterns,
            strategy=EvolutionStrategy.APPEND,
        )

        assert "You are a helpful assistant" in new_prompt
        assert "Learned patterns" in new_prompt
        assert "Check for edge cases" in new_prompt

    def test_evolve_replace_strategy(self, evolver, mock_agent):
        """REPLACE strategy should update learnings section."""
        # First evolution
        patterns1 = [{"type": "issue_identification", "text": "Old pattern"}]
        mock_agent.system_prompt = evolver._evolve_append("Base prompt", patterns1)

        # Second evolution with replace
        patterns2 = [{"type": "issue_identification", "text": "New pattern"}]
        new_prompt = evolver.evolve_prompt(
            mock_agent,
            patterns=patterns2,
            strategy=EvolutionStrategy.REPLACE,
        )

        assert "New pattern" in new_prompt
        # Old pattern should be replaced
        assert new_prompt.count("Learned patterns") == 1

    def test_evolve_hybrid_strategy(self, evolver, mock_agent):
        """HYBRID strategy should use append for short prompts."""
        patterns = [{"type": "improvement_suggestion", "text": "Be concise"}]

        new_prompt = evolver.evolve_prompt(
            mock_agent,
            patterns=patterns,
            strategy=EvolutionStrategy.HYBRID,
        )

        # Short prompt should use append
        assert "Learned patterns" in new_prompt

    def test_evolve_no_patterns(self, evolver, mock_agent):
        """Evolution with no patterns should return original prompt."""
        original = mock_agent.system_prompt
        new_prompt = evolver._evolve_append(original, [])
        assert new_prompt == original


class TestApplyEvolution:
    """Tests for apply_evolution method."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    @pytest.fixture
    def mock_agent(self):
        agent = Mock(spec=Agent)
        agent.name = "test-agent"
        agent.system_prompt = "Base prompt"
        return agent

    def test_apply_evolution_saves_version(self, evolver, mock_agent):
        """apply_evolution should save new version."""
        patterns = [{"type": "improvement_suggestion", "text": "Be clear"}]

        evolver.apply_evolution(mock_agent, patterns=patterns)

        version = evolver.get_prompt_version("test-agent")
        assert version is not None
        assert version.version == 1

    def test_apply_evolution_updates_agent(self, evolver, mock_agent):
        """apply_evolution should update agent's prompt."""
        patterns = [{"type": "improvement_suggestion", "text": "Be clear"}]

        evolver.apply_evolution(mock_agent, patterns=patterns)

        # Agent's set_system_prompt should have been called
        mock_agent.set_system_prompt.assert_called_once()

    def test_apply_evolution_records_history(self, evolver, mock_agent):
        """apply_evolution should record evolution history."""
        patterns = [{"type": "improvement_suggestion", "text": "Be clear"}]

        evolver.apply_evolution(mock_agent, patterns=patterns)

        history = evolver.get_evolution_history("test-agent")
        assert len(history) == 1
        assert history[0]["to_version"] == 1
        assert history[0]["strategy"] == "append"


class TestEvolutionHistory:
    """Tests for evolution history tracking."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    def test_get_evolution_history(self, evolver):
        """Should retrieve evolution history."""
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test-agent"
        mock_agent.system_prompt = "Base"

        # Apply multiple evolutions
        evolver.apply_evolution(mock_agent, [{"type": "test", "text": "p1"}])
        mock_agent.system_prompt = "Updated 1"
        evolver.apply_evolution(mock_agent, [{"type": "test", "text": "p2"}])

        history = evolver.get_evolution_history("test-agent")

        assert len(history) == 2

    def test_get_evolution_history_limit(self, evolver):
        """Should respect limit parameter."""
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test-agent"
        mock_agent.system_prompt = "Base"

        for i in range(10):
            evolver.apply_evolution(mock_agent, [{"type": "test", "text": f"p{i}"}])
            mock_agent.system_prompt = f"Updated {i}"

        history = evolver.get_evolution_history("test-agent", limit=5)

        assert len(history) == 5


class TestPerformanceTracking:
    """Tests for performance metric tracking."""

    @pytest.fixture
    def temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def evolver(self, temp_db):
        return PromptEvolver(db_path=temp_db)

    def test_update_performance(self, evolver):
        """Should update performance metrics for a version."""
        # Create a version first
        evolver.save_prompt_version("claude", "Test prompt")

        # Create mock debate result
        debate = Mock(spec=DebateResult)
        debate.consensus_reached = True
        debate.confidence = 0.85

        evolver.update_performance("claude", version=1, debate_result=debate)

        version = evolver.get_prompt_version("claude", version=1)
        assert version.debates_count == 1
        assert version.consensus_rate == 1.0  # First debate reached consensus
        assert version.performance_score == 0.85

    def test_update_performance_running_average(self, evolver):
        """Should compute running average of consensus rate."""
        evolver.save_prompt_version("claude", "Test prompt")

        # First debate: consensus
        debate1 = Mock(spec=DebateResult)
        debate1.consensus_reached = True
        debate1.confidence = 0.8
        evolver.update_performance("claude", 1, debate1)

        # Second debate: no consensus
        debate2 = Mock(spec=DebateResult)
        debate2.consensus_reached = False
        debate2.confidence = 0.4
        evolver.update_performance("claude", 1, debate2)

        version = evolver.get_prompt_version("claude", 1)
        assert version.debates_count == 2
        assert version.consensus_rate == 0.5  # 1/2 reached consensus
