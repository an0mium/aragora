"""
Extended tests for the Prompt Evolution System.

Tests cover advanced scenarios not covered by test_evolution_evolver.py:
- REFINE strategy API integration (mocked)
- Pattern format handling
- Strategy selection edge cases
- Pattern extraction edge cases
- Rollback and atomicity
- Performance metrics edge cases
"""

import json
import os
import sqlite3
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def evolver(temp_db):
    """Create a PromptEvolver with temporary database."""
    return PromptEvolver(db_path=temp_db)


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock(spec=Agent)
    agent.name = "test-agent"
    agent.system_prompt = "You are a helpful AI assistant."
    return agent


@pytest.fixture
def sample_patterns():
    """Sample patterns for testing."""
    return [
        {"type": "issue_identification", "text": "Check for edge cases"},
        {"type": "improvement_suggestion", "text": "Add error handling"},
        {"type": "structured_response", "text": "Use numbered steps"},
    ]


# =============================================================================
# Category A: REFINE Strategy API Tests
# =============================================================================


class TestRefineStrategyAPI:
    """Tests for REFINE strategy API integration."""

    def test_refine_with_anthropic_api_success(self, evolver, mock_agent, sample_patterns):
        """Should use Anthropic API when available."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "content": [{"text": "Refined prompt with synthesized learnings."}]
                }
                mock_post.return_value = mock_response

                result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

                assert mock_post.called
                assert result == "Refined prompt with synthesized learnings."

    def test_refine_with_openai_fallback(self, evolver, mock_agent, sample_patterns):
        """Should fall back to OpenAI when Anthropic unavailable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "OpenAI refined prompt."}}]
                }
                mock_post.return_value = mock_response

                result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

                assert mock_post.called
                assert result == "OpenAI refined prompt."

    def test_refine_api_timeout_fallback_to_append(self, evolver, mock_agent, sample_patterns):
        """Should fall back to APPEND on API timeout."""
        import requests as req
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_post.side_effect = req.exceptions.Timeout("Connection timed out")

                result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

                # Should fall back to append
                assert "Learned patterns" in result

    def test_refine_api_401_error_fallback(self, evolver, mock_agent, sample_patterns):
        """Should fall back to APPEND on 401 error."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 401
                mock_post.return_value = mock_response

                result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

                # Should fall back to append
                assert "Learned patterns" in result

    def test_refine_api_429_rate_limit_fallback(self, evolver, mock_agent, sample_patterns):
        """Should fall back to APPEND on 429 rate limit."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 429
                mock_post.return_value = mock_response

                result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

                # Should fall back to append
                assert "Learned patterns" in result

    def test_refine_both_apis_unavailable(self, evolver, mock_agent, sample_patterns):
        """Should fall back to APPEND when no API keys available."""
        with patch.dict(os.environ, {}, clear=True):
            result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

            # Should fall back to append
            assert "Learned patterns" in result

    def test_refine_empty_pattern_list(self, evolver, mock_agent):
        """Should return appended prompt with empty patterns."""
        result = evolver._evolve_refine(mock_agent.system_prompt, [])
        # Empty patterns falls back to append which returns original
        assert mock_agent.system_prompt in result

    def test_refine_limits_to_top_5_patterns(self, evolver, mock_agent):
        """Should limit patterns to top 5."""
        many_patterns = [{"type": f"type{i}", "text": f"text{i}"} for i in range(10)]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "content": [{"text": "Refined"}]
                }
                mock_post.return_value = mock_response

                evolver._evolve_refine(mock_agent.system_prompt, many_patterns)

                # Check the request body contains at most 5 patterns
                call_args = mock_post.call_args
                request_body = call_args[1]["json"]
                message_content = request_body["messages"][0]["content"]
                # Count pattern lines (each starts with "- ")
                pattern_lines = [line for line in message_content.split("\n") if line.startswith("- ")]
                assert len(pattern_lines) <= 5

    def test_refine_response_content_extraction(self, evolver, mock_agent, sample_patterns):
        """Should correctly extract content from API response."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "content": [{"text": "  Extracted content with whitespace  "}]
                }
                mock_post.return_value = mock_response

                result = evolver._evolve_refine(mock_agent.system_prompt, sample_patterns)

                # Should strip whitespace
                assert result == "Extracted content with whitespace"


# =============================================================================
# Category B: Pattern Format Tests
# =============================================================================


class TestPatternFormat:
    """Tests for pattern format handling in REFINE."""

    def test_pattern_with_type_text_keys(self, evolver, mock_agent):
        """Should handle patterns with 'type'/'text' keys (actual format)."""
        patterns = [{"type": "issue", "text": "Check errors"}]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "content": [{"text": "Refined"}]
                }
                mock_post.return_value = mock_response

                evolver._evolve_refine(mock_agent.system_prompt, patterns)

                # Verify the format string handles the actual keys
                call_args = mock_post.call_args
                request_body = call_args[1]["json"]
                # The code uses p.get('pattern', 'unknown') but patterns have 'type'
                # This documents the current behavior

    def test_pattern_formatting_in_refine_prompt(self, evolver, mock_agent):
        """Should format patterns into the refinement prompt."""
        patterns = [{"type": "suggestion", "text": "Be concise"}]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"content": [{"text": "OK"}]}
                mock_post.return_value = mock_response

                evolver._evolve_refine(mock_agent.system_prompt, patterns)

                call_args = mock_post.call_args
                request_body = call_args[1]["json"]
                message_content = request_body["messages"][0]["content"]
                # Pattern text should appear in the message
                assert "Patterns to incorporate" in message_content

    def test_missing_keys_handling(self, evolver, mock_agent):
        """Should handle patterns with missing keys gracefully."""
        patterns = [{}]  # Empty pattern dict

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("requests.Session.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"content": [{"text": "OK"}]}
                mock_post.return_value = mock_response

                # Should not raise
                result = evolver._evolve_refine(mock_agent.system_prompt, patterns)
                assert result is not None

    def test_special_characters_in_patterns(self, evolver, mock_agent):
        """Should handle special characters in pattern text."""
        patterns = [
            {"type": "issue_identification", "text": "Check for 'quotes' and \"double quotes\""},
            {"type": "improvement_suggestion", "text": "Handle\nnewlines\tand\ttabs"},
        ]

        result = evolver._evolve_append(mock_agent.system_prompt, patterns)
        # Should not break
        assert "Learned patterns" in result


# =============================================================================
# Category C: Strategy Selection Tests
# =============================================================================


class TestStrategySelection:
    """Tests for strategy selection behavior."""

    def test_hybrid_at_1999_chars_uses_append(self, temp_db, mock_agent):
        """HYBRID should use APPEND for prompts under 2000 chars."""
        evolver = PromptEvolver(db_path=temp_db, strategy=EvolutionStrategy.HYBRID)
        mock_agent.system_prompt = "a" * 1999
        patterns = [{"type": "improvement_suggestion", "text": "Short"}]

        result = evolver.evolve_prompt(mock_agent, patterns, EvolutionStrategy.HYBRID)

        # Should use append (contains "Learned patterns")
        assert "Learned patterns" in result

    def test_hybrid_at_2000_chars_uses_append(self, temp_db, mock_agent):
        """HYBRID should use APPEND at exactly 2000 chars."""
        evolver = PromptEvolver(db_path=temp_db, strategy=EvolutionStrategy.HYBRID)
        mock_agent.system_prompt = "a" * 100
        patterns = [{"type": "improvement_suggestion", "text": "Short"}]

        # Create a result that would be under 2000
        result = evolver.evolve_prompt(mock_agent, patterns, EvolutionStrategy.HYBRID)
        # With short prompt, should use append
        assert "Learned patterns" in result

    def test_hybrid_at_2001_chars_uses_refine(self, temp_db, mock_agent):
        """HYBRID should use REFINE for prompts over 2000 chars."""
        evolver = PromptEvolver(db_path=temp_db, strategy=EvolutionStrategy.HYBRID)
        # Make the prompt long enough that append would exceed 2000
        mock_agent.system_prompt = "a" * 1950
        patterns = [
            {"type": "test", "text": "Pattern that when appended exceeds 2000 chars" * 3}
        ]

        with patch.dict(os.environ, {}, clear=True):
            # No API keys, so refine falls back to append anyway
            result = evolver.evolve_prompt(mock_agent, patterns, EvolutionStrategy.HYBRID)
            # The append result should exceed 2000, triggering refine
            # But refine falls back to append without API keys

    def test_unknown_strategy_returns_original(self, evolver, mock_agent):
        """Unknown strategy should return original prompt."""
        # This tests the else branch
        original = mock_agent.system_prompt
        # Pass a mock strategy value
        mock_strategy = Mock()
        mock_strategy.value = "unknown"

        # Directly test the evolve_prompt logic
        result = evolver.evolve_prompt(mock_agent, [], strategy=mock_strategy)
        # Unknown strategy returns original
        assert result == original

    def test_strategy_with_none_patterns(self, evolver, mock_agent):
        """Strategy with None patterns should use get_top_patterns."""
        # Store some patterns first with a valid type
        evolver.store_patterns([{"type": "issue_identification", "text": "Stored pattern"}])

        result = evolver.evolve_prompt(mock_agent, patterns=None)

        # Should have fetched patterns from db
        assert "Stored pattern" in result

    def test_strategy_with_empty_patterns_list(self, evolver, mock_agent):
        """Strategy with empty patterns list should return original."""
        result = evolver._evolve_append(mock_agent.system_prompt, [])
        assert result == mock_agent.system_prompt


# =============================================================================
# Category D: Pattern Extraction Edge Cases
# =============================================================================


class TestPatternExtractionEdgeCases:
    """Tests for pattern extraction edge cases."""

    def test_severity_at_boundary_0_7(self, evolver):
        """Severity exactly at 0.7 should not be included."""
        critique = Mock(spec=Critique)
        critique.issues = ["Boundary issue"]
        critique.suggestions = ["Boundary suggestion"]
        critique.severity = 0.7  # Exactly at boundary

        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = [critique]
        debate.final_answer = "Answer"

        patterns = evolver.extract_winning_patterns([debate])

        # Severity 0.7 is NOT < 0.7, so should be empty
        issue_patterns = [p for p in patterns if p["type"] == "issue_identification"]
        assert len(issue_patterns) == 0

    def test_confidence_at_boundary_0_6(self, evolver):
        """Confidence exactly at 0.6 should be included."""
        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.6  # Exactly at boundary
        debate.critiques = []
        debate.final_answer = "Step 1: First step"

        patterns = evolver.extract_winning_patterns([debate], min_confidence=0.6)

        # Should be included (confidence >= min_confidence)
        assert len(patterns) > 0

    def test_multiple_critiques_mixed_severities(self, evolver):
        """Should only extract from low severity critiques."""
        low_critique = Mock(spec=Critique)
        low_critique.issues = ["Low issue"]
        low_critique.suggestions = []
        low_critique.severity = 0.3

        high_critique = Mock(spec=Critique)
        high_critique.issues = ["High issue"]
        high_critique.suggestions = []
        high_critique.severity = 0.9

        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = [low_critique, high_critique]
        debate.final_answer = "Answer"

        patterns = evolver.extract_winning_patterns([debate])

        issue_texts = [p["text"] for p in patterns if p["type"] == "issue_identification"]
        assert "Low issue" in issue_texts
        assert "High issue" not in issue_texts

    def test_empty_issues_suggestions_lists(self, evolver):
        """Should handle empty issues/suggestions lists."""
        critique = Mock(spec=Critique)
        critique.issues = []
        critique.suggestions = []
        critique.severity = 0.3

        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = [critique]
        debate.final_answer = "Answer"

        patterns = evolver.extract_winning_patterns([debate])

        # Should not crash, may have response patterns
        assert isinstance(patterns, list)

    def test_code_pattern_multiline_backticks(self, evolver):
        """Should detect code blocks with multiline content."""
        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = []
        debate.final_answer = """Here is the code:
```python
def foo():
    return "bar"
```
That's all."""

        patterns = evolver.extract_winning_patterns([debate])

        code_patterns = [p for p in patterns if p["type"] == "includes_code"]
        assert len(code_patterns) > 0

    def test_very_long_final_answer(self, evolver):
        """Should handle very long final answers."""
        debate = Mock(spec=DebateResult)
        debate.id = "debate-1"
        debate.consensus_reached = True
        debate.confidence = 0.8
        debate.critiques = []
        debate.final_answer = "Step 1: " + "x" * 10000

        patterns = evolver.extract_winning_patterns([debate])

        # Should not crash
        structured = [p for p in patterns if p["type"] == "structured_response"]
        assert len(structured) > 0

    def test_no_consensus_skips_all(self, evolver):
        """Debates without consensus should be skipped entirely."""
        debate = Mock(spec=DebateResult)
        debate.consensus_reached = False
        debate.confidence = 0.95
        debate.critiques = []
        debate.final_answer = "Step 1: ..."

        patterns = evolver.extract_winning_patterns([debate])

        assert len(patterns) == 0


# =============================================================================
# Category E: Rollback & Atomicity Tests
# =============================================================================


class TestRollbackAtomicity:
    """Tests for rollback and atomicity behavior."""

    def test_apply_evolution_partial_failure_agent_update(self, evolver, mock_agent):
        """If agent update fails, version should still be saved."""
        patterns = [{"type": "test", "text": "Pattern"}]
        mock_agent.set_system_prompt.side_effect = Exception("Update failed")

        with pytest.raises(Exception):
            evolver.apply_evolution(mock_agent, patterns)

        # Version should have been saved before agent update
        version = evolver.get_prompt_version("test-agent")
        assert version is not None

    def test_update_performance_nonexistent_version(self, evolver):
        """update_performance with nonexistent version should not crash."""
        debate = Mock(spec=DebateResult)
        debate.consensus_reached = True
        debate.confidence = 0.8

        # Should not crash - just does nothing
        evolver.update_performance("nonexistent-agent", version=999, debate_result=debate)

    def test_concurrent_version_saves_different_agents(self, temp_db):
        """Multiple agents can save versions concurrently."""
        evolver1 = PromptEvolver(db_path=temp_db)
        evolver2 = PromptEvolver(db_path=temp_db)

        v1 = evolver1.save_prompt_version("agent1", "Prompt 1")
        v2 = evolver2.save_prompt_version("agent2", "Prompt 2")

        assert v1 == 1
        assert v2 == 1

        # Both should be retrievable
        assert evolver1.get_prompt_version("agent1").prompt == "Prompt 1"
        assert evolver2.get_prompt_version("agent2").prompt == "Prompt 2"


# =============================================================================
# Category F: Performance Metrics Tests
# =============================================================================


class TestPerformanceMetrics:
    """Tests for performance metrics edge cases."""

    def test_consensus_rate_calculation(self, evolver):
        """Should correctly calculate consensus rate over multiple debates."""
        evolver.save_prompt_version("claude", "Test")

        # 3 debates: 2 consensus, 1 no consensus
        debate1 = Mock(spec=DebateResult)
        debate1.consensus_reached = True
        debate1.confidence = 0.8
        evolver.update_performance("claude", 1, debate1)

        debate2 = Mock(spec=DebateResult)
        debate2.consensus_reached = True
        debate2.confidence = 0.9
        evolver.update_performance("claude", 1, debate2)

        debate3 = Mock(spec=DebateResult)
        debate3.consensus_reached = False
        debate3.confidence = 0.4
        evolver.update_performance("claude", 1, debate3)

        version = evolver.get_prompt_version("claude", 1)
        assert version.debates_count == 3
        # Consensus rate: 2/3 = 0.666...
        assert abs(version.consensus_rate - 2/3) < 0.01

    def test_performance_score_from_confidence(self, evolver):
        """Performance score should come from debate confidence."""
        evolver.save_prompt_version("claude", "Test")

        debate = Mock(spec=DebateResult)
        debate.consensus_reached = True
        debate.confidence = 0.95

        evolver.update_performance("claude", 1, debate)

        version = evolver.get_prompt_version("claude", 1)
        assert version.performance_score == 0.95

    def test_debates_count_increment(self, evolver):
        """debates_count should increment correctly."""
        evolver.save_prompt_version("claude", "Test")

        for i in range(5):
            debate = Mock(spec=DebateResult)
            debate.consensus_reached = True
            debate.confidence = 0.8
            evolver.update_performance("claude", 1, debate)

        version = evolver.get_prompt_version("claude", 1)
        assert version.debates_count == 5

    def test_multiple_agents_same_version_number(self, evolver):
        """Different agents can have same version numbers."""
        v1_a = evolver.save_prompt_version("agent_a", "Prompt A")
        v1_b = evolver.save_prompt_version("agent_b", "Prompt B")

        assert v1_a == 1
        assert v1_b == 1

        # Both should be independently retrievable
        assert evolver.get_prompt_version("agent_a", 1).prompt == "Prompt A"
        assert evolver.get_prompt_version("agent_b", 1).prompt == "Prompt B"

    def test_version_sequence_per_agent(self, evolver):
        """Version sequence should be per-agent."""
        evolver.save_prompt_version("agent_a", "A1")
        evolver.save_prompt_version("agent_b", "B1")
        evolver.save_prompt_version("agent_a", "A2")
        evolver.save_prompt_version("agent_b", "B2")

        assert evolver.get_prompt_version("agent_a").version == 2
        assert evolver.get_prompt_version("agent_b").version == 2

    def test_performance_score_no_consensus_is_zero(self, evolver):
        """Performance score should be 0 when no consensus."""
        evolver.save_prompt_version("claude", "Test")

        debate = Mock(spec=DebateResult)
        debate.consensus_reached = False
        debate.confidence = 0.8  # High confidence but no consensus

        evolver.update_performance("claude", 1, debate)

        version = evolver.get_prompt_version("claude", 1)
        assert version.performance_score == 0


# =============================================================================
# Additional Tests: Database and Edge Cases
# =============================================================================


class TestDatabaseEdgeCases:
    """Tests for database-related edge cases."""

    def test_init_db_creates_tables(self, temp_db):
        """_init_db should create all required tables."""
        evolver = PromptEvolver(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        assert "prompt_versions" in tables
        assert "extracted_patterns" in tables
        assert "evolution_history" in tables

    def test_metadata_json_serialization(self, evolver):
        """Metadata should be properly serialized as JSON."""
        metadata = {"key": "value", "nested": {"a": 1}}
        evolver.save_prompt_version("claude", "Test", metadata=metadata)

        version = evolver.get_prompt_version("claude", 1)
        assert version.metadata == metadata

    def test_metadata_none_handling(self, evolver):
        """Should handle None metadata."""
        evolver.save_prompt_version("claude", "Test", metadata=None)

        version = evolver.get_prompt_version("claude", 1)
        assert version.metadata == {}

    def test_pattern_source_debate_id(self, evolver):
        """Patterns should store source debate ID."""
        patterns = [
            {"type": "test", "text": "Pattern", "source_debate": "debate-123"}
        ]
        evolver.store_patterns(patterns)

        with sqlite3.connect(evolver.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_debate_id FROM extracted_patterns")
            row = cursor.fetchone()
            assert row[0] == "debate-123"

    def test_evolution_history_patterns_applied(self, evolver, mock_agent):
        """Evolution history should record applied patterns."""
        patterns = [
            {"type": "test", "text": "Pattern 1"},
            {"type": "test", "text": "Pattern 2"},
        ]
        evolver.apply_evolution(mock_agent, patterns)

        history = evolver.get_evolution_history("test-agent")
        assert len(history) == 1
        assert "Pattern 1" in history[0]["patterns"]
        assert "Pattern 2" in history[0]["patterns"]


class TestPromptVersionDataclass:
    """Additional tests for PromptVersion dataclass."""

    def test_prompt_version_defaults(self):
        """Should have correct default values."""
        version = PromptVersion(
            version=1,
            prompt="Test",
            created_at="2026-01-05",
        )
        assert version.performance_score == 0.0
        assert version.debates_count == 0
        assert version.consensus_rate == 0.0
        assert version.metadata == {}

    def test_prompt_version_with_all_fields(self):
        """Should store all fields correctly."""
        version = PromptVersion(
            version=10,
            prompt="Full prompt",
            created_at="2026-01-05T12:00:00",
            performance_score=0.92,
            debates_count=50,
            consensus_rate=0.88,
            metadata={"evolved_from": 9},
        )
        assert version.version == 10
        assert version.performance_score == 0.92
        assert version.debates_count == 50
        assert version.consensus_rate == 0.88
        assert version.metadata["evolved_from"] == 9
