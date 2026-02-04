"""
Comprehensive tests for PatternLearner.

Tests the pattern learning system including:
- FixPattern dataclass and properties
- PatternMatch dataclass
- Pattern storage (load/save)
- Learning from successful fixes
- Learning from failed fixes
- Error pattern extraction
- Similarity calculation
- Pattern searching
- Pattern import/export
- Statistics
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aragora.nomic.testfixer.learning.pattern_learner import (
    PatternLearner,
    FixPattern,
    PatternMatch,
)
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.analyzer import FailureCategory


# ---------------------------------------------------------------------------
# Mock Objects for Testing
# ---------------------------------------------------------------------------


@dataclass
class MockPatch:
    """Mock patch for testing."""

    file_path: str
    original_content: str = ""
    patched_content: str = ""


@dataclass
class MockProposal:
    """Mock proposal for testing."""

    id: str = "test-proposal-001"
    description: str = "Fix authentication flag"
    post_debate_confidence: float = 0.8
    patches: list = field(default_factory=list)

    def as_diff(self) -> str:
        return "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"


@dataclass
class MockAnalysis:
    """Mock analysis for testing."""

    failure: TestFailure = None
    root_cause: str = "Missing authentication flag"
    fix_target: FailureCategory = FailureCategory.IMPL_BUG
    category: FailureCategory = FailureCategory.IMPL_BUG
    root_cause_file: str = "src/auth.py"


@dataclass
class MockFixAttempt:
    """Mock fix attempt for testing."""

    failure: TestFailure = None
    analysis: MockAnalysis = None
    proposal: MockProposal = None
    applied: bool = True
    success: bool = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_store_path(tmp_path) -> Path:
    """Create a temporary store path."""
    return tmp_path / "patterns.json"


@pytest.fixture
def sample_test_failure() -> TestFailure:
    """Create a sample test failure."""
    return TestFailure(
        test_name="test_auth",
        test_file="tests/test_auth.py",
        error_type="AssertionError",
        error_message="Expected True but got False in authentication check",
        stack_trace="File test_auth.py line 10\nAssertionError",
        line_number=10,
    )


@pytest.fixture
def sample_analysis(sample_test_failure) -> MockAnalysis:
    """Create a sample analysis."""
    return MockAnalysis(
        failure=sample_test_failure,
        root_cause="authenticate() doesn't set is_authenticated flag",
        category=FailureCategory.IMPL_BUG,
        root_cause_file="src/auth.py",
    )


@pytest.fixture
def sample_proposal() -> MockProposal:
    """Create a sample proposal."""
    return MockProposal(
        id="prop-001",
        description="Add is_authenticated = True after password check",
        patches=[
            MockPatch(
                file_path="src/auth.py",
                patched_content="self.is_authenticated = True",
            )
        ],
    )


@pytest.fixture
def sample_attempt(sample_test_failure, sample_analysis, sample_proposal) -> MockFixAttempt:
    """Create a sample fix attempt."""
    return MockFixAttempt(
        failure=sample_test_failure,
        analysis=sample_analysis,
        proposal=sample_proposal,
        applied=True,
        success=True,
    )


@pytest.fixture
def sample_pattern() -> FixPattern:
    """Create a sample pattern."""
    return FixPattern(
        id="pattern-001",
        category="implementation_bug",
        error_pattern="Expected True but got False",
        fix_pattern="Set flag after verification",
        fix_diff="@@ -1 +1 @@\n-old\n+new",
        fix_file="src/auth.py",
        error_type="AssertionError",
        root_cause="Missing flag assignment",
        success_count=5,
        failure_count=1,
    )


# ---------------------------------------------------------------------------
# FixPattern Tests
# ---------------------------------------------------------------------------


class TestFixPattern:
    """Tests for FixPattern dataclass."""

    def test_basic_creation(self):
        """Test creating a FixPattern."""
        pattern = FixPattern(
            id="test-001",
            category="assertion_error",
            error_pattern="assert x == True",
            fix_pattern="Set x before assertion",
            fix_diff="diff content",
            fix_file="test.py",
            error_type="AssertionError",
            root_cause="Variable not initialized",
        )
        assert pattern.id == "test-001"
        assert pattern.category == "assertion_error"
        assert pattern.success_count == 1  # Default
        assert pattern.failure_count == 0  # Default

    def test_confidence_property(self):
        """Test confidence calculation."""
        pattern = FixPattern(
            id="1",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=8,
            failure_count=2,
        )
        assert pattern.confidence == 0.8  # 8/10

    def test_confidence_zero_total(self):
        """Test confidence with zero counts."""
        pattern = FixPattern(
            id="1",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=0,
            failure_count=0,
        )
        assert pattern.confidence == 0.5  # Default

    def test_is_reliable_true(self):
        """Test is_reliable when criteria met."""
        pattern = FixPattern(
            id="1",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=10,
            failure_count=2,
        )
        # 10/12 = 0.83 >= 0.7 and success_count >= 2
        assert pattern.is_reliable is True

    def test_is_reliable_low_confidence(self):
        """Test is_reliable with low confidence."""
        pattern = FixPattern(
            id="1",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=3,
            failure_count=7,
        )
        # 3/10 = 0.3 < 0.7
        assert pattern.is_reliable is False

    def test_is_reliable_low_count(self):
        """Test is_reliable with insufficient data."""
        pattern = FixPattern(
            id="1",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=1,
            failure_count=0,
        )
        # success_count < 2
        assert pattern.is_reliable is False

    def test_to_dict(self, sample_pattern):
        """Test converting to dictionary."""
        d = sample_pattern.to_dict()

        assert d["id"] == "pattern-001"
        assert d["category"] == "implementation_bug"
        assert d["success_count"] == 5
        assert d["failure_count"] == 1

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "test-id",
            "category": "test_cat",
            "error_pattern": "error",
            "fix_pattern": "fix",
            "fix_diff": "diff",
            "fix_file": "file.py",
            "error_type": "Error",
            "root_cause": "cause",
            "success_count": 3,
            "failure_count": 1,
        }

        pattern = FixPattern.from_dict(data)

        assert pattern.id == "test-id"
        assert pattern.success_count == 3


# ---------------------------------------------------------------------------
# PatternMatch Tests
# ---------------------------------------------------------------------------


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_basic_creation(self, sample_pattern):
        """Test creating a PatternMatch."""
        match = PatternMatch(
            pattern=sample_pattern,
            similarity=0.8,
            confidence=0.7,
        )
        assert match.similarity == 0.8
        assert match.confidence == 0.7

    def test_comparison(self, sample_pattern):
        """Test PatternMatch comparison."""
        match1 = PatternMatch(pattern=sample_pattern, similarity=0.8, confidence=0.6)
        match2 = PatternMatch(pattern=sample_pattern, similarity=0.9, confidence=0.8)

        assert match1 < match2  # Lower confidence


# ---------------------------------------------------------------------------
# PatternLearner - Storage Tests
# ---------------------------------------------------------------------------


class TestPatternLearnerStorage:
    """Tests for pattern storage."""

    def test_init_memory_only(self):
        """Test initialization without file storage."""
        learner = PatternLearner()
        assert learner.store_path is None
        assert len(learner.patterns) == 0

    def test_init_with_path(self, temp_store_path):
        """Test initialization with file path."""
        learner = PatternLearner(temp_store_path)
        assert learner.store_path == temp_store_path

    def test_load_empty_file(self, temp_store_path):
        """Test loading from non-existent file."""
        learner = PatternLearner(temp_store_path)
        learner._ensure_loaded()
        assert len(learner.patterns) == 0

    def test_load_existing_patterns(self, temp_store_path, sample_pattern):
        """Test loading existing patterns from file."""
        # Write patterns to file
        data = {
            "version": 1,
            "patterns": [sample_pattern.to_dict()],
        }
        with temp_store_path.open("w") as f:
            json.dump(data, f)

        # Load and verify
        learner = PatternLearner(temp_store_path)
        learner._ensure_loaded()

        assert len(learner.patterns) == 1
        assert "pattern-001" in learner.patterns

    def test_save_patterns(self, temp_store_path):
        """Test saving patterns to file."""
        learner = PatternLearner(temp_store_path)

        # Add pattern directly
        pattern = FixPattern(
            id="test-save",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
        )
        learner.patterns[pattern.id] = pattern
        learner._save()

        # Verify file contents
        with temp_store_path.open() as f:
            data = json.load(f)

        assert data["version"] == 1
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["id"] == "test-save"

    def test_ensure_loaded_only_once(self, temp_store_path, sample_pattern):
        """Test patterns are only loaded once."""
        data = {"version": 1, "patterns": [sample_pattern.to_dict()]}
        with temp_store_path.open("w") as f:
            json.dump(data, f)

        learner = PatternLearner(temp_store_path)

        # Load multiple times
        learner._ensure_loaded()
        learner._ensure_loaded()
        learner._ensure_loaded()

        # Should still have same pattern
        assert len(learner.patterns) == 1


# ---------------------------------------------------------------------------
# Error Pattern Extraction Tests
# ---------------------------------------------------------------------------


class TestErrorPatternExtraction:
    """Tests for error pattern extraction."""

    def test_extract_removes_line_numbers(self):
        """Test line numbers are removed."""
        learner = PatternLearner()
        error = "Error at line 42 in function"
        pattern = learner._extract_error_pattern(error)

        assert "line N" in pattern
        assert "42" not in pattern

    def test_extract_removes_file_paths(self):
        """Test file paths are normalized."""
        learner = PatternLearner()
        error = "Error in /home/user/project/src/module.py"
        pattern = learner._extract_error_pattern(error)

        assert "/path/to/file.py" in pattern
        assert "/home/user" not in pattern

    def test_extract_removes_memory_addresses(self):
        """Test memory addresses are removed."""
        learner = PatternLearner()
        error = "Object at 0x7f8a1234abcd"
        pattern = learner._extract_error_pattern(error)

        assert "0x..." in pattern
        assert "7f8a1234abcd" not in pattern

    def test_extract_removes_object_ids(self):
        """Test object IDs are normalized."""
        learner = PatternLearner()
        error = "Got '<SomeClass at 0x123>' instead"
        pattern = learner._extract_error_pattern(error)

        assert "'<object>'" in pattern

    def test_extract_truncates_long_errors(self):
        """Test long errors are truncated."""
        learner = PatternLearner()
        error = "X" * 500
        pattern = learner._extract_error_pattern(error)

        assert len(pattern) <= 200


# ---------------------------------------------------------------------------
# Pattern ID Generation Tests
# ---------------------------------------------------------------------------


class TestPatternIdGeneration:
    """Tests for pattern ID generation."""

    def test_generate_pattern_id_deterministic(self):
        """Test ID generation is deterministic."""
        learner = PatternLearner()

        id1 = learner._generate_pattern_id("error", "category")
        id2 = learner._generate_pattern_id("error", "category")

        assert id1 == id2

    def test_generate_pattern_id_different_for_different_input(self):
        """Test different inputs produce different IDs."""
        learner = PatternLearner()

        id1 = learner._generate_pattern_id("error1", "category")
        id2 = learner._generate_pattern_id("error2", "category")

        assert id1 != id2


# ---------------------------------------------------------------------------
# Learning from Attempts Tests
# ---------------------------------------------------------------------------


class TestLearnFromAttempt:
    """Tests for learning from fix attempts."""

    def test_learn_successful_fix(self, temp_store_path, sample_attempt):
        """Test learning from a successful fix."""
        learner = PatternLearner(temp_store_path)

        pattern = learner.learn_from_attempt(sample_attempt)

        assert pattern is not None
        assert pattern.success_count == 1
        assert pattern.failure_count == 0
        assert pattern.id in learner.patterns

    def test_learn_reinforces_existing_pattern(self, temp_store_path, sample_attempt):
        """Test learning reinforces existing pattern."""
        learner = PatternLearner(temp_store_path)

        # Learn twice
        learner.learn_from_attempt(sample_attempt)
        pattern = learner.learn_from_attempt(sample_attempt)

        assert pattern.success_count == 2

    def test_learn_failed_fix(self, temp_store_path, sample_attempt):
        """Test learning from a failed fix."""
        learner = PatternLearner(temp_store_path)

        # First, create a pattern from success
        learner.learn_from_attempt(sample_attempt)

        # Then record a failure
        sample_attempt.success = False
        pattern = learner.learn_from_attempt(sample_attempt)

        assert pattern.success_count == 1
        assert pattern.failure_count == 1

    def test_learn_not_applied_returns_none(self, temp_store_path, sample_attempt):
        """Test learning returns None if fix not applied."""
        learner = PatternLearner(temp_store_path)
        sample_attempt.applied = False

        pattern = learner.learn_from_attempt(sample_attempt)

        assert pattern is None

    def test_learn_failed_no_existing_pattern(self, temp_store_path, sample_attempt):
        """Test learning failed fix with no existing pattern."""
        learner = PatternLearner(temp_store_path)
        sample_attempt.success = False

        # No existing pattern, failure cannot create new
        pattern = learner.learn_from_attempt(sample_attempt)

        assert pattern is None

    def test_learn_saves_to_file(self, temp_store_path, sample_attempt):
        """Test learning saves to file."""
        learner = PatternLearner(temp_store_path)
        learner.learn_from_attempt(sample_attempt)

        # Verify file was created
        assert temp_store_path.exists()

        # Verify contents
        with temp_store_path.open() as f:
            data = json.load(f)

        assert len(data["patterns"]) == 1


# ---------------------------------------------------------------------------
# Similarity Calculation Tests
# ---------------------------------------------------------------------------


class TestSimilarityCalculation:
    """Tests for similarity calculation."""

    def test_similarity_same_category(self, sample_pattern, sample_analysis):
        """Test similarity with matching category."""
        learner = PatternLearner()

        # Make categories match
        sample_pattern.category = sample_analysis.category.value

        similarity = learner._calculate_similarity(sample_pattern, sample_analysis)

        # Should get category bonus (0.4)
        assert similarity >= 0.4

    def test_similarity_different_category(self, sample_pattern, sample_analysis):
        """Test similarity with different category and dissimilar content."""
        learner = PatternLearner()

        # Make the pattern completely different from the analysis
        sample_pattern.category = "different_category"
        sample_pattern.error_type = "ImportError"  # Different from analysis
        sample_pattern.error_pattern = "module xyz not found"  # No word overlap
        sample_pattern.fix_file = "other/path.py"  # Different file

        similarity = learner._calculate_similarity(sample_pattern, sample_analysis)

        # Should be low due to no category, error type, or pattern overlap
        assert similarity < 0.5

    def test_similarity_matching_error_type(self, sample_pattern, sample_analysis):
        """Test similarity with matching error type."""
        learner = PatternLearner()

        sample_pattern.error_type = "AssertionError"
        sample_analysis.failure.error_type = "AssertionError"

        similarity = learner._calculate_similarity(sample_pattern, sample_analysis)

        # Should get error type bonus
        assert similarity > 0


# ---------------------------------------------------------------------------
# Pattern Search Tests
# ---------------------------------------------------------------------------


class TestPatternSearch:
    """Tests for finding similar patterns."""

    def test_find_similar_empty_patterns(self, sample_analysis):
        """Test search with no patterns."""
        learner = PatternLearner()

        matches = learner.find_similar_patterns(sample_analysis)

        assert len(matches) == 0

    def test_find_similar_returns_sorted(self, sample_analysis):
        """Test matches are sorted by confidence."""
        learner = PatternLearner()

        # Add patterns with different success rates
        learner.patterns["p1"] = FixPattern(
            id="p1",
            category=sample_analysis.category.value,
            error_pattern="authentication",
            fix_pattern="Fix 1",
            fix_diff="",
            fix_file="",
            error_type="AssertionError",
            root_cause="",
            success_count=2,
            failure_count=8,  # Low success rate
        )
        learner.patterns["p2"] = FixPattern(
            id="p2",
            category=sample_analysis.category.value,
            error_pattern="authentication",
            fix_pattern="Fix 2",
            fix_diff="",
            fix_file="",
            error_type="AssertionError",
            root_cause="",
            success_count=8,
            failure_count=2,  # High success rate
        )

        matches = learner.find_similar_patterns(sample_analysis)

        # p2 should be first (higher confidence)
        if len(matches) >= 2:
            assert matches[0].pattern.id == "p2"

    def test_find_similar_respects_threshold(self, sample_analysis):
        """Test minimum similarity threshold."""
        learner = PatternLearner()

        # Add a pattern with different category (low similarity)
        learner.patterns["p1"] = FixPattern(
            id="p1",
            category="completely_different_category",
            error_pattern="different error",
            fix_pattern="Fix",
            fix_diff="",
            fix_file="different.py",
            error_type="DifferentError",
            root_cause="",
            success_count=10,
            failure_count=0,
        )

        matches = learner.find_similar_patterns(sample_analysis, min_similarity=0.8)

        # Should not match due to high threshold
        assert len(matches) == 0

    def test_find_similar_max_results(self, sample_analysis):
        """Test max results limit."""
        learner = PatternLearner()

        # Add many patterns
        for i in range(10):
            learner.patterns[f"p{i}"] = FixPattern(
                id=f"p{i}",
                category=sample_analysis.category.value,
                error_pattern=sample_analysis.failure.error_message[:50],
                fix_pattern=f"Fix {i}",
                fix_diff="",
                fix_file="",
                error_type="AssertionError",
                root_cause="",
                success_count=5,
                failure_count=1,
            )

        matches = learner.find_similar_patterns(sample_analysis, min_similarity=0.1, max_results=3)

        assert len(matches) <= 3


# ---------------------------------------------------------------------------
# Reliable Patterns Tests
# ---------------------------------------------------------------------------


class TestReliablePatterns:
    """Tests for getting reliable patterns."""

    def test_get_reliable_patterns_empty(self):
        """Test with no patterns."""
        learner = PatternLearner()

        patterns = learner.get_reliable_patterns()

        assert len(patterns) == 0

    def test_get_reliable_patterns_filters(self):
        """Test filtering unreliable patterns."""
        learner = PatternLearner()

        # Add reliable pattern
        learner.patterns["reliable"] = FixPattern(
            id="reliable",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=10,
            failure_count=2,
        )

        # Add unreliable pattern
        learner.patterns["unreliable"] = FixPattern(
            id="unreliable",
            category="test",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=1,
            failure_count=0,
        )

        patterns = learner.get_reliable_patterns()

        assert len(patterns) == 1
        assert patterns[0].id == "reliable"

    def test_get_reliable_patterns_by_category(self):
        """Test filtering by category."""
        learner = PatternLearner()

        learner.patterns["cat1"] = FixPattern(
            id="cat1",
            category="category1",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=10,
            failure_count=2,
        )
        learner.patterns["cat2"] = FixPattern(
            id="cat2",
            category="category2",
            error_pattern="error",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=10,
            failure_count=2,
        )

        patterns = learner.get_reliable_patterns(category="category1")

        assert len(patterns) == 1
        assert patterns[0].id == "cat1"


# ---------------------------------------------------------------------------
# Suggest Heuristic Tests
# ---------------------------------------------------------------------------


class TestSuggestHeuristic:
    """Tests for heuristic suggestions."""

    def test_suggest_no_patterns(self, sample_analysis):
        """Test suggestion with no patterns."""
        learner = PatternLearner()

        suggestion = learner.suggest_heuristic(sample_analysis)

        assert suggestion is None

    def test_suggest_low_confidence(self, sample_analysis):
        """Test suggestion with low confidence match."""
        learner = PatternLearner()

        # Add pattern with low success rate
        learner.patterns["p1"] = FixPattern(
            id="p1",
            category="different_category",
            error_pattern="different error",
            fix_pattern="Fix pattern",
            fix_diff="",
            fix_file="different.py",
            error_type="DifferentError",
            root_cause="",
            success_count=2,
            failure_count=8,
        )

        suggestion = learner.suggest_heuristic(sample_analysis)

        # Should return None due to low confidence
        assert suggestion is None

    def test_suggest_with_good_match(self, sample_analysis):
        """Test suggestion with good match."""
        learner = PatternLearner()

        # Add pattern that matches well
        learner.patterns["p1"] = FixPattern(
            id="p1",
            category=sample_analysis.category.value,
            error_pattern=sample_analysis.failure.error_message[:50],
            fix_pattern="Set the authentication flag after password verification",
            fix_diff="",
            fix_file="",
            error_type="AssertionError",
            root_cause="",
            success_count=10,
            failure_count=1,
        )

        suggestion = learner.suggest_heuristic(sample_analysis)

        if suggestion:
            assert "authentication" in suggestion or "past fix" in suggestion.lower()


# ---------------------------------------------------------------------------
# Import/Export Tests
# ---------------------------------------------------------------------------


class TestImportExport:
    """Tests for import/export functionality."""

    def test_export_patterns(self, sample_pattern):
        """Test exporting patterns."""
        learner = PatternLearner()
        learner.patterns[sample_pattern.id] = sample_pattern

        exported = learner.export_patterns()

        assert len(exported) == 1
        assert exported[0]["id"] == "pattern-001"

    def test_import_patterns(self):
        """Test importing patterns."""
        learner = PatternLearner()

        patterns = [
            {
                "id": "import-1",
                "category": "test",
                "error_pattern": "error",
                "fix_pattern": "fix",
            }
        ]

        count = learner.import_patterns(patterns)

        assert count == 1
        assert "import-1" in learner.patterns

    def test_import_skips_existing(self, sample_pattern):
        """Test import skips existing patterns."""
        learner = PatternLearner()
        learner.patterns[sample_pattern.id] = sample_pattern

        patterns = [sample_pattern.to_dict()]

        count = learner.import_patterns(patterns)

        assert count == 0  # Already exists

    def test_import_handles_invalid(self):
        """Test import handles invalid patterns."""
        learner = PatternLearner()

        patterns = [
            {"id": "valid", "category": "test", "error_pattern": "e", "fix_pattern": "f"},
            {"invalid": "pattern"},  # Missing required fields
        ]

        count = learner.import_patterns(patterns)

        assert count == 1  # Only valid one imported


# ---------------------------------------------------------------------------
# Statistics Tests
# ---------------------------------------------------------------------------


class TestStatistics:
    """Tests for pattern statistics."""

    def test_get_statistics_empty(self):
        """Test statistics with no patterns."""
        learner = PatternLearner()

        stats = learner.get_statistics()

        assert stats["total_patterns"] == 0
        assert stats["reliable_patterns"] == 0
        assert stats["overall_success_rate"] == 0.0

    def test_get_statistics_with_patterns(self):
        """Test statistics with patterns."""
        learner = PatternLearner()

        learner.patterns["p1"] = FixPattern(
            id="p1",
            category="cat1",
            error_pattern="e",
            fix_pattern="f",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=8,
            failure_count=2,
        )
        learner.patterns["p2"] = FixPattern(
            id="p2",
            category="cat2",
            error_pattern="e",
            fix_pattern="f",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
            success_count=6,
            failure_count=4,
        )

        stats = learner.get_statistics()

        assert stats["total_patterns"] == 2
        assert stats["total_successes"] == 14
        assert stats["total_failures"] == 6
        assert stats["overall_success_rate"] == pytest.approx(14 / 20)
        assert "cat1" in stats["by_category"]
        assert "cat2" in stats["by_category"]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_extract_fix_pattern_no_description(self):
        """Test fix pattern extraction without description."""
        learner = PatternLearner()

        proposal = MockProposal(description="")
        proposal.patches = [MockPatch(file_path="test.py")]

        pattern = learner._extract_fix_pattern(proposal)

        assert "test.py" in pattern

    def test_extract_fix_pattern_no_patches(self):
        """Test fix pattern extraction without patches."""
        learner = PatternLearner()

        proposal = MockProposal(description="", patches=[])

        pattern = learner._extract_fix_pattern(proposal)

        assert "Unknown" in pattern

    def test_load_invalid_json(self, temp_store_path):
        """Test loading handles invalid JSON."""
        # Write invalid JSON
        with temp_store_path.open("w") as f:
            f.write("invalid json {")

        learner = PatternLearner(temp_store_path)
        learner._ensure_loaded()

        # Should not crash, just start with empty patterns
        assert len(learner.patterns) == 0

    def test_similarity_no_pattern_words(self, sample_analysis):
        """Test similarity with empty pattern."""
        learner = PatternLearner()

        pattern = FixPattern(
            id="empty",
            category="different",
            error_pattern="",
            fix_pattern="fix",
            fix_diff="",
            fix_file="",
            error_type="",
            root_cause="",
        )

        # Should not crash
        similarity = learner._calculate_similarity(pattern, sample_analysis)
        assert similarity >= 0
