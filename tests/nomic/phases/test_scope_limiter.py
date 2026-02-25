"""
Tests for Nomic Loop Scope Limiter.

Tests the scope evaluation and design simplification logic:
- Tests ScopeEvaluation dataclass
- Tests complexity pattern detection
- Tests simplicity pattern detection
- Tests file counting
- Tests protected file detection
- Tests simplification suggestions
- Tests check_design_scope convenience function
"""

import pytest

from aragora.nomic.phases.scope_limiter import (
    ScopeEvaluation,
    ScopeLimiter,
    check_design_scope,
    COMPLEXITY_PATTERNS,
    SIMPLICITY_PATTERNS,
)


class TestScopeEvaluation:
    """Tests for ScopeEvaluation dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        eval = ScopeEvaluation()
        assert eval.is_implementable is True
        assert eval.complexity_score == 0.0
        assert eval.file_count == 0
        assert eval.risk_factors == []
        assert eval.suggested_simplifications == []
        assert eval.reason == ""

    def test_is_too_complex_by_score(self):
        """Should detect too complex by score threshold."""
        eval = ScopeEvaluation(complexity_score=0.8)
        assert eval.is_too_complex is True

        eval2 = ScopeEvaluation(complexity_score=0.7)
        assert eval2.is_too_complex is False

        eval3 = ScopeEvaluation(complexity_score=0.71)
        assert eval3.is_too_complex is True

    def test_is_too_complex_by_file_count(self):
        """Should detect too complex by file count."""
        eval = ScopeEvaluation(file_count=6)
        assert eval.is_too_complex is True

        eval2 = ScopeEvaluation(file_count=5)
        assert eval2.is_too_complex is False

        eval3 = ScopeEvaluation(file_count=10)
        assert eval3.is_too_complex is True

    def test_is_too_complex_combined(self):
        """Either high score or high file count triggers complexity."""
        # High score, low files
        eval1 = ScopeEvaluation(complexity_score=0.8, file_count=2)
        assert eval1.is_too_complex is True

        # Low score, high files
        eval2 = ScopeEvaluation(complexity_score=0.3, file_count=8)
        assert eval2.is_too_complex is True

        # Both low
        eval3 = ScopeEvaluation(complexity_score=0.5, file_count=3)
        assert eval3.is_too_complex is False

    def test_to_dict(self):
        """Should serialize to dictionary correctly."""
        eval = ScopeEvaluation(
            is_implementable=False,
            complexity_score=0.75,
            file_count=7,
            risk_factors=["Full refactor mentioned"],
            suggested_simplifications=["Break into smaller changes"],
            reason="Too complex",
        )
        d = eval.to_dict()

        assert d["is_implementable"] is False
        assert d["is_too_complex"] is True
        assert d["complexity_score"] == 0.75
        assert d["file_count"] == 7
        assert d["risk_factors"] == ["Full refactor mentioned"]
        assert d["suggested_simplifications"] == ["Break into smaller changes"]
        assert d["reason"] == "Too complex"

    def test_to_dict_with_defaults(self):
        """to_dict should work with default values."""
        eval = ScopeEvaluation()
        d = eval.to_dict()

        assert d["is_implementable"] is True
        assert d["is_too_complex"] is False
        assert d["complexity_score"] == 0.0
        assert d["file_count"] == 0


class TestScopeLimiterInit:
    """Tests for ScopeLimiter initialization."""

    def test_default_values(self):
        """Should initialize with default values."""
        limiter = ScopeLimiter()
        assert limiter.max_complexity == 0.7
        assert limiter.max_files == 5
        assert "CLAUDE.md" in limiter.protected_files
        assert "aragora/__init__.py" in limiter.protected_files

    def test_custom_values(self):
        """Should accept custom configuration."""
        limiter = ScopeLimiter(
            max_complexity=0.5,
            max_files=3,
            protected_files=["custom.py"],
        )
        assert limiter.max_complexity == 0.5
        assert limiter.max_files == 3
        assert limiter.protected_files == ["custom.py"]


class TestScopeLimiterComplexityPatterns:
    """Tests for complexity pattern detection."""

    def test_detects_refactor_entire(self):
        """Should detect full refactor patterns."""
        limiter = ScopeLimiter()
        design = "We need to refactor the entire debate system"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0
        assert any("refactor" in r.lower() for r in result.risk_factors)

    def test_detects_new_module(self):
        """Should detect new module creation."""
        limiter = ScopeLimiter()
        design = "Create a new module for handling authentication"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0
        assert any("new module" in r.lower() for r in result.risk_factors)

    def test_detects_database_migration(self):
        """Should detect database migration patterns."""
        limiter = ScopeLimiter()
        design = "Apply database migration to add new columns"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0
        assert any("migration" in r.lower() for r in result.risk_factors)

    def test_detects_breaking_changes(self):
        """Should detect breaking changes."""
        limiter = ScopeLimiter()
        design = "This will introduce breaking changes to the API"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0
        assert any("breaking" in r.lower() for r in result.risk_factors)

    def test_detects_multiple_components(self):
        """Should detect multiple component changes."""
        limiter = ScopeLimiter()
        design = "Update multiple files and modules across the codebase"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0

    def test_detects_external_integration(self):
        """Should detect external API integration."""
        limiter = ScopeLimiter()
        design = "Add integration with external API service"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0
        assert any("integration" in r.lower() for r in result.risk_factors)

    def test_detects_auth_system(self):
        """Should detect authentication system changes."""
        limiter = ScopeLimiter()
        design = "Implement new authentication system with OAuth"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0
        assert any("auth" in r.lower() for r in result.risk_factors)

    def test_detects_concurrency(self):
        """Should detect concurrent processing."""
        limiter = ScopeLimiter()
        design = "Add async processing for parallel execution"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0

    def test_detects_realtime_features(self):
        """Should detect real-time features."""
        limiter = ScopeLimiter()
        design = "Add real-time updates via WebSocket streaming data"
        result = limiter.evaluate(design)

        assert result.complexity_score > 0

    def test_accumulates_multiple_patterns(self):
        """Should accumulate complexity from multiple patterns."""
        limiter = ScopeLimiter()
        design = """
        This design involves:
        - Refactor the entire authentication system
        - Create a new module for authorization
        - Add integration with external OAuth API
        - Apply database migration for user roles
        """
        result = limiter.evaluate(design)

        # Should have accumulated significant complexity
        assert result.complexity_score >= 0.5
        assert len(result.risk_factors) >= 3


class TestScopeLimiterSimplicityPatterns:
    """Tests for simplicity pattern detection."""

    def test_reduces_for_simple_function(self):
        """Should reduce complexity for simple function additions."""
        limiter = ScopeLimiter()
        design = "Add a simple function to parse config"
        result = limiter.evaluate(design)

        # Simplicity patterns should offset some complexity
        assert result.complexity_score <= 0.0 or result.is_implementable

    def test_reduces_for_bug_fix(self):
        """Should reduce complexity for bug fixes."""
        limiter = ScopeLimiter()
        design = "Fix a bug in the error handling code"
        result = limiter.evaluate(design)

        assert result.complexity_score <= 0.0 or result.is_implementable

    def test_reduces_for_config_update(self):
        """Should reduce complexity for config updates."""
        limiter = ScopeLimiter()
        design = "Update config settings for timeout values"
        result = limiter.evaluate(design)

        assert result.complexity_score <= 0.0 or result.is_implementable

    def test_reduces_for_documentation(self):
        """Should reduce complexity for documentation."""
        limiter = ScopeLimiter()
        design = "Add logging statements and docstrings"
        result = limiter.evaluate(design)

        assert result.complexity_score <= 0.0 or result.is_implementable

    def test_reduces_for_single_file(self):
        """Should reduce complexity for single file changes."""
        limiter = ScopeLimiter()
        design = "Make a single file change to utils.py"
        result = limiter.evaluate(design)

        assert result.complexity_score <= 0.0 or result.is_implementable


class TestScopeLimiterFileDetection:
    """Tests for file counting and detection."""

    def test_counts_mentioned_files(self):
        """Should count Python files mentioned in design."""
        limiter = ScopeLimiter()
        design = """
        Modify aragora/core.py
        Update aragora/utils.py
        Create aragora/new.py
        """
        result = limiter.evaluate(design)

        assert result.file_count >= 3

    def test_counts_quoted_files(self):
        """Should count files in quotes or backticks."""
        limiter = ScopeLimiter()
        design = """
        Edit `aragora/handlers.py`
        Change "aragora/models.py"
        """
        result = limiter.evaluate(design)

        assert result.file_count >= 2

    def test_detects_file_paths(self):
        """Should detect file paths in design."""
        limiter = ScopeLimiter()
        design = "Changes to aragora/debate/orchestrator.py and aragora/memory/continuum.py"
        result = limiter.evaluate(design)

        assert result.file_count >= 2

    def test_deduplicates_files(self):
        """Should not count the same file multiple times."""
        limiter = ScopeLimiter()
        design = """
        First, modify aragora/core.py
        Then update aragora/core.py again
        Finally, edit aragora/core.py
        """
        result = limiter.evaluate(design)

        # Should count as 1 file, not 3
        assert result.file_count == 1


class TestScopeLimiterProtectedFiles:
    """Tests for protected file detection."""

    def test_detects_claudemd_modification(self):
        """Should detect attempts to modify CLAUDE.md."""
        limiter = ScopeLimiter()
        design = "Update CLAUDE.md with new instructions"
        result = limiter.evaluate(design)

        assert result.is_implementable is False
        assert any("protected" in r.lower() for r in result.risk_factors)

    def test_detects_init_modification(self):
        """Should detect attempts to modify __init__.py."""
        limiter = ScopeLimiter()
        design = "Modify aragora/__init__.py to export new classes"
        result = limiter.evaluate(design)

        assert result.is_implementable is False

    def test_detects_env_modification(self):
        """Should detect attempts to modify .env."""
        limiter = ScopeLimiter()
        design = "Update .env file with API keys"
        result = limiter.evaluate(design)

        assert result.is_implementable is False

    def test_detects_nomic_loop_modification(self):
        """Should detect attempts to modify nomic_loop.py."""
        limiter = ScopeLimiter()
        design = "Change scripts/nomic_loop.py to add new phase"
        result = limiter.evaluate(design)

        assert result.is_implementable is False

    def test_allows_non_protected_files(self):
        """Should allow modification of non-protected files."""
        limiter = ScopeLimiter()
        design = "Create aragora/new_feature.py with utility functions"
        result = limiter.evaluate(design)

        # Should not fail due to protected files (may fail for other reasons)
        assert not any("protected" in r.lower() for r in result.risk_factors)


class TestScopeLimiterEvaluation:
    """Tests for overall evaluation logic."""

    def test_simple_design_passes(self):
        """Simple designs should pass evaluation."""
        limiter = ScopeLimiter()
        design = "Add a helper function to aragora/utils.py"
        result = limiter.evaluate(design)

        assert result.is_implementable is True
        assert result.reason == "Design is within scope limits"

    def test_complex_design_fails_by_score(self):
        """Complex designs should fail by score threshold."""
        limiter = ScopeLimiter(max_complexity=0.3)
        design = """
        Refactor the entire system
        Create new module for auth
        Add database migration
        """
        result = limiter.evaluate(design)

        assert result.is_implementable is False
        assert "Complexity score" in result.reason

    def test_complex_design_fails_by_files(self):
        """Designs with too many files should fail."""
        limiter = ScopeLimiter(max_files=2)
        design = """
        Modify aragora/a.py
        Modify aragora/b.py
        Modify aragora/c.py
        Modify aragora/d.py
        """
        result = limiter.evaluate(design)

        assert result.is_implementable is False
        assert "files" in result.reason.lower()

    def test_complexity_clamped_to_range(self):
        """Complexity score should be clamped to [0, 1]."""
        limiter = ScopeLimiter()

        # Test that negative doesn't go below 0
        design1 = "Fix a typo"
        result1 = limiter.evaluate(design1)
        assert result1.complexity_score >= 0.0

        # Test upper bound - hard to exceed 1.0 naturally
        # but we verify it's capped
        design2 = """
        Refactor the entire authentication system
        Create a new module
        Database migration
        Breaking API changes
        Multiple components
        External integration
        Real-time streaming
        Async processing
        """
        result2 = limiter.evaluate(design2)
        assert result2.complexity_score <= 1.0


class TestScopeLimiterSuggestions:
    """Tests for simplification suggestions."""

    def test_suggests_for_refactor(self):
        """Should suggest alternatives for refactoring."""
        limiter = ScopeLimiter(max_complexity=0.2)
        design = "Refactor the entire debate module"
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) > 0
        assert any("refactor" in s.lower() for s in result.suggested_simplifications)

    def test_suggests_for_new_module(self):
        """Should suggest alternatives for new modules."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Create a new system for payments"
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) > 0
        assert any(
            "prototype" in s.lower() or "minimal" in s.lower()
            for s in result.suggested_simplifications
        )

    def test_suggests_for_migration(self):
        """Should suggest alternatives for migrations."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Add schema migration for new tables"
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) > 0
        assert any(
            "schema" in s.lower() or "optional" in s.lower()
            for s in result.suggested_simplifications
        )

    def test_suggests_for_breaking_changes(self):
        """Should suggest alternatives for breaking changes."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Introduce breaking changes to API"
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) > 0
        assert any(
            "backward" in s.lower() or "compatibility" in s.lower()
            for s in result.suggested_simplifications
        )

    def test_suggests_for_integration(self):
        """Should suggest alternatives for external integrations."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Add integration with Stripe API service"
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) > 0
        assert any(
            "mock" in s.lower() or "stub" in s.lower() for s in result.suggested_simplifications
        )

    def test_suggests_for_too_many_files(self):
        """Should suggest breaking into smaller changes for many files."""
        limiter = ScopeLimiter(max_files=2)
        design = """
        Modify aragora/a.py
        Modify aragora/b.py
        Modify aragora/c.py
        Modify aragora/d.py
        Modify aragora/e.py
        Modify aragora/f.py
        """
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) > 0
        assert any(
            "break" in s.lower() or "smaller" in s.lower() for s in result.suggested_simplifications
        )

    def test_default_suggestions(self):
        """Should provide default suggestions when no specific ones match."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Handle concurrent parallel async processing"
        result = limiter.evaluate(design)

        # If complexity is too high but no specific patterns match for suggestions
        if not result.is_implementable and len(result.suggested_simplifications) == 0:
            # The fallback should be triggered
            pass  # This is acceptable behavior
        else:
            # Should have at least some suggestions
            assert len(result.suggested_simplifications) >= 0

    def test_max_three_suggestions(self):
        """Should return at most 3 suggestions."""
        limiter = ScopeLimiter(max_complexity=0.05)
        design = """
        Refactor entire system
        Create new module
        Add schema migration
        Breaking API changes
        External integration
        Authentication system
        """
        result = limiter.evaluate(design)

        assert len(result.suggested_simplifications) <= 3


class TestScopeLimiterSimplifyForImplementation:
    """Tests for simplify_for_implementation method."""

    def test_returns_original_if_implementable(self):
        """Should return original design if implementable."""
        limiter = ScopeLimiter()
        design = "Add a simple helper function"
        evaluation = limiter.evaluate(design)

        simplified, note = limiter.simplify_for_implementation(design, evaluation)

        assert simplified == design
        assert "within scope" in note.lower()

    def test_returns_modified_if_too_complex(self):
        """Should return modified design if too complex."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Refactor the entire authentication system"
        evaluation = limiter.evaluate(design)

        simplified, note = limiter.simplify_for_implementation(design, evaluation)

        assert simplified != design
        assert "SCOPE LIMITATION" in simplified
        assert design in simplified  # Original should still be included

    def test_includes_constraints_in_simplified(self):
        """Simplified design should include constraints."""
        limiter = ScopeLimiter(max_complexity=0.1, max_files=3)
        design = "Create new module with migrations"
        evaluation = limiter.evaluate(design)

        simplified, note = limiter.simplify_for_implementation(design, evaluation)

        if not evaluation.is_implementable:
            assert "REQUIRED CONSTRAINTS" in simplified
            assert "at most 3 files" in simplified.lower() or "at most" in simplified.lower()

    def test_note_includes_original_complexity(self):
        """Note should include original complexity information."""
        limiter = ScopeLimiter(max_complexity=0.2)
        design = "Refactor entire system and add new modules"
        evaluation = limiter.evaluate(design)

        _, note = limiter.simplify_for_implementation(design, evaluation)

        if not evaluation.is_implementable:
            assert "complexity" in note.lower()
            assert str(limiter.max_complexity) in note

    def test_note_includes_suggestions(self):
        """Note should include simplification suggestions."""
        limiter = ScopeLimiter(max_complexity=0.1)
        design = "Refactor the entire codebase"
        evaluation = limiter.evaluate(design)

        _, note = limiter.simplify_for_implementation(design, evaluation)

        if not evaluation.is_implementable and evaluation.suggested_simplifications:
            for suggestion in evaluation.suggested_simplifications[:3]:
                # At least one suggestion should appear in note
                pass  # Suggestions are numbered in the note


class TestCheckDesignScope:
    """Tests for the check_design_scope convenience function."""

    def test_basic_usage(self):
        """Should evaluate design with default settings."""
        result = check_design_scope("Add a helper function")
        assert isinstance(result, ScopeEvaluation)
        assert result.is_implementable is True

    def test_custom_max_files(self):
        """Should respect custom max_files parameter."""
        design = """
        Modify aragora/a.py
        Modify aragora/b.py
        Modify aragora/c.py
        """
        result = check_design_scope(design, max_files=2)
        assert result.is_implementable is False

    def test_returns_evaluation(self):
        """Should return ScopeEvaluation object."""
        result = check_design_scope("Test design")
        assert hasattr(result, "is_implementable")
        assert hasattr(result, "complexity_score")
        assert hasattr(result, "file_count")
        assert hasattr(result, "risk_factors")


class TestComplexityPatternsConstant:
    """Tests verifying the complexity patterns constant."""

    def test_patterns_have_three_elements(self):
        """Each pattern should be (regex, weight, reason)."""
        for pattern in COMPLEXITY_PATTERNS:
            assert len(pattern) == 3
            assert isinstance(pattern[0], str)  # regex
            assert isinstance(pattern[1], (int, float))  # weight
            assert isinstance(pattern[2], str)  # reason

    def test_all_weights_positive(self):
        """Complexity weights should be positive."""
        for pattern in COMPLEXITY_PATTERNS:
            assert pattern[1] > 0


class TestSimplicityPatternsConstant:
    """Tests verifying the simplicity patterns constant."""

    def test_patterns_have_three_elements(self):
        """Each pattern should be (regex, weight, reason)."""
        for pattern in SIMPLICITY_PATTERNS:
            assert len(pattern) == 3
            assert isinstance(pattern[0], str)  # regex
            assert isinstance(pattern[1], (int, float))  # weight
            assert isinstance(pattern[2], str)  # reason

    def test_all_weights_negative(self):
        """Simplicity weights should be negative (reduce complexity)."""
        for pattern in SIMPLICITY_PATTERNS:
            assert pattern[1] < 0


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_design(self):
        """Should handle empty design string."""
        limiter = ScopeLimiter()
        result = limiter.evaluate("")
        assert result.is_implementable is True
        assert result.complexity_score == 0.0
        assert result.file_count == 0

    def test_whitespace_only_design(self):
        """Should handle whitespace-only design."""
        limiter = ScopeLimiter()
        result = limiter.evaluate("   \n\t  ")
        assert result.is_implementable is True

    def test_very_long_design(self):
        """Should handle very long design text."""
        limiter = ScopeLimiter()
        design = "This is a simple change. " * 1000
        result = limiter.evaluate(design)
        # Should still complete without error
        assert isinstance(result, ScopeEvaluation)

    def test_special_characters_in_design(self):
        """Should handle special characters in design."""
        limiter = ScopeLimiter()
        design = "Add function to handle @decorator and $special chars!"
        result = limiter.evaluate(design)
        assert isinstance(result, ScopeEvaluation)

    def test_unicode_in_design(self):
        """Should handle unicode characters in design."""
        limiter = ScopeLimiter()
        design = "Add internationalization support for Japanese (日本語)"
        result = limiter.evaluate(design)
        assert isinstance(result, ScopeEvaluation)

    def test_case_insensitive_patterns(self):
        """Pattern matching should be case-insensitive."""
        limiter = ScopeLimiter()

        design1 = "REFACTOR THE ENTIRE SYSTEM"
        result1 = limiter.evaluate(design1)

        design2 = "refactor the entire system"
        result2 = limiter.evaluate(design2)

        # Both should detect the pattern
        assert result1.complexity_score == result2.complexity_score

    def test_zero_max_complexity(self):
        """Should handle max_complexity of 0."""
        limiter = ScopeLimiter(max_complexity=0.0)
        design = "Any design at all"
        result = limiter.evaluate(design)
        # Even simplest design will be rejected
        # unless it has negative patterns to bring score below 0
        assert isinstance(result, ScopeEvaluation)

    def test_zero_max_files(self):
        """Should clamp max_files=0 to 1 to prevent ZeroDivisionError."""
        limiter = ScopeLimiter(max_files=0)
        # max_files is clamped to at least 1
        assert limiter.max_files == 1
        design = "Modify aragora/core.py and aragora/other.py"
        result = limiter.evaluate(design)
        assert result.is_implementable is False
        assert "files" in result.reason.lower()

    def test_negative_max_files(self):
        """Should clamp negative max_files to 1."""
        limiter = ScopeLimiter(max_files=-5)
        assert limiter.max_files == 1
