"""
Tests for the aragora.prompts.code_review module.

Tests cover:
- Prompt constants (SECURITY_PROMPT, PERFORMANCE_PROMPT, QUALITY_PROMPT)
- get_focus_prompts function
- build_review_prompt function
- get_role_prompt function
- MAX_DIFF_SIZE constant
- DEFAULT_FOCUS_AREAS constant
"""

import pytest


class TestModuleExports:
    """Tests for module exports."""

    def test_can_import_module(self):
        """Module can be imported."""
        from aragora.prompts import code_review

        assert code_review is not None

    def test_security_prompt_exported(self):
        """SECURITY_PROMPT is exported from package."""
        from aragora.prompts import SECURITY_PROMPT

        assert SECURITY_PROMPT is not None

    def test_performance_prompt_exported(self):
        """PERFORMANCE_PROMPT is exported from package."""
        from aragora.prompts import PERFORMANCE_PROMPT

        assert PERFORMANCE_PROMPT is not None

    def test_quality_prompt_exported(self):
        """QUALITY_PROMPT is exported from package."""
        from aragora.prompts import QUALITY_PROMPT

        assert QUALITY_PROMPT is not None

    def test_build_review_prompt_exported(self):
        """build_review_prompt is exported from package."""
        from aragora.prompts import build_review_prompt

        assert callable(build_review_prompt)

    def test_get_focus_prompts_exported(self):
        """get_focus_prompts is exported from package."""
        from aragora.prompts import get_focus_prompts

        assert callable(get_focus_prompts)

    def test_all_exports_in_init(self):
        """All expected symbols are in __all__."""
        from aragora.prompts import __all__

        expected = [
            "SECURITY_PROMPT",
            "PERFORMANCE_PROMPT",
            "QUALITY_PROMPT",
            "build_review_prompt",
            "get_focus_prompts",
        ]
        for item in expected:
            assert item in __all__


class TestConstants:
    """Tests for module constants."""

    def test_max_diff_size_is_positive(self):
        """MAX_DIFF_SIZE is a positive integer."""
        from aragora.prompts.code_review import MAX_DIFF_SIZE

        assert isinstance(MAX_DIFF_SIZE, int)
        assert MAX_DIFF_SIZE > 0

    def test_max_diff_size_reasonable(self):
        """MAX_DIFF_SIZE is within reasonable bounds."""
        from aragora.prompts.code_review import MAX_DIFF_SIZE

        # Should be at least 10KB for meaningful diffs
        assert MAX_DIFF_SIZE >= 10_000
        # But not too large (less than 1MB)
        assert MAX_DIFF_SIZE < 1_000_000

    def test_max_diff_size_is_50kb(self):
        """MAX_DIFF_SIZE is 50KB as documented."""
        from aragora.prompts.code_review import MAX_DIFF_SIZE

        assert MAX_DIFF_SIZE == 50_000

    def test_default_focus_areas_is_list(self):
        """DEFAULT_FOCUS_AREAS is a list."""
        from aragora.prompts.code_review import DEFAULT_FOCUS_AREAS

        assert isinstance(DEFAULT_FOCUS_AREAS, list)

    def test_default_focus_areas_contains_security(self):
        """DEFAULT_FOCUS_AREAS includes security."""
        from aragora.prompts.code_review import DEFAULT_FOCUS_AREAS

        assert "security" in DEFAULT_FOCUS_AREAS

    def test_default_focus_areas_contains_performance(self):
        """DEFAULT_FOCUS_AREAS includes performance."""
        from aragora.prompts.code_review import DEFAULT_FOCUS_AREAS

        assert "performance" in DEFAULT_FOCUS_AREAS

    def test_default_focus_areas_contains_quality(self):
        """DEFAULT_FOCUS_AREAS includes quality."""
        from aragora.prompts.code_review import DEFAULT_FOCUS_AREAS

        assert "quality" in DEFAULT_FOCUS_AREAS


class TestSecurityPrompt:
    """Tests for SECURITY_PROMPT constant."""

    def test_security_prompt_is_string(self):
        """SECURITY_PROMPT is a string."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert isinstance(SECURITY_PROMPT, str)

    def test_security_prompt_not_empty(self):
        """SECURITY_PROMPT is not empty."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert len(SECURITY_PROMPT.strip()) > 0

    def test_security_prompt_mentions_injection(self):
        """SECURITY_PROMPT mentions injection vulnerabilities."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "Injection" in SECURITY_PROMPT or "injection" in SECURITY_PROMPT.lower()

    def test_security_prompt_mentions_xss(self):
        """SECURITY_PROMPT mentions XSS."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "XSS" in SECURITY_PROMPT

    def test_security_prompt_mentions_csrf(self):
        """SECURITY_PROMPT mentions CSRF."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "CSRF" in SECURITY_PROMPT

    def test_security_prompt_mentions_secrets(self):
        """SECURITY_PROMPT mentions secrets/credentials."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "Secrets" in SECURITY_PROMPT or "credentials" in SECURITY_PROMPT.lower()

    def test_security_prompt_mentions_severity_levels(self):
        """SECURITY_PROMPT includes severity rating guidance."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "CRITICAL" in SECURITY_PROMPT
        assert "HIGH" in SECURITY_PROMPT
        assert "MEDIUM" in SECURITY_PROMPT
        assert "LOW" in SECURITY_PROMPT

    def test_security_prompt_mentions_ssrf(self):
        """SECURITY_PROMPT mentions SSRF."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "SSRF" in SECURITY_PROMPT

    def test_security_prompt_mentions_path_traversal(self):
        """SECURITY_PROMPT mentions path traversal."""
        from aragora.prompts.code_review import SECURITY_PROMPT

        assert "Path Traversal" in SECURITY_PROMPT or "path" in SECURITY_PROMPT.lower()


class TestPerformancePrompt:
    """Tests for PERFORMANCE_PROMPT constant."""

    def test_performance_prompt_is_string(self):
        """PERFORMANCE_PROMPT is a string."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert isinstance(PERFORMANCE_PROMPT, str)

    def test_performance_prompt_not_empty(self):
        """PERFORMANCE_PROMPT is not empty."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert len(PERFORMANCE_PROMPT.strip()) > 0

    def test_performance_prompt_mentions_n_plus_1(self):
        """PERFORMANCE_PROMPT mentions N+1 queries."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert "N+1" in PERFORMANCE_PROMPT

    def test_performance_prompt_mentions_complexity(self):
        """PERFORMANCE_PROMPT mentions algorithmic complexity."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert "Complexity" in PERFORMANCE_PROMPT or "complexity" in PERFORMANCE_PROMPT.lower()

    def test_performance_prompt_mentions_memory(self):
        """PERFORMANCE_PROMPT mentions memory issues."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert "Memory" in PERFORMANCE_PROMPT or "memory" in PERFORMANCE_PROMPT.lower()

    def test_performance_prompt_mentions_caching(self):
        """PERFORMANCE_PROMPT mentions caching."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert "Caching" in PERFORMANCE_PROMPT or "caching" in PERFORMANCE_PROMPT.lower()

    def test_performance_prompt_mentions_severity_levels(self):
        """PERFORMANCE_PROMPT includes severity rating guidance."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert "CRITICAL" in PERFORMANCE_PROMPT
        assert "HIGH" in PERFORMANCE_PROMPT
        assert "MEDIUM" in PERFORMANCE_PROMPT
        assert "LOW" in PERFORMANCE_PROMPT

    def test_performance_prompt_mentions_pagination(self):
        """PERFORMANCE_PROMPT mentions pagination."""
        from aragora.prompts.code_review import PERFORMANCE_PROMPT

        assert "Pagination" in PERFORMANCE_PROMPT or "pagination" in PERFORMANCE_PROMPT.lower()


class TestQualityPrompt:
    """Tests for QUALITY_PROMPT constant."""

    def test_quality_prompt_is_string(self):
        """QUALITY_PROMPT is a string."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert isinstance(QUALITY_PROMPT, str)

    def test_quality_prompt_not_empty(self):
        """QUALITY_PROMPT is not empty."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert len(QUALITY_PROMPT.strip()) > 0

    def test_quality_prompt_mentions_error_handling(self):
        """QUALITY_PROMPT mentions error handling."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert "Error Handling" in QUALITY_PROMPT or "error" in QUALITY_PROMPT.lower()

    def test_quality_prompt_mentions_edge_cases(self):
        """QUALITY_PROMPT mentions edge cases."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert "Edge Cases" in QUALITY_PROMPT or "edge" in QUALITY_PROMPT.lower()

    def test_quality_prompt_mentions_validation(self):
        """QUALITY_PROMPT mentions input validation."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert "Validation" in QUALITY_PROMPT or "validation" in QUALITY_PROMPT.lower()

    def test_quality_prompt_mentions_race_conditions(self):
        """QUALITY_PROMPT mentions race conditions."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert "Race Conditions" in QUALITY_PROMPT or "race" in QUALITY_PROMPT.lower()

    def test_quality_prompt_mentions_severity_levels(self):
        """QUALITY_PROMPT includes severity rating guidance."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert "CRITICAL" in QUALITY_PROMPT
        assert "HIGH" in QUALITY_PROMPT
        assert "MEDIUM" in QUALITY_PROMPT
        assert "LOW" in QUALITY_PROMPT

    def test_quality_prompt_mentions_dead_code(self):
        """QUALITY_PROMPT mentions dead code."""
        from aragora.prompts.code_review import QUALITY_PROMPT

        assert "Dead Code" in QUALITY_PROMPT or "dead code" in QUALITY_PROMPT.lower()


class TestGetFocusPrompts:
    """Tests for get_focus_prompts function."""

    def test_returns_string(self):
        """get_focus_prompts returns a string."""
        from aragora.prompts.code_review import get_focus_prompts

        result = get_focus_prompts()

        assert isinstance(result, str)

    def test_default_includes_all_areas(self):
        """Default call includes all focus areas."""
        from aragora.prompts.code_review import (
            PERFORMANCE_PROMPT,
            QUALITY_PROMPT,
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts()

        assert SECURITY_PROMPT in result
        assert PERFORMANCE_PROMPT in result
        assert QUALITY_PROMPT in result

    def test_security_only(self):
        """Can request only security focus."""
        from aragora.prompts.code_review import (
            PERFORMANCE_PROMPT,
            QUALITY_PROMPT,
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts(["security"])

        assert SECURITY_PROMPT in result
        assert PERFORMANCE_PROMPT not in result
        assert QUALITY_PROMPT not in result

    def test_performance_only(self):
        """Can request only performance focus."""
        from aragora.prompts.code_review import (
            PERFORMANCE_PROMPT,
            QUALITY_PROMPT,
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts(["performance"])

        assert PERFORMANCE_PROMPT in result
        assert SECURITY_PROMPT not in result
        assert QUALITY_PROMPT not in result

    def test_quality_only(self):
        """Can request only quality focus."""
        from aragora.prompts.code_review import (
            PERFORMANCE_PROMPT,
            QUALITY_PROMPT,
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts(["quality"])

        assert QUALITY_PROMPT in result
        assert SECURITY_PROMPT not in result
        assert PERFORMANCE_PROMPT not in result

    def test_multiple_areas(self):
        """Can request multiple specific areas."""
        from aragora.prompts.code_review import (
            PERFORMANCE_PROMPT,
            QUALITY_PROMPT,
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts(["security", "performance"])

        assert SECURITY_PROMPT in result
        assert PERFORMANCE_PROMPT in result
        assert QUALITY_PROMPT not in result

    def test_empty_list_uses_defaults(self):
        """Empty focus list uses defaults (falsy check)."""
        from aragora.prompts.code_review import (
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        # Empty list is falsy, so defaults are used
        result = get_focus_prompts([])

        # Should include default prompts since [] is falsy
        assert SECURITY_PROMPT in result

    def test_none_uses_defaults(self):
        """None focus areas uses defaults (all areas)."""
        from aragora.prompts.code_review import (
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts(None)

        assert SECURITY_PROMPT in result

    def test_invalid_area_ignored(self):
        """Invalid focus area names are ignored."""
        from aragora.prompts.code_review import get_focus_prompts

        result = get_focus_prompts(["invalid_area"])

        # Should return empty since no valid areas matched
        assert result == ""

    def test_mixed_valid_invalid_areas(self):
        """Mixed valid and invalid areas only include valid ones."""
        from aragora.prompts.code_review import (
            SECURITY_PROMPT,
            get_focus_prompts,
        )

        result = get_focus_prompts(["security", "invalid"])

        assert SECURITY_PROMPT in result

    def test_prompts_joined_with_double_newline(self):
        """Multiple prompts are joined with double newlines."""
        from aragora.prompts.code_review import get_focus_prompts

        result = get_focus_prompts(["security", "performance"])

        # Should have double newline separator
        assert "\n\n" in result


class TestBuildReviewPrompt:
    """Tests for build_review_prompt function."""

    def test_returns_string(self):
        """build_review_prompt returns a string."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="+ added line")

        assert isinstance(result, str)

    def test_includes_diff_content(self):
        """Diff content is included in the prompt."""
        from aragora.prompts.code_review import build_review_prompt

        test_diff = "+def new_function():\n+    pass"
        result = build_review_prompt(diff=test_diff)

        assert test_diff in result

    def test_includes_focus_prompts(self):
        """Focus prompts are included by default."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test")

        # Should include security-related content from SECURITY_PROMPT
        assert "Security" in result or "security" in result.lower()

    def test_specific_focus_areas(self):
        """Can specify focus areas."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test", focus_areas=["security"])

        assert "Security" in result
        # Performance section header should NOT be present
        assert "Performance Review Focus" not in result

    def test_includes_response_format(self):
        """Response format instructions are included."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test")

        assert "Response Format" in result
        assert "Severity" in result

    def test_includes_reviewer_role(self):
        """Reviewer role description is included."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test")

        assert "expert code reviewer" in result.lower()

    def test_includes_additional_context(self):
        """Additional context is included when provided."""
        from aragora.prompts.code_review import build_review_prompt

        context = "This is a payment processing module"
        result = build_review_prompt(diff="test", additional_context=context)

        assert context in result
        assert "Additional Context" in result

    def test_no_context_section_when_none(self):
        """No context section when additional_context is None."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test", additional_context=None)

        # Should not have empty "Additional Context" section
        assert "Additional Context\n\n\n" not in result

    def test_diff_in_code_block(self):
        """Diff is wrapped in a code block."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test diff")

        assert "```diff" in result
        assert "```" in result

    def test_truncates_large_diff(self):
        """Large diffs are truncated."""
        from aragora.prompts.code_review import MAX_DIFF_SIZE, build_review_prompt

        large_diff = "x" * (MAX_DIFF_SIZE + 1000)
        result = build_review_prompt(diff=large_diff)

        # Truncation marker should be present
        assert "truncated" in result.lower()
        # Full original diff should NOT be present
        assert large_diff not in result

    def test_truncation_at_max_size(self):
        """Truncation happens exactly at MAX_DIFF_SIZE."""
        from aragora.prompts.code_review import MAX_DIFF_SIZE, build_review_prompt

        # Diff exactly at limit should NOT be truncated
        exact_diff = "x" * MAX_DIFF_SIZE
        result_exact = build_review_prompt(diff=exact_diff)
        assert "truncated" not in result_exact.lower()

        # Diff over limit should be truncated
        over_diff = "x" * (MAX_DIFF_SIZE + 1)
        result_over = build_review_prompt(diff=over_diff)
        assert "truncated" in result_over.lower()

    def test_small_diff_not_truncated(self):
        """Small diffs are not truncated."""
        from aragora.prompts.code_review import build_review_prompt

        small_diff = "short diff"
        result = build_review_prompt(diff=small_diff)

        assert small_diff in result
        assert "truncated" not in result.lower()

    def test_includes_guidelines(self):
        """Important guidelines are included."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test")

        assert "Important Guidelines" in result
        assert "REAL issues" in result

    def test_includes_no_issues_guidance(self):
        """Guidance for no-issues case is included."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="test")

        assert "No issues found" in result

    def test_empty_diff_handled(self):
        """Empty diff is handled gracefully."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(diff="")

        assert isinstance(result, str)
        assert "```diff" in result

    def test_special_characters_in_diff(self):
        """Special characters in diff are preserved."""
        from aragora.prompts.code_review import build_review_prompt

        special_diff = "+user_input = request.args.get('id')\n+query = f\"SELECT * FROM users WHERE id={user_input}\""
        result = build_review_prompt(diff=special_diff)

        assert "user_input" in result
        assert "SELECT" in result

    def test_unicode_in_diff(self):
        """Unicode in diff is preserved."""
        from aragora.prompts.code_review import build_review_prompt

        unicode_diff = "+message = 'Hello, 世界!'\n+emoji = '🎉'"
        result = build_review_prompt(diff=unicode_diff)

        assert "世界" in result
        # Emoji may or may not be preserved depending on encoding


class TestRolePrompts:
    """Tests for role-specific prompts."""

    def test_security_reviewer_role_exists(self):
        """SECURITY_REVIEWER_ROLE constant exists."""
        from aragora.prompts.code_review import SECURITY_REVIEWER_ROLE

        assert isinstance(SECURITY_REVIEWER_ROLE, str)
        assert len(SECURITY_REVIEWER_ROLE) > 0

    def test_performance_reviewer_role_exists(self):
        """PERFORMANCE_REVIEWER_ROLE constant exists."""
        from aragora.prompts.code_review import PERFORMANCE_REVIEWER_ROLE

        assert isinstance(PERFORMANCE_REVIEWER_ROLE, str)
        assert len(PERFORMANCE_REVIEWER_ROLE) > 0

    def test_quality_reviewer_role_exists(self):
        """QUALITY_REVIEWER_ROLE constant exists."""
        from aragora.prompts.code_review import QUALITY_REVIEWER_ROLE

        assert isinstance(QUALITY_REVIEWER_ROLE, str)
        assert len(QUALITY_REVIEWER_ROLE) > 0

    def test_security_reviewer_mentions_owasp(self):
        """Security reviewer role mentions OWASP."""
        from aragora.prompts.code_review import SECURITY_REVIEWER_ROLE

        assert "OWASP" in SECURITY_REVIEWER_ROLE

    def test_performance_reviewer_mentions_optimization(self):
        """Performance reviewer role mentions optimization."""
        from aragora.prompts.code_review import PERFORMANCE_REVIEWER_ROLE

        assert "optimization" in PERFORMANCE_REVIEWER_ROLE.lower()

    def test_quality_reviewer_mentions_design_patterns(self):
        """Quality reviewer role mentions design patterns."""
        from aragora.prompts.code_review import QUALITY_REVIEWER_ROLE

        assert "design patterns" in QUALITY_REVIEWER_ROLE.lower()


class TestGetRolePrompt:
    """Tests for get_role_prompt function."""

    def test_get_role_prompt_exists(self):
        """get_role_prompt function exists."""
        from aragora.prompts.code_review import get_role_prompt

        assert callable(get_role_prompt)

    def test_security_reviewer_role(self):
        """Can get security reviewer role prompt."""
        from aragora.prompts.code_review import SECURITY_REVIEWER_ROLE, get_role_prompt

        result = get_role_prompt("security_reviewer")

        assert result == SECURITY_REVIEWER_ROLE

    def test_performance_reviewer_role(self):
        """Can get performance reviewer role prompt."""
        from aragora.prompts.code_review import PERFORMANCE_REVIEWER_ROLE, get_role_prompt

        result = get_role_prompt("performance_reviewer")

        assert result == PERFORMANCE_REVIEWER_ROLE

    def test_quality_reviewer_role(self):
        """Can get quality reviewer role prompt."""
        from aragora.prompts.code_review import QUALITY_REVIEWER_ROLE, get_role_prompt

        result = get_role_prompt("quality_reviewer")

        assert result == QUALITY_REVIEWER_ROLE

    def test_invalid_role_returns_empty(self):
        """Invalid role name returns empty string."""
        from aragora.prompts.code_review import get_role_prompt

        result = get_role_prompt("invalid_role")

        assert result == ""

    def test_empty_role_returns_empty(self):
        """Empty role name returns empty string."""
        from aragora.prompts.code_review import get_role_prompt

        result = get_role_prompt("")

        assert result == ""

    def test_case_sensitive_role_lookup(self):
        """Role lookup is case sensitive."""
        from aragora.prompts.code_review import get_role_prompt

        # Correct case
        result_correct = get_role_prompt("security_reviewer")
        assert result_correct != ""

        # Wrong case
        result_wrong = get_role_prompt("SECURITY_REVIEWER")
        assert result_wrong == ""


class TestIntegration:
    """Integration tests for prompt module."""

    def test_full_review_prompt_workflow(self):
        """Complete workflow: build prompt with all components."""
        from aragora.prompts.code_review import (
            build_review_prompt,
            get_focus_prompts,
            get_role_prompt,
        )

        # Simulate a code review setup
        diff = """
+def process_payment(user_input):
+    # Potential SQL injection!
+    query = f"SELECT * FROM payments WHERE id={user_input}"
+    return db.execute(query)
"""
        context = "This is a payment processing endpoint"

        # Get role prompt for the reviewing agent
        role = get_role_prompt("security_reviewer")
        assert "OWASP" in role

        # Get focus prompts for security
        focus = get_focus_prompts(["security"])
        assert "Injection" in focus

        # Build the complete review prompt
        prompt = build_review_prompt(
            diff=diff,
            focus_areas=["security"],
            additional_context=context,
        )

        # Verify all components are present
        assert "process_payment" in prompt
        assert "SQL injection" in prompt.lower() or "SELECT" in prompt
        assert context in prompt
        assert "Security" in prompt

    def test_multiple_focus_areas_combined(self):
        """Multiple focus areas combine properly."""
        from aragora.prompts.code_review import build_review_prompt

        result = build_review_prompt(
            diff="test",
            focus_areas=["security", "performance", "quality"],
        )

        # All three sections should be present
        assert "Security Review Focus" in result
        assert "Performance Review Focus" in result
        assert "Quality Review Focus" in result

    def test_all_role_prompts_mention_senior(self):
        """All role prompts describe senior-level reviewers."""
        from aragora.prompts.code_review import (
            PERFORMANCE_REVIEWER_ROLE,
            QUALITY_REVIEWER_ROLE,
            SECURITY_REVIEWER_ROLE,
        )

        for role in [SECURITY_REVIEWER_ROLE, PERFORMANCE_REVIEWER_ROLE, QUALITY_REVIEWER_ROLE]:
            assert "senior" in role.lower()

    def test_prompts_are_well_formatted(self):
        """Prompts have proper markdown formatting."""
        from aragora.prompts.code_review import (
            PERFORMANCE_PROMPT,
            QUALITY_PROMPT,
            SECURITY_PROMPT,
        )

        for prompt in [SECURITY_PROMPT, PERFORMANCE_PROMPT, QUALITY_PROMPT]:
            # Should have markdown headers
            assert "**" in prompt  # Bold text
            # Should have bullet points
            assert "- " in prompt  # List items
