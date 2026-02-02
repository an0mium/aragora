"""
Tests for orchestrator domain classification module.

Tests cover:
- compute_domain_from_task function
- All domain classifications (security, performance, testing, etc.)
- LRU caching behavior
- Backward compatibility alias (_compute_domain_from_task)
- Edge cases (empty strings, mixed keywords)
"""

from __future__ import annotations

import pytest

from aragora.debate.orchestrator_domains import (
    _compute_domain_from_task,
    compute_domain_from_task,
)


# =============================================================================
# Test Security Domain
# =============================================================================


class TestSecurityDomain:
    """Tests for security domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            "review the security of this api",
            "check for security vulnerabilities",
            "implement security best practices",
            "analyze the SECURITY implications",
        ],
    )
    def test_security_keyword(self, task: str):
        """Test 'security' keyword triggers security domain."""
        assert compute_domain_from_task(task.lower()) == "security"

    @pytest.mark.parametrize(
        "task",
        [
            "prevent hack attempts",
            "the system was hacked",
            "implement anti-hack measures",
        ],
    )
    def test_hack_keyword(self, task: str):
        """Test 'hack' keyword triggers security domain."""
        assert compute_domain_from_task(task.lower()) == "security"

    @pytest.mark.parametrize(
        "task",
        [
            "analyze the vulnerability in user input",
            "vulnerability assessment report",
            "patch known vulnerability cve-2024-1234",
        ],
    )
    def test_vulnerability_keyword(self, task: str):
        """Test 'vulnerability' keyword triggers security domain."""
        assert compute_domain_from_task(task.lower()) == "security"

    @pytest.mark.parametrize(
        "task",
        [
            "implement auth flow",
            "fix authentication issues",
            "oauth authentication integration",
            "add basic auth support",
        ],
    )
    def test_auth_keyword(self, task: str):
        """Test 'auth' keyword triggers security domain."""
        assert compute_domain_from_task(task.lower()) == "security"

    @pytest.mark.parametrize(
        "task",
        [
            "add encryption to data at rest",
            "implement end-to-end encrypt",
            "encrypt sensitive fields",
        ],
    )
    def test_encrypt_keyword(self, task: str):
        """Test 'encrypt' keyword triggers security domain."""
        assert compute_domain_from_task(task.lower()) == "security"


# =============================================================================
# Test Performance Domain
# =============================================================================


class TestPerformanceDomain:
    """Tests for performance domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            "analyze performance bottlenecks",
            "improve system performance",
            "performance testing results",
        ],
    )
    def test_performance_keyword(self, task: str):
        """Test 'performance' keyword triggers performance domain."""
        assert compute_domain_from_task(task.lower()) == "performance"

    @pytest.mark.parametrize(
        "task",
        [
            "improve query speed",
            "speed up page load",
            "measure response speed",
        ],
    )
    def test_speed_keyword(self, task: str):
        """Test 'speed' keyword triggers performance domain."""
        assert compute_domain_from_task(task.lower()) == "performance"

    @pytest.mark.parametrize(
        "task",
        [
            "optimize the code",
            "optimize memory usage",
            "optimize rendering",
        ],
    )
    def test_optimize_keyword(self, task: str):
        """Test 'optimize' keyword triggers performance domain."""
        assert compute_domain_from_task(task.lower()) == "performance"

    @pytest.mark.parametrize(
        "task",
        [
            "implement redis cache",
            "add cache layer",
            "cache invalidation strategy",
        ],
    )
    def test_cache_keyword(self, task: str):
        """Test 'cache' keyword triggers performance domain."""
        assert compute_domain_from_task(task.lower()) == "performance"

    @pytest.mark.parametrize(
        "task",
        [
            "reduce api latency",
            "high latency issues",
            "latency optimization",
        ],
    )
    def test_latency_keyword(self, task: str):
        """Test 'latency' keyword triggers performance domain."""
        assert compute_domain_from_task(task.lower()) == "performance"


# =============================================================================
# Test Testing Domain
# =============================================================================


class TestTestingDomain:
    """Tests for testing domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            "write a test for this function",
            "add unit test coverage",
            "test the edge cases",
        ],
    )
    def test_test_keyword(self, task: str):
        """Test 'test' keyword triggers testing domain."""
        assert compute_domain_from_task(task.lower()) == "testing"

    @pytest.mark.parametrize(
        "task",
        [
            "improve testing strategy",
            "integration testing setup",
            "end-to-end testing framework",
        ],
    )
    def test_testing_keyword(self, task: str):
        """Test 'testing' keyword triggers testing domain."""
        assert compute_domain_from_task(task.lower()) == "testing"

    @pytest.mark.parametrize(
        "task",
        [
            "increase code coverage",
            "coverage report analysis",
            "100% coverage goal",
        ],
    )
    def test_coverage_keyword(self, task: str):
        """Test 'coverage' keyword triggers testing domain."""
        assert compute_domain_from_task(task.lower()) == "testing"

    @pytest.mark.parametrize(
        "task",
        [
            "fix regression in payment flow",
            "regression tests failing",
            "prevent regression bugs",
        ],
    )
    def test_regression_keyword(self, task: str):
        """Test 'regression' keyword triggers testing domain."""
        assert compute_domain_from_task(task.lower()) == "testing"


# =============================================================================
# Test Architecture Domain
# =============================================================================


class TestArchitectureDomain:
    """Tests for architecture domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            "design a new microservice",
            "system design review",
            "design the api interface",
        ],
    )
    def test_design_keyword(self, task: str):
        """Test 'design' keyword triggers architecture domain."""
        assert compute_domain_from_task(task.lower()) == "architecture"

    @pytest.mark.parametrize(
        "task",
        [
            "review system architecture",
            "microservices architecture",
            "architecture decision record",
        ],
    )
    def test_architecture_keyword(self, task: str):
        """Test 'architecture' keyword triggers architecture domain."""
        assert compute_domain_from_task(task.lower()) == "architecture"

    @pytest.mark.parametrize(
        "task",
        [
            "implement factory pattern",
            "use singleton pattern",
            "observer pattern for events",
        ],
    )
    def test_pattern_keyword(self, task: str):
        """Test 'pattern' keyword triggers architecture domain."""
        assert compute_domain_from_task(task.lower()) == "architecture"

    @pytest.mark.parametrize(
        "task",
        [
            "refactor code structure",
            "improve project structure",
            "folder structure organization",
        ],
    )
    def test_structure_keyword(self, task: str):
        """Test 'structure' keyword triggers architecture domain."""
        assert compute_domain_from_task(task.lower()) == "architecture"


# =============================================================================
# Test Debugging Domain
# =============================================================================


class TestDebuggingDomain:
    """Tests for debugging domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            "fix the bug in login",
            "investigate bug report",
            "reproduce this bug",
        ],
    )
    def test_bug_keyword(self, task: str):
        """Test 'bug' keyword triggers debugging domain."""
        assert compute_domain_from_task(task.lower()) == "debugging"

    @pytest.mark.parametrize(
        "task",
        [
            "fix error handling",
            "investigate error message",
            "500 error in production",
        ],
    )
    def test_error_keyword(self, task: str):
        """Test 'error' keyword triggers debugging domain."""
        assert compute_domain_from_task(task.lower()) == "debugging"

    @pytest.mark.parametrize(
        "task",
        [
            "fix the broken feature",
            "apply hotfix for issue",
            "fix asap this blocker",
        ],
    )
    def test_fix_keyword(self, task: str):
        """Test 'fix' keyword triggers debugging domain."""
        assert compute_domain_from_task(task.lower()) == "debugging"

    @pytest.mark.parametrize(
        "task",
        [
            "investigate app crash",
            "server crash analysis",
            "crash dump review",
        ],
    )
    def test_crash_keyword(self, task: str):
        """Test 'crash' keyword triggers debugging domain."""
        assert compute_domain_from_task(task.lower()) == "debugging"

    @pytest.mark.parametrize(
        "task",
        [
            "handle null pointer exception",
            "fix timeout exception",
            "exception handling improvement",
        ],
    )
    def test_exception_keyword(self, task: str):
        """Test 'exception' keyword triggers debugging domain."""
        assert compute_domain_from_task(task.lower()) == "debugging"


# =============================================================================
# Test API Domain
# =============================================================================


class TestAPIDomain:
    """Tests for API domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            # Note: "design the new api" matches "design" (architecture) first
            "api versioning strategy",
            "document the api",
            "api rate limiting",
        ],
    )
    def test_api_keyword(self, task: str):
        """Test 'api' keyword triggers api domain."""
        assert compute_domain_from_task(task.lower()) == "api"

    @pytest.mark.parametrize(
        "task",
        [
            "add new endpoint for users",
            "endpoint rate limiting",
            "secure the endpoint",
        ],
    )
    def test_endpoint_keyword(self, task: str):
        """Test 'endpoint' keyword triggers api domain."""
        assert compute_domain_from_task(task.lower()) == "api"

    @pytest.mark.parametrize(
        "task",
        [
            "implement rest api",
            "rest best practices",
            # Note: "restful resource design" matches "design" (architecture) first
            "restful interface",
        ],
    )
    def test_rest_keyword(self, task: str):
        """Test 'rest' keyword triggers api domain."""
        assert compute_domain_from_task(task.lower()) == "api"

    @pytest.mark.parametrize(
        "task",
        [
            "add graphql resolver",
            # Note: "graphql schema design" matches "design" (architecture) first
            "migrate to graphql",
            "graphql query optimization",
        ],
    )
    def test_graphql_keyword(self, task: str):
        """Test 'graphql' keyword triggers api domain."""
        assert compute_domain_from_task(task.lower()) == "api"


# =============================================================================
# Test Database Domain
# =============================================================================


class TestDatabaseDomain:
    """Tests for database domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            # Note: "optimize database queries" matches "optimize" (performance) first
            "database migration needed",
            "database connection pool",
            "database indexing",
        ],
    )
    def test_database_keyword(self, task: str):
        """Test 'database' keyword triggers database domain."""
        assert compute_domain_from_task(task.lower()) == "database"

    @pytest.mark.parametrize(
        "task",
        [
            "write sql for report",
            "sql injection prevention",
            # Note: "optimize sql query" matches "optimize" (performance) first
            "sql join issue",
        ],
    )
    def test_sql_keyword(self, task: str):
        """Test 'sql' keyword triggers database domain."""
        assert compute_domain_from_task(task.lower()) == "database"

    @pytest.mark.parametrize(
        "task",
        [
            # Note: "optimize the query plan" matches "optimize" (performance) first
            "slow query analysis",
            # Note: "query caching strategy" matches "cache" (performance) first
            "query execution plan",
            "complex query issue",
        ],
    )
    def test_query_keyword(self, task: str):
        """Test 'query' keyword triggers database domain."""
        assert compute_domain_from_task(task.lower()) == "database"

    @pytest.mark.parametrize(
        "task",
        [
            "update database schema",
            "schema migration tool",
            "schema versioning",
        ],
    )
    def test_schema_keyword(self, task: str):
        """Test 'schema' keyword triggers database domain."""
        assert compute_domain_from_task(task.lower()) == "database"


# =============================================================================
# Test Frontend Domain
# =============================================================================


class TestFrontendDomain:
    """Tests for frontend domain classification."""

    @pytest.mark.parametrize(
        "task",
        [
            "improve ui responsiveness",
            # Note: "ui design review" matches "design" (architecture) first
            "ui component library",
            "ui update needed",
        ],
    )
    def test_ui_keyword(self, task: str):
        """Test 'ui' keyword triggers frontend domain."""
        assert compute_domain_from_task(task.lower()) == "frontend"

    @pytest.mark.parametrize(
        "task",
        [
            # Note: "frontend performance" matches "performance" first
            # Note: "frontend testing setup" matches "testing" first
            "frontend refactoring",
            "frontend build",
            "frontend rendering",
        ],
    )
    def test_frontend_keyword(self, task: str):
        """Test 'frontend' keyword triggers frontend domain."""
        assert compute_domain_from_task(task.lower()) == "frontend"

    @pytest.mark.parametrize(
        "task",
        [
            "migrate react components",
            "react hooks best practices",
            "react state management",
        ],
    )
    def test_react_keyword(self, task: str):
        """Test 'react' keyword triggers frontend domain."""
        assert compute_domain_from_task(task.lower()) == "frontend"

    @pytest.mark.parametrize(
        "task",
        [
            # Note: "fix css layout issue" matches "fix" (debugging) first
            # Note: "css animation performance" matches "performance" first
            "css grid implementation",
            "css styling changes",
            "css selector issues",
        ],
    )
    def test_css_keyword(self, task: str):
        """Test 'css' keyword triggers frontend domain."""
        assert compute_domain_from_task(task.lower()) == "frontend"

    @pytest.mark.parametrize(
        "task",
        [
            # Note: "fix layout on mobile" matches "fix" (debugging) first
            # Note: "responsive layout design" matches "design" (architecture) first
            "layout grid system",
            "layout adjustments",
            "update the layout",
        ],
    )
    def test_layout_keyword(self, task: str):
        """Test 'layout' keyword triggers frontend domain."""
        assert compute_domain_from_task(task.lower()) == "frontend"


# =============================================================================
# Test General Domain (Fallback)
# =============================================================================


class TestGeneralDomain:
    """Tests for general domain classification (fallback)."""

    @pytest.mark.parametrize(
        "task",
        [
            "review the code",
            "implement new feature",
            "refactor the module",
            "update documentation",
            "prepare for release",
            "set up development environment",
        ],
    )
    def test_general_fallback(self, task: str):
        """Test that non-specific tasks fall back to general domain."""
        assert compute_domain_from_task(task.lower()) == "general"

    def test_empty_string(self):
        """Test empty string returns general domain."""
        assert compute_domain_from_task("") == "general"

    def test_whitespace_only(self):
        """Test whitespace-only string returns general domain."""
        assert compute_domain_from_task("   ") == "general"


# =============================================================================
# Test Domain Priority (First Match Wins)
# =============================================================================


class TestDomainPriority:
    """Tests for domain priority when multiple keywords present."""

    def test_security_before_performance(self):
        """Security keywords should match before performance."""
        # 'security' appears before 'performance' in function
        task = "security audit for performance optimization"
        assert compute_domain_from_task(task.lower()) == "security"

    def test_security_before_testing(self):
        """Security keywords should match before testing."""
        task = "security vulnerability test"
        assert compute_domain_from_task(task.lower()) == "security"

    def test_performance_before_testing(self):
        """Performance keywords should match before testing."""
        task = "performance regression testing"
        # 'performance' is checked before 'testing'
        assert compute_domain_from_task(task.lower()) == "performance"

    def test_architecture_before_api(self):
        """Architecture keywords should match before API."""
        task = "design the api architecture"
        # 'design' is checked in architecture before 'api'
        assert compute_domain_from_task(task.lower()) == "architecture"


# =============================================================================
# Test LRU Cache Behavior
# =============================================================================


class TestLRUCacheBehavior:
    """Tests for LRU caching of domain computation."""

    def test_cache_returns_same_result(self):
        """Test that cached results are consistent."""
        task = "implement security feature"
        result1 = compute_domain_from_task(task)
        result2 = compute_domain_from_task(task)
        assert result1 == result2 == "security"

    def test_cache_info_accessible(self):
        """Test that cache_info is accessible for monitoring."""
        # Clear cache first
        compute_domain_from_task.cache_clear()

        # Make some calls
        compute_domain_from_task("security check")
        compute_domain_from_task("performance test")
        compute_domain_from_task("security check")  # Cache hit

        info = compute_domain_from_task.cache_info()
        assert info.hits >= 1
        assert info.misses >= 2
        assert info.currsize >= 2

    def test_cache_clear_works(self):
        """Test that cache_clear clears the cache."""
        compute_domain_from_task("some task")
        compute_domain_from_task.cache_clear()
        info = compute_domain_from_task.cache_info()
        assert info.currsize == 0


# =============================================================================
# Test Backward Compatibility Alias
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility alias."""

    def test_alias_exists(self):
        """Test that _compute_domain_from_task alias exists."""
        assert _compute_domain_from_task is not None

    def test_alias_is_same_function(self):
        """Test that alias references the same function."""
        assert _compute_domain_from_task is compute_domain_from_task

    def test_alias_returns_same_result(self):
        """Test that alias returns same result as main function."""
        task = "implement authentication"
        assert _compute_domain_from_task(task) == compute_domain_from_task(task)


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from aragora.debate import orchestrator_domains

        expected = {"compute_domain_from_task", "_compute_domain_from_task"}
        assert set(orchestrator_domains.__all__) == expected

    def test_function_is_callable(self):
        """Test exported function is callable."""
        assert callable(compute_domain_from_task)

    def test_function_has_cache_methods(self):
        """Test function has LRU cache methods."""
        assert hasattr(compute_domain_from_task, "cache_info")
        assert hasattr(compute_domain_from_task, "cache_clear")


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in domain classification."""

    def test_partial_keyword_match(self):
        """Test partial keyword matching (e.g., 'tests' contains 'test')."""
        # 'tests' contains 'test'
        assert compute_domain_from_task("run all tests") == "testing"

    def test_keyword_as_substring(self):
        """Test keyword as substring of another word."""
        # 'authenticate' contains 'auth'
        assert compute_domain_from_task("authenticate users") == "security"

    def test_case_sensitivity_warning(self):
        """Test that uppercase input may not match (function expects lowercase)."""
        # The function is designed for lowercase input
        # Uppercase should be lowercased before calling
        upper_task = "SECURITY CHECK"
        lower_task = upper_task.lower()
        # Uppercase will likely return 'general' since no match
        # But lowercase should return 'security'
        assert compute_domain_from_task(lower_task) == "security"

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        task = "security vulnerabilite"
        assert compute_domain_from_task(task) == "security"

    def test_special_characters(self):
        """Test handling of special characters in task."""
        task = "fix bug: api endpoint returns 500"
        assert compute_domain_from_task(task) == "debugging"

    def test_very_long_task(self):
        """Test handling of very long task descriptions."""
        long_prefix = "a" * 1000
        task = f"{long_prefix} security audit needed"
        assert compute_domain_from_task(task) == "security"

    def test_multiple_keywords_same_domain(self):
        """Test multiple keywords from same domain."""
        task = "optimize speed and cache performance latency"
        assert compute_domain_from_task(task) == "performance"
