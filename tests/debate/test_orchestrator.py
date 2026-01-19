"""
Tests for the debate orchestrator module.

Tests cover:
- _compute_domain_from_task utility function
"""

from __future__ import annotations

import pytest

from aragora.debate.orchestrator import _compute_domain_from_task


class TestComputeDomainFromTask:
    """Tests for _compute_domain_from_task utility function."""

    def test_security_domain(self):
        """Test security domain detection."""
        assert _compute_domain_from_task("implement authentication system") == "security"
        assert _compute_domain_from_task("fix the security vulnerability") == "security"
        assert _compute_domain_from_task("add encryption to data storage") == "security"
        assert _compute_domain_from_task("prevent hack attacks") == "security"
        assert _compute_domain_from_task("improve auth flow") == "security"

    def test_performance_domain(self):
        """Test performance domain detection."""
        # Note: Order of checks matters - performance keywords are checked early
        assert _compute_domain_from_task("improve the speed of the application") == "performance"
        assert _compute_domain_from_task("implement cache strategy") == "performance"
        assert _compute_domain_from_task("reduce latency") == "performance"
        assert _compute_domain_from_task("optimize the code") == "performance"

    def test_testing_domain(self):
        """Test testing domain detection."""
        assert _compute_domain_from_task("write unit tests for the module") == "testing"
        assert _compute_domain_from_task("improve test coverage") == "testing"
        assert _compute_domain_from_task("fix regression tests") == "testing"

    def test_architecture_domain(self):
        """Test architecture domain detection."""
        assert _compute_domain_from_task("design the system architecture") == "architecture"
        assert _compute_domain_from_task("implement factory pattern") == "architecture"
        assert _compute_domain_from_task("restructure the codebase") == "architecture"

    def test_debugging_domain(self):
        """Test debugging domain detection."""
        assert _compute_domain_from_task("fix the login bug") == "debugging"
        assert _compute_domain_from_task("resolve null pointer error") == "debugging"
        assert _compute_domain_from_task("debug the crash on startup") == "debugging"
        assert _compute_domain_from_task("handle runtime exception") == "debugging"

    def test_api_domain(self):
        """Test API domain detection."""
        assert _compute_domain_from_task("create new endpoint for users") == "api"
        assert _compute_domain_from_task("implement rest service") == "api"
        # Note: "graphql" is API but comes after other checks
        assert _compute_domain_from_task("graphql mutations") == "api"

    def test_database_domain(self):
        """Test database domain detection."""
        assert _compute_domain_from_task("create new sql table") == "database"
        assert _compute_domain_from_task("write database query") == "database"
        # Note: "schema" could be API (graphql) but database is checked after API

    def test_frontend_domain(self):
        """Test frontend domain detection."""
        assert _compute_domain_from_task("update the ui component") == "frontend"
        assert _compute_domain_from_task("improve layout styling") == "frontend"
        assert _compute_domain_from_task("style with css") == "frontend"
        assert _compute_domain_from_task("react component") == "frontend"
        # Note: "fix layout" triggers "debugging" due to "fix"

    def test_general_domain(self):
        """Test general domain fallback."""
        assert _compute_domain_from_task("do something random") == "general"
        assert _compute_domain_from_task("process the data") == "general"
        assert _compute_domain_from_task("implement feature x") == "general"

    def test_first_match_wins(self):
        """Test that the first matching domain wins (order matters)."""
        # "performance" is checked before "database" so "optimize" wins
        assert _compute_domain_from_task("optimize database queries") == "performance"
        # "security" is checked before "testing" so "auth" wins
        assert _compute_domain_from_task("test auth flow") == "security"

    def test_case_sensitivity(self):
        """Test that function expects lowercase input."""
        # The function expects lowercase input (as documented)
        # Mixed case might not match correctly
        assert _compute_domain_from_task("security") == "security"
        assert _compute_domain_from_task("performance") == "performance"

    def test_partial_word_matches(self):
        """Test that word boundaries are not enforced (substring matching)."""
        # "auth" matches in "authentication"
        assert _compute_domain_from_task("authentication") == "security"
        # "test" matches in "testing"
        assert _compute_domain_from_task("testing") == "testing"
        # "cache" is an exact keyword, "caching" doesn't match
        assert _compute_domain_from_task("cache layer") == "performance"
        # "encrypt" matches in "encryption"
        assert _compute_domain_from_task("encryption") == "security"

    def test_multiple_keywords_first_wins(self):
        """Test that with multiple keywords, first domain check wins."""
        # This has both "speed" (performance) and "bug" (debugging)
        # Performance is checked first, so it wins
        assert _compute_domain_from_task("speed up and fix the bug") == "performance"

        # This has both "security" and "test"
        # Security is checked first
        assert _compute_domain_from_task("security test suite") == "security"

    def test_empty_string(self):
        """Test empty string returns general."""
        assert _compute_domain_from_task("") == "general"

    def test_no_keywords(self):
        """Test string with no domain keywords returns general."""
        assert _compute_domain_from_task("hello world") == "general"
        assert _compute_domain_from_task("please help me") == "general"
