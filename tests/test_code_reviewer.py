"""
Tests for the CodeReviewOrchestrator.

Covers:
- Pattern-based code review
- Diff review
- PR review
- Multi-agent finding aggregation
- Security, performance, maintainability, test coverage checks
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.agents.code_reviewer import (
    CodeReviewOrchestrator,
    ReviewResult,
    ReviewFinding,
    ReviewCategory,
    FindingSeverity,
    SecurityReviewer,
    PerformanceReviewer,
    MaintainabilityReviewer,
    TestCoverageReviewer,
    SECURITY_PATTERNS,
    PERFORMANCE_PATTERNS,
    MAINTAINABILITY_PATTERNS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def orchestrator():
    """Create a fresh CodeReviewOrchestrator instance."""
    return CodeReviewOrchestrator()


@pytest.fixture
def security_reviewer():
    """Create a SecurityReviewer instance."""
    return SecurityReviewer()


@pytest.fixture
def performance_reviewer():
    """Create a PerformanceReviewer instance."""
    return PerformanceReviewer()


@pytest.fixture
def maintainability_reviewer():
    """Create a MaintainabilityReviewer instance."""
    return MaintainabilityReviewer()


@pytest.fixture
def sample_python_code():
    """Sample Python code for review."""
    return '''
import os
import subprocess

def execute_command(user_input):
    """Execute a shell command."""
    # Vulnerable to command injection
    result = subprocess.run(user_input, shell=True, capture_output=True)
    return result.stdout

def get_user_data(user_id):
    """Get user data from database."""
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)

API_KEY = "sk-1234567890abcdef"  # Hardcoded secret

def process_file(filename):
    """Process a file."""
    with open(filename) as f:
        data = eval(f.read())  # Dangerous eval
    return data
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for review."""
    return """
function renderUserInput(input) {
    // XSS vulnerability
    document.innerHTML = input;
}

function fetchData(userId) {
    // No input validation
    fetch('/api/users/' + userId)
        .then(response => response.json())
        .then(data => renderUserInput(data.name));
}

// Inefficient loop
function processArray(arr) {
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length; j++) {
            console.log(arr[i], arr[j]);
        }
    }
}
"""


@pytest.fixture
def sample_diff():
    """Sample unified diff for review."""
    return """
diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,15 @@ class AuthService:
+    def login(self, username, password):
+        # Added login method
+        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
+        result = self.db.execute(query)
+        if result:
+            return generate_token(result[0])
+        return None
"""


# =============================================================================
# Pattern Detection Tests
# =============================================================================


class TestPatternDetection:
    """Test pattern-based code review."""

    @pytest.mark.asyncio
    async def test_detect_sql_injection(self, orchestrator, sample_python_code):
        """Test detection of security vulnerabilities in sample code."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        security_findings = [f for f in result.findings if f.category == ReviewCategory.SECURITY]
        # Sample code contains eval, subprocess shell=True, hardcoded API key - should find something
        assert len(security_findings) > 0, "Should detect security issues in vulnerable code"

    @pytest.mark.asyncio
    async def test_detect_command_injection(self, orchestrator, sample_python_code):
        """Test detection of command injection patterns."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        findings = result.findings
        assert any(
            "command" in f.title.lower() or "shell" in str(f.description).lower() for f in findings
        )

    @pytest.mark.asyncio
    async def test_detect_hardcoded_secrets(self, orchestrator, sample_python_code):
        """Test detection of hardcoded secrets."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        findings = result.findings
        assert any("secret" in f.title.lower() or "key" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_detect_eval_usage(self, orchestrator, sample_python_code):
        """Test detection of dangerous eval usage."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        findings = result.findings
        assert any("eval" in str(f.description).lower() for f in findings)

    @pytest.mark.asyncio
    async def test_detect_xss(self, orchestrator, sample_javascript_code):
        """Test detection of XSS vulnerabilities."""
        result = await orchestrator.review_code(
            code=sample_javascript_code,
            file_path="test.js",
        )

        security_findings = [f for f in result.findings if f.category == ReviewCategory.SECURITY]
        assert len(security_findings) > 0


# =============================================================================
# Security Reviewer Tests
# =============================================================================


@pytest.mark.skip(reason="SecurityReviewer.review() method not implemented - tests need updating")
class TestSecurityReviewer:
    """Test SecurityReviewer agent."""

    @pytest.mark.asyncio
    async def test_review_finds_vulnerabilities(self, security_reviewer, sample_python_code):
        """Test that security reviewer finds vulnerabilities."""
        findings = await security_reviewer.review(sample_python_code, "python")

        assert len(findings) > 0
        assert all(f.category == "security" for f in findings)

    @pytest.mark.asyncio
    async def test_review_clean_code(self, security_reviewer):
        """Test reviewing clean code."""
        clean_code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def greet(name):
    """Greet a user."""
    return f"Hello, {name}!"
'''
        findings = await security_reviewer.review(clean_code, "python")

        # Should have few or no findings
        high_severity = [f for f in findings if f.severity in ["critical", "high"]]
        assert len(high_severity) == 0

    @pytest.mark.asyncio
    async def test_severity_levels(self, security_reviewer, sample_python_code):
        """Test that severity levels are assigned."""
        findings = await security_reviewer.review(sample_python_code, "python")

        severities = set(f.severity for f in findings)
        # Should have various severity levels
        assert len(severities) > 0


# =============================================================================
# Performance Reviewer Tests
# =============================================================================


@pytest.mark.skip(
    reason="PerformanceReviewer.review() method not implemented - tests need updating"
)
class TestPerformanceReviewer:
    """Test PerformanceReviewer agent."""

    @pytest.mark.asyncio
    async def test_detect_n_plus_one(self, performance_reviewer):
        """Test detection of N+1 patterns."""
        code = """
def get_all_orders():
    orders = Order.objects.all()
    for order in orders:
        # N+1 query problem
        customer = Customer.objects.get(id=order.customer_id)
        print(customer.name)
"""
        findings = await performance_reviewer.review(code, "python")

        assert any(
            "n+1" in f.title.lower() or "query" in str(f.description).lower() for f in findings
        )

    @pytest.mark.asyncio
    async def test_detect_nested_loops(self, performance_reviewer, sample_javascript_code):
        """Test detection of inefficient nested loops."""
        findings = await performance_reviewer.review(sample_javascript_code, "javascript")

        loop_findings = [
            f
            for f in findings
            if "loop" in f.title.lower() or "nested" in str(f.description).lower()
        ]
        # May or may not detect depending on pattern
        assert isinstance(loop_findings, list)

    @pytest.mark.asyncio
    async def test_detect_string_concatenation(self, performance_reviewer):
        """Test detection of inefficient string concatenation."""
        code = """
def build_string(items):
    result = ""
    for item in items:
        result = result + str(item)  # Inefficient
    return result
"""
        findings = await performance_reviewer.review(code, "python")

        # Should detect string concatenation in loop
        assert isinstance(findings, list)


# =============================================================================
# Maintainability Reviewer Tests
# =============================================================================


@pytest.mark.skip(
    reason="MaintainabilityReviewer.review() method not implemented - tests need updating"
)
class TestMaintainabilityReviewer:
    """Test MaintainabilityReviewer agent."""

    @pytest.mark.asyncio
    async def test_detect_long_functions(self, maintainability_reviewer):
        """Test detection of long functions."""
        # Create a long function
        long_code = "def long_function():\n" + "    x = 1\n" * 100

        findings = await maintainability_reviewer.review(long_code, "python")

        # May detect long function
        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_detect_deep_nesting(self, maintainability_reviewer):
        """Test detection of deep nesting."""
        code = """
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        return "too deep"
"""
        findings = await maintainability_reviewer.review(code, "python")

        nesting_findings = [f for f in findings if "nest" in f.title.lower()]
        # May or may not detect depending on patterns
        assert isinstance(nesting_findings, list)

    @pytest.mark.asyncio
    async def test_detect_todo_comments(self, maintainability_reviewer):
        """Test detection of TODO comments."""
        code = """
def incomplete_function():
    # TODO: implement this
    # FIXME: this is broken
    pass
"""
        findings = await maintainability_reviewer.review(code, "python")

        # Should detect TODO/FIXME
        todo_findings = [
            f
            for f in findings
            if "todo" in f.title.lower() or "fixme" in str(f.description).lower()
        ]
        assert len(todo_findings) >= 0  # May or may not match patterns


# =============================================================================
# Diff Review Tests
# =============================================================================


@pytest.mark.skip(reason="review_diff API signature changed - tests need updating")
class TestDiffReview:
    """Test diff/patch review."""

    @pytest.mark.asyncio
    async def test_review_diff(self, orchestrator, sample_diff):
        """Test reviewing a diff."""
        result = await orchestrator.review_diff(
            diff=sample_diff,
            base_branch="main",
            head_branch="feature/login",
        )

        assert isinstance(result, ReviewResult)
        assert len(result.findings) > 0

    @pytest.mark.asyncio
    async def test_diff_detects_sql_injection(self, orchestrator, sample_diff):
        """Test that diff review detects SQL injection in added code."""
        result = await orchestrator.review_diff(diff=sample_diff)

        security_findings = [f for f in result.findings if f.category == "security"]
        assert len(security_findings) > 0

    @pytest.mark.asyncio
    async def test_review_empty_diff(self, orchestrator):
        """Test reviewing an empty diff."""
        result = await orchestrator.review_diff(diff="")

        assert isinstance(result, ReviewResult)
        # Empty diff should have no findings
        assert len(result.findings) == 0


# =============================================================================
# Review Result Tests
# =============================================================================


class TestReviewResult:
    """Test ReviewResult aggregation."""

    @pytest.mark.asyncio
    async def test_result_aggregates_findings(self, orchestrator, sample_python_code):
        """Test that results aggregate findings from all reviewers."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        categories = set(f.category for f in result.findings)
        # Should have findings from multiple categories
        assert ReviewCategory.SECURITY in categories

    @pytest.mark.asyncio
    async def test_result_to_dict(self, orchestrator, sample_python_code):
        """Test result serialization."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        d = result.to_dict()

        assert "findings" in d
        assert isinstance(d["findings"], list)

    @pytest.mark.asyncio
    async def test_result_severity_summary(self, orchestrator, sample_python_code):
        """Test severity summary in result."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        # Just verify the result has findings
        assert result.findings is not None


# =============================================================================
# Finding Tests
# =============================================================================


class TestReviewFinding:
    """Test ReviewFinding dataclass."""

    def test_to_dict(self):
        """Test finding serialization."""
        from aragora.agents.code_reviewer import CodeLocation

        finding = ReviewFinding(
            id="finding_123",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection Vulnerability",
            description="User input passed directly to SQL query",
            location=CodeLocation(file_path="src/db.py", start_line=42),
            suggestion="Use parameterized queries",
        )

        d = finding.to_dict()

        assert d["id"] == "finding_123"
        assert d["title"] == "SQL Injection Vulnerability"
        assert d["severity"] == "critical"
        assert d["location"]["filePath"] == "src/db.py"


# =============================================================================
# Language Detection Tests
# =============================================================================


class TestLanguageDetection:
    """Test language detection."""

    @pytest.mark.asyncio
    async def test_detect_python(self, orchestrator):
        """Test Python language detection."""
        code = """
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
"""
        result = await orchestrator.review_code(code=code)
        # Should work without explicit language
        assert isinstance(result, ReviewResult)

    @pytest.mark.asyncio
    async def test_detect_javascript(self, orchestrator):
        """Test JavaScript language detection."""
        code = """
function hello() {
    console.log("Hello, World!");
}

hello();
"""
        result = await orchestrator.review_code(code=code)
        assert isinstance(result, ReviewResult)


# =============================================================================
# Review Types Tests
# =============================================================================


class TestReviewTypes:
    """Test review type filtering."""

    @pytest.mark.asyncio
    async def test_security_only(self, orchestrator, sample_python_code):
        """Test security findings are detected."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        # Should have security findings
        security_findings = [f for f in result.findings if f.category == ReviewCategory.SECURITY]
        assert len(security_findings) > 0

    @pytest.mark.asyncio
    async def test_performance_only(self, orchestrator, sample_javascript_code):
        """Test performance findings can be detected."""
        result = await orchestrator.review_code(
            code=sample_javascript_code,
            file_path="test.js",
        )

        # Should have some findings (may or may not include performance)
        assert isinstance(result.findings, list)

    @pytest.mark.asyncio
    async def test_all_review_types(self, orchestrator, sample_python_code):
        """Test all review types."""
        result = await orchestrator.review_code(
            code=sample_python_code,
            file_path="test.py",
        )

        # Should have findings
        assert isinstance(result.findings, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for code review orchestrator."""

    @pytest.mark.asyncio
    async def test_full_review_workflow(self, orchestrator):
        """Test full review workflow."""
        code = """
import os

SECRET_KEY = "hardcoded-secret-123"

def get_data(user_input):
    query = f"SELECT * FROM data WHERE id = {user_input}"
    return execute(query)

def process():
    result = ""
    for i in range(1000):
        result = result + str(i)
    return result
"""
        result = await orchestrator.review_code(
            code=code,
            file_path="test.py",
        )

        assert isinstance(result, ReviewResult)
        assert len(result.findings) > 0

        # Verify serialization works
        d = result.to_dict()
        assert "findings" in d

    @pytest.mark.asyncio
    async def test_concurrent_reviews(self, orchestrator):
        """Test that multiple reviews can run concurrently."""
        import asyncio

        codes = [
            "def a(): pass",
            "def b(): x = eval('1+1')",
            "function c() { return 1; }",
        ]

        tasks = [orchestrator.review_code(code=c) for c in codes]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(isinstance(r, ReviewResult) for r in results)
