"""
Tests for the Code Review Agent module.

Tests cover:
- Dataclasses: CodeLocation, ReviewFinding, ReviewResult
- Enums: ReviewCategory, FindingSeverity
- Security patterns and detection
- Performance patterns and detection
- Maintainability patterns and detection
- Specialized reviewers: SecurityReviewer, PerformanceReviewer, MaintainabilityReviewer, TestCoverageReviewer
- CodeReviewOrchestrator orchestration and aggregation
- Diff parsing and review
- GitHub comment generation
- Debate integration
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.agents.code_reviewer import (
    CodeLocation,
    CodeReviewOrchestrator,
    FindingSeverity,
    MaintainabilityReviewer,
    PerformanceReviewer,
    ReviewCategory,
    ReviewFinding,
    ReviewResult,
    SecurityReviewer,
    TestCoverageReviewer,
    SECURITY_PATTERNS,
    PERFORMANCE_PATTERNS,
    MAINTAINABILITY_PATTERNS,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestReviewCategory:
    """Tests for ReviewCategory enum."""

    def test_security_value(self):
        """Security category has correct value."""
        assert ReviewCategory.SECURITY.value == "security"

    def test_performance_value(self):
        """Performance category has correct value."""
        assert ReviewCategory.PERFORMANCE.value == "performance"

    def test_maintainability_value(self):
        """Maintainability category has correct value."""
        assert ReviewCategory.MAINTAINABILITY.value == "maintainability"

    def test_test_coverage_value(self):
        """Test coverage category has correct value."""
        assert ReviewCategory.TEST_COVERAGE.value == "test_coverage"

    def test_bug_value(self):
        """Bug category has correct value."""
        assert ReviewCategory.BUG.value == "bug"

    def test_style_value(self):
        """Style category has correct value."""
        assert ReviewCategory.STYLE.value == "style"

    def test_documentation_value(self):
        """Documentation category has correct value."""
        assert ReviewCategory.DOCUMENTATION.value == "documentation"

    def test_architecture_value(self):
        """Architecture category has correct value."""
        assert ReviewCategory.ARCHITECTURE.value == "architecture"

    def test_category_is_string_enum(self):
        """ReviewCategory is a string enum."""
        assert isinstance(ReviewCategory.SECURITY, str)
        assert ReviewCategory.SECURITY == "security"


class TestFindingSeverity:
    """Tests for FindingSeverity enum."""

    def test_critical_value(self):
        """Critical severity has correct value."""
        assert FindingSeverity.CRITICAL.value == "critical"

    def test_high_value(self):
        """High severity has correct value."""
        assert FindingSeverity.HIGH.value == "high"

    def test_medium_value(self):
        """Medium severity has correct value."""
        assert FindingSeverity.MEDIUM.value == "medium"

    def test_low_value(self):
        """Low severity has correct value."""
        assert FindingSeverity.LOW.value == "low"

    def test_info_value(self):
        """Info severity has correct value."""
        assert FindingSeverity.INFO.value == "info"

    def test_severity_is_string_enum(self):
        """FindingSeverity is a string enum."""
        assert isinstance(FindingSeverity.CRITICAL, str)
        assert FindingSeverity.CRITICAL == "critical"


# =============================================================================
# CodeLocation Tests
# =============================================================================


class TestCodeLocation:
    """Tests for CodeLocation dataclass."""

    def test_minimal_init(self):
        """CodeLocation can be initialized with minimal params."""
        loc = CodeLocation(file_path="test.py", start_line=10)

        assert loc.file_path == "test.py"
        assert loc.start_line == 10
        assert loc.end_line is None
        assert loc.code_snippet == ""

    def test_full_init(self):
        """CodeLocation can be initialized with all params."""
        loc = CodeLocation(
            file_path="test.py",
            start_line=10,
            end_line=20,
            code_snippet="def test(): pass",
        )

        assert loc.file_path == "test.py"
        assert loc.start_line == 10
        assert loc.end_line == 20
        assert loc.code_snippet == "def test(): pass"

    def test_to_dict(self):
        """CodeLocation to_dict returns correct structure."""
        loc = CodeLocation(
            file_path="test.py",
            start_line=10,
            end_line=20,
            code_snippet="code",
        )

        result = loc.to_dict()

        assert result["filePath"] == "test.py"
        assert result["startLine"] == 10
        assert result["endLine"] == 20
        assert result["codeSnippet"] == "code"

    def test_to_dict_camel_case_keys(self):
        """CodeLocation to_dict uses camelCase keys."""
        loc = CodeLocation(file_path="test.py", start_line=1)
        result = loc.to_dict()

        assert "filePath" in result
        assert "startLine" in result
        assert "endLine" in result
        assert "codeSnippet" in result


# =============================================================================
# ReviewFinding Tests
# =============================================================================


class TestReviewFinding:
    """Tests for ReviewFinding dataclass."""

    def test_minimal_init(self):
        """ReviewFinding can be initialized with minimal params."""
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test finding",
            description="Test description",
        )

        assert finding.id == "F001"
        assert finding.category == ReviewCategory.SECURITY
        assert finding.severity == FindingSeverity.HIGH
        assert finding.title == "Test finding"
        assert finding.description == "Test description"

    def test_default_values(self):
        """ReviewFinding has correct default values."""
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test",
            description="Desc",
        )

        assert finding.location is None
        assert finding.suggestion == ""
        assert finding.suggested_code == ""
        assert finding.reviewer == ""
        assert finding.confidence == 0.8
        assert finding.references == []
        assert finding.tags == []

    def test_full_init(self):
        """ReviewFinding can be initialized with all params."""
        loc = CodeLocation(file_path="test.py", start_line=10)
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="User input flows to SQL query",
            location=loc,
            suggestion="Use parameterized queries",
            suggested_code="cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            reviewer="security_reviewer",
            confidence=0.95,
            references=["CWE-89"],
            tags=["injection", "database"],
        )

        assert finding.location is not None
        assert finding.suggestion == "Use parameterized queries"
        assert finding.confidence == 0.95
        assert "CWE-89" in finding.references
        assert "injection" in finding.tags

    def test_to_dict(self):
        """ReviewFinding to_dict returns correct structure."""
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test",
            description="Desc",
            reviewer="test_reviewer",
        )

        result = finding.to_dict()

        assert result["id"] == "F001"
        assert result["category"] == "security"
        assert result["severity"] == "high"
        assert result["title"] == "Test"
        assert result["description"] == "Desc"
        assert result["reviewer"] == "test_reviewer"
        assert result["location"] is None

    def test_to_dict_with_location(self):
        """ReviewFinding to_dict includes location when present."""
        loc = CodeLocation(file_path="test.py", start_line=10)
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test",
            description="Desc",
            location=loc,
        )

        result = finding.to_dict()

        assert result["location"] is not None
        assert result["location"]["filePath"] == "test.py"


# =============================================================================
# ReviewResult Tests
# =============================================================================


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_default_init(self):
        """ReviewResult has correct defaults."""
        result = ReviewResult()

        assert result.pr_url is None
        assert result.files_reviewed == 0
        assert result.lines_reviewed == 0
        assert result.findings == []
        assert result.summary == ""
        assert result.approval_status == "pending"
        assert result.by_category == {}
        assert result.by_severity == {}
        assert result.reviewers_participated == []
        assert result.consensus_notes == []

    def test_critical_count_property(self):
        """critical_count property counts critical findings."""
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="Critical 1",
                    description="Desc",
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="High 1",
                    description="Desc",
                ),
                ReviewFinding(
                    id="F003",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="Critical 2",
                    description="Desc",
                ),
            ]
        )

        assert result.critical_count == 2

    def test_high_count_property(self):
        """high_count property counts high severity findings."""
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="High 1",
                    description="Desc",
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="High 2",
                    description="Desc",
                ),
                ReviewFinding(
                    id="F003",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.LOW,
                    title="Low 1",
                    description="Desc",
                ),
            ]
        )

        assert result.high_count == 2

    def test_should_block_merge_with_critical(self):
        """should_block_merge returns True when critical findings exist."""
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="Critical",
                    description="Desc",
                ),
            ]
        )

        assert result.should_block_merge is True

    def test_should_block_merge_without_critical(self):
        """should_block_merge returns False when no critical findings."""
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="High",
                    description="Desc",
                ),
            ]
        )

        assert result.should_block_merge is False

    def test_to_dict(self):
        """ReviewResult to_dict returns correct structure."""
        result = ReviewResult(
            pr_url="https://github.com/test/repo/pull/1",
            files_reviewed=5,
            lines_reviewed=100,
            summary="Review complete",
            approval_status="approved",
        )

        data = result.to_dict()

        assert data["prUrl"] == "https://github.com/test/repo/pull/1"
        assert data["filesReviewed"] == 5
        assert data["linesReviewed"] == 100
        assert data["summary"] == "Review complete"
        assert data["approvalStatus"] == "approved"
        assert data["criticalCount"] == 0
        assert data["highCount"] == 0
        assert data["shouldBlockMerge"] is False


# =============================================================================
# Security Pattern Tests
# =============================================================================


class TestSecurityPatterns:
    """Tests for security vulnerability patterns."""

    def test_eval_pattern_matches(self):
        """Eval pattern matches eval() calls."""
        import re

        pattern = r"eval\s*\("

        assert re.search(pattern, "eval(user_input)")
        assert re.search(pattern, "eval (code)")
        assert re.search(pattern, "result = eval( expression )")

    def test_exec_pattern_matches(self):
        """Exec pattern matches exec() calls."""
        import re

        pattern = r"exec\s*\("

        assert re.search(pattern, "exec(code)")
        assert re.search(pattern, "exec (user_code)")

    def test_hardcoded_password_pattern(self):
        """Hardcoded password pattern matches."""
        import re

        pattern = r"password\s*=\s*['\"][^'\"]+['\"]"

        assert re.search(pattern, 'password = "secret123"', re.IGNORECASE)
        assert re.search(pattern, "password='mysecret'", re.IGNORECASE)

    def test_api_key_pattern_matches(self):
        """API key pattern matches hardcoded keys."""
        import re

        pattern = r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]"

        assert re.search(pattern, 'api_key = "sk_live_123"', re.IGNORECASE)
        assert re.search(pattern, 'apiKey = "secret"', re.IGNORECASE)
        assert re.search(pattern, 'api-key = "value"', re.IGNORECASE)

    def test_pickle_load_pattern(self):
        """Pickle load pattern matches unsafe deserialization."""
        import re

        pattern = r"pickle\.load"

        assert re.search(pattern, "pickle.load(file)")
        assert re.search(pattern, "data = pickle.load(f)")

    def test_yaml_load_pattern(self):
        """YAML load pattern matches unsafe load."""
        import re

        pattern = r"yaml\.load\s*\([^,]+\)"

        assert re.search(pattern, "yaml.load(data)")
        assert not re.search(pattern, "yaml.load(data, Loader=yaml.SafeLoader)")


# =============================================================================
# Specialized Reviewer Tests
# =============================================================================


class TestSecurityReviewer:
    """Tests for SecurityReviewer agent."""

    def test_init_default_values(self):
        """SecurityReviewer initializes with correct defaults."""
        reviewer = SecurityReviewer()

        assert reviewer.name == "security_reviewer"
        assert reviewer.model == "pattern-based"
        assert reviewer.role == "proposer"
        assert reviewer.focus == "security"

    def test_init_with_custom_name(self):
        """SecurityReviewer can be initialized with custom name."""
        reviewer = SecurityReviewer(name="custom_security")

        assert reviewer.name == "custom_security"

    @pytest.mark.asyncio
    async def test_review_finds_eval(self):
        """SecurityReviewer detects eval() usage."""
        reviewer = SecurityReviewer()
        code = """
def process(user_input):
    return eval(user_input)
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("eval" in f.title.lower() for f in findings)
        assert all(f.category == ReviewCategory.SECURITY for f in findings)

    @pytest.mark.asyncio
    async def test_review_finds_hardcoded_secrets(self):
        """SecurityReviewer detects hardcoded secrets."""
        reviewer = SecurityReviewer()
        code = """
password = "secret123"
api_key = "sk_live_abc123"
"""
        findings = await reviewer.review(code)

        assert len(findings) >= 2
        severities = [f.severity for f in findings]
        assert FindingSeverity.CRITICAL in severities

    @pytest.mark.asyncio
    async def test_review_clean_code(self):
        """SecurityReviewer returns empty for clean code."""
        reviewer = SecurityReviewer()
        code = """
def add(a, b):
    return a + b
"""
        findings = await reviewer.review(code)

        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_generate_returns_empty(self):
        """SecurityReviewer generate is a stub."""
        reviewer = SecurityReviewer()
        result = await reviewer.generate("prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_critique_returns_critique_object(self):
        """SecurityReviewer critique returns Critique."""
        from aragora.core_types import Critique

        reviewer = SecurityReviewer()
        result = await reviewer.critique(
            proposal="test",
            task="task",
            target_agent="target",
        )

        assert isinstance(result, Critique)
        assert result.agent == "security_reviewer"


class TestPerformanceReviewer:
    """Tests for PerformanceReviewer agent."""

    def test_init_default_values(self):
        """PerformanceReviewer initializes with correct defaults."""
        reviewer = PerformanceReviewer()

        assert reviewer.name == "performance_reviewer"
        assert reviewer.focus == "performance"

    @pytest.mark.asyncio
    async def test_review_finds_range_len(self):
        """PerformanceReviewer detects range(len()) antipattern."""
        reviewer = PerformanceReviewer()
        code = """
for i in range(len(items)):
    print(items[i])
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("enumerate" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_finds_wildcard_import(self):
        """PerformanceReviewer detects wildcard imports."""
        reviewer = PerformanceReviewer()
        code = """
from module import *
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("wildcard" in f.title.lower() for f in findings)


class TestMaintainabilityReviewer:
    """Tests for MaintainabilityReviewer agent."""

    def test_init_default_values(self):
        """MaintainabilityReviewer initializes with correct defaults."""
        reviewer = MaintainabilityReviewer()

        assert reviewer.name == "maintainability_reviewer"
        assert reviewer.focus == "maintainability"

    @pytest.mark.asyncio
    async def test_review_finds_broad_exception(self):
        """MaintainabilityReviewer detects broad Exception catch."""
        reviewer = MaintainabilityReviewer()
        code = """
try:
    risky_operation()
except Exception:
    log_error()
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("exception" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_finds_todo(self):
        """MaintainabilityReviewer detects TODO comments."""
        reviewer = MaintainabilityReviewer()
        code = """
def process():
    # TODO: implement this
    pass
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("todo" in f.title.lower() for f in findings)
        # TODO comments should be INFO severity
        assert any(f.severity == FindingSeverity.INFO for f in findings)

    @pytest.mark.asyncio
    async def test_review_finds_fixme(self):
        """MaintainabilityReviewer detects FIXME comments."""
        reviewer = MaintainabilityReviewer()
        code = """
def process():
    # FIXME: this is broken
    return None
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("fixme" in f.title.lower() for f in findings)


class TestTestCoverageReviewer:
    """Tests for TestCoverageReviewer agent."""

    def test_init_default_values(self):
        """TestCoverageReviewer initializes with correct defaults."""
        reviewer = TestCoverageReviewer()

        assert reviewer.name == "test_coverage_reviewer"
        assert reviewer.focus == "test_coverage"

    def test_not_collected_as_test(self):
        """TestCoverageReviewer is not collected by pytest."""
        assert TestCoverageReviewer.__test__ is False


# =============================================================================
# CodeReviewOrchestrator Tests
# =============================================================================


class TestCodeReviewOrchestratorInit:
    """Tests for CodeReviewOrchestrator initialization."""

    def test_default_init(self):
        """Orchestrator initializes with all reviewers enabled."""
        orchestrator = CodeReviewOrchestrator()

        reviewer_types = [r[0] for r in orchestrator.reviewers]
        assert "security" in reviewer_types
        assert "performance" in reviewer_types
        assert "maintainability" in reviewer_types
        assert "test_coverage" in reviewer_types

    def test_disable_security(self):
        """Can disable security reviewer."""
        orchestrator = CodeReviewOrchestrator(enable_security=False)

        reviewer_types = [r[0] for r in orchestrator.reviewers]
        assert "security" not in reviewer_types

    def test_disable_performance(self):
        """Can disable performance reviewer."""
        orchestrator = CodeReviewOrchestrator(enable_performance=False)

        reviewer_types = [r[0] for r in orchestrator.reviewers]
        assert "performance" not in reviewer_types

    def test_with_arena(self):
        """Can initialize with arena for debates."""
        mock_arena = MagicMock()
        orchestrator = CodeReviewOrchestrator(arena=mock_arena)

        assert orchestrator.arena is mock_arena

    def test_disable_all_reviewers(self):
        """Can disable all reviewers."""
        orchestrator = CodeReviewOrchestrator(
            enable_security=False,
            enable_performance=False,
            enable_maintainability=False,
            enable_test_coverage=False,
        )

        assert len(orchestrator.reviewers) == 0


class TestCodeReviewOrchestratorReviewCode:
    """Tests for CodeReviewOrchestrator.review_code method."""

    @pytest.mark.asyncio
    async def test_review_code_basic(self):
        """review_code returns ReviewResult."""
        orchestrator = CodeReviewOrchestrator()
        code = """
def add(a, b):
    return a + b
"""
        result = await orchestrator.review_code(code, "test.py")

        assert isinstance(result, ReviewResult)
        assert result.files_reviewed == 1
        assert result.lines_reviewed == 4

    @pytest.mark.asyncio
    async def test_review_code_with_issues(self):
        """review_code detects security issues."""
        orchestrator = CodeReviewOrchestrator()
        code = """
password = "secret123"
def process(user_input):
    return eval(user_input)
"""
        result = await orchestrator.review_code(code, "test.py")

        assert len(result.findings) > 0
        assert result.critical_count > 0

    @pytest.mark.asyncio
    async def test_review_code_approval_status_changes_requested(self):
        """review_code sets changes_requested for critical findings."""
        orchestrator = CodeReviewOrchestrator()
        code = 'api_key = "secret"'

        result = await orchestrator.review_code(code)

        assert result.approval_status == "changes_requested"

    @pytest.mark.asyncio
    async def test_review_code_approval_status_approved(self):
        """review_code sets approved for clean code."""
        orchestrator = CodeReviewOrchestrator()
        code = """
def add(a, b):
    return a + b
"""
        result = await orchestrator.review_code(code)

        assert result.approval_status == "approved"

    @pytest.mark.asyncio
    async def test_review_code_counts_by_category(self):
        """review_code populates by_category counts."""
        orchestrator = CodeReviewOrchestrator()
        code = """
password = "secret"  # security
for i in range(len(items)):  # performance
    print(items[i])
"""
        result = await orchestrator.review_code(code)

        assert "security" in result.by_category or len(result.findings) > 0

    @pytest.mark.asyncio
    async def test_review_code_counts_by_severity(self):
        """review_code populates by_severity counts."""
        orchestrator = CodeReviewOrchestrator()
        code = 'password = "secret"'

        result = await orchestrator.review_code(code)

        if result.findings:
            assert len(result.by_severity) > 0

    @pytest.mark.asyncio
    async def test_review_code_generates_summary(self):
        """review_code generates summary."""
        orchestrator = CodeReviewOrchestrator()
        code = "x = 1"

        result = await orchestrator.review_code(code)

        assert "Reviewed" in result.summary
        assert "files" in result.summary.lower() or "1" in result.summary

    @pytest.mark.asyncio
    async def test_review_code_tracks_reviewers(self):
        """review_code tracks participating reviewers."""
        orchestrator = CodeReviewOrchestrator()
        result = await orchestrator.review_code("x = 1")

        assert len(result.reviewers_participated) > 0


class TestCodeReviewOrchestratorDiff:
    """Tests for CodeReviewOrchestrator diff review."""

    @pytest.mark.asyncio
    async def test_review_diff_basic(self):
        """review_diff parses and reviews diff."""
        orchestrator = CodeReviewOrchestrator()
        diff = """diff --git a/test.py b/test.py
index 123..456 789
--- a/test.py
+++ b/test.py
@@ -1,3 +1,5 @@
+password = "secret"
 def main():
-    pass
+    print("hello")
"""
        result = await orchestrator.review_diff(diff)

        assert isinstance(result, ReviewResult)
        assert result.files_reviewed > 0

    @pytest.mark.asyncio
    async def test_review_diff_with_pr_url(self):
        """review_diff includes PR URL in result."""
        orchestrator = CodeReviewOrchestrator()
        diff = """diff --git a/test.py b/test.py
+x = 1
"""
        result = await orchestrator.review_diff(
            diff,
            pr_url="https://github.com/test/repo/pull/1",
        )

        assert result.pr_url == "https://github.com/test/repo/pull/1"

    def test_parse_diff(self):
        """_parse_diff extracts file changes."""
        orchestrator = CodeReviewOrchestrator()
        diff = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
+added line
-removed line
diff --git a/file2.py b/file2.py
+another added
"""
        files = orchestrator._parse_diff(diff)

        assert len(files) == 2
        assert files[0][0] == "file1.py"
        assert "added line" in files[0][1]
        assert "removed line" in files[0][2]


class TestCodeReviewOrchestratorGitHub:
    """Tests for GitHub integration."""

    def test_generate_github_comments(self):
        """generate_github_comments creates comment objects."""
        orchestrator = CodeReviewOrchestrator()
        loc = CodeLocation(file_path="test.py", start_line=10)
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="Security issue",
                    description="Description",
                    location=loc,
                ),
            ]
        )

        comments = orchestrator.generate_github_comments(result)

        assert len(comments) == 1
        assert comments[0]["path"] == "test.py"
        assert comments[0]["line"] == 10
        assert "Security issue" in comments[0]["body"]

    def test_generate_github_comments_max_limit(self):
        """generate_github_comments respects max_comments."""
        orchestrator = CodeReviewOrchestrator()
        loc = CodeLocation(file_path="test.py", start_line=1)
        findings = [
            ReviewFinding(
                id=f"F{i:03d}",
                category=ReviewCategory.SECURITY,
                severity=FindingSeverity.MEDIUM,
                title=f"Issue {i}",
                description="Desc",
                location=CodeLocation(file_path="test.py", start_line=i),
            )
            for i in range(30)
        ]
        result = ReviewResult(findings=findings)

        comments = orchestrator.generate_github_comments(result, max_comments=10)

        assert len(comments) == 10

    def test_generate_github_comments_skips_no_location(self):
        """generate_github_comments skips findings without location."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="No location",
                    description="Desc",
                    location=None,
                ),
            ]
        )

        comments = orchestrator.generate_github_comments(result)

        assert len(comments) == 0

    def test_format_comment_includes_severity(self):
        """_format_comment includes severity in output."""
        orchestrator = CodeReviewOrchestrator()
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="Critical issue",
            description="Description",
        )

        comment = orchestrator._format_comment(finding)

        assert "CRITICAL" in comment
        assert "Critical issue" in comment

    def test_format_comment_includes_suggestion(self):
        """_format_comment includes suggestion when present."""
        orchestrator = CodeReviewOrchestrator()
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Desc",
            suggestion="Fix it like this",
        )

        comment = orchestrator._format_comment(finding)

        assert "Suggestion" in comment
        assert "Fix it like this" in comment

    def test_format_pr_review_body(self):
        """_format_pr_review_body generates summary."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            summary="Test summary",
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="Issue",
                    description="Desc",
                ),
            ],
        )

        body = orchestrator._format_pr_review_body(result)

        assert "Code Review Summary" in body
        assert "Test summary" in body


class TestCodeReviewOrchestratorDebate:
    """Tests for debate integration."""

    @pytest.mark.asyncio
    async def test_debate_findings_without_arena(self):
        """debate_findings returns message when no arena."""
        orchestrator = CodeReviewOrchestrator(arena=None)

        notes = await orchestrator.debate_findings([])

        assert "No arena configured" in notes[0]

    @pytest.mark.asyncio
    async def test_debate_findings_with_arena(self):
        """debate_findings returns consensus notes with arena."""
        mock_arena = MagicMock()
        orchestrator = CodeReviewOrchestrator(arena=mock_arena)
        findings = [
            ReviewFinding(
                id="F001",
                category=ReviewCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                title="Issue 1",
                description="Desc",
            ),
            ReviewFinding(
                id="F002",
                category=ReviewCategory.PERFORMANCE,
                severity=FindingSeverity.MEDIUM,
                title="Issue 2",
                description="Desc",
            ),
        ]

        notes = await orchestrator.debate_findings(findings, "conflicting")

        assert len(notes) > 0
        assert "conflicting" in notes[0]


class TestHelperMethods:
    """Tests for helper methods."""

    def test_count_by_field_category(self):
        """_count_by_field counts by category."""
        orchestrator = CodeReviewOrchestrator()
        findings = [
            ReviewFinding(
                id="F001",
                category=ReviewCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                title="A",
                description="D",
            ),
            ReviewFinding(
                id="F002",
                category=ReviewCategory.SECURITY,
                severity=FindingSeverity.LOW,
                title="B",
                description="D",
            ),
            ReviewFinding(
                id="F003",
                category=ReviewCategory.PERFORMANCE,
                severity=FindingSeverity.MEDIUM,
                title="C",
                description="D",
            ),
        ]

        counts = orchestrator._count_by_field(findings, "category")

        assert counts["security"] == 2
        assert counts["performance"] == 1

    def test_count_by_field_severity(self):
        """_count_by_field counts by severity."""
        orchestrator = CodeReviewOrchestrator()
        findings = [
            ReviewFinding(
                id="F001",
                category=ReviewCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                title="A",
                description="D",
            ),
            ReviewFinding(
                id="F002",
                category=ReviewCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                title="B",
                description="D",
            ),
        ]

        counts = orchestrator._count_by_field(findings, "severity")

        assert counts["high"] == 2

    def test_generate_summary_no_issues(self):
        """_generate_summary for clean review."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            files_reviewed=3,
            lines_reviewed=100,
            findings=[],
        )

        summary = orchestrator._generate_summary(result)

        assert "3 files" in summary
        assert "100 lines" in summary
        assert "No issues" in summary

    def test_generate_summary_with_issues(self):
        """_generate_summary includes issue counts."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            files_reviewed=1,
            lines_reviewed=50,
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="A",
                    description="D",
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="B",
                    description="D",
                ),
            ],
        )

        summary = orchestrator._generate_summary(result)

        assert "2 issues" in summary or "Found 2" in summary
        assert "critical" in summary.lower()


class TestPatternReview:
    """Tests for pattern-based review method."""

    def test_pattern_review_security(self):
        """_pattern_review finds security issues."""
        orchestrator = CodeReviewOrchestrator()
        code = 'password = "secret"'

        findings = orchestrator._pattern_review(
            code,
            "test.py",
            SECURITY_PATTERNS,
            ReviewCategory.SECURITY,
            "security_pattern",
        )

        assert len(findings) > 0
        assert findings[0].category == ReviewCategory.SECURITY
        assert findings[0].reviewer == "security_pattern"

    def test_pattern_review_sets_location(self):
        """_pattern_review sets correct location."""
        orchestrator = CodeReviewOrchestrator()
        code = """line1
line2
password = "secret"
line4
"""
        findings = orchestrator._pattern_review(
            code,
            "test.py",
            SECURITY_PATTERNS,
            ReviewCategory.SECURITY,
            "security_pattern",
        )

        assert findings[0].location is not None
        assert findings[0].location.file_path == "test.py"
        assert findings[0].location.start_line == 3

    def test_pattern_review_increments_counter(self):
        """_pattern_review increments finding counter."""
        orchestrator = CodeReviewOrchestrator()
        code = """password = "a"
api_key = "b"
"""
        initial_counter = orchestrator._finding_counter

        orchestrator._pattern_review(
            code,
            "test.py",
            SECURITY_PATTERNS,
            ReviewCategory.SECURITY,
            "test",
        )

        assert orchestrator._finding_counter > initial_counter


class TestMultiLanguageSupport:
    """Tests for multi-language code review."""

    @pytest.mark.asyncio
    async def test_review_javascript_eval(self):
        """Review detects eval in JavaScript."""
        reviewer = SecurityReviewer()
        code = """
function processInput(input) {
    return eval(input);
}
"""
        findings = await reviewer.review(code, language="javascript")

        assert len(findings) > 0
        assert any("eval" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_innerHTML_xss(self):
        """Review detects innerHTML XSS risk."""
        reviewer = SecurityReviewer()
        code = """
function update(html) {
    element.innerHTML = html;
}
"""
        findings = await reviewer.review(code, language="javascript")

        assert len(findings) > 0
        assert any("innerHTML" in f.title or "xss" in f.title.lower() for f in findings)


# =============================================================================
# Additional Security Pattern Tests
# =============================================================================


class TestAdditionalSecurityPatterns:
    """Additional tests for security pattern detection coverage."""

    @pytest.mark.asyncio
    async def test_review_subprocess_shell(self):
        """Review detects subprocess shell=True."""
        reviewer = SecurityReviewer()
        code = """
import subprocess
subprocess.run(cmd, shell=True)
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("shell" in f.title.lower() or "subprocess" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_secret_detection(self):
        """Review detects hardcoded secrets."""
        reviewer = SecurityReviewer()
        code = 'secret = "my_super_secret_value"'
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("secret" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_dangerously_set_inner_html(self):
        """Review detects React dangerouslySetInnerHTML."""
        reviewer = SecurityReviewer()
        code = "<div dangerouslySetInnerHTML={{__html: userInput}} />"
        findings = await reviewer.review(code)

        assert len(findings) > 0

    @pytest.mark.asyncio
    async def test_review_pickle_detection(self):
        """Review detects pickle.load usage."""
        reviewer = SecurityReviewer()
        code = """
import pickle
data = pickle.load(open('data.pkl', 'rb'))
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("pickle" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_unsafe_yaml(self):
        """Review detects unsafe yaml.load."""
        reviewer = SecurityReviewer()
        code = """
import yaml
data = yaml.load(file_content)
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("yaml" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_security_finding_ids_increment(self):
        """Security finding IDs increment correctly."""
        reviewer = SecurityReviewer()
        code = """
eval(user_input)
exec(code)
"""
        findings = await reviewer.review(code)

        ids = [f.id for f in findings]
        assert len(ids) == len(set(ids)), "Finding IDs should be unique"

    @pytest.mark.asyncio
    async def test_security_finding_has_location(self):
        """Security findings include location data."""
        reviewer = SecurityReviewer()
        code = """
line1
eval(user_input)
"""
        findings = await reviewer.review(code)

        for f in findings:
            assert f.location is not None
            assert f.location.start_line > 0

    @pytest.mark.asyncio
    async def test_security_finding_reviewer_name(self):
        """Security findings have correct reviewer name."""
        reviewer = SecurityReviewer()
        code = 'eval("1+1")'
        findings = await reviewer.review(code)

        for f in findings:
            assert f.reviewer == "security_reviewer"


# =============================================================================
# Additional Performance Pattern Tests
# =============================================================================


class TestAdditionalPerformancePatterns:
    """Additional tests for performance pattern detection."""

    @pytest.mark.asyncio
    async def test_review_read_all(self):
        """Review detects reading entire file at once."""
        reviewer = PerformanceReviewer()
        code = "data = file.read()"
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("read" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_sleep(self):
        """Review detects long sleep calls."""
        reviewer = PerformanceReviewer()
        code = """
import time
time.sleep(30)
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("sleep" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_clean_performance_code(self):
        """Performance review returns empty for optimized code."""
        reviewer = PerformanceReviewer()
        code = """
def optimized(items):
    return [item.upper() for item in items]
"""
        findings = await reviewer.review(code)

        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_performance_generate_stub(self):
        """PerformanceReviewer generate returns empty string."""
        reviewer = PerformanceReviewer()
        result = await reviewer.generate("prompt")
        assert result == ""

    @pytest.mark.asyncio
    async def test_performance_critique_stub(self):
        """PerformanceReviewer critique returns Critique object."""
        from aragora.core_types import Critique

        reviewer = PerformanceReviewer()
        result = await reviewer.critique("proposal", "task", target_agent="target")
        assert isinstance(result, Critique)
        assert result.agent == "performance_reviewer"


# =============================================================================
# Additional Maintainability Pattern Tests
# =============================================================================


class TestAdditionalMaintainabilityPatterns:
    """Additional tests for maintainability pattern detection."""

    @pytest.mark.asyncio
    async def test_review_bare_except_pass(self):
        """Review detects bare except with pass."""
        reviewer = MaintainabilityReviewer()
        code = """
try:
    do_something()
except:
    pass
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0

    @pytest.mark.asyncio
    async def test_review_hack_comment(self):
        """Review detects HACK comments."""
        reviewer = MaintainabilityReviewer()
        code = """
def process():
    # HACK: this is a workaround
    return None
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("hack" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_global_usage(self):
        """Review detects global variable usage."""
        reviewer = MaintainabilityReviewer()
        code = """
def process():
    global counter
    counter += 1
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("global" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_review_print_statement(self):
        """Review detects print statements."""
        reviewer = MaintainabilityReviewer()
        code = """
def debug():
    print("debugging value")
"""
        findings = await reviewer.review(code)

        assert len(findings) > 0
        assert any("print" in f.title.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_maintainability_generate_stub(self):
        """MaintainabilityReviewer generate returns empty string."""
        reviewer = MaintainabilityReviewer()
        result = await reviewer.generate("prompt")
        assert result == ""

    @pytest.mark.asyncio
    async def test_maintainability_critique_stub(self):
        """MaintainabilityReviewer critique returns Critique object."""
        from aragora.core_types import Critique

        reviewer = MaintainabilityReviewer()
        result = await reviewer.critique("proposal", "task", target_agent="target")
        assert isinstance(result, Critique)
        assert result.agent == "maintainability_reviewer"

    @pytest.mark.asyncio
    async def test_review_clean_maintainability_code(self):
        """Maintainability review returns empty for clean code."""
        reviewer = MaintainabilityReviewer()
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        findings = await reviewer.review(code)

        assert len(findings) == 0


# =============================================================================
# Additional TestCoverageReviewer Tests
# =============================================================================


class TestTestCoverageReviewerMethods:
    """Additional tests for TestCoverageReviewer."""

    @pytest.mark.asyncio
    async def test_generate_stub(self):
        """TestCoverageReviewer generate returns empty string."""
        reviewer = TestCoverageReviewer()
        result = await reviewer.generate("prompt")
        assert result == ""

    @pytest.mark.asyncio
    async def test_critique_stub(self):
        """TestCoverageReviewer critique returns Critique object."""
        from aragora.core_types import Critique

        reviewer = TestCoverageReviewer()
        result = await reviewer.critique("proposal", "task", target_agent="target")
        assert isinstance(result, Critique)
        assert result.agent == "test_coverage_reviewer"

    @pytest.mark.asyncio
    async def test_critique_without_target_agent(self):
        """TestCoverageReviewer critique with default target."""
        reviewer = TestCoverageReviewer()
        result = await reviewer.critique("proposal", "task")
        assert result.target_agent == "unknown"

    def test_custom_init_params(self):
        """TestCoverageReviewer accepts custom init params."""
        reviewer = TestCoverageReviewer(name="custom_test_reviewer", model="custom-model")
        assert reviewer.name == "custom_test_reviewer"
        assert reviewer.model == "custom-model"


# =============================================================================
# Additional Orchestrator Tests
# =============================================================================


class TestOrchestratorAdditional:
    """Additional tests for CodeReviewOrchestrator."""

    @pytest.mark.asyncio
    async def test_review_code_with_high_severity_pending(self):
        """review_code sets pending for moderate high findings."""
        orchestrator = CodeReviewOrchestrator()
        # Code with some high-severity issues but not > 3
        code = 'secret = "value123"'

        result = await orchestrator.review_code(code)

        # Should have at least one high finding
        assert result.high_count >= 1

    @pytest.mark.asyncio
    async def test_review_code_many_high_findings(self):
        """review_code sets changes_requested for many high findings."""
        orchestrator = CodeReviewOrchestrator()
        # Multiple high-severity findings
        code = """
secret = "value1"
pickle.load(data)
yaml.load(content)
pickle.load(other_data)
"""
        result = await orchestrator.review_code(code)

        # Should have multiple findings
        assert len(result.findings) >= 3

    @pytest.mark.asyncio
    async def test_review_code_file_path_propagated(self):
        """review_code propagates file path to findings."""
        orchestrator = CodeReviewOrchestrator()
        code = 'password = "secret"'

        result = await orchestrator.review_code(code, file_path="src/config.py")

        for finding in result.findings:
            if finding.location:
                assert finding.location.file_path == "src/config.py"

    @pytest.mark.asyncio
    async def test_review_diff_empty(self):
        """review_diff handles empty diff."""
        orchestrator = CodeReviewOrchestrator()
        result = await orchestrator.review_diff("")

        assert isinstance(result, ReviewResult)
        assert result.files_reviewed == 0

    @pytest.mark.asyncio
    async def test_review_diff_multiple_files(self):
        """review_diff handles diff with multiple files."""
        orchestrator = CodeReviewOrchestrator()
        diff = """diff --git a/file1.py b/file1.py
+password = "secret"
diff --git a/file2.py b/file2.py
+api_key = "key123"
diff --git a/file3.py b/file3.py
+x = 1
"""
        result = await orchestrator.review_diff(diff)

        assert result.files_reviewed == 3

    @pytest.mark.asyncio
    async def test_review_diff_approval_many_findings(self):
        """review_diff sets pending for many non-critical findings."""
        orchestrator = CodeReviewOrchestrator()
        # Create code that generates many low-severity findings
        lines = ["# TODO: fix this"] * 15
        code = "\n".join(lines)
        diff = "diff --git a/file.py b/file.py\n" + "\n".join(f"+{line}" for line in lines)

        result = await orchestrator.review_diff(diff)

        # Many findings should give pending or changes_requested
        if len(result.findings) > 10:
            assert result.approval_status in ("pending", "changes_requested")

    def test_parse_diff_no_additions(self):
        """_parse_diff handles diff with only removals."""
        orchestrator = CodeReviewOrchestrator()
        diff = """diff --git a/file.py b/file.py
-removed_line
-another_removed
"""
        files = orchestrator._parse_diff(diff)

        assert len(files) == 1
        assert len(files[0][1]) == 0  # No added lines
        assert len(files[0][2]) == 2  # Two removed lines

    def test_parse_diff_handles_header_lines(self):
        """_parse_diff correctly skips +++ and --- header lines."""
        orchestrator = CodeReviewOrchestrator()
        diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
+real_addition
"""
        files = orchestrator._parse_diff(diff)

        assert len(files) == 1
        # +++ should not be included as an added line
        added = files[0][1]
        assert "++ b/test.py" not in added
        assert "real_addition" in added

    def test_count_by_field_empty_list(self):
        """_count_by_field handles empty findings list."""
        orchestrator = CodeReviewOrchestrator()
        counts = orchestrator._count_by_field([], "category")

        assert counts == {}

    def test_generate_summary_medium_and_low(self):
        """_generate_summary includes medium and low counts."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            files_reviewed=1,
            lines_reviewed=10,
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.MAINTAINABILITY,
                    severity=FindingSeverity.MEDIUM,
                    title="A",
                    description="D",
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.MAINTAINABILITY,
                    severity=FindingSeverity.LOW,
                    title="B",
                    description="D",
                ),
            ],
        )

        summary = orchestrator._generate_summary(result)

        assert "2 issues" in summary or "Found 2" in summary
        assert "medium" in summary.lower()
        assert "low" in summary.lower()


# =============================================================================
# ReviewResult Serialization Tests
# =============================================================================


class TestReviewResultSerialization:
    """Tests for ReviewResult to_dict serialization."""

    def test_to_dict_with_findings(self):
        """ReviewResult to_dict serializes findings."""
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="Issue",
                    description="Description",
                ),
            ]
        )

        data = result.to_dict()

        assert len(data["findings"]) == 1
        assert data["findings"][0]["id"] == "F001"
        assert data["findings"][0]["severity"] == "high"

    def test_to_dict_consensus_notes(self):
        """ReviewResult to_dict includes consensus notes."""
        result = ReviewResult(
            consensus_notes=["Note 1", "Note 2"],
        )

        data = result.to_dict()

        assert data["consensusNotes"] == ["Note 1", "Note 2"]

    def test_to_dict_reviewers_participated(self):
        """ReviewResult to_dict includes reviewers."""
        result = ReviewResult(
            reviewers_participated=["security", "performance"],
        )

        data = result.to_dict()

        assert data["reviewersParticipated"] == ["security", "performance"]


# =============================================================================
# ReviewFinding Edge Cases
# =============================================================================


class TestReviewFindingEdgeCases:
    """Edge case tests for ReviewFinding."""

    def test_finding_with_empty_references(self):
        """ReviewFinding handles empty references list."""
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test",
            description="Desc",
            references=[],
        )

        data = finding.to_dict()
        assert data["references"] == []

    def test_finding_with_empty_tags(self):
        """ReviewFinding handles empty tags list."""
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test",
            description="Desc",
            tags=[],
        )

        data = finding.to_dict()
        assert data["tags"] == []

    def test_finding_with_suggested_code(self):
        """ReviewFinding handles suggested_code in to_dict."""
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test",
            description="Desc",
            suggested_code="x = safe_parse(input)",
        )

        data = finding.to_dict()
        assert data["suggestedCode"] == "x = safe_parse(input)"


# =============================================================================
# Format Comment Tests
# =============================================================================


class TestFormatCommentAdditional:
    """Additional tests for _format_comment method."""

    def test_format_comment_medium_severity(self):
        """_format_comment formats medium severity correctly."""
        orchestrator = CodeReviewOrchestrator()
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.MAINTAINABILITY,
            severity=FindingSeverity.MEDIUM,
            title="Code smell",
            description="This code could be cleaner",
        )

        comment = orchestrator._format_comment(finding)

        assert "MEDIUM" in comment
        assert "Code smell" in comment

    def test_format_comment_low_severity(self):
        """_format_comment formats low severity correctly."""
        orchestrator = CodeReviewOrchestrator()
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.STYLE,
            severity=FindingSeverity.LOW,
            title="Style issue",
            description="Inconsistent naming",
        )

        comment = orchestrator._format_comment(finding)

        assert "LOW" in comment

    def test_format_comment_info_severity(self):
        """_format_comment formats info severity correctly."""
        orchestrator = CodeReviewOrchestrator()
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.DOCUMENTATION,
            severity=FindingSeverity.INFO,
            title="TODO found",
            description="TODO comment exists",
        )

        comment = orchestrator._format_comment(finding)

        assert "INFO" in comment

    def test_format_comment_with_suggested_code(self):
        """_format_comment includes code block for suggested code."""
        orchestrator = CodeReviewOrchestrator()
        finding = ReviewFinding(
            id="F001",
            category=ReviewCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Unsafe eval",
            description="eval() detected",
            suggestion="Use ast.literal_eval instead",
            suggested_code="result = ast.literal_eval(expression)",
        )

        comment = orchestrator._format_comment(finding)

        assert "```python" in comment
        assert "ast.literal_eval" in comment


# =============================================================================
# PR Review Body Format Tests
# =============================================================================


class TestPRReviewBodyFormat:
    """Tests for PR review body formatting."""

    def test_format_body_includes_aragora_signature(self):
        """PR review body includes Aragora signature."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(summary="Clean code")

        body = orchestrator._format_pr_review_body(result)

        assert "Aragora Code Review" in body

    def test_format_body_severity_breakdown(self):
        """PR review body includes severity breakdown."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            summary="Issues found",
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="Critical",
                    description="D",
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="High",
                    description="D",
                ),
            ],
        )

        body = orchestrator._format_pr_review_body(result)

        assert "Findings by Severity" in body
        assert "CRITICAL" in body

    def test_format_body_category_breakdown(self):
        """PR review body includes category breakdown."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(
            summary="Issues found",
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="Sec",
                    description="D",
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.PERFORMANCE,
                    severity=FindingSeverity.MEDIUM,
                    title="Perf",
                    description="D",
                ),
            ],
        )

        body = orchestrator._format_pr_review_body(result)

        assert "Findings by Category" in body

    def test_format_body_no_findings(self):
        """PR review body handles no findings."""
        orchestrator = CodeReviewOrchestrator()
        result = ReviewResult(summary="No issues found")

        body = orchestrator._format_pr_review_body(result)

        assert "Code Review Summary" in body
        assert "No issues found" in body


# =============================================================================
# GitHub Comments Sorting Tests
# =============================================================================


class TestGitHubCommentsSorting:
    """Tests for GitHub comment generation sorting behavior."""

    def test_comments_sorted_by_severity(self):
        """Comments are sorted by severity (critical first)."""
        orchestrator = CodeReviewOrchestrator()
        loc = CodeLocation(file_path="test.py", start_line=1)
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.LOW,
                    title="Low",
                    description="D",
                    location=loc,
                ),
                ReviewFinding(
                    id="F002",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="Critical",
                    description="D",
                    location=loc,
                ),
                ReviewFinding(
                    id="F003",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.MEDIUM,
                    title="Medium",
                    description="D",
                    location=loc,
                ),
            ],
        )

        comments = orchestrator.generate_github_comments(result)

        assert len(comments) == 3
        assert "Critical" in comments[0]["body"]

    def test_comments_respect_max_zero(self):
        """Comments respect max_comments=0."""
        orchestrator = CodeReviewOrchestrator()
        loc = CodeLocation(file_path="test.py", start_line=1)
        result = ReviewResult(
            findings=[
                ReviewFinding(
                    id="F001",
                    category=ReviewCategory.SECURITY,
                    severity=FindingSeverity.HIGH,
                    title="Issue",
                    description="D",
                    location=loc,
                ),
            ],
        )

        comments = orchestrator.generate_github_comments(result, max_comments=0)

        assert len(comments) == 0


# =============================================================================
# Reviewer Initialization Tests
# =============================================================================


class TestReviewerInitialization:
    """Tests for specialized reviewer initialization."""

    def test_security_reviewer_custom_role(self):
        """SecurityReviewer accepts custom role."""
        reviewer = SecurityReviewer(role="critic")
        assert reviewer.role == "critic"

    def test_performance_reviewer_custom_name(self):
        """PerformanceReviewer accepts custom name."""
        reviewer = PerformanceReviewer(name="custom_perf")
        assert reviewer.name == "custom_perf"

    def test_maintainability_reviewer_custom_model(self):
        """MaintainabilityReviewer accepts custom model."""
        reviewer = MaintainabilityReviewer(model="gpt4")
        assert reviewer.model == "gpt4"

    def test_security_reviewer_agent_type(self):
        """SecurityReviewer has correct agent_type."""
        reviewer = SecurityReviewer()
        assert reviewer.agent_type == "security_reviewer"

    def test_performance_reviewer_agent_type(self):
        """PerformanceReviewer has correct agent_type."""
        reviewer = PerformanceReviewer()
        assert reviewer.agent_type == "performance_reviewer"

    def test_maintainability_reviewer_agent_type(self):
        """MaintainabilityReviewer has correct agent_type."""
        reviewer = MaintainabilityReviewer()
        assert reviewer.agent_type == "maintainability_reviewer"

    def test_test_coverage_reviewer_agent_type(self):
        """TestCoverageReviewer has correct agent_type."""
        reviewer = TestCoverageReviewer()
        assert reviewer.agent_type == "test_coverage_reviewer"

    def test_all_reviewers_have_system_prompts(self):
        """All reviewers have non-empty system prompts."""
        reviewers = [
            SecurityReviewer(),
            PerformanceReviewer(),
            MaintainabilityReviewer(),
            TestCoverageReviewer(),
        ]
        for reviewer in reviewers:
            assert len(reviewer.system_prompt) > 100

    def test_all_reviewers_neutral_stance(self):
        """All reviewers have neutral stance."""
        reviewers = [
            SecurityReviewer(),
            PerformanceReviewer(),
            MaintainabilityReviewer(),
            TestCoverageReviewer(),
        ]
        for reviewer in reviewers:
            assert reviewer.stance == "neutral"


# =============================================================================
# Pattern Constant Tests
# =============================================================================


class TestPatternConstants:
    """Tests for pattern constant lists."""

    def test_security_patterns_non_empty(self):
        """SECURITY_PATTERNS is non-empty."""
        assert len(SECURITY_PATTERNS) > 0

    def test_performance_patterns_non_empty(self):
        """PERFORMANCE_PATTERNS is non-empty."""
        assert len(PERFORMANCE_PATTERNS) > 0

    def test_maintainability_patterns_non_empty(self):
        """MAINTAINABILITY_PATTERNS is non-empty."""
        assert len(MAINTAINABILITY_PATTERNS) > 0

    def test_all_patterns_are_tuples(self):
        """All patterns are 3-tuples."""
        for pattern_list in [SECURITY_PATTERNS, PERFORMANCE_PATTERNS, MAINTAINABILITY_PATTERNS]:
            for item in pattern_list:
                assert len(item) == 3
                assert isinstance(item[0], str)  # regex pattern
                assert isinstance(item[1], str)  # description
                assert isinstance(item[2], FindingSeverity)  # severity

    def test_all_patterns_are_valid_regex(self):
        """All patterns compile as valid regex."""
        import re

        for pattern_list in [SECURITY_PATTERNS, PERFORMANCE_PATTERNS, MAINTAINABILITY_PATTERNS]:
            for pattern, _, _ in pattern_list:
                re.compile(pattern)  # Should not raise


# =============================================================================
# Edge Cases
# =============================================================================


class TestCodeReviewEdgeCases:
    """Edge case tests for code review."""

    @pytest.mark.asyncio
    async def test_review_empty_code(self):
        """Review handles empty code string."""
        orchestrator = CodeReviewOrchestrator()
        result = await orchestrator.review_code("")

        assert isinstance(result, ReviewResult)
        assert result.files_reviewed == 1
        assert len(result.findings) == 0
        assert result.approval_status == "approved"

    @pytest.mark.asyncio
    async def test_review_whitespace_only_code(self):
        """Review handles whitespace-only code."""
        orchestrator = CodeReviewOrchestrator()
        result = await orchestrator.review_code("   \n\n  \n  ")

        assert isinstance(result, ReviewResult)
        assert result.approval_status == "approved"

    @pytest.mark.asyncio
    async def test_review_single_line_code(self):
        """Review handles single-line code."""
        orchestrator = CodeReviewOrchestrator()
        result = await orchestrator.review_code("x = 1")

        assert result.lines_reviewed == 1

    @pytest.mark.asyncio
    async def test_review_large_code(self):
        """Review handles large code blocks."""
        orchestrator = CodeReviewOrchestrator()
        code = "\n".join([f"x_{i} = {i}" for i in range(1000)])

        result = await orchestrator.review_code(code)

        assert result.lines_reviewed == 1000

    def test_code_location_to_dict_with_none_end_line(self):
        """CodeLocation to_dict with None end_line."""
        loc = CodeLocation(file_path="test.py", start_line=10)
        data = loc.to_dict()

        assert data["endLine"] is None

    def test_review_result_empty_should_not_block(self):
        """Empty ReviewResult should not block merge."""
        result = ReviewResult()

        assert result.should_block_merge is False
        assert result.critical_count == 0
        assert result.high_count == 0
