"""Tests for the autonomous code reviewer agent."""

from __future__ import annotations

import os
import tempfile
import textwrap

import pytest

from aragora.nomic.code_reviewer import (
    CodeReviewerAgent,
    IssueSeverity,
    ReviewConfig,
    ReviewIssue,
    ReviewResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reviewer() -> CodeReviewerAgent:
    return CodeReviewerAgent()


@pytest.fixture
def strict_reviewer() -> CodeReviewerAgent:
    return CodeReviewerAgent(
        ReviewConfig(
            min_score=0.9,
            max_complexity=5,
            max_function_length=20,
            max_nesting_depth=2,
        )
    )


def _write_temp(content: str, suffix: str = ".py") -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, textwrap.dedent(content).encode())
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Clean code
# ---------------------------------------------------------------------------


class TestCleanCode:
    def test_clean_code_scores_perfect(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            import logging

            logger = logging.getLogger(__name__)

            def add(a: int, b: int) -> int:
                return a + b
        """)
        try:
            result = reviewer.review_files([path])
            assert result.score == 1.0
            assert result.approved is True
            assert result.issues == []
            assert "No issues" in result.summary
        finally:
            os.unlink(path)

    def test_empty_file_scores_perfect(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("")
        try:
            result = reviewer.review_files([path])
            assert result.score == 1.0
            assert result.approved is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Pattern checks
# ---------------------------------------------------------------------------


class TestPatternChecks:
    def test_bare_except_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def risky():
                try:
                    do_something()
                except:
                    pass
        """)
        try:
            result = reviewer.review_files([path])
            pattern_issues = [i for i in result.issues if i.category == "pattern"]
            assert len(pattern_issues) >= 1
            assert any("Bare except" in i.description for i in pattern_issues)
            assert result.score < 1.0
        finally:
            os.unlink(path)

    def test_exception_pass_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def bad():
                try:
                    risky()
                except Exception:
                    pass
        """)
        try:
            result = reviewer.review_files([path])
            pattern_issues = [i for i in result.issues if i.category == "pattern"]
            assert any("Silent exception" in i.description for i in pattern_issues)
        finally:
            os.unlink(path)

    def test_print_flagged_as_info(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def debug():
                print("hello")
        """)
        try:
            result = reviewer.review_files([path])
            info_issues = [i for i in result.issues if i.severity == IssueSeverity.INFO]
            assert len(info_issues) >= 1
            assert any("print()" in i.description for i in info_issues)
        finally:
            os.unlink(path)

    def test_handler_missing_handle_errors(self, reviewer: CodeReviewerAgent) -> None:
        # Simulate a file in handlers/ path
        fd, path = tempfile.mkstemp(suffix=".py", dir=None)
        # We need "handlers/" in the path for the check to trigger
        content = textwrap.dedent("""\
            async def do_POST(request):
                return {"status": "ok"}
        """)
        os.write(fd, content.encode())
        os.close(fd)
        try:
            # Pass a fake filepath that contains "handlers/"
            result = reviewer.review_files([], goal="")
            # Use review_diff instead since we can control the "filepath"
            # Or we can test the internal method directly
            issues = reviewer._check_handler_decorators(content, "aragora/server/handlers/test.py")
            assert len(issues) >= 1
            assert any("@handle_errors" in i.description for i in issues)
        finally:
            os.unlink(path)

    def test_handler_with_handle_errors_ok(self, reviewer: CodeReviewerAgent) -> None:
        content = textwrap.dedent("""\
            @handle_errors
            async def do_POST(request):
                return {"status": "ok"}
        """)
        issues = reviewer._check_handler_decorators(content, "aragora/server/handlers/test.py")
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# Security checks
# ---------------------------------------------------------------------------


class TestSecurityChecks:
    def test_eval_flagged_critical(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def dangerous(user_input):
                return eval(user_input)
        """)
        try:
            result = reviewer.review_files([path])
            critical = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL]
            assert len(critical) >= 1
            assert any("eval()" in i.description for i in critical)
        finally:
            os.unlink(path)

    def test_exec_flagged_critical(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def run_code(code):
                exec(code)
        """)
        try:
            result = reviewer.review_files([path])
            critical = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL]
            assert any("exec()" in i.description for i in critical)
        finally:
            os.unlink(path)

    def test_shell_true_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            import subprocess
            def run(cmd):
                subprocess.run(cmd, shell=True)
        """)
        try:
            result = reviewer.review_files([path])
            critical = [i for i in result.issues if i.severity == IssueSeverity.CRITICAL]
            assert any("shell=True" in i.description for i in critical)
        finally:
            os.unlink(path)

    def test_hardcoded_secret_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            API_KEY = "sk-1234567890abcdef"
        """)
        try:
            result = reviewer.review_files([path])
            security_issues = [i for i in result.issues if i.category == "security"]
            assert any(
                "secret" in i.description.lower() or "hardcoded" in i.description.lower()
                for i in security_issues
            )
        finally:
            os.unlink(path)

    def test_sql_injection_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def query(user_id):
                sql = f"SELECT * FROM users WHERE id = {user_id}"
                return sql
        """)
        try:
            result = reviewer.review_files([path])
            security_issues = [i for i in result.issues if i.category == "security"]
            assert any("SQL" in i.description for i in security_issues)
        finally:
            os.unlink(path)

    def test_str_e_in_response_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def handle(request):
                try:
                    do_work()
                except Exception as e:
                    return {"error": str(e)}
        """)
        try:
            result = reviewer.review_files([path])
            security_issues = [i for i in result.issues if i.category == "security"]
            assert any("str(e)" in i.description for i in security_issues)
        finally:
            os.unlink(path)

    def test_str_e_in_logging_not_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            import logging
            logger = logging.getLogger(__name__)
            def handle(request):
                try:
                    do_work()
                except Exception as e:
                    logger.warning("Failed: %s", e)
        """)
        try:
            result = reviewer.review_files([path])
            security_issues = [
                i for i in result.issues if i.category == "security" and "str(e)" in i.description
            ]
            assert len(security_issues) == 0
        finally:
            os.unlink(path)

    def test_pickle_loads_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            import pickle
            def load(data):
                return pickle.loads(data)
        """)
        try:
            result = reviewer.review_files([path])
            security_issues = [i for i in result.issues if i.category == "security"]
            assert any(
                "Pickle" in i.description or "pickle" in i.description for i in security_issues
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Complexity checks
# ---------------------------------------------------------------------------


class TestComplexityChecks:
    def test_high_complexity_detected(self, strict_reviewer: CodeReviewerAgent) -> None:
        # Generate a function with many branches
        path = _write_temp("""\
            def complex_func(a, b, c, d, e, f):
                if a:
                    if b:
                        if c:
                            pass
                if d:
                    pass
                if e:
                    pass
                if f:
                    pass
        """)
        try:
            result = strict_reviewer.review_files([path])
            complexity_issues = [
                i
                for i in result.issues
                if i.category == "complexity" and "cyclomatic" in i.description.lower()
            ]
            assert len(complexity_issues) >= 1
        finally:
            os.unlink(path)

    def test_long_function_detected(self, strict_reviewer: CodeReviewerAgent) -> None:
        # Generate a function exceeding 20 lines
        lines = ["def long_func():"]
        for i in range(25):
            lines.append(f"    x_{i} = {i}")
        path = _write_temp("\n".join(lines) + "\n")
        try:
            result = strict_reviewer.review_files([path])
            length_issues = [
                i
                for i in result.issues
                if i.category == "complexity" and "lines long" in i.description
            ]
            assert len(length_issues) >= 1
        finally:
            os.unlink(path)

    def test_deep_nesting_detected(self, strict_reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def deep():
                if True:
                    for x in range(10):
                        if x > 5:
                            while x > 0:
                                x -= 1
        """)
        try:
            result = strict_reviewer.review_files([path])
            nesting_issues = [
                i
                for i in result.issues
                if i.category == "complexity" and "nesting" in i.description.lower()
            ]
            assert len(nesting_issues) >= 1
        finally:
            os.unlink(path)

    def test_simple_function_passes(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def simple(x):
                if x > 0:
                    return x
                return -x
        """)
        try:
            result = reviewer.review_files([path])
            complexity_issues = [i for i in result.issues if i.category == "complexity"]
            assert len(complexity_issues) == 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Architecture checks
# ---------------------------------------------------------------------------


class TestArchitectureChecks:
    def test_adapter_importing_handler_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            from aragora.server.handlers.base import BaseHandler

            class MyAdapter:
                pass
        """)
        try:
            # Must use a filepath containing "adapters/"
            issues = reviewer._check_architecture(
                "aragora/knowledge/mound/adapters/my_adapter.py",
                __import__("ast").parse(open(path).read()),
            )
            assert len(issues) >= 1
            assert any("architecture" in i.category for i in issues)
        finally:
            os.unlink(path)

    def test_normal_import_not_flagged(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            from aragora.knowledge.bridges import KnowledgeBridgeHub
        """)
        try:
            import ast as _ast

            issues = reviewer._check_architecture(
                "aragora/knowledge/mound/adapters/my_adapter.py",
                _ast.parse(open(path).read()),
            )
            assert len(issues) == 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_starts_at_one(self, reviewer: CodeReviewerAgent) -> None:
        assert reviewer._calculate_score([]) == 1.0

    def test_critical_deducts_030(self, reviewer: CodeReviewerAgent) -> None:
        issues = [
            ReviewIssue(
                severity=IssueSeverity.CRITICAL,
                category="security",
                file="test.py",
                line=1,
                description="test",
            )
        ]
        assert reviewer._calculate_score(issues) == 0.7

    def test_error_deducts_015(self, reviewer: CodeReviewerAgent) -> None:
        issues = [
            ReviewIssue(
                severity=IssueSeverity.ERROR,
                category="pattern",
                file="test.py",
                line=1,
                description="test",
            )
        ]
        assert reviewer._calculate_score(issues) == 0.85

    def test_warning_deducts_005(self, reviewer: CodeReviewerAgent) -> None:
        issues = [
            ReviewIssue(
                severity=IssueSeverity.WARNING,
                category="complexity",
                file="test.py",
                line=1,
                description="test",
            )
        ]
        assert reviewer._calculate_score(issues) == 0.95

    def test_info_deducts_001(self, reviewer: CodeReviewerAgent) -> None:
        issues = [
            ReviewIssue(
                severity=IssueSeverity.INFO,
                category="pattern",
                file="test.py",
                line=1,
                description="test",
            )
        ]
        assert reviewer._calculate_score(issues) == 0.99

    def test_score_floors_at_zero(self, reviewer: CodeReviewerAgent) -> None:
        issues = [
            ReviewIssue(
                severity=IssueSeverity.CRITICAL,
                category="security",
                file="test.py",
                line=1,
                description=f"issue {i}",
            )
            for i in range(10)
        ]
        assert reviewer._calculate_score(issues) == 0.0

    def test_approval_threshold(self, reviewer: CodeReviewerAgent) -> None:
        """Score >= min_score means approved."""
        path = _write_temp("""\
            def clean():
                return 42
        """)
        try:
            result = reviewer.review_files([path])
            assert result.approved is True
        finally:
            os.unlink(path)

    def test_below_threshold_rejected(self) -> None:
        """Code with critical issues below threshold is rejected."""
        reviewer = CodeReviewerAgent(ReviewConfig(min_score=0.9))
        path = _write_temp("""\
            def bad(x):
                return eval(x)
        """)
        try:
            result = reviewer.review_files([path])
            assert result.approved is False
            assert result.score < 0.9
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Diff review
# ---------------------------------------------------------------------------


class TestDiffReview:
    def test_review_diff_detects_issues(self, reviewer: CodeReviewerAgent) -> None:
        diff = textwrap.dedent("""\
            diff --git a/module.py b/module.py
            --- a/module.py
            +++ b/module.py
            @@ -1,3 +1,5 @@
             def process():
            -    pass
            +    try:
            +        risky()
            +    except:
            +        pass
        """)
        result = reviewer.review_diff(diff)
        assert len(result.issues) >= 1
        assert any("Bare except" in i.description for i in result.issues)

    def test_review_diff_clean(self, reviewer: CodeReviewerAgent) -> None:
        diff = textwrap.dedent("""\
            diff --git a/module.py b/module.py
            --- a/module.py
            +++ b/module.py
            @@ -1,3 +1,5 @@
             def process():
            -    pass
            +    return 42
        """)
        result = reviewer.review_diff(diff)
        assert result.score == 1.0

    def test_review_diff_with_goal(self, reviewer: CodeReviewerAgent) -> None:
        diff = "+def simple(): return 1"
        result = reviewer.review_diff(diff, goal="Improve error handling")
        assert "Improve error handling" in result.summary or result.summary.startswith("No issues")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_syntax_error_creates_issue(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("def bad(:\n    pass\n")
        try:
            result = reviewer.review_files([path])
            syntax_issues = [i for i in result.issues if i.category == "syntax"]
            assert len(syntax_issues) == 1
            assert syntax_issues[0].severity == IssueSeverity.ERROR
        finally:
            os.unlink(path)

    def test_missing_file_creates_issue(self, reviewer: CodeReviewerAgent) -> None:
        result = reviewer.review_files(["/nonexistent/file.py"])
        io_issues = [i for i in result.issues if i.category == "io"]
        assert len(io_issues) == 1
        assert io_issues[0].severity == IssueSeverity.ERROR

    def test_multiple_files_reviewed(self, reviewer: CodeReviewerAgent) -> None:
        path1 = _write_temp("def a(): return 1\n")
        path2 = _write_temp("def b(): return 2\n")
        try:
            result = reviewer.review_files([path1, path2])
            assert result.metrics.get("files_reviewed") == 2
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ---------------------------------------------------------------------------
# Summary and recommendations
# ---------------------------------------------------------------------------


class TestSummaryAndRecommendations:
    def test_summary_includes_counts(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def bad(x):
                return eval(x)
        """)
        try:
            result = reviewer.review_files([path])
            assert "critical" in result.summary.lower() or "issue" in result.summary.lower()
        finally:
            os.unlink(path)

    def test_recommendations_per_category(self, reviewer: CodeReviewerAgent) -> None:
        # Code with both security and pattern issues
        path = _write_temp("""\
            def bad(x):
                try:
                    return eval(x)
                except:
                    pass
        """)
        try:
            result = reviewer.review_files([path])
            categories = {r.split(":")[0] for r in result.recommendations}
            assert "Security" in categories or "Patterns" in categories
            assert len(result.recommendations) >= 1
        finally:
            os.unlink(path)

    def test_no_issues_means_no_recommendations(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("def ok(): return 1\n")
        try:
            result = reviewer.review_files([path])
            assert result.recommendations == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_disable_all_checks(self) -> None:
        config = ReviewConfig(
            check_patterns=False,
            check_security=False,
            check_complexity=False,
            check_architecture=False,
        )
        reviewer = CodeReviewerAgent(config)
        path = _write_temp("""\
            def bad(x):
                return eval(x)
        """)
        try:
            result = reviewer.review_files([path])
            assert result.score == 1.0
            assert result.issues == []
        finally:
            os.unlink(path)

    def test_custom_thresholds(self) -> None:
        config = ReviewConfig(min_score=0.5)
        reviewer = CodeReviewerAgent(config)
        # eval() is critical (-0.3), so score=0.7 which is >= 0.5
        path = _write_temp("def f(x): return eval(x)\n")
        try:
            result = reviewer.review_files([path])
            assert result.approved is True  # 0.7 >= 0.5
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_include_function_count(self, reviewer: CodeReviewerAgent) -> None:
        path = _write_temp("""\
            def a(): pass
            def b(): pass
            def c(): pass
        """)
        try:
            result = reviewer.review_files([path])
            assert result.metrics.get("total_functions") == 3
            assert result.metrics.get("files_reviewed") == 1
        finally:
            os.unlink(path)
