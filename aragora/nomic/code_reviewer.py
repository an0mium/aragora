"""Autonomous code reviewer for Nomic Loop implementations.

Evaluates code quality beyond test pass/fail: pattern consistency,
architectural fit, security, and complexity analysis.

All analysis is AST-based and deterministic -- no LLM calls required.

Usage:
    from aragora.nomic.code_reviewer import CodeReviewerAgent, ReviewConfig

    reviewer = CodeReviewerAgent()
    result = reviewer.review_files(["aragora/some/module.py"])
    if not result.approved:
        for issue in result.issues:
            print(f"[{issue.severity.value}] {issue.file}:{issue.line}: {issue.description}")
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class IssueSeverity(Enum):
    """Severity level for review issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ReviewIssue:
    """A single issue found during code review."""

    severity: IssueSeverity
    category: str  # "pattern", "security", "architecture", "complexity"
    file: str
    line: int | None
    description: str
    suggestion: str | None = None


@dataclass
class ReviewConfig:
    """Configuration for the code reviewer."""

    min_score: float = 0.7
    check_patterns: bool = True
    check_security: bool = True
    check_architecture: bool = True
    check_complexity: bool = True
    max_complexity: int = 15
    max_function_length: int = 50
    max_nesting_depth: int = 4


@dataclass
class ReviewResult:
    """Result of a code review."""

    approved: bool
    score: float  # 0.0 - 1.0
    issues: list[ReviewIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_SECURITY_PATTERNS: list[tuple[re.Pattern[str], IssueSeverity, str, str | None]] = [
    (
        re.compile(r"\beval\s*\("),
        IssueSeverity.CRITICAL,
        "Use of eval() -- arbitrary code execution risk",
        "Replace with ast.literal_eval() or a safe parser",
    ),
    (
        re.compile(r"\bexec\s*\("),
        IssueSeverity.CRITICAL,
        "Use of exec() -- arbitrary code execution risk",
        "Refactor to avoid dynamic code execution",
    ),
    (
        re.compile(r"subprocess\.(?:run|call|Popen|check_output)\s*\([^)]*shell\s*=\s*True"),
        IssueSeverity.CRITICAL,
        "subprocess with shell=True -- command injection risk",
        "Use shell=False with a list of arguments",
    ),
    (
        re.compile(r"""(?:api[_-]?key|password|secret|token)\s*=\s*["'][^"']{8,}["']""", re.IGNORECASE),
        IssueSeverity.CRITICAL,
        "Possible hardcoded secret",
        "Use environment variables or a secrets manager",
    ),
    (
        re.compile(r"""f["'].*(?:SELECT|INSERT|UPDATE|DELETE)\b.*\{.*\}""", re.IGNORECASE),
        IssueSeverity.ERROR,
        "SQL with f-string interpolation -- SQL injection risk",
        "Use parameterized queries",
    ),
    (
        re.compile(r"""["'].*(?:SELECT|INSERT|UPDATE|DELETE)\b.*%s.*["']\s*%""", re.IGNORECASE),
        IssueSeverity.ERROR,
        "SQL with % string formatting -- SQL injection risk",
        "Use parameterized queries",
    ),
    (
        re.compile(r"\b__import__\s*\("),
        IssueSeverity.ERROR,
        "Dynamic import via __import__() -- potential security concern",
        "Use importlib.import_module() for clarity, or static imports",
    ),
    (
        re.compile(r"pickle\.loads?\s*\("),
        IssueSeverity.WARNING,
        "Pickle deserialization -- untrusted data could execute arbitrary code",
        "Use JSON or a safer serialization format for untrusted data",
    ),
]

_PATTERN_CHECKS: list[tuple[re.Pattern[str], IssueSeverity, str, str | None]] = [
    (
        re.compile(r"^\s*except\s*:", re.MULTILINE),
        IssueSeverity.ERROR,
        "Bare except clause -- catches SystemExit, KeyboardInterrupt",
        "Catch a specific exception type",
    ),
    (
        re.compile(r"except\s+Exception\s*:\s*\n\s*pass\b"),
        IssueSeverity.ERROR,
        "Silent exception swallowing (except Exception: pass)",
        "Log the exception or handle it explicitly",
    ),
    (
        re.compile(r"print\s*\("),
        IssueSeverity.INFO,
        "print() statement found -- use logging instead",
        "Replace with logger.info() or logger.debug()",
    ),
]

# Patterns for diff-mode detection (applied to added lines only)
_DIFF_ADDED_LINE = re.compile(r"^\+(?!\+\+)(.*)$", re.MULTILINE)

# Architectural dependency rules: (source_pattern, forbidden_import_pattern, description)
_ARCH_RULES: list[tuple[str, str, str]] = [
    ("adapters/", "server.handlers", "Adapters should not import from server handlers"),
    ("adapters/", "cli.", "Adapters should not import CLI modules"),
    ("knowledge/mound/adapters/", "server.", "KM adapters should not depend on server layer"),
]


# ---------------------------------------------------------------------------
# Complexity visitor
# ---------------------------------------------------------------------------


class _ComplexityVisitor(ast.NodeVisitor):
    """Counts cyclomatic complexity of functions."""

    def __init__(self) -> None:
        self.functions: list[tuple[str, int, int, int]] = []  # (name, lineno, complexity, length)
        self._current_complexity = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        saved = self._current_complexity
        self._current_complexity = 1  # base complexity
        self.generic_visit(node)
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        self.functions.append((node.name, node.lineno, self._current_complexity, length))
        self._current_complexity = saved

    def visit_If(self, node: ast.If) -> None:
        self._current_complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._current_complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._current_complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._current_complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._current_complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each `and`/`or` adds a decision point
        self._current_complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._current_complexity += 1
        self.generic_visit(node)


class _NestingVisitor(ast.NodeVisitor):
    """Measures maximum nesting depth of functions."""

    def __init__(self) -> None:
        self.functions: list[tuple[str, int, int]] = []  # (name, lineno, max_depth)
        self._depth = 0
        self._max_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        saved_depth = self._depth
        saved_max = self._max_depth
        self._depth = 0
        self._max_depth = 0
        self.generic_visit(node)
        self.functions.append((node.name, node.lineno, self._max_depth))
        self._depth = saved_depth
        self._max_depth = saved_max

    def _enter_block(self, node: ast.AST) -> None:
        self._depth += 1
        if self._depth > self._max_depth:
            self._max_depth = self._depth
        self.generic_visit(node)
        self._depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self._enter_block(node)

    def visit_For(self, node: ast.For) -> None:
        self._enter_block(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._enter_block(node)

    def visit_While(self, node: ast.While) -> None:
        self._enter_block(node)

    def visit_With(self, node: ast.With) -> None:
        self._enter_block(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._enter_block(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._enter_block(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._enter_block(node)


# ---------------------------------------------------------------------------
# Main reviewer
# ---------------------------------------------------------------------------


class CodeReviewerAgent:
    """Reviews code changes for quality, patterns, security, and architecture.

    All analysis is AST-based and deterministic. No external dependencies
    or LLM calls are required.
    """

    def __init__(self, config: ReviewConfig | None = None) -> None:
        self.config = config or ReviewConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_diff(self, diff: str, goal: str = "", context: dict[str, Any] | None = None) -> ReviewResult:
        """Review a git diff for quality issues.

        Extracts added lines from the diff and runs pattern and security
        checks against them.
        """
        issues: list[ReviewIssue] = []
        added_content = self._extract_added_lines(diff)
        current_file = "<diff>"

        # Determine file from diff headers
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]

        if self.config.check_patterns:
            issues.extend(self._check_content_patterns(added_content, current_file))
        if self.config.check_security:
            issues.extend(self._check_content_security(added_content, current_file))
        if self.config.check_complexity:
            issues.extend(self._check_content_complexity(added_content, current_file))

        score = self._calculate_score(issues)
        return ReviewResult(
            approved=score >= self.config.min_score,
            score=score,
            issues=issues,
            summary=self._generate_summary(issues, goal),
            recommendations=self._generate_recommendations(issues),
        )

    def review_files(self, files: list[str], goal: str = "") -> ReviewResult:
        """Review complete files for quality issues."""
        issues: list[ReviewIssue] = []
        metrics: dict[str, Any] = {"files_reviewed": len(files)}
        total_functions = 0

        for filepath in files:
            try:
                with open(filepath) as f:
                    content = f.read()
            except OSError as e:
                logger.warning("Cannot read file %s: %s", filepath, e)
                issues.append(ReviewIssue(
                    severity=IssueSeverity.ERROR,
                    category="io",
                    file=filepath,
                    line=None,
                    description=f"Cannot read file: {type(e).__name__}",
                ))
                continue

            try:
                tree = ast.parse(content)
            except SyntaxError:
                issues.append(ReviewIssue(
                    severity=IssueSeverity.ERROR,
                    category="syntax",
                    file=filepath,
                    line=None,
                    description="Cannot parse file: SyntaxError",
                ))
                continue

            if self.config.check_patterns:
                issues.extend(self._check_content_patterns(content, filepath))
            if self.config.check_security:
                issues.extend(self._check_content_security(content, filepath))
            if self.config.check_complexity:
                file_issues, func_count = self._check_file_complexity(filepath, tree)
                issues.extend(file_issues)
                total_functions += func_count
            if self.config.check_architecture:
                issues.extend(self._check_architecture(filepath, tree))

        metrics["total_functions"] = total_functions
        score = self._calculate_score(issues)
        return ReviewResult(
            approved=score >= self.config.min_score,
            score=score,
            issues=issues,
            summary=self._generate_summary(issues, goal),
            recommendations=self._generate_recommendations(issues),
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Pattern checks
    # ------------------------------------------------------------------

    def _check_content_patterns(self, content: str, filepath: str) -> list[ReviewIssue]:
        """Check content against known anti-patterns."""
        issues: list[ReviewIssue] = []
        for pattern, severity, desc, suggestion in _PATTERN_CHECKS:
            for match in pattern.finditer(content):
                line = content[:match.start()].count("\n") + 1
                issues.append(ReviewIssue(
                    severity=severity,
                    category="pattern",
                    file=filepath,
                    line=line,
                    description=desc,
                    suggestion=suggestion,
                ))

        # Check handler methods missing @handle_errors
        if "handlers/" in filepath and filepath.endswith(".py"):
            issues.extend(self._check_handler_decorators(content, filepath))

        # Check for str(e) in response bodies (not in logging)
        issues.extend(self._check_str_e_leaks(content, filepath))

        return issues

    def _check_handler_decorators(self, content: str, filepath: str) -> list[ReviewIssue]:
        """Check that handler write methods have @handle_errors."""
        issues: list[ReviewIssue] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if not any(name.startswith(prefix) for prefix in ("do_POST", "do_PUT", "do_PATCH", "do_DELETE", "handle_post", "handle_put", "handle_patch", "handle_delete")):
                continue
            # Check outermost decorator is handle_errors
            has_handle_errors = False
            for dec in node.decorator_list:
                dec_name = ""
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Attribute):
                    dec_name = dec.attr
                if dec_name == "handle_errors":
                    has_handle_errors = True
                    break
            if not has_handle_errors:
                issues.append(ReviewIssue(
                    severity=IssueSeverity.ERROR,
                    category="pattern",
                    file=filepath,
                    line=node.lineno,
                    description=f"Handler method '{name}' missing @handle_errors decorator",
                    suggestion="Add @handle_errors as the outermost decorator",
                ))
        return issues

    def _check_str_e_leaks(self, content: str, filepath: str) -> list[ReviewIssue]:
        """Detect str(e) in response bodies (not logging)."""
        issues: list[ReviewIssue] = []
        # Match str(e) or str(err) or str(ex) in return/response contexts
        # Exclude lines that are clearly logging
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if "str(e)" not in stripped and "str(err)" not in stripped and "str(ex)" not in stripped:
                continue
            # Skip logging lines
            if any(kw in stripped for kw in ("logger.", "logging.", "log.", ".warning(", ".error(", ".info(", ".debug(", ".exception(")):
                continue
            # Skip comments and docstrings
            if stripped.startswith("#"):
                continue
            # Flag response-context uses
            if any(kw in stripped for kw in ("return", "response", "json", "message", "body", "detail", "error")):
                issues.append(ReviewIssue(
                    severity=IssueSeverity.WARNING,
                    category="security",
                    file=filepath,
                    line=i,
                    description="str(e) in response context -- may leak internal details",
                    suggestion="Use a static error message and log the exception separately",
                ))
        return issues

    # ------------------------------------------------------------------
    # Security checks
    # ------------------------------------------------------------------

    def _check_content_security(self, content: str, filepath: str) -> list[ReviewIssue]:
        """Check content for security issues."""
        issues: list[ReviewIssue] = []
        for pattern, severity, desc, suggestion in _SECURITY_PATTERNS:
            for match in pattern.finditer(content):
                line = content[:match.start()].count("\n") + 1
                issues.append(ReviewIssue(
                    severity=severity,
                    category="security",
                    file=filepath,
                    line=line,
                    description=desc,
                    suggestion=suggestion,
                ))
        return issues

    # ------------------------------------------------------------------
    # Complexity checks
    # ------------------------------------------------------------------

    def _check_content_complexity(self, content: str, filepath: str) -> list[ReviewIssue]:
        """Parse content as Python and check complexity."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        issues, _ = self._check_file_complexity(filepath, tree)
        return issues

    def _check_file_complexity(self, filepath: str, tree: ast.Module) -> tuple[list[ReviewIssue], int]:
        """Check complexity metrics for all functions in a parsed AST."""
        issues: list[ReviewIssue] = []

        # Cyclomatic complexity and function length
        cv = _ComplexityVisitor()
        cv.visit(tree)
        func_count = len(cv.functions)

        for name, lineno, complexity, length in cv.functions:
            if complexity > self.config.max_complexity:
                issues.append(ReviewIssue(
                    severity=IssueSeverity.WARNING,
                    category="complexity",
                    file=filepath,
                    line=lineno,
                    description=f"Function '{name}' has cyclomatic complexity {complexity} (max {self.config.max_complexity})",
                    suggestion="Consider breaking this function into smaller pieces",
                ))
            if length > self.config.max_function_length:
                issues.append(ReviewIssue(
                    severity=IssueSeverity.WARNING,
                    category="complexity",
                    file=filepath,
                    line=lineno,
                    description=f"Function '{name}' is {length} lines long (max {self.config.max_function_length})",
                    suggestion="Extract helper functions to reduce length",
                ))

        # Nesting depth
        nv = _NestingVisitor()
        nv.visit(tree)
        for name, lineno, depth in nv.functions:
            if depth > self.config.max_nesting_depth:
                issues.append(ReviewIssue(
                    severity=IssueSeverity.WARNING,
                    category="complexity",
                    file=filepath,
                    line=lineno,
                    description=f"Function '{name}' has nesting depth {depth} (max {self.config.max_nesting_depth})",
                    suggestion="Use early returns or extract nested blocks into helper functions",
                ))

        return issues, func_count

    # ------------------------------------------------------------------
    # Architecture checks
    # ------------------------------------------------------------------

    def _check_architecture(self, filepath: str, tree: ast.Module) -> list[ReviewIssue]:
        """Check architectural dependency rules."""
        issues: list[ReviewIssue] = []
        imports = self._extract_imports(tree)

        for source_pattern, forbidden_pattern, desc in _ARCH_RULES:
            if source_pattern not in filepath:
                continue
            for imp_name, lineno in imports:
                if forbidden_pattern in imp_name:
                    issues.append(ReviewIssue(
                        severity=IssueSeverity.ERROR,
                        category="architecture",
                        file=filepath,
                        line=lineno,
                        description=f"{desc}: imports '{imp_name}'",
                        suggestion="Invert the dependency or use an interface/protocol",
                    ))

        return issues

    def _extract_imports(self, tree: ast.Module) -> list[tuple[str, int]]:
        """Extract all import names and their line numbers from an AST."""
        imports: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append((module, node.lineno))
        return imports

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _calculate_score(self, issues: list[ReviewIssue]) -> float:
        """Calculate review score from issues.

        Starts at 1.0 and deducts based on severity:
        - CRITICAL: -0.3
        - ERROR: -0.15
        - WARNING: -0.05
        - INFO: -0.01
        """
        score = 1.0
        penalties = {
            IssueSeverity.CRITICAL: 0.3,
            IssueSeverity.ERROR: 0.15,
            IssueSeverity.WARNING: 0.05,
            IssueSeverity.INFO: 0.01,
        }
        for issue in issues:
            score -= penalties.get(issue.severity, 0.0)
        return max(0.0, round(score, 4))

    # ------------------------------------------------------------------
    # Summary and recommendations
    # ------------------------------------------------------------------

    def _generate_summary(self, issues: list[ReviewIssue], goal: str) -> str:
        """Generate a human-readable summary of review results."""
        if not issues:
            return "No issues found. Code looks clean."

        counts: dict[str, int] = {}
        for issue in issues:
            counts[issue.severity.value] = counts.get(issue.severity.value, 0) + 1

        parts = []
        for sev in ("critical", "error", "warning", "info"):
            if sev in counts:
                parts.append(f"{counts[sev]} {sev}")

        summary = f"Found {len(issues)} issue(s): {', '.join(parts)}."
        if goal:
            summary += f" Review context: {goal}"
        return summary

    def _generate_recommendations(self, issues: list[ReviewIssue]) -> list[str]:
        """Generate actionable recommendations grouped by category."""
        recommendations: list[str] = []
        categories_seen: set[str] = set()

        for issue in issues:
            if issue.category in categories_seen:
                continue
            categories_seen.add(issue.category)

            cat_issues = [i for i in issues if i.category == issue.category]
            count = len(cat_issues)
            worst = max(cat_issues, key=lambda i: list(IssueSeverity).index(i.severity))

            if issue.category == "security":
                recommendations.append(
                    f"Security: {count} issue(s) found, worst severity: {worst.severity.value}. "
                    "Address all security issues before merging."
                )
            elif issue.category == "pattern":
                recommendations.append(
                    f"Patterns: {count} anti-pattern(s) detected. "
                    "Align with codebase conventions."
                )
            elif issue.category == "complexity":
                recommendations.append(
                    f"Complexity: {count} function(s) exceed thresholds. "
                    "Consider refactoring for maintainability."
                )
            elif issue.category == "architecture":
                recommendations.append(
                    f"Architecture: {count} dependency violation(s). "
                    "Fix import direction to maintain clean architecture."
                )
            elif issue.category == "syntax":
                recommendations.append(
                    f"Syntax: {count} file(s) have parse errors. "
                    "Fix syntax errors before review can proceed."
                )

        return recommendations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_added_lines(self, diff: str) -> str:
        """Extract only added lines from a unified diff."""
        lines = []
        for match in _DIFF_ADDED_LINE.finditer(diff):
            lines.append(match.group(1))
        return "\n".join(lines)
