"""
Error Taxonomy for Nomic Loop Learning.

Provides structured error categorization to enable pattern-based learning
from failures. Errors are classified by:
- Phase (context, debate, design, implement, verify, commit)
- Type (timeout, crash, validation, resource, external)
- Severity (recoverable, critical, fatal)
- Root cause patterns

Usage:
    from scripts.nomic.error_taxonomy import ErrorCategory, classify_error, ErrorPattern

    # Classify an error
    category = classify_error(exception, phase="verify")
    print(f"Category: {category.type}, Recoverable: {category.recoverable}")

    # Record for learning
    pattern = ErrorPattern.from_exception(exception, phase, context)
    pattern_store.record(pattern)
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ErrorType(Enum):
    """Primary error classification."""

    TIMEOUT = "timeout"  # Phase exceeded time limit
    CRASH = "crash"  # Unexpected exception
    VALIDATION = "validation"  # Output didn't meet requirements
    RESOURCE = "resource"  # Out of memory, disk, etc.
    EXTERNAL = "external"  # API failures, network issues
    SYNTAX = "syntax"  # Generated code has syntax errors
    TEST_FAILURE = "test_failure"  # Tests failed
    TYPE_ERROR = "type_error"  # Type checking failures
    IMPORT_ERROR = "import_error"  # Module import failures
    PERMISSION = "permission"  # File/git permission issues
    CONFLICT = "conflict"  # Git merge conflicts
    UNKNOWN = "unknown"


class Severity(Enum):
    """Error severity level."""

    RECOVERABLE = "recoverable"  # Can retry or skip
    CRITICAL = "critical"  # Requires rollback
    FATAL = "fatal"  # Cannot continue


@dataclass
class ErrorCategory:
    """Categorized error information."""

    type: ErrorType
    severity: Severity
    phase: str
    message: str
    recoverable: bool
    suggested_action: str
    root_cause_pattern: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "phase": self.phase,
            "message": self.message,
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action,
            "root_cause_pattern": self.root_cause_pattern,
        }


@dataclass
class ErrorPattern:
    """
    Structured error pattern for learning.

    Stores enough context to identify recurring issues
    and suggest mitigations.
    """

    error_type: ErrorType
    phase: str
    message: str
    stack_trace: str
    timestamp: datetime
    cycle_number: int = 0
    fix_iteration: int = 0
    context: dict = field(default_factory=dict)

    # Pattern matching fields
    file_involved: Optional[str] = None
    function_involved: Optional[str] = None
    test_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.value,
            "phase": self.phase,
            "message": self.message,
            "stack_trace": self.stack_trace[:500],  # Truncate for storage
            "timestamp": self.timestamp.isoformat(),
            "cycle_number": self.cycle_number,
            "fix_iteration": self.fix_iteration,
            "file_involved": self.file_involved,
            "function_involved": self.function_involved,
            "test_name": self.test_name,
            "context": self.context,
        }

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        phase: str,
        cycle_number: int = 0,
        fix_iteration: int = 0,
        context: dict | None = None,
    ) -> "ErrorPattern":
        """Create pattern from an exception."""
        error_type = _detect_error_type(error)
        stack = traceback.format_exception(type(error), error, error.__traceback__)
        stack_trace = "".join(stack)

        # Extract file/function from traceback
        file_involved = None
        function_involved = None
        for line in reversed(stack):
            match = re.search(r'File "([^"]+)", line \d+, in (\w+)', line)
            if match:
                file_involved = match.group(1)
                function_involved = match.group(2)
                break

        return cls(
            error_type=error_type,
            phase=phase,
            message=str(error)[:500],
            stack_trace=stack_trace,
            timestamp=datetime.now(),
            cycle_number=cycle_number,
            fix_iteration=fix_iteration,
            file_involved=file_involved,
            function_involved=function_involved,
            context=context or {},
        )


# Error detection patterns
_TIMEOUT_PATTERNS = [
    r"timed?\s*out",
    r"deadline\s*exceeded",
    r"asyncio\.TimeoutError",
    r"took too long",
]

_SYNTAX_PATTERNS = [
    r"SyntaxError",
    r"IndentationError",
    r"invalid syntax",
    r"unexpected token",
]

_TYPE_PATTERNS = [
    r"TypeError",
    r"type.*mismatch",
    r"incompatible type",
    r"mypy.*error",
]

_IMPORT_PATTERNS = [
    r"ImportError",
    r"ModuleNotFoundError",
    r"cannot import",
    r"No module named",
]

_RESOURCE_PATTERNS = [
    r"MemoryError",
    r"out of memory",
    r"disk.*full",
    r"no space left",
]

_TEST_PATTERNS = [
    r"AssertionError",
    r"FAILED",
    r"test.*failed",
    r"pytest.*error",
]

_EXTERNAL_PATTERNS = [
    r"ConnectionError",
    r"HTTPError",
    r"API.*error",
    r"rate.*limit",
    r"quota.*exceeded",
]

_PERMISSION_PATTERNS = [
    r"PermissionError",
    r"Permission denied",
    r"access.*denied",
]


def _detect_error_type(error: Exception) -> ErrorType:
    """Detect error type from exception."""
    error_str = f"{type(error).__name__}: {str(error)}"

    # Check patterns in order of specificity
    pattern_checks = [
        (_TIMEOUT_PATTERNS, ErrorType.TIMEOUT),
        (_SYNTAX_PATTERNS, ErrorType.SYNTAX),
        (_TYPE_PATTERNS, ErrorType.TYPE_ERROR),
        (_IMPORT_PATTERNS, ErrorType.IMPORT_ERROR),
        (_RESOURCE_PATTERNS, ErrorType.RESOURCE),
        (_TEST_PATTERNS, ErrorType.TEST_FAILURE),
        (_EXTERNAL_PATTERNS, ErrorType.EXTERNAL),
        (_PERMISSION_PATTERNS, ErrorType.PERMISSION),
    ]

    for patterns, error_type in pattern_checks:
        for pattern in patterns:
            if re.search(pattern, error_str, re.IGNORECASE):
                return error_type

    return ErrorType.CRASH if isinstance(error, Exception) else ErrorType.UNKNOWN


def _get_severity(error_type: ErrorType, phase: str) -> Severity:
    """Determine severity based on error type and phase."""
    # Fatal errors
    if error_type == ErrorType.RESOURCE:
        return Severity.FATAL

    # Critical errors by phase
    critical_phases = {"implement", "verify"}
    if phase in critical_phases and error_type in {
        ErrorType.SYNTAX,
        ErrorType.CRASH,
    }:
        return Severity.CRITICAL

    # Most errors are recoverable
    return Severity.RECOVERABLE


def _suggest_action(error_type: ErrorType, phase: str) -> str:
    """Suggest recovery action based on error type."""
    suggestions = {
        ErrorType.TIMEOUT: "Increase timeout or simplify task",
        ErrorType.SYNTAX: "Review generated code for syntax issues",
        ErrorType.TYPE_ERROR: "Check type annotations and fix mismatches",
        ErrorType.IMPORT_ERROR: "Verify module paths and dependencies",
        ErrorType.RESOURCE: "Free resources or reduce batch size",
        ErrorType.TEST_FAILURE: "Review test output and fix failing assertions",
        ErrorType.EXTERNAL: "Retry with backoff or use fallback",
        ErrorType.PERMISSION: "Check file permissions and git state",
        ErrorType.CONFLICT: "Resolve git conflicts manually",
        ErrorType.CRASH: "Review stack trace for root cause",
        ErrorType.VALIDATION: "Check output format requirements",
    }
    return suggestions.get(error_type, "Investigate error details")


def classify_error(
    error: Exception,
    phase: str,
    context: dict | None = None,
) -> ErrorCategory:
    """
    Classify an error into a structured category.

    Args:
        error: The exception to classify
        phase: Which nomic loop phase it occurred in
        context: Additional context about the error

    Returns:
        ErrorCategory with type, severity, and recovery suggestions
    """
    error_type = _detect_error_type(error)
    severity = _get_severity(error_type, phase)
    suggested_action = _suggest_action(error_type, phase)

    # Determine if recoverable
    recoverable = severity != Severity.FATAL

    # Extract root cause pattern
    root_cause = None
    error_msg = str(error)
    if error_type == ErrorType.TEST_FAILURE:
        # Extract test name from error
        match = re.search(r"(test_\w+)", error_msg)
        if match:
            root_cause = f"failing_test:{match.group(1)}"
    elif error_type == ErrorType.SYNTAX:
        # Extract line number
        match = re.search(r"line (\d+)", error_msg)
        if match:
            root_cause = f"syntax_error:line_{match.group(1)}"

    return ErrorCategory(
        type=error_type,
        severity=severity,
        phase=phase,
        message=str(error)[:200],
        recoverable=recoverable,
        suggested_action=suggested_action,
        root_cause_pattern=root_cause,
    )


def extract_test_failures(test_output: str) -> list[dict]:
    """
    Extract structured test failure information from pytest output.

    Returns list of dicts with test name, file, error type.
    """
    failures = []

    # Match pytest failure patterns
    failure_pattern = r"FAILED\s+([\w/]+\.py)::([\w\[\]]+)"
    for match in re.finditer(failure_pattern, test_output):
        failures.append(
            {
                "file": match.group(1),
                "test": match.group(2),
                "type": "assertion",
            }
        )

    # Match error patterns
    error_pattern = r"ERROR\s+([\w/]+\.py)::([\w\[\]]+)"
    for match in re.finditer(error_pattern, test_output):
        failures.append(
            {
                "file": match.group(1),
                "test": match.group(2),
                "type": "error",
            }
        )

    return failures


def format_learning_summary(patterns: list[ErrorPattern]) -> str:
    """
    Format error patterns into a learning summary for agents.

    Used to inform agents about recurring issues to avoid.
    """
    if not patterns:
        return ""

    # Group by error type
    by_type: dict[ErrorType, list[ErrorPattern]] = {}
    for p in patterns:
        by_type.setdefault(p.error_type, []).append(p)

    lines = ["Recent failure patterns to avoid:"]
    for error_type, type_patterns in by_type.items():
        count = len(type_patterns)
        lines.append(f"- {error_type.value}: {count} occurrence(s)")

        # Show specific examples
        for p in type_patterns[:2]:
            if p.file_involved:
                lines.append(f"  - File: {p.file_involved}")
            if p.message:
                lines.append(f"    {p.message[:100]}")

    return "\n".join(lines)


__all__ = [
    "ErrorType",
    "Severity",
    "ErrorCategory",
    "ErrorPattern",
    "classify_error",
    "extract_test_failures",
    "format_learning_summary",
]
