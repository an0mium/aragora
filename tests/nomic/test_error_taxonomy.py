"""
Tests for Nomic Loop Error Taxonomy.

Tests cover:
- Error type detection from exceptions
- Severity classification by phase
- Pattern extraction from test output
- Learning summary formatting
"""

import pytest
from datetime import datetime

from scripts.nomic.error_taxonomy import (
    ErrorType,
    Severity,
    ErrorCategory,
    ErrorPattern,
    classify_error,
    extract_test_failures,
    format_learning_summary,
)


class TestErrorTypeDetection:
    """Tests for error type detection."""

    def test_detects_timeout_error(self):
        """Should detect timeout errors."""
        import asyncio
        error = asyncio.TimeoutError("Operation timed out")
        category = classify_error(error, phase="verify")
        assert category.type == ErrorType.TIMEOUT

    def test_detects_syntax_error(self):
        """Should detect syntax errors."""
        error = SyntaxError("invalid syntax")
        category = classify_error(error, phase="implement")
        assert category.type == ErrorType.SYNTAX

    def test_detects_type_error(self):
        """Should detect type errors."""
        error = TypeError("incompatible type for argument")
        category = classify_error(error, phase="verify")
        assert category.type == ErrorType.TYPE_ERROR

    def test_detects_import_error(self):
        """Should detect import errors."""
        error = ImportError("No module named 'nonexistent'")
        category = classify_error(error, phase="implement")
        assert category.type == ErrorType.IMPORT_ERROR

    def test_detects_memory_error(self):
        """Should detect resource errors."""
        error = MemoryError("out of memory")
        category = classify_error(error, phase="verify")
        assert category.type == ErrorType.RESOURCE

    def test_detects_assertion_as_test_failure(self):
        """Should detect assertion errors as test failures."""
        error = AssertionError("assert 1 == 2")
        category = classify_error(error, phase="verify")
        assert category.type == ErrorType.TEST_FAILURE

    def test_generic_exception_is_crash(self):
        """Generic exceptions should be classified as crash."""
        error = RuntimeError("Something went wrong")
        category = classify_error(error, phase="debate")
        assert category.type == ErrorType.CRASH


class TestSeverityClassification:
    """Tests for severity classification."""

    def test_resource_errors_are_fatal(self):
        """Resource errors should be fatal."""
        error = MemoryError("out of memory")
        category = classify_error(error, phase="any")
        assert category.severity == Severity.FATAL

    def test_syntax_in_implement_is_critical(self):
        """Syntax errors in implement phase should be critical."""
        error = SyntaxError("invalid syntax")
        category = classify_error(error, phase="implement")
        assert category.severity == Severity.CRITICAL

    def test_timeout_in_debate_is_recoverable(self):
        """Timeout in debate phase should be recoverable."""
        import asyncio
        error = asyncio.TimeoutError()
        category = classify_error(error, phase="debate")
        assert category.severity == Severity.RECOVERABLE
        assert category.recoverable is True


class TestErrorCategory:
    """Tests for ErrorCategory dataclass."""

    def test_category_has_suggested_action(self):
        """Category should include suggested action."""
        error = SyntaxError("invalid syntax")
        category = classify_error(error, phase="implement")
        assert category.suggested_action
        assert "syntax" in category.suggested_action.lower()

    def test_category_to_dict(self):
        """Category should serialize to dict."""
        error = TypeError("type mismatch")
        category = classify_error(error, phase="verify")
        d = category.to_dict()
        assert d["type"] == "type_error"
        assert d["phase"] == "verify"
        assert "suggested_action" in d


class TestErrorPattern:
    """Tests for ErrorPattern dataclass."""

    def test_pattern_from_exception(self):
        """Should create pattern from exception."""
        error = ValueError("Invalid value")
        pattern = ErrorPattern.from_exception(
            error,
            phase="implement",
            cycle_number=5,
            fix_iteration=2,
        )
        assert pattern.phase == "implement"
        assert pattern.cycle_number == 5
        assert pattern.fix_iteration == 2
        assert "ValueError" in pattern.message or "Invalid" in pattern.message

    def test_pattern_extracts_file_from_traceback(self):
        """Should extract file info from traceback."""
        try:
            raise RuntimeError("Test error")
        except RuntimeError as e:
            pattern = ErrorPattern.from_exception(e, phase="verify")
            # File should be extracted from traceback
            assert pattern.file_involved is not None or pattern.function_involved is not None

    def test_pattern_to_dict(self):
        """Pattern should serialize to dict."""
        error = ValueError("test")
        pattern = ErrorPattern.from_exception(error, phase="test")
        d = pattern.to_dict()
        assert d["phase"] == "test"
        assert "timestamp" in d
        assert "error_type" in d


class TestExtractTestFailures:
    """Tests for test failure extraction from pytest output."""

    def test_extracts_failed_tests(self):
        """Should extract FAILED test patterns."""
        output = """
        FAILED tests/test_foo.py::test_bar - AssertionError
        FAILED tests/test_baz.py::test_qux[param1] - ValueError
        """
        failures = extract_test_failures(output)
        assert len(failures) == 2
        assert failures[0]["file"] == "tests/test_foo.py"
        assert failures[0]["test"] == "test_bar"
        assert failures[1]["test"] == "test_qux[param1]"

    def test_extracts_error_tests(self):
        """Should extract ERROR test patterns."""
        output = """
        ERROR tests/test_setup.py::test_init - ModuleNotFoundError
        """
        failures = extract_test_failures(output)
        assert len(failures) == 1
        assert failures[0]["type"] == "error"

    def test_returns_empty_for_passing_output(self):
        """Should return empty list for passing tests."""
        output = "===== 10 passed in 1.23s ====="
        failures = extract_test_failures(output)
        assert len(failures) == 0


class TestFormatLearningSummary:
    """Tests for learning summary formatting."""

    def test_formats_patterns_by_type(self):
        """Should format patterns grouped by type."""
        patterns = [
            ErrorPattern(
                error_type=ErrorType.SYNTAX,
                phase="implement",
                message="invalid syntax at line 10",
                stack_trace="",
                timestamp=datetime.now(),
                file_involved="foo.py",
            ),
            ErrorPattern(
                error_type=ErrorType.SYNTAX,
                phase="implement",
                message="unexpected token",
                stack_trace="",
                timestamp=datetime.now(),
                file_involved="bar.py",
            ),
        ]
        summary = format_learning_summary(patterns)
        assert "syntax" in summary.lower()
        assert "2 occurrence" in summary.lower()

    def test_returns_empty_for_no_patterns(self):
        """Should return empty string for no patterns."""
        summary = format_learning_summary([])
        assert summary == ""

    def test_shows_file_info(self):
        """Should show file info in summary."""
        patterns = [
            ErrorPattern(
                error_type=ErrorType.TEST_FAILURE,
                phase="verify",
                message="assert failed",
                stack_trace="",
                timestamp=datetime.now(),
                file_involved="tests/test_foo.py",
            ),
        ]
        summary = format_learning_summary(patterns)
        assert "test_foo.py" in summary or "File:" in summary
