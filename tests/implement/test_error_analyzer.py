"""Unit tests for ErrorAnalyzer and ErrorAnalysis."""

from __future__ import annotations

from aragora.implement.error_analyzer import ErrorAnalysis, ErrorAnalyzer


class TestErrorAnalyzerClassification:
    """Tests for ErrorAnalyzer.analyze() category classification."""

    def setup_method(self) -> None:
        self.analyzer = ErrorAnalyzer()

    def test_classify_syntax_error(self) -> None:
        """SyntaxError maps to category 'syntax' and is retryable."""
        analysis = self.analyzer.analyze(
            'File "main.py", line 10\n    if True\n         ^\nSyntaxError: expected \':\'',
        )
        assert analysis.category == "syntax"
        assert analysis.retryable is True

    def test_classify_import_error(self) -> None:
        """ImportError with 'No module named' maps to category 'import' and is retryable."""
        analysis = self.analyzer.analyze("ImportError: No module named 'foo'")
        assert analysis.category == "import"
        assert analysis.retryable is True

    def test_classify_test_failure(self) -> None:
        """Pytest failure summary maps to category 'test_failure'."""
        analysis = self.analyzer.analyze(
            "FAILED tests/test_x.py::test_y - AssertionError === 3 failed ===",
        )
        assert analysis.category == "test_failure"
        assert analysis.retryable is True

    def test_classify_timeout(self) -> None:
        """TimeoutError maps to category 'timeout'."""
        analysis = self.analyzer.analyze("TimeoutError: operation timed out after 30s")
        assert analysis.category == "timeout"
        assert analysis.retryable is True

    def test_classify_runtime_error(self) -> None:
        """NameError maps to category 'runtime'."""
        analysis = self.analyzer.analyze("NameError: name 'x' is not defined")
        assert analysis.category == "runtime"
        assert analysis.retryable is True

    def test_classify_unknown_error(self) -> None:
        """Unrecognised error text maps to category 'unknown' and is not retryable."""
        analysis = self.analyzer.analyze("Something unexpected happened")
        assert analysis.category == "unknown"
        assert analysis.retryable is False

    def test_empty_error_handled(self) -> None:
        """Empty error string maps to 'unknown' with retryable=False."""
        analysis = self.analyzer.analyze("")
        assert analysis.category == "unknown"
        assert analysis.retryable is False
        assert analysis.error_summary  # should still have a summary


class TestErrorAnalyzerFileExtraction:
    """Tests for file-path extraction from error output."""

    def setup_method(self) -> None:
        self.analyzer = ErrorAnalyzer()

    def test_extract_file_references(self) -> None:
        """File paths in 'File "path.py"' patterns are extracted into relevant_files."""
        error = (
            'Traceback (most recent call last):\n'
            '  File "aragora/foo.py", line 42, in bar\n'
            '    return baz()\n'
            '  File "aragora/baz.py", line 7, in baz\n'
            '    raise ValueError\n'
            'ValueError'
        )
        analysis = self.analyzer.analyze(error)
        assert "aragora/foo.py" in analysis.relevant_files
        assert "aragora/baz.py" in analysis.relevant_files


class TestBuildRetryHint:
    """Tests for ErrorAnalyzer.build_retry_hint() output."""

    def setup_method(self) -> None:
        self.analyzer = ErrorAnalyzer()

    def test_build_retry_hint_contains_error_summary(self) -> None:
        """The retry hint includes the error_summary text."""
        analysis = ErrorAnalysis(
            category="syntax",
            retryable=True,
            suggested_fix="Fix the syntax.",
            error_summary="SyntaxError: unexpected EOF",
        )
        hint = self.analyzer.build_retry_hint(analysis)
        assert "SyntaxError: unexpected EOF" in hint

    def test_build_retry_hint_contains_fix_suggestion(self) -> None:
        """The retry hint includes the suggested_fix text."""
        analysis = ErrorAnalysis(
            category="import",
            retryable=True,
            suggested_fix="Check that the module exists and is spelled correctly.",
            error_summary="ImportError: No module named 'bar'",
        )
        hint = self.analyzer.build_retry_hint(analysis)
        assert "Check that the module exists and is spelled correctly." in hint

    def test_retry_hint_format(self) -> None:
        """The retry hint starts with the '## Previous Attempt Failed' header."""
        analysis = ErrorAnalysis(
            category="runtime",
            retryable=True,
            suggested_fix="Check variable names.",
            error_summary="NameError: name 'x' is not defined",
        )
        hint = self.analyzer.build_retry_hint(analysis)
        assert hint.startswith("## Previous Attempt Failed")

    def test_non_retryable_skips_retry(self) -> None:
        """Unknown errors have retryable=False so callers know not to retry."""
        analysis = self.analyzer.analyze("Something completely unexpected")
        assert analysis.category == "unknown"
        assert analysis.retryable is False
