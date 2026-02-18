"""Structured error analysis for implementation retry enrichment.

Classifies execution errors by pattern (syntax, import, test failure, timeout,
runtime) and generates retry hints that get injected into the next attempt's
prompt. This replaces the naive "was it a timeout?" check with structured
analysis so retries learn from failures.

Usage:
    analyzer = ErrorAnalyzer()
    analysis = analyzer.analyze(error_text, stderr_text)
    if analysis.retryable:
        hint = analyzer.build_retry_hint(analysis, original_prompt)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Patterns: (regex, category, retryable)
_ERROR_PATTERNS: list[tuple[str, str, bool]] = [
    (r"SyntaxError", "syntax", True),
    (r"IndentationError", "syntax", True),
    (r"ImportError.*No module named", "import", True),
    (r"ModuleNotFoundError", "import", True),
    (r"FAILED.*\d+ failed", "test_failure", True),
    (r"ERRORS?.*\d+ error", "test_failure", True),
    (r"TimeoutError|timed out|timeout", "timeout", True),
    (r"NameError.*is not defined", "runtime", True),
    (r"AttributeError.*has no attribute", "runtime", True),
    (r"FileNotFoundError", "runtime", True),
    (r"TypeError.*argument", "runtime", True),
    (r"KeyError", "runtime", True),
]

# Regex to extract file paths from error output
_FILE_PATTERN = re.compile(r'File "([^"]+\.py)"')

# Suggested fix templates per category
_FIX_SUGGESTIONS: dict[str, str] = {
    "syntax": "Fix the syntax error — check for missing colons, parentheses, or indentation.",
    "import": "The module import failed. Check that the module exists and is spelled correctly.",
    "test_failure": "Tests failed. Read the test output carefully and fix the failing assertions.",
    "timeout": "The previous attempt timed out. Simplify the approach or break into smaller steps.",
    "runtime": "A runtime error occurred. Check variable names, attribute access, and types.",
    "unknown": "An unexpected error occurred. Review the error output for clues.",
}


@dataclass
class ErrorAnalysis:
    """Result of analyzing an implementation error."""

    category: str  # "syntax", "import", "test_failure", "timeout", "runtime", "unknown"
    retryable: bool
    suggested_fix: str  # Human-readable hint for the retry prompt
    relevant_files: list[str] = field(default_factory=list)
    error_summary: str = ""  # Concise 1-line summary


class ErrorAnalyzer:
    """Classifies implementation errors and generates retry hints."""

    def analyze(self, error: str, stderr: str = "") -> ErrorAnalysis:
        """Classify an error and extract structured information.

        Args:
            error: Primary error text (e.g. from TaskResult.error).
            stderr: Optional stderr output from the subprocess.

        Returns:
            ErrorAnalysis with category, retryability, and file references.
        """
        combined = f"{error}\n{stderr}".strip()
        if not combined:
            return ErrorAnalysis(
                category="unknown",
                retryable=False,
                suggested_fix=_FIX_SUGGESTIONS["unknown"],
                error_summary="No error output available",
            )

        # Match against patterns (first match wins)
        category = "unknown"
        retryable = False
        for pattern, cat, is_retryable in _ERROR_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                category = cat
                retryable = is_retryable
                break

        # Extract referenced file paths
        relevant_files = list(dict.fromkeys(_FILE_PATTERN.findall(combined)))[:10]

        # Build a concise summary (first meaningful line)
        error_summary = ""
        for line in combined.splitlines():
            line = line.strip()
            if line and not line.startswith("Traceback"):
                error_summary = line[:200]
                break
        if not error_summary:
            error_summary = combined[:200]

        return ErrorAnalysis(
            category=category,
            retryable=retryable,
            suggested_fix=_FIX_SUGGESTIONS.get(category, _FIX_SUGGESTIONS["unknown"]),
            relevant_files=relevant_files,
            error_summary=error_summary,
        )

    def build_retry_hint(self, analysis: ErrorAnalysis, original_prompt: str = "") -> str:
        """Generate a retry hint section to inject into the next attempt's prompt.

        Args:
            analysis: The error analysis from analyze().
            original_prompt: The original implementation prompt (for context).

        Returns:
            A formatted string suitable for appending to the retry prompt.
        """
        parts = [
            "## Previous Attempt Failed",
            f"**Error type:** {analysis.category}",
            f"**Summary:** {analysis.error_summary}",
            f"**Suggested fix:** {analysis.suggested_fix}",
        ]

        if analysis.relevant_files:
            files_str = ", ".join(analysis.relevant_files[:5])
            parts.append(f"**Files involved:** {files_str}")

        parts.append(
            "\nPlease address this error in your implementation. "
            "Do not repeat the same mistake."
        )

        return "\n".join(parts)
