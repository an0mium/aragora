"""Forward-fix system for diagnosing and repairing test failures.

Instead of auto-reverting on test failure (losing work), this module
diagnoses the root cause and attempts targeted fixes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Classification of test failure root causes."""

    ASSERTION_MISMATCH = "assertion_mismatch"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# Patterns used to classify error messages
_ERROR_PATTERNS: list[tuple[FailureType, re.Pattern[str]]] = [
    (FailureType.ASSERTION_MISMATCH, re.compile(r"AssertionError|assert\s+.*==|assert\s+.*!=|assert\s+.*\bin\b")),
    (FailureType.IMPORT_ERROR, re.compile(r"ImportError|ModuleNotFoundError|cannot import name")),
    (FailureType.TYPE_ERROR, re.compile(r"TypeError.*argument|TypeError.*expected|TypeError.*got")),
    (FailureType.ATTRIBUTE_ERROR, re.compile(r"AttributeError.*has no attribute|AttributeError.*object")),
    (FailureType.PERMISSION_ERROR, re.compile(r"PermissionError|permission denied|403|RBAC|Forbidden", re.IGNORECASE)),
    (FailureType.TIMEOUT, re.compile(r"TimeoutError|timed?\s*out|deadline exceeded", re.IGNORECASE)),
]

# Regex for pytest FAILED lines
_FAILED_TEST_RE = re.compile(r"FAILED\s+([\w/.]+::\w[\w:]*)")
# Regex for short test summary lines
_ERROR_LINE_RE = re.compile(r"^E\s+(.+)$", re.MULTILINE)
# Regex for traceback "File" lines
_TRACEBACK_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')


@dataclass
class DiagnosisResult:
    """Result of diagnosing a test failure."""

    failure_type: FailureType
    failed_tests: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    likely_cause: str = ""
    affected_files: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ForwardFix:
    """A proposed fix for a diagnosed failure."""

    fix_type: str
    description: str
    file_path: str
    old_content: str
    new_content: str
    confidence: float = 0.0


class ForwardFixer:
    """Diagnose test failures and propose targeted fixes."""

    def __init__(
        self,
        max_attempts: int = 3,
        min_confidence: float = 0.7,
    ) -> None:
        self.max_attempts = max_attempts
        self.min_confidence = min_confidence

    def diagnose_failure(
        self,
        test_output: str,
        diff: str | None = None,
    ) -> DiagnosisResult:
        """Parse pytest output and classify the failure.

        If *diff* is provided, correlates changed lines with failures
        to improve confidence.
        """
        parsed = self._parse_pytest_output(test_output)
        failed_tests = [p["test"] for p in parsed]
        error_messages = [p["error"] for p in parsed]
        affected_files = list({p["file"] for p in parsed if p.get("file")})

        failure_type = self._classify_error("\n".join(error_messages))
        confidence = 0.5 if failure_type != FailureType.UNKNOWN else 0.2

        likely_cause = ""
        if diff:
            cause = self._correlate_with_diff(diff, parsed)
            if cause:
                likely_cause = cause
                confidence = min(1.0, confidence + 0.3)

        if not likely_cause and error_messages:
            likely_cause = error_messages[0][:200]

        # Boost confidence for well-understood failure types
        if failure_type in (FailureType.IMPORT_ERROR, FailureType.ATTRIBUTE_ERROR):
            confidence = min(1.0, confidence + 0.1)

        return DiagnosisResult(
            failure_type=failure_type,
            failed_tests=failed_tests,
            error_messages=error_messages,
            likely_cause=likely_cause,
            affected_files=affected_files,
            confidence=round(confidence, 2),
        )

    def suggest_fix(self, diagnosis: DiagnosisResult) -> ForwardFix | None:
        """Propose a fix based on diagnosis. Returns None if unsure."""
        if diagnosis.confidence < self.min_confidence:
            return None

        if diagnosis.failure_type == FailureType.UNKNOWN:
            return None

        if diagnosis.failure_type == FailureType.IMPORT_ERROR:
            return self._suggest_import_fix(diagnosis)

        if diagnosis.failure_type == FailureType.ATTRIBUTE_ERROR:
            return self._suggest_attribute_fix(diagnosis)

        if diagnosis.failure_type == FailureType.ASSERTION_MISMATCH:
            return self._suggest_assertion_fix(diagnosis)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_pytest_output(self, output: str) -> list[dict[str, str]]:
        """Extract test names, error messages, and file paths from pytest output."""
        results: list[dict[str, str]] = []

        # Find failed test names
        failed_tests = _FAILED_TEST_RE.findall(output)

        # Find error lines (E  ...)
        error_lines = _ERROR_LINE_RE.findall(output)

        # Find traceback file references
        tb_files = _TRACEBACK_FILE_RE.findall(output)

        if failed_tests:
            for i, test in enumerate(failed_tests):
                entry: dict[str, str] = {"test": test, "error": "", "file": ""}
                if i < len(error_lines):
                    entry["error"] = error_lines[i].strip()
                # Associate file from traceback
                test_file = test.split("::")[0] if "::" in test else ""
                entry["file"] = test_file
                results.append(entry)
        elif error_lines:
            # No FAILED lines but have error lines (unusual)
            for err in error_lines:
                results.append({"test": "", "error": err.strip(), "file": ""})

        # Add traceback files to affected list
        for filepath, _lineno in tb_files:
            if filepath and not any(r.get("file") == filepath for r in results):
                if not results:
                    results.append({"test": "", "error": "", "file": filepath})

        return results

    def _classify_error(self, error_text: str) -> FailureType:
        """Classify error text into a FailureType."""
        for ftype, pattern in _ERROR_PATTERNS:
            if pattern.search(error_text):
                return ftype
        return FailureType.UNKNOWN

    def _correlate_with_diff(self, diff: str, errors: list[dict[str, str]]) -> str:
        """Find which changed lines likely caused the failure."""
        # Extract changed file paths from diff
        changed_files: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                path = line.split("/", 1)[1] if "/" in line else ""
                if path and path != "/dev/null":
                    changed_files.append(path)

        # Check overlap with error files
        error_files = {e.get("file", "") for e in errors if e.get("file")}
        overlap = set(changed_files) & error_files
        if overlap:
            return f"Changes in {', '.join(sorted(overlap))} likely caused the failure"

        # Check if error messages reference changed content
        for e in errors:
            err = e.get("error", "")
            for cf in changed_files:
                basename = cf.rsplit("/", 1)[-1] if "/" in cf else cf
                if basename and basename in err:
                    return f"Error references changed file {cf}"

        return ""

    def _suggest_import_fix(self, diagnosis: DiagnosisResult) -> ForwardFix | None:
        """Suggest adding a missing import."""
        for msg in diagnosis.error_messages:
            # "cannot import name 'Foo' from 'bar.baz'"
            m = re.search(r"cannot import name '(\w+)' from '([\w.]+)'", msg)
            if m:
                name, module = m.group(1), m.group(2)
                return ForwardFix(
                    fix_type="add_import",
                    description=f"Add missing import: {name} from {module}",
                    file_path=diagnosis.affected_files[0] if diagnosis.affected_files else "",
                    old_content="",
                    new_content=f"from {module} import {name}",
                    confidence=diagnosis.confidence,
                )
            # "No module named 'foo.bar'"
            m = re.search(r"No module named '([\w.]+)'", msg)
            if m:
                module = m.group(1)
                return ForwardFix(
                    fix_type="fix_module_path",
                    description=f"Module not found: {module}",
                    file_path=diagnosis.affected_files[0] if diagnosis.affected_files else "",
                    old_content=module,
                    new_content=module,  # Placeholder - needs manual resolution
                    confidence=max(0.5, diagnosis.confidence - 0.2),
                )
        return None

    def _suggest_attribute_fix(self, diagnosis: DiagnosisResult) -> ForwardFix | None:
        """Suggest fixing a wrong attribute name."""
        for msg in diagnosis.error_messages:
            m = re.search(r"has no attribute '(\w+)'", msg)
            if m:
                attr = m.group(1)
                return ForwardFix(
                    fix_type="fix_attribute",
                    description=f"Attribute '{attr}' not found on object",
                    file_path=diagnosis.affected_files[0] if diagnosis.affected_files else "",
                    old_content=attr,
                    new_content=attr,  # Placeholder - needs context
                    confidence=max(0.5, diagnosis.confidence - 0.1),
                )
        return None

    def _suggest_assertion_fix(self, diagnosis: DiagnosisResult) -> ForwardFix | None:
        """Suggest updating a test assertion."""
        for msg in diagnosis.error_messages:
            # "assert X == Y" pattern
            m = re.search(r"assert\s+(.+?)\s*==\s*(.+)", msg)
            if m:
                actual, expected = m.group(1).strip(), m.group(2).strip()
                return ForwardFix(
                    fix_type="update_assertion",
                    description=f"Test expected {expected} but got {actual}",
                    file_path=diagnosis.failed_tests[0].split("::")[0] if diagnosis.failed_tests else "",
                    old_content=expected,
                    new_content=actual,
                    confidence=max(0.5, diagnosis.confidence - 0.1),
                )
        return None
