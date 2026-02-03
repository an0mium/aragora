"""
FailureAnalyzer - Analyze test failures to determine root cause.

Uses AI to:
1. Categorize the type of failure
2. Identify the likely root cause
3. Determine which code needs to change (test vs implementation)
4. Extract relevant context for fix proposals
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from aragora.nomic.testfixer.runner import TestFailure, TestResult


class FailureCategory(str, Enum):
    """Categories of test failures."""

    # Test issues
    TEST_ASSERTION = "test_assertion"  # Test expectation wrong
    TEST_SETUP = "test_setup"  # Test setup/fixture issue
    TEST_ASYNC = "test_async"  # Missing await, async issues
    TEST_MOCK = "test_mock"  # Mock configuration wrong
    TEST_IMPORT = "test_import"  # Import error in test

    # Implementation issues
    IMPL_BUG = "impl_bug"  # Bug in implementation
    IMPL_MISSING = "impl_missing"  # Missing method/attribute
    IMPL_TYPE = "impl_type"  # Type error in implementation
    IMPL_API_CHANGE = "impl_api_change"  # API changed, test outdated

    # Environment issues
    ENV_DEPENDENCY = "env_dependency"  # Missing dependency
    ENV_CONFIG = "env_config"  # Configuration issue
    ENV_RESOURCE = "env_resource"  # Resource unavailable (DB, network)

    # Complex issues
    RACE_CONDITION = "race_condition"  # Timing/race issue
    FLAKY = "flaky"  # Intermittent failure
    UNKNOWN = "unknown"  # Cannot determine


class FixTarget(str, Enum):
    """Where the fix should be applied."""

    TEST_FILE = "test_file"  # Fix the test
    IMPL_FILE = "impl_file"  # Fix the implementation
    BOTH = "both"  # Both need changes
    CONFIG = "config"  # Fix configuration
    SKIP = "skip"  # Skip/mark as xfail
    HUMAN = "human"  # Requires human intervention


@dataclass
class FailureAnalysis:
    """Analysis of a test failure."""

    failure: TestFailure
    analyzed_at: datetime = field(default_factory=datetime.now)

    # Categorization
    category: FailureCategory = FailureCategory.UNKNOWN
    confidence: float = 0.5  # 0.0 to 1.0
    fix_target: FixTarget = FixTarget.TEST_FILE

    # Root cause
    root_cause: str = ""
    root_cause_file: str = ""
    root_cause_line: int | None = None

    # Context for fix
    relevant_code: dict[str, str] = field(default_factory=dict)  # file -> code
    suggested_approach: str = ""

    # Risk assessment
    fix_complexity: str = "low"  # low, medium, high
    regression_risk: str = "low"  # low, medium, high

    # Metadata
    analysis_notes: list[str] = field(default_factory=list)

    def to_fix_prompt(self) -> str:
        """Generate prompt for fix proposal."""
        lines = [
            "## Failure Analysis",
            f"Category: {self.category.value}",
            f"Fix Target: {self.fix_target.value}",
            f"Confidence: {self.confidence:.0%}",
            "",
            "### Root Cause",
            self.root_cause,
            "",
            "### Suggested Approach",
            self.suggested_approach,
            "",
            "### Original Failure",
            self.failure.to_prompt_context(),
        ]

        if self.relevant_code:
            lines.extend(["", "### Relevant Code"])
            for file_path, code in self.relevant_code.items():
                lines.extend(
                    [
                        "",
                        f"#### {file_path}",
                        "```python",
                        code,
                        "```",
                    ]
                )

        return "\n".join(lines)


# Heuristic patterns for failure categorization
CATEGORY_PATTERNS: dict[FailureCategory, list[tuple[str, float]]] = {
    FailureCategory.TEST_ASYNC: [
        (r"coroutine.*was never awaited", 0.95),
        (r"RuntimeWarning.*coroutine", 0.9),
        (r"await.*missing", 0.85),
        (r"async.*not.*await", 0.8),
    ],
    FailureCategory.TEST_MOCK: [
        (r"MagicMock.*has no attribute", 0.9),
        (r"mock.*not called", 0.85),
        (r"call_args.*None", 0.8),
        (r"AssertionError.*mock", 0.75),
    ],
    FailureCategory.IMPL_MISSING: [
        (r"AttributeError.*has no attribute", 0.9),
        (r"NameError.*not defined", 0.9),
        (r"ModuleNotFoundError", 0.85),
        (r"ImportError", 0.8),
    ],
    FailureCategory.IMPL_TYPE: [
        (r"TypeError.*argument", 0.85),
        (r"TypeError.*expected", 0.85),
        (r"cannot unpack", 0.8),
    ],
    FailureCategory.TEST_ASSERTION: [
        (r"AssertionError", 0.7),
        (r"assert.*==", 0.65),
        (r"Expected.*but got", 0.8),
    ],
    FailureCategory.ENV_DEPENDENCY: [
        (r"ModuleNotFoundError.*No module named", 0.9),
        (r"pip install", 0.8),
    ],
    FailureCategory.RACE_CONDITION: [
        (r"timeout", 0.6),
        (r"deadlock", 0.9),
        (r"race condition", 0.95),
    ],
}


def categorize_by_heuristics(failure: TestFailure) -> tuple[FailureCategory, float]:
    """Categorize failure using pattern matching."""
    combined_text = (
        failure.error_type + " " + failure.error_message + " " + failure.stack_trace
    ).lower()

    best_category = FailureCategory.UNKNOWN
    best_confidence = 0.0

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern, confidence in patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                if confidence > best_confidence:
                    best_category = category
                    best_confidence = confidence

    return best_category, best_confidence


def determine_fix_target(category: FailureCategory, failure: TestFailure) -> FixTarget:
    """Determine where the fix should be applied."""
    # Test issues -> fix test
    if category in [
        FailureCategory.TEST_ASSERTION,
        FailureCategory.TEST_SETUP,
        FailureCategory.TEST_ASYNC,
        FailureCategory.TEST_MOCK,
    ]:
        return FixTarget.TEST_FILE

    # Implementation issues -> fix implementation
    if category in [
        FailureCategory.IMPL_BUG,
        FailureCategory.IMPL_MISSING,
        FailureCategory.IMPL_TYPE,
    ]:
        return FixTarget.IMPL_FILE

    # API changes often need both
    if category == FailureCategory.IMPL_API_CHANGE:
        return FixTarget.BOTH

    # Environment -> config (or implementation if missing dependency is in app code)
    if category in [FailureCategory.ENV_DEPENDENCY, FailureCategory.ENV_CONFIG]:
        if failure.involved_files:
            for f in failure.involved_files:
                if "test" not in f.lower():
                    return FixTarget.IMPL_FILE
        return FixTarget.CONFIG

    # Complex issues might need human
    if category in [FailureCategory.RACE_CONDITION, FailureCategory.FLAKY]:
        return FixTarget.HUMAN

    return FixTarget.TEST_FILE  # Default to test


def extract_relevant_code(
    failure: TestFailure,
    repo_path: Path,
    context_lines: int = 10,
) -> dict[str, str]:
    """Extract code snippets from involved files."""
    code_snippets = {}

    for file_path in failure.involved_files[:5]:  # Limit to 5 files
        full_path = repo_path / file_path

        if not full_path.exists():
            continue

        try:
            content = full_path.read_text()
            lines = content.split("\n")

            # Try to find the relevant line from stack trace
            line_match = re.search(rf"{re.escape(file_path)}.*line (\d+)", failure.stack_trace)

            if line_match:
                target_line = int(line_match.group(1))
                start = max(0, target_line - context_lines)
                end = min(len(lines), target_line + context_lines)
                snippet = "\n".join(
                    f"{i + 1:4d}: {line}" for i, line in enumerate(lines[start:end], start=start)
                )
            else:
                # Just take the first 50 lines
                snippet = "\n".join(f"{i + 1:4d}: {line}" for i, line in enumerate(lines[:50]))

            code_snippets[file_path] = snippet

        except Exception:
            pass

    return code_snippets


def generate_approach_heuristic(
    category: FailureCategory,
    failure: TestFailure,
) -> str:
    """Generate fix approach suggestion based on category."""
    approaches = {
        FailureCategory.TEST_ASYNC: (
            "The test is missing 'await' for an async function call. "
            "Add @pytest.mark.asyncio decorator if missing, and await the "
            "coroutine that's being called."
        ),
        FailureCategory.TEST_MOCK: (
            "The mock is not configured correctly. Check that:\n"
            "1. The mock method exists with the right name\n"
            "2. Return values are set with the expected attributes\n"
            "3. The mock is patching the right location"
        ),
        FailureCategory.IMPL_MISSING: (
            "A method or attribute is missing. Either:\n"
            "1. Add the missing method/attribute to the implementation\n"
            "2. Or update the test if the API has changed intentionally"
        ),
        FailureCategory.IMPL_TYPE: (
            "There's a type mismatch. Check:\n"
            "1. Function signatures match expected arguments\n"
            "2. Return types are correct\n"
            "3. Data structure shapes match expectations"
        ),
        FailureCategory.TEST_ASSERTION: (
            "The assertion is failing. Determine if:\n"
            "1. The expected value in the test is wrong (update test)\n"
            "2. The implementation is returning wrong value (fix impl)\n"
            "3. The test setup is missing something"
        ),
        FailureCategory.ENV_DEPENDENCY: (
            "A dependency is missing. Add it to requirements.txt or "
            "pyproject.toml, or mock it if it's optional."
        ),
        FailureCategory.IMPL_API_CHANGE: (
            "The API has changed. Update the test to match the new API, "
            "or update the implementation if the change was unintentional."
        ),
    }

    return approaches.get(
        category, "Analyze the error message and stack trace to determine the root cause."
    )


class AIAnalyzer(Protocol):
    """Protocol for AI-based analysis."""

    async def analyze(
        self,
        failure: TestFailure,
        code_context: dict[str, str],
    ) -> tuple[str, str, float]:
        """Analyze failure and return (root_cause, approach, confidence)."""
        ...


class FailureAnalyzer:
    """Analyzes test failures to determine root cause and fix approach.

    Combines heuristic analysis with optional AI-powered deep analysis.

    Example:
        analyzer = FailureAnalyzer(repo_path=Path("/path/to/repo"))

        analysis = await analyzer.analyze(failure)

        print(f"Category: {analysis.category}")
        print(f"Fix target: {analysis.fix_target}")
        print(f"Root cause: {analysis.root_cause}")
    """

    def __init__(
        self,
        repo_path: Path,
        ai_analyzer: AIAnalyzer | None = None,
        context_lines: int = 10,
    ):
        """Initialize the analyzer.

        Args:
            repo_path: Path to repository root
            ai_analyzer: Optional AI-based analyzer for deep analysis
            context_lines: Lines of context to extract around errors
        """
        self.repo_path = Path(repo_path)
        self.ai_analyzer = ai_analyzer
        self.context_lines = context_lines

    async def analyze(self, failure: TestFailure) -> FailureAnalysis:
        """Analyze a test failure.

        Args:
            failure: The failure to analyze

        Returns:
            FailureAnalysis with categorization and fix guidance
        """
        # Start with heuristic categorization
        category, heuristic_confidence = categorize_by_heuristics(failure)

        # Determine fix target
        fix_target = determine_fix_target(category, failure)

        # Extract relevant code
        relevant_code = extract_relevant_code(
            failure,
            self.repo_path,
            self.context_lines,
        )

        # Generate approach suggestion
        suggested_approach = generate_approach_heuristic(category, failure)

        # Default root cause from heuristics
        root_cause = f"{category.value}: {failure.error_message}"

        # Optionally use AI for deeper analysis
        confidence = heuristic_confidence
        if self.ai_analyzer and heuristic_confidence < 0.8:
            try:
                ai_root_cause, ai_approach, ai_confidence = await self.ai_analyzer.analyze(
                    failure,
                    relevant_code,
                )
                if ai_confidence > heuristic_confidence:
                    root_cause = ai_root_cause
                    suggested_approach = ai_approach
                    confidence = ai_confidence
            except Exception:
                pass  # Fall back to heuristics

        # Determine root cause file
        root_cause_file = failure.test_file
        if fix_target == FixTarget.IMPL_FILE and failure.involved_files:
            # Find non-test file in involved files
            for f in failure.involved_files:
                if "test" not in f.lower():
                    root_cause_file = f
                    break
        # For missing dependencies, prefer the file that imports the missing module.
        if category == FailureCategory.ENV_DEPENDENCY and failure.involved_files:
            missing_mod = None
            match = re.search(r"No module named '([^']+)'", failure.error_message)
            if match:
                missing_mod = match.group(1)
            if not missing_mod:
                match = re.search(r"No module named '([^']+)'", failure.stack_trace)
                if match:
                    missing_mod = match.group(1)
            if missing_mod:
                for f in failure.involved_files:
                    if "test" in f.lower():
                        continue
                    path = self.repo_path / f
                    if not path.exists():
                        continue
                    try:
                        content = path.read_text()
                    except Exception:
                        continue
                    if f"import {missing_mod}" in content or f"from {missing_mod}" in content:
                        root_cause_file = f
                        break

        # Estimate complexity
        fix_complexity = "low"
        if category in [FailureCategory.IMPL_API_CHANGE, FailureCategory.RACE_CONDITION]:
            fix_complexity = "high"
        elif category in [FailureCategory.IMPL_BUG, FailureCategory.TEST_MOCK]:
            fix_complexity = "medium"

        return FailureAnalysis(
            failure=failure,
            category=category,
            confidence=confidence,
            fix_target=fix_target,
            root_cause=root_cause,
            root_cause_file=root_cause_file,
            relevant_code=relevant_code,
            suggested_approach=suggested_approach,
            fix_complexity=fix_complexity,
            regression_risk="low" if fix_target == FixTarget.TEST_FILE else "medium",
        )

    async def analyze_result(self, result: TestResult) -> list[FailureAnalysis]:
        """Analyze all failures in a test result.

        Args:
            result: Test result with failures

        Returns:
            List of FailureAnalysis for each failure
        """
        analyses = []
        for failure in result.failures:
            analysis = await self.analyze(failure)
            analyses.append(analysis)
        return analyses
