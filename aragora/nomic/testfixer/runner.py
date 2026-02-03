"""
TestRunner - Execute tests and capture failures.

Runs test commands and parses output to extract:
- Which tests failed
- The error messages and stack traces
- Which files are involved
- Timing information
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class TestFramework(str, Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    UNKNOWN = "unknown"


@dataclass
class TestFailure:
    """A single test failure."""

    test_name: str
    test_file: str
    error_type: str
    error_message: str
    stack_trace: str
    line_number: int | None = None
    relevant_code: str = ""
    duration_seconds: float = 0.0

    # Parsed context
    involved_files: list[str] = field(default_factory=list)
    involved_functions: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format failure for LLM analysis."""
        lines = [
            f"## Test Failure: {self.test_name}",
            f"File: {self.test_file}",
            f"Error Type: {self.error_type}",
            "",
            "### Error Message",
            self.error_message,
            "",
            "### Stack Trace",
            self.stack_trace,
        ]

        if self.relevant_code:
            lines.extend(
                [
                    "",
                    "### Relevant Code",
                    "```python",
                    self.relevant_code,
                    "```",
                ]
            )

        if self.involved_files:
            lines.extend(
                [
                    "",
                    "### Involved Files",
                    *[f"- {f}" for f in self.involved_files],
                ]
            )

        return "\n".join(lines)


@dataclass
class TestResult:
    """Result of a test run."""

    command: str
    exit_code: int
    stdout: str
    stderr: str

    # Parsed results
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0

    failures: list[TestFailure] = field(default_factory=list)

    duration_seconds: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    framework: TestFramework = TestFramework.UNKNOWN

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0

    @property
    def first_failure(self) -> TestFailure | None:
        return self.failures[0] if self.failures else None

    def summary(self) -> str:
        """Human-readable summary."""
        if self.success:
            return f"✓ All {self.total_tests} tests passed in {self.duration_seconds:.1f}s"
        else:
            return (
                f"✗ {self.failed} failed, {self.errors} errors, "
                f"{self.passed} passed of {self.total_tests} tests "
                f"in {self.duration_seconds:.1f}s"
            )


def detect_framework(command: str) -> TestFramework:
    """Detect test framework from command."""
    cmd_lower = command.lower()

    if "pytest" in cmd_lower:
        return TestFramework.PYTEST
    elif "jest" in cmd_lower:
        return TestFramework.JEST
    elif "mocha" in cmd_lower:
        return TestFramework.MOCHA
    elif "go test" in cmd_lower:
        return TestFramework.GO_TEST
    elif "cargo test" in cmd_lower:
        return TestFramework.CARGO_TEST
    elif "unittest" in cmd_lower or "python -m unittest" in cmd_lower:
        return TestFramework.UNITTEST

    return TestFramework.UNKNOWN


def parse_pytest_output(stdout: str, stderr: str) -> tuple[dict[str, int], list[TestFailure]]:
    """Parse pytest output to extract failures."""
    stats = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    failures = []

    combined = stdout + "\n" + stderr

    # More comprehensive summary parsing
    for match in re.finditer(
        r"(\d+) (failed|passed|error|skipped|warning)", combined, re.IGNORECASE
    ):
        count = int(match.group(1))
        status = match.group(2).lower()
        if status == "passed":
            stats["passed"] = count
        elif status == "failed":
            stats["failed"] = count
        elif status == "error":
            stats["errors"] = count
        elif status == "skipped":
            stats["skipped"] = count

    stats["total"] = stats["passed"] + stats["failed"] + stats["errors"] + stats["skipped"]

    # Parse FAILED lines like "FAILED tests/foo/test_bar.py::TestClass::test_method"
    failed_pattern = re.compile(r"FAILED\s+([^\s:]+)::([^\s]+)")
    for match in failed_pattern.finditer(combined):
        test_file = match.group(1)
        test_name = match.group(2)

        # Try to extract error details from the failure block
        error_type = "AssertionError"
        error_message = ""
        stack_trace = ""

        # Look for the failure block
        failure_block_pattern = re.compile(
            rf"_{(2,)}\s+{re.escape(test_name)}\s+_{(2,)}(.*?)(?=_{(2,)}|\Z)", re.DOTALL
        )
        block_match = failure_block_pattern.search(combined)
        if block_match:
            block = block_match.group(1)

            # Extract error type and message
            error_line_match = re.search(r"([\w.]+Error|[\w.]+Exception):\s*(.+?)(?:\n|$)", block)
            if error_line_match:
                error_type = error_line_match.group(1)
                error_message = error_line_match.group(2).strip()

            stack_trace = block.strip()

        failures.append(
            TestFailure(
                test_name=test_name,
                test_file=test_file,
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
            )
        )

    # Also check for short test summary info
    short_summary_match = re.search(
        r"=+ short test summary info =+(.*?)(?==+|\Z)", combined, re.DOTALL
    )
    if short_summary_match and not failures:
        summary_block = short_summary_match.group(1)
        for line in summary_block.strip().split("\n"):
            if line.startswith("FAILED"):
                parts = line.replace("FAILED ", "").split("::")
                if len(parts) >= 2:
                    test_file = parts[0]
                    test_name = "::".join(parts[1:]).split(" - ")[0]

                    error_message = ""
                    if " - " in line:
                        error_message = line.split(" - ", 1)[1]

                    failures.append(
                        TestFailure(
                            test_name=test_name,
                            test_file=test_file,
                            error_type="TestFailure",
                            error_message=error_message,
                            stack_trace="",
                        )
                    )

    return stats, failures


def extract_involved_files(failure: TestFailure, repo_path: Path) -> list[str]:
    """Extract files mentioned in stack trace."""
    files = set()

    # Look for file paths in stack trace
    file_pattern = re.compile(r'File "([^"]+)"')
    for match in file_pattern.finditer(failure.stack_trace):
        filepath = match.group(1)
        # Filter to repo files
        try:
            rel_path = Path(filepath).relative_to(repo_path)
            files.add(str(rel_path))
        except ValueError:
            pass

    # Add the test file itself
    files.add(failure.test_file)

    return list(files)


class TestRunner:
    """Runs tests and captures failures.

    Example:
        runner = TestRunner(
            repo_path=Path("/path/to/repo"),
            test_command="pytest tests/ -q --maxfail=1",
        )

        result = await runner.run()

        if not result.success:
            for failure in result.failures:
                print(f"Failed: {failure.test_name}")
                print(f"  {failure.error_message}")
    """

    def __init__(
        self,
        repo_path: Path,
        test_command: str,
        timeout_seconds: float = 300.0,
        env: dict[str, str] | None = None,
    ):
        """Initialize the runner.

        Args:
            repo_path: Path to repository root
            test_command: Command to run tests
            timeout_seconds: Timeout for test execution
            env: Environment variables for test process
        """
        self.repo_path = Path(repo_path)
        self.test_command = test_command
        self.timeout = timeout_seconds
        self.env = env
        self.framework = detect_framework(test_command)

    async def run(self) -> TestResult:
        """Run tests and return parsed results.

        Returns:
            TestResult with parsed failures
        """
        start_time = time.time()
        started_at = datetime.now()

        try:
            # Run the test command
            process = await asyncio.create_subprocess_shell(
                self.test_command,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return TestResult(
                    command=self.test_command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Test execution timed out after {self.timeout}s",
                    failures=[
                        TestFailure(
                            test_name="timeout",
                            test_file="",
                            error_type="TimeoutError",
                            error_message=f"Tests did not complete within {self.timeout}s",
                            stack_trace="",
                        )
                    ],
                    duration_seconds=self.timeout,
                    started_at=started_at,
                    framework=self.framework,
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode or 0

        except Exception as e:
            return TestResult(
                command=self.test_command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                failures=[
                    TestFailure(
                        test_name="execution_error",
                        test_file="",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        stack_trace="",
                    )
                ],
                duration_seconds=time.time() - start_time,
                started_at=started_at,
                framework=self.framework,
            )

        duration = time.time() - start_time

        # Parse output based on framework
        if self.framework == TestFramework.PYTEST:
            stats, failures = parse_pytest_output(stdout, stderr)
        else:
            # Fallback: just check exit code
            stats = {
                "total": 0,
                "passed": 0 if exit_code != 0 else 1,
                "failed": 1 if exit_code != 0 else 0,
                "skipped": 0,
                "errors": 0,
            }
            failures = []
            if exit_code != 0:
                failures.append(
                    TestFailure(
                        test_name="unknown",
                        test_file="",
                        error_type="TestFailure",
                        error_message=f"Tests exited with code {exit_code}",
                        stack_trace=stdout + "\n" + stderr,
                    )
                )

        # Enrich failures with file information
        for failure in failures:
            failure.involved_files = extract_involved_files(failure, self.repo_path)

        return TestResult(
            command=self.test_command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            total_tests=stats["total"],
            passed=stats["passed"],
            failed=stats["failed"],
            skipped=stats["skipped"],
            errors=stats["errors"],
            failures=failures,
            duration_seconds=duration,
            started_at=started_at,
            framework=self.framework,
        )

    async def run_single_test(self, test_id: str) -> TestResult:
        """Run a single specific test.

        Args:
            test_id: Test identifier (e.g., "tests/foo.py::test_bar")

        Returns:
            TestResult for the single test
        """
        if self.framework == TestFramework.PYTEST:
            command = f"pytest {test_id} -v"
        else:
            command = f"{self.test_command} {test_id}"

        runner = TestRunner(
            repo_path=self.repo_path,
            test_command=command,
            timeout_seconds=self.timeout,
            env=self.env,
        )
        return await runner.run()
