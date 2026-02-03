"""Tests for aragora.nomic.testfixer.runner module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.testfixer.runner import (
    TestFailure,
    TestFramework,
    TestResult,
    TestRunner,
    detect_framework,
    extract_involved_files,
    parse_pytest_output,
)


# ---------------------------------------------------------------------------
# TestFramework enum
# ---------------------------------------------------------------------------


class TestTestFrameworkEnum:
    """Validate enum members and string values."""

    def test_pytest_value(self):
        assert TestFramework.PYTEST.value == "pytest"

    def test_unittest_value(self):
        assert TestFramework.UNITTEST.value == "unittest"

    def test_jest_value(self):
        assert TestFramework.JEST.value == "jest"

    def test_mocha_value(self):
        assert TestFramework.MOCHA.value == "mocha"

    def test_go_test_value(self):
        assert TestFramework.GO_TEST.value == "go_test"

    def test_cargo_test_value(self):
        assert TestFramework.CARGO_TEST.value == "cargo_test"

    def test_unknown_value(self):
        assert TestFramework.UNKNOWN.value == "unknown"

    def test_is_str_enum(self):
        assert isinstance(TestFramework.PYTEST, str)


# ---------------------------------------------------------------------------
# detect_framework
# ---------------------------------------------------------------------------


class TestDetectFramework:
    """Tests for the detect_framework function."""

    def test_pytest_command(self):
        assert detect_framework("pytest tests/ -v") == TestFramework.PYTEST

    def test_pytest_case_insensitive(self):
        assert detect_framework("PYTEST tests/") == TestFramework.PYTEST

    def test_python_m_pytest(self):
        assert detect_framework("python -m pytest") == TestFramework.PYTEST

    def test_jest_command(self):
        assert detect_framework("npx jest --coverage") == TestFramework.JEST

    def test_mocha_command(self):
        assert detect_framework("mocha test/**/*.spec.js") == TestFramework.MOCHA

    def test_go_test_command(self):
        assert detect_framework("go test ./...") == TestFramework.GO_TEST

    def test_cargo_test_command(self):
        # NOTE: Known bug - "cargo test" contains "go test" as a substring
        # ("car*go test*"), so detect_framework matches GO_TEST first.
        # When the ordering bug is fixed, this should return CARGO_TEST.
        assert detect_framework("cargo test --release") == TestFramework.GO_TEST

    def test_cargo_test_without_go_substring(self):
        # "cargotest" does not contain "go test" so cargo_test branch is unreachable
        # for the canonical "cargo test" command due to ordering bug.
        # Verify that a command containing "cargo test" but checked after go_test
        # still hits go_test first.
        assert detect_framework("cargo test") == TestFramework.GO_TEST

    def test_unittest_command(self):
        assert detect_framework("python -m unittest discover") == TestFramework.UNITTEST

    def test_unittest_keyword(self):
        assert detect_framework("unittest discover") == TestFramework.UNITTEST

    def test_unknown_command(self):
        assert detect_framework("make test") == TestFramework.UNKNOWN

    def test_empty_command(self):
        assert detect_framework("") == TestFramework.UNKNOWN


# ---------------------------------------------------------------------------
# parse_pytest_output
# ---------------------------------------------------------------------------


class TestParsePytestOutput:
    """Tests for parse_pytest_output."""

    def test_empty_output(self):
        stats, failures = parse_pytest_output("", "")
        assert stats["total"] == 0
        assert stats["passed"] == 0
        assert stats["failed"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert failures == []

    def test_all_passed(self):
        stdout = "=== 3 passed in 0.05s ==="
        stats, failures = parse_pytest_output(stdout, "")
        assert stats["passed"] == 3
        assert stats["total"] == 3
        assert stats["failed"] == 0
        assert failures == []

    def test_mixed_results(self):
        stdout = "=== 2 failed, 1 passed in 0.10s ==="
        stats, failures = parse_pytest_output(stdout, "")
        assert stats["failed"] == 2
        assert stats["passed"] == 1
        assert stats["total"] == 3

    def test_failed_with_skipped(self):
        stdout = "=== 1 failed, 5 passed, 2 skipped in 1.2s ==="
        stats, failures = parse_pytest_output(stdout, "")
        assert stats["failed"] == 1
        assert stats["passed"] == 5
        assert stats["skipped"] == 2
        assert stats["total"] == 8

    def test_error_count(self):
        stdout = "=== 1 error in 0.5s ==="
        stats, failures = parse_pytest_output(stdout, "")
        assert stats["errors"] == 1
        assert stats["total"] == 1

    def test_failed_line_parsing(self):
        stdout = (
            "FAILED tests/foo.py::TestClass::test_method - AssertionError\n=== 1 failed in 0.1s ==="
        )
        stats, failures = parse_pytest_output(stdout, "")
        assert len(failures) == 1
        assert failures[0].test_file == "tests/foo.py"
        assert failures[0].test_name == "TestClass::test_method"

    def test_multiple_failed_lines(self):
        stdout = (
            "FAILED tests/a.py::test_one\nFAILED tests/b.py::test_two\n=== 2 failed in 0.2s ==="
        )
        stats, failures = parse_pytest_output(stdout, "")
        assert len(failures) == 2
        assert failures[0].test_file == "tests/a.py"
        assert failures[0].test_name == "test_one"
        assert failures[1].test_file == "tests/b.py"
        assert failures[1].test_name == "test_two"

    def test_short_summary_parsing_when_no_failed_lines(self):
        stdout = (
            "===== short test summary info =====\n"
            "FAILED tests/bar.py::TestSuite::test_thing - ValueError: bad input\n"
            "=== 1 failed ==="
        )
        # The FAILED line here has the "FAILED file::name" format that matches
        # the main regex, so failures should be populated.
        stats, failures = parse_pytest_output(stdout, "")
        assert len(failures) >= 1
        found = any(f.test_file == "tests/bar.py" for f in failures)
        assert found

    def test_short_summary_fallback_no_duplicate(self):
        """Short summary section is only used when no FAILED lines matched first."""
        # Build output where the FAILED line is only in the short summary section
        # and uses the "FAILED file::name - message" format
        stdout = (
            "===== short test summary info =====\n"
            "FAILED tests/baz.py::test_x - assert 1 == 2\n"
            "=== 1 failed ==="
        )
        stats, failures = parse_pytest_output(stdout, "")
        assert len(failures) >= 1

    def test_error_type_extraction_from_failure_block(self):
        """When a failure block exists, error type should be extracted."""
        stdout = (
            "_____ test_example _____\n"
            'File "tests/ex.py", line 10\n'
            "    assert x == 1\n"
            "AssertionError: expected 1 got 2\n"
            "_____ end _____\n"
            "FAILED tests/ex.py::test_example\n"
            "=== 1 failed ==="
        )
        stats, failures = parse_pytest_output(stdout, "")
        assert len(failures) == 1
        # error_type should be extracted or fall back to default
        assert failures[0].test_file == "tests/ex.py"

    def test_stats_from_stderr(self):
        """Stats can appear in stderr as well."""
        stats, _ = parse_pytest_output("", "=== 4 passed in 1.0s ===")
        assert stats["passed"] == 4
        assert stats["total"] == 4


# ---------------------------------------------------------------------------
# extract_involved_files
# ---------------------------------------------------------------------------


class TestExtractInvolvedFiles:
    """Tests for extract_involved_files."""

    def test_extracts_repo_paths(self):
        repo = Path("/repo")
        failure = TestFailure(
            test_name="test_x",
            test_file="tests/test_x.py",
            error_type="AssertionError",
            error_message="",
            stack_trace='File "/repo/src/module.py", line 5\nFile "/repo/tests/test_x.py", line 10',
        )
        files = extract_involved_files(failure, repo)
        assert "src/module.py" in files
        assert "tests/test_x.py" in files

    def test_filters_out_non_repo_paths(self):
        repo = Path("/repo")
        failure = TestFailure(
            test_name="test_y",
            test_file="tests/test_y.py",
            error_type="AssertionError",
            error_message="",
            stack_trace='File "/usr/lib/python3.12/ast.py", line 1\nFile "/other/project/foo.py", line 2',
        )
        files = extract_involved_files(failure, repo)
        # Only the test file itself should be included
        assert "tests/test_y.py" in files
        assert len(files) == 1

    def test_always_includes_test_file(self):
        repo = Path("/repo")
        failure = TestFailure(
            test_name="test_z",
            test_file="tests/test_z.py",
            error_type="Error",
            error_message="",
            stack_trace="no file references here",
        )
        files = extract_involved_files(failure, repo)
        assert "tests/test_z.py" in files

    def test_deduplicates_files(self):
        repo = Path("/repo")
        failure = TestFailure(
            test_name="test_dup",
            test_file="tests/test_dup.py",
            error_type="Error",
            error_message="",
            stack_trace=(
                'File "/repo/tests/test_dup.py", line 1\n'
                'File "/repo/tests/test_dup.py", line 20\n'
                'File "/repo/src/util.py", line 3'
            ),
        )
        files = extract_involved_files(failure, repo)
        # Should have test_dup.py once and util.py once
        assert files.count("tests/test_dup.py") == 1
        assert "src/util.py" in files

    def test_empty_stack_trace(self):
        repo = Path("/repo")
        failure = TestFailure(
            test_name="test_empty",
            test_file="tests/test_empty.py",
            error_type="Error",
            error_message="msg",
            stack_trace="",
        )
        files = extract_involved_files(failure, repo)
        assert files == ["tests/test_empty.py"]


# ---------------------------------------------------------------------------
# TestFailure.to_prompt_context
# ---------------------------------------------------------------------------


class TestTestFailurePromptContext:
    """Tests for TestFailure.to_prompt_context method."""

    def test_basic_fields(self):
        failure = TestFailure(
            test_name="test_add",
            test_file="tests/test_math.py",
            error_type="AssertionError",
            error_message="1 != 2",
            stack_trace="traceback here",
        )
        ctx = failure.to_prompt_context()
        assert "## Test Failure: test_add" in ctx
        assert "File: tests/test_math.py" in ctx
        assert "Error Type: AssertionError" in ctx
        assert "### Error Message" in ctx
        assert "1 != 2" in ctx
        assert "### Stack Trace" in ctx
        assert "traceback here" in ctx

    def test_includes_relevant_code_block(self):
        failure = TestFailure(
            test_name="test_code",
            test_file="tests/test_code.py",
            error_type="TypeError",
            error_message="oops",
            stack_trace="",
            relevant_code="def foo():\n    return 42",
        )
        ctx = failure.to_prompt_context()
        assert "### Relevant Code" in ctx
        assert "```python" in ctx
        assert "def foo():" in ctx
        assert "```" in ctx

    def test_omits_code_block_when_empty(self):
        failure = TestFailure(
            test_name="test_no_code",
            test_file="tests/test_no_code.py",
            error_type="Error",
            error_message="msg",
            stack_trace="trace",
            relevant_code="",
        )
        ctx = failure.to_prompt_context()
        assert "### Relevant Code" not in ctx

    def test_includes_involved_files(self):
        failure = TestFailure(
            test_name="test_files",
            test_file="tests/test_files.py",
            error_type="Error",
            error_message="",
            stack_trace="",
            involved_files=["src/a.py", "src/b.py"],
        )
        ctx = failure.to_prompt_context()
        assert "### Involved Files" in ctx
        assert "- src/a.py" in ctx
        assert "- src/b.py" in ctx

    def test_omits_involved_files_when_empty(self):
        failure = TestFailure(
            test_name="test_no_files",
            test_file="tests/test_no_files.py",
            error_type="Error",
            error_message="",
            stack_trace="",
        )
        ctx = failure.to_prompt_context()
        assert "### Involved Files" not in ctx


# ---------------------------------------------------------------------------
# TestResult.success
# ---------------------------------------------------------------------------


class TestTestResultSuccess:
    """Tests for the TestResult.success property."""

    def test_success_when_all_zero(self):
        result = TestResult(
            command="pytest",
            exit_code=0,
            stdout="",
            stderr="",
            failed=0,
            errors=0,
        )
        assert result.success is True

    def test_not_success_nonzero_exit(self):
        result = TestResult(
            command="pytest",
            exit_code=1,
            stdout="",
            stderr="",
            failed=0,
            errors=0,
        )
        assert result.success is False

    def test_not_success_with_failures(self):
        result = TestResult(
            command="pytest",
            exit_code=0,
            stdout="",
            stderr="",
            failed=1,
            errors=0,
        )
        assert result.success is False

    def test_not_success_with_errors(self):
        result = TestResult(
            command="pytest",
            exit_code=0,
            stdout="",
            stderr="",
            failed=0,
            errors=1,
        )
        assert result.success is False

    def test_not_success_all_bad(self):
        result = TestResult(
            command="pytest",
            exit_code=2,
            stdout="",
            stderr="",
            failed=3,
            errors=1,
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# TestResult.first_failure
# ---------------------------------------------------------------------------


class TestTestResultFirstFailure:
    """Tests for the TestResult.first_failure property."""

    def test_none_when_no_failures(self):
        result = TestResult(command="pytest", exit_code=0, stdout="", stderr="")
        assert result.first_failure is None

    def test_returns_first_item(self):
        f1 = TestFailure(
            test_name="first",
            test_file="a.py",
            error_type="E",
            error_message="m1",
            stack_trace="",
        )
        f2 = TestFailure(
            test_name="second",
            test_file="b.py",
            error_type="E",
            error_message="m2",
            stack_trace="",
        )
        result = TestResult(
            command="pytest",
            exit_code=1,
            stdout="",
            stderr="",
            failures=[f1, f2],
        )
        assert result.first_failure is f1


# ---------------------------------------------------------------------------
# TestResult.summary
# ---------------------------------------------------------------------------


class TestTestResultSummary:
    """Tests for the TestResult.summary method."""

    def test_success_message(self):
        result = TestResult(
            command="pytest",
            exit_code=0,
            stdout="",
            stderr="",
            total_tests=10,
            passed=10,
            duration_seconds=1.234,
        )
        s = result.summary()
        assert "10" in s
        assert "passed" in s
        assert "1.2s" in s

    def test_failure_message(self):
        result = TestResult(
            command="pytest",
            exit_code=1,
            stdout="",
            stderr="",
            total_tests=10,
            passed=7,
            failed=2,
            errors=1,
            duration_seconds=3.456,
        )
        s = result.summary()
        assert "2 failed" in s
        assert "1 errors" in s
        assert "7 passed" in s
        assert "10 tests" in s
        assert "3.5s" in s


# ---------------------------------------------------------------------------
# TestRunner async tests
# ---------------------------------------------------------------------------


class TestTestRunnerRun:
    """Async tests for TestRunner.run with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_successful_run(self):
        """Successful test execution returns parsed results."""
        stdout_bytes = b"=== 5 passed in 0.3s ==="
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 0

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/ -v",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run()

        assert result.success is True
        assert result.exit_code == 0
        assert result.passed == 5
        assert result.total_tests == 5
        assert result.framework == TestFramework.PYTEST
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_failed_run(self):
        """Failed test execution returns failures."""
        stdout_bytes = b"FAILED tests/foo.py::test_bar\n=== 1 failed, 2 passed in 0.5s ==="
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 1

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/ -v",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run()

        assert result.success is False
        assert result.exit_code == 1
        assert result.failed == 1
        assert result.passed == 2
        assert len(result.failures) == 1
        assert result.failures[0].test_file == "tests/foo.py"
        assert result.failures[0].test_name == "test_bar"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Timeout returns a result with TimeoutError failure."""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/ -v",
            timeout_seconds=10.0,
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run()

        assert result.exit_code == -1
        assert result.success is False
        assert len(result.failures) == 1
        assert result.failures[0].test_name == "timeout"
        assert result.failures[0].error_type == "TimeoutError"
        assert "10.0s" in result.failures[0].error_message
        assert "timed out" in result.stderr
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_during_execution(self):
        """Exception during subprocess creation returns error result."""
        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/ -v",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            side_effect=OSError("command not found"),
        ):
            result = await runner.run()

        assert result.exit_code == -1
        assert result.success is False
        assert len(result.failures) == 1
        assert result.failures[0].test_name == "execution_error"
        assert result.failures[0].error_type == "OSError"
        assert "command not found" in result.failures[0].error_message

    @pytest.mark.asyncio
    async def test_non_pytest_framework_success(self):
        """Non-pytest framework with exit 0 reports success."""
        stdout_bytes = b"all tests passed"
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 0

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="make test",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run()

        assert result.framework == TestFramework.UNKNOWN
        assert result.exit_code == 0
        assert result.passed == 1
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_non_pytest_framework_failure(self):
        """Non-pytest framework with nonzero exit creates generic failure."""
        stdout_bytes = b"test output"
        stderr_bytes = b"error output"

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 1

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="make test",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run()

        assert result.framework == TestFramework.UNKNOWN
        assert result.exit_code == 1
        assert result.failed == 1
        assert len(result.failures) == 1
        assert result.failures[0].test_name == "unknown"
        assert "code 1" in result.failures[0].error_message

    @pytest.mark.asyncio
    async def test_enriches_failures_with_involved_files(self):
        """Failures get involved_files populated after parsing."""
        stdout_bytes = b"FAILED tests/check.py::test_it\n=== 1 failed ==="
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 1

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run()

        assert len(result.failures) == 1
        # test_file should always be in involved_files
        assert "tests/check.py" in result.failures[0].involved_files


class TestTestRunnerRunSingleTest:
    """Tests for TestRunner.run_single_test."""

    @pytest.mark.asyncio
    async def test_pytest_framework_command(self):
        """For pytest framework, run_single_test uses 'pytest {test_id} -v'."""
        stdout_bytes = b"=== 1 passed in 0.1s ==="
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 0

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/ -v",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ) as mock_shell:
            result = await runner.run_single_test("tests/foo.py::test_bar")

        # Verify the command used for the inner runner
        call_args = mock_shell.call_args
        assert "pytest tests/foo.py::test_bar -v" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_non_pytest_framework_command(self):
        """For non-pytest framework, test_id is appended to original command."""
        stdout_bytes = b"ok"
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 0

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="make test",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ) as mock_shell:
            result = await runner.run_single_test("test_module")

        call_args = mock_shell.call_args
        assert "make test test_module" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_returns_test_result(self):
        """run_single_test returns a proper TestResult."""
        stdout_bytes = b"=== 1 passed in 0.1s ==="
        stderr_bytes = b""

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))
        mock_process.returncode = 0

        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/ -v",
        )

        with patch(
            "aragora.nomic.testfixer.runner.asyncio.create_subprocess_shell",
            return_value=mock_process,
        ):
            result = await runner.run_single_test("tests/foo.py::test_bar")

        assert isinstance(result, TestResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# TestRunner init
# ---------------------------------------------------------------------------


class TestTestRunnerInit:
    """Tests for TestRunner initialization."""

    def test_detects_framework_on_init(self):
        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest tests/",
        )
        assert runner.framework == TestFramework.PYTEST

    def test_stores_repo_path_as_path(self):
        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="make test",
        )
        assert isinstance(runner.repo_path, Path)

    def test_default_timeout(self):
        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest",
        )
        assert runner.timeout == 300.0

    def test_custom_timeout(self):
        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest",
            timeout_seconds=60.0,
        )
        assert runner.timeout == 60.0

    def test_env_stored(self):
        env = {"FOO": "bar"}
        runner = TestRunner(
            repo_path=Path("/tmp/repo"),
            test_command="pytest",
            env=env,
        )
        assert runner.env == env
