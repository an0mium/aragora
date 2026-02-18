"""Tests for the DebugLoop module -- iterative test-failure-feedback-retry cycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.debug_loop import (
    DebugAttempt,
    DebugLoop,
    DebugLoopConfig,
    DebugLoopResult,
)
from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline


# ---------------------------------------------------------------------------
# 1. Config / dataclass tests (10 tests)
# ---------------------------------------------------------------------------


class TestDebugLoopConfig:
    def test_defaults(self):
        config = DebugLoopConfig()
        assert config.max_retries == 3
        assert config.test_timeout == 120
        assert config.max_failure_context_chars == 3000
        assert config.agent_timeout == 300

    def test_custom_values(self):
        config = DebugLoopConfig(
            max_retries=5,
            test_timeout=60,
            max_failure_context_chars=500,
            agent_timeout=180,
        )
        assert config.max_retries == 5
        assert config.test_timeout == 60
        assert config.max_failure_context_chars == 500
        assert config.agent_timeout == 180

    def test_single_override(self):
        config = DebugLoopConfig(max_retries=1)
        assert config.max_retries == 1
        assert config.test_timeout == 120  # default preserved


class TestDebugAttempt:
    def test_defaults(self):
        attempt = DebugAttempt(attempt_number=1, prompt="Do the thing")
        assert attempt.attempt_number == 1
        assert attempt.prompt == "Do the thing"
        assert attempt.tests_passed == 0
        assert attempt.tests_failed == 0
        assert attempt.test_output == ""
        assert attempt.success is False
        assert attempt.agent_stdout == ""
        assert attempt.agent_stderr == ""

    def test_populated_attempt(self):
        attempt = DebugAttempt(
            attempt_number=2,
            prompt="Fix auth",
            tests_passed=10,
            tests_failed=2,
            test_output="FAILED test_login",
            success=False,
            agent_stdout="Running...",
            agent_stderr="Warning: ...",
        )
        assert attempt.tests_passed == 10
        assert attempt.tests_failed == 2
        assert attempt.test_output == "FAILED test_login"
        assert attempt.agent_stdout == "Running..."


class TestDebugLoopResult:
    def test_defaults(self):
        result = DebugLoopResult(
            subtask_id="sub_1", success=False, total_attempts=0
        )
        assert result.subtask_id == "sub_1"
        assert result.success is False
        assert result.total_attempts == 0
        assert result.attempts == []
        assert result.final_tests_passed == 0
        assert result.final_tests_failed == 0
        assert result.final_files_changed == []

    def test_to_dict_serialization(self):
        attempt = DebugAttempt(
            attempt_number=1,
            prompt="fix",
            tests_passed=5,
            tests_failed=1,
            success=False,
        )
        result = DebugLoopResult(
            subtask_id="sub_42",
            success=False,
            total_attempts=1,
            attempts=[attempt],
            final_tests_passed=5,
            final_tests_failed=1,
            final_files_changed=["auth.py"],
        )
        d = result.to_dict()
        assert d["subtask_id"] == "sub_42"
        assert d["success"] is False
        assert d["total_attempts"] == 1
        assert d["final_tests_passed"] == 5
        assert d["final_tests_failed"] == 1
        assert d["final_files_changed"] == ["auth.py"]
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["attempt_number"] == 1
        assert d["attempts"][0]["tests_passed"] == 5
        assert d["attempts"][0]["tests_failed"] == 1
        assert d["attempts"][0]["success"] is False

    def test_to_dict_empty_attempts(self):
        result = DebugLoopResult(
            subtask_id="x", success=True, total_attempts=0
        )
        d = result.to_dict()
        assert d["attempts"] == []
        assert d["final_files_changed"] == []

    def test_with_populated_attempts(self):
        a1 = DebugAttempt(attempt_number=1, prompt="p1", tests_passed=3, tests_failed=2, success=False)
        a2 = DebugAttempt(attempt_number=2, prompt="p2", tests_passed=5, tests_failed=0, success=True)
        result = DebugLoopResult(
            subtask_id="sub_multi",
            success=True,
            total_attempts=2,
            attempts=[a1, a2],
            final_tests_passed=5,
            final_tests_failed=0,
            final_files_changed=["a.py", "b.py"],
        )
        assert len(result.attempts) == 2
        assert result.attempts[0].success is False
        assert result.attempts[1].success is True
        assert result.final_files_changed == ["a.py", "b.py"]

    def test_attempts_list_isolation(self):
        """Each instance should have its own attempts list."""
        r1 = DebugLoopResult(subtask_id="a", success=False, total_attempts=0)
        r2 = DebugLoopResult(subtask_id="b", success=False, total_attempts=0)
        r1.attempts.append(DebugAttempt(attempt_number=1, prompt="x"))
        assert len(r2.attempts) == 0


# ---------------------------------------------------------------------------
# 2. Single attempt -- pass on first try (10 tests)
# ---------------------------------------------------------------------------


class TestSingleAttemptSuccess:
    @pytest.mark.asyncio
    async def test_pass_on_first_try(self):
        """Agent succeeds, tests pass -> success=True, total_attempts=1."""
        loop = DebugLoop()
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("ok", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 10, "failed": 0, "output": "10 passed"},
            ),
            patch.object(loop, "_get_changed_files", return_value=["fix.py"]),
        ):
            result = await loop.execute_with_retry(
                instruction="Fix the bug",
                worktree_path="/tmp/wt",
                test_scope=["tests/auth/"],
                subtask_id="sub_1",
            )

        assert result.success is True
        assert result.total_attempts == 1
        assert result.final_tests_passed == 10
        assert result.final_tests_failed == 0
        assert result.final_files_changed == ["fix.py"]
        assert len(result.attempts) == 1
        assert result.attempts[0].success is True

    @pytest.mark.asyncio
    async def test_no_tests_found_is_not_success(self):
        """0 passed, 0 failed -> not success (needs >0 passed)."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 0, "failed": 0, "output": "no tests ran"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Do something", worktree_path="/tmp/wt"
            )

        assert result.success is False
        assert result.total_attempts == 1
        assert result.final_tests_passed == 0

    @pytest.mark.asyncio
    async def test_agent_import_error_still_runs_tests(self):
        """When ClaudeCodeHarness import fails, tests still execute."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        # _run_agent returns empty on ImportError internally
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 3, "failed": 0, "output": "3 passed"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Fix it", worktree_path="/tmp/wt"
            )

        assert result.success is True
        assert result.final_tests_passed == 3

    @pytest.mark.asyncio
    async def test_agent_output_captured(self):
        """Agent stdout/stderr captured in attempt."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(
                loop, "_run_agent", new_callable=AsyncMock,
                return_value=("agent output here", "some warning"),
            ),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "1 passed"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Task", worktree_path="/tmp/wt"
            )

        assert result.attempts[0].agent_stdout == "agent output here"
        assert result.attempts[0].agent_stderr == "some warning"

    @pytest.mark.asyncio
    async def test_prompt_truncated_in_attempt(self):
        """The prompt stored in DebugAttempt is truncated to 500 chars."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        long_prompt = "x" * 1000
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "ok"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction=long_prompt, worktree_path="/tmp/wt"
            )

        assert len(result.attempts[0].prompt) == 500

    @pytest.mark.asyncio
    async def test_agent_stdout_truncated(self):
        """Agent stdout truncated to 1000 chars."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        long_stdout = "y" * 2000
        with (
            patch.object(
                loop, "_run_agent", new_callable=AsyncMock,
                return_value=(long_stdout, ""),
            ),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "ok"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="task", worktree_path="/tmp/wt"
            )

        assert len(result.attempts[0].agent_stdout) == 1000

    @pytest.mark.asyncio
    async def test_agent_stderr_truncated(self):
        """Agent stderr truncated to 500 chars."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        long_stderr = "z" * 800
        with (
            patch.object(
                loop, "_run_agent", new_callable=AsyncMock,
                return_value=("", long_stderr),
            ),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "ok"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="task", worktree_path="/tmp/wt"
            )

        assert len(result.attempts[0].agent_stderr) == 500

    @pytest.mark.asyncio
    async def test_subtask_id_passed_through(self):
        """subtask_id flows into the result."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "1 passed"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Fix", worktree_path="/tmp/wt", subtask_id="my_sub"
            )

        assert result.subtask_id == "my_sub"

    @pytest.mark.asyncio
    async def test_test_scope_passed_to_run_tests(self):
        """test_scope is forwarded to _run_tests."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        scope = ["tests/auth/", "tests/rbac/"]
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "ok"},
            ) as mock_run_tests,
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            await loop.execute_with_retry(
                instruction="Fix", worktree_path="/tmp/wt", test_scope=scope
            )

        mock_run_tests.assert_called_once_with("/tmp/wt", scope)

    @pytest.mark.asyncio
    async def test_default_config_when_none(self):
        """DebugLoop(config=None) uses default DebugLoopConfig."""
        loop = DebugLoop(config=None)
        assert loop.config.max_retries == 3
        assert loop.config.test_timeout == 120


# ---------------------------------------------------------------------------
# 3. Retry behavior (15 tests)
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_fail_then_pass_on_retry(self):
        """Fail on first attempt, pass on second -> success, total_attempts=2."""
        loop = DebugLoop(DebugLoopConfig(max_retries=3))
        test_results = [
            {"passed": 3, "failed": 2, "output": "2 failed"},
            {"passed": 5, "failed": 0, "output": "5 passed"},
        ]
        call_count = 0

        async def mock_run_tests(wt, scope):
            nonlocal call_count
            r = test_results[call_count]
            call_count += 1
            return r

        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(loop, "_run_tests", side_effect=mock_run_tests),
            patch.object(loop, "_get_changed_files", return_value=["fixed.py"]),
        ):
            result = await loop.execute_with_retry(
                instruction="Fix tests", worktree_path="/tmp/wt"
            )

        assert result.success is True
        assert result.total_attempts == 2
        assert result.final_tests_passed == 5
        assert result.final_tests_failed == 0

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """All retries fail -> success=False, total_attempts=max_retries."""
        loop = DebugLoop(DebugLoopConfig(max_retries=3))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 2, "failed": 1, "output": "1 failed"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Unfixable", worktree_path="/tmp/wt"
            )

        assert result.success is False
        assert result.total_attempts == 3
        assert len(result.attempts) == 3

    @pytest.mark.asyncio
    async def test_retry_prompt_includes_failure_output(self):
        """The retry prompt fed to _run_agent includes test failure output."""
        loop = DebugLoop(DebugLoopConfig(max_retries=2))
        prompts_received: list[str] = []

        async def capture_agent(prompt, wt):
            prompts_received.append(prompt)
            return ("", "")

        test_results = [
            {"passed": 1, "failed": 1, "output": "FAILED test_auth_login"},
            {"passed": 2, "failed": 0, "output": "2 passed"},
        ]
        call_idx = 0

        async def mock_tests(wt, scope):
            nonlocal call_idx
            r = test_results[call_idx]
            call_idx += 1
            return r

        with (
            patch.object(loop, "_run_agent", side_effect=capture_agent),
            patch.object(loop, "_run_tests", side_effect=mock_tests),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            await loop.execute_with_retry(
                instruction="Fix auth", worktree_path="/tmp/wt"
            )

        assert len(prompts_received) == 2
        # First call gets the original instruction
        assert prompts_received[0] == "Fix auth"
        # Second call gets a retry prompt with failure context
        assert "RETRY ATTEMPT" in prompts_received[1]
        assert "FAILED test_auth_login" in prompts_received[1]

    @pytest.mark.asyncio
    async def test_retry_prompt_includes_original_instruction(self):
        """The retry prompt includes the original objective."""
        loop = DebugLoop(DebugLoopConfig(max_retries=2))
        prompts_received: list[str] = []

        async def capture_agent(prompt, wt):
            prompts_received.append(prompt)
            return ("", "")

        test_results = [
            {"passed": 0, "failed": 1, "output": "error"},
            {"passed": 1, "failed": 0, "output": "1 passed"},
        ]
        idx = 0

        async def mock_tests(wt, scope):
            nonlocal idx
            r = test_results[idx]
            idx += 1
            return r

        with (
            patch.object(loop, "_run_agent", side_effect=capture_agent),
            patch.object(loop, "_run_tests", side_effect=mock_tests),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            await loop.execute_with_retry(
                instruction="Refactor the database module",
                worktree_path="/tmp/wt",
            )

        assert "Refactor the database module" in prompts_received[1]

    @pytest.mark.asyncio
    async def test_test_output_truncated_at_max_chars(self):
        """Retry prompt truncates test output at max_failure_context_chars."""
        loop = DebugLoop(DebugLoopConfig(max_retries=2, max_failure_context_chars=50))
        prompts_received: list[str] = []

        async def capture_agent(prompt, wt):
            prompts_received.append(prompt)
            return ("", "")

        long_output = "E" * 200
        test_results = [
            {"passed": 0, "failed": 1, "output": long_output},
            {"passed": 1, "failed": 0, "output": "ok"},
        ]
        idx = 0

        async def mock_tests(wt, scope):
            nonlocal idx
            r = test_results[idx]
            idx += 1
            return r

        with (
            patch.object(loop, "_run_agent", side_effect=capture_agent),
            patch.object(loop, "_run_tests", side_effect=mock_tests),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            await loop.execute_with_retry(
                instruction="Fix it", worktree_path="/tmp/wt"
            )

        retry_prompt = prompts_received[1]
        assert "... [truncated]" in retry_prompt
        # The full 200 chars should NOT appear
        assert long_output not in retry_prompt

    @pytest.mark.asyncio
    async def test_each_attempt_recorded(self):
        """Each attempt is appended to result.attempts."""
        loop = DebugLoop(DebugLoopConfig(max_retries=3))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 0, "failed": 1, "output": "fail"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="x", worktree_path="/tmp/wt"
            )

        assert len(result.attempts) == 3
        for i, attempt in enumerate(result.attempts, 1):
            assert attempt.attempt_number == i

    @pytest.mark.asyncio
    async def test_final_tests_reflect_last_attempt_on_failure(self):
        """On failure, final_tests_passed/failed reflect the last attempt."""
        loop = DebugLoop(DebugLoopConfig(max_retries=2))
        test_results = [
            {"passed": 1, "failed": 3, "output": "3 failed"},
            {"passed": 2, "failed": 1, "output": "1 failed"},
        ]
        idx = 0

        async def mock_tests(wt, scope):
            nonlocal idx
            r = test_results[idx]
            idx += 1
            return r

        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(loop, "_run_tests", side_effect=mock_tests),
            patch.object(loop, "_get_changed_files", return_value=["partial.py"]),
        ):
            result = await loop.execute_with_retry(
                instruction="x", worktree_path="/tmp/wt"
            )

        assert result.success is False
        assert result.final_tests_passed == 2
        assert result.final_tests_failed == 1

    @pytest.mark.asyncio
    async def test_files_changed_on_failure(self):
        """_get_changed_files is called even on failure."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 0, "failed": 1, "output": "fail"},
            ),
            patch.object(
                loop, "_get_changed_files", return_value=["broken.py"]
            ) as mock_changed,
        ):
            result = await loop.execute_with_retry(
                instruction="x", worktree_path="/tmp/wt"
            )

        assert result.final_files_changed == ["broken.py"]
        mock_changed.assert_called_with("/tmp/wt")

    @pytest.mark.asyncio
    async def test_max_retries_one_single_attempt(self):
        """max_retries=1 means exactly one attempt, no retries."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 0, "failed": 1, "output": "fail"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="x", worktree_path="/tmp/wt"
            )

        assert result.total_attempts == 1
        assert len(result.attempts) == 1

    @pytest.mark.asyncio
    async def test_success_on_third_attempt(self):
        """Fail twice, succeed on third -> success=True, total_attempts=3."""
        loop = DebugLoop(DebugLoopConfig(max_retries=3))
        test_results = [
            {"passed": 0, "failed": 2, "output": "2 failed"},
            {"passed": 1, "failed": 1, "output": "1 failed"},
            {"passed": 3, "failed": 0, "output": "3 passed"},
        ]
        idx = 0

        async def mock_tests(wt, scope):
            nonlocal idx
            r = test_results[idx]
            idx += 1
            return r

        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(loop, "_run_tests", side_effect=mock_tests),
            patch.object(loop, "_get_changed_files", return_value=["final.py"]),
        ):
            result = await loop.execute_with_retry(
                instruction="Hard fix", worktree_path="/tmp/wt"
            )

        assert result.success is True
        assert result.total_attempts == 3
        assert result.final_tests_passed == 3

    @pytest.mark.asyncio
    async def test_no_retry_after_success(self):
        """Success on first attempt stops the loop -- no further attempts."""
        loop = DebugLoop(DebugLoopConfig(max_retries=5))
        call_count = 0

        async def counting_tests(wt, scope):
            nonlocal call_count
            call_count += 1
            return {"passed": 1, "failed": 0, "output": "ok"}

        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(loop, "_run_tests", side_effect=counting_tests),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Easy fix", worktree_path="/tmp/wt"
            )

        assert result.success is True
        assert result.total_attempts == 1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_no_test_scope(self):
        """test_scope=None still works (pytest runs without specific scope)."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "ok"},
            ) as mock_tests,
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Fix", worktree_path="/tmp/wt", test_scope=None
            )

        mock_tests.assert_called_once_with("/tmp/wt", None)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_subtask_id_default(self):
        """subtask_id defaults to empty string."""
        loop = DebugLoop(DebugLoopConfig(max_retries=1))
        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(
                loop, "_run_tests", new_callable=AsyncMock,
                return_value={"passed": 1, "failed": 0, "output": "ok"},
            ),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Fix", worktree_path="/tmp/wt"
            )

        assert result.subtask_id == ""

    @pytest.mark.asyncio
    async def test_progressive_improvement_across_retries(self):
        """Tests show progressive improvement across retries."""
        loop = DebugLoop(DebugLoopConfig(max_retries=3))
        test_results = [
            {"passed": 1, "failed": 5, "output": "5 failed"},
            {"passed": 4, "failed": 2, "output": "2 failed"},
            {"passed": 6, "failed": 0, "output": "6 passed"},
        ]
        idx = 0

        async def mock_tests(wt, scope):
            nonlocal idx
            r = test_results[idx]
            idx += 1
            return r

        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("", "")),
            patch.object(loop, "_run_tests", side_effect=mock_tests),
            patch.object(loop, "_get_changed_files", return_value=[]),
        ):
            result = await loop.execute_with_retry(
                instruction="Progressive fix", worktree_path="/tmp/wt"
            )

        assert result.success is True
        assert result.attempts[0].tests_failed == 5
        assert result.attempts[1].tests_failed == 2
        assert result.attempts[2].tests_failed == 0


# ---------------------------------------------------------------------------
# 4. Retry prompt construction (8 tests)
# ---------------------------------------------------------------------------


class TestBuildRetryPrompt:
    def test_includes_retry_attempt_header(self):
        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=3, tests_failed=2,
            test_output="FAILED test_x",
        )
        prompt = loop._build_retry_prompt("original", attempt)
        assert "RETRY ATTEMPT" in prompt

    def test_includes_attempt_number_incremented(self):
        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=2, prompt="p", tests_passed=1, tests_failed=1,
            test_output="fail",
        )
        prompt = loop._build_retry_prompt("orig", attempt)
        assert "RETRY ATTEMPT 3" in prompt

    def test_includes_original_objective(self):
        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=0, tests_failed=1,
            test_output="err",
        )
        prompt = loop._build_retry_prompt("Refactor authentication module", attempt)
        assert "Refactor authentication module" in prompt
        assert "ORIGINAL OBJECTIVE" in prompt

    def test_includes_test_count_summary(self):
        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=7, tests_failed=3,
            test_output="output",
        )
        prompt = loop._build_retry_prompt("orig", attempt)
        assert "7 tests passed" in prompt
        assert "3 tests failed" in prompt

    def test_includes_test_output(self):
        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=0, tests_failed=1,
            test_output="FAILED tests/auth/test_login.py::test_invalid_token - AssertionError",
        )
        prompt = loop._build_retry_prompt("orig", attempt)
        assert "FAILED tests/auth/test_login.py::test_invalid_token" in prompt

    def test_truncates_long_test_output(self):
        loop = DebugLoop(DebugLoopConfig(max_failure_context_chars=100))
        long_output = "FAIL " * 200  # 1000 chars
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=0, tests_failed=5,
            test_output=long_output,
        )
        prompt = loop._build_retry_prompt("orig", attempt)
        assert "... [truncated]" in prompt
        # Full output should NOT be in the prompt
        assert long_output not in prompt

    def test_references_fix_not_revert(self):
        """Prompt should instruct to fix, not revert."""
        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=0, tests_failed=1,
            test_output="err",
        )
        prompt = loop._build_retry_prompt("orig", attempt)
        assert "Fix" in prompt or "fix" in prompt
        assert "Do not revert" in prompt or "not revert" in prompt

    def test_original_instruction_truncated_to_1000(self):
        """Original instruction in retry prompt is truncated to 1000 chars."""
        loop = DebugLoop()
        long_instruction = "A" * 2000
        attempt = DebugAttempt(
            attempt_number=1, prompt="p", tests_passed=0, tests_failed=1,
            test_output="err",
        )
        prompt = loop._build_retry_prompt(long_instruction, attempt)
        # The section between ORIGINAL OBJECTIVE and TEST FAILURES should
        # contain at most 1000 A's
        objective_section = prompt.split("ORIGINAL OBJECTIVE:")[1].split("TEST FAILURES")[0]
        a_count = objective_section.count("A")
        assert a_count == 1000


# ---------------------------------------------------------------------------
# 5. Pipeline integration (7 tests)
# ---------------------------------------------------------------------------


class TestPipelineDebugLoopIntegration:
    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """_execute_with_debug_loop returns None when enable_debug_loop=False."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(enable_debug_loop=False)
        )
        mock_instruction = MagicMock()
        result = await pipeline._execute_with_debug_loop(
            mock_instruction, "/tmp/wt"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_debug_loop_runs_with_require_approval(self):
        """Debug loop runs even when require_approval=True (gate removed).

        The approval gate was removed so the debug loop is reachable under
        default config. This test verifies it attempts execution rather
        than short-circuiting to None.
        """
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_debug_loop=True,
                require_approval=True,
                autonomous=False,
            )
        )
        mock_instruction = MagicMock()
        mock_instruction.to_agent_prompt.return_value = "test prompt"

        # Mock DebugLoop to avoid real agent/harness execution
        with patch(
            "aragora.nomic.self_improve.DebugLoop",
            side_effect=ImportError("mocked"),
        ) if False else patch(
            "aragora.nomic.debug_loop.DebugLoop"
        ) as MockDebug:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.final_files_changed = []
            mock_result.final_tests_passed = 1
            mock_result.final_tests_failed = 0
            mock_result.total_attempts = 1
            MockDebug.return_value.execute_with_retry = AsyncMock(
                return_value=mock_result
            )
            result = await pipeline._execute_with_debug_loop(
                mock_instruction, "/tmp/wt"
            )

        # Debug loop now runs and returns a result dict (not None)
        assert result is not None
        assert isinstance(result, dict)
        assert result["debug_loop_success"] is True

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        """Returns None when DebugLoop import fails."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_debug_loop=True,
                require_approval=False,
            )
        )
        mock_instruction = MagicMock()
        mock_instruction.to_agent_prompt.return_value = "prompt"
        mock_instruction.subtask_id = "sub_1"

        with patch(
            "aragora.nomic.debug_loop.DebugLoop",
            side_effect=ImportError("no module"),
        ):
            # Patching the class reference triggers ImportError on construction
            pass

        # Actually patch at the import site within the method
        with patch.dict("sys.modules", {"aragora.nomic.debug_loop": None}):
            result = await pipeline._execute_with_debug_loop(
                mock_instruction, "/tmp/wt"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_result_dict_on_success(self):
        """Returns a result dict when debug loop succeeds."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_debug_loop=True,
                require_approval=False,
            )
        )
        mock_instruction = MagicMock()
        mock_instruction.to_agent_prompt.return_value = "fix the auth bug"
        mock_instruction.subtask_id = "sub_1"

        mock_debug_result = MagicMock()
        mock_debug_result.final_files_changed = ["auth.py"]
        mock_debug_result.final_tests_passed = 10
        mock_debug_result.final_tests_failed = 0
        mock_debug_result.total_attempts = 2
        mock_debug_result.success = True

        mock_debug_loop = MagicMock()
        mock_debug_loop.execute_with_retry = AsyncMock(return_value=mock_debug_result)

        with patch("aragora.nomic.debug_loop.DebugLoop", return_value=mock_debug_loop):
            with patch("aragora.nomic.debug_loop.DebugLoopConfig"):
                result = await pipeline._execute_with_debug_loop(
                    mock_instruction, "/tmp/wt"
                )

        assert result is not None
        assert result["files_changed"] == ["auth.py"]
        assert result["tests_passed"] == 10
        assert result["tests_failed"] == 0
        assert result["debug_loop_attempts"] == 2
        assert result["debug_loop_success"] is True

    @pytest.mark.asyncio
    async def test_autonomous_bypasses_approval_gate(self):
        """When autonomous=True, require_approval does not block debug loop."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_debug_loop=True,
                require_approval=True,
                autonomous=True,
            )
        )
        mock_instruction = MagicMock()
        mock_instruction.to_agent_prompt.return_value = "prompt"
        mock_instruction.subtask_id = "sub_1"

        mock_debug_result = MagicMock()
        mock_debug_result.final_files_changed = []
        mock_debug_result.final_tests_passed = 1
        mock_debug_result.final_tests_failed = 0
        mock_debug_result.total_attempts = 1
        mock_debug_result.success = True

        mock_debug_loop = MagicMock()
        mock_debug_loop.execute_with_retry = AsyncMock(return_value=mock_debug_result)

        with patch("aragora.nomic.debug_loop.DebugLoop", return_value=mock_debug_loop):
            with patch("aragora.nomic.debug_loop.DebugLoopConfig"):
                result = await pipeline._execute_with_debug_loop(
                    mock_instruction, "/tmp/wt"
                )

        assert result is not None
        assert result["debug_loop_success"] is True

    @pytest.mark.asyncio
    async def test_falls_back_to_dispatch_when_debug_loop_returns_none(self):
        """When debug loop returns None, pipeline falls back to dispatch_to_claude_code."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_debug_loop=False,  # disabled => returns None
                require_approval=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                use_meta_planner=False,
            )
        )

        mock_instruction = MagicMock()
        mock_instruction.subtask_id = "sub_fb"
        mock_instruction.to_agent_prompt.return_value = "prompt"
        mock_instruction.to_dict.return_value = {}
        mock_instruction.worktree_path = "/tmp/wt"

        dispatch_result = {
            "files_changed": ["fallback.py"],
            "tests_passed": 3,
            "tests_failed": 0,
        }

        with (
            patch(
                "aragora.nomic.execution_bridge.ExecutionBridge"
            ) as MockBridge,
            patch.object(
                pipeline, "_dispatch_to_claude_code",
                new_callable=AsyncMock, return_value=dispatch_result,
            ) as mock_dispatch,
            patch.object(
                pipeline, "_write_instruction_to_worktree",
                return_value=True,
            ),
        ):
            MockBridge.return_value.create_instruction.return_value = mock_instruction
            result = await pipeline._execute_single("some task", "cycle_1")

        # debug loop is disabled so dispatch should have been called
        mock_dispatch.assert_called_once()
        assert result["files_changed"] == ["fallback.py"]

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none(self):
        """RuntimeError inside debug loop returns None gracefully."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_debug_loop=True,
                require_approval=False,
            )
        )
        mock_instruction = MagicMock()
        mock_instruction.to_agent_prompt.return_value = "prompt"
        mock_instruction.subtask_id = "sub_err"

        mock_debug_loop = MagicMock()
        mock_debug_loop.execute_with_retry = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with patch("aragora.nomic.debug_loop.DebugLoop", return_value=mock_debug_loop):
            with patch("aragora.nomic.debug_loop.DebugLoopConfig"):
                result = await pipeline._execute_with_debug_loop(
                    mock_instruction, "/tmp/wt"
                )

        assert result is None


# ---------------------------------------------------------------------------
# 6. Test scope inference (5 tests)
# ---------------------------------------------------------------------------


class TestInferTestScope:
    def test_maps_aragora_to_tests(self):
        """aragora/foo/ maps to tests/foo/."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = ["aragora/auth/oidc.py", "aragora/auth/mfa.py"]
        del subtask.goal

        result = pipeline._infer_test_scope(subtask)
        assert "tests/auth" in result

    def test_preserves_tests_prefix(self):
        """Paths already starting with tests/ are kept as-is."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = ["tests/rbac/test_checker.py"]
        del subtask.goal

        result = pipeline._infer_test_scope(subtask)
        assert "tests/rbac/test_checker.py" in result

    def test_empty_file_hints_returns_empty(self):
        """No file_hints -> empty list."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = []
        del subtask.goal

        result = pipeline._infer_test_scope(subtask)
        assert result == []

    def test_deduplicates_test_dirs(self):
        """Multiple files in same module -> single test dir."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = [
            "aragora/auth/oidc.py",
            "aragora/auth/mfa.py",
            "aragora/auth/sessions.py",
        ]
        del subtask.goal

        result = pipeline._infer_test_scope(subtask)
        assert result.count("tests/auth") == 1

    def test_falls_back_to_goal_file_hints(self):
        """Uses subtask.goal.file_hints when file_scope is absent."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock(spec=[])  # no attributes by default

        goal = MagicMock()
        goal.file_hints = ["aragora/billing/cost_tracker.py"]
        subtask.goal = goal

        result = pipeline._infer_test_scope(subtask)
        assert "tests/billing" in result


# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------


class TestDebugLoopImports:
    def test_all_exports(self):
        """All __all__ exports are importable."""
        from aragora.nomic.debug_loop import __all__

        assert "DebugLoop" in __all__
        assert "DebugLoopConfig" in __all__
        assert "DebugAttempt" in __all__
        assert "DebugLoopResult" in __all__
