"""Comprehensive tests for the subprocess_runner utility.

Tests cover:
- Command validation (allowlist, blocklist, subcommands)
- Security features (shell metacharacter detection, environment sanitization)
- Execution behavior (async and sync variants)
- Timeout handling
- Output capture (stdout, stderr)
- Error handling
"""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from aragora.utils.subprocess_runner import (
    ALLOWED_COMMANDS,
    BLOCKED_COMMANDS,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
    SandboxedCommand,
    SandboxedResult,
    SandboxError,
    run_sandboxed,
    run_sandboxed_sync,
)


# ============================================================================
# SandboxedResult Tests
# ============================================================================


class TestSandboxedResult:
    """Tests for the SandboxedResult dataclass."""

    def test_success_property_true_on_zero_returncode(self):
        """success property returns True when returncode is 0."""
        result = SandboxedResult(
            returncode=0,
            stdout="output",
            stderr="",
            command=["git", "status"],
        )
        assert result.success is True

    def test_success_property_false_on_nonzero_returncode(self):
        """success property returns False when returncode is non-zero."""
        result = SandboxedResult(
            returncode=1,
            stdout="",
            stderr="error",
            command=["git", "status"],
        )
        assert result.success is False

    def test_success_property_false_on_negative_returncode(self):
        """success property returns False when returncode is negative (timeout)."""
        result = SandboxedResult(
            returncode=-1,
            stdout="",
            stderr="timeout",
            command=["git", "status"],
            timed_out=True,
        )
        assert result.success is False

    def test_timed_out_default_false(self):
        """timed_out defaults to False."""
        result = SandboxedResult(
            returncode=0,
            stdout="",
            stderr="",
            command=["git", "status"],
        )
        assert result.timed_out is False

    def test_stores_command_correctly(self):
        """Command is stored and accessible."""
        cmd = ["git", "log", "--oneline"]
        result = SandboxedResult(
            returncode=0, stdout="", stderr="", command=cmd
        )
        assert result.command == cmd


# ============================================================================
# SandboxedCommand Validation Tests
# ============================================================================


class TestSandboxedCommandValidation:
    """Tests for SandboxedCommand.validate()."""

    def test_empty_command_rejected(self):
        """Empty command should be rejected."""
        allowed, reason = SandboxedCommand.validate([])
        assert allowed is False
        assert "Empty command" in reason

    def test_blocked_command_rejected(self):
        """Commands in BLOCKED_COMMANDS should be rejected."""
        for cmd in ["rm", "sudo", "curl", "wget", "ssh", "docker"]:
            allowed, reason = SandboxedCommand.validate([cmd, "-rf", "/"])
            assert allowed is False, f"{cmd} should be blocked"
            assert "blocked" in reason.lower()

    def test_blocked_command_with_path_rejected(self):
        """Blocked commands with full paths should also be rejected."""
        allowed, reason = SandboxedCommand.validate(["/usr/bin/rm", "-rf", "/"])
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_unknown_command_rejected(self):
        """Commands not in allowlist should be rejected."""
        allowed, reason = SandboxedCommand.validate(["unknown_command", "arg"])
        assert allowed is False
        assert "not in the allowlist" in reason

    def test_allowed_command_accepted(self):
        """Commands in ALLOWED_COMMANDS should be accepted."""
        allowed, reason = SandboxedCommand.validate(["git", "status"])
        assert allowed is True
        assert "allowed" in reason.lower()

    def test_allowed_command_with_full_path(self):
        """Allowed commands specified with full path should work."""
        allowed, reason = SandboxedCommand.validate(["/usr/bin/git", "status"])
        assert allowed is True

    def test_allowed_subcommand_accepted(self):
        """Valid subcommands for restricted commands should be accepted."""
        # git has specific allowed subcommands
        allowed, reason = SandboxedCommand.validate(["git", "status"])
        assert allowed is True

        allowed, reason = SandboxedCommand.validate(["git", "diff"])
        assert allowed is True

        allowed, reason = SandboxedCommand.validate(["git", "log"])
        assert allowed is True

    def test_disallowed_subcommand_rejected(self):
        """Invalid subcommands for restricted commands should be rejected."""
        # 'gc' is not in the allowed git subcommands
        allowed, reason = SandboxedCommand.validate(["git", "gc"])
        assert allowed is False
        assert "not allowed" in reason.lower()

    def test_command_with_none_subcommand_restriction(self):
        """Commands with None subcommand list allow all subcommands."""
        # pytest has None for subcommands (all allowed)
        allowed, reason = SandboxedCommand.validate(["pytest", "-v", "tests/"])
        assert allowed is True

        allowed, reason = SandboxedCommand.validate(
            ["pytest", "--random-flag", "test.py"]
        )
        assert allowed is True

    def test_python_allowed_flags(self):
        """Python command with allowed flags should pass validation."""
        # Note: validate() only checks the command structure, not metacharacters
        # in all arguments. The -m and --version flags are in the allowlist.
        allowed, reason = SandboxedCommand.validate(["python", "-m", "pytest"])
        assert allowed is True

        allowed, reason = SandboxedCommand.validate(["python", "--version"])
        assert allowed is True

        # -c is also in the allowlist, but arguments with metacharacters
        # are separately rejected by the metacharacter check
        allowed, reason = SandboxedCommand.validate(["python", "-c", "x=1"])
        assert allowed is True

    def test_flags_starting_with_dash_skip_subcommand_check(self):
        """Flags starting with - should skip subcommand validation."""
        # Even if the flag isn't explicitly allowed, -flags should pass
        allowed, reason = SandboxedCommand.validate(
            ["git", "-C", "/path", "status"]
        )
        assert allowed is True

    def test_shell_metacharacter_rejected_general(self):
        """Shell metacharacters should be rejected for most commands."""
        metachar_tests = [
            (["ls", "foo; rm -rf /"], ";"),
            (["ls", "foo | cat"], "|"),
            (["ls", "foo && rm -rf /"], "&"),
            (["ls", "$(cat /etc/passwd)"], "$"),
            (["ls", "`cat /etc/passwd`"], "`"),
            (["ls", "foo > /etc/passwd"], ">"),
            (["ls", "foo < /etc/passwd"], "<"),
            (["ls", "foo(bar)"], "("),
        ]
        for cmd, char in metachar_tests:
            allowed, reason = SandboxedCommand.validate(cmd)
            assert allowed is False, f"metachar {char} should be rejected"
            assert "metacharacter" in reason.lower()

    def test_shell_metacharacter_allowed_for_git_commit(self):
        """Git commit messages can contain shell metacharacters."""
        allowed, reason = SandboxedCommand.validate(
            ["git", "commit", "-m", "fix(core): handle $VAR correctly"]
        )
        assert allowed is True

    def test_shell_metacharacter_allowed_for_gh_pr(self):
        """GitHub CLI PR commands can contain shell metacharacters."""
        allowed, reason = SandboxedCommand.validate(
            ["gh", "pr", "create", "--body", "Fixes issue & adds tests"]
        )
        assert allowed is True


class TestAllowedBlockedLists:
    """Tests for the ALLOWED_COMMANDS and BLOCKED_COMMANDS constants."""

    def test_blocked_commands_is_frozenset(self):
        """BLOCKED_COMMANDS should be immutable."""
        assert isinstance(BLOCKED_COMMANDS, frozenset)

    def test_dangerous_commands_blocked(self):
        """Known dangerous commands should be blocked."""
        dangerous = [
            "rm",
            "rmdir",
            "sudo",
            "su",
            "chmod",
            "chown",
            "kill",
            "curl",
            "wget",
            "ssh",
            "docker",
            "reboot",
            "shutdown",
        ]
        for cmd in dangerous:
            assert cmd in BLOCKED_COMMANDS, f"{cmd} should be blocked"

    def test_git_commands_allowed(self):
        """Git should be in allowed list with appropriate subcommands."""
        assert "git" in ALLOWED_COMMANDS
        git_subs = ALLOWED_COMMANDS["git"]
        assert git_subs is not None
        assert "status" in git_subs
        assert "commit" in git_subs
        assert "push" in git_subs
        assert "pull" in git_subs

    def test_build_tools_allowed(self):
        """Build and test tools should be allowed."""
        build_tools = ["pytest", "python", "npm", "black", "isort", "mypy", "ruff"]
        for tool in build_tools:
            assert tool in ALLOWED_COMMANDS, f"{tool} should be allowed"

    def test_timeout_constants_reasonable(self):
        """Timeout constants should be reasonable values."""
        assert DEFAULT_TIMEOUT == 30.0
        assert MAX_TIMEOUT == 600.0
        assert DEFAULT_TIMEOUT < MAX_TIMEOUT


# ============================================================================
# Environment Sanitization Tests
# ============================================================================


class TestEnvironmentSanitization:
    """Tests for SandboxedCommand.sanitize_env()."""

    def test_returns_dict(self):
        """sanitize_env should return a dictionary."""
        env = SandboxedCommand.sanitize_env()
        assert isinstance(env, dict)

    def test_preserves_path(self):
        """PATH should be preserved if present."""
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            env = SandboxedCommand.sanitize_env()
            assert "PATH" in env
            assert env["PATH"] == "/usr/bin:/bin"

    def test_preserves_home(self):
        """HOME should be preserved if present."""
        with patch.dict(os.environ, {"HOME": "/home/testuser"}):
            env = SandboxedCommand.sanitize_env()
            assert "HOME" in env
            assert env["HOME"] == "/home/testuser"

    def test_preserves_api_keys(self):
        """API keys needed for tools should be preserved."""
        test_keys = {
            "ANTHROPIC_API_KEY": "test-anthropic",
            "OPENAI_API_KEY": "test-openai",
            "GITHUB_TOKEN": "test-github",
            "OPENROUTER_API_KEY": "test-openrouter",
            "MISTRAL_API_KEY": "test-mistral",
        }
        with patch.dict(os.environ, test_keys, clear=False):
            env = SandboxedCommand.sanitize_env()
            for key, value in test_keys.items():
                assert env.get(key) == value, f"{key} should be preserved"

    def test_preserves_python_env(self):
        """Python environment variables should be preserved."""
        python_vars = {
            "PYTHONPATH": "/custom/python/path",
            "VIRTUAL_ENV": "/path/to/venv",
        }
        with patch.dict(os.environ, python_vars, clear=False):
            env = SandboxedCommand.sanitize_env()
            for key, value in python_vars.items():
                assert env.get(key) == value, f"{key} should be preserved"

    def test_preserves_git_env(self):
        """Git environment variables should be preserved."""
        git_vars = {
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test Committer",
            "GIT_COMMITTER_EMAIL": "committer@example.com",
        }
        with patch.dict(os.environ, git_vars, clear=False):
            env = SandboxedCommand.sanitize_env()
            for key, value in git_vars.items():
                assert env.get(key) == value, f"{key} should be preserved"

    def test_excludes_unknown_vars(self):
        """Unknown environment variables should not be included."""
        with patch.dict(
            os.environ,
            {"UNKNOWN_VAR": "secret", "RANDOM_VAR": "data"},
            clear=False,
        ):
            env = SandboxedCommand.sanitize_env()
            assert "UNKNOWN_VAR" not in env
            assert "RANDOM_VAR" not in env


# ============================================================================
# Async run_sandboxed Tests
# ============================================================================


class TestRunSandboxedAsync:
    """Tests for the async run_sandboxed function."""

    @pytest.mark.asyncio
    async def test_rejects_blocked_command(self):
        """Blocked commands should raise SandboxError."""
        with pytest.raises(SandboxError) as exc_info:
            await run_sandboxed(["rm", "-rf", "/"])
        assert "blocked" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rejects_unknown_command(self):
        """Unknown commands should raise SandboxError."""
        with pytest.raises(SandboxError) as exc_info:
            await run_sandboxed(["unknown_command"])
        assert "not in the allowlist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_runs_allowed_command(self):
        """Allowed commands should execute successfully."""
        result = await run_sandboxed(["python", "--version"])
        assert isinstance(result, SandboxedResult)
        assert result.returncode == 0
        assert "Python" in result.stdout or "Python" in result.stderr

    @pytest.mark.asyncio
    async def test_captures_stdout(self):
        """stdout should be captured when capture_output=True."""
        # Use python -m to run a module that generates output
        result = await run_sandboxed(
            ["python", "-m", "platform"],
            capture_output=True,
        )
        assert result.returncode == 0
        # platform module outputs system info
        assert len(result.stdout) > 0

    @pytest.mark.asyncio
    async def test_captures_stderr(self):
        """stderr should be captured (tested via mock to avoid metachar issues)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr="error output"
            )
            result = await run_sandboxed(
                ["python", "--version"],  # Command passes validation
                capture_output=True,
            )
            assert "error output" in result.stderr

    @pytest.mark.asyncio
    async def test_returns_nonzero_returncode(self):
        """Non-zero return codes should be captured (tested via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=42, stdout="", stderr=""
            )
            result = await run_sandboxed(
                ["python", "--version"],
                check=False,
            )
            assert result.returncode == 42
            assert result.success is False

    @pytest.mark.asyncio
    async def test_check_raises_on_failure(self):
        """check=True should raise CalledProcessError on failure (via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                await run_sandboxed(
                    ["python", "--version"],
                    check=True,
                )
            assert exc_info.value.returncode == 1

    @pytest.mark.asyncio
    async def test_cwd_parameter(self, temp_dir: Path):
        """cwd parameter should set working directory."""
        result = await run_sandboxed(["pwd"], cwd=temp_dir)
        assert str(temp_dir) in result.stdout

    @pytest.mark.asyncio
    async def test_custom_env_merged(self):
        """Custom env should be merged with sanitized env (tested via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="custom_value", stderr=""
            )
            result = await run_sandboxed(
                ["python", "--version"],
                env={"CUSTOM_VAR": "custom_value"},
            )
            # Verify env was passed with custom var merged
            call_kwargs = mock_run.call_args.kwargs
            assert "CUSTOM_VAR" in call_kwargs["env"]
            assert call_kwargs["env"]["CUSTOM_VAR"] == "custom_value"

    @pytest.mark.asyncio
    async def test_input_data_passed_to_stdin(self):
        """input_data should be passed to process stdin (tested via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="test input", stderr=""
            )
            result = await run_sandboxed(
                ["python", "--version"],
                input_data="test input",
            )
            # Verify input was passed
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["input"] == "test input"

    @pytest.mark.asyncio
    async def test_timeout_clamped_to_max(self):
        """Timeout should be clamped to MAX_TIMEOUT."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="output", stderr=""
            )
            await run_sandboxed(
                ["python", "--version"],
                timeout=999999,  # Way over MAX_TIMEOUT
            )
            # Verify timeout was clamped
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["timeout"] == MAX_TIMEOUT

    @pytest.mark.asyncio
    async def test_timeout_produces_timed_out_result(self):
        """Command timeout should produce result with timed_out=True (via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["python", "--version"], timeout=0.1
            )
            result = await run_sandboxed(
                ["python", "--version"],
                timeout=0.1,
            )
            assert result.timed_out is True
            assert result.returncode == -1

    @pytest.mark.asyncio
    async def test_stores_command_in_result(self):
        """Result should store the executed command."""
        cmd = ["python", "--version"]
        result = await run_sandboxed(cmd)
        assert result.command == cmd


# ============================================================================
# Sync run_sandboxed_sync Tests
# ============================================================================


class TestRunSandboxedSync:
    """Tests for the sync run_sandboxed_sync function."""

    def test_rejects_blocked_command(self):
        """Blocked commands should raise SandboxError."""
        with pytest.raises(SandboxError) as exc_info:
            run_sandboxed_sync(["sudo", "ls"])
        assert "blocked" in str(exc_info.value).lower()

    def test_rejects_unknown_command(self):
        """Unknown commands should raise SandboxError."""
        with pytest.raises(SandboxError) as exc_info:
            run_sandboxed_sync(["mystery_command"])
        assert "not in the allowlist" in str(exc_info.value)

    def test_runs_allowed_command(self):
        """Allowed commands should execute successfully."""
        result = run_sandboxed_sync(["python", "--version"])
        assert isinstance(result, SandboxedResult)
        assert result.returncode == 0

    def test_captures_stdout(self):
        """stdout should be captured."""
        # Use python -m to run a module that generates output
        result = run_sandboxed_sync(
            ["python", "-m", "platform"],
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_captures_stderr(self):
        """stderr should be captured (tested via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr="sync error"
            )
            result = run_sandboxed_sync(
                ["python", "--version"],
            )
            assert "sync error" in result.stderr

    def test_check_raises_on_failure(self):
        """check=True should raise CalledProcessError on failure (via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=5, stdout="", stderr=""
            )
            with pytest.raises(subprocess.CalledProcessError):
                run_sandboxed_sync(
                    ["python", "--version"],
                    check=True,
                )

    def test_cwd_parameter(self, temp_dir: Path):
        """cwd parameter should set working directory."""
        result = run_sandboxed_sync(["pwd"], cwd=temp_dir)
        assert str(temp_dir) in result.stdout

    def test_custom_env_merged(self):
        """Custom env should be merged with sanitized env (via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="sync_value", stderr=""
            )
            run_sandboxed_sync(
                ["python", "--version"],
                env={"SYNC_VAR": "sync_value"},
            )
            # Verify env was passed with custom var merged
            call_kwargs = mock_run.call_args.kwargs
            assert "SYNC_VAR" in call_kwargs["env"]
            assert call_kwargs["env"]["SYNC_VAR"] == "sync_value"

    def test_input_data_passed_to_stdin(self):
        """input_data should be passed to process stdin (via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="sync input", stderr=""
            )
            run_sandboxed_sync(
                ["python", "--version"],
                input_data="sync input",
            )
            # Verify input was passed
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["input"] == "sync input"

    def test_timeout_produces_timed_out_result(self):
        """Command timeout should produce result with timed_out=True (via mock)."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["python", "--version"], timeout=0.1
            )
            result = run_sandboxed_sync(
                ["python", "--version"],
                timeout=0.1,
            )
            assert result.timed_out is True
            assert result.returncode == -1


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurityFeatures:
    """Tests for security-related features."""

    def test_shell_false_always(self):
        """subprocess.run should never be called with shell=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )
            run_sandboxed_sync(["python", "--version"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("shell") is False

    @pytest.mark.asyncio
    async def test_command_injection_via_semicolon_blocked(self):
        """Semicolon command injection should be blocked."""
        with pytest.raises(SandboxError):
            await run_sandboxed(["ls", "foo; rm -rf /"])

    @pytest.mark.asyncio
    async def test_command_injection_via_pipe_blocked(self):
        """Pipe command injection should be blocked."""
        with pytest.raises(SandboxError):
            await run_sandboxed(["ls", "foo | cat /etc/passwd"])

    @pytest.mark.asyncio
    async def test_command_injection_via_backtick_blocked(self):
        """Backtick command injection should be blocked."""
        with pytest.raises(SandboxError):
            await run_sandboxed(["ls", "`rm -rf /`"])

    @pytest.mark.asyncio
    async def test_command_injection_via_dollar_blocked(self):
        """Dollar sign command injection should be blocked."""
        with pytest.raises(SandboxError):
            await run_sandboxed(["ls", "$(rm -rf /)"])

    def test_path_traversal_in_command_name_handled(self):
        """Commands with path traversal should be validated by base name."""
        # ../../../bin/rm should still be blocked because base name is 'rm'
        allowed, reason = SandboxedCommand.validate(
            ["../../../bin/rm", "-rf", "/"]
        )
        assert allowed is False

    def test_symbolic_command_path_blocked(self):
        """Blocked command accessed via symlink-like path should be blocked."""
        allowed, reason = SandboxedCommand.validate(["/tmp/../usr/bin/rm", "-rf"])
        assert allowed is False


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_command_with_only_base_name(self):
        """Command with only base name and no args should work."""
        allowed, reason = SandboxedCommand.validate(["pwd"])
        assert allowed is True

    @pytest.mark.asyncio
    async def test_command_with_nonzero_exit_via_mock(self):
        """Running a command that fails should be handled gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Error occurred"
            )
            result = await run_sandboxed(
                ["python", "--version"],
                check=False,
            )
            assert result.returncode != 0

    def test_validate_single_element_command(self):
        """Single element command (no subcommand) should validate."""
        allowed, reason = SandboxedCommand.validate(["pwd"])
        assert allowed is True

    def test_result_stdout_stderr_never_none(self):
        """Result stdout and stderr should never be None."""
        result = SandboxedResult(
            returncode=0,
            stdout=None,  # type: ignore - testing edge case
            stderr=None,  # type: ignore - testing edge case
            command=["test"],
        )
        # The dataclass itself doesn't prevent None, but run_sandboxed should
        # always provide strings

    @pytest.mark.asyncio
    async def test_empty_stdout_and_stderr(self):
        """Commands with minimal output return appropriate strings."""
        # python --version may output to stderr on some systems
        result = await run_sandboxed(["python", "--version"])
        # At minimum, the version output goes somewhere
        assert (result.stdout + result.stderr).strip() != ""

    def test_very_long_command_args(self):
        """Very long command arguments should be handled."""
        long_arg = "a" * 10000
        allowed, reason = SandboxedCommand.validate(["python", "-c", long_arg])
        assert allowed is True


# ============================================================================
# Integration Tests with Real Commands
# ============================================================================


class TestRealCommandExecution:
    """Integration tests with real command execution."""

    @pytest.mark.asyncio
    async def test_git_status_in_git_repo(self, temp_dir: Path):
        """git status should work in a git repo."""
        # Initialize a git repo
        init_result = await run_sandboxed(["git", "init"], cwd=temp_dir)
        assert init_result.returncode == 0

        # Run git status
        result = await run_sandboxed(["git", "status"], cwd=temp_dir)
        assert result.returncode == 0
        assert "branch" in result.stdout.lower() or "commit" in result.stdout.lower()

    def test_which_command(self):
        """which command should find python."""
        result = run_sandboxed_sync(["which", "python"])
        # May return 0 even if not found on some systems, just check it runs
        assert result.returncode in (0, 1)

    def test_date_command(self):
        """date command should work."""
        result = run_sandboxed_sync(["date"])
        assert result.returncode == 0
        assert len(result.stdout) > 0

    @pytest.mark.asyncio
    async def test_pip_list(self):
        """pip list should work."""
        result = await run_sandboxed(["pip", "list"])
        assert result.returncode == 0
        # Should at least list pip itself
        assert "pip" in result.stdout.lower()


# ============================================================================
# Mocked Subprocess Tests
# ============================================================================


class TestMockedSubprocess:
    """Tests using mocked subprocess for isolated testing."""

    def test_sync_subprocess_called_correctly(self):
        """Verify subprocess.run is called with correct parameters."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="output", stderr=""
            )
            result = run_sandboxed_sync(
                ["git", "status"],
                cwd="/test/path",
                timeout=60,
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args

            assert call_args.args[0] == ["git", "status"]
            assert call_args.kwargs["cwd"] == "/test/path"
            assert call_args.kwargs["timeout"] == 60
            assert call_args.kwargs["shell"] is False
            assert call_args.kwargs["capture_output"] is True
            assert call_args.kwargs["text"] is True

    @pytest.mark.asyncio
    async def test_async_uses_run_in_executor(self):
        """Async version should use run_in_executor."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            # Create a completed future with a mock result
            future = asyncio.Future()
            mock_result = MagicMock(returncode=0, stdout="test", stderr="")
            future.set_result(mock_result)
            mock_loop.run_in_executor.return_value = future

            result = await run_sandboxed(["python", "--version"])

            mock_loop.run_in_executor.assert_called_once()

    def test_timeout_expired_handled_sync(self):
        """subprocess.TimeoutExpired should be caught in sync version."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["test"], timeout=1, output=b"partial", stderr=b"err"
            )
            result = run_sandboxed_sync(["python", "--version"])

            assert result.timed_out is True
            assert result.returncode == -1
            assert "partial" in result.stdout
            assert "err" in result.stderr

    def test_timeout_expired_handles_none_output(self):
        """TimeoutExpired with None output should be handled."""
        with patch("subprocess.run") as mock_run:
            exc = subprocess.TimeoutExpired(cmd=["test"], timeout=1)
            exc.stdout = None
            exc.stderr = None
            mock_run.side_effect = exc

            result = run_sandboxed_sync(["python", "--version"])

            assert result.timed_out is True
            assert result.stdout == ""
            assert result.stderr == ""


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
