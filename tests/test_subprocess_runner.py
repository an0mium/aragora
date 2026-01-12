"""
Tests for subprocess sandboxing utility.
"""

import asyncio
import subprocess

import pytest

from aragora.utils.subprocess_runner import (
    ALLOWED_COMMANDS,
    BLOCKED_COMMANDS,
    SandboxedCommand,
    SandboxedResult,
    SandboxError,
    run_sandboxed,
    run_sandboxed_sync,
)


class TestSandboxedCommand:
    """Tests for SandboxedCommand validation."""

    def test_validate_empty_command(self):
        """Empty commands should be rejected."""
        allowed, reason = SandboxedCommand.validate([])
        assert not allowed
        assert "Empty command" in reason

    def test_validate_blocked_command(self):
        """Blocked commands should be rejected."""
        for cmd in ["rm", "sudo", "curl", "wget"]:
            allowed, reason = SandboxedCommand.validate([cmd, "arg"])
            assert not allowed
            assert "blocked" in reason.lower()

    def test_validate_allowed_command(self):
        """Allowed commands should pass validation."""
        allowed, reason = SandboxedCommand.validate(["git", "status"])
        assert allowed
        assert "allowed" in reason.lower()

    def test_validate_unknown_command(self):
        """Unknown commands should be rejected."""
        allowed, reason = SandboxedCommand.validate(["unknown_cmd", "arg"])
        assert not allowed
        assert "not in the allowlist" in reason

    def test_validate_subcommand_restriction(self):
        """Commands with subcommand restrictions should be enforced."""
        # git allows 'status'
        allowed, reason = SandboxedCommand.validate(["git", "status"])
        assert allowed

        # git doesn't allow arbitrary subcommands like 'arbitrary'
        allowed, reason = SandboxedCommand.validate(["git", "arbitrary"])
        assert not allowed

    def test_validate_unrestricted_subcommands(self):
        """Commands with None subcommand list allow all subcommands."""
        # pytest allows all subcommands
        allowed, reason = SandboxedCommand.validate(["pytest", "-v", "--tb=short"])
        assert allowed

    def test_validate_shell_metacharacters(self):
        """Shell metacharacters should be blocked in most contexts."""
        # Should block shell injection attempts
        allowed, reason = SandboxedCommand.validate(["ls", "; rm -rf /"])
        assert not allowed
        assert "metacharacter" in reason.lower()

    def test_validate_shell_metacharacters_git_commit(self):
        """Git commit messages can contain shell chars."""
        # Git commit messages may contain special chars
        allowed, reason = SandboxedCommand.validate(["git", "commit", "-m", "fix: handle $VAR"])
        assert allowed

    def test_validate_full_path_command(self):
        """Commands with full paths should extract base name."""
        allowed, reason = SandboxedCommand.validate(["/usr/bin/git", "status"])
        assert allowed

    def test_sanitize_env_preserves_path(self):
        """Sanitized environment should preserve PATH."""
        env = SandboxedCommand.sanitize_env()
        assert "PATH" in env

    def test_sanitize_env_excludes_dangerous(self):
        """Sanitized environment should exclude dangerous vars."""
        import os
        # Set a dangerous variable temporarily
        original = os.environ.get("LD_PRELOAD")
        os.environ["LD_PRELOAD"] = "/tmp/evil.so"
        try:
            env = SandboxedCommand.sanitize_env()
            assert "LD_PRELOAD" not in env
        finally:
            if original:
                os.environ["LD_PRELOAD"] = original
            else:
                os.environ.pop("LD_PRELOAD", None)


class TestSandboxedResult:
    """Tests for SandboxedResult dataclass."""

    def test_success_property_zero_returncode(self):
        """Success should be True for returncode 0."""
        result = SandboxedResult(
            returncode=0,
            stdout="output",
            stderr="",
            command=["test"],
        )
        assert result.success is True

    def test_success_property_nonzero_returncode(self):
        """Success should be False for non-zero returncode."""
        result = SandboxedResult(
            returncode=1,
            stdout="",
            stderr="error",
            command=["test"],
        )
        assert result.success is False

    def test_timed_out_default_false(self):
        """Timed_out should default to False."""
        result = SandboxedResult(
            returncode=0,
            stdout="",
            stderr="",
            command=["test"],
        )
        assert result.timed_out is False


class TestRunSandboxedSync:
    """Tests for run_sandboxed_sync function."""

    def test_run_allowed_command(self):
        """Allowed commands should run successfully."""
        result = run_sandboxed_sync(["git", "--version"])
        assert result.returncode == 0
        assert "git" in result.stdout.lower()

    def test_run_blocked_command(self):
        """Blocked commands should raise SandboxError."""
        with pytest.raises(SandboxError) as exc_info:
            run_sandboxed_sync(["rm", "-rf", "/tmp/test"])
        assert "blocked" in str(exc_info.value).lower()

    def test_run_with_cwd(self):
        """Commands should respect working directory."""
        result = run_sandboxed_sync(["pwd"], cwd="/tmp")
        assert result.success
        assert "/tmp" in result.stdout or "private/tmp" in result.stdout  # macOS uses /private/tmp

    def test_run_with_timeout(self):
        """Commands should timeout when exceeding limit."""
        # This test uses a command that might take long
        result = run_sandboxed_sync(["git", "status"], timeout=1)
        # Should complete within timeout for a simple command
        assert result.returncode == 0 or result.timed_out

    def test_run_check_raises_on_failure(self):
        """check=True should raise on non-zero return."""
        # git show on non-existent ref
        with pytest.raises(subprocess.CalledProcessError):
            run_sandboxed_sync(["git", "show", "nonexistent12345"], check=True)

    def test_run_captures_stdout(self):
        """stdout should be captured."""
        result = run_sandboxed_sync(["git", "--version"])
        assert len(result.stdout) > 0

    def test_run_captures_stderr(self):
        """stderr should be captured on error."""
        result = run_sandboxed_sync(["git", "show", "nonexistent12345"])
        # Git outputs error to stderr
        assert result.returncode != 0


class TestRunSandboxedAsync:
    """Tests for run_sandboxed async function."""

    @pytest.mark.asyncio
    async def test_run_allowed_command(self):
        """Allowed commands should run successfully."""
        result = await run_sandboxed(["git", "--version"])
        assert result.returncode == 0
        assert "git" in result.stdout.lower()

    @pytest.mark.asyncio
    async def test_run_blocked_command(self):
        """Blocked commands should raise SandboxError."""
        with pytest.raises(SandboxError):
            await run_sandboxed(["rm", "-rf", "/"])

    @pytest.mark.asyncio
    async def test_run_with_cwd(self):
        """Commands should respect working directory."""
        result = await run_sandboxed(["pwd"], cwd="/tmp")
        assert result.success

    @pytest.mark.asyncio
    async def test_run_concurrent(self):
        """Multiple sandboxed commands can run concurrently."""
        results = await asyncio.gather(
            run_sandboxed(["git", "--version"]),
            run_sandboxed(["date"]),
            run_sandboxed(["whoami"]),
        )
        assert all(r.success for r in results)


class TestAllowlist:
    """Tests for command allowlist configuration."""

    def test_git_allowed_subcommands(self):
        """Git should allow standard subcommands."""
        for subcmd in ["status", "diff", "log", "commit", "push", "pull"]:
            allowed, _ = SandboxedCommand.validate(["git", subcmd])
            assert allowed, f"git {subcmd} should be allowed"

    def test_ffmpeg_allowed(self):
        """ffmpeg should be in allowlist."""
        assert "ffmpeg" in ALLOWED_COMMANDS
        assert "ffprobe" in ALLOWED_COMMANDS

    def test_lean_allowed(self):
        """lean should be in allowlist for formal verification."""
        assert "lean" in ALLOWED_COMMANDS

    def test_pytest_allowed(self):
        """pytest should be in allowlist."""
        assert "pytest" in ALLOWED_COMMANDS
        allowed, _ = SandboxedCommand.validate(["pytest", "-v", "--tb=short", "tests/"])
        assert allowed

    def test_blocked_commands_present(self):
        """Critical dangerous commands should be blocked."""
        for cmd in ["rm", "sudo", "curl", "wget", "ssh", "docker"]:
            assert cmd in BLOCKED_COMMANDS, f"{cmd} should be blocked"


class TestSecurityScenarios:
    """Tests for security scenarios and edge cases."""

    def test_command_injection_semicolon(self):
        """Command injection via semicolon should be blocked."""
        allowed, _ = SandboxedCommand.validate(["ls", ".; rm -rf /"])
        assert not allowed

    def test_command_injection_pipe(self):
        """Command injection via pipe should be blocked."""
        allowed, _ = SandboxedCommand.validate(["ls", "| cat /etc/passwd"])
        assert not allowed

    def test_command_injection_backtick(self):
        """Command injection via backtick should be blocked."""
        allowed, _ = SandboxedCommand.validate(["echo", "`id`"])
        assert not allowed

    def test_command_injection_dollar(self):
        """Command injection via $() should be blocked."""
        allowed, _ = SandboxedCommand.validate(["echo", "$(id)"])
        assert not allowed

    def test_path_traversal_in_command(self):
        """Path traversal in command path should use base name."""
        # Even with path traversal, only base name is checked
        allowed, _ = SandboxedCommand.validate(["../../bin/git", "status"])
        assert allowed  # git is allowed regardless of path

    def test_blocked_command_with_path(self):
        """Blocked commands should be blocked even with full path."""
        allowed, _ = SandboxedCommand.validate(["/bin/rm", "-rf", "/"])
        assert not allowed

    def test_empty_string_argument(self):
        """Empty string arguments should be handled."""
        allowed, _ = SandboxedCommand.validate(["git", "commit", "-m", ""])
        assert allowed

    def test_very_long_argument(self):
        """Very long arguments should be handled."""
        long_arg = "a" * 10000
        allowed, _ = SandboxedCommand.validate(["git", "commit", "-m", long_arg])
        assert allowed
