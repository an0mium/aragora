"""End-to-end tests proving HarnessHealthRegistry wiring at real harness call sites.

Round 30g (#6922 continuation). These tests exercise
:class:`aragora.harnesses.claude_code.ClaudeCodeHarness` and
:class:`aragora.harnesses.codex.CodexHarness` with their underlying
subprocess / OpenAI calls mocked. The goal is to demonstrate that:

  1. ``record_attempt`` fires when ``analyze_repository`` /
     ``analyze_files`` runs.
  2. On success, the registry records a success outcome.
  3. On auth-style failures (401-equivalent), the harness pins
     permanently with ``last_failure_category == "auth"``.
  4. On quota-style failures (429-equivalent), the harness pins
     permanently with ``last_failure_category == "quota"``.
  5. On transient failures (5xx-equivalent), a single failure does
     not pin, but enough transient failures within the rolling window
     do trip the pin.

These tests do **not** call the real Claude Code or OpenAI API. The
boundary that's mocked is the lowest-level subprocess/HTTP wrapper
on each harness; everything between that wrapper and the registry is
real production code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aragora.harnesses.base import AnalysisType
from aragora.harnesses.claude_code import ClaudeCodeConfig, ClaudeCodeHarness
from aragora.harnesses.codex import CodexConfig, CodexHarness
from aragora.swarm.harness_health import (
    TRANSIENT_FAILURE_THRESHOLD,
    get_harness_health_registry,
    reset_harness_health_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_harness_health_registry()
    yield
    reset_harness_health_registry()


@pytest.fixture
def temp_repo() -> Path:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "a.py").write_text("print('a')\n")
        (repo / "b.py").write_text("print('b')\n")
        yield repo


# =============================================================================
# ClaudeCodeHarness (subprocess-mocked)
# =============================================================================


class TestClaudeCodeHarnessWiring:
    @pytest.mark.asyncio
    async def test_success_records_success(self, temp_repo: Path) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig())
        with patch.object(
            harness,
            "_run_claude_code",
            new=AsyncMock(return_value=("[]", "")),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is True
        snap = get_harness_health_registry().snapshot("claude-code")
        assert snap.last_outcome == "success"
        assert snap.available is True
        assert snap.permanent_pin_reason is None

    @pytest.mark.asyncio
    async def test_auth_failure_pins_with_auth_category(self, temp_repo: Path) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig())
        # Simulate an auth-flavoured failure surfacing through the
        # generic exception path. The registry classifies on the
        # textual reason; HTTP status code is not always reachable.
        with patch.object(
            harness,
            "_run_claude_code",
            new=AsyncMock(side_effect=RuntimeError("Unauthorized: invalid API key")),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is False
        snap = get_harness_health_registry().snapshot("claude-code")
        assert snap.available is False
        assert snap.last_outcome == "failure"
        # AUTH/QUOTA failures pin permanently; the canonical signal is
        # permanent_pin_reason. The registry stores the category as a
        # prefix in that string and does not also append to the
        # transient _failures window.
        assert snap.permanent_pin_reason is not None
        assert snap.permanent_pin_reason.startswith("auth:")

    @pytest.mark.asyncio
    async def test_quota_failure_pins_with_quota_category(self, temp_repo: Path) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig())
        with patch.object(
            harness,
            "_run_claude_code",
            new=AsyncMock(side_effect=RuntimeError("rate limit exceeded (429)")),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is False
        snap = get_harness_health_registry().snapshot("claude-code")
        assert snap.available is False
        assert snap.permanent_pin_reason is not None
        assert snap.permanent_pin_reason.startswith("quota:")

    @pytest.mark.asyncio
    async def test_single_transient_does_not_pin(self, temp_repo: Path) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig())
        with patch.object(
            harness,
            "_run_claude_code",
            new=AsyncMock(side_effect=RuntimeError("connection reset by peer")),
        ):
            await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        snap = get_harness_health_registry().snapshot("claude-code")
        # A single transient failure must not pin permanently.
        assert snap.available is True
        assert snap.last_outcome == "failure"
        assert snap.last_failure_category == "transient"
        assert snap.transient_failure_count_in_window == 1

    @pytest.mark.asyncio
    async def test_repeated_transient_eventually_pins(self, temp_repo: Path) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig())
        with patch.object(
            harness,
            "_run_claude_code",
            new=AsyncMock(side_effect=RuntimeError("connection reset by peer")),
        ):
            for _ in range(TRANSIENT_FAILURE_THRESHOLD):
                await harness.analyze_repository(
                    repo_path=temp_repo,
                    analysis_type=AnalysisType.GENERAL,
                )
        snap = get_harness_health_registry().snapshot("claude-code")
        # After enough transients in the rolling window, harness pins.
        assert snap.available is False
        assert snap.permanent_pin_reason is not None

    @pytest.mark.asyncio
    async def test_empty_files_no_op_records_failure_without_pin(self) -> None:
        harness = ClaudeCodeHarness(ClaudeCodeConfig())
        result = await harness.analyze_files(
            files=[],
            analysis_type=AnalysisType.GENERAL,
        )
        # Caller-side bug: no files. The harness records the no-op
        # but treats it as a single failure (not enough to pin).
        assert result.success is False
        snap = get_harness_health_registry().snapshot("claude-code")
        assert snap.last_outcome == "failure"
        assert snap.available is True


# =============================================================================
# CodexHarness (OpenAI-call mocked)
# =============================================================================


class TestCodexHarnessWiring:
    @pytest.mark.asyncio
    async def test_success_records_success(self, temp_repo: Path) -> None:
        harness = CodexHarness(CodexConfig(api_key="test"))
        with patch.object(
            harness,
            "_call_openai",
            new=AsyncMock(return_value="[]"),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is True
        snap = get_harness_health_registry().snapshot("codex")
        assert snap.last_outcome == "success"
        assert snap.available is True

    @pytest.mark.asyncio
    async def test_auth_failure_pins(self, temp_repo: Path) -> None:
        harness = CodexHarness(CodexConfig(api_key="test"))
        with patch.object(
            harness,
            "_call_openai",
            new=AsyncMock(side_effect=ValueError("Unauthorized: invalid API key")),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is False
        snap = get_harness_health_registry().snapshot("codex")
        assert snap.available is False
        assert snap.permanent_pin_reason is not None
        assert snap.permanent_pin_reason.startswith("auth:")

    @pytest.mark.asyncio
    async def test_quota_failure_pins(self, temp_repo: Path) -> None:
        harness = CodexHarness(CodexConfig(api_key="test"))
        with patch.object(
            harness,
            "_call_openai",
            new=AsyncMock(side_effect=RuntimeError("rate limit exceeded (429)")),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is False
        snap = get_harness_health_registry().snapshot("codex")
        assert snap.available is False
        assert snap.permanent_pin_reason is not None
        assert snap.permanent_pin_reason.startswith("quota:")

    @pytest.mark.asyncio
    async def test_single_transient_does_not_pin(self, temp_repo: Path) -> None:
        harness = CodexHarness(CodexConfig(api_key="test"))
        with patch.object(
            harness,
            "_call_openai",
            new=AsyncMock(side_effect=RuntimeError("upstream returned 503")),
        ):
            result = await harness.analyze_repository(
                repo_path=temp_repo,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is False
        snap = get_harness_health_registry().snapshot("codex")
        assert snap.available is True
        assert snap.last_failure_category == "transient"

    @pytest.mark.asyncio
    async def test_no_files_records_success_without_pin(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d)
            harness = CodexHarness(CodexConfig(api_key="test"))
            result = await harness.analyze_repository(
                repo_path=empty,
                analysis_type=AnalysisType.GENERAL,
            )
        assert result.success is True
        snap = get_harness_health_registry().snapshot("codex")
        # Empty work surface is success, not failure.
        assert snap.last_outcome == "success"
        assert snap.available is True


# =============================================================================
# record_harness_result helper edge cases
# =============================================================================


class TestRecordHarnessResultHelper:
    def test_success_path(self) -> None:
        from aragora.swarm.harness_health import record_harness_result

        record_harness_result(harness="x", success=True)
        snap = get_harness_health_registry().snapshot("x")
        assert snap.last_outcome == "success"

    def test_failure_uses_error_message_when_present(self) -> None:
        from aragora.swarm.harness_health import record_harness_result

        record_harness_result(
            harness="x",
            success=False,
            error_message="Unauthorized: invalid API key",
            status_code=401,
        )
        snap = get_harness_health_registry().snapshot("x")
        assert snap.available is False
        # AUTH classification surfaces as a 'auth:' permanent pin.
        assert snap.permanent_pin_reason is not None
        assert snap.permanent_pin_reason.startswith("auth:")

    def test_failure_falls_back_to_error_output_tail(self) -> None:
        from aragora.swarm.harness_health import record_harness_result

        record_harness_result(
            harness="x",
            success=False,
            error_output="some\nlong\nstderr\nUnauthorized at line 42",
        )
        snap = get_harness_health_registry().snapshot("x")
        assert snap.permanent_pin_reason is not None
        assert snap.permanent_pin_reason.startswith("auth:")

    def test_failure_with_no_signal_records_transient(self) -> None:
        from aragora.swarm.harness_health import record_harness_result

        record_harness_result(harness="x", success=False)
        snap = get_harness_health_registry().snapshot("x")
        assert snap.last_outcome == "failure"
        # Unknown reason classifies as transient (not pinning).
        assert snap.available is True
