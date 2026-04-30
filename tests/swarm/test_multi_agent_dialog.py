"""Tests for aragora.swarm.multi_agent_dialog."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from aragora.swarm.multi_agent_dialog import (
    CLAUDE_FLAGS,
    CODEX_FLAGS,
    DROID_FLAGS,
    DROID_REVIEW_SYSTEM_PROMPT,
    DEFAULT_CLAUDE_TIMEOUT,
    DEFAULT_CODEX_TIMEOUT,
    DEFAULT_DROID_TIMEOUT,
    DEFAULT_MODEL_TIMEOUT,
    MAX_OUTPUT_BYTES,
    RC_BINARY_NOT_FOUND,
    RC_DISPATCH_ERROR,
    AgentSpec,
    DialogRound,
    DialogTurn,
    dispatch_round,
    render_transcript_markdown,
    run_round_and_persist,
    write_round_jsonl,
    write_transcript_markdown,
    _atomic_write_text,
    _dispatch_one,
    _escape_md_fence,
    _strip_ansi,
    _truncate_output,
    _validate_round_id,
)


def test_agent_spec_claude_flags() -> None:
    spec = AgentSpec.claude()
    assert spec.name == "claude"
    assert spec.binary == "claude"
    assert spec.base_flags == CLAUDE_FLAGS
    assert spec.stdin_mode == "stdin"
    assert spec.timeout_seconds == DEFAULT_CLAUDE_TIMEOUT


def test_agent_spec_codex_flags_include_minimal_reasoning() -> None:
    spec = AgentSpec.codex()
    assert "reasoning_effort=minimal" in spec.base_flags
    assert "sandbox_mode=read-only" in spec.base_flags
    assert spec.timeout_seconds == DEFAULT_CODEX_TIMEOUT


def test_agent_spec_droid_flags_are_read_search_only() -> None:
    spec = AgentSpec.droid()
    assert "exec" in spec.base_flags
    assert "--auto" not in spec.base_flags
    assert "--disabled-tools" in spec.base_flags
    assert "Execute" in spec.base_flags
    assert DROID_REVIEW_SYSTEM_PROMPT in spec.base_flags
    assert spec.timeout_seconds == DEFAULT_DROID_TIMEOUT


def test_agent_spec_custom_timeout() -> None:
    spec = AgentSpec.claude(timeout_seconds=30)
    assert spec.timeout_seconds == 30


def test_dialog_round_render_prompt_no_context() -> None:
    r = DialogRound(round_id="t", prompt="hello")
    assert r.render_prompt() == "hello"


def test_dialog_round_render_prompt_with_context() -> None:
    r = DialogRound(round_id="t", prompt="hello", extra_context="more info")
    assert "hello" in r.render_prompt()
    assert "more info" in r.render_prompt()


def test_dialog_turn_succeeded_clean() -> None:
    t = DialogTurn(
        agent="x",
        started_at="2026-04-30T00:00:00+00:00",
        finished_at="2026-04-30T00:00:01+00:00",
        elapsed_seconds=1.0,
        returncode=0,
        stdout="OK",
        stderr="",
        timed_out=False,
        error=None,
    )
    assert t.succeeded()


def test_dialog_turn_succeeded_false_on_timeout() -> None:
    t = DialogTurn(
        agent="x",
        started_at="t1",
        finished_at="t2",
        elapsed_seconds=60.0,
        returncode=0,
        stdout="",
        stderr="",
        timed_out=True,
    )
    assert not t.succeeded()


def test_dialog_turn_succeeded_false_on_error() -> None:
    t = DialogTurn(
        agent="x",
        started_at="t1",
        finished_at="t2",
        elapsed_seconds=0.0,
        returncode=-1,
        stdout="",
        stderr="",
        timed_out=False,
        error="binary not found",
    )
    assert not t.succeeded()


def test_dialog_turn_succeeded_false_on_nonzero_rc() -> None:
    t = DialogTurn(
        agent="x",
        started_at="t1",
        finished_at="t2",
        elapsed_seconds=1.0,
        returncode=1,
        stdout="",
        stderr="boom",
        timed_out=False,
    )
    assert not t.succeeded()


@pytest.mark.asyncio
async def test_dispatch_one_succeeds_with_echo() -> None:
    spec = AgentSpec(
        name="echo",
        binary="/bin/sh",
        base_flags=("-c", "cat"),
        timeout_seconds=10,
        stdin_mode="stdin",
    )
    turn = await _dispatch_one(spec, rendered_prompt="hello")
    assert turn.succeeded()
    assert "hello" in turn.stdout
    assert turn.returncode == 0


@pytest.mark.asyncio
async def test_dispatch_one_handles_missing_binary() -> None:
    spec = AgentSpec(
        name="missing",
        binary="/no/such/binary-9b13c2e4",
        base_flags=(),
        timeout_seconds=5,
        stdin_mode="stdin",
    )
    turn = await _dispatch_one(spec, rendered_prompt="hi")
    assert not turn.succeeded()
    assert turn.error is not None
    assert "binary not found" in turn.error


@pytest.mark.asyncio
async def test_dispatch_one_handles_timeout() -> None:
    spec = AgentSpec(
        name="sleeper",
        binary="/bin/sh",
        base_flags=("-c", "sleep 5"),
        timeout_seconds=1,
        stdin_mode="argv",
    )
    turn = await _dispatch_one(spec, rendered_prompt="x")
    assert turn.timed_out
    assert not turn.succeeded()


@pytest.mark.asyncio
async def test_dispatch_round_isolates_failures() -> None:
    """A failing agent must not break the others."""
    good = AgentSpec(
        name="good",
        binary="/bin/sh",
        base_flags=("-c", "echo OK"),
        timeout_seconds=5,
        stdin_mode="argv",
    )
    bad = AgentSpec(
        name="bad",
        binary="/no/such/binary-9b13c2e4",
        base_flags=(),
        timeout_seconds=5,
        stdin_mode="argv",
    )
    round_ = DialogRound(round_id="t", prompt="hi")
    turns = await dispatch_round(round_, [good, bad])
    assert len(turns) == 2
    by_agent = {t.agent: t for t in turns}
    assert by_agent["good"].succeeded()
    assert not by_agent["bad"].succeeded()


@pytest.mark.asyncio
async def test_dispatch_round_empty_agents() -> None:
    round_ = DialogRound(round_id="t", prompt="hi")
    turns = await dispatch_round(round_, [])
    assert turns == []


def test_write_round_jsonl_creates_file(tmp_path: Path) -> None:
    round_ = DialogRound(round_id="r1", prompt="p")
    turns = [
        DialogTurn(
            agent="a",
            started_at="t1",
            finished_at="t2",
            elapsed_seconds=1.0,
            returncode=0,
            stdout="x",
            stderr="",
            timed_out=False,
        )
    ]
    out = write_round_jsonl(round_, turns, tmp_path)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    head = json.loads(lines[0])
    assert head["type"] == "round"
    assert head["round_id"] == "r1"
    body = json.loads(lines[1])
    assert body["type"] == "turn"
    assert body["agent"] == "a"


def test_render_transcript_markdown_includes_summary() -> None:
    round_ = DialogRound(round_id="r1", prompt="Review this")
    turns = [
        DialogTurn(
            agent="a1",
            started_at="t1",
            finished_at="t2",
            elapsed_seconds=1.0,
            returncode=0,
            stdout="great work",
            stderr="",
            timed_out=False,
        ),
        DialogTurn(
            agent="a2",
            started_at="t1",
            finished_at="t2",
            elapsed_seconds=2.0,
            returncode=1,
            stdout="",
            stderr="oops",
            timed_out=False,
        ),
    ]
    md = render_transcript_markdown(round_, turns)
    assert "Cross-agent dialog" in md
    assert "Successful: **1**" in md
    assert "Failed: **1**" in md
    assert "great work" in md
    assert "FAILED" in md


def test_render_transcript_markdown_with_extra_context() -> None:
    round_ = DialogRound(round_id="r1", prompt="Review", extra_context="some source")
    md = render_transcript_markdown(round_, [])
    assert "Extra context" in md
    assert "some source" in md


def test_write_transcript_markdown_creates_file(tmp_path: Path) -> None:
    round_ = DialogRound(round_id="r1", prompt="p")
    out = write_transcript_markdown(round_, [], tmp_path)
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "Cross-agent dialog" in body


@pytest.mark.asyncio
async def test_run_round_and_persist_end_to_end(tmp_path: Path) -> None:
    spec = AgentSpec(
        name="echo",
        binary="/bin/sh",
        base_flags=("-c", "echo HELLO"),
        timeout_seconds=5,
        stdin_mode="argv",
    )
    round_ = DialogRound(round_id="e2e", prompt="ping")
    jsonl_path, md_path, turns = await run_round_and_persist(round_, [spec], tmp_path)
    assert jsonl_path.exists()
    assert md_path.exists()
    assert len(turns) == 1
    assert turns[0].succeeded()
    md = md_path.read_text(encoding="utf-8")
    assert "HELLO" in md


def test_constants_match_round_30d_verification() -> None:
    """These flag tuples are the exact verified-working flag sets.

    If any of these change, the round's agent_dispatch_verification step
    must be re-run before merging."""
    assert CLAUDE_FLAGS == ("--print",)
    assert "reasoning_effort=minimal" in CODEX_FLAGS
    assert "sandbox_mode=read-only" in CODEX_FLAGS
    assert "--skip-git-repo-check" in CODEX_FLAGS
    assert DROID_FLAGS == (
        "exec",
        "--disabled-tools",
        "Execute",
        "--append-system-prompt",
        DROID_REVIEW_SYSTEM_PROMPT,
    )


# --------------------------------------------------------------------- #
# Heterogeneous model factories (round 30e Phase B)                     #
# --------------------------------------------------------------------- #


class TestWithModel:
    def test_with_model_claude_appends_model_flag(self) -> None:
        spec = AgentSpec.with_model("claude", "opus")
        assert spec.binary == "claude"
        # Model flag must be appended *after* the base CLAUDE_FLAGS so
        # the CLI's positional parser still sees ``--print`` first.
        assert spec.base_flags == (*CLAUDE_FLAGS, "--model", "opus")
        assert spec.name == "claude:opus"
        assert spec.stdin_mode == "stdin"

    def test_with_model_droid_appends_short_m_flag(self) -> None:
        spec = AgentSpec.with_model("droid", "gpt-5.4")
        assert spec.binary == "droid"
        assert spec.base_flags == (*DROID_FLAGS, "-m", "gpt-5.4")
        assert spec.name == "droid:gpt-5.4"

    def test_with_model_custom_name_overrides_default(self) -> None:
        spec = AgentSpec.with_model("droid", "kimi-k2.5", name="bear-kimi")
        assert spec.name == "bear-kimi"

    def test_with_model_codex_raises_value_error(self) -> None:
        # codex CLI doesn't expose a per-invocation model flag yet;
        # the harness must explicitly fail rather than silently use
        # the default model.
        with pytest.raises(ValueError, match="codex"):
            AgentSpec.with_model("codex", "gpt-5.4")

    def test_with_model_unknown_cli_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown cli"):
            AgentSpec.with_model("gpt", "gpt-5.4")

    def test_with_model_case_insensitive_cli(self) -> None:
        spec = AgentSpec.with_model("DROID", "kimi-k2.5")
        assert spec.binary == "droid"

    def test_with_model_custom_timeout_propagates(self) -> None:
        spec = AgentSpec.with_model("droid", "kimi-k2.5", timeout_seconds=200)
        assert spec.timeout_seconds == 200


class TestNamedHeterogeneousFactories:
    def test_claude_opus(self) -> None:
        spec = AgentSpec.claude_opus()
        assert spec.binary == "claude"
        assert "opus" in spec.base_flags
        assert spec.name == "claude-opus"

    def test_claude_sonnet(self) -> None:
        spec = AgentSpec.claude_sonnet()
        assert spec.binary == "claude"
        assert "sonnet" in spec.base_flags
        assert spec.name == "claude-sonnet"

    def test_droid_gpt5(self) -> None:
        spec = AgentSpec.droid_gpt5()
        assert spec.binary == "droid"
        assert "gpt-5.4" in spec.base_flags
        assert spec.name == "droid-gpt5"

    def test_droid_gemini(self) -> None:
        spec = AgentSpec.droid_gemini()
        assert spec.binary == "droid"
        assert "gemini-3.1-pro-preview" in spec.base_flags
        assert spec.name == "droid-gemini"

    def test_droid_kimi(self) -> None:
        spec = AgentSpec.droid_kimi()
        assert spec.binary == "droid"
        assert "kimi-k2.5" in spec.base_flags

    def test_droid_glm(self) -> None:
        spec = AgentSpec.droid_glm()
        assert spec.binary == "droid"
        assert "glm-5.1" in spec.base_flags


class TestHeterogeneousPanel:
    def test_panel_has_six_distinct_models(self) -> None:
        panel = AgentSpec.heterogeneous_panel()
        assert len(panel) == 6
        # Names must be unique so per-agent transcript files don't collide.
        assert len({s.name for s in panel}) == 6

    def test_panel_spans_at_least_three_families(self) -> None:
        panel = AgentSpec.heterogeneous_panel()
        # Heuristic: family inferred from the model substring in flags.
        families: set[str] = set()
        for spec in panel:
            joined = " ".join(spec.base_flags).lower()
            if "opus" in joined or "sonnet" in joined:
                families.add("anthropic")
            elif "gpt-" in joined:
                families.add("openai-gpt")
            elif "gemini" in joined:
                families.add("google")
            elif "kimi" in joined or "glm" in joined:
                families.add("chinese")
            elif spec.binary == "codex":
                # codex CLI talks to OpenAI; counts as openai-codex.
                families.add("openai-codex")
        assert len(families) >= 3, families

    def test_panel_codex_timeout_is_higher(self) -> None:
        panel = AgentSpec.heterogeneous_panel(codex_timeout=200, model_timeout=90)
        codex_spec = next(s for s in panel if s.binary == "codex")
        assert codex_spec.timeout_seconds == 200
        # All other specs share model_timeout.
        for spec in panel:
            if spec.binary != "codex":
                assert spec.timeout_seconds == 90

    def test_panel_defaults_use_quality_budgets(self) -> None:
        panel = AgentSpec.heterogeneous_panel()
        codex_spec = next(s for s in panel if s.binary == "codex")
        assert codex_spec.timeout_seconds == DEFAULT_CODEX_TIMEOUT
        for spec in panel:
            if spec.binary != "codex":
                assert spec.timeout_seconds == DEFAULT_MODEL_TIMEOUT


def test_cli_parse_args_uses_quality_timeout_defaults() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "multi_agent_dialog.py"
    spec = importlib.util.spec_from_file_location("multi_agent_dialog_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--round-id",
            "r",
            "--prompt",
            "p",
            "--output-dir",
            "/tmp/dialog",
        ]
    )
    assert args.claude_timeout == DEFAULT_CLAUDE_TIMEOUT
    assert args.codex_timeout == DEFAULT_CODEX_TIMEOUT
    assert args.droid_timeout == DEFAULT_DROID_TIMEOUT
    assert args.model_timeout == DEFAULT_MODEL_TIMEOUT


# --------------------------------------------------------------------- #
# Round 30e Phase C — gauntlet hardening regression tests              #
# --------------------------------------------------------------------- #


class TestStripAnsi:
    def test_strip_csi_color(self) -> None:
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strip_osc_hyperlink(self) -> None:
        # OSC-8 hyperlink: ESC ] 8 ; ; URL ESC \ TEXT ESC ] 8 ; ; ESC \
        s = "\x1b]8;;https://x\x1b\\link\x1b]8;;\x1b\\"
        assert _strip_ansi(s) == "link"

    def test_strip_idempotent_on_clean_text(self) -> None:
        assert _strip_ansi("plain text\nline 2") == "plain text\nline 2"


class TestTruncateOutput:
    def test_truncate_under_limit_passthrough(self) -> None:
        assert _truncate_output("short") == "short"

    def test_truncate_over_limit_inserts_sentinel(self) -> None:
        big = "x" * (MAX_OUTPUT_BYTES + 10)
        out = _truncate_output(big)
        assert "[TRUNCATED:" in out
        assert len(out) < len(big) + 100

    def test_truncate_handles_multibyte(self) -> None:
        # Each emoji is 4 bytes UTF-8; should still truncate cleanly.
        s = "🦀" * (MAX_OUTPUT_BYTES // 4 + 5)
        out = _truncate_output(s)
        assert "[TRUNCATED:" in out


class TestEscapeMdFence:
    def test_escape_at_line_start(self) -> None:
        # The fence is the very first 3 chars; escape inserts ZWSP.
        out = _escape_md_fence("```evil")
        assert out.startswith("\u200b```evil")

    def test_escape_with_indent(self) -> None:
        out = _escape_md_fence("    ```evil")
        assert "\u200b```" in out
        # Original leading whitespace preserved.
        assert out.startswith("    \u200b```")

    def test_escape_does_not_touch_inline_backticks(self) -> None:
        # Single backticks aren't a fence; should pass through.
        assert _escape_md_fence("inline `code` here") == "inline `code` here"


class TestValidateRoundId:
    def test_valid_simple(self) -> None:
        _validate_round_id("round-30e")

    def test_valid_alnum_underscore(self) -> None:
        _validate_round_id("e2e_smoke_001")

    def test_reject_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            _validate_round_id("../etc/passwd")

    def test_reject_slash(self) -> None:
        with pytest.raises(ValueError):
            _validate_round_id("a/b")

    def test_reject_empty(self) -> None:
        with pytest.raises(ValueError):
            _validate_round_id("")

    def test_reject_too_long(self) -> None:
        with pytest.raises(ValueError):
            _validate_round_id("x" * 200)


class TestAtomicWriteText:
    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        _atomic_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("old")
        _atomic_write_text(target, "new")
        assert target.read_text() == "new"

    def test_atomic_write_no_tmp_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        _atomic_write_text(target, "x")
        # No .tmp.* files should remain in the directory.
        leftover = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftover == []


class TestDispatchHardening:
    def test_dispatch_one_filenotfound_returns_distinct_rc(self) -> None:
        # Round 30e Phase C fix #3: rc=-127 for missing binary.
        spec = AgentSpec(
            name="ghost",
            binary="/nonexistent/path/to/xyz",
            base_flags=(),
            timeout_seconds=5,
            stdin_mode="stdin",
        )
        turn = asyncio.run(_dispatch_one(spec, "hi"))
        assert turn.returncode == RC_BINARY_NOT_FOUND
        assert turn.error is not None
        assert "binary not found" in turn.error
        assert not turn.succeeded()

    def test_dispatch_round_handles_per_agent_exception(self) -> None:
        # Round 30e Phase C fix #6: one agent's failure does not
        # cascade. We simulate by mixing a real (missing-binary) spec
        # with a dispatchable real binary so we verify *peer* survival.
        ghost = AgentSpec(
            name="ghost-dispatch-test",
            binary="/nonexistent/x",
            base_flags=(),
            timeout_seconds=5,
            stdin_mode="stdin",
        )
        # ``true`` is a real binary on every POSIX machine.
        good = AgentSpec(
            name="good",
            binary="true",
            base_flags=(),
            timeout_seconds=5,
            stdin_mode="stdin",
        )
        round_ = DialogRound(round_id="cascade", prompt="ping")
        turns = asyncio.run(dispatch_round(round_, [ghost, good]))
        assert len(turns) == 2
        # ghost failed cleanly with sentinel rc
        assert turns[0].returncode == RC_BINARY_NOT_FOUND
        # good succeeded — ghost's failure did not cascade
        assert turns[1].succeeded(), turns[1]


class TestPersistenceHardening:
    def test_write_round_jsonl_rejects_path_traversal(self, tmp_path: Path) -> None:
        bad = DialogRound(round_id="../escape", prompt="x")
        with pytest.raises(ValueError):
            write_round_jsonl(bad, [], tmp_path)

    def test_write_transcript_markdown_rejects_path_traversal(self, tmp_path: Path) -> None:
        bad = DialogRound(round_id="../escape", prompt="x")
        with pytest.raises(ValueError):
            write_transcript_markdown(bad, [], tmp_path)

    def test_write_round_jsonl_atomic_no_tmp_left(self, tmp_path: Path) -> None:
        round_ = DialogRound(round_id="atomic-test", prompt="x")
        write_round_jsonl(round_, [], tmp_path)
        leftover = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftover == []


class TestRenderHardening:
    def test_render_escapes_nested_fence_in_stdout(self) -> None:
        round_ = DialogRound(round_id="esc", prompt="p")
        turn = DialogTurn(
            agent="evil",
            started_at="2026-04-30T00:00:00+00:00",
            finished_at="2026-04-30T00:00:01+00:00",
            elapsed_seconds=1.0,
            returncode=0,
            stdout="```\nbreak out\n```\nuninvited content",
            stderr="",
            timed_out=False,
        )
        md = render_transcript_markdown(round_, [turn])
        # The nested ``` should have a zero-width space inserted before
        # it so the surrounding fence isn't broken.
        assert "\u200b```" in md
        # The "uninvited content" line must not appear *outside* a fence:
        # i.e. the document still has matched fences.
        assert md.count("```") % 2 == 0, md

    def test_render_marks_timed_out_turn(self) -> None:
        round_ = DialogRound(round_id="to", prompt="p")
        turn = DialogTurn(
            agent="slow",
            started_at="2026-04-30T00:00:00+00:00",
            finished_at="2026-04-30T00:00:30+00:00",
            elapsed_seconds=30.0,
            returncode=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )
        md = render_transcript_markdown(round_, [turn])
        assert "[TIMED OUT]" in md

    def test_render_distinguishes_failed_from_timed_out(self) -> None:
        round_ = DialogRound(round_id="fail", prompt="p")
        timed_out_turn = DialogTurn(
            agent="slow",
            started_at="2026-04-30T00:00:00+00:00",
            finished_at="2026-04-30T00:00:30+00:00",
            elapsed_seconds=30.0,
            returncode=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )
        plain_failed_turn = DialogTurn(
            agent="bad",
            started_at="2026-04-30T00:00:00+00:00",
            finished_at="2026-04-30T00:00:01+00:00",
            elapsed_seconds=1.0,
            returncode=2,
            stdout="",
            stderr="oops",
            timed_out=False,
        )
        md = render_transcript_markdown(round_, [timed_out_turn, plain_failed_turn])
        assert "FAILED [TIMED OUT]" in md
        # Plain FAILED should NOT have the timed-out badge.
        bad_section = md.split("### `bad`")[1]
        assert "[TIMED OUT]" not in bad_section


class TestSentinelExports:
    def test_sentinels_distinct(self) -> None:
        assert RC_BINARY_NOT_FOUND != RC_DISPATCH_ERROR
        # Both should be negative (out of normal POSIX rc range) so
        # they can't be confused with real exit codes 0..255.
        assert RC_BINARY_NOT_FOUND < 0
        assert RC_DISPATCH_ERROR < 0


# --------------------------------------------------------------------- #
# Phase D follow-up: heterogeneous-review regression tests              #
# --------------------------------------------------------------------- #
#
# These cover the two real findings from the 6-model panel review of
# Phase C: codex (truncation overshoot) and claude-opus (DoS amplifier
# from running ANSI strip *before* truncation).


class TestTruncateOutputPhaseD:
    def test_truncated_output_total_bytes_under_cap(self) -> None:
        big = "x" * (MAX_OUTPUT_BYTES + 1000)
        out = _truncate_output(big)
        # The total persisted bytes (including sentinel) must not
        # exceed the cap — codex's Phase D finding.
        assert len(out.encode("utf-8")) <= MAX_OUTPUT_BYTES

    def test_truncated_output_sentinel_still_present(self) -> None:
        big = "x" * (MAX_OUTPUT_BYTES + 1000)
        out = _truncate_output(big)
        assert "[TRUNCATED:" in out
        assert "original" in out

    def test_truncated_output_with_small_cap(self) -> None:
        # Sanity: a tiny cap doesn't crash on the sentinel-budget edge
        # case (sentinel bytes >= max_bytes).
        out = _truncate_output("hello world", max_bytes=5)
        assert "[TRUNCATED:" in out
