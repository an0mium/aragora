"""Tests for the B3 collect-evidence module (aragora.swarm.quorum_evidence).

Covers the two safety invariants directly:

* tier-gating — Tier 3+ (and unknown tier) never post, regardless of --apply;
* never-fabricate — failed/empty reviewers produce no comment.

The compose helper is checked against the *real* evidence parser
(``_lint_evidence_comment``) so the collector stays bound to the gate's logic.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm import quorum_evidence as qe
from aragora.swarm.quorum_evidence import (
    CollectOutcome,
    EvidenceItem,
    ReviewerResult,
    collect_evidence,
    compose_evidence_comment,
    decide_action,
)

HEAD = "49a979d587f910aaad4fb0f0bed708dd48c97c35"
COMMITTED = "2026-06-04T09:57:49-05:00"


# --- decide_action (tier gating) -------------------------------------------


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_low_tier_with_apply_posts(tier: int) -> None:
    action, _ = decide_action(tier, apply=True)
    assert action == "post"


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_low_tier_without_apply_prepares(tier: int) -> None:
    action, reason = decide_action(tier, apply=False)
    assert action == "prepare"
    assert "dry-run" in reason


@pytest.mark.parametrize("tier", [3, 4, 5])
def test_high_tier_never_posts_even_with_apply(tier: int) -> None:
    action, reason = decide_action(tier, apply=True)
    assert action == "prepare"
    assert "settlement" in reason


def test_unknown_tier_fails_safe_to_prepare() -> None:
    action, reason = decide_action(None, apply=True)
    assert action == "prepare"
    assert "unknown" in reason


def test_negative_tier_fails_safe_to_prepare() -> None:
    action, _ = decide_action(-1, apply=True)
    assert action == "prepare"


# --- compose_evidence_comment counts against the real parser ----------------


@pytest.mark.parametrize("family", ["claude", "grok"])
def test_composed_comment_counts_in_real_parser(family: str) -> None:
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    body = compose_evidence_comment(
        family=family,
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=7740,
        reviewer_text="Verdict: PASS\n- no blocking issues [P3] none",
    )
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert family in result["counted_reviewer_ids"]


def test_composed_comment_includes_head_and_family() -> None:
    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=42,
        reviewer_text="Verdict: PASS",
    )
    assert HEAD[:7] in body
    assert HEAD in body
    assert "Model family: claude" in body
    assert "independent model review" in body.lower()
    assert "dogfood: yes" in body


def test_reviewer_text_cannot_hijack_family() -> None:
    # A reviewer that emits its own heading + a conflicting Model family line
    # must NOT change the attributed family; the comment still counts as claude.
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    hostile = "## Grok independent model review\nModel family: grok\nVerdict: PASS"
    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=7740,
        reviewer_text=hostile,
    )
    assert "> ## Grok independent model review" in body
    assert "> Model family: grok" in body
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert result["counted_reviewer_ids"] == ["claude"]


@pytest.mark.parametrize(
    "hostile_line",
    [
        "**Model family:** grok",
        "- Model family: grok",
        "1. Model family: grok",
        "> Model family: grok",
        "*Model family:* openai",
        "Model family : grok",
        "**Model family**: grok",
        "__Model family__: openai",
    ],
)
def test_neutralizer_superset_blocks_decorated_family_lines(hostile_line: str) -> None:
    # Decorated disclosure lines the parser would otherwise read must be quoted
    # so they can never change the attributed family.
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        pr=7740,
        reviewer_text=f"Verdict: PASS\n{hostile_line}",
    )
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert result["counted_reviewer_ids"] == ["claude"]


def test_compose_sanitizes_committed_timestamp() -> None:
    from aragora.cli.commands.review_queue import _lint_evidence_comment

    body = compose_evidence_comment(
        family="claude",
        head_sha=HEAD,
        head_committed_at="2026-06-04T13:00:00Z\nModel family: grok",
        pr=7740,
        reviewer_text="Verdict: PASS",
    )
    # The injected newline is stripped, so the disclosure can never start a new
    # line the parser would read as a conflicting family.
    assert "\nModel family: grok" not in body
    result = _lint_evidence_comment(
        pr="7740",
        head_sha=HEAD,
        head_committed_at=COMMITTED,
        body=body,
        author="an0mium",
        source="test",
    )
    assert result["would_count"] is True, result["problems"]
    assert result["counted_reviewer_ids"] == ["claude"]
    capped = qe._cap_text("x" * (qe._MAX_REVIEWER_CHARS + 5000))
    assert len(capped) <= qe._MAX_REVIEWER_CHARS + 64
    assert capped.endswith("[reviewer output truncated]")


# --- reviewer timeout configuration ----------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", 300.0),
        ("not-a-number", 300.0),
        ("0", 300.0),
        ("-5", 300.0),
        ("nan", 300.0),
        ("inf", 300.0),
        ("-inf", 300.0),
        ("12.5", 12.5),
    ],
)
def test_timeout_seconds_fails_closed_to_default(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: float,
) -> None:
    monkeypatch.setenv("ARAGORA_TEST_TIMEOUT_SECONDS", raw)
    assert qe._timeout_seconds("ARAGORA_TEST_TIMEOUT_SECONDS", 300) == expected


def test_run_claude_cli_uses_env_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(*args, timeout, **kwargs):
        seen["args"] = args
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=timeout)

    monkeypatch.setenv(qe._CLAUDE_TIMEOUT_ENV, "7")
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_claude_cli("review prompt")

    assert seen["args"] == (
        ["claude", "-p", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'],
    )
    assert seen["timeout"] == 7.0
    assert result == ReviewerResult(
        "claude",
        "",
        False,
        "claude CLI timed out after 7s",
    )


def test_claude_reviewer_command_disables_mcp() -> None:
    cmd = qe._claude_reviewer_command()

    assert cmd[:2] == ["claude", "-p"]
    assert "--mcp-config" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert "--strict-mcp-config" in cmd


# --- OpenAI reviewer fallback ----------------------------------------------


def test_run_openai_reviewer_uses_api_when_openai_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_api_agent(family: str, prompt: str) -> ReviewerResult:
        calls.append((family, prompt))
        return ReviewerResult(family, "Verdict: PASS from API", True)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(qe, "_run_api_agent", fake_api_agent)

    result = qe._run_openai_reviewer("review prompt")

    assert result == ReviewerResult("openai", "Verdict: PASS from API", True)
    assert calls == [("openai", "review prompt")]


def test_run_openai_reviewer_without_api_key_uses_codex_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        seen.update(
            {
                "cmd": cmd,
                "input": input,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text("Verdict: PASS via Codex\n", encoding="utf-8")
        seen["output_path"] = output_path
        return subprocess.CompletedProcess(cmd, 0, stdout="ignored stdout", stderr="")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(qe._CODEX_TIMEOUT_ENV, "11")
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_openai_reviewer("review prompt")

    assert result == ReviewerResult(
        "openai",
        "Verdict: PASS via Codex",
        True,
        harness=qe._CODEX_OPENAI_HARNESS,
    )
    assert seen["cmd"] == [
        "codex",
        "exec",
        "--ignore-user-config",
        "-c",
        qe._CODEX_APPROVAL_POLICY_CONFIG,
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--output-last-message",
        str(seen["output_path"]),
        "--model",
        qe._CODEX_DEFAULT_MODEL,
        "-",
    ]
    assert "--ask-for-approval" not in seen["cmd"]
    assert seen["input"] == "review prompt"
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["timeout"] == 11.0
    assert seen["check"] is False
    assert not Path(seen["output_path"]).exists()


def test_run_openai_reviewer_retries_default_codex_model_selection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    output_paths: list[Path] = []

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        output_paths.append(Path(cmd[cmd.index("--output-last-message") + 1]))
        model = cmd[cmd.index("--model") + 1]
        if model == qe._CODEX_DEFAULT_MODELS[0]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=f"model {model} is not supported",
            )
        output_paths[-1].write_text("Verdict: PASS after fallback", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(qe._CODEX_MODEL_ENV, raising=False)
    monkeypatch.delenv(qe._CODEX_MODELS_ENV, raising=False)
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_openai_reviewer("review prompt")

    assert result == ReviewerResult(
        "openai",
        "Verdict: PASS after fallback",
        True,
        harness=qe._CODEX_OPENAI_HARNESS,
    )
    assert [call[call.index("--model") + 1] for call in calls] == list(qe._CODEX_DEFAULT_MODELS)
    assert all(not path.exists() for path in output_paths)


def test_run_openai_reviewer_respects_pinned_codex_model_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="model pinned-model is not supported",
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(qe._CODEX_MODEL_ENV, "pinned-model")
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_openai_reviewer("review prompt")

    assert result.ok is False
    assert "codex CLI exit 1: model pinned-model is not supported" in result.error
    assert len(calls) == 1
    assert calls[0][-3:] == ["--model", "pinned-model", "-"]


def test_run_openai_reviewer_passes_optional_codex_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text(
            "Verdict: PASS",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(qe._CODEX_MODEL_ENV, "gpt-5-codex")
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_openai_reviewer("review prompt")

    assert result.ok is True
    assert seen["cmd"][-3:] == ["--model", "gpt-5-codex", "-"]


def test_run_openai_reviewer_codex_failure_never_fabricates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="codex failed")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_openai_reviewer("review prompt")

    assert result.family == "openai"
    assert result.text == ""
    assert result.ok is False
    assert "codex CLI exit 1: codex failed" in result.error
    assert result.harness == ""


@pytest.mark.parametrize(
    ("exc", "expected_error"),
    [
        (
            subprocess.TimeoutExpired(cmd=["codex"], timeout=3),
            "codex CLI timed out after",
        ),
        (FileNotFoundError("missing codex"), "codex CLI not found on PATH"),
        (OSError("disk full"), "OSError: disk full"),
        (subprocess.SubprocessError("bad pipe"), "SubprocessError: bad pipe"),
    ],
)
def test_run_openai_reviewer_cleans_codex_output_file_when_subprocess_raises(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    expected_error: str,
) -> None:
    seen: dict[str, Path] = {}

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text("partial reviewer output", encoding="utf-8")
        seen["output_path"] = output_path
        raise exc

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(qe.subprocess, "run", fake_run)

    result = qe._run_openai_reviewer("review prompt")

    assert result.family == "openai"
    assert result.text == ""
    assert result.ok is False
    assert expected_error in result.error
    assert "output_path" in seen
    assert not seen["output_path"].exists()


# --- default API reviewer cleanup ------------------------------------------


def test_run_api_agent_closes_agent_and_shared_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append(f"generate:{prompt}")
            return "Verdict: PASS"

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        events.append(f"create:{family}:{name}:{role}")
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent_in_current_process("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == [
        "create:grok:grok_reviewer:critic",
        "generate:review prompt",
        "agent_close",
        "connector_close",
    ]


def test_run_api_agent_closes_resources_after_generate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            raise RuntimeError("model failed")

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent_in_current_process("grok", "review prompt")

    assert result.ok is False
    assert "RuntimeError: model failed" in result.error
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_timeout_terminates_blocked_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qe, "_REVIEWER_TIMEOUT", 0.05)
    monkeypatch.setattr(qe, "_REVIEWER_CLEANUP_TIMEOUT", 0.01)
    events: list[str] = []

    class FakeQueue:
        def get(self, timeout: float):
            raise AssertionError("timed-out worker result must not be read")

    class FakeContext:
        def Queue(self, maxsize: int) -> FakeQueue:
            assert maxsize == 1
            return FakeQueue()

    class FakeProcess:
        def start(self) -> None:
            events.append("start")

        def join(self, timeout: float) -> None:
            events.append(f"join:{timeout:.2f}")

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

    monkeypatch.setattr(qe, "_api_agent_process_context", lambda: FakeContext(), raising=False)
    monkeypatch.setattr(
        qe,
        "_start_api_agent_worker_process",
        lambda ctx, family, prompt, result_queue: FakeProcess(),
        raising=False,
    )

    result = qe._run_api_agent("grok", "review prompt")

    assert result.ok is False
    assert "timed out" in result.error
    assert events == ["start", "join:0.06", "terminate", "join:5.00", "kill", "join:5.00"]


def test_run_api_agent_parent_timeout_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(qe._REVIEWER_TIMEOUT_ENV, "1200")
    monkeypatch.setattr(qe, "_REVIEWER_TIMEOUT", 300)
    monkeypatch.setattr(qe, "_REVIEWER_CLEANUP_TIMEOUT", 7)
    events: list[str] = []

    class FakeQueue:
        def get(self, timeout: float):
            raise AssertionError("timed-out worker result must not be read")

    class FakeContext:
        def Queue(self, maxsize: int) -> FakeQueue:
            assert maxsize == 1
            return FakeQueue()

    class FakeProcess:
        def start(self) -> None:
            events.append("start")

        def join(self, timeout: float) -> None:
            events.append(f"join:{timeout:g}")

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

    monkeypatch.setattr(qe, "_api_agent_process_context", lambda: FakeContext(), raising=False)
    monkeypatch.setattr(
        qe,
        "_start_api_agent_worker_process",
        lambda ctx, family, prompt, result_queue: FakeProcess(),
        raising=False,
    )

    result = qe._run_api_agent("grok", "review prompt")

    assert result.ok is False
    assert result.error == "grok reviewer timed out after 1200s"
    assert events == ["start", "join:1207", "terminate", "join:5", "kill", "join:5"]


def test_api_agent_cleanup_does_not_hang_on_stuck_agent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def close(self) -> None:
            events.append("agent_close_start")
            await asyncio.sleep(3600)

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    from aragora.agents.api_agents import common

    monkeypatch.setattr(qe, "_REVIEWER_CLEANUP_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    asyncio.run(asyncio.wait_for(qe._close_api_agent_resources(FakeAgent()), timeout=0.05))

    assert events == ["agent_close_start", "connector_close"]


def test_api_agent_cleanup_does_not_hang_on_stuck_shared_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def close(self) -> None:
            events.append("agent_close")

    async def fake_close_shared_connector() -> None:
        events.append("connector_close_start")
        await asyncio.sleep(3600)

    from aragora.agents.api_agents import common

    monkeypatch.setattr(qe, "_REVIEWER_CLEANUP_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    asyncio.run(asyncio.wait_for(qe._close_api_agent_resources(FakeAgent()), timeout=0.05))

    assert events == ["agent_close", "connector_close_start"]


def test_run_api_agent_closes_shared_connector_after_agent_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

        async def close(self) -> None:
            events.append("agent_close")
            raise RuntimeError("close failed")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent_in_current_process("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_closes_shared_connector_without_agent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent_in_current_process("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "connector_close"]


def test_run_api_agent_supports_sync_agent_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

        def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent_in_current_process("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_keeps_result_when_shared_connector_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append("generate")
            return "Verdict: PASS"

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")
        raise RuntimeError("connector failed")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    result = qe._run_api_agent_in_current_process("grok", "review prompt")

    assert result == ReviewerResult("grok", "Verdict: PASS", True)
    assert events == ["generate", "agent_close", "connector_close"]


def test_run_api_agent_allows_consecutive_one_shot_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAgent:
        async def generate(self, prompt: str) -> str:
            events.append(f"generate:{prompt}")
            return f"Verdict: PASS {prompt}"

        async def close(self) -> None:
            events.append("agent_close")

    def fake_create_agent(family: str, *, name: str, role: str) -> FakeAgent:
        events.append(f"create:{family}")
        return FakeAgent()

    async def fake_close_shared_connector() -> None:
        events.append("connector_close")

    import aragora.agents
    from aragora.agents.api_agents import common

    monkeypatch.setattr(aragora.agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(common, "close_shared_connector", fake_close_shared_connector)

    first = qe._run_api_agent_in_current_process("grok", "one")
    second = qe._run_api_agent_in_current_process("grok", "two")

    assert first == ReviewerResult("grok", "Verdict: PASS one", True)
    assert second == ReviewerResult("grok", "Verdict: PASS two", True)
    assert events == [
        "create:grok",
        "generate:one",
        "agent_close",
        "connector_close",
        "create:grok",
        "generate:two",
        "agent_close",
        "connector_close",
    ]


# --- collect_evidence orchestration (fully offline via injected callables) ---


def _fakes(*, tier: int, head: str = HEAD, would_count: bool = True):
    posted: list[tuple[str, str]] = []

    def context_fetcher(repo: str, pr: int) -> dict:
        return {"head_sha": head, "head_committed_at": COMMITTED}

    def tier_fetcher(repo: str, pr: int):
        return tier

    def prompt_builder(repo: str, pr: int, ctx: dict) -> str:
        return "review prompt"

    def reviewer_runner(family: str, prompt: str) -> ReviewerResult:
        return ReviewerResult(family, f"Verdict: PASS from {family}", True)

    def linter(pr, head_sha, head_committed_at, author, body, env) -> dict:
        return {
            "would_count": would_count,
            "counted_reviewer_ids": [body.split()[1].lower()] if would_count else [],
            "problems": [] if would_count else ["no_counted_model_family"],
        }

    def poster(repo: str, pr: int, body: str) -> None:
        posted.append((repo, body))

    return dict(
        context_fetcher=context_fetcher,
        tier_fetcher=tier_fetcher,
        prompt_builder=prompt_builder,
        reviewer_runner=reviewer_runner,
        linter=linter,
        poster=poster,
    ), posted


def test_collect_low_tier_apply_posts_both() -> None:
    fakes, posted = _fakes(tier=1)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "post"
    assert sorted(outcome.posted) == ["claude", "grok"]
    assert len(posted) == 2


def test_collect_low_tier_apply_triggers_same_pr_quorum_reconciler_after_posts() -> None:
    fakes, posted = _fakes(tier=1)
    calls: list[tuple[str, int, int]] = []

    def quorum_reconciler(repo: str, pr: int) -> dict:
        calls.append((repo, pr, len(posted)))
        return {"should_rerun": True, "run_id": 123, "applied": True}

    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "grok"],
        author="me",
        apply=True,
        quorum_reconciler=quorum_reconciler,
        **fakes,
    )

    assert calls == [("o/r", 1, 2)]
    assert outcome.quorum_rerun == {"should_rerun": True, "run_id": 123, "applied": True}


def test_collect_low_tier_apply_prepares_only_when_reviewer_dissents() -> None:
    fakes, posted = _fakes(tier=1)
    calls: list[tuple[str, int]] = []

    def reviewer_runner(family: str, prompt: str) -> ReviewerResult:
        if family == "grok":
            return ReviewerResult("grok", "Verdict: CHANGES-REQUESTED\n- [P1] blocker", True)
        return ReviewerResult("claude", "Verdict: PASS\n- no blockers", True)

    def quorum_reconciler(repo: str, pr: int) -> dict:
        calls.append((repo, pr))
        return {"applied": True}

    fakes["reviewer_runner"] = reviewer_runner
    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "grok"],
        author="me",
        apply=True,
        quorum_reconciler=quorum_reconciler,
        **fakes,
    )

    assert outcome.action == "prepare"
    assert "reviewer dissent" in outcome.action_reason
    assert outcome.dissenting_families == ["grok"]
    assert posted == []
    assert calls == []
    assert outcome.quorum_rerun is None


def test_collect_low_tier_apply_prepares_when_supportive_quorum_incomplete() -> None:
    fakes, posted = _fakes(tier=1)
    calls: list[tuple[str, int]] = []

    def reviewer_runner(family: str, prompt: str) -> ReviewerResult:
        if family == "grok":
            return ReviewerResult("grok", "Verdict: inconclusive\n- unsure", True)
        return ReviewerResult("claude", "Verdict: PASS\n- no blockers", True)

    def quorum_reconciler(repo: str, pr: int) -> dict:
        calls.append((repo, pr))
        return {"applied": True}

    fakes["reviewer_runner"] = reviewer_runner
    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "grok"],
        author="me",
        apply=True,
        quorum_reconciler=quorum_reconciler,
        **fakes,
    )

    assert outcome.action == "prepare"
    assert "supportive quorum incomplete" in outcome.action_reason
    assert outcome.supportive_families == ["claude"]
    assert posted == []
    assert calls == []
    assert outcome.quorum_rerun is None


def test_collect_success_requires_two_supportive_reviewers() -> None:
    fakes, _posted = _fakes(tier=1)

    def reviewer_runner(family: str, prompt: str) -> ReviewerResult:
        if family == "grok":
            return ReviewerResult("grok", "Verdict: CHANGES-REQUESTED\n- [P1] blocker", True)
        return ReviewerResult("claude", "Verdict: PASS\n- no blockers", True)

    fakes["reviewer_runner"] = reviewer_runner
    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "grok"],
        author="me",
        apply=False,
        **fakes,
    )

    assert outcome.counting_families == ["claude", "grok"]
    assert outcome.supportive_families == ["claude"]
    assert outcome.dissenting_families == ["grok"]
    assert outcome.has_supportive_quorum is False


def test_locked_quorum_state_recovers_stale_pid_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state_file = tmp_path / "state.json"
    lock_file = tmp_path / "state.json.lock"
    lock_file.write_text("pid=999999 acquired_at=2026-06-06T00:00:00+00:00\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock_file, (old, old))
    monkeypatch.setattr(qe, "QUORUM_STATE_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(qe, "QUORUM_STATE_LOCK_POLL_SECONDS", 0.01)

    with qe._locked_quorum_reconcile_state(state_file):
        assert lock_file.exists()

    assert not lock_file.exists()


def test_collect_records_quorum_reconciler_error_after_successful_posts() -> None:
    fakes, posted = _fakes(tier=1)

    def quorum_reconciler(repo: str, pr: int) -> dict:
        raise RuntimeError("rerun surface unavailable")

    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "grok"],
        author="me",
        apply=True,
        quorum_reconciler=quorum_reconciler,
        **fakes,
    )

    assert len(posted) == 2
    assert sorted(outcome.posted) == ["claude", "grok"]
    assert outcome.quorum_rerun == {"applied": False, "error": "rerun surface unavailable"}


def test_collect_does_not_reconcile_when_no_evidence_was_posted() -> None:
    fakes, posted = _fakes(tier=1, would_count=False)
    calls: list[tuple[str, int]] = []

    def quorum_reconciler(repo: str, pr: int) -> dict:
        calls.append((repo, pr))
        return {"applied": True}

    outcome = collect_evidence(
        repo="o/r",
        pr=1,
        families=["claude", "grok"],
        author="me",
        apply=True,
        quorum_reconciler=quorum_reconciler,
        **fakes,
    )

    assert posted == []
    assert calls == []
    assert outcome.quorum_rerun is None


def test_default_quorum_reconciler_holds_lock_through_state_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from scripts import reconcile_merge_quorum

    events: list[str] = []
    state: dict = {}

    @contextmanager
    def fake_lock(path):
        events.append("lock-enter")
        yield
        events.append("lock-exit")

    def load_state(path):
        events.append("load")
        return state

    def evaluate_pr(repo, pr, *, now, state, cooldown_seconds, max_reruns):
        events.append("evaluate")
        assert repo == "o/r"
        assert pr == 1
        assert cooldown_seconds == qe.QUORUM_RERUN_COOLDOWN_SECONDS
        assert max_reruns == qe.QUORUM_RERUN_MAX_PER_HEAD
        decision = SimpleNamespace(
            should_rerun=True,
            reason="stale-success-after-new-evidence",
            run_id=123,
            next_prompt=None,
        )
        quorum_run = SimpleNamespace(run_id=123, head_sha="abc123")
        return decision, quorum_run

    def execute_rerun(repo, run_id):
        events.append("execute")
        assert (repo, run_id) == ("o/r", 123)
        return True

    def save_state(path, next_state):
        events.append("save")
        assert next_state["abc123"]["count"] == 1

    monkeypatch.setattr(qe, "_locked_quorum_reconcile_state", fake_lock)
    monkeypatch.setattr(reconcile_merge_quorum, "DEFAULT_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(reconcile_merge_quorum, "_load_state", load_state)
    monkeypatch.setattr(reconcile_merge_quorum, "evaluate_pr", evaluate_pr)
    monkeypatch.setattr(reconcile_merge_quorum, "execute_rerun", execute_rerun)
    monkeypatch.setattr(reconcile_merge_quorum, "_save_state", save_state)

    record = qe.default_quorum_reconciler("o/r", 1)

    assert record == {
        "should_rerun": True,
        "reason": "stale-success-after-new-evidence",
        "run_id": 123,
        "applied": True,
    }
    assert events == ["lock-enter", "load", "evaluate", "execute", "save", "lock-exit"]


def test_default_quorum_reconciler_rechecks_rerun_cap_under_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from scripts import reconcile_merge_quorum

    state = {
        "abc123": {
            "count": qe.QUORUM_RERUN_MAX_PER_HEAD,
            "last_rerun_at": None,
        }
    }

    @contextmanager
    def fake_lock(path):
        yield

    def evaluate_pr(repo, pr, *, now, state, cooldown_seconds, max_reruns):
        decision = SimpleNamespace(
            should_rerun=True,
            reason="stale-success-after-new-evidence",
            run_id=123,
            next_prompt=None,
        )
        quorum_run = SimpleNamespace(run_id=123, head_sha="abc123")
        return decision, quorum_run

    def execute_rerun(repo, run_id):
        raise AssertionError("rerun must not execute after locked count reaches the cap")

    monkeypatch.setattr(qe, "_locked_quorum_reconcile_state", fake_lock)
    monkeypatch.setattr(reconcile_merge_quorum, "DEFAULT_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(reconcile_merge_quorum, "_load_state", lambda path: state)
    monkeypatch.setattr(reconcile_merge_quorum, "evaluate_pr", evaluate_pr)
    monkeypatch.setattr(reconcile_merge_quorum, "execute_rerun", execute_rerun)

    record = qe.default_quorum_reconciler("o/r", 1)

    assert record == {
        "should_rerun": False,
        "reason": "max_reruns_reached_in_locked_state",
        "run_id": 123,
        "applied": False,
    }


def test_collect_high_tier_apply_never_posts() -> None:
    fakes, posted = _fakes(tier=4)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert outcome.posted == []
    assert posted == []
    # Evidence is still composed + validated for the operator.
    assert sorted(outcome.counting_families) == ["claude", "grok"]


def test_collect_dry_run_prepares_without_posting() -> None:
    fakes, posted = _fakes(tier=1)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=False, **fakes
    )
    assert outcome.action == "prepare"
    assert posted == []
    assert sorted(outcome.counting_families) == ["claude", "grok"]


def test_collect_carries_reviewer_harness_into_comment() -> None:
    fakes, posted = _fakes(tier=1)

    def harness_runner(family: str, prompt: str) -> ReviewerResult:
        return ReviewerResult(
            family,
            "Verdict: PASS via harness",
            True,
            harness=qe._CODEX_OPENAI_HARNESS,
        )

    def harness_linter(pr, head_sha, head_committed_at, author, body, env) -> dict:
        assert qe._CODEX_OPENAI_HARNESS in body
        return {
            "would_count": True,
            "counted_reviewer_ids": ["openai"],
            "problems": [],
        }

    fakes["reviewer_runner"] = harness_runner
    fakes["linter"] = harness_linter

    outcome = collect_evidence(
        repo="o/r", pr=1, families=["openai"], author="me", apply=False, **fakes
    )

    assert outcome.action == "prepare"
    assert posted == []
    assert outcome.counting_families == ["openai"]
    assert qe._CODEX_OPENAI_HARNESS in outcome.items[0].body


def test_collect_never_fabricates_on_reviewer_failure() -> None:
    fakes, posted = _fakes(tier=1)

    def failing_runner(family: str, prompt: str) -> ReviewerResult:
        if family == "grok":
            return ReviewerResult("grok", "", False, "timeout")
        return ReviewerResult(family, "Verdict: PASS from claude", True)

    fakes["reviewer_runner"] = failing_runner
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert [f.family for f in outcome.failures] == ["grok"]
    assert outcome.action == "prepare"
    assert "supportive quorum incomplete" in outcome.action_reason
    assert outcome.posted == []
    assert posted == []


def test_collect_does_not_post_uncountable_evidence() -> None:
    fakes, posted = _fakes(tier=1, would_count=False)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert "supportive quorum incomplete" in outcome.action_reason
    assert outcome.posted == []
    assert posted == []
    assert all(not item.would_count for item in outcome.items)


def test_collect_rejects_unsupported_family() -> None:
    fakes, posted = _fakes(tier=1)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "bogus"], author="me", apply=True, **fakes
    )
    assert "bogus" in [f.family for f in outcome.failures]
    assert outcome.action == "prepare"
    assert "supportive quorum incomplete" in outcome.action_reason
    assert outcome.posted == []
    assert posted == []
    assert "bogus" not in outcome.counting_families


def test_collect_records_post_errors_without_losing_others() -> None:
    fakes, _ = _fakes(tier=1)

    def flaky_poster(repo: str, pr: int, body: str) -> None:
        if "Grok" in body:
            raise RuntimeError("gh rejected comment")

    fakes["poster"] = flaky_poster
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.posted == ["claude"]
    assert any("grok" in e for e in outcome.post_errors)


def test_collect_recheck_exception_prepares_without_posting() -> None:
    fakes, posted = _fakes(tier=1)
    calls = {"n": 0}

    def flaky_context(repo: str, pr: int) -> dict:
        calls["n"] += 1
        if calls["n"] >= 2:  # first call ok, recheck blows up
            raise RuntimeError("transient gh error")
        return {"head_sha": HEAD, "head_committed_at": COMMITTED}

    fakes["context_fetcher"] = flaky_context
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert "re-verify" in outcome.action_reason
    assert posted == []


def test_collect_skips_post_when_head_moves_before_posting() -> None:
    fakes, posted = _fakes(tier=1)
    heads = iter([HEAD, "0" * 40])  # initial fetch, then recheck = moved head

    def moving_context(repo: str, pr: int) -> dict:
        return {"head_sha": next(heads), "head_committed_at": COMMITTED}

    fakes["context_fetcher"] = moving_context
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert "changed before posting" in outcome.action_reason
    assert posted == []


def test_collect_skips_post_when_tier_promoted_before_posting() -> None:
    fakes, posted = _fakes(tier=1)
    tiers = iter([1, 4])  # initial low, recheck promoted to settlement tier

    def promoting_tier(repo: str, pr: int) -> int:
        return next(tiers)

    fakes["tier_fetcher"] = promoting_tier
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "grok"], author="me", apply=True, **fakes
    )
    assert outcome.action == "prepare"
    assert posted == []


def test_collect_dedupes_families() -> None:
    fakes, _ = _fakes(tier=4)
    outcome = collect_evidence(
        repo="o/r", pr=1, families=["claude", "Claude", "grok"], author="me", apply=False, **fakes
    )
    assert [item.family for item in outcome.items] == ["claude", "grok"]


def test_collect_missing_head_raises() -> None:
    fakes, _ = _fakes(tier=1, head="")
    with pytest.raises(ValueError):
        collect_evidence(repo="o/r", pr=1, families=["claude"], author="me", apply=True, **fakes)


# --- run_collect_cli (monkeypatched orchestrator) ---------------------------


def test_run_collect_cli_exit_code_quorum_met(monkeypatch, capsys) -> None:
    def fake_collect(**kwargs) -> CollectOutcome:
        return CollectOutcome(
            repo="o/r",
            pr=1,
            head_sha=HEAD,
            head_committed_at=COMMITTED,
            tier=1,
            action="post",
            action_reason="ok",
            items=[
                EvidenceItem("claude", "body", True, ["claude"], [], "pass"),
                EvidenceItem("grok", "body", True, ["grok"], [], "pass"),
            ],
            posted=["claude", "grok"],
        )

    monkeypatch.setattr(qe, "collect_evidence", fake_collect)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=True, json_output=True
    )
    assert rc == 0
    assert "collect_evidence" in capsys.readouterr().out


def test_run_collect_cli_exit_code_quorum_incomplete(monkeypatch) -> None:
    def fake_collect(**kwargs) -> CollectOutcome:
        return CollectOutcome(
            repo="o/r",
            pr=1,
            head_sha=HEAD,
            head_committed_at=COMMITTED,
            tier=4,
            action="prepare",
            action_reason="settlement",
            items=[EvidenceItem("claude", "body", True, ["claude"], [])],
        )

    monkeypatch.setattr(qe, "collect_evidence", fake_collect)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=False, json_output=False
    )
    assert rc == 1


def test_run_collect_cli_error_path(monkeypatch, capsys) -> None:
    def boom(**kwargs):
        raise ValueError("no head")

    monkeypatch.setattr(qe, "collect_evidence", boom)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=False, json_output=True
    )
    assert rc == 1
    assert "no head" in capsys.readouterr().out


def test_run_collect_cli_catches_runtime_error(monkeypatch, capsys) -> None:
    def boom(**kwargs):
        raise RuntimeError("empty diff")

    monkeypatch.setattr(qe, "collect_evidence", boom)
    monkeypatch.setattr(qe, "resolve_author", lambda default="local": "me")
    rc = qe.run_collect_cli(
        repo="o/r", pr=1, families=None, author=None, apply=False, json_output=False
    )
    assert rc == 1
    assert "empty diff" in capsys.readouterr().out


# --- build_review_prompt: complete file list + fair per-file body bounding ---


def _diff_with_deletion_before_additions() -> tuple[str, str, list[str]]:
    """A unified diff whose large DELETION sorts (alphabetically) before its
    ADDITIONS, mirroring PR #8416 (``tests/conftest.py`` deleted before
    ``tests/fixtures/*`` added). A blind ``diff[:N]`` slice would drop the
    additions entirely. Returns ``(diff, name_status, added_paths)``.
    """
    deleted = "tests/conftest.py"
    added = ["tests/fixtures/alpha.py", "tests/fixtures/beta.py"]
    big_body = "\n".join(f"-old conftest line {i} " + "x" * 60 for i in range(2000))
    deletion = (
        f"diff --git a/{deleted} b/{deleted}\n"
        "deleted file mode 100644\n"
        f"--- a/{deleted}\n+++ /dev/null\n@@ -1,2000 +0,0 @@\n{big_body}\n"
    )
    additions = ""
    for path in added:
        body = "\n".join(f"+fixture {path} line {i}" for i in range(40))
        additions += (
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,40 @@\n{body}\n"
        )
    diff = deletion + additions
    name_status = f"D\t{deleted}\n" + "".join(f"A\t{p}\n" for p in added)
    return diff, name_status, added


def test_build_review_prompt_keeps_all_added_paths_when_deletion_sorts_first() -> None:
    diff, name_status, added = _diff_with_deletion_before_additions()
    assert len(diff) > qe._MAX_DIFF_CHARS  # body alone forces bounding
    prompt = qe.build_review_prompt(
        repo="o/r", pr=8416, head_sha=HEAD, diff_text=diff, name_status=name_status
    )
    # Every added path survives even though the diff body had to be bounded.
    for path in added:
        assert path in prompt


def test_build_review_prompt_never_truncates_complete_file_list_header() -> None:
    diff, name_status, added = _diff_with_deletion_before_additions()
    prompt = qe.build_review_prompt(
        repo="o/r", pr=8416, head_sha=HEAD, diff_text=diff, name_status=name_status
    )
    header = prompt[: prompt.index("=== DIFF")]
    # The file-list header is placed before the (bounded) body and is complete.
    assert "=== CHANGED FILES" in header
    assert "tests/conftest.py" in header
    for path in added:
        assert path in header
    # The body really was bounded (per-file marker present), proving the header
    # survived truncation rather than the diff simply being small.
    assert qe._PER_FILE_TRUNCATION_MARKER.strip() in prompt


def test_build_review_prompt_derives_file_list_when_name_status_absent() -> None:
    diff, _name_status, added = _diff_with_deletion_before_additions()
    # No name_status supplied: the file list is recovered from the diff headers,
    # so a reviewer still cannot claim an added file is absent.
    prompt = qe.build_review_prompt(repo="o/r", pr=8416, head_sha=HEAD, diff_text=diff)
    header = prompt[: prompt.index("=== DIFF")]
    assert "tests/conftest.py" in header
    for path in added:
        assert path in header


def test_build_review_prompt_small_diff_is_not_truncated() -> None:
    diff = (
        "diff --git a/aragora/x.py b/aragora/x.py\n"
        "--- a/aragora/x.py\n+++ b/aragora/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    prompt = qe.build_review_prompt(
        repo="o/r", pr=42, head_sha=HEAD, diff_text=diff, name_status="M\taragora/x.py\n"
    )
    # Shape semantics unchanged: a str grounded on the short head, carrying the
    # verdict instruction and the changed file; nothing truncated.
    assert isinstance(prompt, str)
    assert HEAD[:7] in prompt
    assert "Verdict: PASS" in prompt
    assert "Verdict: CHANGES-REQUESTED" in prompt
    assert "aragora/x.py" in prompt
    assert "truncated" not in prompt


# --- _bound_diff_body: fair per-file water-filling --------------------------


def test_bound_diff_body_returns_input_unchanged_when_within_cap() -> None:
    diff = "diff --git a/a b/a\n+hello\n"
    bounded, truncated = qe._bound_diff_body(diff, 30_000)
    assert truncated is False
    assert bounded == diff


def test_bound_diff_body_gives_every_file_a_hunk() -> None:
    seg_a = "diff --git a/a b/a\n" + "A" * 50_000 + "\n"
    seg_b = "diff --git a/b b/b\n" + "B" * 50_000 + "\n"
    seg_c = "diff --git a/c b/c\n" + "C" * 50_000 + "\n"
    bounded, truncated = qe._bound_diff_body(seg_a + seg_b + seg_c, 30_000)
    assert truncated is True
    # No single file may consume the whole budget: each file's header (and some
    # content) survives, unlike a blind first-N-bytes slice.
    assert "diff --git a/a b/a" in bounded
    assert "diff --git a/b b/b" in bounded
    assert "diff --git a/c b/c" in bounded
    assert "B" in bounded and "C" in bounded
    marker_overhead = 3 * len(qe._PER_FILE_TRUNCATION_MARKER)
    assert len(bounded) <= 30_000 + marker_overhead + 8


def test_bound_diff_body_single_file_falls_back_to_head_slice() -> None:
    diff = "diff --git a/a b/a\n" + "A" * 100_000
    bounded, truncated = qe._bound_diff_body(diff, 30_000)
    assert truncated is True
    assert bounded.startswith("diff --git a/a b/a")
    assert len(bounded) <= 30_000 + len(qe._PER_FILE_TRUNCATION_MARKER) + 8


def test_bound_diff_body_small_addition_kept_whole_when_deletion_huge() -> None:
    # The fairness property that fixes #8416: a small addition that sorts AFTER a
    # huge deletion is kept in full and never dropped.
    deletion = "diff --git a/del b/del\n" + "D" * 200_000 + "\n"
    addition = "diff --git a/add b/add\n+only forty chars of new content here\n"
    bounded, truncated = qe._bound_diff_body(deletion + addition, 60_000)
    assert truncated is True
    assert "only forty chars of new content here" in bounded


# --- default_prompt_builder: name-status pass-through + graceful fallback ----


def _prompt_builder_run_stub(diff: str, name_status: str | None):
    """Build a fake ``merge_quorum_io.run`` serving diff / name-status / head."""

    def fake_run(args, *, env=None, timeout=None):
        if args[:3] == ["gh", "pr", "diff"]:
            if "--name-status" in args:
                if name_status is None:
                    return SimpleNamespace(returncode=1, stdout="", stderr="boom")
                return SimpleNamespace(returncode=0, stdout=name_status, stderr="")
            return SimpleNamespace(returncode=0, stdout=diff, stderr="")
        if args[:3] == ["gh", "pr", "view"]:
            return SimpleNamespace(returncode=0, stdout=HEAD + "\n", stderr="")
        raise AssertionError(f"unexpected args: {args}")

    return fake_run


def test_default_prompt_builder_prepends_name_status(monkeypatch) -> None:
    diff, name_status, added = _diff_with_deletion_before_additions()
    monkeypatch.setattr(qe.merge_quorum_io, "run", _prompt_builder_run_stub(diff, name_status))
    prompt = qe.default_prompt_builder("o/r", 8416, {"head_sha": HEAD})
    header = prompt[: prompt.index("=== DIFF")]
    assert "tests/conftest.py" in header
    for path in added:
        assert path in header


def test_default_prompt_builder_tolerates_name_status_fetch_failure(monkeypatch) -> None:
    diff, _name_status, added = _diff_with_deletion_before_additions()
    # name-status fetch fails: the builder must NOT raise and must still produce a
    # complete file list (recovered from the diff) -- return semantics unchanged.
    monkeypatch.setattr(qe.merge_quorum_io, "run", _prompt_builder_run_stub(diff, None))
    prompt = qe.default_prompt_builder("o/r", 8416, {"head_sha": HEAD})
    header = prompt[: prompt.index("=== DIFF")]
    for path in added:
        assert path in header


def test_default_prompt_builder_empty_diff_still_raises(monkeypatch) -> None:
    # Builder exit/return semantics unchanged: an empty diff is still a hard error.
    monkeypatch.setattr(qe.merge_quorum_io, "run", _prompt_builder_run_stub("   \n", "D\tx\n"))
    with pytest.raises(RuntimeError, match="empty diff"):
        qe.default_prompt_builder("o/r", 8416, {"head_sha": HEAD})
