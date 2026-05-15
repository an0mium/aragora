from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aragora.swarm.agent_bridge.exceptions import TransportOutputParseError
from aragora.swarm.agent_bridge.harnesses.codex import CodexTransport


def _fixture_text(name: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "tests" / "fixtures" / "agent_bridge" / name).read_text(encoding="utf-8")


class FakeRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.commands: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        self.kwargs.append(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr="")


def test_codex_launch_parses_thread_started_agent_messages_and_usage(tmp_path: Path) -> None:
    fake_runner = FakeRunner(_fixture_text("codex_start.jsonl"))
    transport = CodexTransport(
        cwd=tmp_path,
        model="gpt-5.4",
        runner=fake_runner,
        binary_resolver=lambda _: "/usr/bin/codex",
    )

    result = transport.launch("Review the plan", allowed_roles={"reviewer", "implementer"})

    assert result.session_id == "019db172-4d01-7072-860c-99114afe8792"
    assert result.usage == {"input_tokens": 27138, "output_tokens": 64}
    assert result.parsed_turn.parse_status == "ok"
    assert result.parsed_turn.footer is not None
    assert result.parsed_turn.footer.next_actor == "implementer"
    assert fake_runner.commands[0] == [
        "codex",
        "exec",
        "--json",
        "--model",
        "gpt-5.4",
        "Review the plan",
    ]
    assert fake_runner.kwargs[0]["stdin"] == subprocess.DEVNULL


def test_codex_plaintext_fallback_parses_session_id_and_footer(tmp_path: Path) -> None:
    transport = CodexTransport(
        cwd=tmp_path,
        binary_resolver=lambda _: "/usr/bin/codex",
    )
    raw_stdout = (
        "session id: 019db152-df99-77c0-9863-08cf1a2a994f\n\n"
        "Fallback body.\n\n"
        "---BRIDGE-FOOTER---\n"
        "summary: Used plaintext fallback\n"
        "next_actor: reviewer\n"
        "needs_human: false\n"
        "done: false\n"
        "artifacts: []\n"
        "tests_run: []\n"
        "---BRIDGE-FOOTER-END---\n"
    )

    parsed = transport.parse_output(
        raw_stdout,
        allowed_roles={"reviewer"},
        session_id=None,
        is_resume=False,
    )

    assert parsed.session_id == "019db152-df99-77c0-9863-08cf1a2a994f"
    assert parsed.parsed_turn.parse_status == "ok"
    assert parsed.parsed_turn.footer is not None
    assert parsed.parsed_turn.footer.summary == "Used plaintext fallback"


def test_codex_resume_rejects_mismatched_thread_started(tmp_path: Path) -> None:
    transport = CodexTransport(
        cwd=tmp_path,
        binary_resolver=lambda _: "/usr/bin/codex",
    )
    raw_stdout = '{"type":"thread.started","thread_id":"unexpected-thread"}\n' + _fixture_text(
        "codex_resume.jsonl"
    )

    with pytest.raises(TransportOutputParseError, match="mismatched thread.started"):
        transport.parse_output(
            raw_stdout,
            allowed_roles={"reviewer", "implementer"},
            session_id="expected-thread",
            is_resume=True,
        )
