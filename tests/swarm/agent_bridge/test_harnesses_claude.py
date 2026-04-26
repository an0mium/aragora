from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from aragora.swarm.agent_bridge.harnesses.claude import ClaudeTransport


class FakeRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr="")


def _fixture_text(name: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "tests" / "fixtures" / "agent_bridge" / name).read_text(encoding="utf-8")


def test_claude_launch_assigns_uuid4_and_resume_uses_resume_flag(tmp_path: Path) -> None:
    fake_runner = FakeRunner(_fixture_text("claude_start.txt"))
    session_id = uuid.UUID("123e4567-e89b-42d3-a456-426614174000")
    transport = ClaudeTransport(
        cwd=tmp_path,
        model="claude-opus-4-7",
        runner=fake_runner,
        binary_resolver=lambda _: "/usr/bin/claude",
        uuid_factory=lambda: session_id,
    )

    launched = transport.launch("Review this", allowed_roles={"reviewer"})
    fake_runner.stdout = _fixture_text("claude_resume.txt")
    resumed = transport.resume(str(session_id), "Repair this", allowed_roles={"reviewer"})

    assert launched.session_id == str(session_id)
    assert launched.parsed_turn.parse_status == "ok"
    assert fake_runner.commands[0] == [
        "claude",
        "-p",
        "--session-id",
        str(session_id),
        "--model",
        "claude-opus-4-7",
        "Review this",
    ]
    assert fake_runner.commands[1] == [
        "claude",
        "-p",
        "--resume",
        str(session_id),
        "--model",
        "claude-opus-4-7",
        "Repair this",
    ]
    assert resumed.session_id == str(session_id)
