from __future__ import annotations

import json
import subprocess
from pathlib import Path

from aragora.swarm.agent_bridge.harnesses.droid import DroidTransport
from aragora.swarm.agent_bridge.harnesses import create_transport


def _fixture_text(name: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return (root / "tests" / "fixtures" / "agent_bridge" / name).read_text(encoding="utf-8")


class FakeRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr="")


def test_droid_launch_parses_session_id_usage_and_default_auto_high(tmp_path: Path) -> None:
    fake_runner = FakeRunner(_fixture_text("droid_start.json"))
    transport = DroidTransport(
        cwd=tmp_path,
        model="claude-opus-4-7",
        runner=fake_runner,
        binary_resolver=lambda _: "/usr/bin/droid",
    )

    result = transport.launch("Synthesize", allowed_roles={"reviewer", "implementer"})

    assert result.session_id == "16329fce-3484-47a4-ad98-6676fdfb7477"
    assert result.usage["input_tokens"] == 6
    assert result.parsed_turn.parse_status == "ok"
    assert fake_runner.commands[0] == [
        "droid",
        "exec",
        "--auto",
        "high",
        "--output-format",
        "json",
        "--model",
        "claude-opus-4-7",
        "--cwd",
        str(tmp_path),
        "Synthesize",
    ]


def test_droid_filesystem_fallback_discovers_session_id(tmp_path: Path) -> None:
    payload = json.loads(_fixture_text("droid_resume.json"))
    payload.pop("session_id")
    home = tmp_path / "home"
    session_dir = home / ".factory" / "sessions" / str(tmp_path.resolve()).replace("/", "-")
    session_dir.mkdir(parents=True)
    (session_dir / "last.settings.json").write_text("{}", encoding="utf-8")
    concrete = session_dir / "fallback-session.settings.json"
    concrete.write_text("{}", encoding="utf-8")

    transport = DroidTransport(
        cwd=tmp_path,
        binary_resolver=lambda _: "/usr/bin/droid",
        home=home,
    )
    parsed = transport.parse_output(
        json.dumps(payload),
        allowed_roles={"reviewer", "implementer"},
        session_id=None,
        is_resume=False,
    )

    assert parsed.session_id == "fallback-session"
    assert parsed.usage["output_tokens"] == 12


def test_droid_resume_uses_session_flag(tmp_path: Path) -> None:
    fake_runner = FakeRunner(_fixture_text("droid_resume.json"))
    transport = DroidTransport(
        cwd=tmp_path,
        runner=fake_runner,
        binary_resolver=lambda _: "/usr/bin/droid",
    )

    transport.resume(
        "16329fce-3484-47a4-ad98-6676fdfb7477",
        "Continue",
        allowed_roles={"reviewer", "implementer"},
    )

    assert fake_runner.commands[0][:9] == [
        "droid",
        "exec",
        "--auto",
        "high",
        "--output-format",
        "json",
        "-s",
        "16329fce-3484-47a4-ad98-6676fdfb7477",
        "--cwd",
    ]


def test_factory_harness_alias_uses_droid_transport(tmp_path: Path) -> None:
    transport = create_transport(
        "factory",
        cwd=tmp_path,
        binary_resolver=lambda _: "/usr/bin/droid",
    )

    assert isinstance(transport, DroidTransport)
    assert transport.harness == "droid"
