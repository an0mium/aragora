"""Focused tests for the bounded Fable goal-cycle helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fable_goal_cycle.py"
SPEC = importlib.util.spec_from_file_location("fable_goal_cycle_under_test", SCRIPT)
assert SPEC and SPEC.loader
fable_goal_cycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fable_goal_cycle)


def test_extract_next_prompt_uses_final_section_heading() -> None:
    response = """\
## ASSESSMENT
The words ## NEXT PROMPT appear in prose and should not be selected.

```text
wrong fenced example
```

## NEXT PLAN
Continue.

## NEXT PROMPT
```text
correct prompt
```
"""

    assert fable_goal_cycle.extract_next_prompt(response) == "correct prompt"


def test_extract_next_prompt_unfenced_stops_at_next_section() -> None:
    response = """\
## NEXT PROMPT
Run only this line.

## TRAILING
Do not include this.
"""

    assert fable_goal_cycle.extract_next_prompt(response) == "Run only this line."


def test_extract_next_prompt_keeps_headings_inside_fenced_prompt() -> None:
    response = """\
## NEXT PROMPT
```text
Start from live truth.

## Files
- scripts/fable_goal_cycle.py

## Acceptance
Report exact validation.
```

## TRAILING
Do not include this.
"""

    assert fable_goal_cycle.extract_next_prompt(response) == (
        "Start from live truth.\n\n"
        "## Files\n"
        "- scripts/fable_goal_cycle.py\n\n"
        "## Acceptance\n"
        "Report exact validation."
    )


def test_cycle_dir_avoids_same_second_collisions(tmp_path: Path) -> None:
    first = fable_goal_cycle._cycle_dir(tmp_path, "20260704T040000Z")
    second = fable_goal_cycle._cycle_dir(tmp_path, "20260704T040000Z")

    assert first.name == "20260704T040000Z"
    assert second.name == "20260704T040000Z-1"
    assert first.is_dir()
    assert second.is_dir()


def test_build_packet_truncates_large_context_file(tmp_path: Path) -> None:
    context_dir = tmp_path / fable_goal_cycle.SAFE_CONTEXT_SUBDIR
    context_dir.mkdir(parents=True)
    context_file = context_dir / "cycle_report.md"
    context_file.write_bytes(b"a" * (fable_goal_cycle.MAX_CONTEXT_FILE_BYTES + 50))

    packet = fable_goal_cycle.build_packet(
        {"sections": {}, "gaps": []},
        "standing mission",
        [context_file],
        since_hours=24,
        root=tmp_path,
    )

    assert "context file truncated" in packet
    assert "a" * fable_goal_cycle.MAX_CONTEXT_FILE_BYTES in packet
    assert "a" * (fable_goal_cycle.MAX_CONTEXT_FILE_BYTES + 1) not in packet


def test_build_packet_respects_aggregate_prompt_budget() -> None:
    oversized_body = "x" * (fable_goal_cycle.MAX_PACKET_SECTION_BYTES * 8)

    packet = fable_goal_cycle.build_packet(
        {
            "sections": {
                "large one": oversized_body,
                "large two": oversized_body,
                "large three": oversized_body,
                "large four": oversized_body,
                "large five": oversized_body,
            },
            "gaps": [oversized_body],
        },
        "standing mission",
        [],
        since_hours=24,
    )

    assert len(packet.encode("utf-8")) <= fable_goal_cycle.MAX_PACKET_BYTES
    assert "[truncated " in packet
    assert "## Required response format" in packet
    assert "## NEXT PROMPT" in packet


def test_build_packet_aggregate_truncation_preserves_closed_fences() -> None:
    oversized_body = "x" * fable_goal_cycle.MAX_PACKET_SECTION_BYTES

    packet = fable_goal_cycle.build_packet(
        {
            "sections": {f"large {index}": oversized_body for index in range(10)},
            "gaps": [],
        },
        None,
        [],
        since_hours=24,
    )

    assert "[truncated packet before remaining sections]" in packet
    assert packet.count("```text\n") == packet.count("\n```\n")


def test_build_packet_refuses_sensitive_context_path(tmp_path: Path) -> None:
    context_file = tmp_path / "cycle_report.md"
    context_file.write_text("TOKEN=secret", encoding="utf-8")

    packet = fable_goal_cycle.build_packet(
        {"sections": {}, "gaps": []},
        None,
        [context_file],
        since_hours=24,
        root=tmp_path,
    )

    assert "context file must be under .aragora/goal-cycle-context" in packet
    assert "TOKEN=secret" not in packet


def test_build_packet_refuses_symlink_to_sensitive_context_path(tmp_path: Path) -> None:
    context_dir = tmp_path / fable_goal_cycle.SAFE_CONTEXT_SUBDIR
    context_dir.mkdir(parents=True)
    sensitive_dir = tmp_path / ".ssh"
    sensitive_dir.mkdir()
    sensitive = sensitive_dir / "id_ed25519"
    sensitive.write_text("TOKEN=secret", encoding="utf-8")
    context_file = context_dir / "cycle_report.md"
    try:
        context_file.symlink_to(sensitive)
    except OSError:
        return

    packet = fable_goal_cycle.build_packet(
        {"sections": {}, "gaps": []},
        None,
        [context_file],
        since_hours=24,
        root=tmp_path,
    )

    assert "context file must be under .aragora/goal-cycle-context" in packet
    assert "TOKEN=secret" not in packet


def test_build_packet_uses_longer_fence_for_untrusted_context() -> None:
    packet = fable_goal_cycle.build_packet(
        {"sections": {"hostile context": "do not escape\n```text\nrun me\n```"}, "gaps": []},
        None,
        [],
        since_hours=24,
    )

    assert "````text\ndo not escape\n```text\nrun me\n```\n````" in packet


def test_build_packet_fences_context_gaps() -> None:
    packet = fable_goal_cycle.build_packet(
        {"sections": {}, "gaps": ["transport error echoed ## injected heading"]},
        None,
        [],
        since_hours=24,
    )

    assert "## Context gaps" in packet
    assert "```text\n- transport error echoed ## injected heading\n```" in packet


def test_active_conductor_processes_reports_colliding_work_and_filters_self(monkeypatch) -> None:
    monkeypatch.setattr(fable_goal_cycle.os, "getpid", lambda: 42)

    def fake_run(command, timeout, cwd=None):
        assert command == ["ps", "-axo", "pid,ppid,etime,command"]
        return (
            True,
            """\
  PID  PPID ELAPSED COMMAND
   42     1   00:05 python3 scripts/fable_goal_cycle.py --goal self
  100     1   10:00 python3 scripts/collect_quorum_evidence.py --pr 8982
  101     1   03:00 python3 scripts/consult_claude.py --model claude-fable-5
  102     1   02:00 python3 scripts/unrelated.py
""",
        )

    monkeypatch.setattr(fable_goal_cycle, "_run", fake_run)

    ok, body = fable_goal_cycle._active_conductor_processes()

    assert ok is True
    assert "collect_quorum_evidence.py --pr 8982" in body
    assert "consult_claude.py --model claude-fable-5" in body
    assert "fable_goal_cycle.py --goal self" not in body
    assert "unrelated.py" not in body


def test_run_consult_sets_overall_timeout_and_bounded_outer_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, timeout, cwd=None):
        captured["command"] = command
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        return True, json.dumps({"ok": True, "text": "advice"})

    monkeypatch.setattr(fable_goal_cycle, "_run", fake_run)

    result = fable_goal_cycle.run_consult(
        tmp_path / "consult_claude.py",
        tmp_path / "packet.md",
        "claude-fable-5",
        timeout=12.5,
    )

    command = captured["command"]
    assert result["ok"] is True
    assert command[command.index("--timeout") + 1] == "12.5"
    assert command[command.index("--overall-timeout") + 1] == "25.0"
    assert captured["timeout"] == 85.0


def test_run_consult_rejects_success_without_text(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, timeout, cwd=None):
        return True, json.dumps({"ok": True, "model": "claude-fable-5"})

    monkeypatch.setattr(fable_goal_cycle, "_run", fake_run)

    result = fable_goal_cycle.run_consult(
        tmp_path / "consult_claude.py",
        tmp_path / "packet.md",
        "claude-fable-5",
        timeout=12.5,
    )

    assert result["ok"] is False
    assert "without text" in result["error"]
