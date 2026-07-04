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


def test_cycle_dir_avoids_same_second_collisions(tmp_path: Path) -> None:
    first = fable_goal_cycle._cycle_dir(tmp_path, "20260704T040000Z")
    second = fable_goal_cycle._cycle_dir(tmp_path, "20260704T040000Z")

    assert first.name == "20260704T040000Z"
    assert second.name == "20260704T040000Z-1"
    assert first.is_dir()
    assert second.is_dir()


def test_build_packet_truncates_large_context_file(tmp_path: Path) -> None:
    context_file = tmp_path / "cycle_report.md"
    context_file.write_bytes(b"a" * (fable_goal_cycle.MAX_CONTEXT_FILE_BYTES + 50))

    packet = fable_goal_cycle.build_packet(
        {"sections": {}, "gaps": []},
        "standing mission",
        [context_file],
        since_hours=24,
    )

    assert "context file truncated" in packet
    assert "a" * fable_goal_cycle.MAX_CONTEXT_FILE_BYTES in packet
    assert "a" * (fable_goal_cycle.MAX_CONTEXT_FILE_BYTES + 1) not in packet


def test_build_packet_refuses_sensitive_context_path(tmp_path: Path) -> None:
    context_file = tmp_path / ".env"
    context_file.write_text("TOKEN=secret", encoding="utf-8")

    packet = fable_goal_cycle.build_packet(
        {"sections": {}, "gaps": []},
        None,
        [context_file],
        since_hours=24,
    )

    assert "refused potentially sensitive context path" in packet
    assert "TOKEN=secret" not in packet


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
    assert command[command.index("--overall-timeout") + 1] == "12.5"
    assert captured["timeout"] == 72.5
