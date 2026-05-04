"""Tests for scripts/check_codex_desktop_automations.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _write_automation(
    root: Path,
    automation_id: str,
    *,
    name: str,
    prompt: str,
    byminute: int,
    status: str = "ACTIVE",
) -> None:
    path = root / automation_id
    path.mkdir(parents=True)
    (path / "automation.toml").write_text(
        "\n".join(
            [
                "version = 1",
                f'id = "{automation_id}"',
                'kind = "cron"',
                f'name = "{name}"',
                f'prompt = "{prompt}"',
                f'status = "{status}"',
                f'rrule = "FREQ=HOURLY;INTERVAL=1;BYMINUTE={byminute}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_detects_paused_core_writer(tmp_path: Path) -> None:
    import check_codex_desktop_automations as mod

    for automation_id, minute in mod.CORE_WRITERS.items():
        _write_automation(
            tmp_path,
            automation_id,
            name=f"{automation_id} Writer",
            prompt="Paused during queue drain.",
            byminute=minute,
        )

    payload = mod.build_payload(tmp_path)

    codes = {issue["code"] for issue in payload["issues"]}
    assert "active_paused_prompt" in codes
    assert "missing_prompt_word_outbox" in codes


def test_audit_accepts_staggered_writer_contracts(tmp_path: Path) -> None:
    import check_codex_desktop_automations as mod

    prompt = "Read memory, repair one branch, validate locally, run preflight, then refresh outbox."
    for automation_id, minute in mod.CORE_WRITERS.items():
        _write_automation(
            tmp_path,
            automation_id,
            name=f"{automation_id} Writer",
            prompt=prompt,
            byminute=minute,
        )

    payload = mod.build_payload(tmp_path)

    assert payload["summary"] == {"active_count": 4, "error_count": 0, "warning_count": 0}


def test_audit_warns_writer_missing_preflight(tmp_path: Path) -> None:
    import check_codex_desktop_automations as mod

    prompt = "Read memory, repair one branch, validate locally, then refresh outbox."
    for automation_id, minute in mod.CORE_WRITERS.items():
        _write_automation(
            tmp_path,
            automation_id,
            name=f"{automation_id} Writer",
            prompt=prompt,
            byminute=minute,
        )

    payload = mod.build_payload(tmp_path)

    assert {
        issue["automation_id"]
        for issue in payload["issues"]
        if issue["code"] == "missing_prompt_word_preflight"
    } == set(mod.CORE_WRITERS)


def test_summary_only_payload_omits_core_writer_prompts(tmp_path: Path) -> None:
    import check_codex_desktop_automations as mod

    prompt = "Read memory, repair one branch, validate locally, run preflight, then refresh outbox."
    for automation_id, minute in mod.CORE_WRITERS.items():
        _write_automation(
            tmp_path,
            automation_id,
            name=f"{automation_id} Writer",
            prompt=prompt,
            byminute=minute,
        )

    payload = mod.build_payload(tmp_path)
    compact = mod.summary_only_payload(payload)

    assert compact["summary"] == payload["summary"]
    assert compact["prompt_details_omitted"] is True
    assert "prompt" in payload["core_writers"]["engineering-autopilot"]
    assert "prompt" not in compact["core_writers"]["engineering-autopilot"]
    assert compact["core_writers"]["engineering-autopilot"]["byminute"] == 5


def test_main_summary_only_json_omits_prompts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import check_codex_desktop_automations as mod

    prompt = "Read memory, repair one branch, validate locally, run preflight, then refresh outbox."
    for automation_id, minute in mod.CORE_WRITERS.items():
        _write_automation(
            tmp_path,
            automation_id,
            name=f"{automation_id} Writer",
            prompt=prompt,
            byminute=minute,
        )

    assert mod.main(["--root", str(tmp_path), "--json", "--summary-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt_details_omitted"] is True
    assert all("prompt" not in record for record in payload["core_writers"].values())


def test_audit_warns_duplicate_writer_minutes(tmp_path: Path) -> None:
    import check_codex_desktop_automations as mod

    prompt = "Read memory, repair one branch, validate locally, run preflight, then refresh outbox."
    for automation_id in mod.CORE_WRITERS:
        _write_automation(
            tmp_path,
            automation_id,
            name=f"{automation_id} Writer",
            prompt=prompt,
            byminute=5,
        )

    payload = mod.build_payload(tmp_path)

    assert any(issue["code"] == "duplicate_writer_minute" for issue in payload["issues"])
    assert any(issue["code"] == "writer_not_staggered" for issue in payload["issues"])
