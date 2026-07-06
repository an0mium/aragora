"""Tests for ``scripts/write_prompt_handoff_outbox.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_script(script_name: str, module_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script("write_prompt_handoff_outbox.py", "write_prompt_handoff_outbox_under_test")


def test_dry_run_emits_publisher_compatible_payload(tmp_path: Path, capsys: Any) -> None:
    rc = mod.main(
        [
            "--prompt",
            "Start from live repo truth.\nDo exactly one bounded unit.",
            "--task",
            "Prompt handoff for queue routing",
            "--source",
            "python3 scripts/build_next_prompt.py --json",
            "--pr",
            "8845",
            "--expected-head",
            "abc123",
            "--outbox-dir",
            str(tmp_path),
            "--created-at",
            "2026-07-06T15:41:00Z",
            "--json",
        ]
    )

    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["wrote"] is False
    assert not Path(result["outbox_path"]).exists()

    payload = result["payload"]
    for required in (
        "task",
        "requires_github",
        "requested_action",
        "repo",
        "local_evidence",
        "validation",
        "idempotency_key",
        "created_at",
    ):
        assert payload[required]
    assert payload["requires_github"] is True
    assert payload["requested_action"]["type"] == "prompt_handoff"
    assert payload["requested_action"]["target"] == {"pr": 8845, "expected_head": "abc123"}
    assert payload["local_evidence"]["kind"] == "prompt_handoff"
    assert (
        payload["local_evidence"]["prompt_sha256"] == payload["requested_action"]["prompt_sha256"]
    )
    assert payload["local_evidence"]["prompt"].startswith("Start from live repo truth")
    assert payload["expires_at"] == "2026-07-09T15:41:00Z"


def test_apply_writes_deterministic_outbox_file(tmp_path: Path, capsys: Any) -> None:
    rc = mod.main(
        [
            "--prompt",
            "Route this through the existing publisher.",
            "--task",
            "Publish prompt handoff",
            "--outbox-dir",
            str(tmp_path),
            "--created-at",
            "2026-07-06T15:42:00Z",
            "--apply",
            "--json",
        ]
    )

    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    outbox_path = Path(result["outbox_path"])
    assert result["wrote"] is True
    assert outbox_path.exists()
    saved = json.loads(outbox_path.read_text(encoding="utf-8"))
    assert saved == result["payload"]
    assert saved["idempotency_key"].startswith("prompt-handoff-publish-prompt-handoff-")


def test_written_payload_is_loadable_by_automation_handoff_publisher(
    tmp_path: Path, capsys: Any
) -> None:
    rc = mod.main(
        [
            "--prompt",
            "Start from live repo truth and read the handoff issue.",
            "--task",
            "Publish prompt handoff through existing publisher",
            "--outbox-dir",
            str(tmp_path),
            "--created-at",
            "2026-07-06T15:44:00Z",
            "--apply",
            "--json",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    publisher = _load_script(
        "publish_automation_handoffs.py", "publish_automation_handoffs_for_prompt_test"
    )
    handoffs, skipped = publisher._load_outbox_handoffs_with_skip_reasons(
        tmp_path,
        outbox_dir=tmp_path,
        receipt_dir=tmp_path / "receipts",
        now=publisher.datetime(2026, 7, 6, 15, 45, tzinfo=publisher.UTC),
    )

    assert skipped == {}
    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish prompt handoff through existing publisher"
    assert "Requested Action:" in handoffs[0].body
    assert "prompt_handoff" in handoffs[0].body


def test_apply_refuses_existing_file_without_force(tmp_path: Path, capsys: Any) -> None:
    args = [
        "--prompt",
        "Same prompt.",
        "--task",
        "Same task",
        "--outbox-dir",
        str(tmp_path),
        "--created-at",
        "2026-07-06T15:43:00Z",
        "--apply",
    ]
    assert mod.main(args) == 0
    assert mod.main(args) == 2
    assert "handoff already exists" in capsys.readouterr().err


def test_default_idempotency_key_includes_target_metadata(tmp_path: Path, capsys: Any) -> None:
    base_args = [
        "--prompt",
        "Same prompt for different targets.",
        "--task",
        "Same task",
        "--outbox-dir",
        str(tmp_path),
        "--created-at",
        "2026-07-06T15:43:00Z",
        "--apply",
        "--json",
    ]

    assert mod.main([*base_args, "--pr", "100"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert mod.main([*base_args, "--pr", "101"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["payload"]["idempotency_key"] != second["payload"]["idempotency_key"]
    assert first["outbox_path"] != second["outbox_path"]
    assert Path(first["outbox_path"]).exists()
    assert Path(second["outbox_path"]).exists()


def test_empty_prompt_returns_2(capsys: Any) -> None:
    rc = mod.main(["--prompt", "  "])
    assert rc == 2
    assert "prompt must not be empty" in capsys.readouterr().err
