"""Tests for the local Codex dispatch queue runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


queue = _load_module("codex_dispatch_queue.py")


def test_enqueue_writes_pending_job(tmp_path: Path) -> None:
    args = queue.build_parser().parse_args(
        [
            "enqueue",
            "--to",
            "codex-owner",
            "--prompt",
            "queued prompt",
            "--queue-root",
            str(tmp_path / "queue"),
            "--json",
        ]
    )

    result = queue.enqueue(args)

    assert result["ok"] is True
    path = Path(result["path"])
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == queue.JOB_SCHEMA_VERSION
    assert payload["selectors"]["to"] == "codex-owner"
    assert payload["prompt_sha256"]


def test_dry_run_runner_blocks_without_writing_mailbox(tmp_path: Path) -> None:
    enqueue_args = queue.build_parser().parse_args(
        [
            "enqueue",
            "--to",
            "codex-owner",
            "--prompt",
            "queued prompt",
            "--queue-root",
            str(tmp_path / "queue"),
        ]
    )
    queue.enqueue(enqueue_args)
    run_args = queue.build_parser().parse_args(
        [
            "run",
            "--queue-root",
            str(tmp_path / "queue"),
            "--steering-inbox-root",
            str(tmp_path / "steering"),
            "--receipt-root",
            str(tmp_path / "dispatch-receipts"),
            "--json",
        ]
    )

    result = queue.run_queue(run_args)

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["processed"][0]["status"] == "blocked"
    assert result["processed"][0]["error"] == "dry-run; no prompt delivered"
    assert not (tmp_path / "steering").exists()
    assert list((tmp_path / "queue" / "blocked").glob("*.json"))
    assert list((tmp_path / "queue" / "blocked").glob("*.receipt.json"))
