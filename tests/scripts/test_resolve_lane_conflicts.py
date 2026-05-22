"""Tests for ``scripts/resolve_lane_conflicts.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
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


resolver = _load_module("resolve_lane_conflicts.py")
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "resolve_lane_conflicts.py"


def test_detects_completed_owner_conflict_without_mutating(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P104-ssd-cleanup-continuation",
                    "owner_session": "codex-P104",
                    "status": "conflict",
                    "conflict_session": "codex-R03",
                    "conflict_reason": "stale cleanup overlap",
                },
                {
                    "lane_id": "R03-post-p102-harvest-followthrough",
                    "owner_session": "codex-R03",
                    "status": "completed",
                },
            ]
        ),
        encoding="utf-8",
    )

    candidates = resolver.find_resolvable_conflicts(registry)

    assert [candidate["lane_id"] for candidate in candidates] == ["P104-ssd-cleanup-continuation"]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "conflict"


def test_apply_marks_conflict_superseded_and_writes_receipt(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P104-ssd-cleanup-continuation",
                    "owner_session": "codex-P104",
                    "status": "conflict",
                    "conflict_session": "codex-R03",
                    "conflict_reason": "stale cleanup overlap",
                },
                {
                    "lane_id": "R03-post-p102-harvest-followthrough",
                    "owner_session": "codex-R03",
                    "status": "released",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = resolver.resolve_conflicts(
        registry_path=registry,
        receipt_dir=receipt_dir,
        apply=True,
        resolved_at="2026-05-21T23:30:00Z",
    )

    rows = {row["lane_id"]: row for row in json.loads(registry.read_text(encoding="utf-8"))}
    assert result["resolved_count"] == 1
    assert rows["P104-ssd-cleanup-continuation"]["status"] == "superseded"
    receipts = sorted(receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "aragora-lane-conflict-resolution/1.0"
    assert receipt["lane_id"] == "P104-ssd-cleanup-continuation"
    assert receipt["new_status"] == "superseded"


def test_apply_supersedes_only_exact_conflict_row(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "shared-lane",
                    "owner_session": "codex-conflict-a",
                    "status": "conflict",
                    "conflict_session": "codex-done",
                },
                {
                    "lane_id": "shared-lane",
                    "owner_session": "codex-conflict-b",
                    "status": "conflict",
                    "conflict_session": "codex-unknown",
                },
                {
                    "lane_id": "done-lane",
                    "owner_session": "codex-done",
                    "status": "completed",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = resolver.resolve_conflicts(
        registry_path=registry,
        receipt_dir=receipt_dir,
        apply=True,
        resolved_at="2026-05-21T23:45:00Z",
    )

    rows = json.loads(registry.read_text(encoding="utf-8"))
    by_owner = {row["owner_session"]: row for row in rows}
    assert result["resolved_count"] == 1
    assert result["unknown_session_count"] == 1
    assert by_owner["codex-conflict-a"]["status"] == "superseded"
    assert by_owner["codex-conflict-b"]["status"] == "conflict"


def test_concurrent_apply_preserves_registry_json(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": f"conflict-{idx:02d}",
                    "owner_session": f"codex-conflict-{idx:02d}",
                    "status": "conflict",
                    "conflict_session": f"codex-done-{idx:02d}",
                }
                for idx in range(8)
            ]
            + [
                {
                    "lane_id": f"done-{idx:02d}",
                    "owner_session": f"codex-done-{idx:02d}",
                    "status": "completed",
                }
                for idx in range(8)
            ]
        ),
        encoding="utf-8",
    )

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--apply",
                "--registry-path",
                str(registry),
                "--receipt-dir",
                str(receipt_dir),
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _idx in range(4)
    ]
    results = [proc.communicate(timeout=30) + (proc.returncode,) for proc in procs]

    assert all(returncode == 0 for _stdout, _stderr, returncode in results), results
    payload = json.loads(registry.read_text(encoding="utf-8"))
    by_lane = {row["lane_id"]: row for row in payload}
    assert all(by_lane[f"conflict-{idx:02d}"]["status"] == "superseded" for idx in range(8))
