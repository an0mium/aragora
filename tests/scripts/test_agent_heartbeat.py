"""Tests for ``scripts/agent_heartbeat.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


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


heartbeat = _load_module("agent_heartbeat.py")


def test_heartbeat_upserts_owner_identity(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "heartbeats.json"

    row = heartbeat.record_heartbeat(
        heartbeat_path=heartbeat_path,
        lane_id="P106-merge-gate-settlement",
        owner_session="droid-P106-merge-gate-settlement-20260521T2118Z",
        thread_id="thread-123",
        pid=12345,
        cwd="/tmp/aragora",
        worktree="/tmp/aragora",
        branch="claude/recover-merge-gate-reconciliation",
        pr_number=7423,
        last_seen_at="2026-05-21T23:00:00Z",
    )

    assert row["lane_id"] == "P106-merge-gate-settlement"
    assert row["owner_session"] == "droid-P106-merge-gate-settlement-20260521T2118Z"
    assert row["last_seen_at"] == "2026-05-21T23:00:00Z"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload == [row]


def test_heartbeat_rejects_path_traversal_owner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe owner_session"):
        heartbeat.record_heartbeat(
            heartbeat_path=tmp_path / "heartbeats.json",
            lane_id="P106",
            owner_session="../escape",
        )
