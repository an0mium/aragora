from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aragora.cli.parser import build_parser
from aragora.cli.commands.work_board import (
    cmd_work_graph,
    cmd_work_list,
    cmd_work_robot,
    cmd_work_show,
)


def _args(tmp_path: Path, **kwargs) -> argparse.Namespace:
    defaults = {"repo": str(tmp_path), "json": True, "scope": "current", "work_id": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _capture_json(capsys: pytest.CaptureFixture) -> dict:
    return json.loads(capsys.readouterr().out)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def test_work_list_degrades_gracefully_without_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)

    assert cmd_work_list(_args(tmp_path, scope="current")) == 0
    payload = _capture_json(capsys)

    assert payload["schema_version"] == "aragora.work.v1"
    assert payload["items"] == []
    assert any(
        h["source"] == "github_pr" and h["status"] == "degraded" for h in payload["source_health"]
    )


def test_work_parser_registers_read_only_robot_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["work", "robot", "--json"])

    assert args.command == "work"
    assert args.work_cmd == "robot"
    assert args.json is True


def test_work_list_reads_current_outbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "handoff.json").write_text(
        json.dumps({"task": "Open PR for repair lane", "branch": "codex/repair"}),
        encoding="utf-8",
    )

    assert cmd_work_list(_args(tmp_path, scope="current")) == 0
    payload = _capture_json(capsys)

    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "automation-outbox:handoff"
    assert payload["items"][0]["branch"] == "codex/repair"


def test_work_robot_marks_tier_four_pr_human_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: "/usr/bin/gh")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["gh"],
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "number": 42,
                        "title": "Modify merge authority parser",
                        "url": "https://github.com/synaptent/aragora/pull/42",
                        "state": "OPEN",
                        "isDraft": False,
                        "headRefName": "codex/gate",
                        "headRefOid": "abc123",
                        "updatedAt": _now_iso(),
                        "createdAt": _now_iso(),
                        "reviewDecision": "APPROVED",
                        "mergeStateStatus": "CLEAN",
                        "labels": [{"name": "tier-4"}],
                        "assignees": [{"login": "codex"}],
                    }
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("aragora.work.sources.subprocess.run", fake_run)

    assert cmd_work_robot(_args(tmp_path)) == 0
    payload = _capture_json(capsys)

    assert payload["recommendations"][0]["item_id"] == "pr:42"
    assert payload["recommendations"][0]["classification"] == "human-gated"
    assert payload["recommendations"][0]["item"]["metadata"]["tier"] == 4


def test_work_show_finds_historical_receipt_in_all_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)
    receipts = tmp_path / ".aragora" / "automation-receipts"
    receipts.mkdir(parents=True)
    (receipts / "done.json").write_text(
        json.dumps(
            {
                "task": "Published old handoff",
                "status": "already_satisfied",
                "recorded_at": "2026-05-14T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert cmd_work_show(_args(tmp_path, work_id="automation-receipt:done")) == 0
    payload = _capture_json(capsys)

    assert payload["found"] is True
    assert payload["item"]["scope"] == "historical"


def test_work_graph_includes_bead_dependency_edges(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)
    bead_dir = tmp_path / ".aragora_beads"
    bead_dir.mkdir()
    (bead_dir / "beads.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "bead_type": "task",
                        "status": "pending",
                        "title": "A",
                        "updated_at": _now_iso(),
                        "dependencies": ["b"],
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "bead_type": "task",
                        "status": "pending",
                        "title": "B",
                        "updated_at": _now_iso(),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert cmd_work_graph(_args(tmp_path, work_id="bead:a")) == 0
    payload = _capture_json(capsys)

    assert {item["id"] for item in payload["items"]} == {"bead:a", "bead:b"}
    assert payload["edges"] == [{"from": "bead:a", "relation": "depends_on", "to": "bead:b"}]


def test_work_robot_ranks_actionable_current_work(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "repair.json").write_text(
        json.dumps({"task": "repair queue health", "branch": "codex/repair"}),
        encoding="utf-8",
    )

    assert cmd_work_robot(_args(tmp_path)) == 0
    payload = _capture_json(capsys)

    assert payload["mutations"] == []
    assert payload["recommendations"][0]["item_id"] == "automation-outbox:repair"
    assert payload["recommendations"][0]["classification"] == "needs-polish"
    assert payload["recommendations"][0]["action"] == "publish_or_reconcile_handoff"


def test_work_robot_emits_ready_for_polished_bead(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)
    bead_dir = tmp_path / ".aragora_beads"
    bead_dir.mkdir()
    (bead_dir / "beads.jsonl").write_text(
        json.dumps(
            {
                "id": "polished",
                "status": "pending",
                "title": "Repair broker live capture",
                "updated_at": _now_iso(),
                "claimed_by": "factory",
                "metadata": {
                    "objective": "Capture broker-launched sessions in operator-snapshot",
                    "context": "Desktop transcripts are historical, not live truth",
                    "acceptance_criteria": ["snapshot shows one active broker run"],
                    "validation": "focused bridge tests",
                    "mutation_boundary": "read-only bridge discovery",
                    "dependencies_declared": True,
                },
            }
        ),
        encoding="utf-8",
    )

    assert cmd_work_robot(_args(tmp_path)) == 0
    payload = _capture_json(capsys)

    assert payload["recommendations"][0]["item_id"] == "bead:polished"
    assert payload["recommendations"][0]["classification"] == "ready"
    assert payload["recommendations"][0]["score"]["bead_quality"] >= 0.8


def test_work_list_current_excludes_completed_broker_runs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("aragora.work.sources.shutil.which", lambda name: None)
    run_dir = tmp_path / ".aragora" / "agent_bridge" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "completed",
                "task": "Old desktop transcript import",
                "created_at": "2026-05-14T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert cmd_work_list(_args(tmp_path, scope="current")) == 0
    current = _capture_json(capsys)
    assert current["items"] == []

    assert cmd_work_list(_args(tmp_path, scope="all")) == 0
    all_items = _capture_json(capsys)
    assert all_items["items"][0]["id"] == "broker-run:run-1"
    assert all_items["items"][0]["scope"] == "historical"
