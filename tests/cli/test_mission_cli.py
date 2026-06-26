"""Tests for the native mission CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.cli.commands.mission import cmd_mission
from aragora.cli.parser import build_parser
from aragora.missions import Feature, MissionState, Status


def test_mission_parser_accepts_public_subcommands(tmp_path: Path) -> None:
    parser = build_parser()
    state_path = tmp_path / "mission.json"

    seed = parser.parse_args(
        [
            "mission",
            "seed",
            "Refactor auth",
            "--state",
            str(state_path),
            "--budget",
            "150.50",
            "--max-hours",
            "6.5",
            "--relay",
            "slack",
            "--auto-settle-max-tier",
            "1",
            "--tracks",
            "sme,qa",
        ]
    )
    assert seed.command == "mission"
    assert seed.mission_action == "seed"
    assert seed.goal == ["Refactor auth"]
    assert seed.state == str(state_path)
    assert seed.autonomy == "report"

    run = parser.parse_args(
        [
            "mission",
            "run",
            "--state",
            str(state_path),
            "--autonomy",
            "auto-drain",
            "--max-ticks",
            "2",
        ]
    )
    assert run.mission_action == "run"
    assert run.state == str(state_path)
    assert run.autonomy == "auto-drain"
    assert run.max_ticks == 2

    reconcile = parser.parse_args(["mission", "reconcile", "--autonomy", "safe-clean"])
    assert reconcile.mission_action == "reconcile"
    assert reconcile.autonomy == "safe-clean"


def test_mission_parser_keeps_legacy_goal_alias() -> None:
    parser = build_parser()

    args = parser.parse_args(["mission", "Do something"])

    assert args.command == "mission"
    assert args.mission_action == "Do something"
    assert args.goal == []


def test_mission_parser_rejects_invalid_relay() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["mission", "seed", "goal", "--relay", "invalid-relay"])


def test_cmd_mission_seed_writes_native_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "1")
    parser = build_parser()
    state_path = tmp_path / "state.json"
    args = parser.parse_args(
        [
            "mission",
            "seed",
            "Refactor auth",
            "--state",
            str(state_path),
            "--budget",
            "50",
            "--relay",
            "email",
            "--tracks",
            "sme",
        ]
    )

    exit_code = cmd_mission(args)

    assert exit_code == 0
    loaded = MissionState.load(state_path)
    assert loaded.goal == "Refactor auth"
    assert loaded.features[0].metadata["budget_usd"] == 50.0
    assert loaded.features[0].metadata["relay"] == "email"
    assert loaded.features[0].metadata["tracks"] == ["sme"]
    captured = capsys.readouterr()
    assert "Seeded mission" in captured.out
    assert str(state_path) in captured.out


def test_cmd_mission_legacy_alias_seeds_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "1")
    parser = build_parser()
    state_path = tmp_path / "state.json"
    args = parser.parse_args(["mission", "Refactor auth", "--state", str(state_path)])

    assert cmd_mission(args) == 0
    assert MissionState.load(state_path).goal == "Refactor auth"
    assert "Seeded mission" in capsys.readouterr().out


def test_cmd_mission_status_prints_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    MissionState(
        mission_id="mission-test",
        goal="g",
        milestones=["m"],
        features=[
            Feature(id="done", description="done", milestone="m", status=Status.COMPLETED),
            Feature(id="blocked", description="blocked", milestone="m", status=Status.BLOCKED),
        ],
    ).save(state_path)
    parser = build_parser()
    args = parser.parse_args(["mission", "status", "--state", str(state_path)])

    assert cmd_mission(args) == 0

    out = capsys.readouterr().out
    assert "mission-test" in out
    assert "1/2 completed" in out
    assert "1 blocked" in out


def test_cmd_mission_seed_refuses_when_native_mission_flag_is_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARAGORA_ENABLE_NATIVE_MISSION", raising=False)
    parser = build_parser()
    state_path = tmp_path / "state.json"
    args = parser.parse_args(["mission", "seed", "Refactor auth", "--state", str(state_path)])

    assert cmd_mission(args) == 1

    assert not state_path.exists()
    assert "Native mission engine is disabled" in capsys.readouterr().err


def test_cmd_mission_run_report_mode_parks_without_false_completion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "1")
    state_path = tmp_path / "state.json"
    MissionState(
        mission_id="mission-test",
        goal="g",
        milestones=["m"],
        features=[Feature(id="f1", description="inspect", milestone="m")],
    ).save(state_path)
    parser = build_parser()
    args = parser.parse_args(
        ["mission", "run", "--state", str(state_path), "--autonomy", "report", "--max-ticks", "2"]
    )

    assert cmd_mission(args) == 0

    feature = MissionState.load(state_path).get("f1")
    assert feature.status == Status.BLOCKED
    assert "report autonomy does not dispatch feature f1" in feature.notes
    assert "Mission run: 0/1 completed" in capsys.readouterr().out


def test_cmd_mission_run_refuses_paused_mission(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "1")
    state_path = tmp_path / "state.json"
    state_path.with_name("PAUSED").write_text("operator pause\n", encoding="utf-8")
    MissionState(
        mission_id="mission-test",
        goal="g",
        milestones=["m"],
        features=[Feature(id="f1", description="inspect", milestone="m")],
    ).save(state_path)
    parser = build_parser()
    args = parser.parse_args(["mission", "run", "--state", str(state_path)])

    assert cmd_mission(args) == 1

    assert MissionState.load(state_path).get("f1").status == Status.PENDING
    assert "mission is paused" in capsys.readouterr().err


def test_cmd_mission_reconcile_outputs_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARAGORA_ENABLE_NATIVE_MISSION", "1")
    fixture = tmp_path / "artifacts.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "artifact_id": "wt-merged",
                    "kind": "worktree",
                    "clean": True,
                    "already_merged": True,
                },
                {"artifact_id": "wt-dirty", "kind": "worktree", "clean": False},
            ]
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "mission",
            "reconcile",
            "--autonomy",
            "safe-clean",
            "--artifact-fixture",
            str(fixture),
            "--json",
        ]
    )

    assert cmd_mission(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "safe-clean"
    assert [item["artifact_id"] for item in payload["authorized_cleanup"]] == ["wt-merged"]
    assert [item["artifact_id"] for item in payload["parked"]] == ["wt-dirty"]
