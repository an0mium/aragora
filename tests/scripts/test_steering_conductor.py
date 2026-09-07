"""Tests for ``scripts/steering_conductor.py``."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "steering_conductor.py"
    spec = importlib.util.spec_from_file_location("steering_conductor_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


NOW = dt.datetime(2026, 7, 7, 17, 0, 0, tzinfo=dt.UTC)


def _lane(
    lane_id: str,
    owner_session: str,
    *,
    pr_number: int | None = 9001,
    branch: str = "codex/live-branch",
    status: str = "active",
    updated_at: str = "2026-07-07T16:55:00Z",
    next_action: str = "re-ground and avoid duplicate evidence lanes",
    possible_unpushed_work: bool | None = None,
) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "owner_session": owner_session,
        "status": status,
        "pr_number": pr_number,
        "branch": branch,
        "updated_at": updated_at,
        "last_heartbeat_at": updated_at,
        "next_action": next_action,
        "possible_unpushed_work": possible_unpushed_work,
    }


def _write_lanes(path: Path, lanes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lanes), encoding="utf-8")


def _write_receipt(
    inbox: Path,
    *,
    message_filename: str,
    message_sha256: str,
    outcome: str,
) -> None:
    receipt_dir = inbox / "_read_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / f"{outcome}.json").write_text(
        json.dumps(
            {
                "message_filename": message_filename,
                "message_sha256": message_sha256,
                "outcome": outcome,
            }
        ),
        encoding="utf-8",
    )


def _runner(open_prs: list[dict[str, Any]] | None = None) -> Any:
    def fake_runner(
        command: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, timeout, check
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, "## main...origin/main\n", "")
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, "head-sha\norigin-main-sha\n", "")
        if command[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(open_prs or []), "")
        if command == ["python3", "scripts/list_active_agent_sessions.py", "--json"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({"open_prs": []}), "")
        if command == ["python3", "scripts/agent_bridge.py", "operator-snapshot", "--json"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({"lanes": []}), "")
        if command == ["python3", "scripts/fleet_sentinel.py", "--json"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({"ok": True}), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return fake_runner


def _config(tmp_path: Path, lanes_path: Path, *, dry_run: bool = False) -> Any:
    return sc.CycleConfig(
        repo_root=tmp_path / "repo",
        ledger_path=tmp_path / "ledger.json",
        lane_registry_path=lanes_path,
        steering_inbox_root=tmp_path / "operator-steering",
        dry_run=dry_run,
        skip_fetch=True,
    )


def test_cycle_sends_one_message_and_records_ledger(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a")])

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is True
    assert result["selected"]["target_key"] == "pr:9001"
    messages = list((tmp_path / "operator-steering" / "owner-a").glob("*.json"))
    assert len(messages) == 1
    payload = json.loads(messages[0].read_text(encoding="utf-8"))
    assert payload["from"] == "steering-conductor"
    assert "Target: PR #9001 at live observed head abc123" in payload["body"]
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["consecutive_no_send"] == 0
    assert ledger["entries"][-1]["target_key"] == "pr:9001"


def test_unread_pending_message_blocks_duplicate_send(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a")])
    inbox = tmp_path / "operator-steering" / "owner-a"
    inbox.mkdir(parents=True)
    message = sc.send_operator_steering.build_message(
        to_session="owner-a",
        body="already pending",
    )
    (inbox / "pending.json").write_text(json.dumps(message), encoding="utf-8")

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["no_send_reason"] == "no eligible live owner target"
    assert len(list(inbox.glob("*.json"))) == 1
    assert result["candidate_skips"][0]["reason"] == "unread pending steering"


def test_read_receipt_does_not_clear_pending_message(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a")])
    inbox = tmp_path / "operator-steering" / "owner-a"
    inbox.mkdir(parents=True)
    message = sc.send_operator_steering.build_message(
        to_session="owner-a",
        body="already pending",
    )
    message_path = inbox / "pending.json"
    message_path.write_text(json.dumps(message), encoding="utf-8")
    _write_receipt(
        inbox,
        message_filename=message_path.name,
        message_sha256=message["message_sha256"],
        outcome="read",
    )

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["no_send_reason"] == "no eligible live owner target"
    assert result["candidate_skips"][0]["reason"] == "unread pending steering"


def test_resolved_receipt_clears_pending_message(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a")])
    inbox = tmp_path / "operator-steering" / "owner-a"
    inbox.mkdir(parents=True)
    message = sc.send_operator_steering.build_message(
        to_session="owner-a",
        body="already handled",
    )
    message_path = inbox / "pending.json"
    message_path.write_text(json.dumps(message), encoding="utf-8")
    _write_receipt(
        inbox,
        message_filename=message_path.name,
        message_sha256=message["message_sha256"],
        outcome="completed",
    )

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is True
    assert result["selected"]["target_key"] == "pr:9001"


def test_invalid_owner_session_skips_record_without_aborting(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(
        lanes_path,
        [
            _lane("bad", "owner/bad", pr_number=9001),
            _lane("good", "owner-good", pr_number=9002),
        ],
    )

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner(
            [
                {"number": 9001, "headRefOid": "abc123"},
                {"number": 9002, "headRefOid": "def456"},
            ]
        ),
        now=NOW,
    )

    assert result["sent"] is True
    assert result["selected"]["target_key"] == "pr:9002"
    reasons = {skip["target_key"]: skip["reason"] for skip in result["candidate_skips"]}
    assert reasons["pr:9001"].startswith("invalid owner_session:")


def test_duplicate_active_owners_block_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(
        lanes_path,
        [
            _lane("lane-a", "owner-a", pr_number=9001),
            _lane("lane-b", "owner-b", pr_number=9001),
        ],
    )

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["no_send_reason"] == "no eligible live owner target"
    assert len(list((tmp_path / "operator-steering").glob("*/*.json"))) == 0
    duplicate_skips = [
        skip for skip in result["candidate_skips"] if skip["reason"] == "multiple active owners"
    ]
    assert len(duplicate_skips) == 2
    assert duplicate_skips[0]["owner_sessions"] == ["owner-a", "owner-b"]


def test_duplicate_active_owners_do_not_block_clean_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(
        lanes_path,
        [
            _lane("lane-a", "owner-a", pr_number=9001),
            _lane("lane-b", "owner-b", pr_number=9001),
            _lane("lane-c", "owner-c", pr_number=9002),
        ],
    )

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner(
            [
                {"number": 9001, "headRefOid": "abc123"},
                {"number": 9002, "headRefOid": "def456"},
            ]
        ),
        now=NOW,
    )

    assert result["sent"] is True
    assert result["selected"]["target_key"] == "pr:9002"
    assert len(list((tmp_path / "operator-steering" / "owner-c").glob("*.json"))) == 1
    assert any(skip["reason"] == "multiple active owners" for skip in result["candidate_skips"])


def test_closed_pr_lane_is_excluded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a", pr_number=9001)])

    result = sc.run_cycle(_config(tmp_path, lanes_path), command_runner=_runner([]), now=NOW)

    assert result["sent"] is False
    assert result["no_send_reason"] == "no eligible live owner target"
    assert result["candidate_skips"][0]["reason"] == "PR is not open"


def test_recent_target_rotation_prefers_different_lane(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(
        lanes_path,
        [
            _lane("lane-a", "owner-a", pr_number=9001),
            _lane("lane-b", "owner-b", pr_number=9002),
        ],
    )
    ledger = {
        "schema_version": sc.SCHEMA_VERSION,
        "consecutive_no_send": 0,
        "entries": [
            {
                "timestamp": "2026-07-07T16:30:00Z",
                "sent": True,
                "target_key": "pr:9001",
                "owner_session": "owner-a",
            }
        ],
    }
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner(
            [
                {"number": 9001, "headRefOid": "abc123"},
                {"number": 9002, "headRefOid": "def456"},
            ]
        ),
        now=NOW,
    )

    assert result["sent"] is True
    assert result["selected"]["target_key"] == "pr:9002"
    assert len(list((tmp_path / "operator-steering" / "owner-a").glob("*.json"))) == 0
    assert len(list((tmp_path / "operator-steering" / "owner-b").glob("*.json"))) == 1


def test_recent_target_cycles_zero_uses_no_cycle_suppression() -> None:
    ledger = {
        "entries": [
            {
                "timestamp": "2026-07-07T16:30:00Z",
                "target_key": "pr:9001",
            }
        ]
    }

    assert (
        sc._recent_target_keys(
            ledger,
            now=NOW,
            recent_cycles=0,
            recent_hours=0.0,
        )
        == set()
    )


def test_load_lane_records_merges_default_user_and_repo_registries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    user_lanes = tmp_path / "user" / "lanes.json"
    repo_lanes = tmp_path / "repo" / "lanes.json"
    _write_lanes(user_lanes, [_lane("user-lane", "owner-user", pr_number=9001)])
    _write_lanes(repo_lanes, [_lane("repo-lane", "owner-repo", pr_number=9002)])
    monkeypatch.setattr(sc, "USER_LANE_REGISTRY_DEFAULT", user_lanes)
    monkeypatch.setattr(sc, "LANE_REGISTRY_DEFAULT", repo_lanes)

    records = sc.load_lane_records(repo_lanes)

    assert [record["lane_id"] for record in records] == ["user-lane", "repo-lane"]


def test_default_paths_use_canonical_owner_lookup_state_root() -> None:
    assert sc.LANE_REGISTRY_DEFAULT == sc.owner_lookup.LANE_REGISTRY_DEFAULT
    assert sc.STEERING_INBOX_ROOT_DEFAULT == sc.owner_lookup.STEERING_INBOX_ROOT_DEFAULT
    assert sc.CycleConfig().lane_registry_path == sc.owner_lookup.LANE_REGISTRY_DEFAULT
    assert sc.CycleConfig().steering_inbox_root == sc.owner_lookup.STEERING_INBOX_ROOT_DEFAULT


def test_stale_terminal_and_unpushed_lanes_are_excluded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(
        lanes_path,
        [
            _lane("old", "owner-old", pr_number=9001, updated_at="2026-07-07T12:00:00Z"),
            _lane("done", "owner-done", pr_number=9002, status="completed"),
            _lane("unpushed", "owner-unpushed", pr_number=9003, possible_unpushed_work=True),
        ],
    )

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    reasons = {skip["target_key"]: skip["reason"] for skip in result["candidate_skips"]}
    assert reasons["pr:9001"].startswith("stale lane age")
    assert reasons["pr:9002"] == "terminal status completed"
    assert reasons["pr:9003"] == "possible_unpushed_work"


def test_third_no_send_cycle_sets_stop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [])
    ledger = {
        "schema_version": sc.SCHEMA_VERSION,
        "consecutive_no_send": 2,
        "entries": [],
    }
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    result = sc.run_cycle(_config(tmp_path, lanes_path), command_runner=_runner(), now=NOW)

    assert result["sent"] is False
    assert result["stop"] is True
    assert result["stop_reason"] == "three consecutive no-send cycles"
    updated = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert updated["consecutive_no_send"] == 3


def test_third_drift_no_send_cycle_sets_stop(tmp_path: Path, monkeypatch: Any) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    ledger = {
        "schema_version": sc.SCHEMA_VERSION,
        "consecutive_no_send": 2,
        "entries": [],
    }
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    calls = 0

    def fake_load_lane_records(_path: Path) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_lane("lane-a", "owner-a", pr_number=9001)]
        return []

    monkeypatch.setattr(sc, "load_lane_records", fake_load_lane_records)

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["stop"] is True
    assert result["stop_reason"] == "three consecutive no-send cycles"
    assert result["ledger_consecutive_no_send"] == 3


def test_heartbeat_only_drift_still_sends(tmp_path: Path, monkeypatch: Any) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    calls = 0

    def fake_load_lane_records(_path: Path) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_lane("lane-a", "owner-a", pr_number=9001)]
        return [
            _lane(
                "lane-a",
                "owner-a",
                pr_number=9001,
                updated_at="2026-07-07T16:56:00Z",
            )
        ]

    monkeypatch.setattr(sc, "load_lane_records", fake_load_lane_records)

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is True
    assert result["selected"]["target_key"] == "pr:9001"
    assert len(list((tmp_path / "operator-steering" / "owner-a").glob("*.json"))) == 1


def test_target_drift_blocks_send(tmp_path: Path, monkeypatch: Any) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    calls = 0

    def fake_load_lane_records(_path: Path) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_lane("lane-a", "owner-a", pr_number=9001)]
        return [_lane("lane-a", "owner-a", pr_number=9002)]

    monkeypatch.setattr(sc, "load_lane_records", fake_load_lane_records)

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner(
            [
                {"number": 9001, "headRefOid": "abc123"},
                {"number": 9002, "headRefOid": "def456"},
            ]
        ),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["no_send_reason"] == "candidate target changed to pr:9002"
    assert result["ledger_consecutive_no_send"] == 1


def test_pr_list_failure_short_circuits_without_counting_no_send(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a", pr_number=9001)])

    def failing_pr_runner(
        command: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(command, 1, "", "auth failed")
        return _runner([])(
            command,
            cwd=cwd,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
        )

    result = sc.run_cycle(_config(tmp_path, lanes_path), command_runner=failing_pr_runner, now=NOW)

    assert result["ok"] is False
    assert result["sent"] is False
    assert result["no_send_reason"] == "open PR list unavailable"
    assert result["open_pr_error"] == "auth failed"
    assert result["ledger_consecutive_no_send"] == 0
    assert result["ledger_updated"] is False
    assert not (tmp_path / "ledger.json").exists()


def test_duplicate_sent_body_is_not_resent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    lane = _lane("lane-a", "owner-a", pr_number=9001)
    _write_lanes(lanes_path, [lane])
    body_hash = sc._body_hash(
        sc._message_body(lane, repo_root=tmp_path / "repo", open_pr={"headRefOid": "abc123"})
    )
    ledger = {
        "schema_version": sc.SCHEMA_VERSION,
        "consecutive_no_send": 0,
        "entries": [
            {
                "timestamp": "2026-07-07T16:30:00Z",
                "sent": True,
                "target_key": "pr:9001",
                "body_sha256": body_hash,
            }
        ],
    }
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["no_send_reason"] == "duplicate steering body already sent"
    assert len(list((tmp_path / "operator-steering" / "owner-a").glob("*.json"))) == 0
    updated = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert updated["consecutive_no_send"] == 1
    assert updated["entries"][-1]["body_sha256"] == body_hash


def test_duplicate_active_owner_appearing_before_send_blocks_target(
    tmp_path: Path, monkeypatch: Any
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    calls = 0

    def fake_load_lane_records(_path: Path) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_lane("lane-a", "owner-a", pr_number=9001)]
        return [
            _lane("lane-a", "owner-a", pr_number=9001),
            _lane("lane-b", "owner-b", pr_number=9001),
        ]

    monkeypatch.setattr(sc, "load_lane_records", fake_load_lane_records)

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["no_send_reason"] == "multiple active owners"
    assert result["owner_sessions"] == ["owner-a", "owner-b"]
    assert len(list((tmp_path / "operator-steering").glob("*/*.json"))) == 0


def test_non_dry_run_acquires_and_releases_ledger_lock(tmp_path: Path, monkeypatch: Any) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a")])
    lock_events: list[str] = []

    def fake_acquire(_path: Path) -> object:
        lock_events.append("acquire")
        return object()

    def fake_release(_handle: object) -> None:
        lock_events.append("release")

    monkeypatch.setattr(sc, "_acquire_cycle_lock", fake_acquire)
    monkeypatch.setattr(sc, "_release_cycle_lock", fake_release)

    result = sc.run_cycle(
        _config(tmp_path, lanes_path),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is True
    assert lock_events == ["acquire", "release"]


def test_dry_run_does_not_write_message_or_ledger(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lanes_path = tmp_path / "lanes.json"
    _write_lanes(lanes_path, [_lane("lane-a", "owner-a")])

    result = sc.run_cycle(
        _config(tmp_path, lanes_path, dry_run=True),
        command_runner=_runner([{"number": 9001, "headRefOid": "abc123"}]),
        now=NOW,
    )

    assert result["sent"] is False
    assert result["selected"]["target_key"] == "pr:9001"
    assert not (tmp_path / "operator-steering").exists()
    assert not (tmp_path / "ledger.json").exists()
