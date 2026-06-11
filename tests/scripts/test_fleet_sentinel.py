"""Tests for ``scripts/fleet_sentinel.py`` — Dead Man's Signals fleet sentinel.

Plan v2 Phase 0.1 (Pillar 6). All checks are exercised through injected
fixture paths and injected command runners — no network, no live state.
"""

from __future__ import annotations

import importlib.util
import json
import os
import plistlib
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


sentinel = _load_module("fleet_sentinel.py")

NOW = sentinel.parse_iso("2026-06-10T12:00:00Z")
HOUR = 3600.0


def _touch(path: Path, *, age_hours: float = 0.0, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    ts = NOW.timestamp() - age_hours * HOUR
    os.utime(path, (ts, ts))
    return path


def _healthy_publisher_payload() -> dict[str, Any]:
    return {
        "generated_at": "2026-06-10T11:00:00Z",
        "github_health": {"auth_ok": True, "api_ok": True, "ready": True},
    }


# ---------------------------------------------------------------------------
# publisher_status
# ---------------------------------------------------------------------------


def test_publisher_status_ok(tmp_path: Path) -> None:
    p = _touch(tmp_path / "status.json", content=json.dumps(_healthy_publisher_payload()))
    result = sentinel.check_publisher_status(p, max_age_hours=24, now=NOW)
    assert result["check"] == "publisher_status"
    assert result["status"] == "ok"


def test_publisher_status_stale_mtime_breaches(tmp_path: Path) -> None:
    p = _touch(
        tmp_path / "status.json",
        age_hours=25,
        content=json.dumps(_healthy_publisher_payload()),
    )
    result = sentinel.check_publisher_status(p, max_age_hours=24, now=NOW)
    assert result["status"] == "breach"
    assert "stale" in result["detail"].lower() or "old" in result["detail"].lower()


def test_publisher_status_auth_not_ok_breaches(tmp_path: Path) -> None:
    payload = _healthy_publisher_payload()
    payload["github_health"]["auth_ok"] = False
    p = _touch(tmp_path / "status.json", content=json.dumps(payload))
    result = sentinel.check_publisher_status(p, max_age_hours=24, now=NOW)
    assert result["status"] == "breach"
    assert "auth_ok" in result["detail"]


def test_publisher_status_missing_file_breaches(tmp_path: Path) -> None:
    result = sentinel.check_publisher_status(tmp_path / "absent.json", max_age_hours=24, now=NOW)
    assert result["status"] == "breach"


def test_publisher_status_corrupt_json_breaches(tmp_path: Path) -> None:
    p = _touch(tmp_path / "status.json", content="{not json")
    result = sentinel.check_publisher_status(p, max_age_hours=24, now=NOW)
    assert result["status"] == "breach"


def test_breach_replay_may18_publisher_incident(tmp_path: Path) -> None:
    """Acceptance: replaying the real May-18 publisher status raises the alarm.

    Fixture replicates the status structure captured during the
    May-18 -> Jun-08 outage: ``auth_ok: false`` and ``generated_at``
    of 2026-05-18.  Even with a *fresh* file mtime the sentinel must breach.

    Provenance: the original incident snapshot lived at the old
    ``.aragora/automation-publisher-status.json`` path, whose writer moved
    away around 2026-05-24 (the file became an orphan).  The check now
    watches the live writer's path —
    ``.aragora/automation-github-status/latest.json``, written by
    ``scripts/cache_codex_automation_github_status.py`` on every publisher
    pass — which carries the same ``generated_at`` +
    ``github_health.auth_ok`` structure, so the breach semantics replayed
    here are unchanged.
    """
    incident = {
        "generated_at": "2026-05-18T15:10:46.775938Z",
        "github_health": {
            "api_ok": False,
            "auth_ok": False,
            "error": (
                "error connecting to api.github.com\n"
                "check your internet connection or https://githubstatus.com"
            ),
            "mode": "connectivity_failed",
            "ready": False,
        },
        "github_queue": {"available": False, "reason": "connectivity_failed"},
        "github_repo": "synaptent/aragora",
    }
    p = _touch(tmp_path / "latest.json", content=json.dumps(incident))
    result = sentinel.check_publisher_status(p, max_age_hours=24, now=NOW)
    assert result["status"] == "breach"
    assert "auth_ok" in result["detail"]


def test_publisher_status_default_is_live_cache_path() -> None:
    """The default must point at the live writer's file, not the orphan.

    The publisher's status writer moved to
    ``.aragora/automation-github-status/latest.json``
    (scripts/cache_codex_automation_github_status.py) around 2026-05-24;
    the old ``automation-publisher-status.json`` stopped being written.
    A sentinel watching the orphan would breach forever on stale data —
    or worse, report "ok" on a frozen healthy snapshot.
    """
    args = sentinel.build_parser().parse_args([])
    default = Path(args.publisher_status)
    assert default.parts[-3:] == (".aragora", "automation-github-status", "latest.json")


# ---------------------------------------------------------------------------
# boss_metrics_heartbeat
# ---------------------------------------------------------------------------


def test_boss_metrics_fresh_ok(tmp_path: Path) -> None:
    p = _touch(tmp_path / "boss_metrics.jsonl", age_hours=1)
    result = sentinel.check_boss_metrics(p, max_age_hours=48, now=NOW)
    assert result["status"] == "ok"


def test_boss_metrics_stale_breaches(tmp_path: Path) -> None:
    p = _touch(tmp_path / "boss_metrics.jsonl", age_hours=49)
    result = sentinel.check_boss_metrics(p, max_age_hours=48, now=NOW)
    assert result["status"] == "breach"


def test_boss_metrics_missing_breaches(tmp_path: Path) -> None:
    result = sentinel.check_boss_metrics(tmp_path / "absent.jsonl", max_age_hours=48, now=NOW)
    assert result["status"] == "breach"


# ---------------------------------------------------------------------------
# launchd_plists  (motivating incident: the empty boss-loop plist of 2026-06-10)
# ---------------------------------------------------------------------------


def _write_plist(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        plistlib.dump(payload, fh)
    return path


def test_launchd_plists_all_valid_ok(tmp_path: Path) -> None:
    _write_plist(tmp_path / "com.aragora.good.plist", {"Label": "com.aragora.good"})
    result = sentinel.check_launchd_plists(tmp_path)
    assert result["status"] == "ok"


def test_launchd_empty_plist_breaches(tmp_path: Path) -> None:
    _write_plist(tmp_path / "com.aragora.good.plist", {"Label": "com.aragora.good"})
    (tmp_path / "com.aragora.boss-loop.plist").write_bytes(b"")
    result = sentinel.check_launchd_plists(tmp_path)
    assert result["status"] == "breach"
    assert "com.aragora.boss-loop.plist" in result["detail"]


def test_launchd_unparseable_plist_breaches(tmp_path: Path) -> None:
    (tmp_path / "com.aragora.broken.plist").write_text("definitely not a plist")
    result = sentinel.check_launchd_plists(tmp_path)
    assert result["status"] == "breach"
    assert "com.aragora.broken.plist" in result["detail"]


def test_launchd_ignores_non_aragora_plists(tmp_path: Path) -> None:
    (tmp_path / "com.other.vendor.plist").write_bytes(b"")
    result = sentinel.check_launchd_plists(tmp_path)
    assert result["status"] == "ok"


def test_launchd_missing_dir_is_ok(tmp_path: Path) -> None:
    result = sentinel.check_launchd_plists(tmp_path / "absent")
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# gh_auth
# ---------------------------------------------------------------------------


def test_gh_auth_ok_with_zero_exit() -> None:
    result = sentinel.check_gh_auth(runner=lambda cmd: 0)
    assert result["status"] == "ok"


def test_gh_auth_nonzero_breaches() -> None:
    result = sentinel.check_gh_auth(runner=lambda cmd: 1)
    assert result["status"] == "breach"


def test_gh_auth_runner_exception_is_unknown() -> None:
    def boom(cmd: list[str]) -> int:
        raise FileNotFoundError("gh not installed")

    result = sentinel.check_gh_auth(runner=boom)
    assert result["status"] == "unknown"


# ---------------------------------------------------------------------------
# checkout_invariant
# ---------------------------------------------------------------------------


def test_checkout_on_main_ok(tmp_path: Path) -> None:
    result = sentinel.check_checkout_invariant(tmp_path, branch_reader=lambda repo: "main")
    assert result["status"] == "ok"


def test_checkout_off_main_breaches(tmp_path: Path) -> None:
    result = sentinel.check_checkout_invariant(tmp_path, branch_reader=lambda repo: "feature/x")
    assert result["status"] == "breach"
    assert "feature/x" in result["detail"]


def test_checkout_reader_failure_is_unknown(tmp_path: Path) -> None:
    def boom(repo: Path) -> str:
        raise subprocess.CalledProcessError(128, ["git"])

    result = sentinel.check_checkout_invariant(tmp_path, branch_reader=boom)
    assert result["status"] == "unknown"


# ---------------------------------------------------------------------------
# outbox_depth
# ---------------------------------------------------------------------------


def test_outbox_within_limits_ok(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    for i in range(3):
        _touch(outbox / f"item-{i}.json", age_hours=1)
    result = sentinel.check_outbox(outbox, max_items=50, max_age_days=7, now=NOW)
    assert result["status"] == "ok"


def test_outbox_depth_above_max_breaches(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    for i in range(4):
        _touch(outbox / f"item-{i}.json", age_hours=1)
    result = sentinel.check_outbox(outbox, max_items=3, max_age_days=7, now=NOW)
    assert result["status"] == "breach"
    assert "4" in result["detail"]


def test_outbox_archive_subdir_excluded(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    _touch(outbox / "live.json", age_hours=1)
    for i in range(10):
        _touch(outbox / "archive" / f"old-{i}.json", age_hours=1)
    result = sentinel.check_outbox(outbox, max_items=3, max_age_days=7, now=NOW)
    assert result["status"] == "ok"


def test_outbox_oldest_item_too_old_breaches(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    _touch(outbox / "ancient.json", age_hours=8 * 24)
    result = sentinel.check_outbox(outbox, max_items=50, max_age_days=7, now=NOW)
    assert result["status"] == "breach"
    assert "ancient.json" in result["detail"]


def test_outbox_missing_dir_is_ok(tmp_path: Path) -> None:
    result = sentinel.check_outbox(tmp_path / "absent", max_items=50, max_age_days=7, now=NOW)
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# disk_free
# ---------------------------------------------------------------------------


class _Usage:
    def __init__(self, free_gib: float) -> None:
        self.total = 500 * 2**30
        self.used = self.total - int(free_gib * 2**30)
        self.free = int(free_gib * 2**30)


def test_disk_free_above_min_ok(tmp_path: Path) -> None:
    result = sentinel.check_disk_free(tmp_path, min_free_gib=25, usage_fn=lambda p: _Usage(100))
    assert result["status"] == "ok"


def test_disk_free_below_min_breaches(tmp_path: Path) -> None:
    result = sentinel.check_disk_free(tmp_path, min_free_gib=25, usage_fn=lambda p: _Usage(10))
    assert result["status"] == "breach"


# ---------------------------------------------------------------------------
# report / exit contract / ledger / notify
# ---------------------------------------------------------------------------


def _fixture_env(tmp_path: Path) -> dict[str, Path]:
    """A fully healthy filesystem fixture for end-to-end main() runs."""
    paths = {
        "publisher": _touch(
            tmp_path / "status.json", content=json.dumps(_healthy_publisher_payload())
        ),
        "metrics": _touch(tmp_path / "boss_metrics.jsonl", age_hours=1),
        "launch_agents": tmp_path / "LaunchAgents",
        "outbox": tmp_path / "outbox",
        "ledger": tmp_path / "ledger" / "ledger.jsonl",
    }
    _write_plist(paths["launch_agents"] / "com.aragora.ok.plist", {"Label": "ok"})
    _touch(paths["outbox"] / "one.json", age_hours=1)
    return paths


def _argv(paths: dict[str, Path], **overrides: Any) -> list[str]:
    argv = [
        "--json",
        "--now",
        "2026-06-10T12:00:00Z",
        "--publisher-status",
        str(paths["publisher"]),
        "--boss-metrics",
        str(paths["metrics"]),
        "--launch-agents-dir",
        str(paths["launch_agents"]),
        "--outbox-dir",
        str(paths["outbox"]),
        "--ledger",
        str(paths["ledger"]),
        "--checks",
        overrides.pop(
            "checks",
            "publisher_status,boss_metrics_heartbeat,launchd_plists,outbox_depth,disk_free",
        ),
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return argv


def test_main_all_ok_exits_zero_and_emits_contract(tmp_path: Path, capsys: Any) -> None:
    paths = _fixture_env(tmp_path)
    code = sentinel.main(_argv(paths))
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generated_at"] == "2026-06-10T12:00:00Z"
    assert report["breaches"] == 0
    assert report["blind_checks"] == 0
    assert {c["check"] for c in report["checks"]} == {
        "publisher_status",
        "boss_metrics_heartbeat",
        "launchd_plists",
        "outbox_depth",
        "disk_free",
    }
    for check in report["checks"]:
        assert set(check) >= {"check", "status", "detail"}


def test_main_breach_exits_one(tmp_path: Path, capsys: Any) -> None:
    paths = _fixture_env(tmp_path)
    paths["publisher"].write_text(
        json.dumps({"generated_at": "2026-05-18T15:10:46Z", "github_health": {"auth_ok": False}})
    )
    code = sentinel.main(_argv(paths))
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["breaches"] >= 1


def test_main_unknown_exits_two_never_zero(tmp_path: Path, capsys: Any) -> None:
    """Silence/unknown is never success: an errored check forces exit 2."""
    paths = _fixture_env(tmp_path)
    code = sentinel.main(
        _argv(paths, checks="publisher_status,gh_auth", gh_auth_cmd="/nonexistent-gh-binary")
    )
    assert code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blind_checks"] >= 1


def test_unknown_takes_precedence_over_breach() -> None:
    checks = [
        {"check": "a", "status": "breach", "detail": "x"},
        {"check": "b", "status": "unknown", "detail": "y"},
    ]
    assert sentinel.exit_code_for(checks) == 2
    assert sentinel.exit_code_for(checks[:1]) == 1
    assert sentinel.exit_code_for([{"check": "c", "status": "ok", "detail": ""}]) == 0


def test_main_appends_ledger_line(tmp_path: Path, capsys: Any) -> None:
    paths = _fixture_env(tmp_path)
    assert sentinel.main(_argv(paths)) == 0
    assert sentinel.main(_argv(paths)) == 0
    capsys.readouterr()
    lines = paths["ledger"].read_text().strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["breaches"] == 0
    assert entry["generated_at"] == "2026-06-10T12:00:00Z"


def test_notify_cmd_invoked_with_breach_summary(tmp_path: Path, capsys: Any) -> None:
    paths = _fixture_env(tmp_path)
    paths["metrics"].unlink()  # boss_metrics breach
    record = tmp_path / "notify-args.json"
    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        "import json, sys, pathlib\n"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    code = sentinel.main(_argv(paths, notify_cmd=f"{sys.executable} {recorder} {{summary}}"))
    assert code == 1
    capsys.readouterr()
    args = json.loads(record.read_text())
    assert len(args) == 1
    assert "boss_metrics_heartbeat" in args[0]


def test_notify_cmd_not_invoked_when_all_ok(tmp_path: Path, capsys: Any) -> None:
    paths = _fixture_env(tmp_path)
    record = tmp_path / "notify-args.json"
    recorder = tmp_path / "recorder.py"
    recorder.write_text(f"import pathlib\npathlib.Path({str(record)!r}).write_text('called')\n")
    code = sentinel.main(_argv(paths, notify_cmd=f"{sys.executable} {recorder} {{summary}}"))
    assert code == 0
    capsys.readouterr()
    assert not record.exists()


def test_notify_bare_placeholder_token_passes_summary_raw() -> None:
    """A standalone ``{summary}`` token is argv-level: pass the text through."""
    captured: dict[str, list[str]] = {}

    def runner(tokens: list[str]) -> int:
        captured["tokens"] = tokens
        return 0

    sentinel.notify("notifier --msg {summary}", 'has "quotes" and \\slash', runner=runner)
    assert captured["tokens"] == ["notifier", "--msg", 'has "quotes" and \\slash']


# ---------------------------------------------------------------------------
# lane_liveness  (failure class A: the silent c06/c07/c08 lane deaths of
# 2026-06-10/11 — three coordinator-spawned lanes died at setup, left empty
# branches on origin, and nobody noticed until a manual morning sweep)
# ---------------------------------------------------------------------------


def _write_lane_ledger(
    lanes_dir: Path,
    lane: str,
    *,
    branch: str,
    launched_at: str,
    status: str = "in_progress",
) -> Path:
    lanes_dir.mkdir(parents=True, exist_ok=True)
    path = lanes_dir / f"{lane}.json"
    path.write_text(
        json.dumps(
            {
                "lane": lane,
                "agent_id": "a0000000000000000",
                "branch": branch,
                "brief": f"brief for {lane}",
                "launched_at": launched_at,
                "status": status,
            }
        )
    )
    return path


def _liveness(
    lanes_dir: Path,
    *,
    heads: dict[str, str],
    ahead: dict[str, int | None],
    dates: dict[str, str] | None = None,
    lane_max_age_hours: float = 3.0,
    orphan_age_hours: float = 24.0,
) -> Any:
    date_map = {sha: sentinel.parse_iso(d) for sha, d in (dates or {}).items()}
    return sentinel.check_lane_liveness(
        str(lanes_dir),
        lane_max_age_hours=lane_max_age_hours,
        orphan_age_hours=orphan_age_hours,
        now=NOW,
        remote_heads=lambda: heads,
        ahead_counter=lambda sha: ahead.get(sha),
        commit_dater=lambda sha: date_map.get(sha),
    )


def test_lane_liveness_fresh_in_progress_ok(tmp_path: Path) -> None:
    _write_lane_ledger(tmp_path, "q2", branch="elves/run-x-q2", launched_at="2026-06-10T11:00:00Z")
    result = _liveness(tmp_path, heads={"elves/run-x-q2": "aaa"}, ahead={"aaa": 0})
    assert result["status"] == "ok"


def test_lane_liveness_stale_zero_ahead_breaches(tmp_path: Path) -> None:
    _write_lane_ledger(tmp_path, "q2", branch="elves/run-x-q2", launched_at="2026-06-10T07:00:00Z")
    result = _liveness(tmp_path, heads={"elves/run-x-q2": "aaa"}, ahead={"aaa": 0})
    assert result["status"] == "breach"
    assert "q2" in result["detail"]
    assert "zero commits ahead" in result["detail"]


def test_lane_liveness_stale_branch_absent_breaches(tmp_path: Path) -> None:
    _write_lane_ledger(tmp_path, "q2", branch="elves/run-x-q2", launched_at="2026-06-10T07:00:00Z")
    result = _liveness(tmp_path, heads={}, ahead={})
    assert result["status"] == "breach"
    assert "absent from origin" in result["detail"]


def test_lane_liveness_stale_with_commits_ok(tmp_path: Path) -> None:
    _write_lane_ledger(tmp_path, "q2", branch="elves/run-x-q2", launched_at="2026-06-10T07:00:00Z")
    result = _liveness(tmp_path, heads={"elves/run-x-q2": "aaa"}, ahead={"aaa": 2})
    assert result["status"] == "ok"


def test_lane_liveness_unresolvable_ahead_is_not_dead(tmp_path: Path) -> None:
    """None from the ahead counter means unfetched unique commits — never dead."""
    _write_lane_ledger(tmp_path, "q2", branch="elves/run-x-q2", launched_at="2026-06-10T07:00:00Z")
    result = _liveness(tmp_path, heads={"elves/run-x-q2": "aaa"}, ahead={"aaa": None})
    assert result["status"] == "ok"


def test_lane_liveness_non_in_progress_ignored(tmp_path: Path) -> None:
    _write_lane_ledger(
        tmp_path,
        "q2",
        branch="elves/run-x-q2",
        launched_at="2026-06-10T07:00:00Z",
        status="dead",
    )
    result = _liveness(tmp_path, heads={}, ahead={})
    assert result["status"] == "ok"


def test_lane_liveness_orphan_branch_breaches(tmp_path: Path) -> None:
    result = _liveness(
        tmp_path,
        heads={"elves/run-20260610-c06-dead": "aaa"},
        ahead={"aaa": 0},
        dates={"aaa": "2026-06-09T06:00:00Z"},  # 30h old at NOW
    )
    assert result["status"] == "breach"
    assert "orphan branch elves/run-20260610-c06-dead" in result["detail"]


def test_lane_liveness_boss_pattern_orphan_breaches(tmp_path: Path) -> None:
    result = _liveness(
        tmp_path,
        heads={"aragora/boss-harvest/issue-1-boss-x": "bbb"},
        ahead={"bbb": 0},
        dates={"bbb": "2026-06-08T12:00:00Z"},
    )
    assert result["status"] == "breach"


def test_lane_liveness_young_empty_branch_ok(tmp_path: Path) -> None:
    result = _liveness(
        tmp_path,
        heads={"elves/run-x-young": "aaa"},
        ahead={"aaa": 0},
        dates={"aaa": "2026-06-10T10:00:00Z"},  # 2h old
    )
    assert result["status"] == "ok"


def test_lane_liveness_old_branch_with_commits_ok(tmp_path: Path) -> None:
    result = _liveness(
        tmp_path,
        heads={"elves/run-x-real-work": "aaa"},
        ahead={"aaa": 1},
        dates={"aaa": "2026-06-01T00:00:00Z"},
    )
    assert result["status"] == "ok"


def test_lane_liveness_non_pattern_branch_ignored(tmp_path: Path) -> None:
    result = _liveness(
        tmp_path,
        heads={"feature/unrelated": "aaa"},
        ahead={"aaa": 0},
        dates={"aaa": "2026-06-01T00:00:00Z"},
    )
    assert result["status"] == "ok"


def test_lane_liveness_remote_failure_is_unknown(tmp_path: Path) -> None:
    def boom() -> dict[str, str]:
        raise subprocess.CalledProcessError(128, ["git"])

    result = sentinel.check_lane_liveness(
        str(tmp_path),
        lane_max_age_hours=3,
        orphan_age_hours=24,
        now=NOW,
        remote_heads=boom,
        ahead_counter=lambda sha: 0,
        commit_dater=lambda sha: None,
    )
    assert result["status"] == "unknown"


def test_lane_liveness_unreadable_ledger_is_unknown(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "broken.json").write_text("{not json")
    result = _liveness(tmp_path, heads={}, ahead={})
    assert result["status"] == "unknown"
    assert "broken.json" in result["detail"]


def test_breach_replay_jun10_silent_lane_death(tmp_path: Path) -> None:
    """Acceptance: replaying the real 2026-06-10/11 incident raises the alarm.

    Three coordinator-spawned lanes (c06/c07/c08) died at setup overnight.
    Their branches sat on origin with zero commits ahead of main until a
    manual morning sweep found them.  With ledgers present the ledger rule
    fires; without ledgers (as on the real morning after) the orphan-branch
    rule fires for every one of them.
    """
    heads = {
        "elves/run-20260610-c06-mixed-family-fix": "c06sha",
        "elves/run-20260610-c07-auto-evidence": "c07sha",
        "elves/run-20260610-c08-b1-retrigger": "c08sha",
    }
    ahead: dict[str, int | None] = {"c06sha": 0, "c07sha": 0, "c08sha": 0}
    dates = dict.fromkeys(ahead, "2026-06-09T02:00:00Z")  # ~34h old at NOW
    # Ledger-less variant (the real morning-after state):
    result = _liveness(tmp_path / "lanes", heads=heads, ahead=ahead, dates=dates)
    assert result["status"] == "breach"
    for branch in heads:
        assert branch in result["detail"]
    # Ledger-present variant: the ledger rule also catches each lane.
    lanes = tmp_path / "run-20260610" / "lanes"
    for lane, branch in (
        ("c06", "elves/run-20260610-c06-mixed-family-fix"),
        ("c07", "elves/run-20260610-c07-auto-evidence"),
        ("c08", "elves/run-20260610-c08-b1-retrigger"),
    ):
        _write_lane_ledger(lanes, lane, branch=branch, launched_at="2026-06-09T02:00:00Z")
    result = _liveness(lanes, heads=heads, ahead=ahead, dates=dates)
    assert result["status"] == "breach"
    assert result["detail"].count("lane c0") == 3


# ---------------------------------------------------------------------------
# github_api_health  (failure class B: GraphQL 502/504 streaks of 2026-06-10/11
# stalled the publisher; the cached github_health flipped auth_ok:false so the
# sentinel breached without distinguishing transient from persistent)
# ---------------------------------------------------------------------------


def _publisher_log(tmp_path: Path, passes: list[str], *, errors: list[str] | None = None) -> Path:
    """Build a publisher log: one start/end marker pair per pass outcome."""
    lines: list[str] = []
    for i, outcome in enumerate(passes):
        lines.append(
            f"2026-06-11T0{i % 10}:00:00Z [codex-automation-publisher] starting branch publish pass"
        )
        for err in errors or []:
            lines.append(f'    "error": "{err}",')
        lines.append(
            f"2026-06-11T0{i % 10}:01:00Z [codex-automation-publisher] branch publish pass {outcome}"
        )
    path = tmp_path / "codex-automation-publisher.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_github_api_health_probe_ok_no_streak(tmp_path: Path) -> None:
    log = _publisher_log(tmp_path, ["complete", "complete"])
    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 0
    )
    assert result["status"] == "ok"
    assert "streak=0" in result["detail"]


def test_github_api_health_transient_blip_visible_but_quiet(tmp_path: Path) -> None:
    """Probe down + short streak = transient: recorded in detail, NO breach."""
    log = _publisher_log(
        tmp_path,
        ["complete", "failed", "failed"],
        errors=["HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)"],
    )
    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 1
    )
    assert result["status"] == "ok"
    assert "streak=2" in result["detail"]
    assert "HTTP 502" in result["detail"]


def test_github_api_health_persistent_degradation_breaches(tmp_path: Path) -> None:
    log = _publisher_log(
        tmp_path,
        ["complete", "failed", "failed", "failed"],
        errors=["HTTP 504: We couldn't respond in time (https://api.github.com/graphql)"],
    )
    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 1
    )
    assert result["status"] == "breach"
    assert "streak=3" in result["detail"]
    assert "HTTP 504" in result["detail"]


def test_github_api_health_probe_recovered_long_streak_ok(tmp_path: Path) -> None:
    """Streak alone never breaches: a green probe means the API recovered."""
    log = _publisher_log(tmp_path, ["failed", "failed", "failed", "failed"])
    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 0
    )
    assert result["status"] == "ok"
    assert "streak=4" in result["detail"]


def test_github_api_health_streak_resets_on_complete(tmp_path: Path) -> None:
    log = _publisher_log(tmp_path, ["failed", "failed", "complete", "failed"])
    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 1
    )
    assert result["status"] == "ok"
    assert "streak=1" in result["detail"]


def test_github_api_health_attempt_lines_not_counted_as_pass_failures(tmp_path: Path) -> None:
    lines = [
        "2026-06-11T00:00:00Z [codex-automation-publisher] starting branch publish pass",
        "2026-06-11T00:00:30Z [codex-automation-publisher] branch publish pass attempt 1/3 failed (exit 1); retrying in 30s",
        "2026-06-11T00:01:30Z [codex-automation-publisher] branch publish pass complete",
    ]
    log = tmp_path / "pub.log"
    log.write_text("\n".join(lines) + "\n")
    streak, _ = sentinel.publisher_failure_streak(lines)
    assert streak == 0
    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 1
    )
    assert result["status"] == "ok"


def test_github_api_health_log_missing_is_unknown(tmp_path: Path) -> None:
    result = sentinel.check_github_api_health(
        tmp_path / "absent.log", persist_threshold=3, tail_lines=2000, probe_runner=lambda cmd: 0
    )
    assert result["status"] == "unknown"


def test_github_api_health_probe_crash_is_unknown(tmp_path: Path) -> None:
    log = _publisher_log(tmp_path, ["complete"])

    def boom(cmd: list[str]) -> int:
        raise FileNotFoundError("gh not installed")

    result = sentinel.check_github_api_health(
        log, persist_threshold=3, tail_lines=2000, probe_runner=boom
    )
    assert result["status"] == "unknown"


def test_new_checks_registered_in_all_checks() -> None:
    assert "lane_liveness" in sentinel.ALL_CHECKS
    assert "github_api_health" in sentinel.ALL_CHECKS


def test_main_github_api_health_wired_end_to_end(tmp_path: Path, capsys: Any) -> None:
    log = _publisher_log(tmp_path, ["complete"])
    code = sentinel.main(
        [
            "--json",
            "--no-ledger",
            "--now",
            "2026-06-10T12:00:00Z",
            "--checks",
            "github_api_health",
            "--publisher-log",
            str(log),
            "--rate-limit-cmd",
            f"{sys.executable} -c pass",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checks"][0]["check"] == "github_api_health"


def test_main_lane_liveness_blind_repo_exits_two(tmp_path: Path, capsys: Any) -> None:
    """Wiring test without network: a non-repo root makes the check blind."""
    code = sentinel.main(
        [
            "--json",
            "--no-ledger",
            "--now",
            "2026-06-10T12:00:00Z",
            "--checks",
            "lane_liveness",
            "--repo-root",
            str(tmp_path),
            "--lanes-glob",
            str(tmp_path / "run-*" / "lanes"),
        ]
    )
    assert code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["blind_checks"] == 1


def test_notify_embedded_placeholder_neutralizes_quote_injection() -> None:
    """{summary} embedded inside a larger token (e.g. an AppleScript string
    literal, as in the installer's default osascript notify command) must not
    let quotes/backslashes in the summary escape the host string literal."""
    captured: dict[str, list[str]] = {}

    def runner(tokens: list[str]) -> int:
        captured["tokens"] = tokens
        return 0

    cmd = 'osascript -e "display notification \\"{summary}\\" with title \\"Aragora Fleet Sentinel\\""'
    sentinel.notify(cmd, 'detail "x" \\ end', runner=runner)
    tokens = captured["tokens"]
    assert tokens[0] == "osascript"
    script = tokens[2]
    # The two delimiting quotes around the summary plus the two around the
    # title are the ONLY double quotes left — none smuggled in via summary.
    assert script.count('"') == 4
    assert "\\" not in script
    assert "detail 'x' / end" in script
