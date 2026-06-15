"""Tests for ``scripts/identify_lane_owner.py`` — Phase A consolidator.

Fixture-driven; never calls the real ``agent_bridge`` subprocess and
never reads the live ``~/.codex/`` / ``~/.claude/`` / ``~/.factory/``
directories. All discovery sources are pointed at ``tmp_path``
fixtures so tests are deterministic and isolated.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "identify_lane_owner.py"
    spec = importlib.util.spec_from_file_location("identify_lane_owner_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ilo = _load_module()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


SAMPLE_LANES: list[dict[str, Any]] = [
    {
        "lane_id": "P19-repair-7292-stage2-blockers",
        "owner_session": "codex-p19-repair-7292",
        "source": "codex",
        "status": "active",
        "branch": "droid/P16-stage2-auto-merge-bucket-a-20260518-002325",
        "worktree": "/private/tmp/p19-fixture-wt",
        "pr_number": 7292,
        "goal": "Repair #7292 Stage 2 auto-merge blockers",
        "updated_at": "2026-05-18T04:19:24Z",
    },
    {
        "lane_id": "P20-model-pins-frontier-aligned",
        "owner_session": "droid-F473CDBF",
        "source": "droid",
        "status": "active",
        "branch": "droid/P20-model-pins-frontier-aligned-20260518-041438",
        "worktree": "/private/tmp/p20-fixture-wt",
        "pr_number": None,
        "updated_at": "2026-05-18T04:14:38Z",
    },
    {
        "lane_id": "P28-with-rich-identity",
        "owner_session": "codex-test-rich",
        "source": "codex",
        "status": "active",
        "branch": "codex/with-identity",
        "worktree": "/private/tmp/p28-rich-wt",
        "pr_number": 9000,
        "codex_thread_id": "019e3942-e27e-7e72-b8d6-b61d981fd532",
        "codex_rollout_path": None,  # set per-test
        "desktop_label": "Test Codex Desktop Tab",
        "session_title": "Rich identity claim",
        "updated_at": "2026-05-18T04:30:00Z",
    },
]


def write_lane_registry(tmp_path: Path, lanes: list[dict[str, Any]] | None = None) -> Path:
    if lanes is None:
        lanes = SAMPLE_LANES
    registry_dir = tmp_path / ".aragora" / "agent-bridge"
    registry_dir.mkdir(parents=True, exist_ok=True)
    p = registry_dir / "lanes.json"
    p.write_text(json.dumps(lanes), encoding="utf-8")
    return p


def fake_snapshot_records(
    records: list[dict[str, Any]],
    *,
    by_role: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fake operator-snapshot payload matching the live contract."""

    return {"process_census": {"by_role": by_role or {}, "records": records}}


def test_default_state_root_prefers_local_lane_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worktree = tmp_path / "worktree"
    local_registry = worktree / ".aragora" / "agent-bridge" / "lanes.json"
    local_registry.parent.mkdir(parents=True)
    local_registry.write_text("[]", encoding="utf-8")

    def fail_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("git lookup should not run when local registry exists")

    monkeypatch.setattr(ilo.subprocess, "run", fail_run)

    assert ilo._default_state_root(worktree) == worktree / ".aragora"


def test_default_state_root_uses_git_common_dir_for_linked_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worktree = tmp_path / "linked" / "aragora"
    canonical = tmp_path / "main" / "aragora"
    worktree.mkdir(parents=True)
    canonical.mkdir(parents=True)

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, f"{canonical / '.git'}\n", "")

    monkeypatch.setattr(ilo.subprocess, "run", fake_run)

    assert ilo._default_state_root(worktree) == canonical / ".aragora"


def test_default_state_root_honors_automation_state_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_root = tmp_path / "state-root"
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))

    assert ilo._default_state_root(tmp_path / "worktree") == state_root / ".aragora"


# ---------------------------------------------------------------------------
# load_lane_records / find_lane
# ---------------------------------------------------------------------------


class TestLoadAndFind:
    def test_missing_registry_returns_empty_list(self, tmp_path: Path) -> None:
        assert ilo.load_lane_records(tmp_path / "nope.json") == []

    def test_unparseable_registry_returns_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "lanes.json"
        p.write_text("not valid json {{{", encoding="utf-8")
        assert ilo.load_lane_records(p) == []

    def test_find_by_exact_lane_id(self) -> None:
        r = ilo.find_lane(SAMPLE_LANES, lane_id="P19-repair-7292-stage2-blockers")
        assert r is not None
        assert r["owner_session"] == "codex-p19-repair-7292"

    def test_find_by_exact_lane_id_preserves_registry_order(self) -> None:
        lanes = [
            {
                "lane_id": "duplicate-lane-id",
                "owner_session": "codex-original",
                "status": "released",
                "updated_at": "2026-05-18T04:00:00Z",
            },
            {
                "lane_id": "duplicate-lane-id",
                "owner_session": "codex-newer-active",
                "status": "active",
                "updated_at": "2026-05-18T05:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, lane_id="duplicate-lane-id")
        assert r is not None
        assert r["owner_session"] == "codex-original"

    def test_find_by_pr_number(self) -> None:
        r = ilo.find_lane(SAMPLE_LANES, pr=7292)
        assert r is not None
        assert r["lane_id"] == "P19-repair-7292-stage2-blockers"

    def test_find_by_pr_prefers_active_over_stale_history(self) -> None:
        lanes = [
            {
                "lane_id": "old-completed",
                "owner_session": "codex-old",
                "status": "completed",
                "pr_number": 7292,
                "updated_at": "2026-05-18T05:00:00Z",
            },
            {
                "lane_id": "current-active",
                "owner_session": "codex-current",
                "status": "active",
                "pr_number": 7292,
                "updated_at": "2026-05-18T04:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, pr=7292)
        assert r is not None
        assert r["lane_id"] == "current-active"

    def test_find_by_pr_uses_newest_historical_when_unowned(self) -> None:
        lanes = [
            {
                "lane_id": "older-completed",
                "owner_session": "codex-old",
                "status": "completed",
                "pr_number": 7292,
                "updated_at": "2026-05-18T04:00:00Z",
            },
            {
                "lane_id": "newer-released",
                "owner_session": "codex-new",
                "status": "released",
                "pr_number": 7292,
                "updated_at": "2026-05-18T05:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, pr=7292)
        assert r is not None
        assert r["lane_id"] == "newer-released"


class TestHeartbeatSummary:
    def test_build_owner_info_includes_fresh_heartbeat(self, tmp_path: Path) -> None:
        heartbeat_path = tmp_path / "heartbeats.json"
        heartbeat_path.write_text(
            json.dumps(
                [
                    {
                        "schema_version": "aragora-agent-heartbeat/1.0",
                        "lane_id": "P19-repair-7292-stage2-blockers",
                        "owner_session": "codex-p19-repair-7292",
                        "pid": 1234,
                        "cwd": "/tmp/aragora",
                        "worktree": "/private/tmp/p19-fixture-wt",
                        "branch": "droid/P16-stage2-auto-merge-bucket-a-20260518-002325",
                        "pr_number": 7292,
                        "last_seen_at": "2026-05-22T00:05:00Z",
                    }
                ]
            ),
            encoding="utf-8",
        )

        info = ilo.build_owner_info(
            SAMPLE_LANES[0],
            snapshot_provider=lambda: None,
            sessions_root=tmp_path / "codex",
            projects_root=tmp_path / "claude",
            bg_path=tmp_path / "factory.json",
            steering_inbox_root=tmp_path / "steering",
            heartbeat_path=heartbeat_path,
            heartbeat_now="2026-05-22T00:10:00Z",
        )

        assert info.latest_heartbeat is not None
        assert info.latest_heartbeat["fresh"] is True
        assert info.latest_heartbeat["age_seconds"] == 300
        assert info.latest_heartbeat["pid"] == 1234
        assert info.latest_heartbeat["cwd"] == "/tmp/aragora"
        assert info.latest_heartbeat["worktree"] == "/private/tmp/p19-fixture-wt"
        assert (
            info.latest_heartbeat["branch"]
            == "droid/P16-stage2-auto-merge-bucket-a-20260518-002325"
        )
        assert info.latest_heartbeat["pr_number"] == 7292
        assert info.owner_state == "owned"
        assert info.liveness_state == "fresh_heartbeat"
        assert info.cleanup_state == "preserve_live_owner"
        assert info.recommended_operator_action == (
            "route work through owner_session; do not cleanup without owner release"
        )

    def test_build_owner_info_marks_stale_heartbeat(self, tmp_path: Path) -> None:
        heartbeat_path = tmp_path / "heartbeats.json"
        heartbeat_path.write_text(
            json.dumps(
                [
                    {
                        "lane_id": "P19-repair-7292-stage2-blockers",
                        "owner_session": "codex-p19-repair-7292",
                        "last_seen_at": "2026-05-22T00:00:00Z",
                    }
                ]
            ),
            encoding="utf-8",
        )

        info = ilo.build_owner_info(
            SAMPLE_LANES[0],
            snapshot_provider=lambda: None,
            sessions_root=tmp_path / "codex",
            projects_root=tmp_path / "claude",
            bg_path=tmp_path / "factory.json",
            steering_inbox_root=tmp_path / "steering",
            heartbeat_path=heartbeat_path,
            heartbeat_now="2026-05-22T00:20:00Z",
        )

        assert info.latest_heartbeat is not None
        assert info.latest_heartbeat["fresh"] is False
        assert info.latest_heartbeat["age_seconds"] == 1200
        assert info.owner_state == "owned"
        assert info.liveness_state == "stale_heartbeat"
        assert info.cleanup_state == "preserve_stale_owner"
        assert info.recommended_operator_action == (
            "preserve; refresh heartbeat or contact owner before mutation or cleanup"
        )

    def test_build_owner_info_prefers_claimed_owner_heartbeat(self, tmp_path: Path) -> None:
        heartbeat_path = tmp_path / "heartbeats.json"
        heartbeat_path.write_text(
            json.dumps(
                [
                    {
                        "lane_id": "P19-repair-7292-stage2-blockers",
                        "owner_session": "other-owner",
                        "branch": "droid/P16-stage2-auto-merge-bucket-a-20260518-002325",
                        "pr_number": 7292,
                        "last_seen_at": "2026-05-22T00:10:00Z",
                    },
                    {
                        "lane_id": "P19-repair-7292-stage2-blockers",
                        "owner_session": "codex-p19-repair-7292",
                        "branch": "droid/P16-stage2-auto-merge-bucket-a-20260518-002325",
                        "pr_number": 7292,
                        "last_seen_at": "2026-05-22T00:05:00Z",
                    },
                ]
            ),
            encoding="utf-8",
        )

        info = ilo.build_owner_info(
            SAMPLE_LANES[0],
            snapshot_provider=lambda: None,
            sessions_root=tmp_path / "codex",
            projects_root=tmp_path / "claude",
            bg_path=tmp_path / "factory.json",
            steering_inbox_root=tmp_path / "steering",
            heartbeat_path=heartbeat_path,
            heartbeat_now="2026-05-22T00:20:00Z",
        )

        assert info.latest_heartbeat is not None
        assert info.latest_heartbeat["owner_session"] == "codex-p19-repair-7292"
        assert info.latest_heartbeat["age_seconds"] == 900

    def test_build_owner_info_requires_target_lane_heartbeat(self, tmp_path: Path) -> None:
        heartbeat_path = tmp_path / "heartbeats.json"
        heartbeat_path.write_text(
            json.dumps(
                [
                    {
                        "lane_id": "other-lane",
                        "owner_session": "codex-p19-repair-7292",
                        "last_seen_at": "2026-05-22T00:10:00Z",
                    },
                    {
                        "lane_id": "P19-repair-7292-stage2-blockers",
                        "owner_session": "codex-p19-repair-7292",
                        "last_seen_at": "2026-05-22T00:00:00Z",
                    },
                ]
            ),
            encoding="utf-8",
        )

        info = ilo.build_owner_info(
            SAMPLE_LANES[0],
            snapshot_provider=lambda: None,
            sessions_root=tmp_path / "codex",
            projects_root=tmp_path / "claude",
            bg_path=tmp_path / "factory.json",
            steering_inbox_root=tmp_path / "steering",
            heartbeat_path=heartbeat_path,
            heartbeat_now="2026-05-22T00:20:00Z",
        )

        assert info.latest_heartbeat is not None
        assert info.latest_heartbeat["lane_id"] == "P19-repair-7292-stage2-blockers"
        assert info.latest_heartbeat["age_seconds"] == 1200

    def test_find_by_pr_prefers_conflict_over_newer_released_history(self) -> None:
        lanes = [
            {
                "lane_id": "newer-released",
                "owner_session": "codex-released",
                "status": "released",
                "pr_number": 7292,
                "updated_at": "2026-05-18T05:00:00Z",
            },
            {
                "lane_id": "older-conflict",
                "owner_session": "codex-conflict",
                "status": "conflict",
                "pr_number": 7292,
                "updated_at": "2026-05-18T04:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, pr=7292)
        assert r is not None
        assert r["lane_id"] == "older-conflict"

    def test_find_by_pr_treats_bad_or_missing_updated_at_as_oldest(self) -> None:
        lanes = [
            {
                "lane_id": "bad-time",
                "owner_session": "codex-bad",
                "status": "released",
                "pr_number": 7292,
                "updated_at": "not-a-timestamp",
            },
            {
                "lane_id": "missing-time",
                "owner_session": "codex-missing",
                "status": "released",
                "pr_number": 7292,
            },
            {
                "lane_id": "valid-time",
                "owner_session": "codex-valid",
                "status": "completed",
                "pr_number": 7292,
                "updated_at": "2026-05-18T04:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, pr=7292)
        assert r is not None
        assert r["lane_id"] == "valid-time"

    def test_find_by_branch(self) -> None:
        r = ilo.find_lane(
            SAMPLE_LANES, branch="droid/P20-model-pins-frontier-aligned-20260518-041438"
        )
        assert r is not None
        assert r["lane_id"] == "P20-model-pins-frontier-aligned"

    def test_find_by_branch_uses_duplicate_lane_ranking(self) -> None:
        lanes = [
            {
                "lane_id": "newer-released",
                "owner_session": "codex-released",
                "status": "released",
                "branch": "codex/shared-branch",
                "updated_at": "2026-05-18T05:00:00Z",
            },
            {
                "lane_id": "older-conflict",
                "owner_session": "codex-conflict",
                "status": "conflict",
                "branch": "codex/shared-branch",
                "updated_at": "2026-05-18T04:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, branch="codex/shared-branch")
        assert r is not None
        assert r["lane_id"] == "older-conflict"

    def test_find_by_worktree_uses_duplicate_lane_ranking(self) -> None:
        lanes = [
            {
                "lane_id": "older-released",
                "owner_session": "codex-released",
                "status": "released",
                "worktree": "/private/tmp/shared-worktree",
                "updated_at": "2026-05-18T05:00:00Z",
            },
            {
                "lane_id": "current-active",
                "owner_session": "codex-active",
                "status": "active",
                "worktree": "/private/tmp/shared-worktree",
                "updated_at": "2026-05-18T04:00:00Z",
            },
        ]
        r = ilo.find_lane(lanes, worktree="/private/tmp/shared-worktree/")
        assert r is not None
        assert r["lane_id"] == "current-active"

    def test_find_by_worktree_path_normalised(self) -> None:
        # Trailing-slash variant must match the registry's path.
        r = ilo.find_lane(SAMPLE_LANES, worktree="/private/tmp/p19-fixture-wt/")
        assert r is not None
        assert r["lane_id"] == "P19-repair-7292-stage2-blockers"

    def test_find_by_worktree_exact(self) -> None:
        r = ilo.find_lane(SAMPLE_LANES, worktree="/private/tmp/p19-fixture-wt")
        assert r is not None
        assert r["lane_id"] == "P19-repair-7292-stage2-blockers"

    def test_no_match_returns_none(self) -> None:
        assert ilo.find_lane(SAMPLE_LANES, lane_id="does-not-exist") is None
        assert ilo.find_lane(SAMPLE_LANES, pr=999999) is None
        assert ilo.find_lane(SAMPLE_LANES, branch="unknown") is None
        assert ilo.find_lane(SAMPLE_LANES, worktree="/nowhere") is None


# ---------------------------------------------------------------------------
# lookup_live_process
# ---------------------------------------------------------------------------


class TestLookupLiveProcess:
    def test_matches_codex_cli_pid_by_cwd(self) -> None:
        lane = {"worktree": "/private/tmp/p19-fixture-wt"}
        snap = fake_snapshot_records(
            [
                {"pid": 12345, "role": "codex_cli", "cwd": "/private/tmp/p19-fixture-wt"},
                {"pid": 12346, "role": "codex_cli", "cwd": "/elsewhere"},
                {"pid": 22222, "role": "claude_code", "cwd": "/another/dir"},
            ],
            by_role={"codex_cli": 2, "claude_code": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is True
        assert r["pid"] == 12345
        assert r["family"] == "codex_cli"

    def test_no_worktree_returns_not_found(self) -> None:
        r = ilo.lookup_live_process({}, snapshot_provider=lambda: fake_snapshot_records([]))
        assert r["found"] is False
        assert "no worktree" in r["reason"]

    def test_snapshot_unavailable_returns_not_found(self) -> None:
        r = ilo.lookup_live_process({"worktree": "/x"}, snapshot_provider=lambda: None)
        assert r["found"] is False
        assert "snapshot unavailable" in r["reason"]

    def test_no_process_match_returns_not_found(self) -> None:
        lane = {"worktree": "/private/tmp/nope"}
        snap = fake_snapshot_records(
            [{"pid": 1, "role": "codex_cli", "cwd": "/elsewhere"}],
            by_role={"codex_cli": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is False
        assert "no process_census entry matched" in r["reason"]

    def test_real_snapshot_shape_without_cwd_fails_closed(self) -> None:
        lane = {"worktree": "/private/tmp/shared-wt"}
        snap = fake_snapshot_records(
            [
                {
                    "pid": 11111,
                    "role": "claude_code",
                    "elapsed": "00:01:00",
                    "summary": "Claude Code local session process",
                },
                {
                    "pid": 22222,
                    "role": "codex_cli",
                    "elapsed": "00:02:00",
                    "summary": "Codex CLI session process",
                },
            ],
            by_role={"claude_code": 1, "codex_cli": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is False
        assert "no cwd-bearing process records" in r["reason"]

    def test_real_summary_snapshot_shape_without_records_fails_closed(self) -> None:
        lane = {"worktree": "/private/tmp/shared-wt"}
        snap = {"process_census": {"by_role": {"claude_code": 1, "codex_cli": 1}}}
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is False
        assert "no cwd-bearing process records" in r["reason"]

    def test_multiple_families_same_worktree_uses_lane_source(self) -> None:
        lane = {"source": "claude", "worktree": "/private/tmp/shared-wt"}
        snap = fake_snapshot_records(
            [
                {"pid": 11111, "role": "codex_cli", "cwd": "/private/tmp/shared-wt"},
                {"pid": 22222, "role": "claude_code", "cwd": "/private/tmp/shared-wt"},
            ],
            by_role={"codex_cli": 1, "claude_code": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is True
        assert r["pid"] == 22222
        assert r["family"] == "claude_code"
        assert "disambiguated" in r["matched_via"]

    def test_multiple_families_same_worktree_uses_owner_session_family(self) -> None:
        lane = {"owner_session": "droid-ABC12345", "worktree": "/private/tmp/shared-wt"}
        snap = fake_snapshot_records(
            [
                {"pid": 11111, "role": "codex_cli", "cwd": "/private/tmp/shared-wt"},
                {"pid": 33333, "role": "factory_droid", "cwd": "/private/tmp/shared-wt"},
            ],
            by_role={"codex_cli": 1, "factory_droid": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is True
        assert r["pid"] == 33333
        assert r["family"] == "factory_droid"

    def test_multiple_families_same_worktree_without_hint_fails_closed(self) -> None:
        lane = {"worktree": "/private/tmp/shared-wt"}
        snap = fake_snapshot_records(
            [
                {"pid": 11111, "role": "codex_cli", "cwd": "/private/tmp/shared-wt"},
                {"pid": 22222, "role": "claude_code", "cwd": "/private/tmp/shared-wt"},
            ],
            by_role={"codex_cli": 1, "claude_code": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is False
        assert "ambiguous_same_worktree" in r["reason"]
        assert [m["family"] for m in r["matches"]] == ["claude_code", "codex_cli"]

    def test_multiple_hinted_matches_same_worktree_fails_closed(self) -> None:
        lane = {"source": "codex", "worktree": "/private/tmp/shared-wt"}
        snap = fake_snapshot_records(
            [
                {"pid": 44444, "role": "codex_app_server", "cwd": "/private/tmp/shared-wt"},
                {"pid": 11111, "role": "codex_cli", "cwd": "/private/tmp/shared-wt"},
            ],
            by_role={"codex_app_server": 1, "codex_cli": 1},
        )
        r = ilo.lookup_live_process(lane, snapshot_provider=lambda: snap)
        assert r["found"] is False
        assert "ambiguous_same_worktree" in r["reason"]
        assert "still matched 2 entries" in r["reason"]


# ---------------------------------------------------------------------------
# lookup_codex_thread
# ---------------------------------------------------------------------------


class TestLookupCodexThread:
    def _make_rollout(self, sessions_root: Path, thread_id: str, body: str = "") -> Path:
        # Filename convention: rollout-YYYY-MM-DDTHH-MM-SS-<thread_id>.jsonl
        day_dir = sessions_root / "2026" / "05" / "18"
        day_dir.mkdir(parents=True, exist_ok=True)
        p = day_dir / f"rollout-2026-05-18T04-37-00-{thread_id}.jsonl"
        p.write_text(body or '{"event": "noop"}\n', encoding="utf-8")
        return p

    def test_exact_match_via_codex_rollout_path(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "codex_sessions"
        p = self._make_rollout(sessions_root, "abcd1234")
        lane = {"codex_rollout_path": str(p), "worktree": "/anywhere"}
        r = ilo.lookup_codex_thread(lane, sessions_root=sessions_root)
        assert r["found"] is True
        assert r["matched_via"] == "lane.codex_rollout_path (exact)"
        assert r["thread_id"] == "abcd1234"

    def test_exact_match_via_codex_thread_id_filename(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "codex_sessions"
        thread_id = "019e3942-e27e-7e72-b8d6-b61d981fd532"
        self._make_rollout(sessions_root, thread_id)
        lane = {"codex_thread_id": thread_id, "worktree": "/anywhere"}
        r = ilo.lookup_codex_thread(lane, sessions_root=sessions_root)
        assert r["found"] is True
        assert "exact filename match" in r["matched_via"]
        assert r["thread_id"] == thread_id

    def test_fuzzy_match_via_worktree_in_rollout_body(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "codex_sessions"
        wt = "/private/tmp/p19-fuzzy-target"
        body = '{"event":"tool_call","cwd":"' + wt + '","payload":"..."}\n'
        p = self._make_rollout(sessions_root, "ffff0000", body=body)
        lane = {"worktree": wt}
        r = ilo.lookup_codex_thread(
            lane,
            sessions_root=sessions_root,
            now=p.stat().st_mtime + 60,  # within freshness window
        )
        assert r["found"] is True
        assert "fuzzy" in r["matched_via"]
        assert r["thread_id"] == "ffff0000"

    def test_fuzzy_no_recent_match(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "codex_sessions"
        wt = "/private/tmp/p19-fuzzy-target"
        p = self._make_rollout(sessions_root, "ffff0001", body=f"cwd:{wt}\n")
        lane = {"worktree": wt}
        # Set now far in the future so the rollout is outside the fuzzy window.
        future_now = p.stat().st_mtime + (10 * 60 * 60)  # 10h later
        r = ilo.lookup_codex_thread(
            lane,
            sessions_root=sessions_root,
            now=future_now,
            fuzzy_max_age_seconds=60,
        )
        assert r["found"] is False
        assert "no recent codex rollout" in r["reason"]

    def test_missing_sessions_root(self, tmp_path: Path) -> None:
        r = ilo.lookup_codex_thread({"worktree": "/x"}, sessions_root=tmp_path / "nope")
        assert r["found"] is False
        assert "sessions root absent" in r["reason"]


# ---------------------------------------------------------------------------
# lookup_claude_session
# ---------------------------------------------------------------------------


class TestLookupClaudeSession:
    def test_finds_session_by_worktree_encoding(self, tmp_path: Path) -> None:
        projects_root = tmp_path / "claude_projects"
        cwd = "/Users/armand/Development/aragora/.worktrees/codex-auto/foo"
        # Claude encodes '/' → '-' and prefixes with a leading '-'.
        encoded = ilo._encode_cwd_for_claude(cwd)
        project_dir = projects_root / encoded
        project_dir.mkdir(parents=True)
        # Two sessions; lookup should return the most-recent.
        older = project_dir / "old-uuid-1111.jsonl"
        older.write_text('{"event":"a"}\n', encoding="utf-8")
        import os as _os
        import time as _time

        _os.utime(older, (_time.time() - 1000, _time.time() - 1000))
        newer = project_dir / "new-uuid-2222.jsonl"
        newer.write_text('{"event":"b"}\n', encoding="utf-8")
        lane = {"worktree": cwd}
        r = ilo.lookup_claude_session(lane, projects_root=projects_root)
        assert r["found"] is True
        assert r["session_uuid"] == "new-uuid-2222"
        assert "most-recent" in r["matched_via"]

    def test_no_matching_project_dir(self, tmp_path: Path) -> None:
        projects_root = tmp_path / "claude_projects"
        projects_root.mkdir()
        lane = {"worktree": "/nowhere/expected"}
        r = ilo.lookup_claude_session(lane, projects_root=projects_root)
        assert r["found"] is False
        assert "no claude project dir matched" in r["reason"]

    def test_project_dir_with_no_session_files(self, tmp_path: Path) -> None:
        projects_root = tmp_path / "claude_projects"
        cwd = "/Users/armand/Development/aragora"
        encoded = ilo._encode_cwd_for_claude(cwd)
        (projects_root / encoded).mkdir(parents=True)
        # No .jsonl files inside.
        r = ilo.lookup_claude_session({"worktree": cwd}, projects_root=projects_root)
        assert r["found"] is False
        assert "no .jsonl session files" in r["reason"]


# ---------------------------------------------------------------------------
# lookup_factory_droid
# ---------------------------------------------------------------------------


class TestLookupFactoryDroid:
    def test_matches_by_branch(self, tmp_path: Path) -> None:
        bg = tmp_path / "background-processes.json"
        bg.write_text(
            json.dumps(
                [
                    {"id": "p1", "branch": "droid/X-1"},
                    {"id": "p2", "branch": "droid/X-2"},
                ]
            ),
            encoding="utf-8",
        )
        lane = {"branch": "droid/X-2"}
        r = ilo.lookup_factory_droid(lane, bg_path=bg)
        assert r["found"] is True
        assert r["process_id"] == "p2"
        assert "branch" in r["matched_via"]

    def test_matches_by_worktree(self, tmp_path: Path) -> None:
        bg = tmp_path / "background-processes.json"
        bg.write_text(
            json.dumps(
                {
                    "processes": [
                        {"id": "p9", "worktree": "/some/where/X"},
                        {"id": "p10", "cwd": "/private/tmp/target"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        lane = {"worktree": "/private/tmp/target"}
        r = ilo.lookup_factory_droid(lane, bg_path=bg)
        assert r["found"] is True
        assert r["process_id"] == "p10"

    def test_missing_file(self, tmp_path: Path) -> None:
        r = ilo.lookup_factory_droid({"branch": "x"}, bg_path=tmp_path / "absent.json")
        assert r["found"] is False
        assert "absent" in r["reason"]


# ---------------------------------------------------------------------------
# steering_inbox_for
# ---------------------------------------------------------------------------


class TestSteeringInbox:
    def test_missing_inbox_dir_returns_zero_count(self, tmp_path: Path) -> None:
        path, count, receipt_summary = ilo.steering_inbox_for(
            "nobody-1", root=tmp_path / "steering"
        )
        assert count == 0
        assert path == tmp_path / "steering" / "nobody-1"
        assert receipt_summary["read_receipt_count"] == 0
        assert receipt_summary["unread_message_count"] == 0
        assert receipt_summary["latest_read_receipt"] is None

    def test_counts_only_dot_json_files(self, tmp_path: Path) -> None:
        inbox = tmp_path / "steering" / "claude-X"
        inbox.mkdir(parents=True)
        (inbox / "msg-a.json").write_text("{}", encoding="utf-8")
        (inbox / "msg-b.json").write_text("{}", encoding="utf-8")
        (inbox / "README.md").write_text("docs only", encoding="utf-8")
        path, count, receipt_summary = ilo.steering_inbox_for(
            "claude-X", root=tmp_path / "steering"
        )
        assert count == 2
        assert path == inbox
        assert receipt_summary["read_receipt_count"] == 0
        assert receipt_summary["unread_message_count"] == 2
        assert receipt_summary["latest_read_receipt"] is None

    def test_summarizes_read_receipts_without_changing_pending_count(self, tmp_path: Path) -> None:
        inbox = tmp_path / "steering" / "claude-X"
        receipts = inbox / "_read_receipts"
        receipts.mkdir(parents=True)
        (inbox / "msg-a.json").write_text(
            json.dumps(
                {
                    "schema_version": "aragora-operator-steering/1.0",
                    "message_sha256": "aaa",
                    "sent_at_utc": "2026-05-18T01:00:00.000Z",
                }
            ),
            encoding="utf-8",
        )
        (inbox / "msg-b.json").write_text(
            json.dumps(
                {
                    "schema_version": "aragora-operator-steering/1.0",
                    "message_sha256": "bbb",
                    "sent_at_utc": "2026-05-18T02:00:00.000Z",
                }
            ),
            encoding="utf-8",
        )
        (receipts / "receipt-a.json").write_text(
            json.dumps(
                {
                    "schema_version": "aragora-operator-steering-read-receipt/1.0",
                    "owner_session": "claude-X",
                    "read_by_session": "reader",
                    "read_at_utc": "2026-05-18T03:00:00.000Z",
                    "message_filename": "msg-a.json",
                    "message_sha256": "aaa",
                    "outcome": "stale",
                    "subject": "msg-a",
                }
            ),
            encoding="utf-8",
        )

        path, count, receipt_summary = ilo.steering_inbox_for(
            "claude-X", root=tmp_path / "steering"
        )

        assert path == inbox
        assert count == 2
        assert receipt_summary["read_receipt_count"] == 1
        assert receipt_summary["unread_message_count"] == 1
        assert receipt_summary["latest_read_receipt"]["message_filename"] == "msg-a.json"
        assert receipt_summary["latest_read_receipt"]["outcome"] == "stale"


# ---------------------------------------------------------------------------
# build_owner_info (composition)
# ---------------------------------------------------------------------------


class TestBuildOwnerInfo:
    def test_composes_all_fields_for_rich_identity_lane(self, tmp_path: Path) -> None:
        # Sources are all tmp dirs so lookups are deterministic.
        sessions_root = tmp_path / "codex_sessions"
        projects_root = tmp_path / "claude_projects"
        bg = tmp_path / "factory_bg.json"
        bg.write_text("[]", encoding="utf-8")
        lane = dict(SAMPLE_LANES[2])  # P28-with-rich-identity
        info = ilo.build_owner_info(
            lane,
            snapshot_provider=lambda: fake_snapshot_records([]),
            sessions_root=sessions_root,
            projects_root=projects_root,
            bg_path=bg,
            steering_inbox_root=tmp_path / "steering",
        )
        assert info.lane_id == "P28-with-rich-identity"
        assert info.owner_session == "codex-test-rich"
        assert info.codex_thread_id == "019e3942-e27e-7e72-b8d6-b61d981fd532"
        assert info.desktop_label == "Test Codex Desktop Tab"
        assert info.session_title == "Rich identity claim"
        assert info.live_prompt_dispatchable is True
        assert info.mailbox_dispatchable is True
        assert info.pending_message_count == 0
        assert info.read_receipt_count == 0
        assert info.unread_message_count == 0
        assert info.latest_read_receipt is None
        # Live lookups all return found=False because tmp dirs are empty.
        assert info.live_process["found"] is False
        assert info.claude_session["found"] is False
        assert info.factory_droid["found"] is False
        assert info.owner_state == "owned"
        assert info.liveness_state == "missing_heartbeat"
        assert info.cleanup_state == "preserve_unverified_owner"
        assert info.owner_state_reason == "active lane has no heartbeat evidence"

    def test_contact_metadata_surfaces_and_controls_dispatch_split(self, tmp_path: Path) -> None:
        bg = tmp_path / "factory_bg.json"
        bg.write_text("[]", encoding="utf-8")
        lane = {
            "lane_id": "tmux-lane",
            "owner_session": "codex-tmux",
            "status": "active",
            "contact_method": "tmux:aragora:2",
            "contact_payload": {"target": "aragora:2"},
            "last_mailbox_check_at": "2026-05-20T01:00:00Z",
            "last_delivery_at": "2026-05-20T01:01:00Z",
            "last_ack_at": "2026-05-20T01:02:00Z",
        }

        info = ilo.build_owner_info(
            lane,
            snapshot_provider=lambda: fake_snapshot_records([]),
            sessions_root=tmp_path / "codex_sessions",
            projects_root=tmp_path / "claude_projects",
            bg_path=bg,
            steering_inbox_root=tmp_path / "steering",
        )

        assert info.contact_method == "tmux:aragora:2"
        assert info.contact_payload == {"target": "aragora:2"}
        assert info.last_mailbox_check_at == "2026-05-20T01:00:00Z"
        assert info.last_delivery_at == "2026-05-20T01:01:00Z"
        assert info.last_ack_at == "2026-05-20T01:02:00Z"
        assert info.mailbox_dispatchable is True
        assert info.live_prompt_dispatchable is True

    def test_owner_state_marks_conflict_as_duplicate_preserve(self, tmp_path: Path) -> None:
        bg = tmp_path / "factory_bg.json"
        bg.write_text("[]", encoding="utf-8")
        lane = {
            "lane_id": "duplicate-lane",
            "owner_session": "codex-conflict",
            "status": "conflict",
            "worktree": "/tmp/duplicate-worktree",
        }

        info = ilo.build_owner_info(
            lane,
            snapshot_provider=lambda: fake_snapshot_records([]),
            sessions_root=tmp_path / "codex_sessions",
            projects_root=tmp_path / "claude_projects",
            bg_path=bg,
            steering_inbox_root=tmp_path / "steering",
        )

        assert info.owner_state == "duplicate"
        assert info.liveness_state == "missing_heartbeat"
        assert info.cleanup_state == "preserve_duplicate_owner"
        assert info.dispatchable is False
        assert info.recommended_operator_action == (
            "resolve the lane conflict before mutation or cleanup"
        )

    def test_owner_state_marks_completed_lane_as_stale_historical(self, tmp_path: Path) -> None:
        bg = tmp_path / "factory_bg.json"
        bg.write_text("[]", encoding="utf-8")
        lane = {
            "lane_id": "completed-lane",
            "owner_session": "codex-finished",
            "status": "completed",
            "worktree": "/tmp/completed-worktree",
        }

        info = ilo.build_owner_info(
            lane,
            snapshot_provider=lambda: fake_snapshot_records([]),
            sessions_root=tmp_path / "codex_sessions",
            projects_root=tmp_path / "claude_projects",
            bg_path=bg,
            steering_inbox_root=tmp_path / "steering",
        )

        assert info.owner_state == "stale"
        assert info.liveness_state == "missing_heartbeat"
        assert info.cleanup_state == "historical_requires_cleanup_inspect"
        assert info.dispatchable is False
        assert info.owner_state_reason == "lane status is completed"
        assert info.recommended_operator_action == (
            "treat as historical; run fresh cleanup inspection before any deletion"
        )


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    def _cli_args(self, registry: Path, tmp_path: Path) -> list[str]:
        return [
            "--registry-path",
            str(registry),
            "--codex-sessions-root",
            str(tmp_path / "no_codex"),
            "--claude-projects-root",
            str(tmp_path / "no_claude"),
            "--factory-bg-path",
            str(tmp_path / "no_factory.json"),
            "--steering-inbox-root",
            str(tmp_path / "no_steering"),
        ]

    def test_no_criteria_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        registry = write_lane_registry(tmp_path)
        rc = ilo.main(self._cli_args(registry, tmp_path))
        assert rc == 2
        assert "at least one of" in capsys.readouterr().err

    def test_missing_registry_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = ilo.main(
            [
                "--lane-id",
                "P19-repair-7292-stage2-blockers",
                "--registry-path",
                str(tmp_path / "absent.json"),
            ]
        )
        assert rc == 2
        assert "lane registry empty or missing" in capsys.readouterr().err

    def test_no_match_exits_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        registry = write_lane_registry(tmp_path)
        rc = ilo.main(["--lane-id", "does-not-exist", *self._cli_args(registry, tmp_path)])
        assert rc == 1
        assert "no lane matched" in capsys.readouterr().err

    def test_happy_path_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        registry = write_lane_registry(tmp_path)
        rc = ilo.main(
            [
                "--pr",
                "7292",
                "--json",
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["lane_id"] == "P19-repair-7292-stage2-blockers"
        assert data["owner_session"] == "codex-p19-repair-7292"
        assert data["pr_number"] == 7292
        assert data["live_process"]["found"] is False  # no snapshot integration in CLI default path
        assert data["pending_message_count"] == 0
        assert data["read_receipt_count"] == 0
        assert data["unread_message_count"] == 0
        assert data["latest_read_receipt"] is None
        assert data["dispatchable"] is True
        assert data["dispatch_blocker"] is None
        assert data["harness_confidence"] == "mailbox_only"
        assert "send_operator_steering.py --to codex-p19-repair-7292" in data["steering_command"]

    def test_completed_lane_reports_mailbox_only_but_not_dispatchable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry = write_lane_registry(
            tmp_path,
            [
                {
                    "lane_id": "q25-finished",
                    "owner_session": "codex-finished",
                    "source": "codex",
                    "status": "released",
                    "branch": "codex/finished",
                    "worktree": "/tmp/finished",
                    "pr_number": 7370,
                    "updated_at": "2026-05-19T17:49:14Z",
                }
            ],
        )

        rc = ilo.main(["--pr", "7370", "--json", *self._cli_args(registry, tmp_path)])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["owner_session"] == "codex-finished"
        assert data["dispatchable"] is False
        assert data["dispatch_blocker"] == (
            "lane status is released; claim an active lane before steering"
        )
        assert data["steering_command"] is None
        assert data["harness_confidence"] == "mailbox_only"

    def test_happy_path_human(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        registry = write_lane_registry(tmp_path)
        rc = ilo.main(
            [
                "--branch",
                "droid/P20-model-pins-frontier-aligned-20260518-041438",
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "lane_id:" in out
        assert "P20-model-pins-frontier-aligned" in out
        assert "owner_session:" in out
        assert "droid-F473CDBF" in out


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------


class TestEncodeCwdForClaude:
    def test_basic_encoding(self) -> None:
        assert ilo._encode_cwd_for_claude("/Users/x") == "-Users-x"

    def test_trailing_slash_stripped(self) -> None:
        assert ilo._encode_cwd_for_claude("/Users/x/") == ilo._encode_cwd_for_claude("/Users/x")

    def test_no_leading_slash_gets_dash(self) -> None:
        assert ilo._encode_cwd_for_claude("rel/path") == "-rel-path"


# ---------------------------------------------------------------------------
# Owner-lease liveness + stale-claim advisory (issue #8318)
# ---------------------------------------------------------------------------


LIVENESS_NOW = "2026-06-13T12:00:00Z"


def _liveness_now() -> Any:
    return ilo._parse_iso_utc(LIVENESS_NOW)


def _hours_ago(hours: float) -> str:
    from datetime import timedelta

    return (_liveness_now() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def write_lane_ledger(tmp_path: Path, entries: list[dict[str, Any]]) -> str:
    """Write lane-ledger fixtures in the lane_janitor layout; return runs glob."""

    lanes_dir = tmp_path / ".aragora" / "run-20260613-liveness" / "lanes"
    lanes_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries):
        name = str(entry.get("lane") or f"lane-{i}")
        (lanes_dir / f"{name}.json").write_text(json.dumps(entry), encoding="utf-8")
    return str(tmp_path / ".aragora" / "run-*" / "lanes")


def write_preservation_outbox(
    tmp_path: Path,
    *,
    lane_id: str,
    branch: str,
    desired_head_sha: str,
) -> Path:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"open-pr-{branch.replace('/', '-')}-{desired_head_sha[:8]}.json"
    path.write_text(
        json.dumps(
            {
                "lane_id": lane_id,
                "branch": branch,
                "desired_head_sha": desired_head_sha,
            }
        ),
        encoding="utf-8",
    )
    return path


def completed(
    cmd: list[str], *, stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")


def safe_inspect_payload(*, exists: bool) -> str:
    blockers = [] if exists else ["missing_path"]
    return json.dumps(
        {
            "exists": exists,
            "tracked_worktree": exists,
            "active_session": False,
            "dirty": False,
            "blockers": blockers,
            "cleanup_safety": {
                "classification": "cleanup_candidate" if exists else "absent_noop",
                "decision": "cleanup_candidate" if exists else "noop",
            },
        }
    )


class TestOwnerLeaseLiveness:
    def test_live_owner_no_advisory(self) -> None:
        lane = {
            "lane_id": "Q1-live",
            "owner_session": "codex-q1",
            "status": "active",
            "branch": "codex/q1",
            "updated_at": _hours_ago(1.0),
        }
        ledger = {"lane": "Q1-live", "status": "in_progress", "launched_at": _hours_ago(1.0)}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        liveness = result["owner_liveness"]
        assert liveness["assessed"] == "live"
        assert liveness["lane_status"] == "in_progress"
        assert liveness["lease_age_seconds"] == 3600
        assert liveness["last_heartbeat_at"] is None
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] is None

    def test_terminal_completed_lane_yields_advisory(self) -> None:
        lane = {
            "lane_id": "Q2-done",
            "owner_session": "codex-q2",
            "status": "active",
            "branch": "codex/q2",
            "updated_at": _hours_ago(7.0),
        }
        ledger = {"lane": "Q2-done", "status": "completed", "launched_at": _hours_ago(7.0)}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        assert result["owner_liveness"]["assessed"] == "terminal"
        assert result["owner_liveness"]["lane_status"] == "completed"
        advisory = result["stale_claim_advisory"]
        assert advisory is not None
        assert advisory["available"] is True
        assert advisory["protocol"] == "stale-claim-override"
        assert advisory["required_ledger_record"] == (
            "overriding lane must write an override entry naming the stale lane id"
        )
        assert any("terminal" in c for c in advisory["conditions_met"])
        assert result["advisory_withheld"] is None

    @pytest.mark.parametrize("status", ["failed", "cancelled"])
    def test_failed_and_cancelled_ledger_statuses_are_terminal(self, status: str) -> None:
        lane = {"lane_id": "Q3", "owner_session": "x", "updated_at": _hours_ago(7.0)}
        ledger = {"lane": "Q3", "status": status, "launched_at": _hours_ago(7.0)}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        assert result["owner_liveness"]["assessed"] == "terminal"
        assert result["stale_claim_advisory"] is not None

    def test_stale_in_progress_without_heartbeat_yields_advisory(self) -> None:
        lane = {
            "lane_id": "Q4-stale",
            "owner_session": "codex-q4",
            "status": "active",
            "branch": "codex/q4",
            "updated_at": _hours_ago(7.0),
        }
        ledger = {"lane": "Q4-stale", "status": "in_progress", "launched_at": _hours_ago(7.0)}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        liveness = result["owner_liveness"]
        assert liveness["assessed"] == "stale"
        assert liveness["lane_status"] == "in_progress"
        assert liveness["lease_age_seconds"] == 7 * 3600
        advisory = result["stale_claim_advisory"]
        assert advisory is not None
        assert advisory["available"] is True
        assert any("lease_age_seconds" in c for c in advisory["conditions_met"])
        assert any("no heartbeat" in c for c in advisory["conditions_met"])
        assert result["advisory_withheld"] is None

    def test_worktree_reference_withholds_advisory(self) -> None:
        lane = {
            "lane_id": "Q5-wt",
            "owner_session": "codex-q5",
            "status": "active",
            "worktree": "/private/tmp/q5-worktree",
            "updated_at": _hours_ago(7.0),
        }
        ledger = {"lane": "Q5-wt", "status": "in_progress", "launched_at": _hours_ago(7.0)}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        assert result["owner_liveness"]["assessed"] == "stale"
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] == "possible_unpushed_work"

    def test_uncommitted_work_claim_withholds_advisory(self) -> None:
        lane = {"lane_id": "Q6-dirty", "owner_session": "codex-q6", "updated_at": _hours_ago(7.0)}
        ledger = {
            "lane": "Q6-dirty",
            "status": "completed",
            "launched_at": _hours_ago(7.0),
            "uncommitted_changes": True,
        }
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        assert result["owner_liveness"]["assessed"] == "terminal"
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] == "possible_unpushed_work"

    def test_unknown_timestamps_never_produce_advisory(self) -> None:
        lane = {"lane_id": "Q7-unknown", "owner_session": "codex-q7", "status": "active"}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=None, heartbeat=None, now=_liveness_now()
        )
        liveness = result["owner_liveness"]
        assert liveness["assessed"] == "unknown"
        assert liveness["lease_age_seconds"] is None
        assert liveness["lane_status"] == "unknown"
        assert liveness["last_heartbeat_at"] is None
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] is None

    def test_lease_just_under_stale_hours_is_live(self) -> None:
        # 1 minute inside the default 6h window → live, no advisory.
        from datetime import timedelta

        updated = (_liveness_now() - timedelta(hours=6) + timedelta(minutes=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        lane = {"lane_id": "Q8-boundary", "owner_session": "codex-q8", "updated_at": updated}
        ledger = {"lane": "Q8-boundary", "status": "in_progress", "launched_at": updated}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=ledger, heartbeat=None, now=_liveness_now()
        )
        assert result["owner_liveness"]["assessed"] == "live"
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] is None

    def test_fresh_heartbeat_keeps_old_lease_live(self) -> None:
        lane = {"lane_id": "Q9-hb", "owner_session": "codex-q9", "updated_at": _hours_ago(7.0)}
        heartbeat = {"last_seen_at": _hours_ago(0.1)}
        result = ilo.assess_owner_liveness(
            lane, ledger_entry=None, heartbeat=heartbeat, now=_liveness_now()
        )
        liveness = result["owner_liveness"]
        assert liveness["assessed"] == "live"
        assert liveness["last_heartbeat_at"] == _hours_ago(0.1)
        assert result["stale_claim_advisory"] is None

    def test_find_lane_ledger_entry_matches_by_branch_and_picks_newest(
        self, tmp_path: Path
    ) -> None:
        runs_glob = write_lane_ledger(
            tmp_path,
            [
                {
                    "lane": "older-attempt",
                    "branch": "codex/shared",
                    "status": "dead",
                    "launched_at": _hours_ago(30.0),
                },
                {
                    "lane": "newer-attempt",
                    "branch": "codex/shared",
                    "status": "in_progress",
                    "launched_at": _hours_ago(2.0),
                },
            ],
        )
        lane = {"lane_id": "not-in-ledger", "branch": "codex/shared"}
        entry = ilo.find_lane_ledger_entry(lane, runs_glob=runs_glob)
        assert entry is not None
        assert entry["lane"] == "newer-attempt"
        assert entry["status"] == "in_progress"

    def test_find_lane_ledger_entry_missing_returns_none(self, tmp_path: Path) -> None:
        runs_glob = write_lane_ledger(tmp_path, [])
        assert ilo.find_lane_ledger_entry({"lane_id": "nope"}, runs_glob=runs_glob) is None


class TestWorktreeReferencePreservationProof:
    def test_q467_absent_worktree_remote_branch_exact_head_yields_advisory(
        self, tmp_path: Path
    ) -> None:
        desired_sha = "4966b95bec51fac1ae102443d5e7a2974e03065d"
        branch = "codex/measure-work-loss-pending-outbox-primary-20260610"
        lane = {
            "lane_id": "Q467-primary-measure-work-loss-pending-outbox",
            "owner_session": "codex-q467",
            "branch": branch,
            "worktree": str(tmp_path / "absent-q467"),
            "updated_at": _hours_ago(7.0),
        }
        ledger = {
            "lane": lane["lane_id"],
            "branch": branch,
            "status": "in_progress",
            "launched_at": _hours_ago(7.0),
        }
        write_preservation_outbox(
            tmp_path,
            lane_id=lane["lane_id"],
            branch=branch,
            desired_head_sha=desired_sha,
        )
        calls: list[list[str]] = []

        def runner(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if "safe_worktree_cleanup.py" in " ".join(cmd):
                return completed(cmd, stdout=safe_inspect_payload(exists=False), returncode=1)
            if cmd[:3] == ["git", "ls-remote", "origin"]:
                return completed(cmd, stdout=f"{desired_sha}\trefs/heads/{branch}\n")
            raise AssertionError(f"unexpected command: {cmd}")

        proof = ilo.build_worktree_reference_preservation_proof(
            lane,
            ledger_entry=ledger,
            repo_root=tmp_path,
            state_root=tmp_path / ".aragora",
            runner=runner,
        )
        assert proof["available"] is True
        assert proof["upstream_preservation"]["method"] == "remote_branch_exact_head"
        assert not any(cmd[:2] == ["gh", "api"] for cmd in calls)

        result = ilo.assess_owner_liveness(
            lane,
            ledger_entry=ledger,
            heartbeat=None,
            now=_liveness_now(),
            local_work_preservation=proof,
        )
        assert result["stale_claim_advisory"]["available"] is True
        assert result["advisory_withheld"] is None

    def test_q379_absent_worktree_merged_pr_commit_yields_advisory_when_remote_gone(
        self, tmp_path: Path
    ) -> None:
        desired_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        branch = "codex/salvage-collect-evidence-quorum-rerun-20260606"
        lane = {
            "lane_id": "Q379-primary-salvage",
            "owner_session": "codex-q379",
            "branch": branch,
            "worktree": str(tmp_path / "absent-q379"),
            "updated_at": _hours_ago(7.0),
        }
        ledger = {
            "lane": lane["lane_id"],
            "branch": branch,
            "status": "completed",
            "launched_at": _hours_ago(7.0),
        }
        write_preservation_outbox(
            tmp_path,
            lane_id=lane["lane_id"],
            branch=branch,
            desired_head_sha=desired_sha,
        )

        def runner(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
            if "safe_worktree_cleanup.py" in " ".join(cmd):
                return completed(cmd, stdout=safe_inspect_payload(exists=False), returncode=1)
            if cmd[:3] == ["git", "ls-remote", "origin"]:
                return completed(cmd, stdout="")
            if cmd == ["git", "remote", "get-url", "origin"]:
                return completed(cmd, stdout="https://github.com/synaptent/aragora.git\n")
            if cmd[:2] == ["gh", "api"] and f"commits/{desired_sha}/pulls" in cmd[-1]:
                return completed(
                    cmd, stdout=json.dumps([{"number": 8396, "merged_at": LIVENESS_NOW}])
                )
            if cmd[:2] == ["gh", "api"] and "pulls/8396/commits" in cmd[-1]:
                return completed(cmd, stdout=json.dumps([{"sha": desired_sha}]))
            raise AssertionError(f"unexpected command: {cmd}")

        proof = ilo.build_worktree_reference_preservation_proof(
            lane,
            ledger_entry=ledger,
            repo_root=tmp_path,
            state_root=tmp_path / ".aragora",
            runner=runner,
        )
        assert proof["available"] is True
        assert proof["upstream_preservation"]["method"] == "merged_pr_commit_list"

        result = ilo.assess_owner_liveness(
            lane,
            ledger_entry=ledger,
            heartbeat=None,
            now=_liveness_now(),
            local_work_preservation=proof,
        )
        assert result["stale_claim_advisory"]["available"] is True
        assert result["advisory_withheld"] is None

    def test_present_worktree_still_withholds_possible_unpushed_work(self, tmp_path: Path) -> None:
        desired_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        branch = "codex/present-worktree"
        lane = {
            "lane_id": "Q-present",
            "owner_session": "codex-present",
            "branch": branch,
            "worktree": str(tmp_path / "present-worktree"),
            "updated_at": _hours_ago(7.0),
        }
        ledger = {
            "lane": lane["lane_id"],
            "branch": branch,
            "status": "in_progress",
            "launched_at": _hours_ago(7.0),
        }
        write_preservation_outbox(
            tmp_path,
            lane_id=lane["lane_id"],
            branch=branch,
            desired_head_sha=desired_sha,
        )

        def runner(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
            if "safe_worktree_cleanup.py" in " ".join(cmd):
                return completed(cmd, stdout=safe_inspect_payload(exists=True))
            raise AssertionError(f"unexpected command: {cmd}")

        proof = ilo.build_worktree_reference_preservation_proof(
            lane,
            ledger_entry=ledger,
            repo_root=tmp_path,
            state_root=tmp_path / ".aragora",
            runner=runner,
        )
        result = ilo.assess_owner_liveness(
            lane,
            ledger_entry=ledger,
            heartbeat=None,
            now=_liveness_now(),
            local_work_preservation=proof,
        )
        assert proof["available"] is False
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] == "possible_unpushed_work"

    def test_absent_worktree_without_upstream_proof_still_withholds(self, tmp_path: Path) -> None:
        desired_sha = "cccccccccccccccccccccccccccccccccccccccc"
        branch = "codex/no-upstream-proof"
        lane = {
            "lane_id": "Q-no-proof",
            "owner_session": "codex-no-proof",
            "branch": branch,
            "worktree": str(tmp_path / "absent-no-proof"),
            "updated_at": _hours_ago(7.0),
        }
        ledger = {
            "lane": lane["lane_id"],
            "branch": branch,
            "status": "in_progress",
            "launched_at": _hours_ago(7.0),
        }
        write_preservation_outbox(
            tmp_path,
            lane_id=lane["lane_id"],
            branch=branch,
            desired_head_sha=desired_sha,
        )

        def runner(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
            if "safe_worktree_cleanup.py" in " ".join(cmd):
                return completed(cmd, stdout=safe_inspect_payload(exists=False), returncode=1)
            if cmd[:3] == ["git", "ls-remote", "origin"]:
                return completed(cmd, stdout="")
            if cmd == ["git", "remote", "get-url", "origin"]:
                return completed(cmd, stdout="https://github.com/synaptent/aragora.git\n")
            if cmd[:2] == ["gh", "api"] and f"commits/{desired_sha}/pulls" in cmd[-1]:
                return completed(cmd, stdout="[]")
            raise AssertionError(f"unexpected command: {cmd}")

        proof = ilo.build_worktree_reference_preservation_proof(
            lane,
            ledger_entry=ledger,
            repo_root=tmp_path,
            state_root=tmp_path / ".aragora",
            runner=runner,
        )
        result = ilo.assess_owner_liveness(
            lane,
            ledger_entry=ledger,
            heartbeat=None,
            now=_liveness_now(),
            local_work_preservation=proof,
        )
        assert proof["available"] is False
        assert proof["reason"] == "upstream_preservation_unproven"
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] == "possible_unpushed_work"

    def test_dirty_marker_set_still_withholds_possible_unpushed_work(self, tmp_path: Path) -> None:
        branch = "codex/dirty-marker"
        lane = {
            "lane_id": "Q-dirty-marker",
            "owner_session": "codex-dirty-marker",
            "branch": branch,
            "worktree": str(tmp_path / "absent-dirty"),
            "local_work": True,
            "updated_at": _hours_ago(7.0),
        }
        ledger = {
            "lane": lane["lane_id"],
            "branch": branch,
            "status": "in_progress",
            "launched_at": _hours_ago(7.0),
        }

        def runner(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
            raise AssertionError(f"unexpected command: {cmd}")

        proof = ilo.build_worktree_reference_preservation_proof(
            lane,
            ledger_entry=ledger,
            repo_root=tmp_path,
            state_root=tmp_path / ".aragora",
            runner=runner,
        )
        result = ilo.assess_owner_liveness(
            lane,
            ledger_entry=ledger,
            heartbeat=None,
            now=_liveness_now(),
            local_work_preservation=proof,
        )
        assert proof["available"] is False
        assert proof["reason"] == "local_work_claim_present"
        assert result["stale_claim_advisory"] is None
        assert result["advisory_withheld"] == "possible_unpushed_work"


class TestLivenessCLI:
    def _cli_args(self, registry: Path, tmp_path: Path) -> list[str]:
        return [
            "--registry-path",
            str(registry),
            "--codex-sessions-root",
            str(tmp_path / "no_codex"),
            "--claude-projects-root",
            str(tmp_path / "no_claude"),
            "--factory-bg-path",
            str(tmp_path / "no_factory.json"),
            "--steering-inbox-root",
            str(tmp_path / "no_steering"),
            "--heartbeat-path",
            str(tmp_path / "no_heartbeats.json"),
        ]

    def _stale_fixture(self, tmp_path: Path) -> tuple[Path, str]:
        registry = write_lane_registry(
            tmp_path,
            [
                {
                    "lane_id": "Q379-stale-owner",
                    "owner_session": "codex-q379",
                    "source": "codex",
                    "status": "active",
                    "branch": "codex/q379",
                    "pr_number": 7825,
                    "updated_at": _hours_ago(7.0),
                }
            ],
        )
        runs_glob = write_lane_ledger(
            tmp_path,
            [
                {
                    "lane": "Q379-stale-owner",
                    "branch": "codex/q379",
                    "status": "in_progress",
                    "launched_at": _hours_ago(7.0),
                }
            ],
        )
        return registry, runs_glob

    def test_json_includes_owner_liveness_and_advisory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry, runs_glob = self._stale_fixture(tmp_path)
        rc = ilo.main(
            [
                "--lane-id",
                "Q379-stale-owner",
                "--json",
                "--runs-glob",
                runs_glob,
                "--now",
                LIVENESS_NOW,
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Existing fields still present and unchanged.
        assert data["lane_id"] == "Q379-stale-owner"
        assert data["owner_session"] == "codex-q379"
        assert data["pr_number"] == 7825
        # New advisory-only enrichment.
        assert data["owner_liveness"]["assessed"] == "stale"
        assert data["owner_liveness"]["lane_status"] == "in_progress"
        assert data["owner_liveness"]["lease_age_seconds"] == 7 * 3600
        assert data["stale_claim_advisory"]["available"] is True
        assert data["stale_claim_advisory"]["protocol"] == "stale-claim-override"
        assert data["advisory_withheld"] is None

    def test_custom_stale_hours_flag_keeps_owner_live(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry, runs_glob = self._stale_fixture(tmp_path)
        rc = ilo.main(
            [
                "--lane-id",
                "Q379-stale-owner",
                "--json",
                "--runs-glob",
                runs_glob,
                "--now",
                LIVENESS_NOW,
                "--stale-hours",
                "8",
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["owner_liveness"]["assessed"] == "live"
        assert data["stale_claim_advisory"] is None

    def test_no_liveness_output_is_byte_identical_to_legacy_schema(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import dataclasses as _dataclasses

        registry, runs_glob = self._stale_fixture(tmp_path)
        rc = ilo.main(
            [
                "--lane-id",
                "Q379-stale-owner",
                "--json",
                "--no-liveness",
                "--runs-glob",
                runs_glob,
                "--now",
                LIVENESS_NOW,
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        legacy_fields = {f.name for f in _dataclasses.fields(ilo.LaneOwnerInfo)}
        assert set(data.keys()) == legacy_fields
        # Byte-identical to the pre-#8318 serialization of the same info.
        lane = ilo.find_lane(ilo.load_lane_records(registry), lane_id="Q379-stale-owner")
        assert lane is not None
        info = ilo.build_owner_info(
            lane,
            sessions_root=tmp_path / "no_codex",
            projects_root=tmp_path / "no_claude",
            bg_path=tmp_path / "no_factory.json",
            steering_inbox_root=tmp_path / "no_steering",
            heartbeat_path=tmp_path / "no_heartbeats.json",
        )
        expected = json.dumps(_dataclasses.asdict(info), indent=2, sort_keys=True) + "\n"
        assert out == expected

    def test_human_output_gains_single_summary_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry, runs_glob = self._stale_fixture(tmp_path)
        rc = ilo.main(
            [
                "--lane-id",
                "Q379-stale-owner",
                "--runs-glob",
                runs_glob,
                "--now",
                LIVENESS_NOW,
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        summary_lines = [line for line in out.splitlines() if line.startswith("owner_liveness: ")]
        assert len(summary_lines) == 1
        assert "assessed=stale" in summary_lines[0]
        assert "stale_claim_advisory=available" in summary_lines[0]

    def test_human_output_omits_summary_with_no_liveness(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry, runs_glob = self._stale_fixture(tmp_path)
        rc = ilo.main(
            [
                "--lane-id",
                "Q379-stale-owner",
                "--no-liveness",
                "--runs-glob",
                runs_glob,
                *self._cli_args(registry, tmp_path),
            ]
        )
        assert rc == 0
        assert "owner_liveness: " not in capsys.readouterr().out
