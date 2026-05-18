"""Tests for ``scripts/identify_lane_owner.py`` — Phase A consolidator.

Fixture-driven; never calls the real ``agent_bridge`` subprocess and
never reads the live ``~/.codex/`` / ``~/.claude/`` / ``~/.factory/``
directories. All discovery sources are pointed at ``tmp_path``
fixtures so tests are deterministic and isolated.
"""

from __future__ import annotations

import importlib.util
import json
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

    def test_find_by_pr_number(self) -> None:
        r = ilo.find_lane(SAMPLE_LANES, pr=7292)
        assert r is not None
        assert r["lane_id"] == "P19-repair-7292-stage2-blockers"

    def test_find_by_branch(self) -> None:
        r = ilo.find_lane(
            SAMPLE_LANES, branch="droid/P20-model-pins-frontier-aligned-20260518-041438"
        )
        assert r is not None
        assert r["lane_id"] == "P20-model-pins-frontier-aligned"

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
        path, count = ilo.steering_inbox_for("nobody-1", root=tmp_path / "steering")
        assert count == 0
        assert path == tmp_path / "steering" / "nobody-1"

    def test_counts_only_dot_json_files(self, tmp_path: Path) -> None:
        inbox = tmp_path / "steering" / "claude-X"
        inbox.mkdir(parents=True)
        (inbox / "msg-a.json").write_text("{}", encoding="utf-8")
        (inbox / "msg-b.json").write_text("{}", encoding="utf-8")
        (inbox / "README.md").write_text("docs only", encoding="utf-8")
        path, count = ilo.steering_inbox_for("claude-X", root=tmp_path / "steering")
        assert count == 2
        assert path == inbox


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
        assert info.pending_message_count == 0
        # Live lookups all return found=False because tmp dirs are empty.
        assert info.live_process["found"] is False
        assert info.claude_session["found"] is False
        assert info.factory_droid["found"] is False


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
