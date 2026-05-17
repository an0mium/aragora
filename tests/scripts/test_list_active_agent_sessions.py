"""Focused tests for `scripts/list_active_agent_sessions.py`.

All tests are fixture-driven: no network, no subprocess to live tools,
no git invocations. The script's optional external calls (``git
worktree list``, ``gh pr list``, ``check_codex_desktop_automations.py``)
are exercised via patched stubs.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import list_active_agent_sessions as detector  # noqa: E402


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=dt.UTC)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stable_branch_name_strips_refs_heads() -> None:
    assert detector._stable_branch_name("refs/heads/main") == "main"
    assert detector._stable_branch_name("main") == "main"
    assert detector._stable_branch_name("") is None
    assert detector._stable_branch_name(None) is None


def test_detect_worktrees_parses_porcelain_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sample = (
        "worktree /repo/main\nHEAD abcd\nbranch refs/heads/main\n\n"
        "worktree /tmp/wt-a\nHEAD efgh\nbranch refs/heads/feature/a\nlocked\n\n"
        "worktree /tmp/wt-b\nHEAD ijkl\ndetached\n"
    )

    class _Proc:
        returncode = 0
        stdout = sample
        stderr = ""

    with patch.object(detector.subprocess, "run", return_value=_Proc()):
        out = detector.detect_worktrees(repo)

    assert out == [
        {"path": "/repo/main", "head": "abcd", "branch": "main"},
        {"path": "/tmp/wt-a", "head": "efgh", "branch": "feature/a", "locked": True},
        {"path": "/tmp/wt-b", "head": "ijkl", "detached": True},
    ]


def test_detect_worktrees_returns_empty_when_no_git_dir(tmp_path: Path) -> None:
    assert detector.detect_worktrees(tmp_path) == []


def test_detect_recent_jsonl_files_respects_age_threshold(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.json"
    stale = tmp_path / "stale.json"
    _write_json(fresh, {"branch": "feature/fresh", "issue_number": 7257})
    _write_json(stale, {"branch": "feature/stale"})

    now = _ts("2026-05-17T12:00:00Z")
    # set mtimes
    import os

    os.utime(fresh, times=(now.timestamp() - 60, now.timestamp() - 60))
    os.utime(stale, times=(now.timestamp() - 3600 * 5, now.timestamp() - 3600 * 5))

    rows = detector.detect_recent_jsonl_files(tmp_path, now=now, max_age_minutes=60.0, limit=10)
    names = [r["name"] for r in rows]
    assert "fresh.json" in names
    assert "stale.json" not in names
    fresh_row = next(r for r in rows if r["name"] == "fresh.json")
    assert fresh_row["branch"] == "feature/fresh"
    assert fresh_row["issue_number"] == 7257


def test_detect_recent_jsonl_files_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert (
        detector.detect_recent_jsonl_files(
            tmp_path / "missing",
            now=_ts("2026-05-17T12:00:00Z"),
            max_age_minutes=60.0,
        )
        == []
    )


def test_detect_fleet_coordination_handles_missing(tmp_path: Path) -> None:
    assert detector.detect_fleet_coordination(tmp_path) == {}


def test_detect_fleet_coordination_reads_json_and_lock(tmp_path: Path) -> None:
    (tmp_path / ".aragora").mkdir()
    _write_json(tmp_path / ".aragora" / "fleet_coordination.json", {"active": []})
    (tmp_path / ".aragora" / "fleet_coordination.lock").write_text(
        "agent=claude\n", encoding="utf-8"
    )
    out = detector.detect_fleet_coordination(tmp_path)
    assert out["fleet_coordination_json"] == {"active": []}
    assert "agent=claude" in out["fleet_coordination_lock"]


def test_detect_codex_desktop_automations_returns_empty_when_script_missing(
    tmp_path: Path,
) -> None:
    assert detector.detect_codex_desktop_automations(tmp_path) == {}


def test_detect_codex_cli_sessions_filters_by_age(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions" / "2026" / "05" / "17"
    sessions.mkdir(parents=True)
    fresh = sessions / "fresh.jsonl"
    stale = sessions / "stale.jsonl"
    fresh.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "thread-1",
                    "source": "exec",
                    "cwd": "/repo/worktree-a",
                    "git": {
                        "branch": "refs/heads/feature/fresh",
                        "commit_hash": "abc123",
                        "repository_url": "https://github.com/example/repo.git",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stale.write_text("{}\n", encoding="utf-8")
    now = _ts("2026-05-17T12:00:00Z")
    import os

    os.utime(fresh, times=(now.timestamp() - 60, now.timestamp() - 60))
    os.utime(stale, times=(now.timestamp() - 3600 * 10, now.timestamp() - 3600 * 10))
    out = detector.detect_codex_cli_sessions(tmp_path, now=now, max_age_minutes=120.0)
    names = [r["name"] for r in out]
    assert "fresh.jsonl" in names
    assert "stale.jsonl" not in names
    fresh_row = next(r for r in out if r["name"] == "fresh.jsonl")
    assert fresh_row["relative_path"] == "2026/05/17/fresh.jsonl"
    assert fresh_row["thread_id"] == "thread-1"
    assert fresh_row["source"] == "exec"
    assert fresh_row["cwd"] == "/repo/worktree-a"
    assert fresh_row["branch"] == "feature/fresh"
    assert fresh_row["commit_hash"] == "abc123"


def test_read_codex_session_metadata_ignores_message_content(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": "do not expose this",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "thread-2",
                            "cwd": "/repo",
                            "git": {"branch": "main", "commit_hash": "def456"},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert detector.read_codex_session_metadata(path) == {
        "thread_id": "thread-2",
        "cwd": "/repo",
        "branch": "main",
        "commit_hash": "def456",
    }


def test_detect_codex_cli_sessions_respects_output_limit(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions" / "2026" / "05" / "17"
    sessions.mkdir(parents=True)
    now = _ts("2026-05-17T12:00:00Z")
    import os

    for index in range(3):
        path = sessions / f"session-{index}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        mtime = now.timestamp() - index
        os.utime(path, times=(mtime, mtime))

    out = detector.detect_codex_cli_sessions(
        tmp_path,
        now=now,
        max_age_minutes=120.0,
        limit=2,
        scan_limit=3,
    )
    assert [row["name"] for row in out] == ["session-0.jsonl", "session-1.jsonl"]


def test_detect_agent_process_census_parses_summary(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "agent_bridge.py"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    class _Proc:
        returncode = 0
        stdout = json.dumps(
            {
                "ok": True,
                "total": 42,
                "by_role": {"codex_cli": 2, "factory_droid": 3},
            }
        )
        stderr = ""

    with patch.object(detector.subprocess, "run", return_value=_Proc()):
        out = detector.detect_agent_process_census(tmp_path)

    assert out == {
        "ok": True,
        "total": 42,
        "by_role": {"codex_cli": 2, "factory_droid": 3},
    }


def test_detect_agent_process_census_returns_empty_when_script_missing(tmp_path: Path) -> None:
    assert detector.detect_agent_process_census(tmp_path) == {}


def test_fetch_open_prs_returns_empty_when_gh_missing() -> None:
    with patch.object(detector.shutil, "which", return_value=None):
        assert detector.fetch_open_prs() == []


def test_fetch_open_prs_parses_gh_output() -> None:
    class _Proc:
        returncode = 0
        stdout = json.dumps(
            [
                {
                    "number": 7257,
                    "title": "feat: observer truth on FastAPI",
                    "headRefName": "refs/heads/droid/phase1",
                    "author": {"login": "an0mium"},
                    "isDraft": True,
                    "createdAt": "2026-05-17T00:00:00Z",
                    "updatedAt": "2026-05-17T01:00:00Z",
                    "url": "https://github.com/synaptent/aragora/pull/7257",
                }
            ]
        )
        stderr = ""

    with (
        patch.object(detector.shutil, "which", return_value="/usr/bin/gh"),
        patch.object(detector.subprocess, "run", return_value=_Proc()),
    ):
        out = detector.fetch_open_prs(limit=5)
    assert len(out) == 1
    assert out[0]["number"] == 7257
    assert out[0]["branch"] == "droid/phase1"
    assert out[0]["author"] == "an0mium"
    assert out[0]["is_draft"] is True


def test_build_overlap_report_flags_cross_source_branches() -> None:
    report = detector.build_overlap_report(
        worktrees=[
            {"path": "/tmp/wt-a", "branch": "feature/x"},
            {"path": "/tmp/wt-b", "branch": "feature/y"},
        ],
        dispatch_contracts=[{"branch": "feature/x", "name": "issue-1.json"}],
        issue_claims=[{"issue_number": 7257, "name": "claim-7257.json"}],
        automation_outbox=[{"branch": "feature/z", "idempotency_key": "k"}],
        codex_cli_sessions=[
            {
                "relative_path": "2026/05/17/session.jsonl",
                "branch": "feature/x",
                "cwd": "/tmp/wt-b",
            }
        ],
        open_prs=[
            {"branch": "feature/x", "number": 9001},
            {"branch": "feature/y", "number": 9002},
            {"branch": "feature/z", "number": 9003},
        ],
    )

    overlaps_by_value = {ov["value"]: ov for ov in report["overlaps"]}
    # feature/x appears in git_worktree + dispatch_contract + open_pr
    assert "feature/x" in overlaps_by_value
    assert set(overlaps_by_value["feature/x"]["sources"]) == {
        "codex_cli_session",
        "git_worktree",
        "dispatch_contract",
        "open_pr",
    }
    # feature/y in git_worktree + open_pr
    assert set(overlaps_by_value["feature/y"]["sources"]) == {
        "git_worktree",
        "open_pr",
    }
    # feature/z in automation_outbox + open_pr
    assert set(overlaps_by_value["feature/z"]["sources"]) == {
        "automation_outbox",
        "open_pr",
    }
    assert set(overlaps_by_value["/tmp/wt-b"]["sources"]) == {
        "codex_cli_session",
        "git_worktree",
    }
    # issue 7257 alone (issue_claim only) is not an overlap
    assert "7257" not in overlaps_by_value
    assert report["overlap_count"] == 4


def test_build_overlap_report_handles_empty_inputs() -> None:
    report = detector.build_overlap_report(
        worktrees=[],
        dispatch_contracts=[],
        issue_claims=[],
        automation_outbox=[],
        codex_cli_sessions=[],
        open_prs=[],
    )
    assert report == {"counts": {}, "overlaps": [], "overlap_count": 0}


def test_build_payload_assembles_top_level_keys(tmp_path: Path) -> None:
    payload = detector.build_payload(
        repo_root=tmp_path,
        codex_home=tmp_path / "codex",
        now=_ts("2026-05-17T12:00:00Z"),
        max_age_minutes=120.0,
        skip_gh=True,
        skip_codex_desktop=True,
        skip_process_census=True,
    )
    expected_keys = {
        "schema_version",
        "generated_at",
        "repo_root",
        "codex_home",
        "max_age_minutes",
        "codex_session_scan_limit",
        "skip_gh",
        "skip_codex_desktop",
        "skip_process_census",
        "worktrees",
        "dispatch_contracts",
        "issue_claims",
        "work_leases",
        "automation_outbox",
        "fleet_coordination",
        "codex_desktop_automations",
        "codex_cli_sessions",
        "process_census",
        "open_prs",
        "overlap_report",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["schema_version"] == detector.SCHEMA_VERSION
    assert payload["generated_at"] == "2026-05-17T12:00:00Z"
    assert payload["skip_gh"] is True
    assert payload["skip_process_census"] is True
    assert payload["overlap_report"]["overlap_count"] == 0


def test_render_text_contains_expected_sections() -> None:
    payload: dict[str, Any] = {
        "generated_at": "2026-05-17T12:00:00Z",
        "repo_root": "/repo",
        "codex_home": "/home/u/.codex",
        "worktrees": [
            {"path": "/repo", "branch": "main"},
            {"path": "/tmp/wt-a", "branch": "feature/x", "locked": True},
        ],
        "dispatch_contracts": [
            {"name": "issue-7257.json", "issue_number": 7257, "branch": "droid/x", "age_minutes": 5}
        ],
        "issue_claims": [],
        "work_leases": [],
        "automation_outbox": [],
        "codex_cli_sessions": [],
        "codex_desktop_automations": {
            "core_writers": {"engineering-autopilot": {"status": "PAUSED"}}
        },
        "process_census": {
            "ok": True,
            "total": 5,
            "by_role": {"codex_cli": 1, "factory_droid": 4},
        },
        "open_prs": [
            {
                "number": 7257,
                "title": "feat",
                "branch": "droid/x",
                "is_draft": True,
                "author": "an0mium",
            }
        ],
        "overlap_report": {
            "overlap_count": 1,
            "overlaps": [
                {"kind": "branch", "value": "droid/x", "sources": ["dispatch_contract", "open_pr"]}
            ],
        },
    }

    out = detector.render_text(payload)
    assert "git worktrees (2)" in out
    assert "[LOCKED]" in out
    assert "dispatch contracts (recent within max_age, 1)" in out
    assert "engineering-autopilot: status=PAUSED" in out
    assert "agent process census: ok=True total=5" in out
    assert "factory_droid: 4" in out
    assert "open PRs (1)" in out
    assert "overlap report (count=1)" in out
    assert "branch=droid/x" in out


def test_main_json_mode_produces_parseable_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = detector.main(
        [
            "--repo-root",
            str(tmp_path),
            "--codex-home",
            str(tmp_path / "codex"),
            "--skip-gh",
            "--skip-codex-desktop",
            "--skip-process-census",
            "--max-age-minutes",
            "5",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["schema_version"] == detector.SCHEMA_VERSION
    assert payload["skip_gh"] is True
    assert payload["skip_codex_desktop"] is True
    assert payload["skip_process_census"] is True


def test_main_text_mode_prints_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = detector.main(
        [
            "--repo-root",
            str(tmp_path),
            "--codex-home",
            str(tmp_path / "codex"),
            "--skip-gh",
            "--skip-codex-desktop",
            "--skip-process-census",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("Active agent sessions")
    assert "overlap report" in out
