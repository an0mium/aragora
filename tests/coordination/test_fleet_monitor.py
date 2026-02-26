"""Tests for fleet coordination monitor and merge queue enforcement."""

from __future__ import annotations

from pathlib import Path

from aragora.coordination.fleet import FleetCoordinator


def test_claim_paths_blocks_conflict_without_override(tmp_path: Path) -> None:
    fleet = FleetCoordinator(repo_root=tmp_path)

    first = fleet.claim_paths("track:core", ["aragora/server/handlers/coordination.py"])
    second = fleet.claim_paths("track:qa", ["aragora/server/handlers/coordination.py"])

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["applied"] is False
    assert second["conflicts"][0]["owner"] == "track:core"


def test_claim_paths_override_allows_conflict(tmp_path: Path) -> None:
    fleet = FleetCoordinator(repo_root=tmp_path)

    fleet.claim_paths("track:core", ["aragora/cli/commands/worktree.py"])
    result = fleet.claim_paths(
        "track:qa",
        ["aragora/cli/commands/worktree.py"],
        override=True,
    )

    assert result["ok"] is True
    assert result["override"] is True
    claims = fleet.get_claims()
    assert claims["conflict_count"] == 1


def test_merge_queue_advance_blocked_on_red_checks(monkeypatch, tmp_path: Path) -> None:
    fleet = FleetCoordinator(repo_root=tmp_path)
    enqueue = fleet.enqueue_merge(owner="track:core", branch="codex/core-123")
    assert enqueue["ok"] is True

    monkeypatch.setattr(
        fleet,
        "_collect_sessions",
        lambda **_: [
            {
                "owner": "track:core",
                "session_id": "core",
                "branch": "codex/core-123",
                "required_checks_state": "red",
            }
        ],
    )
    monkeypatch.setattr(
        fleet,
        "get_claims",
        lambda: {
            "generated_at": "",
            "claims": [],
            "conflicts": [],
            "conflict_count": 0,
        },
    )

    result = fleet.advance_merge_queue()

    assert result["ok"] is True
    assert result["advanced"] is False
    assert result["reason"] == "blocked"
    assert "required_checks_red" in result["blockers"]


def test_merge_queue_advance_ready_when_green(monkeypatch, tmp_path: Path) -> None:
    fleet = FleetCoordinator(repo_root=tmp_path)
    enqueue = fleet.enqueue_merge(owner="track:core", branch="codex/core-123")
    assert enqueue["ok"] is True

    monkeypatch.setattr(
        fleet,
        "_collect_sessions",
        lambda **_: [
            {
                "owner": "track:core",
                "session_id": "core",
                "branch": "codex/core-123",
                "required_checks_state": "green",
            }
        ],
    )
    monkeypatch.setattr(
        fleet,
        "get_claims",
        lambda: {
            "generated_at": "",
            "claims": [],
            "conflicts": [],
            "conflict_count": 0,
        },
    )

    result = fleet.advance_merge_queue()

    assert result["ok"] is True
    assert result["advanced"] is True
    assert result["item"]["status"] == "ready"


def test_fleet_status_collects_drift_and_log_tail(monkeypatch, tmp_path: Path) -> None:
    wt = tmp_path / "wt-core"
    wt.mkdir(parents=True)
    (wt / ".sprint-agent.log").write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    fleet = FleetCoordinator(repo_root=tmp_path)

    monkeypatch.setattr(
        fleet,
        "_manifest_entries",
        lambda: [
            {
                "track": "core",
                "worktree": str(wt),
                "agent": "codex",
                "goal": "stabilize fleet monitor",
                "files_claimed": ["aragora/server/handlers/coordination.py"],
                "status": "active",
                "pid": 1234,
                "started_at": "2026-02-26T00:00:00+00:00",
            }
        ],
    )
    monkeypatch.setattr(fleet, "_autopilot_entries", lambda: [])
    monkeypatch.setattr(
        fleet,
        "_git_worktree_map",
        lambda: {str(wt.resolve()): {"branch": "codex/core-123", "detached": False}},
    )
    monkeypatch.setattr(fleet, "_git_head_sha", lambda _: "abc111")
    monkeypatch.setattr(fleet, "_git_remote_sha", lambda _: "def222")
    monkeypatch.setattr(
        fleet,
        "_resolve_pr_status",
        lambda _: {
            "pr": {"number": 365, "url": "https://example/pr/365", "title": "Fleet"},
            "required_checks_state": "green",
            "required_checks": [],
            "failing_checks": [],
        },
    )

    payload = fleet.fleet_status(tail_lines=2)

    assert payload["total_sessions"] == 1
    assert payload["active_sessions"] == 1
    session = payload["sessions"][0]
    assert session["branch"] == "codex/core-123"
    assert session["drift"] is True
    assert session["log_tail"] == ["line-2", "line-3"]
    assert session["claimed_paths"] == ["aragora/server/handlers/coordination.py"]


def test_write_report_creates_markdown(tmp_path: Path) -> None:
    fleet = FleetCoordinator(repo_root=tmp_path)

    status = {
        "generated_at": "2026-02-26T12:00:00+00:00",
        "active_sessions": 1,
        "total_sessions": 1,
        "sessions": [
            {
                "owner": "track:core",
                "branch": "codex/core-123",
                "active": True,
                "required_checks_state": "green",
                "drift": False,
            }
        ],
        "claims": {"conflict_count": 0},
        "merge_queue": {"total": 0, "items": []},
        "actionable_failures": [],
    }

    out_dir = tmp_path / "docs" / "status"
    result = fleet.write_report(status=status, output_dir=out_dir)

    report_path = Path(result["report_path"])
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Fleet Status Report" in text
    assert "Actionable Failures" in text
