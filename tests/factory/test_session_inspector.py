from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from aragora.factory import session_inspector as inspector


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _repo_with_lanes(tmp_path: Path, lanes: list[dict[str, object]]) -> Path:
    repo = tmp_path / "repo"
    registry = repo / ".aragora" / "agent-bridge" / "lanes.json"
    _write_json(registry, lanes)
    return repo


def _factory_home_with_index(tmp_path: Path, sessions: list[dict[str, object]]) -> Path:
    home = tmp_path / "factory"
    _write_json(home / "sessions-index.json", sessions)
    return home


def test_brief_redacts_index_metadata_and_matches_active_lane(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    factory_home = _factory_home_with_index(
        tmp_path,
        [
            {
                "id": "droid-ABC123",
                "cwd": str(worktree),
                "branch": "droid/P16-stage2",
                "head": "a" * 40,
                "pr_number": 7292,
                "updated_at": int(time.time()),
                "title": "repair #7292 with sk-proj-FAKE-FACTORY-SECRET",
            }
        ],
    )
    repo = _repo_with_lanes(
        tmp_path,
        [
            {
                "lane_id": "P16-stage2",
                "owner_session": "droid-ABC123",
                "status": "active",
                "pr_number": 7292,
                "branch": "droid/P16-stage2",
                "worktree": str(worktree),
                "updated_at": "2026-05-19T12:00:00Z",
            }
        ],
    )

    briefs = inspector.build_factory_session_briefs(
        factory_home=factory_home,
        repo_root=repo,
        since=timedelta(hours=4),
    )
    payload = [brief.to_dict() for brief in briefs]
    serialized = json.dumps(payload)

    assert len(payload) == 1
    row = payload[0]
    assert row["provider"] == "factory"
    assert row["session_id"] == "droid-ABC123"
    assert row["pr_number"] == 7292
    assert row["branch"] == "droid/P16-stage2"
    assert row["head"] == "a" * 40
    assert row["matched_lane"]["lane_id"] == "P16-stage2"
    assert row["prompt_needed"] is False
    assert row["prompt_needed_reason"] == "active_lane_owned"
    assert "sk-proj-FAKE-FACTORY-SECRET" not in serialized
    assert "repair #7292" not in serialized


def test_default_inspector_does_not_read_history_or_raw_session_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory_home = _factory_home_with_index(
        tmp_path,
        [
            {
                "id": "droid-RAW",
                "cwd": str(tmp_path / "worktree"),
                "updated_at": int(time.time()),
            }
        ],
    )
    forbidden_paths = {
        factory_home / "history.json",
        factory_home / "sessions" / "droid-RAW" / "transcript.jsonl",
        factory_home / "logs" / "droid-log-single.log",
    }
    for path in forbidden_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("RAW_FACTORY_SECRET_SHOULD_NOT_BE_READ", encoding="utf-8")
    original_read_text: Any = Path.read_text

    def guarded_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self in forbidden_paths:
            raise AssertionError(f"raw Factory file was read: {self}")
        return str(original_read_text(self, *args, **kwargs))

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    briefs = inspector.build_factory_session_briefs(
        factory_home=factory_home,
        repo_root=_repo_with_lanes(tmp_path, []),
        since=timedelta(hours=4),
    )
    serialized = json.dumps([brief.to_dict() for brief in briefs])

    assert len(briefs) == 1
    assert "RAW_FACTORY_SECRET_SHOULD_NOT_BE_READ" not in serialized


def test_duplicate_active_pr_owner_reports_conflict_and_steering(tmp_path: Path) -> None:
    factory_home = _factory_home_with_index(
        tmp_path,
        [
            {
                "id": "droid-DUPLICATE",
                "cwd": str(tmp_path / "duplicate"),
                "branch": "droid/duplicate",
                "pr_number": 7292,
                "updated_at": int(time.time()),
            }
        ],
    )
    repo = _repo_with_lanes(
        tmp_path,
        [
            {
                "lane_id": "Q01-owner",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7292,
                "branch": "droid/current-owner",
                "updated_at": "2026-05-19T12:00:00Z",
            }
        ],
    )

    brief = inspector.build_factory_session_briefs(
        factory_home=factory_home,
        repo_root=repo,
        since=timedelta(hours=4),
    )[0]
    payload = brief.to_dict()

    assert payload["conflict_risk"] == "active-owner-overlap"
    assert payload["direct_steering_available"] is False
    assert payload["matched_lane"]["owner_session"] == "codex-owner"
    assert "codex-owner" in payload["router"]["recommended_next_prompt"]
    assert "Do not edit" in payload["router"]["recommended_next_prompt"]


def test_direct_steering_does_not_use_shared_cwd_without_session_or_lane_match(
    tmp_path: Path,
) -> None:
    shared_cwd = tmp_path / "repo"
    shared_cwd.mkdir()
    factory_home = _factory_home_with_index(
        tmp_path,
        [
            {
                "id": "factory-uuid-session",
                "cwd": str(shared_cwd),
                "updated_at": int(time.time()),
            }
        ],
    )
    repo = _repo_with_lanes(tmp_path, [])
    _write_json(
        repo / ".aragora" / "tmux-sessions" / "factory-review.meta.json",
        {
            "name": "factory-review",
            "agent": "droid",
            "cwd": str(shared_cwd),
            "tmux_window_target": "aragora:factory-review",
        },
    )

    brief = inspector.build_factory_session_briefs(
        factory_home=factory_home,
        repo_root=repo,
        since=timedelta(hours=4),
    )[0]

    assert brief.direct_steering_available is False
    assert brief.steering_command is None


def test_direct_steering_ignores_secret_like_tmux_target_names(tmp_path: Path) -> None:
    factory_home = _factory_home_with_index(
        tmp_path,
        [
            {
                "id": "factory-visible",
                "pr_number": 7359,
                "updated_at": int(time.time()),
            }
        ],
    )
    secret_target = "factory-ghp_FAKELEAK12345678901234"
    repo = _repo_with_lanes(
        tmp_path,
        [
            {
                "lane_id": "P80-owner",
                "owner_session": secret_target,
                "status": "active",
                "pr_number": 7359,
                "updated_at": "2026-05-19T12:00:00Z",
            }
        ],
    )
    _write_json(
        repo / ".aragora" / "tmux-sessions" / "factory-secret.meta.json",
        {
            "name": secret_target,
            "agent": "factory",
        },
    )

    brief = inspector.build_factory_session_briefs(
        factory_home=factory_home,
        repo_root=repo,
        since=timedelta(hours=4),
    )[0]
    payload = brief.to_dict()
    serialized = json.dumps(payload)

    assert payload["matched_lane"]["owner_session"] == "factory-[REDACTED]"
    assert payload["direct_steering_available"] is False
    assert payload["steering_command"] is None
    assert secret_target not in serialized


def test_direct_steering_ignores_path_like_tmux_target_names(tmp_path: Path) -> None:
    factory_home = _factory_home_with_index(
        tmp_path,
        [
            {
                "id": "factory-path-target",
                "updated_at": int(time.time()),
            }
        ],
    )
    repo = _repo_with_lanes(tmp_path, [])
    _write_json(
        repo / ".aragora" / "tmux-sessions" / "factory-path.meta.json",
        {
            "name": "factory/path-target",
            "agent": "droid",
        },
    )

    brief = inspector.build_factory_session_briefs(
        factory_home=factory_home,
        repo_root=repo,
        since=timedelta(hours=4),
    )[0]

    assert brief.direct_steering_available is False
    assert brief.steering_command is None


def test_unknown_session_returns_paste_needed(tmp_path: Path) -> None:
    factory_home = _factory_home_with_index(tmp_path, [])

    brief = inspector.paste_needed_brief("missing-session")
    payload = brief.to_dict()

    assert payload["session_id"] == "missing-session"
    assert payload["router"]["category"] == "paste-needed"
    assert "Paste the relevant Factory excerpt" in payload["router"]["recommended_next_prompt"]
    assert (
        inspector.build_factory_session_briefs(
            factory_home=factory_home,
            repo_root=_repo_with_lanes(tmp_path, []),
            since=timedelta(hours=4),
            session="missing-session",
        )[0].router.category
        == "paste-needed"
    )
