from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import render_worktree_harvest_unblock_map as unblock_map  # noqa: E402


def _candidate(
    *,
    path: str,
    classification: str,
    size_bytes: int,
    cleanup_candidate: bool = False,
    dirty: bool = False,
    active_session: bool = False,
    lock_files: list[str] | None = None,
    ahead: int = 0,
    open_prs: list[dict[str, object]] | None = None,
    outbox_files: list[str] | None = None,
    receipt_files: list[str] | None = None,
) -> dict[str, object]:
    return {
        "path": path,
        "repo_path": f"{path}/aragora",
        "classification": classification,
        "decision": "cleanup_candidate" if cleanup_candidate else "preserve",
        "cleanup_candidate": cleanup_candidate,
        "size_bytes": size_bytes,
        "active_session": active_session,
        "lock_files": lock_files or [],
        "proof": [classification],
        "git": {
            "branch": "codex/demo",
            "head": "abc123",
            "ahead": ahead,
            "behind": 0,
            "dirty": dirty,
            "lookup_failed": classification == "lookup_failed",
        },
        "links": {
            "open_prs": open_prs or [],
            "outbox_files": outbox_files or [],
            "receipt_files": receipt_files or [],
        },
    }


def _inventory() -> dict[str, object]:
    return {
        "schema": "aragora-worktree-harvest/1.0",
        "generated_at": "2026-05-16T23:31:28Z",
        "root": "/Users/armand/.codex/worktrees",
        "summary": {
            "total_candidates": 6,
            "cleanup_candidate_count": 2,
            "harvest_candidate_count": 1,
            "known_size_bytes": 21 * 1024 * 1024,
        },
        "candidates": [
            _candidate(
                path="/Users/armand/.codex/worktrees/dirty",
                classification="active_or_dirty",
                size_bytes=6 * 1024 * 1024,
                dirty=True,
            ),
            _candidate(
                path="/Users/armand/.codex/worktrees/lock",
                classification="active_or_dirty",
                size_bytes=5 * 1024 * 1024,
                active_session=True,
                lock_files=["/Users/armand/.codex/worktrees/lock/.codex_session_active"],
            ),
            _candidate(
                path="/Users/armand/.codex/worktrees/pr",
                classification="open_pr_or_outbox",
                size_bytes=4 * 1024 * 1024,
                open_prs=[{"number": 123, "url": "https://example.test/pr/123"}],
            ),
            _candidate(
                path="/Users/armand/.codex/worktrees/receipt",
                classification="receipt_protected",
                size_bytes=3 * 1024 * 1024,
                receipt_files=[".aragora/automation-receipts/demo.json"],
            ),
            _candidate(
                path="/Users/armand/.codex/worktrees/clean",
                classification="patch_equivalent_or_merged",
                size_bytes=2 * 1024 * 1024,
                cleanup_candidate=True,
            ),
            _candidate(
                path="/Users/armand/.codex/worktrees/unique",
                classification="unique_unharvested",
                size_bytes=1 * 1024 * 1024,
                ahead=1,
            ),
        ],
    }


def test_render_unblock_map_groups_blockers_and_candidates() -> None:
    result = unblock_map.render_unblock_map(_inventory(), limit=10)

    families = {item["family"]: item for item in result["blocker_families"]}
    assert families["dirty uncommitted changes"]["count"] == 1
    assert families["active session lock"]["count"] == 1
    assert families["open PR protected"]["count"] == 1
    assert families["receipt/outbox protected"]["count"] == 1
    assert families["cleanup candidate"]["count"] == 1
    assert families["unique commits not on origin/main"]["count"] == 1

    assert result["top_cleanup_candidates"][0]["path"] == "/Users/armand/.codex/worktrees/clean"
    assert result["top_human_review_candidates"][0]["path"] == (
        "/Users/armand/.codex/worktrees/dirty"
    )
    assert result["summary"]["cleanup_candidate_count"] == 2
    assert result["summary"]["harvest_candidate_count"] == 1


def test_rendered_next_commands_are_read_only_guidance() -> None:
    result = unblock_map.render_unblock_map(_inventory(), limit=10)
    rows = result["top_cleanup_candidates"] + result["top_human_review_candidates"]
    assert rows
    for row in rows:
        command = row["next_command"]
        assert command.startswith("DO NOT RUN: ")
        assert "/Users/armand/Development/aragora" not in command
        assert " rm " not in f" {command} "
        assert "git branch -D" not in command
        assert "git push --force" not in command


def test_main_writes_json_output(tmp_path: Path) -> None:
    input_path = tmp_path / "inventory.json"
    output_path = tmp_path / "unblock-map.json"
    input_path.write_text(json.dumps(_inventory()), encoding="utf-8")

    assert unblock_map.main(["--input", str(input_path), "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source_inventory"]["generated_at"] == "2026-05-16T23:31:28Z"
    assert payload["top_cleanup_candidates"]
