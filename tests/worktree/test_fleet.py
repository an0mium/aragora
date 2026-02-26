"""Tests for shared fleet coordination utilities."""

from __future__ import annotations

from pathlib import Path

from aragora.worktree.fleet import (
    FleetCoordinationStore,
    infer_orchestration_pattern,
)


def test_infer_orchestration_pattern_from_framework() -> None:
    pattern = infer_orchestration_pattern({"framework": "CrewAI"})
    assert pattern == "crewai"


def test_infer_orchestration_pattern_from_command() -> None:
    pattern = infer_orchestration_pattern({"command": "python scripts/gastown_migrate_state.py"})
    assert pattern == "gastown"


def test_claim_paths_detects_conflicts(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    first = store.claim_paths(
        session_id="session-a",
        paths=["aragora/server/handlers/a.py"],
        mode="exclusive",
    )
    assert first["conflicts"] == []

    second = store.claim_paths(
        session_id="session-b",
        paths=["aragora/server/handlers/a.py"],
        mode="exclusive",
    )
    assert len(second["conflicts"]) == 1
    assert second["conflicts"][0]["session_id"] == "session-a"


def test_release_paths_by_subset(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.claim_paths(session_id="session-a", paths=["a.py", "b.py"])
    result = store.release_paths(session_id="session-a", paths=["a.py"])
    assert result["released"] == 1
    claims = store.list_claims()
    assert len(claims) == 1
    assert claims[0]["path"] == "b.py"


def test_enqueue_merge_deduplicates_active_branch(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    first = store.enqueue_merge(session_id="session-a", branch="codex/session-a", priority=70)
    assert first["queued"] is True
    second = store.enqueue_merge(session_id="session-b", branch="codex/session-a", priority=80)
    assert second["duplicate"] is True
    queue = store.list_merge_queue()
    assert len(queue) == 1
