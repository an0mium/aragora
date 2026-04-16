"""Unit tests for scripts/codex_worktree_autopilot.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_parse_worktree_porcelain_includes_branch_and_detached():
    import codex_worktree_autopilot as mod

    porcelain = (
        "worktree /repo\n"
        "HEAD abc\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /repo/.worktrees/codex-auto/s1\n"
        "HEAD def\n"
        "branch refs/heads/codex/s1\n"
        "\n"
        "worktree /repo/.worktrees/codex-auto/s2\n"
        "HEAD 123\n"
        "detached\n"
        "\n"
    )
    entries = mod._parse_worktree_porcelain(porcelain)
    assert len(entries) == 3
    assert entries[0].branch == "main"
    assert entries[1].branch == "codex/s1"
    assert entries[2].detached is True
    assert entries[2].branch is None


def test_prune_stale_state_keeps_existing_orphan_paths_and_removes_missing(tmp_path):
    import codex_worktree_autopilot as mod

    existing_path = tmp_path / "existing"
    missing_path = tmp_path / "missing"
    existing_path.mkdir()

    state = {
        "sessions": [
            {"session_id": "existing", "path": str(existing_path)},
            {"session_id": "missing", "path": str(missing_path)},
        ]
    }
    active: set[str] = set()
    pruned, removed = mod._prune_stale_state(state, active)
    assert removed == 1
    assert len(pruned["sessions"]) == 1
    assert pruned["sessions"][0]["session_id"] == "existing"


def test_choose_reusable_session_prefers_latest_last_seen():
    import codex_worktree_autopilot as mod

    state = {
        "sessions": [
            {
                "agent": "codex",
                "session_id": "old",
                "path": "/repo/.worktrees/codex-auto/old",
                "last_seen_at": "2026-02-24T00:00:00+00:00",
            },
            {
                "agent": "codex",
                "session_id": "new",
                "path": "/repo/.worktrees/codex-auto/new",
                "last_seen_at": "2026-02-24T01:00:00+00:00",
            },
        ]
    }
    chosen = mod._choose_reusable_session(
        state,
        agent="codex",
        session_id=None,
        active_paths={
            "/repo/.worktrees/codex-auto/old",
            "/repo/.worktrees/codex-auto/new",
        },
    )
    assert chosen is not None
    assert chosen["session_id"] == "new"


def test_choose_reusable_session_honors_session_id_filter():
    import codex_worktree_autopilot as mod

    state = {
        "sessions": [
            {"agent": "codex", "session_id": "a", "path": "/repo/.worktrees/codex-auto/a"},
            {"agent": "codex", "session_id": "b", "path": "/repo/.worktrees/codex-auto/b"},
        ]
    }
    chosen = mod._choose_reusable_session(
        state,
        agent="codex",
        session_id="a",
        active_paths={
            "/repo/.worktrees/codex-auto/a",
            "/repo/.worktrees/codex-auto/b",
        },
    )
    assert chosen is not None
    assert chosen["session_id"] == "a"


def test_choose_reusable_session_skips_branch_mismatches():
    import codex_worktree_autopilot as mod

    state = {
        "sessions": [
            {
                "agent": "codex",
                "session_id": "s1",
                "branch": "codex/s1",
                "path": "/repo/.worktrees/codex-auto/s1",
            }
        ]
    }
    chosen = mod._choose_reusable_session(
        state,
        agent="codex",
        session_id="s1",
        active_paths={"/repo/.worktrees/codex-auto/s1"},
        active_branches_by_path={"/repo/.worktrees/codex-auto/s1": "codex/unrelated"},
    )
    assert chosen is None


def test_cleanup_parser_defaults_to_delete_branches():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["cleanup"])
    assert args.delete_branches is True


def test_ensure_parser_defaults_to_ff_only_strategy():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["ensure"])
    assert args.strategy == "ff-only"


def test_reconcile_parser_defaults_to_ff_only_strategy():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["reconcile"])
    assert args.strategy == "ff-only"


def test_cleanup_parser_allows_no_delete_branches_toggle():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["cleanup", "--no-delete-branches"])
    assert args.delete_branches is False


def test_maintain_parser_allows_no_delete_branches_toggle():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["maintain", "--no-delete-branches"])
    assert args.delete_branches is False


def test_maintain_parser_defaults_to_ff_only_strategy():
    import codex_worktree_autopilot as mod

    parser = mod._build_parser()
    args = parser.parse_args(["maintain"])
    assert args.strategy == "ff-only"


def test_parse_ts_normalizes_naive_timestamp_to_utc():
    import codex_worktree_autopilot as mod

    parsed = mod._parse_ts("2026-02-24T12:00:00")
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc


def test_create_managed_worktree_reuses_unattached_existing_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    managed_root = repo_root / ".worktrees" / "codex-auto"
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(mod, "_ensure_fetched", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_branch_exists", lambda _repo, branch: branch == "codex/swarm-123")
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=repo_root, branch="main")],
    )
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")

    def _run_git(
        _repo_root: Path, *args: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args=("git", *args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run_git", _run_git)

    session = mod._create_managed_worktree(
        repo_root,
        managed_root,
        agent="codex",
        base="main",
        session_id="swarm-123",
    )

    assert session["branch"] == "codex/swarm-123"
    assert any(
        call[:4]
        == ("worktree", "add", str((managed_root / "swarm-123").resolve()), "codex/swarm-123")
        for call in calls
    )
    assert not any(call[:3] == ("worktree", "add", "-b") for call in calls)


def test_create_managed_worktree_retries_add_time_branch_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    managed_root = repo_root / ".worktrees" / "codex-auto"
    calls: list[tuple[str, ...]] = []
    uuid_values = iter(["race0001", "b16b00b5", "cafe1234"])

    monkeypatch.setattr(mod, "_ensure_fetched", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_branch_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=repo_root, branch="main")],
    )
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(
        mod,
        "uuid4",
        lambda: type("FakeUUID", (), {"hex": next(uuid_values)})(),
    )

    def _run_git(
        _repo_root: Path, *args: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:3] == ("worktree", "add", "-b") and args[3] == "codex/swarm-race":
            return subprocess.CompletedProcess(
                args=("git", *args),
                returncode=128,
                stdout="",
                stderr="fatal: a branch named 'codex/swarm-race' already exists",
            )
        return subprocess.CompletedProcess(args=("git", *args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_run_git", _run_git)

    session = mod._create_managed_worktree(
        repo_root,
        managed_root,
        agent="codex",
        base="main",
        session_id="swarm-race",
    )

    assert session["branch"] == "codex/swarm-race-b16b"
    assert (
        "worktree",
        "add",
        "-b",
        "codex/swarm-race",
        str((managed_root / "swarm-race").resolve()),
        "origin/main",
    ) in calls
    assert (
        "worktree",
        "add",
        "-b",
        "codex/swarm-race-b16b",
        str((managed_root / "swarm-race").resolve()),
        "origin/main",
    ) in calls


def test_create_managed_worktree_reuses_existing_unattached_retry_branch_for_same_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    managed_root = repo_root / ".worktrees" / "codex-auto"
    attached_path = tmp_path / "attached"
    attached_path.mkdir()
    calls: list[tuple[str, ...]] = []
    now = datetime(2026, 4, 4, 5, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(mod, "_ensure_fetched", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [
            mod.WorktreeEntry(path=repo_root, branch="main"),
            mod.WorktreeEntry(path=attached_path, branch="codex/swarm-race"),
        ],
    )
    monkeypatch.setattr(
        mod,
        "_local_branches_with_prefix",
        lambda _repo, prefix: [prefix, f"{prefix}-b16b"],
    )
    monkeypatch.setattr(mod, "_branch_exists", lambda *_args, **_kwargs: True)

    def _run_git(
        _repo_root: Path, *args: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:2] == ("worktree", "add") and args[2] != "-b":
            assert args[3] == "codex/swarm-race-b16b"
            return subprocess.CompletedProcess(
                args=("git", *args), returncode=0, stdout="", stderr=""
            )
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(mod, "_run_git", _run_git)

    session = mod._create_managed_worktree(
        repo_root,
        managed_root,
        agent="codex",
        base="main",
        session_id="swarm-race",
    )

    assert session["branch"] == "codex/swarm-race-b16b"
    assert (
        "worktree",
        "add",
        str((managed_root / "swarm-race").resolve()),
        "codex/swarm-race-b16b",
    ) in calls


def test_create_managed_worktree_clears_stale_initializing_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    managed_root = repo_root / ".worktrees" / "codex-auto"
    stale_admin = repo_root / ".git" / "worktrees" / "swarm-stale"
    stale_admin.mkdir(parents=True)
    (stale_admin / "locked").write_text("initializing", encoding="utf-8")
    (stale_admin / "gitdir").write_text(
        str((managed_root / "swarm-stale" / ".git").resolve()),
        encoding="utf-8",
    )
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(mod, "_ensure_fetched", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_git_common_dir", lambda _repo: repo_root / ".git")
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=repo_root, branch="main")],
    )
    monkeypatch.setattr(mod, "_local_branches_with_prefix", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mod, "_branch_exists", lambda *_args, **_kwargs: False)

    def _run_git(
        _repo_root: Path, *args: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:2] == ("worktree", "add"):
            assert not stale_admin.exists()
            return subprocess.CompletedProcess(
                args=("git", *args), returncode=0, stdout="", stderr=""
            )
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(mod, "_run_git", _run_git)

    session = mod._create_managed_worktree(
        repo_root,
        managed_root,
        agent="codex",
        base="main",
        session_id="swarm-stale",
    )

    assert session["branch"] == "codex/swarm-stale"
    assert not stale_admin.exists()
    assert (
        "worktree",
        "add",
        "-b",
        "codex/swarm-stale",
        str((managed_root / "swarm-stale").resolve()),
        "origin/main",
    ) in calls


def test_create_managed_worktree_cleans_leaked_branch_before_second_source_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    managed_root = repo_root / ".worktrees" / "codex-auto"
    existing_branches: set[str] = set()
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(mod, "_ensure_fetched", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=repo_root, branch="main")],
    )
    monkeypatch.setattr(mod, "_local_branches_with_prefix", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        mod,
        "_branch_exists",
        lambda _repo, branch: branch in existing_branches,
    )

    def _proc(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)

    def _run_git(
        _repo_root: Path, *args: str, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:2] == ("worktree", "add"):
            branch = args[3]
            source = args[5]
            if source == "origin/main":
                existing_branches.add(branch)
                return _proc(
                    128,
                    "fatal: '/repo/.worktrees/codex-auto/swarm-race' is a missing but already "
                    "registered worktree; use 'add -f' to override",
                )
            if source == "main":
                assert branch not in existing_branches
                return _proc(0)
        if args[:2] == ("branch", "-D"):
            existing_branches.discard(args[2])
            return _proc(0)
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(mod, "_run_git", _run_git)

    session = mod._create_managed_worktree(
        repo_root,
        managed_root,
        agent="codex",
        base="main",
        session_id="swarm-race",
    )

    assert session["branch"] == "codex/swarm-race"
    assert ("branch", "-D", "codex/swarm-race") in calls
    assert (
        "worktree",
        "add",
        "-b",
        "codex/swarm-race",
        str((managed_root / "swarm-race").resolve()),
        "main",
    ) in calls


def test_cmd_cleanup_keeps_session_when_worktree_remove_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    active_path = tmp_path / "active-wt"
    active_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "s1",
                "agent": "codex",
                "branch": "codex/s1",
                "path": str(active_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=active_path, branch="codex/s1")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_remove_worktree", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "_delete_branch", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "prune"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        ttl_hours=0,
        force_unmerged=False,
        delete_branches=True,
        json=True,
    )
    rc = mod.cmd_cleanup(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == 0
    assert payload["kept"] == 1
    assert payload["failed_worktree_removals"] == 1
    assert payload["failed_branch_deletions"] == 0
    assert len(saved_state["sessions"]) == 1


def test_cmd_cleanup_reports_failed_branch_deletions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    stale_path = tmp_path / "stale-wt"
    state = {
        "sessions": [
            {
                "session_id": "s2",
                "agent": "codex",
                "branch": "codex/s2",
                "path": str(stale_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: [])
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_delete_branch", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "prune"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        ttl_hours=24,
        force_unmerged=False,
        delete_branches=True,
        json=True,
    )
    rc = mod.cmd_cleanup(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == 1
    assert payload["kept"] == 0
    assert payload["failed_worktree_removals"] == 0
    assert payload["failed_branch_deletions"] == 1
    assert saved_state["sessions"] == []


def test_cmd_reconcile_preserves_last_seen_at_for_skipped_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    now = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    original_last_seen = "2026-03-28T12:30:00+00:00"
    state = {
        "sessions": [
            {
                "session_id": "s-grace",
                "agent": "codex",
                "branch": "codex/s-grace",
                "path": str(worktree),
                "created_at": "2026-03-28T10:00:00+00:00",
                "last_seen_at": original_last_seen,
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=worktree, branch="codex/s-grace")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda *_args, **_kwargs: {
            "lookup_failed": False,
            "has_live_lease": False,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_id": None,
            "lease_expires_at": None,
        },
    )
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_safe_worktree_dirty", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(
        mod,
        "_integrate_worktree",
        lambda *_args, **_kwargs: pytest.fail("grace sessions should not be integrated"),
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        strategy="ff-only",
        all=True,
        path=None,
        ttl_hours=24,
        json=True,
    )
    rc = mod.cmd_reconcile(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["status"] == "skipped_grace"
    assert saved_state["sessions"][0]["last_seen_at"] == original_last_seen


def test_cmd_reconcile_preserves_last_seen_at_for_safe_to_clean_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    original_last_seen = "2026-03-20T12:00:00+00:00"
    state = {
        "sessions": [
            {
                "session_id": "s-safe",
                "agent": "codex",
                "branch": "codex/s-safe",
                "path": str(worktree),
                "created_at": "2026-03-20T10:00:00+00:00",
                "last_seen_at": original_last_seen,
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=worktree, branch="codex/s-safe")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda *_args, **_kwargs: {
            "lookup_failed": False,
            "has_live_lease": False,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_id": None,
            "lease_expires_at": None,
        },
    )
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_safe_worktree_dirty", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "_integrate_worktree", lambda *_args, **_kwargs: (True, "up_to_date"))

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        strategy="ff-only",
        all=True,
        path=None,
        ttl_hours=24,
        json=True,
    )
    rc = mod.cmd_reconcile(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["status"] == "up_to_date"
    assert saved_state["sessions"][0]["last_seen_at"] == original_last_seen


def test_cmd_status_prunes_missing_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    missing_path = tmp_path / "missing-wt"
    state = {
        "sessions": [
            {
                "session_id": "s-missing",
                "agent": "codex",
                "branch": "codex/s-missing",
                "path": str(missing_path),
                "created_at": "2026-03-20T10:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: [])
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        ttl_hours=24,
        json=True,
    )
    rc = mod.cmd_status(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"] == []
    assert saved_state["sessions"] == []


def test_cmd_ensure_refreshes_active_paths_for_new_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    now = datetime(2026, 3, 24, 0, 0, tzinfo=timezone.utc)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    created_path = tmp_path / "managed" / "session-1"
    created_path.mkdir(parents=True)
    state = {"sessions": []}
    saved_state: dict[str, object] = {}

    entries_iter = iter(
        [
            [],
            [mod.WorktreeEntry(path=created_path, branch="codex/session-1")],
        ]
    )

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: next(entries_iter))
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(
        mod,
        "_create_managed_worktree",
        lambda *_args, **_kwargs: {
            "session_id": "session-1",
            "agent": "codex",
            "branch": "codex/session-1",
            "path": str(created_path),
            "base_branch": "main",
            "created_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
        },
    )
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
            "lookup_failed": False,
        },
    )
    monkeypatch.setattr(
        mod,
        "_worktree_status",
        lambda *_args, **_kwargs: {"dirty": False, "ahead": 0, "behind": 0},
    )
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        agent="codex",
        base="main",
        session_id=None,
        force_new=True,
        reconcile=True,
        strategy="ff-only",
        print_path=False,
        json=True,
    )
    rc = mod.cmd_ensure(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] is True
    assert payload["session"]["tracked_worktree"] is True
    assert payload["session"]["lifecycle_state"] == "grace"
    assert saved_state["sessions"][0]["tracked_worktree"] is True
    assert saved_state["sessions"][0]["lifecycle_state"] == "grace"


def test_cmd_ensure_replaces_ahead_reusable_session_with_fresh_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    now = datetime(2026, 4, 13, 16, 30, tzinfo=timezone.utc)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    drifted_path = tmp_path / "managed" / "swarm-5d9f0302-micro-2"
    created_path = tmp_path / "managed" / "swarm-clean"
    drifted_path.mkdir(parents=True)
    created_path.mkdir(parents=True)
    state = {
        "sessions": [
            {
                "session_id": "swarm-5d9f0302-micro-2",
                "agent": "codex",
                "branch": "codex/swarm-5d9f0302-micro-2",
                "path": str(drifted_path),
                "base_branch": "main",
                "created_at": now.isoformat(),
                "last_seen_at": now.isoformat(),
            }
        ]
    }
    saved_state: dict[str, object] = {}

    entries_iter = iter(
        [
            [mod.WorktreeEntry(path=drifted_path, branch="codex/swarm-5d9f0302-micro-2")],
            [
                mod.WorktreeEntry(path=drifted_path, branch="codex/swarm-5d9f0302-micro-2"),
                mod.WorktreeEntry(path=created_path, branch="codex/swarm-clean"),
            ],
        ]
    )

    def _worktree_status(_repo: Path, path: Path, _base: str) -> dict[str, int | bool]:
        if path == drifted_path:
            return {"dirty": False, "ahead": 1, "behind": 0}
        if path == created_path:
            return {"dirty": False, "ahead": 0, "behind": 0}
        raise AssertionError(f"unexpected worktree path: {path}")

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: next(entries_iter))
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(
        mod,
        "_create_managed_worktree",
        lambda *_args, **_kwargs: {
            "session_id": "swarm-clean",
            "agent": "codex",
            "branch": "codex/swarm-clean",
            "path": str(created_path),
            "base_branch": "main",
            "created_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
        },
    )
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
            "lookup_failed": False,
        },
    )
    monkeypatch.setattr(mod, "_worktree_status", _worktree_status)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        agent="codex",
        base="main",
        session_id=None,
        force_new=False,
        reconcile=True,
        strategy="ff-only",
        print_path=False,
        json=True,
    )
    rc = mod.cmd_ensure(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] is True
    assert payload["session"]["path"] == str(created_path)
    assert payload["session"]["branch"] == "codex/swarm-clean"
    assert saved_state["sessions"][0]["reconcile_status"] == "rejected_drifted_reusable_session"
    assert saved_state["sessions"][1]["path"] == str(created_path)


def test_cmd_ensure_preserves_explicit_session_id_when_replacing_drifted_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    now = datetime(2026, 4, 13, 16, 30, tzinfo=timezone.utc)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    drifted_path = tmp_path / "managed" / "session-1"
    created_path = tmp_path / "managed" / "session-1-replaced"
    drifted_path.mkdir(parents=True)
    created_path.mkdir(parents=True)
    state = {
        "sessions": [
            {
                "session_id": "session-1",
                "agent": "codex",
                "branch": "codex/session-1",
                "path": str(drifted_path),
                "base_branch": "main",
                "created_at": now.isoformat(),
                "last_seen_at": now.isoformat(),
            }
        ]
    }
    saved_state: dict[str, object] = {}
    create_calls: list[str | None] = []

    entries_iter = iter(
        [
            [mod.WorktreeEntry(path=drifted_path, branch="codex/session-1")],
            [
                mod.WorktreeEntry(path=drifted_path, branch="codex/session-1"),
                mod.WorktreeEntry(path=created_path, branch="codex/session-1"),
            ],
        ]
    )

    def _worktree_status(_repo: Path, path: Path, _base: str) -> dict[str, int | bool]:
        if path == drifted_path:
            return {"dirty": False, "ahead": 1, "behind": 0}
        if path == created_path:
            return {"dirty": False, "ahead": 0, "behind": 0}
        raise AssertionError(f"unexpected worktree path: {path}")

    def _create_managed_worktree(
        *_args: object,
        session_id: str | None,
        **_kwargs: object,
    ) -> dict[str, object]:
        create_calls.append(session_id)
        return {
            "session_id": session_id or "unexpected-generated-session",
            "agent": "codex",
            "branch": "codex/session-1",
            "path": str(created_path),
            "base_branch": "main",
            "created_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
        }

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: next(entries_iter))
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_create_managed_worktree", _create_managed_worktree)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
            "lookup_failed": False,
        },
    )
    monkeypatch.setattr(mod, "_worktree_status", _worktree_status)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        agent="codex",
        base="main",
        session_id="session-1",
        force_new=False,
        reconcile=True,
        strategy="ff-only",
        print_path=False,
        json=True,
    )
    rc = mod.cmd_ensure(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] is True
    assert payload["session"]["session_id"] == "session-1"
    assert payload["session"]["branch"] == "codex/session-1"
    assert create_calls == ["session-1"]
    assert len(saved_state["sessions"]) == 1
    assert saved_state["sessions"][0]["reconcile_status"] == "rejected_drifted_reusable_session"
    assert saved_state["sessions"][0]["session_id"] == "session-1"


def test_create_managed_worktree_retries_when_branch_is_created_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    managed_root = tmp_path / "managed"
    now = datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc)
    branches_seen: list[str] = []

    def _proc(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)

    def _run_git(_repo: Path, *args: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:1] == ("for-each-ref",):
            return _proc(0)
        if args[:2] == ("worktree", "add"):
            branch = args[3]
            branches_seen.append(branch)
            if len(branches_seen) == 1:
                return _proc(
                    128,
                    f"fatal: a branch named '{branch}' already exists",
                )
            return _proc(0)
        raise AssertionError(f"unexpected git args: {args}")

    class _UUID:
        def __init__(self, value: str) -> None:
            self.hex = value

    uuids = iter(
        [
            _UUID("sessiontokdeadbeef"),
            _UUID("retryabcd12345678"),
        ]
    )

    monkeypatch.setattr(mod, "_ensure_fetched", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_branch_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mod, "_run_git", _run_git)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "uuid4", lambda: next(uuids))

    session = mod._create_managed_worktree(
        repo_root,
        managed_root,
        agent="codex",
        base="main",
        session_id="swarm-ec71e047-subtask_1",
    )

    assert branches_seen[0] == "codex/swarm-ec71e047-subtask_1"
    assert session["branch"] == "codex/swarm-ec71e047-subtask_1-retr"
    assert branches_seen[1] == session["branch"]


def test_cmd_ensure_evicts_branch_mismatch_before_recreate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    now = datetime(2026, 3, 30, 7, 50, tzinfo=timezone.utc)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    mismatched_path = tmp_path / "stale-session"
    mismatched_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "swarm-run-subtask_1",
                "agent": "codex",
                "branch": "codex/swarm-run-subtask_1",
                "path": str(mismatched_path),
                "created_at": now.isoformat(),
                "last_seen_at": now.isoformat(),
            }
        ]
    }
    saved_state: dict[str, object] = {}
    removed_paths: list[str] = []
    create_calls: list[dict[str, object]] = []

    entries_iter = iter(
        [
            [mod.WorktreeEntry(path=mismatched_path, branch="codex/unrelated")],
            [],
            [],
        ]
    )

    def _create(*_args: object, **kwargs: object) -> dict[str, object]:
        create_calls.append(dict(kwargs))
        return {
            "session_id": "swarm-run-subtask_1",
            "agent": "codex",
            "branch": "codex/swarm-run-subtask_1",
            "path": str(mismatched_path),
            "base_branch": "main",
            "created_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
        }

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: next(entries_iter))
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_create_managed_worktree", _create)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_has_active_lease", lambda _repo, _path: False)
    monkeypatch.setattr(
        mod,
        "_remove_worktree",
        lambda _repo, path: removed_paths.append(str(path)) or True,
    )
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
            "lookup_failed": False,
        },
    )
    monkeypatch.setattr(
        mod,
        "_worktree_status",
        lambda *_args, **_kwargs: {"dirty": False, "ahead": 0, "behind": 0},
    )
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")
    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        agent="codex",
        base="main",
        session_id="swarm-run-subtask_1",
        force_new=False,
        reconcile=True,
        strategy="ff-only",
        print_path=False,
        json=True,
    )
    rc = mod.cmd_ensure(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] is True
    assert removed_paths == [str(mismatched_path)]
    assert create_calls and create_calls[0]["session_id"] == "swarm-run-subtask_1"
    assert saved_state["sessions"][0]["branch"] == "codex/swarm-run-subtask_1"


def test_cmd_cleanup_skips_worktree_with_active_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    leased_path = tmp_path / "leased-wt"
    leased_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "lease1",
                "agent": "codex",
                "branch": "codex/lease1",
                "path": str(leased_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=leased_path, branch="codex/lease1")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": "lease-1",
            "lease_status": "active",
            "last_heartbeat_at": "2026-02-24T00:00:00+00:00",
            "lease_expires_at": "2026-02-24T08:00:00+00:00",
            "owner_agent": "codex",
            "owner_session_id": "sess-1",
            "branch": "codex/lease1",
            "title": "leased",
            "has_live_lease": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "prune"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        ttl_hours=0,
        force_unmerged=False,
        delete_branches=True,
        json=True,
    )
    rc = mod.cmd_cleanup(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == 0
    assert payload["kept"] == 1
    assert payload["skipped_grace"] == 1
    assert payload["skipped_active_session"] == 0
    assert payload["results"][0]["status"] == "skipped_grace"
    assert payload["results"][0]["lifecycle_state"] == "grace"
    assert saved_state["sessions"] == state["sessions"]


def test_classify_session_fails_closed_when_lease_lookup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    path = tmp_path / "lease-error-wt"
    path.mkdir()
    session = {
        "session_id": "lease-error",
        "agent": "codex",
        "branch": "codex/lease-error",
        "path": str(path),
        "created_at": "2026-02-01T00:00:00+00:00",
        "last_seen_at": "2026-02-01T00:00:00+00:00",
    }

    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
            "lookup_failed": True,
        },
    )
    monkeypatch.setattr(mod, "_resolve_ref_sha", lambda *_args, **_kwargs: "abc123")

    metadata = mod._classify_session(
        repo_root,
        session,
        active_paths=set(),
        ttl=mod.timedelta(hours=24),
    )

    assert metadata["lifecycle_state"] == "grace"
    assert metadata["cleanup_lock"] is True
    assert metadata["cleanup_lock_reason"] == "lease_lookup_error"
    assert metadata["lease_lookup_failed"] is True


def test_has_active_session_detects_codex_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    mod._live_cwd_paths.cache_clear()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".codex_session_active").write_text("pid=1234\n", encoding="utf-8")

    def _fake_kill(pid: int, sig: int) -> None:
        if pid == 1234:
            return
        raise ProcessLookupError

    monkeypatch.setattr(mod.os, "kill", _fake_kill)
    assert mod._has_active_session(worktree) is True
    mod._live_cwd_paths.cache_clear()


def test_has_active_session_detects_live_cwd_process_without_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_autopilot as mod

    mod._live_cwd_paths.cache_clear()
    worktree = tmp_path / "wt"
    nested = worktree / "nested"
    nested.mkdir(parents=True)

    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["lsof", "-a", "-d", "cwd", "-Fpn"],
            returncode=0,
            stdout=f"p4242\nn{nested}\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    assert mod._has_active_session(worktree) is True
    mod._live_cwd_paths.cache_clear()


def test_cmd_reconcile_skips_active_session_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    active_path = tmp_path / "active-wt"
    active_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "s1",
                "agent": "codex",
                "branch": "codex/s1",
                "path": str(active_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=active_path, branch="codex/s1")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: True)
    monkeypatch.setattr(
        mod,
        "_integrate_worktree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not reconcile")),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        strategy="ff-only",
        all=True,
        path=None,
        json=True,
    )
    rc = mod.cmd_reconcile(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["skipped_active_session"] == 1
    assert payload["results"][0]["status"] == "skipped_active_session"
    assert saved_state["sessions"][0]["reconcile_status"] == "skipped_active_session"


def test_cmd_reconcile_skips_grace_lane_with_live_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    grace_path = tmp_path / "grace-wt"
    grace_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "grace-1",
                "agent": "codex",
                "branch": "codex/grace-1",
                "path": str(grace_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=grace_path, branch="codex/grace-1")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": "lease-1",
            "lease_status": "active",
            "last_heartbeat_at": "2026-02-24T00:00:00+00:00",
            "lease_expires_at": "2026-02-24T08:00:00+00:00",
            "owner_agent": "codex",
            "owner_session_id": "sess-1",
            "branch": "codex/grace-1",
            "title": "grace",
            "has_live_lease": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "_integrate_worktree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not reconcile")),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        strategy="ff-only",
        ttl_hours=24,
        all=True,
        path=None,
        json=True,
    )
    rc = mod.cmd_reconcile(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["skipped_grace"] == 1
    assert payload["results"][0]["status"] == "skipped_grace"
    assert payload["results"][0]["lifecycle_state"] == "grace"
    assert saved_state["sessions"][0]["reconcile_status"] == "skipped_grace"


def test_cmd_cleanup_archives_before_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    stale_path = tmp_path / "stale-wt"
    stale_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "stale-1",
                "agent": "codex",
                "branch": "codex/stale-1",
                "path": str(stale_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}
    removed_paths: list[Path] = []

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=stale_path, branch="codex/stale-1")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
        },
    )
    monkeypatch.setattr(mod, "_worktree_status", lambda *_args, **_kwargs: {"dirty": False})
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        mod,
        "_archive_session",
        lambda _repo_root, _session, _metadata: (True, "/tmp/archive/stale-1"),
    )
    monkeypatch.setattr(
        mod,
        "_remove_worktree",
        lambda _repo_root, path: removed_paths.append(path) or True,
    )
    monkeypatch.setattr(mod, "_delete_branch", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "prune"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        ttl_hours=0,
        force_unmerged=False,
        delete_branches=True,
        json=True,
    )
    rc = mod.cmd_cleanup(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["archived"] == 1
    assert payload["removed"] == 1
    assert payload["failed_archives"] == 0
    assert payload["results"][0]["archive_path"] == "/tmp/archive/stale-1"
    assert removed_paths == [stale_path]
    assert saved_state["sessions"] == []


def test_cmd_cleanup_skips_live_cwd_process_without_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    mod._live_cwd_paths.cache_clear()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    active_path = tmp_path / "active-wt"
    active_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "active-1",
                "agent": "codex",
                "branch": "codex/active-1",
                "path": str(active_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod,
        "_get_worktree_entries",
        lambda _repo: [mod.WorktreeEntry(path=active_path, branch="codex/active-1")],
    )
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_has_live_cwd_process", lambda _path: True)
    monkeypatch.setattr(
        mod,
        "_archive_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not archive")),
    )
    monkeypatch.setattr(
        mod,
        "_remove_worktree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not remove")),
    )
    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "prune"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        ttl_hours=0,
        force_unmerged=False,
        delete_branches=True,
        json=True,
    )
    rc = mod.cmd_cleanup(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == 0
    assert payload["kept"] == 1
    assert payload["skipped_active_session"] == 1
    assert payload["results"][0]["status"] == "skipped_active_session"
    assert len(saved_state["sessions"]) == 1
    mod._live_cwd_paths.cache_clear()


def test_cmd_cleanup_archives_orphaned_existing_directory_before_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    orphan_path = tmp_path / "orphan-wt"
    orphan_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "orphan-1",
                "agent": "codex",
                "branch": "codex/orphan-1",
                "path": str(orphan_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}
    deleted_paths: list[Path] = []

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: [])
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(
        mod,
        "_lease_snapshot",
        lambda _repo_root, _path: {
            "lease_id": None,
            "lease_status": None,
            "last_heartbeat_at": None,
            "lease_expires_at": None,
            "owner_agent": None,
            "owner_session_id": None,
            "branch": None,
            "title": None,
            "has_live_lease": False,
            "lookup_failed": False,
        },
    )
    monkeypatch.setattr(mod, "_branch_ahead_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        mod,
        "_archive_session",
        lambda _repo_root, _session, metadata: (
            metadata["tracked_worktree"] is False,
            "/tmp/archive/orphan-1",
        ),
    )
    monkeypatch.setattr(
        mod.shutil,
        "rmtree",
        lambda path, ignore_errors=True: deleted_paths.append(Path(path)),
    )
    monkeypatch.setattr(mod, "_delete_branch", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "prune"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        ttl_hours=0,
        force_unmerged=False,
        delete_branches=True,
        json=True,
    )
    rc = mod.cmd_cleanup(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["archived"] == 1
    assert payload["removed"] == 1
    assert payload["results"][0]["archive_path"] == "/tmp/archive/orphan-1"
    assert deleted_paths == [orphan_path]
    assert saved_state["sessions"] == []


def test_cmd_reconcile_preserves_drifted_existing_session_in_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    drifted_path = tmp_path / "drifted-wt"
    drifted_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "drifted-1",
                "agent": "codex",
                "branch": "codex/drifted-1",
                "path": str(drifted_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            }
        ]
    }
    saved_state: dict[str, object] = {}

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: [])
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(
        mod, "_save_state", lambda _state_file, payload: saved_state.update(payload)
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        base="main",
        strategy="ff-only",
        ttl_hours=24,
        all=True,
        path=None,
        json=True,
    )
    rc = mod.cmd_reconcile(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 0
    assert len(saved_state["sessions"]) == 1
    assert saved_state["sessions"][0]["session_id"] == "drifted-1"


def test_cmd_status_reports_lifecycle_and_lock_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import codex_worktree_autopilot as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    active_path = tmp_path / "active"
    safe_path = tmp_path / "safe"
    active_path.mkdir()
    safe_path.mkdir()
    state = {
        "sessions": [
            {
                "session_id": "s1",
                "agent": "codex",
                "branch": "codex/s1",
                "path": str(active_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            },
            {
                "session_id": "s2",
                "agent": "codex",
                "branch": "codex/s2",
                "path": str(safe_path),
                "created_at": "2026-02-01T00:00:00+00:00",
            },
        ]
    }

    metadata_rows = iter(
        [
            {
                "lifecycle_state": "active",
                "cleanup_lock": True,
                "cleanup_lock_reason": "active_session",
                "base_branch": "main",
                "base_sha": "abc123",
                "last_heartbeat_at": "2026-02-24T00:00:00+00:00",
                "lease_status": "active",
                "lease_expires_at": "2026-02-24T08:00:00+00:00",
            },
            {
                "lifecycle_state": "safe-to-clean",
                "cleanup_lock": False,
                "cleanup_lock_reason": None,
                "base_branch": "main",
                "base_sha": "def456",
                "last_heartbeat_at": None,
                "lease_status": None,
                "lease_expires_at": None,
            },
        ]
    )

    monkeypatch.setattr(mod, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(mod, "_load_state", lambda _state_file: state)
    monkeypatch.setattr(mod, "_get_worktree_entries", lambda _repo: [])
    monkeypatch.setattr(
        mod,
        "_annotate_session",
        lambda *_args, **_kwargs: next(metadata_rows),
    )

    args = argparse.Namespace(
        repo=".",
        managed_dir=".worktrees/codex-auto",
        ttl_hours=24,
        json=True,
    )
    rc = mod.cmd_status(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"][0]["lifecycle_state"] == "active"
    assert payload["sessions"][0]["cleanup_lock_reason"] == "active_session"
    assert payload["sessions"][1]["lifecycle_state"] == "safe-to-clean"
