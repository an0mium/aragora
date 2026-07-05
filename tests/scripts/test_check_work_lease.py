"""Tests for ``scripts/check_work_lease.py`` (lease-rule preflight, #8851 item 2)."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


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


cwl = _load_module("check_work_lease.py")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ARAGORA_DEV_COORDINATION_DB",
        "ARAGORA_SESSION_ID",
        "ARAGORA_AGENT_SESSION_ID",
        "ARAGORA_SWARM_SESSION_ID",
        "ARAGORA_AGENT",
        "ARAGORA_AGENT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", "README.md")
    _run(repo, "git", "commit", "-m", "initial")
    return repo


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _main(repo: Path, *argv: str) -> int:
    return cwl.main(["--repo", str(repo), *argv])


def _db_path(repo: Path) -> Path:
    return cwl.resolve_db_path(repo)


def test_lease_row_work_id_accepts_legacy_pr_number_metadata() -> None:
    lease = cwl.LeaseRow(
        lease_id="lease-1",
        task_id="branch:feat-x",
        title="legacy PR lease",
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="feat-x",
        worktree_path="/tmp/repo",
        status="active",
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        metadata={"pr_number": "8852"},
    )

    assert lease.work_id == "pr:8852"


def test_no_lease_without_claim_fails(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _main(repo, "feat-x", "--session-id", "sess-a")
    assert code == 1
    out = capsys.readouterr().out
    assert "no active lease" in out
    assert "--claim" in out


def test_claim_then_check_holds(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--agent", "claude") == 0
    assert "claimed lease" in capsys.readouterr().out
    # Second invocation by the same session holds without --claim.
    assert _main(repo, "feat-x", "--session-id", "sess-a") == 0
    assert "holding lease" in capsys.readouterr().out


def test_conflict_reports_owner_one_line(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--agent", "claude") == 0
    capsys.readouterr()
    code = _main(repo, "feat-x", "--session-id", "sess-b")
    assert code == 1
    out = capsys.readouterr().out
    assert "LEASE CONFLICT" in out
    assert "sess-a" in out
    assert "claude" in out
    # One human-readable line.
    assert len([line for line in out.strip().splitlines() if "LEASE CONFLICT" in line]) == 1


def test_claim_by_other_session_blocked_at_store(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    capsys.readouterr()
    # The second claim must be rejected transactionally by claim_lease via
    # the synthetic branch-lock glob, not by the read-only pre-check.
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-b") == 1
    out = capsys.readouterr().out
    assert "LEASE CONFLICT" in out
    assert "sess-a" in out


def test_branch_lock_glob_conflicts_for_store_direct_same_lock(repo: Path) -> None:
    # Two helper-style claims conflict AT THE STORE: a direct claim_lease
    # call carrying the same branch-lock glob raises LeaseConflictError.
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    from aragora.nomic.dev_coordination import DevCoordinationStore, LeaseConflictError

    store = DevCoordinationStore(repo_root=repo)
    lock = cwl.BRANCH_LOCK_GLOB_TEMPLATE.format(branch="feat-x")
    with pytest.raises(LeaseConflictError):
        store.claim_lease(
            task_id="branch:feat-x",
            title="competing claim",
            owner_agent="codex",
            owner_session_id="sess-b",
            branch="feat-x",
            worktree_path=str(repo),
            allowed_globs=[lock],
        )


def test_store_direct_branch_lease_triggers_backoff(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A store-direct claimant (e.g. swarm) holds the branch with a file
    # scope that does not overlap the branch-lock glob: claim_lease cannot
    # see it, so the post-claim double-check must back off and release the
    # just-claimed lease.
    from aragora.nomic.dev_coordination import DevCoordinationStore

    store = DevCoordinationStore(repo_root=repo)
    store.claim_lease(
        task_id="wo-1",
        title="swarm work order",
        owner_agent="swarm",
        owner_session_id="sess-swarm",
        branch="feat-x",
        worktree_path=str(repo),
        allowed_globs=["aragora/swarm/*.py"],
    )
    code = _main(repo, "feat-x", "--claim", "--session-id", "sess-b")
    assert code == 1
    out = capsys.readouterr().out
    assert "LEASE CONFLICT" in out
    assert "sess-swarm" in out
    survivors = cwl.active_leases_for_branch(cwl.resolve_db_path(repo), "feat-x")
    assert [lease.owner_session_id for lease in survivors] == ["sess-swarm"]


def test_stale_dead_worker_lease_reclaimed_on_claim(repo: Path) -> None:
    # A foreign lease with a FUTURE expires_at but no worker_pid and no
    # heartbeat for >30min must not squat the branch: the --claim path goes
    # straight to store.claim_lease, which reaps it first.
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with sqlite3.connect(_db_path(repo)) as conn:
        conn.execute("UPDATE leases SET updated_at = ?, created_at = ?", (stale, stale))
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-b") == 0


def test_other_branch_unaffected(repo: Path) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    assert _main(repo, "feat-y", "--claim", "--session-id", "sess-b") == 0


def test_release_then_reclaim_by_other_session(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    assert _main(repo, "feat-x", "--release", "--session-id", "sess-a") == 0
    assert "released lease" in capsys.readouterr().out
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-b") == 0


def test_release_is_idempotent(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--release", "--session-id", "sess-a") == 0
    assert "no-op" in capsys.readouterr().out


def test_release_by_non_owner_blocked(repo: Path) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    assert _main(repo, "feat-x", "--release", "--session-id", "sess-b") == 1


def test_renew_extends_expiry(repo: Path) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--ttl-hours", "1") == 0
    before = _lease_expiry(repo, "feat-x")
    assert _main(repo, "feat-x", "--renew", "--session-id", "sess-a", "--ttl-hours", "12") == 0
    after = _lease_expiry(repo, "feat-x")
    assert after > before


def test_renew_without_lease_fails(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--renew", "--session-id", "sess-a") == 1
    assert "no active lease" in capsys.readouterr().out


def test_expired_foreign_lease_does_not_block_claim(repo: Path) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    db = _db_path(repo)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE leases SET expires_at = ?", (past,))
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-b") == 0


def test_unreachable_store_warns_fail_open(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a sqlite database at all........")
    code = _main(repo, "feat-x", "--db", str(corrupt), "--session-id", "sess-a")
    assert code == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "fail-open" in err


def test_unreachable_store_strict_fails(repo: Path, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a sqlite database at all........")
    code = _main(repo, "feat-x", "--db", str(corrupt), "--session-id", "sess-a", "--strict")
    assert code == 1


def test_wal_sidecars_absent_readonly_dir_still_reads(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Clean close removes -wal/-shm; a copied DB never has them. If the
    # store directory is also not writable by the invoking UID, mode=ro
    # cannot create them — the immutable=1 fallback must still read.
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    capsys.readouterr()
    db = _db_path(repo)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    db.parent.chmod(0o555)
    try:
        code = _main(repo, "feat-x", "--session-id", "sess-b")
        captured = capsys.readouterr()
    finally:
        db.parent.chmod(0o755)
    assert code == 1
    assert "LEASE CONFLICT" in captured.out
    assert "WARNING: lease store unreachable" not in captured.err


def test_readonly_open_falls_back_to_immutable(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Deterministically exercise the fallback ladder: the plain mode=ro
    # attempt raises OperationalError (as when sidecars are absent and the
    # directory is unwritable), the immutable=1 retry succeeds.
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    capsys.readouterr()
    real_connect = sqlite3.connect

    def fake_connect(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        if (
            isinstance(database, str)
            and database.startswith("file:")
            and "mode=ro" in database
            and "immutable=1" not in database
        ):
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cwl.sqlite3, "connect", fake_connect)
    code = _main(repo, "feat-x", "--session-id", "sess-b")
    captured = capsys.readouterr()
    assert code == 1
    assert "LEASE CONFLICT" in captured.out
    assert "WARNING: lease store unreachable" not in captured.err


def test_missing_db_is_no_lease_not_unreachable(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "never-created.db"
    code = _main(repo, "feat-x", "--db", str(missing), "--session-id", "sess-a")
    assert code == 1
    captured = capsys.readouterr()
    assert "no active lease" in captured.out
    assert "WARNING: lease store unreachable" not in captured.err


def test_record_lane_sidecar_roundtrip(repo: Path) -> None:
    assert (
        _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--record-lane", "lane-1") == 0
    )
    sidecar = repo / ".aragora" / "agent-bridge" / "lane-leases.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["lane-1"]["branch"] == "feat-x"
    assert payload["lane-1"]["owner_session_id"] == "sess-a"
    assert payload["lane-1"]["lease_id"]
    assert (
        _main(repo, "feat-x", "--release", "--session-id", "sess-a", "--record-lane", "lane-1") == 0
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "lane-1" not in payload


def test_json_output(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["action"] == "claim"
    assert payload["lease_id"]


def test_verify_only_matching_work_id_succeeds_without_sidecar_write(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _main(
            repo,
            "feat-x",
            "--claim",
            "--session-id",
            "sess-a",
            "--agent",
            "claude",
            "--work-id",
            "pr:8852",
        )
        == 0
    )
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--record-lane",
            "lane-1",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["action"] == "check"
    assert payload["reason"] is None
    assert payload["work_id"] == "pr:8852"
    assert payload["branch"] == "feat-x"
    assert payload["owner_session_id"] == "sess-a"
    assert payload["lease_id"]
    assert not (repo / ".aragora" / "agent-bridge" / "lane-leases.json").exists()


def test_verify_only_branch_work_id_matches_store_direct_branch_lease(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aragora.nomic.dev_coordination import DevCoordinationStore

    store = DevCoordinationStore(repo_root=repo)
    lease = store.claim_lease(
        task_id="WO-branch-owner",
        title="swarm work order",
        owner_agent="swarm",
        owner_session_id="sess-swarm",
        branch="feat-x",
        worktree_path=str(repo),
        allowed_globs=["aragora/swarm/*.py"],
    )
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-swarm",
            "--work-id",
            "branch:feat-x",
            "--json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["reason"] is None
    assert payload["lease_id"] == lease.lease_id
    assert payload["owner_session_id"] == "sess-swarm"


def test_verify_only_branch_work_id_reports_foreign_store_direct_lease(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aragora.nomic.dev_coordination import DevCoordinationStore

    store = DevCoordinationStore(repo_root=repo)
    lease = store.claim_lease(
        task_id="WO-branch-owner",
        title="swarm work order",
        owner_agent="swarm",
        owner_session_id="sess-swarm",
        branch="feat-x",
        worktree_path=str(repo),
        allowed_globs=["aragora/swarm/*.py"],
    )
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-other",
            "--work-id",
            "branch:feat-x",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "wrong_owner"
    assert payload["lease_id"] == lease.lease_id
    assert payload["owner_session_id"] == "sess-swarm"


def test_verify_only_pr_work_id_reports_nonmatching_foreign_branch_lease(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--work-id", "pr:8852") == 0
    from aragora.nomic.dev_coordination import DevCoordinationStore

    store = DevCoordinationStore(repo_root=repo)
    lease = store.claim_lease(
        task_id="WO-nonmatching",
        title="swarm work order",
        owner_agent="swarm",
        owner_session_id="sess-swarm",
        branch="feat-x",
        worktree_path=str(repo),
        allowed_globs=["aragora/swarm/*.py"],
    )
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "wrong_owner"
    assert payload["lease_id"] == lease.lease_id
    assert payload["owner_session_id"] == "sess-swarm"


def test_verify_only_pr_work_id_accepts_owned_branch_level_lease(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["reason"] is None
    assert payload["owner_session_id"] == "sess-a"


def test_verify_only_pr_work_id_rejects_foreign_exact_pr_lease_despite_branch_fallback(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    with sqlite3.connect(_db_path(repo)) as conn:
        conn.execute(
            """
            INSERT INTO leases (
                lease_id, task_id, title, owner_agent, owner_session_id, branch, worktree_path,
                allowed_globs_json, claimed_paths_json, expected_tests_json, status, created_at,
                updated_at, expires_at, metadata_json
            )
            SELECT
                'lease-foreign-pr', 'pr:8852', title, 'codex', 'sess-b', branch, worktree_path,
                allowed_globs_json, claimed_paths_json, expected_tests_json, status, created_at,
                updated_at, expires_at, '{"work_id": "pr:8852"}'
            FROM leases
            WHERE branch = ?
            """,
            ("feat-x",),
        )
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] in {"ambiguous_owner", "wrong_owner"}
    assert payload["owner_session_id"] == "sess-b"


def test_verify_only_rejects_mismatched_work_id_with_stable_reason(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--work-id", "pr:8852") == 0
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:9999",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "missing_lease"
    assert payload["work_id"] == "pr:9999"
    assert payload["branch"] == "feat-x"
    assert payload["lease_id"] is None


def test_advisory_preserves_missing_lease_reason_but_exits_zero(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--advisory",
            "--session-id",
            "sess-a",
            "--work-id",
            "branch:feat-x",
            "--json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "missing_lease"
    assert payload["work_id"] == "branch:feat-x"
    assert payload["branch"] == "feat-x"


def test_advisory_rejected_for_mutating_modes(repo: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _main(repo, "feat-x", "--claim", "--advisory", "--session-id", "sess-a")

    assert excinfo.value.code == 2


def test_verify_only_wrong_owner_reason(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--work-id", "pr:8852") == 0
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-b",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "wrong_owner"
    assert payload["owner_session_id"] == "sess-a"
    assert payload["lease_id"]


def test_verify_only_expired_lease_reason(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--work-id", "pr:8852") == 0
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(_db_path(repo)) as conn:
        conn.execute("UPDATE leases SET expires_at = ?", (past,))
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "expired_lease"
    assert payload["owner_session_id"] == "sess-a"


def test_verify_only_bridge_sidecar_without_dev_lease_reason(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sidecar = repo / ".aragora" / "agent-bridge" / "lane-leases.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "lane-1": {
                    "branch": "feat-x",
                    "lease_id": "missing-lease",
                    "owner_session_id": "sess-a",
                    "work_id": "pr:8852",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--record-lane",
            "lane-1",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "bridge_only_no_dev_lease"
    assert payload["lease_id"] == "missing-lease"


def test_verify_only_ambiguous_owner_reason(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a", "--work-id", "pr:8852") == 0
    with sqlite3.connect(_db_path(repo)) as conn:
        conn.execute(
            """
            INSERT INTO leases (
                lease_id, task_id, title, owner_agent, owner_session_id, branch, worktree_path,
                allowed_globs_json, claimed_paths_json, expected_tests_json, status, created_at,
                updated_at, expires_at, metadata_json
            )
            SELECT
                'lease-ambiguous', task_id, title, 'codex', 'sess-b', branch, worktree_path,
                allowed_globs_json, claimed_paths_json, expected_tests_json, status, created_at,
                updated_at, expires_at, metadata_json
            FROM leases
            WHERE branch = ?
            """,
            ("feat-x",),
        )
    capsys.readouterr()

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--session-id",
            "sess-a",
            "--work-id",
            "pr:8852",
            "--json",
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "ambiguous_owner"
    assert payload["owners"] == ["sess-a", "sess-b"]


def test_verify_only_store_unreachable_json_reason(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a sqlite database at all........")

    assert (
        _main(
            repo,
            "feat-x",
            "--verify-only",
            "--strict",
            "--advisory",
            "--db",
            str(corrupt),
            "--json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "store_unreachable"


def test_session_id_from_environment(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARAGORA_SESSION_ID", "sess-env")
    assert _main(repo, "feat-x", "--claim") == 0
    capsys.readouterr()
    assert _main(repo, "feat-x") == 0
    assert "sess-env" in capsys.readouterr().out


def _lease_expiry(repo: Path, branch: str) -> str:
    with sqlite3.connect(_db_path(repo)) as conn:
        row = conn.execute(
            "SELECT expires_at FROM leases WHERE branch = ? AND status = 'active'", (branch,)
        ).fetchone()
    assert row is not None
    return str(row[0])
