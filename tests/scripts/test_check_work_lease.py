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


def test_claim_by_other_session_blocked(repo: Path) -> None:
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-a") == 0
    assert _main(repo, "feat-x", "--claim", "--session-id", "sess-b") == 1


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
