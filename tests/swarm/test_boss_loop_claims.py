"""Tests for the boss loop issue dispatch claim lock module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aragora.swarm.boss_loop_claims import (
    claim_issue,
    claim_owned_by,
    filter_claimed_issues,
    has_active_foreign_claim,
    issue_claim_owner_pid,
    issue_claim_path,
    read_issue_claim_payload,
    reap_stale_claim,
    release_claim,
)


@pytest.fixture()
def claims_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    claims = tmp_path / ".aragora" / "issue_claims"
    claims.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return claims


class TestClaimPayload:
    def test_read_valid_payload(self, claims_dir: Path) -> None:
        path = claims_dir / "123.lock"
        path.write_text(json.dumps({"pid": 999, "run_id": "test-run"}))
        payload = read_issue_claim_payload(path)
        assert payload is not None
        assert payload["pid"] == 999

    def test_read_missing_file(self, claims_dir: Path) -> None:
        assert read_issue_claim_payload(claims_dir / "missing.lock") is None

    def test_read_empty_file(self, claims_dir: Path) -> None:
        path = claims_dir / "empty.lock"
        path.write_text("")
        assert read_issue_claim_payload(path) is None

    def test_read_invalid_json(self, claims_dir: Path) -> None:
        path = claims_dir / "bad.lock"
        path.write_text("not json")
        assert read_issue_claim_payload(path) is None


class TestOwnership:
    def test_owner_pid_valid(self) -> None:
        assert issue_claim_owner_pid({"pid": 42}) == 42

    def test_owner_pid_none(self) -> None:
        assert issue_claim_owner_pid(None) is None

    def test_owner_pid_zero(self) -> None:
        assert issue_claim_owner_pid({"pid": 0}) is None

    def test_claim_owned_by_self(self) -> None:
        payload = {"run_id": "my-run", "pid": os.getpid()}
        assert claim_owned_by(payload, "my-run") is True

    def test_claim_not_owned_wrong_run(self) -> None:
        payload = {"run_id": "other-run", "pid": os.getpid()}
        assert claim_owned_by(payload, "my-run") is False

    def test_claim_not_owned_wrong_pid(self) -> None:
        payload = {"run_id": "my-run", "pid": 99999}
        assert claim_owned_by(payload, "my-run") is False


class TestClaimAndRelease:
    def test_claim_creates_lock_file(self, claims_dir: Path) -> None:
        ok, err = claim_issue(42, "run-1")
        assert ok is True
        assert err is None
        assert (claims_dir / "42.lock").exists()

    def test_claim_idempotent_same_run(self, claims_dir: Path) -> None:
        claim_issue(42, "run-1")
        ok, err = claim_issue(42, "run-1")
        assert ok is True

    def test_claim_rejected_by_foreign(self, claims_dir: Path) -> None:
        path = claims_dir / "42.lock"
        path.write_text(
            json.dumps(
                {
                    "run_id": "other-run",
                    "pid": os.getpid(),
                    "host": "localhost",
                }
            )
        )
        ok, err = claim_issue(42, "my-run")
        assert ok is False
        assert "already claimed" in (err or "")

    def test_release_removes_own_claim(self, claims_dir: Path) -> None:
        claim_issue(42, "run-1")
        release_claim(42, "run-1")
        assert not (claims_dir / "42.lock").exists()

    def test_release_preserves_foreign_claim(self, claims_dir: Path) -> None:
        path = claims_dir / "42.lock"
        path.write_text(json.dumps({"run_id": "other", "pid": 99999}))
        release_claim(42, "my-run")
        assert path.exists()


class TestReapStale:
    def test_reap_expired(self, claims_dir: Path) -> None:
        path = claims_dir / "42.lock"
        path.write_text(json.dumps({"run_id": "old", "pid": 99999}))
        os.utime(path, (0, 0))  # set mtime to epoch
        reaped = reap_stale_claim(42, path, {"run_id": "old", "pid": 99999}, "my-run")
        assert reaped is True

    def test_reap_missing_file(self, claims_dir: Path) -> None:
        path = claims_dir / "missing.lock"
        assert reap_stale_claim(42, path, None, "my-run") is True


class TestForeignClaim:
    def test_no_claim_file(self, claims_dir: Path) -> None:
        assert has_active_foreign_claim(42, "my-run") is False

    def test_own_claim_not_foreign(self, claims_dir: Path) -> None:
        claim_issue(42, "my-run")
        assert has_active_foreign_claim(42, "my-run") is False


class TestFilterClaimed:
    def test_filters_claimed_issues(self, claims_dir: Path) -> None:
        from types import SimpleNamespace

        claim_issue(1, "other-run")
        # Change the claim to look foreign
        path = claims_dir / "1.lock"
        path.write_text(
            json.dumps({"run_id": "other-run", "pid": os.getpid() + 1, "host": "localhost"})
        )

        issues = [SimpleNamespace(number=1), SimpleNamespace(number=2)]
        filtered = filter_claimed_issues(issues, "my-run")
        # Issue 1 should be skipped (foreign claim), issue 2 passes
        assert len(filtered) == 1
        assert filtered[0].number == 2
