"""Tests for the swarm session coordinator facade."""

from __future__ import annotations

import json
import os
import time

from aragora.coordination.registry import SessionRegistry
from aragora.swarm.session_coordinator import (
    claim_pr,
    get_my_assignment,
    list_findings,
    read_directives,
    report_finding,
    set_assignment,
)


class TestSessionCoordinator:
    @staticmethod
    def _rewrite_heartbeat(
        tmp_path,
        session_id: str,
        *,
        last_heartbeat: float,
    ) -> None:
        session_path = tmp_path / ".aragora_coordination" / "sessions" / f"{session_id}.json"
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        payload["last_heartbeat"] = last_heartbeat
        session_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_set_assignment_roundtrip(self, tmp_path):
        payload = set_assignment(
            "codex-a",
            "SDK parity consolidation",
            scope=["#2684"],
            constraints=["no queue drain"],
            issued_by="boss-codex",
            repo_root=tmp_path,
        )

        assert payload["target"] == "codex-a"
        assignment = get_my_assignment("codex-a", repo_root=tmp_path)
        assert assignment is not None
        assert assignment["task"] == "SDK parity consolidation"
        assert assignment["scope"] == ["#2684"]

    def test_claim_pr_contested(self, tmp_path):
        first = claim_pr(2684, "codex-a", repo_root=tmp_path)
        second = claim_pr(2684, "codex-b", repo_root=tmp_path)

        assert first["status"] == "granted"
        assert second["status"] == "contested"
        assert second["contested_by"][0]["session_id"] == "codex-a"

    def test_claim_pr_reaps_stale_session_claims_before_contest(self, tmp_path):
        registry = SessionRegistry(repo_path=tmp_path, stale_timeout_seconds=1)
        stale_session = registry.register(
            agent="codex",
            worktree="/tmp/wt1",
            pid=os.getpid(),
        )
        first = claim_pr(2684, stale_session.session_id, repo_root=tmp_path)
        self._rewrite_heartbeat(
            tmp_path,
            stale_session.session_id,
            last_heartbeat=time.time() - 1000,
        )

        second = claim_pr(2684, "codex-b", repo_root=tmp_path)

        assert first["status"] == "granted"
        assert second["status"] == "granted"
        assert second["contested_by"] == []

    def test_report_and_list_findings(self, tmp_path):
        report_finding(
            "Bandit B310 in auth/oidc.py",
            "codex-b",
            kind="blocker",
            pr=2679,
            scope=["aragora/auth/oidc.py"],
            repo_root=tmp_path,
        )

        findings = list_findings(repo_root=tmp_path, kind="blocker", pr=2679)
        assert len(findings) == 1
        assert findings[0]["message"] == "Bandit B310 in auth/oidc.py"
        assert findings[0]["source_session"] == "codex-b"

    def test_read_directives_aggregates_state(self, tmp_path):
        set_assignment("codex-a", "Own parity lane", repo_root=tmp_path)
        claim_pr(2684, "codex-a", repo_root=tmp_path)
        report_finding("Verification route bug", "review-codex", pr=2677, repo_root=tmp_path)

        view = read_directives(repo_root=tmp_path, findings_limit=5)
        assert view["summary"]["directive_count"] == 1
        assert view["summary"]["claim_count"] == 1
        assert view["summary"]["finding_count"] == 1
        assert view["directives"][0]["target"] == "codex-a"

    def test_read_directives_clears_dead_session_assignments(self, tmp_path):
        session = SessionRegistry(repo_path=tmp_path).register(
            agent="codex",
            worktree="/tmp/wt1",
            pid=999999999,
        )
        set_assignment(session.session_id, "Stale assignment", repo_root=tmp_path)

        view = read_directives(repo_root=tmp_path)

        assert view["summary"]["directive_count"] == 0
        assert view["summary"]["session_count"] == 0
        assert view["summary"]["dead_session_count"] == 1
        assert view["reaped_sessions"][0]["status"] == "dead"
        assert get_my_assignment(session.session_id, repo_root=tmp_path) is None

    def test_read_directives_reaps_stale_live_sessions_and_claims(self, tmp_path):
        registry = SessionRegistry(repo_path=tmp_path, stale_timeout_seconds=1)
        session = registry.register(
            agent="codex",
            worktree="/tmp/wt1",
            pid=os.getpid(),
        )
        set_assignment(session.session_id, "Stale assignment", repo_root=tmp_path)
        claim_pr(2754, session.session_id, repo_root=tmp_path)
        self._rewrite_heartbeat(
            tmp_path,
            session.session_id,
            last_heartbeat=time.time() - 1000,
        )

        view = read_directives(repo_root=tmp_path)

        assert view["summary"]["directive_count"] == 0
        assert view["summary"]["session_count"] == 0
        assert view["summary"]["claim_count"] == 0
        assert view["summary"]["stale_session_count"] == 1
        assert view["summary"]["reaped_session_count"] == 1
        assert view["reaped_sessions"][0]["status"] == "stale"
        assert view["reaped_sessions"][0]["pid_alive"] is True
        assert view["reaped_sessions"][0]["heartbeat_stale"] is True
        assert view["reaped_sessions"][0]["directive_cleared"] is True
        assert view["reaped_sessions"][0]["claims_released"] == 1
