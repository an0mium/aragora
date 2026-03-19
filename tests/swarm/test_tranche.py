from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aragora.nomic.dev_coordination import LeaseConflictError
from aragora.swarm.tranche import (
    TrancheArtifactStore,
    TrancheInspector,
    TrancheLaneArtifact,
    TrancheManifest,
    load_tranche_manifest,
    parse_github_reference_url,
)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", "README.md")
    _run(repo, "git", "commit", "-m", "initial")
    _run(repo, "git", "remote", "add", "origin", str(repo))
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


class _FakeReferenceClient:
    def __init__(
        self,
        *,
        pulls: dict[int, dict] | None = None,
        issues: dict[int, dict] | None = None,
    ) -> None:
        self.pulls = pulls or {}
        self.issues = issues or {}

    def get_pr(self, repo: str, number: int) -> dict:
        payload = dict(self.pulls[number])
        payload.setdefault("url", f"https://github.com/{repo}/pull/{number}")
        return payload

    def get_issue(self, repo: str, number: int) -> dict:
        payload = dict(self.issues[number])
        payload.setdefault("url", f"https://github.com/{repo}/issues/{number}")
        return payload


def _historical_manifest() -> TrancheManifest:
    return TrancheManifest.from_dict(
        {
            "manifest_version": 1,
            "manifest_id": "boss-live-proof-tranche-2026-03-19",
            "repo": {
                "name": "synaptent/aragora",
                "root": "/tmp/repo",
                "base_ref": "origin/main",
                "base_sha": "1eb030814",
            },
            "references": {
                "gates": {
                    "pr_1061": {
                        "kind": "pull_request",
                        "url": "https://github.com/synaptent/aragora/pull/1061",
                        "state": "merged",
                    },
                    "pr_1065": {
                        "kind": "pull_request",
                        "url": "https://github.com/synaptent/aragora/pull/1065",
                        "state": "open",
                    },
                    "pr_1060": {
                        "kind": "pull_request",
                        "url": "https://github.com/synaptent/aragora/pull/1060",
                        "state": "closed",
                    },
                },
                "live_target": {
                    "issue_1064": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1064",
                        "state": "open",
                        "label": "boss-loop-test",
                    }
                },
                "retired_targets": {
                    "issue_873": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/873",
                        "state": "closed",
                    },
                    "issue_909": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/909",
                        "state": "closed",
                    },
                },
            },
            "gates": {
                "gate_1061": {
                    "source_ref": "pr_1061",
                    "state": "satisfied",
                    "required_for": ["codex_a_live_gate"],
                    "satisfy_when": "merged_to_main",
                },
                "gate_1065": {
                    "source_ref": "pr_1065",
                    "state": "pending",
                    "required_for": ["codex_a_live_gate", "codex_b_gatekeeper"],
                    "satisfy_when": "merged to main",
                },
                "replace_1060": {
                    "source_ref": "pr_1060",
                    "state": "satisfied",
                    "required_for": ["codex_b_gatekeeper"],
                    "satisfy_when": "closed without merge",
                },
            },
            "lanes": [
                {
                    "lane_id": "codex_a_live_gate",
                    "owner_role": "critical_path_engineer",
                    "branch": {"convention": "codex/<swarm-fix-or-live-proof>"},
                    "worktree": {"convention": ".worktrees/codex-auto/<session-id>"},
                    "allowed_write_scope": ["aragora/swarm/**"],
                    "dependencies": ["gate_1061", "gate_1065", "issue_1064"],
                    "verification_commands": ["python -m aragora.cli.main swarm boss-loop --json"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["boss loop JSON"],
                },
                {
                    "lane_id": "codex_b_gatekeeper",
                    "owner_role": "pr_gate_and_repo_ops",
                    "branch": {"convention": "codex/<repo-ops-or-gate-task>"},
                    "worktree": {"convention": ".worktrees/codex-auto/<session-id>"},
                    "allowed_write_scope": [],
                    "dependencies": ["gate_1065", "replace_1060"],
                    "verification_commands": ["gh pr view 1065 --json state"],
                    "stop_conditions": ["required check blocking"],
                    "expected_receipts_artifacts": ["gate summary"],
                },
            ],
            "terminal_outcomes": {
                "success": {"definition": "proof completed"},
                "needs_human": {"definition": "blocked safely"},
                "stop_and_replan": {"definition": "target stale"},
            },
        }
    )


def _historical_client(*, gate_1065_state: str = "OPEN") -> _FakeReferenceClient:
    pulls = {
        1061: {
            "number": 1061,
            "state": "MERGED",
            "mergedAt": "2026-03-19T15:27:39Z",
            "title": "fix(swarm): stop one-tick boss-loop live runs after dispatch",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "APPROVED",
        },
        1065: {
            "number": 1065,
            "state": gate_1065_state,
            "mergedAt": "2026-03-19T17:22:00Z" if gate_1065_state == "MERGED" else None,
            "title": "fix(swarm): close stdin on non-detached workers and guard script(1) PTY",
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN" if gate_1065_state == "MERGED" else "BLOCKED",
            "reviewDecision": "APPROVED" if gate_1065_state == "MERGED" else "REVIEW_REQUIRED",
        },
        1060: {
            "number": 1060,
            "state": "CLOSED",
            "mergedAt": None,
            "title": "stale boss-model cleanup",
            "mergeable": "UNKNOWN",
            "mergeStateStatus": "UNKNOWN",
            "reviewDecision": "",
        },
    }
    issues = {
        1064: {
            "number": 1064,
            "state": "OPEN",
            "closedAt": None,
            "title": "Boss-loop execution: bump @supabase/supabase-js from 2.99.1 to 2.99.3 in /aragora/live",
            "labels": [{"name": "boss-loop-test"}],
        },
        873: {
            "number": 873,
            "state": "CLOSED",
            "closedAt": "2026-03-19T15:00:00Z",
            "title": "stale dependency lane",
            "labels": [{"name": "boss-loop-test"}],
        },
        909: {
            "number": 909,
            "state": "CLOSED",
            "closedAt": "2026-03-19T15:00:00Z",
            "title": "stale benchmark issue",
            "labels": [{"name": "boss-loop-test"}],
        },
    }
    return _FakeReferenceClient(pulls=pulls, issues=issues)


def test_parse_github_reference_url() -> None:
    ref = parse_github_reference_url("https://github.com/synaptent/aragora/pull/1065")
    assert ref.owner == "synaptent"
    assert ref.repo == "aragora"
    assert ref.kind == "pull_request"
    assert ref.number == 1065


def test_load_tracked_example_manifest() -> None:
    path = Path("docs/examples/boss-lane-manifest-2026-03-19.yaml")
    manifest = load_tranche_manifest(path)
    assert manifest.manifest_id == "boss-live-proof-tranche-2026-03-19"
    assert "retired_targets" in manifest.references
    assert "issue_873" in manifest.references["retired_targets"]


def test_manifest_validation_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="manifest_id"):
        TrancheManifest.from_dict(
            {
                "repo": {"name": "synaptent/aragora"},
                "references": {},
                "gates": {},
                "lanes": [],
                "terminal_outcomes": {},
            }
        )


def test_inspect_reports_historical_gate_states_and_stale_targets(tmp_path: Path) -> None:
    manifest = _historical_manifest()
    artifact_store = TrancheArtifactStore(tmp_path)
    artifact_store.save(
        manifest.manifest_id,
        TrancheLaneArtifact(
            lane_id="claude_a_runtime_verifier",
            source_ref="issue_1064",
            status="approved_with_risk",
            commands=["gh pr view 1065 --json state"],
            urls=["https://github.com/synaptent/aragora/pull/1065"],
            run_id="run-873",
            worktree_path="/tmp/worktree",
            residual_risk="multi-tick path still unmerged",
            next_actions=["Merge #1065"],
        ),
    )
    inspector = TrancheInspector(
        repo_root=tmp_path,
        reference_client=_historical_client(gate_1065_state="OPEN"),
        artifact_store=artifact_store,
    )

    payload = inspector.inspect(manifest)

    assert payload["gates"]["gate_1061"]["state"] == "satisfied"
    assert payload["gates"]["gate_1065"]["state"] == "pending"
    assert payload["references"]["issue_1064"]["status"] == "actionable"
    assert payload["references"]["issue_873"]["status"] == "stale"
    assert payload["references"]["issue_909"]["status"] == "stale"
    assert payload["recommended_action"]["kind"] == "resolve_gate"
    assert payload["recommended_action"]["source_ref"] == "pr_1065"
    verifier_lane = next(
        lane for lane in payload["lanes"] if lane["lane_id"] == "codex_b_gatekeeper"
    )
    assert verifier_lane["claimable"] is False
    assert payload["artifacts"][0]["lane_id"] == "claude_a_runtime_verifier"


def test_inspect_recommends_lane_once_pending_gate_is_satisfied(tmp_path: Path) -> None:
    manifest = _historical_manifest()
    inspector = TrancheInspector(
        repo_root=tmp_path,
        reference_client=_historical_client(gate_1065_state="MERGED"),
        artifact_store=TrancheArtifactStore(tmp_path),
    )

    payload = inspector.inspect(manifest)

    assert payload["gates"]["gate_1065"]["state"] == "satisfied"
    assert payload["recommended_action"] == {
        "kind": "run_lane",
        "lane_id": "codex_a_live_gate",
        "reason": "All dependencies satisfied",
    }


def test_declared_scope_conflict_blocks_only_writable_lanes(tmp_path: Path) -> None:
    manifest = _historical_manifest()
    manifest.lanes.append(
        manifest.lane("codex_a_live_gate").__class__.from_dict(
            {
                "lane_id": "codex_c_overlap",
                "owner_role": "supporting_engineer",
                "branch": {"convention": "codex/<support>"},
                "worktree": {"convention": ".worktrees/codex-auto/<session-id>"},
                "allowed_write_scope": ["aragora/swarm/worker_launcher.py"],
                "dependencies": ["gate_1061"],
                "verification_commands": ["pytest tests/swarm/test_worker_launcher.py -q"],
                "stop_conditions": ["scope leak"],
                "expected_receipts_artifacts": ["worker receipt"],
            }
        )
    )
    inspector = TrancheInspector(
        repo_root=tmp_path,
        reference_client=_historical_client(gate_1065_state="MERGED"),
        artifact_store=TrancheArtifactStore(tmp_path),
    )

    payload = inspector.inspect(manifest)

    assert len(payload["scope_conflicts"]) == 1
    conflict = payload["scope_conflicts"][0]
    assert {conflict["left_lane_id"], conflict["right_lane_id"]} == {
        "codex_a_live_gate",
        "codex_c_overlap",
    }
    assert "codex_b_gatekeeper" not in {conflict["left_lane_id"], conflict["right_lane_id"]}


def test_prepare_lane_claim_rejects_read_only_lane(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = _historical_manifest()
    inspector = TrancheInspector(
        repo_root=repo,
        reference_client=_historical_client(gate_1065_state="MERGED"),
        artifact_store=TrancheArtifactStore(repo),
    )

    with pytest.raises(ValueError, match="read-only"):
        inspector.prepare_lane_claim(
            manifest,
            lane_id="codex_b_gatekeeper",
            task_id="task-1",
            title="gatekeeper",
            owner_agent="codex",
            owner_session_id="sess-1",
            branch="codex/gatekeeper",
            worktree_path=str(repo),
        )


def test_prepare_lane_claim_detects_scope_conflicts(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = _historical_manifest()
    manifest.lanes.append(
        manifest.lane("codex_a_live_gate").__class__.from_dict(
            {
                "lane_id": "codex_c_overlap",
                "owner_role": "supporting_engineer",
                "branch": {"convention": "codex/<support>"},
                "worktree": {"convention": ".worktrees/codex-auto/<session-id>"},
                "allowed_write_scope": ["aragora/swarm/worker_launcher.py"],
                "dependencies": ["gate_1061"],
                "verification_commands": ["pytest tests/swarm/test_worker_launcher.py -q"],
                "stop_conditions": ["scope leak"],
                "expected_receipts_artifacts": ["worker receipt"],
            }
        )
    )
    inspector = TrancheInspector(
        repo_root=repo,
        reference_client=_historical_client(gate_1065_state="MERGED"),
        artifact_store=TrancheArtifactStore(repo),
    )

    inspector.prepare_lane_claim(
        manifest,
        lane_id="codex_a_live_gate",
        task_id="task-a",
        title="primary lane",
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/primary-lane",
        worktree_path=str(repo),
    )

    with pytest.raises(LeaseConflictError):
        inspector.prepare_lane_claim(
            manifest,
            lane_id="codex_c_overlap",
            task_id="task-c",
            title="overlap lane",
            owner_agent="codex",
            owner_session_id="sess-c",
            branch="codex/overlap-lane",
            worktree_path=str(repo),
        )


def test_lane_artifact_round_trip_preserves_receipt_fields(tmp_path: Path) -> None:
    store = TrancheArtifactStore(tmp_path)
    artifact = TrancheLaneArtifact(
        lane_id="codex_a_live_gate",
        source_ref="issue_1064",
        status="running",
        commands=["python -m aragora.cli.main swarm boss-loop --json"],
        urls=[
            "https://github.com/synaptent/aragora/issues/1064",
            "https://github.com/synaptent/aragora/pull/1065",
        ],
        run_id="8c06b6cc-a1f",
        worktree_path="/tmp/repo/.worktrees/swarm-8c06b6cc-subtask_1",
        residual_risk="worker still executing npm install",
        next_actions=["Inspect the active supervisor run before starting another tick."],
        metadata={"worker_status": "running"},
    )

    store.save("boss-live-proof-tranche-2026-03-19", artifact)
    loaded = store.load("boss-live-proof-tranche-2026-03-19", "codex_a_live_gate")

    assert loaded is not None
    assert loaded.lane_id == artifact.lane_id
    assert loaded.urls == artifact.urls
    assert loaded.run_id == artifact.run_id
    assert loaded.worktree_path == artifact.worktree_path
    assert loaded.next_actions == artifact.next_actions
