from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.dev_coordination import LeaseConflictError
from aragora.swarm.tranche import (
    TrancheArtifactStore,
    TrancheExecutor,
    TrancheInspector,
    TrancheLaneArtifact,
    TrancheLane,
    TrancheManifest,
    TranchePlanner,
    _lane_review_model,
    _lane_spec_from_manifest,
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


def test_plan_from_prompt_bundle_compiles_manifest_and_derives_source_refs(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    bundle = tmp_path / "prompt-pack.yaml"
    bundle.write_text(
        """
bundle_id: pmf-tranche
objective: Ship the PMF tranche
lanes:
  - lane_id: pmf_impl
    owner_role: critical_path_engineer
    title: Implement PMF path
    prompt: |
      Make the end-to-end PMF flow work.
    source_refs:
      - https://github.com/synaptent/aragora/issues/1046
    target_agent: codex
    allowed_write_scope:
      - aragora/api/**
    verification_commands:
      - pytest tests/api/test_pmf.py -q
    stop_conditions:
      - needs_human returned
    expected_receipts_artifacts:
      - PR URL
""".strip(),
        encoding="utf-8",
    )
    planner = TranchePlanner(repo_root=repo)

    manifest, output_path = planner.plan_from_prompt_bundle(bundle)

    assert manifest.manifest_id == "pmf-tranche"
    assert output_path.exists()
    lane = manifest.lane("pmf_impl")
    assert lane.dependencies == ["issue_1046"]
    assert lane.metadata["prompt"].startswith("Make the end-to-end PMF flow work.")
    assert manifest.references["source_refs"]["issue_1046"].url.endswith("/issues/1046")


def test_prepare_creates_workspace_artifact_for_ready_lane(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    executor = TrancheExecutor(
        repo_root=repo,
        artifact_store=TrancheArtifactStore(repo),
        reference_client=_FakeReferenceClient(
            issues={
                1046: {
                    "number": 1046,
                    "state": "OPEN",
                    "closedAt": None,
                    "title": "PMF issue",
                    "labels": [],
                }
            }
        ),
    )

    def _fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if "codex_worktree_autopilot.py" in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{repo}/.worktrees/codex-auto/lane-1\n"
            )
        if cmd[:3] == ["git", "switch", "-C"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        raise AssertionError(f"Unexpected command: {cmd}")

    with patch("aragora.swarm.tranche.subprocess.run", side_effect=_fake_run):
        payload = executor.prepare(manifest, lane_id="pmf_impl")

    prepared = payload["prepared_lanes"][0]
    assert prepared["status"] == "prepared"
    assert prepared["metadata"]["branch"].startswith("codex/")
    assert prepared["worktree_path"].endswith("/lane-1")
    saved = executor.artifact_store.load(manifest.manifest_id, "pmf_impl")
    assert saved is not None
    assert saved.status == "prepared"
    assert saved.worktree_path == f"{repo}/.worktrees/codex-auto/lane-1"


def test_prepare_preserves_nested_base_branch_names(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(repo, "git", "update-ref", "refs/remotes/origin/release/v2", "HEAD")
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {
                "name": "synaptent/aragora",
                "root": str(repo),
                "base_ref": "origin/release/v2",
            },
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    executor = TrancheExecutor(
        repo_root=repo,
        artifact_store=TrancheArtifactStore(repo),
        reference_client=_FakeReferenceClient(
            issues={
                1046: {
                    "number": 1046,
                    "state": "OPEN",
                    "closedAt": None,
                    "title": "PMF issue",
                    "labels": [],
                }
            }
        ),
    )
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(list(cmd))
        if "codex_worktree_autopilot.py" in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{repo}/.worktrees/codex-auto/lane-release\n"
            )
        if cmd[:3] == ["git", "switch", "-C"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        raise AssertionError(f"Unexpected command: {cmd}")

    with patch("aragora.swarm.tranche.subprocess.run", side_effect=_fake_run):
        executor.prepare(manifest, lane_id="pmf_impl")

    branch_switch = next(cmd for cmd in commands if cmd[:3] == ["git", "switch", "-C"])
    assert branch_switch[-1] == "origin/release/v2"


@pytest.mark.asyncio
async def test_run_dispatches_prepared_lane_and_records_review(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                    "acceptance_criteria": ["pytest tests/api/test_pmf.py -q passes"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    store = TrancheArtifactStore(repo)
    store.save(
        manifest.manifest_id,
        TrancheLaneArtifact(
            lane_id="pmf_impl",
            source_ref="issue_1046",
            status="prepared",
            commands=["prepare"],
            urls=["https://github.com/synaptent/aragora/issues/1046"],
            worktree_path="/tmp/controller-worktree",
            metadata={"branch": "codex/pmf-impl", "target_agent": "codex"},
        ),
    )
    fake_reviewer = AsyncMock(
        return_value=SimpleNamespace(
            status="passed",
            findings=[],
            to_dict=lambda: {
                "required": True,
                "review_model": "claude",
                "status": "passed",
                "findings": [],
                "reviewed_at": "2026-03-19T00:00:00+00:00",
                "raw_review": {},
            },
        )
    )
    executor = TrancheExecutor(
        repo_root=repo,
        artifact_store=store,
        reference_client=_FakeReferenceClient(
            issues={
                1046: {
                    "number": 1046,
                    "state": "OPEN",
                    "closedAt": None,
                    "title": "PMF issue",
                    "labels": [],
                }
            }
        ),
        reviewer=SimpleNamespace(review=fake_reviewer),
    )

    with patch(
        "aragora.swarm.boss_loop.dispatch_bounded_spec",
        new=AsyncMock(
            return_value={
                "status": "completed",
                "outcome": "deliverable_created",
                "run_id": "run-123",
                "deliverable": {
                    "type": "pr",
                    "pr_url": "https://github.com/synaptent/aragora/pull/2000",
                },
                "run": {
                    "run_id": "run-123",
                    "status": "completed",
                    "work_orders": [{"worktree_path": "/tmp/worker-worktree"}],
                },
            }
        ),
    ):
        payload = await executor.run(manifest, lane_id="pmf_impl")

    result = payload["results"][0]
    assert result["status"] == "review_passed"
    assert result["run_id"] == "run-123"
    assert "https://github.com/synaptent/aragora/pull/2000" in result["urls"]
    saved = store.load(manifest.manifest_id, "pmf_impl")
    assert saved is not None
    assert saved.status == "review_passed"
    assert saved.worktree_path == "/tmp/worker-worktree"


@pytest.mark.asyncio
async def test_artifact_from_run_result_persists_receipt_and_lease_ids(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    executor = TrancheExecutor(repo_root=repo, artifact_store=TrancheArtifactStore(repo))
    prepared = TrancheLaneArtifact(
        lane_id="pmf_impl",
        source_ref="issue_1046",
        status="prepared",
        worktree_path="/tmp/controller-worktree",
        metadata={"branch": "codex/pmf-impl", "target_agent": "codex"},
    )

    artifact = await executor._artifact_from_run_result(
        manifest,
        manifest.lane("pmf_impl"),
        prepared=prepared,
        result={
            "status": "completed",
            "outcome": "deliverable_created",
            "run_id": "run-123",
            "run": {
                "run_id": "run-123",
                "status": "completed",
                "work_orders": [
                    {
                        "worktree_path": "/tmp/worker-worktree",
                        "receipt_id": "receipt-xyz",
                        "lease_id": "lease-abc",
                    }
                ],
            },
        },
        review_model="claude",
        skip_review=True,
    )

    assert artifact.metadata["receipt_id"] == "receipt-xyz"
    assert artifact.metadata["lease_id"] == "lease-abc"


@pytest.mark.asyncio
async def test_artifact_from_run_result_omits_raw_issue_text_when_sanitized(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    executor = TrancheExecutor(repo_root=repo, artifact_store=TrancheArtifactStore(repo))
    prepared = TrancheLaneArtifact(
        lane_id="pmf_impl",
        source_ref="issue_1046",
        status="prepared",
        worktree_path="/tmp/controller-worktree",
        metadata={"branch": "codex/pmf-impl", "target_agent": "codex"},
    )

    artifact = await executor._artifact_from_run_result(
        manifest,
        manifest.lane("pmf_impl"),
        prepared=prepared,
        result={
            "status": "needs_human",
            "outcome": "sanitation_failed",
            "run_id": "run-123",
            "sanitizer_outcome": "rewritten",
            "checks_failed": ["missing_validation"],
            "original_issue_body": "raw body",
            "sanitized_issue_body": "rewritten body",
            "reasons": ["needs update"],
            "next_actions": ["fix it"],
        },
        review_model="claude",
        skip_review=True,
    )

    assert artifact.metadata["issue_text"] == {
        "sanitized_body": "rewritten body",
        "changed": True,
        "sanitizer_outcome": "rewritten",
        "checks_failed": ["missing_validation"],
    }
    assert "original_body" not in artifact.metadata["issue_text"]


@pytest.mark.asyncio
async def test_artifact_from_run_result_keeps_single_copy_when_text_is_unchanged(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    executor = TrancheExecutor(repo_root=repo, artifact_store=TrancheArtifactStore(repo))
    prepared = TrancheLaneArtifact(
        lane_id="pmf_impl",
        source_ref="issue_1046",
        status="prepared",
        worktree_path="/tmp/controller-worktree",
        metadata={"branch": "codex/pmf-impl", "target_agent": "codex"},
    )

    artifact = await executor._artifact_from_run_result(
        manifest,
        manifest.lane("pmf_impl"),
        prepared=prepared,
        result={
            "status": "needs_human",
            "outcome": "sanitation_checked",
            "run_id": "run-123",
            "sanitizer_outcome": "accepted",
            "original_issue_body": "same body",
            "sanitized_issue_body": "same body",
            "reasons": ["needs update"],
            "next_actions": ["fix it"],
        },
        review_model="claude",
        skip_review=True,
    )

    assert artifact.metadata["issue_text"] == {
        "sanitized_body": "same body",
        "changed": False,
        "sanitizer_outcome": "accepted",
    }


def test_lane_spec_from_manifest_emits_explicit_single_work_order() -> None:
    from aragora.swarm.tranche import TrancheManifest, _lane_spec_from_manifest

    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "overnight",
            "repo": {
                "name": "synaptent/aragora",
                "root": "/tmp/repo",
                "base_ref": "origin/main",
            },
            "objective": "Ship one bounded overnight slice.",
            "lanes": [
                {
                    "lane_id": "product-loop-cli-slice",
                    "owner_role": "implementation_engineer",
                    "allowed_write_scope": ["aragora/cli/**", "tests/cli/**"],
                    "verification_commands": [
                        "python3 -m pytest tests/cli/test_api_key_command.py -q"
                    ],
                    "prompt": "Implement one bounded CLI slice.",
                    "title": "Product loop CLI slice",
                    "target_agent": "codex",
                    "review_model": "claude",
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )

    spec = _lane_spec_from_manifest(manifest, manifest.lane("product-loop-cli-slice"))

    assert len(spec.work_orders) == 1
    work_order = spec.work_orders[0]
    assert work_order["work_order_id"] == "product-loop-cli-slice"
    assert work_order["file_scope"] == ["aragora/cli/**", "tests/cli/**"]
    assert work_order["expected_tests"] == [
        "python3 -m pytest tests/cli/test_api_key_command.py -q"
    ]
    assert work_order["target_agent"] == "codex"
    assert work_order["reviewer_agent"] == "claude"


@pytest.mark.asyncio
async def test_run_reprepares_failed_artifact_before_dispatch(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    store = TrancheArtifactStore(repo)
    store.save(
        manifest.manifest_id,
        TrancheLaneArtifact(
            lane_id="pmf_impl",
            source_ref="issue_1046",
            status="failed",
            commands=["prepare"],
            urls=["https://github.com/synaptent/aragora/issues/1046"],
            worktree_path="/tmp/stale-worktree",
            metadata={"branch": "codex/stale", "target_agent": "codex"},
        ),
    )
    executor = TrancheExecutor(
        repo_root=repo,
        artifact_store=store,
        reference_client=_FakeReferenceClient(
            issues={
                1046: {
                    "number": 1046,
                    "state": "OPEN",
                    "closedAt": None,
                    "title": "PMF issue",
                    "labels": [],
                }
            }
        ),
    )
    fresh_prepared = TrancheLaneArtifact(
        lane_id="pmf_impl",
        source_ref="issue_1046",
        status="prepared",
        commands=["prepare"],
        urls=["https://github.com/synaptent/aragora/issues/1046"],
        worktree_path="/tmp/fresh-worktree",
        metadata={"branch": "codex/fresh", "target_agent": "codex"},
    )

    with (
        patch.object(
            executor,
            "_prepare_lane_workspace",
            return_value=fresh_prepared,
        ) as mock_prepare,
        patch(
            "aragora.swarm.boss_loop.dispatch_bounded_spec",
            new=AsyncMock(
                return_value={
                    "status": "running",
                    "run_id": "run-456",
                }
            ),
        ),
    ):
        payload = await executor.run(manifest, lane_id="pmf_impl", skip_review=True)

    mock_prepare.assert_called_once()
    result = payload["results"][0]
    assert result["status"] == "running"
    assert result["worktree_path"] == "/tmp/fresh-worktree"
    saved = store.load(manifest.manifest_id, "pmf_impl")
    assert saved is not None
    assert saved.worktree_path == "/tmp/fresh-worktree"


@pytest.mark.asyncio
async def test_run_disables_managed_session_script_by_default(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    store = TrancheArtifactStore(repo)
    store.save(
        manifest.manifest_id,
        TrancheLaneArtifact(
            lane_id="pmf_impl",
            source_ref="issue_1046",
            status="prepared",
            commands=["prepare"],
            urls=["https://github.com/synaptent/aragora/issues/1046"],
            worktree_path="/tmp/controller-worktree",
            metadata={"branch": "codex/pmf-impl", "target_agent": "codex"},
        ),
    )
    executor = TrancheExecutor(repo_root=repo, artifact_store=store)
    captured_kwargs: dict[str, object] = {}

    async def _dispatch(_: object, **kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {
            "status": "running",
            "run_id": "run-789",
        }

    with patch("aragora.swarm.boss_loop.dispatch_bounded_spec", side_effect=_dispatch):
        await executor.run(manifest, lane_id="pmf_impl", skip_review=True)

    assert captured_kwargs.get("use_managed_session_script") is True


@pytest.mark.asyncio
async def test_run_allows_explicit_managed_session_script_override(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manifest = TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": str(repo), "base_ref": "origin/main"},
            "references": {
                "source_refs": {
                    "issue_1046": {
                        "kind": "issue",
                        "url": "https://github.com/synaptent/aragora/issues/1046",
                        "state": "open",
                    }
                }
            },
            "gates": {},
            "lanes": [
                {
                    "lane_id": "pmf_impl",
                    "owner_role": "critical_path_engineer",
                    "title": "Implement PMF path",
                    "prompt": "Make the end-to-end PMF flow work.",
                    "target_agent": "codex",
                    "review_model": "claude",
                    "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
                    "allowed_write_scope": ["aragora/api/**"],
                    "dependencies": ["issue_1046"],
                    "verification_commands": ["pytest tests/api/test_pmf.py -q"],
                    "stop_conditions": ["needs_human returned"],
                    "expected_receipts_artifacts": ["PR URL"],
                    "use_managed_session_script": True,
                }
            ],
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )
    store = TrancheArtifactStore(repo)
    store.save(
        manifest.manifest_id,
        TrancheLaneArtifact(
            lane_id="pmf_impl",
            source_ref="issue_1046",
            status="prepared",
            commands=["prepare"],
            urls=["https://github.com/synaptent/aragora/issues/1046"],
            worktree_path="/tmp/controller-worktree",
            metadata={"branch": "codex/pmf-impl", "target_agent": "codex"},
        ),
    )
    executor = TrancheExecutor(repo_root=repo, artifact_store=store)
    captured_kwargs: dict[str, object] = {}

    async def _dispatch(_: object, **kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {
            "status": "running",
            "run_id": "run-790",
        }

    with patch("aragora.swarm.boss_loop.dispatch_bounded_spec", side_effect=_dispatch):
        await executor.run(manifest, lane_id="pmf_impl", skip_review=True)

    assert captured_kwargs.get("use_managed_session_script") is True


def test_lane_spec_coerces_string_boolean_metadata() -> None:
    manifest = TrancheManifest(
        manifest_id="m1",
        repo={"name": "synaptent/aragora", "root": "/tmp/repo", "base_ref": "origin/main"},
        references={},
        gates={},
        lanes=[],
        terminal_outcomes={"success": {"definition": "done"}},
    )
    lane = TrancheLane(
        lane_id="lane-a",
        owner_role="engineer",
        allowed_write_scope=["aragora/swarm/**"],
        verification_commands=[],
        metadata={
            "prompt": "Ship the fix.",
            "target_agent": "codex",
            "review_model": "codex",
            "requires_approval": "false",
            "enforce_cross_model_review": "false",
        },
    )

    spec = _lane_spec_from_manifest(manifest, lane)

    assert spec.work_orders[0]["approval_required"] is False
    assert spec.requires_approval is False
    assert _lane_review_model(lane, target_agent="codex") == "codex"
