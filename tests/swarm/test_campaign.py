from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.nomic.task_decomposer import SubTask
from aragora.swarm.campaign import (
    CampaignDependency,
    CampaignExecutionState,
    CampaignManifest,
    CampaignPlanner,
    CampaignProject,
    CampaignProjectStatus,
    CampaignReviewGate,
    CampaignReviewStatus,
    CampaignRunOutcome,
    CampaignStopReason,
    CampaignExecutor,
    _compute_stop_reason,
    load_campaign_manifest,
    save_campaign_manifest,
)
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.supervisor import SwarmSupervisor
from aragora.swarm.worker_launcher import WorkerLauncher, WorkerProcess

UTC = timezone.utc
from aragora.swarm.supervisor import (
    CAMPAIGN_BLOCKERS_METADATA_KEY,
    CAMPAIGN_OUTCOME_METADATA_KEY,
    CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY,
)


def _bounded_spec(goal: str, scope: list[str] | None = None) -> SwarmSpec:
    return SwarmSpec(
        raw_goal=goal,
        refined_goal=goal,
        acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
        constraints=["do not widen scope"],
        file_scope_hints=scope or ["aragora/swarm/campaign.py"],
        budget_limit_usd=5.0,
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


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


async def _record_timeout_run(repo: Path) -> str:
    store = DevCoordinationStore(repo_root=repo)
    head = _head(repo)
    stale = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    run_record = store.create_supervisor_run(
        goal="timeout campaign bridge",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "timeout campaign bridge"},
        work_orders=[
            {
                "work_order_id": "wo-timeout",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "pid": 4242,
                "initial_head": head,
                "dispatched_at": stale,
                "last_progress_at": stale,
                "progress_fingerprint": {
                    "head_sha": head,
                    "changed_paths": [],
                    "diff_lines": 0,
                },
            }
        ],
        status="active",
    )

    launcher = MagicMock(spec=WorkerLauncher)
    launcher.collect_finished = AsyncMock(return_value=[])
    launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": True,
            "head_sha": head,
            "changed_paths": [],
            "diff_lines": 0,
        }
    )
    launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=60.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)
    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        await supervisor.collect_finished_results(run_record["run_id"])
    return run_record["run_id"]


async def _record_crash_run(repo: Path) -> str:
    store = DevCoordinationStore(repo_root=repo)
    head = _head(repo)
    run_record = store.create_supervisor_run(
        goal="crash campaign bridge",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "crash campaign bridge"},
        work_orders=[
            {
                "work_order_id": "wo-crash",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "initial_head": head,
            }
        ],
        status="active",
    )

    launcher = MagicMock(spec=WorkerLauncher)
    launcher.collect_finished = AsyncMock(
        return_value=[
            WorkerProcess(
                work_order_id="wo-crash",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                exit_code=1,
                changed_paths=[],
                commit_shas=[],
                head_sha=head,
                stderr="worker crashed",
            )
        ]
    )
    launcher.config = SimpleNamespace(auto_commit=False, no_progress_timeout_seconds=60.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)
    await supervisor.collect_finished_results(run_record["run_id"])
    return run_record["run_id"]


class TestCampaignPlanner:
    def test_plan_from_source_file_is_deterministic(self, tmp_path: Path) -> None:
        source = tmp_path / "roadmap.md"
        source.write_text(
            "- Add retry ledger in aragora/swarm/campaign.py\n"
            "- Improve status output in aragora/cli/commands/swarm.py\n",
            encoding="utf-8",
        )
        planner = CampaignPlanner(repo_root=tmp_path)

        manifest_a = planner.plan_from_source_file(source)
        manifest_b = planner.plan_from_source_file(source)

        ids_a = [project.project_id for project in manifest_a.projects]
        ids_b = [project.project_id for project in manifest_b.projects]
        assert ids_a == ids_b
        assert [project.title for project in manifest_a.projects] == [
            project.title for project in manifest_b.projects
        ]

    def test_plan_splits_decomposed_items_and_sorts_dependencies(self, tmp_path: Path) -> None:
        planner = CampaignPlanner(repo_root=tmp_path)
        sub_a = SimpleNamespace(
            id="a",
            title="Add manifest model",
            description="Add manifest model",
            dependencies=[],
            estimated_complexity="medium",
            file_scope=["aragora/swarm/campaign.py"],
            success_criteria={"tests": "campaign tests pass"},
        )
        sub_b = SimpleNamespace(
            id="b",
            title="Wire CLI",
            description="Wire CLI",
            dependencies=["a"],
            estimated_complexity="medium",
            file_scope=["aragora/cli/commands/swarm.py"],
            success_criteria={"tests": "CLI tests pass"},
        )
        decomposition = SimpleNamespace(
            should_decompose=True,
            subtasks=[sub_b, sub_a],
            complexity_level="high",
        )
        planner.decomposer = MagicMock()
        planner.decomposer.analyze.return_value = decomposition

        manifest = planner.plan_from_items(
            ["Implement campaign support"], source_kind="source_file", source_ref="roadmap.md"
        )

        assert [project.title for project in manifest.projects] == [
            "Add manifest model",
            "Wire CLI",
        ]
        assert manifest.projects[1].dependencies == [
            CampaignDependency(project_id="proj-001", reason="subtask_dependency")
        ]

    def test_overlap_adds_dependency_finding(self, tmp_path: Path) -> None:
        planner = CampaignPlanner(repo_root=tmp_path)
        planner.decomposer = MagicMock()
        planner.decomposer.analyze.side_effect = [
            SimpleNamespace(should_decompose=False, subtasks=[], complexity_level="medium"),
            SimpleNamespace(should_decompose=False, subtasks=[], complexity_level="medium"),
        ]

        manifest = planner.plan_from_items(
            [
                "Update aragora/swarm/campaign.py manifest persistence",
                "Refine aragora/swarm/campaign.py review gate",
            ],
            source_kind="source_file",
            source_ref="plan.md",
        )

        assert any("overlapping scope" in finding for finding in manifest.planning_findings)
        assert manifest.projects[1].dependencies[0].project_id == manifest.projects[0].project_id

    def test_model_planner_falls_back_to_heuristic_and_records_finding(
        self, tmp_path: Path
    ) -> None:
        planner = CampaignPlanner(
            repo_root=tmp_path,
            planner_model="codex",
            planner_strategy="model",
        )
        planner.decomposer = MagicMock()
        planner.decomposer.analyze_with_model_sync.side_effect = RuntimeError("planner timeout")
        planner.decomposer.analyze.return_value = SimpleNamespace(
            should_decompose=False,
            subtasks=[],
            complexity_level="medium",
        )

        manifest = planner.plan_from_items(
            ["Enable quality gates in aragora/nomic/hardened_orchestrator.py"],
            source_kind="source_file",
            source_ref="roadmap.md",
        )

        planner.decomposer.analyze_with_model_sync.assert_called_once()
        planner.decomposer.analyze.assert_called_once()
        assert any(
            "planner fallback to heuristic" in finding for finding in manifest.planning_findings
        )


class TestCampaignManifestIO:
    def test_round_trip_yaml(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-test",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Add campaign manifest",
                    spec=_bounded_spec("Add campaign manifest"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                )
            ],
            execution_state=CampaignExecutionState(),
        )
        save_campaign_manifest(manifest_path, manifest)
        loaded = load_campaign_manifest(manifest_path)

        assert loaded.campaign_id == "campaign-test"
        assert loaded.projects[0].project_id == "proj-001"
        assert loaded.projects[0].spec.is_dispatch_bounded() is True

    def test_same_model_review_requires_opt_in(self) -> None:
        enforced = CampaignManifest(
            campaign_id="campaign-enforced",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            worker_model="codex",
            review_model="codex",
            enforce_cross_model_review=True,
        )
        allowed = CampaignManifest(
            campaign_id="campaign-allowed",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            worker_model="codex",
            review_model="codex",
            enforce_cross_model_review=False,
        )

        assert enforced.review_model == "claude"
        assert allowed.review_model == "codex"


class TestCampaignExecutor:
    @pytest.mark.asyncio
    async def test_execute_once_uses_model_planner_to_create_explicit_work_orders(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-model-planner",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            planner_model="claude",
            planner_strategy="model",
            worker_model="claude",
            review_model="claude",
            enforce_cross_model_review=False,
            experiment_id="exp-001",
            experiment_label="claude-all-the-way",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Enable quality gates by default",
                    spec=_bounded_spec(
                        "Enable quality gates by default",
                        ["aragora/nomic/hardened_orchestrator.py"],
                    ),
                    file_scope_hints=["aragora/nomic/hardened_orchestrator.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)
        executor.decomposer.analyze_with_model = AsyncMock(
            return_value=SimpleNamespace(
                subtasks=[
                    SubTask(
                        id="subtask_1",
                        title="Planner lane",
                        description="Wire default quality gates",
                        dependencies=[],
                        estimated_complexity="medium",
                        file_scope=["aragora/nomic/hardened_orchestrator.py"],
                        success_criteria={
                            "tests": "python -m pytest tests/swarm/test_campaign.py -q"
                        },
                    )
                ]
            )
        )

        with patch(
            "aragora.swarm.campaign.dispatch_bounded_spec",
            new=AsyncMock(
                return_value={
                    "status": "needs_human",
                    "outcome": CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
                    "run_id": "run-model-planner",
                    "run": {
                        "run_id": "run-model-planner",
                        "status": "completed",
                        "work_orders": [],
                    },
                }
            ),
        ) as mock_dispatch:
            await executor.execute_once()

        passed_spec = mock_dispatch.await_args.args[0]
        assert passed_spec.work_orders
        work_order = passed_spec.work_orders[0]
        assert work_order["target_agent"] == "claude"
        assert work_order["reviewer_agent"] == "claude"
        assert work_order["metadata"]["planner_strategy_requested"] == "model"
        assert work_order["metadata"]["planner_strategy_used"] == "model"
        assert work_order["metadata"]["experiment_id"] == "exp-001"
        executor.decomposer.analyze_with_model.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_once_redispatches_needs_revision_with_review_findings(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-review-retry",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Retry with review findings",
                    spec=_bounded_spec("Retry with review findings"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.NEEDS_REVISION.value,
                    review=CampaignReviewGate(
                        required=True,
                        review_model="claude",
                        status=CampaignReviewStatus.CHANGES_REQUESTED.value,
                        findings=["Preserve ticket auditability."],
                    ),
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        dispatch = AsyncMock(
            return_value={
                "status": "needs_human",
                "outcome": CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
                "run_id": "run-review-retry",
                "run": {
                    "run_id": "run-review-retry",
                    "status": "completed",
                    "work_orders": [],
                },
            }
        )
        with patch("aragora.swarm.campaign.dispatch_bounded_spec", new=dispatch):
            payload = await executor.execute_once()

        retry_spec = dispatch.await_args.args[0]
        assert (
            "Address prior review finding: Preserve ticket auditability." in retry_spec.constraints
        )
        assert payload["dispatched_projects"][0]["project_id"] == "proj-001"

    @pytest.mark.asyncio
    async def test_execute_once_records_completed_project_after_review(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-exec",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            worker_model="codex",
            review_model="claude",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Implement manifest",
                    spec=_bounded_spec("Implement manifest"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    estimated_cost_usd=1.0,
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        dispatch_result = {
            "status": "completed",
            "outcome": CampaignRunOutcome.DELIVERABLE_CREATED.value,
            "run_id": "run-123",
            "deliverable": {"type": "pr", "pr_url": "https://github.com/example/pull/1"},
            "run": {
                "run_id": "run-123",
                "status": "completed",
                "work_orders": [
                    {
                        "status": "completed",
                        "pr_url": "https://github.com/example/pull/1",
                        "receipt_id": "receipt-1",
                    }
                ],
            },
        }
        review_gate = CampaignReviewGate(
            required=True,
            review_model="claude",
            status=CampaignReviewStatus.PASSED.value,
            findings=[],
        )

        with (
            patch(
                "aragora.swarm.campaign.dispatch_bounded_spec",
                new=AsyncMock(return_value=dispatch_result),
            ),
            patch.object(executor.reviewer, "review", new=AsyncMock(return_value=review_gate)),
        ):
            payload = await executor.execute_once()

        reloaded = load_campaign_manifest(manifest_path)
        project = reloaded.projects[0]
        assert project.status == CampaignProjectStatus.COMPLETED.value
        assert project.run_id == "run-123"
        assert project.pr_url == "https://github.com/example/pull/1"
        assert project.review.status == CampaignReviewStatus.PASSED.value
        assert payload["dispatched_projects"][0]["project_id"] == "proj-001"

    @pytest.mark.asyncio
    async def test_execute_once_halts_before_dispatch_when_remaining_budget_is_insufficient(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-budget-preflight",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            budget_limit_usd=1.0,
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Would exceed remaining budget",
                    spec=_bounded_spec("Would exceed remaining budget"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    estimated_cost_usd=0.50,
                )
            ],
            execution_state=CampaignExecutionState(total_cost_usd=0.75),
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        with patch(
            "aragora.swarm.campaign.dispatch_bounded_spec",
            new=AsyncMock(
                return_value={
                    "status": "unexpected-dispatch",
                    "outcome": CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
                    "run_id": "run-unexpected",
                    "run": {
                        "run_id": "run-unexpected",
                        "status": "completed",
                        "work_orders": [],
                    },
                }
            ),
        ) as mock_dispatch:
            payload = await executor.execute_once()

        reloaded = load_campaign_manifest(manifest_path)
        assert mock_dispatch.await_count == 0
        assert payload["stop_reason"] == CampaignStopReason.BUDGET_EXHAUSTED.value
        assert payload["dispatched_projects"] == []
        assert reloaded.execution_state.total_cost_usd == pytest.approx(0.75)
        assert reloaded.projects[0].status in {
            CampaignProjectStatus.PENDING.value,
            CampaignProjectStatus.READY.value,
        }

    @pytest.mark.asyncio
    async def test_execute_once_marks_clean_exit_no_deliverable_needs_revision(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-clean-exit",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Do thing",
                    spec=_bounded_spec("Do thing"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        with patch(
            "aragora.swarm.campaign.dispatch_bounded_spec",
            new=AsyncMock(
                return_value={
                    "status": "needs_human",
                    "outcome": CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
                    "run_id": "run-456",
                    "run": {"run_id": "run-456", "status": "completed", "work_orders": []},
                }
            ),
        ):
            await executor.execute_once()

        project = load_campaign_manifest(manifest_path).projects[0]
        assert project.status == CampaignProjectStatus.NEEDS_REVISION.value
        assert project.last_run_outcome == CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value
        assert project.retry_count == 1

    @pytest.mark.asyncio
    async def test_execute_once_halts_before_over_budget_dispatch(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-budget-cap",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            budget_limit_usd=1.0,
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Too expensive",
                    spec=_bounded_spec("Too expensive"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    estimated_cost_usd=2.0,
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        with patch(
            "aragora.swarm.campaign.dispatch_bounded_spec", new=AsyncMock()
        ) as mock_dispatch:
            payload = await executor.execute_once()

        reloaded = load_campaign_manifest(manifest_path)
        status_payload = executor.status()

        assert mock_dispatch.await_count == 0
        assert payload["stop_reason"] == CampaignStopReason.BUDGET_EXHAUSTED.value
        assert payload["budget_blocked_projects"] == ["proj-001"]
        assert payload["budget"]["available_budget_usd"] == 1.0
        assert reloaded.projects[0].status in {
            CampaignProjectStatus.PENDING.value,
            CampaignProjectStatus.READY.value,
        }
        assert status_payload["stop_reason"] == CampaignStopReason.BUDGET_EXHAUSTED.value
        assert status_payload["budget"]["available_budget_usd"] == 1.0
        assert status_payload["projects"][0]["estimated_cost_usd"] == 2.0

    def test_reconcile_active_needs_human_run_blocks_and_emits_receipt(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        store = DevCoordinationStore(repo_root=repo)
        run_record = store.create_supervisor_run(
            goal="needs human campaign bridge",
            target_branch="main",
            supervisor_agents={},
            approval_policy={},
            spec={"raw_goal": "needs human campaign bridge"},
            work_orders=[
                {
                    "work_order_id": "wo-human",
                    "status": "needs_human",
                    "worktree_path": str(repo),
                    "branch": "main",
                    "target_agent": "codex",
                    "dispatch_error": "Worker requires human input.",
                }
            ],
            status="active",
        )

        manifest_path = repo / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="phase0b-needs-human",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Blocked run",
                    spec=_bounded_spec("Blocked run", ["aragora/swarm/campaign.py"]),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id=run_record["run_id"],
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=repo)

        executor._reconcile_active_projects(manifest)

        project = manifest.projects[0]
        receipt_path = repo / "docs" / "receipts" / "phase0b-needs-human" / "proj-001.yaml"
        assert project.status == CampaignProjectStatus.BLOCKED.value
        assert project.last_run_outcome == CampaignRunOutcome.NEEDS_HUMAN.value
        assert project.receipt_id == "docs/receipts/phase0b-needs-human/proj-001.yaml"
        assert receipt_path.exists()

    @pytest.mark.asyncio
    async def test_reconcile_active_crash_run_uses_supervisor_failure_metadata(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        run_id = await _record_crash_run(repo)
        manifest_path = repo / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="phase0b-crash",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            max_retries_per_project=2,
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Crash recovery",
                    spec=_bounded_spec("Crash recovery", ["aragora/swarm/campaign.py"]),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id=run_id,
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=repo)

        executor._reconcile_active_projects(manifest)

        project = manifest.projects[0]
        assert project.status == CampaignProjectStatus.NEEDS_REVISION.value
        assert project.last_run_outcome == CampaignRunOutcome.CRASH.value
        assert project.receipt_id is None
        assert project.attempt_history[-1]["requeue_eligible"] is True
        assert project.attempt_history[-1]["failure_detail"] == "worker crashed"

    @pytest.mark.asyncio
    async def test_reconcile_active_no_progress_timeout_surfaces_stalled_state(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        run_id = await _record_timeout_run(repo)
        manifest_path = repo / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="phase0b-timeout",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            max_retries_per_project=2,
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Timeout recovery",
                    spec=_bounded_spec("Timeout recovery", ["aragora/swarm/campaign.py"]),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id=run_id,
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=repo)

        executor._reconcile_active_projects(manifest)

        project = manifest.projects[0]
        receipt_path = repo / "docs" / "receipts" / "phase0b-timeout" / "proj-001.yaml"

        assert project.status == CampaignProjectStatus.STALLED.value
        assert project.last_run_outcome == CampaignRunOutcome.STALLED.value
        assert project.receipt_id == "docs/receipts/phase0b-timeout/proj-001.yaml"
        assert receipt_path.exists()
        assert project.attempt_history[-1]["requeue_eligible"] is False
        assert "no-progress timeout" in project.attempt_history[-1]["failure_detail"]
        assert _compute_stop_reason(manifest) == CampaignStopReason.CAMPAIGN_STALLED.value

    @pytest.mark.asyncio
    async def test_execute_once_reconciles_active_project_without_redispatch(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-reconcile",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Active project",
                    spec=_bounded_spec("Active project"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id="run-live",
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        with (
            patch.object(
                executor,
                "_refresh_run_dict",
                return_value={
                    "run_id": "run-live",
                    "status": "completed",
                    "work_orders": [
                        {
                            "status": "completed",
                            "adopted_pr": "#857",
                            "receipt_id": "receipt-live",
                        }
                    ],
                },
            ),
            patch("aragora.swarm.campaign.dispatch_bounded_spec", new=AsyncMock()) as mock_dispatch,
        ):
            payload = await executor.execute_once()

        project = load_campaign_manifest(manifest_path).projects[0]
        assert mock_dispatch.await_count == 0
        assert project.last_run_outcome == CampaignRunOutcome.PR_ADOPTED.value
        assert payload["stop_reason"] in {"still_running", "campaign_blocked", "campaign_complete"}

    @pytest.mark.asyncio
    async def test_execute_once_surfaces_waiting_conflict_deadlock_as_campaign_stalled(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-deadlock",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Deadlocked project",
                    spec=_bounded_spec("Deadlocked project"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id="run-deadlock",
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Downstream blocked by deadlock",
                    spec=_bounded_spec("Downstream blocked by deadlock"),
                    file_scope_hints=["aragora/swarm/reconciler.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_reconciler.py"],
                    constraints=["do not widen scope"],
                    dependencies=[
                        CampaignDependency(project_id="proj-001", reason="subtask_dependency")
                    ],
                ),
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        with patch.object(
            executor,
            "_refresh_run_dict",
            return_value={
                "run_id": "run-deadlock",
                "status": "needs_human",
                "work_orders": [
                    {"status": "completed"},
                    {"status": "waiting_conflict"},
                    {"status": "waiting_conflict"},
                    {"status": "failed"},
                ],
            },
        ):
            payload = await executor.execute_once()

        reloaded = load_campaign_manifest(manifest_path)
        assert reloaded.projects[0].status == CampaignProjectStatus.STALLED.value
        assert reloaded.projects[0].last_run_outcome == CampaignRunOutcome.STALLED.value
        assert payload["stop_reason"] == CampaignStopReason.CAMPAIGN_STALLED.value

    @pytest.mark.asyncio
    async def test_execute_once_requeues_timeout_outcome_and_preserves_attempt_audit(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-timeout-retry",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            worker_model="codex",
            review_model="claude",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Recoverable timeout",
                    spec=_bounded_spec("Recoverable timeout"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    estimated_cost_usd=1.0,
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        timeout_result = {
            "status": "failed",
            "outcome": CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
            "run_id": "run-timeout-1",
            "run": {
                "run_id": "run-timeout-1",
                "status": "completed",
                "metadata": {
                    CAMPAIGN_OUTCOME_METADATA_KEY: CampaignRunOutcome.TIMEOUT.value,
                    CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY: True,
                    CAMPAIGN_BLOCKERS_METADATA_KEY: ["worker exceeded no-progress timeout (60s)"],
                },
                "work_orders": [
                    {
                        "status": "needs_human",
                        "worker_outcome": "timeout_no_progress",
                        "dispatch_error": "worker exceeded no-progress timeout (60s)",
                        "receipt_id": "worker-receipt-1",
                    }
                ],
            },
        }
        success_result = {
            "status": "completed",
            "outcome": CampaignRunOutcome.DELIVERABLE_CREATED.value,
            "run_id": "run-success-2",
            "deliverable": {"type": "pr", "pr_url": "https://github.com/example/pull/2"},
            "run": {
                "run_id": "run-success-2",
                "status": "completed",
                "work_orders": [
                    {
                        "status": "completed",
                        "pr_url": "https://github.com/example/pull/2",
                        "receipt_id": "worker-receipt-2",
                    }
                ],
            },
        }
        review_gate = CampaignReviewGate(
            required=True,
            review_model="claude",
            status=CampaignReviewStatus.PASSED.value,
            findings=[],
        )

        with (
            patch(
                "aragora.swarm.campaign.dispatch_bounded_spec",
                new=AsyncMock(side_effect=[timeout_result, success_result]),
            ),
            patch.object(executor.reviewer, "review", new=AsyncMock(return_value=review_gate)),
        ):
            first_payload = await executor.execute_once()
            first_project = load_campaign_manifest(manifest_path).projects[0]
            status_payload = executor.status()
            second_payload = await executor.execute_once()

        assert first_payload["stop_reason"] == CampaignStopReason.STILL_RUNNING.value
        assert first_project.status == CampaignProjectStatus.NEEDS_REVISION.value
        assert first_project.last_run_outcome == CampaignRunOutcome.TIMEOUT.value
        assert first_project.worker_receipt_id == "worker-receipt-1"
        assert len(first_project.attempt_history) == 1
        assert first_project.attempt_history[0]["requeue_eligible"] is True
        assert "no-progress timeout" in first_project.attempt_history[0]["failure_detail"]
        assert status_payload["projects"][0]["last_run_outcome"] == CampaignRunOutcome.TIMEOUT.value
        assert status_payload["projects"][0]["recovery_eligible"] is True
        assert "no-progress timeout" in status_payload["projects"][0]["last_failure_detail"]

        reloaded = load_campaign_manifest(manifest_path).projects[0]
        assert second_payload["stop_reason"] in {
            CampaignStopReason.STILL_RUNNING.value,
            CampaignStopReason.CAMPAIGN_COMPLETE.value,
        }
        assert reloaded.status == CampaignProjectStatus.COMPLETED.value
        assert reloaded.pr_url == "https://github.com/example/pull/2"
        assert reloaded.worker_receipt_id == "worker-receipt-2"
        assert len(reloaded.attempt_history) == 2
        assert reloaded.attempt_history[0]["outcome"] == CampaignRunOutcome.TIMEOUT.value
        assert (
            reloaded.attempt_history[1]["outcome"] == CampaignRunOutcome.DELIVERABLE_CREATED.value
        )

    @pytest.mark.asyncio
    async def test_execute_once_returns_still_running_when_active_projects_exist(
        self, tmp_path: Path
    ) -> None:
        """Finding 1: active in-flight projects must yield still_running, not campaign_blocked."""
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-inflight",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="In-flight project",
                    spec=_bounded_spec("In-flight project"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["tests pass"],
                    constraints=["stay in scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id="run-inflight",
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Blocked downstream",
                    spec=_bounded_spec("Blocked downstream", ["aragora/cli/commands/swarm.py"]),
                    file_scope_hints=["aragora/cli/commands/swarm.py"],
                    acceptance_criteria=["tests pass"],
                    constraints=["stay in scope"],
                    dependencies=[
                        CampaignDependency(project_id="proj-001", reason="subtask_dependency")
                    ],
                ),
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        # Run is still in progress — not terminal
        with patch.object(
            executor,
            "_refresh_run_dict",
            return_value={
                "run_id": "run-inflight",
                "status": "running",
                "work_orders": [],
            },
        ):
            payload = await executor.execute_once()

        assert payload["stop_reason"] == "still_running"
        assert payload["dispatched_projects"] == []
        # Project should still be active, not blocked
        reloaded = load_campaign_manifest(manifest_path)
        assert reloaded.projects[0].status == CampaignProjectStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_execute_once_keeps_running_while_active_budget_is_reserved(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-budget-reserved",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            budget_limit_usd=1.0,
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="In-flight expensive task",
                    spec=_bounded_spec("In-flight expensive task"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["tests pass"],
                    constraints=["stay in scope"],
                    status=CampaignProjectStatus.ACTIVE.value,
                    run_id="run-expensive",
                    estimated_cost_usd=0.75,
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Ready but unaffordable",
                    spec=_bounded_spec("Ready but unaffordable", ["docs/CLI_REFERENCE.md"]),
                    file_scope_hints=["docs/CLI_REFERENCE.md"],
                    acceptance_criteria=["tests pass"],
                    constraints=["stay in scope"],
                    estimated_cost_usd=0.50,
                ),
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        with (
            patch.object(
                executor,
                "_refresh_run_dict",
                return_value={"run_id": "run-expensive", "status": "running", "work_orders": []},
            ),
            patch("aragora.swarm.campaign.dispatch_bounded_spec", new=AsyncMock()) as mock_dispatch,
        ):
            payload = await executor.execute_once()

        status_payload = executor.status()

        assert mock_dispatch.await_count == 0
        assert payload["stop_reason"] == CampaignStopReason.STILL_RUNNING.value
        assert payload["budget_blocked_projects"] == ["proj-002"]
        assert payload["budget"]["reserved_cost_usd"] == 0.75
        assert payload["budget"]["available_budget_usd"] == 0.25
        assert status_payload["budget"]["reserved_cost_usd"] == 0.75
        assert status_payload["budget"]["available_budget_usd"] == 0.25

    def test_status_reports_invalid_manifest_truthfully(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "campaign_id: bad\nsource_kind: source_file\nsource_ref: x\nprojects:\n"
            "  - project_id: proj-001\n    title: bad\n    spec: {raw_goal: bad, refined_goal: bad}\n",
            encoding="utf-8",
        )
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)
        with pytest.raises(ValueError):
            executor.status()

    def test_status_exposes_budget_accounting_and_review_state(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-budget-status",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            budget_limit_usd=3.0,
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Reviewed project",
                    spec=_bounded_spec("Reviewed project"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                    constraints=["do not widen scope"],
                    status=CampaignProjectStatus.DELIVERED.value,
                    review=CampaignReviewGate(
                        required=True,
                        review_model="claude",
                        status=CampaignReviewStatus.PENDING.value,
                    ),
                )
            ],
            execution_state=CampaignExecutionState(total_cost_usd=1.25),
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        status = executor.status()

        assert status["budget_limit_usd"] == pytest.approx(3.0)
        assert status["total_cost_usd"] == pytest.approx(1.25)
        assert status["projects"][0]["review_status"] == CampaignReviewStatus.PENDING.value


class TestComputeStopReason:
    """Dogfood finding F2: status and run must agree on blocked state."""

    def test_unreachable_pending_projects_are_campaign_blocked(self) -> None:
        """Pending projects whose deps are skipped/failed are unreachable."""
        manifest = CampaignManifest(
            campaign_id="campaign-unreachable",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Head project",
                    spec=_bounded_spec("Head project"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.SKIPPED.value,
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Downstream",
                    spec=_bounded_spec("Downstream", ["docs/CLI_REFERENCE.md"]),
                    file_scope_hints=["docs/CLI_REFERENCE.md"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.PENDING.value,
                    dependencies=[CampaignDependency(project_id="proj-001", reason="sequential")],
                ),
            ],
        )
        assert _compute_stop_reason(manifest) == CampaignStopReason.CAMPAIGN_BLOCKED.value

    def test_reachable_pending_projects_are_still_running(self) -> None:
        """Pending projects with no deps or all-completed deps are reachable."""
        manifest = CampaignManifest(
            campaign_id="campaign-reachable",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Completed head",
                    spec=_bounded_spec("Completed head"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.COMPLETED.value,
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Ready downstream",
                    spec=_bounded_spec("Ready downstream", ["docs/CLI_REFERENCE.md"]),
                    file_scope_hints=["docs/CLI_REFERENCE.md"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.PENDING.value,
                    dependencies=[CampaignDependency(project_id="proj-001", reason="sequential")],
                ),
            ],
        )
        assert _compute_stop_reason(manifest) == CampaignStopReason.STILL_RUNNING.value

    def test_mixed_skipped_and_independent_pending_is_still_running(self) -> None:
        """A dependency-free pending project keeps the campaign alive."""
        manifest = CampaignManifest(
            campaign_id="campaign-mixed",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Skipped head",
                    spec=_bounded_spec("Skipped head"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.SKIPPED.value,
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Independent project",
                    spec=_bounded_spec("Independent project", ["docs/FAQ.md"]),
                    file_scope_hints=["docs/FAQ.md"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.PENDING.value,
                    dependencies=[],
                ),
            ],
        )
        assert _compute_stop_reason(manifest) == CampaignStopReason.STILL_RUNNING.value

    def test_status_agrees_with_run_on_unreachable_blocked(self, tmp_path: Path) -> None:
        """End-to-end: executor.status() must report campaign_blocked for unreachable projects."""
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-status-blocked",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Skipped head",
                    spec=_bounded_spec("Skipped head"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.SKIPPED.value,
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Unreachable downstream",
                    spec=_bounded_spec("Unreachable downstream", ["docs/CLI_REFERENCE.md"]),
                    file_scope_hints=["docs/CLI_REFERENCE.md"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.PENDING.value,
                    dependencies=[CampaignDependency(project_id="proj-001", reason="sequential")],
                ),
            ],
        )
        save_campaign_manifest(manifest_path, manifest)
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)
        status = executor.status()
        assert status["stop_reason"] == CampaignStopReason.CAMPAIGN_BLOCKED.value

    def test_stalled_project_stop_reason_is_distinct_from_campaign_blocked(self) -> None:
        manifest = CampaignManifest(
            campaign_id="campaign-stalled",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Stalled head",
                    spec=_bounded_spec("Stalled head"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.STALLED.value,
                    last_run_outcome=CampaignRunOutcome.STALLED.value,
                ),
                CampaignProject(
                    project_id="proj-002",
                    title="Dependent project",
                    spec=_bounded_spec("Dependent project", ["docs/CLI_REFERENCE.md"]),
                    file_scope_hints=["docs/CLI_REFERENCE.md"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                    status=CampaignProjectStatus.PENDING.value,
                    dependencies=[CampaignDependency(project_id="proj-001", reason="sequential")],
                ),
            ],
        )

        assert _compute_stop_reason(manifest) == CampaignStopReason.CAMPAIGN_STALLED.value


class TestCampaignCLI:
    def _args(self, **overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "swarm_action_or_goal": "campaign",
            "swarm_goal": "status",
            "swarm_campaign_target": None,
            "spec": None,
            "skip_interrogation": False,
            "dry_run": False,
            "budget_limit": 50.0,
            "require_approval": False,
            "save_spec": None,
            "from_obsidian": None,
            "obsidian_vault": None,
            "no_obsidian_receipts": False,
            "profile": "developer",
            "autonomy": "propose",
            "max_parallel": 20,
            "no_loop": False,
            "target_branch": "main",
            "concurrency_cap": 8,
            "managed_dir_pattern": ".worktrees/{agent}-auto",
            "json": True,
            "run_id": None,
            "status_limit": 20,
            "refresh_scaling": False,
            "no_dispatch": False,
            "watch": False,
            "interval_seconds": 0.0,
            "max_ticks": 2,
            "all_runs": False,
            "dispatch_only": False,
            "no_wait": False,
            "freshness_ttl": 3600.0,
            "boss_repo": None,
            "boss_label_filter": None,
            "max_consecutive_failures": 3,
            "source_file": None,
            "issue_list": None,
            "github_query": None,
            "planner_model": "claude",
            "worker_model": "codex",
            "review_model": "claude",
            "manifest": "unused.yaml",
            "output": None,
            "max_parallel_ready_projects": 1,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_swarm_parser_accepts_campaign_plan(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "swarm",
                "campaign",
                "plan",
                "--source-file",
                "ROADMAP.md",
                "--manifest",
                ".aragora/campaign_manifest.yaml",
                "--planner-model",
                "claude",
                "--worker-model",
                "codex",
                "--review-model",
                "claude",
            ]
        )
        assert args.swarm_action_or_goal == "campaign"
        assert args.swarm_goal == "plan"
        assert args.source_file == "ROADMAP.md"

    def test_swarm_parser_accepts_campaign_run(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "swarm",
                "campaign",
                "run",
                "--source-file",
                "ROADMAP.md",
                "--worker-model",
                "codex",
                "--review-model",
                "claude",
            ]
        )
        assert args.swarm_action_or_goal == "campaign"
        assert args.swarm_goal == "run"
        assert args.source_file == "ROADMAP.md"

    def test_swarm_parser_accepts_campaign_experiment_flags(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "swarm",
                "campaign",
                "plan",
                "--source-file",
                "ROADMAP.md",
                "--planner-strategy",
                "model",
                "--planner-model",
                "codex",
                "--worker-model",
                "claude",
                "--review-model",
                "claude",
                "--allow-same-model-review",
                "--experiment-id",
                "exp-001",
                "--experiment-label",
                "planner-benchmark",
            ]
        )

        assert args.planner_strategy == "model"
        assert args.allow_same_model_review is True
        assert args.experiment_id == "exp-001"

    def test_campaign_run_plans_then_executes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aragora.cli.commands.swarm import cmd_swarm

        source = tmp_path / "roadmap.md"
        source.write_text(
            "- Add retry ledger in aragora/swarm/campaign.py\n",
            encoding="utf-8",
        )
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        args = self._args(
            swarm_goal="run",
            source_file=str(source),
            manifest=str(manifest_path),
        )

        execute_result = {
            "stop_reason": "still_running",
            "dispatched_projects": [
                {"project_id": "proj-001", "status": "completed", "outcome": "deliverable_created"}
            ],
        }
        with patch("aragora.swarm.campaign.CampaignExecutor") as executor_cls:
            executor_cls.return_value.execute_once = AsyncMock(return_value=execute_result)
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["mode"] == "campaign-run"
        assert parsed["invocation_mode"] == "planned_then_executed"
        assert parsed["manifest_path"] == str(manifest_path)
        assert parsed["campaign_id"]
        assert parsed["stop_reason"] == "still_running"
        assert len(parsed["dispatched_projects"]) == 1
        assert manifest_path.exists()

    def test_campaign_run_resumes_existing_manifest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aragora.cli.commands.swarm import cmd_swarm
        from aragora.swarm.campaign import (
            CampaignManifest,
            CampaignProject,
            save_campaign_manifest,
        )

        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-resume",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="roadmap.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Already planned",
                    spec=_bounded_spec("Already planned"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["tests pass"],
                    constraints=["stay in scope"],
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)

        args = self._args(
            swarm_goal="run",
            manifest=str(manifest_path),
        )
        execute_result = {
            "stop_reason": "campaign_complete",
            "dispatched_projects": [],
        }
        with patch("aragora.swarm.campaign.CampaignExecutor") as executor_cls:
            executor_cls.return_value.execute_once = AsyncMock(return_value=execute_result)
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["mode"] == "campaign-run"
        assert parsed["invocation_mode"] == "resumed"
        assert parsed["manifest_path"] == str(manifest_path)
        assert parsed["campaign_id"] == "campaign-resume"
        assert parsed["stop_reason"] == "campaign_complete"

    def test_campaign_run_errors_without_input_or_manifest(self, tmp_path: Path) -> None:
        from aragora.cli.commands.swarm import cmd_swarm

        manifest_path = tmp_path / ".aragora" / "nonexistent.yaml"
        args = self._args(
            swarm_goal="run",
            manifest=str(manifest_path),
        )
        with pytest.raises(ValueError, match="campaign run requires"):
            cmd_swarm(args)

    def test_campaign_run_errors_with_conflicting_inputs(self, tmp_path: Path) -> None:
        from aragora.cli.commands.swarm import cmd_swarm

        source = tmp_path / "roadmap.md"
        source.write_text("- Build campaign runner\n", encoding="utf-8")
        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        args = self._args(
            swarm_goal="run",
            manifest=str(manifest_path),
            source_file=str(source),
            issue_list="1,2,3",
        )

        with pytest.raises(
            ValueError, match="exactly one of --source-file, --issue-list, or --github-query"
        ):
            cmd_swarm(args)

    def test_campaign_run_from_issue_list_plans_then_executes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aragora.cli.commands.swarm import cmd_swarm

        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        args = self._args(
            swarm_goal="run",
            manifest=str(manifest_path),
            issue_list="101, 102",
        )
        execute_result = {
            "stop_reason": "still_running",
            "dispatched_projects": [
                {"project_id": "proj-001", "status": "active", "outcome": "deliverable_created"}
            ],
        }
        with (
            patch("aragora.swarm.campaign.CampaignPlanner.plan_from_issue_list") as plan_issue_list,
            patch("aragora.swarm.campaign.CampaignExecutor") as executor_cls,
        ):
            plan_issue_list.return_value = CampaignManifest(
                campaign_id="campaign-issues",
                created_at="2026-03-10T00:00:00+00:00",
                source_kind="issue_list",
                source_ref="101,102",
                projects=[
                    CampaignProject(
                        project_id="proj-001",
                        title="Issue-backed project",
                        spec=_bounded_spec("Issue-backed project"),
                        file_scope_hints=["aragora/swarm/campaign.py"],
                        acceptance_criteria=["tests pass"],
                        constraints=["stay in scope"],
                    )
                ],
            )
            executor_cls.return_value.execute_once = AsyncMock(return_value=execute_result)
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["invocation_mode"] == "planned_then_executed"
        assert parsed["campaign_id"] == "campaign-issues"

    def test_campaign_run_from_github_query_plans_then_executes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aragora.cli.commands.swarm import cmd_swarm

        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        args = self._args(
            swarm_goal="run",
            manifest=str(manifest_path),
            github_query="label:campaign is:open",
        )
        execute_result = {
            "stop_reason": "still_running",
            "dispatched_projects": [
                {"project_id": "proj-009", "status": "active", "outcome": "deliverable_created"}
            ],
        }
        with (
            patch("aragora.swarm.campaign.CampaignPlanner.plan_from_github_query") as plan_query,
            patch("aragora.swarm.campaign.CampaignExecutor") as executor_cls,
        ):
            plan_query.return_value = CampaignManifest(
                campaign_id="campaign-query",
                created_at="2026-03-10T00:00:00+00:00",
                source_kind="github_query",
                source_ref="label:campaign is:open",
                projects=[
                    CampaignProject(
                        project_id="proj-009",
                        title="Query-backed project",
                        spec=_bounded_spec("Query-backed project"),
                        file_scope_hints=["aragora/swarm/campaign.py"],
                        acceptance_criteria=["tests pass"],
                        constraints=["stay in scope"],
                    )
                ],
            )
            executor_cls.return_value.execute_once = AsyncMock(return_value=execute_result)
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["invocation_mode"] == "planned_then_executed"
        assert parsed["campaign_id"] == "campaign-query"

    def test_campaign_run_errors_with_source_when_manifest_exists(self, tmp_path: Path) -> None:
        """Finding 3: conflicting inputs should error even when manifest exists."""
        from aragora.cli.commands.swarm import cmd_swarm
        from aragora.swarm.campaign import (
            CampaignManifest,
            CampaignProject,
            save_campaign_manifest,
        )

        manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
        manifest = CampaignManifest(
            campaign_id="campaign-exists",
            created_at="2026-03-10T00:00:00+00:00",
            source_kind="source_file",
            source_ref="old.md",
            projects=[
                CampaignProject(
                    project_id="proj-001",
                    title="Existing",
                    spec=_bounded_spec("Existing"),
                    file_scope_hints=["aragora/swarm/campaign.py"],
                    acceptance_criteria=["pass"],
                    constraints=["scope"],
                )
            ],
        )
        save_campaign_manifest(manifest_path, manifest)

        args = self._args(
            swarm_goal="run",
            manifest=str(manifest_path),
            source_file="new_roadmap.md",
        )
        with pytest.raises(ValueError, match="cannot supply --source-file"):
            cmd_swarm(args)

    def test_cmd_swarm_campaign_status_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        from aragora.cli.commands.swarm import cmd_swarm

        args = self._args()
        with patch("aragora.swarm.campaign.CampaignExecutor") as executor_cls:
            executor_cls.return_value.status.return_value = {
                "mode": "campaign-status",
                "campaign_id": "campaign-1",
                "stop_reason": "still_running",
                "counts": {"ready": 1},
                "projects": [],
            }
            cmd_swarm(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["mode"] == "campaign-status"
        assert parsed["campaign_id"] == "campaign-1"
