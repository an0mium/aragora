from __future__ import annotations

import subprocess
from pathlib import Path

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.swarm.initiative_loop import (
    InitiativeExecutor,
    STATUS_ACTIVE,
    STATUS_BLOCKED,
    STATUS_MERGED,
    STATUS_NEEDS_HUMAN,
    STATUS_QUEUED,
    STATUS_SUPERSEDED,
)
from aragora.swarm.initiative_models import (
    InitiativeCheckpoint,
    InitiativeMilestone,
    InitiativeRecord,
    InitiativeSlice,
)
from aragora.swarm.initiative_store import InitiativeStore


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


def _store_and_executor(
    repo: Path,
) -> tuple[InitiativeStore, InitiativeExecutor, DevCoordinationStore]:
    store = InitiativeStore(repo_root=repo)
    coordination = DevCoordinationStore(repo_root=repo)
    executor = InitiativeExecutor(repo_root=repo, store=store, coordination_store=coordination)
    return store, executor, coordination


def _record(
    *,
    initiative_id: str = "initiative-executor",
    slices: list[InitiativeSlice],
    checkpoints: list[InitiativeCheckpoint] | None = None,
    milestones: list[InitiativeMilestone] | None = None,
) -> InitiativeRecord:
    return InitiativeRecord(
        initiative_id=initiative_id,
        title="Initiative executor",
        goal="Execute roadmap slices only when dependencies are ready.",
        rationale="PR2 needs dependency-aware initiative execution.",
        slices=slices,
        checkpoints=list(checkpoints or []),
        milestones=list(milestones or []),
    )


def _record_receipt(
    *,
    coordination: DevCoordinationStore,
    initiative_id: str,
    slice_id: str,
    metadata: dict[str, object] | None = None,
    pr_url: str | None = None,
) -> None:
    task_id = f"initiative:{initiative_id}:slice:{slice_id}"
    lease = coordination.claim_lease(
        task_id=task_id,
        title=f"Slice {slice_id}",
        owner_agent="codex",
        owner_session_id="codex-a",
        branch=f"codex/{slice_id}",
        worktree_path=f"/tmp/{slice_id}",
        claimed_paths=[f"aragora/{slice_id}.py"],
        expected_tests=[f"python3 -m pytest tests/{slice_id}.py -q"],
    )
    coordination.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="codex-a",
        branch=f"codex/{slice_id}",
        worktree_path=f"/tmp/{slice_id}",
        commit_shas=["deadbeef"],
        changed_paths=[f"aragora/{slice_id}.py"],
        tests_run=[f"python3 -m pytest tests/{slice_id}.py -q"],
        pr_url=pr_url,
        metadata=metadata,
    )


def test_refresh_marks_dependency_ready_slices_and_blocked_dependents(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, _coordination = _store_and_executor(repo)
    store.save(
        _record(
            slices=[
                InitiativeSlice(
                    slice_id="slice-1",
                    title="First",
                    description="First slice",
                ),
                InitiativeSlice(
                    slice_id="slice-2",
                    title="Second",
                    description="Second slice",
                    dependencies=["slice-1"],
                ),
            ]
        )
    )

    snapshot = executor.refresh("initiative-executor")

    assert snapshot.status == STATUS_QUEUED
    assert snapshot.ready_slice_ids == ["slice-1"]
    assert snapshot.slice_statuses["slice-1"] == STATUS_QUEUED
    assert snapshot.slice_statuses["slice-2"] == STATUS_BLOCKED


def test_dispatch_ready_slices_assigns_one_owner_per_slice_without_duplicates(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, _coordination = _store_and_executor(repo)
    store.save(
        _record(
            slices=[
                InitiativeSlice(slice_id="slice-1", title="One", description="One"),
                InitiativeSlice(slice_id="slice-2", title="Two", description="Two"),
            ]
        )
    )

    snapshot = executor.dispatch_ready_slices(
        "initiative-executor",
        owner_targets=["codex-a", "codex-b"],
        assigned_by="initiative-boss",
    )

    assert snapshot.dispatched_slice_ids == ["slice-1", "slice-2"]
    assert snapshot.slice_statuses["slice-1"] == STATUS_ACTIVE
    assert snapshot.slice_statuses["slice-2"] == STATUS_ACTIVE
    assert snapshot.owner_targets == {"slice-1": "codex-a", "slice-2": "codex-b"}

    repeated = executor.dispatch_ready_slices(
        "initiative-executor",
        owner_targets=["codex-a", "codex-b"],
        assigned_by="initiative-boss",
    )

    assert repeated.dispatched_slice_ids == []
    assert len(executor.directive_board.list()) == 2


def test_refresh_dependency_readiness_is_order_independent(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, _coordination = _store_and_executor(repo)
    store.save(
        _record(
            slices=[
                InitiativeSlice(
                    slice_id="slice-2",
                    title="Second",
                    description="Second slice",
                    dependencies=["slice-1"],
                ),
                InitiativeSlice(
                    slice_id="slice-1",
                    title="First",
                    description="First slice",
                    status=STATUS_MERGED,
                ),
            ]
        )
    )

    snapshot = executor.refresh("initiative-executor")

    assert snapshot.slice_statuses["slice-1"] == STATUS_MERGED
    assert snapshot.slice_statuses["slice-2"] == STATUS_QUEUED
    assert snapshot.ready_slice_ids == ["slice-2"]
    assert snapshot.status == STATUS_QUEUED


def test_dispatch_ready_slices_ignores_completed_directives_for_redispatch(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, _coordination = _store_and_executor(repo)
    store.save(
        _record(
            slices=[
                InitiativeSlice(slice_id="slice-1", title="One", description="One"),
            ]
        )
    )

    first = executor.dispatch_ready_slices(
        "initiative-executor",
        owner_targets=["codex-a"],
        assigned_by="initiative-boss",
    )
    assert first.dispatched_slice_ids == ["slice-1"]

    directive = executor.directive_board.get("codex-a")
    assert directive is not None
    executor.directive_board.assign(
        "codex-a",
        directive.task,
        scope=directive.scope,
        constraints=directive.constraints,
        assigned_by=directive.assigned_by,
        status="completed",
    )

    refreshed = executor.refresh("initiative-executor")
    assert refreshed.slice_statuses["slice-1"] == STATUS_QUEUED
    assert refreshed.ready_slice_ids == ["slice-1"]

    redispatched = executor.dispatch_ready_slices(
        "initiative-executor",
        owner_targets=["codex-a"],
        assigned_by="initiative-boss",
    )

    assert redispatched.dispatched_slice_ids == ["slice-1"]
    assert redispatched.slice_statuses["slice-1"] == STATUS_ACTIVE


def test_receipt_backed_slice_becomes_terminal_and_checkpoint_blocks_follow_on_dispatch(
    tmp_path,
) -> None:
    repo = _init_repo(tmp_path)
    store, executor, coordination = _store_and_executor(repo)
    store.save(
        _record(
            slices=[
                InitiativeSlice(slice_id="slice-1", title="One", description="One"),
                InitiativeSlice(
                    slice_id="slice-2",
                    title="Two",
                    description="Two",
                    dependencies=["slice-1"],
                ),
            ],
            checkpoints=[
                InitiativeCheckpoint(
                    checkpoint_id="checkpoint-1",
                    title="Review slice one",
                    dependencies=["slice-1"],
                )
            ],
        )
    )
    _record_receipt(
        coordination=coordination,
        initiative_id="initiative-executor",
        slice_id="slice-1",
        pr_url="https://github.com/synaptent/aragora/pull/9999",
    )

    snapshot = executor.refresh("initiative-executor")

    assert snapshot.status == STATUS_NEEDS_HUMAN
    assert snapshot.slice_statuses["slice-1"] == STATUS_NEEDS_HUMAN
    assert snapshot.checkpoint_statuses["checkpoint-1"] == STATUS_NEEDS_HUMAN
    assert snapshot.ready_slice_ids == []
    assert snapshot.boundary_blockers == ["checkpoint-1"]


def test_resolved_checkpoint_releases_next_dependency_ready_slice(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, coordination = _store_and_executor(repo)
    initiative = _record(
        slices=[
            InitiativeSlice(slice_id="slice-1", title="One", description="One"),
            InitiativeSlice(
                slice_id="slice-2",
                title="Two",
                description="Two",
                dependencies=["slice-1"],
            ),
        ],
        checkpoints=[
            InitiativeCheckpoint(
                checkpoint_id="checkpoint-1",
                title="Review slice one",
                dependencies=["slice-1"],
            )
        ],
    )
    store.save(initiative)
    _record_receipt(
        coordination=coordination,
        initiative_id="initiative-executor",
        slice_id="slice-1",
    )
    executor.refresh("initiative-executor")

    updated = store.get("initiative-executor")
    assert updated is not None
    updated.checkpoints[0].status = STATUS_MERGED
    store.save(updated)

    snapshot = executor.refresh("initiative-executor")

    assert snapshot.status == STATUS_QUEUED
    assert snapshot.ready_slice_ids == ["slice-2"]
    assert snapshot.slice_statuses["slice-2"] == STATUS_QUEUED


def test_milestone_boundary_stops_after_completed_slice_group(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, coordination = _store_and_executor(repo)
    initiative = _record(
        slices=[
            InitiativeSlice(slice_id="slice-1", title="One", description="One"),
            InitiativeSlice(slice_id="slice-2", title="Two", description="Two"),
        ],
        milestones=[
            InitiativeMilestone(
                milestone_id="milestone-1",
                title="Pause for review",
                slice_ids=["slice-1"],
            )
        ],
    )
    store.save(initiative)
    _record_receipt(
        coordination=coordination,
        initiative_id="initiative-executor",
        slice_id="slice-1",
    )

    snapshot = executor.refresh("initiative-executor")

    assert snapshot.milestone_statuses["milestone-1"] == STATUS_NEEDS_HUMAN
    assert snapshot.ready_slice_ids == []

    updated = store.get("initiative-executor")
    assert updated is not None
    updated.milestones[0].status = STATUS_MERGED
    store.save(updated)

    resumed = executor.refresh("initiative-executor")

    assert resumed.ready_slice_ids == ["slice-2"]


def test_receipt_metadata_can_mark_slice_merged_or_superseded(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    store, executor, coordination = _store_and_executor(repo)
    store.save(
        _record(
            slices=[
                InitiativeSlice(slice_id="slice-1", title="Merged", description="Merged"),
                InitiativeSlice(slice_id="slice-2", title="Superseded", description="Superseded"),
            ]
        )
    )
    _record_receipt(
        coordination=coordination,
        initiative_id="initiative-executor",
        slice_id="slice-1",
        metadata={"initiative_terminal_status": STATUS_MERGED},
    )
    _record_receipt(
        coordination=coordination,
        initiative_id="initiative-executor",
        slice_id="slice-2",
        metadata={"initiative_terminal_status": STATUS_SUPERSEDED},
    )

    snapshot = executor.refresh("initiative-executor")

    assert snapshot.slice_statuses["slice-1"] == STATUS_MERGED
    assert snapshot.slice_statuses["slice-2"] == STATUS_SUPERSEDED
    assert snapshot.status == STATUS_MERGED
