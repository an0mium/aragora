"""Self-improvement CLI command -- unified hardened pipeline.

Runs the full worktree-isolated self-improvement pipeline:
1. MetaPlanner debate -> prioritize goals
2. TaskDecomposer -> break into subtasks per track
3. WorktreeManager -> create isolated worktrees per subtask
4. HardenedOrchestrator -> execute with gauntlet validation + mode enforcement
5. BranchCoordinator -> merge passing branches, reject failing ones
6. DecisionReceipt -> generate audit receipts per subtask

Usage:
    aragora self-improve "Make Aragora the best decision platform for SMEs"
    aragora self-improve "Improve test coverage" --tracks qa --budget-limit 5
    aragora self-improve "Harden security" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any


def cmd_self_improve(args: argparse.Namespace) -> None:
    """Handle 'self-improve' command -- unified hardened pipeline."""
    goal = args.goal
    tracks = args.tracks.split(",") if args.tracks else None
    dry_run = args.dry_run
    max_cycles = args.max_cycles
    require_approval = args.require_approval
    budget_limit = args.budget_limit
    spectate = args.spectate
    generate_receipts = args.receipt
    sessions = args.sessions
    codebase_path = Path(args.path).resolve() if args.path else Path.cwd()
    verbose = getattr(args, "verbose", False)

    # Validate tracks
    valid_track_values = {"sme", "developer", "self_hosted", "qa", "core", "security"}
    if tracks:
        invalid = [t for t in tracks if t.lower() not in valid_track_values]
        if invalid:
            print(f"\nError: Invalid track(s): {', '.join(invalid)}")
            print(f"Valid tracks: {', '.join(sorted(valid_track_values))}")
            return

    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT PIPELINE (Hardened + Coordinated)")
    print("=" * 60)
    print(f"\nGoal: {goal}")
    print(f"Codebase: {codebase_path}")
    if tracks:
        print(f"Tracks: {', '.join(tracks)}")
    if dry_run:
        print("Mode: DRY RUN (preview only)")
    else:
        print(f"Max cycles: {max_cycles}")
        if budget_limit:
            print(f"Budget limit: ${budget_limit:.2f}")
        if sessions:
            print(f"Sessions: {sessions}")
        print(f"Worktree isolation: ON")
        print(f"Gauntlet validation: ON")
        print(f"Mode enforcement: ON")
        print(f"Spectate: {'ON' if spectate else 'OFF'}")
        print(f"Receipts: {'ON' if generate_receipts else 'OFF'}")
        if require_approval:
            print("Approval: Required at gates")
    print()

    if dry_run:
        _run_self_improve_dry_run(goal, tracks, verbose)
    else:
        _run_self_improve_pipeline(
            goal=goal,
            tracks=tracks,
            max_cycles=max_cycles,
            require_approval=require_approval,
            budget_limit=budget_limit,
            spectate=spectate,
            generate_receipts=generate_receipts,
            sessions=sessions,
            codebase_path=codebase_path,
            verbose=verbose,
        )


def _run_self_improve_dry_run(
    goal: str,
    tracks: list[str] | None,
    verbose: bool,
) -> None:
    """Preview the self-improvement plan without executing."""
    print("-" * 60)
    print("PLAN PREVIEW")
    print("-" * 60)

    # Step 1: MetaPlanner heuristic preview
    try:
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            Track,
        )

        planner = MetaPlanner(MetaPlannerConfig(quick_mode=True))

        available_tracks = None
        if tracks:
            track_map = {t.value: t for t in Track}
            available_tracks = [track_map[t] for t in tracks if t in track_map]

        goals = asyncio.run(
            planner.prioritize_work(
                objective=goal,
                available_tracks=available_tracks,
            )
        )

        print(f"\nMetaPlanner goals ({len(goals)}):")
        for pg in goals:
            print(f"  {pg.priority}. [{pg.track.value}] {pg.description}")
            print(f"     Impact: {pg.estimated_impact}")
            if pg.rationale:
                print(f"     Rationale: {pg.rationale}")
            print()

    except ImportError:
        print("\nMetaPlanner unavailable, showing direct decomposition:\n")

    # Step 2: TaskDecomposer preview
    try:
        from aragora.nomic.task_decomposer import DecomposerConfig, TaskDecomposer
        from aragora.nomic.autonomous_orchestrator import AgentRouter, Track

        decomposer = TaskDecomposer(config=DecomposerConfig(complexity_threshold=4))
        router = AgentRouter()

        enriched_goal = goal
        if tracks:
            enriched_goal = f"{goal}\n\nFocus tracks: {', '.join(tracks)}"

        decomposition = decomposer.analyze(enriched_goal)

        print(f"Complexity: {decomposition.complexity_score}/10 ({decomposition.complexity_level})")
        print(f"Subtasks: {len(decomposition.subtasks)}")
        print()

        allowed_tracks = {Track(t.lower()) for t in tracks} if tracks else set(Track)

        for i, subtask in enumerate(decomposition.subtasks, 1):
            track = router.determine_track(subtask)
            agent = router.select_agent_type(subtask, track)
            skip = " [SKIP]" if track not in allowed_tracks else ""

            print(f"  {i}. {subtask.title}{skip}")
            print(
                f"     Track: {track.value} | Agent: {agent} | Complexity: {subtask.estimated_complexity}"
            )
            if subtask.file_scope:
                files = ", ".join(subtask.file_scope[:3])
                extra = f" +{len(subtask.file_scope) - 3}" if len(subtask.file_scope) > 3 else ""
                print(f"     Files: {files}{extra}")
            print()

    except ImportError:
        print("TaskDecomposer unavailable for preview")

    # Step 3: Worktree plan
    print("-" * 60)
    print("WORKTREE PLAN")
    print("-" * 60)
    print()
    print("Each subtask will get an isolated git worktree:")
    print("  - Branch: dev/<track>-<subtask-id>-<timestamp>")
    print("  - Tests run inside worktree before merge")
    print("  - Gauntlet validation after execution")
    print("  - Auto-merge to main on success")
    print()
    print("To execute this plan:")

    cmd_parts = [f'aragora self-improve "{goal}"']
    if tracks:
        cmd_parts.append(f"--tracks {','.join(tracks)}")
    print(f"  {' '.join(cmd_parts)}")
    print()


def _run_self_improve_pipeline(
    goal: str,
    tracks: list[str] | None,
    max_cycles: int,
    require_approval: bool,
    budget_limit: float | None,
    spectate: bool,
    generate_receipts: bool,
    sessions: int | None,
    codebase_path: Path,
    verbose: bool,
) -> None:
    """Run the full coordinated self-improvement pipeline."""
    from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

    checkpoints: list[tuple[str, dict[str, Any]]] = []

    def on_checkpoint(phase: str, data: dict[str, Any]) -> None:
        checkpoints.append((phase, data))
        if verbose:
            print(f"  [Checkpoint] {phase}: {data.get('orchestration_id', '')}")
        if require_approval and phase in ("decomposed", "assigned"):
            _prompt_approval(phase, data)

    print("-" * 60)
    print("STARTING SELF-IMPROVEMENT PIPELINE")
    print("-" * 60)

    orchestrator = HardenedOrchestrator(
        aragora_path=codebase_path,
        require_human_approval=require_approval,
        max_parallel_tasks=sessions or 4,
        on_checkpoint=on_checkpoint,
        use_debate_decomposition=True,
        # Hardened defaults: all ON
        use_worktree_isolation=True,
        enable_gauntlet_validation=True,
        enable_mode_enforcement=True,
        enable_prompt_defense=True,
        enable_audit_reconciliation=True,
        enable_meta_planning=True,
        budget_limit_usd=budget_limit,
        generate_receipts=generate_receipts,
        spectate_stream=spectate,
    )

    try:
        # Use coordinated execution (MetaPlanner + BranchCoordinator)
        result = asyncio.run(
            orchestrator.execute_goal_coordinated(
                goal=goal,
                tracks=tracks,
                max_cycles=max_cycles,
            )
        )

        _print_pipeline_result(result, orchestrator, verbose)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print("Active worktrees may need cleanup:")
        print("  ./scripts/cleanup_worktrees.sh --all")

    except (OSError, RuntimeError, ValueError) as e:
        print(f"\nPipeline failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        print("\nClean up worktrees:")
        print("  ./scripts/cleanup_worktrees.sh --all")


def _prompt_approval(phase: str, data: dict[str, Any]) -> None:
    """Prompt for human approval at a gate."""
    print(f"\n{'=' * 40}")
    print(f"APPROVAL GATE: {phase.upper()}")
    print(f"{'=' * 40}")

    if phase == "decomposed":
        count = data.get("subtask_count", 0)
        print(f"The goal has been decomposed into {count} subtasks.")
    elif phase == "assigned":
        count = data.get("assignment_count", 0)
        print(f"{count} assignments have been created.")

    while True:
        response = input("\nProceed? [y/n/details]: ").strip().lower()
        if response in ("y", "yes"):
            print("Approved. Continuing...\n")
            return
        elif response in ("n", "no"):
            raise KeyboardInterrupt("User rejected at approval gate")
        elif response in ("d", "details"):
            print(json.dumps(data, indent=2, default=str))
        else:
            print("Please enter 'y' to approve, 'n' to reject, or 'd' for details.")


def _print_pipeline_result(
    result: Any,
    orchestrator: Any,
    verbose: bool,
) -> None:
    """Print pipeline results including receipts and spectate events."""
    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT RESULT")
    print("=" * 60)

    status = "SUCCESS" if result.success else "FAILED"
    print(f"\nStatus: {status}")
    print(f"Goal: {result.goal}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print()
    print(f"Total subtasks: {result.total_subtasks}")
    print(f"  Completed: {result.completed_subtasks}")
    print(f"  Failed: {result.failed_subtasks}")
    if result.skipped_subtasks:
        print(f"  Skipped: {result.skipped_subtasks}")

    if result.summary:
        print(f"\n{result.summary}")

    # Receipts
    receipts = getattr(orchestrator, "_receipts", [])
    if receipts:
        print(f"\nDecision Receipts: {len(receipts)}")
        for receipt in receipts:
            task = getattr(receipt, "task", "unknown")
            integrity = getattr(receipt, "integrity_hash", "n/a")
            print(f"  - {task[:60]} [{integrity[:12]}...]")

    # Spectate events
    events = getattr(orchestrator, "_spectate_events", [])
    if events and verbose:
        print(f"\nSpectate Events: {len(events)}")
        for event in events[-10:]:  # Show last 10
            print(f"  [{event.get('type', '?')}] {event.get('timestamp', '')}")

    if result.error:
        print(f"\nError: {result.error}")

    print()
