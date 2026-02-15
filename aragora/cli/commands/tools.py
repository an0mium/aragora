"""
Tool and utility CLI commands.

Contains commands for operational modes, templates, self-improvement,
and codebase context building.
"""

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.modes import ModeRegistry

if TYPE_CHECKING:
    from aragora.nomic.autonomous_orchestrator import OrchestrationResult


def cmd_modes(args: argparse.Namespace) -> None:
    """Handle 'modes' command - list available operational modes."""
    modes = ModeRegistry.get_all()

    print("\n" + "=" * 60)
    print("AVAILABLE OPERATIONAL MODES")
    print("=" * 60 + "\n")

    if not modes:
        print("No modes registered. This shouldn't happen!")
        return

    verbose = getattr(args, "verbose", False)

    for mode in modes:
        # Mode header
        print(f"[{mode.name}]")
        print(f"  {mode.description}")

        # Show tool access
        tools = []
        from aragora.modes.tool_groups import ToolGroup

        if ToolGroup.READ in mode.tool_groups:
            tools.append("read")
        if ToolGroup.EDIT in mode.tool_groups:
            tools.append("edit")
        if ToolGroup.COMMAND in mode.tool_groups:
            tools.append("command")
        if ToolGroup.BROWSER in mode.tool_groups:
            tools.append("browser")
        if ToolGroup.DEBATE in mode.tool_groups:
            tools.append("debate")

        print(f"  Tools: {', '.join(tools) if tools else 'none'}")

        if verbose:
            # Show full system prompt in verbose mode
            prompt = mode.get_system_prompt()
            # Truncate for display
            lines = prompt.strip().split("\n")
            preview = "\n    ".join(lines[:10])
            if len(lines) > 10:
                preview += "\n    ..."
            print(f"\n  System Prompt:\n    {preview}\n")
        else:
            print()

    print("-" * 60)
    print("Usage: aragora ask 'task' --mode <mode-name>")
    print("       aragora modes --verbose  (show full system prompts)")


def cmd_templates(args: argparse.Namespace) -> None:
    """Handle 'templates' command - list available debate templates."""
    from aragora.templates import list_templates

    templates = list_templates()

    print("\n" + "=" * 60)
    print("\U0001f4cb AVAILABLE DEBATE TEMPLATES")
    print("=" * 60 + "\n")

    for t in templates:
        print(f"[{t['type']}] {t['name']}")
        print(f"  {t['description'][:60]}...")
        print(f"  Agents: {t['agents']}, Domain: {t['domain']}")
        print()


def cmd_improve(args: argparse.Namespace) -> None:
    """Handle 'improve' command - self-improvement mode using AutonomousOrchestrator."""
    from aragora.nomic.autonomous_orchestrator import (
        Track,
    )

    goal = args.goal
    tracks = args.tracks.split(",") if args.tracks else None
    dry_run = args.dry_run
    max_cycles = args.max_cycles
    require_approval = args.require_approval
    use_debate = args.debate
    max_parallel = args.max_parallel
    codebase_path = Path(args.path).resolve() if args.path else Path.cwd()
    verbose = getattr(args, "verbose", False)

    # Validate tracks if provided
    valid_tracks = {t.value for t in Track}
    if tracks:
        invalid = [t for t in tracks if t.lower() not in valid_tracks]
        if invalid:
            print(f"\nError: Invalid track(s): {', '.join(invalid)}")
            print(f"Valid tracks: {', '.join(sorted(valid_tracks))}")
            return

    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT MODE (AutonomousOrchestrator)")
    print("=" * 60)
    print(f"\nGoal: {goal}")
    print(f"Codebase: {codebase_path}")
    if tracks:
        print(f"Tracks: {', '.join(tracks)}")
    if dry_run:
        print("Mode: DRY RUN (preview only)")
    else:
        print(f"Max cycles: {max_cycles}")
        print(f"Max parallel: {max_parallel}")
        if require_approval:
            print("Approval: Required at gates")
    if use_debate:
        print("Decomposition: Multi-agent debate")
    print()

    if dry_run:
        # Dry run: just show the decomposition plan
        _run_dry_run(goal, tracks, use_debate, verbose)
    else:
        # Full execution via orchestrator
        _run_orchestration(
            goal=goal,
            tracks=tracks,
            max_cycles=max_cycles,
            require_approval=require_approval,
            use_debate=use_debate,
            max_parallel=max_parallel,
            codebase_path=codebase_path,
            verbose=verbose,
        )


def _run_dry_run(
    goal: str,
    tracks: list[str] | None,
    use_debate: bool,
    verbose: bool,
) -> None:
    """Run dry-run mode: show decomposition without executing."""
    from aragora.nomic.task_decomposer import TaskDecomposer, DecomposerConfig
    from aragora.nomic.autonomous_orchestrator import AgentRouter, Track

    print("-" * 60)
    print("TASK DECOMPOSITION PREVIEW")
    print("-" * 60)

    decomposer = TaskDecomposer(config=DecomposerConfig(complexity_threshold=4))
    router = AgentRouter()

    # Enrich goal with track context
    if tracks:
        enriched_goal = f"{goal}\n\nFocus tracks: {', '.join(tracks)}"
    else:
        enriched_goal = goal

    if use_debate:
        print("\nUsing multi-agent debate for decomposition...")
        print("(This may take a minute and consume API tokens)\n")
        decomposition = asyncio.run(decomposer.analyze_with_debate(enriched_goal))
    else:
        print("\nUsing heuristic decomposition...\n")
        decomposition = decomposer.analyze(enriched_goal)

    # Show analysis results
    print(
        f"Complexity Score: {decomposition.complexity_score}/10 ({decomposition.complexity_level})"
    )
    print(f"Should Decompose: {'Yes' if decomposition.should_decompose else 'No'}")
    print(f"Rationale: {decomposition.rationale}")
    print()

    if not decomposition.subtasks:
        print("No subtasks generated. Goal may be too simple or abstract.")
        print("\nTips:")
        print("  - Add specific file paths or areas to focus on")
        print("  - Use --debate for abstract goals like 'maximize utility'")
        return

    # Filter by allowed tracks if specified
    allowed_tracks = {Track(t.lower()) for t in tracks} if tracks else set(Track)

    print(f"Subtasks ({len(decomposition.subtasks)}):")
    print("-" * 40)

    for i, subtask in enumerate(decomposition.subtasks, 1):
        track = router.determine_track(subtask)
        agent = router.select_agent_type(subtask, track)

        # Check if this track is allowed
        if track not in allowed_tracks:
            status = " [SKIPPED - track not selected]"
        else:
            status = ""

        print(f"\n{i}. {subtask.title}{status}")
        print(f"   Track: {track.value}")
        print(f"   Agent: {agent}")
        print(f"   Complexity: {subtask.estimated_complexity}")

        if subtask.file_scope:
            print(f"   Files: {', '.join(subtask.file_scope[:3])}")
            if len(subtask.file_scope) > 3:
                print(f"          ... and {len(subtask.file_scope) - 3} more")

        if subtask.dependencies:
            print(f"   Dependencies: {', '.join(subtask.dependencies)}")

        if verbose:
            print(f"   Description: {subtask.description[:200]}...")

    print("\n" + "-" * 60)
    print("To execute this plan, run without --dry-run:")
    cmd_parts = ["aragora improve", f'--goal "{goal}"']
    if tracks:
        cmd_parts.append(f"--tracks {','.join(tracks)}")
    if use_debate:
        cmd_parts.append("--debate")
    print(f"  {' '.join(cmd_parts)}")
    print()


def _run_orchestration(
    goal: str,
    tracks: list[str] | None,
    max_cycles: int,
    require_approval: bool,
    use_debate: bool,
    max_parallel: int,
    codebase_path: Path,
    verbose: bool,
    use_hardened: bool = False,
    use_worktree: bool = False,
    spectate: bool = False,
    generate_receipts: bool = False,
    budget_limit: float | None = None,
    coordinated: bool = False,
) -> None:
    """Run full orchestration via AutonomousOrchestrator or HardenedOrchestrator."""
    checkpoints: list[tuple[str, dict[str, Any]]] = []

    def on_checkpoint(phase: str, data: dict[str, Any]) -> None:
        """Callback for orchestration checkpoints."""
        checkpoints.append((phase, data))
        if verbose:
            print(f"  [Checkpoint] {phase}: {data.get('orchestration_id', '')}")

        # Handle approval gates if required
        if require_approval and phase in ("decomposed", "assigned"):
            _prompt_approval(phase, data)

    print("-" * 60)
    print("STARTING ORCHESTRATION")
    if use_hardened or use_worktree:
        print("  Mode: HARDENED (worktree isolation, gauntlet, mode enforcement)")
    if spectate:
        print("  Spectate: ON")
    if generate_receipts:
        print("  Receipts: ON")
    if budget_limit:
        print(f"  Budget: ${budget_limit:.2f}")
    if coordinated:
        print("  Pipeline: COORDINATED (MetaPlanner -> BranchCoordinator)")
    print("-" * 60)

    # Use HardenedOrchestrator when hardened flags are set
    if use_hardened or use_worktree or coordinated:
        from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

        orchestrator = HardenedOrchestrator(
            aragora_path=codebase_path,
            require_human_approval=require_approval,
            max_parallel_tasks=max_parallel,
            on_checkpoint=on_checkpoint,
            use_debate_decomposition=use_debate,
            use_worktree_isolation=True,
            enable_gauntlet_validation=use_hardened,
            budget_limit_usd=budget_limit,
            generate_receipts=generate_receipts,
            spectate_stream=spectate,
        )
    else:
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orchestrator = AutonomousOrchestrator(
            aragora_path=codebase_path,
            require_human_approval=require_approval,
            max_parallel_tasks=max_parallel,
            on_checkpoint=on_checkpoint,
            use_debate_decomposition=use_debate,
        )

    try:
        if coordinated and hasattr(orchestrator, "execute_goal_coordinated"):
            result = asyncio.run(
                orchestrator.execute_goal_coordinated(
                    goal=goal,
                    tracks=tracks,
                    max_cycles=max_cycles,
                )
            )
        else:
            result = asyncio.run(
                orchestrator.execute_goal(
                    goal=goal,
                    tracks=tracks,
                    max_cycles=max_cycles,
                )
            )
        _print_result(result, verbose)

    except KeyboardInterrupt:
        print("\n\nOrchestration interrupted by user.")
        print("Active assignments may be left in progress.")

    except Exception as e:
        print(f"\nOrchestration failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()


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
            import json

            print(json.dumps(data, indent=2, default=str))
        else:
            print("Please enter 'y' to approve, 'n' to reject, or 'd' for details.")


def _print_result(result: "OrchestrationResult", verbose: bool) -> None:
    """Print orchestration result."""

    print("\n" + "=" * 60)
    print("ORCHESTRATION RESULT")
    print("=" * 60)

    status = "SUCCESS" if result.success else "FAILED"
    print(f"\nStatus: {status}")
    print(f"Goal: {result.goal}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print()
    print(f"Total subtasks: {result.total_subtasks}")
    print(f"  Completed: {result.completed_subtasks}")
    print(f"  Failed: {result.failed_subtasks}")
    print(f"  Skipped: {result.skipped_subtasks}")

    if result.error:
        print(f"\nError: {result.error}")

    if result.summary:
        print(f"\n{result.summary}")

    if verbose and result.assignments:
        print("\n" + "-" * 40)
        print("ASSIGNMENT DETAILS")
        print("-" * 40)

        for a in result.assignments:
            status_icon = "+" if a.status == "completed" else ("-" if a.status == "failed" else "?")
            print(f"\n{status_icon} {a.subtask.title}")
            print(f"  Track: {a.track.value}")
            print(f"  Agent: {a.agent_type}")
            print(f"  Status: {a.status}")
            if a.started_at and a.completed_at:
                duration = (a.completed_at - a.started_at).total_seconds()
                print(f"  Duration: {duration:.1f}s")
            if a.result and a.status == "failed":
                error = a.result.get("error", "Unknown error")
                print(f"  Error: {error[:100]}...")

    print()


def cmd_context(args: argparse.Namespace) -> None:
    """Handle 'context' command - build codebase context for RLM."""
    from aragora.rlm.codebase_context import CodebaseContextBuilder

    root = Path(args.path or ".").resolve()
    include_tests: bool | None = None
    if args.include_tests:
        include_tests = True
    elif args.exclude_tests:
        include_tests = False

    builder = CodebaseContextBuilder(
        root_path=root,
        max_context_bytes=args.max_bytes or 0,
        include_tests=include_tests,
        full_corpus=args.full_corpus,
    )

    async def _run() -> None:
        index = await builder.build_index()
        print("\n" + "=" * 60)
        print("\U0001f4da CODEBASE CONTEXT")
        print("=" * 60)
        print(f"Root: {index.root_path}")
        print(
            f"Files: {index.total_files} | Lines: {index.total_lines} | "
            f"Bytes: {index.total_bytes} | ~Tokens: {index.total_tokens_estimate}"
        )
        print(f"Index build time: {index.build_time_seconds:.2f}s")

        if args.rlm:
            print("\n\U0001f50d Building RLM context (TRUE RLM preferred)...")
            context = await builder.build_rlm_context()
            if context is None:
                print("\u26a0\ufe0f  RLM context unavailable (missing RLM package or disabled).")
            else:
                print("\u2705 RLM context ready.")

        if args.summary_out or args.preview:
            context = await builder.build_debate_context()
            if args.summary_out:
                output_path = Path(args.summary_out).resolve()
                output_path.write_text(context, encoding="utf-8")
                print(f"\n\U0001f4dd Context summary written to: {output_path}")
            if args.preview:
                print("\n\U0001f4c4 Context preview (first 40 lines):")
                for line in context.splitlines()[:40]:
                    print(line)
                if len(context.splitlines()) > 40:
                    print("... (truncated)")

        print()

    asyncio.run(_run())
