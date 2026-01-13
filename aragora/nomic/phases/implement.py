"""
Implementation phase for nomic loop.

Phase 3: Code generation
- Hybrid multi-model implementation
- Task planning and execution
- Crash recovery via checkpoints
- Pre-verification review
"""

import asyncio
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Set

from . import ImplementResult
from .scope_limiter import ScopeLimiter, ScopeEvaluation

# Default protected files
DEFAULT_PROTECTED_FILES = [
    "CLAUDE.md",
    "core.py",
    "aragora/__init__.py",
    ".env",
    "scripts/nomic_loop.py",
]

SAFETY_PREAMBLE = """You are implementing code in the aragora project.
SAFETY RULES:
1. NEVER delete or modify protected files
2. NEVER remove existing functionality - only ADD new code
3. NEVER simplify code by removing features
4. Preserve ALL existing imports, classes, and functions"""


class ImplementPhase:
    """
    Handles code implementation from design specifications.

    Supports hybrid multi-model implementation with crash recovery.
    """

    def __init__(
        self,
        aragora_path: Path,
        plan_generator: Optional[Callable[[str, Path], Any]] = None,
        executor: Optional[Any] = None,
        progress_loader: Optional[Callable[[Path], Any]] = None,
        progress_saver: Optional[Callable[[Any, Path], None]] = None,
        progress_clearer: Optional[Callable[[Path], None]] = None,
        protected_files: Optional[List[str]] = None,
        cycle_count: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
        stream_emit_fn: Optional[Callable[..., None]] = None,
        record_replay_fn: Optional[Callable[..., None]] = None,
        save_state_fn: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize the implement phase.

        Args:
            aragora_path: Path to the aragora project root
            plan_generator: Function to generate implementation plan from design
            executor: Executor instance for running tasks
            progress_loader: Function to load progress checkpoint
            progress_saver: Function to save progress checkpoint
            progress_clearer: Function to clear progress checkpoint
            protected_files: List of files that cannot be modified
            cycle_count: Current cycle number
            log_fn: Function to log messages
            stream_emit_fn: Function to emit streaming events
            record_replay_fn: Function to record replay events
            save_state_fn: Function to save phase state
        """
        self.aragora_path = aragora_path
        self._plan_generator = plan_generator
        self._executor = executor
        self._progress_loader = progress_loader
        self._progress_saver = progress_saver
        self._progress_clearer = progress_clearer
        self.protected_files = protected_files or DEFAULT_PROTECTED_FILES
        self.cycle_count = cycle_count
        self._log = log_fn or print
        self._stream_emit = stream_emit_fn or (lambda *args: None)
        self._record_replay = record_replay_fn or (lambda *args: None)
        self._save_state = save_state_fn or (lambda state: None)

    async def execute(self, design: str) -> ImplementResult:
        """
        Execute the implementation phase.

        Args:
            design: Implementation design from design phase

        Returns:
            ImplementResult with implementation status and diff
        """
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 3: IMPLEMENTATION (Hybrid)")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "implement", self.cycle_count, {})
        self._record_replay("phase", "system", "implement")

        use_hybrid = os.environ.get("ARAGORA_HYBRID_IMPLEMENT", "1") == "1"
        scope_check = os.environ.get("ARAGORA_SCOPE_CHECK", "1") == "1"

        # Check design scope before proceeding
        if scope_check:
            limiter = ScopeLimiter(protected_files=self.protected_files)
            scope_eval = limiter.evaluate(design)

            if not scope_eval.is_implementable:
                self._log(f"  [scope] Design rejected: {scope_eval.reason}")
                for factor in scope_eval.risk_factors[:3]:
                    self._log(f"  [scope]   - {factor}")
                for suggestion in scope_eval.suggested_simplifications[:2]:
                    self._log(f"  [scope] Suggestion: {suggestion}")

                self._stream_emit(
                    "on_phase_end",
                    "implement",
                    self.cycle_count,
                    False,
                    (datetime.now() - phase_start).total_seconds(),
                    {"error": "scope_exceeded", "evaluation": scope_eval.to_dict()},
                )

                return ImplementResult(
                    success=False,
                    error=f"Design too complex: {scope_eval.reason}",
                    data={"scope_evaluation": scope_eval.to_dict()},
                    duration_seconds=(datetime.now() - phase_start).total_seconds(),
                    files_modified=[],
                    diff_summary="",
                )
            else:
                self._log(
                    f"  [scope] Design approved (complexity: {scope_eval.complexity_score:.2f})"
                )

        if not use_hybrid:
            return await self._legacy_implement(design)

        design_hash = hashlib.md5(design.encode()).hexdigest()

        # Check for crash recovery
        progress = self._progress_loader(self.aragora_path) if self._progress_loader else None
        if progress and hasattr(progress, "plan") and progress.plan.design_hash == design_hash:
            self._log("  Resuming from checkpoint...")
            plan = progress.plan
            completed: Set[str] = set(progress.completed_tasks)
            stash_ref = progress.git_stash_ref
        else:
            # Generate new plan
            if self._plan_generator:
                try:
                    self._log("  Generating implementation plan...")
                    plan = await self._plan_generator(design, self.aragora_path)
                    self._log(f"  Plan generated: {len(plan.tasks)} tasks")
                except Exception as e:
                    self._log(f"  Plan generation failed: {e}")
                    return ImplementResult(
                        success=False,
                        error=str(e),
                        data={},
                        duration_seconds=(datetime.now() - phase_start).total_seconds(),
                        files_modified=[],
                        diff_summary="",
                    )
            else:
                self._log("  No plan generator configured, using legacy mode")
                return await self._legacy_implement(design)

            completed = set()
            stash_ref = await self._git_stash_create()

            if self._progress_saver:
                self._progress_saver(
                    {
                        "plan": plan,
                        "completed_tasks": [],
                        "git_stash_ref": stash_ref,
                    },
                    self.aragora_path,
                )

        # Save state
        self._save_state(
            {
                "phase": "implement",
                "stage": "executing",
                "total_tasks": len(plan.tasks),
                "completed_tasks": len(completed),
            }
        )

        # Execute tasks
        if not self._executor:
            self._log("  No executor configured, falling back to legacy mode")
            return await self._legacy_implement(design)

        def on_task_complete(task_id: str, result):
            completed.add(task_id)
            self._log(f"  Task {task_id}: {'completed' if result.success else 'failed'}")
            if self._progress_saver:
                self._progress_saver(
                    {
                        "plan": plan,
                        "completed_tasks": list(completed),
                        "current_task": None,
                        "git_stash_ref": stash_ref,
                    },
                    self.aragora_path,
                )

        try:
            results = await self._executor.execute_plan(
                plan.tasks,
                completed,
                on_task_complete=on_task_complete,
            )

            all_success = all(r.success for r in results)
            tasks_completed = len([r for r in results if r.success])

            if all_success and tasks_completed == len(plan.tasks):
                if self._progress_clearer:
                    self._progress_clearer(self.aragora_path)
                self._log(f"  All {tasks_completed} tasks completed successfully")

                diff = await self._get_git_diff()
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end",
                    "implement",
                    self.cycle_count,
                    True,
                    phase_duration,
                    {"tasks_completed": tasks_completed},
                )

                return ImplementResult(
                    success=True,
                    data={
                        "tasks_completed": tasks_completed,
                        "tasks_total": len(plan.tasks),
                    },
                    duration_seconds=phase_duration,
                    files_modified=await self._get_modified_files(),
                    diff_summary=diff[:2000] if diff else "",
                )
            else:
                failed = [r for r in results if not r.success]
                self._log(f"  {len(failed)} tasks failed")
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end",
                    "implement",
                    self.cycle_count,
                    False,
                    phase_duration,
                    {"tasks_failed": len(failed)},
                )

                diff = await self._get_git_diff()
                return ImplementResult(
                    success=False,
                    error=failed[0].error if failed else "Unknown error",
                    data={
                        "tasks_completed": tasks_completed,
                        "tasks_total": len(plan.tasks),
                    },
                    duration_seconds=phase_duration,
                    files_modified=await self._get_modified_files(),
                    diff_summary=diff[:2000] if diff else "",
                )

        except Exception as e:
            self._log(f"  Catastrophic failure: {e}")
            self._log("  Rolling back changes...")
            await self._git_stash_pop(stash_ref)
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end",
                "implement",
                self.cycle_count,
                False,
                phase_duration,
                {"error": str(e)},
            )
            self._stream_emit("on_error", "implement", str(e), True)

            return ImplementResult(
                success=False,
                error=str(e),
                data={},
                duration_seconds=phase_duration,
                files_modified=[],
                diff_summary="",
            )

    async def _legacy_implement(self, design: str) -> ImplementResult:
        """Legacy single-Codex implementation (fallback)."""
        phase_start = datetime.now()
        self._log("  Using legacy Codex-only mode...")

        prompt = f"""{SAFETY_PREAMBLE}

Implement this design in the aragora codebase:

{design}

Write the actual code. Create or modify files as needed.
Follow aragora's existing code style and patterns.
Include docstrings and type hints.

CRITICAL SAFETY RULES:
- NEVER delete or modify these protected files: {self.protected_files}
- NEVER remove existing functionality - only ADD new code
- NEVER simplify code by removing features - complexity is acceptable
- If a file seems "too complex", DO NOT simplify it
- Preserve ALL existing imports, classes, and functions"""

        try:
            proc = await asyncio.create_subprocess_exec(
                "codex",
                "exec",
                "-C",
                str(self.aragora_path),
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1200)

            phase_duration = (datetime.now() - phase_start).total_seconds()
            success = proc.returncode == 0

            self._stream_emit(
                "on_phase_end", "implement", self.cycle_count, success, phase_duration, {}
            )

            diff = await self._get_git_diff()
            return ImplementResult(
                success=success,
                data={"output": stdout.decode() if stdout else ""},
                duration_seconds=phase_duration,
                files_modified=await self._get_modified_files(),
                diff_summary=diff[:2000] if diff else "",
            )

        except asyncio.TimeoutError:
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end",
                "implement",
                self.cycle_count,
                False,
                phase_duration,
                {"error": "timeout"},
            )
            return ImplementResult(
                success=False,
                error="Implementation timed out",
                data={},
                duration_seconds=phase_duration,
                files_modified=[],
                diff_summary="",
            )
        except Exception as e:
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end",
                "implement",
                self.cycle_count,
                False,
                phase_duration,
                {"error": str(e)},
            )
            return ImplementResult(
                success=False,
                error=str(e),
                data={},
                duration_seconds=phase_duration,
                files_modified=[],
                diff_summary="",
            )

    async def _git_stash_create(self) -> Optional[str]:
        """Create a git stash for rollback."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "stash",
                "create",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip() if proc.returncode == 0 else None
        except (OSError, FileNotFoundError, asyncio.TimeoutError) as e:
            self._log(f"  [git] Failed to create stash: {e}")
            return None

    async def _git_stash_pop(self, stash_ref: Optional[str]) -> bool:
        """Pop a git stash to rollback changes."""
        if not stash_ref:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "stash",
                "apply",
                stash_ref,
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return True
        except (OSError, FileNotFoundError, asyncio.TimeoutError) as e:
            self._log(f"  [git] Failed to apply stash: {e}")
            return False

    async def _get_git_diff(self) -> str:
        """Get current git diff."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--stat",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode() if proc.returncode == 0 else ""
        except (OSError, FileNotFoundError, asyncio.TimeoutError) as e:
            self._log(f"  [git] Failed to get diff: {e}")
            return ""

    async def _get_modified_files(self) -> List[str]:
        """Get list of modified files."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--name-only",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                return [f.strip() for f in stdout.decode().strip().split("\n") if f.strip()]
        except (OSError, FileNotFoundError, asyncio.TimeoutError) as e:
            self._log(f"  [git] Failed to get modified files: {e}")
        return []


__all__ = ["ImplementPhase"]
