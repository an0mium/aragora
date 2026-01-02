"""
Hybrid multi-model executor.

Updated routing based on empirical performance data (Dec 2025):
- Claude: ALL implementation tasks (37% faster than alternatives, best code quality)
- Codex: Code review / QA after implementation (high quality review, latency-tolerant)
- Gemini: Planning only (handled by planner.py, leverages 1M context window)

Research sources:
- Claude completed projects in 1h17m vs Gemini's 2h2m (37% faster)
- Codex has known latency issues (5-20 min for simple tasks per GitHub issues)
- Claude produces "production-ready codebase with organized folders"
- Codex excels at review/QA where latency isn't critical
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

from aragora.agents.cli_agents import ClaudeAgent, CodexAgent

from .types import ImplementTask, TaskResult


TASK_PROMPT_TEMPLATE = """Implement this task in the codebase:

## Task
{description}

## Files to Create/Modify
{files}

## Repository
Working directory: {repo_path}

## Instructions

1. Create or modify the files listed above
2. Follow existing code style and patterns
3. Include docstrings and type hints
4. Make only the changes necessary for this task
5. Do not break existing functionality

IMPORTANT: Only make changes that are safe and reversible.
"""


class HybridExecutor:
    """
    Executes implementation tasks using Claude, with optional Codex review.

    Updated routing strategy (Dec 2025):
    - ALL tasks: Claude (fastest, best quality for implementation)
    - Post-implementation: Codex review (optional QA phase)
    - Fallback: Codex if Claude times out (resilience)

    Rationale:
    - Codex has severe latency issues (GitHub #5149, #1811, #6990)
    - Claude is 37% faster and produces more organized code
    - Codex quality shines in review mode where latency is acceptable

    Resilience features (Jan 2026):
    - Retry failed tasks with 2x timeout
    - Model fallback on timeout (Claude → Codex)
    - Continue execution after failures (collect, retry at end)
    """

    def __init__(
        self,
        repo_path: Path,
        claude_timeout: int = 1200,  # 20 min - doubled from 600
        codex_timeout: int = 1200,   # 20 min - doubled from 600
        max_retries: int = 2,
    ):
        self.repo_path = repo_path

        # Initialize agents lazily (created on first use)
        self._claude: Optional[ClaudeAgent] = None
        self._codex: Optional[CodexAgent] = None

        self.claude_timeout = claude_timeout
        self.codex_timeout = codex_timeout
        self.max_retries = max_retries

    @property
    def claude(self) -> ClaudeAgent:
        if self._claude is None:
            self._claude = ClaudeAgent(
                name="claude-implementer",
                model="claude",
                role="implementer",
                timeout=self.claude_timeout,
            )
            self._claude.system_prompt = """You are implementing code changes in a repository.
Be precise, follow existing patterns, and make only necessary changes.
Include proper type hints and docstrings."""
        return self._claude

    @property
    def codex(self) -> CodexAgent:
        if self._codex is None:
            self._codex = CodexAgent(
                name="codex-specialist",
                model="o3",
                role="implementer",
                timeout=self.codex_timeout,
            )
            self._codex.system_prompt = """You are implementing a focused code change.
Make only the changes specified. Follow existing code style."""
        return self._codex

    def _select_agent(self, complexity: str):
        """Select the appropriate agent based on task complexity.

        Updated Dec 2025: Always use Claude for implementation.
        Codex latency issues make it unsuitable for interactive implementation.
        Codex is now used only for post-implementation review.
        """
        # Always use Claude for implementation (fastest, best quality)
        # Complexity only affects timeout expectations
        return self.claude, "claude"

    def _build_prompt(self, task: ImplementTask) -> str:
        """Build the implementation prompt for a task."""
        files_str = "\n".join(f"- {f}" for f in task.files) if task.files else "- (determine from description)"

        return TASK_PROMPT_TEMPLATE.format(
            description=task.description,
            files=files_str,
            repo_path=str(self.repo_path),
        )

    def _get_git_diff(self) -> str:
        """Get the current git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout
        except Exception:
            return ""

    async def execute_task(
        self,
        task: ImplementTask,
        attempt: int = 1,
        use_fallback: bool = False,
    ) -> TaskResult:
        """
        Execute a single implementation task with retry and fallback support.

        Args:
            task: The task to execute
            attempt: Current attempt number (1-based)
            use_fallback: If True, use Codex instead of Claude

        Returns:
            TaskResult with success status and diff
        """
        # Select agent - use fallback (Codex) if primary (Claude) failed
        if use_fallback:
            agent = self.codex
            model_name = "codex-fallback"
            # Use 2x timeout for fallback
            agent.timeout = self.codex_timeout * 2
            print(f"  Retry [{task.complexity}] {task.id} with {model_name} (attempt {attempt})...")
        else:
            agent, model_name = self._select_agent(task.complexity)
            # Increase timeout on retry
            if attempt > 1:
                agent.timeout = self.claude_timeout * attempt
                print(f"  Retry [{task.complexity}] {task.id} with {model_name} (attempt {attempt}, timeout {agent.timeout}s)...")
            else:
                print(f"  Executing [{task.complexity}] {task.id} with {model_name}...")

        prompt = self._build_prompt(task)
        start_time = time.time()

        try:
            # Execute with the selected agent
            await agent.generate(prompt, context=[])

            # Get the diff to see what changed
            diff = self._get_git_diff()
            duration = time.time() - start_time

            print(f"    Completed in {duration:.1f}s")
            if diff:
                print(f"    Changes:\n{diff[:200]}...")

            return TaskResult(
                task_id=task.id,
                success=True,
                diff=diff,
                model_used=model_name,
                duration_seconds=duration,
            )

        except TimeoutError as e:
            duration = time.time() - start_time
            print(f"    Timeout after {duration:.1f}s")
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Timeout: {e}",
                model_used=model_name,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"    Error: {e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                model_used=model_name,
                duration_seconds=duration,
            )

    async def execute_task_with_retry(self, task: ImplementTask) -> TaskResult:
        """
        Execute a task with automatic retry and model fallback.

        Retry strategy:
        1. First attempt with Claude (primary)
        2. If timeout, retry with Claude + 2x timeout
        3. If still fails, try Codex as fallback with 2x timeout

        Returns:
            Best TaskResult from attempts
        """
        # Attempt 1: Claude with normal timeout
        result = await self.execute_task(task, attempt=1, use_fallback=False)
        if result.success:
            return result

        # Check if it was a timeout (worth retrying) vs other error
        is_timeout = result.error and "timeout" in result.error.lower()

        if is_timeout and self.max_retries >= 2:
            # Attempt 2: Claude with 2x timeout
            print(f"    Retrying {task.id} with extended timeout...")
            result = await self.execute_task(task, attempt=2, use_fallback=False)
            if result.success:
                return result

        if is_timeout and self.max_retries >= 3:
            # Attempt 3: Fallback to Codex
            print(f"    Falling back to Codex for {task.id}...")
            result = await self.execute_task(task, attempt=3, use_fallback=True)

        return result

    async def execute_plan(
        self,
        tasks: list[ImplementTask],
        completed: set[str],
        on_task_complete=None,
        stop_on_failure: bool = False,
    ) -> list[TaskResult]:
        """
        Execute all tasks in a plan, respecting dependencies.

        Updated Jan 2026: Now continues after failures by default and retries.

        Args:
            tasks: List of tasks to execute
            completed: Set of already-completed task IDs
            on_task_complete: Optional callback after each task
            stop_on_failure: If True, stop on first failure (legacy behavior)

        Returns:
            List of TaskResults for executed tasks
        """
        results = []
        failed_tasks = []

        # First pass: execute all tasks, collecting failures
        for task in tasks:
            # Skip already completed
            if task.id in completed:
                continue

            # Check dependencies
            deps_met = all(dep in completed for dep in task.dependencies)
            if not deps_met:
                print(f"  Skipping {task.id} - dependencies not met")
                continue

            # Execute with retry
            result = await self.execute_task_with_retry(task)
            results.append(result)

            if result.success:
                completed.add(task.id)
                if on_task_complete:
                    on_task_complete(task.id, result)
            else:
                failed_tasks.append(task)
                if stop_on_failure:
                    print(f"  Stopping execution due to failure in {task.id}")
                    break
                else:
                    print(f"  Task {task.id} failed, continuing with remaining tasks...")

        # Second pass: retry failed tasks once more (dependencies may now be met)
        if failed_tasks and not stop_on_failure:
            print(f"\n  Retrying {len(failed_tasks)} failed tasks...")
            for task in failed_tasks:
                # Check if dependencies are now met
                deps_met = all(dep in completed for dep in task.dependencies)
                if not deps_met:
                    print(f"  Skipping retry of {task.id} - dependencies still not met")
                    continue

                # Already tried with retry, try one more time with max timeout
                print(f"  Final retry for {task.id}...")
                result = await self.execute_task(task, attempt=self.max_retries + 1, use_fallback=True)

                # Update results (replace the failed one)
                for i, r in enumerate(results):
                    if r.task_id == task.id:
                        results[i] = result
                        break

                if result.success:
                    completed.add(task.id)
                    if on_task_complete:
                        on_task_complete(task.id, result)

        return results

    async def review_with_codex(self, diff: str, timeout: int = 2400) -> dict:  # 40 min - Codex is slow but thorough
        """
        Run Codex code review on implemented changes.

        Codex is slow (~5-20min) but produces high-quality review.
        Use this as a QA step after Claude implementation.

        Args:
            diff: The git diff to review
            timeout: Max time to wait (default 10 min)

        Returns:
            dict with 'approved', 'issues', and 'suggestions'
        """
        if not diff.strip():
            return {"approved": True, "issues": [], "suggestions": []}

        review_prompt = f"""Review this code change for quality and safety issues.

## Git Diff
```
{diff}
```

## Review Checklist
1. Are there any bugs or logic errors?
2. Are there security vulnerabilities (injection, XSS, etc.)?
3. Does the code follow consistent style?
4. Are there missing error handlers or edge cases?
5. Is there unnecessary complexity that could be simplified?

## Response Format
Provide your review as:
- APPROVED: yes/no
- ISSUES: List any problems that MUST be fixed
- SUGGESTIONS: List any improvements that would be nice

Be concise and actionable."""

        print("  Running Codex code review (this may take several minutes)...")
        start_time = time.time()

        try:
            # Use codex with extended timeout for review
            self._codex = CodexAgent(
                name="codex-reviewer",
                model="o3",
                role="reviewer",
                timeout=timeout,
            )
            self._codex.system_prompt = """You are a senior code reviewer.
Focus on correctness, security, and maintainability.
Be constructive but thorough."""

            response = await self._codex.generate(review_prompt, context=[])
            duration = time.time() - start_time

            print(f"    Review completed in {duration:.1f}s")

            # Parse response (basic parsing)
            response_lower = response.lower() if response else ""
            approved = "approved: yes" in response_lower or "approved:yes" in response_lower

            return {
                "approved": approved,
                "review": response,
                "duration_seconds": duration,
                "model": "codex-o3",
            }

        except Exception as e:
            duration = time.time() - start_time
            print(f"    Review failed after {duration:.1f}s: {e}")
            return {
                "approved": None,
                "error": str(e),
                "duration_seconds": duration,
            }
