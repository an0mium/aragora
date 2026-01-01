"""
Hybrid multi-model executor.

Routes implementation tasks to the appropriate model based on complexity:
- simple: Codex (fast, reliable for isolated tasks)
- moderate/complex: Claude (better at multi-file coordination)
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
    Executes implementation tasks using the appropriate model.

    Complexity routing:
    - simple: CodexAgent (o3, fast for isolated tasks)
    - moderate: ClaudeAgent (claude, balanced)
    - complex: ClaudeAgent with extended timeout
    """

    def __init__(
        self,
        repo_path: Path,
        claude_timeout: int = 600,
        codex_timeout: int = 300,
    ):
        self.repo_path = repo_path

        # Initialize agents lazily (created on first use)
        self._claude: Optional[ClaudeAgent] = None
        self._codex: Optional[CodexAgent] = None

        self.claude_timeout = claude_timeout
        self.codex_timeout = codex_timeout

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
        """Select the appropriate agent based on task complexity."""
        if complexity == "simple":
            return self.codex, "codex"
        else:  # moderate or complex
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

    async def execute_task(self, task: ImplementTask) -> TaskResult:
        """
        Execute a single implementation task.

        Args:
            task: The task to execute

        Returns:
            TaskResult with success status and diff
        """
        agent, model_name = self._select_agent(task.complexity)
        prompt = self._build_prompt(task)

        print(f"  Executing [{task.complexity}] {task.id} with {model_name}...")
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

    async def execute_plan(
        self,
        tasks: list[ImplementTask],
        completed: set[str],
        on_task_complete=None,
    ) -> list[TaskResult]:
        """
        Execute all tasks in a plan, respecting dependencies.

        Args:
            tasks: List of tasks to execute
            completed: Set of already-completed task IDs
            on_task_complete: Optional callback after each task

        Returns:
            List of TaskResults for executed tasks
        """
        results = []

        for task in tasks:
            # Skip already completed
            if task.id in completed:
                continue

            # Check dependencies
            deps_met = all(dep in completed for dep in task.dependencies)
            if not deps_met:
                print(f"  Skipping {task.id} - dependencies not met")
                continue

            # Execute
            result = await self.execute_task(task)
            results.append(result)

            if result.success:
                completed.add(task.id)
                if on_task_complete:
                    on_task_complete(task.id, result)
            else:
                # Stop on first failure
                print(f"  Stopping execution due to failure in {task.id}")
                break

        return results
