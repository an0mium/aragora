"""
Autonomous Development Orchestrator.

Coordinates heterogeneous agents to develop specs, implement features,
and maintain the Aragora codebase with minimal human intervention.

This module ties together:
- TaskDecomposer: Break high-level goals into subtasks
- WorkflowEngine: Execute multi-step workflows with checkpoints
- NomicLoop: Run improvement cycles on individual tasks
- Gates: Approval checkpoints for safety

Usage:
    from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

    orchestrator = AutonomousOrchestrator()

    # High-level goal with track hints
    result = await orchestrator.execute_goal(
        goal="Maximize utility for SME SMB users",
        tracks=["sme", "qa"],
        max_cycles=5,
    )
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aragora.nomic.task_decomposer import (
    TaskDecomposer,
    TaskDecomposition,
    SubTask,
    DecomposerConfig,
)
from aragora.workflow.engine import WorkflowEngine, get_workflow_engine
from aragora.workflow.types import (
    WorkflowDefinition,
    StepDefinition,
)
from aragora.observability import get_logger

logger = get_logger(__name__)


class Track(Enum):
    """Development tracks for domain-based routing."""

    SME = "sme"  # Small business features
    DEVELOPER = "developer"  # SDK, API, docs
    SELF_HOSTED = "self_hosted"  # Docker, deployment, ops
    QA = "qa"  # Tests, CI/CD
    CORE = "core"  # Core debate engine (requires approval)


@dataclass
class TrackConfig:
    """Configuration for a development track."""

    name: str
    folders: List[str]  # Folders this track owns
    protected_folders: List[str] = field(default_factory=list)  # Cannot modify
    agent_types: List[str] = field(default_factory=list)  # Preferred agents
    max_concurrent_tasks: int = 2


# Default track configurations aligned with AGENT_ASSIGNMENTS.md
DEFAULT_TRACK_CONFIGS: Dict[Track, TrackConfig] = {
    Track.SME: TrackConfig(
        name="SME",
        folders=["aragora/live/", "aragora/server/handlers/"],
        protected_folders=["aragora/debate/", "aragora/agents/", "core.py"],
        agent_types=["claude", "gemini"],
        max_concurrent_tasks=2,
    ),
    Track.DEVELOPER: TrackConfig(
        name="Developer",
        folders=["sdk/", "docs/", "tests/sdk/"],
        protected_folders=["aragora/debate/", "aragora/live/src/app/"],
        agent_types=["claude", "codex"],
        max_concurrent_tasks=2,
    ),
    Track.SELF_HOSTED: TrackConfig(
        name="Self-Hosted",
        folders=["scripts/", "docker/", "docs/deployment/", "aragora/backup/"],
        protected_folders=["aragora/debate/", "aragora/server/handlers/"],
        agent_types=["claude", "codex"],
        max_concurrent_tasks=1,
    ),
    Track.QA: TrackConfig(
        name="QA",
        folders=["tests/", "aragora/live/e2e/", ".github/workflows/"],
        protected_folders=["aragora/debate/"],
        agent_types=["claude", "gemini"],
        max_concurrent_tasks=3,
    ),
    Track.CORE: TrackConfig(
        name="Core",
        folders=["aragora/debate/", "aragora/agents/", "aragora/memory/"],
        protected_folders=[],  # Can modify core, but requires approval
        agent_types=["claude"],  # Only Claude for core changes
        max_concurrent_tasks=1,
    ),
}


@dataclass
class AgentAssignment:
    """Assignment of a subtask to an agent."""

    subtask: SubTask
    track: Track
    agent_type: str
    priority: int = 0
    status: str = "pending"  # pending, running, completed, failed
    attempt_count: int = 0
    max_attempts: int = 3
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class OrchestrationResult:
    """Result of an orchestration run."""

    goal: str
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    skipped_subtasks: int
    assignments: List[AgentAssignment]
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    summary: str = ""


class AgentRouter:
    """
    Routes subtasks to appropriate agents based on domain and track.

    Uses heuristics to determine:
    1. Which track owns a subtask (based on file patterns)
    2. Which agent type is best suited (based on task complexity)
    """

    def __init__(self, track_configs: Optional[Dict[Track, TrackConfig]] = None):
        self.track_configs = track_configs or DEFAULT_TRACK_CONFIGS
        self._file_to_track_cache: Dict[str, Track] = {}

    def determine_track(self, subtask: SubTask) -> Track:
        """Determine which track should handle a subtask."""
        # Check file scope first
        for file_path in subtask.file_scope:
            track = self._file_to_track(file_path)
            if track:
                return track

        # Infer from task description
        description_lower = subtask.description.lower()
        title_lower = subtask.title.lower()
        combined = f"{title_lower} {description_lower}"

        # Track detection patterns
        patterns = {
            Track.SME: ["ui", "frontend", "user", "dashboard", "workspace", "admin"],
            Track.DEVELOPER: ["sdk", "api", "documentation", "docs", "client"],
            Track.SELF_HOSTED: [
                "docker",
                "deploy",
                "backup",
                "restore",
                "ops",
                "kubernetes",
            ],
            Track.QA: ["test", "e2e", "ci", "coverage", "quality", "playwright"],
            Track.CORE: ["debate", "consensus", "arena", "agent", "memory"],
        }

        for track, keywords in patterns.items():
            if any(kw in combined for kw in keywords):
                return track

        # Default to developer track for unclassified tasks
        return Track.DEVELOPER

    def _file_to_track(self, file_path: str) -> Optional[Track]:
        """Map a file path to its owning track."""
        if file_path in self._file_to_track_cache:
            return self._file_to_track_cache[file_path]

        for track, config in self.track_configs.items():
            for folder in config.folders:
                if file_path.startswith(folder):
                    self._file_to_track_cache[file_path] = track
                    return track

        return None

    def select_agent_type(self, subtask: SubTask, track: Track) -> str:
        """Select the best agent type for a subtask."""
        config = self.track_configs.get(track, DEFAULT_TRACK_CONFIGS[Track.DEVELOPER])

        if not config.agent_types:
            return "claude"  # Default

        # High complexity -> Claude (better reasoning)
        if subtask.estimated_complexity == "high":
            return "claude"

        # Code generation -> prefer Codex
        if "implement" in subtask.title.lower() or "code" in subtask.description.lower():
            if "codex" in config.agent_types:
                return "codex"

        # Default to first preferred agent
        return config.agent_types[0]

    def check_conflicts(
        self,
        subtask: SubTask,
        active_assignments: List[AgentAssignment],
    ) -> List[str]:
        """Check for potential conflicts with active assignments."""
        conflicts = []

        for assignment in active_assignments:
            if assignment.status != "running":
                continue

            # Check file overlap
            active_files = set(assignment.subtask.file_scope)
            new_files = set(subtask.file_scope)
            overlap = active_files & new_files

            if overlap:
                conflicts.append(f"File conflict with {assignment.subtask.id}: {overlap}")

            # Check track overlap (some tracks shouldn't run in parallel)
            if assignment.track == Track.CORE and self.determine_track(subtask) == Track.CORE:
                conflicts.append("Core track conflict: only one core task at a time")

        return conflicts


class FeedbackLoop:
    """
    Manages feedback from verification back to earlier phases.

    When verification fails, determines:
    1. Root cause (test failure, lint error, etc.)
    2. Which phase to return to (design, implement, or new subtask)
    3. How to modify the approach
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self._iteration_counts: Dict[str, int] = {}

    def analyze_failure(
        self,
        assignment: AgentAssignment,
        error_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze a failure and determine next steps."""
        subtask_id = assignment.subtask.id
        self._iteration_counts[subtask_id] = self._iteration_counts.get(subtask_id, 0) + 1

        if self._iteration_counts[subtask_id] >= self.max_iterations:
            return {
                "action": "escalate",
                "reason": f"Max iterations ({self.max_iterations}) reached",
                "require_human": True,
            }

        error_type = error_info.get("type", "unknown")
        error_message = error_info.get("message", "")

        # Test failures -> adjust implementation
        if error_type == "test_failure":
            return {
                "action": "retry_implement",
                "reason": "Test failures require implementation adjustment",
                "hints": self._extract_test_hints(error_message),
            }

        # Lint/type errors -> quick fix
        if error_type in ("lint_error", "type_error"):
            return {
                "action": "quick_fix",
                "reason": "Static analysis errors can be auto-fixed",
                "hints": error_message,
            }

        # Design issues -> revisit design
        if error_type == "design_issue":
            return {
                "action": "redesign",
                "reason": "Implementation revealed design flaws",
                "hints": error_info.get("suggestion", ""),
            }

        # Unknown -> escalate
        return {
            "action": "escalate",
            "reason": f"Unknown error type: {error_type}",
            "require_human": True,
        }

    def _extract_test_hints(self, error_message: str) -> str:
        """Extract hints from test failure messages."""
        lines = error_message.split("\n")
        hints = []

        for line in lines:
            if "AssertionError" in line or "Expected" in line or "Actual" in line:
                hints.append(line.strip())

        return "\n".join(hints[:5]) if hints else "Review test output"


class AutonomousOrchestrator:
    """
    Orchestrates autonomous development across multiple agents and tracks.

    Coordinates:
    - Goal decomposition into track-aligned subtasks
    - Agent assignment based on domain expertise
    - Parallel execution with conflict detection
    - Feedback loops for failed tasks
    - Human approval checkpoints
    """

    def __init__(
        self,
        aragora_path: Optional[Path] = None,
        track_configs: Optional[Dict[Track, TrackConfig]] = None,
        workflow_engine: Optional[WorkflowEngine] = None,
        task_decomposer: Optional[TaskDecomposer] = None,
        require_human_approval: bool = False,
        max_parallel_tasks: int = 4,
        on_checkpoint: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            aragora_path: Path to the aragora project
            track_configs: Custom track configurations
            workflow_engine: Custom workflow engine
            task_decomposer: Custom task decomposer
            require_human_approval: Whether to require approval at gates
            max_parallel_tasks: Maximum concurrent tasks across all tracks
            on_checkpoint: Callback for checkpoint events
        """
        self.aragora_path = aragora_path or Path.cwd()
        self.track_configs = track_configs or DEFAULT_TRACK_CONFIGS
        self.workflow_engine = workflow_engine or get_workflow_engine()
        self.task_decomposer = task_decomposer or TaskDecomposer(
            config=DecomposerConfig(complexity_threshold=4)
        )
        self.require_human_approval = require_human_approval
        self.max_parallel_tasks = max_parallel_tasks
        self.on_checkpoint = on_checkpoint

        self.router = AgentRouter(self.track_configs)
        self.feedback_loop = FeedbackLoop()

        # State
        self._active_assignments: List[AgentAssignment] = []
        self._completed_assignments: List[AgentAssignment] = []
        self._orchestration_id: Optional[str] = None

    async def execute_goal(
        self,
        goal: str,
        tracks: Optional[List[str]] = None,
        max_cycles: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Execute a high-level goal by decomposing and orchestrating subtasks.

        Args:
            goal: High-level goal description
            tracks: Optional list of track names to focus on
            max_cycles: Maximum improvement cycles per subtask
            context: Additional context for the orchestration

        Returns:
            OrchestrationResult with completion status
        """
        self._orchestration_id = f"orch_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now(timezone.utc)
        context = context or {}

        logger.info(
            "orchestration_started",
            orchestration_id=self._orchestration_id,
            goal=goal[:100],
            tracks=tracks,
        )

        self._checkpoint("started", {"goal": goal, "tracks": tracks})

        try:
            # Step 1: Decompose the goal
            decomposition = await self._decompose_goal(goal, tracks)
            self._checkpoint("decomposed", {"subtask_count": len(decomposition.subtasks)})

            if not decomposition.subtasks:
                return OrchestrationResult(
                    goal=goal,
                    total_subtasks=0,
                    completed_subtasks=0,
                    failed_subtasks=0,
                    skipped_subtasks=0,
                    assignments=[],
                    duration_seconds=0,
                    success=True,
                    summary="Goal decomposed to zero subtasks (may be trivial)",
                )

            # Step 2: Create assignments
            assignments = self._create_assignments(decomposition, tracks)
            self._checkpoint("assigned", {"assignment_count": len(assignments)})

            # Step 3: Execute assignments
            await self._execute_assignments(assignments, max_cycles)

            # Step 4: Compute result
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            completed = sum(1 for a in assignments if a.status == "completed")
            failed = sum(1 for a in assignments if a.status == "failed")
            skipped = sum(1 for a in assignments if a.status == "skipped")

            result = OrchestrationResult(
                goal=goal,
                total_subtasks=len(assignments),
                completed_subtasks=completed,
                failed_subtasks=failed,
                skipped_subtasks=skipped,
                assignments=assignments,
                duration_seconds=duration,
                success=failed == 0,
                summary=self._generate_summary(assignments),
            )

            self._checkpoint("completed", {"result": result.summary})
            logger.info(
                "orchestration_completed",
                orchestration_id=self._orchestration_id,
                completed=completed,
                failed=failed,
                duration_seconds=duration,
            )

            return result

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.exception(
                "orchestration_failed",
                orchestration_id=self._orchestration_id,
                error=str(e),
            )

            return OrchestrationResult(
                goal=goal,
                total_subtasks=len(self._active_assignments),
                completed_subtasks=0,
                failed_subtasks=len(self._active_assignments),
                skipped_subtasks=0,
                assignments=self._active_assignments,
                duration_seconds=duration,
                success=False,
                error=str(e),
            )

    async def _decompose_goal(
        self,
        goal: str,
        tracks: Optional[List[str]] = None,
    ) -> TaskDecomposition:
        """Decompose a high-level goal into subtasks."""
        # Enrich goal with track context if provided
        if tracks:
            track_context = f"\n\nFocus tracks: {', '.join(tracks)}"
            enriched_goal = f"{goal}{track_context}"
        else:
            enriched_goal = goal

        return self.task_decomposer.analyze(enriched_goal)

    def _create_assignments(
        self,
        decomposition: TaskDecomposition,
        tracks: Optional[List[str]] = None,
    ) -> List[AgentAssignment]:
        """Create agent assignments from decomposed subtasks."""
        assignments = []
        allowed_tracks = {Track(t.lower()) for t in tracks} if tracks else set(Track)

        for i, subtask in enumerate(decomposition.subtasks):
            track = self.router.determine_track(subtask)

            # Skip if track not in allowed list
            if track not in allowed_tracks:
                logger.debug(f"Skipping subtask {subtask.id}: track {track} not allowed")
                continue

            agent_type = self.router.select_agent_type(subtask, track)

            assignments.append(
                AgentAssignment(
                    subtask=subtask,
                    track=track,
                    agent_type=agent_type,
                    priority=len(decomposition.subtasks) - i,  # Higher index = lower priority
                )
            )

        # Sort by priority (highest first)
        assignments.sort(key=lambda a: a.priority, reverse=True)
        return assignments

    async def _execute_assignments(
        self,
        assignments: List[AgentAssignment],
        max_cycles: int,
    ) -> None:
        """Execute assignments with parallel coordination."""
        pending = list(assignments)
        running: List[asyncio.Task] = []

        while pending or running:
            # Start new tasks up to max parallel
            while pending and len(running) < self.max_parallel_tasks:
                assignment = pending.pop(0)

                # Check for conflicts
                conflicts = self.router.check_conflicts(
                    assignment.subtask,
                    [a for a in assignments if a.status == "running"],
                )

                if conflicts:
                    logger.warning(
                        "assignment_delayed",
                        subtask_id=assignment.subtask.id,
                        conflicts=conflicts,
                    )
                    pending.append(assignment)  # Re-queue
                    break  # Wait for running tasks to complete

                # Start the assignment
                assignment.status = "running"
                assignment.started_at = datetime.now(timezone.utc)
                self._active_assignments.append(assignment)

                task = asyncio.create_task(self._execute_single_assignment(assignment, max_cycles))
                running.append(task)

            if not running:
                break

            # Wait for at least one task to complete
            done, pending_tasks = await asyncio.wait(
                running,
                return_when=asyncio.FIRST_COMPLETED,
            )
            running = list(pending_tasks)

            # Process completed tasks
            for task in done:
                try:
                    await task
                except Exception as e:
                    logger.exception(f"Task failed: {e}")

    async def _execute_single_assignment(
        self,
        assignment: AgentAssignment,
        max_cycles: int,
    ) -> None:
        """Execute a single assignment with retry logic."""
        subtask = assignment.subtask

        logger.info(
            "assignment_started",
            subtask_id=subtask.id,
            track=assignment.track.value,
            agent=assignment.agent_type,
        )

        try:
            # Build workflow for this subtask
            workflow = self._build_subtask_workflow(assignment)

            # Execute workflow
            result = await self.workflow_engine.execute(
                workflow,
                inputs={
                    "subtask": subtask.description,
                    "files": subtask.file_scope,
                    "complexity": subtask.estimated_complexity,
                    "max_cycles": max_cycles,
                },
            )

            if result.success:
                assignment.status = "completed"
                assignment.result = {"workflow_result": result.final_output}
            else:
                # Handle failure with feedback loop
                feedback = self.feedback_loop.analyze_failure(
                    assignment,
                    {"type": "workflow_failure", "message": result.error or ""},
                )

                if feedback["action"] == "escalate":
                    assignment.status = "failed"
                    assignment.result = {"error": result.error, "feedback": feedback}
                else:
                    # Retry based on feedback
                    assignment.attempt_count += 1
                    if assignment.attempt_count < assignment.max_attempts:
                        await self._execute_single_assignment(assignment, max_cycles)
                    else:
                        assignment.status = "failed"

        except Exception as e:
            logger.exception(
                "assignment_failed",
                subtask_id=subtask.id,
                error=str(e),
            )
            assignment.status = "failed"
            assignment.result = {"error": str(e)}

        finally:
            assignment.completed_at = datetime.now(timezone.utc)
            self._completed_assignments.append(assignment)
            self._active_assignments.remove(assignment)

    def _build_subtask_workflow(self, assignment: AgentAssignment) -> WorkflowDefinition:
        """Build a workflow definition for a subtask."""
        subtask = assignment.subtask

        # Create workflow with phases aligned to nomic loop
        steps = [
            StepDefinition(
                id="design",
                name="Design Solution",
                step_type="agent",
                config={
                    "agent_type": assignment.agent_type,
                    "prompt_template": "design",
                    "task": subtask.description,
                },
                next_steps=["implement"],
            ),
            StepDefinition(
                id="implement",
                name="Implement Changes",
                step_type="agent",
                config={
                    "agent_type": assignment.agent_type,
                    "prompt_template": "implement",
                    "files": subtask.file_scope,
                },
                next_steps=["verify"],
            ),
            StepDefinition(
                id="verify",
                name="Verify Changes",
                step_type="agent",
                config={
                    "agent_type": "claude",  # Always use Claude for verification
                    "prompt_template": "verify",
                    "run_tests": True,
                },
                next_steps=[],
            ),
        ]

        return WorkflowDefinition(
            id=f"subtask_{subtask.id}",
            name=f"Execute: {subtask.title}",
            description=subtask.description,
            steps=steps,
            entry_step="design",
        )

    def _checkpoint(self, phase: str, data: Dict[str, Any]) -> None:
        """Create a checkpoint for the orchestration."""
        if self.on_checkpoint:
            self.on_checkpoint(
                phase,
                {
                    "orchestration_id": self._orchestration_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **data,
                },
            )

    def _generate_summary(self, assignments: List[AgentAssignment]) -> str:
        """Generate a summary of the orchestration."""
        by_track: Dict[str, List[str]] = {}
        for a in assignments:
            track_name = a.track.value
            if track_name not in by_track:
                by_track[track_name] = []
            status_icon = "+" if a.status == "completed" else "-"
            by_track[track_name].append(f"{status_icon} {a.subtask.title}")

        lines = ["Orchestration Summary:"]
        for track, tasks in by_track.items():
            lines.append(f"\n{track.upper()}:")
            for task in tasks:
                lines.append(f"  {task}")

        return "\n".join(lines)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def execute_track(
        self,
        track: str,
        focus_areas: Optional[List[str]] = None,
        max_cycles: int = 3,
    ) -> OrchestrationResult:
        """
        Execute work for a specific track.

        Args:
            track: Track name (sme, developer, self_hosted, qa)
            focus_areas: Optional list of focus areas within the track
            max_cycles: Maximum cycles per subtask

        Returns:
            OrchestrationResult
        """
        goal = f"Improve {track} track capabilities"
        if focus_areas:
            goal += f", focusing on: {', '.join(focus_areas)}"

        return await self.execute_goal(
            goal=goal,
            tracks=[track],
            max_cycles=max_cycles,
        )

    def get_active_assignments(self) -> List[AgentAssignment]:
        """Get currently active assignments."""
        return self._active_assignments.copy()

    def get_completed_assignments(self) -> List[AgentAssignment]:
        """Get completed assignments."""
        return self._completed_assignments.copy()


# Singleton instance
_orchestrator_instance: Optional[AutonomousOrchestrator] = None


def get_orchestrator(
    **kwargs: Any,
) -> AutonomousOrchestrator:
    """Get or create the singleton orchestrator instance."""
    global _orchestrator_instance

    if _orchestrator_instance is None:
        _orchestrator_instance = AutonomousOrchestrator(**kwargs)

    return _orchestrator_instance


def reset_orchestrator() -> None:
    """Reset the singleton (for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = None


__all__ = [
    "AutonomousOrchestrator",
    "AgentRouter",
    "FeedbackLoop",
    "Track",
    "TrackConfig",
    "AgentAssignment",
    "OrchestrationResult",
    "get_orchestrator",
    "reset_orchestrator",
]
