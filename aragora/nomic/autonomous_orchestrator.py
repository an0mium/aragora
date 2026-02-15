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
import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Callable

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
    SECURITY = "security"  # Vulnerability scanning, auth hardening, OWASP


# Agents that have dedicated agentic coding harnesses (can edit files autonomously)
AGENTS_WITH_CODING_HARNESS = {"claude", "codex"}

# Mapping from model types to their KiloCode provider_id for coding tasks
# These models don't have dedicated harnesses but can use KiloCode
KILOCODE_PROVIDER_MAPPING = {
    "gemini": "openrouter/google/gemini-3-pro-preview",  # Gemini via OpenRouter
    "gemini-cli": "openrouter/google/gemini-3-pro-preview",
    "grok": "openrouter/x-ai/grok-4",  # Grok via OpenRouter
    "grok-cli": "openrouter/x-ai/grok-4",
    "deepseek": "openrouter/deepseek/deepseek-chat-v3-0324",  # DeepSeek via OpenRouter
    "qwen": "openrouter/qwen/qwen-2.5-coder-32b-instruct",  # Qwen via OpenRouter
}


@dataclass
class TrackConfig:
    """Configuration for a development track."""

    name: str
    folders: list[str]  # Folders this track owns
    protected_folders: list[str] = field(default_factory=list)  # Cannot modify
    agent_types: list[str] = field(default_factory=list)  # Preferred agents
    max_concurrent_tasks: int = 2
    # Whether to use KiloCode as coding harness for models without one
    use_kilocode_harness: bool = True


# Default track configurations aligned with AGENT_ASSIGNMENTS.md
DEFAULT_TRACK_CONFIGS: dict[Track, TrackConfig] = {
    Track.SME: TrackConfig(
        name="SME",
        folders=["aragora/live/", "aragora/server/handlers/"],
        protected_folders=["aragora/debate/", "aragora/agents/", "aragora/core/"],
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
    Track.SECURITY: TrackConfig(
        name="Security",
        folders=["aragora/security/", "aragora/audit/", "aragora/auth/", "aragora/rbac/"],
        protected_folders=["aragora/debate/"],  # Don't modify core debate
        agent_types=["claude"],  # Claude for security audits
        max_concurrent_tasks=2,
    ),
}


@dataclass
class HierarchyConfig:
    """Configuration for Planner/Worker/Judge agent hierarchy.

    When enabled, the orchestration workflow becomes:
      1. Planner designs the solution (design step)
      2. Plan approval gate: judge reviews the plan before implementation
      3. Workers implement the changes (implement step)
      4. Standard verification (verify step)
      5. Judge reviews the final result before completion

    This separation of concerns ensures no single agent both designs and
    approves its own work, improving quality and catching design flaws early.
    """

    enabled: bool = False

    # The planner agent handles design/decomposition (defaults to orchestrator's choice)
    planner_agent: str = "claude"

    # Worker agents handle implementation (list allows round-robin or selection)
    worker_agents: list[str] = field(default_factory=lambda: ["claude", "codex"])

    # The judge agent reviews plans and final output (should differ from planner)
    judge_agent: str = "claude"

    # Whether the plan approval gate blocks on rejection (vs. warning-only)
    plan_gate_blocking: bool = True

    # Whether the final judge review blocks on rejection
    final_review_blocking: bool = True

    # Maximum plan revision attempts before escalating
    max_plan_revisions: int = 2


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
    result: dict[str, Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class OrchestrationResult:
    """Result of an orchestration run."""

    goal: str
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    skipped_subtasks: int
    assignments: list[AgentAssignment]
    duration_seconds: float
    success: bool
    error: str | None = None
    summary: str = ""


class AgentRouter:
    """
    Routes subtasks to appropriate agents based on domain and track.

    Uses heuristics to determine:
    1. Which track owns a subtask (based on file patterns)
    2. Which agent type is best suited (based on task complexity)
    """

    def __init__(self, track_configs: dict[Track, TrackConfig] | None = None):
        self.track_configs = track_configs or DEFAULT_TRACK_CONFIGS
        self._file_to_track_cache: dict[str, Track] = {}

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
            Track.SECURITY: [
                "security",
                "vuln",
                "auth",
                "encrypt",
                "secret",
                "owasp",
                "xss",
                "csrf",
                "injection",
            ],
        }

        for track, keywords in patterns.items():
            if any(kw in combined for kw in keywords):
                return track

        # Default to developer track for unclassified tasks
        return Track.DEVELOPER

    def _file_to_track(self, file_path: str) -> Track | None:
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

    def get_coding_harness(
        self,
        agent_type: str,
        track: Track,
    ) -> dict[str, str] | None:
        """Determine the coding harness to use for an agent.

        For agents with native coding harnesses (claude, codex), returns None.
        For other agents (gemini, grok, etc.), returns KiloCode configuration
        if the track allows it.

        Args:
            agent_type: The selected agent type
            track: The development track

        Returns:
            None if agent has native harness, otherwise dict with:
            - harness: "kilocode"
            - provider_id: The KiloCode provider to use
            - mode: The KiloCode mode (code, architect, etc.)
        """
        # Agents with native coding harnesses don't need KiloCode
        if agent_type in AGENTS_WITH_CODING_HARNESS:
            return None

        # Check if track allows KiloCode harness
        config = self.track_configs.get(track, DEFAULT_TRACK_CONFIGS[Track.DEVELOPER])
        if not config.use_kilocode_harness:
            return None

        # Get KiloCode provider for this agent type
        provider_id = KILOCODE_PROVIDER_MAPPING.get(agent_type)
        if not provider_id:
            # No KiloCode mapping for this agent
            return None

        return {
            "harness": "kilocode",
            "provider_id": provider_id,
            "mode": "code",  # Use code mode for implementation tasks
        }

    def check_conflicts(
        self,
        subtask: SubTask,
        active_assignments: list[AgentAssignment],
    ) -> list[str]:
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
        self._iteration_counts: dict[str, int] = {}

    def analyze_failure(
        self,
        assignment: AgentAssignment,
        error_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze a failure and determine next steps.

        Anti-fragile design: on first failure, tries reassigning to a different
        agent before retrying the same one. This handles cases where an agent
        type is fundamentally incompatible with a task (timeout, rate limit,
        capability mismatch).
        """
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

        # Agent-level failures -> try a different agent before retrying same one
        if error_type in ("agent_timeout", "agent_error", "workflow_failure"):
            if assignment.attempt_count == 0:
                # First failure: try reassigning to a different agent
                return {
                    "action": "reassign_agent",
                    "reason": f"Agent {assignment.agent_type} failed on first attempt; "
                    f"trying alternative agent",
                    "original_agent": assignment.agent_type,
                }

        # CI failures -> adjust implementation
        if error_type == "ci_failure":
            return {
                "action": "retry_implement",
                "reason": "CI test failures require implementation adjustment",
                "hints": error_info.get("ci_failures", []),
            }

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
        aragora_path: Path | None = None,
        track_configs: dict[Track, TrackConfig] | None = None,
        workflow_engine: WorkflowEngine | None = None,
        task_decomposer: TaskDecomposer | None = None,
        require_human_approval: bool = False,
        max_parallel_tasks: int = 4,
        on_checkpoint: Callable[[str, dict[str, Any]], None] | None = None,
        use_debate_decomposition: bool = False,
        enable_curriculum: bool = True,
        curriculum_config: Any | None = None,
        branch_coordinator: Any | None = None,
        hierarchy: HierarchyConfig | None = None,
        hierarchical_coordinator: Any | None = None,
        enable_gauntlet_gate: bool = False,
        use_decision_plan: bool = False,
        enable_convoy_tracking: bool = False,
        workspace_manager: Any | None = None,
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
            use_debate_decomposition: Use multi-agent debate for goal decomposition
                (slower but better for abstract goals)
            enable_curriculum: Enable SOAR curriculum for failed tasks
            curriculum_config: Optional curriculum configuration
            branch_coordinator: Optional BranchCoordinator for worktree isolation
            hierarchy: Optional Planner/Worker/Judge hierarchy configuration
            hierarchical_coordinator: Optional HierarchicalCoordinator for
                plan-execute-judge cycle delegation
            enable_gauntlet_gate: Insert adversarial gauntlet step between
                design and implement phases
            use_decision_plan: Use DecisionPlanFactory to generate risk-aware
                workflows from debate results (falls back to standard workflow)
            enable_convoy_tracking: Track orchestration lifecycle with
                Convoy/Bead persistence for crash recovery
            workspace_manager: Optional WorkspaceManager for convoy/bead tracking
        """
        self.aragora_path = aragora_path or Path.cwd()
        self.track_configs = track_configs or DEFAULT_TRACK_CONFIGS
        self.branch_coordinator = branch_coordinator
        self.workflow_engine = workflow_engine or get_workflow_engine()
        self.task_decomposer = task_decomposer or TaskDecomposer(
            config=DecomposerConfig(complexity_threshold=4)
        )
        self.require_human_approval = require_human_approval
        self.max_parallel_tasks = max_parallel_tasks
        self.on_checkpoint = on_checkpoint
        self.use_debate_decomposition = use_debate_decomposition
        self.enable_gauntlet_gate = enable_gauntlet_gate
        self.use_decision_plan = use_decision_plan
        self.enable_convoy_tracking = enable_convoy_tracking
        self.workspace_manager = workspace_manager

        self.hierarchy = hierarchy or HierarchyConfig()
        self.hierarchical_coordinator = hierarchical_coordinator
        self.router = AgentRouter(self.track_configs)
        self.feedback_loop = FeedbackLoop()
        if enable_curriculum:
            try:
                from aragora.nomic.curriculum.integration import CurriculumAwareFeedbackLoop

                self.feedback_loop = CurriculumAwareFeedbackLoop(  # type: ignore[assignment]
                    max_iterations=self.feedback_loop.max_iterations,
                    config=curriculum_config,
                )
                logger.info("SOAR curriculum enabled for autonomous orchestrator")
            except Exception as e:
                logger.debug("SOAR curriculum unavailable: %s" % e)

        # Concurrency semaphore for parallel task execution
        self._semaphore = asyncio.Semaphore(max_parallel_tasks)

        # File-based approval gate
        self._auto_approve = not require_human_approval
        self._approval_gate_dir = self.aragora_path / ".aragora_beads" / "approval_gates"

        # State
        self._active_assignments: list[AgentAssignment] = []
        self._completed_assignments: list[AgentAssignment] = []
        self._orchestration_id: str | None = None
        # Convoy/bead IDs for tracking (populated when convoy tracking enabled)
        self._convoy_id: str | None = None
        self._bead_ids: dict[str, str] = {}  # subtask_id -> bead_id

    async def execute_goal(
        self,
        goal: str,
        tracks: list[str] | None = None,
        max_cycles: int = 5,
        context: dict[str, Any] | None = None,
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

        # Delegate to HierarchicalCoordinator if provided
        if self.hierarchical_coordinator is not None:
            h_result = await self.hierarchical_coordinator.coordinate(
                goal=goal,
                tracks=tracks,
                context=context,
            )
            return self._hierarchical_to_orchestration_result(h_result, goal, start_time)

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

            # Step 2b: Create convoy for tracking if enabled
            if self.enable_convoy_tracking:
                await self._create_convoy_for_goal(goal, assignments)

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

            # Step 4b: Complete convoy tracking
            if self.enable_convoy_tracking and self._convoy_id:
                await self._complete_convoy(failed == 0)

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

            # Fail the convoy on exception
            if self.enable_convoy_tracking and self._convoy_id:
                try:
                    await self._complete_convoy(success=False, error=str(e))
                except Exception:
                    logger.debug("Failed to update convoy on error")

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
        tracks: list[str] | None = None,
    ) -> TaskDecomposition:
        """Decompose a high-level goal into subtasks."""
        # Enrich goal with track context if provided
        if tracks:
            track_context = f"\n\nFocus tracks: {', '.join(tracks)}"
            enriched_goal = f"{goal}{track_context}"
        else:
            enriched_goal = goal

        # Use debate-based decomposition for abstract goals
        if self.use_debate_decomposition:
            logger.info("Using debate-based decomposition for goal")
            return await self.task_decomposer.analyze_with_debate(enriched_goal)

        return self.task_decomposer.analyze(enriched_goal)

    def _create_assignments(
        self,
        decomposition: TaskDecomposition,
        tracks: list[str] | None = None,
    ) -> list[AgentAssignment]:
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
        assignments: list[AgentAssignment],
        max_cycles: int,
    ) -> None:
        """Execute assignments with parallel coordination.

        When a BranchCoordinator is configured, creates isolated worktree
        branches before execution and merges completed branches afterward.
        """
        # Create worktree branches for isolation if coordinator is available
        if self.branch_coordinator is not None:
            await self._create_branches_for_assignments(assignments)

        pending = list(assignments)
        running: list[asyncio.Task] = []

        while pending or running:
            # Start new tasks up to max parallel (semaphore enforces limit)
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

                task = asyncio.create_task(self._execute_with_semaphore(assignment, max_cycles))
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

        # Merge completed branches and cleanup worktrees
        if self.branch_coordinator is not None:
            await self._merge_and_cleanup(assignments)

    async def _execute_with_semaphore(
        self,
        assignment: AgentAssignment,
        max_cycles: int,
    ) -> None:
        """Acquire the concurrency semaphore then execute the assignment."""
        async with self._semaphore:
            await self._execute_single_assignment(assignment, max_cycles)

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

        # Update bead status to RUNNING
        await self._update_bead_status(subtask.id, "running")

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
                await self._update_bead_status(subtask.id, "done")
            else:
                # Handle failure with feedback loop
                feedback = self.feedback_loop.analyze_failure(
                    assignment,
                    {"type": "workflow_failure", "message": result.error or ""},
                )
                if inspect.isawaitable(feedback):
                    feedback = await feedback

                if feedback["action"] == "escalate":
                    assignment.status = "failed"
                    assignment.result = {"error": result.error, "feedback": feedback}
                    await self._update_bead_status(subtask.id, "failed", error=result.error)
                elif feedback["action"] == "reassign_agent":
                    # Anti-fragile: try a different agent type
                    alt_agent = self._select_alternative_agent(assignment)
                    if alt_agent and alt_agent != assignment.agent_type:
                        logger.info(
                            "reassigning_agent",
                            subtask_id=subtask.id,
                            from_agent=assignment.agent_type,
                            to_agent=alt_agent,
                        )
                        assignment.agent_type = alt_agent
                    assignment.attempt_count += 1
                    if assignment.attempt_count < assignment.max_attempts:
                        await self._execute_single_assignment(assignment, max_cycles)
                    else:
                        assignment.status = "failed"
                        await self._update_bead_status(subtask.id, "failed")
                else:
                    # Retry based on feedback
                    assignment.attempt_count += 1
                    if assignment.attempt_count < assignment.max_attempts:
                        await self._execute_single_assignment(assignment, max_cycles)
                    else:
                        assignment.status = "failed"
                        await self._update_bead_status(subtask.id, "failed")

        except Exception as e:
            logger.exception(
                "assignment_failed",
                subtask_id=subtask.id,
                error=str(e),
            )
            assignment.status = "failed"
            assignment.result = {"error": str(e)}
            await self._update_bead_status(subtask.id, "failed", error=str(e))

        finally:
            assignment.completed_at = datetime.now(timezone.utc)
            self._completed_assignments.append(assignment)
            self._active_assignments.remove(assignment)

    def _select_alternative_agent(self, assignment: AgentAssignment) -> str | None:
        """Select an alternative agent type for reassignment on failure.

        Picks the next available agent from the track's preferred list,
        skipping the current agent. Falls back to 'claude' as the most
        capable general-purpose agent.
        """
        config = self.track_configs.get(
            assignment.track,
            DEFAULT_TRACK_CONFIGS[Track.DEVELOPER],
        )
        candidates = [a for a in config.agent_types if a != assignment.agent_type]
        if candidates:
            return candidates[0]
        # Fallback: if current agent isn't claude, try claude
        if assignment.agent_type != "claude":
            return "claude"
        return None

    def _build_subtask_workflow(self, assignment: AgentAssignment) -> WorkflowDefinition:
        """Build a workflow definition for a subtask.

        Uses the gold path: agent(design) -> implementation -> verification.

        The "implementation" step type bridges to HybridExecutor which spawns
        Claude/Codex subprocesses to write code. The "verification" step type
        runs pytest against the changed files.
        """
        subtask = assignment.subtask

        # Check if agent needs a coding harness (e.g., KiloCode for Gemini)
        coding_harness = self.router.get_coding_harness(
            assignment.agent_type,
            assignment.track,
        )

        # Resolve repo path: prefer worktree path for isolation
        repo_path = self.aragora_path
        if self.branch_coordinator is not None:
            # Look up if this assignment's track has a worktree
            for branch, wt_path in getattr(self.branch_coordinator, "_worktree_paths", {}).items():
                if assignment.track.value in branch:
                    repo_path = wt_path
                    break

        # Build implementation step config matching ImplementationStep's expected format
        implement_config: dict[str, Any] = {
            "task_id": subtask.id,
            "description": subtask.description,
            "files": subtask.file_scope,
            "complexity": subtask.estimated_complexity,
            "repo_path": str(repo_path),
        }

        # If agent needs a coding harness, add it to the config
        if coding_harness:
            implement_config["coding_harness"] = coding_harness
            logger.info(
                f"subtask_using_kilocode agent={assignment.agent_type} "
                f"provider={coding_harness['provider_id']} "
                f"track={assignment.track.value}"
            )

        # Derive test paths from file scope for verification
        test_paths = self._infer_test_paths(subtask.file_scope)

        # Create workflow with phases aligned to nomic loop gold path
        #
        # With hierarchy enabled, the workflow becomes:
        #   design (planner) -> plan_approval (judge) -> [gauntlet] -> implement (worker)
        #   -> verify -> judge_review (judge)
        #
        # With gauntlet gate enabled (no hierarchy):
        #   design -> gauntlet -> implement -> verify
        #
        # Without either:
        #   design -> implement -> verify

        hierarchy_enabled = self.hierarchy.enabled

        # Determine the step that follows design (before implement)
        # Priority: plan_approval (hierarchy) > gauntlet > implement
        if hierarchy_enabled:
            design_next = ["plan_approval"]
        elif self.enable_gauntlet_gate:
            design_next = ["gauntlet"]
        else:
            design_next = ["implement"]

        # The step that feeds into implement
        if self.enable_gauntlet_gate and hierarchy_enabled:
            plan_approval_next = ["gauntlet"]
        elif hierarchy_enabled:
            plan_approval_next = ["implement"]
        else:
            plan_approval_next = ["implement"]

        # Override agent types when hierarchy is active
        design_agent = self.hierarchy.planner_agent if hierarchy_enabled else assignment.agent_type
        implement_agent = assignment.agent_type
        if hierarchy_enabled and self.hierarchy.worker_agents:
            # Pick the first worker agent that matches the track, or fallback to first
            implement_agent = self.hierarchy.worker_agents[0]
            for wa in self.hierarchy.worker_agents:
                if wa in (
                    config.agent_types
                    if (config := self.track_configs.get(assignment.track))
                    else []
                ):
                    implement_agent = wa
                    break

        steps = [
            StepDefinition(
                id="design",
                name="Design Solution",
                step_type="agent",
                config={
                    "agent_type": design_agent,
                    "prompt_template": "design",
                    "task": subtask.description,
                },
                next_steps=design_next,
            ),
        ]

        if hierarchy_enabled:
            steps.append(
                StepDefinition(
                    id="plan_approval",
                    name="Plan Approval Gate",
                    step_type="agent",
                    config={
                        "agent_type": self.hierarchy.judge_agent,
                        "prompt_template": "review",
                        "task": (
                            f"Review the design plan for: {subtask.description}\n\n"
                            "Evaluate the plan for:\n"
                            "1. Feasibility: Can this be implemented as described?\n"
                            "2. Completeness: Are all edge cases addressed?\n"
                            "3. Risk: Are there security or correctness risks?\n\n"
                            "Respond with APPROVE or REJECT (with reasons)."
                        ),
                        "gate": True,
                        "blocking": self.hierarchy.plan_gate_blocking,
                        "max_revisions": self.hierarchy.max_plan_revisions,
                    },
                    next_steps=plan_approval_next,
                )
            )

        # Insert gauntlet adversarial validation step between design and implement
        if self.enable_gauntlet_gate:
            # Use stricter threshold for high-complexity subtasks
            severity_threshold = "medium" if subtask.estimated_complexity == "high" else "high"
            steps.append(
                StepDefinition(
                    id="gauntlet",
                    name="Adversarial Validation",
                    step_type="gauntlet",
                    config={
                        "input_key": "content",
                        "severity_threshold": severity_threshold,
                        "require_passing": True,
                        "attack_categories": [
                            "prompt_injection",
                            "hallucination",
                            "safety",
                        ],
                        "probe_categories": [
                            "reasoning",
                            "consistency",
                        ],
                    },
                    next_steps=["implement"],
                )
            )

        steps.append(
            StepDefinition(
                id="implement",
                name="Implement Changes",
                step_type="implementation",
                config={
                    **implement_config,
                    "agent_type": implement_agent,
                },
                next_steps=["verify"],
            ),
        )

        verify_next = ["judge_review"] if hierarchy_enabled else []
        steps.append(
            StepDefinition(
                id="verify",
                name="Verify Changes",
                step_type="verification",
                config={
                    "run_tests": True,
                    "test_paths": test_paths,
                    "test_count": len(test_paths),
                },
                next_steps=verify_next,
            ),
        )

        if hierarchy_enabled:
            steps.append(
                StepDefinition(
                    id="judge_review",
                    name="Judge Final Review",
                    step_type="agent",
                    config={
                        "agent_type": self.hierarchy.judge_agent,
                        "prompt_template": "review",
                        "task": (
                            f"Final review for: {subtask.description}\n\n"
                            "Review the implementation and verification results.\n"
                            "Check that the implementation matches the approved plan.\n"
                            "Respond with APPROVE or REJECT (with reasons)."
                        ),
                        "gate": True,
                        "blocking": self.hierarchy.final_review_blocking,
                    },
                    next_steps=[],
                )
            )

        return WorkflowDefinition(
            id=f"subtask_{subtask.id}",
            name=f"Execute: {subtask.title}",
            description=subtask.description,
            steps=steps,
            entry_step="design",
        )

    @staticmethod
    def _infer_test_paths(file_scope: list[str]) -> list[str]:
        """Infer test file paths from source file paths.

        Maps source files like ``aragora/foo/bar.py`` to
        ``tests/foo/test_bar.py`` if no explicit test paths are provided.
        """
        test_paths: list[str] = []
        for path in file_scope:
            if path.startswith("tests/"):
                test_paths.append(path)
                continue
            # aragora/foo/bar.py -> tests/foo/test_bar.py
            if path.startswith("aragora/"):
                rel = path[len("aragora/") :]
                parts = rel.rsplit("/", 1)
                if len(parts) == 2:
                    directory, filename = parts
                    if filename.endswith(".py"):
                        test_file = f"tests/{directory}/test_{filename}"
                        test_paths.append(test_file)
        return test_paths

    def _checkpoint(self, phase: str, data: dict[str, Any]) -> None:
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

    def _generate_summary(self, assignments: list[AgentAssignment]) -> str:
        """Generate a summary of the orchestration."""
        by_track: dict[str, list[str]] = {}
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
    # File-based approval gate
    # =========================================================================

    async def request_approval(
        self,
        gate_id: str,
        description: str,
        metadata: dict[str, Any] | None = None,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> bool:
        """Request human approval via a file-based gate.

        Writes a JSON request to ``.aragora_beads/approval_gates/<gate_id>.json``.
        Polls for a ``.approved`` or ``.rejected`` marker file.

        If ``--auto-approve`` / ``self._auto_approve`` is True, returns
        immediately without waiting.

        Args:
            gate_id: Unique identifier for this approval gate
            description: Human-readable description of what's being approved
            metadata: Additional context for the reviewer
            poll_interval: Seconds between polls
            timeout: Maximum seconds to wait

        Returns:
            True if approved, False if rejected or timed out
        """
        if self._auto_approve:
            logger.info("approval_auto_approved", gate=gate_id)
            return True

        import json as _json
        import time as _time

        gate_dir = self._approval_gate_dir
        gate_dir.mkdir(parents=True, exist_ok=True)

        request_file = gate_dir / f"{gate_id}.json"
        approved_file = gate_dir / f"{gate_id}.approved"
        rejected_file = gate_dir / f"{gate_id}.rejected"

        # Write request
        request_data = {
            "gate_id": gate_id,
            "description": description,
            "metadata": metadata or {},
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "orchestration_id": self._orchestration_id,
        }
        request_file.write_text(_json.dumps(request_data, indent=2))

        logger.info("approval_requested", gate=gate_id, file=str(request_file))

        # Poll for approval/rejection
        start = _time.monotonic()
        while (_time.monotonic() - start) < timeout:
            if approved_file.exists():
                logger.info("approval_granted", gate=gate_id)
                # Clean up marker
                approved_file.unlink(missing_ok=True)
                request_file.unlink(missing_ok=True)
                return True

            if rejected_file.exists():
                logger.warning("approval_rejected", gate=gate_id)
                rejected_file.unlink(missing_ok=True)
                request_file.unlink(missing_ok=True)
                return False

            await asyncio.sleep(poll_interval)

        logger.warning("approval_timeout", gate=gate_id, seconds=timeout)
        request_file.unlink(missing_ok=True)
        return False

    # =========================================================================
    # Branch coordination helpers
    # =========================================================================

    async def _create_branches_for_assignments(
        self,
        assignments: list[AgentAssignment],
    ) -> None:
        """Create worktree branches for each unique track in the assignments.

        Groups assignments by track and creates one branch per track so
        multiple assignments targeting the same track share a worktree.
        """
        if self.branch_coordinator is None:
            return

        from aragora.nomic.meta_planner import PrioritizedGoal
        from aragora.nomic.meta_planner import Track as MetaTrack

        seen_tracks: set[str] = set()
        for assignment in assignments:
            track_value = assignment.track.value
            if track_value in seen_tracks:
                continue
            seen_tracks.add(track_value)

            # Map orchestrator Track to meta_planner Track
            try:
                meta_track = MetaTrack(track_value)
            except ValueError:
                meta_track = MetaTrack.DEVELOPER

            await self.branch_coordinator.create_track_branch(
                track=meta_track,
                goal=assignment.subtask.description[:60],
            )
            logger.info(
                "worktree_created",
                track=track_value,
                subtask_id=assignment.subtask.id,
            )

    async def _merge_and_cleanup(
        self,
        assignments: list[AgentAssignment],
    ) -> None:
        """Merge completed branches back to base and cleanup worktrees."""
        if self.branch_coordinator is None:
            return

        completed_tracks: set[str] = set()
        failed_tracks: set[str] = set()
        for assignment in assignments:
            if assignment.status == "completed":
                completed_tracks.add(assignment.track.value)
            elif assignment.status == "failed":
                failed_tracks.add(assignment.track.value)

        # Merge branches where all assignments in the track completed
        for branch, wt_path in dict(
            getattr(self.branch_coordinator, "_worktree_paths", {})
        ).items():
            # Check if any track in this branch completed and none failed
            track_value = None
            for t in completed_tracks:
                if t in branch:
                    track_value = t
                    break

            if track_value and track_value not in failed_tracks:
                merge_result = await self.branch_coordinator.safe_merge(branch)
                if merge_result.success:
                    logger.info(
                        "branch_merged",
                        branch=branch,
                        commit_sha=merge_result.commit_sha,
                    )
                else:
                    logger.warning(
                        "branch_merge_failed",
                        branch=branch,
                        error=merge_result.error,
                    )

        # Cleanup all worktrees
        if hasattr(self.branch_coordinator, "cleanup_all_worktrees"):
            removed = self.branch_coordinator.cleanup_all_worktrees()
        elif hasattr(self.branch_coordinator, "cleanup_worktrees"):
            removed = self.branch_coordinator.cleanup_worktrees()
        else:
            removed = 0
        if removed:
            logger.info("worktrees_cleaned", count=removed)

    # =========================================================================
    # DecisionPlan integration
    # =========================================================================

    def _build_workflow_from_plan(
        self,
        assignment: AgentAssignment,
        debate_result: Any,
    ) -> WorkflowDefinition | None:
        """Build a risk-aware workflow using DecisionPlanFactory.

        When ``use_decision_plan`` is enabled and a debate result is available,
        creates a DecisionPlan which includes risk assessment, verification
        plan, and approval routing based on risk level.

        Returns None if the factory is unavailable or the debate result is
        missing, in which case the caller should fall back to
        ``_build_subtask_workflow``.
        """
        if not self.use_decision_plan or debate_result is None:
            return None

        try:
            from aragora.pipeline.decision_plan.factory import DecisionPlanFactory
            from aragora.pipeline.decision_plan.core import ApprovalMode

            plan = DecisionPlanFactory.from_debate_result(
                debate_result,
                approval_mode=ApprovalMode.RISK_BASED,
                repo_path=self.aragora_path,
            )

            # Convert the plan's implement tasks to workflow steps
            steps: list[StepDefinition] = []
            if plan.implement_plan and plan.implement_plan.tasks:
                for i, task in enumerate(plan.implement_plan.tasks):
                    step_id = f"plan_task_{task.id}"
                    next_id = (
                        f"plan_task_{plan.implement_plan.tasks[i + 1].id}"
                        if i + 1 < len(plan.implement_plan.tasks)
                        else "verify"
                    )
                    steps.append(
                        StepDefinition(
                            id=step_id,
                            name=f"Implement: {task.description[:50]}",
                            step_type="implementation",
                            config={
                                "task_id": task.id,
                                "description": task.description,
                                "files": task.files,
                                "complexity": task.complexity,
                                "repo_path": str(self.aragora_path),
                                "agent_type": assignment.agent_type,
                            },
                            next_steps=[next_id],
                        )
                    )

            # Add verification step
            test_paths = self._infer_test_paths(assignment.subtask.file_scope)
            steps.append(
                StepDefinition(
                    id="verify",
                    name="Verify Changes",
                    step_type="verification",
                    config={
                        "run_tests": True,
                        "test_paths": test_paths,
                    },
                    next_steps=[],
                ),
            )

            # Add approval gate if plan requires human approval
            if plan.requires_human_approval:
                # Insert approval step before implementation
                approval_step = StepDefinition(
                    id="risk_approval",
                    name="Risk-Based Approval Gate",
                    step_type="agent",
                    config={
                        "agent_type": "claude",
                        "prompt_template": "review",
                        "task": (
                            f"Risk review for: {assignment.subtask.description}\n"
                            f"Risk level: {plan.risk_register.max_risk_level if plan.risk_register else 'unknown'}\n"
                            "Review and approve/reject."
                        ),
                        "gate": True,
                        "blocking": True,
                    },
                    next_steps=[steps[0].id] if steps else [],
                )
                steps.insert(0, approval_step)

            entry = steps[0].id if steps else "verify"
            return WorkflowDefinition(
                id=f"plan_{assignment.subtask.id}",
                name=f"DecisionPlan: {assignment.subtask.title}",
                description=assignment.subtask.description,
                steps=steps,
                entry_step=entry,
            )

        except ImportError:
            logger.debug("DecisionPlanFactory not available, using standard workflow")
            return None
        except Exception as e:
            logger.warning(f"Failed to build DecisionPlan workflow: {e}")
            return None

    # =========================================================================
    # Convoy/Bead tracking helpers
    # =========================================================================

    async def _create_convoy_for_goal(
        self,
        goal: str,
        assignments: list[AgentAssignment],
    ) -> None:
        """Create a convoy and beads for tracking the orchestration lifecycle."""
        if not self.enable_convoy_tracking or self.workspace_manager is None:
            return

        try:
            # Create a rig for this orchestration
            rig = await self.workspace_manager.create_rig(
                name=f"orch-{self._orchestration_id}",
            )

            # Create bead specs from assignments
            bead_specs = [
                {
                    "title": a.subtask.title,
                    "description": a.subtask.description,
                    "payload": {
                        "subtask_id": a.subtask.id,
                        "track": a.track.value,
                        "agent_type": a.agent_type,
                    },
                }
                for a in assignments
            ]

            convoy = await self.workspace_manager.create_convoy(
                rig_id=rig.rig_id,
                name=f"Goal: {goal[:50]}",
                description=goal,
                bead_specs=bead_specs,
            )

            self._convoy_id = convoy.convoy_id
            await self.workspace_manager.start_convoy(convoy.convoy_id)

            # Map subtask IDs to bead IDs for status updates
            beads = await self.workspace_manager._bead_manager.list_beads(
                convoy_id=convoy.convoy_id,
            )
            for bead in beads:
                subtask_id = bead.payload.get("subtask_id", "")
                if subtask_id:
                    self._bead_ids[subtask_id] = bead.bead_id

            logger.info(
                "convoy_created",
                convoy_id=convoy.convoy_id,
                bead_count=len(beads),
            )

        except Exception as e:
            logger.warning("Failed to create convoy: %s", e)

    async def _update_bead_status(
        self,
        subtask_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update bead status for a subtask."""
        if not self.enable_convoy_tracking or self.workspace_manager is None:
            return

        bead_id = self._bead_ids.get(subtask_id)
        if not bead_id:
            return

        try:
            if status == "running":
                await self.workspace_manager._bead_manager.start_bead(bead_id)
            elif status == "done":
                await self.workspace_manager.complete_bead(bead_id)
            elif status == "failed":
                await self.workspace_manager.fail_bead(bead_id, error or "Unknown error")
        except Exception as e:
            logger.debug("Failed to update bead %s: %s", bead_id, e)

    async def _complete_convoy(
        self,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Mark the convoy as completed or failed."""
        if not self.enable_convoy_tracking or self.workspace_manager is None:
            return
        if not self._convoy_id:
            return

        try:
            if success:
                await self.workspace_manager.complete_convoy(self._convoy_id)
            else:
                tracker = self.workspace_manager._convoy_tracker
                await tracker.fail_convoy(self._convoy_id, error or "Orchestration failed")
        except Exception as e:
            logger.debug("Failed to complete convoy: %s", e)

    def _hierarchical_to_orchestration_result(
        self,
        h_result: Any,
        goal: str,
        start_time: datetime,
    ) -> OrchestrationResult:
        """Convert HierarchicalResult to OrchestrationResult for backward compat."""
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Convert worker reports to AgentAssignments
        assignments: list[AgentAssignment] = []
        for report in h_result.worker_reports:
            subtask = SubTask(
                id=report.assignment_id,
                title=report.subtask_title,
                description="",
            )
            assignment = AgentAssignment(
                subtask=subtask,
                track=Track.DEVELOPER,
                agent_type=self.config.judge_agent
                if hasattr(self, "config") and hasattr(self.config, "judge_agent")
                else "claude",
                status="completed" if report.success else "failed",
                result=report.output,
            )
            assignments.append(assignment)

        completed = sum(1 for r in h_result.worker_reports if r.success)
        failed = sum(1 for r in h_result.worker_reports if not r.success)

        return OrchestrationResult(
            goal=goal,
            total_subtasks=len(h_result.worker_reports),
            completed_subtasks=completed,
            failed_subtasks=failed,
            skipped_subtasks=0,
            assignments=assignments,
            duration_seconds=duration,
            success=h_result.success,
            summary=f"Hierarchical coordination: {completed}/{len(h_result.worker_reports)} tasks completed "
            f"in {h_result.cycles_used} cycles",
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def execute_track(
        self,
        track: str,
        focus_areas: list[str] | None = None,
        max_cycles: int = 3,
    ) -> OrchestrationResult:
        """
        Execute work for a specific track.

        Args:
            track: Track name (sme, developer, self_hosted, qa, core, security)
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

    def get_active_assignments(self) -> list[AgentAssignment]:
        """Get currently active assignments."""
        return self._active_assignments.copy()

    def get_completed_assignments(self) -> list[AgentAssignment]:
        """Get completed assignments."""
        return self._completed_assignments.copy()


# Singleton instance
_orchestrator_instance: AutonomousOrchestrator | None = None


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
    "HierarchyConfig",
    "Track",
    "TrackConfig",
    "AgentAssignment",
    "OrchestrationResult",
    "get_orchestrator",
    "reset_orchestrator",
]
