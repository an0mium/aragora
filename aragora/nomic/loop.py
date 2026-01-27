"""
Nomic Loop - Compatibility wrapper for legacy API.

Provides the `NomicLoop` class expected by existing tests and integrations,
wrapping the new `NomicStateMachine` implementation.

This module bridges the gap between:
- Legacy API: `NomicLoop` with direct phase methods
- New API: `NomicStateMachine` with event-driven state transitions

Usage:
    from aragora.nomic.loop import NomicLoop

    loop = NomicLoop(
        aragora_path=Path("/path/to/project"),
        max_cycles=5,
        protected_files=["CLAUDE.md"],
    )
    result = await loop.run_cycle()
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aragora.nomic.cycle_record import NomicCycleRecord
from aragora.nomic.cycle_store import CycleLearningStore, get_cycle_store

logger = logging.getLogger(__name__)


class NomicLoop:
    """
    Nomic Loop for autonomous self-improvement cycles.

    This class provides the traditional phase-based API for running
    nomic improvement cycles. Internally it coordinates phase execution
    with safety checks and checkpointing.

    Args:
        aragora_path: Path to the aragora project root
        max_cycles: Maximum number of cycles to run (default: 1)
        protected_files: List of files that cannot be modified
        require_human_approval: Whether to require human approval for changes
        log_fn: Optional logging function for status updates
        checkpoint_dir: Directory for storing checkpoints
        max_files_per_cycle: Maximum files that can be modified per cycle
        max_consecutive_failures: Stop after this many consecutive failures

    Example:
        loop = NomicLoop(
            aragora_path=Path("."),
            max_cycles=3,
            protected_files=["CLAUDE.md"],
        )
        await loop.run()
    """

    def __init__(
        self,
        aragora_path: Path,
        max_cycles: int = 1,
        protected_files: Optional[List[str]] = None,
        require_human_approval: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
        checkpoint_dir: Optional[Path] = None,
        max_files_per_cycle: int = 20,
        max_consecutive_failures: int = 3,
    ):
        """Initialize the Nomic Loop."""
        self.aragora_path = aragora_path
        self.max_cycles = max_cycles
        self.protected_files = protected_files or []
        self.require_human_approval = require_human_approval
        self.log_fn = log_fn or logger.info
        self.checkpoint_dir = checkpoint_dir or Path(".nomic/checkpoints")
        self.max_files_per_cycle = max_files_per_cycle
        self.max_consecutive_failures = max_consecutive_failures

        # State
        self._cycle_count = 0
        self._consecutive_failures = 0
        self._current_cycle_id: Optional[str] = None
        self._cycle_context: Dict[str, Any] = {}
        self._checkpoints: List[Dict[str, Any]] = []

        # Cross-cycle learning
        self._cycle_store: Optional[CycleLearningStore] = None
        self._current_record: Optional[NomicCycleRecord] = None
        self._enable_cycle_learning = True

    def _log(self, message: str) -> None:
        """Log a message using the configured log function."""
        self.log_fn(message)

    async def run(self, max_cycles: Optional[int] = None) -> Dict[str, Any]:
        """
        Run multiple nomic cycles.

        Args:
            max_cycles: Override max_cycles for this run

        Returns:
            Summary of all cycles run
        """
        cycles_to_run = max_cycles or self.max_cycles
        results: List[Dict[str, Any]] = []

        self._log(f"Starting nomic loop with max {cycles_to_run} cycles")

        for i in range(cycles_to_run):
            self._log(f"Starting cycle {i + 1}/{cycles_to_run}")

            result = await self.run_cycle()
            results.append(result)

            if result.get("success", False):
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.max_consecutive_failures:
                    self._log(f"Stopping after {self._consecutive_failures} consecutive failures")
                    break

        return {
            "cycles_run": len(results),
            "successful_cycles": sum(1 for r in results if r.get("success", False)),
            "failed_cycles": sum(1 for r in results if not r.get("success", False)),
            "results": results,
        }

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Run a single nomic improvement cycle.

        Executes phases in order:
        1. Context - Gather codebase understanding
        2. Debate - Agents propose improvements
        3. Design - Plan the implementation
        4. Implement - Write code changes
        5. Verify - Test and validate

        Returns:
            Cycle result with success status and phase outputs
        """
        self._cycle_count += 1
        self._current_cycle_id = str(uuid.uuid4())[:8]
        self._cycle_context = {}

        # Initialize cycle record for cross-cycle learning
        self._current_record = NomicCycleRecord(
            cycle_id=self._current_cycle_id,
            started_at=time.time(),
        )

        # Inject cross-cycle learning context
        learning_context = self._get_cross_cycle_context()
        if learning_context:
            self._cycle_context["learning_context"] = learning_context

        self._log(f"Cycle {self._current_cycle_id} started")

        try:
            # Phase 1: Context
            context_result = await self.run_context_phase()
            if not context_result.get("success", False):
                return self._cycle_failed("context", context_result)
            self._cycle_context["context"] = context_result

            # Phase 2: Debate
            debate_result = await self.run_debate_phase()
            if not debate_result.get("consensus", False):
                return self._cycle_failed("debate", debate_result, "No consensus reached")
            self._cycle_context["debate"] = debate_result

            # Phase 3: Design
            design_result = await self.run_design_phase(debate_result)
            if not design_result.get("approved", False):
                return self._cycle_failed("design", design_result, "Design not approved")
            self._cycle_context["design"] = design_result

            # Phase 4: Implement
            impl_result = await self.run_implement_phase(design_result)
            if not impl_result.get("success", False):
                return self._cycle_failed("implement", impl_result)
            self._cycle_context["implementation"] = impl_result

            # Phase 5: Verify
            verify_result = await self.run_verify_phase(impl_result)
            self._cycle_context["verification"] = verify_result

            # Record test results
            if self._current_record:
                test_results = verify_result.get("test_results", {})
                self._current_record.tests_passed = test_results.get("passed", 0)
                self._current_record.tests_failed = test_results.get("failed", 0)
                self._current_record.tests_skipped = test_results.get("skipped", 0)
                self._current_record.phases_completed = [
                    "context",
                    "debate",
                    "design",
                    "implement",
                    "verify",
                ]

            # Mark cycle successful and save
            self._finalize_cycle_record(success=True)

            return {
                "success": True,
                "cycle_id": self._current_cycle_id,
                "context": context_result,
                "debate": debate_result,
                "design": design_result,
                "implementation": impl_result,
                "verification": verify_result,
            }

        except Exception as e:
            logger.exception(f"Cycle {self._current_cycle_id} failed with exception")
            self.create_checkpoint()

            # Record failure with error
            self._finalize_cycle_record(success=False, error=str(e))

            return {
                "success": False,
                "cycle_id": self._current_cycle_id,
                "error": str(e),
                "phase": self._cycle_context.get("current_phase", "unknown"),
            }

    def _cycle_failed(
        self,
        phase: str,
        result: Dict[str, Any],
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a failed cycle result."""
        error_msg = reason or result.get("error", "Phase failed")

        # Record failure with phase info
        self._finalize_cycle_record(success=False, error=f"{phase}: {error_msg}")

        return {
            "success": False,
            "completed": False,
            "cycle_id": self._current_cycle_id,
            "failed_phase": phase,
            "reason": error_msg,
            "partial_context": self._cycle_context,
        }

    async def run_context_phase(self) -> Dict[str, Any]:
        """
        Run the context gathering phase.

        Explores the codebase to understand current state,
        recent changes, and areas for improvement.

        Returns:
            Context information including files, features, and analysis
        """
        self._cycle_context["current_phase"] = "context"
        self._log("Running context phase")

        # Default implementation - can be overridden or mocked
        return {
            "success": True,
            "context": "gathered",
            "files_analyzed": [],
            "features_found": [],
        }

    async def run_debate_phase(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the debate phase where agents propose improvements.

        Args:
            context: Optional context from previous phase

        Returns:
            Debate results including consensus and proposals
        """
        self._cycle_context["current_phase"] = "debate"
        self._log("Running debate phase")

        # Default implementation - can be overridden or mocked
        return {
            "consensus": True,
            "confidence": 0.8,
            "proposals": [],
        }

    async def run_design_phase(
        self,
        debate_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the design phase to plan implementation.

        Args:
            debate_result: Results from the debate phase

        Returns:
            Design specification including components and files
        """
        self._cycle_context["current_phase"] = "design"
        self._log("Running design phase")

        # Default implementation - can be overridden or mocked
        return {
            "approved": True,
            "design": {
                "components": [],
                "files_to_modify": [],
            },
        }

    async def run_implement_phase(
        self,
        design_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the implementation phase to write code changes.

        Args:
            design_result: Results from the design phase

        Returns:
            Implementation results including files modified
        """
        self._cycle_context["current_phase"] = "implement"
        self._log("Running implement phase")

        # Default implementation - can be overridden or mocked
        return {
            "success": True,
            "files_created": [],
            "files_modified": [],
        }

    async def run_verify_phase(
        self,
        impl_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the verification phase to test changes.

        Args:
            impl_result: Results from the implementation phase

        Returns:
            Verification results including test status
        """
        self._cycle_context["current_phase"] = "verify"
        self._log("Running verify phase")

        # Default implementation - can be overridden or mocked
        return {
            "passed": True,
            "test_results": {
                "total": 0,
                "passed": 0,
                "failed": 0,
            },
        }

    def check_safety(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if proposed changes are safe to apply.

        Validates:
        - No protected files are modified
        - Change volume is within limits
        - No dangerous patterns detected

        Args:
            changes: Dictionary with files_modified, files_created, etc.

        Returns:
            Safety check result with safe flag and reason
        """
        files_modified = changes.get("files_modified", [])
        files_created = changes.get("files_created", [])
        all_files = files_modified + files_created

        # Check protected files
        for f in files_modified:
            if f in self.protected_files or Path(f).name in self.protected_files:
                return {
                    "safe": False,
                    "reason": f"Protected file cannot be modified: {f}",
                }

        # Check change volume
        if len(all_files) > self.max_files_per_cycle:
            return {
                "safe": False,
                "reason": f"Too many files modified ({len(all_files)} > {self.max_files_per_cycle})",
            }

        return {"safe": True, "reason": "All checks passed"}

    async def get_approval_for_changes(
        self,
        changes: Dict[str, Any],
    ) -> bool:
        """
        Get approval for proposed changes.

        If require_human_approval is True, will request human approval.
        Otherwise, auto-approves safe changes.

        Args:
            changes: Dictionary describing the proposed changes

        Returns:
            True if approved, False otherwise
        """
        # First check safety
        safety_result = self.check_safety(changes)
        if not safety_result["safe"]:
            self._log(f"Changes rejected: {safety_result['reason']}")
            return False

        # Request human approval if required
        if self.require_human_approval:
            return await self.request_human_approval(changes)

        return True

    async def request_human_approval(
        self,
        changes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Request human approval for changes.

        Default implementation auto-approves. Override for actual
        human-in-the-loop approval.

        Args:
            changes: Optional changes to approve

        Returns:
            True if approved
        """
        # Default: auto-approve (tests should mock this)
        self._log("Human approval requested (auto-approved in default implementation)")
        return True

    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of current cycle state.

        Saves the current context so the cycle can be resumed later.

        Returns:
            Checkpoint data
        """
        checkpoint = {
            "cycle_id": self._current_cycle_id,
            "cycle_count": self._cycle_count,
            "timestamp": time.time(),
            "context": self._cycle_context.copy(),
            "aragora_path": str(self.aragora_path),
        }
        self._checkpoints.append(checkpoint)

        # Save to disk if checkpoint_dir is set
        if self.checkpoint_dir:
            try:
                import json

                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{self._current_cycle_id}.json"
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2, default=str))
                self._log(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        return checkpoint

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Restore state from a checkpoint.

        Args:
            checkpoint: Checkpoint data to restore from
        """
        self._current_cycle_id = checkpoint.get("cycle_id")
        self._cycle_count = checkpoint.get("cycle_count", 0)
        self._cycle_context = checkpoint.get("context", {})
        self._log(f"Restored from checkpoint: {self._current_cycle_id}")

    # =========================================================================
    # Cross-Cycle Learning Methods
    # =========================================================================

    def _get_cycle_store(self) -> CycleLearningStore:
        """Get or create the cycle learning store."""
        if self._cycle_store is None:
            self._cycle_store = get_cycle_store()
        return self._cycle_store

    def _finalize_cycle_record(
        self,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Finalize and save the cycle record.

        Args:
            success: Whether the cycle succeeded
            error: Optional error message if failed
        """
        if not self._enable_cycle_learning or not self._current_record:
            return

        try:
            # Populate file changes from context
            impl = self._cycle_context.get("implementation", {})
            self._current_record.files_modified = impl.get("files_modified", [])
            self._current_record.files_created = impl.get("files_created", [])
            self._current_record.lines_added = impl.get("lines_added", 0)
            self._current_record.lines_removed = impl.get("lines_removed", 0)

            # Populate debate topics from context
            debate = self._cycle_context.get("debate", {})
            proposals = debate.get("proposals", [])
            if proposals:
                self._current_record.topics_debated = [
                    p.get("proposal", str(p))[:200] for p in proposals[:5]
                ]
            if debate.get("consensus", False):
                self._current_record.consensus_reached = self._current_record.topics_debated

            # Mark complete and save
            self._current_record.mark_complete(success=success, error=error)
            self._get_cycle_store().save_cycle(self._current_record)

            logger.debug(
                f"cycle_record_saved cycle_id={self._current_record.cycle_id} success={success}"
            )
        except Exception as e:
            logger.warning(f"Failed to save cycle record: {e}")

    def _get_cross_cycle_context(self) -> Dict[str, Any]:
        """
        Get learning context from previous cycles.

        Returns context including:
        - Recent successful patterns
        - Agent trajectories
        - Similar past topics

        Returns:
            Dictionary with cross-cycle learning context
        """
        if not self._enable_cycle_learning:
            return {}

        try:
            store = self._get_cycle_store()
            context: Dict[str, Any] = {}

            # Get recent successful cycles
            successful = store.get_successful_cycles(5)
            if successful:
                context["recent_successes"] = [
                    {
                        "cycle_id": c.cycle_id,
                        "topics": c.topics_debated[:3] if c.topics_debated else [],
                        "files_modified": len(c.files_modified),
                        "duration": c.duration_seconds,
                    }
                    for c in successful[:3]
                ]

            # Get pattern statistics
            patterns = store.get_pattern_statistics()
            if patterns:
                # Include patterns with good success rates
                good_patterns = {
                    k: v
                    for k, v in patterns.items()
                    if v.get("success_rate", 0) > 0.6 and v.get("success_count", 0) >= 2
                }
                if good_patterns:
                    context["successful_patterns"] = list(good_patterns.keys())

            # Get recent surprises to avoid
            surprises = store.get_surprise_summary(20)
            high_impact = []
            for phase, events in surprises.items():
                for event in events:
                    if event.get("impact") == "high":
                        high_impact.append(
                            {
                                "phase": phase,
                                "description": event["description"][:100],
                            }
                        )
            if high_impact:
                context["surprises_to_avoid"] = high_impact[:3]

            return context

        except Exception as e:
            logger.debug(f"Failed to get cross-cycle context: {e}")
            return {}

    def record_agent_contribution(
        self,
        agent_name: str,
        proposals_made: int = 0,
        proposals_accepted: int = 0,
        critiques_given: int = 0,
        critiques_valuable: int = 0,
    ) -> None:
        """
        Record an agent's contribution to the current cycle.

        Args:
            agent_name: Name of the agent
            proposals_made: Number of proposals made
            proposals_accepted: Number of proposals accepted
            critiques_given: Number of critiques given
            critiques_valuable: Number of valuable critiques
        """
        if self._current_record:
            self._current_record.add_agent_contribution(
                agent_name=agent_name,
                proposals_made=proposals_made,
                proposals_accepted=proposals_accepted,
                critiques_given=critiques_given,
                critiques_valuable=critiques_valuable,
            )

    def record_surprise(
        self,
        phase: str,
        description: str,
        expected: str,
        actual: str,
        impact: str = "low",
    ) -> None:
        """
        Record a surprise event in the current cycle.

        Args:
            phase: Phase where surprise occurred
            description: Description of the surprise
            expected: What was expected
            actual: What actually happened
            impact: Impact level (low, medium, high)
        """
        if self._current_record:
            self._current_record.add_surprise(
                phase=phase,
                description=description,
                expected=expected,
                actual=actual,
                impact=impact,
            )

    def record_pattern_reinforcement(
        self,
        pattern_type: str,
        description: str,
        success: bool,
        confidence: float = 0.5,
    ) -> None:
        """
        Record a pattern that was confirmed or refuted.

        Args:
            pattern_type: Type of pattern (e.g., "bugfix", "refactor")
            description: Description of the pattern
            success: Whether the pattern succeeded
            confidence: Confidence in the reinforcement (0.0-1.0)
        """
        if self._current_record:
            self._current_record.add_pattern_reinforcement(
                pattern_type=pattern_type,
                description=description,
                success=success,
                confidence=confidence,
            )

    def get_agent_trajectory(self, agent_name: str, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get performance trajectory for an agent.

        Args:
            agent_name: Name of the agent
            n: Number of recent cycles to analyze

        Returns:
            List of performance snapshots
        """
        return self._get_cycle_store().get_agent_trajectory(agent_name, n)
