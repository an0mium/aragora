"""
Execution phase step executors and the MoleculeEngine.

This module contains executors for executing molecule steps:
- ShellStepExecutor: Execute shell commands
- ParallelStepExecutor: Execute steps across multiple agents in parallel
- ConditionalStepExecutor: Execute steps conditionally based on previous results
- MoleculeEngine: Main engine for executing molecules with checkpointing
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.nomic.beads import BeadStore
from aragora.nomic.molecules.base import (
    CompensatingAction,
    CyclicDependencyError,
    DeadlockError,
    Molecule,
    MoleculeResult,
    MoleculeStatus,
    MoleculeStep,
    MoleculeTransaction,
    StepStatus,
    TransactionState,
)
from aragora.nomic.molecules.proposal import AgentStepExecutor, StepExecutor
from aragora.nomic.molecules.verification import (
    DependencyGraph,
    get_deadlock_detector,
)

logger = logging.getLogger(__name__)


class ShellStepExecutor(StepExecutor):
    """Execute steps as shell commands using sandboxed subprocess runner.

    Security: Uses subprocess_runner module which enforces command allowlisting
    and blocks dangerous operations.
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step as shell command with security validation."""
        import shlex

        from aragora.utils.subprocess_runner import SandboxError, run_sandboxed

        command = step.config.get("command", "echo 'No command'")
        logger.info("Shell executing: %s", command)

        try:
            # Parse command string into arguments for secure execution
            args = shlex.split(command)
            if not args:
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "Empty command",
                }

            # Use sandboxed runner with command whitelist validation
            result = await run_sandboxed(
                args,
                timeout=min(step.timeout_seconds, 300.0),  # Cap at 5 minutes
                capture_output=True,
            )

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except SandboxError as e:
            # Command failed security validation
            logger.warning("Step %s blocked by sandbox: %s", step.name, e)
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Security: {e}",
            }
        except ValueError as e:
            # shlex.split failed
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Invalid command syntax: {e}",
            }
        except asyncio.TimeoutError:
            raise TimeoutError(f"Step {step.name} timed out after {step.timeout_seconds}s")


class ParallelStepExecutor(StepExecutor):
    """
    Execute steps by fanning out to multiple agents in parallel.

    Each agent receives the same task, and results are aggregated.

    Config options:
        agents: List of agent names to use
        task: The task for all agents
        aggregate: How to combine results (all, first, majority, custom)
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step across multiple agents in parallel."""
        agents_config = step.config.get("agents", ["claude", "gpt4"])
        task = step.config.get("task", step.name)
        aggregate = step.config.get("aggregate", "all")

        try:
            from aragora.agents.registry import AgentRegistry

            # Get agents
            agents = []
            for agent_name in agents_config:
                if AgentRegistry.is_registered(agent_name):
                    agent = AgentRegistry.create(agent_name)
                    if agent:
                        agents.append(agent)

            if not agents:
                return {"status": "skipped", "reason": "No agents available"}

            # Execute in parallel
            async def run_agent(agent):
                try:
                    result = await agent.generate(task)
                    return {"agent": agent.name, "result": result, "success": True}
                except (RuntimeError, OSError, ValueError) as e:
                    logger.warning("Agent %s execution failed: %s", agent.name, e)
                    return {"agent": agent.name, "error": f"Agent execution failed: {type(e).__name__}", "success": False}

            results = await asyncio.gather(*[run_agent(a) for a in agents])

            # Aggregate results
            successful = [r for r in results if r.get("success")]

            return {
                "status": "parallel_completed",
                "total_agents": len(agents),
                "successful": len(successful),
                "results": results,
                "aggregate": aggregate,
            }
        except ImportError as e:
            logger.warning("Agent modules not available: %s", e)
            return {"status": "skipped", "reason": "Agent modules not available"}
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Parallel execution failed: %s", e)
            raise


class ConditionalStepExecutor(StepExecutor):
    """
    Execute steps conditionally based on previous step results.

    Enables branching logic in molecules.

    Config options:
        condition: Expression to evaluate (simple key lookup or callable name)
        condition_key: Key in previous result to check
        expected_value: Value to compare against
        operator: Comparison operator (eq, ne, gt, lt, contains, exists)
        if_true: Action if condition is true (continue, skip, branch)
        if_false: Action if condition is false
        branch_step: Step ID to branch to if branching
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step conditionally based on context."""
        condition_key = step.config.get("condition_key", "status")
        expected_value = step.config.get("expected_value", "success")
        operator = step.config.get("operator", "eq")
        if_true = step.config.get("if_true", "continue")
        if_false = step.config.get("if_false", "skip")

        # Get the value from context (previous step results)
        previous_results = context.get("previous_results", {})
        actual_value = None

        # Try to find the value in context
        for key, value in previous_results.items():
            if isinstance(value, dict) and condition_key in value:
                actual_value = value[condition_key]
                break
        if actual_value is None:
            actual_value = previous_results.get(condition_key)

        # Evaluate condition
        condition_met = self._evaluate_condition(actual_value, expected_value, operator)

        action = if_true if condition_met else if_false

        result = {
            "status": "conditional_evaluated",
            "condition_key": condition_key,
            "expected": expected_value,
            "actual": actual_value,
            "operator": operator,
            "condition_met": condition_met,
            "action": action,
        }

        if action == "skip":
            result["should_skip"] = True
        elif action == "branch":
            result["branch_to"] = step.config.get("branch_step")

        return result

    def _evaluate_condition(self, actual: Any, expected: Any, operator: str) -> bool:
        """Evaluate a condition."""
        if operator == "eq":
            return actual == expected
        elif operator == "ne":
            return actual != expected
        elif operator == "gt":
            return actual > expected
        elif operator == "lt":
            return actual < expected
        elif operator == "gte":
            return actual >= expected
        elif operator == "lte":
            return actual <= expected
        elif operator == "contains":
            return expected in str(actual)
        elif operator == "exists":
            return actual is not None
        elif operator == "not_exists":
            return actual is None
        else:
            logger.warning("Unknown operator: %s, defaulting to equality", operator)
            return actual == expected


class MoleculeEngine:
    """
    Executes molecules with step-level persistence.

    Provides:
    - Automatic checkpointing before/after each step
    - Recovery from crashes
    - Pluggable step executors
    - Parallel execution where possible
    - Transaction safety with begin/commit/rollback semantics
    - Dependency graph validation
    - Deadlock detection for parallel molecules
    """

    def __init__(
        self,
        bead_store: BeadStore | None = None,
        checkpoint_dir: Path | None = None,
        enable_transactions: bool = True,
        validate_dependencies: bool = True,
        enable_deadlock_detection: bool = True,
    ):
        """
        Initialize the molecule engine.

        Args:
            bead_store: Optional bead store for tracking
            checkpoint_dir: Directory for checkpoints
            enable_transactions: Enable transaction wrapper with rollback support
            validate_dependencies: Validate step dependencies before execution
            enable_deadlock_detection: Enable deadlock detection for parallel molecules
        """
        self.bead_store = bead_store
        self.checkpoint_dir = checkpoint_dir or Path(".molecules")
        self._executors: dict[str, StepExecutor] = {
            "agent": AgentStepExecutor(),
            "shell": ShellStepExecutor(),
        }
        self._molecules: dict[str, Molecule] = {}
        self._transactions: dict[str, MoleculeTransaction] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._enable_transactions = enable_transactions
        self._validate_dependencies = validate_dependencies
        self._enable_deadlock_detection = enable_deadlock_detection

    def register_executor(self, step_type: str, executor: StepExecutor) -> None:
        """Register a custom step executor."""
        self._executors[step_type] = executor

    async def initialize(self) -> None:
        """Initialize the engine, loading checkpoints."""
        if self._initialized:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        await self._load_molecules()
        self._initialized = True
        logger.info("MoleculeEngine initialized with %s molecules", len(self._molecules))

    async def _load_molecules(self) -> None:
        """Load molecules from checkpoint files."""
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    molecule = Molecule.from_dict(data)
                    self._molecules[molecule.id] = molecule
            except (RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to load molecule from %s: %s", checkpoint_file, e)

    async def _checkpoint(self, molecule: Molecule) -> None:
        """Save a checkpoint for a molecule."""
        checkpoint_file = self.checkpoint_dir / f"{molecule.id}.json"
        molecule.updated_at = datetime.now(timezone.utc)

        try:
            with open(checkpoint_file, "w") as f:
                json.dump(molecule.to_dict(), f, indent=2)
            logger.debug(
                "Checkpointed molecule %s at step %s", molecule.id, molecule.current_step_index
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Failed to checkpoint molecule %s: %s", molecule.id, e)

    def validate_dependencies(self, molecule: Molecule) -> tuple[bool, str | None]:
        """
        Validate step dependencies before execution.

        Args:
            molecule: The molecule to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not molecule.steps:
            return True, None

        graph = DependencyGraph(molecule.steps)
        is_valid, error_details = graph.validate()

        if not is_valid and error_details:
            # Check if it's a missing dependency or a cycle
            if any(":" in detail for detail in error_details):
                # Missing dependencies
                return False, f"Missing dependencies: {error_details}"
            else:
                # Cycle detected
                return False, f"Cyclic dependency detected: {' -> '.join(error_details)}"

        return True, None

    async def _begin_transaction(self, molecule: Molecule) -> MoleculeTransaction | None:
        """Begin a transaction for molecule execution."""
        if not self._enable_transactions:
            return None

        # Take a snapshot of the current state
        snapshot = molecule.to_dict() if molecule else None

        transaction = MoleculeTransaction.begin(
            molecule_id=molecule.id,
            checkpoint_dir=self.checkpoint_dir,
            snapshot=snapshot,
        )
        self._transactions[transaction.transaction_id] = transaction
        logger.info("Started transaction %s for molecule %s", transaction.transaction_id, molecule.id)
        return transaction

    async def _commit_transaction(self, transaction: MoleculeTransaction) -> bool:
        """Commit a transaction."""
        if not transaction:
            return True

        success = await transaction.commit()
        if success:
            # Clean up transaction log after successful commit
            log_file = self.checkpoint_dir / f"txn_{transaction.transaction_id}.log"
            try:
                if log_file.exists():
                    log_file.unlink()
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to clean up transaction log: %s", e)
        return success

    async def _rollback_transaction(
        self, transaction: MoleculeTransaction, molecule: Molecule
    ) -> bool:
        """Rollback a transaction and restore molecule state."""
        if not transaction:
            return True

        success = await transaction.rollback()

        if success and transaction.pre_transaction_snapshot:
            # Restore molecule from snapshot
            try:
                restored = Molecule.from_dict(transaction.pre_transaction_snapshot)
                self._molecules[molecule.id] = restored
                await self._checkpoint(restored)
                logger.info("Restored molecule %s from pre-transaction state", molecule.id)
            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Failed to restore molecule state: %s", e)
                success = False

        # Release any locks held by this molecule
        if self._enable_deadlock_detection:
            detector = get_deadlock_detector()
            await detector.release_all_locks(molecule.id)

        return success

    async def _acquire_step_resources(self, molecule_id: str, step: MoleculeStep) -> bool:
        """Acquire resource locks for a step (deadlock prevention)."""
        if not self._enable_deadlock_detection:
            return True

        # Extract resource requirements from step config
        resources = step.config.get("resources", [])
        if not resources:
            # Default: lock by step type and name
            resources = [f"{step.step_type}:{step.name}"]

        detector = get_deadlock_detector()
        for resource_id in resources:
            try:
                acquired = await detector.acquire_lock(molecule_id, resource_id)
                if not acquired:
                    logger.warning(
                        "Failed to acquire lock on %s for molecule %s", resource_id, molecule_id
                    )
                    return False
            except DeadlockError as e:
                logger.error("Deadlock detected: %s", e)
                raise

        return True

    async def _release_step_resources(self, molecule_id: str, step: MoleculeStep) -> None:
        """Release resource locks after step completion."""
        if not self._enable_deadlock_detection:
            return

        resources = step.config.get("resources", [])
        if not resources:
            resources = [f"{step.step_type}:{step.name}"]

        detector = get_deadlock_detector()
        for resource_id in resources:
            await detector.release_lock(molecule_id, resource_id)

    async def _execute_step(
        self,
        step: MoleculeStep,
        context: dict[str, Any],
        transaction: MoleculeTransaction | None = None,
        molecule: Molecule | None = None,
    ) -> Any:
        """Execute a single step with transaction support."""
        executor = self._executors.get(step.step_type)
        if not executor:
            raise ValueError(f"No executor for step type: {step.step_type}")

        # Acquire resource locks for deadlock prevention
        if molecule:
            await self._acquire_step_resources(molecule.id, step)

        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        step.attempt_count += 1

        # Create compensating action for rollback
        if transaction and molecule:
            compensating = CompensatingAction(
                step_id=step.id,
                step_name=step.name,
                action_type="restore_checkpoint",
                checkpoint_path=self.checkpoint_dir / f"{molecule.id}.json",
                metadata={"step_config": step.config},
            )
            transaction.add_compensating_action(compensating)

        try:
            result = await asyncio.wait_for(
                executor.execute(step, context),
                timeout=step.timeout_seconds,
            )
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            step.result = result

            # Mark step completed in transaction
            if transaction:
                transaction.mark_step_completed(step.id)

            return result

        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error_message = f"Timeout after {step.timeout_seconds}s"
            raise

        except (RuntimeError, OSError, ValueError) as e:
            step.status = StepStatus.FAILED
            step.error_message = f"Step execution failed: {type(e).__name__}"
            raise

        finally:
            # Release resource locks
            if molecule:
                await self._release_step_resources(molecule.id, step)

    async def execute(
        self,
        molecule: Molecule,
        auto_rollback: bool = True,
    ) -> MoleculeResult:
        """
        Execute a molecule with automatic checkpointing and transaction safety.

        Args:
            molecule: The molecule to execute
            auto_rollback: Automatically rollback on failure (default True)

        Returns:
            MoleculeResult with execution details

        Raises:
            CyclicDependencyError: If cyclic dependencies are detected
            DependencyValidationError: If dependencies are invalid
            DeadlockError: If deadlock is detected during execution
        """
        async with self._lock:
            start_time = datetime.now(timezone.utc)
            transaction: MoleculeTransaction | None = None

            # Validate dependencies before execution
            if self._validate_dependencies and molecule.steps:
                is_valid, error_msg = self.validate_dependencies(molecule)
                if not is_valid:
                    molecule.status = MoleculeStatus.FAILED
                    molecule.error_message = error_msg
                    logger.error("Dependency validation failed: %s", error_msg)
                    return MoleculeResult(
                        molecule_id=molecule.id,
                        status=MoleculeStatus.FAILED,
                        completed_steps=0,
                        failed_steps=0,
                        total_steps=len(molecule.steps),
                        duration_seconds=0.0,
                        step_results={},
                        error_message=error_msg,
                    )

            # Begin transaction
            transaction = await self._begin_transaction(molecule)

            molecule.status = MoleculeStatus.RUNNING
            molecule.started_at = molecule.started_at or start_time

            self._molecules[molecule.id] = molecule
            await self._checkpoint(molecule)

            context: dict[str, Any] = {"step_results": {}}
            step_results: dict[str, Any] = {}
            execution_failed = False
            failure_reason: str | None = None

            try:
                # Execute steps in order (respecting dependencies)
                while True:
                    runnable_steps = molecule.get_next_runnable_steps()
                    if not runnable_steps:
                        # Check if all done or stuck
                        pending = [s for s in molecule.steps if s.status == StepStatus.PENDING]
                        if not pending:
                            break
                        else:
                            # Dependencies not met - likely circular
                            raise CyclicDependencyError(
                                [s.name for s in pending],
                                f"Circular dependency or unmet dependencies: "
                                f"{[s.name for s in pending]}",
                            )

                    # Execute runnable steps (could parallelize here)
                    for step in runnable_steps:
                        logger.info("Executing step: %s", step.name)
                        await self._checkpoint(molecule)

                        try:
                            result = await self._execute_step(step, context, transaction, molecule)
                            step_results[step.id] = result
                            context["step_results"][step.name] = result
                        except DeadlockError:
                            # Deadlock - need to rollback
                            execution_failed = True
                            failure_reason = f"Deadlock detected while executing {step.name}"
                            raise
                        except (RuntimeError, OSError, ValueError) as e:
                            logger.warning("Step %s failed: %s", step.name, e)
                            if step.can_retry():
                                step.status = StepStatus.PENDING
                                logger.info(
                                    "Will retry step %s (%d/%d)",
                                    step.name,
                                    step.attempt_count,
                                    step.max_attempts,
                                )
                            else:
                                step_results[step.id] = {"error": f"Step execution failed: {type(e).__name__}"}
                                molecule.status = MoleculeStatus.FAILED
                                molecule.error_message = f"Step {step.name} failed: {type(e).__name__}"
                                execution_failed = True
                                failure_reason = molecule.error_message
                                await self._checkpoint(molecule)
                                break

                        molecule.current_step_index += 1
                        await self._checkpoint(molecule)

                    if molecule.status == MoleculeStatus.FAILED:
                        break

                # Check final status
                failed_steps = [s for s in molecule.steps if s.status == StepStatus.FAILED]
                if failed_steps:
                    molecule.status = MoleculeStatus.FAILED
                    execution_failed = True
                else:
                    molecule.status = MoleculeStatus.COMPLETED

                molecule.completed_at = datetime.now(timezone.utc)
                await self._checkpoint(molecule)

            except (CyclicDependencyError, DeadlockError) as e:
                molecule.status = MoleculeStatus.FAILED
                molecule.error_message = f"Molecule failed: {type(e).__name__}"
                molecule.completed_at = datetime.now(timezone.utc)
                execution_failed = True
                failure_reason = molecule.error_message
                await self._checkpoint(molecule)
                logger.warning("Molecule %s failed: %s", molecule.id, e)

            except (RuntimeError, OSError, ValueError) as e:
                molecule.status = MoleculeStatus.FAILED
                molecule.error_message = f"Molecule execution failed: {type(e).__name__}"
                molecule.completed_at = datetime.now(timezone.utc)
                execution_failed = True
                failure_reason = molecule.error_message
                await self._checkpoint(molecule)
                logger.warning("Molecule %s failed: %s", molecule.id, e)

            # Handle transaction commit/rollback
            if transaction:
                if execution_failed and auto_rollback:
                    logger.info("Auto-rolling back transaction for molecule %s", molecule.id)
                    await self._rollback_transaction(transaction, molecule)
                elif not execution_failed:
                    await self._commit_transaction(transaction)
                else:
                    # Failed but no auto-rollback - just mark transaction as failed
                    transaction.state = TransactionState.FAILED
                    transaction.error_message = failure_reason

            # Release all locks held by this molecule
            if self._enable_deadlock_detection:
                detector = get_deadlock_detector()
                await detector.release_all_locks(molecule.id)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            return MoleculeResult(
                molecule_id=molecule.id,
                status=molecule.status,
                completed_steps=len(
                    [s for s in molecule.steps if s.status == StepStatus.COMPLETED]
                ),
                failed_steps=len([s for s in molecule.steps if s.status == StepStatus.FAILED]),
                total_steps=len(molecule.steps),
                duration_seconds=duration,
                step_results=step_results,
                error_message=molecule.error_message,
            )

    async def resume(self, molecule_id: str) -> MoleculeResult:
        """
        Resume a molecule from its last checkpoint.

        Args:
            molecule_id: ID of the molecule to resume

        Returns:
            MoleculeResult with execution details
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            # Try to load from checkpoint
            checkpoint_file = self.checkpoint_dir / f"{molecule_id}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    molecule = Molecule.from_dict(data)
                    self._molecules[molecule_id] = molecule
            else:
                raise ValueError(f"Molecule {molecule_id} not found")

        # Reset any RUNNING steps back to PENDING
        for step in molecule.steps:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.PENDING

        molecule.status = MoleculeStatus.RUNNING
        logger.info("Resuming molecule %s from step %s", molecule_id, molecule.current_step_index)

        return await self.execute(molecule)

    async def get_molecule(self, molecule_id: str) -> Molecule | None:
        """Get a molecule by ID."""
        return self._molecules.get(molecule_id)

    async def list_molecules(
        self,
        status: MoleculeStatus | None = None,
    ) -> list[Molecule]:
        """List molecules with optional status filter."""
        molecules = list(self._molecules.values())
        if status:
            molecules = [m for m in molecules if m.status == status]
        return molecules

    async def cancel(self, molecule_id: str) -> bool:
        """
        Cancel a running molecule.

        Args:
            molecule_id: ID of the molecule to cancel

        Returns:
            True if cancelled, False if not found
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            return False

        molecule.status = MoleculeStatus.CANCELLED
        molecule.completed_at = datetime.now(timezone.utc)
        await self._checkpoint(molecule)

        logger.info("Cancelled molecule %s", molecule_id)
        return True

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about molecules."""
        molecules = list(self._molecules.values())
        by_status: dict[str, int] = {}
        for m in molecules:
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1

        # Include transaction statistics
        txn_by_state: dict[str, int] = {}
        for txn in self._transactions.values():
            txn_by_state[txn.state.value] = txn_by_state.get(txn.state.value, 0) + 1

        return {
            "total_molecules": len(molecules),
            "by_status": by_status,
            "total_steps": sum(len(m.steps) for m in molecules),
            "transactions": {
                "total": len(self._transactions),
                "by_state": txn_by_state,
            },
        }

    async def get_transaction(self, transaction_id: str) -> MoleculeTransaction | None:
        """Get a transaction by ID."""
        return self._transactions.get(transaction_id)

    async def list_transactions(
        self,
        molecule_id: str | None = None,
        state: TransactionState | None = None,
    ) -> list[MoleculeTransaction]:
        """
        List transactions with optional filters.

        Args:
            molecule_id: Filter by molecule ID
            state: Filter by transaction state

        Returns:
            List of matching transactions
        """
        transactions = list(self._transactions.values())
        if molecule_id:
            transactions = [t for t in transactions if t.molecule_id == molecule_id]
        if state:
            transactions = [t for t in transactions if t.state == state]
        return transactions

    async def rollback_molecule(self, molecule_id: str) -> bool:
        """
        Manually rollback a molecule to its pre-execution state.

        Args:
            molecule_id: ID of the molecule to rollback

        Returns:
            True if rollback succeeded, False otherwise
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            logger.warning("Molecule %s not found for rollback", molecule_id)
            return False

        # Find the most recent active/failed transaction for this molecule
        matching_txns = [
            t
            for t in self._transactions.values()
            if t.molecule_id == molecule_id
            and t.state in (TransactionState.ACTIVE, TransactionState.FAILED)
        ]

        if not matching_txns:
            logger.warning("No active transaction found for molecule %s", molecule_id)
            return False

        # Rollback the most recent transaction
        transaction = max(matching_txns, key=lambda t: t.started_at)
        return await self._rollback_transaction(transaction, molecule)

    async def recover_incomplete_transactions(self) -> dict[str, bool]:
        """
        Recover incomplete transactions after a crash.

        Scans for transaction logs and rolls back any that were
        in progress when the system crashed.

        Returns:
            Dict mapping transaction_id to recovery success status
        """
        results: dict[str, bool] = {}

        # Scan for transaction log files
        for log_file in self.checkpoint_dir.glob("txn_*.log"):
            try:
                with open(log_file) as f:
                    log_data = json.load(f)

                txn_id = log_data.get("transaction_id")
                molecule_id = log_data.get("molecule_id")
                status = log_data.get("status")

                if status in ("committed", "rolled_back"):
                    # Transaction was completed - just clean up the log
                    log_file.unlink()
                    results[txn_id] = True
                    continue

                # Transaction was incomplete - attempt rollback
                logger.warning("Found incomplete transaction %s for molecule %s", txn_id, molecule_id)

                molecule = self._molecules.get(molecule_id)
                if molecule:
                    # Recreate transaction from log
                    transaction = MoleculeTransaction(
                        transaction_id=txn_id,
                        molecule_id=molecule_id,
                        state=TransactionState.ACTIVE,
                        started_at=datetime.fromisoformat(log_data.get("timestamp", "")),
                        checkpoint_dir=self.checkpoint_dir,
                        completed_steps=log_data.get("completed_steps", []),
                    )
                    success = await self._rollback_transaction(transaction, molecule)
                    results[txn_id] = success
                else:
                    logger.error(
                        "Cannot recover transaction %s: molecule %s not found", txn_id, molecule_id
                    )
                    results[txn_id] = False

            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Error processing transaction log %s: %s", log_file, e)
                results[str(log_file)] = False

        return results
