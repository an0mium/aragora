"""
Memory Coordination System - Atomic writes across multiple memory systems.

Provides transaction semantics for coordinating writes to:
- ContinuumMemory (tiered learning)
- ConsensusMemory (historical outcomes)
- CritiqueStore (patterns and reputations)
- KnowledgeMound (unified knowledge)

Features:
- Atomic multi-system writes
- Rollback on partial failures
- Configurable write targets
- Parallel or sequential execution
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from collections.abc import Callable, Awaitable

from aragora.memory.surprise import ContentSurpriseScorer

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.debate.context import DebateContext
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import ConsensusMemory, ConsensusRecord
    from aragora.memory.store import CritiqueStore
    from aragora.knowledge.mound import KnowledgeMound
    from aragora.knowledge.mound.adapters import SupermemoryAdapter


# Protocols for the memory systems used in coordinator
# These define the minimal interfaces needed for memory coordination


@runtime_checkable
class ContinuumMemoryProtocol(Protocol):
    """Protocol for ContinuumMemory-like objects used in coordinator."""

    def store_pattern(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        """Store a pattern and return its ID."""
        ...

    def delete(
        self,
        memory_id: str,
        archive: bool = True,
        reason: str = "",
    ) -> dict[str, Any]:
        """Delete a memory entry."""
        ...


@runtime_checkable
class ConsensusMemoryProtocol(Protocol):
    """Protocol for ConsensusMemory-like objects used in coordinator."""

    def store_consensus(
        self,
        topic: str,
        conclusion: str,
        strength: Any,  # ConsensusStrength
        confidence: float,
        participating_agents: list[str],
        agreeing_agents: list[str],
        winner: str | None = None,
        domain: str = "general",
        rounds: int = 0,
        **kwargs: Any,
    ) -> ConsensusRecord:
        """Store a consensus record and return it."""
        ...

    def delete_consensus(
        self,
        consensus_id: str,
        cascade_dissents: bool = True,
    ) -> bool:
        """Delete a consensus record."""
        ...


@runtime_checkable
class CritiqueStoreProtocol(Protocol):
    """Protocol for CritiqueStore-like objects used in coordinator."""

    def store_result(self, result: Any) -> None:
        """Store a debate result."""
        ...

    def delete_debate(
        self,
        debate_id: str,
        cascade_critiques: bool = True,
    ) -> bool:
        """Delete a debate record."""
        ...


@runtime_checkable
class KnowledgeMoundProtocol(Protocol):
    """Protocol for KnowledgeMound-like objects used in coordinator."""

    async def ingest_debate_outcome(
        self,
        debate_id: str,
        task: str,
        conclusion: str,
        confidence: float,
        domain: str,
        consensus_reached: bool,
        winner: str | None = None,
        key_claims: list[str] | None = None,
    ) -> str:
        """Ingest a debate outcome and return the item ID."""
        ...

    async def store_knowledge(
        self,
        content: str,
        source: str,
        source_id: str,
        confidence: float,
        metadata: dict[str, Any],
    ) -> str:
        """Store knowledge and return the item ID."""
        ...

    async def delete_entry(
        self,
        km_id: str,
        archive: bool = True,
        reason: str = "",
    ) -> bool:
        """Delete a knowledge entry."""
        ...

    async def delete_node_async(self, node_id: str) -> bool:
        """Delete a knowledge node asynchronously."""
        ...


@runtime_checkable
class SupermemoryAdapterProtocol(Protocol):
    """Protocol for SupermemoryAdapter-like objects used in coordinator."""

    async def sync_debate_outcome(
        self,
        debate_result: Any,
        container_tag: str | None = None,
    ) -> Any:
        """Sync debate outcome to Supermemory. Returns SyncOutcomeResult."""
        ...


logger = logging.getLogger(__name__)


class WriteStatus(Enum):
    """Status of a write operation."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"  # Operation intentionally not executed (e.g., low confidence)


@dataclass
class WriteOperation:
    """A single write operation within a transaction."""

    id: str
    target: str  # continuum, consensus, critique, mound
    status: WriteStatus = WriteStatus.PENDING
    data: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def mark_success(self, result: Any = None) -> None:
        """Mark operation as successful."""
        self.status = WriteStatus.SUCCESS
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark operation as failed."""
        self.status = WriteStatus.FAILED
        self.error = error

    def mark_skipped(self, reason: str) -> None:
        """Mark operation as skipped (not executed due to conditions not met)."""
        self.status = WriteStatus.SKIPPED
        self.error = reason


@dataclass
class SkippedOperation:
    """
    Record of an operation that was intentionally not executed.

    This provides visibility into why certain operations were skipped,
    such as confidence thresholds not being met.
    """

    target: str  # Which system would have been written to
    reason: str  # Why the operation was skipped
    threshold: float | None = None  # Threshold that wasn't met
    actual_value: float | None = None  # The actual value that was below threshold
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SupermemoryRollbackMarker:
    """
    Marker for supermemory entries that should be deleted on next sync.

    Since Supermemory is an external service, rollback cannot be immediate.
    These markers are persisted and processed on the next sync cycle.

    Attributes:
        memory_id: The Supermemory ID of the entry to delete
        transaction_id: The transaction that triggered the rollback
        marked_at: When the entry was marked for deletion
        reason: Why the entry needs to be deleted
        retry_count: Number of deletion attempts (for retry logic)
    """

    memory_id: str
    transaction_id: str
    marked_at: datetime = field(default_factory=datetime.now)
    reason: str = "transaction_rollback"
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class MemoryTransaction:
    """A transaction containing multiple write operations."""

    id: str
    debate_id: str
    operations: list[WriteOperation] = field(default_factory=list)
    skipped_operations: list[SkippedOperation] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    rolled_back: bool = False

    @property
    def success(self) -> bool:
        """Check if all operations succeeded (skipped operations don't count as failures)."""
        return all(
            op.status in (WriteStatus.SUCCESS, WriteStatus.SKIPPED) for op in self.operations
        )

    @property
    def partial_failure(self) -> bool:
        """Check if some but not all operations failed."""
        statuses = {op.status for op in self.operations}
        return WriteStatus.FAILED in statuses and WriteStatus.SUCCESS in statuses

    @property
    def has_skipped(self) -> bool:
        """Check if any operations were skipped."""
        return len(self.skipped_operations) > 0

    def get_failed_operations(self) -> list[WriteOperation]:
        """Get all failed operations."""
        return [op for op in self.operations if op.status == WriteStatus.FAILED]

    def get_successful_operations(self) -> list[WriteOperation]:
        """Get all successful operations."""
        return [op for op in self.operations if op.status == WriteStatus.SUCCESS]

    def get_skipped_operations(self) -> list[SkippedOperation]:
        """Get all skipped operations with reasons."""
        return self.skipped_operations


@dataclass
class CoordinatorOptions:
    """Configuration for memory coordination behavior."""

    # Which systems to write to
    write_continuum: bool = True
    write_consensus: bool = True
    write_critique: bool = True
    write_mound: bool = True
    write_supermemory: bool = False  # External memory (opt-in, disabled by default)

    # Behavior
    rollback_on_failure: bool = True
    parallel_writes: bool = False  # Sequential by default for safety
    min_confidence_for_mound: float = 0.7  # Only write to mound if confidence >= threshold
    min_confidence_for_supermemory: float = 0.7  # Only sync to Supermemory if >= threshold
    timeout_seconds: float = 30.0  # Timeout for entire transaction

    # Retry behavior
    max_retries: int = 2
    retry_delay_seconds: float = 0.5

    # Supermemory-specific options
    supermemory_container_tag: str | None = None  # Optional container tag override

    # Retention gate integration
    enable_retention_gate: bool = False


@dataclass
class CoordinatorMetrics:
    """Metrics for memory coordination operations."""

    total_transactions: int = 0
    successful_transactions: int = 0
    partial_failures: int = 0
    full_failures: int = 0
    rollbacks_performed: int = 0
    total_writes: int = 0
    writes_per_target: dict[str, int] = field(default_factory=dict)


class MemoryCoordinator:
    """
    Coordinates writes across multiple memory systems with transaction semantics.

    Ensures data consistency when storing debate outcomes across:
    - ContinuumMemory for tiered learning
    - ConsensusMemory for historical outcomes
    - CritiqueStore for pattern extraction
    - KnowledgeMound for unified knowledge

    Example:
        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        transaction = await coordinator.commit_debate_outcome(
            ctx=debate_context,
            options=CoordinatorOptions(
                parallel_writes=True,
                rollback_on_failure=True,
            ),
        )

        if not transaction.success:
            logger.error("Failed to store outcome: %s", transaction.get_failed_operations())
    """

    def __init__(
        self,
        continuum_memory: ContinuumMemory | None = None,
        consensus_memory: ConsensusMemory | None = None,
        critique_store: CritiqueStore | None = None,
        knowledge_mound: KnowledgeMound | None = None,
        supermemory_adapter: SupermemoryAdapter | None = None,
        options: CoordinatorOptions | None = None,
        surprise_scorer: ContentSurpriseScorer | None = None,
        retention_gate: Any | None = None,
    ):
        """
        Initialize the memory coordinator.

        Args:
            continuum_memory: ContinuumMemory instance for tiered learning
            consensus_memory: ConsensusMemory for historical outcomes
            critique_store: CritiqueStore for patterns
            knowledge_mound: KnowledgeMound for unified knowledge
            supermemory_adapter: SupermemoryAdapter for external memory persistence
            options: Default coordinator options
            surprise_scorer: Titans-inspired surprise scorer for gating writes
            retention_gate: RetentionGate instance for surprise-driven retention
        """
        self.continuum_memory = continuum_memory
        self.consensus_memory = consensus_memory
        self.critique_store = critique_store
        self.knowledge_mound = knowledge_mound
        self.supermemory_adapter = supermemory_adapter
        self.options = options or CoordinatorOptions()
        self.metrics = CoordinatorMetrics()
        self.surprise_scorer = surprise_scorer or ContentSurpriseScorer()
        self.retention_gate = retention_gate

        # Supermemory entries marked for deletion (processed on next sync)
        self._supermemory_rollback_markers: list[SupermemoryRollbackMarker] = []

        # Rollback handlers for each target
        self._rollback_handlers: dict[str, Callable[[WriteOperation], Awaitable[bool]]] = {}

        # Register default rollback handlers for available systems
        self._register_default_rollback_handlers()

    def _register_default_rollback_handlers(self) -> None:
        """Register default rollback handlers for each memory system.

        Handlers are registered for all systems that support deletion:
        - ContinuumMemory: delete()
        - ConsensusMemory: delete_consensus()
        - CritiqueStore: delete_debate()
        - KnowledgeMound: delete_entry() or delete_node_async()
        """
        # Continuum memory has delete() method
        if self.continuum_memory:
            self._rollback_handlers["continuum"] = self._rollback_continuum

        # Consensus memory has delete_consensus() method
        if self.consensus_memory:
            self._rollback_handlers["consensus"] = self._rollback_consensus

        # Critique store has delete_debate() method
        if self.critique_store:
            self._rollback_handlers["critique"] = self._rollback_critique

        # Knowledge mound has delete methods via semantic/graph stores
        if self.knowledge_mound:
            self._rollback_handlers["mound"] = self._rollback_mound

        # Supermemory adapter - external memory (rollback not supported, log only)
        if self.supermemory_adapter:
            self._rollback_handlers["supermemory"] = self._rollback_supermemory

    async def _rollback_continuum(self, op: WriteOperation) -> bool:
        """Rollback a continuum memory write."""
        if not self.continuum_memory or not op.result:
            return False

        try:
            # op.result contains the entry_id returned from store_pattern
            result = self.continuum_memory.delete(
                memory_id=op.result,
                archive=True,
                reason="transaction_rollback",
            )
            return result.get("deleted", False)
        except Exception as e:  # noqa: BLE001 - graceful degradation, rollback is best-effort
            logger.error("[coordinator] Continuum rollback failed: %s", e)
            return False

    async def _rollback_consensus(self, op: WriteOperation) -> bool:
        """Rollback a consensus memory write."""
        if not self.consensus_memory or not op.result:
            return False

        try:
            # op.result contains the consensus_id returned from store_consensus
            return self.consensus_memory.delete_consensus(
                consensus_id=op.result,
                cascade_dissents=True,
            )
        except Exception as e:  # noqa: BLE001 - graceful degradation, rollback is best-effort
            logger.error("[coordinator] Consensus rollback failed: %s", e)
            return False

    async def _rollback_critique(self, op: WriteOperation) -> bool:
        """Rollback a critique store write."""
        if not self.critique_store or not op.result:
            return False

        try:
            # op.result contains the debate_id passed to store_result
            return self.critique_store.delete_debate(
                debate_id=op.result,
                cascade_critiques=True,
            )
        except Exception as e:  # noqa: BLE001 - graceful degradation, rollback is best-effort
            logger.error("[coordinator] Critique rollback failed: %s", e)
            return False

    async def _rollback_mound(self, op: WriteOperation) -> bool:
        """Rollback a knowledge mound write."""
        if not self.knowledge_mound or not op.result:
            return False

        try:
            # op.result contains the item_id returned from ingest_debate_outcome
            if hasattr(self.knowledge_mound, "delete_entry"):
                return await self.knowledge_mound.delete_entry(
                    km_id=op.result,
                    archive=True,
                    reason="transaction_rollback",
                )
            elif hasattr(self.knowledge_mound, "delete_node_async"):
                return await self.knowledge_mound.delete_node_async(op.result)
            else:
                logger.warning("[coordinator] Mound has no delete method")
                return False
        except Exception as e:  # noqa: BLE001 - graceful degradation, rollback is best-effort
            logger.error("[coordinator] Mound rollback failed: %s", e)
            return False

    async def _rollback_supermemory(self, op: WriteOperation) -> bool:
        """Rollback a supermemory write.

        Since Supermemory is an external service, immediate rollback may not be possible.
        Instead, we create a rollback marker that will be processed on the next sync cycle.
        This ensures eventual consistency even if the external service is unavailable.

        Returns:
            True if marker was created, False if no action needed.
        """
        if not self.supermemory_adapter or not op.result:
            return False

        # Create a rollback marker for deferred deletion
        marker = SupermemoryRollbackMarker(
            memory_id=str(op.result),
            transaction_id=op.id,
            reason="transaction_rollback",
        )
        self._supermemory_rollback_markers.append(marker)

        logger.info(
            "[coordinator] Supermemory rollback marker created for %s (transaction %s). "
            "Entry will be deleted on next sync cycle.",
            op.result,
            op.id,
        )
        return True

    def get_pending_supermemory_rollbacks(self) -> list[SupermemoryRollbackMarker]:
        """Get list of supermemory entries pending deletion.

        Use this to check for entries that need to be cleaned up on the next sync.
        """
        return list(self._supermemory_rollback_markers)

    async def process_supermemory_rollbacks(self) -> tuple[int, int]:
        """Process pending supermemory rollback markers.

        Attempts to delete entries marked for rollback. Entries that fail
        to delete are kept in the marker list for retry (up to max_retries).

        Returns:
            Tuple of (successful_deletions, failed_deletions)
        """
        if not self.supermemory_adapter:
            return 0, 0

        successful = 0
        failed = 0
        remaining_markers: list[SupermemoryRollbackMarker] = []

        for marker in self._supermemory_rollback_markers:
            try:
                # Attempt to delete from supermemory
                if hasattr(self.supermemory_adapter, "delete_memory"):
                    await self.supermemory_adapter.delete_memory(marker.memory_id)
                    successful += 1
                    logger.info(
                        "[coordinator] Successfully deleted supermemory entry %s",
                        marker.memory_id,
                    )
                else:
                    # Adapter doesn't support deletion - mark as failed
                    marker.retry_count += 1
                    if marker.retry_count < marker.max_retries:
                        remaining_markers.append(marker)
                    failed += 1
                    logger.warning("[coordinator] Supermemory adapter does not support deletion")
            except (ConnectionError, TimeoutError, OSError) as e:
                # Network error - retry later
                marker.retry_count += 1
                if marker.retry_count < marker.max_retries:
                    remaining_markers.append(marker)
                    logger.warning(
                        "[coordinator] Supermemory deletion failed (attempt %d/%d): %s",
                        marker.retry_count,
                        marker.max_retries,
                        e,
                    )
                else:
                    logger.error(
                        "[coordinator] Supermemory deletion abandoned after %d attempts: %s",
                        marker.max_retries,
                        marker.memory_id,
                    )
                failed += 1
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                # Data error - don't retry
                failed += 1
                logger.error(
                    "[coordinator] Supermemory deletion failed (data error): %s",
                    e,
                )

        self._supermemory_rollback_markers = remaining_markers
        return successful, failed

    async def commit_debate_outcome(
        self,
        ctx: DebateContext,
        options: CoordinatorOptions | None = None,
    ) -> MemoryTransaction:
        """
        Commit a debate outcome to all configured memory systems.

        Args:
            ctx: The DebateContext with completed debate
            options: Override default options for this transaction

        Returns:
            MemoryTransaction with status of all write operations
        """
        opts = options or self.options
        transaction = MemoryTransaction(
            id=str(uuid.uuid4()),
            debate_id=ctx.debate_id,
        )

        self.metrics.total_transactions += 1

        result = ctx.result
        if not result:
            logger.warning("Cannot commit outcome: no result in context")
            return transaction

        # Score surprise BEFORE writing to decide which systems to target
        surprise = self.surprise_scorer.score_debate_outcome(
            conclusion=result.final_answer or "",
            domain=ctx.domain,
            confidence=result.confidence,
        )
        logger.info(
            "Surprise score for debate %s: combined=%.3f should_store=%s",
            ctx.debate_id,
            surprise.combined,
            surprise.should_store,
        )

        # Build operations based on options (returns both operations and skipped)
        operations, skipped = self._build_operations(ctx, result, opts)
        transaction.operations = operations
        transaction.skipped_operations = skipped

        # Log skipped operations for visibility
        if skipped:
            logger.info(
                "Transaction %s: %d operations skipped due to thresholds: %s",
                transaction.id,
                len(skipped),
                ", ".join(f"{s.target} ({s.reason})" for s in skipped),
            )

        try:
            if opts.parallel_writes:
                await self._execute_parallel(transaction, opts)
            else:
                await self._execute_sequential(transaction, opts)

            # Handle partial failures
            if transaction.partial_failure and opts.rollback_on_failure:
                await self._rollback_successful(transaction)
                self.metrics.rollbacks_performed += 1

            transaction.completed_at = datetime.now()

            # Update metrics
            if transaction.success:
                self.metrics.successful_transactions += 1

                # Emit coordination event
                try:
                    from aragora.events.dispatcher import dispatch_event

                    dispatch_event(
                        "memory_coordination",
                        {
                            "transaction_id": transaction.id,
                            "debate_id": transaction.debate_id,
                            "success": transaction.success,
                            "operations_count": len(transaction.operations),
                            "skipped_count": len(transaction.skipped_operations) if transaction.skipped_operations else 0,
                        },
                    )
                except (ImportError, RuntimeError, AttributeError) as e:
                    logger.debug("Memory coordination event emission unavailable: %s", e)

            elif transaction.partial_failure:
                self.metrics.partial_failures += 1
            else:
                self.metrics.full_failures += 1

            self._update_write_metrics(transaction)

        except asyncio.TimeoutError:
            logger.error("Transaction %s timed out", transaction.id)
            for op in transaction.operations:
                if op.status == WriteStatus.PENDING:
                    op.mark_failed("timeout")

        except Exception as e:  # noqa: BLE001 - graceful degradation, mark remaining ops as failed
            logger.error("Transaction %s failed: %s", transaction.id, e)
            for op in transaction.operations:
                if op.status == WriteStatus.PENDING:
                    op.mark_failed(str(e))

        return transaction

    def _build_operations(
        self,
        ctx: DebateContext,
        result: DebateResult,
        opts: CoordinatorOptions,
    ) -> tuple[list[WriteOperation], list[SkippedOperation]]:
        """Build write operations based on configuration.

        Returns:
            Tuple of (operations_to_execute, skipped_operations).
            Skipped operations include details about why they were not executed.
        """
        operations: list[WriteOperation] = []
        skipped: list[SkippedOperation] = []

        if opts.write_continuum and self.continuum_memory:
            operations.append(
                WriteOperation(
                    id=str(uuid.uuid4()),
                    target="continuum",
                    data={
                        "debate_id": ctx.debate_id,
                        "task": ctx.env.task,
                        "final_answer": result.final_answer or "",
                        "confidence": result.confidence,
                        "domain": ctx.domain,
                        "consensus_reached": result.consensus_reached,
                    },
                )
            )

        if opts.write_consensus and self.consensus_memory:
            operations.append(
                WriteOperation(
                    id=str(uuid.uuid4()),
                    target="consensus",
                    data={
                        "debate_id": ctx.debate_id,
                        "topic": ctx.env.task,
                        "conclusion": result.final_answer or "",
                        "confidence": result.confidence,
                        "domain": ctx.domain,
                        "agents": [a.name for a in ctx.agents],
                        "winner": result.winner,
                        "rounds_used": result.rounds_used,
                    },
                )
            )

        if opts.write_critique and self.critique_store:
            operations.append(
                WriteOperation(
                    id=str(uuid.uuid4()),
                    target="critique",
                    data={
                        "result": result,
                        "debate_id": ctx.debate_id,
                    },
                )
            )

        # Write to mound only if confidence meets threshold
        if opts.write_mound and self.knowledge_mound:
            if result.confidence >= opts.min_confidence_for_mound:
                operations.append(
                    WriteOperation(
                        id=str(uuid.uuid4()),
                        target="mound",
                        data={
                            "debate_id": ctx.debate_id,
                            "task": ctx.env.task,
                            "conclusion": result.final_answer or "",
                            "confidence": result.confidence,
                            "domain": ctx.domain,
                            "consensus_reached": result.consensus_reached,
                            "winner": result.winner,
                            "key_claims": getattr(result, "key_claims", []),
                        },
                    )
                )
            else:
                # Track why mound write was skipped
                skipped.append(
                    SkippedOperation(
                        target="mound",
                        reason=f"Confidence {result.confidence:.2f} below threshold {opts.min_confidence_for_mound}",
                        threshold=opts.min_confidence_for_mound,
                        actual_value=result.confidence,
                    )
                )

        # Write to Supermemory only if enabled and confidence meets threshold
        if opts.write_supermemory and self.supermemory_adapter:
            if result.confidence >= opts.min_confidence_for_supermemory:
                operations.append(
                    WriteOperation(
                        id=str(uuid.uuid4()),
                        target="supermemory",
                        data={
                            "debate_result": result,
                            "container_tag": opts.supermemory_container_tag,
                        },
                    )
                )
            else:
                # Track why supermemory write was skipped
                skipped.append(
                    SkippedOperation(
                        target="supermemory",
                        reason=f"Confidence {result.confidence:.2f} below threshold {opts.min_confidence_for_supermemory}",
                        threshold=opts.min_confidence_for_supermemory,
                        actual_value=result.confidence,
                    )
                )

        return operations, skipped

    async def _execute_sequential(
        self,
        transaction: MemoryTransaction,
        opts: CoordinatorOptions,
    ) -> None:
        """Execute operations sequentially."""
        for op in transaction.operations:
            await self._execute_operation(op, opts)

            # Stop on failure if rollback enabled
            if op.status == WriteStatus.FAILED and opts.rollback_on_failure:
                break

    async def _execute_parallel(
        self,
        transaction: MemoryTransaction,
        opts: CoordinatorOptions,
    ) -> None:
        """Execute operations in parallel."""
        tasks = [self._execute_operation(op, opts) for op in transaction.operations]
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=opts.timeout_seconds,
        )

    async def _execute_operation(
        self,
        op: WriteOperation,
        opts: CoordinatorOptions,
    ) -> None:
        """Execute a single write operation with retries."""
        for attempt in range(opts.max_retries + 1):
            try:
                if op.target == "continuum":
                    result = await self._write_continuum(op.data)
                elif op.target == "consensus":
                    result = await self._write_consensus(op.data)
                elif op.target == "critique":
                    result = await self._write_critique(op.data)
                elif op.target == "mound":
                    result = await self._write_mound(op.data)
                elif op.target == "supermemory":
                    result = await self._write_supermemory(op.data)
                else:
                    op.mark_failed(f"Unknown target: {op.target}")
                    return

                op.mark_success(result)
                return

            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                if attempt < opts.max_retries:
                    await asyncio.sleep(opts.retry_delay_seconds)
                else:
                    op.mark_failed(str(e))
            except Exception as e:  # noqa: BLE001 - coordinator boundary must catch all subsystem failures
                op.mark_failed(str(e))
                return

    async def _write_continuum(self, data: dict[str, Any]) -> str:
        """Write to ContinuumMemory."""
        if not self.continuum_memory:
            raise ValueError("ContinuumMemory not configured")

        content = f"Debate: {data['task']}\nConclusion: {data['final_answer']}"
        importance = data["confidence"]
        metadata = {
            "debate_id": data["debate_id"],
            "domain": data["domain"],
            "consensus_reached": data["consensus_reached"],
        }

        # Use store_pattern if available (protocol method), otherwise fall back to add
        if hasattr(self.continuum_memory, "store_pattern"):
            store_pattern_fn: Callable[..., str] = getattr(self.continuum_memory, "store_pattern")
            entry_id = store_pattern_fn(
                content=content,
                importance=importance,
                metadata=metadata,
            )
        else:
            # Fallback to the standard add() method
            entry = self.continuum_memory.add(
                id=data["debate_id"],
                content=content,
                importance=importance,
                metadata=metadata,
            )
            entry_id = entry.id

        logger.debug("[coordinator] Stored in continuum: %s", entry_id)
        return entry_id

    async def _write_consensus(self, data: dict[str, Any]) -> str:
        """Write to ConsensusMemory."""
        if not self.consensus_memory:
            raise ValueError("ConsensusMemory not configured")

        from aragora.memory.consensus import ConsensusStrength

        # Map confidence to strength
        confidence: float = data["confidence"]
        if confidence >= 0.9:
            strength = ConsensusStrength.UNANIMOUS
        elif confidence >= 0.75:
            strength = ConsensusStrength.STRONG
        elif confidence >= 0.6:
            strength = ConsensusStrength.MODERATE
        elif confidence >= 0.5:
            strength = ConsensusStrength.WEAK
        else:
            strength = ConsensusStrength.SPLIT

        # Build metadata with winner if provided
        metadata: dict[str, Any] = {}
        winner = data.get("winner")
        if winner:
            metadata["winner"] = winner

        record = self.consensus_memory.store_consensus(
            topic=data["topic"],
            conclusion=data["conclusion"],
            strength=strength,
            confidence=confidence,
            participating_agents=data["agents"],
            agreeing_agents=data["agents"],  # All agents agree for coordinator
            domain=data["domain"],
            rounds=data["rounds_used"],
            metadata=metadata,
        )
        logger.debug("[coordinator] Stored in consensus: %s", record.id)
        return record.id

    async def _write_critique(self, data: dict[str, Any]) -> str:
        """Write to CritiqueStore."""
        if not self.critique_store:
            raise ValueError("CritiqueStore not configured")

        result = data["result"]
        debate_id: str = data["debate_id"]

        # Store the debate result using store_result if available, else store_debate
        if hasattr(self.critique_store, "store_result"):
            store_result_fn: Callable[[Any], None] = getattr(self.critique_store, "store_result")
            store_result_fn(result)
        else:
            # Standard CritiqueStore uses store_debate
            self.critique_store.store_debate(result)

        logger.debug("[coordinator] Stored in critique: %s", debate_id)
        return debate_id

    async def _write_mound(self, data: dict[str, Any]) -> str:
        """Write to KnowledgeMound."""
        if not self.knowledge_mound:
            raise ValueError("KnowledgeMound not configured")

        item_id: str

        # Use the mound's native ingest method which handles item creation
        # The mound.ingest_debate_outcome is the preferred API
        if hasattr(self.knowledge_mound, "ingest_debate_outcome"):
            ingest_fn: Callable[..., Awaitable[str]] = getattr(
                self.knowledge_mound, "ingest_debate_outcome"
            )
            item_id = await ingest_fn(
                debate_id=data["debate_id"],
                task=data["task"],
                conclusion=data["conclusion"],
                confidence=data["confidence"],
                domain=data["domain"],
                consensus_reached=data["consensus_reached"],
                winner=data.get("winner"),
                key_claims=data.get("key_claims", []),
            )
        elif hasattr(self.knowledge_mound, "store_knowledge"):
            # Fallback: use store_knowledge if available
            store_knowledge_fn: Callable[..., Awaitable[str]] = getattr(
                self.knowledge_mound, "store_knowledge"
            )
            item_id = await store_knowledge_fn(
                content=f"{data['task']}\n\nConclusion: {data['conclusion']}",
                source="debate",
                source_id=data["debate_id"],
                confidence=data["confidence"],
                metadata={
                    "domain": data["domain"],
                    "consensus_reached": data["consensus_reached"],
                    "winner": data.get("winner"),
                    "key_claims": data.get("key_claims", []),
                },
            )
        else:
            raise ValueError(
                "KnowledgeMound has neither ingest_debate_outcome nor store_knowledge method"
            )

        logger.debug("[coordinator] Stored in mound: %s", item_id)
        return item_id

    async def _write_supermemory(self, data: dict[str, Any]) -> str | None:
        """Write to Supermemory external memory."""
        if not self.supermemory_adapter:
            raise ValueError("SupermemoryAdapter not configured")

        debate_result = data["debate_result"]
        container_tag = data.get("container_tag")

        # Use the adapter's sync_debate_outcome method
        sync_result = await self.supermemory_adapter.sync_debate_outcome(
            debate_result=debate_result,
            container_tag=container_tag,
        )

        if sync_result.success:
            logger.debug("[coordinator] Synced to supermemory: %s", sync_result.memory_id)
            return sync_result.memory_id
        else:
            # Sync returned success=False but didn't raise an exception
            # This can happen for below-threshold syncs or client unavailable
            if sync_result.error:
                raise ValueError(f"Supermemory sync failed: {sync_result.error}")
            # Skipped due to threshold - not an error
            logger.debug("[coordinator] Supermemory sync skipped: %s", sync_result.error)
            return None

    async def _rollback_successful(self, transaction: MemoryTransaction) -> None:
        """Roll back successful operations after a partial failure."""
        successful = transaction.get_successful_operations()

        for op in reversed(successful):
            try:
                handler = self._rollback_handlers.get(op.target)
                if handler:
                    await handler(op)
                    op.status = WriteStatus.ROLLED_BACK
                    logger.debug("[coordinator] Rolled back %s: %s", op.target, op.id)
                else:
                    # No rollback handler - mark but can't undo
                    logger.warning(
                        "[coordinator] No rollback handler for %s, cannot undo %s",
                        op.target,
                        op.id,
                    )
            except Exception as e:  # noqa: BLE001 - graceful degradation, rollback is best-effort
                logger.error("[coordinator] Rollback failed for %s: %s", op.id, e)

        transaction.rolled_back = True

    def register_rollback_handler(
        self,
        target: str,
        handler: Callable[[WriteOperation], Awaitable[bool]],
    ) -> None:
        """Register a rollback handler for a target system."""
        self._rollback_handlers[target] = handler

    def _update_write_metrics(self, transaction: MemoryTransaction) -> None:
        """Update write metrics from a completed transaction."""
        for op in transaction.operations:
            if op.status == WriteStatus.SUCCESS:
                self.metrics.total_writes += 1
                self.metrics.writes_per_target[op.target] = (
                    self.metrics.writes_per_target.get(op.target, 0) + 1
                )

    async def evaluate_retention(
        self,
        transaction: MemoryTransaction,
    ) -> list[Any] | None:
        """Post-write hook: evaluate retention for committed items.

        If retention_gate is configured, evaluates all successful writes
        and returns RetentionDecision list for downstream processing.

        Args:
            transaction: Completed transaction to evaluate

        Returns:
            List of RetentionDecision objects, or None if gate not configured.
        """
        if not self.retention_gate or not self.options.enable_retention_gate:
            return None

        decisions = []
        for op in transaction.get_successful_operations():
            if op.result and op.target in ("continuum", "mound"):
                confidence = op.data.get("confidence", 0.5)
                content = op.data.get("task", "") or op.data.get("conclusion", "")

                decision = self.retention_gate.evaluate(
                    item_id=str(op.result),
                    source=op.target,
                    content=content,
                    outcome_surprise=0.5,  # Default; callers can override
                    current_confidence=confidence,
                )
                decisions.append(decision)

        return decisions if decisions else None

    def get_metrics(self) -> dict[str, Any]:
        """Get current coordinator metrics."""
        return {
            "total_transactions": self.metrics.total_transactions,
            "successful_transactions": self.metrics.successful_transactions,
            "partial_failures": self.metrics.partial_failures,
            "full_failures": self.metrics.full_failures,
            "rollbacks_performed": self.metrics.rollbacks_performed,
            "total_writes": self.metrics.total_writes,
            "writes_per_target": self.metrics.writes_per_target,
            "success_rate": (
                self.metrics.successful_transactions / self.metrics.total_transactions
                if self.metrics.total_transactions > 0
                else 0.0
            ),
        }


__all__ = [
    "MemoryCoordinator",
    "MemoryTransaction",
    "WriteOperation",
    "WriteStatus",
    "CoordinatorOptions",
    "CoordinatorMetrics",
]
