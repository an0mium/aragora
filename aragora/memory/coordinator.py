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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable, Awaitable

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.debate.context import DebateContext
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import ConsensusMemory
    from aragora.memory.store import CritiqueStore
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class WriteStatus(Enum):
    """Status of a write operation."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class WriteOperation:
    """A single write operation within a transaction."""

    id: str
    target: str  # continuum, consensus, critique, mound
    status: WriteStatus = WriteStatus.PENDING
    data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def mark_success(self, result: Any = None) -> None:
        """Mark operation as successful."""
        self.status = WriteStatus.SUCCESS
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark operation as failed."""
        self.status = WriteStatus.FAILED
        self.error = error


@dataclass
class MemoryTransaction:
    """A transaction containing multiple write operations."""

    id: str
    debate_id: str
    operations: List[WriteOperation] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    rolled_back: bool = False

    @property
    def success(self) -> bool:
        """Check if all operations succeeded."""
        return all(op.status == WriteStatus.SUCCESS for op in self.operations)

    @property
    def partial_failure(self) -> bool:
        """Check if some but not all operations failed."""
        statuses = {op.status for op in self.operations}
        return WriteStatus.FAILED in statuses and WriteStatus.SUCCESS in statuses

    def get_failed_operations(self) -> List[WriteOperation]:
        """Get all failed operations."""
        return [op for op in self.operations if op.status == WriteStatus.FAILED]

    def get_successful_operations(self) -> List[WriteOperation]:
        """Get all successful operations."""
        return [op for op in self.operations if op.status == WriteStatus.SUCCESS]


@dataclass
class CoordinatorOptions:
    """Configuration for memory coordination behavior."""

    # Which systems to write to
    write_continuum: bool = True
    write_consensus: bool = True
    write_critique: bool = True
    write_mound: bool = True

    # Behavior
    rollback_on_failure: bool = True
    parallel_writes: bool = False  # Sequential by default for safety
    min_confidence_for_mound: float = 0.7  # Only write to mound if confidence >= threshold
    timeout_seconds: float = 30.0  # Timeout for entire transaction

    # Retry behavior
    max_retries: int = 2
    retry_delay_seconds: float = 0.5


@dataclass
class CoordinatorMetrics:
    """Metrics for memory coordination operations."""

    total_transactions: int = 0
    successful_transactions: int = 0
    partial_failures: int = 0
    full_failures: int = 0
    rollbacks_performed: int = 0
    total_writes: int = 0
    writes_per_target: Dict[str, int] = field(default_factory=dict)


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
        continuum_memory: Optional["ContinuumMemory"] = None,
        consensus_memory: Optional["ConsensusMemory"] = None,
        critique_store: Optional["CritiqueStore"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        options: Optional[CoordinatorOptions] = None,
    ):
        """
        Initialize the memory coordinator.

        Args:
            continuum_memory: ContinuumMemory instance for tiered learning
            consensus_memory: ConsensusMemory for historical outcomes
            critique_store: CritiqueStore for patterns
            knowledge_mound: KnowledgeMound for unified knowledge
            options: Default coordinator options
        """
        self.continuum_memory = continuum_memory
        self.consensus_memory = consensus_memory
        self.critique_store = critique_store
        self.knowledge_mound = knowledge_mound
        self.options = options or CoordinatorOptions()
        self.metrics = CoordinatorMetrics()

        # Rollback handlers for each target
        self._rollback_handlers: Dict[str, Callable[[WriteOperation], Awaitable[bool]]] = {}

        # Register default rollback handlers for available systems
        self._register_default_rollback_handlers()

    def _register_default_rollback_handlers(self) -> None:
        """Register default rollback handlers for each memory system.

        Handlers are only registered for systems that support deletion.
        Systems without delete support log warnings during rollback.
        """
        # Continuum memory has delete() method
        if self.continuum_memory:
            self._rollback_handlers["continuum"] = self._rollback_continuum

        # Knowledge mound has delete methods via semantic/graph stores
        if self.knowledge_mound:
            self._rollback_handlers["mound"] = self._rollback_mound

        # Note: ConsensusMemory and CritiqueStore don't currently support deletion
        # Rollback for these will log warnings (handled in _rollback_successful)

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
        except Exception as e:
            logger.error("[coordinator] Continuum rollback failed: %s", e)
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
        except Exception as e:
            logger.error("[coordinator] Mound rollback failed: %s", e)
            return False

    async def commit_debate_outcome(
        self,
        ctx: "DebateContext",
        options: Optional[CoordinatorOptions] = None,
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

        # Build operations based on options
        operations = self._build_operations(ctx, result, opts)
        transaction.operations = operations

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

        except Exception as e:
            logger.error("Transaction %s failed: %s", transaction.id, e)
            for op in transaction.operations:
                if op.status == WriteStatus.PENDING:
                    op.mark_failed(str(e))

        return transaction

    def _build_operations(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
        opts: CoordinatorOptions,
    ) -> List[WriteOperation]:
        """Build write operations based on configuration."""
        operations = []

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

        # Only write to mound if confidence meets threshold
        if (
            opts.write_mound
            and self.knowledge_mound
            and result.confidence >= opts.min_confidence_for_mound
        ):
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

        return operations

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
                else:
                    op.mark_failed(f"Unknown target: {op.target}")
                    return

                op.mark_success(result)
                return

            except Exception as e:
                if attempt < opts.max_retries:
                    await asyncio.sleep(opts.retry_delay_seconds)
                else:
                    op.mark_failed(str(e))

    async def _write_continuum(self, data: Dict[str, Any]) -> str:
        """Write to ContinuumMemory."""
        if not self.continuum_memory:
            raise ValueError("ContinuumMemory not configured")

        entry_id = self.continuum_memory.store_pattern(
            content=f"Debate: {data['task']}\nConclusion: {data['final_answer']}",
            importance=data["confidence"],
            metadata={
                "debate_id": data["debate_id"],
                "domain": data["domain"],
                "consensus_reached": data["consensus_reached"],
            },
        )
        logger.debug("[coordinator] Stored in continuum: %s", entry_id)
        return entry_id

    async def _write_consensus(self, data: Dict[str, Any]) -> str:
        """Write to ConsensusMemory."""
        if not self.consensus_memory:
            raise ValueError("ConsensusMemory not configured")

        from aragora.memory.consensus import ConsensusStrength

        # Map confidence to strength
        confidence = data["confidence"]
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

        consensus_id = self.consensus_memory.store_consensus(
            topic=data["topic"],
            conclusion=data["conclusion"],
            strength=strength,
            confidence=confidence,
            participating_agents=data["agents"],
            winner=data.get("winner"),
            domain=data["domain"],
            rounds=data["rounds_used"],
        )
        logger.debug("[coordinator] Stored in consensus: %s", consensus_id)
        return consensus_id

    async def _write_critique(self, data: Dict[str, Any]) -> str:
        """Write to CritiqueStore."""
        if not self.critique_store:
            raise ValueError("CritiqueStore not configured")

        result = data["result"]
        debate_id = data["debate_id"]

        # Store the debate result
        self.critique_store.store_result(result)
        logger.debug("[coordinator] Stored in critique: %s", debate_id)
        return debate_id

    async def _write_mound(self, data: Dict[str, Any]) -> str:
        """Write to KnowledgeMound."""
        if not self.knowledge_mound:
            raise ValueError("KnowledgeMound not configured")

        # Use the mound's native ingest method which handles item creation
        # The mound.ingest_debate_outcome is the preferred API
        if hasattr(self.knowledge_mound, "ingest_debate_outcome"):
            item_id = await self.knowledge_mound.ingest_debate_outcome(
                debate_id=data["debate_id"],
                task=data["task"],
                conclusion=data["conclusion"],
                confidence=data["confidence"],
                domain=data["domain"],
                consensus_reached=data["consensus_reached"],
                winner=data.get("winner"),
                key_claims=data.get("key_claims", []),
            )
        else:
            # Fallback: use store_knowledge if available
            item_id = await self.knowledge_mound.store_knowledge(
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

        logger.debug("[coordinator] Stored in mound: %s", item_id)
        return item_id

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
            except Exception as e:
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

    def get_metrics(self) -> Dict[str, Any]:
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
