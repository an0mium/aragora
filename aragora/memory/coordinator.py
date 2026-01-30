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
from typing import TYPE_CHECKING, Any, Optional, Callable, Awaitable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.debate.context import DebateContext
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import ConsensusMemory, ConsensusRecord
    from aragora.memory.store import CritiqueStore
    from aragora.knowledge.mound import KnowledgeMound


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
    ) -> "ConsensusRecord":
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


@dataclass
class MemoryTransaction:
    """A transaction containing multiple write operations."""

    id: str
    debate_id: str
    operations: list[WriteOperation] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
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

    def get_failed_operations(self) -> list[WriteOperation]:
        """Get all failed operations."""
        return [op for op in self.operations if op.status == WriteStatus.FAILED]

    def get_successful_operations(self) -> list[WriteOperation]:
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
        continuum_memory: Optional["ContinuumMemory"] = None,
        consensus_memory: Optional["ConsensusMemory"] = None,
        critique_store: Optional["CritiqueStore"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        options: CoordinatorOptions | None = None,
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
        except (KeyError, TypeError, ValueError, AttributeError) as e:
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
        except (KeyError, TypeError, ValueError, AttributeError) as e:
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
        except (KeyError, TypeError, ValueError, AttributeError) as e:
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
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.error("[coordinator] Mound rollback failed: %s", e)
            return False

    async def commit_debate_outcome(
        self,
        ctx: "DebateContext",
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

        except (ValueError, TypeError, KeyError, AttributeError) as e:
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
    ) -> list[WriteOperation]:
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

            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                if attempt < opts.max_retries:
                    await asyncio.sleep(opts.retry_delay_seconds)
                else:
                    op.mark_failed(str(e))
            except Exception as e:
                # Catch-all for unexpected errors (e.g., database errors)
                op.mark_failed(str(e))
                return  # Don't retry unexpected errors

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
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
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
