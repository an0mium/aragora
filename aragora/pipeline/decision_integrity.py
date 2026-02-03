"""Decision Integrity pipeline helpers.

Builds a Decision Integrity package from a debate artifact:
- Decision receipt (audit trail)
- Implementation plan (for multi-agent execution)
- Context snapshot (memory + knowledge state for auditability)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aragora.core_types import DebateResult
from aragora.gauntlet.receipt import DecisionReceipt
from aragora.implement import create_single_task_plan, generate_implement_plan
from aragora.implement.types import ImplementPlan

logger = logging.getLogger(__name__)


@dataclass
class ContextSnapshot:
    """Snapshot of knowledge/memory context available at decision time.

    Captures state from all memory tiers and knowledge sources so that
    the full evidentiary basis for a decision can be audited later.
    """

    continuum_entries: list[dict[str, Any]] = field(default_factory=list)
    cross_debate_context: str = ""
    cross_debate_ids: list[str] = field(default_factory=list)
    knowledge_items: list[dict[str, Any]] = field(default_factory=list)
    knowledge_sources: list[str] = field(default_factory=list)
    document_items: list[dict[str, Any]] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    total_context_tokens: int = 0
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "continuum_entries": self.continuum_entries,
            "cross_debate_context": self.cross_debate_context,
            "cross_debate_ids": self.cross_debate_ids,
            "knowledge_items": self.knowledge_items,
            "knowledge_sources": self.knowledge_sources,
            "document_items": self.document_items,
            "evidence_items": self.evidence_items,
            "total_context_tokens": self.total_context_tokens,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass
class DecisionIntegrityPackage:
    """Bundle of artifacts for decision implementation."""

    debate_id: str
    receipt: DecisionReceipt | None
    plan: ImplementPlan | None
    context_snapshot: ContextSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "context_snapshot": self.context_snapshot.to_dict() if self.context_snapshot else None,
        }


def _coerce_debate_result(debate: dict[str, Any]) -> DebateResult:
    """Best-effort conversion from stored debate dict to DebateResult."""
    agents = debate.get("agents", []) or []
    if isinstance(agents, str):
        agents = [a.strip() for a in agents.split(",") if a.strip()]

    return DebateResult(
        debate_id=str(debate.get("debate_id") or debate.get("id") or ""),
        task=str(debate.get("task") or debate.get("question") or ""),
        final_answer=str(debate.get("final_answer") or debate.get("conclusion") or ""),
        confidence=float(debate.get("confidence") or 0.0),
        consensus_reached=bool(debate.get("consensus_reached") or False),
        rounds_used=int(debate.get("rounds_used") or debate.get("rounds") or 0),
        rounds_completed=int(debate.get("rounds_completed") or 0),
        status=str(debate.get("status") or ""),
        participants=list(agents),
        metadata=dict(debate.get("metadata") or {}),
    )


def coerce_debate_result(debate: dict[str, Any]) -> DebateResult:
    """Public helper to coerce a stored debate dictionary into a DebateResult."""
    return _coerce_debate_result(debate)


async def capture_context_snapshot(
    task: str,
    *,
    continuum_memory: Any = None,
    cross_debate_memory: Any = None,
    knowledge_mound: Any = None,
    document_store: Any = None,
    evidence_store: Any = None,
    debate_id: str | None = None,
    max_entries: int = 10,
    max_tokens: int = 2000,
) -> ContextSnapshot:
    """Capture a snapshot of all available memory/knowledge context.

    Queries Continuum Memory, Cross-Debate Memory, and Knowledge Mound
    to produce an auditable record of what context was available when a
    decision was made.

    Args:
        task: The debate task/question used as the query.
        continuum_memory: Optional ContinuumMemory instance.
        cross_debate_memory: Optional CrossDebateMemory instance.
        knowledge_mound: Optional KnowledgeMound instance.
        max_entries: Max entries to retrieve from each source.
        max_tokens: Max tokens for cross-debate context.
    """
    start = time.monotonic()
    snapshot = ContextSnapshot()

    # 1. Continuum Memory (slow + glacial tiers for institutional memory)
    if continuum_memory is not None:
        try:
            from aragora.memory.tier_manager import MemoryTier

            entries = continuum_memory.retrieve(
                query=task,
                tiers=[MemoryTier.SLOW, MemoryTier.GLACIAL],
                limit=max_entries,
                min_importance=0.3,
            )
            snapshot.continuum_entries = [
                asdict(e) if hasattr(e, "__dataclass_fields__") else {"content": str(e)}
                for e in entries
            ]
        except Exception as exc:
            logger.debug("Continuum memory retrieval failed: %s", exc)

    # 2. Cross-Debate Memory
    if cross_debate_memory is not None:
        try:
            context = await cross_debate_memory.get_relevant_context(
                task=task,
                max_tokens=max_tokens,
            )
            snapshot.cross_debate_context = context or ""
            # Try to extract referenced debate IDs from the memory store
            if hasattr(cross_debate_memory, "_entries"):
                task_lower = task.lower()
                snapshot.cross_debate_ids = [
                    eid
                    for eid, entry in list(cross_debate_memory._entries.items())[:50]
                    if hasattr(entry, "task") and task_lower in str(entry.task).lower()
                ][:5]
        except Exception as exc:
            logger.debug("Cross-debate memory retrieval failed: %s", exc)

    # 3. Knowledge Mound
    if knowledge_mound is not None:
        try:
            result = await knowledge_mound.query(
                query=task,
                limit=max_entries,
            )
            if hasattr(result, "items"):
                snapshot.knowledge_items = [
                    item.to_dict() if hasattr(item, "to_dict") else {"content": str(item)}
                    for item in result.items
                ]
            if hasattr(result, "sources"):
                snapshot.knowledge_sources = [
                    s.value if hasattr(s, "value") else str(s) for s in result.sources
                ]
        except Exception as exc:
            logger.debug("Knowledge Mound query failed: %s", exc)

    # 4. Document store (uploaded documents)
    if document_store is not None:
        try:
            items = document_store.list_all()
            if isinstance(items, list):
                snapshot.document_items = items[:max_entries]
        except Exception as exc:
            logger.debug("Document store listing failed: %s", exc)

    # 5. Evidence store (debate-linked evidence)
    if evidence_store is not None:
        try:
            if debate_id:
                evidence = evidence_store.get_debate_evidence(debate_id)
            else:
                evidence = evidence_store.search_evidence(task, limit=max_entries)
            if isinstance(evidence, list):
                snapshot.evidence_items = evidence[:max_entries]
        except Exception as exc:
            logger.debug("Evidence store retrieval failed: %s", exc)

    # Estimate token count
    token_count = 0
    for e in snapshot.continuum_entries:
        token_count += len(str(e.get("content", ""))) // 4
    token_count += len(snapshot.cross_debate_context) // 4
    for item in snapshot.knowledge_items:
        token_count += len(str(item.get("content", ""))) // 4
    for item in snapshot.document_items:
        token_count += len(str(item.get("preview", ""))) // 4
    for item in snapshot.evidence_items:
        token_count += len(str(item.get("snippet", ""))) // 4
    snapshot.total_context_tokens = token_count
    snapshot.retrieval_time_ms = (time.monotonic() - start) * 1000

    return snapshot


async def build_decision_integrity_package(
    debate: dict[str, Any],
    *,
    include_receipt: bool = True,
    include_plan: bool = True,
    include_context: bool = False,
    plan_strategy: str = "single_task",
    repo_path: Path | None = None,
    continuum_memory: Any = None,
    cross_debate_memory: Any = None,
    knowledge_mound: Any = None,
    document_store: Any = None,
    evidence_store: Any = None,
) -> DecisionIntegrityPackage:
    """Build a Decision Integrity package from a debate payload.

    Args:
        debate: Debate payload (dict) from storage.
        include_receipt: Whether to generate a DecisionReceipt.
        include_plan: Whether to generate an implementation plan.
        include_context: Whether to capture a context snapshot.
        plan_strategy: "single_task" (default) or "gemini" (best-effort).
        repo_path: Repository root (defaults to cwd).
        continuum_memory: Optional ContinuumMemory for context snapshot.
        cross_debate_memory: Optional CrossDebateMemory for context snapshot.
        knowledge_mound: Optional KnowledgeMound for context snapshot.
        document_store: Optional DocumentStore for context snapshot.
        evidence_store: Optional EvidenceStore for context snapshot.
    """
    debate_result = _coerce_debate_result(debate)
    receipt = DecisionReceipt.from_debate_result(debate_result) if include_receipt else None

    plan: ImplementPlan | None = None
    if include_plan:
        repo_root = repo_path or Path.cwd()
        design = debate_result.final_answer or debate_result.task
        if plan_strategy == "gemini":
            try:
                plan = await generate_implement_plan(design=design, repo_path=repo_root)
            except Exception:
                plan = create_single_task_plan(design=design, repo_path=repo_root)
        else:
            plan = create_single_task_plan(design=design, repo_path=repo_root)

    context_snapshot: ContextSnapshot | None = None
    if include_context:
        context_snapshot = await capture_context_snapshot(
            task=debate_result.task,
            continuum_memory=continuum_memory,
            cross_debate_memory=cross_debate_memory,
            knowledge_mound=knowledge_mound,
            document_store=document_store,
            evidence_store=evidence_store,
            debate_id=debate_result.debate_id,
        )

    return DecisionIntegrityPackage(
        debate_id=debate_result.debate_id or str(debate.get("id") or ""),
        receipt=receipt,
        plan=plan,
        context_snapshot=context_snapshot,
    )
