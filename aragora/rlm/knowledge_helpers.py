"""
TRUE RLM REPL helpers for Knowledge Mound context navigation.

Based on arXiv:2512.24601 "Recursive Language Models":
These helpers enable LLMs to programmatically navigate Knowledge Mound
content stored as Python variables in a REPL environment.

The Knowledge Mound is Aragora's organizational memory system that stores:
- Facts: Verified information with confidence scores
- Claims: Assertions from debates awaiting validation
- Evidence: Supporting documents and citations
- Relationships: Graph connections between knowledge items

Usage in TRUE RLM REPL:
    # Context is stored as a variable, not in the prompt
    km = load_knowledge_context(mound, workspace_id)

    # LLM writes code to query knowledge
    facts = get_facts(km, "rate limiting")
    high_conf = filter_by_confidence(km.facts, min_confidence=0.8)
    related = get_related(km, fact_id="f123")

    # Recursive calls for synthesis
    summary = RLM_M("Synthesize findings", subset=related)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.core import KnowledgeMound


@dataclass
class KnowledgeItem:
    """A single knowledge item (fact, claim, or evidence)."""

    id: str
    content: str
    source: str  # "fact", "claim", "evidence"
    confidence: float
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)  # IDs of related items


@dataclass
class KnowledgeREPLContext:
    """
    Structured Knowledge Mound context for TRUE RLM REPL navigation.

    The knowledge content is stored in Python data structures that the
    LLM can query programmatically rather than stuffing into prompts.
    """

    # Workspace info
    workspace_id: str

    # Knowledge items indexed by type
    facts: list[KnowledgeItem]
    claims: list[KnowledgeItem]
    evidence: list[KnowledgeItem]

    # All items for searching
    all_items: list[KnowledgeItem]

    # Quick lookup by ID
    by_id: dict[str, KnowledgeItem]

    # Relationship graph (item_id -> list of related item_ids)
    relationships: dict[str, list[str]]

    # Statistics
    total_items: int
    avg_confidence: float


def load_knowledge_context(
    mound: "KnowledgeMound",
    workspace_id: str,
    limit: int = 1000,
) -> KnowledgeREPLContext:
    """
    Load Knowledge Mound content into a structured context for REPL navigation.

    Args:
        mound: The KnowledgeMound instance
        workspace_id: Workspace to load knowledge from
        limit: Maximum items to load

    Returns:
        KnowledgeREPLContext with indexed access to knowledge

    Example in TRUE RLM REPL:
        >>> km = load_knowledge_context(mound, "ws-123")
        >>> print(f"Facts: {len(km.facts)}, Claims: {len(km.claims)}")
    """
    facts: list[KnowledgeItem] = []
    claims: list[KnowledgeItem] = []
    evidence: list[KnowledgeItem] = []
    all_items: list[KnowledgeItem] = []
    by_id: dict[str, KnowledgeItem] = {}
    relationships: dict[str, list[str]] = {}

    # Load from mound (handle both async and sync patterns)
    try:
        # Try to get facts
        if hasattr(mound, "get_facts"):
            raw_facts = mound.get_facts(workspace_id, limit=limit)
            for f in raw_facts or []:
                item = _to_knowledge_item(f, "fact")
                facts.append(item)
                all_items.append(item)
                by_id[item.id] = item

        # Try to get claims
        if hasattr(mound, "get_claims"):
            raw_claims = mound.get_claims(workspace_id, limit=limit)
            for c in raw_claims or []:
                item = _to_knowledge_item(c, "claim")
                claims.append(item)
                all_items.append(item)
                by_id[item.id] = item

        # Try to get evidence
        if hasattr(mound, "get_evidence"):
            raw_evidence = mound.get_evidence(workspace_id, limit=limit)
            for e in raw_evidence or []:
                item = _to_knowledge_item(e, "evidence")
                evidence.append(item)
                all_items.append(item)
                by_id[item.id] = item

        # Build relationship graph
        for item in all_items:
            relationships[item.id] = item.relationships

    except Exception:
        # Graceful degradation if mound doesn't have expected methods
        pass

    # Calculate stats
    total_items = len(all_items)
    avg_confidence = sum(i.confidence for i in all_items) / total_items if total_items > 0 else 0.0

    return KnowledgeREPLContext(
        workspace_id=workspace_id,
        facts=facts,
        claims=claims,
        evidence=evidence,
        all_items=all_items,
        by_id=by_id,
        relationships=relationships,
        total_items=total_items,
        avg_confidence=avg_confidence,
    )


def _to_knowledge_item(raw: Any, source: str) -> KnowledgeItem:
    """Convert raw knowledge data to KnowledgeItem."""
    if isinstance(raw, dict):
        return KnowledgeItem(
            id=raw.get("id", ""),
            content=raw.get("content", raw.get("text", "")),
            source=source,
            confidence=raw.get("confidence", 0.5),
            created_at=raw.get("created_at", ""),
            metadata=raw.get("metadata", {}),
            relationships=raw.get("relationships", raw.get("related_ids", [])),
        )
    elif hasattr(raw, "model_dump"):
        data = raw.model_dump()
        return _to_knowledge_item(data, source)
    elif hasattr(raw, "__dict__"):
        return _to_knowledge_item(vars(raw), source)
    else:
        return KnowledgeItem(
            id=str(hash(str(raw))),
            content=str(raw),
            source=source,
            confidence=0.5,
            created_at="",
            metadata={},
            relationships=[],
        )


def get_facts(
    context: KnowledgeREPLContext,
    query: Optional[str] = None,
    min_confidence: float = 0.0,
) -> list[KnowledgeItem]:
    """
    Get facts from knowledge context.

    Args:
        context: The knowledge REPL context
        query: Optional text query to filter
        min_confidence: Minimum confidence threshold

    Returns:
        List of matching facts

    Example in TRUE RLM REPL:
        >>> facts = get_facts(km, "rate limiting", min_confidence=0.7)
        >>> for f in facts:
        ...     print(f"[{f.confidence:.2f}] {f.content[:80]}...")
    """
    results = context.facts
    if min_confidence > 0:
        results = [f for f in results if f.confidence >= min_confidence]
    if query:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = [f for f in results if pattern.search(f.content)]
    return results


def get_claims(
    context: KnowledgeREPLContext,
    query: Optional[str] = None,
    validated_only: bool = False,
) -> list[KnowledgeItem]:
    """
    Get claims from knowledge context.

    Args:
        context: The knowledge REPL context
        query: Optional text query to filter
        validated_only: Only return validated claims

    Returns:
        List of matching claims

    Example in TRUE RLM REPL:
        >>> claims = get_claims(km, validated_only=True)
        >>> print(f"Found {len(claims)} validated claims")
    """
    results = context.claims
    if validated_only:
        results = [c for c in results if c.metadata.get("validated", False)]
    if query:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = [c for c in results if pattern.search(c.content)]
    return results


def get_evidence(
    context: KnowledgeREPLContext,
    query: Optional[str] = None,
    source_type: Optional[str] = None,
) -> list[KnowledgeItem]:
    """
    Get evidence from knowledge context.

    Args:
        context: The knowledge REPL context
        query: Optional text query to filter
        source_type: Filter by source type (e.g., "document", "debate")

    Returns:
        List of matching evidence items

    Example in TRUE RLM REPL:
        >>> evidence = get_evidence(km, source_type="document")
        >>> for e in evidence:
        ...     print(f"[{e.metadata.get('source_type')}] {e.content[:60]}...")
    """
    results = context.evidence
    if source_type:
        results = [e for e in results if e.metadata.get("source_type") == source_type]
    if query:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results = [e for e in results if pattern.search(e.content)]
    return results


def filter_by_confidence(
    items: list[KnowledgeItem],
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
) -> list[KnowledgeItem]:
    """
    Filter knowledge items by confidence score.

    Args:
        items: List of knowledge items
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold

    Returns:
        Filtered list of items

    Example in TRUE RLM REPL:
        >>> high_conf = filter_by_confidence(km.all_items, min_confidence=0.8)
        >>> low_conf = filter_by_confidence(km.all_items, max_confidence=0.3)
    """
    return [item for item in items if min_confidence <= item.confidence <= max_confidence]


def group_by_source(
    context: KnowledgeREPLContext,
) -> dict[str, list[KnowledgeItem]]:
    """
    Group all knowledge items by source type.

    Args:
        context: The knowledge REPL context

    Returns:
        Dictionary mapping source type to items

    Example in TRUE RLM REPL:
        >>> grouped = group_by_source(km)
        >>> for source, items in grouped.items():
        ...     print(f"{source}: {len(items)} items")
    """
    result: dict[str, list[KnowledgeItem]] = {}
    for item in context.all_items:
        if item.source not in result:
            result[item.source] = []
        result[item.source].append(item)
    return result


def search_knowledge(
    context: KnowledgeREPLContext,
    pattern: str,
    case_insensitive: bool = True,
) -> list[KnowledgeItem]:
    """
    Search knowledge items using regex pattern.

    This is the "grep" operation from the RLM paper.

    Args:
        context: The knowledge REPL context
        pattern: Regex pattern to match
        case_insensitive: Whether to ignore case

    Returns:
        List of matching items

    Example in TRUE RLM REPL:
        >>> api_items = search_knowledge(km, r"api|endpoint|rest")
        >>> security = search_knowledge(km, r"auth|security|permission")
    """
    flags = re.IGNORECASE if case_insensitive else 0
    regex = re.compile(pattern, flags)
    return [item for item in context.all_items if regex.search(item.content)]


def get_related(
    context: KnowledgeREPLContext,
    item_id: str,
    depth: int = 1,
) -> list[KnowledgeItem]:
    """
    Get related knowledge items via relationship graph.

    Args:
        context: The knowledge REPL context
        item_id: ID of the source item
        depth: How many relationship hops to follow

    Returns:
        List of related items

    Example in TRUE RLM REPL:
        >>> related = get_related(km, "fact-123", depth=2)
        >>> print(f"Found {len(related)} related items")
    """
    visited: set[str] = set()
    to_visit = [item_id]
    related: list[KnowledgeItem] = []

    for _ in range(depth):
        next_level = []
        for current_id in to_visit:
            if current_id in visited:
                continue
            visited.add(current_id)

            # Add related items
            for rel_id in context.relationships.get(current_id, []):
                if rel_id not in visited and rel_id in context.by_id:
                    related.append(context.by_id[rel_id])
                    next_level.append(rel_id)

        to_visit = next_level

    return related


def get_item(context: KnowledgeREPLContext, item_id: str) -> Optional[KnowledgeItem]:
    """
    Get a specific knowledge item by ID.

    Args:
        context: The knowledge REPL context
        item_id: ID of the item

    Returns:
        The knowledge item, or None if not found

    Example in TRUE RLM REPL:
        >>> item = get_item(km, "fact-456")
        >>> if item:
        ...     print(f"[{item.confidence:.2f}] {item.content}")
    """
    return context.by_id.get(item_id)


def partition_by_topic(
    context: KnowledgeREPLContext,
    topics: list[str],
) -> dict[str, list[KnowledgeItem]]:
    """
    Partition knowledge items by topic keywords.

    This is the "partition-map" operation from the RLM paper.

    Args:
        context: The knowledge REPL context
        topics: List of topic keywords

    Returns:
        Dictionary mapping topics to matching items

    Example in TRUE RLM REPL:
        >>> partitions = partition_by_topic(km, ["security", "performance", "api"])
        >>> for topic, items in partitions.items():
        ...     summary = RLM_M(f"Summarize {topic} findings", subset=items)
    """
    result: dict[str, list[KnowledgeItem]] = {topic: [] for topic in topics}
    result["other"] = []

    for item in context.all_items:
        matched = False
        for topic in topics:
            if topic.lower() in item.content.lower():
                result[topic].append(item)
                matched = True
                break
        if not matched:
            result["other"].append(item)

    return result


# RLM Primitives (for use in REPL)


def RLM_M(query: str, subset: Optional[list[KnowledgeItem]] = None) -> str:
    """
    Recursive RLM call placeholder.

    In TRUE RLM, this triggers a recursive LLM call on a subset of knowledge.
    This placeholder is replaced by the actual RLM runtime.

    Args:
        query: The query to answer
        subset: Optional subset of knowledge items

    Returns:
        Answer from recursive call

    Example in TRUE RLM REPL:
        >>> facts_summary = RLM_M("What are the verified facts?", subset=km.facts)
        >>> claims_summary = RLM_M("What claims need validation?", subset=km.claims)
        >>> FINAL(f"Facts: {facts_summary}. Claims: {claims_summary}")
    """
    raise NotImplementedError(
        "RLM_M must be called within a TRUE RLM REPL environment. "
        "Install with: pip install aragora[rlm]"
    )


def FINAL(answer: str) -> str:
    """
    Signal final answer in RLM.

    From the paper: The LLM produces FINAL(<ans>) when ready to terminate.

    Args:
        answer: The final answer

    Returns:
        The answer (for use in expressions)
    """
    return answer


def get_knowledge_helpers(
    mound: Optional["KnowledgeMound"] = None,
) -> dict[str, Any]:
    """
    Get all knowledge REPL helpers as a dictionary.

    This is used to inject helpers into a TRUE RLM REPL environment.

    Args:
        mound: Optional KnowledgeMound for context loading

    Returns:
        Dictionary of helper functions

    Example:
        >>> from aragora.rlm.knowledge_helpers import get_knowledge_helpers
        >>> helpers = get_knowledge_helpers(mound)
        >>> rlm_env.inject_helpers(helpers)
    """
    helpers: dict[str, Any] = {
        # Types
        "KnowledgeItem": KnowledgeItem,
        "KnowledgeREPLContext": KnowledgeREPLContext,
        # Context loading
        "load_knowledge_context": load_knowledge_context,
        # Queries
        "get_facts": get_facts,
        "get_claims": get_claims,
        "get_evidence": get_evidence,
        "get_item": get_item,
        "get_related": get_related,
        # Filtering
        "filter_by_confidence": filter_by_confidence,
        "group_by_source": group_by_source,
        "search_knowledge": search_knowledge,
        "partition_by_topic": partition_by_topic,
        # RLM primitives
        "RLM_M": RLM_M,
        "FINAL": FINAL,
    }

    # If mound provided, add a convenience loader
    if mound is not None:
        helpers["km_load"] = lambda ws_id: load_knowledge_context(mound, ws_id)

    return helpers


__all__ = [
    "KnowledgeItem",
    "KnowledgeREPLContext",
    "load_knowledge_context",
    "get_facts",
    "get_claims",
    "get_evidence",
    "get_item",
    "get_related",
    "filter_by_confidence",
    "group_by_source",
    "search_knowledge",
    "partition_by_topic",
    "RLM_M",
    "FINAL",
    "get_knowledge_helpers",
]
