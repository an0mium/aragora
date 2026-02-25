#!/usr/bin/env python3
"""
Golden Path 5: Knowledge-Enriched Debate
=========================================

Demonstrates how to query the Knowledge Mound for prior knowledge, then
inject that context into a debate so agents can reference organizational
data when making decisions.

The flow:
  1. Create a mock knowledge store with sample items
  2. Query the store with semantic search
  3. Inject retrieved knowledge as debate context
  4. Run a debate where agents reference the context
  5. Show how knowledge improves decision quality

No API keys or database required -- uses in-memory mock store.

Usage:
    python examples/golden_paths/knowledge_query/main.py

Expected runtime: <5 seconds
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a standalone script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aragora_debate import Arena, DebateConfig, StyledMockAgent


# ----------------------------------------------------------------
# Mock Knowledge Store
# ----------------------------------------------------------------
# In production, this would be KnowledgeMound backed by SQLite/PostgreSQL.
# The mock below replicates the query API for self-contained execution.


@dataclass
class KnowledgeItem:
    """A knowledge item with content, source, and importance score."""
    id: str
    title: str
    content: str
    source: str
    importance: float  # 0.0 to 1.0
    created_at: str
    tags: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of a knowledge store query."""
    items: list[KnowledgeItem]
    total_count: int
    query: str


class MockKnowledgeStore:
    """In-memory knowledge store that simulates KnowledgeMound queries.

    In production, replace with:
        from aragora.knowledge.mound.core import KnowledgeMound
        km = KnowledgeMound(workspace_id="your-workspace")
        results = await km.query("your query", limit=5)
    """

    def __init__(self) -> None:
        self._items: list[KnowledgeItem] = []

    def ingest(self, item: KnowledgeItem) -> str:
        """Add a knowledge item to the store."""
        self._items.append(item)
        return item.id

    async def query(self, query: str, *, limit: int = 5) -> QueryResult:
        """Search knowledge items by keyword matching.

        The real KnowledgeMound uses vector embeddings for semantic search.
        This mock uses simple keyword matching for demonstration.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[float, KnowledgeItem]] = []
        for item in self._items:
            text = f"{item.title} {item.content} {' '.join(item.tags)}".lower()
            # Score = keyword overlap * importance
            overlap = sum(1 for w in query_words if w in text)
            score = overlap * item.importance
            if score > 0:
                scored.append((score, item))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        items = [item for _, item in scored[:limit]]

        return QueryResult(
            items=items,
            total_count=len(scored),
            query=query,
        )


# ----------------------------------------------------------------
# Seed the knowledge store with sample organizational data
# ----------------------------------------------------------------

def seed_knowledge_store() -> MockKnowledgeStore:
    """Create and populate a knowledge store with sample data."""
    store = MockKnowledgeStore()

    items = [
        KnowledgeItem(
            id=uuid.uuid4().hex[:12],
            title="Q4 2025 Revenue Report",
            content=(
                "Revenue grew 23% YoY to $4.2M in Q4 2025. SaaS ARR reached "
                "$12.8M. Enterprise segment grew 41%, SMB grew 12%. Net revenue "
                "retention: 118%. Customer acquisition cost decreased 15%."
            ),
            source="finance",
            importance=0.95,
            created_at="2026-01-15T10:00:00Z",
            tags=["revenue", "q4", "growth", "metrics"],
        ),
        KnowledgeItem(
            id=uuid.uuid4().hex[:12],
            title="Customer Churn Analysis (Jan 2026)",
            content=(
                "Monthly churn rate dropped to 2.1% (from 3.4% in Q3) after "
                "onboarding improvements. Key drivers: 1) New interactive "
                "tutorials reduced time-to-value by 40%. 2) Proactive health "
                "scoring identified at-risk accounts 2 weeks earlier. Top churn "
                "reasons: product complexity (34%), pricing (28%), competition (22%)."
            ),
            source="customer_success",
            importance=0.85,
            created_at="2026-02-01T09:00:00Z",
            tags=["churn", "retention", "onboarding", "metrics"],
        ),
        KnowledgeItem(
            id=uuid.uuid4().hex[:12],
            title="Engineering Velocity Report",
            content=(
                "Sprint velocity increased 18% in Q4 after migrating to trunk-based "
                "development. Deployment frequency: 12x/day (up from 3x/day). "
                "Mean time to recovery (MTTR): 22 minutes (down from 4 hours). "
                "Test coverage: 87%. Tech debt ratio: 14% (target: <15%)."
            ),
            source="engineering",
            importance=0.80,
            created_at="2026-01-20T14:00:00Z",
            tags=["engineering", "velocity", "deployment", "metrics"],
        ),
        KnowledgeItem(
            id=uuid.uuid4().hex[:12],
            title="Competitor Analysis: Acme Corp Product Launch",
            content=(
                "Acme Corp launched their AI decision platform in January 2026 "
                "targeting enterprise customers. Key differentiators they claim: "
                "real-time streaming, SOC 2 compliance, and Salesforce integration. "
                "Pricing: $500/seat/month. Weakness: single-model (GPT-4 only), "
                "no multi-agent debate, no decision receipts."
            ),
            source="product",
            importance=0.75,
            created_at="2026-02-10T11:00:00Z",
            tags=["competitor", "market", "strategy", "pricing"],
        ),
        KnowledgeItem(
            id=uuid.uuid4().hex[:12],
            title="Board Meeting Notes: Growth Strategy",
            content=(
                "Board approved Q1 2026 priorities: 1) Expand enterprise sales "
                "team (3 new AEs). 2) Launch self-serve tier for SMBs. 3) Achieve "
                "SOC 2 Type II certification by June. 4) Explore vertical-specific "
                "packaging (healthcare, finance, legal). Budget: $2.1M for Q1."
            ),
            source="leadership",
            importance=0.90,
            created_at="2026-01-28T16:00:00Z",
            tags=["strategy", "growth", "board", "priorities"],
        ),
        KnowledgeItem(
            id=uuid.uuid4().hex[:12],
            title="Previous Debate: API Rate Limiting Design",
            content=(
                "Prior debate concluded with consensus (82% confidence) on hybrid "
                "rate limiting: token bucket per-user + sliding window global. "
                "Key finding: Redis sorted sets provide O(log N) window queries. "
                "Dissent: one agent advocated for leaky bucket approach."
            ),
            source="debate",
            importance=0.70,
            created_at="2026-02-05T13:00:00Z",
            tags=["debate", "architecture", "rate-limiting", "api"],
        ),
    ]

    for item in items:
        store.ingest(item)

    return store


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

async def main() -> None:
    print("=" * 64)
    print("  Aragora Golden Path: Knowledge-Enriched Debate")
    print("=" * 64)
    print()

    # ----------------------------------------------------------------
    # Step 1: Set up the knowledge store
    # ----------------------------------------------------------------
    store = seed_knowledge_store()
    print(f"Knowledge store initialized with {len(store._items)} items.")
    print()

    # ----------------------------------------------------------------
    # Step 2: Query for relevant knowledge
    # ----------------------------------------------------------------
    query = "What are our key business metrics and growth strategy?"
    print(f"Query: \"{query}\"")
    print()

    results = await store.query(query, limit=4)
    print(f"Retrieved {results.total_count} matching items (showing top {len(results.items)}):")
    print()

    for i, item in enumerate(results.items, 1):
        print(f"  {i}. [{item.source}] {item.title} (importance: {item.importance:.0%})")
        # Show first 100 chars of content
        preview = item.content[:100].rstrip() + "..."
        print(f"     {preview}")
        print()

    # ----------------------------------------------------------------
    # Step 3: Build debate context from retrieved knowledge
    # ----------------------------------------------------------------
    context_parts = []
    for item in results.items:
        context_parts.append(f"[{item.source.upper()}] {item.title}:\n{item.content}")
    context = "\n\n".join(context_parts)

    print("--- Knowledge Context (injected into debate) ---")
    for line in context.split("\n"):
        print(f"  {line}")
    print()

    # ----------------------------------------------------------------
    # Step 4: Run a knowledge-enriched debate
    # ----------------------------------------------------------------
    question = (
        "Based on our latest business metrics and board-approved strategy, "
        "what should be our top product priority for Q1 2026? Consider "
        "revenue growth, churn reduction, competitive positioning, and "
        "engineering capacity."
    )

    agents = [
        StyledMockAgent(
            "growth-strategist",
            style="supportive",
            proposal=(
                "Based on the knowledge context, our top priority should be launching "
                "the self-serve SMB tier. Rationale: 1) Q4 revenue grew 23% but SMB "
                "segment only grew 12% vs enterprise at 41% -- the SMB tier addresses "
                "this gap. 2) Churn analysis shows product complexity as the #1 reason "
                "(34%), and a simplified self-serve tier directly targets this. "
                "3) Board approved this as Priority #2, with $2.1M Q1 budget. "
                "4) Competitor Acme charges $500/seat -- our self-serve tier at "
                "$99/seat captures the market they're ignoring."
            ),
        ),
        StyledMockAgent(
            "risk-analyst",
            style="critical",
            proposal=(
                "I challenge the self-serve prioritization. The data shows enterprise "
                "grew 41% vs SMB at 12% -- we should double down on what's working. "
                "Concerns: 1) Self-serve requires significant engineering investment "
                "that diverts from enterprise features. 2) SOC 2 Type II certification "
                "(board Priority #3) is a prerequisite for enterprise deals -- this "
                "should come first. 3) Our 118% net revenue retention proves enterprise "
                "expansion is our strongest growth lever. 4) Prior rate-limiting debate "
                "shows we still have architectural work to support high-volume self-serve."
            ),
        ),
        StyledMockAgent(
            "product-lead",
            style="balanced",
            proposal=(
                "Both perspectives have merit. I recommend a staged approach: "
                "Weeks 1-6: Achieve SOC 2 Type II (unblocks enterprise pipeline and "
                "is a board priority). Weeks 4-12: Parallel-track the self-serve tier "
                "with a lean team (2-3 engineers). This is feasible given our 18% "
                "velocity increase and 12x/day deployment frequency. Key metric: "
                "launch self-serve beta by end of Q1 with 50 pilot customers, while "
                "maintaining enterprise NRR above 115%. The churn data supports both "
                "-- simplified onboarding benefits all segments."
            ),
        ),
    ]

    config = DebateConfig(
        rounds=2,
        consensus_method="majority",
        early_stopping=True,
    )

    arena = Arena(
        question=question,
        agents=agents,
        config=config,
        context=context,  # Inject the knowledge context
    )

    print(f"Question: {question[:80]}...")
    print(f"Agents:   {', '.join(a.name for a in agents)}")
    print(f"Context:  {len(results.items)} knowledge items injected")
    print()

    result = await arena.run()

    # ----------------------------------------------------------------
    # Step 5: Display results
    # ----------------------------------------------------------------
    print("--- Debate Result ---")
    print(f"Status:     {result.status}")
    print(f"Consensus:  {'Reached' if result.consensus_reached else 'Not reached'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Rounds:     {result.rounds_used}")
    print()

    # Show the winning proposal
    print("--- Winning Recommendation ---")
    answer_lines = result.final_answer.split("\n")
    for line in answer_lines:
        print(f"  {line}")
    print()

    # Show dissenting views
    if result.dissenting_views:
        print("--- Dissenting Views ---")
        for view in result.dissenting_views:
            print(f"  {view[:120]}...")
        print()

    # Show the receipt
    if result.receipt:
        print(f"--- Decision Receipt: {result.receipt.receipt_id} ---")
        print(f"  Verdict:    {result.receipt.verdict.value}")
        print(f"  Confidence: {result.receipt.confidence:.0%}")
        print(f"  Agents:     {', '.join(result.receipt.agents)}")
    print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("-" * 64)
    print("Knowledge-enriched debate complete. The agents referenced")
    print("organizational data (revenue, churn, strategy) to ground")
    print("their proposals in evidence rather than generic advice.")
    print()
    print("In production, knowledge comes from KnowledgeMound backed")
    print("by SQLite/PostgreSQL with semantic search via embeddings.")
    print("Cross-debate memory ensures each decision builds on prior ones.")


if __name__ == "__main__":
    asyncio.run(main())
