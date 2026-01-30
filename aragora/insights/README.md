# Insights

Extract, aggregate, and learn from debate outcomes.

## Overview

This package provides tools for extracting structured learnings from debates:

| Component | Module | Purpose |
|-----------|--------|---------|
| Extractor | `extractor.py` | Extract insights from debates |
| Store | `store.py` | SQLite persistence |
| Postgres Store | `postgres_store.py` | PostgreSQL with caching |
| Flip Detector | `flip_detector.py` | Track position changes |
| Aggregator | `aggregator.py` | Cross-debate patterns |

## Quick Start

```python
from aragora.insights import (
    InsightExtractor,
    InsightStore,
    DebateInsights,
)

# Extract insights from completed debate
extractor = InsightExtractor()
insights = await extractor.extract(debate_result)

# Store for later analysis
store = InsightStore("insights.db")
await store.save(insights)

# Query patterns
patterns = await store.get_patterns(
    insight_type=InsightType.WINNING_ARGUMENT,
    min_confidence=0.8,
)
```

## Insight Types

```python
from aragora.insights import InsightType

InsightType.WINNING_ARGUMENT    # Arguments that led to consensus
InsightType.COMMON_GROUND       # Shared positions across agents
InsightType.CONTROVERSY_POINT   # High-disagreement areas
InsightType.EXPERTISE_SIGNAL    # Agent domain strengths
InsightType.META_LEARNING       # Debate process improvements
InsightType.FLIP_EVENT          # Position reversals
```

## Extracting Insights

```python
from aragora.insights import InsightExtractor, DebateInsights

extractor = InsightExtractor(
    min_confidence=0.6,
    extract_agent_performance=True,
    detect_patterns=True,
)

# From debate result
insights: DebateInsights = await extractor.extract(result)

print(f"Debate: {insights.debate_id}")
print(f"Consensus: {insights.consensus_reached}")
print(f"Key takeaway: {insights.key_takeaway}")

for insight in insights.all_insights():
    print(f"  [{insight.type}] {insight.title}")
    print(f"    Confidence: {insight.confidence:.2f}")
```

## Flip Detection

Track when agents change positions:

```python
from aragora.insights import FlipDetector, FlipEvent

detector = FlipDetector(
    similarity_threshold=0.8,  # Consider similar if >80%
    min_confidence_change=0.2,  # Significant if >20% change
)

# Analyze agent history
flips = await detector.detect_flips(
    agent_name="claude",
    debate_history=recent_debates,
)

for flip in flips:
    print(f"Agent: {flip.agent_name}")
    print(f"  Original: {flip.original_claim}")
    print(f"  New: {flip.new_claim}")
    print(f"  Confidence change: {flip.original_confidence} → {flip.new_confidence}")
```

## Agent Consistency Scoring

```python
from aragora.insights import FlipDetector, format_consistency_for_ui

detector = FlipDetector()

# Get consistency scores
scores = await detector.calculate_consistency_scores(
    agent_names=["claude", "gpt-4", "gemini"],
    time_window_days=30,
)

for agent, score in scores.items():
    print(f"{agent}: {score.consistency_score:.2f}")
    print(f"  Flips: {score.flip_count}")
    print(f"  Debates: {score.debate_count}")

# Format for UI display
ui_data = format_consistency_for_ui(scores)
```

## PostgreSQL Store

Production-ready storage with caching:

```python
from aragora.insights.postgres_store import PostgresInsightStore

store = PostgresInsightStore(
    connection_string="postgresql://...",
    cache_ttl=300,  # 5 minute cache
)

# Store debate insights (batch operation)
count = await store.store_debate_insights(insights)

# Query with caching
recent = await store.get_recent_insights(limit=20)
agent_stats = await store.get_agent_stats("claude")

# Search
results = await store.search(
    query="consensus building",
    insight_type=InsightType.WINNING_ARGUMENT,
    agent="claude",
    limit=10,
)
```

## Integration with Knowledge Mound

Insights sync to Knowledge Mound:

```python
from aragora.insights.postgres_store import PostgresInsightStore
from aragora.knowledge.mound.adapters import InsightsAdapter

# Configure bidirectional sync
store = PostgresInsightStore(connection_string)
km_adapter = InsightsAdapter(knowledge_mound)
store.set_km_adapter(km_adapter)

# Insights automatically sync to KM on store
await store.store_debate_insights(insights)
```
