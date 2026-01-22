---
title: Memory Analytics
description: Memory Analytics
---

# Memory Analytics

Aragora's memory analytics system provides visibility into how the multi-tier memory system is performing, including tier utilization, promotion effectiveness, and learning velocity.

## Overview

The memory analytics module tracks:
- Memory distribution across tiers (Fast, Medium, Slow, Glacial)
- Promotion and demotion patterns
- Retrieval effectiveness
- Learning velocity over time

## Memory Tiers

| Tier | TTL | Purpose | Typical Size |
|------|-----|---------|--------------|
| **Fast** | 1 minute | Immediate context, current debate | 10-50 items |
| **Medium** | 1 hour | Session memory, recent interactions | 100-500 items |
| **Slow** | 1 day | Cross-session learning, patterns | 500-2000 items |
| **Glacial** | 1 week | Long-term insights, stable knowledge | 1000-5000 items |

## API Endpoints

### Get Comprehensive Analytics

```http
GET /api/memory/analytics?days=30
```

**Response:**
```json
{
  "summary": {
    "total_memories": 3250,
    "active_memories": 2100,
    "tier_distribution": {
      "fast": 45,
      "medium": 320,
      "slow": 1200,
      "glacial": 1685
    }
  },
  "promotions": {
    "fast_to_medium": 156,
    "medium_to_slow": 89,
    "slow_to_glacial": 34,
    "promotion_rate": 0.12
  },
  "demotions": {
    "glacial_to_slow": 12,
    "slow_to_medium": 45,
    "demotion_rate": 0.03
  },
  "learning_velocity": {
    "current": 4.5,
    "trend": "increasing",
    "percentile_7d": 0.75
  },
  "retrieval_stats": {
    "avg_latency_ms": 12.3,
    "hit_rate": 0.89,
    "most_retrieved_topics": ["ethics", "technology", "philosophy"]
  },
  "recommendations": [
    {
      "type": "optimization",
      "message": "Consider increasing slow tier capacity - 85% utilized",
      "priority": "medium"
    }
  ]
}
```

### Get Tier-Specific Stats

```http
GET /api/memory/analytics/tier/slow?days=7
```

**Response:**
```json
{
  "tier": "slow",
  "stats": {
    "total_items": 1200,
    "capacity_used": 0.85,
    "avg_age_hours": 18.5,
    "avg_access_count": 3.2,
    "promotions_in": 89,
    "promotions_out": 34,
    "demotions_in": 45,
    "expirations": 23
  },
  "top_topics": [
    {"topic": "ethics", "count": 145, "avg_importance": 0.82},
    {"topic": "technology", "count": 132, "avg_importance": 0.78},
    {"topic": "science", "count": 98, "avg_importance": 0.71}
  ],
  "daily_activity": [
    {"date": "2026-01-09", "additions": 45, "removals": 12},
    {"date": "2026-01-10", "additions": 52, "removals": 18}
  ]
}
```

### Take Manual Snapshot

```http
POST /api/memory/analytics/snapshot
Content-Type: application/json

{
  "label": "pre-optimization",
  "include_samples": true
}
```

**Response:**
```json
{
  "snapshot_id": "snap_abc123",
  "created_at": "2026-01-10T12:00:00Z",
  "tier_counts": {
    "fast": 45,
    "medium": 320,
    "slow": 1200,
    "glacial": 1685
  }
}
```

## Python API

### Basic Analytics

```python
from aragora.memory.tier_analytics import TierAnalyticsTracker

# Initialize tracker
tracker = TierAnalyticsTracker(db_path="memory_analytics.db")

# Get comprehensive analytics
analytics = tracker.get_analytics(days=30)

print(f"Total memories: {analytics.summary.total_memories}")
print(f"Learning velocity: {analytics.learning_velocity.current}")

# Check recommendations
for rec in analytics.recommendations:
    print(f"[{rec.priority}] {rec.message}")
```

### Tier-Specific Analysis

```python
# Get slow tier stats
slow_stats = tracker.get_tier_stats("slow", days=7)

print(f"Slow tier utilization: {slow_stats.capacity_used:.1%}")
print(f"Avg access count: {slow_stats.avg_access_count:.1f}")

# Check top topics
for topic in slow_stats.top_topics[:5]:
    print(f"  {topic.name}: {topic.count} memories")
```

### Snapshot Comparison

```python
# Take snapshots before/after optimization
before = tracker.take_snapshot(label="before")

# ... make optimizations ...

after = tracker.take_snapshot(label="after")

# Compare
comparison = tracker.compare_snapshots(before.id, after.id)
print(f"Memory change: {comparison.total_change:+d}")
print(f"Promotion rate change: {comparison.promotion_rate_change:+.2f}")
```

## Metrics Explained

### Learning Velocity

Learning velocity measures how quickly new knowledge is being consolidated:

```
velocity = (promotions_to_glacial / total_new_memories) * 100
```

Higher velocity indicates effective learning and pattern recognition.

### Promotion Rate

The rate at which memories are promoted to higher tiers:

```
promotion_rate = promotions / (total_memories * time_period)
```

### Hit Rate

Retrieval effectiveness:

```
hit_rate = successful_retrievals / total_retrieval_attempts
```

## Recommendations Engine

The analytics system generates actionable recommendations:

| Type | Trigger | Recommendation |
|------|---------|----------------|
| `capacity` | Tier > 85% full | Increase tier capacity or adjust TTL |
| `velocity` | Learning velocity < 1.0 | Review promotion criteria |
| `retrieval` | Hit rate < 70% | Improve memory indexing |
| `stale` | Avg age > 2x TTL | Trigger cleanup cycle |
| `imbalance` | Tier ratio > 10:1 | Rebalance tier distribution |

## Database Schema

```sql
-- Analytics snapshots
CREATE TABLE memory_snapshots (
    id TEXT PRIMARY KEY,
    label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tier_counts TEXT,  -- JSON
    metadata TEXT      -- JSON
);

-- Daily aggregates
CREATE TABLE memory_daily_stats (
    date TEXT NOT NULL,
    tier TEXT NOT NULL,
    additions INTEGER DEFAULT 0,
    removals INTEGER DEFAULT 0,
    promotions_in INTEGER DEFAULT 0,
    promotions_out INTEGER DEFAULT 0,
    avg_importance REAL,
    avg_access_count REAL,
    PRIMARY KEY (date, tier)
);

-- Retrieval logs (sampled)
CREATE TABLE retrieval_logs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tier TEXT,
    query_hash TEXT,
    latency_ms REAL,
    hit BOOLEAN,
    result_count INTEGER
);
```

## Best Practices

1. **Regular Monitoring**: Check analytics daily to catch issues early

2. **Baseline Snapshots**: Take snapshots before major changes

3. **Tier Balancing**: Aim for roughly 1:10:100:1000 ratio across tiers

4. **Capacity Planning**: Set alerts at 80% capacity

5. **Velocity Tracking**: Monitor learning velocity trends weekly

## Troubleshooting

### Low Hit Rate

```python
# Check retrieval patterns
stats = tracker.get_retrieval_stats(days=7)

if stats.hit_rate < 0.7:
    # Check if queries are too specific
    print(f"Avg result count: {stats.avg_result_count}")

    # Check topic coverage
    for topic in stats.missed_topics[:10]:
        print(f"Frequently missed: \{topic\}")
```

### High Demotion Rate

```python
# Investigate demotions
demotions = tracker.get_demotions(days=7)

for d in demotions[:10]:
    print(f"Demoted: {d.memory_id}")
    print(f"  Reason: {d.reason}")
    print(f"  Access count: {d.access_count}")
```

## See Also

- [Memory Strategy](./memory-strategy) - Memory tier architecture
- [Architecture](./architecture) - System overview
- [API Reference](../api/reference) - Full endpoint documentation
