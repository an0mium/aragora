# Position Flip Detection

Detects when agents reverse their positions on claims, tracks flip patterns, and computes consistency scores.

## Overview

Position flips occur when an agent takes a stance that contradicts their prior position. Tracking flips helps identify:
- Agents susceptible to persuasion
- Claims where evidence is genuinely ambiguous
- Potential inconsistency issues

## Flip Types

| Type | Description | Severity |
|------|-------------|----------|
| **Contradiction** | Direct opposite (e.g., "X is good" → "X is bad") | Highest (1.0) |
| **Retraction** | Complete withdrawal ("I was wrong about X") | High (0.7) |
| **Qualification** | Adding nuance ("X is sometimes true") | Medium (0.3) |
| **Refinement** | Minor adjustment ("X is mostly true") | Low (0.1) |

## How It Works

### Detection Algorithm

```python
from aragora.insights.flip_detector import FlipDetector

detector = FlipDetector()

# Detect flips for an agent
flips = detector.detect_flips_for_agent("claude", lookback_positions=50)

for flip in flips:
    print(f"{flip.flip_type}: {flip.original_claim[:50]}...")
    print(f"  → {flip.new_claim[:50]}...")
    print(f"  Similarity: {flip.similarity_score:.0%}")
```

### Classification Logic

```python
def _classify_flip_type(original, new, orig_conf, new_conf):
    # Check contradiction signals
    contradiction_signals = [
        ("not ", " not "),
        ("isn't", "is"),
        ("bad", "good"),
        ("false", "true"),
        ("disagree", "agree"),
    ]

    # Check retraction signals
    retraction_signals = ["was wrong", "reconsider", "take back", "withdraw"]

    # Check qualification signals
    qualification_signals = ["sometimes", "partially", "in some cases"]

    # Default based on similarity
    if similarity > 0.7:
        return "refinement"
    elif similarity < 0.3:
        return "contradiction"
    else:
        return "qualification"
```

## Consistency Scoring

### Per-Agent Score

```python
score = detector.get_agent_consistency("claude")

print(f"Consistency: {score.consistency_score:.0%}")  # 1.0 = perfectly consistent
print(f"Flip rate: {score.flip_rate:.0%}")
print(f"Contradictions: {score.contradictions}")
print(f"Problem domains: {score.domains_with_flips}")
```

### Score Formula

```python
@property
def consistency_score(self) -> float:
    if self.total_positions == 0:
        return 1.0

    # Weight contradictions more heavily
    weighted_flips = (
        self.contradictions * 1.0 +
        self.retractions * 0.7 +
        self.qualifications * 0.3 +
        self.refinements * 0.1
    )
    return max(0.0, 1.0 - (weighted_flips / self.total_positions))
```

## Data Structures

### FlipEvent

```python
@dataclass
class FlipEvent:
    id: str
    agent_name: str
    original_claim: str
    new_claim: str
    original_confidence: float
    new_confidence: float
    original_debate_id: str
    new_debate_id: str
    similarity_score: float  # High = contradiction likely
    flip_type: str           # contradiction, refinement, retraction, qualification
    domain: str | None
    detected_at: str
```

### AgentConsistencyScore

```python
@dataclass
class AgentConsistencyScore:
    agent_name: str
    total_positions: int
    total_flips: int
    contradictions: int
    refinements: int
    retractions: int
    qualifications: int
    avg_confidence_on_flip: float
    domains_with_flips: list[str]

    # Computed properties
    consistency_score: float  # 0.0-1.0
    flip_rate: float          # flips / positions
```

## Batch Operations

For efficiency with multiple agents:

```python
# Avoid N+1 queries
scores = detector.get_agents_consistency_batch(["claude", "gpt4", "gemini"])
for agent, score in scores.items():
    print(f"{agent}: {score.consistency_score:.0%}")
```

## Dashboard Summary

```python
summary = detector.get_flip_summary()
# {
#     "total_flips": 47,
#     "by_type": {"contradiction": 12, "refinement": 25, ...},
#     "by_agent": {"claude": 15, "gpt4": 20, ...},
#     "recent_24h": 5
# }
```

## UI Formatting

```python
from aragora.insights.flip_detector import format_flip_for_ui

ui_flip = format_flip_for_ui(flip)
# {
#     "id": "flip-123",
#     "agent": "claude",
#     "type": "contradiction",
#     "type_emoji": "🔄",
#     "before": {"claim": "X is good...", "confidence": "85%"},
#     "after": {"claim": "X is bad...", "confidence": "70%"},
#     "similarity": "65%",
#     "timestamp": "2026-01-30T..."
# }
```

## Knowledge Mound Integration

Flips are automatically synced to Knowledge Mound for cross-debate learning:

```python
detector = FlipDetector(km_adapter=insights_adapter)

# Flips are stored locally AND synced to KM
flips = detector.detect_flips_for_agent("claude")
# Each flip triggers: km_adapter.store_flip(flip)
```

## Database Schema

```sql
CREATE TABLE detected_flips (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    original_claim TEXT NOT NULL,
    new_claim TEXT NOT NULL,
    original_confidence REAL,
    new_confidence REAL,
    original_debate_id TEXT,
    new_debate_id TEXT,
    similarity_score REAL,
    flip_type TEXT,
    domain TEXT,
    detected_at TEXT
);

CREATE INDEX idx_flips_agent ON detected_flips(agent_name);
CREATE INDEX idx_flips_type ON detected_flips(flip_type);
```

## Implementation Files

| Component | Source |
|-----------|--------|
| FlipDetector | `aragora/insights/flip_detector.py` |
| Position Ledger | `aragora/insights/position_ledger.py` |
| KM Integration | `aragora/knowledge/mound/adapters/insights_adapter.py` |

## Related Documentation

- [ELO Calibration](./ELO_CALIBRATION.md) - Consistency affects agent ratings
- [Belief Network](./BELIEF_NETWORK.md) - Flips can trigger belief re-propagation
