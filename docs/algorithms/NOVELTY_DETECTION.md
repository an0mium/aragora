# Novelty Detection

Tracks semantic distance from prior proposals to prevent convergence to mediocrity.

## Overview

When agents propose ideas too similar to what's already been said, novelty scores drop. This signals potential groupthink or hollow consensus, triggering interventions.

```
Novelty Score = 1 - max(similarity to any prior proposal)
```

| Novelty Level | Range | Interpretation |
|---------------|-------|----------------|
| High | > 0.7 | Fresh, divergent ideas |
| Medium | 0.3 - 0.7 | Building on prior ideas |
| Low | < 0.15 | Too similar, may need intervention |

## How It Works

### Per-Proposal Novelty

For each new proposal, compute similarity against ALL prior proposals:

```python
from aragora.debate.novelty import NoveltyTracker

tracker = NoveltyTracker()

# Round 1 - first proposals are maximally novel (1.0)
result1 = tracker.compute_novelty(proposals_round1, round_num=1)
tracker.add_to_history(proposals_round1)

# Round 2 - compared against round 1
result2 = tracker.compute_novelty(proposals_round2, round_num=2)
if result2.has_low_novelty():
    # Trigger trickster intervention
    pass
tracker.add_to_history(proposals_round2)
```

### Similarity Backends

Uses the same backends as convergence detection:

| Backend | Accuracy | Speed | Availability |
|---------|----------|-------|--------------|
| SentenceTransformer | Best | Moderate | Requires `sentence-transformers` |
| TF-IDF | Good | Fast | Always available |
| Jaccard | Basic | Fastest | Always available |

Backend selection is automatic - best available is used.

## Data Structures

### NoveltyScore

Per-agent measurement:

```python
@dataclass
class NoveltyScore:
    agent: str
    round_num: int
    novelty: float           # 1 - max_similarity (0-1)
    max_similarity: float    # Highest similarity to any prior
    most_similar_to: str     # Agent whose proposal was most similar
    prior_proposals_count: int
```

### NoveltyResult

Aggregate round result:

```python
@dataclass
class NoveltyResult:
    round_num: int
    per_agent_novelty: dict[str, float]
    avg_novelty: float
    min_novelty: float
    max_novelty: float
    low_novelty_agents: list[str]
```

## Codebase Novelty Checking

Separate from debate novelty - compares proposals against existing features:

```python
from aragora.debate.novelty import CodebaseNoveltyChecker

checker = CodebaseNoveltyChecker(codebase_context)
result = checker.check_proposal("Add WebSocket streaming", "agent-1")

if not result.is_novel:
    print(f"Warning: {result.warning}")
    # "Proposal may duplicate existing feature: 'WebSocket Support'
    #  (module: aragora.connectors, similarity: 0.72)"
```

Catches agents proposing features that already exist in the codebase.

### Feature Detection

The checker extracts features from context using:
- Table-formatted features (`| Feature | Module |`)
- Bullet-point features (`- Feature: description`)
- Capitalized phrases (e.g., "WebSocket Streaming")

### Synonym Matching

Built-in synonyms for common features:

```python
FEATURE_SYNONYMS = {
    "streaming": ["websocket", "real-time", "live", "push", "sse"],
    "spectator": ["viewer", "read-only", "observer", "watcher"],
    "dashboard": ["panel", "control center", "admin", "monitor"],
    # ...
}
```

## Integration with Trickster

Low novelty triggers trickster interventions:

```
Round → Novelty Check → Low Novelty Detected → Trickster Injection
                              ↓
                    Challenge consensus
                    Introduce edge cases
                    Force position defense
```

## Trajectory Analysis

Track how novelty changes over the debate:

```python
trajectory = tracker.get_agent_novelty_trajectory("claude")
# [1.0, 0.72, 0.45, 0.31]  # Declining novelty

summary = tracker.get_debate_novelty_summary()
# {
#     "overall_avg": 0.62,
#     "overall_min": 0.31,
#     "rounds_with_low_novelty": 1,
#     "total_rounds": 4,
#     "low_novelty_agents_by_round": {4: ["claude", "gpt4"]}
# }
```

## Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `low_novelty_threshold` | 0.15 | Alert threshold |
| Codebase `novelty_threshold` | 0.65 | Feature duplication threshold |

## Implementation Files

| Component | Source |
|-----------|--------|
| NoveltyTracker | `aragora/debate/novelty.py` |
| CodebaseNoveltyChecker | `aragora/debate/novelty.py` |
| Trickster integration | `aragora/debate/trickster.py` |

## Related Documentation

- [Convergence Detection](./CONVERGENCE.md) - Uses same similarity backends
- [Consensus Mechanism](./CONSENSUS.md) - Novelty affects consensus quality
