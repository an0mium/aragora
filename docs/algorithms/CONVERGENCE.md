# Convergence Detection

This document describes how Aragora detects when agent positions have converged, enabling early termination of debates.

## Overview

Convergence detection identifies when further debate rounds would provide diminishing returns. It uses **semantic similarity** between consecutive rounds combined with **advanced metrics** for nuanced analysis.

## Similarity Backends

The system uses a 3-tier fallback for computing text similarity:

| Backend | Accuracy | Dependency | Speed |
|---------|----------|------------|-------|
| SentenceTransformer | Best | `sentence-transformers` | Slower |
| TF-IDF | Good | `scikit-learn` | Fast |
| Jaccard | Basic | None | Fastest |

### Backend Selection

Automatic selection (best available):
```python
detector = ConvergenceDetector()  # Auto-selects
```

Manual override via environment:
```bash
export ARAGORA_CONVERGENCE_BACKEND=tfidf  # or: jaccard, sentence
```

### SentenceTransformer Backend

Uses `all-MiniLM-L6-v2` model for semantic embeddings:
- Captures meaning, not just word overlap
- Handles paraphrasing and synonyms
- Caches embeddings per debate (scoped by `debate_id`)

### TF-IDF Backend

Uses term frequency-inverse document frequency:
- Fast for medium-length texts
- Good for technical content with consistent terminology
- No GPU required

### Jaccard Backend

Simple token overlap:
```python
similarity = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
```
- Always available (zero dependencies)
- Works well for short, keyword-dense text

## Convergence Thresholds

| Status | Min Similarity | Interpretation |
|--------|----------------|----------------|
| `converged` | >= 85% | Agents have reached agreement |
| `refining` | 40-85% | Still improving positions |
| `diverging` | < 40% | Positions are splitting |

### Consecutive Rounds

Convergence requires `consecutive_rounds_needed` stable rounds (default: 1).

## Convergence Result

```python
@dataclass
class ConvergenceResult:
    converged: bool                    # Should debate terminate?
    status: str                        # "converged", "refining", "diverging"
    min_similarity: float              # Lowest agent similarity
    avg_similarity: float              # Average across agents
    per_agent_similarity: dict         # Per-agent breakdown
    consecutive_stable_rounds: int     # Stability counter
```

## Algorithm

For each round (after `min_rounds_before_check`):

1. **Get common agents** between current and previous round
2. **Compute pairwise similarity** for each agent's responses
3. **Calculate min and avg** similarity scores
4. **Determine status**:
   - If `min_similarity >= convergence_threshold`: increment stable counter
   - If stable counter >= `consecutive_rounds_needed`: `converged = True`
   - If `min_similarity < divergence_threshold`: reset counter, `diverging`
   - Otherwise: reset counter, `refining`

```python
def check_convergence(current, previous, round_num):
    if round_num <= min_rounds_before_check:
        return None

    common_agents = current.keys() & previous.keys()
    similarities = {
        agent: backend.compute_similarity(current[agent], previous[agent])
        for agent in common_agents
    }

    min_sim = min(similarities.values())

    if min_sim >= 0.85:
        consecutive_stable_count += 1
        if consecutive_stable_count >= consecutive_rounds_needed:
            return ConvergenceResult(converged=True, status="converged", ...)
    elif min_sim < 0.40:
        consecutive_stable_count = 0
        return ConvergenceResult(converged=False, status="diverging", ...)
    else:
        consecutive_stable_count = 0
        return ConvergenceResult(converged=False, status="refining", ...)
```

## Advanced Metrics

For deeper analysis, `AdvancedConvergenceAnalyzer` computes additional signals:

### Argument Diversity

Measures how different the arguments are across agents:

```python
@dataclass
class ArgumentDiversityMetric:
    unique_arguments: int     # Arguments with <70% similarity to others
    total_arguments: int
    diversity_score: float    # unique / total (0-1)
```

**Interpretation**:
- High diversity (>0.7): Agents exploring different points
- Low diversity (<0.3): Arguments converging

### Evidence Convergence

Measures citation overlap between agents:

```python
@dataclass
class EvidenceConvergenceMetric:
    shared_citations: int     # Citations used by 2+ agents
    total_citations: int
    overlap_score: float      # shared / total (0-1)
```

**Interpretation**:
- High overlap (>0.6): Agents citing same sources (strong agreement)
- Low overlap: Different evidence bases

### Stance Volatility

Tracks how often agents change positions:

```python
@dataclass
class StanceVolatilityMetric:
    stance_changes: int       # Position flips across rounds
    total_responses: int
    volatility_score: float   # changes / responses (0-1)
```

**Stances detected**: `support`, `oppose`, `neutral`, `mixed`

**Interpretation**:
- Low volatility (<0.2): Stable positions
- High volatility: Ongoing disagreement

## Overall Convergence Score

The weighted combination:

```python
def compute_overall_score():
    weights = {
        "semantic": 0.4,
        "diversity": 0.2,
        "evidence": 0.2,
        "stability": 0.2
    }

    score = semantic_similarity * 0.4
    score += (1 - diversity_score) * 0.2      # Lower diversity = more converged
    score += evidence_overlap_score * 0.2
    score += (1 - volatility_score) * 0.2     # Lower volatility = more converged

    return clamp(score, 0, 1)
```

## Configuration

```python
detector = ConvergenceDetector(
    convergence_threshold=0.85,    # When to consider converged
    divergence_threshold=0.40,     # When to consider diverging
    min_rounds_before_check=1,     # Don't check too early
    consecutive_rounds_needed=1,   # Stable rounds required
    debate_id="debate-123"         # For scoped embedding cache
)
```

## Embedding Cache

For performance, embeddings are cached per debate:

```python
# Scoped cache (recommended)
cache = get_scoped_embedding_cache(debate_id)

# Cleanup after debate
cleanup_embedding_cache(debate_id)

# Global reset (for testing)
reset_embedding_cache()
```

## Usage Example

```python
from aragora.debate.convergence import ConvergenceDetector

detector = ConvergenceDetector(debate_id="debate-001")

# After each round
result = detector.check_convergence(
    current_responses={"claude": "...", "gpt4": "..."},
    previous_responses={"claude": "...", "gpt4": "..."},
    round_number=3
)

if result and result.converged:
    print(f"Debate converged! Avg similarity: {result.avg_similarity:.2%}")
```

## Related Documentation

- [Consensus Mechanism](./CONSENSUS.md) - How final consensus is determined
- [ELO Calibration](./ELO_CALIBRATION.md) - Agent skill tracking
- [Debate Phases](../DEBATE_PHASES.md) - The debate lifecycle
