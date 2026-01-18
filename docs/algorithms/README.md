# Aragora Algorithms

This directory contains detailed documentation for the core algorithms that power Aragora's multi-agent debate system.

## Contents

| Document | Description |
|----------|-------------|
| [CONSENSUS.md](./CONSENSUS.md) | How consensus is determined from agent votes, evidence chains, and dissent tracking |
| [CONVERGENCE.md](./CONVERGENCE.md) | Semantic convergence detection for early debate termination |
| [ELO_CALIBRATION.md](./ELO_CALIBRATION.md) | Agent skill ratings, domain-specific ELO, and calibration scoring |

## Algorithm Overview

### Debate Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Debate Lifecycle                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Start ──► Round 1 ──► Round 2 ──► ... ──► Convergence Check   │
│                                                                 │
│                              │                                  │
│                              ▼                                  │
│                    ┌─────────────────┐                         │
│                    │  Converged?     │                         │
│                    └────────┬────────┘                         │
│                             │                                  │
│              ┌──────────────┼──────────────┐                   │
│              ▼              ▼              ▼                   │
│         converged       refining       diverging               │
│          (≥85%)        (40-85%)         (<40%)                 │
│              │              │              │                   │
│              ▼              │              │                   │
│     Generate Consensus      └──────┬───────┘                   │
│           Proof                    │                           │
│                                    ▼                           │
│                           Continue Debate                      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| System | Primary Metric | Purpose |
|--------|----------------|---------|
| Convergence | Semantic similarity (0-1) | Detect when positions align |
| Consensus | Agreement ratio (0-1) | Measure debate outcome quality |
| ELO | Rating (800-2200 typical) | Track agent skill over time |
| Calibration | Brier score (0-1) | Measure prediction accuracy |

### How They Interact

1. **During Debate**: Convergence detection monitors agent responses each round
2. **At Termination**: Consensus mechanism extracts final claims, votes, and evidence
3. **Post-Debate**: ELO system updates ratings based on match outcome
4. **Over Time**: Calibration scores influence future K-factors and agent selection

## Quick Reference

### Convergence Thresholds

```python
converged   = min_similarity >= 0.85  # Stop debate
refining    = 0.40 <= min_similarity < 0.85  # Continue
diverging   = min_similarity < 0.40  # Positions splitting
```

### Consensus Strength

```python
strong_consensus = (
    consensus_reached and
    agreement_ratio > 0.80 and
    confidence > 0.70
)
```

### ELO Change

```python
# Expected score
E = 1 / (1 + 10^((R_opponent - R_self) / 400))

# New rating
R_new = R_old + K * (actual_score - E)
```

### Calibration Score

```python
brier = (predicted_prob - actual_outcome)^2  # Per prediction
calibration = (1 - avg_brier) * confidence_from_sample_size
```

## Implementation Files

| Algorithm | Primary Source |
|-----------|----------------|
| Consensus | `aragora/debate/consensus.py` |
| Convergence | `aragora/debate/convergence.py` |
| ELO | `aragora/ranking/elo.py` |
| Calibration | `aragora/ranking/calibration_engine.py` |

## See Also

- [Debate Phases](../DEBATE_PHASES.md) - The full debate lifecycle
- [Agent Selection](../AGENT_SELECTION.md) - How agents are chosen for debates
- [Memory Strategy](../MEMORY_STRATEGY.md) - How debate history is persisted
