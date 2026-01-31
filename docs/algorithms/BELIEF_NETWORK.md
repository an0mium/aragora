# Bayesian Belief Network

Probabilistic graphical model for nuanced reasoning over debate claims.

## Overview

The Belief Network extends the Claims Kernel with:
- **BeliefNode**: Claims with probability distributions over truth values
- **BeliefNetwork**: Factor graph for message passing
- **Loopy belief propagation**: For cyclic argument graphs
- **Centrality analysis**: Identifying load-bearing claims
- **Crux detection**: Finding debate-pivotal claims

Moves Aragora from binary accept/reject to nuanced probabilistic reasoning.

## Core Concepts

### Belief Distribution

Probability over claim truth values:

```python
@dataclass
class BeliefDistribution:
    p_true: float = 0.5
    p_false: float = 0.5
    p_unknown: float = 0.0  # Undecidable probability mass

    @property
    def entropy(self) -> float:
        """Shannon entropy - high = uncertain."""

    @property
    def confidence(self) -> float:
        """Max probability."""
        return max(self.p_true, self.p_false, self.p_unknown)

    @property
    def expected_truth(self) -> float:
        """E[truth]: true=1, false=0, unknown=0.5."""
        return self.p_true + 0.5 * self.p_unknown
```

### Belief Node

A claim wrapped with probabilistic state:

```python
@dataclass
class BeliefNode:
    node_id: str
    claim_id: str
    claim_statement: str
    author: str

    prior: BeliefDistribution      # Initial belief
    posterior: BeliefDistribution  # After propagation
    status: BeliefStatus           # PRIOR, UPDATED, CONVERGED, CONTESTED

    centrality: float              # How important (0-1)
    parent_ids: list[str]          # Claims this depends on
    child_ids: list[str]           # Claims depending on this
```

### Factor

Encodes relationships between claims:

```python
@dataclass
class Factor:
    factor_id: str
    relation_type: RelationType  # SUPPORTS, CONTRADICTS, DEPENDS_ON
    source_node_id: str
    target_node_id: str
    strength: float = 1.0
```

Factor potentials for message passing:

| Relation | Source True | Target True | Potential |
|----------|-------------|-------------|-----------|
| SUPPORTS | True | True | 0.7 + 0.3×strength |
| SUPPORTS | True | False | 0.3 - 0.2×strength |
| CONTRADICTS | True | True | 0.2 - 0.15×strength |
| CONTRADICTS | True | False | 0.8 + 0.15×strength |
| DEPENDS_ON | False | True | 0.1 (unlikely) |

## Usage

### Building a Network

```python
from aragora.reasoning.belief import BeliefNetwork

network = BeliefNetwork(debate_id="debate-001")

# Add claims
node1 = network.add_claim(
    claim_id="c1",
    statement="AI improves productivity",
    author="claude",
    initial_confidence=0.8
)

node2 = network.add_claim(
    claim_id="c2",
    statement="Productivity gains benefit workers",
    author="gpt4",
    initial_confidence=0.6
)

# Add relationship
network.add_factor(
    source_claim_id="c1",
    target_claim_id="c2",
    relation_type=RelationType.SUPPORTS,
    strength=0.9
)
```

### From Claims Kernel

```python
from aragora.reasoning.claims import ClaimsKernel

kernel = ClaimsKernel()
# ... add claims and relations ...

network = BeliefNetwork().from_claims_kernel(kernel)
```

### Running Propagation

```python
result = network.propagate()

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Max change: {result.max_change:.4f}")

for node_id, posterior in result.node_posteriors.items():
    print(f"{node_id}: P(true)={posterior.p_true:.2f}")
```

## Belief Propagation Algorithm

### Sum-Product Message Passing

1. Initialize all messages to uniform
2. Iterate until convergence:
   - Send messages along all factors
   - Update posteriors from prior × incoming messages
   - Check max change against threshold
3. Mark nodes as CONVERGED or UPDATED

### Damping

For numerical stability on cyclic graphs:

```python
# Damped message update
msg_new = damping * msg_old + (1 - damping) * msg_computed
```

Default damping: 0.5

### Convergence

```python
max_iterations = 100           # BELIEF_MAX_ITERATIONS
convergence_threshold = 0.001  # BELIEF_CONVERGENCE_THRESHOLD
```

## Analysis Methods

### Most Uncertain Claims

```python
uncertain = network.get_most_uncertain_claims(limit=5)
for node, entropy in uncertain:
    print(f"{node.claim_statement[:50]}: entropy={entropy:.2f}")
```

### Load-Bearing Claims

Claims that affect many others (high centrality):

```python
load_bearing = network.get_load_bearing_claims(limit=5)
for node, centrality in load_bearing:
    print(f"{node.claim_statement[:50]}: centrality={centrality:.3f}")
```

### Contested Claims

Claims with contradictory incoming messages:

```python
contested = network.get_contested_claims()
# Nodes where incoming messages disagree by > 0.3
```

### Conditional Probability

```python
# P(query | evidence)
posterior = network.conditional_probability(
    query_claim_id="c2",
    evidence={"c1": True}  # Assume c1 is true
)
print(f"P(c2=true | c1=true) = {posterior.p_true:.2f}")
```

### Sensitivity Analysis

```python
sensitivities = network.sensitivity_analysis(target_claim_id="c5")
# {
#     "c1": 0.42,  # Changing c1 shifts c5 by 0.42
#     "c2": 0.15,
#     "c3": 0.08,
# }
```

## Centrality Computation

PageRank-like algorithm:

```python
def _compute_centralities(self):
    # Initialize uniform
    centralities = {nid: 1/n for nid in nodes}

    # Iterate
    for _ in range(20):
        for node_id, node in nodes.items():
            rank = (1 - damping) / n
            for child_id in node.child_ids:
                # Weight by entropy (uncertain = important)
                weight = child.posterior.entropy
                rank += damping * centralities[child_id] * weight / len(child.parent_ids)

    # Normalize
    return {k: v/sum(values) for k, v in centralities.items()}
```

## Knowledge Mound Integration

### Seeding from KM

```python
network = BeliefNetwork(km_adapter=belief_adapter)

# Seed with prior beliefs from past debates
seeded_count = network.seed_from_km(
    topic="AI productivity",
    min_confidence=0.7
)
print(f"Seeded {seeded_count} beliefs from Knowledge Mound")
```

### Sync to KM

High-confidence converged beliefs are automatically stored:

```python
# After propagation, beliefs with confidence >= 0.8 are synced
result = network.propagate()
# Triggers: km_adapter.store_belief(...) for each high-confidence node
```

## Crux Detection

Identifies claims that would change the debate outcome if flipped:

```python
from aragora.reasoning.belief import CruxDetector

detector = CruxDetector(network)
cruxes = detector.identify_cruxes(
    target_claim_id="final_conclusion",
    threshold=0.3
)

for crux in cruxes:
    print(f"Crux: {crux.statement}")
    print(f"  Impact: {crux.impact_score:.2f}")
    print(f"  Current: P(true)={crux.current_belief:.2f}")
```

## Serialization

```python
# To dict/JSON
data = network.to_dict()
json_str = network.to_json(indent=2)

# From dict
network = BeliefNetwork.from_dict(data)
```

## Summary Generation

```python
summary = network.generate_summary()
# Returns markdown with:
# - Most certain claims
# - Most uncertain claims
# - Load-bearing claims (high centrality)
```

## Implementation Files

| Component | Source |
|-----------|--------|
| BeliefNetwork | `aragora/reasoning/belief.py` |
| CruxDetector | `aragora/reasoning/crux_detector.py` |
| Claims Kernel | `aragora/reasoning/claims.py` |
| KM Adapter | `aragora/knowledge/mound/adapters/belief_adapter.py` |

## Related Documentation

- [Consensus Mechanism](./CONSENSUS.md) - Belief states inform consensus
- [Convergence Detection](./CONVERGENCE.md) - Semantic convergence vs belief convergence
