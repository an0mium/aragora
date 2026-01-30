# aragora.reasoning

Structured reasoning primitives for debates. Provides typed claims with logical
relationships, cryptographic provenance chains for evidence integrity, and
Bayesian belief propagation for probabilistic reasoning over argument graphs.

## Modules

| File | Purpose |
|------|---------|
| `claims.py` | Typed claims, evidence, argument chains, and the `ClaimsKernel` |
| `belief.py` | Bayesian belief network with loopy propagation and centrality analysis |
| `provenance.py` | Cryptographic provenance records, hash chains, and Merkle tree verification |
| `provenance_enhanced.py` | Staleness detection, git/web source tracking, revalidation triggers |
| `crux_detector.py` | Identifies debate-pivotal claims and runs what-if counterfactual analysis |
| `citations.py` | Scholarly citation extraction, grounding verdicts with academic references |
| `reliability.py` | Confidence scoring for claims and evidence based on source quality |
| `evidence_bridge.py` | Connects `aragora.evidence` collection to reasoning/provenance systems |
| `evidence_grounding.py` | Links collected evidence to claims, produces `GroundedVerdict` objects |
| `position_tracker.py` | Tracks agent position shifts and vote pivots across debate rounds |

## Key Concepts

- **TypedClaim / ClaimsKernel** -- Claims have a `ClaimType` (assertion, objection, synthesis, etc.) and are connected by `RelationType` edges (supports, contradicts, refines). The kernel manages the argument graph.
- **ProvenanceChain** -- Immutable, content-addressed hash chain tracking evidence from source through every transformation. Verified via `ProvenanceVerifier`.
- **BeliefNetwork** -- Factor graph where each node holds a probability distribution over truth values. Loopy belief propagation converges to posterior confidence estimates.
- **Crux Detection** -- Finds claims with high influence, high disagreement, and high uncertainty whose resolution would most shift the debate outcome.
- **ReliabilityScorer** -- Evaluates claim confidence from source authority, citation coverage, contradiction signals, and verification status.

## Usage

```python
from aragora.reasoning import (
    ClaimsKernel, ClaimType, RelationType, EvidenceType,
    BeliefNetwork, ProvenanceManager, ReliabilityScorer,
)

# Build an argument graph
kernel = ClaimsKernel()
c1 = kernel.add_claim("Rate limiting prevents abuse", ClaimType.ASSERTION, author="claude")
c2 = kernel.add_claim("Token bucket is best", ClaimType.PROPOSAL, author="gpt")
kernel.add_relation(c2, c1, RelationType.SUPPORTS)

# Track evidence provenance
pm = ProvenanceManager()
record = pm.record_evidence("Cloudflare uses token bucket", source_type="web_search", source_id="https://...")
chain = pm.get_chain(record.id)

# Probabilistic belief propagation
network = BeliefNetwork(kernel=kernel)
network.propagate()
node = network.get_node(c1)
print(f"P(true) = {node.distribution.p_true:.2f}")
```
