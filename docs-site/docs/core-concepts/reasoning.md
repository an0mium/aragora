---
title: Reasoning Module
description: Reasoning Module
---

# Reasoning Module

The `aragora/reasoning/` module provides belief networks, claim extraction, provenance tracking, and evidence quality analysis for debates.

## Overview

| File | Size | Description |
|------|------|-------------|
| `belief.py` | 50KB | Belief network construction and propagation |
| `provenance.py` | 24KB | Source tracking and citation chains |
| `provenance_enhanced.py` | 26KB | Enhanced provenance with confidence |
| `claims.py` | 25KB | Claim extraction and categorization |
| `citations.py` | 16KB | Citation management and validation |
| `reliability.py` | 14KB | Source reliability scoring |
| `evidence_grounding.py` | 13KB | Ground claims to evidence |

## Belief Networks

The belief network system (`belief.py`) constructs probabilistic graphs of claims and their relationships.

### Key Classes

```python
from aragora.reasoning.belief import (
    BeliefNetwork,
    BeliefNode,
    BeliefEdge,
    PropagationResult,
)

# Create a belief network
network = BeliefNetwork()

# Add claims as nodes
node = network.add_claim(
    claim="The Earth is round",
    confidence=0.99,
    source="scientific consensus"
)

# Add supporting/contradicting relationships
network.add_edge(
    source_id=node.id,
    target_id=other_node.id,
    relationship="supports",
    strength=0.8
)

# Propagate beliefs through network
result = network.propagate()
```

### Network Analysis

```python
# Find critical claims (high centrality)
load_bearing = network.get_load_bearing_claims(threshold=0.7)

# Identify cruxes (pivotal disagreements)
cruxes = network.identify_cruxes(agent_beliefs)

# Export for visualization
graph_data = network.export_d3_json()
```

## Provenance Tracking

The provenance system (`provenance.py`) tracks the origin and transformation of claims through debates.

### Key Classes

```python
from aragora.reasoning.provenance import (
    ProvenanceTracker,
    ProvenanceChain,
    SourceRecord,
)

# Track claim origin
tracker = ProvenanceTracker()
tracker.record_claim(
    claim_id="claim_001",
    text="Market cap is $1T",
    source="agent:gpt-4",
    evidence_ids=["ev_001", "ev_002"]
)

# Build provenance chain
chain = tracker.build_chain("claim_001")
print(chain.depth)  # How many transformations
print(chain.original_sources)  # Root sources
```

### Enhanced Provenance

The enhanced version (`provenance_enhanced.py`) adds confidence propagation:

```python
from aragora.reasoning.provenance_enhanced import EnhancedProvenanceTracker

tracker = EnhancedProvenanceTracker()
tracker.record_with_confidence(
    claim_id="claim_001",
    confidence=0.9,
    decay_factor=0.95  # Confidence decay per hop
)
```

## Claim Extraction

The claims module (`claims.py`) extracts and categorizes claims from debate text.

### Usage

```python
from aragora.reasoning.claims import (
    ClaimExtractor,
    ClaimType,
    ExtractedClaim,
)

extractor = ClaimExtractor()
claims = extractor.extract(
    text="The policy will reduce costs by 20% and improve efficiency.",
    context={"topic": "budget reform"}
)

for claim in claims:
    print(f"{claim.type}: {claim.text}")
    # QUANTITATIVE: "reduce costs by 20%"
    # QUALITATIVE: "improve efficiency"
```

### Claim Types

- `FACTUAL` - Verifiable facts
- `QUANTITATIVE` - Numerical claims
- `QUALITATIVE` - Quality/opinion claims
- `CAUSAL` - Cause-effect relationships
- `PREDICTIVE` - Future predictions
- `NORMATIVE` - Value judgments

## Citations

The citations module (`citations.py`) validates and manages evidence citations.

```python
from aragora.reasoning.citations import (
    CitationManager,
    Citation,
    validate_citation,
)

manager = CitationManager()

# Add citation
citation = manager.add_citation(
    text="According to Smith (2024)...",
    source_url="https://example.com/paper",
    source_type="academic"
)

# Validate citation format
is_valid = validate_citation(citation)

# Find citations for claim
related = manager.find_citations_for(claim_id)
```

## Reliability Scoring

The reliability module (`reliability.py`) scores source trustworthiness.

```python
from aragora.reasoning.reliability import (
    ReliabilityScorer,
    SourceProfile,
)

scorer = ReliabilityScorer()

# Score a source
score = scorer.score_source(
    source_type="academic",
    domain="nature.com",
    publication_date="2024-01-15"
)

print(f"Reliability: {score.value:.2f}")  # 0.0 - 1.0
print(f"Factors: {score.factors}")
```

### Scoring Factors

- **Source type** - Academic > news > social media
- **Domain reputation** - Known reliable domains score higher
- **Recency** - Recent sources preferred for current topics
- **Citation count** - Well-cited sources score higher
- **Author expertise** - Verified experts score higher

## Evidence Grounding

The evidence grounding module (`evidence_grounding.py`) links claims to supporting evidence.

```python
from aragora.reasoning.evidence_grounding import (
    EvidenceGrounder,
    GroundingResult,
)

grounder = EvidenceGrounder()

# Ground a claim to evidence
result = grounder.ground_claim(
    claim="Climate change is accelerating",
    evidence_pool=evidence_list
)

print(f"Grounding score: {result.score}")
print(f"Supporting evidence: {result.supporting}")
print(f"Contradicting evidence: {result.contradicting}")
```

## API Endpoints

The reasoning module is exposed via the BeliefHandler:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/belief/\{debate_id\}/network` | GET | Get belief network |
| `/api/belief/\{debate_id\}/cruxes` | GET | Identify cruxes |
| `/api/belief/\{debate_id\}/load-bearing` | GET | Get load-bearing claims |
| `/api/belief/\{debate_id\}/provenance/\{claim_id\}` | GET | Get claim provenance |

## See Also

- [DEBATE_INTERNALS.md](../DEBATE_INTERNALS.md) - Core debate architecture
- [EVIDENCE.md](../EVIDENCE.md) - Evidence collection system
- [FORMAL_VERIFICATION.md](../FORMAL_VERIFICATION.md) - Claim verification
