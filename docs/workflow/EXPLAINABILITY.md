# Decision Explainability

The Explainability module generates human-readable explanations for debate decisions, enabling users to understand **why** and **how** a particular conclusion was reached.

## Overview

When a debate concludes, Aragora can generate a `Decision` entity that captures:
- **Evidence chains**: How evidence supported the conclusion
- **Vote pivots**: Which agent votes were most influential
- **Belief changes**: How agents updated their positions during debate
- **Confidence attribution**: What factors contributed to confidence level
- **Counterfactual analysis**: What would change the outcome

## Quick Start

```python
from aragora import Arena, Environment
from aragora.explainability import ExplanationBuilder, Decision

# Run a debate
arena = Arena(
    environment=Environment(task="Should we adopt microservices?"),
    agents=agents,
    protocol=protocol,
)
result = await arena.run()

# Generate explanation
builder = ExplanationBuilder()
decision = await builder.build(result)

# Get human-readable summary
summary = builder.generate_summary(decision)
print(summary)
```

## Key Components

### ExplanationBuilder

The main entry point for generating explanations.

```python
from aragora.explainability import ExplanationBuilder

# Initialize with optional tracking systems
builder = ExplanationBuilder(
    evidence_tracker=evidence_tracker,     # For grounding scores
    belief_network=belief_network,         # For belief state analysis
    calibration_tracker=calibration_tracker, # For confidence adjustments
    elo_system=elo_system,                 # For agent skill ratings
    provenance_tracker=provenance_tracker, # For claim lineage
)

# Build Decision from debate result
decision = await builder.build(
    result=debate_result,
    context=debate_context,  # Optional
    include_counterfactuals=True,  # Compute "what-if" scenarios
)
```

### Decision Entity

The `Decision` dataclass aggregates all explainability data:

```python
@dataclass
class Decision:
    decision_id: str
    debate_id: str
    timestamp: str
    conclusion: str
    consensus_reached: bool
    confidence: float
    consensus_type: str  # "majority", "unanimous", "supermajority"
    task: str
    domain: str
    rounds_used: int
    agents_participated: List[str]

    # Explainability components
    evidence_chain: List[EvidenceLink]
    vote_pivots: List[VotePivot]
    belief_changes: List[BeliefChange]
    confidence_attribution: List[ConfidenceAttribution]
    counterfactuals: List[Counterfactual]

    # Summary metrics
    evidence_quality_score: float
    agent_agreement_score: float
    belief_stability_score: float
```

## Explainability Components

### Evidence Chains

Track how evidence supported the conclusion:

```python
@dataclass
class EvidenceLink:
    id: str
    content: str
    source: str              # Agent name, user, or external source
    relevance_score: float   # 0-1, how relevant to conclusion
    quality_scores: Dict[str, float]  # accuracy, specificity, etc.
    cited_by: List[str]      # Which agents cited this evidence
    grounding_type: str      # "claim", "fact", "opinion", "citation"
```

**Example output:**
```
Evidence Chain:
1. [Claude] "Microservices enable independent scaling" (relevance: 0.92)
   - Cited by: GPT-4, Gemini
   - Quality: accuracy=0.88, specificity=0.85

2. [GPT-4] "Netflix handles 2B API calls/day with microservices" (relevance: 0.85)
   - Cited by: Claude
   - Grounding: citation (external)
```

### Vote Pivots

Identify which votes most influenced the outcome:

```python
@dataclass
class VotePivot:
    agent: str
    choice: str
    confidence: float
    weight: float           # Computed from ELO + calibration
    reasoning_summary: str
    influence_score: float  # How much this vote affected outcome
    flip_detected: bool     # Did agent change position?
```

**Example output:**
```
Key Votes:
1. Claude: "adopt" (confidence: 0.82, influence: 0.35)
   Reasoning: Strong evidence for scalability benefits

2. GPT-4: "adopt" (confidence: 0.78, influence: 0.30)
   Reasoning: Industry adoption validates approach
   ⚠️ Position flip detected in round 2
```

### Belief Changes

Track how agents updated their positions:

```python
@dataclass
class BeliefChange:
    agent: str
    round: int
    topic: str
    prior_belief: str
    posterior_belief: str
    prior_confidence: float
    posterior_confidence: float
    trigger: str           # What caused the change
    trigger_source: str    # Who provided the trigger
```

**Example output:**
```
Belief Evolution:
- GPT-4 (Round 2):
  Prior: "maintain monolith" (0.65)
  → Posterior: "adopt microservices" (0.78)
  Trigger: "scalability evidence from Claude"
```

### Confidence Attribution

Explain what factors contributed to confidence:

```python
@dataclass
class ConfidenceAttribution:
    factor: str          # consensus_strength, evidence_quality, etc.
    contribution: float  # 0-1, contribution to final confidence
    explanation: str
    raw_value: float     # Underlying metric
```

**Example output:**
```
Confidence Breakdown (0.85 overall):
- Consensus strength: 30% contribution
  "4/5 agents agreed on final answer"

- Evidence quality: 25% contribution
  "High-quality citations with external validation"

- Agent agreement: 20% contribution
  "Low position variance across rounds"
```

### Counterfactual Analysis

Explore "what-if" scenarios:

```python
@dataclass
class Counterfactual:
    id: str
    description: str      # What was changed
    original_outcome: str
    alternative_outcome: str
    confidence_delta: float
    sensitivity: float    # How sensitive outcome is to this change
```

**Example output:**
```
Counterfactuals:
1. "If Claude voted 'reject'..."
   Outcome would change: adopt → reject
   Sensitivity: HIGH (0.85)

2. "If 2 more rounds were added..."
   Outcome unchanged
   Confidence would increase: +0.05
```

## Integration with Debate System

### Automatic Explanation Generation

Enable automatic explanation generation via Arena config:

```python
arena = Arena(
    environment=env,
    agents=agents,
    protocol=protocol,
    enable_explainability=True,  # Auto-generate after debate
)

result = await arena.run()
# result.explanation is automatically populated
```

### With Evidence Tracker

For richer evidence chains, integrate with EvidenceTracker:

```python
from aragora.evidence import EvidenceTracker

evidence_tracker = EvidenceTracker()
builder = ExplanationBuilder(evidence_tracker=evidence_tracker)

# Evidence will include grounding scores and source verification
decision = await builder.build(result)
```

### With Belief Network

For detailed belief change analysis:

```python
from aragora.reasoning import BeliefNetwork

belief_network = BeliefNetwork()
arena = Arena(
    ...,
    belief_network=belief_network,
    enable_belief_tracking=True,
)

builder = ExplanationBuilder(belief_network=belief_network)
decision = await builder.build(result)
# decision.belief_changes populated with detailed transitions
```

## Output Formats

### Human-Readable Summary

```python
summary = builder.generate_summary(decision)
# Returns formatted markdown string
```

### JSON Export

```python
json_data = decision.to_dict()
# Returns serializable dictionary
```

### Structured Report

```python
report = builder.generate_report(decision, format="html")
# Returns HTML report with visualizations
```

## Best Practices

1. **Enable tracking systems**: For richer explanations, use EvidenceTracker, BeliefNetwork, and CalibrationTracker.

2. **Include counterfactuals for important decisions**: They help identify decision sensitivity.

3. **Review confidence attribution**: Understand which factors most influenced the final confidence.

4. **Check for position flips**: Agents changing positions may indicate persuasive arguments or instability.

5. **Validate evidence chains**: Ensure key evidence links are grounded and properly sourced.

## API Reference

| Class | Purpose |
|-------|---------|
| `ExplanationBuilder` | Main builder for Decision entities |
| `Decision` | Aggregate entity for all explainability data |
| `EvidenceLink` | Evidence chain entry |
| `VotePivot` | Influential vote analysis |
| `BeliefChange` | Belief evolution tracking |
| `ConfidenceAttribution` | Confidence factor breakdown |
| `Counterfactual` | What-if scenario analysis |
| `InfluenceType` | Enum for influence types |

## See Also

- [ORCHESTRATION.md](./ORCHESTRATION.md) - Debate execution details
- [REASONING.md](./REASONING.md) - Belief network and provenance
- [CONSENSUS.md](./CONSENSUS.md) - Consensus detection mechanisms
