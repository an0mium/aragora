# Explainability - Decision Explanation System

Generate comprehensive explanations for how and why debates reach conclusions, with evidence chains, vote analysis, belief tracking, and counterfactual reasoning.

## Quick Start

```python
from aragora.explainability import ExplanationBuilder, Decision

# Initialize builder with optional integrations
builder = ExplanationBuilder(
    evidence_tracker=tracker,
    calibration_tracker=cal_tracker,
    elo_system=elo,
)

# Build explanation from debate result
decision = await builder.build(debate_result, context)

# Generate human-readable summary
summary = builder.generate_summary(decision)
print(summary)

# Export as JSON
json_str = decision.to_json()
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `ExplanationBuilder` | `builder.py` | Constructs Decision from debate results |
| `Decision` | `decision.py` | Aggregates all explanation data |
| `EvidenceLink` | `decision.py` | Evidence supporting conclusion |
| `VotePivot` | `decision.py` | Influential votes analysis |
| `BeliefChange` | `decision.py` | Agent belief evolution |
| `ConfidenceAttribution` | `decision.py` | Confidence factor breakdown |
| `Counterfactual` | `decision.py` | Sensitivity analysis |

## Architecture

```
explainability/
├── __init__.py        # Public API exports
├── builder.py         # ExplanationBuilder class
└── decision.py        # Decision entity and data classes
```

## Five Explainability Components

### 1. Evidence Chain
Evidence supporting the conclusion with quality scores.

```python
@dataclass
class EvidenceLink:
    id: str
    content: str              # Truncated to 500 chars
    source: str               # Agent or external source
    relevance_score: float    # 0-1 relevance rating
    quality_scores: dict      # semantic_relevance, authority, freshness
    cited_by: list[str]       # Agents who cited this
    grounding_type: str       # "claim", "fact", "opinion", "citation"

# Query top evidence
top_evidence = decision.get_top_evidence(n=5)
```

### 2. Vote Pivots
Votes that significantly influenced outcome.

```python
@dataclass
class VotePivot:
    agent: str
    choice: str
    confidence: float
    weight: float             # ELO + calibration adjusted
    influence_score: float    # 0-1 influence on outcome
    elo_rating: float
    flip_detected: bool       # Position changed during debate

# Get pivotal votes
pivots = decision.get_pivotal_votes(threshold=0.3)
```

### 3. Belief Changes
How agent beliefs evolved during debate.

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
    trigger: str              # "critique", "argument", "evidence"
    trigger_source: str       # What caused the change

# Get significant changes
changes = decision.get_significant_belief_changes(min_delta=0.2)
```

### 4. Confidence Attribution
Attribution of final confidence to different factors.

```python
@dataclass
class ConfidenceAttribution:
    factor: str               # Factor name
    contribution: float       # 0-1, normalized contribution
    explanation: str          # Human explanation
    raw_value: float          # Underlying metric value

# Standard factors:
# - consensus_strength (40%) - Agreement margin
# - evidence_quality (30%) - Average relevance
# - agent_calibration (20%) - Historical accuracy
# - debate_efficiency (10%) - Speed to consensus

factors = decision.get_major_confidence_factors(threshold=0.1)
```

### 5. Counterfactuals
Sensitivity analysis - what would change the outcome.

```python
@dataclass
class Counterfactual:
    condition: str            # What-if condition
    outcome_change: str       # Predicted outcome
    likelihood: float         # 0-1 probability
    sensitivity: float        # 0-1 outcome sensitivity
    affected_agents: list[str]

# Examples generated:
# - "If claude's vote removed" → sensitivity: 0.8
# - "If evidence X removed" → sensitivity: 0.5
# - "With fewer agents" → sensitivity: 0.3

counterfactuals = decision.get_high_sensitivity_counterfactuals(threshold=0.5)
```

## Decision Entity

```python
@dataclass
class Decision:
    # Identity
    decision_id: str
    debate_id: str
    timestamp: datetime

    # Outcome
    conclusion: str
    consensus_reached: bool
    confidence: float
    consensus_type: str

    # Context
    task: str
    domain: str
    rounds_used: int
    agents_participated: list[str]

    # Explainability Components
    evidence_chain: list[EvidenceLink]
    vote_pivots: list[VotePivot]
    belief_changes: list[BeliefChange]
    confidence_attribution: list[ConfidenceAttribution]
    counterfactuals: list[Counterfactual]

    # Summary Metrics
    evidence_quality_score: float
    agent_agreement_score: float
    belief_stability_score: float

    # Serialization
    def to_dict(self) -> dict
    def to_json(self, indent=2) -> str
    @classmethod
    def from_dict(cls, data: dict) -> Decision
```

## Summary Generation

```python
summary = builder.generate_summary(decision)

# Output format (markdown):
# ## Decision Summary
#
# **Conclusion:** Use JWT for authentication
# **Consensus:** Reached (3/4 agents agreed)
# **Confidence:** 0.85
# **Rounds:** 3
#
# ### Key Evidence
# 1. JWT is stateless and scalable (relevance: 0.92)
# 2. Industry standard for APIs (relevance: 0.88)
#
# ### Pivotal Votes
# - claude: JWT (influence: 0.45, ELO: 1523)
# - gpt4: JWT (influence: 0.35, ELO: 1498)
#
# ### Confidence Factors
# - Consensus strength: 40% contribution
# - Evidence quality: 30% contribution
```

## HTTP API

```
GET  /api/v1/debates/{id}/explanation     # Full decision
GET  /api/v1/debates/{id}/evidence        # Evidence chain
GET  /api/v1/debates/{id}/votes/pivots    # Vote analysis
GET  /api/v1/debates/{id}/counterfactuals # Counterfactuals
GET  /api/v1/debates/{id}/summary         # Human summary
GET  /api/v1/explain/{debate_id}          # Shortcut

POST /api/v1/explainability/batch         # Batch processing
POST /api/v1/explainability/compare       # Compare decisions
```

## Optional Integrations

ExplanationBuilder enhances explanations when these systems are available:

```python
builder = ExplanationBuilder(
    evidence_tracker=tracker,       # Quality scores for evidence
    belief_network=network,         # Belief change tracking
    calibration_tracker=cal,        # Historical accuracy
    elo_system=elo,                 # Agent skill ratings
    provenance_tracker=provenance,  # Claim lineage
)
```

## Influence Types

```python
class InfluenceType(str, Enum):
    EVIDENCE = "evidence"
    VOTE = "vote"
    ARGUMENT = "argument"
    CALIBRATION = "calibration"
    ELO = "elo"
    CONSENSUS = "consensus"
    USER = "user"
```

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Debate](../debate/README.md) - Debate orchestration
- [Reasoning](../reasoning/README.md) - Belief and provenance tracking
- [Verification](../verification/README.md) - Formal proof verification
