# Advanced Features Guide

This guide covers advanced capabilities in Aragora for power users who need to leverage sophisticated patterns beyond basic debate orchestration.

## Table of Contents

1. [RLM (Recursive Language Models)](#rlm-recursive-language-models)
2. [Belief Network](#belief-network)
3. [Explainability](#explainability)
4. [Matrix/Graph Debates](#matrixgraph-debates)
5. [Workflow Templates](#workflow-templates)

---

## RLM (Recursive Language Models)

RLM enables models to programmatically examine large contexts through a REPL interface rather than stuffing everything into the prompt.

### Overview

Based on the ["Recursive Language Models" paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601), RLM fundamentally changes how LLMs interact with long contexts:

| Approach | How It Works | When to Use |
|----------|--------------|-------------|
| **True RLM** | Model writes code to query/grep/partition context via REPL | Primary - when `rlm` package installed |
| **Compression Fallback** | Pre-processing hierarchical summarization | Fallback - when `rlm` unavailable |

**Key Insight**: Instead of feeding long context directly into neural networks, context is stored in a REPL environment where the LLM can programmatically examine, search, and recursively query it.

### Installation

```bash
# Install Aragora with RLM support
pip install aragora[rlm]

# Or install the official RLM library directly
pip install rlm
```

### Basic Usage

```python
from aragora.rlm import AragoraRLM, DebateContextAdapter, HAS_OFFICIAL_RLM

# Check if official RLM is available
if HAS_OFFICIAL_RLM:
    # Create RLM instance with official library (TRUE RLM)
    rlm = AragoraRLM(backend="openai", model="gpt-4o")
    adapter = DebateContextAdapter(rlm)

    # Query a debate with REPL-based context access
    answer = await adapter.query_debate(
        "What were the main disagreements?",
        debate_result
    )
else:
    # Fallback to hierarchical compression
    from aragora.rlm import HierarchicalCompressor
    compressor = HierarchicalCompressor()
    result = await compressor.compress(content, source_type="debate")
```

### Bridge and Handler Configuration

The `AragoraRLM` class serves as the primary bridge between the official RLM library and Aragora:

```python
from aragora.rlm import AragoraRLM, RLMBackendConfig, RLMConfig

# Configure the backend
backend_config = RLMBackendConfig(
    backend="openai",           # openai, anthropic, openrouter, litellm
    model_name="gpt-4o",
    sub_model_name="gpt-4o-mini",  # Cheaper model for sub-calls
    environment_type="local",   # local, docker, modal
    environment_timeout=120,
    max_depth=1,                # Maximum recursion depth
    max_iterations=30,          # Maximum iterations per execution
    verbose=False,
)

# Configure Aragora-specific settings
aragora_config = RLMConfig(
    target_tokens=4000,         # Target size per level
    overlap_tokens=200,         # Overlap between chunks
    compression_ratio=0.3,      # Target compression per level
    preserve_structure=True,    # Maintain document structure
    cache_compressions=True,    # Cache results
    cache_ttl_seconds=3600,     # Cache TTL (1 hour)
)

# Create the RLM instance
rlm = AragoraRLM(
    backend_config=backend_config,
    aragora_config=aragora_config,
    enable_caching=True,
)
```

### Abstraction Levels

RLM maintains content at multiple abstraction levels for efficient navigation:

| Level | Description | Compression |
|-------|-------------|-------------|
| `FULL` | Original full content | 0% |
| `DETAILED` | Detailed summary | ~50% |
| `SUMMARY` | Key points | ~80% |
| `ABSTRACT` | High-level overview | ~95% |
| `METADATA` | Tags and routing info | ~99% |

### Decomposition Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `PEEK` | Inspect initial sections | Understanding structure |
| `GREP` | Keyword/regex search | Finding specific mentions |
| `PARTITION_MAP` | Chunk and recurse | Large documents |
| `SUMMARIZE` | Extract key points | Getting overview |
| `HIERARCHICAL` | Navigate abstraction tree | Multi-level queries |
| `AUTO` | Let RLM decide | General use |

### Querying Debates

```python
from aragora.rlm import DebateContextAdapter, create_aragora_rlm

# Create RLM for debate analysis
rlm = create_aragora_rlm(backend="anthropic", model="claude-3-5-sonnet-20241022")
adapter = DebateContextAdapter(rlm)

# Load a debate result
debate = await arena.run()

# Query the debate history
answer = await adapter.query_debate(
    "What arguments changed during the debate?",
    debate
)

# Get specific information
claims = await adapter.query_debate(
    "List all verifiable claims made by the claude agent",
    debate
)
```

### Knowledge Mound Integration

```python
from aragora.rlm import KnowledgeMoundAdapter

adapter = KnowledgeMoundAdapter(rlm, knowledge_mound)

# Recursive query with abstraction navigation
result = await adapter.query(
    "What are the security best practices mentioned?",
    start_level=AbstractionLevel.SUMMARY,
    max_depth=2,
)
```

### Iterative Refinement

```python
# Query with iterative refinement for complex questions
result = await rlm.query_with_refinement(
    query="Analyze the trade-offs discussed",
    context=context,
    max_iterations=3,
    start_level="SUMMARY",
)

# Check refinement history
for i, answer in enumerate(result.refinement_history):
    print(f"Iteration {i}: {answer[:100]}...")
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARAGORA_RLM_BACKEND` | Override backend (openai, anthropic, openrouter) |
| `ARAGORA_RLM_MODEL` | Override model name |
| `ARAGORA_RLM_FALLBACK_BACKEND` | Fallback backend on error |
| `ARAGORA_RLM_FALLBACK_MODEL` | Fallback model name |
| `ARAGORA_RLM_CONTEXT_DIR` | Directory for externalized context files |

---

## Belief Network

The Belief Network provides probabilistic graphical model capabilities for tracking claim provenance, confidence propagation, and identifying critical debate points.

### Overview

The belief network system moves Aragora from binary accept/reject to nuanced probabilistic reasoning:

- **BeliefNode**: Claims with probability distributions over truth values
- **BeliefNetwork**: Factor graph for message passing
- **Loopy belief propagation**: For cyclic argument graphs
- **Centrality analysis**: Identify load-bearing claims
- **Crux detection**: Find debate-pivotal claims

### Key Classes

```python
from aragora.reasoning.belief import (
    BeliefNetwork,
    BeliefNode,
    BeliefDistribution,
    BeliefStatus,
)

# Create a belief network
network = BeliefNetwork()

# Add claims as nodes with probability distributions
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

# Propagate beliefs through the network
result = network.propagate()
```

### Belief Distribution

Each node has a probability distribution over truth values:

```python
from aragora.reasoning.belief import BeliefDistribution

# Create from confidence score
dist = BeliefDistribution.from_confidence(0.85, lean_true=True)
print(f"P(true): {dist.p_true}")    # 0.85
print(f"P(false): {dist.p_false}")  # 0.15
print(f"Entropy: {dist.entropy}")   # Uncertainty measure
print(f"Confidence: {dist.confidence}")  # Max probability

# Uniform (maximum uncertainty)
uniform = BeliefDistribution.uniform()  # p_true=0.5, p_false=0.5
```

### Belief States

| Status | Description |
|--------|-------------|
| `PRIOR` | Initial belief before evidence |
| `UPDATED` | Updated via propagation |
| `CONVERGED` | Stable after propagation |
| `CONTESTED` | Multiple conflicting updates |

### Network Analysis

```python
# Find critical claims (high centrality)
load_bearing = network.get_load_bearing_claims(threshold=0.7)

# Identify cruxes (pivotal disagreements)
cruxes = network.identify_cruxes(agent_beliefs)

# Export for visualization (D3.js compatible)
graph_data = network.export_d3_json()
```

### Provenance Tracking

Track the origin and transformation of claims:

```python
from aragora.reasoning.provenance import ProvenanceTracker, ProvenanceChain

tracker = ProvenanceTracker()

# Record claim origin
tracker.record_claim(
    claim_id="claim_001",
    text="Market cap is $1T",
    source="agent:gpt-4",
    evidence_ids=["ev_001", "ev_002"]
)

# Build provenance chain
chain = tracker.build_chain("claim_001")
print(chain.depth)           # How many transformations
print(chain.original_sources)  # Root sources
```

### Enhanced Provenance with Confidence Decay

```python
from aragora.reasoning.provenance_enhanced import EnhancedProvenanceTracker

tracker = EnhancedProvenanceTracker()
tracker.record_with_confidence(
    claim_id="claim_001",
    confidence=0.9,
    decay_factor=0.95  # Confidence decay per hop
)
```

### Claim Extraction

```python
from aragora.reasoning.claims import ClaimExtractor, ClaimType

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

**Claim Types:**
- `FACTUAL` - Verifiable facts
- `QUANTITATIVE` - Numerical claims
- `QUALITATIVE` - Quality/opinion claims
- `CAUSAL` - Cause-effect relationships
- `PREDICTIVE` - Future predictions
- `NORMATIVE` - Value judgments

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/belief/{debate_id}/network` | GET | Get belief network |
| `/api/belief/{debate_id}/cruxes` | GET | Identify cruxes |
| `/api/belief/{debate_id}/load-bearing` | GET | Get load-bearing claims |
| `/api/belief/{debate_id}/provenance/{claim_id}` | GET | Get claim provenance |

---

## Explainability

The explainability system provides comprehensive insight into why and how decisions were made during debates.

### Overview

The `Decision` entity aggregates all explainability data:

- **Evidence chains**: Links from conclusion to supporting evidence
- **Vote pivots**: Which votes were most influential
- **Belief changes**: How agent beliefs evolved
- **Confidence attribution**: What factors contributed to confidence
- **Counterfactual analysis**: What would change the outcome

### Building Explanations

```python
from aragora.explainability import ExplanationBuilder

builder = ExplanationBuilder(
    evidence_tracker=evidence_tracker,
    belief_network=belief_network,
    calibration_tracker=calibration_tracker,
    elo_system=elo_system,
    provenance_tracker=provenance_tracker,
)

# Build decision entity from debate result
decision = await builder.build(
    result=debate_result,
    context=debate_context,
    include_counterfactuals=True,
)

# Generate human-readable summary
summary = builder.generate_summary(decision)
print(summary)
```

### Decision Entity

```python
from aragora.explainability import Decision

# Access decision components
print(f"Consensus: {decision.consensus_reached}")
print(f"Confidence: {decision.confidence:.0%}")
print(f"Rounds: {decision.rounds_used}")

# Evidence quality
print(f"Evidence score: {decision.evidence_quality_score:.2f}")
print(f"Agreement score: {decision.agent_agreement_score:.2f}")
print(f"Belief stability: {decision.belief_stability_score:.2f}")

# Export to JSON
json_data = decision.to_json()
```

### Factor Decomposition

Understand what contributed to confidence:

```python
from aragora.explainability import ConfidenceAttribution

# Get major confidence factors (contribution >= 15%)
major_factors = decision.get_major_confidence_factors(threshold=0.15)

for factor in major_factors:
    print(f"{factor.factor}: {factor.contribution:.0%}")
    print(f"  {factor.explanation}")

# Example output:
# consensus_strength: 40%
#   Agreement level among agents (85% margin)
# evidence_quality: 30%
#   Quality of supporting evidence (78% average)
# agent_calibration: 20%
#   Historical accuracy of participating agents
# debate_efficiency: 10%
#   Reached consensus in 3 rounds
```

### Counterfactual Generation

Analyze what would change the outcome:

```python
from aragora.explainability import Counterfactual

# Get high-sensitivity counterfactuals
sensitive = decision.get_high_sensitivity_counterfactuals(threshold=0.5)

for cf in sensitive:
    print(f"If: {cf.condition}")
    print(f"Then: {cf.outcome_change}")
    print(f"Sensitivity: {cf.sensitivity:.0%}")
    print(f"Affected agents: {cf.affected_agents}")

# Example output:
# If: claude had voted differently
# Then: Possible change in consensus or confidence
# Sensitivity: 65%
# Affected agents: ['claude']
```

### Vote Pivot Analysis

Identify influential votes:

```python
from aragora.explainability import VotePivot

# Get pivotal votes (influence >= 30%)
pivotal = decision.get_pivotal_votes(threshold=0.3)

for vote in pivotal:
    print(f"{vote.agent}: voted '{vote.choice}'")
    print(f"  Influence: {vote.influence_score:.0%}")
    print(f"  Weight: {vote.weight:.2f}")
    print(f"  ELO: {vote.elo_rating}")
    print(f"  Flipped: {vote.flip_detected}")
```

### Belief Change Tracking

Track how agents changed their positions:

```python
from aragora.explainability import BeliefChange

# Get significant belief changes (delta >= 20%)
changes = decision.get_significant_belief_changes(min_delta=0.2)

for change in changes:
    print(f"{change.agent} (round {change.round}):")
    print(f"  Prior: {change.prior_belief} ({change.prior_confidence:.0%})")
    print(f"  Posterior: {change.posterior_belief} ({change.posterior_confidence:.0%})")
    print(f"  Trigger: {change.trigger} from {change.trigger_source}")
```

### Natural Language Summary

```python
summary = builder.generate_summary(decision)
print(summary)

# Output:
# ## Decision Summary
#
# **Consensus:** Reached
# **Confidence:** 85%
# **Rounds:** 3
#
# ### Conclusion
# Based on the analysis, microservices are recommended for...
#
# ### Key Evidence
# - **anthropic-api**: The scalability requirements suggest...
# - **openai-api**: Historical data shows that teams of this size...
#
# ### Most Influential Votes
# - **claude** voted 'microservices' (influence: 45%)
# - **gpt-4** voted 'microservices' (influence: 35%)
#
# ### Confidence Factors
# - consensus_strength: Agreement level among agents (85% margin)
# - evidence_quality: Quality of supporting evidence (78% average)
```

---

## Matrix/Graph Debates

Advanced debate topologies for exploring complex decision spaces.

### Matrix Debates

Matrix debates run the same question across multiple scenarios in parallel, identifying universal conclusions (true in all scenarios) vs conditional conclusions (specific to certain contexts).

**When to Use:**
- Technology selection across different scales
- Policy analysis for different stakeholder groups
- Risk assessment under various failure conditions

```python
# Python SDK
result = client.matrix_debates.create(
    task="Should we adopt microservices?",
    scenarios=[
        {"name": "small_team", "parameters": {"team_size": 5}},
        {"name": "large_team", "parameters": {"team_size": 50}},
        {"name": "startup", "parameters": {"budget": "low"}},
        {"name": "enterprise", "parameters": {"budget": "high"}},
    ]
)

# Get universal vs conditional conclusions
conclusions = client.matrix_debates.get_conclusions(result.matrix_id)
print("Universal (true in all scenarios):", conclusions.universal)
print("Conditional:", conclusions.conditional)
```

**API Example:**

```bash
POST /api/debates/matrix
```

```json
{
  "task": "What database architecture should we use?",
  "agents": ["anthropic-api", "openai-api"],
  "max_rounds": 3,
  "scenarios": [
    {
      "name": "High-write workload",
      "parameters": {"writes_per_second": 100000},
      "constraints": ["Must support ACID transactions"],
      "is_baseline": true
    },
    {
      "name": "Read-heavy analytics",
      "parameters": {"reads_per_second": 500000},
      "constraints": ["Sub-second query latency required"]
    },
    {
      "name": "Global distribution",
      "parameters": {"regions": 12, "consistency": "eventual"},
      "constraints": ["Must comply with GDPR data residency"]
    }
  ]
}
```

**Response includes:**
- Per-scenario results with consensus status
- Universal conclusions (true across all scenarios)
- Conditional conclusions (scenario-specific)
- Comparison matrix with aggregate statistics

### Graph Debates

Graph debates allow branching when agents fundamentally disagree, exploring divergent perspectives in parallel before synthesizing.

**When to Use:**
- Exploring genuine trade-offs without clear "right" answers
- Multi-stakeholder analysis with different priorities
- Adversarial red-teaming for failure modes

```python
# Python SDK
result = client.graph_debates.create(
    task="Should AI models be open-source or closed?",
    agents=["anthropic-api", "openai-api", "gemini"],
    max_rounds=4,
    branch_policy={
        "min_disagreement": 0.6,  # Branch when 60%+ disagreement
        "max_branches": 2,
        "merge_strategy": "synthesis"
    }
)

# Get all branches
branches = client.graph_debates.get_branches(result.debate_id)
for branch in branches:
    print(f"Branch: {branch.name}, Nodes: {len(branch.nodes)}")
```

**API Example:**

```bash
POST /api/debates/graph
```

```json
{
  "task": "Should we prioritize renewable energy or nuclear power?",
  "agents": ["anthropic-api", "openai-api", "gemini"],
  "max_rounds": 5,
  "branch_policy": {
    "min_disagreement": 0.7,
    "max_branches": 3,
    "auto_merge": true,
    "merge_strategy": "synthesis"
  }
}
```

**Branch Reasons:**
- `disagreement_on_scalability` - Agents disagree on scaling approach
- `opposing_evidence` - Agents cite contradicting evidence
- `different_frameworks` - Incompatible analytical frameworks
- `value_conflict` - Different value priorities

**Merge Strategies:**

| Strategy | Description |
|----------|-------------|
| `synthesis` | Create unified position acknowledging both branches |
| `vote` | Agents vote on which branch has stronger argument |
| `best_evidence` | Branch with highest evidence quality wins |

### Matrix vs Graph: When to Use

| Feature | Matrix Debates | Graph Debates |
|---------|---------------|---------------|
| **Purpose** | Explore same question across contexts | Explore divergent perspectives |
| **Structure** | Parallel independent scenarios | Tree with branching and merging |
| **Output** | Universal + conditional conclusions | Synthesized multi-perspective view |
| **Best for** | Technology selection, policy analysis | Trade-offs, adversarial analysis |

---

## Workflow Templates

Pre-built workflow templates for common enterprise use cases, powered by the Workflow Engine.

### Overview

Aragora provides 40+ workflow templates across multiple categories:

| Category | Templates | Use Cases |
|----------|-----------|-----------|
| **Legal** | Contract Review, Due Diligence, Compliance Audit | Legal document analysis |
| **Healthcare** | HIPAA Assessment, Clinical Review, PHI Audit | Healthcare compliance |
| **Code** | Security Audit, Architecture Review, Code Quality | Development workflows |
| **Accounting** | Financial Audit, SOX Compliance | Financial compliance |
| **AI/ML** | Model Deployment, AI Governance, Prompt Engineering | ML operations |
| **DevOps** | CI/CD Review, Incident Response, Infrastructure Audit | Operations |
| **Product** | PRD Review, Feature Spec, Launch Readiness | Product management |
| **Marketing** | Ad Performance, Lead Sync, Analytics | Marketing automation |
| **SME** | Vendor Evaluation, Hiring, Budget Allocation | Business decisions |

### Using Templates

```python
from aragora.workflow.templates import get_template, list_templates

# List available templates
templates = list_templates(category="code")
for t in templates:
    print(f"{t['id']}: {t['name']}")

# Get a specific template
template = get_template("code/security-audit")

# Execute with workflow engine
from aragora.workflow.engine import WorkflowEngine

engine = WorkflowEngine()
result = await engine.execute(template, inputs={
    "codebase": "/path/to/code",
    "scope": "authentication module",
})
```

### Pattern Factories

Create workflows from reusable patterns:

```python
from aragora.workflow.templates.patterns import (
    create_hive_mind_workflow,
    create_map_reduce_workflow,
    create_review_cycle_workflow,
)

# Hive Mind: Parallel agent execution with consensus
workflow = create_hive_mind_workflow(
    name="Risk Assessment",
    agents=["claude", "gpt4", "gemini"],
    task="Assess risks in this proposal",
    consensus_mode="weighted",
    consensus_threshold=0.7,
    include_dissent=True,
)

# MapReduce: Split, process in parallel, aggregate
workflow = create_map_reduce_workflow(
    name="Document Analysis",
    split_strategy="chunks",
    chunk_size=4000,
    map_agent="claude",
    reduce_agent="gpt4",
)

# Review Cycle: Iterative refinement until convergence
workflow = create_review_cycle_workflow(
    name="Code Review",
    reviewer_agent="claude",
    max_iterations=3,
    convergence_threshold=0.9,
)
```

### Available Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **HiveMind** | Parallel agent execution, consensus merge | Multi-perspective analysis |
| **Sequential** | Linear pipeline with data passing | Step-by-step processing |
| **MapReduce** | Split, parallel process, aggregate | Large document analysis |
| **Hierarchical** | Manager-worker delegation | Complex task breakdown |
| **ReviewCycle** | Iterative refinement, convergence check | Quality improvement |
| **Dialectic** | Thesis-antithesis-synthesis | Exploring trade-offs |
| **PostDebate** | Automated post-debate processing | Decision follow-up |

### Custom Node Creation

Create custom workflow nodes for specialized functionality:

```python
from aragora.workflow.step import WorkflowStep, WorkflowContext, StepResult

class CustomAnalysisStep(WorkflowStep):
    """Custom step for specialized analysis."""

    async def execute(
        self,
        context: WorkflowContext,
        inputs: dict,
    ) -> StepResult:
        # Access previous step outputs
        data = inputs.get("data")

        # Perform custom analysis
        result = await self._analyze(data)

        return StepResult(
            status=StepStatus.COMPLETED,
            outputs={"analysis": result},
        )

# Register with engine
engine = WorkflowEngine()
engine.register_step_type("custom_analysis", CustomAnalysisStep)
```

### Built-in Node Types

| Node Type | Description |
|-----------|-------------|
| `agent` | Execute agent task |
| `parallel` | Run multiple steps concurrently |
| `conditional` | Branch based on expressions |
| `loop` | Iterate until condition met |
| `human_checkpoint` | Require human approval |
| `memory_read` / `memory_write` | Knowledge Mound integration |
| `debate` | Execute Aragora debate |
| `decision` | Conditional branching |
| `task` | Generic task execution |
| `connector` | External system integration |
| `nomic_loop` | Self-improvement cycle |
| `gauntlet` | Adversarial validation |
| `knowledge_pipeline` | Document ingestion |
| `knowledge_pruning` | Automatic maintenance |

### Template Composition

Compose complex workflows from simpler templates:

```python
from aragora.workflow.templates import (
    SECURITY_AUDIT_TEMPLATE,
    CODE_QUALITY_TEMPLATE,
)
from aragora.workflow.engine import WorkflowEngine

# Combine templates into a comprehensive review
combined_workflow = {
    "id": "comprehensive-code-review",
    "name": "Comprehensive Code Review",
    "steps": [
        {
            "id": "security",
            "type": "workflow",
            "workflow": SECURITY_AUDIT_TEMPLATE,
        },
        {
            "id": "quality",
            "type": "workflow",
            "workflow": CODE_QUALITY_TEMPLATE,
            "depends_on": ["security"],
        },
        {
            "id": "synthesis",
            "type": "agent",
            "agent": "claude",
            "prompt": "Synthesize the security and quality findings",
            "inputs": {
                "security_report": "{{steps.security.outputs.report}}",
                "quality_report": "{{steps.quality.outputs.report}}",
            },
        },
    ],
}

engine = WorkflowEngine()
result = await engine.execute(combined_workflow, inputs={"codebase": "..."})
```

### SME Templates

Specialized templates for small-medium enterprise decisions:

```python
from aragora.workflow.templates.sme import (
    create_vendor_evaluation_workflow,
    create_hiring_decision_workflow,
    create_budget_allocation_workflow,
    create_performance_review_workflow,
    create_feature_prioritization_workflow,
)

# Vendor evaluation with multi-criteria analysis
workflow = create_vendor_evaluation_workflow(
    vendors=["Vendor A", "Vendor B", "Vendor C"],
    criteria=["cost", "reliability", "support", "features"],
)

# Hiring decision with structured evaluation
workflow = create_hiring_decision_workflow(
    position="Senior Engineer",
    candidates=["Alice", "Bob", "Carol"],
    interview_data=interview_results,
)
```

### Quickstart Templates

Rapid decision-making templates:

```python
from aragora.workflow.templates.quickstart import (
    quick_decision,
    quick_analysis,
    quick_risks,
    quick_ideas,
)

# Quick yes/no decision
result = await quick_decision("Should we migrate to Kubernetes?")

# Quick pros/cons analysis
result = await quick_analysis("GraphQL vs REST for our API")

# Quick risk assessment
result = await quick_risks("Launching without beta testing")

# Quick brainstorming
result = await quick_ideas("Improve user onboarding")
```

---

## Related Documentation

- [RLM User Guide](RLM_USER_GUIDE.md) - Detailed RLM documentation
- [Reasoning Module](REASONING.md) - Belief networks and claims
- [Workflows](WORKFLOWS.md) - Common workflow patterns
- [Matrix Debates](MATRIX_DEBATES.md) - Full matrix debate reference
- [Graph Debates](GRAPH_DEBATES.md) - Full graph debate reference
- [Evidence System](EVIDENCE.md) - Evidence collection and provenance
