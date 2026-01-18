# ML Integration Guide

This guide covers Aragora's local ML capabilities and how to integrate them with the debate system for improved agent selection, quality assessment, and consensus prediction.

## Overview

The `aragora.ml` module provides:

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **LocalEmbeddingService** | Text embeddings | Semantic similarity, search |
| **QualityScorer** | Response quality assessment | Filter low-quality responses |
| **ConsensusPredictor** | Consensus likelihood estimation | Early debate termination |
| **AgentRouter** | Task-based agent selection | Intelligent team formation |
| **LocalFineTuner** | PEFT/LoRA fine-tuning | Domain adaptation |

All models run locally without external API dependencies.

## Quick Start

```python
from aragora.ml import (
    get_embedding_service,
    get_quality_scorer,
    get_consensus_predictor,
    get_agent_router,
)

# Embeddings
service = get_embedding_service()
embedding = service.embed("Design a rate limiter")

# Quality scoring
scorer = get_quality_scorer()
score = scorer.score("The response text", context="task description")
print(f"Quality: {score.overall:.2f}, High quality: {score.is_high_quality}")

# Consensus prediction
predictor = get_consensus_predictor()
prediction = predictor.predict([
    ("claude", "I recommend approach A"),
    ("gpt-4", "I agree, approach A is best"),
])
print(f"Consensus probability: {prediction.probability:.2f}")

# Agent routing
router = get_agent_router()
decision = router.route(
    task="Implement a binary search algorithm",
    available_agents=["claude", "gpt-4", "codex"],
    team_size=2,
)
print(f"Selected: {decision.selected_agents}")
print(f"Task type: {decision.task_type}")
```

## REST API

The ML module is exposed via REST endpoints:

### Agent Routing
```
POST /api/ml/route
```

Request:
```json
{
  "task": "Implement a caching layer",
  "available_agents": ["claude", "gpt-4", "codex", "gemini"],
  "team_size": 3,
  "constraints": {"require_code": true}
}
```

Response:
```json
{
  "selected_agents": ["codex", "claude", "gpt-4"],
  "task_type": "coding",
  "confidence": 0.85,
  "reasoning": ["task_type=coding", "codex_strong_at_coding"],
  "agent_scores": {"codex": 0.92, "claude": 0.85, "gpt-4": 0.78},
  "diversity_score": 0.67
}
```

### Quality Scoring
```
POST /api/ml/score
```

Request:
```json
{
  "text": "The response to evaluate",
  "context": "Optional task context"
}
```

Response:
```json
{
  "overall": 0.75,
  "coherence": 0.80,
  "completeness": 0.70,
  "relevance": 0.78,
  "clarity": 0.72,
  "confidence": 0.65,
  "is_high_quality": true,
  "needs_review": false
}
```

### Batch Scoring
```
POST /api/ml/score-batch
```

### Consensus Prediction
```
POST /api/ml/consensus
```

Request:
```json
{
  "responses": [
    ["agent1", "I agree with approach A"],
    ["agent2", "Approach A is the best choice"]
  ],
  "context": "Design a rate limiter",
  "current_round": 2,
  "total_rounds": 3
}
```

Response:
```json
{
  "probability": 0.85,
  "confidence": 0.70,
  "convergence_trend": "converging",
  "estimated_rounds": 2,
  "likely_consensus": true,
  "early_termination_safe": true,
  "key_factors": ["high_semantic_similarity", "stance_agreement"]
}
```

### Embeddings
```
POST /api/ml/embed
POST /api/ml/search
```

### Training Export
```
POST /api/ml/export-training
```

### Model Info
```
GET /api/ml/models
GET /api/ml/stats
```

## Debate Integration

### ML-Enhanced Team Selection

The `MLDelegationStrategy` integrates with the debate system's team selection:

```python
from aragora.debate.ml_integration import (
    MLDelegationStrategy,
    create_ml_team_selector,
)

# Create ML-enhanced team selector
selector = create_ml_team_selector(
    elo_system=elo,
    calibration_tracker=calibration,
    ml_weight=0.3,  # 30% ML, 70% ELO+calibration
)

# Use in arena
from aragora.debate import Arena, ArenaConfig

config = ArenaConfig(
    team_selector=selector,
    # ... other config
)
arena = Arena(env, agents, protocol, config=config)
```

### Quality Gates

Filter low-quality responses before consensus:

```python
from aragora.debate.ml_integration import QualityGate

gate = QualityGate(threshold=0.6)

# Filter responses
responses = [
    ("claude", "Comprehensive analysis..."),
    ("gpt-4", "ok"),  # Low quality, will be filtered
]

filtered = gate.filter_responses(responses, context="Design task")
# Only high-quality responses remain
```

### Early Termination

Use consensus prediction to end debates early:

```python
from aragora.debate.ml_integration import ConsensusEstimator

estimator = ConsensusEstimator(
    early_termination_threshold=0.85,
    min_rounds=2,
)

# During debate loop
if estimator.should_terminate_early(responses, current_round=2):
    print("Consensus likely - safe to terminate early")
    break
```

### Full Integration Example

```python
from aragora.debate import Arena, Environment, DebateProtocol
from aragora.debate.ml_integration import (
    MLDelegationStrategy,
    QualityGate,
    ConsensusEstimator,
    get_training_exporter,
)

# Setup ML components
ml_delegation = MLDelegationStrategy()
quality_gate = QualityGate(threshold=0.6)
consensus_estimator = ConsensusEstimator()
exporter = get_training_exporter()

# Configure arena with ML delegation
config = ArenaConfig(
    delegation_strategy=ml_delegation,
)

env = Environment(task="Design a distributed cache")
arena = Arena(env, agents, protocol, config=config)

# Run debate with quality gates
result = await arena.run()

# Export for training
training_data = exporter.export_debate(
    task=env.task,
    consensus_response=result.consensus.content,
    rejected_responses=[m.content for m in result.messages if m != result.consensus],
)
training_data.to_jsonl("debate_training.jsonl")
```

## Embedding Models

Available models via `EmbeddingModel` enum:

| Model | Dimensions | Speed | Use Case |
|-------|------------|-------|----------|
| `MINILM` | 384 | Fast | General purpose |
| `MINILM_L12` | 384 | Medium | Better quality |
| `MPNET` | 768 | Slow | Highest quality |
| `MULTILINGUAL` | 384 | Medium | 50+ languages |
| `CODE` | 768 | Medium | Code search |

```python
from aragora.ml import LocalEmbeddingService, LocalEmbeddingConfig, EmbeddingModel

config = LocalEmbeddingConfig(
    model=EmbeddingModel.MPNET,  # High quality
    device="cuda",  # GPU acceleration
)
service = LocalEmbeddingService(config)
```

## Agent Router

### Task Types

The router classifies tasks into:

- `coding` - Implementation, algorithms, bug fixes
- `analysis` - Evaluation, comparison, review
- `creative` - Writing, brainstorming, design
- `reasoning` - Logic, deduction, argumentation
- `research` - Investigation, fact-finding
- `math` - Calculations, proofs, equations
- `general` - Unclassified tasks

### Agent Capabilities

Default capabilities are defined for common agents:

```python
from aragora.ml import AgentRouter
from aragora.ml.agent_router import AgentCapabilities, TaskType

router = AgentRouter()

# Register custom agent
router.register_agent(AgentCapabilities(
    agent_id="custom-coder",
    strengths=[TaskType.CODING, TaskType.MATH],
    weaknesses=[TaskType.CREATIVE],
    speed_tier=1,  # Fast
    cost_tier=2,   # Medium
    elo_rating=1050,
))
```

### Recording Performance

Improve routing over time:

```python
router.record_performance("claude", "coding", success=True)
router.record_performance("codex", "coding", success=True)
router.record_performance("gpt-4", "creative", success=False)

# Check stats
stats = router.get_agent_stats("claude")
print(f"Success rate: {stats['overall_success_rate']:.2%}")
```

## Fine-Tuning

### Creating Training Data

```python
from aragora.ml import TrainingData, TrainingExample

# From debate outcomes
debates = [
    {
        "task": "Design a rate limiter",
        "consensus": "Use token bucket algorithm...",
        "rejected": ["Simple counter approach..."],
    }
]

training_data = TrainingData.from_debates(debates)
training_data.to_jsonl("training.jsonl")
```

### Running Fine-Tuning

```python
from aragora.ml import LocalFineTuner, FineTuneConfig

config = FineTuneConfig(
    base_model="microsoft/phi-2",
    lora_r=8,
    lora_alpha=32,
    epochs=3,
    use_4bit=True,  # Memory efficient
)

tuner = LocalFineTuner(config)
result = tuner.train(training_data)

if result.success:
    print(f"Model saved to: {result.model_path}")
    print(f"Training loss: {result.metrics['train_loss']:.4f}")
```

### Using Fine-Tuned Model

```python
tuner.load_trained_model("./fine_tuned_model")
response = tuner.generate("Implement a caching strategy")
```

## Configuration

### Environment Variables

```bash
# Embedding model cache
SENTENCE_TRANSFORMERS_HOME=/path/to/cache

# GPU device
CUDA_VISIBLE_DEVICES=0

# Disable embeddings (use fallback scoring)
ARAGORA_ML_DISABLE_EMBEDDINGS=true
```

### MLIntegrationConfig

```python
from aragora.debate.ml_integration import MLIntegrationConfig

config = MLIntegrationConfig(
    # Agent routing
    use_ml_routing=True,
    ml_routing_weight=0.4,
    fallback_to_elo=True,

    # Quality gates
    enable_quality_gates=True,
    quality_threshold=0.6,
    min_confidence=0.4,

    # Consensus estimation
    enable_early_termination=True,
    early_termination_threshold=0.85,
    min_rounds_before_termination=2,

    # Performance
    cache_routing_decisions=True,
    cache_ttl_seconds=300,
)
```

## Performance

### Latency Benchmarks

| Operation | Latency (CPU) | Latency (GPU) |
|-----------|---------------|---------------|
| Single embedding | ~10ms | ~2ms |
| Batch embed (100) | ~200ms | ~20ms |
| Quality score | ~5ms | ~5ms |
| Consensus predict | ~15ms | ~10ms |
| Agent route | ~8ms | ~8ms |

### Memory Usage

| Model | RAM | GPU Memory |
|-------|-----|------------|
| MiniLM | ~100MB | ~200MB |
| MPNET | ~400MB | ~500MB |
| Multilingual | ~200MB | ~300MB |

## Troubleshooting

### Model Download Issues

```python
# Pre-download models
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
```

### GPU Not Detected

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Quality Scores Too Low

Adjust thresholds:

```python
gate = QualityGate(
    threshold=0.5,      # Lower threshold
    min_confidence=0.3, # Accept lower confidence
)
```

### Routing Not Diverse

Enable diversity optimization:

```python
from aragora.ml.agent_router import AgentRouterConfig

config = AgentRouterConfig(
    prefer_diversity=True,
    max_same_provider=1,  # Max 1 agent per provider
)
```
