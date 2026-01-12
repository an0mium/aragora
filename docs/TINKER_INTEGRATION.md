# Tinker Integration Guide

Aragora integrates with [Tinker](https://thinkingmachines.ai/tinker/) to enable fine-tuning of open-source LLMs on debate data. This creates a closed-loop learning system where debate outcomes continuously improve agent performance.

## Overview

The integration provides:

- **Training Data Export**: Extract SFT, DPO, and adversarial training data from debates
- **TinkerClient**: API wrapper for fine-tuning and inference
- **TinkerAgent**: Debate agent using fine-tuned models
- **TrainingScheduler**: Batch job management
- **TinkerEvaluator**: A/B testing framework
- **ModelRegistry**: Adapter lifecycle management

## Quick Start

### 1. Set Up Environment

```bash
export TINKER_API_KEY="your-api-key"
export TINKER_BASE_MODEL="llama-3.3-70b"  # optional
```

### 2. Export Training Data

```bash
# Export SFT data from winning debates
python -m aragora.cli.training export-sft -o sft_data.jsonl --min-confidence 0.8

# Export DPO preference pairs
python -m aragora.cli.training export-dpo -o dpo_data.jsonl --min-elo-diff 100

# Export all data types
python -m aragora.cli.training export-all -d training_data/
```

### 3. Train a Model

```bash
# Train SFT model
python -m aragora.cli.training train-sft --model llama-3.3-70b --limit 1000

# Train DPO model
python -m aragora.cli.training train-dpo --model llama-3.3-70b --beta 0.1

# Combined pipeline (SFT then DPO)
python -m aragora.cli.training train-combined --adapter-name aragora-v1
```

### 4. Use in Debates

```python
from aragora.agents.api_agents import TinkerAgent
from aragora.debate import Arena

# Create agent with fine-tuned model
agent = TinkerAgent(
    name="aragora-expert",
    model_id="aragora-sft-v1",
    role="proposer",
)

# Use in debate
arena = Arena(env, [agent, other_agents], protocol)
result = await arena.run()
```

## Training Paradigms

### Supervised Fine-Tuning (SFT)

Trains models on successful debate patterns:

```python
from aragora.training import SFTExporter

exporter = SFTExporter()
data = exporter.export(
    min_confidence=0.8,      # High-confidence debates only
    min_success_rate=0.7,    # Successful patterns only
    include_debates=True,
    include_patterns=True,
    include_critiques=True,
)
```

**Data sources:**
- Winning debate responses (task → answer)
- Successful critique patterns (issue → suggestion)
- Expert domain responses

### Direct Preference Optimization (DPO)

Trains on preference pairs (chosen vs rejected):

```python
from aragora.training import DPOExporter

exporter = DPOExporter()
data = exporter.export(
    min_elo_difference=100,  # Clear skill gap
    include_head_to_head=True,
    include_calibration=True,
    include_domain_specific=True,
)
```

**Data sources:**
- ELO win/loss pairs (winner = chosen)
- Calibration quality (well-calibrated = chosen)
- Domain expertise (expert = chosen)

### Adversarial Training

Trains against Gauntlet vulnerability patterns:

```python
from aragora.training import GauntletExporter

exporter = GauntletExporter()
data = exporter.export(
    include_attack_patterns=True,
    include_defense_training=True,
)
```

## TinkerClient API

### Training

```python
from aragora.training import TinkerClient, TinkerModel

async with TinkerClient() as client:
    # SFT training
    result = await client.train_sft(
        training_data=data,
        model=TinkerModel.LLAMA_3_3_70B,
        adapter_name="my-adapter",
        lora_rank=16,
        learning_rate=1e-4,
    )

    # DPO training
    result = await client.train_dpo(
        preference_data=data,
        model="llama-3.3-70b",
        beta=0.1,  # Temperature parameter
    )
```

### Inference

```python
async with TinkerClient() as client:
    # Generate from fine-tuned model
    response = await client.sample(
        prompt="Critique this proposal: ...",
        model_id="aragora-sft-v1",
        temperature=0.7,
    )

    # Stream generation
    async for chunk in client.sample_stream(prompt, model_id=model_id):
        print(chunk, end="")
```

### Checkpoints

```python
# Save checkpoint
path = await client.save_checkpoint(model_id, "/checkpoints/my-model")

# Load checkpoint
model_id = await client.load_checkpoint(path)
```

## TinkerAgent

Use fine-tuned models as debate agents:

```python
from aragora.agents.api_agents import TinkerAgent

# Base model
agent = TinkerAgent(name="tinker", model="llama-3.3-70b")

# Fine-tuned model
agent = TinkerAgent(
    name="tinker-security",
    model="llama-3.3-70b",
    model_id="aragora-security-v1",
    adapter="security-expert",
)

# Specialized variants
from aragora.agents.api_agents import (
    TinkerLlamaAgent,    # Llama 3.3 70B
    TinkerQwenAgent,     # Qwen 2.5 72B
    TinkerDeepSeekAgent, # DeepSeek V3
)
```

### Hot-Swapping Adapters

```python
agent = TinkerAgent(name="multi-expert")

# Switch domains dynamically
agent.set_adapter("security-expert")
response = await agent.respond(security_task, context)

agent.set_adapter("performance-expert")
response = await agent.respond(performance_task, context)
```

## Training Scheduler

Manage batch training jobs:

```python
from aragora.training import TrainingScheduler

scheduler = TrainingScheduler()

# Schedule SFT job
job = await scheduler.schedule_sft(
    model="llama-3.3-70b",
    adapter_name="aragora-v1",
    min_confidence=0.8,
)

# Wait for completion
completed = await scheduler.wait_for_job(job.job_id)
print(f"Model ID: {completed.model_id}")

# Schedule combined pipeline
job = await scheduler.schedule_combined(
    model="llama-3.3-70b",
    sft_limit=1000,
    dpo_limit=500,
)
```

### Job Management

```python
# List jobs
jobs = scheduler.list_jobs(status=JobStatus.RUNNING)

# Cancel job
scheduler.cancel_job(job_id)

# Save/load state
scheduler.save_state("scheduler_state.json")
scheduler.load_state("scheduler_state.json")
```

## TinkerEvaluator

A/B testing framework for comparing models:

```python
from aragora.training import TinkerEvaluator

evaluator = TinkerEvaluator()

# Compare fine-tuned vs baseline
result = await evaluator.a_b_test(
    tasks=["Design a rate limiter", "Optimize database queries"],
    fine_tuned_agent=my_tinker_agent,
    baseline_agent=my_baseline_agent,
    num_trials=10,
)

print(f"Win rate: {result.fine_tuned_win_rate:.1%}")
print(f"Statistically significant: {result.is_significant}")
```

### Benchmark Evaluation

```python
results = await evaluator.evaluate_on_benchmark(
    agent=my_agent,
    benchmark_tasks=standard_tasks,
    baseline_agents=[claude, gpt4, gemini],
    trials_per_task=5,
)

print(f"Overall win rate: {results['overall_win_rate']:.1%}")
print(f"ELO gain: {results['overall_elo_gain']:.0f}")
```

## Model Registry

Track and manage trained models:

```python
from aragora.training import ModelRegistry, ModelMetadata, get_registry

registry = get_registry()

# Register a model
registry.register(ModelMetadata(
    model_id="aragora-security-v1",
    base_model="llama-3.3-70b",
    adapter_name="security-expert",
    training_type="sft",
    primary_domain="security",
))

# Get best model for domain
model = registry.get_best_for_domain("security")

# Update metrics after evaluation
registry.update_metrics(
    "aragora-security-v1",
    elo_rating=1250,
    win_rate=0.65,
    calibration_score=0.85,
)

# Lifecycle management
registry.deprecate("old-model-v1", notes="Superseded by v2")
registry.archive("very-old-model")
```

## CLI Commands

```bash
# Data export
aragora training export-sft --help
aragora training export-dpo --help
aragora training export-gauntlet --help
aragora training export-all --help

# Training
aragora training train-sft --help
aragora training train-dpo --help
aragora training train-combined --help

# Model management
aragora training list-models
aragora training sample --model-id <id> "Your prompt"
aragora training stats

# API testing
aragora training test-connection
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TINKER_API_KEY` | Tinker API key (required) | - |
| `TINKER_BASE_MODEL` | Default base model | `llama-3.3-70b` |
| `TINKER_MODEL_REGISTRY` | Registry file path | `model_registry.json` |

## Supported Models

| Model | ID | Parameters |
|-------|----|-----------:|
| Llama 3.3 | `llama-3.3-70b` | 70B |
| Llama 3.1 | `llama-3.1-8b` | 8B |
| Qwen 2.5 | `qwen-2.5-72b` | 72B |
| Qwen 3 | `qwen-3-32b` | 32B |
| DeepSeek V3 | `deepseek-v3` | 236B |
| DeepSeek R1 | `deepseek-r1` | 236B |

## Best Practices

### Data Quality

1. **Confidence threshold**: Start with `min_confidence=0.8` for SFT
2. **ELO difference**: Use `min_elo_difference=100` for clear preference signals
3. **Data diversity**: Include multiple domains in training data
4. **Replay buffer**: Mix 20% historical data to prevent forgetting

### Training

1. **Start with SFT**: Train base debate patterns first
2. **Add DPO**: Refine with preference learning
3. **LoRA rank**: Start with 16, increase to 32 for domain experts
4. **Batch size**: 4 with gradient accumulation of 4

### Evaluation

1. **Minimum trials**: At least 10 per comparison for significance
2. **Diverse tasks**: Test on multiple task types
3. **Multiple baselines**: Compare against several models
4. **Track calibration**: Monitor confidence accuracy

## Troubleshooting

### Connection Issues

```bash
# Test API connection
python -m aragora.cli.training test-connection
```

### No Training Data

```bash
# Check database statistics
python -m aragora.cli.training stats
```

### Training Failures

Check scheduler state for error details:

```python
scheduler = TrainingScheduler()
job = scheduler.get_job("job-id")
print(job.error)
```

## Architecture

```
aragora/training/
├── __init__.py           # Module exports
├── tinker_client.py      # Tinker API wrapper
├── training_scheduler.py # Batch job scheduling
├── evaluator.py          # A/B testing framework
├── model_registry.py     # Adapter management
└── exporters/
    ├── base.py           # Base exporter class
    ├── sft_exporter.py   # SFT data export
    ├── dpo_exporter.py   # DPO data export
    └── gauntlet_exporter.py  # Adversarial data
```

## See Also

- [Tinker Documentation](https://thinkingmachines.ai/tinker/)
- [Training Plan](/Users/armand/.claude/plans/rustling-enchanting-key.md)
- [Agent Selection Guide](docs/AGENT_SELECTION.md)
