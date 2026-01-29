# Workflow - DAG-Based Automation Engine

Declarative, multi-step orchestration for automating complex multi-agent processes with sequential, parallel, and conditional execution patterns.

## Quick Start

```python
from aragora.workflow import WorkflowEngine, WorkflowDefinition, StepDefinition

definition = WorkflowDefinition(
    id="review_pipeline",
    name="Code Review Pipeline",
    steps=[
        StepDefinition(
            id="extract",
            name="Extract Issues",
            step_type="agent",
            config={"agent": "claude", "prompt": "Find code issues"},
            next_steps=["analyze"]
        ),
        StepDefinition(
            id="analyze",
            name="Risk Analysis",
            step_type="debate",
            config={"agents": ["claude", "gpt4"]}
        )
    ],
    entry_step="extract"
)

engine = WorkflowEngine()
result = await engine.execute(definition, inputs={"code": "..."})
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `WorkflowEngine` | `engine.py` | Main execution runtime |
| `WorkflowDefinition` | `types.py` | Workflow structure |
| `StepDefinition` | `types.py` | Step configuration |
| `CheckpointStore` | `checkpoint_store.py` | State persistence |
| `TemplateLoader` | `template_loader.py` | YAML template loading |

## Architecture

```
workflow/
├── engine.py             # Main execution engine
├── engine_v2.py          # Resource-aware execution
├── types.py              # Core type definitions
├── step.py               # Step protocol and base implementations
├── checkpoint_store.py   # Persistence (File, Redis, Postgres)
├── persistent_store.py   # Workflow storage
├── template_loader.py    # YAML template loading
├── schema.py             # Validation
├── safe_eval.py          # Secure condition evaluation
├── nodes/                # 17 step type implementations
│   ├── agent.py          # AI agent execution
│   ├── debate.py         # Aragora debate
│   ├── parallel.py       # Concurrent execution
│   ├── conditional.py    # Branching logic
│   ├── human_checkpoint.py  # Human approval
│   └── connector.py      # 100+ integrations
├── patterns/             # 8 reusable patterns
│   ├── hivemind.py       # Parallel consensus
│   ├── sequential.py     # Linear pipeline
│   ├── mapreduce.py      # Split-process-aggregate
│   └── review_cycle.py   # Iterative refinement
└── templates/            # Industry templates
    ├── legal/
    ├── healthcare/
    ├── finance/
    └── code/
```

## Execution Patterns

### Sequential
```python
from aragora.workflow.patterns import SequentialPattern

pattern = SequentialPattern(
    agents=["claude", "gpt4"],
    task="Review this code"
)
workflow = pattern.create_workflow()
```

### Parallel (HiveMind)
```python
from aragora.workflow.patterns import HiveMindPattern

pattern = HiveMindPattern(
    agents=["claude", "gpt4", "gemini"],
    merge_mode="weighted"
)
```

### Map-Reduce
```python
from aragora.workflow.patterns import MapReducePattern

pattern = MapReducePattern(
    mapper_agent="claude",
    reducer_agent="gpt4",
    split_strategy="chunks"
)
```

### Review Cycle
```python
from aragora.workflow.patterns import ReviewCyclePattern

pattern = ReviewCyclePattern(
    max_iterations=5,
    convergence_threshold=0.9
)
```

## Step Types

| Type | Purpose |
|------|---------|
| `agent` | Execute AI agent task |
| `debate` | Run Aragora debate |
| `parallel` | Concurrent execution |
| `conditional` | Branch on condition |
| `loop` | Iterate until convergence |
| `human_checkpoint` | Human approval gate |
| `memory_read/write` | Knowledge Mound access |
| `connector` | External integrations |
| `gauntlet` | Adversarial validation |
| `decision/switch` | Multi-way branching |
| `browser` | Web automation |

## Status Flow

```
PENDING → RUNNING → COMPLETED
             ↓
   FAILED / SKIPPED / WAITING
```

## Checkpointing

```python
engine = WorkflowEngine(
    checkpoint_interval=5,  # Every 5 steps
    checkpoint_store=RedisCheckpointStore(redis_url)
)

# Resume from checkpoint
result = await engine.resume(workflow_id, checkpoint_id)
```

## Templates by Industry

| Category | Templates |
|----------|-----------|
| Legal | Contract review, due diligence |
| Healthcare | Clinical review, HIPAA compliance |
| Finance | Investment analysis, trading |
| Accounting | Financial workflows |
| Code | Development, DevOps |
| Compliance | Regulatory workflows |
| SME | Business decisions |

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Control Plane](../control_plane/README.md) - Task scheduling
- [Connectors](../connectors/README.md) - External integrations
