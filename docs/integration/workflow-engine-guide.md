# Workflow Engine Guide

This guide covers the DAG-based workflow engine for orchestrating multi-step processes across the Aragora platform.

## Overview

The Workflow Engine enables:
- **DAG-based execution**: Define workflows as directed acyclic graphs
- **Step types**: Tasks, decisions, debates, memory operations
- **Checkpointing**: Resume long-running workflows
- **Templates**: Pre-built workflows for common use cases

## Quick Start

### Creating a Simple Workflow

```python
from aragora.workflow import (
    WorkflowEngine,
    WorkflowDefinition,
    StepDefinition,
    ExecutionPattern,
)

# Define workflow steps
steps = [
    StepDefinition(
        id="research",
        step_type="task",
        config={"action": "search", "query": "topic"},
        next_steps=["analyze"],
    ),
    StepDefinition(
        id="analyze",
        step_type="task",
        config={"action": "analyze_results"},
        next_steps=["report"],
    ),
    StepDefinition(
        id="report",
        step_type="task",
        config={"action": "generate_report"},
        next_steps=[],
    ),
]

# Create workflow definition
workflow = WorkflowDefinition(
    id="research-workflow",
    name="Research Pipeline",
    steps=steps,
    entry_step="research",
)

# Execute workflow
engine = WorkflowEngine()
result = await engine.execute(workflow, context={"topic": "AI safety"})
```

### Using Templates

```python
from aragora.workflow import TemplateLoader

loader = TemplateLoader()

# List available templates
templates = loader.list_templates()
# ['legal_contract_review', 'healthcare_hipaa', 'software_code_review', ...]

# Load and instantiate a template
workflow = loader.load_template(
    "legal_contract_review",
    variables={"document_id": "doc-123"},
)

# Execute
engine = WorkflowEngine()
result = await engine.execute(workflow)
```

## Workflow Definition

### Step Types

#### Task Steps

Basic task execution:

```python
StepDefinition(
    id="my_task",
    step_type="task",
    config={
        "action": "log",
        "message": "Hello from workflow",
    },
)
```

Available task actions:
- `log` - Log a message
- `set_state` - Set workflow state
- `delay` - Wait for duration
- Custom actions via task registry

#### Decision Steps

Conditional branching:

```python
StepDefinition(
    id="check_approval",
    step_type="decision",
    config={
        "condition": "context.approval_status == 'approved'",
        "true_step": "proceed",
        "false_step": "request_review",
    },
)
```

#### Debate Steps

Multi-agent debate integration:

```python
StepDefinition(
    id="analyze_contract",
    step_type="debate",
    config={
        "topic": "Analyze contract for risks",
        "agents": ["claude", "gpt-4"],
        "rounds": 3,
        "consensus_threshold": 0.8,
    },
)
```

#### Memory Steps

Memory operations:

```python
StepDefinition(
    id="store_result",
    step_type="memory",
    config={
        "operation": "store",
        "tier": "slow",
        "key": "analysis_result",
        "value_from": "previous_step.output",
    },
)
```

### Transitions

Define conditional transitions between steps:

```python
from aragora.workflow import TransitionRule

transitions = [
    TransitionRule(
        from_step="analyze",
        to_step="high_risk_review",
        condition="result.risk_score > 0.8",
        priority=1,
    ),
    TransitionRule(
        from_step="analyze",
        to_step="normal_review",
        condition="result.risk_score <= 0.8",
        priority=2,
    ),
]

workflow = WorkflowDefinition(
    ...,
    transitions=transitions,
)
```

### Execution Patterns

#### Sequential Execution

Steps execute one after another (default):

```python
StepDefinition(
    id="step1",
    execution_pattern=ExecutionPattern.SEQUENTIAL,
    next_steps=["step2"],
)
```

#### Parallel Execution

Multiple steps execute concurrently:

```python
StepDefinition(
    id="fan_out",
    execution_pattern=ExecutionPattern.PARALLEL,
    next_steps=["task_a", "task_b", "task_c"],
)
```

## Persistence

### Workflow Store

```python
from aragora.workflow import PersistentWorkflowStore

store = PersistentWorkflowStore(db_path="workflows.db")

# Save workflow definition
await store.save_workflow(workflow)

# List workflows
workflows, total = await store.list_workflows(limit=10, offset=0)

# Load workflow
loaded = await store.get_workflow("research-workflow")

# Get workflow versions
versions = await store.get_workflow_versions("research-workflow")
```

### Checkpointing

For long-running workflows:

```python
from aragora.workflow import EnhancedWorkflowEngine

engine = EnhancedWorkflowEngine(
    checkpoint_interval=60,  # Save every 60 seconds
    checkpoint_store=store,
)

# Execute with checkpointing
result = await engine.execute(workflow)

# Resume from checkpoint
result = await engine.resume("execution-id")
```

## Enhanced Engine Features

### Resource Limits

```python
from aragora.workflow import EnhancedWorkflowEngine, ResourceLimits

limits = ResourceLimits(
    max_tokens=100000,
    max_cost=10.0,  # USD
    max_time=3600,  # Seconds
)

engine = EnhancedWorkflowEngine(resource_limits=limits)
result = await engine.execute(workflow)
```

### Metrics Callbacks

```python
def on_step_complete(step_id: str, duration: float, result: Any):
    logger.info(f"Step {step_id} completed in {duration}s")

engine = EnhancedWorkflowEngine()
engine.on_step_complete(on_step_complete)
```

## Industry Templates

Pre-built templates in `aragora/workflow/templates/`:

### Legal
- `legal_contract_review` - Contract analysis workflow
- `legal_due_diligence` - Due diligence process

### Healthcare
- `healthcare_hipaa_compliance` - HIPAA compliance check
- `healthcare_clinical_review` - Clinical review process

### Software
- `software_code_review` - Code review workflow
- `software_security_audit` - Security audit process

### Accounting
- `accounting_financial_audit` - Financial audit workflow

### Academic
- `academic_citation_verification` - Citation verification

### Regulatory
- `regulatory_compliance_assessment` - Compliance assessment

### Loading Templates

```python
# Load by category
templates = loader.list_templates(category="legal")

# Load with customization
workflow = loader.load_template(
    "software_code_review",
    variables={
        "repository": "github.com/org/repo",
        "branch": "main",
        "reviewers": ["claude", "gpt-4"],
    },
)
```

## Custom Step Types

### Registering Custom Steps

```python
from aragora.workflow.nodes import register_step_handler

@register_step_handler("my_custom_step")
async def handle_custom_step(config: dict, context: dict) -> dict:
    # Implement custom logic
    result = await do_something(config["param"])
    return {"output": result}

# Use in workflow
StepDefinition(
    id="custom",
    step_type="my_custom_step",
    config={"param": "value"},
)
```

## Error Handling

### Step Retries

```python
StepDefinition(
    id="flaky_step",
    step_type="task",
    config={...},
    retry_count=3,
    retry_delay=5.0,
)
```

### Error Callbacks

```python
async def on_error(step_id: str, error: Exception):
    await notify_admin(f"Step {step_id} failed: {error}")

engine.on_error(on_error)
```

### Timeout Handling

```python
StepDefinition(
    id="slow_step",
    step_type="task",
    config={...},
    timeout=300,  # 5 minutes
)
```

## API Reference

Core modules:
- `aragora/workflow/engine.py` - Basic engine
- `aragora/workflow/engine_v2.py` - Enhanced engine
- `aragora/workflow/types.py` - Type definitions
- `aragora/workflow/persistent_store.py` - Persistence
- `aragora/workflow/template_loader.py` - Template management
- `aragora/workflow/nodes/` - Step type implementations
