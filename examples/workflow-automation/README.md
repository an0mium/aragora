# Workflow Automation

A business process automation example using Aragora's WorkflowEngine for DAG-based task orchestration. Demonstrates sequential, parallel, and conditional step execution with checkpointing, retries, and event tracking.

## How It Works

This example implements a **content publishing pipeline** -- a realistic business process with branching logic:

```
           +------------------+
           |   Draft Content  |
           +--------+---------+
                    |
          +---------+---------+
          |                   |
  +-------v------+  +--------v-------+
  | Security     |  | Editorial      |
  | Review       |  | Review         |   (parallel execution)
  +-------+------+  +--------+-------+
          |                   |
          +---------+---------+
                    |
              +-----v-----+
              | Both pass? |  (conditional branching)
              +-----+------+
               /          \
          YES /            \ NO
   +--------v---+    +-----v--------+
   |  Publish   |    |  Request     |
   |  Content   |    |  Revision    |
   +--------+---+    +-----+--------+
            \              /
             +------+-----+
                    |
            +-------v-------+
            |    Notify     |
            | Stakeholders  |
            +---------------+
```

## Quick Start (Demo Mode)

No API keys required:

```bash
python examples/workflow-automation/main.py --demo
```

## Full Setup

### 1. Install Dependencies

```bash
pip install -r examples/workflow-automation/requirements.txt
```

### 2. Set Environment Variables

```bash
# At least one AI provider key (for live agent drafting)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### 3. Run the Workflow

```bash
# Run with a custom topic
python examples/workflow-automation/main.py --topic "API security guidelines"

# Just view the workflow DAG
python examples/workflow-automation/main.py --show-dag

# Demo mode (sample content, no API calls)
python examples/workflow-automation/main.py --demo --topic "Cloud migration checklist"

# JSON output for programmatic use
python examples/workflow-automation/main.py --demo --json
```

## Features

- **DAG execution**: Steps run in dependency order with parallel branches
- **Conditional branching**: Publish or revise based on review outcomes
- **Parallel reviews**: Security and editorial reviews run concurrently
- **Checkpointing**: WorkflowEngine saves state for long-running workflows
- **Retry support**: Configurable per-step retries with backoff
- **Event tracking**: Every step emits events for observability
- **JSON output**: Use `--json` for machine-readable results (piping, CI/CD)
- **Custom steps**: Extend `BaseStep` to add your own workflow actions

## Custom Step Implementation

To add your own steps to a workflow:

```python
from aragora.workflow.step import BaseStep, WorkflowContext

class MyCustomStep(BaseStep):
    @property
    def name(self) -> str:
        return "my_custom_step"

    async def execute(self, context: WorkflowContext) -> dict:
        # Read inputs from previous steps
        prev_output = context.get_step_output("previous_step")

        # Do work
        result = {"processed": True}

        # Emit events for observability
        context.emit_event("custom_step_done", result)

        return result
```

Then register it with the engine:

```python
engine = WorkflowEngine(
    step_registry={"my_custom_step": MyCustomStep}
)
```

## Architecture

The example uses three key Aragora components:

1. **`WorkflowEngine`** -- Executes workflow DAGs with configurable timeout, checkpointing, and parallel execution
2. **`WorkflowDefinition`** / **`StepDefinition`** -- Declarative workflow structure with transitions and conditions
3. **`BaseStep`** -- Protocol for implementing custom step logic

## Output Example

```
======================================================================
  Workflow: Content Publishing Pipeline
  ID: content-publishing-pipeline-a1b2c3
======================================================================

  Status: SUCCESS
  Steps: 5/6 completed, 0 failed, 1 skipped
  Duration: 342ms

  --- Step Results ---

  [OK] Draft Content (draft) -- 12ms
       title: Guide: AI governance best practices

  [OK] Security Review (security_review) -- 105ms
       passed: True
       issues_found: 0

  [OK] Editorial Review (editorial_review) -- 103ms
       passed: True
       score: 1.0

  [OK] Publish Content (publish) -- 8ms
       published: True
       url: https://docs.example.com/guides/guide-ai-governance-best-practices

  [SKIP] Request Revision (request_revision) -- 0ms

  [OK] Notify Stakeholders (notify) -- 5ms
       status: published

  --- Events (5) ---

  [10:30:01] content_drafted: {"title": "Guide: AI governance best practices"}
  [10:30:01] security_review_complete: {"passed": true, "issues": 0}
  [10:30:01] editorial_review_complete: {"passed": true, "score": 1.0}
  [10:30:01] content_published: {"url": "https://docs.example.com/..."}
  [10:30:01] stakeholders_notified: {"status": "published"}

======================================================================
```
