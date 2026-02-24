# Workflow Automation

A 3-step automated workflow using Aragora's WorkflowEngine to gather context,
run a multi-agent debate, and produce a structured report.

## How it works

The workflow executes three steps in sequence:

```
[Gather Context] --> [Multi-Agent Debate] --> [Generate Report]
```

1. **Gather** -- An agent collects background information, constraints,
   and stakeholder concerns for the given topic.
2. **Debate** -- Multiple agents debate the best approach, considering
   trade-offs, risks, and feasibility.
3. **Report** -- An agent compiles the debate outcome into a structured
   report with recommendations and implementation steps.

The WorkflowEngine handles step transitions, timeouts, and event callbacks.

## Setup

```bash
# Install aragora
pip install aragora

# Set at least one API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# Basic usage
python examples/workflow_automation/main.py --topic "API rate limiting strategy"

# More debate rounds for thorough analysis
python examples/workflow_automation/main.py --topic "Database migration plan" --rounds 5

# JSON output for programmatic consumption
python examples/workflow_automation/main.py --topic "Microservice decomposition" --json
```

## Key concepts

### WorkflowDefinition

Declarative workflow configuration with steps, transitions, and metadata.
Can be serialized to YAML/JSON for configuration-driven execution.

### StepDefinition

Each step has a type (`agent`, `debate`, `parallel`, etc.), configuration,
timeout, and links to next steps. The engine resolves step types from its
built-in registry.

### TransitionRule

Conditional transitions between steps. Conditions are evaluated against
step outputs, enabling dynamic workflow routing.

### WorkflowEngine

The runtime that executes workflow definitions. Supports checkpointing,
parallel execution, event callbacks, and timeout management.

## Customization

Edit `main.py` to:
- Add more steps (e.g., a validation step after the report)
- Use conditional transitions based on debate confidence
- Enable checkpointing for long-running workflows
- Add parallel steps for gathering from multiple sources
