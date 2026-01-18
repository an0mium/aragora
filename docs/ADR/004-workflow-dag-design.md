# ADR-004: Workflow DAG Design

## Status
Accepted

## Context
The Enterprise Multi-Agent Control Plane needed a generalized workflow system that:
- Supports arbitrary multi-step workflows beyond debates
- Enables parallel and conditional execution
- Provides checkpointing for long-running tasks
- Integrates with the Knowledge Mound system

The existing PhaseExecutor pattern was too debate-specific.

## Decision
We implemented a **DAG-based Workflow Engine** with the following design:

### Core Components

**WorkflowDefinition** (`aragora/workflow/types.py`):
```python
@dataclass
class WorkflowDefinition:
    id: str
    name: str
    steps: List[StepDefinition]
    transitions: List[TransitionRule]
    entry_step: str
```

**StepDefinition**:
```python
@dataclass
class StepDefinition:
    id: str
    step_type: str  # "task", "agent", "debate", "decision"
    config: Dict[str, Any]
    next_steps: List[str]  # Default transitions
    execution_pattern: ExecutionPattern  # SEQUENTIAL, PARALLEL
```

**TransitionRule**:
```python
@dataclass
class TransitionRule:
    from_step: str
    to_step: str
    condition: str  # Python expression
    priority: int
```

### Execution Engines

**WorkflowEngine** (`aragora/workflow/engine.py`):
- Basic engine with sequential/parallel execution
- Checkpointing support
- Step type registry

**EnhancedWorkflowEngine** (`aragora/workflow/engine_v2.py`):
- Resource limits (tokens, cost, time)
- Metrics callbacks
- Enhanced error handling

### Step Types
Located in `aragora/workflow/nodes/`:
- `task.py` - Basic task execution
- `decision.py` - Conditional branching
- `debate.py` - Multi-agent debate step
- `memory.py` - Memory operations
- `knowledge_pipeline.py` - Knowledge Mound integration

### Persistence
**PersistentWorkflowStore** (`aragora/workflow/persistent_store.py`):
- SQLite-backed storage
- Workflow versioning
- Execution history

### Templates
Located in `aragora/workflow/templates/`:
- Industry-specific templates (legal, healthcare, etc.)
- YAML-based definitions
- Template loader for easy instantiation

## Consequences
**Positive:**
- Highly flexible workflow composition
- Reusable across enterprise verticals
- Checkpointing enables resumption
- Templates accelerate deployment

**Negative:**
- Complexity in DAG cycle detection
- Transition conditions evaluated at runtime (security concern)
- Large codebase (660K lines in control plane)

## References
- `aragora/workflow/engine.py` - Core engine
- `aragora/workflow/engine_v2.py` - Enhanced engine
- `aragora/workflow/types.py` - Type definitions
- `aragora/workflow/templates/` - Industry templates
- `docs/WORKFLOWS.md` - Workflow documentation
