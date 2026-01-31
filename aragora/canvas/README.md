# Canvas System

Interactive visual workspace for multi-agent orchestration and collaborative knowledge mapping.

## Overview

The Canvas system provides a real-time collaborative workspace where:

- **Agents visualize debates** with live updates showing proposals, critiques, and consensus
- **Users build workflows** by dragging and connecting nodes on the canvas
- **Agents create UIs** using primitives like buttons, forms, and charts
- **Multiple users collaborate** with synchronized state via WebSocket

## Architecture

```
aragora/canvas/
    __init__.py     # Package exports and module documentation
    models.py       # Core data structures (Canvas, CanvasNode, CanvasEdge)
    primitives.py   # A2UI components (Button, Form, Chart, etc.)
    renderer.py     # Export to JSON, SVG, Mermaid formats
    manager.py      # State management and real-time sync
```

## Node Types

### Core Nodes

| Type | Description |
|------|-------------|
| `AGENT` | AI agent (Claude, GPT-4, Gemini, etc.) |
| `DEBATE` | Active debate session |
| `KNOWLEDGE` | Knowledge Mound item |
| `WORKFLOW` | Workflow step |
| `DECISION` | Decision node |
| `EVIDENCE` | Evidence item |
| `CONNECTOR` | External integration |
| `BROWSER` | Browser automation |

### A2UI Primitives

Agent-to-User Interface primitives enable agents to create interactive UIs:

| Primitive | Description |
|-----------|-------------|
| `ButtonPrimitive` | Clickable button with action callback |
| `FormPrimitive` | Form with multiple input fields |
| `SelectPrimitive` | Dropdown selection control |
| `ChartPrimitive` | Data visualization (line, bar, pie, etc.) |
| `ProgressPrimitive` | Progress indicator for long operations |
| `AlertPrimitive` | Notification display (info, success, warning, error) |
| `CardPrimitive` | Information card with title, content, actions |
| `TablePrimitive` | Data table with sorting and pagination |

## Quick Start

### Basic Canvas Operations

```python
from aragora.canvas import Canvas, CanvasNodeType, Position, EdgeType

# Create a canvas
canvas = Canvas(id="my-canvas", name="Debate Workspace")

# Add an agent node
agent = canvas.add_node(
    CanvasNodeType.AGENT,
    position=Position(100, 100),
    label="Claude",
    data={"model": "claude-3-opus"},
)

# Add a debate node
debate = canvas.add_node(
    CanvasNodeType.DEBATE,
    position=Position(300, 100),
    label="Should we adopt microservices?",
)

# Connect them with an edge
canvas.add_edge(
    agent.id,
    debate.id,
    edge_type=EdgeType.DATA_FLOW,
    label="participates in",
)

# Get all agent nodes
agents = canvas.get_nodes_by_type(CanvasNodeType.AGENT)
```

### Real-Time Collaboration

```python
from aragora.canvas import get_canvas_manager, CanvasNodeType, Position

# Get the global manager
manager = get_canvas_manager()

# Create a canvas
canvas = await manager.create_canvas(
    name="Team Canvas",
    owner_id="user-123",
    workspace_id="workspace-456",
)

# Add a node (broadcasts to all subscribers)
node = await manager.add_node(
    canvas.id,
    CanvasNodeType.AGENT,
    Position(100, 100),
    label="Claude",
    user_id="user-123",
)

# Subscribe to canvas events
async def on_canvas_event(event):
    print(f"Event: {event.event_type.value}")
    print(f"Data: {event.data}")

await manager.subscribe(canvas.id, on_canvas_event)

# Move a node (subscribers receive NODE_MOVE event)
await manager.move_node(canvas.id, node.id, 200, 200, user_id="user-123")
```

### Using A2UI Primitives

```python
from aragora.canvas import (
    ButtonPrimitive,
    ButtonVariant,
    FormPrimitive,
    FormField,
    ChartPrimitive,
    ChartType,
    ChartSeries,
    ChartDataPoint,
    AlertPrimitive,
    AlertSeverity,
)

# Create an interactive button
button = ButtonPrimitive(
    label="Run Analysis",
    action="run_analysis",
    variant=ButtonVariant.PRIMARY,
    tooltip="Analyze the selected data",
)

# Create a form for user input
form = FormPrimitive(
    fields=[
        FormField(name="topic", field_type="text", label="Debate Topic", required=True),
        FormField(name="rounds", field_type="number", label="Rounds", default_value=3),
        FormField(
            name="model",
            field_type="select",
            label="Model",
            options=[
                {"value": "claude", "label": "Claude"},
                {"value": "gpt4", "label": "GPT-4"},
            ],
        ),
    ],
    submit_action="start_debate",
    title="Configure Debate",
)

# Create a chart
chart = ChartPrimitive(
    chart_type=ChartType.LINE,
    series=[
        ChartSeries(
            name="ELO Score",
            data=[
                ChartDataPoint(x="Round 1", y=1200),
                ChartDataPoint(x="Round 2", y=1350),
                ChartDataPoint(x="Round 3", y=1500),
            ],
            color="#3498db",
        )
    ],
    title="Agent Performance",
    x_axis_label="Round",
    y_axis_label="ELO",
)

# Show an alert
alert = AlertPrimitive(
    message="Debate completed successfully!",
    severity=AlertSeverity.SUCCESS,
    title="Complete",
    auto_dismiss_seconds=5,
)

# Serialize for web client
button_data = button.to_dict()
```

### Rendering Canvas

```python
from aragora.canvas import Canvas, CanvasRenderer

canvas = Canvas(id="demo")
# ... add nodes and edges ...

renderer = CanvasRenderer(canvas)

# Export as JSON for React Flow or similar
json_data = renderer.to_json()

# Export as SVG image
svg_string = renderer.to_svg(width=1200, height=800)

# Export as Mermaid diagram
mermaid_syntax = renderer.to_mermaid()
print(mermaid_syntax)
# Output:
# graph LR
#     node-1[Claude]
#     node-2[Debate: Microservices?]
#     node-1 -->|participates in| node-2
```

## Event Types

Canvas events are broadcast via WebSocket for real-time updates:

| Category | Events |
|----------|--------|
| Connection | `CONNECT`, `DISCONNECT` |
| State | `STATE`, `SYNC`, `CANVAS_UPDATE` |
| Node | `NODE_CREATE`, `NODE_UPDATE`, `NODE_MOVE`, `NODE_RESIZE`, `NODE_DELETE`, `NODE_SELECT` |
| Edge | `EDGE_CREATE`, `EDGE_UPDATE`, `EDGE_DELETE` |
| Action | `ACTION`, `ACTION_RESULT` |
| Agent | `AGENT_MESSAGE`, `AGENT_THINKING`, `AGENT_COMPLETE` |
| Debate | `DEBATE_START`, `DEBATE_ROUND`, `DEBATE_VOTE`, `DEBATE_CONSENSUS`, `DEBATE_END` |
| A2UI | `BUTTON_CLICK`, `FORM_SUBMIT`, `SELECT_CHANGE`, `PROGRESS_UPDATE`, `ALERT_DISMISS` |
| Error | `ERROR` |

## Edge Types

Edges represent relationships between nodes:

| Type | Use Case |
|------|----------|
| `DEFAULT` | Generic connection |
| `DATA_FLOW` | Data flows from source to target |
| `CONTROL_FLOW` | Execution order dependency |
| `REFERENCE` | Source references target |
| `DEPENDENCY` | Source depends on target |
| `CRITIQUE` | Source critiques target (debate) |
| `SUPPORT` | Source supports target (debate) |

## Integration Points

- **WebSocket Handlers**: `aragora.server.stream.canvas_stream` provides WebSocket endpoints
- **Debate Engine**: Canvas can trigger debates via `execute_action("start_debate", ...)`
- **Workflow Engine**: Canvas can run workflows via `execute_action("run_workflow", ...)`
- **Knowledge Mound**: Canvas can query knowledge via `execute_action("query_knowledge", ...)`

## API Reference

See the module docstrings for detailed API documentation:

- `aragora.canvas.models` - Core data structures
- `aragora.canvas.primitives` - A2UI components
- `aragora.canvas.renderer` - Export formats
- `aragora.canvas.manager` - State management
