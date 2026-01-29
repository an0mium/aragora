# Computer Use Guide

Safe computer-use automation with Claude Computer Use API integration.

## Overview

The `aragora.computer_use` module provides controlled access to computer-use capabilities:

- **Action definitions**: Screenshot, click, type, scroll, key presses
- **Policy enforcement**: Action allowlists, domain restrictions, credential blocking
- **Multi-turn orchestration**: Agentic loops with validation and audit trails
- **Safety features**: Coordinate bounds, rate limiting, human approval callbacks

## Quick Start

```python
from aragora.computer_use import (
    ComputerUseOrchestrator,
    ComputerPolicy,
    MockActionExecutor,
    create_default_computer_policy,
)

# Create orchestrator with default policy
policy = create_default_computer_policy()
executor = MockActionExecutor()  # Replace with real executor
orchestrator = ComputerUseOrchestrator(
    executor=executor,
    policy=policy,
)

# Execute a task
result = await orchestrator.run_task(
    goal="Open settings and enable dark mode",
    max_steps=10,
)

print(f"Status: {result.status}")
print(f"Steps: {len(result.steps)}")
```

## Core Concepts

### Actions

Actions are the atomic operations that can be executed:

| Action | Description | Tool Input |
|--------|-------------|------------|
| `ScreenshotAction` | Capture current screen | `{"action": "screenshot"}` |
| `ClickAction` | Click at coordinates | `{"action": "click", "coordinate": [x, y]}` |
| `TypeAction` | Type text | `{"action": "type", "text": "..."}` |
| `KeyAction` | Press special key | `{"action": "key", "text": "Return"}` |
| `ScrollAction` | Scroll screen | `{"action": "scroll", "coordinate": [0, delta]}` |
| `DragAction` | Drag between points | `{"action": "drag", ...}` |
| `WaitAction` | Pause execution | Custom wait logic |

```python
from aragora.computer_use import ClickAction, TypeAction, Keys

# Create actions
click = ClickAction(x=100, y=200)
type_text = TypeAction(text="Hello, World!")
key_press = KeyAction(key=Keys.ENTER)

# Convert to Claude tool input
print(click.to_tool_input())  # {"action": "click", "coordinate": [100, 200]}
```

### Policies

Policies control what actions are allowed:

```python
from aragora.computer_use import (
    ComputerPolicy,
    ComputerPolicyChecker,
    create_default_computer_policy,
    create_strict_computer_policy,
    create_readonly_computer_policy,
)

# Default policy: allows standard actions, blocks credentials
policy = create_default_computer_policy()

# Strict policy: audits all actions, lower limits
strict = create_strict_computer_policy()

# Readonly policy: only screenshot/scroll, no clicks/typing
readonly = create_readonly_computer_policy()

# Custom policy
custom = ComputerPolicy(name="custom")
custom.add_action_allowlist([ActionType.CLICK, ActionType.TYPE])
custom.add_domain_allowlist([r"^https://trusted\.com.*$"])
custom.add_sensitive_text_patterns([r"password", r"secret"])
```

### Policy Checker

The policy checker validates actions before execution:

```python
from aragora.computer_use import ComputerPolicyChecker, ClickAction

checker = ComputerPolicyChecker(policy)

# Check an action
action = ClickAction(x=100, y=200)
allowed, reason = checker.check_action(action, current_url="http://localhost:8080")

if not allowed:
    print(f"Blocked: {reason}")
```

### Orchestrator

The orchestrator manages multi-turn computer-use sessions:

```python
from aragora.computer_use import (
    ComputerUseOrchestrator,
    ComputerUseConfig,
)

config = ComputerUseConfig(
    model="claude-sonnet-4-20250514",
    max_steps=50,
    action_timeout_seconds=10.0,
    total_timeout_seconds=300.0,
    take_screenshot_after_action=True,
)

orchestrator = ComputerUseOrchestrator(
    executor=executor,
    policy=policy,
    config=config,
)

# Run task with metadata
result = await orchestrator.run_task(
    goal="Navigate to settings page",
    max_steps=10,
    initial_context="User is on the home page",
    metadata={"user_id": "123", "session_id": "abc"},
)

# Check metrics
print(orchestrator.metrics.to_dict())
```

## Safety Features

### Credential Blocking

Default policy blocks common credential patterns:

```python
# These patterns are blocked by default:
# - password = xxx
# - api_key = xxx
# - Bearer tokens
# - OpenAI API keys (sk-...)
# - GitHub tokens (ghp_...)
```

### Coordinate Bounds

Actions outside screen bounds are rejected:

```python
# Default bounds: 0-3840 (x), 0-2160 (y)
action = ClickAction(x=5000, y=100)
allowed, reason = checker.check_action(action)
# False, "X coordinate 5000 outside bounds [0, 3840]"
```

### Rate Limiting

Configurable rate limits prevent runaway automation:

```python
from aragora.computer_use import RateLimits

limits = RateLimits(
    max_actions_per_minute=60,
    max_clicks_per_minute=30,
    max_keystrokes_per_minute=300,
    cooldown_after_error_seconds=2.0,
)
```

### Human Approval

Optional callback for human-in-the-loop approval:

```python
def approval_callback(action):
    # Custom logic to approve/deny
    if action.action_type == ActionType.TYPE:
        return confirm_with_user(action.text)
    return True

config = ComputerUseConfig(
    require_approval_callback=approval_callback,
)
```

## API Reference

### ComputerUseOrchestrator

Main entry point for computer-use automation.

| Method | Description |
|--------|-------------|
| `run_task(goal, max_steps, ...)` | Execute a computer-use task |
| `cancel_task()` | Cancel the current running task |
| `get_audit_log()` | Get the policy audit log |

### ComputerPolicyChecker

Policy validation engine.

| Method | Description |
|--------|-------------|
| `check_action(action, current_url)` | Validate an action |
| `record_success()` | Record successful action |
| `record_error()` | Record failed action |
| `reset()` | Reset session state |
| `get_stats()` | Get session statistics |

### Action Classes

| Class | Parameters |
|-------|------------|
| `ScreenshotAction` | None |
| `ClickAction` | `x, y, button, double_click` |
| `TypeAction` | `text` |
| `KeyAction` | `key` |
| `ScrollAction` | `direction, amount, x, y` |
| `MoveAction` | `x, y` |
| `DragAction` | `start_x, start_y, end_x, end_y` |
| `WaitAction` | `duration_ms, wait_for` |

## Integration

### With Agent Fabric

```python
from aragora.fabric import AgentFabric
from aragora.computer_use import ComputerUseOrchestrator

# Register computer-use as an agent tool
fabric = AgentFabric()
orchestrator = ComputerUseOrchestrator(...)

# Computer-use actions can be scheduled through fabric
```

### With Workspace

```python
from aragora.workspace import WorkspaceManager
from aragora.computer_use import ComputerUseOrchestrator

# Create bead for computer-use task
ws = WorkspaceManager()
convoy = await ws.create_convoy(
    rig_id="rig-123",
    name="UI Automation",
    bead_specs=[{
        "title": "Navigate to settings",
        "payload": {"goal": "Open settings page"},
    }],
)
```

## Examples

### Basic Screenshot

```python
from aragora.computer_use import ScreenshotAction, MockActionExecutor

executor = MockActionExecutor()
action = ScreenshotAction()
result = await executor.execute(action)

print(f"Success: {result.success}")
print(f"Screenshot captured: {result.screenshot_b64 is not None}")
```

### Form Filling

```python
from aragora.computer_use import (
    ClickAction,
    TypeAction,
    KeyAction,
    Keys,
)

# Click on input field
click = ClickAction(x=200, y=150)

# Type username
type_user = TypeAction(text="johndoe")

# Press Tab to next field
tab = KeyAction(key=Keys.TAB)

# Type password (blocked by policy if matches credential pattern)
type_pass = TypeAction(text="mysecretpassword")  # Will be blocked!

# Submit
submit = KeyAction(key=Keys.ENTER)
```

### Custom Policy

```python
from aragora.computer_use import (
    ComputerPolicy,
    ActionType,
    PolicyDecision,
)

policy = ComputerPolicy(
    name="restricted",
    description="Restricted policy for specific domain",
    default_decision=PolicyDecision.DENY,
    max_actions_per_task=20,
    timeout_per_action_seconds=5.0,
)

# Only allow certain domains
policy.add_domain_allowlist(
    [r"^https://internal\.company\.com.*$"],
    reason="Internal only",
)

# Allow observation and typing
policy.add_action_allowlist(
    [ActionType.SCREENSHOT, ActionType.SCROLL, ActionType.TYPE],
    reason="Read and input only",
)
```

---

*Part of Aragora control plane for multi-agent robust decisionmaking*
