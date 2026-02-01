# OpenClaw Integration Guide

OpenClaw is Aragora's enterprise gateway for computer-use AI agents, providing secure, policy-controlled execution of shell commands, file operations, and browser automation.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Aragora Server                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐    ┌───────────────────┐    ┌────────────────────┐    │
│   │ Aragora SDK │───>│ OpenClaw Gateway  │───>│ OpenClaw Proxy     │    │
│   │  (client)   │    │ Handler           │    │                    │    │
│   └─────────────┘    │ - Sessions        │    │ - Policy Engine    │    │
│                      │ - Actions         │    │ - Approval Queue   │    │
│   ┌─────────────┐    │ - Credentials     │    │ - Audit Log        │    │
│   │ Workflow    │───>│ - Policies        │    │ - Action Executor  │    │
│   │ Engine      │    └───────────────────┘    └────────────────────┘    │
│   └─────────────┘                                       │               │
│                                                         v               │
│                                          ┌────────────────────────────┐ │
│                                          │    Container / Host        │ │
│                                          │    - Shell execution       │ │
│                                          │    - File system access    │ │
│                                          │    - Browser automation    │ │
│                                          └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Variables

```bash
# OpenClaw gateway configuration
export OPENCLAW_ENABLED=true
export OPENCLAW_PROXY_URL=http://localhost:8100  # OpenClaw server
export OPENCLAW_DEFAULT_TIMEOUT=60               # Action timeout (seconds)
export OPENCLAW_REQUIRE_APPROVAL=false           # Default approval mode
```

### 2. Start the Server

```bash
python -m aragora.server.unified_server --port 8080
```

The OpenClaw gateway is available at `/api/gateway/openclaw/` and `/api/v1/openclaw/`.

## SDK Usage

### Initialize Client

```python
from aragora import AragoraClient

client = AragoraClient(base_url="http://localhost:8080")
```

### Session Management

```python
# Create a session
session = client.openclaw.create_session(
    user_id="user-123",
    tenant_id="tenant-abc",
    workspace_id="/workspace",
    roles=["developer", "admin"]
)
print(f"Session: {session.session_id}")

# Execute actions
result = client.openclaw.execute_action(
    session_id=session.session_id,
    action_type="shell",
    command="ls -la /workspace"
)
if result.success:
    print(result.result)

# End session when done
client.openclaw.end_session(session.session_id)
```

### Action Types

```python
# Shell command
client.openclaw.execute_action(
    session_id=session_id,
    action_type="shell",
    command="python script.py"
)

# Read file
client.openclaw.execute_action(
    session_id=session_id,
    action_type="file_read",
    path="/workspace/config.json"
)

# Write file
client.openclaw.execute_action(
    session_id=session_id,
    action_type="file_write",
    path="/workspace/output.txt",
    content="Hello, World!"
)

# Browser automation
client.openclaw.execute_action(
    session_id=session_id,
    action_type="browser",
    url="https://example.com",
    params={"action": "click", "selector": "#submit"}
)

# Screenshot
client.openclaw.execute_action(
    session_id=session_id,
    action_type="screenshot",
    url="https://example.com"
)
```

### Async Usage

```python
import asyncio

async def run_agent():
    session = await client.openclaw.create_session_async(
        user_id="user-123",
        tenant_id="tenant-abc"
    )

    result = await client.openclaw.execute_action_async(
        session_id=session.session_id,
        action_type="shell",
        command="echo 'Hello from async!'"
    )

    await client.openclaw.end_session_async(session.session_id)
    return result

result = asyncio.run(run_agent())
```

## API Reference

### Session Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/openclaw/sessions` | Create session |
| GET | `/api/v1/openclaw/sessions` | List sessions |
| GET | `/api/v1/openclaw/sessions/:id` | Get session |
| POST | `/api/v1/openclaw/sessions/:id/end` | End session |
| DELETE | `/api/v1/openclaw/sessions/:id` | Close session |

### Action Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/openclaw/actions` | Execute action |
| GET | `/api/v1/openclaw/actions/:id` | Get action status |
| POST | `/api/v1/openclaw/actions/:id/cancel` | Cancel action |

### Policy Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/openclaw/policy` | Get current policy |
| GET | `/api/v1/openclaw/policy/rules` | List policy rules |
| POST | `/api/v1/openclaw/policy/rules` | Create rule |
| DELETE | `/api/v1/openclaw/policy/rules/:name` | Delete rule |

### Approval Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/openclaw/approvals` | List pending approvals |
| POST | `/api/v1/openclaw/approvals/:id/approve` | Approve action |
| POST | `/api/v1/openclaw/approvals/:id/deny` | Deny action |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/openclaw/health` | Gateway health |
| GET | `/api/v1/openclaw/stats` | Gateway statistics |
| GET | `/api/v1/openclaw/audit` | Audit trail |

## Workflow Nodes

Use OpenClaw in Aragora workflows for automated agent pipelines.

### OpenClawSessionStep

Create and manage sessions within workflows:

```python
from aragora.workflow import Workflow
from aragora.workflow.nodes import OpenClawSessionStep, OpenClawActionStep

workflow = Workflow("agent-pipeline")

# Create session
workflow.add_step(OpenClawSessionStep(
    name="create_session",
    config={
        "operation": "create",
        "user_id": "{input.user_id}",
        "tenant_id": "{input.tenant_id}",
        "workspace_id": "/workspace"
    }
))
```

### OpenClawActionStep

Execute actions through the gateway:

```python
# Execute shell command
workflow.add_step(OpenClawActionStep(
    name="run_tests",
    config={
        "action_type": "shell",
        "session_id": "{step.create_session.session_id}",
        "command": "pytest tests/ -v",
        "timeout_seconds": 120,
        "on_failure": "error"
    }
))

# Read results
workflow.add_step(OpenClawActionStep(
    name="read_results",
    config={
        "action_type": "file_read",
        "session_id": "{step.create_session.session_id}",
        "path": "/workspace/test-results.xml"
    }
))

# End session
workflow.add_step(OpenClawSessionStep(
    name="end_session",
    config={
        "operation": "end",
        "session_id": "{step.create_session.session_id}"
    }
))
```

### Failure Handling

Configure how action failures are handled:

```python
OpenClawActionStep(
    name="risky_operation",
    config={
        "action_type": "shell",
        "command": "rm -rf /workspace/temp",
        "on_failure": "skip",    # Options: error, skip, retry
        "require_approval": True # Require human approval
    }
)
```

## Policy Configuration

### Creating Policy Rules

```python
# Deny destructive commands
client.openclaw.create_policy_rule(
    name="block-destructive-rm",
    action_types=["shell"],
    decision="deny",
    priority=100,
    config={
        "pattern": "rm\\s+-rf\\s+/(?!workspace)",
        "reason": "Destructive rm outside workspace"
    }
)

# Require approval for file writes
client.openclaw.create_policy_rule(
    name="approve-file-writes",
    action_types=["file_write", "file_delete"],
    decision="require_approval",
    priority=50,
    config={
        "paths": ["/etc/*", "/usr/*"],
        "reason": "System file modification"
    }
)

# Allow workspace operations
client.openclaw.create_policy_rule(
    name="allow-workspace",
    action_types=["shell", "file_read", "file_write"],
    decision="allow",
    priority=10,
    config={
        "workspace_only": True
    }
)
```

### Policy Decision Order

1. Rules are evaluated by priority (highest first)
2. First matching rule determines the decision
3. If no rule matches, the default policy applies

## SKILL.md Migration

If migrating from a SKILL.md-based agent, map capabilities to OpenClaw actions:

| SKILL.md Capability | OpenClaw Action |
|---------------------|-----------------|
| `bash` | `action_type: "shell"` |
| `computer` | `action_type: "browser"` or `"screenshot"` |
| `text_editor_20241022` | `action_type: "file_read"` / `"file_write"` |

### Example Migration

**Before (SKILL.md):**
```markdown
# Skills
- bash: Execute shell commands
- text_editor_20241022: Read and write files
```

**After (OpenClaw):**
```python
session = client.openclaw.create_session(user_id, tenant_id)

# bash -> shell action
client.openclaw.execute_action(
    session_id=session.session_id,
    action_type="shell",
    command="./build.sh"
)

# text_editor -> file actions
client.openclaw.execute_action(
    session_id=session.session_id,
    action_type="file_read",
    path="/workspace/README.md"
)
```

## Security Considerations

1. **Session Isolation**: Each session runs in an isolated context
2. **Policy Enforcement**: All actions pass through the policy engine
3. **Audit Logging**: Every action is logged with full context
4. **Approval Workflows**: Sensitive actions can require human approval
5. **Credential Rotation**: Stored credentials support automatic rotation

## Troubleshooting

### Common Issues

**Session creation fails:**
- Check `OPENCLAW_ENABLED=true`
- Verify the OpenClaw proxy is running

**Actions timeout:**
- Increase `timeout_seconds` in action config
- Check `OPENCLAW_DEFAULT_TIMEOUT` environment variable

**Actions denied:**
- Review policy rules: `GET /api/v1/openclaw/policy/rules`
- Check audit log for denial reason: `GET /api/v1/openclaw/audit`

**Approval pending indefinitely:**
- List pending approvals: `client.openclaw.list_approvals()`
- Approve manually or configure auto-approval rules
