# OpenClaw Integration Guide

## Overview

[OpenClaw](https://github.com/openclaw) is an enterprise gateway for computer-use AI agents. It provides sandboxed environments where AI agents can execute shell commands, read/write files, and interact with browsers under controlled conditions.

Aragora integrates with OpenClaw through a **secure proxy layer** that adds policy-based access control, RBAC enforcement, audit logging, and approval workflows on top of OpenClaw's execution capabilities. This means every action an AI agent attempts -- whether running a shell command or navigating a browser -- is evaluated against configurable security policies before it reaches the OpenClaw backend.

Key capabilities of the integration:

- **Session management** -- create isolated proxy sessions scoped to users and tenants
- **Policy enforcement** -- allow, deny, or require approval for actions based on type, path, command patterns, and user roles
- **Credential management** -- securely store and rotate credentials used by agents
- **Audit trail** -- every action is logged with full context for compliance
- **Workflow integration** -- use OpenClaw actions as steps in Aragora DAG workflows

## Architecture

```
+------------------+       +-------------------------+       +---------------------+
|                  |       |     Aragora Server      |       |                     |
|   Aragora SDK    | HTTP  |  OpenClawGatewayHandler |       |   OpenClaw Server   |
|   (Python)       +------>+          |              +------>+   (Backend)         |
|                  |       |  OpenClawSecureProxy    |       |                     |
| client.openclaw  |       |          |              |       |  Shell / Files /    |
|                  |       |   OpenClawPolicy        |       |  Browser / API      |
+------------------+       +-----+---+---+-----------+       +---------------------+
                                 |   |   |
                           +-----+   |   +------+
                           |         |          |
                      RBAC Check  Policy    Audit Log
                                Evaluation
```

**Request flow:**

1. SDK client calls `/api/v1/openclaw/*` endpoints
2. `OpenClawGatewayHandler` validates the request, checks permissions, and applies rate limits
3. `OpenClawSecureProxy` evaluates the action against `OpenClawPolicy` rules
4. If allowed, the proxy forwards the action to the OpenClaw backend
5. Results and audit entries are returned to the caller

## Quick Start

### Environment Variables

```bash
# Required: Aragora server configuration
export ARAGORA_API_TOKEN="your-api-token"

# Optional: OpenClaw backend URL (if connecting to a live instance)
export OPENCLAW_URL="http://localhost:9222"

# Optional: Policy file path (defaults to built-in enterprise policy)
export OPENCLAW_POLICY_FILE="/path/to/policy.yaml"
```

### Start the Server

The OpenClaw gateway handler is registered automatically when you start the Aragora server:

```bash
python -m aragora.server.unified_server --port 8080
```

The gateway endpoints will be available at `/api/v1/openclaw/*` and `/api/gateway/openclaw/*`.

### Verify the Gateway

```bash
curl http://localhost:8080/api/gateway/openclaw/health
```

Expected response:
```json
{
  "status": "healthy",
  "healthy": true,
  "active_sessions": 0,
  "pending_actions": 0,
  "running_actions": 0
}
```

## SDK Usage

The Aragora Python SDK exposes OpenClaw operations through `client.openclaw`.

### Session Lifecycle

```python
from aragora.client import AragoraClient

client = AragoraClient(base_url="http://localhost:8080", token="your-token")

# Create a session
session = client.openclaw.create_session(
    user_id="user-123",
    tenant_id="acme-corp",
    workspace_id="/workspace/project",
    roles=["developer"],
)
print(f"Session: {session.session_id}")

# Execute actions within the session
result = client.openclaw.execute_shell(session.session_id, "ls -la /workspace")
print(f"Decision: {result.decision}, Output: {result.result}")

# Read a file
result = client.openclaw.execute_file_read(session.session_id, "/workspace/README.md")

# Write a file
result = client.openclaw.execute_file_write(
    session.session_id, "/workspace/output.txt", "Hello from OpenClaw"
)

# Browser action
result = client.openclaw.execute_browser(session.session_id, "https://example.com")

# End the session when done
client.openclaw.end_session(session.session_id)
```

### Async Usage

Every SDK method has an `_async` variant:

```python
session = await client.openclaw.create_session_async(
    user_id="user-123", tenant_id="acme-corp"
)
result = await client.openclaw.execute_shell_async(session.session_id, "whoami")
await client.openclaw.end_session_async(session.session_id)
```

### Handling Approval-Required Actions

Some actions may require human approval based on policy rules:

```python
result = client.openclaw.execute_action(session.session_id, "shell", command="sudo apt install curl")

if result.requires_approval:
    print(f"Approval needed: {result.approval_id}")
    # An admin can approve via:
    client.openclaw.approve_action(result.approval_id, approver_id="admin-1", reason="Approved")
```

### Audit Trail

```python
records = client.openclaw.query_audit(user_id="user-123", limit=20)
for record in records:
    print(f"{record.event_type}: {record.action_type} (success={record.success})")
```

### Statistics

```python
stats = client.openclaw.get_stats()
print(f"Active sessions: {stats.active_sessions}")
print(f"Actions allowed: {stats.actions_allowed}")
print(f"Actions denied:  {stats.actions_denied}")
```

## API Reference

All endpoints are under `/api/v1/openclaw/` (or `/api/gateway/openclaw/`).

| Method | Endpoint | Permission | Description |
|--------|----------|------------|-------------|
| POST | `/sessions` | `gateway:sessions.create` | Create a new session |
| GET | `/sessions` | `gateway:sessions.read` | List sessions (filtered by user/tenant) |
| GET | `/sessions/:id` | `gateway:sessions.read` | Get session details |
| POST | `/sessions/:id/end` | `gateway:sessions.delete` | End a session |
| DELETE | `/sessions/:id` | `gateway:sessions.delete` | Close a session |
| POST | `/actions` | `gateway:actions.execute` | Execute an action |
| GET | `/actions/:id` | `gateway:actions.read` | Get action status |
| POST | `/actions/:id/cancel` | `gateway:actions.cancel` | Cancel a running action |
| POST | `/credentials` | `gateway:credentials.create` | Store a credential |
| GET | `/credentials` | `gateway:credentials.read` | List credentials (no secrets) |
| DELETE | `/credentials/:id` | `gateway:credentials.delete` | Delete a credential |
| POST | `/credentials/:id/rotate` | `gateway:credentials.rotate` | Rotate a credential secret |
| GET | `/policy/rules` | `gateway:policy.read` | List policy rules |
| POST | `/policy/rules` | `gateway:policy.write` | Add a policy rule |
| DELETE | `/policy/rules/:name` | `gateway:policy.write` | Remove a policy rule |
| GET | `/approvals` | `gateway:approvals.read` | List pending approvals |
| POST | `/approvals/:id/approve` | `gateway:approvals.write` | Approve a pending action |
| POST | `/approvals/:id/deny` | `gateway:approvals.write` | Deny a pending action |
| GET | `/health` | (public) | Gateway health check |
| GET | `/metrics` | `gateway:metrics.read` | Gateway metrics |
| GET | `/stats` | `gateway:metrics.read` | Proxy statistics |
| GET | `/audit` | `gateway:audit.read` | Query audit log |

## Workflow Nodes

The integration provides two workflow step types for use in Aragora DAG workflows.

### OpenClawSessionStep

Manages session lifecycle within a workflow:

```python
from aragora.workflow.nodes.openclaw import OpenClawSessionStep, OpenClawActionStep

# Create a session
create_session = OpenClawSessionStep(
    name="create_session",
    config={
        "operation": "create",
        "workspace_id": "/workspace/project",
        "roles": ["developer"],
    },
)

# End a session (referencing the output of the create step)
end_session = OpenClawSessionStep(
    name="end_session",
    config={
        "operation": "end",
        "session_id": "{step.create_session.session_id}",
    },
)
```

### OpenClawActionStep

Executes actions within a session. Supported `action_type` values: `shell`, `file_read`, `file_write`, `file_delete`, `browser`, `screenshot`, `api`.

```python
# Shell command
list_files = OpenClawActionStep(
    name="list_files",
    config={
        "action_type": "shell",
        "session_id": "{step.create_session.session_id}",
        "command": "ls -la /workspace",
    },
)

# File write
write_config = OpenClawActionStep(
    name="write_config",
    config={
        "action_type": "file_write",
        "session_id": "{step.create_session.session_id}",
        "path": "/workspace/config.json",
        "content": '{"debug": true}',
        "on_failure": "skip",  # Options: error, skip, retry
    },
)

# Browser navigation
open_docs = OpenClawActionStep(
    name="open_docs",
    config={
        "action_type": "browser",
        "session_id": "{step.create_session.session_id}",
        "url": "https://docs.example.com",
        "timeout_seconds": 30.0,
    },
)
```

Template variables like `{step.<step_name>.<field>}` are resolved at runtime from prior step outputs.

To register the step types with the workflow engine:

```python
from aragora.workflow.nodes.openclaw import register_openclaw_steps
register_openclaw_steps()
```

## Policy Configuration

The proxy uses a YAML-based policy engine. The default enterprise policy (`create_enterprise_policy()`) ships with sensible defaults.

### Default Enterprise Policy Behavior

- **Deny** access to system directories (`/etc`, `/sys`, `/proc`, `/root`, `/boot`, `/dev`)
- **Deny** dangerous shell commands (`rm -rf /`, `mkfs`, `dd if=...of=/dev/...`)
- **Require approval** for `sudo` and destructive commands
- **Allow** file operations within the workspace directory
- **Rate limit** API calls

### Custom Policy via YAML

```yaml
version: 1
default_decision: deny

rules:
  - name: allow_workspace_reads
    action_types: [file_read]
    decision: allow
    priority: 50
    path_patterns:
      - "/workspace/**"
    description: "Allow reading files in workspace"

  - name: block_secrets
    action_types: [file_read, file_write]
    decision: deny
    priority: 100
    path_patterns:
      - "**/.env"
      - "**/credentials*"
      - "**/*.pem"
    description: "Block access to secret files"

  - name: approve_installs
    action_types: [shell]
    decision: require_approval
    priority: 80
    command_patterns:
      - "apt install.*"
      - "pip install.*"
    description: "Require approval for package installs"

  - name: rate_limit_api
    action_types: [api]
    decision: allow
    priority: 10
    rate_limit: 30
    rate_limit_window: 60
    description: "Allow API calls with rate limiting"
```

### Adding Rules at Runtime

```python
from aragora.client.resources.openclaw import PolicyRule

rule = PolicyRule(
    name="block_production_db",
    action_types=["shell"],
    decision="deny",
    priority=100,
    description="Block direct access to production database",
    config={"command_deny_patterns": [r"psql.*production", r"mysql.*prod"]},
)
client.openclaw.add_rule(rule)

# List current rules
rules = client.openclaw.get_policy_rules()

# Remove a rule
client.openclaw.remove_rule("block_production_db")
```

### Using the Proxy Directly

```python
from aragora.gateway.openclaw_proxy import OpenClawSecureProxy
from aragora.gateway.openclaw_policy import OpenClawPolicy

# Load from YAML file
proxy = OpenClawSecureProxy(policy_file="/path/to/policy.yaml")

# Or use the default enterprise policy
proxy = OpenClawSecureProxy()

# Or construct programmatically
policy = OpenClawPolicy()
policy.add_rule(...)
proxy = OpenClawSecureProxy(policy=policy, audit_callback=my_audit_handler)
```

## SKILL.md Migration

If you previously defined OpenClaw agent capabilities in a `SKILL.md` file (a common pattern for configuring computer-use agents), you can migrate to Aragora's native integration:

| SKILL.md Pattern | Aragora Equivalent |
|------------------|--------------------|
| Allowed commands list | Policy rule with `command_patterns` (decision: allow) |
| Blocked commands list | Policy rule with `command_deny_patterns` (decision: deny) |
| File access paths | Policy rule with `path_patterns` / `path_deny_patterns` |
| Approval-required actions | Policy rule with `decision: require_approval` |
| Role restrictions | Policy rule with `allowed_roles` / `denied_roles` |
| Rate limits | Policy rule with `rate_limit` and `rate_limit_window` |

**Migration steps:**

1. Identify the action restrictions in your `SKILL.md`
2. Create a YAML policy file mapping each restriction to a `PolicyRule`
3. Set `OPENCLAW_POLICY_FILE` to your new policy file, or add rules via the API
4. Test with `client.openclaw.execute_action()` to verify policy decisions match expectations
5. Remove the `SKILL.md` once all rules are ported

**Source files:**

- Handler: `aragora/server/handlers/openclaw_gateway.py`
- SDK client: `aragora/client/resources/openclaw.py`
- Workflow nodes: `aragora/workflow/nodes/openclaw.py`
- Proxy: `aragora/gateway/openclaw_proxy.py`
- Policy engine: `aragora/gateway/openclaw_policy.py`
