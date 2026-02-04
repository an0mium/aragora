# OpenClaw Gateway Handlers

REST API handlers for the OpenClaw gateway integration, providing session management, action execution, credential storage, and policy enforcement.

## Modules

| Module | Purpose |
|--------|---------|
| `gateway.py` | Main handler class routing requests to sub-handlers |
| `orchestrator.py` | Session lifecycle and action execution orchestration |
| `credentials.py` | Secure credential storage with rotation and rate limiting |
| `policies.py` | Policy rules, enforcement, and approval workflows |
| `models.py` | Data models (Session, Action, Credential, AuditEntry) |
| `validation.py` | Input validation and sanitization functions |
| `store.py` | SQLite-backed persistent storage with async operations |

## Endpoints

### Sessions
- `GET /api/gateway/openclaw/sessions` - List active sessions
- `GET /api/gateway/openclaw/sessions/{id}` - Get session details
- `POST /api/gateway/openclaw/sessions` - Create new session
- `DELETE /api/gateway/openclaw/sessions/{id}` - Close session

### Actions
- `GET /api/gateway/openclaw/actions/{id}` - Get action status
- `POST /api/gateway/openclaw/actions` - Execute action within session

### Credentials
- `GET /api/gateway/openclaw/credentials` - List stored credentials
- `POST /api/gateway/openclaw/credentials` - Store new credential
- `DELETE /api/gateway/openclaw/credentials/{name}` - Delete credential

### Policy
- `GET /api/gateway/openclaw/policy/rules` - List policy rules
- `POST /api/gateway/openclaw/policy/rules` - Create policy rule
- `DELETE /api/gateway/openclaw/policy/rules/{id}` - Delete policy rule
- `GET /api/gateway/openclaw/approvals` - List pending approvals

### Admin
- `GET /api/gateway/openclaw/health` - Health check
- `GET /api/gateway/openclaw/metrics` - Gateway metrics
- `GET /api/gateway/openclaw/audit` - Audit log entries
- `GET /api/gateway/openclaw/stats` - Usage statistics

## RBAC Permissions

| Permission | Description |
|------------|-------------|
| `openclaw:read` | View sessions, actions, credentials |
| `openclaw:write` | Create sessions, execute actions |
| `openclaw:admin` | Manage policies, view audit logs |
| `openclaw:credentials` | Manage stored credentials |

## Usage

```python
from aragora.server.handlers.openclaw import (
    OpenClawGatewayHandler,
    get_openclaw_gateway_handler,
)

# Get singleton handler instance
handler = get_openclaw_gateway_handler()

# Check circuit breaker status
from aragora.server.handlers.openclaw import get_openclaw_circuit_breaker_status
status = get_openclaw_circuit_breaker_status()
```

## Features

- **Session Management**: Create, track, and close gateway sessions
- **Action Execution**: Execute actions with input validation and timeout
- **Credential Security**: Encrypted storage with rotation and rate limiting
- **Policy Enforcement**: Rule-based approval workflows
- **Circuit Breaker**: Fault tolerance for upstream gateway failures
- **Audit Logging**: Complete audit trail for compliance

## Tests

166 tests covering validation, credentials, orchestration, and integration.
