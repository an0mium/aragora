# Workspace Handlers

Enterprise workspace management APIs providing data isolation, access control, retention policies, and privacy compliance.

## Modules

| Module | Purpose |
|--------|---------|
| `crud.py` | Workspace CRUD operations (create, read, update, delete) |
| `members.py` | Member management, roles, and RBAC profile assignment |
| `invites.py` | Workspace invitation workflows with expiration |
| `policies.py` | Retention policy creation, execution, and scheduling |
| `settings.py` | Sensitivity classification and audit endpoints |
| `workspace_utils.py` | Circuit breaker and validation utilities |
| `sensitivity.py` | Sensitivity level definitions |

## Endpoints

### Workspace CRUD
- `GET /api/v1/workspaces` - List workspaces for user
- `POST /api/v1/workspaces` - Create workspace
- `GET /api/v1/workspaces/{id}` - Get workspace details
- `PUT /api/v1/workspaces/{id}` - Update workspace
- `DELETE /api/v1/workspaces/{id}` - Delete workspace

### Members
- `POST /api/v1/workspaces/{id}/members` - Add member
- `DELETE /api/v1/workspaces/{id}/members/{user_id}` - Remove member
- `GET /api/v1/workspaces/{id}/roles` - List available roles
- `PUT /api/v1/workspaces/{id}/members/{user_id}/role` - Update member role
- `GET /api/v1/workspaces/profiles` - List RBAC profiles

### Invites
- `POST /api/v1/workspaces/{id}/invites` - Create invite
- `GET /api/v1/workspaces/{id}/invites` - List pending invites
- `POST /api/v1/workspaces/invites/{token}/accept` - Accept invite
- `DELETE /api/v1/workspaces/{id}/invites/{invite_id}` - Revoke invite

### Retention Policies
- `GET /api/v1/workspaces/{id}/policies` - List policies
- `POST /api/v1/workspaces/{id}/policies` - Create policy
- `GET /api/v1/workspaces/{id}/policies/{policy_id}` - Get policy
- `PUT /api/v1/workspaces/{id}/policies/{policy_id}` - Update policy
- `DELETE /api/v1/workspaces/{id}/policies/{policy_id}` - Delete policy
- `POST /api/v1/workspaces/{id}/policies/{policy_id}/execute` - Execute policy
- `GET /api/v1/workspaces/{id}/policies/expiring` - Get expiring items

### Settings & Audit
- `GET /api/v1/workspaces/{id}/sensitivity` - Get sensitivity settings
- `PUT /api/v1/workspaces/{id}/sensitivity` - Update sensitivity
- `GET /api/v1/workspaces/{id}/audit` - Get audit log

## RBAC Permissions

| Permission | Description |
|------------|-------------|
| `workspace:read` | View workspace and members |
| `workspace:write` | Create/update workspaces |
| `workspace:delete` | Delete workspaces |
| `workspace:share` | Add/remove members |
| `workspace:admin` | Manage policies and settings |

## RBAC Profiles

Three pre-configured profiles for different team sizes:

| Profile | Roles | Features |
|---------|-------|----------|
| `lite` | owner, admin, member | Basic sharing, audit log |
| `standard` | owner, admin, editor, viewer, member | Custom permissions, advanced audit |
| `enterprise` | Full RBAC hierarchy | SSO, compliance controls |

## Usage

```python
from aragora.server.handlers.workspace import (
    WorkspaceHandler,
    get_workspace_circuit_breaker_status,
)

# Check circuit breaker status
status = get_workspace_circuit_breaker_status()
```

## Features

- **Data Isolation**: Tenant-level workspace separation
- **RBAC Profiles**: Pre-configured role hierarchies (lite/standard/enterprise)
- **Retention Policies**: Automated data lifecycle management
- **Sensitivity Classification**: Data classification enforcement
- **Audit Logging**: Complete compliance audit trail
- **Circuit Breaker**: Fault tolerance for storage failures
