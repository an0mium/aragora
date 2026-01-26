# RBAC and ABAC Authorization Strategy

This document defines the authorization strategy for Aragora, providing practical guidance for developers implementing access control in handlers.

## Overview

Aragora uses a **layered authorization model**:

| Layer | System | Purpose | Performance |
|-------|--------|---------|-------------|
| 1 | **RBAC** | Coarse-grained, organization-wide permissions | Fast (cached) |
| 2 | **ABAC** | Fine-grained, resource-level access | Per-request |

```
Request → [RBAC Check] → [ABAC Check] → Handler Logic
              ↓               ↓
           DENY            DENY
```

## When to Use Each System

### Use RBAC Only

For **capability-based** operations that don't depend on specific resource ownership:

- Creating new resources (e.g., `debates:create`)
- Listing resources the user has access to (e.g., `workflows:list`)
- Administrative actions (e.g., `settings:update`)
- Billing and subscription management

```python
from aragora.rbac.decorators import require_permission

@require_permission("debates:create")
async def create_debate(context: AuthorizationContext, request: DebateCreateRequest):
    # User has organization-wide permission to create debates
    ...
```

### Use RBAC + ABAC

For **resource-specific** operations where you need both:

1. Permission to perform the action type (RBAC)
2. Access to the specific resource (ABAC)

- Updating a specific debate
- Deleting a specific workflow
- Sharing a document with others
- Exporting sensitive data

```python
from aragora.rbac.decorators import require_permission
from aragora.server.middleware.abac import check_resource_access, ResourceType, Action

@require_permission("debates:update", resource_id_param="debate_id")
async def update_debate(context: AuthorizationContext, debate_id: str, updates: dict):
    # Step 1: RBAC passed - user can update debates in general

    # Step 2: ABAC - check access to this specific debate
    debate = await get_debate(debate_id)
    decision = check_resource_access(
        user_id=context.user_id,
        user_role=context.system_role,
        user_plan=context.plan,
        resource_type=ResourceType.DEBATE,
        resource_id=debate_id,
        action=Action.WRITE,
        resource_owner_id=debate.owner_id,
        resource_workspace_id=debate.workspace_id,
        user_workspace_id=context.workspace_id,
        shared_with=debate.shared_with,
    )

    if not decision.allowed:
        raise ForbiddenError(decision.reason)

    # Proceed with update
    ...
```

### Use ABAC Only

**Not recommended.** Always pair with RBAC for:

- Consistent audit trails
- Cache efficiency
- Permission visibility in admin UI

## Authorization Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Handler Entry Point                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  @require_permission   │
              │  "resource:action"     │
              │  ─────────────────     │
              │  • Cached (Redis)      │
              │  • O(1) lookup         │
              │  • Audited             │
              └────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │ ALLOW                   │ DENY
              ▼                         ▼
   ┌────────────────────┐    ┌────────────────────┐
   │  Fetch Resource    │    │  Return 403        │
   │  from Database     │    │  PermissionDenied  │
   └────────────────────┘    └────────────────────┘
              │
              ▼
   ┌────────────────────────┐
   │  check_resource_access │
   │  ────────────────────  │
   │  • Owner check         │
   │  • Workspace check     │
   │  • Shared access       │
   │  • Sensitivity level   │
   └────────────────────────┘
              │
   ┌──────────┴──────────┐
   │ ALLOW               │ DENY
   ▼                     ▼
┌────────────────┐  ┌────────────────────┐
│ Execute Logic  │  │ Return 403         │
│                │  │ Resource access    │
└────────────────┘  │ denied             │
                    └────────────────────┘
```

## Permission Key Format

Use colon-separated format: `resource:action`

```python
# Correct
@require_permission("debates:create")
@require_permission("workflows:execute")
@require_permission("knowledge:export")

# Incorrect (legacy dot format)
@require_permission("debates.create")  # Don't use
```

### Standard Actions

| Action | Description | Typical Use |
|--------|-------------|-------------|
| `create` | Create new resource | POST endpoints |
| `read` | View resource | GET endpoints |
| `update` | Modify resource | PUT/PATCH endpoints |
| `delete` | Remove resource | DELETE endpoints |
| `list` | List resources | GET collection endpoints |
| `execute` | Run/trigger | Workflow execution |
| `export` | Export data | Data export endpoints |
| `admin` | Administrative ops | Settings, configuration |

## ABAC Evaluation Order

The ABAC evaluator checks access in this order (first match wins):

1. **System Admin** → Allow (if action in admin_actions)
2. **Resource Owner** → Allow (if action in owner_actions)
3. **Workspace Admin** → Allow (if action in workspace_admin_actions)
4. **Workspace Member** → Allow (if action in workspace_member_actions)
5. **Shared Access** → Allow (if action in shared_user_actions)
6. **Public Access** → Allow (if allow_public_read and action is READ)
7. **Sensitivity Check** → Deny if plan insufficient
8. **Default** → Deny

## Code Examples

### Example 1: Simple RBAC Protection

```python
from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext

@require_permission("agents:list")
async def list_agents(context: AuthorizationContext) -> list[Agent]:
    """List all agents. Requires agents:list permission."""
    return await agent_repository.list_all(org_id=context.org_id)
```

### Example 2: RBAC with Resource ID

```python
@require_permission("debates:update", resource_id_param="debate_id")
async def update_debate(
    context: AuthorizationContext,
    debate_id: str,
    updates: DebateUpdate
) -> Debate:
    """Update a debate. Permission check includes resource_id for audit."""
    # RBAC passed, now check ABAC
    debate = await debate_repository.get(debate_id)

    decision = check_resource_access(
        user_id=context.user_id,
        user_role="user",
        user_plan=context.metadata.get("plan", "free"),
        resource_type=ResourceType.DEBATE,
        resource_id=debate_id,
        action=Action.WRITE,
        resource_owner_id=debate.owner_id,
        resource_workspace_id=debate.workspace_id,
        user_workspace_id=context.metadata.get("workspace_id"),
    )

    if not decision.allowed:
        raise ForbiddenError(decision.reason)

    return await debate_repository.update(debate_id, updates)
```

### Example 3: Role-Based Access

```python
from aragora.rbac.decorators import require_role, require_admin

@require_role("admin", "owner")
async def manage_organization(context: AuthorizationContext, org_id: str):
    """Requires admin or owner role."""
    ...

@require_admin()  # Shorthand for @require_role("admin", "owner")
async def delete_organization(context: AuthorizationContext, org_id: str):
    """Requires admin privileges."""
    ...
```

### Example 4: Self-or-Admin Pattern

```python
from aragora.rbac.decorators import require_self_or_admin

@require_self_or_admin(user_id_param="target_user_id")
async def update_user_settings(
    context: AuthorizationContext,
    target_user_id: str,
    settings: UserSettings
):
    """User can update own settings, admin can update any user."""
    ...
```

### Example 5: Organization Scoping

```python
from aragora.rbac.decorators import require_permission, require_org_access

@require_permission("reports:read")
@require_org_access(org_id_param="org_id")
async def get_org_report(
    context: AuthorizationContext,
    org_id: str,
    report_type: str
):
    """View organization reports. Must belong to the org."""
    ...
```

## Adding RBAC to New Handlers

### Step 1: Identify the Permission

Determine the resource and action:

```python
# Resource: what entity is being accessed
# Action: what operation is performed

# Examples:
# - Creating a workflow → "workflows:create"
# - Viewing analytics → "analytics:read"
# - Deleting a template → "templates:delete"
```

### Step 2: Check if Permission Exists

```bash
# Search existing permissions
grep -r "require_permission" aragora/server/handlers/ | grep "your_resource"
```

If not, add to `aragora/rbac/defaults.py`:

```python
# In DEFAULT_PERMISSIONS
Permission(
    key="workflows:execute",
    name="Execute Workflows",
    description="Run workflow automations",
    resource="workflows",
    action="execute",
    category=PermissionCategory.WORKFLOWS,
)
```

### Step 3: Add Decorator

```python
from aragora.rbac.decorators import require_permission

@require_permission("workflows:execute", resource_id_param="workflow_id")
async def execute_workflow(
    context: AuthorizationContext,
    workflow_id: str,
    inputs: dict
):
    ...
```

### Step 4: Add ABAC if Needed

For resource-specific operations, add ABAC after RBAC:

```python
# After RBAC decorator passes
workflow = await workflow_repository.get(workflow_id)

decision = check_resource_access(
    user_id=context.user_id,
    user_role=context.system_role,
    user_plan=context.plan,
    resource_type=ResourceType.WORKFLOW,
    resource_id=workflow_id,
    action=Action.EXECUTE,
    resource_owner_id=workflow.owner_id,
    resource_workspace_id=workflow.workspace_id,
    user_workspace_id=context.workspace_id,
)

if not decision.allowed:
    raise ForbiddenError(decision.reason)
```

## Testing Authorization

### Unit Tests

```python
import pytest
from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.models import AuthorizationContext

@pytest.fixture
def admin_context():
    return AuthorizationContext(
        user_id="admin-1",
        org_id="org-1",
        roles={"admin"},
    )

@pytest.fixture
def user_context():
    return AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        roles={"member"},
    )

async def test_admin_can_delete(admin_context):
    result = await delete_resource(admin_context, resource_id="res-1")
    assert result.success

async def test_user_cannot_delete(user_context):
    with pytest.raises(PermissionDeniedError):
        await delete_resource(user_context, resource_id="res-1")
```

### Integration Tests

```python
async def test_owner_can_update_own_debate(client, auth_headers):
    # Create debate as user
    debate = await client.post("/api/v1/debates", headers=auth_headers, json={...})
    debate_id = debate.json()["id"]

    # Update should succeed (owner)
    response = await client.patch(
        f"/api/v1/debates/{debate_id}",
        headers=auth_headers,
        json={"title": "Updated"}
    )
    assert response.status_code == 200

async def test_other_user_cannot_update_debate(client, other_auth_headers, debate_id):
    # Other user trying to update should fail
    response = await client.patch(
        f"/api/v1/debates/{debate_id}",
        headers=other_auth_headers,
        json={"title": "Hacked"}
    )
    assert response.status_code == 403
```

## Migration Guide

### Converting from Dot to Colon Format

```python
# Before (deprecated)
@require_permission("debates.create")

# After (current)
@require_permission("debates:create")
```

### Converting ABAC-only to RBAC+ABAC

```python
# Before (ABAC only)
async def update_debate(user: User, debate_id: str):
    debate = await get_debate(debate_id)
    decision = check_resource_access(user_id=user.id, ...)
    if not decision.allowed:
        raise ForbiddenError(...)
    ...

# After (RBAC + ABAC)
@require_permission("debates:update", resource_id_param="debate_id")
async def update_debate(context: AuthorizationContext, debate_id: str):
    # RBAC already passed via decorator
    debate = await get_debate(debate_id)
    decision = check_resource_access(user_id=context.user_id, ...)
    if not decision.allowed:
        raise ForbiddenError(...)
    ...
```

## Related Documentation

- [ADR-017: RBAC and ABAC Unification](../analysis/adr-017-rbac-abac-unification.md) - Architecture decision record
- [RBAC Module](../../aragora/rbac/) - Implementation code
- [ABAC Middleware](../../aragora/server/middleware/abac.py) - ABAC evaluator
- [Authentication](./authentication.md) - Authentication setup

## Summary

| Scenario | Use |
|----------|-----|
| Create/list operations | RBAC only |
| Read/update/delete specific resource | RBAC + ABAC |
| Admin-only features | `@require_admin()` |
| User modifying own data | `@require_self_or_admin()` |
| Organization-scoped data | `@require_org_access()` |
| Sensitive data export | RBAC + ABAC + sensitivity check |
