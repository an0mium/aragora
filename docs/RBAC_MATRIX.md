# RBAC Permission Matrix

This document defines the Role-Based Access Control (RBAC) system used in Aragora for fine-grained authorization.

## Overview

Aragora uses a hierarchical RBAC system with:
- **8 System Roles** with predefined permissions
- **100+ Permissions** covering all resource types
- **Role Hierarchy** for permission inheritance
- **Custom Roles** for organization-specific needs

## System Roles

| Role | Priority | Description | Use Case |
|------|----------|-------------|----------|
| **Owner** | 100 | Full control over organization | Organization founders |
| **Admin** | 80 | Manage users and resources (no billing) | IT administrators |
| **Compliance Officer** | 75 | Data governance and audit | Security/compliance teams |
| **Team Lead** | 55 | Manage team membership | Engineering managers |
| **Debate Creator** | 50 | Create and run debates | Power users |
| **Member** | 40 | Standard organization access | Regular employees |
| **Analyst** | 30 | Read-only analytics access | Data analysts |
| **Viewer** | 10 | Minimal read access | External stakeholders |

## Role Hierarchy

```
Owner
  └── Admin
        ├── Compliance Officer
        │     └── Analyst
        │           └── Viewer
        └── Debate Creator
              └── Team Lead
                    └── Member
                          └── Viewer
```

Roles inherit all permissions from their descendants in the hierarchy.

## Permission Categories

### Core Permissions

#### Debates (`debate.*`)
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `debate.create` | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | - |
| `debate.read` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `debate.update` | ✓ | ✓ | - | ✓ | ✓ | - | - | - |
| `debate.delete` | ✓ | ✓ | - | - | - | - | - | - |
| `debate.run` | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | - |
| `debate.stop` | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | - |
| `debate.fork` | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | - |

#### Agents (`agent.*`)
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `agent.create` | ✓ | ✓ | - | - | - | - | - | - |
| `agent.read` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `agent.update` | ✓ | ✓ | - | - | - | - | - | - |
| `agent.delete` | ✓ | ✓ | - | - | - | - | - | - |
| `agent.deploy` | ✓ | ✓ | - | - | - | - | - | - |

#### Users (`user.*`)
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `user.read` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `user.invite` | ✓ | ✓ | - | - | - | - | - | - |
| `user.remove` | ✓ | ✓ | - | - | - | - | - | - |
| `user.change_role` | ✓ | ✓ | - | - | - | - | - | - |
| `user.impersonate` | ✓ | - | - | - | - | - | - | - |

#### Organization (`organization.*`)
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `organization.read` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `organization.update` | ✓ | ✓ | - | - | - | - | - | - |
| `organization.manage_billing` | ✓ | - | - | - | - | - | - | - |
| `organization.view_audit` | ✓ | ✓ | ✓ | - | - | - | - | - |
| `organization.export_data` | ✓ | ✓ | - | - | - | - | - | - |

### Enterprise Permissions

#### Gauntlet (Adversarial Testing)
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `gauntlet.run` | ✓ | ✓ | - | ✓ | ✓ | ✓ | - | - |
| `gauntlet.read` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `gauntlet.delete` | ✓ | ✓ | - | - | - | - | - | - |
| `gauntlet.sign` | ✓ | ✓ | - | - | - | - | - | - |
| `gauntlet.compare` | ✓ | ✓ | - | ✓ | ✓ | - | - | - |
| `gauntlet.export_data` | ✓ | ✓ | - | ✓ | ✓ | - | - | - |

#### Compliance & Data Governance
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `data_classification.read` | ✓ | - | ✓ | - | - | - | - | - |
| `data_classification.classify` | ✓ | - | ✓ | - | - | - | - | - |
| `data_retention.read` | ✓ | - | ✓ | - | - | - | - | - |
| `data_retention.update` | ✓ | - | ✓ | - | - | - | - | - |
| `pii.read` | ✓ | - | ✓ | - | - | - | - | - |
| `pii.redact` | ✓ | - | ✓ | - | - | - | - | - |
| `audit_log.read` | ✓ | - | ✓ | - | - | - | - | - |
| `audit_log.export` | ✓ | - | ✓ | - | - | - | - | - |

#### Control Plane
| Permission | Owner | Admin | Compliance | Team Lead | Creator | Member | Analyst | Viewer |
|------------|:-----:|:-----:|:----------:|:---------:|:-------:|:------:|:-------:|:------:|
| `control_plane.read` | ✓ | ✓ | ✓ | - | - | - | - | - |
| `control_plane.submit` | ✓ | ✓ | - | - | - | - | - | - |
| `control_plane.cancel` | ✓ | ✓ | - | - | - | - | - | - |
| `control_plane.deliberate` | ✓ | ✓ | - | - | - | - | - | - |

## Handler Permission Mapping

### Admin Handlers (`/api/v1/admin/*`)

| Endpoint | Method | Permission | Notes |
|----------|--------|------------|-------|
| `/admin/organizations` | GET | `admin.organizations.list` | List all orgs |
| `/admin/users` | GET | `admin.users.list` | List all users |
| `/admin/stats` | GET | `admin.stats.read` | System statistics |
| `/admin/metrics` | GET | `admin.metrics.read` | System metrics |
| `/admin/revenue` | GET | `admin.revenue.read` | Revenue stats |
| `/admin/users/{id}/impersonate` | POST | `admin.users.impersonate` | User impersonation |
| `/admin/users/{id}/deactivate` | POST | `admin.users.deactivate` | Deactivate user |
| `/admin/users/{id}/activate` | POST | `admin.users.activate` | Activate user |
| `/admin/users/{id}/unlock` | POST | `admin.users.unlock` | Unlock account |
| `/admin/nomic/status` | GET | `admin.nomic.read` | Nomic loop status |
| `/admin/nomic/circuit-breakers` | GET | `admin.nomic.read` | Circuit breakers |
| `/admin/nomic/reset` | POST | `admin.nomic.write` | Reset nomic phase |
| `/admin/nomic/pause` | POST | `admin.nomic.write` | Pause nomic |
| `/admin/nomic/resume` | POST | `admin.nomic.write` | Resume nomic |

### Security Handlers (`/api/v1/admin/security/*`)

| Endpoint | Method | Permission | Notes |
|----------|--------|------------|-------|
| `/admin/security/status` | GET | `admin.security.status` | Encryption status |
| `/admin/security/health` | GET | `admin.security.health` | Security health |
| `/admin/security/keys` | GET | `admin.security.keys` | List keys |
| `/admin/security/rotate-key` | POST | `admin.security.rotate` | Rotate key |

### Billing Handlers (`/api/v1/billing/*`)

| Endpoint | Method | Permission | Notes |
|----------|--------|------------|-------|
| `/billing/usage` | GET | `org:billing` | Usage stats |
| `/billing/subscription` | GET | `org:billing` | Subscription details |
| `/billing/checkout` | POST | `org:billing` | Create checkout |
| `/billing/portal` | POST | `org:billing` | Billing portal |
| `/billing/cancel` | POST | `org:billing` | Cancel subscription |
| `/billing/resume` | POST | `org:billing` | Resume subscription |
| `/billing/audit-log` | GET | `admin:audit` | Billing audit log |
| `/billing/usage/export` | GET | `org:billing` | Export usage |
| `/billing/usage/forecast` | GET | `org:billing` | Usage forecast |
| `/billing/invoices` | GET | `org:billing` | Invoice history |

## Custom Role Creation

Organizations can create custom roles based on system roles:

```python
from aragora.rbac.defaults import create_custom_role

engineering_role = create_custom_role(
    name="engineering",
    display_name="Engineering Team",
    description="Engineering with agent management",
    permission_keys={
        "agent.create",
        "agent.update",
        "connector.create",
    },
    org_id="org-123",
    base_role="debate_creator",  # Inherit permissions
)
```

### Role Templates

| Template | Base Role | Additional Permissions |
|----------|-----------|----------------------|
| `engineering` | debate_creator | agent.create, agent.update, connector.create |
| `research` | analyst | training.create, debate.create, debate.run |
| `support` | viewer | user.read, organization.view_audit |
| `external` | viewer | (none) |

## Permission Enforcement

### Decorator Usage

```python
from aragora.rbac.decorators import require_permission

@require_permission("debate.create")
async def create_debate(context: AuthorizationContext, ...):
    ...

@require_permission("user.impersonate", resource_id_param="user_id")
async def impersonate_user(context: AuthorizationContext, user_id: str, ...):
    ...
```

### Manual Check

```python
from aragora.rbac import check_permission, AuthorizationContext

context = AuthorizationContext(
    user_id="user-123",
    roles={"admin"},
    org_id="org-456",
)

decision = check_permission(context, "debate.delete", resource_id="debate-789")
if decision.allowed:
    # Proceed with deletion
    ...
```

## Audit Logging

All RBAC decisions are logged for compliance:

```json
{
  "timestamp": "2026-01-25T12:00:00Z",
  "user_id": "user-123",
  "permission": "debate.delete",
  "resource_id": "debate-789",
  "allowed": true,
  "reason": "Permission granted via admin role",
  "ip_address": "192.168.1.1"
}
```

## Best Practices

1. **Principle of Least Privilege**: Assign the minimum role needed
2. **Role Inheritance**: Use hierarchy instead of duplicating permissions
3. **Custom Roles**: Create org-specific roles for unique needs
4. **Audit Regularly**: Review role assignments quarterly
5. **MFA Enforcement**: Require MFA for admin/owner roles

## Related Documentation

- [Enterprise Features](ENTERPRISE_FEATURES.md)
- [Security Configuration](SECURITY.md)
- [API Reference](API_REFERENCE.md)
