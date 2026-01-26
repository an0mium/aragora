# RBAC Role Hierarchy

Complete reference for system roles and their inheritance in Aragora.

## Summary

| Role | Priority | Permissions | Description |
|------|----------|-------------|-------------|
| Owner | 100 | 143 | Full control over organization |
| Admin | 80 | 95 | Administrative access (no billing) |
| Compliance Officer | 75 | 42 | Data governance and audit |
| Team Lead | 55 | 45 | Team management + member permissions |
| Debate Creator | 50 | 43 | Create and manage debates |
| Member | 40 | 31 | Standard organization member |
| Analyst | 30 | 15 | Read-only analytics access |
| Viewer | 10 | 4 | Minimal read-only access |

---

## Role Hierarchy Diagram

```
                    ┌─────────┐
                    │  Owner  │  (All 143 permissions)
                    │  P:100  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  Admin  │  (95 permissions, no billing)
                    │  P:80   │
                    └────┬────┘
              ┌──────────┼──────────┐
              │          │          │
      ┌───────▼──┐  ┌────▼────┐  ┌──▼──────┐
      │Compliance│  │ Debate  │  │ Analyst │
      │ Officer  │  │ Creator │  │  P:30   │
      │  P:75    │  │  P:50   │  └────┬────┘
      └────┬─────┘  └────┬────┘       │
           │             │            │
           │       ┌─────▼────┐       │
           │       │Team Lead │       │
           │       │  P:55    │       │
           │       └────┬─────┘       │
           │            │             │
           │       ┌────▼────┐        │
           └───────►│ Member │◄───────┘
                   │  P:40   │
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Viewer  │
                   │  P:10   │
                   └─────────┘

P = Priority (higher overrides lower in conflicts)
```

---

## Role Definitions

### Owner

**Full control over the organization.**

- **Priority:** 100 (highest)
- **Permissions:** All 143 permissions
- **System Role:** Yes
- **Typical Use:** Organization founders, billing contacts

**Unique Capabilities:**
- Billing management (`organization.billing`)
- User impersonation (`user.impersonate`)
- Organization deletion (`organization.delete`)
- All admin configuration (`admin.config`, `admin.all`)
- Backup & disaster recovery
- All policy management
- Session creation and management

---

### Administrator

**Manage users and resources without billing access.**

- **Priority:** 80
- **Permissions:** 95 permissions
- **System Role:** Yes
- **Inherits From:** None (standalone)
- **Typical Use:** IT administrators, operations managers

**Key Capabilities:**
- All debate operations
- All agent management
- User management (invite, remove, change roles)
- Organization settings (no billing)
- All workflow management
- All webhook management
- System metrics access
- Queue and orchestration management

**Cannot:**
- Access billing
- Delete organization
- Impersonate users
- Configure admin settings
- Manage backups

---

### Compliance Officer

**Manage compliance policies, data governance, and audit trails.**

- **Priority:** 75
- **Permissions:** 42 permissions
- **System Role:** Yes
- **Typical Use:** Compliance managers, data protection officers, auditors

**Key Capabilities:**
- Full data governance (classification, retention, lineage)
- PII management (read, redact, mask)
- Compliance policy management
- Full audit log access
- Vendor approval
- Session management
- MFA enforcement

**Cannot:**
- Modify debates or agents
- Create workflows
- Manage connectors
- Access billing

---

### Debate Creator

**Create, run, and manage debates.**

- **Priority:** 50
- **Permissions:** 43 permissions
- **System Role:** Yes
- **Typical Use:** Researchers, project leads, power users

**Key Capabilities:**
- Full debate operations (create, update, run, fork)
- Workflow creation and execution
- Evidence management
- Checkpoint management
- Gauntlet testing (run, compare, export)
- Marketplace publishing
- Explainability (read and batch)
- Findings management (no bulk)

**Cannot:**
- Create/delete agents
- Manage connectors
- Invite/remove users
- Access admin features

---

### Team Lead

**Manage team membership and share resources.**

- **Priority:** 55
- **Permissions:** 45 permissions
- **System Role:** Yes
- **Inherits From:** Member
- **Typical Use:** Team managers, project coordinators

**Key Capabilities:**
- All member permissions plus:
- Team management (update, add/remove members, share)
- Quota visibility
- Cost center visibility
- Approval requests
- Findings assignment
- Memory updates

**Cannot:**
- Create/delete teams
- Create agents
- Manage billing
- Configure quotas

---

### Member

**Default organization member with standard access.**

- **Priority:** 40
- **Permissions:** 31 permissions
- **System Role:** Yes
- **Typical Use:** Standard team members, contributors

**Key Capabilities:**
- Create and run debates
- Fork debates
- Create workflows
- View analytics
- Import from marketplace
- Create evidence
- Generate API keys
- Workspace sharing

**Cannot:**
- Delete debates
- Manage agents
- Manage users
- Access admin features
- Export analytics

---

### Analyst

**View debates, analytics, and reports.**

- **Priority:** 30
- **Permissions:** 15 permissions
- **System Role:** Yes
- **Typical Use:** Data analysts, business intelligence users

**Key Capabilities:**
- Read-only access to debates
- Full analytics (read + export)
- Training data read access
- Evidence read access
- Marketplace browsing
- Findings visibility

**Cannot:**
- Create or modify any resources
- Run debates
- Execute workflows

---

### Viewer

**Minimal read-only access.**

- **Priority:** 10 (lowest)
- **Permissions:** 4 permissions
- **System Role:** Yes
- **Typical Use:** External stakeholders, observers

**Key Capabilities:**
- View debates
- View agents
- View organization info
- View findings

**Cannot:**
- Create, modify, or delete anything
- View analytics
- Access workflows

---

## Role Templates

Pre-defined templates for quick custom role creation:

### Engineering Template

- **Base:** Debate Creator
- **Additional:** `agent.create`, `agent.update`, `connector.create`
- **Use Case:** Engineering team with agent management

### Research Template

- **Base:** Analyst
- **Additional:** `training.create`, `debate.create`, `debate.run`
- **Use Case:** Research team with training data access

### Support Template

- **Base:** Viewer
- **Additional:** `user.read`, `organization.audit`
- **Use Case:** Support team with user visibility

### External Template

- **Base:** Viewer
- **Additional:** None
- **Use Case:** External collaborators with minimal access

---

## Creating Custom Roles

```python
from aragora.rbac.defaults import create_custom_role

# Create a custom role for data science team
role = create_custom_role(
    name="data_scientist",
    display_name="Data Scientist",
    description="Access to analytics and training data",
    permission_keys={
        "analytics.read",
        "analytics.export",
        "training.read",
        "training.create",
        "debate.read",
    },
    org_id="org_123",
    base_role="analyst",  # Inherit analyst permissions
)
```

---

## Permission Resolution

When a user has multiple roles, permissions are resolved as follows:

1. **Union of permissions:** User gets all permissions from all assigned roles
2. **Priority for conflicts:** Higher priority role wins in edge cases
3. **Inheritance included:** Parent role permissions are automatically included

```python
from aragora.rbac.defaults import get_role_permissions

# Get all permissions for team_lead (includes inherited from member and viewer)
perms = get_role_permissions("team_lead", include_inherited=True)

# Get only direct permissions (no inheritance)
direct_perms = get_role_permissions("team_lead", include_inherited=False)
```

---

## Role Assignment Best Practices

1. **Principle of Least Privilege:** Assign the minimum role needed
2. **Use Custom Roles:** Create organization-specific roles rather than over-assigning
3. **Regular Audits:** Review role assignments quarterly
4. **Separation of Duties:** Compliance Officer should not have Admin role
5. **Document Custom Roles:** Maintain documentation for custom role purposes

---

## See Also

- [RBAC Permission Reference](RBAC_PERMISSION_REFERENCE.md) - Complete permission catalog
- [Enterprise Features](ENTERPRISE_FEATURES.md) - Enterprise security capabilities
- [API Reference](API_REFERENCE.md) - Protected endpoint documentation
