# RBAC Permission Reference

Complete catalog of all permissions in the Aragora RBAC system.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Permissions | 156+ |
| Permission Categories | 28 |
| System Roles | 8 |
| Role Templates | 4 |

## Permission Format

Permissions use the format `resource.action`:
- **Resource**: The type of entity (e.g., `debate`, `agent`, `user`)
- **Action**: The operation (e.g., `create`, `read`, `update`, `delete`)

Example: `debate.create` grants the ability to create new debates.

Some handler permissions use a colon‑separated format (`resource:action`), e.g.
`debates:read`. Both formats are supported by the permission checker.

---

## Core Permissions

### Debate Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `debate.create` | Create new debates | Owner, Admin, Debate Creator, Team Lead, Member |
| `debate.read` | View debates | All roles |
| `debate.update` | Modify debate settings | Owner, Admin, Debate Creator, Team Lead |
| `debate.delete` | Delete debates | Owner, Admin |
| `debate.run` | Execute debates | Owner, Admin, Debate Creator, Team Lead, Member |
| `debate.stop` | Stop running debates | Owner, Admin, Debate Creator, Team Lead, Member |
| `debate.fork` | Fork debates | Owner, Admin, Debate Creator, Team Lead, Member |

### Debate Service Permissions (colon format)

| Permission | Description | Roles |
|------------|-------------|-------|
| `debates:read` | Read debate details and intervention state | Owner, Admin, Debate Creator, Team Lead, Member |
| `debates:write` | Pause/resume debates, inject arguments, adjust weights/thresholds | Owner, Admin, Debate Creator, Team Lead |

### Decision Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `decisions:create` | Create a decision request | Owner, Admin, Debate Creator, Team Lead, Member |
| `decisions:read` | Read decision results/status | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |

### Agent Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `agent.create` | Create new agents | Owner, Admin |
| `agent.read` | View agent configurations | All roles |
| `agent.update` | Modify agent settings | Owner, Admin |
| `agent.delete` | Delete agents | Owner, Admin |
| `agent.deploy` | Deploy agents to production | Owner, Admin |

### User Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `user.read` | View user profiles | Owner, Admin, Compliance Officer, Debate Creator, Team Lead, Analyst, Member |
| `user.invite` | Invite new users | Owner, Admin |
| `user.remove` | Remove users | Owner, Admin |
| `user.change_role` | Change user roles | Owner, Admin |
| `user.impersonate` | Impersonate users (support) | Owner |

### Organization Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `organization.read` | View organization info | All roles |
| `organization.update` | Modify organization settings | Owner, Admin |
| `organization.billing` | Manage billing | Owner |
| `organization.audit` | View audit logs | Owner, Admin, Compliance Officer |
| `organization.export` | Export organization data | Owner, Admin |
| `organization.delete` | Delete organization | Owner |

### Auth & SSO Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `auth:read` | Access SSO login, callback, refresh, logout, provider list | Owner, Admin |

### API Key Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `api.generate_key` | Generate API keys | Owner, Admin, Debate Creator, Team Lead, Member |
| `api.revoke_key` | Revoke API keys | Owner, Admin |
| `api_key.list_all` | List all organization API keys | Owner |
| `api_key.export_secret` | Export API key secrets | Owner |

---

## Memory & Knowledge Permissions

### Memory Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `memory.read` | Read memory data | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `memory.update` | Update memory data | Owner, Admin, Debate Creator, Team Lead |
| `memory.delete` | Delete memory data | Owner, Admin |

### Evidence Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `evidence.read` | View evidence | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `evidence.create` | Add evidence | Owner, Admin, Debate Creator, Team Lead, Member |
| `evidence.delete` | Delete evidence | Owner |

### Document & Upload Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `documents.read` | View document metadata and queries | Owner, Admin, Analyst |
| `documents.create` | Upload documents | Owner, Admin |
| `upload.create` | Create document folder uploads | Owner, Admin |

### Speech Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `speech.create` | Generate speech transcripts | Owner, Admin |

### Knowledge Analytics Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `knowledge.analytics.read` | View knowledge mound statistics, sharing stats, federation stats | Owner, Admin, Analyst |

### Knowledge Maintenance Permissions (colon format)

| Permission | Description | Roles |
|------------|-------------|-------|
| `knowledge:read` | Read deduplication and maintenance reports | Owner, Admin, Analyst |
| `federation:read` | View federation status and regions | Owner, Admin, Analyst |

### Knowledge Notifications Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `knowledge.notifications.read` | View sharing notifications, preferences | All roles |
| `knowledge.notifications.write` | Mark notifications as read, update preferences | All roles |

---

## Workflow & Automation Permissions

### Workflow Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `workflow.create` | Create workflows | Owner, Admin, Debate Creator, Team Lead, Member |
| `workflow.read` | View workflows | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `workflow.run` | Execute workflows | Owner, Admin, Debate Creator, Team Lead, Member |
| `workflow.delete` | Delete workflows | Owner, Admin |

### Checkpoint Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `checkpoint.read` | View checkpoints | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `checkpoint.create` | Create checkpoints | Owner, Admin, Debate Creator, Team Lead, Member |
| `checkpoint.delete` | Delete checkpoints | Owner, Admin |

### Template Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `template.create` | Create templates | Owner, Admin |
| `template.read` | View templates | Owner, Admin |
| `template.update` | Modify templates | Owner, Admin |
| `template.delete` | Delete templates | Owner, Admin |

---

## Analytics & Training Permissions

### Analytics Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `analytics.read` | View analytics | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `analytics.export` | Export analytics data | Owner, Admin, Analyst |

### Training Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `training.read` | View training data | Owner, Admin, Analyst |
| `training.create` | Create training data | Owner, Admin |

## Billing & Accounting Permissions (colon format)

| Permission | Description | Roles |
|------------|-------------|-------|
| `costs:read` | Read cost dashboards and recommendations | Owner, Admin, Analyst |
| `ap:read` | Read accounts payable invoices/forecasts/discounts | Owner, Admin |
| `ar:read` | Read accounts receivable invoices/reports/collections | Owner, Admin |

---

## Connector & Integration Permissions

### Connector Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `connector.read` | View connectors | Owner, Admin |
| `connector.create` | Create connectors | Owner, Admin |
| `connector.delete` | Delete connectors | Owner, Admin |
| `connector.update` | Update connector settings | Owner |
| `connector.authorize` | Authorize connector OAuth | Owner |
| `connector.rotate` | Rotate connector credentials | Owner |
| `connector.test` | Test connector connectivity | Owner |
| `connector.rollback` | Rollback connector version | Owner |

### Webhook Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `webhook.read` | View webhooks | Owner, Admin |
| `webhook.create` | Create webhooks | Owner, Admin |
| `webhook.delete` | Delete webhooks | Owner, Admin |
| `webhook.admin` | Administer all webhooks | Owner, Admin |

### Bot Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `bots.read` | View bot status (Discord, Teams, Telegram, etc.) | Owner, Admin |
| `bots.write` | Configure bot settings | Owner, Admin |

---

## Quality & Testing Permissions

### Code Review Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `code_review.read` | View code review results | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `code_review.write` | Create code reviews and scans | Owner, Admin, Debate Creator, Team Lead, Member |

### Codebase Audit Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `codebase_audit.read` | View security scan results, findings, dashboard | Owner, Admin, Compliance Officer |
| `codebase_audit.write` | Run security scans (SAST, secrets, dependencies) | Owner, Admin |

### Gauntlet Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `gauntlet.run` | Run gauntlet tests | Owner, Admin, Debate Creator, Team Lead, Member |
| `gauntlet.read` | View gauntlet results | Owner, Admin, Debate Creator, Team Lead, Analyst, Compliance Officer, Member |
| `gauntlet.delete` | Delete gauntlet results | Owner, Admin |
| `gauntlet.sign` | Sign gauntlet receipts | Owner, Admin |
| `gauntlet.compare` | Compare gauntlet runs | Owner, Admin, Debate Creator |
| `gauntlet.export` | Export gauntlet data | Owner, Admin, Debate Creator |

### Findings Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `findings.read` | View findings | All roles |
| `findings.update` | Update findings | Owner, Admin, Debate Creator, Team Lead, Compliance Officer, Member |
| `findings.assign` | Assign findings | Owner, Admin, Debate Creator, Team Lead |
| `findings.bulk` | Bulk operations on findings | Owner, Admin |

### Explainability Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `explainability.read` | View explanations | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `explainability.batch` | Generate batch explanations | Owner, Admin, Debate Creator |

---

## Marketplace Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `marketplace.read` | Browse marketplace | Owner, Admin, Debate Creator, Team Lead, Analyst, Member |
| `marketplace.publish` | Publish to marketplace | Owner, Admin, Debate Creator |
| `marketplace.import` | Import from marketplace | Owner, Admin, Debate Creator, Team Lead, Member |
| `marketplace.rate` | Rate marketplace items | Owner, Admin, Debate Creator, Team Lead, Member |
| `marketplace.review` | Review marketplace items | Owner, Admin, Debate Creator, Team Lead, Member |
| `marketplace.delete` | Delete marketplace items | Owner, Admin |

---

## Decision & Policy Permissions

### Decision Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `decision.create` | Create decisions | Owner, Admin, Debate Creator, Team Lead, Member |
| `decision.read` | View decisions | Owner, Admin, Debate Creator, Team Lead, Member |

### Policy Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `policy.read` | View policies | Owner |
| `policy.create` | Create policies | Owner |
| `policy.update` | Update policies | Owner |
| `policy.delete` | Delete policies | Owner |

### Compliance Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `compliance.read` | View compliance status | Owner |
| `compliance.update` | Update compliance settings | Owner |
| `compliance.check` | Run compliance checks | Owner |

---

## Enterprise Data Governance Permissions

### Data Classification Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `data_classification.read` | View data classifications | Owner, Compliance Officer |
| `data_classification.classify` | Classify data | Owner, Compliance Officer |
| `data_classification.update` | Update classifications | Owner, Compliance Officer |

### Data Retention Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `data_retention.read` | View retention policies | Owner, Compliance Officer |
| `data_retention.update` | Update retention policies | Owner, Compliance Officer |

### Data Lineage Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `data_lineage.read` | View data lineage | Owner, Compliance Officer |

### PII Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `pii.read` | View PII data | Owner, Compliance Officer |
| `pii.redact` | Redact PII | Owner, Compliance Officer |
| `pii.mask` | Mask PII | Owner, Compliance Officer |

---

## Enterprise Compliance Permissions

### Compliance Policy Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `compliance_policy.read` | View compliance policies | Owner, Compliance Officer |
| `compliance_policy.update` | Update compliance policies | Owner, Compliance Officer |
| `compliance_policy.enforce` | Enforce compliance policies | Owner, Compliance Officer |

### Audit Log Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `audit_log.read` | View audit logs | Owner, Compliance Officer |
| `audit_log.export` | Export audit logs | Owner, Compliance Officer |
| `audit_log.search` | Search audit logs | Owner, Compliance Officer |
| `audit_log.stream` | Stream audit logs | Owner, Compliance Officer |
| `audit_log.configure` | Configure audit logging | Owner, Compliance Officer |
| `audit_log.delete` | Delete audit logs | Owner |

### Vendor Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `vendor.read` | View vendors | Owner, Compliance Officer |
| `vendor.approve` | Approve vendors | Owner, Compliance Officer |

---

## Enterprise Team & Workspace Permissions

### Team Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `team.create` | Create teams | Owner, Admin |
| `team.read` | View teams | Owner, Admin, Team Lead |
| `team.update` | Update team settings | Owner, Admin, Team Lead |
| `team.delete` | Delete teams | Owner, Admin |
| `team.add_member` | Add team members | Owner, Admin, Team Lead |
| `team.remove_member` | Remove team members | Owner, Admin, Team Lead |
| `team.share` | Share resources with team | Owner, Admin, Team Lead |
| `team.dissolve` | Dissolve teams | Owner |

### Workspace Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `workspace.create` | Create workspaces | Owner, Admin |
| `workspace.read` | View workspaces | Owner, Admin, Member |
| `workspace.update` | Update workspaces | Owner, Admin |
| `workspace.delete` | Delete workspaces | Owner, Admin |
| `workspace.member_add` | Add workspace members | Owner, Admin |
| `workspace.member_remove` | Remove workspace members | Owner, Admin |
| `workspace.member_change_role` | Change member roles | Owner, Admin |
| `workspace.share` | Share workspace resources | Owner, Admin, Member |

---

## Enterprise Cost & Quota Permissions

### Quota Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `quota.read` | View quotas | Owner, Admin, Team Lead |
| `quota.update` | Update quotas | Owner, Admin |
| `quota.override` | Override quota limits | Owner |

### Cost Center Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `cost_center.read` | View cost centers | Owner, Admin, Team Lead |
| `cost_center.update` | Update cost centers | Owner, Admin |

### Budget Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `budget.read` | View budgets | Owner, Admin |
| `budget.update` | Update budgets | Owner, Admin |
| `budget.override` | Override budget limits | Owner |

### Billing Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `billing.read` | View billing info | Owner |
| `billing.recommendations_read` | View cost recommendations | Owner |
| `billing.recommendations_apply` | Apply cost recommendations | Owner |
| `billing.forecast_read` | View billing forecasts | Owner |
| `billing.forecast_simulate` | Simulate billing scenarios | Owner |
| `billing.export_history` | Export billing history | Owner |

---

## Enterprise Session & Auth Permissions

### Session Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `session.create` | Create sessions | Owner |
| `session.read` | View sessions | Owner, Compliance Officer |
| `session.revoke` | Revoke sessions | Owner, Compliance Officer |

### Authentication Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `authentication.reset_password` | Reset user passwords | Owner |
| `authentication.require_mfa` | Require MFA for users | Owner, Compliance Officer |

---

## Enterprise Approval Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `approval.request` | Request approvals | Owner, Admin, Team Lead |
| `approval.grant` | Grant approvals | Owner, Admin, Compliance Officer |
| `approval.read` | View approvals | Owner, Admin, Team Lead, Compliance Officer |

---

## Enterprise Role Management Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `role.create` | Create custom roles | Owner |
| `role.read` | View roles | Owner, Admin |
| `role.update` | Update roles | Owner |
| `role.delete` | Delete roles | Owner |

---

## System Operations Permissions

### Control Plane Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `control_plane.read` | View control plane status, agents, queue, metrics | Owner, Admin |
| `control_plane.write` | Pause/resume agents, prioritize queue | Owner, Admin |
| `control_plane.submit` | Submit tasks | Owner |
| `control_plane.cancel` | Cancel tasks | Owner |
| `control_plane.deliberate` | Trigger deliberation | Owner |

### Queue Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `queue.read` | View queue status | Owner, Admin |
| `queue.manage` | Manage queue items | Owner, Admin |
| `queue.admin` | Queue administration | Owner, Admin |

### Nomic Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `nomic.read` | View nomic loop status | Owner, Admin |
| `nomic.admin` | Nomic administration | Owner, Admin |

### Orchestration Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `orchestration.read` | View orchestration status | Owner, Admin |
| `orchestration.execute` | Execute orchestration | Owner, Admin |

### System Health Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `system.health_read` | View system health | Owner, Admin |

---

## Admin Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `admin.config` | System configuration | Owner |
| `admin.metrics` | View system metrics | Owner, Admin |
| `admin.features` | Toggle feature flags | Owner |
| `admin.all` | Full admin access | Owner |

---

## Backup & Disaster Recovery Permissions

### Backup Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `backup.create` | Create backups | Owner |
| `backup.read` | View backups | Owner |
| `backup.restore` | Restore from backup | Owner |
| `backup.delete` | Delete backups | Owner |

### Disaster Recovery Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `disaster_recovery.read` | View DR status | Owner |
| `disaster_recovery.execute` | Execute DR procedures | Owner |

---

## Permission Lookup Functions

```python
from aragora.rbac.defaults import (
    get_permission,
    get_role,
    get_role_permissions,
    SYSTEM_PERMISSIONS,
    SYSTEM_ROLES,
)

# Get a permission by key
perm = get_permission("debate.create")
print(perm.description)  # "Create new debates"

# Get a role by name
role = get_role("admin")
print(role.display_name)  # "Administrator"

# Get all permissions for a role (including inherited)
perms = get_role_permissions("team_lead", include_inherited=True)
print(len(perms))  # 35+
```

---

## See Also

- [RBAC Role Hierarchy](RBAC_ROLE_HIERARCHY.md) - Role definitions and inheritance
- [Enterprise Features](ENTERPRISE_FEATURES.md) - Enterprise security capabilities
- [API Reference](API_REFERENCE.md) - Protected endpoint documentation
