# RBAC Coverage Status Report

**Last Updated:** 2026-01-26
**Coverage:** 22% (170/773 handlers protected)

## Executive Summary

This report documents the current state of Role-Based Access Control (RBAC) coverage across Aragora's HTTP handler layer. Key findings:

- **22% of handlers** have explicit RBAC protection via `@require_permission`
- **1 handler** uses ABAC (debates handler) for fine-grained resource access
- **75% of handlers** rely solely on middleware authentication
- **20+ distinct permission keys** are actively used

## Coverage Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Total Handler Files | 200 | - |
| Total Handler Functions | 773 | - |
| With RBAC Decorators | 170 (22%) | 80%+ |
| With ABAC Checks | ~10 (1%) | 20% for resource-level |
| Unprotected (auth only) | ~580 (75%) | <20% |

## Protected Handlers by Category

### Financial Operations (35 handlers)
- `accounting.py` - `finance:read`, `finance:write`
- `payments.py` - `finance:approve`
- `expenses.py` - `finance:write`
- `ap_automation.py`, `ar_automation.py` - `finance:write`, `connectors:create`

### Admin & Audit (25 handlers)
- `auditing.py`, `audit_trail.py` - `admin:audit`
- `audit_export.py` - `audit:export`
- `admin/billing.py` - `org:billing`
- `admin/system.py` - `admin:system`

### Governance & Control Plane (30 handlers)
- `policy.py` - `policies:read`, `policies:update`
- `control_plane.py` - `controlplane:*` permissions
- `compliance_handler.py` - `compliance:audit`

### Security & DR (20 handlers)
- `backup_handler.py` - `backups:read`, `backups:create`
- `dr_handler.py` - `dr:read`, `dr:drill`
- `codebase/security.py` - `secrets:read`, `secrets:scan`

## Top Permission Keys by Usage

| Permission | Uses | Category |
|------------|------|----------|
| `org:billing` | 14 | Billing |
| `finance:write` | 10 | Financial |
| `admin:audit` | 9 | Audit |
| `admin:system` | 8 | System Admin |
| `org:usage:read` | 7 | Usage Monitoring |
| `policies:read` | 6 | Governance |
| `finance:approve` | 5 | Financial |
| `finance:read` | 5 | Financial |

## High-Priority Unprotected Handlers

The following handlers lack RBAC protection and handle sensitive operations:

### Critical Risk
1. **`email_services.py`** (16 handlers) - Email integration
2. **`workflows.py`** (14 handlers) - Workflow execution
3. **`training.py`** (7 handlers) - Training data export
4. **`code_review.py`** (6 handlers) - Code review operations
5. **`gauntlet_v1.py`** (7 handlers) - Adversarial testing

### High Risk
6. **`cross_pollination.py`** (9 handlers) - Knowledge sharing
7. **`inbox/action_items.py`** (8 handlers) - Team inbox
8. **`debates/intervention.py`** (7 handlers) - Debate participation
9. **`knowledge_chat.py`** (6 handlers) - Knowledge interactions
10. **`voice/handler.py`** (4 handlers) - Voice/TTS operations

## ABAC Usage

Currently only `aragora/server/handlers/debates/handler.py` uses ABAC for fine-grained access:

```python
from aragora.server.middleware.abac import check_resource_access

access_decision = check_resource_access(
    action=Action.READ,
    resource_type=ResourceType.DEBATE,
    resource_id=debate_id,
    user_id=context.user_id,
)
```

**Recommended for ABAC expansion:**
- Document ownership checks
- Workspace membership validation
- Knowledge base access levels
- Template sharing permissions

## Permission Naming Convention

**Standard Format:** `resource:action`

Examples:
- `debates:create`, `debates:read`, `debates:delete`
- `finance:read`, `finance:write`, `finance:approve`
- `policies:read`, `policies:update`

**Control Plane Format:** `controlplane:subdomain.action`

Examples:
- `controlplane:health.read`
- `controlplane:tasks.complete`
- `controlplane:agents.read`

## Remediation Roadmap

See [ADR-017: RBAC and ABAC Unification](./adr-017-rbac-abac-unification.md) for the architectural approach.

### Phase 1: Critical Handler Protection
- Add `@require_permission` to email, workflow, and training handlers
- Estimated: 40 handlers

### Phase 2: Permission Standardization
- Migrate `admin.credits.view` to `admin:credits:view` format
- Audit all permission key names for consistency

### Phase 3: ABAC Expansion
- Apply ABAC to document and workspace handlers
- Implement ownership-based access for sensitive resources

### Phase 4: Coverage Target (80%+)
- Protect remaining handlers with appropriate permissions
- Document which handlers intentionally remain public

## Related Documentation

- [ADR-017: RBAC and ABAC Unification](./adr-017-rbac-abac-unification.md)
- [Enterprise Features: RBAC](../enterprise/features.md#rbac)
- [Security Overview](../security/overview.md)
