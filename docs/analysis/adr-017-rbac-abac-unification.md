# ADR-017: RBAC and ABAC Authorization Unification

## Status
**Proposed** - Awaiting team review

## Context

Aragora currently has two separate authorization systems:

1. **RBAC (Role-Based Access Control)** in `aragora/rbac/`
   - Mature, well-tested implementation (15+ test files)
   - Decorator-based: `@require_permission("debates.create")`
   - Cached with Redis support and O(1) invalidation
   - Rich conditions: time, IP, owner, status, tags
   - Audited with `AuthorizationAuditor`
   - 50+ built-in permissions, 6 system roles
   - Used in 30+ handler files

2. **ABAC (Attribute-Based Access Control)** in `aragora/server/middleware/abac.py`
   - Single-file implementation
   - Policy-registry pattern with `AccessEvaluator`
   - Attribute-based: subject, resource, action, environment
   - Default policies for debate, workspace, document, etc.
   - Used only in debates handler
   - No caching

This creates several problems:
- **Two code paths** for authorization
- **Different action enums** (RBAC: strings, ABAC: hardcoded enums)
- **Separate audit trails**
- **No caching for ABAC**
- **Maintenance burden** of two systems

## Decision

We propose a **layered approach** with RBAC as the foundation:

### Phase 1: Unify Enums and Types

Create shared types in `aragora/rbac/types.py`:

```python
from enum import Enum

class Action(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"  # Maps to ABAC "write"
    DELETE = "delete"
    SHARE = "share"
    EXECUTE = "execute"  # Maps to ABAC "run"
    EXPORT = "export"
    ADMIN = "admin"

class ResourceType(str, Enum):
    DEBATE = "debate"
    WORKSPACE = "workspace"
    DOCUMENT = "document"
    TEMPLATE = "template"
    AGENT = "agent"
    WORKFLOW = "workflow"
    EVIDENCE = "evidence"
    KNOWLEDGE = "knowledge"
```

### Phase 2: Dual-Check Pattern

For resource-level access (debates, documents), use both systems:

```python
from aragora.rbac.checker import check_permission
from aragora.rbac.abac import check_resource_attributes

async def authorized_resource_access(
    context: AuthorizationContext,
    resource_type: ResourceType,
    action: Action,
    resource: Resource,
) -> AccessDecision:
    # 1. Check RBAC permission (cached)
    rbac_decision = check_permission(
        context,
        f"{resource_type.value}.{action.value}",
        resource_id=resource.id
    )
    if not rbac_decision.allowed:
        return rbac_decision

    # 2. Check ABAC attributes (ownership, workspace, sensitivity)
    abac_decision = check_resource_attributes(
        subject=context,
        resource=resource,
        action=action
    )

    return abac_decision
```

### Phase 3: Migrate ABAC Policies to RBAC

Convert ABAC `ResourcePolicy` definitions to RBAC conditional permissions:

```python
# Before (ABAC)
DEFAULT_POLICIES = {
    ResourceType.DEBATE: ResourcePolicy(
        allow_public_read=False,
        owner_actions={Action.READ, Action.WRITE, Action.DELETE},
        member_actions={Action.READ, Action.WRITE},
        shared_actions={Action.READ},
    )
}

# After (RBAC with conditions)
DEBATE_PERMISSIONS = [
    Permission(
        resource="debates",
        action="read",
        conditions={
            "owner_or_member_or_shared": True
        }
    ),
    Permission(
        resource="debates",
        action="write",
        conditions={
            "owner_or_member": True
        }
    ),
]
```

### Phase 4: Consolidate Middleware

Create unified authorization middleware:

```python
# aragora/server/middleware/authorization.py

from aragora.rbac import check_permission, PermissionDeniedError
from aragora.rbac.conditions import evaluate_conditions

async def check_access(
    request: Request,
    resource_type: str,
    action: str,
    resource_id: Optional[str] = None,
) -> AccessDecision:
    context = await get_auth_context(request)

    # Check permission with automatic condition evaluation
    decision = check_permission(
        context,
        f"{resource_type}.{action}",
        resource_id=resource_id,
        evaluate_conditions=True  # New flag
    )

    # Audit the decision
    audit_access_decision(context, decision)

    return decision
```

## Consequences

### Positive
- **Single authorization code path** - easier to reason about
- **Consistent caching** - all decisions use RBAC cache
- **Unified audit trail** - one audit log format
- **Reduced maintenance** - one system to maintain
- **Better performance** - ABAC decisions cached

### Negative
- **Migration effort** - need to update debates handler
- **Risk during transition** - temporary dual systems
- **Learning curve** - team needs to understand unified model

### Neutral
- **ABAC concepts preserved** - conditions support attribute checks
- **Backward compatible** - existing RBAC decorators unchanged

## Implementation Plan

### Step 1: Create Shared Types (Low Risk)
- Add `aragora/rbac/types.py` with unified enums
- Update imports in existing code

### Step 2: Add Condition Evaluation to RBAC (Medium Risk)
- Extend `PermissionChecker` to evaluate conditions
- Add `evaluate_conditions` flag to `check_permission`

### Step 3: Migrate Debates Handler (High Risk)
- Replace `check_resource_access()` calls with RBAC
- Add workspace-scoped permissions
- Test thoroughly

### Step 4: Deprecate ABAC Module (Low Risk)
- Mark `aragora/server/middleware/abac.py` as deprecated
- Remove after one release cycle

## Alternatives Considered

### A: Keep Both Systems
- **Pro:** No migration risk
- **Con:** Continued maintenance burden, cognitive load

### B: Replace RBAC with ABAC
- **Pro:** Simpler attribute model
- **Con:** Loses caching, audit, delegation features

### C: Third-Party Solution (OPA, Casbin)
- **Pro:** Industry standard, well-tested
- **Con:** External dependency, learning curve, integration cost

## Related ADRs
- ADR-007: Selection Plugin Architecture
- ADR-009: Control Plane Architecture

## References
- [NIST ABAC Guide](https://csrc.nist.gov/publications/detail/sp/800-162/final)
- [RBAC vs ABAC Comparison](https://www.okta.com/identity-101/role-based-access-control-vs-attribute-based-access-control/)
- Current RBAC: `aragora/rbac/`
- Current ABAC: `aragora/server/middleware/abac.py`
