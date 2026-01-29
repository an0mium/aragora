# Aragora v2.5.0+ Comprehensive Development Plan

*Generated: January 29, 2026*
*Current Version: v2.4.0*

## Executive Summary

Based on comprehensive analysis across 6 dimensions (features, tests, docs, DX, security, infrastructure), Aragora is at **89% commercial readiness** with **0 production blockers**. This plan outlines prioritized next steps to achieve 95%+ readiness for enterprise GA.

### Key Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Lines of Code | 696,000+ | - |
| Tests | 45,100+ (1,220 files) | 48,000+ |
| API Endpoints | 1,177 | - |
| SDK Namespaces | 105 TS / 106 PY | 100% parity |
| RBAC Coverage | 90% handlers | 100% |
| Documentation | 375 files | Reorganized |
| Production Readiness | 72% | 90% |

---

## Priority Matrix

### Tier 1: Critical (v2.5.0 Release Blockers)
*Timeline: 1-2 weeks*

| # | Task | Impact | Effort | Owner |
|---|------|--------|--------|-------|
| 1.1 | Complete RBAC handler coverage (50 handlers) | Security | Medium | Backend |
| 1.2 | Enable input validation middleware blocking mode | Security | Low | Backend |
| 1.3 | Fix test mock cleanup in SDK tests | Quality | Low | SDK |
| 1.4 | Document SLA/SLO targets | Commercial | Low | Docs |
| 1.5 | Add container image vulnerability scanning | Security | Low | DevOps |

### Tier 2: High Priority (v2.5.0 Quality)
*Timeline: 2-3 weeks*

| # | Task | Impact | Effort | Owner |
|---|------|--------|--------|-------|
| 2.1 | Add module READMEs (debate/, agents/, memory/, server/, knowledge/) | Adoption | Medium | Docs |
| 2.2 | Increase test coverage for client module (13% → 60%) | Quality | High | QA |
| 2.3 | Increase test coverage for scheduler module (16% → 60%) | Quality | Medium | QA |
| 2.4 | Persist audit logs to database | Compliance | Medium | Backend |
| 2.5 | Add auto-migration on startup | Deployment | Medium | Backend |
| 2.6 | Configure Redis cluster mode for HA | Reliability | Medium | DevOps |

### Tier 3: Medium Priority (v2.6.0)
*Timeline: 3-4 weeks*

| # | Task | Impact | Effort | Owner |
|---|------|--------|--------|-------|
| 3.1 | Reorganize documentation (consolidate 264 files) | Adoption | High | Docs |
| 3.2 | Implement KMS for per-tenant encryption keys | Security | High | Security |
| 3.3 | Add MCP tool module tests (18 modules untested) | Quality | Medium | QA |
| 3.4 | Document upgrade path (v1→v2 migration) | Adoption | Medium | Docs |
| 3.5 | Implement GitOps with ArgoCD | Deployment | Medium | DevOps |
| 3.6 | Add network policies to K8s deployments | Security | Low | DevOps |

### Tier 4: Enhancement (v2.7.0+)
*Timeline: 4-6 weeks*

| # | Task | Impact | Effort | Owner |
|---|------|--------|--------|-------|
| 4.1 | Blue-green/canary deployment | Reliability | High | DevOps |
| 4.2 | Automated multi-region failover | Reliability | High | DevOps |
| 4.3 | Feature flags integration | Velocity | Medium | Backend |
| 4.4 | SIEM integration (if needed) | Compliance | Medium | Security |
| 4.5 | AI Act compliance mapping | Compliance | Medium | Compliance |

---

## Detailed Task Specifications

### 1.1 Complete RBAC Handler Coverage

**Current State:** 90% of 562 handlers have permission checks
**Gap:** ~50 handlers missing `@require_permission` decorators

**Files to audit:**
```
aragora/server/handlers/
├── admin/           # Priority: HIGH
├── integrations/    # Priority: HIGH
├── inbox/           # Priority: MEDIUM
├── features/        # Priority: MEDIUM
└── bots/            # Priority: LOW (already standardized)
```

**Implementation:**
1. Run RBAC audit script: `python scripts/audit_rbac_coverage.py`
2. Add `@require_permission("resource:action")` to uncovered handlers
3. Add integration tests for each permission check
4. Update docs/RBAC.md with permission matrix

**Acceptance Criteria:**
- [ ] 100% handler coverage
- [ ] Integration tests for new permissions
- [ ] RBAC.md updated with all 128+ resource types

---

### 1.2 Enable Input Validation Blocking Mode

**Current State:** Validation middleware runs in warning mode (`blocking=False`)
**Risk:** Invalid requests accepted, potential injection attacks

**Files:**
- `aragora/server/middleware/validation.py` (line 270)

**Implementation:**
```python
# Change default from:
blocking: bool = False
# To:
blocking: bool = True
```

**Acceptance Criteria:**
- [ ] Blocking mode enabled by default
- [ ] Invalid requests return 400 with structured error
- [ ] All existing tests pass (may need test fixes)

---

### 1.3 Fix SDK Test Mock Cleanup

**Current State:** `new-namespaces.test.ts` had mock cleanup issues
**Status:** Fixed in session - need to verify CI passes

**Files:**
- `sdk/typescript/src/__tests__/new-namespaces.test.ts`

**Acceptance Criteria:**
- [ ] All 174 SDK tests pass
- [ ] CI/CD green on main branch

---

### 1.4 Document SLA/SLO Targets

**Current State:** SLO_DEFINITIONS.md exists but incomplete
**Gap:** No legally-binding SLA for enterprise customers

**Deliverables:**
1. `docs/SLA.md` - Service Level Agreement template
2. Update `docs/SLO_DEFINITIONS.md` with metrics:
   - API availability: 99.9%
   - Debate completion: <30s P95
   - Webhook delivery: <5s P99
   - Error rate: <1%

**Acceptance Criteria:**
- [ ] SLA document with tiered support levels
- [ ] Prometheus SLO recording rules
- [ ] Grafana SLO dashboard

---

### 1.5 Add Container Image Scanning

**Current State:** No vulnerability scanning in CI/CD
**Risk:** Deploying images with known CVEs

**Implementation:**
Add to `.github/workflows/docker.yml`:
```yaml
- name: Scan image for vulnerabilities
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
    severity: 'CRITICAL,HIGH'
    exit-code: '1'
```

**Acceptance Criteria:**
- [ ] Trivy or Snyk scanning in Docker workflow
- [ ] Fails build on CRITICAL/HIGH vulnerabilities
- [ ] Weekly scheduled scan of deployed images

---

### 2.1 Add Module READMEs

**Current State:** Only 2/100+ subsystems have READMEs
**Gap:** New developers can't understand module purposes

**Required READMEs:**
```
aragora/debate/README.md       # 123 files
aragora/agents/README.md       # 35 files
aragora/memory/README.md       # 23 files
aragora/server/README.md       # 72 files
aragora/knowledge/README.md    # 15 files
```

**Template:**
```markdown
# Module Name

Brief description (1-2 sentences).

## Quick Start
\`\`\`python
from aragora.module import MainClass
\`\`\`

## Key Components
- `ComponentA` - Purpose
- `ComponentB` - Purpose

## Architecture
[Brief architecture description]

## Related Documentation
- [Link to detailed docs]
```

---

### 2.2-2.3 Increase Test Coverage

**Modules with Critical Coverage Gaps:**

| Module | Current | Target | Priority Tests |
|--------|---------|--------|----------------|
| client | 13% (5/38) | 60% | Resource modules (16 untested) |
| scheduler | 16% (2/12) | 60% | Rotation handlers, DR drills |
| MCP | 14% (3/21) | 50% | 18 tool modules |
| training | 16% (2/12) | 50% | Exporters (DPO, SFT) |
| plugins | 9% (1/11) | 50% | Plugin lifecycle |

**Test Strategy:**
1. Create test templates for each module type
2. Use mocking for external dependencies
3. Focus on integration tests over unit tests (matches existing patterns)

---

### 2.4 Persist Audit Logs to Database

**Current State:** Audit logs in-memory with 100-entry limit
**Risk:** Audit trail lost on restart, compliance failure

**Implementation:**
1. Create `audit_log` table in PostgreSQL
2. Add async batch writer for audit events
3. Implement retention policy (90 days default)
4. Add query endpoint for audit log retrieval

**Files:**
- `aragora/tenancy/isolation.py` (line 439+)
- New: `aragora/audit/persistence.py`

---

### 2.5 Add Auto-Migration on Startup

**Current State:** Manual `python -m aragora.persistence.migrations.runner` required
**Risk:** Deployment fails if migrations forgotten

**Implementation:**
```python
# In aragora/server/startup.py
async def run_startup_migrations():
    from aragora.persistence.migrations.runner import MigrationRunner
    runner = MigrationRunner()
    pending = runner.get_pending_migrations()
    if pending:
        logger.info(f"Running {len(pending)} pending migrations")
        runner.upgrade()
```

---

### 2.6 Configure Redis Cluster Mode

**Current State:** Single Redis instance in docker-compose
**Risk:** Single point of failure

**Options:**
1. Redis Sentinel (3 nodes) - Simpler, automatic failover
2. Redis Cluster (6 nodes) - Sharding, higher throughput

**Recommendation:** Start with Sentinel for v2.5.0

---

## Documentation Reorganization (3.1)

**Current Issues:**
- 264 files in docs/ root - overwhelming
- Duplicate content across 8+ locations
- No clear progression path for users

**Proposed Structure:**
```
docs/
├── getting-started/          # New users
│   ├── quickstart.md
│   ├── installation.md
│   └── first-debate.md
├── guides/                   # How-to guides
│   ├── deployment/
│   ├── integration/
│   └── administration/
├── reference/                # Technical reference
│   ├── api/
│   ├── cli/
│   └── configuration/
├── architecture/             # System design
│   ├── adr/
│   └── diagrams/
├── enterprise/               # Enterprise features
│   ├── security/
│   ├── compliance/
│   └── multi-tenancy/
└── INDEX.md                  # Central navigation
```

---

## Security Improvements (3.2)

**Per-Tenant Encryption Key Derivation:**

**Current (VULNERABLE):**
```python
key = hashlib.sha256(f"aragora_tenant_key_{tenant_id}".encode())
```

**Proposed (KMS-backed):**
```python
from aragora.security.kms import get_tenant_key
key = await kms_client.get_data_key(
    key_id=f"alias/aragora/tenant/{tenant_id}",
    context={"tenant": tenant_id}
)
```

---

## Success Metrics

### v2.5.0 Release Criteria
- [ ] 100% RBAC handler coverage
- [ ] Input validation blocking enabled
- [ ] All 174 SDK tests pass
- [ ] SLA documentation complete
- [ ] Container scanning enabled
- [ ] CI/CD fully green

### v2.6.0 Goals
- [ ] Test coverage: client 60%, scheduler 60%, MCP 50%
- [ ] Documentation reorganized
- [ ] KMS integration for tenant keys
- [ ] Redis Sentinel configured
- [ ] Auto-migration on startup

### v2.7.0+ Goals
- [ ] Blue-green deployments
- [ ] Automated failover
- [ ] Feature flags
- [ ] 95%+ production readiness score

---

## Resource Allocation

**Recommended Team Focus:**

| Role | Primary Tasks | Secondary Tasks |
|------|---------------|-----------------|
| Backend | 1.1, 1.2, 2.4, 2.5 | 3.2 |
| SDK | 1.3, SDK parity | TypeScript types |
| QA | 2.2, 2.3, 3.3 | Integration tests |
| DevOps | 1.5, 2.6, 3.5, 3.6 | 4.1, 4.2 |
| Docs | 1.4, 2.1, 3.1, 3.4 | READMEs |
| Security | 3.2, 4.4 | Audit |

---

## Appendix: Agent Analysis Sources

1. **Feature Completeness** - 89% overall, core stable
2. **Test Coverage** - 45,100 tests, 583 skipped, key module gaps
3. **Documentation** - 375 files, disorganized, 94.8% docstrings
4. **SDK/DX** - 100% parity, excellent types, good error handling
5. **Security** - RBAC solid, validation/encryption gaps
6. **Infrastructure** - 72% prod-ready, needs HA and scanning

*This plan synthesizes findings from 6 parallel exploration agents.*
