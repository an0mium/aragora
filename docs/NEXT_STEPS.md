# NEXT_STEPS.md - Prioritized Technical Improvement Plan

> Generated: 2026-01-30 | Based on 8-agent comprehensive assessment

## Executive Summary

This plan synthesizes findings from an 8-agent comprehensive assessment of the Aragora codebase covering type safety, test suite health, technical debt, architecture, documentation, security, SDK/API, and performance.

**Overall Health Score: 8.1/10** - The codebase is in strong shape with excellent documentation (9/10), robust security (8.7/10), and healthy test coverage (95K+ tests). Key areas for improvement are type annotations, orchestrator decomposition, and performance observability gaps.

### Assessment Scores

| Area | Score | Key Finding |
|------|-------|-------------|
| Documentation | 9/10 | 401 markdown files, 120-170% docstring coverage |
| Security | 8.7/10 | Excellent secrets management, auth, RBAC, encryption |
| Architecture | 8.2/10 | 310+ modules, minimal circular imports |
| Test Suite | 8/10 | 95,153 tests, 65% coverage target |
| Performance | 7.2/10 | Good SLOs, needs tracing integration |
| Type Safety | 6/10 | 1,476 errors across 315 files |

---

## Immediate Priorities (Next 1-2 Weeks)

### 1. Fix Critical Type Annotation Gaps in Handlers

**What:** Add type annotations to the 401+ handler functions with `no-untyped-def` errors. Focus on `ServerContext` return types and request parameter typing.

**Why:**
- Type errors represent 34% of all mypy issues
- Handler type safety directly impacts API reliability
- Enables better IDE support for 345+ handler modules

**Effort:** 3-5 days (parallelizable across team)

**Dependencies:** None

**Files to prioritize:**
- `aragora/server/handlers/debates/handler.py` (1,675 lines)
- `aragora/server/handlers/admin/dashboard.py` (1,732 lines)
- `aragora/server/handlers/control_plane.py` (1,726 lines)

---

### 2. Fix 2 Identified Flaky Tests

**What:** Stabilize the 2 flaky tests identified in the test suite assessment.

**Why:**
- Flaky tests erode CI reliability
- Block confident merging of changes
- Low effort, high impact on developer experience

**Effort:** 0.5-1 day

**Dependencies:** None

---

### 3. Address MFA Bypass for Service Accounts

**What:** Review and document MFA bypass policy for service accounts, add audit logging for bypass events.

**Why:**
- Security assessment flagged this as needing attention
- Service accounts with MFA bypass are potential attack vectors
- Compliance requirements may mandate tracking

**Effort:** 1-2 days

**Dependencies:** None

**Location:** `aragora/auth/` - review OIDC/SAML integration points

---

### 4. Fix Token Validation Dev Mode Fallback

**What:** Ensure token validation in development mode cannot leak sensitive data or allow unauthorized access.

**Why:**
- Security assessment identified this as a potential issue
- Development shortcuts should not compromise production-like behavior

**Effort:** 0.5-1 day

**Dependencies:** None

---

## Short-Term (1-4 Weeks)

### 5. Decompose orchestrator.py (2,520 lines) - CRITICAL

**What:** Extract remaining logic from `orchestrator.py` into dedicated modules following the established phase pattern.

**Why:**
- Largest file in codebase at 2,520 lines
- Technical debt assessment marked this as CRITICAL
- Pattern already established with `aragora/debate/phases/` (35 modules)

**Effort:** 5-7 days

**Dependencies:** None (well-isolated)

**Approach:**
1. Identify remaining cohesive blocks in `orchestrator.py`
2. Create new modules following `orchestrator_hooks.py`, `orchestrator_memory.py`, `orchestrator_agents.py` pattern
3. Maintain backward compatibility via re-exports
4. Add unit tests for extracted modules

---

### 6. Apply Strategy Pattern to ArenaConfig

**What:** Refactor ArenaConfig (50+ optional fields) to use strategy/builder patterns for logical groupings.

**Why:**
- Current config has ~400 lines of field definitions
- Difficult to understand which fields relate to which features
- Config groups already partially documented

**Effort:** 3-5 days

**Dependencies:** Complete item #5 (orchestrator decomposition)

---

### 7. Complete Logging Migration (threading.local -> contextvars)

**What:** Migrate remaining `threading.local` usages to `contextvars` for proper async context propagation.

**Why:**
- Performance assessment identified incomplete migration
- `contextvars` properly handles async context in Python 3.7+
- Current mix causes context loss in async code paths

**Effort:** 2-3 days

**Dependencies:** None

---

### 8. Handler Base Class Refactoring

**What:** Create typed handler base classes that enforce consistent signatures.

**Why:**
- 262 attribute definition errors in handlers
- Inconsistent method signatures cause type errors
- Enables better testing via dependency injection

**Effort:** 3-4 days

**Dependencies:** Item #1 (handler annotations)

---

### 9. v1 API Sunset Preparation

**What:** Prepare migration path for v1 API sunset (deadline: 2026-06-01).

**Why:**
- 6 months until sunset
- SDK assessment confirmed sunset date
- Need migration guides and deprecation warnings

**Effort:** 3-5 days

**Dependencies:** None

**Actions:**
1. Audit all v1-specific endpoints
2. Create migration guide (v1 -> v2)
3. Add sunset headers to v1 endpoints
4. Document breaking changes

---

## Medium-Term (1-2 Months)

### 10. Integrate Distributed Tracing (OpenTelemetry)

**What:** Connect existing tracing infrastructure to OpenTelemetry for distributed tracing.

**Why:**
- Performance assessment identified this as a GAP
- Tracing middleware exists but not integrated with external collectors
- Critical for production debugging

**Effort:** 5-7 days

**Dependencies:** Item #7 (logging migration)

---

### 11. Establish Load Testing Framework

**What:** Expand existing Locust-based load testing into a comprehensive framework.

**Why:**
- Performance assessment identified missing load testing framework
- SLO targets defined but not automated

**Effort:** 7-10 days

**Dependencies:** None

---

### 12. Handler Consolidation

**What:** Consolidate 345+ handler modules into logical domains.

**Why:**
- Architecture assessment identified consolidation opportunity
- Many small handlers could be grouped
- Reduces import complexity

**Effort:** 10-15 days

**Dependencies:** Items #1, #8 (handler typing and base classes)

---

### 13. Reduce Skipped Tests

**What:** Address 583 skipped tests (57% missing_feature, 33% optional_dependency).

**Why:**
- Skipped tests represent untested code paths
- Missing features may now be implemented

**Effort:** 5-7 days (spread over time)

**Dependencies:** None

---

### 14. Address Code Duplication

**What:** Extract duplicated patterns into shared utilities (10+ common classes identified).

**Why:**
- Technical debt assessment identified duplication
- Reduces maintenance burden

**Effort:** 3-5 days

**Dependencies:** None

---

## Long-Term (2-3 Months)

### 15. Large File Decomposition Campaign

**What:** Systematically address 227 files over 1000 lines.

**Priority files:**
- `orchestrator.py` (2,520 lines) - Item #5
- `workflows.py` (1,941 lines)
- `gauntlet.py` (1,938 lines)
- `permissions.py` (1,873 lines)

**Effort:** 2-3 weeks (ongoing)

**Dependencies:** Items #5, #6

---

### 16. TypeScript SDK Consolidation (v3.0.0)

**What:** Complete TypeScript SDK consolidation.

**Why:**
- SDK assessment noted consolidation in progress
- Target: Single unified `@aragora/sdk` package

**Effort:** 2-3 weeks

**Dependencies:** Item #9 (v1 API sunset preparation)

---

### 17. Breaking Changes Documentation System

**What:** Establish automated breaking changes documentation.

**Why:**
- Documentation assessment identified minor gap
- Improves upgrade experience

**Effort:** 3-5 days

**Dependencies:** None

---

### 18. Complete Return Type Annotations

**What:** Address 65 return type violations in handlers.

**Effort:** 2-3 days

**Dependencies:** Items #1, #8 (handler typing)

---

## Summary by Impact

| Priority | Items | Total Effort | Unblocks |
|----------|-------|--------------|----------|
| Immediate | #1-4 | 5-9 days | Handler typing, Security |
| Short-Term | #5-9 | 16-24 days | Architecture, API stability |
| Medium-Term | #10-14 | 30-44 days | Observability, Testing |
| Long-Term | #15-18 | 5-8 weeks | Maintainability |

## Quick Wins (Can Start Today)

1. **Fix 2 flaky tests** (#2) - 0.5 days
2. **Fix token validation dev mode** (#4) - 0.5 days
3. **Document MFA bypass policy** (#3) - 1 day
4. **Start handler type annotations** (#1) - parallelizable

## Dependencies Graph

```
#1 Handler Annotations ──────┬──► #8 Handler Base Classes ──► #12 Handler Consolidation
                             │
#5 Orchestrator Decompose ───┼──► #6 ArenaConfig Strategy
                             │
#7 Logging Migration ────────┴──► #10 Distributed Tracing

#9 v1 API Sunset ───────────────► #16 TypeScript SDK v3.0.0
```

---

## Critical Files Reference

| File | Issue | Action |
|------|-------|--------|
| `aragora/debate/orchestrator.py` | 2,520 lines | Decomposition target |
| `aragora/debate/arena_config.py` | 50+ fields | Strategy pattern refactor |
| `aragora/server/handlers/base.py` | Base patterns | Extend for typed handlers |
| `aragora/server/middleware/tracing.py` | Foundation exists | Integrate with OTLP |
| `aragora/logging_config.py` | Mixed patterns | Complete contextvars migration |
