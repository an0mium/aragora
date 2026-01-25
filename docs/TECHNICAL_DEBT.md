# Technical Debt Tracking

Last Updated: 2026-01-25

## Large File Refactoring Queue

### HIGH Priority (6,527 lines combined)

#### 1. `aragora/observability/metrics.py` (3,536 lines)
**Issue:** 130 repetitive recording functions, 115+ global variables, bloated `_init_metrics()` function.

**Recommendation:**
- Split into feature-area modules: `metrics/request.py`, `metrics/debate.py`, `metrics/knowledge.py`
- Use factory pattern for metric definitions
- Keep only facade/public API in `__init__.py`

**Estimated Effort:** 4-6 hours

#### 2. `aragora/observability/metrics/core.py` (2,991 lines)
**Issue:** 121 near-identical recording functions, duplicate patterns with metrics.py.

**Recommendation:**
- Consolidate with metrics.py or clearly define responsibility boundary
- Extract recording function templates into a shared factory
- Consider code generation for repetitive patterns

**Estimated Effort:** 3-4 hours

### MEDIUM Priority (5,121 lines combined)

#### 3. `aragora/events/cross_subscribers.py` (2,572 lines)
**Issue:** Single monolithic `CrossSubscriberManager` class handling multiple event types.

**Recommendation:**
- Extract subscriber types: `MemorySubscriber`, `ELOSubscriber`, `KnowledgeSubscriber`
- Use composition: manager delegates to specialized handlers
- Each handler manages own circuit breaker and metrics

**Estimated Effort:** 3-4 hours

#### 4. `aragora/server/handlers/admin/health.py` (2,549 lines)
**Issue:** `HealthHandler` class with 21 methods covering diverse health checks.

**Recommendation:**
- Split into focused handlers: `LivenessHandler`, `ReadinessHandler`, `StorageHealthHandler`
- Use composition for unified health endpoint
- Improves testing granularity

**Estimated Effort:** 2-3 hours

### Keep As-Is (Justified Design)

| File | LOC | Reason |
|------|-----|--------|
| `control_plane/policy.py` | 2,565 | Well-organized domain model, 12 classes with clear purpose |
| `storage/user_store.py` | 2,439 | Intentional facade pattern over 7 repositories |
| `debate/orchestrator.py` | 2,180 | Core Arena engine, already partially extracted to phases/ |
| `server/startup.py` | 2,397 | Server initialization complexity justified |
| `server/handlers/social/slack.py` | 2,384 | Slack API handlers, complex by nature |
| `connectors/chat/slack.py` | 2,309 | Slack connector, complex by nature |
| `connectors/enterprise/communication/gmail.py` | 2,264 | Gmail API integration, justified |
| `services/threat_intelligence.py` | 2,164 | Security service, justified |
| `server/openapi/schemas.py` | 2,045 | API schemas, appropriate grouping |
| `cli/main.py` | 2,013 | CLI entrypoint, appropriate grouping |

## Type Ignore Reduction

Current: 1,628 type: ignore comments (down from 1,643)

**Reduction Strategy:**
1. Focus on `arg-type` ignores (most common, 400+)
2. Add proper type annotations to agent classes
3. Use TypedDict for dictionary parameters
4. Add generics to DataLoader and similar classes

## Exception Handling

BLE001 (blind exception) rule enabled with per-file ignores.
Fixed: 5 high-risk handlers in Phase 3.

**Remaining cleanup:**
- Audit remaining per-file BLE001 ignores
- Add specific exception types where appropriate
- Ensure all exceptions are logged before re-raise

## Test Coverage Gaps

**Comprehensive:** Payment webhooks, chat E2E, CDC-KM integration

**Potential gaps to investigate:**
- MySQL/SQL Server CDC (PostgreSQL and MongoDB covered)
- CDC event concurrency/race conditions
- Full CDC → Debate context injection (mocked only)

## Deprecated Module Status

The following deprecated modules are **properly maintained** as backwards-compatibility shims:

| Module | Replacement | Status |
|--------|-------------|--------|
| `aragora.modes.gauntlet` | `aragora.gauntlet` | Shim active, emits DeprecationWarning |
| `aragora.crawlers` | `aragora.connectors.repository_crawler` | Shim active, emits DeprecationWarning |
| `aragora.connectors.email.gmail_sync` | New Gmail implementation | Shim active |
| `sdk/python/aragora` | `aragora-client` (PyPI) | README deprecation notice added |

**Action:** No removal needed until major version bump. Monitor usage via DeprecationWarning logs.
