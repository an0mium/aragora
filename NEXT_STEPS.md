# Next Steps for Aragora

Prioritized work items for continued development. Updated: 2026-02-03.

---

## Immediate Priorities

### 1. Test Isolation & Pollution
**Status:** Strategic improvement needed
**Impact:** 100+ tests fail together but pass individually

The test suite has test pollution issues where shared state bleeds between tests:
- Async event loops not properly reset
- Module-level singletons (connectors, handlers)
- Mock state not isolated

**Recommendations:**
- Add `pytest-randomly` to CI to detect pollution early
- Use `@pytest.mark.forked` for problematic test modules
- Create teardown fixtures for global state reset

### 2. Connector Exception Handling Cleanup
**Status:** 39 locations need attention
**Impact:** Improved debuggability and test reliability

Multiple connector files have bare `except Exception: pass` clauses that swallow errors:
- `aragora/connectors/enterprise/base.py:548`
- `aragora/connectors/enterprise/database/postgres.py:201`
- `aragora/connectors/sec.py:204,249,299`
- And 30+ more locations

**Fix:** Change to `except Exception as e: logger.debug("...", e)`

### 3. Commit Remaining Uncommitted Changes
**Status:** ~60 files with uncommitted changes
**Impact:** Clean working tree

Review and commit the following categories:
- Connector improvements (11 files)
- Handler restructuring (15+ files)
- New KM adapters (compliance, debate, workflow)
- New personas package
- Pipeline decision_plan package

---

## Medium-Term Goals

### 4. Knowledge Mound Phase A3
**Status:** Phase A2 complete (950+ tests)
**Impact:** Enhanced evidence extraction and cross-debate learning

Phase A3 features:
- Automated evidence extraction from debate outcomes
- Cross-debate knowledge synthesis
- Improved contradiction detection

### 5. Multi-Instance Deployment Documentation
**Status:** Code complete, docs needed
**Impact:** Production readiness

Document Redis requirements for:
- Rate limiting in multi-instance mode
- Session storage
- Cache coordination
- Circuit breaker state sharing

### 6. SDK Parity
**Status:** 140 TypeScript namespaces, Python SDK needs review
**Impact:** Developer experience

Ensure Python and TypeScript SDKs have equivalent coverage for:
- All 461 API endpoints
- WebSocket streaming
- Authentication methods

---

## Strategic Improvements

### 7. Nomic Loop v2: Self-Healing Tests
**Status:** Conceptual
**Impact:** Autonomous test maintenance

Extend Nomic Loop to:
- Detect flaky tests automatically
- Propose fixes via debate
- Apply fixes with verification

### 8. Security Debate API Enhancement
**Status:** API documented, needs hardening
**Impact:** Compliance and audit capabilities

Enhance multi-agent security debates:
- Integration with CI/CD pipelines
- Automated vulnerability triage
- Compliance report generation

### 9. n8n/Linear/Obsidian Integration Expansion
**Status:** Core complete
**Impact:** Competitive positioning

Expand automation capabilities:
- Additional n8n workflow templates
- Linear issue sync bidirectional
- Obsidian plugin development

---

## Commercial Readiness

Current: **92%** (up from 90%)

### Remaining Gaps:
| Gap | Impact | Effort |
|-----|--------|--------|
| Test pollution | CI confidence | Medium |
| Deployment runbooks | Operations | Low |
| SDK examples | Adoption | Low |
| Rate limit docs | Production | Low |

### Recent Completions:
- n8n/Linear/Obsidian integration
- MCP server implementation
- Security hardening (SSRF, MFA governance)
- Handler restructuring (costs, analytics, orchestration, payments)

---

## File Reference

Key files for each priority:

1. **Test Pollution:** `tests/conftest.py`, `pyproject.toml` (pytest config)
2. **Connector Cleanup:** `aragora/connectors/enterprise/base.py`
3. **KM Phase A3:** `aragora/knowledge/mound/adapters/`
4. **Deployment Docs:** `docs/DEPLOYMENT.md` (create)
5. **SDK Parity:** `sdk/python/`, `sdk/typescript/`
