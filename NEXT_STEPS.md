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
**Status:** Completed (2026-02-03)
**Impact:** Clean working tree

All uncommitted changes have been committed and pushed, including:
- Connector improvements, handler restructuring, KM adapters
- TestFixer module with 231 tests
- Secrets Manager integration across codebase
- Security debate API and audit scheduling

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
**Status:** Core implemented (TestFixer module)
**Impact:** Autonomous test maintenance

TestFixer module now provides:
- Automated test failure detection and categorization (14 failure categories)
- Hegelian debate-based fix proposal with cross-critique
- Retry loop orchestrator with stuck detection and revert-on-failure
- JSONL persistence for learning from fix attempts
- CLI entry point: `aragora testfix`

Remaining work:
- Integration with Nomic Loop phases for continuous self-healing
- Flaky test detection heuristics
- Learning from historical fix patterns

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

Current: **93%** (up from 92%)

### Remaining Gaps:
| Gap | Impact | Effort |
|-----|--------|--------|
| Test pollution | CI confidence | Medium |
| Deployment runbooks | Operations | Low |
| SDK examples | Adoption | Low |
| Rate limit docs | Production | Low |

### Recent Completions:
- TestFixer module with 231 tests (automated test self-healing)
- Secrets Manager integration (AWS preferred over .env)
- Security Debate API and audit scheduling
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
