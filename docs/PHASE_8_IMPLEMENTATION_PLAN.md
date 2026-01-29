# Phase 8: Production Hardening & Quality Implementation Plan

**Generated:** 2026-01-29
**Status:** Ready for implementation
**Based on:** Comprehensive 5-dimensional codebase assessment

---

## Executive Summary

After completing Phase 7E (98.8% test pass rate), a ground-up reassessment identified the following priority areas:

| Dimension | Score | Key Finding |
|-----------|-------|-------------|
| **Security** | 95% | Excellent - production ready |
| **Production Readiness** | 78% | Good - 2 critical blockers |
| **Test Coverage** | 85% | Good - critical module gaps |
| **Code Quality** | 70% | Moderate - tech debt accumulation |
| **Documentation** | 88% | Strong - minor gaps |

**Critical Path to Production:** Fix 2 blockers (migration locking, DB connectivity test), add tests for untested critical handlers, then proceed with code quality improvements.

---

## Phase 8A: Production Blockers (CRITICAL - 1 day)

### 8A-1: Database Migration Version Locking

**Problem:** No version locking table - race condition risk with multiple pods running migrations simultaneously.

**File:** `aragora/migrations/runner.py`

**Implementation:**
```python
# Add advisory lock acquisition before running migrations
async def acquire_migration_lock(conn) -> bool:
    """Acquire exclusive lock for migrations."""
    result = await conn.fetchval(
        "SELECT pg_try_advisory_lock(hashtext('aragora_migration'))"
    )
    return result

async def release_migration_lock(conn) -> None:
    """Release migration lock."""
    await conn.execute(
        "SELECT pg_advisory_unlock(hashtext('aragora_migration'))"
    )

# Add migration_versions table
CREATE TABLE IF NOT EXISTS migration_versions (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    applied_by VARCHAR(100)
);
```

**Test:** `tests/migrations/test_concurrent_migrations.py`

### 8A-2: Database Connectivity Validation at Startup

**Problem:** Config validator checks format but not connectivity - will fail at runtime if DB unreachable.

**File:** `aragora/server/config_validator.py` (line ~251-301)

**Implementation:**
```python
async def _validate_database_connectivity(self) -> ValidationResult:
    """Test actual database connection, not just URL format."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return ValidationResult(False, "DATABASE_URL not set")

    try:
        conn = await asyncpg.connect(db_url, timeout=5.0)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return ValidationResult(True, "Database connectivity verified")
    except asyncpg.PostgresError as e:
        return ValidationResult(False, f"Database connection failed: {e}")
    except asyncio.TimeoutError:
        return ValidationResult(False, "Database connection timeout")
```

**Test:** `tests/server/test_config_validator_connectivity.py`

---

## Phase 8B: Critical Test Coverage (HIGH - 2-3 days)

### 8B-1: Knowledge Base Handler Tests

**Gap:** 4 handlers with 0 tests - critical to KnowledgeMound system

**Files to test:**
- `aragora/server/handlers/knowledge_base/handler.py` (8.5KB - main entry)
- `aragora/server/handlers/knowledge_base/query.py`
- `aragora/server/handlers/knowledge_base/search.py`
- `aragora/server/handlers/knowledge_base/facts.py`

**Test files to create:**
- `tests/server/handlers/knowledge_base/test_handler.py`
- `tests/server/handlers/knowledge_base/test_query.py`
- `tests/server/handlers/knowledge_base/test_search.py`
- `tests/server/handlers/knowledge_base/test_facts.py`

**Coverage targets:**
- Route handling (can_handle, handle methods)
- Input validation (query params, body)
- Permission checks (@require_permission)
- Error handling (404, 400, 500 responses)

### 8B-2: Metrics Handler Tests

**Gap:** 17KB handler with 0 tests - production telemetry at risk

**Files to test:**
- `aragora/server/handlers/metrics/handler.py` (17KB)
- `aragora/server/handlers/metrics/formatters.py`
- `aragora/server/handlers/metrics/tracking.py`

**Test files to create:**
- `tests/server/handlers/metrics/test_handler.py`
- `tests/server/handlers/metrics/test_formatters.py`
- `tests/server/handlers/metrics/test_tracking.py`

### 8B-3: Fix Remaining Skipped Tests

**Files with skipped tests to resolve:**
- `tests/server/handlers/features/test_folder_upload.py:151` - path.count bug
- `tests/server/handlers/features/test_speech.py:320` - pathlib.Path mock
- `tests/server/handlers/features/test_scheduler.py:275,295` - enum mocking

---

## Phase 8C: Code Quality Quick Wins (MEDIUM-HIGH - 2-3 days)

### 8C-1: Automated Type Modernization

**Gap:** 1,562 files with deprecated typing patterns

**Script:** `scripts/modernize_types.py`
```python
#!/usr/bin/env python
"""Modernize type hints to Python 3.10+ syntax."""
import re
import subprocess
from pathlib import Path

def modernize_types():
    # Use ruff for safe, fast automated fixes
    subprocess.run([
        "ruff", "check", "aragora/",
        "--select", "UP006,UP007,UP035",  # typing modernization rules
        "--fix"
    ])

if __name__ == "__main__":
    modernize_types()
```

**Patterns to fix:**
| Before | After | Files |
|--------|-------|-------|
| `Optional[T]` | `T \| None` | 1,562 |
| `Dict[K, V]` | `dict[K, V]` | 827 |
| `List[T]` | `list[T]` | 668 |
| `Tuple[T, ...]` | `tuple[T, ...]` | 105 |
| `Set[T]` | `set[T]` | 106 |

### 8C-2: Exception Handling Audit

**Gap:** 132 broad `except Exception:` clauses masking bugs

**Priority files (security-critical):**
1. `aragora/services/threat_intelligence.py` (3+ instances)
2. `aragora/agents/fallback.py` (3 instances)
3. `aragora/storage/schema.py` (3 instances)
4. `aragora/persistence/migrations/runner.py` (3 instances)

**Pattern to apply:**
```python
# Before
try:
    result = await self._fetch_url(url)
except Exception as e:
    logger.error(f"Failed: {e}")
    return None

# After
try:
    result = await self._fetch_url(url)
except aiohttp.ClientTimeout:
    logger.warning(f"Timeout fetching {url}")
    return None
except aiohttp.ClientError as e:
    logger.error(f"HTTP error fetching {url}: {e}")
    raise FetchError(f"Failed to fetch {url}") from e
```

**Script:** `scripts/audit_exception_handling.py`

---

## Phase 8D: Monolith Refactoring (MEDIUM - 1-2 weeks)

### Priority Files (>2,000 LOC)

| Rank | File | LOC | Split Into |
|------|------|-----|-----------|
| 1 | `services/threat_intelligence.py` | 2,164 | `threat_intel/config.py`, `clients.py`, `cache.py`, `events.py` |
| 2 | `server/handlers/_oauth_impl.py` | 2,119 | `oauth/google.py`, `github.py`, `microsoft.py`, `apple.py`, `oidc.py` |
| 3 | `server/debate_origin.py` | 2,055 | `debate_origin/routing.py`, `state.py`, `delivery.py` |
| 4 | `cli/main.py` | 2,027 | `cli/commands/ask.py`, `stats.py`, `agents.py`, `nomic.py` |
| 5 | `services/spam_classifier.py` | 2,021 | `spam/model.py`, `features.py`, `classify.py` |
| 6 | `connectors/chat/teams.py` | 2,000 | `teams/messages.py`, `events.py`, `send.py` |

### Refactoring Pattern
1. Extract public interface into `__init__.py` re-exports
2. Move implementation into domain-specific submodules
3. Add deprecation aliases for backwards compatibility
4. Update imports incrementally
5. Run full test suite after each file

---

## Phase 8E: CI/CD Security Enhancements (LOW - 1 day)

### 8E-1: Image Vulnerability Scanning

**File:** `.github/workflows/build.yml`

**Add:**
```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'ghcr.io/${{ github.repository }}:${{ github.sha }}'
    format: 'table'
    exit-code: '1'
    ignore-unfixed: true
    vuln-type: 'os,library'
    severity: 'CRITICAL,HIGH'
```

### 8E-2: Post-Deploy Smoke Tests

**File:** `.github/workflows/deploy.yml`

**Add:**
```yaml
- name: Smoke test deployment
  run: |
    # Wait for deployment
    sleep 30
    # Test health endpoint
    curl -f https://${{ env.DEPLOY_URL }}/healthz || exit 1
    # Test readiness
    curl -f https://${{ env.DEPLOY_URL }}/readyz || exit 1
    # Test basic API
    curl -f https://${{ env.DEPLOY_URL }}/api/v1/status || exit 1
```

---

## Execution Timeline

```
Week 1:
├── Day 1: Phase 8A - Production Blockers (migration locking, DB connectivity)
├── Day 2-3: Phase 8B-1 - Knowledge Base handler tests
├── Day 4-5: Phase 8B-2 - Metrics handler tests

Week 2:
├── Day 1-2: Phase 8C-1 - Automated type modernization
├── Day 2-3: Phase 8C-2 - Exception handling audit
├── Day 4-5: Phase 8B-3 - Fix skipped tests

Week 3-4:
├── Phase 8D - Monolith refactoring (incremental)
├── Phase 8E - CI/CD security enhancements
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test pass rate | 98.8% | 99.5% |
| Handler test coverage | 85% | 95% |
| Files >1,500 LOC | 45 | <20 |
| `except Exception:` | 132 | <30 |
| Deprecated typing | 1,562 | 0 |
| Production blockers | 2 | 0 |

---

## Critical Files Summary

### Phase 8A (Production Blockers)
- `aragora/migrations/runner.py` - Add version locking
- `aragora/server/config_validator.py` - Add connectivity test

### Phase 8B (Test Coverage)
- `tests/server/handlers/knowledge_base/*.py` - Create 4 test files
- `tests/server/handlers/metrics/*.py` - Create 3 test files

### Phase 8C (Code Quality)
- `scripts/modernize_types.py` - Create automation script
- `scripts/audit_exception_handling.py` - Create audit script
- `aragora/services/threat_intelligence.py` - Fix exception handling
- `aragora/agents/fallback.py` - Fix exception handling

### Phase 8D (Refactoring)
- 6 monolithic files to split into ~18 modules

### Phase 8E (CI/CD)
- `.github/workflows/build.yml` - Add vulnerability scanning
- `.github/workflows/deploy.yml` - Add smoke tests

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|-----------|
| 8A | Migration lock could deadlock | Add timeout, logging |
| 8B | New tests might fail initially | Follow existing patterns |
| 8C | Type changes could break runtime | Run full test suite |
| 8D | Refactoring could break imports | Deprecation aliases |
| 8E | CI changes could block deploys | Test in staging first |

---

## Approval Checklist

- [ ] Phase 8A: Production blockers fixed and tested
- [ ] Phase 8B: Critical handlers have >80% test coverage
- [ ] Phase 8C: No deprecated typing patterns remain
- [ ] Phase 8C: Exception handling audit complete
- [ ] Phase 8D: All files <1,500 LOC
- [ ] Phase 8E: CI/CD security enhancements deployed
- [ ] Full test suite passes (>99% pass rate)
- [ ] Production deployment validated
