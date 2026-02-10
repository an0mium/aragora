# Migration Guide

This guide covers migrating to newer Aragora versions.

---

## Deprecated Modules

The following modules are deprecated and will be removed in v2.3.0:

### `aragora.crawlers` → `aragora.connectors`

The `aragora.crawlers` module is deprecated. Use the connectors module instead.

**Why:** The connectors module provides better AST-based symbol extraction, concurrent processing, provenance tracking, and reliability scoring.

**Migration:**

```python
# Old (deprecated)
from aragora.crawlers import RepositoryCrawler

# New (recommended)
from aragora.connectors.repository_crawler import RepositoryCrawler
```

| Old Module | New Module |
|------------|------------|
| `aragora.crawlers.RepositoryCrawler` | `aragora.connectors.repository_crawler.RepositoryCrawler` |
| `aragora.crawlers.BaseCrawler` | `aragora.connectors.base.BaseCrawler` |
| Local docs crawling | `aragora.connectors.local_docs` |
| Web content | `aragora.connectors.web` |

**Removal:** v2.3.0 (estimated Q2 2026)

### `aragora.modes.gauntlet` → `aragora.gauntlet`

The `aragora.modes.gauntlet` module is deprecated. Use the gauntlet package instead.

**Why:** The new package provides a more feature-complete API with templates, receipt generation, and persona-based regulatory testing.

**Migration:**

```python
# Old (deprecated)
from aragora.modes.gauntlet import GauntletOrchestrator

# New (recommended)
from aragora.gauntlet import GauntletOrchestrator, GauntletRunner
```

| Feature | Old API | New API |
|---------|---------|---------|
| Simple 3-phase execution | Custom implementation | `GauntletRunner` |
| Full 5-phase execution | `GauntletOrchestrator` | `GauntletOrchestrator` |
| Templates | Not available | `aragora.gauntlet.templates` |
| Receipts | Not available | `aragora.gauntlet.receipts` |

See `docs/GAUNTLET_ARCHITECTURE.md` for detailed migration guide.

**Removal:** v2.3.0 (estimated Q2 2026)

---

## API v1 → v2

For comprehensive API migration from v1 to v2, see [**MIGRATION_V1_TO_V2.md**](./MIGRATION_V1_TO_V2.md).

**Key Changes:**
- Response format: Data wrapped in `{"data": ..., "meta": ...}`
- Endpoint names: Pluralized and RESTful
- New features: Calibration, probes, privacy compliance

**Timeline:** API v1 sunset is **June 1, 2026**.

---

## 0.8.x → 1.0.0

### Breaking Changes

None. v1.0.0 is API-compatible with 0.8.x.

### New Features

#### LRU Caching for Consensus Queries

The `ConsensusMemory` class now uses LRU caching for `get_consensus()` and `get_dissents()` methods:
- **TTL**: 5 minutes
- **Max entries**: 500
- **Auto-invalidation**: Cache is cleared on writes

This improves performance for repeated consensus queries without code changes.

#### VoteCollector Module

Vote collection logic has been extracted to `aragora/debate/phases/vote_collector.py`:
- `VoteCollectorConfig` - Configuration dataclass
- `VoteCollectorCallbacks` - Callback hooks for vote events
- `VoteCollectorDeps` - Dependency injection container
- `VoteCollector` - Main vote collection class with timeout protection

#### VoteWeighter Module

Vote weighting logic has been extracted to `aragora/debate/phases/vote_weighter.py`:
- `VoteWeighterDeps` - Dependency injection for weighting
- `VoteWeighter` - Vote weighting and calibration logic

### API Documentation

Full API versioning policy is now documented:
- `docs/API_VERSIONING.md` - Versioning strategy and conventions
- `docs/DEPRECATION_POLICY.md` - Deprecation timelines and procedures

### Upgrade Steps

1. Update your dependency:
   ```bash
   pip install aragora==1.0.0
   # or
   pip install --upgrade aragora
   ```

2. No code changes required for existing integrations.

3. (Optional) Review new documentation:
   - `docs/API_VERSIONING.md` for API stability guarantees
   - `docs/DEPRECATION_POLICY.md` for deprecation procedures

### Environment Variables

No changes to environment variables between 0.8.x and 1.0.0.

### Database Migrations

No database schema changes in 1.0.0. Existing databases are compatible.

---

## 0.7.x → 0.8.x

### Breaking Changes

1. **Agent Names Standardized**
   - `claude` → `anthropic-api`
   - `codex` → `openai-api`

2. **GitHub Action Inputs Updated**
   - See `action.yml` for new parameter names

### New Features

- Gauntlet mode for adversarial stress-testing
- Shareable review links via `--share` flag
- Demo mode via `--demo` flag

### Upgrade Steps

1. Update agent name references in your code
2. Update GitHub Action workflow files if using Aragora Action

---

## 0.6.x → 0.7.x

### Breaking Changes

1. **API Prefix Added**
   - All endpoints now use `/api/v1/` prefix
   - Old endpoints still work but are deprecated

2. **Authentication**
   - New `ARAGORA_API_TOKEN` environment variable for auth
   - Rate limiting enabled by default

### Upgrade Steps

1. Update API endpoint URLs to use `/api/v1/` prefix
2. Set `ARAGORA_API_TOKEN` if using authentication
3. Review rate limiting configuration

---

## Getting Help

- GitHub Issues: https://github.com/an0mium/aragora/issues
- Documentation: https://aragora.ai/docs
