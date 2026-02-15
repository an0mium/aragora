# Planned Breaking Changes for v3.0

Comprehensive catalog of deprecations and planned removals for the Aragora v3.0 release. Each entry includes the migration path so you can update your code today.

For the general deprecation policy, see [reference/DEPRECATION_POLICY.md](reference/DEPRECATION_POLICY.md).
For v1-to-v2 migration details, see [guides/V1_TO_V2_MIGRATION.md](guides/V1_TO_V2_MIGRATION.md).

---

## Summary

| Category | Planned Removals | Severity |
|----------|-----------------|----------|
| [API v1 Sunset](#api-v1-sunset) | All `/api/v1/` endpoints | High |
| [Module Relocations](#module-relocations) | 6 backward-compat shims | Medium |
| [Arena Config Consolidation](#arena-config-consolidation) | ~20 individual kwargs | Medium |
| [Deprecated Functions](#deprecated-functions) | 5 function-level deprecations | Low |
| [Legacy Configuration](#legacy-configuration) | `aragora.config.legacy` module + DB path constants | Medium |
| [Agent Spec String Parsing](#agent-spec-string-parsing) | `AgentSpec.parse()`, `parse_list()`, colon format | Low |
| [RLM Backend Parameter](#rlm-backend-parameter) | `rlm_backend` kwarg on cognitive limiters | Low |
| [Type Aliases](#type-aliases) | `ValidationResult` alias | Low |
| [datetime.utcnow() Cleanup](#datetimeutcnow-cleanup) | 317 call sites across 101 modules | Internal |

---

## API v1 Sunset

**Category:** API
**Deprecated in:** v2.0.0 (Jan 2026)
**Sunset date:** June 1, 2026
**Removed in:** v3.0.0

All `/api/v1/` endpoints will be removed. The deprecation middleware already adds RFC 8594 `Deprecation` and `Sunset` headers to every v1 response.

| v1 Endpoint | v2 Replacement |
|-------------|---------------|
| `GET /api/v1/debates` | `GET /api/v2/debates` |
| `POST /api/v1/debate` | `POST /api/v2/debates` |
| `GET /api/v1/agents` | `GET /api/v2/agents` |
| `GET /api/v1/health` | `GET /api/v2/system/health` |
| `GET /api/v1/rankings` | `GET /api/v2/rankings` |
| `POST /api/v1/probes/run` | `POST /api/v1/probes/capability` (then v2) |
| All other `/api/v1/*` routes | Corresponding `/api/v2/*` routes |

**Post-sunset enforcement:** Set `ARAGORA_BLOCK_SUNSET_ENDPOINTS=true` to return HTTP 410 Gone for v1 requests before v3.0 ships.

**Migration:**

```python
# Before
response = client.post("/api/v1/debate", {"topic": "Design a cache", "max_rounds": 3})
debates = response["debates"]

# After
response = client.post("/api/v2/debates", {"task": "Design a cache", "rounds": 3})
debates = response["data"]["debates"]
```

**Source:** `aragora/server/versioning/constants.py` -- `V1_SUNSET_DATE = date(2026, 6, 1)`

---

## Module Relocations

**Category:** Internal
**Removed in:** v3.0.0

These modules are backward-compatibility shims that re-export from the canonical location. All emit `DeprecationWarning` on import today.

| Deprecated Import | Replacement Import | Deprecated Since |
|-------------------|--------------------|------------------|
| `aragora.modes.gauntlet` | `aragora.gauntlet` | v2.0.0 |
| `aragora.schedulers` | `aragora.scheduler` | v2.0.0 |
| `aragora.operations` | `aragora.ops` | v2.0.0 |
| `aragora.gateway.decision_router` | `aragora.core.decision_router` | v2.0.0 |
| `aragora.observability.logging` | `aragora.logging_config` | v2.0.0 |
| `aragora.connectors.email.gmail_sync` | `aragora.connectors.enterprise.communication.gmail.GmailConnector` | v2.0.0 |
| `aragora.server.handlers.knowledge` (monolithic) | `aragora.server.handlers.knowledge_base` (modular) | v2.0.0 |

**Migration:** Update your imports. Each shim's docstring shows the exact replacement. For example:

```python
# Before
from aragora.modes.gauntlet import GauntletOrchestrator

# After
from aragora.gauntlet import GauntletOrchestrator
```

```python
# Before
from aragora.schedulers import ReceiptRetentionScheduler

# After
from aragora.scheduler.receipt_retention import ReceiptRetentionScheduler
```

**Source files:**
- `aragora/modes/gauntlet.py`
- `aragora/schedulers/__init__.py`
- `aragora/operations/__init__.py`
- `aragora/gateway/decision_router.py`
- `aragora/observability/logging.py`
- `aragora/connectors/email/gmail_sync.py`
- `aragora/server/handlers/knowledge.py`

---

## Arena Config Consolidation

**Category:** Configuration
**Deprecated in:** v2.9.0
**Removed in:** v3.0.0

Individual keyword arguments on `Arena()` / `ArenaConfig` for supermemory, RLM, cross-debate memory, knowledge, evolution, and ML features are deprecated. Use the corresponding config dataclass objects instead.

| Deprecated kwargs | Config Object | Example Fields |
|-------------------|---------------|----------------|
| `enable_supermemory`, `supermemory_adapter`, `supermemory_inject_on_start`, `supermemory_max_context_items`, `supermemory_context_container_tag`, `supermemory_sync_on_conclusion`, `supermemory_min_confidence_for_sync`, `supermemory_outcome_container_tag`, `supermemory_enable_privacy_filter` | `SupermemoryConfig` | All supermemory_* fields |
| `use_rlm_limiter`, `rlm_limiter`, `rlm_compression_threshold`, `rlm_max_recent_messages`, `rlm_summary_level`, `rlm_compression_round_threshold` | `MemoryConfig` | All rlm_* fields |
| `cross_debate_memory`, `enable_cross_debate_memory` | `MemoryConfig` | Cross-debate fields |
| Knowledge bridge params | `KnowledgeConfig` | Knowledge Mound settings |
| Evolution/prompt params | `EvolutionConfig` | Prompt evolution settings |
| ML delegation params | `MLConfig` | ML delegation, quality gates, consensus estimation |

**Migration:**

```python
# Before (deprecated)
arena = Arena(
    env, agents, protocol,
    enable_supermemory=True,
    supermemory_max_context_items=50,
    use_rlm_limiter=True,
    rlm_compression_threshold=3000,
)

# After
from aragora.debate.orchestrator_config import MemoryConfig, SupermemoryConfig

arena = Arena(
    env, agents, protocol,
    supermemory_config=SupermemoryConfig(
        enable_supermemory=True,
        max_context_items=50,
    ),
    memory_config=MemoryConfig(
        use_rlm_limiter=True,
        rlm_compression_threshold=3000,
    ),
)
```

**Source:** `aragora/debate/orchestrator_config.py` lines 572-649

---

## Deprecated Functions

**Category:** Internal
**Removed in:** v3.0.0

| Function | Replacement | Location |
|----------|-------------|----------|
| `DebateFactory._get_persona_prompt()` | `aragora.agents.personas.get_persona_prompt()` | `aragora/server/debate_factory.py` |
| `DebateFactory._apply_persona_params()` | `aragora.agents.personas.apply_persona_to_agent()` | `aragora/server/debate_factory.py` |
| `debate_utils.wrap_agent_for_streaming()` | `aragora.server.stream.arena_hooks.wrap_agent_for_streaming()` | `aragora/server/debate_utils.py` |
| `stream.state_manager.get_state_manager()` | `get_stream_state_manager()` or `aragora.server.state.get_state_manager()` | `aragora/server/stream/state_manager.py` |

**Migration:** Replace each call with the replacement shown above. All deprecated functions currently delegate to their replacement internally.

---

## Legacy Configuration

**Category:** Configuration
**Deprecated in:** v2.9.0 (Jan 2026)
**Removed in:** v3.0.0

The entire `aragora.config.legacy` module will be removed. This includes all `DB_*_PATH` constants (`DB_ELO_PATH`, `DB_MEMORY_PATH`, `DB_INSIGHTS_PATH`, `DB_CONSENSUS_PATH`, `DB_CALIBRATION_PATH`, etc.) and legacy helper functions.

| Deprecated | Replacement |
|-----------|-------------|
| `from aragora.config.legacy import DB_ELO_PATH` | `from aragora.persistence.db_config import get_db_path; get_db_path("elo")` |
| `from aragora.config.legacy import DB_MEMORY_PATH` | `get_db_path("memory")` |
| `from aragora.config.legacy import get_api_key` | `from aragora.config.settings import get_settings; get_settings().api_key` |
| `from aragora.config.legacy import DB_TIMEOUT_SECONDS` | `from aragora.config import DB_TIMEOUT_SECONDS` (still available via `__init__`) |
| `from aragora.config.legacy import validate_configuration` | `from aragora.config.settings import get_settings` |
| All `DB_*_PATH` constants | `aragora.persistence.db_config.get_db_path(name)` |
| Concurrency constants | `aragora.config.settings.get_settings().concurrency` |

**Note:** `aragora.config.__init__` lazy-loads legacy constants via `__getattr__` for convenience. Those aliases that come from `config.legacy` will also stop working in v3.0.

**Source:** `aragora/config/legacy.py` (explicit v3.0.0 removal stated in module docstring)

---

## Agent Spec String Parsing

**Category:** Configuration
**Removed in:** v3.0.0

String-based agent specification parsing (`AgentSpec.parse()` and `AgentSpec.parse_list()`) and the legacy colon format (`provider:persona`) are deprecated in favor of explicit field construction.

| Deprecated | Replacement |
|-----------|-------------|
| `AgentSpec.parse("anthropic:philosopher")` | `AgentSpec(provider="anthropic", persona="philosopher")` |
| `AgentSpec.parse_list("anthropic,openai")` | `AgentSpec.create_team([{"provider": "anthropic"}, {"provider": "openai"}])` |
| Colon format strings in agent lists | Dict-based specs or `AgentSpec()` constructors |

**Migration:**

```python
# Before (deprecated)
specs = AgentSpec.parse_list("anthropic:philosopher,openai:critic")

# After
specs = AgentSpec.create_team([
    {"provider": "anthropic", "persona": "philosopher"},
    {"provider": "openai", "role": "critic"},
])
```

**Source:** `aragora/agents/spec.py` lines 322-328, 410-416

---

## RLM Backend Parameter

**Category:** Configuration
**Deprecated in:** v2.9.0
**Removed in:** v3.0.0

The `rlm_backend` parameter on `RLMCognitiveLoadLimiter`, `RLMCognitiveLoadLimiter.for_stress_level()`, and the top-level `create_rlm_limiter()` factory is deprecated. The backend is now determined automatically by the `rlm_model` parameter.

| Deprecated | Replacement |
|-----------|-------------|
| `RLMCognitiveLoadLimiter(rlm_backend="openai", rlm_model="gpt-4o")` | `RLMCognitiveLoadLimiter(rlm_model="gpt-4o")` |
| `create_rlm_limiter(rlm_backend="anthropic")` | `create_rlm_limiter(rlm_model="claude")` |

**Source:** `aragora/debate/cognitive_limiter_rlm.py` lines 152-174, 318-336

---

## Type Aliases

**Category:** Internal
**Removed in:** v3.0.0

| Deprecated Alias | Canonical Name | Location |
|------------------|---------------|----------|
| `ValidationResult` | `FeatureValidationResult` | `aragora/debate/feature_validator.py` |
| `RankingAdapter` | `PerformanceAdapter` | `aragora/debate/subsystem_coordinator.py` |

**Migration:** Find-and-replace the old name with the new one.

---

## datetime.utcnow() Cleanup

**Category:** Internal
**Target:** v3.0.0

`datetime.utcnow()` is deprecated in Python 3.12+ (PEP 657). There are currently 317 call sites across 101 source files in `aragora/` and 17 more in `scripts/`.

| Deprecated | Replacement |
|-----------|-------------|
| `datetime.utcnow()` | `datetime.now(timezone.utc)` |
| `datetime.utcnow().isoformat()` | `datetime.now(timezone.utc).isoformat()` |
| `field(default_factory=datetime.utcnow)` | `field(default_factory=lambda: datetime.now(timezone.utc))` |

This does not affect public API behavior but will eliminate runtime `DeprecationWarning` noise on Python 3.12+.

---

## Legacy Storage Path (.gt)

**Category:** Internal
**Removed in:** v3.0.0

The legacy `.gt` bead/convoy storage directory is supported via a fallback in the path resolver. In v3.0, only `.aragora_beads` will be recognized.

| Deprecated | Replacement |
|-----------|-------------|
| `<workspace>/.gt/` | `<workspace>/.aragora_beads/` |
| `NOMIC_CANONICAL_STORE_PERSIST` env var | `ARAGORA_CANONICAL_STORE_PERSIST` |

**Migration:** Rename your `.gt` directory to `.aragora_beads`, or set `ARAGORA_STORE_DIR` to point to your preferred location.

**Source:** `aragora/nomic/stores/paths.py` lines 8, 54, 79

---

## Environment Variable Renames

**Category:** Configuration
**Removed in:** v3.0.0

| Deprecated Variable | Replacement |
|---------------------|-------------|
| `ARAGORA_REQUIRE_DISTRIBUTED_STATE` | `ARAGORA_REQUIRE_DISTRIBUTED` |
| `NOMIC_CANONICAL_STORE_PERSIST` | `ARAGORA_CANONICAL_STORE_PERSIST` |
| `ARAGORA_BEAD_DIR` | `ARAGORA_STORE_DIR` (both still work, `STORE_DIR` preferred) |

---

## How to Prepare

1. **Enable deprecation warnings** in your test suite:
   ```bash
   python -W default::DeprecationWarning -m pytest tests/
   ```

2. **Check v1 API usage** with the built-in metrics:
   ```bash
   curl http://localhost:8080/metrics | grep aragora_v1_api
   ```

3. **Set enforcement early** to catch remaining v1 calls:
   ```bash
   export ARAGORA_LOG_DEPRECATED_USAGE=true
   ```

4. **Run the deprecation audit script:**
   ```bash
   python scripts/add_v1_deprecation.py --report
   ```

---

## Timeline

| Date | Event |
|------|-------|
| Jan 2026 | v2.0.0 released; v1 API deprecated, module shims created |
| Jan 2026 | v2.9.0: Arena config consolidation warnings, legacy config warnings |
| Jun 1, 2026 | v1 API sunset date (HTTP 410 enforced if `BLOCK_SUNSET_ENDPOINTS=true`) |
| Q3 2026 | v3.0.0 release: all items above removed |

---

*Last updated: 2026-02-15*
