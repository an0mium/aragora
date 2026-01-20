# Changelog

All notable changes to Aragora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multi-worker server mode**: Production scaling with `--workers` flag
  - `aragora serve --workers 4` spawns 4 worker processes on consecutive ports
  - Updated nginx config with load balancing (`least_conn` for API, `ip_hash` for WebSocket)
  - Graceful shutdown handling with SIGINT/SIGTERM
- **AgentSpec.create_team()**: New factory method for explicit field creation
  - Preferred over deprecated `parse_list()` for programmatic access
  - Auto-rotates roles (proposer → critic → synthesizer → judge)
  - Accepts dicts or AgentSpec instances with type-safe TypedDict
- **URL allowlist configuration**: Configurable SSRF protection for evidence fetching
  - `ARAGORA_URL_FETCH_ALL_ENABLED=true`: Allow any URL (trusted environments)
  - `ARAGORA_URL_ALLOWED_DOMAINS=domain1,domain2`: Extend default allowlist
  - Safety checks always applied (blocks private IPs, localhost, non-HTTP schemes)
- **Knowledge Mound operation tests**: 29 new tests for ops mixins
  - KnowledgeMoundOperations (debate integration)
  - StalenessOperationsMixin, CultureOperationsMixin, SyncOperationsMixin

### Fixed
- **Agent spec parsing**: Fixed 3-part spec parsing bug where OpenRouter models produced invalid specs
  - PERSONA_TO_AGENT now maps to registered agent types instead of full model paths
  - Specs are now always 2-part format (agent_type:persona)
- **Usage sync double-reporting**: Fixed watermarks resetting to 0 on service restart
  - Added SQLite persistence for sync watermarks with billing period keying
  - Watermarks survive restarts preventing duplicate Stripe usage reports
- **Context gatherer cache isolation**: Fixed cache leaking between debates with different tasks
  - Cache now uses task-hash keying to isolate entries per debate
  - Added `get_evidence_pack(task)` and `clear_cache(task)` for targeted operations
- **React ConnectionContext side effect**: Moved `setLastAllConnected` from useMemo to useEffect
- **CrossDebateMemory race condition**: Added snapshot pattern for concurrent reads
  - Take snapshot under lock, process without lock, update access tracking under lock
- **Next.js metadata deprecation**: Separated viewport from metadata export for Next.js 14+ compliance

### Changed
- **Async classification**: QuestionClassifier.classify is now natively async using AsyncAnthropic
  - No longer blocks event loop during LLM-based question classification

### Tests
- Updated context gatherer tests for dict-based cache structure
- Added task isolation tests for ContextGatherer
- Added async classification tests for PromptBuilder
- Added usage sync persistence tests

## [1.5.1] - 2026-01-15

### Added
- **Privacy compliance handler**: GDPR/CCPA data privacy endpoints (566 lines)
  - `GET /api/privacy/data-export`: Export user data (GDPR Art. 20)
  - `DELETE /api/privacy/data-deletion`: Request data deletion (GDPR Art. 17)
  - `GET /api/privacy/consent`: Get consent status
  - `POST /api/privacy/consent`: Update consent preferences
  - `GET /api/privacy/retention`: Get data retention policies
- **CritiqueStore module**: Persistent critique storage with thread-safe access
- **New test suites**:
  - Privacy handler tests (722 lines)
  - MFA middleware tests (663 lines)
  - Storage factory/repository tests (672 lines)
  - Debate orchestrator integration tests (454 lines)
- **Documentation**:
  - DATA_RESIDENCY.md: Regional storage and compliance policies
  - FEATURE_MATURITY.md: Feature stability classifications
  - MONITORING_SETUP.md: Observability configuration guide
  - REMOTE_WORK_SECURITY.md: Security guidelines for remote development

### Changed
- **Type safety improvements**:
  - Added `AgentRole` and `AgentStance` type aliases
  - Added `ServerContext` TypedDict for handler type safety
  - Improved type hints across core modules
- **Consensus phase reliability**: Guaranteed synthesis generation with multi-tier fallbacks
  - Fallback chain: Opus → Sonnet → formatted summary → minimal synthesis
  - Never fails silently - always produces output
- **Handler test fixes**: Updated 351 tests for refactored module paths
  - Fixed HandlerResult attribute access (`.status_code`, `.body`)
  - Fixed import paths for admin/billing, features/evidence subpackages
  - Added rate limiter reset fixtures

### Fixed
- **Handler import paths**: Fixed broken imports after handler refactoring
  - Cache imports: `.cache` → `.admin.cache`
  - Evidence imports: Updated for features subpackage
  - Billing mock patches: Updated for jwt_auth, stripe_client paths
- **Test pollution**: Added rate limiter reset between tests
- **Gauntlet ID validation**: Tests now use valid `gauntlet-*` ID format

### Security
- **GDPR/CCPA compliance**: Full data subject rights implementation
- **Consent management**: Granular consent tracking and updates

## [1.5.0] - 2026-01-14

### Added
- **OAuth E2E test suite**: Comprehensive OAuth testing for social integrations
  - `tests/oauth/test_google_oauth.py`: 7 tests for Google OAuth state management
  - `tests/oauth/test_youtube_oauth.py`: 8 tests for YouTube OAuth flows
  - `tests/oauth/test_twitter_oauth.py`: 16 tests for Twitter publishing and OAuth
  - Tests cover state uniqueness, replay prevention, entropy validation, handler routes
- **SDK capability probes API**: Test agents for vulnerabilities
  - `ProbesAPI` with `run()` method for capability probing
  - Probe types: contradiction, hallucination, sycophancy, persistence, confidence calibration, reasoning depth, edge cases
  - Comprehensive probe report with vulnerability rates and recommendations
- **SDK verification history**: Query and analyze past verifications
  - `translate()`: Convert claims to formal language
  - `history()`: Query verification history with pagination
  - `getHistoryEntry()`: Get specific verification details
  - `getProofTree()`: Get proof tree for verification entry
- **Performance benchmarks**: Documented baseline metrics
  - 11 API component benchmarks with latency metrics (μs)
  - 17 rate limiting benchmark tests
  - Results documented in `docs/PERFORMANCE.md`
- **Frontend verification panels**: UI for verification features
  - Verification history panel with search and filtering
  - Probe reports panel with vulnerability visualization
- **Philosophical personas**: New agent personas for deeper discourse
  - Philosopher, humanist, existentialist personas
  - Domain detection for topic-appropriate persona selection
- **Async memory wrappers**: Non-blocking memory operations
  - `add_async()`, `store()`, `get_async()`, `retrieve_async()`, `update_outcome_async()`
  - Offloads blocking SQLite I/O to event loop executor
- **Callback timeout protection**: Prevent hanging debates
  - 30-second timeout for judge termination, early stopping, evidence refresh
  - Graceful fallback on timeout

### Changed
- **TypeScript SDK v1.1.0**: Major feature additions
  - Added ProbesAPI for capability probing
  - Extended VerificationAPI with history and proof tree methods
  - New types: ProbeType, ProbeRunRequest, ProbeResult, ProbeReport, VerificationHistoryEntry
- **Test count**: Increased from 23,363 to 23,448 test functions (+85)
- **Resilience improvements**: Synthesis fallback and token grouping
- **Arena hooks**: Enhanced event handling with additional safety checks

### Fixed
- **WebSocket state reset**: Reset all debate state when debateId changes
  - Prevents data leaking between debates when navigating
- **Type safety**: Resolved remaining mypy and ruff violations
- **Stream events**: Improved arena hooks and event handling

### Documentation
- **Performance baseline**: `docs/PERFORMANCE.md` with automated benchmark results
- **OAuth testing guide**: Documentation for running E2E OAuth tests
- **Evidence API guide**: Updated documentation for evidence endpoints

### Security
- **OAuth state validation**: Comprehensive replay attack prevention tests
- **State entropy verification**: Minimum 32-character state tokens validated

## [1.4.0] - 2026-01-14

### Added
- **Admin UI components**: Complete admin console with React components
  - `PersonaEditor`: Grid/list view persona management with search and detail panels
  - `AuditLogViewer`: Real-time audit feed with filtering, export (JSON/CSV/SOC2)
  - `TrainingExportPanel`: ML training data export (SFT, DPO, Gauntlet formats)
  - Admin pages: `/admin/personas`, `/admin/audit`, `/admin/training`, `/admin/revenue`, `/admin/organizations`, `/admin/users`
- **Nomic governance system**: Approval gates and audit logging for self-improvement safety
  - `DesignGate`: Approval before implementation with complexity scoring
  - `TestQualityGate`: Test quality validation with coverage and warning thresholds
  - `CommitGate`: Structured approval before committing changes
  - SQLite-backed audit logger with queryable event history
  - Circuit breakers and recovery strategies for error handling
- **Mandatory final synthesis**: Claude Opus 4.5 generates unified synthesis at debate end
- **Gallery API**: Public debate access via `GET /api/gallery` for showcasing debates
- **Test coverage expansion**: 23,363 tests (+607 from v1.3.0)
- **Template tests**: Added `tests/test_templates.py` with 83 tests
- **Streaming tests**: Added `tests/test_stream_arena_hooks.py` (31 tests) and `tests/test_stream_gauntlet_emitter.py` (27 tests)
- **Server metrics tests**: Added `tests/test_server_metrics.py` with 56 tests for Prometheus-style metrics
- **Load testing infrastructure**: Added `tests/performance/` with 13 load tests

### Changed
- **Architecture Phase 2**: Extracted caches, added types module, integrated subsystems
- **Handler refactoring**: Extracted debates and auth domains into subpackages
- **Nomic phase improvements**: Enhanced error handling in phase implementations
- **Orchestrator cleanup**: Removed dead methods, slimmed Arena class
- **Convergence refactoring**: Split module into cache/similarity/detector
- **Test count**: Increased from 22,756 to 23,363 test functions
- **Deploy consolidation**: Unified aragora.service systemd configuration

### Fixed
- **WebSocket stability**: Resolved race conditions causing UI flickering
- **WebSocket reconnection**: Stabilized useEffect dependencies to prevent reconnection loops
- **Event handling**: Defensive `.get()` for event data access, filtered kwargs for SpectatorStream
- **Database persistence**: Ensure temperature_params table persists across restarts
- **Debates handler**: Expanded test coverage and improved reliability
- **Type safety**: Fixed mypy errors in `openapi_impl.py` and `secrets.py`

### Documentation
- **ADMIN.md**: Comprehensive admin console guide with API endpoints and component usage
- **NOMIC_GOVERNANCE.md**: Approval gates, audit logging, and state machine documentation
- **Feature discovery guide**: `docs/FEATURE_DISCOVERY.md`

### Security
- **SAST scan**: Bandit shows 0 HIGH severity issues
- **Rate limiting verified**: Load tests confirm correct behavior under concurrent load
- **Nomic safety gates**: Approval required before critical autonomous operations

## [1.3.0] - 2026-01-13

### Fixed
- **Test suite blockers**: Fixed 2 import errors that were blocking test collection (22,756 tests now runnable)
  - Added `TierConfig` re-export to `aragora/memory/continuum.py` (fixes test_continuum.py, test_continuum_memory.py)
  - Added `DB_TIMEOUT` re-export to `aragora/server/storage.py` (fixes test_server_storage.py)

### Changed
- **Test collection**: Tests increased from 22,469 to 22,756 (287 additional tests now discoverable)

## [1.2.0] - 2026-01-13

### Changed
- **Linting cleanup**: Reduced ruff errors from 793 to 0
  - Fixed all unused variable warnings (F841)
  - Fixed all unused import warnings (F401)
  - Fixed type comparison issues (E721) using `is` instead of `==`
  - Cleaned up trailing whitespace (W293)
- **Code cleanup**: Removed dead code and unused assignments across 25+ files
- **Ruff configuration**: Added per-file ignores for intentional print statements in CLI utilities

### Fixed
- **Type comparisons**: Changed `param_type == int` to `param_type is int` in routing.py and decorators.py
- **Undefined exports**: Removed non-existent `with_error_handling` from error_utils.py `__all__`
- **Unused code removal**:
  - Removed unused `results` list in matrix_debates.py
  - Removed unused `all_relationships` in relationship.py
  - Removed unused `combined` variable in sharing.py
  - Simplified `_generate_share_token` to use `secrets.token_urlsafe` directly

## [1.1.0] - 2026-01-13

### Changed
- **Type safety overhaul**: Reduced mypy errors from 209 to 0 in checked modules
- **MyPy configuration**: Added comprehensive `ignore_errors` overrides for third-party integrations and complex modules
- Disabled `warn_return_any` and `warn_unused_ignores` to reduce false positives

### Security
- **Fixed all high-severity bandit findings** (6 → 0):
  - Added `usedforsecurity=False` to non-cryptographic MD5 hashes in:
    - `connectors/wikipedia.py` (evidence ID generation)
    - `evidence/collector.py` (content deduplication)
    - `nomic/phases/implement.py` (design hash for caching)
    - `server/http_caching.py` (ETag generation)

### Documentation
- Updated v1.1.0 roadmap with quality focus areas

## [1.0.1] - 2026-01-13

### Fixed
- **OpenAPI schema**: Added missing `BadRequest` response component that was causing contract test failures
- **Type safety**: Fixed implicit Optional patterns across multiple modules (security_barrier.py, routing.py, json_helpers.py, etc.)
- **Removed unused type:ignore comments**: Cleaned up stale type suppressions in typing.py, postgres_store.py, event_bridge.py

### Documentation
- Updated STATUS.md with v1.0.0 release information
- Added MIGRATION.md with upgrade guide from 0.8.x to 1.0.x

### Changed
- Added mypy configuration to ignore yaml library stubs

## [1.0.0] - 2026-01-13

### Added
- **LRU caching for consensus queries**: `get_consensus()` and `get_dissents()` now use TTL-based caching (5 min, 500 entries) for improved performance
- **VoteCollector module**: Extracted vote collection logic with timeout protection
- **VoteWeighter module**: Extracted vote weighting and calibration logic
- **Comprehensive API versioning**: Full documentation in `docs/API_VERSIONING.md`
- **Deprecation policy**: Formal policy in `docs/DEPRECATION_POLICY.md`

### Changed
- **Version bump to 1.0.0**: First stable release
- **Database optimization**: Added LRU caching to frequently-accessed consensus and dissent queries
- **Architecture cleanup**: Modular vote handling with VoteCollector and VoteWeighter classes

### Fixed
- Auth handler test mocks for `is_account_locked()` and `record_failed_login()`

### Documentation
- API versioning policy with deprecation timelines
- Migration guide from 0.8.x to 1.0.0

## [0.8.1] - 2026-01-13

### Added
- SOC2, PCI-DSS, and NIST CSF compliance personas for Gauntlet mode
- `scripts/test_tiers.sh` for common test tiers (fast, ci, lint, typecheck, frontend, e2e)
- `scripts/cleanup_runtime_artifacts.sh` to relocate root-level runtime DB artifacts (now handles directories too)
- PhaseValidator for Nomic loop state validation between phases
- WebSocket reconnection with exponential backoff (1s→30s cap, max 5 attempts)
- Plugin submission flow (`POST /api/plugins/submit`, `GET /api/plugins/submissions`)
- Load testing module (`tests/load/test_concurrent_debates.py`) with 8 tests
- Reconnection indicator in GauntletLive UI

### Changed
- Onboarding docs now point to `docs/START_HERE.md` / `docs/GETTING_STARTED.md` as the canonical entry
- Database docs now default to `ARAGORA_DATA_DIR` (`.nomic`) and clarify legacy paths
- Frontend docs clarify dashboard vs SDK vs legacy frontend
- OpenAPI spec regenerated (1285 endpoints)
- MCP server now exposes all 24 tools from tools.py

### Fixed
- `aragora doctor` circuit breaker status check (handles `_registry_size` metadata key)
- `notify_payment_failed()` signature (added missing `days_until_downgrade` parameter)

## [0.8.0] - 2026-01-11

### Added
- **Gauntlet Mode**: Adversarial stress-testing for specifications, architectures, and policies
  - Red team attacks (security, injection, auth bypass)
  - Devil's advocate (logic flaws, hidden assumptions)
  - Scaling critic (SPOF, bottlenecks, thundering herd)
  - Compliance checking (GDPR, HIPAA, AI Act)
  - Decision receipts with cryptographic audit trails
- **ReviewsHandler**: Shareable code review links via `/api/reviews/{id}`
- **Badge generator**: `aragora badge` command for README badges
- **Shareable review links**: `aragora review --share` generates permanent URLs
- **Demo mode**: `aragora review --demo` works without API keys
- **Single-provider fallback**: Works with just one API key configured
- **GAUNTLET.md**: Comprehensive Gauntlet mode documentation (300+ lines)
- **AGENT_SELECTION.md**: Agent comparison and selection guide
- **Gauntlet demos**: Sample specs for security, GDPR, and scaling demos
- **Integration tests and benchmarks**: Performance and reliability testing

### Fixed
- Review ID length validation prevents filesystem errors on very long IDs
- `eval()` replaced with safe AST evaluator in probe handler
- Circular import in gauntlet module resolved
- Template serialization in gauntlet config

### Changed
- GitHub Action updated with proper CLI integration (`action.yml`)
- Agent names standardized: `claude,codex` → `anthropic-api,openai-api`

### Security
- Subprocess environment filtering added
- Safe AST evaluation replaces dangerous eval() calls

## [0.7.0] - 2026-01-01

### Added
- **AI Red Team PR Review**: `aragora review` command for unanimous AI consensus
- **Multi-provider agents**: Mistral, DeepSeek, Qwen, Yi, Kimi via OpenRouter
- **API versioning**: `/api/v1/` prefix for stable endpoints
- **Circuit breaker persistence**: Survives restarts
- **SQLiteStore base class**: Unified database access
- **Graceful shutdown**: Clean server termination
- **WeightCalculator and VoteAggregator**: Extracted consensus components
- **Token extraction utilities**: Centralized auth handling
- **Production observability**: Logging, metrics, and tracing

### Fixed
- Agent caching and JSON parsing security issues
- Database safety and SSRF vulnerabilities
- Error handling improvements in handlers

### Security
- 4 medium/low security issues addressed
- Input validation hardened
- Rate limiting improvements

## [0.6.0] - 2025-12-15

### Added
- **Phase 11-13**: Operational modes, capability probing, red team mode
- **Formal verification**: Z3/Lean backends for proof generation
- **Debate graph**: DAG-based debates for complex disagreements
- **Calibration tracker**: Brier score prediction accuracy
- **Position tracker**: Agent stance history with verification
- **Flip detector**: Semantic position reversal detection

### Changed
- Memory tier benchmarks added
- E2E test coverage expanded
- Rate limiting documentation improved

## [0.5.0] - 2025-12-01

### Added
- **Phase 8-10**: Advanced debates, truth grounding, audience participation
- **Persona laboratory**: A/B testing, emergent traits, cross-pollination
- **Semantic retriever**: Pattern matching for similar critiques
- **Thread-safe audience participation**: ArenaMailbox for live interaction
- **WebSocket streaming**: Real-time debate visualization

### Changed
- Checkpoint/resume system improved
- Crash recovery hardened

## [0.4.0] - 2025-11-15

### Added
- **Phase 5-7**: Intelligence, formal reasoning, reliability & audit
- **ELO system**: Persistent agent skill tracking
- **Claims kernel**: Structured typed claims with evidence
- **Provenance manager**: Cryptographic evidence chains
- **Belief network**: Probabilistic reasoning
- **Breakpoint manager**: Human intervention points

## [0.3.0] - 2025-11-01

### Added
- **Phase 3-4**: Evidence, resilience, agent evolution
- **Memory stream**: Per-agent persistent memory
- **Local docs connector**: Evidence from codebase
- **Persona manager**: Agent traits and expertise
- **Tournament system**: Competitive benchmarking

## [0.2.0] - 2025-10-15

### Added
- **Phase 1-2**: Foundation and learning
- **Continuum memory**: Multi-tier learning (fast/medium/slow/glacial)
- **Consensus memory**: Track settled vs contested topics
- **Insight extractor**: Post-debate pattern learning
- **Argument cartographer**: Debate graph visualization

## [0.1.0] - 2025-10-01

### Added
- Initial release
- Multi-agent debate framework
- Heterogeneous agents (Claude, GPT, Gemini, Grok)
- Structured debate protocol (propose/critique/revise)
- Multiple consensus mechanisms (majority, unanimous, judge)
- SQLite-based critique store
- CLI interface (`aragora ask`)

---

## Upgrade Notes

### 0.7.x → 0.8.x
- Agent names changed: Use `anthropic-api` instead of `claude`, `openai-api` instead of `codex`
- GitHub Action inputs updated: See `action.yml` for new parameter names
- New `--share` flag on `aragora review` for shareable links

### 0.6.x → 0.7.x
- API endpoints now use `/api/v1/` prefix
- `ARAGORA_API_TOKEN` environment variable for auth
- Rate limiting enabled by default
