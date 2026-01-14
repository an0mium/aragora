# Aragora Project Status

*Last updated: January 13, 2026 (23:30 UTC)*

## Current Release

### v1.3.0 - GA Preparation Release (January 2026)

**Production Ready** - Aragora 1.3.0 represents comprehensive security hardening, type safety, and production readiness improvements through Phases 11-15.

#### Key Highlights
- **22,924+ tests** collected and passing
- **116+ documentation files** with comprehensive coverage
- **41 modular HTTP handlers** for clean API architecture
- **67 fully integrated features** including memory systems, ELO rankings, and formal verification
- **0 HIGH severity security issues** (Bandit scan clean)
- **0 mypy errors** in 12 core modules (expanded from 9)
- **0 ruff lint violations** (all auto-fixed)
- **Gitleaks secret scanning** in CI pipeline
- **OpenAPI spec** committed to docs/api/
- **9.0/10 production readiness score**

#### What's New in 1.3.0 (Phases 11-15 Complete)

**Phase 15 (Latest)**
- **Code Quality**: Fixed all ruff lint violations (0 errors)
- **Secret Scanning**: Added gitleaks job to CI for commit-level secret detection
- **API Documentation**: Generated and committed OpenAPI spec (JSON + YAML) to docs/api/
- **Frontend Tests**: Added component tests for TrainingExportPanel and PublicGallery
- **CI Enhancement**: docs-sync job validates OpenAPI spec stays in sync with code

**Phase 14**
- **Type Safety Expansion**: typecheck-core expanded to 12 modules (consensus.py, resilience.py, belief.py)
- **E2E Test Suite**: 17 Playwright E2E test files covering critical user flows
- **Accessibility Testing**: axe-core integration for WCAG compliance
- **CI Enhancement**: E2E workflow with backend/frontend integration

**Phase 13**
- **Token Revocation UI**: "Logout All Devices" button in Settings panel
- **Type Safety Expansion**: typecheck-core expanded to 9 modules (orchestrator.py, continuum.py)
- **Storage Layer Fixes**: Fixed mypy errors in factory.py and webhook_store.py
- **Training Export Page**: New /training route for ML training data export
- **Production Checklist**: Comprehensive 300+ line PRODUCTION_CHECKLIST.md
- **Public Gallery Page**: /gallery route for browsing debate history

**Phase 12**
- **TrainingExportPanel**: 470-line component for SFT/DPO/Gauntlet exports
- **PublicGallery**: 743-line component for debate browsing with search/filters
- **CI/CD Hardening**: ESLint now blocking (was informational)
- **Bug Fixes**: Fixed showBoot variable declaration order in page.tsx

**Phase 11**
- **Security Hardening**: Full SQL injection audit of high-risk files
  - Audited and documented 16 B608 warnings across storage layer
  - Added `# nosec B608` comments with validation explanations
  - 0 HIGH severity Bandit issues across entire codebase
- **Type Safety**: Fixed secrets.py mypy error (json.loads return type)
- **AWS Integration**: Secrets Manager support for production deployments
- **Genesis/Gauntlet UI**: New frontend components for evolution and stress-testing
- **CI/CD Hardening**: typecheck-core job for strict type checking on core modules

#### What's New in 1.0.1
- Type safety: Fixed all mypy errors (error_monitoring.py Sentry callback types)
- Extended DebateStorageProtocol with 8 new methods
- Fixed AgentRating attribute names (elo, debates_count)
- Fixed EloSystem.get_elo_history method name
- Nomic loop phased workflow documentation
- .env.example with nomic loop settings

#### What's Included (from 1.0.0)
- Multi-agent debate orchestration with consensus detection
- Memory systems: ContinuumMemory, ConsensusMemory with LRU caching
- ELO rankings and tournament system
- Agent fallback via OpenRouter on quota errors
- CircuitBreaker for agent failure handling
- WebSocket event streaming
- Formal verification with Z3 backend
- Gauntlet mode for adversarial stress-testing

## Previous State

### Stabilization Target (0.8.1) - COMPLETE

All stabilization items addressed:

- [x] Canonical onboarding path (START_HERE -> GETTING_STARTED) and doc consistency
- [x] Test tiers + CI alignment for fast/local vs full runs
- [x] Runtime data hygiene (prefer `.nomic` via `ARAGORA_DATA_DIR`, cleanup script)
- [x] Prometheus metrics and Grafana dashboards (deploy/grafana/)
- [x] OpenAPI spec regenerated (1285 endpoints)
- [x] PhaseValidator integrated into Nomic loop
- [x] WebSocket reconnection with exponential backoff
- [x] Plugin submission flow added
- [x] aragora doctor bug fixed (circuit breaker metadata handling)

### Test Status
- **Total Tests**: 12,349 collected (massive expansion via parametrized tests)
- **Frontend Tests**: 34 Jest tests (DebateListPanel, AgentComparePanel)
- **Recent Fixes (2026-01-05)**:
  - Fixed `_get_belief_classes()` → `_get_belief_analyzer()` typo in orchestrator.py
  - Fixed all 7 unanimous consensus tests (were failing due to above typo)
  - ELO tests (3), calibration test (1), replay tests (4) fixed in previous session
- **Code Fixes**:
  - Calibration bucket boundary now includes confidence=1.0
  - Belief analyzer function name corrected in orchestrator.py:2213

### Nomic Loop
- **Cycle**: 1 (running with 6-cycle budget)
- **Phase**: context gathering → debate
- **Last Consensus**: "Stream Gateway & Multiplexer" (80% consensus) - identified verify_system_health.py issue
- **Recent Fixes (2026-01-08 Session 19)**:
  - Fixed TypedDict access in context phase (`result["key"]` not `result.key`)
  - Fixed debate phase empty agents (`_create_debate_phase()` now calls `_select_debate_team()`)
  - Fixed `verify_system_health.py` to check refactored `stream/` package
  - All critical systems now healthy (timeout handling, CLI integration, loop_id routing)
- **Blocking Issues FIXED**:
  - Missing `agent_type` attribute in GeminiAgent (now added to all API agents)
  - RelationshipTracker.get_influence_network() parameter mismatch (fixed)
  - OpenRouterAgent broken super().__init__() call (fixed)
  - `_get_belief_classes()` undefined (fixed - was typo for `_get_belief_analyzer()`)
  - Design phase 0% consensus now detected and cycle skipped (new safeguard)
  - Context phase crash with TypedDict access (fixed 2026-01-08)
  - Debate phase 0/0 agents (fixed 2026-01-08)
- **Position Ledger**: Implemented in `aragora/agents/grounded.py`
- **NomicIntegration**: Fully wired up (probing, belief analysis, checkpointing, staleness)

### Active Agents (default config, 8 total)
| Agent | Model | API |
|-------|-------|-----|
| `grok` | grok-3 | xAI |
| `anthropic-api` | claude-opus-4-5-20251101 | Anthropic |
| `openai-api` | gpt-5.2 | OpenAI |
| `deepseek` | deepseek/deepseek-chat-v3-0324 | OpenRouter |
| `mistral-api` | mistral-large-2512 | Mistral |
| `gemini` | gemini-3-pro-preview | Google |
| `qwen-max` | qwen/qwen-max | OpenRouter |
| `kimi` | moonshot-v1-8k | Moonshot |

### Recent Changes (2026-01-10)
- **Mistral API Integration**:
  - Added `MistralAPIAgent` for direct Mistral API access
  - Added `CodestralAgent` for code-specialized tasks
  - Uses `MISTRAL_API_KEY` environment variable
  - Registered in AgentRegistry with `mistral-api` and `codestral` names
  - Added to `DEFAULT_AGENTS` and `STREAMING_CAPABLE_AGENTS`
- **Technical Debt Reduction**:
  - Extracted `TeamSelector` from orchestrator (-53 LOC)
  - Replaced 7 bare exception catches with specific types (13→6 remaining)
  - All remaining bare catches are correct patterns (transaction rollback)
- **UI Enhancements**:
  - Wired `TricksterAlertPanel` to dashboard
  - Created `RhetoricalObserverPanel` (165 LOC) for debate pattern analysis
  - Added debate mode selector (Standard/Graph/Matrix) to DebateInput
- **Documentation Update**:
  - Updated README.md with Mistral and multi-provider support
  - Updated AGENTS.md with 20+ agent types
  - Updated ENVIRONMENT.md with MISTRAL_API_KEY
  - Comprehensive cross-document consistency check

### Recent Changes (2026-01-09 Night)
- **Nomic Loop Extraction (Waves 1-4 Complete)**:
  - Wave 1: DeadlockManager, ContextFormatter, BackupManager (~990 LOC)
  - Wave 2: DeepAuditRunner, DisagreementHandler, GraphDebateRunner, ForkingRunner (~550 LOC)
  - Wave 3: ArenaFactory (~291 LOC), PostDebateProcessor (~454 LOC)
  - Wave 4: DebatePhase and DesignPhase validated as complete
  - Total: ~2,300 LOC extracted from nomic_loop.py into modular components
- **Runtime Artifact Cleanup**:
  - Removed ~548 files from git tracking (.nomic/backups, .nomic/replays, etc.)
  - Updated .gitignore to comprehensively exclude .nomic state files
  - 262k+ lines of runtime data removed from repository
- **Dependency Alignment**:
  - Standardized installs on `pyproject.toml` + `uv.lock` (pip install .)
  - Removed legacy requirements.txt usage from deploy/docs paths

### Recent Changes (2026-01-13)
- **0.8.1 Stabilization Verification**:
  - Verified EvidenceHandler integration (8 endpoints, STABLE status)
  - Verified timeout middleware exports
  - Added 35 EvidenceHandler tests (34 passing, 1 skipped pending handler fix)
  - Confirmed legacy /api/auth/revoke already migrated to AuthHandler
- **Test Coverage**:
  - New test file: `tests/test_handlers_evidence.py` (35 tests)
  - Covers: list, get, search, collect, associate, statistics, delete endpoints
  - Proper rate limiter isolation via fixture

### Recent Changes (2026-01-09 Evening)
- **Feature Integration Sprint**:
  - Wired PerformanceMonitor to Arena and AutonomicExecutor (tracking for generate/critique/vote)
  - Wired CalibrationTracker via `enable_calibration` protocol flag
  - Added AirlockProxy option via `use_airlock` ArenaConfig flag
  - Wired AgentTelemetry to AutonomicExecutor with `enable_telemetry` flag
  - Wired RhetoricalObserver to DebateRoundsPhase with `enable_rhetorical_observer` flag
  - Added `enable_trickster` and `trickster_sensitivity` protocol flags
  - Added Genesis evolution wiring (population_manager, auto_evolve, breeding_threshold)
- **New Protocol Flags**:
  - `enable_calibration: bool` - Record prediction accuracy for calibration curves
  - `enable_rhetorical_observer: bool` - Passive commentary on debate dynamics
  - `enable_trickster: bool` - Hollow consensus detection
  - `trickster_sensitivity: float` - Threshold for trickster challenges (default 0.7)
- **New ArenaConfig Options**:
  - `performance_monitor` / `enable_performance_monitor` - Agent call telemetry
  - `enable_telemetry` - Prometheus/Blackbox emission
  - `use_airlock` / `airlock_config` - Timeout protection
  - `population_manager` / `auto_evolve` / `breeding_threshold` - Genesis evolution
- **New API Endpoints**:
  - `POST /api/debates/graph` - Run graph-structured debates with branching
  - `GET /api/debates/graph/{id}` - Get graph debate by ID
  - `POST /api/debates/matrix` - Run parallel scenario debates
  - `GET /api/debates/matrix/{id}` - Get matrix debate results
- **New Event Type**:
  - `RHETORICAL_OBSERVATION` - Rhetorical pattern detected in debate rounds

### Recent Changes (2026-01-09 Morning)
- **Demo Consensus Fixtures**:
  - Created `aragora/fixtures/` package with demo consensus data
  - Added `load_demo_consensus()` and `ensure_demo_data()` functions
  - Created `demo_consensus.json` with 5 sample debate topics (architecture domain)
  - Added auto-seed on server startup for empty databases
  - Fixed ConsensusStrength enum mapping (`MEDIUM` → `MODERATE`)
  - Added `[tool.setuptools.package-data]` to include JSON fixtures in deployment
- **New API Endpoint**:
  - `GET /api/consensus/seed-demo` - Manually trigger demo data seeding
- **Nomic Loop Fixes**:
  - Fixed empty agent list crash in design phase (`_select_debate_team()` fallback)
  - Added fallback to default_team when AgentSelector returns empty Team
- **Search Functionality**:
  - Search now works independently of nomic loop (uses HTTP REST, not WebSocket)
  - Demo data ensures search has content even with no live debates

### Recent Changes (2026-01-08 Session 19)
- **Nomic Loop Fixes**:
  - Fixed TypedDict access patterns in context phase (use `result["key"]` not `result.key`)
  - Fixed debate phase agent creation (`_create_debate_phase()` now calls `_select_debate_team()`)
  - Added `topic_hint` parameter to `_create_debate_phase()` for better agent selection
  - Debate phase now correctly shows "Agent weights: 4/4 reliable" instead of "0/0"
- **System Health Verification**:
  - Updated `verify_system_health.py` for refactored `stream/` package
  - Fixed misleading logic (presence of timeout handling is good, not bad)
  - System now correctly reports "All critical systems healthy"
- **Debug Logging**:
  - Added debug logging for dropped critiques in `relationships.py:221-229`
  - Logs reason: missing critic, missing target, or self-critique
- **Documentation Hygiene**:
  - Removed deprecated `docs/API.md` (replaced by `API_REFERENCE.md`)
  - Updated `CLAUDE.md` with stream package refactoring, debate phases, agent modules
  - Updated architecture diagram with `server/stream/` package structure

### Recent Changes (2026-01-08 Earlier)
- **Nomic Loop Phase Extraction**: All 6 phases now have modular implementations
  - `ContextPhase` - Multi-agent codebase exploration
  - `DebatePhase` - Improvement proposal with PostDebateHooks
  - `DesignPhase` - Architecture planning with BeliefContext
  - `ImplementPhase` - Hybrid multi-model code generation
  - `VerifyPhase` - Tests and quality checks
  - `CommitPhase` - Git commit with safety checks
  - Opt-in via `USE_EXTRACTED_PHASES=1` environment variable
  - 29 new tests for phase factories and result types
- **Test Isolation**: Added global `clear_handler_cache` fixture to conftest.py
- **Flaky Test Fixes**: Fixed pulse handler tests that were making real API calls
- **Test Count**: Expanded to 12,001 tests (from 3,400+)

### Recent Changes (2026-01-07)
- **Database Consolidation**: Implemented full migration system for 22→4 databases
  - Created unified schemas in `aragora/persistence/schemas/` (core.sql, analytics.sql, memory.sql, agents.sql)
  - Added `db_config.py` for centralized database path management
  - Migration script with dry-run, rollback, and verification: `scripts/migrate_databases.py`
- **Type Annotations**: Added mypy configuration and Protocol definitions
  - `aragora/protocols.py` with 8 Protocol definitions (StorageBackend, MemoryBackend, EloBackend, etc.)
  - Enhanced mypy config in pyproject.toml with per-module strict settings
- **Performance**: Added LRU caching to ELO system
  - `aragora/utils/cache.py` with TTLCache, lru_cache_with_ttl decorator
  - Cached leaderboards and ratings with automatic invalidation
- **Memory Backend**: Extracted TierManager for configurable memory tiers
  - `aragora/memory/tier_manager.py` with tier configuration and transition metrics
  - ContinuumMemory now uses TierManager for promotion/demotion decisions
- **Stream Architecture**: Extracted ServerBase for common server functionality
  - `aragora/server/stream/server_base.py` with rate limiting, state caching, client management

### Recent Changes (2026-01-06)
- Extracted `SecurityBarrier` and `TelemetryVerifier` to `debate/security_barrier.py` (213 lines)
- Extracted `MemoryManager` to `debate/memory_manager.py`
- Extracted `PromptBuilder` to `debate/prompt_builder.py`
- Fixed Z3 test availability handling (proper pytest.skip when Z3 unavailable)
- Reduced orchestrator.py from 3,758 to 3,545 LOC
- Added comprehensive documentation for Phases 16-19 in FEATURES.md
- Updated ENVIRONMENT.md with telemetry and belief network config options
- Fixed CLAUDE.md CircuitBreaker location (was orchestrator.py, now resilience.py)
- Nomic loop state reset for fresh cycle after root cause analysis

### Recent Changes (2026-01-05)
- Fixed `_get_belief_classes()` → `_get_belief_analyzer()` typo in orchestrator.py
- Fixed all 7 unanimous consensus tests
- Added test baseline capture to nomic loop
- Improved fix targeting prompts (include failing file paths)

### Recent Changes (2026-01-04)
- Added OpenRouter support (DeepSeek, Llama, Mistral)
- Added GrokAgent for xAI API
- Updated default agents to streaming-capable models
- Fixed security: removed API keys from .env.example
- Fixed security: restricted exec() builtins in proofs.py
- Exported KiloCodeAgent for codebase exploration
- Improved debate scrolling (calc(100vh-280px))
- **NEW**: Activated Position Ledger by default in server startup
- **NEW**: Added IP-based rate limiting (DoS protection without auth)
- **NEW**: Initialized Debate Embeddings by default for historical memory
- **NEW**: Added TournamentPanel UI component to dashboard
- **NEW**: Added agent routing hints to DebateInput (domain detection + recommendations)
- **NEW**: Added `/api/tournaments` endpoint to list tournaments
- **NEW**: Added CruxPanel for Belief Network visualization
- **NEW**: Added MemoryInspector for Continuum Memory browsing
- **NEW**: Fixed asyncio.gather timeout in debate orchestrator (prevents 50% timeout failures)
- **NEW**: Fixed BeliefPropagationAnalyzer import in server
- **NEW**: Added LaboratoryPanel for emergent traits and cross-pollinations
- **NEW**: Implemented LLM-based Lean theorem translation (formal.py)
- **NEW**: Implemented LLM-based prompt refinement (evolver.py)
- **NEW**: Fixed security: unvalidated float parameter in /api/critiques/patterns
- **NEW**: Fixed security: path traversal in raw document upload
- **NEW**: Improved error handling in DocumentStore.list_all()
- **NEW**: Added `agent_type` attribute to all API agents (fixes nomic loop blocking issue)
- **NEW**: Fixed OpenRouterAgent broken super().__init__() call
- **NEW**: Fixed RelationshipTracker.get_influence_network() parameter issue in nomic_loop.py
- **NEW**: Added ReasoningDepthProbe and EdgeCaseProbe to prober.py
- **NEW**: Added compute_relationship_metrics() to EloSystem (rivalry/alliance scores)
- **NEW**: Added get_rivals() and get_allies() methods to EloSystem
- **NEW**: Created comprehensive ELO ranking tests (tests/test_elo.py)
- **NEW**: Fixed GitHub connector timeouts (180s → 30s/60s)
- **NEW**: Added __init__.py for evidence, pulse, uncertainty modules
- **NEW**: Implemented MomentDetector for significant debate events (Emergent Persona Lab v2)
- **NEW**: Exported 24+ new classes from main __init__.py (grounded personas, evidence, pulse, uncertainty)
- **NEW**: Created .env.example template for secure API key management
- **NEW**: Exported BeliefNetwork, ReliabilityScorer, WebhookDispatcher in main __init__.py
- **NEW**: Added record_redteam_result() to EloSystem (Red Team → ELO feedback loop)
- **NEW**: Added pattern injection to debate prompts (learned patterns from InsightStore)
- **NEW**: Added 5 UI components: DebateExportModal, OperationalModesPanel, AgentNetworkPanel, RedTeamAnalysisPanel, CapabilityProbePanel
- **NEW**: Fixed security: sanitized error messages in unified_server.py (prevents info disclosure)
- **NEW**: Fixed security: added slug length validation in storage.py (DoS prevention)
- **NEW**: Fixed security: added GitHub connector input validation (repo format + query length)
- **NEW**: Added belief analysis → design phase context (contested/crux claims guide design)
- **NEW**: Fixed stale claims injection to queue ALL stale claims (not just high-severity)
- **NEW**: Fixed security: checkpoint ID validation for git branch names (command injection prevention)
- **NEW**: Fixed security: path segment validation in api.py (400 error on invalid IDs)
- **NEW**: Connected ContinuumMemory.update_outcome() after debates (surprise-based learning)
- **NEW**: Added CritiqueStore.fail_pattern() tracking for failed debates (balanced learning)
- **NEW**: Added belief network persistence across nomic cycles (cross-cycle learning)
- **NEW**: Fixed Python 3.10 compatibility (asyncio.timeout → asyncio.wait_for)
- **NEW**: Fixed aiohttp WebSocket security (origin validation, payload validation, rate limiting)
- **NEW**: Fixed CORS header fallback behavior (don't send Allow-Origin for unauthorized origins)
- **NEW**: Added agent introspection API endpoints (/api/introspection/*)
- **NEW**: Added formal verification integration for decidable claims (Z3 backend)
- **NEW**: Added plugins API endpoints (/api/plugins, /api/plugins/{name}, /api/plugins/{name}/run)
- **NEW**: Added genesis API endpoints (/api/genesis/stats, /api/genesis/events, /api/genesis/lineage/*, /api/genesis/tree/*)
- **NEW**: Connected Z3 SMT solver to post-debate claim verification (auto-verifies decidable claims)
- **NEW**: Added Replay Theater visualization (ReplayGenerator, self-contained HTML replays)
- **NEW**: Wired SpectatorStream events to WebSocket broadcast (real-time UI updates)
- **NEW**: Added optional WebSocket authentication (check_auth integration)
- **NEW**: Added AUDIENCE_DRAIN event type for audience event processing
- **NEW**: Fixed security: path traversal protection in code.py (CodeReader)
- **NEW**: Added deadline enforcement to nomic loop verify-fix cycle (prevents infinite hangs)
- **NEW**: Exported 50+ new classes to main __init__.py (modes, spectate, pipeline, visualization, replay, introspection)
- **NEW**: Connected belief network cruxes to fix guidance prompts (targeted fixing)
- **NEW**: Added ELO confidence weighting from probe results (low-confidence debates = reduced ELO impact)
- **NEW**: Added TournamentManager class for reading tournament SQLite databases
- **NEW**: Wired /api/tournaments with real tournament data from nomic_dir
- **NEW**: Added /api/tournaments/{tournament_id} endpoint for tournament details
- **NEW**: Added TOKEN_START, TOKEN_DELTA, TOKEN_END event mapping to SpectatorStream bridge
- **NEW**: Added /api/agent/{name}/consistency endpoint for FlipDetector scores
- **NEW**: Added /api/agent/{name}/network endpoint for relationship data (rivals, allies)
- **NEW**: Emit MATCH_RECORDED WebSocket event after ELO match recording in orchestrator
- **NEW**: Fixed security: path traversal protection in custom.py (CustomModeLoader)
- **NEW**: Fixed security: environment variable bracket access in formal.py and evolver.py
- **NEW**: Updated CLI serve command to use unified server (aragora serve)
- **NEW**: Added crux cache clearing at cycle start (prevents context bleeding between cycles)
- **NEW**: Added real-time flip_detected WebSocket listener in InsightsPanel (auto-switches to flips tab)
- **NEW**: Added missing event types to events.ts (audience_drain, match_recorded, flip_detected, memory_recall, token_*)
- **NEW**: Added loop_id to FLIP_DETECTED events for multi-loop isolation
- **NEW**: Added original_confidence, new_confidence, domain fields to flip event data
- **NEW**: Fixed security: exec() timeout protection in proofs.py (5s limit prevents CPU exhaustion)
- **NEW**: Fixed security: sanitized error messages in api.py (replay endpoint)
- **NEW**: Fixed AgentRelationship export in __init__.py (was exporting wrong name)
- **NEW**: Added 4 unused UI components to dashboard (AgentNetworkPanel, CapabilityProbePanel, OperationalModesPanel, RedTeamAnalysisPanel)
- **NEW**: Added circuit breaker persistence across nomic cycles (saves/restores cooldowns)
- **NEW**: Added circuit breaker filtering to agent selection (skips agents in cooldown)
- **NEW**: Fixed security: sanitized error messages in token streaming (unified_server.py)
- **NEW**: Added mood/sentiment event types (MOOD_DETECTED, MOOD_SHIFT, DEBATE_ENERGY)
- **NEW**: Added critique and consensus event handling to DebateViewer
- **NEW**: Added ContraryViewsPanel for displaying dissenting opinions
- **NEW**: Added RiskWarningsPanel for domain-specific risk assessment
- **NEW**: Fixed CSP security: removed unsafe-eval from script-src (blocks eval/new Function)
- **NEW**: Added 8 database indexes to elo.py (elo_history, matches, domain_calibration, relationships)
- **NEW**: Exported 4 new modules: audience, plugins, nomic, learning (25+ new public APIs)
- **NEW**: Fixed N+1 query pattern in get_rivals/get_allies (single DB query instead of N+1)
- **NEW**: Wired AUDIENCE_SUMMARY and INSIGHT_EXTRACTED events to WebSocket stream

### Recent Changes (2026-01-05 Session 5)
- **NEW**: Modular handlers expanded to 41 total (agents, analytics, audio, auditing, auth, belief, billing, breakpoints, broadcast, cache, calibration, consensus, critique, dashboard, debates, documents, evolution, gallery, genesis, graph_debates, insights, introspection, laboratory, leaderboard, learning, matrix_debates, memory, metrics, moments, persona, plugins, probes, pulse, relationship, replays, routing, social, system, tournaments, verification)
- **NEW**: Migrated opponent briefing API to AgentsHandler (removed from unified_server.py legacy routes)
- **NEW**: Fork debate initial messages support (Arena accepts initial_messages parameter)
- **NEW**: Fork tests added (TestForkInitialMessages - 6 tests)
- **NEW**: User event queue overflow tests (TestUserEventQueue - 7 tests)
- **NEW**: Webhook logger fix for atexit shutdown (catches ValueError when logger closed)
- **NEW**: Resource availability logging at startup (_log_resource_availability)
- **NEW**: Updated .gitignore for nomic loop state files (.nomic/checkpoints/, .nomic/replays/, *.db files)
- **NEW**: Comprehensive tests for checkpoint, evolution, laboratory, belief modules
- **TOTAL TESTS**: 3,400+ collected (expanded via parametrized tests)

### Recent Changes (2026-01-05 Session 4)
- **NEW**: Modular HTTP handlers framework (base.py, debates.py, agents.py, system.py)
- **NEW**: Wired modular handlers into unified_server.py for gradual migration
- **NEW**: Added DebateListPanel component (debate history browser with filters)
- **NEW**: Added AgentComparePanel component (side-by-side agent comparison)
- **NEW**: Added 34 Jest tests for new frontend components
- **NEW**: Added 35 WebSocket tests (StreamEvent, SyncEventEmitter, TokenBucket)
- **NEW**: Added PhaseRecovery class to nomic_loop.py (structured error handling)
- **NEW**: Added Docker support for frontend (Dockerfile + next.config.js update)
- **NEW**: Added "agents" parameter to query whitelist for /api/agent/compare
- **NEW**: Nomic loop running SwarmAgent design proposal (audience participation)
- **TOTAL TESTS**: 789+ passed (754 Python + 35 WebSocket)

### Recent Changes (2026-01-05 Session 3)
- **NEW**: Fixed security: SAFE_ID_PATTERN bug (was string, needed re.match)
- **NEW**: Fixed security: Added symlink protection in _serve_file() (prevents directory escape)
- **NEW**: Added tests/test_security.py with token validation and SQL injection tests
- **NEW**: Added try/except error handling around all nomic loop phase calls
- **NEW**: Added phase crash recovery (context, debate, design, implement, verify phases)
- **NEW**: Fixed MemoryInspector endpoint (added /api/memory/tier-stats alias)
- **NEW**: Added design fallback mechanism (uses highest-voted design if no consensus)
- **NEW**: Added design arbitration (judge picks between competing designs on close votes)
- **NEW**: Fixed TypeScript errors in page.tsx (AgentNetworkPanel, RedTeamAnalysisPanel)
- **NEW**: Added missing timedelta import to nomic_loop.py
- **NEW**: Updated test count to 402 passed

### Recent Changes (2026-01-05 Session 2)
- **NEW**: Added design consensus safeguard (skips implementation if design has 0% consensus)
- **NEW**: Fixed security: X-Forwarded-For header only trusted from TRUSTED_PROXIES
- **NEW**: Added AnalyticsPanel for disagreements, early-stops, and role rotation visualization
- **NEW**: Added CalibrationPanel for agent confidence accuracy curves
- **NEW**: Added ConsensusKnowledgeBase for browsing settled topics and searching similar debates
- **NEW**: Killed stalled nomic loop (was stuck 10+ hours on empty plan) and reset state for cycle 2
- **NEW**: Updated Fully Integrated feature count to 54

## Feature Integration Status

### Fully Integrated (63)
| Feature | Status | Location |
|---------|--------|----------|
| Multi-Agent Debate | Active | `aragora/debate/orchestrator.py` |
| Token Streaming | Active | `aragora/agents/api_agents.py` |
| ELO Rankings | Active | `aragora/ranking/elo.py` |
| FlipDetector + Vote Weight | Active | `aragora/insights/flip_detector.py` (→ orchestrator.py:1423-1433) |
| Position Ledger | Active | `aragora/agents/grounded.py` |
| Calibration Tracking | Active | `aragora/agents/calibration.py` |
| Convergence Detection + Early Exit | Active | `aragora/debate/convergence.py` + `orchestrator.py:1635` |
| Role Rotation | Active | `aragora/debate/roles.py` |
| PersonaSynthesizer | Active | `aragora/agents/grounded.py` |
| MomentDetector | Active | `aragora/agents/grounded.py` |
| Relationship Metrics | Active | `aragora/ranking/elo.py` |
| Red Team → ELO | Active | `aragora/ranking/elo.py:record_redteam_result()` |
| Pattern Injection | Active | `aragora/debate/orchestrator.py:_format_patterns_for_prompt()` |
| Belief → Design Context | Active | `scripts/nomic_loop.py:phase_design()` (contested/crux claims) |
| Stale Claims Feedback | Active | `scripts/nomic_loop.py:run_cycle()` → `phase_debate()` |
| ContinuumMemory Outcomes | Active | `aragora/debate/orchestrator.py:_update_continuum_memory_outcomes()` |
| Failed Pattern Tracking | Active | `aragora/debate/orchestrator.py` (calls CritiqueStore.fail_pattern) |
| Cross-Cycle Beliefs | Active | `scripts/nomic_loop.py:phase_debate()` (loads prev cycle beliefs) |
| Belief Network | Exported | `aragora/reasoning/belief.py` (exported in __init__.py) |
| Reliability Scoring | Exported | `aragora/reasoning/reliability.py` (exported in __init__.py) |
| Webhook Integration | Exported | `aragora/integrations/webhooks.py` (exported in __init__.py) |
| Z3 Formal Verification | Active | `aragora/debate/orchestrator.py:_verify_claims_formally()` |
| Introspection API | Active | `aragora/server/unified_server.py` (/api/introspection/*) |
| Plugins API | Active | `aragora/server/unified_server.py` (/api/plugins/*) |
| Genesis API | Active | `aragora/server/unified_server.py` (/api/genesis/*) |
| Deadline Enforcement | Active | `scripts/nomic_loop.py` (verify-fix cycle timeout) |
| Crux → Fix Guidance | Active | `scripts/nomic_loop.py` (belief network → fix prompts) |
| Probe → ELO Weighting | Active | `aragora/ranking/elo.py` (confidence_weight parameter) |
| Path Traversal Protection | Active | `aragora/tools/code.py` (_resolve_path validation) |
| Agent Consistency API | Active | `aragora/server/handlers/agents.py` (/api/agent/{name}/consistency) |
| Agent Network API | Active | `aragora/server/handlers/agents.py` (/api/agent/{name}/network) |
| MATCH_RECORDED Event | Active | `aragora/debate/orchestrator.py` (WebSocket emission) |
| Custom Mode Security | Active | `aragora/modes/custom.py` (path traversal protection) |
| Crux Cache Lifecycle | Active | `scripts/nomic_loop.py:run_cycle()` (cleared at cycle start) |
| Unified Serve CLI | Active | `aragora/cli/main.py:cmd_serve()` (unified server integration) |
| Circuit Breaker Persistence | Active | `scripts/nomic_loop.py` (saves/restores across cycles) |
| Circuit Breaker Agent Filtering | Active | `scripts/nomic_loop.py:_select_debate_team()` |
| AgentNetworkPanel | Active | `aragora/live/src/components/AgentNetworkPanel.tsx` |
| CapabilityProbePanel | Active | `aragora/live/src/components/CapabilityProbePanel.tsx` |
| OperationalModesPanel | Active | `aragora/live/src/components/OperationalModesPanel.tsx` |
| RedTeamAnalysisPanel | Active | `aragora/live/src/components/RedTeamAnalysisPanel.tsx` |
| Mood Event Types | Active | `aragora/server/stream/serializers.py` (MOOD_DETECTED, MOOD_SHIFT, DEBATE_ENERGY) |
| ContraryViewsPanel | Active | `aragora/live/src/components/ContraryViewsPanel.tsx` |
| RiskWarningsPanel | Active | `aragora/live/src/components/RiskWarningsPanel.tsx` |
| AnalyticsPanel | Active | `aragora/live/src/components/AnalyticsPanel.tsx` (disagreements, roles, early-stops) |
| CalibrationPanel | Active | `aragora/live/src/components/CalibrationPanel.tsx` (confidence accuracy) |
| ConsensusKnowledgeBase | Active | `aragora/live/src/components/ConsensusKnowledgeBase.tsx` (settled topics) |
| DebateViewer Critique Handling | Active | `aragora/live/src/components/DebateViewer.tsx` (critique + consensus) |
| ArgumentCartographer | Active | `aragora/debate/orchestrator.py` (graph visualization) |
| Graph Export API | Active | `aragora/server/handlers/debates.py` (/api/debate/{loop_id}/graph/*) |
| Audience Clusters API | Active | `aragora/server/handlers/debates.py` (/api/debate/{loop_id}/audience/clusters) |
| Replay Export API | Active | `aragora/server/handlers/replays.py` (/api/replays/*) |
| Database Query Indexes | Active | `aragora/ranking/elo.py` (8 indexes for common queries) |
| N+1 Query Optimization | Active | `aragora/ranking/elo.py` (get_rivals/get_allies batch) |
| Fork Initial Messages | Active | `aragora/debate/orchestrator.py` (initial_messages parameter) |
| Modular HTTP Handlers | Active | `aragora/server/handlers/` (41 handler modules) |
| Resource Availability Logging | Active | `aragora/server/unified_server.py` (_log_resource_availability) |
| Demo Consensus Fixtures | Active | `aragora/fixtures/__init__.py` (auto-seed on server startup) |
| Seed Demo API | Active | `aragora/server/handlers/consensus.py` (/api/consensus/seed-demo) |
| Broadcast Audio Generation | Active | `aragora/broadcast/` (TTS, mixing, storage) |
| Podcast RSS Feed | Active | `aragora/server/handlers/audio.py` (/api/podcast/feed.xml) |
| Audio File Serving | Active | `aragora/server/handlers/audio.py` (/audio/{id}.mp3) |
| Mistral Direct API | Active | `aragora/agents/api_agents/mistral.py` (MistralAPIAgent, CodestralAgent) |
| TeamSelector | Active | `aragora/debate/team_selector.py` (ELO+calibration scoring) |
| TricksterAlertPanel | Active | `aragora/live/src/components/TricksterAlertPanel.tsx` |
| RhetoricalObserverPanel | Active | `aragora/live/src/components/RhetoricalObserverPanel.tsx` |
| TrainingExportPanel | Active | `aragora/live/src/components/TrainingExportPanel.tsx` (SFT/DPO/Gauntlet export) |
| PublicGallery | Active | `aragora/live/src/components/PublicGallery.tsx` (debate browsing) |
| Token Revocation UI | Active | `aragora/live/src/components/SettingsPanel.tsx` (Logout All Devices) |
| Production Checklist | Active | `docs/PRODUCTION_CHECKLIST.md` (deployment guide) |

### Recently Surfaced (6)
| Feature | Status | Location |
|---------|--------|----------|
| Tournament System | TournamentPanel added | `aragora/live/src/components/TournamentPanel.tsx` |
| Agent Routing | Integrated in DebateInput | `aragora/live/src/components/DebateInput.tsx` |
| Belief Network | CruxPanel added | `aragora/live/src/components/CruxPanel.tsx` |
| Continuum Memory | MemoryInspector added | `aragora/live/src/components/MemoryInspector.tsx` |
| Persona Laboratory | LaboratoryPanel added | `aragora/live/src/components/LaboratoryPanel.tsx` |
| Prompt Evolution | LLM refinement implemented | `aragora/evolution/evolver.py` |

### Server Endpoints (72+ total)
- **Used by Frontend**: ~18%
- **Available but Unused**: ~50 endpoints
- **Key Gap**: Frontend uses WebSocket events, bypasses most REST endpoints
- **New APIs**: Introspection (3), Plugins (3), Genesis (4), Graph (3), Audience (1), Replay (2)

### Handler Integration Status (29 handlers)
| Handler | File | Routes | Status |
|---------|------|--------|--------|
| DebatesHandler | debates.py | 9+ | ✅ Active |
| AgentsHandler | agents.py | 6+ | ✅ Active |
| SystemHandler | system.py | 4 | ✅ Active |
| PulseHandler | pulse.py | 3 | ✅ Active |
| AnalyticsHandler | analytics.py | 4 | ✅ Active |
| MetricsHandler | metrics.py | 2 | ✅ Active |
| ConsensusHandler | consensus.py | 8 | ✅ Active |
| BeliefHandler | belief.py | 6 | ✅ Active |
| CritiqueHandler | critique.py | 3 | ✅ Active |
| GenesisHandler | genesis.py | 5 | ✅ Active |
| ReplaysHandler | replays.py | 4 | ✅ Active |
| TournamentHandler | tournaments.py | 3 | ✅ Active |
| MemoryHandler | memory.py | 5 | ✅ Active |
| LeaderboardViewHandler | leaderboard.py | 2 | ✅ Active |
| RelationshipHandler | relationship.py | 3 | ✅ Active |
| MomentsHandler | moments.py | 2 | ✅ Active |
| DocumentHandler | documents.py | 4 | ✅ Active |
| VerificationHandler | verification.py | 2 | ✅ Active |
| AuditingHandler | auditing.py | 3 | ✅ Active |
| DashboardHandler | dashboard.py | 3 | ✅ Active |
| PersonaHandler | persona.py | 4 | ✅ Active |
| IntrospectionHandler | introspection.py | 3 | ✅ Active |
| CalibrationHandler | calibration.py | 3 | ✅ Active |
| RoutingHandler | routing.py | 2 | ✅ Active |
| EvolutionHandler | evolution.py | 3 | ✅ Active |
| PluginsHandler | plugins.py | 3 | ✅ Active |
| BroadcastHandler | broadcast.py | 4 | ✅ Active |
| LaboratoryHandler | laboratory.py | 4 | ✅ Active |
| ProbesHandler | probes.py | 2 | ✅ Active |

**Handler Architecture**: All handlers inherit from `BaseHandler` with:
- `can_handle(path)` - Route matching
- `handle(path, query_params, http_handler)` - GET processing
- `handle_post(path, query_params, http_handler)` - POST processing
- `read_json_body(handler)` - Request body parsing
- `ttl_cache` decorator - Response caching
- `handle_errors` decorator - Centralized error handling

**Handler Test Coverage** (January 2026):
| Handler | Test File | Tests | Status |
|---------|-----------|-------|--------|
| DebatesHandler | test_handlers_debates.py | 30+ | STABLE |
| BeliefHandler | test_handlers_belief.py | 35 | STABLE |
| CalibrationHandler | test_handlers_calibration.py | 27 | STABLE |
| IntrospectionHandler | test_handlers_introspection.py | 25 | STABLE |
| MemoryHandler | test_handlers_memory.py | 22 | STABLE |
| SystemHandler | test_handlers_system.py | 20+ | STABLE |
| ConsensusHandler | test_handlers_consensus.py | 15+ | STABLE |
| BroadcastHandler | test_handlers_broadcast.py | 15+ | STABLE |

## Security Status

### Fixed (2026-01-04)
- Removed real API keys from `.env.local.example`
- Replaced full `__builtins__` with `SAFE_BUILTINS` in proofs.py
- Input validation on all POST endpoints
- Agent type allowlist prevents injection
- **IP-based rate limiting** (120 req/min per IP, DoS protection without auth)
- **Path traversal prevention** (SAFE_ID_PATTERN validation on replay_id, tournament_id)
- **Thread pool for debates** (max 10 concurrent, prevents resource exhaustion)
- **Rate limiter memory bounds** (LRU eviction when >10k entries)
- **CSP hardening** (removed unsafe-eval, blocks eval()/new Function() XSS vectors)

### Remaining Considerations
- ~~Token revocation mechanism not implemented~~ - **DONE** (Phase 13: JWT token versioning + UI)
- Consider API versioning for backwards compatibility

## Recommendations

### High Priority
1. ~~Activate Position Ledger by default~~ - **DONE** (initialized in server startup)
2. ~~Surface Tournament UI~~ - **DONE** (TournamentPanel added)
3. **Enable Belief Network visualization** - Crux analysis available but hidden

### Medium Priority
1. ~~Create Agent Routing UI~~ - **DONE** (integrated in DebateInput)
2. Implement Continuum Memory inspector
3. Add emergent traits browser from PersonaLaboratory

### Nomic Loop Improvements
1. **Better Task Splitting**: Decompose large tasks to avoid timeouts
2. **Pattern-based Agent Selection**: Route tasks to agents with best track record
3. **Cross-cycle Learning**: Persist insights between cycles via continuum.db

## Architecture Notes

### Nomic Loop Phase Architecture

The nomic loop (`scripts/nomic_loop.py`) implements a 6-phase self-improvement cycle:

| Phase | Class | Location | Purpose |
|-------|-------|----------|---------|
| 0 | `ContextPhase` | `scripts/nomic/phases/context.py` | Multi-agent codebase exploration |
| 1 | `DebatePhase` | `scripts/nomic/phases/debate.py` | Improvement proposal with hooks |
| 2 | `DesignPhase` | `scripts/nomic/phases/design.py` | Architecture planning |
| 3 | `ImplementPhase` | `scripts/nomic/phases/implement.py` | Hybrid code generation |
| 4 | `VerifyPhase` | `scripts/nomic/phases/verify.py` | Tests and quality checks |
| 5 | `CommitPhase` | `scripts/nomic/phases/commit.py` | Git commit with safety |

**Migration Status**: All phases have opt-in modular implementations via `USE_EXTRACTED_PHASES=1`.

**Factory Methods**: Each phase has a corresponding factory method in NomicLoop:
- `_create_context_phase()`, `_create_debate_phase()`, `_create_design_phase()`
- `_create_implement_phase()`, `_create_verify_phase()`, `_create_commit_phase()`

**PostDebateHooks**: 10 callback hooks for debate post-processing:
- `on_consensus_stored`, `on_calibration_recorded`, `on_insights_extracted`
- `on_memories_recorded`, `on_persona_recorded`, `on_patterns_extracted`
- `on_meta_analyzed`, `on_elo_recorded`, `on_claims_extracted`, `on_belief_network_built`

The codebase is **feature-rich with improving exposure**:
- 64+ API endpoints, ~15% used by frontend
- Many sophisticated features now surfaced via new APIs
- WebSocket-first architecture for real-time, REST for data access

**Recent Progress**:
- Z3 formal verification now active in post-debate flow
- Plugins API enables sandboxed evidence gathering
- Genesis API exposes evolution/lineage tracking
- Introspection API enables agent self-awareness
- Surprise-based ContinuumMemory learning now connected

**Key Insight**: Continuing to expose hidden features via REST APIs increases system utility without new core logic.

## Deployment

### Docker Deployment
```bash
# Quick start (requires .env file with API keys)
docker-compose up -d

# With frontend
docker-compose --profile with-frontend up -d

# View logs
docker-compose logs -f aragora
```

### Environment Variables
Required (at least one):
- `ANTHROPIC_API_KEY` - Anthropic Claude API
- `OPENAI_API_KEY` - OpenAI GPT API

Optional:
- `GEMINI_API_KEY` - Google Gemini API
- `XAI_API_KEY` - xAI Grok API
- `MISTRAL_API_KEY` - Mistral AI API (Large, Codestral)
- `OPENROUTER_API_KEY` - OpenRouter (DeepSeek, Llama, Qwen, Yi)
- `ARAGORA_API_TOKEN` - Optional authentication token
- `ARAGORA_ALLOWED_ORIGINS` - CORS origins (default: http://localhost:3000)

### Health Check
```bash
curl http://localhost:8080/api/health
```

### Recent Additions (2026-01-05)
- **End-to-End Integration Tests**: 22 new tests covering full debate flows
- **Dockerfile**: Production-ready container with non-root user
- **docker-compose.yml**: Complete orchestration with volume persistence
