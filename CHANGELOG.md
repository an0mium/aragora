# Changelog

All notable changes to Aragora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.4] - 2026-01-21

### Added

- **Unified Audit Module** - Centralized audit logging:
  - Consistent audit trail format across all operations
  - Structured event types with metadata
  - Integration with immutable log for compliance
  - 678 lines of audit infrastructure

- **Security Migration Utilities** - Credential rotation support:
  - API key rotation workflows
  - Secret migration between providers
  - Rollback capabilities for failed migrations
  - Audit trail for all credential changes

- **Checkpoint Edge Case Tests** - Enhanced checkpoint reliability:
  - Compression roundtrip verification
  - Concurrent checkpoint creation handling
  - Large message content handling
  - Expiration detection tests

- **Handler Integration Tests** - Comprehensive handler coverage:
  - Secure handler with RBAC tests (811 lines)
  - External integrations handler tests (389 lines)
  - Connector tests (Google Chat, Voice Bridge, HackerNews)

- **Encryption Performance Benchmarks** - Security performance validation:
  - Field encryption throughput tests
  - Key derivation benchmarks
  - Envelope encryption performance

### Changed

- **AdminHandler** - Enhanced with RBAC permission checks
- **Shutdown Sequence** - Improved graceful cleanup
- **Startup** - Enhanced initialization with validation

### Fixed

- **Circuit Breaker Tests** - Fixed flaky test with UUID-based service names
- **Checkpoint Resume Tests** - Fixed SQLite fallback issue
- **Import Errors** - Fixed BaseHandler import in admin module

## [2.1.3] - 2026-01-21

### Added

- **Approval Gate Middleware** - Human-in-the-loop approval for sensitive operations:
  - Risk level classification (low, medium, high, critical)
  - Customizable approval checklists
  - Integration with GovernanceStore for persistence
  - Decorator and middleware usage patterns

- **Secure Handler Base** - Security-enhanced HTTP handler:
  - Automatic JWT verification
  - Built-in RBAC permission enforcement
  - Audit trail logging for security events
  - Security metrics emission

- **Webhook Delivery Manager** - Reliable webhook delivery:
  - Delivery status tracking (pending, delivered, failed, dead-lettered)
  - Retry queue with exponential backoff
  - Dead-letter queue for consistently failing webhooks
  - Delivery SLA metrics

- **DecisionHandler Tests** - Comprehensive test coverage for `/api/decisions`:
  - Route matching tests
  - GET/POST operation tests
  - Result caching tests

- **Cache Invalidation** - Enhanced decision routing middleware:
  - Tagged entries for selective invalidation
  - Workspace-scoped invalidation
  - Policy version tracking
  - Cache statistics and monitoring

### Changed

- **Redis now optional** for initial production deployment (with warning)
- **RBAC enforcement** added to connectors handler

### Fixed

- **Migration script metrics** - Added tracking for migration outcomes
- **ENCRYPTED_FIELDS export** - Now properly exported from gmail_token_store

## [2.1.2] - 2026-01-21

### Added

- **Cloud KMS Provider** - Multi-cloud key management integration:
  - AWS KMS support via boto3
  - Azure Key Vault support via azure-identity
  - GCP Cloud KMS support via google-cloud-kms
  - Auto-detection of cloud platform from environment
  - Envelope encryption with data key decrypt

- **Encrypted Fields** - Field-level encryption for sensitive data:
  - Automatic encryption of OAuth tokens, API keys, secrets
  - Platform-specific credential encryption (Slack, Discord, Telegram, etc.)
  - Transparent encrypt/decrypt on storage operations

- **Decision Routing Middleware** - Smart request routing:
  - `aragora/server/middleware/decision_routing.py` - Request routing based on decision context
  - RBAC-aware routing decisions

- **Re-enabled Integration Tests** - 47 previously skipped tests now passing:
  - TestArenaRun (7 tests)
  - TestRoundExecutionFlow (5 tests)
  - TestPhaseTransitions (5 tests)
  - TestEarlyTerminationOnConvergence (5 tests)
  - TestConsensusDetection (5 tests)
  - TestErrorHandling (5 tests)
  - TestMemoryIntegration (5 tests)
  - TestFullDebateFlow (7 tests)
  - TestAgentFailureHandling (3 tests)
  - Debate performance benchmarks re-enabled

- **Production Deployment Documentation**:
  - `docs/PRODUCTION_DEPLOYMENT.md` - Comprehensive ops guide
  - `docs/SUPABASE_SETUP.md` - Hosted PostgreSQL setup

- **Migration Configuration Tests** - 7 new tests verifying Alembic setup:
  - `test_alembic_config_exists` - Validates alembic.ini and directory structure
  - `test_initial_schema_exists` - Verifies schema SQL file
  - `test_initial_schema_has_required_tables` - Checks all 11 required tables
  - `test_init_postgres_script_exists` - Validates initialization script
  - `test_migration_env_uses_asyncpg` - Confirms async support
  - `test_migration_version_exists` - Verifies migration versions

- **Architecture Documentation** - Added Storage Architecture section:
  - PostgreSQL store implementations table (13 stores)
  - Database migration commands
  - Transaction safety documentation

### Fixed

- **Circuit Breaker Test Isolation** - Fixed flaky `test_get_metrics_provides_summary`:
  - Uses UUID-based service names to avoid cross-test interference
  - Added specific assertion for created breaker in metrics

### Changed

- **Documentation Updates**:
  - `docs/ENVIRONMENT.md` - Added Alembic migration commands
  - `docs/ARCHITECTURE.md` - Added Storage Architecture section with PostgreSQL details
  - Updated performance metrics (38,000+ tests, 250+ strict modules, 468 storage tests)

## [2.1.1] - 2026-01-20

### Added

- **Twilio Voice Integration** - Bidirectional voice for phone-triggered debates:
  - Inbound call handling with speech-to-text transcription
  - TwiML response generation for interactive prompts
  - Outbound calls with TTS for debate results
  - Call session tracking and management
  - Webhook signature verification
  - New endpoints: `/api/voice/inbound`, `/api/voice/status`, `/api/voice/gather`

- **Enhanced Observability** - Comprehensive metrics and health monitoring:
  - Circuit breaker state tracking
  - Debate throughput metrics
  - Memory usage monitoring
  - Readiness and liveness probe support

- **PostgreSQL Backends** - Full horizontal scaling support for all 11 storage modules:
  - `PostgresWebhookConfigStore` - Webhook configuration persistence
  - `PostgresIntegrationStore` - Chat platform integrations
  - `PostgresGmailTokenStore` - OAuth token storage
  - `PostgresFindingWorkflowStore` - Audit workflow state
  - `PostgresGauntletRunStore` - In-flight gauntlet runs
  - `PostgresApprovalRequestStore` - Human approval requests
  - `PostgresJobQueueStore` - Background job queue
  - `PostgresMarketplaceStore` - Template marketplace
  - `PostgresTokenBlacklistStore` - JWT revocation
  - `PostgresFederationRegistryStore` - Multi-region federation
  - `PostgresGovernanceStore` - Decision governance artifacts
  - Unified schema in `migrations/sql/001_initial_schema.sql` (383 lines)
  - Alembic migration framework with async asyncpg support
  - Atomic transaction handling for multi-table operations
  - Factory functions support `ARAGORA_DB_BACKEND=postgres` environment variable

### Changed

- **Migrations Restructure** - Moved schema to `migrations/sql/001_initial_schema.sql`
- **Operations Documentation** - Updated with monitoring and alerting guidance
- **Transaction Safety** - Added explicit transactions to:
  - `PostgresGovernanceStore.cleanup_old_records_async()` - Atomic multi-table cleanup
  - `PostgresFederationRegistryStore.update_sync_status()` - Atomic counter increments

## [2.1.0] - 2026-01-20

### Added

- **JWT Webhook Verification** - Proper cryptographic verification for chat platform webhooks:
  - Microsoft Teams: JWT validation against Azure AD public keys
  - Google Chat: JWT validation against Google's JWKS endpoint
  - Graceful fallback when PyJWT not installed (with security warning)
  - JWKS client caching with hourly refresh

- **AgentSpec Test Suite** - Comprehensive tests for unified agent specification:
  - Creation with explicit fields
  - Deprecated string parsing with warnings
  - Team creation from dicts
  - String serialization and equality

- **Chat Webhook Router Tests** - Platform detection and routing tests:
  - Slack, Discord, Telegram, WhatsApp detection
  - Connector caching behavior
  - Signature format verification

### Changed

- **rlm_backend Deprecation** - Improved deprecation handling:
  - Now uses sentinel value to detect ANY explicit use
  - Warns for all uses (not just non-default values)
  - Documented removal planned for v3.0

### Security

- **Webhook Signature Verification** - Teams and Google Chat webhooks now properly validate JWT signatures instead of just checking for Bearer token presence

## [2.0.8] - 2026-01-20

### Added

- **Cross-Pollination Health Endpoint** - `/api/health/cross-pollination` for monitoring feature integrations:
  - ELO weighting status
  - Calibration tracking health
  - Evidence quality scoring
  - RLM hierarchy caching metrics
  - Knowledge Mound operations
  - Pulse trending topics

- **Cross-Pollination CLI Flags** - Control features via CLI:
  - `--no-elo-weighting` - Disable ELO-based vote weights
  - `--no-calibration` - Disable calibration tracking
  - `--no-evidence-weighting` - Disable evidence quality scoring
  - `--no-trending` - Disable Pulse trending topics

- **Cross-Pollination Grafana Dashboard** - `deploy/grafana/dashboards/cross-pollination.json`:
  - RLM cache hit rate gauge
  - Calibration error (ECE) gauge
  - Voting accuracy rate
  - ELO weight adjustments
  - Calibration adjustments by agent
  - Evidence quality bonuses
  - Knowledge Mound operations

- **CultureAdapter** - Bridges Culture Accumulator to Knowledge Mound:
  - Culture pattern storage after debates
  - Pattern retrieval for protocol configuration
  - Cross-workspace culture promotion

- **MultiInboxManager** - Unified Gmail account management:
  - Multi-account authentication
  - Cross-account sender intelligence
  - Unified prioritization

- **Template Marketplace** - Community workflow template sharing:
  - SQLite-backed storage
  - Rating and review system
  - Category browsing

- **GauntletRunStore** - Persistent storage for in-flight gauntlet runs:
  - InMemory, SQLite, and Redis backends
  - Status tracking (pending/running/completed/failed/cancelled)
  - Template and workspace filtering
  - Active run listing with Redis index support

- **ApprovalRequestStore** - Persistent storage for human approval requests:
  - InMemory, SQLite, and Redis backends
  - Workflow and step tracking
  - Expiration management
  - Response recording with responder tracking
  - Priority-based ordering

### Fixed

- **EmailHandler Interface** - Added missing `can_handle` and `handle` methods required by BaseHandler

### Changed

- **Cross-Pollination Enabled by Default** - All features now enabled out of the box
- **Quick Start Documentation** - Added to CROSS_POLLINATION.md
- **Strict Type Checking Expansion** - Added 19 modules to strict mypy (Phase 27):
  - Workflow templates: `__init__`, `package`, `patterns`, `legal`, `healthcare`, `accounting`, `ai_ml`, `code`, `devops`, `product`
  - Misc modules: `server.stream.events`, `client.resources.leaderboard`, `debate.safety`, `replay.schema`, `modes.builtin.orchestrator`, `server.handlers.knowledge`, `broadcast.mixer`
  - Continuum extractions: `continuum_glacial`, `continuum_snapshot`

- **Continuum Memory Refactoring** - Reduced `continuum.py` (1,995 → 1,612 lines):
  - Extracted `ContinuumGlacialMixin` to `continuum_glacial.py` (222 lines)
  - Extracted `ContinuumSnapshotMixin` to `continuum_snapshot.py` (267 lines)
  - `ContinuumMemory` now inherits from mixins for cleaner separation

## [2.0.7] - 2026-01-20

### Added

- **Batch Explainability Panel** - UI component for multi-debate analysis
- **Culture Integration Tests** - Tests for debate culture profile processing
- **Mode Handoff Tests** - Tests for transition contexts between modes

### Fixed

- **BeliefNetwork API Mismatch** - Fixed `WinnerSelector.analyze_belief_network()` to use correct `add_claim()` signature and `CruxDetector` for crux identification
- **MockDebateResult Test Fixture** - Added missing attributes (`consensus_reached`, `proposals`, `participants`, etc.)
- **Knowledge Mound Query Pagination** - Added `offset` parameter to `QueryOperationsMixin.query()` for proper pagination support
- **Knowledge Bridge Tests** - Fixed `MockKnowledgeMound` missing `workspace_id` attribute
- **Checkpoint Restore** - Graceful handling of corrupted checkpoint files with proper error messages
- **Checkpoint Test Timing** - Changed `duration_ms > 0` assertion to `>= 0` for fast operations
- **Analytics Handler** - Fixed `get_knowledge_mound()` being incorrectly awaited (it's synchronous)
- **Federation Tests** - Added proper mocking of persistent store to isolate tests from SQLite data

### Changed

- **Knowledge Mound Federation** - Methods now fall back to class-level cache when persistent store unavailable
- **Test Isolation** - Federation tests now properly isolated from persistent SQLite data

## [2.0.6] - 2026-01-20

### Added

- **Notification Delivery Metrics** - Prometheus metrics for notification tracking:
  - `aragora_notification_sent_total` - Counts by channel/severity/priority/status
  - `aragora_notification_latency_seconds` - Delivery latency histogram by channel
  - `aragora_notification_errors_total` - Error counts by channel and error type
  - `aragora_notification_queue_size` - Gauge for queue size by channel
  - Integrated into Slack, Email, and Webhook providers

- **Redis Cluster Support** - High-availability Redis for multi-instance deployments:
  - `RedisClusterClient` with automatic cluster/standalone detection
  - `get_redis_client()` utility for unified Redis access
  - Connection pooling with health monitoring
  - Graceful failover and automatic reconnection
  - Read replica support for scaling reads
  - Hash tag support for slot affinity

- **Enhanced Consensus Ingestion Metrics** - Track dissent and evolution:
  - `aragora_consensus_dissent_ingested_total` - Dissenting views captured
  - `aragora_consensus_evolution_tracked_total` - Supersession relationships
  - `aragora_consensus_evidence_linked_total` - Evidence linking counts
  - `aragora_consensus_agreement_ratio` - Agreement ratio distribution

- **Database Migration** - `v20260120000000_channel_governance_stores.py`:
  - `integration_configs` - Chat platform configurations
  - `gmail_tokens` - Gmail OAuth token storage
  - `gmail_sync_jobs` - Gmail sync job state
  - `finding_workflows` - Audit finding workflow state
  - `federation_registry` - Multi-region federation config

### Fixed

- **ELO System Tests** - Fixed tests to properly persist ratings to database
- **Metrics Package Import** - Fixed missing submodules in metrics package refactoring:
  - Created `aragora/observability/metrics/bridge.py` for cross-pollination metrics
  - Created `aragora/observability/metrics/km.py` for Knowledge Mound metrics
  - Created `aragora/observability/metrics/notification.py` for notification metrics
  - Added backward compatibility shims for `_init_metrics` and `_init_noop_metrics`
- **Workflow Templates Schema** - Removed pattern templates from `WORKFLOW_TEMPLATES` registry
  as they use a different schema (factory patterns vs fixed step workflows)

### Changed

- **Belief Module Refactoring** - Reduced `belief.py` (1,593 → 1,001 lines):
  - Extracted `CruxDetector`, `CruxClaim`, `CruxAnalysisResult` to `crux_detector.py`
  - Extracted `BeliefPropagationAnalyzer` to `crux_detector.py`
  - Backward compatible re-exports maintained

- **Strict Type Checking Expansion** - Added 21 modules to strict mypy (Phase 25-26):
  - Phase 25: `crux_detector`, `metrics/bridge`, `metrics/km`, `metrics/notification`
  - Phase 25: `streaming_mixin`, `resources/gauntlet`, `resources/graph_debates`
  - Phase 25: `resources/matrix_debates`, `resources/replay`
  - Phase 26: `resources/agents`, `resources/memory`, `resources/verification`
  - Phase 26: `modes/tool_groups`, `modes/builtin/*` (architect, coder, debugger, reviewer)
  - Phase 26: `ranking/verification`, `billing/jwt_auth`, `spectate/events`, `mcp/tools_module/gauntlet`

### Documentation

- Updated `docs/ENVIRONMENT.md` with Redis cluster configuration variables
- Added pluggable storage backend documentation
- Updated `docs/ARCHITECTURE.md` with RLM and client module structure
- Updated architecture metrics (35,784 tests, 38 strict modules, 495K LOC)

### Testing

- Added integration tests for notification metrics (14 tests)
- Added integration tests for Redis clustering (16 tests)
- Added consensus dissent tracking tests (400+ lines)
- Audited skipped tests: 158 conditional (optional deps) + 17 unconditional (E2E/future API)

## [2.0.5] - 2026-01-20

### Changed

- **RLM Bridge Refactoring** - Further reduced `bridge.py` (979 → 691 lines):
  - Extracted streaming methods to `streaming_mixin.py` (330 lines)
  - `RLMStreamingMixin` provides `query_stream()`, `query_with_refinement_stream()`, `compress_stream()`
  - `AragoraRLM` now inherits from `RLMStreamingMixin` for cleaner separation of concerns

- **Client Module Refactoring** - Reduced `client.py` (1,101 → 694 lines):
  - Extracted `GauntletAPI` to `resources/gauntlet.py`
  - Extracted `GraphDebatesAPI` to `resources/graph_debates.py`
  - Extracted `MatrixDebatesAPI` to `resources/matrix_debates.py`
  - Extracted `ReplayAPI` to `resources/replay.py`

### Added

- **New Prometheus Metrics** for observability:
  - TTS metrics: `aragora_tts_synthesis_total`, `aragora_tts_latency_seconds`
  - Convergence metrics: `aragora_convergence_check_total`, `aragora_convergence_blocked_total`
  - RLM metrics: `aragora_rlm_ready_quorum_total`
  - Vote bonus metrics: `aragora_evidence_citation_bonus_total`, `aragora_process_evaluation_bonus_total`

- **Integration Tests** for new features (19 tests):
  - TTS integration: initialization, event bus registration, agent message handling, rate limiting
  - Convergence tracker: initialization, reset, convergence check, novelty tracking
  - Vote bonus calculator: evidence citation bonuses, quality weighting
  - RLM streaming mixin: event sequence, error handling, node examination
  - Cross-feature integration: singleton patterns, metrics recording

### Technical

- Metrics wired in: `tts_integration.py`, `convergence_tracker.py`, `vote_bonus_calculator.py`
- Type ignore audit: 208 total (mostly legitimate decorator/mixin/optional-import patterns)

## [2.0.4] - 2026-01-20

### Changed

- **Phase 24: Module Extractions** - Reduced large file sizes for maintainability:
  - `prometheus.py` (1,460 → 994 lines): Extracted to `prometheus_nomic.py`, `prometheus_control_plane.py`, `prometheus_rlm.py`, `prometheus_knowledge.py`
  - `feedback_phase.py` (1,628 → 1,191 lines): Extracted to `feedback_elo.py`, `feedback_persona.py`, `feedback_evolution.py`
  - `bridge.py` (1,701 → 979 lines): Extracted to `debate_adapter.py`, `knowledge_adapter.py`, `hierarchy_cache.py`

- **Strict Type Checking** - Expanded mypy strict mode to 29 modules (from 19):
  - Added: `prometheus_nomic`, `prometheus_control_plane`, `prometheus_rlm`
  - Added: `feedback_elo`, `feedback_persona`, `feedback_evolution`
  - Added: `debate_adapter`, `knowledge_adapter`, `hierarchy_cache`, `rlm/types`

### Fixed

- Fixed undefined name errors (`AgentSpec`, `KnowledgeItem`, `DB_KNOWLEDGE_PATH`, `asyncio`)
- Fixed circular imports in prometheus module re-exports
- Fixed duplicate imports in `openai_compatible.py`
- Fixed ambiguous variable name `l` → `label` in `email_priority.py`
- Removed unused local variables in `backup/manager.py`
- Updated import paths for metrics functions to use extracted submodules

### Documentation

- Updated `.github/workflows/lint.yml` to include 29 strict-checked modules
- Added comments documenting intentional broad exception handlers with `# noqa: BLE001`

## [2.0.3] - 2026-01-20

### Added

- **Cross-Functional Integration** - 7 major features now fully wired:
  - `KnowledgeBridgeHub` - Unified access to MetaLearner, Evidence, and Pattern bridges
  - `MemoryCoordinator` - Atomic writes across multiple memory systems
  - `SelectionFeedbackLoop` - Performance-based agent selection weights
  - `CrossDebateMemory` - Institutional knowledge injection from past debates
  - `EvidenceBridge` - Persist collected evidence in KnowledgeMound
  - `CultureAccumulator` - Extract organizational patterns from debates
  - Post-debate workflow triggers for automated refinement

- **New Prometheus Metrics** (7 metrics for cross-functional features):
  - `aragora_knowledge_cache_hits_total` / `aragora_knowledge_cache_misses_total`
  - `aragora_memory_coordinator_writes_total`
  - `aragora_selection_feedback_adjustments_total`
  - `aragora_workflow_triggers_total`
  - `aragora_evidence_stored_total`
  - `aragora_culture_patterns_total`

- **Documentation**:
  - `docs/CROSS_FUNCTIONAL_FEATURES.md` - Comprehensive usage guide
  - Updated `docs/STATUS.md` with 88 fully integrated features

- **Tests**:
  - Knowledge mound ops handler tests (793 LOC)
  - Trickster calibrator tests
  - Cross-pollination integration tests
  - Modes module tests (base, deep_audit, redteam, tool_groups)

### Changed

- `AgentSpec` now always assigns a default role of "proposer" when not specified
- Legacy colon format (`claude:critic`) sets persona, not role (pipe format for explicit roles)
- Updated CLI `parse_agents` tests to match new AgentSpec behavior

### Fixed

- CLI test failures for `parse_agents` function
- Arena.from_config `enable_adaptive_rounds` NameError

## [2.0.2] - 2026-01-19

### Added

- UI Enhancement Release with improved connectors and uncertainty pages
- Inbox components for notifications
- Knowledge Mound federation operations and visibility controls
- Snapshot/restore for checkpoint integration
- Calibration and learning feedback loops

### Fixed

- Agent spec parsing 3-part spec bug
- Usage sync persistence watermarks
- Context gatherer cache task-hash keying
- React side effect in setLastAllConnected
- Cross-debate memory snapshot race condition

## [2.0.1] - 2026-01-18

### Added

- Feature Integration & Consolidation Release
- OAuth endpoints (Google, GitHub)
- Workflow API endpoints
- Workspace management endpoints
- Leader election for distributed coordination

## [2.0.0] - 2026-01-17

### Added

- Enterprise & Production Hardening Release
- Graph and Matrix debate APIs
- Breakpoint API for human-in-the-loop
- Billing and subscription endpoints
- Memory analytics endpoints
- A/B testing for agent evolution

## [1.5.x] - 2026-01-15

### Added

- E2E Testing & SOC 2 Compliance
- OAuth, SDK Probes & Performance improvements
- Admin UI & Governance features

---

For detailed feature status, see [docs/STATUS.md](docs/STATUS.md).
