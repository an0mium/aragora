# Changelog

All notable changes to Aragora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Documentation

- Updated `docs/ENVIRONMENT.md` with Redis cluster configuration variables
- Added pluggable storage backend documentation

### Testing

- Added integration tests for notification metrics (14 tests)
- Added integration tests for Redis clustering (16 tests)
- Added consensus dissent tracking tests (400+ lines)

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
