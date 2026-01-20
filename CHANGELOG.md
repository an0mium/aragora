# Changelog

All notable changes to Aragora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
