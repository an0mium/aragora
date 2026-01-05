# Aragora Project Status

*Last updated: January 5, 2026 (03:30 UTC)*

## Current State

### Nomic Loop
- **Cycle**: 1
- **Phase**: debate (Round 1 - Live Position Flip Telemetry proposal)
- **Last Proposal**: Claude's "Persona Laboratory v2" (won 2/3 consensus)
- **Implementation**: Failed on verification (timeout issues)
- **Blocking Issues FIXED**:
  - Missing `agent_type` attribute in GeminiAgent (now added to all API agents)
  - RelationshipTracker.get_influence_network() parameter mismatch (fixed)
  - OpenRouterAgent broken super().__init__() call (fixed)
- **Position Ledger**: Implemented in `aragora/agents/grounded.py`

### Active Agents (4 default, 12+ total)
| Agent | Model | API |
|-------|-------|-----|
| `grok` | Grok 4 | xAI |
| `anthropic-api` | Claude Opus 4.5 | Anthropic |
| `openai-api` | GPT 5.2 | OpenAI |
| `deepseek-r1` | DeepSeek V3.2 | OpenRouter |

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

## Feature Integration Status

### Fully Integrated (41)
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
| Agent Consistency API | Active | `aragora/server/stream.py` (/api/agent/{name}/consistency) |
| Agent Network API | Active | `aragora/server/stream.py` (/api/agent/{name}/network) |
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

### Recently Surfaced (6)
| Feature | Status | Location |
|---------|--------|----------|
| Tournament System | TournamentPanel added | `aragora/live/src/components/TournamentPanel.tsx` |
| Agent Routing | Integrated in DebateInput | `aragora/live/src/components/DebateInput.tsx` |
| Belief Network | CruxPanel added | `aragora/live/src/components/CruxPanel.tsx` |
| Continuum Memory | MemoryInspector added | `aragora/live/src/components/MemoryInspector.tsx` |
| Persona Laboratory | LaboratoryPanel added | `aragora/live/src/components/LaboratoryPanel.tsx` |
| Prompt Evolution | LLM refinement implemented | `aragora/evolution/evolver.py` |

### Server Endpoints (64+ total)
- **Used by Frontend**: ~15%
- **Available but Unused**: ~50 endpoints
- **Key Gap**: Frontend uses WebSocket events, bypasses most REST endpoints
- **New APIs**: Introspection (3), Plugins (3), Genesis (4)

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

### Remaining Considerations
- Token revocation mechanism not implemented
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
