# Aragora Project Status

*Last updated: January 5, 2026 (02:00 UTC)*

## Current State

### Test Status
- **Total Tests**: 605 passed, 0 failures, 9 skipped
- **Recent Fixes (2026-01-05)**:
  - Fixed `_get_belief_classes()` → `_get_belief_analyzer()` typo in orchestrator.py
  - Fixed all 7 unanimous consensus tests (were failing due to above typo)
  - ELO tests (3), calibration test (1), replay tests (4) fixed in previous session
- **Code Fixes**:
  - Calibration bucket boundary now includes confidence=1.0
  - Belief analyzer function name corrected in orchestrator.py:2213

### Nomic Loop
- **Cycle**: 2 (starting fresh after stalled cycle 1)
- **Phase**: debate (starting)
- **Last Proposal**: "Debate Mood Ring" (80% consensus, stalled at 0% design consensus)
- **Cycle 1 Issue**: Design phase had 0% consensus → plan generation failed → loop stalled
- **Fix Applied**: Added design consensus check before implementation (skips if no design)
- **Blocking Issues FIXED**:
  - Missing `agent_type` attribute in GeminiAgent (now added to all API agents)
  - RelationshipTracker.get_influence_network() parameter mismatch (fixed)
  - OpenRouterAgent broken super().__init__() call (fixed)
  - `_get_belief_classes()` undefined (fixed - was typo for `_get_belief_analyzer()`)
  - Design phase 0% consensus now detected and cycle skipped (new safeguard)
- **Position Ledger**: Implemented in `aragora/agents/grounded.py`
- **NomicIntegration**: Fully wired up (probing, belief analysis, checkpointing, staleness)

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

### Fully Integrated (54)
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
| Mood Event Types | Active | `aragora/server/stream.py` (MOOD_DETECTED, MOOD_SHIFT, DEBATE_ENERGY) |
| ContraryViewsPanel | Active | `aragora/live/src/components/ContraryViewsPanel.tsx` |
| RiskWarningsPanel | Active | `aragora/live/src/components/RiskWarningsPanel.tsx` |
| AnalyticsPanel | Active | `aragora/live/src/components/AnalyticsPanel.tsx` (disagreements, roles, early-stops) |
| CalibrationPanel | Active | `aragora/live/src/components/CalibrationPanel.tsx` (confidence accuracy) |
| ConsensusKnowledgeBase | Active | `aragora/live/src/components/ConsensusKnowledgeBase.tsx` (settled topics) |
| DebateViewer Critique Handling | Active | `aragora/live/src/components/DebateViewer.tsx` (critique + consensus) |
| ArgumentCartographer | Active | `aragora/debate/orchestrator.py` (graph visualization) |
| Graph Export API | Active | `aragora/server/stream.py` (/api/debate/{loop_id}/graph/*) |
| Audience Clusters API | Active | `aragora/server/stream.py` (/api/debate/{loop_id}/audience/clusters) |
| Replay Export API | Active | `aragora/server/stream.py` (/api/replays/*) |
| Database Query Indexes | Active | `aragora/ranking/elo.py` (8 indexes for common queries) |
| N+1 Query Optimization | Active | `aragora/ranking/elo.py` (get_rivals/get_allies batch) |

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
- `OPENROUTER_API_KEY` - OpenRouter (DeepSeek, Llama, Mistral)
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
