# Aragora Project Status

*Last updated: January 4, 2026*

## Current State

### Nomic Loop
- **Cycle**: 1
- **Phase**: debate (arena_starting - was blocked, now fixed)
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

## Feature Integration Status

### Fully Integrated (8)
| Feature | Status | Location |
|---------|--------|----------|
| Multi-Agent Debate | Active | `aragora/debate/orchestrator.py` |
| Token Streaming | Active | `aragora/agents/api_agents.py` |
| ELO Rankings | Active | `aragora/ranking/elo.py` |
| FlipDetector | Active | `aragora/insights/flip_detector.py` |
| Position Ledger | Active | `aragora/agents/grounded.py` |
| Calibration Tracking | Active | `aragora/agents/calibration.py` |
| Convergence Detection | Active | `aragora/debate/convergence.py` |
| Role Rotation | Active | `aragora/debate/roles.py` |

### Implemented but Underutilized (1)
| Feature | Issue | Location |
|---------|-------|----------|
| Formal Verification | Lean backend not connected | `aragora/verification/formal.py` |

### Recently Surfaced (6)
| Feature | Status | Location |
|---------|--------|----------|
| Tournament System | TournamentPanel added | `aragora/live/src/components/TournamentPanel.tsx` |
| Agent Routing | Integrated in DebateInput | `aragora/live/src/components/DebateInput.tsx` |
| Belief Network | CruxPanel added | `aragora/live/src/components/CruxPanel.tsx` |
| Continuum Memory | MemoryInspector added | `aragora/live/src/components/MemoryInspector.tsx` |
| Persona Laboratory | LaboratoryPanel added | `aragora/live/src/components/LaboratoryPanel.tsx` |
| Prompt Evolution | LLM refinement implemented | `aragora/evolution/evolver.py` |

### Server Endpoints (54 total)
- **Used by Frontend**: ~10%
- **Available but Unused**: 52 endpoints
- **Key Gap**: Frontend uses WebSocket events, bypasses most REST endpoints

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

The codebase is **feature-rich but under-exposed**:
- 54 API endpoints, ~10% used by frontend
- Many sophisticated features implemented but not surfaced
- WebSocket-first architecture means REST endpoints underutilized

**Key Insight**: Enabling more optional features by default would significantly increase system capability without new code.
