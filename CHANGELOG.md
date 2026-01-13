# Changelog

All notable changes to Aragora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SOC2, PCI-DSS, and NIST CSF compliance personas for Gauntlet mode
- `scripts/test_tiers.sh` for common test tiers (fast, ci, lint, typecheck, frontend, e2e)
- `scripts/cleanup_runtime_artifacts.sh` to relocate root-level runtime DB artifacts

### Changed
- Onboarding docs now point to `docs/START_HERE.md` / `docs/GETTING_STARTED.md` as the canonical entry
- Database docs now default to `ARAGORA_DATA_DIR` (`.nomic`) and clarify legacy paths
- Frontend docs clarify dashboard vs SDK vs legacy frontend

### Focus
- Stabilization target: onboarding, test tiers, and runtime data hygiene (0.8.1)

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
