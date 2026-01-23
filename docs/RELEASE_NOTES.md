# Aragora Release Notes

---

# v2.0.6 - Stability Release

**Release Date:** January 20, 2026
**Version:** 2.0.6
**Codename:** Pulse Ascension

---

## Overview

Aragora 2.0.6 promotes Pulse (trending topics) to stable status, fixes all deprecated asyncio patterns, and adds bidirectional Knowledge Mound adapters. This release focuses on stability, test quality, and cross-pollination completion.

---

## What's New

### Pulse Promotion to Stable

Pulse is now production-ready with 358+ tests passing:

| Component | Tests | Status |
|-----------|-------|--------|
| Quality Filtering | 45+ | ✅ Stable |
| Freshness Scoring | 30+ | ✅ Stable |
| Source Weighting | 25+ | ✅ Stable |
| Ingestors (HN/Reddit/Twitter) | 80+ | ✅ Stable |
| Scheduler | 40+ | ✅ Stable |
| Store | 50+ | ✅ Stable |

```python
from aragora.pulse.ingestor import PulseManager

manager = PulseManager(
    enable_hackernews=True,
    enable_reddit=True,
    quality_threshold=0.6,
)

topics = await manager.get_trending_topics(limit=5)
```

### ANN Similarity Backend

Fast approximate nearest neighbor search for convergence detection:

- FAISS index integration for large-scale similarity
- 26 ANN-specific tests
- Configurable via `ARAGORA_CONVERGENCE_BACKEND=ann`

### Asyncio Modernization

Fixed 34 deprecated `get_event_loop().run_until_complete()` patterns across 11 test files. All tests now use `asyncio.run()`.

### Bidirectional Knowledge Adapters

New adapters with reverse data flows:

- `BidirectionalCoordinator` - Manages two-way data sync
- Pulse adapter with bidirectional support
- Tests for adapter persistence and coordination

### Cross-Pollination Benchmarks

New benchmark suite measuring cross-pollination performance:

- Weight calculation: <0.5ms per call
- ELO lookups: <1ms per call
- Cache operations: <0.01ms per call
- Calibration records: <10ms per call

---

## Test Improvements

- Fixed role rotation test (role_matching priority issue)
- Fixed feedback phase test (delegated objects pattern)
- Added fast Jaccard backend for test performance
- 38,100+ tests across 1,047 files

---

## Breaking Changes

None.

---

## Upgrade Notes

No special upgrade steps required. The release is backwards compatible.

---

# v2.0.3 - Cross-Functional Integration

**Release Date:** January 20, 2026
**Version:** 2.0.3
**Codename:** Unified Intelligence

---

## Overview

Aragora 2.0.3 activates 700+ lines of previously implemented but disconnected features, creating a unified system where subsystems share data and enhance each other. This release connects KnowledgeMound, MemoryCoordinator, SelectionFeedbackLoop, and other components into a cohesive whole.

---

## What's New

### Cross-Functional Features

Seven major integrations now fully wired:

| Feature | Purpose | Default |
|---------|---------|---------|
| `KnowledgeBridgeHub` | Unified access to MetaLearner, Evidence, Pattern bridges | Auto-enabled with KnowledgeMound |
| `MemoryCoordinator` | Atomic writes across memory systems | `enable_coordinated_writes=True` |
| `SelectionFeedbackLoop` | Performance-based agent selection | `enable_performance_feedback=True` |
| `CrossDebateMemory` | Institutional knowledge injection | `enable_cross_debate_memory=True` |
| `EvidenceBridge` | Persist collected evidence | Auto-enabled with KnowledgeBridgeHub |
| `CultureAccumulator` | Extract organizational patterns | Auto-enabled with KnowledgeMound |
| Post-debate workflows | Automated processing after debates | `enable_post_debate_workflow=False` |

```python
from aragora.debate.arena_config import ArenaConfig

config = ArenaConfig(
    # Enable all cross-functional features
    enable_knowledge_retrieval=True,
    enable_cross_debate_memory=True,
    enable_coordinated_writes=True,
    enable_performance_feedback=True,
)
```

### New Prometheus Metrics

7 new metrics for monitoring cross-functional features:

- `aragora_knowledge_cache_hits_total` / `aragora_knowledge_cache_misses_total`
- `aragora_memory_coordinator_writes_total`
- `aragora_selection_feedback_adjustments_total`
- `aragora_workflow_triggers_total`
- `aragora_evidence_stored_total`
- `aragora_culture_patterns_total`

### New Prometheus Alerts

6 alerts for cross-functional feature health:

| Alert | Trigger |
|-------|---------|
| `LowKnowledgeCacheHitRate` | Cache hit rate < 50% for 15m |
| `MemoryCoordinatorFailures` | Failed atomic writes detected |
| `HighSelectionFeedbackVolatility` | >10 weight adjustments/sec for 10m |
| `WorkflowTriggerFailures` | >20% workflow failures for 10m |
| `NoEvidenceBeingStored` | No evidence stored despite 10+ debates/hr |
| `NoCulturePatternsExtracted` | No patterns extracted despite 20+ debates/6hr |

### Performance Benchmarks

MemoryCoordinator benchmarks with SLO verification:

| Operation | Performance |
|-----------|-------------|
| Metrics update | ~681ns (1.47M ops/sec) |
| Build operations | ~50μs (20K ops/sec) |
| Sequential write | SLO verified |
| Parallel write | SLO verified |

### API Updates

- OpenAPI spec regenerated with **2210 endpoints**
- 64 API tags for comprehensive documentation
- All new cross-functional endpoints documented

---

## Upgrade Guide

### From v2.0.2

No breaking changes. New features are opt-in via ArenaConfig flags.

```python
# Minimal upgrade - all new features use sensible defaults
arena = Arena.from_config(env, agents, protocol)

# Full cross-functional activation
arena = Arena.from_config(
    env, agents, protocol,
    config=ArenaConfig(
        enable_cross_debate_memory=True,
        enable_coordinated_writes=True,
        enable_performance_feedback=True,
    ),
)
```

### Configuration Reference

See [docs/CROSS_FUNCTIONAL_FEATURES.md](CROSS_FUNCTIONAL_FEATURES.md) for detailed configuration options.

---

## Known Issues

- Knowledge pipeline tests marked as xfail pending API alignment
- TypeScript SDK needs update for new endpoints

---

## Contributors

- Aragora Team
- Claude Opus 4.5 (Co-Author)

---
---

# v1.0.0 - Production Ready

**Release Date:** January 13, 2026
**Version:** 1.0.0
**Codename:** Production Ready

---

## Overview

Aragora 1.0 marks our first production-ready release. This version delivers enterprise-grade security, comprehensive TypeScript SDK support, and high-availability deployment capabilities. The release includes 22,209 tests across 507 test files, ensuring stability and reliability for production workloads.

---

## What's New

### Security Enhancements

#### Account Lockout Protection
Brute-force attack prevention with intelligent exponential backoff:
- **5 failed attempts**: 1-minute lockout
- **10 failed attempts**: 15-minute lockout
- **15+ failed attempts**: 1-hour lockout

Independent tracking by email AND IP address ensures attackers can't bypass by switching accounts or proxies.

```python
from aragora.auth.lockout import get_lockout_tracker

tracker = get_lockout_tracker()

# Check before login
if tracker.is_locked(email=email, ip=client_ip):
    remaining = tracker.get_remaining_time(email=email, ip=client_ip)
    return error(f"Account locked for {remaining} seconds")
```

#### Multi-Factor Authentication (MFA)
Full TOTP/HOTP support with backup codes:
- Setup flow with QR code generation
- 6-digit time-based codes (30-second validity)
- 10 backup recovery codes
- Admin-assisted unlock capability

### TypeScript SDK

Complete client library with 23 API namespaces:

| Namespace | Description |
|-----------|-------------|
| `auth` | Authentication, sessions, MFA |
| `debates` | Create and manage debates |
| `agents` | Agent configuration and status |
| `consensus` | Consensus tracking and proofs |
| `calibration` | Prediction accuracy (Brier scores) |
| `insights` | Post-debate pattern extraction |
| `beliefNetwork` | Probabilistic reasoning graphs |
| `crux` | Critical disagreement identification |
| `tournaments` | Competitive agent benchmarking |
| `gauntlet` | Adversarial stress testing |
| ... | And 13 more |

```typescript
import { AragoraClient, streamDebate } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: 'https://aragora.example.com',
  apiToken: process.env.ARAGORA_API_TOKEN,
});

// Start a debate
const debate = await client.debates.create({
  topic: 'Should we implement feature X?',
  agents: ['claude', 'gpt-4o', 'gemini-pro'],
  protocol: { rounds: 3, consensus: 'majority' },
});

// Stream events
const stream = streamDebate('https://aragora.example.com', debate.debate_id);
for await (const event of stream) {
  const eventLoopId = event.loop_id || event.data?.debate_id || event.data?.loop_id;
  if (eventLoopId && eventLoopId !== debate.debate_id) continue;
  console.log(event.type, event.data);
}
```

### High-Availability Deployment

Production-ready Kubernetes manifests:

- **Horizontal Pod Autoscaler (HPA)**: Auto-scale 2-10 pods based on CPU (70% threshold)
- **Pod Disruption Budget (PDB)**: Minimum 1 pod always available
- **Anti-affinity rules**: Spread across nodes and zones
- **Redis shared state**: Sessions, rate limits, lockouts across replicas

```bash
# Deploy HA configuration
kubectl apply -k deploy/kubernetes/

# Verify
kubectl -n aragora get hpa
kubectl -n aragora get pdb
```

### Load Testing CI

Automated performance validation on every merge to main:

- **k6 load tests** for API endpoints
- **WebSocket burst tests** for real-time streaming
- **SLO threshold enforcement**:
  - p50 latency < 200ms
  - p95 latency < 500ms
  - p99 latency < 2000ms
  - Error rate < 1%
  - Throughput > 50 RPS

### Database Optimizations

- **LRU caching** for consensus queries (5-min TTL, 500 entries)
- **Extracted modules**: VoteCollector, VoteWeighter for maintainability
- **Query optimization**: Indexed lookups for frequent operations

---

## API Changes

### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/auth/mfa/setup` | POST | Initialize MFA setup |
| `/api/v2/auth/mfa/enable` | POST | Enable MFA with verification |
| `/api/v2/auth/mfa/verify` | POST | Verify MFA code at login |
| `/api/v2/admin/users/{id}/unlock` | POST | Admin unlock locked account |
| `/api/v2/calibration/scores` | GET | Brier score leaderboard |
| `/api/v2/calibration/history/{agent}` | GET | Agent calibration history |
| `/api/v2/insights/extract` | POST | Extract patterns from debate |
| `/api/v2/consensus/proofs/{id}` | GET | Cryptographic consensus proof |

### Deprecated Endpoints (Sunset: July 2026)

| Old Endpoint | Replacement |
|--------------|-------------|
| `/api/debates` | `/api/v2/debates` |
| `/api/agents` | `/api/v2/agents` |
| `/api/health` | `/api/v2/health` |

All V1 endpoints return `Deprecation` and `Sunset` headers with migration guidance.

---

## Performance

Benchmarks on 4-core, 8GB RAM instance:

| Metric | Value |
|--------|-------|
| API p50 latency | 45ms |
| API p95 latency | 120ms |
| API p99 latency | 280ms |
| Max concurrent debates | 50+ |
| Max WebSocket connections | 1000+ |
| Memory per debate | ~50MB |

---

## Breaking Changes

1. **V1 API Deprecation**: All `/api/` endpoints without version prefix are deprecated. Use `/api/v2/` for new integrations.

2. **Agent Names**: Use canonical names (`anthropic-api`, `openai-api`) not aliases (`claude`, `codex`).

3. **Rate Limiting**: Enabled by default. Configure via `ARAGORA_RATE_LIMIT_*` environment variables.

4. **Redis Required for HA**: Multi-replica deployments require Redis for session/lockout state.

---

## Migration

See [deprecated/migrations/MIGRATION_0.8_to_1.0.md](deprecated/migrations/MIGRATION_0.8_to_1.0.md) for detailed upgrade instructions.

**Quick checklist:**
- [ ] Update API calls to use `/api/v2/` prefix
- [ ] Configure Redis for distributed deployments
- [ ] Set `ARAGORA_ENABLE_MFA=true` if using MFA
- [ ] Update SDK to `@aragora/sdk@1.0.0`
- [ ] Review rate limit configuration

---

## Known Issues

1. **MFA Recovery**: If user loses device and all backup codes, admin must manually reset MFA via database.

2. **WebSocket Reconnection**: Occasional connection drops under high load (>500 concurrent). Automatic reconnection handles this.

3. **Large Debate Memory**: Debates with 100+ rounds may consume significant memory. Use `max_rounds` limit.

---

## Contributors

Thanks to all contributors who made 1.0 possible. Special recognition to the test automation improvements that brought coverage to 22,209 tests.

---

## What's Next (1.1 Roadmap)

- **Multi-region deployment** support
- **GraphQL API** alongside REST
- **Advanced consensus mechanisms** (stake-weighted, reputation)
- **Plugin marketplace** for community extensions
- **Real-time collaboration** features

---

## Support

- Documentation: https://aragora.ai/docs
- Issues: https://github.com/an0mium/aragora/issues
- Discussions: https://github.com/an0mium/aragora/discussions
