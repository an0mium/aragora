# Aragora 1.0 Release Notes

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
import { AragoraClient } from '@aragora/sdk';

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
client.debates.subscribe(debate.id, (event) => {
  console.log(event.type, event.data);
});
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

See [MIGRATION_0.8_to_1.0.md](MIGRATION_0.8_to_1.0.md) for detailed upgrade instructions.

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
