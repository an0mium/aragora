# Aragora Documentation Hub

Quick navigation for the Aragora platform documentation.

## What Are You Trying To Do?

### Getting Started
| Goal | Document |
|------|----------|
| First time setup | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Quick API test | [API_QUICK_START.md](API_QUICK_START.md) |
| Run a debate | [API_EXAMPLES.md](API_EXAMPLES.md) |

### Build an Integration
| Goal | Document |
|------|----------|
| Python SDK | [SDK_GUIDE.md](SDK_GUIDE.md) |
| TypeScript SDK | [SDK_TYPESCRIPT.md](SDK_TYPESCRIPT.md) |
| WebSocket streaming | [WEBSOCKET_EVENTS.md](WEBSOCKET_EVENTS.md) |
| Full API reference | [API_REFERENCE.md](API_REFERENCE.md) |

### Deploy to Production
| Goal | Document |
|------|----------|
| Deployment guide | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Docker setup | [deployment/DOCKER.md](deployment/DOCKER.md) |
| Production readiness | [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) |
| CI/CD security | [CI_CD_SECURITY.md](CI_CD_SECURITY.md) |

### Troubleshoot Issues
| Goal | Document |
|------|----------|
| General troubleshooting | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Alert runbooks | [ALERT_RUNBOOKS.md](ALERT_RUNBOOKS.md) |
| Operations guide | [OPERATIONS.md](OPERATIONS.md) |

---

## Core System Documentation

### Debate Engine
The core multi-agent debate orchestration system.

```
ARCHITECTURE.md          # Overall system design
    └── DEBATE_INTERNALS.md  # How debates work
        └── CONSENSUS.md     # Consensus mechanisms
```

### Memory Systems
Multi-tier memory for cross-debate learning.

```
MEMORY.md                # Overview of memory architecture
├── MEMORY_TIERS.md      # Fast/Medium/Slow/Glacial tiers
├── MEMORY_STRATEGY.md   # Consolidation and decay strategies
└── MEMORY_ANALYTICS.md  # Memory usage metrics
```

**Key Concepts:**
- **Fast tier** (1 min TTL): Immediate context within a debate
- **Medium tier** (1 hour): Session-level memory
- **Slow tier** (1 day): Cross-session patterns
- **Glacial tier** (1 week): Long-term institutional knowledge

### Knowledge Mound
Centralized knowledge storage with validation and retrieval.

```
KNOWLEDGE_MOUND.md            # Architecture and concepts
└── KNOWLEDGE_MOUND_OPERATIONS.md  # Operational procedures
```

**Key Features:**
- 14 adapters (Continuum, Consensus, Evidence, ELO, etc.)
- Semantic search with validation feedback
- RBAC-governed access
- Contradiction detection

### Control Plane
Enterprise orchestration layer for multi-agent systems.

```
CONTROL_PLANE.md         # Architecture overview
└── CONTROL_PLANE_GUIDE.md   # Operational guide
```

**Features:** Agent registry, task scheduler, policy governance, health monitoring

---

## Integration Channels

| Channel | Doc | Description |
|---------|-----|-------------|
| Slack | [BOT_INTEGRATIONS.md](BOT_INTEGRATIONS.md) | Slack bot setup |
| Telegram | [CHAT_CONNECTOR_GUIDE.md](CHAT_CONNECTOR_GUIDE.md) | Telegram integration |
| WhatsApp | [CHAT_CONNECTOR_GUIDE.md](CHAT_CONNECTOR_GUIDE.md) | WhatsApp Business |
| Email | [EMAIL_PRIORITIZATION.md](EMAIL_PRIORITIZATION.md) | Email processing |
| GitHub | [GITHUB_PR_REVIEW.md](GITHUB_PR_REVIEW.md) | PR review automation |

---

## Enterprise Features

| Feature | Doc | Description |
|---------|-----|-------------|
| Authentication | [AUTH_GUIDE.md](AUTH_GUIDE.md) | SSO, MFA, API keys |
| RBAC | [GOVERNANCE.md](GOVERNANCE.md) | Role-based access control |
| Compliance | [COMPLIANCE.md](COMPLIANCE.md) | SOC 2, GDPR |
| Billing | [BILLING.md](BILLING.md) | Usage metering and costs |
| Security | [SECURITY.md](SECURITY.md) | Security architecture |

---

## API Quick Reference

| Endpoint Pattern | Purpose | Doc Section |
|-----------------|---------|-------------|
| `POST /api/v1/debates` | Start debate | [API_EXAMPLES.md#debates](API_EXAMPLES.md) |
| `GET /ws/debate/{id}` | Stream events | [WEBSOCKET_EVENTS.md](WEBSOCKET_EVENTS.md) |
| `POST /api/v1/documents` | Ingest documents | [DOCUMENTS.md](DOCUMENTS.md) |
| `GET /api/v1/knowledge/search` | Search knowledge | [KNOWLEDGE_MOUND.md](KNOWLEDGE_MOUND.md) |

---

## CLI Reference

```bash
aragora ask "Question"      # Run a debate
aragora serve               # Start server
aragora setup               # Interactive setup wizard
aragora backup create       # Create database backup
aragora backup restore      # Restore from backup
```

See [CLI_REFERENCE.md](CLI_REFERENCE.md) for full documentation.

---

## Document Index

For the complete document listing, see [INDEX.md](INDEX.md).

For deprecated documentation, see [deprecated/](deprecated/).
