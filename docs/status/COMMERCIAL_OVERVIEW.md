# Aragora -- The Decision Integrity Platform

*Version 2.6.3 | Commercial Overview*
*Status: Internal snapshot; metrics are directional unless sourced in docs/STATUS.md.*

## Executive Summary

Aragora is the **Decision Integrity Platform** -- orchestrating 42+ agent types to adversarially vet decisions against your organization's knowledge, then delivering audit-ready decision receipts to any channel.

**You don't just get an answer. You get a defensible decision trail.**

Unlike chatbots, Aragora builds institutional memory with full audit trails. Vetted decisionmaking is the engine. The product is a defensible decision record.

---

## Five Pillars

Aragora is built on five architectural commitments that together produce something no single-model tool can offer.

| Pillar | What It Means |
|--------|---------------|
| **1. SMB-Ready, Enterprise-Grade** | Useful to a 5-person startup on day one; scales to regulated enterprise without rearchitecting. Security and compliance built in, not bolted on. |
| **2. Leading-Edge Memory and Context** | 4-tier Continuum Memory, Knowledge Mound (28 registered adapters), and RLM context compression enable coherence across long multi-round sessions and large document sets. |
| **3. Extensible and Modular** | Connectors, SDKs (Python + TypeScript, 140 namespaces), 2,000+ API operations, OpenClaw integration, workflow engine, marketplace. |
| **4. Multi-Agent Robustness** | Heterogeneous agents (Claude, GPT, Gemini, Grok, Mistral, DeepSeek, Qwen, Kimi) produce outputs more robust, less biased, and higher quality than single models. |
| **5. Self-Healing and Self-Extending** | Nomic Loop autonomous improvement, red-team stress-testing, multi-agent code editing with human approval gates. |

---

## What Aragora Does

| Capability | Description | Business Value |
|------------|-------------|----------------|
| **Omnivorous Input** | Ingest from documents, APIs, databases, web, voice | Single platform for all information sources |
| **Multi-Channel Access** | Query via web, Slack, Telegram, WhatsApp, API | Meet users where they already work |
| **Multi-Agent Consensus** | 42+ heterogeneous agent types debate to conclusions | Diverse perspectives, defensible decisions |
| **Bidirectional Dialogue** | Ask follow-ups, refine questions, drill into details | Interactive human-AI collaboration |
| **Evidence Trails** | Cryptographic audit chains with provenance tracking | Compliance-ready documentation |
| **Learn and Improve** | 4-tier memory with cross-session pattern learning | Continuously improving accuracy |

---

## Core Value Proposition

### For Engineering Leaders
- **Reduce review bottlenecks**: AI agents provide first-pass critique 24/7
- **Catch blind spots**: Different AI models notice different issues
- **Accelerate decisions**: Get multi-perspective analysis in minutes, not days

### For Compliance Officers
- **Audit trails**: Every debate produces a cryptographic receipt
- **Dissent tracking**: Know what was contested and why
- **Provenance chains**: Trace every claim to its source

### For Security Teams
- **Adversarial testing**: Built-in red team mode attacks your specs
- **Gauntlet mode**: Systematic stress-testing with risk heatmaps
- **Pattern learning**: System remembers past vulnerabilities

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ARAGORA PLATFORM                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    AGENT LAYER (42+ Agent Types)                  │   │
│  │  Claude │ GPT │ Gemini │ Grok │ Mistral │ DeepSeek │ Qwen │ Kimi │   │
│  │                     + Local Models (Ollama, LM Studio)            │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DEBATE ENGINE                                  │   │
│  │  • 9-round structured protocol (Propose → Critique → Synthesize)  │   │
│  │  • Graph debates with branching │ Matrix debates for scenarios    │   │
│  │  • Consensus detection │ Convergence analysis │ Forking support   │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    KNOWLEDGE LAYER                                │   │
│  │  • Belief networks with Bayesian propagation                      │   │
│  │  • Claims kernel with typed relationships                         │   │
│  │  • Evidence provenance with hash chains                           │   │
│  │  • Citation tracking and reliability scoring                      │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY SYSTEM (4 tiers)                        │   │
│  │  Fast → Medium → Slow → Glacial                                   │   │
│  │  • Surprise-based learning │ Consolidation scoring                │   │
│  │  • Cross-session pattern extraction                               │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    OUTPUT LAYER                                   │   │
│  │  Decision Receipts │ Risk Heatmaps │ Dissent Trails │ Proofs     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Commercial Readiness Assessment

### Overall: 90% Production Ready (internal estimate)

| Category | Score | Status | Notes |
|----------|-------|--------|-------|
| Error Handling & Resilience | 95% | Ready | Circuit breakers, retry policies, graceful degradation |
| Security & Authentication | 92% | Ready | OIDC/SAML, MFA, encryption, audit logging |
| Scalability & Performance | 92% | Ready | Connection pooling, caching, rate limiting |
| Observability & Monitoring | 90% | Ready | Prometheus, Grafana, OpenTelemetry |
| Testing & QA | 95% | Ready | 130,000+ tests across 3,000+ files |
| Documentation | 91% | Ready | API docs, runbooks, compliance guides |
| Compliance & Governance | 88% | Ready | RBAC v2 with 50+ permissions, role hierarchy |
| SDK & Integrations | 90% | Ready | 140 TypeScript namespaces, 8 bot handlers |
| **OVERALL** | **90%** | **SMB Ready** | Enterprise-grade features integrated |

### Deployment Readiness

- Docker container with non-root user
- Kubernetes manifests in `/deploy/kubernetes/`
- Health checks and readiness probes
- Prometheus metrics endpoint
- Grafana dashboards included

---

## Key Differentiators

### 1. Heterogeneous Agent Orchestration
Unlike single-model solutions, Aragora runs debates across 42+ agent types/providers. Different models catch different issues—Claude excels at reasoning, GPT at breadth, Gemini at design, Grok at lateral thinking.

### 2. Audit-Ready Output
Every debate produces a **Decision Receipt** with:
- Cryptographic hash chain
- Evidence provenance
- Dissent tracking
- Timestamp verification

### 3. Adversarial Testing Built-In
**Gauntlet Mode** provides systematic stress-testing:
- Security red-team attacks
- Devil's advocate logic testing
- Scaling critic analysis
- Compliance verification (GDPR, HIPAA, SOC 2, AI Act)

### 4. Learning Memory System
The 4-tier **Continuum Memory** enables:
- Pattern extraction from successful critiques
- Cross-session learning
- Institutional knowledge accumulation
- Surprise-based prioritization

### 5. Enterprise-Grade Security
- OIDC/SAML SSO integration
- MFA support (TOTP/HOTP)
- AES-256-GCM encryption at rest
- Multi-tenant isolation with quotas

---

## Use Cases

### Specification Review
```bash
aragora gauntlet spec.md --profile thorough --output receipt.html
```
Stress-test API specifications, architecture documents, and technical designs before implementation.

### Compliance Audit
```bash
aragora gauntlet policy.yaml --input-type policy --persona gdpr
```
Automated compliance checking against GDPR, HIPAA, SOC 2, and AI Act requirements.

### Code Review
```bash
git diff main | aragora review
```
AI red-team review of pull requests with unanimous consensus highlighting.

### Decision Validation
```python
from aragora import Arena, Environment, DebateProtocol

env = Environment(task="Should we adopt microservices?")
protocol = DebateProtocol(rounds=5, consensus="majority")
arena = Arena(env, agents, protocol)
result = await arena.run()
```
Structured debate for strategic decisions with evidence-based recommendations.

---

## Platform Statistics

| Metric | Source |
|--------|--------|
| Codebase size & tests | See `docs/STATUS.md` |
| Agent catalog | `AGENTS.md` |
| Connector catalog | `docs/CONNECTORS.md` |
| Memory tiers | 4 (see `docs/MEMORY_TIERS.md`) |

---

## Deployment Options

### Self-Hosted
- Docker Compose for single-node deployment
- Kubernetes for scale-out deployment
- Supports SQLite (dev) or PostgreSQL (prod)

### Cloud
- AWS Lightsail (current production)
- Any Kubernetes-compatible cloud (AWS EKS, GCP GKE, Azure AKS)
- Cloudflare Tunnel for secure ingress

### Hybrid
- On-premises control plane
- Cloud-based agent APIs
- Air-gapped deployment support

---

## Integration Points

### Chat Platforms
- Slack (bot + connector)
- Discord (bot + connector)
- Microsoft Teams (bot + connector)
- Google Chat (connector)

### Data Sources
- GitHub, GitLab
- SharePoint, Confluence, Notion
- ArXiv, Wikipedia, news APIs
- Healthcare systems (HL7/FHIR)
- SEC filings, legal databases

### Observability
- Prometheus metrics export
- Grafana dashboards included
- OpenTelemetry tracing
- SIEM integration

---

## Enterprise Readiness

| Capability | Resolution | Status |
|------------|------------|--------|
| Fine-grained RBAC | RBAC v2 with 7 roles, 50+ permissions | Complete |
| Automated backups | BackupManager with incremental support | Complete |
| Bot handler consolidation | BotHandlerMixin across 8 platforms | Complete |
| TypeScript SDK | 140 namespaces wired to client | Complete |
| OpenClaw integration | Portable agent governance | Complete |
| Knowledge Mound Phase A2 | Contradiction detection, confidence decay, RBAC governance | Complete |
| SLA documentation | Legally-binding service levels | In Progress |
| Distributed rate limiting | Redis-backed cluster-aware limiting | In Progress |

---

## Pricing Considerations

*Placeholder for commercial discussion*

### Potential Models
1. **Per-seat licensing** - Based on user count
2. **Usage-based** - Per debate/API call
3. **Tier-based** - SMB / Enterprise / Enterprise+
4. **Hybrid** - Base license + usage overage

### Cost Factors
- AI provider API costs (passed through or absorbed)
- Compute and storage
- Support tier (community, business, enterprise)
- Compliance certifications

---

## Getting Started

### Quick Start (5 minutes)
```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
export ANTHROPIC_API_KEY=your-key
aragora ask "Design a rate limiter" --agents anthropic-api,openai-api
```

### Production Deployment
See [PRODUCTION_READINESS.md](../deployment/PRODUCTION_READINESS.md) for the complete checklist.

### API Integration
See [SDK_GUIDE.md](../SDK_GUIDE.md) for the Python SDK reference.

---

## Contact

- **Domain**: [aragora.ai](https://aragora.ai)
- **Documentation**: [docs/](.)
- **API Reference**: [API_REFERENCE.md](../api/API_REFERENCE.md)

---

*Document generated from comprehensive codebase exploration. Feature counts verified against actual module inventory (February 2026).*
