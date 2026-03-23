# Aragora Product Roadmap

**Last Updated:** March 23, 2026

---

## Current Status (March 2026)

Aragora has shipped most of the closed-loop backbone (CLB) infrastructure and completed the 14/14 issue sprint. The March 21-23 merge stream materially improved the product loop on `main`, and current `main` now includes a mocked founder-loop E2E proof plus the tightened quickstart contract. Launch readiness is still gated first by **live** PMF closure, not by enterprise certification.

**By the numbers:**
- 3,846 Python files under `aragora/`
- 5,174 test files under `tests/`
- 674 Markdown docs under `docs/`
- 42 Knowledge Mound adapter specs registered
- 43 agent types available across CLI, API, local, and proxy providers
- SOC 2 controls framework: 98% implemented

**Completed since January 2026:**
- Nomic Loop end-to-end self-improvement (66 E2E tests passing)
- Knowledge Mound Phase A2 (contradiction detection, confidence decay, RBAC governance, analytics)
- Unified Memory Gateway (MemoryGateway + RetentionGate + CrossSystemDedupEngine + RLMMemoryNavigator + ClaudeMemAdapter)
- Pipeline orchestration — 4-stage Idea-to-Execution (Ideas → Goals → Workflows → Orchestration)
- Compliance CLI (`aragora compliance export`) for EU AI Act artifact bundles
- Settlement hooks with cryptographic Gauntlet receipts
- Voice/TTS integration wired end-to-end (STT + TTS)
- Multi-tenancy isolation, resource quotas, and usage metering
- RBAC v2 fully integrated across all KM adapters
- Closed-loop backbone sprint (14 CLB issues: canonical contracts, spec gates, deliberation handoff, execution bundles, verification bundles, receipt normalization, outcome feedback, trust-tier propagation, external verifier, golden-path test, dogfood profile)
- PR watch daemon fleet (3 Mac machines, 30 autonomous reviews/hour)
- Dev swarm coordination layer
- LLM-powered scope validation and blocker classification (6 async methods with fail-closed semantics)
- Ralph observability dashboard (7 Prometheus metrics, 10 REST API endpoints, YAML state data service)
- File-scope propagation fix for swarm work orders (#884)
- LLM-first vague-goal expansion replacing keyword templates (#888)
- Live settings/API-key tab wiring to backend auth endpoints ([#1146](https://github.com/synaptent/aragora/pull/1146))
- Live debate creation from the debates page ([#1147](https://github.com/synaptent/aragora/pull/1147))
- Truthful partial-public status surface ([#1148](https://github.com/synaptent/aragora/pull/1148))
- Refresh-aware pipeline feedback scoping ([#1149](https://github.com/synaptent/aragora/pull/1149))
- Visible pipeline golden-path summary ([#1150](https://github.com/synaptent/aragora/pull/1150))
- `PipelineKMBridge` precedent loading before debate ([#1151](https://github.com/synaptent/aragora/pull/1151))
- Queue max-parallel-two safety slice ([#1141](https://github.com/synaptent/aragora/pull/1141))
- Queue harvest command ([#1164](https://github.com/synaptent/aragora/pull/1164))

**Remaining tracked priority work:**
- Prove the canonical founder loop live on current `main`: readiness -> live debate -> receipt -> visible result -> KM ingestion
- Turn any live founder-loop failure into a bounded PMF blocker with an exact command transcript and truthful stop condition
- Use `aragora pipeline dogfood` and bounded swarm/nomic lanes only on those founder-loop blockers
- Re-run the founder loop after each landed blocker tranche until it is repeatable without manual rescue
- Dogfood the inbox trust wedge only after the founder loop holds
- Keep pentest / SOC 2 preparation warm, but after PMF proof ([#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), [#509](https://github.com/synaptent/aragora/issues/509))

**EU AI Act enforcement date: August 2, 2026** — the compliance CLI and audit trail infrastructure
position Aragora as a natural adoption path for enterprises facing this deadline.

---

## Vision

Aragora is the control plane for multi-agent vetted decisionmaking across organizational knowledge and channels. We orchestrate heterogeneous AI agents to debate, synthesize, and deliver defensible decisions through structured vetted decisionmaking—building institutional memory with full audit trails.

---

## Current Capabilities (v2.8)

### Core Platform
- Multi-agent debate orchestration with configurable protocols
- 43 agent types across 10+ providers (Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, etc.)
- Real-time WebSocket streaming of debate progress
- Consensus detection with formal verification proofs
- ELO-based agent skill tracking and team selection

### Knowledge & Memory
- Knowledge Mound for organizational knowledge accumulation
- Continuum Memory with 4-tier retention (fast/medium/slow/glacial)
- Evidence collection from 11+ sources (ArXiv, GitHub, Wikipedia, etc.)
- Cross-debate learning and pattern recognition

### Enterprise Features
- Multi-tenant workspaces with RBAC
- OIDC/SAML authentication
- Audit logging and compliance reporting
- Control Plane for multi-instance orchestration
- Workflow engine for complex debate pipelines

### Integrations
- Slack, Discord, Microsoft Teams bots
- Email-to-debate routing
- REST API with 3,100+ operations across 2,600+ paths
- WebSocket real-time API
- MCP server for Claude Desktop

---

## Q1-Q2 2026: SME & Developer Focus

> **Backlog:** See `docs/FEATURE_GAP_LIST.md` for current backlog.

### Track 1: SME Starter Pack
- [x] Slack integration (OAuth, slash commands, thread debates)
- [x] Microsoft Teams integration (Bot Framework, Adaptive Cards)
- [x] Decision Receipts v1 (cryptographic signatures, PDF/JSON/HTML export)
- [x] Budget controls and cost tracking per debate
- [x] Usage dashboard with spend analytics

### Track 2: Developer Platform
- [x] OpenAPI 3.1 specification (3,100+ operations)
- [x] TypeScript SDK feature parity with Python
- [x] SDK code generation pipeline
- [x] Backend runtime entrypoint and compatibility policy documented (ADR-017)
- [ ] Interactive API explorer at docs.aragora.ai/api
- [ ] Example apps (Slack code review, document analysis)

### Track 3: Self-Hosted Deployment
- [x] Docker Compose production stack
- [ ] Guided setup CLI (`aragora setup`)
- [x] Minimal dependency mode (SQLite + in-memory)
- [x] Backup & restore CLI
- [x] Helm chart for Kubernetes

### Enterprise Readiness (After PMF)
- [ ] Complete third-party penetration testing after provider routing, one working user journey, and KM retrieval are live end-to-end
- [ ] Deploy public status page at status.aragora.ai
- [x] Implement quarterly disaster recovery drills (BackupScheduler with DR integration)
- [x] Finalize data classification policy (runtime enforcement, CI PII gate, evidence bundles)
- [x] MFA enforcement for admin access (TOTP/HOTP)
- [x] Enhanced circuit breaker coverage for all connectors
- [x] Redis Sentinel/Cluster support (RedisHAClient)
- [ ] 99.9% uptime target with public SLA

---

## Q3 2026: Scale & Performance

### Performance Optimization
- [ ] Debate execution time reduction (target: 50%)
- [ ] Streaming response improvements
- [ ] Efficient batch debate processing
- [ ] Memory optimization for large knowledge bases

### Horizontal Scaling
- [ ] Kubernetes Operator for automated scaling
- [ ] Global edge deployment
- [ ] Debate sharding for high-throughput workloads
- [x] Redis Cluster mode support

### Cost Optimization
- [ ] Smart provider routing based on cost/quality
- [ ] Token usage analytics dashboard
- [x] Budget controls and alerts
- [ ] Cached response optimization

---

## Q4 2026: Platform Ecosystem

### Marketplace
- [ ] Agent marketplace for sharing custom agents
- [x] Workflow template library (50+ pre-built templates across 6 categories)
- [ ] Integration connectors from community
- [ ] Revenue sharing for creators

### Extended Integrations
- [x] Zapier / Make.com connectors
- [x] GitHub Actions for CI/CD debates (`aragora-review-gate.yml` shipped)
- [ ] Jupyter notebook integration
- [ ] VS Code extension

### Analytics & Insights
- [ ] Debate outcome analytics dashboard
- [ ] Agent performance benchmarking
- [ ] Knowledge gap identification
- [ ] ROI measurement tools

---

## 2027 Vision

### Autonomous Agents
- [x] Self-improving debate protocols (Nomic Loop, operational as of Q1 2026)
- [x] Autonomous knowledge acquisition (Knowledge Mound Phase A2)
- [ ] Proactive insight generation
- [x] Human-in-the-loop governance (approval gates in self-improvement pipeline)

### Industry Solutions
- [ ] Legal document review suite (vertical package planned Q3 2026)
- [ ] Medical diagnosis support (Healthcare FHIR vertical planned Q3 2026)
- [ ] Financial analysis platform (Financial SOX vertical planned Q3 2026)
- [ ] Research acceleration tools

### Platform Capabilities
- [ ] 1M+ concurrent debates
- [ ] Sub-second debate initiation
- [ ] 99.99% availability
- [ ] Global compliance (HIPAA, FedRAMP)

### 2027 R&D
- Prover-Estimator debate protocol
- Canvas GUI 8-stage visual pipeline
- Market resolution mechanism for long-horizon settlement
- ERC-8004 on-chain agent identity deployment to mainnet (contracts written, pending deployment)
- Decision-Integrity UI Workbench
- OpenClaw E2E demo and production integration
- Cloud marketplace listings (AWS Marketplace, Azure Marketplace)

---

## Q2-Q4 2026 Forward Plan

This section captures the prioritized forward roadmap as of March 2026, organized by quarter and theme.
Execution priority source of truth: [docs/status/NEXT_STEPS_CANONICAL.md](docs/status/NEXT_STEPS_CANONICAL.md). This roadmap summarizes quarter-level themes and does not supersede canonical execution priorities.

The March 2026 product cohesion assessment found ~25% effective feature completeness for actual use, no complete user journey, provider routing not yet wired to runtime selection, Knowledge Mound retrieval not yet enriching default debates, and a shell-heavy frontend surface. The March 21-23 merge stream changed those specific facts on `main`, but it did not eliminate the need for live PMF proof. The near-term roadmap still prioritizes proving the founder loop before widening enterprise-readiness work.

**EU AI Act enforcement: August 2, 2026.** This remains a real forcing function, but the compliance package only matters commercially if the core PMF loop is usable enough to demo and adopt.

### Q2 2026 Priorities
- [ ] Prove the live founder loop using the runbook in [docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md](docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md)
- [ ] Reconstitute the PMF blocker queue from observed founder-loop failures instead of stale March umbrella issues
- [ ] Use the idea-to-execution pipeline and swarm queue to clear only those PMF blockers
- [ ] Productize the founder-loop result surfaces so the live outcome is understandable without operator context
- [ ] Dogfood the second workflow (inbox trust wedge) after the founder loop is repeatable
- [x] Agent-first beta: OpenClaw fleet deployed on 3 machines, running `aragora review` on real PRs via REST API
- [x] GitHub Actions pre-merge gate (`aragora-review-gate.yml` shipped)
- [x] Public demo at aragora.ai/demo (PR #705; standalone demo page live)
- [ ] EU AI Act compliance package — keep packaging warm without displacing PMF loop closure

### Q3 2026 Priorities
- [ ] 10+ agent coordination at enterprise scale ([#815](https://github.com/synaptent/aragora/issues/815))
- [ ] Pentest / SOC 2 engagement kickoff after PMF proof ([#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), [#509](https://github.com/synaptent/aragora/issues/509))
- [ ] ERC-8004 on-chain deployment ([#816](https://github.com/synaptent/aragora/issues/816))
- [ ] Cloud marketplace listings: AWS Marketplace and Azure Marketplace
- [ ] Vertical packages: Healthcare (FHIR/HIPAA), Financial Services (SOX/audit), Legal
- [ ] Skills marketplace pilot (community agent templates)
- [ ] Kubernetes Operator for automated horizontal scaling

### Q4 2026 Priorities
- [ ] Cross-organization federation foundation
- [ ] Decision-Integrity UI Workbench (visual debate canvas)

### 2027 Horizon
- Prover-Estimator debate protocol
- Canvas GUI 8-stage visual pipeline
- Market resolution mechanism (long-horizon settlement)
- ERC-8004 on-chain deployment to mainnet

---

## Feature Requests

We actively track feature requests from customers. Top requested features:

| Feature | Votes | Status |
|---------|-------|--------|
| Dark mode for live dashboard | 89 | **Shipped v2.1** |
| Mobile app | 67 | Under consideration |
| Offline debate mode | 45 | Researching |
| Voice input for debates | 38 | **Shipped Q1 2026** |
| Debate replay/rewind | 34 | Planned Q2 |

Submit feature requests: https://github.com/aragora/aragora/discussions

---

## Release Cadence

| Release Type | Frequency | Notes |
|--------------|-----------|-------|
| Patch (x.x.X) | Weekly | Bug fixes, security patches |
| Minor (x.X.0) | Monthly | New features, improvements |
| Major (X.0.0) | Quarterly | Breaking changes (with migration guides) |

---

## Contributing

Aragora is open to contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority contribution areas:
- Evidence connectors for new sources
- Language translations
- Documentation improvements
- Test coverage expansion

---

## Contact

- **Product Feedback**: product@aragora.ai
- **Enterprise Sales**: sales@aragora.ai
- **Security Issues**: security@aragora.ai
- **General Support**: support@aragora.ai

---

*This roadmap represents our current plans and is subject to change based on customer feedback and market conditions.*
