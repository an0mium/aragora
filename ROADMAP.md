# Aragora Product Roadmap

**Last Updated:** January 2026
**Current Version:** 2.4.0

---

## Vision

Aragora is the control plane for multi-agent vetted decisionmaking across organizational knowledge and channels. We orchestrate heterogeneous AI agents to debate, synthesize, and deliver defensible decisions through structured vetted decisionmaking—building institutional memory with full audit trails.

---

## Current Capabilities (v2.1)

### Core Platform
- Multi-agent debate orchestration with configurable protocols
- 15+ AI provider integrations (Anthropic, OpenAI, Google, xAI, DeepSeek, etc.)
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
- REST API with 275+ endpoints
- WebSocket real-time API
- MCP server for Claude Desktop

---

## Q1-Q2 2026: SME & Developer Focus

> **Detailed Backlog:** See [docs/BACKLOG_Q1_Q2_2026.md](docs/BACKLOG_Q1_Q2_2026.md) for 36 issues across 8 sprints (16 weeks).
> **GitHub Issues:** Import from [docs/BACKLOG_ISSUES.csv](docs/BACKLOG_ISSUES.csv)

### Track 1: SME Starter Pack
- [ ] Slack integration (OAuth, slash commands, thread debates)
- [ ] Microsoft Teams integration (Bot Framework, Adaptive Cards)
- [ ] Decision Receipts v1 (cryptographic signatures, PDF/JSON export)
- [ ] Budget controls and cost tracking per debate
- [ ] Usage dashboard with spend analytics

### Track 2: Developer Platform
- [ ] OpenAPI 3.1 specification (275+ endpoints)
- [ ] TypeScript SDK feature parity with Python
- [ ] SDK code generation pipeline
- [ ] Interactive API explorer at docs.aragora.ai/api
- [ ] Example apps (Slack code review, document analysis)

### Track 3: Self-Hosted Deployment
- [ ] Docker Compose production stack
- [ ] Guided setup CLI (`aragora setup`)
- [ ] Minimal dependency mode (SQLite + in-memory)
- [ ] Backup & restore CLI
- [ ] Helm chart for Kubernetes (stretch)

### Enterprise Readiness (Ongoing)
- [ ] Complete third-party penetration testing
- [ ] Deploy public status page at status.aragora.ai
- [x] Implement quarterly disaster recovery drills (BackupScheduler with DR integration)
- [ ] Finalize data classification policy
- [ ] MFA enforcement for admin access
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
- [x] Redis Cluster mode support (v2.1.14)

### Cost Optimization
- [ ] Smart provider routing based on cost/quality
- [ ] Token usage analytics dashboard
- [ ] Budget controls and alerts
- [ ] Cached response optimization

---

## Q4 2026: Platform Ecosystem

### Marketplace
- [ ] Agent marketplace for sharing custom agents
- [ ] Workflow template library
- [ ] Integration connectors from community
- [ ] Revenue sharing for creators

### Extended Integrations
- [ ] Zapier / Make.com connectors
- [ ] GitHub Actions for CI/CD debates
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
- Self-improving debate protocols
- Autonomous knowledge acquisition
- Proactive insight generation
- Human-in-the-loop governance

### Industry Solutions
- Legal document review suite
- Medical diagnosis support
- Financial analysis platform
- Research acceleration tools

### Platform Capabilities
- 1M+ concurrent debates
- Sub-second debate initiation
- 99.99% availability
- Global compliance (HIPAA, FedRAMP)

---

## Feature Requests

We actively track feature requests from customers. Top requested features:

| Feature | Votes | Status |
|---------|-------|--------|
| Dark mode for live dashboard | 89 | **Shipped v2.1** |
| Mobile app | 67 | Under consideration |
| Offline debate mode | 45 | Researching |
| Voice input for debates | 38 | Planned Q3 |
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

Aragora is open to contributions. See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

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
