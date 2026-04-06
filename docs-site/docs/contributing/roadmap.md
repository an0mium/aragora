---
title: Aragora Product Roadmap
description: Aragora Product Roadmap
---

# Aragora Product Roadmap

**Last Updated:** March 25, 2026

---

## Vision

Aragora is the **Decision Integrity Platform** — the control plane for multi-agent vetted decision-making across organizational knowledge and channels. We orchestrate heterogeneous AI agents to debate, synthesize, and deliver defensible decisions with full audit trails and cryptographic receipts.

The long-term goal: every consequential AI-assisted decision in an organization flows through Aragora, producing an inspectable, verifiable record of what was decided, by whom, with what evidence, and with what dissent.

---

## Operating Principle: Build the Full Vision, Make Everything Work

This roadmap is not a stop-doing list. The platform has massive infrastructure already built — 3,800+ Python modules, 216,000+ tests, 42 Knowledge Mound adapters, 43 agent types, 4-stage pipeline, swarm orchestration, compliance framework, and more.

**The priority is:**
1. Make everything that exists work correctly and reliably
2. Wire disconnected subsystems together into coherent user journeys
3. Improve performance, quality, and polish across the board
4. Build out the full vision using the autonomous pipeline infrastructure

**The autonomous execution system (overnight queue, boss loop, Nomic loop) is proven and should be used continuously** to improve the platform. Every code change should make the demo better, the debates faster, the receipts clearer, or the integrations more reliable.

---

## Current Status (March 2026)

The live founder loop is **proven repeatable** (5/5 consecutive runs, 35-62s). Production is live at api.aragora.ai. The demo surface works. 41 founder strategy docs were produced autonomously overnight and merged.

**What works end-to-end today:**
- Multi-agent debates with real LLM providers (Claude, GPT-4, Gemini, Mistral, Grok)
- Cryptographic decision receipts with SHA-256 audit trails
- Production API serving live debates at api.aragora.ai
- Demo surface at aragora.ai/demo with real backend
- Prompt-to-spec engine (`aragora spec`) in ~23s
- Inbox trust wedge CLI (`aragora triage`) with staged performance profile
- Autonomous overnight queue producing real strategy artifacts
- Deploy pipeline auto-deploying on push to main
- 216,000+ tests across 4,700+ test files

---

## Continuous Improvement Tracks

These tracks run in parallel, driven by the autonomous pipeline. Each produces measurable improvements that can be verified by running the product.

### Track 1: Performance & Speed
Make every user-facing flow faster.

- [ ] Debate execution time: target sub-20s for quickstart (currently 35-62s)
- [ ] Triage per-email time: target sub-10s for fast tier (currently ~18s)
- [ ] Prompt-to-spec latency: target sub-15s (currently ~23s)
- [ ] Streaming response improvements for live debate views
- [ ] Batch debate processing for inbox triage at scale
- [ ] Provider routing optimization (select fastest available model)
- [ ] Reduce unnecessary sidecars (research, KM retrieval when empty)

### Track 2: Code Correctness & Reliability
Make everything that exists work correctly.

- [ ] Full test suite health pass (find and fix regressions from recent 50+ PR sprint)
- [ ] Provider fallback reliability (Anthropic → OpenRouter → local)
- [ ] Async/sync boundary correctness (PostgreSQL, asyncpg, event loop issues)
- [ ] Resource cleanup (unclosed sockets, transport warnings)
- [ ] Error message quality (user-facing errors should be helpful, not stack traces)
- [ ] Receipt integrity verification across all paths
- [ ] KM ingestion/retrieval consistency (embedding dimension management)

### Track 3: Integration Wiring
Wire disconnected subsystems into coherent flows.

- [ ] Provider routing visible in debate results (show which models were selected and why)
- [ ] KM retrieval actually enriching debate context (not just write-only)
- [ ] Debate outcomes feeding back into agent ELO ratings
- [ ] Pipeline stage transitions visible in the UI
- [ ] Receipt store ↔ dashboard ↔ API ↔ share links fully connected
- [ ] Compliance artifacts auto-generated from debate receipts
- [ ] Webhook delivery for debate completion events

### Track 4: Frontend & UX Polish
Make the product look and feel professional.

- [ ] All frontend pages rendering real data (not shells)
- [ ] Consistent typography and spacing across themes
- [ ] Mobile responsiveness for key flows
- [ ] Loading states and progress indicators for long operations
- [ ] Error boundaries with helpful recovery actions
- [ ] Debate detail page: clean verdict, agent positions, receipt link
- [ ] Receipts page: searchable, filterable, exportable

### Track 5: Multi-Agent Orchestration
Scale and improve the debate engine.

- [ ] 10+ agent debates at production quality
- [ ] Prover-Estimator consensus mode hardened for real use
- [ ] Cross-verification quality metrics visible in receipts
- [ ] Truth scorer influence visible in vote weights
- [ ] Agent performance benchmarking and selection improvement
- [ ] Debate replay and comparison tools
- [ ] Configurable debate protocols from the UI

### Track 6: Knowledge & Memory
Make the Knowledge Mound genuinely useful.

- [ ] KM retrieval enriching debate context with relevant precedents
- [ ] Semantic search returning quality results (embedding consistency)
- [ ] Cross-debate learning producing measurable quality improvement
- [ ] Knowledge gap identification from debate patterns
- [ ] KM health dashboard showing adapter status and data quality
- [ ] Evidence collection reliability across all sources

### Track 7: Pipeline & Automation
Build out the idea-to-execution vision.

- [ ] DAG-based pipeline visualization in the UI
- [ ] Interactive stage transitions (Ideas → Goals → Actions → Orchestration)
- [ ] Pipeline feedback loops (execution results inform future planning)
- [ ] Autonomous improvement cycles via Nomic Loop
- [ ] Boss loop handling real engineering tasks continuously
- [ ] Queue-based overnight runs producing and merging real improvements

### Track 8: Enterprise & Compliance
Prepare for enterprise adoption.

- [ ] EU AI Act compliance artifacts auto-generated (enforcement: August 2, 2026)
- [ ] SOC 2 controls: close remaining 2% gap
- [ ] Pentest readiness (when enterprise prospect requires it)
- [ ] Multi-tenant isolation hardening
- [ ] Audit log completeness and export
- [ ] HIPAA/FedRAMP compliance paths for healthcare/government verticals

---

## Quarterly Themes

### Q2 2026: Make It Work, Make It Fast
- Complete the dogfood loop (inbox wedge, design partner demos)
- Performance improvements across all user-facing flows
- Wire disconnected subsystems together
- Polish frontend for professional presentation
- Stripe integration and first paid users

### Q3 2026: Scale & Expand
- 10+ agent debates at enterprise quality
- Vertical packages (Healthcare, Financial, Legal)
- Cloud marketplace listings (AWS, Azure)
- Kubernetes Operator for horizontal scaling
- Skills marketplace pilot

### Q4 2026: Platform Ecosystem
- Cross-organization federation
- Decision-Integrity UI Workbench (visual debate canvas)
- Community integration connectors
- Advanced analytics and ROI measurement
- ERC-8004 on-chain deployment

### 2027: Industry Standard
- 1M+ concurrent debates
- Sub-second debate initiation
- Global compliance (HIPAA, FedRAMP)
- Market resolution mechanism for long-horizon settlement
- Canvas GUI 8-stage visual pipeline

---

## Autonomous Execution

The platform improves itself using its own infrastructure:

- **Overnight queue**: Produces bounded improvements while the team sleeps
- **Boss loop**: Dispatches work to AI workers against labeled GitHub issues
- **Nomic Loop**: Self-improvement cycles with human approval gates
- **Swarm orchestration**: Multi-worker parallel execution with lease-based coordination

The operating principle: if an improvement can be specified as a bounded task with clear acceptance criteria, it should be dispatched to the autonomous pipeline, not wait for a human to implement it.

---

## Contributing

Aragora is open to contributions. See [CONTRIBUTING.md](./guide) for guidelines.

Priority contribution areas:
- Evidence connectors for new sources
- Language translations
- Documentation improvements
- Test coverage expansion
- Performance optimization

---

## Contact

- **GitHub**: https://github.com/synaptent/aragora
- **Product Feedback**: product@aragora.ai
- **Enterprise Sales**: sales@aragora.ai
- **Security Issues**: security@aragora.ai

---

*This roadmap represents our current plans and is subject to change based on user feedback and market conditions.*
