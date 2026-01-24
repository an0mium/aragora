# Aragora Q1 2026 Backlog (12 Weeks)

**Version:** 2.1.14
**Start Date:** Week of Jan 27, 2026
**Tracks:** SME Starter Pack | Developer Platform | Self-Hosted Deployment

---

## Executive Summary

| Metric | Current | Target |
|--------|---------|--------|
| Commercial Readiness | 85% | 95% |
| SOC 2 Readiness | 98% | 100% |
| Test Coverage | 46,868 tests | 48,000+ |
| SDK Parity (TS) | ~70% | 95% |

**Key Deliverables:**
1. **SME Starter Pack** - First debate + decision receipt in <15 minutes
2. **Developer Platform** - SDK parity, OpenAPI spec, example apps
3. **Self-Hosted** - Single docker-compose, <30 min setup

---

## Sprint 1 (Weeks 1-2): Foundations & Audit

### SME Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| SME-01 | Define SME starter pack scope and success metrics | P0 | 3d |
| SME-02 | Audit existing Slack/Email/Drive integrations | P0 | 2d |
| SME-03 | Design 15-minute onboarding flow | P1 | 2d |

### Developer Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| DEV-01 | Build OpenAPI spec generation pipeline | P0 | 3d |
| DEV-02 | Document API reference with examples | P1 | 2d |
| DEV-03 | Audit SDK parity gaps (TS vs Python) | P0 | 1d |

### Self-Hosted Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| HOST-01 | Create baseline docker-compose.yml | P0 | 2d |
| HOST-02 | Create .env.example with all variables | P0 | 1d |
| HOST-03 | Draft self-hosted quickstart guide | P1 | 2d |

### Release Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| REL-01 | Version alignment validation in CI | P0 | 1d |
| REL-02 | Update CHANGELOG for v2.1.14 | P0 | 0.5d |

---

## Sprint 2 (Weeks 3-4): SME Starter Pack v0.1 & Dev Docs

### SME Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| SME-04 | Implement guided onboarding wizard (CLI/Web) | P0 | 5d |
| SME-05 | First debate → receipt flow in <15 min | P0 | 3d |
| SME-06 | Decision receipt export (PDF/Markdown) | P0 | 3d |

### Developer Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| DEV-04 | TS SDK: debates/streaming/receipts parity | P0 | 5d |
| DEV-05 | Update SDK_PARITY.md with coverage metrics | P1 | 1d |
| DEV-06 | Add streaming example to SDK docs | P1 | 2d |

### Self-Hosted Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| HOST-04 | Health check endpoints documentation | P0 | 1d |
| HOST-05 | Smoke test script for self-hosted | P1 | 2d |

---

## Sprint 3 (Weeks 5-6): Cost Controls & Examples

### SME Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| SME-07 | Usage dashboard with spend tracking | P0 | 4d |
| SME-08 | Workspace budget caps and alerts | P0 | 3d |
| SME-09 | SME workflow templates library (8-12) | P1 | 3d |

### Developer Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| DEV-07 | Example app: TypeScript streaming + receipts | P0 | 3d |
| DEV-08 | Example app: Python SDK basic usage | P0 | 2d |
| DEV-09 | examples/ directory with READMEs | P1 | 1d |

### Self-Hosted Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| HOST-06 | Security hardening defaults | P0 | 2d |
| HOST-07 | TLS configuration guide | P1 | 1d |
| HOST-08 | Environment variable documentation | P1 | 1d |

---

## Sprint 4 (Weeks 7-8): Integration Polish & Error Handling

### SME Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| SME-10 | Slack integration setup wizard | P0 | 4d |
| SME-11 | Google Drive/Email integration wizard | P0 | 4d |
| SME-12 | Integration audit trails | P1 | 2d |

### Developer Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| DEV-10 | Align SDK/API error models | P0 | 3d |
| DEV-11 | Retry semantics in SDKs | P0 | 2d |
| DEV-12 | Error code documentation | P1 | 2d |

### Self-Hosted Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| HOST-09 | Observability bundle (metrics/logging) | P0 | 3d |
| HOST-10 | Sample Grafana dashboards | P1 | 2d |

### QA Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| QA-01 | Integration test matrix for connectors | P0 | 3d |
| QA-02 | Mocked coverage in CI | P1 | 2d |

---

## Sprint 5 (Weeks 9-10): Admin Features & Docs Polish

### SME Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| SME-13 | Workspace admin UI (invites, roles) | P0 | 4d |
| SME-14 | RBAC-lite for workspace members | P0 | 3d |
| SME-15 | Audit log UI | P1 | 2d |

### Developer Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| DEV-13 | SDK documentation portal landing page | P0 | 3d |
| DEV-14 | Unified "Developer quickstart" path | P1 | 2d |
| DEV-15 | API versioning documentation | P1 | 1d |

### Self-Hosted Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| HOST-11 | Backup and restore scripts | P0 | 3d |
| HOST-12 | Upgrade runbook | P1 | 2d |
| HOST-13 | Disaster recovery documentation | P1 | 2d |

### Release Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| REL-03 | Automated changelog generation | P1 | 2d |
| REL-04 | Pre-release validation gate in CI | P0 | 2d |

---

## Sprint 6 (Weeks 11-12): GA Polish & Release

### SME Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| SME-16 | ROI/usage dashboard (cost, debates, receipts) | P0 | 4d |
| SME-17 | SME starter pack GA documentation | P0 | 2d |
| SME-18 | User feedback collection mechanism | P1 | 2d |

### Developer Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| DEV-16 | SDK parity pass #2 (>95% coverage) | P0 | 4d |
| DEV-17 | API coverage tests | P0 | 2d |
| DEV-18 | Developer portal GA | P1 | 2d |

### Self-Hosted Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| HOST-14 | Self-hosted GA sign-off (<30 min setup) | P0 | 2d |
| HOST-15 | Production deployment checklist | P0 | 1d |
| HOST-16 | Self-hosted vs hosted comparison docs | P1 | 1d |

### QA Track
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| QA-03 | End-to-end smoke tests (hosted + self-hosted) | P0 | 3d |
| QA-04 | Nightly CI runs | P1 | 2d |

---

## Stretch Goals (Weeks 13-16)

| ID | Title | Track | Priority |
|----|-------|-------|----------|
| SME-19 | Notion/Confluence integration | SME | P2 |
| SME-20 | Jira integration | SME | P2 |
| DEV-19 | CLI `aragora init` scaffolding | DEV | P2 |
| DEV-20 | Event streaming samples | DEV | P2 |
| HOST-17 | Helm chart alpha | HOST | P2 |
| HOST-18 | Kubernetes operator alpha | HOST | P3 |

---

## Success Metrics

### SME Starter Pack
- [ ] New workspace → first receipt in <15 minutes
- [ ] 3+ integration wizards (Slack, Email, Drive)
- [ ] Budget controls configurable per workspace
- [ ] 10+ workflow templates available

### Developer Platform
- [ ] TypeScript SDK ≥95% parity with Python
- [ ] OpenAPI spec auto-generated in CI
- [ ] 3+ example apps (TS streaming, Python basic, React)
- [ ] Unified developer documentation portal

### Self-Hosted
- [ ] Single `docker-compose up` → running in <5 min
- [ ] Complete setup (with config) in <30 min
- [ ] Observability bundle included
- [ ] Documented backup/restore/upgrade paths

---

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| Third-party pentest delays | Schedule immediately; use internal scanning meanwhile |
| SDK parity effort underestimated | Focus on core endpoints first (debates, receipts) |
| Integration API changes | Pin versions; add regression tests |
| Self-hosted complexity | Start with docker-compose only; defer k8s |

---

## Team Allocation (Suggested)

| Track | Engineers | Focus |
|-------|-----------|-------|
| SME | 2 | Onboarding, integrations, templates |
| Developer | 1-2 | SDK, docs, examples |
| Self-Hosted | 1 | Docker, scripts, runbooks |
| QA | 1 (shared) | Test matrix, CI automation |

---

## Related Documents

- [ROADMAP.md](../ROADMAP.md) - Strategic roadmap
- [STATUS.md](STATUS.md) - Feature implementation status
- [SDK_PARITY.md](SDK_PARITY.md) - SDK coverage tracking
- [SUPABASE_SETUP.md](SUPABASE_SETUP.md) - Database configuration
- [TESTING.md](TESTING.md) - Test guidelines

---

*Generated: 2026-01-24*
*Next Review: Sprint 2 Planning*
