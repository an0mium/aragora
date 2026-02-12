# Aragora 30/60/90 Roadmap

> **Goal:** Ship a verifiable AI operations platform with three first-class pillars:
> adversarial decisioning, policy-gated execution (OpenClaw), and provenance attestation (ERC-8004).

---

## Day 0: Current State (Feb 12, 2026)

| Surface | Python SDK | TypeScript SDK | Python Client | Server | Tests |
|---------|-----------|---------------|---------------|--------|-------|
| **OpenClaw** | 22/22 endpoints | 5/22 endpoints, 2 path mismatches | 22/22 endpoints | 22 handlers | Partial |
| **Blockchain** | 6/7 endpoints | 6/7 endpoints | N/A | 5 impl + 2 stub (501) | 182 tests |
| **Debate** | Via client | N/A | Via client | Full | 13,500+ |

**Critical gaps:**
1. TypeScript OpenClaw: 77% of endpoints missing, 2 path mismatches
2. SDK_PARITY.md reports blockchain at 0% (stale — actual is ~86%)
3. No contract parity tests (drift can recur silently)
4. ERC-8004 agent list/register endpoints return 501

### Completed in this run (Feb 12, 2026)

- OpenClaw SDK contract alignment shipped:
  - `sdk/python/aragora_sdk/namespaces/openclaw.py` now targets gateway endpoints
  - `sdk/typescript/src/namespaces/openclaw.ts` paths aligned to server (`/actions`, `/sessions/{id}/end`)
- Blockchain SDK namespace coverage added:
  - `sdk/python/aragora_sdk/namespaces/blockchain.py` (new)
  - `sdk/typescript/src/namespaces/blockchain.ts` (new)
  - client wiring updated in `sdk/python/aragora_sdk/client.py` and `sdk/typescript/src/client.ts`
- SDK coverage tests updated:
  - `tests/sdk/test_sdk_coverage_expansion.py` now validates OpenClaw gateway and blockchain namespace registration
- Reliability hardening included:
  - `aragora/connectors/blockchain/connector.py` now degrades gracefully on unexpected connector exceptions

**Validation gates passed in this run:**
- `pytest tests/sdk/test_sdk_coverage_expansion.py tests/sdk/test_endpoint_parity.py tests/sdk/test_sdk_parity.py`
- `pytest tests/client/test_openclaw_api.py tests/server/handlers/test_erc8004.py tests/blockchain tests/connectors/blockchain`

---

## Days 1–30: Contract Unification & Parity Tests

> **Theme:** Make every public API surface tell the same story.

### P0: Fix TypeScript OpenClaw SDK (Week 1)

| Task | File | Test Gate |
|------|------|-----------|
| Fix `executeAction` path | `sdk/typescript/src/namespaces/openclaw.ts` | Path matches `/api/v1/openclaw/actions` |
| Fix `closeSession` → `endSession` | same file | Path matches `/api/v1/openclaw/sessions/{id}/end` |
| Add `getAction`, `cancelAction` | same file | Methods exist, paths match server |
| Add credential lifecycle (4 methods) | same file | `listCredentials`, `storeCredential`, `rotateCredential`, `deleteCredential` |
| Add policy rules (3 methods) | same file | `getPolicyRules`, `addPolicyRule`, `removePolicyRule` |
| Add approvals (3 methods) | same file | `listApprovals`, `approveAction`, `denyAction` |
| Add service introspection (4 methods) | same file | `health`, `metrics`, `audit`, `stats` |

**Test gate:** `pytest tests/sdk/ -k openclaw` — all pass, 22 endpoint paths verified.

### P1: Contract Parity Tests (Week 2)

| Task | File | Test Gate |
|------|------|-----------|
| OpenClaw parity test | `tests/sdk/test_openclaw_parity.py` | Verifies Python SDK, TS SDK, and server OpenAPI define same endpoints |
| Blockchain parity test | `tests/sdk/test_blockchain_parity.py` | Verifies both SDKs match server handler endpoints |
| Update SDK_PARITY.md | `docs/guides/SDK_PARITY.md` | blockchain shows 86%, openclaw shows 100% (both SDKs) |

**Test gate:** `pytest tests/sdk/test_*_parity.py` — 0 failures.

### P2: Harden Blockchain Surface (Weeks 3–4)

| Task | File | Test Gate |
|------|------|-----------|
| Implement `GET /api/v1/blockchain/agents` | `aragora/server/handlers/erc8004.py` | Returns paginated agent list (not 501) |
| Implement `POST /api/v1/blockchain/agents` | same file | Registers agent on-chain (or returns clear error if no web3) |
| Add SDK methods for new endpoints | Both SDK namespaces | `listAgents()` and `registerAgent()` in Python + TS |
| Document endpoint status | `docs/api/BLOCKCHAIN_API.md` | All 7 endpoints documented with request/response schemas |

**Test gate:** `pytest tests/server/handlers/test_erc8004.py` — all pass including new endpoint tests.

---

## Days 31–60: Product Packaging & Developer Experience

> **Theme:** Make it easy to try, easy to buy, easy to extend.

### P3: Standalone Package Polish (Weeks 5–6)

| Task | File | Test Gate |
|------|------|-----------|
| Add reference agent implementations | `aragora-debate/src/aragora_debate/agents.py` | Claude + OpenAI agents with real API calls |
| Add test suite for standalone package | `aragora-debate/tests/` | `pytest aragora-debate/tests/` — 30+ tests |
| Publish to PyPI (test) | `aragora-debate/pyproject.toml` | `pip install aragora-debate` works from test.pypi.org |
| Write integration guide | `aragora-debate/docs/INTEGRATION.md` | How to use with CrewAI, LangGraph, AutoGen |

### P4: Tiered Module Packaging (Weeks 7–8)

| Tier | Modules | Status |
|------|---------|--------|
| **Core** (always installed) | `debate/`, `gauntlet/`, `ranking/`, `knowledge/mound/` | Stable |
| **Gateway** (OpenClaw pack) | `compat/openclaw/`, `server/handlers/openclaw/` | Stable |
| **Blockchain** (ERC-8004 pack) | `blockchain/`, `connectors/blockchain/` | Stable (web3 optional) |
| **Enterprise** (security pack) | `auth/`, `rbac/`, `tenancy/`, `compliance/` | Stable |
| **Connectors** (integration pack) | `connectors/chat/`, `connectors/enterprise/` | Stable |
| **Experimental** | `genesis/`, `visualization/`, `sandbox/` | Alpha |

| Task | File | Test Gate |
|------|------|-----------|
| Document tier definitions | `docs/PACKAGING.md` | Clear install paths per tier |
| Create optional dependency groups | `pyproject.toml` | `pip install aragora[gateway]`, `pip install aragora[blockchain]` |
| Validate each tier imports independently | `tests/packaging/test_tiers.py` | Each tier importable without others |

---

## Days 61–90: Go-to-Market & Validation

> **Theme:** Prove value with real users.

### P5: OpenClaw as Flagship Capability (Weeks 9–10)

| Task | File | Test Gate |
|------|------|-----------|
| End-to-end OpenClaw demo | `examples/openclaw_gateway.py` | Session → policy → action → approval → audit flow |
| Standalone gateway quickstart | `aragora/compat/openclaw/README.md` | Docker Compose up in < 2 minutes |
| Security audit of gateway surface | `docs/security/OPENCLAW_AUDIT.md` | No OWASP Top 10 issues in gateway handlers |

### P6: Verifiable Decision Pipeline (Weeks 11–12)

| Task | File | Test Gate |
|------|------|-----------|
| Decision → Action → Audit → Attestation demo | `examples/decision_pipeline.py` | Full pipeline produces receipt + on-chain anchor |
| Cross-org trust portability doc | `docs/TRUST_PORTABILITY.md` | ERC-8004 reputation transfer between orgs |
| EU AI Act compliance demo | `examples/eu_ai_act_compliance.py` | Produces Art. 12/13/14 compliant artifacts |

### P7: Metrics & Validation (Ongoing)

| Metric | Target | How to Measure |
|--------|--------|---------------|
| SDK parity | 100% across all namespaces | `pytest tests/sdk/test_*_parity.py` |
| Test suite health | 0 persistent failures, <50 skips | `pytest --tb=no -q` |
| OpenClaw endpoint coverage | 22/22 in both SDKs | Parity test |
| Blockchain endpoint coverage | 7/7 in both SDKs | Parity test |
| Standalone package | Publishable on PyPI | `pip install aragora-debate` |
| Decision receipt generation | < 5 seconds | Benchmark test |

---

## Success Criteria

### Day 30
- [ ] TypeScript OpenClaw SDK: 22/22 endpoints, 0 path mismatches
- [ ] Automated parity tests prevent future contract drift
- [ ] Blockchain endpoints: 7/7 implemented (no more 501s)
- [ ] SDK_PARITY.md accurate and automated

### Day 60
- [ ] `aragora-debate` on PyPI (test)
- [ ] Tiered packaging documented and tested
- [ ] Reference agent implementations for Claude + OpenAI

### Day 90
- [ ] OpenClaw standalone gateway demo running
- [ ] Full decision-to-attestation pipeline demonstrated
- [ ] EU AI Act compliance artifacts generated automatically
