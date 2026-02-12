# Aragora 30/60/90 Roadmap

> **Goal:** Ship a verifiable AI operations platform with three first-class pillars:
> adversarial decisioning, policy-gated execution (OpenClaw), and provenance attestation (ERC-8004).

---

## Day 0: Current State (Feb 12, 2026)

| Surface | Python SDK | TypeScript SDK | Python Client | Server | Tests |
|---------|-----------|---------------|---------------|--------|-------|
| **OpenClaw** | 22/22 endpoints | 22/22 endpoints (incl. deleteSession) | 22/22 endpoints | 22 handlers | 10 parity + 41 client |
| **Blockchain** | 8/8 endpoints | 8/8 endpoints | N/A | 8 impl (0 stubs) | 45 handler + 10 parity |
| **Debate** | Via client | N/A | Via client | Full | 13,500+ |

**All critical gaps resolved:**
1. ~~TypeScript OpenClaw: 77% of endpoints missing~~ — Fixed: 22/22 endpoints, 0 path mismatches
2. ~~SDK_PARITY.md reports blockchain at 0%~~ — Updated to 85.7%
3. ~~No contract parity tests~~ — Added: `tests/sdk/test_contract_parity.py` (10 tests)
4. ~~ERC-8004 agent list/register endpoints return 501~~ — Implemented with full pagination and wallet signing

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

### Completed in this run (continued, Feb 12, 2026)

- ERC-8004 handler endpoints are now implemented end-to-end for both list and register:
  - `GET /api/v1/blockchain/agents`
  - `POST /api/v1/blockchain/agents`
  - with pagination validation, identity-registry configuration checks, and wallet credential error clarity.
- Contract parity hardening expanded:
  - `tests/sdk/test_contract_parity.py` now enforces OpenClaw + Blockchain endpoint parity with no xfail fallback for Python client coverage.
- Python client OpenClaw parity gaps closed in `aragora/client/resources/openclaw.py`:
  - action status/cancel, credential lifecycle, and health/metrics coverage are all represented in client resource calls.
- Parity documentation regenerated using current sources:
  - `docs/guides/SDK_PARITY.md`
  - `docs/api/SDK_ENDPOINT_AUDIT.md`

---

## Days 1–30: Contract Unification & Parity Tests

> **Theme:** Make every public API surface tell the same story.

### P0: Fix TypeScript OpenClaw SDK (Week 1) — COMPLETE

All 22 endpoints aligned. Paths corrected (`/actions`, `/sessions/{id}/end`), missing methods added.

**Test gate:** `pytest tests/sdk/test_contract_parity.py` — 10/10 passed.

### P1: Contract Parity Tests (Week 2) — COMPLETE

`tests/sdk/test_contract_parity.py` enforces OpenClaw (21 endpoints) and Blockchain (8 endpoints) parity across Python SDK, TypeScript SDK, and Python client. No xfail gaps remain.
| Update SDK_PARITY.md | `docs/guides/SDK_PARITY.md` | blockchain shows 86%, openclaw shows 100% (both SDKs) |

**Test gate:** `pytest tests/sdk/test_*_parity.py` — 0 failures.

### P2: Harden Blockchain Surface (Weeks 3–4) — COMPLETE

- `GET /api/v1/blockchain/agents` — paginated listing with identity registry check
- `POST /api/v1/blockchain/agents` — on-chain registration with wallet signer
- `listAgents()` / `registerAgent()` added to Python SDK, TypeScript SDK, and Python client
- Remaining: `docs/api/BLOCKCHAIN_API.md` endpoint documentation

**Test gate:** `pytest tests/server/handlers/test_erc8004.py` — 45/45 passed.

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
| Blockchain endpoint coverage | 8/8 in both SDKs | Parity test |
| Standalone package | Publishable on PyPI | `pip install aragora-debate` |
| Decision receipt generation | < 5 seconds | Benchmark test |

---

## Success Criteria

### Day 30
- [x] TypeScript OpenClaw SDK: 22/22 endpoints, 0 path mismatches
- [x] Automated parity tests prevent future contract drift
- [x] Blockchain endpoints: 8/8 implemented (no more 501s)
- [x] SDK_PARITY.md updated (blockchain 100%, gateway 100%)

### Day 60
- [ ] `aragora-debate` on PyPI (test)
- [ ] Tiered packaging documented and tested
- [ ] Reference agent implementations for Claude + OpenAI

### Day 90
- [ ] OpenClaw standalone gateway demo running
- [ ] Full decision-to-attestation pipeline demonstrated
- [ ] EU AI Act compliance artifacts generated automatically
