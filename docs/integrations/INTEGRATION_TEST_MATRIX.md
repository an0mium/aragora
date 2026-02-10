# Integration Test Matrix

This document provides a comprehensive overview of test coverage across all testing layers.

## Test Infrastructure Summary

| Layer | Count | Location |
|-------|-------|----------|
| Total Python Tests | 1,873 files | `tests/` |
| E2E Tests (Python) | 48+ files | `tests/e2e/` |
| Integration Tests | 50+ files | `tests/integration/` |
| Playwright E2E | 44 files | `aragora/live/e2e/` |

---

## Feature Coverage Matrix

| Feature | Unit | Integration | E2E (Python) | E2E (Playwright) | Status |
|---------|:----:|:-----------:|:------------:|:----------------:|--------|
| **Authentication** | | | | | |
| Login/Logout | Y | Y | Y | Y | Complete |
| OAuth/SSO | Y | Y | Y | Y | Complete |
| MFA | Y | Y | Y | - | Complete |
| JWT Tokens | Y | Y | Y | - | Complete |
| **Debates** | | | | | |
| Create/View | Y | Y | Y | Y | Complete |
| Graph Debates | Y | Y | - | Y | Complete |
| Matrix Debates | Y | Y | - | - | Partial |
| Consensus | Y | Y | Y | - | Complete |
| **Knowledge Mound** | | | | | |
| Facts/Search | Y | Y | Y | - | Complete |
| Federation | Y | Y | Y | - | Complete |
| Governance | Y | Y | - | - | Complete |
| **Agents** | | | | | |
| Selection | Y | Y | Y | - | Complete |
| Fallback | Y | Y | - | - | Complete |
| Calibration | Y | Y | - | - | Complete |
| **Billing** | | | | | |
| Usage Tracking | Y | Y | Y | Y | Complete |
| Cost Analysis | Y | Y | Y | - | Complete |
| Budgets | Y | Y | - | - | Complete |
| **Control Plane** | | | | | |
| Scheduler | Y | Y | Y | Y | Complete |
| Health | Y | Y | Y | - | Complete |
| Policy | Y | Y | Y | - | Complete |
| **Gauntlet** | | | | | |
| Findings | Y | Y | Y | Y | Complete |
| Receipts | Y | Y | - | - | Partial |
| Defense | Y | - | - | - | Partial |
| **Security** | | | | | |
| RBAC | Y | Y | Y | - | Complete |
| Rate Limiting | Y | Y | Y | - | Complete |
| Encryption | Y | Y | - | - | Complete |
| **Compliance** | | | | | |
| GDPR/DSAR | Y | Y | Y | - | Complete |
| Audit Logs | Y | Y | Y | - | Complete |
| Retention | Y | Y | Y | - | Complete |
| **SDK** | | | | | |
| Python Client | Y | Y | Y | - | Partial |
| TypeScript Client | Y | - | - | - | Partial |
| **Connectors** | | | | | |
| Slack | Y | Y | Y | Y | Complete |
| GitHub | Y | Y | - | - | Complete |
| Telegram/WhatsApp | Y | Y | Y | - | Complete |
| **Backup/Recovery** | | | | | |
| Full Backup | Y | Y | Y | - | Complete |
| Incremental | Y | Y | - | - | Complete |
| Restore | Y | Y | Y | - | Complete |

---

## Browser Matrix (Playwright)

| Browser | Desktop | Mobile | Status |
|---------|:-------:|:------:|--------|
| Chromium | Y | - | Active |
| Firefox | Y | - | Active |
| WebKit | Y | - | Active |
| Chrome (Android) | - | Y | Pixel 5 viewport |
| Safari (iOS) | - | Y | iPhone 12 viewport |

---

## Environment Matrix

| Environment | Unit | Integration | E2E | Notes |
|-------------|:----:|:-----------:|:---:|-------|
| Local (localhost) | Y | Y | Y | Primary dev environment |
| CI (GitHub Actions) | Y | Y | Y | Automated on every PR |
| Staging | - | - | Y | Pre-production validation |
| Production | - | - | Monitoring | Health checks only |

---

## Python E2E Tests (`tests/e2e/`)

| Test File | Coverage Area |
|-----------|---------------|
| `test_auth_e2e.py` | Authentication flows, JWT, sessions |
| `test_backup_recovery_e2e.py` | Backup/restore operations |
| `test_billing_accuracy_e2e.py` | Billing calculations |
| `test_compliance_e2e.py` | GDPR, audit logging |
| `test_compliance_deletion_e2e.py` | Data deletion workflows |
| `test_control_plane_workflows.py` | Scheduler, policy |
| `test_debate_lifecycle.py` | Full debate flow |
| `test_connector_sync.py` | External integrations |
| `test_gauntlet_e2e.py` | Security testing |
| `test_knowledge_mound_e2e.py` | Knowledge operations |
| `test_resilience_e2e.py` | Circuit breaker, retry |
| `test_security_e2e.py` | Security controls |

---

## Playwright E2E Tests (`aragora/live/e2e/`)

| Test File | Coverage Area |
|-----------|---------------|
| `critical-flows.spec.ts` | Core user journeys |
| `auth.spec.ts` | Login, logout, session |
| `auth-multi-provider.spec.ts` | OAuth providers |
| `debate-creation.spec.ts` | Create debate flow |
| `debate-viewing.spec.ts` | View debate flow |
| `admin.spec.ts` | Admin panel |
| `billing.spec.ts` | Billing UI |
| `connectors.spec.ts` | Integration UI |
| `control-plane.spec.ts` | Control plane UI |
| `gauntlet.spec.ts` | Gauntlet UI |
| `accessibility.spec.ts` | A11y compliance |

---

## Integration Tests (`tests/integration/`)

| Test File | Integration Type |
|-----------|------------------|
| `test_agent_fallback_chain.py` | Agent → OpenRouter fallback |
| `test_api_integration.py` | API endpoint chains |
| `test_auth_flow.py` | Auth → session → API |
| `test_billing_flow.py` | Usage → cost → budget |
| `test_cdc_km_integration.py` | CDC → Knowledge Mound |
| `test_knowledge_debate_integration.py` | KM ↔ Debate |
| `test_connector_enterprise.py` | Slack/Teams/GitHub |
| `test_webhook_delivery.py` | Webhook dispatch |

---

## Test Markers (pytest)

```bash
# Run by marker
pytest -m unit                    # Fast unit tests
pytest -m integration             # Requires external services
pytest -m integration_minimal     # No external services
pytest -m e2e                     # Full E2E (requires server)
pytest -m slow                    # Long-running (>30s)
pytest -m load                    # Load/stress tests
pytest -m network                 # Real API calls
pytest -m serial                  # Must run serially
```

---

## CI/CD Pipeline

| Job | Tests Run | Timeout |
|-----|-----------|---------|
| Unit Tests | `pytest -m "not integration and not e2e"` | 10 min |
| Integration Tests | `pytest -m integration` | 20 min |
| Python E2E | `pytest tests/e2e/` | 20 min |
| Playwright E2E | `npx playwright test` | 30 min |
| Visual Regression | `npx playwright test --project=visual` | 15 min |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# E2E only (requires running server)
pytest tests/e2e/ -v --timeout=120

# Integration with Docker services
docker compose -f docker-compose.test.yml up -d
pytest tests/integration/ -v

# Playwright E2E
cd aragora/live && npx playwright test

# Specific browser
cd aragora/live && npx playwright test --project=chromium
```

---

## Coverage Gaps (To Address)

| Area | Gap | Priority |
|------|-----|----------|
| SDK E2E | Python SDK endpoint coverage | High |
| Gauntlet | Receipt verification E2E | Medium |
| Matrix Debates | Full E2E coverage | Low |
| Defense Mode | E2E attack/defend cycles | Low |

---

## Related Documentation

- [Test README](../tests/README.md) - Test infrastructure guide
- [E2E Conftest](../tests/e2e/conftest.py) - E2E fixtures
- [Playwright Config](../aragora/live/playwright.config.ts) - Browser config
- [CI/CD Workflow](../.github/workflows/e2e.yml) - E2E pipeline
