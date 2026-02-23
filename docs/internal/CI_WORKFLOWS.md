# CI Workflow Inventory

Comprehensive catalog of all GitHub Actions workflows in `.github/workflows/`.

## Workflow Categories

### Quality Gates (PR blocking)

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Lint | `lint.yml` | Ruff format/check on changed Python files, connector exception hygiene, agent registry sync, mypy typecheck, frontend ESLint, Helm lint, TODO/FIXME audit, docs sync | push/PR to main (path-filtered), dispatch |
| Tests | `test.yml` | Multi-tier test suite: version alignment, baseline determinism, fast parallel tests (4 categories), randomized pollution guard, compatibility matrix (3 OS x 4 Python), golden path smoke, CLI smoke, nightly slow tier, frontend E2E, security scan, typecheck, migration tests, quality gates | push/PR to main, nightly schedule, dispatch |
| Coverage Gate | `coverage.yml` | Runs full test suite with coverage, enforces 70% minimum, generates PR coverage comment, checks for new zero-coverage files, coverage diff for changed files | push/PR to main (path-filtered), dispatch |
| Smoke Tests | `smoke.yml` | Core smoke tests (dataclass creation, debate lifecycle), offline golden path, server smoke (health, CORS, WebSocket), decision-to-action smoke | push/PR to main (path-filtered), dispatch |
| Offline Golden Path | `smoke-offline.yml` | Verifies offline/demo mode works end-to-end without API keys | push/PR to main (path-filtered), dispatch |
| Core Suites | `core-suites.yml` | Focused decision integrity test suites via `ci_core_suites.sh` | push/PR to main (path-filtered), dispatch |
| OpenAPI Spec | `openapi.yml` | Generates OpenAPI spec, validates structure (top-level keys, paths, security schemes), generates SDK types (TS, Python, Live), validates handler route coverage, SDK contracts, namespace parity, contract drift. Syncs spec to main on push. | push/PR to main (path-filtered), dispatch |
| Benchmarks | `benchmark.yml` | pytest-benchmark with regression detection, orchestration speed smoke, nightly latency tracking | push/PR to main (path-filtered), nightly schedule, dispatch |
| Performance Regression | `benchmarks.yml` | PR-only benchmark comparison against main baseline, posts regression report as PR comment | PR to main (path-filtered) |

### Security

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Security | `security.yml` | CodeQL (Python + JS), Bandit scan, Aragora security scanner, dependency vulnerability check (Safety + pip-audit + npm audit), pentest findings gate, RBAC coverage check, secret scanning (Gitleaks + TruffleHog) | push/PR to main (path-filtered), weekly schedule, dispatch |
| Build & Publish Docker | `build.yml` | Multi-platform Docker build (amd64/arm64), SBOM generation, Cosign signing, Trivy config + image vulnerability scanning | push/PR to main (path-filtered), tags, dispatch |

### SDK & API

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| SDK Tests | `sdk-test.yml` | Python SDK tests (multi-version matrix), TypeScript SDK tests (build + test) | push/PR to main (SDK path-filtered), dispatch |
| SDK Parity | `sdk-parity.yml` | Verifies Python and TypeScript SDKs expose equivalent API surface | push/PR to main, dispatch |
| SDK Generate | `sdk-generate.yml` | Generates TypeScript SDK types from OpenAPI spec, builds and validates | push/PR to main (SDK/OpenAPI path-filtered), dispatch |
| Contract Drift Governance | `contract-drift-governance.yml` | Enforces contract drift ratchet (no new regressions vs baseline), generates drift backlog | push/PR to main (handler/SDK path-filtered) |
| Connector Registry | `connector-registry.yml` | Validates connector registry consistency | push/PR to main (connectors path-filtered) |

### Deployment

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Deploy (Secure) | `deploy-secure.yml` | Production deployment via AWS OIDC + SSM (no SSH keys), with environment approval gates | push to main, dispatch |
| Deploy Canary | `deploy-canary.yml` | Progressive canary deployment with ALB traffic shifting, CloudWatch health gates, auto-rollback | dispatch only (manual) |
| Deploy Frontend | `deploy-frontend.yml` | Deploys Next.js frontend to Cloudflare Pages or Docker | push to main (live path-filtered), dispatch |
| Deploy Lightsail | `deploy-lightsail.yml` | Deploys to AWS Lightsail instance | dispatch only (manual) |
| Deploy Multi-Region | `deploy-multi-region.yml` | Multi-region deployment with region selection | push to main, dispatch |
| Docker Build | `docker.yml` | Builds backend/frontend/operator Docker images, Trivy scans, PR integration test | push/PR to main (Docker path-filtered), tags, dispatch |
| Status Page | `status-page.yml` | Deploys Uptime Kuma status page infrastructure | push to main (deploy/uptime-kuma path-filtered), dispatch |

### Publishing

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Release | `release.yml` | Full release pipeline: validate version, run tests (parallel matrix), build package, publish to PyPI, create GitHub Release | tag push (v*), dispatch |
| Publish aragora | `publish-aragora.yml` | Publishes main `aragora` package to PyPI (manual confirmation required) | dispatch only |
| Publish aragora-debate | `publish-aragora-debate.yml` | Publishes standalone `aragora-debate` package to PyPI | tag push (aragora-debate-v*), dispatch |
| Publish Python SDK | `publish-sdk-python.yml` | Publishes `aragora-sdk` to PyPI | tag push (sdk-python-v*), dispatch |
| Publish TypeScript SDK | `publish-sdk-typescript.yml` | Publishes `@aragora/sdk` to npm | tag push (sdk-ts-v*), dispatch |
| Publish VS Code Extension | `publish-vscode.yml` | Publishes VS Code extension | tag push (vscode-v*), dispatch |
| Release Notes | `release-notes.yml` | DEPRECATED -- use release.yml. Kept for manual standalone release note generation. | dispatch only |
| Generate SBOM | `sbom.yml` | Generates Python + TypeScript SBOMs on release | release published, dispatch |

### Testing (Specialized)

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| E2E Tests | `e2e.yml` | Full Playwright E2E (backend + frontend), Python E2E tests, visual regression tests | push/PR to main (live path-filtered), nightly schedule, dispatch |
| Integration | `integration.yml` | Self-hosting compose validation, cross-module integration tests | push/PR to main (path-filtered) |
| Load Tests | `load-tests.yml` | Locust load testing with configurable concurrency | push/PR to main (load test path-filtered), nightly schedule, dispatch |
| Migration Tests | `migration-tests.yml` | Database migration up/down/up cycle against PostgreSQL | push/PR to main (migration path-filtered) |
| New Features CI | `new-features.yml` | E2E tests for recently added features (marketplace, webhooks, explainability) | push/PR to main (specific handler path-filtered) |
| Nightly Integration | `nightly-integration.yml` | Live debate E2E using real API keys (nightly only) | nightly schedule, dispatch |
| Capability Gap | `capability-gap.yml` | Generates capability surface report and matrix delta | push/PR to main (capabilities path-filtered) |

### Monitoring & Operations

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Uptime Monitor | `monitor.yml` | Checks frontend and backend health every 30 minutes | schedule (*/30 * * * *), dispatch |
| Production Monitor | `production-monitor.yml` | Deep production health check every 6 hours | schedule (0 */6 * * *), dispatch |
| Backup Verification | `backup-verification.yml` | Weekly backup integrity check, optional full DR drill | weekly schedule (Sunday 3 AM UTC), dispatch |

### AI-Assisted

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Aragora PR Review | `pr-debate.yml` | AI-powered PR review using multi-agent debate (uses pull_request_target for security) | PR opened/synchronized, dispatch |
| Aragora Gauntlet Review | `aragora-gauntlet.yml` | Runs Gauntlet review on PR changes | PR opened/synchronized/reopened |
| TestFixer Auto | `testfixer-auto.yml` | Auto-fixes failing tests after Tests workflow completes using AI agents | workflow_run (after Tests), dispatch |

### Nomic (Self-Improvement)

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Nomic Branch CI | `nomic-ci.yml` | Targeted tests on dev/** branches created by the Nomic Loop | push to dev/** branches |

### Documentation

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Deploy Documentation | `deploy-docs.yml` | Deploys docs-site to GitHub Pages | push to main (docs path-filtered), dispatch |
| Build Documentation | `docs-build.yml` | PR check that docs build successfully, generates doc stats | PR to main (docs path-filtered) |
| Lighthouse CI | `lighthouse.yml` | Lighthouse performance audit on frontend | push/PR to main (live path-filtered), dispatch |

## Complementary vs Redundant Analysis

### Complementary Pairs (intentional overlap, different scopes)

- **benchmark.yml + benchmarks.yml**: benchmark.yml tracks trends and runs nightly; benchmarks.yml is PR-only regression comparison. Both needed.
- **smoke.yml + smoke-offline.yml**: smoke.yml validates server-connected paths; smoke-offline.yml specifically tests no-API-key operation. Both needed.
- **test.yml + core-suites.yml**: test.yml is the comprehensive matrix; core-suites.yml runs a focused subset via `ci_core_suites.sh`. Both needed.
- **test.yml + coverage.yml**: test.yml includes coverage in the compat matrix job; coverage.yml provides dedicated threshold enforcement and PR comments. Both needed.
- **build.yml + docker.yml**: build.yml builds single monolith image with signing/SBOM; docker.yml builds separate backend/frontend/operator images with Trivy scans. Different targets.
- **security.yml (in lint.yml) + security.yml**: lint.yml has a `security` job (Bandit); security.yml is the full security suite (CodeQL, Bandit, dep check, secrets). lint.yml security is fast/light; security.yml is comprehensive.
- **deploy-secure.yml + deploy-canary.yml + deploy-lightsail.yml + deploy-multi-region.yml**: Different deployment strategies for different scenarios.

### Merged (previously redundant)

- **openapi.yml + openapi-validate.yml**: MERGED into `openapi.yml`. openapi-validate.yml was a strict subset of openapi.yml (spec generation + structure validation). The merged workflow now includes all validation steps from both.

### Potential Consolidation Candidates

- **release.yml + release-notes.yml**: release-notes.yml is already deprecated (workflow_dispatch only). Can be removed once confirmed unused.
- **test.yml migration-test job + migration-tests.yml**: test.yml has a `migration-test` job; migration-tests.yml is a dedicated workflow. The dedicated workflow has more specific path filters. Low priority to merge since path filters prevent unnecessary runs.

## Trigger Summary

| Trigger Type | Workflows |
|-------------|-----------|
| push + PR to main | lint, test, build, security, smoke, smoke-offline, coverage, openapi, benchmark, sdk-test, sdk-parity, sdk-generate, contract-drift-governance, connector-registry, core-suites, e2e, integration, load-tests, migration-tests, new-features, capability-gap, docker, lighthouse |
| PR only | benchmarks (regression), docs-build, aragora-gauntlet |
| Tags (v*) | build, release |
| Package tags | publish-aragora-debate, publish-sdk-python, publish-sdk-typescript, publish-vscode |
| Schedule only | monitor (*/30min), production-monitor (6h), backup-verification (weekly) |
| Schedule + push/PR | test (nightly), load-tests (nightly), security (weekly), e2e (nightly) |
| Schedule + dispatch only | nightly-integration |
| Dispatch only | deploy-canary, deploy-lightsail, publish-aragora, release-notes |
| workflow_run | testfixer-auto (after Tests) |
| Nomic branches | nomic-ci (dev/**) |
| Deploy on push | deploy-secure, deploy-frontend, deploy-multi-region, deploy-docs, status-page |
