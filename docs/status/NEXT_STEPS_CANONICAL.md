# Next Steps (Canonical)

Last updated: 2026-02-13

This is the single source of truth for project-wide execution priorities.
Legacy planning documents should link here.

## Execution Order

### 1) Test Isolation and Pollution Hardening
- Owner: Platform + QA
- Goal: deterministic green runs across randomized order and parallel execution
- Acceptance:
  - Randomized test gate passes for 3 fixed seeds
  - No cross-test state leaks in global fixtures/singletons/event loops
  - Full PR test suite remains green
  - Skip marker debt tracked and burned down from audited baseline (`docs/status/TEST_SKIP_BURNDOWN.md`)
- CI gates:
  - `.github/workflows/test.yml` `test-pollution-randomized`

### 2) Connector Exception Handling Hygiene
- Owner: Backend (Connectors)
- Goal: no silent broad exception swallowing in connector implementations
- Acceptance:
  - Zero `except Exception: pass` / equivalent silent handlers in `aragora/connectors/**`
  - Exceptions are either typed + handled or logged with enough context
- CI gates:
  - `.github/workflows/lint.yml` `connector-exception-hygiene`
  - Script: `scripts/check_connector_exception_handling.py`

### 3) Offline and Demo Golden Path Enforcement
- Owner: CLI + Runtime
- Goal: `--demo` and offline/local execution remain network-free and quiet
- Acceptance:
  - Offline tests validate no audience/network probes and no network-backed subsystems
  - Demo CLI smoke run passes with `ARAGORA_OFFLINE=1`
  - Local reproducibility script passes: `scripts/run_offline_golden_path.sh`
- CI gates:
  - `.github/workflows/smoke.yml` `offline-golden-path`
  - Tests: `tests/cli/test_offline_golden_path.py`

### 4) Documentation and Registry Drift Control
- Owner: Docs + Platform
- Goal: docs reflect runtime agent registry + allowlist exactly
- Acceptance:
  - AGENTS.md counts match runtime registry and allowlist
  - AGENTS table exactly matches `list_available_agents()`
- CI gates:
  - `.github/workflows/lint.yml` `agent-registry-sync`
  - Script: `scripts/check_agent_registry_sync.py`

### 5) SDK and Version Alignment
- Owner: SDK
- Goal: prevent SDK/server/doc version drift and keep parity checks blocking
- Acceptance:
  - `check_version_alignment` passes on SDK parity changes
  - SDK parity report remains clean for handler changes
  - SDK parity debt follows weekly budget reduction (`scripts/baselines/check_sdk_parity_budget.json`)
- CI gates:
  - `.github/workflows/sdk-parity.yml` (version alignment + parity)

### 6) Self-Hosted Production Readiness Validation
- Owner: DevOps + SRE
- Goal: production compose + required env wiring stay valid
- Acceptance:
  - Required services/dependencies/env keys validated in CI
  - Redis Sentinel and DB dependencies remain correctly wired
- CI gates:
  - `.github/workflows/integration.yml` `self-host-readiness`
  - Script: `scripts/check_self_host_compose.py`

### 7) External Pentest Closure Gate
- Owner: Security
- Goal: unresolved HIGH/CRITICAL pentest findings cannot drift unnoticed
- Acceptance:
  - Findings tracked in `security/pentest/findings.json`
  - Execution plan maintained in `security/pentest/EXECUTION_PLAN.md`
  - CI fails if unresolved HIGH/CRITICAL findings exist
- CI gates:
  - `.github/workflows/security.yml` `pentest-findings`
  - Script: `scripts/check_pentest_findings.py`

## Operating Rules
- Any new roadmap or status doc must link this file instead of redefining priorities.
- If priorities change, update this file first, then update linked summaries.
- Legacy planning docs should be compatibility pointers, not alternate priority sources.
