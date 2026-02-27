# Agent Task Assignments

**Purpose:** Suggested focus areas for AI coding agents to minimize conflicts.

---

## Recommended Agent Setup

Run up to 3-4 agents in parallel, each on a different track:

| Agent | Track | Focus | Start Command |
|-------|-------|-------|---------------|
| Agent 1 | SME | User-facing features | "Work on SME track issues" |
| Agent 2 | Developer | SDKs and documentation | "Work on Developer track issues" |
| Agent 3 | Self-Hosted | Deployment and ops | "Work on Self-Hosted track issues" |
| Agent 4 | QA | Tests and CI | "Work on QA track issues" |
| Agent 5 | Security | Vulnerability scanning | "Work on Security track issues" |

---

## Track Details

### SME Track (Small Business Users)

**Goal:** Make Aragora usable for non-technical business users

**Previous issues (#91, #92, #99, #100) all CLOSED.** Current work:

**Priority Issues:**
1. **#293 [EPIC] Enterprise Communication Hub & Active Triage** - Active triage, communication integration
   - Files: `aragora/connectors/`, `aragora/integrations/`, `aragora/server/handlers/`
   - Needs: Inbox auto-debate routing, notification wiring

2. **#292 [EPIC] Debate Engine Integration & Marketplace Synergy** - Marketplace + debate UX
   - Files: `aragora/marketplace/`, `aragora/debate/`, `aragora/live/src/`
   - Needs: Template marketplace, debate experience polish

3. **Product quality UX** (PRs #403, #400, #396) - Onboarding, CEO-mode, persist/retry/share
   - Files: `aragora/live/src/`, `aragora/server/handlers/`
   - Needs: Frontend wiring, API integration, swarm UX

**Starter prompt:**
```
Work on the SME track for Aragora. Focus on making the product
usable for small business users who aren't technical.

Priority: Epic #293 (Enterprise Communication Hub) and UX quality PRs

Stay within these folders:
- aragora/live/src/ (frontend)
- aragora/server/handlers/ (API endpoints)
- aragora/connectors/ (integration connectors)

Don't modify: aragora/debate/orchestrator.py, aragora/agents/, core.py
```

---

### Developer Track (SDK & API Users)

**Goal:** Make Aragora easy to integrate programmatically

**Previous issues (#94, #102, #103) all CLOSED.** SDK at 185 namespaces, 100% parity.

**Priority Issues:**
1. **#297 [EPIC] SDK Parity, Golden Paths & Developer Experience** - SDK golden paths, DX improvements
   - Files: `sdk/`, `docs/`, `aragora/server/`
   - Needs: SDK stale endpoint cleanup, golden path examples, developer quickstart

2. **#294 [EPIC] Idea-to-Execution Pipeline & Safe Code Execution** - Pipeline and sandbox
   - Files: `aragora/pipeline/`, `aragora/sandbox/`, `tests/pipeline/`
   - Needs: Pipeline persistence, safe execution, stage advancement

3. **#323 Integrate FastAPI v2 marketplace/orchestration routes** - API modernization
   - Files: `aragora/server/`, `aragora/marketplace/`
   - Needs: FastAPI route integration, contract-safe rollout

**Starter prompt:**
```
Work on the Developer track for Aragora. Focus on SDK golden paths
and pipeline integration.

Priority: Epic #297 (SDK Parity & Golden Paths)

Stay within these folders:
- sdk/ (Python and TypeScript SDKs)
- docs/ (documentation)
- aragora/pipeline/ (idea-to-execution pipeline)
- tests/sdk/ (SDK tests)

Don't modify: aragora/debate/orchestrator.py, aragora/live/src/app/
```

---

### Self-Hosted Track (On-Premise Deployment)

**Goal:** Enable customers to run Aragora on their own infrastructure

**Previous issues (#88, #96, #105, #106) all CLOSED.** Deployment, backup, and observability are production-ready.

**Priority Issues:**
1. **#273 [EPIC] Enterprise Assurance Closure** - Production hardening, security sign-off
   - Files: `aragora/ops/`, `docker/`, `scripts/`
   - Needs: Runtime validation, self-host smoke tests, deployment verification

2. **Self-host runtime stability** - Ongoing CI fixes (PRs #375, #376, #379)
   - Files: `scripts/`, `.github/workflows/`, `docker/`
   - Needs: Runtime probe hardening, compose port handling, readiness fallbacks

3. **Release readiness gate** - Deterministic dep resolution (PR #369)
   - Files: `.github/workflows/`, `scripts/`
   - Needs: Optional deps policy, worktree hygiene checks

**Starter prompt:**
```
Work on the Self-Hosted track for Aragora. Focus on runtime
stability and enterprise assurance closure.

Priority: Epic #273 (Enterprise Assurance Closure)

Stay within these folders:
- scripts/ (automation scripts)
- docker/ (container configs)
- aragora/ops/ (deployment validation)
- .github/workflows/ (CI pipelines)

Don't modify: aragora/debate/orchestrator.py, aragora/server/handlers/
```

---

### QA Track (Quality Assurance)

**Goal:** Ensure reliability and catch regressions

**Previous issues (#90, #107, #108) all CLOSED.** 208,000+ tests, nightly CI active, E2E smoke tests running.

**Priority Issues:**
1. **E2E golden path stabilization** - Fix flaky integration tests (PRs #372, #377, #381)
   - Files: `tests/e2e/`, `aragora/live/e2e/`, `.github/workflows/`
   - Needs: Playwright deduplication, auth/workflow/email test stabilization

2. **Benchmark noise floor** - CI benchmark regression gate (PR #394)
   - Files: `.github/workflows/`, `tests/`
   - Needs: Noise floor calculation, false-positive reduction

3. **#295 [EPIC] Nomic Loop Safety Gates & Observable Evolution** - Self-improvement test coverage
   - Files: `tests/nomic/`, `aragora/nomic/`, `scripts/`
   - Needs: Gate verification, evolution observability tests

**Starter prompt:**
```
Work on the QA track for Aragora. Focus on E2E test stability
and CI pipeline reliability.

Priority: E2E golden path stabilization (PRs #372, #377, #381)

Stay within these folders:
- tests/ (all tests)
- aragora/live/e2e/ (Playwright tests)
- .github/workflows/ (CI config)

Don't modify: aragora/debate/orchestrator.py, aragora/server/ (except adding tests)
```

---

### Security Track (Security Hardening)

**Goal:** Identify and fix security vulnerabilities, harden production

**Priority Issues:**
1. **#274 Execute external penetration test and remediate findings** *(priority:critical)*
   - Files: `aragora/security/`, `aragora/audit/`
   - Needs: Third-party pentest coordination, finding remediation

2. **#273 [EPIC] Enterprise Assurance Closure** *(priority:critical)*
   - Files: `aragora/auth/`, `aragora/rbac/`, `aragora/compliance/`
   - Needs: Security sign-off, compliance verification

3. **#296 [EPIC] Compliance Dashboard & Verifiable Decision Receipts**
   - Files: `aragora/compliance/`, `aragora/gauntlet/`
   - Needs: Dashboard wiring, receipt verification UI

4. **Secrets sync and runtime hardening** (completed in PR #365)
   - Settlement hooks, secrets sync, and runtime hardening all merged
   - Ongoing: error sanitization maintenance (no `str(e)` in responses)

**Starter prompt:**
```
Work on the Security track for Aragora. Focus on enterprise
assurance closure and compliance.

Priority: Epic #273 (Enterprise Assurance Closure)

Stay within these folders:
- aragora/security/ (encryption, key rotation)
- aragora/audit/ (security scanner, bug detector)
- aragora/auth/ (authentication)
- aragora/rbac/ (authorization)
- aragora/compliance/ (compliance framework)

Scripts available:
- python scripts/security_audit.py --fail-on-critical
- python scripts/security_checklist.py --ci

Don't modify: core debate engine without approval
```

---

## Quick Start Templates

### For Claude Code Sessions

Copy-paste when starting a session:

```
I'm working on Aragora. Check .claude/COORDINATION.md first.

I want to work on: [TRACK NAME] track
Specifically: [ISSUE NUMBER or DESCRIPTION]

Before changing code:
1. Tell me your plan in plain language
2. List the files you'll modify
3. Wait for my OK

After changes:
1. Run tests: pytest tests/ -x --timeout=60 -m "not slow"
2. Update .claude/COORDINATION.md
3. Commit with descriptive message
```

### For Codex Sessions

```
Project: Aragora (multi-agent AI decision platform)
Task: [SPECIFIC TASK]
Constraints:
- Only modify files in: [FOLDER LIST]
- Run tests before committing
- Keep changes focused and small
```

---

## What NOT to Modify

These files require extra caution:

| File | Reason | Who Can Modify |
|------|--------|----------------|
| `aragora/core.py` | Core types, many dependencies | Explicit approval only |
| `aragora/debate/orchestrator.py` | Central debate logic | Explicit approval only |
| `CLAUDE.md` | AI instructions | Manual only |
| `.env*` | Secrets | Never commit |
| `scripts/nomic_loop.py` | Self-improvement safety | Explicit approval only |

---

## Checking for Conflicts

Before starting work, run:

```bash
# See what's changed recently
git log --oneline -10

# See uncommitted changes
git status

# See who's working on what
cat .claude/COORDINATION.md
```

If you see unexpected changes, ask before proceeding.
