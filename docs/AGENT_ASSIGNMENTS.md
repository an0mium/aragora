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

---

## Track Details

### SME Track (Small Business Users)

**Goal:** Make Aragora usable for non-technical business users

**Priority Issues:**
1. **#91 Workspace admin UI** - Add invite/role management UI
   - Files: `aragora/live/src/app/(app)/workspace/`
   - Needs: React components, API integration

2. **#92 RBAC-lite** - Simple permissions for workspace members
   - Files: `aragora/rbac/`, `aragora/server/handlers/`
   - Needs: Permission checks, UI toggles

3. **#99 ROI/usage dashboard** - Show value to customers
   - Files: `aragora/live/src/app/(app)/dashboard/`
   - Needs: Metrics aggregation, charts

**Starter prompt:**
```
Work on the SME track for Aragora. Focus on making the product
usable for small business users who aren't technical.

Priority: Issue #91 (Workspace admin UI)

Stay within these folders:
- aragora/live/src/ (frontend)
- aragora/server/handlers/ (API endpoints)

Don't modify: aragora/debate/, aragora/agents/, core.py
```

---

### Developer Track (SDK & API Users)

**Goal:** Make Aragora easy to integrate programmatically

**Priority Issues:**
1. **#102 SDK parity pass** - Ensure Python/TypeScript SDKs match
   - Files: `sdk/`, `aragora/live/src/api/`
   - Needs: Compare endpoints, add missing methods

2. **#94 SDK docs portal** - Create documentation site
   - Files: `docs/`, `sdk/*/README.md`
   - Needs: API reference, examples

3. **#103 API coverage tests** - Ensure all endpoints tested
   - Files: `tests/server/handlers/`
   - Needs: Test coverage analysis, new tests

**Starter prompt:**
```
Work on the Developer track for Aragora. Focus on SDK quality
and API documentation.

Priority: Issue #102 (SDK parity)

Stay within these folders:
- sdk/ (Python and TypeScript SDKs)
- docs/ (documentation)
- tests/sdk/ (SDK tests)

Don't modify: aragora/debate/, aragora/live/src/app/
```

---

### Self-Hosted Track (On-Premise Deployment)

**Goal:** Enable customers to run Aragora on their own infrastructure

**Priority Issues:**
1. **#96 Backup and restore scripts** - Data safety
   - Files: `scripts/`, `aragora/backup/`
   - Needs: Shell scripts, restore verification

2. **#106 Production deployment checklist** - Go-live guide
   - Files: `docs/deployment/`, `docker/`
   - Needs: Step-by-step docs, validation scripts

3. **#88 Observability bundle** - Monitoring out of box
   - Files: `docker/`, `aragora/server/prometheus*.py`
   - Needs: Grafana dashboards, alert rules

**Starter prompt:**
```
Work on the Self-Hosted track for Aragora. Focus on making
deployment easy and reliable for ops teams.

Priority: Issue #96 (Backup scripts)

Stay within these folders:
- scripts/ (automation scripts)
- docker/ (container configs)
- docs/deployment/ (deployment docs)
- aragora/backup/ (backup module)

Don't modify: aragora/debate/, aragora/server/handlers/
```

---

### QA Track (Quality Assurance)

**Goal:** Ensure reliability and catch regressions

**Priority Issues:**
1. **#107 E2E smoke tests** - Critical path testing
   - Files: `aragora/live/e2e/`, `tests/`
   - Needs: Playwright tests, CI integration

2. **#90 Integration test matrix** - Connector coverage
   - Files: `tests/connectors/`
   - Needs: Test all connector combinations

3. **#108 Nightly CI runs** - Automated quality gates
   - Files: `.github/workflows/`
   - Needs: CI config, notification setup

**Starter prompt:**
```
Work on the QA track for Aragora. Focus on test coverage
and CI/CD reliability.

Priority: Issue #107 (E2E smoke tests)

Stay within these folders:
- tests/ (all tests)
- aragora/live/e2e/ (Playwright tests)
- .github/workflows/ (CI config)

Don't modify: aragora/debate/, aragora/server/ (except adding tests)
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
