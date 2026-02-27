# AI Agent Coordination

**Last updated:** 2026-02-26
**Maintainer:** Update this file when starting/finishing work

---

## Quick Reference

| Track | Focus Area | Key Folders |
|-------|------------|-------------|
| **SME** | Small business features | `aragora/live/`, `aragora/server/handlers/` |
| **Developer** | SDKs, APIs, docs | `sdk/`, `docs/`, `aragora/server/` |
| **Self-Hosted** | Docker, deployment | `docker/`, `scripts/`, `aragora/ops/` |
| **QA** | Tests, CI/CD | `tests/`, `.github/` |
| **Release** | Versioning, changelog | Root files, `docs/` |

---

## Active Work

> **Instructions:** Before starting work, add your session below.
> When done, move to "Recently Completed" section.

### Currently Active

<!-- Copy this template for each active session:

### [Agent Type] - [Brief Task]
- **Started:** YYYY-MM-DD HH:MM
- **Working on:** One sentence description
- **Files:** List main files being modified
- **Issue:** #number (if applicable)
- **Status:** In Progress / Blocked / Almost Done

-->

**19 open PRs** under active review or development:

| PR | Branch | Focus |
|----|--------|-------|
| #403 | `feat/product-quality-ux` | End-to-end UX: persist, retry, share, monitor |
| #402 | `docs/loop-first-evolution-design` | Loop-first evolution design document |
| #401 | `feat/obsidian-bidirectional-sync` | Obsidian <-> Knowledge Mound bidirectional sync |
| #400 | `feat/onboarding-critical-path` | Onboarding: frontend to backend API wiring |
| #399 | `feat/decision-plan-exporters` | Decision plan exporters (Jira, Linear, webhooks) |
| #398 | `feat/inbox-auto-debate` | Inbox auto-debate router for high-priority triage |
| #396 | `feat/ceo-mode-extended` | CEO-mode for non-developer swarm experience |
| #395 | `fix/pipeline-receipt-test-v2` | Pipeline receipt test mock fix |
| #394 | `fix/benchmark-noise-floor` | CI benchmark noise floor gate |
| #393 | `feat/landing-competitive-moats` | Landing page competitive moats |
| #392 | `worktree-regen-openapi-spec` | Regenerate OpenAPI spec for fleet endpoints |
| #390 | `fix/hardening-and-polish` | Hardening, swarm CEO-mode, doc polish |
| #387 | `fix/handler-can-handle` | Default can_handle() for handler base classes |
| #386 | `fix/normalize-text-9px-to-10px` | Normalize text sizes across frontend |
| #385 | `feat/ui-polish-and-handler-improvements` | Standardize font sizes, improve swarm UX |
| #382 | `chore/landing-swarm-tune` | Landing copy sharpening, swarm defaults |
| #381 | `fix/ci-stabilization-and-frontend-polish-20260226` | CI stabilization + frontend fixes |
| #378 | `fix/pipeline-gates-shadow-tokens` | Pipeline persistence, nomic gates, shadow token alignment |
| #377 | `fix/e2e-workflow-cleanup` | Deduplicate Playwright E2E steps |

---

### Recently Completed (Last 7 Days)

| Date | Agent | Task | PR | Commit |
|------|-------|------|----|--------|
| 2026-02-26 | Claude/Codex | Mypy reporter + pipeline CLI type fixes | #397 | f44d079 |
| 2026-02-26 | Codex | CI integration gate: broadened event exception handlers | #391 | d926330..8e72b23 |
| 2026-02-26 | Codex | Pipeline self-improve bridge: planning output into execution handoff | #383 | 9b75de4 |
| 2026-02-26 | Codex | Fleet control plane: session status, ownership claims, merge queue | #370 | 9107477 |
| 2026-02-26 | Codex | E2E stabilization: auth/workflow/email golden path tests | #372 | e83d035 |
| 2026-02-26 | Codex | Self-host runtime: readiness fallback, auth-gate, port 0 fixes | #375, #376, #379 | various |
| 2026-02-25 | Claude/Codex | Settlement hooks, secrets sync, runtime hardening | #365 | a7d55b7 |
| 2026-02-25 | Claude/Codex | Swarm commander CLI + requirement-spec workflow | #364 | ec6cf95 |
| 2026-02-25 | Claude/Codex | Vercel frontend deploy migration, debate routes | #360, #363 | 7398db3, 84647800 |
| 2026-02-25 | Claude/Codex | Mode/settlement metadata propagation in live UI | #356 | b0838fc |
| 2026-02-25 | Codex | Weekly epistemic KPI extraction workflow | #354 | c795123 |
| 2026-02-25 | Codex | Release readiness gate + worktree hygiene checks | - | 7492cf7 |
| 2026-02-24 | Claude/Codex | Mypy zero errors (318 errors eliminated across 60+ files) | - | various |
| 2026-02-24 | Claude/Codex | All 36 push-triggered workflows: concurrency groups | - | various |

---

## Domain Ownership

To avoid conflicts, agents should stay within their assigned domains:

```
Session 1 claims: aragora/connectors/
Session 2 claims: aragora/server/handlers/
Session 3 claims: tests/
```

### Current Claims

*No domains currently claimed*

---

## Issue Priority by Track

> All original GA issues (#88-#108) are **CLOSED**. Current work is organized around epics.

### Active Epics

**Product & UX:**
- [ ] #292 [EPIC] Debate Engine Integration & Marketplace Synergy *(debate, stabilization)*
- [ ] #293 [EPIC] Enterprise Communication Hub & Active Triage *(integration, stabilization)*
- [ ] #323 Integrate FastAPI v2 marketplace/orchestration routes

**Developer Experience:**
- [ ] #297 [EPIC] SDK Parity, Golden Paths & Developer Experience *(documentation, sdk)*
- [ ] #294 [EPIC] Idea-to-Execution Pipeline & Safe Code Execution *(pipeline, stabilization)*

**Security & Compliance:**
- [ ] #273 [EPIC] Enterprise Assurance Closure *(priority:critical, security)*
- [ ] #274 Execute external penetration test and remediate findings *(priority:critical, pentest)*
- [ ] #296 [EPIC] Compliance Dashboard & Verifiable Decision Receipts *(security, compliance)*

**Self-Improvement:**
- [ ] #295 [EPIC] Nomic Loop Safety Gates & Observable Evolution *(self-improvement, stabilization)*

### Current Focus Areas (Feb 26)

| Priority | Area | Active PRs |
|----------|------|------------|
| **P0** | User-facing product quality (UX, shareability) | #403, #400, #393, #396 |
| **P0** | Landing page + onboarding critical path | #400, #382, #385, #386 |
| **P1** | Pipeline persistence + self-improve bridge | #378, #383, #395 |
| **P1** | CI stabilization + E2E test reliability | #381, #394, #377 |
| **P2** | Obsidian sync, decision plan exporters | #401, #399 |
| **P2** | OpenAPI spec regeneration, handler base classes | #392, #387 |

---

## How to Use This File

### Starting a Session

1. Check "Currently Active" - avoid working on same files
2. Add your session using the template
3. Claim a domain if doing substantial work
4. Reference an issue number if applicable

### During Work

- Update status if blocked
- Note any files you unexpectedly needed to modify
- If you need files someone else claimed, coordinate first

### Finishing a Session

1. Move your entry to "Recently Completed"
2. Release any domain claims
3. Note the commit hash
4. Update issue status on GitHub if applicable

---

## Conflict Resolution

If you encounter merge conflicts or overlapping work:

1. **Don't force push** - you may overwrite others' work
2. **Pull latest:** `git pull origin main`
3. **Check this file** - see who was working on conflicting files
4. **Ask Claude to resolve** - AI is good at semantic merges
5. **Run tests:** `pytest tests/ -x --timeout=60`

---

## Communication Shortcuts

When starting a Claude session, paste this:

```
Check .claude/COORDINATION.md for active work by other agents.
Before making changes, tell me your plan.
Stay within: [YOUR ASSIGNED DOMAIN]
Update COORDINATION.md when done.
```

---

## Test Commands

Quick validation before committing:

```bash
# Fast check (2 min)
pytest tests/ -x --timeout=60 -q -m "not slow"

# Full suite (10+ min)
pytest tests/ --timeout=120

# Specific area
pytest tests/server/ -v --timeout=60
pytest tests/connectors/ -v --timeout=60
```

---

## Autonomous Orchestration (Experimental)

Aragora can orchestrate its own development using the `AutonomousOrchestrator`:

```python
from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

orchestrator = AutonomousOrchestrator()

# Execute a high-level goal
result = await orchestrator.execute_goal(
    goal="Maximize utility for SME SMB users",
    tracks=["sme", "qa"],
    max_cycles=5,
)

# Or focus on a specific track
result = await orchestrator.execute_track(
    track="developer",
    focus_areas=["SDK documentation", "API coverage"],
)
```

**Components:**
- `AgentRouter`: Routes subtasks to appropriate agents based on domain
- `FeedbackLoop`: Handles verification failures with retry/redesign logic
- `TrackConfig`: Defines folders, protected files, and agent preferences per track

**Safety Features:**
- Domain isolation prevents file conflicts
- Core track limited to 1 concurrent task
- Approval gates for dangerous changes
- Checkpoint callbacks for monitoring

See `aragora/nomic/autonomous_orchestrator.py` for full API.

---

## Recent Patterns to Follow

Based on recent commits, follow these patterns:

- **Commit messages:** `type(scope): description` (e.g., `fix(tests): add mock`)
- **Co-author:** Add `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- **Test before commit:** Always run tests
- **Small commits:** One logical change per commit
