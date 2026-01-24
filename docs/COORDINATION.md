# AI Agent Coordination

**Last updated:** 2026-01-24
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
- **Status:** 🟡 In Progress / 🔴 Blocked / 🟢 Almost Done

-->

*No active sessions recorded*

---

### Recently Completed (Last 7 Days)

| Date | Agent | Task | Issue | Commit |
|------|-------|------|-------|--------|
| 2026-01-24 | Claude | SentenceTransformer mock for tests | - | 520adee |
| 2026-01-24 | Claude | OAuth prefix test fix | - | a0b7f27 |

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

### P0 - Must Do (Blocking Release)

**SME Track:**
- [ ] #100 SME starter pack GA documentation
- [ ] #99 ROI/usage dashboard
- [ ] #92 RBAC-lite for workspace members
- [ ] #91 Workspace admin UI

**Developer Track:**
- [ ] #103 API coverage tests
- [ ] #102 SDK parity pass #2
- [ ] #94 SDK docs portal landing page

**Self-Hosted Track:**
- [ ] #106 Production deployment checklist
- [ ] #105 Self-hosted GA sign-off
- [ ] #96 Backup and restore scripts

**QA Track:**
- [ ] #107 E2E smoke tests
- [ ] #90 Integration test matrix

### P1 - Should Do

- [ ] #108 Nightly CI smoke test runs
- [ ] #104 Developer portal GA
- [ ] #101 User feedback collection
- [ ] #98 Automated changelog generation

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
- **Co-author:** Add `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`
- **Test before commit:** Always run tests
- **Small commits:** One logical change per commit
