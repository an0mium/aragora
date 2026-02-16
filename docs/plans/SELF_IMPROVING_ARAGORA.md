# Self-Improving Aragora: Strategic Plan

## Vision

Aragora accepts a vague natural-language goal, decomposes it into actionable work,
coordinates heterogeneous AI agents across isolated worktrees, produces tested code,
merges it, and learns from the results. The market gap is not in individual agent
capability but in **coordination of many heterogeneous agents** working on a shared
codebase efficiently and safely.

## Current State (Feb 2026)

The infrastructure is **80% complete**. The missing 20% is connective tissue:

| Component | Status | Gap |
|-----------|--------|-----|
| TaskDecomposer | Works (debate + heuristic) | No learning from past decompositions |
| AutonomousOrchestrator | Works (routes to agents) | HardenedOrchestrator not default, no git commit |
| HybridExecutor | Works (Claude/Codex CLI) | No worktree awareness |
| WorktreeManager | Exists (not wired) | Opt-in only, never default |
| BranchCoordinator | Exists (partial) | Conflict resolution is detection-only |
| MetaPlanner | Exists (dead-end) | Never called by orchestrator |
| OpenClaw | Exists (isolated) | Not connected to orchestration |
| Prompt injection | Zero implementation | Critical for autonomous operation |
| Frontend | Strong (4-level progressive) | Minor UX polish |

## Architecture Target

```
Human: "Maximize utility for SMEs"
    ↓
MetaPlanner.prioritize_work()          ← WIRE IN (currently dead-end)
    ↓
TaskDecomposer.analyze_with_debate()   ← WORKS
    ↓
HardenedOrchestrator.execute_goal()    ← MAKE DEFAULT
    ↓ (per subtask)
WorktreeManager.create_worktree()      ← WIRE IN (currently opt-in)
    ↓
HybridExecutor.execute_in_worktree()   ← ADD worktree param
    ↓
VerificationStep.run_tests()           ← WORKS
    ↓
BranchCoordinator.merge_if_passing()   ← ADD auto-merge
    ↓
AuditReconciliation.cross_check()      ← ADD second-agent review
    ↓
KnowledgeMound.record_outcome()        ← ADD cross-cycle learning
```

## Phase 1: Wire the Gold Path ~~(Week 1)~~ COMPLETE

**Goal:** `self_develop.py --goal "..." --autonomous` produces tested, committed code.

### 1A. HardenedOrchestrator as Default ✅
- `use_worktree_isolation=True` by default
- Prompt injection scanning via SkillScanner on all goal inputs
- Budget tracking with `budget_limit_usd` and `BudgetEnforcementConfig`

### 1B. Git Commit in Orchestrator ✅
- Auto-stage + commit with structured message (subtask ID, track, agent)
- Push to worktree branch (not main)
- Revert on test failure

### 1C. MetaPlanner → Orchestrator Bridge ✅
- `execute_goal_coordinated()` calls `MetaPlanner.prioritize_work()`
- `quick_mode=True` for concrete goals, debate for vague goals
- Cross-cycle learning via KnowledgeMound `NomicCycleAdapter`

### 1D. Merge Gate ✅
- `_run_merge_gate()` runs scoped pytest in worktree before merge
- BranchCoordinator merges on success, rejects on failure
- Merge outcome recorded in KnowledgeMound

## Phase 2: Security Hardening ~~(Week 1-2)~~ COMPLETE

**Goal:** Safe autonomous execution even with untrusted inputs.

### 2A. Prompt Injection Defense ✅
- SkillScanner input sanitization before TaskDecomposer
- Canary tokens in system prompts (`get_canary_directive()`)
- Output validation scans diff for canary leaks + dangerous patterns
- `_check_canary_leak()` detects system prompt leaks

### 2B. Code Review Gate ✅
- `_run_review_gate()` scores diff (0-10), blocks below `review_gate_min_score`
- `_cross_agent_review()` selects a DIFFERENT agent via `_select_best_agent(exclude_agents=...)`
- OpenClaw SkillScanner scans generated code for dangerous patterns

### 2C. Sandbox Execution ✅
- `_run_sandbox_validation()` validates Python syntax via `py_compile`
- Docker sandbox via `SandboxExecutor` when available
- Configurable timeout via `sandbox_timeout`

## Phase 3: Multi-Agent Coordination (Week 2-3)

**Goal:** 6-12 agents working on different subtasks in parallel worktrees.

### 3A. Agent Pool Manager
- Assign agents from heterogeneous pool (Claude, GPT, Gemini, Codestral)
- Match agent capabilities to subtask requirements
- Track per-agent success rates and adjust assignments
- Circuit breaker per agent (3 failures → exclude for 30 min)

### 3B. Cross-Agent Review
- Agent A implements, Agent B reviews, Agent C verifies
- No agent reviews its own output
- Disagreements trigger debate (existing Arena infrastructure)
- Consensus required for merge (>= 2 of 3 agents agree)

### 3C. Work Stealing
- If an agent's worktree completes early, steal pending work from queue
- Priority: blocked tasks that just became unblocked
- Never steal partially-completed work

## Phase 4: OpenClaw Integration (Week 3-4)

**Goal:** Autonomous computer use for tasks that require it.

### 4A. Orchestrator → OpenClaw Bridge
- Add "computer_use" execution mode to orchestrator
- Route UI tasks (Playwright tests, visual regression) through OpenClaw
- Route code tasks through HybridExecutor (existing)
- Decision criteria: task mentions "browser", "UI", "visual", "click"

### 4B. Secure Execution
- OpenClaw actions sandboxed in Docker
- Network allowlist (only project URLs)
- Screenshot logging for audit trail
- Budget caps per computer-use session

## Phase 5: Self-Improvement Loop (Week 4+)

**Goal:** Aragora improves itself via the Nomic Loop.

### 5A. Dogfooding Pipeline
- Run `self_develop.py --goal "Fix all failing tests" --autonomous` nightly
- Run `self_develop.py --goal "Improve test coverage" --autonomous` weekly
- Human reviews PRs before merge to main

### 5B. Cross-Cycle Learning
- Record every orchestration: goal, decomposition, assignments, outcomes
- Feed successful patterns into next decomposition
- Feed failures into "avoid this" context
- Use KnowledgeMound for persistent storage

### 5C. Calibration Feedback
- Track Brier scores for agent predictions vs. actual outcomes
- Weight agent selection by calibration score
- Surface calibration data in ERC-8004 reputation

## Worktree Setup Guide

### For Your Current Sessions

```bash
# Create 6 worktrees (one per track)
./scripts/setup_worktrees.sh 6

# Start Claude Code in each:
cd ../aragora-worktrees/orchestration && claude
cd ../aragora-worktrees/security && claude
cd ../aragora-worktrees/frontend && claude
cd ../aragora-worktrees/testing && claude
cd ../aragora-worktrees/integration && claude
cd ../aragora-worktrees/sdk && claude
```

### Assignment Per Worktree

| Worktree | Focus | First Task |
|----------|-------|------------|
| orchestration | Phase 1A-1D | Wire HardenedOrchestrator as default |
| security | Phase 2A-2C | Prompt injection defense layer |
| frontend | UX polish | Mode selector visibility, landing page CTA |
| testing | Test health | Fix remaining flaky tests, CI completion |
| integration | Phase 4A-4B | OpenClaw → orchestrator bridge |
| sdk | SDK parity | Contract tests, TypeScript sync |

### Merging Back

```bash
# After sessions complete:
./scripts/merge_worktrees.sh --dry-run  # Preview merges
./scripts/merge_worktrees.sh            # Execute merges
```

### Avoiding Drift

- **Merge frequency:** Every 4-6 hours, merge completed worktrees to main
- **Rebase often:** Each worktree should `git fetch origin && git rebase main` before starting new work
- **Non-overlapping files:** Assign different directories per worktree
- **CI must complete:** Don't merge until tests pass in the worktree

## When Can Aragora Do This Job?

### Today (with human coordination):
- `self_develop.py --goal "..." --debate --dry-run` → see the decomposition
- `self_develop.py --goal "..." --require-approval` → human approves each step
- Manual worktree setup + manual merge

### After Phase 1 (~1 week):
- `self_develop.py --goal "..." --autonomous` → tested, committed code in worktrees
- `merge_worktrees.sh` → automated merge with test gates

### After Phase 3 (~3 weeks):
- Multi-agent parallel execution in worktrees
- Cross-agent review (no agent reviews its own code)
- Automated merge with consensus gate

### After Phase 5 (~1 month):
- Nightly self-improvement runs
- Learning from past cycles
- Calibrated agent selection
- Full autonomous loop with human review of PRs

## Key Insight

The market gap is NOT in code generation (every lab has that). It's in:

1. **Coordination** — assigning non-overlapping work to heterogeneous agents
2. **Isolation** — worktrees/containers so agents don't conflict
3. **Verification** — test gates before merge, cross-agent review
4. **Learning** — recording outcomes and improving decomposition
5. **Safety** — prompt injection defense, sandbox execution, audit trails

Aragora has infrastructure for all five. The work is wiring them together.
