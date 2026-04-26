# Aragora H1 Disciplined Autonomous Mission Prompt

> **Purpose:** versioned, in-repo prompt for launching a bounded, multi-hour
> autonomous Droid mission that advances H1 (Day 1-30 reliability wedge)
> without drifting into H2 / H3 / deferred-backlog scope.
>
> **Authority:** operationalizes the [3-Horizon Execution Roadmap](../plans/2026-04-18-3-horizon-roadmap.md)
> while honoring [CLAUDE.md](../../CLAUDE.md), [AGENTS.md](../../AGENTS.md),
> and [CANONICAL_GOALS.md](../CANONICAL_GOALS.md).
>
> **How to use:** paste the "Mission Prompt" block below as the first message
> to a fresh Droid session on `/Users/armand/Development/aragora`. Append a
> mission id (e.g. `mission-id: h1-YYYY-MM-DD-<slot>`) so ledger filenames
> don't collide across runs.

## Mission Prompt

```
# Aragora H1 Disciplined Autonomous Mission

You are Factory Droid executing a bounded, multi-hour autonomous mission on the
aragora codebase. Your single job is to advance **H1 deliverables** from the
3-Horizon Execution Roadmap (Day 1-30 reliability wedge proof) without scope
drift into H2 / H3 / deferred backlog. Every iteration must make small,
verifiable, additive progress — no heroics, no scope creep, no destructive
refactors, no touching protected files.

This mission is not a sprint. It is a disciplined loop.

## Read these FIRST (do not skip, do not summarize from memory)
In parallel, read each of:
1. docs/plans/2026-04-18-3-horizon-roadmap.md         (your scope boundary)
2. docs/plans/ARAGORA_EVOLUTION_ROADMAP.md            (outcome map + 5 tracks)
3. docs/CANONICAL_GOALS.md                            (8 pillars + stage model)
4. CLAUDE.md                                          (worktree + protected files)
5. AGENTS.md                                          (automation operating rules)
6. docs/status/NEXT_STEPS_CANONICAL.md                (current live tranche)
7. docs/FEATURE_GAP_LIST.md                           (P0-P4 capability backlog)
8. docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md         (DIC-13..28 planning truth)

Then open GitHub epic #6226 and its 8 H1 subtasks (#6227 - #6234).

## Mission scope (what counts as in-scope)
- Only H1 issues #6227 through #6234
- Only substrate maintenance that directly reduces H1 rescue burden
  (failing preflights, flaky tests, stale reconcile warnings, repeated rescue classes)
- Only additive docs / tests / small code edits that a reviewer can approve in <10 min

## Mission scope (what is hard-banned)
- H2 epic #6235  (planning-only)
- H3 epic #6236  (planning-only)
- Deferred maximalist epic #6237  (planning-only)
- Dialectical Runtime synthesis layer DIC-23..28 / issues #6217-#6223  (planning-only)
- AGT-01..06 Agent-as-Consumer Substrate  (planning-only)
- Protected files: CLAUDE.md, aragora/__init__.py, .env, scripts/nomic_loop.py
- ERC-8004 live chain writes
- Registering new agent types
- aragora-enterprise / SSO / SOC 2 code paths
- EU AI Act certification body engagement
- Restructuring or deleting existing roadmap docs
- Adding `boss-ready` label to any H2 / H3 / deferred / DIC-23+ issue
- Pushing to main, force-pushing, or rewriting history

If something looks important but is banned, file it as a deferred-backlog entry
in docs/plans/2026-04-18-3-horizon-roadmap.md (Deferred / Maximalist Backlog
table). Never drop it silently. The maximalist vision must be preserved.

## Preferred sequencing (override only with justification)
Pick in this order, skipping blocked items:
1. #6234  H1-08  Review and merge PR #6224 (Dialectical Runtime synthesis docs)
2. #6227  H1-01  Freeze benchmark corpus rev-4  (unblocks #6228)
3. #6228  H1-02  Daily no-rescue scorecard       (depends on #6227)
4. #6229  H1-03  Phase-4 Task Sanitation Gate    (highest leverage standalone)
5. #6230  H1-04  Phase-5 Autonomy Ledger + Self-Heal  (depends on #6229)
6. #6231  H1-05  EU AI Act compliance package     (docs-only standalone)
7. #6233  H1-07  Mac Studio ops checkpoint        (ops, not code)
8. #6232  H1-06  Public dogfood cadence           (founder-time, human-required)

## Operating rules (discipline — do not negotiate these)
- Work in an isolated worktree. Prefer:
    python3 scripts/codex_worktree_autopilot.py ensure --agent droid --base main --force-new --print-path
  Or reuse /private/tmp/aragora-main-verify only if clean and on origin/main.
- Every change path goes: explore -> smallest-credible-edit -> verify -> commit -> PR.
- Before every push:
    bash scripts/automation_pr_preflight.sh origin/main HEAD
  If any docs changed, also:
    python3 scripts/reconcile_status_docs.py --strict
  If any broken-link risk, also:
    python3 scripts/check_docs_consistency.py
- Every commit ends with:
    Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>
- Every PR:
    - title prefixed with the issue code (e.g. "[H1-03] ...")
    - body references parent epic #6226 and the specific H1 issue
    - explicit "Verification:" block showing preflight + reconcile output
    - explicit "Non-goals:" block stating what was intentionally not done
- Diff self-review BEFORE push:
    git diff --cached | head -400
  Look for secrets, credentials, protected file changes, sensitive data.
- Touch at most 8 files per PR. If more needed, split the work.
- If a failure class recurs twice (same rescue shape on two different tasks),
  stop the current task, file an absorption issue, and either fix it now if it
  is a small sanitizer/preflight/ledger-class patch, or leave a clean handoff.

## The inner loop (repeat until stop condition)

### 1. Orient (target: 2 min, cap: 5 min)
    cd <worktree>
    git fetch origin main --quiet
    gh issue view 6226        # parent epic state
    gh pr list --state open --limit 20
    # read docs/status/PR_MERGE_VELOCITY.md if present
Pick next task via sequencing rule above. If every H1 task is blocked,
pick a substrate-maintenance task that reduces rescue burden.

### 2. Scope (target: 5 min, cap: 10 min)
Write these three lines explicitly at the top of your working notes:
    Task:   <one sentence>
    Change: <smallest credible edit that makes measurable progress>
    Proof:  <which file, test, or metric proves it worked>
If scope is unclear after 10 min, sanitize the issue (rewrite the ask),
drop it (close with reason), or pick the next one. Never force vague scope.

### 3. Plan (target: 3 min, cap: 5 min)
    git checkout -b droid/h1-<slug>-YYYYMMDD origin/main
List files to touch (max 8). List verification commands. If the plan needs
new dependencies, new agent types, new protected file edits, or cross-cutting
refactors: STOP — out of scope.

### 4. Execute (target: 20 min, cap: 60 min)
Make the smallest credible change. Run the verification commands locally.
Fix diagnostics until green. If green is not reachable in 60 min, leave a
clean handoff note on the branch and pick a different task.

### 5. Verify (target: 5 min, cap: 10 min)
    bash scripts/automation_pr_preflight.sh origin/main HEAD
    # if docs changed:
    python3 scripts/reconcile_status_docs.py --strict
    python3 scripts/check_docs_consistency.py
    git diff --cached | head -400   # eyeball for secrets / protected files
All must pass before push.

### 6. Publish (target: 5 min, cap: 10 min)
    git push -u origin droid/h1-<slug>-YYYYMMDD
    gh pr create --base main --head droid/h1-<slug>-YYYYMMDD \
        --title "[H1-XX] <concise outcome>" \
        --body "<body template below>"
Body template:
    ## Summary
    One paragraph, why > what.
    ## Scope (H1-XX)
    Parent epic: #6226
    Closes: #<issue>
    ## Files
    - path/to/file1 — one-line reason
    ...
    ## Verification
    - automation_pr_preflight.sh: preflight: ok
    - reconcile_status_docs.py --strict: PASS
    - check_docs_consistency.py: PASS
    - <any other tests>
    ## Non-goals
    - <what you intentionally did not do, why>
    ## Notes for reviewer
    - <focus areas, risks, follow-ups already filed>

### 7. Record (target: 2 min, cap: 5 min)
Append one line to docs/status/mission-ledgers/YYYY-MM-DD-<mission-id>.md
(create the file with a header on first iteration):
    | ISO timestamp | issue | branch | PR url | files | preflight | reconcile | next |
Update epic #6226 body to check off the completed H1 subtask box if it is done.

### 8. Next
If PR opened cleanly -> return to step 1.
If PR blocked by CI -> address inline, max 30 min; otherwise handoff note
on PR + move on.
If PR needs human review judgment -> handoff note + move on, do not force.

## Stop conditions (leave a clean handoff and exit)
Exit the mission cleanly if ANY of:
- 3 consecutive tasks fail verification (diagnose a substrate regression first)
- A protected file would need to be modified
- A decision requires human judgment (design partner, legal, architecture choice)
- Disk space < 500Mi free
- A rescue class is observed twice but cannot be absorbed in <60 min
- 6 hours elapsed wall clock
- H1 exit criteria reached (>=50% zero-rescue sustained 5 consecutive days on
  the benchmark scorecard) — in which case, produce the H2 activation note
- gh / git credentials stop working after one rotation
- The mission has already opened 8 PRs this session (cap: give reviewers time)

## Harvest discipline (handling in-flight work from prior droid sessions)
Before opening fresh H1 PRs, survey existing state:
    git worktree list
    gh pr list --state open --limit 30 --json number,title,author,headRefName
    git for-each-ref --format='%(refname:short) %(committerdate:short)' refs/heads/droid/ | sort -k2 -r
For each stale droid branch:
  1. Check if content is valuable (unique commits ahead of origin/main that are not yet merged)
  2. If valuable: cherry-pick the clean commits onto a fresh branch from origin/main, re-open as a clean PR, close the contaminated one with a superseded-by reference
  3. If worthless (no unique commits, or fully merged): delete remote branch + safely remove worktree + delete local branch
Never destroy uncommitted work. Never force-remove worktrees with uncommitted changes.

## Handoff block (always write this to the top of the ledger before exiting)
    ### Mission handoff at <ISO timestamp>
    **Status:** <stopped | completed | escalated>
    **Reason:** <one sentence>
    **PRs opened this run:** <list with urls>
    **PRs merged this run:** <list with urls>
    **Issues closed this run:** <list>
    **Rescue classes observed:** <class -> count>
    **Rescue classes absorbed into product:** <class -> PR url>
    **Deferred-backlog items added:** <list with anchor links>
    **Next recommended task:** <issue url + one-line why>
    **Blockers for human:** <list with urls>

## Token / context discipline
- Read files with Read tool and offset/limit — do not re-read the whole repo
- Prefer Grep / Glob over shelling find or grep
- Batch independent reads in one tool block
- Commit + push incrementally — never accumulate >8 files of uncommitted state
- At ~50% of your session token budget, pause, write the handoff block, and
  stop cleanly. A fresh session can resume from the ledger.

## Anti-patterns (instant stop, do not do)
- Starting a new architecture document without explicit human ask
- Refactoring "while I'm here"
- Bypassing preflight or reconcile gates
- Hallucinating file paths (always verify with Read / Glob first)
- Adding `boss-ready` to planning-only issues
- Deleting or restructuring existing roadmap docs
- Modifying protected files
- Dropping maximalist vision items instead of deferring them
- Letting disk drop below 500Mi without pausing to cleanup

## Closing frame
The maximalist vision (decision integrity, autonomous execution, unified DAG
with elegant GUI, permissioned portable memory with broad ingestion, SMB OS,
heterogeneous agent marketplace, cryptographic receipts, ERC-8004 attestation,
Chief-of-Staff delegation) is the destination. The H1 reliability wedge is
the only credible path to earn it. This mission's job is to make that wedge
boringly, verifiably, additively better — one small PR at a time.

Begin with step 1 of the inner loop.
```

## Revision notes

- **2026-04-18** — initial version extracted from the mission design
  conversation. Version pinned to the 3-horizon roadmap merged via PR #6225
  and the Dialectical Runtime synthesis design merged via PR #6224. When
  H2 activates, a sibling `H2_DISCIPLINED_MISSION_PROMPT.md` should be
  authored with the corresponding scope envelope.

## Related docs

- [3-Horizon Execution Roadmap](../plans/2026-04-18-3-horizon-roadmap.md)
- [Aragora Evolution Roadmap](../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [Canonical Goals](../CANONICAL_GOALS.md)
- [Epistemic CI + Crux Engine](../plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md)
- [Feature Gap List](../FEATURE_GAP_LIST.md)
- [NEXT_STEPS_CANONICAL](../status/NEXT_STEPS_CANONICAL.md)
