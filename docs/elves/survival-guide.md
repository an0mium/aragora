# ARCHIVAL SURVIVAL GUIDE — CLOSE-THE-LOOP RUN

> This was the Survival Guide for the 2026-07-01 close-the-loop run. It is preserved as a
> historical run artifact, not a current operating runbook. After compaction, read it to
> understand the run, then verify live repo/owner state before acting. If this file conflicts
> with later entries in `docs/elves/execution-log.md`, the later execution-log entry supersedes
> it.
>
> Read order after compaction: this file → `.elves-session.json` → `docs/elves/learnings.md` →
> the plan → `docs/elves/execution-log.md` → `docs/AGENT_OPERATING_CONTRACT.md` →
> `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.

---

## Mission

Executed the Close the Loop strategic plan (`docs/plans/2026-07-01-strategic-plan-close-the-loop.md`,
epic #8762): drain the stuck ready-PR queue, build the mission intake bridge, merge executor, and
harvest engine, and start the ODR compliance wedge — all under the aragora validation gate
(local truth → adversarial review → verified receipt → tier settlement). Invariants: never merge
by default; Tier 3-4 prepare-only; no shared-root mutation; no edits to
`aragora/swarm/quorum_evidence.py` or `aragora/cli/commands/review_queue.py` during the run while
the Codex timeout-family lane owned them. Future operators must re-check live ownership/steering
instead of treating this archival freeze as current.

---

## Run Control

- **Run mode:** open-ended
- **Stop policy:** blocker-only (operator said "as much as possible"; stop when all batches done or blocked on human settlement)
- **User intent:** "I approve G1/G2 sign-off on the queue-drain cleanup, and the arming decision for the Tier 0-2 merge executor once #8759 demonstrates its dry-run. and all kinds of other good things to improve the repo as much as possible" (recorded at https://github.com/synaptent/aragora/issues/8762#issuecomment-4860561852)
- **Checkpoint due by:** none (post progress comments on epic #8762 at each batch close)
- **Checkpoint semantics:** delivery target only
- **May continue after checkpoint:** yes
- **Actual stop conditions:** all batches complete or parked/awaiting settlement; operator stop; operating-contract auto-halt
- **Workspace ownership:** dedicated worktree `.claude/worktrees/elves-close-the-loop-20260701`, branch `elves/close-the-loop-20260701` — coordinator-owned. Batch code ships from per-batch branches/PRs; this branch holds run artifacts only.
- **Branch tip at start (collision tripwire):** 7439a1466f6b906a72c2b423486364eb9abf9b4b (origin/main at staging)
- **Merge policy:** user-merges (default). NO merge-on-green preference recorded. The Tier 0-2 merge *executor* (#8759) is a deliverable, not a license: its arming is Tier 4 with conditional pre-approval (see Settlement notes).
- **Highest tier auto-settle allowed:** Tier 2
- **Final-response policy:** allowed at batch closes; keep going while unblocked work remains
- **Gate attempt cap:** 2 full gate cycles per batch, then park
- **Parallel fan-out allowance:** max 2 concurrent lanes; only file-disjoint batches (B2 `aragora/missions/` vs B3/B4 `scripts/` are disjoint). Lane ledger at `.aragora/run-close-the-loop-20260701/lanes/`.
- **Per-batch wall-clock budget:** 90 min implement+gate; **per-phase:** no batch un-closed/un-parked for 2 h → no-progress halt
- **External-wait pacing:** PR CI ≈ every 4-5 min (verify workflows actually started on first check; `gh run rerun` if never ran); quorum evidence ≈ every 15 min. Foreground bounded until-loops only — no background watchers.
- **Receipt dir:** `.aragora/run-close-the-loop-20260701/receipts/`
- **Batch completion rule:** complete only after gate steps 1-7 pass and the closing commit+push lands.
- **Re-read rule:** immediately after every commit and push, re-read this survival guide.

---

## Stop Gate

- **Planned batches remaining:** 0 unblocked (B1,B3,B4,B6 done; B2 parked; B5 packet delivered; B7,B8 skipped w/ reasons)
- **Batches blocked on human settlement (Tier 3-4):** none for this archival run. Later execution-log entries record the operator-only residuals and supersede earlier queue snapshots.
- **Stop allowed right now:** yes
- **Why:** every remaining item is blocked on human settlement, operator decisions, or external CI.
- **Next required action:** none from this file. Use the latest execution-log section and live GitHub/repo state before choosing any new action.

---

## Forbidden Stop Reasons

Not valid reasons to stop while unblocked work remains: a checkpoint time was reached; a commit/push
succeeded; CI is green; a receipt was written; the user is silent; you wrote a summary; the current
batch is done but later batches remain; "this is a lot for one turn"; "this feels like a natural
place to check in". Update docs, commit, push, re-read this file, continue.

A genuine Tier 3-4 human-settlement block **is** a valid pause for *that batch only*.

---

## Non-Negotiables

- **Never merge by default. Never approve a merge.** Tier 3-4 requires explicit human risk
  acceptance before counting as landed.
- **Every batch produces a verified `DecisionReceipt`.** No receipt → not done.
- **Unresolved adversarial dissent blocks the batch** (parking is the legal disposition after the attempt cap).
- **Never modify a test to make it pass.** Total test count never decreases.
- **Respect approval-required surfaces and auto-halts** (operating contract; gate doc).
- One coordinator owns this run. Surprise tip move on an owned branch = collision → stop that lane.
- No destructive git: `reset --hard`, `checkout .`, `clean -fd`, `push --force`, rebase on shared.
- Stage specific files; never `git add -A`. Scope ≤800 LOC delta per batch.
- Closing commits carry `Co-authored-by: claude[bot]`.
- **Project-specific:** (1) During this run, do not touch `aragora/swarm/quorum_evidence.py`,
  `aragora/cli/commands/review_queue.py`, or PRs #8726/#8720 because an active Codex conductor
  lane owned the timeout family. Future work must re-check live owner/steering state rather than
  inheriting this archival freeze. (2) Check mailbox/owner state (`scripts/agent_bridge` / operator steering)
  before claiming ANY existing PR in the drain campaign. (3) Shared root
  (the main repo checkout, `git rev-parse --show-toplevel` of the primary clone) is read-only —
  it is behind origin/main with untracked dirt.
  (4) Evidence collection: use the repo review path
  (`python3 -m aragora.cli.main review-queue collect-evidence --reviewers claude openai` with all
  3 CI flags honored; OpenRouter fallback per `ARAGORA_ENABLE_OPENROUTER_REVIEWER_FALLBACK=1`).
  Local raw API keys are intentionally absent — never write keys into env or files.

---

## Batch Plan (pre-classified tiers)

| # | Batch | Surface | Tier | Autonomy |
|---|---|---|---|---|
| B1 | Drain campaign wave 1 (#8761): triage ready PRs >7d, one at a time — refresh stale heads where staleness caused CI failure, run bounded quorum evidence, settle Tier 0-2 that pass, prepare packets for Tier 3-4 (#8713, #8519), file advisory P2/P3 as follow-up issues, park honestly | existing PRs + evidence tooling (read/execute only) | 2 | auto per-PR; Tier 3-4 PRs → packets |
| B2 | Mission intake→decomposition bridge (#8758): dispatch.py intake path calls TaskDecomposer, branch-backed Features, tests | `aragora/missions/`, `tests/missions/` | 2 | auto |
| B3 | Merge executor (#8759): `scripts/merge_executor.py` composing auto_merge_quorum_green + settle_plan, dry-run default, receipts, auto-halt-on-main-red; demonstrate dry-run | `scripts/`, `tests/` | 2 (impl) / **4 (arming)** | impl auto; **arming = human step with dry-run evidence (conditionally pre-approved, must present evidence + get final confirm)** |
| B4 | Harvest engine (#8760): recurring classifier over closed/merged/parked PRs + stale branches, WIP-capped issue creation, drain ledger, dry-run default | `scripts/`, `aragora/nomic/` (compose only) | 2 | auto |
| B5 | Adjudicator wiring design packet (#8748): design + exact diff proposal for wiring ReviewAdjudicator into the quorum stall path | merge/evidence authority | **4** | **prepare-only — do NOT implement autonomously**; queue packet |
| B6 | Queue-drain cleanup execution (G1/G2 GRANTED): follow `docs/plans/2026-06-30-queue-drain-diagnosis-and-cleanup-plan.md` exactly — archive (tags/bundles + restore manifest) BEFORE any deletion; close classified-churn PRs with recorded rationale; delete only fully-archived orphaned branches | branches/PRs (destructive, reversible-by-design) | **4 (pre-approved)** | execute per plan's own gates; any branch whose commits are not verifiably archived → skip and log |
| B7 | ODR-1 receipt schema (#8223): vendor-neutral JSON Schema + JCS canonicalization, docs + additive code | `aragora/gauntlet/`/`docs/` | 1 | auto |
| B8 | ODR-3 offline verifier (#8223): pip-installable `aragora-verify` package skeleton + real receipt fixture test (check #8389 outcome from B1 first — may supersede) | new package dir | 2 | auto |

Order: B1 → (B2 ∥ B3) → B4 → B6 → B7 → B8; B5 packet whenever convenient. Re-check issue/PR
ownership immediately before starting each batch — other fleet agents may claim #8758-#8761.

---

## Launch Readiness

- [x] Plan cleaned and saved to disk (`docs/plans/2026-07-01-strategic-plan-close-the-loop.md`, PR #8763)
- [x] Survival guide updated from the current plan
- [x] Learnings + execution log initialized with batch breakdown and preflight notes
- [x] Dedicated worktree created; branch + checkout ownership confirmed; tip recorded (7439a146)
- [ ] Preflight results recorded (mypy + pytest slice running at staging; see execution log)
- [x] `aragora --help` OK; API keys intentionally absent locally (Secrets Manager policy) — evidence via repo review path
- [x] Batches pre-classified by tier; Tier 3-4 flagged
- [x] Run mode, merge policy, non-negotiables recorded
- [x] Stop Gate initialized `no`
- [x] Launch prompt prepared

---

## Current Phase

**Status:** Archived complete. This section records the pre-launch state and is superseded by the
later execution-log waves.
**Active batch:** none
**What was just finished:** the run artifacts were harvested into this PR for preservation.
**Single next action:** none from this archival file.

---

## Next Exact Batch

**Historical next batch at staging:** B1: Drain campaign wave 1 (#8761). Do not execute this
section as a current prompt; later execution-log entries record which parts completed, parked,
or were superseded.
**Predicted merge tier:** 2 (operations; per-PR tier governs each settlement)
**Scope:**
- Probe live state; list ready PRs >7d; exclude Codex-owned (#8726/#8720) and mailbox-frozen items
- For each unowned PR, oldest first, ONE at a time: check head freshness → rebase/refresh only if the PR author lane is inactive and staleness caused CI failure → run bounded quorum evidence (claude+openai, bounded timeouts) → settle Tier 0-2 on clean quorum → prepare Tier 3-4 packets → file advisory follow-ups → park honestly with reasons
- Log every disposition on issue #8761
**Acceptance criteria:**
- [ ] Every ready PR >7d has a disposition logged (settled / packet-prepared / parked+reason)
- [ ] No gate weakened; no evidence posted on dissent; receipts stored per settled PR
**Risk:** reviewer transport flake (Codex is fixing collector preflight in #8726) — if collect-evidence fails twice identically on a PR, park that PR, not the batch.
**Rollback tag:** `elves/pre-batch-1` _(create before starting)_

---

## Acceptance Checks (per batch)

Run the full gate in `.claude/skills/elves-aragora/references/validation-gate-aragora.md`. Summary:

- [ ] Rollback tag created before the batch started
- [ ] Local truth green: `pre-commit run --all-files` (changed-scope acceptable if all-files impractical — log it), `mypy` vs `.mypy-baseline`, `pytest` slice; test count not decreased
- [ ] Adversarial review through aragora review path; quorum facts recorded (head SHA, families, independence, recommendation, dissent)
- [ ] No unresolved dissent (or parked per attempt cap)
- [ ] `DecisionReceipt` produced and verified
- [ ] Tier classified; Tier 0-2 settled autonomously / Tier 3-4 queued for human settlement
- [ ] Execution log + survival guide updated; closing commit (`Co-authored-by: claude[bot]`) pushed
- [ ] Survival guide re-read after push

---

## Tool Configuration (aragora ground truth)

```yaml
lint: pre-commit run --all-files          # changed-scope fallback allowed, must be logged
typecheck: mypy aragora                   # vs .mypy-baseline; no new errors
test: pytest <relevant slice>
mypy-baseline: .mypy-baseline

review: repo review path (preferred, no local keys needed)
review-cmd: python3 -m aragora.cli.main review-queue collect-evidence --repo synaptent/aragora --pr <N> --reviewers claude openai --json
review-fallback: ARAGORA_ENABLE_OPENROUTER_REVIEWER_FALLBACK=1 (grok/deepseek via OpenRouter)
merge-quorum-workflow: aragora-merge-quorum.yml   # enforcing; 3 flags ON (severity-gated+tiered+advisory)

receipt-verify: aragora verify <receipt.json>
receipt-dir: .aragora/run-close-the-loop-20260701/receipts

tier-policy-doc: docs/REVIEW_AUTHORITY_PRINCIPLES.md
human-settlement-signal: aragora/human-settlement
approvals-dir: .approvals

notification: comment on epic #8762 per batch close
```

---

## Rollback and Safety Rules

1. Tag before every batch: `git tag elves/pre-batch-N && git push origin elves/pre-batch-N`.
2. Never force-push or rebase the working branch.
3. Never merge by default. Tier 3-4 never auto-settles.
4. On serious breakage: branch from last good tag, document, stop. Leave the original branch intact.
5. Stage specific files; know what you commit.
6. Surprise branch-tip move = collision → stop, surface to user.
7. B6 cleanup: NOTHING is deleted before its archive (tag/bundle) is verified restorable; keep a restore manifest in the run dir.

---

## After Any Compaction

1. Read this file (doing it now).
2. Read Run Control + Stop Gate.
3. Read `.elves-session.json` (current batch, receipt paths, `continuation_guard`).
4. Read learnings → plan → execution log (last completed batch + receipt + settlement state).
5. Skim operating contract auto-halts + tier table.
6. If `continuation_guard.stop_allowed` is false, continue without re-deciding.
7. Resume the first unblocked incomplete batch. A verified receipt in the log = done; do not redo.

---

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
