# Reconcile Lane — Design Spec (the first mission on the native orchestrator)

**Date:** 2026-06-26
**Status:** Design / buildable backlog
**Parent:** the canonical mission-orchestrator design (`docs/plans/2026-06-25-native-mission-orchestrator-spec.md`, lands via PR #8655 — not yet on `main`, so referenced rather than linked to avoid a dangling reference). This document is **subordinate** to it: it does not redefine `MissionState`, the stateless tick loop, the handoff/triage protocol, validator injection, or the 6 primitives. It specifies the **first real mission** that runs on that spine.

**Merge dependency:** this PR is safe to land as a design/spec artifact before #8655, but implementation is hard-stacked after #8655. Until `aragora/missions/` is on `main`, references to mission-spine types below are target surfaces, not buildable current-main imports.
**Author:** distilled from a manual reconcile pass run this session (empirical results in §10).

---

## 1. Thesis

`aragora reconcile` is a **merge-first closed control loop** that keeps the repo's git surface (branches, worktrees, open PRs) bounded and clean. It is the proving-ground mission the parent spec calls for ("run a *small* real mission on it … as the proving ground", parent §7.2) — except the work units are not feature implementations but **recovery/cleanup verdicts**, each one gated, receipted, and survivable exactly like any other mission feature.

The lane wraps existing, proven recovery primitives (it **does not rewrite them** — see §4). Its novelty is three things:

1. A **single survivable control loop** over primitives that today run as ~20 uncoordinated launchd daemons.
2. **Adversarial inspection** of substantive stale branches (cheap, because value-density is very low — §10), producing a `DecisionReceipt` per batch.
3. Three governance rules that close the failure modes that killed every prior automation (§3).

This is the merge-side counterpart to Factory-with-provenance: where the parent spec governs *what gets built*, the reconcile lane governs *what gets retired* — and every retirement is debate-gated and receipted.

---

## 2. The seven stages

`aragora reconcile run` executes seven **idempotent** stages. **Dry-run is the default; `--apply` is required to mutate.** Each stage maps to one mission `Feature` (or a small ordered block of features) on the parent's `MissionState`; the stage order is encoded by feature array order + `preconditions` (§5).

| # | Stage | Mutates? | Wraps (does not rewrite) |
|---|-------|----------|--------------------------|
| 1 | **Prune** safe dead branches/worktrees | branches, worktrees | `codex_worktree_autopilot.py cleanup`, `safe_worktree_cleanup.py` |
| 2 | **Triage** stale branches | no (classify) | `harvest_salvage_branches.py` classifier, `codex_worktree_value_inventory.py` |
| 3 | **Inspect** substantive branches (adversarial) | no (vote) | NEW pluggable runner → workflow fan-out *or* API quorum; emits `DecisionReceipt` |
| 4 | **Harvest** preserve-verdicts → draft PRs | draft PRs | `harvest_salvage_branches.py` classifier + new conductor PR adapter |
| 5 | **Cut/Park** policy-allowed retirements | branches + receipts | `harvest_salvage_branches.py` discard path + protected-SHA receipts + human queue |
| 6 | **Settle** green-quorum Tier 0-2 | merges | `auto_merge_green.py` + `scripts/auto_merge_quorum_green.py` |
| 7 | **Govern** WIP backpressure | signal files | `wip_budget.py` + `scripts/backlog_gate.py` |

Every stage is **re-runnable**: re-running after a partial pass converges to the same end-state (the same idempotency contract the parent's `Dispatch` requires). Re-running a completed stage is a no-op that re-emits its terminal receipts.

### Stage 1 — Prune (safe dead branches/worktrees)

- **Goal:** delete branches/worktrees that are provably valueless, with a recovery net.
- **Input:** repo root, `--base origin/main`, `--ttl-hours` (default 24).
- **Eligible (delete):** merged-into-`origin/main` (incl. squash-merge proven by `codex_worktree_autopilot._branch_effectively_merged` / `is_patch_equivalent`) or empty-diff (`harvest_salvage_branches._is_trivial_diff` → "no files changed"/"zero net LOC"). `[gone]` upstream is only a candidate signal; it is never deletion authority unless fresh merged/equivalent/zero-diff proof also exists.
- **Guardrails (NEVER delete):** a **dirty** worktree (`_safe_worktree_dirty`), a worktree with a **live process** (`_has_active_session` via lock files + `lsof` cwd), a **PR-backing branch** (open PR head; cross-checked against `gh pr list`, `codex_worktree_value_inventory.py --include-pr-state`, and outbox/owner evidence), or a branch whose current ref no longer equals the manifest `head_sha`. These are the existing autopilot guardrails plus exact-head binding — the lane only *invokes* them.
- **Recovery net:** deletion never relies on a branch reflog or on `.git/worktree-archive/` alone, because neither is a durable ref that prevents Git garbage collection. Before deletion, the lane writes a durable manifest receipt with `{branch, head_sha, reason}` and creates an actual keep-ref such as `refs/archive/reconcile/<safe-branch>/<timestamp>` pointing at `head_sha` until the retention window expires. Recovery is `git branch <name> <head_sha>` from the receipt while the protected SHA is retained by that ref.
- **Output:** `pruned[]` (branch, sha, reason), `skipped[]` (branch, guardrail or `head_moved`). **Idempotency:** already-deleted → no-op; a re-appearing dirty/live/head-moved worktree is skipped, never force-removed.
- **Calls:** dry-run never calls `codex_worktree_autopilot.py cleanup`, because that helper archives/removes worktrees even when branch deletion is disabled. Dry-run uses the read-only inventory/inspect path (`codex_worktree_value_inventory.py --base origin/main --dry-run --json` plus `safe_worktree_cleanup.py inspect` for candidate worktrees) after a fresh `git fetch origin main`. Apply mode re-runs fresh inspect evidence, creates the keep-ref, resolves each branch ref immediately before deletion, requires `current_head == manifest.head_sha`, removes manifest-listed worktrees via `safe_worktree_cleanup.py remove` **without** relying on that helper's raw branch-delete path, then deletes branches with an atomic exact-head operation (`git update-ref -d refs/heads/<branch> <expected_sha>`) or an equivalent new helper API. The bulk autopilot `cleanup` helper may be used only behind a conductor adapter that proves its entire live result set equals the already-inspected manifest and deletes branches through the same exact-head API. The eligibility base and execution base must match the remote-tracking branch; local `main` is never delete authority.

### Stage 2 — Triage (fingerprint + classify stale branches)

- **Goal:** turn the surviving stale-branch population into a classified worklist, **deduped**.
- **Input:** the post-prune branch list, `--base`.
- **Process:**
  - **Fingerprint** each branch by `commits-ahead + diffstat` (`harvest_salvage_branches._diff_stat` + `_commit_log`).
  - **Dedup retry-series** by signature: branches that are re-derivations of the same work (same base subject + near-identical diffstat, e.g. `…-retry-2`, `…-rebase`) collapse to one representative; the rest are tagged `superseded` and routed to Cut.
  - **Classify** each representative into `empty` / `trivial` / `tiny` / `substantive` (extends the existing trivial/auto-PR/operator-review classifier).
- **Output:** `triage.jsonl` — one record per representative `{branch, head_sha, added, removed, file_count, commit_count, signature, class, superseded_by?}`. Mirrors the existing `salvage-decisions-*.json` shape so downstream stages read a familiar artifact.
- **Idempotency:** pure function of branch heads; re-run produces identical classification for unchanged heads. No mutation.

### Stage 3 — Inspect (adversarial agent fan-out, the value gate)

- **Goal:** for each **tiny** or **substantive** branch, decide preserve-vs-cut by **reading the real diff** — not by a heuristic. This is the only stage that spends model tokens, and §10 shows it is cheap relative to the value it protects.
- **Input:** the `tiny` and `substantive` subsets of the Triage worklist, batched (default batch size 16). `tiny` exists specifically to avoid an orphan class: it is too non-empty to auto-cut by heuristic, so it must either get a preserve/cut inspection receipt or escalate.
- **Process:** adversarial fan-out reads each branch's actual diff + commit log, and votes **preserve | cut | needs-human** with a one-line **reason** per branch. Only a clean preserve majority becomes a Stage-4 preserve verdict. Quorum disagreement emits `needs-human-at-<sha>` and preserves the branch in place for the manual queue (§11); it must not be converted into an automatic draft PR.
- **Pluggable runner** (the key seam; mirrors the parent's pluggable `Dispatch`):
  - **interactive** = a `aragora/workflow/` fan-out (debate nodes over the diffs);
  - **headless** = `aragora/agents/api_agents/*` as a **heterogeneous model quorum** (e.g. `claude + grok`/`claude + deepseek` via OpenRouter — the codex-free reliable pair), so the lane runs on EC2 with no subscription-CLI 402 ceiling (parent Phase D).
  The runner is selected by config/env; both implement one `InspectRunner` protocol returning `[{branch, verdict, reason}]`.
- **Output:** a signed **`DecisionReceipt`** (`aragora/gauntlet/receipt_models.DecisionReceipt`, signed via `gauntlet/signing.SignedReceipt`) **per batch**, recording the diffs inspected, the quorum, and every preserve/cut/needs-human verdict with reason. Receipts persist via `gauntlet/receipt_store`.
- **Idempotency:** keyed on `(branch, head_sha)`; a re-run with an unchanged head reuses the prior receipt rather than re-spending the quorum (receipt-cache lookup first).

### Stage 4 — Harvest (preserve verdicts → draft PRs)

- **Goal:** every clean Stage-3 `preserve` verdict becomes a **DRAFT PR** carrying the inspection rationale, deduped against existing open PRs.
- **Input:** Stage-3 clean `preserve` verdicts + their receipts. `needs-human` / split-inspection receipts are excluded and stay in the manual queue.
- **Process:** for each preserved branch, open a draft PR (rationale = the receipt's verdict reason; body links the `DecisionReceipt` id). Dedup against `gh pr list`, `codex_worktree_value_inventory.py --include-pr-state`, and outbox/owner evidence so a re-run never opens a second PR for an already-represented branch.
- **Why draft:** admission rule (§3a) — a preserved branch now has a **declared path to merge** (the draft PR enters the normal review-queue/quorum gate) instead of lingering as orphaned value.
- **Calls:** `harvest_salvage_branches.py` only for archetype/routing evidence; it is classify-only and never opens PRs. Draft PR creation is new conductor-owned glue around `gh pr create --draft`, with the same pause guard, dedup check, exact branch head, and receipt emission as other mutating adapters.
- **Output:** `harvested[]` (branch, pr_number, receipt_id). **Idempotency:** already-has-PR → no-op.

### Stage 5 — Cut/Park (retire only what policy allows)

- **Goal:** retire everything classified for removal: Triage `empty`/`trivial`/`superseded` + Stage-3 `cut` verdicts. Autonomous deletion is limited to branches with no approval-required risk: already merged/equivalent to `origin/main`, gone upstream with no unique commits, or empty/trivial with fresh zero-diff proof. Any branch with unmerged commits is parked with a `needs-human-at-<sha>` receipt instead of deletion.
- **Guardrails (subtract before deleting):** never cut a branch that is **preserved** (Stage 3/4), **PR-backing** (open-PR head), backed by a **dirty or live worktree/session**, has **unmerged commits requiring approval**, has moved since its receipt/manifest `head_sha`, or touches a **protected surface** from §11. Re-checks live PR state, dirty/live worktree state, active-session locks/processes, merge/equivalence proof, exact branch head, and protected-path classification at cut time (heads move); protected-surface, active/dirty, unmerged, and head-moved branches route to `needs-human-at-<sha>` instead of deletion.
- **Recovery net:** same as Prune — a real `refs/archive/reconcile/...` keep-ref plus a **manifest receipt** listing every cut `{branch, sha, class, reason, receipt_id?}`. Recovery must use the receipt SHA; branch reflog and archive directories are not treated as durable evidence after deletion.
- **Output:** `cut_manifest` receipt for policy-allowed deletions plus `needs-human-at-<sha>` receipts for unmerged/protected/active cut candidates. **Idempotency:** already-deleted → no-op; a branch that gained a PR since Triage is skipped (guardrail re-check).

### Stage 6 — Settle (repaired auto-merge on green quorum, Tier 0-2)

- **Goal:** take the human out of the per-PR merge loop for the safe tier band — the real throughput unlock.
- **Input:** open non-draft PRs (incl. the Harvest drafts once marked ready), their merge-packets and live check states.
- **Process:** `aragora/swarm/auto_merge_green.decide_auto_merge` per PR — **pure** re-check of the already-authorized Tier 0-2 admin-squash conditions (merge-quorum green + all required checks green + mergeable + packet satisfied + tier ≤ 2). `apply_merges(..., dry_run=not apply)` executes head-bound (`--match-head-commit`) squashes.
- **Tier 3-4 → human queue:** never auto-settled; routed to the operator escalation fork (parent primitive 6). This is identical to the dispatch boundary in `aragora/missions/dispatch.BossLoopDispatch` (operator_tier=3).
- **Output:** `settled[]` / `skipped[]` (with blocker list — `decide_auto_merge` accumulates *every* blocker). **Idempotency:** already-merged → reported `skip`; head-moved → re-decided next pass.

### Stage 7 — Govern (WIP backpressure on all generators)

- **Goal:** enforce the merge-first admission rule (§3a) at the fleet level — if WIP is over cap, **signal all generators to drain, not generate**.
- **Input:** live open PR count, including draft PRs, plus in-flight reconcile items and harvested-but-unsettled `preserved-blocked` receipts.
- **Process:** `WIP = open non-draft PRs + open draft PRs + harvested-but-unsettled preserved branches + other in-flight items`. Compute `classify_wip` (`aragora/swarm/wip_budget`) and run `scripts/backlog_gate.py` to write the `.aragora/backpressure.json` signal (`mode: generate | shepherd`). Over cap → every generation lane (boss-loop, publisher, swarm) reads `shepherd` and creates **no new work** until WIP drains.
- **Output:** the backpressure signal file + a `wip_decision` receipt. **Idempotency:** a pure read→classify→atomic-write; re-run overwrites the signal with the current count. Never fabricates a count (over-cap requires a real count AND a real ceiling — `wip_budget`'s fail-safe).

---

## 3. Three governance rules (folded in from the independent Codex review)

These are invariants the whole lane upholds, not stages.

### (a) MERGE-FIRST ADMISSION RULE

**No worker creates new work unless the system has a declared path to merge, park, or retire it.** Operationally: Stage 7 is a **precondition** the generators consult (`backlog_gate` `mode`), and Stage 4 only preserves a branch by giving it a draft PR (a merge path). A branch with no path to merge is either parked with a `needs-human-at-<sha>` receipt or retired only when Stage 5's full guardrail recheck proves deletion is policy-allowed. This is why §10's sprawl could exist at all — generation outran settlement; the rule structurally prevents recurrence.

### (b) ONE CONDUCTOR

The independent head-moving daemons — **boss-loop, publisher, merge-arbiter, merge-shepherd, overnight-watchdog** — become **callable subroutines/lanes of the conductor** (the `MissionOrchestrator` driving the reconcile mission), **not independent launchd jobs**. Concretely:

- They are invoked as `Dispatch`-style callables inside the single orchestrator tick, under the parent's `mission_owner_lock` (exclusive). Two head-movers can no longer race.
- The single launchd job (Phase 2) launches **the conductor**, which calls them in order — replacing N cron entries with one.
- This directly closes the "4-fleet contention" halting class (parent §1).

### (c) THE PAUSE/LOCK MUST BE REAL

There is a confirmed bug: the publisher **never read its pause manifest** (`~/.aragora/fleet-pause/paused-agents.txt`) — a code search this session found **zero references** to it, so the pause was decorative. The reconcile lane fixes this structurally:

- **One central pause/lock**, checked by **every repo-touching action** before it may mutate local or remote repo state: spawn agents, create PRs, push, merge, publish, delete branches, remove/prune worktrees, or write outbound handoff state. Implemented as a single guard (e.g. `reconcile.pause.is_paused()`) read by the conductor-owned mutation adapter used by Stages 1, 4, 5, 6 and by every generator the conductor calls.
- Paused → the stage is a **no-op that emits a signed `paused` operational receipt** (not a silent skip — §9 alerting). This is not one of the five artifact terminal states because no artifact exits the lane while paused.
- The check is the conductor's, not each daemon's, so there is exactly one manifest and one reader. No mutation path may bypass it: tests enumerate the mutation adapter surface (`git push`, `gh pr create/edit/merge/close`, exact-head branch delete via `git update-ref -d`, `git worktree remove/prune`, cleanup helpers, publisher/outbox writes) and assert each path routes through the guard before executing.

---

## 4. What the lane wraps (do NOT rewrite)

| Primitive | Module | Role in lane |
|-----------|--------|--------------|
| Worktree cleanup + guardrails | `scripts/codex_worktree_autopilot.py` (`cleanup`), `scripts/safe_worktree_cleanup.py` | Stage 1 |
| Worktree value inventory | `scripts/codex_worktree_value_inventory.py` (`classify_candidate`, value classes) | Stage 1/2 evidence |
| Salvage classifier (discard / auto-PR-candidate / operator-review) | `scripts/harvest_salvage_branches.py` (`_is_trivial_diff`, `_matches_auto_pr_archetype`, `_diff_stat`, `_commit_log`) | Stages 2,4,5 classification only |
| PR/outbox representation evidence | `scripts/codex_worktree_value_inventory.py --include-pr-state`, `scripts/reconcile_automation_outbox.py`, `scripts/identify_lane_owner.py` | PR-backing / representation evidence (guardrails) |
| Outbox reconcile | `scripts/reconcile_automation_outbox.py` (`--apply`) | settles satisfied handoffs before Cut |
| Auto-merge decision core | `aragora/swarm/auto_merge_green.py` (`decide_auto_merge`, `apply_merges`), `scripts/auto_merge_quorum_green.py` | Stage 6 |
| WIP / backpressure | `aragora/swarm/wip_budget.py` (`classify_wip`), `scripts/backlog_gate.py` | Stage 7 |
| Decision receipts | `aragora/gauntlet/receipt_models.DecisionReceipt`, `gauntlet/signing.SignedReceipt`, `gauntlet/receipt_store` | every terminal receipt |

The lane is **orchestration + governance + the inspection runner**. Everything else already exists and is battle-tested.

---

## 5. Architecture — mapping onto MissionState / ledger / orchestrator

After #8655 lands, the reconcile lane is **a mission**, expressed in the parent's mission types (`aragora/missions/`), with **no new spine code**:

- **`MissionState`** (`state.py`): `goal = "reconcile the repo git surface"`. The seven stages are **`Feature`s** in array order, milestone-grouped (`milestone="prune"`, …, `"govern"`). Stage ordering is `preconditions` (`feature:prune` → `feature:triage` → …). Substantive-branch inspection fans out as **per-batch features** under the `inspect` milestone, each `fulfills` a `VAL-*` assertion ("every substantive branch has a preserve/cut verdict with a receipt").
- **`MissionOrchestrator`** (`orchestrator.py`): drives the stages one survivable tick at a time. Each stage's work is a `Dispatch` callable returning a `Handoff`. A `kill -9` mid-Cut resumes from the persisted checkpoint with zero double-cut (Stage idempotency = dispatch idempotency).
- **`Ledger`** (`ledger.py`) + **`run_worker`** (`swarm.py`): the Inspect stage's per-batch fan-out runs as a **swarm** — many workers atomic-claim batches via `select_for`, vote, and write verdicts as ledger `discoveries` (advisory). `reconcile_from_ledger` folds completions back. The propose/accept boundary holds: workers *propose* verdicts; only the orchestrator + receipt turn a verdict into a Harvest/Cut action.
- **`Handoff`**: a stage returns `success` (advance), `discovered[]` (e.g. "47 superseded branches"), `follow_ups[]` (advisory — e.g. "branch X needs a manual split"; **never** auto-accepted, so a buggy inspection can't widen scope), and `terminal=True` for operator forks (Tier 3-4, ambiguous split → the manual queue, §11).
- **Conductor binding (§3b):** the orchestrator *is* the one conductor. boss-loop/publisher/merge-arbiter become `Dispatch` callables it invokes inside Stages 4/6, all under the single `mission_owner_lock` exclusive fence — so they cannot race each other or a second conductor.

### Data flow

```
[live git + GitHub]
   │  (re-derived every tick — never trust in-memory carry-over, parent A4)
   ▼
Prune ──▶ surviving branches ──▶ Triage ──▶ triage.jsonl
                                              │   ├─ empty/trivial/superseded ─────────────┐
                                              │   └─ tiny/substantive ─▶ Inspect (quorum)   │
                                              │                         │ preserve │ cut     │ needs-human
                                              │                         ▼          ▼        ▼
                                              │                      Harvest     ───▶ Cut/Park ─▶ human queue
                                              │                     (draft PRs)     (receipt)
                                              ▼
                                          Settle (Tier 0-2 green) ──▶ merges ; Tier 3-4 ──▶ human queue
                                              ▼
                                          Govern (WIP) ──▶ .aragora/backpressure.json ──▶ all generators
```

Every arrow that mutates (Prune delete, Harvest PR, Cut delete, Settle merge) passes the §3c pause check and emits the appropriate receipt: a §9 artifact terminal receipt for real exits, or a non-terminal operational `paused` receipt when the pause guard prevents mutation.

---

## 6. Per-stage interface contract

Each stage implements the parent's `Dispatch = Callable[[Feature], Handoff]`. Uniform contract:

| Property | Contract |
|----------|----------|
| **Input** | `Feature` (carries stage params in `notes`/config) + live git/GitHub re-derived at dispatch time |
| **Output** | `Handoff(success, discovered[], follow_ups[], terminal, blocked_reason)` + zero-or-more artifact terminal receipts and/or operational receipts |
| **Idempotency** | re-run with unchanged heads → same result; mutations are no-ops when already applied; receipts cache on `(artifact, sha)` |
| **Dry-run** | default; emits the *plan* (would-prune/would-cut/would-merge) and writes non-terminal preview receipts marked `dry_run=true`; `--apply` flips to real mutation and is the only mode that may emit terminal artifact-exit receipts |
| **Pause** | every mutating stage checks §3c first; paused → `Handoff(success=False, terminal=True, blocked_reason="fleet paused")` + signed non-terminal `paused` operational receipt |
| **Calls** | the wrapped primitive(s) in §4 — the stage is glue, not new cleanup logic |

---

## 7. Error handling & resilience (each documented prior-failure mode, addressed)

| Prior failure | Mechanism in the lane |
|---------------|-----------------------|
| **Silent GitHub-auth `None`** | **Fail-loud auth:** refresh the token **per stage**; on mint failure **raise + alert** (§9) — never proceed with a silent `None`. A stage that cannot authenticate returns `terminal` blocked, not a false-clean pass. |
| **Git wedge blocks the whole pass** | **Per-branch timeouts** on every git call (the primitives already take `timeout=`/`--git-timeout`); a stuck worktree is **quarantined** (moved to archive + flagged) and the pass continues. One wedged worktree never blocks the lane. |
| **Non-idempotent re-runs** | Every stage re-runnable; **dry-run default**; mutations no-op when already applied; receipt-cache prevents re-spending the inspection quorum. |
| **Silence instead of alerting** | Token-mint failure, **N consecutive merge failures**, **quarantined worktrees**, and **WIP-cap-hit** all emit a notification (`control_plane/notifications.py`). Silence is a bug. |
| **No record of what happened** | **A terminal-state receipt for EVERY artifact**, one of: `merged`, `closed-superseded`, `preserved-blocked` (→ draft PR), `deleted` (with recovery SHA), `needs-human-at-<sha>`. No branch/worktree/PR exits the lane without a receipt. |

Crash safety is inherited from the spine: `MissionState.save` is atomic (`os.replace`), the orchestrator persists before dispatch, and `mission_owner_lock` makes a second conductor / live swarm fail fast (`MissionOwnershipError`).

---

## 8. Testing strategy

- **Temp-repo fixtures with synthetic sprawl.** A fixture builds a throwaway git repo and fabricates the full taxonomy: merged-into-main, `[gone]`, empty-diff, dirty worktree, live-process worktree (sentinel lock file), PR-backing branch, retry-series duplicates, `trivial`/`tiny`/`substantive` branches, and a Tier-0..4 spread of open PRs. No network: `gh`, outbox/owner probes, and the model quorum are injected (the primitives already take injectable list/merge fns — `apply_merges(merge_fn=…)`, `backlog_gate.run_gate(list_prs=…)`, and the inventory/outbox scripts accept cached or fixture-backed data).
- **Dry-run vs apply parity.** For each stage, assert the dry-run **plan** exactly equals the set the `--apply` run mutates (same branches pruned/cut, same PRs merged) — the core safety property. A divergence is a bug.
- **Guardrail tests.** Assert Prune/Cut **never** touch a dirty / live-process / active-session / PR-backing / preserved / protected-surface / unmerged-commit branch even when it otherwise classifies for removal.
- **Exact-head deletion tests.** Advance a branch after dry-run/inspection but before apply; assert Prune/Cut skip deletion and emit `needs-human-at-<sha>` or `head_moved`, preserving both the old protected SHA and the new branch tip.
- **Atomic delete tests.** Assert branch deletion uses `git update-ref -d refs/heads/<branch> <expected_sha>` or a helper wrapping that exact compare-and-delete operation, never raw `git branch -D`, for manifest-listed branch deletion.
- **Split-inspection routing.** Assert a split inspection emits `needs-human-at-<sha>` and does not feed Stage 4's draft-PR creation path.
- **Receipt emission.** Assert every artifact that exits any apply stage has exactly one terminal receipt in one of the five terminal states, signature-valid (`SignedReceipt`), and recoverable (deleted-receipt SHA re-creates the branch because an actual `refs/archive/reconcile/...` keep-ref protects it). Dry-run preview receipts are explicitly non-terminal and must not consume the artifact's terminal receipt slot.
- **WIP accounting.** Assert Stage 7 counts open draft PRs and harvested-but-unsettled preserved branches, not only non-draft PRs, before it allows generators to produce more work.
- **Idempotency.** Run each stage twice on the same fixture; second run mutates nothing and re-emits identical receipts; the inspection quorum is **not** re-invoked (receipt-cache hit).
- **Pause is real.** With the pause manifest set, assert every mutating stage emits a non-terminal `paused` operational receipt and mutates nothing. The §3c regression test must also enumerate the full conductor-owned mutation adapter surface — PR creation/edits/closures, pushes, merges, branch deletion, worktree removal/pruning, cleanup helpers, and publisher/outbox writes — and prove every path calls the pause guard before executing.
- **Crash/resume.** `kill -9` mid-Cut (between two deletes); relaunch; assert no double-delete and the manifest receipt is consistent — reuses the parent's Phase-A exit test.

---

## 9. Receipts — the five terminal states

Every artifact exits in exactly one state, each a signed `DecisionReceipt`:

| Terminal state | Meaning | Recovery |
|----------------|---------|----------|
| `merged` | Tier 0-2 settled on green quorum | n/a |
| `closed-superseded` | retry-series duplicate or obsoleted branch retired only when merged/equivalent/no-unique-work proof makes deletion policy-allowed | protected SHA in receipt |
| `preserved-blocked` | inspection said preserve → draft PR opened | PR number in receipt |
| `deleted` | safe dead branch/worktree pruned/cut | `git branch <name> <sha>` from receipt |
| `needs-human-at-<sha>` | Tier 3-4, ambiguous split, or protected-surface — manual queue | operator fork |

Receipts persist via `gauntlet/receipt_store`; the per-batch inspection receipt additionally records the quorum and per-branch reasons (the audit trail that makes "we cut 2,260 branches" defensible).

---

## 10. Empirical grounding (why this design is correct)

A **manual run of exactly this flow this session** validated it:

- **Branches: 2,749 → 489.** **Worktrees: 319 → 1.**
- Adversarial inspection of **288 substantive stale branches found exactly 4 worth preserving.** The archived rollup identified **235 now-obsolete automation-plumbing** and **47 superseded** rows among the non-preserved set; the signed manifest remains the only delete authority, and any count mismatch fails closed into manual reconciliation instead of becoming implicit cut authority.

Two design conclusions follow directly:

1. **The flow works** end-to-end at scale on the real repo.
2. **The value-density of sprawl is very low** (~1.4% of even the *substantive* subset was worth keeping). Therefore **aggressive classification + cheap inspection is the correct posture**: the expected cost of over-preserving (carrying thousands of dead branches) vastly exceeds the expected cost of inspecting (a few cents of quorum per substantive branch, receipt-cached). This is the empirical justification for classifying stale work quickly, deleting only policy-allowed no-unique-work branches, and parking unmerged/protected cases with receipts — and for the fail-safe (split → preserve) being rare enough to route to a human without flooding the queue.

---

## 11. What stays manual (never auto-mutated)

- **Tier 3-4 settlement** — server / persistence / security / public-API surfaces. Routed to the operator fork as `needs-human-at-<sha>`. Identical boundary to `dispatch.BossLoopDispatch.operator_tier=3`.
- **Agent-split ambiguous branches** — when inspection quorum is *split* (no clean preserve/cut majority), the branch is **preserved into the manual queue**, not auto-cut and not auto-PR'd. §10 says these are rare.
- **Protected files / workflows / public API / control-plane surfaces** — per the Agent Operating Contract and repo-local agent rules: a branch touching `AGENTS.md`, `CLAUDE.md`, `docs/AGENT_OPERATING_CONTRACT.md`, `docs/governance/**`, `aragora/__init__.py`, `scripts/nomic_loop.py`, `.github/workflows/*`, `.env*`, `secrets/**`, or a public API surface is never auto-cut or auto-merged here; it escalates.
- **`--apply` itself** — the lane is dry-run by default; a human (or a Phase-2 scheduled job with an explicit `--apply` and an honored pause manifest) authorizes mutation.

---

## 12. Delivery — 4 incremental phases

Each phase is independently useful and testable.

### Phase 1 — On-demand `aragora reconcile run`
- Stages **1-5** (Prune + Triage + Inspect + Harvest + Cut), **dry-run default**.
- Inspection runner = interactive workflow fan-out (operator present).
- Ships the temp-repo fixture + dry-run/apply parity + guardrail + receipt tests (§8).
- **Value:** one command takes the repo from sprawl to clean, every cut receipted and recoverable. This is the §10 flow, productized.

### Phase 2 — Scheduled, headless, alerting
- **One** launchd job runs the conductor (replacing the ~20 daemons — §3b).
- **Headless inspection runner** = API-agent heterogeneous quorum (no subscription CLI; EC2-safe — parent Phase D).
- **Alerting** wired (§7): token failure, N-consecutive-merge-fail, quarantined worktrees, WIP-cap-hit.
- **Value:** the lane runs unattended and only pings on genuine forks.

### Phase 3 — WIP-governor binding
- Stage **7** (Govern) wired so **all** generators (boss-loop, publisher, swarm) consult `backlog_gate` `mode` before generating, and **all** repo-touching actions honor the **single real pause manifest** (§3c) — including the regression grep-test.
- **Value:** the merge-first admission rule (§3a) is enforced fleet-wide; generation can no longer outrun settlement (the root cause of §10's sprawl).

### Phase 4 — Repaired auto-merge (Tier 0-2)
- Stage **6** (Settle) live: `decide_auto_merge` + `apply_merges(dry_run=False)` on green-quorum Tier 0-2; Tier 3-4 → human queue.
- **Value:** the human is out of the per-PR merge loop for the safe tier band — the throughput unlock, gated and receipted.

---

## 13. The wedge (one line)

The parent spec gives *Factory with provenance* for **building**. The reconcile lane gives the same for **retiring**: the only autonomous system where even a *branch deletion* is debate-gated, receipted, and recoverable — and where, empirically, 2,749 branches collapse to 489 with exactly 4 human-confirmed keeps, all on one survivable control loop.
