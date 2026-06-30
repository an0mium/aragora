# Proof-First Benchmark-Truth Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the **`elves-aragora`** skill to
> execute this plan. It is the aragora-governed autonomous driver: each batch is gated
> by adversarial debate → verifiable DecisionReceipt → tier-appropriate settlement, and
> it manages its own worktrees. (Generic `subagent-driven-development` / `executing-plans`
> are *not* used here — they lack the receipt-backed quorum and tier halts this run
> requires.) Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a fresh, genuinely-rebuilt, verified B0/TW03 benchmark-truth artifact
on clean `origin/main`, then stop on success and hand steady-state to a daily guardian.

**Architecture:** A short sequence of `elves-aragora` batches operating aragora's own
proof-surface tooling from a clean `origin/main` observer worktree. Diagnose → refresh
→ verify → (bounded, governance-gated fix if needed) → publish → install guardian. The
run is bounded by a falsifiable exit condition, not by elapsed time.

**Tech Stack:** Python 3.11 (pyenv), aragora CLI 2.9.x, bash, `gh`, the `schedule`
skill (cron). All proof tooling already exists under `scripts/`.

**Design doc:** `docs/superpowers/specs/2026-06-04-proof-first-benchmark-truth-run-design.md`

---

## Standing Rules (apply to EVERY batch — do not violate)

- [ ] **Observer discipline.** Read runtime truth only from a clean `origin/main`
  worktree pinned to a SHA. Never trust the founder root. Re-pin per batch.
- [ ] **Substrate-freeze trip (HARD HALT).** If a batch's work turns into
  orchestration / settlement / queue / publisher *substrate* rather than the proof
  artifact → stop that line, publish what already exists, and exit with a report.
- [ ] **WIP cap = 1.** Never open work beyond what the artifact strictly needs.
- [ ] **No queue collision.** Touch only proof-surface files/PRs. Do **not** drain the
  automation outbox or the general PR queue (the live boss-loop owns those).
- [ ] **Tier gate.** Tier 0–2 fixes settle autonomously via genuine claude+grok quorum;
  any **Tier 3/4** change → draft PR + settlement packet + **STOP for operator**.
- [ ] **Budget cap.** Stop if metered API spend exceeds the operator-set ceiling
  (subscription claude/codex CLIs are free; this caps API spend only).
- [ ] **No-progress halt.** If 2 consecutive batches make no progress toward freshness,
  stop and report.

---

## File / Surface Map

| Path | Role | Touched by |
|------|------|-----------|
| `scripts/probe_proof_surface_freshness.py` | read-only freshness gate (exit 0 iff fresh) | all batches (read) |
| `scripts/refresh_proof_surfaces.sh` | idempotent refresh + promote (`--check`/`--surface`/`--commit`) | Batch 1, 2 |
| `scripts/build_benchmark_truth_artifact.py` | rebuild corpus-linked truth artifact | Batch 1 |
| `scripts/check_benchmark_regression.py` | verify rebuild is honest | Batch 1, 2 |
| `scripts/render_benchmark_truth_status.py` | render B0 status doc | Batch 1 |
| `scripts/render_rescue_productization_status.py` | render TW03 status doc | Batch 2 |
| `scripts/observer_truth_probe.py` | clean-origin/main truth read | Batch 0 |
| `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` | B0 (TW-02) surface | Batch 1 (output) |
| `docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md` | TW03 surface | Batch 2 (output) |
| guardian routine (new) | daily steady-state freshness | Batch 5 |

---

## Batch 0: Diagnose (read-only)

**Goal:** Establish the clean observer and determine *why* (if at all) each surface is
stale. No mutations.

**Files:** none modified (read-only).

- [ ] **Step 1: Provision a clean observer pinned to origin/main**

```bash
git fetch origin --prune
git worktree add --detach /tmp/proof-observer origin/main
cd /tmp/proof-observer
git rev-parse HEAD   # record the pinned SHA in the batch receipt
```

Expected: a clean worktree; `git status --porcelain` empty.

- [ ] **Step 2: Capture exact tool flags (avoid guessing)**

```bash
python3 scripts/probe_proof_surface_freshness.py --help
bash   scripts/refresh_proof_surfaces.sh --help
python3 scripts/build_benchmark_truth_artifact.py --help
python3 scripts/check_benchmark_regression.py --help
```

Expected: help text for each. Record any flags that differ from this plan and prefer
the live `--help` contract over this document.

- [ ] **Step 3: Probe current freshness (the exit gate, measured up front)**

```bash
python3 scripts/probe_proof_surface_freshness.py --surfaces b0,tw03 --max-age-days 7; echo "exit=$?"
```

Expected: JSON with one record per surface (`surface,last_updated,age_days,fresh`).
- If `exit=0` (both fresh) → **the exit condition may already hold**; skip to Batch 4
  to confirm via a genuine rebuild + regression check before declaring done.
- If `exit!=0` → note which surface(s) are stale and proceed.

- [ ] **Step 4: Classify the staleness cause (report-only refresh)**

```bash
bash scripts/refresh_proof_surfaces.sh --check
```

Classify each stale surface into exactly one:
- `data-changed` — corpus moved; a rebuild will refresh it.
- `promote-rot` — publisher wrote `.aragora/<surface>/` but the tracked
  `docs/status/generated/<surface>/` copy was not promoted (documented failure mode in
  the refresh script header).
- `real-regression` — the engine is genuinely less honest on the corpus.

- [ ] **Step 5: Record the diagnosis as the batch DecisionReceipt**

Write the pinned SHA, per-surface freshness, and classification into the batch receipt.
This receipt is the input contract for Batches 1–3.

---

## Batch 1: Refresh + verify B0 (TW-02)

**Goal:** Produce a genuinely-rebuilt, regression-checked B0 surface.

**Files:** Modify `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` (+ its generated copy) via
the refresh script; no hand-editing of generated surfaces.

- [ ] **Step 1: Rebuild the B0 artifact from current data**

```bash
bash scripts/refresh_proof_surfaces.sh --surface b0
```

Expected: the publisher rebuilds the artifact and promotes the tracked copy. If the
script reports "no change" but Batch 0 said B0 was stale, this is `promote-rot` →
proceed to Batch 3 (do not hand-edit the timestamp).

- [ ] **Step 2: Verify the rebuild is honest, not a timestamp bump**

```bash
python3 scripts/check_benchmark_regression.py; echo "exit=$?"
```

Expected: `exit=0` (no regression). If `exit!=0` → this is `real-regression` →
**do not publish**; proceed to Batch 3.

- [ ] **Step 3: Confirm B0 now passes the freshness gate**

```bash
python3 scripts/probe_proof_surface_freshness.py --surfaces b0 --max-age-days 7; echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 4: Settle the B0 refresh under governance**

Hand the B0 surface change to the `elves-aragora` batch gate (claude+grok quorum +
receipt). If the change is Tier 0–2 → settle autonomously. If it touches anything
Tier 3/4 → draft PR + STOP. Commit message: `chore(proof): refresh B0 benchmark truth`.

---

## Batch 2: Refresh + verify TW03

**Goal:** Produce a genuinely-rebuilt TW03 rescue-productization surface, handling the
known promote-step rot.

**Files:** Modify `docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md` (+ generated copy)
via the refresh script.

- [ ] **Step 1: Rebuild + promote TW03**

```bash
bash scripts/refresh_proof_surfaces.sh --surface tw03
```

Expected: publisher rebuilds `.aragora/rescue_productization/latest.json` and promotes
it to `docs/status/generated/rescue_productization/latest.json`, then renders the
tracked status doc.

- [ ] **Step 2: Confirm TW03 passes the freshness gate**

```bash
python3 scripts/probe_proof_surface_freshness.py --surfaces tw03 --max-age-days 7; echo "exit=$?"
```

Expected: `exit=0`. If still stale after a successful rebuild → `promote-rot` →
Batch 3.

- [ ] **Step 3: Settle the TW03 refresh under governance**

Same gate as Batch 1 Step 4. Commit message: `chore(proof): refresh TW03 rescue productization`.

---

## Batch 3 (CONDITIONAL): Bounded governance-gated fix

**Run only if** Batch 0–2 surfaced `promote-rot` or `real-regression`. Skip otherwise.

**Goal:** Restore an honest, publishable proof surface by fixing the *real* defect — not
by masking it.

**Files:** determined at execution by the diagnosis (e.g. the promote step inside
`scripts/refresh_proof_surfaces.sh`, the renderer, or the corpus/regression logic). The
exact diff is produced by `elves-aragora` under TDD within the batch; it is deliberately
**not** pre-written here because inventing a diff for an undiagnosed defect would be a
placeholder.

- [ ] **Step 1: Reproduce the defect with a failing check**

Capture the exact failing command + output from Batch 1/2 (the regression `exit!=0`, or
the post-refresh probe still stale). This is the red state the fix must turn green.

- [ ] **Step 2: Implement the minimal fix under the batch governance gate**

Let `elves-aragora` drive the fix TDD-style. Constraint: the fix must make the Step 1
check pass **and** leave `check_benchmark_regression.py` at `exit=0`.

- [ ] **Step 3: Re-verify the full gate**

```bash
python3 scripts/check_benchmark_regression.py; echo "regression_exit=$?"
python3 scripts/probe_proof_surface_freshness.py --surfaces b0,tw03 --max-age-days 7; echo "fresh_exit=$?"
```

Expected: both `exit=0`.

- [ ] **Step 4: Settle under governance, respecting the tier gate**

Tier 0–2 → autonomous quorum settle. Tier 3/4 (e.g. if the fix touches protected
publisher/workflow surfaces) → draft PR + settlement packet + **STOP for operator**.

---

## Batch 4: Publish + confirm the exit condition

**Goal:** Confirm the exit contract holds on clean `origin/main` and the artifact is
published.

- [ ] **Step 1: Re-pin a fresh clean observer (state may have advanced)**

```bash
# from any existing checkout of the aragora repo (worktree add works from any worktree):
git fetch origin --prune
git worktree add --detach /tmp/proof-observer-final origin/main
cd /tmp/proof-observer-final
```

- [ ] **Step 2: Genuine rebuild + regression + freshness on the published main**

```bash
python3 scripts/check_benchmark_regression.py; echo "regression_exit=$?"
python3 scripts/probe_proof_surface_freshness.py --surfaces b0,tw03 --max-age-days 7; echo "fresh_exit=$?"
```

Expected: both `exit=0`. This is the **exit condition**.

- [ ] **Step 3: Emit the run's success receipt**

Record: pinned SHA, both exit codes, the published surface paths, and the merge/draft-PR
state of each refresh. If all green → the run is **done**; proceed to Batch 5, then stop.
If a Tier 3/4 draft PR is pending operator settlement → stop here and report that as the
single remaining blocker.

---

## Batch 5: Install the steady-state guardian

**Goal:** Replace any future foreground sessions with a cheap daily freshness check.

**Files:** a new `schedule` routine (cron). No repo code change required.

- [ ] **Step 1: Define the guardian command**

The guardian runs, from a clean `origin/main` checkout:

```bash
python3 scripts/probe_proof_surface_freshness.py --surfaces b0,tw03 --max-age-days 7
# fresh (exit 0) -> log no-op
# stale (exit !=0) -> bash scripts/refresh_proof_surfaces.sh --commit   (low-risk auto)
#                     else open a draft refresh PR + alert the operator
```

- [ ] **Step 2: Create the daily routine via the `schedule` skill**

Use the `schedule` skill to register a daily run of the Step 1 logic (suggested: early
morning local time). The routine must be read-only on the fresh path and only mutate via
the existing idempotent `refresh_proof_surfaces.sh --commit` on the stale path.

- [ ] **Step 3: Dry-run the guardian against a deliberately-stale fixture**

Verify it detects staleness and produces the refresh (or alert) with no false positive on
fresh input. Record the dry-run result in the final report.

---

## Self-Review (completed)

- **Spec coverage:** exit condition (Batch 0 S3 / Batch 4), observer discipline (Standing
  Rules + Batch 0/4), refresh+verify (Batch 1/2), bounded Tier 0–2 fix (Batch 3),
  substrate-freeze + WIP + no-collision guardrails (Standing Rules), outbox out-of-scope
  (Standing Rule "No queue collision"), steady-state guardian (Batch 5), self-proving
  done-ness (Batch 4 S2/S3). All spec sections map to a task.
- **Placeholder scan:** the only intentionally-unwritten code is Batch 3's fix diff,
  which is *correctly* deferred to execution-time diagnosis under `elves-aragora` TDD
  (documented why). All operational steps carry exact commands + expected exit codes.
- **Consistency:** surface names (`b0`, `tw03`), script paths, and the freshness/
  regression gates are used identically across Batches 0–5.
