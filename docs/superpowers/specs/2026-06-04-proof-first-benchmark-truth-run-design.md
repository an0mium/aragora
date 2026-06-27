# Proof-First Benchmark-Truth Run — Design

**Date:** 2026-06-04
**Status:** Approved (brainstorming), pending implementation plan
**Author:** Claude Code (brainstorming session with operator)

## Purpose

Define a *bounded, long-running, autonomous* objective for the aragora project that
produces real product proof rather than more orchestration substrate. The objective
is a corrective replacement for the prior framing "run for the maximum conceivable
amount of time before you stop" — which optimizes **duration** (a make-work
incentive) instead of **value**.

This run optimizes for a single, falsifiable outcome and **stops when that outcome is
true**.

## North Star

Publish a **fresh, genuinely-rebuilt, verified B0/TW03 benchmark-truth artifact** —
the canonical "the decision-integrity engine is still honest on the corpus" proof.
This dogfoods aragora's own thesis (decision integrity) and aligns with the standing
**substrate-freeze / external-proof** posture: prove the product, freeze the plumbing.

## Exit Condition (the contract)

The run **stops and reports success** when, on a clean `origin/main` observer:

1. `scripts/probe_proof_surface_freshness.py --surfaces b0,tw03 --max-age-days 7`
   exits `0` (both surfaces fresh ≤ 7 days), **and**
2. the underlying artifact was **rebuilt from current data** — verified via
   `scripts/build_benchmark_truth_artifact.py` + `scripts/check_benchmark_regression.py`
   (a real rebuild, not a timestamp bump), **and**
3. the refreshed surfaces are **published** — either landed on `main` via the
   Tier 0–2 settlement path, or left as a draft PR if any change touches a higher tier.

## Hard Stops (run halts and asks for human input)

- Any **Tier 3/4** gate is reached (the run prepares a draft PR + settlement packet,
  never settles it).
- The **metered-spend budget cap** is hit.
- **Substrate-freeze trip** (see Guardrails) fires.
- `N` consecutive batches make **no progress** toward freshness (default `N = 2`).
- Live state contradicts the observer (head drift, dirty tree where clean expected).

## Observer Discipline (non-negotiable)

All truth-reading happens in a **clean `origin/main` worktree pinned to a SHA**
(`scripts/observer_truth_probe.py`), never the dirty founder root. This matches the
operating contract enforced across prior runs: the founder checkout is frequently
dirty/stale and must not be trusted as runtime truth.

## Work Loop (elves-aragora batches)

Each batch is gated by aragora's own governance: adversarial debate → verifiable
DecisionReceipt → tier-appropriate settlement (the `elves-aragora` default).

- **Batch 0 — Diagnose.** Provision the clean observer; run the freshness probe;
  classify *why* each surface is stale into one of:
  - `data-changed` (corpus moved; just needs a rebuild),
  - `promote-rot` (publisher wrote to untracked `.aragora/<surface>/` but the tracked
    `docs/status/generated/<surface>/` copy was never promoted — the known failure
    mode documented in `scripts/refresh_proof_surfaces.sh`),
  - `real-regression` (the engine is genuinely less honest on the corpus).
- **Batches 1..n — Refresh + verify per surface.**
  `scripts/refresh_proof_surfaces.sh --surface b0|tw03` → rebuild artifact →
  `check_benchmark_regression.py` + render → promote tracked copy → commit.
- **Bounded fixes (in scope).** If diagnosis surfaces a real defect (e.g. the rotted
  promote step, or a regression in the proof pipeline), fix it gated by governance.
  - **Tier 0–2:** settle autonomously via genuine claude+grok quorum.
  - **Tier 3/4:** draft PR + stop for operator settlement.
- **Publish.** Refreshed surfaces land on `main` (Tier 0–2) or as a draft PR.

## Guardrails (the anti-make-work core)

These are the mechanisms that fix the original goal's duration-maximizing flaw.

- **Substrate-freeze trip (hard halt).** If a batch's work turns into
  orchestration / settlement / queue / publisher *substrate* rather than the proof
  artifact, **stop that line, publish what exists, and exit.** (Operator's standing
  memory rule, promoted here to an enforced halt.)
- **WIP cap = 1.** Never open work beyond what the artifact strictly needs. Drain,
  don't grow.
- **No queue collision.** The live boss-loop / Codex Desktop fleet owns the general
  PR queue and the automation outbox. This run touches **only** proof-surface PRs.
- **Budget cap.** Stop if metered spend exceeds the ceiling. (Subscription CLIs —
  claude/codex — are effectively free; this caps API spend only.)

## Explicitly Out of Scope: the automation outbox

Investigated during brainstorming and deliberately **excluded** from this run.

- `.aragora/automation-outbox/` holds **129 untracked, operator-local handoffs**
  (publisher cache agrees: outbox = 129, cache = 129 — real, not a stale-count
  artifact).
- They are **blocked publish-intents**, not 129 distinct improvements:
  **127/129 are blocked on `github_unavailable`** (DNS/connectivity failure + `gh`
  unauthenticated **in the codex sandbox**); only **26 distinct branches** across
  129 files (heavy r3/r4 retry duplication); sampled branches are **not on the
  remote** (the pushes never landed). Reconcile dry-run: **127 still protect active
  work, 2 are archivable.** Live GitHub has 11 open PRs, not 129.
- **Verdict: do not bulk-convert to PRs or merge.** That would create
  duplicate/stale/superseded noise and collide with the live loop that already drains
  this when GitHub is healthy.
- **The actual leverage** (a separate, optional task — not this run): 127/129 share
  one root cause — the Codex Desktop sandbox lacks GitHub connectivity/auth. Fixing
  that (or adding an authenticated drain step) lets the existing fleet publish
  naturally. That is a one-time infra fix, worth far more than 129 manual merges.
- Per the substrate-freeze guardrail, the outbox **is** the orchestration substrate,
  so it stays out of the proof-first objective. At most it becomes a bounded
  "reconcile + publish one highest-value handoff" lane later — never a foreground
  drain.

## Steady-State Guardian (the "long objective without make-work" answer)

After the bounded run reaches green and publishes, hand maintenance to a cheap daily
`schedule` routine instead of a perpetual foreground session:

- Run the **read-only** `probe_proof_surface_freshness.py`.
- **Fresh** → log a no-op.
- **Stale** → auto-run `refresh_proof_surfaces.sh --commit` if low-risk; otherwise
  open a draft refresh + alert.

This keeps the proof artifact fresh indefinitely with ~2-minute daily invocations,
avoiding the drift-into-make-work that an always-on session would cause.

## Done-ness Is Self-Proving

The run's success is itself receipt-checkable: a published artifact + a green
freshness probe on clean `main`. The objective validates the thesis it dogfoods —
there is no separate, softer definition of "done."

## Components & Interfaces (runnable surface)

| Unit | Path | Role |
|------|------|------|
| Freshness probe | `scripts/probe_proof_surface_freshness.py` | read-only; exit 0 iff fresh; the exit gate |
| Surface refresher | `scripts/refresh_proof_surfaces.sh` | idempotent refresh + promote; `--check` / `--commit` |
| Artifact builder | `scripts/build_benchmark_truth_artifact.py` | rebuild corpus-linked truth artifact (TW-01/TW-02) |
| Regression check | `scripts/check_benchmark_regression.py` | verify rebuild is honest, not a bump |
| Renderers | `scripts/render_benchmark_truth_status.py`, `render_rescue_productization_status.py` | tracked status docs |
| B0 scorecard | `scripts/measure_b0_scorecard.py`, `measure_b0_progress.py` | corpus scorecard |
| Observer | `scripts/observer_truth_probe.py` | clean-origin/main truth read |
| Driver | `elves-aragora` skill | batched autonomous execution with receipt-gated settlement |
| Guardian | `schedule` routine | daily steady-state freshness |

Proof surfaces:
- B0 (TW-02): `docs/status/B0_BENCHMARK_TRUTH_STATUS.md`
- TW03: `docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md`

## Testing / Verification

- The exit gate **is** the test: `probe_proof_surface_freshness.py` exit 0 on clean
  `origin/main` after a genuine rebuild + regression check.
- Each in-scope fix is verified by aragora's own batch governance (debate → receipt →
  Tier 0–2 quorum) before settlement.
- Guardian correctness: a dry-run of the daily routine on a deliberately-stale fixture
  must detect staleness and produce the refresh (or alert) without false positives on
  fresh input.
