# Mission Schema — Design Spec

Status: design spec (medium-term item 2 of epic [#8344](https://github.com/synaptent/aragora/issues/8344),
§5.2 of `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md`).
Written 2026-06-13. Implementable; proposes a concrete schema, a loader/executor
contract, and how validation contracts and readiness probes plug in.

## 1. Why this exists: the orchestrator must be swappable

The program doc states the load-bearing principle plainly: **no load-bearing
component should assume a specific model.** The 2026-06-12 Fable-5 suspension
made it concrete — the heterogeneous merge quorum sailed through unaffected,
because heterogeneity is built into its construction, while anything *pinned to a
single model* (a mission orchestrator) was exposed and stalled.

Today `docs/superpowers/plans/*` mission files are prose. They are excellent for
humans and for a model that happens to be running, but they are not
*machine-loadable*: a Claude Code session, a Codex goal-loop, and a Factory
mission each re-interpret the same prose differently, and each re-researches the
repo's merge gate from scratch (the program's harness-comparison table calls
this out as Factory's weakness — "had to re-research the repo's merge gate from
scratch"). A machine-readable mission schema turns the prose into an artifact any
harness can load and execute identically. The orchestrator becomes a swappable
runtime, not a single point of model failure.

This is the program's "absorb into aragora as repo artifacts" item #3 (mission
schema) composed with #1 (validation contracts) and #2 (readiness probes).

## 2. Prior art: the Factory mission structure

Factory missions are the explicit prior art (harness-comparison table:
"readiness probes, validation contracts, milestone gates, pointed clarifying
questions"). A Factory mission already carries, in spirit:

- a **goal** and decomposition into **phases / milestones**,
- **gates** between milestones (don't advance until the gate passes),
- **validation contracts** (machine-checkable acceptance per goal),
- a **readiness probe** (a dependency check before spending budget),
- **clarifying questions** surfaced before execution.

The Aragora loop's complementary superpower is the *governed merge transport*:
tiers, quorum, receipts, 24/7 operation. The mission schema's job is to make a
mission file carry **both** — Factory's planning discipline and Aragora's
tiering/yield/budget governance — in one portable document. The existing
`docs/superpowers/plans/*` files (e.g. this program doc, the codebase-health
program, the steering-leverage plans) are the corpus the schema must be able to
represent without losing information.

## 3. Format: YAML front-matter + Markdown body

A mission file stays a readable Markdown document with a **YAML front-matter
block** carrying the machine-readable contract. This keeps the human narrative
(which is where the *why* lives) while making the executable fields parseable
with zero new file types. The front-matter validates against a published
JSON-schema (`docs/superpowers/schema/mission.schema.json`). Rationale for
front-matter over a separate `.json`: the prose and the contract drift apart if
they live in two files; co-locating them keeps the spec and its rationale in one
reviewable diff (the same discipline the codebase already applies to receipts).

```markdown
---
mission_schema_version: 1
id: conveyor-event-substrate
title: Event substrate over polling
goal: >
  Retire most PR-state polling by pushing GitHub events to a local collector
  that maintains the shared PR-state cache; double the collector as the TET
  external witness.
tier: 2                      # max tier any phase reaches; per-phase tier may differ
budgets:
  wall_clock_minutes: 240
  model_calls: 400
  usd: 5.00
readiness_probe: mission-readiness/event-substrate   # see §6
yield_rules:                  # ODR / path-freeze behavior
  path_freeze:
    - "scripts/pr_state_cache.py"      # reader contract is frozen; do not edit
  odr_yield:                  # owner-declared-region yield: steer via comments, do not edit
    - surface: "review_queue*"
      owner: scarmani
phases:
  - id: p0-shadow
    name: Collector alongside poll (shadow)
    tier: 2
    human_checkpoint: false
    acceptance:                # validation contract refs (§5)
      - contract: webhook-cache-divergence-bounded
      - contract: no-lane-reads-webhook-cache-yet
  - id: p1-canonical
    name: Collector writes canonical cache; poll demoted
    tier: 2
    human_checkpoint: false
    depends_on: [p0-shadow]
    acceptance:
      - contract: graphql-consumption-dropped
      - contract: reconcile-delta-near-empty
  - id: p2-witness
    name: Witness wiring
    tier: 4                    # inherits TET identity discipline
    human_checkpoint: true     # scarmani settlement / operator preapproval
    depends_on: [p1-canonical]
    acceptance:
      - contract: trail-reconcile-breaches-on-unmatched
validation_contracts: docs/superpowers/contracts/event-substrate/
references:
  - "#8344"
  - "#8339"
  - "docs/specs/TAMPER_EVIDENT_TRAIL.md"
---

# Event substrate over polling

(human narrative — the design doc body, unchanged prose...)
```

## 4. Required fields (the schema contract)

The JSON-schema (`mission.schema.json`) requires:

| Field | Type | Meaning |
|---|---|---|
| `mission_schema_version` | int | schema version for forward compat |
| `id` | string (slug) | stable mission id; matches a lane ledger entry |
| `title` | string | human title |
| `goal` | string | one-paragraph north star |
| `tier` | int 0–4 | max tier reached by any phase (`docs/REVIEW_AUTHORITY_PRINCIPLES.md`) |
| `budgets` | object | at minimum `wall_clock_minutes`; optionally `model_calls`, `usd` |
| `phases` | array | ordered phases, each with the phase contract below |
| `yield_rules` | object | `path_freeze` (paths no lane may edit) + `odr_yield` (owner-declared regions; steer via comments) |
| `validation_contracts` | path | directory/file of machine-checkable acceptance assertions (§5) |
| `readiness_probe` | string | id of the pre-launch probe to run (§6) |
| `references` | array | issues/docs this mission threads (e.g. `#8344`) |

Each **phase** requires:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | phase slug |
| `name` | string | human name |
| `tier` | int 0–4 | this phase's tier (may exceed mission `tier` only if mission `tier` is raised to match) |
| `human_checkpoint` | bool | if true, the harness MUST pause for explicit operator authorization before/after the phase (Tier 3–4 phases are always `true`) |
| `acceptance` | array of `{contract: <id>}` | the validation contracts that must pass for the phase to count as done |
| `depends_on` | array of phase ids (optional) | DAG ordering |

The schema is intentionally a superset-compatible *extension* of the Factory
mission shape: goal, phases-with-gates, validation contracts, readiness probe,
checkpoints. The aragora-specific additions are `tier`, `budgets`, and
`yield_rules` — the governance fields that make the mission honor this repo's
merge authority and claim-discipline rules.

## 5. Validation contracts (machine-checkable acceptance)

A validation contract is the program's absorb-item #1: a *machine-checkable
acceptance assertion per goal, evaluated before a lane's work counts as done*.
Each contract is a small, harness-agnostic, executable check living under
`docs/superpowers/contracts/<mission-id>/<contract-id>.{sh,py,yaml}` with a
declared verdict protocol:

- exit 0 = satisfied; nonzero = not satisfied; the check prints a one-line
  human-readable verdict and a machine `{"contract": id, "pass": bool,
  "evidence": ...}` line.
- contracts are **read-only and deterministic** where possible (they assert over
  repo state, CI status, cache files, metrics) — they never mutate.
- a phase's `acceptance` list is ANDed: every referenced contract must pass for
  the phase to advance.

Example contract ids from the event-substrate mission above:
`graphql-consumption-dropped` (asserts measured GraphQL pts/hr fell below a
threshold after Phase 1), `reconcile-delta-near-empty` (asserts the reconcile
poll found ≤N deltas the webhooks missed), `trail-reconcile-breaches-on-unmatched`
(asserts the reconciler raises a breach on a simulated unmatched witness event —
the TET T5 replay test). These are exactly the *falsifiable exit metrics* the
TET spec and the event-substrate spec already write in prose; the schema makes
them executable and binding.

Why this matters: today a lane self-reports "done" from its transcript, and the
"do not trust transcript state" doctrine then forces re-verification by polling.
A validation contract moves the done-judgment out of the transcript into a
deterministic check any harness (or the sentinel, or the operator) can re-run —
it is the planning-side analog of the merge quorum's evidence-first discipline.

## 6. Pre-launch readiness probe

The program's absorb-item #2: a **5-minute dependency check before any lane
spends budget** (3 of the run's silent deaths were probe-preventable — failure
class A). A probe is referenced by id from the mission and lives under
`docs/superpowers/probes/<probe-id>.{sh,py}`. It asserts the mission's
preconditions and refuses to launch on failure:

- **auth**: `gh auth status` healthy; the *right identity* (App token vs
  `an0mium`) is active for this mission's traffic class.
- **keys**: required provider keys present for the heterogeneity this mission
  needs (and explicitly NOT assuming a single model — if the configured
  orchestrator model is unavailable, the probe must confirm a fallback exists,
  not fail the mission; this is the Fable-5 lesson encoded).
- **CLI surfaces**: the commands the mission invokes resolve (e.g.
  `review-queue merge-packet --help`, `pr_state_cache.py read --help`).
- **worktree**: an isolated worktree exists and is clean (per CLAUDE.md
  worktree-isolation rule).
- **gate context**: the merge gate is reachable and its tier rules are loadable
  — so the harness does not "re-research the repo's merge gate from scratch."

Probe verdict protocol mirrors validation contracts: exit 0 / nonzero + a
machine line. The harness MUST run the probe and gate launch on it; a failed
probe parks the mission (does not disclose-and-proceed — program operating
lesson: "tier escalations park by default").

## 7. How a harness loads and executes a mission

The contract any harness implements (Claude Code, Codex, Factory). Reference
loader/runner: `scripts/mission_run.py` (stdlib + the schema validator),
deliberately thin so each harness can re-implement it natively if preferred:

1. **Parse + validate** front-matter against `mission.schema.json`. Invalid →
   refuse, print the schema error. (No execution on an unvalidated mission.)
2. **Readiness probe**: run `readiness_probe`; park on failure (§6).
3. **For each phase in DAG order** (`depends_on`):
   a. If `human_checkpoint` or `tier >= 3`: pause for explicit operator
      authorization (any channel — program lesson 6: "a human must decide ≠ a
      human must type"; unforgeability comes from credential isolation, not
      ceremony). For Tier 4, the authorization is the scarmani-settled
      preapproval per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.
   b. Enforce `yield_rules`: never edit `path_freeze` paths; in `odr_yield`
      surfaces, steer via comments to the declared owner rather than editing.
   c. Track budget consumption against `budgets`; on breach, park and report
      (never silently overrun — backpressure at admission, program §1).
   d. Do the work (harness-native).
   e. Run the phase's `acceptance` validation contracts (§5). All must pass to
      mark the phase done. A failing contract parks the phase with the contract's
      evidence — it does NOT advance on a self-reported done.
4. **Mission done** only when all phases are contract-satisfied. Emit a mission
   receipt (binds to the contracts that passed and their evidence), suitable for
   the lane ledger and — for Tier 3+/witnessed missions — the TET intent chain.

The harness is swappable because every decision point is data in the schema
(tiers, checkpoints, budgets, yield rules) or an external executable (probe,
contracts), not a model judgment baked into one orchestrator. Swap the model
running step 3d and the governance still holds.

## 8. Migration / rollout

- **Phase A — schema + validator, no enforcement.** Publish
  `mission.schema.json` and `scripts/mission_run.py validate`. Add front-matter
  to ONE existing plan (this event-substrate spec is the natural first subject,
  since its phases and exit metrics are already written). Tier 1 (additive).
- **Phase B — probes + contracts for that one mission.** Author the
  readiness probe and the validation contracts referenced by the front-matter;
  prove `mission_run.py` parks correctly on a failing probe and on a failing
  contract. Tier 2.
- **Phase C — backfill the corpus.** Add front-matter to the other active
  `docs/superpowers/plans/*` missions; the schema must round-trip them without
  information loss (if a real mission can't be expressed, the schema is wrong —
  extend it). Tier 1–2.
- **Phase D — harness adoption.** Document the loader contract (§7) so Codex and
  Factory runs consume the same missions; the orchestrator is then demonstrably
  swappable. Tier 2.

Exit metric (falsifiable): the same mission file, executed by two different
harnesses/models, reaches the same per-phase contract verdicts. If they diverge,
a decision that should be data is still living in a model's head — find it and
move it into the schema, a probe, or a contract.

## Cross-references

- Program: `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md` (§4 harness comparison + absorb items; §5.2).
- Epic: [#8344](https://github.com/synaptent/aragora/issues/8344) (medium-term phase 2; phase 3 validation contracts + readiness probes).
- Tier classification + checkpoint semantics: `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.
- Witness/receipt anchoring for Tier 3+ missions: `docs/specs/TAMPER_EVIDENT_TRAIL.md`.
- First subject mission: `docs/superpowers/plans/2026-06-13-event-substrate-design.md`.
- Related blocked work this discipline protects: #8315, #8316, #8343.
