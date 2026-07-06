---
title: Tamper-Evident Audit Trail (TET) — Specification
description: Tamper-Evident Audit Trail (TET) — Specification
---

# Tamper-Evident Audit Trail (TET) — Specification

**Status:** Tier-2 build spec, operator-requested 2026-06-11. Build phases are
loop-executable; the two identity-hardening items marked Tier 4 require
scarmani settlement.

## Threat model (stated honestly)

The adversary **controls the operator's laptop and every credential on it**
(the post-incident lockdown's own assumption — see the May incident and the
`synaptent-incident-readonly` IAM artifact). Under that model:

- **Prevention is impossible.** With the operator's tokens, the adversary *is*
  the operator to every API. No design changes that.
- **Perfect attribution at action-time is impossible.** A log written from the
  compromised machine proves nothing.
- **Detection is achievable** if three properties hold:
  1. **External witness** — events recorded on infrastructure whose write
     credentials never touch the laptop;
  2. **Append-only** — stolen credentials can add entries, never rewrite
     history;
  3. **Reconciliation** — what *happened* (witness) is continuously diffed
     against what was *intended* (anchored intent records); any unmatched
     action is an alarm.

Residual risk that no system removes: a patient adversary acting slowly and
plausibly inside normal agent behavior. TET's goal is that even that adversary
operates **on the record**, betting against future forensics rather than in
the dark.

## Identity layers (already partially live, 2026-06-11)

| Layer | Identity | Property |
|---|---|---|
| Execution | `an0mium` + `aragora-automation-fable[bot]` (App) | attributable automation; App budget separate |
| Evidence | model-quorum receipts, head-bound lineage comments | adversarial review record |
| Oversight | `scarmani` | human-only; credentials never on an agent machine (today: separate browser; target: passkey-on-phone, browser-only sign-in at settle time) |

Hardening items (each Tier 4, scarmani-settled):
- **H1 — CODEOWNERS pin**: `.github/workflows/`, `aragora/cli/commands/review_queue.py`,
  `scripts/settle_*.py` owned by `@scarmani`; branch protection requires
  code-owner review on those paths. Gate changes become mechanically
  impossible without a scarmani approval no agent can produce.
- **H2 — settlement-creator pin**: the quorum evaluation verifies the
  `aragora/human-settlement` status creator is `scarmani` (the statuses API
  exposes `creator.login`; the precedent settlement on #8169 was recorded by
  `an0mium`, which is exactly the gap this closes).

## Component 1 — External witness

Two variants, side by side. **The org upgraded to Enterprise Cloud on
2026-06-11, so Variant B is available and primary; Variant A remains
documented as the fallback (plan downgrade, other orgs/repos, defense in
depth).**

| | **Variant B — Enterprise audit-log streaming** (primary) | **Variant A — org webhook → collector** (fallback) |
|---|---|---|
| Mechanism | GitHub streams the org/enterprise audit log (git events incluse) to S3 continuously | Org webhook fires per event to a small collector (Lambda/CF worker) that appends to S3 |
| Coverage | Full audit-log schema: pushes, merges, branch-protection changes, token/deploy-key/App changes, member/role changes, settings | Webhook event types only (push, PR, issues, member, …) — narrower: token creation and several admin events are NOT webhook-visible |
| Who runs it | GitHub (managed) | We own the collector (more moving parts, more control) |
| Tamper resistance | Stream target: S3 bucket with **Object Lock (compliance mode)** + lifecycle; bucket write-creds = GitHub's streaming role only, NEVER on the laptop | Same bucket pattern; collector creds live only in the cloud function |
| Cost | included in Enterprise ($42/mo for 2 seats, already purchased) | ~$0 (webhook free; pennies of S3/Lambda) |
| Setup | Enterprise settings → Audit log → Streaming → S3 (operator, ~10 min, browser) | Org settings → Webhooks + deploy one function (loop-buildable) |
| Failure mode | GitHub stream lag/outage (GitHub reports stream health) | Collector outage = blind window (sentinel must watch collector liveness) |

Decision rule: run **B now**; build **A only if/when** (a) plan changes, (b) a
second org/repo without Enterprise needs the trail, or (c) we want redundancy
for the highest-assurance period. Do not build A speculatively (YAGNI).

In both variants the bucket is the **witness root**: S3 Object Lock
compliance-mode, region us-east-2, account 767371459652, written by a role the
laptop never holds. The existing MFA-gated `AragoraSecretsRead` posture
extends: a new `AragoraTrailWitnessWrite` role assumable ONLY by GitHub's
streaming integration / the collector function.

## Component 2 — Intent anchoring

The loop already writes intent records (lane ledgers, receipts, settlement
records, run logs). TET makes them **non-repudiable**:

1. **Hash chain**: each intent record gets `prev_hash` + `record_hash`
   (SHA-256 over canonical JSON), appended to
   `.aragora/trail/intent-chain.jsonl` (local working copy).
2. **External anchor**: every N records or T minutes, the chain head hash is
   anchored outside the laptop. Anchor targets, cheapest-first:
   - a GitHub commit status / gist on a dedicated anchor repo (server-side
     timestamped — and the audit stream then *witnesses the anchor itself*);
   - optional upgrade: Sigstore **Rekor** transparency log (public,
     append-only, free) — aligns with Plan v2 Pillar 5 (Open Receipt
     Standard) and is the productizable version.
3. **Contract**: any repo-mutating agent action (publish, merge, settle,
   branch delete) must reference an intent record anchored **before** the
   action. The auto-evidence cycle, publisher, janitor, and settle scripts
   each add their intent-ids to the chain (small wiring, Tier 2 each).

Forging an action now requires compromising BOTH the laptop AND the anchor
target before acting — two independent systems, which is the detection
property we can actually buy.

## Component 3 — Reconciliation (the alarm)

A new sentinel check `trail_reconcile` (extends the live fleet sentinel):

- Pull the last window of witness events (S3 via read-only role with MFA for
  humans; the check itself reads a replicated, read-only view).
- Pull the anchored intent chain.
- **Diff**: every witness event of class \{push, merge, branch delete,
  workflow change, token/key/App change, member change\} must match an
  anchored intent (by repo/ref/SHA/actor-class/time-window).
- Unmatched witness event → **breach** (macOS notification + ledger + morning
  report), severity by event class (token/key/member/workflow = critical).
- Witness silence beyond expected cadence → **blind-period incident** (the
  audit's "silence is never success" rule).
- Token/key/App-change events have NO legitimate agent intent class → always
  critical unless a scarmani-anchored intent exists.

## Build phases (loop-executable; sizes are honest)

| Phase | Deliverable | Tier | Size |
|---|---|---|---|
| T0 | Operator: enable Enterprise audit streaming → S3 (browser); create Object-Lock bucket + `AragoraTrailWitnessWrite` role (CLI commands provided at build time) | operator | ~30 min |
| T1 | Intent hash-chain library + wiring into publisher/auto-evidence/janitor/settle intents | 2 | 1 lane |
| T2 | Anchor publisher (commit-status anchor first; Rekor optional flag — made real by #8231: `aragora/trail/rekor.py` submits hashedrekord entries to the public log with ephemeral-key signing; response-consistency checked, full inclusion-proof/SET verification deferred to ODR-3) | 2 | 1 lane |
| T3 | `trail_reconcile` sentinel check + blind-period accounting | 2 | 1 lane |
| T4 | H1 CODEOWNERS pin + H2 settlement-creator pin | **4 — scarmani settles** | 1 lane + 1 browser approval |
| T5 | Incident replay test: re-enact the May incident class (token created from unknown context) against the reconciler; permanent acceptance test | 2 | 1 lane |

Exit metric (falsifiable): a simulated unauthorized action (T5) raises a
critical breach within one sentinel cycle, with zero alarms on a normal day's
~30-merge agent traffic. If normal traffic can't run alarm-free after tuning,
the matching rules are wrong — fix or honestly downgrade to advisory.

## Product note (why this is not just hygiene)

TET is Plan v2 Pillar 5 + Pillar 6 composed: anchored receipts + fleet
mortality signals + reconciliation = **decision receipts that survive
adversarial conditions** — Aragora's core pitch, dogfooded on its own repo,
demonstrable to any auditor in one sentinel report. The Rekor anchor variant
is the seed of the public Open Receipt Standard verifier.
