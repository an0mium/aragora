# Inbox Trust Wedge Activation Plan

Last updated: 2026-04-17
Status: planning-truth artifact; not a live `boss-ready` lane under the current queue gate

Related:
- `docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md`
- `docs/strategy/PROOF_AND_EVIDENCE.md`
- `docs/status/NEXT_STEPS_CANONICAL.md`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`
- `docs/status/DESIGN_PARTNER_PROGRAM.md`
- `docs/examples/inbox-trust-wedge-activation-prompt-pack.yaml`
- `docs/examples/inbox-trust-wedge-activation-sources.yaml`
- `docs/templates/INBOX_TRUST_WEDGE_PROOF_PACK_TEMPLATE.md`
- `aragora/cli/commands/triage.py`
- `aragora/cli/commands/inbox_wedge.py`

## Purpose

Turn the repo's existing inbox trust wedge thesis into durable execution truth:

- one stable operator plan
- one automation-ready prompt pack
- one machine-readable source manifest
- one proof-pack template
- one bounded GitHub issue set mapped to the canonical roadmap codes

This plan exists because Aragora already has a real kernel, but the repo is too
broad and too internally complex to justify widening the story without repeated
workflow proof. The right move is not more substrate breadth. The right move is
making one consequential inbox workflow repeatable, auditable, and easy for a
non-builder to run.

## Queue Posture

This document does **not** change the live queue rule in
`docs/status/NEXT_STEPS_CANONICAL.md`.

Current canonical posture:

- `CS-01..03` remains the only short-horizon **Do now** set
- trust-wedge inbox work may exist as planning truth and bounded backlog
- no `TW-*` inbox activation issue created from this plan should carry
  `boss-ready` until the canonical queue gate is deliberately changed

Use this plan to preserve execution truth and prepare bounded follow-on work,
not to silently widen the live queue.

## Why This Wedge

The repo already commits to this wedge in multiple places:

- `docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md` says the founder loop is proven and
  the current objective is the inbox trust wedge plus design-partner readiness
- `docs/strategy/PROOF_AND_EVIDENCE.md` defines internal dogfood success around
  repeated inbox runs, operator transferability, truthful stopping, and proof
  packs
- `aragora/cli/commands/triage.py` already exposes the founder-facing command
  surface: `auth`, `run`, `status`, `queue`, `label`, `digest`, `audit`, and
  `calibrate`
- `aragora/cli/commands/inbox_wedge.py` already exposes the receipt loop:
  `create`, `review`, `show`, `list`, `execute`, `report`, and `export`

This is the fastest route from "there is real machinery here" to
"someone else can trust and use one workflow repeatedly."

## Current Product Truth

What is already true on current `main`:

- the inbox trust wedge is structurally implemented
- receipt-before-action is part of the intended execution contract
- Gmail auth, dry-run, and approval-gated execution surfaces exist
- operator telemetry surfaces already exist for queue, digest, audit, and
  calibration
- the proof ladder already defines the exact unlock bar for internal dogfood

What is not yet proven:

- 10 consecutive live runs over at least 5 business days on a real internal
  inbox
- <=10 minute time-to-first-useful-result for 2 non-builder operators
- >=70% accepted-action rate over a 2-week window
- zero false-success incidents across that window
- one compact proof pack that a design partner can inspect

## Activation Criteria

Treat the inbox trust wedge as activated for internal dogfood only when all of
the following are true:

1. 10 consecutive live runs complete over at least 5 business days on a real
   internal inbox.
2. 2 internal operators other than the primary builder can authenticate, run
   the workflow, and reach first useful result in 10 minutes or less from the
   written runbook.
3. Over a 2-week window, at least 70% of runs end in an accepted action, and
   every remaining run stops truthfully with a blocker class and next action.
4. Zero false-success incidents occur.
5. Every executed action has a persisted receipt and reviewable state.

These are the only criteria that unlock the next rung. Connector breadth,
additional docs, and generic platform polish do not.

## Non-Goals

Do not use this plan to justify:

- `AGT-*` activation
- broad provider or connector expansion
- reply, send, or forward automation
- generalized org substrate claims
- dashboard-first polish detached from the live inbox path
- enterprise assurance sprawl before wedge proof
- broad cleanup work not tied to inbox proof or proof-pack generation

## 30-Day Execution Order

### Week 1: Internal baseline and first truthful runs

- run `aragora triage status`
- complete `aragora triage auth`
- run the wedge in `--dry-run` mode on a real internal inbox slice
- switch to approval-gated live mode only after receipt creation and review are
  visible and correct
- fix only blockers that prevent repeated real runs

### Week 2: Operator transferability

- put 2 non-builder operators through the written path
- measure setup-to-first-useful-result time
- tighten auth, runbook, and blocker messages until the path is usable without
  founder repo spelunking
- keep the allowed action surface narrow: `ARCHIVE`, `STAR`, `LABEL`, `IGNORE`

### Week 3: Proof-pack assembly

- finish the 10-run / 5-business-day internal proof bar
- export representative receipts and blocker examples
- compile the accepted-action rate, truthful-stop rate, latency, and override
  evidence into one proof pack
- keep claims narrow: "receipt-gated inbox triage works repeatably on internal
  consequential workflows"

### Week 4: Design-partner prep

- freeze one repeatable runbook and one proof pack
- run one bounded external dry-run or approval-gated design-partner rehearsal
- decide whether the wedge is ready for recurring partner use, still internal
  only, or not yet fit

## Exact Operator Surfaces

Readiness and auth:

```bash
python3 -m aragora.cli.main triage status
python3 -m aragora.cli.main triage auth
```

Dry-run and live founder/operator use:

```bash
python3 -m aragora.cli.main triage run --batch 5 --dry-run
python3 -m aragora.cli.main triage run --batch 5
python3 -m aragora.cli.main triage run --batch 5 --auto-approve
```

Receipt review and inspection:

```bash
python3 -m aragora.cli.main inbox-wedge list --limit 20
python3 -m aragora.cli.main inbox-wedge show <receipt_id>
python3 -m aragora.cli.main inbox-wedge review <receipt_id> --choice approve
python3 -m aragora.cli.main inbox-wedge execute <receipt_id>
```

Telemetry and evidence export:

```bash
python3 -m aragora.cli.main triage queue --limit 20
python3 -m aragora.cli.main triage digest --hours 24
python3 -m aragora.cli.main triage audit --batch 20
python3 -m aragora.cli.main triage calibrate --json
python3 -m aragora.cli.main inbox-wedge report --limit 200
python3 -m aragora.cli.main inbox-wedge export /tmp/inbox-wedge.jsonl --limit 200
```

## Automation-Ready Artifacts

Use these machine-readable artifacts for bounded long-running execution:

- prompt pack: `docs/examples/inbox-trust-wedge-activation-prompt-pack.yaml`
- source manifest: `docs/examples/inbox-trust-wedge-activation-sources.yaml`

Suggested tranche workflow:

```bash
python3 -m aragora.cli.main swarm tranche plan \
  --from-prompts docs/examples/inbox-trust-wedge-activation-prompt-pack.yaml \
  --output .aragora/tranches/inbox-trust-wedge-activation/tranche.yaml \
  --json

python3 -m aragora.cli.main swarm tranche inspect \
  --manifest .aragora/tranches/inbox-trust-wedge-activation/tranche.yaml \
  --json

python3 -m aragora.cli.main swarm tranche design-review \
  --manifest .aragora/tranches/inbox-trust-wedge-activation/tranche.yaml \
  --json
```

Operational rule:

- use tranche automation for bounded code/doc/status lanes
- keep actual inbox actioning human-gated until the activation criteria are met
- stop rather than bluff when the lane needs live credentials, operator policy,
  or external partner input

## Proof-Pack Contract

Every inbox activation cycle should leave behind:

- exact commands run
- run window and operator identity
- receipt bundle paths or IDs
- accepted-action counts and truthful-stop counts
- latency, cost, override, and time-to-first-result metrics
- representative blocker examples with next actions
- one explicit gate decision:
  - `internal_ready`
  - `partner_dry_run_ready`
  - `not_ready`

Use `docs/templates/INBOX_TRUST_WEDGE_PROOF_PACK_TEMPLATE.md` as the canonical
shape.

## Mapping To Canonical Roadmap Codes

This plan maps directly to the existing trust-wedge backlog:

- `TW-04` Preserve receipt-before-action guarantees for inbox workflows
  ([#6159](https://github.com/synaptent/aragora/issues/6159))
- `TW-05` Reuse contracts, memory, and approval policies across non-code actions
  ([#6160](https://github.com/synaptent/aragora/issues/6160))
- `TW-06` Capture operator feedback tied to receipts
  ([#6161](https://github.com/synaptent/aragora/issues/6161))
- `TW-10` Establish weekly design-partner operating cadence on real workloads
  ([#6162](https://github.com/synaptent/aragora/issues/6162))
- `TW-11` Publish truthful proof packs and before/after benchmarks
  ([#6163](https://github.com/synaptent/aragora/issues/6163))
- `TW-12` Decide which wedges graduate to packaged offerings
  ([#6164](https://github.com/synaptent/aragora/issues/6164))

Issue creation from this plan should preserve those codes verbatim and keep the
issues out of the live `boss-ready` queue until the short-horizon queue gate is
updated explicitly.

## Stop Conditions

Stop and re-plan if any of the following happen:

- a run executes an action without a persisted valid receipt
- the workflow claims success while the operator would classify it as a blocker
- the queue or plan widens into broad substrate or connector work
- operator transferability stalls because the path still requires repo
  spelunking
- the proof pack cannot be assembled from durable receipts and visible outputs

## Success Definition

This plan succeeds when Aragora can truthfully say:

> The inbox trust wedge is repeatable on internal consequential work, every
> action is receipt-gated, operators can reach first useful result quickly from
> a written path, and the resulting proof pack is strong enough to support a
> bounded design-partner dry run.
