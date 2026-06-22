# review_queue.py Decomposition — Tier-4 Preapproval Artifact

Status: **preapproval artifact** for a Tier-4 merge-authority self-modification
(medium-term item 3 of epic [#8344](https://github.com/synaptent/aragora/issues/8344),
§5.3 of `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md`).
Written 2026-06-13.

> **THIS IS A PLAN, NOT AN IMPLEMENTATION.** Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`,
> any change to `aragora/cli/commands/review_queue.py` is **Tier 4** because the
> model-quorum logic that gates the change *is* the code being changed — the
> artifact under review would otherwise be its own arbiter. Tier 4 requires
> **human preapproval before implementation AND human settlement (scarmani) on
> each PR before merge.** This document is the preapproval artifact the operator
> reviews to authorize the work. **No lane may begin implementation until this
> plan is operator-approved, and each sub-PR below requires its own scarmani
> settlement.** Do NOT propose or perform this unattended.

## 1. Why decompose: the Tier-4 bottleneck

`aragora/cli/commands/review_queue.py` is ~5,293 LOC (measured 2026-06-13)
against a lint ratchet. It is the single Tier-4 merge-authority module, and
three pieces of near-term work all queue behind its size and its
self-arbitration property:

- **#8315** — GraphQL→REST fallback for the merge-authority read path.
- **#8316** — `settle_one` retry + `auto_evidence` `transport_blocked_prs`
  surfacing (failure class B; "after the review_queue split or via extraction"
  in the epic's own checklist).
- **#8343** — `collect-evidence` CLI registration (Tier 4, checkpoint flow).

Every edit to this file is a full Tier-4 ceremony, and the file's size makes
each edit's blast radius large and its diff hard to settle. Decomposing it into
cohesive submodules **does not loosen the gate** — it makes each future Tier-4
change small, legible, and surface-pinned, so the operator can settle it
honestly. The CODEOWNERS pin must follow the surface via a glob (`review_queue*`,
per program §3 and the #8324 thread), so modularization never moves
merge-authority code out from under scarmani's pin.

## 2. Precedent / template (#8324)

The decomposition is not novel — it is the continuation of an established
pattern. Already extracted and present on `main`:

- `review_queue_transport.py` (~7.8 KB) — `gh` CLI transport helpers,
  transport-error classification (`transport_blocked`, the GitHub-transport
  markers), JSON-with-retries. (Established by the #8324 line of work.)
- `review_queue_parsers.py` (~5.3 KB) — argparse subparser registration helpers.
- `review_queue_conductor.py` (~54.6 KB) — the owner-aware, read-only conductor
  packet builder, which already imports back from `review_queue` and
  `review_queue_transport` via the re-export discipline below.

The #8324 `review_queue_rest_fallback.py` extraction is **not yet on main** (it
is the in-flight Tier-4 template). This plan sequences the remaining splits the
same way #8324 did its REST-fallback extraction: extract a cohesive unit, leave
a re-export shim in `review_queue.py`, migrate the unit's tests, settle as its
own Tier-4 PR.

## 3. Proposed module boundaries

The thin facade `review_queue.py` retains the public CLI surface (parser wiring
+ `cmd_*` dispatch) and re-exports every symbol other modules or tests import,
so no public import path breaks. Cohesive units extracted:

| Module | Responsibility | Representative symbols (from current `review_queue.py`) | Status |
|---|---|---|---|
| `review_queue_transport.py` | `gh` transport + transport-error classification | `_gh_json`, `_gh_json_with_transport_retries`, `_gh_error_kind`, `_is_github_transport_error` | **exists** |
| `review_queue_parsers.py` | argparse subparser registration | `add_*_parser` helpers | **exists** |
| `review_queue_conductor.py` | read-only owner-aware conductor packet | `QUEUE_CONDUCTOR_*`, conductor builder | **exists** |
| `review_queue_rest_fallback.py` | GraphQL→REST fallback for the check/status read path | `_fetch_direct_commit_check_runs`, `_latest_direct_check_runs_by_name`, `_direct_check_run_*`, `_fetch_required_status_check_protection` | **planned in #8324** (this plan sequences after it) |
| `review_queue_checks.py` | required-check surface summarization + rollup diagnostics | `_fetch_required_pr_check_surface`, `_summarize_required_pr_checks`, `_required_pr_check_bucket`, `_rollup_*`, `_build_check_surface_diagnostics`, `_status_check_*` | new |
| `review_queue_packet.py` | merge-authorization packet building | `_build_packet`, `_build_merge_authorization_packet`, `_explicit_merged_pr_merge_packet_entry`, `_tier_requirement` | new |
| `review_queue_quorum.py` | model-review quorum + tier classification (the gate core) | `_build_model_review_quorum`, `_classify_model_review_tier`, `_reviewer_signals_from_protocol`, `_infer_model_reviewer_from_text`, `ModelReviewIdentity` | new |
| `review_queue_evidence_lint.py` | evidence-lint command logic | `_cmd_evidence_lint` and its helpers | new |
| `review_queue_settlement.py` | settlement-creator pin + settlement receipts | `_post_human_settlement_status`, `_has_recorded_*_settlement`, `_has_tier_four_human_preapproval_comment`, `SettlementReceipt`, `RecordedSettlementResult`, `_resolve_settlement_repo_slug` | new |
| `review_queue.py` (facade) | CLI parser wiring, `cmd_*` dispatch, re-export shims | `add_review_queue_parser`, `cmd_review_queue`, `_cmd_*` | **thin facade** |

Boundary rationale:

- **`review_queue_quorum.py` is the most sensitive unit** — it is literally the
  merge-authority gate (quorum counting, tier classification, the family
  recognizer `_infer_model_reviewer_from_text` whose change
  `docs/REVIEW_AUTHORITY_PRINCIPLES.md` already names as Tier 4). It is extracted
  *last* and most carefully; its PR is the one where self-arbitration risk is
  highest, so its scarmani settlement scrutiny is highest.
- **`review_queue_settlement.py`** carries the H2 settlement-creator pin logic
  (`creator.login == scarmani`); it must land only after #8274's pins are in
  place (§7), so the extraction moves code that is already pin-protected.
- **`review_queue_checks.py` / `review_queue_packet.py`** are mechanical,
  read-only surfaces with the largest LOC win and lowest semantic risk — they go
  early to relieve the lint ratchet for #8315/#8316/#8343.

## 4. Re-export-shim discipline (public symbols)

The pattern already in use (`review_queue_conductor.py` imports
`_build_merge_authorization_packet`, `_fetch_required_pr_check_surface`, etc.
back from `review_queue`). Discipline for every extraction:

1. Move the implementation to the new module.
2. In `review_queue.py`, add `from aragora.cli.commands.review_queue_<unit> import (...)`
   re-exporting **every** symbol that any other module or test referenced from
   `review_queue` — so existing import paths keep working unchanged. The facade
   becomes the stable public surface; the units are the implementation.
3. Avoid import cycles: shared low-level helpers (transport, identity dataclasses)
   live in the leaf modules; the facade and conductor import *from* the units,
   never the reverse for anything load-bearing. Where the current conductor
   imports back from `review_queue`, redirect those imports to the new units once
   the units exist (a follow-up cleanup PR, not a blocker).
4. Each shim PR is verified by: full import of `aragora.cli.commands.review_queue`
   succeeds; every previously-importable symbol still resolves (a characterization
   test enumerates the public symbol set and asserts it is unchanged across the
   extraction — this is the safety net against accidentally narrowing the gate).

## 5. Test-migration plan

- Before any extraction: add a **characterization test** that pins the current
  public symbol set of `review_queue` and the current gate verdicts on a corpus
  of representative packets (Tier 0–4). This is the inversion-proof: the
  decomposition must not change a single gate verdict. (Pattern named in
  `docs/REVIEW_AUTHORITY_PRINCIPLES.md`: governance tests in `tests/governance/`
  that characterize current gate behavior.)
- Per extraction: move the unit's tests alongside the unit
  (`tests/.../test_review_queue_<unit>.py`), keep a thin re-import test ensuring
  the facade still exposes the moved symbols, and re-run the characterization
  test — green characterization is the per-PR acceptance gate.
- The `review_queue_quorum.py` extraction additionally re-runs the
  `tests/governance/` suite that pins family-eligibility-by-tier (the payload of
  `docs/REVIEW_AUTHORITY_PRINCIPLES.md` §"Model family eligibility") to prove the
  gate's counting rules are byte-for-byte preserved.

## 6. What stays in the thin facade

`review_queue.py` keeps only: `add_review_queue_parser` (parser wiring,
delegating to `review_queue_parsers`), `cmd_review_queue` dispatch, the `_cmd_*`
handlers (which orchestrate the extracted units), the public dataclasses kept
for back-compat re-export, and the re-export block. Target: a facade well under
the lint ratchet, with each future Tier-4 change touching one cohesive unit.

## 7. Sequencing — AFTER #8274 (scarmani pins)

This work is sequenced **after #8274 lands** (the 4-commit scarmani sitting: H1
CODEOWNERS pin incl. the `review_queue*` glob, H2 settlement-creator pin). Two
reasons, both load-bearing:

1. The `review_queue*` **glob pin must exist first** so that as code moves into
   `review_queue_*.py` modules, it stays under scarmani's CODEOWNERS authority
   (program §3: "pins must follow surfaces ... because modularization otherwise
   moves merge-authority code out from under the pin"). Decomposing before the
   glob pin would briefly orphan merge-authority code from its owner.
2. `review_queue_settlement.py` extracts the H2 settlement-creator-pin logic;
   that logic should already be the pinned, settled behavior before it is moved.

Also requires the operator to have completed the epic's near-term item: "check
existing broad `@an0mium` CODEOWNERS rules won't freeze automation" (an author
cannot approve their own PR; broad rules would freeze the pipeline) before
require-code-owner-review is enabled.

## 8. Exact sub-PR sequence (each Tier 4, each scarmani-settled)

Each PR below is its own Tier-4 merge-authority PR requiring operator preapproval
(this artifact) plus a per-PR scarmani settlement. They are ordered
lowest-risk-first to relieve the lint ratchet early and defer the gate core last.

| # | PR | Extract | Risk | Unblocks |
|---|---|---|---|---|
| 0 | **prerequisite** | #8274 scarmani pins land; #8324 `review_queue_rest_fallback.py` lands | — | the glob pin + the template |
| 1 | char-test | `tests/governance/` characterization test pinning public symbols + Tier 0–4 verdicts | low | the safety net for all that follow |
| 2 | checks | `review_queue_checks.py` (required-check surface + rollup diagnostics) | low (read-only) | LOC relief for #8315/#8316/#8343 |
| 3 | packet | `review_queue_packet.py` (merge-authorization packet building) | medium (read-only, gate-adjacent) | #8315 (REST fallback in packet path), #8316 |
| 4 | evidence-lint | `review_queue_evidence_lint.py` | low | #8343 collect-evidence registration |
| 5 | settlement | `review_queue_settlement.py` (incl. H2 creator-pin logic) | medium-high (settlement semantics) | clean settlement surface; AFTER #8274 |
| 6 | quorum | `review_queue_quorum.py` (model quorum + tier classification — the gate core) | **highest** (self-arbitration) | the legibility win on the most sensitive code |
| 7 | shim cleanup | redirect conductor's back-imports to the new units; tighten facade | low | final facade < lint ratchet |

Each PR: characterization test green (no gate-verdict change), full
`review_queue` import resolves, public symbol set unchanged, then scarmani
settlement. If any PR's characterization test changes a verdict, that PR is
**not a refactor** — it is a gate behavior change and must be split out and
settled as such (or rejected).

## 9. Explicit Tier-4 guardrails (restated)

- This plan is the **preapproval artifact**; implementation begins only after the
  operator approves it.
- Each of PRs 1–7 requires its own **scarmani settlement** (the
  `aragora/human-settlement` status, creator `scarmani`, per
  `docs/REVIEW_AUTHORITY_PRINCIPLES.md` and the H2 pin).
- A lane may **prepare** the diffs and characterization tests, but MUST park at
  the settlement boundary and MUST NOT self-merge — "tier escalations park by
  default; disclose-and-proceed races the operator" (program operating lesson 4).
- Pure refactor only: zero gate-verdict changes. Any behavior change (incl.
  touching `_infer_model_reviewer_from_text` family eligibility) is a separate,
  separately-preapproved Tier-4 change with its own `docs/specs/` artifact, per
  the family-additive governance rule in `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.

## Cross-references

- Program: `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md` (§3 identity/pins-follow-surfaces, §5.3).
- Epic: [#8344](https://github.com/synaptent/aragora/issues/8344) (medium-term phase 5; near-term #8274 sitting).
- Tier-4 / merge-authority self-modification rule + family eligibility: `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.
- Pins: #8274 (scarmani H1/H2, `review_queue*` glob). Template: #8324 (`review_queue_rest_fallback.py`).
- Unblocked work: #8315 (GraphQL→REST fallback), #8316 (settle_one retry / transport surfacing), #8343 (collect-evidence registration).
