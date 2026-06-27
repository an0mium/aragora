# ADC v0.1 → v0.4 Stack-Coherence Audit

**Date:** 2026-05-19T13:35:00Z
**Auditor:** claude-B061F80D
**Trigger:** Factory recommended a 3-check audit before fanning out to v0.5–v0.8
**Verdict:** **GREEN** — all checks pass; the four versions compose cleanly without merge conflicts.

## Why this audit exists

Aragora Delegation Contract v0.1 → v0.4 was built by **four agents in parallel** in one session:

| Agent | Family | Shipped | PR |
|---|---|---|---|
| claude-B061F80D | Claude Code (Anthropic) | v0.1 schema + predicate oracle | [#7357](https://github.com/synaptent/aragora/pull/7357) |
| claude-B061F80D | Claude Code | v0.2 lane-registry hookup | [#7358](https://github.com/synaptent/aragora/pull/7358) |
| droid-6D2D7294 | Factory Droid (`droid exec --auto high`) | v0.3 progress-ledger periodic evaluator | [#7360](https://github.com/synaptent/aragora/pull/7360) |
| droid-adc-v04-1779196049 | Factory Droid | v0.4 HMAC-SHA256 signing | [#7361](https://github.com/synaptent/aragora/pull/7361) |

The parallel build itself is the protocol-being-built working in real-time: the pattern ADC is meant to govern is the pattern that just shipped it. But without an audit, parallel-build risk is real: independent contributors can drift on schema, fork the trust kernel, or introduce silent merge conflicts.

## Checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| A | v0.3 imports v0.1's PredicateOracle (not a fork) | **PASS** | `from aragora.policy import evaluate_predicate` in `scripts/evaluate_goal_progress.py` |
| B | v0.4 populates v0.1's `signature` field (no parallel field) | **PASS** | `_SIGNATURE_FIELD = "signature"` constant; v0.1's `validate()` relaxed in-place from "must be None" → "None OR valid hex, verified when env key set" |
| C | v0.3 progress keyed via correct lane→contract→goal chain | **PASS** | Keyed off `GoalSpec.goal_id` per v0.1 design; multi-step `lane (delegation_contract_id) → contract (goal_id) → ledger` chain holds |
| D | v0.3 + v0.4 don't conflict on `aragora/policy/__init__.py` | **PASS** | v0.3 makes no `__init__.py` changes; v0.4 appends below v0.1 |
| E | v0.4 properly updated v0.1's signature test | **PASS** | `test_root_contract_rejects_v01_signature` renamed to `..._accepts_hex_signature_in_v04`; new companion `..._rejects_non_hex_signature` added |
| F | v0.3 ↔ v0.4 merge | **PASS** | `git merge-tree` reports 0 conflict markers |
| G | v0.2 ↔ {v0.3, v0.4} merges | **PASS** | `git merge-tree` reports 0 conflict markers for both |

## Cross-cutting observations

1. **Import discipline held.** Both droids `from aragora.policy import …` rather than reaching into module internals. Trust kernel remains unified.
2. **Modification policy held.** v0.4 modifies v0.1 files (necessary for signature relaxation), but only in narrow version-explicit places (one validate() block, one test rename). v0.3 doesn't touch any v0.1 file.
3. **Schema versioning is clean.** `CONTRACT_SCHEMA_VERSION` stayed `"aragora-delegation-contract/0.1"`; v0.4 introduced a separate `SIGNING_SCHEMA_VERSION = "aragora-contract-signing/0.4"` — additive, not breaking.
4. **`is_signed` property** added by v0.4 as a derived predicate, not a stored field. Correct.
5. **Receipts.** v0.4 ships `sign_receipt` / `verify_receipt` helpers that match the contract-signing canonical-JSON discipline. v0.3's progress ledger entries are unsigned in v0.3 itself — they can be wrapped by `sign_receipt` later without ledger-schema change.

## Recommended merge order

All four PRs are mergeable in any order (zero conflicts). The cleanest cognitive order minimizes rebase cost:

| Step | PR | Why this order |
|---|---|---|
| 1 | **#7357 (v0.1)** | Schema foundation. |
| 2 | **#7361 (v0.4)** | Modifies v0.1 files (validate() relaxation + test rename). Landing v0.4 next means v0.2 and v0.3 see the relaxed validator + the new `contract_signing` module on main, eliminating rebase risk for them. |
| 3 | **#7358 (v0.2)** | Lane-registry hookup. Independent of v0.4; rebases cleanly to a 2-file diff once v0.1 + v0.4 are on main. |
| 4 | **#7360 (v0.3)** | Progress ledger. Independent of v0.2 + v0.4; rebases to a 2-file diff once v0.1 lands. |

Steps 3 + 4 can also be parallel.

## What this audit doesn't cover

- **Runtime end-to-end semantics.** Audit verifies composition at code-shape and merge-tree level. The full sign-claim-evaluate-progress stack hasn't been smoke-tested end-to-end. That's a Stage 2 audit, runnable after all four PRs land on main.
- **Cross-family policy intersection.** Factory's point that v0.8 is the real validation that this protocol governs the families that built it is the dogfooding moment, addressed in the resequenced roadmap.

## Roadmap resequencing (per Factory's recommendation)

| Order | Version | Owner | Rationale |
|---|---|---|---|
| Next | **v0.8** (cross-family adapter — `--parent-contract` on launchers) | **inline (claude + factory co-development)** | Highest leverage; dogfoods immediately on the very session that built it. Cross-family negotiation needs human steering, not droid work. |
| Parallel with v0.8 | **v0.7** (three-tier reversibility: pause / halt / revoke) | droid mission | Independent surface; addresses safety-incident path. |
| After v0.8 + v0.7 land | **v0.5** (continuous adversarial spot-check) | droid mission | Now checks contracted workers, not just self. |
| After v0.5 lands | **v0.6** (ELO trust multiplier) | droid mission | Now adjusts trust on adversarial-check + contracted-receipt evidence across families. |

## My refinement on Factory's audit protocol

Factory's audit listed three composition checks. I added four more (D-G) covering cross-PR merge feasibility + test-rename correctness + cross-cutting `__init__.py` discipline. Future N-version-in-parallel sessions should run the full A-G set, not just A-C, because **composing at the import/schema level (A-C) does not imply composing at the merge-tree level (D-G).** The two can fail independently.

I also added the **merge-order plan** as a twin artifact alongside this audit. Right-shape composition (audit A-G GREEN) can still produce N-PR rebase pain if landed out of order. Audit + merge-order plan are twin durable artifacts for every future parallel-build session.

## Significance

Four versions of a multi-PR authority-protocol stack shipped end-to-end in one session, by two different agent families (Claude Code + Factory Droid), through a third coordination protocol (Aragora lane registry + receipt-trio convention), and composed cleanly without a single merge conflict.

That's the protocol-being-built working in real-time. The cross-family permission-laundering gap that motivated ADC is the gap that v0.8 closes — and v0.8 is the dogfooding moment: if v0.8 ships and the next droid dispatch in this session is contract-bound, we will have built and self-applied an authority protocol in one continuous run.
