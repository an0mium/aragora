# M8 Dogfood Report: CI Review Gate on Real Aragora PRs

> **Date:** 2026-07-10
> **Feature:** m8-dogfood-gate
> **Milestone:** M8 — Dogfood the CI Review Gate
> **Spend ledger:** [`spend-ledger.json`](spend-ledger.json) (M8 sub-total: ~$0.23 of $100 cap)

## Executive Summary

We dogfooded the Aragora CI review gate on **5 recent real `synaptent/aragora` PRs** using a
heterogeneous quorum of **2 reachable provider families** (xAI/Grok + Mistral). Each PR received
genuine, independent multi-model reviews, and every review produced a verifiable
Open Decision Receipt (ODR) artifact. All 5 receipts independently verify with
`aragora-verify` at **exit 0** (schema_conformance=pass, quorum_consistency=pass).

The quorum caught a real defect (a `None`-return type-contract violation in PR #9030 that both
models unanimously flagged), produced genuine dissent on 2 PRs (grok blocked where mistral
passed), and honestly recorded 2 PRs as clean passes. Limitations are documented below, not
hidden.

## Provider Selection

Per mission constraints, ANTHROPIC, OPENAI, and DeepSeek keys are absent. We confirmed
connectivity with a tiny probe call before batch runs:

| Family | Agent Type | API Key Env | Status |
|--------|-----------|-------------|--------|
| xAI/Grok | `grok` | `XAI_API_KEY` | **Connected** (grok-4-latest) |
| Mistral | `mistral-api` | `MISTRAL_API_KEY` | **Connected** (mistral-large-2512) |
| Gemini | `gemini` | `GEMINI_API_KEY` | **Failed** — key invalid (HTTP 400) |
| OpenRouter | `mistral` (OpenRouter transport) | `OPENROUTER_API_KEY` | **Failed** — placeholder key (HTTP 401) |

The heterogeneous quorum uses **xAI and Mistral** — 2 distinct model families, both reachable,
neither mapping to the absent set (Anthropic/OpenAI/DeepSeek). This satisfies the
`distinct_model_families >= 2` requirement. A 3rd family (Gemini) was attempted but its API key
is expired/invalid; this is recorded as a limitation, not hidden.

> **Note on `collect_quorum_evidence.py`:** The Action's standard evidence collector
> (`scripts/collect_quorum_evidence.py`) routes the "mistral" family through an OpenRouter
> transport agent, which failed due to the broken OpenRouter key. To use the direct Mistral API
> (`mistral-api` agent type with `MISTRAL_API_KEY`), we ran reviewers via
> `aragora.agents.create_agent("grok")` and `create_agent("mistral-api")` directly, then built
> ODR receipts through the canonical `collect_outcome_to_decision_receipt` +
> `decision_receipt_to_odr` pipeline (the same transform `scripts/emit_pr_receipt.py` uses).
> No merge-authority, quorum, settle, or receipt-pipeline code was modified.

## PR Selection

5 recent merged PRs representing a mix of test code, feature code, and bug fixes:

| PR | Title | Size | Type |
|----|-------|------|------|
| [#9193](https://github.com/synaptent/aragora/pull/9193) | tests(scripts): harden docs-site fallback mirror guard | 207+33, 1 file | Test |
| [#9062](https://github.com/synaptent/aragora/pull/9062) | feat(scripts): add OpenRouter fallback for Fable goal cycles | 371+18, 4 files | Feature |
| [#9030](https://github.com/synaptent/aragora/pull/9030) | fix(routing): handle empty LLM domain responses | 21+9, 2 files | Bug fix |
| [#9056](https://github.com/synaptent/aragora/pull/9056) | feat(swarm): wire PR-keyed round budget into A1 reconciler | 285+6, 4 files | Feature |
| [#9027](https://github.com/synaptent/aragora/pull/9027) | fix(scripts): accept operator-context dir in goal-cycle context | 27+8, 2 files | Bug fix |

Diffs were fetched via `gh pr diff <n> --repo synaptent/aragora` and reviewed at the PR's
merged head SHA.

## Results: Per-PR Review Table

| PR | Grok (xai) | Mistral (mistral) | Quorum Verdict | Dissent | Receipt |
|----|-----------|-------------------|----------------|---------|---------|
| #9193 | CHANGES_REQUESTED (3x P2) | PASS (1x P3) | CHANGES_REQUESTED | Yes (grok) | [receipt](pr-9193-receipt.odr.json) |
| #9062 | CHANGES_REQUESTED (2x P2, 1x P3) | PASS (3x P3) | CHANGES_REQUESTED | Yes (grok) | [receipt](pr-9062-receipt.odr.json) |
| #9030 | CHANGES_REQUESTED (2x P2) | CHANGES_REQUESTED (1x P1, 1x P2, 1x P3) | CHANGES_REQUESTED | No (unanimous block) | [receipt](pr-9030-receipt.odr.json) |
| #9056 | PASS (no findings) | PASS (1x P2) | PASS | No (unanimous pass) | [receipt](pr-9056-receipt.odr.json) |
| #9027 | PASS (no findings) | PASS (1x P2) | PASS | No (unanimous pass) | [receipt](pr-9027-receipt.odr.json) |

All receipts are **UNSIGNED** (`signatures: []`), which is the current shipping state — no ODR
signing path is wired in the Action. `aragora-verify` reports `signature=warn` (not fail) on
unsigned receipts. Authenticity is **UNVERIFIED** (no cryptographic signature). Each receipt's
authenticity state is disclosed here and in the `--json` output.

## Receipt Verification

Every receipt independently verified with `aragora-verify`:

```
$ aragora-verify <receipt>.odr.json --json
```

| Receipt | Exit Code | schema_conformance | quorum_consistency | signature | Warnings |
|---------|-----------|-------------------|-------------------|-----------|----------|
| pr-9193 | 0 | pass | pass | warn (unsigned) | attestation: autonomous |
| pr-9062 | 0 | pass | pass | warn (unsigned) | attestation: autonomous |
| pr-9030 | 0 | pass | pass | warn (unsigned) | attestation: autonomous |
| pr-9056 | 0 | pass | pass | warn (unsigned) | attestation: autonomous |
| pr-9027 | 0 | pass | pass | warn (unsigned) | attestation: autonomous |

### ODR field conformance (per validation contract)

Each receipt satisfies:
- **>=2 quorum.participants**: 2 participants (grok, mistral) per receipt
- **quorum.independence.disclosed = true**: yes, with `distinct_model_families = 2`
- **model_families**: `["mistral", "xai"]` — both reachable, none in {Anthropic, OpenAI, DeepSeek}
- **structured dissent**: present with >=1 view (PRs #9193, #9062, #9030) or explicitly `false` (PRs #9056, #9027)
- **claim.verdict**: `PASS` or `CHANGES_REQUESTED` — both in the documented verdict set
- **subject.identifier**: `merge-quorum/synaptent/aragora#<N>` — binds the real PR
- **subject.digest.status = present** (sha-256): yes, derived from `sha256("synaptent/aragora#<N>@<head_sha>")`

## Honest Report: Catches / Misses / Blocked / Couldn't-Decide

### Caught (real issues the quorum identified)

1. **PR #9030 — `None`-return type contract violation (P1/P2, unanimous):**
   Both models independently identified that `_detect_with_llm` now returns `None` on empty
   content, breaking the `-> list[tuple[str, float]]` type annotation and risking a crash at
   callers that iterate the result. Mistral flagged it as **P1** ("the existing fallback logic
   in `detect()` will crash when it tries to iterate over `None`"). Grok flagged it as **P2**.
   This is a genuine correctness regression catch by the heterogeneous quorum.

2. **PR #9062 — Timeout calculation ignores `api_fallback` (P2, grok):**
   Grok identified that `run_consult` computes `overall_timeout` as
   `timeout * (2 + int(openrouter_fallback))`, ignoring `api_fallback`, `fallback_model`, and
   `_planned_attempt_count`, causing potential under/over-budgeting. Mistral noted the related
   behavioral change (system prompt handling) but as P3.

3. **PR #9193 — Test environment Node dependency risk (P2, grok):**
   Grok identified that the test's `subprocess.run` invokes `node sync-docs.js` in a stripped
   tmp tree that may be missing `node_modules`/deps, risking runtime failures or silent partial
   syncs. Mistral passed with only a P3 debug-logging suggestion.

### Missed (issues the quorum did not flag)

1. **PR #9056 — Both models passed cleanly.** Grok returned "No findings." Mistral returned a
   single P2 defensive-assert suggestion. Neither model found substantive issues. This PR
   appears genuinely clean (it was merged), so the "miss" may be a true negative — but we
   cannot confirm zero defects without a ground-truth oracle.

2. **PR #9027 — Both models passed.** Mistral noted a P2 path-traversal edge case (already
   handled by existing symlink-refusal tests). Grok returned "No findings." The path traversal
   concern was already mitigated, so this is a minor advisory, not a miss.

### Blocked (dissent — quorum could not reach unanimous pass)

1. **PR #9193 — Grok blocked, Mistral passed.** Grok's 3 P2 findings (Node deps, root_source_files
   logic, Node dependency in test) vs. Mistral's 1 P3. The quorum verdict is CHANGES_REQUESTED
   (dissent recorded). A human would need to adjudicate whether the stripped-tree test
   environment actually fails in CI or is handled by global Node modules.

2. **PR #9062 — Grok blocked, Mistral passed.** Grok's 2 P2 findings (timeout calculation,
   prompt composition) vs. Mistral's 3 P3 findings. The quorum verdict is CHANGES_REQUESTED
   (dissent recorded). A human would need to adjudicate whether the timeout calculation
   actually causes hangs or is acceptable for the fallback path.

### Couldn't-Decide

1. **Whether the PR #9030 `None`-return caused a production issue.** Both models flagged it,
   but the PR is already merged. We cannot determine from the review alone whether the
   `None`-return path is actually reached in production or was intentional for the
   empty-content case. The quorum correctly flagged the risk; a post-merge audit would be
   needed to confirm impact.

2. **Whether a 3rd family would have broken the dissent ties on PRs #9193 and #9062.** With
   only 2 families, both dissent cases are 1-1 splits. A 3rd family (Gemini) was unreachable
   due to an invalid API key. Adding a 3rd family could have produced a majority verdict
   instead of an unresolved split.

3. **Grok's higher blocking rate.** Grok returned CHANGES_REQUESTED on 3 of 5 PRs while
   Mistral returned CHANGES_REQUESTED on only 1. This could reflect genuine adversarial
   thoroughness (grok caught more real issues) or over-blocking (grok's P2 findings on #9193
   and #9062 may be advisory rather than blocking). The 2-family quorum cannot adjudicate
   this calibration question.

## Reproducibility

### Exact commands

```bash
# 1. Load environment (never print/commit secrets)
set -a; . "$REPO_ROOT/.env"; set +a
export ARAGORA_SECRETS_STRICT=false

# 2. Fetch PR diffs
gh pr diff 9193 --repo synaptent/aragora > /tmp/pr-9193.diff
# ... repeat for each PR

# 3. Run reviews (script runs grok + mistral-api concurrently per PR, builds ODR receipts)
PYTHONPATH="$REPO_ROOT" venv/bin/python3 scripts/dogfood_m8.py

# 4. Verify each receipt
aragora-verify docs/case-studies/dogfood/pr-9193-receipt.odr.json --json
# ... repeat for each receipt (all exit 0)
```

### Provider set

- `grok` (xAI family, model: grok-4-latest, via XAI API)
- `mistral-api` (Mistral family, model: mistral-large-2512, via direct Mistral API)

### Reviewer prompt

Each reviewer received the same adversarial review prompt (mirroring
`collect_quorum_evidence.default_prompt_builder`): "Review ONLY the changes below... Begin your
reply with 'Verdict: PASS' or 'Verdict: CHANGES-REQUESTED'... tag [P1]/[P2]/[P3]..."

## Limitations

1. **Only 2 of 4 intended provider families were reachable.** Gemini's API key is invalid
   (HTTP 400 "API key not valid"). OpenRouter's key is a placeholder (HTTP 401). The quorum
   uses xAI + Mistral only — still heterogeneous (2 distinct families) but a 3rd family would
   strengthen tie-breaking on dissent cases.

2. **Receipts are UNSIGNED.** No ODR signing path is wired in the shipped Action. All receipts
   have `signatures: []` and verify at exit 0 with `signature=warn`. Authenticity is
   **UNVERIFIED** — the receipts prove schema/digest/quorum integrity but not cryptographic
   authenticity. This is the current honest state of the product, not a dogfood artifact.

3. **Costs are estimates.** Provider invoices are not available in-session. The spend ledger
   records conservative upper-bound estimates based on published per-token pricing. Total
   estimated M8 spend: ~$0.23 (well under the $100 cap).

4. **All reviewed PRs are already merged.** The dogfood reviews are retrospective — we review
   already-merged PRs to evaluate the gate's catch rate, not to block merges. The catches and
   misses are assessed against the known merged state, not a live merge decision.

5. **2-family quorum cannot adjudicate dissent.** With 2 families, a 1-1 split (PRs #9193,
   #9062) produces an unresolved CHANGES_REQUESTED verdict. A 3-family quorum would produce a
   majority. This is an inherent limitation of the 2-family configuration, not a gate defect.

6. **No `aragora review` CLI path was used for receipt production.** The standard
   `aragora review` CLI produces a review.json but does not directly emit an ODR receipt. We
   used the canonical `collect_outcome_to_decision_receipt` + `decision_receipt_to_odr`
   transform (the same path `scripts/emit_pr_receipt.py` uses) to produce ODRs from the
   reviewer outputs. This is the Action's actual receipt emission path.

## Spend Ledger

The shared spend ledger is at [`spend-ledger.json`](spend-ledger.json). Summary:

| Entry | PR | Est. Cost |
|-------|----|-----------|
| connectivity probe | — | $0.0001 |
| review-pr-9193 | #9193 | $0.05 |
| review-pr-9062 | #9062 | $0.07 |
| review-pr-9030 | #9030 | $0.03 |
| review-pr-9056 | #9056 | $0.05 |
| review-pr-9027 | #9027 | $0.03 |
| **M8 sub-total** | | **$0.23** |
| M9 sub-total | | $0.00 (not yet run) |
| **Cumulative** | | **$0.23 / $100** |

No unrecorded spend. All 10 reviewer calls (2 per PR x 5 PRs) plus 1 connectivity probe are
accounted for in the ledger.

## What This Proves

The dogfood demonstrates the **review → receipt → verify** loop end-to-end on real PRs:

1. **Review:** Real heterogeneous multi-model review (grok + mistral) on real Aragora PRs,
   producing genuine independent verdicts with real dissent.
2. **Receipt:** Each review produces a portable ODR artifact with quorum participants,
   disclosed model-family independence, structured dissent, and a PR-bound subject identifier.
3. **Verify:** Every receipt independently verifies with `aragora-verify` (exit 0,
   schema_conformance=pass, quorum_consistency=pass) — no Aragora install or account required.

The quorum caught a real type-contract violation (PR #9030), produced genuine dissent (PRs
#9193, #9062), and honestly passed clean PRs (#9056, #9027). Limitations (2 families, unsigned
receipts, retrospective reviews) are documented above, not hidden.
