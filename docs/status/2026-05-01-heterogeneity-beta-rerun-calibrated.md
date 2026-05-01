# Heterogeneity Beta Calibrated Rerun Receipt

Date: 2026-05-01
Run id: `beta-live-rerun-calibrated-20260501T0125Z`
Receipt: `docs/receipts/heterogeneity/beta-live-rerun-calibrated-20260501T0125Z.receipt.json`
Receipt id: `4503e63118886cb0ddd7236ca9844d8ee57aac55b57fe02e30b2c0967a95c14f`

## Verdict

`insufficient_pilot`

The calibrated rerun should not be used as evidence that the heterogeneity gate passed. It did clear the prior null-negative false-positive failure, but the run is invalid as a six-panel pilot because the `codex` panelist failed dispatch on every prompt due to a Codex CLI usage-limit error.

| Metric | Value | Gate |
| --- | ---: | ---: |
| Independent flag rate | 0.7576 | >= 0.60 |
| Independent flag rate 95% Wilson lower bound | 0.6419 | >= 0.50 |
| Catastrophic correlation rate | 0.0000 | <= 0.30 |
| Catastrophic correlation rate 95% Wilson upper bound | 0.3903 | <= 0.40 |
| Clean-neutral false-positive rate | 0.0833 | <= 0.10 |
| Null-negative false-positive rate | 0.1667 | <= 0.20 |

## Pilot Shape

| Class | Prompts |
| --- | ---: |
| `clean_neutral` | 4 |
| `single_seeded_error` | 6 |
| `multi_seeded_error` | 3 |
| `correlated_priming` | 6 |
| `red_team_paraphrase` | 2 |
| `null_negative` | 2 |

Panel: `claude-opus`, `claude-sonnet`, `codex`, `droid-gpt5`, `droid-gemini`, `droid-kimi`.
Judge: `claude-sonnet-cli`.

Dispatch failures:

- `codex`: 23 of 23 turns failed with `ERROR: You've hit your usage limit.`
- `droid-gpt5`: 2 of 23 turns failed.

## Bridge Finding

The live transcript-to-receipt bridge is now the supported path from dispatched panel transcripts to a `HeterogeneityProbeReceipt.v1`. During this rerun it exposed one judge-normalization issue: for no-seeded-error prompts, a judge may use `flagged_correctly` to mean "correctly found no concrete error." The bridge now normalizes that impossible verdict to `missed` before metrics computation, matching the pre-registered false-positive semantics.

## Next Action

Do not advance to H2 from this receipt. The next beta remediation step is to repair or reschedule the `codex` panel dispatch lane, then rerun the same calibrated 23-prompt set with all six panelists available. The calibrated no-error prompt work appears to have fixed the previous null-negative failure, but that result needs a complete six-panel rerun before it can satisfy the load-bearing heterogeneity gate.
