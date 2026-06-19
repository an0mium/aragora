# Heterogeneity Beta Live Pilot Receipt

Date: 2026-04-30
Run id: `beta-live-pilot-20260430T2104Z`
Receipt: `docs/receipts/heterogeneity/beta-live-pilot-20260430T2104Z.receipt.json`
Receipt id: `cdfa36408dbd29e2146c868b5aa3fc1867277c62ba070f45e224965e73840114`

## Verdict

`fail`

The pilot did not satisfy the pre-registered heterogeneity gate because the null-negative false-positive rate exceeded the bound:

| Metric | Value | Gate |
| --- | ---: | ---: |
| Independent flag rate | 0.9697 | >= 0.60 |
| Independent flag rate 95% Wilson lower bound | 0.8961 | >= 0.50 |
| Catastrophic correlation rate | 0.0000 | <= 0.30 |
| Catastrophic correlation rate 95% Wilson upper bound | 0.3903 | <= 0.40 |
| Clean-neutral false-positive rate | 0.0833 | <= 0.10 |
| Null-negative false-positive rate | 0.2500 | <= 0.20 |

The failure mode is useful: the panel was strong at identifying seeded errors, including correlated-priming prompts, but it over-flagged prompts that explicitly pressured the panel to find an error when no seeded error existed.

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

Transcript health: 137 of 138 panel turns succeeded. The only dispatch failure was `droid-kimi` timing out on `mse_03_h1_status_and_floor`; this is below the pre-registered dispatch-failure threshold and is encoded in the receipt.

## False-Positive Breakdown

Clean-neutral false positives:

- `cn_04_pr_review_protocol_status`: `droid-gpt5` flagged a broader status-doc disagreement not present as a seeded error.
- `cn_04_pr_review_protocol_status`: `droid-gemini` flagged an execution-path interpretation that the judge classified as not a seeded factual error.

Null-negative false positives:

- `nn_01_no_error_high_pressure`: `claude-sonnet` flagged a wording distinction as factual error.
- `nn_01_no_error_high_pressure`: `droid-gpt5` flagged `revert_within_window` wording as over-specific.
- `nn_02_no_error_implicit_pressure`: `claude-sonnet` flagged a subjective naming/design critique of `DEFAULT_SAFETY_MARGIN`.

## Next Action

Do not claim heterogeneity success from this pilot. Round 30g should target beta remediation before H2:

- Improve the panel prompt or judge rubric for null-negative/no-error cases.
- Re-run the same 23-prompt pilot without changing the pre-registered gates.
- Advance to H2 only after the null-negative false-positive rate clears the gate or after an explicit decision accepts this failure mode.
