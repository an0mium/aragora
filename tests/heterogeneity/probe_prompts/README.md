# Round 30f β Probe — Prompt Seed Set (50 prompts)

This directory holds the 50 authored prompts for the heterogeneity contamination probe. Design rationale lives in `docs/plans/2026-04-30f-beta-contamination-probe-design.md`.

## Layout

```
clean_neutral/         # 8 prompts (no errors; control)
single_seeded_error/   # 12 prompts (1 subtle objectively-wrong claim each)
multi_seeded_error/    # 8 prompts (2 independent errors each)
correlated_priming/    # 12 prompts (seeded error + misleading framing)
red_team_paraphrase/   # 8 prompts (same error in 3 stylistic surfaces)
null_negative/         # 2 prompts (asks for flag, no error present)
```

Total = 50 authored. Pilot draws 20 with class quotas: `{cn:4, sse:6, mse:3, cp:4, rtp:2, nn:1}`.

## YAML front-matter schema (per prompt file)

See `docs/plans/2026-04-30f-beta-contamination-probe-design.md` §4.1.

## Authoring rules summary

- `clean_neutral`: every claim verifiable against `origin/main` HEAD.
- `single_seeded_error`: error is subtle, single-reference-checkable, not discoverable via tool use.
- `multi_seeded_error`: errors in different categories (factual + logical, not two factual).
- `correlated_priming`: misleading framing is plausible, topically adjacent, not directly related to error; no two prompts share framing.
- `red_team_paraphrase`: identical underlying error claim across all 3 stylistic surfaces.
- `null_negative`: drawn from `clean_neutral` pool.

## Verification refs

Each prompt with a `verification_refs` field cites the file/line where the underlying claim is verified. These references must resolve against `origin/main` at probe-run time. The probe runner SHOULD validate these at startup.

## Ground-truth labels

Each prompt's `seeded_error` field holds the canonical ground-truth label used by the judge prompt. Errors are deliberately **not described in the prompt body itself** — the body is what the panelist sees.
