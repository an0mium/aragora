# Model Lineage Disclosure for Reviewer Attestations

**Status**: superseded implementation note. The original #7490
preapproval proposed a soft record-only lineage variant. The later
#7472 design approval selected stricter lineage-bound quorum counting:
counted PR-comment evidence is keyed by canonical underlying model
family, not by router or product surface marker.

This document now records the implemented strict contract so the older
soft-variant text does not conflict with live merge-quorum behavior.

## Problem

The old recognizer inferred a reviewer from the first markdown heading
and counted the surface marker it found: `claude`, `codex`, `tesla`,
`harvey`, `factory`, `grok`, or `gemini`.

That collapsed harness identity over underlying model lineage. A comment
headed `## Factory independent semantic review...` could be produced by
Factory routing to OpenAI, Claude, Gemini, DeepSeek, or another model,
but the merge packet only saw `factory`. Likewise, `## Factory ...` and
`## Codex ...` could look like two heterogeneous reviewers while both
were actually OpenAI-lineage.

The strict contract closes that gap by making the counted unit the
disclosed canonical model family.

## Counted Comment Contract

Router/product comments that should count must include a nearby
structured metadata block immediately after the first markdown heading:

```md
## Factory independent semantic review on head <full-sha>

**Reviewer harness:** factory
**Model family:** openai
**Model id:** gpt-6-astra
**Receipt artifact:** <local path or URL>
```

Canonical counted `model_family` IDs are:

`claude`, `openai`, `gemini`, `grok`, `mistral`, `deepseek`, `qwen`,
`kimi`, `yi`, `glm`, `minimax`, `hermes`.

Rules:

1. Router/product markers `factory`, `codex`, `tesla`, and `harvey`
   do not count by themselves.
2. Router/product markers count only when `**Model family:**` resolves
   to a canonical family.
3. Direct family headings such as `## Claude ...`, `## OpenAI ...`,
   `## Gemini ...`, and `## Grok ...` may self-map when no explicit
   `Model family` line exists.
4. A direct heading plus conflicting explicit `Model family` line is
   rejected as identity-conflicted.
5. Body prose, quoted diffs, fenced code, and later headings cannot
   override the first heading plus nearby structured metadata block.
6. Missing receipt artifact is reported as an identity diagnostic. It is
   not used as the lineage count key.

## Merge-Packet Contract

`counted_reviewer_ids` remains for compatibility, but it now contains
canonical model-family IDs. `counted_model_families` is also emitted as
the explicit canonical lineage list.

Each PR-comment signal includes:

- `surface_reviewer_id`
- `model_family`
- `model_id`
- `identity_source`
- `identity_problems`

Comments with missing, unknown, or conflicting identity metadata remain
visible in `reviewer_signals` or `dogfood_evidence` when their first
heading names a known reviewer surface, but they do not contribute to
quorum counting.

## Preserved Safety Properties

The implementation preserves:

- exact-head grounding
- stale-comment exclusion
- first-heading safety
- GitHub Actions exclusion
- unknown-reviewer fail-closed behavior
- focused dogfood requirements for tiers that require dogfood evidence

## Out Of Scope

This implementation does not modify #7480 settlement-recording behavior,
branch protection, admin-merge behavior, or Tier 4 settlement authority.
The implementation PR itself remains Tier 4 and requires exact-head
model evidence plus explicit operator settlement before merge.
