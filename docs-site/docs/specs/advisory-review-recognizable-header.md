---
title: Lineage-Bound Model Review Quorum (Tier 4 Pre-Approval)
description: Lineage-Bound Model Review Quorum (Tier 4 Pre-Approval)
---

# Lineage-Bound Model Review Quorum (Tier 4 Pre-Approval)

**Status:** design doc / pre-approval artifact for a future Tier 4
merge-authority change. **Not implemented in this PR.** This file and
`tests/governance/test_advisory_review_recognizable_header.py` are the
pre-approval artifact required by
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`.

## Problem Statement

`aragora-merge-quorum.yml` counts model-review signals through
`aragora/cli/commands/review_queue.py`. Today the comment recognizer
infers a reviewer from the first markdown heading and counts the
surface marker it finds: `claude`, `codex`, `tesla`, `harvey`,
`factory`, `grok`, or `gemini`.

That is not enough for router-style tools. `Factory`, `Codex`, `Tesla`,
and `Harvey` are harness/product identities, not necessarily underlying
model lineages. A comment headed `## Factory independent semantic
review...` may disclose `Factory Droid (GPT-5.5)` in the body, but the
merge packet only records `factory`. A second comment headed
`## Codex independent semantic review...` then counts as `codex`, even
though both signals may be OpenAI-lineage. The packet shows two
heterogeneous surface reviewers while proving only one underlying model
family.

This is a quorum-integrity gap, not a cosmetic metadata issue. It can
overstate heterogeneity in Tier 1-2 packets, and it is especially
dangerous for Tier 4 self-modification evidence where the audit trail
must show what model lineages actually reviewed the gate change.

The older version of this design only made the advisory workflow emit a
recognizable family heading. That is insufficient. The future
implementation must bind counted signals to structured underlying model
family metadata.

## Counting Contract

Counted PR-comment evidence must use this structure:

```md
## Factory independent semantic review on head <full-sha>

**Reviewer harness:** factory
**Model family:** openai
**Model id:** gpt-5.5
**Receipt artifact:** <local path or URL>
```

The canonical counted `model_family` IDs are:

`claude`, `openai`, `gemini`, `grok`, `mistral`, `deepseek`, `qwen`,
`kimi`, `yi`, `glm`, `minimax`, `hermes`.

Rules:

1. Router/product surface markers (`factory`, `codex`, `tesla`,
   `harvey`) are not counted families by themselves. They count only
   when the structured `**Model family:** ...` line resolves to a
   canonical model family.
2. Direct family headings (`## Claude ...`, `## Gemini ...`,
   `## Grok ...`, `## OpenAI ...`) may self-map to their model family
   when no explicit `Model family` line is present.
3. If a direct family heading conflicts with an explicit `Model family`
   line, the signal is rejected as identity-conflicted. Example:
   `## Claude ...` plus `**Model family:** openai` does not count.
4. The parser reads only the first markdown heading plus the nearby
   structured metadata block. Body prose, subheadings, quoted diffs,
   and model output text cannot override identity.
5. Comments missing the current head SHA, posted by `github-actions`,
   or resolving to unknown/conflicting identity remain visible in
   `reviewer_signals` but do not contribute to the counted quorum.

## Merge-Packet Contract

For compatibility, `counted_reviewer_ids` remains in the JSON packet,
but after the implementation it contains canonical model-family IDs,
not router/product markers.

The implementation also adds:

- `counted_model_families`: sorted canonical family IDs used for quorum
  counting.
- Per-signal identity fields:
  - `surface_reviewer_id`
  - `model_family`
  - `model_id`
  - `identity_source`
  - `identity_problems`

Router comments with missing or invalid lineage metadata still appear
in `reviewer_signals` with `identity_problems`, so the audit trail shows
why a visible review did not count.

## Tier Policy

This change is Tier 4 because it changes which evidence satisfies the
model-quorum gate. It touches `review_queue.py` behavior in the future
implementation PR, and that code is merge-authority self-modification
per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.

This PR is only the pre-approval artifact:

- design doc
- governance tests that characterize the current unsafe behavior
- no workflow change
- no recognizer/counting implementation change
- no merge/settlement action

The implementation PR requires explicit operator preapproval at the
implementation step and exact-head human settlement before merge.

## Implementation PR Plan

The separate implementation PR will:

1. Add a structured identity parser in
   `aragora/cli/commands/review_queue.py` that returns both the surface
   reviewer marker and the canonical model family.
2. Update comment review-signal extraction and dogfood extraction to
   attach identity metadata and identity problems.
3. Update quorum counting to dedupe/count by canonical model family.
4. Preserve existing exact-head grounding, first-heading safety,
   GitHub Actions exclusion, unknown-reviewer fail-closed behavior, and
   stale-comment exclusion.
5. Update evidence-lint output to report missing model-family
   disclosure, unknown model-family disclosure, and heading/body
   conflicts.
6. Update the advisory-review emitter/workflow contract so generated
   comments include `Reviewer harness`, `Model family`, `Model id`, and
   `Receipt artifact`.

Do not combine this with:

- #7480 settlement-recording work
- PR-A2 family expansion
- any merge-quorum settlement action

## Governance Test Intent

`tests/governance/test_advisory_review_recognizable_header.py`
currently pins the gap that the implementation will invert:

- Factory without `Model family` counts as `factory` today; after the
  implementation it must be advisory-only with
  `missing_model_family_disclosure`.
- Codex without `Model family` counts as `codex` today; after the
  implementation it must be advisory-only unless it discloses lineage.
- Factory(OpenAI) + Codex(OpenAI) counts as two surface reviewers today;
  after the implementation it must count as one model family.
- Factory(OpenAI) + Claude(Claude) currently counts for the wrong
  reason (`factory` + `claude`); after the implementation it must count
  as `openai` + `claude`.
- `## Claude ...` plus `Model family: openai` counts as `claude` today;
  after the implementation it must be rejected as conflicted.
- Body-only and diff-quoted family names must continue not to override
  the first heading.

## Operator Preapproval Requested

Approving this PR constitutes preapproval to draft the separate
lineage-bound implementation PR described above. It does not authorize
merge of the implementation PR. That future PR remains Tier 4 and
requires exact-head model evidence plus explicit operator settlement
before merge.
