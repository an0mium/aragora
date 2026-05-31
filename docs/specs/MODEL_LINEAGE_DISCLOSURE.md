# Model Lineage Disclosure for Reviewer Attestations (Tier 4 Pre-Approval)

**Status**: design doc / pre-approval artifact for a Tier 4 merge-authority
change. **Not implemented in this PR.** This file + the failing-current-
state governance tests in
`tests/governance/test_model_lineage_disclosure_recognizer.py` are the
pre-approval artifact for the future implementation PR, per
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`.

## Problem statement

The recognizer `_infer_model_reviewer_from_text` in
`aragora/cli/commands/review_queue.py` (lines 2266–2290 as of
`4dfb8729c9`) reads the first markdown heading of a PR comment and
maps it to one of seven family markers:
`claude / codex / tesla / harvey / factory / grok / gemini`. The
returned family marker is what the merge-quorum evaluator counts.

**The recognizer collapses harness identity over underlying model
lineage.** A comment headed `## Factory focused dogfood` is counted
as the `factory` family regardless of which underlying model Factory
actually ran. Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md` Tier-
eligibility table, "factory" is a Western family for jurisdiction
purposes — but Factory.ai is a *router* that can dispatch to any
underlying model the operator has configured (GPT-5.5, Claude
Opus 4.7, Gemini, DeepSeek, etc.). The same harness identity can
deliver Western or non-Western lineage on the same PR with no
recognizer-visible difference.

This was independently surfaced three times in the same week:

1. **Operator devil's-advocate**: "factory is a harness and can run
   Western or non-Western models" — flagged the lineage gap during
   the #7451 settlement cycle.
2. **Codex audit (2026-05-27)**: "Model identity remains a governance
   weak point. The GPT-5.5/GPT-5.2 confusion and Factory-as-router
   issue show that app labels, harness names, and underlying model
   lineage must be treated as separate facts."
3. **PR #7451 evidence** (head `113a706c92`): comment
   `## Factory independent semantic review on head ...` from
   `an0mium`, body declaring `**Reviewer:** Factory Droid (GPT-5.5)`.
   The recognizer counted it as `factory` (Western harness) but the
   actual technical review was produced by OpenAI's GPT-5.5
   (Western lineage in this case — but the gate has no machinery to
   verify that, and the same shape could deliver Chinese-routed
   lineage undetected).

Two concrete failure modes the current state permits:

- **Heterogeneity laundering**: two `## Factory ...` signals from
  two Factory runs that both routed to Anthropic Claude would look
  heterogeneous to the recognizer (two distinct family markers
  expected — though here both are "factory" so it's actually the
  same family, but consider `## Factory ...` + `## Codex ...` both
  routed to OpenAI by Factory; that LOOKS heterogeneous at the
  family level and IS NOT at the model level).
- **Jurisdiction laundering**: a `## Factory ...` heading produced
  by a Factory run routed to a Chinese-routed model would count as
  Western for the Tier 2 "≥1 Western family" floor — silently
  defeating the jurisdictional payload boundary the principles doc
  is built to defend.

## Non-goals

- **No change to the seven-marker recognizer tuple.** Family marker
  expansion (e.g., adding `openai`, `anthropic`, `mistral`, `deepseek`,
  `qwen`, `kimi`) is governed by separate pre-approval (PR #7472 and
  the family-expansion design in #7450). This design is orthogonal: it
  layers lineage disclosure on top of the existing seven markers.
- **No change to `aragora-merge-quorum.yml`** semantics for which
  Tier needs which family floor. The Tier-eligibility table in
  `REVIEW_AUTHORITY_PRINCIPLES.md` is unchanged.
- **No change to `scripts/settle_tier4_pr.py`**. The trusted-operator-
  allowlist, exact-head binding, and admin-squash settlement path
  remain unchanged.
- **No automatic rejection of router-based signals** that lack
  lineage disclosure. This design *records* lineage when present and
  *flags* signals when absent; the counting behavior is in the
  variant-choice section below and is operator-decided.

## Proposed contract

The recognizer is extended to additionally parse a structured
`**Reviewer:**` (or `**Reviewer lineage:**`) line from the comment
body. The line declares the actual model that produced the review,
separately from the harness identity in the heading.

### Comment body contract

For a comment to be a *fully-disclosed* signal, the body must
contain a line matching the regex:

```python
LINEAGE_REGEX = re.compile(
    r"^\s*\*{0,2}Reviewer(?:\s+lineage)?:\*{0,2}\s*"
    r"(?P<harness>[A-Za-z][A-Za-z0-9 .\-_/]*?)"
    r"\s*\((?P<model_id>[a-z][a-z0-9 .\-_/]*)\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
```

This matches lines like:

- `**Reviewer:** Factory Droid (gpt-5.5)`
- `**Reviewer:** Factory Droid (claude-opus-4-7)`
- `**Reviewer:** Claude Code (claude-sonnet-4-5)`
- `**Reviewer:** Codex CLI (gpt-5-codex)`
- `**Reviewer:** Aragora harness (grok-4-3-mini)`
- `**Reviewer lineage:** Factory Droid (gpt-5.5)`

The `model_id` capture is normalized by a new model-lineage prefix
table introduced by the implementation PR. The table's provider names
must stay aligned with the real agent-provider vocabulary validated by
`aragora/agents/spec.py::AgentSpec` via
`aragora.config.settings::ALLOWED_AGENT_TYPES`; it does not depend on a
pre-existing provider allowlist constant in `aragora/agents/spec.py`.

| model_id prefix | normalized model family |
| --- | --- |
| `gpt-*`, `openai-*`, `o1-*`, `o3-*` | `openai` |
| `claude-*`, `anthropic-*` | `anthropic` |
| `gemini-*`, `palm-*` | `google` |
| `grok-*`, `xai-*` | `xai` |
| `mistral-*`, `codestral-*` | `mistral` |
| `deepseek-*` | `deepseek` |
| `qwen-*` | `qwen` |
| `kimi-*`, `moonshot-*` | `kimi` |
| `llama-*` | `meta` |
| anything else | `unknown_model_lineage` |

The harness identity is recorded but not normalized for counting
purposes — it's audit-trail metadata.

The parser applies `LINEAGE_REGEX` only to body prose outside fenced
Markdown code blocks. A reviewer-shaped line inside a fenced code block,
for example `Reviewer: SomeAgent (some-model-v1)`, is example text, not
an operator attestation, and must not populate `model_lineage`.

### Recognizer return shape

The recognizer currently returns `str` (a family marker like
`"factory"` or `"unknown_model_reviewer"`). The proposed return is a
`tuple[str, str | None]` or a small dataclass:

```python
class RecognizedReviewer(NamedTuple):
    family_marker: str          # e.g. "factory" (from heading)
    model_lineage: str | None   # e.g. "openai" (from body), or None if undisclosed
```

`family_marker` retains the exact current behavior. `model_lineage`
is `None` when the comment body has no parseable Reviewer-lineage
line. Existing call sites that consume only the family marker
continue to work via `recognized.family_marker`; new call sites can
choose to weight by `model_lineage`.

### Two enforcement variants — operator choice

The recognizer extension is the same in both variants; what differs
is how the merge-quorum evaluator treats `model_lineage is None`.

**Variant A — soft (record-only)**. The merge-quorum evaluator
counts the signal regardless of lineage disclosure but records
`lineage_undeclared: true` on the per-signal record in the merge-
packet. Operators can audit lineage-undeclared rates and ratchet
toward Variant B over time. This is the minimum-disruption choice.

**Variant B — hard (structural requirement)**. The merge-quorum
evaluator does NOT count signals whose `model_lineage is None`
for Tier 2+ PRs. Tier 0-1 signals continue to count regardless
(matches the lower-stakes principle). This is the
maximum-discipline choice and prevents jurisdiction laundering.

The implementation PR proposes **Variant A first**, with a planned
follow-on PR (also Tier 4 pre-approval) to migrate to Variant B
after a documented N-week observation period where the
`lineage_undeclared` rate is measured.

## Tier classification

Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md` Tier 4 row ("merge-authority
self-modifications") AND family-additive change governance, this is
**Tier 4**: it changes the merge-quorum-counting contract that gates
which PRs can merge to `main`.

The implementation PR will need:

- this design doc, merged;
- the governance tests in
  `tests/governance/test_model_lineage_disclosure_recognizer.py`,
  merged in this PR (they pin the current state as the regression
  floor);
- explicit operator Tier 4 preapproval *at the implementation step*
  before the recognizer code changes;
- explicit operator Tier 4 preapproval *at the merge step* before
  the recognizer-counting change goes live on `main`.

## Risk dimensions

| # | Risk | Mitigation |
| --- | --- | --- |
| 1 | Brittle parsing — natural-language variation in how reviewers describe their underlying model. | The regex requires the `**Reviewer:**` (or `**Reviewer lineage:**`) prefix with a model ID in parentheses. This is a structural convention, not free-form prose. Governance tests pin ≥10 positive and ≥10 negative examples. Operator-attestation comments are templated documents, not natural-language essays, so the contract is enforceable. |
| 2 | Mis-disclosure — operator (or compromised credential) declares `**Reviewer:** Factory Droid (claude-opus-4-7)` when the run was actually `gpt-5.5`. | This design *records* the disclosure; it does not verify it cryptographically. The next-tier hardening (out of scope of this design) would be receipt-binding: requiring a receipt artifact under `.aragora/proof/<topic>/<timestamp>/` and checking the receipt's manifest matches the disclosed lineage. Pre-approval for that hardening is a separate future Tier 4 design queued behind this one. |
| 3 | Backwards compatibility — existing comments without the `**Reviewer:**` field (e.g., the 5+ in-flight signals on #7479, #7451, #7480) would become lineage-undeclared. | Variant A grandfathers existing comments (they continue to count). Variant B (when adopted) applies only to comments posted after the variant-B-implementation PR lands; previously-posted comments retain their counting status. |
| 4 | Heading/body disagreement — comment headed `## Claude ...` but body declares `**Reviewer:** Factory Droid (gpt-5.5)`. Is this a recognizer-bug, an operator-error, or a deliberate cross-attribution claim? | The recognizer returns `family_marker="claude"` (from heading) and `model_lineage="openai"` (from body). The merge-quorum evaluator surfaces the discrepancy as `family_lineage_mismatch: true` in the per-signal record, and (in Variant B) optionally treats the signal as `unknown_model_reviewer`. The variant-B treatment is operator-decided in the follow-on pre-approval. |
| 5 | Model-ID drift — new model IDs (e.g., `gpt-5.6`, `claude-opus-5-0`) won't match the prefix table. | The prefix table is a *living* document. Adding new model-ID prefixes to the normalization table is a Tier 4 pre-approval pattern (same shape as #7472 for family additions). Governance tests pin the current prefix table; updates require pre-approval. |
| 6 | Counted-quorum impact — Variant B would retroactively un-count many existing signals on open PRs. | Variant A is recommended for first implementation precisely to avoid this. Variant B's follow-on PR ships with an explicit observation-period and a per-PR transition policy (e.g., `lineage_disclosed_after: <date>` field in the merge-packet). |
| 7 | Fenced-code spoofing — a review body includes example code containing `Reviewer: SomeAgent (some-model-v1)`, and a naive regex treats it as an attestation. | The parser contract is regex-plus-context: skip fenced Markdown code blocks before applying `LINEAGE_REGEX`. Governance tests pin this as a required safety floor so implementation cannot accidentally count example text as model lineage. |

## Failing governance tests (regression floor)

`tests/governance/test_model_lineage_disclosure_recognizer.py` pins
the current state as the regression floor:

- **Current recognizer return shape**: the recognizer returns a
  bare `str` today, NOT a NamedTuple/dataclass. The implementation
  PR changes this. If the test fails on the current code, the
  recognizer has been changed without pre-approval discipline.
- **Current collapse behavior**: `## Factory ...` headers all return
  `"factory"` regardless of body content. Pin the current uniform
  behavior so the implementation PR can be verified to break it
  intentionally.
- **Parser contract for `**Reviewer:**`**: ≥10 positive examples
  matching the regex outside fenced code blocks (covering all major
  model families); ≥10 negative examples (missing prefix, missing
  parens, missing model-id, mis-spelled keyword, fenced-code example
  text).
- **Model-ID prefix normalization**: each entry in the prefix table
  has a positive test. Unknown prefixes produce `unknown_model_lineage`.
- **Body-detection precedence**: the parser scans the entire body,
  not just the first heading. (This is a DIFFERENT scope from the
  current recognizer's heading-only scan; the lineage parser is a
  body-scan addition, not a replacement.)
- **Backwards-compatibility floor**: comments without the
  `**Reviewer:**` field continue to return `model_lineage=None` (not
  some default-guess); under Variant A this still counts toward
  quorum, under Variant B it does not.

After the implementation PR lands and the recognizer is extended,
the current-collapse-behavior tests invert (they assert the new
disambiguated behavior), per the regression-floor pattern in
`tests/governance/test_model_quorum_recognizer_gaps.py` and
`tests/governance/test_advisory_review_recognizable_header.py`.

## Implementation PR plan (out of scope of this PR)

A future PR will:

1. Replace `def _infer_model_reviewer_from_text(text: str) -> str`
   with `def _infer_model_reviewer_from_text(text: str) -> RecognizedReviewer`
   (or equivalent NamedTuple). Existing call sites use
   `.family_marker`; new call sites use `.model_lineage`.
2. Add the body-scanning `LINEAGE_REGEX` parser, fenced-code-block
   filtering, and the prefix-normalization table as module-level
   constants/helpers in `aragora/cli/commands/review_queue.py`.
3. Update `aragora-merge-quorum.yml` (or its evaluator script) to
   record `model_lineage`, `lineage_undeclared`, and
   `family_lineage_mismatch` per per-signal record in the merge-
   packet (Variant A behavior).
4. Update the recognizer's docstring with the new return shape and
   the variant-A counting behavior.
5. Convert the current-state regression-floor tests in this PR's
   governance suite to positive-present assertions, per the pattern
   in #7472 / #7450.
6. Dogfood on a real PR (e.g., post a `## Factory ...` comment with
   and without the `**Reviewer:**` field and verify the merge-
   packet differentiates them).

## Operator preapproval requested

Approving this PR constitutes preapproval to *draft* the implementation
PR described above with **Variant A** (soft, record-only). Variant B
(hard, structural requirement) requires a *separate* future pre-approval
artifact after observation-period data is collected.

This PR contains no live recognizer change. Until the implementation
PR lands, the recognizer continues to collapse harness identity over
underlying model lineage, and the Factory-as-router gap remains the
governance weak point Codex's audit named.
