# Quorum-evidence runbook: getting `aragora-merge-quorum` to count model review signals

Operator runbook for the single most common blocker on otherwise-green PRs:
`aragora-merge-quorum` reporting **0/2** (no counted heterogeneous model-review
signals on the exact head). This documents the **supported, non-fabricated** path
to attach review evidence the gate will count. It changes no merge-authority code;
it only explains existing tooling. Read with `docs/specs/MODEL_LINEAGE_DISCLOSURE.md`
(the counted-comment contract) and `docs/governance/MERGE_GATE_RECONCILIATION.md`.

## How the gate decides (read-only)

`.github/workflows/aragora-merge-quorum.yml` runs:

```
aragora review-queue merge-packet --pr <N> --repo <repo> --json
```

and reads the PR's packet `entry.status`:

- `satisfied` → **pass** (Tier 0-2: a model quorum alone authorizes settlement).
- `repair_or_wait` / `needs_model_review_quorum` → **fail** (checks failing/pending,
  or not enough counted model signals).
- `unresolved_dissent` → **fail** (needs a human).
- `human_preapproval_required` / `human_risk_settlement_required` → **pass only if**
  an `aragora/human-settlement` commit status = `success` exists for the exact head
  (Tier 3-4). Quorum evidence alone never greens a Tier 3-4 PR.

Draft PRs short-circuit to pass ("deferred until ready_for_review").

## What actually gets counted

`merge-packet` builds `counted_model_families` from **PR comments** (and any stored
debate protocol) via `aragora.cli.commands.review_queue`. A comment contributes a
counted family only when **all** hold:

1. **Exact-head grounding.** The comment body cites the head SHA (≥7-char prefix)
   **or** its `createdAt` is at/after the head commit's `committedDate`. A comment
   that predates the head and cites no SHA is treated as stale and dropped.
2. **Non-GitHub-Actions author.** `github-actions` / `github-actions[bot]` never count.
3. **Recognizable reviewer surface** in the **first markdown heading** (not deep in
   the body): a direct family (`Claude`, `OpenAI`, `Gemini`, `Grok`, …) or a router
   surface (`factory`, `codex`, `tesla`, `harvey`).
4. **Canonical model family disclosed.** Router surfaces **must** include a
   `**Model family:** <canonical>` line in the structured block right after the
   heading. Direct family headings self-map (and must not conflict with an explicit
   `Model family` line).
5. **A trigger token** appears in the body:
   - dogfood evidence: one of `dogfood`, `adversarial`, `cross-author`, `recheck`;
   - model-review signal: one of `independent model review`,
     `model-family semantic signal`, `independent semantic review`,
     `codex review`, `claude review`, `grok independent`, `gemini independent`.
   Include **both** kinds so the comment satisfies the dogfood requirement *and*
   the signal count.

**Quorum is met** when `counted_model_families` has **≥ the tier's required count**
(2 for the Tier 1-2 default) **distinct** families, **and** (for tiers requiring it)
at least one dogfood comment from a known model reviewer. Two comments disclosing
two *different* canonical families is the normal way to reach 2/2.

Canonical families: `claude`, `openai`, `gemini`, `grok`, `mistral`, `deepseek`,
`qwen`, `kimi`, `yi`, `glm`, `minimax`, `hermes`.

## The non-negotiable: no fabrication

These comments must reflect **real** reviews produced by the disclosed model family.
Do not hand-author a "claude" and an "openai" comment yourself. Generate genuine
heterogeneous-model reviews of the exact head diff via aragora's own review tooling
(the validation-gate "adversarial review = a debate, not a comment" step), e.g. an
`aragora ask` / review pass over `git diff <base>...<head>` per model family, then
format each model's output per the template below. The `Receipt artifact` should
point at the saved review output. (Missing receipt is a diagnostic, not a count
blocker — but record it.)

## Counted-comment template

```md
## Claude independent model review on head <FULL_HEAD_SHA>

**Reviewer harness:** claude
**Model family:** claude
**Model id:** claude-opus-4
**Receipt artifact:** <local path or URL to the saved review>

Independent model review (adversarial dogfood recheck) of head <FULL_HEAD_SHA>.
- <substantive finding 1>
- <substantive finding 2 / "no blocking issues">
```

For a router surface, the heading names the harness and the `Model family` line
carries the lineage:

```md
## Factory independent semantic review on head <FULL_HEAD_SHA>

**Reviewer harness:** factory
**Model family:** openai
**Model id:** gpt-5.5
**Receipt artifact:** <...>

Independent model review (adversarial cross-author recheck) ...
```

## Pre-flight every comment with `evidence-lint` (read-only, no network)

Before posting, dry-run each body against the **same parsers** the gate uses:

```
aragora review-queue evidence-lint \
  --pr <N> --head-sha <FULL_HEAD_SHA> --author <your-login> \
  --body-file review.md --json
```

Require `"would_count": true` and an empty `problems` list. Validated examples
(against PR #7876, head `9a5ac49f3b…`):

- Direct `## Claude …` body with `Model family: claude` + head SHA + tokens →
  `would_count: true`, `counted_model_families: ["claude"]`, `problems: []`.
- Router `## Factory …` body **without** a `Model family:` line →
  `would_count: false`, `problems: ["missing_model_family_disclosure", …]`.
- Body that cites **no** head SHA →
  `would_count: false`, `problems: ["missing_current_head_grounding", …]`.

## Post, then re-evaluate

1. Post each linted comment to the PR with a non-GHA identity:
   `gh pr comment <N> --body-file review.md`.
2. **Re-trigger the gate.** Posting a comment does **not** itself re-run
   `aragora-merge-quorum` (its triggers are opened/synchronize/reopened/
   ready_for_review). Re-run the latest run for the head
   (`gh run rerun <run-id>`) or push a new head commit. The job checks out the
   default branch and reads the live comments, so a rerun picks up new evidence.
3. Confirm locally any time with
   `aragora review-queue merge-packet --pr <N> --repo <repo> --json` →
   `counted_model_families` should list ≥2 families and `status` should flip to
   `satisfied` (Tier 0-2).

## Tier 3-4 caveat

For Tier 3-4 PRs (merge-authority, security/RBAC, public API, migrations, etc.),
a satisfied model quorum only prepares the packet. The check still requires the
operator's exact-head `aragora/human-settlement` commit status. Use
`scripts/settle_tier4_pr.py --check --pr <N> --head <SHA>` and record settlement
per `docs/governance/MERGE_GATE_RECONCILIATION.md`.

## Scope

This is operator documentation for existing, supported tooling
(`review-queue evidence-lint` / `merge-packet`) and the lineage-disclosure contract.
It is Tier 0 (docs): it modifies no merge-authority code, gate, or workflow.
