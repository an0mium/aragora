# Frontier AI Tooling vs the Human→AI Leverage Ratio — Research Brief

> **Status:** NON-CANONICAL RESEARCH BRIEF.
> **Authority:** None. This document is analysis and framing, not roadmap. It does
> not bind architecture, create tracks, add scope, or authorize tool adoption.
> **Gate:** Per [docs/FOCUS.md](../FOCUS.md) Sprint 2 anti-goals, the current
> obligation is load-bearing product proof, not new tooling/substrate. Any
> recommendation here that graduates into work must do so through an existing
> track and survive an evidence artifact — never as a new lane.
> **Tier:** Tier 0 (docs-only). Nothing here changes review-authority code,
> merge-quorum/review-gate workflows, or family-eligibility (those are Tier 3/4).
> **Date:** 2026-05-29

## Purpose

Survey AI tooling released by frontier labs in the last ~90 days and decide which
(if any) should be adopted to raise the **leverage ratio** on Aragora, *without*
exacerbating the project's documented imbalances. The brief states the question
operationally, grounds the imbalances in measured signals, classifies each
candidate by its effect on the leverage ratio, and proposes one bounded
validation pilot. Vendor performance claims are flagged as requiring local
validation before they bind anything.

## The question, stated operationally

Define the leverage ratio:

```
L = (verified, accepted AI output) / (human cognition spent steering + reviewing + settling)
```

A tool is valuable here **only if it raises L**. A tool whose primary effect is
more raw AI generation throughput *lowers* L for this project, because the
binding constraint is not production capacity — it is one human's capacity to
comprehend, verify, and settle what agents already produce.

## The imbalances, measured (2026-05-29)

These are not assertions; they are signals pulled from the repo and Aragora's own
read-only tooling on this date.

| Signal | Value | Source command |
|---|---|---|
| Remote branches | **467** (codex **223**, claude 12, droid 3; rest `worktree-*`/fix/feat/dependabot) | `git branch -r` |
| Local worktrees | **167** | `git worktree list` |
| Open automation-outbox handoffs | **91** | `ls .aragora/automation-outbox/` |
| Live ranked work items | **182** | `aragora work robot --json` |
| Commits, last 14 days | 264 (238 by operator identities `an0mium`+`Armand`; ~7 harness) | `git log --since="14 days ago"` |
| Accountable Tier 3-4 risk settlers | **1** (the operator) | [REVIEW_AUTHORITY_PRINCIPLES.md](../REVIEW_AUTHORITY_PRINCIPLES.md) |

The project's own vocabulary already names this: **"producer:merger ratio,"**
**"substrate-overbuild,"** and **"breadth over depth"** ([FOCUS.md](../FOCUS.md)).
The top `aragora work robot` recommendation (PR #7519) is classified `ready` but
is `BLOCKED` with **owner unknown**; many outbox items are `needs-polish` with
*objective / acceptance-criteria / owner missing*. Codex is the dominant
producer; the human is the sole high-stakes verifier. **That gap is the low
leverage ratio.**

The imbalances any recommendation must NOT worsen:

1. **Producer:merger gap** — more branches/PRs/handoffs produced than one human can review/settle.
2. **Substrate-overbuild** — agents building settlement/review/steering meta-tooling instead of product proof (explicit Sprint 2 anti-goal).
3. **Breadth scope creep** — ~1.48M LOC, ~25% off the core decision-integrity thesis.
4. **Owner-clarity / acceptance-criteria gap** — queued work lacking owner and a machine-checkable "done".
5. **Calibration gap** — AI reviewers not yet trusted to replace human risk settlement for Tier 3-4.

## Candidate tools, classified by effect on L

| Tool (release) | What it does | Effect on **L** | Verdict |
|---|---|---|---|
| **Opus 4.8 — honesty/calibration gain** (2026-05-28) | Less overconfident; better citation precision | **Raises** — attacks the *calibration gap* that keeps the human in every Tier 3-4 loop | **Pilot** on the existing advisory review-gate; validate locally before trusting |
| **Opus 4.8 — Effort Control** (2026-05-28) | Dial reasoning effort per call | **Raises** — spend more compute on review/verification, less on routine generation | **Adopt** on the quorum / settlement-evidence path |
| **Codex `/goal`** (2026-05) | Durable objective + *verifiable stopping condition*; self-validates for hours | **Both** — raises L for bounded, machine-verifiable, product-load-bearing targets; **lowers** L for open-ended feature work | **Adopt narrowly** — only goals with a checkable "done" |
| **Claude Code "Follow a plan"** | Approve plan once, agent executes within boundary | **Raises** — front-loads steering into one approval | **Adopt** — matches existing spec→dispatch pattern |
| **Extended thinking / effort levels** (`ultrathink` deprecated 2026-01-17) | Tiered thinking budget | **Raises** when applied to verification/synthesis | **Adopt**; retire the dead `ultrathink` keyword |
| **`/security-review` + GitHub Action** | Automated independent security pass | **Raises** — adds an independent verification signal | **Adopt** as an extra *advisory* quorum signal |
| **Analytics API / Cowork OpenTelemetry** | Telemetry on agent activity/spend | **Raises** — more operator visibility into the fleet | **Adopt** if harness usage justifies |
| **Cowork Dispatch / scheduled tasks** | Fire-and-forget autonomous task generation | **Lowers** — manufactures more unreviewed work | **Avoid** for production; OK for read-only digests |
| **More parallel agents / worktrees** | Wider fan-out | **Lowers** — 167 worktrees already exceed review capacity | **Avoid** |
| **Any new settlement/review/steering meta-tooling** | — | **Lowers** — textbook substrate-overbuild | **Avoid** (trips Sprint 2 anti-goal) |

## Determination

A specific subset helps, and the selection rule is the one the project already
lives by. **Adopt** the tools on the **verification / steering / calibration**
side of the frontier:

- Opus 4.8 honesty + Effort Control aimed at the *review and settlement* path.
- Codex `/goal` only against *verifiable, bounded, product-load-bearing* objectives
  (e.g. drive B0 `truth_success_rate` upward; prune dead `server/` handlers until
  tests stay green) — never open-ended feature generation.
- "Follow a plan" and effort/extended-thinking on synthesis and verification.
- `/security-review` and fleet telemetry as additional independent signals.

**Reject** the tools whose primary effect is more autonomous generation — Cowork
Dispatch for production, additional parallel worktrees, `/goal` on open-ended
work, and any new meta-tooling. Those deepen the exact 467-branch / 182-item /
one-settler gap this brief measures.

**Highest-leverage single move:** use Opus 4.8's honesty/calibration gain to raise
the ceiling of what the heterogeneous model quorum can settle *without* the human
— but only after validating the gain against Aragora's own calibration metrics,
and never by silently changing which family counts at which Tier.

## Claims requiring local validation

These vendor claims are cited for context and **do not bind anything** until
measured locally on Aragora's own corpus:

- Opus 4.8 "~4x honesty gain" and coding-benchmark gains over Opus 4.7.
- Opus 4.8 Fast Mode "~2.5x faster / ~3x cheaper".
- Codex `/goal` "works independently for many hours" reliability.

Aragora's calibration bar (Brier scores, ELO, dissent preservation in
[REVIEW_AUTHORITY_PRINCIPLES.md](../REVIEW_AUTHORITY_PRINCIPLES.md)) is the
authority, not the press release.

## Proposed bounded validation pilot (proposal only — no behavior change)

This proposal does **not** wire anything into CI and changes no gate. It is
specified here so it can be approved or rejected as a unit later.

1. Run Opus 4.8 (with raised Effort) as an **additional advisory reviewer** on the
   existing `aragora-review-gate.yml` lane only — reusing current infrastructure,
   adding **no** new tooling, scripts, lanes, or worktrees.
2. On a sample of already-settled Tier 0-2 PRs, capture the existing quorum
   evidence fields (head SHA, model identity, recommendation, dissent) plus a
   calibration score.
3. **Success metric:** does Opus 4.8's advisory verdict agree with the operator's
   eventual settlement at a higher *calibrated* rate than Opus 4.7? That measured
   number — not the vendor claim — is the only thing that could later justify
   proposing raised quorum trust.

### Explicitly out of scope (Tier 4 — requires preapproval discipline)

Changing *which model family counts at which Tier*, or otherwise loosening the
merge-quorum gate, is a **Tier 4 merge-authority self-modification**. Per
[REVIEW_AUTHORITY_PRINCIPLES.md](../REVIEW_AUTHORITY_PRINCIPLES.md) it requires a
`docs/specs/` design doc, failing `tests/governance/` tests pinning the current
behavior, and operator preapproval before implementation and before merge. None
of that is authorized by this brief.

## Sources

- Anthropic — *Introducing Claude Opus 4.8* (anthropic.com/news/claude-opus-4-8) and Claude release notes (support.claude.com), 2026-05-28.
- The New Stack / DigitalApplied — Opus 4.8 effort controls + dynamic workflows coverage, 2026-05-28.
- OpenAI — *Follow a goal* / `/goal` use case (developers.openai.com/codex/use-cases/follow-goals), 2026-05.
- Decode Claude / ClaudeLog — `ultrathink` deprecation → effort levels, 2026-01-17.
- Aragora repo, 2026-05-29: `git branch -r`, `git worktree list`, `aragora work robot --json`, `.aragora/automation-outbox/`, `git log`.
