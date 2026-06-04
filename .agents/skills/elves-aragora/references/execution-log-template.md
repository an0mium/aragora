# Execution Log — <run name>

> Chronological, append-only record of work, decisions, commands, review outcomes, receipts, and
> settlement state. This is the proof of what is done. If a batch appears here as complete with a
> verified receipt, it is complete — do not re-implement it.

- **Plan:** [path]
- **Branch / worktree:** [feat/...] / [../aragora-<branch>]
- **Default branch:** [main]
- **Started:** [YYYY-MM-DD HH:MM tz]

---

## Preflight

- `pre-commit run --all-files`: [result]
- `mypy aragora` vs `.mypy-baseline`: [result]
- `pytest <slice>`: [N passed, baseline count]
- `aragora --help` / `aragora api-key list`: [ok / missing keys]
- Worktree + branch ownership confirmed: [yes], tip recorded: [SHA]
- Blockers: [none / list]

---

## Batch N: <name>

- **Predicted tier / final tier:** [X / Y]
- **Tier 4 pre-approval (if applicable):** [n/a | obtained before implementation at <ts/ref>]
- **Rollback tag:** `elves/<branch>/pre-batch-N` (pushed: [yes])
- **Draft PR:** [#NNNN] — head SHA: [SHA]
- **Scope delivered:** [bullets]
- **Commands run + results:**
  - `pre-commit run --all-files` → [pass]
  - `mypy aragora` → [no new errors above baseline]
  - `pytest <slice>` → [N passed; total count not decreased]
- **Required checks (`gh pr checks <pr> --required`):** [all green @ SHA / list reds]
- **Model-quorum evidence (head-grounded):**
  - head SHA reviewed: [SHA]
  - `aragora review-pr` reviewers + families: [claude→anthropic / codex→openai / ...]
  - independent of authoring lane: [yes/no]
  - recommendation: [accept / changes]
  - dissent: [none / summary + disposition]
  - `review-queue evidence-lint` would_count: [true] — evidence comment: [link/id]
  - adversarial dogfood: [note]
- **Receipt:** [.aragora/receipts/<file>.json] — `aragora verify` → [PASS]
- **Authorization surfaces:**
  - `review-queue merge-packet --pr <pr>` → [satisfied / blocked: ...]
  - `scripts/settle_one_pr.py --pr <pr>` → [blockers == ['PR is draft']]
- **Settlement:**
  - Tier 0-2 → [marked ready; protected squash (never --admin); settlement recorded]
  - Tier 3 → [PAUSED — awaiting `aragora/human-settlement`; packet prepared at <path>; PR kept draft]
  - Tier 4 → [pre-approved before impl; PAUSED for human merge approval; `settle_tier4_pr.py --check` <result>]
- **Commit:** [SHA] (`Co-authored-by: codex[bot]`) — pushed: [yes]
- **Decisions made:** [durable notes worth keeping]
- **Docs touched:** [files / "none needed"]
- **Time:** [Xm]

---

## Completed Archive

> When this log gets large, move older fully-complete + settled batch entries here in place.

---

## Open human-settlement queue

| Batch | Tier | Receipt | Packet path | Requested at | Status |
| --- | --- | --- | --- | --- | --- |
| [N] | [3/4] | [path] | [path] | [ts] | [pending/accepted/rejected] |
