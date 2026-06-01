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
- **Rollback tag:** `elves/pre-batch-N` (pushed: [yes])
- **Scope delivered:** [bullets]
- **Commands run + results:**
  - `pre-commit run --all-files` → [pass]
  - `mypy aragora` → [no new errors above baseline]
  - `pytest <slice>` → [N passed; total count not decreased]
- **Adversarial review (aragora debate):**
  - head SHA reviewed: [SHA]
  - reviewer families: [anthropic / openai / grok / ...]
  - independent of authoring lane: [yes/no]
  - recommendation: [accept / changes]
  - dissent: [none / summary + disposition]
  - evidence: [debate id / dogfood note]
- **Receipt:** [.aragora/receipts/<file>.json] — `aragora verify` → [PASS]
- **Settlement:**
  - Tier 0-2 → [auto-settled, packet recorded]
  - Tier 3-4 → [PAUSED — awaiting `aragora/human-settlement`; packet prepared at <path>]
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
| --- | --- | --- | --- | --- |
| [N] | [3/4] | [path] | [path] | [ts] | [pending/accepted/rejected] |
