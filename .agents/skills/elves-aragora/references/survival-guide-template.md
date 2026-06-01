# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

> This is the Survival Guide — your persistent memory across compactions and restarts. After any
> compaction event, read this file before touching code. If it contradicts what you think you
> remember, trust this file. Your memory is gone; this is not.
>
> Core pattern: the Ralph Loop — try, check, feed back, repeat, one sprint-sized batch at a time.
> In aragora the "check" is not just tests: it is the **aragora validation gate** (debate →
> receipt → tier settlement) defined in `references/validation-gate-aragora.md`. The user plans
> and settles; you run the loop in the middle. You never merge unless a merge-on-green preference
> is recorded below, and never for Tier 3-4.
>
> Recommended read order after compaction: survival guide → `.elves-session.json` → learnings →
> plan → execution log → `docs/AGENT_OPERATING_CONTRACT.md` → `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.

---

## Mission

[2-3 sentences. Be specific about scope and the non-negotiable invariants — e.g. "must not change
the SDK public surface", "no schema migrations".]

---

## Run Control

- **Run mode:** [finite | open-ended]
- **Stop policy:** [deadline | explicit-user-stop | blocker-only]
- **User intent:** [exact controlling instruction, e.g. "Back at 8am ET"]
- **Checkpoint due by:** [YYYY-MM-DD HH:MM tz | none]
- **Checkpoint semantics:** [delivery target only | hard stop | none]
- **May continue after checkpoint:** [yes | no]
- **Actual stop conditions:** [one sentence]
- **Workspace ownership:** [dedicated worktree at `../aragora-<branch>`] — never shared with another active agent
- **Branch tip at start (collision tripwire):** [`git rev-parse HEAD` recorded at staging]
- **Merge policy:** [user-merges (default — you never merge) | merge-commit-on-green (opt-in, Tier 0-2 only, never squash)]
- **Highest tier auto-settle allowed:** [Tier 2] — Tier 3-4 always hard-stop for human settlement
- **Final-response policy:** [allowed | disallowed until stop]
- **Batch completion rule:** A batch is complete only after the aragora gate passes (gate steps 1-7) and the closing commit+push lands.
- **Re-read rule:** Immediately after every commit and push, re-read this survival guide.

---

## Stop Gate

> Rewrite in place. The explicit answer to "may I stop now?"

- **Planned batches remaining:** [N]
- **Batches blocked on human settlement (Tier 3-4):** [list, or none]
- **Stop allowed right now:** [yes | no]
- **Why:** [one sentence]
- **Next required action:** [one sentence]

If batches remain and no real stop condition applies, `Stop allowed right now` is `no`. A Tier 3-4
batch awaiting human settlement does **not** end the run if other unblocked batches remain — move
to the next unblocked batch. Only checkpoint-and-wait when every remaining batch is blocked.

---

## Forbidden Stop Reasons

Not valid reasons to stop while unblocked work remains: a checkpoint time was reached; a commit/push
succeeded; CI is green; a receipt was written; the user is silent; you wrote a summary; the current
batch is done but later batches remain; "this is a lot for one turn" (the volume is the point);
"this feels like a natural place to check in" (there is no one to check in with). Update docs,
commit, push, re-read this file, continue.

A genuine Tier 3-4 human-settlement block **is** a valid pause for *that batch only*.

---

## Non-Negotiables

- **Never merge by default. Never approve a merge.** Tier 3-4 requires explicit human risk
  acceptance (`aragora/human-settlement`) before counting as landed.
- **Every batch produces a verified `DecisionReceipt`** (`aragora verify`). No receipt → not done.
- **Unresolved adversarial dissent blocks the batch.** Never silence a reviewer to clear the gate.
- **Never modify a test to make it pass.** Fix the code. Total test count never decreases.
- **Respect the operating contract's approval-required surfaces and auto-halts** (see gate doc).
- One run owns one branch + one worktree. Surprise tip move = collision → stop.
- No destructive git: `reset --hard`, `checkout .`, `clean -fd`, `push --force`, rebase on shared.
- Stage specific files; never `git add -A`. Scope ≤800 LOC delta per batch.
- Every closing commit carries `Co-authored-by: codex[bot]` (or `claude[bot]`).
- [Project-specific non-negotiable from the plan, e.g. "do not touch `review_queue.py`"]

---

## Launch Readiness

- [ ] Plan cleaned and saved to disk
- [ ] Survival guide updated from the current plan
- [ ] Learnings + execution log initialized with batch breakdown and preflight notes
- [ ] Dedicated worktree created; branch + checkout ownership confirmed; tip recorded
- [ ] Preflight run (`pre-commit run --all-files`, `mypy`, target `pytest` slice) and critical failures cleared
- [ ] `aragora --help` confirmed on PATH; review agents/API keys available (`aragora api-key list`)
- [ ] Each planned batch pre-classified by merge tier (0-4); Tier 3-4 flagged as human-settlement stops
- [ ] Run mode, return time, merge policy, and non-negotiables recorded above
- [ ] Stop Gate initialized with `Stop allowed right now: no`
- [ ] Launch prompt prepared for the next call

---

## Current Phase

> Rewrite in place. History belongs in the execution log.

**Status:** [Staging / Launch-ready / In progress / Awaiting human settlement / All batches complete / Blocked]
**Active batch:** [Batch N: Name — Tier X]
**What was just finished:** [one sentence incl. receipt path + settlement state]
**Single next action:** [one sentence]

---

## Next Exact Batch

**Batch:** [N: Name]
**Predicted merge tier:** [0-4] — if 3-4, this batch ends in a human-settlement stop, not auto-continue
**Scope:**
- [Task 1]
- [Task 2]
**Acceptance criteria:**
- [ ] [Criterion]
**Risk:** [one sentence]
**Rollback tag:** `elves/pre-batch-N` _(create before starting)_

---

## Acceptance Checks (per batch)

Run the full gate in `references/validation-gate-aragora.md`. Summary:

- [ ] Rollback tag created before the batch started
- [ ] Local truth green: `pre-commit run --all-files`, `mypy` (no new errors vs `.mypy-baseline`), `pytest` slice; test count not decreased
- [ ] Adversarial review run through aragora; quorum facts recorded (head SHA, families, independence, recommendation, dissent)
- [ ] No unresolved dissent
- [ ] `DecisionReceipt` produced and `aragora verify` passes
- [ ] Merge tier classified; Tier 0-2 settled autonomously / Tier 3-4 paused for human settlement
- [ ] Execution log updated (SHA, commands, results, receipt path, tier, settlement state)
- [ ] Survival guide Current Phase / Stop Gate / Next Exact Batch updated
- [ ] Closing commit (with `Co-authored-by:` trailer) + push done before any later work
- [ ] Survival guide re-read immediately after that push

---

## Tool Configuration (aragora ground truth)

```yaml
# --- Local truth ---
lint: pre-commit run --all-files
typecheck: mypy aragora        # compare against .mypy-baseline; no new errors
test: pytest                   # narrow to the relevant slice per batch
mypy-baseline: .mypy-baseline

# --- Adversarial review (replaces github-pr-comments) ---
review: aragora-debate
review-cmd: aragora ask "<adversarial review prompt>" --agents anthropic-api,openai-api,grok --context-file <diff> --format json
review-gate-workflow: aragora-review-gate.yml      # advisory
merge-quorum-workflow: aragora-merge-quorum.yml    # enforcing, required check on main

# --- Receipts ---
receipt-verify: aragora verify <receipt.json>
receipt-dir: .aragora/receipts

# --- Settlement / tiers ---
tier-policy-doc: docs/REVIEW_AUTHORITY_PRINCIPLES.md
human-settlement-signal: aragora/human-settlement   # commit status set by operator
approvals-dir: .approvals

# --- Notification ---
notification: pr-comment        # or slack-webhook via ELVES_SLACK_WEBHOOK
```

---

## Rollback and Safety Rules

1. Tag before every batch: `git tag elves/pre-batch-N && git push origin elves/pre-batch-N`.
2. Never force-push or rebase the working branch (invalidates rollback tags).
3. Never merge by default — not even fast-forward. Tier 3-4 never auto-settles.
4. On serious breakage, branch from the last good tag (`git checkout -b recovery/from-elves-pre-batch-N elves/pre-batch-N`), document in the execution log, stop. Leave the original branch intact.
5. Stage specific files; know what you commit.
6. Surprise branch-tip move = collision → stop, do not commit on top, surface to user.

---

## After Any Compaction

1. Read this file (doing it now).
2. Read Run Control + Stop Gate. Recover run mode, stop policy, checkpoint semantics, and which batches are blocked on human settlement.
3. Read `.elves-session.json` (current batch, receipt paths, test baseline, `continuation_guard`).
4. Read learnings, then the plan (compare hash), then the execution log (last completed batch + last receipt + last settlement state).
5. Skim `docs/AGENT_OPERATING_CONTRACT.md` auto-halts and `docs/REVIEW_AUTHORITY_PRINCIPLES.md` tiers.
6. If `continuation_guard.stop_allowed` is false, continue without re-deciding.
7. Identify the first unblocked incomplete batch and resume. Do not redo completed batches — a verified receipt in the log means done.

---

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
