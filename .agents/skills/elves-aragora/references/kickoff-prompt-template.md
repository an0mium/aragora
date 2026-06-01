# Kickoff Prompt Template — aragora

> Two-call handoff: **stage**, then **launch** in a fresh call. Most "the elves stopped" failures
> come from cramming a giant plan and the launch instructions into one message. The plan lives on
> disk; the launch prompt stays short and behavior-heavy.

## Step 1: Stage

```
Stage this elves-aragora run. Do not start implementing batches in this call.

**Plan:** [docs/plans/<plan>.md]
**Branch:** [feat/<name>]   (create in a dedicated worktree: git worktree add -b <branch> ../aragora-<branch>)
**Survival guide:** [docs/elves/survival-guide.md]   (generate from references/survival-guide-template.md)
**Execution log:** [docs/elves/execution-log.md]      (generate from references/execution-log-template.md)
**Learnings:** [docs/elves/learnings.md]

Your job in this call:
- Tighten the plan so it survives compaction without this conversation
- Generate the survival guide, execution log, and learnings file
- **Pre-classify every batch by merge tier (0-4)** per docs/REVIEW_AUTHORITY_PRINCIPLES.md. Flag Tier 3 batches as human-settlement stops AFTER the gate, and **Tier 4 (and approval-required surfaces) as human PRE-APPROVAL stops BEFORE any implementation** — record both in the survival guide
- Set `## Run Control` explicitly: run mode, checkpoint semantics, merge policy (default: you never admin-merge / never bypass a gate; Tier 0-2 settle by marking the draft ready for the protected squash + `aragora-merge-quorum` check), highest auto-settle tier, workspace ownership (dedicated worktree), branch tip tripwire
- Create/switch to the dedicated worktree + branch; confirm no other agent owns this checkout
- Run preflight: pre-commit run --all-files, mypy aragora (vs .mypy-baseline), the relevant pytest slice; confirm `aragora --help`, `gh auth status`, and review agents/keys (`aragora api-key list`)
- Log warnings/blockers
- Prepare a short launch prompt for the next call

Non-negotiables:
- Never admin-merge; never bypass a gate. Tier 3 stops for human settlement (aragora/human-settlement); Tier 4 stops for human pre-approval BEFORE implementation and again before merge
- The batch gate is the repo's real proof-first loop: draft PR → `gh pr checks <pr> --required` green → `aragora review-pr <pr> --reviewer claude` + `--reviewer codex` evidence (`evidence-lint`) → verified DecisionReceipt (`aragora verify`) → `review-queue merge-packet` + `scripts/settle_one_pr.py` clean (blockers == ['PR is draft']) → tier settlement. Local tests alone never close a batch.
- Every batch produces a verified DecisionReceipt and clears adversarial dissent
- Respect the operating contract's approval-required surfaces and auto-halts
- [project-specific hard rule]

Stop condition for this call:
- Stop only once the run is launch-ready and you've handed me the launch prompt.
```

## Step 2: Launch (fresh call, keep it short)

```
The run is staged. Start now.
Read docs/elves/survival-guide.md first, then `.elves-session.json` if it exists, then
docs/elves/learnings.md, then the plan, then docs/elves/execution-log.md, then skim
docs/AGENT_OPERATING_CONTRACT.md and docs/REVIEW_AUTHORITY_PRINCIPLES.md.
I am going offline until [WHEN]. By [WHEN] I want [CHECKPOINT]. This is a [delivery checkpoint / hard stop].
Run the full aragora validation gate on every batch (references/validation-gate-aragora.md):
local truth → draft PR → required checks green (`gh pr checks <pr> --required`) → independent
model-quorum evidence at the exact head (`aragora review-pr <pr> --reviewer claude` and
`--reviewer codex`, lint with `review-queue evidence-lint`) → verified DecisionReceipt
(`aragora verify`) → `review-queue merge-packet` + `scripts/settle_one_pr.py` clean (only blocker
is 'PR is draft') → tier settlement. Tier 0-2: mark the draft ready for the protected squash —
never `--admin`, never a bypass. Tier 3: HARD STOP, queue for my settlement, move to the next
unblocked batch. Tier 4 / approval-required surfaces: HARD STOP for my PRE-APPROVAL *before* you
implement anything — do not write the change first.
Do not admin-merge or bypass any gate. Do not silence reviewer dissent. Do not modify tests to make them pass.
Every completed batch ends with commit + push, then re-read the survival guide.
Do not wait for me to acknowledge checkpoints or commits. If unblocked work remains, keep going.
Honor every operating-contract auto-halt (MAIN RED, approval-required surfaces, 800 LOC cap).
Keep going until the plan is done, every remaining batch is blocked on my settlement, I stop you,
or you hit a true blocker.
```

## Tips

- **Stage and launch in separate calls.** If a single message has a big plan plus "run now", treat it as a staging request and push back: "I'll stage this and wait for your launch command."
- **Pre-classify tiers at staging.** Knowing which batches are Tier 3-4 up front lets the run reorder so unblocked (Tier 0-2) work runs unattended while Tier 3-4 packets wait for you.
- **`ra:` to ride along.** Prefix a mid-run message with `ra:` (or `ride-along:`) to add context without stopping the run.
- **Final readiness review is mandatory.** Before the final handoff: fresh `git diff <default-branch>...HEAD` review, read every review comment, re-run sensible tests, confirm all receipts verify and all Tier 3-4 settlements are resolved or clearly queued. Then deliver the report and stop for the user to merge (or land a Tier 0-2 merge commit only if merge-on-green was set).
