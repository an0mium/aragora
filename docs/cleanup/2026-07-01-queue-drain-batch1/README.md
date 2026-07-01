# Queue-Drain Cleanup — Batch 1 Manifests (2026-07-01)

Lane B6 of autonomous run `close-the-loop-20260701`, executing **batch 1 only** of
`docs/plans/2026-06-30-queue-drain-diagnosis-and-cleanup-plan.md` (plan commit `325169c8ca`,
branch `codex/reconcile-settle-report`).

## Authorization

- **G1** (close other agents' classified-churn PRs, one-time scoped) and
  **G2** (delete orphaned/patch-equivalent branches) granted by operator (armand),
  recorded by scarmani at
  https://github.com/synaptent/aragora/issues/8762#issuecomment-4860561852
  (2026-07-01T22:33:08Z). Epic: #8762.

## State at execution time (differs from plan's Jun-30 snapshot)

- Open PRs: **46** (plan said 231 — prior lanes/settlement drained most).
- Remote branches: **942** total; **124** orphaned (no merge-base with `origin/main`;
  plan said 645 — most already removed by earlier reconcile-lane cleanup).

## Pre-step: content spot-check (10 sampled orphans)

See `partC_spotcheck.json`. All 10 samples are pre-history-rewrite branches
(tips 2026-03-18 → 2026-03-28, PR-era #1000-1500); diff vs current main is ~5,600-6,000
files / ~950k deletions, confirming the rewrite orphaning. Per-tip verification:

- `fix-1522` (prompt-engine timing fix): fix content present on `origin/main`
  (`prompt_engine_stream.py:289` float normalization) — harvested.
- `worktree-agent-a03f0ccb` (canonical assessment compiler): `aragora/nomic/canonical_assessment.py`
  and `aragora/cli/commands/assess.py` exist on main — harvested.
- `worktree-agent-a15c678d` (inbox/audit test fixes): `tests/server/handlers/test_audit_trail_async.py`
  exists on main — harvested.
- `worktree-agent-a89f9ea7` (auth API key endpoints): `tests/server/handlers/auth/test_api_keys.py`
  exists on main; `api_keys.py` has later history — harvested/superseded.
- `pr1111-fix-ci`: one-line CI shim for a March-era workflow — obsolete.
- `rebase-1331`, `rebase-1311`, `rebase-1308`, `fix-1337`, `fix-1315`: **VALUE-FLAGGED.**
  Tips are March-25 strategy/outreach memo drafts whose exact files are NOT on main
  (e.g. `docs/outreach/BUYER_ANALYST_FAQ.md`, `docs/strategy/COMPETITIVE_POSITIONING_2026_03.md`).
  Main carries a *consolidated* canonical set covering the same themes
  (`docs/strategy/POSITIONING_AND_MESSAGING.md`, `PROOF_AND_EVIDENCE.md`,
  `RECEIPTS_DISSENT_EVIDENCE_NARRATIVE_2026_03.md`, `PRECISION_AND_TERMS.md`,
  `docs/outreach/OBJECTIONS_AND_TRUST.md`, `DESIGN_PARTNER_*`), so the content domain was
  harvested by consolidation — but per the value-flag rule the **entire 44-branch March
  memo cluster** (`rebase-1[23]xx` / `fix-1[23]xx`, 22 unique memos duplicated across two
  prefixes) is EXCLUDED from batch-1 deletion and left for operator/batch-2 review.
  See `partC_flagged_memo_cluster_EXCLUDED.json`.

## Part B — patch-equivalent branch deletes (44)

`partB_patch_equivalent_branch_deletes.json`. Criteria: has merge-base with main;
`git cherry origin/main origin/<branch>` returns all `-` (proof embedded per branch);
verified NO open PR via `gh pr list --head` per branch; not in any exclusion pattern.
One candidate excluded: `codex/next-steps-proof-surface-primary-20260606` (open PR #8128).
Cap 45; 44 final. Recovery: `git branch <name> <tip_sha> && git push origin <name>`
(objects retained by GitHub ~90d; also recoverable from this repo's local refs).

## Part C — orphaned branch deletes (80 of 124)

`partC_orphaned_branch_deletes.json`. Criteria (all): empty `git merge-base origin/main origin/<branch>`;
no open PR (per-branch `gh pr list --head` recheck at manifest time); tip date < 2026-06-01
(actual range 2026-03-18 → 2026-04-09); not matching exclusions
(`vision-incubator/*`, `AGT|DIC|TCP`, `structex/p4a-*`, `claude/fusion-*`, `elves/*`,
`codex/reconcile-settle-report*`, run branches). 124 qualified; 44 excluded by value-flag
(memo cluster above); 80 deleted this batch (cap 100). Remaining for batch 2: the 44
flagged (pending review) — no other orphans remain.

## Part A — churn PR closes (4 of 46)

`partA_churn_pr_closes.json`. Deterministic filter (agent-lane substrate-repair title
pattern + CONFLICTING/DIRTY or stale draft + all exclusion lists + no an0mium commits in
7d) yielded only **4** confident candidates (cap 30): #8128, #8143, #8405, #8406.
#8488 considered and excluded (MERGEABLE/CLEAN, borderline staleness). Closes are fully
reversible (`gh pr reopen`); head branches retained.

## Execution order

Part B (provably safe) → Part C (manifest-recoverable) → Part A (reversible closes),
manifests committed and pushed before any destructive action. Batch 1 of each part only;
Part D (worktrees) NOT in this lane.
