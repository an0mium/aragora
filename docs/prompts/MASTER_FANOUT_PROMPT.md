# Master Fan-Out Prompt (v14)

Canonical idempotent prompt for autonomous parallel agents working the Aragora
proof-first loop. Drop this prompt unchanged into any agent (Claude Code, Codex
CLI, Codex Desktop, Factory Droid) running against `/Users/armand/Development/aragora`
or any worktree thereof. Multiple sessions can run concurrently — coordination
is by file-locked lane registry and append-only journal, not by shared context.

**Why this exists:** Earlier versions of this prompt inflated to >1M tokens by
inlining historical chat logs, prior receipts, and command output. The agent
already has observers that can discover that history programmatically; this
prompt relies entirely on Phase 0 discovery and the on-disk journal.

**Version history:** v3..v13 prompt-bug rows are in `docs/status/AGENT_FANOUT_JOURNAL.md`.
v14 incorporates audited corrections — see the closing changelog.

---

## SESSION BINDING

```
SESSION_ID="<family>-$(uuidgen | cut -c1-8 | tr a-z A-Z)"   # e.g. droid-C08FA2B1
AGENT_FAMILY="<claude | codex | droid | other>"
```

If running inside Claude Code, Codex CLI, Codex Desktop, or Factory Droid, the
canonical env vars (`CLAUDE_SESSION_ID`, `CODEX_THREAD_ID`, `CODEX_ROLLOUT_PATH`,
`FACTORY_DROID_SESSION`) are picked up automatically by
`scripts/claim_active_agent_lane.py` (Phase E primitive, PR #7328 on main).

## OPERATING LIMITS

- Operate strictly under [`docs/AGENT_OPERATING_CONTRACT.md`](../AGENT_OPERATING_CONTRACT.md)
  and [`docs/governance/OPERATOR_DELEGATION_POLICY.md`](../governance/OPERATOR_DELEGATION_POLICY.md).
- **Protected files (never edit):** `CLAUDE.md`, `AGENTS.md`, `aragora/__init__.py`,
  `.env`, `.envrc`, `scripts/nomic_loop.py`, anything under `docs/governance/`
  unless the contract authorizes it for your lane.
- **Secrets:** All API keys must resolve via AWS Secrets Manager
  (`aragora/config/secrets.py`). Never export raw keys; never modify `.env`.
- **Branching:** Never push or force-push to `main`. Use `--force-with-lease` on
  your own branch only.
- **Held PRs (canonical, from OPERATOR_DELEGATION_POLICY.md):**
  `#7173, #7215, #7240, #7243, #7245, #7249, #7252, #4990`, the BC-12 soak,
  and the `#7209` lane. Never push to, label, mark-ready, or merge a held PR.
- **Trusted authors for Bucket A auto-merge:** `@an0mium` only.
- **Net LOC ceiling:** ≤1500 for Bucket A; larger diffs route to Bucket C.

---

## PHASE 0 — PROGRAMMATIC LIVE TRUTH (read-only, ≤5 min)

```bash
# 1. Fetch main and read the live log
git fetch origin main
git log --oneline -20 origin/main

# 2. Read the cross-session memory file
cat docs/status/AGENT_FANOUT_JOURNAL.md | tail -120

# 3. Canonical observability stack
python3 scripts/list_active_agent_sessions.py --json --max-pr-fetch 50
python3 scripts/agent_bridge.py --json operator-snapshot --summary-only
python3 scripts/agent_bridge.py --json health
python3 scripts/check_canonical_metrics.py --all --write-receipt
python3 scripts/triage_open_prs.py --json   # NOTE: no --pr flag exists; parse output

# 4. Lane-registry ground truth (Phase 0 read raw to dodge CLI-vs-file mismatch)
python3 -c "import json; r=json.load(open('.aragora/agent-bridge/lanes.json')); \
  active=[l for l in r if l.get('status') in ('active','running','pending','queued','claimed')]; \
  print(json.dumps(active, indent=2))"

# 5. Lane hygiene — surface stuck-active lanes before claiming
python3 scripts/sweep_stale_lane_claims.py            # dry-run by default

# 6. Status receipts on disk
ls docs/status/generated/canonical_metrics/latest.json \
   docs/status/generated/publication_freshness_probe \
   docs/status/generated/worktree_value_inventory
```

Write read-only findings to `docs/status/SESSION_BRIEF_${SESSION_ID}.md` with
a 5-bullet live state summary and a list of active sibling lanes.

> **Known surfacing quirk:** `agent_bridge.py lanes --json` can undercount vs
> the raw `.aragora/agent-bridge/lanes.json` file (PR #7308 P28-A journal row).
> Trust the raw file in Phase 0.

> **Held-branch hygiene:** if `git diff origin/main..HEAD` is empty but you
> see uncommitted edits in the founder root, you may be observing a dirty
> founder-root checkout. Run `python3 scripts/observer_truth_probe.py --json`
> to assert the observer is at clean `origin/main` before drawing
> conclusions.

---

## PHASE 1 — DERIVE CANONICAL PHASE LIST & EXCLUSIONS

Sort outstanding work deterministically by priority:

1. **Strategic Rollout Milestones** — [`docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md`](../roadmap/OPERATOR_DELEGATION_ROLLOUT.md)
2. **Proof-Loop Stability** — B0 truth refreshes, publication freshness probe,
   canonical-metrics `warn`/`fail` rows
3. **Canonical-Metrics Drift** — fields from `docs/status/generated/canonical_metrics/latest.json`
4. **Disk/Worktree Hygiene** — `docs/status/generated/worktree_value_inventory/latest.json`,
   purging safe-removable worktrees, clearing orphan remote branches

**SKIP rules (must hold simultaneously to claim):**

- Skip any task matching a lane marked `active` / `running` / `pending` /
  `queued` / `claimed` in the registry — token-set normalize to catch
  near-duplicate lane_ids (`P28-worktree-inventory-refresh` vs
  `P28-refresh-worktree-value-inventory` are the SAME work — see journal
  2026-05-18T05:18:00Z prompt-bug).
- Skip any task whose phase-id and primary noun match a journal row marked
  `shipped` / `finish-existing` within the last 12 hours.
- Skip held PRs (see Operating Limits).

**Trust Gate (CS-01..03):** Any task that updates a documentation metric
must narrow or exactly equal what `check_canonical_metrics.py` measures.
Never inflate a claim beyond measured proof. If a claim is undercounted by a
buggy counter (e.g. v8 P24 case: `^\s*def test_` regex missed `async def
test_`), apply honesty rule H2 — fix the counter, don't lower the claim.

---

## PHASE 2 — ATOMIC CLAIM LOCK

Select the highest-priority non-skipped task ID. Use canonical lane-id
schema `P<num>-<verb>-<noun>` to keep collision detection effective.

```bash
PHASE_ID="P<num>-<verb>-<noun>"
BRANCH="${AGENT_FAMILY}/${PHASE_ID}-$(date +%Y%m%d)"

python3 scripts/claim_active_agent_lane.py \
  --lane-id "$PHASE_ID" \
  --owner-session "$SESSION_ID" \
  --branch "$BRANCH" \
  --status active \
  --json

# Verify the claim was recorded (raw file, not CLI surfacing)
python3 scripts/agent_bridge.py --json health
```

If health reports a collision or token-overlap, change your status to
`released` and try the next phase:

```bash
python3 scripts/claim_active_agent_lane.py \
  --lane-id "$PHASE_ID" --owner-session "$SESSION_ID" --status released --json
```

Once secured, spin up an isolated managed worktree:

```bash
WT=$(python3 scripts/codex_worktree_autopilot.py ensure \
  --agent "$AGENT_FAMILY" --base main --force-new --print-path | tail -1)
cd "$WT"
git checkout -b "$BRANCH"
```

---

## PHASE 3 — WORKTREE ISOLATED EXECUTION (≤45 min)

- All source modifications occur inside your worktree path.
- Every commit message must include `[lane: $PHASE_ID]`.
- **Test floors:**
  - Utility scripts: ≥5 tests
  - Batchers / multi-input scanners: ≥8 tests
  - State mutations (lane registry, journal, queue): ≥12 tests
  - Data-only / doc-only refreshes: 0 tests if no script code is touched
- Before committing, all of the following must be green:

```bash
ruff check .
ruff format --check .
bash scripts/automation_pr_preflight.sh origin/main HEAD
bash scripts/preflight_mypy.sh --diff-base origin/main      # P71 helper, opt-in but recommended
```

- Push to origin and open a **draft** PR.
- Triage your PR by parsing the full triage output:

```bash
python3 scripts/triage_open_prs.py --json | \
  python3 -c "import json,sys; \
    prs=json.load(sys.stdin); \
    me=next((p for p in prs.get('prs',prs) if p.get('number')==<YOUR_PR>), None); \
    print(json.dumps(me, indent=2))"
```

- Flip from draft to ready-for-review **only if** Bucket A (auto-mergeable).
  Bucket C stays draft — operator or the Stage-3 batcher will pick it up.
- **Stuck budget:** If blocked >30 min, write a descriptive fallback note in
  your receipt and proceed to Phase 4 with outcome `deferred` or `no-work`.

> **Worktree inventory caveat:** if your phase touches the worktree value
> inventory, pair `--smart-merge-detection` with `--include-pr-state` (P58
> #7330 on main); `--skip-gh` alone misclassifies open-PR branches as
> harvest candidates.

> **Amend safety:** never `git commit --amend` a pushed commit; use
> `scripts/guard_amend_pushed.sh` if you want a hard guard (P72 #7338).

---

## PHASE 4 — RECEIPT, JOURNAL APPEND, RELEASE

1. **Write a receipt** to the main checkout path (operator-visible):

   ```
   docs/status/${PHASE_ID}_RECEIPT_${SESSION_ID}.md
   ```

   Sections: goal, what shipped or deferred, triage bucket classification,
   6-observer dogfood quorum (your own results from Phase 0 observers +
   2 sibling agents' fresh runs if available).

2. **Append a single-line record** to the shared ledger:

   ```
   TIMESTAMP_UTC | SESSION_ID | AGENT_FAMILY | PHASE_ID | PR# | OUTCOME
   ```

   Outcomes: `shipped | finish-existing | deferred | no-work | conflict`.

3. **If you found a structural defect** in this prompt, append an extra row:

   ```
   TIMESTAMP_UTC | SESSION_ID | prompt-bug: <description> | workaround: <fix>
   ```

4. **Synchronize and clear the lock:**

   ```bash
   git pull --rebase origin main
   git add docs/status/*
   git commit -m "docs(status): session receipt [lane: $PHASE_ID]"
   git push origin main

   python3 scripts/claim_active_agent_lane.py \
     --lane-id "$PHASE_ID" --owner-session "$SESSION_ID" --status completed --json
   ```

5. **Final exit line** (one line, exact format):

   ```
   Phase <PHASE_ID> claimed; PR #<n> <opened|rebased|flipped-ready|none>; receipt written; lane released.
   ```

---

## ANTI-PATTERNS

Things every prior version of this prompt got wrong at least once:

- ❌ `triage_open_prs.py --pr <number>` — no such flag exists; parse JSON.
- ❌ Treating `agent_bridge.py lanes --json` count as ground truth — read the raw file.
- ❌ Same work, different lane-id — `P28-worktree-inventory-refresh` vs
  `P28-refresh-worktree-value-inventory` slipped past the collision detector.
- ❌ Claiming a lane with `--status active` and never transitioning to
  `released`/`completed` — leaves the registry full of zombies and breaks
  subsequent strict claim rules.
- ❌ Inflating a documentation metric to match an undercounted claim —
  always honesty-rule H2 the counter first.
- ❌ Inlining historical command output into the prompt context — Phase 0
  re-discovers it cheaper.
- ❌ Marking docs-only PRs ready-for-review without preflight; CI lanes care.
- ❌ Running `--smart-merge-detection` with `--skip-gh` alone (misses open PRs).

## v14 CHANGELOG vs the human-distilled draft

Corrections applied after auditing scripts and journal:

| Fix | Reason |
|-----|--------|
| Hold list completed: `#7240, #7243, #7252, BC-12 soak, #7209 lane` added | Per `docs/governance/OPERATOR_DELEGATION_POLICY.md` |
| `triage_open_prs.py --pr` replaced with JSON-parse pattern | Flag doesn't exist |
| `agent_bridge.py operator-snapshot --json --summary-only` order kept; `--json` is global, must precede subcommand | Live `--help` output |
| `sweep_stale_lane_claims.py` (P63) added to Phase 0 | Detects stuck-active lanes |
| `observer_truth_probe.py` (Q09) referenced as dirty-observer guard | Per NEXT_STEPS_CANONICAL Observer rule |
| `preflight_mypy.sh` (P71) added to commit preflight | Production mypy gate |
| Canonical lane-id schema `P<num>-<verb>-<noun>` codified | Prevents token-overlap collisions |
| Worktree inventory caveat documents `--include-pr-state` (P58) | `--skip-gh` alone misclassifies |
| Receipt path convention `docs/status/${PHASE_ID}_RECEIPT_${SESSION_ID}.md` documented | Already in journal practice |
| Raw-file read codified for lane truth (Phase 0 step 4) | CLI vs file mismatch from P28-A |

## Steering templates (for the operator)

See [`docs/prompts/STEERING_TEMPLATES.md`](STEERING_TEMPLATES.md) if/when it
ships. The two canonical templates:

- **Substrate Freeze (Tier-4 redirection):** halts new orchestration verbs;
  redirects the next phase to running an existing benchmark/vertical and
  producing one external-proof artifact.
- **Tier-4 Human Settlement Authorization:** authorizes a specific
  `admin_squash_merge` on a named head SHA after structural delta review.

These are operator-side; agents must never self-author either.
