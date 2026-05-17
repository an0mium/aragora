# Paused Codex Desktop core-writer remediation (2026-05-16)

> **Status**: planning-only; **do not unpause any writer** based on this doc
> without operator sign-off. Each proposed action is reversible; each is
> bound by SHA-256 to the receipt referenced below.

## Receipt binding

- **Receipt (committed)**: `docs/receipts/paused-writer-remediation-20260517T013316Z.json`
- **Receipt (working copy, gitignored)**: `.aragora/codex_insights/paused-writer-remediation-20260517T013316Z.json`
- **SHA-256**: `58c0d6d3ccee122a1d8c435b30868cb8ee8818c185cf7814fed7e3e3ac3745ee`
- **HMAC**: unsigned in this environment (`ARAGORA_CONTEXT_SIGNING_KEY`
  not set). Operator should re-emit with the signing key set if the
  unsigned form is unacceptable for archival.
- **Generated**: 2026-05-17T01:33:16 UTC
- **Schema**: `aragora-paused-writer-remediation/1.0`

## Context

`scripts/check_codex_desktop_automations.py` reports **4 paused codex
desktop core writers** out of 18 total automations (11 active, 4 paused
warning, 3 supporting). All 4 paused are role=`writer` — the bulk
producers of the autonomous fleet's output. While paused, the writer
side of the codex automation lane is offline.

This audit was produced by dogfooding the inspector shipped in #7240 +
the insights layer shipped in #7245 against the live `~/.codex/`
corpus. No `~/.codex/` state was modified; no AI provider keys were
consumed; no merge/label/mark-ready actions were taken.

## Per-writer diagnosis (from the receipt)

| Writer | Role | Cron (m of hour) | Paused at (UTC) | Silent for | Diagnosis class | Proposed action | Severity |
|---|---|---|---|---|---|---|---|
| `engineering-autopilot` | Primary Writer | `:05` | 2026-05-16T18:25:49 | ~7h | `operator_pause_during_session` | `resume_when_operator_session_ends` | low |
| `engineering-autopilot-2` | Repair Writer | `:20` | 2026-05-16T19:07:08 | ~6.5h | `operator_pause_during_session` | `resume_when_operator_session_ends` | low |
| `engineering-autopilot-3-2` | Improver Writer | `:50` | 2026-05-16T18:25:46 | ~7h | `operator_pause_during_session` | `resume_when_operator_session_ends` | low |
| `engineering-autopilot-3` | Branch Salvage Writer | `:35` | **2026-05-13T03:57:35** | **~3d 21h** | `long_pause` | `investigate_then_resume_or_retire` | medium |

### Cluster signal

- **3 of 4 writers paused within 41 minutes today**
  (`engineering-autopilot` and `engineering-autopilot-3-2` paused in the
  same second 18:25:46 → 18:25:49; `engineering-autopilot-2` followed
  at 19:07:08). This pattern is consistent with a deliberate operator
  bulk-pause during an active session, not an infrastructure failure.
- **Branch Salvage is the outlier** (paused 2026-05-13 03:57 UTC,
  ~3 days before the others). This suggests a different cause and
  needs separate investigation before resume.

### Why "operator pause during session" is the working hypothesis

- The 18:25 pause time overlaps the operator's session in which:
  - PR #7240 (codex desktop inspector) was opened and reviewed
  - PR #7245 (insights layer) was built and pushed
  - The rotation gate was lifted for three previously-leaked AI keys
  - Several other PRs were merged on `main` (#7244, #7247, #7248)
- Writers paused while the operator is actively merging/reviewing PRs
  prevent autonomous writers from creating conflicts with in-flight
  manual work, and prevent them from consuming AI keys before rotation
  is confirmed propagated.
- This is consistent with the explicit holds the operator placed earlier
  this session (AI-key consumption, merge/label/mark-ready mutation).

## Recommended bounded next action per writer

These are recommendations only. Each requires explicit operator sign-off
before any change to the automation.toml or `status` field.

### Primary Writer — `engineering-autopilot`

- **Diagnosis**: cluster pause during operator session.
- **Action**: resume by setting `status = "ACTIVE"` in
  `~/.codex/automations/engineering-autopilot/automation.toml` once the
  operator has signed off on PR #7240, PR #7245, and #7215, and once
  any AI-key rotation has propagated to launchd inherited env.
- **Verification after resume**: confirm one cron tick fires at `:05`
  past the next hour by checking for a new `source=exec` rollout in
  `~/.codex/sessions/YYYY/MM/DD/` with `cwd` under the worktree root
  the automation creates.
- **Rollback**: re-set `status = "PAUSED"`.

### Repair Writer — `engineering-autopilot-2`

- **Diagnosis**: cluster pause during operator session.
- **Action**: same as Primary; resume at the same time.
- **Verification**: cron tick at `:20` past the next hour.
- **Rollback**: same as Primary.

### Improver Writer — `engineering-autopilot-3-2`

- **Diagnosis**: cluster pause during operator session (paused in the
  same second as Primary).
- **Action**: same as Primary; resume at the same time.
- **Verification**: cron tick at `:50` past the next hour.
- **Rollback**: same as Primary.

### Branch Salvage Writer — `engineering-autopilot-3`

- **Diagnosis**: longer-term pause, separate from the operator cluster
  pattern. Could be:
  - Deliberate longer-term policy decision (e.g., branch salvage was
    superseded by an automated worktree maintainer)
  - Genuine infra issue (model gate, cwd accessibility, worktree
    runner health)
- **Action**: **investigate before resuming**. Specifically:
  1. Check git log on `automation.toml` for any commit message
     explaining the pause.
  2. Look at the writer's `memory.md` for self-reported failure modes.
  3. Confirm whether `scripts/codex_worktree_autopilot.py` is now
     covering the branch salvage use case (it has a cleanup mode that
     overlaps).
- **Action choice (post-investigation)**:
  - If superseded → leave paused with a note in `automation.toml` or
    add a comment-only commit explaining the retire decision.
  - If real issue → fix root cause then resume.
  - If no clear cause → resume with monitoring; re-pause if the next
    3 cron ticks fail.

## What this audit deliberately did NOT do

- Did not modify any `automation.toml`
- Did not unpause any writer
- Did not open new boss-ready / autonomous issues
- Did not interact with the #7209 lane, #7173, #7215, #4990, or BC-12
  soak
- Did not consume any AI provider key
- Did not install or modify any launchd job
- Did not write to anything outside `.aragora/codex_insights/` (the
  receipt) and `docs/plans/` (this file)

## Verification (operator-facing)

```bash
# Re-emit the receipt at any time to compare against the SHA below.
# Inputs are read-only; output is deterministic given the same inputs.
# (Re-running will produce a NEW timestamp; the per-writer diagnosis
# fields should remain identical until pause state changes.)

python3 scripts/check_codex_desktop_automations.py --json --summary-only \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print([a for a in d['core_writers'].values() if a['status']=='PAUSED'])"

# Sanity-check the receipt hash:
sha256sum .aragora/codex_insights/paused-writer-remediation-20260517T013316Z.json
# Compare against the JSON file's "sha256" field — should match the canonical content hash.

# Optional: re-run with signing for archival:
ARAGORA_CONTEXT_SIGNING_KEY=$(python3 -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())") \
  python3 -m aragora.cli.main codex insights digest --since 24h --emit-receipt
```

## Sequencing

- This doc-only PR is independent of #7240 and #7245 (the inspector +
  insights PRs whose tools were used to produce this audit).
- This PR is **draft** and stays draft until operator sign-off.
- No labels (`boss-ready`, `autonomous`, etc.) — this is a
  recommendation artifact, not a dispatch target.

## Holds respected

- `#7209` lane (untouchable)
- `#7173`, `#7215`, `#4990`, BC-12 soak (held)
- AI-provider-key consumption (held)
- merge / label / mark-ready / launchd install (held)
- No `~/.codex/` mutation
