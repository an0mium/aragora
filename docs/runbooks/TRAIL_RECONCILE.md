# Trail Reconcile — Tamper-Evident Trail Alarm (TET T3)

Spec: `docs/specs/TAMPER_EVIDENT_TRAIL.md` (Component 3). Check:
`trail_reconcile` in `scripts/fleet_sentinel.py`.

## What a trail breach means

The sentinel diffs **what happened** (the external witness — today the
interim GitHub REST events feed or a local replica via
`--trail-witness-replica`; the S3 audit-log stream once operator phase T0 is
done) against **what was intended** (the anchored intent chain,
`.aragora/trail/intent-chain.jsonl`, read and hash-verified via
`aragora.trail.intent_chain`).

**The contract: every mutating action needs a pre-anchored intent.** An
intent anchored *after* the action (beyond a small clock-skew grace, default
2 min) does not count — post-hoc anchoring cannot legitimize anything. A
mutating witness event with no matching intent (repo + ref/sha + compatible
actor class + anchored within the match window, default 15 min before) is a
breach: someone — possibly an adversary holding our own credentials — acted
off the record.

## Severity tiers

| Tier | Event classes | Why |
|---|---|---|
| **critical** | token/deploy-key/app/secret changes, member/role changes, workflow changes; intent-chain tamper (`verify_chain` broken) | Credential and member events have **no legitimate agent intent class** — only a scarmani-anchored intent matches. This is the May-incident shape. |
| **high** | push, merge, branch deletion without anchored intent | Normal agent verbs gone off the record. |
| **unknown** (exit 2) | witness unreadable, intent-chain module absent/unpopulated, witness silent beyond 4× cadence | Silence is never success; unknown outranks breach in the exit contract. |
| note (still ok) | witness silence beyond cadence (default 6 h) but under 4× | Blind-period note: visible but quiet. |
| note (still ok) | `coverage limited` — interim events-API witness in use | The GitHub REST events witness **cannot see token/deploy-key/member admin events** (the May-incident class). Until operator phase T0 (S3 audit stream) is live, an `ok` only covers push/merge/branch-deletion traffic; every report says so. |

## Operator response to a critical breach

1. **Rotate credentials now** — GitHub tokens, deploy keys, App keys; assume
   the laptop's credentials are hostile (the spec's threat model).
2. **Freeze settlement** — no Tier-2+ merges/settles until reconciled; pause
   the publisher/auto-evidence loops if the breach touches workflows.
3. **Investigate via the S3 witness** — the Object-Lock bucket is the
   append-only ground truth; pull the window around the breach event and
   compare with the intent chain (`seq`/`intent_id` in the sentinel detail).
   The check's detail enumerates every unmatched event and the chain-verify
   result so nothing unaccounted-for is hidden.
4. **Record the disposition** as an anchored intent (scarmani-anchored for
   credential events) so the next cycle reconciles clean — or keep the alarm
   firing on purpose until it is truly resolved.

False alarms on normal ~30-merge agent traffic mean the matching rules are
wrong: tune `--trail-match-window-mins` / actor classes, or honestly
downgrade to advisory (spec exit metric). The permanent regression guards are
`test_breach_replay_may_incident_unauthorized_credential` and
`test_replay_normal_day_thirty_merges_zero_false_alarms` in
`tests/scripts/test_fleet_sentinel.py`.
