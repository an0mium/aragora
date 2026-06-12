# Codebase Health Program — 2026-06-12

**Epic:** #8257 | **Basis:** [`docs/audits/2026-06-12-codebase-health-audit.md`](../../audits/2026-06-12-codebase-health-audit.md) (three measured passes at head `a5cf5fc70b`) | **Baseline:** [`docs/audits/2026-06-12-codebase-health-baseline.json`](../../audits/2026-06-12-codebase-health-baseline.json)

## Mission

Point the repo's own proven governance machinery (shrink-only baselines, ratchets, drift
checks, evidence loops) at the three surfaces it was never aimed at: macro-architecture
layering, gate rigor, and the outsider's first hour (packaging/docs/hygiene). Every batch
lands a falsifiable definition of done, most expressible as a baseline-JSON number moving
the right direction.

## Standing constraints (floor — cannot be weakened by any wave)

- elves-aragora v0.2.1 gate per batch: local truth → adversarial debate (≥2 model families)
  → verified DecisionReceipt → tier settlement. Lanes finish synchronously; every lane
  registers in the run's lane registry with `next_action`, `last_heartbeat_at`.
- Check the operator-steering mailbox (`scripts/read_operator_steering.py`) before
  mutating lane state; write outcome receipts when messages are read.
- Tier rules per `docs/AGENT_OPERATING_CONTRACT.md`: Tier 0-2 settle autonomously with
  countable evidence; Tier 3 prepares evidence and parks for operator settlement;
  workflow-file edits are approval-required (prepare branches, never apply).
- `bash scripts/automation_pr_preflight.sh origin/main HEAD` before publishing any branch.
- Never `--amend` a pushed commit (`scripts/guard_amend_pushed.sh`).

## Right-of-way (claim discipline)

- **The ODR program leads.** On any file conflict with
  `docs/superpowers/plans/2026-06-11-odr-program.md` waves (epic #8223), HEALTH yields.
  Gauntlet/receipt surfaces are ODR territory; HEALTH-6 (#8263) explicitly coordinates
  with ODR-3 (#8226) on verifier/wedge packaging.
- Before claiming any HEALTH issue: read this plan, the ODR plan, the lane registry, and
  open PRs. A missing lane file is not proof of no owner (lesson: #8246 duplicated #8239
  inside a create-to-merge window). Prefer issues with no fresher signal anywhere.
- **No `boss-ready` on HEALTH issues until the audit/plan PR merges** (the DoDs reference
  the baseline it carries).

## Wave structure

### Wave 1 — launch immediately (file-disjoint, low risk)
| Lane | Issue | Scope | Tier |
|---|---|---|---|
| H1 | #8258 | Repo-root hygiene: relocate 11 screenshots/.docx + research md | 1 |
| H2 | #8259 | Shrink-only layering baseline test + first shrink (`debate -> server.metrics` edge via `observability/`) | 2 |
| H3 | #8260 | `/metrics` fail-closed + `session.revoke` ownership check (evidence: `.aragora/prepared-evidence/pr-8163.json`) | 3 — prepare + park |

### Wave 2 — after any Wave 1 batch merges
| Lane | Issue | Scope | Tier |
|---|---|---|---|
| H4 | #8261 | mypy `check_untyped_defs` + ruff `BLE001`/`G004` per-package grow-only enrollment (seed 3 clean packages) | 2 |
| H5 | #8262 | Docs canonicalization: 1 quickstart + 1 self-hosting guide, version under Version Alignment | 2 |

### Wave 3 — gated (conditions explicit)
| Lane | Issue | Trigger | Tier |
|---|---|---|---|
| H6 | #8263 | Aragora debate run, decision receipt attached to the issue (one-name vs wedge-split); coordinate ODR-3 | 3 |
| H7 | #8264 | H2 merged (so movement is measured): telemetry shim retirement, streaming home, memory via gateway — 3 separate PRs | 2-3 |
| H8 | #8265 | Workflow inventory + canceller-masking fix; workflow edits prepared and parked for operator | 3-4 — prepare only |

### Continuous (every coordinator check-in, no lane needed)
- Re-measure the baseline JSON commands; flag any `shrink` metric that grew as a
  regression with the offending commit range.
- Queue drain per the standing automation contract is unaffected by this program.

## Breakers

- MAIN RED >30 min → bisect/fix first; this program halts.
- Two same-wave CI failures for distinct reasons → halt the wave, checkpoint.
- Any HEALTH change that breaks a gauntlet/receipt surface → revert first, then
  re-approach with ODR coordination.
- No batch closed or parked in 2h → checkpoint + postmortem + re-read this plan.

## Exit metrics (falsifiable)

1. Every `shrink`/`grow` metric in the baseline JSON is at-or-better than baseline,
   enforced by CI where harnesses exist (H2, H4).
2. Clean-machine first hour passes end to end: documented `pip install` → documented CLI
   exists → zero-key demo produces a receipt.
3. `/metrics` and `session.revoke` fail closed (tests assert both).
4. Workflow count ≤73 with no governance-relevant cancelled-terminal runs on ready PRs
   (one reconciler cycle reruns them).

All waves complete or parked → final report, update `docs/FOCUS.md`, close #8257.
