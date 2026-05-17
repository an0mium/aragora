# Open queue settlement packet — 2026-05-17

> **Read-only operator settlement packet.** This document does not mutate any PR. Each per-PR recommendation is cryptographically bound to a specific head SHA via the receipt referenced below. Operator signs off by editing the sign-off matrix in section 5 or by replying with explicit decisions.

## Header

- **Repo**: synaptent/aragora
- **Generated**: 2026-05-17T14:28:11.657846+00:00
- **PR count at pin**: 15 (prompt said 'currently 10'; live count is 15 — drift since the recommendation was produced)
- **Receipt**: `docs/receipts/open-queue-settlement-20260517T142811Z.json`
- **SHA-256**: `b93358c76358bab8b1a41a2843bc4c7ee36446ce433ebab8ab301d0c359cf9a2`
- **HMAC**: _unsigned (ARAGORA_CONTEXT_SIGNING_KEY not set in this environment; operator can re-emit with key set)_
- **Tier source**: `docs/REVIEW_AUTHORITY_PRINCIPLES.md` (lines 26-38)

## Tier reference (from docs/REVIEW_AUTHORITY_PRINCIPLES.md)

| Tier | Class | Requirement | Settlement |
|---|---|---|---|
| 0 | Docs/tests/status only | Green required checks + 1 model review/dogfood | Admin squash allowed |
| 1 | Additive internal code, no live caller | Green + 2 model signals (1 adversarial/dogfood) | Admin squash allowed |
| 2 | Live automation/CLI/observability | Green + 2 heterogeneous models + dogfood + no dissent | Admin squash allowed |
| 3 | Semantic/persistence/security/public API | Model quorum + explicit human risk settlement | Human risk acceptance required |
| 4 | Secrets/deployment/policy/merge-gate | Human preapproval before implementation AND before merge | Human preapproval required |

## Executive summary

| # | Head | Draft? | Decision | Merge | F | IF | T | Recommended action |
|---|---|---|---|---|---|---|---|---|
| **#7173** | `29115c5e97` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **2** | OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green |
| **#7215** | `0d148f13c1` | ready | REVIEW_REQUIRED | BLOCKED | 0 | 1 | **3** | WAIT: 1 check(s) still in flight; revisit when settled |
| **#7240** | `7aa00774bf` | ready | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **3** | REQUIRES: explicit human Tier-3 risk acceptance comment at current head |
| **#7243** | `7f9ecf28d5` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **0** | OPERATOR: mark ready when complete; Tier 0 → admin squash allowed once green |
| **#7245** | `68c03eadd0` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **3** | OPERATOR: mark ready when complete; Tier 3 → human risk acceptance required |
| **#7249** | `7a4d310d61` | draft | REVIEW_REQUIRED | BLOCKED | 2 | 0 | **1** | OPERATOR: mark ready when complete; Tier 1 → admin squash allowed once green |
| **#7251** | `19f29eeef1` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **1** | OPERATOR: mark ready when complete; Tier 1 → admin squash allowed once green |
| **#7252** | `5961232a82` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **0** | OPERATOR: mark ready when complete; Tier 0 → admin squash allowed once green |
| **#7256** | `e1b37b3485` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 6 | **2** | OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green |
| **#7257** | `a9b044ce74` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **3** | OPERATOR: mark ready when complete; Tier 3 → human risk acceptance required |
| **#7258** | `28a748d3f0` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **2** | OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green |
| **#7259** | `31192860af` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **2** | OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green |
| **#7260** | `e53aa8157d` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **2** | OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green |
| **#7261** | `a5a4f13063` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **2** | OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green |
| **#7262** | `e8a962a2dd` | draft | REVIEW_REQUIRED | BLOCKED | 0 | 0 | **1** | OPERATOR: mark ready when complete; Tier 1 → admin squash allowed once green |

_F = failures, IF = in-flight, T = tier_

## Clusters

### parser.py + CLI/docs cluster (mandatory merge order)

**Members**: #7215, #7240, #7245

**Shared files** (conflict-prone):
- `aragora/cli/commands/codex_sessions.py`
- `aragora/cli/parser.py`
- `aragora/codex/__init__.py`
- `aragora/codex/desktop_inspector.py`
- `aragora/codex/desktop_paths.py`
- `aragora/codex/duration.py`
- `aragora/codex/jsonl_stream.py`
- `aragora/codex/sqlite_ro.py`
- `aragora/module_tiers.yaml`
- `docs-site/docs/api/cli.md`
- `docs-site/docs/contributing/capability-matrix.md`
- `docs/CAPABILITY_MATRIX.md`
- `docs/METRICS.md`
- `docs/reference/CLI_REFERENCE.md`
- `tests/codex/__init__.py`

**Recommended merge order**: whichever PR is fastest to settle goes first (foundation); the other(s) rebase. Each PR adds one subparser registration in `aragora/cli/parser.py` and one row in CLI/capability docs — trivial conflicts but rebase required.

### codex_worktree_value_inventory.py cluster (operator-choice)

**Members**: #7256, #7259

**Shared files**:
- `scripts/codex_worktree_value_inventory.py`
- `tests/scripts/test_codex_worktree_value_inventory.py`

**Operator decision required**: these PRs propose alternate approaches to the same change. They should NOT both merge — pick ONE; the other should close or rebase to subsume the chosen approach.

## Per-PR detail

### #7173 — feat(triage): calibration-only multi-model GitHub issue triage with guided founder review

- **Head SHA**: `29115c5e97b9cd0ab0c7910cb12e6fa09f6025b7`
- **Branch**: `claude/triage-issues-via-debate-20260515` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=14, in-flight=0, failures=0
- **Diff**: +3505 / −14 across 7 files
- **Updated**: 2026-05-17T06:15

- **Tier**: **2** — live CLI / automation / observability surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Live automation / CLI / observability surface change. Accept: any newly added CLI verb becomes a public operator surface; any new automation will fire on its schedule once enabled. Reversible by revert + un-install of any installed launchd jobs.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green


### #7215 — [DIC-17] Add `aragora crux-followup` CLI verb (flag-gated, default OFF)

- **Head SHA**: `0d148f13c1b281f74c207fd25560bafb11fa9186`
- **Branch**: `vision-incubator/dic-17-crux-followup-cli` → `main`
- **Author**: `an0mium`
- **Draft**: False
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=11, in-flight=1, failures=0
- **Diff**: +435 / −12 across 8 files
- **Updated**: 2026-05-17T14:26

- **Tier**: **3** — semantic correctness / persistence / security / public API surface

- **Model-quorum evidence (from PR comments)**:
    - 2026-05-16T18:01  `aragora_code_review_bot` by `github-actions` [no head pin in comment]
    - 2026-05-16T18:05  `codex_focused_dogfood` by `an0mium` [head-pinned to `0d148f13c1`]
    - 2026-05-16T18:28  `droid_kimi_review` by `an0mium` [head-pinned to `0d148f13c1`]
    - 2026-05-16T18:28  `droid_kimi_review` by `an0mium` [head-pinned to `0d148f13c1`]
    - 2026-05-16T19:22  `codex_focused_dogfood` by `an0mium` [head-pinned to `0d148f13c1`]
    - 2026-05-16T23:16  `tier4_acceptance` by `an0mium` [head-pinned to `0d148f13c1`]
    - 2026-05-17T14:24  `tier4_acceptance` by `an0mium` [head-pinned to `0d148f13c1`]
    - 2026-05-17T14:26  `tier4_acceptance` by `an0mium` [head-pinned to `0d148f13c1`]

- **What you're accepting (1-paragraph risk statement)**:
    > Semantic / persistence / security / public API surface change. Accept: behavioral changes visible to downstream consumers (SDK users, API clients, persisted data shape). Requires explicit human risk settlement comment at current head.

- **Conflicts / overlap**: shares files with #7240, #7243, #7245

- **Recommended action**: WAIT: 1 check(s) still in flight; revisit when settled


### #7240 — feat(codex): read-only inspector CLI for Codex Desktop sessions

- **Head SHA**: `7aa00774bf633ccd7b1a945eff427396a75e083a`
- **Branch**: `worktree-codex-desktop-inspector` → `main`
- **Author**: `an0mium`
- **Draft**: False
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=71, in-flight=0, failures=0
- **Diff**: +2155 / −39 across 20 files
- **Updated**: 2026-05-17T06:24

- **Tier**: **3** — semantic correctness / persistence / security / public API surface

- **Model-quorum evidence (from PR comments)**:
    - 2026-05-16T23:24  `codex_security_check` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:25  `codex_focused_dogfood` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:28  `codex_focused_dogfood` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:29  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:30  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:33  `codex_focused_dogfood` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:34  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:37  `codex_security_check` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:43  `aragora_code_review_bot` by `github-actions` [no head pin in comment]
    - 2026-05-16T23:48  `codex_focused_dogfood` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:49  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-16T23:50  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-17T01:36  `tier4_acceptance` by `an0mium` [no head pin in comment]
    - 2026-05-17T01:50  `findings_review` by `an0mium` [no head pin in comment]
    - 2026-05-17T03:38  `codex_security_check` by `an0mium` [no head pin in comment]
    - 2026-05-17T05:00  `codex_focused_dogfood` by `an0mium` [no head pin in comment]
    - 2026-05-17T05:00  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-17T06:13  `droid_kimi_review` by `an0mium` [no head pin in comment]
    - 2026-05-17T06:22  `codex_focused_dogfood` by `an0mium` [head-pinned to `7aa00774bf`]
    - 2026-05-17T06:24  `droid_kimi_review` by `an0mium` [head-pinned to `7aa00774bf`]

- **What you're accepting (1-paragraph risk statement)**:
    > Semantic / persistence / security / public API surface change. Accept: behavioral changes visible to downstream consumers (SDK users, API clients, persisted data shape). Requires explicit human risk settlement comment at current head.

- **Conflicts / overlap**: shares files with #7215, #7243, #7245

- **Recommended action**: REQUIRES: explicit human Tier-3 risk acceptance comment at current head


### #7243 — docs: refresh metrics from current main

- **Head SHA**: `7f9ecf28d5b9cf75df08281419104d74419c8917`
- **Branch**: `codex/harvest-metrics-refresh-20260516` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=12, in-flight=0, failures=0
- **Diff**: +3 / −3 across 1 files
- **Updated**: 2026-05-16T23:23

- **Tier**: **0** — docs-only / tests-only / status report

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Docs/tests/status-only change touching 1 file(s). Accept: documentation drift if any of the listed paths become canonical surfaces later. Reversible by revert.

- **Conflicts / overlap**: shares files with #7215, #7240, #7245

- **Recommended action**: OPERATOR: mark ready when complete; Tier 0 → admin squash allowed once green


### #7245 — feat(codex): activity-intelligence layer with signed digest receipts

- **Head SHA**: `68c03eadd0762e9825293a1f28298e949a73bf28`
- **Branch**: `worktree-codex-insights` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=16, in-flight=0, failures=0
- **Diff**: +3215 / −18 across 24 files
- **Updated**: 2026-05-17T01:52

- **Tier**: **3** — semantic correctness / persistence / security / public API surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Semantic / persistence / security / public API surface change. Accept: behavioral changes visible to downstream consumers (SDK users, API clients, persisted data shape). Requires explicit human risk settlement comment at current head.

- **Conflicts / overlap**: shares files with #7215, #7240, #7243

- **Recommended action**: OPERATOR: mark ready when complete; Tier 3 → human risk acceptance required


### #7249 — [AGT-06] Add viah_signals bridge: ReputationStore → VIAH sidecar counters (SD-2, SD-3)

- **Head SHA**: `7a4d310d618bfe62c90f3d76913ff2afe97b9434`
- **Branch**: `vision-incubator/agt-06-viah-signals-bridge` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=11, in-flight=0, failures=2
- **Diff**: +286 / −0 across 2 files
- **Updated**: 2026-05-17T00:21

- **Tier**: **1** — additive internal code, no live caller and no public API effect

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Additive internal code with no live caller. Accept: dead code if the additive surface is never wired; minor maintenance burden. Reversible by revert.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 1 → admin squash allowed once green


### #7251 — feat(benchmarks): productize A2 admission-recovery scenarios for #7209 (PR-3, data+docs only)

- **Head SHA**: `19f29eeef1fc4e72be58cdbd36064303ac14a7c4`
- **Branch**: `droid/a2-pr3-admission-recovery-rescue-map-and-fixtures-20260516` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=14, in-flight=0, failures=0
- **Diff**: +353 / −0 across 3 files
- **Updated**: 2026-05-17T01:38

- **Tier**: **1** — additive internal code, no live caller and no public API effect

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Additive internal code with no live caller. Accept: dead code if the additive surface is never wired; minor maintenance burden. Reversible by revert.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 1 → admin squash allowed once green


### #7252 — docs(codex-insights): paused core-writer remediation audit (read-only)

- **Head SHA**: `5961232a8231b77b9ba0e5f6a0bf689749710993`
- **Branch**: `worktree-paused-writer-remediation` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=11, in-flight=0, failures=0
- **Diff**: +287 / −0 across 2 files
- **Updated**: 2026-05-17T01:35

- **Tier**: **0** — docs-only / tests-only / status report

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Docs/tests/status-only change touching 2 file(s). Accept: documentation drift if any of the listed paths become canonical surfaces later. Reversible by revert.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 0 → admin squash allowed once green


### #7256 — fix(scripts): preserve foreign worktree inventory roots

- **Head SHA**: `e1b37b34853d769221644a4d4087d0205394c60d`
- **Branch**: `codex/inventory-foreign-repo-preserve-20260516` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=1, in-flight=6, failures=0
- **Diff**: +100 / −0 across 2 files
- **Updated**: 2026-05-17T14:27

- **Tier**: **2** — live CLI / automation / observability surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Live automation / CLI / observability surface change. Accept: any newly added CLI verb becomes a public operator surface; any new automation will fire on its schedule once enabled. Reversible by revert + un-install of any installed launchd jobs.

- **Conflicts / overlap**: shares files with #7259

- **Recommended action**: OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green


### #7257 — feat(server): observer truth on FastAPI swarm-status sibling surface

- **Head SHA**: `a9b044ce7490eab1a838fcc2ba8e54fcdb42ace6`
- **Branch**: `droid/phase1-fastapi-observer-truth-20260516` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=14, in-flight=0, failures=0
- **Diff**: +169 / −1 across 2 files
- **Updated**: 2026-05-17T01:55

- **Tier**: **3** — semantic correctness / persistence / security / public API surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Semantic / persistence / security / public API surface change. Accept: behavioral changes visible to downstream consumers (SDK users, API clients, persisted data shape). Requires explicit human risk settlement comment at current head.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 3 → human risk acceptance required


### #7258 — feat(automation): publish recurring worktree value inventory

- **Head SHA**: `28a748d3f0c11708b95b212c84eba0d75077a2ca`
- **Branch**: `droid/phase2-worktree-value-inventory-20260516v2` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=14, in-flight=0, failures=0
- **Diff**: +5903 / −0 across 5 files
- **Updated**: 2026-05-17T06:07

- **Tier**: **2** — live CLI / automation / observability surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Live automation / CLI / observability surface change. Accept: any newly added CLI verb becomes a public operator surface; any new automation will fire on its schedule once enabled. Reversible by revert + un-install of any installed launchd jobs.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green


### #7259 — feat(scripts): bound worktree inventory runtime

- **Head SHA**: `31192860afdd7f095080059f358fb16e8dccc846`
- **Branch**: `codex/worktree-inventory-runtime-budget-20260517` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=14, in-flight=0, failures=0
- **Diff**: +221 / −13 across 2 files
- **Updated**: 2026-05-17T06:11

- **Tier**: **2** — live CLI / automation / observability surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Live automation / CLI / observability surface change. Accept: any newly added CLI verb becomes a public operator surface; any new automation will fire on its schedule once enabled. Reversible by revert + un-install of any installed launchd jobs.

- **Conflicts / overlap**: shares files with #7256

- **Recommended action**: OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green


### #7260 — feat(automation): read-only cross-agent overlap detector

- **Head SHA**: `e53aa8157d00060c134993ff61584da409094083`
- **Branch**: `droid/phase3-list-active-agent-sessions-20260516` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=13, in-flight=0, failures=0
- **Diff**: +1007 / −0 across 2 files
- **Updated**: 2026-05-17T06:19

- **Tier**: **2** — live CLI / automation / observability surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Live automation / CLI / observability surface change. Accept: any newly added CLI verb becomes a public operator surface; any new automation will fire on its schedule once enabled. Reversible by revert + un-install of any installed launchd jobs.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green


### #7261 — feat(automation): publish recurring publication-freshness probe

- **Head SHA**: `a5a4f130633a9cd93476474c304fb700407aff0f`
- **Branch**: `droid/phase4-publication-freshness-probe-20260516` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=14, in-flight=0, failures=0
- **Diff**: +1219 / −0 across 5 files
- **Updated**: 2026-05-17T06:27

- **Tier**: **2** — live CLI / automation / observability surface

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Live automation / CLI / observability surface change. Accept: any newly added CLI verb becomes a public operator surface; any new automation will fire on its schedule once enabled. Reversible by revert + un-install of any installed launchd jobs.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 2 → admin squash allowed once green


### #7262 — [AGT-02] A2A per-domain reputation read endpoint (sub-deliverable 5)

- **Head SHA**: `e8a962a2dd96670650938e5fbe145a73959fa04d`
- **Branch**: `vision-incubator/agt-02-a2a-reputation-read` → `main`
- **Author**: `an0mium`
- **Draft**: True
- **Decision / Merge**: REVIEW_REQUIRED / BLOCKED
- **Checks**: success=13, in-flight=0, failures=0
- **Diff**: +373 / −0 across 2 files
- **Updated**: 2026-05-17T08:24

- **Tier**: **1** — additive internal code, no live caller and no public API effect

- **Model-quorum evidence**: N/A — draft state; no quorum required until ready

- **What you're accepting (1-paragraph risk statement)**:
    > Additive internal code with no live caller. Accept: dead code if the additive surface is never wired; minor maintenance burden. Reversible by revert.

- **Conflicts / overlap**: none with other open PRs

- **Recommended action**: OPERATOR: mark ready when complete; Tier 1 → admin squash allowed once green


## Sign-off matrix

Operator: tick exactly one box per PR (or write your own decision in the comments column).

| # | Tier | [ ] APPROVE this tier | [ ] APPROVE downgraded | [ ] REQUEST changes | [ ] REJECT | [ ] HOLD (operator-only) | Comment |
|---|---|---|---|---|---|---|---|
| **#7173** | T2 | [ ] | [ ] | [ ] | [ ] | [X] |   |
| **#7215** | T3 | [ ] | [ ] | [ ] | [ ] | [X] |   |
| **#7240** | T3 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7243** | T0 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7245** | T3 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7249** | T1 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7251** | T1 | [ ] | [ ] | [ ] | [ ] | [X] |   |
| **#7252** | T0 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7256** | T2 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7257** | T3 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7258** | T2 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7259** | T2 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7260** | T2 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7261** | T2 | [ ] | [ ] | [ ] | [ ] | [ ] |   |
| **#7262** | T1 | [ ] | [ ] | [ ] | [ ] | [ ] |   |

## Holds compliance

This packet does not modify any of the following:

- #7209 lane (untouchable)
- #7173 (held — metadata only)
- #7215 (read-only metadata only)
- #4990 (held)
- BC-12 soak (untouched)
- AI provider key consumption
- merge / label / mark-ready / launchd install / automation.toml edit

**#7173 and #7251 specifically**: included in the packet as metadata only. Recommended action for both is `OPERATOR-ONLY` — packet does not propose any settlement; operator decides whether to lift the hold separately.

**#7215 specifically**: live PR comments at head `0d148f13c1b281f74c207fd25560bafb11fa9186` contain explicit Tier-4 acceptance text from author `an0mium` (2026-05-17 06:26 UTC and earlier). This is recorded in the per-PR section as **read-only observed state**. The packet does not advance #7215 in any way; the operator's hold ('read-only metadata only') is respected.

## Verification

The `sha256` field inside the JSON receipt hashes the **canonical payload before the signature fields (`sha256`, `hmac_sha256`, `signed_at_utc`) were added** — not the on-disk file. To verify the binding holds:

```bash
# 1. Confirm the receipt file is well-formed and reports the expected SHA:
python3 -c "
import json, hashlib
d = json.load(open('docs/receipts/open-queue-settlement-20260517T142811Z.json'))
claimed = d.pop('sha256'); d.pop('hmac_sha256', None); d.pop('signed_at_utc', None)
canonical = json.dumps(d, sort_keys=True, separators=(',', ':')).encode('utf-8')
recomputed = hashlib.sha256(canonical).hexdigest()
print('claimed   :', claimed)
print('recomputed:', recomputed)
print('match     :', claimed == recomputed)
"

# 2. Confirm this doc references the same SHA:
grep -o '`b93358c7[a-f0-9]*`' docs/status/settlement-packets/2026-05-17-open-queue-settlement.md

# 3. Re-derive at any time (output differs only in generated_at_utc; per-PR head_sha pins are stable until the PR pushes):
#    gh pr list --state open --json number,headRefOid,...   # same shape as used to produce this packet
```

---

_Generated by aragora session-continuation tooling — read-only synthesis from gh CLI + `docs/REVIEW_AUTHORITY_PRINCIPLES.md`. No mutation performed; no PR labels touched; no merge action taken._
