# VibeProxy inference allowlist execution log

## Run Digest

- **Last updated:** 2026-07-20 17:54 America/Chicago
- **Current phase:** Final validation and exact-head rereview
- **Active batch:** Batch 1: Inventory and enforcement
- **Last completed batch:** none
- **Next exact batch:** none; final review/cleanup only
- **Active PR:** #9439
- **Docs promoted this run:** none
- **Latest Elves Report:** pending final exact-head review

## Session Setup: 2026-07-20 17:06 America/Chicago

**Phase:** Launch started
**Plan:** `docs/plans/2026-07-20-vibeproxy-inference-allowlist.md`
**Survival guide:** `docs/elves/vibeproxy-inference-allowlist-9409/survival-guide.md`
**Learnings:** `docs/elves/vibeproxy-inference-allowlist-9409/learnings.md`
**Execution log:** `docs/elves/vibeproxy-inference-allowlist-9409/execution-log.md`
**Branch:** `codex/vibeproxy-inference-allowlist-9409`
**PR:** #9439
**Run mode:** finite | **Hard stop:** 2026-07-20 20:15 America/Chicago
**Checkpoint semantics:** hard stop | **Actual stop conditions:** final readiness, true blocker, or hard deadline
**Active compute at launch:** none
**Continuation guard:** stop_allowed=no | remaining_batches=1 | checkpoint_is_stop=yes | next_required_action=Implement Batch 1, validate, and review

**Approved order:**

1. Current PR: inference-site inventory/static allowlist.
2. Separate follow-on PR: exact-match OpenAI Chat/Responses routing.
3. Separate follow-on PR: endpoint authentication/pinning before broad rollout.

**Live grounding:**

- `origin/main`: `9247f44918534e9ac29d37b50be53e4b978b41c8`
- PR #9431: merged externally; old lane marked complete.
- Issue #9409: open, no comments, no overlapping open VibeProxy PR.
- Mailbox: branch lease required before push; chartered removals and per-object ownership remain binding.
- New lane: `codex-vibeproxy-inference-allowlist-9409-20260720`, owner `elves-vibeproxy-inference-allowlist-9409-20260720`.
- Lease: `225e738e-cea`, issue `#9409`, four-hour TTL.

**Preflight:**

- Git remote / `gh` auth: PASS
- Dedicated worktree / collision tripwire: PASS
- Mailbox / owner / overlap checks: PASS
- Validation gate dry run: PASS; 97 tests passed across the existing VibeProxy transport, sanitized diagnostic, and AI-audit surfaces.
- Elves install doctor: WARN, optional v2.10.3 update available; installed v1.12.0 retained for this run.
- Automation preflight: PASS against `origin/main...HEAD`.
- Launch readiness: READY; the user's latest message is the fresh execution approval.

## Batch 1 Contract: 2026-07-20 17:12 America/Chicago

**Behaviors:**

- Discover production OpenAI/Anthropic inference call sites deterministically.
- Require every discovered site to have an exact allowlist entry classified `proxy-eligible` or `direct-only`.
- Reject stale inventory entries, missing rationales for direct-only entries, and every use of port 8317.

**Build on:**

- Existing standard-library checker and JSON output conventions under `scripts/`.
- Existing audit patterns under `tests/audit/` and `tests/scripts/`.
- Existing `ModelTransportPolicy` authority under `aragora/agents/transports/` without modifying runtime behavior.

**Acceptance criteria:**

- [ ] Current inventory exact-match is green.
- [ ] Synthetic unclassified and stale entries fail deterministically.
- [ ] Protected direct-only categories and port 8317 are tested.
- [ ] Focused gates, automation preflight, independent review, and final cumulative readiness are clean.

**Blast radius:**

- New tooling/manifest/tests/docs only; no existing runtime surface is planned to change.
- `.gitignore` receives one additive Elves ephemeral-artifact entry.
- Risk: low to medium because false negatives would permit uncontrolled routing growth and false positives would create noisy maintenance.

**Pre-implementation survey:**

- Complete. Existing checker/audit patterns and the current inference surface were mapped before implementation.
- The generic `elves/pre-batch-1` tag already existed in the shared repository; this run uses the collision-free `elves/vibeproxy-inference-allowlist-9409/pre-batch-1` tag instead.

## Batch 1 Implementation and Validation: 2026-07-20 17:54 America/Chicago

**Implementation:**

- Added `scripts/check_inference_site_allowlist.py`, a standard-library AST/JSON checker using stable path/symbol/provider/protocol anchors.
- Added the generated, line-reviewable `scripts/inference_site_allowlist.json`: 139 sites, 143 detections, 138 direct-only classifications, and one deliberate proxy-eligible `scripts/consult_claude.py::_run_vibeproxy` entry.
- Added 23 checker tests covering exact inventory, missing/stale/count drift, protected paths, aliases, method-receiver filtering, unreadable syntax, templates, manifest validation, and literal/zero-padded port 8317.
- No runtime routing, inference request, evidence, governance, workflow, API, auth/pinning, settlement, or merge change was made.

**Validation:**

- `uv run pytest -q tests/scripts/test_check_inference_site_allowlist.py tests/scripts/test_check_vibeproxy.py tests/agents/transports/test_vibeproxy.py tests/audit/test_ai_systems_audit.py`: PASS, 120 tests, 12 pre-existing deprecation warnings, 32.66s.
- `uv run ruff check ...`: PASS.
- `uv run mypy scripts/check_inference_site_allowlist.py`: PASS.
- `uv run python scripts/check_inference_site_allowlist.py --json`: PASS, 139 sites / 143 detections / 4,762 scanned files / zero scan errors.
- `uv run python scripts/check_charter_compliance.py --range origin/main...HEAD --format json`: PASS.
- `bash scripts/automation_pr_preflight.sh origin/main HEAD`: PASS.
- Non-generated implementation plus test delta: 798 lines; generated manifest is 146 stable review lines.

**Independent review round 1 (`344ab114a6`):**

- P1 whole-file port exemption: fixed at `d3ca833cd3`; only the exact central prohibition declaration is exempt.
- P2 constructor aliases: fixed and covered for OpenAI and Anthropic.
- P2 unreadable/invalid sources: fixed fail-closed with `scan_errors`.
- P2 suffix false positives: fixed by requiring a client receiver; unrelated `store.responses.create()` is rejected by test.
- P2 zero-padded port and one-line manifest: fixed; `:0*8317` is rejected and the generated manifest has one site per line.
- P2 scope/stale Elves state: product/test delta is within 800; ephemeral run artifacts and the unrelated broad `.gitignore` entry are scheduled for final cleanup.

**Next action:** commit and push these review fixes, then obtain a fresh independent review on the new exact head before cleanup/readiness.
