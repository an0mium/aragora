# Session Findings — claude-C1CE7926 (extended soak)

**Window:** 2026-05-21 03:08Z .. 03:25Z (~17 min wall clock; original plan had ~3-5h budget)
**Plan:** `~/.claude/plans/aragora-substrate-soak-2026-05-20.md` (9 planned phases; ran 11 actual)
**Companion files:**
- `SESSION_BRIEF_claude-C1CE7926.md` (Phase 0 posture)
- `SESSION_RECEIPT_claude-C1CE7926.md` (9-phase wrap)
- `BUCKET_C_REPORT_claude-C1CE7926.md` (PR triage)
- `docs/compliance/EU_AI_ACT_ARTIFACT_claude-C1CE7926/` (PR #7391, demo-data bundle)
- `docs/compliance/EU_AI_ACT_REAL_claude-C1CE7926/` (PR #7392, real-receipt bundle)

## TL;DR — product findings the operator should act on

1. **EU AI Act gap map (highest priority)** — PR #7392's real-receipt bundle reports Article 12 FAIL, Article 15 FAIL, Articles 9/13/14 PARTIAL with 4 binding recommendations. **This is the canonical Aug-2-2026-deadline punch list.** Each recommendation maps to an existing aragora subsystem; see PR #7392 RECEIPT.md.

2. **Receipt verify is broken end-to-end.** `aragora demo --receipt <path>` produces receipts that fail `aragora receipt verify`:
   - **Real-mode receipt:** `[FAIL] hash mismatch: stored=da72228e1425, expected=8daa1747e181` + missing `timestamp` field
   - **Synthetic bundle:** missing `artifact_hash`, `verdict`, `timestamp`, `confidence`
   - Both report INVALID (1/3 checks pass)
   - Root cause appears to be producer/consumer canonicalization mismatch between `aragora demo` and `aragora receipt verify`. **Not** related to the EU AI Act bundle's own integrity hash (which works fine).
   - **Recommended fix order:** write a `tests/cli/test_receipt_roundtrip.py` that produces a receipt via `aragora demo --receipt` and verifies it via `aragora receipt verify`. Drive the gap from there.

3. **HIPAA compliance check is keyword-only.** `aragora compliance check --frameworks hipaa` on synthetic PHI (`SSN 123-45-6789, DOB 1980-03-15, ICD-10: E11.9, insurance member ID BC-9876543`) flagged only the keyword "diagnosis" (1 critical). SSNs, ICD codes, insurance IDs, DOBs, etc. were NOT detected.
   - This **violates** [`feedback_use_real_intelligence.md`](../../.claude/projects/-Users-armand-Development-aragora/memory/feedback_use_real_intelligence.md): "frontier LLMs for all classification/routing/disambiguation."
   - Current rules in `aragora/compliance/` are regex/keyword-based. They should be either (a) augmented with frontier-LLM classification, or (b) at minimum extended with the standard PHI regex set (SSN, DOB, ICD-10, NPI, MRN, etc.).
   - This is one of the four EU AI Act recommendations from #7392 ("Improve robustness score"), now with a concrete repro.

4. **`aragora doctor` flags "LLM Provider: NO API KEY SET" as a failure.** This is incorrect for the canonical local posture (AWS Secrets Manager via `aragora/config/secrets.py` per [`feedback_no_api_keys_in_local_env.md`](../../.claude/projects/-Users-armand-Development-aragora/memory/feedback_no_api_keys_in_local_env.md)). Should be either downgraded to "optional" / "info" with hint about AWS Secrets Manager, or doctor should attempt to load via the canonical secrets path before flagging.

5. **Production is healthy.** `https://api.aragora.ai/health` returns `{"status": "ok"}` in 143ms.

6. **Canonical-metrics is fully passing.** 10 pass / 0 warn / 0 fail after this session's P91 refresh. Prior `warn` on `test_definitions.count` resolved organically — recent test-coverage commits (#7377-#7380) pushed the live counter back over the 80% floor.

## Phases summary (11 lanes)

| Lane | Outcome | Output |
|------|---------|--------|
| P90-master-fanout-prompt-v14-pr | **shipped** | PR #7390 |
| P91-canonical-metrics-receipt-refresh | **shipped** | receipt 9P/1W → 10P/0W |
| P92-canonical-test-count-drift | no-work | resolved by P91 |
| P93-proof-surface-refresh | no-work | B0+TW03 fresh (1.96d) |
| P94-worktree-inventory-refresh-with-pr-state | deferred | codex worktrees active |
| P95-lane-registry-stale-sweep | no-work | 0 stale of 139 |
| P96-triage-scan-bucket-a-flips | **shipped** | BUCKET_C_REPORT.md |
| P97-eu-ai-act-compliance-artifact | **shipped** | PR #7391 (demo data) |
| P98-real-receipt-and-bundle | **shipped** | PR #7392 (real data, **headline**) |
| P99-session-wrap | shipped | SESSION_RECEIPT.md |
| P100-extended-soak-wrap | (this) | SESSION_FINDINGS.md |

**Shipped artifacts:** 3 PRs (#7390, #7391, #7392) + 4 status docs on main + 1 status doc on this branch.

## What this soak proved about the substrate

The fan-out infrastructure (lane claims, journal append, canonical metrics, observers, codex_worktree_autopilot, automation_pr_preflight) all worked end-to-end without modification. Zero prompt-bug rows added to the journal. v14 of the master prompt held up under real execution.

The substrate-freeze posture worked: of 11 lanes claimed, 0 introduced new tooling. All output came from existing CLIs (`aragora demo`, `aragora compliance eu-ai-act generate`, `aragora compliance check`, `aragora receipt verify`, `aragora doctor`), existing scripts (`check_canonical_metrics`, `sweep_stale_lane_claims`, `triage_open_prs`), and existing docs surfaces.

## Concurrent agents (not interfered with)

Throughout the soak, the following codex worktrees were active in parallel. None claimed lanes that overlapped with mine; I deferred Phase 4 (proof-surface) and Phase 5 (worktree-inventory) explicitly to avoid stepping on them:
- codex/b0-truth-refresh-after-corpus-merges-20260520
- codex/harvest-adc-follow-on-deepening-20260520 (+ refresh-r2)
- codex/salvage-publish-rescue-dry-run-output-20260520
- codex/salvage-github-connectivity-tokens-20260521
- codex/salvage-reconcile-open-pr-unknown-state-20260520
- codex/stage2-subprocess-cwd-hardening-primary-20260520
- codex/worktree-inventory-size-none-20260518

The codex `salvage-github-connectivity-tokens-20260521` lane is the natural owner of the gh GraphQL 504 issues I hit during Phase 7.

The `harvest-adc-follow-on-deepening` lanes are the natural owners of settling ADC v0.2/v0.3/v0.4 (#7358/#7360/#7361) — explicitly OUT of this soak's scope per operator-tier review requirement.

## Follow-up work the operator should batch

| Priority | Item | Why |
|----------|------|-----|
| P0 | Review + ready-flip PR #7392 (real-receipt EU AI Act artifact) | Headline external proof; closes the demo-vs-real loop |
| P0 | Open issue for "receipt verify is broken end-to-end" (finding #2 above) | Affects every receipt produced by demo mode |
| P1 | Open issue for "HIPAA check is keyword-only" (finding #3) | Maps directly to EU AI Act Article 15 recommendation |
| P1 | Review + ready-flip PR #7390 (master prompt v14) | Operationalizes the v14 prompt for next sessions |
| P2 | Review + ready-flip PR #7391 (demo-data EU AI Act artifact) | Confirms the bundle CLI smoke-tests cleanly |
| P2 | Rebase wave for claude #7336/#7348/#7349/#7351/#7358/#7360/#7361 | Unblocks ADC v0.2..v0.4 chain |
| P3 | Fix `aragora doctor` API-key reporting per canonical secrets pattern | UX, not correctness |

## Final state

| Surface | State |
|---------|-------|
| Open PRs | 50 (was 47; +3 from this session) |
| canonical-metrics | 10 pass / 0 warn / 0 fail |
| Lane registry | 140 rows (was 139; +1 net from this session's release lifecycle) |
| Production health | green (143ms /health) |
| Session-owned active lanes | 1 (P100, will release after committing this) |
| Worktrees under .worktrees/ | unchanged (Phase 5 deferred) |
| Journal rows | +13 by this session |
