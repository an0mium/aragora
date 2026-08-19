# AWS Retirement Migration Execution Log

## Run Digest

- **Last updated:** 2026-08-19 09:16 America/Chicago
- **Current phase:** Staging
- **Active batch:** none
- **Last completed batch:** none
- **Next exact batch:** Batch 1: Provider-neutral secret custody
- **Active PR:** pending
- **Docs promoted:** none
- **Elves Report:** not generated

## Session Setup: 2026-08-19 09:10 America/Chicago

**Phase:** staging in progress

**Branch:** `codex/aws-retirement-migration-9391`

**Worktree:** `$HOME/.codex/worktrees/aws-retirement-migration-9391-20260819/aragora`

**Base/collision tripwire:** `6955ab420ed959dcce9cece4120b298453adc9c3`

**Lease:** `803c32bd-760`, work ID `issue:9391`, owner
`codex-elves-aws-retirement-9391-20260819`

**Authority:** operator approved the provider-neutral migration direction and ordered execution.
The run records Tier-4 implementation preapproval for this bounded migration plan; it does not
infer exact-head settlement or merge authority.

**Batch breakdown:**

1. Provider-neutral secret custody.
2. Non-AWS ODR signing.
3. Canary deployment pack.
4. External canary verifier.
5. Live canary and evidence.
6. AWS retirement follow-up packet, without workflow edits.

**Live grounding:**

- Issue #9391 remains open and still describes AWS recovery as the next action.
- Current main is `6955ab420ed959dcce9cece4120b298453adc9c3`.
- Five runners labeled `aragora` are online.
- No relevant open PR was found from the initial title/branch scan.
- A stale `contingency/hetzner-prod-pack` branch has two July 17 worktrees with uncommitted ODR
  signing experiments, no process, lane, lease, or PR. Preserved untouched.
- The fresh branch had no steering message before claim.

**Architecture survey:**

- `aragora/config/secrets.py` is the shared AWS-first secret abstraction and strict production
  policy surface.
- `aragora/gauntlet/odr_signing.py` directly loads a dedicated AWS secret.
- `deploy/self-hosted/docker-compose.yml` is the reusable container baseline.
- `deploy/helm/aragora/values-supabase.yaml` proves Supabase is already standard external
  PostgreSQL, not an application-specific backend requirement.
- `scripts/verify_websocket.sh` and `scripts/verify_receipt.py` provide existing verification
  primitives to compose rather than replace.

**Preflight:**

- Git remote, push dry-run, and `gh` authentication: PASS.
- Dedicated worktree and branch ownership: PASS.
- Lease: PASS.
- Focused tests after matching CI-side-loaded packages: 125 passed.
- Targeted Ruff: PASS.
- Targeted mypy: PASS.
- Existing self-hosted Compose config: PASS with expected missing-password warning.
- Repository-wide Ruff: WARN, ambient failure in
  `.github/workflows/contract_drift_trusted_launcher.py`; protected and outside scope.
- Docker CLI installed but daemon unavailable: WARN.
- Supabase CLI authenticated: PASS.
- Hetzner and Cloudflare auth: unavailable; live canary Batch 5 gated.
- No paid or long-running resource started.

**Decision notes:**

- Do not salvage or clean stale contingency worktrees. They are user-owned prior art.
- Keep the first PR under the 800-line discipline and exclude workflow/governance cleanup.
- Treat the current approval as bounded Tier-4 implementation preapproval only. Require separate
  exact-head human settlement before merge.
- Do not create paid Hetzner infrastructure without a named existing target or explicit budget.

**Next:** commit and push staging artifacts, open the draft PR, reframe #9391 as the migration
ledger, record the PR number and plan hash, then stop at the mandatory fresh-launch boundary.
