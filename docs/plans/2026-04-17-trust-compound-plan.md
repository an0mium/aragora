# Trust-Compound Plan (TCP)

> **Created:** 2026-04-17
> **Source:** [docs/audits/2026-04-17-codex-claude-joint-assessment.md](../audits/2026-04-17-codex-claude-joint-assessment.md)
> **Goal:** make the repo legible without subtracting any preserved subsystem
> **Activation gate:** TCP-1..TCP-7 are planning truth, not new boss-ready scope. Codex autopilot may pick up TCP issues only when the proof-first reconciler permits it.

## Sequencing rationale

Each step was chosen for **leverage per week** and for compatibility with the substrate-first activation gate. Earlier steps unlock or ratify later ones; none requires a code freeze. Long-running steps run in the background while normal work continues.

| Order | Code | Step | Owner | Estimated effort | Long-running execution surface |
|-------|------|------|-------|------------------|--------------------------------|
| 1 | TCP-1 | Canonical-metrics claim manifest + CI verifier | Codex autopilot or Claude Code | 1 day | `docs/status/claims/canonical_metrics.yaml` + `aragora.epistemic.ClaimVerifier` runs in CI on every push and on a daily schedule |
| 2 | TCP-2 | Packaging identity consolidation | Founder, 30 min | 30 min | One-shot |
| 3 | TCP-3 | Hotspot-file split proposals (1 file/month) | Droid deep audit | 1 month per file | Droid produces a split proposal per file; PRs land monthly |
| 4 | TCP-4 | Wire / showcase / shelve classification | Claude Code | 1 week | Extends `docs/STRANDED_FEATURES_AUDIT.md`; verified periodically by a tagged-state check in CI |
| 5 | TCP-5 | Generated-artifacts CI migration | Codex autopilot | 1-2 weeks | Build pipeline change; reduces conflict storms ongoing |
| 6 | TCP-6 | README top-of-fold rewrite to lead with bounded wedge | Founder, 30 min | 30 min | One-shot |
| 7 | TCP-7 | One narrow public AGT-* activation (e.g. `aragora.ai/cruxes`) | Claude Code session | 1 week + ongoing | Static page rebuilt nightly from the CruxSet store; the "click-able proof" output |

## Step contracts

### TCP-1 — Canonical-metrics executable claim manifest

**Why first:** highest signal-to-noise. Mismatch between `CANONICAL_GOALS.md` (claims 42 KM adapters) and `python3 scripts/doc_stats.py` (reports 0) is exactly the failure class the project's own DIC-13 / DIC-14 epistemic-CI doctrine is designed to prevent. Dogfooding the doctrine compounds trust faster than any other single move.

**Deliverables:**

- `docs/status/claims/canonical_metrics.yaml` — claim manifest in the existing schema, one claim per headline number in `CANONICAL_GOALS.md`
- `scripts/check_canonical_metrics.py` — extracts the claimed value from `CANONICAL_GOALS.md` and reconciles against the live count; prints structured JSON; exit 0 if consistent, non-zero on drift
- `tests/integration/test_canonical_metrics_manifest.py` — verifies the manifest schema-validates, the script runs end-to-end, and at least one claim's verifier exits 0 against the current repo
- CI workflow integration: run the script on every push to `main` and on a daily schedule

**Acceptance criteria:**

- All claims in the manifest schema-validate
- The script either confirms each claim or fails clearly with which value drifted
- The 42-vs-0 KM-adapters drift is either resolved or explicitly downgraded to a known-broken claim with a follow-up issue

**Long-running execution:** the manifest and verifier are intended to run in CI indefinitely. New canonical claims (test counts, agent counts, version) are added by appending to the YAML.

### TCP-2 — Packaging identity consolidation

**Why second:** highest first-impression ROI. 30 minutes of founder time fixes a top-of-page trust signal.

**Decisions to make (one-shot):**

- Is `aragora-debate` (the pip package) a separate product or a slice of the main install? Document both shape and relationship in README.
- Fix `pyproject.toml` git URL: currently `github.com/an0mium/aragora`, repo lives at `github.com/synaptent/aragora`
- README install table: keep the three install surfaces if all three are intended product offerings; merge or remove if not

**Acceptance criteria:**

- `pyproject.toml` `[project.urls]` matches reality
- README install table has one canonical "start here" path with the others labeled as alternatives
- One-paragraph README section explaining the relationship between `aragora` and `aragora-debate`

### TCP-3 — Hotspot-file split proposals (rolling)

**Why third:** behavior-preserving legibility win. Splitting `boss_loop.py` into a subpackage doesn't remove a single line — it just makes 4.5k LOC findable in 4-5 ~1k-line files with clear module docstrings.

**Order of attack:**

1. `aragora/swarm/boss_loop.py` (~4.5k) — most read by external reviewers since it's the autonomy entry point
2. `aragora/nomic/dev_coordination.py` (~5.3k) — largest absolute size
3. `aragora/server/handlers/playground.py` (~4.2k)
4. `aragora/pipeline/idea_to_execution.py` (~4.0k)
5. `aragora/cli/parser.py` (~3.6k) — easiest, most mechanical (subparser registration sprawl)

**Acceptance criteria for each split PR:**

- Public API unchanged (top-level imports re-export everything)
- All existing tests pass without modification
- New file structure has one clear concern per file with docstrings
- New subpackage has a `README.md` (or top-level docstring) explaining the layout

**Long-running execution:** Droid deep-audit packet (already drafted earlier in the joint-assessment session) produces a split proposal per file. PRs land at most one per month to avoid review fatigue.

### TCP-4 — Wire / showcase / shelve classification

**Why:** converts visible sprawl into honest cataloguing. Nothing is deleted; the inspector reads the labels.

**Deliverables:**

- Extend `docs/STRANDED_FEATURES_AUDIT.md` with a three-column table for every subpackage under `aragora/`:
  - `(A) wire` — currently on the critical path to the bounded-execution wedge or the AGT-* track
  - `(B) showcase` — demonstrates the platform end-to-end (Inbox Trust Wedge, AGT pipeline dry-run, etc.)
  - `(C) shelved-revisit` — preserved on disk, maintenance suspended, revival criteria specified
- One sentence per `(C)` row stating what condition would revive maintenance
- A claim manifest entry under TCP-1 verifying the classification is non-empty and covers all subpackages

**Preservation guarantee:** the table CANNOT be a deletion plan. Anything classified `(C)` stays on disk.

### TCP-5 — Generated artifacts as build outputs

**Why:** OpenAPI/SDK conflict storms are a CI workflow problem, not a structural one. Two paths:

- **Lazy-rebuild path:** keep the generated files committed but use `.gitattributes merge=theirs` / `merge=ours` to auto-resolve; rebuild on every push to `main`
- **Build-output path:** move generated files to `build/` (gitignored), publish per-tag as build artifacts; consumers fetch by tag

**Acceptance criteria:**

- No more PRs blocked by `docs/api/openapi*.{json,yaml}` merge conflicts
- A clear documented procedure for downstream consumers to fetch the latest schema

**Long-running execution:** once the workflow lands, ongoing maintenance is zero.

### TCP-6 — README top-of-fold rewrite

**Why:** the README already says the right thing ("govern AI-assisted work with receipts, review, and truthful gates"). Lead harder with that. Move the agent-civilization vision to a clearly-labeled `## Roadmap` section so the bounded-execution wedge gets the first 30 seconds of attention.

**Acceptance criteria:**

- First two screens of the README sell the bounded wedge (current product)
- A `## Roadmap` or `## Forward bet` section explicitly captures the AGT-* / agent-civilization direction
- Install table has one canonical path

### TCP-7 — One narrow public AGT-* activation

**Why:** the unlock that flips Codex's verdict from "might be impressive in 60 days" to "clearly doing something useful today." Pick one AGT-* flag, activate narrowly, publish output.

**Suggested wedge:** emit CruxSets for one debate category (e.g. "ship/hold decisions"), publish them as a static page at `aragora.ai/cruxes` rebuilt nightly from the CruxSet store. Shows the system identifying load-bearing disagreements on real questions.

**Acceptance criteria:**

- One AGT-* flag activated in production via env var (not just merged behind default-off)
- Output is publicly readable at a stable URL
- At least 10 real CruxSets published before the activation is rated successful

**Long-running execution:** the page rebuilds nightly; the CruxSet count grows over time; failure modes (zero new cruxes for >7 days, signature verification failure, etc.) flag follow-up issues via DIC-17.

## Cross-cutting rules

- **Nothing in `aragora/` is deleted by this plan.** `verticals/`, `extensions/`, `genesis/`, `modes/`, marketplace/, etc. are all preserved (see audit doc preservation list).
- **No code freeze.** TCP-3 (hotspot splits) runs in parallel with all other work, one file per month.
- **Substrate-first gate is unchanged.** TCP issues do NOT carry `boss-ready` until the proof-first reconciler permits the upper-layer tranche, in line with how DIC-13..22 and AGT-01..06 are governed.
- **TCP-1 is the keystone.** Once the canonical-metrics manifest runs in CI, every subsequent step (TCP-3 splits, TCP-4 classification) gets verified by it — drift becomes a build failure rather than a slow trust erosion.

## Dispatch matrix (who does what)

| Step | Best-fit dispatcher | Spec source |
|------|---------------------|-------------|
| TCP-1 | Codex autopilot (issue spec is well-bounded) or Claude Code | This doc + the existing `docs/status/claims/proof_first_claims.yaml` template |
| TCP-2 | Founder, no delegation | This doc |
| TCP-3 | Droid deep audit (`droid exec -m claude-opus-4-7 -r high --auto low`); split proposals only, no PRs | Droid dispatch packet drafted in joint-assessment session; extend with split mandate |
| TCP-4 | Claude Code session | This doc + `docs/STRANDED_FEATURES_AUDIT.md` |
| TCP-5 | Codex autopilot | This doc |
| TCP-6 | Founder, no delegation | This doc |
| TCP-7 | Claude Code (continuation of AGT-* track familiarity) | This doc + `docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md` |

## Tracking

- GitHub epic: see follow-up issue created from this plan
- Per-step issues: TCP-1..TCP-7 each get one issue, all linked to the epic
- Status: `in_progress` once a PR exists; `closed` when acceptance criteria pass

## When to revisit this plan

- After TCP-1 lands and runs in CI for 7 days — confirm the manifest catches at least one drift event; tighten thresholds
- After TCP-3 ships its first hotspot split — measure subjective "could a new contributor understand this in an hour" against the pre-split baseline
- After TCP-7 publishes 10 CruxSets — re-read the joint assessment and update the verdict
- Otherwise: every quarter, regenerate the joint assessment from a cold-read perspective and diff against this plan
