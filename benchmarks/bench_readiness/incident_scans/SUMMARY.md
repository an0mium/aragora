# Trufflehog Scan Summary — 2026-04-17

Scope: every public repo under `synaptent/*` and `an0mium/*` GitHub orgs, plus
the local `aragora` clone. Scanned with `trufflehog --only-verified`.

## Repos scanned

20 public repos: A-revolution-detector, Ancient-vase-profiles, aragora,
Chemdata, Cone-shell-evolution, contour-extraction-for-R, Evolution-Revolutions,
Global-Pot-Project, Global-Pot-Project-Sebastian-, Iznik, Modern-vases,
Pace-of-Modern-Culture, RingRift, Selection_simulations, shapeClassification,
Shell-collection, The-Lexicon-of-Iznik-Ornament, Women-in-the-BMJ,
Women-in-the-BMJ-public, plus the `aragora` worktree.

## Findings

| Repo | Verified secrets | Severity | Action taken |
|------|------------------|----------|--------------|
| `aragora` | 0 (only test fixtures) | none | 9 false-positive secret-scanning alerts dismissed via `gh api --method PATCH … resolution=used_in_tests` |
| `RingRift` | Multiple verified secrets (raw scan retained off-repo) | **high** | Owner notified separately; raw scan kept out of git history to avoid re-exposure |
| All other 18 repos | 0 | none | — |

## Why the raw scan output is not committed

Per-repo trufflehog raw output (`raw/RingRift.json`, etc.) contains the literal
secret values that were exposed. Committing them to this repo — which is itself
public — would re-expose those secrets to the same secret-scanning crawlers and
attackers that found them the first time. Only this redacted summary is kept.

The raw scan files are stored locally at:

    ~/Development/aragora/benchmarks/bench_readiness/incident_scans/raw/

(gitignored). They are needed for active remediation of `RingRift` and can be
regenerated with the command in `incident_2026-04-07_high-gravity.md`.

## Cross-reference

- `incident_2026-04-07_high-gravity.md` — the parent security incident write-up
  (HIGH-GRAVITY harvester repo, leaked Anthropic key)
- `docs/audits/2026-04-17-codex-claude-joint-assessment.md` — the audit that
  drove the scan
- `.gitleaks.toml` — local pre-commit policy that flagged this scan output as
  containing secrets (working as intended)
