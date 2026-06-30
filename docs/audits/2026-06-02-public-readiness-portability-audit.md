# Public-Readiness & Portability Audit + Remediation Spec

**Date:** 2026-06-02
**Audited revision:** `origin/main` @ `7875dab643`
**Author:** autonomous audit (Droid), read-only `git grep` over tracked files
**Scope decision:** Full categorized audit (not the narrow launchd-only slice)
**Owner-identity decision:** make owner/repo **env/config overridable, default `synaptent/aragora`**

---

## 1. Executive summary

The repository is intended to be cloned and run by the general public, but a large
amount of **install-time / author-machine state has been captured as if it were durable
repository truth**. Concretely:

- **152 tracked files** hardcode the author's home directory `/Users/armand`.
- **127 tracked files** reference the legacy GitHub username `an0mium` — and the canonical
  repo has since moved to **`synaptent/aragora`**, so several of these are now *wrong*, not
  merely private.
- **14 tracked files** hardcode `.venv/bin/python` as the runtime interpreter; when that
  virtualenv is absent (as it currently is on the author's machine), the launchd/systemd
  services fail permanently with **exit 126** instead of recovering.

This is **systemic**, not a single bug. The root cause is a pattern, not a typo: automation
and agent-generated scripts/docs baked the author's machine into committed files, and there
is **no regression guard** to stop new leaks from landing.

The good news: **the correct pattern already exists in the codebase.** `aragora/__version__.py`
declares `REPOSITORY = "https://github.com/synaptent/aragora"`, and multiple modules already
resolve owner/repo from env with a `synaptent/aragora` default (`ARAGORA_BUILD_REPO`,
`ARAGORA_GITHUB_OWNER`/`ARAGORA_GITHUB_REPO`, `GITHUB_REPOSITORY`, `ARAGORA_REPO_ROOT`).
Remediation is therefore mostly **propagating an existing convention** and adding a guard —
not inventing a new architecture.

### One active bug (not cosmetic)

`.github/workflows/nightly-integration.yml:18`:

```yaml
if: github.repository == 'an0mium/aragora'
```

The canonical repo is `synaptent/aragora`, so **this workflow never runs on the real repo.**
This should use the GitHub-native, fork-portable expression and/or the correct slug.

---

## 2. Scope & methodology

Read-only `git grep` over tracked files at `origin/main` `7875dab643`. Counts are
"number of tracked files containing at least one match":

| Pattern | Files | Nature |
|---|---:|---|
| `/Users/armand` | **152** | private home dir → portability + privacy |
| `an0mium` | **127** | legacy username; some now *wrong* (repo is `synaptent`) |
| `/home/` | 53 | **mostly legitimate** deploy/service-account paths (see §5) |
| `.venv/bin/python` | **14** | install-time interpreter capture |
| `synaptent/aragora` | 395 | canonical default — **mostly fine**, make overridable |

`/Users/armand` by file type: **80 json, 35 md, 26 py, 5 yaml, 3 sh, 1 plist, 1 jsonl, 1 csv**.

`/Users/armand` by top-level dir: **115 docs, 15 scripts, 13 tests**, plus a few misc.

---

## 3. Categorized findings

Severity tiers reflect *impact on a third-party user who clones the public repo*.

### Tier A — Public-blocking (breaks or misleads non-author users)

#### A1. `.github/` repo-slug leakage (CI + copyable templates)

| File | Line(s) | Issue | Fix |
|---|---|---|---|
| `workflows/nightly-integration.yml` | 18 | **gated to `an0mium/aragora` → never runs** | `${{ github.repository }}` or `synaptent/aragora` |
| `ISSUE_TEMPLATE/config.yml` | 4,7,10,13 | help URLs point to `an0mium/aragora` | `synaptent/aragora` |
| `actions/aragora-review/README.md` | 33,44,56,68 | `uses: an0mium/aragora/...` examples | `synaptent/aragora` |
| `workflows/publish-aragora-debate.yml` | 205,206 | changelog/readme URLs | `synaptent/aragora` |
| `workflows/benchmark.yml` | 108 | `alert-comment-cc-users: '@an0mium'` | configurable / org handle |
| `workflows/templates/aragora-gauntlet-template.yml` | 11,197 | **template users copy** → propagates wrong slug | `${{ github.repository }}` |
| `workflows/templates/aragora-review-template.yml` | 8 | doc URL in copyable template | `${{ github.repository }}` |

> In workflows, prefer the GitHub-native `${{ github.repository }}` /
> `${{ github.repository_owner }}` so **every fork is automatically correct** without edits.
> Static markdown URLs (issue templates, READMEs) should use the canonical `synaptent/aragora`.

#### A2. Runtime interpreter capture (launchd / systemd) — the `.venv/bin/python` slice

Services that capture an absolute interpreter path at install time and then fail forever if
the venv moves/disappears:

- `scripts/install_boss_loop_launchd.sh`
- `scripts/install_merge_arbiter_launchd.sh`
- `scripts/run_boss_cycle.sh`
- `scripts/run_codex_insights_digest.sh`
- `scripts/run_offline_golden_path.sh`
- `deploy/systemd/aragora-pr-watcher.service`  ← same class, **Linux**

This is the surface the previously-pasted "Public-Ready Runtime Installers" plan addresses.
That plan is correct and is folded in here as **A2** (extended to cover systemd, not just
launchd). `scripts/aragora_runtime.sh` does **not** yet exist.

#### A3. Hardcoded private paths in runtime scripts

Real defaults / `cwd` / filesystem prefixes (break for other users):

| File | Line | Usage |
|---|---|---|
| `scripts/_tighten_knowledge_exceptions.py` | 196,205 | `cwd="/Users/armand/Development/aragora"`, `os.path.join(...)` |
| `scripts/publish_automation_handoffs.py` | 33 | `DEFAULT_CODEX_HOME = Path("/Users/armand/.codex")` |
| `scripts/disk_recovery_coordinator.py` | 278 | `prefix = "/Users/armand/.codex/worktrees/"` |
| `scripts/audit_test_skips.py` | 444,687 | `.replace("/Users/armand/Development/aragora/", "")` |

Leaked into agent **prompt / help strings** (degrade gracefully but leak identity and mislead):

- `scripts/settle_one_pr.py:992,1004`, `scripts/settlement_followup.py:331`,
  `scripts/pr_check_followup.py:598`, `scripts/root_guarded_queue.py:203`
  — all embed `Start from live truth in /Users/armand/Development/aragora` into prompts.
- `scripts/audit_branch_backlog_parallel.py:16`, `scripts/claim_active_agent_lane.py:61`
  — `--help` usage examples.
- `scripts/rename_to_aragora.py:152` — one-off migration example (low priority).

Shell/plist:

- `scripts/create-sprint-worktrees.sh`, `scripts/demo_design_partner.sh`,
  `scripts/run_codex_automation_publisher.sh`, `scripts/runners/com.aragora.runner-health.plist`.

#### A4. Hardcoded owner-username in source logic (trusted-author allowlists)

- `aragora/swarm/merge_arbiter.py:55` — `"an0mium"` in the trusted-author set (next to
  `"github-actions[bot]"`) used for auto-merge decisions.
- `aragora/triage/evidence.py:32` — `"an0mium"` in trusted-author markers (next to `"[bot]"`,
  `"boss-loop"`).

These are *functional config*, not docs: on a fork owned by someone else, the owner's own
automation would not be trusted. Make the allowlist env/config-overridable with the existing
default.

### Tier B — Public hygiene (privacy/professionalism; does not break)

- `/Users/armand` in **~115 docs** (status reports, runbooks, examples).
- `an0mium` in docs (**569 hits**) — example URLs, PR references, status logs.
- Mechanical replacement: `~` / `$HOME` / `<repo-root>` / `synaptent` placeholder.

### Tier C — Committed artifact hygiene

~81 committed `json`/`jsonl`/`csv` files with absolute `/Users/armand` paths baked in:

| Dir | Files |
|---|---:|
| `docs/status` | 45 |
| `docs/experiments` | 17 |
| `docs/plans` | 15 |
| `docs/receipts` | 3 |
| `benchmarks/bench_readiness` | 1 |
| `.gt/config.json` | 1 (machine-local state — likely should not be tracked) |

These are agent-generated receipts/reports. Decide per-class: **sanitize** (rewrite to
relative paths) or **stop committing** (gitignore the generator output). `.gt/config.json`
should almost certainly be untracked.

---

## 4. Explicit NON-issues (do **not** scrub)

Accuracy matters as much as coverage — these look similar but are legitimate:

- **`/home/aragora`, `/home/ubuntu`, `/home/ec2-user` under `deploy/`** — deployment and
  service-account conventions (Docker image user, EC2/Lightsail origins). Not private leaks.
- **`aragora/server/auth_checks.py:304`** — a comment on a redaction *guard* that prevents
  leaking `/home/ec2-user/aragora/`. This is a security control, keep it.
- **`synaptent/aragora` defaults in Python** (`__version__.py`, CLI parser defaults, badge,
  publish/export footers) — already correct and mostly env-overridable. Keep as default.
- **`.github/CODEOWNERS` `@an0mium`** — CODEOWNERS *requires* a real handle; per the
  owner-identity decision this stays (it is the actual repo owner).
- **Test fixtures** with `an0mium` as author data (`aragora/live/**/__tests__/*`,
  `tests/**`) — intentional sample data.

---

## 5. Remediation design (built on existing conventions)

### 5.1 Owner identity — env/config overridable, default `synaptent/aragora`

Standardize on the pattern the repo *already* uses:

- **Single source of truth:** `aragora/__version__.py:REPOSITORY`. Add a small resolver, e.g.
  `aragora/config/identity.py: resolve_github_repo()` →
  `GITHUB_REPOSITORY` env → `ARAGORA_GITHUB_OWNER`/`ARAGORA_GITHUB_REPO` env →
  default `synaptent/aragora`. (All three env hooks already exist in the codebase.)
- **Python call sites:** route literals through the resolver / existing CLI defaults.
- **`.github/` workflows:** use `${{ github.repository }}` / `${{ github.repository_owner }}`
  (auto-correct on any fork). Static markdown URLs → `synaptent/aragora`.
- **Trusted-author allowlists (A4):** seed from `resolve_github_repo()` owner +
  `ARAGORA_TRUSTED_AUTHORS` env (comma-separated), default keeps `an0mium` + bots so behavior
  is unchanged on the canonical repo.

### 5.2 Interpreter resolution — `scripts/aragora_runtime.sh` + wrappers

- New `scripts/aragora_runtime.sh` exposing `resolve_aragora_python "$REPO_ROOT"`, resolution
  order: `ARAGORA_PYTHON` → repo-local `.venv/bin/python3` → `python3` → `python` →
  `pyenv which python3`; fail with a clear diagnostic listing accepted sources.
- Wrapper scripts (`run_boss_cycle.sh`, new `run_merge_arbiter.sh`, `run_codex_insights_digest.sh`,
  `run_offline_golden_path.sh`) source the helper and resolve **every launch**.
- Installers (`install_*_launchd.sh`) generate plists that call **repo-owned wrappers**, never
  an embedded resolved path or an install-time `ARAGORA_PYTHON` export.
- `deploy/systemd/aragora-pr-watcher.service`: `ExecStart` calls a wrapper (or reads
  `ARAGORA_PYTHON` from an `EnvironmentFile`).
- Generated per-user plists/units **may** still contain the local repo path + log path (they
  are per-user generated files) — but must never contain `/Users/armand` unless that is the
  installing user's actual path.

### 5.3 Private paths in scripts (A3)

- Replace hardcoded `cwd`/prefixes with `git rev-parse --show-toplevel`,
  `Path(__file__).resolve().parents[N]`, or `ARAGORA_REPO_ROOT` (already used elsewhere).
- `DEFAULT_CODEX_HOME` → `Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))`.
- Prompt/help strings → interpolate the resolved repo root instead of a literal.

### 5.4 Prevention guard (highest leverage)

- `scripts/check_portability.py` (ripgrep-backed) + a pre-commit hook + a CI job
  (`.github/workflows/portability-lint.yml` or a step in existing lint) that **fails on new**
  tracked occurrences of: `/Users/<name>`, `/home/<name>` (excluding the `deploy/` allowlist),
  `.venv/bin/python`, and `an0mium/` URLs (excluding CODEOWNERS + test fixtures).
- Ship with an **allowlist file** seeded from the current known hits so the guard is
  **non-breaking** — it only blocks *new* regressions while Tier B/C are worked down.

---

## 6. Phased PR sequencing

Each PR is independently reviewable, additive, and reversible.

| PR | Title | Scope | Risk | Rollback |
|---|---|---|---|---|
| **0** | Portability guard | §5.4 lint + pre-commit + seeded allowlist | low | revert; guard is additive |
| **1** | CI slug correctness | A1 (`.github/` slugs; fix nightly gating bug) | low | revert workflow edits |
| **2** | Runtime installer resilience | A2 (`aragora_runtime.sh` + wrappers + installers + systemd) | med | wrappers fall back to `.venv`; restore plists from backup |
| **3** | Script path/identity parameterization | A3 + A4 | med | env defaults preserve current behavior |
| **4** | Docs scrub | Tier B mechanical replacement | low | docs-only |
| **5** | Artifact hygiene | Tier C sanitize / gitignore `.gt/config.json` etc. | low | re-add if needed |

**Sequencing rationale:** PR 0 first so the guard protects everything after it. PR 1 is a
real bug fix and trivially safe. PR 2 carries the pasted plan. PRs 3–5 grind down the long
tail under guard protection.

### Test plans (per PR, summary)

- **PR 0:** unit-test the checker (flags a planted `/Users/x` line; passes on allowlisted
  files; respects `deploy/` exceptions). Run in CI against the tree (expect green with the
  seeded allowlist).
- **PR 1:** `actionlint`/workflow parse; assert no `an0mium/aragora` remains in `.github/`
  except CODEOWNERS; render issue-template URLs resolve.
- **PR 2:** shell tests for `resolve_aragora_python` (honors `ARAGORA_PYTHON`; ignores stale
  `.venv`; falls back to `python3`; fails loudly when nothing imports deps). Installer tests
  generate plists into a temp `HOME` and assert **no `/Users/armand`** and **no captured
  `.venv/bin/python3`**, and that plists call the wrappers. Do **not** mutate real launchd.
- **PR 3:** unit tests that path/identity resolvers honor env and default to current values;
  `merge_arbiter`/`evidence` allowlist tests with overridden `ARAGORA_TRUSTED_AUTHORS`.
- **PR 4/5:** doc-link lint; assert sanitized artifacts contain no absolute home paths.

### Safety constraints (carried from the operating contract)

- No `.venv` rebuild, no launchd/systemd mutation, no service reinstall during tests.
- `.github/workflows/**` and CODEOWNERS are protected surfaces — PR 1 must be human-reviewed.
- All changes additive and behavior-preserving on the canonical `synaptent/aragora` repo.

---

## 7. Appendix — full Tier-A file lists

**A1 `.github/`:** `CODEOWNERS`*(keep)*, `ISSUE_TEMPLATE/config.yml`,
`actions/aragora-review/README.md`, `workflows/benchmark.yml`,
`workflows/nightly-integration.yml`, `workflows/publish-aragora-debate.yml`,
`workflows/templates/aragora-gauntlet-template.yml`,
`workflows/templates/aragora-review-template.yml`.

**A2 `.venv/bin/python` (code):** `scripts/install_boss_loop_launchd.sh`,
`scripts/install_merge_arbiter_launchd.sh`, `scripts/run_boss_cycle.sh`,
`scripts/run_codex_insights_digest.sh`, `scripts/run_offline_golden_path.sh`,
`deploy/systemd/aragora-pr-watcher.service`. *(also non-code: benchmarks/*, docs/*, tests/review/test_health.py)*

**A3 non-test `.py`:** `_tighten_knowledge_exceptions.py`, `audit_branch_backlog_parallel.py`,
`audit_test_skips.py`, `claim_active_agent_lane.py`, `disk_recovery_coordinator.py`,
`pr_check_followup.py`, `publish_automation_handoffs.py`, `rename_to_aragora.py`,
`root_guarded_queue.py`, `settle_one_pr.py`, `settlement_followup.py` (+ `test_debate.py` at repo root).
**A3 shell/plist:** `scripts/create-sprint-worktrees.sh`, `scripts/demo_design_partner.sh`,
`scripts/run_codex_automation_publisher.sh`, `scripts/runners/com.aragora.runner-health.plist`.

**A4 source:** `aragora/swarm/merge_arbiter.py:55`, `aragora/triage/evidence.py:32`.

---

*Generated by read-only audit on `origin/main@7875dab643`. No files were modified by the audit;
this document is the proposed plan for review before any remediation work begins.*
