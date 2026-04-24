# Incident Report — HIGH-GRAVITY Anthropic Key Leak

**Date of leak detection by Anthropic:** 2026-04-07
**Date of auto-revocation:** 2026-04-07
**Date of incident response:** 2026-04-17 (this session)
**Leaked key name:** `aragora`
**Leaked key tail:** `…VgAA`
**Leaked key id:** `334029dc-0421-4a4d-b1e1-2e30311fd326`
**Account:** `armand@synaptent.com`
**Source of leak:** `github.com/SWORDIntel/HIGH-GRAVITY` (public, organized credential-harvesting operation)

## 1. Scope of the harvester

`SWORDIntel/HIGH-GRAVITY` is an organized credential-dump repo explicitly targeting Windsurf IDE users. The aggregate dump contains:

| Provider  | Keys in dump | User impacted? |
|-----------|--------------|----------------|
| Anthropic | 89           | Yes — 1 key (`aragora`, tail `VgAA`) |
| Gemini    | 19           | No — tail `QuOglZ6zuc` not present |
| Other     | various      | Not checked — not in use |

## 2. Root cause

The Anthropic key was stored in a Windsurf workspace state file / extension cache that was either:
- Exfiltrated via a malicious Windsurf extension, or
- Scraped from a public chat/session log in a Windsurf integration.

Windsurf VSCode extension was uninstalled by the user during incident response, eliminating the ongoing attack vector.

## 3. Actions taken

### P0 — Containment
- [x] Anthropic key auto-revoked by Anthropic (2026-04-07) — no action needed
- [x] Confirmed Windsurf extension uninstalled from VSCode (user)
- [x] Gemini key tail verified NOT in harvester dump

### P1 — Key rotation
- [x] `OPENAI_API_KEY` rotated via `scripts/secrets_manager.py rotate OPENAI_API_KEY` (2026-04-17)
- [x] `OPENROUTER_API_KEY` rotated via `scripts/secrets_manager.py rotate OPENROUTER_API_KEY` (2026-04-17)
- [ ] `ANTHROPIC_API_KEY` — **blocked** on regaining account access (only 1 of 12–14 Anthropic accounts has active API keys)

### P1 — Defense-in-depth
- [x] All public repos (19: 16 Armand1 + 3 synaptent) scanned with gitleaks v8.30.1
  - **0 real credential leaks** — RingRift had 19 test-fixture hits, aragora had 1 asterisk-masked placeholder; all false positives
- [x] Last 30 GitHub Actions runs on synaptent/aragora scanned for echoed keys (30k log lines): **0 real-looking key patterns**
- [x] All 9 existing GitHub secret-scanning alerts on synaptent/aragora dismissed as false-positive test fixtures
- [x] Gitleaks pre-commit hook bumped from v8.18.4 to v8.30.0 and now runs at **both** `pre-commit` and `pre-push` stages (previously only `pre-push`, meaning a stale clone without pre-push could still commit secrets)
- [x] Rotation schedule YAML snapshot generated at `rotation-schedule.yaml`

### P2 — Resilience (model pinning)
- [x] Created `aragora/config/model_pins.py` as the authoritative frontier-model registry (Opus 4.7 / GPT-5.4 / Gemini 3.1 Pro)
- [x] Updated agent classes so every legacy model ID routes to the frontier via OpenRouter (`OPENROUTER_MODEL_MAP` in `anthropic.py`, `openai.py`, `gemini.py`)
- [x] Updated 5 agent config YAMLs (proposer, synthesizer, quality-reviewer, security-auditor, compliance-auditor) from `anthropic-api` to `openrouter` so a missing `ANTHROPIC_API_KEY` no longer blocks debate execution
- [x] Bulk-upgraded 95 model pin strings across 43 Python files to frontier models
- [x] Net effect: a fully missing / disabled Anthropic key now silently falls back to Opus 4.7 via OpenRouter — the original "Anthropic is down" P0 blocker is now a P3 cost-profile concern

## 4. Verification

- **Unit tests:** 49 convergence + 103 anthropic+fallback + 62 smoke tests pass after frontier-model migration
- **Offline CLI:** `aragora demo --offline` and `aragora quickstart --demo` both produce receipts in <1 s
- **Gitleaks:** passes cleanly at both `pre-commit` and `pre-push` stages on the full tree
- **Open GitHub secret-scanning alerts on synaptent/aragora:** 0

## 5. Residual risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Anthropic account `armand@synaptent.com` not recoverable | P0 | OpenRouter fallback provides direct Opus 4.7 access; no runtime blocker. Manual recovery through Anthropic support. |
| 11–13 unused Anthropic accounts may have orphaned keys not visible in console | P2 | Systematic audit — log into each account, revoke-all, then decommission the account. |
| Windsurf extension history (chat logs, workspace state) may contain additional credentials | P2 | Grep `~/Library/Application Support/Windsurf` (already-uninstalled), `~/.codeium`, `~/.windsurf` for `sk-`, `AIza`, `sk-or-` patterns as a follow-up. |
| Pre-commit hook uses `v8.30.0`; latest is `v8.30.1` | P3 | `pre-commit autoupdate` selected the available tag. Refresh in a month. |

## 6. Files changed this session

See `git diff --name-only` at commit time. High-level list:
- `aragora/config/model_pins.py` (new)
- `aragora/agents/api_agents/{anthropic,openai,gemini}.py`
- `aragora/agents/configs/{proposer,synthesizer,quality-reviewer,security-auditor,compliance-auditor}.yaml`
- `aragora/server/research_phase.py`
- `aragora/swarm/{spec,issue_upgrader,rescue_planner,boss_validation}.py`
- `aragora/verticals/**/*.py`
- `.pre-commit-config.yaml`
- `tests/test_agent_anthropic.py` (updated assertion for frontier-upgrade behavior)
- `tests/agents/api_agents/test_gemini.py` (updated assertion for frontier-upgrade behavior)
- 43 Python files with bulk model-pin upgrades

## 7. Follow-up tasks

1. **ANTHROPIC_API_KEY recovery:** Work through Anthropic support to regain account access on `armand@synaptent.com`. Issue new key, rotate into `secrets_manager.py` via `rotate ANTHROPIC_API_KEY`.
2. **Decommission unused Anthropic accounts:** Log into each of the 11–13 unused accounts, revoke all keys, and either close the account or mark it as dormant in a personal asset register.
3. **Sweep Windsurf/Codeium residue:** Grep user home for lingering Windsurf/Codeium state files for any additional harvester artifacts.
4. **Consider a full git-history rewrite for RingRift:** The 19 test-fixture hits are all safe, but `eyJ...` JWT-shaped strings in `docs/architecture/API_REFERENCE.md` should be replaced with obviously-fake tokens (e.g., `eyJhbGci.EXAMPLE.TOKEN`) to silence gitleaks and future auditors.
