# Incident Report — HIGH-GRAVITY Anthropic Key Leak

| Field | Value |
|---|---|
| Date of leak detection by Anthropic | 2026-04-07 |
| Date of Anthropic auto-revocation | 2026-04-07 |
| Date of full incident response | 2026-04-17 |
| Leaked key name | `aragora` |
| Leaked key tail | `…VgAA` |
| Leaked key id | `334029dc-0421-4a4d-b1e1-2e30311fd326` |
| Account | `armand@synaptent.com` |
| Source of leak | `github.com/SWORDIntel/HIGH-GRAVITY` (public credential-harvesting op) |

## 1. Scope of the harvester

`SWORDIntel/HIGH-GRAVITY` is an organized credential dump targeting Windsurf IDE
users. The aggregate dump contains:

| Provider  | Keys in dump | User impacted? |
|-----------|--------------|----------------|
| Anthropic | 89           | Yes — 1 key (`aragora`, tail `VgAA`) |
| Gemini    | 19           | No — user's key tail `QuOglZ6zuc` not in dump |
| Other     | various      | Not in use |

## 2. Root cause

The Anthropic key was stored in a Windsurf VS Code extension workspace state
file. The extension either exfiltrated the key directly or it was scraped from
a session log. The Windsurf extension was uninstalled by the user during
incident response, eliminating the active exfiltration vector.

A secondary forensic signal was discovered on 2026-04-17: **Factory.app's
Anthropic integration was still holding and re-injecting the revoked VgAA-tailed
key into every Droid shell session.** This suggests a second storage location
of the leaked key, independent of Windsurf. See §6 for remediation.

## 3. Actions taken

### P0 — Containment

- [x] Anthropic auto-revoked the leaked key on 2026-04-07
- [x] Windsurf VS Code extension uninstalled by the user
- [x] Confirmed user's Gemini key NOT in harvester dump

### P1 — Key rotation (all via `scripts/secrets_manager.py rotate`)

| Key | Rotated on | Status |
|---|---|---|
| `OPENAI_API_KEY` | 2026-04-17 | ACTIVE, synced to local / AWS us-east-1 / AWS us-east-2 / GitHub Secrets |
| `OPENROUTER_API_KEY` | 2026-04-17 | ACTIVE, synced across all backends |
| `ANTHROPIC_API_KEY` | 2026-04-17 | ACTIVE, synced across all backends; live probe with `claude-opus-4-7` returned "Pong!" |

### P2 — Local-machine hardening (2026-04-17 afternoon)

Assumes the local machine may be compromised. Goal: a full clone of this
machine is useless without AWS credentials.

- [x] **Deleted `~/.claude/.api_key`** (held a dead hgAA-tailed Anthropic key)
- [x] **Removed `use-api` alias from `~/.zshrc`** that loaded a plaintext key
      from `~/.claude/.api_key`; replaced with a commented-out AWS-sourced
      alternative
- [x] **Blanked LLM API keys in `/Users/armand/Development/aragora/.env`**:
      `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
      `GEMINI_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`,
      `ELEVENLABS_API_KEY` — all verified identical to AWS before blanking
      (see §5 for drift on XAI/GROK/MISTRAL/FAL)
- [x] **Added LLM keys to `CRITICAL_SECRETS`** in `aragora/config/secrets.py`
      (ANTHROPIC, OPENAI, OPENROUTER, GEMINI) so strict mode refuses env-var
      fallback for them
- [x] **Defaulted `ARAGORA_SECRETS_STRICT=true`** in `aragora/.env` so even
      local dev fails loud if AWS is unreachable, rather than silently
      consuming a stale .env value

### Defense-in-depth (completed earlier in session)

- [x] 9 GitHub secret-scanning alerts on `synaptent/aragora` dismissed as
      test-fixture false positives
- [x] 19 public repos scanned with gitleaks v8.30.1: 0 real credential leaks
      (20 gitleaks hits were all placeholder strings like `sk-1234567890abcdef`,
      `test-jwt-secret-…`)
- [x] Last 30 GitHub Actions workflow runs on `synaptent/aragora` scanned
      (30k log lines): 0 key-shaped patterns
- [x] `gitleaks` pre-commit hook bumped v8.18.4 → v8.30.0 and now runs at
      **both** `pre-commit` AND `pre-push` stages

### Model pinning (completed earlier in session)

- [x] Frontier pins: Opus 4.7 / GPT-5.4 / Gemini 3.1 Pro
- [x] Every legacy model ID routes to the frontier via OpenRouter
- [x] A missing `ANTHROPIC_API_KEY` now auto-falls-back to
      `anthropic/claude-opus-4.7` via OpenRouter at every call site

## 4. Verification (post-hardening)

Reproduced 2026-04-17 18:20 local (on the just-hardened machine):

| Test | Result |
|---|---|
| `get_secret("ANTHROPIC_API_KEY")` with blanked env + strict mode | Returns AWS value (sha16 `7d418451706d571a`) |
| Same path via `legacy.get_api_key("ANTHROPIC_API_KEY")` | Returns same AWS value |
| Live `claude-opus-4-7` call via anthropic SDK with AWS-sourced key | Returns `"Pong!"` |
| Strict mode with AWS access blocked | Raises `SecretNotFoundError` (fail-loud, no silent fallback) |
| `aragora demo --offline` | Produces Consensus receipt in < 1 s |
| 49 convergence tests | PASS |
| 62 smoke tests | PASS |
| `gitleaks` pre-commit + pre-push on full tree | PASS |

## 5. Residual risks & follow-ups

### P0 — manual, cannot be automated from this session

| Risk | Mitigation |
|---|---|
| **Factory.app still holds a revoked VgAA key** and re-injects it into every Droid shell. | Open Factory → Settings → Integrations → Anthropic and **delete** the integration. Do NOT paste the newly-rotated key back in. Either rely on Claude Max subscription, or create a scoped sub-key with a $5/day budget specifically for Factory. |
| 11–13 unused Anthropic accounts may have orphaned keys. | Log into each, revoke all keys, close or mark dormant. |

### P1 — follow-up engineering passes

| Item | Current state | Proposed fix |
|---|---|---|
| Non-agent code reading `os.environ.get("ANTHROPIC_API_KEY")` directly (21 callsites across `computer_use/`, `evaluation/`, `evolution/`, `verification/`, `nomic/`, `swarm/`) | Currently returns empty string after .env blanking; underlying SDK raises. Fail-loud, but not ideal. | Migrate to `aragora.config.legacy.get_api_key()` (AWS-first wrapper). Scope: ~2-hour pass with test coverage. |
| `.env` still has XAI_API_KEY, GROK_API_KEY, MISTRAL_API_KEY, FAL_API_KEY with drift vs AWS | Not critical (not on harvester target list) | Run `python scripts/secrets_manager.py sync <NAME>` for each, then blank in .env. |
| `ARAGORA_JWT_SECRET`, `ARAGORA_SECRET_KEY`, `ARAGORA_RECEIPT_SIGNING_KEY`, `SUPABASE_DB_PASSWORD`, `GITHUB_PERSONAL_ACCESS_TOKEN`, etc. still in .env | Critical infrastructure keys. Already in CRITICAL_SECRETS. | Blank in .env after verifying all match AWS. Requires similar audit pass. |

### P2 — monitoring

| Item | Proposed |
|---|---|
| No runtime alert when `os.environ.get` returns the blank placeholder | Add a one-time startup logger that reports any CRITICAL_SECRETS not present in the AWS cache. |
| No automated rotation reminder | Calendar reminder from `rotation-schedule.yaml` (next LLM rotation due 2026-07-16). |

## 6. Files changed this session (post-hardening)

| File | Change |
|---|---|
| `~/.claude/.api_key` | **DELETED** (backed up at `~/.aragora_hardening_snapshot_2026-04-17/claude_api_key.bak`) |
| `~/.zshrc` | Removed `use-api` alias, documented AWS-sourced alternative |
| `/Users/armand/Development/aragora/.env` | Blanked 7 LLM keys; added `ARAGORA_SECRETS_STRICT=true` |
| `aragora/config/secrets.py` | Added `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` to `CRITICAL_SECRETS` |
| `benchmarks/bench_readiness/incident_2026-04-07_high-gravity.md` | This file |
| `benchmarks/bench_readiness/rotation-schedule.yaml` | Next-due calendar reflecting 2026-04-17 rotations |
| `benchmarks/bench_readiness/manifest.json` | Post-hardening reproducibility snapshot |

Safety snapshots of all modified files are at
`~/.aragora_hardening_snapshot_2026-04-17/` (not checked into git).

## 7. Net posture

A full clone of the local machine no longer yields working LLM credentials.
Running `aragora` requires AWS credentials in `~/.aws/credentials`; without
them, strict mode raises `SecretNotFoundError` and refuses to start. The
weakest remaining link is Factory.app's stale Anthropic integration, which is
a manual UI fix.
