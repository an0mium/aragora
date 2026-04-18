# Hardening Follow-up (2026-04-17)

Residual work identified while closing the HIGH-GRAVITY incident hardening
pass (P0–P3). Captured here so the next pass has exact scope; nothing here
is urgent.

## 1. Migrate direct `os.environ.get(LLM_API_KEY)` call-sites to the AWS-first helper

**Canonical helper:** `aragora.config.legacy.get_api_key(*env_vars, required=True)`
(priority: AWS Secrets Manager bundle `aragora/production` when
`ARAGORA_USE_SECRETS_MANAGER=true`, then env vars).

**Scope (2026-04-17 grep):** 57 call-sites across 25 files. Categorized:

### Safe mechanical migration (actual key consumers, ~17 sites)

These files fetch a key and pass it to an SDK client. Replace
`os.environ.get("X_API_KEY")` with `get_api_key("X_API_KEY", required=False)`.

- `aragora/services/expense_tracker.py:812,816`
- `aragora/computer_use/claude_bridge.py:100`
- `aragora/evaluation/llm_judge.py:835`
- `aragora/evolution/evolver.py:509,541,542`
- `aragora/verification/deepseek_prover.py:96`
- `aragora/verification/formal.py:422,449,450,681,715,716`
- `aragora/nomic/task_decomposer.py:1547,1576`
- `aragora/harnesses/codex.py:167`
- `aragora/swarm/issue_upgrader.py:541,558`
- `aragora/swarm/rescue_planner.py:115`
- `aragora/connectors/whisper.py:215`
- `aragora/security/openrouter_rotator.py:137`
- `aragora/routing/domain_matcher.py:406`
- `aragora/agents/cli_agents.py:307`

### Do NOT migrate blindly — routing / presence gates (~30 sites)

These call-sites use key presence as a boolean ("if ANY provider key is
available, enable agent X") to decide which agents to include in a debate.
Migrating to AWS-first would change behavior: a call that currently returns
empty (and correctly excludes the agent) would start returning the AWS value
and include the agent. Each site needs judgment: should it participate in the
strict-mode/AWS-first flow, or should it remain a pure env-presence check?

- `aragora/cli/review.py:153,157,161,166`
- `aragora/cli/demo.py:610,611,612`
- `aragora/cli/commands/triage.py:704`
- `aragora/cli/commands/debate.py:1993`
- `aragora/cli/setup.py:429,444`
- `aragora/config/minimal.py:140,141,142,145`
- `aragora/inbox/triage_runner.py:425,432,439,447,454,461,469,476,483`
- `aragora/swarm/review_routing.py:237`
- `aragora/swarm/boss_loop_selection.py:32,36`
- `aragora/prompt_engine/{researcher,interrogator,spec_builder,decomposer}.py`
- `aragora/rlm/bridge.py:288`
- `aragora/agents/fallback.py:307`

### Untouchable (by design)

- `aragora/nomic/task_decomposer.py:2745` — this is the only call that
  **writes** to `os.environ` (setting `OPENROUTER_API_KEY` for a subprocess).
  Leave as-is.

## 2. Local/AWS key drift to resolve

Comparison of local `.env` vs `aragora/production` bundle (both us-east-1/us-east-2
carry the same bundle). Tails shown; full values are sensitive.

| Key                | .env tail | bundle tail | Decision needed |
|--------------------|-----------|-------------|-----------------|
| `XAI_API_KEY`      | `ZG63`    | `GyIR`      | Determine which is current at api.x.ai; sync the other direction |
| `GROK_API_KEY`     | `ZG63`    | *missing*   | Same value as XAI locally; bundle doesn't track. Either drop the alias or add to bundle |
| `MISTRAL_API_KEY`  | `UOmh`    | `ZZfP`      | Determine which is current at api.mistral.ai |
| `GEMINI_API_KEY`   | empty     | `Ekso`      | Local was blanked; AWS canonical. Verify AWS value works against the Gemini API |
| `FAL_API_KEY`      | `814f`    | *missing*   | Either drop local or add to bundle |
| `KIMI_API_KEY`     | empty     | `ysXx`      | AWS canonical; aragora reads it from bundle via `get_secret` |
| `DEEPSEEK_API_KEY` | empty     | `3e97`      | AWS canonical |
| `ELEVENLABS_API_KEY` | empty   | `52b8`      | AWS canonical |

Use `python scripts/secrets_manager.py sync <NAME>` after deciding the
direction. Do NOT blank local values in `.env` until AWS is verified live
against the provider, or the next run in strict mode will fail loud on
whichever agent needs it.

## 3. AWS secret layout note — do NOT delete

The `aragora/api/anthropic`, `aragora/api/openai`, `aragora/api/openrouter`
singular secrets were initially suspected to be orphaned after the LaunchAgent
stopped consuming them. Audit (`rg "aragora/api/(anthropic|openai|openrouter)"`)
confirmed they are **actively used** by:

- `scripts/rotate_keys.py` — rotation automation (writes both bundle and singular paths)
- `scripts/secrets_manager.py` — the interactive rotation tool itself
- `aragora/security/api_key_proxy.py:221` — proxy's standalone-secret path
- `tests/security/test_openrouter_rotator.py` — coverage

**Do not delete these secrets.** Removing them breaks rotation and the
OpenRouter rotator flow.

## 4. LaunchAgent changes (already applied 2026-04-17 18:49)

For reference only; no action needed:

- `~/.local/bin/aragora-codex-env.sh` — `SECRET_BINDINGS` array + `fetch_secret()`
  + AWS CLI check removed. Still hydrates `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`,
  `ARAGORA_REVIEW_PROVIDER_ORDER`, `ARAGORA_CLAUDE_REVIEW_PROFILES` into launchctl
  user env at login.
- Backup at `~/.aragora_hardening_snapshot_2026-04-17/aragora-codex-env.sh.bak`.
- Log: `~/.aragora/logs/codex-env.log`.

## 5. Factory.app / launchctl verification

After Factory was relaunched at 2026-04-17 ~19:00:

- `launchctl getenv ANTHROPIC_API_KEY` → empty
- `launchctl getenv OPENAI_API_KEY` → empty
- `launchctl getenv OPENROUTER_API_KEY` → empty
- Leaked HIGH-GRAVITY VgAA-tailed key (sha16 `fea6ff9a7261558d`) is gone
  from every user-space env.

## 6. Open items outside the hardening tail

These are tracked in the parent `README.md` ranked remediation list:

1. Publish Tier-1 benchmark (standalone `aragora-debate` vs. solo Opus 4.7).
2. Fix silent failures in `aragora gauntlet --local` and `aragora demo`
   (no `--offline`).
3. Refactor `aragora/debate/orchestrator.py` (1,270 LOC → ≤300 LOC per module).
4. Narrow product positioning to one vertical with one headline use-case.
