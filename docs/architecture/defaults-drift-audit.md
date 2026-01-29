# Defaults Drift Audit

**Date:** 2026-01-29  
**Owner:** Platform  
**Scope:** Debate defaults (rounds, agents, consensus, max rounds)

---

## Canonical Defaults (Server)

From `aragora/config/settings.py` + `aragora/config/legacy.py`:

- **DEFAULT_ROUNDS:** 9  
- **MAX_ROUNDS:** 12  
- **DEFAULT_CONSENSUS:** judge  
- **DEFAULT_AGENTS:** grok, anthropic-api, openai-api, deepseek, mistral, gemini, qwen, kimi

---

## Alignment Status

**Aligned**
- API defaults and server handlers (config + legacy constants).
- OpenAPI examples and constraints (default 9 rounds, max 12).
- Live UI defaults for agents/rounds/consensus.

**Fixed in this pass**
- Live UI `MAX_ROUNDS` default updated to 12 (`aragora/live/src/config.ts`).

---

## Intentional Deviations (Documented)

These are feature-specific defaults and should remain **explicitly documented**:

- **LangChain integration** defaults to legacy agent names (`aragora/integrations/langchain/tools.py`, `chains.py`).
- **Twilio voice** uses a minimal agent set (`aragora/integrations/twilio_voice.py`).
- **Telegram bot** limits agents to a subset for latency (`aragora/server/handlers/bots/telegram.py`).
- **Canvas manager** has its own fallback agent list (`aragora/canvas/manager.py`).

---

## Guardrails Added

**Test:** `tests/test_defaults_alignment.py`  
