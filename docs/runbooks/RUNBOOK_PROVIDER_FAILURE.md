# Provider Failure Runbook

Procedures for handling AI provider outages.

## Quick Diagnosis

```bash
# Check provider status in health response
curl -s http://localhost:8080/api/health | jq .providers
```

## Provider Status Pages

| Provider | Status Page |
|----------|-------------|
| Anthropic | https://status.anthropic.com |
| OpenAI | https://status.openai.com |
| Google (Gemini) | https://status.cloud.google.com |
| Mistral | https://status.mistral.ai |
| xAI | Check Twitter @xaboratory |

## Automatic Fallback

Aragora automatically falls back to OpenRouter when primary providers fail:

1. Provider returns 429 (rate limit) or 5xx error
2. CircuitBreaker trips after 3 consecutive failures
3. Request automatically routed to OpenRouter

**Ensure `OPENROUTER_API_KEY` is set for fallback to work.**

## Manual Provider Disable

If a provider is causing issues, temporarily disable it:

```bash
# In .env or systemd override
ANTHROPIC_API_KEY=  # Empty to disable
```

Then restart:
```bash
sudo systemctl restart aragora
```

## Rate Limit Exhaustion

### Symptoms
- 429 errors in logs
- Debates failing mid-conversation
- Health check shows provider "degraded"

### Resolution

1. **Check usage** in provider dashboard
2. **Wait for reset** (usually hourly or daily)
3. **Enable OpenRouter fallback** as backup
4. **Consider upgrading** tier if persistent

## Provider-Specific Issues

### Anthropic

```bash
# Test connectivity
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
```

### OpenAI

```bash
# Test connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

### OpenRouter

```bash
# Test connectivity
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models
```

## Degraded Mode Operation

When all providers are down:

1. Debates queue but don't execute
2. Health endpoint returns "degraded"
3. Existing cached responses still work
4. New debates return provider unavailable error

## Recovery Verification

After provider recovers:

```bash
# 1. Check health shows available
curl -s http://localhost:8080/api/health | jq .providers

# 2. Run test debate
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{"task":"Test: Is 2+2=4?","agents":["claude","gpt4"]}'

# 3. Verify completion
curl http://localhost:8080/api/debates/{debate_id}
```

## Escalation

If provider outage lasts >1 hour:
1. Check provider status page for updates
2. Switch to backup providers
3. Consider notifying users if service significantly degraded
