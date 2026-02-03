# ADR-002: Agent Fallback via OpenRouter

## Status
Accepted

## Context

API agents frequently encounter quota/billing errors from providers:
- Anthropic: "credit balance is too low"
- OpenAI: Rate limit exceeded (429)
- Provider outages and temporary failures

Without fallback, debates would fail entirely when a single provider has issues. This creates:
- Poor user experience
- Unreliable debate execution
- Wasted partial progress

## Decision

Implement automatic fallback to OpenRouter when primary providers fail:

1. **QuotaFallbackMixin**: Shared mixin for all API agents
   ```python
   class AnthropicAPIAgent(QuotaFallbackMixin, APIAgent):
       OPENROUTER_MODEL_MAP = {
           "claude-opus-4-5-20251101": "anthropic/claude-sonnet-4",
           # ...
       }
   ```

2. **Quota Detection**: Check for billing/quota errors
   ```python
   def is_quota_error(self, status: int, error_text: str) -> bool:
       return status == 402 or "credit balance" in error_text.lower()
   ```

3. **Transparent Fallback**: Same interface, different backend
   ```python
   async def fallback_generate(self, prompt, context, original_status):
       if not os.getenv("OPENROUTER_API_KEY"):
           return None  # No fallback available
       # Use OpenRouter with mapped model
   ```

4. **Model Mapping**: Map provider models to OpenRouter equivalents
   - Claude Opus 4.5 → anthropic/claude-sonnet-4
   - GPT-4 → openai/gpt-4-turbo
   - etc.

## Consequences

### Positive
- **Resilience**: Debates continue despite provider issues
- **Cost optimization**: Can use cheaper fallback models
- **Transparency**: Same API, automatic failover
- **Logging**: Fallback events logged for monitoring

### Negative
- **Model differences**: Fallback model may behave differently
- **Additional dependency**: Requires OPENROUTER_API_KEY for fallback
- **Complexity**: More code paths to test and maintain

### Neutral
- Fallback is opt-in via `enable_fallback` parameter
- OpenRouter costs tracked separately

## Related
- `aragora/agents/fallback.py` - QuotaFallbackMixin
- `aragora/agents/api_agents/*.py` - Agent implementations
- `aragora/resilience.py` - CircuitBreaker for repeated failures
