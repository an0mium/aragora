# Agent Selection Guide

This guide helps you choose the right AI agents for your Aragora stress-tests based on task type, cost, and capability requirements.

Perspective coverage note: Mistral adds an EU lens, and Chinese models like DeepSeek, Qwen, and Kimi provide a Chinese perspective (use the providers and keys listed below).

## Available Agents

### Primary Providers (Direct API)

| Agent ID | Provider | Model | Best For | Cost |
|----------|----------|-------|----------|------|
| `anthropic-api` | Anthropic | Claude 3.5 Sonnet | Code review, reasoning | $$ |
| `openai-api` | OpenAI | GPT-4 Turbo | General tasks, creativity | $$ |
| `gemini-api` | Google | Gemini Pro | Long context, analysis | $ |
| `mistral-api` | Mistral | Mistral Large | European compliance, multilingual | $$ |
| `grok-api` | xAI | Grok | Real-time knowledge | $$ |

### OpenRouter Providers (Fallback/Alternative)

| Agent ID | Model | Best For | Cost |
|----------|-------|----------|------|
| `openrouter-api` | Auto-routes | Fallback when primary fails | Varies |
| `deepseek-api` | DeepSeek V3 | Code, math, reasoning | $ |
| `qwen-api` | Qwen 2.5 | Multilingual, code | $ |
| `llama-api` | Llama 3.3 70B | General, open weights | $ |
| `yi-api` | Yi 34B | Chinese/English | $ |

**Cost Legend:** $ = Low ($0.001-0.01/1K tokens), $$ = Medium ($0.01-0.05/1K), $$$ = High ($0.05+/1K)

## Task-Based Recommendations

### Code Review

**Recommended:** `anthropic-api,openai-api`

```bash
git diff main | aragora review --agents anthropic-api,openai-api
```

Why:
- Claude excels at code understanding and security analysis
- GPT-4 provides creative edge case detection
- Consensus between them = high confidence findings

**Budget alternative:** `anthropic-api,deepseek-api`
- DeepSeek V3 is excellent at code for 1/10th the cost

### Architecture Design

**Recommended:** `anthropic-api,openai-api,gemini-api`

```bash
aragora ask "Review this microservices architecture" \
  --agents anthropic-api,openai-api,gemini-api \
  --rounds 3
```

Why:
- Three diverse perspectives catch more issues
- Gemini handles long architecture documents well
- Multiple rounds allow deeper exploration

### Compliance Audits

**Recommended:** `anthropic-api,mistral-api`

```bash
aragora gauntlet policy.md --agents anthropic-api,mistral-api --persona gdpr
```

Why:
- Mistral is trained with European data/compliance focus
- Claude provides strong reasoning for legal interpretation
- Both have strong safety training

### Quick Validation

**Recommended:** `anthropic-api` (single agent)

```bash
aragora review --demo  # Or single agent for speed
aragora review --agents anthropic-api
```

Why:
- Fastest response time
- Claude alone catches most critical issues
- Use for early-stage development feedback

### High-Stakes Decisions

**Recommended:** `anthropic-api,openai-api,gemini-api,mistral-api`

```bash
aragora gauntlet critical_spec.md \
  --agents anthropic-api,openai-api,gemini-api,mistral-api \
  --profile thorough
```

Why:
- Four perspectives maximize coverage
- Different training data catches different blind spots
- Worth the cost for critical decisions

## Cost Optimization

### Single API Key Strategy

If you only have one API key:

```bash
# Anthropic only
export ANTHROPIC_API_KEY=your_key
aragora review  # Auto-detects and uses available agent

# OpenAI only
export OPENAI_API_KEY=your_key
aragora review  # Auto-detects and uses available agent
```

### Budget-Conscious Setup

Use OpenRouter for cost-effective multi-agent:

```bash
export OPENROUTER_API_KEY=your_key
aragora review --agents deepseek-api,qwen-api
```

Estimated costs per 10K token stress-test:
- `anthropic-api,openai-api`: ~$0.30
- `deepseek-api,qwen-api`: ~$0.03 (10x cheaper)

### Automatic Fallback

Aragora auto-falls back to OpenRouter on rate limits:

```bash
# Primary keys + fallback
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export OPENROUTER_API_KEY=fallback_key

# If primary hits rate limit, falls back automatically
aragora review --agents anthropic-api,openai-api
```

## Capability Matrix

| Capability | Claude | GPT-4 | Gemini | Mistral | DeepSeek |
|------------|--------|-------|--------|---------|----------|
| Code Understanding | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Security Analysis | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| Reasoning Depth | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| Long Context | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Multilingual | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| Creativity | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |
| Safety/Refusals | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| Response Speed | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ |

## Role Assignment

Aragora auto-assigns roles based on agent order:

```bash
aragora ask "Design auth system" --agents anthropic-api,openai-api,gemini-api
```

| Position | Role | Best Agent Type |
|----------|------|-----------------|
| 1st | **Proposer** | Strong reasoning (Claude, GPT-4) |
| 2nd | **Critic** | Detail-oriented (Claude, Mistral) |
| 3rd | **Synthesizer** | Balanced (GPT-4, Gemini) |

You can also specify roles explicitly:
```bash
--agents anthropic-api:proposer,openai-api:critic,gemini-api:synthesizer
```

## Environment Variables

```bash
# Primary providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AI...
export MISTRAL_API_KEY=...
export XAI_API_KEY=...

# Fallback provider
export OPENROUTER_API_KEY=sk-or-...
```

## Recommendations by Use Case

| Use Case | Agents | Rounds | Profile |
|----------|--------|--------|---------|
| Quick PR review | `anthropic-api,openai-api` | 2 | - |
| Security audit | `anthropic-api,openai-api,mistral-api` | 3 | `thorough` |
| Architecture review | `anthropic-api,openai-api,gemini-api` | 3 | `thorough` |
| GDPR compliance | `anthropic-api,mistral-api` | 2 | `policy` |
| Code refactoring | `anthropic-api,deepseek-api` | 2 | `code` |
| Budget review | `deepseek-api,qwen-api` | 2 | `quick` |
| High-stakes decision | All 4 primary | 4 | `thorough` |

## Troubleshooting

### "No API keys configured"
Set at least one provider key:
```bash
export ANTHROPIC_API_KEY=your_key
```

### Rate limiting
Add OpenRouter fallback:
```bash
export OPENROUTER_API_KEY=fallback_key
```

### Slow responses
- Use fewer agents
- Use `--rounds 1` for faster results
- Use `gemini-api` (fastest response time)

### Inconsistent results
- Add more agents for consensus
- Use `--rounds 3` or more
- Prefer Claude for consistent reasoning

## Related Documentation

- [Environment Variables](./ENVIRONMENT.md) - Full API key reference
- [Gauntlet Mode](./GAUNTLET.md) - Stress testing configuration
- [API Reference](./API_REFERENCE.md) - Programmatic agent selection
