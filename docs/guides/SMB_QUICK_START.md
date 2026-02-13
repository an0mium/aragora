# Aragora SMB Quick Start Guide

Get started with Aragora in 5 minutes. This guide is designed for small and medium businesses looking to add AI-powered deliberation to their decision-making processes.

## Prerequisites

- Python 3.10+ or Node.js 18+
- An API key from at least one LLM provider (Anthropic, OpenAI, or OpenRouter)
- 10 minutes for initial setup

## Step 1: Install the SDK

Choose your preferred language:

**Python:**
```bash
pip install aragora-sdk
```

**TypeScript/JavaScript:**
```bash
npm install @aragora/sdk
# or
yarn add @aragora/sdk
```

## Step 2: Configure API Keys

Set your LLM provider API key as an environment variable:

```bash
# Pick one (or use OpenRouter for access to multiple models)
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
# or
export OPENROUTER_API_KEY="your-key-here"  # Recommended for SMB
```

**Why OpenRouter?** OpenRouter provides access to 40+ models through a single API key, with automatic fallback when one provider is unavailable. Perfect for SMB budgets.

## Step 3: Run Your First Debate

**Python:**
```python
from aragora import Client

# Initialize client
client = Client()

# Start a debate
result = await client.debates.create(
    task="Should we implement a 4-day work week for our 50-person team?",
    rounds=3,  # 3 rounds for faster results
)

print(f"Decision: {result.consensus}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Key considerations: {result.summary}")
```

**TypeScript:**
```typescript
import { createClient } from '@aragora/sdk';

const client = createClient();

const result = await client.debates.create({
  task: "Should we implement a 4-day work week for our 50-person team?",
  rounds: 3,
});

console.log(`Decision: ${result.consensus}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(0)}%`);
```

## Step 4: Connect to Your Chat Platform (Optional)

Aragora can receive questions from Slack, Teams, or Discord and return answers directly to your team.

**Slack Integration:**
```bash
# 1. Create a Slack app at api.slack.com/apps
# 2. Configure OAuth scopes: chat:write, app_mentions:read
# 3. Set webhook URL to: https://your-domain.com/api/webhooks/slack

export SLACK_BOT_TOKEN="xoxb-your-token"
export SLACK_SIGNING_SECRET="your-secret"
```

**Teams Integration:**
```bash
# 1. Register bot in Azure Portal
# 2. Configure messaging endpoint

export TEAMS_APP_ID="your-app-id"
export TEAMS_APP_PASSWORD="your-password"
```

## Common SMB Use Cases

### 1. Product Decisions
```python
result = await client.debates.create(
    task="Should we add feature X that 30% of customers requested, "
         "given our 3-person dev team and Q2 deadline?",
    context={
        "customer_requests": 150,
        "dev_capacity_hours": 200,
        "deadline": "2024-06-30",
    }
)
```

### 2. Hiring Decisions
```python
result = await client.debates.create(
    task="Should we hire a senior developer at $150K or two juniors at $80K each?",
    context={
        "current_team_size": 5,
        "mentorship_capacity": "limited",
        "project_complexity": "high",
    }
)
```

### 3. Process Changes
```python
result = await client.debates.create(
    task="Should we switch from weekly to bi-weekly sprints?",
    context={
        "current_velocity": 20,
        "team_feedback": "sprint planning takes too long",
    }
)
```

### 4. Vendor Selection
```python
result = await client.debates.create(
    task="Which CRM should we choose: Salesforce ($500/mo), HubSpot ($200/mo), or Pipedrive ($100/mo)?",
    context={
        "team_size": 10,
        "sales_team": 3,
        "current_tools": ["Slack", "Notion"],
    }
)
```

## Cost Optimization Tips

1. **Use fewer rounds for simple decisions**
   - 1-2 rounds: Yes/no questions, quick sanity checks
   - 3-5 rounds: Standard business decisions
   - 7-9 rounds: Complex strategic decisions

2. **Leverage OpenRouter fallback**
   ```python
   client = Client(
       fallback_enabled=True,  # Uses cheaper models when primary is unavailable
   )
   ```

3. **Cache repeated decisions**
   ```python
   # Enable knowledge mound for learning from past decisions
   client = Client(
       enable_knowledge_mound=True,
   )
   ```

4. **Set budget limits**
   ```python
   result = await client.debates.create(
       task="...",
       max_cost_usd=1.00,  # Cap spending per debate
   )
   ```

## Monitoring Your Usage

Check your usage dashboard:
```python
usage = await client.billing.get_usage(period="month")
print(f"Debates this month: {usage.debate_count}")
print(f"Total cost: ${usage.total_cost:.2f}")
print(f"Avg cost per debate: ${usage.avg_cost_per_debate:.2f}")
```

## Next Steps

- **[API Reference](../api/API_QUICK_START.md)** - Full API documentation
- **[Pricing Tiers](../reference/PRICING_TIERS.md)** - Compare plans
- **[Chat Integration](../integrations/CHANNELS.md)** - Connect Slack/Teams/Discord
- **[Best Practices](./SME_GA_GUIDE.md)** - Tips for better decisions

## Getting Help

- **Community Discord**: Join our SMB users channel
- **Email Support**: support@aragora.ai (Pro/Enterprise)
- **Documentation**: docs.aragora.ai

## FAQ

**Q: How much does a typical debate cost?**
A: With OpenRouter and 3 rounds, expect $0.05-0.20 per debate. Complex 9-round debates with Claude/GPT-4 may cost $0.50-2.00.

**Q: Can I run Aragora locally?**
A: Yes! Use Ollama for local inference (free, but slower):
```bash
export ARAGORA_LLM_PROVIDER=ollama
```

**Q: How long does a debate take?**
A: 3-round debates typically complete in 30-60 seconds. 9-round debates may take 2-5 minutes.

**Q: Is my data private?**
A: Yes. We don't store your prompts or decisions unless you enable the Knowledge Mound feature for learning. See our [Privacy Policy](../enterprise/PRIVACY_POLICY.md).
