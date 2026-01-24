# SME Quick Start Guide

Get your first AI-facilitated debate running in under 15 minutes.

---

## Prerequisites

- A workspace on [aragora.ai](https://aragora.ai) or self-hosted instance
- Admin access to create integrations
- A Slack workspace (recommended) or email access

---

## Step 1: Create Your Workspace (2 min)

1. Sign up at [aragora.ai/signup](https://aragora.ai/signup)
2. Enter your organization name
3. Choose a workspace URL (e.g., `acme.aragora.ai`)
4. Verify your email

---

## Step 2: Connect Slack (3 min)

1. Go to **Settings > Integrations**
2. Click **Add Slack**
3. Authorize the Aragora app in your Slack workspace
4. Select a channel for debate results

**What Aragora can do in Slack:**
- Read messages in connected channels (for evidence gathering)
- Post debate results and receipts
- Respond to `/aragora` commands

---

## Step 3: Run Your First Debate (5 min)

### Option A: From the Web UI

1. Go to **Debates > New Debate**
2. Select a template:
   - **Quick Decision** - 2 rounds, 2 agents
   - **Team Consensus** - 3 rounds, 3 agents
   - **Deep Analysis** - 4 rounds, 4 agents
3. Enter your topic (e.g., "Should we adopt TypeScript for our frontend?")
4. Click **Start Debate**

### Option B: From Slack

Type in your connected channel:
```
/aragora debate "Should we adopt TypeScript for our frontend?"
```

**Debate Options:**
```
/aragora debate --template "vendor-selection" "Which CI/CD tool should we use?"
/aragora debate --rounds 4 "How should we structure our Q2 goals?"
```

---

## Step 4: Review Results (3 min)

After the debate completes:

1. **View the Summary** - Consensus decision with confidence score
2. **Explore Arguments** - See what each AI agent proposed
3. **Check Dissent** - Review any minority opinions
4. **Download Receipt** - PDF or Markdown for records

### Understanding the Receipt

| Field | Meaning |
|-------|---------|
| **Verdict** | The consensus decision |
| **Confidence** | How strongly agents agreed (0-100%) |
| **Risk Level** | LOW (>70%), MEDIUM (50-70%), HIGH (<50%) |
| **Participants** | Which AI models participated |
| **Receipt ID** | Unique identifier for audit trail |

---

## Step 5: Customize Your Setup

### Add Team Members

1. Go to **Settings > Team**
2. Click **Invite Member**
3. Enter email addresses
4. Choose role: **Admin** or **Member**

### Set Budget Limits

1. Go to **Settings > Billing**
2. Set monthly spend cap
3. Configure alerts (50%, 75%, 90%)

### Explore Templates

View all workflow templates:
```
/aragora templates
```

Popular templates:
- **Hiring Decision** - Evaluate candidates
- **Feature Prioritization** - Rank backlog items
- **Vendor Selection** - Compare tools/services
- **Policy Review** - Draft or update policies

---

## API Access (Optional)

For programmatic access:

### Python SDK

```bash
pip install aragora-client
```

```python
from aragora_client import AragoraClient

client = AragoraClient(
    base_url="https://acme.aragora.ai",
    api_key="your-api-key"
)

# Start a debate
result = await client.debates.create(
    task="Should we migrate to Kubernetes?",
    rounds=3,
    agents=["claude", "gpt-4o", "gemini"]
)

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.0%}")
```

### TypeScript SDK

```bash
npm install @aragora/sdk
```

```typescript
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: 'https://acme.aragora.ai',
  apiKey: 'your-api-key'
});

const result = await client.debates.create({
  task: 'Should we migrate to Kubernetes?',
  rounds: 3,
  agents: ['claude', 'gpt-4o', 'gemini']
});

console.log(`Verdict: ${result.verdict}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(0)}%`);
```

---

## Troubleshooting

### Debate takes too long

- Reduce rounds (2-3 is usually sufficient)
- Use fewer agents (2-3 for quick decisions)
- Check your network connection

### Slack integration not working

1. Verify the app is installed: `/aragora status`
2. Check channel permissions
3. Re-authorize at Settings > Integrations > Slack

### Low confidence scores

- Add more context to your question
- Try different agents for the topic
- Use more debate rounds

---

## Next Steps

- [Full Documentation](https://docs.aragora.ai)
- [Template Library](https://docs.aragora.ai/templates)
- [API Reference](https://docs.aragora.ai/api)
- [Community Discord](https://discord.aragora.ai)

---

*Need help? Email support@aragora.ai or use the chat widget in the app.*
