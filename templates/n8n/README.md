# n8n Workflow Templates for Aragora

Pre-built automation recipes connecting Obsidian, Aragora, and Linear.

## Available Workflows

### 1. Obsidian → Aragora → Linear Pipeline

**File:** `obsidian-aragora-linear.json`

Complete thought-to-action automation:

```
Obsidian Note (#ready) → Aragora Debate → Decision Integrity → Linear Issue → Obsidian → Slack
```

#### What it does

1. **Watches** Obsidian vault for notes tagged `#ready`
2. **Launches** multi-agent debate via Aragora API
3. **Builds** decision integrity package (receipt + plan)
4. **Creates** Linear issue from decision
5. **Writes** decision integrity note back to Obsidian
6. **Notifies** team via Slack

#### Setup

1. Import `obsidian-aragora-linear.json` into n8n
2. Configure environment variables:

```env
# Obsidian
OBSIDIAN_VAULT_PATH=/path/to/your/vault

# Aragora
ARAGORA_API_URL=https://api.aragora.ai
ARAGORA_API_KEY=your_api_key

# Linear
LINEAR_API_KEY=lin_api_xxx
LINEAR_TEAM_ID=team_id
LINEAR_DECISION_LABEL_ID=label_id

# Slack (optional)
SLACK_CHANNEL=#decisions
```

3. Configure Slack credentials in n8n (optional)
4. Activate the workflow

#### Usage

Tag any Obsidian note with `#ready` to trigger:

```markdown
---
title: API Design Decision
date: 2026-02-03
tags:
  - architecture
  - api
---

# Should we use REST or GraphQL?

Context and considerations here... #ready
```

The workflow will:
- Remove `#ready` tag and add `#processed`
- Add `aragora_id` and `linear_issue` to frontmatter
- Create `decisions/2026-02-03-API-Design-Decision-integrity.md`

---

## Creating Custom Workflows

### Aragora API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/debates` | POST | Launch debate |
| `/api/v2/debates/{id}` | GET | Get status |
| `/api/v1/debates/{id}/decision-integrity` | POST | Build receipt + plan |
| `/api/v2/gauntlet/run` | POST | Stress test |
| `/api/v2/knowledge/search` | GET | Search knowledge |

### Webhook Events

Subscribe to these events from Aragora:

- `debate_start` - Debate initiated
- `debate_end` - Debate completed
- `consensus` - Consensus reached
- `decision_made` - Final decision recorded
- `gauntlet_complete` - Stress test finished

### Example: Webhook Trigger

```json
{
  "type": "n8n-nodes-base.webhook",
  "parameters": {
    "path": "aragora-webhook",
    "httpMethod": "POST"
  }
}
```

Configure in Aragora:
```bash
curl -X POST https://api.aragora.ai/api/v2/webhooks \
  -H "Authorization: Bearer $ARAGORA_API_KEY" \
  -d '{
    "url": "https://your-n8n.com/webhook/aragora-webhook",
    "events": ["debate_end", "consensus"]
  }'
```

---

## Support

- [Aragora Documentation](https://docs.aragora.ai)
- [n8n Documentation](https://docs.n8n.io)
- [Linear API](https://developers.linear.app)
