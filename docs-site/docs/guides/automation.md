---
title: Automation Platform Integrations
description: Automation Platform Integrations
---

# Automation Platform Integrations

> **Last Updated:** 2026-01-20

Aragora integrates with popular workflow automation platforms to enable no-code/low-code automation of multi-agent debates and decision-making workflows.

## Table of Contents

- [Overview](#overview)
- [Zapier Integration](#zapier-integration)
- [Make (Integromat) Integration](#make-integromat-integration)
- [n8n Integration](#n8n-integration)
- [LangChain Integration](#langchain-integration)
- [API Endpoints](#api-endpoints)
- [Authentication](#authentication)
- [Examples](#examples)

---

## Overview

Aragora provides native integrations for workflow automation platforms:

| Platform | Type | Use Case |
|----------|------|----------|
| **Zapier** | No-code | Connect Aragora with 5000+ apps |
| **Make** | Low-code | Complex multi-step automations |
| **n8n** | Self-hosted | Privacy-first workflow automation |
| **LangChain** | Developer | AI application development |

### Common Capabilities

All automation platforms support:
- **Triggers**: Webhook-based notifications when debates complete
- **Actions**: Start debates, submit evidence, query knowledge
- **Filtering**: Filter events by workspace, tags, confidence level

---

## Zapier Integration

### Setup

1. Go to **Settings > Integrations > Zapier** in Aragora
2. Click **Connect to Zapier** to generate API credentials
3. Search for "Aragora" in the Zapier app directory
4. Use your API key and secret to authenticate

### Triggers

| Trigger | Description | Output |
|---------|-------------|--------|
| `debate_completed` | Fires when any debate finishes | Debate ID, result, consensus |
| `consensus_reached` | Fires when consensus is achieved | Debate ID, consensus details |
| `gauntlet_completed` | Fires when stress-test completes | Gauntlet ID, risks, score |
| `decision_made` | Fires when a decision is finalized | Decision ID, verdict, evidence |

### Actions

| Action | Description | Inputs |
|--------|-------------|--------|
| `start_debate` | Start a new debate | Topic, agents, rounds, tags |
| `get_debate_status` | Get current debate status | Debate ID |
| `submit_evidence` | Submit evidence to a debate | Debate ID, content, source |

### Example Zap

**Trigger**: New row in Google Sheets
**Action**: Start Aragora debate with row content
**Result**: Multi-agent analysis of each new entry

```
Google Sheets → Aragora "Start Debate" → Slack notification
```

---

## Make (Integromat) Integration

### Setup

1. In Make, add the **Aragora** module
2. Create a new connection with your API credentials
3. Choose a trigger or action module

### Trigger Modules

| Module | Description |
|--------|-------------|
| **Watch Debates** | Monitor for new or completed debates |
| **Watch Consensus** | Monitor for consensus events |
| **Watch Decisions** | Monitor for final decisions |
| **Watch Gauntlet** | Monitor stress-test completions |

### Action Modules

| Module | Description |
|--------|-------------|
| **Create Debate** | Start a new multi-agent debate |
| **Get Debate** | Retrieve debate details and results |
| **Submit Evidence** | Add evidence to an active debate |
| **Get Agents** | List available AI agents |
| **Run Gauntlet** | Execute stress-test on content |

### Example Scenario

```
Notion Database Updated
    ↓
Aragora: Create Debate
    ↓
Router (based on consensus)
    ↓ (Yes)              ↓ (No)
Asana: Create Task    Slack: Alert Team
```

---

## n8n Integration

### Setup (Self-Hosted)

1. Install the Aragora n8n community node:
   ```bash
   npm install @aragora/n8n-nodes-aragora
   ```

2. Add credentials in n8n:
   - API URL: `https://api.aragora.ai`
   - API Key: Your Aragora API key
   - API Secret: Your Aragora API secret

### Node Types

#### Aragora Trigger Node

Webhook-based trigger for debate events:
- **Events**: debate.created, debate.completed, consensus.reached
- **Filtering**: By workspace, tags, confidence threshold

#### Aragora Node

CRUD operations on Aragora resources:

| Resource | Operations |
|----------|------------|
| Debate | Create, Get, GetAll, Update, Delete |
| Agent | Get, GetAll |
| Evidence | Create, Get, GetAll |
| Decision | Get, GetAll |
| Gauntlet | Execute, Get |
| Knowledge | Query, Create, Update |

### Example Workflow

```json
{
  "nodes": [
    {
      "name": "Aragora Trigger",
      "type": "@aragora/n8n-nodes-aragora.aragoraTrigger",
      "parameters": {
        "events": ["debate.completed"],
        "filter": {
          "minConfidence": 0.8
        }
      }
    },
    {
      "name": "Aragora",
      "type": "@aragora/n8n-nodes-aragora.aragora",
      "parameters": {
        "resource": "decision",
        "operation": "get",
        "debateId": "={{$json.debate_id}}"
      }
    }
  ]
}
```

---

## LangChain Integration

### Installation

```bash
pip install aragora[langchain]
# or
pip install aragora langchain-core
```

### Components

#### AragoraTool

Use Aragora as a LangChain tool:

```python
from aragora.integrations.langchain import AragoraTool

# Create the tool
tool = AragoraTool(
    api_base="https://api.aragora.ai",
    api_key="your-api-key"
)

# Run a debate
result = tool.run("Should we use microservices or monolith architecture?")
print(result)
```

#### AragoraRetriever

Use Aragora's knowledge base as a retriever:

```python
from aragora.integrations.langchain import AragoraRetriever

retriever = AragoraRetriever(
    api_base="https://api.aragora.ai",
    api_key="your-api-key"
)

# Get relevant documents
docs = retriever.get_relevant_documents("database architecture patterns")
for doc in docs:
    print(doc.page_content)
```

#### AragoraCallbackHandler

Stream debate events to your application:

```python
from aragora.integrations.langchain import AragoraCallbackHandler

handler = AragoraCallbackHandler(
    on_debate_start=lambda debate_id: print(f"Started: \{debate_id\}"),
    on_round_complete=lambda round: print(f"Round \{round\} done"),
    on_consensus=lambda result: print(f"Consensus: \{result\}")
)

tool = AragoraTool(
    api_base="https://api.aragora.ai",
    api_key="your-api-key",
    callbacks=[handler]
)
```

#### With LangGraph

```python
from langgraph.graph import StateGraph
from aragora.integrations.langchain import AragoraTool

def debate_node(state):
    tool = AragoraTool(api_base="...", api_key="...")
    result = tool.run(state["question"])
    return {"debate_result": result}

graph = StateGraph()
graph.add_node("debate", debate_node)
# ... configure graph
```

---

## API Endpoints

All automation integrations use these REST endpoints:

### External Integration API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/integrations/external/providers` | GET | List available providers |
| `/api/integrations/external/connect` | POST | Connect a provider |
| `/api/integrations/external/disconnect` | POST | Disconnect a provider |
| `/api/integrations/external/status` | GET | Get connection status |
| `/api/integrations/external/test` | POST | Test a connection |

### Webhook Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/webhooks` | GET | List all webhooks |
| `/api/webhooks` | POST | Create a webhook |
| `/api/webhooks/\{id\}` | DELETE | Delete a webhook |
| `/api/webhooks/\{id\}/test` | POST | Test a webhook |

---

## Authentication

### API Key Authentication

All automation platforms use API key authentication:

```http
Authorization: Bearer YOUR_API_KEY
X-Aragora-Secret: YOUR_API_SECRET
```

### Webhook Signature Verification

Webhooks include a signature header for verification:

```http
X-Aragora-Signature: sha256=HMAC_SIGNATURE
```

Verify in your application:

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256=\{expected\}", signature)
```

---

## Examples

### Automated Decision Workflow

```
Email received (Gmail)
    ↓
Extract decision question (OpenAI)
    ↓
Run Aragora debate (3 rounds, 5 agents)
    ↓
Branch on consensus?
    ↓ Yes                    ↓ No
Create Jira ticket       Schedule meeting (Calendar)
with decision            for human review
```

### Continuous Compliance Check

```
GitHub PR opened
    ↓
Get PR diff content
    ↓
Run Aragora Gauntlet (security persona)
    ↓
Post results as PR comment
    ↓
If high risk → Block merge
```

### Knowledge Base Sync

```
Notion page updated (trigger)
    ↓
Aragora: Submit evidence
    ↓
Aragora: Update knowledge base
    ↓
Slack: Notify team of new knowledge
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Webhook not firing | Check webhook URL is publicly accessible |
| Authentication failed | Verify API key and secret are correct |
| Timeout errors | Debates may take time; increase timeout |
| Rate limited | Implement exponential backoff |

### Rate Limits

| Platform | Limit |
|----------|-------|
| Zapier | 100 requests/minute |
| Make | 100 requests/minute |
| n8n | No limit (self-hosted) |
| LangChain | API rate limits apply |

---

## See Also

- [BOT_INTEGRATIONS.md](./BOT_INTEGRATIONS.md) - Chat platform integrations
- [API_REFERENCE.md](./API_REFERENCE.md) - Full API documentation
- [WEBHOOKS.md](./WEBHOOKS.md) - Webhook configuration guide
