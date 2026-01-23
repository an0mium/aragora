---
title: Smart Inbox Guide
description: Unified inbox setup, prioritization, and vetted decisionmaking workflows.
---

# Smart Inbox Guide

Aragora's Smart Inbox uses multi-agent vetted decisionmaking to intelligently
prioritize, thread, and manage your email. This guide covers setup,
configuration, and best practices.

## Overview

The Smart Inbox provides:

| Feature | Description |
|---------|-------------|
| **AI Prioritization** | 3-tier scoring system for optimal latency/accuracy |
| **Email Threading** | Automatic conversation reconstruction |
| **Cross-Channel Context** | Integrates Slack, Calendar, Drive signals |
| **VIP Management** | Special handling for important contacts |
| **Action Learning** | Improves over time based on your behavior |
| **Multi-Account** | Manage multiple Gmail and Outlook accounts |

## Quick Start

```python
from aragora.services.email_prioritization import EmailPrioritizer, EmailPrioritizationConfig
from aragora.connectors.enterprise.communication.gmail import GmailConnector

# Connect to Gmail
gmail = await GmailConnector.from_credentials(
    credentials_path="credentials.json",
    token_path="token.json",
)

# Create prioritizer
prioritizer = EmailPrioritizer(
    gmail_connector=gmail,
    config=EmailPrioritizationConfig(
        vip_domains={"yourcompany.com"},
        vip_addresses={"ceo@partner.com", "investor@vc.com"},
    ),
)

# Score a single email
result = await prioritizer.score_email(email)
print(f"Priority: {result.priority.name}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Rationale: {result.rationale}")

# Rank entire inbox
ranked = await prioritizer.rank_inbox(emails, limit=50)
for r in ranked:
    print(f"[{r.priority.name}] {r.email_id}: {r.rationale}")
```

## Unified Inbox API (v1)

Use the unified inbox endpoints to connect accounts, sync messages, and triage
across providers.

```
POST /api/v1/inbox/connect          - Connect Gmail or Outlook account
GET  /api/v1/inbox/accounts         - List connected accounts
DELETE /api/v1/inbox/accounts/{id}  - Disconnect an account
GET  /api/v1/inbox/messages         - Prioritized messages across accounts
POST /api/v1/inbox/triage           - Multi-agent triage for a message
GET  /api/v1/inbox/stats            - Inbox health metrics
```

Notes:
- Gmail OAuth tokens are persisted in `GmailTokenStore`; Outlook tokens are stored
  via the integration store under the `outlook_email` integration type.
- Unified inbox accounts, messages, and triage results are persisted via the
  Unified Inbox store. Default backend is SQLite; for production use PostgreSQL
  by setting `ARAGORA_INBOX_STORE_BACKEND=postgres` and `DATABASE_URL`.
- OAuth credentials are read from environment variables (see [OAuth setup](../security/oauth-setup)).

## Priority Levels

| Priority | Value | Description | Typical Action |
|----------|-------|-------------|----------------|
| **CRITICAL** | 1 | Immediate attention required | Respond within 1 hour |
| **HIGH** | 2 | Important, respond today | Review and respond today |
| **MEDIUM** | 3 | Standard priority | Review within 2-3 days |
| **LOW** | 4 | Can wait | Review when time allows |
| **DEFER** | 5 | Archive or auto-file | Auto-archive or batch review |
| **BLOCKED** | 6 | Sender is blocked | Auto-archive/delete |

## 3-Tier Scoring Architecture

The prioritizer uses a tiered approach for optimal performance:

### Tier 1: Rule-Based (< 200ms)

Fast, deterministic scoring using:
- Sender reputation (VIP, internal, response history)
- Gmail labels (Important, Starred)
- Urgency keywords ("urgent", "ASAP", "deadline")
- Newsletter/bulk email detection
- Blocklist checking

```python
# Force Tier 1 only
from aragora.services.email_prioritization import ScoringTier

result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)
```

### Tier 2: Lightweight AI (< 500ms)

Single-agent analysis for ambiguous cases when Tier 1 confidence is low:
- Uses small, fast language model
- Analyzes email content and context
- Provides explanatory rationale

```python
# Tier 2 kicks in when Tier 1 confidence < 0.7
config = EmailPrioritizationConfig(
    tier_1_confidence_threshold=0.7,  # Below this, escalate to Tier 2
)
```

### Tier 3: Multi-Agent Vetted Decisionmaking (< 30s)

Full vetted decisionmaking for complex decisions when Tier 2 is still uncertain:
- Multiple specialized agents collaborate
- Cross-references Knowledge Mound
- Produces vetted decisionmaking rationale

```python
# Tier 3 kicks in when Tier 2 confidence < 0.6
config = EmailPrioritizationConfig(
    tier_2_confidence_threshold=0.6,  # Below this, escalate to Tier 3
    debate_agent_count=3,
    debate_timeout_seconds=30.0,
)
```
