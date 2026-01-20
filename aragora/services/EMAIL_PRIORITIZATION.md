# Email Prioritization System

Intelligent, multi-tier email prioritization using AI-powered analysis.

## Overview

The Email Prioritization System helps users manage their inbox by automatically scoring and ranking emails based on importance, urgency, and historical patterns. It uses a 3-tier scoring system that balances speed and accuracy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Email Prioritization                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Tier 1    │    │   Tier 2    │    │   Tier 3    │         │
│  │  Rule-Based │───▶│  Lightweight│───▶│ Multi-Agent │         │
│  │   <200ms    │    │   <500ms    │    │   <5s       │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Sender History Service                  │          │
│  │  - Reputation tracking                            │          │
│  │  - Response patterns                              │          │
│  │  - VIP/blocked management                         │          │
│  └──────────────────────────────────────────────────┘          │
│                           │                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │         Cross-Channel Context Service             │          │
│  │  - Slack activity signals                         │          │
│  │  - Calendar integration                           │          │
│  │  - Google Drive relevance                         │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Priority Levels

| Priority | Value | Description | Examples |
|----------|-------|-------------|----------|
| CRITICAL | 5 | Requires immediate attention | VIP sender, urgent deadline |
| HIGH | 4 | Important, respond today | Manager email, time-sensitive |
| MEDIUM | 3 | Standard priority | Regular work email |
| LOW | 2 | Can wait | FYI emails, CC'd emails |
| DEFER | 1 | Read later or archive | Newsletters, marketing |

## 3-Tier Scoring System

### Tier 1: Rule-Based (< 200ms)
Fast pattern matching for high-confidence decisions:
- VIP sender detection
- Newsletter/marketing identification
- Urgency keyword detection
- Deadline pattern matching
- Auto-archive sender matching

### Tier 2: Lightweight Agent (< 500ms)
Single-agent analysis for ambiguous cases:
- Content summarization
- Context-aware scoring
- Sender reputation integration

### Tier 3: Multi-Agent Debate (< 5s)
Full debate-based prioritization for complex decisions:
- Multiple specialized agents
- Sender Reputation Agent
- Content Urgency Agent
- Context Relevance Agent
- Billing/Legal Agent
- Timeline Agent

## API Endpoints

### Score Single Email
```http
POST /api/email/prioritize
Content-Type: application/json

{
  "email": {
    "id": "msg_123",
    "subject": "Urgent: Quarterly Review",
    "from_address": "boss@company.com",
    "body_text": "Please review by EOD...",
    "labels": ["INBOX"]
  },
  "force_tier": null  // Optional: 1, 2, or 3
}
```

Response:
```json
{
  "success": true,
  "result": {
    "email_id": "msg_123",
    "priority": "high",
    "confidence": 0.85,
    "score": 0.78,
    "tier_used": 1,
    "rationale": "Urgency keywords detected; sender is in your frequent contacts",
    "processing_time_ms": 45
  }
}
```

### Rank Inbox
```http
POST /api/email/rank-inbox
Content-Type: application/json

{
  "emails": [...],
  "limit": 50
}
```

### Fetch and Rank (Gmail Connected)
```http
GET /api/email/inbox?user_id=default&limit=50&labels=INBOX
```

### Record Feedback
```http
POST /api/email/feedback
Content-Type: application/json

{
  "email_id": "msg_123",
  "action": "replied",  // opened, archived, deleted, replied, starred
  "email": {...}  // Optional: full email for context
}
```

### VIP Management
```http
# Add VIP
POST /api/email/vip
{"sender_email": "important@company.com"}

# Remove VIP
DELETE /api/email/vip
{"sender_email": "important@company.com"}
```

### Configuration
```http
# Get config
GET /api/email/config

# Update config
PUT /api/email/config
{
  "vip_addresses": ["ceo@company.com"],
  "auto_archive_senders": ["noreply@marketing.com"],
  "tier_1_threshold": 0.8
}
```

## Gmail OAuth Integration

### 1. Get OAuth URL
```http
POST /api/email/gmail/oauth-url
{
  "user_id": "user_123",
  "redirect_uri": "https://app.example.com/inbox/callback"
}
```

### 2. Handle Callback
```http
POST /api/email/gmail/callback
{
  "code": "auth_code_from_google",
  "state": "user_123",
  "redirect_uri": "https://app.example.com/inbox/callback"
}
```

## Sender History Service

Tracks sender reputation and interaction patterns:

```python
from aragora.services import SenderHistoryService

service = SenderHistoryService(db_path="sender_history.db")
await service.initialize()

# Record interaction
await service.record_interaction(
    user_id="user@example.com",
    sender_email="important@company.com",
    action="replied",
    response_time_minutes=30,
)

# Get reputation
reputation = await service.get_sender_reputation(
    user_id="user@example.com",
    sender_email="important@company.com",
)

print(f"Reputation: {reputation.reputation_score}")
print(f"Category: {reputation.category}")  # vip, important, normal, low_priority
```

### Reputation Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Open Rate | 40% | Percentage of emails opened |
| Reply Rate | 40% | Percentage of emails replied to |
| Recency | 20% | Bonus for recent interactions |
| VIP Status | +0.3 | Manual VIP designation |
| Fast Responder | +0.1 | Avg response < 30 mins |
| High Delete Rate | -0.15 | > 50% emails deleted |

## Cross-Channel Context

Integrates signals from other platforms:

```python
from aragora.services import CrossChannelContextService

service = CrossChannelContextService(
    slack_connector=slack,
    knowledge_mound=mound,
)

# Get sender context
context = await service.get_user_context("sender@company.com")

print(f"Slack online: {context.slack.is_online}")
print(f"Activity score: {context.overall_activity_score}")
```

### Signal Types

- **Slack Activity**: Online status, active channels, urgent threads
- **Calendar**: Meeting density, availability windows
- **Drive**: Recent document activity, shared files

## Configuration Options

```python
from aragora.services import EmailPrioritizationConfig

config = EmailPrioritizationConfig(
    # VIP settings
    vip_domains={"important-company.com"},
    vip_addresses={"ceo@company.com"},

    # Internal domain boost
    internal_domains={"mycompany.com"},

    # Auto-archive
    auto_archive_senders={"noreply@marketing.com"},

    # Tier thresholds
    tier_1_confidence_threshold=0.8,  # Stay in Tier 1 if confidence >= 0.8
    tier_2_confidence_threshold=0.5,  # Escalate to Tier 3 if < 0.5

    # Cross-channel
    enable_slack_signals=True,
    enable_calendar_signals=True,

    # Urgent keywords
    urgent_keywords=["urgent", "asap", "critical", "deadline"],
)
```

## Web UI

The inbox view is available at `/inbox` and includes:

1. **Gmail Connection**: OAuth-based authentication
2. **Priority Inbox**: AI-ranked email list with priority badges
3. **Tier Summary**: Shows T1/T2/T3 distribution
4. **Priority Filters**: Filter by priority level
5. **Feedback Buttons**: Thumbs up/down for learning
6. **Config Panel**: VIP management, context toggles

## Testing

Run tests with:
```bash
pytest aragora/tests/services/test_email_prioritization.py -v
pytest aragora/tests/services/test_sender_history.py -v
```

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Tier 1 scoring | < 200ms | Rule-based only |
| Tier 2 scoring | < 500ms | Single agent call |
| Tier 3 scoring | < 5s | Full debate |
| Inbox ranking (50 emails) | < 2s | Batch processing |
| Reputation lookup | < 10ms | Cached |

## Future Enhancements

- [ ] Slack connector integration for real-time signals
- [ ] Calendar integration for meeting-aware prioritization
- [ ] Response time prediction
- [ ] Custom rules engine
- [ ] Email summarization
- [ ] Smart reply suggestions
