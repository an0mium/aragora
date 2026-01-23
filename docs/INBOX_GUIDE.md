# Smart Inbox Guide

Aragora's Smart Inbox uses multi-agent vetted decisionmaking to intelligently prioritize, thread, and manage your email. This guide covers setup, configuration, and best practices.

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
- Unified inbox messages are cached in-memory by default; configure Redis/Postgres
  sync backends for production-grade durability.
- OAuth credentials are read from environment variables (see `OAUTH_SETUP.md`).

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

## Configuration

### Basic Configuration

```python
from aragora.services.email_prioritization import EmailPrioritizationConfig

config = EmailPrioritizationConfig(
    # VIP settings - always prioritized
    vip_domains={"partner.com", "investor.com"},
    vip_addresses={"ceo@company.com", "boss@work.com"},
    internal_domains={"yourcompany.com"},

    # Auto-archive patterns
    auto_archive_senders={"noreply@marketing.com"},
    newsletter_patterns=[
        r"unsubscribe",
        r"email preferences",
        r"no.?reply",
    ],

    # Urgency detection
    urgent_keywords=[
        "urgent", "asap", "immediately", "critical",
        "emergency", "deadline", "today", "eod",
    ],

    # Confidence thresholds
    tier_1_confidence_threshold=0.7,
    tier_2_confidence_threshold=0.6,
)
```

### Cross-Channel Integration

Enable context from other channels:

```python
config = EmailPrioritizationConfig(
    enable_slack_signals=True,   # Boost emails from active Slack contacts
    enable_calendar_signals=True, # Boost emails about upcoming meetings
    enable_drive_signals=True,    # Boost emails about shared documents
)
```

## Email Threading

Reconstruct conversations from fragmented inbox:

```python
from aragora.services.email_threading import EmailThreader

# Create threader
threader = EmailThreader(
    min_participant_overlap=0.5,  # 50% participant overlap to merge
    enable_semantic_matching=False,  # Enable AI-based matching
)

# Thread emails
threads = threader.thread_emails(emails)

for thread in threads:
    print(f"Thread: {thread.subject}")
    print(f"  Participants: {len(thread.participants)}")
    print(f"  Messages: {thread.message_count}")
    print(f"  Unread: {thread.unread_count}")
```

### Threading Strategies

The threader uses multiple strategies in order:

1. **In-Reply-To Header** (RFC 5322) - Most reliable
2. **References Header** - Follows full reply chain
3. **Gmail Thread ID** - Native Gmail threading
4. **Subject Matching** - Normalized subject comparison
5. **Participant Overlap** - Contacts + timeframe

### Thread Summaries

Generate AI summaries of conversations:

```python
# Get thread summary
summary = threader.get_thread_summary(thread)

print(f"Summary: {summary.summary}")
print(f"Key Points: {summary.key_points}")
print(f"Action Items: {summary.action_items}")
print(f"Urgency: {summary.urgency}")
```

### Thread Management

```python
# Merge two threads
merged = threader.merge_threads(thread_id_1, thread_id_2)

# Split messages into new thread
new_thread = threader.split_thread(
    thread_id="thread_abc123",
    message_ids=["msg_1", "msg_2"],
)

# Find related threads
related = threader.find_related_threads(thread_id, max_results=5)
```

## Multi-Account Support

Manage multiple Gmail accounts:

```python
from aragora.storage.gmail_tokens import GmailTokenStore

# Initialize token store
store = GmailTokenStore(db_path="~/.aragora/gmail_tokens.db")

# Store token for an account
await store.store_token(
    user_id="user_123",
    email="work@company.com",
    token_data={
        "access_token": "ya29...",
        "refresh_token": "1//...",
        "expires_at": 1700000000,
    },
)

# List all accounts
accounts = await store.list_accounts(user_id="user_123")
for account in accounts:
    print(f"Account: {account['email']}")

# Aggregate inbox across accounts
from aragora.services.inbox_aggregator import InboxAggregator

aggregator = InboxAggregator(token_store=store, user_id="user_123")
all_emails = await aggregator.fetch_all_inboxes(limit_per_account=50)
ranked = await aggregator.prioritize_aggregated(all_emails)
```

## Learning from Actions

The system learns from your behavior:

```python
# Record user action
await prioritizer.record_user_action(
    email_id="msg_abc123",
    action="replied",  # read, opened, archived, deleted, replied, starred, important
    email=email,
    user_id="user_123",
    response_time_minutes=30,
)
```

### Action Effects on Learning

| Action | Learning Effect |
|--------|-----------------|
| `replied` quickly | Boost sender priority |
| `archived` without reading | Lower sender priority |
| `starred` | Mark as important pattern |
| `deleted` | Strong negative signal |
| `marked_spam` | Add to blocklist |

## Sender Management

### VIP Senders

```python
# Add VIP at runtime
prioritizer.config.vip_addresses.add("important@partner.com")
prioritizer.config.vip_domains.add("bigclient.com")

# VIP senders get automatic priority boost
```

### Blocklist

```python
from aragora.services.sender_history import SenderHistoryService

history = SenderHistoryService()
await history.initialize()

# Block a sender
await history.block_sender(
    user_id="user_123",
    sender_email="spam@badactor.com",
    reason="Persistent spam",
)

# Check if blocked
is_blocked = await history.is_blocked("user_123", "spam@badactor.com")

# Unblock
await history.unblock_sender("user_123", "spam@badactor.com")
```

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inbox/prioritize` | POST | Prioritize inbox |
| `/api/inbox/email/{id}/priority` | GET | Get email priority |
| `/api/inbox/threads` | GET | List threaded conversations |
| `/api/inbox/thread/{id}` | GET | Get thread details |
| `/api/inbox/thread/{id}/summary` | GET | Get thread summary |
| `/api/inbox/accounts` | GET | List connected accounts |
| `/api/inbox/action` | POST | Record user action |

### WebSocket Events

```javascript
// Connect to inbox sync stream
const ws = new WebSocket('wss://api.aragora.ai/ws/inbox');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'email_received':
      console.log('New email:', data.email_id);
      break;
    case 'priority_updated':
      console.log('Priority changed:', data.email_id, data.priority);
      break;
    case 'thread_updated':
      console.log('Thread updated:', data.thread_id);
      break;
  }
};
```

## Best Practices

### Performance

1. **Use Tier 1 for bulk** - Process large inboxes with Tier 1 only first
2. **Selective Tier 3** - Only run vetted decisionmaking on truly ambiguous emails
3. **Cache sender profiles** - Reuse sender reputation data
4. **Batch processing** - Use `rank_inbox()` for multiple emails

### Accuracy

1. **Configure VIPs** - Add important contacts upfront
2. **Set internal domains** - Mark your company domain as internal
3. **Tune thresholds** - Adjust confidence thresholds based on your needs
4. **Enable learning** - Record actions to improve over time

### Privacy

1. **Local processing** - Email content stays on your infrastructure
2. **Minimal retention** - Only metadata stored for learning
3. **Opt-out available** - Users can disable learning

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Slow prioritization | Check Tier 2/3 usage; adjust thresholds |
| Wrong priorities | Configure VIP senders; tune keywords |
| Missed threads | Lower `min_participant_overlap` |
| Too many newsletters | Add patterns to `newsletter_patterns` |

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger("aragora.services.email_prioritization").setLevel(logging.DEBUG)
logging.getLogger("aragora.services.email_threading").setLevel(logging.DEBUG)

# Check scoring details
result = await prioritizer.score_email(email)
print(f"Tier used: {result.tier_used}")
print(f"Scores: {result.to_dict()['scores']}")
print(f"Cross-channel: {result.to_dict()['cross_channel_boosts']}")
```

## Related Documentation

- [API Reference](./API_REFERENCE.md) - Full API documentation
- [Connectors Setup](./CONNECTORS_SETUP.md) - Gmail connector setup
- [Control Plane Guide](./CONTROL_PLANE_GUIDE.md) - Enterprise inbox management
