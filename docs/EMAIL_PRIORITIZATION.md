# Email Prioritization & Inbox Operations

Intelligent inbox management powered by multi-agent deliberation, sender history,
and cross-channel context.

## Overview

The Email Prioritization system scores and ranks emails based on urgency,
importance, and historical behavior. It supports real-time inbox ranking,
follow-up tracking, and snooze recommendations to help teams focus on the
highest-impact messages.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMAIL PRIORITIZATION                         │
├─────────────────────────────────────────────────────────────────┤
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
│                           │                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │    Follow-Up Tracker + Snooze Recommender         │          │
│  │  - Awaiting replies                               │          │
│  │  - Smart snooze suggestions                       │          │
│  └──────────────────────────────────────────────────┘          │
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
- Sender reputation
- Content urgency
- Context relevance
- Timeline and compliance perspectives

## API Endpoints

All endpoints are served under `/api/v1/email` unless otherwise noted.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/email/prioritize` | Score a single email |
| POST | `/api/v1/email/prioritize/batch` | Score multiple emails |
| POST | `/api/v1/email/rank-inbox` | Rank a provided list |
| GET | `/api/v1/email/inbox` | Fetch + rank inbox (Gmail) |
| POST | `/api/v1/email/feedback` | Record a user action |
| POST | `/api/v1/email/feedback/batch` | Batch feedback |
| GET | `/api/v1/email/config` | Fetch per-user config |
| PUT | `/api/v1/email/config` | Update per-user config |
| POST | `/api/v1/email/vip` | Add VIP sender |
| DELETE | `/api/v1/email/vip` | Remove VIP sender |
| POST | `/api/v1/email/categorize` | Categorize a single email |
| POST | `/api/v1/email/categorize/batch` | Categorize multiple emails |
| POST | `/api/v1/email/categorize/apply-label` | Apply category label |
| POST | `/api/v1/email/gmail/oauth/url` | Start Gmail OAuth |
| POST | `/api/v1/email/gmail/oauth/callback` | Handle Gmail callback |
| GET | `/api/v1/email/gmail/status` | Gmail connection status |
| GET | `/api/v1/email/context/boost` | Cross-channel context boost |
| GET | `/api/email/daily-digest` | Daily digest stats (alias: `/api/inbox/daily-digest`) |
| POST | `/api/v1/email/followups/mark` | Mark follow-up |
| GET | `/api/v1/email/followups/pending` | List pending follow-ups |
| POST | `/api/v1/email/followups/{id}/resolve` | Resolve follow-up |
| POST | `/api/v1/email/followups/check-replies` | Check for replies |
| POST | `/api/v1/email/followups/auto-detect` | Auto-detect follow-ups |
| GET | `/api/v1/email/{id}/snooze-suggestions` | Snooze suggestions |
| POST | `/api/v1/email/{id}/snooze` | Apply snooze |
| DELETE | `/api/v1/email/{id}/snooze` | Cancel snooze |
| GET | `/api/v1/email/snoozed` | List snoozed emails |
| GET | `/api/v1/email/categories` | List categories |
| POST | `/api/v1/email/categories/learn` | Category feedback |

### Inbox Command Center APIs

The inbox command center exposes endpoints for quick actions and bulk
operations:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/inbox/command` | Fetch prioritized inbox |
| POST | `/api/inbox/actions` | Execute quick action |
| POST | `/api/inbox/bulk-actions` | Execute bulk action |
| POST | `/api/inbox/reprioritize` | Trigger AI re-prioritization |
| GET | `/api/inbox/sender-profile` | Sender profile |

### Score Single Email
```http
POST /api/v1/email/prioritize
Content-Type: application/json

{
  "email": {
    "id": "msg_123",
    "subject": "Urgent: Quarterly Review",
    "from_address": "boss@company.com",
    "body_text": "Please review by EOD...",
    "labels": ["INBOX"]
  },
  "force_tier": null
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
    "tier_used": "tier_1_rules",
    "rationale": "Urgency keywords detected; sender is VIP",
    "processing_time_ms": 45
  }
}
```

### Rank Inbox
```http
POST /api/v1/email/rank-inbox
Content-Type: application/json

{
  "emails": [...],
  "limit": 50
}
```

### Fetch + Rank (Gmail Connected)
```http
GET /api/v1/email/inbox?user_id=default&limit=50&labels=INBOX
```

### Record Feedback
```http
POST /api/v1/email/feedback
Content-Type: application/json

{
  "email_id": "msg_123",
  "action": "replied",
  "email": {...}
}
```

## Gmail OAuth Integration

### 1. Get OAuth URL
```http
POST /api/v1/email/gmail/oauth/url
{
  "user_id": "user_123",
  "redirect_uri": "https://app.example.com/inbox/callback"
}
```

### 2. Handle Callback
```http
POST /api/v1/email/gmail/oauth/callback
{
  "code": "auth_code_from_google",
  "state": "user_123",
  "redirect_uri": "https://app.example.com/inbox/callback"
}
```

## Inbox UI Surfaces

The control plane UI includes inbox components for daily summary, bulk actions,
and sender-level insights:

- `DailyDigestWidget` (`aragora/live/src/components/inbox/DailyDigestWidget.tsx`)
  renders a daily digest panel and calls `/api/email/daily-digest` with a local
  mock fallback.
- `QuickActionsBar` (`aragora/live/src/components/inbox/QuickActionsBar.tsx`)
  provides single-email and bulk actions (archive, snooze, reprioritize).
- `SenderInsightsPanel` (`aragora/live/src/components/inbox/SenderInsightsPanel.tsx`)
  surfaces sender profile statistics, AI analysis, and quick actions.

## Shared Inbox & Routing Rules

Collaborative inbox workflows (assignment, status, routing rules) are exposed via
the shared inbox API and UI. See [SHARED_INBOX.md](SHARED_INBOX.md) for the
endpoints and workflow model.

## Sender History Service

Tracks sender reputation and interaction patterns:

```python
from aragora.services.sender_history import SenderHistoryService

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

## Follow-Up Tracker

Tracks sent emails that are awaiting replies and surfaces overdue follow-ups.

```python
from datetime import datetime, timedelta
from aragora.services.followup_tracker import FollowUpTracker

tracker = FollowUpTracker()

item = await tracker.mark_awaiting_reply(
    email_id="msg_123",
    thread_id="thread_456",
    subject="Vendor security review",
    recipient="security@vendor.com",
    expected_by=datetime.now() + timedelta(days=3),
)

pending = await tracker.get_pending_followups(user_id="default")
```

Notes:
- The follow-up tracker uses in-memory storage by default.
- For production, persist follow-ups to a durable store.

## Snooze Recommender

Provides smart snooze suggestions based on sender patterns, work schedule,
and optional calendar availability.

```python
from aragora.services.snooze_recommender import SnoozeRecommender

recommender = SnoozeRecommender(sender_history=service)
recommendation = await recommender.recommend_snooze(email, priority_result)

for suggestion in recommendation.suggestions:
    print(suggestion.label, suggestion.snooze_until, suggestion.reason.value)
```

## Notes on Persistence

- Sender history uses SQLite for local persistence.
- Follow-ups and snooze recommendations are in-memory by default.
- For multi-instance deployments, use a shared store for user state.
