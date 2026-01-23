---
title: Shared Inbox & Routing Rules
description: Shared Inbox & Routing Rules
---

# Shared Inbox & Routing Rules

Aragora provides shared inbox APIs for collaborative email triage, ownership,
and routing automation. Teams can assign messages, track status, and enforce
rules that label or escalate incoming mail.

## Overview

Shared inbox features include:

- Create and manage shared inboxes per workspace.
- Assign, track, and resolve messages with status metadata.
- Define routing rules to auto-assign, label, or escalate messages.
- Store shared inbox metadata in a persistent email store (if configured).

UI route: `/shared-inbox`

## API Endpoints

### Shared Inboxes

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/inbox/shared` | Create shared inbox |
| GET | `/api/v1/inbox/shared` | List shared inboxes |
| GET | `/api/v1/inbox/shared/\{id\}` | Get inbox details |
| GET | `/api/v1/inbox/shared/\{id\}/messages` | List inbox messages |
| POST | `/api/v1/inbox/shared/\{id\}/messages/\{msg_id\}/assign` | Assign message |
| POST | `/api/v1/inbox/shared/\{id\}/messages/\{msg_id\}/status` | Update status |
| POST | `/api/v1/inbox/shared/\{id\}/messages/\{msg_id\}/tag` | Add tag |

### Routing Rules

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/inbox/routing/rules` | Create routing rule |
| GET | `/api/v1/inbox/routing/rules` | List routing rules |
| PATCH | `/api/v1/inbox/routing/rules/\{id\}` | Update routing rule |
| DELETE | `/api/v1/inbox/routing/rules/\{id\}` | Delete routing rule |
| POST | `/api/v1/inbox/routing/rules/\{id\}/test` | Test routing rule |

## Create a Shared Inbox

```http
POST /api/v1/inbox/shared
Content-Type: application/json

{
  "workspace_id": "ws_123",
  "name": "Support Inbox",
  "description": "Customer support inquiries",
  "email_address": "support@company.com",
  "connector_type": "gmail",
  "team_members": ["user1", "user2"],
  "admins": ["admin1"]
}
```

## Assign a Message

```http
POST /api/v1/inbox/shared/inbox_123/messages/msg_456/assign
Content-Type: application/json

{
  "assignee": "user2"
}
```

## Create a Routing Rule

```http
POST /api/v1/inbox/routing/rules
Content-Type: application/json

{
  "workspace_id": "ws_123",
  "name": "Escalate VIP",
  "conditions": [
    { "field": "priority", "operator": "equals", "value": "critical" }
  ],
  "condition_logic": "AND",
  "actions": [
    { "type": "assign", "target": "oncall-security" },
    { "type": "label", "target": "vip" }
  ],
  "priority": 1
}
```

## Notes

- Shared inbox data is cached in memory with optional persistence via
  `aragora/storage/email_store.py`.
- Routing rule actions include assign, label, escalate, archive, notify, and
  forward.
- For command center endpoints (daily digest, bulk actions), see
  `docs/EMAIL_PRIORITIZATION.md`.
