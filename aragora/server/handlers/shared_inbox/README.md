# Shared Inbox Handlers

Team inbox management APIs with intelligent message routing, assignment workflows, and rule-based automation.

## Modules

| Module | Purpose |
|--------|---------|
| `handler.py` | Main SharedInboxHandler class with route dispatch |
| `inbox_handlers.py` | Inbox CRUD and message management endpoints |
| `rule_handlers.py` | Routing rule creation, testing, and management |
| `rules_engine.py` | Rule evaluation and message routing logic |
| `validators.py` | Input validation, regex safety, rate limiting |
| `models.py` | Data models (SharedInbox, RoutingRule, Message) |
| `storage.py` | Storage utilities and in-memory caches |

## Endpoints

### Inboxes
- `GET /api/v1/shared-inboxes` - List team inboxes
- `POST /api/v1/shared-inboxes` - Create inbox
- `GET /api/v1/shared-inboxes/{id}` - Get inbox details
- `GET /api/v1/shared-inboxes/{id}/messages` - List inbox messages

### Messages
- `POST /api/v1/shared-inboxes/{id}/messages` - Add message to inbox
- `PUT /api/v1/shared-inboxes/{inbox_id}/messages/{msg_id}/assign` - Assign message
- `PUT /api/v1/shared-inboxes/{inbox_id}/messages/{msg_id}/status` - Update status
- `POST /api/v1/shared-inboxes/{inbox_id}/messages/{msg_id}/tags` - Add tag

### Routing Rules
- `GET /api/v1/shared-inboxes/{id}/routing-rules` - List rules
- `POST /api/v1/shared-inboxes/{id}/routing-rules` - Create rule
- `PUT /api/v1/shared-inboxes/{id}/routing-rules/{rule_id}` - Update rule
- `DELETE /api/v1/shared-inboxes/{id}/routing-rules/{rule_id}` - Delete rule
- `POST /api/v1/shared-inboxes/{id}/routing-rules/{rule_id}/test` - Test rule

## RBAC Permissions

| Permission | Description |
|------------|-------------|
| `inbox:read` | View inboxes and messages |
| `inbox:write` | Create inboxes, manage messages |
| `inbox:assign` | Assign messages to team members |
| `inbox:rules` | Create and manage routing rules |

## Rule Conditions

Rules can match on these message fields:

| Field | Operators |
|-------|-----------|
| `from` | equals, contains, regex |
| `to` | equals, contains, regex |
| `subject` | equals, contains, regex |
| `body` | contains, regex |
| `priority` | equals, greater_than, less_than |
| `tags` | contains, not_contains |

## Rule Actions

| Action | Description |
|--------|-------------|
| `assign_to_user` | Route to specific team member |
| `assign_to_group` | Route to team group |
| `add_tag` | Apply tag to message |
| `set_priority` | Override message priority |
| `forward_to_inbox` | Move to another inbox |
| `archive` | Auto-archive matching messages |

## Usage

```python
from aragora.server.handlers.shared_inbox import (
    SharedInboxHandler,
    validate_routing_rule,
    apply_routing_rules_to_message,
)

# Validate a routing rule before creation
result = validate_routing_rule(rule_data)
if not result.valid:
    print(f"Errors: {result.errors}")

# Apply rules to incoming message
matched_rules = apply_routing_rules_to_message(message, rules)
```

## Features

- **Team Inboxes**: Shared mailboxes for team collaboration
- **Intelligent Routing**: Rule-based message assignment
- **Circular Detection**: Prevents infinite routing loops
- **Safe Regex**: ReDoS-protected regex pattern matching
- **Rate Limiting**: Protects against rule creation abuse
- **Activity Logging**: Complete audit trail of actions
