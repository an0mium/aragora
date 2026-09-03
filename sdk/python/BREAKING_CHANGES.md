# Python SDK Breaking Changes

This document tracks breaking changes specific to the Aragora Python SDK. For core API breaking changes, see the main [BREAKING_CHANGES.md](../../docs/BREAKING_CHANGES.md).

---

## Version 2.x

### Unreleased (2026-09-03)

#### Breaking Changes

Removed 12 `debates` methods (sync `DebatesAPI` and async `AsyncDebatesAPI`)
that targeted routes no server handler dispatches. Every one of them was
already marked DEPRECATED and emitted a `DeprecationWarning`; each call
fell through to the debate slug lookup and returned 404, so no working
integration depended on them.

| Removed Method | Route | Migration |
|----------------|-------|-----------|
| `debates.restore(debate_id)` | `POST /api/v1/debates/{id}/restore` | `debates.update(debate_id, status="active")` |
| `debates.make_permanent(debate_id)` | `POST /api/v1/debates/{id}/make-permanent` | None needed; completed debates persist automatically |
| `debates.find_similar(debate_id, limit)` | `GET /api/v1/debates/{id}/similar` | `consensus.get_similar_debates(topic, limit)` (`GET /api/v1/consensus/similar`) with the debate task text as `topic` |
| `debates.get_quality(debate_id)` | `GET /api/v1/debates/{id}/quality` | `debates.get_verification_report(debate_id)` |
| `debates.get_notes(debate_id)` | `GET /api/v1/debates/{id}/notes` | No replacement (server has no notes feature) |
| `debates.add_note(debate_id, content)` | `POST /api/v1/debates/{id}/notes` | No replacement (server has no notes feature) |
| `debates.delete_note(debate_id, note_id)` | `DELETE /api/v1/debates/{id}/notes/{note_id}` | No replacement (server has no notes feature) |
| `debates.get_batch_results(batch_id)` | `GET /api/v1/debates/batch/{id}/results` | `debates.get_batch_status(batch_id)` (includes per-job results) |
| `debates.cancel_batch(batch_id)` | `POST /api/v1/debates/batch/{id}/cancel` | Cancel individual debates with `debates.cancel(debate_id)` |
| `debates.retry_batch(batch_id)` | `POST /api/v1/debates/batch/{id}/retry` | Re-submit failed jobs with `debates.submit_batch(...)` |
| `debates.get_agent_statistics(debate_id)` | `GET /api/v1/debates/{id}/agent-statistics` | `debates.get_agent_stats()` (aggregate per-agent statistics) |
| `debates.get_debate_health(debate_id)` | `GET /api/v1/debates/{id}/health` | `debates.get_health()` for system health, `debates.get(debate_id)` for a debate's status |

Removed 9 `webhooks` methods (sync `WebhooksAPI` and async `AsyncWebhooksAPI`)
that targeted per-webhook sub-routes no server handler dispatches. The
webhook handler only routes `/api/v1/webhooks/{id}` and
`/api/v1/webhooks/{id}/test` below a webhook id, so every one of these calls
returned 404 and no working integration depended on them.

| Removed Method | Route | Migration |
|----------------|-------|-----------|
| `webhooks.list_deliveries(webhook_id, status, limit, offset)` | `GET /api/v1/webhooks/{id}/deliveries` | `webhooks.get_delivery_stats(webhook_id)` for per-webhook counts; `webhooks.list_dead_letter()` for failed deliveries |
| `webhooks.get_delivery(webhook_id, delivery_id)` | `GET /api/v1/webhooks/{id}/deliveries/{delivery_id}` | `webhooks.get_dead_letter(dead_letter_id)` for a failed delivery; successful deliveries are not exposed individually |
| `webhooks.retry_delivery(webhook_id, delivery_id)` | `POST /api/v1/webhooks/{id}/deliveries/{delivery_id}/retry` | `webhooks.retry_dead_letter(dead_letter_id)` |
| `webhooks.subscribe_events(webhook_id, events)` | `POST /api/v1/webhooks/{id}/events` | `webhooks.update(webhook_id, events=[...])` with the full event list |
| `webhooks.unsubscribe_events(webhook_id, events)` | `DELETE /api/v1/webhooks/{id}/events` | `webhooks.update(webhook_id, events=[...])` with the remaining events |
| `webhooks.get_retry_policy(webhook_id)` | `GET /api/v1/webhooks/{id}/retry-policy` | No replacement; the retry policy is server-configured, not per webhook |
| `webhooks.update_retry_policy(webhook_id, **policy)` | `PUT /api/v1/webhooks/{id}/retry-policy` | No replacement; the retry policy is server-configured, not per webhook |
| `webhooks.rotate_secret(webhook_id)` | `POST /api/v1/webhooks/{id}/rotate-secret` | `webhooks.delete(webhook_id)` then `webhooks.create(...)`; the secret is returned once on creation |
| `webhooks.get_signing_info(webhook_id)` | `GET /api/v1/webhooks/{id}/signing` | Deliveries are signed with HMAC-SHA256 as `sha256=<hex>`; see `docs/api/WEBHOOKS.md` |

---

### v2.4.0 (2026-01-25)

**No breaking changes.** Added new namespace resources.

**New Features:**
- Added resources: `orgs`, `tenants`, `policies`, `codebase`, `costs`, `decisions`, `onboarding`, `notifications`, `gmail`, `explainability`
- Fixed payloads for billing and RBAC endpoints

---

### v2.2.0 (2026-01-24)

**No breaking changes.** Version aligned with core package.

---

### v2.0.0 (2026-01-17)

#### Breaking Changes

| Change | Before | After | Migration |
|--------|--------|-------|-----------|
| API version default | v1 | v2 | Explicit `api_version="v1"` to keep old behavior |
| Method naming | Camel case methods | Namespace-based methods | `client.getDebates()` becomes `client.debates.list()` |
| Response format | Direct data | Wrapped in `data`/`meta` | Access via `response["data"]` |

#### Method Renames

| Old Method | New Method |
|------------|------------|
| `client.getDebates()` | `client.debates.list()` |
| `client.createDebate(...)` | `client.debates.create(...)` |
| `client.getDebate(id)` | `client.debates.get(id)` |
| `client.getAgents()` | `client.agents.list()` |
| `client.getAgent(name)` | `client.agents.get(name)` |
| `client.submitVote(...)` | `client.debates.vote(debate_id, ...)` |
| `client.getConsensus(id)` | `client.consensus.get(id)` |

#### Migration Example

```python
# Before (v1.x)
from aragora.client import AragoraClient

client = AragoraClient(base_url="https://api.aragora.ai")
debates = client.getDebates()
debate = client.createDebate(topic="Should we use GraphQL?", max_rounds=3)

# After (v2.x)
from aragora import AragoraClient

client = AragoraClient(
    base_url="https://api.aragora.ai",
    api_version="v2"  # Optional, v2 is now default
)
response = client.debates.list()
debates = response["data"]["debates"]

response = client.debates.create(
    task="Should we use GraphQL?",  # 'topic' renamed to 'task'
    rounds=3  # 'max_rounds' renamed to 'rounds'
)
debate = response["data"]
```

#### Response Format Change

```python
# Before (v1.x) - Direct data access
response = client.debates.list()
debates = response["debates"]
count = response["count"]

# After (v2.x) - Wrapped response
response = client.debates.list()
debates = response["data"]["debates"]
count = response["data"]["count"]
meta = response["meta"]  # Contains version, timestamp, request_id
```

---

## Version 1.x

### v1.0.0 (2026-01-14)

**Initial stable release.** No breaking changes from pre-1.0 beta versions.

---

## Upcoming Breaking Changes

### Scheduled for v3.0.0

No breaking changes currently scheduled.

---

## Migration Guides

- [API v1 to v2 Migration](../../docs/api/V1_TO_V2_MIGRATION.md) - Complete guide for API migration
- [SDK Guide](../../docs/SDK_GUIDE.md) - Full SDK documentation

---

## Deprecation Warnings

The SDK emits `DeprecationWarning` for deprecated methods. Enable warnings to see them:

```python
import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module="aragora")
```

Or run Python with `-W default`:

```bash
python -W default your_script.py
```

---

*Last updated: 2026-01-31*
