# TypeScript SDK Breaking Changes

This document tracks breaking changes specific to the Aragora TypeScript SDK. For core API breaking changes, see the main [BREAKING_CHANGES.md](../../docs/BREAKING_CHANGES.md).

---

## Version 2.x

### Unreleased (2026-09-03)

#### Breaking Changes

Removed 12 `debates` methods from `DebatesAPI` that targeted routes no server
handler dispatches. Every one of them was already marked `@deprecated`; each
call fell through to the debate slug lookup and returned 404, so no working
integration depended on them. The response interfaces used only by those
methods (`DebateHealthDetail`, `BatchResults`, `ArgumentQualityAnalysis`,
`DebateNote`) were removed with them.

| Removed Method | Route | Migration |
|----------------|-------|-----------|
| `debates.restore(debateId)` | `POST /api/v1/debates/{id}/restore` | `debates.update(debateId, { status: 'active' })` |
| `debates.makePermanent(debateId)` | `POST /api/v1/debates/{id}/make-permanent` | None needed; completed debates persist automatically |
| `debates.findSimilar(debateId, limit)` | `GET /api/v1/debates/{id}/similar` | `consensus.findSimilar({ topic, limit })` (`GET /api/consensus/similar`) with the debate task text as `topic` |
| `debates.analyzeArgumentQuality(debateId)` | `GET /api/v1/debates/{id}/quality` | `debates.getSummary(debateId)` |
| `debates.getNotes(debateId)` | `GET /api/v1/debates/{id}/notes` | No replacement (server has no notes feature) |
| `debates.addNote(debateId, note)` | `POST /api/v1/debates/{id}/notes` | No replacement (server has no notes feature) |
| `debates.deleteNote(debateId, noteId)` | `DELETE /api/v1/debates/{id}/notes/{noteId}` | No replacement (server has no notes feature) |
| `debates.getBatchResults(batchId)` | `GET /api/v1/debates/batch/{id}/results` | `debates.getBatchStatus(batchId)` (includes per-job state) |
| `debates.cancelBatch(batchId)` | `POST /api/v1/debates/batch/{id}/cancel` | Cancel individual debates with `debates.cancel(debateId)` |
| `debates.retryBatch(batchId, options)` | `POST /api/v1/debates/batch/{id}/retry` | Re-submit failed debates with `debates.submitBatch(...)` |
| `debates.getDebateAgentStatistics(debateId)` | `GET /api/v1/debates/{id}/agent-statistics` | `debates.getStatsAgents()` (aggregate per-agent statistics) |
| `debates.getHealth(debateId)` | `GET /api/v1/debates/{id}/health` | `debates.listHealth()` for system-wide debate health |

Removed 10 `webhooks` methods from `WebhooksAPI` that targeted per-webhook
sub-routes no server handler dispatches. The webhook handler only routes
`/api/v1/webhooks/{id}` and `/api/v1/webhooks/{id}/test` below a webhook id,
so every one of these calls returned 404 and no working integration depended
on them. The interfaces used only by those methods (`WebhookDeliveryAttempt`,
`WebhookRetryPolicy`) were removed with them.

| Removed Method | Route | Migration |
|----------------|-------|-----------|
| `webhooks.listDeliveries(webhookId, options)` | `GET /api/v1/webhooks/{id}/deliveries` | `webhooks.listDeadLetter()` for failed deliveries; successful deliveries are not listed |
| `webhooks.getDelivery(webhookId, deliveryId)` | `GET /api/v1/webhooks/{id}/deliveries/{deliveryId}` | `webhooks.getDeadLetter(id)` for a failed delivery; successful deliveries are not exposed individually |
| `webhooks.retryDelivery(webhookId, deliveryId)` | `POST /api/v1/webhooks/{id}/deliveries/{deliveryId}/retry` | `webhooks.retryDeadLetter(id)` |
| `webhooks.getDeliveryStats(webhookId, options)` | `GET /api/v1/webhooks/{id}/stats` | No replacement; per-webhook delivery statistics are not served (`webhooks.getQueueStats()` reports the shared delivery queue, not a single webhook) |
| `webhooks.subscribeEvents(webhookId, events)` | `POST /api/v1/webhooks/{id}/events` | `webhooks.update(webhookId, { events })` with the full event list |
| `webhooks.unsubscribeEvents(webhookId, events)` | `DELETE /api/v1/webhooks/{id}/events` | `webhooks.update(webhookId, { events })` with the remaining events |
| `webhooks.getRetryPolicy(webhookId)` | `GET /api/v1/webhooks/{id}/retry-policy` | No replacement; the retry policy is server-configured, not per webhook |
| `webhooks.updateRetryPolicy(webhookId, policy)` | `PUT /api/v1/webhooks/{id}/retry-policy` | No replacement; the retry policy is server-configured, not per webhook |
| `webhooks.rotateSecret(webhookId)` | `POST /api/v1/webhooks/{id}/rotate-secret` | `webhooks.delete(webhookId)` then `webhooks.create(...)`; the secret is returned once on creation |
| `webhooks.getSigningInfo(webhookId)` | `GET /api/v1/webhooks/{id}/signing` | Deliveries are signed with HMAC-SHA256 as `sha256=<hex>`; see `docs/api/WEBHOOKS.md` |

Removed 10 `admin` methods from `AdminAPI` that targeted routes no server
handler dispatches. Below an organization or user id the admin handler only
serves the `/activate`, `/deactivate` and `/unlock` user actions; impersonation
is served as `POST /api/v1/admin/impersonate/{userId}` and credits under
`/api/v1/admin/credits/{orgId}/...`. Every call below returned 405 from that
handler, so no working integration depended on them. The
`AdminClientInterface` entries used only by those methods (`getCreditAccount`,
`listCreditTransactions`, `adjustCreditBalance`, `getExpiringCredits`) were
removed with them. The legacy flat methods on `AragoraClient` that target the
same routes (`getAdminOrganization`, `updateAdminOrganization`, `getAdminUser`,
`suspendAdminUser`, `impersonateUser`, `issueCredits`, `getCreditAccount`,
`listCreditTransactions`, `adjustCreditBalance`, `getExpiringCredits`) are
unchanged here and still return 405; do not migrate to them.

| Removed Method | Route | Migration |
|----------------|-------|-----------|
| `admin.getOrganization(orgId)` | `GET /api/v1/admin/organizations/{id}` | `organizations.get(orgId)` (`GET /api/v1/org/{id}`) |
| `admin.updateOrganization(orgId, updates)` | `PUT /api/v1/admin/organizations/{id}` | `organizations.update(orgId, { name, settings })` (`PUT /api/v1/org/{id}`, served but not yet in the spec; org-admin scoped and accepts only `name` and `settings`) |
| `admin.getUser(userId)` | `GET /api/v1/admin/users/{id}` | `admin.listUsers(...)` and filter by id |
| `admin.suspendUser(userId, reason)` | `POST /api/v1/admin/users/{id}/suspend` | `admin.deactivateUser(userId)` |
| `admin.impersonateUser(userId)` | `POST /api/v1/admin/users/{id}/impersonate` | `openapi.requestPostApiV1AdminImpersonateByUserId(userId)` (`POST /api/v1/admin/impersonate/{userId}`) |
| `admin.issueCredits(orgId, amount, reason, expiresAt)` | `POST /api/v1/admin/organizations/{id}/credits` | ``client.request('POST', `/api/v1/admin/credits/${orgId}/issue`, { body })`` |
| `admin.adjustCredits(orgId, amount, reason)` | `POST /api/v1/admin/organizations/{id}/credits` | ``client.request('POST', `/api/v1/admin/credits/${orgId}/adjust`, { body })`` |
| `admin.getCreditAccount(orgId)` | `GET /api/v1/admin/organizations/{id}/credits` | ``client.request('GET', `/api/v1/admin/credits/${orgId}`)`` |
| `admin.listCreditTransactions(orgId, params)` | `GET /api/v1/admin/organizations/{id}/credits/transactions` | ``client.request('GET', `/api/v1/admin/credits/${orgId}/transactions`, { params })`` |
| `admin.getExpiringCredits(orgId)` | `GET /api/v1/admin/organizations/{id}/credits/expiring` | ``client.request('GET', `/api/v1/admin/credits/${orgId}/expiring`, { params: { within_days: 30 } })`` (`within_days` 1-365, default 30) |

---

## Migration Guides

- [API v1 to v2 Migration](../../docs/api/V1_TO_V2_MIGRATION.md) - Complete guide for API migration
- [SDK Guide](../../docs/SDK_GUIDE.md) - Full SDK documentation

---
