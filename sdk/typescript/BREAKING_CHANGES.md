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
| `debates.findSimilar(debateId, limit)` | `GET /api/v1/debates/{id}/similar` | `debates.search(query)` or `debates.compare(debateIds)` |
| `debates.analyzeArgumentQuality(debateId)` | `GET /api/v1/debates/{id}/quality` | `debates.getSummary(debateId)` |
| `debates.getNotes(debateId)` | `GET /api/v1/debates/{id}/notes` | No replacement (server has no notes feature) |
| `debates.addNote(debateId, note)` | `POST /api/v1/debates/{id}/notes` | No replacement (server has no notes feature) |
| `debates.deleteNote(debateId, noteId)` | `DELETE /api/v1/debates/{id}/notes/{noteId}` | No replacement (server has no notes feature) |
| `debates.getBatchResults(batchId)` | `GET /api/v1/debates/batch/{id}/results` | `debates.getBatchStatus(batchId)` (includes per-job state) |
| `debates.cancelBatch(batchId)` | `POST /api/v1/debates/batch/{id}/cancel` | Cancel individual debates with `debates.cancel(debateId)` |
| `debates.retryBatch(batchId, options)` | `POST /api/v1/debates/batch/{id}/retry` | Re-submit failed debates with `debates.submitBatch(...)` |
| `debates.getDebateAgentStatistics(debateId)` | `GET /api/v1/debates/{id}/agent-statistics` | `debates.getStatsAgents()` (aggregate per-agent statistics) |
| `debates.getHealth(debateId)` | `GET /api/v1/debates/{id}/health` | `debates.listHealth()` for system-wide debate health |

---

## Migration Guides

- [API v1 to v2 Migration](../../docs/api/V1_TO_V2_MIGRATION.md) - Complete guide for API migration
- [SDK Guide](../../docs/SDK_GUIDE.md) - Full SDK documentation

---
