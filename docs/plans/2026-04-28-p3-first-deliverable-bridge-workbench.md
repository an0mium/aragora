# P3 (DAG / GUI) First Deliverable: Agent Bridge Run Inspector

**Status:** spec only — text proposal for operator review.
**Author:** droid (Factory) overnight 2026-04-28.
**Scope:** narrow workbench surface for read-only inspection of agent bridge runs landed in #6420 (read-only UI), #6465 (handler wiring), and #6629 (write-gated ops plane).
**Non-scope:** no implementation overnight, no PR. Stop at spec.

## Why this exists

The 2026-04-21..28 reassessment surfaced that Pillar 3 (Unified DAG and Optional Interactivity) received **0.4%** of merged PRs in the 7-day window — the lowest of any pillar. Only one PR, #6612 (canvas transition approval), was workbench-adjacent. Meanwhile the **agent bridge** stack landed five substantive PRs in the same window:

- #6392 backend core on PR 2a schema
- #6407 read-only HTTP API on PR 2b
- #6420 read-only autonomous UI on PR 2c
- #6465 handler wired into registry and autonomous nav
- #6629 write-gated ops plane

But a verification grep of `aragora/live/src/` (the Next.js frontend) returns **zero matches** for `agent_bridge` or `AgentBridge`. The new ops plane has no workbench surface for operators to:
- See live and historical agent bridge runs
- Inspect a single run's turn-by-turn transcript
- View per-run footers (verdicts, evidence links, claim refs)
- Trigger a write-gated dispatch with operator approval

This is a textbook case of "the GUI and DAG are not decoration" (CANONICAL_GOALS.md):
> A beautiful GUI that is not backed by the same contracts and ledger is an anti-goal.

The complement is also true: a backend with no workbench surface erodes the human-legibility promise of Pillar 3. This spec proposes the **smallest first deliverable** that closes that erosion.

## Surface in scope

Today's read-only HTTP API:
- `GET /api/agent-bridge/runs` — list of runs (paginated, etag-aware)
- `GET /api/agent-bridge/runs/{run_id}` — single run with turns + footer
- handlers in `aragora/server/handlers/agent_bridge.py` (742 lines)
- types in `aragora/swarm/agent_bridge/types.py`
- store in `aragora/swarm/agent_bridge/store.py`

Today's write-gated HTTP API (per #6629):
- `POST /api/agent-bridge/runs` — operator-gated dispatch (RBAC: `agent_bridge:write`)

Today's frontend: `aragora/live/src/app/`, `aragora/live/src/components/`, `aragora/live/src/hooks/` — zero references to agent_bridge.

## The first deliverable: Bridge Run Inspector page

**Single new route** in the Next.js app: `/agent-bridge/runs` and `/agent-bridge/runs/[id]`, **read-only**, no write actions in the first cut.

### Page 1 — Runs list (`/agent-bridge/runs`)

Server-rendered React page that hits `GET /api/agent-bridge/runs`. Renders:
- Table: `run_id`, `started_at`, `harness` (claude / codex / droid), `agent_name`, `turn_count`, `footer.verdict`
- Filter by harness, by date range, by verdict
- Pagination via cursor
- Click a row → navigate to detail page

### Page 2 — Run detail (`/agent-bridge/runs/[id]`)

Server-rendered detail page that hits `GET /api/agent-bridge/runs/{id}`. Renders:
- **Header**: run_id, harness, agent_name, started_at, ended_at, turn_count
- **Turn-by-turn transcript**: `TurnRecord` list with role, content, timestamp; collapsible
- **Footer panel**: `BridgeFooter` fields — verdict, evidence_links, claim_refs, signature
- **Receipt link**: if a footer's verdict points to a decision receipt, link to it (this is the integration seam with Worker A's P8 audit recommendation — receipts as agent-readable JSON)

### Out of scope for the first cut
- Write actions (operator-gated dispatch button) — second cut, after RBAC binding is exercised in production
- Live streaming (websocket follow) — third cut
- Multi-run comparison views — third cut
- Search across content — third cut, requires backend search

## Why this is the right scope

Per the "narrow first deliverable" pattern:
- ~400 LOC of TypeScript / React in `aragora/live/src/app/agent-bridge/`
- Reuses existing read-only HTTP API; no backend changes needed
- Reuses existing Next.js layout / navigation patterns from `aragora/live/src/app/`
- No new dependencies
- No conflict with Codex's automation lane or Claude's docs port lane

It also unlocks the **P8 (agents-as-consumers) recommendation** from Worker A's audit: when the agent-readable receipt endpoint lands, the run-detail page can link directly to it, demonstrating parity between agent-form and human-form receipt access from the same UI.

## Suggested file layout

```
aragora/live/src/app/agent-bridge/
  layout.tsx          # shared chrome + breadcrumbs
  runs/
    page.tsx          # runs list (Page 1)
    [id]/
      page.tsx        # run detail (Page 2)
aragora/live/src/components/agent-bridge/
  RunsTable.tsx
  RunDetailHeader.tsx
  TurnRecordCard.tsx
  FooterPanel.tsx
aragora/live/src/lib/api/agent-bridge.ts   # typed fetcher
aragora/live/__tests__/agent-bridge/
  runs-list.test.tsx
  run-detail.test.tsx
```

Total: ~10 new files, ~400 LOC of TypeScript + React, ~150 LOC of tests. All additive.

## Sequencing suggestion

If the operator approves implementation:
1. PR 1 (this spec): typed API client + RunsTable component (~150 LOC, with mock data tests). No new routes yet.
2. PR 2: `/agent-bridge/runs` route + RunsTable wired (~120 LOC).
3. PR 3: `/agent-bridge/runs/[id]` route + detail components (~150 LOC).
4. PR 4 (later, gated on P8 receipt-endpoint landing): add the link-to-agent-receipt CTA (~30 LOC).

Each PR opens with auto-merge OFF, narrow scope, no regression risk.

## What this spec does NOT solve

- **Write-gated dispatch UI**: deferred until the read-only inspector is in operator hands long enough to validate the data model.
- **Live streaming**: would require a new websocket subscription on the backend (separate spec).
- **Multi-run comparison**: would require new backend query primitives.
- **Cross-pillar GUI for canvas, debate, or knowledge mound**: out of scope; this is a single bounded P3 deliverable.

## Stop conditions

This spec is finalized when an operator approves it. It is text only. No implementation initiative is requested. If rejected, the existing zero-frontend state continues without harm — the read-only HTTP API is fully usable via curl or MCP today.

---

*End of spec. No PR is filed by this document.*
