# Agent-Dispatch Reach Plan — Progress Snapshot

Generated: 2026-05-19T04:25:00Z by claude-B061F80D

Tracks the 5-phase reach plan introduced in PR #7327 (P54), which closes four "weak / does-not-exist-as-advertised" capability gaps in the aragora agent-dispatch surface:

1. Reach **INTO** a Droid CLI session running in a Factory.ai web-CLI or local terminal that aragora did NOT launch.
2. Reach **INTO** a Codex Desktop tab running in the macOS Codex.app, whose IPC is private to the Electron process.
3. **Direct programmatic dispatch to "all currently-active agents"** including ones not registered via `agent_bridge.py launch`.
4. **One-button wake-up** for any agent — operator types one CLI command and the right agent gets the right prompt via the right backend.

## Phase status

| Phase | Lane | PR | State | Description |
|---|---|---|---|---|
| Plan | P54-agent-dispatch-reach-plan | [#7327](https://github.com/synaptent/aragora/pull/7327) | draft | 5-phase plan in `docs/governance/AGENT_DISPATCH_REACH_PLAN.md` |
| R01 | R01-reach-plan-contact-method-field | [#7336](https://github.com/synaptent/aragora/pull/7336) | draft | `contact_method` + `contact_payload` fields on `LaneRecord`; tmux auto-detect |
| R02 | R02-wake-agent-cli | [#7348](https://github.com/synaptent/aragora/pull/7348) | draft | `scripts/wake_agent.sh` unified dispatch CLI |
| R03 | (not yet claimed) | — | — | `scripts/codex_desktop_inject.sh` — see [research finding on #7327](https://github.com/synaptent/aragora/pull/7327#issuecomment-4481751598) |
| R04a | (not yet claimed) | — | — | Droid local-tmux reach (uses existing tmux backend) |
| R04b | (not yet claimed) | — | — | `scripts/droid_inbox_poller.py` polling sidecar |
| R05 | R05-sweep-lane-contact-methods | [#7351](https://github.com/synaptent/aragora/pull/7351) | draft | `scripts/sweep_lane_contact_methods.py` backfills `contact_method` on existing lanes |

## Architectural summary

```
operator/agent → scripts/wake_agent.sh --lane <id>
                          │
                          ↓
                identify_lane_owner.py (Phase A, on main)
                          │
                          ↓
                        switch on contact_method (R01)
                          │
        ┌─────────────────┼─────────────────┬──────────────────┐
        ↓                 ↓                 ↓                  ↓
    tmux:NAME       mailbox-only       osascript:*       factory-api:*
    (R02 ✓)         (R02 ✓)          (R03 future)       (R04b opt-in)
        │                 │                 │                  │
        ↓                 ↓                 ↓                  ↓
  tmux_send_prompt.sh  send_operator    codex_desktop      factory_api_send.py
   (already shipped)    _steering.py     _inject.sh         (R04b future)
                       (Phase B, #7310)  (R03 future)

  Every dispatch writes .aragora/dispatch-receipts/<utc>-<lane>-<sha8>.json
```

## What's mergeable today

- **R01 (#7336)** — schema-additive field on LaneRecord. Independent. 19 tests passing.
- **R02 (#7348)** — wake_agent.sh ships independently of R01 via graceful degradation (missing contact_method → mailbox-only fallback). 24 tests passing.
- **R05 (#7351)** — sweep_lane_contact_methods.py ships independently of R01 in `--dry-run` mode (works today against current registry). `--apply` mode gates on R01 landing. 12 tests passing.

## Discovery worth reviewing before R03 starts

**Codex Desktop does NOT expose a localhost HTTP API.** But it DOES ship an official IPC layer via `codex app-server` with subcommands:
- `daemon { enable-remote-control | disable-remote-control | start | stop }`
- `proxy --sock <PATH>` — proxies stdio bytes to the running control socket
- `generate-json-schema --out <DIR>` — emits the protocol schema (39 message types including `thread/inject_items`)

This makes R03 dramatically simpler than the original plan assumed: the osascript / Accessibility-API path becomes a fallback rather than the primary mechanism. The primary R03 path:

1. Operator one-time bootstrap: `codex app-server daemon enable-remote-control`
2. R03 wraps `codex app-server proxy --sock <PATH>` and pipes a JSON-RPC payload
3. Schema is officially generated, not reverse-engineered

Full research findings: [#7327 comment](https://github.com/synaptent/aragora/pull/7327#issuecomment-4481751598)

## Suggested merge order (informational)

1. **#7336 (R01)** first — unblocks R05 `--apply` mode and lets new lanes auto-populate
2. **#7348 (R02)** any time — independent
3. **#7351 (R05)** after R01 — sweeper can then backfill the existing 75-row registry
4. **#7327 (P54)** — review the plan + my R03 research finding before claiming R03

## Adjacent shipped this session

- **#7349 (H02)** — `scripts/refresh_proof_surfaces.sh` — closes the recurring TW03 staleness loop codex flagged.
- **#7337 (H01)** — orphan `<<<<<<< HEAD` marker removal from journal.
- Direct-to-main: TW03 freshness restore (`bb39676048`), PR queue triage (`c87c255fe7`).

## Cross-references

- Plan PR: [#7327](https://github.com/synaptent/aragora/pull/7327)
- Agent-steering primitive chain (Phase A/B/C/D/E — separate but adjacent track): [#7308 merged], [#7310 ready], [#7311 ready], P52/D deferred, [#7328 P53/E draft]
- Operator-snapshot integration via Phase C surfaces this whole track in operator queries
