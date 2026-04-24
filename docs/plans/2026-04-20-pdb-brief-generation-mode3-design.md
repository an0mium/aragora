# PDB Brief Generation — Mode 3 (on-demand) Thinnest Viable Slice

Last updated: 2026-04-20
Status: design doc — ready for implementation mission

**Parent plan:** [`2026-04-19-pr-intelligence-brief.md`](2026-04-19-pr-intelligence-brief.md)
**Addendum:** [`2026-04-19-pr-intelligence-brief-addendum.md`](2026-04-19-pr-intelligence-brief-addendum.md)
**Tracking:** #6306 (PRReviewProtocol), #6307 (receipts), #6305 (cost/budget), #6304 (UI)

## Purpose

The review-queue UI (PDB UI v0) is live but the Brief sub-system that gives
it meaning is not. The addendum spec'd three timing modes:

- Mode 1 — webhook on PR-open (auto)
- Mode 2 — batched `review-queue build` (auto)
- Mode 3 — `review-queue brief <pr>` (on-demand)

Per the joint Claude+Codex recommendation, **Mode 3 is the right first slice**:
it's the smallest path that makes the UI real without requiring webhooks,
always-on workers, or queue orchestration. Once Mode 3 works end-to-end,
layering Mode 2 (batched) and then Mode 1 (webhook) is incremental.

## Non-goals for this slice

- Mode 1 webhook listener — deferred
- Mode 2 batched auto-generation — deferred (CLI `build` keeps listing PRs)
- Dual-synthesis check (Protocol C dual-synthesis) — deferred; only Protocol B single-synthesis in v0.3
- Adaptive escalation rules — deferred (flat Protocol B for every on-demand request)
- GLM / Ernie adapters — deferred
- Multi-tenancy, cost attribution across tenants — deferred
- Web UI refinements beyond state wiring — deferred

## Brief lifecycle state machine

One authoritative state per `(pr_number, head_sha)` pair. All UI and backend
code reads + writes through this enum.

```
absent → queued → running → ready ─┐
                        │           │
                        └ → failed  │
                                    │
                        ready ──stale── (when head_sha changes)
```

Semantics:

- **absent**: no brief exists; `.aragora/review-queue/briefs/pr-{n}-{sha}.json`
  is not on disk, no index entry for this `(pr, sha)`.
- **queued**: request accepted; work item on disk at
  `.aragora/review-queue/briefs/queued/pr-{n}-{sha}.json` with `requested_at`
  timestamp. Worker will pick up.
- **running**: worker has started. State file at
  `.aragora/review-queue/briefs/running/pr-{n}-{sha}.json` with
  `started_at`, `panel_models`, `current_phase` (findings_round |
  critique_round | synthesis).
- **ready**: final brief JSON at
  `.aragora/review-queue/briefs/pr-{n}-{sha}.json`, signed, indexed.
- **failed**: worker completed abnormally. Error record at
  `.aragora/review-queue/briefs/failed/pr-{n}-{sha}.json` with
  `error_message`, `failed_phase`, `cost_usd_so_far`.
- **stale**: brief exists but `head_sha` no longer matches current GitHub
  head. Moved to `.aragora/review-queue/briefs/invalidated/` with reason
  `head_advanced`.

Transitions are UNIDIRECTIONAL within a lifecycle. A `stale` brief on
a new head starts a fresh `absent → queued → running → ready` cycle.

## API shape

Backend endpoints (add to `aragora/server/handlers/review_queue.py`):

### `POST /api/v1/review-queue/prs/{number}/brief/generate`

Request body: empty, or optional `{ "protocol": "B" | "C" }` (default B).

Response 202 (Accepted):
```json
{
  "pr_number": 6328,
  "head_sha": "6a7dfc5e5135",
  "state": "queued",
  "queued_at": "2026-04-20T21:00:00Z",
  "estimated_completion_seconds": 180
}
```

Response 409 if a brief is already `running`, `queued`, or `ready` for that
head_sha. Body indicates current state so caller can subscribe to status.

### `GET /api/v1/review-queue/prs/{number}/brief/state`

Returns the lifecycle state without the brief content.

```json
{
  "pr_number": 6328,
  "head_sha": "6a7dfc5e5135",
  "state": "running",
  "current_phase": "critique_round",
  "started_at": "2026-04-20T21:00:15Z",
  "cost_usd_so_far": 2.41
}
```

### `GET /api/v1/review-queue/prs/{number}/brief` (existing)

Unchanged endpoint. Returns the full Brief JSON when state is `ready`;
404 otherwise. (Extended: `Retry-After` header when state is `queued` or
`running`.)

### `DELETE /api/v1/review-queue/prs/{number}/brief/generate`

Cancels a queued or running generation. No-op if state is already
`ready`, `failed`, or `absent`. Returns the pre-cancel state.

## Job model

### Worker: simple foreground thread pool

For Mode 3 v0.3, the worker runs **in-process** inside `aragora serve`.
A `BriefGenerationWorker` singleton with:

- A bounded queue (max 5 concurrent generations, configurable via
  `ARAGORA_PDB_MAX_CONCURRENT_BRIEFS`)
- Each request spawns one `asyncio.Task`
- Cancellation propagates via `Task.cancel()` on DELETE
- Graceful shutdown on server stop: finish running tasks, reject new queued

No separate worker process, no Redis queue, no Celery. That's Mode 1/2
infrastructure. Mode 3 is in-process only — which is fine because the
founder is the only user in Phase A.

### Single generation flow (Protocol B only)

1. Load PR packet from GitHub (`gh pr view` + `gh pr diff`) — ~3s
2. For each of 8 panel models in parallel: issue findings prompt — ~20–40s
3. For each of 8 panel models in parallel: issue critique prompt with
   round-1 findings context — ~20–40s
4. Synthesizer (`pdb.core.claude`) consumes 16 findings, produces
   role-structured brief — ~30s
5. Sign artifact (ed25519) and write to `.aragora/review-queue/briefs/`
6. Update index + ledger (`pdb_brief_generated` event)

Total wall clock: ~90–120s for Protocol B. Budget: 16 + 1 = 17 LLM calls,
approximately $5–8 per brief at current frontier pricing.

## Storage paths

```
.aragora/review-queue/briefs/
├── pr-6328-6a7dfc5e5135.json       (ready)
├── queued/
│   └── pr-6329-abc123def456.json   (queued — has requested_at, expected_panel_models)
├── running/
│   └── pr-6330-def456abc789.json   (running — has started_at, current_phase)
├── failed/
│   └── pr-6331-789abcdef012.json   (failed — has error_message, failed_phase, cost_usd_so_far)
├── invalidated/
│   └── pr-6332-oldabc123456.json   (moved here when head_sha changes)
└── index.jsonl                     (append-only index of all state transitions)
```

Lifecycle state is determined by disk presence — no DB. Simpler, transparent,
matches rest of repo's flat-file patterns.

## UI state changes

### `useReviewQueue` hook additions

New method `generateBrief(pr_number: number, protocol?: 'B' | 'C'): Promise<{ state, estimated_completion_seconds }>`
wraps `POST /api/v1/review-queue/prs/{n}/brief/generate`.

New method `getBriefState(pr_number: number): Promise<{ state, current_phase?, cost_usd_so_far? }>`
wraps `GET /api/v1/review-queue/prs/{n}/brief/state`.

### `ReviewQueueCard.handleApprove` three-way dialog

Replace the browser `window.confirm` with a proper modal component
(`ApproveDecisionModal`) that exposes:

- **Approve** — settles immediately, same as current behavior
- **Generate brief** — calls `generateBrief()`, dismisses modal, card
  shows `state=queued` badge with spinner; auto-expands when `ready`;
  user can re-open the Approve flow after brief exists
- **Cancel** — closes modal, no side effect

### `BriefPanel` state-aware rendering

```
state=absent    →  "No brief generated yet. [Generate brief] button."
state=queued    →  "Brief queued — waiting for worker slot. [Cancel]"
state=running   →  "Brief in progress — phase: <current_phase>. Cost so far: $X.XX. [Cancel]"
state=ready     →  existing role-structured rendering
state=failed    →  "Brief generation failed at <failed_phase>: <error_message>. [Retry]"
state=stale     →  "Brief is stale (head advanced). [Regenerate]"
```

### `ReviewQueueCard` badge tint while generating

Badge background pulses softly (`animation: pulse 2s ease-in-out infinite`)
while state is `queued` or `running`. Badge tone follows verdict once `ready`.

## Config + flags

```yaml
# aragora/config/pdb_panel.yaml (new file)
panel:
  core:
    - slot: pdb.core.claude
      provider: anthropic
      model: claude-opus-4-7
    - slot: pdb.core.gpt
      provider: openai
      model: gpt-5-3
  heterodox:
    - slot: pdb.heterodox.gemini
      provider: gemini
      model: gemini-2-5-pro
    - slot: pdb.heterodox.grok
      provider: grok
      model: grok-4
    - slot: pdb.heterodox.deepseek
      provider: openrouter
      model: deepseek/deepseek-chat-v3-0324
    - slot: pdb.heterodox.kimi
      provider: openrouter
      model: moonshotai/kimi-k2-0905
    - slot: pdb.heterodox.qwen
      provider: openrouter
      model: qwen/qwen3-max
  regulatory:
    - slot: pdb.regulatory.mistral
      provider: mistral
      model: mistral-large-2512
synthesizer_slot: pdb.core.claude

budgets:
  per_brief_max_usd: 20.0
  per_day_max_usd: 200.0

worker:
  max_concurrent_briefs: 5
  findings_timeout_s: 90
  synthesis_timeout_s: 120
```

Feature flag `ARAGORA_PDB_BRIEF_GENERATION_ENABLED=1` — default OFF. With
flag off, the `generate` endpoint returns 503 and the Generate brief button
is hidden (renders as a disabled "Coming soon" state).

## Rollout order

**Revision (2026-04-20 evening, post-Codex review):** The original plan
bundled budget enforcement + panel config into Step 1. Codex correctly
noted those only matter once the executor is calling models, and that
a tighter PR 1 scope is more reviewable. Revised below. The cumulative
work is unchanged; only the PR boundaries shifted.

### PR 1 — storage + state machine only (~1.5 days)

Files:
- `aragora/pdb/__init__.py`
- `aragora/pdb/brief_state.py` — `BriefLifecycleState` enum + transitions
- `aragora/pdb/storage.py` — read/write to `.aragora/review-queue/briefs/` and subdirs
- `aragora/server/handlers/review_queue.py` — refactor the existing
  `/brief` read path to use the new storage layer (pure refactor, no
  behavior change; proves the abstraction works with real code)
- Tests: `tests/pdb/test_brief_state.py`, `tests/pdb/test_storage.py`,
  existing `tests/server/handlers/test_review_queue.py` continues to pass

Acceptance: existing page doesn't regress; all 6 states representable
on disk; stale detection moves ready briefs to `invalidated/` on SHA
change; writes are atomic via `os.replace`.

No execution, no endpoints, no UI, no budget, no panel config.

### PR 2 — Protocol B executor + panel config + budget (~3 days)

- `aragora/pdb/panel_config.py` — loads `aragora/config/pdb_panel.yaml`,
  resolves slots to provider+model
- `aragora/config/pdb_panel.yaml` — default panel config
- `aragora/pdb/budget.py` — per-brief + per-day budget enforcement,
  UTC-midnight rollover
- `aragora/pdb/protocol.py` — `PRReviewProtocol.run()` that orchestrates
  findings round + critique round + synthesis, uses existing agent
  adapters from `aragora/agents/api_agents/`, signs output via
  `DurableFileSigner`, writes ledger event
- Tests: `tests/pdb/test_panel_config.py`, `test_budget.py`,
  `test_protocol.py` (with mocked agents)

Budget + panel config land here because this is the first PR where
model calls actually happen. No endpoints yet.

### PR 3 — backend endpoints + in-process worker (~2 days)

- `aragora/server/handlers/review_queue_brief.py` — the 4 new endpoints
  (generate, state, cancel, get) layered on top of existing handler
- In-process `BriefGenerationWorker` with `asyncio.Semaphore(max_concurrent)`
- Tests: `tests/server/handlers/test_review_queue_brief.py` with fake protocol

### PR 4 — UI integration + polish (~3 days, folds Step 5 into UI PR)

- `useReviewQueue` hook adds `generateBrief`, `getBriefState`, polling helpers
- `ApproveDecisionModal` component with 3 options
- `BriefPanel` state-aware rendering for all 6 states
- `ReviewQueueCard` badge pulse animation while generating
- Tests: `aragora/live/__tests__/ApproveDecisionModal.test.tsx`

*(Step 5 polish folded into PR 4: user guide, feature-flag instructions,
screenshot, end-to-end dogfood verification all land with the UI PR.)*

**Total: ~9.5 days focused engineering time** for Mode 3 v0.3 across
4 reviewable PRs.

**Branch naming convention** (per Codex's recommendation):
- PR 1: `codex/pdb-mode3-storage`
- PR 2: `codex/pdb-mode3-executor`
- PR 3: `codex/pdb-mode3-endpoints`
- PR 4: `codex/pdb-mode3-ui`

## Open questions

- **Cancel guarantees** — if user clicks Cancel while in round-2 critique, do
  we kill all 8 in-flight calls cleanly? Proposal: yes, `Task.cancel()` each,
  record partial cost in failed record.
- **Budget exhaustion mid-run** — if we breach `per_day_max_usd` at
  synthesis, do we abort or honor already-spent compute? Proposal:
  abort at phase boundary, NOT mid-phase. Record as `failed` with
  reason `budget_exhausted_before_phase=synthesis`.
- **Rate limits** — per-provider rate limits differ. Should the protocol
  retry with exponential backoff, or fail fast? Proposal: backoff
  5s→15s→45s, then fail with `rate_limit_exhausted`.
- **Temperature and system prompts** — each lens (core/heterodox/regulatory)
  may benefit from lens-specific system prompt framing. Defer to Step 2
  implementation; start with neutral prompts and iterate.
- **Stale detection** — should we periodically scan all `ready` briefs for
  SHA staleness, or only check when the user opens a card? Proposal:
  check on card-open only. Not worth a background sweep.

## Success signal

Step 5 complete + flag enabled →

- Click Generate brief on an open PR
- See state transition `absent → queued → running → ready` in the UI within
  ~2 minutes
- Brief appears in BriefPanel with logic/security/maintainability/skeptic
  sections populated from real LLM output
- Dissent-by-lens visible in the synthesis
- Signed artifact on disk at `.aragora/review-queue/briefs/pr-N-HEAD.json`
- Ledger event `pdb_brief_generated` written
- Approve button no longer fires the "no brief on file" warning

Once one brief works end-to-end, the core PDB product promise is validated.
Mode 2 (batched) and Mode 1 (webhook) become scaling problems, not
existence problems.
