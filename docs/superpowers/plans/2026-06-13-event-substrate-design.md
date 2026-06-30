# Event Substrate over Polling — Design Spec

Status: design spec (medium-term item 1 of epic [#8344](https://github.com/synaptent/aragora/issues/8344),
§5.1 of `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md`).
Written 2026-06-13. Implementable; names the components, the data flow, and a
phased rollout. Tier of the eventual build: per-component (mostly Tier 2; the
witness/credential pieces inherit the Tier-2 build / Tier-4 identity split from
`docs/specs/TAMPER_EVIDENT_TRAIL.md`).

## 1. Problem: polling is the quota-starvation root cause

This retires **failure class E** (quota starvation as architecture) at the
root. The program doc and epic frame it precisely: one GitHub identity serves N
concurrent pollers, each re-verifying full PR state every cycle. The "do not
trust transcript state" doctrine is correct, but applied per-lane it multiplies
GraphQL reads by fleet size and exhausts the 5,000-pts/hr budget mid-hour while
REST sits healthy.

Three layers already ameliorate this without removing the polls:

- `scripts/pr_state_cache.py` (#8339): ONE budgeted REST-only poller maintains
  `.aragora/pr-state-cache.json`; lanes read the cache and spend exactly one
  targeted REST `verify` call immediately before any mutation (the binding
  **reader contract** documented in that module's header).
- ETag/conditional requests (#8339/#8324): 304s are free, so the single poller
  re-lists cheaply.
- App-token routing (operator runbook): the GitHub App installation token has
  its own rate budget, separate from `an0mium`.

These shrink the cost of polling. They do **not** remove the structural fact
that state is *pulled* on a clock. The event substrate inverts that: GitHub
**pushes** state changes to a local collector, which updates the same
`.aragora/pr-state-cache.json` that #8339 already ships. The single poller
degrades from "the source of truth" to "a periodic reconcile backstop."

## 2. Architecture: webhook → collector → shared cache

```
   GitHub org/repo                  local collector                lanes
   ┌───────────────┐   webhook   ┌──────────────────────┐  reads  ┌─────────┐
   │ pull_request  │──POST────▶  │ event-cache-collector │◀────────│ lane A  │
   │ check_run     │  (HMAC)     │  (append-only ingest) │  read   │ lane B  │
   │ issue_comment │             │   ↓ atomic write      │◀────────│ lane C  │
   │ status        │             │ .aragora/pr-state-    │  read   │  ...    │
   └───────────────┘             │   cache.json          │         └─────────┘
          │                      │ .aragora/trail/       │              │
          │  (also)              │   webhook-events.jsonl│ verify(1 REST│ before
          ▼                      └──────────────────────┘  exact head) │ mutate
   external witness                                                     ▼
   (TET intent reconciliation)                                      GitHub API
```

The cache file format and reader contract are **unchanged** from #8339 — that
is the whole point. Lanes already speak `pr_state_cache.py read` / `verify`;
the substrate only changes *who writes the cache* (pushed events instead of a
clock-driven poll), not how lanes consume it. This is what makes the migration
low-risk: the consumer side never moves.

## 3. Webhook events to subscribe

Org-level webhook (or repo-level for `synaptent/aragora` first), delivering to
the collector endpoint. Subscribe to exactly the event types that move the
funnel state the cache tracks:

| Event | Why the cache needs it | Cache field updated |
|---|---|---|
| `pull_request` | open/close/reopen, ready-for-review, head-SHA moves (`synchronize`), merge | PR entry create/retire, `state`, `head_sha`, `draft`, `merged` |
| `check_run` | per-check conclusion/status transitions (the funnel's quorum-green signal) | per-PR `{check_name: conclusion-or-status}`, `checks_fetched_at` |
| `status` | commit-status contexts incl. `aragora/human-settlement` and `aragora-merge-quorum` | status contexts on the head SHA |
| `issue_comment` | operator steering comments, Tier-4 human-preapproval comments, supersession hints | comment-derived steering/settlement hints |

Deliberately **not** subscribed (out of cache scope, and `check_suite` would
duplicate `check_run`): `push` to non-PR refs, `workflow_run` internals,
`deployment`. The witness role (§6) widens coverage separately via the
Enterprise audit stream, which sees admin events webhooks never expose.

Delivery hardening:

- HMAC-SHA256 signature verification (`X-Hub-Signature-256`) on every POST; the
  shared secret lives only in the collector's environment.
- `X-GitHub-Delivery` id recorded per event for dedup and replay.
- Per-event ordering is not assumed — the collector reconciles by
  `(pr_number, head_sha, event timestamp)` and never lets an older event
  overwrite a newer head (monotonic-by-timestamp merge).

## 4. The local collector (append-only)

New script: `scripts/event_cache_collector.py`. Stdlib-only by design (mirrors
`pr_state_cache.py`), so it runs anywhere the receiving host can reach. Two
responsibilities, cleanly separated:

1. **Ingest (append-only).** Every verified webhook is appended verbatim to
   `.aragora/trail/webhook-events.jsonl` (one JSON object per line, with the
   delivery id, received-at, and HMAC-verified flag). This file is never
   rewritten — it is the local working copy of the event log and the seed of
   the witness role (§6).
2. **Project (cache write).** From each event, project the relevant fields onto
   the in-memory cache and write `.aragora/pr-state-cache.json` atomically
   (mkstemp + `os.replace`, never a partial file — identical discipline to
   `pr_state_cache.py`). The projection is monotonic: an event whose
   `head_sha`/timestamp is older than the cached entry annotates and is
   skipped, never regresses the cache.

The cache schema carries `generated_at` and per-PR `checks_fetched_at` already;
the collector additionally stamps `source: "webhook"` and `last_delivery_id`
per entry so a reader (or the reconciler) can tell webhook-fresh entries from
poll-refreshed ones. `SCHEMA_VERSION` bumps by one and `pr_state_cache.py read`
tolerates both `source` values.

Transport mechanics (pick at build time, both documented):

- **Receiver**: a tiny HTTP endpoint. Cheapest viable form is a single-file
  cloud function (CF worker / Lambda) that verifies HMAC and appends to a
  durable queue; the local collector drains the queue and writes the cache. For
  a single-host fleet, the receiver can be a localhost listener behind a tunnel
  (cloudflared/ngrok) — acceptable for the dogfood phase, not for the witness
  role.
- **Decoupling**: the receiver's only job is verify-and-enqueue; the projection
  into the cache happens in the local collector loop. This keeps the
  agent-reachable host out of the credential path (§7).

## 5. How lanes consume it (unchanged reader contract)

No lane code changes between the polling world and the event world. Lanes:

1. `pr_state_cache.py read <pr>` for routine state (cheap, local, free).
2. exactly one `pr_state_cache.py verify <pr>` REST call for the exact head
   immediately before any mutation.
3. never treat the cache as authority for settlement/merge gates —
   `review-queue merge-packet` for the exact head SHA remains the gate
   authority (see `docs/REVIEW_AUTHORITY_PRINCIPLES.md`).

The substrate makes step 1 *fresher* (push latency, not poll cadence) and makes
step 2's `verify` almost always confirm what the cache already said — but the
`verify` stays mandatory: a single live REST call before mutation is the
fail-closed guarantee that survives a missed or delayed webhook (§8). The epic's
near-term item "wire `pr_state_cache read/verify` into lane recursive-prompt
preambles" is the prerequisite; the substrate inherits it for free.

## 6. Dual role: the tamper-evident-trail external witness

The collector is also Variant A of the external witness in
`docs/specs/TAMPER_EVIDENT_TRAIL.md` (Component 1, "org webhook → collector").
That spec already decided Variant B (Enterprise audit-log streaming → S3
Object-Lock) is **primary** and Variant A is the documented fallback / defense
in depth — so this substrate's witness role is explicitly the *secondary*
witness, valuable because:

- It runs even on plan downgrade or for a second org/repo without Enterprise.
- It provides redundancy during the highest-assurance period.
- Its `webhook-events.jsonl` feeds the same **intent reconciliation** detector:
  the `trail_reconcile` sentinel diffs witness events against pre-registered,
  externally-anchored intent records (TET Component 2/3), and an unmatched
  witness event is a breach.

To serve the witness role honestly, the append path must be **append-only with
external anchoring** — the local `.jsonl` is the working copy, but the witness
root is a bucket whose write credential the laptop never holds (TET §"witness
root", `AragoraTrailWitnessWrite` role). The webhook collector's coverage is
narrower than the audit stream (it cannot see token/deploy-key/App creation),
which is exactly why TET keeps Variant B primary. This design does not
re-litigate that decision; it implements the fallback collector so the cache
update and the witness ingest share one component instead of two.

## 7. Security note: collector credentials never touch agent-reachable disk

Two distinct credentials, neither on a lane-reachable host:

- **Webhook secret** (HMAC shared secret): lives only in the receiver's
  environment (cloud-function secret store). Agents read the *cache*, never the
  secret; possessing the cache grants no ability to forge events.
- **Witness write credential** (`AragoraTrailWitnessWrite`, the bucket
  Object-Lock writer): assumable ONLY by the collector function, never present
  in any agent keyring/env — identical posture to TET. The threat model is
  TET's: even an adversary who controls the laptop and every credential on it
  cannot rewrite the witness, because the write credential isn't there.

The collector's *cache projection* (what lanes consume) is intentionally
low-privilege: it only writes `.aragora/pr-state-cache.json`, a non-authoritative
read cache. The high-privilege append (witness root) is isolated in the
cloud-function half. This split is the load-bearing security property — do not
collapse the two halves onto one agent-reachable host during implementation.

## 8. Failure modes and backstops

| Failure | Detection | Backstop |
|---|---|---|
| Missed/dropped webhook | cache entry's `generated_at`/`checks_fetched_at` ages past `--max-age-seconds`; `read` returns `stale` (exit 3) | the periodic reconcile poll (§9) re-lists via REST+ETag and repairs the entry; the mandatory pre-mutation `verify` catches a stale head in the moment |
| Collector outage (blind window) | sentinel watches collector liveness (TET notes this exact failure mode for Variant A) | reconcile poll keeps the cache usable; sentinel raises a blind-period incident ("silence is never success") |
| Out-of-order delivery | monotonic-by-timestamp/head merge in the projector | older event annotated and skipped, never regresses the cache |
| Forged/replayed event | HMAC verify + `X-GitHub-Delivery` dedup | unverified/duplicate events dropped before projection; witness records the drop |
| Webhook coverage gap (admin events) | known-by-design (webhooks don't expose token/App changes) | TET Variant B audit stream covers it; this collector never claims to |

The reconcile poll is the safety net that lets the substrate fail open *toward
correctness*: if events stop, the system degrades to exactly today's polling
behavior (one budgeted REST poller), not to blindness.

## 9. Migration path (phased, run-alongside-first)

Strictly additive at every step; polling is retired only after the substrate
proves itself against it.

- **Phase 0 — collector alongside poll (shadow).** Deploy
  `event_cache_collector.py` writing to a *separate* cache file
  (`.aragora/pr-state-cache.webhook.json`). The existing `pr_state_cache.py poll`
  keeps writing the canonical cache. A small comparator
  (`scripts/event_cache_collector.py compare`) diffs the two each cycle and
  reports divergence. No lane reads the webhook cache yet. Tier 2.
- **Phase 1 — collector writes the canonical cache; poll demoted to
  reconcile.** Once Phase 0 shows webhook freshness matches/beats poll freshness
  with bounded divergence, the collector writes the canonical
  `.aragora/pr-state-cache.json`. The poller drops to a *reconcile* cadence
  (e.g. every 5–10 min instead of every cycle) whose only job is repairing
  missed-webhook gaps. Most GraphQL/REST list traffic disappears here — this is
  where the quota win lands. Tier 2.
- **Phase 2 — witness wiring.** Point the collector's append at the witness root
  per TET (Variant A fallback) and wire `webhook-events.jsonl` into
  `trail_reconcile`. Tier 2 build; the credential-isolation steps inherit TET's
  Tier-4 identity discipline.
- **Phase 3 — retire residual polling.** When Phase 1 has run for a sustained
  window with zero missed-webhook-caused stale reads (measured: reconcile poll
  finds no deltas the webhook missed), reduce the reconcile poll to a low-cadence
  backstop only. Never remove it entirely — it is the §8 backstop. Tier 2.

Exit metric (falsifiable, mirrors TET's discipline): after Phase 1, fleet
GraphQL consumption per hour drops by the bulk of the polling fan-out, *and* the
reconcile poll reports a near-empty delta set (webhooks were carrying the state).
If the reconcile poll keeps finding deltas the webhooks missed, the subscription
set or the projector is wrong — fix it before advancing, do not retire polling.

## 10. Components named (for the executing lane)

| Component | Path | Tier | Status |
|---|---|---|---|
| Webhook receiver (HMAC verify + enqueue) | cloud function (CF worker / Lambda) | 2 (build) / operator (deploy) | new |
| Local collector (ingest + project) | `scripts/event_cache_collector.py` | 2 | new |
| Shared cache (reader contract) | `.aragora/pr-state-cache.json` via `scripts/pr_state_cache.py` | — | exists (#8339) |
| Webhook event log (witness working copy) | `.aragora/trail/webhook-events.jsonl` | 2 | new |
| Reconcile poller (backstop) | `scripts/pr_state_cache.py poll` (demoted cadence) | — | exists (#8339) |
| Witness root + write role | S3 Object-Lock + `AragoraTrailWitnessWrite` | 4 (identity) | per TET |
| Reconciler check | `trail_reconcile` sentinel | 2 | per TET |

## Cross-references

- Program: `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md` (§2 class E, §3 tamper evidence, §5.1).
- Epic: [#8344](https://github.com/synaptent/aragora/issues/8344) (medium-term phase 1; near-term cache wiring).
- Witness/reconciliation: `docs/specs/TAMPER_EVIDENT_TRAIL.md` (Component 1 Variant A, Components 2–3).
- Shared cache + reader contract: #8339; REST fallback + ETag: #8324; quota class root: #8315/#8316.
- Gate authority that the cache must never substitute for: `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.
