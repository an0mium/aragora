# SpecUpgrader Design — 2026-04-17

## Goal

Convert Aragora's boss-loop dispatch pipeline from a gate (rejects weak specs) to a repair loop (upgrades weak specs, dispatches the upgraded result, audits the transformation). Target failure class: `blocked_not_dispatch_bounded` (currently 42% of terminal outcomes, 0 of 305 historical ticks produced a merged PR).

v1 addresses the narrow production blocker. The mechanism generalises to other integration points (CLI, pipeline stages) in follow-ups.

## Context

Current behaviour on dispatch:

- `spec.is_dispatch_bounded()` at `aragora/swarm/spec.py:482` returns False → dispatch short-circuits with `blocked_not_dispatch_bounded`.
- A preflight worker runs (`aragora/swarm/preflight.py:1680`) and emits a `WorkerContract`; if the emitted contract diverges from the expected contract, `_enforce_expected_contract()` at `preflight.py:1566` fails the dispatch.
- Both failure classes currently surface as dead ends. The existing `dispatch_followups.maybe_upgrade_dispatch_spec()` at `aragora/swarm/boss_worker_lifecycle.py:801` calls `issue_upgrader.upgrade_issue_heuristic()` at `aragora/swarm/issue_upgrader.py:378`, which is limited to four narrow categories (`test_coverage`, `broad_exception`, `silent_exception`, `type_annotation`) and is gated by a heuristic that rejects most real issues. The LLM-backed sibling `upgrade_issue_llm()` at `issue_upgrader.py:491` exists but is not wired into the current follow-up hook and retains the same narrow-category assumption.

Structured diagnostics already exist: `missing_dispatch_bounds()` at `spec.py:486` returns a list of missing fields; preflight surfaces contract drift at `preflight.py:1138`. Neither signal currently feeds a repair attempt.

## Architecture

Single public library module `aragora/swarm/spec_upgrader.py` with one entry point:

```python
def upgrade_spec(
    spec: SwarmSpec,
    failure_context: UpgradeFailureContext,
    *,
    issue_number: int | None = None,
    max_attempts: int = 2,
) -> UpgradeResult
```

`UpgradeResult` is a tagged union — `status="upgraded"` (spec populated) or `status="escalated"` (unresolved_questions populated). No other outcomes.

Integration points (v1) — **two seams**, preserving the existing pre-gate repair and adding a new post-gate repair:

1. **Seam A (pre-contract-gate, replaces existing call)**: `dispatch_followups.maybe_upgrade_dispatch_spec()` at `boss_worker_lifecycle.py:801` currently calls `upgrade_issue_heuristic()`. Replace that call with `upgrade_spec()`. This is the first repair: it runs before the contract gate and handles unbounded specs.
2. **Seam B (post-contract-gate, new)**: When `dispatch_contract_gate()` at `boss_worker_lifecycle.py:876` fails with contract drift, call `upgrade_spec()` again with the drift diagnostic populating `UpgradeFailureContext.preflight_diff`. This is the second repair: drift feedback into a targeted re-upgrade.
3. **Shared attempt budget**: both seams share one `max_attempts=2` counter per issue (durable across ticks via the `[spec-upgraded]` comment). If Seam A consumes one attempt, Seam B has one remaining before escalation.
4. `boss_loop.py:419` is a last-resort fallback; in normal flow, issues reach it only if Seams A and B are somehow bypassed.

Integration points (v2, out of scope):
- CLI: `aragora upgrade <issue-N>` wraps `upgrade_spec()`.
- Pipeline: add a "Spec Upgrade" stage between Goals and Workflow in `aragora/pipeline/`.

## Components

### `UpgradeFailureContext` (dataclass)

Fields: `missing_bounds: list[str]`, `preflight_diff: dict | None`, `prior_attempts: int`, `original_issue_body: str`, `issue_title: str`, `track_tag: str | None` (e.g. `TW-02`, `CS-01`).

Constructed from `spec.missing_dispatch_bounds()`, preflight drift diagnostics, and the durable attempt count.

### A-path: deterministic enrichment (Tier 1, no LLM)

Attempts to bound the spec from static signals:

1. Parse `missing_bounds` → fill corresponding fields with concrete defaults:
   - Missing acceptance criterion → synthesise from issue body's first imperative sentence, or from track-tag conventions.
   - Missing file-scope hint → regex-extract `path/to/file.py` mentions from body, validate against repo.
   - Missing constraint → populate from preflight contract's `files` and `forbidden_paths`.
   - Missing work order → derive from title + body's top-level bullets.
2. Translate preflight contract-drift diagnostic into an acceptance criterion ("Worker must scope changes to: <files>. Reject any drift from this list.").
3. Infer **candidate** scope hints from track-tag prefixes, treated as low-confidence. A hint is only merged into the spec if it passes validation against the repo — the referenced file exists AND is either mentioned in the issue body OR returned by `git grep` for the issue's key identifiers. Starter table (extensible constant in `spec_upgrader.py`):
   - `TW-*`, `CS-*`, `RS-*` → likely live under `aragora/swarm/` subtree; propose candidates but require validation before merging into scope.
   - `DIC-*`, `AGT-*` → vision-layer, design-heavy. Do NOT infer scope from the prefix — fall through to Tier 2 (LLM) or, if already in Tier 2, escalate.
   - Unknown prefix → fall through to Tier 2, do not guess.
   Validation failure on all candidates → fall through to Tier 2 or escalate. Track-tag inference must never produce scope hints the upgrader hasn't verified against the current repo.
4. Opportunistically delegate to `issue_upgrader.upgrade_issue_heuristic()` when the module category matches — no dependency on its success.

If the resulting spec passes `is_dispatch_bounded()`, return immediately without an LLM call.

### A-path: LLM enrichment (Tier 2, fallback)

Invoked only when Tier 1 cannot bound the spec. New prompt, not routed through `upgrade_issue_llm()`:

- Input: current spec state, `UpgradeFailureContext`, repo file tree at two levels of depth.
- Model/client: use the configured upgrader agent (via the existing `aragora.agents` factory). Default model, fallback chain, and quota behavior come from config — likely `ARAGORA_UPGRADER_MODEL` env or `aragora/config/`. The design does not pin a specific model so policy can change without touching the upgrader.
- Output: structured JSON matching `SwarmSpec` field names, validated before merge into the spec.
- Transient infrastructure failure (5xx, timeout, connection error): raise `SpecUpgraderUnavailable` exception — caller treats the issue as "skip for this tick, retry next tick". Does NOT consume an attempt. Keeps the 2-state `UpgradeResult` invariant.
- Logic failure (LLM returns malformed or ungrounded output after one local retry): return `Escalated` with the failure summary in `unresolved_questions`. Consumes the attempt.

### PreflightGate (integration, not new code)

After the upgrader mutates the spec, the dispatch flow re-enters `dispatch_contract_gate()`, which calls `run_contract_preflight_receipt()` at `preflight.py:1138`. The gate result is evaluated via `evaluate_preflight_receipt_gate()` at `preflight.py:1246`. Drift is surfaced inside preflight by `_enforce_expected_contract()` at `preflight.py:1566`. If the contract still drifts after an upgrade, the drift diagnostic populates the next `UpgradeFailureContext.preflight_diff` — this is the Seam-B feedback.

### AuditPersistence

Idempotent upsert of a single `[spec-upgraded]` comment on the issue:

- **Marker format (strict)**: `<!-- spec-upgraded:v1 attempt=N -->` on the first line of the comment body, where N is an integer 0 ≤ N ≤ 2.
- **On entry**: scan issue comments for marker. Three cases:
  1. Marker present and parseable → use the parsed `attempt_count` as durable source of truth.
  2. Marker present but unparseable/corrupted → treat as **max-attempts reached**, escalate immediately, and rewrite a valid marker with a loud warning log (`upgrade_audit_marker_corrupted`). Do NOT reset to 0 — that would create unbounded retry loops on malformed comments.
  3. No marker → `attempt_count=0`, create marker on first write.
- **On success**: update comment in place with the new attempt's audit markdown, bumping `attempt=N` in the marker.
- **On GitHub API failure** (write path): dispatch anyway, set `upgrade_audit_failed=true` in metrics, log warning. Audit persistence is best-effort; upgrade logic is not.

### Escalator (C-path)

Triggered when `attempt_count == 2` and preflight still fails. Actions:

1. Apply label `needs-clarification` via `gh issue edit`.
2. Post a comment (distinct from the audit comment) with the unresolved questions and the latest `UpgradeFailureContext` summary.
3. Record `upgrade_escalated=true` in metrics.

If either `gh` mutation fails: fail closed — do NOT dispatch, record a distinct `upgrade_escalation_failed` metric, emit a loud log warning. The boss loop skips the issue on its next tick.

## Data flow

The actual dispatch order in `boss_worker_lifecycle.py` is: follow-up (upgrade-if-unbounded) first, then contract gate. SpecUpgrader slots into both seams:

```
dispatch_issue
  │
  ▼
Seam A: dispatch_followups.maybe_upgrade_dispatch_spec()
  │  [reads [spec-upgraded] comment for attempt_count]
  │  [if attempt_count >= 2 OR corrupted marker: Escalator → skip]
  │
  ├── spec is bounded? ──yes──► (no upgrade needed) ────┐
  │                                                      │
  │ not bounded                                          │
  ▼                                                      │
upgrade_spec(spec, ctx={missing_bounds})                 │
  │                                                      │
  ▼                                                      │
Tier 1 deterministic → Tier 2 LLM if needed              │
  │                                                      │
  ├── Upgraded ──► AuditPersistence (attempt=1) ─────────┤
  │                                                      │
  └── Escalated ──► Escalator → skip dispatch            │
                                                         │
                                                         ▼
                                            dispatch_contract_gate()
                                               [Seam B opportunity]
                                                         │
                               pass ──► dispatch worker ─┘
                                                         │
                                                         fail (drift)
                                                         │
                                                         ▼
Seam B: upgrade_spec(spec, ctx={missing_bounds, preflight_diff})
  │  [attempt_count now 1 if Seam A upgraded, else 0 — from marker]
  │  [if attempt_count would exceed max=2: Escalator → skip]
  │
  ▼
Tier 1 deterministic (now targeting the drift) → Tier 2 LLM if needed
  │
  ├── Upgraded ──► AuditPersistence (attempt=1 or 2) ──► re-enter dispatch_contract_gate()
  │                                                         │
  │                                            pass ──► dispatch worker
  │                                                         │
  │                                                      fail
  │                                                         │
  │                                                         ▼
  │                                               Escalator → skip
  │
  └── Escalated ──► Escalator → skip dispatch
```

Key properties:
- **Seam A handles the unbounded-spec class** (issue body lacks acceptance criteria, file scope, etc.). Same class the existing `maybe_upgrade_dispatch_spec` already targets — we're replacing its narrow heuristic implementation with the richer SpecUpgrader.
- **Seam B handles the contract-drift class** (preflight worker emits a contract that deviates from expected). This is a net-new repair path; today drift is a dead end.
- **Attempt budget is shared across seams** (durable via audit marker). An issue that exhausts both attempts via Seam A+B chain escalates.
- **`dispatch_contract_gate()` is not modified** — the new integration wraps its failure return path. This preserves the existing contract-receipt architecture.

## Error handling precedence

| Failure | Behaviour |
|---|---|
| LLM transient (5xx/timeout/connection) | 1 local retry + backoff; if still failing, raise `SpecUpgraderUnavailable` — caller skips issue for this tick, does NOT consume an attempt |
| LLM logic (malformed output, ungrounded response) after 1 local retry | Return `Escalated` with failure summary in `unresolved_questions` — consumes the attempt |
| Preflight timeout | Count as failed attempt; next seam or escalation decides next step |
| Audit comment persistence fail (GitHub API) | Dispatch anyway, set `upgrade_audit_failed=true` |
| Audit marker corrupted on entry | Treat as max-attempts reached, escalate, rewrite valid marker, emit `upgrade_audit_marker_corrupted` |
| Escalation `gh` mutation fail (label/comment) | **Fail closed** — do NOT dispatch, record `upgrade_escalation_failed`, loud warning |
| Budget guard (max 2 attempts across both seams, 1 LLM call + 1 preflight per attempt) | Escalate immediately on limit breach |

No third state. `UpgradeResult.status` is exclusively `"upgraded"` or `"escalated"`. Transient infrastructure errors bubble up as exceptions and are handled by the caller, not as an upgrade outcome.

## Telemetry

Upgrade activity must be observable **whether or not dispatch follows** — escalated issues never reach the existing dispatch record. Emit a dedicated per-upgrade row to `boss_metrics.jsonl`, and link it from the dispatch row (if one occurs):

**Per-upgrade row** (emitted on every `upgrade_spec()` call, even on escalation):
- `event: "spec_upgrade"`
- `upgrade_id: <uuid>`
- `issue_number: int`
- `seam: "A" | "B"`
- `attempt_count: int` (1 or 2)
- `status: "upgraded" | "escalated"`
- `upgrade_path: "deterministic" | "llm" | "deterministic+llm" | null` (null when escalated without attempting a path)
- `wall_clock_ms: int`
- `audit_failed: bool`
- `escalation_failed: bool` (only when status=escalated)
- `llm_tokens_in/out: int` (when LLM tier invoked)
- `failure_reasons: list[str]` (from `UpgradeFailureContext`)

**Dispatch row additions** (existing record, when dispatch occurs):
- `upgrade_refs: list[str]` — list of `upgrade_id` values for upgrades that contributed to this dispatch (usually one; can be two if Seam A and Seam B both ran)

This lets the Stage-Gate Conductor compute: upgrade attempt rate, upgrade success rate by seam, attempt distribution, LLM cost drift, and the `blocked_not_dispatch_bounded` delta post-deployment.

## Testing

### Unit (new test file `tests/swarm/test_spec_upgrader.py`)

- `UpgradeFailureContext` construction from `missing_dispatch_bounds()` and preflight drift dicts.
- Tier 1 deterministic enrichment per missing-bound category: synthesise acceptance criterion, file-scope hint, constraint, work order. Happy path + empty-body path + unparseable-body path.
- Tier 2 LLM enrichment with mocked client: fixed input → expected structured output; malformed LLM output → fallthrough to escalation.
- `AuditPersistence` upsert idempotency: first run creates comment, second run updates in place with incremented `attempt_count`.
- `Escalator` label + comment correctness; failure of either mutation triggers `upgrade_escalation_failed`.
- Attempt-count recovery: given a fixture issue with an existing `[spec-upgraded]` comment showing `attempt_count=1`, the next call correctly increments.

### Integration

- Real preflight + mocked LLM: verify the retry feedback loop (drift → upgrade → re-preflight → drift → upgrade → pass).

### End-to-end regression

Frozen fixtures at `tests/swarm/fixtures/spec_upgrader/issue_5898.json` and `issue_5903.json` (captured title + body + comments at time of design). Mock LLM, mock preflight. Assertion: `upgrade_spec()` returns `status="upgraded"` with the missing bounds filled. This is the "would have worked" test — the primary deliverable signal.

A separate manual-smoke test (not in CI) re-runs against live `gh issue view 5898/5903` output to validate the fixtures remain representative.

## Acceptance criteria

1. `aragora/swarm/spec_upgrader.py` exists and exports `upgrade_spec()`, `UpgradeFailureContext`, `UpgradeResult`.
2. `dispatch_followups.maybe_upgrade_dispatch_spec()` at `boss_worker_lifecycle.py:801` calls `upgrade_spec()`; contract gate at `boss_worker_lifecycle.py:876` feeds drift back into next attempt.
3. Unit + integration + E2E regression tests all pass.
4. Fixture-level E2E for #5898 and #5903 produces `status="upgraded"` with bounded specs.
5. `boss_metrics.jsonl` includes per-upgrade `event="spec_upgrade"` rows emitted on every `upgrade_spec()` call; dispatch rows include `upgrade_refs` listing contributing upgrade IDs when dispatch occurs.
6. `boss_worker_lifecycle.py` LOC ratchet remains under its hard limit.

## Out of scope (explicit v2 candidates)

- Shape 3 B-path: `Arena(consensus="spec_upgrade")` — deferred until A-path telemetry shows where it falls short.
- **Tier 2b two-model ping-pong refinement**: model A proposes an upgraded spec, model B critiques for missing bounds / hallucinated scope / unvalidated claims, model A revises; capped rounds (suggested 3), transcript persisted as part of the audit comment. Lower-cost precursor to full Arena consensus and natural extension of Tier 2. Wait until Tier 2 telemetry reveals the single-model failure shape before designing the critique prompt.
- Generalized pipeline-stage upgrading (Ideas → Goals → Workflows).
- Pre-ingestion upgrading at issue-create time.
- Autonomous multi-turn Q&A answering (self-play variant of A-path).
- CLI `aragora upgrade <issue-N>` — added in v1.1 once the library boundary is proven.

## Risks

- **LLM hallucinations fill in plausible-but-wrong bounds.** Mitigation: preflight is the gate. If the upgraded spec would produce contract drift, the second attempt sees it; if still wrong, escalation.
- **Attempt-count parsing from comment is fragile.** Mitigation: strict versioned marker format `<!-- spec-upgraded:v1 attempt=N -->`. If parsing fails, treat as max-attempts reached, escalate, and rewrite a valid marker with a loud warning. Erring toward escalation prevents unbounded retry loops on malformed comments at the cost of occasional false escalations (acceptable — the human review that follows is cheap and catches both).
- **LLM / API cost escalation.** Mitigation: hard cap of 2 LLM calls per issue via attempt counter; Stage-Gate Conductor flags weekly LLM-cost drift. Actual model(s) and pricing come from config, so this bound holds regardless of model choice.
- **Integration seam moves under us.** `dispatch_followups.maybe_upgrade_dispatch_spec()` is called from boss-lifecycle code that may refactor. Mitigation: integration test covers the full dispatch path end-to-end, not just the upgrader in isolation.

## References

- `aragora/swarm/spec.py:482` — `is_dispatch_bounded()`
- `aragora/swarm/spec.py:486` — `missing_dispatch_bounds()`
- `aragora/swarm/boss_worker_lifecycle.py:801` — `dispatch_followups.maybe_upgrade_dispatch_spec()` (Seam A integration)
- `aragora/swarm/boss_worker_lifecycle.py:876` — contract gate failure path (Seam B integration)
- `aragora/swarm/preflight.py:1138` — `run_contract_preflight_receipt()`
- `aragora/swarm/preflight.py:1246` — `evaluate_preflight_receipt_gate()`
- `aragora/swarm/preflight.py:1566` — `_enforce_expected_contract()` (drift surface)
- `aragora/swarm/issue_upgrader.py:378` — `upgrade_issue_heuristic()` (opportunistic reuse, not required)
- `aragora/swarm/issue_upgrader.py:491` — `upgrade_issue_llm()` (v1 does NOT depend on this)
- `docs/benchmarks/corpus_honesty_audit_2026-04-17.md` — corpus hollowness finding motivating this work
- `docs/reviews/swarm_prs_2026-04-17.md` — related substrate-PR review
