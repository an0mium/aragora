# Agent Bridge — First Real Run Scenario

**Status:** Scenario design — runnable as soon as the bridge write API (POST `/api/v1/agent-bridge/runs` and POST `/runs/{id}/dispatch`, codex's in-flight work) merges and the `agent_bridge_write` feature gate is enabled.

**Purpose:** This is the **first non-smoke** validation of the bridge. The existing `scripts/agent_bridge_live_smoke.py` proves transport works; this scenario proves the bridge produces a useful receipt on real roadmap-relevant work.

## What this validates beyond the smoke

The smoke checks: Claude/codex/droid CLIs launch, sessions resume, footers parse. This scenario additionally checks:

1. **End-to-end via the new write API**, not the script-only CLI — confirms HTTP dispatch works.
2. **Real implementer ↔ reviewer dialog** on actual code, not a no-op prompt.
3. **Footer chain over multiple turns** (≥ 3 turns) including at least one `next_actor` handoff.
4. **Receipt artifact** that future bridge runs can be measured against (latency, footer-parse-rate, repair-loop frequency).

## Scenario: Cross-check review of #6608 closure path

Once Factory's step C (CLI surfacing) lands, all four #6608 deferred code items are in: A (event source, #6615 ✓), B (scheduler, #6616 ✓), C (CLI), D (per-class numerator, #6625). The natural roadmap question is: **does this set actually compose into a working empirical-threshold loop end-to-end?**

That's exactly the kind of cross-cutting question a single harness tends to miss. Perfect first real bridge target.

### Actors

| Role | Harness | Model | Why |
|---|---|---|---|
| `implementer` | `codex` | `gpt-5.4` | Owned step B (scheduler/`ThresholdUpdateReceipt`) — defends design choices |
| `reviewer` | `claude_code` | `claude-opus-4-7` | Independent-perspective second opinion; code reading + thesis-alignment check |
| `synthesist` *(optional, third turn)* | `droid` | `claude-opus-4-7` | Reconciles implementer + reviewer if they disagree; produces a settlement note |

The implementer/reviewer pairing is the minimum viable adversarial cross-check. The synthesist is optional but lets us validate the 3-actor footer chain in one run.

### Task

```
Cross-check the #6608 closure path end-to-end.

Files in scope:
  - aragora/review/invalidation.py (vocabulary, classifier, baseline computer, threshold deriver)
  - aragora/review/invalidation_event_source.py (event source from receipt stores)
  - aragora/review/threshold_recalibration.py (scheduler + ThresholdUpdateReceipt)
  - aragora/cli/commands/review_queue.py (operator CLI baseline subcommand)
  - tests/review/test_invalidation*.py (38+13+N tests)

Question: does this set compose into a working empirical-threshold loop?

Specifically check:
  1. Can a CLI baseline run feed a scheduler invocation that emits a
     ThresholdUpdateReceipt without round-tripping through manual code?
  2. Does the per-class breakdown produce useful per-class thresholds, or
     does the safety-margin floor swallow the per-class signal?
  3. Is anything load-bearing for THESIS.md Commitment 3 still missing
     beyond the calendar-bound "real data accumulation" step?
  4. What's the smallest follow-up PR that would let the founder's
     overnight cycle generate the first ThresholdUpdateReceipt against
     real settlement data?
```

### Turn sequence

| Turn | Role | Prompt |
|---|---|---|
| 1 | `implementer` | "Read the four files above. Walk through what happens end-to-end when an operator runs `aragora review-queue baseline`. Identify any seams where the loop is incomplete or where the framework returns a placeholder threshold instead of a measured one. Emit a footer with `next_actor: reviewer`." |
| 2 | `reviewer` | "Read turn 1. Independently walk through the same files and report: (a) where you agree with the implementer, (b) where you disagree, (c) any gap they missed. Specifically check thesis-alignment against `docs/THESIS.md` Commitment 3. Emit a footer with `next_actor: synthesist` if there's substantive disagreement, else `next_actor: null` and `done: true`." |
| 3 *(if dispatched)* | `synthesist` | "Reconcile turns 1 and 2. Where they agreed, restate the consensus. Where they disagreed, propose a settlement (or surface the question to the operator if neither side is clearly correct). Emit a footer with `done: true` and `artifacts` listing any follow-up issues you'd recommend filing. Do NOT actually file them." |

### Expected footer fields per turn

All turns must emit:
- `summary`: one-sentence what-was-done
- `next_actor`: `implementer` | `reviewer` | `synthesist` | `null`
- `needs_human`: boolean (true if the agent hits something it can't decide)
- `done`: boolean (true on the final turn)
- `artifacts`: array of `{type, path}` (e.g., transcript paths, recommended issue titles)
- `tests_run`: array — empty for this read-only scenario

### Success criteria

The run is **successful** if:

1. All ≥ 3 turns complete without invoking the repair loop more than once per turn.
2. Each footer parses on first try (no malformed output).
3. The reviewer (turn 2) **disagrees with at least one specific point** from turn 1. (If they don't, the bridge isn't producing real adversarial signal — investigate prompts.)
4. The synthesist (turn 3, if dispatched) produces a `settlement_note` artifact — a 200-300 word reconciliation that an operator could act on.
5. The combined footer chain identifies **at least one concrete follow-up** for #6608 closure (or honestly states none exist).
6. Total cost stays under $5 (bounded read-only review on ~5 files).

The run is **partially successful** if 1–4 hold but 5–6 don't — the infrastructure works, the prompts need tuning.

The run is a **failure** if any of: footer parse fails twice, `needs_human=true` on turn 1, transport drops a session.

### Invocation (once write API lands)

```bash
# Start the run
curl -sX POST http://localhost:8080/api/v1/agent-bridge/runs \
  -H "Authorization: Bearer ${ARAGORA_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Cross-check the #6608 closure path end-to-end (see docs/operations/agent_bridge_first_real_run.md)",
    "actors": [
      {"role": "implementer", "harness": "codex", "model": "gpt-5.4"},
      {"role": "reviewer", "harness": "claude_code", "model": "claude-opus-4-7"},
      {"role": "synthesist", "harness": "droid", "model": "claude-opus-4-7"}
    ]
  }' | jq .

# Save the run_id from the response, then dispatch turn 1
RUN_ID=...

curl -sX POST "http://localhost:8080/api/v1/agent-bridge/runs/${RUN_ID}/dispatch" \
  -H "Authorization: Bearer ${ARAGORA_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d @docs/operations/agent_bridge_first_real_run_turn1.txt
```

(Subsequent turns can use the auto-step endpoint once it lands.)

### What we'd learn from this run

Independent of whether the cross-check finds bugs in the #6608 path, the bridge run itself is the artifact. It tells us:

- Does the write API work in a real operator flow?
- Do the harnesses produce footer-compliant output on substantive prompts (not just smoke)?
- Is the dispatch latency tolerable for interactive operator use?
- What's the actual cost-per-decision for an adversarial cross-check?

These data points feed Phase 1 probe #6610 (with-receipt A/B) — the bridge IS the production receipt-generation surface; this run is the first real receipt produced by it.

### Sequel runs (after the first)

Once the first run produces a clean transcript, the next bridge runs in priority order:

1. **Phase 1 probe #6610** (with-receipt vs without-receipt A/B) — the bridge becomes the data-collection surface.
2. **Cross-check codex's bridge implementation PR** through the bridge itself (recursive validation).
3. **Cross-check #6614** (AGT-05 vision-layer skeleton) — sanity on whether the vision-layer slice is coherent before merging into main.

## Open questions for the operator

- Should the synthesist run by default, or only when reviewer's footer flags substantive disagreement? (Recommend: default-on for the first run, revisit after observing.)
- Should the bridge run cost be capped at the actor level (e.g., $2 per actor per turn), at the run level (e.g., $5 per run), or both? (Recommend: per-run cap as primary control.)
- Should the run write to a real receipt store (`aragora/inbox/trust_wedge.py` style) or remain in `~/.aragora/agent_bridge/runs/` only? (Recommend: bridge runs are the receipt store; no separate persistence needed.)

## Related

- `scripts/agent_bridge_live_smoke.py` — the existing smoke (transport-only)
- `aragora/swarm/agent_bridge/` — broker + harnesses (shipped)
- `aragora/server/handlers/agent_bridge.py` — read API (shipped); write API (in flight, codex)
- #6608 — the closure-path epic this scenario reviews
- #6610 — Phase 1 probe that this scenario's receipt feeds into
