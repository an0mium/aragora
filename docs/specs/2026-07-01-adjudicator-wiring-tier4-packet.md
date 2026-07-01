# Tier-4 Preapproval Packet: wire ReviewAdjudicator into the quorum stall path (#8748)

**Status:** PREPARE-ONLY — awaiting (a) human Tier-4 preapproval, (b) release of the Codex
timeout-family freeze on the target files. Produced by the close-the-loop run (epic #8762, B5).

## What already exists (verified on main, 2026-07-01)

- `aragora/swarm/review_adjudicator.py` (#8749, merged): pure, flag-gated
  (`ARAGORA_ENABLE_REVIEW_ADJUDICATOR`, default OFF) `adjudicate(items) ->
  SETTLE | BLOCK | ESCALATE | NOT_APPLICABLE`. Hard bar ([P0]/[P1]) evaluated without the
  scorer; single-scoring invariant; conservative bias (SETTLE only when *every* dissent is thin).
- `#8756` (open): M0a operator-approved advisory settlement records — the posting primitive.
- Advisory-settle reachability in the enforcing quorum job (#8741, merged).

## Proposed wiring (exact, for the human to preapprove)

1. **Call site A — `aragora/swarm/quorum_evidence.py`** (currently FROZEN, Codex lane): in the
   outcome-aggregation step where the collector concludes `action=prepare` with
   `dissenting_families` non-empty AND `supportive_families` non-empty (the stall shape observed
   live on #8389 round-2: claude PASS / openai CHANGES-REQUESTED), call
   `review_adjudicator.adjudicate()` over the reviewer findings when the flag is ON. Record the
   `AdjudicationResult` in the collect outcome JSON (`adjudication` key) — no behavior change to
   posting in M0-wiring step 1 (observe-only).
2. **Call site B — `aragora/cli/commands/review_queue.py` merge-packet** (Tier 4 self-arbiter
   surface): map a recorded `adjudication.verdict == SETTLE` on the exact head SHA to a new
   packet verdict `adjudicated_settle` that satisfies Tier 1-2 settlement *only when* the
   existing advisory_settle_surface_clear predicate also holds. `BLOCK`/`ESCALATE` never weaken
   anything: they add reasons.
3. **Receipt**: every adjudication emits a DecisionReceipt (ConsensusBuilder) naming suppressed
   findings + severities; suppressed advisory findings are auto-filed as follow-up issues
   (compose the same WIP-capped filing used by the harvest engine #8760).

## Rollout preconditions (all must hold before implementation)

1. Human Tier-4 preapproval recorded on #8748 (this packet is the artifact).
2. Codex timeout-family lane released or merged (#8726) — no concurrent edits to
   `quorum_evidence.py`.
3. Step-1 (observe-only) ships first and runs on ≥3 real stalls with the operator eyeballing the
   `adjudication` outputs before step-2 (packet consumption) is enabled.
4. Governance tests in `tests/governance/` pin: flag OFF → byte-identical packet output; [P0]/[P1]
   present → never `adjudicated_settle`; ESCALATE → packet gains a human-crux reason.

## Live evidence motivating this (from the run)

- #8389: two full gate cycles; round-2 claude PASS / openai new [P1] — a genuine hard-bar case
  (adjudicator would correctly BLOCK; the [P1] is real, filed as #8765).
- #8519: openai [P2] contradicting an intentional characterization test — a genuine ESCALATE
  case (material two-sided disagreement; human crux).
- #8460: clean 2-0 after evidence — NOT_APPLICABLE case (no stall).
The three verdict classes all occurred naturally within one drain wave, which is strong evidence
the adjudicator's taxonomy carves the space correctly.
