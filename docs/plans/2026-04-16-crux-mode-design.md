# Crux Finder Mode — Design

**Goal:** Elevate aragora's existing crux-detection machinery into a first-class, user-facing debate goal with distinct positioning: *"aragora can give you an answer, or aragora can show you where reasonable people diverge."* Ship a thin-wiring MVP (Approach A) that exposes the latent `CruxDetector` via a new consensus mode, a dedicated signed receipt, and a `aragora crux` CLI verb. Gate the deeper protocol-shaping work (Approach B) on dogfood signal from A.

**Architecture:** Additive only. No new subsystems; no breaking changes to existing debate/consensus/receipt code. One new consensus mode string (`crux_finder`), one new proof-builder function (mirrors `build_proof_from_prover_estimator`), one new dataclass (`CruxReceipt`), one new CLI command module. All leverage existing `aragora.reasoning.crux_detector.CruxDetector`, existing `BeliefNetwork` population in `aragora.debate.phases.belief_analysis`, and existing SHA-256 signing pattern from `aragora.gauntlet.receipt_models.ConsensusProof.checksum`.

**Tech Stack:** Python 3.11, existing aragora modules (no new dependencies).

**Working directory:** worktree at `.claude/worktrees/<agent>-<hash>/` (do not work in repo root — see CLAUDE.md worktree isolation rules).

**Tier / canonical home:** This is the implementation-level plan for **DIC-15** in the Decision Integrity Core tranche per [docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md](EPISTEMIC_CI_AND_CRUX_ENGINE.md). That canonical doc gates the tranche on proof-first Foreman reliability. This design doc is a ready-to-execute implementation breakdown for when the gate opens; it does not itself change the gate.

**Correction:** An earlier version of this doc placed this work under FEATURE_GAP_LIST P1 and opened a parallel epic (#6035) with subtasks #6036–#6039. Those were filed before I pulled the commits that introduced `EPISTEMIC_CI_AND_CRUX_ENGINE.md` and the DIC-13..22 sequence. The parallel epic + subtasks have been closed as superseded; this doc is re-routed as implementation-level planning under **#6025 (DIC-15)** and #6026 (DIC-16) for the CruxReceipt slice.

**GitHub tracking (canonical):**
- [#6025](https://github.com/synaptent/aragora/issues/6025) **DIC-15** — CruxSet contract and crux-finder consensus mode (this doc is the implementation plan)
- [#6026](https://github.com/synaptent/aragora/issues/6026) **DIC-16** — receipt and Knowledge Mound provenance (covers CruxReceipt slice)
- [#6027](https://github.com/synaptent/aragora/issues/6027) **DIC-17** — unresolved cruxes → bounded follow-up (Track B, gated on triggers in this doc)

**Related code:**
- `aragora/reasoning/crux_detector.py` — CruxDetector, CruxClaim, CruxAnalysisResult (already exists)
- `aragora/reasoning/belief.py` — BeliefNetwork (already exists)
- `aragora/debate/consensus.py:997` — build_proof_from_prover_estimator (reference pattern)
- `aragora/debate/protocol.py:218-235` — DebateProtocol consensus Literal + mode-specific config
- `aragora/gauntlet/receipt_models.py` — DecisionReceipt, ConsensusProof (reference pattern)
- `aragora/explainability/decision.py` — Counterfactual dataclass (reuse for crux validation)
- `docs/FEATURE_GAP_LIST.md` — P1 tier placement

---

## Context: What Exists vs What's Missing

### Already built (no change required)

- **`CruxDetector`** (`aragora/reasoning/crux_detector.py:85-436`): Counterfactual-based crux scoring. Weighs influence × disagreement × uncertainty × centrality × resolution_impact. Emits `crux_detected` events. Syncs to Knowledge Mound via `BeliefAdapter`.
- **`CruxClaim`, `CruxAnalysisResult`** dataclasses (`crux_detector.py:31-82`): `to_dict()` serialization, ranked-focus output.
- **`BeliefNetwork`** population from debate claims (`aragora/debate/phases/belief_analysis.py`, `analytics_phase.py`, `winner_selector.py`).
- **HTTP handlers** (`aragora/server/handlers/belief.py`, `decisions/explain.py`) and SDK (`sdk/python/aragora_sdk/namespaces/belief.py`).
- **`Counterfactual`** dataclass (`aragora/explainability/decision.py:154-160`): `condition`, `outcome_change`, `likelihood` — direct fit for crux validation.
- **Receipt signing pattern**: `ConsensusProof.checksum` property (`aragora/debate/consensus.py:276-294`) — SHA-256 over sorted-keys JSON.

### Latent but not wired

- **No `consensus="crux_finder"` mode.** `DebateProtocol.consensus` Literal at `aragora/debate/protocol.py:218` does not include it. Cruxes currently fall out as a post-analysis byproduct in `winner_selector`, not as a first-class debate goal.
- **No `CruxReceipt`.** Cruxes hide inside `DecisionReceipt.live_explainability`. There is no signed, exportable artifact where cruxes are the headline.
- **No `aragora crux` CLI verb.**
- **No crux-shaping prompt templates.** Agents debate to converge; cruxes fall out of that. Whether *shaping* the debate to locate cruxes would materially improve output is an open question — answered by Approach B dogfood only after A ships.

This plan elevates the latent capability into a first-class feature without rebuilding anything.

---

## Approach: A First, B Gated on Signal

**Approach A (this plan)**: Thin wiring. New consensus mode runs a normal debate and short-circuits to `CruxDetector.detect_cruxes()` at the end, producing a `CruxReceipt`. CLI verb wraps this for single-command use. No new prompts, no new debate dynamics.

**Approach B (deferred follow-up)**: Crux-shaping protocol. New prompt templates direct agents to locate disagreement rather than converge; per-round claim targeting picks highest-influence claims for next-round focus; Nomic Loop integration feeds `OutcomeFeedbackRecord` for self-identified fragile reasoning.

**B trigger criteria (explicit, measurable):**
1. A has been dogfooded on ≥10 distinct real-question debates.
2. Observed: either (a) <3 distinct cruxes per run on average (indicating incidental-only discovery), or (b) user feedback that cruxes feel "after-the-fact" rather than "what the debate explored", or (c) crux scores cluster below 0.4 (below the KM-sync default threshold).
3. If no trigger fires in 30 days of dogfood, close B as "not needed" — A is sufficient.

---

## Track A: Thin-Wiring MVP

### Task A1 — New consensus mode `crux_finder`

**Files:**
- Modify: `aragora/debate/protocol.py` (Literal + three mode-specific fields)
- Modify: `aragora/debate/consensus.py` (new builder function `build_proof_from_crux_finder`)
- Modify: `aragora/debate/orchestrator_runner.py` or the phase executor that dispatches `protocol.consensus` (follow `prover_estimator` dispatch pattern)
- Create: `aragora/debate/crux_mode.py` (new module — run-time orchestration for the mode)

**Step 1 — Extend `DebateProtocol`**

In `aragora/debate/protocol.py` around line 218, add `"crux_finder"` to the `consensus` Literal and three mode-specific fields:

```python
consensus: Literal[
    "majority", "judge", "prover_estimator", "crux_finder", ...
] = "judge"

# Crux-finder mode config
crux_finder_top_k: int = 5
crux_finder_min_score: float = 0.3
crux_finder_counterfactual_validation: bool = True
```

Default values mirror `CruxDetector.detect_cruxes` defaults and the existing KM-sync threshold (0.3).

**Step 2 — Define `CruxFinderResult`**

Create `aragora/debate/crux_mode.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from aragora.reasoning.crux_detector import CruxAnalysisResult, CruxClaim

@dataclass
class CruxFinderResult:
    """Result of a crux-finder debate.

    Distinct from a ConsensusProof because the deliverable is *not* a verdict.
    Carries all fields needed to build both ConsensusProof (for downstream
    compatibility) and CruxReceipt (for signed export).
    """

    debate_id: str
    question: str
    analysis: CruxAnalysisResult
    counterfactuals: list[dict]  # Validation evidence: each crux produces one
    agents: list[str]
    rounds: int
    raw_claims: list[dict] = field(default_factory=list)  # Full claim log
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_cruxes(self) -> list[CruxClaim]:
        return self.analysis.cruxes

    def convergence_barrier(self) -> float:
        return self.analysis.convergence_barrier

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "question": self.question,
            "analysis": self.analysis.to_dict(),
            "counterfactuals": self.counterfactuals,
            "agents": self.agents,
            "rounds": self.rounds,
            "raw_claims": self.raw_claims,
            "metadata": self.metadata,
        }
```

**Step 3 — Implement the mode orchestrator**

Add to `crux_mode.py`:

```python
async def run_crux_finder(
    arena: "Arena",
    protocol: "DebateProtocol",
    question: str,
) -> CruxFinderResult:
    """Run a debate in crux-finder mode.

    MVP behavior (Approach A): run the standard debate protocol, then
    extract cruxes from the populated BeliefNetwork. No prompt shaping.
    """
    # Run the underlying debate as usual. The existing belief_analysis
    # phase already populates a BeliefNetwork from agent claims.
    debate_result = await arena.run_standard_rounds()

    network = debate_result.belief_network
    if network is None:
        raise RuntimeError(
            "crux_finder mode requires belief_analysis phase to have populated "
            "a BeliefNetwork. Check arena phase configuration."
        )

    from aragora.reasoning.crux_detector import CruxDetector
    detector = CruxDetector(network=network)
    analysis = detector.detect_cruxes(
        top_k=protocol.crux_finder_top_k,
        min_score=protocol.crux_finder_min_score,
    )

    counterfactuals = []
    if protocol.crux_finder_counterfactual_validation:
        for crux in analysis.cruxes:
            # compute_resolution_impact already does counterfactual analysis
            # per crux. Package it as Counterfactual-shaped record for the
            # receipt.
            counterfactuals.append({
                "claim_id": crux.claim_id,
                "condition": f"Resolve '{crux.statement}' to high confidence",
                "outcome_change": (
                    f"Reduces total network uncertainty by "
                    f"{crux.resolution_impact:.3f}"
                ),
                "likelihood": crux.uncertainty_score,
                "affected_claims": crux.affected_claims,
            })

    return CruxFinderResult(
        debate_id=debate_result.debate_id,
        question=question,
        analysis=analysis,
        counterfactuals=counterfactuals,
        agents=[a.name for a in arena.agents],
        rounds=protocol.rounds,
        raw_claims=[c.to_dict() for c in debate_result.all_claims],
        metadata={"mode": "crux_finder", "approach": "A"},
    )
```

**Step 4 — Add `build_proof_from_crux_finder` in `consensus.py`**

Mirror `build_proof_from_prover_estimator` (`consensus.py:997`). The proof's `final_claim` should be a stable sentinel indicating "no verdict by design", and `unresolved_tensions` carries the cruxes:

```python
def build_proof_from_crux_finder(result: "CruxFinderResult") -> ConsensusProof:
    """Build a ConsensusProof from a CruxFinderResult.

    Note: the proof exists for protocol compatibility. The canonical
    artifact for crux_finder mode is CruxReceipt (see gauntlet/).
    final_claim is set to a stable sentinel so downstream consumers
    that assume a verdict can detect and skip these proofs.
    """
    builder = ConsensusBuilder(
        debate_id=result.debate_id,
        task=result.question,
    )

    for crux in result.analysis.cruxes:
        claim = builder.add_claim(
            statement=crux.statement,
            author=crux.author,
            confidence=1.0 - crux.uncertainty_score,
        )
        for contester in crux.contesting_agents:
            builder.record_dissent(claim.claim_id, by_agent=contester)

    # Sentinel marker — NOT a verdict.
    builder.set_final_claim(
        "__CRUX_MAP__: no verdict by design; see CruxReceipt.cruxes"
    )
    builder.mark_unresolved_tensions(
        [c.statement for c in result.analysis.cruxes]
    )
    return builder.build()
```

**Step 5 — Wire dispatcher**

In whichever file dispatches on `protocol.consensus` (grep for `"prover_estimator"` to find it), add a parallel branch:

```python
elif protocol.consensus == "crux_finder":
    from aragora.debate.crux_mode import run_crux_finder, CruxFinderResult
    from aragora.debate.consensus import build_proof_from_crux_finder
    cf_result = await run_crux_finder(arena, protocol, question=task)
    proof = build_proof_from_crux_finder(cf_result)
    # Stash CruxFinderResult on the arena result so the receipt builder
    # can find it later. See Task A2.
    debate_output.crux_finder_result = cf_result
    return proof
```

**Step 6 — Unit tests**

Create `tests/debate/test_crux_mode.py` with:
- `test_crux_finder_literal_accepted`: `DebateProtocol(consensus="crux_finder")` does not raise.
- `test_crux_finder_result_serialization`: `CruxFinderResult.to_dict()` round-trips.
- `test_build_proof_from_crux_finder_sentinel`: Proof's `final_claim` contains `__CRUX_MAP__`.
- `test_crux_finder_produces_ranked_cruxes`: Integration with mock Arena populating a simple 4-node BeliefNetwork — detect_cruxes returns ≥1 crux.

Expected: all pass.

---

### Task A2 — `CruxReceipt` dataclass + SHA-256 signing + exporters

**Files:**
- Modify: `aragora/gauntlet/receipt_models.py` (add `CruxReceipt` dataclass)
- Modify: `aragora/gauntlet/receipt.py` (re-export)
- Modify: `aragora/gauntlet/receipt_exporters.py` (markdown + JSON exporters)
- Create: `aragora/gauntlet/crux_receipt.py` (builder and signer helper — optional, can live in `receipt_models.py` if small)

**Step 1 — Define `CruxReceipt`**

In `aragora/gauntlet/receipt_models.py`, after `DecisionReceipt`:

```python
@dataclass
class CruxReceipt:
    """A signed map of load-bearing disagreement.

    Distinct from DecisionReceipt: this artifact is explicitly NOT a
    verdict. It says 'here is where reasonable people diverge', with
    counterfactual evidence that each listed claim is load-bearing.
    """

    receipt_id: str           # e.g. "crux-<uuid8>"
    debate_id: str
    question: str
    timestamp: str            # ISO-8601 UTC
    agents: list[str]
    rounds: int

    # The map of disagreement
    cruxes: list[dict]                # CruxClaim.to_dict() entries
    convergence_barrier: float        # from CruxAnalysisResult
    counterfactuals: list[dict]       # Validation per crux
    recommended_focus: list[str]      # Claim IDs in priority order

    # Optional resolution strategies from CruxDetector
    resolution_strategies: list[dict] = field(default_factory=list)

    # Provenance
    raw_claims_hash: str = ""         # SHA-256 of sorted-keys raw_claims JSON
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        """SHA-256 over canonical-JSON, 16 hex chars. Matches ConsensusProof pattern."""
        import json, hashlib
        content = json.dumps(
            {
                "receipt_id": self.receipt_id,
                "debate_id": self.debate_id,
                "question": self.question,
                "timestamp": self.timestamp,
                "cruxes": self.cruxes,
                "convergence_barrier": self.convergence_barrier,
                "counterfactuals": self.counterfactuals,
                "recommended_focus": self.recommended_focus,
                "raw_claims_hash": self.raw_claims_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "debate_id": self.debate_id,
            "question": self.question,
            "timestamp": self.timestamp,
            "agents": self.agents,
            "rounds": self.rounds,
            "cruxes": self.cruxes,
            "convergence_barrier": self.convergence_barrier,
            "counterfactuals": self.counterfactuals,
            "recommended_focus": self.recommended_focus,
            "resolution_strategies": self.resolution_strategies,
            "raw_claims_hash": self.raw_claims_hash,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }
```

**Step 2 — Builder function**

In `aragora/gauntlet/receipt_models.py` (or a new `crux_receipt.py`):

```python
def build_crux_receipt(
    result: "CruxFinderResult",
    resolution_strategies: list[dict] | None = None,
) -> CruxReceipt:
    import uuid, json, hashlib
    from datetime import datetime, timezone

    receipt_id = f"crux-{uuid.uuid4().hex[:8]}"
    raw_hash = hashlib.sha256(
        json.dumps(result.raw_claims, sort_keys=True).encode()
    ).hexdigest()

    return CruxReceipt(
        receipt_id=receipt_id,
        debate_id=result.debate_id,
        question=result.question,
        timestamp=datetime.now(timezone.utc).isoformat(),
        agents=result.agents,
        rounds=result.rounds,
        cruxes=[c.to_dict() for c in result.analysis.cruxes],
        convergence_barrier=result.analysis.convergence_barrier,
        counterfactuals=result.counterfactuals,
        recommended_focus=result.analysis.recommended_focus,
        resolution_strategies=resolution_strategies or [],
        raw_claims_hash=raw_hash,
        metadata=result.metadata,
    )
```

**Step 3 — Markdown exporter**

In `aragora/gauntlet/receipt_exporters.py`, add:

```python
def crux_receipt_to_markdown(receipt: CruxReceipt) -> str:
    lines = [
        f"# Crux Map — {receipt.question}",
        "",
        f"**Receipt:** `{receipt.receipt_id}`  **Checksum:** `{receipt.checksum}`  ",
        f"**Debate:** `{receipt.debate_id}`  **Agents:** {', '.join(receipt.agents)}  **Rounds:** {receipt.rounds}",
        "",
        f"**Convergence barrier:** {receipt.convergence_barrier:.3f}  ",
        "*(higher = harder to reach consensus; cruxes below are the highest-leverage disagreement points)*",
        "",
        "## Cruxes",
        "",
    ]
    for i, crux in enumerate(receipt.cruxes, 1):
        lines += [
            f"### {i}. {crux['statement']}",
            f"- **Crux score:** {crux['crux_score']:.3f}",
            f"- **Influence:** {crux['influence_score']:.3f}  |  "
            f"**Disagreement:** {crux['disagreement_score']:.3f}  |  "
            f"**Uncertainty:** {crux['uncertainty_score']:.3f}",
            f"- **Contesting agents:** {', '.join(crux.get('contesting_agents', [])) or '—'}",
            f"- **Affected claims:** {len(crux.get('affected_claims', []))}",
            "",
        ]
    lines += ["---", f"_Generated by aragora crux-finder mode at {receipt.timestamp}_"]
    return "\n".join(lines)
```

**Step 4 — Tests**

In `tests/gauntlet/test_crux_receipt.py`:
- `test_crux_receipt_checksum_stable`: Same input → same checksum.
- `test_crux_receipt_checksum_changes_on_mutation`: Mutating any crux field changes checksum.
- `test_build_crux_receipt_from_result`: Round-trip from `CruxFinderResult`.
- `test_crux_receipt_markdown_contains_all_cruxes`: Every crux appears in output.

Expected: all pass.

---

### Task A3 — `aragora crux` CLI verb

**Files:**
- Create: `aragora/cli/commands/crux.py`
- Modify: `aragora/cli/commands/__init__.py` (register parser)
- Modify: `aragora/cli/__main__.py` (dispatch)

**Step 1 — CLI module**

Mirror `aragora/cli/commands/consensus.py`. Support:
- `aragora crux find "<question>"` — run the debate, output crux map to stdout (markdown) and optionally to a receipt file.
- `aragora crux validate <receipt-path>` — re-run counterfactual analysis on the receipt's cruxes against the original BeliefNetwork snapshot (if present) and report whether each crux is still load-bearing.
- Flags: `--top-k 5`, `--min-score 0.3`, `--agents <list>`, `--rounds 3`, `--receipt <out.json>`, `--format {markdown,json}`.

**Step 2 — Implementation sketch**

```python
# aragora/cli/commands/crux.py
def add_crux_parser(subparsers):
    parser = subparsers.add_parser(
        "crux",
        help="Find load-bearing disagreement points instead of a verdict.",
        description=(
            "Run a debate that surfaces cruxes — the specific claims where, if "
            "you flipped your belief on them, your overall conclusion would flip. "
            "Output is a signed crux receipt, not a verdict."
        ),
    )
    sub = parser.add_subparsers(dest="crux_command", required=True)

    find = sub.add_parser("find", help="Find cruxes for a question")
    find.add_argument("question")
    find.add_argument("--top-k", type=int, default=5)
    find.add_argument("--min-score", type=float, default=0.3)
    find.add_argument("--agents", nargs="+", default=["claude", "gpt-4o", "mistral-large"])
    find.add_argument("--rounds", type=int, default=3)
    find.add_argument("--receipt", help="Write signed receipt JSON to this path")
    find.add_argument("--format", choices=["markdown", "json"], default="markdown")

    validate = sub.add_parser("validate", help="Re-validate a crux receipt")
    validate.add_argument("receipt_path")

    return parser

async def run_crux_find(args):
    from aragora import Arena, Environment
    from aragora.debate.protocol import DebateProtocol
    from aragora.gauntlet.receipt_models import build_crux_receipt, crux_receipt_to_markdown

    protocol = DebateProtocol(
        consensus="crux_finder",
        crux_finder_top_k=args.top_k,
        crux_finder_min_score=args.min_score,
        rounds=args.rounds,
    )
    env = Environment(task=args.question)
    arena = Arena(env, agents=_resolve_agents(args.agents), protocol=protocol)

    cf_result = await arena.run_crux_mode()  # thin wrapper that calls run_crux_finder
    receipt = build_crux_receipt(cf_result)

    if args.format == "markdown":
        print(crux_receipt_to_markdown(receipt))
    else:
        import json
        print(json.dumps(receipt.to_dict(), indent=2))

    if args.receipt:
        from pathlib import Path
        Path(args.receipt).write_text(json.dumps(receipt.to_dict(), indent=2))
```

**Step 3 — Tests**

`tests/cli/test_crux_cli.py`:
- `test_crux_find_help_lists_subcommands`: `aragora crux find --help` works.
- `test_crux_find_runs_with_mocked_agents`: End-to-end with stub agents, produces markdown output.
- `test_crux_validate_reads_receipt`: Validate command reads a receipt JSON without crashing.

---

### Task A4 — Dogfood: 10 real-question live runs

**Files:**
- Create: `docs/dogfood/2026-04-crux-mode-runs.md`

**Step 1 — Run list**

Run `aragora crux find` on 10 questions spanning different domains:
1. "Should we migrate the orchestrator from async to structured concurrency?"
2. "Is the Nomic Loop worth continuing, or should we cut it?"
3. "Is our current auth-provider choice reversible?"
4. "Should aragora be open-source or source-available?"
5. "Does the KM Mound need sharding before 10M entries?"
6. ...(5 more drawn from recent founder/team disagreements)

**Step 2 — Capture**

For each run, record: `receipt_id`, `checksum`, number of cruxes returned, convergence_barrier, and a one-line founder assessment: "cruxes land on the real disagreement" / "cruxes miss the real disagreement".

**Step 3 — Decision gate**

After 10 runs, check the B trigger criteria (top of this doc). If any fires, open the B-track issues. If none fires in 30 days, mark B as "not needed".

---

## Acceptance Criteria

**Track A ships when all of these are true:**

1. `DebateProtocol(consensus="crux_finder")` constructs without error and is accepted by the dispatcher.
2. `build_proof_from_crux_finder` returns a `ConsensusProof` with `final_claim` containing the `__CRUX_MAP__` sentinel.
3. `CruxReceipt.checksum` is stable across serialization and changes on any field mutation.
4. `aragora crux find "<question>"` on a live debate (not mocked) produces markdown output with ≥1 crux on a topic with real disagreement.
5. The live-produced `CruxReceipt` saved via `--receipt` round-trips through JSON without losing the checksum.
6. `aragora crux validate <receipt>` reports counterfactual-stability for each crux.
7. New tests pass: `pytest tests/debate/test_crux_mode.py tests/gauntlet/test_crux_receipt.py tests/cli/test_crux_cli.py`.
8. No existing test regressions: `pytest tests/debate/ tests/gauntlet/ tests/cli/` stays green.
9. Dogfood log `docs/dogfood/2026-04-crux-mode-runs.md` has ≥10 entries within 2 weeks of landing.

---

## Risks & Mitigations

**Risk 1: `BeliefNetwork` population is gated on the `belief_analysis` phase being active.**
Mitigation: Task A1 Step 3's orchestrator explicitly checks for `debate_result.belief_network is None` and raises a clear error pointing at phase configuration. Don't silently fall back.

**Risk 2: Cruxes cluster below 0.3 on real questions, producing empty receipts.**
Mitigation: CLI `--min-score` is user-tunable. Dogfood Step 2 records crux score distribution; if scores cluster low across runs, B-trigger (c) fires and protocol shaping becomes justified.

**Risk 3: `CruxReceipt` positioning confuses users expecting a verdict.**
Mitigation: The `__CRUX_MAP__` sentinel in `ConsensusProof.final_claim` is detectable by downstream consumers. CLI output's opening line says "No verdict by design — this is a disagreement map." Receipt markdown's first heading is "Crux Map — <question>" not "Decision — <question>".

**Risk 4: Existing `winner_selector`/`analytics_phase` callers of `CruxDetector` break because crux detection now has two invocation sites.**
Mitigation: `CruxDetector.detect_cruxes` is pure (given a populated network). Calling it twice in a run is safe; the second call produces the same result. No change needed to existing callers.

**Risk 5: Scope creep toward B.**
Mitigation: B trigger criteria are explicit and measurable (top of doc). Do not add prompt shaping, per-round targeting, or Nomic Loop integration in this track even if they feel natural — they are B work.

---

## Follow-up (Track B — DEFERRED, gated on A dogfood signal)

Tracked as separate issues opened only when B trigger criteria fire:

- **B1** — Crux-shaping prompt templates in `aragora/debate/prompts/crux_prompts.py`. Direct agents to name their strongest opposing case and the one fact whose change would flip their conclusion.
- **B2** — Per-round claim targeting: after round N, use `CruxDetector.detect_cruxes(top_k=3)` on the in-progress network to generate round N+1's focus prompts.
- **B3** — Nomic Loop integration: emit `OutcomeFeedbackRecord` entries from high-resolution-impact cruxes; MetaPlanner uses them as goal-prioritization signal. Killer demo: the system identifies where its own reasoning is fragile.

---

## Quick Reference

| Track | Key files | Tests |
|-------|-----------|-------|
| A1: Consensus mode | `aragora/debate/protocol.py`, `consensus.py`, `crux_mode.py` | `tests/debate/test_crux_mode.py` |
| A2: CruxReceipt | `aragora/gauntlet/receipt_models.py`, `receipt_exporters.py` | `tests/gauntlet/test_crux_receipt.py` |
| A3: CLI verb | `aragora/cli/commands/crux.py`, `__init__.py`, `__main__.py` | `tests/cli/test_crux_cli.py` |
| A4: Dogfood | `docs/dogfood/2026-04-crux-mode-runs.md` | Manual — 10 runs, founder assessment |

**Estimate:** 3–5 engineering days for Tracks A1–A3; 2 weeks of founder-intermittent dogfood for A4.
