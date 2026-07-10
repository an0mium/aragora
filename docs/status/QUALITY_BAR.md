# Quality Bar — the ≥8/10 gate for external human proof

**Founder decision (2026-07-10, binding):** no external human is recruited for
the #8858 outsider verification run until the project measures **≥8/10 on all
dimensions below**. Founder's holistic assessment at decision time: **~4/10**.
Until the bar is met, the instrument for the outsider experience is a
**simulated stranger run**: a clean VM/container, public docs only, no repo
context, scripted persona, every failure recorded before anything is fixed.

Rationale: a real stranger's first impression is a scarce, unrepeatable
resource. The bar prevents spending it on a 4/10 experience. The counterweight
(recorded so it is not forgotten): internal scores drift optimistic, so every
dimension below is tied to a repeatable instrument, not a self-assessment, and
simulated stranger runs must stay adversarial (the runner's incentive is to
find failures, not to succeed).

Scores are updated only with a fresh instrument run linked in the row.
Baseline scores anchor to the 2026-06-12 codebase-health audit
(`docs/audits/2026-06-12-codebase-health-audit.md`) and to measured settlement
telemetry from 2026-07-09/10; they await founder calibration.

| # | Dimension | Instrument (repeatable) | Baseline | Target (what 8 means) |
|---|-----------|-------------------------|---------|------------------------|
| 1 | First-hour stranger experience | Simulated stranger run: clean VM, README/quickstart only, time-to-first-verified-receipt + failure count | **6** ([run 1, 2026-07-10](../artifacts/2026-07-10-stranger-sim-run-1.md)) | 8: zero blocking failures; verified receipt in <15 min; no undocumented step |
| 2 | Packaging & install truthfulness | `pip install aragora` → `aragora demo --offline --receipt` → `aragora receipt verify` on clean env; INSTALL.md vs pyproject consistency diff | **5** ([run 1](../artifacts/2026-07-10-stranger-sim-run-1.md)) | 8: round trip green on macOS+Linux clean envs; zero doc/pyproject contradictions |
| 3 | Docs coherence / single positioning | Docs-consistency CI + positioning-drift scan (README / WHY_ARAGORA / COMMERCIAL_OVERVIEW / EXTENDED_README / CLAUDE.md) | **4** ([run 1](../artifacts/2026-07-10-stranger-sim-run-1.md): 3 verify surfaces, 3 stories) | 8: one canonical positioning; zero contradictory capability or metric claims |
| 4 | Receipt verifiability end-to-end | Fresh production receipt → `aragora-verify` offline, including signature (needs #8809 signing path) | ~5 (unsigned prod receipts) | 8: signed prod receipt verifies offline incl. issuer authenticity, documented key rotation |
| 5 | Settlement signal integrity | Quorum family counting rate; fake-failure (cancellation) incidents/week; `DOES NOT count ()` occurrences | ~4 (claude family invisible; ~6 cancel incidents in one session) | 8: all configured families count when they review; <1 unexplained terminal-cancel/week (#9129, #9133) |
| 6 | Macro-architecture | Import-cycle ratchet count; `aragora.server` imported-by count (shrink-only baselines) | 4 (audit) | 8: cycles <30; server imported-by ≤5; no new layering violations for 30 days |
| 7 | Test trustworthiness | Shard-pollution incidents/week on PR CI; randomized-order lane green streak | ~5 (known pollution class blocking PRs) | 8: zero passes-alone-fails-in-shard incidents in 14 days; randomized lane green |
| 8 | Core workflow utility | `aragora review-pr` / `gauntlet` / `ask` succeed with ONE provider key on a clean env, useful output, no crash | ~5 | 8: all three green on clean env; outputs judged useful in the weekly digest review |

## Operating consequences

- The 30-day plan's W1 exit criterion "#8858 artifact w/ named human" and the
  Jul 16 replan trigger are **deferred** (amended in the same PR as this doc).
  W1's replacement gate: first simulated stranger run executed and its failure
  list converted to issues.
- The EU AI Act bundle (W3, Jul 30) does **not** depend on the outsider run
  and keeps its date.
- Dimensions 5–7 are already in flight (#9129, #9133, swarm-status salvage,
  existing ratchets). Dimensions 1–3 and 8 are the gap: they are product/docs
  work the work-mix governor should prefer.
- This bar governs **recruiting an external human**. It does not gate internal
  dogfooding, receipts, or the compliance bundle.
