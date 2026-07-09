# The Maximalist Vision — Tool → Organization Substrate

**Status:** Canonical home of the long-horizon vision. Moved here from the README on
2026-07-08 (founder decision: *focus, don't amputate* — the README carries the earned
near-term claim; this document keeps the full arc, built out piece by piece, section
by section, in integrated fashion).
**Related:** `docs/CANONICAL_GOALS.md`, `docs/vision/ADJACENT_POSSIBLE.md`,
`docs/plans/2026-07-08-vision-audit-and-work-mix-governor.md` (§7 staged-integration
track), capability checkpoints CP-1..5 (README, Discipline section).

---

## The long arc — five stages

Aragora is designed to climb five stages: **Tool → Teammate → Foreman → Chief of
Staff → Organization Substrate** — from bounded useful results today to a cross-org
agentic operating system.

```
Tool ──▶ Teammate ──▶ Foreman ──▶ Chief of Staff ──▶ Organization Substrate
 one      assists      runs bounded   plans & routes      agents + humans as
 review   a human      backlogs       across backlogs     co-equal consumers
 +receipt              unattended     with receipts       on one runtime truth
(today's wedge) ───────────────────────────────────────▶ (long-horizon thesis)
```

The wedge and the vision are the same system at different stages. Each stage is
entered only when the previous stage's claims are externally provable — the vision is
**bounded by checkpoints, not open-ended**, and failing a checkpoint downscales the
next investment; it does not kill the vision.

## The eight foundational pillars of the substrate

① **Adversarial heterogeneous consensus + crux-finding** — disagreement between
independent frontier models is the signal; the load-bearing dissent is named, not
averaged away.

② **Reliable autonomous execution** — contracts, preflight, repair, fail-closed
escalation; bounded backlogs run unattended with receipts.

③ **A unified DAG** — ideas → goals → actions → orchestration, with optional
interactivity at every node.

④ **Permissioned, portable, attributable memory** — across repos, docs, APIs, chat,
inbox, and telemetry; institutional knowledge that survives sessions and staff.

⑤ **Cryptographic receipts & auditability** — with eventual proof-carrying code;
every consequential decision leaves a verifiable, dissent-preserving artifact.

⑥ **SMB operator leverage** — intent → action in under 10 minutes; the substrate is
not enterprise-only.

⑦ **Self-improvement on the *same* substrate as user-facing work** — Aragora governs
Aragora's own development; the dogfood loop is the standing demonstration and the
calibration-data flywheel.

⑧ **Agents and humans as co-equal consumers** — parity surfaces backed by one
runtime truth.

## Skin in the game — the epistemic enforcement layer

The substrate's end-state epistemics are not merely reputational; they are
**staked**. The ERC-8004 track (`aragora/blockchain/`) exists for this: claims →
stakes → resolution against external ground truth → reputation deltas → dispatch
eligibility. An agent that is wrong pays; an agent that dissents correctly compounds.
Near-term, public anchoring is served by Sigstore Rekor (ODR-7, #8231) and reputation
by ELO/Brier calibration; the staking layer activates at checkpoint CP-4 (a
reputation delta changing real dispatch), when it can be wired to live debate
outcomes rather than run as scaffolding. This is deliberate sequencing, not
deprecation.

## The frontier tracks

- **Agent Civilization Substrate (AGT-01..06):** live CruxDetector, an A2A consumer
  surface (agents register/discover/transact/consume receipts), Manifold prediction
  calibration, synthetic GitHub markets with verifiable resolution, ERC-8004
  reputation flow, and the VIAH (Verifiable Improvements per Agent-Hour)
  self-justification metric.
- **Epistemic CI / Decision Integrity Core (DIC-13..22):** receipts beyond debates —
  executable claims with freshness SLAs, proof-carrying code units that fail closed
  when assumptions decay, epistemic-decay repair proposals, a read-only
  organizational truth map.
- **Verticals** (`aragora/verticals/`): healthcare, legal, financial, accounting
  specialists behind the receipt wedge — activated per-vertical when a pilot signs.
- **Marketplace** (`aragora/marketplace/`, skills): the distribution surface once
  external users exist.
- **Communication surfaces** (`aragora/broadcast/`, channels): decisions delivered
  where organizations actually live.

## How the vision is built — integrate, don't archive

Every dormant engine in the codebase is either on the near-term product target list
(crux detector, Pareto router, inbox web GUI) or holds a named integration criterion
in the staged-integration track (`docs/plans/2026-07-08-vision-audit-and-work-mix-governor.md`
§7). Nothing is cut for being early; things are sequenced by proof. The near-term
focus — the signed, dissent-preserving, offline-verifiable Decision Receipt — is
pillar ⑤ carrying the flag for the other seven until each earns its activation.
