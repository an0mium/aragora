# Aragora Stop-Doing Ledger — March 2026

This ledger turns strategy into an explicit filter for roadmap and sprint work.
Its purpose is simple: if a project does not strengthen Aragora's current wedge,
the team should mark it as `defer` or `reject` instead of carrying it as
ambient "important later" work.

This document complements:

- [ROADMAP](../../ROADMAP.md)
- [COMPETITIVE_POSITIONING_2026_03](COMPETITIVE_POSITIONING_2026_03.md)
- [WHEN_TO_USE_ARAGORA_VS_EXECUTION_SUBSTRATES](WHEN_TO_USE_ARAGORA_VS_EXECUTION_SUBSTRATES.md)

## Current Wedge

Aragora's current wedge is:

**auditable multi-model execution and review for consequential engineering work,
with receipts, provenance, and truthful stopping behavior**

In product terms, near-term work should strengthen one or more of these:

- `aragora review` as a complete, trustworthy product path
- the inbox trust wedge as a second dogfoodable workflow
- operator-facing receipts, blocker visibility, and review evidence
- repeatable live demos and design-partner adoption

## How To Use This Ledger

For any proposed project, ask four questions:

1. Does it make `aragora review` or the inbox trust wedge more usable now?
2. Does it improve receipts, provenance, blocker truthfulness, or operator trust?
3. Does it shorten time-to-first-useful-result for a real design partner?
4. Does it generate proof that multi-model review beats a single strong model?

If the answer is "no" across the board, the default action is not "keep it on
the roadmap anyway." The default action is `defer` or `reject`.

## Decision Meanings

| Status | Meaning |
|---|---|
| `allow` | Actively strengthens the wedge now; can compete for near-term capacity |
| `defer` | Potentially valuable later, but not before the wedge is repeatable with external users |
| `reject` | Attractive but strategically dilutive in the current phase; do not start unless the wedge changes |

## Stop-Doing Ledger

| Work class | Status | Why | Re-open trigger |
|---|---|---|---|
| More provider breadth, agent-count bragging, or connector-count marketing as a headline | `reject` | Breadth is table stakes and makes Aragora sound interchangeable with worker substrates | Only re-open if a specific provider/channel is required to close a live design-partner workflow |
| Generic orchestration infrastructure without direct pull from review/inbox product paths | `reject` | Infrastructure without user pull increases complexity without sharpening the moat | Re-open only with clear evidence that the current wedge is blocked by the missing infrastructure |
| "Whole orchestra" default UX or large overlapping swarms for routine work | `reject` | The strategy is bounded delegation with explicit ownership, not complexity theater | Re-open only when measured quality gains beat lead-agent-plus-bounded-workers on real tasks |
| Marketplace, creator-economy, or community-template platform work | `reject` | Ecosystem surface area is not the beachhead and does not prove the control-plane wedge | Re-open after repeatable external usage and evidence of inbound ecosystem pull |
| ERC-8004 or cryptoeconomic productization as a near-term bet | `reject` | Accountability economics only matter after there is meaningful receipt volume and calibration data | Re-open after durable outcome labeling and real decision volume exist |
| Net-new channels or integrations that do not strengthen review or inbox trust workflows | `defer` | More surfaces widen the product before the core loop is obviously worth adopting | Re-open if a concrete design partner requires the integration for a live wedge workflow |
| Pentest, SOC 2 expansion, and enterprise packaging beyond keeping the path warm | `defer` | Important later, but design-partner proof and PMF closure are the actual gate today | Re-open after a repeatable live demo and active external design-partner demand |
| Cloud marketplace listings and procurement surface work | `defer` | Distribution polish does not matter before buyers want the core workflow | Re-open after external pull and a sales motion that is blocked on procurement channels |
| 10+ agent coordination, large-scale sharding, Kubernetes operator work, and scale-first infrastructure | `defer` | The current problem is product truthfulness and adoption, not throughput ceilings | Re-open after real usage shows the lead-agent-plus-bounded-workers pattern is no longer sufficient |
| Vertical packages for legal, medical, financial, or other industries | `defer` | Verticalization before core workflow proof fragments focus and multiplies claims | Re-open after the core engineering wedge is validated and a vertical shows concrete pull |
| Canvas/UI workbench ambitions that do not directly improve receipt, review, or blocker clarity | `defer` | A large visual shell can disguise rather than solve product-truth gaps | Re-open after current CLI/web flows are repeatedly used and the next bottleneck is operator comprehension |
| Example apps, demos, and docs that directly reduce time-to-first-useful-result for `aragora review` or inbox trust wedge | `allow` | These sharpen the wedge and improve design-partner readiness | Keep investing while they shorten activation or improve proof quality |
| Receipt visibility, review summaries, blocker truthfulness, merge/publish clarity, and operator-facing evidence | `allow` | This is the differentiated control-plane surface | Continue by default |
| Measurement of catch-rate delta, trust outcomes, calibration quality, and live workflow usage | `allow` | The wedge needs proof, not just narrative | Continue by default |

## Default Response Patterns

When a project does not strengthen the wedge, use explicit language:

- `reject`: "This increases generic substrate breadth without improving the auditable review/inbox wedge."
- `defer`: "This may matter later, but the re-open trigger is repeatable external usage of the current wedge."
- `allow`: "This improves the current wedge by increasing trust, operator clarity, or real workflow adoption."

## Exceptions

This ledger is not a ban on all adjacent work.

Exceptions are reasonable when:

- a bug outside the wedge blocks the wedge directly
- a customer/design partner requirement is concrete and immediate
- a compliance/security task is necessary to keep a live pilot running

The burden of proof stays on the proposer to show the wedge connection plainly.
