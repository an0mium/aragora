# Founder Risk Register

Last updated: 2026-03-25

This is the founder-level risk register for the current PMF window.
It sits below the roadmap and above issue-by-issue execution.

Scope:

- protect the proven founder loop from false confidence
- keep the inbox trust wedge and design-partner motion honest
- force direct responses when product truth, onboarding, or operating repeatability drift

This is not the enterprise assurance register.
It is the short-horizon founder operating register for March-April 2026.

## Current Operating Context

- The canonical founder loop is proven repeatable on `main`:
  5/5 consecutive live runs on March 24, 2026, with valid receipts visible via API/dashboard.
- The next gates are:
  - dogfood the inbox trust wedge on a real Gmail inbox
  - prepare for design-partner outreach with a repeatable live demo
- The main failure mode now is not "missing infrastructure."
  It is shipping or selling ahead of what the product can truthfully prove end to end.

## How To Use This Register

- Review it before weekly PMF planning.
- Re-check it after every failed live dogfood run or partner demo.
- If a tripwire is crossed, treat the linked mitigation as the default next action.
- Do not offset a red tripwire with unrelated infrastructure wins.

## Register

| Area | Founder risk | Why it matters now | Tripwire | Direct mitigation |
|---|---|---|---|---|
| PMF | Internal proof outpaces external pull. | The founder loop is proven, but the second workflow and design-partner repetition are not yet proven live. | No real inbox-wedge dogfood run with receipt by the end of the next working week, or no design partner reaches first receipt within 14 days of outreach starting. | Freeze net-new roadmap expansion. Run only founder-loop, inbox-wedge, and partner-demo blockers. Rebuild the live blocker list from observed failed runs and partner friction, then route only those fixes through the pipeline. |
| Trust | Product claims outrun truthful product behavior. | Aragora's wedge is receipts, provenance, and truthful stopping. A single demo-shaped shortcut damages the whole story. | Any silent fallback to demo, any live run without a reviewable receipt, any UI/API surface claiming readiness that cannot be reproduced with an exact command transcript, or more than one manual rescue in a week for the default demo path. | Fail closed by default. Narrow external claims to the founder loop and any inbox-wedge behaviors that have direct live proof. Make every stop reason and missing prerequisite explicit on product surfaces before adding polish work. |
| Onboarding | First-use success depends on founder intervention. | Quickstart is live, but design partners will judge the product on time-to-first-useful-result, not backend completeness. | A fresh user cannot reach first receipt from documented steps in 10 minutes, readiness does not explain the next missing credential or action clearly, or partner setup requires ad hoc Slack/terminal coaching twice in a row. | Treat onboarding as a product bug surface, not documentation debt. Put readiness first, collapse the default path to one recommended entry point per workflow, and patch every failed setup with exact copy, command, or UI guidance before recruiting more users. |
| Ops | Founder demos and dogfood runs are not operationally repeatable. | One 5/5 proof set is strong evidence, but PMF depends on weekly repeatability under normal operating conditions. | Weekly live founder-loop success drops below 80%, runtime expands above 90 seconds for the default proof path twice in a week, or a failed run cannot be reconstructed from exact command/output evidence. | Run a fixed weekly dogfood cadence with command transcripts and receipt capture. Keep the repair queue bounded to reproducible failures only. Do not let enterprise assurance, cleanup, or speculative infra work displace operational proof maintenance. |
| Technical cohesion | Shipped slices do not add up to one believable product path. | Quickstart, prompt-to-spec, receipts, truth-seeking, onboarding, and the inbox wedge are all stronger individually than they are as one cohesive product story. | A partner demo requires jumping across more than two disconnected surfaces, the same capability is described differently across docs/UI/CLI, or the main user journey needs three or more caveats to explain what is and is not live. | Enforce one default story per wedge. Unify naming, proof points, and result surfaces around the canonical flow. Prefer deleting or hiding orphaned paths from the founder narrative over carrying extra surface area that weakens coherence. |

## Tripwire Response Order

When multiple tripwires fire at once, respond in this order:

1. Trust
2. PMF
3. Onboarding
4. Ops
5. Technical cohesion

Reason:
If trust breaks, sales and dogfood evidence become unreliable.
If PMF is unclear, optimization work is premature.
If onboarding is weak, external usage data will be biased by founder rescue.
If ops are unstable, learning velocity collapses.
If cohesion is weak, the product still sounds larger than it feels.

## Immediate Founder Rules

- Do not market enterprise readiness ahead of weekly PMF proof.
- Do not treat inbox-wedge code-complete status as proof until a real inbox run produces receipts and reviewable actions.
- Do not add new workflow surfaces if the current default path still needs founder narration to succeed.
- Do not let compliance packaging or infrastructure breadth substitute for recurring design-partner use.

## Linked Source Documents

- [PMF Dogfood Execution Plan](./PMF_DOGFOOD_EXECUTION_PLAN.md)
- [Next Steps (Canonical)](../status/NEXT_STEPS_CANONICAL.md)
- [PMF Scorecard](../status/PMF_SCORECARD.md)
- [Design Partner Program](../status/DESIGN_PARTNER_PROGRAM.md)
