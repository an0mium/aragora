# Design-Partner Onboarding Readiness Checklist

Last updated: 2026-03-25

## Goal

Define the minimum bar Aragora must satisfy before inviting the next design
partner into the live founder loop.

This checklist is intentionally stricter than an internal dogfood pass and
intentionally narrower than enterprise readiness. The question is not "is the
whole platform complete?" The question is "can a founder-led live session run
truthfully, produce value quickly, and end with a credible next step for the
partner?"

## Scope And Assumptions

- This is for a founder-led live session, not a fully self-serve onboarding
  funnel.
- The session uses the **live founder loop** as the primary workflow.
- The inbox trust wedge may be shown only as a secondary proof, not as the
  onboarding dependency.
- Manual founder guidance is allowed; hidden product rescue is not.
- The bar is one trustworthy external design-partner session on current `main`,
  not broad PMF proof or enterprise certification.

## Exit Rule

Do not invite the next design partner until every **must-pass** item below is
green with linked evidence. If a must-pass item fails, record the exact blocker,
owner, and next verification step instead of widening scope.

## Must-Pass Checklist

### 1. Live founder-loop execution is still proven on current `main`

- [ ] Run the canonical founder-loop command on current `main` and capture the
  exact command transcript.
- [ ] Complete **5 consecutive live runs** without hidden manual rescue.
- [ ] Keep runtime inside the current demonstrated band or explain the
  regression explicitly if it widens beyond it.
- [ ] Confirm each run ends in one of two truthful states only:
  - useful result delivered
  - direct blocker with precise stop reason

Evidence:
- command transcript
- run times for all 5 runs
- receipt paths or URLs for all 5 runs

### 2. Readiness and fail-closed behavior are externally legible

- [ ] Readiness state is visible before the session starts.
- [ ] Provider and credential state are explicit; no ambient shell magic is
  required to explain why the run can proceed.
- [ ] If the environment is not ready, Aragora fails closed quickly with a
  direct reason rather than silently dropping to demo behavior.
- [ ] The founder can show the partner where the system reports readiness and
  what "not ready" looks like.

Evidence:
- readiness output or screenshot
- one verified failure-mode example with the exact surfaced message

### 3. Receipt and result visibility work on product surfaces

- [ ] Every live run emits a structured receipt that can be inspected and
  verified.
- [ ] The resulting receipt is visible on at least one product surface the
  partner can inspect during the session.
- [ ] The founder can move from completed run to visible receipt/result without
  ad hoc spelunking.
- [ ] Share-link or API visibility is stable enough to use in a live demo.

Evidence:
- receipt verification output
- screenshot or URL of the visible receipt/result surface

### 4. The session reaches first value fast enough to feel productized

- [ ] Time from fresh session start to first useful result is consistently short
  enough for a live call.
- [ ] The founder has one canonical prompt/topic that reliably demonstrates the
  wedge.
- [ ] The partner can substitute their own consequential question without
  breaking the flow contract.
- [ ] Operator noise is bounded: logs, warnings, or summaries do not force the
  founder to explain away obvious product roughness mid-demo.

Evidence:
- timed rehearsal from fresh start
- canonical demo prompt
- one partner-style custom prompt rehearsal

### 5. The product story matches the actual behavior

- [ ] The founder can explain, in one short paragraph, why Aragora is better
  than a single execution substrate for this session.
- [ ] The demonstrated wedge is receipts, provenance, disagreement, and truthful
  stopping behavior, not generic model breadth.
- [ ] Known limitations are stated plainly before or during the session when
  relevant.
- [ ] The product surface shown in the session matches the current roadmap truth
  and does not imply enterprise readiness that does not exist yet.

Evidence:
- demo script or talk track
- list of limitations the founder will say out loud

### 6. Partner-safe operating discipline exists

- [ ] There is a single named founder/operator for the session.
- [ ] There is a session checklist for preflight, live operation, and follow-up.
- [ ] The founder knows the fallback path if the primary provider or workflow
  fails during the call.
- [ ] A post-call artifact is defined: receipt pack, notes, blockers, and next
  action.

Evidence:
- founder session runbook
- fallback path note
- post-call artifact template

## Should-Pass Checklist

These do not block the invite on their own, but they materially improve the
quality of the session and should be pushed toward green.

- [ ] Inbox trust wedge has at least one real dogfood proof on a live inbox.
- [ ] Fresh-user onboarding is timed and documented under 10 minutes.
- [ ] A second operator can reproduce the session without private founder
  context.
- [ ] The partner can access a small proof pack after the call without manual
  reconstruction.

## Evidence Pack Required Before Invite

Before sending the invitation, assemble one compact evidence pack containing:

- latest 5/5 founder-loop proof
- readiness screenshot or transcript
- one verified receipt and one visible product-surface result
- canonical demo prompt and fallback prompt
- known limitations / truthful caveats
- named owner for the live session

If this pack cannot be assembled in under 15 minutes, readiness is still too
fragile for the next design-partner invite.

## Explicit Non-Gates

The following are important, but they are **not** blockers for the next
founder-led design-partner session:

- full enterprise hardening
- pentest completion
- SOC 2 completion
- broad connector coverage
- full self-serve onboarding for every user type

Those remain downstream of design-partner validation, not prerequisites for it.

## Open Questions

- Should the next design partner see only the founder loop, or should the call
  also include the inbox trust wedge as a second proof?
- What exact runtime ceiling still feels acceptable on a live call: 60 seconds,
  90 seconds, or "truthful if longer"?
- Which product surface is the canonical receipt/result view during the call:
  CLI, dashboard, or share link?

## Next Actions

1. Rehearse the founder loop on current `main` and collect the 5-run evidence
   pack.
2. Write the founder session runbook that maps directly to this checklist.
3. Dogfood the inbox trust wedge separately so it can be shown as optional
   follow-on proof, not onboarding risk.
4. Invite the next design partner only after all must-pass items are green with
   evidence links.
