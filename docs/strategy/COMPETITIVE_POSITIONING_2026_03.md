# Aragora Competitive Positioning — March 2026

> This document augments, not replaces, the broader vision in
> [CANONICAL_GOALS](../CANONICAL_GOALS.md),
> [WHY_ARAGORA](../WHY_ARAGORA.md), and
> [COMMERCIAL_OVERVIEW](../COMMERCIAL_OVERVIEW.md). It adds competitive
> context and priority ordering based on the current agent-tool landscape.

## The Landscape Shift

Multi-model orchestration is now commodity.

Tools like OpenCode, Pi, and model-routing plugin ecosystems already provide:

- provider-agnostic model access
- task-type-based routing
- terminal and IDE-native execution
- extensible tool/plugin surfaces
- lightweight multi-agent coordination

Aragora should not try to win on those dimensions alone.

## What Aragora Should Actually Be

Aragora is best framed as an **auditable execution control plane for
AI-assisted work**.

That means:

- multiple models can contribute
- disagreement is surfaced, not hidden
- review and approval gates stay explicit
- receipts and provenance are first-class outputs
- automation stops truthfully when evidence is insufficient

This is a different category from "AI coding assistant" or "agent shell."

## Where Aragora Has A Real Wedge

### 1. Adversarial disagreement as evidence

Most tools treat model disagreement as a nuisance to route around.
Aragora should treat disagreement as data:

- where models converge after challenge, confidence is stronger
- where they diverge, the dissent trail tells the human where judgment is still needed

### 2. Receipts, provenance, and truthful gates

Aragora's strongest product distinction is not "many models" but:

- receipts
- provenance
- review outcomes
- merge gates
- truthful blocker handling

The system should always be able to answer:

- who said what
- what evidence was used
- why the system advanced or stopped
- what the next human action actually is

### 3. Control-plane coordination above worker runtimes

OpenCode, Pi, Codex, Claude Code, and similar tools can be excellent worker
runtimes. Aragora does not need to replace them.

The better strategy is to own the layer above them:

- policy
- routing
- bounded delegation
- review
- receipts
- publish and merge truthfulness

### 4. Long-term calibrated accountability

Aragora already has the ingredients for a deeper moat:

- ELO and calibration tracking
- receipts
- historical outcomes
- heterogeneous model comparison

That creates the foundation for evidence-based trust weighting and, later,
cryptoeconomic accountability.

## What Is Table Stakes Now

These are necessary, but no longer differentiators:

- multi-provider support
- plugin/extensibility stories
- generic "43 agent types" breadth
- broad connector counts
- generic workflow orchestration

If Aragora leads with these, it will sound interchangeable with stronger,
larger ecosystems.

## Beachhead

The most credible near-term beachhead is:

**auditable multi-model execution and review for consequential engineering work**

Why this works:

- engineering teams already use AI tooling
- PR review and execution quality are measurable
- receipts and blocker truthfulness are valuable immediately
- the repo now contains real evidence from queue-produced work, not just demos

## First-Deal Decision Ownership

The first deals will usually have four distinct decision owners, even if a
smaller company compresses them into two people:

| Role | Owns this decision | What Aragora must prove |
|------|--------------------|-------------------------|
| Buyer | Is this worth budget and organizational attention right now? | Clear ROI wedge, narrow pilot, fast time-to-first-receipt |
| Daily user | Will I actually run this workflow every week? | Low-friction operator path, useful summaries, precise next actions |
| Evaluator | Do the receipts, dissent, and controls meet the bar for consequential work? | Provenance, review evidence, explicit gates, truthful stopping |
| Blocker | Is there any security, legal, procurement, or policy issue that still stops rollout? | Exact blocker handling, deployment answers, audit/export story |

If Aragora collapses these roles into a generic "champion," the team will
misread where a deal is stuck. Product and GTM should track them separately and
build artifacts for each.

## Competitor Map

| Dimension | Aragora | LangGraph/CrewAI | OpenCode/Pi |
|-----------|---------|------------------|-------------|
| Multi-model routing | yes | yes | yes |
| Adversarial debate | yes, core primitive | mostly cooperative | no |
| Decision receipts | yes, explicit output | logging-first | no |
| Calibration tracking | yes | limited | no |
| Truthful blocker handling | yes | uneven | uneven |
| Worker-runtime breadth | secondary | strong | strong |
| Primary moat | control-plane truthfulness | orchestration ecosystem | lightweight execution |

Aragora should not compete on generic orchestration breadth. It should compete
on decision quality, provenance, and accountable execution.

## ERC-8004 And Cryptoeconomic Accountability

This is not a sideshow, but it is not the beachhead.

ERC-8004-style staking and identity become strategically valuable only after
Aragora has enough real decision volume to justify:

- per-model, per-domain track records
- durable outcome labeling
- reputation weighting
- stronger external trust guarantees

So the correct sequence is:

1. generate real decision and execution data
2. preserve receipts and outcomes truthfully
3. calibrate model reliability empirically
4. later, make that accountability economically meaningful

## Current Priority Order

### P0

- make the product surface reflect the real wedge
- keep unattended execution truthful
- make `aragora review` / execution flows feel like product, not internal plumbing

### P1

- strengthen audit-ready PR and review workflows
- expose routing, review, and blocker evidence cleanly to operators
- validate with real external users on consequential tasks

### P2

- unify the full idea-to-execution workbench
- turn upstream ideas/goals/actions into the default shell around the control plane

## 90-Day Execution Stack

### Weeks 1-4

- make `aragora review` feel like a complete product path
- keep unattended execution and merge gating truthful
- tighten receipt, review, and blocker summaries for operators
- package first-deal artifacts by owner: buyer brief, daily-user workflow, evaluator evidence pack, blocker FAQ

### Weeks 5-8

- get 3-5 external users through real review and execution flows
- measure single-model vs multi-model quality deltas on real work
- publish concrete case studies with findings and receipts

### Weeks 9-12

- package compliance-ready artifacts around the real review path
- prepare pentest and audit work only after the beachhead flow is repeatedly used
- treat cryptoeconomic accountability as follow-on leverage, not the first sale

## Key Metrics To Track

| Metric | Why it matters |
|--------|----------------|
| Active deals with named buyer, daily user, evaluator, and blocker | shows whether deal ownership is explicit instead of assumed |
| External users running `aragora review` | proves the beachhead is real |
| Receipts generated per week | measures decision volume and provenance capture |
| Live multi-model debates per week | measures real usage, not surface clicks |
| Bug/finding catch-rate delta vs single model | proves the epistemic wedge |
| Time from install to first useful result | shows whether the path is productized |
| Outcome-labeled calibration samples by domain | determines whether long-term accountability is becoming real |

## What This Means For Development

Build product surface and control-plane truthfulness, not more generic substrate.

The right next steps are things like:

- clearer review and receipt UX
- stronger publish / merge visibility
- truthful stage transitions
- better operator-facing summaries

The wrong next steps are:

- more generic orchestration infrastructure without user pull
- selling provider breadth as if it were a moat
- building huge agent orchestras before the control plane is unquestionably truthful

## Simple Strategic Test

If a user asks, "Why not just use OpenCode or Pi plus plugins?" the answer
should not be "because Aragora supports more models."

The answer should be:

**Because Aragora governs AI-assisted execution with receipts, review, provenance,
and truthful stopping behavior.**

That is the category worth owning.
