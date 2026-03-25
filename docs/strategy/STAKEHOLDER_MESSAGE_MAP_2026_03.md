# Aragora Stakeholder Message Map

Last updated: 2026-03-25

This document translates Aragora's current positioning into stakeholder-specific
message discipline for sales, partnerships, design-partner outreach, and
technical evaluation.

It should be read alongside:

- `docs/strategy/COMPETITIVE_POSITIONING_2026_03.md`
- `docs/strategy/WHEN_TO_USE_ARAGORA_VS_EXECUTION_SUBSTRATES.md`
- `docs/outreach/STAKEHOLDER_NARRATIVE_VARIANTS.md`

## Core Category Claim

Aragora is an **auditable execution control plane for AI-assisted work**.

The category claim stays fixed across stakeholders. What changes is the problem
framing, proof surface, and first workflow to land.

## Messaging Rules

- Lead with consequential work, not general-purpose AI productivity
- Lead with receipts, review, provenance, and truthful gating
- Treat worker runtimes as complements and substrates, not enemies
- Use model breadth as supporting evidence, never as the moat
- Do not promise full autonomy; promise bounded execution under explicit policy

## Stakeholder Matrix

| Stakeholder | Primary pain | Promise | Proof surface | Likely objection | Best first workflow |
|---|---|---|---|---|---|
| Founder | AI increases velocity but also hidden decision risk | More leverage without losing accountability | Receipt trail, dissent visibility, bounded execution | "Is this just orchestration overhead?" | High-value founder-owned review path |
| Operator | Ad hoc AI use creates inconsistent workflows and unclear ownership | Repeatable operations with explicit terminal states | Queueing, receipt-before-action, blocker summaries | "Will this add friction to the team?" | Recurring triage, review, or approval workflow |
| Security reviewer | Agents are opaque and hard to constrain | Governable AI-assisted execution with audit evidence | Approval gates, provenance, self-hosting, compliance artifacts | "How do you prevent silent unsafe actions?" | Review-heavy workflow with human approval |
| Technical buyer | Existing agent tools already handle execution | Control plane above execution substrates | Debate, receipts, calibration, API and SDK surfaces | "Why not just use Codex, Claude Code, or LangGraph?" | Consequential engineering review path |

## Objection Handling

### "Why not just use a strong single model?"

Because the problem is not only generation quality. It is whether the team can
inspect disagreement, preserve provenance, and justify why a decision advanced
or stopped. Strong models can execute. Aragora governs.

### "Why not just use OpenCode, Pi, Codex, or Claude Code?"

Those are excellent execution substrates. Aragora sits above them when the work
needs multi-model challenge, receipts, review gates, and truthful blocker
handling.

### "Is this too heavy for day-to-day use?"

Not if it is attached to the right workflow. Aragora should be introduced where
a wrong answer is expensive, a review trail matters, or unattended execution
needs honest terminal states.

### "Are you selling compliance software?"

No. Compliance artifacts are a consequence of the control-plane design. The
beachhead is consequential review and execution, with security and compliance as
amplifiers for buyers that need them.

## Proof Priority By Stakeholder

### Founder

1. Explain the leverage story
2. Show one receipt that demonstrates accountability
3. Show how bounded execution increases trust rather than removing humans

### Operator

1. Show a workflow trigger and terminal state
2. Show how blockers and handoffs are made explicit
3. Show that the next action is lower-friction, not higher-friction

### Security Reviewer

1. Show trust boundaries and approval points
2. Show provenance and receipt artifacts
3. Show deployment and control options that fit the environment

### Technical Buyer

1. Show where Aragora sits relative to existing agent tools
2. Show the review and evidence layer, not generic orchestration
3. Show how it plugs into current engineering systems

## Anti-Patterns

- Do not pitch Aragora as "many agents in one place"
- Do not center connector count, provider count, or workflow template count
- Do not imply that disagreement disappears; show that it is made legible
- Do not sell cryptoeconomic accountability before the buyer accepts the
  control-plane wedge

## Recommended Language

Preferred phrases:

- auditable execution control plane
- receipt-before-action
- adversarial review
- provenance and dissent trail
- truthful stop/go gating
- bounded delegation

Avoid as lead phrases:

- model marketplace
- orchestration framework
- autonomous swarm platform
- AI employees
- no-human-in-the-loop automation

## Strategic Summary

Each stakeholder should hear a different reason to care, but the same reason to
believe:

**Aragora makes consequential AI-assisted work governable by producing receipts,
preserving provenance, surfacing disagreement, and stopping truthfully when
confidence is not earned.**
