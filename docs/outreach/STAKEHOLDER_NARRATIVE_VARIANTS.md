# Aragora Stakeholder Narrative Variants

Last updated: 2026-03-25

This document gives stakeholder-specific narrative variants that keep the same
core positioning:

**Aragora is the Decision Integrity Platform for AI-assisted work.**

It is not just multi-model execution. It is the control plane that adds
adversarial review, decision receipts, provenance, and truthful stop/go gates
before consequential work ships.

---

## Shared Message Spine

Every variant should preserve these truths:

- Aragora sits above worker runtimes such as Codex, Claude Code, OpenCode, and Pi
- The wedge is not model breadth; it is receipts, review, provenance, and
  truthful blocker handling
- Aragora helps teams move faster on consequential work by making delegation
  governable rather than opaque
- The output is a decision receipt, not just an answer

---

## Founder Variant

### One-line framing

**Aragora lets a small team operate with board-room rigor without adding board-room drag.**

### Core narrative

You are already using AI to move faster, but speed stops being an advantage the
moment nobody can explain why an important decision was made. Aragora gives a
founder-led team a way to delegate more work to AI without creating hidden risk.
It runs adversarial review across multiple models, surfaces disagreement before
it becomes an incident, and produces a receipt you can share with customers,
investors, auditors, or your own team.

The real value is leverage with control. Instead of hiring a larger staff just
to create decision coverage, Aragora gives your existing team a repeatable way
to vet specs, PRs, vendor choices, policy changes, and other consequential
calls. You move faster because review is structured, not because governance is
removed.

### What the founder cares about

- Shipping faster without accumulating invisible AI risk
- Getting more leverage from a small team
- Having a credible story for customers, investors, and diligence
- Avoiding operational chaos from ungoverned agent use

### Proof points to emphasize

- Multi-agent debate with explicit dissent, not a single opaque model opinion
- Decision receipts with provenance and confidence trails
- Bounded execution with receipt gates and truthful stopping behavior
- Real review and execution flows already wired end to end on `main`

### Founder CTA

Start with one recurring high-value workflow where speed matters but an
unexplained mistake would be expensive.

---

## Operator Variant

### One-line framing

**Aragora turns ad hoc AI usage into a repeatable operating system for consequential work.**

### Core narrative

Operators do not need more model demos. They need a workflow that runs the same
way on Monday night as it does during an incident on Friday afternoon. Aragora
is useful when work has to be reviewed, routed, approved, and handed off without
losing the evidence trail. It takes inputs from channels your team already uses,
runs structured multi-agent vetting, and returns a receipt that makes the next
action obvious: proceed, escalate, or stop.

This matters because most AI usage fails operationally before it fails
technically. Ownership gets fuzzy, outputs are hard to audit, and people cannot
tell whether a task completed cleanly or merely sounded confident. Aragora adds
truthful stage transitions, explicit blockers, and publish/merge gates so teams
can operationalize AI without pretending uncertainty does not exist.

### What the operator cares about

- Clear ownership and low-friction handoffs
- Repeatable workflow behavior across Slack, GitHub, inbox, and APIs
- Faster throughput on bounded recurring work
- Honest terminal states instead of false success

### Proof points to emphasize

- Receipt-before-action workflow design
- Queue, supervisor, lease, and merge-policy patterns for bounded execution
- Operator-facing summaries for consensus, blockers, and next actions
- Exportable artifacts for reporting and postmortems

### Operator CTA

Pick one bounded operational workflow with a clear trigger, owner, and success
condition, then make Aragora the review and receipt layer for that path.

---

## Security Reviewer Variant

### One-line framing

**Aragora is the governance layer that makes AI-assisted execution reviewable, constrainable, and auditable.**

### Core narrative

Security teams should be skeptical of any agent platform that claims autonomy
without showing its control surfaces. Aragora's value is that it does not ask
you to trust a model. It gives you explicit review stages, provenance,
cryptographic receipts, bounded delegation, and truthful stopping behavior when
the evidence is weak or the state is ambiguous.

The system is strongest when presented as a control plane above worker runtimes,
not as an unbounded autonomous actor. It preserves who said what, what evidence
was used, how consensus or dissent formed, and why a workflow advanced or
stopped. That aligns with how security and compliance teams actually evaluate
risk: not "was the model smart?" but "what constrained the action, and what
artifact exists for review afterward?"

### What the security reviewer cares about

- Clear trust boundaries and bounded execution
- Explicit approval and merge gates
- Tamper-evident audit trails and provenance
- Deployment options that fit regulated or isolated environments

### Proof points to emphasize

- SHA-256 decision receipts and exportable compliance artifacts
- Truthful blocker handling rather than hidden retries or false positives
- Enterprise controls: SSO, RBAC, encryption, multi-tenancy, offline/self-hosted
- EU AI Act, SOC 2, HIPAA, and governance-aligned artifact generation

### What not to say

- Do not lead with "43 agent types" or generic orchestration breadth
- Do not imply Aragora removes human accountability
- Do not position it as open-ended autonomy without policy and receipt gates

### Security CTA

Start with a review-heavy workflow where auditability, bounded action, and
evidence retention matter more than raw task volume.

---

## Technical Buyer Variant

### One-line framing

**Aragora gives technical teams the control plane they do not get from worker runtimes alone.**

### Core narrative

Technical buyers already know they can get raw execution from Codex, Claude
Code, OpenCode, Pi, or a homegrown agent harness. The question is what they add
when the work becomes consequential. Aragora is the layer that governs that
execution: multi-model challenge, structured review, provenance, calibration,
and receipts that make the output usable inside real engineering systems.

This is why Aragora should not be pitched as "another coding agent." It is the
product you buy when you want AI-assisted execution to survive code review,
security review, and operational scrutiny. It integrates with existing
substrates instead of forcing a rip-and-replace, which lowers adoption friction
for engineering organizations that already have strong opinions about their
worker tools.

### What the technical buyer cares about

- Whether Aragora replaces or complements existing agent tools
- Whether it can plug into current review and delivery workflows
- Whether the evidence quality is high enough for consequential engineering work
- Whether the architecture supports future governance and calibration needs

### Proof points to emphasize

- Aragora sits above execution substrates rather than competing with them
- Adversarial debate and dissent trails are core primitives, not add-ons
- Receipts, review outcomes, and blocker visibility are first-class outputs
- API, SDK, and channel surfaces exist to embed Aragora into current systems

### Technical buyer CTA

Evaluate Aragora on one consequential review path where a fast answer is not
enough and the team needs evidence for why a change should ship.

---

## Talk Tracks By Moment

| Moment | Best stakeholder lead |
|---|---|
| Fundraising or board-pressure conversation | Founder variant |
| Workflow design, support burden, or throughput discussion | Operator variant |
| Security architecture or compliance review | Security reviewer variant |
| Tooling evaluation against Codex, Claude Code, or LangGraph | Technical buyer variant |

## Message Discipline

If a conversation drifts, return to the same category claim:

**Aragora governs AI-assisted execution with receipts, review, provenance, and truthful stopping behavior.**
