# Aragora: Design Partner Brief

**Stop shipping AI-assisted decisions you can't explain.**

Aragora governs consequential AI-assisted work with multi-model review,
decision receipts, and truthful gates, so teams can move faster without losing
provenance or human control.

Last updated: 2026-03-24

---

## The Problem

Your team uses AI every day -- generating code, drafting plans, evaluating options.
But when something goes wrong, nobody can answer: *why did we decide that?*

- **No audit trail.** AI outputs are ephemeral. Decisions evaporate.
- **Single-model blind spots.** One LLM = one perspective. No adversarial vetting.
- **Compliance gaps.** Regulators want explainability. You have chat logs. EU AI Act enforcement begins August 2, 2026.

## What Aragora Does

Aragora is a **Decision Integrity Platform**. It orchestrates multiple AI agents
to adversarially vet decisions, then delivers audit-ready **decision receipts**
to any channel.

```
Your input -> multi-agent debate -> consensus + receipt -> KM feedback -> Slack / GitHub / API
```

## Core Proof Points

1. **Disagreement becomes useful evidence.** Multiple models challenge each
   other before work advances, so dissent shows exactly where human judgment is
   still needed.

2. **Every consequential action has a receipt.** Aragora produces
   audit-ready decision receipts with provenance, votes, confidence, and the
   explicit next action.

3. **Execution is bounded and truthful.** Approval gates, lease-based
   coordination, and explicit blocker handling are first-class parts of the
   system, not afterthoughts.

4. **It fits above the tools you already use.** Aragora complements GitHub,
   Slack, the terminal, and existing worker runtimes instead of forcing a stack
   replacement.

## What Works End-to-End Today (March 23, 2026)

The product loop is **structurally closed** on `main`. The complete path:

```
Onboarding wizard -> API key setup -> ProviderRouter-backed debate
  -> KM-enriched context -> consensus + receipt -> KM writeback
  -> live dashboard -> demo surface
```

23 PRs merged March 21-23 closing 15 issues to wire this loop shut:

| Capability | Status | Evidence |
|------------|--------|----------|
| Multi-agent debate engine (43 agent types, 10+ concurrent) | Production | PR #1182 |
| Smart provider routing (cost/quality/latency Pareto) | Wired | PR #1167 |
| KM-enriched debate context + outcome writeback | Wired | PRs #1168, #1176 |
| Interactive onboarding wizard (landing to first debate) | Shipped | PR #1170 |
| API key management (real backend, no client-side fakes) | Shipped | PR #1169 |
| Live dashboard with active debate tracking | Shipped | PR #1175 |
| Demo surface hitting real backend | Shipped | PR #1177 |
| Decision receipts with SHA-256 audit trail | Production | |
| Autonomous repo improvement (Ralph loop) | Validated | V14 benchmark |
| Swarm orchestration (supervisor, leases, receipts) | Production | |
| Inbox trust wedge (Gmail -> debate -> receipt -> action) | Shipped | |
| EU AI Act compliance artifacts (85/100 score) | Production | |
| 210,000+ tests across 5,000+ test files | CI | |

## Three Proof Surfaces for Partners

**1. Decision Review.** Run any artifact (spec, PR, architecture proposal) through
multi-agent debate. Get a receipt with consensus, dissent, confidence, and provenance.
Share it on Slack, export as PDF/MD/JSON.

**2. Autonomous Repo Improvement.** Point Aragora at bounded engineering work.
The Ralph loop: spec -> deliverable -> cross-model review -> repair -> PR -> merge.
Zero operator intervention on the validated benchmark path.

**3. Inbox Trust Wedge.** Gmail -> adversarial debate -> signed receipt -> CLI approval
-> action (archive/star/label). Receipt-before-action is non-negotiable.

## Common Objections

**"Why not just use Codex, Claude Code, or OpenCode?"**
Those tools are worker runtimes. Aragora governs the layer above them: review,
receipts, provenance, dissent, and truthful stopping behavior.

**"Won't this be slower than a single model?"**
Yes, and that is the point. Use the simplest layer that preserves the needed
truthfulness. Aragora is for consequential work, not every autocomplete.

**"Are receipts just logs?"**
No. A receipt explains who said what, what evidence was used, why the system
advanced or stopped, and what the next human action is.

**"Do we have to replace our current tools?"**
No. Aragora sits above the tools your team already uses and adds governance
only where the risk justifies it.

## What Partners Get

- A self-improving platform that learns from every decision it vets
- Autonomous repo maintenance under explicit policy and receipt gates
- Audit-ready decision receipts (SHA-256 signed, exportable, verifiable)
- Priority support, roadmap influence, and direct access to the team
- Early access to enterprise features (OIDC/SAML SSO, RBAC, multi-tenancy)

## What We Need From Partners

| From You | From Us |
|----------|---------|
| One bounded recurring workflow with a clear trigger | Free access during partner period |
| 30 min onboarding + weekly 30 min check-in for 4-6 weeks | Hands-on onboarding to first receipt |
| Real artifacts (sanitized OK): inbox batch, design doc, bounded backlog | Priority feature requests and weekly iteration |
| Candid feedback on quality, friction, and trust | Co-marketing opportunity (anonymized case studies OK) |

## Who This Is For

We are looking for **3-5 design partners** who:

- Feel real pain from review latency, manual triage, audit evidence work, or bounded engineering backlog throughput
- Have one recurring workflow with a clear trigger, owner, and success/failure outcome
- Can start narrow (one receipt-gated workflow) before expanding
- Have a champion who can provide artifacts, review receipts, and join a weekly loop

**Best-fit segments:** Regulated SaaS, FinTech, HealthTech, platform/security teams,
founder-led teams with painful inbox triage, AI-native teams frustrated by single-model trust gaps.

## Numbers

- **216,000+ tests** across 4,700+ test files
- **3,800+ Python modules** | **3,100+ API operations**
- **42 knowledge adapters** for cross-system learning
- **43 agent types** across 6+ LLM providers
- **190+ WebSocket event types** for real-time streaming
- **Python + TypeScript SDKs** (185 Py / 183 TS namespaces)

## Next Step

**Primary CTA:** Book a 15-minute workflow scoping call.

**Secondary CTA:** Send one bounded artifact or recurring workflow and we will
show what the receipt path looks like.

**Tertiary CTA:** Reply with the team's current review bottleneck if you are
not ready for a call yet.

Contact options:

- Email: [your-email]
- Calendar: [booking-link]
- GitHub: https://github.com/synaptent/aragora

---

*Aragora is open source under the Apache 2.0 license.*
*Design partners get priority support, roadmap influence, and early access to enterprise features.*
