# Aragora Objection-Handling Library

Last updated: 2026-03-24

Related:
- `docs/outreach/DESIGN_PARTNER_ONEPAGER.md`
- `docs/strategy/COMPETITIVE_POSITIONING_2026_03.md`
- `docs/strategy/WHEN_TO_USE_ARAGORA_VS_EXECUTION_SUBSTRATES.md`
- `docs/status/DESIGN_PARTNER_PROGRAM.md`

---

## Purpose

This library is for live sales calls, partner outreach, follow-up emails, and
pilot scoping. It is not a general FAQ. It exists to answer the five objections
that will predictably appear once a buyer understands what Aragora actually is:

1. security
2. trust
3. false positives
4. integration burden
5. "why are existing tools not enough?"

The goal is not to win an abstract debate. The goal is to move the buyer toward
one bounded workflow with a clear approval path and a first receipt.

---

## Message Discipline

Use this structure for every objection:

1. Acknowledge the concern in plain language.
2. Reframe Aragora around one bounded workflow, not platform sprawl.
3. Tie the answer to a shipped proof surface: decision review, inbox trust
   wedge, Ralph/swarm, or receipt-gated review.
4. End on the lowest-risk next step: one artifact, one gate, one receipt.

Do not:

- promise zero false positives
- promise broad unattended autonomy on day one
- sell model count or connector count as the moat
- position Aragora as a replacement for every worker runtime

---

## Fast Answer Table

| Objection | Short answer | Proof to use | Best next step |
|---|---|---|---|
| "This sounds risky from a security standpoint." | Aragora is designed to keep authority narrow: receipt before action, explicit approval gates, and self-hosted deployment where needed. | Narrow allowed actions in the inbox trust wedge; self-hosted/offline posture; SSO, RBAC, and audit artifacts | Offer a pilot on sanitized artifacts or self-hosted evaluation |
| "Why should we trust AI to judge AI?" | You should not trust one model. Aragora makes models challenge each other, records dissent, and keeps the human approval path explicit. | Multi-model debate, decision receipts, calibration, dissent trails | Run one real artifact and review the receipt together |
| "Won't this create more false positives and slow us down?" | Aragora is supposed to expose uncertainty, not hide it. Start with a narrow workflow, human override, and measure approval versus override rate. | Receipt confidence, dissent capture, bounded allowed actions, override tracking in partner motion | Pilot one recurring workflow with a narrow policy |
| "Integration looks heavy." | Do not roll out a platform. Start with one workflow using an artifact the team already produces. | Onboarding, API key setup, SDKs, CLI, real demo surface, one-pager activation paths | Pick one artifact source: PR, spec, inbox slice, or bounded backlog lane |
| "Why not just use Codex, Claude Code, LangGraph, or our existing tools?" | Those tools are strong worker substrates. Aragora sits above them when a decision needs receipts, dissent, provenance, and truthful stopping. | Execution-substrate strategy, review receipts, approval gates, swarm receipts | Identify where current tools stop and a human still needs defensible proof |

---

## Objection 1: Security

### What the buyer usually means

- "I do not want another AI tool with broad permissions."
- "I cannot send sensitive data to an ungoverned black box."
- "Security and compliance will block this unless there is a narrow control story."

### Short answer

Aragora should be introduced as a controlled review layer, not an always-on
autonomous actor. The default posture is narrow authority, explicit approval,
and auditable receipts.

### Full answer

The security posture is strongest when Aragora is framed correctly. We are not
asking for open-ended action rights across the environment. The product starts
with bounded workflows, explicit review gates, and persisted receipts before any
action occurs. For sensitive environments, the deployment story is self-hosted,
offline-capable, and compatible with enterprise controls like SSO, RBAC, audit
logging, and restricted data boundaries.

The right first pilot is not "let the agents operate everywhere." It is "take
one workflow that already has a human approver and make the decision path
auditable."

### Proof points to cite

- Inbox trust wedge uses a narrow allowed action set: `ARCHIVE`, `STAR`,
  `LABEL`, `IGNORE`
- Receipt-before-action is the default posture for partner workflows
- Enterprise deployment options include self-hosted and offline/air-gapped modes
- Enterprise controls include OIDC/SAML SSO, RBAC, encryption, and audit-ready
  artifacts

### Best proof surface

Decision review or inbox trust wedge.

### Discovery question

"Which workflow would your security team actually approve first if the action
surface were narrow and a human gate remained in place?"

### Avoid saying

- "It is fully autonomous."
- "Security can be dealt with later."
- "Just trust the model."

---

## Objection 2: Trust

### What the buyer usually means

- "LLMs hallucinate. Why should I rely on this?"
- "If models disagree, who decides?"
- "I need something I can defend to my team, auditors, or customers."

### Short answer

Aragora is not asking the buyer to trust a single model. It turns model
disagreement into visible evidence, preserves dissent, and gives the operator a
receipt they can inspect before acting.

### Full answer

Trust is the wrong frame if it implies blind confidence in AI output. Aragora's
core claim is the opposite: frontier models are unreliable witnesses, so the
system treats them as witnesses that must challenge each other. When models
converge after critique, that is more informative than an isolated answer. When
they do not converge, the dissent trail shows exactly where human judgment is
still required.

What the buyer should trust is not "the AI." They should trust the process:
heterogeneous model challenge, explicit dissent, calibrated weighting over time,
and a receipt that shows why the system advanced or stopped.

### Proof points to cite

- Adversarial multi-model debate is the core product primitive
- Decision receipts preserve consensus, dissent, provenance, and signatures
- Calibration systems track agent reliability over time
- Truthful blocker handling is part of the control-plane story

### Best proof surface

Receipt-gated design review or architecture review.

### Discovery question

"What decision today gets made with AI help but leaves you with no durable proof
of why the team accepted it?"

### Avoid saying

- "The answer is more accurate because more models voted."
- "Consensus means certainty."
- "You can remove the human reviewer immediately."

---

## Objection 3: False Positives

### What the buyer usually means

- "I already have enough noisy alerts and review comments."
- "I do not want a system that blocks work for weak reasons."
- "If the tool is overcautious, the team will route around it."

### Short answer

Aragora should not be sold as "never wrong." It should be sold as a way to
surface uncertainty explicitly, keep override paths clear, and measure whether
the review layer is helping or just creating noise.

### Full answer

False positives matter because an overactive control plane destroys trust faster
than a quiet one. The right answer is not to promise that Aragora will never
over-flag. The right answer is that Aragora keeps uncertainty legible instead of
hiding it behind one confident response. Dissent, calibrated confidence, and
approval gates make it possible to tune the operating threshold without losing
the evidence trail.

This is why pilots should start narrow. Pick one recurring workflow, leave the
human approver in the loop, and measure approval rate, override rate, and
decision-change rate. If the system creates more noise than signal, the receipt
history will show it quickly.

### Proof points to cite

- Partner program explicitly tracks approval rate and override rate
- Receipt confidence and dissent capture make weak cases visible
- Allowed action surfaces stay narrow early in a rollout
- Expansion is by bounded wins, not by narrative confidence

### Best proof surface

Inbox trust wedge or bounded review workflow with a human approver.

### Discovery question

"Where are false positives most expensive for your team today: triage noise,
review delay, or over-blocking on release decisions?"

### Avoid saying

- "False positives disappear with more models."
- "The right rollout is org-wide from day one."
- "Every flagged issue should block the workflow."

---

## Objection 4: Integration Burden

### What the buyer usually means

- "This looks like a six-month platform project."
- "My team will not wire another complex AI stack just to run a pilot."
- "We need value before security, procurement, and platform teams get dragged in."

### Short answer

Do not sell Aragora as a full-platform rollout. Sell one bounded activation path
using an artifact the team already has: a PR, spec, inbox slice, or bounded
backlog task.

### Full answer

The fastest way to lose a design partner is to make the first step feel like an
infrastructure migration. Aragora has a large platform surface, but the partner
motion should stay intentionally narrow: pick one workflow, map one trigger,
keep one approval owner, and produce one receipt that matters. The current
product loop already supports onboarding, API key setup, live debate, receipt
generation, and dashboard visibility without requiring a broad integration
program first.

The integration story gets stronger when the team sees a real receipt on a real
artifact. Only after that should broader SDK, connector, or workflow adoption
enter the conversation.

### Proof points to cite

- Interactive onboarding and API key management are shipped
- CLI, API, and SDK paths already exist
- Decision review is the default activation path
- The partner program is explicitly scoped to one recurring workflow first

### Best proof surface

Decision review on a real spec, PR, or policy artifact.

### Discovery question

"What is the smallest recurring workflow where a receipt would matter but the
setup is still low enough to try this week?"

### Avoid saying

- "We should start by integrating everything."
- "You need a broad platform rollout to see value."
- "Connector breadth is the product story."

---

## Objection 5: Why Existing Tools Are Not Enough

### What the buyer usually means

- "We already use Codex, Claude Code, OpenCode, or LangGraph."
- "We already have workflow tools, code review, and governance dashboards."
- "Why add another layer?"

### Short answer

Existing tools are often the right worker runtimes. Aragora matters when the
workflow becomes consequential enough to need adversarial review, receipts,
provenance, explicit approval, and truthful stopping behavior.

### Full answer

This objection should never be answered by attacking execution substrates.
Codex, Claude Code, OpenCode, Pi, LangGraph, and CrewAI are useful. They are
fast, practical, and often the right default when the task only needs raw
execution. Aragora is the layer above them for decisions and actions that need
defensible governance. The competitive answer is not "we support more models" or
"we have more connectors." The answer is that Aragora governs AI-assisted work
with disagreement, receipts, provenance, and blocker truthfulness.

Use worker runtimes when speed is the only requirement. Use Aragora when a human
still needs to know what evidence was considered, who disagreed, why the system
advanced, and what the next approval action is.

### Proof points to cite

- Strategy docs explicitly frame existing tools as substrates, not enemies
- Aragora's moat is the control-plane layer: receipts, review, provenance, and
  truthful gates
- Swarm orchestration is bounded by leases, receipts, and integrator-controlled
  merge authority
- Design review and inbox trust wedge both show receipt-gated behavior that
  generic execution tools do not provide

### Best proof surface

Any workflow where a team currently says, "we got the answer, but we still do not
have proof."

### Discovery question

"Where do your current tools stop and leave a human responsible for a decision
without a defensible receipt?"

### Avoid saying

- "Your current tools are obsolete."
- "Aragora replaces all coding agents."
- "The moat is that we support more providers."

---

## Closing Moves

Use one of these closes after answering the objection:

- "Let's keep the approval gate in place and prove this on one artifact."
- "We do not need a platform rollout to validate this. We need one workflow and
  one receipt."
- "If the receipt is not useful to your approver, we will know quickly."
- "The first question is not whether Aragora can run everywhere. It is whether
  one recurring decision becomes safer and easier to defend."

---

## Default Pilot Recommendation

If the objection is serious but not a hard blocker, steer to the same next step:

1. choose one recurring workflow
2. keep a human approval gate
3. use sanitized artifacts or self-hosted deployment if needed
4. generate a first receipt
5. inspect approval, override, and decision-change rates before expanding

That is the lowest-risk path to converting skepticism into evidence.
