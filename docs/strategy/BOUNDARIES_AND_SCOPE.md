# Boundaries And Scope

Consolidated from:
- `docs/strategy/NON_GOALS_LEDGER.md`
- `docs/strategy/STOP_DOING_LEDGER_2026_03.md`
- `docs/strategy/WHEN_TO_USE_ARAGORA_VS_EXECUTION_SUBSTRATES.md`
- `docs/strategy/HUMAN_IN_THE_LOOP_BOUNDARIES.md`

Last updated: 2026-03-25

---

## Part 1: What We Are And Are Not

### Category Statement

Aragora exists to govern consequential AI-assisted execution with:

- explicit review and approval points
- receipts and provenance
- surfaced disagreement and dissent
- bounded delegation
- truthful blocker handling and terminal states

Aragora does **not** exist to be the default runtime for every autonomous agent
workflow.

If a roadmap item, launch post, or sales claim conflicts with this ledger, the
ledger wins unless leadership intentionally updates it.

### Explicit Non-Goals

| Non-goal | Why it is out of scope | What Aragora does instead |
|---|---|---|
| Be a generic autonomous-agent platform for arbitrary work | That framing collapses Aragora into a crowded substrate category and hides the real wedge | Govern consequential execution paths where truthfulness, receipts, and review matter |
| Replace Codex, Claude Code, OpenCode, Pi, or similar worker runtimes | Those tools are better optimized for direct execution speed and low-friction operator loops | Sit above worker runtimes as the control plane for routing, review, receipts, and publish/merge gates |
| Win by advertising the biggest swarm, the most agents, or the most connectors | Breadth is table stakes and reads as interchangeable commodity orchestration | Win on accountable execution quality, provenance, and trustworthy terminalization |
| Sell lights-out autonomy as the default operating mode | Unbounded autonomy erodes trust, weakens operator clarity, and invites overselling | Default to supervised automation with explicit receipts, blocker handling, and human escalation points |
| Treat bigger heterogeneous orchestras as a roadmap goal by themselves | Coordination complexity grows faster than truthfulness unless delegation is tightly bounded | Use a lead-plus-bounded-workers pattern and only expand orchestration when it measurably improves review quality |
| Lead with vertical or enterprise sprawl before the core wedge is repeatedly used | Packaging without repeated consequential workflows creates narrative inflation instead of product pull | Prioritize review, receipt, and controlled execution paths that prove the beachhead with real users |
| Position cryptoeconomic identity or on-chain mechanics as the near-term product story | Those mechanisms only matter after Aragora has durable decision volume and calibration data | Preserve receipts and empirical reliability first; layer stronger accountability later if justified |

### Roadmap Guardrails

A roadmap item is in bounds only if it strengthens one or more of these:

- review quality on consequential work
- provenance, receipts, or auditability
- truthful stopping behavior and blocker precision
- bounded delegation and operator control
- measurable quality deltas versus a single-worker baseline

A roadmap item is out of bounds if its main value is one of these:

- making Aragora sound like a generic autonomous-agent platform
- increasing agent count, connector count, or orchestration complexity without a truthfulness gain
- replacing worker runtimes instead of governing them
- implying unattended autonomy without explicit evidence and terminal states

### Sales And Packaging Guardrails

Preferred framing:

- auditable execution control plane for AI-assisted work
- review, receipts, provenance, and truthful gates for consequential tasks
- governance layer above worker runtimes

Avoid framing Aragora as:

- an autonomous-agent platform for any workflow
- a general-purpose agent shell or IDE replacement
- a biggest-swarm orchestration vendor
- a model-breadth or connector-count story

### Simple Litmus Test

If a prospect asks, "Why not just use Codex, Claude Code, OpenCode, or Pi
directly?" the answer should be about:

- receipts
- review
- provenance
- bounded delegation
- truthful stopping behavior

If the answer depends mainly on "we have more agents" or "we automate more of
the org," the message has drifted outside Aragora's category.

---

## Part 2: Stop-Doing Ledger

This ledger turns strategy into an explicit filter for roadmap and sprint work.
Its purpose is simple: if a project does not strengthen Aragora's current wedge,
the team should mark it as `defer` or `reject` instead of carrying it as
ambient "important later" work.

### Current Wedge

Aragora's current wedge is:

**auditable multi-model execution and review for consequential engineering work,
with receipts, provenance, and truthful stopping behavior**

In product terms, near-term work should strengthen one or more of these:

- `aragora review` as a complete, trustworthy product path
- the inbox trust wedge as a second dogfoodable workflow
- operator-facing receipts, blocker visibility, and review evidence
- repeatable live demos and design-partner adoption

### How To Use This Ledger

For any proposed project, ask four questions:

1. Does it make `aragora review` or the inbox trust wedge more usable now?
2. Does it improve receipts, provenance, blocker truthfulness, or operator trust?
3. Does it shorten time-to-first-useful-result for a real design partner?
4. Does it generate proof that multi-model review beats a single strong model?

If the answer is "no" across the board, the default action is not "keep it on
the roadmap anyway." The default action is `defer` or `reject`.

### Decision Meanings

| Status | Meaning |
|---|---|
| `allow` | Actively strengthens the wedge now; can compete for near-term capacity |
| `defer` | Potentially valuable later, but not before the wedge is repeatable with external users |
| `reject` | Attractive but strategically dilutive in the current phase; do not start unless the wedge changes |

### Stop-Doing Entries

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

### Default Response Patterns

When a project does not strengthen the wedge, use explicit language:

- `reject`: "This increases generic substrate breadth without improving the auditable review/inbox wedge."
- `defer`: "This may matter later, but the re-open trigger is repeatable external usage of the current wedge."
- `allow`: "This improves the current wedge by increasing trust, operator clarity, or real workflow adoption."

### Exceptions

This ledger is not a ban on all adjacent work.

Exceptions are reasonable when:

- a bug outside the wedge blocks the wedge directly
- a customer/design partner requirement is concrete and immediate
- a compliance/security task is necessary to keep a live pilot running

The burden of proof stays on the proposer to show the wedge connection plainly.

---

## Part 3: When To Use Aragora Vs Execution Substrates

### Default Rule

Use the **simplest layer that preserves the needed truthfulness**.

If the task only needs raw execution, use a worker runtime.
If the task needs receipts, review, provenance, or truthful blocker handling,
use Aragora.

### The Buyer's Actual Menu

Most teams are not choosing between agent research papers. They are choosing
between a few practical defaults.

| Default buyer choice | Best when | What breaks first | Move to Aragora when |
|---|---|---|---|
| Status quo coordination: Slack, docs, meetings, checklists | Work is infrequent, ambiguous, and one owner can carry context | Decisions vanish into chat, handoffs get slow, and no one can reconstruct why something shipped | The same consequential workflow repeats and rework or audit pain is now visible |
| Generic agent: Codex, Claude Code, OpenCode, Pi, ChatGPT | A bounded task has one owner and speed matters most | Human arbitration stays implicit, evidence is thin, and blocker handling varies by run | AI-assisted work needs explicit review, delegation, or truthful stopping behavior |
| Bespoke workflow: scripts, prompts, GitHub Actions, MCP glue | One narrow path is stable enough to script around | Logic sprawls across prompts and scripts, ownership is brittle, and auditability is poor | You need one control plane across multiple workflows with policy, receipts, and outcome tracking |
| Human-only review | Trust matters more than throughput and volume is low | Review becomes the bottleneck and evidence quality depends on heroics | Humans still approve, but they need pre-structured evidence, dissent, and provenance |

### Decision Table

| Situation | Best default | Why |
|---|---|---|
| Manual coordination across Slack / docs / meetings | Status quo | Cheapest path when the work is rare and one human can absorb the ambiguity |
| Small code edit with clear scope | Single strong coding agent | Lowest coordination overhead |
| One owner, 2-4 bounded parallel subtasks | Lead agent plus bounded subagents | Keeps ownership clear while getting real parallelism |
| Vague natural-language request | Lead agent first | Someone has to frame, slice, and own integration |
| Recurring workflow already held together by scripts, prompts, and GitHub Actions | Aragora | Replace brittle bespoke governance with explicit review, receipts, and truthful stopping |
| High-risk approval where a human must own the final call | Human-only review, optionally prepared by Aragora | Keep the human gate explicit while reducing evidence-prep cost |
| High-stakes review or merge decision | Aragora | Receipts, dissent, gates, and blocker truth matter |
| Unattended multi-step execution | Aragora | Queue, watch, integrate, and truthful terminalization are the point |
| Pure terminal productivity for one developer | Codex / Claude Code / OpenCode / Pi | Fastest path, lowest ceremony |
| Model-routing experiments | OpenCode/Pi or direct worker harnesses | Good substrate for worker-level routing |
| Building Aragora itself | Lead agent plus bounded subagents, optionally under Aragora for proof runs | Orchestration overhead must stay bounded |

### Recommended Operating Modes

#### 1. Single-agent mode

Use when:
- the task is small
- scope is obvious
- integration risk is low

Best tools: Codex, Claude Code, other direct coding agents.

#### 2. Lead agent plus bounded subagents

Use when:
- the prompt is vague
- you need a decomposition pass
- there are a few independent sidecar tasks

This is the default mode for building Aragora itself.

Why:
- one agent owns framing and integration
- a few workers handle isolated slices
- coordination stays legible

#### 3. Aragora control-plane mode

Use when:
- the work is consequential
- a receipt is valuable
- review and publish behavior must be explicit
- unattended execution needs truthful stopping behavior

Why:
- Aragora's value starts where worker runtimes stop
- it owns governance, not just execution

#### 4. Worker-runtime mode

Use OpenCode, Pi, Codex, Claude Code, or similar tools directly when:
- you mainly need execution speed
- auditability is not the main bottleneck
- the task does not need a control plane

These tools are better treated as substrates than as strategic enemies.

### A Note On "Whole Orchestras"

Large heterogeneous swarms sound appealing, but they fail quickly when:

- the prompt is vague
- scopes overlap
- state propagation lies
- verification evidence is weak
- no one clearly owns integration

So the default should not be "spawn a huge orchestra."

The better sequence is:

1. one lead agent frames the task
2. a few bounded workers handle independent slices
3. Aragora governs when the work becomes consequential enough to need receipts, gates, and truthfulness

### Strategic Implication

Aragora should not try to beat OpenCode or Pi at being execution substrates.

Aragora should sit above them:

- selecting when deeper governance is needed
- preserving receipts and provenance
- making disagreement useful
- turning "needs human" into a precise, low-cost next action

That same logic applies to manual coordination, bespoke automation, and
human-only review: Aragora wins when the cost of being fast but unexplainable
has become higher than the overhead of governance.

---

## Part 4: Human-In-The-Loop Boundaries

Aragora is allowed to automate work, not to erase accountable human judgment.
The system should always be in one of three states:

- `propose`: generate drafts, plans, critiques, patches, and bounded actions
- `must ask`: pause for an explicit human decision before continuing
- `must stop`: terminate the lane truthfully and emit the blocker / next action

The purpose of this boundary is simple: keep automation useful on reversible
work, keep approval explicit on consequential transitions, and keep failure
truthful instead of silently improvising.

### Default Rule

Use the lowest-autonomy mode that preserves truthfulness.

- If the work is reversible, bounded, and inside an explicit contract, Aragora
  can propose.
- If the next step changes authority, risk, spend, publication state, or scope,
  Aragora must ask.
- If the system cannot proceed safely or truthfully, Aragora must stop.

### 1. When Aragora Can Propose

Aragora can act without an approval pause when all of the following are true:

- the task is inside an explicit scope, lease, issue, or validation contract
- the output is reversible or low-cost to discard
- the action does not publish, merge, send, purchase, or bind the organization
- the system has enough evidence to explain why it chose the next step
- a human can still review, override, or reject the result before a
  consequential transition

Typical `propose` work:

- write or revise docs, specs, plans, and internal analysis
- generate code patches in a bounded file scope
- run local validation, tests, linting, and review flows
- decompose a vague request into candidate tasks and assumptions
- prepare drafts of PRs, issues, receipts, runbooks, or operator summaries

`Propose` means "draft or stage the next move," not "silently finalize the
decision."

### 2. When Aragora Must Ask

Aragora must ask for explicit human approval before continuing when any of the
following become true:

- the next step creates an irreversible or externally visible effect
- the system needs to merge, publish, deploy, send messages, or trigger
  real-world execution
- the task requires choosing among materially different strategies, priorities,
  or tradeoffs not already settled in the contract
- scope needs to widen beyond the leased files, stated issue, or accepted plan
- money, legal exposure, compliance posture, vendor commitments, or customer
  promises are involved
- the system is about to touch production data, secrets, credentials, access
  controls, or user accounts
- the system sees meaningful disagreement, ambiguous evidence, or multiple
  reasonable interpretations that affect the outcome
- a human role with authority is required by policy, governance, or law

Typical `must ask` gates:

- "merge this PR or keep iterating?"
- "deploy this change to a live environment?"
- "send this message to a customer or regulator?"
- "expand from `docs/**` into application code?"
- "treat this recommendation as the final organizational decision?"

When Aragora asks, it should present:

- the exact pending action
- why approval is required
- the evidence and dissent that matter
- the lowest-cost options available next

### 3. When Aragora Must Stop

Aragora must stop, not negotiate past the boundary, when any of these are true:

- the task is outside authorized scope and no approval path is available
- the required human decision-maker is unavailable
- the evidence is insufficient to continue truthfully
- validation fails in a way the lane cannot repair inside scope
- instructions conflict in a way that changes the outcome materially
- the requested action is unsafe, prohibited, deceptive, or would conceal
  uncertainty from the operator
- the system detects that continuing would fabricate state, fake completion, or
  claim authority it does not have

Typical `must stop` outputs:

- a blocker receipt
- exact reason for termination
- files or artifacts examined
- what remains unresolved
- the next human action required to resume

Stopping is a feature, not a failure mode. Aragora should prefer a truthful
halt over a smooth but false continuation.

### Decision Table

| Situation | State | Required behavior |
|---|---|---|
| Drafting a plan, patch, or review in approved scope | `propose` | Produce the draft, show assumptions, keep it reversible |
| Running local checks and collecting evidence | `propose` | Execute and attach receipts / results |
| Need to merge, deploy, publish, spend, or contact outsiders | `must ask` | Pause and request explicit approval |
| Need to widen scope or choose between materially different strategies | `must ask` | Surface options, tradeoffs, and recommend one |
| Missing authority, missing evidence, unsafe request, or unresolved contradiction | `must stop` | Terminate truthfully with blocker and next action |

### Operating Principles

#### 1. Humans own commitment

Aragora may prepare commitments, but humans own the final commitment when the
organization becomes bound by the result.

#### 2. Reversibility buys autonomy

The more reversible the work, the more Aragora can do without waiting.
Irreversibility moves the system from `propose` to `must ask`.

#### 3. Ambiguity is a gate, not a license

When material ambiguity appears, the system should narrow, ask, or stop. It
should not improvise a hidden policy.

#### 4. Truthful stopping beats fake completion

The system must never mask uncertainty with authoritative tone, invented
progress, or unearned "done" states.

#### 5. Override must stay real

Operators need a real ability to inspect, disregard, reverse, or interrupt the
system. Human oversight is not satisfied by a decorative approval checkbox.

### Product Implication

Aragora's core control-plane promise is not maximum autonomy. It is
**bounded autonomy with explicit human authority at consequential edges**.

That means the product should optimize for:

- clear state transitions between `propose`, `must ask`, and `must stop`
- receipts that explain why a boundary was crossed
- low-friction approval UX at real decision points
- truthful blocker summaries instead of silent retries or scope drift

If Aragora gets this boundary right, it becomes trustworthy precisely because
it does not pretend every problem should be solved autonomously.
