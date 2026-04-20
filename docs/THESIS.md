# The Aragora Thesis

> Canonical source of authority. Every other strategic doc links up to this.
> Last updated: 2026-04-20. Status: v1 draft, under review.

---

## The thesis

Advanced AI creates a new problem. It generates more output than any human
can meaningfully review, while generating output humans cannot safely
trust. The untrustworthiness is systematic, not incidental — it stems
from bad actors, prompt injection, training-data poisoning and bias,
spiky capabilities, hallucinations, knowledge cutoffs, and the ordinary
fact that any single model has blind spots it cannot detect alone.
Human attention becomes the scarce resource, and it cannot be rescued
by delegating review back to AI.

What humans and AI agents both actually need — and they both need this,
not just humans — is **infrastructure for truth-seeking**: tooling that
surfaces the structure of a claim (inputs, outputs, assumptions, values,
cruxes, dependencies, scope), cross-checks it adversarially across
heterogeneous lenses, tests it against outcomes, and distills the result
to a volume and form that informed consent, rejection, or feedback
becomes possible in the time a human actually has.

Aragora is that infrastructure. Its first domain is its own codebase.
Its endpoint is wherever the pattern generalizes.

---

## The five premises

1. **Bandwidth.** AI produces more output than humans can meaningfully process.
2. **Trust.** AI output is systematically untrustworthy — from adversarial
   inputs, biased or poisoned training data, spiky capabilities,
   hallucinations, knowledge cutoffs, and individual-model blind spots.
3. **No safe delegation.** Neither humans nor AI agents can safely defer
   to any single other agent's judgment, including their own.
4. **Structure.** Claims, conclusions, and decisions are tractable in
   terms of their inputs, outputs, assumptions, values, cruxes,
   dependencies, and scopes — and making that structure explicit is the
   prerequisite for cross-checking anything.
5. **Outcomes.** Truth-seeking is a process, not an oracle lookup.
   Claims that prove harmful or false get downweighted; claims that
   prove beneficial or true get upweighted. The weights are observable,
   auditable, and subject to revision.

---

## What Aragora therefore is

A truth-seeking substrate for **both humans and AI agents**, built on
four coordinated components:

1. **Structural decomposition.** Every consequential claim, conclusion,
   or decision gets decomposed into inputs, assumptions, cruxes,
   dependencies, and scope — before it is evaluated. Unstructured intent
   becomes a structured object that can be reasoned about.
2. **Adversarial cross-checking.** Heterogeneous-model ensembles — each
   with different training, architectures, and blind spots — surface
   what single models miss. Dissent is first-class and preserved, not
   collapsed into a false consensus.
3. **Outcome-weighted feedback.** Claims, agents, and decisions carry
   track records. Calibration is measured. What proved harmful or false
   gets downweighted; what proved beneficial or true gets upweighted.
   The downweighting itself is evidence-linked and auditable.
4. **Distillation to human-scale.** Firehoses become briefs. Briefs
   preserve the load-bearing structure (cruxes, dependencies, dissent,
   outcome weights) so informed consent, rejection, or feedback is
   possible in the time a human actually has.

The audience includes AI agents. An agent evaluating whether to defer to
another agent's output needs the same structural decomposition,
adversarial cross-check, outcome weighting, and distilled form a human
does. The substrate serves both populations.

---

## What Aragora is NOT

Anti-claims — things the thesis explicitly does *not* commit to:

- **Not an oracle.** Aragora does not claim to know what is true. It
  claims to structure a process that approaches *relatively more true,
  less wrong* through evidence and outcome tests.
- **Not a rubber stamp.** Approving bot-generated work on "CI green"
  alone defeats the thesis. Human settlement remains the final gate for
  consequential decisions.
- **Not a replacement for human judgment.** The goal is to make human
  decisions faster, more informed, and more structured — not to remove
  them.
- **Not value-neutral.** Truth-seeking requires a stance: some outputs
  are worse than others, and the system must take a position by
  downweighting, surfacing dissent, or refusing to proceed.
- **Not an arbiter of contested values.** Claims about which outcomes
  are beneficial versus harmful in hard ethical cases are inputs to the
  system, not its output.

---

## Where this thesis does NOT yet apply

Honest edges — regions the thesis does not claim to cover today:

- **Fundamentally value-laden decisions** (religious, ethical x-risk
  tradeoffs, contested political questions) where adversarial
  cross-check alone does not arbitrate. The system can structure such
  decisions and surface dissent, but cannot conclude them.
- **Decisions without decomposable structure.** Some problems resist
  structural decomposition (aesthetic judgment, novel research
  direction-setting). Truth-seeking machinery is weaker here.
- **Decisions where outcomes are not observable, or are observable only
  after long delays** (strategic bets, hiring, long-horizon R&D).
  Outcome-weighting requires feedback that may not arrive in useful
  time. The system must flag this limit explicitly rather than pretend
  to weight the unweighted.
- **Low-consequence, high-volume decisions** where the overhead of
  structural decomposition exceeds the value of the decision. The
  product's internal rule: structure-first applies to consequential
  decisions; trivial decisions get a fast path.

Naming the edges honestly is part of the thesis. A truth-seeking
substrate that pretends to cover everything fails premise 2 on itself.

---

## Load-bearing assumptions (testable)

The thesis rests on five claims that must prove true in practice or the
product fails on its own terms:

| Claim | How to test | Horizon |
|------|-------------|---------|
| Heterogeneous AI ensembles detect what individual models miss | Per-model vs panel accuracy on benchmark corpus; dissent-surfacing rate | H1 |
| Humans given distilled advisory packets make better decisions than raw output | Decision-quality A/B on matched PR populations; override-correlated-with-outcome rate | H1–H2 |
| Structural decomposition is tractable for most consequential decisions | Percent of intent objects that decompose to testable cruxes; incompletions per class | H1–H2 |
| Cryptographic receipts produce trust that matters to buyers | Design-partner willingness-to-pay conditional on receipt presence | H2 |
| The pattern generalizes beyond software execution | Cross-domain benchmark once a software-execution wedge is proven | H3 |

If any of these fail under test, the corresponding component is wrong
or overreaching and the thesis has to be revised. The point of naming
the assumptions is to make revision cheap.

---

## How outcomes actually close the loop

Premise 5 depends on outcomes being observable and fed back. Concretely:

1. **Every consequential decision is emitted as a receipt** with
   structure, evidence, dissent, and a verdict.
2. **Outcomes are recorded against receipts** — test pass rate, merge
   stability, incident linkage, downstream revert, human override,
   design-partner adoption, compliance findings.
3. **Weights update from outcomes** — per-agent calibration scores,
   per-claim verification rates, per-lens dissent usefulness, per-
   decision-class override rates.
4. **New decisions consult updated weights** — truth-ratio weighting in
   consensus, selection feedback in agent picking, claim staleness in
   belief network, refused-to-proceed flags from repeated harm.

If any link in that chain is missing or unobserved, the outcome loop is
broken and premise 5 holds only in claim, not in practice.

---

## Generalization path (from codebase to wider domains)

The thesis commits to a sequenced rollout from software execution to
organizational substrate. The stages are:

1. **Own codebase (H1).** Aragora maintains Aragora. Dogfood proves
   bandwidth / trust / structure / outcomes on a domain where ground
   truth (code runs, tests pass, merge sticks) is cheap to observe.
2. **External software execution (H2).** Bounded autonomous engineering
   work on design-partner repos. The cryptographic-receipts assumption
   gets tested here.
3. **Consequential non-software decisions (H2→H3).** Risk, compliance,
   incident response, clinical and legal review. Ground truth is more
   expensive to observe; outcome feedback slows. The substrate has to
   flag where it is weaker.
4. **Organization substrate (H3).** Coordinated agentic work across
   functions on one graph with permissioned memory, shared receipts,
   portfolio-level truth-seeking. The endpoint of the thesis, not a
   near-term promise.

Each stage is gated on the previous stage's load-bearing assumption
being validated, not assumed.

---

## How existing capabilities map

Every load-bearing subsystem should answer a premise. If it doesn't,
it's either redundant or should be reframed.

| Premise | Existing Aragora substrate | Measurable property |
|---------|----------------------------|---------------------|
| Bandwidth | Batched-triage advisory packets (#6279); review-queue CLI + UI (#6280 / #6288 / #6328); settlement loop (#6297) | PRs settled per session; time-per-settlement |
| Trust | Heterogeneous-model ensembles; Arena debate engine; circuit breaker; airlock; task sanitizer; trickster; cross-verification | Dissent-surfacing rate; hallucination-catch rate |
| No safe delegation | Human settlement gate in merge-arbiter; EU AI Act Article 14 wedge (H1-05); advisory-only machine review (#6279) | Percent of merges with human settlement; override-correlated-with-outcome rate |
| Structure | Belief network (claims + provenance); reasoning module; PR review protocol scaffold (#6355); planned Brief schema | Percent of decisions with decomposed cruxes; structure-completeness score |
| Outcomes | ELO tracking; persona evolution; calibration tracker; outcome feedback; selection feedback; receipt store | Calibration error; per-agent track record; outcome-weight half-life |
| Tested against reality | Benchmark corpus rev-4; B0 truth publication; proof-first queue; gauntlet receipts; evidence staleness | Zero-rescue rate on bounded tasks; claim verification rate |
| Adversarial cross-check | Arena topologies; Prover-Estimator consensus; rhetorical observer; trickster; Recursive Language Models | Ensemble-vs-single-model delta; dissent preservation rate |
| Distillation | PDB UI v0 (#6328); batched review-queue (#6288); progressive disclosure (brief A/B/C densities); receipt summaries | Time-to-decision; brief-coverage-of-load-bearing-structure |
| Informed consent / feedback | Settlement signals (#6297); BriefReceipt + SettlementLinkage (#6353); dissent preservation in receipts | Signal-to-settlement rate; human-override outcome correlation |
| Self-test on own codebase | Nomic loop; self-develop CLI; H1 dogfood wedge; this review-queue rollout | H1 exit criteria; dogfood session cadence |

**The strongest proof point is that the product is being applied to
itself.** The arc from problem statement → heterogeneous critique →
declaw of the auto-approver → design-doc-first discipline → settlement-
loop rollout is a worked example of the thesis in action. This very
document was produced by the same loop: three AI agents in adversarial
dialogue, human arbitration, structured output, committed as evidence.

---

## Commitments this thesis makes

Four concrete commitments follow from taking the thesis seriously:

1. **No auto-merge of substantive code without human settlement.**
   Removing the human gate defeats premise 3. EU AI Act Article 14
   alignment falls out of this as a natural consequence, not a bolt-on.
2. **Dissent is preserved in receipts, not collapsed into majority.**
   Majority rule without a dissent trail is indistinguishable from
   false consensus.
3. **Outcome feedback is the measurement of whether the thesis holds.**
   If calibration scores do not improve over time, or if heterogeneous
   ensembles do not surface dissent that changes decisions, the product
   has failed its own test and must be revised rather than shipped.
4. **The limits named in "Where this thesis does NOT yet apply" are
   respected in product scope.** Aragora will not claim to arbitrate
   what it cannot arbitrate, even when asked.

---

## What this replaces and what it does not

- **Replaces:** the scattered top-level framing across
  `WHY_ARAGORA.md`, `CANONICAL_GOALS.md` intros, `EXTENDED_README.md`
  openers, and `FEATURE_DISCOVERY.md` preambles. Those remain for
  operational detail; their introductions should cite this doc as
  source of authority.
- **Does not replace:** the 3-horizon roadmap, architecture references,
  feature catalogs, or operational runbooks. Those encode *how*. This
  doc encodes *why*.

---

## Single sentence

**Aragora is infrastructure for truth-seeking when AI output outpaces
human review and cannot be safely trusted — decomposing claims into
structure, cross-checking them adversarially across heterogeneous
lenses, weighting them by outcomes, and distilling the result to a form
humans and AI agents can actually use — starting with its own
codebase and ending wherever the pattern generalizes.**
