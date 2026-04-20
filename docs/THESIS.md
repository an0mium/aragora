# The Aragora Thesis

> Canonical source of authority. Every other strategic doc links up to this.
> Last updated: 2026-04-20. Status: v1 draft, revised in response to
> codex adversarial review (REQUEST_CHANGES verdict, 5 required changes
> applied); awaiting second review before merge.

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
Whether the pattern generalizes beyond software execution is a
load-bearing assumption, not a promise (see § Load-bearing assumptions).

---

## The five premises

1. **Bandwidth.** AI produces more output than humans can meaningfully process.
2. **Trust.** AI output is systematically untrustworthy — from adversarial
   inputs, biased or poisoned training data, spiky capabilities,
   hallucinations, knowledge cutoffs, and individual-model blind spots.
3. **No safe delegation.** Neither humans nor AI agents can safely defer
   to any single other agent's judgment, including their own.
   Convergence across agents only counts as evidence when the agents
   have *different priors*, *different evidence*, and *active incentive
   to dissent* — homogeneous convergence (multiple models trained on
   similar data agreeing) is spurious.
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
   with different training, architectures, and blind spots — combined
   with actively engineered input-diversification (rotated prompts,
   separate retrieval paths, provider-differentiated tooling) and
   explicit dissent incentives, surface what single models miss.
   Formal heterogeneity without input diversity is fake heterogeneity
   and does not satisfy premise 3. Dissent is first-class and
   preserved, not collapsed into a false consensus.
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

## What we mean by "true"

This thesis commits to an operational meaning of "relatively more true,
less wrong" rather than a metaphysical one. Four tiers of claim, each
with a different evidential basis, each separately expressible in a
decision receipt:

1. **Agent-relative belief quality** — internal model output that
   reduces A's surprise and improves A's decisions under A's goals and
   constraints. Colloquially "truth for A," but strictly this is
   instrumental fit, not truth in the correspondence sense. Naming it
   precisely avoids relativist drift.
2. **Convergent truth** — the subset of agent-relative truths that
   remain stable under heterogeneous adversarial cross-checking by
   agents with different priors, different evidence, and active
   incentive to dissent.
3. **Operational objective truth** — the subset of convergent truth
   that continues to predict successfully under out-of-distribution
   interventions and over long time horizons. Today, Aragora can
   plausibly emit tier-3 claims only in narrow domains with short
   feedback loops and observable interventions (bounded software
   tasks). Tier-3 claims in broad or long-horizon domains are
   aspirational and must be flagged as such.
4. **Metaphysical objective truth** — the hypothesized structure of
   reality that best explains why (3) continues to hold. The product
   does not claim direct access to this tier; it bets that (3)
   approximates it.

A finite system can emit claims at tiers (1)–(3). It cannot emit (4).
Aragora commits to labeling which tier any given output occupies
rather than marketing all outputs as unqualified "truth." A receipt
saying *"convergent across five heterogeneous lenses with dissent
preserved"* is a weaker and more honest claim than *"true,"* and also
a more actionable one.

This position has precedent in pragmatism and convergent-inquiry
epistemology; the innovation is architectural, not philosophical,
namely building it as shipping software rather than essay. Readers
who want the academic mapping can see the footnote at the bottom of
this document.

[^philosophy]: Closest precedent: Charles Sanders Peirce's long-run
convergent inquiry (1878). Shares instincts with pragmatism (James,
Dewey) and predictive-processing (Friston). Explicitly not Tarski-style
correspondence — Aragora does not claim agent-independent access to
truth, only convergence under adversarial constraints.

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
- **Closed belief systems that maintain surprise-reduction through
  hermeneutic reinterpretation rather than prediction.** A framework
  that explains away anomalies after they occur is not the same as a
  framework that predicts them in advance. Truth-seeking machinery
  works on claims testable under genuine intervention pressure and
  out-of-distribution prediction; it does not adjudicate beliefs that
  survive by being unfalsifiable.
- **Fake heterogeneity from shared context.** Multiple frontier models
  can share correlated failure modes, especially when they consume the
  same context bundle, retrieval sources, tool outputs, or prompt-
  injection vectors. The heterogeneity required by premise 3 degrades
  into theater if the agents are formally diverse but epistemically
  collapsed by shared inputs. Genuinely independent challenge must be
  actively engineered (separate retrieval, rotated prompts, provider-
  differentiated tooling, adversarial prompting across lenses), not
  assumed from the provider list.

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
| Ensembles produce genuinely independent challenge in practice, not just formal heterogeneity | Shared-context contamination probe: feed poisoned / adversarial context to a panel; measure percent of lenses that catch it vs correlate on failure. Success: >60% of lenses independently flag; <30% catastrophic correlation | H1–H2 |
| Cryptographic receipts produce trust that matters to buyers | Controlled with-receipt vs without-receipt A/B on identical decisions: approval-latency delta, pilot continuation rate, willingness-to-deploy under matched evidence. Willingness-to-pay alone is confounded and not sufficient. | H2 |
| The pattern generalizes beyond software execution | Success: at least one non-software domain wedge reaches tier-3 (operational objective truth, see § "What we mean by true") under its domain's intervention schedule, without domain-specific hand-tuning that fails to transfer. Failure: wedges either fail to converge or converge only via per-domain engineering that blocks generalization. | H3 |

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

Each substrate item is tagged `[shipped]`, `[scaffolded]`, `[docs-only]`,
or `[planned]`. The table must survive skeptical reading; status is
explicit rather than implied.

| Premise | Existing Aragora substrate | Measurable property |
|---------|----------------------------|---------------------|
| Bandwidth | Batched-triage advisory packets (#6279) `[shipped]`; review-queue CLI (#6280, #6288) `[shipped]`; PDB UI v0 (#6328) `[scaffolded]`; settlement loop (#6297) `[shipped]` | PRs settled per session; time-per-settlement |
| Trust | Heterogeneous-model ensembles `[shipped]`; Arena debate engine `[shipped]`; circuit breaker, airlock, task sanitizer, trickster, cross-verification `[shipped]` | Dissent-surfacing rate; hallucination-catch rate |
| No safe delegation | Human settlement gate in merge-arbiter `[shipped]`; advisory-only machine review (#6279) `[shipped]`; EU AI Act Article 14 wedge (H1-05) `[docs-only]` | Percent of review-queue-path merges with human settlement; override-correlated-with-outcome rate |
| Structure | Belief network (claims + provenance) `[shipped]`; reasoning module `[shipped]`; `ReviewBrief` schema in `aragora/review/protocol.py` `[shipped]`; `BriefReceipt` + `SettlementLinkage` (#6353) `[shipped]`; PR review protocol packet scaffold (#6355) `[scaffolded]` | Percent of decisions with decomposed cruxes; structure-completeness score |
| Outcomes | ELO tracking `[shipped]`; persona evolution `[shipped]`; calibration tracker `[shipped]`; outcome feedback loop `[shipped]`; selection feedback `[shipped]`; receipt store `[shipped]` | Calibration error; per-agent track record; outcome-weight half-life |
| Tested against reality | Benchmark corpus rev-4 `[shipped]`; B0 truth publication `[shipped]`; proof-first queue `[shipped]`; gauntlet receipts `[shipped]`; evidence staleness `[shipped]` | Zero-rescue rate on bounded tasks; claim verification rate |
| Adversarial cross-check | Arena topologies `[shipped]`; Prover-Estimator consensus `[shipped]`; rhetorical observer `[shipped]`; trickster `[shipped]`; Recursive Language Models `[shipped]` | Ensemble-vs-single-model delta; dissent preservation rate |
| Distillation | Batched review-queue (#6288) `[shipped]`; PDB UI v0 (#6328) `[scaffolded]`; receipt summaries `[shipped]`; progressive disclosure (brief A/B/C densities) `[planned]` | Time-to-decision; brief-coverage-of-load-bearing-structure |
| Informed consent / feedback | Settlement signals (#6297) `[shipped]`; `BriefReceipt` + `SettlementLinkage` (#6353) `[shipped]`; dissent preservation in receipts `[shipped]` | Signal-to-settlement rate; human-override outcome correlation |
| Self-test on own codebase | Nomic loop `[shipped]`; self-develop CLI `[shipped]`; H1 dogfood wedge `[in progress]`; this review-queue rollout `[in progress]` | H1 exit criteria; dogfood session cadence |

**The strongest proof point is that the product is being applied to
itself.** The arc from problem statement → heterogeneous critique →
declaw of the auto-approver → design-doc-first discipline → settlement-
loop rollout is a worked example of the thesis in action. This very
document was produced by the same loop: multiple AI agents in
adversarial dialogue (including a codex review that requested changes
and was applied as a third commit), human arbitration, structured
output, committed as evidence.

---

## Commitments this thesis makes

Four concrete commitments follow from taking the thesis seriously:

1. **No auto-merge of substantive code from the thesis-conformant
   review-queue path without human settlement.** Removing the human
   gate on the heterogeneous review path defeats premise 3. EU AI Act
   Article 14 alignment falls out of this as a natural consequence,
   not a bolt-on.

   **Known exceptions, named honestly:** the repo currently retains
   two pre-thesis auto-merge paths that this commitment does *not*
   yet cover: (a) `fire_and_forget` low-risk auto-merge in
   `aragora/swarm/tranche_integrate.py` and (b) `admin_merge_allowed`
   admin-merge bypass documented in `docs/STATUS.md`. These are
   grandfathered legacy paths from before #6279. They are not
   endorsed by this thesis. Their scope, usage, and continued
   existence is itself a load-bearing product question that must be
   resolved (either by gating them behind human settlement or by
   removing them) before the thesis's Commitment #1 can be claimed
   without qualification. Failure to resolve this contradicts the
   thesis on its own first commitment.
2. **Dissent is preserved in receipts, not collapsed into majority.**
   Majority rule without a dissent trail is indistinguishable from
   false consensus.
3. **Outcome feedback is the measurement of whether the thesis holds.**
   Specifically: if per-agent calibration error does not decrease over
   any rolling 30-day window on the benchmark corpus, or if the
   fraction of panel decisions where dissent materially changes the
   human verdict drops below 15% over a rolling 30-day window, the
   product has failed its own test on that window and must be revised
   (either architecturally, via input-diversification; or
   operationally, via expanded panel heterogeneity) before shipping
   further capability on top. Thresholds are provisional and subject
   to recalibration after 30 days of real settlement data.
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
