# The Adjacent Possible

**Exploratory vision extensions. Not roadmap commitments.**

This document sketches four candidate extensions of the Aragora thesis. None is
scheduled, resourced, or promised. Each is included because it passes the same
prioritization filter the platform already runs on: **invest where each unit of
work makes the decision receipt more load-bearing.** The receipt — a verifiable,
dissent-preserving record that a decision was adversarially examined — is the
moat ([WHY_ARAGORA.md](../WHY_ARAGORA.md)); the doctrinal frame these extend is
[CANONICAL_GOALS.md](../CANONICAL_GOALS.md), especially Pillar 5 (Cryptographic
Receipts and Auditability) and the Epistemic CI direction. Anything here that
stops strengthening the receipt should be cut without ceremony.

---

## 1. The Disagreement Atlas

Every settled PR in this repository leaves behind something no lab has at
scale: naturalistic, adjudicated records of frontier models disagreeing about
real code under real stakes — verdicts at exact head SHAs, severity-tagged
findings, dissent that was later proven right or wrong by an adjudication and
by what actually happened to the code. Benchmark datasets are synthetic;
RLHF preference data is single-turn and private; nobody has merge-gate-scale
disagreement data with ground-truth resolution attached. The extension:
publish this as a living, receipted dataset and benchmark — which model
families dissent about what, how often dissent is vindicated, what
adjudication decided and why. It composes directly with the review-failure
taxonomy artifact and gives the adjudicator an evaluation corpus it currently
lacks. Each published disagreement is a receipt made load-bearing twice: once
as a merge record, once as a research artifact a stranger can verify.

## 2. Institutional epistemics as infrastructure

Decision receipts currently attest that a decision *was examined*. The
extension makes them attest that a decision *is still valid*: receipts for
human and organizational decisions that carry explicit falsification
conditions and check-by dates, and that **reopen themselves** when a
kill-switch observation fires — the harvest engine (#8760) already sweeps
settled work for follow-ups, and the falsification-condition machinery (#8815)
already gives claims testable failure modes; composing them turns "we decided
X in Q2" from a stale wiki page into a live object that files its own
reopening issue when its premises die. No docs tool, OKR tracker, or decision
log approaches this: they all record decisions as text, not as claims with
enforcement. This is the "Time Is Part of Settlement" doctrine made into an
organizational primitive, and it makes every receipt load-bearing for as long
as the decision it records stays consequential.

## 3. Time-aware truth

A recurring failure class in this project's own operations is staleness
masquerading as truth: changelogs that lag releases, CDN caches serving dead
docs, merge commits that reference vanished branches, baselines that drift
from reality. The pattern generalizes — most false claims in an organization
were once true. The extension is a freshness layer for the knowledge and
receipt substrate: every claim carries a last-verified timestamp and a
staleness policy (how long before this claim degrades from *asserted* to
*unverified* to *presumed stale*), with re-verification as a schedulable,
receipted act. The pulse subsystem (`aragora/pulse/`, `freshness.py`) already
scores freshness for trending sources; extending that discipline to internal
claims makes the receipt time-aware — a receipt that says *when* it was last
true is strictly more load-bearing than one that only says it was true once.

## 4. Epistemic middleware

The interrogation batteries, falsifier questions, crux-finding, and
adversarial review that gate this repository's merges are currently packaged
as an application. The extension exposes them as a service boundary any agent
framework can call: **"have Aragora doubt this before you act on it."** A
LangGraph pipeline, a CrewAI crew, or a bare tool-using agent submits a claim,
a plan, or a diff; Aragora returns severity-tagged dissent, the load-bearing
cruxes, and a receipt — without the caller adopting Aragora's orchestration,
memory, or UI. This is the Pillar 8 (co-equal consumers) posture applied to
the doubt machinery itself, and it is the purest expression of the moat rule:
every external system that routes decisions through the doubt service is a
system whose actions are now backed by an Aragora receipt.

---

## The filter, restated

Four extensions, one test each unit of work must pass: does it make the
receipt more load-bearing — carried by more decisions (Atlas), for longer
(institutional epistemics), with truer timestamps (time-aware truth), across
more systems (middleware)? Extensions that grow the platform without growing
the receipt's weight are scope creep by the
[CANONICAL_GOALS.md](../CANONICAL_GOALS.md) product-boundary rule, however
attractive they look.

*Exploratory. Revisit when the active operating focus (reliable autonomous
execution + receipt legibility) is boringly stable.*
