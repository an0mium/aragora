# Operator Questions

**How to ask questions that make agent fleets investigate and act well.**

Agents with tools are only as good as the questions that direct them. This guide
collects the question patterns that reliably produce investigation, verification,
and useful action — and the patterns that quietly produce nothing.

---

## The core pattern: evaluate, with permission to act

There are three ways to task an agent:

| Form | Example | Failure mode |
|---|---|---|
| Execution order | "Add a retry to the fetcher" | Executes even when the premise is wrong |
| Opinion request | "Do you think our retries are good?" | Produces prose, changes nothing |
| **Evaluative + act** | "Is our retry behavior actually correct under provider outages? Verify it, and fix what you find." | — |

The third form wins because it forces the agent to establish ground truth
before acting, and removes the round-trip where it reports findings and waits
for permission. The general template:

> **"Is X true / good / necessary? Go find out. Proceed on what you find."**

The evaluation makes the agent investigate; the attached permission makes the
investigation consequential. Either half alone underperforms.

## Outsider-falsification templates

These questions all share one move: they take a claim you believe and demand
that it survive contact with someone (or something) that doesn't share your
context.

- **Prove it for a stranger.** "Prove claim X is true for someone who isn't
  us — no internal context, no charity. If it isn't provable, make it true."
- **The stranger test.** "A competent developer with zero context lands on our
  README. Walk their path. Where do they stall, and what do they conclude we
  are?"
- **The moat audit.** "What can we prove that no competitor can fake? Verify
  that proof end-to-end, as an outsider would, before we claim it."
- **Path-to-value trace.** "Trace the shortest real path from 'never heard of
  us' to 'got value.' Time every step. Which step kills the most people?"
- **The deletion question.** "If we deleted X tomorrow, what would actually
  break, and who would notice? If the answer is 'nothing and nobody,' why does
  X exist?"
- **Belief-reality audit.** "List five things we believe about the system but
  haven't verified in the last 30 days. Verify them now. Report which were
  false."
- **Boring but load-bearing.** "What is the least glamorous component the whole
  system depends on? When did anyone last check it deliberately?"

## Questions to ask any frontier model (that people don't)

Models answer what you ask. These questions change what an answer *is*:

1. **"What did you NOT verify?"** — Forces the assumption surface into the
   open. Every confident answer has one.
2. **"What would make this wrong, and how would we know within N days?"** —
   Attaches a falsification condition and a check-by date to the claim, instead
   of accepting it as timelessly settled.
3. **"Give me the strongest case against, before I accept this."** — Cheap
   adversarial pass. Models are good at this when asked and silent when not.
4. **"Go look before answering."** — For any agent with tools: prohibit
   answering from priors when the ground truth is one command away.
5. **"What do you need from me?"** — Surfaces missing credentials, ambiguous
   scope, and blocked decisions *before* the agent improvises around them.
6. **"What's the smallest version that proves it?"** — Replaces a plan with an
   experiment.
7. **"What did you assume?"** — Post-hoc twin of #1; ask it after delivery,
   every time.
8. **"Should this be done at all?"** — The question execution orders skip.
   Agents will faithfully build the wrong thing forever if you let them.

## Anti-patterns

**Date-gated planning.** A plan that says "review on July 29" deserves the
question: *why July 29 and not today?* If the date exists because data genuinely
arrives then (a 30-day measurement window, an external deadline), it's a real
data window — keep it. If the date is decorative, it's a schedule artifact: the
work is checkable now and the date is just deferred attention. Most review
dates in agent-written plans are schedule artifacts. Ask "what will we know
then that we don't know now?" — if the answer is "nothing," review it today.

**Passive waiting on autonomous systems.** When an autonomous pipeline goes
quiet, the operator failure mode is waiting politely for it to resume. The
correct question is immediate and evaluative: *"Why is nothing happening? Go
find the actual blocker."* Silence is data. Queues that never drain, gates that
never settle, and loops that never loop all have specific, findable causes —
but only if someone asks. (This repo's queue-drain investigation found a
645-branch orphaning that had been silently blocking harvest for weeks; the
unlock was someone finally asking why the backlog never moved.)

## Putting it together

A good operator prompt usually reads like this:

> "We claim [X]. Verify it as a stranger would, end to end. Tell me what you
> did NOT verify. If the claim is false or unprovable, make it true or narrow
> the claim — proceed on what you find, and give me the strongest case against
> whatever you conclude."

One sentence of claim, one sentence of falsification demand, one sentence of
permission. That structure — not model choice, not prompt length — is what
separates fleets that investigate from fleets that generate plausible text.

---

*See also: [WHY_ARAGORA.md](../WHY_ARAGORA.md) for why adversarial challenge is
the product's core stance, and
[artifacts/2026-07-decision-integrity-dogfooding.md](../artifacts/2026-07-decision-integrity-dogfooding.md)
for what these questions look like when a merge gate asks them automatically.*
