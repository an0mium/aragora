# Round 30f — H2 Candidate-Selection Rubric

*Author:* Factory/Claude, Round 30f planning lane.
*Status:* DRAFT — pending β verdict and a future measured #6375 threshold outcome. #6898 landed the conservative δ insufficiency surface but did not close #6375.
*Purpose:* Score candidate external repositories for the **first H2 pilot** that will run *only after* Round 30f's δ and β both pass (or after explicit operator decision to accept their failure modes per §6 of `2026-04-30f-round-spec.md`).

This document **scores; it does not recommend.** The operator picks from the shortlist. The rubric is here so the pick is reviewable rather than vibed.

---

## 1. Five-axis rubric (each 0-3, total /15)

### Axis A — Friction-of-access

How hard is it for a Droid/agent-bridge to file a review packet against this repo?

| Score | Meaning |
| --- | --- |
| 3 | Public repo, formal PR review process documented, contributor onboarding clear, bots not blocked. |
| 2 | Public repo, review process informal but maintainers responsive (median time-to-review ≤ 7 days). |
| 1 | Public repo, no documented review process or maintainers slow (>14 days). |
| 0 | Closed/private repo, or contributor process requires CLA/IP-assignment friction. |

### Axis B — Distributional distance from Aragora

How different is this repo from our own codebase? (We want **maximum distance** to test out-of-distribution generalization.)

| Score | Meaning |
| --- | --- |
| 3 | Different domain (not LLM/agent infra), different language stack (not Python-primary), different team size (small vs our 1-3-person workflow), different review volume. |
| 2 | Two of the four "different" axes above. |
| 1 | One of the four. |
| 0 | Same domain (another agent/LLM substrate) — high overlap, weak signal. |

### Axis C — Receipt-receivable

Will the maintainer accept a structured PR-review packet (as a PR comment, structured JSON, or signed receipt link) without rejecting it as "bot noise"?

| Score | Meaning |
| --- | --- |
| 3 | Maintainer explicitly receptive (e.g., already uses review bots, has docs welcoming structured reviews). |
| 2 | Maintainer neutral (no stance; community is collaborative). |
| 1 | Maintainer wary (community has had bot-spam problems; receipts must be substantive or be flagged). |
| 0 | Maintainer hostile or repo has explicit "no bots" policy. |

### Axis D — Outcome-observable

Can we measure the receipt's effect on downstream events?

| Score | Meaning |
| --- | --- |
| 3 | Full GitHub timeline available; PR merged/rejected/follow-up incidents/reverts all queryable; maintainer often cites review reasoning explicitly. |
| 2 | Full GitHub timeline; outcome inferable but not always cited. |
| 1 | Partial timeline (e.g., reviews happen off-platform) or sparse activity. |
| 0 | No way to observe outcome (private merges, no incident-tracking). |

### Axis E — Risk-asymmetry

What is the worst-case if our review packet is wrong, low-signal, or actively harmful?

| Score | Meaning |
| --- | --- |
| 3 | Clearly bounded — withdrawal procedure documented; maintainer can mute/ignore at will; no reputational/legal consequence; single-PR scope. |
| 2 | Mostly bounded — bad review wastes maintainer time; recoverable. |
| 1 | Some asymmetric risk — bad review on a high-traffic repo could create reputational damage to Aragora/Synaptent. |
| 0 | Catastrophic asymmetry — e.g., security-critical repo where wrong review could mask a vulnerability; legal/regulated domain. |

### Composite score

Total = A + B + C + D + E, range 0-15. **Threshold for shortlist: ≥ 10.** No candidate <10 should be considered for the first H2 pilot regardless of strategic appeal — the failure-asymmetry is too high while the signal-quality of the rubric is still untested.

---

## 2. Pre-registered H2 success criterion

The first H2 pilot is **successful** iff *all four* of the following hold within 30 days of the receipt being filed:

1. Receipt is produced for ≥1 PR in the target repo.
2. Receipt is *received* — i.e., posted to the PR (as a comment / linked artifact) without being deleted by the maintainer or auto-flagged as spam.
3. Receipt is *acted upon* — i.e., maintainer either accepts (incorporates a finding into the PR or final merge), rejects (replies engaging substantively), or substantively engages (asks a clarifying question, requests revision). "Ignored" or "auto-archived" does NOT count as acted upon.
4. Outcome is recorded — i.e., final disposition (merged / rejected / abandoned) is captured in the receipt store within 30 days.

AND **at least one** of the following load-bearing observations:

5. Receipt's specific finding is cited in the maintainer's merge/rejection rationale.
6. Receipt's named issue is caught by CI or further review on the same PR.
7. Receipt's named issue surfaces in a post-merge incident, regression, or revert within 90 days.

This is asymmetric: the first 4 are mandatory; the 5th-7th are evidence the receipt was *load-bearing*, not just received.

---

## 3. Pre-registered H2 failure modes

Any of these counts as a **honest failure**, not a "redo with different framing":

- **Ignored entirely** — receipt posted, no engagement within 30 days. (Suggests receipt format is illegible to humans.)
- **Rejected as low-signal** — maintainer explicitly says the receipt is noise. (Suggests heterogeneity-panel claims are weaker than expected.)
- **Receipt acted upon but wrongly** — maintainer accepts a flagged issue that turns out not to be real, causing a regression. (Suggests false-positive rate in production is higher than β probe estimated.)
- **Receipt's claims fail to land** — maintainer engages, agrees, but the issue does not surface in any subsequent CI/review/incident. (Suggests claimed issues are real but unimportant.)

A failed H2 pilot is **a successful experiment**. It generates concrete evidence about receipt quality that no amount of internal dogfooding can.

---

## 4. Candidate shortlist (scored, not recommended)

**Important:** I do not have visibility into the operator's actual professional/employer GitHub access or specific friendly-OSS relationships. The candidates below are *categories* with concrete examples. The operator should refine them with specific repo URLs before Round 30g.

### Category 1 — Synaptent properties

| Candidate | A (friction) | B (distance) | C (receivable) | D (observable) | E (risk) | Total | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `synaptent/rocket-rocks` | 3 | 2 | 3 | 3 | 3 | **14** | Internal repo? Same org, low B. Strong A/C/D/E. |
| `synaptent/<other>` | ? | ? | ? | ? | ? | ? | Operator should fill in actual properties. |

**Recommendation if Category 1 is picked:** treat the receipt as an internal stress-test, not a load-bearing H2 result. Synaptent properties have low B — they don't test out-of-distribution generalization. They are a useful *warm-up* before a true H2 pilot but should not be reported as the H2 wedge.

### Category 2 — Friendly OSS

| Candidate (example) | A | B | C | D | E | Total | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `httpx/httpx` (Python HTTP client) | 3 | 2 | 2 | 3 | 3 | **13** | Active, formal review, but Python+infra-adjacent. |
| `openai/openai-python` | 3 | 1 | 1 | 3 | 1 | **9** | Below threshold — too LLM-adjacent + bot-wary. |
| `pydantic/pydantic` | 3 | 2 | 3 | 3 | 3 | **14** | Strong all-around. Different team, different domain (data validation). |
| `astral-sh/ruff` | 3 | 3 | 2 | 3 | 3 | **14** | Rust, lint domain, large team, very different. |
| `nushell/nushell` | 3 | 3 | 2 | 3 | 3 | **14** | Rust shell, totally orthogonal domain. |
| `redis/redis-py` | 3 | 2 | 2 | 3 | 3 | **13** | Maintainer responsive, but DB-client domain only mildly distant. |

**Recommendation among OSS:** `astral-sh/ruff` (Rust lint, maximum distributional distance, team famously open to structured feedback) or `pydantic/pydantic` (Python validation, different domain than agent infra, very responsive maintainer). Both score 14/15.

### Category 3 — Operator's professional/employer domain

| Candidate | A | B | C | D | E | Total | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `<operator-employer>/<repo>` | ? | ? | ? | ? | ? | ? | **Operator must fill in.** Likely high B (different domain than Aragora). E may be low (reputation risk). |

**Recommendation among employer category:** highest distributional value but highest reputational asymmetry. Should be picked only with explicit operator consent and a withdrawal/disclaimer plan.

---

## 5. Decision-tree handoff to Round 30g

After Round 30f resolves, the operator picks one candidate from the **shortlist of repos scoring ≥ 10** AND meeting these tiebreakers:

1. Highest B (distributional distance) wins among ties.
2. Among equal B: highest C (receivable) wins.
3. Among equal C: highest E (risk asymmetry, low-risk preferred) wins.

If no candidate scores ≥ 10, **the operator does not run an H2 pilot in Round 30g.** Instead, Round 30g focuses on rubric-revision (e.g., we may have over-scored Axis E, or we may need to find candidates the operator has personal relationships with). H2 pilot waits until at least one candidate scores ≥ 10.

---

## 6. What this rubric is NOT

- **Not a recommendation.** I score; the operator decides.
- **Not a substitute for operator judgment about specific repos.** I cannot see the operator's professional relationships, prior commitments, or strategic considerations.
- **Not load-bearing on the first pilot succeeding.** The first H2 pilot exists to *test the receipt-quality hypothesis*. Failure is informative.
- **Not a one-time exercise.** This rubric is iterated after each H2 pilot to incorporate observed failure modes.

---

## 7. Why this matters

The 2026-04-25 reassessment named three risks the project has been carrying:

1. **Substrate-without-application risk.** 657 commits in 14 days, all internal.
2. **Heterogeneity-theater risk.** Panel mechanism shipped, contamination probe not run.
3. **Receipt-without-recipient risk.** Receipts produced, never published to non-Aragora context.

The H2 pilot directly addresses risk 3. The β probe addresses risk 2. The δ #6375 closure addresses the H1 hold-out. Round 30f is the first round where all three risks are simultaneously on the bounded queue rather than being deferred.

The project earns the right to claim "Decision Integrity Platform" only after at least one external receipt has been produced, received, and acted upon by a maintainer outside Synaptent. This rubric is the runway to that pilot.

— Round 30f planning lane (Factory/Claude), 2026-04-30.
