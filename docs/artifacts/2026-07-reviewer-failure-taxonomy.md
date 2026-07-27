# When Reviewers Are Wrong: A Taxonomy of Multi-Model Review Failures, With Receipts

**Date:** 2026-07-05
**Repo:** [synaptent/aragora](https://github.com/synaptent/aragora)
**Audience:** ML/eval engineers evaluating LLM-as-reviewer systems, and anyone
deciding whether to trust an "AI review gate" with merge authority.
**Companion to:** [Decision Integrity, Dogfooded](2026-07-decision-integrity-dogfooding.md)
(the same gate's successes over 30 days).

---

## The claim

Every vendor publishes the bugs their AI reviewers caught. Nobody publishes the
times their AI reviewers were confidently wrong. This document does the second
thing, from four days of this repository's public merge-gate record
(2026-07-02 to 2026-07-05): six distinct reviewer failure classes, one
receipted case study each, the exact machine refutation used against each, and
the resolution mechanism that unstuck the pipeline — plus the control group of
reviews where the same gate worked exactly as designed.

The headline, stated up front and defended below: **in every documented failure,
the reviewer error was a false negative — a true claim wrongly doubted. We
found no case in this window where a reviewer PASS vouched for a claim that
was false.** The gate errs toward doubt, not false trust. That asymmetry is
the property you actually want from a merge gate, and it is measurable.

Everything here is drawn from public PR/issue comments of this repository or
from evidence-collector records committed alongside this document as
machine-readable eval fixtures
([`tests/governance/fixtures/adjudicator_eval_cases.json`](../../tests/governance/fixtures/adjudicator_eval_cases.json),
10 cases, verbatim reviewer bodies at exact head SHAs). Every quote links to
its source; where a quoted review body was produced by the evidence collector
in a prepare-only round (recorded but not posted verbatim to the thread), the
fixture file is the committed source and the linked thread comment restates the
finding.

## The setup, in one paragraph

Merges in this repository are gated by adversarial review from independent
frontier-model families — Claude (Anthropic) and GPT (OpenAI) as the counting
"western frontier" pair — each producing a verdict (`PASS` /
`CHANGES-REQUESTED`) with per-finding severities (`[P0]`–`[P3]`) at the exact
PR head SHA. [Severity-gated dissent](https://github.com/synaptent/aragora/pull/8574)
makes `[P0]`/`[P1]` blocking and `[P2]`/`[P3]` advisory (non-blocking but
preserved, never discarded); the
[tiered gate](https://github.com/synaptent/aragora/pull/8638) scales required
evidence to blast radius; Tier 3–4 changes always end in recorded human risk
acceptance. The full mechanism, with its 30-day numbers, is in the
[companion artifact](2026-07-decision-integrity-dogfooding.md). This document
is about the rounds where the reviewers — not the code — were the problem.

## The window

Seven PRs, 2026-07-02 to 2026-07-05, all ultimately merged. Roughly 30 review
rounds and ~60 reviewer verdicts; the fixture file commits 20 verbatim bodies,
and the remainder are receipted through in-thread evidence-round comments that
quote them.

| PR | Rounds | What happened | Outcome |
|---|---|---|---|
| [#8824](https://github.com/synaptent/aragora/pull/8824) | 7 | Both reviewers repeatedly reported a README link dead; the target file existed on `main` the whole time | Merged after machine refutation + absolute-URL premise removal |
| [#8834](https://github.com/synaptent/aragora/pull/8834) | 4 | One reviewer read a stale PyPI index; the other called a UTC date "in the future" from its local clock | Merged after timestamped refutation; one dissent premise self-expired at local midnight |
| [#8800](https://github.com/synaptent/aragora/pull/8800) | 4 | One reviewer re-raised its round-1 finding after the demanded audit was delivered | Operator-settled with the other family's counting PASS |
| [#8802](https://github.com/synaptent/aragora/pull/8802) | 8 | Findings alternated between reviewers for seven adversarial rounds, several targeting a file the PR was fenced from touching; three rounds found real bugs (one a genuine security bug) | Merged; out-of-scope findings re-filed as [#8810](https://github.com/synaptent/aragora/issues/8810) |
| [#8811](https://github.com/synaptent/aragora/pull/8811) | 3 | **Control:** both reviewers converged on the same real harmlessness bug in round 1 | Fixed; clean 2-0 PASS in round 3 |
| [#8852](https://github.com/synaptent/aragora/pull/8852) | 3 | **Control:** three real `[P2]`s in round 1, including a reproduced WAL edge case | Fixed; clean 2-0 PASS twice |
| [#8803](https://github.com/synaptent/aragora/pull/8803) | 1 | **Control:** clean 2-0 PASS, zero findings, single round | Merged |

## The taxonomy

| # | Failure class | Specimen | Who erred | Resolution mechanism |
|---|---|---|---|---|
| 1 | Diff-blind grounding | #8824 r1–r3 | both families | evidence-post + premise removal; grounding fix filed as [#8825](https://github.com/synaptent/aragora/issues/8825) |
| 2 | Stale-external-world grounding | #8834 r1–r2 | openai | timestamped fresh-fetch refutation |
| 3 | Temporal reasoning | #8834 r1–r2 | claude | timezone-explicit timestamps + premise self-expiry |
| 4 | Verbatim-repeat dissent | #8800 r3 | openai | severity-gating + operator adjudication |
| 5 | Out-of-scope carousel | #8802 r4/r6 | both families | re-filing (finding preserved as [#8810](https://github.com/synaptent/aragora/issues/8810)) |
| 6 | Cross-family contradiction | #8802 r5 vs r7 | both, by definition | human adjudication (reference-implementation parity) |
| — | Control: convergent real findings, clean passes | #8811, #8852, #8803 | nobody | the gate, working |

### 1. Diff-blind grounding — the reviewer reasons over the diff, not the tree

**Case: [#8824](https://github.com/synaptent/aragora/pull/8824), rounds 1–3.**
A docs-only PR linked `docs/artifacts/2026-07-decision-integrity-dogfooding.md`
from the README. That file had been on `main` since
[#8801](https://github.com/synaptent/aragora/pull/8801) (merged
2026-07-03T05:25Z, blob `3c0458f1`). Both reviewers, for three consecutive
rounds, reported the link dead. Round 1, claude, as a blocking `[P1]`:

> "The marquee evidence link `[Decision Integrity, Dogfooded](docs/artifacts/2026-07-decision-integrity-dogfooding.md)`
> is dead: neither the file nor the `docs/artifacts/` directory exists at this
> head, and this PR changes only `README.md`"
> — claude review at head `1bbf572` (fixture `pr8824_r1_diff_blind_grounding`)

Round 3, openai's wording exposes the mechanism:

> "this PR's complete changed-file list does not include that artifact and the
> reviewed checkout has no tracked target at that path"
> — openai review at head `75893e9` (fixture `pr8824_r3_diff_blind_unanimous`;
> quoted in the [round-3 disposition comment](https://github.com/synaptent/aragora/pull/8824#issuecomment-4879423633))

The reviewer grounding reasons over the diff plus the changed-file list — it
cannot see the base tree where the file already lives. The refutation was
machine evidence, posted in-thread
([round 2](https://github.com/synaptent/aragora/pull/8824#issuecomment-4879405120)):

> "disproven at the branch tree with machine evidence: `git ls-tree HEAD --
> docs/artifacts/` → blob 3c0458f1 present; `rg --files docs/artifacts/` now
> lists the file (negation effective); `git check-ignore` → not ignored; the
> file merged to main in #8801."

Honest wrinkle — the reviewers were half right. There *was* a landmine, just
not the one they named
([round 1 comment](https://github.com/synaptent/aragora/pull/8824#issuecomment-4879384313)):

> "both were wrong about the tree … but right about the landmine: the repo's
> 'artifacts/' gitignore rule shadowed the tracked file from every
> ignore-respecting tool, including reviewer grounding."

#8824 itself fixed the `.gitignore` shadowing (a `!docs/artifacts/` negation).
The dead-link claim survived even that; it stopped only when the README link
became an absolute URL to `main` — verifiable without tree access — after which
[openai PASSed at head `7e18207`](https://github.com/synaptent/aragora/pull/8824#issuecomment-4879436485).
The structural fix is filed as
[#8825 — "Reviewer grounding blind spot: evidence reviewers cannot verify
files/links outside the PR diff"](https://github.com/synaptent/aragora/issues/8825).

A closing recursion, for the record: the #8824 negation whitelisted exactly one
filename, and a claude advisory `[P3]` in a later prepare-only round (collector
record at head `56fc92a`) predicted the consequence — "Any sibling tracked
artifact under `docs/artifacts/` stays invisible to the gitignore-respecting
tools this change is meant to satisfy; prefer whitelisting `*.md` … rather than
a single hard-coded filename that must be edited for every future artifact."
Committing *this document* tripped exactly that foot-gun and required adding
its own negation line to `.gitignore`. The advisory finding was right.

### 2. Stale-external-world grounding — the reviewer's fetch of reality is out of date

**Case: [#8834](https://github.com/synaptent/aragora/pull/8834), rounds 1–2.**
A docs-only PR asserted `aragora-verify` 0.1.1 was live on PyPI. openai
fetched PyPI and disagreed. Round 1:

> "PyPI currently exposes only `0.1.0` for `aragora-verify`
> ([PyPI JSON](https://pypi.org/pypi/aragora-verify/json)). This makes the
> install command fail and overstates availability of the key_id binding fix."
> — openai review at head `05487fa` (fixture `pr8834_r1_temporal_and_stale_external`)

Round 2, same premise, escalated framing: "This creates a false security-release
record for the signer-label binding fix." The refutation was a timestamped
fresh fetch plus a clean-venv install, posted in-thread
([refutation comment](https://github.com/synaptent/aragora/pull/8834#issuecomment-4880630705)):

> "stale fetch. Live at 2026-07-04 04:26 UTC:
> https://pypi.org/pypi/aragora-verify/json → latest **0.1.1**, uploaded
> 2026-07-04T03:28Z, and a fresh venv `pip install 'aragora-verify>=0.1.1'`
> resolves and installs 0.1.1"

The release had been uploaded ~54 minutes before openai's first review of this
PR; the reviewer's view (CDN cache or pre-propagation snapshot) was stale.
Re-verified while preparing this document (2026-07-05): the PyPI JSON reports
0.1.1 with wheel upload time `2026-07-04T03:28:00.547565Z`. openai returned
PASS at the same head in the re-gate
([posted 2026-07-04T05:03:39Z](https://github.com/synaptent/aragora/pull/8834#issuecomment-4880751502)).

### 3. Temporal reasoning — the reviewer's clock is not the world's clock

**Case: [#8834](https://github.com/synaptent/aragora/pull/8834), same rounds,
other reviewer.** claude blocked the same PR with a `[P1]` because the release
date looked impossible:

> "the release is dated **2026-07-04, which is in the future** — today is
> 2026-07-03."
> — claude review at head `05487fa` (fixture `pr8834_r1_temporal_and_stale_external`)

And in round 2, with full confidence in the wrong direction:

> "The PR asserts 0.1.1 was **published 2026-07-04 03:28 UTC**, but today is
> **2026-07-03**. UTC 2026-07-04 03:28 is in the future in *every* timezone"
> — claude review at head `8955a4e` (fixture `pr8834_r2_temporal_and_stale_external`)

It was not in the future in any timezone: the review ran at roughly
2026-07-04T04:27Z, an hour *after* the upload. The reviewer's local calendar
(UTC−5) still read July 3, and it projected that local date onto UTC. Two
resolutions were applied, both visible in-thread. First, the legitimate kernel
of the finding was fixed: every release timestamp in the docs became
timezone-explicit ("2026-07-04 (03:28 UTC)"). Second — this is the striking
part — the
[operator settlement packet](https://github.com/synaptent/aragora/pull/8834#issuecomment-4880639145)
explicitly priced in premise self-expiry:

> "(b) wait ~1h until the reviewer clock passes local midnight (the P1's
> premise self-expires) and run one premise-changed re-gate"

The re-gate run minutes after the reviewer environment's local midnight
returned openai PASS (posted 05:03:39Z ≈ 00:03 local), and claude PASSed the
next head at
[05:54Z](https://github.com/synaptent/aragora/pull/8834#issuecomment-4880875980).
A dissent premise that expires on a wall-clock boundary is a failure mode
worth naming: it is refutable by *waiting*.

### 4. Verbatim-repeat dissent — the reviewer re-raises an answered finding

**Case: [#8800](https://github.com/synaptent/aragora/pull/8800), round 3.**
Round 1 on this CI-baseline PR demanded justification for a skip-count baseline
move (68→75). The demanded audit was delivered: a per-skip table in the PR
body tracing all 7 net-new skips to already-merged PRs, every one a
conditional environment guard
([revision comment](https://github.com/synaptent/aragora/pull/8800#issuecomment-4871650194)).
Round 2's distinct finding (doc-history inconsistency) was also fixed. Round
3, openai:

> "[P2] `tests/.skip_baseline:1` raises the enforced skip budget from 68 to 75
> without any corresponding test changes in this PR. … keep the baseline at 68
> unless the PR itself introduces and justifies the 7 new skips."
> — [openai review at head `1817ae0`](https://github.com/synaptent/aragora/pull/8800#issuecomment-4878231496)

That is the round-1 finding restated, with no engagement with the delivered
audit — the [park note](https://github.com/synaptent/aragora/pull/8800#issuecomment-4873062395)
characterizes it as "openai re-raises round 1 verbatim … without engaging the
provided audit," and quotes the round-1 wording it repeats. (Limitation: the
round-1 review body itself was a prepare-only record not posted verbatim, so
"verbatim" rests on the in-thread quote; the round-3 body is on record in full
— fixture `pr8800_r3_verbatim_repeat_answered`.) Meanwhile claude's round-3
PASS did the opposite of repeating itself: it ran the audit tooling at the
head and verified the numbers
("audit returns exactly 75, DIFF=0" —
[claude review](https://github.com/synaptent/aragora/pull/8800#issuecomment-4878231397)).
Resolution:
[operator settlement](https://github.com/synaptent/aragora/pull/8800#issuecomment-4878231586)
under severity-gated dissent — the `[P2]` is advisory and on record; the
counting western-frontier PASS settles Tier 0.

### 5. Out-of-scope carousel — findings rotate onto files the PR cannot touch

**Case: [#8802](https://github.com/synaptent/aragora/pull/8802), seven
adversarial rounds before the clean final.** This compliance-walkthrough PR
was explicitly fenced from `aragora/gauntlet/odr_verify.py` (reserved to the
#8765 lane). Findings alternated between reviewers round after round, and
several targeted exactly the fenced file. Round 4, openai:

> "[P2] … this PR only hardens `aragora-verify`; the in-repo
> `aragora.gauntlet.odr_verify` path used by server/internal verification is
> not updated and can still diverge on multi-signature cases."
> — [openai review at head `21a4ac4`](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871391696)

Round 6 was claude's turn to land on the same fenced file (the
unsigned-with-key parity gap in `odr_verify.py`). True findings — about a file
this PR was forbidden to change. The resolution was re-filing, not suppression:
[#8810 — cross-verifier parity](https://github.com/synaptent/aragora/issues/8810)
now carries them, scoped to the lane that owns the file
([park packet](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871373189),
[settlement](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871391778)).

Honesty requires splitting this class down the middle: the carousel was
**mixed value, not pure noise**. Of the seven rounds, three surfaced real,
in-scope bugs that were fixed and regression-tested:

- **Round 1:** fixture-generator nondeterminism via an ambient calibration DB
  ([fixed](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871197557))
- **Round 3:** a genuine security bug — "aragora-verify 0.1.0 reported VERIFIED
  for a cryptographically valid signature even when its recorded `key_id` had
  been relabeled" ([fixed in 0.1.1 with a regression test](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871349915))
- **Round 5:** a real precedence divergence between the standalone and in-repo
  verifiers ([fixed](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871658875))

The pathology is not that reviewers find things; it is that an adversarial
reviewer with no scope model will eventually wander outside the PR's reachable
surface, and each such finding costs a full round.

### 6. Cross-family contradiction — two reviewers demand opposite designs

**Case: [#8802](https://github.com/synaptent/aragora/pull/8802) again, rounds
5 vs 7.** Round 5, claude *required* a specific signature-verification
precedence, citing the in-repo reference implementation:

> "The in-repo reference verifier `aragora/gauntlet/odr_verify.py:596-641`
> deliberately uses a *separate* `key_id_mismatch` flag checked **after**
> `verified_any`, precisely so that a receipt containing one valid+matching
> signature PASSes even if another entry carries a valid-but-relabeled
> signature. … Fix: mirror the sibling's separate `key_id_mismatch` flag with
> `verified_any`-wins ordering."
> — claude round-5 review at head `05d8ef9` (collector record; requirement
> restated in the
> [round-5 fix comment](https://github.com/synaptent/aragora/pull/8802#issuecomment-4871658875))

The fix was implemented exactly as demanded, with a two-signature regression
test. Then round 7 (fresh head after a main merge), openai objected to
precisely that ordering:

> "[P2] `aragora-verify/src/aragora_verify/verifier.py:231` - `verified_any`
> wins before `key_id_mismatch`, so an attacker can append a relabeled
> duplicate signature … Fail or mark unverified when any evaluated signature
> verifies with the supplied key but has a mismatched `key_id`."
> — openai review at head `787d1f5` (fixture
> `pr8802_r7_cross_family_contradiction`; summarized in the
> [in-thread evidence note](https://github.com/synaptent/aragora/pull/8802#issuecomment-4879461357))

Two frontier families, opposite positions, on one deliberate and documented
design decision. No revision can satisfy both. This is the case that
adversarial-review systems must route to a human — and the in-thread record
already calls it by name: "a textbook ESCALATE specimen for the #8748/#8811
adjudicator, resolved here by reference-implementation parity." The residual
position is pinned on [#8810](https://github.com/synaptent/aragora/issues/8810);
the PR reached a clean 2-0 PASS at the final head
([claude](https://github.com/synaptent/aragora/pull/8802#issuecomment-4879458963),
[openai](https://github.com/synaptent/aragora/pull/8802#issuecomment-4880146715)).

## The control group — the same gate, working

If the six classes above were the whole story, the fix would be "fire the
reviewers." They are not the whole story.

- **[#8811](https://github.com/synaptent/aragora/pull/8811) round 1 — clean
  convergence on a real bug.** Both families independently found the same
  `[P2]`: the observe-only adjudicator hook was wired into the settlement hot
  path with no exception guard, so a crash in an *observational* feature could
  break evidence collection exactly where harmlessness was required
  ([round-1 record](https://github.com/synaptent/aragora/pull/8811#issuecomment-4879898764):
  "both reviewers converge on the same real [P2] … No other findings; no
  carousel — clean convergent dissent"). Fixed; round 3 was a 2-0 PASS
  ([claude](https://github.com/synaptent/aragora/pull/8811#issuecomment-4881139661),
  [openai](https://github.com/synaptent/aragora/pull/8811#issuecomment-4881139703)).

- **[#8852](https://github.com/synaptent/aragora/pull/8852) round 1 — three
  real `[P2]`s, one with a reproduction.** claude reproduced a WAL read-only
  open failure ("Confirmed repro: sidecars present → RO read OK; sidecars
  removed → `mode=ro FAILED: unable to open database file`; read-only
  directory → same failure" — fixture `pr8852_r1_convergent_real_findings`);
  openai found a branch-uniqueness gap and a stale-lease squat. All three
  fixed with new tests
  ([fix comment](https://github.com/synaptent/aragora/pull/8852#issuecomment-4884476574));
  both families then PASSed twice
  ([r2](https://github.com/synaptent/aragora/pull/8852#issuecomment-4884498921),
  [r3](https://github.com/synaptent/aragora/pull/8852#issuecomment-4884676821)).

- **[#8803](https://github.com/synaptent/aragora/pull/8803) — the boring ideal.**
  Clean 2-0 PASS, zero findings, one round
  ([claude](https://github.com/synaptent/aragora/pull/8803#issuecomment-4871390438),
  [openai](https://github.com/synaptent/aragora/pull/8803#issuecomment-4871390511)).
  Good PRs do exit the gate in one pass — and
  [#8802's final round](https://github.com/synaptent/aragora/pull/8802#issuecomment-4880146715)
  shows even a seven-round PR converging to an uncoerced 2-0.

The thesis of this document is **not** "reviewers are bad." It is: reviewer
errors exist, they are classifiable, and *in this record they all point the
same direction* — doubt.

## Error direction: the asymmetry that matters

Tally the documented reviewer errors in this window:

| Error | Direction | True state |
|---|---|---|
| Dead-link claims, #8824 (6 assertions across 4 heads, both families) | false negative | file existed on `main` (blob `3c0458f1`) |
| Stale-PyPI claims, #8834 (3 assertions, openai) | false negative | 0.1.1 live, uploaded 03:28Z |
| Future-date claims, #8834 (2 assertions, claude, one `[P1]`) | false negative | date was valid UTC, already past |
| Verbatim re-raise, #8800 (openai) | false negative | audit already delivered in PR body |
| Out-of-scope findings, #8802 (both families) | false negative for *this* PR | real findings, wrong lane — preserved as #8810 |
| Precedence objection, #8802 r7 (openai) | design dissent | legitimate two-sided crux, not an error of fact |

Every factual error withheld a PASS from a claim that was true. In the same
window, the same reviewers found a real security bug (#8802 r3), a real
crash-in-hot-path (#8811 r1), and a reproduced storage edge case (#8852 r1) —
and no reviewer PASS in this window vouched for a claim later shown false.
False alarms cost rounds and operator attention; false trust would cost
correctness of merged code. This gate spent the cheap currency, not the
expensive one. (See Limitations for why false trust is also structurally
harder to observe — we state the asymmetry as this window's record, not a
theorem.)

## Resolution mechanisms, catalogued

Each failure class came with a working countermeasure, all visible in-thread:

| Mechanism | What it is | Where it worked |
|---|---|---|
| Evidence-post | Machine refutation posted into the review thread (`git ls-tree`, `check-ignore`, timestamped fetches, clean-venv installs) | #8824, #8834 |
| Premise removal | Restate the claim so the reviewer's blind spot no longer applies (absolute URL instead of tree-relative link) | #8824 |
| Premise self-expiry | Wait until a time-indexed dissent premise expires, then re-gate | #8834 |
| Severity gating ([#8574](https://github.com/synaptent/aragora/pull/8574)) | `[P2]`/`[P3]` dissent is advisory: preserved, non-blocking, non-counting | #8800, #8802 |
| Operator adjudication | Human settles a stalled-but-answered record, in-thread, head-pinned | #8800, #8802, #8834 |
| Re-filing | Out-of-scope findings become issues owned by the right lane, never discarded | #8802 → #8810 |
| Grounding fix (filed) | Give reviewers a tree/file-existence oracle, not just the diff | [#8825](https://github.com/synaptent/aragora/issues/8825) |

## What this data enables

**1. Adjudicator evaluation.** The
[review adjudicator](../../aragora/swarm/review_adjudicator.py) (M0,
[#8748](https://github.com/synaptent/aragora/issues/8748)) exists to escape
exactly these stalls by classifying them SETTLE / BLOCK / ESCALATE. The ten
fixture cases in
[`tests/governance/fixtures/adjudicator_eval_cases.json`](../../tests/governance/fixtures/adjudicator_eval_cases.json)
carry verbatim reviewer bodies plus the human-settled ground truth, and
[`tests/governance/test_adjudicator_eval_fixtures.py`](../../tests/governance/test_adjudicator_eval_fixtures.py)
pins the current M0 verdict on each. The current agreement matrix:

| Case | M0 verdict today | Human ground truth | Delta |
|---|---|---|---|
| #8824 r1, #8834 r1/r2 (unrefuted `[P1]`) | BLOCK (hard bar) | block-at-snapshot | agrees — correct fail-safe |
| #8824 r3, #8811 r1, #8852 r1 (unanimous CR) | abstain (not a stall) | block | compatible — base gate blocks |
| #8803 (clean 2-0) | abstain (no dissent) | settle | compatible |
| #8802 r7 (contradiction) | ESCALATE | escalate | **exact agreement** |
| #8800 r3 (verbatim repeat, answered) | ESCALATE | settle | **gap: thread-history blindness** |
| #8802 r4 (out-of-scope) | ESCALATE | settle + re-file | **gap: scope blindness** |

The two gaps are the eval targets: an adjudicator that can see (a) whether a
repeated finding was already answered in the thread and (b) whether a finding
targets the PR's reachable scope would convert both ESCALATEs into the SETTLEs
the human record shows are correct — without ever weakening the hard bar.
Escalating to a human in the meantime is the safe failure mode, and the
fixtures make the improvement measurable instead of anecdotal.

**2. Grounding fixes.** Classes 1–3 are all grounding failures, not judgment
failures: the diff-blind class has a concrete fix specified in
[#8825](https://github.com/synaptent/aragora/issues/8825) (tree listing /
file-existence oracle in the reviewer context, plus a regression fixture:
"a docs-only PR linking a pre-existing tracked file yields no dead-link
finding"). Stale-world and temporal classes suggest the same pattern:
reviewers should be *equipped* to check (fresh fetch, timezone-normalized
clock) rather than trusted to infer.

**3. A benchmark seed.** Ten labeled cases from one week is not a benchmark,
but the schema (verbatim multi-model verdicts + exact head + human disposition
+ resolution mechanism) is designed to accumulate: every future stall the gate
survives is a fixture candidate.

## Limitations — read before citing

1. **Single repo, single week, two counting families.** Six classes observed
   in four days at one codebase (plus Grok in one round) is an existence
   proof and a taxonomy seed, not a frequency estimate. Class frequencies
   here say more about this repo's docs-heavy week than about the models.
2. **Selection effects, in both directions.** These seven PRs were selected
   *because* their threads were eventful or cleanly exemplary. And the
   headline asymmetry has an observation bias: a false reviewer PASS on a
   false claim would surface only if something else caught it later —
   false trust is structurally quieter than false doubt. Within this window
   we also checked the direction the bias hides: post-merge, an independent
   dry-run on #8852 surfaced additional real findings
   ([post-merge note](https://github.com/synaptent/aragora/pull/8852#issuecomment-4884683135))
   — findings a 2-0 PASS had not raised at an earlier head. The gate's PASS
   is "no blocking finding found," not "no bug exists."
3. **Prepared vs posted review bodies.** Some quoted review bodies come from
   prepare-only evidence rounds: recorded by the collector at the exact head
   and quoted in-thread, but not posted verbatim as comments at the time.
   These are committed in full in the fixture file; each fixture marks
   `posted_to_thread` per item. The #8800 round-1 body specifically was not
   recoverable in full, so the "verbatim repeat" characterization rests on
   the in-thread park note's quotation of it.
4. **Second-precision timing is approximate.** The #8834 "local midnight"
   flip is receipted at comment granularity (openai PASS posted
   2026-07-04T05:03:39Z ≈ 00:03 reviewer-local); the reviewer's internal
   completion time is not independently verifiable from the public record.
5. **The taxonomy is descriptive, not exhaustive.** Reviewer transport
   failures, rate limits, and harness hangs — the dominant *availability*
   failure mode per the
   [companion artifact's failure ledger](2026-07-decision-integrity-dogfooding.md#what-the-gate-does-not-catch-read-this-section)
   — are deliberately out of scope here; this document is about reviewers
   that answered and were wrong.
6. **The operator is not a neutral referee.** Settlements quoted here were
   made by the repo operator under recorded policy (severity-gated dissent,
   tiered settlement, head-pinned authorizations). The record is public and
   auditable, but "ground truth" labels in the fixtures encode the
   operator's dispositions — dispute them by filing an issue against the
   fixture, which is exactly the kind of correction this artifact exists to
   invite.

---

*Prepared 2026-07-05 as the second dogfooding-class artifact
([#8856](https://github.com/synaptent/aragora/issues/8856) item 3, tracked as
[#8860](https://github.com/synaptent/aragora/issues/8860)), companion to
[Decision Integrity, Dogfooded](2026-07-decision-integrity-dogfooding.md).
All PR/issue links public. Corrections welcome — file an issue.*
