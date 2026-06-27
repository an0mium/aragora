# B0 Corpus rev-6 Reconciliation — Decision Memo (founder-owned)

> **Status: DECISION APPLIED (Option B1) — founder merges. Do not auto-merge.**
> The founder selected **retire 4 + restock 4 fresh** on 2026-06-05. This PR now
> carries the concrete rev-6 edit (corpus membership, retired-reasons, republished
> truth/scorecard artifacts, cleared freshness map). It still touches the
> human-owned benchmark *denominator*, so per the standing guardrail — *never
> silently mutate the denominator* — **the merge remains the founder's**; the
> change is staged in this PR for review, not auto-settled.
>
> **Selected restock members (criterion-valid):** #5182, #5183, #5184, #5186 —
> all OPEN B0-cohort `missing_test_coverage` issues with recorded `boss_metrics`
> dispatch history, same execution class as the graduated rev-5 verified cohort.
> (The exception_narrowing candidates first considered had *no* dispatch history
> and so failed the membership criterion — they were not used.)

**Prepared:** 2026-06-05 · Closes the standing restock tracker **#5839** once the
chosen option is applied. Supersedes the stale rev-4 freshness map entry.

---

## 1. Current measured truth (authoritative, dry-run — not yet republished)

Source: `python3 scripts/build_benchmark_truth_artifact.py --corpus docs/benchmarks/corpus.json --dry-run`
against live GitHub state on 2026-06-05.

| Metric | Published surface (2026-06-02) | **Live measured (2026-06-05)** |
| --- | --- | --- |
| Verified `truth_success_rate_verified` | 100% | **100%** |
| Full-corpus `truth_success_rate` | 53.8% (7/13) | **61.5% (8/13)** |
| In-progress graduation rate | 25% (2/8) | **37.5% (3/8)** |
| `corpus_freshness.status` | fresh | fresh (0 stale-closed) |

The 53.8%→61.5% movement is **honest graduation**, not a denominator trick: three
in-progress members closed by their own linked merged PRs since the last publish.
Simply re-running the recurring publication (no corpus edit) will move the public
headline to 61.5% / 37.5%. That republish is denominator-neutral and is the
routine automation's job — not part of this memo's decision.

## 2. Per-member classification of the 8 in-progress entries

| Issue | Live state | Closing PR linkage | Truth state | Honest class |
| --- | --- | --- | --- | --- |
| #5426 | CLOSED | **#7517** (linked, merged) | `merged_pr` ✓ | **Graduation** |
| #5427 | CLOSED | **#7516** (linked, merged) | `merged_pr` ✓ | **Graduation** |
| #5844 | CLOSED | **#7789** (linked, merged) | `merged_pr` ✓ | **Graduation** (this session) |
| #5428 | CLOSED | none (closed as "superseded; landed via #5431") | `no_linked_pr` | **Retire** |
| #5764 | CLOSED | none ("superseded; landed via #5763, dup of #5759") | `no_linked_pr` | **Retire** |
| #5789 | CLOSED | none ("superseded; landed via #5964") | `no_linked_pr` | **Retire** |
| #5790 | CLOSED | none ("superseded; landed via commit b427aad5da") | `no_linked_pr` | **Retire** |
| #5839 | OPEN | — | `in_progress_open` | Keep (this tracker) |

**Why the four `no_linked_pr` are "retire," not "graduation":** each had real
autonomous dispatch history and the underlying code *did* land (the cited PRs
#5431/#5763/#5964 and commit b427aad5da are all merged). But none was resolved by
a **dedicated autonomous graduation of that bounded issue** — each was closed as
*"already satisfied / superseded"* by sibling work done under a different task.
Counting a "satisfied-by-sibling" close as a benchmark success is exactly the
self-grading the guardrail prohibits, and matches the precedent set by the
**rev-3 honesty pass** (which retired five entries "closed by a founder-driven
… PR or manually as stale … none represent autonomous execution"). So they are
retired, not credited.

## 3. The denominator decision (founder-owned)

Three ways to handle the four retired entries. Only the founder should pick,
because it changes what the benchmark *denominator* measures.

| Option | Members | Full-corpus rate | Honest? | Note |
| --- | --- | --- | --- | --- |
| **B0 — keep as failures** | 13 | 8/13 = **61.5%** | Yes | 4 dead closed issues sit as permanent `no_linked_pr` failures; no path to re-attempt; rate can only fall. |
| **B1 — retire 4 + restock 4 (rev-6)** ✅ recommended | 13 | 8/13 = **61.5%** | Yes | Denominator-neutral. Replaces 4 un-re-attemptable failures with 4 fresh *live* bounded members the loop can actually graduate. |
| ~~B2 — retire 4, no restock~~ | 9 | 8/9 = **88.9%** | **No** | Headline inflation by removing failures. **Do not do this.** |

**Recommendation: B1.** It is denominator-neutral (the headline stays 61.5%
today — retiring four 0-numerator failures and adding four 0-numerator fresh
members does not move the rate), it keeps the corpus measuring *live* autonomous
bounded execution, and it restores a path for the benchmark to climb honestly.
B0 is also acceptable and strictly conservative; B2 is prohibited.

## 4. Restock candidate pool (founder selects 4 for B1)

A vetted staging pool exists at `tests/benchmarks/corpus_rev4.json` (33 bounded
B0-cohort entries; 21 not currently promoted). Candidates must be **re-verified
OPEN + bounded + single-PR + no existing resolving PR** at selection time — some
pool entries (e.g. #5176, #5180) have already graduated and are NOT eligible.

Same-execution-class replacements for the four retired (keeps class mix stable):

- For #5789, #5790 (`exception_narrowing`): candidates `#5788`, `#5791`, `#5792`, `#5793`, `#5794`, `#5801` (verify open).
- For #5428 (`small_refactor`) and #5764 (`validation_tightening`): pull two from the `silent_exception_replacement` / bounded-refactor pool (`#5808`–`#5811`, verify open) or the operator backlog (9 open `boss-ready` issues at time of writing).

Founder picks any 4 still-open bounded issues; the exact numbers are not
load-bearing for the denominator, only for *what live work gets measured next*.

## 5. Exact rev-6 corpus edit to apply once Option + members are chosen

1. Move #5428, #5764, #5789, #5790 out of `issues[]` into a new
   `retired_in_revision_6` array, each with the per-issue reason from §2
   ("closed as superseded-by-sibling; landed via &lt;PR&gt;; not a dedicated
   autonomous graduation — retired under the rev-3 honesty precedent").
2. (B1) Add the 4 founder-selected fresh entries to `issues[]` with
   `expected_status: in_progress` and their `execution_class` / `scope_hint`.
3. Bump `revision` to `6`, set `recorded_on`, and append a `revision_log` entry
   mirroring the rev-3/rev-4 wording.
4. Refresh `docs/benchmarks/benchmark_corpus_freshness.json` (currently stale at
   rev-4 referencing #5844/#5903/#5887) to the rev-6 reality, and re-link/close
   the **#5839** tracker.
5. Re-run `scripts/build_benchmark_truth_artifact.py --publish` to republish the
   headline (61.5% / 37.5%, or whatever the fresh members measure).

## 6. What was NOT done here (by design)

- `corpus.json` is **unchanged** — the denominator edit is the founder's.
- No truth artifact was **published** — avoided colliding with the recurring
  publish fleet and kept the headline change founder-gated.
- The three real graduations (#5426/#5427/#5844) need **no** corpus edit; they
  already sit in membership and will show as `merged_pr` on the next publish.
