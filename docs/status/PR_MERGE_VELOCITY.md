# PR Merge Velocity

Last updated: 2026-04-27T03:50Z

H1-07 tracks whether the Mac Studio boss-loop path can run overnight while the
founder checkpoint stays bounded to 15 minutes. This surface is intentionally
operator-facing: it records queue pressure, recent merge throughput, and any
runner blockers that affect overnight autonomy.

## Current Snapshot

| Metric | Value |
| --- | --- |
| Open PRs | 18 |
| Open PRs created today (UTC, 2026-04-27) | 14 |
| Open PRs carried from yesterday (UTC, 2026-04-26) | 4 |
| Open Dependabot PRs | 11 |
| Merged PRs in the last 7 days | >=200 (GitHub query hit the 200-result cap) |
| `boss-loop-test` labeled PRs merged in the last 7 days | 0 |
| `boss-loop-test` keyword fallback | #6669 merged in 7m27s |

## Open PR Classes

| Class | PRs | Operator action |
| --- | --- | --- |
| Overnight H1 Codex lanes | #6719, #6720 | Await CI/review; do not auto-merge while founder is asleep. |
| Dependabot burst | #6708-#6718 | Hold for batch dependency review; outside proof-first safe artifact policy. |
| Vision / roadmap held | #6650, #6655, #6707 | Hold for founder review; do not merge or relabel overnight. |
| Pre-existing implementation held | #6658, #6659 | Hold; #6658 needs OpenAPI regeneration, #6659 is conflicting. |

## Runner / Soak Incident

- Proof-first shift `proof-first-20260426T183511Z` stopped at
  2026-04-27T02:52:49Z after 178 cycles and 23,610s.
- Stop reason: `RuntimeFailure: benchmark publication failed with an
  unclassified runtime error`.
- Triggering workflow: Benchmark Truth Publication run `24974102382`.
- Runner failure: `mac-studio-m3ultra` reported `No space left on device` while
  writing an Actions runner diagnostic log.
- Retry run `24974267989` later ended `cancelled`.
- Post-cleanup verification: Self-Hosted Fleet Probe run `24975379404`
  completed successfully for both Hetzner and Mac Studio probe jobs.

## Morning 15-Minute Checkpoint

1. Confirm no proof-first shift was restarted while unattended.
2. Confirm latest Benchmark Truth Publication after runner cleanup succeeds
   before starting another 12h canonical-green window.
3. Review #6719 and #6720 for CI outcome and scope; they are H1 follow-ups from
   the overnight window.
4. Leave Dependabot PRs batched unless one is explicitly urgent; they are not
   proof-first safe artifact merges.
5. Keep #6650, #6655, #6658, #6659, and #6707 held unless the founder gives a
   new directive.
