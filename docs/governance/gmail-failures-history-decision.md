# Decision: gmail_failures.json git-history handling

- Status: Open (operator decision required)
- Date filed: 2026-06-13
- Owner: repository operator / founder
- Related: HEALTH-1 (#8258), Structural Excellence epic (#8257), assertions VAL-P1-002 and VAL-P1-010

## Context

`gmail_failures.json` is an inbox-triage failure dump that was tracked at the
repository root. It contains at least one real personal email address, so it has
been untracked (`git rm --cached`, file kept on disk) and added to `.gitignore`
as part of P1 repo-root hygiene.

Untracking removes the file from the working tree of future clones, but the
historical blob remains reachable in git history (for example commit
`0d30808d81`). The Structural Excellence mission never rewrites git history under
any circumstances, so purging the blob from history is explicitly out of scope
for the automated mission and is recorded here as an operator decision item.

## Options

1. Accept (do nothing further). Leave the historical blob in place. The working
   copy is untracked and ignored; the only residue is the email address inside
   old history. Lowest effort, and it performs no history rewrite.
2. Operator-run `git filter-repo` window. The operator schedules a coordinated
   maintenance window, freezes pushes, runs `git filter-repo` (or BFG) to excise
   the blob, force-updates `main`, and has every collaborator re-clone. This
   rewrites history and must be performed by a human, never by the mission.
3. Contact the data owner. Reach out to the person whose email appears in the
   dump, confirm whether removal from history is required (for example for a
   privacy or GDPR reason), and only then choose option 1 or option 2.

## Recommendation

Default to option 1 (accept) unless the data owner or a privacy/compliance
requirement forces a history rewrite, in which case the operator performs
option 2 in a controlled window. This document records the decision item; the
chosen resolution is tracked in the mission operator-approvals ledger.
