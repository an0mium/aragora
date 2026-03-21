# PR Review Remote-Head Discipline

Use this rule for every manual or agentic PR review:

**Review the remote PR head, not the ambient local worktree.**

## Why

We hit repeated false findings because review agents inspected whatever branch
their local worktree happened to be on rather than the actual PR branch under
review. That produces stale-snapshot reviews, blocks valid PRs, and erodes
operator trust.

## Required Commands

Fetch the PR metadata and branch first:

```bash
gh pr view <pr-number> --repo synaptent/aragora \
  --json headRefName,headRefOid,baseRefName
git fetch origin <headRefName> --quiet
```

Use remote diff or remote file reads as the source of truth:

```bash
gh pr diff <pr-number> --repo synaptent/aragora
git diff origin/<baseRefName>...origin/<headRefName>
git show origin/<headRefName>:<path>
```

## Review Rules

- A review finding must be reproducible from the remote PR head.
- If the local worktree disagrees with the remote branch, the remote branch
  wins.
- If the remote branch cannot be fetched or inspected safely, mark the review
  `blocked_nonreviewable` instead of relying on local files.

## Minimal Checklist

1. Confirm `headRefOid`.
2. Inspect the PR diff from GitHub or `git diff origin/<base>...origin/<head>`.
3. Inspect any cited file with `git show origin/<head>:<path>`.
4. Only then write findings or approval.
