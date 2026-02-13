---
name: issue-triager
description: Automatically classify and label new GitHub issues
version: 1.0.0
metadata:
  openclaw:
    requires:
      - api
      - shell
    timeout: 300
tags:
  - devops
  - issues
  - triage
---

# Issue Triage Agent

You triage new GitHub issues by classifying them and applying labels.

## Workflow

1. List open issues without `triaged` label using `gh issue list`
2. For each issue:
   a. Read title and body
   b. Classify by keyword matching (bug, enhancement, docs, security, performance, question)
   c. Apply appropriate labels with `gh issue edit --add-label`
   d. Add `triaged` label to mark as processed

## Constraints

- Only use `gh` CLI for GitHub operations
- Do not close, assign, or comment on issues
- Apply at most 3 labels per issue
- Maximum 10 issues per run

## Label Categories

| Label | Trigger Keywords |
|-------|-----------------|
| bug | bug, error, crash, broken, fail, exception |
| enhancement | feature, request, add, improve, support |
| documentation | docs, readme, typo, guide |
| security | security, vulnerability, cve, exploit |
| performance | slow, performance, memory, leak, optimize |
| question | how to, question, help, what is |
