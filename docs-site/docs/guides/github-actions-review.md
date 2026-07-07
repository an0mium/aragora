---
title: GitHub Action Setup
description: Add multi-model CI review and receipts to pull requests.
---

# GitHub Action Setup

Add Aragora multi-model review to pull requests and, when needed, emit an
independently verifiable Decision Receipt.

## Use the Root Action

External repositories should use the root `synaptent/aragora` action. It works
from any repository and supports `emit-receipt`.

```yaml
name: Aragora AI Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

concurrency:
  group: aragora-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    name: AI Code Review
    runs-on: ubuntu-latest
    if: github.event.pull_request.draft == false && github.actor != 'dependabot[bot]'
    steps:
      - name: Run Aragora Review
        id: review
        uses: synaptent/aragora@8b600a3a8dbf076f4027ae27f3dcbbf48e75409f
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          post-comment: 'true'
          fail-on-critical: 'false'
```

Add at least one provider key under **Settings > Secrets and variables >
Actions**. For best results, set both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

## Emit a Receipt

Set `emit-receipt: 'true'` when the review should produce a portable Open
Decision Receipt artifact:

```yaml
with:
  anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
  openai-api-key: ${{ secrets.OPENAI_API_KEY }}
  post-comment: 'true'
  emit-receipt: 'true'
  receipt-reviewers: 'claude openai'
```

The action validates the receipt schema and canonical digest before exposing
receipt outputs. Unsigned receipts still verify for schema and digest, but the
offline verifier reports that the signature set is empty.

## Root vs. Nested Actions

This page is for the root `synaptent/aragora` action. The composite actions
inside this repository, `.github/actions/aragora-code-review` and
`.github/actions/aragora-review`, are same-repository snippets. GitHub only
resolves `uses: ./path` inside the repository that owns the workflow, so another
repository must vendor those directories before using them. Those nested actions
also do not have an `emit-receipt` input.

## Next Steps

| Guide | What it covers |
|-------|----------------|
| [Open Decision Receipt](../specs/open-decision-receipt) | Portable receipt format |
| [Receipt Lineage Reconciliation](../specs/receipt-lineage-reconciliation) | Native record vs. portable ODR |
| [Independent Verifier Guide](../specs/independent-verifier-guide) | Offline receipt verification |
