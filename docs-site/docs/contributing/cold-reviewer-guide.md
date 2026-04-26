---
title: Cold Reviewer Guide
description: Fast path for reviewing Aragora's thesis, proof surfaces, and shipped boundary.
sidebar_position: 2
---

# Cold Reviewer Guide

Last updated: 2026-04-24

Aragora is an auditable execution control plane for consequential AI-assisted
work. It coordinates heterogeneous agents, requires evidence-backed review,
preserves receipts and provenance, and stops truthfully when a task lacks enough
support to proceed.

## What It Is Good For Today

- PR review and merge settlement where multiple agents inspect the same diff,
  dissent is preserved, and merge readiness is tied to evidence.
- Autonomous or semi-autonomous software execution where work orders, validation
  plans, receipts, and stop states matter more than raw coding speed.
- Decision records that need provenance, claims, disagreement, and later
  reassessment instead of ephemeral chat transcripts.
- Calibration loops that compare agent recommendations against later outcomes.
- Fail-closed automation where admin-scoped approvals and truth surfaces are
  required before high-impact actions advance.

## What Is Still Aspirational

The long-term roadmap is larger than the current wedge: an organization substrate
for the full `idea -> goal -> plan -> action -> receipt -> later settlement`
loop. Reviewers should treat future-facing documents as roadmap material unless
the capability has tests, receipts, and proof surfaces.

## Inspect These First

- [Supported API Surface](/docs/api/supported-surface)
- [Canonical Goals](/docs/contributing/canonical-goals)
- [Evolution Roadmap](/docs/contributing/aragora-evolution-roadmap)
- [Next Steps Canonical](/docs/contributing/next-steps-canonical)
- [Benchmark Truth Status](/docs/contributing/b0-benchmark-truth-status)
- [Rescue Productization Status](/docs/contributing/tw03-rescue-productization-status)
- [GitHub PR Review API](/docs/api/github-pr-review)

## Fast Verification

Run these from the repository root:

```bash
python scripts/inspect_cold_review_surface.py
python scripts/check_version_alignment.py
npm run build --prefix docs-site
```

The cold-review inspector checks public framing, docs-site announcement drift,
reviewer/auditor entry points, supported-surface documentation, live queue
alignment, and OpenAPI scale summary.

## Bottom Line

Aragora is valuable if it keeps tightening the gap between AI-generated work and
auditable organizational trust. The next phase should bias toward fewer,
better-proofed surfaces: PR settlement, receipts, calibration, review gates,
proof-carrying code, and cold-reviewable documentation.
