---
title: Supported API Surface
description: Stability tiers for Aragora's API, SDK, and generated OpenAPI surface.
sidebar_position: 2
---

# Supported API Surface

Last updated: 2026-04-24

Aragora has a large generated API catalog because the platform includes debates,
agents, memory, workflows, integrations, autonomy controls, and operational
surfaces. That catalog is useful for discovery, but it is not a blanket promise
that every route is equally stable.

## Stability Tiers

| Tier | Meaning | Change Policy |
|------|---------|---------------|
| Supported | Intended for external use and source-compatible across patch releases | Breaking changes require migration notes and versioning |
| Beta | Useful and documented, but still hardening | Breaking changes are allowed with release notes |
| Internal | Used by Aragora operators, automations, or generated tools | No external stability promise |
| Experimental | Research, roadmap, or proof-of-concept surface | May change or disappear without migration |

## Supported Today

- Core debate package install path: `pip install aragora-debate`.
- Python SDK package: `aragora-sdk`.
- TypeScript SDK package: `@aragora/sdk`.
- [GitHub PR Review API](/docs/api/github-pr-review).
- Receipt-backed review, decision, and merge-readiness surfaces covered by tests
  and canonical status docs.
- Published OpenAPI reference as a discovery surface, with this document acting
  as the stability boundary.

## Beta / Productizing

- Autonomous execution and worker-contract control-plane APIs.
- Benchmark truth and rescue-productization publication surfaces.
- Decision Integrity Core slices such as executable claims, crux analysis,
  coherence checks, and gardening passes.
- Operator inbox and non-code action loops.

## Promotion Checklist

An API moves into the supported tier only when it has clear documentation,
examples, auth and failure semantics, focused tests, SDK or CLI coverage, and a
receipt, audit event, or provenance hook for consequential actions.

If the tier is unclear, the API should remain beta or internal until the
contract is explicit.
