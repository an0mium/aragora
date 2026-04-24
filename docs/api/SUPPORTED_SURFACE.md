# Supported API Surface

Last updated: 2026-04-24

Aragora has a large generated API catalog because the platform includes debates,
agents, memory, workflows, integrations, autonomy controls, and operational
surfaces. That catalog is useful for discovery, but it is not a blanket promise
that every route is equally stable.

This document defines the contract a cold reviewer should use when judging API
quality and maintainability.

## Stability Tiers

| Tier | Meaning | Change Policy |
|------|---------|---------------|
| Supported | Intended for external use and should remain source-compatible across patch releases | Breaking changes require migration notes and versioning |
| Beta | Useful and documented, but still subject to shape changes | Breaking changes are allowed with release notes |
| Internal | Used by Aragora operators, automations, or generated tools | No external stability promise |
| Experimental | Research, roadmap, or proof-of-concept surface | May change or disappear without migration |

## Supported Today

These surfaces should meet the highest documentation and regression bar:

- Core debate package install path: `pip install aragora-debate`.
- Python SDK package: `aragora-sdk`.
- TypeScript SDK package: `@aragora/sdk`.
- GitHub PR review and settlement APIs documented in
  [GitHub PR Review API](../integrations/GITHUB_PR_REVIEW.md).
- Receipt-backed review, decision, and merge-readiness surfaces that are covered
  by tests and canonical status docs.
- Published OpenAPI reference as a discovery surface, with this document acting
  as the stability boundary.

## Beta / Productizing

These surfaces are valuable but should be described as still hardening:

- Autonomous execution and worker-contract control-plane APIs.
- Benchmark truth and rescue-productization publication surfaces.
- Decision Integrity Core slices such as executable claims, crux analysis,
  coherence checks, and gardening passes.
- Operator inbox and non-code action loops.

## Internal Or Experimental

These should not be sold as stable external contracts without a promotion pass:

- Large generated route families that lack public docs, SDK coverage, or
  explicit stability notes.
- Provider-specific agent plumbing and local runner internals.
- Broad DAG workbench and multi-host expansion routes that are not yet backed by
  repeated green proof surfaces.
- Research-oriented persona, ranking, and nomic-loop internals that are still
  evolving.

## Promotion Checklist

An API moves into the supported tier only when it has:

- A clear user story and owner-facing documentation.
- Request and response examples.
- Auth, permissions, and failure semantics.
- At least one SDK or CLI path, or a documented reason why HTTP is the only
  contract.
- Focused tests that cover success, validation failure, and authorization or
  admission failure where applicable.
- A receipt, audit event, or provenance hook for consequential actions.
- Versioning or migration notes for breaking changes.

## Reviewer Guidance

When inspecting a PR that adds API surface, ask three questions:

1. Is this route part of the supported external contract, a beta surface, or an
   internal helper?
2. Does the documentation make that tier obvious to a new integrator?
3. Does the implementation fail closed when evidence, authorization, or
   validation is missing?

If the answer is unclear, the API should remain beta or internal until the
contract is explicit.
