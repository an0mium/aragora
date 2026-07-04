---
title: Aragora Documentation Index
description: Aragora Documentation Index
---

# Aragora Documentation Index

Canonical documentation lives in `docs/` and is mirrored into `docs-site/`.

This index intentionally links to actively maintained docs with validated paths.

## Getting Started

- [Cold Reviewer Guide](COLD_REVIEWER_GUIDE.md)
- [Getting Started](../getting-started/overview)
- [SDK Guide (Python)](../guides/sdk)
- [CLI Reference (generated)](../api/cli)

## Receipts & Verification

`docs/specs/` is not mirrored into `docs-site/` (see "Notes" below), so these
are intentionally external GitHub links rather than in-site links that would
404 on `docs.aragora.ai`. For release-matched audits, replace `main` in the
URL with the audited tag or commit.

- [Open Decision Receipt Spec](https://github.com/synaptent/aragora/blob/main/docs/specs/OPEN_DECISION_RECEIPT.md)
- [Receipt Lineage Reconciliation](https://github.com/synaptent/aragora/blob/main/docs/specs/RECEIPT_LINEAGE_RECONCILIATION.md)
- [Independent Verifier Guide](https://github.com/synaptent/aragora/blob/main/docs/specs/INDEPENDENT_VERIFIER_GUIDE.md)

## API

- [API Reference](../api/reference)
- [Supported API Surface](api/SUPPORTED_SURFACE.md)
- [API Endpoint Catalog](../api/endpoints)
- [API Examples](../api/examples)
- [API Versioning](../api/versioning)
- [Webhooks](../api/webhooks)

## Core Concepts

- [Architecture](../core-concepts/architecture)
- [Debate Internals](../core-concepts/debate-internals)
- [Agent System](../core-concepts/agents)
- [Knowledge Mound](../core-concepts/knowledge-mound)
- [Workflow Engine](../core-concepts/workflow-engine)

## Operations

- [Production Deployment](../deployment/production-deployment)
- [Deployment Guide](../deployment/overview)
- [Security Deployment](../deployment/security)
- [Runbook](../operations/runbook)
- [Incident Response](../operations/incident-response)
- [Aragora Conductor Workflow](../guides/conductor-workflow)
- [Aragora Worker Prompt Pack](../guides/worker-prompt-pack)
- [Dev Swarm Coordination](./dev-swarm-coordination)

## Architecture and Planning

- [Conductor Control Plane Implementation Spec](./conductor-control-plane-implementation-spec)

## Security and Compliance

- [Security Overview](../security/overview)
- [Authentication Guide](../security/authentication)
- [SSO Setup](../enterprise/sso)
- [Compliance](../enterprise/compliance)
- [RBAC Matrix](../deployment/RBAC_MATRIX)

## Product Status and Planning

- [Status](./status)
- [Feature Discovery](./feature-discovery)
- [Feature Gap List](./feature-gap-list)
- [Next Steps (Canonical)](./next-steps-canonical)
- [Active 6-Week Execution Plan](./execution-next-6-weeks-2026-03-05)
- [Documentation Hygiene Register](./documentation-hygiene-and-gap-register)
- [Roadmap](./roadmap)

## Reference

- [Environment Variables](../getting-started/environment)
- [Library Usage](../guides/library-usage)

## Contributing

- [Contributing Guide](./guide)
- [Reference Index](./documentation-index)
- [Deprecation Policy](./deprecation)

## Notes

- Deprecated and historical docs are in `docs/deprecated/`.
- For link-health checks, run `python scripts/validate_doc_links.py`.
- `docs/specs/` is not mirrored into `docs-site/` by `docs-site/scripts/sync-docs.js`;
  link to it with absolute `github.com/synaptent/aragora/blob/main/...` URLs, not
  relative paths, so the docs-site mirror doesn't 404.
