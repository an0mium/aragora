---
title: Aragora Documentation Index
description: Aragora Documentation Index
---

# Aragora Documentation Index

Canonical documentation lives in `docs/` and is mirrored into `docs-site/`.

This index intentionally links to actively maintained docs with validated paths.

## Getting Started

- [Getting Started](../getting-started/overview)
- [SDK Guide (Python)](../guides/sdk)
- [CLI Reference (generated)](../api/cli)

## API

- [API Reference](../api/reference)
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
- [Aragora Conductor Workflow](guides/CONDUCTOR_WORKFLOW.md)
- [Aragora Worker Prompt Pack](guides/WORKER_PROMPT_PACK.md)
- [Dev Swarm Coordination](architecture/DEV_SWARM_COORDINATION.md)

## Architecture and Planning

- [Conductor Control Plane Implementation Spec](plans/2026-03-07-conductor-control-plane.md)

## Security and Compliance

- [Security Overview](../security/overview)
- [Authentication Guide](../security/authentication)
- [SSO Setup](../enterprise/sso)
- [Compliance](../enterprise/compliance)
- [RBAC Matrix](../deployment/RBAC_MATRIX)

## Product Status and Planning

- [Status](./status)
- [Feature Discovery](status/FEATURE_DISCOVERY.md)
- [Feature Gap List](FEATURE_GAP_LIST.md)
- [Next Steps (Canonical)](status/NEXT_STEPS_CANONICAL.md)
- [Active 6-Week Execution Plan](status/EXECUTION_NEXT_6_WEEKS_2026-03-05.md)
- [Documentation Hygiene Register](status/DOCUMENTATION_HYGIENE_AND_GAP_REGISTER.md)
- [Roadmap](../ROADMAP.md)

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
