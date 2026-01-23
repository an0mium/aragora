---
slug: 016-marketplace-architecture
title: "ADR 016: Agent Template Marketplace Architecture"
description: "ADR 016: Agent Template Marketplace Architecture"
---

# ADR 016: Agent Template Marketplace Architecture

## Status

Accepted

## Context

As Aragora matures, users want to share and reuse successful agent configurations, debate formats, and workflow templates. We need a system that:

1. Allows local storage and management of templates
2. Supports community sharing via a remote marketplace
3. Provides versioning and ratings
4. Integrates seamlessly with existing Arena and Workflow systems

## Decision

We implement a three-layer marketplace architecture:

### Layer 1: Data Models

Three template types using Python dataclasses:

- **AgentTemplate**: System prompts, capabilities, constraints, model configs
- **DebateTemplate**: Task templates, agent roles, protocols, evaluation criteria
- **WorkflowTemplate**: DAG nodes, edges, inputs, outputs

All templates include:
- `TemplateMetadata`: ID, name, description, version, author, category, tags
- `content_hash()`: SHA-256 hash for integrity verification
- `to_dict()`: JSON serialization

### Layer 2: Local Registry

SQLite-based `TemplateRegistry` providing:

- CRUD operations for templates
- Full-text search by query, category, type, tags
- Rating system (1-5 stars with reviews)
- Download/star tracking
- Import/export as JSON
- Built-in templates loaded on initialization

Design choices:
- **SQLite**: Zero configuration, portable, sufficient for local use
- **Lazy loading**: Built-ins loaded on first access
- **Unique ratings**: One rating per user per template (upsert)

### Layer 3: Remote Client

Async `MarketplaceClient` for community sharing:

- aiohttp-based HTTP client
- Token authentication
- Full CRUD for templates
- Rating and starring
- Featured/popular/recent queries
- User template management

## Consequences

### Positive

- **Composability**: Templates work with existing Arena/Workflow systems
- **Offline-first**: Local registry works without network
- **Extensibility**: Easy to add new template types
- **Community**: Enables template sharing ecosystem

### Negative

- **Sync complexity**: Local/remote sync not automatic
- **Versioning**: No automatic dependency resolution
- **Trust**: Remote templates need validation

### Mitigations

- Content hashing for integrity verification
- Rating system for quality signals
- Built-in templates as trusted baseline
- Clear separation of local vs remote

## Alternatives Considered

1. **Git-based sharing**: Too complex for non-developers
2. **Central database only**: No offline support
3. **File-based registry**: Harder to search and manage
4. **No marketplace**: Limits ecosystem growth

## Implementation

```
aragora/marketplace/
├── __init__.py      # Public API
├── models.py        # AgentTemplate, DebateTemplate, WorkflowTemplate
├── registry.py      # TemplateRegistry (SQLite)
└── client.py        # MarketplaceClient (async HTTP)
```

Tests: 38 tests covering models, registry operations, and built-in templates.
