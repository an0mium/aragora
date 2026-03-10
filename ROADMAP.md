# Aragora Roadmap

**Last Updated:** March 10, 2026

## Purpose

This roadmap is the engineering source of truth for making Aragora coherent,
robust, and production-grade without reducing its ambition or scope.

Aragora already has a real core:
- backend runtime centered on `python -m aragora.server`
- debate kernel centered on `aragora/debate`
- frontend runtime in `aragora/live`
- background work through `scripts/queue_worker.py`

The current problem is not lack of scope. The problem is that scope has grown
faster than consolidation. This roadmap prioritizes runtime truth, subsystem
boundaries, deploy correctness, test confidence, and observability.

## Current Assessment

### What Is Real

- `aragora/server/__main__.py` is the clearest canonical backend entrypoint.
- `aragora/server/unified_server.py` remains a real operational center of gravity.
- `aragora/server/fastapi/factory.py` is a substantive parallel API surface.
- `scripts/queue_worker.py` is a real worker entrypoint.
- `aragora/live` is a real Next.js application, not placeholder UI.

### What Is Messy

- Backend runtime is split across unified server and FastAPI migration paths.
- `/api/v1` and `/api/v2` coexist with broad compatibility burden.
- Deploy surfaces drift across Compose, Helm, docs, and packaging.
- Memory, knowledge, storage, and orchestration concepts overlap.
- The test suite is huge, but a large portion is mock-heavy rather than
  end-to-end proof of system integrity.

## System Classification

### Canonical

These should be the backbone of the system:
- `aragora/server/__main__.py`
- `aragora/server/unified_server.py`
- `aragora/server/debate_controller.py`
- `aragora/server/fastapi/factory.py`
- `aragora/debate`
- `aragora/core`
- `scripts/queue_worker.py`
- `aragora/live/src/app`
- `deploy/docker-compose.production.yml`
- `Makefile`

### Messy But Core

These are important, but need consolidation and clearer contracts:
- `aragora/nomic`
- `aragora/memory`
- `aragora/knowledge`
- `aragora/storage`
- `aragora/server/handler_registry`
- `aragora/server/initialization.py`
- `aragora/routing`
- `aragora/queue`
- `aragora/auth`
- `aragora/rbac`

### Expansion Surface

These are likely real product surface area, but should consume platform
contracts rather than define new infrastructure patterns:
- `aragora/connectors`
- `aragora/gateway`
- `aragora/workflow`
- `aragora/gauntlet`
- `aragora/compliance`
- `aragora/billing`
- `aragora/analytics`
- `aragora/audit`
- `aragora/notifications`
- `aragora/live/src/components`

### Compatibility And Drift

These require active cleanup:
- `aragora/server/app.py`
- `aragora/server/fastapi/compat.py`
- `aragora/server/handlers`
- `aragora/server/fastapi/routes`
- `deploy/kubernetes/helm/aragora/values.yaml`
- `pyproject.toml`

## 90-Day Plan

### P0: Days 1-30

Goal: establish runtime truth and stop further architectural drift.

#### Milestone: Canonical Runtime Defined

- Publish ADR for canonical backend runtime
- Publish ADR for canonical worker model
- Document frontend runtime contract
- Publish deploy truth table

#### Milestone: Subsystem And Entrypoint Inventory Complete

- Classify major subsystems as `canonical`, `core-but-messy`, `expansion`,
  `compatibility`, or `defer`
- Inventory backend, CLI, worker, frontend, websocket, and scheduled entrypoints
- Assign an owner to each canonical subsystem

#### Milestone: Deploy Drift Removed

- Align Compose, Helm, service files, and docs to the same real commands
- Normalize worker invocation
- Correct package and CLI truth in `pyproject.toml`

#### Milestone: Expansion Freeze For New Runtime Surfaces

- No new servers, queues, worker models, or compatibility layers without ADR
- No new top-level subsystem without review

### P1: Days 31-60

Goal: make the core hang together.

#### Milestone: Backend Surface Rationalized

- Define the relationship between unified server and FastAPI
- Classify `/api/v1` and `/api/v2` route families
- Reduce duplicate startup logic

#### Milestone: Domain Boundaries Enforced

- Define allowed dependencies between:
  - `server`
  - `debate`
  - `nomic`
  - `memory`
  - `knowledge`
  - `storage`
  - `connectors`
  - `live`
- Add static or CI checks for forbidden cross-layer imports

#### Milestone: State Model Clarified

- Map memory, knowledge, and persistence responsibilities
- Identify overlapping abstractions
- Propose canonical state flows

#### Milestone: Nomic Governance Added

- Identify canonical orchestrators
- Mark legacy and fallback paths
- Document runtime expectations and failure modes

### P2: Days 61-90

Goal: raise confidence and reduce concentration risk.

#### Milestone: Truth-Suite In CI

Add small but real composed tests for:
- backend startup
- debate execution
- worker flow
- persistence flow
- one frontend-backed API path

#### Milestone: Observability Baseline Complete

- Health checks for canonical subsystems
- Startup diagnostics
- Error counters and key flow metrics
- Operator-facing subsystem status signals

#### Milestone: Large-File Decomposition Begins

- Select highest-risk giant files in `server`, `nomic`, and `debate`
- Split only after interfaces are defined

#### Milestone: Frontend-Backend Coherence Pass

- Map major pages to canonical backend contracts
- Flag shell, partial, and deprecated UI surfaces

## Workstreams

### Epic 1: Establish System Truth

- Runtime ADR: declare canonical backend
- Runtime ADR: declare canonical worker model
- Subsystem ledger: classify top-level domains
- Entrypoint inventory: backend, CLI, worker, frontend, scheduled jobs
- Deploy drift fix: normalize runtime commands
- Packaging truth pass
- Admission policy: no new runtime surfaces without ADR

### Epic 2: Consolidate Backend Surface

- Backend classification: unified server vs FastAPI
- API surface matrix: `/api/v1` and `/api/v2`
- Remove duplicate startup logic

### Epic 3: Align Deployment Reality

- Reconcile Compose, Helm, service files, and docs
- Normalize worker command across environments
- Correct package metadata and CLI story
- Add CI checks for deploy/runtime drift

### Epic 4: Define Domain Boundaries

- Define allowed import directions
- Add static boundary enforcement
- Document public subsystem interfaces

### Epic 5: Rationalize State And Memory

- Audit `ContinuumMemory`, `CrossDebateMemory`, `Knowledge Mound`, and storage
- Map write and read paths
- Identify overlap and deprecation candidates

### Epic 6: Harden Nomic

- Inventory `nomic` entrypoints and long-running flows
- Mark canonical orchestrators
- Mark legacy and fallback orchestrators
- Add subsystem spec and smoke path

### Epic 7: Rebalance Testing

- Separate contract, integration, smoke, e2e, and benchmark tiers
- Build a small required truth-suite
- Report mock-heavy test ratios by subsystem

### Epic 8: Observability Baseline

- Require health, logs, and counters for canonical subsystems
- Add missing status and readiness signals
- Add operator-visible core flow diagnostics

### Epic 9: Frontend-Backend Cohesion

- Map major UI routes to backend contracts
- Flag shell and partial routes
- Surface deprecated backend dependencies

### Epic 10: Large-File Decomposition

- Audit top 10-25 largest files
- Split by responsibility after interface decisions

### Epic 11: Admission Control For Future Work

- Add feature metadata template
- Add ADR requirement for new runtime surfaces
- Add experimental and compatibility policy

## Issue Backlog

### P0

- `P0-1` Runtime ADR: Declare Canonical Backend
- `P0-2` Runtime ADR: Declare Canonical Worker Model
- `P0-3` Subsystem Ledger: Classify Top-Level Domains
- `P0-4` Entrypoint Inventory
- `P0-5` Deploy Drift Fix: Normalize Runtime Commands
- `P0-6` Packaging Truth Pass
- `P0-7` Admission Policy: No New Runtime Surfaces Without ADR

### P1

- `P1-8` Backend Classification: Unified Server vs FastAPI
- `P1-9` API Surface Matrix: `/api/v1` and `/api/v2`
- `P1-10` Remove Duplicate Startup Logic
- `P1-11` Domain Boundary Rules
- `P1-12` Static Boundary Enforcement
- `P1-13` Memory/Knowledge Architecture Map
- `P1-14` Nomic Subsystem Spec

### P2

- `P2-15` Truth-Suite: Backend Startup Smoke
- `P2-16` Truth-Suite: Debate Flow
- `P2-17` Truth-Suite: Worker Flow
- `P2-18` Truth-Suite: Persistence Flow
- `P2-19` Observability Baseline For Canonical Subsystems
- `P2-20` Frontend Contract Map
- `P2-21` Large-File Refactor Plan
- `P2-22` Test Taxonomy And CI Tiering

## Success Criteria By Day 90

- One agreed backend runtime story
- One agreed worker model
- One deploy story per environment
- Subsystem ownership and classification in place
- Domain boundary rules enforced
- Truth-suite running in CI
- Top architectural drift points identified and under active reduction

## Non-Goals For The First 90 Days

- No repo rewrite
- No blind scope reduction
- No giant-file cleanup without interface decisions
- No major new runtime surfaces unless they strengthen canonical paths

## Guiding Principle

Aragora should keep its ambition. The standard is not "smaller." The standard is
"broader, but governed." Every important surface should have a clear status,
entrypoint, owner, contract, test tier, and observability story.
