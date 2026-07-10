# SDK parity #9086: exact Option A classification

**Snapshot:** `origin/main` at `3fe2e5cf561fc094221008d064040ac84625bd4e` on
2026-07-10 UTC. This is an evidence and authorization packet for issue
[#9086](https://github.com/synaptent/aragora/issues/9086), not authorization to
change SDKs, baselines, workflows, branch protection, or the main-red halt.

## Decision requested

Grant one bounded Tier 4 repair that removes these two SDK-only methods from
both language SDKs and removes only their direct tests:

- `/api/modes/{param}`
- `/api/spectate/{param}/stream`

The repair may touch exactly:

- `sdk/python/aragora_sdk/namespaces/modes.py`
- `sdk/python/aragora_sdk/namespaces/spectate.py`
- `sdk/typescript/src/namespaces/modes.ts`
- `sdk/typescript/src/namespaces/spectate.ts`
- `tests/sdk/test_modes_ns.py`
- `tests/sdk/test_spectate_ns.py`
- `tests/sdk/test_sdk_namespaces_new.py`

It may not change baselines, generated SDKs or docs, server routes, CI
workflows, branch protection, or `.aragora/merge_executor.halt`. The repair
must be based on the exact main head above or be revalidated on a newer head.

## Gate snapshot

All commands ran with `TZ=UTC` from the exact snapshot above.

| Gate | Result |
| --- | --- |
| `check_sdk_parity.py --strict --baseline ... --budget ...` | **Fail:** 53 stale Python SDK paths versus the current maximum of 51. There are 281 handlers, 1,877 public routes, and zero routes missing from both SDKs. |
| `check_sdk_namespace_parity.py --strict --baseline ...` | **Pass:** all focused namespaces are current and there are zero regressions. |
| `check_cross_sdk_parity.py --strict --baseline ...` | **Pass:** 2,257 Python paths, 2,373 TypeScript paths, zero Python-only paths, 116 baseline TypeScript-only paths, and zero regressions. |

An in-memory removal of the two proposed paths from both extracted SDK path
sets produced:

- stale Python paths: **53 -> 51**;
- stale TypeScript paths: **79 -> 77**;
- routes missing from both SDKs: **0 -> 0**;
- Python-only cross-SDK paths: **0 -> 0**;
- TypeScript-only cross-SDK paths: **116 -> 116**; and
- new cross-SDK regressions: **0**.

This simulation changed no repository file. It demonstrates the smallest
known source correction that restores the weekly stale-path budget without
transferring debt to another parity gate.

## Classification rule

- **A - remove:** no matching server route exists, the SDK path was introduced
  speculatively, and removal is proven not to create cross-SDK or server
  coverage drift.
- **B - reconcile live route:** a matching server route exists through dynamic
  dispatch or decorator registration, but the parity extractor does not see
  it. Deleting the SDK method would hide a checker/route-declaration gap.
- **C - decide intent:** no matching live route was found, but the family has
  broader product, migration, or duplication ambiguity. Resolve API intent
  before adding a server route or deleting SDK surface.

The conservative result is **A=2, B=16, C=35**.

## Complete 53-path classification

| # | Stale Python SDK path | Class | Evidence or next decision |
| ---: | --- | :---: | --- |
| 1 | `/api/admin/organizations/{param}` | C | Admin organization detail intent is unresolved; current admin/tenant APIs use other route families. |
| 2 | `/api/admin/organizations/{param}/credits` | C | Credit-management intent exists, but no exact registered route was found. |
| 3 | `/api/admin/organizations/{param}/credits/expiring` | C | Same organization-credit migration ambiguity as #2. |
| 4 | `/api/admin/organizations/{param}/credits/transactions` | C | Same organization-credit migration ambiguity as #2. |
| 5 | `/api/admin/users/{param}` | C | Admin user detail intent is unresolved; do not infer deletion from extractor output. |
| 6 | `/api/admin/users/{param}/activate` | B | Live decorator route in `aragora/server/handlers/admin/users.py`; extraction misses it. |
| 7 | `/api/admin/users/{param}/deactivate` | B | Live decorator route in `aragora/server/handlers/admin/users.py`; extraction misses it. |
| 8 | `/api/admin/users/{param}/impersonate` | C | Security-sensitive admin intent requires an explicit keep/remove decision. |
| 9 | `/api/admin/users/{param}/suspend` | C | Admin lifecycle intent overlaps deactivate but is not proven equivalent. |
| 10 | `/api/admin/users/{param}/unlock` | B | Live decorator route in `aragora/server/handlers/admin/users.py`; extraction misses it. |
| 11 | `/api/agents/stats` | C | The live agents handler exposes other stats/ranking surfaces; exact compatibility intent is unclear. |
| 12 | `/api/agents/{param}/calibrate` | C | Current agent routes use singular `/api/agent/*/calibration`; migration intent must be decided. |
| 13 | `/api/agents/{param}/disable` | C | No exact live route found; agent-administration intent remains plausible. |
| 14 | `/api/agents/{param}/elo` | C | Rating data exists under other APIs; compatibility intent is unresolved. |
| 15 | `/api/agents/{param}/enable` | C | No exact live route found; pair with #13 for one lifecycle decision. |
| 16 | `/api/agents/{param}/quota` | C | No exact live route found; quota ownership and API version need a product decision. |
| 17 | `/api/audit/resource/{param}/history` | B | Live dynamic route in `workspace/settings.py` and `workspace_module.py`; extraction misses dispatch registration. |
| 18 | `/api/control-plane/audit` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 19 | `/api/control-plane/audit/stats` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 20 | `/api/control-plane/audit/verify` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 21 | `/api/control-plane/breakers` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 22 | `/api/control-plane/notifications` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 23 | `/api/control-plane/notifications/stats` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 24 | `/api/control-plane/queue/metrics` | B | Live legacy-compatible dispatch in `control_plane/__init__.py`. |
| 25 | `/api/inbox/mentions/{param}/acknowledge` | B | Live dynamic route in `team_inbox.py`; extraction misses it. |
| 26 | `/api/index/{param}` | C | Current knowledge/index handlers use different route shapes; index namespace ownership is duplicated. |
| 27 | `/api/index/{param}/documents` | C | Same index-version and namespace ambiguity as #26. |
| 28 | `/api/index/{param}/documents/{param}` | C | Same index-version and namespace ambiguity as #26. |
| 29 | `/api/index/{param}/optimize` | C | Same index-version and namespace ambiguity as #26. |
| 30 | `/api/index/{param}/rebuild` | C | Same index-version and namespace ambiguity as #26. |
| 31 | `/api/index/{param}/stats` | C | Same index-version and namespace ambiguity as #26. |
| 32 | `/api/media/audio/{param}` | C | Media storage and generated-audio routes exist, but not this exact resource contract. |
| 33 | `/api/media/audio/{param}/convert` | C | Conversion intent is plausible but no exact registered route was found. |
| 34 | `/api/media/audio/{param}/transcription` | C | Transcription intent is plausible but no exact registered route was found. |
| 35 | `/api/modes/{param}` | **A** | Only `/api/modes` is live; this SDK-only detail route was introduced without a server counterpart. |
| 36 | `/api/partners/keys/{param}` | B | Live dynamic key revoke/rotation paths exist in `partner.py`; static `ROUTES` omit the parameterized forms. |
| 37 | `/api/payments/customer/{param}` | B | Live routes are registered in `payments/plans.py` and handled in `payments/billing.py`. |
| 38 | `/api/payments/subscription/{param}` | B | Live routes are registered in `payments/plans.py` and handled in `payments/billing.py`. |
| 39 | `/api/payments/transaction/{param}` | B | Live route is registered in `payments/plans.py` and handled in `payments/stripe.py`. |
| 40 | `/api/pipelines/{param}/items/{param}/transition` | C | Current transition APIs use singular `/api/v1/pipeline`; plural-v2 intent is unresolved. |
| 41 | `/api/pipelines/{param}/items/{param}/transition/rollback` | C | Same pipeline version/migration ambiguity as #40. |
| 42 | `/api/pipelines/{param}/items/{param}/transition/validate` | C | Same pipeline version/migration ambiguity as #40. |
| 43 | `/api/pipelines/{param}/items/{param}/transitions` | C | Same pipeline version/migration ambiguity as #40. |
| 44 | `/api/pipelines/{param}/items/{param}/transitions/available` | C | Same pipeline version/migration ambiguity as #40. |
| 45 | `/api/podcast/episodes/{param}` | C | Episode listing exists; detail-route intent and duplicated podcast namespaces need one decision. |
| 46 | `/api/spectate/{param}/stream` | **A** | Canonical SSE is `/api/v1/spectate/stream?debate_id=...`; the path-parameter SDK route never had a server counterpart. |
| 47 | `/api/tenants/{param}` | C | Current organization APIs use `/api/org/*`; tenant naming/version migration is unresolved. |
| 48 | `/api/tenants/{param}/members` | C | Same tenant-to-organization migration ambiguity as #47. |
| 49 | `/api/tenants/{param}/members/invite` | C | Same tenant-to-organization migration ambiguity as #47. |
| 50 | `/api/tenants/{param}/quotas` | C | Same tenant-to-organization migration ambiguity as #47. |
| 51 | `/api/tenants/{param}/reactivate` | C | Same tenant-to-organization migration ambiguity as #47. |
| 52 | `/api/tenants/{param}/suspend` | C | Same tenant-to-organization migration ambiguity as #47. |
| 53 | `/api/tenants/{param}/usage` | C | Same tenant-to-organization migration ambiguity as #47. |

## Why the two A paths are safe

Commit `af5c2ea1a160b13d7a9cadf30aaf28c3a4e77411` introduced the Python and
TypeScript `modes` and `spectate` namespaces together with direct SDK tests. It
did not introduce matching parameterized server routes.

For modes, the current server declares and dispatches only `GET /api/modes` in
`aragora/server/handlers/nomic.py`. Repository history contains no server-side
`/api/modes/<name>` route. The SDK `get_mode`/`getMode` methods therefore test
only a mock-client URL, not a public server contract.

For spectate, commit `73bfd30cd25f6e68a0a5eb75f557b981120d2087`
established the canonical SSE endpoint as `GET /api/v1/spectate/stream`, with
an optional `debate_id` query parameter. The SDK `connect_sse`/`connectSSE`
methods instead call `/api/spectate/<id>/stream`. History shows no server route
with that shape.

There is consequently no server-route removal commit to cite for either path:
the parameterized routes were SDK-only additions. That is evidence of a client
contract error, not evidence that a public endpoint was later removed.

## Required repair proof

The granted repair is complete only when all of these hold on its exact head:

1. The two methods and only their direct tests are removed from the seven-file
   grant scope.
2. SDK parity strict reports at most 51 stale Python paths and zero routes
   missing from both SDKs.
3. Namespace parity strict remains green with zero regressions.
4. Cross-SDK parity strict remains green with zero new Python-only or
   TypeScript-only paths.
5. Focused Python SDK tests for the affected namespace files pass after their
   obsolete method cases are removed.
6. The PR records Tier 4 exact-head settlement before any merge attempt.

## Follow-up ownership

Class B should become one bounded parity-extractor/route-declaration issue,
with fixtures for decorator, dynamic-dispatch, and framework-registered routes.
Class C should be split by API family and settled as keep-and-implement,
compatibility alias, or remove-from-both-SDKs. Neither class belongs in the
two-path weekly-budget repair.
