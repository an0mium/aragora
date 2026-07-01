# P4a EventBus / Job-Queue Registry Inversion (events + queue relocate-UP)

> **Status:** Design (Tier 1-2 doc). **Milestone:** `p4a-layering-foundation`.
> **Supersedes:** the cancelled relocate batches `p4a-events-subscribers-relocate` (2a)
> and `p4a-queue-workers-relocate` (2b).
> **Scope:** resolves the layering inversion for BOTH `aragora.events`
> (`cross_subscribers` + `subscribers` + `security_events`/`security_dispatcher`/`arena_bridge`)
> AND the sibling `aragora.queue` (`worker` + `workers.*`), with grimp evidence.
> **Does NOT** implement the moves; it specifies homes, the domain-free registry,
> the bootstrap, the relocate-UP no-shim exemption, the batch breakdown, and the
> exact AGENTS.md / shims.md policy text.

## 0. TL;DR

`aragora.events` (infrastructure) and `aragora.queue` (infrastructure) each import
UP into `domain`/`application`/`interface` because domain-coupled *handlers*
(`cross_subscribers`/`subscribers`) and *workers* (`queue.workers.*`) live inside
the infra package. The sanctioned fix (per
`docs/architecture/P4A_LAYERING_DISPOSITION.md` lines 185-190, 337) is the
**EventBus / job-handler registry inversion**:

1. The events/queue packages keep only a **domain-free registry + dispatch surface**
   (`register_subscriber`/`register_factory`/`get`/`reset`/`SubscriberStats`,
   and the queue transport). ZERO domain imports (eager, lazy, or `TYPE_CHECKING`).
2. Each domain-coupled handler/worker **moves to a home at or above the highest
   layer it imports** (prefer the coupled domain package; use an existing
   application/interface home for multi-domain/server couplings). It
   **self-registers on import**.
3. A single explicit **bootstrap** at each composition root imports the relevant
   subscriber/worker modules to guarantee registration.
4. The moved paths get **NO re-export shim** at their old `events`/`queue`
   locations (a shim re-creates the layering edge - the proven failure mode of
   worker `5f92db63`). ALL consumers (runtime + the ~7 monkeypatching test files)
   are **repointed** to the new homes. This is a *sanctioned relocate-UP
   exemption* from the usual "shim every moved path" rule (§8).

Grimp proves (§7) that after the inversion,
`check_import_contracts.py --layers foundation,infrastructure` clears
`events -> {agents,debate,knowledge,memory,nomic,ranking,reasoning,workflow}` and
`queue -> {agents,debate,gauntlet,integrations,memory,nomic,ranking,server}` plus
the subscriber-side `events -> server`, **with zero new un-baselined edge**. The
residual `events -> server` is owned by Batch 1b-sweep + Batch 2c (core
dispatcher), not this batch.

## 1. Why relocate-UP + a re-export shim is impossible (the 5f92db63 result)

Worker `5f92db63` (handoff
`handoffs/2026-07-01T02-52-44-111Z__p4a-events-subscribers-relocate__5f92db63*.json`)
empirically proved that **relocate-UP plus a mandatory re-export shim CANNOT clear
`events -> domain`**:

- The usual VAL-CROSS-004 rule ("shim every moved path") requires an
  identity-preserving re-export at the old path so external callers and the ~7
  test files that `@patch(...)` the *old* submodule paths keep working
  (e.g. `@patch("aragora.events.cross_subscribers.handlers.validation.record_km_inbound_event")`).
- An **eager** identity-preserving shim at `aragora.events.<x>` re-imports the
  relocated module, so `aragora.events.<x> -> <home> -> aragora.<domain>` is
  re-created. grimp counts indirect **and** lazy chains through the shim, so
  `resolved_violations == {}` after relocation + eager shim.
- No relocation target avoids a violation *while a shim exists*: a domain/unlayered
  home keeps `events -> domain`; an `application` home creates a NEW un-baselined
  `events -> application` (fail-on-new); a `foundation` home makes
  `foundation -> domain`.

**Conclusion:** the only way to clear the edge is to *remove the events/queue
package's import of the domain module entirely* - i.e. the domain-coupled code must
leave `events`/`queue` **and leave behind no re-export**. That forces (a) a
domain-free registry surface that stays, and (b) a no-shim relocate-UP with all
consumers repointed. Both are specified below.

## 2. How the checker attributes edges (empirically proven: direct first-hop)

`scripts/ci/check_import_contracts.py` records each violation as
`f"{importer} -> {imported}"` at **layer-package granularity**
(`aragora.events -> aragora.debate`) straight from import-linter's
`LayersContract` metadata; import-linter in turn calls grimp's
`find_illegal_dependencies_for_layers`. We replicated that exact primitive with
grimp 3.14 against the local tree (`/tmp/sx_grimp_evidence.py`). Decisive finding:

> **Every `events -> domain` / `queue -> domain` route is `[head] -> [tail]` with
> an EMPTY middle.** grimp attributes an illegal layer edge to the **direct
> first-hop importer**. A transitive `events.dispatcher -> server.* -> ... ->
> debate.*` chain is **NOT** collapsed into `events -> debate`; it is attributed
> as `events -> server` (the first illegal hop) plus a legal downward
> `server -> debate`.

Sample (verbatim grimp output, current tree at `fce9aa5e09`):

```
aragora.events -> aragora.debate   (9 route(s))
      route: [aragora.events.cross_subscribers.handlers.basic] -> [aragora.debate.selection_feedback]
      route: [aragora.events.security_dispatcher] -> [aragora.debate.orchestrator]
aragora.events -> aragora.server   (10 route(s))
      route: [aragora.events.dispatcher] -> [aragora.server.middleware.tracing]
```

**Implication (the batch's core enabler):** cutting the *direct* subscriber/handler
imports clears `events -> domain` even though the dispatcher-side `events -> server`
persists (that is Batch 1b-sweep + Batch 2c, §7). Likewise for `queue`.

Two attribution nuances captured for accuracy:
- **Unlayered pass-through.** `subscribers.notification_handlers` shows up under
  `events -> server` because it imports the *unlayered* `aragora.notifications`
  package, through which grimp routes to `server`. Moving `notification_handlers`
  to a delivery home removes it from the `events -> server` head set.
- **`registry` head of `events -> server`** is a transitive artifact through
  `security_events` (which stays but is de-coupled in §5.1); `events.registry`
  has **no direct** `server` import. It is immaterial - `events -> server` stays
  regardless because `dispatcher`/`async_dispatcher` import `server.middleware.tracing`
  directly, and that edge is baselined + owned by Batch 1b-sweep/2c.

## 3. Current coupling map (per-edge direct contributors)

From grimp (`find_illegal_dependencies_for_layers`, layers = the 5 `.importlinter`
layers, container `aragora`). Heads are relative to `aragora.events.` / `aragora.queue.`.

### 3.1 `aragora.events` (baseline: 9 edges)

| Edge | Direct contributor modules (heads) |
|---|---|
| events -> agents | `security_events` |
| events -> debate | `arena_bridge`, `cross_subscribers.handlers.basic`, `cross_subscribers.handlers.strategic`, `security_dispatcher`, `security_events`, `subscribers.debate_handlers` |
| events -> knowledge | `cross_subscribers.handlers.{basic,culture,knowledge_mound,strategic,validation}`, `subscribers.{debate_handlers,execution_handlers,mound_handlers,testfixer_handlers}` |
| events -> memory | `cross_subscribers.handlers.{basic,knowledge_mound,validation}`, `subscribers.mound_handlers` |
| events -> nomic | `subscribers.testfixer_handlers` |
| events -> ranking | `subscribers.execution_handlers` |
| events -> reasoning | `cross_subscribers.handlers.basic` |
| events -> server | subscriber-side (THIS batch): `cross_subscribers.handlers.{basic,culture}`, `subscribers.{execution_handlers,notification_handlers}`; core-side (Batch 1b-sweep/2c): `dispatcher`, `async_dispatcher`, `registry` |
| events -> workflow | `cross_subscribers.handlers.strategic`, `subscribers.workflow_automation` |

### 3.2 `aragora.queue` (baseline: 8 edges)

| Edge | Direct contributor modules (heads) |
|---|---|
| queue -> agents | `worker` (`agents.base`), `workers.gauntlet_worker` (`agents.base`) |
| queue -> debate | `worker` (`debate.orchestrator.Arena`) |
| queue -> gauntlet | `workers.gauntlet_worker` (`gauntlet`, `gauntlet.storage`) |
| queue -> integrations | `workers.routing_worker` (`integrations.email_reply_loop`) |
| queue -> memory | `workers.consensus_healing_worker` (`memory.consensus`) |
| queue -> nomic | `workers.testfixer_worker` (`nomic.testfixer.http_api`) |
| queue -> ranking | `workers.gauntlet_worker` (`ranking.elo.EloSystem`) |
| queue -> server | `workers.gauntlet_worker` (`server.stream.gauntlet_emitter`), `workers.routing_worker` (`server.debate_origin`) |

> The Batch-1 `queue -> server` contributors (`queue/tracing.py`,
> `queue/webhook_worker.py`) already cleared in Batch 1b/1c, so **`queue -> server`
> is owned entirely by this batch** (both remaining heads are workers).

## 4. The inversion design

### 4.1 Domain-free registry surface (STAYS in events/queue; no move, no shim)

`aragora.events.cross_subscribers` keeps a **domain-free** registry + dispatch core:

```python
# aragora/events/cross_subscribers/__init__.py + manager.py  (ZERO domain imports)
def register_subscriber(name: str, subscriber: object) -> None: ...
def register_factory(name: str, factory: Callable[[], object]) -> None: ...
def get_cross_subscriber_manager() -> "CrossSubscriberManager": ...   # public accessor - UNCHANGED
def reset_cross_subscriber_manager() -> None: ...

@dataclass
class SubscriberStats:            # domain-free telemetry record
    events_processed: int = 0
    errors: int = 0
    ...

class CrossSubscriberManager:     # dispatch-only; iterates registered subscribers
    def dispatch(self, event: StreamEvent) -> None: ...
```

- Today `manager.py` (570 LOC) composes domain-coupled **mixins**
  (`BasicHandlersMixin`, `CultureHandlersMixin`, ...). The inversion converts the
  manager into a **registry-backed dispatcher**: it holds a list/dict of registered
  subscriber objects and fans events out to them. The mixin imports are removed as
  each handler group relocates (§4.4).
- `get_cross_subscriber_manager()` **stays at
  `aragora.events.cross_subscribers`** and keeps returning a working manager. This
  is the IDEAL outcome required by the spec: the ~13 lazy consumers
  (debate/memory/server/cli) that call `get_cross_subscriber_manager()` do **not**
  move or break - only the manager's *internals* change (domain-free), not its
  public accessor or import path.
- `aragora.queue` keeps the transport (`base`, `config`, `job`, `retry`, the
  Redis-Streams `Worker` base) plus a **domain-free job-handler registry**
  (`register_worker`/`register_job_handler` + `get`/`reset`). The
  domain-coupled `DebateExecutor`/`create_default_executor` re-exports are
  **dropped** from `queue/__init__.py` (no re-export shim - a re-export would
  create `queue -> application`, §5.2).

### 4.2 Home-assignment rule (the governing invariant)

> **A relocated handler/worker must live in a package whose layer is at or above
> the highest layer among all packages it imports.** ("home layer >= max import
> layer.")

Because cross-imports *within* a layer are legal (siblings are colon-separated =
non-independent in `.importlinter`), a handler that touches only `domain` packages
can live in ANY domain package and import its sibling domains freely. A handler
that reaches `application` (workflow/nomic) must live in `application`; one that
reaches `interface` (server/notifications-delivery) must live in `interface`.

This rule is what guarantees **no new upward edge** at the destination (§7 proves
the source edges clear; this rule proves the destination introduces none).

### 4.3 Home assignments (prefer the coupled domain; existing app/interface homes)

Handlers are currently multi-reaction classes; the inversion **splits each class by
reaction** and files each reaction under the home matching its target layer. No new
top-level package is introduced (no `aragora/orchestration`); every home already
exists, so **no `module_tiers.yaml` drift** results (§9).

| New home (self-registering module) | Layer | Reactions relocated here (source) |
|---|---|---|
| `aragora/knowledge/event_subscribers.py` | domain | KM ingest/validation/mound reactions from `handlers.{knowledge_mound,validation,basic,culture,strategic}`, `subscribers.{mound_handlers,debate_handlers,execution_handlers,testfixer_handlers}` (the knowledge-writing parts) |
| `aragora/memory/event_subscribers.py` | domain | memory/continuum reactions from `handlers.{basic,knowledge_mound,validation}`, `subscribers.mound_handlers` |
| `aragora/debate/event_subscribers.py` | domain | debate reactions from `subscribers.debate_handlers`, `handlers.{basic,strategic}`; `arena_bridge` relocation (§5.1) |
| `aragora/reasoning/event_subscribers.py` | domain | belief reaction from `handlers.basic` (`reasoning.belief`) |
| `aragora/ranking/event_subscribers.py` | domain | elo reaction from `subscribers.execution_handlers` (`ranking.elo`) |
| `aragora/workflow/event_subscribers.py` | application | `subscribers.workflow_automation` (PostDebateWorkflowSubscriber) + `handlers.strategic` workflow reaction (`workflow.engine.get_workflow_engine`) + `handlers.basic._handle_debate_end_to_workflow` |
| `aragora/nomic/...` (existing testfixer surface) | application | `subscribers.testfixer_handlers` (`nomic.testfixer.http_api`, `nomic.improvement_queue`) |
| `aragora/server/event_subscribers.py` | interface | server-coupled reactions: `handlers.basic` webhook delivery (`server.handlers.webhooks.get_webhook_store`), `handlers.culture` (`server.stream.state_manager`), `subscribers.execution_handlers` emitter (`server.stream.emitter`), `subscribers.notification_handlers` (delivery via `aragora.notifications`) |

Rationale: this honors "subscribers belong with their domains" for the pure-domain
reactions, and uses **existing** application (`workflow`, `nomic`) and interface
(`server`) homes for the multi-domain/server-coupled reactions - never inventing a
new unlayered package.

### 4.4 Self-registration + explicit bootstrap (layered composition roots)

Each home module registers its subscriber(s) at import time:

```python
# aragora/knowledge/event_subscribers.py  (domain; imports sibling domains only)
from aragora.events.cross_subscribers import register_subscriber   # infra, downward = legal
class KnowledgeEventSubscriber: ...
register_subscriber("knowledge", KnowledgeEventSubscriber())
```

Because registration only happens when the home module is imported, a **single
explicit bootstrap** at each composition root imports the relevant modules:

- **Interface composition roots** (server startup, CLI) call the *superset*
  `bootstrap_event_subscribers()` which imports the domain **and** application
  **and** interface home modules. Interface may import every layer, so this is legal
  and guarantees full registration for the running product.
- **Domain/library composition root** (Arena / `aragora.debate`) calls a
  *domain-subset* bootstrap that imports only the domain home modules
  (`knowledge`/`memory`/`debate`/`reasoning`/`ranking` `event_subscribers`) -
  domain->domain, legal. Application/interface reactions (workflow/nomic/server)
  register when *those* subsystems initialize (they are no-ops in a pure library
  debate that has no server/workflow engine, matching today's try/except fallbacks).
- **Direct-constructor tests** either call the appropriate bootstrap or import the
  specific home module before asserting registration.

> **Bootstrap location constraint:** the superset bootstrap imports application +
> interface modules, so it must itself live in `interface` (e.g.
> `aragora/server/startup/` or CLI). It must NOT live in `debate`/`events` (that
> would recreate an upward import). The domain-subset bootstrap lives in `debate`
> (domain) and imports only sibling-domain home modules.

## 5. SPLIT modules and cross-handler coupling inversion

Three events modules and one queue module are **SPLIT** rather than fully moved:
they have a domain-free public surface with **many external consumers**, so the
surface STAYS in `events`/`queue` and only the domain-coupled functions relocate.

### 5.1 events SPLITs

- **`security_events.py` (864 LOC).** KEEP (domain-free, stays in events): the
  enums/dataclasses/emitter/factories that external callers import -
  `SecurityEventType`, `SecuritySeverity`, `SecurityFinding`, `SecurityEvent`,
  `SecurityEventEmitter`, `create_vulnerability_event`, `create_secret_event`,
  `create_scan_completed_event`, `get_security_emitter`, `set_security_emitter`.
  MOVE (domain-coupled): `build_security_debate_question` (imports
  `debate.protocol`, `debate.orchestrator`) and the agent-runner path (imports
  `agents.factory`, `agents.api_agents.{anthropic,openai}`) -> a debate/security
  home (e.g. `aragora/debate/security_response.py`, domain). Clears
  `events -> agents` and the `security_events` contributor to `events -> debate`.
- **`security_dispatcher.py` (499 LOC).** Only ONE domain import
  (`debate.orchestrator.Arena`, lazy at line ~363). KEEP the domain-free dispatch
  routing surface in events; MOVE the single Arena-running function to the debate
  home. Clears the `security_dispatcher` contributor to `events -> debate`.
- **`arena_bridge.py` (184 LOC).** Its only domain import
  (`from aragora.debate.event_bus import DebateEvent, EventBus`) is
  **`TYPE_CHECKING`-only**. Two acceptable fixes; prefer (a):
  (a) **Relocate** `arena_bridge` to `aragora/debate/` (domain) - it is really a
  debate-side adapter that bridges the domain `EventBus` to the domain-free
  cross-subscriber registry; `debate -> events.cross_subscribers` is downward,
  legal. (b) **Drop-type-only-import** per the sanctioned `billing -> agents`
  precedent (P4A_LAYERING_DISPOSITION §Batch 3c): the file already has
  `from __future__ import annotations`, so delete the `TYPE_CHECKING` block and use
  stringized annotations - "no runtime coupling, no shim, NOT `ignore_imports`."
  Either clears the `arena_bridge` contributor to `events -> debate`.

### 5.2 queue SPLIT

- **`queue/worker.py` (353 LOC).** KEEP (domain-free, stays in queue): the
  transport base (`Worker`, job dispatch loop) importing only
  `queue.{base,config,job,retry}`, `core`, `exceptions`. MOVE (domain-coupled):
  `DebateExecutor` + `create_default_executor` (import `agents.base`,
  `debate.orchestrator`) -> a debate/application executor home
  (e.g. `aragora/debate/queue_executor.py`). DROP the `DebateExecutor`/
  `create_default_executor`/`DebateWorker` re-exports from `queue/__init__.py`
  (no shim). Clears `queue -> debate` and the `worker` contributor to
  `queue -> agents`.

### 5.3 Cross-handler coupling must invert to event emission (not direct import)

`handlers.basic._handle_debate_end_to_workflow` and
`subscribers.debate_handlers` currently **lazily import**
`PostDebateWorkflowSubscriber` (application-tier) to trigger post-debate workflows.
If a domain-home handler kept that import it would create a NEW `domain -> workflow`
edge. The inversion removes the direct call: `PostDebateWorkflowSubscriber` is
itself a subscriber that independently reacts to the `debate_end` event via the
domain-free registry. So domain-home handlers **emit/participate**, and the
application-home workflow subscriber reacts **in parallel** - no cross-layer import.
This is the general pattern: **handlers communicate through the domain-free event
bus, never by importing each other across layers.**

> Note: `handlers.strategic` also imports `control_plane.registry` (unlayered) and
> `handlers.notification_handlers` imports `aragora.notifications` (unlayered).
> Unlayered targets impose no layer constraint, but the reaction still moves to the
> home matching its *highest layered* target (workflow=application for strategic's
> workflow reaction; delivery/interface for notification_handlers).

## 6. Consumer census (proves no-shim safety) + repoint list

A no-shim relocate-UP is only safe if **no public/external consumer** depends on the
moved paths. Census (grep across `aragora/`, `tests/`, `scripts/`, `sdk/`):

### 6.1 Moved handler paths (`cross_subscribers.handlers.*`, `subscribers.<handler>`)

- **No `sdk/` consumer. No `aragora/__init__.py` consumer. No non-test runtime
  consumer** outside `aragora/events/`. Runtime access is via
  `subscribers/__init__.py` + `manager.py` (both internal to events, edited in
  place) and via the `get_cross_subscriber_manager()` accessor (which STAYS).
- **Only test consumers** (the ~7 monkeypatching files) require repointing:
  - `tests/memory/test_tier_transition_events.py` (`handlers.basic`)
  - `tests/integration/test_e2e_debate_km_flow.py` (`handlers.validation.record_km_inbound_event`)
  - `tests/events/test_strategic_handlers.py` (`handlers.strategic`)
  - `tests/events/test_consensus_ingestion.py` (`handlers.validation.record_km_inbound_event`)
  - `tests/events/test_workflow_to_supermemory.py` (`handlers.basic`)
  - `tests/nomic/testfixer/test_event_integration.py` (`subscribers.testfixer_handlers`)
  - `tests/events/test_post_debate_workflow.py` (`subscribers.workflow_automation.PostDebateWorkflowSubscriber`)

### 6.2 Moved worker paths (`queue.workers.*`, `DebateExecutor`, `create_default_executor`)

- **No `sdk/` consumer.** Runtime consumers are **interface-layer + scripts**, all
  repointable to the new homes:
  - `aragora/server/startup/workers.py` (`gauntlet_worker`, `testfixer_worker`)
  - `aragora/server/handlers/gauntlet/runner.py` (`gauntlet_worker.enqueue_gauntlet_job`)
  - `scripts/queue_worker.py` (`create_default_executor`)
- Plus queue/worker tests (`tests/queue/workers/*`, `tests/server/startup/test_workers.py`,
  `tests/handlers/gauntlet/test_runner.py`, `tests/queue/test_consensus_healing_worker.py`, ...).
- `aragora/debate/model_combinations.py`'s `SingleDebateExecutor` is an unrelated
  `Callable` type alias, NOT a consumer of `queue.worker.DebateExecutor`.

**Verdict:** no public/external/SDK consumer of any moved path -> the no-shim
relocate-UP exemption (§8) is safe. The repoint set is finite and enumerated above.

## 7. Grimp evidence: the edges clear with zero new un-baselined edge

We simulated the end state in-memory with grimp (`/tmp/sx_grimp_sim.py`): FULL-MOVE
modules are removed from the events/queue subtree (they relocate with no re-export),
and SPLIT modules keep their surface but lose their domain imports. Then we re-ran
`find_illegal_dependencies_for_layers` (the import-linter primitive) and diffed the
edge set. Verbatim result (local tree at `fce9aa5e09`, grimp 3.14, 4238 modules):

```
BEFORE events targets: ['agents', 'debate', 'knowledge', 'memory', 'nomic', 'ranking', 'reasoning', 'server', 'workflow']
BEFORE queue  targets: ['agents', 'debate', 'gauntlet', 'integrations', 'memory', 'nomic', 'ranking', 'server']

FULL-MOVE (remove_module): events cross_subscribers.handlers.*, subscribers.{debate,execution,mound,testfixer,notification}_handlers, workflow_automation;
                           queue workers.{gauntlet,routing,testfixer,consensus_healing,transcription}_worker
SPLIT (remove domain imports only): security_events (-> debate.protocol, debate.orchestrator, agents.api_agents.{anthropic,openai}),
                                    security_dispatcher (-> debate.orchestrator), arena_bridge (-> debate.event_bus),
                                    queue.worker (-> agents.base, debate.orchestrator)

================ RESULT ================
AFTER events targets: ['server']
AFTER queue  targets: []

events->DOMAIN edges remaining (must be empty): []
  events->server remaining heads: ['aragora.events.async_dispatcher', 'aragora.events.dispatcher', 'aragora.events.registry']
queue-> edges remaining (must be empty): []

NEW illegal edges introduced by the simulation (must be empty): []

SIMULATION PASS
```

Reading of the result:

- **events** reduces to `{server}` only: all eight upward edges
  `events -> {agents,debate,knowledge,memory,nomic,ranking,reasoning,workflow}`
  are **cleared**.
- **queue** reduces to `{}` (empty): all eight edges
  `queue -> {agents,debate,gauntlet,integrations,memory,nomic,ranking,server}`
  are **cleared** - including `queue -> server` (both contributors were workers).
- The **subscriber-side** `events -> server` heads (`basic`, `culture`,
  `execution_handlers`, `notification_handlers`) are gone; the **residual**
  `events -> server` heads are `dispatcher`/`async_dispatcher`/`registry`, owned by
  **Batch 1b-sweep** (`dispatcher`/`async_dispatcher` -> `server.middleware.tracing`,
  confirmed direct) and **Batch 2c** (`dispatcher`/`async_dispatcher` ->
  `server.handlers.webhooks`, Tier-3). `events -> server` therefore stays a
  (baselined) edge after this batch - as expected; it is NOT this batch's target.
- **Zero NEW illegal edges** were introduced (the removals cannot add edges; the
  home-assignment rule §4.2 guarantees the *destination* introduces none either -
  every relocated reaction lands in a home at or above its highest import).

> **Fidelity note.** The simulation removes both a handler and any sibling handler
> it couples to (e.g. `debate_handlers` + `workflow_automation`), which is only
> valid because §5.3 inverts that coupling to event emission. If the real
> implementation kept a domain-home handler importing the application `workflow`
> subscriber, it would introduce `debate -> workflow`; the inversion (event
> emission, not import) is therefore load-bearing and is called out as a batch
> acceptance criterion (§10).

**Reproduce:** from the worktree root,
`SX_REPO_ROOT="$PWD" python3 /tmp/sx_grimp_sim.py` (evidence scripts are throwaway,
not committed). The authoritative gate remains
`python3 scripts/ci/check_import_contracts.py --layers foundation,infrastructure`
run after each impl batch lands.

## 8. Shim policy: the relocate-UP no-shim exemption

The usual rule (VAL-CROSS-004 / shims.md) is "shim every moved path" so callers of
the old import path keep working. **That rule is INVERTED for a layering relocate-UP**
because an identity-preserving re-export shim at the old `events`/`queue` path
re-imports the relocated (now higher-layer) module and thereby **re-creates the very
`infra -> {domain,application,interface}` edge the move was meant to delete** (proven
by worker `5f92db63`, §1). For this specific case the correct action is: **move with
NO re-export shim, and repoint every consumer.**

Guardrails that make the exemption safe (all satisfied here):
1. The **domain-free surface stays** in `events`/`queue` (registry, dispatch,
   transport, `get_cross_subscriber_manager`, security types/emitter). Only the
   domain-coupled code moves. Public accessors do not move.
2. A **consumer census** (§6) confirms **no public/external/SDK consumer** of the
   moved paths - only internal runtime (repointed) and the ~7 monkeypatching tests
   (repointed). If any external consumer existed, the exemption would NOT apply and
   an alternative (keep the surface + invert, or a *domain-free* facade) would be
   required instead.
3. grimp shows the edges clear with **no new un-baselined edge** (§7).

### 8.1 Proposed AGENTS.md addition (under "P4a Contracts-Thread Shared Rules")

```md
- **Relocate-UP no-shim exemption (layering only).** The standard "shim every moved
  path" rule (VAL-CROSS-004) is SUSPENDED when a module is moved *up* a layer to
  clear a lower-layer `importer -> imported` contract edge (e.g. moving a
  domain-coupled handler/worker out of infra `events`/`queue`). A re-export shim at
  the old path re-imports the relocated higher-layer module and RE-CREATES the exact
  edge the move deletes (grimp counts indirect, lazy, and TYPE_CHECKING chains
  through the shim), so the shim is forbidden here. Instead: (a) keep only a
  DOMAIN-FREE surface at the old package (registry/dispatch/transport/public
  accessors with zero domain imports, eager or lazy); (b) move the domain-coupled
  code to a home whose layer is >= the highest layer it imports (prefer the coupled
  domain package; use an existing application/interface home for multi-domain/server
  couplings; do not invent a new top-level package without recording module_tiers
  drift); (c) REPOINT every consumer of the moved path - runtime AND tests
  (including `@patch("<old.module.path>")` monkeypatches) - to the new home. This
  exemption REQUIRES a committed consumer census proving no public/external/SDK
  consumer of the moved path, plus grimp evidence that the target edge clears with
  no new un-baselined edge. If any external consumer exists, the exemption does not
  apply.
```

### 8.2 Proposed `{missionDir}/library/shims.md` policy note

```md
## Relocate-UP exception (no shim)

Do NOT add a re-export shim when the move is a layering relocate-UP intended to
clear an import-contract edge (infra -> domain/application/interface). A shim at the
old path re-imports the relocated module and re-creates the edge (empirically:
worker 5f92db63, resolved_violations == {} after relocation + eager shim; grimp
counts indirect/lazy/TYPE_CHECKING chains through the shim). Correct pattern:
- keep a domain-free surface at the old package (registry/dispatch/transport +
  public accessors, zero domain imports);
- move domain-coupled code to a home at or above its highest import layer;
- repoint ALL consumers (runtime + tests, including monkeypatch string paths); no
  shim.
Precondition: a consumer census showing no public/external/SDK consumer of the moved
path, plus grimp evidence the edge clears with no new un-baselined edge. This is the
ONLY sanctioned exception to "shim every moved path". (See
docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md.)
```

## 9. module_tiers drift

The home assignments in §4.3 use **only existing packages**
(`knowledge`, `memory`, `debate`, `reasoning`, `ranking`, `workflow`, `nomic`,
`server`) - **no new top-level package is created**, so there is **no
`aragora/module_tiers.yaml` drift** and no operator Decision line is required for
package addition. A brand-new `aragora/orchestration` package was explicitly
rejected (it would be unlayered/new and would still not solve the server-coupled
reactions, which need the interface layer). If a future implementer decides a new
package is warranted, they MUST add it to `module_tiers.yaml` + `.importlinter` and
record the drift per VAL-P0-006.

## 10. Batch breakdown (<=800 LOC coherent impl sub-features)

Ordered; each is Tier 1-2 (touches only `aragora/events`, `aragora/queue`, and the
domain/application/interface home packages + their tests) unless noted. LOC is an
estimate of code touched (moved + edited + repointed + tests); a couple sit near the
cap and carry a noted split point. **`events.dispatcher`/`async_dispatcher` ->
`server` (webhook signing + tracing) is OUT OF SCOPE** here - that is Batch 1b-sweep
+ Batch 2c (2c is Tier-3, `aragora/server/handlers/webhooks.py`).

### Events (supersedes cancelled Batch 2a)

| # | Sub-feature | Scope | Clears | ~LOC |
|---|---|---|---|---|
| E1 | Domain-free cross-subscriber registry core | `cross_subscribers/{__init__,manager}.py` -> registry-backed dispatch; add `register_subscriber`/`register_factory`/`get`/`reset`/`SubscriberStats`; keep `get_cross_subscriber_manager` stable; add interface superset + domain-subset bootstrap skeletons. Handlers still in place (register via new API). | none (enabler; no new edge) | ~600 |
| E2 | knowledge-domain home | Move KM reactions from `handlers.{knowledge_mound,validation}` (+ knowledge bits of `basic`/`culture`/`strategic`, `subscribers.mound_handlers`) -> `aragora/knowledge/event_subscribers.py`; self-register; NO shim; repoint `test_e2e_debate_km_flow`, `test_consensus_ingestion`. **Split E2a(knowledge_mound)/E2b(validation) if >800.** | contributes `events->{knowledge,memory}` | ~750 |
| E3 | memory + reasoning + ranking homes | Move memory reactions (`mound_handlers`, `knowledge_mound`/`validation` memory bits), `basic` reasoning (`reasoning.belief`), `execution_handlers` ranking (`ranking.elo`) -> `aragora/{memory,reasoning,ranking}/event_subscribers.py`; repoint tests. | `events->{reasoning,ranking}`; contributes `events->memory` | ~600 |
| E4 | debate-domain home | Move debate reactions from `subscribers.debate_handlers` + `handlers.{basic,strategic}` debate bits -> `aragora/debate/event_subscribers.py`; repoint `test_strategic_handlers`, `test_tier_transition_events`. | contributes `events->debate` | ~700 |
| E5 | application homes (workflow + nomic) + coupling inversion | Move `subscribers.workflow_automation` + `handlers.strategic` workflow reaction + `basic._handle_debate_end_to_workflow` -> `aragora/workflow/event_subscribers.py`; move `subscribers.testfixer_handlers` -> `aragora/nomic/...`; **invert `basic`/`debate_handlers` -> workflow to event emission** (§5.3); repoint `test_post_debate_workflow`, `test_event_integration`. | `events->{workflow,nomic}` | ~700 |
| E6 | interface home (server-coupled reactions) | Move `basic` webhook delivery, `culture` state_manager, `execution_handlers` emitter, `notification_handlers` -> `aragora/server/event_subscribers.py`; register at server startup; repoint `test_workflow_to_supermemory`. | subscriber-side `events->server` | ~700 |
| E7 | security split + arena_bridge + manager de-mixin | SPLIT `security_events` (move `build_security_debate_question` + agent-runner -> `aragora/debate/security_response.py`); SPLIT `security_dispatcher` (move Arena-runner fn); relocate `arena_bridge` -> `aragora/debate/` (or drop TYPE_CHECKING import); finalize `manager.py` domain-free (remove last mixin imports). | `events->agents` + remaining `events->debate` | ~800 |

After E1-E7: `events -> {agents,debate,knowledge,memory,nomic,ranking,reasoning,workflow}`
cleared; `events -> server` remains (Batch 1b-sweep/2c).

### Queue (supersedes cancelled Batch 2b)

| # | Sub-feature | Scope | Clears | ~LOC |
|---|---|---|---|---|
| Q1 | Domain-free job-handler registry | `queue/__init__.py` + `queue/worker.py`: expose `register_worker`/`register_job_handler` + `get`/`reset`; keep transport base; DROP `DebateExecutor`/`create_default_executor`/`DebateWorker` re-exports from `__init__` (no shim). | none (enabler; no new edge) | ~500 |
| Q2 | DebateExecutor -> executor home | Move `DebateExecutor` + `create_default_executor` out of `queue/worker.py` -> `aragora/debate/queue_executor.py`; repoint `scripts/queue_worker.py`, tests. | `queue->debate` + `worker` share of `queue->agents` | ~500 |
| Q3 | gauntlet + ranking + memory workers | Move `workers.gauntlet_worker` -> `aragora/gauntlet/`, `workers.consensus_healing_worker` -> `aragora/memory/`; repoint `server/startup/workers.py`, `server/handlers/gauntlet/runner.py`, tests. **Split gauntlet vs consensus if >800.** | `queue->{gauntlet,ranking,memory}` + gauntlet share of `queue->{agents,server}` | ~800 |
| Q4 | routing + testfixer + transcription workers | Move `workers.routing_worker` -> integrations/application home, `workers.testfixer_worker` -> `aragora/nomic/`, `workers.transcription_worker` -> `aragora/transcription/`; repoint tests. | `queue->{integrations,nomic}` + routing share of `queue->server` | ~800 |

After Q1-Q4: `queue -> {agents,debate,gauntlet,integrations,memory,nomic,ranking,server}`
fully cleared.

**Per-batch acceptance criteria:** (1)
`check_import_contracts.py --layers foundation,infrastructure` shows no NEW edge and
the targeted contributor(s) removed; (2) NO re-export shim at any moved path;
(3) all repointed consumers/tests pass; (4) `get_cross_subscriber_manager()` import
path unchanged.

## 11. References

- `docs/architecture/P4A_LAYERING_DISPOSITION.md` (lines 185-190, 326-349: preferred
  EventBus/job-handler registry inversion; Batch 2a/2b/2c decomposition;
  `billing -> agents` TYPE_CHECKING drop precedent).
- Handoff `5f92db63` (relocate-UP + shim impossibility diagnosis).
- `{missionDir}/library/p4a-deferred-shim-callers.md`,
  `{missionDir}/library/p4a-layering-disposition-batches.md`,
  `{missionDir}/library/shims.md`.
- `.importlinter` (5-layer contract), `scripts/ci/check_import_contracts.py`,
  `scripts/baselines/import_contracts_baseline.json` (frozen at `5ce80610c6`).
- Grimp evidence scripts (throwaway, not committed): `/tmp/sx_grimp_evidence.py`,
  `/tmp/sx_grimp_sim.py`.
