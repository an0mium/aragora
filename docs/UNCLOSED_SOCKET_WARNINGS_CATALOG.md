# Unclosed Socket/Transport Warnings Catalog

Audit of code patterns that produce (or may produce) `ResourceWarning: unclosed`
messages for sockets, transports, connectors, and event loops.

Generated: 2026-03-28

---

## Executive Summary

The test suite (216k+ tests) does **not** currently emit unclosed socket/transport
`ResourceWarning`s at runtime — Python's default warning filters suppress them unless
`-W all` or `PYTHONWARNINGS=all` is set. However, the codebase contains **multiple
patterns** that would generate these warnings under strict resource tracking. The
existing `aragora.debate.runtime_blockers` module already classifies these warnings
when they appear in subprocess stderr during debate dogfooding.

**Key numbers:**
- 25+ files with lazy `aiohttp.ClientSession()` stored on `self._session` (no `async with`)
- 4 files with raw `socket.socket()` without context managers
- 6+ test files creating `asyncio.new_event_loop()` manually
- 0 warnings currently surfaced by default pytest config (no `filterwarnings` for ResourceWarning)

---

## Category 1: Lazy aiohttp.ClientSession (Stored on Instance)

These create a `ClientSession` on first use and store it on `self`. If the owning
object is garbage-collected without an explicit `.close()` call, the session's
underlying TCP connector and sockets are leaked.

### Pattern
```python
# Lazy init — no async with
if self._session is None:
    self._session = aiohttp.ClientSession(...)
```

### Affected Files

| File | Line(s) | Has close()? | Notes |
|------|---------|-------------|-------|
| `aragora/client/client.py` | 486, 603, 724, 841, 912 | Yes (`close_async` L1129) | 5 duplicate lazy-init sites; callers must remember to close |
| `aragora/gateway/federation/registry.py` | 762, 942 | Yes (shutdown method) | Used for health checks and capability fetches |
| `aragora/gateway/enterprise/proxy/core.py` | 201 | Unclear | Session with custom TCP connector |
| `aragora/marketplace/client.py` | 48 | Yes (L51) | |
| `aragora/services/threat_intelligence/service.py` | 222 | Yes (L273) | |
| `aragora/connectors/ecommerce/shopify/client.py` | 107 | Check needed | |
| `aragora/connectors/ecommerce/shopify.py` | 390 | Check needed | |
| `aragora/connectors/enterprise/documents/onedrive.py` | 290 | Check needed | |
| `aragora/connectors/enterprise/documents/dropbox.py` | 190 | Check needed | |
| `aragora/connectors/ecommerce/woocommerce/client.py` | 120 | Check needed | |
| `aragora/connectors/accounting/base.py` | 200 | Check needed | `self._http_client` variant |
| `aragora/agents/api_agents/external_framework.py` | 254 | Check needed | |
| `aragora/integrations/slack.py` | 114 | Check needed | |
| `aragora/integrations/base.py` | 147 | Check needed | Base class — all integrations inherit |
| `aragora/integrations/email.py` | 311 | Check needed | |
| `aragora/integrations/slack_debate.py` | 746 | Check needed | |
| `aragora/integrations/teams.py` | 157 | Check needed | |
| `aragora/integrations/discord.py` | 107 | Check needed | |
| `aragora/integrations/telegram.py` | 134 | Check needed | |
| `aragora/integrations/whatsapp.py` | 147 | Check needed | |
| `aragora/integrations/matrix.py` | 129 | Check needed | |
| `aragora/integrations/zoom.py` | 200 | Check needed | |
| `aragora/integrations/exporters/jira_adapter.py` | 93 | Check needed | |
| `aragora/integrations/exporters/linear_adapter.py` | 111 | Check needed | |
| `aragora/integrations/exporters/webhook_adapter.py` | 87 | Check needed | |
| `aragora/integrations/openclaw/client.py` | 121 | Check needed | |

### Safe Pattern (no warning risk)

These files use `async with aiohttp.ClientSession()` — session is always closed:

- `aragora/memory/embeddings.py` (L206, 237, 284, 320)
- `aragora/gateway/health.py` (L271)
- `aragora/gateway/enterprise/audit_interceptor/interceptor.py` (L685)
- `aragora/gateway/openclaw/sandbox.py` (L312)
- `aragora/gateway/openclaw/adapter.py` (L838, 988)
- `aragora/connectors/legal/docusign.py` (L333, 374, 409)
- `aragora/core/embeddings/backends/ollama.py` (L103)
- `aragora/core/embeddings/backends/openai.py` (L124)
- `aragora/core/embeddings/backends/gemini.py` (L82)
- `aragora/notifications/receipt_delivery.py` (L502)
- `aragora/notifications/providers.py` (L271, 291, 582)
- `aragora/agents/errors/decorators.py` (L294)
- `aragora/connectors/accounting/qbo.py` (L204, 275)
- `aragora/connectors/accounting/gusto.py` (L395, 441)
- `aragora/http_client.py` (L86 — factory, callers use `async with`)

---

## Category 2: Raw Socket Without Context Manager

These create `socket.socket()` and call `.close()` manually. If an exception is
raised between creation and `.close()`, the socket leaks.

### Pattern
```python
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
result = sock.connect_ex((host, port))
sock.close()  # skipped if connect_ex raises
```

### Affected Files

| File | Line(s) | Notes |
|------|---------|-------|
| `aragora/server/handlers/oauth_wizard.py` | 756-759 | SMTP connectivity check; no try/finally around close |
| `tests/test_debate_embeddings.py` | 20-23 | Ollama detection helper; same pattern |
| `aragora/server/handlers/integration_management.py` | Similar | Socket connectivity check |

### Safe Examples (use context manager)

- `aragora/memory/embeddings.py` (L398-402): `with socket.socket(...) as sock:`
- `aragora/core/embeddings/backends/ollama.py` (L69-72): `with socket.socket(...) as sock:`
- `tests/e2e/server_fixture.py` (L32-35): `with closing(socket.socket(...)) as s:`

---

## Category 3: Manual Event Loop Creation in Tests

Tests that create `asyncio.new_event_loop()` can leak transports if the loop is
not properly drained before closing.

### Affected Files

| File | Line(s) | Properly Closed? | Notes |
|------|---------|-------------------|-------|
| `tests/test_streams_extended.py` | 46-50 | Yes (with broad `except`) | Silently swallows close errors |
| `tests/memory/test_benchmark.py` | 353-365 | Yes | Resets event loop to None |
| `tests/server/handlers/test_rlm_handler.py` | 33-37 | Yes | Module-scope fixture |
| `tests/server/test_startup_validation_runner.py` | 42-46 | Yes | |
| `tests/server/test_postgres_storage.py` | ~37 | Yes | Mentions "stale event loop" issue |
| `tests/test_handlers_system.py` | 45-49 | Yes | Helper `_run()` function |
| `aragora/billing/budget_alert_notifier.py` | 121-133 | Yes (try/finally) | Production code, not test |

---

## Category 4: Existing Detection Infrastructure

The codebase already has infrastructure to detect and classify these warnings:

### `aragora/debate/runtime_blockers.py`

Classifies stderr from debate subprocesses:
- `resource_warning` — matches `\bResourceWarning\b`
- `unclosed_connector` — matches `\bunclosed connector\b`
- `unclosed_transport` — matches `\bunclosed transport\b`
- `tracemalloc_hint` — matches `\benable tracemalloc\b`
- `leaked_semaphore_warning` — matches `\bleaked semaphore objects\b`

Used to distinguish warning-only runs from blocker failures in debate dogfooding.

### `aragora/inbox/triage_diagnostics.py` (L261)

Enables `warnings.simplefilter("always", ResourceWarning)` during triage runs to
surface resource leaks in diagnostics output.

### `tests/debate/test_runtime_blockers.py`

Tests that verify the classifier correctly handles:
- `ResourceWarning: unclosed <socket.socket ...>` (L10)
- `Unclosed connector` (L12)
- `ResourceWarning: unclosed transport <_SelectorSocketTransport fd=63>` (L35)

---

## Category 5: Unawaited Coroutines (Related)

During test runs, several `RuntimeWarning: coroutine ... was never awaited` warnings
appear from `tests/nomic/test_postgres_cycle_store.py::TestSyncWrappers`. These come
from sync wrappers calling async methods of `PostgresCycleLearningStore` without
awaiting them. While not socket warnings per se, they indicate async resource
management issues in that module.

---

## Recommendations

1. **Add pytest filterwarnings config** to surface ResourceWarnings during CI:
   ```toml
   [tool.pytest.ini_options]
   filterwarnings = [
       "error::ResourceWarning",
   ]
   ```

2. **Fix raw socket patterns** — replace manual `.close()` with context managers in:
   - `aragora/server/handlers/oauth_wizard.py:756`
   - `tests/test_debate_embeddings.py:20`

3. **Audit lazy ClientSession classes** — ensure all 25+ classes with
   `self._session = aiohttp.ClientSession()` implement `__aenter__`/`__aexit__`
   or have explicit `close()` called in their lifecycle.

4. **Consider `aragora/integrations/base.py`** — since it's the base class for all
   integrations, adding proper session lifecycle there would fix 10+ subclasses.

5. **Fix unawaited coroutines** in `tests/nomic/test_postgres_cycle_store.py` sync
   wrapper tests.
