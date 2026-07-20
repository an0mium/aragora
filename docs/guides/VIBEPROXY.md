# VibeProxy Local Transport

Aragora can use a locally running VibeProxy for developer and operator model
calls when an operator opts in. VibeProxy remains a transport: provider family
and requested model identity do not change, and it does not create an
additional review signal.

## Current Scope

- Supported now: bounded Fable/Claude advisory consults through the Anthropic
  Messages protocol. Consults use the direct CLI path unless VibeProxy is
  explicitly selected.
- Direct by default: normal agents, CI, production servers, credential checks,
  public gateways, and merge-quorum evidence collection.
- Deferred until contract-tested: web search, tools, embeddings, image, audio,
  and other media capabilities.

The local implementation uses `http://127.0.0.1:8318/v1`. This is VibeProxy's
loopback-bound core endpoint and is provisional. Port `8317` is not selected:
the tested macOS application listened on all interfaces there, and Aragora
rejects it even when explicitly configured.

## Configuration

```bash
export ARAGORA_MODEL_TRANSPORT=vibeproxy-prefer
export ARAGORA_VIBEPROXY_BASE_URL=http://127.0.0.1:8318
```

Modes:

- `direct`: retain existing behavior. This is the project and advisory-consult
  default.
- `vibeproxy-prefer`: try the exact catalog model through VibeProxy, then use
  the existing backend order if the proxy is unavailable.
- `vibeproxy-required`: fail closed if VibeProxy cannot serve the exact model.

Optional settings are `ARAGORA_VIBEPROXY_API_KEY`,
`ARAGORA_VIBEPROXY_CATALOG_TTL_SECONDS`, and
`ARAGORA_VIBEPROXY_MODEL_MAP`. Model mappings are explicit JSON; no semantic
substitution occurs by default.

Plaintext endpoints must use a literal loopback IP. Remote endpoints require
HTTPS and an explicit key. Credentials are never included in diagnostics.
Requests ignore ambient HTTP proxy settings and reject redirects so prompts
and authorization headers cannot escape the resolved endpoint.
Loopback prevents network exposure but does not authenticate the local server.
Opting in therefore trusts the process bound to the configured endpoint with
the full consult prompt. Use `direct` on a shared or untrusted host; broader
rollout requires a separate server-authentication or endpoint-pinning design.

## Readiness Diagnostic

Check the configured endpoint without sending a prompt or making any inference
request:

```bash
python3 scripts/check_vibeproxy.py
python3 scripts/check_vibeproxy.py --json
```

The command always performs a fresh `GET /v1/models`. If that required catalog
request succeeds, it may also perform `GET /` to read advertised routes and an
allowlisted version header. It never calls `/messages`, `/chat/completions`,
`/completions`, or another prompt-bearing route. Exit code `0` means the live
catalog was non-empty and well formed. Configuration, availability, timeout,
redirect, and malformed-response failures exit nonzero; `--json` still emits
exactly one schema-versioned object.

Schema version `1` contains:

- `endpoint`: the normalized URL and literal-loopback classification. Unsafe
  input is rejected before this field is populated, so userinfo and query data
  are never echoed.
- `version`: an allowlisted HTTP-header value and its exact source, or
  `{ "value": null, "source": "unknown" }`. The diagnostic does not guess a
  version from model names or make remote use depend on a local app bundle.
- `protocols.advertised`: sanitized method/path pairs reported by `GET /`.
  `verified_no_inference` separately lists only `GET /v1/models`, which this
  run actually exercised. `aragora_implemented_not_probed` records the
  Anthropic Messages route implemented by Aragora without claiming the
  diagnostic verified it.
- `model_inventory`: a sorted model-ID list and count.
- `catalog_freshness`: the age and configured TTL of the forced live catalog
  observation. Age and freshness are process-local monotonic-clock values, not
  server timestamps. With a zero TTL, the observation is intentionally not
  cache-fresh even though the request succeeded.
- `latency_ms`: catalog, optional metadata, and total wall-clock timings. One
  total budget covers both GETs; metadata cannot reset it. Set the budget with
  `--timeout-seconds` or
  `ARAGORA_VIBEPROXY_DIAGNOSTIC_TIMEOUT_SECONDS`.
- `error`: `null` on readiness, otherwise a stable sanitized category and
  message. Response bodies, redirect locations, authorization headers, API
  keys, prompts, and token-bearing URL data are never included.

The diagnostic preserves the same trust boundary as transport requests:
plaintext is literal-loopback-only, remote endpoints require HTTPS and an
explicit API key, port `8317` is prohibited, ambient proxies are disabled,
redirects are denied, and response size plus wall-clock reads are bounded.
Catalog readiness is not proof that a prompt-bearing protocol works or that the
local process is trustworthy.

## Verify

```bash
python3 scripts/consult_claude.py --json "Reply with exactly DIRECT"
ARAGORA_MODEL_TRANSPORT=vibeproxy-prefer \
  python3 scripts/consult_claude.py --json "Reply with exactly PREFER_PROXY"
ARAGORA_MODEL_TRANSPORT=vibeproxy-required \
  python3 scripts/consult_claude.py --json "Reply with exactly PROXY_ONLY"
```

## Fallback Rules

`vibeproxy-prefer` tries the requested Claude model and configured fallback
model through VibeProxy before the existing Claude CLI and opt-in paid API
fallbacks. `vibeproxy-required` never falls through to those backends. Timeout
budgets include every enabled VibeProxy attempt.
