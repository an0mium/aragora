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
