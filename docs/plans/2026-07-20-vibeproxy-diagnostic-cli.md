# Plan: Sanitized VibeProxy Diagnostic CLI

## Mission

Deliver the first bounded work unit from issue #9409: a safe, machine-readable
`scripts/check_vibeproxy.py --json` command that tells an operator whether the
configured VibeProxy endpoint is usable without making an inference request or
exposing credentials. The command, its fake-proxy tests, and its documentation
must be review-ready on one dedicated PR while all later routing, adapter,
metrics, trust, burn-in, shadow, and governance work remains separate.

## Scope

### In Scope

- Add `scripts/check_vibeproxy.py` with human-readable output and a stable JSON
  schema.
- Reuse the secure request and URL-validation behavior in
  `aragora/agents/transports/vibeproxy.py`; make only the smallest additive
  transport changes needed to expose sanitized diagnostic metadata.
- Report these fields without sending a model prompt:
  - schema version and overall readiness
  - sanitized endpoint and loopback classification
  - server/app version when safely discoverable, otherwise an explicit unknown
  - advertised and actually verified protocol surfaces
  - sorted model inventory and count
  - catalog freshness/TTL information
  - bounded catalog and total latency
- Add fake-loopback-proxy tests for success, unavailable, malformed, timeout,
  redirect, unsafe endpoint, and redaction behavior.
- Preserve the existing direct transport default and the existing
  `consult_claude.py` fallback semantics.
- Document the command, JSON semantics, exit codes, trust boundary, and the
  fact that the diagnostic performs no inference call.

### Out of Scope

- OpenAI chat/Responses routing.
- Anthropic, Grok, Gemini, or Kimi adapter work.
- New transport metrics, trace metadata, or inference call-site allowlists.
- Server authentication, endpoint pinning, or any relaxation of the current
  loopback/HTTPS safety policy.
- Seven-day burn-in, reviewer shadows, countable evidence, merge-quorum policy,
  or any Tier-4 governance change.
- Selecting or probing prohibited port 8317.
- Merging this PR; the user retains merge authority.

## Output Contract

`--json` emits exactly one JSON object. The implementation may use nested
objects, but it must preserve these top-level concepts:

- `schema_version`: integer, initially `1`
- `ok`: boolean readiness result
- `endpoint`: sanitized URL and loopback status; never credentials or query data
- `version`: safely discovered value plus source, or an explicit unknown value
- `protocols`: distinguish server-advertised routes from protocol behavior the
  diagnostic actually verified; do not claim a prompt-bearing protocol was
  verified when no inference request was sent
- `model_inventory`: sorted model IDs and count
- `catalog_freshness`: age, configured TTL, and whether the observation is fresh
- `latency_ms`: bounded catalog and total timings
- `error`: `null` on success or a stable sanitized category/message on failure

Exit `0` only when the required catalog probe succeeds and the response is
well-formed. Configuration, timeout, unavailability, and malformed-response
failures exit nonzero while still emitting valid JSON under `--json`.

## Batches

### Batch 1: Diagnostic Command, Tests, and Documentation

**Tasks:**

- [ ] Define the diagnostic result model and CLI parser in
  `scripts/check_vibeproxy.py`.
- [ ] Extend the existing VibeProxy client only where necessary to obtain
  sanitized metadata through the same no-proxy, no-redirect, bounded-read path.
- [ ] Add category-level fake-proxy and CLI tests.
- [ ] Run focused regression tests for the existing transport and direct
  `consult_claude.py` behavior.
- [ ] Update `docs/guides/VIBEPROXY.md` with usage and field semantics.
- [ ] Complete an independent cumulative review, respond to PR feedback, and
  run the final readiness gate on the exact branch tip.

**Acceptance criteria:**

- [ ] `python3 scripts/check_vibeproxy.py --json` always emits parseable,
  credential-free JSON and uses meaningful exit codes.
- [ ] A fake loopback proxy proves success without any `/messages`,
  `/chat/completions`, or other inference request.
- [ ] Fake-proxy tests cover slow/malformed/redirecting responses and prove the
  total diagnostic deadline cannot be extended by slow-drip reads.
- [ ] Port 8317, non-loopback plaintext, URL credentials/query data, and ambient
  proxy/redirect escape remain rejected.
- [ ] Existing `VibeProxyClient`, transport-policy, and direct
  `consult_claude.py` tests remain green with no tests removed, skipped, or
  weakened.
- [ ] Changed Python files pass focused mypy and changed-file pre-commit checks.
- [ ] `bash scripts/automation_pr_preflight.sh origin/main HEAD` passes before
  the branch is called review-ready.
- [ ] Documentation states exactly which protocols are advertised versus
  verified and explains unavailable version values rather than guessing.
- [ ] PR diff remains within the issue's single diagnostic work unit and below
  the operating contract's 800-LOC limit.

**Docs likely touched:**

- `docs/guides/VIBEPROXY.md`
- this plan during staging only; Elves session artifacts are removed at final
  completion

**Risk:** Medium. The command inspects a credential-bearing local transport;
the main risks are leaking sensitive configuration, overstating protocol
support, or creating a second unsafe HTTP path instead of reusing the existing
bounded client.

## Non-Negotiables

- VibeProxy remains a transport, never a reviewer/provider family.
- The diagnostic makes no model inference request and never emits API keys,
  authorization headers, raw response bodies, URL credentials, or query data.
- Never select port 8317; preserve literal-loopback-only plaintext, HTTPS for
  remote endpoints, disabled ambient proxies, disabled redirects, bounded
  reads, and wall-clock deadlines.
- Preserve direct-by-default behavior for normal agents, CI, production,
  credential checks, public gateways, and merge evidence.
- Do not modify workflows, required checks, merge/evidence/settlement authority,
  protected governance docs, or public APIs.
- Never weaken, skip, delete, or rewrite an existing test to make the batch pass.
- Never merge this PR; prepare it for user review.

## Test Strategy

- **Baseline/focused tests:**
  `python3 -m pytest tests/agents/transports/test_vibeproxy.py tests/scripts/test_consult_claude.py -q`
- **New diagnostic tests:**
  `python3 -m pytest tests/scripts/test_check_vibeproxy.py -q`
- **Typecheck:** changed Python files with the repository mypy configuration and
  skipped transitive imports if required by the existing baseline.
- **Lint:** changed-file pre-commit hooks only.
- **Repository preflight:**
  `bash scripts/automation_pr_preflight.sh origin/main HEAD`
- **Smoke:** fake proxy is required; optional live loopback smoke may run only
  against port 8318 and must not send a prompt.
- **Preview/E2E:** no deployment or browser flow applies to this operator CLI.

## Batch Sizing

```yaml
team-size: 1
sprint-length: 1 day
```

## Notes

- Source issue: https://github.com/synaptent/aragora/issues/9409
- Foundation PR #9408 is merged at `bc9652d70fb1b31f30fda2d012dbccee6ae5a748`.
- The current transport already centralizes URL safety, catalog caching,
  deadline-bounded reads, no-proxy behavior, no redirects, and sanitized
  status output. Extend that surface rather than creating a parallel HTTP
  client.
- The live port-8318 root advertises OpenAI-compatible routes while the Aragora
  transport has separately exercised Anthropic Messages. The diagnostic must
  report those as different evidence categories rather than conflating them.
- The installed macOS app may provide a bundle version even when the server
  omits version headers. Any local fallback must be optional, clearly sourced,
  and must not make remote diagnostics macOS-specific.
