# Project Learnings: VibeProxy Diagnostic Run

## Repo Conventions

- 2026-07-20: Extend `VibeProxyClient` for transport diagnostics rather than
  creating a second HTTP client; it already enforces URL safety, no ambient
  proxies, no redirects, bounded response sizes, and wall-clock reads.
- 2026-07-20: Each #9409 work unit belongs on a separate bounded PR. This run
  owns only the diagnostic command.

## Validation and Tooling

- 2026-07-20: Disposable worktrees do not contain their own Python environment;
  use the shared repository environment for focused pytest, pre-commit, and
  mypy, while keeping committed commands portable.
- 2026-07-20: The diagnostic's required integration proof is a fake loopback
  HTTP proxy. A live 8318 probe is optional and must not send a prompt.

## Review Heuristics

- 2026-07-20: Reviewers previously found deadline-extension, malformed-config
  fallback, credential-cache isolation, redirect/proxy escape, and bounded-read
  defects in this transport family. Treat those as regression categories, not
  one-off examples.

## Product and Domain Invariants

- 2026-07-20: VibeProxy is a transport, not a reviewer/provider family; logical
  family identity and exact model identity remain independent of the route.
- 2026-07-20: Port 8317 is prohibited. Plaintext requires a literal loopback IP;
  remote endpoints require HTTPS and an explicit key.
- 2026-07-20: Server-advertised protocols, no-prompt verified protocols, and
  Aragora's implemented transport capabilities are distinct facts and must not
  be conflated in diagnostics.

## Known Traps

- 2026-07-20: The current 8318 server exposes version-header names through CORS
  but may omit actual version header values. Report unknown or a clearly sourced
  local bundle fallback; never invent a version.
- 2026-07-20: A new process has no durable in-memory catalog cache. Catalog
  freshness fields must have explicit semantics rather than implying cross-run
  cache persistence.

## Retired Learnings

- None.
