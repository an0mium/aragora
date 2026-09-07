# Worker and Single-Instance Environment Variables

**Status:** Governance reference
**Last updated:** March 13, 2026

This note documents the deployment intent for `WORKER_MODE` and
`ARAGORA_SINGLE_INSTANCE`. It closes the documentation gap identified in
[DRIFT-002](./deploy-truth-table.md#drift-002-kubernetes-worker-has-no-explicit-entrypoint)
and [DRIFT-007](./deploy-truth-table.md#drift-007-aragora_single_instancetrue-undocumented).

## Summary

| Variable | Intended use | Safe values | Primary risk |
|----------|--------------|-------------|--------------|
| `WORKER_MODE` | Mark a process as running in a worker deployment context | `true`, unset | Setting it without an explicit worker command can still start the server |
| `ARAGORA_SINGLE_INSTANCE` | Declare that the deployment is a single-node install without distributed coordination | `true`, `false`, unset | Enabling it in a multi-node deployment suppresses Redis-backed safety checks |

## `WORKER_MODE`

`WORKER_MODE=true` currently appears in the Kubernetes debate-worker
deployment. By itself, this variable does not define the process entrypoint.
The deployment still needs an explicit worker command such as:

```bash
python -m scripts.queue_worker --concurrency 3
```

Operational guidance:

- Use `WORKER_MODE=true` only on pods or services that are intended to run the
  queue worker path.
- Pair it with an explicit worker `command:` or equivalent process manager
  setting.
- Do not rely on `WORKER_MODE=true` to convert the image default CMD into a
  worker automatically unless that behavior is explicitly implemented and
  tested.

This is the documented mitigation for
[DRIFT-002](./deploy-truth-table.md#drift-002-kubernetes-worker-has-no-explicit-entrypoint):
the env var is deployment metadata, not a substitute for the worker entrypoint.

## `ARAGORA_SINGLE_INSTANCE`

`ARAGORA_SINGLE_INSTANCE=true` declares that Aragora is running as a single
node and should not require distributed coordination services for leader
election and shared runtime state.

Observed runtime behavior:

- In production startup validation, `ARAGORA_SINGLE_INSTANCE=true` suppresses
  the requirement for `REDIS_URL` when the deployment is intentionally
  single-node.
- In control-plane leader election, it disables the default assumption that a
  production deployment needs Redis-backed distributed state.
- In degraded-mode and validator messaging, it is the documented fallback when
  operators choose a single-node deployment without Redis.

Operational guidance:

- Set `ARAGORA_SINGLE_INSTANCE=true` for EC2 or other single-node deployments
  that intentionally run without separate workers or Redis-backed coordination.
- Leave it unset or set it to `false` for distributed deployments.
- Do not combine `ARAGORA_SINGLE_INSTANCE=true` with a deployment that runs
  multiple backend replicas unless distributed-state requirements are reviewed
  explicitly.

This documents the previously missing operator contract called out in
[DRIFT-007](./deploy-truth-table.md#drift-007-aragora_single_instancetrue-undocumented).

## Recommended deployment patterns

| Topology | `WORKER_MODE` | `ARAGORA_SINGLE_INSTANCE` | Notes |
|----------|---------------|---------------------------|-------|
| Single-node EC2 | Unset | `true` | Server runs without Redis-backed coordination |
| Docker production with separate worker | Unset on server, optional `true` on worker | `false` or unset | Worker still needs explicit worker command |
| Kubernetes multi-pod | `true` on worker pods only | `false` or unset | Requires explicit worker entrypoint and Redis-backed distributed state |

## Related governance records

- [Deploy Truth Table](./deploy-truth-table.md)
- [Runtime Entrypoint Inventory](./entrypoint-inventory.md)
