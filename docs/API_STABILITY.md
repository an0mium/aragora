# API Stability

Aragora tags every OpenAPI operation with `x-aragora-stability` to signal how
safe it is to depend on an endpoint. This keeps the public SDK surface reliable
while allowing rapid iteration on newer features.

## Stability Levels

- `stable`: Covered by both TypeScript and Python SDKs. Backwards-compatible by default.
- `beta`: Supported but may change with advance notice.
- `experimental`: Available for early adopters; breaking changes may happen without notice.
- `internal`: Intended for internal services or admin operations only.
- `deprecated`: Still available but scheduled for removal.

## How Stability Is Assigned

Stability is derived from an SDK parity manifest generated from the current
OpenAPI spec and SDK coverage.

Manifest location:

- `aragora/server/openapi/stability_manifest.json`

Regenerate it after SDK updates:

```bash
python scripts/sdk_parity_audit.py \
  --stable-out aragora/server/openapi/stability_manifest.json \
  --stable-require both
```

If you need to promote or demote individual endpoints, update the manifest and
re-export the OpenAPI spec.

You can also add manual overrides:

- `beta`: endpoints to surface as beta
- `internal`: endpoints reserved for internal use

These overrides are preserved when regenerating the manifest.

## OpenAPI Integration

`aragora/server/openapi_impl.py` applies the stability marker to every operation
when generating `docs/api/openapi.json` and `docs/api/openapi.yaml`.
