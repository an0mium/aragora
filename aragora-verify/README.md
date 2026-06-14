# aragora-verify

**Verify an [Open Decision Receipt](https://github.com/synaptent/aragora/blob/main/docs/specs/OPEN_DECISION_RECEIPT.md) offline — no Aragora install, no server, no account.**

Action-level receipts (Microsoft AGT, SCITT, in-toto/SLSA) prove *what happened
and whether policy allowed it*. An **Open Decision Receipt (ODR)** proves the
layer above: *why it was decided, who adversarially examined it with what model
diversity, who dissented, how calibrated the confidence was, and whether an
accountable human accepted the risk.*

`aragora-verify` is the free, standalone tool that lets anyone — an auditor, a
customer, a skeptic — check such a receipt is genuine and well-formed:

- **Schema conformance** to the ODR v0.1 content profile.
- **Canonical digest** — recomputes `SHA-256(JCS(receipt − signatures))` per
  RFC 8785, the value any detached signature covers.
- **Ed25519 signature** — verifies detached signatures with only the public key.
- **Quorum consistency** — every supporting/dissenting agent is a disclosed
  participant (a mismatch is a tamper/malformed signal).
- **Hash-chain linkage** — when a chain is supplied, the receipt is anchored in
  it and the links are continuous.

It depends only on the Python standard library plus `cryptography`.

## Install

```bash
pip install aragora-verify
```

## Use

```bash
# Structural + canonical-digest check
aragora-verify receipt.odr.json

# Full authenticity check against the issuer's published public key
aragora-verify receipt.odr.json --pubkey aragora-odr-signing-key.pem

# Also confirm the receipt is anchored in a hash chain
aragora-verify receipt.odr.json --pubkey key.pem --chain intent-chain.jsonl

# Machine-readable result
aragora-verify receipt.odr.json --pubkey key.pem --json
```

Exit code `0` means verified (no failed checks); `1` means a check failed;
`2` is a usage/input error.

The public key for receipts emitted by an Aragora deployment is published at
`GET /.well-known/aragora-odr-signing-key` and `GET /api/v2/receipts/signing-key`.

### Weakening vs. failing

Absent markers (`{"status": "absent", ...}`) and `"undisclosed"` model families
are **honesty signals** — a receipt full of them is visibly weak, not a
strong-looking fabrication. They are reported as *weakening signals* and do
**not** fail verification; the policy thresholds (e.g. "require ≥2 model
families", "require human attestation") are yours to apply on top.

## Library

```python
from aragora_verify import verify, load_public_key

result = verify(receipt_dict, public_key=load_public_key(pem_bytes))
print(result.ok, result.odr_digest)
for check in result.checks:
    print(check.name, check.status, check.detail)
```

## What this is part of

ODR-3 of the [Open Decision Receipt epic](https://github.com/synaptent/aragora/issues/8223).
The verifier is free and standalone by design — the *emitter* (adversarial
debate + signed decision receipts) is the product. See the
[content-profile spec](https://github.com/synaptent/aragora/blob/main/docs/specs/OPEN_DECISION_RECEIPT.md).

## License

MIT
