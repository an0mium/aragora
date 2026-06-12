# ODR Signing Keys

Ed25519 keys for Open Decision Receipt detached signatures
([spec](../specs/OPEN_DECISION_RECEIPT.md) §6, issue #8225).

## What lives where

| Artifact | Location | Notes |
|---|---|---|
| Private key (32-byte seed) | AWS Secrets Manager, secret `ARAGORA_ODR_SIGNING_KEY` (base64 or hex) | **Never** in the repo, never in `.env`, never exported as a raw shell variable. Loaded through `aragora.config.secrets.get_secret`. |
| Public key | This directory (`odr-ed25519-<key_id>.pub`, base64 raw 32 bytes) and the live endpoint | Safe to publish anywhere. |
| Live endpoint | `GET /api/v2/receipts/signing-key` and `GET /.well-known/aragora-odr-signing-key` | Serves `key_id` + base64 public key derived from the configured private key. |

The `key_id` is derived from the public key
(`ed25519-` + first 16 hex chars of SHA-256 of the raw public key), so a
verifier can match a receipt's `signatures[].key_id` against published keys
without any registry.

## Provisioning (operator action)

```bash
python - <<'EOF'
from aragora.gauntlet.odr_signing import (
    export_private_seed_b64,
    generate_odr_keypair,
    public_key_b64,
)

signer = generate_odr_keypair()
print("key_id:    ", signer.key_id)
print("public_key:", public_key_b64(signer))
print("seed (store in Secrets Manager, then discard):", export_private_seed_b64(signer))
EOF
```

1. Store the seed in AWS Secrets Manager under `ARAGORA_ODR_SIGNING_KEY`
   (in the `aragora/production` secret bundle).
2. Commit the public key here as `odr-ed25519-<key_id>.pub` (one line, base64).
3. Discard the local copy of the seed.

No public key file in this directory means no production ODR signing key has
been provisioned yet — receipts are emitted with `signatures: []`.

## Rotation

Follows the `aragora/security/key_rotation.py` philosophy: overlap, don't
cut over.

1. Generate the new keypair; store the new seed under
   `ARAGORA_ODR_SIGNING_KEY` (the old seed moves to
   `ARAGORA_ODR_SIGNING_KEY_PREVIOUS` for the overlap window).
2. Commit the new public key file here; **keep** the old `.pub` file —
   receipts signed with the old key must stay verifiable forever. Public
   keys are append-only in this directory.
3. New receipts sign with the new key automatically (`key_id` changes).
4. Verifiers select the key by `signatures[].key_id`; old receipts verify
   against the old published key.

Revocation (suspected key compromise) is a different event from rotation:
add a `REVOKED-<key_id>` marker file with the incident reference, and treat
signatures from that key as untrusted after the compromise window. The
receipts themselves remain tamper-evident via the content digest and any
external anchoring (Rekor, #8231).
