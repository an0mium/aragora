# Changelog

All notable changes to `aragora-verify` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic
versioning.

## [0.1.0] — unreleased

### Added
- Initial release: standalone offline verifier for Open Decision Receipts (ODR v0.1).
- `aragora-verify <receipt.json> [--pubkey KEY] [--chain JSONL] [--json]` CLI.
- Library API: `verify`, `load_public_key`, `compute_key_id`, `validate_structure`,
  `jcs_canonicalize`, `odr_content_digest`.
- Checks: ODR v0.1 schema conformance (stdlib structural validator, with optional
  `jsonschema` rigor), RFC 8785 (JCS) canonical digest recomputation, Ed25519
  detached-signature verification, quorum participant consistency, and hash-chain
  linkage/anchoring.
- Absent markers and `"undisclosed"` model families surfaced as non-failing
  weakening signals.
- Dependencies: Python standard library plus `cryptography`; `jsonschema` optional.
