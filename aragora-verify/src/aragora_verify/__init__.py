"""aragora-verify — standalone offline verifier for Open Decision Receipts.

Validate an ODR v0.1 receipt with nothing but the receipt JSON (and optionally
a public key and a hash chain): schema conformance, RFC 8785 (JCS) canonical
digest, Ed25519 detached signature, quorum consistency, and hash-chain linkage.

No Aragora install, no server, no account. The emitter is the product; the
verifier is free.

    from aragora_verify import verify, load_public_key
    result = verify(receipt_dict, public_key=load_public_key(pem_bytes))
    assert result.ok
"""

from __future__ import annotations

from .jcs import jcs_canonicalize, odr_content_digest
from .schema import ODR_PROFILE_URI, ODR_VERSION, validate_structure
from .verifier import (
    Check,
    VerificationError,
    VerifyResult,
    compute_key_id,
    load_public_key,
    verify,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "verify",
    "load_public_key",
    "compute_key_id",
    "validate_structure",
    "jcs_canonicalize",
    "odr_content_digest",
    "Check",
    "VerifyResult",
    "VerificationError",
    "ODR_VERSION",
    "ODR_PROFILE_URI",
]
