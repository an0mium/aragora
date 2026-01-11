"""
Verification module for executable and formal proofs.

Provides:
- VerificationProof: Executable code that verifies claims
- ProofExecutor: Safe execution environment for proofs
- VerificationResult: Outcome of proof execution
- ClaimVerifier: Links claims to their verification proofs
- FormalVerificationBackend: Interface for theorem provers (Lean, Z3)
"""

from aragora.verification.proofs import (
    VerificationProof,
    ProofType,
    ProofStatus,
    VerificationResult,
    ProofExecutor,
    ClaimVerifier,
    VerificationReport,
    ProofBuilder,
)
from aragora.verification.formal import (
    FormalVerificationBackend,
    FormalVerificationManager,
    FormalProofResult,
    FormalProofStatus,
    FormalLanguage,
    LeanBackend,
    Z3Backend,
    get_formal_verification_manager,
)
from aragora.verification.sandbox import (
    ProofSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    run_sandboxed,
)

__all__ = [
    # Executable proofs
    "VerificationProof",
    "ProofType",
    "ProofStatus",
    "VerificationResult",
    "ProofExecutor",
    "ClaimVerifier",
    "VerificationReport",
    "ProofBuilder",
    # Formal verification (stub interface)
    "FormalVerificationBackend",
    "FormalVerificationManager",
    "FormalProofResult",
    "FormalProofStatus",
    "FormalLanguage",
    "LeanBackend",
    "Z3Backend",
    "get_formal_verification_manager",
    # Sandbox execution
    "ProofSandbox",
    "SandboxConfig",
    "SandboxResult",
    "SandboxStatus",
    "run_sandboxed",
]
