# Proof Generation (Z3/Lean)

Formal verification backends for machine-verified proofs of debate claims.

## Overview

Aragora can attempt to formally verify mathematical and logical claims using:
- **Z3 SMT Solver**: For decidable fragments (arithmetic, logic, constraints)
- **Lean 4**: For complex mathematical proofs with LLM-assisted translation

## Important Limitations

- LLM translations may produce valid proofs that don't match the original claim
- Always check `is_high_confidence` for reliable results
- For critical decisions, manually review the `formal_statement`
- Semantic verification helps but doesn't guarantee correctness

## Usage

### Quick Start

```python
from aragora.verification.formal import get_formal_verification_manager

manager = get_formal_verification_manager()

result = await manager.attempt_formal_verification(
    claim="For all n, n + 0 = n",
    claim_type="MATHEMATICAL",
    timeout_seconds=60.0
)

if result.is_high_confidence:
    print("Verified with high confidence!")
elif result.is_verified:
    print(f"Proof compiles but: {result.confidence_warning}")
else:
    print(f"Verification failed: {result.error_message}")
```

### Using Specific Backends

```python
from aragora.verification.formal import LeanBackend, Z3Backend

# Lean 4 for mathematical proofs
lean = LeanBackend()
if lean.is_available:
    result = await lean.verify_claim(
        "All prime numbers greater than 2 are odd",
        verify_semantic_match=True
    )

# Z3 for decidable logic
z3 = Z3Backend()
if z3.is_available:
    result = await z3.prove("""
        (declare-const x Int)
        (declare-const y Int)
        (assert (not (=> (> x y) (> (+ x 1) y))))
        (check-sat)
    """)
```

## Proof Status

```python
class FormalProofStatus(Enum):
    NOT_ATTEMPTED = "not_attempted"
    TRANSLATION_FAILED = "translation_failed"  # Couldn't translate
    PROOF_FOUND = "proof_found"                # Success
    PROOF_FAILED = "proof_failed"              # Theorem false/unprovable
    TIMEOUT = "timeout"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    NOT_SUPPORTED = "not_supported"            # Claim type unsuitable
```

## Z3 Backend

SMT solver for decidable fragments.

### Supported Claim Types

- Linear/nonlinear arithmetic
- Boolean satisfiability
- Bitvector operations
- Array theory
- Quantifier-free theories

### Claim Detection

```python
z3 = Z3Backend()

# These return True:
z3.can_verify("If X > Y and Y > Z, then X > Z")
z3.can_verify("This value is in range [0, 100]")
z3.can_verify("(assert (> x 0))", claim_type="constraint")

# Already in SMT-LIB2 format:
z3.can_verify("(declare-const x Int) (assert (> x 0)) (check-sat)")
```

### SMT-LIB2 Format

For validity checking, assert the **negation**:

```lisp
; Prove: for all x, y: x > y implies x + 1 > y
(declare-const x Int)
(declare-const y Int)
(assert (not (=> (> x y) (> (+ x 1) y))))  ; Negate the claim
(check-sat)
; unsat = original claim is valid
; sat = counterexample found
```

### Result Interpretation

| Z3 Result | Meaning |
|-----------|---------|
| `unsat` | Negation unsatisfiable → claim VALID |
| `sat` | Counterexample found → claim FALSE |
| `unknown` | Timeout or undecidable |

### Caching

```python
z3 = Z3Backend(
    cache_size=100,          # Max cached results
    cache_ttl_seconds=3600   # 1 hour TTL
)

# Clear cache
cleared = z3.clear_cache()
```

## Lean 4 Backend

For complex mathematical proofs with LLM-assisted translation.

### Prerequisites

- Lean 4 toolchain installed (`lean`, `lake`)
- API key for translation (Anthropic, OpenAI, or OpenRouter)

### Claim Detection

```python
lean = LeanBackend()

# Math patterns detected:
lean.can_verify("For all n in naturals, n + 0 = n")
lean.can_verify("There exists a prime between n and 2n")
lean.can_verify("∀ x ∈ ℕ, x ≥ 0")

# Explicit claim types:
lean.can_verify("...", claim_type="MATHEMATICAL")
lean.can_verify("...", claim_type="THEOREM")
```

### Translation Models

```python
from aragora.verification.formal import TranslationModel

lean = LeanBackend(translation_model=TranslationModel.AUTO)

# Options:
# - AUTO: DeepSeek-Prover first, then Claude/GPT-4
# - DEEPSEEK_PROVER: Best for math proofs
# - CLAUDE: General purpose
# - OPENAI: GPT-4 fallback
```

### Complete Verification Pipeline

```python
result = await lean.verify_claim(
    claim="All primes > 2 are odd",
    context="Number theory context",
    verify_semantic_match=True  # Check theorem matches claim
)

print(f"Status: {result.status.value}")
print(f"Translation confidence: {result.translation_confidence:.0%}")
print(f"Semantic match verified: {result.semantic_match_verified}")
print(f"High confidence: {result.is_high_confidence}")

if result.confidence_warning:
    print(f"Warning: {result.confidence_warning}")
```

### Semantic Verification

LLM checks if the proven theorem actually proves the claim:

```python
matches, confidence, explanation = await lean.verify_semantic_match(
    original_claim="All primes > 2 are odd",
    formal_statement="theorem claim_1 : 1 = 1 := rfl"  # Wrong theorem!
)
# matches=False, confidence=0.9, explanation="Theorem proves 1=1, not the claim"
```

## FormalProofResult

```python
@dataclass
class FormalProofResult:
    status: FormalProofStatus
    language: FormalLanguage

    formal_statement: str | None    # Translated code
    proof_text: str | None          # Proof (or counterexample)
    proof_hash: str | None          # For caching

    translation_time_ms: float
    proof_search_time_ms: float
    error_message: str
    prover_version: str

    # Confidence tracking
    translation_confidence: float   # 0.0-1.0
    original_claim: str
    semantic_match_verified: bool
    confidence_warning: str

    @property
    def is_verified(self) -> bool:
        """Proof compiles (may not match claim)."""

    @property
    def is_high_confidence(self) -> bool:
        """Verified AND semantically matches AND confidence >= 0.8."""
```

## Manager API

```python
manager = get_formal_verification_manager()

# Get available backends
backends = manager.get_available_backends()

# Find best backend for claim
backend = manager.get_backend_for_claim(
    "All primes > 2 are odd",
    claim_type="MATHEMATICAL"
)

# Status report
status = manager.status_report()
# {
#     "backends": [
#         {"language": "z3_smt", "available": True},
#         {"language": "lean4", "available": False}
#     ],
#     "any_available": True
# }
```

## Backend Selection

The manager tries backends in order:

1. **Z3** (first) - Simpler, faster for decidable claims
2. **Lean 4** (fallback) - For complex mathematical proofs

## Sandbox Execution

Lean code runs in a sandboxed subprocess:

```python
lean = LeanBackend(
    sandbox_timeout=60.0,      # Max execution time
    sandbox_memory_mb=1024     # Memory limit
)
```

## Example: Full Workflow

```python
from aragora.verification.formal import (
    get_formal_verification_manager,
    FormalProofStatus
)

async def verify_debate_claim(claim: str, claim_type: str):
    manager = get_formal_verification_manager()

    # Check if we can verify
    backend = manager.get_backend_for_claim(claim, claim_type)
    if not backend:
        return {"verifiable": False, "reason": "No suitable backend"}

    # Attempt verification
    result = await manager.attempt_formal_verification(
        claim=claim,
        claim_type=claim_type,
        timeout_seconds=60.0
    )

    response = {
        "verifiable": True,
        "backend": result.language.value,
        "status": result.status.value,
    }

    if result.is_high_confidence:
        response["verdict"] = "PROVEN"
        response["proof_hash"] = result.proof_hash
    elif result.status == FormalProofStatus.PROOF_FAILED:
        response["verdict"] = "DISPROVEN"
        if result.proof_text:
            response["counterexample"] = result.proof_text[:200]
    elif result.is_verified:
        response["verdict"] = "PROOF_COMPILES"
        response["warning"] = result.confidence_warning
    else:
        response["verdict"] = "UNVERIFIED"
        response["error"] = result.error_message

    return response
```

## Implementation Files

| Component | Source |
|-----------|--------|
| Main Module | `aragora/verification/formal.py` |
| Sandbox | `aragora/verification/sandbox.py` |
| DeepSeek Prover | `aragora/verification/deepseek_prover.py` |

## Related Documentation

- [Belief Network](./BELIEF_NETWORK.md) - Verified claims boost belief confidence
- [Consensus Mechanism](./CONSENSUS.md) - Formal proofs strengthen consensus
