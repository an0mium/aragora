# Verification - Formal and Executable Proof System

Formal mathematical verification (Lean 4, Z3) and sandboxed executable proofs for validating debate claims with cryptographic-grade confidence.

## Quick Start

```python
# Formal verification (Lean 4)
from aragora.verification import LeanBackend

backend = LeanBackend()
result = await backend.verify_claim(
    "For all natural numbers n, n + 0 = n",
    verify_semantic_match=True
)
if result.is_high_confidence:
    print("High-confidence proof verified")

# Executable proof
from aragora.verification import ProofExecutor, ProofBuilder

proof = ProofBuilder("claim-1").assertion(
    "Array length is non-negative",
    "arr = [1,2,3]",
    "len(arr) >= 0"
)
executor = ProofExecutor()
result = await executor.execute(proof)
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `LeanBackend` | `formal.py` | Lean 4 theorem prover |
| `Z3Backend` | `formal.py` | SMT solver for decidable logic |
| `FormalVerificationManager` | `formal.py` | Coordinates backends |
| `ProofExecutor` | `proofs.py` | Sandboxed proof execution |
| `ClaimVerifier` | `proofs.py` | Manages proof-claim relationships |
| `ProofSandbox` | `sandbox.py` | Subprocess isolation |
| `DeepSeekProverTranslator` | `deepseek_prover.py` | NL-to-Lean translation |

## Architecture

```
verification/
├── __init__.py              # Public API exports
├── formal.py                # Formal verification backends
├── proofs.py                # Executable verification proofs
├── sandbox.py               # Sandboxed execution environment
└── deepseek_prover.py       # DeepSeek-Prover integration
```

## Formal Verification

### Lean 4 Backend

```python
from aragora.verification import LeanBackend, FormalProofResult

backend = LeanBackend()
result: FormalProofResult = await backend.verify_claim(
    claim="All prime numbers greater than 2 are odd",
    verify_semantic_match=True  # Guards against hallucination
)

# Result properties
result.status          # PROOF_FOUND, PROOF_FAILED, TIMEOUT, etc.
result.proof_text      # Generated Lean 4 code
result.confidence      # Translation confidence (0-1)
result.semantic_match_verified  # True if theorem matches claim
result.is_high_confidence  # True if proof_found + confidence >= 0.8 + semantic verified
```

**Supported Claim Types:**
- Mathematical (primes, sums, integrals)
- Logical (quantifiers, implications)
- Arithmetic (divisibility, parity)
- Theorems and lemmas

### Z3 Backend (SMT Solver)

```python
from aragora.verification import Z3Backend

backend = Z3Backend()
result = await backend.prove(
    claim="x > 0 implies x + 1 > 1",
    timeout=30
)
```

**Best For:**
- Arithmetic constraints
- Boolean satisfiability
- Bitvector operations
- Array theory

### Translation Pipeline

```
Natural Language → LLM Translation → Formal Statement → Proof Search → Semantic Verification
                   (DeepSeek-Prover)    (Lean 4/Z3)       (Type checker)    (Guards hallucination)
```

## Executable Proofs

### Proof Types

```python
from aragora.verification import ProofType

ProofType.ASSERTION      # Boolean expression evaluation
ProofType.CODE_EXECUTION # Run code and capture output
ProofType.COMPUTATION    # Mathematical verification
ProofType.TEST_SUITE     # Run test suite
ProofType.PROPERTY_CHECK # Property-based testing
ProofType.API_CALL       # Data fetching
ProofType.STATIC_ANALYSIS # Code analysis
ProofType.MANUAL         # Requires human verification
```

### Creating Proofs

```python
from aragora.verification import ProofBuilder, VerificationProof

# Using builder
proof = ProofBuilder("claim-id").assertion(
    description="List sorting is stable",
    setup="items = [(1, 'a'), (1, 'b')]",
    assertion="sorted(items, key=lambda x: x[0]) == [(1, 'a'), (1, 'b')]"
)

# Direct creation
proof = VerificationProof(
    id="proof-123",
    claim_id="claim-456",
    proof_type=ProofType.COMPUTATION,
    description="Verify factorial correctness",
    code="import math; result = math.factorial(5)",
    expected_output="120"
)
```

### Executing Proofs

```python
from aragora.verification import ProofExecutor

executor = ProofExecutor(
    allow_network=False,
    allow_filesystem=False
)

result = await executor.execute(proof)

result.status      # PASSED, FAILED, ERROR, TIMEOUT
result.output      # Execution output
result.duration_ms # Execution time
result.error       # Error message if failed
```

## Sandboxed Execution

```python
from aragora.verification import ProofSandbox, SandboxConfig

config = SandboxConfig(
    timeout_seconds=5,
    max_memory_mb=256,
    allow_network=False,
    allow_filesystem=False
)

sandbox = ProofSandbox(config)

# Execute Lean code
result = await sandbox.execute_lean(lean_code)

# Execute Z3 code
result = await sandbox.execute_z3(smt_code)

# Execute Python code (heavily sandboxed)
result = await sandbox.execute(python_code)
```

**Security Features:**
- Subprocess isolation with hard timeout
- 50+ safe builtins whitelist
- 25+ dangerous patterns blocked (exec, eval, __import__, etc.)
- Environment sanitization (API keys removed)
- Resource limits (memory, CPU, file descriptors)

## Claim Verification

```python
from aragora.verification import ClaimVerifier, VerificationReport

verifier = ClaimVerifier()

# Add proofs for claims
verifier.add_proof(proof1)
verifier.add_proof(proof2)

# Verify specific claim
results = await verifier.verify_claim("claim-123")

# Verify all claims
all_results = await verifier.verify_all()

# Generate report
report = VerificationReport(
    debate_id="debate-456",
    results=all_results,
    claims=claims
)

print(f"Verification rate: {report.verification_rate()}")
print(f"Pass rate: {report.pass_rate()}")
print(report.generate_summary())  # Markdown summary
```

## Proof Status Flow

```
PENDING → RUNNING → PASSED
              ↓
        FAILED / ERROR / TIMEOUT / SKIPPED
```

## Formal Proof Status

```python
from aragora.verification import FormalProofStatus

FormalProofStatus.NOT_ATTEMPTED      # Not tried yet
FormalProofStatus.TRANSLATION_FAILED # NL→Formal failed
FormalProofStatus.PROOF_FOUND        # Successfully proved
FormalProofStatus.PROOF_FAILED       # Could not prove
FormalProofStatus.TIMEOUT            # Exceeded time limit
FormalProofStatus.BACKEND_UNAVAILABLE # Lean/Z3 not installed
FormalProofStatus.NOT_SUPPORTED      # Claim type not supported
```

## DeepSeek-Prover Integration

```python
from aragora.verification import DeepSeekProverTranslator

translator = DeepSeekProverTranslator()

# Single translation
result = await translator.translate(
    "The sum of first n natural numbers is n(n+1)/2"
)
print(result.lean_code)
print(result.confidence)

# Batch translation
results = await translator.translate_batch(claims)
```

## Configuration

```python
# Execution timeout (default: 5 seconds)
EXEC_TIMEOUT_SECONDS = 5.0

# DeepSeek model
PRIMARY_MODEL = "deepseek/deepseek-prover-v2"
FALLBACK_MODEL = "deepseek/deepseek-chat"

# Z3 cache TTL (default: 1 hour)
CACHE_TTL = 3600

# Max output size (default: 10KB)
MAX_OUTPUT_SIZE = 10240
```

## Safety Considerations

1. **Hallucination Protection**: Semantic verification ensures theorem matches original claim
2. **Two-Layer Sandboxing**: Pattern detection + subprocess isolation
3. **Environment Sanitization**: API keys removed before execution
4. **Proof Caching**: SHA256-based caching for deterministic proofs
5. **Output Truncation**: Max 10KB captured per proof

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Explainability](../explainability/README.md) - Decision explanations
- [Reasoning](../reasoning/README.md) - Belief and claim tracking
- [Gauntlet](../gauntlet/README.md) - Adversarial validation
