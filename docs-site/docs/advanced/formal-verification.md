---
title: Formal Verification
description: Formal Verification
---

# Formal Verification

Aragora integrates formal verification backends for machine-verified proofs of claims made during debates. This enables a higher level of trust when agents make mathematical, logical, or constraint-based assertions.

## Table of Contents

- [Overview](#overview)
- [Supported Backends](#supported-backends)
- [Z3 Backend](#z3-backend)
- [Lean 4 Backend](#lean-4-backend)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Limitations](#limitations)

---

## Overview

Formal verification provides machine-checkable proofs for claims. When an agent asserts something like "for all n, n + 0 = n", the formal verification system can:

1. **Translate** the natural language claim to a formal statement
2. **Attempt proof** using theorem provers or SMT solvers
3. **Return verified result** with proof artifact

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 FormalVerificationManager                    │
├─────────────────────────────────────────────────────────────┤
│  - Coordinates multiple backends                             │
│  - Selects appropriate backend for claim type                │
│  - Returns FormalProofResult                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Z3Backend   │ │  LeanBackend  │ │  (Future)     │
│   (SMT-LIB2)  │ │   (Lean 4)    │ │  Coq/Isabelle │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## Supported Backends

| Backend | Language | Best For | Status |
|---------|----------|----------|--------|
| **Z3** | SMT-LIB2 | Arithmetic, constraints, decidable fragments | Implemented |
| **Lean 4** | Lean | Complex theorems, mathematical proofs | Implemented |
| Coq | Gallina | (Future) | Planned |
| Isabelle | Isar | (Future) | Planned |

---

## Z3 Backend

The Z3 backend uses the [Z3 SMT solver](https://github.com/Z3Prover/z3) for decidable verification.

### Capabilities

- Linear and nonlinear arithmetic
- Boolean satisfiability
- Bitvector operations
- Array theory
- Quantifier-free theories

### Claim Types

Z3 works best for:
- Assertions: "X > Y and Y > Z implies X > Z"
- Constraints: "These variables cannot all be true"
- Arithmetic: "The sum of 1 to n equals n(n+1)/2"
- Bounds checking: "This value is always in range [0, 100]"

### Installation

```bash
pip install z3-solver
```

### Example

```python
from aragora.verification import Z3Backend

backend = Z3Backend()

# Check if Z3 is available
print(backend.is_available)  # True if z3-solver installed

# Verify a transitivity claim
claim = "If X > Y and Y > Z, then X > Z"
if backend.can_verify(claim):
    # Translate to SMT-LIB2
    formal = await backend.translate(claim)
    # formal = """
    # (declare-const X Int)
    # (declare-const Y Int)
    # (declare-const Z Int)
    # (assert (not (=> (and (> X Y) (> Y Z)) (> X Z))))
    # (check-sat)
    # """

    # Attempt proof
    result = await backend.prove(formal)
    print(result.status)  # PROOF_FOUND (negation is unsat)
    print(result.is_verified)  # True
```

### SMT-LIB2 Format

Z3 uses SMT-LIB2 format. To verify a claim, we prove by contradiction:
1. Assert the **negation** of the claim
2. If Z3 returns `unsat`, the original claim is valid
3. If Z3 returns `sat`, a counterexample exists

```smt2
; Example: Prove transitivity of >
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

; Assert negation: NOT (x > y AND y > z IMPLIES x > z)
(assert (not (=> (and (> x y) (> y z)) (> x z))))

(check-sat)  ; Returns "unsat" = claim is valid
```

---

## Lean 4 Backend

The Lean 4 backend uses the [Lean theorem prover](https://leanprover.github.io/) with LLM-assisted translation.

### Capabilities

- Full dependent type theory
- Mathematical proofs
- Program verification
- Mathlib integration (when available)

### Claim Types

Lean works best for:
- Mathematical theorems: "All prime numbers greater than 2 are odd"
- Properties: "This function is associative"
- Invariants: "This data structure maintains sorted order"
- Complex proofs requiring human-like reasoning

### Installation

```bash
# Install Lean 4 via elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Verify installation
lean --version
```

### Example

```python
from aragora.verification import LeanBackend

backend = LeanBackend(sandbox_timeout=60.0, sandbox_memory_mb=1024)

# Check if Lean is available
print(backend.is_available)  # True if lean command found
print(backend.lean_version)  # e.g., "Lean 4.x.x"

# Verify a mathematical claim
claim = "For all natural numbers n, n + 0 = n"

if backend.can_verify(claim, claim_type="MATHEMATICAL"):
    # Translate using LLM (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
    formal = await backend.translate(claim)
    # formal = """
    # import Mathlib.Tactic
    # theorem claim_1 : ∀ n : Nat, n + 0 = n := by simp
    # """

    # Verify via Lean type checker
    result = await backend.prove(formal)
    print(result.status)  # PROOF_FOUND
    print(result.proof_hash)  # Hash for caching/verification
```

### Pattern Matching

The Lean backend recognizes mathematical patterns:
- Quantifiers: `for all`, `forall`, `exists`, `there exists`
- Logical: `iff`, `implies`, `if and only if`
- Mathematical: `prove`, `theorem`, `lemma`
- Arithmetic: `prime`, `divisible`, `even`, `odd`
- Unicode: `∀∃→←↔∧∨¬⊢⊨≡≠≤≥∈∉⊂⊃∩∪∅ℕℤℚℝℂ`

### Sandboxed Execution

Lean code runs in a sandboxed subprocess with:
- Hard timeout enforcement
- Memory limits (default 1024 MB)
- CPU time limits
- File descriptor limits
- Process/thread limits

See [ProofSandbox](#proofsandbox) for details.

---

## Usage

### Using FormalVerificationManager

The manager automatically selects the best backend for each claim:

```python
from aragora.verification import get_formal_verification_manager

manager = get_formal_verification_manager()

# Check available backends
status = manager.status_report()
print(status)
# {
#   "backends": [
#     {"language": "z3_smt", "available": True},
#     {"language": "lean4", "available": True}
#   ],
#   "any_available": True
# }

# Verify a claim (auto-selects backend)
result = await manager.attempt_formal_verification(
    claim="X > Y and Y > Z implies X > Z",
    claim_type="LOGICAL",
    timeout_seconds=30.0,
)

if result.is_verified:
    print("Claim verified!")
    print(f"Proof hash: {result.proof_hash}")
    print(f"Backend: {result.language.value}")
    print(f"Time: {result.proof_search_time_ms:.1f}ms")
else:
    print(f"Verification failed: {result.error_message}")
```

### In Debates

Formal verification integrates with the debate system:

```python
from aragora import Arena, DebateProtocol

protocol = DebateProtocol(
    rounds=3,
    enable_formal_verification=True,  # Enable for mathematical claims
)

# During debate, claims can be annotated:
# Agent: "I claim that for all n, n + 0 = n [MATHEMATICAL]"
# System: Automatically verifies and adds proof to claim metadata
```

---

## API Reference

### FormalProofStatus

```python
class FormalProofStatus(Enum):
    NOT_ATTEMPTED = "not_attempted"
    TRANSLATION_FAILED = "translation_failed"
    PROOF_FOUND = "proof_found"
    PROOF_FAILED = "proof_failed"
    TIMEOUT = "timeout"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    NOT_SUPPORTED = "not_supported"
```

### FormalProofResult

```python
@dataclass
class FormalProofResult:
    status: FormalProofStatus
    language: FormalLanguage
    formal_statement: Optional[str] = None
    proof_text: Optional[str] = None
    proof_hash: Optional[str] = None
    translation_time_ms: float = 0.0
    proof_search_time_ms: float = 0.0
    error_message: str = ""
    prover_version: str = ""
    timestamp: datetime

    @property
    def is_verified(self) -> bool:
        return self.status == FormalProofStatus.PROOF_FOUND
```

### FormalLanguage

```python
class FormalLanguage(Enum):
    LEAN4 = "lean4"
    COQ = "coq"
    ISABELLE = "isabelle"
    AGDA = "agda"
    Z3_SMT = "z3_smt"
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | For LLM translation (Lean) | None |
| `OPENAI_API_KEY` | Alternative for LLM translation | None |

### Backend Options

**Z3Backend**:
```python
Z3Backend(llm_translator=None)  # Optional custom translator
```

**LeanBackend**:
```python
LeanBackend(
    sandbox_timeout=60.0,    # Max seconds for Lean execution
    sandbox_memory_mb=1024,  # Memory limit in MB
)
```

---

## ProofSandbox

All formal verification runs in a sandboxed environment:

```python
from aragora.verification import ProofSandbox, run_sandboxed

# Quick verification
result = await run_sandboxed(
    code="theorem t : 1 + 1 = 2 := rfl",
    language="lean",
    timeout=30.0,
    memory_mb=512,
)

# Or use sandbox directly
sandbox = ProofSandbox(timeout=30.0, memory_mb=512)
result = await sandbox.execute_lean(lean_code)
result = await sandbox.execute_z3(smtlib_code)
```

### Security Features

- **Subprocess isolation**: Code runs in separate process
- **Resource limits**: Memory, CPU, file descriptors, processes
- **Timeout enforcement**: Hard kill after timeout
- **Temporary directory cleanup**: No persistent artifacts
- **Restricted PATH**: Only essential directories
- **Network disabled**: `no_proxy=*` environment

### SandboxResult

```python
@dataclass
class SandboxResult:
    status: SandboxStatus  # SUCCESS, TIMEOUT, MEMORY_LIMIT, etc.
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    memory_used_mb: float
    error_message: str

    @property
    def is_success(self) -> bool:
        return self.status == SandboxStatus.SUCCESS and self.exit_code == 0
```

---

## Limitations

### Current Limitations

1. **LLM Translation**: Translation quality depends on LLM capability
2. **Complex Proofs**: Very complex proofs may timeout
3. **Mathlib**: Full Mathlib integration requires project setup
4. **Network Provers**: Online proof services not yet supported

### Not Suitable For

- Claims requiring external data/APIs
- Non-formalizable statements ("This code is elegant")
- Claims requiring probabilistic reasoning
- Statements about AI behavior or emergent properties

### Future Enhancements

- Coq and Isabelle backends
- Proof caching across sessions
- Parallel proof search
- Mathlib project templates
- Integration with ProvenanceManager

---

## See Also

- [Verification Module](aragora/verification/) - Source code
- [Sandbox Security](./SECURITY.md#proof-sandbox) - Security details
- [WebSocket Events](./WEBSOCKET_EVENTS.md#verification-events) - Streaming events
