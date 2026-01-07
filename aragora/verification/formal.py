"""
Formal Verification Backends - Interface for theorem provers.

Provides a protocol for integrating formal proof assistants like Lean, Coq,
or Isabelle. Currently a stub interface to future-proof the architecture
for when LLM-to-Lean translation matures.

Status: Interface defined, implementation pending (estimated 2025-2026)

Rationale (from aragora self-debate):
- LLM-to-Lean tools improving rapidly (DeepSeek-Prover-V2, LeanDojo)
- Trust differentiation: "machine-verified proof" is valuable signaling
- Minimal integration cost now, significant optionality later
- Can connect with ProvenanceManager for "verified provenance"
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class FormalProofStatus(Enum):
    """Status of a formal proof attempt."""
    NOT_ATTEMPTED = "not_attempted"
    TRANSLATION_FAILED = "translation_failed"  # Couldn't translate to formal language
    PROOF_FOUND = "proof_found"
    PROOF_FAILED = "proof_failed"  # Theorem false or unprovable
    TIMEOUT = "timeout"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    NOT_SUPPORTED = "not_supported"  # Claim type not suitable for formal proof


class FormalLanguage(Enum):
    """Supported formal proof languages."""
    LEAN4 = "lean4"
    COQ = "coq"
    ISABELLE = "isabelle"
    AGDA = "agda"
    Z3_SMT = "z3_smt"  # SMT solver (simpler, more practical)


@dataclass
class FormalProofResult:
    """Result of a formal verification attempt."""
    status: FormalProofStatus
    language: FormalLanguage

    # The formal statement (if translation succeeded)
    formal_statement: Optional[str] = None

    # The proof (if found)
    proof_text: Optional[str] = None
    proof_hash: Optional[str] = None  # For verification/caching

    # Timing
    translation_time_ms: float = 0.0
    proof_search_time_ms: float = 0.0

    # Error info
    error_message: str = ""

    # Metadata
    prover_version: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_verified(self) -> bool:
        return self.status == FormalProofStatus.PROOF_FOUND

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "language": self.language.value,
            "formal_statement": self.formal_statement,
            "proof_hash": self.proof_hash,
            "is_verified": self.is_verified,
            "translation_time_ms": self.translation_time_ms,
            "proof_search_time_ms": self.proof_search_time_ms,
            "error_message": self.error_message,
            "prover_version": self.prover_version,
            "timestamp": self.timestamp.isoformat(),
        }


@runtime_checkable
class FormalVerificationBackend(Protocol):
    """
    Protocol for formal proof backends.

    Implementations should handle:
    1. Translation: natural language claim → formal theorem
    2. Proof search: attempt to prove the theorem
    3. Verification: check that a proof is valid

    Example future implementations:
    - LeanBackend: Lean 4 with LLM-assisted translation
    - Z3Backend: SMT solver for decidable fragments
    - MathlibLookup: Check if claim exists in Mathlib
    """

    @property
    def language(self) -> FormalLanguage:
        """The formal language this backend uses."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the backend is available (toolchain installed, etc.)."""
        ...

    def can_verify(self, claim: str, claim_type: Optional[str] = None) -> bool:
        """
        Check if this backend can attempt to verify the claim.

        Args:
            claim: Natural language claim
            claim_type: Optional hint about claim type (math, logic, protocol, etc.)

        Returns:
            True if the backend should attempt verification
        """
        ...

    async def translate(self, claim: str, context: str = "") -> Optional[str]:
        """
        Translate a natural language claim to a formal statement.

        Args:
            claim: Natural language claim
            context: Additional context to help translation

        Returns:
            Formal statement in the backend's language, or None if translation fails
        """
        ...

    async def prove(self, formal_statement: str, timeout_seconds: float = 60.0) -> FormalProofResult:
        """
        Attempt to prove a formal statement.

        Args:
            formal_statement: Statement in the backend's formal language
            timeout_seconds: Maximum time for proof search

        Returns:
            FormalProofResult with status and proof if found
        """
        ...

    async def verify_proof(self, formal_statement: str, proof: str) -> bool:
        """
        Verify that a proof is valid for a statement.

        Args:
            formal_statement: The theorem to verify
            proof: The purported proof

        Returns:
            True if the proof is valid
        """
        ...


class LeanBackend:
    """
    Lean 4 formal verification backend.

    Status: STUB - Not yet implemented.

    Future implementation will:
    - Use LLM-assisted translation (DeepSeek-Prover, Lean Copilot)
    - Shell out to `lake build` for proof checking
    - Query Mathlib for existing theorems
    - Cache successful proofs

    Prerequisites for implementation:
    - Lean 4 toolchain installed
    - Mathlib available
    - LLM with Lean training (Claude, DeepSeek-Prover)
    """

    @property
    def language(self) -> FormalLanguage:
        return FormalLanguage.LEAN4

    @property
    def is_available(self) -> bool:
        """Check if Lean 4 toolchain (lean and lake) is available."""
        import shutil
        return shutil.which("lean") is not None and shutil.which("lake") is not None

    def can_verify(self, claim: str, claim_type: Optional[str] = None) -> bool:
        # Stub: Not yet implemented
        return False

    async def translate(self, claim: str, context: str = "") -> Optional[str]:
        """
        Use LLM to translate a natural language claim to a Lean 4 theorem.

        Returns None if translation fails or no LLM is available.
        Example output: "theorem claim_123 : ∀ n : ℕ, n + 0 = n := by simp"
        """
        import os
        import aiohttp

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        # Create translation prompt
        prompt = f"""Translate the following natural language claim into a Lean 4 theorem statement.
If the claim cannot be expressed as a formal theorem, return "UNTRANSLATABLE".

Claim: {claim}
{f'Context: {context}' if context else ''}

Guidelines:
- Use valid Lean 4 syntax
- Include necessary imports if needed
- Use "sorry" as a placeholder proof
- Keep the theorem name simple (e.g., claim_1, main_theorem)

Return ONLY the Lean 4 code, no explanations. Example:
theorem claim_1 : ∀ n : Nat, n + 0 = n := by simp"""

        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            openai_key = os.environ.get("OPENAI_API_KEY")

            if anthropic_key:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                }
            elif openai_key:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "gpt-4o",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                # No API key available for LLM translation
                return None

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        logger.warning(f"LLM API returned status {response.status} for Lean translation")
                        return None

                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        logger.warning(f"Failed to parse LLM API response as JSON: {e}")
                        return None

                    try:
                        if os.environ.get("ANTHROPIC_API_KEY"):
                            result = data["content"][0]["text"].strip()
                        else:
                            result = data["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError, TypeError) as e:
                        logger.warning(f"Unexpected LLM response format for Lean translation: {e}")
                        return None

                    if "UNTRANSLATABLE" in result:
                        return None
                    return result

        except aiohttp.ClientError as e:
            logger.warning(f"Network error translating claim to Lean 4: {e}")
            return None
        except asyncio.TimeoutError:
            logger.warning("Timeout translating claim to Lean 4")
            return None
        except Exception as e:
            logger.warning(f"Failed to translate claim to Lean 4: {type(e).__name__}: {e}")
            return None

    async def prove(self, formal_statement: str, timeout_seconds: float = 60.0) -> FormalProofResult:
        return FormalProofResult(
            status=FormalProofStatus.BACKEND_UNAVAILABLE,
            language=FormalLanguage.LEAN4,
            error_message="Lean backend not yet implemented. See aragora/verification/formal.py",
        )

    async def verify_proof(self, formal_statement: str, proof: str) -> bool:
        return False


class Z3Backend:
    """
    Z3 SMT solver backend for decidable verification.

    Handles decidable fragments of first-order logic:
    - Linear/nonlinear arithmetic
    - Boolean satisfiability
    - Bitvector operations
    - Array theory
    - Quantifier-free theories

    Good for claims like:
    - "If X > Y and Y > Z, then X > Z"
    - "This function returns a value in range [0, 100]"
    - "These constraints have no solution"

    Translation strategy:
    1. If claim is already SMT-LIB2 format, use directly
    2. If LLM translator is provided, use it for natural language
    3. Fall back to pattern-based translation for simple claims
    """

    def __init__(self, llm_translator: Optional[Any] = None):
        """
        Initialize Z3 backend.

        Args:
            llm_translator: Optional async callable(claim, context) -> str
                           for LLM-assisted translation to SMT-LIB2
        """
        self._llm_translator = llm_translator
        self._z3_version: Optional[str] = None

    @property
    def language(self) -> FormalLanguage:
        return FormalLanguage.Z3_SMT

    @property
    def is_available(self) -> bool:
        """Check if z3 Python package is installed."""
        try:
            import z3
            self._z3_version = f"z3-{z3.get_version_string()}"
            return True
        except ImportError:
            return False

    @property
    def z3_version(self) -> str:
        """Get Z3 version string."""
        if self._z3_version is None and self.is_available:
            pass  # is_available sets the version
        return self._z3_version or "unknown"

    def can_verify(self, claim: str, claim_type: Optional[str] = None) -> bool:
        """
        Check if this backend can attempt to verify the claim.

        Returns True for:
        - Claims already in SMT-LIB2 format
        - Claim types: assertion, precondition, arithmetic, constraint, logical
        - Claims containing quantifiable patterns (for all, exists, implies, etc.)
        """
        import re

        if not self.is_available:
            return False

        # Already SMT-LIB2 format
        if claim.strip().startswith("(") and ("assert" in claim or "declare" in claim):
            return True

        # Explicit claim types we handle
        z3_types = {
            "assertion", "precondition", "postcondition",
            "arithmetic", "constraint", "logical", "invariant",
            "LOGICAL", "FACTUAL",  # ClaimType enum values
        }
        if claim_type and claim_type in z3_types:
            return True

        # Heuristic: check for patterns that suggest formal logic
        quantifiable_patterns = r"\b(for all|forall|exists|there exists|greater than|less than|equal to|equals|implies|if.*then|and|or|not|sum|product|divides|prime|even|odd|positive|negative|integer|real|boolean)\b"
        if re.search(quantifiable_patterns, claim, re.IGNORECASE):
            return True

        # Check for mathematical notation
        math_patterns = r"[<>=≤≥≠∀∃→∧∨¬+\-*/^]"
        if re.search(math_patterns, claim):
            return True

        return False

    def _is_smtlib2(self, text: str) -> bool:
        """Check if text is already in SMT-LIB2 format."""
        text = text.strip()
        return text.startswith("(") and any(
            kw in text for kw in ["declare-", "assert", "check-sat", "define-"]
        )

    def _validate_smtlib2(self, smtlib: str) -> bool:
        """Validate SMT-LIB2 syntax using Z3 parser."""
        try:
            import z3
            ctx = z3.Context()
            z3.parse_smt2_string(smtlib, ctx=ctx)
            return True
        except Exception as e:
            logger.debug(f"SMT-LIB2 validation failed: {type(e).__name__}: {e}")
            return False

    def _simple_translate(self, claim: str) -> Optional[str]:
        """
        Pattern-based translation for simple claims.

        Handles basic arithmetic and logical claims without LLM.
        """
        import re

        claim_lower = claim.lower().strip()

        # Pattern: "X > Y and Y > Z implies X > Z" (transitivity)
        trans_match = re.match(
            r"(?:if\s+)?(\w+)\s*([<>=]+)\s*(\w+)\s+and\s+(\w+)\s*([<>=]+)\s*(\w+)\s*(?:,?\s*then\s+|implies\s+)(\w+)\s*([<>=]+)\s*(\w+)",
            claim_lower,
        )
        if trans_match:
            a, op1, b, c, op2, d, e, op3, f = trans_match.groups()
            return f"""
(declare-const {a} Int)
(declare-const {b} Int)
(declare-const {c} Int)
(declare-const {d} Int)
(declare-const {e} Int)
(declare-const {f} Int)
(assert (not (=> (and ({op1} {a} {b}) ({op2} {c} {d})) ({op3} {e} {f}))))
(check-sat)
"""

        # Pattern: "for all n, n + 0 = n"
        forall_match = re.match(
            r"for all (\w+),?\s+(.+)",
            claim_lower,
        )
        if forall_match:
            var, body = forall_match.groups()
            # Very simplified - just create a quantified assertion
            # Real implementation would parse the body properly
            return None  # Defer to LLM for complex quantified statements

        return None

    async def translate(self, claim: str, context: str = "") -> Optional[str]:
        """
        Translate a natural language claim to SMT-LIB2 format.

        Strategy:
        1. If already SMT-LIB2, validate and return
        2. Try pattern-based translation for simple claims
        3. Use LLM translator if available
        """
        import time

        start = time.time()

        # Already in SMT-LIB2 format
        if self._is_smtlib2(claim):
            if self._validate_smtlib2(claim):
                return claim
            # Invalid SMT-LIB2, try to fix or reject
            return None

        # Try simple pattern-based translation
        simple = self._simple_translate(claim)
        if simple and self._validate_smtlib2(simple):
            return simple

        # Use LLM translator if available
        if self._llm_translator is not None:
            try:
                prompt = f"""Translate this claim to SMT-LIB2 format for Z3 solver.

Claim: {claim}
Context: {context}

Requirements:
- Use (declare-const name Type) for variables
- Use (assert (not ...)) to check validity (prove by showing negation is unsat)
- End with (check-sat)
- Use Int, Real, or Bool types
- Common operators: +, -, *, /, <, >, <=, >=, =, and, or, not, =>

Return ONLY the SMT-LIB2 code, no explanation."""

                smtlib = await self._llm_translator(prompt, context)

                # Extract SMT-LIB2 from response (may have markdown)
                if "```" in smtlib:
                    import re
                    match = re.search(r"```(?:smt2?|smtlib2?)?\n?(.*?)```", smtlib, re.DOTALL)
                    if match:
                        smtlib = match.group(1)

                smtlib = smtlib.strip()

                if self._validate_smtlib2(smtlib):
                    return smtlib

            except Exception as e:
                # LLM translation failed, log and continue to return None
                logger.debug(f"LLM translation to SMT-LIB2 failed: {e}")

        return None

    async def prove(self, formal_statement: str, timeout_seconds: float = 60.0) -> FormalProofResult:
        """
        Attempt to prove a formal statement using Z3.

        The statement should be in SMT-LIB2 format with the claim negated.
        If Z3 returns 'unsat', the original claim is proven.
        If Z3 returns 'sat', a counterexample exists.
        """
        import time

        if not self.is_available:
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=FormalLanguage.Z3_SMT,
                error_message="Z3 Python package not installed. Run: pip install z3-solver",
            )

        try:
            import z3

            start = time.time()

            # Create solver with timeout
            solver = z3.Solver()
            solver.set("timeout", int(timeout_seconds * 1000))  # ms

            # Parse and add assertions
            try:
                assertions = z3.parse_smt2_string(formal_statement)
                for a in assertions:
                    solver.add(a)
            except z3.Z3Exception as e:
                return FormalProofResult(
                    status=FormalProofStatus.TRANSLATION_FAILED,
                    language=FormalLanguage.Z3_SMT,
                    formal_statement=formal_statement,
                    error_message=f"Failed to parse SMT-LIB2: {e}",
                    prover_version=self.z3_version,
                )

            # Check satisfiability
            result = solver.check()
            elapsed_ms = (time.time() - start) * 1000

            if result == z3.unsat:
                # Negation is unsatisfiable → original claim is valid
                proof_text = "QED (negation is unsatisfiable)"
                proof_hash = hashlib.sha256(formal_statement.encode()).hexdigest()[:16]

                return FormalProofResult(
                    status=FormalProofStatus.PROOF_FOUND,
                    language=FormalLanguage.Z3_SMT,
                    formal_statement=formal_statement,
                    proof_text=proof_text,
                    proof_hash=proof_hash,
                    proof_search_time_ms=elapsed_ms,
                    prover_version=self.z3_version,
                )

            elif result == z3.sat:
                # Counterexample found
                model = solver.model()
                counterexample = f"COUNTEREXAMPLE: {model}"

                return FormalProofResult(
                    status=FormalProofStatus.PROOF_FAILED,
                    language=FormalLanguage.Z3_SMT,
                    formal_statement=formal_statement,
                    proof_text=counterexample,
                    proof_search_time_ms=elapsed_ms,
                    error_message="Claim is false - counterexample found",
                    prover_version=self.z3_version,
                )

            else:
                # Unknown (timeout or undecidable)
                return FormalProofResult(
                    status=FormalProofStatus.TIMEOUT,
                    language=FormalLanguage.Z3_SMT,
                    formal_statement=formal_statement,
                    proof_search_time_ms=elapsed_ms,
                    error_message="Solver returned unknown (timeout or undecidable)",
                    prover_version=self.z3_version,
                )

        except Exception as e:
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=FormalLanguage.Z3_SMT,
                formal_statement=formal_statement,
                error_message=f"Z3 error: {e}",
                prover_version=self.z3_version,
            )

    async def verify_proof(self, formal_statement: str, proof: str) -> bool:
        """
        Verify a proof by re-running Z3.

        For Z3, proofs are implicit (unsat result), so we re-run the solver
        and check if we get the same result.
        """
        result = await self.prove(formal_statement, timeout_seconds=30.0)
        return result.status == FormalProofStatus.PROOF_FOUND


class FormalVerificationManager:
    """
    Manages formal verification backends.

    Coordinates between multiple backends, choosing the most
    appropriate one for each claim type.
    """

    def __init__(self):
        self.backends: list[FormalVerificationBackend] = [
            Z3Backend(),  # Try Z3 first (simpler, faster)
            LeanBackend(),  # Fall back to Lean for complex proofs
        ]

    def get_available_backends(self) -> list[FormalVerificationBackend]:
        """Get all available backends."""
        return [b for b in self.backends if b.is_available]

    def get_backend_for_claim(
        self, claim: str, claim_type: Optional[str] = None
    ) -> Optional[FormalVerificationBackend]:
        """Find the best backend for a claim."""
        for backend in self.backends:
            if backend.is_available and backend.can_verify(claim, claim_type):
                return backend
        return None

    async def attempt_formal_verification(
        self,
        claim: str,
        claim_type: Optional[str] = None,
        context: str = "",
        timeout_seconds: float = 60.0,
    ) -> FormalProofResult:
        """
        Attempt to formally verify a claim using available backends.

        Args:
            claim: Natural language claim to verify
            claim_type: Optional hint about claim type
            context: Additional context for translation
            timeout_seconds: Maximum time for proof search

        Returns:
            FormalProofResult with verification status
        """
        backend = self.get_backend_for_claim(claim, claim_type)

        if backend is None:
            return FormalProofResult(
                status=FormalProofStatus.NOT_SUPPORTED,
                language=FormalLanguage.LEAN4,  # Default
                error_message="No suitable formal verification backend available",
            )

        # Translate claim to formal language
        formal_statement = await backend.translate(claim, context)

        if formal_statement is None:
            return FormalProofResult(
                status=FormalProofStatus.TRANSLATION_FAILED,
                language=backend.language,
                error_message="Could not translate claim to formal language",
            )

        # Attempt proof
        result = await backend.prove(formal_statement, timeout_seconds)
        result.formal_statement = formal_statement

        return result

    def status_report(self) -> dict[str, Any]:
        """Get status of all backends."""
        return {
            "backends": [
                {
                    "language": b.language.value,
                    "available": b.is_available,
                }
                for b in self.backends
            ],
            "any_available": any(b.is_available for b in self.backends),
        }


# Singleton instance for easy access
_manager: Optional[FormalVerificationManager] = None


def get_formal_verification_manager() -> FormalVerificationManager:
    """Get the global formal verification manager."""
    global _manager
    if _manager is None:
        _manager = FormalVerificationManager()
    return _manager
