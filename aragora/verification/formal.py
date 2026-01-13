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
import subprocess
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

    async def prove(
        self, formal_statement: str, timeout_seconds: float = 60.0
    ) -> FormalProofResult:
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


class TranslationModel(Enum):
    """Available models for NL-to-Lean translation."""

    DEEPSEEK_PROVER = "deepseek_prover"  # Best for mathematical proofs
    CLAUDE = "claude"  # General-purpose, good at reasoning
    OPENAI = "openai"  # GPT-4, solid alternative
    AUTO = "auto"  # Automatically select best available


class LeanBackend:
    """
    Lean 4 formal verification backend.

    Uses LLM-assisted translation to convert natural language claims to Lean 4
    theorems, then runs the Lean type checker to verify proofs.

    Features:
    - DeepSeek-Prover-V2 integration for state-of-the-art translation
    - Fallback to Claude/GPT-4 for translation
    - Sandboxed Lean execution with resource limits
    - Proof caching for repeated verification
    - Support for common mathematical patterns

    Prerequisites:
    - Lean 4 toolchain installed (lean, lake)
    - API key for translation (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY)
    """

    # Claim types suitable for Lean verification
    LEAN_CLAIM_TYPES = {
        "MATHEMATICAL",
        "LOGICAL",
        "ARITHMETIC",
        "PROOF",
        "THEOREM",
        "LEMMA",
        "PROPERTY",
        "INVARIANT",
    }

    # Patterns indicating mathematical content
    MATH_PATTERNS = [
        r"\bfor all\b",
        r"\bforall\b",
        r"\bexists\b",
        r"\bthere exists\b",
        r"\biff\b",
        r"\bimplies\b",
        r"\bif and only if\b",
        r"\bprove\b",
        r"\bproof\b",
        r"\btheorem\b",
        r"\blemma\b",
        r"\bprime\b",
        r"\bdivisible\b",
        r"\beven\b",
        r"\bodd\b",
        r"\bsum\b",
        r"\bproduct\b",
        r"\bintegral\b",
        r"\bderivative\b",
        r"[∀∃→←↔∧∨¬⊢⊨≡≠≤≥∈∉⊂⊃∩∪∅ℕℤℚℝℂ]",
    ]

    def __init__(
        self,
        sandbox_timeout: float = 60.0,
        sandbox_memory_mb: int = 1024,
        translation_model: TranslationModel = TranslationModel.AUTO,
    ):
        """
        Initialize Lean backend.

        Args:
            sandbox_timeout: Maximum seconds for Lean execution.
            sandbox_memory_mb: Memory limit for Lean process.
            translation_model: Which model to use for NL-to-Lean translation.
                              AUTO will prefer DeepSeek-Prover if available.
        """
        self._sandbox_timeout = sandbox_timeout
        self._sandbox_memory_mb = sandbox_memory_mb
        self._translation_model = translation_model
        self._proof_cache: dict[str, FormalProofResult] = {}
        self._lean_version: Optional[str] = None
        self._deepseek_translator: Optional[Any] = None

    @property
    def language(self) -> FormalLanguage:
        return FormalLanguage.LEAN4

    @property
    def is_available(self) -> bool:
        """Check if Lean 4 toolchain (lean and lake) is available."""
        import shutil

        has_lean = shutil.which("lean") is not None
        if has_lean and self._lean_version is None:
            try:
                result = subprocess.run(
                    ["lean", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self._lean_version = result.stdout.strip().split("\n")[0]
            except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
                logger.debug("Lean version check failed: %s", e)
        return has_lean

    @property
    def lean_version(self) -> str:
        """Get Lean version string."""
        if self._lean_version is None and self.is_available:
            pass  # is_available sets version
        return self._lean_version or "unknown"

    def can_verify(self, claim: str, claim_type: Optional[str] = None) -> bool:
        """
        Check if this backend can attempt to verify the claim.

        Returns True for:
        - Claims with explicit mathematical claim types
        - Claims containing mathematical notation or keywords
        - Claims that appear to be theorems or proofs
        """
        import re

        if not self.is_available:
            return False

        # Check explicit claim type
        if claim_type and claim_type.upper() in self.LEAN_CLAIM_TYPES:
            return True

        # Check for mathematical patterns
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, claim, re.IGNORECASE):
                return True

        return False

    def _get_deepseek_translator(self) -> Optional[Any]:
        """Get DeepSeek-Prover translator instance."""
        if self._deepseek_translator is None:
            try:
                from aragora.verification.deepseek_prover import DeepSeekProverTranslator

                translator = DeepSeekProverTranslator()
                if translator.is_available:
                    self._deepseek_translator = translator
            except ImportError:
                pass
        return self._deepseek_translator

    async def translate(self, claim: str, context: str = "") -> Optional[str]:
        """
        Use LLM to translate a natural language claim to a Lean 4 theorem.

        Returns None if translation fails or no LLM is available.
        Example output: "theorem claim_123 : ∀ n : ℕ, n + 0 = n := by simp"

        Translation model selection (for AUTO mode):
        1. DeepSeek-Prover-V2 (best for mathematical proofs)
        2. Claude (fallback, good reasoning)
        3. GPT-4 (fallback)
        """
        import os

        import aiohttp

        # Try DeepSeek-Prover first if configured
        if self._translation_model in (TranslationModel.AUTO, TranslationModel.DEEPSEEK_PROVER):
            translator = self._get_deepseek_translator()
            if translator:
                result = await translator.translate(claim, context)
                if result.success and result.lean_code:
                    logger.debug(
                        f"DeepSeek-Prover translation succeeded (confidence: {result.confidence:.2f})"
                    )
                    return result.lean_code
                elif self._translation_model == TranslationModel.DEEPSEEK_PROVER:
                    # Explicit DeepSeek selection but failed
                    logger.warning(f"DeepSeek-Prover translation failed: {result.error_message}")
                    return None

        # Fall back to Claude/OpenAI
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        # Create translation prompt
        prompt = f"""Translate the following natural language claim into a Lean 4 theorem with proof.

Claim: {claim}
{f'Context: {context}' if context else ''}

Requirements:
1. Use valid Lean 4 syntax (not Lean 3)
2. Include necessary imports (e.g., import Mathlib.Tactic)
3. Provide a complete proof using tactics like simp, ring, omega, decide, etc.
4. If the claim is false, prove its negation instead and note this
5. Keep theorem name simple (claim_1, main_theorem)

If the claim cannot be expressed as a Lean theorem, return exactly: UNTRANSLATABLE

Return ONLY the Lean 4 code, no explanations. Example:
```lean
import Mathlib.Tactic

theorem claim_1 : ∀ n : Nat, n + 0 = n := by simp
```"""

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
                    "max_tokens": 2048,
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
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                return None

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"LLM API returned status {response.status} for Lean translation"
                        )
                        return None

                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        logger.warning(f"Failed to parse LLM API response as JSON: {e}")
                        return None

                    try:
                        if anthropic_key:
                            result = data["content"][0]["text"].strip()
                        else:
                            result = data["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError, TypeError) as e:
                        logger.warning(f"Unexpected LLM response format for Lean translation: {e}")
                        return None

                    if "UNTRANSLATABLE" in result:
                        return None

                    # Extract Lean code from markdown if present
                    import re

                    lean_match = re.search(r"```(?:lean4?|lean)?\n?(.*?)```", result, re.DOTALL)
                    if lean_match:
                        return lean_match.group(1).strip()
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

    async def prove(
        self, formal_statement: str, timeout_seconds: float = 60.0
    ) -> FormalProofResult:
        """
        Attempt to verify a Lean 4 theorem using the Lean type checker.

        Runs the Lean code in a sandboxed subprocess and checks if it
        type-checks successfully (meaning the proof is valid).

        Args:
            formal_statement: Lean 4 code with theorem and proof.
            timeout_seconds: Maximum time for Lean execution.

        Returns:
            FormalProofResult with verification status.
        """
        import time

        if not self.is_available:
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=FormalLanguage.LEAN4,
                error_message="Lean 4 not installed. Install from https://leanprover.github.io/",
            )

        # Check cache
        cache_key = hashlib.sha256(formal_statement.encode()).hexdigest()
        if cache_key in self._proof_cache:
            cached = self._proof_cache[cache_key]
            logger.debug(f"Lean proof cache hit for {cache_key[:8]}")
            return cached

        start_time = time.time()

        try:
            from aragora.verification.sandbox import ProofSandbox, SandboxStatus

            sandbox = ProofSandbox(
                timeout=timeout_seconds,
                memory_mb=self._sandbox_memory_mb,
            )

            result = await sandbox.execute_lean(formal_statement)
            elapsed_ms = (time.time() - start_time) * 1000

            if result.status == SandboxStatus.TIMEOUT:
                proof_result = FormalProofResult(
                    status=FormalProofStatus.TIMEOUT,
                    language=FormalLanguage.LEAN4,
                    formal_statement=formal_statement,
                    proof_search_time_ms=elapsed_ms,
                    error_message=f"Lean execution exceeded {timeout_seconds}s timeout",
                    prover_version=self.lean_version,
                )
            elif result.status == SandboxStatus.SETUP_FAILED:
                proof_result = FormalProofResult(
                    status=FormalProofStatus.BACKEND_UNAVAILABLE,
                    language=FormalLanguage.LEAN4,
                    formal_statement=formal_statement,
                    error_message=result.error_message,
                    prover_version=self.lean_version,
                )
            elif result.is_success:
                # Lean returned 0 = proof type-checked successfully
                proof_hash = hashlib.sha256(formal_statement.encode()).hexdigest()[:16]
                proof_result = FormalProofResult(
                    status=FormalProofStatus.PROOF_FOUND,
                    language=FormalLanguage.LEAN4,
                    formal_statement=formal_statement,
                    proof_text=formal_statement,
                    proof_hash=proof_hash,
                    proof_search_time_ms=elapsed_ms,
                    prover_version=self.lean_version,
                )
            else:
                # Lean returned non-zero = type error or proof failure
                error_msg = result.stderr or result.stdout or "Unknown error"
                # Check for common error patterns
                if "sorry" in error_msg.lower() or "declaration uses 'sorry'" in error_msg:
                    error_msg = "Proof incomplete (contains 'sorry')"
                elif "type mismatch" in error_msg.lower():
                    error_msg = f"Type error in proof: {error_msg[:200]}"

                proof_result = FormalProofResult(
                    status=FormalProofStatus.PROOF_FAILED,
                    language=FormalLanguage.LEAN4,
                    formal_statement=formal_statement,
                    proof_search_time_ms=elapsed_ms,
                    error_message=error_msg[:500],
                    prover_version=self.lean_version,
                )

            # Cache the result
            self._proof_cache[cache_key] = proof_result
            return proof_result

        except ImportError:
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=FormalLanguage.LEAN4,
                formal_statement=formal_statement,
                error_message="ProofSandbox not available",
            )
        except Exception as e:
            return FormalProofResult(
                status=FormalProofStatus.BACKEND_UNAVAILABLE,
                language=FormalLanguage.LEAN4,
                formal_statement=formal_statement,
                error_message=f"Lean execution error: {type(e).__name__}: {e}",
                prover_version=self.lean_version,
            )

    async def verify_proof(self, formal_statement: str, proof: str) -> bool:
        """
        Verify that a proof is valid by re-running Lean.

        For Lean, we combine the statement and proof and check if it type-checks.
        """
        # If proof is separate from statement, combine them
        if proof and proof.strip() not in formal_statement:
            full_code = f"{formal_statement}\n\n{proof}"
        else:
            full_code = formal_statement

        result = await self.prove(full_code, timeout_seconds=30.0)
        return result.status == FormalProofStatus.PROOF_FOUND

    def clear_cache(self) -> int:
        """Clear the proof cache. Returns number of entries cleared."""
        count = len(self._proof_cache)
        self._proof_cache.clear()
        return count


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

    def __init__(
        self,
        llm_translator: Optional[Any] = None,
        cache_size: int = 100,
        cache_ttl_seconds: float = 3600.0,
    ):
        """
        Initialize Z3 backend.

        Args:
            llm_translator: Optional async callable(claim, context) -> str
                           for LLM-assisted translation to SMT-LIB2
            cache_size: Maximum number of proof results to cache
            cache_ttl_seconds: Time-to-live for cached results (default: 1 hour)
        """
        self._llm_translator = llm_translator
        self._z3_version: Optional[str] = None
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl_seconds
        # Cache: hash -> (timestamp, FormalProofResult)
        self._proof_cache: dict[str, tuple[float, FormalProofResult]] = {}

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
            "assertion",
            "precondition",
            "postcondition",
            "arithmetic",
            "constraint",
            "logical",
            "invariant",
            "LOGICAL",
            "FACTUAL",  # ClaimType enum values
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

    def _cache_key(self, formal_statement: str) -> str:
        """Generate cache key from formal statement."""
        return hashlib.sha256(formal_statement.encode()).hexdigest()[:16]

    def _get_cached(self, formal_statement: str) -> Optional[FormalProofResult]:
        """Get cached proof result if valid and not expired."""
        import time

        key = self._cache_key(formal_statement)
        if key not in self._proof_cache:
            return None

        timestamp, result = self._proof_cache[key]
        if time.time() - timestamp > self._cache_ttl:
            # Expired, remove from cache
            del self._proof_cache[key]
            return None

        logger.debug(f"Z3 proof cache hit for key {key}")
        return result

    def _cache_result(self, formal_statement: str, result: FormalProofResult) -> None:
        """Cache a proof result with LRU eviction."""
        import time

        key = self._cache_key(formal_statement)

        # LRU eviction if at capacity
        if len(self._proof_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = min(self._proof_cache.keys(), key=lambda k: self._proof_cache[k][0])
            del self._proof_cache[oldest_key]

        self._proof_cache[key] = (time.time(), result)

    def clear_cache(self) -> int:
        """Clear the proof cache. Returns number of entries cleared."""
        count = len(self._proof_cache)
        self._proof_cache.clear()
        return count

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

    async def prove(
        self, formal_statement: str, timeout_seconds: float = 60.0
    ) -> FormalProofResult:
        """
        Attempt to prove a formal statement using Z3.

        The statement should be in SMT-LIB2 format with the claim negated.
        If Z3 returns 'unsat', the original claim is proven.
        If Z3 returns 'sat', a counterexample exists.

        Results are cached by statement hash with configurable TTL.
        """
        import time

        # Check cache first
        cached = self._get_cached(formal_statement)
        if cached is not None:
            return cached

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

                proof_result = FormalProofResult(
                    status=FormalProofStatus.PROOF_FOUND,
                    language=FormalLanguage.Z3_SMT,
                    formal_statement=formal_statement,
                    proof_text=proof_text,
                    proof_hash=proof_hash,
                    proof_search_time_ms=elapsed_ms,
                    prover_version=self.z3_version,
                )
                self._cache_result(formal_statement, proof_result)
                return proof_result

            elif result == z3.sat:
                # Counterexample found
                model = solver.model()
                counterexample = f"COUNTEREXAMPLE: {model}"

                proof_result = FormalProofResult(
                    status=FormalProofStatus.PROOF_FAILED,
                    language=FormalLanguage.Z3_SMT,
                    formal_statement=formal_statement,
                    proof_text=counterexample,
                    proof_search_time_ms=elapsed_ms,
                    error_message="Claim is false - counterexample found",
                    prover_version=self.z3_version,
                )
                self._cache_result(formal_statement, proof_result)
                return proof_result

            else:
                # Unknown (timeout or undecidable) - don't cache timeouts
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
