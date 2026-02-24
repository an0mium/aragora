"""
Verification Namespace API

Provides methods for formal verification of decisions:
- Verification service status
- Formal proof generation (Z3/Lean backends)
- Consensus verification with proof hashes
- Claim validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class VerificationAPI:
    """
    Synchronous Verification API.

    Provides formal verification of decisions using Z3 and Lean
    verification backends, including consensus proofs and claim validation.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.verification.get_status()
        >>> result = client.verification.formal_verify(
        ...     debate_id="debate_123",
        ...     proof_type="consensus",
        ... )
        >>> if result["valid"]:
        ...     print("Verified:", result["proof_hash"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_status(self) -> dict[str, Any]:
        """
        Get verification service status.

        Returns:
            Dict with service status, available backends (Z3, Lean),
            and verification capabilities.
        """
        return self._client.request("GET", "/api/v1/verification/status")

    def formal_verify(self, **kwargs: Any) -> dict[str, Any]:
        """
        Run formal verification on a decision or claim.

        Args:
            **kwargs: Verification parameters including:
                - debate_id: Debate to verify
                - proof_type: Type of proof (consensus, claim, decision)
                - backend: Verification backend (z3, lean)

        Returns:
            Dict with verification result including:
            - valid: Whether the proof is valid
            - proof_hash: Hash of the verification proof
            - backend_used: Which backend performed verification
        """
        return self._client.request("POST", "/api/v1/verification/formal-verify", json=kwargs)

    def get_proofs(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get generated verification proofs.

        Args:
            **kwargs: Filter parameters including:
                - debate_id: Filter proofs by debate
                - proof_type: Filter by proof type
                - limit: Maximum proofs to return

        Returns:
            Dict with verification proofs and their metadata.
        """
        # TODO: server route not yet implemented
        return self._client.request("GET", "/api/v1/verification/proofs", params=kwargs or None)

    def validate(self, **kwargs: Any) -> dict[str, Any]:
        """
        Validate a specific claim or assertion.

        Args:
            **kwargs: Validation parameters including:
                - claim: The claim text to validate
                - evidence: Supporting evidence
                - context: Context for the claim

        Returns:
            Dict with validation result including confidence and reasoning.
        """
        # TODO: server route not yet implemented
        return self._client.request("POST", "/api/v1/verification/validate", json=kwargs)


class AsyncVerificationAPI:
    """
    Asynchronous Verification API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.verification.formal_verify(
        ...         debate_id="debate_123"
        ...     )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_status(self) -> dict[str, Any]:
        """Get verification service status."""
        return await self._client.request("GET", "/api/v1/verification/status")

    async def formal_verify(self, **kwargs: Any) -> dict[str, Any]:
        """Run formal verification on a decision or claim."""
        return await self._client.request("POST", "/api/v1/verification/formal-verify", json=kwargs)

    async def get_proofs(self, **kwargs: Any) -> dict[str, Any]:
        """Get generated verification proofs."""
        # TODO: server route not yet implemented
        return await self._client.request(
            "GET", "/api/v1/verification/proofs", params=kwargs or None
        )

    async def validate(self, **kwargs: Any) -> dict[str, Any]:
        """Validate a specific claim or assertion."""
        # TODO: server route not yet implemented
        return await self._client.request("POST", "/api/v1/verification/validate", json=kwargs)
