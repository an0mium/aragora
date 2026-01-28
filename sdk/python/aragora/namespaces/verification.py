"""
Verification Namespace API

Provides methods for formal verification of decisions:
- Consensus verification
- Formal proof generation
- Claim verification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class VerificationAPI:
    """
    Synchronous Verification API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.verification.verify_consensus("debate_123")
        >>> if result["valid"]:
        ...     print("Consensus verified:", result["proof_hash"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def verify_consensus(self, debate_id: str) -> dict[str, Any]:
        """
        Verify consensus for a debate.

        Args:
            debate_id: Debate ID to verify

        Returns:
            Verification result with validity and proof details
        """
        return self._client.request("POST", f"/api/v1/verification/debates/{debate_id}/consensus")

    def generate_proof(
        self,
        debate_id: str,
        backend: str = "z3",
    ) -> dict[str, Any]:
        """
        Generate a formal proof for a debate.

        Args:
            debate_id: Debate ID
            backend: Verification backend (z3, lean, etc.)

        Returns:
            Generated proof with verification status
        """
        return self._client.request(
            "POST",
            f"/api/v1/verification/debates/{debate_id}/proof",
            json={"backend": backend},
        )

    def get_proof(self, debate_id: str) -> dict[str, Any]:
        """
        Get existing proof for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Proof details if available
        """
        return self._client.request("GET", f"/api/v1/verification/debates/{debate_id}/proof")

    def verify_claim(
        self,
        claim: str,
        context: str | None = None,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Verify a specific claim.

        Args:
            claim: Claim text to verify
            context: Optional context for verification
            evidence: Optional supporting evidence

        Returns:
            Claim verification result
        """
        data: dict[str, Any] = {"claim": claim}
        if context:
            data["context"] = context
        if evidence:
            data["evidence"] = evidence

        return self._client.request("POST", "/api/v1/verification/claims", json=data)

    def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """
        Verify a decision receipt's integrity.

        Args:
            receipt_id: Receipt ID to verify

        Returns:
            Verification result with hash and validity
        """
        return self._client.request("POST", f"/api/v1/verification/receipts/{receipt_id}")

    def verify_batch(self, receipt_ids: list[str]) -> dict[str, Any]:
        """
        Verify multiple receipts in batch.

        Args:
            receipt_ids: List of receipt IDs to verify

        Returns:
            Batch verification results
        """
        return self._client.request(
            "POST",
            "/api/v1/verification/receipts/batch",
            json={"receipt_ids": receipt_ids},
        )

    def get_verification_status(self, debate_id: str) -> dict[str, Any]:
        """
        Get verification status for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Current verification status
        """
        return self._client.request("GET", f"/api/v1/verification/debates/{debate_id}/status")

    def list_verified_debates(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List debates with verified consensus.

        Args:
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of verified debates
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/verification/debates", params=params)


class AsyncVerificationAPI:
    """
    Asynchronous Verification API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.verification.verify_consensus("debate_123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def verify_consensus(self, debate_id: str) -> dict[str, Any]:
        """Verify consensus for a debate."""
        return await self._client.request(
            "POST", f"/api/v1/verification/debates/{debate_id}/consensus"
        )

    async def generate_proof(
        self,
        debate_id: str,
        backend: str = "z3",
    ) -> dict[str, Any]:
        """Generate a formal proof for a debate."""
        return await self._client.request(
            "POST",
            f"/api/v1/verification/debates/{debate_id}/proof",
            json={"backend": backend},
        )

    async def get_proof(self, debate_id: str) -> dict[str, Any]:
        """Get existing proof for a debate."""
        return await self._client.request("GET", f"/api/v1/verification/debates/{debate_id}/proof")

    async def verify_claim(
        self,
        claim: str,
        context: str | None = None,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify a specific claim."""
        data: dict[str, Any] = {"claim": claim}
        if context:
            data["context"] = context
        if evidence:
            data["evidence"] = evidence

        return await self._client.request("POST", "/api/v1/verification/claims", json=data)

    async def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Verify a decision receipt's integrity."""
        return await self._client.request("POST", f"/api/v1/verification/receipts/{receipt_id}")

    async def verify_batch(self, receipt_ids: list[str]) -> dict[str, Any]:
        """Verify multiple receipts in batch."""
        return await self._client.request(
            "POST",
            "/api/v1/verification/receipts/batch",
            json={"receipt_ids": receipt_ids},
        )

    async def get_verification_status(self, debate_id: str) -> dict[str, Any]:
        """Get verification status for a debate."""
        return await self._client.request("GET", f"/api/v1/verification/debates/{debate_id}/status")

    async def list_verified_debates(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List debates with verified consensus."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/verification/debates", params=params)
