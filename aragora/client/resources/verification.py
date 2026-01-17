"""VerificationAPI resource for the Aragora client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import (
    VerifyClaimRequest,
    VerifyClaimResponse,
    VerifyStatusResponse,
)

if TYPE_CHECKING:
    from ..client import AragoraClient


class VerificationAPI:
    """API interface for formal verification of claims."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def verify(
        self,
        claim: str,
        context: str | None = None,
        backend: str = "z3",
        timeout: int = 30,
    ) -> VerifyClaimResponse:
        """
        Verify a claim using formal methods.

        Args:
            claim: The claim to verify in natural language.
            context: Optional context for the claim.
            backend: Verification backend (z3, lean, coq).
            timeout: Verification timeout in seconds.

        Returns:
            VerifyClaimResponse with status, proof, or counterexample.
        """
        request = VerifyClaimRequest(
            claim=claim,
            context=context,
            backend=backend,
            timeout=timeout,
        )

        response = self._client._post("/api/verify/claim", request.model_dump())
        return VerifyClaimResponse(**response)

    async def verify_async(
        self,
        claim: str,
        context: str | None = None,
        backend: str = "z3",
        timeout: int = 30,
    ) -> VerifyClaimResponse:
        """Async version of verify()."""
        request = VerifyClaimRequest(
            claim=claim,
            context=context,
            backend=backend,
            timeout=timeout,
        )

        response = await self._client._post_async("/api/verify/claim", request.model_dump())
        return VerifyClaimResponse(**response)

    def status(self) -> VerifyStatusResponse:
        """
        Check verification backend availability.

        Returns:
            VerifyStatusResponse with available backends.
        """
        response = self._client._get("/api/verify/status")
        return VerifyStatusResponse(**response)

    async def status_async(self) -> VerifyStatusResponse:
        """Async version of status()."""
        response = await self._client._get_async("/api/verify/status")
        return VerifyStatusResponse(**response)
