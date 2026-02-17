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

    def get_status(self) -> dict[str, Any]:
        """Get verification service status."""
        return self._client.request("GET", "/api/v1/verification/status")

    def formal_verify(self, **kwargs: Any) -> dict[str, Any]:
        """Run formal verification on a decision or claim."""
        return self._client.request("POST", "/api/v1/verification/formal-verify", json=kwargs)


class AsyncVerificationAPI:
    """
    Asynchronous Verification API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.verification.verify_consensus("debate_123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_status(self) -> dict[str, Any]:
        """Get verification service status."""
        return await self._client.request("GET", "/api/v1/verification/status")

    async def formal_verify(self, **kwargs: Any) -> dict[str, Any]:
        """Run formal verification on a decision or claim."""
        return await self._client.request("POST", "/api/v1/verification/formal-verify", json=kwargs)
