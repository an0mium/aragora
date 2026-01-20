"""Gauntlet API resource for adversarial validation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient

from aragora.client.errors import AragoraAPIError
from aragora.client.models import (
    GauntletReceipt,
    GauntletRunRequest,
    GauntletRunResponse,
)


class GauntletAPI:
    """API interface for gauntlet (adversarial validation)."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def run(
        self,
        input_content: str,
        input_type: str = "text",
        persona: str = "security",
        profile: str = "default",
    ) -> GauntletRunResponse:
        """
        Start a gauntlet analysis run.

        Args:
            input_content: Content to analyze.
            input_type: Type of content (text, policy, code).
            persona: Analysis persona (security, gdpr, hipaa, etc).
            profile: Analysis depth (quick, default, thorough).

        Returns:
            GauntletRunResponse with gauntlet_id.
        """
        request = GauntletRunRequest(
            input_content=input_content,
            input_type=input_type,
            persona=persona,
            profile=profile,
        )

        response = self._client._post("/api/gauntlet/run", request.model_dump())
        return GauntletRunResponse(**response)

    async def run_async(
        self,
        input_content: str,
        input_type: str = "text",
        persona: str = "security",
        profile: str = "default",
    ) -> GauntletRunResponse:
        """Async version of run()."""
        request = GauntletRunRequest(
            input_content=input_content,
            input_type=input_type,
            persona=persona,
            profile=profile,
        )

        response = await self._client._post_async("/api/gauntlet/run", request.model_dump())
        return GauntletRunResponse(**response)

    def get_receipt(self, gauntlet_id: str) -> GauntletReceipt:
        """
        Get the decision receipt for a gauntlet run.

        Args:
            gauntlet_id: The gauntlet run ID.

        Returns:
            GauntletReceipt with verdict and findings.
        """
        response = self._client._get(f"/api/gauntlet/{gauntlet_id}/receipt")
        return GauntletReceipt(**response)

    async def get_receipt_async(self, gauntlet_id: str) -> GauntletReceipt:
        """Async version of get_receipt()."""
        response = await self._client._get_async(f"/api/gauntlet/{gauntlet_id}/receipt")
        return GauntletReceipt(**response)

    def run_and_wait(
        self,
        input_content: str,
        input_type: str = "text",
        persona: str = "security",
        profile: str = "default",
        timeout: int = 900,
    ) -> GauntletReceipt:
        """
        Run gauntlet and wait for completion.

        Args:
            input_content: Content to analyze.
            input_type: Type of content.
            persona: Analysis persona.
            profile: Analysis depth.
            timeout: Maximum wait time in seconds.

        Returns:
            GauntletReceipt with full results.
        """
        response = self.run(input_content, input_type, persona, profile)
        gauntlet_id = response.gauntlet_id

        start = time.time()
        while time.time() - start < timeout:
            try:
                return self.get_receipt(gauntlet_id)
            except AragoraAPIError as e:
                if e.status_code != 404:
                    raise
            time.sleep(5)

        raise TimeoutError(f"Gauntlet {gauntlet_id} did not complete within {timeout}s")


__all__ = ["GauntletAPI"]
