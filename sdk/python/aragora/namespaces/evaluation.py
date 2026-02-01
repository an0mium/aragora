"""
Evaluation Namespace API

Provides methods for LLM-as-Judge evaluation operations:
- Evaluating responses against quality dimensions
- Comparing two responses head-to-head
- Managing evaluation dimensions and profiles
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class EvaluationAPI:
    """
    Synchronous Evaluation API.

    Provides methods for LLM-as-Judge evaluation:
    - Evaluate responses for quality across multiple dimensions
    - Compare two responses head-to-head
    - List and retrieve evaluation dimensions and profiles

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.evaluation.evaluate(
        ...     response="The capital of France is Paris.",
        ...     prompt="What is the capital of France?",
        ...     dimensions=["accuracy", "clarity", "completeness"],
        ... )
        >>> comparison = client.evaluation.compare(
        ...     response_a="First answer...",
        ...     response_b="Second answer...",
        ...     prompt="Original question",
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Evaluation Operations
    # ===========================================================================

    def evaluate(
        self,
        response: str,
        prompt: str | None = None,
        context: str | None = None,
        dimensions: list[str] | None = None,
        profile: str | None = None,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single response for quality.

        Args:
            response: The response text to evaluate
            prompt: The original prompt that generated the response (optional)
            context: Additional context for evaluation (optional)
            dimensions: List of dimension IDs to evaluate against (optional)
            profile: Evaluation profile ID to use (optional)
            reference: Reference answer for comparison (optional)

        Returns:
            Dict with evaluation results including:
            - overall_score: Aggregate quality score
            - dimension_scores: Scores per dimension
            - feedback: Qualitative feedback
            - strengths: List of identified strengths
            - weaknesses: List of identified weaknesses
            - suggestions: Improvement suggestions
            - profile_used: ID of the profile used
        """
        data: dict[str, Any] = {"response": response}
        if prompt is not None:
            data["prompt"] = prompt
        if context is not None:
            data["context"] = context
        if dimensions is not None:
            data["dimensions"] = dimensions
        if profile is not None:
            data["profile"] = profile
        if reference is not None:
            data["reference"] = reference
        return self._client.request("POST", "/api/v1/evaluate", json=data)

    def compare(
        self,
        response_a: str,
        response_b: str,
        prompt: str | None = None,
        context: str | None = None,
        dimensions: list[str] | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """
        Compare two responses head-to-head.

        Args:
            response_a: First response to compare
            response_b: Second response to compare
            prompt: The original prompt (optional)
            context: Additional context for comparison (optional)
            dimensions: List of dimension IDs to compare against (optional)
            profile: Evaluation profile ID to use (optional)

        Returns:
            Dict with comparison results including:
            - winner: 'A', 'B', or 'tie'
            - margin: Score difference between responses
            - response_a_score: Overall score for response A
            - response_b_score: Overall score for response B
            - dimension_comparison: Per-dimension comparison with scores
            - reasoning: Explanation of the comparison result
        """
        data: dict[str, Any] = {
            "response_a": response_a,
            "response_b": response_b,
        }
        if prompt is not None:
            data["prompt"] = prompt
        if context is not None:
            data["context"] = context
        if dimensions is not None:
            data["dimensions"] = dimensions
        if profile is not None:
            data["profile"] = profile
        return self._client.request("POST", "/api/v1/evaluate/compare", json=data)

    # ===========================================================================
    # Dimensions and Profiles
    # ===========================================================================

    def list_dimensions(self) -> dict[str, Any]:
        """
        List available evaluation dimensions.

        Returns:
            Dict with dimensions array containing:
            - id: Dimension identifier
            - name: Human-readable name
            - description: What the dimension measures
            - weight: Default weight in scoring
            - criteria: List of evaluation criteria
        """
        return self._client.request("GET", "/api/v1/evaluate/dimensions")

    def list_profiles(self) -> dict[str, Any]:
        """
        List available evaluation profiles.

        Returns:
            Dict with profiles array containing:
            - id: Profile identifier
            - name: Human-readable name
            - description: Profile purpose
            - dimensions: Dimension weights mapping
            - default: Whether this is the default profile
        """
        return self._client.request("GET", "/api/v1/evaluate/profiles")

    def get_dimension(self, dimension_id: str) -> dict[str, Any]:
        """
        Get a specific evaluation dimension by ID.

        Args:
            dimension_id: The dimension ID to retrieve

        Returns:
            Dict with dimension details

        Raises:
            ValueError: If dimension is not found
        """
        result = self.list_dimensions()
        dimensions: list[dict[str, Any]] = result.get("dimensions", [])
        for dimension in dimensions:
            if dimension.get("id") == dimension_id:
                return dimension
        raise ValueError(f"Dimension not found: {dimension_id}")

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        """
        Get a specific evaluation profile by ID.

        Args:
            profile_id: The profile ID to retrieve

        Returns:
            Dict with profile details

        Raises:
            ValueError: If profile is not found
        """
        result = self.list_profiles()
        profiles: list[dict[str, Any]] = result.get("profiles", [])
        for profile in profiles:
            if profile.get("id") == profile_id:
                return profile
        raise ValueError(f"Profile not found: {profile_id}")


class AsyncEvaluationAPI:
    """
    Asynchronous Evaluation API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.evaluation.evaluate(
        ...         response="The capital of France is Paris.",
        ...         prompt="What is the capital of France?",
        ...     )
        ...     dimensions = await client.evaluation.list_dimensions()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Evaluation Operations
    # ===========================================================================

    async def evaluate(
        self,
        response: str,
        prompt: str | None = None,
        context: str | None = None,
        dimensions: list[str] | None = None,
        profile: str | None = None,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single response for quality.

        Args:
            response: The response text to evaluate
            prompt: The original prompt that generated the response (optional)
            context: Additional context for evaluation (optional)
            dimensions: List of dimension IDs to evaluate against (optional)
            profile: Evaluation profile ID to use (optional)
            reference: Reference answer for comparison (optional)

        Returns:
            Dict with evaluation results including overall_score, dimension_scores,
            feedback, strengths, weaknesses, suggestions, and profile_used.
        """
        data: dict[str, Any] = {"response": response}
        if prompt is not None:
            data["prompt"] = prompt
        if context is not None:
            data["context"] = context
        if dimensions is not None:
            data["dimensions"] = dimensions
        if profile is not None:
            data["profile"] = profile
        if reference is not None:
            data["reference"] = reference
        return await self._client.request("POST", "/api/v1/evaluate", json=data)

    async def compare(
        self,
        response_a: str,
        response_b: str,
        prompt: str | None = None,
        context: str | None = None,
        dimensions: list[str] | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """
        Compare two responses head-to-head.

        Args:
            response_a: First response to compare
            response_b: Second response to compare
            prompt: The original prompt (optional)
            context: Additional context for comparison (optional)
            dimensions: List of dimension IDs to compare against (optional)
            profile: Evaluation profile ID to use (optional)

        Returns:
            Dict with comparison results including winner, margin, scores,
            dimension_comparison, and reasoning.
        """
        data: dict[str, Any] = {
            "response_a": response_a,
            "response_b": response_b,
        }
        if prompt is not None:
            data["prompt"] = prompt
        if context is not None:
            data["context"] = context
        if dimensions is not None:
            data["dimensions"] = dimensions
        if profile is not None:
            data["profile"] = profile
        return await self._client.request("POST", "/api/v1/evaluate/compare", json=data)

    # ===========================================================================
    # Dimensions and Profiles
    # ===========================================================================

    async def list_dimensions(self) -> dict[str, Any]:
        """List available evaluation dimensions."""
        return await self._client.request("GET", "/api/v1/evaluate/dimensions")

    async def list_profiles(self) -> dict[str, Any]:
        """List available evaluation profiles."""
        return await self._client.request("GET", "/api/v1/evaluate/profiles")

    async def get_dimension(self, dimension_id: str) -> dict[str, Any]:
        """
        Get a specific evaluation dimension by ID.

        Args:
            dimension_id: The dimension ID to retrieve

        Returns:
            Dict with dimension details

        Raises:
            ValueError: If dimension is not found
        """
        result = await self.list_dimensions()
        dimensions: list[dict[str, Any]] = result.get("dimensions", [])
        for dimension in dimensions:
            if dimension.get("id") == dimension_id:
                return dimension
        raise ValueError(f"Dimension not found: {dimension_id}")

    async def get_profile(self, profile_id: str) -> dict[str, Any]:
        """
        Get a specific evaluation profile by ID.

        Args:
            profile_id: The profile ID to retrieve

        Returns:
            Dict with profile details

        Raises:
            ValueError: If profile is not found
        """
        result = await self.list_profiles()
        profiles: list[dict[str, Any]] = result.get("profiles", [])
        for profile in profiles:
            if profile.get("id") == profile_id:
                return profile
        raise ValueError(f"Profile not found: {profile_id}")
