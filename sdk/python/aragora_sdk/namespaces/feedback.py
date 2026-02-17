"""
Feedback Namespace API

Provides methods for collecting user feedback including NPS surveys,
feature requests, bug reports, and general suggestions.

Features:
- NPS (Net Promoter Score) submission
- General feedback submission
- NPS analytics (admin)
- Feedback prompts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

FeedbackType = Literal["feature_request", "bug_report", "general", "debate_quality"]


class FeedbackAPI:
    """
    Synchronous Feedback API.

    Provides methods for submitting and managing user feedback:
    - NPS surveys
    - Feature requests
    - Bug reports
    - General feedback
    - Feedback analytics (admin)

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> client.feedback.submit_nps(score=9, comment="Great product!")
        >>> prompts = client.feedback.get_prompts()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # NPS (Net Promoter Score)
    # ===========================================================================

    def submit_nps(
        self,
        score: int,
        comment: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit an NPS score."""
        body: dict[str, Any] = {"score": score, **kwargs}
        if comment:
            body["comment"] = comment
        return self._client.request("POST", "/api/v1/feedback/nps", json=body)

    def get_nps_summary(self) -> dict[str, Any]:
        """Get NPS summary and analytics."""
        return self._client.request("GET", "/api/v1/feedback/nps/summary")

    # ===========================================================================
    # General Feedback
    # ===========================================================================

    def submit_feedback(
        self,
        comment: str,
        feedback_type: FeedbackType = "general",
        score: int | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit general feedback."""
        body: dict[str, Any] = {
            "comment": comment,
            "type": feedback_type,
            **kwargs,
        }
        if score is not None:
            body["score"] = score
        if context:
            body["context"] = context
        return self._client.request("POST", "/api/v1/feedback/general", json=body)

    def submit_feature_request(
        self,
        comment: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a feature request."""
        return self.submit_feedback(
            comment=comment,
            feedback_type="feature_request",
            context=context,
        )

    def submit_bug_report(
        self,
        comment: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a bug report."""
        return self.submit_feedback(
            comment=comment,
            feedback_type="bug_report",
            context=context,
        )

    def submit_debate_quality_feedback(
        self,
        debate_id: str,
        comment: str,
        score: int | None = None,
    ) -> dict[str, Any]:
        """Submit debate quality feedback."""
        return self.submit_feedback(
            comment=comment,
            feedback_type="debate_quality",
            score=score,
            context={"debate_id": debate_id},
        )

    # ===========================================================================
    # Feedback Prompts
    # ===========================================================================

    def get_prompts(self) -> dict[str, Any]:
        """Get feedback prompts configuration."""
        return self._client.request("GET", "/api/v1/feedback/prompts")


class AsyncFeedbackAPI:
    """
    Asynchronous Feedback API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     await client.feedback.submit_nps(score=9, comment="Great!")
        ...     prompts = await client.feedback.get_prompts()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # NPS (Net Promoter Score)
    # ===========================================================================

    async def submit_nps(
        self,
        score: int,
        comment: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit an NPS score."""
        body: dict[str, Any] = {"score": score, **kwargs}
        if comment:
            body["comment"] = comment
        return await self._client.request("POST", "/api/v1/feedback/nps", json=body)

    async def get_nps_summary(self) -> dict[str, Any]:
        """Get NPS summary and analytics."""
        return await self._client.request("GET", "/api/v1/feedback/nps/summary")

    # ===========================================================================
    # General Feedback
    # ===========================================================================

    async def submit_feedback(
        self,
        comment: str,
        feedback_type: FeedbackType = "general",
        score: int | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit general feedback."""
        body: dict[str, Any] = {
            "comment": comment,
            "type": feedback_type,
            **kwargs,
        }
        if score is not None:
            body["score"] = score
        if context:
            body["context"] = context
        return await self._client.request("POST", "/api/v1/feedback/general", json=body)

    async def submit_feature_request(
        self,
        comment: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a feature request."""
        return await self.submit_feedback(
            comment=comment,
            feedback_type="feature_request",
            context=context,
        )

    async def submit_bug_report(
        self,
        comment: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a bug report."""
        return await self.submit_feedback(
            comment=comment,
            feedback_type="bug_report",
            context=context,
        )

    async def submit_debate_quality_feedback(
        self,
        debate_id: str,
        comment: str,
        score: int | None = None,
    ) -> dict[str, Any]:
        """Submit debate quality feedback."""
        return await self.submit_feedback(
            comment=comment,
            feedback_type="debate_quality",
            score=score,
            context={"debate_id": debate_id},
        )

    # ===========================================================================
    # Feedback Prompts
    # ===========================================================================

    async def get_prompts(self) -> dict[str, Any]:
        """Get feedback prompts configuration."""
        return await self._client.request("GET", "/api/v1/feedback/prompts")
