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

    def submit_feature_request(
        self,
        comment: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a feature request.

        Convenience method for submit_feedback with type="feature_request".

        Args:
            comment: Description of the requested feature
            context: Optional metadata

        Returns:
            Dict with success status and feedback_id
        """
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
        """
        Submit a bug report.

        Convenience method for submit_feedback with type="bug_report".

        Args:
            comment: Description of the bug
            context: Optional metadata (e.g., steps to reproduce)

        Returns:
            Dict with success status and feedback_id
        """
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
        """
        Submit debate quality feedback.

        Convenience method for providing feedback on a specific debate.

        Args:
            debate_id: ID of the debate being reviewed
            comment: Quality feedback comment
            score: Optional quality score

        Returns:
            Dict with success status and feedback_id
        """
        return self.submit_feedback(
            comment=comment,
            feedback_type="debate_quality",
            score=score,
            context={"debate_id": debate_id},
        )

    # ===========================================================================
    # Feedback Prompts

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

