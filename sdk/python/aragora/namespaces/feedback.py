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
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit NPS (Net Promoter Score) feedback.

        Args:
            score: NPS score from 0-10
                   - 0-6: Detractor
                   - 7-8: Passive
                   - 9-10: Promoter
            comment: Optional comment explaining the score
            context: Optional metadata (e.g., feature being used)

        Returns:
            Dict with success status and feedback_id

        Raises:
            ValidationError: If score is not 0-10
        """
        data: dict[str, Any] = {"score": score}
        if comment:
            data["comment"] = comment
        if context:
            data["context"] = context

        return self._client.request("POST", "/api/v1/feedback/nps", json=data)

    def get_nps_summary(self, days: int = 30) -> dict[str, Any]:
        """
        Get NPS summary analytics (admin only).

        Args:
            days: Number of days to include in summary (default: 30)

        Returns:
            Dict with:
            - nps_score: Overall NPS (-100 to 100)
            - total_responses: Number of responses
            - promoters: Count of 9-10 scores
            - passives: Count of 7-8 scores
            - detractors: Count of 0-6 scores
            - period_days: Period covered

        Requires: feedback.update permission (admin)
        """
        return self._client.request(
            "GET",
            "/api/v1/feedback/nps/summary",
            params={"days": days},
        )

    # ===========================================================================
    # General Feedback
    # ===========================================================================

    def submit_feedback(
        self,
        comment: str,
        feedback_type: FeedbackType = "general",
        score: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit general feedback.

        Args:
            comment: Feedback content (required)
            feedback_type: Type of feedback:
                - "feature_request": New feature idea
                - "bug_report": Bug or issue
                - "general": General feedback
                - "debate_quality": Feedback on debate quality
            score: Optional rating score
            context: Optional metadata

        Returns:
            Dict with success status and feedback_id
        """
        data: dict[str, Any] = {
            "type": feedback_type,
            "comment": comment,
        }
        if score is not None:
            data["score"] = score
        if context:
            data["context"] = context

        return self._client.request("POST", "/api/v1/feedback/general", json=data)

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

    # ===========================================================================
    # Feedback Prompts
    # ===========================================================================

    def get_prompts(self) -> dict[str, Any]:
        """
        Get active feedback prompts for the user.

        Returns prompts based on user activity and timing.

        Returns:
            Dict with prompts array containing question configs
        """
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
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit NPS (Net Promoter Score) feedback."""
        data: dict[str, Any] = {"score": score}
        if comment:
            data["comment"] = comment
        if context:
            data["context"] = context

        return await self._client.request("POST", "/api/v1/feedback/nps", json=data)

    async def get_nps_summary(self, days: int = 30) -> dict[str, Any]:
        """Get NPS summary analytics (admin only)."""
        return await self._client.request(
            "GET",
            "/api/v1/feedback/nps/summary",
            params={"days": days},
        )

    # ===========================================================================
    # General Feedback
    # ===========================================================================

    async def submit_feedback(
        self,
        comment: str,
        feedback_type: FeedbackType = "general",
        score: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit general feedback."""
        data: dict[str, Any] = {
            "type": feedback_type,
            "comment": comment,
        }
        if score is not None:
            data["score"] = score
        if context:
            data["context"] = context

        return await self._client.request("POST", "/api/v1/feedback/general", json=data)

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

    # ===========================================================================
    # Feedback Prompts
    # ===========================================================================

    async def get_prompts(self) -> dict[str, Any]:
        """Get active feedback prompts for the user."""
        return await self._client.request("GET", "/api/v1/feedback/prompts")
