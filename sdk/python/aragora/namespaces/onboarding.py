"""
Onboarding Namespace API

Provides methods for user onboarding:
- Onboarding flow management
- Template recommendations
- First debate assistance
- Quick-start configurations
- Analytics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

QuickStartProfile = Literal["developer", "security", "executive", "product", "compliance", "sme"]
FlowAction = Literal["next", "previous", "complete", "skip"]


class OnboardingAPI:
    """
    Synchronous Onboarding API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="your-key")
        >>> flow = client.onboarding.get_flow()
        >>> if flow.get("needs_onboarding"):
        ...     client.onboarding.init_flow(quick_start_profile="sme")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Flow Management
    # ===========================================================================

    def get_flow(
        self,
        user_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get current onboarding flow state.

        Args:
            user_id: Optional user ID to check flow for
            organization_id: Optional organization ID

        Returns:
            Flow state with needs_onboarding, exists, flow, recommended_templates
        """
        params = {}
        if user_id:
            params["user_id"] = user_id
        if organization_id:
            params["organization_id"] = organization_id
        return self._client.request("GET", "/api/v1/onboarding/flow", params=params or None)

    def init_flow(
        self,
        use_case: str | None = None,
        quick_start_profile: QuickStartProfile | None = None,
        skip_to_step: str | None = None,
    ) -> dict[str, Any]:
        """
        Initialize a new onboarding flow.

        Args:
            use_case: The user's primary use case
            quick_start_profile: Quick-start profile (developer, security, etc.)
            skip_to_step: Optional step to skip to

        Returns:
            Created onboarding flow
        """
        data = {}
        if use_case:
            data["use_case"] = use_case
        if quick_start_profile:
            data["quick_start_profile"] = quick_start_profile
        if skip_to_step:
            data["skip_to_step"] = skip_to_step
        return self._client.request("POST", "/api/v1/onboarding/flow", json=data or None)

    def update_step(
        self,
        action: FlowAction,
        step_data: dict[str, Any] | None = None,
        jump_to_step: str | None = None,
    ) -> dict[str, Any]:
        """
        Update onboarding step progress.

        Args:
            action: The action to take (next, previous, complete, skip)
            step_data: Optional data for the current step
            jump_to_step: Optional step to jump to

        Returns:
            Updated flow state with next_step
        """
        data: dict[str, Any] = {"action": action}
        if step_data:
            data["step_data"] = step_data
        if jump_to_step:
            data["jump_to_step"] = jump_to_step
        return self._client.request("PUT", "/api/v1/onboarding/flow/step", json=data)

    def skip(self) -> dict[str, Any]:
        """Skip the onboarding flow."""
        return self.update_step(action="skip")

    def complete(self) -> dict[str, Any]:
        """Mark onboarding as complete."""
        return self.update_step(action="complete")

    # ===========================================================================
    # Templates
    # ===========================================================================

    def get_templates(
        self,
        use_case: str | None = None,
        profile: QuickStartProfile | None = None,
    ) -> dict[str, Any]:
        """
        Get recommended starter templates.

        Args:
            use_case: Filter by use case
            profile: Filter by quick-start profile

        Returns:
            List of recommended templates
        """
        params = {}
        if use_case:
            params["use_case"] = use_case
        if profile:
            params["profile"] = profile
        return self._client.request("GET", "/api/v1/onboarding/templates", params=params or None)

    # ===========================================================================
    # First Debate
    # ===========================================================================

    def start_first_debate(
        self,
        template_id: str | None = None,
        topic: str | None = None,
        use_example: bool = False,
    ) -> dict[str, Any]:
        """
        Start a guided first debate.

        Creates a debate optimized for new users with guidance and
        simplified configuration.

        Args:
            template_id: Optional template to use
            topic: Custom topic (if not using example)
            use_example: Use the template's example topic

        Returns:
            Debate creation result with debate_id and guidance
        """
        data: dict[str, Any] = {}
        if template_id:
            data["template_id"] = template_id
        if topic:
            data["topic"] = topic
        if use_example:
            data["use_example"] = use_example
        return self._client.request("POST", "/api/v1/onboarding/first-debate", json=data or None)

    # ===========================================================================
    # Quick-Start
    # ===========================================================================

    def apply_quick_start(self, profile: QuickStartProfile) -> dict[str, Any]:
        """
        Apply quick-start configuration.

        Configures the workspace with defaults appropriate for the selected profile.

        Args:
            profile: The quick-start profile to apply

        Returns:
            Applied configuration details
        """
        return self._client.request(
            "POST",
            "/api/v1/onboarding/quick-start",
            json={"profile": profile},
        )

    def quick_debate(
        self,
        topic: str,
        template_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Start a quick one-click debate.

        Args:
            topic: The debate topic
            template_id: Optional template to use

        Returns:
            Quick debate result
        """
        data: dict[str, Any] = {"topic": topic}
        if template_id:
            data["template_id"] = template_id
        return self._client.request("POST", "/api/v1/onboarding/quick-debate", json=data)

    # ===========================================================================
    # Analytics
    # ===========================================================================

    def get_analytics(
        self,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get onboarding funnel analytics (admin only).

        Args:
            organization_id: Optional organization to get analytics for

        Returns:
            Onboarding analytics with completion rates
        """
        params = {}
        if organization_id:
            params["organization_id"] = organization_id
        return self._client.request("GET", "/api/v1/onboarding/analytics", params=params or None)


class AsyncOnboardingAPI:
    """
    Asynchronous Onboarding API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     flow = await client.onboarding.get_flow()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_flow(
        self,
        user_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """Get current onboarding flow state."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        if organization_id:
            params["organization_id"] = organization_id
        return await self._client.request("GET", "/api/v1/onboarding/flow", params=params or None)

    async def init_flow(
        self,
        use_case: str | None = None,
        quick_start_profile: QuickStartProfile | None = None,
        skip_to_step: str | None = None,
    ) -> dict[str, Any]:
        """Initialize a new onboarding flow."""
        data = {}
        if use_case:
            data["use_case"] = use_case
        if quick_start_profile:
            data["quick_start_profile"] = quick_start_profile
        if skip_to_step:
            data["skip_to_step"] = skip_to_step
        return await self._client.request("POST", "/api/v1/onboarding/flow", json=data or None)

    async def update_step(
        self,
        action: FlowAction,
        step_data: dict[str, Any] | None = None,
        jump_to_step: str | None = None,
    ) -> dict[str, Any]:
        """Update onboarding step progress."""
        data: dict[str, Any] = {"action": action}
        if step_data:
            data["step_data"] = step_data
        if jump_to_step:
            data["jump_to_step"] = jump_to_step
        return await self._client.request("PUT", "/api/v1/onboarding/flow/step", json=data)

    async def skip(self) -> dict[str, Any]:
        """Skip the onboarding flow."""
        return await self.update_step(action="skip")

    async def complete(self) -> dict[str, Any]:
        """Mark onboarding as complete."""
        return await self.update_step(action="complete")

    async def get_templates(
        self,
        use_case: str | None = None,
        profile: QuickStartProfile | None = None,
    ) -> dict[str, Any]:
        """Get recommended starter templates."""
        params = {}
        if use_case:
            params["use_case"] = use_case
        if profile:
            params["profile"] = profile
        return await self._client.request(
            "GET", "/api/v1/onboarding/templates", params=params or None
        )

    async def start_first_debate(
        self,
        template_id: str | None = None,
        topic: str | None = None,
        use_example: bool = False,
    ) -> dict[str, Any]:
        """Start a guided first debate."""
        data: dict[str, Any] = {}
        if template_id:
            data["template_id"] = template_id
        if topic:
            data["topic"] = topic
        if use_example:
            data["use_example"] = use_example
        return await self._client.request(
            "POST", "/api/v1/onboarding/first-debate", json=data or None
        )

    async def apply_quick_start(self, profile: QuickStartProfile) -> dict[str, Any]:
        """Apply quick-start configuration."""
        return await self._client.request(
            "POST",
            "/api/v1/onboarding/quick-start",
            json={"profile": profile},
        )

    async def quick_debate(
        self,
        topic: str,
        template_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a quick one-click debate."""
        data: dict[str, Any] = {"topic": topic}
        if template_id:
            data["template_id"] = template_id
        return await self._client.request("POST", "/api/v1/onboarding/quick-debate", json=data)

    async def get_analytics(
        self,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """Get onboarding funnel analytics (admin only)."""
        params = {}
        if organization_id:
            params["organization_id"] = organization_id
        return await self._client.request(
            "GET", "/api/v1/onboarding/analytics", params=params or None
        )
