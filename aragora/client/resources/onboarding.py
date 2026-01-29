"""
Onboarding API resource for the Aragora client.

Provides methods for user onboarding:
- Onboarding flow management
- Template recommendations
- First debate assistance
- Quick-start configurations
- Analytics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class StarterTemplate:
    """A template recommended for onboarding."""

    id: str
    name: str
    description: str
    use_cases: list[str]
    agents_count: int
    rounds: int
    estimated_minutes: int
    example_prompt: str
    tags: list[str] = field(default_factory=list)
    difficulty: str = "beginner"


@dataclass
class OnboardingFlow:
    """Complete onboarding flow state."""

    id: str
    current_step: str
    completed_steps: list[str]
    use_case: str | None = None
    selected_template_id: str | None = None
    first_debate_id: str | None = None
    quick_start_profile: str | None = None
    team_invites_count: int = 0
    progress_percentage: int = 0
    started_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    skipped: bool = False


@dataclass
class QuickStartConfig:
    """A quick-start profile configuration."""

    profile: str
    default_template: str
    suggested_templates: list[str]
    default_agents: list[str]
    default_rounds: int
    focus_areas: list[str]


@dataclass
class OnboardingAnalytics:
    """Onboarding funnel analytics."""

    started: int
    first_debate: int
    completed: int
    completion_rate: float
    step_completion: dict[str, int] = field(default_factory=dict)
    total_events: int = 0


class OnboardingAPI:
    """API interface for user onboarding."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Flow Management
    # =========================================================================

    def get_flow(
        self,
        user_id: str | None = None,
        organization_id: str | None = None,
    ) -> tuple[bool, OnboardingFlow | None, list[StarterTemplate]]:
        """
        Get current onboarding flow state.

        Args:
            user_id: Optional user ID.
            organization_id: Optional organization ID.

        Returns:
            Tuple of (needs_onboarding, flow if exists, recommended templates).
        """
        params: dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        if organization_id:
            params["organization_id"] = organization_id

        response = self._client._get("/api/v1/onboarding/flow", params=params)

        flow = None
        if response.get("exists"):
            flow = self._parse_flow(response.get("flow", {}))

        templates = [self._parse_template(t) for t in response.get("recommended_templates", [])]

        return response.get("needs_onboarding", True), flow, templates

    async def get_flow_async(
        self,
        user_id: str | None = None,
        organization_id: str | None = None,
    ) -> tuple[bool, OnboardingFlow | None, list[StarterTemplate]]:
        """Async version of get_flow()."""
        params: dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        if organization_id:
            params["organization_id"] = organization_id

        response = await self._client._get_async("/api/v1/onboarding/flow", params=params)

        flow = None
        if response.get("exists"):
            flow = self._parse_flow(response.get("flow", {}))

        templates = [self._parse_template(t) for t in response.get("recommended_templates", [])]

        return response.get("needs_onboarding", True), flow, templates

    def init_flow(
        self,
        use_case: str | None = None,
        quick_start_profile: str | None = None,
        skip_to_step: str | None = None,
    ) -> OnboardingFlow:
        """
        Initialize a new onboarding flow.

        Args:
            use_case: Initial use case selection.
            quick_start_profile: Quick-start profile to apply.
            skip_to_step: Step to skip to.

        Returns:
            OnboardingFlow object.
        """
        body: dict[str, Any] = {}
        if use_case:
            body["use_case"] = use_case
        if quick_start_profile:
            body["quick_start_profile"] = quick_start_profile
        if skip_to_step:
            body["skip_to_step"] = skip_to_step

        response = self._client._post("/api/v1/onboarding/flow", body)
        return self._parse_flow(response)

    async def init_flow_async(
        self,
        use_case: str | None = None,
        quick_start_profile: str | None = None,
        skip_to_step: str | None = None,
    ) -> OnboardingFlow:
        """Async version of init_flow()."""
        body: dict[str, Any] = {}
        if use_case:
            body["use_case"] = use_case
        if quick_start_profile:
            body["quick_start_profile"] = quick_start_profile
        if skip_to_step:
            body["skip_to_step"] = skip_to_step

        response = await self._client._post_async("/api/v1/onboarding/flow", body)
        return self._parse_flow(response)

    def update_step(
        self,
        action: str = "next",
        step_data: Optional[dict[str, Any]] = None,
        jump_to_step: str | None = None,
    ) -> dict[str, Any]:
        """
        Update onboarding step progress.

        Args:
            action: Action to take (next, previous, complete, skip).
            step_data: Data collected in current step.
            jump_to_step: Specific step to jump to.

        Returns:
            Updated flow state.
        """
        body: dict[str, Any] = {"action": action}
        if step_data:
            body["step_data"] = step_data
        if jump_to_step:
            body["jump_to_step"] = jump_to_step

        return self._client._put("/api/v1/onboarding/flow/step", body)

    async def update_step_async(
        self,
        action: str = "next",
        step_data: Optional[dict[str, Any]] = None,
        jump_to_step: str | None = None,
    ) -> dict[str, Any]:
        """Async version of update_step()."""
        body: dict[str, Any] = {"action": action}
        if step_data:
            body["step_data"] = step_data
        if jump_to_step:
            body["jump_to_step"] = jump_to_step

        return await self._client._put_async("/api/v1/onboarding/flow/step", body)

    def skip_onboarding(self) -> dict[str, Any]:
        """
        Skip the onboarding flow.

        Returns:
            Updated flow state.
        """
        return self.update_step(action="skip")

    async def skip_onboarding_async(self) -> dict[str, Any]:
        """Async version of skip_onboarding()."""
        return await self.update_step_async(action="skip")

    def complete_onboarding(self) -> dict[str, Any]:
        """
        Mark onboarding as complete.

        Returns:
            Updated flow state.
        """
        return self.update_step(action="complete")

    async def complete_onboarding_async(self) -> dict[str, Any]:
        """Async version of complete_onboarding()."""
        return await self.update_step_async(action="complete")

    # =========================================================================
    # Templates
    # =========================================================================

    def get_templates(
        self,
        use_case: str | None = None,
        profile: str | None = None,
    ) -> list[StarterTemplate]:
        """
        Get recommended starter templates.

        Args:
            use_case: Filter by use case.
            profile: Quick-start profile to prioritize.

        Returns:
            List of StarterTemplate objects.
        """
        params: dict[str, Any] = {}
        if use_case:
            params["use_case"] = use_case
        if profile:
            params["profile"] = profile

        response = self._client._get("/api/v1/onboarding/templates", params=params)
        return [self._parse_template(t) for t in response.get("templates", [])]

    async def get_templates_async(
        self,
        use_case: str | None = None,
        profile: str | None = None,
    ) -> list[StarterTemplate]:
        """Async version of get_templates()."""
        params: dict[str, Any] = {}
        if use_case:
            params["use_case"] = use_case
        if profile:
            params["profile"] = profile

        response = await self._client._get_async("/api/v1/onboarding/templates", params=params)
        return [self._parse_template(t) for t in response.get("templates", [])]

    # =========================================================================
    # First Debate
    # =========================================================================

    def start_first_debate(
        self,
        template_id: str | None = None,
        topic: str | None = None,
        use_example: bool = False,
    ) -> dict[str, Any]:
        """
        Start a guided first debate.

        Args:
            template_id: Template to use.
            topic: Custom topic (if not using template).
            use_example: Use template's example prompt.

        Returns:
            Debate creation result.
        """
        body: dict[str, Any] = {}
        if template_id:
            body["template_id"] = template_id
        if topic:
            body["topic"] = topic
        if use_example:
            body["use_example"] = use_example

        return self._client._post("/api/v1/onboarding/first-debate", body)

    async def start_first_debate_async(
        self,
        template_id: str | None = None,
        topic: str | None = None,
        use_example: bool = False,
    ) -> dict[str, Any]:
        """Async version of start_first_debate()."""
        body: dict[str, Any] = {}
        if template_id:
            body["template_id"] = template_id
        if topic:
            body["topic"] = topic
        if use_example:
            body["use_example"] = use_example

        return await self._client._post_async("/api/v1/onboarding/first-debate", body)

    # =========================================================================
    # Quick-Start
    # =========================================================================

    def apply_quick_start(self, profile: str) -> dict[str, Any]:
        """
        Apply quick-start configuration.

        Args:
            profile: Quick-start profile (developer, security, executive, product, compliance).

        Returns:
            Applied configuration and next action.
        """
        body = {"profile": profile}
        return self._client._post("/api/v1/onboarding/quick-start", body)

    async def apply_quick_start_async(self, profile: str) -> dict[str, Any]:
        """Async version of apply_quick_start()."""
        body = {"profile": profile}
        return await self._client._post_async("/api/v1/onboarding/quick-start", body)

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_analytics(self, organization_id: str | None = None) -> OnboardingAnalytics:
        """
        Get onboarding funnel analytics.

        Args:
            organization_id: Optional organization ID filter.

        Returns:
            OnboardingAnalytics object.
        """
        params: dict[str, Any] = {}
        if organization_id:
            params["organization_id"] = organization_id

        response = self._client._get("/api/v1/onboarding/analytics", params=params)
        return self._parse_analytics(response)

    async def get_analytics_async(self, organization_id: str | None = None) -> OnboardingAnalytics:
        """Async version of get_analytics()."""
        params: dict[str, Any] = {}
        if organization_id:
            params["organization_id"] = organization_id

        response = await self._client._get_async("/api/v1/onboarding/analytics", params=params)
        return self._parse_analytics(response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_flow(self, data: dict[str, Any]) -> OnboardingFlow:
        """Parse flow data into OnboardingFlow object."""
        started_at = None
        updated_at = None
        completed_at = None

        if data.get("started_at"):
            try:
                started_at = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return OnboardingFlow(
            id=data.get("id", data.get("flow_id", "")),
            current_step=data.get("current_step", "welcome"),
            completed_steps=data.get("completed_steps", []),
            use_case=data.get("use_case"),
            selected_template_id=data.get("selected_template_id"),
            first_debate_id=data.get("first_debate_id"),
            quick_start_profile=data.get("quick_start_profile"),
            team_invites_count=data.get("team_invites_count", 0),
            progress_percentage=data.get("progress_percentage", 0),
            started_at=started_at,
            updated_at=updated_at,
            completed_at=completed_at,
            skipped=data.get("skipped", False),
        )

    def _parse_template(self, data: dict[str, Any]) -> StarterTemplate:
        """Parse template data into StarterTemplate object."""
        return StarterTemplate(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            use_cases=data.get("use_cases", []),
            agents_count=data.get("agents_count", 3),
            rounds=data.get("rounds", 2),
            estimated_minutes=data.get("estimated_minutes", 5),
            example_prompt=data.get("example_prompt", ""),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "beginner"),
        )

    def _parse_analytics(self, data: dict[str, Any]) -> OnboardingAnalytics:
        """Parse analytics data into OnboardingAnalytics object."""
        funnel = data.get("funnel", {})
        return OnboardingAnalytics(
            started=funnel.get("started", 0),
            first_debate=funnel.get("first_debate", 0),
            completed=funnel.get("completed", 0),
            completion_rate=funnel.get("completion_rate", 0.0),
            step_completion=data.get("step_completion", {}),
            total_events=data.get("total_events", 0),
        )


__all__ = [
    "OnboardingAPI",
    "OnboardingFlow",
    "StarterTemplate",
    "QuickStartConfig",
    "OnboardingAnalytics",
]
