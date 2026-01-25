"""Onboarding API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class OnboardingStep(BaseModel):
    """Onboarding step model."""

    id: str
    name: str
    description: str | None = None
    status: str = "pending"  # pending, in_progress, completed, skipped
    required: bool = True
    order: int = 0
    completed_at: str | None = None


class OnboardingFlow(BaseModel):
    """Onboarding flow model."""

    id: str
    user_id: str
    template: str = "default"
    status: str = "in_progress"
    steps: list[OnboardingStep] = []
    current_step: str | None = None
    progress_percent: float = 0.0
    started_at: str | None = None
    completed_at: str | None = None


class OnboardingTemplate(BaseModel):
    """Onboarding template model."""

    id: str
    name: str
    description: str | None = None
    steps: list[dict[str, Any]] = []
    estimated_minutes: int = 15
    target_audience: str | None = None


class UpdateStepRequest(BaseModel):
    """Update step request."""

    status: str
    data: dict[str, Any] | None = None


class OnboardingInvitation(BaseModel):
    """Onboarding invitation model."""

    id: str
    email: str
    template: str
    status: str = "pending"
    invited_by: str | None = None
    created_at: str | None = None
    expires_at: str | None = None
    accepted_at: str | None = None


class CreateInvitationRequest(BaseModel):
    """Create invitation request."""

    email: str
    template: str = "default"
    message: str | None = None
    expires_in_days: int = 7


class OnboardingAPI:
    """API for onboarding operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def get_flow(self) -> OnboardingFlow | None:
        """Get the current user's onboarding flow."""
        try:
            data = await self._client._get("/api/v1/onboarding/flow")
            return OnboardingFlow.model_validate(data)
        except Exception:
            return None

    async def init_flow(self, template: str = "default") -> OnboardingFlow:
        """Initialize a new onboarding flow for the current user."""
        data = await self._client._post(
            "/api/v1/onboarding/flow", {"template": template}
        )
        return OnboardingFlow.model_validate(data)

    async def update_step(
        self,
        step_id: str,
        status: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> OnboardingFlow:
        """Update an onboarding step."""
        request = UpdateStepRequest(status=status, data=data)
        result = await self._client._patch(
            f"/api/v1/onboarding/steps/{step_id}",
            request.model_dump(exclude_none=True),
        )
        return OnboardingFlow.model_validate(result)

    async def complete_step(
        self,
        step_id: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> OnboardingFlow:
        """Mark an onboarding step as completed."""
        return await self.update_step(step_id, "completed", data=data)

    async def skip_step(self, step_id: str) -> OnboardingFlow:
        """Skip an onboarding step."""
        return await self.update_step(step_id, "skipped")

    async def list_templates(self) -> list[OnboardingTemplate]:
        """List available onboarding templates."""
        data = await self._client._get("/api/v1/onboarding/templates")
        return [
            OnboardingTemplate.model_validate(t) for t in data.get("templates", [])
        ]

    async def get_template(self, template_id: str) -> OnboardingTemplate:
        """Get a specific onboarding template."""
        data = await self._client._get(f"/api/v1/onboarding/templates/{template_id}")
        return OnboardingTemplate.model_validate(data)

    async def run_first_debate(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the user's first debate as part of onboarding."""
        return await self._client._post(
            "/api/v1/onboarding/first-debate",
            {"task": task, "agents": agents},
        )

    async def quick_start(
        self,
        task: str,
        *,
        template: str = "default",
    ) -> dict[str, Any]:
        """Quick start onboarding with a single call."""
        return await self._client._post(
            "/api/v1/onboarding/quick-start",
            {"task": task, "template": template},
        )

    async def get_analytics(self) -> dict[str, Any]:
        """Get onboarding analytics for the organization."""
        return await self._client._get("/api/v1/onboarding/analytics")

    # Invitations
    async def list_invitations(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OnboardingInvitation]:
        """List onboarding invitations."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = await self._client._get("/api/v1/onboarding/invitations", params=params)
        return [
            OnboardingInvitation.model_validate(i) for i in data.get("invitations", [])
        ]

    async def create_invitation(
        self,
        email: str,
        *,
        template: str = "default",
        message: str | None = None,
        expires_in_days: int = 7,
    ) -> OnboardingInvitation:
        """Create an onboarding invitation."""
        request = CreateInvitationRequest(
            email=email,
            template=template,
            message=message,
            expires_in_days=expires_in_days,
        )
        data = await self._client._post(
            "/api/v1/onboarding/invitations", request.model_dump()
        )
        return OnboardingInvitation.model_validate(data)

    async def resend_invitation(self, invitation_id: str) -> OnboardingInvitation:
        """Resend an onboarding invitation."""
        data = await self._client._post(
            f"/api/v1/onboarding/invitations/{invitation_id}/resend", {}
        )
        return OnboardingInvitation.model_validate(data)

    async def revoke_invitation(self, invitation_id: str) -> None:
        """Revoke an onboarding invitation."""
        await self._client._delete(f"/api/v1/onboarding/invitations/{invitation_id}")
