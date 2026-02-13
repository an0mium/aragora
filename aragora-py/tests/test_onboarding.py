"""Tests for the Onboarding API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora_client.onboarding import (
    OnboardingFlow,
    OnboardingInvitation,
    OnboardingTemplate,
)


class TestOnboardingAPI:
    """Tests for OnboardingAPI methods."""

    @pytest.mark.asyncio
    async def test_get_flow(self, mock_client, mock_response):
        """Test getting the current onboarding flow."""
        response_data = {
            "id": "flow-123",
            "user_id": "user-123",
            "template": "default",
            "status": "in_progress",
            "steps": [
                {"id": "step-1", "name": "Welcome", "status": "completed", "order": 0},
                {
                    "id": "step-2",
                    "name": "Setup Profile",
                    "status": "in_progress",
                    "order": 1,
                },
            ],
            "current_step": "step-2",
            "progress_percent": 50.0,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.get_flow()

        assert isinstance(result, OnboardingFlow)
        assert result.id == "flow-123"
        assert result.progress_percent == 50.0
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_init_flow(self, mock_client, mock_response):
        """Test initializing a new onboarding flow."""
        response_data = {
            "id": "flow-new",
            "user_id": "user-123",
            "template": "enterprise",
            "status": "in_progress",
            "steps": [],
            "progress_percent": 0.0,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.init_flow(template="enterprise")

        assert isinstance(result, OnboardingFlow)
        assert result.template == "enterprise"

    @pytest.mark.asyncio
    async def test_update_step(self, mock_client, mock_response):
        """Test updating an onboarding step."""
        response_data = {
            "id": "flow-123",
            "user_id": "user-123",
            "template": "default",
            "status": "in_progress",
            "steps": [
                {"id": "step-1", "name": "Welcome", "status": "completed", "order": 0},
            ],
            "progress_percent": 25.0,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.update_step(
            "step-1", "completed", data={"time_spent": 60}
        )

        assert isinstance(result, OnboardingFlow)

    @pytest.mark.asyncio
    async def test_complete_step(self, mock_client, mock_response):
        """Test completing an onboarding step."""
        response_data = {
            "id": "flow-123",
            "user_id": "user-123",
            "template": "default",
            "status": "in_progress",
            "steps": [
                {"id": "step-1", "name": "Welcome", "status": "completed", "order": 0},
            ],
            "progress_percent": 50.0,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.complete_step("step-1")

        assert isinstance(result, OnboardingFlow)

    @pytest.mark.asyncio
    async def test_skip_step(self, mock_client, mock_response):
        """Test skipping an onboarding step."""
        response_data = {
            "id": "flow-123",
            "user_id": "user-123",
            "template": "default",
            "status": "in_progress",
            "steps": [
                {"id": "step-1", "name": "Welcome", "status": "skipped", "order": 0},
            ],
            "progress_percent": 25.0,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.skip_step("step-1")

        assert isinstance(result, OnboardingFlow)

    @pytest.mark.asyncio
    async def test_list_templates(self, mock_client, mock_response):
        """Test listing onboarding templates."""
        response_data = {
            "templates": [
                {
                    "id": "default",
                    "name": "Default Onboarding",
                    "description": "Standard user onboarding",
                    "estimated_minutes": 15,
                },
                {
                    "id": "enterprise",
                    "name": "Enterprise Onboarding",
                    "description": "For enterprise users",
                    "estimated_minutes": 30,
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.list_templates()

        assert len(result) == 2
        assert isinstance(result[0], OnboardingTemplate)
        assert result[0].name == "Default Onboarding"

    @pytest.mark.asyncio
    async def test_get_template(self, mock_client, mock_response):
        """Test getting a specific template."""
        response_data = {
            "id": "default",
            "name": "Default Onboarding",
            "description": "Standard user onboarding",
            "steps": [
                {"id": "welcome", "name": "Welcome"},
                {"id": "profile", "name": "Setup Profile"},
            ],
            "estimated_minutes": 15,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.get_template("default")

        assert isinstance(result, OnboardingTemplate)
        assert result.id == "default"
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_run_first_debate(self, mock_client, mock_response):
        """Test running the first debate."""
        response_data = {
            "debate_id": "debate-123",
            "status": "completed",
            "result": "The consensus reached was...",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.run_first_debate(
            task="What is the best programming language?",
            agents=["claude", "gpt4"],
        )

        assert result["debate_id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_quick_start(self, mock_client, mock_response):
        """Test quick start onboarding."""
        response_data = {
            "flow_id": "flow-123",
            "debate_id": "debate-123",
            "status": "completed",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.quick_start(
            task="Help me decide on a tech stack"
        )

        assert result["flow_id"] == "flow-123"

    @pytest.mark.asyncio
    async def test_get_analytics(self, mock_client, mock_response):
        """Test getting onboarding analytics."""
        response_data = {
            "total_flows": 100,
            "completed_flows": 85,
            "average_completion_time": 600,
            "dropout_rate": 0.15,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.get_analytics()

        assert result["total_flows"] == 100
        assert result["completed_flows"] == 85

    @pytest.mark.asyncio
    async def test_list_invitations(self, mock_client, mock_response):
        """Test listing onboarding invitations."""
        response_data = {
            "invitations": [
                {
                    "id": "inv-123",
                    "email": "user@example.com",
                    "template": "default",
                    "status": "pending",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.list_invitations()

        assert len(result) == 1
        assert isinstance(result[0], OnboardingInvitation)
        assert result[0].email == "user@example.com"

    @pytest.mark.asyncio
    async def test_create_invitation(self, mock_client, mock_response):
        """Test creating an invitation."""
        response_data = {
            "id": "inv-new",
            "email": "new@example.com",
            "template": "enterprise",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00Z",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.create_invitation(
            email="new@example.com",
            template="enterprise",
            message="Welcome to the team!",
        )

        assert isinstance(result, OnboardingInvitation)
        assert result.id == "inv-new"

    @pytest.mark.asyncio
    async def test_resend_invitation(self, mock_client, mock_response):
        """Test resending an invitation."""
        response_data = {
            "id": "inv-123",
            "email": "user@example.com",
            "template": "default",
            "status": "pending",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.onboarding.resend_invitation("inv-123")

        assert isinstance(result, OnboardingInvitation)

    @pytest.mark.asyncio
    async def test_revoke_invitation(self, mock_client, mock_response):
        """Test revoking an invitation."""
        mock_client._client.request = AsyncMock(return_value=mock_response(204, {}))

        # Should not raise
        await mock_client.onboarding.revoke_invitation("inv-123")
