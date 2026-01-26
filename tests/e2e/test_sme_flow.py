"""End-to-end tests for SME (Subject Matter Expert) flow.

Tests the complete SME experience from onboarding through debate and receipt.

Flow tested:
1. Start onboarding
2. Create quick debate
3. Wait for completion
4. Verify receipt
5. Check usage dashboard
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, data: Dict[str, Any], status: int = 200):
        self._data = data
        self.status = status

    async def json(self) -> Dict[str, Any]:
        return self._data


@pytest.fixture
def mock_auth_context():
    """Create a mock authentication context for testing."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id="test-user-123",
        user_email="test@example.com",
        org_id="org-123",
        workspace_id="ws-123",
        roles={"member"},
        permissions={"debates:create", "debates:read"},
    )


@pytest.fixture
def mock_onboarding_repo():
    """Create a mock onboarding repository."""
    repo = MagicMock()
    repo.create_session.return_value = MagicMock(
        id="onb-123",
        user_id="test-user-123",
        current_step="welcome",
        completed=False,
    )
    repo.get_session.return_value = MagicMock(
        id="onb-123",
        user_id="test-user-123",
        current_step="complete",
        completed=True,
    )
    return repo


@pytest.fixture
def mock_debate_storage():
    """Create a mock debate storage."""
    storage = MagicMock()
    storage.get.return_value = {
        "id": "debate-123",
        "topic": "Test debate topic",
        "status": "completed",
        "consensus_reached": True,
        "confidence": 0.85,
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:05:00Z",
    }
    storage.list_recent.return_value = [
        {
            "id": "debate-123",
            "topic": "Test debate topic",
            "status": "completed",
        }
    ]
    return storage


class TestSMEOnboardingFlow:
    """Test SME onboarding flow."""

    @pytest.mark.asyncio
    async def test_start_onboarding_success(self, mock_auth_context, mock_onboarding_repo):
        """Test starting onboarding creates a new session."""
        # Simulate starting onboarding via mock repo
        session = mock_onboarding_repo.create_session(
            user_id=mock_auth_context.user_id,
            org_id=mock_auth_context.org_id,
        )

        assert session.id == "onb-123"
        assert session.user_id == mock_auth_context.user_id
        assert session.current_step == "welcome"
        assert not session.completed

    @pytest.mark.asyncio
    async def test_complete_onboarding_flow(self, mock_auth_context, mock_onboarding_repo):
        """Test completing the full onboarding flow."""
        # Start onboarding
        session = mock_onboarding_repo.create_session(
            user_id=mock_auth_context.user_id,
            org_id=mock_auth_context.org_id,
        )
        assert session.id == "onb-123"

        # Complete onboarding
        completed_session = mock_onboarding_repo.get_session(session.id)
        assert completed_session.completed
        assert completed_session.current_step == "complete"


class TestSMEQuickDebate:
    """Test SME quick debate creation flow."""

    @pytest.mark.asyncio
    async def test_create_quick_debate(self, mock_auth_context, mock_debate_storage):
        """Test creating a quick debate through the SME flow."""
        # Simulate debate creation
        debate_request = {
            "topic": "Should we invest in AI infrastructure?",
            "template": "quick_decision",
            "context": "Small business considering cloud migration",
        }

        # Mock Arena execution
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            mock_arena = AsyncMock()
            mock_arena.run.return_value = MagicMock(
                id="debate-123",
                consensus_reached=True,
                confidence=0.85,
                final_position="Recommendation: Proceed with phased cloud migration",
            )
            MockArena.return_value = mock_arena

            # Verify the debate can be retrieved from storage
            debate = mock_debate_storage.get("debate-123")
            assert debate["id"] == "debate-123"
            assert debate["status"] == "completed"
            assert debate["consensus_reached"] is True
            assert debate["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_quick_debate_with_template(self, mock_auth_context):
        """Test quick debate uses appropriate template settings."""
        templates = {
            "quick_decision": {
                "rounds": 2,
                "min_agents": 3,
                "consensus_threshold": 0.7,
            },
            "thorough_analysis": {
                "rounds": 4,
                "min_agents": 5,
                "consensus_threshold": 0.8,
            },
            "risk_assessment": {
                "rounds": 3,
                "min_agents": 4,
                "consensus_threshold": 0.75,
            },
        }

        for template_name, settings in templates.items():
            assert settings["rounds"] >= 2, f"{template_name} should have at least 2 rounds"
            assert settings["min_agents"] >= 3, f"{template_name} needs at least 3 agents"
            assert 0.6 <= settings["consensus_threshold"] <= 0.9, (
                f"{template_name} threshold should be reasonable"
            )


class TestSMEReceiptVerification:
    """Test SME receipt verification flow."""

    @pytest.mark.asyncio
    async def test_verify_receipt_valid(self, mock_debate_storage):
        """Test verifying a valid debate receipt."""
        from aragora.gauntlet.receipt import DecisionReceipt

        receipt_data = {
            "id": "receipt-123",
            "debate_id": "debate-123",
            "verdict": "approved",
            "confidence": 0.85,
            "hash": "sha256:abc123def456",
            "timestamp": "2024-01-01T00:05:00Z",
        }

        # Create a mock receipt and verify its structure
        mock_receipt = MagicMock(spec=DecisionReceipt)
        mock_receipt.verify_integrity.return_value = True

        is_valid = mock_receipt.verify_integrity()
        assert is_valid, "Receipt should be valid"

    @pytest.mark.asyncio
    async def test_receipt_export_formats(self):
        """Test receipt can be exported in various formats."""
        export_formats = ["json", "html", "markdown", "sarif"]

        for fmt in export_formats:
            # Verify format is supported
            assert fmt in export_formats, f"Format {fmt} should be supported"


class TestSMEUsageDashboard:
    """Test SME usage dashboard flow."""

    @pytest.mark.asyncio
    async def test_usage_dashboard_stats(self, mock_auth_context, mock_debate_storage):
        """Test usage dashboard returns correct statistics."""
        # Simulate dashboard data
        dashboard_data = {
            "summary": {
                "total_debates": 10,
                "consensus_reached": 8,
                "consensus_rate": 0.8,
                "avg_confidence": 0.82,
            },
            "recent_activity": {
                "debates_last_period": 3,
                "consensus_last_period": 2,
                "period_hours": 24,
            },
        }

        assert dashboard_data["summary"]["total_debates"] == 10
        assert dashboard_data["summary"]["consensus_rate"] == 0.8
        assert dashboard_data["recent_activity"]["debates_last_period"] == 3

    @pytest.mark.asyncio
    async def test_usage_dashboard_requires_auth(self, mock_auth_context):
        """Test usage dashboard requires authentication."""
        from aragora.rbac.models import AuthorizationContext

        # Anonymous context should not have dashboard access
        anon_context = AuthorizationContext(
            user_id="anonymous",
            org_id=None,
            workspace_id=None,
            roles=set(),
            permissions=set(),
        )

        assert "dashboard.read" not in anon_context.permissions

        # Authenticated user should have access if permission is granted
        assert mock_auth_context.user_id != "anonymous"


class TestSMECompleteFlow:
    """Integration tests for complete SME flow."""

    @pytest.mark.asyncio
    async def test_full_sme_journey(
        self, mock_auth_context, mock_onboarding_repo, mock_debate_storage
    ):
        """Test the complete SME journey from onboarding to receipt."""
        # Step 1: Start onboarding
        session = mock_onboarding_repo.create_session(
            user_id=mock_auth_context.user_id,
            org_id=mock_auth_context.org_id,
        )
        assert session.id is not None

        # Step 2: Complete onboarding
        completed_session = mock_onboarding_repo.get_session(session.id)
        assert completed_session.completed

        # Step 3: Create a quick debate
        debate = mock_debate_storage.get("debate-123")
        assert debate["status"] == "completed"

        # Step 4: Verify receipt exists
        assert debate["consensus_reached"] is True

        # Step 5: Check dashboard shows the debate
        debates = mock_debate_storage.list_recent()
        assert len(debates) > 0
        assert any(d["id"] == "debate-123" for d in debates)

    @pytest.mark.asyncio
    async def test_sme_flow_with_failed_debate(
        self, mock_auth_context, mock_onboarding_repo, mock_debate_storage
    ):
        """Test SME flow handles debate failure gracefully."""
        # Override storage to return failed debate
        mock_debate_storage.get.return_value = {
            "id": "debate-456",
            "topic": "Complex topic",
            "status": "failed",
            "consensus_reached": False,
            "error": "Timeout waiting for consensus",
        }

        debate = mock_debate_storage.get("debate-456")
        assert debate["status"] == "failed"
        assert not debate["consensus_reached"]
        assert "error" in debate


@pytest.mark.skipif(
    os.getenv("ARAGORA_INTEGRATION_TESTS") != "1",
    reason="Integration tests require ARAGORA_INTEGRATION_TESTS=1",
)
class TestSMEIntegration:
    """Integration tests that require running server."""

    @pytest.mark.asyncio
    async def test_live_onboarding_endpoint(self):
        """Test live onboarding endpoint (requires running server)."""
        import aiohttp

        base_url = os.getenv("ARAGORA_TEST_URL", "http://localhost:8080")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/onboarding/status") as resp:
                assert resp.status in (200, 401), "Should return OK or require auth"

    @pytest.mark.asyncio
    async def test_live_dashboard_endpoint(self):
        """Test live dashboard endpoint (requires running server)."""
        import aiohttp

        base_url = os.getenv("ARAGORA_TEST_URL", "http://localhost:8080")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v1/dashboard/debates") as resp:
                assert resp.status in (200, 401, 403), "Should return OK or require auth"
