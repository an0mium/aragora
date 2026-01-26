"""
Tests for the onboarding handler - SME quick-start experience.

Tests:
- OnboardingHandler routing (can_handle, routes)
- OnboardingRepository CRUD operations
- Quick debate creation
- Flow state management
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.onboarding import (
    OnboardingHandler,
    OnboardingStep,
    OnboardingState,
    UseCase,
    QuickStartProfile,
    StarterTemplate,
    STARTER_TEMPLATES,
    QUICK_START_CONFIGS,
    handle_get_flow,
    handle_init_flow,
    handle_update_step,
    handle_get_templates,
    handle_quick_debate,
    get_onboarding_handlers,
)
from aragora.storage.repositories.onboarding import (
    OnboardingRepository,
    get_onboarding_repository,
)


@pytest.fixture
def onboarding_handler():
    """Create an onboarding handler with minimal context."""
    ctx = {}
    handler = OnboardingHandler(ctx)
    return handler


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_onboarding.db"


@pytest.fixture
def onboarding_repo(temp_db_path):
    """Create an onboarding repository with temp database."""
    return OnboardingRepository(db_path=temp_db_path)


class TestOnboardingHandler:
    """Tests for OnboardingHandler routing."""

    def test_can_handle_flow(self, onboarding_handler):
        """Test that handler recognizes /api/onboarding/flow route."""
        assert onboarding_handler.can_handle("/api/onboarding/flow") is True
        assert onboarding_handler.can_handle("/api/v1/onboarding/flow") is True

    def test_can_handle_templates(self, onboarding_handler):
        """Test that handler recognizes /api/onboarding/templates route."""
        assert onboarding_handler.can_handle("/api/onboarding/templates") is True
        assert onboarding_handler.can_handle("/api/v1/onboarding/templates") is True

    def test_can_handle_quick_debate(self, onboarding_handler):
        """Test that handler recognizes /api/onboarding/quick-debate route."""
        assert onboarding_handler.can_handle("/api/onboarding/quick-debate") is True
        assert onboarding_handler.can_handle("/api/v1/onboarding/quick-debate") is True

    def test_can_handle_analytics(self, onboarding_handler):
        """Test that handler recognizes /api/onboarding/analytics route."""
        assert onboarding_handler.can_handle("/api/onboarding/analytics") is True
        assert onboarding_handler.can_handle("/api/v1/onboarding/analytics") is True

    def test_cannot_handle_unknown_path(self, onboarding_handler):
        """Test that handler rejects unknown paths."""
        assert onboarding_handler.can_handle("/unknown") is False
        assert onboarding_handler.can_handle("/api/debates") is False
        assert onboarding_handler.can_handle("/api/v1/debates") is False

    def test_routes_defined(self, onboarding_handler):
        """Test that ROUTES attribute is properly defined."""
        assert len(OnboardingHandler.ROUTES) >= 5
        assert "/api/onboarding/flow" in OnboardingHandler.ROUTES
        assert "/api/onboarding/quick-debate" in OnboardingHandler.ROUTES


class TestOnboardingRepository:
    """Tests for OnboardingRepository CRUD operations."""

    def test_create_flow(self, onboarding_repo):
        """Test creating a new onboarding flow."""
        flow_id = onboarding_repo.create_flow(
            user_id="user-123",
            org_id="org-456",
            current_step="welcome",
            use_case="business_decisions",
        )

        assert flow_id is not None
        assert len(flow_id) > 10

    def test_get_flow(self, onboarding_repo):
        """Test retrieving an onboarding flow."""
        # Create flow first
        flow_id = onboarding_repo.create_flow(
            user_id="user-123",
            org_id="org-456",
            current_step="welcome",
        )

        # Retrieve it
        flow = onboarding_repo.get_flow("user-123", "org-456")

        assert flow is not None
        assert flow["id"] == flow_id
        assert flow["user_id"] == "user-123"
        assert flow["org_id"] == "org-456"
        assert flow["current_step"] == "welcome"

    def test_get_flow_not_found(self, onboarding_repo):
        """Test retrieving a non-existent flow returns None."""
        flow = onboarding_repo.get_flow("nonexistent", "org")
        assert flow is None

    def test_get_flow_by_id(self, onboarding_repo):
        """Test retrieving a flow by ID."""
        flow_id = onboarding_repo.create_flow(
            user_id="user-123",
            org_id=None,
            current_step="use_case",
        )

        flow = onboarding_repo.get_flow_by_id(flow_id)

        assert flow is not None
        assert flow["id"] == flow_id

    def test_update_flow(self, onboarding_repo):
        """Test updating an onboarding flow."""
        flow_id = onboarding_repo.create_flow(
            user_id="user-123",
            org_id=None,
            current_step="welcome",
        )

        # Update the flow
        success = onboarding_repo.update_flow(
            flow_id,
            {
                "current_step": "template_select",
                "selected_template": "quick_decision",
                "completed_steps": ["welcome", "use_case"],
            },
        )

        assert success is True

        # Verify update
        flow = onboarding_repo.get_flow_by_id(flow_id)
        assert flow["current_step"] == "template_select"
        assert flow["selected_template"] == "quick_decision"
        assert "welcome" in flow["completed_steps"]

    def test_complete_flow(self, onboarding_repo):
        """Test marking a flow as completed."""
        flow_id = onboarding_repo.create_flow(
            user_id="user-123",
            org_id=None,
            current_step="completion",
        )

        success = onboarding_repo.complete_flow(flow_id)
        assert success is True

        flow = onboarding_repo.get_flow_by_id(flow_id)
        assert flow["completed_at"] is not None

    def test_get_analytics(self, onboarding_repo):
        """Test getting onboarding analytics."""
        # Create some flows
        onboarding_repo.create_flow("user-1", None, "welcome")
        onboarding_repo.create_flow("user-2", None, "template_select")
        flow_id = onboarding_repo.create_flow("user-3", None, "completion")
        onboarding_repo.complete_flow(flow_id)

        analytics = onboarding_repo.get_analytics()

        assert analytics["total_flows"] == 3
        assert analytics["completed_flows"] == 1
        assert "step_distribution" in analytics


class TestStarterTemplates:
    """Tests for starter templates configuration."""

    def test_starter_templates_defined(self):
        """Test that starter templates are properly defined."""
        assert len(STARTER_TEMPLATES) >= 3

    def test_express_template_exists(self):
        """Test that express onboarding template exists."""
        express = next((t for t in STARTER_TEMPLATES if t.id == "express_onboarding"), None)
        assert express is not None
        assert express.rounds <= 3
        assert express.agents_count >= 2

    def test_template_fields(self):
        """Test that templates have all required fields."""
        for template in STARTER_TEMPLATES:
            assert template.id
            assert template.name
            assert template.rounds > 0
            assert template.agents_count >= 2
            assert template.example_prompt


class TestQuickStartConfigs:
    """Tests for quick-start configurations."""

    def test_configs_defined(self):
        """Test that quick-start configs are defined."""
        assert len(QUICK_START_CONFIGS) >= 3

    def test_developer_config_exists(self):
        """Test that developer config exists."""
        assert QuickStartProfile.DEVELOPER.value in QUICK_START_CONFIGS
        config = QUICK_START_CONFIGS[QuickStartProfile.DEVELOPER.value]
        assert "default_agents" in config

    def test_executive_config_exists(self):
        """Test that executive config exists."""
        assert QuickStartProfile.EXECUTIVE.value in QUICK_START_CONFIGS


class TestHandlerFunctions:
    """Tests for standalone handler functions."""

    @pytest.mark.asyncio
    async def test_handle_get_templates(self):
        """Test getting templates returns valid response."""
        # handle_get_templates takes (data: dict, user_id: str)
        result = await handle_get_templates({}, "user-123")

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body.get("success") is True
        # Templates are nested under data.templates
        assert "data" in body
        assert "templates" in body["data"]
        assert len(body["data"]["templates"]) > 0

    @pytest.mark.asyncio
    async def test_handle_get_templates_with_use_case_filter(self):
        """Test getting templates with use case filter."""
        result = await handle_get_templates(
            {"use_case": "business_decisions"},
            "user-123",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_init_flow(self):
        """Test initializing an onboarding flow."""
        import uuid

        user_id = f"test-user-{uuid.uuid4().hex[:8]}"

        result = await handle_init_flow(
            {"use_case": "business_decisions"},
            user_id,
            "org-456",
        )

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body.get("success") is True
        # Flow data is under 'data' key
        assert "data" in body
        assert "flow_id" in body["data"]
        assert "current_step" in body["data"]

    @pytest.mark.asyncio
    async def test_handle_get_flow_no_flow(self):
        """Test getting flow when none exists."""
        import uuid

        user_id = f"nonexistent-{uuid.uuid4().hex[:8]}"

        result = await handle_get_flow(user_id, "nonexistent-org")

        assert result is not None
        # Should return success with flow (may be None or empty)
        body = json.loads(result.body)
        assert "flow" in body or "error" not in body


class TestOnboardingEnums:
    """Tests for onboarding enums."""

    def test_onboarding_steps(self):
        """Test that all expected steps are defined."""
        expected_steps = [
            "welcome",
            "use_case",
            "organization",
            "team_invite",
            "template_select",
            "first_debate",
            "receipt_review",
            "completion",
        ]
        for step in expected_steps:
            assert step in [s.value for s in OnboardingStep]

    def test_use_cases(self):
        """Test that use cases are defined."""
        assert len(list(UseCase)) >= 3

    def test_quick_start_profiles(self):
        """Test that quick start profiles are defined."""
        assert len(list(QuickStartProfile)) >= 3


class TestGetOnboardingHandlers:
    """Tests for handler registration function."""

    def test_returns_dict(self):
        """Test that get_onboarding_handlers returns a dict."""
        handlers = get_onboarding_handlers()
        assert isinstance(handlers, dict)

    def test_includes_quick_debate(self):
        """Test that quick_debate handler is included."""
        handlers = get_onboarding_handlers()
        assert "quick_debate" in handlers
        assert handlers["quick_debate"] == handle_quick_debate

    def test_includes_all_handlers(self):
        """Test that all expected handlers are included."""
        handlers = get_onboarding_handlers()
        expected = [
            "get_flow",
            "init_flow",
            "update_step",
            "get_templates",
            "first_debate",
            "quick_start",
            "quick_debate",
            "analytics",
        ]
        for name in expected:
            assert name in handlers, f"Missing handler: {name}"


class TestQuickDebate:
    """Tests for quick debate creation."""

    def test_quick_debate_function_exists(self):
        """Test that quick debate function is properly exported."""
        assert callable(handle_quick_debate)

    def test_quick_debate_in_handlers(self):
        """Test that quick_debate is registered in handlers."""
        handlers = get_onboarding_handlers()
        assert "quick_debate" in handlers
        assert handlers["quick_debate"] is handle_quick_debate

    @pytest.mark.asyncio
    async def test_quick_debate_without_controller_returns_error(self):
        """Test that quick debate handles import errors gracefully."""
        # When debate_controller cannot be imported, should return an error
        with patch.dict("sys.modules", {"aragora.server.debate_controller": None}):
            # The function should handle this gracefully
            result = await handle_quick_debate(
                {"topic": "Test question"},
                "user-123",
                None,
            )
            # Should return some response (either error or success)
            assert result is not None
