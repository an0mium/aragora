"""
Tests for Onboarding Flow Repository & Quick-Start (6D).

Tests cover:
- OnboardingRepository CRUD operations (create, get, update, complete)
- SQLite persistence and schema initialization
- OnboardingHandler routing and endpoint dispatch
- 8-step onboarding flow state machine
- Quick-start profile application
- Template recommendations
- Analytics tracking
- Handler registration verification
- Edge cases (duplicate flows, missing flows, invalid steps)
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.onboarding import (
    OnboardingHandler,
    OnboardingState,
    OnboardingStep,
    QuickStartProfile,
    QUICK_START_CONFIGS,
    STARTER_TEMPLATES,
    UseCase,
    _get_next_step,
    _get_recommended_templates,
    _get_step_order,
    _onboarding_flows,
    _onboarding_lock,
    get_onboarding_handlers,
    handle_get_flow,
    handle_init_flow,
    handle_quick_start,
    handle_update_step,
    handle_get_templates,
    handle_first_debate,
    handle_analytics,
)
from aragora.storage.repositories.onboarding import (
    OnboardingRepository,
    get_onboarding_repository,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    return tmp_path / "test_onboarding.db"


@pytest.fixture
def repo(temp_db):
    """Create an OnboardingRepository with a temp database."""
    return OnboardingRepository(db_path=temp_db)


@pytest.fixture(autouse=True)
def clear_in_memory_state():
    """Clear in-memory onboarding state between tests."""
    with _onboarding_lock:
        _onboarding_flows.clear()
    yield
    with _onboarding_lock:
        _onboarding_flows.clear()


@pytest.fixture
def handler():
    """Create an OnboardingHandler instance."""
    return OnboardingHandler({})


def _parse_response(result) -> dict[str, Any]:
    """Parse HandlerResult body as JSON."""
    if isinstance(result.body, bytes):
        return json.loads(result.body.decode("utf-8"))
    return json.loads(result.body)


# ===========================================================================
# Repository Tests
# ===========================================================================


class TestOnboardingRepository:
    """Test OnboardingRepository CRUD operations."""

    def test_create_flow(self, repo):
        flow_id = repo.create_flow(
            user_id="user-1",
            org_id="org-1",
            current_step="welcome",
            use_case="team_decisions",
        )
        assert flow_id is not None
        assert len(flow_id) > 0

    def test_get_flow_by_user(self, repo):
        repo.create_flow(user_id="user-1", org_id="org-1", current_step="welcome")
        flow = repo.get_flow("user-1", "org-1")
        assert flow is not None
        assert flow["user_id"] == "user-1"
        assert flow["org_id"] == "org-1"
        assert flow["current_step"] == "welcome"

    def test_get_flow_returns_none_for_missing(self, repo):
        result = repo.get_flow("nonexistent", None)
        assert result is None

    def test_get_flow_personal(self, repo):
        repo.create_flow(user_id="user-2", org_id=None, current_step="use_case")
        flow = repo.get_flow("user-2", None)
        assert flow is not None
        assert flow["current_step"] == "use_case"

    def test_update_flow(self, repo):
        flow_id = repo.create_flow(user_id="user-3", org_id=None, current_step="welcome")
        updated = repo.update_flow(
            flow_id,
            {
                "current_step": "use_case",
                "completed_steps": ["welcome"],
                "use_case": "security_audit",
            },
        )
        assert updated is True

        flow = repo.get_flow("user-3", None)
        assert flow["current_step"] == "use_case"
        assert "welcome" in flow["completed_steps"]
        assert flow["use_case"] == "security_audit"

    def test_update_flow_not_found(self, repo):
        updated = repo.update_flow("nonexistent", {"current_step": "welcome"})
        assert updated is False

    def test_complete_flow(self, repo):
        flow_id = repo.create_flow(user_id="user-4", org_id=None, current_step="completion")
        completed = repo.complete_flow(flow_id)
        assert completed is True

        flow = repo.get_flow("user-4", None)
        assert flow["completed_at"] is not None

    def test_get_flow_by_id(self, repo):
        flow_id = repo.create_flow(user_id="user-5", org_id="org-5", current_step="welcome")
        flow = repo.get_flow_by_id(flow_id)
        assert flow is not None
        assert flow["id"] == flow_id

    def test_get_flow_by_id_not_found(self, repo):
        result = repo.get_flow_by_id("nonexistent-id")
        assert result is None

    def test_metadata_persistence(self, repo):
        flow_id = repo.create_flow(
            user_id="user-6",
            org_id=None,
            current_step="welcome",
            metadata={"quick_start": True, "profile": "developer"},
        )
        flow = repo.get_flow_by_id(flow_id)
        assert flow["metadata"]["quick_start"] is True
        assert flow["metadata"]["profile"] == "developer"

    def test_analytics(self, repo):
        repo.create_flow(user_id="u1", org_id=None, current_step="welcome")
        repo.create_flow(user_id="u2", org_id=None, current_step="completion")
        analytics = repo.get_analytics()
        assert analytics["total_flows"] == 2

    def test_update_ignores_disallowed_fields(self, repo):
        flow_id = repo.create_flow(user_id="user-7", org_id=None, current_step="welcome")
        updated = repo.update_flow(flow_id, {"user_id": "hacker", "current_step": "use_case"})
        assert updated is True
        flow = repo.get_flow_by_id(flow_id)
        assert flow["user_id"] == "user-7"  # Not changed
        assert flow["current_step"] == "use_case"  # Changed


# ===========================================================================
# Step Order & Navigation Tests
# ===========================================================================


class TestStepNavigation:
    """Test onboarding step ordering and navigation."""

    def test_step_order_has_8_steps(self):
        steps = _get_step_order()
        assert len(steps) == 8

    def test_step_order_starts_with_welcome(self):
        steps = _get_step_order()
        assert steps[0] == OnboardingStep.WELCOME

    def test_step_order_ends_with_completion(self):
        steps = _get_step_order()
        assert steps[-1] == OnboardingStep.COMPLETION

    def test_get_next_step(self):
        assert _get_next_step(OnboardingStep.WELCOME) == OnboardingStep.USE_CASE
        assert _get_next_step(OnboardingStep.USE_CASE) == OnboardingStep.ORGANIZATION

    def test_get_next_step_at_completion(self):
        result = _get_next_step(OnboardingStep.COMPLETION)
        assert result == OnboardingStep.COMPLETION  # Stays at completion


# ===========================================================================
# Template Recommendation Tests
# ===========================================================================


class TestTemplateRecommendation:
    """Test template recommendation logic."""

    def test_all_templates_returns_list(self):
        templates = _get_recommended_templates(None)
        assert len(templates) > 0
        assert len(templates) <= 8

    def test_use_case_filter(self):
        templates = _get_recommended_templates(UseCase.SECURITY_AUDIT.value)
        # Security templates should appear first
        if templates:
            first_template = templates[0]
            assert UseCase.SECURITY_AUDIT.value in first_template.use_cases

    def test_starter_templates_have_required_fields(self):
        for t in STARTER_TEMPLATES:
            assert t.id
            assert t.name
            assert t.description
            assert t.agents_count > 0
            assert t.rounds > 0


# ===========================================================================
# Quick-Start Profile Tests
# ===========================================================================


class TestQuickStartProfiles:
    """Test quick-start profile configurations."""

    def test_all_profiles_have_configs(self):
        for profile in QuickStartProfile:
            assert profile.value in QUICK_START_CONFIGS

    def test_configs_have_required_keys(self):
        for name, config in QUICK_START_CONFIGS.items():
            assert "default_template" in config
            assert "suggested_templates" in config
            assert "default_agents" in config
            assert "default_rounds" in config
            assert "focus_areas" in config

    def test_sme_profile_has_budget(self):
        sme_config = QUICK_START_CONFIGS["sme"]
        assert sme_config.get("budget_enabled") is True
        assert sme_config.get("max_debates_free", 0) > 0


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestOnboardingHandlerRouting:
    """Test OnboardingHandler can_handle and routing."""

    def test_can_handle_v1_flow(self, handler):
        assert handler.can_handle("/api/v1/onboarding/flow") is True

    def test_can_handle_v1_templates(self, handler):
        assert handler.can_handle("/api/v1/onboarding/templates") is True

    def test_can_handle_v1_quick_start(self, handler):
        assert handler.can_handle("/api/v1/onboarding/quick-start") is True

    def test_can_handle_v1_analytics(self, handler):
        assert handler.can_handle("/api/v1/onboarding/analytics") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    async def test_routes_get_flow(self, handler):
        with patch(
            "aragora.server.handlers.onboarding.get_onboarding_repository"
        ) as mock_repo_fn:
            mock_repo = MagicMock()
            mock_repo.get_flow.return_value = None
            mock_repo_fn.return_value = mock_repo

            result = await handler.handle(
                "/api/v1/onboarding/flow",
                "GET",
                user_id="test-user",
            )
            assert result.status_code == 200

    async def test_routes_unknown_returns_404(self, handler):
        result = await handler.handle(
            "/api/v1/onboarding/unknown-endpoint",
            "GET",
            user_id="test-user",
        )
        assert result.status_code == 404


# ===========================================================================
# Handler Function Tests
# ===========================================================================


class TestHandlerFunctions:
    """Test individual onboarding handler functions."""

    async def test_get_flow_no_existing(self):
        with patch(
            "aragora.server.handlers.onboarding.get_onboarding_repository"
        ) as mock_repo_fn:
            mock_repo = MagicMock()
            mock_repo.get_flow.return_value = None
            mock_repo_fn.return_value = mock_repo

            result = await handle_get_flow(user_id="new-user")
            body = _parse_response(result)
            # success_response wraps in {"success": true, "data": {...}}
            data = body.get("data", body)
            assert data["needs_onboarding"] is True
            assert data["exists"] is False

    async def test_init_flow_creates_state(self):
        with patch(
            "aragora.server.handlers.onboarding.get_onboarding_repository"
        ) as mock_repo_fn:
            mock_repo = MagicMock()
            mock_repo.create_flow.return_value = "flow-123"
            mock_repo_fn.return_value = mock_repo

            result = await handle_init_flow(
                data={"use_case": "team_decisions"},
                user_id="user-1",
            )
            body = _parse_response(result)
            data = body.get("data", body)
            assert "flow_id" in data
            assert data["use_case"] == "team_decisions"
            assert data["current_step"] == "welcome"

    async def test_init_flow_with_quick_start_profile(self):
        with patch(
            "aragora.server.handlers.onboarding.get_onboarding_repository"
        ) as mock_repo_fn:
            mock_repo = MagicMock()
            mock_repo.create_flow.return_value = "flow-456"
            mock_repo_fn.return_value = mock_repo

            result = await handle_init_flow(
                data={"quick_start_profile": "developer"},
                user_id="user-2",
            )
            body = _parse_response(result)
            data = body.get("data", body)
            assert data["quick_start_profile"] == "developer"

    async def test_update_step_next(self):
        # First create a flow in memory
        with _onboarding_lock:
            _onboarding_flows["user-3:personal"] = OnboardingState(
                id="flow-test",
                user_id="user-3",
                organization_id=None,
                current_step=OnboardingStep.WELCOME,
                completed_steps=[],
                use_case=None,
                selected_template_id=None,
                first_debate_id=None,
                quick_start_profile=None,
                team_invites=[],
                started_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                completed_at=None,
            )

        with patch("aragora.server.handlers.onboarding._sync_flow_to_repo"):
            result = await handle_update_step(
                data={"action": "next"},
                user_id="user-3",
            )
            body = _parse_response(result)
            data = body.get("data", body)
            assert data["current_step"] == "use_case"
            assert "welcome" in data["completed_steps"]

    async def test_update_step_skip(self):
        with _onboarding_lock:
            _onboarding_flows["user-4:personal"] = OnboardingState(
                id="flow-skip",
                user_id="user-4",
                organization_id=None,
                current_step=OnboardingStep.ORGANIZATION,
                completed_steps=["welcome", "use_case"],
                use_case="general",
                selected_template_id=None,
                first_debate_id=None,
                quick_start_profile=None,
                team_invites=[],
                started_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                completed_at=None,
            )

        with patch("aragora.server.handlers.onboarding._sync_flow_to_repo"):
            result = await handle_update_step(
                data={"action": "skip"},
                user_id="user-4",
            )
            body = _parse_response(result)
            data = body.get("data", body)
            assert data["is_skipped"] is True
            assert data["is_complete"] is True

    async def test_update_step_no_flow_returns_404(self):
        result = await handle_update_step(
            data={"action": "next"},
            user_id="nonexistent-user",
        )
        assert result.status_code == 404


# ===========================================================================
# Handler Registration Tests
# ===========================================================================


class TestHandlerRegistration:
    """Test handler registration and export."""

    def test_get_onboarding_handlers_returns_dict(self):
        handlers = get_onboarding_handlers()
        assert isinstance(handlers, dict)
        assert "get_flow" in handlers
        assert "init_flow" in handlers
        assert "update_step" in handlers
        assert "get_templates" in handlers
        assert "first_debate" in handlers
        assert "quick_start" in handlers
        assert "analytics" in handlers

    def test_handler_class_has_routes(self):
        assert len(OnboardingHandler.ROUTES) > 0
        assert "/api/onboarding/flow" in OnboardingHandler.ROUTES

    def test_onboarding_handler_importable_from_package(self):
        from aragora.server.handlers import OnboardingHandler as ImportedHandler

        assert ImportedHandler is OnboardingHandler

    def test_handler_registry_contains_onboarding(self):
        from aragora.server.handler_registry.admin import OnboardingHandler as RegHandler

        assert RegHandler is not None
