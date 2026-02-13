"""Tests for OnboardingAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.onboarding import (
    OnboardingAnalytics,
    OnboardingAPI,
    OnboardingFlow,
    QuickStartConfig,
    StarterTemplate,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> OnboardingAPI:
    return OnboardingAPI(mock_client)


SAMPLE_TEMPLATE = {
    "id": "tpl-001",
    "name": "Quick Decision",
    "description": "A fast decision-making template",
    "use_cases": ["product", "engineering"],
    "agents_count": 3,
    "rounds": 2,
    "estimated_minutes": 5,
    "example_prompt": "Should we adopt microservices?",
    "tags": ["architecture", "strategy"],
    "difficulty": "beginner",
}

SAMPLE_FLOW = {
    "id": "flow-abc",
    "current_step": "select_template",
    "completed_steps": ["welcome", "use_case"],
    "use_case": "product_decisions",
    "selected_template_id": "tpl-001",
    "first_debate_id": None,
    "quick_start_profile": "developer",
    "team_invites_count": 2,
    "progress_percentage": 40,
    "started_at": "2026-01-15T10:00:00Z",
    "updated_at": "2026-01-15T10:05:00Z",
    "completed_at": None,
    "skipped": False,
}

SAMPLE_FLOW_RESPONSE = {
    "needs_onboarding": True,
    "exists": True,
    "flow": SAMPLE_FLOW,
    "recommended_templates": [SAMPLE_TEMPLATE],
}

SAMPLE_ANALYTICS_RESPONSE = {
    "funnel": {
        "started": 100,
        "first_debate": 60,
        "completed": 40,
        "completion_rate": 0.4,
    },
    "step_completion": {"welcome": 100, "use_case": 80, "template": 60},
    "total_events": 500,
}


class TestGetFlow:
    def test_get_flow_with_existing_flow(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_FLOW_RESPONSE
        needs, flow, templates = api.get_flow()
        assert needs is True
        assert flow is not None
        assert isinstance(flow, OnboardingFlow)
        assert flow.id == "flow-abc"
        assert flow.current_step == "select_template"
        assert len(templates) == 1
        assert isinstance(templates[0], StarterTemplate)
        mock_client._get.assert_called_once()

    def test_get_flow_no_existing_flow(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "needs_onboarding": True,
            "exists": False,
            "recommended_templates": [],
        }
        needs, flow, templates = api.get_flow()
        assert needs is True
        assert flow is None
        assert templates == []

    def test_get_flow_with_user_id(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_FLOW_RESPONSE
        api.get_flow(user_id="user-1")
        params = mock_client._get.call_args[1]["params"]
        assert params["user_id"] == "user-1"

    def test_get_flow_with_organization_id(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_FLOW_RESPONSE
        api.get_flow(organization_id="org-1")
        params = mock_client._get.call_args[1]["params"]
        assert params["organization_id"] == "org-1"

    def test_get_flow_with_both_ids(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_FLOW_RESPONSE
        api.get_flow(user_id="user-1", organization_id="org-1")
        params = mock_client._get.call_args[1]["params"]
        assert params["user_id"] == "user-1"
        assert params["organization_id"] == "org-1"

    def test_get_flow_no_params(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"needs_onboarding": True, "exists": False, "recommended_templates": []}
        api.get_flow()
        mock_client._get.assert_called_once_with("/api/v1/onboarding/flow", params={})

    @pytest.mark.asyncio
    async def test_get_flow_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_FLOW_RESPONSE)
        needs, flow, templates = await api.get_flow_async()
        assert needs is True
        assert flow is not None
        assert flow.id == "flow-abc"
        assert len(templates) == 1

    @pytest.mark.asyncio
    async def test_get_flow_async_with_user_id(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_FLOW_RESPONSE)
        await api.get_flow_async(user_id="user-2")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["user_id"] == "user-2"


class TestInitFlow:
    def test_init_flow_minimal(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FLOW
        flow = api.init_flow()
        assert isinstance(flow, OnboardingFlow)
        assert flow.id == "flow-abc"
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body == {}

    def test_init_flow_with_use_case(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FLOW
        api.init_flow(use_case="product_decisions")
        body = mock_client._post.call_args[0][1]
        assert body["use_case"] == "product_decisions"

    def test_init_flow_with_quick_start_profile(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FLOW
        api.init_flow(quick_start_profile="developer")
        body = mock_client._post.call_args[0][1]
        assert body["quick_start_profile"] == "developer"

    def test_init_flow_with_skip_to_step(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FLOW
        api.init_flow(skip_to_step="template")
        body = mock_client._post.call_args[0][1]
        assert body["skip_to_step"] == "template"

    def test_init_flow_with_all_options(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FLOW
        api.init_flow(use_case="security", quick_start_profile="compliance", skip_to_step="first_debate")
        body = mock_client._post.call_args[0][1]
        assert body["use_case"] == "security"
        assert body["quick_start_profile"] == "compliance"
        assert body["skip_to_step"] == "first_debate"

    def test_init_flow_calls_correct_endpoint(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_FLOW
        api.init_flow()
        assert mock_client._post.call_args[0][0] == "/api/v1/onboarding/flow"

    @pytest.mark.asyncio
    async def test_init_flow_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_FLOW)
        flow = await api.init_flow_async(use_case="engineering")
        assert isinstance(flow, OnboardingFlow)
        assert flow.id == "flow-abc"
        body = mock_client._post_async.call_args[0][1]
        assert body["use_case"] == "engineering"


class TestUpdateStep:
    def test_update_step_default_action(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"status": "ok"}
        result = api.update_step()
        assert result == {"status": "ok"}
        body = mock_client._put.call_args[0][1]
        assert body["action"] == "next"

    def test_update_step_previous(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"status": "ok"}
        api.update_step(action="previous")
        body = mock_client._put.call_args[0][1]
        assert body["action"] == "previous"

    def test_update_step_with_step_data(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"status": "ok"}
        api.update_step(step_data={"selected_template": "tpl-001"})
        body = mock_client._put.call_args[0][1]
        assert body["step_data"]["selected_template"] == "tpl-001"

    def test_update_step_with_jump(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"status": "ok"}
        api.update_step(jump_to_step="first_debate")
        body = mock_client._put.call_args[0][1]
        assert body["jump_to_step"] == "first_debate"

    def test_update_step_calls_correct_endpoint(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"status": "ok"}
        api.update_step()
        assert mock_client._put.call_args[0][0] == "/api/v1/onboarding/flow/step"

    @pytest.mark.asyncio
    async def test_update_step_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put_async = AsyncMock(return_value={"status": "ok"})
        result = await api.update_step_async(action="complete", step_data={"rating": 5})
        assert result == {"status": "ok"}
        body = mock_client._put_async.call_args[0][1]
        assert body["action"] == "complete"
        assert body["step_data"]["rating"] == 5


class TestSkipOnboarding:
    def test_skip_onboarding(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"skipped": True}
        result = api.skip_onboarding()
        assert result == {"skipped": True}
        body = mock_client._put.call_args[0][1]
        assert body["action"] == "skip"

    @pytest.mark.asyncio
    async def test_skip_onboarding_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put_async = AsyncMock(return_value={"skipped": True})
        result = await api.skip_onboarding_async()
        assert result == {"skipped": True}


class TestCompleteOnboarding:
    def test_complete_onboarding(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"completed": True}
        result = api.complete_onboarding()
        assert result == {"completed": True}
        body = mock_client._put.call_args[0][1]
        assert body["action"] == "complete"

    @pytest.mark.asyncio
    async def test_complete_onboarding_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._put_async = AsyncMock(return_value={"completed": True})
        result = await api.complete_onboarding_async()
        assert result == {"completed": True}


class TestGetTemplates:
    def test_get_templates_default(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"templates": [SAMPLE_TEMPLATE]}
        templates = api.get_templates()
        assert len(templates) == 1
        assert isinstance(templates[0], StarterTemplate)
        assert templates[0].id == "tpl-001"
        assert templates[0].name == "Quick Decision"

    def test_get_templates_with_use_case(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"templates": []}
        api.get_templates(use_case="security")
        params = mock_client._get.call_args[1]["params"]
        assert params["use_case"] == "security"

    def test_get_templates_with_profile(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"templates": []}
        api.get_templates(profile="executive")
        params = mock_client._get.call_args[1]["params"]
        assert params["profile"] == "executive"

    def test_get_templates_empty(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"templates": []}
        templates = api.get_templates()
        assert templates == []

    def test_get_templates_calls_correct_endpoint(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"templates": []}
        api.get_templates()
        mock_client._get.assert_called_once_with("/api/v1/onboarding/templates", params={})

    @pytest.mark.asyncio
    async def test_get_templates_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"templates": [SAMPLE_TEMPLATE]})
        templates = await api.get_templates_async(use_case="product")
        assert len(templates) == 1
        assert templates[0].name == "Quick Decision"


class TestStartFirstDebate:
    def test_start_first_debate_minimal(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"debate_id": "deb-001", "status": "started"}
        result = api.start_first_debate()
        assert result["debate_id"] == "deb-001"
        body = mock_client._post.call_args[0][1]
        assert body == {}

    def test_start_first_debate_with_template(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"debate_id": "deb-002"}
        api.start_first_debate(template_id="tpl-001")
        body = mock_client._post.call_args[0][1]
        assert body["template_id"] == "tpl-001"

    def test_start_first_debate_with_topic(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"debate_id": "deb-003"}
        api.start_first_debate(topic="Should we use Kubernetes?")
        body = mock_client._post.call_args[0][1]
        assert body["topic"] == "Should we use Kubernetes?"

    def test_start_first_debate_with_example(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"debate_id": "deb-004"}
        api.start_first_debate(template_id="tpl-001", use_example=True)
        body = mock_client._post.call_args[0][1]
        assert body["template_id"] == "tpl-001"
        assert body["use_example"] is True

    def test_start_first_debate_use_example_false_not_sent(
        self, api: OnboardingAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"debate_id": "deb-005"}
        api.start_first_debate(template_id="tpl-001", use_example=False)
        body = mock_client._post.call_args[0][1]
        assert "use_example" not in body

    def test_start_first_debate_calls_correct_endpoint(
        self, api: OnboardingAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        api.start_first_debate()
        assert mock_client._post.call_args[0][0] == "/api/v1/onboarding/first-debate"

    @pytest.mark.asyncio
    async def test_start_first_debate_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"debate_id": "deb-010"})
        result = await api.start_first_debate_async(topic="Test topic")
        assert result["debate_id"] == "deb-010"


class TestApplyQuickStart:
    def test_apply_quick_start(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"applied": True, "profile": "developer"}
        result = api.apply_quick_start("developer")
        assert result["applied"] is True
        body = mock_client._post.call_args[0][1]
        assert body["profile"] == "developer"

    def test_apply_quick_start_calls_correct_endpoint(
        self, api: OnboardingAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        api.apply_quick_start("security")
        assert mock_client._post.call_args[0][0] == "/api/v1/onboarding/quick-start"

    @pytest.mark.asyncio
    async def test_apply_quick_start_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"applied": True, "profile": "executive"})
        result = await api.apply_quick_start_async("executive")
        assert result["profile"] == "executive"


class TestGetAnalytics:
    def test_get_analytics(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_ANALYTICS_RESPONSE
        analytics = api.get_analytics()
        assert isinstance(analytics, OnboardingAnalytics)
        assert analytics.started == 100
        assert analytics.first_debate == 60
        assert analytics.completed == 40
        assert analytics.completion_rate == 0.4
        assert analytics.step_completion == {"welcome": 100, "use_case": 80, "template": 60}
        assert analytics.total_events == 500

    def test_get_analytics_with_organization(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_ANALYTICS_RESPONSE
        api.get_analytics(organization_id="org-1")
        params = mock_client._get.call_args[1]["params"]
        assert params["organization_id"] == "org-1"

    def test_get_analytics_calls_correct_endpoint(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_ANALYTICS_RESPONSE
        api.get_analytics()
        mock_client._get.assert_called_once_with("/api/v1/onboarding/analytics", params={})

    @pytest.mark.asyncio
    async def test_get_analytics_async(self, api: OnboardingAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_ANALYTICS_RESPONSE)
        analytics = await api.get_analytics_async(organization_id="org-2")
        assert analytics.started == 100
        assert analytics.completion_rate == 0.4


class TestParseFlow:
    def test_parse_flow_full(self, api: OnboardingAPI) -> None:
        flow = api._parse_flow(SAMPLE_FLOW)
        assert flow.id == "flow-abc"
        assert flow.current_step == "select_template"
        assert flow.completed_steps == ["welcome", "use_case"]
        assert flow.use_case == "product_decisions"
        assert flow.selected_template_id == "tpl-001"
        assert flow.first_debate_id is None
        assert flow.quick_start_profile == "developer"
        assert flow.team_invites_count == 2
        assert flow.progress_percentage == 40
        assert flow.started_at is not None
        assert flow.started_at.year == 2026
        assert flow.updated_at is not None
        assert flow.completed_at is None
        assert flow.skipped is False

    def test_parse_flow_minimal(self, api: OnboardingAPI) -> None:
        flow = api._parse_flow({})
        assert flow.id == ""
        assert flow.current_step == "welcome"
        assert flow.completed_steps == []
        assert flow.use_case is None
        assert flow.started_at is None

    def test_parse_flow_falls_back_to_flow_id(self, api: OnboardingAPI) -> None:
        flow = api._parse_flow({"flow_id": "fallback-id"})
        assert flow.id == "fallback-id"

    def test_parse_flow_invalid_datetime(self, api: OnboardingAPI) -> None:
        data = {**SAMPLE_FLOW, "started_at": "not-a-date"}
        flow = api._parse_flow(data)
        assert flow.started_at is None

    def test_parse_flow_completed_at(self, api: OnboardingAPI) -> None:
        data = {**SAMPLE_FLOW, "completed_at": "2026-01-15T11:00:00Z"}
        flow = api._parse_flow(data)
        assert flow.completed_at is not None
        assert flow.completed_at.year == 2026


class TestParseTemplate:
    def test_parse_template_full(self, api: OnboardingAPI) -> None:
        template = api._parse_template(SAMPLE_TEMPLATE)
        assert template.id == "tpl-001"
        assert template.name == "Quick Decision"
        assert template.description == "A fast decision-making template"
        assert template.use_cases == ["product", "engineering"]
        assert template.agents_count == 3
        assert template.rounds == 2
        assert template.estimated_minutes == 5
        assert template.example_prompt == "Should we adopt microservices?"
        assert template.tags == ["architecture", "strategy"]
        assert template.difficulty == "beginner"

    def test_parse_template_defaults(self, api: OnboardingAPI) -> None:
        template = api._parse_template({})
        assert template.id == ""
        assert template.name == ""
        assert template.agents_count == 3
        assert template.rounds == 2
        assert template.estimated_minutes == 5
        assert template.tags == []
        assert template.difficulty == "beginner"


class TestParseAnalytics:
    def test_parse_analytics_full(self, api: OnboardingAPI) -> None:
        analytics = api._parse_analytics(SAMPLE_ANALYTICS_RESPONSE)
        assert analytics.started == 100
        assert analytics.first_debate == 60
        assert analytics.completed == 40
        assert analytics.completion_rate == 0.4
        assert analytics.step_completion == {"welcome": 100, "use_case": 80, "template": 60}
        assert analytics.total_events == 500

    def test_parse_analytics_empty(self, api: OnboardingAPI) -> None:
        analytics = api._parse_analytics({})
        assert analytics.started == 0
        assert analytics.first_debate == 0
        assert analytics.completed == 0
        assert analytics.completion_rate == 0.0
        assert analytics.step_completion == {}
        assert analytics.total_events == 0


class TestDataclasses:
    def test_starter_template_defaults(self) -> None:
        template = StarterTemplate(
            id="t1",
            name="Test",
            description="desc",
            use_cases=["testing"],
            agents_count=3,
            rounds=2,
            estimated_minutes=5,
            example_prompt="example",
        )
        assert template.tags == []
        assert template.difficulty == "beginner"

    def test_starter_template_custom_difficulty(self) -> None:
        template = StarterTemplate(
            id="t2",
            name="Advanced",
            description="desc",
            use_cases=["research"],
            agents_count=5,
            rounds=4,
            estimated_minutes=15,
            example_prompt="example",
            difficulty="advanced",
        )
        assert template.difficulty == "advanced"

    def test_onboarding_flow_defaults(self) -> None:
        flow = OnboardingFlow(
            id="f1",
            current_step="welcome",
            completed_steps=[],
        )
        assert flow.use_case is None
        assert flow.selected_template_id is None
        assert flow.first_debate_id is None
        assert flow.quick_start_profile is None
        assert flow.team_invites_count == 0
        assert flow.progress_percentage == 0
        assert flow.started_at is None
        assert flow.updated_at is None
        assert flow.completed_at is None
        assert flow.skipped is False

    def test_quick_start_config(self) -> None:
        config = QuickStartConfig(
            profile="developer",
            default_template="tpl-dev",
            suggested_templates=["tpl-001", "tpl-002"],
            default_agents=["claude", "gpt"],
            default_rounds=3,
            focus_areas=["code_review", "architecture"],
        )
        assert config.profile == "developer"
        assert config.default_template == "tpl-dev"
        assert len(config.suggested_templates) == 2
        assert config.default_agents == ["claude", "gpt"]
        assert config.default_rounds == 3
        assert config.focus_areas == ["code_review", "architecture"]

    def test_onboarding_analytics_defaults(self) -> None:
        analytics = OnboardingAnalytics(
            started=10,
            first_debate=5,
            completed=3,
            completion_rate=0.3,
        )
        assert analytics.step_completion == {}
        assert analytics.total_events == 0
