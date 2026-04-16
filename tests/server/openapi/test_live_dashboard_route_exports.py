from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.agent_evolution_dashboard import AgentEvolutionDashboardHandler
from aragora.server.handlers.feedback_hub import FeedbackHubHandler
from aragora.server.handlers.outcome_dashboard import OutcomeDashboardHandler
from aragora.server.handlers.system_intelligence import SystemIntelligenceHandler
from aragora.server.openapi import generate_openapi_schema


def test_live_dashboard_handlers_participate_in_all_handlers() -> None:
    handlers = set(ALL_HANDLERS)

    assert SystemIntelligenceHandler in handlers
    assert OutcomeDashboardHandler in handlers
    assert AgentEvolutionDashboardHandler in handlers
    assert FeedbackHubHandler in handlers


def test_live_dashboard_routes_appear_in_generated_openapi() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    assert "/api/v1/system-intelligence/overview" in paths
    assert "/api/v1/outcome-dashboard" in paths
    assert "/api/v1/agent-evolution/timeline" in paths
    assert "/api/v1/feedback-hub/stats" in paths
    assert "/api/v1/feedback-hub/history" in paths
