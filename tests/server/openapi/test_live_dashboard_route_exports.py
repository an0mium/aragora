from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.agent_evolution_dashboard import AgentEvolutionDashboardHandler
from aragora.server.handlers.chat.router import ChatHandler
from aragora.server.handlers.feedback_hub import FeedbackHubHandler
from aragora.server.handlers.outcome_dashboard import OutcomeDashboardHandler
from aragora.server.handlers.system_intelligence import SystemIntelligenceHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


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


def test_chat_handler_participates_in_all_handlers() -> None:
    assert ChatHandler in set(ALL_HANDLERS)


def test_chat_routes_appear_in_generated_openapi_with_declared_methods() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    webhook_paths = [
        "/api/v1/chat/webhook",
        "/api/v1/chat/slack/webhook",
        "/api/v1/chat/teams/webhook",
        "/api/v1/chat/discord/webhook",
        "/api/v1/chat/google_chat/webhook",
        "/api/v1/chat/telegram/webhook",
        "/api/v1/chat/whatsapp/webhook",
    ]

    assert _operation_methods(paths["/api/v1/chat/status"]) == {"get"}
    for path in webhook_paths:
        assert _operation_methods(paths[path]) == {"post"}
