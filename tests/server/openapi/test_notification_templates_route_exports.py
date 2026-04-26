from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.notifications.templates import NotificationTemplatesHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def test_notification_templates_handler_participates_in_all_handlers() -> None:
    assert NotificationTemplatesHandler in set(ALL_HANDLERS)


def test_notification_template_routes_appear_in_generated_openapi() -> None:
    paths = generate_openapi_schema()["paths"]

    assert _operation_methods(paths["/api/v1/notifications/templates"]) == {"get"}
    assert _operation_methods(paths["/api/v1/notifications/templates/{id}"]) == {"get", "put"}
    assert _operation_methods(paths["/api/v1/notifications/templates/{id}/reset"]) == {"post"}
    assert _operation_methods(paths["/api/v1/notifications/templates/{id}/preview"]) == {"post"}
