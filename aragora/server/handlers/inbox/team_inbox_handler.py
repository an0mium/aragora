"""Route wrapper for team inbox collaboration endpoints."""

from __future__ import annotations

import re
from typing import Any

from aragora.server.handlers.base import BaseHandler, HandlerResult, error_response
from aragora.server.handlers.inbox.team_inbox import (
    handle_acknowledge_mention,
    handle_add_note,
    handle_add_team_member,
    handle_get_activity_feed,
    handle_get_mentions,
    handle_get_notes,
    handle_get_team_members,
    handle_remove_team_member,
    handle_start_typing,
    handle_start_viewing,
    handle_stop_typing,
    handle_stop_viewing,
)
from aragora.server.handlers.shared_inbox.handler import SharedInboxHandler

_TEAM_INBOX_PATTERNS = (
    re.compile(r"^/api/v1/inbox/shared/[^/]+/team$"),
    re.compile(r"^/api/v1/inbox/shared/[^/]+/team/[^/]+$"),
    re.compile(r"^/api/v1/inbox/shared/[^/]+/messages/[^/]+/(viewing|typing|notes)$"),
    re.compile(r"^/api/v1/inbox/shared/[^/]+/activity$"),
    re.compile(r"^/api/v1/inbox/mentions$"),
    re.compile(r"^/api/v1/inbox/mentions/[^/]+/acknowledge$"),
)


class TeamInboxHandler(BaseHandler):
    """Dispatch team inbox collaboration routes to the function handlers."""

    ROUTES = [
        "/api/v1/inbox/shared/{inbox_id}/team",
        "/api/v1/inbox/shared/{inbox_id}/team/{member_user_id}",
        "/api/v1/inbox/shared/{inbox_id}/messages/{message_id}/viewing",
        "/api/v1/inbox/shared/{inbox_id}/messages/{message_id}/typing",
        "/api/v1/inbox/shared/{inbox_id}/messages/{message_id}/notes",
        "/api/v1/inbox/shared/{inbox_id}/activity",
        "/api/v1/inbox/mentions",
        "/api/v1/inbox/mentions/{mention_id}/acknowledge",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/inbox/shared/",
        "/api/v1/inbox/mentions/",
    ]

    def can_handle(self, path: str) -> bool:
        return any(pattern.fullmatch(path) for pattern in _TEAM_INBOX_PATTERNS)

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        self._bind_auth_context(handler)

        if path == "/api/v1/inbox/mentions":
            return self._run_async(handle_get_mentions(query_params, user_id=self._get_user_id()))

        shared_parts = self._route_parts(path, "/api/v1/inbox/shared/")
        if shared_parts is None:
            return None
        if len(shared_parts) == 2 and shared_parts[1] == "team":
            return self._run_async(
                handle_get_team_members(
                    query_params, inbox_id=shared_parts[0], user_id=self._get_user_id()
                )
            )
        if len(shared_parts) == 4 and shared_parts[1] == "messages" and shared_parts[3] == "notes":
            return self._run_async(
                handle_get_notes(
                    query_params,
                    inbox_id=shared_parts[0],
                    message_id=shared_parts[2],
                    user_id=self._get_user_id(),
                )
            )
        if len(shared_parts) == 2 and shared_parts[1] == "activity":
            return self._run_async(
                handle_get_activity_feed(
                    query_params,
                    inbox_id=shared_parts[0],
                    user_id=self._get_user_id(),
                )
            )
        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        del query_params
        self._bind_auth_context(handler)
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        mention_parts = self._route_parts(path, "/api/v1/inbox/mentions/")
        if mention_parts is not None:
            if len(mention_parts) == 2 and mention_parts[1] == "acknowledge":
                return self._run_async(
                    handle_acknowledge_mention(
                        body,
                        mention_id=mention_parts[0],
                        user_id=self._get_user_id(),
                    )
                )
            return error_response("Not found", 404)

        shared_parts = self._route_parts(path, "/api/v1/inbox/shared/")
        if shared_parts is None:
            return None
        if len(shared_parts) == 2 and shared_parts[1] == "team":
            return self._run_async(
                handle_add_team_member(body, inbox_id=shared_parts[0], user_id=self._get_user_id())
            )
        if len(shared_parts) == 4 and shared_parts[1] == "messages":
            inbox_id, _, message_id, action = shared_parts
            if action == "viewing":
                return self._run_async(
                    handle_start_viewing(
                        body, inbox_id=inbox_id, message_id=message_id, user_id=self._get_user_id()
                    )
                )
            if action == "typing":
                return self._run_async(
                    handle_start_typing(
                        body, inbox_id=inbox_id, message_id=message_id, user_id=self._get_user_id()
                    )
                )
            if action == "notes":
                return self._run_async(
                    handle_add_note(
                        body, inbox_id=inbox_id, message_id=message_id, user_id=self._get_user_id()
                    )
                )
        return error_response("Not found", 404)

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        del query_params
        self._bind_auth_context(handler)

        shared_parts = self._route_parts(path, "/api/v1/inbox/shared/")
        if shared_parts is None:
            return None
        if len(shared_parts) == 3 and shared_parts[1] == "team":
            return self._run_async(
                handle_remove_team_member(
                    {},
                    inbox_id=shared_parts[0],
                    member_user_id=shared_parts[2],
                    user_id=self._get_user_id(),
                )
            )
        if len(shared_parts) == 4 and shared_parts[1] == "messages":
            inbox_id, _, message_id, action = shared_parts
            if action == "viewing":
                return self._run_async(
                    handle_stop_viewing(
                        {},
                        inbox_id=inbox_id,
                        message_id=message_id,
                        user_id=self._get_user_id(),
                    )
                )
            if action == "typing":
                return self._run_async(
                    handle_stop_typing(
                        {},
                        inbox_id=inbox_id,
                        message_id=message_id,
                        user_id=self._get_user_id(),
                    )
                )
        return None

    def _get_user_id(self) -> str:
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx is not None and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"

    def _bind_auth_context(self, handler: Any) -> None:
        if handler is None:
            return
        auth_ctx = getattr(handler, "_auth_context", None)
        if auth_ctx is not None:
            self.ctx["auth_context"] = auth_ctx

    @staticmethod
    def _route_parts(path: str, prefix: str) -> list[str] | None:
        return SharedInboxHandler._route_parts(path, prefix)

    @staticmethod
    def _run_async(coro: Any) -> HandlerResult:
        return SharedInboxHandler._run_async(coro)


__all__ = ["TeamInboxHandler"]
