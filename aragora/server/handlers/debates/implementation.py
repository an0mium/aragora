"""Decision integrity operations for debates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission
from aragora.server.http_utils import run_async

from aragora.pipeline.decision_integrity import build_decision_integrity_package

from ..base import HandlerResult, error_response, handle_errors, json_response, require_storage
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    from aragora.billing.auth.context import UserAuthContext


logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    ctx: dict[str, Any]

    def get_storage(self) -> Any | None: ...

    def read_json_body(
        self, handler: Any, max_size: int | None = None
    ) -> dict[str, Any] | None: ...

    def get_current_user(self, handler: Any) -> "UserAuthContext | None": ...


class ImplementationOperationsMixin:
    """Mixin providing Decision Integrity endpoints for debates."""

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/decision-integrity",
        summary="Build decision integrity package",
        description="Generate a decision receipt and implementation plan from a debate.",
        tags=["Debates"],
        responses={
            "200": {"description": "Decision integrity package returned"},
            "400": {"description": "Invalid request"},
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:write")
    @require_storage
    @handle_errors("build decision integrity package")
    def _create_decision_integrity(
        self: _DebatesHandlerProtocol, handler: Any, debate_id: str
    ) -> HandlerResult:
        """Generate a decision receipt and implementation plan for a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id) if storage else None
        if not debate:
            return error_response("Debate not found", 404)

        payload = self.read_json_body(handler) or {}
        include_receipt = bool(payload.get("include_receipt", True))
        include_plan = bool(payload.get("include_plan", True))
        plan_strategy = str(payload.get("plan_strategy", "single_task"))

        repo_root = self.ctx.get("repo_root")
        repo_path = Path(repo_root) if repo_root else None

        package = run_async(
            build_decision_integrity_package(
                debate,
                include_receipt=include_receipt,
                include_plan=include_plan,
                plan_strategy=plan_strategy,
                repo_path=repo_path,
            )
        )

        return json_response(package.to_dict())
