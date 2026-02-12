"""Route chat interactions to approval workflows."""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.approvals.tokens import decode_approval_action

logger = logging.getLogger(__name__)


def _extract_token(interaction: Any) -> str | None:
    """Extract approval token from an interaction."""
    if not interaction:
        return None

    # Prefer explicit value payload
    value = getattr(interaction, "value", None)
    if value:
        return value

    action_id = getattr(interaction, "action_id", "") or ""
    if action_id.startswith("approval:") and len(action_id.split(":", 2)) >= 3:
        # approval:{action}:{token}
        return action_id.split(":", 2)[2] or None

    return None


def _require_identity_mapping() -> bool:
    value = os.environ.get("ARAGORA_CHAT_APPROVAL_REQUIRE_IDENTITY", "1").lower()
    return value not in ("0", "false", "no", "off")


def _resolve_actor(interaction: Any) -> tuple[str, str, bool]:
    """Resolve approver identity (best-effort mapping)."""
    user = getattr(interaction, "user", None)
    user_id = getattr(user, "id", None) or "unknown"
    display = (
        getattr(user, "display_name", None)
        or getattr(user, "username", None)
        or getattr(user, "email", None)
        or user_id
    )

    mapped = False
    # Attempt external identity mapping to internal user ID
    try:
        from aragora.storage.repositories.external_identity import (
            get_external_identity_repository,
        )

        provider = getattr(interaction, "platform", None) or ""
        channel = getattr(interaction, "channel", None)
        tenant_id = getattr(channel, "team_id", None) if channel else None
        repo = get_external_identity_repository()
        identity = repo.get_by_external_id(provider, user_id, tenant_id=tenant_id)
        if identity:
            user_id = identity.user_id
            display = identity.display_name or identity.email or display
            mapped = True
    except Exception:
        logger.debug("External identity lookup failed for user %s", user_id, exc_info=True)

    return user_id, display, mapped


class ApprovalInteractionRouter:
    """Handle chat approval interactions."""

    async def handle_interaction(self, event: Any, connector: Any) -> bool:
        interaction = getattr(event, "interaction", None)
        if interaction is None:
            return False

        action_id = getattr(interaction, "action_id", "") or ""
        if not action_id.startswith("approval"):
            return False

        token_value = _extract_token(interaction)
        if not token_value:
            await self._respond(
                connector,
                interaction,
                "Approval token missing or invalid.",
                replace_original=False,
            )
            return True

        token = decode_approval_action(token_value)
        if token is None:
            await self._respond(
                connector,
                interaction,
                "Approval token invalid or expired.",
                replace_original=False,
            )
            return True

        actor_id, actor_display, mapped = _resolve_actor(interaction)
        if _require_identity_mapping() and not mapped:
            await self._respond(
                connector,
                interaction,
                "Approval requires a linked Aragora identity. Please link your account.",
                replace_original=False,
            )
            return True

        ok, message = await self._route_action(token, actor_id, actor_display)
        await self._respond(
            connector,
            interaction,
            message,
            replace_original=True,
        )
        return ok

    async def _route_action(
        self, token: Any, actor_id: str, actor_display: str
    ) -> tuple[bool, str]:
        kind = token.kind
        action = token.action.lower().strip()
        if action not in ("approve", "reject"):
            return False, f"Unsupported approval action: {action}"
        approved = action == "approve"

        if kind == "workflow":
            return self._handle_workflow(token.target_id, approved, actor_id, actor_display)

        if kind == "decision_plan":
            return self._handle_plan(token.target_id, approved, actor_id, actor_display)

        if kind == "computer_use":
            return await self._handle_computer_use(
                token.target_id,
                approved,
                actor_id,
                actor_display,
            )

        return False, f"Unsupported approval type: {kind}"

    def _handle_workflow(
        self,
        request_id: str,
        approved: bool,
        actor_id: str,
        actor_display: str,
    ) -> tuple[bool, str]:
        try:
            from aragora.workflow.nodes.human_checkpoint import (
                ApprovalStatus,
                get_approval_request,
                resolve_approval,
            )
        except ImportError:
            return False, "Workflow approvals unavailable"

        request = get_approval_request(request_id)
        if request and request.status != ApprovalStatus.PENDING:
            return True, f"Approval already {request.status.value}."

        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        ok = resolve_approval(
            request_id=request_id,
            status=status,
            responder_id=actor_id,
            notes=f"Resolved via chat by {actor_display}",
        )
        if not ok:
            return False, "Approval request not found."

        outcome = "approved" if approved else "rejected"
        return True, f"✅ Workflow approval {outcome} by {actor_display}"

    def _handle_plan(
        self,
        plan_id: str,
        approved: bool,
        actor_id: str,
        actor_display: str,
    ) -> tuple[bool, str]:
        try:
            from aragora.pipeline.executor import get_plan, store_plan
            from aragora.pipeline.decision_plan import PlanStatus
        except ImportError:
            return False, "Decision plans unavailable"

        plan = get_plan(plan_id)
        if not plan:
            return False, "Decision plan not found."

        if plan.status not in (PlanStatus.CREATED, PlanStatus.AWAITING_APPROVAL):
            return True, f"Plan already {plan.status.value}."

        if approved:
            plan.approve(approver_id=actor_id, reason=f"Approved via chat by {actor_display}")
        else:
            plan.reject(approver_id=actor_id, reason=f"Rejected via chat by {actor_display}")

        store_plan(plan)

        outcome = "approved" if approved else "rejected"
        return True, f"✅ Plan {plan.id} {outcome} by {actor_display}"

    async def _handle_computer_use(
        self,
        request_id: str,
        approved: bool,
        actor_id: str,
        actor_display: str,
    ) -> tuple[bool, str]:
        try:
            from aragora.server.extensions import get_extension_state
        except ImportError:
            return False, "Computer-use approvals unavailable"

        state = get_extension_state()
        workflow = getattr(state, "computer_approval_workflow", None) if state else None
        if workflow is None:
            return False, "Computer-use approval workflow not configured."

        if approved:
            ok = await workflow.approve(
                request_id=request_id,
                approver_id=actor_id,
                reason=f"Approved via chat by {actor_display}",
            )
        else:
            ok = await workflow.deny(
                request_id=request_id,
                denier_id=actor_id,
                reason=f"Rejected via chat by {actor_display}",
            )

        if not ok:
            return False, "Computer-use approval request not found or already resolved."

        outcome = "approved" if approved else "rejected"
        return True, f"✅ Computer-use action {outcome} by {actor_display}"

    async def _respond(
        self,
        connector: Any,
        interaction: Any,
        message: str,
        replace_original: bool = False,
    ) -> None:
        if connector is None:
            return
        try:
            await connector.respond_to_interaction(
                interaction,
                message,
                replace_original=replace_original,
            )
        except Exception:
            logger.debug("Failed to respond to interaction", exc_info=True)


__all__ = ["ApprovalInteractionRouter"]
