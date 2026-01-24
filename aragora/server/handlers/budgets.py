"""
Budget Management API Handler.

Endpoints:
- GET  /api/v1/budgets              - List budgets for org
- POST /api/v1/budgets              - Create a budget
- GET  /api/v1/budgets/:id          - Get budget details
- PATCH /api/v1/budgets/:id         - Update a budget
- DELETE /api/v1/budgets/:id        - Delete (close) a budget
- GET  /api/v1/budgets/:id/alerts   - Get alerts for a budget
- POST /api/v1/budgets/:id/alerts/:alert_id/acknowledge - Acknowledge alert
- POST /api/v1/budgets/:id/override - Add override for user
- DELETE /api/v1/budgets/:id/override/:user_id - Remove override
- POST /api/v1/budgets/:id/reset    - Reset budget period
- GET  /api/v1/budgets/summary      - Get org budget summary
- POST /api/v1/budgets/check        - Pre-flight cost check
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class BudgetHandler(BaseHandler):
    """Handler for budget management endpoints."""

    ROUTES = [
        "/api/v1/budgets",
        "/api/v1/budgets/summary",
        "/api/v1/budgets/check",
        "/api/v1/budgets/*",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        if path.startswith("/api/v1/budgets"):
            return True
        return False

    @rate_limit(rpm=60)
    async def handle(  # type: ignore[override]
        self,
        path: str,
        method: str,
        handler: Any = None,
        query_params: Optional[dict[str, Any]] = None,
    ) -> Optional[HandlerResult]:
        """Route budget requests to appropriate methods."""
        # Extract org_id from auth context
        org_id = self._get_org_id(handler)
        user_id = self._get_user_id(handler)

        # Parse path
        parts = path.rstrip("/").split("/")
        # /api/v1/budgets -> ["", "api", "v1", "budgets"]
        # /api/v1/budgets/budget-xxx -> ["", "api", "v1", "budgets", "budget-xxx"]

        # GET /api/v1/budgets/summary
        if path == "/api/v1/budgets/summary" and method == "GET":
            return self._get_summary(org_id)

        # POST /api/v1/budgets/check
        if path == "/api/v1/budgets/check" and method == "POST":
            return await self._check_budget(org_id, user_id, handler)

        # GET /api/v1/budgets - List budgets
        if path == "/api/v1/budgets" and method == "GET":
            return self._list_budgets(org_id, handler)

        # POST /api/v1/budgets - Create budget
        if path == "/api/v1/budgets" and method == "POST":
            return await self._create_budget(org_id, user_id, handler)

        # Routes with budget_id
        if len(parts) >= 5:
            budget_id = parts[4]

            # GET /api/v1/budgets/:id
            if len(parts) == 5 and method == "GET":
                return self._get_budget(budget_id, org_id)

            # PATCH /api/v1/budgets/:id
            if len(parts) == 5 and method == "PATCH":
                return await self._update_budget(budget_id, org_id, handler)

            # DELETE /api/v1/budgets/:id
            if len(parts) == 5 and method == "DELETE":
                return self._delete_budget(budget_id, org_id)

            # GET /api/v1/budgets/:id/alerts
            if len(parts) == 6 and parts[5] == "alerts" and method == "GET":
                return self._get_alerts(budget_id, org_id)

            # POST /api/v1/budgets/:id/alerts/:alert_id/acknowledge
            if (
                len(parts) == 8
                and parts[5] == "alerts"
                and parts[7] == "acknowledge"
                and method == "POST"
            ):
                alert_id = parts[6]
                return self._acknowledge_alert(alert_id, user_id)

            # POST /api/v1/budgets/:id/override
            if len(parts) == 6 and parts[5] == "override" and method == "POST":
                return await self._add_override(budget_id, org_id, handler)

            # DELETE /api/v1/budgets/:id/override/:user_id
            if len(parts) == 7 and parts[5] == "override" and method == "DELETE":
                target_user_id = parts[6]
                return self._remove_override(budget_id, org_id, target_user_id)

            # POST /api/v1/budgets/:id/reset
            if len(parts) == 6 and parts[5] == "reset" and method == "POST":
                return self._reset_budget(budget_id, org_id)

        return error_response("Not found", 404)

    def _get_org_id(self, handler: Any) -> str:
        """Extract org_id from request context."""
        if handler and hasattr(handler, "org_id"):
            return handler.org_id

        # Try to extract from auth header
        try:
            from aragora.billing.jwt_auth import extract_user_from_request

            user_ctx = extract_user_from_request(handler, None)
            if user_ctx and user_ctx.org_id:
                return user_ctx.org_id
        except (ImportError, AttributeError):
            pass

        return "default"

    def _get_user_id(self, handler: Any) -> Optional[str]:
        """Extract user_id from request context."""
        if handler and hasattr(handler, "user_id"):
            return handler.user_id

        try:
            from aragora.billing.jwt_auth import extract_user_from_request

            user_ctx = extract_user_from_request(handler, None)
            if user_ctx and user_ctx.user_id:
                return user_ctx.user_id
        except (ImportError, AttributeError):
            pass

        return None

    def _get_budget_manager(self):
        """Get budget manager instance."""
        from aragora.billing.budget_manager import get_budget_manager

        return get_budget_manager()

    # =========================================================================
    # Endpoint Implementations
    # =========================================================================

    def _list_budgets(self, org_id: str, handler: Any) -> HandlerResult:
        """List budgets for organization."""
        try:
            manager = self._get_budget_manager()

            # Parse query params
            active_only = True
            if handler:
                query_str = handler.path.split("?", 1)[1] if "?" in handler.path else ""
                from urllib.parse import parse_qs

                params = parse_qs(query_str)
                active_only = params.get("active_only", ["true"])[0].lower() == "true"

            budgets = manager.get_budgets_for_org(org_id, active_only=active_only)

            return json_response(
                {
                    "budgets": [b.to_dict() for b in budgets],
                    "count": len(budgets),
                    "org_id": org_id,
                }
            )

        except Exception as e:
            logger.error(f"Failed to list budgets: {e}")
            return error_response(f"Failed to list budgets: {str(e)[:100]}", 500)

    async def _create_budget(
        self, org_id: str, user_id: Optional[str], handler: Any
    ) -> HandlerResult:
        """Create a new budget."""
        try:
            from aragora.billing.budget_manager import BudgetPeriod

            body = self.read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            name = body.get("name")
            if not name:
                return error_response("Missing required field: name", 400)

            amount_usd = body.get("amount_usd")
            if amount_usd is None or amount_usd <= 0:
                return error_response("Invalid amount_usd: must be positive", 400)

            period_str = body.get("period", "monthly")
            try:
                period = BudgetPeriod(period_str)
            except ValueError:
                return error_response(f"Invalid period: {period_str}", 400)

            manager = self._get_budget_manager()
            budget = manager.create_budget(
                org_id=org_id,
                name=name,
                amount_usd=float(amount_usd),
                period=period,
                description=body.get("description", ""),
                auto_suspend=body.get("auto_suspend", True),
                created_by=user_id,
            )

            return json_response(budget.to_dict(), status=201)

        except Exception as e:
            logger.error(f"Failed to create budget: {e}")
            return error_response(f"Failed to create budget: {str(e)[:100]}", 500)

    def _get_budget(self, budget_id: str, org_id: str) -> HandlerResult:
        """Get budget details."""
        try:
            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            return json_response(budget.to_dict())

        except Exception as e:
            logger.error(f"Failed to get budget: {e}")
            return error_response(f"Failed to get budget: {str(e)[:100]}", 500)

    async def _update_budget(self, budget_id: str, org_id: str, handler: Any) -> HandlerResult:
        """Update a budget."""
        try:
            from aragora.billing.budget_manager import BudgetStatus

            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            body = self.read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            # Parse status if provided
            status = None
            if "status" in body:
                try:
                    status = BudgetStatus(body["status"])
                except ValueError:
                    return error_response(f"Invalid status: {body['status']}", 400)

            updated = manager.update_budget(
                budget_id=budget_id,
                name=body.get("name"),
                description=body.get("description"),
                amount_usd=body.get("amount_usd"),
                auto_suspend=body.get("auto_suspend"),
                status=status,
            )

            if not updated:
                return error_response("Failed to update budget", 500)

            return json_response(updated.to_dict())

        except Exception as e:
            logger.error(f"Failed to update budget: {e}")
            return error_response(f"Failed to update budget: {str(e)[:100]}", 500)

    def _delete_budget(self, budget_id: str, org_id: str) -> HandlerResult:
        """Delete (close) a budget."""
        try:
            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            manager.delete_budget(budget_id)

            return json_response({"deleted": True, "budget_id": budget_id})

        except Exception as e:
            logger.error(f"Failed to delete budget: {e}")
            return error_response(f"Failed to delete budget: {str(e)[:100]}", 500)

    def _get_summary(self, org_id: str) -> HandlerResult:
        """Get budget summary for organization."""
        try:
            manager = self._get_budget_manager()
            summary = manager.get_summary(org_id)
            return json_response(summary)

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return error_response(f"Failed to get summary: {str(e)[:100]}", 500)

    async def _check_budget(
        self, org_id: str, user_id: Optional[str], handler: Any
    ) -> HandlerResult:
        """Pre-flight cost check."""
        try:
            body = self.read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            estimated_cost = body.get("estimated_cost_usd", 0)
            if estimated_cost <= 0:
                return error_response("Invalid estimated_cost_usd: must be positive", 400)

            manager = self._get_budget_manager()
            allowed, reason, action = manager.check_budget(
                org_id=org_id,
                estimated_cost_usd=float(estimated_cost),
                user_id=user_id,
            )

            return json_response(
                {
                    "allowed": allowed,
                    "reason": reason,
                    "action": action.value if action else None,
                    "estimated_cost_usd": estimated_cost,
                }
            )

        except Exception as e:
            logger.error(f"Failed to check budget: {e}")
            return error_response(f"Failed to check budget: {str(e)[:100]}", 500)

    def _get_alerts(self, budget_id: str, org_id: str) -> HandlerResult:
        """Get alerts for a budget."""
        try:
            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            alerts = manager.get_alerts(budget_id=budget_id)

            return json_response(
                {
                    "alerts": [a.to_dict() for a in alerts],
                    "count": len(alerts),
                    "budget_id": budget_id,
                }
            )

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return error_response(f"Failed to get alerts: {str(e)[:100]}", 500)

    def _acknowledge_alert(self, alert_id: str, user_id: Optional[str]) -> HandlerResult:
        """Acknowledge a budget alert."""
        try:
            if not user_id:
                return error_response("User ID required", 400)

            manager = self._get_budget_manager()
            manager.acknowledge_alert(alert_id, user_id)

            return json_response({"acknowledged": True, "alert_id": alert_id})

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return error_response(f"Failed to acknowledge alert: {str(e)[:100]}", 500)

    async def _add_override(self, budget_id: str, org_id: str, handler: Any) -> HandlerResult:
        """Add budget override for a user."""
        try:
            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            body = self.read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            target_user_id = body.get("user_id")
            if not target_user_id:
                return error_response("Missing required field: user_id", 400)

            duration_hours = body.get("duration_hours")

            manager.add_override(
                budget_id=budget_id,
                user_id=target_user_id,
                duration_hours=float(duration_hours) if duration_hours else None,
            )

            return json_response(
                {
                    "override_added": True,
                    "budget_id": budget_id,
                    "user_id": target_user_id,
                    "duration_hours": duration_hours,
                }
            )

        except Exception as e:
            logger.error(f"Failed to add override: {e}")
            return error_response(f"Failed to add override: {str(e)[:100]}", 500)

    def _remove_override(self, budget_id: str, org_id: str, target_user_id: str) -> HandlerResult:
        """Remove budget override for a user."""
        try:
            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            manager.remove_override(budget_id, target_user_id)

            return json_response(
                {
                    "override_removed": True,
                    "budget_id": budget_id,
                    "user_id": target_user_id,
                }
            )

        except Exception as e:
            logger.error(f"Failed to remove override: {e}")
            return error_response(f"Failed to remove override: {str(e)[:100]}", 500)

    def _reset_budget(self, budget_id: str, org_id: str) -> HandlerResult:
        """Reset budget period."""
        try:
            manager = self._get_budget_manager()
            budget = manager.get_budget(budget_id)

            if not budget:
                return error_response("Budget not found", 404)

            if budget.org_id != org_id:
                return error_response("Access denied", 403)

            updated = manager.reset_period(budget_id)

            if not updated:
                return error_response("Failed to reset budget", 500)

            return json_response(updated.to_dict())

        except Exception as e:
            logger.error(f"Failed to reset budget: {e}")
            return error_response(f"Failed to reset budget: {str(e)[:100]}", 500)


# Handler factory function
def create_budget_handler(server_context: Any) -> BudgetHandler:
    """Factory function for handler registration."""
    return BudgetHandler(server_context)
