"""
Budget Controls API Handlers.

Provides management APIs for organization budget controls:
- GET /api/v1/sme/budgets - List org budgets
- POST /api/v1/sme/budgets - Create budget
- GET /api/v1/sme/budgets/:id - Get budget details
- PATCH /api/v1/sme/budgets/:id - Update budget
- DELETE /api/v1/sme/budgets/:id - Delete budget
- GET /api/v1/sme/budgets/:id/alerts - List alerts
- POST /api/v1/sme/budgets/:id/alerts/ack - Acknowledge alert
- GET /api/v1/sme/budgets/:id/transactions - Transaction history
- POST /api/v1/sme/budgets/check - Pre-check operation cost
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..base import (
    error_response,
    get_string_param,
    handle_errors,
    json_response,
)
from ..utils.responses import HandlerResult
from ..secure import SecureHandler
from aragora.rbac.decorators import require_permission
from aragora.server.validation.query_params import safe_query_int
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for budget APIs (60 requests per minute)
_budget_limiter = RateLimiter(requests_per_minute=60)


class BudgetControlsHandler(SecureHandler):
    """Handler for budget control endpoints.

    Provides APIs for managing organization budgets,
    viewing alerts, and checking spending limits.
    """

    RESOURCE_TYPE = "budget"

    ROUTES = [
        "/api/v1/sme/budgets",
        "/api/v1/sme/budgets/check",
    ]

    # Regex patterns for parameterized routes
    ROUTE_PATTERNS = [
        (re.compile(r"^/api/v1/sme/budgets/([^/]+)/alerts/ack$"), "alert_ack"),
        (re.compile(r"^/api/v1/sme/budgets/([^/]+)/alerts$"), "alerts"),
        (re.compile(r"^/api/v1/sme/budgets/([^/]+)/transactions$"), "transactions"),
        (re.compile(r"^/api/v1/sme/budgets/([^/]+)$"), "budget_detail"),
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        for pattern, _ in self.ROUTE_PATTERNS:
            if pattern.match(path):
                return True
        return False

    def _match_route(self, path: str) -> tuple[str | None, str | None]:
        """Match a path against parameterized routes.

        Returns:
            Tuple of (route_name, extracted_id) or (None, None) if no match.
        """
        for pattern, route_name in self.ROUTE_PATTERNS:
            match = pattern.match(path)
            if match:
                return route_name, match.group(1)
        return None, None

    def handle(
        self,
        path: str,
        query_params: dict,
        handler,
        method: str = "GET",
    ) -> HandlerResult | None:
        """Route budget control requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _budget_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for budget controls: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # Handle static routes
        if path == "/api/v1/sme/budgets":
            if method == "GET":
                return self._list_budgets(handler, query_params)
            elif method == "POST":
                return self._create_budget(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/budgets/check":
            if method == "POST":
                return self._check_spend(handler, query_params)
            return error_response("Method not allowed", 405)

        # Handle parameterized routes
        route_name, param_id = self._match_route(path)
        if route_name:
            if route_name == "budget_detail":
                if method == "GET":
                    return self._get_budget(handler, query_params, param_id)
                elif method == "PATCH":
                    return self._update_budget(handler, query_params, param_id)
                elif method == "DELETE":
                    return self._delete_budget(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "alerts":
                if method == "GET":
                    return self._list_alerts(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "alert_ack":
                if method == "POST":
                    return self._acknowledge_alert(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "transactions":
                if method == "GET":
                    return self._list_transactions(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    def _get_budget_manager(self):
        """Get budget manager instance."""
        from aragora.billing.budget_manager import get_budget_manager

        return get_budget_manager()

    def _get_user_and_org(self, handler, user):
        """Get user and organization from context."""
        user_store = self.ctx.get("user_store")
        if not user_store:
            return None, None, error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return None, None, error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return None, None, error_response("No organization found", 404)

        return db_user, org, None

    @handle_errors("list budgets")
    @require_permission("sme:budgets:read")
    def _list_budgets(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        List budgets for the organization.

        Query Parameters:
            active_only: Only show active budgets (default: true)

        Returns:
            JSON response with budget list:
            {
                "budgets": [...],
                "total": 3
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        active_only = get_string_param(handler, "active_only", "true") == "true"

        manager = self._get_budget_manager()
        budgets = manager.get_budgets_for_org(org.id, active_only=active_only)

        return json_response(
            {
                "budgets": [b.to_dict() for b in budgets],
                "total": len(budgets),
            }
        )

    @handle_errors("get budget")
    @require_permission("sme:budgets:read")
    def _get_budget(
        self,
        handler,
        query_params: dict,
        budget_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Get details for a specific budget.

        Path Parameters:
            budget_id: Budget ID

        Returns:
            JSON response with budget details
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        manager = self._get_budget_manager()
        budget = manager.get_budget(budget_id)

        if not budget:
            return error_response("Budget not found", 404)

        # Verify budget belongs to this org
        if budget.org_id != org.id:
            return error_response("Budget not found", 404)

        return json_response({"budget": budget.to_dict()})

    @handle_errors("create budget")
    @require_permission("sme:budgets:write")
    def _create_budget(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Create a new budget for the organization.

        Request Body:
            {
                "name": "Monthly Operations",
                "description": "General operations budget",
                "amount_usd": 500.00,
                "period": "monthly",
                "auto_suspend": true,
                "allow_overage": false
            }

        Returns:
            JSON response with created budget
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        # Validate required fields
        name = data.get("name")
        amount_usd = data.get("amount_usd")

        if not name:
            return error_response("name is required", 400)
        if amount_usd is None or amount_usd <= 0:
            return error_response("amount_usd must be positive", 400)

        # Parse period
        from aragora.billing.budget_manager import BudgetPeriod

        period_str = data.get("period", "monthly")
        try:
            period = BudgetPeriod(period_str)
        except ValueError:
            valid_periods = [p.value for p in BudgetPeriod]
            return error_response(f"Invalid period. Valid values: {valid_periods}", 400)

        manager = self._get_budget_manager()

        try:
            try:
                amount_usd_float = float(amount_usd)
            except (ValueError, TypeError):
                return error_response("Invalid amount_usd value", 400)
            budget = manager.create_budget(
                org_id=org.id,
                name=name,
                amount_usd=amount_usd_float,
                period=period,
                description=data.get("description", ""),
                auto_suspend=data.get("auto_suspend", True),
                allow_overage=data.get("allow_overage", False),
                overage_rate_multiplier=data.get("overage_rate_multiplier", 1.5),
                max_overage_usd=data.get("max_overage_usd"),
                created_by=db_user.id,
            )
            logger.info(f"Created budget {budget.budget_id} for org {org.id}")
            return json_response({"budget": budget.to_dict()}, status=201)
        except Exception as e:
            logger.error(f"Failed to create budget: {e}")
            return error_response(f"Failed to create budget: {e}", 500)

    @handle_errors("update budget")
    @require_permission("sme:budgets:write")
    def _update_budget(
        self,
        handler,
        query_params: dict,
        budget_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Update an existing budget.

        Path Parameters:
            budget_id: Budget ID

        Request Body:
            {
                "name": "Updated name",
                "amount_usd": 600.00,
                "auto_suspend": false
            }

        Returns:
            JSON response with updated budget
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        manager = self._get_budget_manager()
        budget = manager.get_budget(budget_id)

        if not budget:
            return error_response("Budget not found", 404)

        if budget.org_id != org.id:
            return error_response("Budget not found", 404)

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        # Build update kwargs
        update_kwargs: dict[str, Any] = {}

        if "name" in data:
            update_kwargs["name"] = data["name"]
        if "description" in data:
            update_kwargs["description"] = data["description"]
        if "amount_usd" in data:
            try:
                amount_usd_val = float(data["amount_usd"])
            except (ValueError, TypeError):
                return error_response("Invalid amount_usd value", 400)
            if amount_usd_val <= 0:
                return error_response("amount_usd must be positive", 400)
            update_kwargs["amount_usd"] = amount_usd_val
        if "auto_suspend" in data:
            update_kwargs["auto_suspend"] = bool(data["auto_suspend"])
        if "allow_overage" in data:
            update_kwargs["allow_overage"] = bool(data["allow_overage"])
        if "overage_rate_multiplier" in data:
            try:
                update_kwargs["overage_rate_multiplier"] = float(data["overage_rate_multiplier"])
            except (ValueError, TypeError):
                return error_response("Invalid overage_rate_multiplier value", 400)
        if "max_overage_usd" in data:
            if data["max_overage_usd"] is not None:
                try:
                    update_kwargs["max_overage_usd"] = float(data["max_overage_usd"])
                except (ValueError, TypeError):
                    return error_response("Invalid max_overage_usd value", 400)
            else:
                update_kwargs["max_overage_usd"] = None

        if not update_kwargs:
            return error_response("No update fields provided", 400)

        try:
            updated = manager.update_budget(budget_id, **update_kwargs)
            if not updated:
                return error_response("Failed to update budget", 500)
            logger.info(f"Updated budget {budget_id}")
            return json_response({"budget": updated.to_dict()})
        except Exception as e:
            logger.error(f"Failed to update budget: {e}")
            return error_response(f"Failed to update budget: {e}", 500)

    @handle_errors("delete budget")
    @require_permission("sme:budgets:write")
    def _delete_budget(
        self,
        handler,
        query_params: dict,
        budget_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Delete a budget.

        Path Parameters:
            budget_id: Budget ID

        Returns:
            JSON response confirming deletion
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        manager = self._get_budget_manager()
        budget = manager.get_budget(budget_id)

        if not budget:
            return error_response("Budget not found", 404)

        if budget.org_id != org.id:
            return error_response("Budget not found", 404)

        success = manager.delete_budget(budget_id)

        if not success:
            return error_response("Failed to delete budget", 500)

        logger.info(f"Deleted budget {budget_id} for org {org.id}")

        return json_response(
            {
                "deleted": True,
                "budget_id": budget_id,
            }
        )

    @handle_errors("list budget alerts")
    @require_permission("sme:budgets:read")
    def _list_alerts(
        self,
        handler,
        query_params: dict,
        budget_id: str,
        user=None,
    ) -> HandlerResult:
        """
        List alerts for a budget.

        Path Parameters:
            budget_id: Budget ID

        Query Parameters:
            unacknowledged_only: Only show unacknowledged alerts (default: false)
            limit: Maximum results (default: 50)

        Returns:
            JSON response with alert list
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        manager = self._get_budget_manager()
        budget = manager.get_budget(budget_id)

        if not budget:
            return error_response("Budget not found", 404)

        if budget.org_id != org.id:
            return error_response("Budget not found", 404)

        unack_only = get_string_param(handler, "unacknowledged_only", "false") == "true"
        limit_str = get_string_param(handler, "limit", "50")
        limit = safe_query_int({"limit": [limit_str]}, "limit", default=50, min_val=1, max_val=1000)

        alerts = manager.get_alerts(budget_id, unacknowledged_only=unack_only, limit=limit)

        return json_response(
            {
                "alerts": [a.to_dict() for a in alerts],
                "total": len(alerts),
                "budget_id": budget_id,
            }
        )

    @handle_errors("acknowledge alert")
    @require_permission("sme:budgets:write")
    def _acknowledge_alert(
        self,
        handler,
        query_params: dict,
        budget_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Acknowledge a budget alert.

        Path Parameters:
            budget_id: Budget ID

        Request Body:
            {
                "alert_id": "alert-123"
            }

        Returns:
            JSON response confirming acknowledgment
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        manager = self._get_budget_manager()
        budget = manager.get_budget(budget_id)

        if not budget:
            return error_response("Budget not found", 404)

        if budget.org_id != org.id:
            return error_response("Budget not found", 404)

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        alert_id = data.get("alert_id")
        if not alert_id:
            return error_response("alert_id is required", 400)

        success = manager.acknowledge_alert(alert_id, db_user.id)

        if not success:
            return error_response("Failed to acknowledge alert", 500)

        logger.info(f"Acknowledged alert {alert_id} by user {db_user.id}")

        return json_response(
            {
                "acknowledged": True,
                "alert_id": alert_id,
                "acknowledged_by": db_user.id,
                "acknowledged_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @handle_errors("list transactions")
    @require_permission("sme:budgets:read")
    def _list_transactions(
        self,
        handler,
        query_params: dict,
        budget_id: str,
        user=None,
    ) -> HandlerResult:
        """
        List transactions for a budget.

        Path Parameters:
            budget_id: Budget ID

        Query Parameters:
            limit: Maximum results (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            JSON response with transaction list
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        manager = self._get_budget_manager()
        budget = manager.get_budget(budget_id)

        if not budget:
            return error_response("Budget not found", 404)

        if budget.org_id != org.id:
            return error_response("Budget not found", 404)

        limit_str = get_string_param(handler, "limit", "50")
        offset_str = get_string_param(handler, "offset", "0")
        limit = safe_query_int({"limit": [limit_str]}, "limit", default=50, min_val=1, max_val=1000)
        offset = safe_query_int(
            {"offset": [offset_str]}, "offset", default=0, min_val=0, max_val=10000
        )

        # Get transactions
        transactions = manager.get_transactions(budget_id, limit=limit, offset=offset)

        return json_response(
            {
                "transactions": [t.to_dict() for t in transactions],
                "total": len(transactions),
                "budget_id": budget_id,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("check spend")
    @require_permission("sme:budgets:read")
    def _check_spend(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Pre-check if a spend amount is allowed.

        Request Body:
            {
                "budget_id": "budget-123",  # Optional - uses default if not provided
                "amount_usd": 10.50
            }

        Returns:
            JSON response with spend check result:
            {
                "allowed": true,
                "message": "OK",
                "is_overage": false,
                "budget": {...}
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        amount_usd = data.get("amount_usd")
        if amount_usd is None or amount_usd < 0:
            return error_response("amount_usd is required and must be non-negative", 400)

        budget_id = data.get("budget_id")

        manager = self._get_budget_manager()

        # Get budget - either specified or find default for org
        if budget_id:
            budget = manager.get_budget(budget_id)
            if not budget or budget.org_id != org.id:
                return error_response("Budget not found", 404)
        else:
            # Get first active budget for org
            budgets = manager.get_budgets_for_org(org.id, active_only=True)
            if not budgets:
                # No budget configured - spending is unlimited
                return json_response(
                    {
                        "allowed": True,
                        "message": "No budget configured - spending unlimited",
                        "is_overage": False,
                        "budget": None,
                    }
                )
            budget = budgets[0]

        # Check if spend is allowed
        try:
            amount_usd_float = float(amount_usd)
        except (ValueError, TypeError):
            return error_response("Invalid amount_usd value", 400)
        result = budget.can_spend_extended(amount_usd_float, user_id=db_user.id)

        return json_response(
            {
                "allowed": result.allowed,
                "message": result.message,
                "is_overage": result.is_overage,
                "overage_amount_usd": result.overage_amount_usd,
                "overage_rate_multiplier": result.overage_rate_multiplier,
                "budget": budget.to_dict(),
            }
        )


__all__ = ["BudgetControlsHandler"]
