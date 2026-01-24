"""
Credits Admin API Handler.

Endpoints:
- POST /api/v1/admin/credits/{org_id}/issue - Issue credits to organization
- GET /api/v1/admin/credits/{org_id} - Get credit account details
- GET /api/v1/admin/credits/{org_id}/transactions - List credit transactions
- POST /api/v1/admin/credits/{org_id}/adjust - Make balance adjustment
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from aiohttp import web

from aragora.billing.credits import (
    CreditTransactionType,
    get_credit_manager,
)
from aragora.server.handlers.base import (
    error_response,
    json_response,
)
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.server.handlers.secure import SecureHandler
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)


class CreditsAdminHandler(SecureHandler):
    """Handler for credit administration endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.
    """

    RESOURCE_TYPE = "credits"

    @require_permission("admin.credits.issue")
    async def issue_credit(
        self,
        org_id: str,
        data: Dict[str, Any],
        user_id: str,
    ) -> HandlerResult:
        """Issue credits to an organization.

        Args:
            org_id: Target organization ID
            data: Request data with amount_cents, type, description, expires_days
            user_id: Admin user issuing the credit

        Returns:
            Created transaction details
        """
        # Validate required fields
        amount_cents = data.get("amount_cents")
        if not amount_cents or not isinstance(amount_cents, int) or amount_cents <= 0:
            return error_response("amount_cents must be a positive integer", status=400)

        credit_type_str = data.get("type", "promotional")
        try:
            credit_type = CreditTransactionType(credit_type_str.lower())
        except ValueError:
            valid_types = [
                t.value for t in CreditTransactionType if t != CreditTransactionType.USAGE
            ]
            return error_response(
                f"Invalid credit type. Must be one of: {', '.join(valid_types)}",
                status=400,
            )

        description = data.get("description", "")
        if not description:
            return error_response("description is required", status=400)

        # Calculate expiration if specified
        expires_at = None
        expires_days = data.get("expires_days")
        if expires_days and isinstance(expires_days, int) and expires_days > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

        # Issue the credit
        manager = get_credit_manager()
        transaction = await manager.issue_credit(
            org_id=org_id,
            amount_cents=amount_cents,
            credit_type=credit_type,
            description=description,
            expires_at=expires_at,
            created_by=user_id,
            reference_id=data.get("reference_id"),
        )

        logger.info(f"Admin {user_id} issued {amount_cents} cents to org {org_id}: {description}")

        return json_response(
            {"transaction": transaction.to_dict()},
            status=201,
        )

    @require_permission("admin.credits.view")
    async def get_credit_account(self, org_id: str) -> HandlerResult:
        """Get credit account details for an organization.

        Args:
            org_id: Organization ID

        Returns:
            Credit account summary with balance
        """
        manager = get_credit_manager()
        account = await manager.get_account(org_id)

        return json_response({"account": account.to_dict()})

    @require_permission("admin.credits.view")
    async def list_transactions(
        self,
        org_id: str,
        limit: int = 100,
        offset: int = 0,
        transaction_type: Optional[str] = None,
    ) -> HandlerResult:
        """List credit transactions for an organization.

        Args:
            org_id: Organization ID
            limit: Maximum transactions to return
            offset: Number to skip
            transaction_type: Optional filter by type

        Returns:
            List of transactions
        """
        manager = get_credit_manager()

        # Parse transaction type filter
        type_filter = None
        if transaction_type:
            try:
                type_filter = CreditTransactionType(transaction_type.lower())
            except ValueError:
                return error_response(f"Invalid transaction type: {transaction_type}", status=400)

        transactions = await manager.get_transactions(
            org_id=org_id,
            limit=min(limit, 500),
            offset=offset,
            transaction_type=type_filter,
        )

        return json_response(
            {
                "transactions": [tx.to_dict() for tx in transactions],
                "count": len(transactions),
                "offset": offset,
                "limit": limit,
            }
        )

    @require_permission("admin.credits.adjust")
    async def adjust_balance(
        self,
        org_id: str,
        data: Dict[str, Any],
        user_id: str,
    ) -> HandlerResult:
        """Make a manual balance adjustment.

        Args:
            org_id: Organization ID
            data: Request data with amount_cents (can be negative), description
            user_id: Admin user making the adjustment

        Returns:
            Adjustment transaction details
        """
        amount_cents = data.get("amount_cents")
        if amount_cents is None or not isinstance(amount_cents, int):
            return error_response("amount_cents must be an integer", status=400)

        if amount_cents == 0:
            return error_response("amount_cents cannot be zero", status=400)

        description = data.get("description", "")
        if not description:
            return error_response("description is required for adjustments", status=400)

        manager = get_credit_manager()
        transaction = await manager.adjust_balance(
            org_id=org_id,
            amount_cents=amount_cents,
            description=description,
            created_by=user_id,
        )

        action = "increased" if amount_cents > 0 else "decreased"
        logger.info(
            f"Admin {user_id} {action} org {org_id} balance by {abs(amount_cents)} cents: {description}"
        )

        return json_response({"transaction": transaction.to_dict()})

    @require_permission("admin.credits.view")
    async def get_expiring_credits(
        self,
        org_id: str,
        within_days: int = 30,
    ) -> HandlerResult:
        """Get credits expiring within a period.

        Args:
            org_id: Organization ID
            within_days: Days to look ahead

        Returns:
            List of expiring credit transactions
        """
        manager = get_credit_manager()
        expiring = await manager.get_expiring_credits(
            org_id=org_id,
            within_days=min(within_days, 365),
        )

        # Calculate total expiring amount
        total_expiring = sum(tx.amount_cents for tx in expiring)

        return json_response(
            {
                "expiring_credits": [tx.to_dict() for tx in expiring],
                "total_expiring_cents": total_expiring,
                "total_expiring_usd": total_expiring / 100,
                "within_days": within_days,
            }
        )


def register_credits_admin_routes(app: web.Application, handler: CreditsAdminHandler) -> None:
    """Register credits admin routes with the application.

    Args:
        app: aiohttp Application
        handler: CreditsAdminHandler instance
    """

    async def issue_credit(request: web.Request) -> web.Response:
        org_id = request.match_info["org_id"]
        user_id = request.get("user_id", "admin")
        data = await request.json()
        result = await handler.issue_credit(org_id, data, user_id)
        return web.json_response(result["body"], status=result["status"])

    async def get_credit_account(request: web.Request) -> web.Response:
        org_id = request.match_info["org_id"]
        result = await handler.get_credit_account(org_id)
        return web.json_response(result["body"], status=result["status"])

    async def list_transactions(request: web.Request) -> web.Response:
        org_id = request.match_info["org_id"]
        limit = int(request.query.get("limit", "100"))
        offset = int(request.query.get("offset", "0"))
        tx_type = request.query.get("type")
        result = await handler.list_transactions(org_id, limit, offset, tx_type)
        return web.json_response(result["body"], status=result["status"])

    async def adjust_balance(request: web.Request) -> web.Response:
        org_id = request.match_info["org_id"]
        user_id = request.get("user_id", "admin")
        data = await request.json()
        result = await handler.adjust_balance(org_id, data, user_id)
        return web.json_response(result["body"], status=result["status"])

    async def get_expiring(request: web.Request) -> web.Response:
        org_id = request.match_info["org_id"]
        within_days = int(request.query.get("within_days", "30"))
        result = await handler.get_expiring_credits(org_id, within_days)
        return web.json_response(result["body"], status=result["status"])

    # Register versioned routes
    app.router.add_post("/api/v1/admin/credits/{org_id}/issue", issue_credit)
    app.router.add_get("/api/v1/admin/credits/{org_id}", get_credit_account)
    app.router.add_get("/api/v1/admin/credits/{org_id}/transactions", list_transactions)
    app.router.add_post("/api/v1/admin/credits/{org_id}/adjust", adjust_balance)
    app.router.add_get("/api/v1/admin/credits/{org_id}/expiring", get_expiring)

    # Register non-versioned routes for compatibility
    app.router.add_post("/api/admin/credits/{org_id}/issue", issue_credit)
    app.router.add_get("/api/admin/credits/{org_id}", get_credit_account)
    app.router.add_get("/api/admin/credits/{org_id}/transactions", list_transactions)
    app.router.add_post("/api/admin/credits/{org_id}/adjust", adjust_balance)
    app.router.add_get("/api/admin/credits/{org_id}/expiring", get_expiring)


__all__ = [
    "CreditsAdminHandler",
    "register_credits_admin_routes",
]
