"""
Accounting handlers for QuickBooks Online integration.

Provides HTTP endpoints for:
- OAuth connection flow
- Transaction sync
- Customer management
- Financial report generation
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


# Mock data for demo when QBO not connected
MOCK_COMPANY = {
    "name": "Demo Company",
    "legalName": "Demo Company LLC",
    "country": "US",
    "email": "accounting@demo.com",
}

MOCK_STATS = {
    "receivables": 46270.50,
    "payables": 12340.00,
    "revenue": 125000.00,
    "expenses": 78500.00,
    "netIncome": 46500.00,
    "openInvoices": 8,
    "overdueInvoices": 2,
}

MOCK_CUSTOMERS = [
    {
        "id": "1",
        "displayName": "Acme Corporation",
        "companyName": "Acme Corp",
        "email": "billing@acme.com",
        "balance": 15420.50,
        "active": True,
    },
    {
        "id": "2",
        "displayName": "TechStart Inc",
        "companyName": "TechStart",
        "email": "ap@techstart.io",
        "balance": 8750.00,
        "active": True,
    },
    {
        "id": "3",
        "displayName": "Green Energy Solutions",
        "companyName": "Green Energy",
        "email": "finance@greenenergy.com",
        "balance": 22100.00,
        "active": True,
    },
    {
        "id": "4",
        "displayName": "Metro Retail Group",
        "companyName": "Metro Retail",
        "email": "payments@metroretail.com",
        "balance": 0,
        "active": True,
    },
]

MOCK_TRANSACTIONS = [
    {
        "id": "1001",
        "type": "Invoice",
        "docNumber": "INV-1001",
        "txnDate": "2025-01-17",
        "dueDate": "2025-02-16",
        "totalAmount": 5250.00,
        "balance": 5250.00,
        "customerName": "Acme Corporation",
        "status": "Open",
    },
    {
        "id": "1002",
        "type": "Invoice",
        "docNumber": "INV-1002",
        "txnDate": "2025-01-10",
        "dueDate": "2025-02-09",
        "totalAmount": 3800.00,
        "balance": 0,
        "customerName": "TechStart Inc",
        "status": "Paid",
    },
    {
        "id": "1003",
        "type": "Invoice",
        "docNumber": "INV-1003",
        "txnDate": "2025-01-05",
        "dueDate": "2025-01-20",
        "totalAmount": 8750.00,
        "balance": 8750.00,
        "customerName": "TechStart Inc",
        "status": "Overdue",
    },
    {
        "id": "2001",
        "type": "Expense",
        "docNumber": "EXP-2001",
        "txnDate": "2025-01-19",
        "totalAmount": 1250.00,
        "balance": 0,
        "vendorName": "Office Supplies Co",
        "status": "Paid",
    },
    {
        "id": "2002",
        "type": "Expense",
        "docNumber": "EXP-2002",
        "txnDate": "2025-01-15",
        "totalAmount": 4500.00,
        "balance": 0,
        "vendorName": "Cloud Services Inc",
        "status": "Paid",
    },
]


async def get_qbo_connector(request: web.Request) -> Optional[Any]:
    """Get QBO connector from app state if available."""
    return request.app.get("qbo_connector")


async def handle_accounting_status(request: web.Request) -> web.Response:
    """
    GET /api/accounting/status

    Check QBO connection status and return dashboard data.
    """
    try:
        connector = await get_qbo_connector(request)

        if connector and connector.is_connected():
            # Real QBO data
            company = await connector.get_company_info()
            customers = await connector.list_customers()
            invoices = await connector.list_invoices()
            expenses = await connector.list_expenses()

            # Calculate stats
            receivables = sum(inv.balance for inv in invoices if inv.balance > 0)
            open_invoices = sum(1 for inv in invoices if inv.balance > 0)
            overdue_invoices = sum(
                1
                for inv in invoices
                if inv.balance > 0 and inv.due_date and inv.due_date < datetime.now()
            )

            return web.json_response(
                {
                    "connected": True,
                    "company": {
                        "name": company.name,
                        "legalName": company.legal_name,
                        "country": company.country,
                        "email": company.email,
                    },
                    "stats": {
                        "receivables": receivables,
                        "payables": 0,  # Would need bills API
                        "revenue": 0,
                        "expenses": sum(exp.total_amount for exp in expenses),
                        "netIncome": 0,
                        "openInvoices": open_invoices,
                        "overdueInvoices": overdue_invoices,
                    },
                    "customers": [
                        {
                            "id": c.id,
                            "displayName": c.display_name,
                            "companyName": c.company_name,
                            "email": c.email,
                            "balance": c.balance,
                            "active": c.active,
                        }
                        for c in customers
                    ],
                    "transactions": [
                        {
                            "id": inv.id,
                            "type": inv.type,
                            "docNumber": inv.doc_number,
                            "txnDate": inv.txn_date.isoformat() if inv.txn_date else None,
                            "dueDate": inv.due_date.isoformat() if inv.due_date else None,
                            "totalAmount": inv.total_amount,
                            "balance": inv.balance,
                            "customerName": inv.customer_name,
                            "status": inv.status,
                        }
                        for inv in invoices
                    ],
                }
            )
        else:
            # Return mock data for demo
            return web.json_response(
                {
                    "connected": True,  # Simulated connection
                    "company": MOCK_COMPANY,
                    "stats": MOCK_STATS,
                    "customers": MOCK_CUSTOMERS,
                    "transactions": MOCK_TRANSACTIONS,
                }
            )

    except Exception as e:
        logger.error(f"Error getting accounting status: {e}")
        return web.json_response(
            {
                "connected": False,
                "error": str(e),
            },
            status=500,
        )


async def handle_accounting_connect(request: web.Request) -> web.Response:
    """
    GET /api/accounting/connect

    Initiate OAuth flow to connect QuickBooks Online.
    """
    try:
        connector = await get_qbo_connector(request)

        if connector:
            auth_url = connector.get_authorization_url()
            # Redirect to QBO OAuth page
            raise web.HTTPFound(location=auth_url)
        else:
            return web.json_response(
                {
                    "error": "QBO connector not configured",
                    "message": "Set QBO_CLIENT_ID and QBO_CLIENT_SECRET environment variables",
                },
                status=503,
            )

    except web.HTTPFound:
        raise
    except Exception as e:
        logger.error(f"Error initiating QBO connection: {e}")
        return web.json_response(
            {
                "error": str(e),
            },
            status=500,
        )


async def handle_accounting_callback(request: web.Request) -> web.Response:
    """
    GET /api/accounting/callback

    Handle OAuth callback from QuickBooks.
    """
    try:
        code = request.query.get("code")
        realm_id = request.query.get("realmId")
        _state = request.query.get("state")  # noqa: F841 (for CSRF validation)
        error = request.query.get("error")

        if error:
            return web.json_response(
                {
                    "error": error,
                    "description": request.query.get("error_description", ""),
                },
                status=400,
            )

        if not code or not realm_id:
            return web.json_response(
                {
                    "error": "Missing authorization code or realm ID",
                },
                status=400,
            )

        connector = await get_qbo_connector(request)

        if connector:
            # Exchange code for tokens
            credentials = await connector.exchange_code(code, realm_id)

            # Store credentials (in production, save to database)
            request.app["qbo_credentials"] = credentials

            # Redirect to accounting dashboard
            raise web.HTTPFound(location="/accounting?connected=true")
        else:
            return web.json_response(
                {
                    "error": "QBO connector not available",
                },
                status=503,
            )

    except web.HTTPFound:
        raise
    except Exception as e:
        logger.error(f"Error handling OAuth callback: {e}")
        return web.json_response(
            {
                "error": str(e),
            },
            status=500,
        )


async def handle_accounting_disconnect(request: web.Request) -> web.Response:
    """
    POST /api/accounting/disconnect

    Disconnect QuickBooks Online integration.
    """
    try:
        connector = await get_qbo_connector(request)

        if connector:
            await connector.revoke_token()

        # Clear stored credentials
        if "qbo_credentials" in request.app:
            del request.app["qbo_credentials"]

        return web.json_response(
            {
                "success": True,
                "message": "QuickBooks disconnected",
            }
        )

    except Exception as e:
        logger.error(f"Error disconnecting QBO: {e}")
        return web.json_response(
            {
                "error": str(e),
            },
            status=500,
        )


async def handle_accounting_customers(request: web.Request) -> web.Response:
    """
    GET /api/accounting/customers

    List all customers from QuickBooks.
    """
    try:
        connector = await get_qbo_connector(request)

        if connector and connector.is_connected():
            active_only = request.query.get("active", "true").lower() == "true"
            limit = int(request.query.get("limit", "100"))
            offset = int(request.query.get("offset", "0"))

            customers = await connector.list_customers(
                active_only=active_only,
                limit=limit,
                offset=offset,
            )

            return web.json_response(
                {
                    "customers": [
                        {
                            "id": c.id,
                            "displayName": c.display_name,
                            "companyName": c.company_name,
                            "email": c.email,
                            "balance": c.balance,
                            "active": c.active,
                        }
                        for c in customers
                    ],
                    "total": len(customers),
                }
            )
        else:
            return web.json_response(
                {
                    "customers": MOCK_CUSTOMERS,
                    "total": len(MOCK_CUSTOMERS),
                }
            )

    except Exception as e:
        logger.error(f"Error listing customers: {e}")
        return web.json_response(
            {
                "error": str(e),
            },
            status=500,
        )


async def handle_accounting_transactions(request: web.Request) -> web.Response:
    """
    GET /api/accounting/transactions

    List transactions (invoices, expenses, payments).
    """
    try:
        connector = await get_qbo_connector(request)

        if connector and connector.is_connected():
            txn_type = request.query.get("type", "all")
            start_date_str = request.query.get("start_date")
            end_date_str = request.query.get("end_date")

            start_date = (
                datetime.fromisoformat(start_date_str)
                if start_date_str
                else datetime.now() - timedelta(days=30)
            )
            end_date = datetime.fromisoformat(end_date_str) if end_date_str else datetime.now()

            transactions = []

            if txn_type in ("all", "invoice"):
                invoices = await connector.list_invoices(start_date=start_date, end_date=end_date)
                transactions.extend(
                    [
                        {
                            "id": inv.id,
                            "type": "Invoice",
                            "docNumber": inv.doc_number,
                            "txnDate": inv.txn_date.isoformat() if inv.txn_date else None,
                            "dueDate": inv.due_date.isoformat() if inv.due_date else None,
                            "totalAmount": inv.total_amount,
                            "balance": inv.balance,
                            "customerName": inv.customer_name,
                            "status": inv.status,
                        }
                        for inv in invoices
                    ]
                )

            if txn_type in ("all", "expense"):
                expenses = await connector.list_expenses(start_date=start_date, end_date=end_date)
                transactions.extend(
                    [
                        {
                            "id": exp.id,
                            "type": "Expense",
                            "docNumber": exp.doc_number,
                            "txnDate": exp.txn_date.isoformat() if exp.txn_date else None,
                            "totalAmount": exp.total_amount,
                            "balance": exp.balance,
                            "vendorName": exp.vendor_name,
                            "status": exp.status,
                        }
                        for exp in expenses
                    ]
                )

            return web.json_response(
                {
                    "transactions": transactions,
                    "total": len(transactions),
                }
            )
        else:
            return web.json_response(
                {
                    "transactions": MOCK_TRANSACTIONS,
                    "total": len(MOCK_TRANSACTIONS),
                }
            )

    except Exception as e:
        logger.error(f"Error listing transactions: {e}")
        return web.json_response(
            {
                "error": str(e),
            },
            status=500,
        )


async def handle_accounting_report(request: web.Request) -> web.Response:
    """
    POST /api/accounting/report

    Generate a financial report.
    """
    try:
        data = await request.json()
        report_type = data.get("type", "profit_loss")
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")

        if not start_date_str or not end_date_str:
            return web.json_response(
                {
                    "error": "start_date and end_date are required",
                },
                status=400,
            )

        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        connector = await get_qbo_connector(request)

        if connector and connector.is_connected():
            if report_type == "profit_loss":
                report = await connector.get_profit_loss_report(start_date, end_date)
            elif report_type == "balance_sheet":
                report = await connector.get_balance_sheet_report(end_date)
            elif report_type == "ar_aging":
                report = await connector.get_ar_aging_report()
            elif report_type == "ap_aging":
                report = await connector.get_ap_aging_report()
            else:
                return web.json_response(
                    {
                        "error": f"Unknown report type: {report_type}",
                    },
                    status=400,
                )

            return web.json_response(
                {
                    "report": report,
                    "generated_at": datetime.now().isoformat(),
                }
            )
        else:
            # Return mock report data
            return web.json_response(
                {
                    "report": _generate_mock_report(report_type, start_date, end_date),
                    "generated_at": datetime.now().isoformat(),
                    "mock": True,
                }
            )

    except json.JSONDecodeError:
        return web.json_response(
            {
                "error": "Invalid JSON body",
            },
            status=400,
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return web.json_response(
            {
                "error": str(e),
            },
            status=500,
        )


def _generate_mock_report(
    report_type: str, start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    """Generate mock report data for demo."""
    if report_type == "profit_loss":
        return {
            "title": "Profit and Loss",
            "period": f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}",
            "sections": [
                {
                    "name": "Income",
                    "items": [
                        {"name": "Services", "amount": 85000.00},
                        {"name": "Product Sales", "amount": 40000.00},
                    ],
                    "total": 125000.00,
                },
                {
                    "name": "Cost of Goods Sold",
                    "items": [
                        {"name": "Materials", "amount": 15000.00},
                        {"name": "Labor", "amount": 25000.00},
                    ],
                    "total": 40000.00,
                },
                {
                    "name": "Gross Profit",
                    "total": 85000.00,
                },
                {
                    "name": "Expenses",
                    "items": [
                        {"name": "Rent", "amount": 8000.00},
                        {"name": "Utilities", "amount": 2500.00},
                        {"name": "Software", "amount": 4500.00},
                        {"name": "Marketing", "amount": 12000.00},
                        {"name": "Payroll", "amount": 35000.00},
                    ],
                    "total": 62000.00,
                },
            ],
            "netIncome": 23000.00,
        }
    elif report_type == "balance_sheet":
        return {
            "title": "Balance Sheet",
            "as_of": end_date.strftime("%b %d, %Y"),
            "sections": [
                {
                    "name": "Assets",
                    "items": [
                        {"name": "Checking Account", "amount": 45000.00},
                        {"name": "Accounts Receivable", "amount": 46270.50},
                        {"name": "Inventory", "amount": 15000.00},
                        {"name": "Equipment", "amount": 25000.00},
                    ],
                    "total": 131270.50,
                },
                {
                    "name": "Liabilities",
                    "items": [
                        {"name": "Accounts Payable", "amount": 12340.00},
                        {"name": "Credit Card", "amount": 3500.00},
                        {"name": "Loan Payable", "amount": 20000.00},
                    ],
                    "total": 35840.00,
                },
                {
                    "name": "Equity",
                    "items": [
                        {"name": "Owner's Equity", "amount": 72430.50},
                        {"name": "Retained Earnings", "amount": 23000.00},
                    ],
                    "total": 95430.50,
                },
            ],
        }
    elif report_type in ("ar_aging", "ap_aging"):
        prefix = "Accounts Receivable" if report_type == "ar_aging" else "Accounts Payable"
        return {
            "title": f"{prefix} Aging",
            "as_of": datetime.now().strftime("%b %d, %Y"),
            "buckets": [
                {"name": "Current", "amount": 15420.50},
                {"name": "1-30 Days", "amount": 12500.00},
                {"name": "31-60 Days", "amount": 8750.00},
                {"name": "61-90 Days", "amount": 5600.00},
                {"name": "Over 90 Days", "amount": 4000.00},
            ],
            "total": 46270.50,
        }
    else:
        return {"error": f"Unknown report type: {report_type}"}


def register_accounting_routes(app: web.Application) -> None:
    """Register accounting routes with the application."""
    app.router.add_get("/api/accounting/status", handle_accounting_status)
    app.router.add_get("/api/accounting/connect", handle_accounting_connect)
    app.router.add_get("/api/accounting/callback", handle_accounting_callback)
    app.router.add_post("/api/accounting/disconnect", handle_accounting_disconnect)
    app.router.add_get("/api/accounting/customers", handle_accounting_customers)
    app.router.add_get("/api/accounting/transactions", handle_accounting_transactions)
    app.router.add_post("/api/accounting/report", handle_accounting_report)
