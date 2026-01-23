"""Accounting (QuickBooks) endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


ACCOUNTING_ENDPOINTS = {
    "/api/accounting/status": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get accounting status",
            "description": "Fetch QuickBooks connection status and dashboard data.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Status", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/connect": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Start OAuth",
            "description": "Start QuickBooks OAuth flow.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Connect", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/callback": {
        "get": {
            "tags": ["Accounting"],
            "summary": "OAuth callback",
            "description": "Handle QuickBooks OAuth callback.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Callback", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/disconnect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Disconnect",
            "description": "Disconnect QuickBooks integration.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Disconnected", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/customers": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List customers",
            "description": "List QuickBooks customers.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Customers", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/transactions": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List transactions",
            "description": "List QuickBooks transactions.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Transactions", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/report": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Generate report",
            "description": "Generate an accounting report.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": False,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Report", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["ACCOUNTING_ENDPOINTS"]
