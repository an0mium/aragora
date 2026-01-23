"""Accounting (QuickBooks + Gusto) endpoint definitions."""

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
    "/api/accounting/gusto/status": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Gusto status",
            "description": "Fetch Gusto connection status.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Status", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/connect": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Start Gusto OAuth",
            "description": "Start Gusto OAuth flow.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Connect", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/callback": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Gusto OAuth callback",
            "description": "Handle Gusto OAuth callback.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Callback", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/disconnect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Disconnect Gusto",
            "description": "Disconnect Gusto integration.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Disconnected", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/employees": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List employees",
            "description": "List employees from Gusto.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "active", "in": "query", "schema": {"type": "boolean"}},
            ],
            "responses": {
                "200": _ok_response("Employees", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/payrolls": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List payrolls",
            "description": "List payroll runs from Gusto.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "start_date", "in": "query", "schema": {"type": "string"}},
                {"name": "end_date", "in": "query", "schema": {"type": "string"}},
                {"name": "processed", "in": "query", "schema": {"type": "boolean"}},
            ],
            "responses": {
                "200": _ok_response("Payrolls", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/payrolls/{payroll_id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get payroll",
            "description": "Fetch a payroll run by ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "payroll_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Payroll", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/accounting/gusto/payrolls/{payroll_id}/journal-entry": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Generate journal entry",
            "description": "Generate a journal entry for a payroll run.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "payroll_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": False,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Journal entry", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["ACCOUNTING_ENDPOINTS"]
