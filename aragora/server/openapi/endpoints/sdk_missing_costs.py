"""SDK missing endpoints for Accounting features.

This module contains endpoint definitions for advanced accounting features
(AP/AR, invoices, expenses) that are not yet implemented.

NOTE: Costs and Payments endpoints have been removed - they are now fully implemented in:
- aragora/server/handlers/costs/ (registered via register_routes)
- aragora/server/handlers/payments/ (registered via register_payment_routes)
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

# Advanced Accounting endpoints (AP/AR, invoices, expenses)
# Basic accounting endpoints (status, connect, disconnect, customers, transactions)
# are implemented in aragora/server/handlers/accounting.py
SDK_MISSING_COSTS_ENDPOINTS: dict = {
    # Accounting endpoints
    "/api/v1/accounting/ap/batch": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST batch",
            "operationId": "postApBatch",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/discounts": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET discounts",
            "operationId": "getApDiscounts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/forecast": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET forecast",
            "operationId": "getApForecast",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/invoices": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET invoices",
            "operationId": "getApInvoices",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Accounting"],
            "summary": "POST invoices",
            "operationId": "postApInvoices",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET AP invoice by ID",
            "operationId": "getApInvoiceById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/invoices/{id}/payment": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST payment",
            "operationId": "postInvoicesPayment",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ap/optimize": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST optimize",
            "operationId": "postApOptimize",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/aging": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET aging",
            "operationId": "getArAging",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/collections": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET collections",
            "operationId": "getArCollections",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/customers": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST customers",
            "operationId": "postArCustomers",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/customers/{id}/balance": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET balance",
            "operationId": "getCustomersBalance",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET invoices",
            "operationId": "getArInvoices",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Accounting"],
            "summary": "POST invoices",
            "operationId": "postArInvoices",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET AR invoice by ID",
            "operationId": "getArInvoiceById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/payment": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST AR invoice payment",
            "operationId": "postArInvoicePayment",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/reminder": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reminder",
            "operationId": "postInvoicesReminder",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/send": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST send",
            "operationId": "postInvoicesSend",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/connect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST connect",
            "operationId": "postAccountingConnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/customers": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET customers",
            "operationId": "getAccountingCustomers",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/disconnect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST disconnect",
            "operationId": "postAccountingDisconnect",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/expenses/{id}": {
        "delete": {
            "tags": ["Accounting"],
            "summary": "DELETE {id}",
            "operationId": "deleteAccountingExpenses",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Accounting"],
            "summary": "GET {id}",
            "operationId": "getAccountingExpenses",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Accounting"],
            "summary": "PUT {id}",
            "operationId": "putAccountingExpenses",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/expenses/{id}/approve": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST approve",
            "operationId": "postExpensesApprove",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/expenses/{id}/reject": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reject",
            "operationId": "postExpensesReject",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET {id}",
            "operationId": "getAccountingInvoices",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/anomalies": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET anomalies",
            "operationId": "getInvoicesAnomalies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/approve": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST approve",
            "operationId": "postInvoicesApprove",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/match": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST match",
            "operationId": "postInvoicesMatch",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/reject": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reject",
            "operationId": "postInvoicesReject",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/schedule": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST schedule",
            "operationId": "postInvoicesSchedule",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/reports": {
        "post": {
            "tags": ["Accounting"],
            "summary": "POST reports",
            "operationId": "postAccountingReports",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/status": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET status",
            "operationId": "getAccountingStatus",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/accounting/transactions": {
        "get": {
            "tags": ["Accounting"],
            "summary": "GET transactions",
            "operationId": "getAccountingTransactions",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
}

__all__ = ["SDK_MISSING_COSTS_ENDPOINTS"]
