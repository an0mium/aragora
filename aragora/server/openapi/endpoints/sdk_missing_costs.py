"""SDK missing endpoints for Costs, Payments, and Accounting.

This module contains endpoint definitions for cost management, payment processing,
and accounting integration features.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

# Costs endpoints
SDK_MISSING_COSTS_ENDPOINTS: dict = {
    "/api/costs": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET costs",
            "operationId": "getCosts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/alerts": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET alerts",
            "operationId": "getCostsAlerts",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/alerts/{id}/dismiss": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST dismiss",
            "operationId": "postAlertsDismiss",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/breakdown": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET breakdown",
            "operationId": "getCostsBreakdown",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/budget": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST budget",
            "operationId": "postCostsBudget",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/constraints/check": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST check",
            "operationId": "postConstraintsCheck",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/efficiency": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET efficiency",
            "operationId": "getCostsEfficiency",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/estimate": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST estimate",
            "operationId": "postCostsEstimate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/forecast": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET forecast",
            "operationId": "getCostsForecast",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/forecast/detailed": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET detailed",
            "operationId": "getForecastDetailed",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/forecast/simulate": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST simulate",
            "operationId": "postForecastSimulate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET recommendations",
            "operationId": "getCostsRecommendations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/detailed": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET detailed",
            "operationId": "getRecommendationsDetailed",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/{id}": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET recommendation by ID",
            "operationId": "getCostsRecommendationById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/{id}/apply": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST apply",
            "operationId": "postRecommendationsApply",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/recommendations/{id}/dismiss": {
        "post": {
            "tags": ["Costs"],
            "summary": "POST dismiss",
            "operationId": "postRecommendationsDismiss",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/costs/timeline": {
        "get": {
            "tags": ["Costs"],
            "summary": "GET timeline",
            "operationId": "getCostsTimeline",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    # Payments endpoints
    "/api/payments/authorize": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST authorize",
            "operationId": "postPaymentsAuthorize",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/capture": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST capture",
            "operationId": "postPaymentsCapture",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/charge": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST charge",
            "operationId": "postPaymentsCharge",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/customer": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST customer",
            "operationId": "postPaymentsCustomer",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/customer/{id}": {
        "delete": {
            "tags": ["Payments"],
            "summary": "DELETE {id}",
            "operationId": "deletePaymentsCustomer",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Payments"],
            "summary": "GET {id}",
            "operationId": "getPaymentsCustomer",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Payments"],
            "summary": "PUT {id}",
            "operationId": "putPaymentsCustomer",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/refund": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST refund",
            "operationId": "postPaymentsRefund",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/subscription": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST subscription",
            "operationId": "postPaymentsSubscription",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/subscription/{id}": {
        "delete": {
            "tags": ["Payments"],
            "summary": "DELETE {id}",
            "operationId": "deletePaymentsSubscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Payments"],
            "summary": "GET {id}",
            "operationId": "getPaymentsSubscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Payments"],
            "summary": "PUT {id}",
            "operationId": "putPaymentsSubscription",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/transaction/{id}": {
        "get": {
            "tags": ["Payments"],
            "summary": "GET {id}",
            "operationId": "getPaymentsTransaction",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/payments/void": {
        "post": {
            "tags": ["Payments"],
            "summary": "POST void",
            "operationId": "postPaymentsVoid",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
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
