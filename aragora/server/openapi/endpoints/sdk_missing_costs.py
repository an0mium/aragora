"""SDK missing endpoints: Planned AP/AR/invoice/expense features (not yet implemented).

This module contains endpoint definitions for advanced accounting features
that are planned but not yet implemented: accounts payable batch operations,
accounts receivable aging/collections, invoice management, expense approval
workflows, and accounting system integrations.

NOTE: Basic costs and payments endpoints have been removed - they are now
fully implemented in:
- aragora/server/handlers/costs/ (registered via register_routes)
- aragora/server/handlers/payments/ (registered via register_payment_routes)
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    STANDARD_ERRORS,
    _ok_response,
)

# =============================================================================
# Response Schemas
# =============================================================================

_INVOICE_SCHEMA = {
    "id": {"type": "string", "description": "Unique invoice identifier"},
    "number": {"type": "string", "description": "Invoice number for display"},
    "vendor_id": {"type": "string", "description": "Vendor/supplier identifier"},
    "customer_id": {"type": "string", "description": "Customer identifier"},
    "status": {
        "type": "string",
        "enum": ["draft", "pending", "approved", "paid", "overdue", "cancelled"],
        "description": "Current invoice status",
    },
    "amount": {"type": "number", "description": "Invoice total amount"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "due_date": {"type": "string", "format": "date", "description": "Payment due date"},
    "issue_date": {"type": "string", "format": "date", "description": "Invoice issue date"},
    "line_items": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "unit_price": {"type": "number"},
                "amount": {"type": "number"},
            },
        },
        "description": "Invoice line items",
    },
    "payment_terms": {"type": "string", "description": "Payment terms (Net 30, etc.)"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_INVOICE_LIST_SCHEMA = {
    "invoices": {
        "type": "array",
        "items": {"type": "object", "properties": _INVOICE_SCHEMA},
        "description": "List of invoices",
    },
    "total": {"type": "integer", "description": "Total number of invoices"},
    "page": {"type": "integer", "description": "Current page number"},
    "page_size": {"type": "integer", "description": "Items per page"},
}

_EXPENSE_SCHEMA = {
    "id": {"type": "string", "description": "Unique expense identifier"},
    "description": {"type": "string", "description": "Expense description"},
    "amount": {"type": "number", "description": "Expense amount"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "category": {"type": "string", "description": "Expense category"},
    "status": {
        "type": "string",
        "enum": ["pending", "approved", "rejected", "reimbursed"],
        "description": "Approval status",
    },
    "submitter_id": {"type": "string", "description": "User who submitted the expense"},
    "approver_id": {"type": "string", "description": "User who approved/rejected"},
    "receipt_url": {"type": "string", "description": "URL to receipt attachment"},
    "expense_date": {"type": "string", "format": "date", "description": "Date of expense"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_CUSTOMER_SCHEMA = {
    "id": {"type": "string", "description": "Unique customer identifier"},
    "name": {"type": "string", "description": "Customer name"},
    "email": {"type": "string", "format": "email", "description": "Primary contact email"},
    "phone": {"type": "string", "description": "Contact phone number"},
    "billing_address": {
        "type": "object",
        "properties": {
            "street": {"type": "string"},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "postal_code": {"type": "string"},
            "country": {"type": "string"},
        },
        "description": "Billing address",
    },
    "payment_terms": {"type": "string", "description": "Default payment terms"},
    "credit_limit": {"type": "number", "description": "Credit limit amount"},
    "created_at": {"type": "string", "format": "date-time"},
}

_CUSTOMER_LIST_SCHEMA = {
    "customers": {
        "type": "array",
        "items": {"type": "object", "properties": _CUSTOMER_SCHEMA},
        "description": "List of customers",
    },
    "total": {"type": "integer", "description": "Total number of customers"},
}

_CUSTOMER_BALANCE_SCHEMA = {
    "customer_id": {"type": "string", "description": "Customer identifier"},
    "total_outstanding": {"type": "number", "description": "Total outstanding balance"},
    "current": {"type": "number", "description": "Current (not yet due) balance"},
    "overdue_1_30": {"type": "number", "description": "1-30 days overdue"},
    "overdue_31_60": {"type": "number", "description": "31-60 days overdue"},
    "overdue_61_90": {"type": "number", "description": "61-90 days overdue"},
    "overdue_90_plus": {"type": "number", "description": "Over 90 days overdue"},
    "credit_available": {"type": "number", "description": "Available credit"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "last_payment_date": {"type": "string", "format": "date", "description": "Last payment date"},
    "last_payment_amount": {"type": "number", "description": "Last payment amount"},
}

_PAYMENT_SCHEMA = {
    "id": {"type": "string", "description": "Payment identifier"},
    "invoice_id": {"type": "string", "description": "Associated invoice ID"},
    "amount": {"type": "number", "description": "Payment amount"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "method": {
        "type": "string",
        "enum": ["bank_transfer", "credit_card", "check", "cash", "wire"],
        "description": "Payment method",
    },
    "status": {
        "type": "string",
        "enum": ["pending", "processing", "completed", "failed", "refunded"],
        "description": "Payment status",
    },
    "reference": {"type": "string", "description": "Payment reference number"},
    "payment_date": {
        "type": "string",
        "format": "date-time",
        "description": "When payment was made",
    },
    "created_at": {"type": "string", "format": "date-time"},
}

_FORECAST_SCHEMA = {
    "period_start": {"type": "string", "format": "date", "description": "Forecast period start"},
    "period_end": {"type": "string", "format": "date", "description": "Forecast period end"},
    "expected_payables": {"type": "number", "description": "Expected accounts payable"},
    "expected_receivables": {"type": "number", "description": "Expected accounts receivable"},
    "net_cash_flow": {"type": "number", "description": "Projected net cash flow"},
    "confidence": {"type": "number", "description": "Forecast confidence (0-1)"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "breakdown": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
                "payables": {"type": "number"},
                "receivables": {"type": "number"},
            },
        },
        "description": "Daily/weekly breakdown",
    },
}

_AGING_REPORT_SCHEMA = {
    "as_of_date": {"type": "string", "format": "date", "description": "Report date"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "summary": {
        "type": "object",
        "properties": {
            "current": {"type": "number"},
            "days_1_30": {"type": "number"},
            "days_31_60": {"type": "number"},
            "days_61_90": {"type": "number"},
            "days_90_plus": {"type": "number"},
            "total": {"type": "number"},
        },
        "description": "Aging summary by period",
    },
    "accounts": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "account_name": {"type": "string"},
                "current": {"type": "number"},
                "days_1_30": {"type": "number"},
                "days_31_60": {"type": "number"},
                "days_61_90": {"type": "number"},
                "days_90_plus": {"type": "number"},
                "total": {"type": "number"},
            },
        },
        "description": "Per-account aging details",
    },
}

_COLLECTIONS_SCHEMA = {
    "total_outstanding": {"type": "number", "description": "Total outstanding amount"},
    "total_overdue": {"type": "number", "description": "Total overdue amount"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "accounts": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "customer_name": {"type": "string"},
                "outstanding": {"type": "number"},
                "overdue": {"type": "number"},
                "days_overdue": {"type": "integer"},
                "last_contact": {"type": "string", "format": "date-time"},
                "next_action": {"type": "string"},
            },
        },
        "description": "Accounts requiring collection",
    },
}

_DISCOUNT_SCHEMA = {
    "vendor_id": {"type": "string", "description": "Vendor identifier"},
    "vendor_name": {"type": "string", "description": "Vendor name"},
    "invoice_id": {"type": "string", "description": "Invoice identifier"},
    "discount_percentage": {"type": "number", "description": "Discount percentage available"},
    "discount_amount": {"type": "number", "description": "Discount amount in currency"},
    "deadline": {"type": "string", "format": "date", "description": "Deadline to claim discount"},
    "terms": {"type": "string", "description": "Discount terms (2/10 Net 30, etc.)"},
}

_DISCOUNT_LIST_SCHEMA = {
    "discounts": {
        "type": "array",
        "items": {"type": "object", "properties": _DISCOUNT_SCHEMA},
        "description": "Available early payment discounts",
    },
    "total_savings": {"type": "number", "description": "Total potential savings"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
}

_BATCH_RESULT_SCHEMA = {
    "batch_id": {"type": "string", "description": "Batch operation identifier"},
    "status": {
        "type": "string",
        "enum": ["pending", "processing", "completed", "partial", "failed"],
        "description": "Batch status",
    },
    "total_items": {"type": "integer", "description": "Total items in batch"},
    "processed": {"type": "integer", "description": "Successfully processed items"},
    "failed": {"type": "integer", "description": "Failed items"},
    "errors": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "error": {"type": "string"},
            },
        },
        "description": "Error details for failed items",
    },
}

_OPTIMIZATION_SCHEMA = {
    "recommendations": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string"},
                "potential_savings": {"type": "number"},
                "invoice_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
        "description": "Payment optimization recommendations",
    },
    "total_potential_savings": {"type": "number", "description": "Total potential savings"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
}

_ANOMALY_SCHEMA = {
    "anomalies": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["duplicate", "price_variance", "unusual_amount", "timing"],
                },
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                "description": {"type": "string"},
                "confidence": {"type": "number"},
                "related_invoices": {"type": "array", "items": {"type": "string"}},
            },
        },
        "description": "Detected anomalies",
    },
    "checked_at": {"type": "string", "format": "date-time"},
}

_CONNECTION_STATUS_SCHEMA = {
    "connected": {"type": "boolean", "description": "Whether accounting system is connected"},
    "provider": {
        "type": "string",
        "enum": ["quickbooks", "xero", "sage", "netsuite", "freshbooks"],
        "description": "Connected accounting provider",
    },
    "last_sync": {"type": "string", "format": "date-time", "description": "Last sync timestamp"},
    "sync_status": {
        "type": "string",
        "enum": ["synced", "syncing", "error", "stale"],
        "description": "Current sync status",
    },
    "company_name": {"type": "string", "description": "Connected company name"},
}

_TRANSACTION_SCHEMA = {
    "id": {"type": "string", "description": "Transaction identifier"},
    "type": {
        "type": "string",
        "enum": ["invoice", "payment", "refund", "credit_note", "adjustment"],
        "description": "Transaction type",
    },
    "amount": {"type": "number", "description": "Transaction amount"},
    "currency": {"type": "string", "description": "ISO 4217 currency code"},
    "description": {"type": "string", "description": "Transaction description"},
    "date": {"type": "string", "format": "date-time", "description": "Transaction date"},
    "account_id": {"type": "string", "description": "Related account"},
    "reference": {"type": "string", "description": "External reference"},
}

_TRANSACTION_LIST_SCHEMA = {
    "transactions": {
        "type": "array",
        "items": {"type": "object", "properties": _TRANSACTION_SCHEMA},
        "description": "List of transactions",
    },
    "total": {"type": "integer", "description": "Total transaction count"},
    "page": {"type": "integer", "description": "Current page"},
    "page_size": {"type": "integer", "description": "Items per page"},
}

_REPORT_SCHEMA = {
    "report_id": {"type": "string", "description": "Generated report identifier"},
    "type": {
        "type": "string",
        "enum": ["aging", "cash_flow", "profit_loss", "balance_sheet", "custom"],
        "description": "Report type",
    },
    "generated_at": {"type": "string", "format": "date-time"},
    "period_start": {"type": "string", "format": "date"},
    "period_end": {"type": "string", "format": "date"},
    "download_url": {"type": "string", "description": "URL to download report"},
    "format": {"type": "string", "enum": ["pdf", "csv", "xlsx", "json"]},
}

_REMINDER_SCHEMA = {
    "reminder_id": {"type": "string", "description": "Reminder identifier"},
    "invoice_id": {"type": "string", "description": "Associated invoice"},
    "sent_at": {"type": "string", "format": "date-time", "description": "When reminder was sent"},
    "channel": {
        "type": "string",
        "enum": ["email", "sms", "mail"],
        "description": "Delivery channel",
    },
    "status": {
        "type": "string",
        "enum": ["sent", "delivered", "failed", "bounced"],
        "description": "Delivery status",
    },
}

_SCHEDULE_SCHEMA = {
    "schedule_id": {"type": "string", "description": "Schedule identifier"},
    "invoice_id": {"type": "string", "description": "Associated invoice"},
    "payment_date": {"type": "string", "format": "date", "description": "Scheduled payment date"},
    "amount": {"type": "number", "description": "Scheduled payment amount"},
    "status": {
        "type": "string",
        "enum": ["scheduled", "processing", "completed", "cancelled"],
        "description": "Schedule status",
    },
}

_MATCH_RESULT_SCHEMA = {
    "invoice_id": {"type": "string", "description": "Invoice identifier"},
    "matched": {"type": "boolean", "description": "Whether match was successful"},
    "matched_documents": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "document_id": {"type": "string"},
                "match_confidence": {"type": "number"},
            },
        },
        "description": "Matched purchase orders, receipts, etc.",
    },
    "discrepancies": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Any discrepancies found",
    },
}

_ACTION_RESULT_SCHEMA = {
    "success": {"type": "boolean", "description": "Whether action succeeded"},
    "message": {"type": "string", "description": "Result message"},
    "timestamp": {"type": "string", "format": "date-time"},
}

# =============================================================================
# Request Body Schemas
# =============================================================================

_BATCH_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of invoice IDs to process",
        },
        "action": {
            "type": "string",
            "enum": ["approve", "pay", "reject"],
            "description": "Batch action to perform",
        },
    },
    "required": ["invoice_ids", "action"],
}

_INVOICE_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor_id": {"type": "string", "description": "Vendor identifier"},
        "customer_id": {"type": "string", "description": "Customer identifier"},
        "amount": {"type": "number", "description": "Invoice amount"},
        "currency": {"type": "string", "description": "ISO 4217 currency code"},
        "due_date": {"type": "string", "format": "date", "description": "Payment due date"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                },
            },
        },
    },
    "required": ["amount", "due_date"],
}

_PAYMENT_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "description": "Payment amount"},
        "method": {
            "type": "string",
            "enum": ["bank_transfer", "credit_card", "check"],
            "description": "Payment method",
        },
        "reference": {"type": "string", "description": "Payment reference"},
    },
    "required": ["amount", "method"],
}

_CUSTOMER_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Customer name"},
        "email": {"type": "string", "format": "email", "description": "Contact email"},
        "phone": {"type": "string", "description": "Contact phone"},
        "billing_address": {"type": "object", "description": "Billing address"},
        "payment_terms": {"type": "string", "description": "Default payment terms"},
    },
    "required": ["name"],
}

_EXPENSE_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "amount": {"type": "number"},
        "category": {"type": "string"},
        "receipt_url": {"type": "string"},
    },
}

_OPTIMIZE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["maximize_discounts", "minimize_cash_outflow", "balance"],
            "description": "Optimization strategy",
        },
        "available_cash": {"type": "number", "description": "Available cash for payments"},
    },
}

_CONNECT_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "provider": {
            "type": "string",
            "enum": ["quickbooks", "xero", "sage", "netsuite", "freshbooks"],
            "description": "Accounting provider to connect",
        },
        "auth_code": {"type": "string", "description": "OAuth authorization code"},
        "redirect_uri": {"type": "string", "description": "OAuth redirect URI"},
    },
    "required": ["provider", "auth_code"],
}

_REPORT_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["aging", "cash_flow", "profit_loss", "balance_sheet"],
            "description": "Report type",
        },
        "period_start": {"type": "string", "format": "date"},
        "period_end": {"type": "string", "format": "date"},
        "format": {"type": "string", "enum": ["pdf", "csv", "xlsx"]},
    },
    "required": ["type"],
}

_REMINDER_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "channel": {"type": "string", "enum": ["email", "sms"], "description": "Delivery channel"},
        "message": {"type": "string", "description": "Custom reminder message"},
    },
}

_SCHEDULE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "payment_date": {
            "type": "string",
            "format": "date",
            "description": "Scheduled payment date",
        },
        "amount": {"type": "number", "description": "Payment amount (defaults to full invoice)"},
    },
    "required": ["payment_date"],
}

_APPROVAL_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {"type": "string", "description": "Approval notes"},
    },
}

_REJECTION_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "Rejection reason"},
    },
    "required": ["reason"],
}

# =============================================================================
# Endpoint Definitions
# =============================================================================

# Advanced Accounting endpoints (AP/AR, invoices, expenses)
# Basic accounting endpoints (status, connect, disconnect, customers, transactions)
# are implemented in aragora/server/handlers/accounting.py
SDK_MISSING_COSTS_ENDPOINTS: dict = {
    # AP (Accounts Payable) endpoints
    "/api/v1/accounting/ap/batch": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Process AP invoices in batch",
            "description": "Perform batch operations on multiple accounts payable invoices.",
            "operationId": "postApBatch",
            "requestBody": {"content": {"application/json": {"schema": _BATCH_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Batch operation result", _BATCH_RESULT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/accounting/ap/discounts": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List available early payment discounts",
            "description": "Get all available early payment discounts for accounts payable.",
            "operationId": "getApDiscounts",
            "responses": {
                "200": _ok_response("Available discounts", _DISCOUNT_LIST_SCHEMA),
            },
        },
    },
    "/api/v1/accounting/ap/forecast": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get AP cash flow forecast",
            "description": "Forecast upcoming accounts payable cash requirements.",
            "operationId": "getApForecast",
            "parameters": [
                {
                    "name": "days",
                    "in": "query",
                    "schema": {"type": "integer", "default": 30},
                    "description": "Forecast period in days",
                },
            ],
            "responses": {
                "200": _ok_response("AP forecast", _FORECAST_SCHEMA),
            },
        },
    },
    "/api/v1/accounting/ap/invoices": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List AP invoices",
            "description": "Get all accounts payable invoices with optional filtering.",
            "operationId": "getApInvoices",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by status",
                },
                {
                    "name": "vendor_id",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by vendor",
                },
                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {
                "200": _ok_response("AP invoices list", _INVOICE_LIST_SCHEMA),
            },
        },
        "post": {
            "tags": ["Accounting"],
            "summary": "Create AP invoice",
            "description": "Create a new accounts payable invoice.",
            "operationId": "postApInvoices",
            "requestBody": {"content": {"application/json": {"schema": _INVOICE_CREATE_SCHEMA}}},
            "responses": {
                "200": _ok_response("Created invoice", _INVOICE_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/accounting/ap/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get AP invoice by ID",
            "description": "Retrieve a specific accounts payable invoice.",
            "operationId": "getApInvoiceById",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "responses": {
                "200": _ok_response("Invoice details", _INVOICE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/ap/invoices/{id}/payment": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Record payment for AP invoice",
            "description": "Record a payment against an accounts payable invoice.",
            "operationId": "postApInvoicePayment",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _PAYMENT_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Payment recorded", _PAYMENT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/ap/optimize": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Optimize AP payments",
            "description": "Get recommendations for optimizing accounts payable payments.",
            "operationId": "postApOptimize",
            "requestBody": {"content": {"application/json": {"schema": _OPTIMIZE_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Payment optimization", _OPTIMIZATION_SCHEMA),
            },
        },
    },
    # AR (Accounts Receivable) endpoints
    "/api/v1/accounting/ar/aging": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get AR aging report",
            "description": "Get accounts receivable aging report showing overdue amounts.",
            "operationId": "getArAging",
            "parameters": [
                {
                    "name": "as_of",
                    "in": "query",
                    "schema": {"type": "string", "format": "date"},
                    "description": "Report date",
                },
            ],
            "responses": {
                "200": _ok_response("AR aging report", _AGING_REPORT_SCHEMA),
            },
        },
    },
    "/api/v1/accounting/ar/collections": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get AR collections status",
            "description": "Get accounts requiring collection attention.",
            "operationId": "getArCollections",
            "responses": {
                "200": _ok_response("Collections status", _COLLECTIONS_SCHEMA),
            },
        },
    },
    "/api/v1/accounting/ar/customers": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Create AR customer",
            "description": "Create a new customer for accounts receivable.",
            "operationId": "postArCustomers",
            "requestBody": {"content": {"application/json": {"schema": _CUSTOMER_CREATE_SCHEMA}}},
            "responses": {
                "200": _ok_response("Created customer", _CUSTOMER_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/accounting/ar/customers/{id}/balance": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get customer balance",
            "description": "Get detailed balance information for a customer.",
            "operationId": "getArCustomerBalance",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Customer ID",
                }
            ],
            "responses": {
                "200": _ok_response("Customer balance", _CUSTOMER_BALANCE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/ar/invoices": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List AR invoices",
            "description": "Get all accounts receivable invoices with optional filtering.",
            "operationId": "getArInvoices",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by status",
                },
                {
                    "name": "customer_id",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by customer",
                },
                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {
                "200": _ok_response("AR invoices list", _INVOICE_LIST_SCHEMA),
            },
        },
        "post": {
            "tags": ["Accounting"],
            "summary": "Create AR invoice",
            "description": "Create a new accounts receivable invoice.",
            "operationId": "postArInvoices",
            "requestBody": {"content": {"application/json": {"schema": _INVOICE_CREATE_SCHEMA}}},
            "responses": {
                "200": _ok_response("Created invoice", _INVOICE_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get AR invoice by ID",
            "description": "Retrieve a specific accounts receivable invoice.",
            "operationId": "getArInvoiceById",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "responses": {
                "200": _ok_response("Invoice details", _INVOICE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/payment": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Record payment for AR invoice",
            "description": "Record a payment received against an accounts receivable invoice.",
            "operationId": "postArInvoicePayment",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _PAYMENT_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Payment recorded", _PAYMENT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/reminder": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Send payment reminder",
            "description": "Send a payment reminder for an overdue invoice.",
            "operationId": "postArInvoiceReminder",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _REMINDER_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Reminder sent", _REMINDER_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/ar/invoices/{id}/send": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Send invoice to customer",
            "description": "Send an invoice to the customer via email.",
            "operationId": "postArInvoiceSend",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "email": {
                                    "type": "string",
                                    "format": "email",
                                    "description": "Override recipient email",
                                },
                                "message": {"type": "string", "description": "Custom message"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Invoice sent", _ACTION_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    # General Accounting endpoints
    "/api/v1/accounting/connect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Connect accounting system",
            "description": "Connect to an external accounting system (QuickBooks, Xero, etc.).",
            "operationId": "postAccountingConnect",
            "requestBody": {"content": {"application/json": {"schema": _CONNECT_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Connection established", _CONNECTION_STATUS_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/accounting/customers": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List all customers",
            "description": "Get all customers from the connected accounting system.",
            "operationId": "getAccountingCustomers",
            "parameters": [
                {
                    "name": "search",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Search by name/email",
                },
                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {
                "200": _ok_response("Customers list", _CUSTOMER_LIST_SCHEMA),
            },
        },
    },
    "/api/v1/accounting/disconnect": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Disconnect accounting system",
            "description": "Disconnect from the connected accounting system.",
            "operationId": "postAccountingDisconnect",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "confirm": {
                                    "type": "boolean",
                                    "description": "Confirm disconnection",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Disconnected", _ACTION_RESULT_SCHEMA),
            },
        },
    },
    # Expense endpoints
    "/api/v1/accounting/expenses/{id}": {
        "delete": {
            "tags": ["Accounting"],
            "summary": "Delete expense",
            "description": "Delete an expense record.",
            "operationId": "deleteAccountingExpense",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Expense ID",
                }
            ],
            "responses": {
                "200": _ok_response("Expense deleted", _ACTION_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Accounting"],
            "summary": "Get expense by ID",
            "description": "Retrieve a specific expense record.",
            "operationId": "getAccountingExpense",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Expense ID",
                }
            ],
            "responses": {
                "200": _ok_response("Expense details", _EXPENSE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Accounting"],
            "summary": "Update expense",
            "description": "Update an existing expense record.",
            "operationId": "putAccountingExpense",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Expense ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _EXPENSE_UPDATE_SCHEMA}}},
            "responses": {
                "200": _ok_response("Updated expense", _EXPENSE_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/expenses/{id}/approve": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Approve expense",
            "description": "Approve an expense for reimbursement.",
            "operationId": "postExpenseApprove",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Expense ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _APPROVAL_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Expense approved", _EXPENSE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/expenses/{id}/reject": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Reject expense",
            "description": "Reject an expense submission.",
            "operationId": "postExpenseReject",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Expense ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _REJECTION_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Expense rejected", _EXPENSE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    # Invoice management endpoints
    "/api/v1/accounting/invoices/{id}": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get invoice by ID",
            "description": "Retrieve any invoice by its ID.",
            "operationId": "getAccountingInvoice",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "responses": {
                "200": _ok_response("Invoice details", _INVOICE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/anomalies": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Detect invoice anomalies",
            "description": "Detect potential anomalies (duplicates, price variance, etc.) for an invoice.",
            "operationId": "getInvoiceAnomalies",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "responses": {
                "200": _ok_response("Anomaly detection results", _ANOMALY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/approve": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Approve invoice",
            "description": "Approve an invoice for payment.",
            "operationId": "postInvoiceApprove",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _APPROVAL_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Invoice approved", _INVOICE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/match": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Match invoice to documents",
            "description": "Match an invoice to purchase orders, receipts, and other documents.",
            "operationId": "postInvoiceMatch",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "document_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Document IDs to match against",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Match results", _MATCH_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/reject": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Reject invoice",
            "description": "Reject an invoice.",
            "operationId": "postInvoiceReject",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _REJECTION_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Invoice rejected", _INVOICE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/accounting/invoices/{id}/schedule": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Schedule invoice payment",
            "description": "Schedule a payment for an invoice.",
            "operationId": "postInvoiceSchedule",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Invoice ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _SCHEDULE_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Payment scheduled", _SCHEDULE_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    # Reports and status
    "/api/v1/accounting/reports": {
        "post": {
            "tags": ["Accounting"],
            "summary": "Generate accounting report",
            "description": "Generate an accounting report (aging, cash flow, P&L, etc.).",
            "operationId": "postAccountingReports",
            "requestBody": {"content": {"application/json": {"schema": _REPORT_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Generated report", _REPORT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/accounting/status": {
        "get": {
            "tags": ["Accounting"],
            "summary": "Get accounting connection status",
            "description": "Get the current status of accounting system connections.",
            "operationId": "getAccountingStatus",
            "responses": {
                "200": _ok_response("Connection status", _CONNECTION_STATUS_SCHEMA),
            },
        },
    },
    "/api/v1/accounting/transactions": {
        "get": {
            "tags": ["Accounting"],
            "summary": "List transactions",
            "description": "Get all accounting transactions with optional filtering.",
            "operationId": "getAccountingTransactions",
            "parameters": [
                {
                    "name": "type",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by type",
                },
                {
                    "name": "from_date",
                    "in": "query",
                    "schema": {"type": "string", "format": "date"},
                    "description": "Start date",
                },
                {
                    "name": "to_date",
                    "in": "query",
                    "schema": {"type": "string", "format": "date"},
                    "description": "End date",
                },
                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {
                "200": _ok_response("Transactions list", _TRANSACTION_LIST_SCHEMA),
            },
        },
    },
}

__all__ = ["SDK_MISSING_COSTS_ENDPOINTS"]
