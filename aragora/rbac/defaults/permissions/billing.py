"""
RBAC Permissions for Billing, Finance, Cost, and Payment resources.

Contains permissions related to:
- Billing and subscriptions
- Finance operations
- Cost management and budgets
- Payments processing
- Accounts payable
- Expenses
- Quotas
"""

from __future__ import annotations

from aragora.rbac.models import Action, Permission, ResourceType

from ._helpers import _permission

# ============================================================================
# BILLING PERMISSIONS
# ============================================================================

PERM_BILLING_READ = _permission(
    ResourceType.BILLING,
    Action.READ,
    "View Cost Data",
    "View cost analytics and efficiency metrics",
)
PERM_BILLING_RECOMMENDATIONS_READ = _permission(
    ResourceType.BILLING,
    Action.READ,
    "View Cost Recommendations",
    "View cost optimization recommendations",
)
PERM_BILLING_RECOMMENDATIONS_APPLY = _permission(
    ResourceType.BILLING,
    Action.UPDATE,
    "Apply Cost Recommendations",
    "Apply and dismiss cost optimization recommendations",
)
PERM_BILLING_FORECAST_READ = _permission(
    ResourceType.BILLING,
    Action.READ,
    "View Cost Forecasts",
    "View cost forecasts and projections",
)
PERM_BILLING_FORECAST_SIMULATE = _permission(
    ResourceType.BILLING,
    Action.UPDATE,
    "Simulate Cost Scenarios",
    "Run what-if cost simulations",
)
PERM_BILLING_EXPORT_HISTORY = _permission(
    ResourceType.BILLING,
    Action.EXPORT_HISTORY,
    "Export Billing History",
    "Export historical billing data",
)
PERM_BILLING_DELETE = _permission(
    ResourceType.BILLING, Action.DELETE, "Delete Billing", "Remove billing records"
)
PERM_BILLING_CANCEL = _permission(
    ResourceType.BILLING, Action.CANCEL, "Cancel Billing", "Cancel subscriptions"
)

# ============================================================================
# FINANCE PERMISSIONS
# ============================================================================

PERM_FINANCE_READ = _permission(
    ResourceType.FINANCE,
    Action.READ,
    "View Finance",
    "View financial data, invoices, and transactions",
)
PERM_FINANCE_WRITE = _permission(
    ResourceType.FINANCE,
    Action.WRITE,
    "Manage Finance",
    "Create and modify financial records",
)
PERM_FINANCE_APPROVE = _permission(
    ResourceType.FINANCE,
    Action.APPROVE,
    "Approve Finance",
    "Approve financial transactions and invoices",
)

# ============================================================================
# COST PERMISSIONS
# ============================================================================

PERM_COST_READ = _permission(
    ResourceType.COST,
    Action.READ,
    "View Costs",
    "View cost dashboards and optimization recommendations",
)
PERM_COST_WRITE = _permission(
    ResourceType.COST,
    Action.WRITE,
    "Manage Costs",
    "Modify cost allocations and budgets",
)
PERM_COST_CENTER_READ = _permission(
    ResourceType.COST_CENTER, Action.READ, "View Cost Centers", "View cost center assignments"
)
PERM_COST_CENTER_UPDATE = _permission(
    ResourceType.COST_CENTER,
    Action.CHARGEBACK,
    "Manage Cost Centers",
    "Link resources to cost centers for chargeback",
)

# ============================================================================
# QUOTA & BUDGET PERMISSIONS
# ============================================================================

PERM_QUOTA_READ = _permission(
    ResourceType.QUOTA, Action.READ, "View Quotas", "View rate limits and quotas"
)
PERM_QUOTA_UPDATE = _permission(
    ResourceType.QUOTA, Action.SET_LIMIT, "Set Quotas", "Configure rate limits per user/org"
)
PERM_QUOTA_OVERRIDE = _permission(
    ResourceType.QUOTA,
    Action.OVERRIDE,
    "Override Quotas",
    "Emergency quota overrides",
)
PERM_BUDGET_READ = _permission(
    ResourceType.BUDGET, Action.READ, "View Budgets", "View budget limits and usage"
)
PERM_BUDGET_UPDATE = _permission(
    ResourceType.BUDGET, Action.SET_LIMIT, "Set Budgets", "Configure spending limits and alerts"
)
PERM_BUDGET_OVERRIDE = _permission(
    ResourceType.BUDGET,
    Action.OVERRIDE,
    "Override Budget",
    "Emergency budget increases",
)

# ============================================================================
# PAYMENTS PERMISSIONS
# ============================================================================

PERM_PAYMENTS_READ = _permission(
    ResourceType.PAYMENTS, Action.READ, "View Payments", "View payment transactions"
)
PERM_PAYMENTS_CHARGE = _permission(
    ResourceType.PAYMENTS, Action.CHARGE, "Charge Payments", "Process payment charges"
)
PERM_PAYMENTS_AUTHORIZE = _permission(
    ResourceType.PAYMENTS, Action.AUTHORIZE, "Authorize Payments", "Authorize payment transactions"
)
PERM_PAYMENTS_CAPTURE = _permission(
    ResourceType.PAYMENTS, Action.CAPTURE, "Capture Payments", "Capture authorized payments"
)
PERM_PAYMENTS_REFUND = _permission(
    ResourceType.PAYMENTS, Action.REFUND, "Refund Payments", "Process payment refunds"
)
PERM_PAYMENTS_VOID = _permission(
    ResourceType.PAYMENTS, Action.VOID, "Void Payments", "Void pending transactions"
)
PERM_PAYMENTS_CUSTOMER_CREATE = _permission(
    ResourceType.PAYMENTS,
    Action.CREATE,
    "Create Payment Customers",
    "Create payment customer profiles",
)
PERM_PAYMENTS_CUSTOMER_READ = _permission(
    ResourceType.PAYMENTS, Action.READ, "View Payment Customers", "View payment customer profiles"
)
PERM_PAYMENTS_SUBSCRIPTION_CREATE = _permission(
    ResourceType.PAYMENTS, Action.CREATE, "Create Subscriptions", "Create payment subscriptions"
)

# ============================================================================
# ACCOUNTS PAYABLE PERMISSIONS
# ============================================================================

PERM_AP_READ = _permission(ResourceType.AP, Action.READ, "View AP", "View accounts payable data")

# ============================================================================
# EXPENSES PERMISSIONS
# ============================================================================

PERM_EXPENSES_READ = _permission(
    ResourceType.EXPENSES, Action.READ, "View Expenses", "View expense records"
)
PERM_EXPENSES_WRITE = _permission(
    ResourceType.EXPENSES, Action.WRITE, "Manage Expenses", "Create and modify expenses"
)
PERM_EXPENSES_APPROVE = _permission(
    ResourceType.EXPENSES, Action.APPROVE, "Approve Expenses", "Approve expense reports"
)

# ============================================================================
# RECEIPT PERMISSIONS
# ============================================================================

PERM_RECEIPT_READ = _permission(
    ResourceType.RECEIPT,
    Action.READ,
    "View Receipts",
    "View decision receipts and audit trails",
)
PERM_RECEIPT_VERIFY = _permission(
    ResourceType.RECEIPT,
    Action.VERIFY,
    "Verify Receipts",
    "Verify integrity of decision receipts",
)
PERM_RECEIPT_EXPORT = _permission(
    ResourceType.RECEIPT,
    Action.EXPORT_DATA,
    "Export Receipts",
    "Export receipts for compliance reporting",
)
PERM_RECEIPT_SEND = _permission(
    ResourceType.RECEIPT,
    Action.SEND,
    "Send Receipts",
    "Send receipts to stakeholders",
)
PERM_RECEIPT_SHARE = _permission(
    ResourceType.RECEIPT, Action.SHARE, "Share Receipts", "Share receipts with stakeholders"
)
PERM_RECEIPT_SIGN = _permission(
    ResourceType.RECEIPT, Action.SIGN, "Sign Receipts", "Cryptographically sign receipts"
)

# ============================================================================
# RECONCILIATION PERMISSIONS
# ============================================================================

PERM_RECONCILIATION_READ = _permission(
    ResourceType.RECONCILIATION, Action.READ, "View Reconciliation", "View reconciliation data"
)

# All billing-related permission exports
__all__ = [
    # Billing
    "PERM_BILLING_READ",
    "PERM_BILLING_RECOMMENDATIONS_READ",
    "PERM_BILLING_RECOMMENDATIONS_APPLY",
    "PERM_BILLING_FORECAST_READ",
    "PERM_BILLING_FORECAST_SIMULATE",
    "PERM_BILLING_EXPORT_HISTORY",
    "PERM_BILLING_DELETE",
    "PERM_BILLING_CANCEL",
    # Finance
    "PERM_FINANCE_READ",
    "PERM_FINANCE_WRITE",
    "PERM_FINANCE_APPROVE",
    # Cost
    "PERM_COST_READ",
    "PERM_COST_WRITE",
    "PERM_COST_CENTER_READ",
    "PERM_COST_CENTER_UPDATE",
    # Quota & Budget
    "PERM_QUOTA_READ",
    "PERM_QUOTA_UPDATE",
    "PERM_QUOTA_OVERRIDE",
    "PERM_BUDGET_READ",
    "PERM_BUDGET_UPDATE",
    "PERM_BUDGET_OVERRIDE",
    # Payments
    "PERM_PAYMENTS_READ",
    "PERM_PAYMENTS_CHARGE",
    "PERM_PAYMENTS_AUTHORIZE",
    "PERM_PAYMENTS_CAPTURE",
    "PERM_PAYMENTS_REFUND",
    "PERM_PAYMENTS_VOID",
    "PERM_PAYMENTS_CUSTOMER_CREATE",
    "PERM_PAYMENTS_CUSTOMER_READ",
    "PERM_PAYMENTS_SUBSCRIPTION_CREATE",
    # Accounts Payable
    "PERM_AP_READ",
    # Expenses
    "PERM_EXPENSES_READ",
    "PERM_EXPENSES_WRITE",
    "PERM_EXPENSES_APPROVE",
    # Receipt
    "PERM_RECEIPT_READ",
    "PERM_RECEIPT_VERIFY",
    "PERM_RECEIPT_EXPORT",
    "PERM_RECEIPT_SEND",
    "PERM_RECEIPT_SHARE",
    "PERM_RECEIPT_SIGN",
    # Reconciliation
    "PERM_RECONCILIATION_READ",
]
