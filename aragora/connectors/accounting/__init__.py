"""
Accounting Connectors.

Integrations for accounting and financial services:
- QuickBooks Online (QBO)
- Plaid bank connectivity
- Gusto payroll
"""

from aragora.connectors.accounting.qbo import (
    QuickBooksConnector,
    QBOCredentials,
    QBOEnvironment,
    QBOCustomer,
    QBOTransaction,
    QBOAccount,
    TransactionType,
    get_mock_customers,
    get_mock_transactions,
)
from aragora.connectors.accounting.plaid import (
    PlaidConnector,
    PlaidCredentials,
    PlaidEnvironment,
    PlaidError,
    BankAccount,
    BankTransaction,
    AccountType,
    TransactionCategory,
    CategoryMapping,
    get_mock_accounts,
    get_mock_transactions as get_mock_bank_transactions,
)
from aragora.connectors.accounting.gusto import (
    GustoConnector,
    GustoCredentials,
    Employee,
    EmploymentType,
    PayType,
    PayrollRun,
    PayrollItem,
    PayrollStatus,
    JournalEntry,
    get_mock_employees,
    get_mock_payroll_run,
)

__all__ = [
    # QBO
    "QuickBooksConnector",
    "QBOCredentials",
    "QBOEnvironment",
    "QBOCustomer",
    "QBOTransaction",
    "QBOAccount",
    "TransactionType",
    "get_mock_customers",
    "get_mock_transactions",
    # Plaid
    "PlaidConnector",
    "PlaidCredentials",
    "PlaidEnvironment",
    "PlaidError",
    "BankAccount",
    "BankTransaction",
    "AccountType",
    "TransactionCategory",
    "CategoryMapping",
    "get_mock_accounts",
    "get_mock_bank_transactions",
    # Gusto
    "GustoConnector",
    "GustoCredentials",
    "Employee",
    "EmploymentType",
    "PayType",
    "PayrollRun",
    "PayrollItem",
    "PayrollStatus",
    "JournalEntry",
    "get_mock_employees",
    "get_mock_payroll_run",
]
