"""
Gusto Payroll Connector.

Provides payroll integration via Gusto API:
- OAuth 2.0 authentication
- Employee data sync
- Payroll run retrieval
- Journal entry generation for QBO
- Contractor payments

Dependencies:
    pip install aiohttp

Environment Variables:
    GUSTO_CLIENT_ID - Gusto OAuth client ID
    GUSTO_CLIENT_SECRET - Gusto OAuth client secret
    GUSTO_REDIRECT_URI - OAuth callback URL
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PayrollStatus(str, Enum):
    """Payroll run status."""

    UNPROCESSED = "unprocessed"
    PROCESSED = "processed"
    CANCELLED = "cancelled"


class EmploymentType(str, Enum):
    """Employee type."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACTOR = "contractor"


class PayType(str, Enum):
    """Compensation type."""

    HOURLY = "hourly"
    SALARY = "salary"


@dataclass
class GustoCredentials:
    """OAuth credentials for Gusto."""

    access_token: str
    refresh_token: str
    company_id: str
    company_name: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scope: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.expires_at:
            return True
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class Employee:
    """A Gusto employee."""

    id: str
    first_name: str
    last_name: str
    email: str
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    pay_type: PayType = PayType.SALARY
    department: Optional[str] = None
    job_title: Optional[str] = None
    hire_date: Optional[date] = None
    termination_date: Optional[date] = None
    is_active: bool = True

    # Compensation
    hourly_rate: Optional[Decimal] = None
    annual_salary: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": f"{self.first_name} {self.last_name}",
            "email": self.email,
            "employment_type": self.employment_type.value,
            "pay_type": self.pay_type.value,
            "department": self.department,
            "job_title": self.job_title,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "is_active": self.is_active,
        }


@dataclass
class PayrollItem:
    """An individual employee's payroll entry."""

    employee_id: str
    employee_name: str

    # Earnings
    gross_pay: Decimal = Decimal("0")
    regular_hours: Decimal = Decimal("0")
    overtime_hours: Decimal = Decimal("0")
    bonus: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    reimbursements: Decimal = Decimal("0")

    # Deductions
    federal_tax: Decimal = Decimal("0")
    state_tax: Decimal = Decimal("0")
    local_tax: Decimal = Decimal("0")
    social_security: Decimal = Decimal("0")
    medicare: Decimal = Decimal("0")
    retirement_401k: Decimal = Decimal("0")
    health_insurance: Decimal = Decimal("0")
    other_deductions: Decimal = Decimal("0")

    # Employer contributions
    employer_ss: Decimal = Decimal("0")
    employer_medicare: Decimal = Decimal("0")
    employer_futa: Decimal = Decimal("0")
    employer_suta: Decimal = Decimal("0")
    employer_401k_match: Decimal = Decimal("0")
    employer_health: Decimal = Decimal("0")

    @property
    def net_pay(self) -> Decimal:
        """Calculate net pay."""
        total_deductions = (
            self.federal_tax
            + self.state_tax
            + self.local_tax
            + self.social_security
            + self.medicare
            + self.retirement_401k
            + self.health_insurance
            + self.other_deductions
        )
        return self.gross_pay - total_deductions

    @property
    def total_employer_cost(self) -> Decimal:
        """Calculate total employer cost."""
        return (
            self.gross_pay
            + self.employer_ss
            + self.employer_medicare
            + self.employer_futa
            + self.employer_suta
            + self.employer_401k_match
            + self.employer_health
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "gross_pay": float(self.gross_pay),
            "net_pay": float(self.net_pay),
            "federal_tax": float(self.federal_tax),
            "state_tax": float(self.state_tax),
            "social_security": float(self.social_security),
            "medicare": float(self.medicare),
            "retirement_401k": float(self.retirement_401k),
            "health_insurance": float(self.health_insurance),
            "total_employer_cost": float(self.total_employer_cost),
        }


@dataclass
class PayrollRun:
    """A payroll run (pay period)."""

    id: str
    company_id: str
    pay_period_start: date
    pay_period_end: date
    check_date: date
    status: PayrollStatus = PayrollStatus.PROCESSED

    # Totals
    total_gross_pay: Decimal = Decimal("0")
    total_net_pay: Decimal = Decimal("0")
    total_employer_taxes: Decimal = Decimal("0")
    total_employee_taxes: Decimal = Decimal("0")
    total_deductions: Decimal = Decimal("0")
    total_reimbursements: Decimal = Decimal("0")

    # Individual entries
    payroll_items: List[PayrollItem] = field(default_factory=list)

    # Metadata
    processed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_employer_cost(self) -> Decimal:
        """Total cost to employer."""
        return self.total_gross_pay + self.total_employer_taxes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "pay_period_start": self.pay_period_start.isoformat(),
            "pay_period_end": self.pay_period_end.isoformat(),
            "check_date": self.check_date.isoformat(),
            "status": self.status.value,
            "total_gross_pay": float(self.total_gross_pay),
            "total_net_pay": float(self.total_net_pay),
            "total_employer_taxes": float(self.total_employer_taxes),
            "total_employee_taxes": float(self.total_employee_taxes),
            "total_employer_cost": float(self.total_employer_cost),
            "employee_count": len(self.payroll_items),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


@dataclass
class JournalEntry:
    """A journal entry for QBO."""

    date: date
    memo: str
    lines: List[Dict[str, Any]] = field(default_factory=list)

    def add_debit(
        self,
        account_id: str,
        account_name: str,
        amount: Decimal,
        description: str = "",
    ) -> None:
        """Add a debit line."""
        self.lines.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "debit": float(amount),
                "credit": 0,
                "description": description,
            }
        )

    def add_credit(
        self,
        account_id: str,
        account_name: str,
        amount: Decimal,
        description: str = "",
    ) -> None:
        """Add a credit line."""
        self.lines.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "debit": 0,
                "credit": float(amount),
                "description": description,
            }
        )

    @property
    def is_balanced(self) -> bool:
        """Check if debits equal credits."""
        total_debits = sum(line["debit"] for line in self.lines)
        total_credits = sum(line["credit"] for line in self.lines)
        return abs(total_debits - total_credits) < 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "memo": self.memo,
            "lines": self.lines,
            "is_balanced": self.is_balanced,
            "total_debits": sum(line["debit"] for line in self.lines),
            "total_credits": sum(line["credit"] for line in self.lines),
        }


class GustoConnector:
    """
    Gusto payroll integration connector.

    Handles OAuth authentication and payroll operations.
    """

    BASE_URL = "https://api.gusto.com"
    AUTH_URL = "https://api.gusto.com/oauth/authorize"
    TOKEN_URL = "https://api.gusto.com/oauth/token"

    # Default QBO account mappings
    DEFAULT_ACCOUNTS = {
        "gross_wages": ("6700", "Payroll Expense - Wages"),
        "employer_payroll_taxes": ("6710", "Payroll Expense - Taxes"),
        "employer_benefits": ("6720", "Payroll Expense - Benefits"),
        "payroll_liabilities": ("2100", "Payroll Liabilities"),
        "cash": ("1000", "Checking Account"),
        "federal_tax_payable": ("2110", "Federal Tax Payable"),
        "state_tax_payable": ("2120", "State Tax Payable"),
        "401k_payable": ("2130", "401k Payable"),
        "health_insurance_payable": ("2140", "Health Insurance Payable"),
    }

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        """
        Initialize Gusto connector.

        Args:
            client_id: OAuth client ID (or from GUSTO_CLIENT_ID env var)
            client_secret: OAuth client secret (or from GUSTO_CLIENT_SECRET env var)
            redirect_uri: OAuth callback URL (or from GUSTO_REDIRECT_URI env var)
        """
        self.client_id = client_id or os.getenv("GUSTO_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GUSTO_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("GUSTO_REDIRECT_URI")

        self._credentials: Optional[GustoCredentials] = None
        self._account_mappings = dict(self.DEFAULT_ACCOUNTS)

    @property
    def is_configured(self) -> bool:
        """Check if connector is configured."""
        return bool(self.client_id and self.client_secret)

    @property
    def is_authenticated(self) -> bool:
        """Check if connector has valid credentials."""
        return self._credentials is not None and not self._credentials.is_expired

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Get OAuth authorization URL."""
        import urllib.parse

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
        }
        if state:
            params["state"] = state

        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(
        self,
        authorization_code: str,
    ) -> GustoCredentials:
        """Exchange authorization code for tokens."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": authorization_code,
                    "grant_type": "authorization_code",
                    "redirect_uri": self.redirect_uri,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token exchange failed: {error_text}")

                data = await response.json()

                # Get company info
                company_info = await self._get_current_company(data["access_token"])

                self._credentials = GustoCredentials(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    company_id=company_info["id"],
                    company_name=company_info["name"],
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=data.get("expires_in", 7200)),
                    scope=data.get("scope", ""),
                )

                return self._credentials

    async def refresh_tokens(self) -> GustoCredentials:
        """Refresh OAuth tokens."""
        if not self._credentials:
            raise Exception("No credentials to refresh")

        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self._credentials.refresh_token,
                    "grant_type": "refresh_token",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token refresh failed: {error_text}")

                data = await response.json()

                self._credentials = GustoCredentials(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    company_id=self._credentials.company_id,
                    company_name=self._credentials.company_name,
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=data.get("expires_in", 7200)),
                )

                return self._credentials

    def set_credentials(self, credentials: GustoCredentials) -> None:
        """Set credentials (e.g., from storage)."""
        self._credentials = credentials

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
    ) -> Any:
        """Make authenticated API request."""
        token = access_token
        if not token:
            if not self._credentials:
                raise Exception("Not authenticated")
            if self._credentials.is_expired:
                await self.refresh_tokens()
            token = self._credentials.access_token

        import aiohttp

        url = f"{self.BASE_URL}{endpoint}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=headers,
                json=data,
            ) as response:
                response_data = await response.json()

                if response.status >= 400:
                    error = response_data.get("error", "Unknown error")
                    raise Exception(f"Gusto API error: {error}")

                return response_data

    async def _get_current_company(self, access_token: str) -> Dict[str, Any]:
        """Get current user's company."""
        response = await self._request(
            "GET",
            "/v1/me",
            access_token=access_token,
        )

        # Return first company
        companies = response.get("companies", [])
        if companies:
            return {"id": companies[0]["id"], "name": companies[0]["name"]}
        return {"id": "unknown", "name": "Unknown Company"}

    # =========================================================================
    # Employee Operations
    # =========================================================================

    async def list_employees(
        self,
        active_only: bool = True,
    ) -> List[Employee]:
        """List all employees."""
        if not self._credentials:
            raise Exception("Not authenticated")

        response = await self._request(
            "GET",
            f"/v1/companies/{self._credentials.company_id}/employees",
        )

        employees = []
        for item in response:
            if active_only and item.get("terminated", False):
                continue

            # Determine employment type
            emp_type = EmploymentType.FULL_TIME
            if item.get("employment_status") == "part_time":
                emp_type = EmploymentType.PART_TIME

            # Determine pay type
            pay_type = PayType.SALARY
            if item.get("flsa_status") == "Nonexempt":
                pay_type = PayType.HOURLY

            employees.append(
                Employee(
                    id=str(item["id"]),
                    first_name=item.get("first_name", ""),
                    last_name=item.get("last_name", ""),
                    email=item.get("email", ""),
                    employment_type=emp_type,
                    pay_type=pay_type,
                    department=item.get("department"),
                    job_title=item.get("jobs", [{}])[0].get("title") if item.get("jobs") else None,
                    hire_date=date.fromisoformat(item["hire_date"])
                    if item.get("hire_date")
                    else None,
                    is_active=not item.get("terminated", False),
                )
            )

        return employees

    # =========================================================================
    # Payroll Operations
    # =========================================================================

    async def list_payrolls(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        processed_only: bool = True,
    ) -> List[PayrollRun]:
        """List payroll runs."""
        if not self._credentials:
            raise Exception("Not authenticated")

        # Build query params
        params = []
        if start_date:
            params.append(f"start_date={start_date.isoformat()}")
        if end_date:
            params.append(f"end_date={end_date.isoformat()}")
        if processed_only:
            params.append("processed=true")

        query = "&".join(params)
        endpoint = f"/v1/companies/{self._credentials.company_id}/payrolls"
        if query:
            endpoint += f"?{query}"

        response = await self._request("GET", endpoint)

        payrolls = []
        for item in response:
            status = PayrollStatus.PROCESSED if item.get("processed") else PayrollStatus.UNPROCESSED
            if item.get("cancelled"):
                status = PayrollStatus.CANCELLED

            payrolls.append(
                PayrollRun(
                    id=str(item["id"]),
                    company_id=self._credentials.company_id,
                    pay_period_start=date.fromisoformat(item["pay_period"]["start_date"]),
                    pay_period_end=date.fromisoformat(item["pay_period"]["end_date"]),
                    check_date=date.fromisoformat(item["check_date"]),
                    status=status,
                    total_gross_pay=Decimal(str(item.get("totals", {}).get("gross_pay", 0))),
                    total_net_pay=Decimal(str(item.get("totals", {}).get("net_pay", 0))),
                    total_employer_taxes=Decimal(
                        str(item.get("totals", {}).get("employer_taxes", 0))
                    ),
                    total_employee_taxes=Decimal(
                        str(item.get("totals", {}).get("employee_taxes", 0))
                    ),
                    processed_at=datetime.fromisoformat(item["processed_date"])
                    if item.get("processed_date")
                    else None,
                )
            )

        return payrolls

    async def get_payroll(self, payroll_id: str) -> Optional[PayrollRun]:
        """Get detailed payroll run with employee items."""
        if not self._credentials:
            raise Exception("Not authenticated")

        response = await self._request(
            "GET",
            f"/v1/companies/{self._credentials.company_id}/payrolls/{payroll_id}",
        )

        if not response:
            return None

        status = PayrollStatus.PROCESSED if response.get("processed") else PayrollStatus.UNPROCESSED

        payroll = PayrollRun(
            id=str(response["id"]),
            company_id=self._credentials.company_id,
            pay_period_start=date.fromisoformat(response["pay_period"]["start_date"]),
            pay_period_end=date.fromisoformat(response["pay_period"]["end_date"]),
            check_date=date.fromisoformat(response["check_date"]),
            status=status,
            total_gross_pay=Decimal(str(response.get("totals", {}).get("gross_pay", 0))),
            total_net_pay=Decimal(str(response.get("totals", {}).get("net_pay", 0))),
            total_employer_taxes=Decimal(str(response.get("totals", {}).get("employer_taxes", 0))),
            total_employee_taxes=Decimal(str(response.get("totals", {}).get("employee_taxes", 0))),
        )

        # Parse employee compensations
        for emp_comp in response.get("employee_compensations", []):
            employee = emp_comp.get("employee", {})
            fixed_comps = emp_comp.get("fixed_compensations", [])
            hourly_comps = emp_comp.get("hourly_compensations", [])
            taxes = emp_comp.get("taxes", [])
            benefits = emp_comp.get("benefits", [])

            # Calculate earnings
            gross = sum(Decimal(str(c.get("amount", 0))) for c in fixed_comps)
            gross += sum(
                Decimal(str(c.get("hours", 0))) * Decimal(str(c.get("compensation_multiplier", 1)))
                for c in hourly_comps
            )

            item = PayrollItem(
                employee_id=str(employee.get("id", "")),
                employee_name=f"{employee.get('first_name', '')} {employee.get('last_name', '')}",
                gross_pay=gross,
            )

            # Parse taxes
            for tax in taxes:
                tax_name = tax.get("name", "").lower()
                amount = Decimal(str(tax.get("amount", 0)))

                if "federal" in tax_name:
                    if tax.get("employer"):
                        item.employer_futa = amount
                    else:
                        item.federal_tax = amount
                elif "state" in tax_name:
                    if tax.get("employer"):
                        item.employer_suta = amount
                    else:
                        item.state_tax = amount
                elif "social security" in tax_name:
                    if tax.get("employer"):
                        item.employer_ss = amount
                    else:
                        item.social_security = amount
                elif "medicare" in tax_name:
                    if tax.get("employer"):
                        item.employer_medicare = amount
                    else:
                        item.medicare = amount

            # Parse benefits
            for benefit in benefits:
                benefit_name = benefit.get("name", "").lower()
                emp_amount = Decimal(str(benefit.get("employee_deduction", 0)))
                er_amount = Decimal(str(benefit.get("company_contribution", 0)))

                if "401k" in benefit_name or "retirement" in benefit_name:
                    item.retirement_401k = emp_amount
                    item.employer_401k_match = er_amount
                elif "health" in benefit_name or "medical" in benefit_name:
                    item.health_insurance = emp_amount
                    item.employer_health = er_amount

            payroll.payroll_items.append(item)

        return payroll

    # =========================================================================
    # Journal Entry Generation
    # =========================================================================

    def generate_journal_entry(
        self,
        payroll: PayrollRun,
        account_mappings: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> JournalEntry:
        """
        Generate QBO journal entry from payroll run.

        Standard payroll journal entry:
        - Debit: Payroll Expense (gross wages)
        - Debit: Payroll Tax Expense (employer taxes)
        - Debit: Benefits Expense (employer benefits)
        - Credit: Cash (net pay)
        - Credit: Federal Tax Payable
        - Credit: State Tax Payable
        - Credit: 401k Payable
        - Credit: Health Insurance Payable

        Args:
            payroll: PayrollRun to convert
            account_mappings: Optional custom account mappings

        Returns:
            JournalEntry ready for QBO
        """
        mappings = account_mappings or self._account_mappings

        entry = JournalEntry(
            date=payroll.check_date,
            memo=f"Payroll {payroll.pay_period_start} to {payroll.pay_period_end}",
        )

        # Aggregate totals from items
        total_gross = Decimal("0")
        total_federal = Decimal("0")
        total_state = Decimal("0")
        total_401k = Decimal("0")
        total_health = Decimal("0")
        total_ss = Decimal("0")
        total_medicare = Decimal("0")
        total_employer_taxes = Decimal("0")
        total_employer_benefits = Decimal("0")

        for item in payroll.payroll_items:
            total_gross += item.gross_pay
            total_federal += item.federal_tax
            total_state += item.state_tax
            total_401k += item.retirement_401k
            total_health += item.health_insurance
            total_ss += item.social_security
            total_medicare += item.medicare

            total_employer_taxes += (
                item.employer_ss + item.employer_medicare + item.employer_futa + item.employer_suta
            )
            total_employer_benefits += item.employer_401k_match + item.employer_health

        # Debits
        if total_gross > 0:
            acct = mappings.get("gross_wages", self.DEFAULT_ACCOUNTS["gross_wages"])
            entry.add_debit(acct[0], acct[1], total_gross, "Gross wages")

        if total_employer_taxes > 0:
            acct = mappings.get(
                "employer_payroll_taxes", self.DEFAULT_ACCOUNTS["employer_payroll_taxes"]
            )
            entry.add_debit(acct[0], acct[1], total_employer_taxes, "Employer payroll taxes")

        if total_employer_benefits > 0:
            acct = mappings.get("employer_benefits", self.DEFAULT_ACCOUNTS["employer_benefits"])
            entry.add_debit(acct[0], acct[1], total_employer_benefits, "Employer benefits")

        # Credits
        if payroll.total_net_pay > 0:
            acct = mappings.get("cash", self.DEFAULT_ACCOUNTS["cash"])
            entry.add_credit(acct[0], acct[1], payroll.total_net_pay, "Net pay - direct deposit")

        if total_federal > 0:
            acct = mappings.get("federal_tax_payable", self.DEFAULT_ACCOUNTS["federal_tax_payable"])
            entry.add_credit(acct[0], acct[1], total_federal, "Federal tax withholding")

        if total_state > 0:
            acct = mappings.get("state_tax_payable", self.DEFAULT_ACCOUNTS["state_tax_payable"])
            entry.add_credit(acct[0], acct[1], total_state, "State tax withholding")

        if total_ss + total_medicare > 0:
            acct = mappings.get("federal_tax_payable", self.DEFAULT_ACCOUNTS["federal_tax_payable"])
            entry.add_credit(acct[0], acct[1], total_ss + total_medicare, "FICA withholding")

        if total_401k > 0:
            acct = mappings.get("401k_payable", self.DEFAULT_ACCOUNTS["401k_payable"])
            entry.add_credit(acct[0], acct[1], total_401k, "401k contributions")

        if total_health > 0:
            acct = mappings.get(
                "health_insurance_payable", self.DEFAULT_ACCOUNTS["health_insurance_payable"]
            )
            entry.add_credit(acct[0], acct[1], total_health, "Health insurance deductions")

        # Employer tax/benefit liabilities
        if total_employer_taxes > 0:
            acct = mappings.get("federal_tax_payable", self.DEFAULT_ACCOUNTS["federal_tax_payable"])
            entry.add_credit(
                acct[0], acct[1], total_employer_taxes, "Employer payroll taxes payable"
            )

        if total_employer_benefits > 0:
            acct = mappings.get(
                "health_insurance_payable", self.DEFAULT_ACCOUNTS["health_insurance_payable"]
            )
            entry.add_credit(acct[0], acct[1], total_employer_benefits, "Employer benefits payable")

        return entry

    def set_account_mapping(
        self,
        mapping_key: str,
        account_id: str,
        account_name: str,
    ) -> None:
        """Set a custom account mapping."""
        self._account_mappings[mapping_key] = (account_id, account_name)


# =============================================================================
# Mock Data for Demo
# =============================================================================


def get_mock_employees() -> List[Employee]:
    """Generate mock employee data."""
    return [
        Employee(
            id="emp_001",
            first_name="Alice",
            last_name="Johnson",
            email="alice@company.com",
            employment_type=EmploymentType.FULL_TIME,
            pay_type=PayType.SALARY,
            department="Engineering",
            job_title="Senior Software Engineer",
            hire_date=date(2022, 3, 15),
            annual_salary=Decimal("150000"),
        ),
        Employee(
            id="emp_002",
            first_name="Bob",
            last_name="Smith",
            email="bob@company.com",
            employment_type=EmploymentType.FULL_TIME,
            pay_type=PayType.SALARY,
            department="Engineering",
            job_title="Software Engineer",
            hire_date=date(2023, 1, 10),
            annual_salary=Decimal("120000"),
        ),
        Employee(
            id="emp_003",
            first_name="Carol",
            last_name="Davis",
            email="carol@company.com",
            employment_type=EmploymentType.PART_TIME,
            pay_type=PayType.HOURLY,
            department="Marketing",
            job_title="Marketing Coordinator",
            hire_date=date(2023, 6, 1),
            hourly_rate=Decimal("35"),
        ),
    ]


def get_mock_payroll_run() -> PayrollRun:
    """Generate mock payroll run."""
    today = date.today()
    period_end = today - timedelta(days=today.weekday() + 3)  # Last Friday
    period_start = period_end - timedelta(days=13)  # Two weeks

    return PayrollRun(
        id="payroll_demo_001",
        company_id="company_001",
        pay_period_start=period_start,
        pay_period_end=period_end,
        check_date=period_end + timedelta(days=5),
        status=PayrollStatus.PROCESSED,
        total_gross_pay=Decimal("22500.00"),
        total_net_pay=Decimal("16875.00"),
        total_employer_taxes=Decimal("1721.25"),
        total_employee_taxes=Decimal("4500.00"),
        payroll_items=[
            PayrollItem(
                employee_id="emp_001",
                employee_name="Alice Johnson",
                gross_pay=Decimal("5769.23"),
                federal_tax=Decimal("1153.85"),
                state_tax=Decimal("288.46"),
                social_security=Decimal("357.69"),
                medicare=Decimal("83.65"),
                retirement_401k=Decimal("288.46"),
                health_insurance=Decimal("250.00"),
                employer_ss=Decimal("357.69"),
                employer_medicare=Decimal("83.65"),
                employer_401k_match=Decimal("144.23"),
            ),
            PayrollItem(
                employee_id="emp_002",
                employee_name="Bob Smith",
                gross_pay=Decimal("4615.38"),
                federal_tax=Decimal("923.08"),
                state_tax=Decimal("230.77"),
                social_security=Decimal("286.15"),
                medicare=Decimal("66.92"),
                retirement_401k=Decimal("230.77"),
                health_insurance=Decimal("250.00"),
                employer_ss=Decimal("286.15"),
                employer_medicare=Decimal("66.92"),
                employer_401k_match=Decimal("115.38"),
            ),
        ],
        processed_at=datetime.now(timezone.utc) - timedelta(days=2),
    )


__all__ = [
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
