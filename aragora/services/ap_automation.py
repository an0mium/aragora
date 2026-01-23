"""
Accounts Payable Automation Service.

Automates accounts payable workflows:
- Payment timing optimization (cash flow vs discounts)
- Batch payment processing
- Cash needs forecasting
- Vendor payment scheduling
- Early payment discount capture

Usage:
    from aragora.services.ap_automation import APAutomation

    ap = APAutomation()

    # Optimize payment timing
    schedule = await ap.optimize_payment_timing(invoices)

    # Batch payments
    batch = await ap.batch_payments(invoices)

    # Forecast cash needs
    forecast = await ap.forecast_cash_needs(days_ahead=30)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid4

if TYPE_CHECKING:
    from aragora.connectors.accounting.qbo import QuickBooksConnector

logger = logging.getLogger(__name__)


class PaymentPriority(str, Enum):
    """Payment priority levels."""

    CRITICAL = "critical"  # Must pay now (utilities, payroll)
    HIGH = "high"  # Important vendors, discount expiring
    NORMAL = "normal"  # Standard terms
    LOW = "low"  # Can delay if needed
    HOLD = "hold"  # On hold (dispute, etc.)


class PaymentMethod(str, Enum):
    """Payment methods."""

    ACH = "ach"
    WIRE = "wire"
    CHECK = "check"
    CREDIT_CARD = "credit_card"
    VIRTUAL_CARD = "virtual_card"


@dataclass
class PayableInvoice:
    """An accounts payable invoice."""

    id: str
    vendor_id: str
    vendor_name: str
    invoice_number: str = ""
    invoice_date: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    total_amount: Decimal = Decimal("0.00")
    amount_paid: Decimal = Decimal("0.00")
    balance: Decimal = Decimal("0.00")
    payment_terms: str = "Net 30"
    early_pay_discount: float = 0.0  # e.g., 2% = 0.02
    discount_deadline: Optional[datetime] = None
    priority: PaymentPriority = PaymentPriority.NORMAL
    preferred_payment_method: PaymentMethod = PaymentMethod.ACH
    scheduled_pay_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    is_recurring: bool = False
    notes: str = ""

    @property
    def days_until_due(self) -> Optional[int]:
        """Days until due date."""
        if self.due_date:
            return (self.due_date - datetime.now()).days
        return None

    @property
    def days_until_discount(self) -> Optional[int]:
        """Days until early pay discount expires."""
        if self.discount_deadline:
            return (self.discount_deadline - datetime.now()).days
        return None

    @property
    def discount_amount(self) -> Decimal:
        """Calculate early payment discount amount."""
        if self.early_pay_discount > 0:
            return (self.balance * Decimal(str(self.early_pay_discount))).quantize(Decimal("0.01"))
        return Decimal("0")

    @property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        if self.due_date and self.balance > 0:
            return datetime.now() > self.due_date
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vendorId": self.vendor_id,
            "vendorName": self.vendor_name,
            "invoiceNumber": self.invoice_number,
            "invoiceDate": self.invoice_date.isoformat(),
            "dueDate": self.due_date.isoformat() if self.due_date else None,
            "totalAmount": float(self.total_amount),
            "amountPaid": float(self.amount_paid),
            "balance": float(self.balance),
            "paymentTerms": self.payment_terms,
            "earlyPayDiscount": self.early_pay_discount,
            "discountDeadline": self.discount_deadline.isoformat()
            if self.discount_deadline
            else None,
            "discountAmount": float(self.discount_amount),
            "priority": self.priority.value,
            "preferredPaymentMethod": self.preferred_payment_method.value,
            "scheduledPayDate": self.scheduled_pay_date.isoformat()
            if self.scheduled_pay_date
            else None,
            "daysUntilDue": self.days_until_due,
            "daysUntilDiscount": self.days_until_discount,
            "isOverdue": self.is_overdue,
        }


@dataclass
class PaymentSchedule:
    """Optimized payment schedule."""

    total_amount: Decimal = Decimal("0.00")
    total_discount_captured: Decimal = Decimal("0.00")
    payments: List[Dict[str, Any]] = field(default_factory=list)
    by_date: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    optimization_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "totalAmount": float(self.total_amount),
            "totalDiscountCaptured": float(self.total_discount_captured),
            "payments": self.payments,
            "byDate": {k: v for k, v in self.by_date.items()},
            "optimizationNotes": self.optimization_notes,
        }


@dataclass
class BatchPayment:
    """A batch of payments to process together."""

    id: str
    payment_date: datetime
    total_amount: Decimal = Decimal("0.00")
    payment_count: int = 0
    payment_method: PaymentMethod = PaymentMethod.ACH
    invoices: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "paymentDate": self.payment_date.isoformat(),
            "totalAmount": float(self.total_amount),
            "paymentCount": self.payment_count,
            "paymentMethod": self.payment_method.value,
            "invoices": self.invoices,
            "status": self.status,
            "createdAt": self.created_at.isoformat(),
        }


@dataclass
class CashForecast:
    """Cash flow forecast."""

    forecast_date: datetime = field(default_factory=datetime.now)
    days_ahead: int = 30
    current_balance: Decimal = Decimal("0.00")
    total_payables: Decimal = Decimal("0.00")
    total_receivables: Decimal = Decimal("0.00")
    projected_balance: Decimal = Decimal("0.00")
    daily_forecast: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecastDate": self.forecast_date.isoformat(),
            "daysAhead": self.days_ahead,
            "currentBalance": float(self.current_balance),
            "totalPayables": float(self.total_payables),
            "totalReceivables": float(self.total_receivables),
            "projectedBalance": float(self.projected_balance),
            "dailyForecast": self.daily_forecast,
            "warnings": self.warnings,
        }


class APAutomation:
    """
    Service for automating accounts payable workflows.

    Handles payment optimization, batching, and cash forecasting.
    Includes circuit breaker protection for external service calls.
    """

    def __init__(
        self,
        qbo_connector: Optional[QuickBooksConnector] = None,
        current_cash_balance: Decimal = Decimal("100000"),
        min_cash_reserve: Decimal = Decimal("20000"),
        enable_circuit_breakers: bool = True,
    ):
        """
        Initialize AP automation service.

        Args:
            qbo_connector: QuickBooks connector for sync
            current_cash_balance: Current cash balance
            min_cash_reserve: Minimum cash reserve to maintain
            enable_circuit_breakers: Enable circuit breaker protection
        """
        self.qbo = qbo_connector
        self.current_cash_balance = current_cash_balance
        self.min_cash_reserve = min_cash_reserve
        self._enable_circuit_breakers = enable_circuit_breakers

        # In-memory storage
        self._invoices: Dict[str, PayableInvoice] = {}
        self._batches: Dict[str, BatchPayment] = {}
        self._by_vendor: Dict[str, set] = {}
        self._expected_receivables: List[Tuple[datetime, Decimal]] = []

        # Circuit breakers for external service resilience
        self._circuit_breakers: Dict[str, Any] = {}
        if enable_circuit_breakers:
            from aragora.resilience import get_circuit_breaker

            self._circuit_breakers = {
                "qbo": get_circuit_breaker("ap_automation_qbo", 5, 120.0),
                "payment": get_circuit_breaker("ap_automation_payment", 3, 60.0),
            }

    def _check_circuit_breaker(self, service: str) -> bool:
        """Check if circuit breaker allows the request."""
        if service not in self._circuit_breakers:
            return True
        cb = self._circuit_breakers[service]
        if not cb.can_proceed():
            logger.warning(f"Circuit breaker open for {service}")
            return False
        return True

    def _record_cb_success(self, service: str) -> None:
        """Record successful external call."""
        if service in self._circuit_breakers:
            self._circuit_breakers[service].record_success()

    def _record_cb_failure(self, service: str) -> None:
        """Record failed external call."""
        if service in self._circuit_breakers:
            self._circuit_breakers[service].record_failure()

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            "enabled": self._enable_circuit_breakers,
            "services": {
                svc: {"status": cb.get_status(), "failures": cb.failure_count}
                for svc, cb in self._circuit_breakers.items()
            },
        }

    async def add_invoice(
        self,
        vendor_id: str,
        vendor_name: str,
        total_amount: float,
        invoice_number: str = "",
        invoice_date: Optional[datetime] = None,
        due_date: Optional[datetime] = None,
        payment_terms: str = "Net 30",
        early_pay_discount: float = 0.0,
        discount_days: int = 10,
        priority: PaymentPriority = PaymentPriority.NORMAL,
    ) -> PayableInvoice:
        """
        Add an invoice to the AP system.

        Args:
            vendor_id: Vendor ID
            vendor_name: Vendor name
            total_amount: Invoice amount
            invoice_number: Invoice reference
            invoice_date: Invoice date
            due_date: Due date
            payment_terms: Payment terms (e.g., "2/10 Net 30")
            early_pay_discount: Discount percentage (e.g., 0.02 for 2%)
            discount_days: Days to capture discount
            priority: Payment priority

        Returns:
            Created invoice
        """
        invoice_id = f"ap_{uuid4().hex[:12]}"
        inv_date = invoice_date or datetime.now()

        # Parse payment terms for due date
        if due_date is None:
            if "Net 30" in payment_terms:
                due_date = inv_date + timedelta(days=30)
            elif "Net 15" in payment_terms:
                due_date = inv_date + timedelta(days=15)
            elif "Net 60" in payment_terms:
                due_date = inv_date + timedelta(days=60)
            else:
                due_date = inv_date + timedelta(days=30)

        # Calculate discount deadline
        discount_deadline = None
        if early_pay_discount > 0:
            discount_deadline = inv_date + timedelta(days=discount_days)

        amount = Decimal(str(total_amount))

        invoice = PayableInvoice(
            id=invoice_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            invoice_number=invoice_number,
            invoice_date=inv_date,
            due_date=due_date,
            total_amount=amount,
            balance=amount,
            payment_terms=payment_terms,
            early_pay_discount=early_pay_discount,
            discount_deadline=discount_deadline,
            priority=priority,
        )

        self._store_invoice(invoice)
        return invoice

    def _store_invoice(self, invoice: PayableInvoice) -> None:
        """Store invoice and update indexes."""
        self._invoices[invoice.id] = invoice

        if invoice.vendor_id not in self._by_vendor:
            self._by_vendor[invoice.vendor_id] = set()
        self._by_vendor[invoice.vendor_id].add(invoice.id)

    async def optimize_payment_timing(
        self,
        invoices: Optional[List[PayableInvoice]] = None,
        available_cash: Optional[Decimal] = None,
    ) -> PaymentSchedule:
        """
        Optimize payment timing to balance cash flow and discounts.

        Strategy:
        1. Always pay critical/overdue first
        2. Capture early payment discounts if ROI > threshold
        3. Pay remaining invoices as late as possible while maintaining terms

        Args:
            invoices: Invoices to schedule (default: all unpaid)
            available_cash: Available cash (default: current balance)

        Returns:
            Optimized payment schedule
        """
        if invoices is None:
            invoices = [i for i in self._invoices.values() if i.balance > 0]

        cash = available_cash or self.current_cash_balance
        schedule = PaymentSchedule()

        # Sort by priority and timing
        # 1. Critical/overdue first
        # 2. Discount expiring soon
        # 3. By due date
        def payment_priority(inv: PayableInvoice) -> Tuple:
            is_critical = inv.priority == PaymentPriority.CRITICAL or inv.is_overdue
            discount_days = inv.days_until_discount if inv.days_until_discount else 999
            due_days = inv.days_until_due if inv.days_until_due else 999
            return (0 if is_critical else 1, discount_days, due_days)

        sorted_invoices = sorted(invoices, key=payment_priority)

        for invoice in sorted_invoices:
            payment_date = datetime.now()
            capture_discount = False
            discount_saved = Decimal("0")

            # Determine optimal payment date
            if invoice.priority == PaymentPriority.CRITICAL or invoice.is_overdue:
                # Pay immediately
                payment_date = datetime.now()
                schedule.optimization_notes.append(
                    f"Pay {invoice.vendor_name} immediately (critical/overdue)"
                )
            elif invoice.early_pay_discount > 0 and invoice.discount_deadline:
                # Check if discount is worth it
                # Simple ROI: discount % annualized vs holding cash
                days_early = (
                    (invoice.due_date - invoice.discount_deadline).days if invoice.due_date else 20
                )
                if days_early > 0:
                    annualized_return = (invoice.early_pay_discount / days_early) * 365
                    # If annualized return > 15%, capture discount
                    if annualized_return > 0.15:
                        payment_date = invoice.discount_deadline - timedelta(days=1)
                        capture_discount = True
                        discount_saved = invoice.discount_amount
                        schedule.optimization_notes.append(
                            f"Pay {invoice.vendor_name} early to capture {invoice.early_pay_discount*100:.1f}% discount (${discount_saved})"
                        )

            if not capture_discount and invoice.due_date:
                # Pay on due date (maximize cash holding)
                payment_date = invoice.due_date

            # Check cash availability
            net_amount = invoice.balance - discount_saved
            if cash - net_amount < self.min_cash_reserve:
                schedule.optimization_notes.append(
                    f"Warning: Paying {invoice.vendor_name} would breach cash reserve"
                )

            # Add to schedule
            payment_info = {
                "invoiceId": invoice.id,
                "vendorId": invoice.vendor_id,
                "vendorName": invoice.vendor_name,
                "invoiceNumber": invoice.invoice_number,
                "payDate": payment_date.isoformat(),
                "amount": float(net_amount),
                "originalAmount": float(invoice.balance),
                "discountCaptured": float(discount_saved),
            }

            schedule.payments.append(payment_info)
            schedule.total_amount += net_amount
            schedule.total_discount_captured += discount_saved

            # Group by date
            date_key = payment_date.strftime("%Y-%m-%d")
            if date_key not in schedule.by_date:
                schedule.by_date[date_key] = []
            schedule.by_date[date_key].append(payment_info)

        return schedule

    async def batch_payments(
        self,
        invoices: Optional[List[PayableInvoice]] = None,
        payment_date: Optional[datetime] = None,
        payment_method: PaymentMethod = PaymentMethod.ACH,
    ) -> BatchPayment:
        """
        Create a batch of payments for processing.

        Args:
            invoices: Invoices to batch (default: all approved)
            payment_date: Payment date
            payment_method: Payment method for batch

        Returns:
            Batch payment
        """
        if invoices is None:
            invoices = [
                i for i in self._invoices.values() if i.balance > 0 and i.scheduled_pay_date
            ]

        batch_id = f"batch_{uuid4().hex[:12]}"
        pay_date = payment_date or datetime.now()

        batch = BatchPayment(
            id=batch_id,
            payment_date=pay_date,
            payment_method=payment_method,
        )

        # Group by vendor for efficiency
        by_vendor: Dict[str, List[PayableInvoice]] = {}
        for invoice in invoices:
            if invoice.vendor_id not in by_vendor:
                by_vendor[invoice.vendor_id] = []
            by_vendor[invoice.vendor_id].append(invoice)

        for vendor_id, vendor_invoices in by_vendor.items():
            vendor_total = sum(i.balance for i in vendor_invoices)
            vendor_name = vendor_invoices[0].vendor_name

            batch.invoices.append(
                {
                    "vendorId": vendor_id,
                    "vendorName": vendor_name,
                    "invoiceCount": len(vendor_invoices),
                    "totalAmount": float(vendor_total),
                    "invoices": [
                        {
                            "id": i.id,
                            "invoiceNumber": i.invoice_number,
                            "amount": float(i.balance),
                        }
                        for i in vendor_invoices
                    ],
                }
            )

            batch.total_amount += vendor_total
            batch.payment_count += len(vendor_invoices)

        self._batches[batch_id] = batch
        return batch

    async def forecast_cash_needs(
        self,
        days_ahead: int = 30,
        include_receivables: bool = True,
    ) -> CashForecast:
        """
        Forecast cash needs for the specified period.

        Args:
            days_ahead: Days to forecast
            include_receivables: Include expected receivables

        Returns:
            Cash forecast
        """
        forecast = CashForecast(
            days_ahead=days_ahead,
            current_balance=self.current_cash_balance,
        )

        # Get payables due in period
        end_date = datetime.now() + timedelta(days=days_ahead)
        payables_due = [
            i
            for i in self._invoices.values()
            if i.balance > 0 and i.due_date and i.due_date <= end_date
        ]

        forecast.total_payables = sum(i.balance for i in payables_due)

        # Include expected receivables
        if include_receivables:
            for recv_date, amount in self._expected_receivables:
                if recv_date <= end_date:
                    forecast.total_receivables += amount

        # Calculate projected balance
        forecast.projected_balance = (
            self.current_cash_balance + forecast.total_receivables - forecast.total_payables
        )

        # Generate daily forecast
        daily_balance = self.current_cash_balance
        current_date = datetime.now()

        for day_offset in range(days_ahead + 1):
            check_date = current_date + timedelta(days=day_offset)
            date_str = check_date.strftime("%Y-%m-%d")

            # Payables due on this date
            day_payables = sum(
                i.balance
                for i in payables_due
                if i.due_date and i.due_date.date() == check_date.date()
            )

            # Receivables expected on this date
            day_receivables = sum(
                amount
                for recv_date, amount in self._expected_receivables
                if recv_date.date() == check_date.date()
            )

            daily_balance = daily_balance + day_receivables - day_payables

            forecast.daily_forecast.append(
                {
                    "date": date_str,
                    "payables": float(day_payables),
                    "receivables": float(day_receivables),
                    "balance": float(daily_balance),
                }
            )

            # Check for warnings
            if daily_balance < self.min_cash_reserve:
                forecast.warnings.append(
                    f"Cash below reserve on {date_str}: ${float(daily_balance):,.2f}"
                )
            if daily_balance < 0:
                forecast.warnings.append(
                    f"Negative cash on {date_str}: ${float(daily_balance):,.2f}"
                )

        return forecast

    async def add_expected_receivable(
        self,
        expected_date: datetime,
        amount: float,
    ) -> None:
        """Add an expected receivable for forecasting."""
        self._expected_receivables.append((expected_date, Decimal(str(amount))))

    async def get_invoice(self, invoice_id: str) -> Optional[PayableInvoice]:
        """Get invoice by ID."""
        return self._invoices.get(invoice_id)

    async def list_invoices(
        self,
        vendor_id: Optional[str] = None,
        priority: Optional[PaymentPriority] = None,
        overdue_only: bool = False,
        with_discount: bool = False,
    ) -> List[PayableInvoice]:
        """List invoices with filters."""
        invoices = list(self._invoices.values())

        # Filter to unpaid only
        invoices = [i for i in invoices if i.balance > 0]

        if vendor_id:
            invoices = [i for i in invoices if i.vendor_id == vendor_id]

        if priority:
            invoices = [i for i in invoices if i.priority == priority]

        if overdue_only:
            invoices = [i for i in invoices if i.is_overdue]

        if with_discount:
            invoices = [
                i
                for i in invoices
                if i.early_pay_discount > 0
                and i.discount_deadline
                and i.discount_deadline > datetime.now()
            ]

        invoices.sort(key=lambda x: x.due_date or datetime.max)
        return invoices

    async def get_discount_opportunities(self) -> List[Dict[str, Any]]:
        """Get invoices with available early payment discounts."""
        opportunities = []

        for invoice in self._invoices.values():
            if invoice.balance <= 0:
                continue
            if invoice.early_pay_discount <= 0:
                continue
            if not invoice.discount_deadline or invoice.discount_deadline <= datetime.now():
                continue

            days_left = invoice.days_until_discount or 0
            discount_amount = invoice.discount_amount
            days_early = (
                (invoice.due_date - invoice.discount_deadline).days if invoice.due_date else 20
            )

            # Calculate annualized return
            if days_early > 0:
                annualized_return = (invoice.early_pay_discount / days_early) * 365
            else:
                annualized_return = 0

            opportunities.append(
                {
                    "invoiceId": invoice.id,
                    "vendorName": invoice.vendor_name,
                    "invoiceNumber": invoice.invoice_number,
                    "amount": float(invoice.balance),
                    "discountPercent": invoice.early_pay_discount * 100,
                    "discountAmount": float(discount_amount),
                    "discountDeadline": invoice.discount_deadline.isoformat(),
                    "daysLeft": days_left,
                    "annualizedReturn": annualized_return * 100,
                    "recommended": annualized_return > 0.15,
                }
            )

        # Sort by annualized return
        opportunities.sort(key=lambda x: x["annualizedReturn"], reverse=True)
        return opportunities

    async def record_payment(
        self,
        invoice_id: str,
        amount: float,
        payment_date: Optional[datetime] = None,
    ) -> Optional[PayableInvoice]:
        """Record a payment against an invoice."""
        invoice = self._invoices.get(invoice_id)
        if not invoice:
            return None

        payment = Decimal(str(amount))
        invoice.amount_paid += payment
        invoice.balance = invoice.total_amount - invoice.amount_paid

        if invoice.balance <= 0:
            invoice.paid_at = payment_date or datetime.now()

        return invoice

    def get_stats(self) -> Dict[str, Any]:
        """Get AP statistics."""
        invoices = [i for i in self._invoices.values() if i.balance > 0]

        overdue = [i for i in invoices if i.is_overdue]
        with_discount = [
            i
            for i in invoices
            if i.early_pay_discount > 0
            and i.discount_deadline
            and i.discount_deadline > datetime.now()
        ]

        return {
            "totalPayables": float(sum(i.balance for i in invoices)),
            "invoiceCount": len(invoices),
            "overdueCount": len(overdue),
            "overdueAmount": float(sum(i.balance for i in overdue)),
            "discountOpportunities": len(with_discount),
            "potentialDiscountSavings": float(sum(i.discount_amount for i in with_discount)),
            "vendorCount": len(self._by_vendor),
        }
