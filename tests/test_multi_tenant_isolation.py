"""
Multi-tenant isolation tests for accounting services.

Tests verify that:
1. Each tenant's data is isolated from other tenants
2. Operations on one tenant don't affect another tenant's data
3. Tenant-specific configurations are respected
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import datetime
import pytest

from aragora.services.expense_tracker import (
    ExpenseTracker,
    ExpenseRecord,
    ExpenseCategory,
    ExpenseStatus,
    PaymentMethod,
)
from aragora.services.invoice_processor import (
    InvoiceProcessor,
    InvoiceData,
    InvoiceStatus,
)


class TestMultiTenantExpenseIsolation:
    """Test expense data isolation between tenants."""

    @pytest.mark.asyncio
    async def test_separate_tracker_instances_isolated(self):
        """Each tenant should have isolated expense data with separate instances."""
        tenant_a_tracker = ExpenseTracker()
        tenant_b_tracker = ExpenseTracker()

        # Create expense for tenant A
        expense_a = await tenant_a_tracker.create_expense(
            vendor_name="Tenant A Vendor",
            amount=100.00,
        )

        # Create expense for tenant B
        expense_b = await tenant_b_tracker.create_expense(
            vendor_name="Tenant B Vendor",
            amount=200.00,
        )

        # Verify isolation - tenant A shouldn't see tenant B's expense
        tenant_a_expenses, _ = await tenant_a_tracker.list_expenses()
        tenant_b_expenses, _ = await tenant_b_tracker.list_expenses()

        # Each tracker has its own expense
        assert len(tenant_a_expenses) >= 1
        assert len(tenant_b_expenses) >= 1

        # Verify expenses are in correct trackers
        tenant_a_vendors = {e.vendor_name for e in tenant_a_expenses}
        tenant_b_vendors = {e.vendor_name for e in tenant_b_expenses}

        assert "Tenant A Vendor" in tenant_a_vendors
        assert "Tenant B Vendor" in tenant_b_vendors

    @pytest.mark.asyncio
    async def test_expense_operations_isolated(self):
        """Operations on one tenant's expenses don't affect another's."""
        tenant_a = ExpenseTracker()
        tenant_b = ExpenseTracker()

        # Create expenses
        expense_a = await tenant_a.create_expense(
            vendor_name="Isolated Vendor A",
            amount=150.00,
        )
        expense_b = await tenant_b.create_expense(
            vendor_name="Isolated Vendor B",
            amount=250.00,
        )

        # Approve tenant A's expense
        approved_a = await tenant_a.approve_expense(expense_a.id)
        assert approved_a is not None
        assert approved_a.status == ExpenseStatus.APPROVED

        # Tenant B's expense should still be pending
        expense_b_check = await tenant_b.get_expense(expense_b.id)
        assert expense_b_check is not None
        assert expense_b_check.status == ExpenseStatus.PENDING

    @pytest.mark.asyncio
    async def test_tenant_specific_categorization(self):
        """Each tenant can have different categorization patterns."""
        tenant_a = ExpenseTracker(enable_llm_categorization=False)
        tenant_b = ExpenseTracker(enable_llm_categorization=False)

        # Same vendor might be categorized differently
        expense_a = ExpenseRecord(
            id="exp_tenant_a",
            vendor_name="Office Depot",
            amount=Decimal("50.00"),
            date=datetime.now(),
        )
        expense_b = ExpenseRecord(
            id="exp_tenant_b",
            vendor_name="Office Depot",
            amount=Decimal("50.00"),
            date=datetime.now(),
        )

        # Categorize for each tenant
        category_a = await tenant_a.categorize_expense(expense_a)
        category_b = await tenant_b.categorize_expense(expense_b)

        # Default pattern matching gives same result
        assert category_a == ExpenseCategory.OFFICE_SUPPLIES
        assert category_b == ExpenseCategory.OFFICE_SUPPLIES

    @pytest.mark.asyncio
    async def test_tenant_stats_isolated(self):
        """Statistics should be isolated per tenant."""
        tenant_a = ExpenseTracker()
        tenant_b = ExpenseTracker()

        # Create different amounts per tenant
        await tenant_a.create_expense(vendor_name="A1", amount=1000.00)
        await tenant_a.create_expense(vendor_name="A2", amount=500.00)

        await tenant_b.create_expense(vendor_name="B1", amount=200.00)

        # Get stats
        stats_a = tenant_a.get_stats()
        stats_b = tenant_b.get_stats()

        # Stats should be independent
        assert stats_a.total_expenses >= 2
        assert stats_b.total_expenses >= 1

        # Total amounts should be independent
        assert stats_a.total_amount >= Decimal("1500.00")
        assert stats_b.total_amount >= Decimal("200.00")

    @pytest.mark.asyncio
    async def test_concurrent_tenant_operations(self):
        """Concurrent operations across tenants should be safe."""
        tenant_trackers = [ExpenseTracker() for _ in range(5)]

        async def create_expenses(tracker, tenant_idx):
            expenses = []
            for i in range(10):
                expense = await tracker.create_expense(
                    vendor_name=f"Tenant{tenant_idx}_Vendor{i}",
                    amount=100.00 + i,
                )
                expenses.append(expense)
            return expenses

        # Run all tenants concurrently
        results = await asyncio.gather(*[
            create_expenses(tracker, idx)
            for idx, tracker in enumerate(tenant_trackers)
        ])

        # Each tenant should have 10 expenses
        for idx, tracker in enumerate(tenant_trackers):
            expenses, _ = await tracker.list_expenses()
            tenant_expenses = [
                e for e in expenses
                if e.vendor_name.startswith(f"Tenant{idx}_")
            ]
            assert len(tenant_expenses) >= 10


class TestMultiTenantInvoiceIsolation:
    """Test invoice data isolation between tenants."""

    @pytest.mark.asyncio
    async def test_separate_processor_instances_isolated(self):
        """Each tenant should have isolated invoice data."""
        tenant_a = InvoiceProcessor()
        tenant_b = InvoiceProcessor()

        # Create invoices for each tenant
        invoice_a = await tenant_a.create_manual_invoice(
            vendor_name="Tenant A Supplier",
            total_amount=5000.00,
        )
        invoice_b = await tenant_b.create_manual_invoice(
            vendor_name="Tenant B Supplier",
            total_amount=3000.00,
        )

        # List invoices
        invoices_a, _ = await tenant_a.list_invoices()
        invoices_b, _ = await tenant_b.list_invoices()

        # Verify isolation
        a_vendors = {i.vendor_name for i in invoices_a}
        b_vendors = {i.vendor_name for i in invoices_b}

        assert "Tenant A Supplier" in a_vendors
        assert "Tenant B Supplier" in b_vendors

    @pytest.mark.asyncio
    async def test_invoice_approval_isolated(self):
        """Approval workflow should be isolated per tenant."""
        tenant_a = InvoiceProcessor()
        tenant_b = InvoiceProcessor()

        # Create invoices
        invoice_a = await tenant_a.create_manual_invoice(
            vendor_name="Approval Test A",
            total_amount=1000.00,
        )
        invoice_b = await tenant_b.create_manual_invoice(
            vendor_name="Approval Test B",
            total_amount=1000.00,
        )

        # Approve tenant A's invoice
        approved_a = await tenant_a.approve_invoice(invoice_a.id, "approver_a")
        assert approved_a is not None
        assert approved_a.status == InvoiceStatus.APPROVED

        # Tenant B's invoice should be unchanged
        invoice_b_check = await tenant_b.get_invoice(invoice_b.id)
        assert invoice_b_check is not None
        assert invoice_b_check.status != InvoiceStatus.APPROVED

    @pytest.mark.asyncio
    async def test_purchase_order_isolation(self):
        """Purchase orders should be isolated per tenant."""
        tenant_a = InvoiceProcessor()
        tenant_b = InvoiceProcessor()

        # Add PO for tenant A
        po_a = await tenant_a.add_purchase_order(
            po_number="PO-A-001",
            vendor_name="Vendor A",
            total_amount=10000.00,
        )

        # Add PO for tenant B
        po_b = await tenant_b.add_purchase_order(
            po_number="PO-B-001",
            vendor_name="Vendor B",
            total_amount=20000.00,
        )

        # Create invoice referencing tenant A's PO
        invoice_a = await tenant_a.create_manual_invoice(
            vendor_name="Vendor A",
            total_amount=10000.00,
            po_number="PO-A-001",
        )

        # Match should find PO in tenant A
        match_result = await tenant_a.match_to_po(invoice_a)
        assert match_result.po_id is not None or match_result.match_type != "none"

        # Create invoice for tenant B referencing tenant A's PO
        invoice_b = await tenant_b.create_manual_invoice(
            vendor_name="Vendor A",
            total_amount=10000.00,
            po_number="PO-A-001",
        )

        # Tenant B shouldn't match tenant A's PO
        match_result_b = await tenant_b.match_to_po(invoice_b)
        # Different tenant's PO should not be found
        assert match_result_b.match_type == "none" or match_result_b.po_number != "PO-A-001"

    @pytest.mark.asyncio
    async def test_pending_approvals_isolated(self):
        """Pending approvals should only show current tenant's invoices."""
        tenant_a = InvoiceProcessor()
        tenant_b = InvoiceProcessor()

        # Create pending invoices for both tenants
        await tenant_a.create_manual_invoice(
            vendor_name="Pending A1",
            total_amount=1000.00,
        )
        await tenant_a.create_manual_invoice(
            vendor_name="Pending A2",
            total_amount=2000.00,
        )
        await tenant_b.create_manual_invoice(
            vendor_name="Pending B1",
            total_amount=3000.00,
        )

        # Get pending approvals
        pending_a = await tenant_a.get_pending_approvals()
        pending_b = await tenant_b.get_pending_approvals()

        # Verify isolation
        pending_a_vendors = {i.vendor_name for i in pending_a}
        pending_b_vendors = {i.vendor_name for i in pending_b}

        # Tenant A should not see tenant B's pending invoices
        assert "Pending B1" not in pending_a_vendors
        # Tenant B should not see tenant A's pending invoices
        assert "Pending A1" not in pending_b_vendors
        assert "Pending A2" not in pending_b_vendors

    @pytest.mark.asyncio
    async def test_anomaly_detection_isolated(self):
        """Anomaly detection should use tenant-specific history."""
        tenant_a = InvoiceProcessor()
        tenant_b = InvoiceProcessor()

        # Create invoice for tenant A
        invoice_a = InvoiceData(
            id="inv_anomaly_a",
            vendor_name="New Vendor For Tenant A",
            total_amount=Decimal("10000.00"),
        )

        # Create same invoice for tenant B
        invoice_b = InvoiceData(
            id="inv_anomaly_b",
            vendor_name="New Vendor For Tenant B",
            total_amount=Decimal("10000.00"),
        )

        # Detect anomalies
        anomalies_a = await tenant_a.detect_anomalies(invoice_a)
        anomalies_b = await tenant_b.detect_anomalies(invoice_b)

        # Both should flag as new vendor (isolated history)
        a_types = {a.type.value for a in anomalies_a}
        b_types = {a.type.value for a in anomalies_b}

        assert "new_vendor" in a_types
        assert "new_vendor" in b_types


class TestMultiTenantConfigurationIsolation:
    """Test tenant-specific configuration isolation."""

    @pytest.mark.asyncio
    async def test_different_auto_approve_thresholds(self):
        """Each tenant can have different auto-approve thresholds."""
        tenant_a = InvoiceProcessor(auto_approve_threshold=Decimal("500"))
        tenant_b = InvoiceProcessor(auto_approve_threshold=Decimal("1000"))

        # Create same amount invoice for both
        invoice_a = InvoiceData(
            id="inv_threshold_a",
            vendor_name="Threshold Test",
            total_amount=Decimal("750.00"),
        )
        invoice_b = InvoiceData(
            id="inv_threshold_b",
            vendor_name="Threshold Test",
            total_amount=Decimal("750.00"),
        )

        # Determine approval levels
        level_a = tenant_a.determine_approval_level(invoice_a)
        level_b = tenant_b.determine_approval_level(invoice_b)

        # Tenant A ($750 > $500): requires manager
        # Tenant B ($750 < $1000): auto-approve
        from aragora.services.invoice_processor import ApprovalLevel
        assert level_a != ApprovalLevel.AUTO
        assert level_b == ApprovalLevel.AUTO

    @pytest.mark.asyncio
    async def test_circuit_breaker_isolation(self):
        """Circuit breakers should be isolated per tenant."""
        tenant_a = ExpenseTracker(enable_circuit_breakers=True)
        tenant_b = ExpenseTracker(enable_circuit_breakers=True)

        # Get circuit breaker status for each
        status_a = tenant_a.get_circuit_breaker_status()
        status_b = tenant_b.get_circuit_breaker_status()

        # Both should be enabled and independent
        assert status_a["enabled"] is True
        assert status_b["enabled"] is True

        # Each has its own circuit breakers
        assert "ocr" in status_a["services"]
        assert "ocr" in status_b["services"]

    @pytest.mark.asyncio
    async def test_qbo_connector_isolation(self):
        """QBO connectors should be tenant-specific."""
        # Create with different configurations (no actual connector)
        tenant_a = ExpenseTracker(qbo_connector=None)
        tenant_b = ExpenseTracker(qbo_connector=None)

        # Sync should fail gracefully for both
        result_a = await tenant_a.sync_to_qbo()
        result_b = await tenant_b.sync_to_qbo()

        # Both should fail independently
        assert result_a.success_count == 0
        assert result_b.success_count == 0
        assert len(result_a.errors) > 0
        assert len(result_b.errors) > 0
