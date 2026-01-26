"""
Multi-Tenant Data Isolation Audit Tests.

SOC 2 Control: CC6.1 - Logical Access Security
SOC 2 Control: CC6.3 - Data Confidentiality

These tests verify that tenant data isolation is enforced across all
system components. Results can be exported for SOC 2 audit evidence.

Run with: pytest tests/security/test_tenant_isolation_audit.py -v --tb=short
Generate report: pytest tests/security/test_tenant_isolation_audit.py --json-report
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Core isolation imports
from aragora.tenancy.context import TenantContext, get_current_tenant_id
from aragora.tenancy.context import TenantNotSetError
from aragora.tenancy.isolation import (
    IsolationLevel,
    IsolationViolation,
    TenantDataIsolation,
    TenantIsolationConfig,
)


@dataclass
class AuditTestResult:
    """Result of a single isolation audit test."""

    test_name: str
    category: str
    passed: bool
    tenant_a: str
    tenant_b: str
    resource_type: str
    operation: str
    details: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "category": self.category,
            "passed": self.passed,
            "tenant_a": self.tenant_a,
            "tenant_b": self.tenant_b,
            "resource_type": self.resource_type,
            "operation": self.operation,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IsolationAuditReport:
    """Complete isolation audit report for SOC 2 evidence."""

    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    environment: str = "test"
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    categories_tested: Set[str] = field(default_factory=set)
    resource_types_tested: Set[str] = field(default_factory=set)
    results: List[AuditTestResult] = field(default_factory=list)

    def add_result(self, result: AuditTestResult) -> None:
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.categories_tested.add(result.category)
        self.resource_types_tested.add(result.resource_type)

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "environment": self.environment,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": f"{self.pass_rate:.2f}%",
            "categories_tested": sorted(self.categories_tested),
            "resource_types_tested": sorted(self.resource_types_tested),
            "results": [r.to_dict() for r in self.results],
        }

    def save_report(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global report collector
_audit_report = IsolationAuditReport()


@pytest.fixture(scope="module")
def audit_report():
    """Shared audit report for all tests in this module."""
    global _audit_report
    _audit_report = IsolationAuditReport()
    yield _audit_report
    # Save report at end of test run
    report_path = os.environ.get(
        "ISOLATION_AUDIT_REPORT_PATH",
        "/tmp/tenant_isolation_audit_report.json",
    )
    _audit_report.save_report(report_path)


@pytest.fixture
def isolation_strict():
    """Strict isolation config for audit tests."""
    config = TenantIsolationConfig(
        level=IsolationLevel.STRICT,
        tenant_column="tenant_id",
        encrypt_at_rest=True,
        per_tenant_keys=True,
        strict_validation=True,
        shared_resources=[],
    )
    return TenantDataIsolation(config)


@pytest.fixture
def tenant_a():
    """First test tenant."""
    return "tenant_audit_a"


@pytest.fixture
def tenant_b():
    """Second test tenant (should never see tenant_a data)."""
    return "tenant_audit_b"


# =============================================================================
# Category 1: Query Filter Isolation
# =============================================================================


class TestQueryFilterIsolation:
    """Tests that all query operations are properly tenant-scoped."""

    RESOURCE_TYPES = [
        "debates",
        "agents",
        "rounds",
        "votes",
        "critiques",
        "knowledge_nodes",
        "workflows",
        "checkpoints",
        "audit_entries",
        "findings",
        "decisions",
        "integrations",
        "tokens",
        "messages",
    ]

    @pytest.mark.parametrize("resource_type", RESOURCE_TYPES)
    def test_dict_query_adds_tenant_filter(
        self, isolation_strict, tenant_a, resource_type, audit_report
    ):
        """Verify dict queries always include tenant_id filter."""
        with TenantContext(tenant_id=tenant_a):
            original_query = {"status": "active", "created_at": {"$gt": "2024-01-01"}}
            filtered = isolation_strict.filter_query(original_query, resource_type)

            passed = "tenant_id" in filtered and filtered["tenant_id"] == tenant_a

            audit_report.add_result(
                AuditTestResult(
                    test_name="dict_query_tenant_filter",
                    category="query_filtering",
                    passed=passed,
                    tenant_a=tenant_a,
                    tenant_b="n/a",
                    resource_type=resource_type,
                    operation="filter_query",
                    details=f"tenant_id in result: {passed}",
                )
            )

            assert passed, f"Query for {resource_type} missing tenant_id filter"

    @pytest.mark.parametrize("resource_type", RESOURCE_TYPES)
    def test_sql_query_adds_tenant_filter(
        self, isolation_strict, tenant_a, resource_type, audit_report
    ):
        """Verify SQL queries use parameterized tenant filtering."""
        with TenantContext(tenant_id=tenant_a):
            original_sql = f"SELECT * FROM {resource_type} WHERE status = 'active'"
            modified_sql, params = isolation_strict.filter_sql(original_sql, resource_type)

            # Tenant ID must be in params, not interpolated into SQL
            tenant_in_params = params.get("tenant_id") == tenant_a
            tenant_not_in_sql = tenant_a not in modified_sql
            has_where_clause = "tenant_id" in modified_sql.lower()

            passed = tenant_in_params and tenant_not_in_sql and has_where_clause

            audit_report.add_result(
                AuditTestResult(
                    test_name="sql_query_parameterized",
                    category="query_filtering",
                    passed=passed,
                    tenant_a=tenant_a,
                    tenant_b="n/a",
                    resource_type=resource_type,
                    operation="filter_sql",
                    details=f"parameterized: {tenant_in_params}, not interpolated: {tenant_not_in_sql}",
                )
            )

            assert passed, f"SQL for {resource_type} not properly parameterized"


# =============================================================================
# Category 2: Cross-Tenant Access Prevention
# =============================================================================


class TestCrossTenantAccessPrevention:
    """Tests that cross-tenant data access is blocked."""

    @pytest.fixture
    def resource_tenant_a(self, tenant_a):
        """Resource owned by tenant A."""
        return {
            "id": "res_001",
            "tenant_id": tenant_a,
            "name": "Tenant A Resource",
            "data": "confidential_a",
        }

    @pytest.fixture
    def resource_tenant_b(self, tenant_b):
        """Resource owned by tenant B."""
        return {
            "id": "res_002",
            "tenant_id": tenant_b,
            "name": "Tenant B Resource",
            "data": "confidential_b",
        }

    def test_tenant_b_cannot_access_tenant_a_resource(
        self,
        isolation_strict,
        tenant_a,
        tenant_b,
        resource_tenant_a,
        audit_report,
    ):
        """Tenant B attempting to access Tenant A's resource must fail."""
        with TenantContext(tenant_id=tenant_b):
            try:
                isolation_strict.validate_ownership(resource_tenant_a, tenant_field="tenant_id")
                passed = False
                details = "VIOLATION: Cross-tenant access allowed!"
            except IsolationViolation:
                passed = True
                details = "Access correctly denied via IsolationViolation"

            audit_report.add_result(
                AuditTestResult(
                    test_name="cross_tenant_access_blocked",
                    category="access_prevention",
                    passed=passed,
                    tenant_a=tenant_a,
                    tenant_b=tenant_b,
                    resource_type="generic",
                    operation="validate_ownership",
                    details=details,
                )
            )

            assert passed, "Cross-tenant access was not blocked!"

    def test_tenant_a_can_access_own_resource(
        self,
        isolation_strict,
        tenant_a,
        resource_tenant_a,
        audit_report,
    ):
        """Tenant A accessing own resource must succeed."""
        with TenantContext(tenant_id=tenant_a):
            try:
                isolation_strict.validate_ownership(resource_tenant_a, tenant_field="tenant_id")
                passed = True
                details = "Own-tenant access correctly allowed"
            except IsolationViolation:
                passed = False
                details = "Own-tenant access incorrectly blocked"

            audit_report.add_result(
                AuditTestResult(
                    test_name="own_tenant_access_allowed",
                    category="access_prevention",
                    passed=passed,
                    tenant_a=tenant_a,
                    tenant_b="n/a",
                    resource_type="generic",
                    operation="validate_ownership",
                    details=details,
                )
            )

            assert passed, "Own-tenant access was blocked incorrectly!"


# =============================================================================
# Category 3: SQL Injection Prevention
# =============================================================================


class TestSQLInjectionPrevention:
    """Tests that malicious tenant IDs cannot cause SQL injection."""

    INJECTION_PAYLOADS = [
        "' OR '1'='1",
        "'; DROP TABLE debates; --",
        "' UNION SELECT * FROM users --",
        "1; DELETE FROM debates WHERE '1'='1",
        "tenant' AND 1=1 --",
        "' OR 1=1 --",
        "admin'--",
        "1' ORDER BY 1--+",
        "' AND '1'='1",
    ]

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    def test_sql_injection_payload_parameterized(self, isolation_strict, payload, audit_report):
        """Malicious tenant IDs must be parameterized, not interpolated."""
        with TenantContext(tenant_id=payload):
            original_sql = "SELECT * FROM debates WHERE status = 'active'"
            modified_sql, params = isolation_strict.filter_sql(original_sql, "debates")

            # Payload must be in params, never in SQL string
            payload_in_params = params.get("tenant_id") == payload
            payload_not_in_sql = payload not in modified_sql

            passed = payload_in_params and payload_not_in_sql

            audit_report.add_result(
                AuditTestResult(
                    test_name="sql_injection_prevention",
                    category="sql_injection",
                    passed=passed,
                    tenant_a=payload[:20] + "..." if len(payload) > 20 else payload,
                    tenant_b="n/a",
                    resource_type="debates",
                    operation="filter_sql",
                    details=f"Payload safely parameterized: {passed}",
                )
            )

            assert passed, f"SQL injection payload not safely handled: {payload}"


# =============================================================================
# Category 4: Encryption Key Isolation
# =============================================================================


class TestEncryptionKeyIsolation:
    """Tests that encryption keys are isolated per tenant."""

    def test_different_tenants_get_different_keys(
        self, isolation_strict, tenant_a, tenant_b, audit_report
    ):
        """Each tenant must have a unique encryption key."""
        with TenantContext(tenant_id=tenant_a):
            key_a = isolation_strict.get_encryption_key()

        with TenantContext(tenant_id=tenant_b):
            key_b = isolation_strict.get_encryption_key()

        passed = key_a != key_b

        audit_report.add_result(
            AuditTestResult(
                test_name="per_tenant_encryption_keys",
                category="encryption",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b=tenant_b,
                resource_type="encryption_key",
                operation="get_encryption_key",
                details=f"Keys are different: {passed}",
            )
        )

        assert passed, "Tenants have same encryption key - isolation breach!"

    def test_same_tenant_gets_consistent_key(self, isolation_strict, tenant_a, audit_report):
        """Same tenant must get consistent key across calls."""
        with TenantContext(tenant_id=tenant_a):
            key_1 = isolation_strict.get_encryption_key()
            key_2 = isolation_strict.get_encryption_key()

        passed = key_1 == key_2

        audit_report.add_result(
            AuditTestResult(
                test_name="consistent_tenant_key",
                category="encryption",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b="n/a",
                resource_type="encryption_key",
                operation="get_encryption_key",
                details=f"Keys are consistent: {passed}",
            )
        )

        assert passed, "Tenant key is not consistent across calls!"


# =============================================================================
# Category 5: Context Isolation (Async Safety)
# =============================================================================


class TestContextIsolation:
    """Tests that tenant context is isolated across async operations."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_isolated(self, tenant_a, tenant_b, audit_report):
        """Concurrent requests must maintain separate tenant contexts."""
        results = []

        async def request_handler(tenant_id: str, delay: float) -> str:
            with TenantContext(tenant_id=tenant_id):
                await asyncio.sleep(delay)
                return get_current_tenant_id()

        # Run concurrent requests with overlapping execution
        tasks = [
            request_handler(tenant_a, 0.05),
            request_handler(tenant_b, 0.02),
            request_handler(tenant_a, 0.03),
            request_handler(tenant_b, 0.01),
        ]

        results = await asyncio.gather(*tasks)
        expected = [tenant_a, tenant_b, tenant_a, tenant_b]
        passed = results == expected

        audit_report.add_result(
            AuditTestResult(
                test_name="async_context_isolation",
                category="context_isolation",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b=tenant_b,
                resource_type="context",
                operation="concurrent_requests",
                details=f"Results: {results}, Expected: {expected}",
            )
        )

        assert passed, f"Context leakage detected: {results} != {expected}"

    def test_context_cleanup_on_exit(self, tenant_a, audit_report):
        """Tenant context must be cleaned up after context manager exits."""
        # Before: no tenant
        before = get_current_tenant_id()

        with TenantContext(tenant_id=tenant_a):
            during = get_current_tenant_id()

        # After: no tenant
        after = get_current_tenant_id()

        passed = before is None and during == tenant_a and after is None

        audit_report.add_result(
            AuditTestResult(
                test_name="context_cleanup",
                category="context_isolation",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b="n/a",
                resource_type="context",
                operation="context_cleanup",
                details=f"before={before}, during={during}, after={after}",
            )
        )

        assert passed, "Tenant context not properly cleaned up!"


# =============================================================================
# Category 6: Namespace Key Isolation
# =============================================================================


class TestNamespaceKeyIsolation:
    """Tests that cache/storage keys are tenant-namespaced."""

    def test_namespace_keys_include_tenant(
        self, isolation_strict, tenant_a, tenant_b, audit_report
    ):
        """Cache keys must be namespaced by tenant."""
        with TenantContext(tenant_id=tenant_a):
            key_a = isolation_strict.namespace_key("debate:123")

        with TenantContext(tenant_id=tenant_b):
            key_b = isolation_strict.namespace_key("debate:123")

        # Same logical key, different namespaced keys
        keys_different = key_a != key_b
        tenant_a_in_key = tenant_a in key_a
        tenant_b_in_key = tenant_b in key_b

        passed = keys_different and tenant_a_in_key and tenant_b_in_key

        audit_report.add_result(
            AuditTestResult(
                test_name="namespace_key_isolation",
                category="namespace_isolation",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b=tenant_b,
                resource_type="cache_key",
                operation="namespace_key",
                details=f"key_a={key_a}, key_b={key_b}",
            )
        )

        assert passed, "Namespace keys not properly isolated by tenant!"


# =============================================================================
# Category 7: Isolation Level Enforcement
# =============================================================================


class TestIsolationLevelEnforcement:
    """Tests that isolation levels are properly enforced."""

    def test_strict_level_requires_tenant(self, isolation_strict, audit_report):
        """STRICT isolation must require tenant context."""
        # No tenant context
        try:
            isolation_strict.filter_query({"status": "active"}, "debates")
            passed = False
            details = "Query allowed without tenant context"
        except (IsolationViolation, TenantNotSetError, ValueError):
            passed = True
            details = "Query correctly rejected without tenant"

        audit_report.add_result(
            AuditTestResult(
                test_name="strict_requires_tenant",
                category="isolation_level",
                passed=passed,
                tenant_a="none",
                tenant_b="n/a",
                resource_type="debates",
                operation="filter_query",
                details=details,
            )
        )

        assert passed, "STRICT isolation allowed query without tenant!"


# =============================================================================
# Report Generation
# =============================================================================


class TestAuditReportGeneration:
    """Tests for audit report generation."""

    def test_report_captures_all_categories(self, audit_report):
        """Verify report captures multiple categories."""
        # This test runs last and validates report completeness
        expected_categories = {
            "query_filtering",
            "access_prevention",
            "sql_injection",
            "encryption",
            "context_isolation",
            "namespace_isolation",
            "isolation_level",
        }

        # Check that we have results from multiple categories
        assert len(audit_report.categories_tested) >= 3, (
            f"Report should cover at least 3 categories, got: {audit_report.categories_tested}"
        )

    def test_report_has_passing_tests(self, audit_report):
        """Verify report has high pass rate."""
        assert audit_report.pass_rate >= 90.0, (
            f"Pass rate should be >= 90%, got: {audit_report.pass_rate:.2f}%"
        )


# =============================================================================
# Compliance Matrix Test
# =============================================================================


class TestComplianceMatrix:
    """Cross-reference tests for SOC 2 compliance evidence."""

    SOC2_CONTROLS = {
        "CC6.1": "Logical Access Security",
        "CC6.3": "Data Confidentiality",
        "CC6.6": "Security Event Monitoring",
        "CC6.7": "Access Restriction",
        "CC7.2": "System Monitoring",
    }

    def test_cc6_1_logical_access_query_filtering(self, isolation_strict, tenant_a, audit_report):
        """CC6.1: Verify logical access via query filtering."""
        with TenantContext(tenant_id=tenant_a):
            filtered = isolation_strict.filter_query({}, "debates")
            passed = filtered.get("tenant_id") == tenant_a

        audit_report.add_result(
            AuditTestResult(
                test_name="SOC2_CC6.1_logical_access",
                category="soc2_compliance",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b="n/a",
                resource_type="debates",
                operation="filter_query",
                details="CC6.1: Logical access enforced via tenant filtering",
            )
        )

        assert passed

    def test_cc6_3_data_confidentiality_encryption(
        self, isolation_strict, tenant_a, tenant_b, audit_report
    ):
        """CC6.3: Verify data confidentiality via per-tenant encryption."""
        with TenantContext(tenant_id=tenant_a):
            key_a = isolation_strict.get_encryption_key()
        with TenantContext(tenant_id=tenant_b):
            key_b = isolation_strict.get_encryption_key()

        passed = key_a != key_b

        audit_report.add_result(
            AuditTestResult(
                test_name="SOC2_CC6.3_data_confidentiality",
                category="soc2_compliance",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b=tenant_b,
                resource_type="encryption",
                operation="per_tenant_keys",
                details="CC6.3: Data confidentiality via isolated encryption keys",
            )
        )

        assert passed

    def test_cc6_7_access_restriction_cross_tenant(
        self, isolation_strict, tenant_a, tenant_b, audit_report
    ):
        """CC6.7: Verify access restriction blocks cross-tenant access."""
        resource = {"id": "test", "tenant_id": tenant_a}

        with TenantContext(tenant_id=tenant_b):
            try:
                isolation_strict.validate_ownership(resource)
                passed = False
            except IsolationViolation:
                passed = True

        audit_report.add_result(
            AuditTestResult(
                test_name="SOC2_CC6.7_access_restriction",
                category="soc2_compliance",
                passed=passed,
                tenant_a=tenant_a,
                tenant_b=tenant_b,
                resource_type="generic",
                operation="validate_ownership",
                details="CC6.7: Access restriction enforced via ownership validation",
            )
        )

        assert passed
