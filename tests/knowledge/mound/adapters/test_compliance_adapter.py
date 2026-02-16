"""Tests for the ComplianceAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.compliance_adapter import (
    CheckOutcome,
    ComplianceAdapter,
    ComplianceSearchResult,
    ViolationOutcome,
)


@pytest.fixture
def adapter() -> ComplianceAdapter:
    return ComplianceAdapter()


@pytest.fixture
def sample_check() -> CheckOutcome:
    return CheckOutcome(
        check_id="scan-001",
        compliant=False,
        score=0.72,
        frameworks_checked=["soc2", "gdpr"],
        issue_count=3,
        critical_count=1,
        high_count=1,
        issues_summary=[
            {
                "framework": "soc2",
                "rule_id": "CC6.1",
                "severity": "critical",
                "description": "Missing encryption at rest",
            },
            {
                "framework": "soc2",
                "rule_id": "CC7.2",
                "severity": "high",
                "description": "Incomplete access logging",
            },
            {
                "framework": "gdpr",
                "rule_id": "Art13",
                "severity": "medium",
                "description": "Privacy notice incomplete",
            },
        ],
    )


@pytest.fixture
def sample_violation() -> ViolationOutcome:
    return ViolationOutcome(
        violation_id="vio-001",
        policy_id="pol-soc2",
        rule_id="CC6.1",
        rule_name="Encryption at Rest",
        framework_id="soc2",
        severity="critical",
        status="open",
        description="Database storage is not encrypted at rest",
        source="aragora/storage/postgres_store.py",
        workspace_id="ws-main",
    )


class TestComplianceAdapterInit:
    def test_init_defaults(self, adapter: ComplianceAdapter) -> None:
        assert adapter.adapter_name == "compliance"
        assert adapter.source_type == "compliance"
        assert adapter._pending_checks == []
        assert adapter._pending_violations == []

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = ComplianceAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestCheckOutcome:
    def test_from_check_result(self) -> None:
        mock_result = MagicMock()
        mock_result.compliant = True
        mock_result.score = 0.95
        mock_result.frameworks_checked = ["hipaa"]

        mock_issue = MagicMock()
        mock_issue.framework = "hipaa"
        mock_issue.rule_id = "164.312"
        mock_issue.severity = "low"
        mock_issue.description = "Minor documentation gap"
        mock_result.issues = [mock_issue]

        outcome = CheckOutcome.from_check_result(mock_result, check_id="test-check")

        assert outcome.check_id == "test-check"
        assert outcome.compliant is True
        assert outcome.score == 0.95
        assert outcome.issue_count == 1
        assert outcome.critical_count == 0


class TestViolationOutcome:
    def test_from_violation(self) -> None:
        mock_violation = MagicMock()
        mock_violation.id = "v-123"
        mock_violation.policy_id = "p-1"
        mock_violation.rule_id = "r-1"
        mock_violation.rule_name = "Test Rule"
        mock_violation.framework_id = "soc2"
        mock_violation.severity = "high"
        mock_violation.status = "investigating"
        mock_violation.description = "Test violation"
        mock_violation.source = "test.py"
        mock_violation.workspace_id = "ws-1"
        mock_violation.detected_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_violation.resolved_at = None
        mock_violation.resolution_notes = None

        outcome = ViolationOutcome.from_violation(mock_violation)

        assert outcome.violation_id == "v-123"
        assert outcome.severity == "high"
        assert outcome.status == "investigating"


class TestStoreCheck:
    def test_store_adds_to_pending(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        adapter.store_check(sample_check)
        assert len(adapter._pending_checks) == 1
        assert adapter._pending_checks[0].metadata["km_sync_pending"] is True

    def test_store_emits_event(self, sample_check: CheckOutcome) -> None:
        callback = MagicMock()
        adapter = ComplianceAdapter(event_callback=callback)
        adapter.store_check(sample_check)

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert data["check_id"] == "scan-001"


class TestStoreViolation:
    def test_store_adds_to_pending(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        adapter.store_violation(sample_violation)
        assert len(adapter._pending_violations) == 1
        assert adapter._pending_violations[0].metadata["km_sync_pending"] is True

    def test_store_emits_event(self, sample_violation: ViolationOutcome) -> None:
        callback = MagicMock()
        adapter = ComplianceAdapter(event_callback=callback)
        adapter.store_violation(sample_violation)

        callback.assert_called_once()
        _, data = callback.call_args[0]
        assert data["severity"] == "critical"


class TestGet:
    def test_get_check_with_prefix(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        adapter._synced_checks["scan-001"] = sample_check
        result = adapter.get("cc_scan-001")
        assert result is not None
        assert isinstance(result, CheckOutcome)

    def test_get_violation_with_prefix(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        adapter._synced_violations["vio-001"] = sample_violation
        result = adapter.get("cv_vio-001")
        assert result is not None
        assert isinstance(result, ViolationOutcome)

    def test_get_without_prefix(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        adapter._synced_checks["scan-001"] = sample_check
        result = adapter.get("scan-001")
        assert result is not None


class TestSearchViolations:
    @pytest.mark.asyncio
    async def test_search_by_framework(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        adapter._synced_violations["vio-001"] = sample_violation
        results = await adapter.search_violations(framework="soc2")

        assert len(results) == 1
        assert results[0].framework == "soc2"
        assert results[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_search_by_severity(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        adapter._synced_violations["vio-001"] = sample_violation
        results = await adapter.search_violations(severity="critical")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_by_status(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        adapter._synced_violations["vio-001"] = sample_violation
        results = await adapter.search_violations(status="open")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_no_match(self, adapter: ComplianceAdapter) -> None:
        results = await adapter.search_violations(framework="nonexistent")
        assert len(results) == 0


class TestSearchChecks:
    @pytest.mark.asyncio
    async def test_search_by_framework(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        adapter._synced_checks["scan-001"] = sample_check
        results = await adapter.search_checks(framework="soc2")

        assert len(results) == 1
        assert results[0].score == 0.72

    @pytest.mark.asyncio
    async def test_search_min_score(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        adapter._synced_checks["scan-001"] = sample_check
        results = await adapter.search_checks(min_score=0.9)
        assert len(results) == 0


class TestToKnowledgeItem:
    def test_check_to_knowledge_item(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        item = adapter.check_to_knowledge_item(sample_check)

        assert item.id == "cc_scan-001"
        from aragora.knowledge.unified.types import ConfidenceLevel

        assert item.confidence == ConfidenceLevel.HIGH  # 0.72 maps to HIGH
        assert "non-compliant" in item.content
        assert item.metadata["record_type"] == "check"
        assert item.metadata["critical_count"] == 1

    def test_violation_to_knowledge_item(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        item = adapter.violation_to_knowledge_item(sample_violation)

        assert item.id == "cv_vio-001"
        from aragora.knowledge.unified.types import ConfidenceLevel

        assert (
            item.confidence == ConfidenceLevel.UNVERIFIED
        )  # Critical severity = very low confidence
        assert "critical" in item.content.lower()
        assert item.metadata["record_type"] == "violation"
        assert item.metadata["framework_id"] == "soc2"

    def test_to_knowledge_item_dispatches(
        self,
        adapter: ComplianceAdapter,
        sample_check: CheckOutcome,
        sample_violation: ViolationOutcome,
    ) -> None:
        check_item = adapter.to_knowledge_item(sample_check)
        assert check_item.id.startswith("cc_")

        violation_item = adapter.to_knowledge_item(sample_violation)
        assert violation_item.id.startswith("cv_")


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_checks(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_check(sample_check)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        assert "scan-001" in adapter._synced_checks

    @pytest.mark.asyncio
    async def test_sync_violations(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_violation(sample_violation)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        assert "vio-001" in adapter._synced_violations

    @pytest.mark.asyncio
    async def test_sync_mixed(
        self,
        adapter: ComplianceAdapter,
        sample_check: CheckOutcome,
        sample_violation: ViolationOutcome,
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_check(sample_check)
        adapter.store_violation(sample_violation)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 2

    @pytest.mark.asyncio
    async def test_sync_error_handling(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock(side_effect=RuntimeError("DB down"))

        adapter.store_check(sample_check)
        result = await adapter.sync_to_km(mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1


class TestGetStats:
    def test_stats_empty(self, adapter: ComplianceAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_checks_synced"] == 0
        assert stats["total_violations_synced"] == 0

    def test_stats_with_data(
        self,
        adapter: ComplianceAdapter,
        sample_check: CheckOutcome,
        sample_violation: ViolationOutcome,
    ) -> None:
        adapter._synced_checks["scan-001"] = sample_check
        adapter._synced_violations["vio-001"] = sample_violation
        stats = adapter.get_stats()

        assert stats["total_checks_synced"] == 1
        assert stats["total_violations_synced"] == 1
        assert stats["avg_compliance_score"] == 0.72
        assert stats["open_violations"] == 1
        assert stats["critical_violations"] == 1


class TestMixinMethods:
    def test_record_to_dict_check(
        self, adapter: ComplianceAdapter, sample_check: CheckOutcome
    ) -> None:
        d = adapter._record_to_dict(sample_check, similarity=0.7)
        assert d["type"] == "check"
        assert d["score"] == 0.72

    def test_record_to_dict_violation(
        self, adapter: ComplianceAdapter, sample_violation: ViolationOutcome
    ) -> None:
        d = adapter._record_to_dict(sample_violation, similarity=0.8)
        assert d["type"] == "violation"
        assert d["severity"] == "critical"

    def test_extract_source_id(self, adapter: ComplianceAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "cc_test"}) == "test"
        assert adapter._extract_source_id({"source_id": "cv_test"}) == "test"
        assert adapter._extract_source_id({"source_id": "plain"}) == "plain"

    def test_get_fusion_sources(self, adapter: ComplianceAdapter) -> None:
        sources = adapter._get_fusion_sources()
        assert "debate" in sources
        assert "workflow" in sources
