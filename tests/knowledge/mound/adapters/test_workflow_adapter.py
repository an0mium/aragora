"""Tests for the WorkflowAdapter Knowledge Mound adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.workflow_adapter import (
    WorkflowAdapter,
    WorkflowOutcome,
    WorkflowSearchResult,
)


@pytest.fixture
def adapter() -> WorkflowAdapter:
    return WorkflowAdapter()


@pytest.fixture
def sample_outcome() -> WorkflowOutcome:
    return WorkflowOutcome(
        workflow_id="wf-001",
        definition_id="contract_review_v2",
        success=True,
        total_duration_ms=15000.0,
        step_count=4,
        failed_steps=0,
        step_summaries=[
            {"step_id": "s1", "step_name": "extract", "status": "COMPLETED", "duration_ms": 3000},
            {"step_id": "s2", "step_name": "analyze", "status": "COMPLETED", "duration_ms": 5000},
            {"step_id": "s3", "step_name": "validate", "status": "COMPLETED", "duration_ms": 4000},
            {"step_id": "s4", "step_name": "report", "status": "COMPLETED", "duration_ms": 3000},
        ],
        category="legal",
        template_name="contract_review_v2",
    )


class TestWorkflowAdapterInit:
    def test_init_defaults(self, adapter: WorkflowAdapter) -> None:
        assert adapter.adapter_name == "workflow"
        assert adapter.source_type == "workflow"
        assert adapter._pending_outcomes == []
        assert adapter._synced_outcomes == {}

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = WorkflowAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestWorkflowOutcome:
    def test_from_workflow_result(self) -> None:
        mock_result = MagicMock()
        mock_result.workflow_id = "wf-auto"
        mock_result.definition_id = "template_1"
        mock_result.success = True
        mock_result.total_duration_ms = 5000.0
        mock_result.error = None
        mock_result.checkpoints_created = 2

        mock_step = MagicMock()
        mock_step.step_id = "s1"
        mock_step.step_name = "step_one"
        mock_step.status = "COMPLETED"
        mock_step.duration_ms = 1000.0
        mock_step.retry_count = 0
        mock_step.error = None
        mock_result.steps = [mock_step]

        outcome = WorkflowOutcome.from_workflow_result(
            mock_result, category="healthcare", template_name="template_1"
        )

        assert outcome.workflow_id == "wf-auto"
        assert outcome.category == "healthcare"
        assert outcome.step_count == 1
        assert outcome.failed_steps == 0

    def test_from_workflow_result_with_failures(self) -> None:
        mock_result = MagicMock()
        mock_result.workflow_id = "wf-fail"
        mock_result.definition_id = "bad_template"
        mock_result.success = False
        mock_result.total_duration_ms = 2000.0
        mock_result.error = "Step failed"
        mock_result.checkpoints_created = 0

        mock_step = MagicMock()
        mock_step.step_id = "s1"
        mock_step.step_name = "failing_step"
        mock_step.status = "FAILED"
        mock_step.duration_ms = 500.0
        mock_step.retry_count = 2
        mock_step.error = "Connection timeout"
        mock_result.steps = [mock_step]

        outcome = WorkflowOutcome.from_workflow_result(mock_result)
        assert outcome.failed_steps == 1


class TestStoreExecution:
    def test_store_adds_to_pending(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        adapter.store_execution(sample_outcome)
        assert len(adapter._pending_outcomes) == 1
        assert adapter._pending_outcomes[0].metadata["km_sync_pending"] is True

    def test_store_emits_event(self, sample_outcome: WorkflowOutcome) -> None:
        callback = MagicMock()
        adapter = WorkflowAdapter(event_callback=callback)
        adapter.store_execution(sample_outcome)

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert data["workflow_id"] == "wf-001"


class TestGet:
    def test_get_returns_none_empty(self, adapter: WorkflowAdapter) -> None:
        assert adapter.get("missing") is None

    def test_get_with_prefix(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        adapter._synced_outcomes["wf-001"] = sample_outcome
        result = adapter.get("wf_wf-001")
        assert result is not None
        assert result.workflow_id == "wf-001"


class TestSearchByTemplate:
    @pytest.mark.asyncio
    async def test_search_by_definition_id(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        adapter._synced_outcomes["wf-001"] = sample_outcome
        results = await adapter.search_by_template("contract_review_v2")

        assert len(results) == 1
        assert results[0].workflow_id == "wf-001"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_search_success_only(self, adapter: WorkflowAdapter) -> None:
        success = WorkflowOutcome(
            workflow_id="wf-ok",
            definition_id="tmpl",
            success=True,
            total_duration_ms=1000,
            step_count=1,
            failed_steps=0,
        )
        fail = WorkflowOutcome(
            workflow_id="wf-fail",
            definition_id="tmpl",
            success=False,
            total_duration_ms=500,
            step_count=1,
            failed_steps=1,
        )
        adapter._synced_outcomes["wf-ok"] = success
        adapter._synced_outcomes["wf-fail"] = fail

        results = await adapter.search_by_template("tmpl", success_only=True)
        assert len(results) == 1
        assert results[0].success is True


class TestSearchByCategory:
    @pytest.mark.asyncio
    async def test_search_by_category(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        adapter._synced_outcomes["wf-001"] = sample_outcome
        results = await adapter.search_by_category("legal")

        assert len(results) == 1
        assert results[0].category == "legal"

    @pytest.mark.asyncio
    async def test_search_no_match(self, adapter: WorkflowAdapter) -> None:
        results = await adapter.search_by_category("finance")
        assert len(results) == 0


class TestToKnowledgeItem:
    def test_converts_success(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        item = adapter.to_knowledge_item(sample_outcome)

        assert item.id == "wf_wf-001"
        assert item.source_id == "wf-001"
        from aragora.knowledge.unified.types import ConfidenceLevel

        assert item.confidence == ConfidenceLevel.VERIFIED  # Success = verified confidence
        assert "succeeded" in item.content
        assert item.metadata["definition_id"] == "contract_review_v2"
        assert item.metadata["category"] == "legal"

    def test_converts_failure(self, adapter: WorkflowAdapter) -> None:
        outcome = WorkflowOutcome(
            workflow_id="wf-err",
            definition_id="bad",
            success=False,
            total_duration_ms=500,
            step_count=2,
            failed_steps=1,
            error="Timeout on step 2",
        )
        item = adapter.to_knowledge_item(outcome)

        from aragora.knowledge.unified.types import ConfidenceLevel

        assert item.confidence == ConfidenceLevel.LOW  # Failure = low confidence
        assert "failed" in item.content
        assert "Timeout" in item.content


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_success(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_execution(sample_outcome)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        assert result.records_failed == 0
        assert len(adapter._pending_outcomes) == 0
        assert "wf-001" in adapter._synced_outcomes

    @pytest.mark.asyncio
    async def test_sync_error_handling(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock(side_effect=RuntimeError("DB error"))

        adapter.store_execution(sample_outcome)
        result = await adapter.sync_to_km(mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1


class TestGetStats:
    def test_stats_empty(self, adapter: WorkflowAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["success_rate"] == 0.0

    def test_stats_with_data(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        adapter._synced_outcomes["wf-001"] = sample_outcome
        stats = adapter.get_stats()

        assert stats["total_synced"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration_ms"] == 15000.0
        assert "legal" in stats["categories"]


class TestMixinMethods:
    def test_get_record_by_id(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        adapter._synced_outcomes["wf-001"] = sample_outcome
        record = adapter._get_record_by_id("wf-001")
        assert record is not None

    def test_record_to_dict(
        self, adapter: WorkflowAdapter, sample_outcome: WorkflowOutcome
    ) -> None:
        d = adapter._record_to_dict(sample_outcome, similarity=0.8)
        assert d["id"] == "wf-001"
        assert d["success"] is True
        assert d["similarity"] == 0.8

    def test_extract_source_id(self, adapter: WorkflowAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "wf_test"}) == "test"
        assert adapter._extract_source_id({"source_id": "plain"}) == "plain"

    def test_get_fusion_sources(self, adapter: WorkflowAdapter) -> None:
        sources = adapter._get_fusion_sources()
        assert "debate" in sources
        assert "compliance" in sources
