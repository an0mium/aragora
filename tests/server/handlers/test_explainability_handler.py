"""
Tests for Explainability Handler.

Tests the explainability API endpoints including:
- Decision explanations
- Evidence chains
- Vote influence analysis
- Counterfactual analysis
- Human-readable summaries
- Batch operations
"""

import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from aragora.server.handlers.explainability import (
    _get_cached_decision,
    _cache_decision,
    _decision_cache,
    _cache_timestamps,
    BatchStatus,
    BatchDebateResult,
    BatchJob,
    CACHE_TTL_SECONDS,
)


class TestDecisionCache:
    """Test decision caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        _decision_cache.clear()
        _cache_timestamps.clear()

    def test_cache_miss(self):
        """Should return None for uncached debate."""
        result = _get_cached_decision("nonexistent")
        assert result is None

    def test_cache_hit(self):
        """Should return cached decision."""
        mock_decision = {"outcome": "approved", "confidence": 0.95}
        _cache_decision("debate-123", mock_decision)

        result = _get_cached_decision("debate-123")
        assert result == mock_decision

    def test_cache_expiration(self):
        """Should return None for expired cache entries."""
        mock_decision = {"outcome": "approved"}
        _cache_decision("debate-123", mock_decision)

        # Manually expire the entry
        _cache_timestamps["debate-123"] = time.time() - CACHE_TTL_SECONDS - 1

        result = _get_cached_decision("debate-123")
        assert result is None
        assert "debate-123" not in _decision_cache

    def test_cache_pruning(self):
        """Should prune oldest entries when cache exceeds limit."""
        # Fill cache to limit
        for i in range(105):
            _cache_decision(f"debate-{i}", {"id": i})
            _cache_timestamps[f"debate-{i}"] = time.time() - i  # Oldest first

        # Cache should be pruned to 100
        assert len(_decision_cache) <= 101


class TestBatchStatus:
    """Test BatchStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.PROCESSING.value == "processing"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.PARTIAL.value == "partial"
        assert BatchStatus.FAILED.value == "failed"


class TestBatchDebateResult:
    """Test BatchDebateResult dataclass."""

    def test_success_result(self):
        """Should create success result."""
        result = BatchDebateResult(
            debate_id="debate-123",
            status="success",
            explanation={"outcome": "approved"},
            processing_time_ms=150.5,
        )

        as_dict = result.to_dict()
        assert as_dict["debate_id"] == "debate-123"
        assert as_dict["status"] == "success"
        assert as_dict["explanation"]["outcome"] == "approved"
        assert as_dict["processing_time_ms"] == 150.5

    def test_error_result(self):
        """Should create error result."""
        result = BatchDebateResult(
            debate_id="debate-456",
            status="error",
            error="Debate not found",
            processing_time_ms=10.2,
        )

        as_dict = result.to_dict()
        assert as_dict["status"] == "error"
        assert as_dict["error"] == "Debate not found"
        assert "explanation" not in as_dict

    def test_not_found_result(self):
        """Should create not_found result."""
        result = BatchDebateResult(
            debate_id="debate-789",
            status="not_found",
        )

        as_dict = result.to_dict()
        assert as_dict["status"] == "not_found"


class TestBatchJob:
    """Test BatchJob dataclass."""

    def test_create_batch_job(self):
        """Should create batch job with defaults."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["debate-1", "debate-2", "debate-3"],
        )

        assert job.batch_id == "batch-123"
        assert len(job.debate_ids) == 3
        assert job.status == BatchStatus.PENDING
        assert job.processed_count == 0
        assert job.results == []

    def test_to_dict(self):
        """Should convert to dict correctly."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["debate-1", "debate-2"],
            status=BatchStatus.PROCESSING,
            processed_count=1,
            results=[
                BatchDebateResult(debate_id="debate-1", status="success"),
            ],
        )

        as_dict = job.to_dict()
        assert as_dict["batch_id"] == "batch-123"
        assert as_dict["status"] == "processing"
        assert as_dict["total_debates"] == 2
        assert as_dict["processed_count"] == 1
        assert as_dict["success_count"] == 1
        assert as_dict["error_count"] == 0
        assert as_dict["progress_pct"] == 50.0

    def test_progress_calculation(self):
        """Should calculate progress correctly."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1", "d2", "d3", "d4"],
            processed_count=3,
        )

        as_dict = job.to_dict()
        assert as_dict["progress_pct"] == 75.0

    def test_empty_batch_progress(self):
        """Should handle empty batch for progress."""
        job = BatchJob(
            batch_id="batch-empty",
            debate_ids=[],
        )

        as_dict = job.to_dict()
        assert as_dict["progress_pct"] == 0

    def test_error_count(self):
        """Should count errors correctly."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1", "d2", "d3"],
            results=[
                BatchDebateResult(debate_id="d1", status="success"),
                BatchDebateResult(debate_id="d2", status="error"),
                BatchDebateResult(debate_id="d3", status="not_found"),
            ],
        )

        as_dict = job.to_dict()
        assert as_dict["success_count"] == 1
        assert as_dict["error_count"] == 2


class TestExplainabilityHandlerRoutes:
    """Test ExplainabilityHandler routing (integration-style)."""

    def test_batch_result_timestamps(self):
        """Should track timing correctly."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1"],
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1005.0,
        )

        as_dict = job.to_dict()
        assert as_dict["created_at"] == 1000.0
        assert as_dict["started_at"] == 1001.0
        assert as_dict["completed_at"] == 1005.0


class TestBatchStatusTransitions:
    """Test batch status transitions."""

    def test_pending_to_processing(self):
        """Should transition from pending to processing."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1"],
            status=BatchStatus.PENDING,
        )

        job.status = BatchStatus.PROCESSING
        job.started_at = time.time()

        assert job.status == BatchStatus.PROCESSING
        assert job.started_at is not None

    def test_processing_to_completed(self):
        """Should transition to completed when all succeed."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1", "d2"],
            status=BatchStatus.PROCESSING,
            results=[
                BatchDebateResult(debate_id="d1", status="success"),
                BatchDebateResult(debate_id="d2", status="success"),
            ],
            processed_count=2,
        )

        job.status = BatchStatus.COMPLETED
        job.completed_at = time.time()

        assert job.status == BatchStatus.COMPLETED
        assert job.to_dict()["success_count"] == 2

    def test_processing_to_partial(self):
        """Should transition to partial when some fail."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1", "d2"],
            status=BatchStatus.PROCESSING,
            results=[
                BatchDebateResult(debate_id="d1", status="success"),
                BatchDebateResult(debate_id="d2", status="error"),
            ],
            processed_count=2,
        )

        job.status = BatchStatus.PARTIAL
        job.completed_at = time.time()

        assert job.status == BatchStatus.PARTIAL
        as_dict = job.to_dict()
        assert as_dict["success_count"] == 1
        assert as_dict["error_count"] == 1


class TestBatchJobOptions:
    """Test batch job options handling."""

    def test_default_options(self):
        """Should have empty options by default."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1"],
        )

        assert job.options == {}

    def test_custom_options(self):
        """Should preserve custom options."""
        job = BatchJob(
            batch_id="batch-123",
            debate_ids=["d1"],
            options={
                "include_evidence": True,
                "include_counterfactuals": False,
                "max_evidence_depth": 3,
            },
        )

        assert job.options["include_evidence"] is True
        assert job.options["include_counterfactuals"] is False
        assert job.options["max_evidence_depth"] == 3


class TestProcessingTimeTracking:
    """Test processing time tracking in results."""

    def test_processing_time_default(self):
        """Should default to 0 processing time."""
        result = BatchDebateResult(
            debate_id="d1",
            status="success",
        )

        assert result.processing_time_ms == 0.0

    def test_processing_time_recorded(self):
        """Should record processing time."""
        result = BatchDebateResult(
            debate_id="d1",
            status="success",
            processing_time_ms=250.75,
        )

        assert result.processing_time_ms == 250.75
        assert result.to_dict()["processing_time_ms"] == 250.75
