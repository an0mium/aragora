"""Tests for ContradictionOperationsMixin (aragora/server/handlers/knowledge_base/mound/contradiction.py).

Covers all routes and behavior of the contradiction detection handler mixin:
- POST /api/v1/knowledge/mound/contradictions/detect - Trigger contradiction scan
- GET  /api/v1/knowledge/mound/contradictions         - List unresolved contradictions
- POST /api/v1/knowledge/mound/contradictions/:id/resolve - Resolve a contradiction
- GET  /api/v1/knowledge/mound/contradictions/stats   - Get contradiction statistics

Each method is tested for:
- Success with valid inputs
- Mound not available (503)
- Missing required parameters (400)
- Invalid parameter values (400)
- Internal errors from mound operations (500)
- Edge cases and response structure
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.contradiction import (
    Contradiction,
    ContradictionReport,
    ContradictionType,
    ResolutionStrategy,
)
from aragora.server.handlers.knowledge_base.mound.contradiction import (
    ContradictionOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Mock dataclasses for mound return values
# ---------------------------------------------------------------------------


def _make_contradiction(
    id: str = "ctr-001",
    item_a_id: str = "item-a",
    item_b_id: str = "item-b",
    contradiction_type: ContradictionType = ContradictionType.SEMANTIC,
    similarity_score: float = 0.9,
    conflict_score: float = 0.7,
    resolved: bool = False,
    resolution: ResolutionStrategy | None = None,
    resolved_at: datetime | None = None,
    resolved_by: str | None = None,
    notes: str = "",
) -> Contradiction:
    """Create a Contradiction instance for testing."""
    return Contradiction(
        id=id,
        item_a_id=item_a_id,
        item_b_id=item_b_id,
        contradiction_type=contradiction_type,
        similarity_score=similarity_score,
        conflict_score=conflict_score,
        resolved=resolved,
        resolution=resolution,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        notes=notes,
    )


def _make_report(
    workspace_id: str = "ws-1",
    scanned_items: int = 100,
    contradictions: list[Contradiction] | None = None,
) -> ContradictionReport:
    """Create a ContradictionReport instance for testing."""
    if contradictions is None:
        contradictions = [
            _make_contradiction(id="ctr-001"),
            _make_contradiction(
                id="ctr-002",
                contradiction_type=ContradictionType.NUMERICAL,
                conflict_score=0.9,
                similarity_score=0.95,
            ),
        ]
    return ContradictionReport(
        workspace_id=workspace_id,
        scanned_items=scanned_items,
        contradictions_found=len(contradictions),
        contradictions=contradictions,
        by_type={c.contradiction_type.value: 1 for c in contradictions},
        by_severity={c.severity: 1 for c in contradictions},
        scan_duration_ms=42.5,
    )


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class ContradictionTestHandler(ContradictionOperationsMixin):
    """Concrete handler for testing the contradiction mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with contradiction methods."""
    mound = MagicMock()

    # detect_contradictions returns a ContradictionReport
    mound.detect_contradictions = AsyncMock(return_value=_make_report())

    # get_unresolved_contradictions returns a list of Contradiction objects
    mound.get_unresolved_contradictions = AsyncMock(
        return_value=[
            _make_contradiction(id="ctr-001"),
            _make_contradiction(id="ctr-002", conflict_score=0.9, similarity_score=0.95),
        ]
    )

    # resolve_contradiction returns a resolved Contradiction
    mound.resolve_contradiction = AsyncMock(
        return_value=_make_contradiction(
            id="ctr-001",
            resolved=True,
            resolution=ResolutionStrategy.PREFER_NEWER,
            resolved_by="user-123",
            resolved_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            notes="Resolved via newer data",
        )
    )

    # get_contradiction_stats returns a stats dict (synchronous)
    mound.get_contradiction_stats = MagicMock(
        return_value={
            "total_contradictions": 10,
            "resolved": 3,
            "unresolved": 7,
            "by_type": {"semantic": 5, "numerical": 3, "temporal": 2},
            "by_severity": {"high": 4, "medium": 3, "low": 3},
        }
    )

    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a ContradictionTestHandler with a mocked mound."""
    return ContradictionTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a ContradictionTestHandler with no mound (None)."""
    return ContradictionTestHandler(mound=None)


# ============================================================================
# Tests: detect_contradictions
# ============================================================================


class TestDetectContradictions:
    """Test detect_contradictions (POST /api/knowledge/mound/contradictions/detect)."""

    def test_detect_success(self, handler, mock_mound):
        """Successfully detecting contradictions returns the report."""
        result = _run(
            handler.detect_contradictions(
                workspace_id="ws-1",
                item_ids=["item-a", "item-b"],
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-1"
        assert body["scanned_items"] == 100
        assert body["contradictions_found"] == 2
        assert len(body["contradictions"]) == 2

    def test_detect_calls_mound_with_correct_args(self, handler, mock_mound):
        """Mound.detect_contradictions is called with workspace_id and item_ids."""
        _run(
            handler.detect_contradictions(
                workspace_id="ws-test",
                item_ids=["id-1", "id-2"],
            )
        )
        mock_mound.detect_contradictions.assert_called_once_with(
            workspace_id="ws-test",
            item_ids=["id-1", "id-2"],
        )

    def test_detect_without_item_ids(self, handler, mock_mound):
        """item_ids defaults to None when not provided."""
        _run(handler.detect_contradictions(workspace_id="ws-1"))
        mock_mound.detect_contradictions.assert_called_once_with(
            workspace_id="ws-1",
            item_ids=None,
        )

    def test_detect_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(
            handler_no_mound.detect_contradictions(workspace_id="ws-1")
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_detect_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = _run(handler.detect_contradictions(workspace_id=""))
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    def test_detect_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.detect_contradictions = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        assert _status(result) == 500

    def test_detect_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.detect_contradictions = AsyncMock(side_effect=ValueError("bad data"))
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        assert _status(result) == 500

    def test_detect_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.detect_contradictions = AsyncMock(side_effect=OSError("db fail"))
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        assert _status(result) == 500

    def test_detect_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.detect_contradictions = AsyncMock(side_effect=TypeError("wrong type"))
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        assert _status(result) == 500

    def test_detect_report_contains_by_type(self, handler, mock_mound):
        """Report includes by_type breakdown."""
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        assert "by_type" in body

    def test_detect_report_contains_by_severity(self, handler, mock_mound):
        """Report includes by_severity breakdown."""
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        assert "by_severity" in body

    def test_detect_report_contains_scan_duration(self, handler, mock_mound):
        """Report includes scan_duration_ms."""
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        assert "scan_duration_ms" in body
        assert body["scan_duration_ms"] == 42.5

    def test_detect_empty_report(self, handler, mock_mound):
        """Detection with no contradictions found returns empty report."""
        mock_mound.detect_contradictions = AsyncMock(
            return_value=_make_report(contradictions=[])
        )
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        assert _status(result) == 200
        body = _body(result)
        assert body["contradictions_found"] == 0
        assert body["contradictions"] == []

    def test_detect_with_empty_item_ids_list(self, handler, mock_mound):
        """Passing empty list for item_ids is forwarded to mound."""
        _run(handler.detect_contradictions(workspace_id="ws-1", item_ids=[]))
        mock_mound.detect_contradictions.assert_called_once_with(
            workspace_id="ws-1",
            item_ids=[],
        )

    def test_detect_contradiction_to_dict_called(self, handler, mock_mound):
        """Each contradiction in report is serialized via to_dict."""
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        for c in body["contradictions"]:
            assert "id" in c
            assert "item_a_id" in c
            assert "item_b_id" in c
            assert "contradiction_type" in c
            assert "severity" in c

    def test_detect_report_scanned_at_present(self, handler, mock_mound):
        """Report includes scanned_at timestamp."""
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        assert "scanned_at" in body


# ============================================================================
# Tests: list_contradictions
# ============================================================================


class TestListContradictions:
    """Test list_contradictions (GET /api/knowledge/mound/contradictions)."""

    def test_list_success(self, handler, mock_mound):
        """Successfully listing contradictions returns list with count."""
        result = _run(handler.list_contradictions())
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["contradictions"]) == 2

    def test_list_with_workspace_filter(self, handler, mock_mound):
        """workspace_id filter is forwarded to mound and included in response."""
        result = _run(
            handler.list_contradictions(workspace_id="ws-prod")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-prod"
        mock_mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id="ws-prod",
            min_severity=None,
        )

    def test_list_with_severity_filter(self, handler, mock_mound):
        """min_severity filter is forwarded to mound and included in response."""
        result = _run(
            handler.list_contradictions(min_severity="high")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_severity"] == "high"
        mock_mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id=None,
            min_severity="high",
        )

    def test_list_with_both_filters(self, handler, mock_mound):
        """Both workspace_id and min_severity are forwarded."""
        result = _run(
            handler.list_contradictions(
                workspace_id="ws-1",
                min_severity="critical",
            )
        )
        body = _body(result)
        assert body["workspace_id"] == "ws-1"
        assert body["min_severity"] == "critical"
        mock_mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id="ws-1",
            min_severity="critical",
        )

    def test_list_no_filters(self, handler, mock_mound):
        """No filters defaults to None for both."""
        result = _run(handler.list_contradictions())
        body = _body(result)
        assert body["workspace_id"] is None
        assert body["min_severity"] is None
        mock_mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id=None,
            min_severity=None,
        )

    def test_list_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.list_contradictions())
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_list_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_unresolved_contradictions = AsyncMock(
            side_effect=KeyError("missing")
        )
        result = _run(handler.list_contradictions())
        assert _status(result) == 500

    def test_list_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_unresolved_contradictions = AsyncMock(
            side_effect=ValueError("bad")
        )
        result = _run(handler.list_contradictions())
        assert _status(result) == 500

    def test_list_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_unresolved_contradictions = AsyncMock(
            side_effect=OSError("db fail")
        )
        result = _run(handler.list_contradictions())
        assert _status(result) == 500

    def test_list_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_unresolved_contradictions = AsyncMock(
            side_effect=TypeError("wrong type")
        )
        result = _run(handler.list_contradictions())
        assert _status(result) == 500

    def test_list_empty_results(self, handler, mock_mound):
        """Empty contradiction list returns count=0 and empty list."""
        mock_mound.get_unresolved_contradictions = AsyncMock(return_value=[])
        result = _run(handler.list_contradictions())
        body = _body(result)
        assert body["count"] == 0
        assert body["contradictions"] == []

    def test_list_count_matches_contradictions(self, handler, mock_mound):
        """Count field matches the number of contradictions."""
        result = _run(handler.list_contradictions())
        body = _body(result)
        assert body["count"] == len(body["contradictions"])

    def test_list_contradictions_serialized(self, handler, mock_mound):
        """Each contradiction is serialized via to_dict."""
        result = _run(handler.list_contradictions())
        body = _body(result)
        for c in body["contradictions"]:
            assert "id" in c
            assert "item_a_id" in c
            assert "contradiction_type" in c
            assert "severity" in c
            assert "resolved" in c

    def test_list_single_contradiction(self, handler, mock_mound):
        """Single contradiction in list."""
        mock_mound.get_unresolved_contradictions = AsyncMock(
            return_value=[_make_contradiction(id="ctr-only")]
        )
        result = _run(handler.list_contradictions())
        body = _body(result)
        assert body["count"] == 1
        assert body["contradictions"][0]["id"] == "ctr-only"

    def test_list_severity_values(self, handler, mock_mound):
        """Severity values in response correspond to known levels."""
        valid_severities = {"low", "medium", "high", "critical"}
        result = _run(handler.list_contradictions())
        body = _body(result)
        for c in body["contradictions"]:
            assert c["severity"] in valid_severities


# ============================================================================
# Tests: resolve_contradiction
# ============================================================================


class TestResolveContradiction:
    """Test resolve_contradiction (POST /api/knowledge/mound/contradictions/:id/resolve)."""

    def test_resolve_success(self, handler, mock_mound):
        """Successfully resolving returns success with contradiction data."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
                resolved_by="user-123",
                notes="Resolved via newer data",
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "contradiction" in body
        assert body["contradiction"]["id"] == "ctr-001"
        assert body["contradiction"]["resolved"] is True

    def test_resolve_calls_mound_with_strategy_enum(self, handler, mock_mound):
        """Strategy string is converted to ResolutionStrategy enum before calling mound."""
        _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
                resolved_by="admin",
                notes="test",
            )
        )
        mock_mound.resolve_contradiction.assert_called_once_with(
            contradiction_id="ctr-001",
            strategy=ResolutionStrategy.PREFER_NEWER,
            resolved_by="admin",
            notes="test",
        )

    def test_resolve_prefer_higher_confidence(self, handler, mock_mound):
        """prefer_higher_confidence strategy is accepted."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_higher_confidence",
            )
        )
        assert _status(result) == 200
        mock_mound.resolve_contradiction.assert_called_once()
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["strategy"] == ResolutionStrategy.PREFER_HIGHER_CONFIDENCE

    def test_resolve_merge_strategy(self, handler, mock_mound):
        """merge strategy is accepted."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="merge",
            )
        )
        assert _status(result) == 200
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["strategy"] == ResolutionStrategy.MERGE

    def test_resolve_human_review_strategy(self, handler, mock_mound):
        """human_review strategy is accepted."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="human_review",
            )
        )
        assert _status(result) == 200
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["strategy"] == ResolutionStrategy.HUMAN_REVIEW

    def test_resolve_keep_both_strategy(self, handler, mock_mound):
        """keep_both strategy is accepted."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="keep_both",
            )
        )
        assert _status(result) == 200
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["strategy"] == ResolutionStrategy.KEEP_BOTH

    def test_resolve_prefer_more_sources_strategy(self, handler, mock_mound):
        """prefer_more_sources strategy is accepted."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_more_sources",
            )
        )
        assert _status(result) == 200
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["strategy"] == ResolutionStrategy.PREFER_MORE_SOURCES

    def test_resolve_all_valid_strategies(self, handler, mock_mound):
        """Every valid ResolutionStrategy value is accepted."""
        for strategy in ResolutionStrategy:
            mock_mound.resolve_contradiction = AsyncMock(
                return_value=_make_contradiction(
                    id="ctr-001", resolved=True, resolution=strategy
                )
            )
            result = _run(
                handler.resolve_contradiction(
                    contradiction_id="ctr-001",
                    strategy=strategy.value,
                )
            )
            assert _status(result) == 200, f"Strategy {strategy.value} should be valid"

    def test_resolve_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(
            handler_no_mound.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_resolve_empty_contradiction_id_returns_400(self, handler):
        """Empty contradiction_id returns 400."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 400
        body = _body(result)
        assert "contradiction_id" in body["error"].lower()

    def test_resolve_empty_strategy_returns_400(self, handler):
        """Empty strategy returns 400."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="",
            )
        )
        assert _status(result) == 400
        body = _body(result)
        assert "strategy" in body["error"].lower()

    def test_resolve_invalid_strategy_returns_400(self, handler):
        """Invalid strategy string returns 400 with valid strategies list."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="invalid_strategy",
            )
        )
        assert _status(result) == 400
        body = _body(result)
        assert "invalid strategy" in body["error"].lower() or "must be one of" in body["error"].lower()

    def test_resolve_invalid_strategy_lists_valid_options(self, handler):
        """Invalid strategy error message contains the valid strategy values."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="not_real",
            )
        )
        body = _body(result)
        error_msg = body["error"].lower()
        # Should mention valid strategies
        assert "prefer_newer" in error_msg or "must be one of" in error_msg

    def test_resolve_not_found_returns_404(self, handler, mock_mound):
        """When mound.resolve_contradiction returns None, returns 404."""
        mock_mound.resolve_contradiction = AsyncMock(return_value=None)
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="nonexistent",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_resolve_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.resolve_contradiction = AsyncMock(side_effect=KeyError("missing"))
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 500

    def test_resolve_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.resolve_contradiction = AsyncMock(side_effect=ValueError("bad"))
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 500

    def test_resolve_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.resolve_contradiction = AsyncMock(side_effect=OSError("db fail"))
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 500

    def test_resolve_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.resolve_contradiction = AsyncMock(side_effect=TypeError("wrong"))
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 500

    def test_resolve_default_optional_params(self, handler, mock_mound):
        """Default resolved_by is None and notes is empty string."""
        _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["resolved_by"] is None
        assert call_kwargs["notes"] == ""

    def test_resolve_with_notes(self, handler, mock_mound):
        """Notes are forwarded to mound."""
        _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
                notes="Important resolution note",
            )
        )
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["notes"] == "Important resolution note"

    def test_resolve_with_resolved_by(self, handler, mock_mound):
        """resolved_by is forwarded to mound."""
        _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
                resolved_by="admin-user",
            )
        )
        call_kwargs = mock_mound.resolve_contradiction.call_args.kwargs
        assert call_kwargs["resolved_by"] == "admin-user"

    def test_resolve_response_contains_contradiction_dict(self, handler, mock_mound):
        """Response contains the full contradiction to_dict output."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        body = _body(result)
        contradiction = body["contradiction"]
        assert "id" in contradiction
        assert "item_a_id" in contradiction
        assert "item_b_id" in contradiction
        assert "contradiction_type" in contradiction
        assert "similarity_score" in contradiction
        assert "conflict_score" in contradiction
        assert "severity" in contradiction
        assert "resolved" in contradiction
        assert "resolution" in contradiction

    def test_resolve_response_success_flag(self, handler, mock_mound):
        """Successful resolution always sets success=True."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="merge",
            )
        )
        body = _body(result)
        assert body["success"] is True


# ============================================================================
# Tests: get_contradiction_stats
# ============================================================================


class TestGetContradictionStats:
    """Test get_contradiction_stats (GET /api/knowledge/mound/contradictions/stats)."""

    def test_get_stats_success(self, handler, mock_mound):
        """Successfully getting stats returns stats data."""
        result = _run(handler.get_contradiction_stats())
        assert _status(result) == 200
        body = _body(result)
        assert body["total_contradictions"] == 10
        assert body["resolved"] == 3
        assert body["unresolved"] == 7

    def test_get_stats_contains_by_type(self, handler, mock_mound):
        """Stats include by_type breakdown."""
        result = _run(handler.get_contradiction_stats())
        body = _body(result)
        assert "by_type" in body
        assert body["by_type"]["semantic"] == 5
        assert body["by_type"]["numerical"] == 3
        assert body["by_type"]["temporal"] == 2

    def test_get_stats_contains_by_severity(self, handler, mock_mound):
        """Stats include by_severity breakdown."""
        result = _run(handler.get_contradiction_stats())
        body = _body(result)
        assert "by_severity" in body
        assert body["by_severity"]["high"] == 4
        assert body["by_severity"]["medium"] == 3
        assert body["by_severity"]["low"] == 3

    def test_get_stats_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.get_contradiction_stats())
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_stats_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.get_contradiction_stats = MagicMock(side_effect=KeyError("missing"))
        result = _run(handler.get_contradiction_stats())
        assert _status(result) == 500

    def test_get_stats_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.get_contradiction_stats = MagicMock(side_effect=ValueError("bad"))
        result = _run(handler.get_contradiction_stats())
        assert _status(result) == 500

    def test_get_stats_os_error_returns_500(self, handler, mock_mound):
        """OSError returns 500."""
        mock_mound.get_contradiction_stats = MagicMock(side_effect=OSError("db fail"))
        result = _run(handler.get_contradiction_stats())
        assert _status(result) == 500

    def test_get_stats_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.get_contradiction_stats = MagicMock(side_effect=TypeError("wrong"))
        result = _run(handler.get_contradiction_stats())
        assert _status(result) == 500

    def test_get_stats_is_sync(self, handler, mock_mound):
        """get_contradiction_stats calls mound.get_contradiction_stats (sync, not async)."""
        result = _run(handler.get_contradiction_stats())
        mock_mound.get_contradiction_stats.assert_called_once()
        assert _status(result) == 200

    def test_get_stats_empty_stats(self, handler, mock_mound):
        """Stats with zero values work correctly."""
        mock_mound.get_contradiction_stats = MagicMock(
            return_value={
                "total_contradictions": 0,
                "resolved": 0,
                "unresolved": 0,
                "by_type": {},
                "by_severity": {},
            }
        )
        result = _run(handler.get_contradiction_stats())
        body = _body(result)
        assert body["total_contradictions"] == 0
        assert body["resolved"] == 0
        assert body["unresolved"] == 0
        assert body["by_type"] == {}
        assert body["by_severity"] == {}

    def test_get_stats_custom_fields_passed_through(self, handler, mock_mound):
        """Custom fields in stats are passed through."""
        mock_mound.get_contradiction_stats = MagicMock(
            return_value={"custom_field": "value", "count": 42}
        )
        result = _run(handler.get_contradiction_stats())
        body = _body(result)
        assert body["custom_field"] == "value"
        assert body["count"] == 42


# ============================================================================
# Tests: edge cases and combined scenarios
# ============================================================================


class TestContradictionEdgeCases:
    """Test edge cases across contradiction operations."""

    def test_detect_none_workspace_id_returns_400(self, handler):
        """None workspace_id (falsy) triggers validation and returns 400."""
        # The handler checks `if not workspace_id`, so None should return 400
        result = _run(handler.detect_contradictions(workspace_id=None))
        assert _status(result) == 400

    def test_resolve_none_contradiction_id_returns_400(self, handler):
        """None contradiction_id returns 400."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id=None,
                strategy="prefer_newer",
            )
        )
        assert _status(result) == 400

    def test_resolve_none_strategy_returns_400(self, handler):
        """None strategy returns 400."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy=None,
            )
        )
        assert _status(result) == 400

    def test_detect_report_to_dict_structure(self, handler, mock_mound):
        """Report to_dict produces the expected top-level keys."""
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        expected_keys = {
            "workspace_id",
            "scanned_items",
            "contradictions_found",
            "contradictions",
            "by_type",
            "by_severity",
            "scan_duration_ms",
            "scanned_at",
        }
        assert expected_keys.issubset(set(body.keys()))

    def test_list_response_structure(self, handler, mock_mound):
        """List response has expected keys."""
        result = _run(handler.list_contradictions())
        body = _body(result)
        assert "workspace_id" in body
        assert "min_severity" in body
        assert "count" in body
        assert "contradictions" in body

    def test_resolve_response_structure(self, handler, mock_mound):
        """Resolve response has expected keys."""
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        body = _body(result)
        assert "success" in body
        assert "contradiction" in body

    def test_detect_many_contradictions(self, handler, mock_mound):
        """Report with many contradictions is handled correctly."""
        many = [
            _make_contradiction(id=f"ctr-{i:03d}")
            for i in range(50)
        ]
        mock_mound.detect_contradictions = AsyncMock(
            return_value=_make_report(contradictions=many)
        )
        result = _run(handler.detect_contradictions(workspace_id="ws-1"))
        body = _body(result)
        assert body["contradictions_found"] == 50
        assert len(body["contradictions"]) == 50

    def test_list_many_contradictions(self, handler, mock_mound):
        """Listing many contradictions is handled correctly."""
        many = [
            _make_contradiction(id=f"ctr-{i:03d}")
            for i in range(30)
        ]
        mock_mound.get_unresolved_contradictions = AsyncMock(return_value=many)
        result = _run(handler.list_contradictions())
        body = _body(result)
        assert body["count"] == 30
        assert len(body["contradictions"]) == 30

    def test_resolve_contradiction_to_dict_has_resolution_field(self, handler, mock_mound):
        """Resolved contradiction includes the resolution strategy value."""
        mock_mound.resolve_contradiction = AsyncMock(
            return_value=_make_contradiction(
                id="ctr-001",
                resolved=True,
                resolution=ResolutionStrategy.MERGE,
            )
        )
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="merge",
            )
        )
        body = _body(result)
        assert body["contradiction"]["resolution"] == "merge"

    def test_resolve_contradiction_to_dict_has_notes(self, handler, mock_mound):
        """Resolved contradiction includes notes."""
        mock_mound.resolve_contradiction = AsyncMock(
            return_value=_make_contradiction(
                id="ctr-001",
                resolved=True,
                resolution=ResolutionStrategy.PREFER_NEWER,
                notes="Important note",
            )
        )
        result = _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
                notes="Important note",
            )
        )
        body = _body(result)
        assert body["contradiction"]["notes"] == "Important note"

    def test_detect_workspace_id_forwarded(self, handler, mock_mound):
        """Workspace ID is correctly forwarded to mound."""
        _run(handler.detect_contradictions(workspace_id="custom-ws"))
        mock_mound.detect_contradictions.assert_called_once_with(
            workspace_id="custom-ws",
            item_ids=None,
        )

    def test_list_mound_called_once(self, handler, mock_mound):
        """list_contradictions calls mound exactly once."""
        _run(handler.list_contradictions())
        mock_mound.get_unresolved_contradictions.assert_called_once()

    def test_stats_mound_called_once(self, handler, mock_mound):
        """get_contradiction_stats calls mound exactly once."""
        _run(handler.get_contradiction_stats())
        mock_mound.get_contradiction_stats.assert_called_once()

    def test_detect_mound_called_once(self, handler, mock_mound):
        """detect_contradictions calls mound exactly once."""
        _run(handler.detect_contradictions(workspace_id="ws-1"))
        mock_mound.detect_contradictions.assert_called_once()

    def test_resolve_mound_called_once(self, handler, mock_mound):
        """resolve_contradiction calls mound exactly once."""
        _run(
            handler.resolve_contradiction(
                contradiction_id="ctr-001",
                strategy="prefer_newer",
            )
        )
        mock_mound.resolve_contradiction.assert_called_once()


# ============================================================================
# Tests: routing integration
# ============================================================================


class TestContradictionRouting:
    """Test routing dispatch for contradiction endpoints.

    These tests verify the routing wrappers in routing.py correctly parse
    request bodies/params and call the mixin methods.
    """

    def test_handle_detect_dispatches(self, handler, mock_mound):
        """_handle_detect_contradictions parses body and calls detect."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        body_bytes = json.dumps({
            "workspace_id": "ws-routing",
            "item_ids": ["id-1"],
        }).encode()
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.request.body = body_bytes

        result = RoutingMixin._handle_detect_contradictions(handler, mock_http)
        assert result is not None
        mock_mound.detect_contradictions.assert_called_once()

    def test_handle_list_dispatches(self, handler, mock_mound):
        """_handle_list_contradictions parses query params and calls list."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        result = RoutingMixin._handle_list_contradictions(
            handler,
            {"workspace_id": "ws-q", "min_severity": "high"},
        )
        assert result is not None
        mock_mound.get_unresolved_contradictions.assert_called_once()

    def test_handle_stats_dispatches(self, handler, mock_mound):
        """_handle_contradiction_stats calls get_contradiction_stats."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        result = RoutingMixin._handle_contradiction_stats(handler)
        assert result is not None
        mock_mound.get_contradiction_stats.assert_called_once()

    def test_handle_resolve_dispatches(self, handler, mock_mound):
        """_handle_resolve_contradiction parses body and calls resolve."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        body_bytes = json.dumps({
            "strategy": "prefer_newer",
            "resolved_by": "admin",
            "notes": "test note",
        }).encode()
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.request.body = body_bytes

        result = RoutingMixin._handle_resolve_contradiction(
            handler, "ctr-001", mock_http
        )
        assert result is not None
        mock_mound.resolve_contradiction.assert_called_once()

    def test_handle_resolve_missing_strategy_returns_400(self, handler, mock_mound):
        """_handle_resolve_contradiction returns 400 when strategy is missing."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        body_bytes = json.dumps({
            "resolved_by": "admin",
        }).encode()
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.request.body = body_bytes

        result = RoutingMixin._handle_resolve_contradiction(
            handler, "ctr-001", mock_http
        )
        assert _status(result) == 400
        body = _body(result)
        assert "strategy" in body["error"].lower()
