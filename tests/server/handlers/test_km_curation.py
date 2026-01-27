"""
Tests for KnowledgeMound Curation Operations Mixin.

Tests curation management endpoints:
- GET /api/v1/knowledge/mound/curation/policy - Get curation policy
- POST /api/v1/knowledge/mound/curation/policy - Set curation policy
- GET /api/v1/knowledge/mound/curation/status - Get curation status
- POST /api/v1/knowledge/mound/curation/run - Trigger curation run
- GET /api/v1/knowledge/mound/curation/history - Get curation history
- GET /api/v1/knowledge/mound/curation/scores - Get quality scores
- GET /api/v1/knowledge/mound/curation/tiers - Get tier distribution
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Note: These tests cover the sync handler methods directly.
# POST endpoints require mocking of @require_auth and read_json_body.


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockCurationPolicy:
    """Mock curation policy."""

    policy_id: str = "policy_123"
    workspace_id: str = "default"
    enabled: bool = True
    name: str = "default"
    quality_threshold: float = 0.5
    promotion_threshold: float = 0.85
    demotion_threshold: float = 0.35
    archive_threshold: float = 0.2
    refresh_staleness_threshold: float = 0.7
    usage_window_days: int = 30
    min_retrievals_for_promotion: int = 5


class MockTierLevel(Enum):
    """Mock tier level enum."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    GLACIAL = "glacial"


class MockCurationRecommendation(Enum):
    """Mock curation recommendation."""

    PROMOTE = "promote"
    DEMOTE = "demote"
    ARCHIVE = "archive"
    REFRESH = "refresh"
    KEEP = "keep"


@dataclass
class MockQualityScore:
    """Mock quality score."""

    node_id: str = "node_123"
    overall_score: float = 0.75
    freshness_score: float = 0.8
    confidence_score: float = 0.9
    usage_score: float = 0.6
    relevance_score: float = 0.7
    relationship_score: float = 0.85
    recommendation: MockCurationRecommendation = MockCurationRecommendation.KEEP
    debate_uses: int = 5
    retrieval_count: int = 10


@dataclass
class MockCurationResult:
    """Mock curation run result."""

    promoted_count: int = 5
    demoted_count: int = 2
    archived_count: int = 1
    refreshed_count: int = 3
    flagged_count: int = 0


class MockMound:
    """Mock KnowledgeMound for testing."""

    def __init__(self):
        self._policy = MockCurationPolicy()
        self._status = None
        self._history = []
        self._scores = []
        self._tier_distribution = {
            MockTierLevel.HOT.value: 100,
            MockTierLevel.WARM.value: 200,
            MockTierLevel.COLD.value: 300,
            MockTierLevel.GLACIAL.value: 50,
        }

    async def get_curation_policy(self, workspace_id: str) -> MockCurationPolicy:
        return self._policy

    async def set_curation_policy(self, policy: MockCurationPolicy) -> None:
        self._policy = policy

    async def get_curation_status(self, workspace_id: str) -> dict:
        return self._status or {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "is_running": False,
        }

    async def get_node_count(self, workspace_id: str) -> int:
        return 1000

    async def run_curation(
        self, workspace_id: str, dry_run: bool = False, limit: int = 100
    ) -> MockCurationResult:
        return MockCurationResult()

    async def get_curation_history(self, workspace_id: str, limit: int = 20) -> list:
        return self._history or [
            {"id": "run_1", "timestamp": datetime.now(timezone.utc).isoformat()},
            {"id": "run_2", "timestamp": datetime.now(timezone.utc).isoformat()},
        ]

    async def get_quality_scores(
        self,
        workspace_id: str,
        limit: int = 50,
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> list[MockQualityScore]:
        return self._scores or [MockQualityScore(), MockQualityScore(node_id="node_456")]

    async def get_tier_distribution(self, workspace_id: str) -> dict:
        return self._tier_distribution


# ===========================================================================
# Test Handler Setup
# ===========================================================================


def create_handler(mound=None, rbac_denied=False, authenticated=True):
    """Create a handler instance for testing."""
    from aragora.server.handlers.knowledge_base.mound.curation import CurationOperationsMixin

    class TestHandler(CurationOperationsMixin):
        def __init__(self, mound, rbac_denied, authenticated):
            self._mound_instance = mound
            self._rbac_denied = rbac_denied
            self._authenticated = authenticated

        def _get_mound(self):
            return self._mound_instance

        def _check_knowledge_permission(self, handler, action: str = "read"):
            if self._rbac_denied:
                from aragora.server.handlers.base import error_response

                return error_response("Permission denied: knowledge.update required", 403)
            return None

    return TestHandler(mound, rbac_denied, authenticated)


def create_mock_handler_obj(method: str = "GET", body: dict = None) -> MagicMock:
    """Create a mock handler object (simulates the request handler)."""
    handler = MagicMock()
    handler.command = method
    handler.headers = {}
    handler.headers["Authorization"] = "Bearer test-token"
    if body:
        handler._body = json.dumps(body).encode()
    else:
        handler._body = None
    return handler


class MockAuthConfig:
    """Mock auth configuration for @require_auth decorator."""

    api_token = "test-token"

    def validate_token(self, token: str) -> bool:
        return token == "test-token"


def parse_json_response(result) -> tuple[dict[str, Any], int]:
    """Parse handler result into (data, status_code)."""
    if hasattr(result, "body"):
        body = json.loads(result.body) if result.body else {}
        return body, result.status_code
    return {}, 500


# ===========================================================================
# Test: Get Curation Policy
# ===========================================================================


class TestGetCurationPolicy:
    """Tests for GET /api/v1/knowledge/mound/curation/policy."""

    def test_get_policy_success(self):
        """Should return curation policy successfully."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)

        result = handler._handle_get_curation_policy({"workspace_id": "default"})

        data, status = parse_json_response(result)
        assert status == 200
        assert data["workspace_id"] == "default"
        assert "policy" in data
        assert data["policy"]["enabled"] is True
        assert data["policy"]["quality_threshold"] == 0.5

    def test_get_policy_no_custom_policy(self):
        """Should return default policy when no custom policy set."""
        mock_mound = MagicMock()
        # Remove the method to simulate no method existing
        del mock_mound.get_curation_policy
        handler = create_handler(mound=mock_mound)

        with patch(
            "aragora.knowledge.mound.ops.auto_curation.CurationPolicy",
            MockCurationPolicy,
        ):
            result = handler._handle_get_curation_policy({"workspace_id": "test_ws"})

        data, status = parse_json_response(result)
        assert status == 200
        assert "note" in data or "policy" in data

    def test_get_policy_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)

        result = handler._handle_get_curation_policy({})

        data, status = parse_json_response(result)
        assert status == 503

    def test_get_policy_module_unavailable(self):
        """Should return 501 when curation module not available."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)

        with patch.dict("sys.modules", {"aragora.knowledge.mound.ops.auto_curation": None}):
            result = handler._handle_get_curation_policy({})

        data, status = parse_json_response(result)
        assert status == 501


# ===========================================================================
# Test: Set Curation Policy
# ===========================================================================


class TestSetCurationPolicy:
    """Tests for POST /api/v1/knowledge/mound/curation/policy."""

    def test_set_policy_success(self):
        """Should set curation policy successfully."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)
        mock_handler_obj = create_mock_handler_obj(
            "POST",
            {
                "workspace_id": "test_ws",
                "enabled": True,
                "quality_threshold": 0.6,
            },
        )

        with (
            patch("aragora.server.auth.auth_config", MockAuthConfig()),
            patch(
                "aragora.server.handlers.base.read_json_body",
                return_value={"workspace_id": "test_ws", "enabled": True, "quality_threshold": 0.6},
            ),
            patch(
                "aragora.knowledge.mound.ops.auto_curation.CurationPolicy",
                MockCurationPolicy,
            ),
            patch(
                "aragora.server.http_utils.run_async",
                return_value=None,
            ),
        ):
            result = handler._handle_set_curation_policy(mock_handler_obj)

        data, status = parse_json_response(result)
        assert status == 200
        assert data["success"] is True
        assert data["workspace_id"] == "test_ws"

    def test_set_policy_no_body(self):
        """Should return 400 when no JSON body provided."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)
        mock_handler_obj = create_mock_handler_obj("POST")

        with (
            patch("aragora.server.auth.auth_config", MockAuthConfig()),
            patch(
                "aragora.server.handlers.base.read_json_body",
                return_value=None,
            ),
        ):
            result = handler._handle_set_curation_policy(mock_handler_obj)

        data, status = parse_json_response(result)
        assert status == 400

    def test_set_policy_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)
        mock_handler_obj = create_mock_handler_obj("POST")

        with (
            patch("aragora.server.auth.auth_config", MockAuthConfig()),
            patch(
                "aragora.server.handlers.base.read_json_body",
                return_value={"workspace_id": "test"},
            ),
        ):
            result = handler._handle_set_curation_policy(mock_handler_obj)

        data, status = parse_json_response(result)
        assert status == 503


# ===========================================================================
# Test: Curation Status
# ===========================================================================


class TestCurationStatus:
    """Tests for GET /api/v1/knowledge/mound/curation/status."""

    def test_status_success(self):
        """Should return curation status successfully."""
        mock_mound = MagicMock()
        # Remove get_curation_status method so it falls back to default
        del mock_mound.get_curation_status
        del mock_mound.get_node_count
        handler = create_handler(mound=mock_mound)

        result = handler._handle_curation_status({"workspace_id": "default"})

        data, status = parse_json_response(result)
        assert status == 200
        assert data["workspace_id"] == "default"
        assert "stats" in data
        assert "is_running" in data

    def test_status_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)

        result = handler._handle_curation_status({})

        data, status = parse_json_response(result)
        assert status == 503


# ===========================================================================
# Test: Run Curation
# ===========================================================================


class TestRunCuration:
    """Tests for POST /api/v1/knowledge/mound/curation/run."""

    def test_run_curation_success(self):
        """Should run curation successfully."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)
        mock_handler_obj = create_mock_handler_obj("POST")

        with (
            patch("aragora.server.auth.auth_config", MockAuthConfig()),
            patch(
                "aragora.server.handlers.base.read_json_body",
                return_value={"workspace_id": "default", "dry_run": False},
            ),
            patch(
                "aragora.server.http_utils.run_async",
                return_value=MockCurationResult(),
            ),
        ):
            result = handler._handle_run_curation(mock_handler_obj)

        data, status = parse_json_response(result)
        assert status == 200
        assert data["workspace_id"] == "default"
        assert data["promoted"] == 5
        assert data["demoted"] == 2

    def test_run_curation_dry_run(self):
        """Should run curation in dry run mode."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)
        mock_handler_obj = create_mock_handler_obj("POST")

        with (
            patch("aragora.server.auth.auth_config", MockAuthConfig()),
            patch(
                "aragora.server.handlers.base.read_json_body",
                return_value={"workspace_id": "test_ws", "dry_run": True},
            ),
            patch(
                "aragora.server.http_utils.run_async",
                return_value=MockCurationResult(),
            ),
        ):
            result = handler._handle_run_curation(mock_handler_obj)

        data, status = parse_json_response(result)
        assert status == 200
        assert data["dry_run"] is True

    def test_run_curation_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)
        mock_handler_obj = create_mock_handler_obj("POST")

        with (
            patch("aragora.server.auth.auth_config", MockAuthConfig()),
            patch(
                "aragora.server.handlers.base.read_json_body",
                return_value={},
            ),
        ):
            result = handler._handle_run_curation(mock_handler_obj)

        data, status = parse_json_response(result)
        assert status == 503


# ===========================================================================
# Test: Curation History
# ===========================================================================


class TestCurationHistory:
    """Tests for GET /api/v1/knowledge/mound/curation/history."""

    def test_history_success(self):
        """Should return curation history successfully."""
        mock_mound = MagicMock()
        del mock_mound.get_curation_history  # No method, so returns empty
        handler = create_handler(mound=mock_mound)

        result = handler._handle_curation_history({"workspace_id": "default"})

        data, status = parse_json_response(result)
        assert status == 200
        assert data["workspace_id"] == "default"
        assert "history" in data
        assert data["count"] == 0  # Empty because no method

    def test_history_with_limit(self):
        """Should respect limit parameter."""
        mock_mound = MagicMock()
        del mock_mound.get_curation_history
        handler = create_handler(mound=mock_mound)

        result = handler._handle_curation_history({"workspace_id": "default", "limit": "10"})

        data, status = parse_json_response(result)
        assert status == 200

    def test_history_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)

        result = handler._handle_curation_history({})

        data, status = parse_json_response(result)
        assert status == 503


# ===========================================================================
# Test: Quality Scores
# ===========================================================================


class TestQualityScores:
    """Tests for GET /api/v1/knowledge/mound/curation/scores."""

    def test_scores_success(self):
        """Should return quality scores successfully."""
        mock_mound = MagicMock()
        del mock_mound.get_quality_scores  # No method, so returns empty
        handler = create_handler(mound=mock_mound)

        result = handler._handle_quality_scores({"workspace_id": "default"})

        data, status = parse_json_response(result)
        assert status == 200
        assert data["workspace_id"] == "default"
        assert "scores" in data
        assert data["count"] == 0  # Empty because no method

    def test_scores_with_filters(self):
        """Should respect score filter parameters."""
        mock_mound = MagicMock()
        del mock_mound.get_quality_scores
        handler = create_handler(mound=mock_mound)

        result = handler._handle_quality_scores(
            {
                "workspace_id": "default",
                "min_score": "0.5",
                "max_score": "0.9",
                "limit": "25",
            }
        )

        data, status = parse_json_response(result)
        assert status == 200
        assert data["filters"]["min_score"] == 0.5
        assert data["filters"]["max_score"] == 0.9

    def test_scores_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)

        result = handler._handle_quality_scores({})

        data, status = parse_json_response(result)
        assert status == 503


# ===========================================================================
# Test: Tier Distribution
# ===========================================================================


class TestTierDistribution:
    """Tests for GET /api/v1/knowledge/mound/curation/tiers."""

    def test_tiers_success(self):
        """Should return tier distribution successfully."""
        mock_mound = MagicMock()
        del mock_mound.get_tier_distribution  # No method, uses default from TierLevel
        handler = create_handler(mound=mock_mound)

        with patch(
            "aragora.knowledge.mound.ops.auto_curation.TierLevel",
            MockTierLevel,
        ):
            result = handler._handle_tier_distribution({"workspace_id": "default"})

        data, status = parse_json_response(result)
        assert status == 200
        assert data["workspace_id"] == "default"
        assert "distribution" in data
        assert "percentages" in data

    def test_tiers_no_mound(self):
        """Should return 503 when mound not available."""
        handler = create_handler(mound=None)

        result = handler._handle_tier_distribution({})

        data, status = parse_json_response(result)
        assert status == 503

    def test_tiers_module_unavailable(self):
        """Should return 501 when curation module not available."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)

        with patch.dict("sys.modules", {"aragora.knowledge.mound.ops.auto_curation": None}):
            result = handler._handle_tier_distribution({})

        data, status = parse_json_response(result)
        assert status == 501


# ===========================================================================
# Test: Route Handler
# ===========================================================================


class TestCurationRoutes:
    """Tests for _handle_curation_routes dispatcher."""

    def test_routes_to_get_policy(self):
        """Should route GET policy request correctly."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)
        mock_handler_obj = create_mock_handler_obj("GET")

        with patch(
            "aragora.knowledge.mound.ops.auto_curation.CurationPolicy",
            MockCurationPolicy,
        ):
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/policy",
                {"workspace_id": "default"},
                mock_handler_obj,
            )

        assert result is not None
        data, status = parse_json_response(result)
        assert status == 200

    def test_routes_to_status(self):
        """Should route status request correctly."""
        mock_mound = MockMound()
        handler = create_handler(mound=mock_mound)
        mock_handler_obj = create_mock_handler_obj("GET")

        result = handler._handle_curation_routes(
            "/api/v1/knowledge/mound/curation/status",
            {},
            mock_handler_obj,
        )

        assert result is not None
        data, status = parse_json_response(result)
        assert status == 200

    def test_routes_rbac_denied(self):
        """Should return 403 when RBAC denies access."""
        handler = create_handler(mound=MockMound(), rbac_denied=True)
        mock_handler_obj = create_mock_handler_obj("GET")

        result = handler._handle_curation_routes(
            "/api/v1/knowledge/mound/curation/policy",
            {},
            mock_handler_obj,
        )

        data, status = parse_json_response(result)
        assert status == 403

    def test_routes_unknown_path(self):
        """Should return None for unknown paths."""
        handler = create_handler(mound=MockMound())
        mock_handler_obj = create_mock_handler_obj("GET")

        result = handler._handle_curation_routes(
            "/api/v1/knowledge/mound/curation/unknown",
            {},
            mock_handler_obj,
        )

        assert result is None
