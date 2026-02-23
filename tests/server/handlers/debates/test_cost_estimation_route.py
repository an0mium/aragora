"""Tests for the cost estimation endpoint route wiring."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.debates.handler import DebatesHandler


@pytest.fixture
def handler():
    h = DebatesHandler(ctx={})
    return h


class TestCostEstimationRoute:
    """Verify /api/v1/debates/estimate-cost is routed correctly."""

    def test_estimate_cost_returns_response(self, handler):
        """GET /api/v1/debates/estimate-cost should return cost data."""
        result = handler.handle(
            "/api/v1/debates/estimate-cost",
            {"num_agents": "3", "num_rounds": "3"},
            None,
        )
        assert result is not None
        assert result.status_code == 200

    def test_estimate_cost_with_models(self, handler):
        """Cost estimate should accept model_types parameter."""
        result = handler.handle(
            "/api/v1/debates/estimate-cost",
            {"num_agents": "2", "num_rounds": "5", "model_types": "claude-sonnet-4,gpt-4o"},
            None,
        )
        assert result is not None
        assert result.status_code == 200
        import json

        data = json.loads(result.body)
        assert "total_estimated_cost_usd" in data
        assert data["num_agents"] == 2
        assert data["num_rounds"] == 5

    def test_estimate_cost_default_params(self, handler):
        """Cost estimate should work with default parameters."""
        result = handler.handle(
            "/api/v1/debates/estimate-cost",
            {},
            None,
        )
        assert result is not None
        assert result.status_code == 200
        import json

        data = json.loads(result.body)
        assert data["num_agents"] == 3  # default
        assert data["num_rounds"] == 9  # default

    def test_estimate_cost_invalid_agents(self, handler):
        """Cost estimate should reject invalid num_agents."""
        result = handler.handle(
            "/api/v1/debates/estimate-cost",
            {"num_agents": "0"},
            None,
        )
        assert result is not None
        assert result.status_code == 400

    def test_estimate_cost_no_auth_required(self, handler):
        """Cost estimation should not require authentication."""
        # Passing None handler means no auth header
        result = handler.handle(
            "/api/v1/debates/estimate-cost",
            {"num_agents": "3", "num_rounds": "3"},
            None,
        )
        assert result is not None
        # Should succeed (200), not return 401
        assert result.status_code == 200


class TestBudgetLimitParsing:
    """Test budget_limit_usd parsing in DebateRequest."""

    def test_parse_budget_limit_from_body(self):
        from aragora.server.debate_controller import DebateRequest

        req = DebateRequest.from_dict(
            {
                "question": "test question",
                "budget_limit_usd": 5.0,
            }
        )
        assert req.budget_limit_usd == 5.0

    def test_parse_budget_limit_capped_at_100(self):
        from aragora.server.debate_controller import DebateRequest

        req = DebateRequest.from_dict(
            {
                "question": "test question",
                "budget_limit_usd": 500,
            }
        )
        assert req.budget_limit_usd == 100.0

    def test_parse_budget_limit_negative_ignored(self):
        from aragora.server.debate_controller import DebateRequest

        req = DebateRequest.from_dict(
            {
                "question": "test question",
                "budget_limit_usd": -1,
            }
        )
        assert req.budget_limit_usd is None

    def test_parse_budget_limit_none_default(self):
        from aragora.server.debate_controller import DebateRequest

        req = DebateRequest.from_dict(
            {
                "question": "test question",
            }
        )
        assert req.budget_limit_usd is None
