"""Tests for the Decision Package Assembly Endpoint (Task 14D).

Validates that:
- GET /api/v1/debates/{id}/package returns assembled JSON package
- GET /api/v1/debates/{id}/package/markdown returns markdown
- Package includes receipt, explanation, cost, next_steps
- Graceful degradation when receipt/explanation/argument_map unavailable
- 404 for missing debate, 409 for incomplete debate
- Next steps vary by verdict
- can_handle routes correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def handler():
    """Create a DecisionPackageHandler with mock storage."""
    from aragora.server.handlers.debates.decision_package import (
        DecisionPackageHandler,
    )

    storage = MagicMock()
    ctx = {"storage": storage}
    h = DecisionPackageHandler(ctx=ctx)
    return h, storage


def _make_debate(debate_id: str = "test-1", status: str = "completed", **overrides):
    """Create a mock debate dict."""
    debate = {
        "id": debate_id,
        "question": "Should we adopt microservices?",
        "status": status,
        "agents": ["claude", "gpt4"],
        "messages": [],
        "result": {
            "final_answer": "Yes, microservices are recommended.",
            "consensus_reached": True,
            "confidence": 0.85,
            "status": "completed",
            "participants": ["claude", "gpt4"],
            "total_cost_usd": 0.15,
            "per_agent_cost": {"claude": 0.08, "gpt4": 0.07},
            "explanation_summary": "Strong consensus on microservices benefits.",
        },
    }
    debate.update(overrides)
    return debate


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_package_json(self, handler):
        h, _ = handler
        assert h.can_handle("/api/v1/debates/abc123/package") is True

    def test_can_handle_package_markdown(self, handler):
        h, _ = handler
        assert h.can_handle("/api/v1/debates/abc123/package/markdown") is True

    def test_rejects_wrong_path(self, handler):
        h, _ = handler
        assert h.can_handle("/api/v1/debates/abc123/export") is False
        assert h.can_handle("/api/v1/debates") is False
        assert h.can_handle("/api/v1/debates/abc123/package/pdf") is False


class TestPackageJSON:
    """Tests for GET /api/v1/debates/{id}/package."""

    def test_assembles_complete_package(self, handler):
        h, storage = handler
        debate = _make_debate()
        storage.get_debate.return_value = debate

        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "rcpt-123"
        mock_receipt.verdict = "APPROVED"
        mock_receipt.confidence = 0.85
        mock_receipt.risk_level = "LOW"
        mock_receipt.risk_score = 0.1
        mock_receipt.checksum = "abc123"
        mock_receipt.created_at = "2026-02-15T00:00:00Z"

        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_by_gauntlet.return_value = mock_receipt
            mock_get_store.return_value = mock_store

            result = h._handle_json("test-1")

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["debate_id"] == "test-1"
        assert body["verdict"] == "APPROVED"
        assert body["confidence"] == 0.85
        assert body["final_answer"] == "Yes, microservices are recommended."
        assert body["explanation_summary"] == "Strong consensus on microservices benefits."
        assert body["receipt"]["receipt_id"] == "rcpt-123"
        assert body["cost"]["total_cost_usd"] == 0.15
        assert body["cost"]["per_agent_cost"]["claude"] == 0.08
        assert len(body["next_steps"]) > 0
        assert "json" in body["export_formats"]
        assert "markdown" in body["export_formats"]
        assert body["assembled_at"] is not None

    def test_404_for_missing_debate(self, handler):
        h, storage = handler
        storage.get_debate.return_value = None

        result = h._handle_json("nonexistent")
        assert result.status_code == 404

    def test_409_for_incomplete_debate(self, handler):
        h, storage = handler
        storage.get_debate.return_value = _make_debate(status="in_progress")

        result = h._handle_json("test-1")
        assert result.status_code == 409

    def test_graceful_degradation_no_receipt(self, handler):
        h, storage = handler
        storage.get_debate.return_value = _make_debate()

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=ImportError("no receipt store"),
        ):
            result = h._handle_json("test-1")

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["receipt"] is None
        # Verdict should fall back to result-based calculation
        assert body["verdict"] == "APPROVED"
        assert body["confidence"] == 0.85

    def test_graceful_degradation_no_argument_map(self, handler):
        h, storage = handler
        debate = _make_debate()
        debate["messages"] = [
            {"agent": "claude", "content": "I propose...", "role": "proposal", "round": 1}
        ]
        storage.get_debate.return_value = debate

        with (
            patch(
                "aragora.storage.receipt_store.get_receipt_store",
                side_effect=ImportError("no store"),
            ),
            patch(
                "aragora.visualization.mapper.ArgumentCartographer",
                side_effect=ImportError("no mapper"),
            ),
        ):
            result = h._handle_json("test-1")

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["argument_map"] is None


class TestPackageMarkdown:
    """Tests for GET /api/v1/debates/{id}/package/markdown."""

    def test_returns_markdown(self, handler):
        h, storage = handler
        storage.get_debate.return_value = _make_debate()

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=ImportError("no store"),
        ):
            result = h._handle_markdown("test-1")

        assert result.status_code == 200
        assert "text/markdown" in result.content_type
        content = result.body.decode("utf-8")
        assert "# Decision Package:" in content
        assert "microservices" in content.lower()
        assert "## Summary" in content
        assert "## Next Steps" in content

    def test_markdown_404_for_missing_debate(self, handler):
        h, storage = handler
        storage.get_debate.return_value = None

        result = h._handle_markdown("nonexistent")
        assert result.status_code == 404


class TestNextSteps:
    """Tests for _generate_next_steps logic."""

    def test_approved_high_confidence(self):
        from aragora.server.handlers.debates.decision_package import (
            _generate_next_steps,
        )

        steps = _generate_next_steps("APPROVED", 0.9, True, "Test?")
        actions = [s["action"] for s in steps]
        assert any("Proceed" in a for a in actions)

    def test_needs_review_no_consensus(self):
        from aragora.server.handlers.debates.decision_package import (
            _generate_next_steps,
        )

        steps = _generate_next_steps("NEEDS_REVIEW", 0.4, False, "Test?")
        actions = [s["action"] for s in steps]
        assert any("Escalate" in a for a in actions)
        assert any("follow-up debate" in a for a in actions)
        assert any("Low confidence" in a for a in actions)

    def test_approved_with_conditions(self):
        from aragora.server.handlers.debates.decision_package import (
            _generate_next_steps,
        )

        steps = _generate_next_steps("APPROVED_WITH_CONDITIONS", 0.7, True, "Test?")
        actions = [s["action"] for s in steps]
        assert any("conditions" in a.lower() for a in actions)
