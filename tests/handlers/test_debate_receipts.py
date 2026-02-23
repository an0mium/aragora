"""Tests for auto-receipt generation on ALL debates (Task 14A).

Validates that:
- Receipts are generated for non-onboarding debates
- receipt_id appears in status update
- RECEIPT_GENERATED event is emitted
- Receipt generation failure doesn't crash debate
- Receipt includes debate question and verdict
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_debate_config():
    """Create a mock DebateConfig."""
    config = MagicMock()
    config.question = "Should we adopt microservices?"
    config.agents_str = "claude,gpt4"
    config.rounds = 3
    config.metadata = {}
    config.debate_format = "full"
    return config


@pytest.fixture()
def mock_debate_result():
    """Create a mock debate result."""
    result = MagicMock()
    result.final_answer = "Yes, microservices are recommended."
    result.consensus_reached = True
    result.confidence = 0.85
    result.status = "completed"
    result.agent_failures = []
    result.participants = ["claude", "gpt4"]
    result.grounded_verdict = None
    result.total_cost_usd = 0.12
    result.per_agent_cost = {"claude": 0.07, "gpt4": 0.05}
    result.explanation = None
    result.messages = []
    result.winner = None
    return result


@pytest.fixture()
def controller():
    """Create a DebateController with mocked dependencies."""
    from aragora.server.debate_controller import DebateController

    factory = MagicMock()
    emitter = MagicMock()
    controller = DebateController(
        factory=factory,
        emitter=emitter,
        elo_system=None,
        storage=MagicMock(),
    )
    return controller


class TestAutoReceiptGeneration:
    """Tests for receipt generation on all debates."""

    def test_receipt_generated_for_non_onboarding_debate(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """Receipt should be generated for a regular (non-onboarding) debate."""
        mock_debate_config.metadata = {}  # No is_onboarding flag

        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_test123",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=45.0,
            )

            # Verify receipt was saved
            mock_store.save.assert_called_once()
            receipt_dict = mock_store.save.call_args[0][0]
            assert receipt_dict["debate_id"] == "adhoc_test123"
            assert receipt_dict["is_onboarding"] is False

    def test_receipt_generated_for_onboarding_debate(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """Receipt should still be generated for onboarding debates."""
        mock_debate_config.metadata = {"is_onboarding": True}

        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_onboard1",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=30.0,
            )

            mock_store.save.assert_called_once()
            receipt_dict = mock_store.save.call_args[0][0]
            assert receipt_dict["is_onboarding"] is True

    def test_receipt_id_in_status_update(self, controller, mock_debate_config, mock_debate_result):
        """receipt_id should be added to the debate status update."""
        with (
            patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store,
            patch("aragora.server.debate_controller.update_debate_status") as mock_update,
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_test456",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=20.0,
            )

            # The call to update_debate_status should include receipt_id
            calls = mock_update.call_args_list
            assert len(calls) >= 1
            last_call = calls[-1]
            assert "receipt_id" in last_call.kwargs.get("result", {})

    def test_receipt_generated_event_emitted(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """RECEIPT_GENERATED stream event should be emitted."""
        from aragora.events.types import StreamEventType

        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_evt789",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=15.0,
            )

            # Find the RECEIPT_GENERATED emit call
            emit_calls = controller.emitter.emit.call_args_list
            receipt_events = [
                c for c in emit_calls if c.args[0].type == StreamEventType.RECEIPT_GENERATED
            ]
            assert len(receipt_events) == 1
            event_data = receipt_events[0].args[0].data
            assert event_data["debate_id"] == "adhoc_evt789"
            assert "receipt_id" in event_data
            assert "verdict" in event_data

    def test_receipt_failure_does_not_crash_debate(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """Receipt generation failure should not propagate as an exception."""
        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=ImportError("receipt store not available"),
        ):
            # Should not raise
            controller._generate_debate_receipt(
                debate_id="adhoc_fail",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=10.0,
            )

    def test_receipt_includes_question_and_verdict(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """Receipt should include the debate question and a verdict."""
        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_content",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=25.0,
            )

            receipt_dict = mock_store.save.call_args[0][0]
            assert "microservices" in receipt_dict["input_summary"]
            assert receipt_dict["verdict"] in (
                "APPROVED",
                "APPROVED_WITH_CONDITIONS",
                "NEEDS_REVIEW",
            )
            assert receipt_dict["confidence"] == 0.85

    def test_receipt_verdict_approved_high_confidence(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """High confidence + consensus should produce APPROVED verdict."""
        mock_debate_result.confidence = 0.9
        mock_debate_result.consensus_reached = True

        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_approved",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=30.0,
            )

            receipt_dict = mock_store.save.call_args[0][0]
            assert receipt_dict["verdict"] == "APPROVED"
            assert receipt_dict["risk_level"] == "LOW"

    def test_receipt_verdict_needs_review_no_consensus(
        self, controller, mock_debate_config, mock_debate_result
    ):
        """No consensus should produce NEEDS_REVIEW verdict."""
        mock_debate_result.confidence = 0.4
        mock_debate_result.consensus_reached = False

        with patch("aragora.storage.receipt_store.get_receipt_store") as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            controller._generate_debate_receipt(
                debate_id="adhoc_review",
                config=mock_debate_config,
                result=mock_debate_result,
                duration_seconds=30.0,
            )

            receipt_dict = mock_store.save.call_args[0][0]
            assert receipt_dict["verdict"] == "NEEDS_REVIEW"
