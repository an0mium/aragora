"""
Tests for fork and follow-up debate operations handler.

Tests the ForkOperationsMixin for counterfactual forking,
outcome verification, and crux-based follow-up debates.
"""

import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import tempfile


class TestForkDebate:
    """Tests for counterfactual fork creation."""

    def test_fork_debate_missing_body(self):
        """Should return error when request body is missing."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value=None)
        mixin.get_storage = MagicMock()  # Required by @require_storage

        result = mixin._fork_debate(MagicMock(), "debate_123")

        assert result.status_code == 400
        response = json.loads(result.body.decode())
        assert "Invalid or missing JSON body" in response.get("error", "")

    def test_fork_debate_not_found(self):
        """Should return 404 when debate doesn't exist."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin
        from aragora.server.validation import FORK_REQUEST_SCHEMA

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value={"branch_point": 1})
        mixin.get_storage = MagicMock(
            return_value=MagicMock(get_debate=MagicMock(return_value=None))
        )

        with patch("aragora.server.validation.schema.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = mixin._fork_debate(MagicMock(), "nonexistent_debate")

            assert result.status_code == 404

    def test_fork_debate_invalid_branch_point(self):
        """Should return error when branch point exceeds message count."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value={"branch_point": 100})
        mixin.get_storage = MagicMock(
            return_value=MagicMock(
                get_debate=MagicMock(
                    return_value={"messages": [{"content": "msg1"}, {"content": "msg2"}]}
                )
            )
        )

        with patch("aragora.server.validation.schema.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = mixin._fork_debate(MagicMock(), "debate_123")

            assert result.status_code == 400
            response = json.loads(result.body.decode())
            assert "Branch point" in response.get("error", "")

    def test_fork_debate_success(self):
        """Should successfully create a fork."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            mixin = ForkOperationsMixin()
            mixin.read_json_body = MagicMock(
                return_value={
                    "branch_point": 1,
                    "modified_context": "What if assumption X was true?",
                }
            )
            mixin.get_storage = MagicMock(
                return_value=MagicMock(
                    get_debate=MagicMock(
                        return_value={
                            "messages": [
                                {"content": "msg1", "agent": "claude"},
                                {"content": "msg2", "agent": "gpt4"},
                            ]
                        }
                    )
                )
            )
            mixin.get_nomic_dir = MagicMock(return_value=Path(tmpdir))

            with patch("aragora.server.validation.schema.validate_against_schema") as mock_validate:
                mock_validate.return_value = MagicMock(is_valid=True)

                result = mixin._fork_debate(MagicMock(), "debate_123")

                assert result.status_code == 200
                response = json.loads(result.body.decode())
                assert response["success"] is True
                assert "branch_id" in response
                assert response["parent_debate_id"] == "debate_123"
                assert response["branch_point"] == 1
                assert response["modified_context"] == "What if assumption X was true?"


class TestVerifyOutcome:
    """Tests for outcome verification."""

    def test_verify_outcome_missing_body(self):
        """Should return error when request body is missing."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value=None)

        result = mixin._verify_outcome(MagicMock(), "debate_123")

        assert result.status_code == 400

    def test_verify_outcome_with_position_tracker(self):
        """Should verify outcome using position tracker."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(
            return_value={"correct": True, "source": "manual_verification"}
        )

        mock_tracker = MagicMock()
        mixin.ctx = {"position_tracker": mock_tracker}

        result = mixin._verify_outcome(MagicMock(), "debate_123")

        assert result.status_code == 200
        response = json.loads(result.body.decode())
        assert response["status"] == "verified"
        assert response["correct"] is True
        mock_tracker.record_verification.assert_called_once_with(
            "debate_123", True, "manual_verification"
        )

    def test_verify_outcome_no_tracker_configured(self):
        """Should return error when position tracking not configured."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value={"correct": True})
        mixin.ctx = {}
        mixin.get_nomic_dir = MagicMock(return_value=None)

        result = mixin._verify_outcome(MagicMock(), "debate_123")

        assert result.status_code == 503


class TestFollowupSuggestions:
    """Tests for follow-up suggestions."""

    def test_get_followup_suggestions_debate_not_found(self):
        """Should return 404 when debate doesn't exist."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.get_storage = MagicMock(
            return_value=MagicMock(get_debate=MagicMock(return_value=None))
        )

        result = mixin._get_followup_suggestions("nonexistent")

        assert result.status_code == 404

    def test_get_followup_suggestions_no_cruxes(self):
        """Should return empty suggestions when no cruxes identified."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.get_storage = MagicMock(
            return_value=MagicMock(
                get_debate=MagicMock(
                    return_value={
                        "messages": [],
                        "votes": [],
                        "proposals": {},
                        "uncertainty_metrics": {},
                    }
                )
            )
        )

        with patch("aragora.uncertainty.estimator.DisagreementAnalyzer") as MockAnalyzer:
            mock_analyzer = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.cruxes = []
            mock_analyzer.analyze_disagreement.return_value = mock_metrics
            MockAnalyzer.return_value = mock_analyzer

            result = mixin._get_followup_suggestions("debate_123")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            assert response["suggestions"] == []
            assert "No significant disagreement" in response.get("message", "")


class TestCreateFollowupDebate:
    """Tests for creating follow-up debates."""

    def test_create_followup_missing_body(self):
        """Should return error when request body is missing."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value=None)
        mixin.get_storage = MagicMock()  # Required by @require_storage

        result = mixin._create_followup_debate(MagicMock(), "debate_123")

        assert result.status_code == 400

    def test_create_followup_missing_crux_and_task(self):
        """Should return error when neither crux_id nor task provided."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value={})
        mixin.get_storage = MagicMock()

        result = mixin._create_followup_debate(MagicMock(), "debate_123")

        assert result.status_code == 400
        response = json.loads(result.body.decode())
        assert "Either crux_id or task is required" in response.get("error", "")

    def test_create_followup_parent_not_found(self):
        """Should return 404 when parent debate doesn't exist."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value={"task": "Explore X"})
        mixin.get_storage = MagicMock(
            return_value=MagicMock(get_debate=MagicMock(return_value=None))
        )

        result = mixin._create_followup_debate(MagicMock(), "nonexistent")

        assert result.status_code == 404

    def test_create_followup_with_custom_task(self):
        """Should create follow-up with custom task."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        with tempfile.TemporaryDirectory() as tmpdir:
            mixin = ForkOperationsMixin()
            mixin.read_json_body = MagicMock(return_value={"task": "Custom follow-up task"})
            mixin.get_storage = MagicMock(
                return_value=MagicMock(
                    get_debate=MagicMock(
                        return_value={
                            "agents": ["claude", "gpt4", "gemini"],
                            "uncertainty_metrics": {},
                        }
                    )
                )
            )
            mixin.get_nomic_dir = MagicMock(return_value=Path(tmpdir))

            result = mixin._create_followup_debate(MagicMock(), "debate_123")

            assert result.status_code == 200
            response = json.loads(result.body.decode())
            assert response["success"] is True
            assert "followup_id" in response
            assert response["task"] == "Custom follow-up task"
            assert response["parent_debate_id"] == "debate_123"

    def test_create_followup_crux_not_found(self):
        """Should return 404 when specified crux doesn't exist."""
        from aragora.server.handlers.debates_fork import ForkOperationsMixin

        mixin = ForkOperationsMixin()
        mixin.read_json_body = MagicMock(return_value={"crux_id": "nonexistent"})
        mixin.get_storage = MagicMock(
            return_value=MagicMock(
                get_debate=MagicMock(
                    return_value={
                        "agents": ["claude"],
                        "uncertainty_metrics": {"cruxes": []},
                    }
                )
            )
        )

        result = mixin._create_followup_debate(MagicMock(), "debate_123")

        assert result.status_code == 404
        response = json.loads(result.body.decode())
        assert "Crux not found" in response.get("error", "")


class TestMixinIntegration:
    """Tests for mixin integration with DebatesHandler."""

    def test_fork_methods_available_on_handler(self):
        """Should make fork methods available on handler."""
        from aragora.server.handlers.debates import DebatesHandler

        handler = DebatesHandler({})

        # Check fork methods
        assert hasattr(handler, "_fork_debate")
        assert hasattr(handler, "_verify_outcome")
        assert hasattr(handler, "_get_followup_suggestions")
        assert hasattr(handler, "_create_followup_debate")
