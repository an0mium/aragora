"""
Tests for the 'aragora outcome' CLI commands.

Tests cover:
- outcome record command
- outcome search command
- Argument parsing
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from aragora.cli.commands.outcome import _cmd_record, _cmd_search


class TestOutcomeRecord:
    """Tests for the 'outcome record' CLI command."""

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_record_success(self, mock_get_adapter, capsys):
        mock_adapter = MagicMock()
        mock_adapter.ingest.return_value = True
        mock_get_adapter.return_value = mock_adapter

        args = Namespace(
            debate_id="dbt_123",
            type="success",
            description="Vendor delivered on time",
            impact=0.8,
            lessons="Early screening helps",
            tags="vendor,procurement",
            decision_id=None,
        )

        _cmd_record(args)

        mock_adapter.ingest.assert_called_once()
        call_data = mock_adapter.ingest.call_args[0][0]
        assert call_data["debate_id"] == "dbt_123"
        assert call_data["outcome_type"] == "success"
        assert call_data["impact_score"] == 0.8

        captured = capsys.readouterr()
        assert "Outcome recorded" in captured.out

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_record_failure(self, mock_get_adapter, capsys):
        mock_adapter = MagicMock()
        mock_adapter.ingest.return_value = False
        mock_get_adapter.return_value = mock_adapter

        args = Namespace(
            debate_id="dbt_456",
            type="failure",
            description="Project delayed",
            impact=0.3,
            lessons="",
            tags="",
            decision_id=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            _cmd_record(args)
        assert exc_info.value.code == 1

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_record_with_custom_decision_id(self, mock_get_adapter):
        mock_adapter = MagicMock()
        mock_adapter.ingest.return_value = True
        mock_get_adapter.return_value = mock_adapter

        args = Namespace(
            debate_id="dbt_123",
            type="partial",
            description="Partially met goals",
            impact=0.5,
            lessons="",
            tags="",
            decision_id="custom_dec_id",
        )

        _cmd_record(args)

        call_data = mock_adapter.ingest.call_args[0][0]
        assert call_data["decision_id"] == "custom_dec_id"

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_record_clamps_impact(self, mock_get_adapter):
        mock_adapter = MagicMock()
        mock_adapter.ingest.return_value = True
        mock_get_adapter.return_value = mock_adapter

        args = Namespace(
            debate_id="dbt_123",
            type="success",
            description="test",
            impact=1.5,  # Over 1.0
            lessons="",
            tags="",
            decision_id=None,
        )

        _cmd_record(args)

        call_data = mock_adapter.ingest.call_args[0][0]
        assert call_data["impact_score"] == 1.0


class TestOutcomeSearch:
    """Tests for the 'outcome search' CLI command."""

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_search_empty(self, mock_get_adapter, capsys):
        mock_adapter = MagicMock()
        mock_adapter.get_stats.return_value = {
            "outcomes_processed": 0,
            "total_items_ingested": 0,
            "total_errors": 0,
            "mound_connected": False,
        }
        mock_get_adapter.return_value = mock_adapter

        args = Namespace(query="vendor", tags="", type=None, limit=20)
        _cmd_search(args)

        captured = capsys.readouterr()
        assert "0 outcomes" in captured.out
