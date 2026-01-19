"""Tests for the ranking verification module.

Tests cover:
- calculate_verification_elo_change function
- update_rating_from_verification function
- get_verification_history function
- calculate_verification_impact function
- _validate_agent_name helper
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from aragora.ranking.verification import (
    calculate_verification_elo_change,
    update_rating_from_verification,
    get_verification_history,
    calculate_verification_impact,
    MAX_AGENT_NAME_LENGTH,
    _validate_agent_name,
)


class TestValidateAgentName:
    """Tests for agent name validation."""

    def test_valid_name_passes(self):
        """Valid names should not raise."""
        _validate_agent_name("test-agent")
        _validate_agent_name("a" * MAX_AGENT_NAME_LENGTH)

    def test_too_long_name_fails(self):
        """Names exceeding max length should raise ValueError."""
        long_name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds"):
            _validate_agent_name(long_name)


class TestCalculateVerificationEloChange:
    """Tests for calculate_verification_elo_change function."""

    def test_no_verification_returns_zero(self):
        """Zero counts should return zero change."""
        result = calculate_verification_elo_change(0, 0)
        assert result == 0.0

    def test_verified_only_positive(self):
        """Only verified claims should give positive change."""
        result = calculate_verification_elo_change(2, 0, k_factor=16.0)
        # 2 verified * 16 * 0.5 = 16
        assert result == 16.0

    def test_disproven_only_negative(self):
        """Only disproven claims should give negative change."""
        result = calculate_verification_elo_change(0, 2, k_factor=16.0)
        # -2 disproven * 16 * 0.5 = -16
        assert result == -16.0

    def test_mixed_results(self):
        """Mixed results should net out correctly."""
        result = calculate_verification_elo_change(3, 1, k_factor=16.0)
        # (3 - 1) * 16 * 0.5 = 16
        assert result == 16.0

    def test_custom_k_factor(self):
        """Custom k_factor should be applied."""
        result = calculate_verification_elo_change(1, 0, k_factor=32.0)
        # 1 * 32 * 0.5 = 16
        assert result == 16.0


class TestUpdateRatingFromVerification:
    """Tests for update_rating_from_verification function."""

    def test_updates_overall_elo(self):
        """Should update overall ELO."""
        rating = MagicMock()
        rating.elo = 1500.0
        rating.domain_elos = {}

        update_rating_from_verification(rating, "math", 50.0)

        assert rating.elo == 1550.0

    def test_updates_domain_elo(self):
        """Should update domain-specific ELO."""
        rating = MagicMock()
        rating.elo = 1500.0
        rating.domain_elos = {"math": 1600.0}

        update_rating_from_verification(rating, "math", 50.0)

        assert rating.domain_elos["math"] == 1650.0

    def test_creates_domain_elo_if_missing(self):
        """Should create domain ELO if it doesn't exist."""
        rating = MagicMock()
        rating.elo = 1500.0
        rating.domain_elos = {}

        update_rating_from_verification(rating, "logic", 50.0, default_elo=1500.0)

        assert rating.domain_elos["logic"] == 1550.0

    def test_elo_floor_at_100(self):
        """Should not go below 100 ELO."""
        rating = MagicMock()
        rating.elo = 150.0
        rating.domain_elos = {}

        update_rating_from_verification(rating, "", -100.0)

        assert rating.elo == 100.0

    def test_domain_elo_floor_at_100(self):
        """Domain ELO should not go below 100."""
        rating = MagicMock()
        rating.elo = 500.0
        rating.domain_elos = {"math": 150.0}

        update_rating_from_verification(rating, "math", -100.0)

        assert rating.domain_elos["math"] == 100.0

    def test_updates_timestamp(self):
        """Should update updated_at timestamp."""
        rating = MagicMock()
        rating.elo = 1500.0
        rating.domain_elos = {}
        rating.updated_at = None

        update_rating_from_verification(rating, "", 50.0)

        assert rating.updated_at is not None


class TestGetVerificationHistory:
    """Tests for get_verification_history function."""

    def test_validates_agent_name(self):
        """Should validate agent name."""
        db = MagicMock()
        long_name = "a" * (MAX_AGENT_NAME_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds"):
            get_verification_history(db, long_name)

    def test_queries_database(self):
        """Should query database with correct parameters."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn.cursor.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=None)
        db.connection.return_value = conn

        get_verification_history(db, "test-agent", limit=25)

        db.connection.assert_called_once()
        cursor.execute.assert_called_once()
        # Check the query includes the agent name
        call_args = cursor.execute.call_args[0]
        assert "agent_name = ?" in call_args[0]
        assert "test-agent" in call_args[1]
        assert 25 in call_args[1]

    def test_returns_history_with_changes(self):
        """Should calculate ELO changes from history."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        # Return two rows with ELO progression
        cursor.fetchall.return_value = [
            ("verification:1", 1550.0, "2024-01-01"),
            ("verification:2", 1500.0, "2024-01-02"),
        ]
        conn.cursor.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=None)
        db.connection.return_value = conn

        result = get_verification_history(db, "test-agent")

        # Should return list with change calculations
        assert isinstance(result, list)


class TestCalculateVerificationImpact:
    """Tests for calculate_verification_impact function."""

    def test_returns_summary_dict(self):
        """Should return a summary dictionary."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn.cursor.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=None)
        db.connection.return_value = conn

        result = calculate_verification_impact(db, "test-agent")

        assert "agent_name" in result
        assert result["agent_name"] == "test-agent"
        assert "verification_events" in result
        assert "total_impact" in result
        assert "history" in result

    def test_empty_history_returns_zero_impact(self):
        """Empty history should return zero impact."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn.cursor.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=None)
        db.connection.return_value = conn

        result = calculate_verification_impact(db, "test-agent")

        assert result["verification_events"] == 0
        assert result["total_impact"] == 0


class TestModuleExports:
    """Tests for module exports."""

    def test_exports(self):
        """Module should export expected functions."""
        from aragora.ranking import verification

        assert hasattr(verification, "calculate_verification_elo_change")
        assert hasattr(verification, "update_rating_from_verification")
        assert hasattr(verification, "get_verification_history")
        assert hasattr(verification, "calculate_verification_impact")
