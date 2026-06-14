"""
Tests for Nomic loop phase validation.

Tests the PhaseValidator class that ensures safe state transitions
between phases in the Nomic loop.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from aragora.nomic.phases import (
    PhaseValidator,
    PhaseValidationError,
    validate_agents_list,
)


class TestPhaseValidator:
    """Test PhaseValidator class."""

    def test_validate_none_result(self):
        """Test that None result fails validation."""
        is_valid, error = PhaseValidator.validate("context", None)
        assert not is_valid
        assert "None" in error

    def test_validate_non_dict_result(self):
        """Test that non-dict result fails validation."""
        is_valid, error = PhaseValidator.validate("context", "string result")
        assert not is_valid
        assert "not a dict" in error

    def test_validate_missing_required_field(self):
        """Test that missing required field fails validation."""
        result = {"codebase_summary": "test"}  # Missing 'success'
        is_valid, error = PhaseValidator.validate("context", result)
        assert not is_valid
        assert "success" in error

    def test_validate_context_success(self):
        """Test valid context result passes."""
        result = {
            "success": True,
            "codebase_summary": "Summary of codebase",
            "recent_changes": "",
            "open_issues": [],
        }
        is_valid, error = PhaseValidator.validate("context", result)
        assert is_valid
        assert error is None

    def test_validate_debate_consensus_without_improvement(self):
        """Test debate with consensus but no improvement fails."""
        result = {
            "success": True,
            "consensus_reached": True,
            "improvement": "",  # Empty improvement
            "confidence": 0.8,
        }
        is_valid, error = PhaseValidator.validate("debate", result)
        assert not is_valid
        assert "improvement" in error.lower()

    def test_validate_debate_no_consensus_ok(self):
        """Test debate without consensus is valid (even with empty improvement)."""
        result = {
            "success": False,
            "consensus_reached": False,
            "improvement": "",
            "confidence": 0.0,
        }
        is_valid, error = PhaseValidator.validate("debate", result)
        assert is_valid

    def test_validate_or_raise_raises(self):
        """Test validate_or_raise raises PhaseValidationError."""
        with pytest.raises(PhaseValidationError) as exc_info:
            PhaseValidator.validate_or_raise("context", None)

        assert exc_info.value.phase == "context"
        assert exc_info.value.recoverable is True

    def test_normalize_result_fills_defaults(self):
        """Test normalize_result fills in missing fields."""
        result = PhaseValidator.normalize_result("debate", {"success": True})

        assert "consensus_reached" in result
        assert "improvement" in result
        assert "confidence" in result
        assert "votes" in result

    def test_normalize_result_clamps_confidence(self):
        """Test normalize_result clamps confidence to [0,1]."""
        result = PhaseValidator.normalize_result("debate", {"confidence": 1.5})
        assert result["confidence"] == 1.0

        result = PhaseValidator.normalize_result("debate", {"confidence": -0.5})
        assert result["confidence"] == 0.0

    def test_normalize_result_handles_none(self):
        """Test normalize_result handles None input."""
        result = PhaseValidator.normalize_result("context", None)
        assert isinstance(result, dict)
        assert result["success"] is False

    def test_normalize_result_handles_non_dict(self):
        """Test normalize_result handles non-dict input."""
        result = PhaseValidator.normalize_result("context", "error message")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result

    def test_safe_get_handles_none(self):
        """Test safe_get returns default for None result."""
        value = PhaseValidator.safe_get(None, "field", "default")
        assert value == "default"

    def test_safe_get_handles_non_dict(self):
        """Test safe_get returns default for non-dict result."""
        value = PhaseValidator.safe_get("not a dict", "field", "default")
        assert value == "default"


class TestValidateAgentsList:
    """Test validate_agents_list function."""

    def test_none_agents_fails(self):
        """Test None agents list fails."""
        is_valid, error = validate_agents_list(None)
        assert not is_valid
        assert "None" in error

    def test_non_list_fails(self):
        """Test non-list agents fails."""
        is_valid, error = validate_agents_list("not a list")
        assert not is_valid
        assert "not a list" in error

    def test_empty_list_fails(self):
        """Test empty agents list fails (default min_agents=1)."""
        is_valid, error = validate_agents_list([])
        assert not is_valid
        assert "at least 1" in error

    def test_none_agent_fails(self):
        """Test list containing None agent fails."""
        is_valid, error = validate_agents_list([None])
        assert not is_valid
        assert "None" in error

    def test_agent_without_generate_fails(self):
        """Test agent without generate method fails."""

        class BadAgent:
            pass

        is_valid, error = validate_agents_list([BadAgent()])
        assert not is_valid
        assert "generate" in error

    def test_valid_agents_pass(self):
        """Test valid agents list passes."""
        agent = MagicMock()
        agent.generate = MagicMock()

        is_valid, error = validate_agents_list([agent])
        assert is_valid
        assert error == ""

    def test_multiple_valid_agents_pass(self):
        """Test multiple valid agents pass."""
        agents = [MagicMock() for _ in range(3)]
        for a in agents:
            a.generate = MagicMock()

        is_valid, error = validate_agents_list(agents)
        assert is_valid

    def test_min_agents_enforced(self):
        """Test min_agents parameter is enforced."""
        agent = MagicMock()
        agent.generate = MagicMock()

        is_valid, error = validate_agents_list([agent], min_agents=2)
        assert not is_valid
        assert "at least 2" in error


class TestPhaseValidationIntegration:
    """Integration tests for phase validation flow."""

    def test_context_to_debate_flow(self):
        """Test validation of context->debate transition."""
        # Context result
        context_result = {
            "success": True,
            "codebase_summary": "Project has X, Y, Z",
            "recent_changes": "Added feature A",
            "open_issues": [],
        }
        is_valid, _ = PhaseValidator.validate("context", context_result)
        assert is_valid

        # Pass to debate
        debate_result = {
            "success": True,
            "consensus_reached": True,
            "improvement": "Add feature B",
            "confidence": 0.85,
            "votes": [("agent1", "yes"), ("agent2", "yes")],
        }
        is_valid, _ = PhaseValidator.validate("debate", debate_result)
        assert is_valid

    def test_debate_to_design_flow(self):
        """Test validation of debate->design transition."""
        debate_result = {
            "success": True,
            "consensus_reached": True,
            "improvement": "Add caching layer",
            "confidence": 0.9,
        }
        is_valid, _ = PhaseValidator.validate("debate", debate_result)
        assert is_valid

        # Design result
        design_result = {
            "success": True,
            "design": "Create cache.py with LRU cache...",
            "files_affected": ["aragora/cache.py"],
            "complexity_estimate": "medium",
        }
        is_valid, _ = PhaseValidator.validate("design", design_result)
        assert is_valid

    def test_full_cycle_validation(self):
        """Test validation across full cycle."""
        phases = [
            (
                "context",
                {
                    "success": True,
                    "codebase_summary": "Summary",
                },
            ),
            (
                "debate",
                {
                    "success": True,
                    "consensus_reached": True,
                    "improvement": "Improvement",
                    "confidence": 0.8,
                },
            ),
            (
                "design",
                {
                    "success": True,
                    "design": "Design doc",
                    "files_affected": ["file.py"],
                },
            ),
            (
                "implement",
                {
                    "success": True,
                    "files_modified": ["file.py"],
                },
            ),
            (
                "verify",
                {
                    "success": True,
                    "tests_passed": True,
                    "test_output": "All tests passed",
                },
            ),
            (
                "commit",
                {
                    "success": True,
                    "commit_hash": "abc123",
                    "committed": True,
                },
            ),
        ]

        for phase, result in phases:
            is_valid, error = PhaseValidator.validate(phase, result)
            assert is_valid, f"Phase {phase} failed: {error}"
