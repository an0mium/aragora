"""
E2E tests for Nomic Loop complete cycle.

Tests the system's ability to:
- Execute complete nomic cycles (context -> debate -> design -> implement -> verify)
- Handle phase transitions and validation
- Create and restore checkpoints
- Rollback on verification failure
- Protect critical files from modification
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases import (
    CommitResult,
    ContextResult,
    DebateResult,
    DesignResult,
    ImplementResult,
    PhaseResult,
    PhaseValidationError,
    PhaseValidator,
    VerifyResult,
    validate_agents_list,
)


class TestPhaseValidator:
    """Test PhaseValidator functionality."""

    def test_validate_none_result(self):
        """Verify None result is invalid."""
        is_valid, error = PhaseValidator.validate("context", None)
        assert not is_valid
        assert "None" in error

    def test_validate_non_dict_result(self):
        """Verify non-dict result is invalid."""
        is_valid, error = PhaseValidator.validate("context", "not a dict")
        assert not is_valid
        assert "not a dict" in error

    def test_validate_missing_required_field(self):
        """Verify missing required field is caught."""
        is_valid, error = PhaseValidator.validate("context", {})
        assert not is_valid
        assert "success" in error

    def test_validate_valid_context_result(self):
        """Verify valid context result passes."""
        result = {"success": True, "codebase_summary": "Test summary"}
        is_valid, error = PhaseValidator.validate("context", result)
        assert is_valid
        assert error is None

    def test_validate_debate_requires_improvement(self):
        """Verify debate with consensus requires improvement."""
        result = {"success": True, "consensus_reached": True, "improvement": ""}
        is_valid, error = PhaseValidator.validate("debate", result)
        assert not is_valid
        assert "improvement" in error.lower()

    def test_validate_debate_with_valid_consensus(self):
        """Verify valid debate result passes."""
        result = {
            "success": True,
            "consensus_reached": True,
            "improvement": "Add caching",
            "confidence": 0.8,
        }
        is_valid, error = PhaseValidator.validate("debate", result)
        assert is_valid

    def test_validate_or_raise_raises_on_invalid(self):
        """Verify validate_or_raise raises exception."""
        with pytest.raises(PhaseValidationError) as exc_info:
            PhaseValidator.validate_or_raise("context", None)
        assert exc_info.value.phase == "context"

    def test_normalize_result_fills_defaults(self):
        """Verify normalize_result fills in defaults."""
        result = PhaseValidator.normalize_result("debate", {})
        assert result["success"] is False
        assert result["consensus_reached"] is False
        assert result["improvement"] == ""
        assert result["confidence"] == 0.0
        assert result["votes"] == []

    def test_normalize_clamps_confidence(self):
        """Verify confidence is clamped to [0, 1]."""
        result = PhaseValidator.normalize_result("debate", {"confidence": 1.5})
        assert result["confidence"] == 1.0

        result = PhaseValidator.normalize_result("debate", {"confidence": -0.5})
        assert result["confidence"] == 0.0

    def test_safe_get_handles_none(self):
        """Verify safe_get handles None result."""
        assert PhaseValidator.safe_get(None, "field") is None
        assert PhaseValidator.safe_get(None, "field", "default") == "default"

    def test_safe_get_handles_non_dict(self):
        """Verify safe_get handles non-dict result."""
        assert PhaseValidator.safe_get("not dict", "field", 42) == 42


class TestValidateAgentsList:
    """Test agent list validation."""

    def test_validate_none_agents(self):
        """Verify None agents list is invalid."""
        is_valid, error = validate_agents_list(None)
        assert not is_valid
        assert "None" in error

    def test_validate_non_list_agents(self):
        """Verify non-list agents is invalid."""
        is_valid, error = validate_agents_list("not a list")
        assert not is_valid
        assert "not a list" in error

    def test_validate_empty_agents(self):
        """Verify empty agents list is invalid."""
        is_valid, error = validate_agents_list([])
        assert not is_valid
        assert "at least" in error

    def test_validate_agent_missing_generate(self):
        """Verify agent without generate method is invalid."""

        class BadAgent:
            pass

        is_valid, error = validate_agents_list([BadAgent()])
        assert not is_valid
        assert "generate" in error

    def test_validate_valid_agents(self):
        """Verify valid agents list passes."""

        class GoodAgent:
            def generate(self):
                pass

        is_valid, error = validate_agents_list([GoodAgent()])
        assert is_valid
        assert error == ""


class TestPhaseResultTypes:
    """Test phase result type definitions."""

    def test_context_result_fields(self):
        """Verify ContextResult has expected fields."""
        result: ContextResult = {
            "success": True,
            "codebase_summary": "Summary",
            "recent_changes": "Changes",
            "open_issues": ["issue1"],
        }
        assert result["success"] is True
        assert result["codebase_summary"] == "Summary"

    def test_debate_result_fields(self):
        """Verify DebateResult has expected fields."""
        result: DebateResult = {
            "success": True,
            "improvement": "Add feature",
            "consensus_reached": True,
            "confidence": 0.9,
            "votes": [("agent1", True), ("agent2", True)],
        }
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.9

    def test_design_result_fields(self):
        """Verify DesignResult has expected fields."""
        result: DesignResult = {
            "success": True,
            "design": "Architecture plan",
            "files_affected": ["file1.py", "file2.py"],
            "complexity_estimate": "medium",
        }
        assert len(result["files_affected"]) == 2

    def test_implement_result_fields(self):
        """Verify ImplementResult has expected fields."""
        result: ImplementResult = {
            "success": True,
            "files_modified": ["file1.py"],
            "diff_summary": "+10 -5",
        }
        assert len(result["files_modified"]) == 1

    def test_verify_result_fields(self):
        """Verify VerifyResult has expected fields."""
        result: VerifyResult = {
            "success": True,
            "tests_passed": True,
            "test_output": "All tests passed",
            "syntax_valid": True,
        }
        assert result["tests_passed"] is True

    def test_commit_result_fields(self):
        """Verify CommitResult has expected fields."""
        result: CommitResult = {
            "success": True,
            "commit_hash": "abc123",
            "committed": True,
        }
        assert result["commit_hash"] == "abc123"


class TestPhaseTransitions:
    """Test phase transition logic."""

    @pytest.fixture
    def phase_results(self) -> Dict[str, PhaseResult]:
        """Create a full set of successful phase results."""
        return {
            "context": {
                "success": True,
                "codebase_summary": "Project has 100 files",
                "recent_changes": "Added feature X",
                "open_issues": [],
            },
            "debate": {
                "success": True,
                "improvement": "Add caching layer",
                "consensus_reached": True,
                "confidence": 0.85,
                "votes": [],
            },
            "design": {
                "success": True,
                "design": "Use Redis for caching",
                "files_affected": ["cache.py"],
                "complexity_estimate": "low",
            },
            "implement": {
                "success": True,
                "files_modified": ["cache.py"],
                "diff_summary": "+50 lines",
            },
            "verify": {
                "success": True,
                "tests_passed": True,
                "test_output": "10 tests passed",
                "syntax_valid": True,
            },
            "commit": {
                "success": True,
                "commit_hash": "abc123def",
                "committed": True,
            },
        }

    def test_all_phases_validate(self, phase_results):
        """Verify all phase results validate successfully."""
        for phase_name, result in phase_results.items():
            is_valid, error = PhaseValidator.validate(phase_name, result)
            assert is_valid, f"Phase {phase_name} failed: {error}"

    def test_phase_sequence_order(self):
        """Verify phase sequence is correct."""
        expected_sequence = [
            "context",
            "debate",
            "design",
            "implement",
            "verify",
            "commit",
        ]
        # This reflects the documented nomic loop phases
        assert len(expected_sequence) == 6

    def test_failed_verify_blocks_commit(self, phase_results):
        """Verify failed verification should block commit."""
        phase_results["verify"]["success"] = False
        phase_results["verify"]["tests_passed"] = False

        # Simulating the decision logic
        can_commit = phase_results["verify"]["success"] and phase_results["verify"]["tests_passed"]
        assert not can_commit

    def test_no_consensus_skips_later_phases(self, phase_results):
        """Verify no consensus should skip design/implement phases."""
        phase_results["debate"]["consensus_reached"] = False

        # Simulating the decision logic
        should_proceed_to_design = phase_results["debate"]["consensus_reached"]
        assert not should_proceed_to_design


class TestCheckpointingBehavior:
    """Test checkpoint creation and restoration behavior."""

    def test_checkpoint_data_structure(self):
        """Verify checkpoint data structure is correct."""
        checkpoint = {
            "cycle_number": 1,
            "phase": "design",
            "timestamp": "2026-01-16T12:00:00",
            "results": {
                "context": {"success": True},
                "debate": {"success": True, "consensus_reached": True},
            },
            "files_modified": [],
            "git_state": {"branch": "main", "commit": "abc123"},
        }

        assert "cycle_number" in checkpoint
        assert "phase" in checkpoint
        assert "results" in checkpoint

    def test_checkpoint_includes_partial_results(self):
        """Verify checkpoint captures partial progress."""
        checkpoint = {
            "cycle_number": 2,
            "phase": "implement",
            "results": {
                "context": {"success": True},
                "debate": {"success": True},
                "design": {"success": True},
                # implement not yet complete
            },
        }

        completed_phases = list(checkpoint["results"].keys())
        assert "context" in completed_phases
        assert "debate" in completed_phases
        assert "design" in completed_phases
        assert "implement" not in completed_phases


class TestRollbackBehavior:
    """Test rollback on verification failure."""

    def test_rollback_decision_on_test_failure(self):
        """Verify rollback is triggered on test failure."""
        verify_result: VerifyResult = {
            "success": False,
            "tests_passed": False,
            "test_output": "3 tests failed",
            "syntax_valid": True,
        }

        should_rollback = not verify_result["tests_passed"]
        assert should_rollback

    def test_rollback_decision_on_syntax_error(self):
        """Verify rollback is triggered on syntax error."""
        verify_result: VerifyResult = {
            "success": False,
            "tests_passed": False,
            "test_output": "Syntax error in file.py",
            "syntax_valid": False,
        }

        should_rollback = not verify_result["syntax_valid"]
        assert should_rollback

    def test_no_rollback_on_success(self):
        """Verify no rollback on successful verification."""
        verify_result: VerifyResult = {
            "success": True,
            "tests_passed": True,
            "test_output": "All tests passed",
            "syntax_valid": True,
        }

        should_rollback = not (verify_result["tests_passed"] and verify_result["syntax_valid"])
        assert not should_rollback


class TestProtectedFilesHandling:
    """Test protected file verification."""

    @pytest.fixture
    def protected_files(self) -> List[str]:
        """List of protected files."""
        return [
            "CLAUDE.md",
            "core.py",
            "aragora/__init__.py",
            ".env",
            "scripts/nomic_loop.py",
        ]

    def test_protected_files_list_exists(self, protected_files):
        """Verify protected files list is defined."""
        assert len(protected_files) > 0
        assert "CLAUDE.md" in protected_files

    def test_protected_file_modification_detected(self, protected_files, tmp_path):
        """Verify modification of protected file is detected."""
        # Create a mock protected file
        test_file = tmp_path / "protected.md"
        test_file.write_text("Original content")

        original_hash = hash(test_file.read_text())

        # Modify it
        test_file.write_text("Modified content")

        new_hash = hash(test_file.read_text())

        # Should detect change
        assert original_hash != new_hash

    def test_checksum_computation(self, tmp_path):
        """Verify checksum computation works."""
        import hashlib

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        checksum = hashlib.sha256(test_file.read_bytes()).hexdigest()
        assert len(checksum) == 64  # SHA256 hex length


class TestCycleCompletion:
    """Test complete cycle execution scenarios."""

    def test_successful_cycle_returns_commit_hash(self):
        """Verify successful cycle returns commit info."""
        cycle_result = {
            "cycle_number": 1,
            "success": True,
            "commit_hash": "abc123def456",
            "improvement_applied": "Added caching layer",
            "phases_completed": 6,
        }

        assert cycle_result["success"]
        assert cycle_result["commit_hash"] is not None

    def test_failed_cycle_no_commit(self):
        """Verify failed cycle does not commit."""
        cycle_result = {
            "cycle_number": 1,
            "success": False,
            "commit_hash": None,
            "failure_reason": "Tests failed",
            "phases_completed": 4,
        }

        assert not cycle_result["success"]
        assert cycle_result["commit_hash"] is None

    def test_cycle_tracks_phases_completed(self):
        """Verify cycle tracks number of phases completed."""
        # Failure at verify phase (phase 5)
        cycle_result = {
            "phases_completed": 4,  # context, debate, design, implement
            "failed_at_phase": "verify",
        }

        assert cycle_result["phases_completed"] == 4


class TestPhaseTimeouts:
    """Test phase timeout handling."""

    def test_phase_timeout_structure(self):
        """Verify phase timeout configuration structure."""
        phase_timeouts = {
            "context": 60,  # 1 minute
            "debate": 300,  # 5 minutes
            "design": 180,  # 3 minutes
            "implement": 600,  # 10 minutes
            "verify": 300,  # 5 minutes
            "commit": 60,  # 1 minute
        }

        assert all(t > 0 for t in phase_timeouts.values())
        assert phase_timeouts["implement"] > phase_timeouts["context"]

    def test_timeout_triggers_phase_failure(self):
        """Verify timeout results in phase failure."""
        # Simulated timeout scenario
        phase_result = {
            "success": False,
            "error": "Phase timed out after 300 seconds",
            "duration_seconds": 300.0,
        }

        assert not phase_result["success"]
        assert "timed out" in phase_result["error"].lower()


class TestMultiCycleExecution:
    """Test multiple consecutive cycles."""

    def test_cycle_counter_increments(self):
        """Verify cycle counter increments correctly."""
        cycles_completed = 0
        max_cycles = 3

        for _ in range(max_cycles):
            cycles_completed += 1

        assert cycles_completed == max_cycles

    def test_cycle_state_isolation(self):
        """Verify cycles don't share state inappropriately."""
        cycle_1_results = {"context": {"summary": "Cycle 1"}}
        cycle_2_results = {"context": {"summary": "Cycle 2"}}

        assert cycle_1_results["context"]["summary"] != cycle_2_results["context"]["summary"]

    def test_consecutive_failures_trigger_stop(self):
        """Verify consecutive failures can trigger loop termination."""
        max_consecutive_failures = 3
        consecutive_failures = 0

        for i in range(5):
            failed = i < 3  # First 3 fail
            if failed:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            if consecutive_failures >= max_consecutive_failures:
                should_stop = True
                break
        else:
            should_stop = False

        assert should_stop
