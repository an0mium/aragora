"""Tests for response formatting utilities (aragora/server/handlers/debates/response_formatting.py).

Covers all public functions and constants:
- STATUS_MAP / STATUS_REVERSE_MAP constants
- CACHE_TTL_* constants
- normalize_status() -- forward mapping
- denormalize_status() -- reverse mapping
- normalize_debate_response() -- full debate dict normalization
  - None input
  - Status normalization
  - Missing status default
  - debate_id / id aliasing
  - consensus_proof promotion
  - consensus_reached / confidence helpers
  - rounds_used derivation
  - duration_seconds default
  - confidence <-> agreement aliasing
  - conclusion <-> final_answer aliasing
  - Edge cases (empty dicts, nested None values, mixed fields)
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from aragora.server.handlers.debates.response_formatting import (
    CACHE_TTL_CONVERGENCE,
    CACHE_TTL_DEBATES_LIST,
    CACHE_TTL_IMPASSE,
    CACHE_TTL_SEARCH,
    STATUS_MAP,
    STATUS_REVERSE_MAP,
    denormalize_status,
    normalize_debate_response,
    normalize_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestConstants:
    """Verify module-level constant values."""

    def test_status_map_active(self):
        assert STATUS_MAP["active"] == "running"

    def test_status_map_concluded(self):
        assert STATUS_MAP["concluded"] == "completed"

    def test_status_map_archived(self):
        assert STATUS_MAP["archived"] == "completed"

    def test_status_map_starting(self):
        assert STATUS_MAP["starting"] == "created"

    def test_status_map_in_progress(self):
        assert STATUS_MAP["in_progress"] == "running"

    def test_status_reverse_map_running(self):
        assert STATUS_REVERSE_MAP["running"] == "active"

    def test_status_reverse_map_completed(self):
        assert STATUS_REVERSE_MAP["completed"] == "concluded"

    def test_status_reverse_map_pending(self):
        assert STATUS_REVERSE_MAP["pending"] == "active"

    def test_status_reverse_map_created(self):
        assert STATUS_REVERSE_MAP["created"] == "active"

    def test_status_reverse_map_in_progress(self):
        assert STATUS_REVERSE_MAP["in_progress"] == "active"

    def test_cache_ttl_debates_list(self):
        assert CACHE_TTL_DEBATES_LIST == 30

    def test_cache_ttl_search(self):
        assert CACHE_TTL_SEARCH == 60

    def test_cache_ttl_convergence(self):
        assert CACHE_TTL_CONVERGENCE == 120

    def test_cache_ttl_impasse(self):
        assert CACHE_TTL_IMPASSE == 120


# ===========================================================================
# normalize_status Tests
# ===========================================================================


class TestNormalizeStatus:
    """Tests for normalize_status()."""

    def test_active_becomes_running(self):
        assert normalize_status("active") == "running"

    def test_concluded_becomes_completed(self):
        assert normalize_status("concluded") == "completed"

    def test_archived_becomes_completed(self):
        assert normalize_status("archived") == "completed"

    def test_starting_becomes_created(self):
        assert normalize_status("starting") == "created"

    def test_in_progress_becomes_running(self):
        assert normalize_status("in_progress") == "running"

    def test_paused_passthrough(self):
        """Status not in map should pass through unchanged."""
        assert normalize_status("paused") == "paused"

    def test_failed_passthrough(self):
        assert normalize_status("failed") == "failed"

    def test_cancelled_passthrough(self):
        assert normalize_status("cancelled") == "cancelled"

    def test_unknown_passthrough(self):
        assert normalize_status("some_unknown_status") == "some_unknown_status"

    def test_empty_string_passthrough(self):
        assert normalize_status("") == ""


# ===========================================================================
# denormalize_status Tests
# ===========================================================================


class TestDenormalizeStatus:
    """Tests for denormalize_status()."""

    def test_running_becomes_active(self):
        assert denormalize_status("running") == "active"

    def test_completed_becomes_concluded(self):
        assert denormalize_status("completed") == "concluded"

    def test_pending_becomes_active(self):
        assert denormalize_status("pending") == "active"

    def test_created_becomes_active(self):
        assert denormalize_status("created") == "active"

    def test_in_progress_becomes_active(self):
        assert denormalize_status("in_progress") == "active"

    def test_paused_passthrough(self):
        assert denormalize_status("paused") == "paused"

    def test_failed_passthrough(self):
        assert denormalize_status("failed") == "failed"

    def test_cancelled_passthrough(self):
        assert denormalize_status("cancelled") == "cancelled"

    def test_unknown_passthrough(self):
        assert denormalize_status("xyz") == "xyz"

    def test_empty_string_passthrough(self):
        assert denormalize_status("") == ""

    def test_roundtrip_active(self):
        """normalize then denormalize should round-trip for mapped values."""
        assert denormalize_status(normalize_status("active")) == "active"

    def test_roundtrip_concluded(self):
        assert denormalize_status(normalize_status("concluded")) == "concluded"


# ===========================================================================
# normalize_debate_response Tests
# ===========================================================================


class TestNormalizeDebateResponseNone:
    """Tests for None and empty-dict inputs."""

    def test_none_returns_none(self):
        assert normalize_debate_response(None) is None

    def test_empty_dict_gets_defaults(self):
        result = normalize_debate_response({})
        assert result is not None
        # Missing status defaults to "completed"
        assert result["status"] == "completed"
        # rounds_used defaults to 0
        assert result["rounds_used"] == 0
        # duration_seconds defaults to 0
        assert result["duration_seconds"] == 0
        # consensus_reached defaults to False
        assert result["consensus_reached"] is False


class TestNormalizeDebateResponseStatus:
    """Tests for status normalization within debate dicts."""

    def test_active_status_normalized(self):
        debate = {"status": "active"}
        result = normalize_debate_response(debate)
        assert result["status"] == "running"

    def test_concluded_status_normalized(self):
        debate = {"status": "concluded"}
        result = normalize_debate_response(debate)
        assert result["status"] == "completed"

    def test_archived_status_normalized(self):
        debate = {"status": "archived"}
        result = normalize_debate_response(debate)
        assert result["status"] == "completed"

    def test_paused_status_passthrough(self):
        debate = {"status": "paused"}
        result = normalize_debate_response(debate)
        assert result["status"] == "paused"

    def test_missing_status_defaults_to_completed(self):
        debate = {"id": "d1"}
        result = normalize_debate_response(debate)
        assert result["status"] == "completed"


class TestNormalizeDebateResponseIdAliasing:
    """Tests for debate_id / id aliasing."""

    def test_debate_id_creates_id_alias(self):
        debate = {"debate_id": "abc-123"}
        result = normalize_debate_response(debate)
        assert result["id"] == "abc-123"
        assert result["debate_id"] == "abc-123"

    def test_id_creates_debate_id_alias(self):
        debate = {"id": "xyz-789"}
        result = normalize_debate_response(debate)
        assert result["debate_id"] == "xyz-789"
        assert result["id"] == "xyz-789"

    def test_both_id_fields_present_unchanged(self):
        debate = {"id": "id1", "debate_id": "did1"}
        result = normalize_debate_response(debate)
        assert result["id"] == "id1"
        assert result["debate_id"] == "did1"

    def test_neither_id_field_no_error(self):
        debate = {"status": "active"}
        result = normalize_debate_response(debate)
        assert "id" not in result
        assert "debate_id" not in result


class TestNormalizeDebateResponseConsensusProofPromotion:
    """Tests for promoting consensus_proof into consensus."""

    def test_consensus_proof_promoted(self):
        debate = {
            "consensus_proof": {
                "reached": True,
                "confidence": 0.85,
                "final_answer": "Use microservices",
                "vote_breakdown": {"agent-a": True, "agent-b": True, "agent-c": False},
            }
        }
        result = normalize_debate_response(debate)
        consensus = result["consensus"]
        assert consensus["reached"] is True
        assert consensus["confidence"] == 0.85
        assert consensus["agreement"] == 0.85
        assert consensus["final_answer"] == "Use microservices"
        assert consensus["conclusion"] == "Use microservices"
        assert "agent-a" in consensus["supporting_agents"]
        assert "agent-b" in consensus["supporting_agents"]
        assert "agent-c" in consensus["dissenting_agents"]

    def test_consensus_proof_not_promoted_when_consensus_exists(self):
        """If consensus already present, consensus_proof should not overwrite it."""
        debate = {
            "consensus": {"reached": True, "confidence": 0.9},
            "consensus_proof": {"reached": False, "confidence": 0.1},
        }
        result = normalize_debate_response(debate)
        assert result["consensus"]["reached"] is True
        assert result["consensus"]["confidence"] == 0.9

    def test_consensus_proof_empty_vote_breakdown(self):
        debate = {
            "consensus_proof": {
                "reached": False,
                "confidence": 0.3,
                "final_answer": None,
                "vote_breakdown": {},
            }
        }
        result = normalize_debate_response(debate)
        consensus = result["consensus"]
        assert consensus["reached"] is False
        assert consensus["supporting_agents"] == []
        assert consensus["dissenting_agents"] == []

    def test_consensus_proof_none_value(self):
        """consensus_proof is None (falsy) -- should still create empty consensus."""
        debate = {"consensus_proof": None}
        result = normalize_debate_response(debate)
        consensus = result["consensus"]
        assert consensus["reached"] is False
        assert consensus["supporting_agents"] == []
        assert consensus["dissenting_agents"] == []

    def test_consensus_proof_missing_vote_breakdown(self):
        """consensus_proof without vote_breakdown key."""
        debate = {
            "consensus_proof": {
                "reached": True,
                "confidence": 0.75,
                "final_answer": "Go with option A",
            }
        }
        result = normalize_debate_response(debate)
        consensus = result["consensus"]
        assert consensus["reached"] is True
        assert consensus["supporting_agents"] == []
        assert consensus["dissenting_agents"] == []


class TestNormalizeDebateResponseConsensusReached:
    """Tests for consensus_reached helper field."""

    def test_consensus_reached_from_consensus(self):
        debate = {"consensus": {"reached": True}}
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is True

    def test_consensus_not_reached_from_consensus(self):
        debate = {"consensus": {"reached": False}}
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is False

    def test_consensus_reached_already_present(self):
        """Pre-existing consensus_reached should not be overwritten."""
        debate = {"consensus_reached": True, "consensus": {"reached": False}}
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is True

    def test_consensus_reached_no_consensus(self):
        debate = {}
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is False

    def test_consensus_reached_consensus_none(self):
        """consensus key set to None."""
        debate = {"consensus": None}
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is False


class TestNormalizeDebateResponseConfidence:
    """Tests for top-level confidence extraction from consensus."""

    def test_confidence_from_consensus_confidence(self):
        debate = {"consensus": {"confidence": 0.92}}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.92

    def test_confidence_from_consensus_agreement(self):
        """If consensus has agreement but not confidence, use agreement."""
        debate = {"consensus": {"agreement": 0.88}}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.88

    def test_confidence_already_present(self):
        debate = {"confidence": 0.5, "consensus": {"confidence": 0.9}}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.5

    def test_confidence_none_in_consensus(self):
        debate = {"consensus": {"confidence": None, "agreement": None}}
        result = normalize_debate_response(debate)
        assert "confidence" not in result

    def test_confidence_missing_from_consensus(self):
        debate = {"consensus": {"reached": True}}
        result = normalize_debate_response(debate)
        # Neither confidence nor agreement in consensus, so no top-level confidence added
        assert "confidence" not in result


class TestNormalizeDebateResponseRoundsUsed:
    """Tests for rounds_used derivation."""

    def test_rounds_used_already_present(self):
        debate = {"rounds_used": 5, "rounds": 3}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 5

    def test_rounds_used_from_int_rounds(self):
        debate = {"rounds": 4}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 4

    def test_rounds_used_from_list_rounds(self):
        debate = {"rounds": [{"round": 1}, {"round": 2}, {"round": 3}]}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 3

    def test_rounds_used_from_empty_list(self):
        debate = {"rounds": []}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 0

    def test_rounds_used_from_string_rounds(self):
        """Non-int, non-list rounds should yield 0."""
        debate = {"rounds": "three"}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 0

    def test_rounds_used_from_none_rounds(self):
        debate = {"rounds": None}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 0

    def test_rounds_used_missing_rounds(self):
        debate = {}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 0


class TestNormalizeDebateResponseDurationSeconds:
    """Tests for duration_seconds default."""

    def test_duration_seconds_default(self):
        debate = {}
        result = normalize_debate_response(debate)
        assert result["duration_seconds"] == 0

    def test_duration_seconds_preserved(self):
        debate = {"duration_seconds": 42}
        result = normalize_debate_response(debate)
        assert result["duration_seconds"] == 42


class TestNormalizeDebateResponseFieldAliases:
    """Tests for confidence/agreement and conclusion/final_answer aliasing."""

    def test_confidence_creates_agreement_alias(self):
        debate = {"confidence": 0.77}
        result = normalize_debate_response(debate)
        assert result["agreement"] == 0.77

    def test_agreement_creates_confidence_alias(self):
        debate = {"agreement": 0.65}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.65

    def test_both_confidence_and_agreement_present(self):
        """If both exist, neither overwrites the other."""
        debate = {"confidence": 0.8, "agreement": 0.7}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.8
        assert result["agreement"] == 0.7

    def test_conclusion_creates_final_answer_alias(self):
        debate = {"conclusion": "Go with option B"}
        result = normalize_debate_response(debate)
        assert result["final_answer"] == "Go with option B"

    def test_final_answer_creates_conclusion_alias(self):
        debate = {"final_answer": "Use monolith"}
        result = normalize_debate_response(debate)
        assert result["conclusion"] == "Use monolith"

    def test_both_conclusion_and_final_answer_present(self):
        debate = {"conclusion": "A", "final_answer": "B"}
        result = normalize_debate_response(debate)
        assert result["conclusion"] == "A"
        assert result["final_answer"] == "B"

    def test_neither_conclusion_nor_final_answer(self):
        debate = {}
        result = normalize_debate_response(debate)
        assert "conclusion" not in result
        assert "final_answer" not in result


class TestNormalizeDebateResponseIntegration:
    """Integration tests combining multiple normalization behaviors."""

    def test_full_debate_normalization(self):
        """A realistic debate dict with all fields gets fully normalized."""
        debate = {
            "debate_id": "debate-001",
            "status": "active",
            "rounds": 3,
            "duration_seconds": 120,
            "consensus_proof": {
                "reached": True,
                "confidence": 0.91,
                "final_answer": "Adopt Kubernetes",
                "vote_breakdown": {
                    "claude": True,
                    "gpt4": True,
                    "gemini": False,
                },
            },
        }
        result = normalize_debate_response(debate)

        # Status normalized
        assert result["status"] == "running"
        # ID aliased
        assert result["id"] == "debate-001"
        assert result["debate_id"] == "debate-001"
        # consensus promoted from proof
        assert result["consensus"]["reached"] is True
        assert result["consensus"]["confidence"] == 0.91
        assert result["consensus"]["final_answer"] == "Adopt Kubernetes"
        assert set(result["consensus"]["supporting_agents"]) == {"claude", "gpt4"}
        assert result["consensus"]["dissenting_agents"] == ["gemini"]
        # consensus_reached helper
        assert result["consensus_reached"] is True
        # confidence propagated up
        assert result["confidence"] == 0.91
        # agreement aliased from confidence
        assert result["agreement"] == 0.91
        # rounds_used
        assert result["rounds_used"] == 3
        # duration preserved
        assert result["duration_seconds"] == 120

    def test_mutates_input_dict(self):
        """normalize_debate_response mutates the input dict in-place."""
        debate = {"status": "active"}
        result = normalize_debate_response(debate)
        assert result is debate
        assert debate["status"] == "running"

    def test_minimal_debate(self):
        """Minimal debate with only an ID."""
        debate = {"id": "min-1"}
        result = normalize_debate_response(debate)
        assert result["status"] == "completed"
        assert result["debate_id"] == "min-1"
        assert result["rounds_used"] == 0
        assert result["duration_seconds"] == 0
        assert result["consensus_reached"] is False

    def test_consensus_with_agreement_field(self):
        """consensus dict has agreement (not confidence) -- should propagate."""
        debate = {
            "consensus": {
                "reached": True,
                "agreement": 0.95,
                "conclusion": "Final decision",
            }
        }
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is True
        # confidence extracted from consensus.agreement
        assert result["confidence"] == 0.95
        # top-level aliases
        assert result["agreement"] == 0.95

    def test_zero_confidence_propagated(self):
        """A confidence of 0 should still be set (not treated as falsy)."""
        debate = {"consensus": {"confidence": 0}}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0

    def test_consensus_proof_with_none_fields(self):
        """consensus_proof with all-None fields."""
        debate = {
            "consensus_proof": {
                "reached": None,
                "confidence": None,
                "final_answer": None,
                "vote_breakdown": None,
            }
        }
        result = normalize_debate_response(debate)
        consensus = result["consensus"]
        assert consensus["reached"] is None
        assert consensus["confidence"] is None
        assert consensus["final_answer"] is None
        assert consensus["supporting_agents"] == []
        assert consensus["dissenting_agents"] == []
