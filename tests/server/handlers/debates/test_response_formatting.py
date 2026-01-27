"""Tests for response formatting utilities."""

import pytest

from aragora.server.handlers.debates.response_formatting import (
    STATUS_MAP,
    STATUS_REVERSE_MAP,
    CACHE_TTL_DEBATES_LIST,
    CACHE_TTL_SEARCH,
    CACHE_TTL_CONVERGENCE,
    CACHE_TTL_IMPASSE,
    normalize_status,
    denormalize_status,
    normalize_debate_response,
)


# =============================================================================
# Test Status Normalization
# =============================================================================


class TestStatusNormalization:
    """Tests for status normalization functions."""

    def test_normalize_active_to_running(self):
        """Should normalize 'active' to 'running'."""
        assert normalize_status("active") == "running"

    def test_normalize_concluded_to_completed(self):
        """Should normalize 'concluded' to 'completed'."""
        assert normalize_status("concluded") == "completed"

    def test_normalize_archived_to_completed(self):
        """Should normalize 'archived' to 'completed'."""
        assert normalize_status("archived") == "completed"

    def test_normalize_starting_to_created(self):
        """Should normalize 'starting' to 'created'."""
        assert normalize_status("starting") == "created"

    def test_normalize_in_progress_to_running(self):
        """Should normalize 'in_progress' to 'running'."""
        assert normalize_status("in_progress") == "running"

    def test_normalize_unknown_status_unchanged(self):
        """Should leave unknown statuses unchanged."""
        assert normalize_status("paused") == "paused"
        assert normalize_status("custom_status") == "custom_status"

    def test_denormalize_running_to_active(self):
        """Should denormalize 'running' to 'active'."""
        assert denormalize_status("running") == "active"

    def test_denormalize_completed_to_concluded(self):
        """Should denormalize 'completed' to 'concluded'."""
        assert denormalize_status("completed") == "concluded"

    def test_denormalize_pending_to_active(self):
        """Should denormalize 'pending' to 'active'."""
        assert denormalize_status("pending") == "active"

    def test_denormalize_unknown_status_unchanged(self):
        """Should leave unknown statuses unchanged."""
        assert denormalize_status("failed") == "failed"


# =============================================================================
# Test Debate Response Normalization
# =============================================================================


class TestNormalizeDebateResponse:
    """Tests for debate response normalization."""

    def test_normalize_none_returns_none(self):
        """Should return None for None input."""
        assert normalize_debate_response(None) is None

    def test_normalize_status_field(self):
        """Should normalize status field."""
        debate = {"status": "active", "id": "123"}
        result = normalize_debate_response(debate)
        assert result["status"] == "running"

    def test_add_default_status_if_missing(self):
        """Should add default 'completed' status if missing."""
        debate = {"id": "123"}
        result = normalize_debate_response(debate)
        assert result["status"] == "completed"

    def test_add_id_alias_from_debate_id(self):
        """Should add 'id' field from 'debate_id'."""
        debate = {"debate_id": "debate-123"}
        result = normalize_debate_response(debate)
        assert result["id"] == "debate-123"

    def test_add_debate_id_alias_from_id(self):
        """Should add 'debate_id' field from 'id'."""
        debate = {"id": "123"}
        result = normalize_debate_response(debate)
        assert result["debate_id"] == "123"

    def test_promote_consensus_proof(self):
        """Should promote consensus_proof to consensus."""
        debate = {
            "id": "123",
            "consensus_proof": {
                "reached": True,
                "confidence": 0.85,
                "final_answer": "Yes, proceed.",
                "vote_breakdown": {"claude": True, "gpt4": True, "gemini": False},
            },
        }
        result = normalize_debate_response(debate)

        assert "consensus" in result
        assert result["consensus"]["reached"] is True
        assert result["consensus"]["confidence"] == 0.85
        assert result["consensus"]["final_answer"] == "Yes, proceed."
        assert "claude" in result["consensus"]["supporting_agents"]
        assert "gemini" in result["consensus"]["dissenting_agents"]

    def test_add_consensus_reached_field(self):
        """Should add consensus_reached from consensus object."""
        debate = {
            "id": "123",
            "consensus": {"reached": True},
        }
        result = normalize_debate_response(debate)
        assert result["consensus_reached"] is True

    def test_add_confidence_from_consensus(self):
        """Should add confidence from consensus object."""
        debate = {
            "id": "123",
            "consensus": {"confidence": 0.9},
        }
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.9

    def test_add_confidence_from_agreement(self):
        """Should add confidence from consensus agreement field."""
        debate = {
            "id": "123",
            "consensus": {"agreement": 0.75},
        }
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.75

    def test_rounds_used_from_int(self):
        """Should set rounds_used from int rounds."""
        debate = {"id": "123", "rounds": 5}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 5

    def test_rounds_used_from_list(self):
        """Should set rounds_used from list length."""
        debate = {"id": "123", "rounds": [{}, {}, {}]}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 3

    def test_rounds_used_default_zero(self):
        """Should default rounds_used to 0."""
        debate = {"id": "123"}
        result = normalize_debate_response(debate)
        assert result["rounds_used"] == 0

    def test_duration_seconds_default(self):
        """Should default duration_seconds to 0."""
        debate = {"id": "123"}
        result = normalize_debate_response(debate)
        assert result["duration_seconds"] == 0

    def test_confidence_agreement_alias(self):
        """Should add agreement alias from confidence."""
        debate = {"id": "123", "confidence": 0.8}
        result = normalize_debate_response(debate)
        assert result["agreement"] == 0.8

    def test_agreement_confidence_alias(self):
        """Should add confidence alias from agreement."""
        debate = {"id": "123", "agreement": 0.7}
        result = normalize_debate_response(debate)
        assert result["confidence"] == 0.7

    def test_conclusion_final_answer_alias(self):
        """Should add final_answer alias from conclusion."""
        debate = {"id": "123", "conclusion": "The answer is yes."}
        result = normalize_debate_response(debate)
        assert result["final_answer"] == "The answer is yes."

    def test_final_answer_conclusion_alias(self):
        """Should add conclusion alias from final_answer."""
        debate = {"id": "123", "final_answer": "The answer is no."}
        result = normalize_debate_response(debate)
        assert result["conclusion"] == "The answer is no."


# =============================================================================
# Test Cache TTL Constants
# =============================================================================


class TestCacheConstants:
    """Tests for cache TTL constants."""

    def test_cache_ttl_debates_list(self):
        """Should have reasonable TTL for debates list."""
        assert CACHE_TTL_DEBATES_LIST == 30

    def test_cache_ttl_search(self):
        """Should have reasonable TTL for search."""
        assert CACHE_TTL_SEARCH == 60

    def test_cache_ttl_convergence(self):
        """Should have reasonable TTL for convergence."""
        assert CACHE_TTL_CONVERGENCE == 120

    def test_cache_ttl_impasse(self):
        """Should have reasonable TTL for impasse."""
        assert CACHE_TTL_IMPASSE == 120


# =============================================================================
# Test Status Maps
# =============================================================================


class TestStatusMaps:
    """Tests for status mapping dictionaries."""

    def test_status_map_coverage(self):
        """Should map all known internal statuses."""
        assert "active" in STATUS_MAP
        assert "concluded" in STATUS_MAP
        assert "archived" in STATUS_MAP
        assert "starting" in STATUS_MAP
        assert "in_progress" in STATUS_MAP

    def test_status_reverse_map_coverage(self):
        """Should reverse map all known SDK statuses."""
        assert "running" in STATUS_REVERSE_MAP
        assert "completed" in STATUS_REVERSE_MAP
        assert "pending" in STATUS_REVERSE_MAP
        assert "created" in STATUS_REVERSE_MAP
        assert "in_progress" in STATUS_REVERSE_MAP

    def test_maps_are_symmetric(self):
        """Should have consistent forward and reverse mappings."""
        # active -> running -> active
        assert STATUS_REVERSE_MAP[STATUS_MAP["active"]] == "active"
        # concluded -> completed -> concluded
        assert STATUS_REVERSE_MAP[STATUS_MAP["concluded"]] == "concluded"
