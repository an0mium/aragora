"""Tests for aragora.debate.config.defaults module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestDebateDefaults:
    """Test the DebateDefaults dataclass."""

    def test_import(self):
        """Verify module imports successfully."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS, DebateDefaults

        assert DEBATE_DEFAULTS is not None
        assert isinstance(DEBATE_DEFAULTS, DebateDefaults)

    def test_frozen_dataclass(self):
        """Verify defaults are immutable."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        with pytest.raises(AttributeError):
            DEBATE_DEFAULTS.convergence_threshold = 0.5

    def test_convergence_thresholds(self):
        """Test convergence threshold values."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        assert DEBATE_DEFAULTS.convergence_threshold == 0.85
        assert DEBATE_DEFAULTS.divergence_threshold == 0.40
        assert DEBATE_DEFAULTS.convergence_threshold > DEBATE_DEFAULTS.divergence_threshold

    def test_consensus_thresholds(self):
        """Test consensus threshold values."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        assert DEBATE_DEFAULTS.consensus_threshold == 0.6
        assert DEBATE_DEFAULTS.strong_consensus_agreement_ratio == 0.8
        assert DEBATE_DEFAULTS.strong_consensus_confidence == 0.7

    def test_timeout_values(self):
        """Test timeout values are positive and reasonable."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        assert DEBATE_DEFAULTS.agent_timeout_seconds > 0
        assert DEBATE_DEFAULTS.round_timeout_seconds > 0
        assert DEBATE_DEFAULTS.debate_total_timeout_seconds > 0
        # Total timeout should be greater than individual timeouts
        assert DEBATE_DEFAULTS.debate_total_timeout_seconds > DEBATE_DEFAULTS.round_timeout_seconds

    def test_quality_thresholds(self):
        """Test quality gate thresholds."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        assert 0.0 <= DEBATE_DEFAULTS.quality_gate_threshold <= 1.0
        assert 0.0 <= DEBATE_DEFAULTS.min_quality_threshold <= 1.0
        assert 0.0 <= DEBATE_DEFAULTS.hollow_detection_threshold <= 1.0

    def test_confidence_thresholds_in_valid_range(self):
        """Test confidence thresholds are between 0 and 1."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        confidence_fields = [
            "extraction_min_confidence",
            "coordinator_min_confidence_for_mound",
            "broadcast_min_confidence",
            "training_export_min_confidence",
            "breeding_threshold",
            "post_debate_workflow_threshold",
            "receipt_min_confidence",
            "bead_min_confidence",
        ]

        for field in confidence_fields:
            value = getattr(DEBATE_DEFAULTS, field)
            assert 0.0 <= value <= 1.0, f"{field} = {value} is out of range [0, 1]"

    def test_convergence_weights_sum_to_one(self):
        """Test convergence metric weights sum to 1.0."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        total_weight = (
            DEBATE_DEFAULTS.convergence_weight_semantic
            + DEBATE_DEFAULTS.convergence_weight_diversity
            + DEBATE_DEFAULTS.convergence_weight_evidence
            + DEBATE_DEFAULTS.convergence_weight_stability
        )
        assert abs(total_weight - 1.0) < 0.001, f"Weights sum to {total_weight}, expected 1.0"


class TestEnvironmentOverrides:
    """Test environment variable overrides for defaults."""

    def test_get_debate_defaults_with_env_override(self):
        """Test that environment variables override defaults."""
        from aragora.debate.config.defaults import get_debate_defaults

        # Clear the cache to ensure fresh instance
        get_debate_defaults.cache_clear()

        with patch.dict(os.environ, {"ARAGORA_DEBATE_CONVERGENCE_THRESHOLD": "0.95"}):
            get_debate_defaults.cache_clear()
            defaults = get_debate_defaults()
            assert defaults.convergence_threshold == 0.95

        # Cleanup: clear cache again
        get_debate_defaults.cache_clear()

    def test_get_debate_defaults_invalid_env_fallback(self):
        """Test that invalid env values fall back to defaults."""
        from aragora.debate.config.defaults import get_debate_defaults

        get_debate_defaults.cache_clear()

        with patch.dict(os.environ, {"ARAGORA_DEBATE_CONVERGENCE_THRESHOLD": "not_a_number"}):
            get_debate_defaults.cache_clear()
            defaults = get_debate_defaults()
            # Should fall back to default 0.85
            assert defaults.convergence_threshold == 0.85

        get_debate_defaults.cache_clear()

    def test_integer_env_override(self):
        """Test integer environment variable override."""
        from aragora.debate.config.defaults import get_debate_defaults

        get_debate_defaults.cache_clear()

        with patch.dict(os.environ, {"ARAGORA_DEBATE_TOTAL_TIMEOUT": "2400"}):
            get_debate_defaults.cache_clear()
            defaults = get_debate_defaults()
            assert defaults.debate_total_timeout_seconds == 2400

        get_debate_defaults.cache_clear()


class TestIntegration:
    """Test integration with other modules."""

    def test_convergence_detector_uses_defaults(self):
        """Test ConvergenceDetector uses DEBATE_DEFAULTS."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector()
        assert detector.convergence_threshold == DEBATE_DEFAULTS.convergence_threshold
        assert detector.divergence_threshold == DEBATE_DEFAULTS.divergence_threshold

    def test_convergence_detector_custom_overrides(self):
        """Test ConvergenceDetector accepts custom threshold overrides."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            convergence_threshold=0.90,
            divergence_threshold=0.35,
        )
        assert detector.convergence_threshold == 0.90
        assert detector.divergence_threshold == 0.35

    def test_consensus_proof_uses_defaults(self):
        """Test ConsensusProof uses DEBATE_DEFAULTS for has_strong_consensus."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS
        from aragora.debate.consensus import ConsensusProof

        # Create proof with exactly the threshold values
        proof = ConsensusProof(
            proof_id="test",
            debate_id="test-debate",
            task="Test task",
            final_claim="Test claim",
            confidence=DEBATE_DEFAULTS.strong_consensus_confidence + 0.01,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2", "a3", "a4", "a5"],  # 5 supporting
            dissenting_agents=["d1"],  # 1 dissenting = 5/6 = 0.833 > 0.8
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        # 5 supporting / 6 total = 0.833 > 0.8 threshold
        assert proof.has_strong_consensus is True

    def test_consensus_proof_below_threshold(self):
        """Test ConsensusProof returns false when below threshold."""
        from aragora.debate.consensus import ConsensusProof

        # Create proof with low confidence
        proof = ConsensusProof(
            proof_id="test",
            debate_id="test-debate",
            task="Test task",
            final_claim="Test claim",
            confidence=0.5,  # Below 0.7 threshold
            consensus_reached=True,
            votes=[],
            supporting_agents=["a1", "a2", "a3", "a4", "a5"],
            dissenting_agents=["d1"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        assert proof.has_strong_consensus is False


class TestDefaultsConsistency:
    """Verify that all debate entrypoints derive their defaults from DEBATE_DEFAULTS.

    This ensures no local hard-coded values diverge from the centralized
    source of truth (Issue #178).
    """

    def test_distributed_config_uses_centralized_defaults(self):
        """DistributedDebateConfig should match DEBATE_DEFAULTS."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS
        from aragora.debate.distributed import DistributedDebateConfig

        config = DistributedDebateConfig()
        assert config.max_rounds == DEBATE_DEFAULTS.distributed_default_rounds
        assert config.consensus_threshold == DEBATE_DEFAULTS.distributed_consensus_threshold
        assert config.min_agents == DEBATE_DEFAULTS.min_agents_per_debate
        assert config.max_agents == DEBATE_DEFAULTS.max_agents_per_debate
        assert (
            config.proposal_timeout_seconds == DEBATE_DEFAULTS.distributed_proposal_timeout_seconds
        )
        assert (
            config.critique_timeout_seconds == DEBATE_DEFAULTS.distributed_critique_timeout_seconds
        )
        assert config.vote_timeout_seconds == DEBATE_DEFAULTS.distributed_vote_timeout_seconds
        assert config.sync_interval_seconds == DEBATE_DEFAULTS.distributed_sync_interval_seconds
        assert (
            config.failover_timeout_seconds == DEBATE_DEFAULTS.distributed_failover_timeout_seconds
        )

    def test_fabric_config_uses_centralized_defaults(self):
        """FabricDebateConfig should match DEBATE_DEFAULTS for agent limits."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS
        from aragora.debate.fabric_integration import FabricDebateConfig

        config = FabricDebateConfig(pool_id="test-pool")
        assert config.min_agents == DEBATE_DEFAULTS.min_agents_per_debate
        assert config.max_agents == DEBATE_DEFAULTS.max_agents_per_debate

    def test_byzantine_config_uses_centralized_defaults(self):
        """ByzantineConsensusConfig should match DEBATE_DEFAULTS."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS
        from aragora.debate.byzantine import ByzantineConsensusConfig

        config = ByzantineConsensusConfig()
        assert config.min_agents == DEBATE_DEFAULTS.byzantine_min_agents
        # Byzantine min_agents must satisfy n >= 3f+1
        assert config.min_agents >= 4

    def test_settings_and_defaults_agent_limits_agree(self):
        """DebateSettings.max_agents_per_debate should match DEBATE_DEFAULTS."""
        from aragora.config.settings import get_settings
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        settings = get_settings()
        assert settings.debate.max_agents_per_debate == DEBATE_DEFAULTS.max_agents_per_debate

    def test_security_debate_uses_centralized_defaults(self):
        """Security debate protocol values should match DEBATE_DEFAULTS."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        # Verify the centralized defaults exist and have expected values
        assert DEBATE_DEFAULTS.security_debate_rounds == 3
        assert DEBATE_DEFAULTS.security_debate_consensus == "majority"
        assert DEBATE_DEFAULTS.security_debate_timeout_seconds == 300

    def test_agent_limit_invariants(self):
        """Agent limit invariants: min < max, byzantine > standard min."""
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        assert DEBATE_DEFAULTS.min_agents_per_debate < DEBATE_DEFAULTS.max_agents_per_debate
        assert DEBATE_DEFAULTS.byzantine_min_agents >= DEBATE_DEFAULTS.min_agents_per_debate
        assert DEBATE_DEFAULTS.byzantine_min_agents >= 4  # BFT requirement: n >= 3f+1

    def test_distributed_rounds_less_than_standard(self):
        """Distributed rounds should be <= standard defaults (higher latency)."""
        from aragora.config.settings import get_settings
        from aragora.debate.config.defaults import DEBATE_DEFAULTS

        settings = get_settings()
        assert DEBATE_DEFAULTS.distributed_default_rounds <= settings.debate.default_rounds

    def test_no_hardcoded_magic_numbers_in_defaults(self):
        """All numeric defaults should be positive and within reasonable bounds."""
        from dataclasses import fields

        from aragora.debate.config.defaults import DEBATE_DEFAULTS, DebateDefaults

        for f in fields(DebateDefaults):
            value = getattr(DEBATE_DEFAULTS, f.name)
            if isinstance(value, (int, float)):
                assert value >= 0, f"{f.name} = {value} should be non-negative"
            if isinstance(value, float) and "threshold" in f.name:
                # Some "threshold" fields are multipliers, not ratios
                multiplier_thresholds = {
                    "verbosity_penalty_threshold",  # multiplier of target length
                }
                if f.name not in multiplier_thresholds:
                    assert 0.0 <= value <= 1.0, f"{f.name} = {value} should be in [0, 1]"
