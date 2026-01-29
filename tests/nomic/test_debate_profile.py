"""Tests for NomicDebateProfile - full-power debate configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora.nomic.debate_profile import (
    DEFAULT_NOMIC_AGENTS,
    NomicDebateProfile,
)


class TestDefaultConstants:
    """Tests for module-level constants."""

    def test_default_agents_count(self):
        assert len(DEFAULT_NOMIC_AGENTS) == 8

    def test_default_agents_are_strings(self):
        for agent in DEFAULT_NOMIC_AGENTS:
            assert isinstance(agent, str)
            assert len(agent) > 0

    def test_default_agents_unique(self):
        assert len(DEFAULT_NOMIC_AGENTS) == len(set(DEFAULT_NOMIC_AGENTS))

    def test_default_agents_expected_models(self):
        expected = {
            "grok",
            "anthropic-api",
            "openai-api",
            "deepseek",
            "mistral",
            "gemini",
            "qwen",
            "kimi",
        }
        assert set(DEFAULT_NOMIC_AGENTS) == expected


class TestNomicDebateProfileDefaults:
    """Tests for default profile configuration."""

    def test_default_agents(self):
        profile = NomicDebateProfile()
        assert profile.agents == list(DEFAULT_NOMIC_AGENTS)

    def test_default_rounds(self):
        profile = NomicDebateProfile()
        assert profile.rounds == 9

    def test_default_structured_phases(self):
        profile = NomicDebateProfile()
        assert profile.use_structured_phases is True

    def test_default_round_phases_count(self):
        profile = NomicDebateProfile()
        assert len(profile.round_phases) == 9  # Rounds 0-8

    def test_default_consensus_mode(self):
        profile = NomicDebateProfile()
        assert profile.consensus_mode == "judge"

    def test_default_judge_selection(self):
        profile = NomicDebateProfile()
        assert profile.judge_selection == "elo_ranked"

    def test_default_consensus_threshold(self):
        profile = NomicDebateProfile()
        assert profile.consensus_threshold == 0.6

    def test_default_proposer_count_all(self):
        profile = NomicDebateProfile()
        assert profile.proposer_count == -1  # All agents

    def test_default_critic_count_all(self):
        profile = NomicDebateProfile()
        assert profile.critic_count == -1  # All agents

    def test_default_role_rotation(self):
        profile = NomicDebateProfile()
        assert profile.role_rotation is True

    def test_default_asymmetric_stances(self):
        profile = NomicDebateProfile()
        assert profile.asymmetric_stances is True

    def test_default_early_stop_threshold_high(self):
        profile = NomicDebateProfile()
        assert profile.early_stop_threshold == 0.95

    def test_default_min_rounds_before_early_stop(self):
        profile = NomicDebateProfile()
        assert profile.min_rounds_before_early_stop == 8

    def test_default_agreement_intensity(self):
        profile = NomicDebateProfile()
        assert profile.agreement_intensity == 2

    def test_default_convergence_detection(self):
        profile = NomicDebateProfile()
        assert profile.convergence_detection is True


class TestNomicDebateProfileProperties:
    """Tests for computed properties."""

    def test_agent_names_returns_copy(self):
        profile = NomicDebateProfile()
        names = profile.agent_names
        assert names == list(DEFAULT_NOMIC_AGENTS)
        names.append("extra")
        assert len(profile.agent_names) == 8  # Original unchanged

    def test_agent_count(self):
        profile = NomicDebateProfile()
        assert profile.agent_count == 8

    def test_agent_count_custom(self):
        profile = NomicDebateProfile(agents=["a", "b", "c"])
        assert profile.agent_count == 3

    def test_total_phases(self):
        profile = NomicDebateProfile()
        assert profile.total_phases == 9

    def test_total_phases_custom(self):
        from aragora.debate.protocol import RoundPhase

        custom_phases = [
            RoundPhase(
                number=0, name="Only", description="One phase", focus="Test", cognitive_mode="Test"
            )
        ]
        profile = NomicDebateProfile(round_phases=custom_phases)
        assert profile.total_phases == 1


class TestToProtocol:
    """Tests for to_protocol() conversion."""

    def test_protocol_rounds(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.rounds == 9

    def test_protocol_structured_phases(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.use_structured_phases is True

    def test_protocol_consensus(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.consensus == "judge"

    def test_protocol_judge_selection(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.judge_selection == "elo_ranked"

    def test_protocol_proposer_count(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.proposer_count == -1

    def test_protocol_asymmetric_stances(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.asymmetric_stances is True

    def test_protocol_early_stopping(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.early_stopping is True
        assert protocol.early_stop_threshold == 0.95
        assert protocol.min_rounds_before_early_stop == 8

    def test_protocol_agreement_intensity(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.agreement_intensity == 2

    def test_protocol_convergence(self):
        profile = NomicDebateProfile()
        protocol = profile.to_protocol()
        assert protocol.convergence_detection is True

    def test_protocol_custom_rounds(self):
        profile = NomicDebateProfile(rounds=5)
        protocol = profile.to_protocol()
        assert protocol.rounds == 5

    def test_protocol_custom_consensus(self):
        profile = NomicDebateProfile(consensus_mode="majority")
        protocol = profile.to_protocol()
        assert protocol.consensus == "majority"


class TestToDebateConfig:
    """Tests for to_debate_config() conversion."""

    def test_debate_config_rounds(self):
        profile = NomicDebateProfile()
        config = profile.to_debate_config()
        assert config.rounds == 9

    def test_debate_config_consensus_mode(self):
        profile = NomicDebateProfile()
        config = profile.to_debate_config()
        assert config.consensus_mode == "judge"

    def test_debate_config_judge_selection(self):
        profile = NomicDebateProfile()
        config = profile.to_debate_config()
        assert config.judge_selection == "elo_ranked"

    def test_debate_config_proposer_count(self):
        profile = NomicDebateProfile()
        config = profile.to_debate_config()
        assert config.proposer_count == -1

    def test_debate_config_role_rotation(self):
        profile = NomicDebateProfile()
        config = profile.to_debate_config()
        assert config.role_rotation is True

    def test_debate_config_asymmetric_stances(self):
        profile = NomicDebateProfile()
        config = profile.to_debate_config()
        assert config.asymmetric_stances is True


class TestMinimalProfile:
    """Tests for minimal() factory method."""

    def test_minimal_agents(self):
        profile = NomicDebateProfile.minimal()
        assert profile.agents == ["anthropic-api", "openai-api", "deepseek"]

    def test_minimal_agent_count(self):
        profile = NomicDebateProfile.minimal()
        assert profile.agent_count == 3

    def test_minimal_rounds(self):
        profile = NomicDebateProfile.minimal()
        assert profile.rounds == 3

    def test_minimal_early_stop(self):
        profile = NomicDebateProfile.minimal()
        assert profile.min_rounds_before_early_stop == 2

    def test_minimal_no_asymmetric(self):
        profile = NomicDebateProfile.minimal()
        assert profile.asymmetric_stances is False

    def test_minimal_to_protocol(self):
        profile = NomicDebateProfile.minimal()
        protocol = profile.to_protocol()
        assert protocol.rounds == 3


class TestFromEnv:
    """Tests for from_env() factory method."""

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove any NOMIC_ env vars
            env = {k: v for k, v in os.environ.items() if not k.startswith("NOMIC_")}
            with patch.dict(os.environ, env, clear=True):
                profile = NomicDebateProfile.from_env()
                assert profile.agents == list(DEFAULT_NOMIC_AGENTS)
                assert profile.rounds == 9
                assert profile.agreement_intensity == 2

    def test_from_env_custom_agents(self):
        with patch.dict(os.environ, {"NOMIC_AGENTS": "claude,gpt,gemini"}):
            profile = NomicDebateProfile.from_env()
            assert profile.agents == ["claude", "gpt", "gemini"]

    def test_from_env_custom_rounds(self):
        with patch.dict(os.environ, {"NOMIC_DEBATE_ROUNDS": "5"}):
            profile = NomicDebateProfile.from_env()
            assert profile.rounds == 5

    def test_from_env_custom_agreement(self):
        with patch.dict(os.environ, {"NOMIC_AGREEMENT_INTENSITY": "7"}):
            profile = NomicDebateProfile.from_env()
            assert profile.agreement_intensity == 7

    def test_from_env_empty_agents_uses_default(self):
        with patch.dict(os.environ, {"NOMIC_AGENTS": ""}):
            profile = NomicDebateProfile.from_env()
            assert profile.agents == list(DEFAULT_NOMIC_AGENTS)

    def test_from_env_strips_whitespace(self):
        with patch.dict(os.environ, {"NOMIC_AGENTS": " claude , gpt , gemini "}):
            profile = NomicDebateProfile.from_env()
            assert profile.agents == ["claude", "gpt", "gemini"]


class TestCustomProfiles:
    """Tests for custom profile configurations."""

    def test_custom_agents(self):
        profile = NomicDebateProfile(agents=["a", "b"])
        assert profile.agent_count == 2

    def test_custom_rounds(self):
        profile = NomicDebateProfile(rounds=12)
        assert profile.rounds == 12

    def test_independent_instances(self):
        p1 = NomicDebateProfile()
        p2 = NomicDebateProfile()
        p1.agents.append("extra")
        assert len(p2.agents) == 8  # Unaffected

    def test_round_phases_independent(self):
        p1 = NomicDebateProfile()
        p2 = NomicDebateProfile()
        p1.round_phases.pop()
        assert len(p2.round_phases) == 9

    def test_all_consensus_modes(self):
        for mode in ["judge", "majority", "unanimous"]:
            profile = NomicDebateProfile(consensus_mode=mode)
            protocol = profile.to_protocol()
            assert protocol.consensus == mode
