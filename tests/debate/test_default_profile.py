from aragora.config import DEFAULT_AGENTS, DEFAULT_CONSENSUS, DEFAULT_ROUNDS
from aragora.config.settings import AgentSettings, DebateSettings
from aragora.debate.protocol import (
    DebateProtocol,
    STRUCTURED_ROUND_PHASES,
    resolve_default_protocol,
)
from aragora.debate.service import DebateOptions
from aragora.nomic.debate_profile import NomicDebateProfile


def test_debate_settings_align_with_legacy_defaults() -> None:
    settings = DebateSettings()
    assert settings.default_rounds == DEFAULT_ROUNDS
    assert settings.default_consensus == DEFAULT_CONSENSUS


def test_agent_settings_align_with_legacy_defaults() -> None:
    settings = AgentSettings()
    assert settings.default_agents == DEFAULT_AGENTS
    assert settings.default_agent_list == [
        agent.strip() for agent in DEFAULT_AGENTS.split(",") if agent.strip()
    ]


def test_structured_round_phases_match_default_rounds() -> None:
    assert len(STRUCTURED_ROUND_PHASES) == DEFAULT_ROUNDS


def test_resolve_default_protocol_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_DEBATE_PROFILE", raising=False)
    protocol = resolve_default_protocol(None)
    assert protocol.rounds == DEFAULT_ROUNDS
    assert protocol.consensus == DEFAULT_CONSENSUS


def test_debate_options_defaults_align() -> None:
    opts = DebateOptions()
    assert opts.rounds == DEFAULT_ROUNDS
    assert opts.consensus == DEFAULT_CONSENSUS


def test_nomic_profile_defaults_align() -> None:
    profile = NomicDebateProfile()
    assert profile.rounds == DEFAULT_ROUNDS
    assert profile.agent_names == AgentSettings().default_agent_list
    assert profile.total_phases == len(STRUCTURED_ROUND_PHASES)
