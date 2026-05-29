from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.config.secrets import SecretNotFoundError, SecretPresence


def test_optional_get_api_key_treats_strict_secret_miss_as_absent(monkeypatch: pytest.MonkeyPatch):
    """Optional provider probes must not fail a selected-provider path."""
    from aragora.config.legacy import get_api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")

    def strict_secret_presence(name: str) -> SecretPresence:
        return SecretPresence(
            name=name,
            source="blocked_by_strict_mode",
            critical=True,
            managed=True,
        )

    monkeypatch.setattr("aragora.config.secrets.get_secret_presence", strict_secret_presence)
    monkeypatch.setattr(
        "aragora.config.secrets.get_secret",
        lambda name: (_ for _ in ()).throw(AssertionError("value probe should be skipped")),
    )

    assert get_api_key("OPENROUTER_API_KEY", required=False) is None


def test_primary_key_is_used_before_openrouter_fallback_probe():
    """Direct providers should not fail on OpenRouter fallback when primary key exists."""
    from aragora.agents.api_agents import common

    def fake_get_api_key(*env_vars: str, required: bool = True) -> str | None:
        if env_vars == ("XAI_API_KEY", "GROK_API_KEY") and required is False:
            return "xai-primary-key"
        if env_vars == ("OPENROUTER_API_KEY",):
            raise AssertionError("OpenRouter fallback should not be probed before primary")
        raise AssertionError(f"unexpected api key lookup: {env_vars!r}, required={required!r}")

    with patch.object(common, "get_api_key", side_effect=fake_get_api_key):
        assert (
            common.get_primary_api_key(
                "XAI_API_KEY",
                "GROK_API_KEY",
                allow_openrouter_fallback=True,
            )
            == "xai-primary-key"
        )


def test_mistral_api_is_allowed_in_ask_agent_specs():
    """Provider readiness and ask parsing must agree on the direct Mistral provider name."""
    from aragora.agents.registry import AgentRegistry
    from aragora.agents.spec import AgentSpec

    parsed = AgentSpec.parse("mistral-api:proposer", _warn=False)

    assert parsed.provider == "mistral-api"
    assert parsed.role == "proposer"
    assert AgentRegistry.validate_allowed("mistral-api")


@pytest.mark.asyncio
async def test_run_debate_passes_resolved_provider_key_to_direct_agent():
    """Local ask should reuse the same provider key surface validate-env checks."""
    from aragora.cli.main import run_debate

    created_agents: list[dict] = []

    def track_create(*args, **kwargs):
        created_agents.append(kwargs)
        agent = MagicMock()
        agent.name = kwargs.get("name", "grok_proposer")
        return agent

    with patch("aragora.cli.commands.debate.create_agent", side_effect=track_create):
        with patch("aragora.cli.commands.debate.Arena") as mock_arena:
            mock_result = MagicMock()
            mock_arena.return_value.run = AsyncMock(return_value=mock_result)
            with patch(
                "aragora.cli.api_keys.get_provider_key",
                return_value=("xai-provider-key", "environment (XAI_API_KEY)"),
            ):
                await run_debate(
                    task="Test",
                    agents_str="grok",
                    rounds=1,
                    learn=False,
                    offline=True,
                )

    assert created_agents[0]["api_key"] == "xai-provider-key"


def test_grok_registry_accepts_explicit_api_key():
    """The direct Grok API agent must accept the key passed by local ask."""
    from aragora.agents.registry import AgentRegistry, register_all_agents

    register_all_agents()

    assert AgentRegistry._registry["grok"].accepts_api_key is True


def test_semantic_store_auto_detect_falls_back_on_optional_secret_miss(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Knowledge setup should not block a selected ask provider on missing embedding keys."""
    import aragora.config as config
    from aragora.knowledge.mound.semantic_store import EmbeddingProvider, SemanticStore

    def optional_secret_miss(*env_vars: str, required: bool = True) -> str | None:
        assert required is False
        raise SecretNotFoundError(env_vars[0])

    monkeypatch.setattr(config, "get_api_key", optional_secret_miss)

    with patch("socket.socket") as mock_socket:
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket.return_value.__exit__ = MagicMock(return_value=None)

        store = SemanticStore(str(tmp_path / "semantic.db"))

    assert isinstance(store.embedding_provider, EmbeddingProvider)
    assert store.embedding_dimension == 256


@pytest.mark.asyncio
async def test_synthesis_uses_combined_output_when_optional_anthropic_missing():
    """Single-provider asks should not probe Anthropic for mandatory synthesis."""
    from aragora.debate.phases.synthesis_generator import SynthesisGenerator

    ctx = SimpleNamespace(
        env=SimpleNamespace(task="Return a short answer"),
        proposals={"grok_proposer": "provider reached"},
        result=SimpleNamespace(
            confidence=0.5,
            debate_id="debate-test",
            final_answer="provider reached",
            synthesis=None,
            winner=None,
        ),
    )
    generator = SynthesisGenerator(protocol=SimpleNamespace(enable_llm_synthesis=True, rounds=1))

    with patch.object(generator, "_anthropic_synthesis_available", return_value=False):
        ok = await generator.generate_mandatory_synthesis(ctx)

    assert ok is True
    assert "provider reached" in ctx.result.synthesis
    assert "provider reached" in ctx.result.final_answer


@pytest.mark.asyncio
async def test_round_knowledge_refresh_treats_external_embedding_failure_as_optional():
    """Expired unrelated embedding providers must not fail an explicit-provider ask."""
    from aragora.debate.orchestrator_delegates import ArenaDelegatesMixin
    from aragora.exceptions import ExternalServiceError

    class Harness(ArenaDelegatesMixin):
        def __init__(self) -> None:
            self.enable_knowledge_retrieval = True
            self.env = SimpleNamespace(task="Provider isolation validation")
            self._km_manager = SimpleNamespace(
                fetch_context=AsyncMock(
                    side_effect=ExternalServiceError(
                        service="Gemini Embedding",
                        reason="API key expired",
                        status_code=400,
                    )
                )
            )

    harness = Harness()

    refreshed = await harness._refresh_knowledge_context_for_round(
        "explicit Grok provider returned a response",
        SimpleNamespace(),
        1,
    )

    assert refreshed == 0
    harness._km_manager.fetch_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_judge_skips_when_optional_anthropic_missing(monkeypatch: pytest.MonkeyPatch):
    """Optional post-debate judging must not become an unrelated provider failure."""
    from aragora.evaluation.llm_judge import LLMJudge

    monkeypatch.setattr("aragora.config.get_api_key", lambda *args, **kwargs: None)

    with patch.object(LLMJudge, "_call_judge", new_callable=AsyncMock) as call_judge:
        result = await LLMJudge().evaluate(query="q", response="r")

    call_judge.assert_not_awaited()
    assert result.summary == "Evaluation skipped: ANTHROPIC_API_KEY not configured"
