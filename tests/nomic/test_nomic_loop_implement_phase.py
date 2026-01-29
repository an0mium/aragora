import pytest

from aragora.nomic.convoy_executor import GastownConvoyExecutor
from scripts.nomic_loop import NomicLoop


class DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.asyncio
async def test_create_implement_phase_prefers_gastown_executor(tmp_path):
    loop = NomicLoop(aragora_path=str(tmp_path))

    # Provide fallback agents so _create_agent_for_implement returns something.
    loop.claude = DummyAgent("anthropic-api")
    loop.codex = DummyAgent("openai-api")

    phase = loop._create_implement_phase()

    assert isinstance(phase._executor, GastownConvoyExecutor)
