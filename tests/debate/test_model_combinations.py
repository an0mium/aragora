from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aragora.debate.model_combinations import (
    ExecutionMode,
    ModelCombination,
    MultiModelDebateRunner,
    parse_model_combinations,
)


def _agent_factory(spec, index):
    return SimpleNamespace(
        name=f"{spec.provider}-{spec.role}-{index}",
        model=spec.model or f"default-{spec.provider}",
        role=spec.role,
        agent_type=spec.provider,
    )


class TestModelCombinationNormalization:
    def test_parse_model_combinations_supports_shorthand(self) -> None:
        combinations = parse_model_combinations(
            [
                ["anthropic-api", "openai-api", "gemini"],
                {
                    "name": "explicit-team",
                    "agents": [
                        {"provider": "openai-api", "role": "proposer"},
                        {"provider": "gemini", "role": "critic", "model": "gemini-3.1-pro-preview"},
                    ],
                    "metadata": {"tier": "candidate"},
                    "tags": ["test"],
                },
            ]
        )

        assert len(combinations) == 2
        assert combinations[0].name == "anthropic-api + openai-api + gemini"
        assert combinations[1].name == "explicit-team"
        assert combinations[1].agents[1].model == "gemini-3.1-pro-preview"
        assert combinations[1].metadata == {"tier": "candidate"}
        assert combinations[1].tags == ["test"]

    def test_from_dict_accepts_models_alias(self) -> None:
        combination = ModelCombination.from_dict(
            {
                "name": "alias-team",
                "models": ["anthropic-api", "openai-api"],
                "max_rounds": 5,
            }
        )

        assert combination.name == "alias-team"
        assert [agent.provider for agent in combination.agents] == [
            "anthropic-api",
            "openai-api",
        ]
        assert combination.max_rounds == 5

    def test_parse_model_combinations_requires_values(self) -> None:
        with pytest.raises(ValueError, match="At least one model combination"):
            parse_model_combinations([])


class TestMultiModelDebateRunner:
    @pytest.mark.asyncio
    async def test_runs_sequentially(self) -> None:
        active = 0
        max_active = 0

        async def executor(environment, agents, protocol, combination):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(
                final_answer=f"answer-{combination.name}",
                confidence=0.8,
                consensus_reached=True,
                winner=agents[0].name,
                rounds_used=protocol.rounds,
            )

        runner = MultiModelDebateRunner(
            debate_executor=executor,
            agent_factory=_agent_factory,
            max_parallel=3,
        )

        result = await runner.run(
            "Evaluate architecture tradeoffs",
            [
                ["anthropic-api", "openai-api", "gemini"],
                ["openai-api", "gemini", "anthropic-api"],
            ],
            mode=ExecutionMode.SEQUENTIAL,
        )

        assert result.mode is ExecutionMode.SEQUENTIAL
        assert max_active == 1
        assert len(result.results) == 2
        assert result.success_count == 2
        assert all(item.rounds_used == 9 for item in result.results)

    @pytest.mark.asyncio
    async def test_runs_in_parallel_batches(self) -> None:
        active = 0
        max_active = 0

        async def executor(environment, agents, protocol, combination):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return SimpleNamespace(
                final_answer=f"answer-{combination.name}",
                confidence=0.9,
                consensus_reached=True,
                winner=agents[0].name,
                rounds_used=protocol.rounds,
            )

        runner = MultiModelDebateRunner(
            debate_executor=executor,
            agent_factory=_agent_factory,
            max_parallel=2,
        )

        result = await runner.run(
            "Pick an API design",
            [
                ["anthropic-api", "openai-api", "gemini"],
                ["openai-api", "gemini", "anthropic-api"],
                ["gemini", "anthropic-api", "openai-api"],
            ],
            mode=ExecutionMode.PARALLEL,
        )

        assert result.mode is ExecutionMode.PARALLEL
        assert max_active == 2
        assert len(result.results) == 3
        assert result.success_count == 3

    @pytest.mark.asyncio
    async def test_captures_failure_per_combination(self) -> None:
        async def executor(environment, agents, protocol, combination):
            if combination.name == "failing-team":
                raise RuntimeError("combination exploded")
            return SimpleNamespace(
                final_answer="good result",
                confidence=0.7,
                consensus_reached=False,
                winner=None,
                rounds_used=protocol.rounds,
            )

        runner = MultiModelDebateRunner(
            debate_executor=executor,
            agent_factory=_agent_factory,
            max_parallel=2,
        )

        result = await runner.run(
            "Stress a plan",
            [
                {
                    "name": "healthy-team",
                    "agents": ["anthropic-api", "openai-api", "gemini"],
                },
                {
                    "name": "failing-team",
                    "agents": ["openai-api", "gemini", "anthropic-api"],
                },
            ],
        )

        healthy = next(item for item in result.results if item.combination_name == "healthy-team")
        failing = next(item for item in result.results if item.combination_name == "failing-team")

        assert healthy.status == "completed"
        assert failing.status == "failed"
        assert failing.error == "combination exploded"
        assert result.success_count == 1
        assert result.failure_count == 1

    @pytest.mark.asyncio
    async def test_auto_assigns_default_roles(self) -> None:
        captured_roles: list[str] = []

        def recording_agent_factory(spec, index):
            captured_roles.append(spec.role)
            return _agent_factory(spec, index)

        async def executor(environment, agents, protocol, combination):
            return SimpleNamespace(
                final_answer="done",
                confidence=1.0,
                consensus_reached=True,
                winner=agents[0].name,
                rounds_used=protocol.rounds,
            )

        runner = MultiModelDebateRunner(
            debate_executor=executor,
            agent_factory=recording_agent_factory,
            max_parallel=1,
        )

        await runner.run(
            "Evaluate platform choices",
            [["anthropic-api", "openai-api", "gemini", "qwen"]],
        )

        assert captured_roles == ["proposer", "critic", "synthesizer", "analyst"]

    @pytest.mark.asyncio
    async def test_preserves_agent_metadata_in_results(self) -> None:
        async def executor(environment, agents, protocol, combination):
            assert environment.task == "Debate the rollout"
            assert environment.context == "Use existing constraints."
            return SimpleNamespace(
                final_answer="ship it",
                confidence=0.88,
                consensus_reached=True,
                winner=agents[0].name,
                rounds_used=protocol.rounds,
            )

        runner = MultiModelDebateRunner(
            debate_executor=executor,
            agent_factory=_agent_factory,
            max_parallel=1,
        )

        result = await runner.run(
            "Debate the rollout",
            [
                {
                    "name": "team-a",
                    "agents": [
                        {"provider": "anthropic-api", "model": "claude-opus-4-6"},
                        {"provider": "openai-api", "model": "gpt-4.1"},
                    ],
                    "metadata": {"seed": "baseline"},
                }
            ],
            base_context="Use existing constraints.",
            max_rounds=4,
        )

        entry = result.results[0]

        assert entry.combination_name == "team-a"
        assert entry.agents[0]["provider"] == "anthropic-api"
        assert entry.agents[0]["model"] == "claude-opus-4-6"
        assert entry.agents[1]["role"] == "critic"
        assert entry.rounds_used == 4
        assert entry.metadata["combination_metadata"] == {"seed": "baseline"}
