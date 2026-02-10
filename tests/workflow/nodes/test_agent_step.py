"""Tests for AgentStep enhancements (agent pool + power sampling)."""

import pytest

from aragora.workflow.step import AgentStep, WorkflowContext


class StubAgent:
    def __init__(self, response: str):
        self._response = response

    async def generate(self, prompt: str):
        return self._response


class CountingAgent:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str):
        self.calls += 1
        return f"resp {self.calls}"


@pytest.mark.asyncio
async def test_agent_pool_selects_best_response(monkeypatch):
    import aragora.agents as agents_module

    responses = {
        "agent_a": "good response therefore conclusion.",
        "agent_b": "short response.",
    }

    def create_agent_stub(agent_type: str):
        return StubAgent(responses[agent_type])

    monkeypatch.setattr(agents_module, "create_agent", create_agent_stub)

    def scorer(response: str, prompt: str) -> float:
        return 1.0 if "good" in response else 0.1

    step = AgentStep(
        name="Pool Step",
        config={
            "agent_pool": ["agent_a", "agent_b"],
            "include_candidates": True,
            "response_scorer": scorer,
            "prompt_template": "Test prompt",
        },
    )

    ctx = WorkflowContext(workflow_id="wf1", definition_id="def1", inputs={})
    result = await step.execute(ctx)

    assert result["agent_type"] == "agent_a"
    assert result.get("best_score") == 1.0
    assert "candidates" in result


@pytest.mark.asyncio
async def test_agent_pool_power_sampling_uses_multiple_samples(monkeypatch):
    import aragora.agents as agents_module

    agent = CountingAgent()

    def create_agent_stub(agent_type: str):
        return agent

    monkeypatch.setattr(agents_module, "create_agent", create_agent_stub)

    def scorer(response: str, prompt: str) -> float:
        # Prefer later samples
        try:
            return float(response.split()[-1])
        except Exception:
            return 0.0

    step = AgentStep(
        name="Pool Step",
        config={
            "agent_pool": ["agent_a"],
            "samples_per_agent": 3,
            "power_sampling": {"n_samples": 3},
            "response_scorer": scorer,
            "prompt_template": "Test prompt",
        },
    )

    ctx = WorkflowContext(workflow_id="wf1", definition_id="def1", inputs={})
    result = await step.execute(ctx)

    assert agent.calls == 3
    assert result["agent_type"] == "agent_a"
