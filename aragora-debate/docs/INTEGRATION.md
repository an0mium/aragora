# Integration Guide

Use `aragora-debate` as a drop-in adversarial debate engine in any Python project.

## Installation

```bash
# Core (zero dependencies)
pip install aragora-debate

# With Claude support
pip install aragora-debate[anthropic]

# With OpenAI support
pip install aragora-debate[openai]

# Both providers
pip install aragora-debate[all]
```

## Quick Start

```python
import asyncio
from aragora_debate import Arena, ClaudeAgent, OpenAIAgent

async def main():
    agents = [
        ClaudeAgent("analyst", model="claude-sonnet-4-5-20250929"),
        OpenAIAgent("challenger", model="gpt-4o"),
    ]
    arena = Arena(
        question="Should we migrate from PostgreSQL to a distributed database?",
        agents=agents,
    )
    result = await arena.run()

    print(result.receipt.to_markdown())
    print(f"\nVerdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.0%}")

asyncio.run(main())
```

## Custom Agents

Implement the `Agent` ABC to wrap any LLM or decision-making system:

```python
from aragora_debate import Agent, Critique, Message, Vote

class MyAgent(Agent):
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        # Call your LLM here
        return "My proposal..."

    async def critique(
        self, proposal: str, task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        best = max(proposals, key=lambda k: len(proposals[k]))
        return Vote(agent=self.name, choice=best, confidence=0.8)
```

## Framework Integrations

### CrewAI

```python
from crewai import Agent as CrewAgent, Task, Crew
from aragora_debate import Arena, Agent, Critique, Message, Vote

class CrewAIDebateAgent(Agent):
    """Wrap a CrewAI agent as a debate participant."""

    def __init__(self, name: str, crew_agent: CrewAgent):
        super().__init__(name=name, model=crew_agent.llm or "")
        self._crew_agent = crew_agent

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        task = Task(description=prompt, agent=self._crew_agent)
        crew = Crew(agents=[self._crew_agent], tasks=[task])
        result = crew.kickoff()
        return str(result)

    async def critique(self, proposal, task, context=None, target_agent=None):
        prompt = f"Critique this proposal for '{task}':\n{proposal}"
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=[await self.generate(prompt)],
        )

    async def vote(self, proposals, task):
        prompt = f"For '{task}', which is best?\n"
        for name, content in proposals.items():
            prompt += f"\n{name}: {content[:200]}"
        response = await self.generate(prompt)
        choice = list(proposals.keys())[0]  # parse from response
        return Vote(agent=self.name, choice=choice, confidence=0.7)
```

### LangGraph

```python
from langchain_openai import ChatOpenAI
from aragora_debate import Arena, Agent, Critique, Message, Vote

class LangChainDebateAgent(Agent):
    """Wrap a LangChain LLM as a debate participant."""

    def __init__(self, name: str, llm: ChatOpenAI):
        super().__init__(name=name, model=llm.model_name)
        self._llm = llm

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        response = await self._llm.ainvoke(prompt)
        return response.content

    async def critique(self, proposal, task, context=None, target_agent=None):
        prompt = f"Critique this proposal for '{task}':\n{proposal}"
        response = await self._llm.ainvoke(prompt)
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=[response.content],
        )

    async def vote(self, proposals, task):
        prompt = f"Vote for the best proposal for '{task}':\n"
        for name, content in proposals.items():
            prompt += f"\n{name}: {content[:200]}"
        response = await self._llm.ainvoke(prompt)
        return Vote(agent=self.name, choice=list(proposals.keys())[0], confidence=0.7)

# Use in a LangGraph state machine:
# node "debate" -> Arena.run() -> node "act_on_result"
```

### AutoGen

```python
from autogen import AssistantAgent
from aragora_debate import Arena, Agent, Critique, Message, Vote

class AutoGenDebateAgent(Agent):
    def __init__(self, name: str, autogen_agent: AssistantAgent):
        super().__init__(name=name, model=autogen_agent.llm_config.get("model", ""))
        self._agent = autogen_agent

    async def generate(self, prompt, context=None):
        reply = self._agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        return reply

    async def critique(self, proposal, task, context=None, target_agent=None):
        reply = await self.generate(f"Critique for '{task}': {proposal}")
        return Critique(
            agent=self.name, target_agent=target_agent or "unknown",
            target_content=proposal, issues=[reply],
        )

    async def vote(self, proposals, task):
        return Vote(agent=self.name, choice=list(proposals.keys())[0], confidence=0.7)
```

## Configuration

```python
from aragora_debate import Arena, DebateConfig, ConsensusMethod

config = DebateConfig(
    rounds=5,                                    # Max debate rounds
    consensus_method=ConsensusMethod.SUPERMAJORITY,  # 2/3 agreement required
    consensus_threshold=0.7,                     # Confidence threshold
    early_stopping=True,                         # Stop when consensus found
    early_stop_threshold=0.85,                   # High-confidence early stop
    min_rounds=2,                                # Minimum rounds before early stop
)

arena = Arena(
    question="Should we adopt Kubernetes?",
    agents=agents,
    config=config,
    context="We currently run on bare EC2 instances with 50 microservices.",
)
```

## Decision Receipts

Every debate produces a cryptographic `DecisionReceipt`:

```python
result = await arena.run()

# Markdown summary
print(result.receipt.to_markdown())

# JSON for API responses
from aragora_debate import ReceiptBuilder
json_str = ReceiptBuilder.to_json(result.receipt)

# HTML for reports
html = ReceiptBuilder.to_html(result.receipt)

# HMAC signing for tamper detection
ReceiptBuilder.sign_hmac(result.receipt, "your-secret-key")
assert ReceiptBuilder.verify_hmac(result.receipt, "your-secret-key")
```

## Assigned Stances

Force agents to argue specific positions:

```python
agents = [
    ClaudeAgent("proponent", stance="affirmative"),
    OpenAIAgent("opponent", stance="negative"),
    ClaudeAgent("judge", stance="neutral", model="claude-opus-4-6"),
]
```

## Cost Tracking

```python
result = await arena.run()
print(f"Total tokens: {result.total_tokens}")
print(f"Total cost: ${result.total_cost_usd:.4f}")
for agent, cost in result.per_agent_cost.items():
    print(f"  {agent}: ${cost:.4f}")
```
