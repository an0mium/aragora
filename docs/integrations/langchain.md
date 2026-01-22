# LangChain Integration Guide

Integrate Aragora's multi-agent debate capabilities into your LangChain applications.

## Installation

```bash
pip install aragora langchain langchain-openai
```

## Basic Integration

### Using Aragora as a LangChain Tool

```python
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from aragora import AragoraClientSync

# Initialize Aragora client
aragora = AragoraClientSync(
    base_url="https://api.aragora.ai",
    api_key="your-aragora-api-key"
)

def run_debate(query: str) -> str:
    """Run a multi-agent debate on a topic and return the consensus."""
    result = aragora.create_debate(
        question=query,
        agents=["claude", "gpt-4", "gemini"],
        rounds=3
    )

    # Poll for completion
    import time
    while True:
        debate = aragora.get_debate(result["debate_id"])
        if debate["status"] == "completed":
            break
        time.sleep(2)

    consensus = debate.get("consensus", {})
    return consensus.get("final_answer", "No consensus reached")

# Create LangChain tool
debate_tool = Tool(
    name="multi_agent_debate",
    func=run_debate,
    description="""Useful when you need multiple AI perspectives on a complex topic.
    Use this tool to stress-test ideas, validate decisions, or get balanced viewpoints.
    Input should be a clear question or topic to debate."""
)

# Create agent with the debate tool
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can run multi-agent debates for complex decisions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, [debate_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[debate_tool], verbose=True)

# Use the agent
result = agent_executor.invoke({
    "input": "Should we use GraphQL or REST for our new API? Consider scalability, developer experience, and maintenance."
})
print(result["output"])
```

### Custom Aragora Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from aragora import AragoraClientSync

class AragoraDebateChain:
    """Custom chain that uses Aragora for decision validation."""

    def __init__(self, aragora_api_key: str):
        self.aragora = AragoraClientSync(
            base_url="https://api.aragora.ai",
            api_key=aragora_api_key
        )
        self.llm = ChatOpenAI(model="gpt-4")

    def run(self, question: str, context: str = "") -> dict:
        # First, refine the question using LLM
        refine_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Given this context: {context}

Refine this question for a multi-agent debate: {question}

Refined question:"""
        )
        chain = LLMChain(llm=self.llm, prompt=refine_prompt)
        refined = chain.run(question=question, context=context)

        # Run the debate
        result = self.aragora.create_debate(
            question=refined.strip(),
            agents=["claude", "gpt-4", "gemini"],
            rounds=3
        )

        # Wait for completion
        import time
        while True:
            debate = self.aragora.get_debate(result["debate_id"])
            if debate["status"] in ["completed", "failed"]:
                break
            time.sleep(2)

        # Summarize with LLM
        summarize_prompt = PromptTemplate(
            input_variables=["debate_result"],
            template="""Summarize this debate result for a decision-maker:

{debate_result}

Executive Summary:"""
        )
        summary_chain = LLMChain(llm=self.llm, prompt=summarize_prompt)
        summary = summary_chain.run(debate_result=str(debate))

        return {
            "debate_id": result["debate_id"],
            "consensus": debate.get("consensus", {}),
            "summary": summary
        }

# Usage
chain = AragoraDebateChain(aragora_api_key="your-key")
result = chain.run(
    question="What cloud provider should we use?",
    context="We're a startup with 10 engineers, building a SaaS product"
)
print(result["summary"])
```

## Advanced: Async Integration

```python
import asyncio
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from aragora import AragoraClient

class DebateInput(BaseModel):
    question: str = Field(description="The question to debate")
    agents: list[str] = Field(
        default=["claude", "gpt-4"],
        description="AI agents to participate"
    )
    rounds: int = Field(default=3, description="Number of debate rounds")

async def async_debate(question: str, agents: list[str], rounds: int) -> str:
    async with AragoraClient(
        base_url="https://api.aragora.ai",
        api_key="your-key"
    ) as client:
        result = await client.create_debate(
            question=question,
            agents=agents,
            rounds=rounds
        )

        # Poll for completion
        while True:
            debate = await client.get_debate(result["debate_id"])
            if debate["status"] == "completed":
                return debate.get("consensus", {}).get("final_answer", "No consensus")
            await asyncio.sleep(2)

debate_tool = StructuredTool.from_function(
    func=lambda **kwargs: asyncio.run(async_debate(**kwargs)),
    name="debate",
    description="Run a multi-agent debate",
    args_schema=DebateInput
)
```

## Best Practices

1. **Caching**: Cache debate results for repeated questions
2. **Timeouts**: Set appropriate timeouts for long-running debates
3. **Error Handling**: Handle API errors and rate limits gracefully
4. **Context**: Provide relevant context in your questions for better results

## Related Resources

- [Aragora Python SDK](https://docs.aragora.ai/sdk/python)
- [LangChain Documentation](https://python.langchain.com/)
- [API Reference](https://docs.aragora.ai/api)
