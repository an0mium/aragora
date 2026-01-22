# LlamaIndex Integration Guide

Integrate Aragora's multi-agent debate capabilities into your LlamaIndex RAG applications.

## Installation

```bash
pip install aragora llama-index llama-index-llms-openai
```

## Basic Integration

### Aragora as a Query Engine Tool

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from aragora import AragoraClientSync

# Initialize clients
aragora = AragoraClientSync(
    base_url="https://api.aragora.ai",
    api_key="your-aragora-api-key"
)
llm = OpenAI(model="gpt-4", temperature=0)

def debate_query(question: str) -> str:
    """
    Run a multi-agent debate to get balanced perspectives on a question.

    Args:
        question: The question or topic to debate

    Returns:
        The consensus answer from the debate
    """
    result = aragora.create_debate(
        question=question,
        agents=["claude", "gpt-4", "gemini"],
        rounds=3
    )

    # Wait for completion
    import time
    while True:
        debate = aragora.get_debate(result["debate_id"])
        if debate["status"] == "completed":
            break
        time.sleep(2)

    consensus = debate.get("consensus", {})
    return f"""
Consensus: {consensus.get("final_answer", "No consensus")}
Confidence: {consensus.get("confidence", 0):.0%}
Method: {consensus.get("method", "unknown")}
"""

# Create tool
debate_tool = FunctionTool.from_defaults(
    fn=debate_query,
    name="multi_agent_debate",
    description="Use this to get multiple AI perspectives on complex decisions or controversial topics"
)

# Create agent
agent = ReActAgent.from_tools(
    tools=[debate_tool],
    llm=llm,
    verbose=True
)

# Query
response = agent.chat("Should our company adopt a 4-day work week?")
print(response)
```

### Combining RAG with Debate

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from aragora import AragoraClientSync

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Initialize Aragora
aragora = AragoraClientSync(api_key="your-key")

def debate_with_context(question: str, context: str = "") -> str:
    """Run a debate with additional context."""
    full_question = f"{question}\n\nContext: {context}" if context else question

    result = aragora.create_debate(
        question=full_question,
        agents=["claude", "gpt-4"],
        rounds=2
    )

    import time
    while True:
        debate = aragora.get_debate(result["debate_id"])
        if debate["status"] == "completed":
            break
        time.sleep(2)

    return debate.get("consensus", {}).get("final_answer", "No consensus")

# Create tools
rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Search internal documents for factual information"
)

debate_tool = FunctionTool.from_defaults(
    fn=debate_with_context,
    name="debate_decision",
    description="Get multiple AI perspectives on a decision. Use after gathering facts."
)

# Create agent with both tools
agent = ReActAgent.from_tools(
    tools=[rag_tool, debate_tool],
    llm=OpenAI(model="gpt-4"),
    verbose=True,
    system_prompt="""You are a decision support assistant.
    First search documents for relevant facts, then use the debate tool
    to get balanced perspectives on important decisions."""
)

# Use the agent
response = agent.chat(
    "Based on our company policies, should we allow remote work for the engineering team?"
)
print(response)
```

### Custom Query Engine with Debate Validation

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.openai import OpenAI
from aragora import AragoraClientSync

class DebateValidatedQueryEngine(CustomQueryEngine):
    """Query engine that validates answers through multi-agent debate."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    aragora: AragoraClientSync

    def custom_query(self, query_str: str) -> str:
        # Get initial answer from RAG
        nodes = self.retriever.retrieve(query_str)
        initial_response = self.response_synthesizer.synthesize(
            query_str, nodes=nodes
        )

        # Validate through debate
        validation_query = f"""
        Question: {query_str}

        Proposed Answer: {initial_response}

        Is this answer accurate and complete? What nuances or caveats should be added?
        """

        result = self.aragora.create_debate(
            question=validation_query,
            agents=["claude", "gpt-4"],
            rounds=2
        )

        import time
        while True:
            debate = self.aragora.get_debate(result["debate_id"])
            if debate["status"] == "completed":
                break
            time.sleep(2)

        consensus = debate.get("consensus", {})

        # Combine results
        return f"""
## Answer
{initial_response}

## Validation ({consensus.get("confidence", 0):.0%} confidence)
{consensus.get("final_answer", "No additional validation")}
"""

# Usage
index = VectorStoreIndex.from_documents(documents)
query_engine = DebateValidatedQueryEngine(
    retriever=index.as_retriever(),
    response_synthesizer=index.as_query_engine().get_response_synthesizer(),
    aragora=AragoraClientSync(api_key="your-key")
)

response = query_engine.query("What is our refund policy?")
print(response)
```

## Async Integration

```python
import asyncio
from llama_index.core.tools import FunctionTool
from aragora import AragoraClient

async def async_debate(question: str) -> str:
    async with AragoraClient(api_key="your-key") as client:
        result = await client.create_debate(
            question=question,
            agents=["claude", "gpt-4"],
        )

        while True:
            debate = await client.get_debate(result["debate_id"])
            if debate["status"] == "completed":
                return debate.get("consensus", {}).get("final_answer", "")
            await asyncio.sleep(2)

# Wrap for sync use
def debate_sync(question: str) -> str:
    return asyncio.run(async_debate(question))

debate_tool = FunctionTool.from_defaults(fn=debate_sync, name="debate")
```

## Best Practices

1. **Use Debate for Decisions**: RAG for facts, debate for opinions/decisions
2. **Provide Context**: Include relevant retrieved context in debate questions
3. **Handle Timeouts**: Debates can take 30-60 seconds; handle gracefully
4. **Cache Results**: Cache debate results for similar questions

## Related Resources

- [Aragora Python SDK](https://docs.aragora.ai/sdk/python)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [API Reference](https://docs.aragora.ai/api)
