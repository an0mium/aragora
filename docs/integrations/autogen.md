# AutoGen Integration Guide

Integrate Aragora's multi-agent debate capabilities with Microsoft AutoGen.

## Installation

```bash
pip install aragora pyautogen
```

## Basic Integration

### Aragora as an AutoGen Agent

```python
from autogen import ConversableAgent, UserProxyAgent
from aragora import AragoraClientSync

class AragoraDebateAgent(ConversableAgent):
    """AutoGen agent that uses Aragora for multi-perspective analysis."""

    def __init__(self, name: str, aragora_api_key: str, **kwargs):
        super().__init__(
            name=name,
            system_message="""You are a debate coordinator that uses multi-agent AI debates
            to provide balanced perspectives on complex questions.""",
            **kwargs
        )
        self.aragora = AragoraClientSync(
            base_url="https://api.aragora.ai",
            api_key=aragora_api_key
        )

    def generate_reply(self, messages=None, sender=None, **kwargs):
        # Extract the question from the last message
        if messages:
            question = messages[-1].get("content", "")
        else:
            return "I need a question to debate."

        # Run the debate
        try:
            result = self.aragora.create_debate(
                question=question,
                agents=["claude", "gpt-4", "gemini"],
                rounds=3
            )

            # Wait for completion
            import time
            for _ in range(60):  # Max 2 minutes
                debate = self.aragora.get_debate(result["debate_id"])
                if debate["status"] == "completed":
                    break
                time.sleep(2)

            consensus = debate.get("consensus", {})
            return f"""
## Multi-Agent Debate Results

**Question:** {question}

**Consensus:** {consensus.get("final_answer", "No consensus reached")}

**Confidence:** {consensus.get("confidence", 0):.0%}

**Method:** {consensus.get("method", "unknown")}

**Debate ID:** {result["debate_id"]}
"""
        except Exception as e:
            return f"Debate failed: {str(e)}"

# Usage
debate_agent = AragoraDebateAgent(
    name="Debate Coordinator",
    aragora_api_key="your-aragora-api-key",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=False
)

user.initiate_chat(
    debate_agent,
    message="Should we use Kubernetes or Docker Swarm for our infrastructure?"
)
```

### Multi-Agent Collaboration with Debate Validation

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from aragora import AragoraClientSync

# Initialize Aragora
aragora = AragoraClientSync(api_key="your-key")

# Define agents
architect = AssistantAgent(
    name="Architect",
    system_message="You are a software architect. Propose system designs.",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

security_expert = AssistantAgent(
    name="SecurityExpert",
    system_message="You are a security expert. Review designs for vulnerabilities.",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

def validate_with_debate(proposal: str) -> str:
    """Validate a proposal through Aragora debate."""
    result = aragora.create_debate(
        question=f"Evaluate this proposal for potential issues: {proposal}",
        agents=["claude", "gpt-4"],
        rounds=2
    )

    import time
    while True:
        debate = aragora.get_debate(result["debate_id"])
        if debate["status"] == "completed":
            return debate.get("consensus", {}).get("final_answer", "")
        time.sleep(2)

validator = AssistantAgent(
    name="Validator",
    system_message="""You validate proposals using multi-agent debate.
    Call the validate_with_debate function with the proposal text.""",
    llm_config={"config_list": [{"model": "gpt-4"}]},
    function_map={"validate_with_debate": validate_with_debate}
)

user = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    code_execution_config=False
)

# Create group chat
group_chat = GroupChat(
    agents=[user, architect, security_expert, validator],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

# Start the conversation
user.initiate_chat(
    manager,
    message="Design an authentication system for our new microservices architecture."
)
```

### Aragora-Powered Decision Making

```python
from autogen import AssistantAgent, UserProxyAgent
from aragora import AragoraClientSync
import json

aragora = AragoraClientSync(api_key="your-key")

def get_decision_receipt(question: str, options: list[str]) -> dict:
    """Get a formal decision receipt from Aragora."""
    formatted_question = f"""
    Decision needed: {question}

    Options to evaluate:
    {json.dumps(options, indent=2)}

    Analyze each option and recommend the best choice with reasoning.
    """

    result = aragora.create_debate(
        question=formatted_question,
        agents=["claude", "gpt-4", "gemini"],
        rounds=3
    )

    import time
    while True:
        debate = aragora.get_debate(result["debate_id"])
        if debate["status"] == "completed":
            break
        time.sleep(2)

    return {
        "debate_id": result["debate_id"],
        "decision": debate.get("consensus", {}).get("final_answer"),
        "confidence": debate.get("consensus", {}).get("confidence"),
        "verification_url": f"https://aragora.ai/debates/{result['debate_id']}"
    }

decision_agent = AssistantAgent(
    name="DecisionMaker",
    system_message="""You help make important decisions using multi-agent debate.
    Use the get_decision_receipt function to get validated decisions.""",
    llm_config={"config_list": [{"model": "gpt-4"}]},
    function_map={"get_decision_receipt": get_decision_receipt}
)

user = UserProxyAgent(name="User", human_input_mode="NEVER")

user.initiate_chat(
    decision_agent,
    message="""We need to choose a database for our new project. Options are:
    1. PostgreSQL - mature, ACID compliant
    2. MongoDB - flexible schema, good for documents
    3. DynamoDB - managed, scales automatically
    4. CockroachDB - distributed, PostgreSQL compatible

    Our requirements: high availability, moderate scale (1M users), complex queries"""
)
```

## Best Practices

1. **Use for Complex Decisions**: AutoGen agents for routine work, Aragora for important decisions
2. **Set Timeouts**: Debates take time; configure appropriate timeouts
3. **Cache Results**: Store debate IDs for audit trails
4. **Combine Perspectives**: Use AutoGen's multi-agent + Aragora's debate for comprehensive analysis

## Related Resources

- [Aragora Python SDK](https://docs.aragora.ai/sdk/python)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [API Reference](https://docs.aragora.ai/api)
