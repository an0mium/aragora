"""
Base utilities for creating agents.
"""

from typing import Literal, Union

AgentType = Literal[
    # CLI-based
    "codex", "claude", "openai", "gemini-cli", "grok-cli", "qwen-cli", "deepseek-cli",
    # API-based
    "gemini", "ollama", "anthropic-api", "openai-api",
]


def create_agent(
    model_type: AgentType,
    name: str = None,
    role: str = "proposer",
    model: str = None,
    api_key: str = None,
):
    """
    Factory function to create agents by type.

    Args:
        model_type: Type of agent to create:
            - "codex": OpenAI Codex CLI
            - "claude": Claude CLI (claude-code)
            - "openai": OpenAI CLI
            - "gemini-cli": Google Gemini CLI
            - "grok-cli": xAI Grok CLI
            - "qwen-cli": Alibaba Qwen Code CLI
            - "deepseek-cli": Deepseek CLI
            - "gemini": Google Gemini API
            - "ollama": Local Ollama API
            - "anthropic-api": Direct Anthropic API
            - "openai-api": Direct OpenAI API
        name: Agent name (defaults to model_type)
        role: Agent role ("proposer", "critic", "synthesizer")
        model: Specific model to use (optional)
        api_key: API key for API-based agents (optional, uses env vars)

    Returns:
        Agent instance
    """
    # CLI-based agents
    if model_type == "codex":
        from aagora.agents.cli_agents import CodexAgent
        return CodexAgent(
            name=name or "codex",
            model=model or "gpt-5.2-codex",
            role=role,
        )
    elif model_type == "claude":
        from aagora.agents.cli_agents import ClaudeAgent
        return ClaudeAgent(
            name=name or "claude",
            model=model or "claude-sonnet-4",
            role=role,
        )
    elif model_type == "openai":
        from aagora.agents.cli_agents import OpenAIAgent
        return OpenAIAgent(
            name=name or "openai",
            model=model or "gpt-4o",
            role=role,
        )
    elif model_type == "gemini-cli":
        from aagora.agents.cli_agents import GeminiCLIAgent
        return GeminiCLIAgent(
            name=name or "gemini",
            model=model or "gemini-2.5-pro",
            role=role,
        )
    elif model_type == "grok-cli":
        from aagora.agents.cli_agents import GrokCLIAgent
        return GrokCLIAgent(
            name=name or "grok",
            model=model or "grok-3-latest",
            role=role,
        )
    elif model_type == "qwen-cli":
        from aagora.agents.cli_agents import QwenCLIAgent
        return QwenCLIAgent(
            name=name or "qwen",
            model=model or "qwen3-coder",
            role=role,
        )
    elif model_type == "deepseek-cli":
        from aagora.agents.cli_agents import DeepseekCLIAgent
        return DeepseekCLIAgent(
            name=name or "deepseek",
            model=model or "deepseek-v3",
            role=role,
        )

    # API-based agents
    elif model_type == "gemini":
        from aagora.agents.api_agents import GeminiAgent
        return GeminiAgent(
            name=name or "gemini",
            model=model or "gemini-2.0-flash-exp",
            role=role,
            api_key=api_key,
        )
    elif model_type == "ollama":
        from aagora.agents.api_agents import OllamaAgent
        return OllamaAgent(
            name=name or "ollama",
            model=model or "llama3.2",
            role=role,
        )
    elif model_type == "anthropic-api":
        from aagora.agents.api_agents import AnthropicAPIAgent
        return AnthropicAPIAgent(
            name=name or "claude-api",
            model=model or "claude-sonnet-4-20250514",
            role=role,
            api_key=api_key,
        )
    elif model_type == "openai-api":
        from aagora.agents.api_agents import OpenAIAPIAgent
        return OpenAIAPIAgent(
            name=name or "openai-api",
            model=model or "gpt-4o",
            role=role,
            api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Valid types: codex, claude, openai, gemini-cli, grok-cli, qwen-cli, deepseek-cli, gemini, ollama, anthropic-api, openai-api"
        )


def list_available_agents() -> dict:
    """List all available agent types and their requirements."""
    return {
        "codex": {
            "type": "CLI",
            "requires": "codex CLI (npm install -g @openai/codex)",
            "env_vars": None,
        },
        "claude": {
            "type": "CLI",
            "requires": "claude CLI (npm install -g @anthropic-ai/claude-code)",
            "env_vars": None,
        },
        "openai": {
            "type": "CLI",
            "requires": "openai CLI (pip install openai)",
            "env_vars": "OPENAI_API_KEY",
        },
        "gemini-cli": {
            "type": "CLI",
            "requires": "gemini CLI (npm install -g @google/gemini-cli)",
            "env_vars": None,
        },
        "grok-cli": {
            "type": "CLI",
            "requires": "grok CLI (npm install -g grok-cli)",
            "env_vars": None,
        },
        "qwen-cli": {
            "type": "CLI",
            "requires": "qwen CLI (npm install -g @qwen-code/qwen-code)",
            "env_vars": None,
        },
        "deepseek-cli": {
            "type": "CLI",
            "requires": "deepseek CLI (pip install deepseek-cli)",
            "env_vars": "DEEPSEEK_API_KEY",
        },
        "gemini": {
            "type": "API",
            "requires": None,
            "env_vars": "GEMINI_API_KEY or GOOGLE_API_KEY",
        },
        "ollama": {
            "type": "API",
            "requires": "Ollama running locally (brew install ollama && ollama serve)",
            "env_vars": "OLLAMA_HOST (optional, defaults to localhost:11434)",
        },
        "anthropic-api": {
            "type": "API",
            "requires": None,
            "env_vars": "ANTHROPIC_API_KEY",
        },
        "openai-api": {
            "type": "API",
            "requires": None,
            "env_vars": "OPENAI_API_KEY",
        },
    }
