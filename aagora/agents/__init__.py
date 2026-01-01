"""
Agent implementations for various AI models.

Supports both CLI-based agents (codex, claude) and API-based agents
(Gemini, Ollama, direct OpenAI/Anthropic APIs).
"""

from aagora.agents.cli_agents import CodexAgent, ClaudeAgent, OpenAIAgent
from aagora.agents.api_agents import (
    GeminiAgent,
    OllamaAgent,
    AnthropicAPIAgent,
    OpenAIAPIAgent,
)
from aagora.agents.base import create_agent

__all__ = [
    # CLI-based
    "CodexAgent",
    "ClaudeAgent",
    "OpenAIAgent",
    # API-based
    "GeminiAgent",
    "OllamaAgent",
    "AnthropicAPIAgent",
    "OpenAIAPIAgent",
    # Factory
    "create_agent",
]
