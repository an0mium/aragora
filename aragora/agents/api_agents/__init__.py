"""
API-based agent implementations.

These agents call AI APIs directly (HTTP), enabling use without CLI tools.
Supports Gemini, Ollama (local), and direct OpenAI/Anthropic API calls.
"""

from __future__ import annotations

# Re-export all agents for backward compatibility
from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.gemini import GeminiAgent
from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
from aragora.agents.api_agents.openai import OpenAIAPIAgent
from aragora.agents.api_agents.grok import GrokAgent
from aragora.agents.api_agents.ollama import OllamaAgent
from aragora.agents.api_agents.lm_studio import LMStudioAgent
from aragora.agents.api_agents.openrouter import (
    OpenRouterAgent,
    DeepSeekAgent,
    DeepSeekReasonerAgent,
    DeepSeekV3Agent,
    LlamaAgent,
    MistralAgent,
    QwenAgent,
    QwenMaxAgent,
    YiAgent,
    KimiAgent,
)
from aragora.agents.api_agents.mistral import MistralAPIAgent, CodestralAgent
from aragora.agents.api_agents.tinker import (
    TinkerAgent,
    TinkerLlamaAgent,
    TinkerQwenAgent,
    TinkerDeepSeekAgent,
)
from aragora.agents.api_agents.rate_limiter import (
    OpenRouterRateLimiter,
    OpenRouterTier,
    OPENROUTER_TIERS,
    get_openrouter_limiter,
    set_openrouter_tier,
)
from aragora.agents.api_agents.common import MAX_STREAM_BUFFER_SIZE
from aragora.agents.api_agents.openai_compatible import OpenAICompatibleMixin

__all__ = [
    # Base classes
    "APIAgent",
    "OpenAICompatibleMixin",
    # Provider agents
    "GeminiAgent",
    "AnthropicAPIAgent",
    "OpenAIAPIAgent",
    "GrokAgent",
    "OllamaAgent",
    "LMStudioAgent",
    # Mistral direct API
    "MistralAPIAgent",
    "CodestralAgent",
    # OpenRouter and subclasses
    "OpenRouterAgent",
    "DeepSeekAgent",
    "DeepSeekReasonerAgent",
    "DeepSeekV3Agent",
    "LlamaAgent",
    "MistralAgent",
    "QwenAgent",
    "QwenMaxAgent",
    "YiAgent",
    "KimiAgent",
    # Tinker (fine-tuned models)
    "TinkerAgent",
    "TinkerLlamaAgent",
    "TinkerQwenAgent",
    "TinkerDeepSeekAgent",
    # Rate limiting
    "OpenRouterRateLimiter",
    "OpenRouterTier",
    "OPENROUTER_TIERS",
    "get_openrouter_limiter",
    "set_openrouter_tier",
    # Constants
    "MAX_STREAM_BUFFER_SIZE",
]
