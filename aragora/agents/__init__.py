"""
Agent implementations for various AI models.

Supports both CLI-based agents (codex, claude) and API-based agents
(Gemini, Ollama, direct OpenAI/Anthropic APIs).

Also includes persona management and the Emergent Persona Laboratory
for evolving agent specializations.
"""

from aragora.agents.airlock import (
    AirlockConfig,
    AirlockMetrics,
    AirlockProxy,
    wrap_agent,
    wrap_agents,
)
from aragora.agents.config_loader import (
    AgentConfig,
    AgentConfigLoader,
    ConfigValidationError,
    load_agent_configs,
)
from aragora.agents.api_agents import (
    AnthropicAPIAgent,
    DeepSeekAgent,
    DeepSeekReasonerAgent,
    DeepSeekV3Agent,
    GeminiAgent,
    GrokAgent,
    LlamaAgent,
    LMStudioAgent,
    MistralAgent,
    OllamaAgent,
    OpenAIAPIAgent,
    OpenRouterAgent,
)
from aragora.agents.base import create_agent, list_available_agents
from aragora.agents.calibration import (
    CalibrationBucket,
    CalibrationSummary,
    CalibrationTracker,
)
from aragora.agents.cli_agents import (
    ClaudeAgent,
    CodexAgent,
    DeepseekCLIAgent,
    GeminiCLIAgent,
    GrokCLIAgent,
    KiloCodeAgent,
    OpenAIAgent,
    QwenCLIAgent,
)
from aragora.agents.demo_agent import DemoAgent
from aragora.agents.fallback import (
    QUOTA_ERROR_KEYWORDS,
    AgentFallbackChain,
    AllProvidersExhaustedError,
    FallbackMetrics,
    QuotaFallbackMixin,
)
from aragora.agents.laboratory import (
    EmergentTrait,
    PersonaExperiment,
    PersonaLaboratory,
    TraitTransfer,
)
from aragora.agents.local_llm_detector import (
    LocalLLMDetector,
    LocalLLMServer,
    LocalLLMStatus,
    detect_local_llms,
    detect_local_llms_sync,
)
from aragora.agents.performance_monitor import (
    AgentMetric,
    AgentPerformanceMonitor,
    AgentStats,
)
from aragora.agents.personas import EXPERTISE_DOMAINS, PERSONALITY_TRAITS, Persona, PersonaManager
from aragora.agents.registry import AgentRegistry, register_all_agents
from aragora.agents.telemetry import (
    AgentTelemetry,
    TelemetryContext,
    get_telemetry_stats,
    register_telemetry_collector,
    reset_telemetry,
    setup_default_collectors,
    unregister_telemetry_collector,
    with_telemetry,
)


def get_agents_by_names(names: list[str]) -> list:
    """Get agent instances by their type names.

    Creates agent instances for each valid name in the list.
    Invalid names are silently skipped.

    Args:
        names: List of agent type names (e.g., ["anthropic-api", "openai-api"])

    Returns:
        List of agent instances for valid names

    Example:
        >>> agents = get_agents_by_names(["anthropic-api", "openai-api"])
        >>> len(agents)
        2
    """
    # Ensure all agents are registered
    register_all_agents()

    agents = []
    for name in names:
        try:
            if AgentRegistry.is_registered(name):
                agent = AgentRegistry.create(name)
                agents.append(agent)
        except (ValueError, ImportError, RuntimeError):
            # Skip invalid agent names
            pass
    return agents


__all__ = [
    # CLI-based
    "CodexAgent",
    "ClaudeAgent",
    "OpenAIAgent",
    "GeminiCLIAgent",
    "GrokCLIAgent",
    "QwenCLIAgent",
    "DeepseekCLIAgent",
    "KiloCodeAgent",
    # Built-in
    "DemoAgent",
    # API-based (direct)
    "GeminiAgent",
    "OllamaAgent",
    "LMStudioAgent",
    "AnthropicAPIAgent",
    "OpenAIAPIAgent",
    "GrokAgent",
    # API-based (OpenRouter)
    "OpenRouterAgent",
    "DeepSeekAgent",
    "DeepSeekReasonerAgent",
    "DeepSeekV3Agent",
    "LlamaAgent",
    "MistralAgent",
    # Factory
    "create_agent",
    "get_agents_by_names",
    "list_available_agents",
    "AgentRegistry",
    "register_all_agents",
    # Personas
    "Persona",
    "PersonaManager",
    "EXPERTISE_DOMAINS",
    "PERSONALITY_TRAITS",
    # Laboratory
    "PersonaLaboratory",
    "PersonaExperiment",
    "EmergentTrait",
    "TraitTransfer",
    # Calibration
    "CalibrationTracker",
    "CalibrationBucket",
    "CalibrationSummary",
    # Fallback
    "AgentFallbackChain",
    "AllProvidersExhaustedError",
    "FallbackMetrics",
    "QuotaFallbackMixin",
    "QUOTA_ERROR_KEYWORDS",
    # Airlock (resilience)
    "AirlockProxy",
    "AirlockConfig",
    "AirlockMetrics",
    "wrap_agent",
    "wrap_agents",
    # Telemetry
    "AgentTelemetry",
    "with_telemetry",
    "TelemetryContext",
    "register_telemetry_collector",
    "unregister_telemetry_collector",
    "setup_default_collectors",
    "get_telemetry_stats",
    "reset_telemetry",
    # Performance Monitor
    "AgentPerformanceMonitor",
    "AgentMetric",
    "AgentStats",
    # Local LLM Detection
    "LocalLLMDetector",
    "LocalLLMServer",
    "LocalLLMStatus",
    "detect_local_llms",
    "detect_local_llms_sync",
    # YAML Configuration
    "AgentConfig",
    "AgentConfigLoader",
    "ConfigValidationError",
    "load_agent_configs",
    # Email Agents
    "SenderReputationAgent",
    "ContentUrgencyAgent",
    "ContextRelevanceAgent",
    "BillingCriticalityAgent",
    "TimelineAgent",
    "get_email_agent_team",
]

# Email-specific agents for inbox prioritization
from aragora.agents.email_agents import (
    SenderReputationAgent,
    ContentUrgencyAgent,
    ContextRelevanceAgent,
    BillingCriticalityAgent,
    TimelineAgent,
    get_email_agent_team,
)
