"""
Agent implementations for various AI models.

Supports both CLI-based agents (codex, claude) and API-based agents
(Gemini, Ollama, direct OpenAI/Anthropic APIs).

Also includes persona management and the Emergent Persona Laboratory
for evolving agent specializations.
"""

from aragora.agents.cli_agents import (
    CodexAgent,
    ClaudeAgent,
    OpenAIAgent,
    GeminiCLIAgent,
    GrokCLIAgent,
    QwenCLIAgent,
    DeepseekCLIAgent,
    KiloCodeAgent,
)
from aragora.agents.demo_agent import DemoAgent
from aragora.agents.api_agents import (
    GeminiAgent,
    OllamaAgent,
    LMStudioAgent,
    AnthropicAPIAgent,
    OpenAIAPIAgent,
    GrokAgent,
    OpenRouterAgent,
    DeepSeekAgent,
    DeepSeekReasonerAgent,
    DeepSeekV3Agent,
    LlamaAgent,
    MistralAgent,
)
from aragora.agents.local_llm_detector import (
    LocalLLMDetector,
    LocalLLMServer,
    LocalLLMStatus,
    detect_local_llms,
    detect_local_llms_sync,
)
from aragora.agents.base import create_agent
from aragora.agents.personas import Persona, PersonaManager, EXPERTISE_DOMAINS, PERSONALITY_TRAITS
from aragora.agents.laboratory import (
    PersonaLaboratory,
    PersonaExperiment,
    EmergentTrait,
    TraitTransfer,
)
from aragora.agents.calibration import (
    CalibrationTracker,
    CalibrationBucket,
    CalibrationSummary,
)
from aragora.agents.fallback import (
    AgentFallbackChain,
    AllProvidersExhaustedError,
    FallbackMetrics,
    QuotaFallbackMixin,
    QUOTA_ERROR_KEYWORDS,
)
from aragora.agents.airlock import (
    AirlockProxy,
    AirlockConfig,
    AirlockMetrics,
    wrap_agent,
    wrap_agents,
)
from aragora.agents.telemetry import (
    AgentTelemetry,
    with_telemetry,
    TelemetryContext,
    register_telemetry_collector,
    unregister_telemetry_collector,
    setup_default_collectors,
    get_telemetry_stats,
    reset_telemetry,
)
from aragora.agents.performance_monitor import (
    AgentPerformanceMonitor,
    AgentMetric,
    AgentStats,
)

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
]
