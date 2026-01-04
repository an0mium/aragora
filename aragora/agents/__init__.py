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
)
from aragora.agents.api_agents import (
    GeminiAgent,
    OllamaAgent,
    AnthropicAPIAgent,
    OpenAIAPIAgent,
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

__all__ = [
    # CLI-based
    "CodexAgent",
    "ClaudeAgent",
    "OpenAIAgent",
    "GeminiCLIAgent",
    "GrokCLIAgent",
    "QwenCLIAgent",
    "DeepseekCLIAgent",
    # API-based
    "GeminiAgent",
    "OllamaAgent",
    "AnthropicAPIAgent",
    "OpenAIAPIAgent",
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
]
