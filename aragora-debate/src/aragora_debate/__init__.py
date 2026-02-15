"""
aragora-debate: Adversarial multi-model debate engine with decision receipts.

Run structured adversarial debates across multiple LLM providers,
detect consensus, track dissent, and produce cryptographic decision receipts.

Quick start (5-line API)::

    from aragora_debate import Debate, create_agent

    debate = Debate(topic="Should we migrate to microservices?")
    debate.add_agent(create_agent("anthropic", model="claude-sonnet-4-5-20250929"))
    debate.add_agent(create_agent("openai", model="gpt-4o"))
    result = await debate.run()
    print(result.receipt.to_markdown())

Advanced usage::

    from aragora_debate import Arena, Agent, DebateConfig

    agents = [MyAgent("claude"), MyAgent("gpt4")]
    arena = Arena(question="Should we migrate?", agents=agents)
    result = await arena.run()
    print(result.receipt.to_markdown())
"""

# --- High-level API (recommended) ---
from aragora_debate.debate import Debate, create_agent

# --- Core types ---
from aragora_debate.types import (
    Agent,
    AgentResponse,
    Claim,
    Consensus,
    ConsensusMethod,
    Critique,
    DebateConfig,
    DebateResult,
    DecisionReceipt,
    DissentRecord,
    Evidence,
    Message,
    Phase,
    Proposal,
    Verdict,
    Vote,
)

# --- Advanced API ---
from aragora_debate.arena import Arena
from aragora_debate.receipt import ReceiptBuilder

# --- Analysis & Detection ---
from aragora_debate.evidence import (
    EvidenceQualityAnalyzer,
    EvidenceQualityScore,
    HollowConsensusAlert,
    HollowConsensusDetector,
)
from aragora_debate.convergence import ConvergenceDetector, ConvergenceResult
from aragora_debate.events import EventEmitter, EventType, DebateEvent
from aragora_debate.trickster import (
    EvidencePoweredTrickster,
    TricksterConfig,
    TricksterIntervention,
)
from aragora_debate.cross_analysis import CrossProposalAnalyzer, CrossProposalAnalysis

# --- Mock agents (always available) ---
from aragora_debate._mock import MockAgent
from aragora_debate.styled_mock import StyledMockAgent

__version__ = "0.2.0"

# Optional provider agents (require extra dependencies)
try:
    from aragora_debate.agents import ClaudeAgent
except ImportError:
    ClaudeAgent = None  # type: ignore[assignment,misc]

try:
    from aragora_debate.agents import OpenAIAgent
except ImportError:
    OpenAIAgent = None  # type: ignore[assignment,misc]

try:
    from aragora_debate.agents import MistralAgent
except ImportError:
    MistralAgent = None  # type: ignore[assignment,misc]

try:
    from aragora_debate.agents import GeminiAgent
except ImportError:
    GeminiAgent = None  # type: ignore[assignment,misc]

__all__ = [
    # High-level API
    "Debate",
    "create_agent",
    # Core types
    "Agent",
    "AgentResponse",
    "Arena",
    "Claim",
    "ClaudeAgent",
    "Consensus",
    "ConsensusMethod",
    "Critique",
    "DebateConfig",
    "DebateResult",
    "DecisionReceipt",
    "DissentRecord",
    "Evidence",
    "Message",
    "MockAgent",
    "StyledMockAgent",
    "OpenAIAgent",
    "MistralAgent",
    "GeminiAgent",
    "Phase",
    "Proposal",
    "ReceiptBuilder",
    "Verdict",
    "Vote",
    # Analysis & Detection
    "EvidenceQualityAnalyzer",
    "EvidenceQualityScore",
    "HollowConsensusAlert",
    "HollowConsensusDetector",
    "ConvergenceDetector",
    "ConvergenceResult",
    "EventEmitter",
    "EventType",
    "DebateEvent",
    "EvidencePoweredTrickster",
    "TricksterConfig",
    "TricksterIntervention",
    "CrossProposalAnalyzer",
    "CrossProposalAnalysis",
]
