"""
aragora-debate: Adversarial multi-model debate engine with decision receipts.

Run structured adversarial debates across multiple LLM providers,
detect consensus, track dissent, and produce cryptographic decision receipts.

Quick start::

    from aragora_debate import Arena, Agent, DebateConfig

    agents = [MyAgent("claude"), MyAgent("gpt4")]
    arena = Arena(
        question="Should we migrate to microservices?",
        agents=agents,
    )
    result = await arena.run()
    print(result.receipt.to_markdown())
"""

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
    Vote,
)
from aragora_debate.arena import Arena
from aragora_debate.receipt import ReceiptBuilder

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResponse",
    "Arena",
    "Claim",
    "Consensus",
    "ConsensusMethod",
    "Critique",
    "DebateConfig",
    "DebateResult",
    "DecisionReceipt",
    "DissentRecord",
    "Evidence",
    "Message",
    "Phase",
    "Proposal",
    "ReceiptBuilder",
    "Vote",
]
