"""
aagora (Agent Agora): A Multi-Agent Debate Framework

A society of heterogeneous AI agents that discuss, critique, improve
each other's responses, and learn from successful patterns.

Inspired by:
- Stanford Generative Agents (memory + reflection)
- ChatArena (game environments)
- LLM Multi-Agent Debate (consensus mechanisms)
- UniversalBackrooms (multi-model conversations)
- Project Sid (emergent civilization)
"""

from aagora.core import Agent, Critique, DebateResult, Environment
from aagora.debate.orchestrator import Arena, DebateProtocol
from aagora.memory.store import CritiqueStore
from aagora.memory.embeddings import SemanticRetriever
from aagora.evolution.evolver import PromptEvolver, EvolutionStrategy

__version__ = "0.2.0"
__all__ = [
    # Core
    "Agent",
    "Critique",
    "DebateResult",
    "Environment",
    # Debate
    "Arena",
    "DebateProtocol",
    # Memory
    "CritiqueStore",
    "SemanticRetriever",
    # Evolution
    "PromptEvolver",
    "EvolutionStrategy",
]
