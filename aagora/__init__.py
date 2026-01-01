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
from aagora.debate.meta import MetaCritiqueAnalyzer, MetaCritique, MetaObservation
from aagora.debate.forking import (
    DebateForker,
    ForkDetector,
    Branch,
    ForkDecision,
    ForkPoint,
    MergeResult,
)
from aagora.memory.store import CritiqueStore
from aagora.memory.embeddings import SemanticRetriever
from aagora.memory.streams import MemoryStream, Memory, RetrievedMemory
from aagora.evolution.evolver import PromptEvolver, EvolutionStrategy
from aagora.agents.personas import Persona, PersonaManager, PerformanceRecord
from aagora.ranking import EloSystem, AgentRating, MatchResult
from aagora.tournaments import (
    Tournament,
    TournamentFormat,
    TournamentTask,
    TournamentMatch,
    TournamentStanding,
    TournamentResult,
    create_default_tasks,
)

__version__ = "0.5.0"
__all__ = [
    # Core
    "Agent",
    "Critique",
    "DebateResult",
    "Environment",
    # Debate Orchestration
    "Arena",
    "DebateProtocol",
    # Meta-Critique
    "MetaCritiqueAnalyzer",
    "MetaCritique",
    "MetaObservation",
    # Debate Forking
    "DebateForker",
    "ForkDetector",
    "Branch",
    "ForkDecision",
    "ForkPoint",
    "MergeResult",
    # Memory
    "CritiqueStore",
    "SemanticRetriever",
    "MemoryStream",
    "Memory",
    "RetrievedMemory",
    # Evolution
    "PromptEvolver",
    "EvolutionStrategy",
    # Personas
    "Persona",
    "PersonaManager",
    "PerformanceRecord",
    # Ranking
    "EloSystem",
    "AgentRating",
    "MatchResult",
    # Tournaments
    "Tournament",
    "TournamentFormat",
    "TournamentTask",
    "TournamentMatch",
    "TournamentStanding",
    "TournamentResult",
    "create_default_tasks",
]
