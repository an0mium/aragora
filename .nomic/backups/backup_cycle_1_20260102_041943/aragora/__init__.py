"""
aragora (Agent Agora): A Multi-Agent Debate Framework

A society of heterogeneous AI agents that discuss, critique, improve
each other's responses, and learn from successful patterns.

Features (v0.7.0):
- Multi-agent debate with propose/critique/synthesize protocol
- CLI agents: claude, codex, gemini, grok, qwen, deepseek
- Memory streams for persistent agent memory
- ELO ranking and tournament systems
- Debate forking when agents disagree
- Meta-critique for process analysis
- Red-team mode for adversarial testing
- Human intervention breakpoints
- Domain-specific debate templates
- Adaptive agent selection
- Self-improvement mode for code editing
- GitHub Action for PR reviews
- Graph-based debates with counterfactual branching
- Evidence provenance chains
- Consensus memory with dissent retrieval
- Scenario matrix debates
- Executable verification proofs

Inspired by:
- Stanford Generative Agents (memory + reflection)
- ChatArena (game environments)
- LLM Multi-Agent Debate (consensus mechanisms)
- UniversalBackrooms (multi-model conversations)
- Project Sid (emergent civilization)
"""

# Core
from aragora.core import Agent, Critique, DebateResult, Environment

# Debate Orchestration
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.meta import MetaCritiqueAnalyzer, MetaCritique, MetaObservation
from aragora.debate.forking import (
    DebateForker,
    ForkDetector,
    Branch,
    ForkDecision,
    ForkPoint,
    MergeResult,
)
from aragora.debate.traces import DebateTracer, DebateReplayer, DebateTrace, TraceEvent
from aragora.debate.consensus import ConsensusProof, ConsensusBuilder, Claim, Evidence
from aragora.debate.breakpoints import BreakpointManager, Breakpoint, HumanGuidance
from aragora.debate.graph import (
    DebateGraph,
    DebateNode,
    BranchPolicy,
    BranchReason,
    MergeStrategy,
    ConvergenceScorer,
    GraphReplayBuilder,
    GraphDebateOrchestrator,
    NodeType,
)
from aragora.debate.scenarios import (
    ScenarioMatrix,
    Scenario,
    ScenarioType,
    ScenarioResult,
    MatrixResult,
    MatrixDebateRunner,
    OutcomeCategory,
)

# Memory
from aragora.memory.store import CritiqueStore
from aragora.memory.embeddings import SemanticRetriever
from aragora.memory.streams import MemoryStream, Memory, RetrievedMemory
from aragora.memory.consensus import (
    ConsensusMemory,
    ConsensusRecord,
    ConsensusStrength,
    DissentRecord,
    DissentType,
    DissentRetriever,
)

# Evolution
from aragora.evolution.evolver import PromptEvolver, EvolutionStrategy

# Agents
from aragora.agents.personas import Persona, PersonaManager
from aragora.agents.laboratory import (
    PersonaLaboratory,
    PersonaExperiment,
    EmergentTrait,
    TraitTransfer,
)

# Ranking
from aragora.ranking import EloSystem, AgentRating, MatchResult

# Tournaments
from aragora.tournaments import (
    Tournament,
    TournamentFormat,
    TournamentTask,
    TournamentMatch,
    TournamentStanding,
    TournamentResult,
    create_default_tasks,
)

# Reasoning
from aragora.reasoning import (
    ClaimsKernel,
    TypedClaim,
    TypedEvidence,
    ClaimType,
    ProvenanceManager,
    ProvenanceChain,
    ProvenanceRecord,
    CitationGraph,
    SourceType,
)

# Modes
from aragora.modes import RedTeamMode, RedTeamResult, Attack, AttackType

# Tools
from aragora.tools import CodeReader, CodeWriter, SelfImprover, CodeProposal

# Routing
from aragora.routing import AgentSelector, AgentProfile, TaskRequirements, TeamComposition

# Templates
from aragora.templates import (
    DebateTemplate,
    TemplateType,
    CODE_REVIEW_TEMPLATE,
    DESIGN_DOC_TEMPLATE,
    INCIDENT_RESPONSE_TEMPLATE,
    RESEARCH_SYNTHESIS_TEMPLATE,
    get_template,
    list_templates,
)

# Verification
from aragora.verification import (
    VerificationProof,
    ProofType,
    ProofStatus,
    VerificationResult,
    ProofExecutor,
    ClaimVerifier,
    VerificationReport,
    ProofBuilder,
    # Formal verification (stub interface for Lean/Z3)
    FormalVerificationBackend,
    FormalVerificationManager,
    FormalProofStatus,
    FormalLanguage,
)

__version__ = "0.07"
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
    # Traces
    "DebateTracer",
    "DebateReplayer",
    "DebateTrace",
    "TraceEvent",
    # Consensus
    "ConsensusProof",
    "ConsensusBuilder",
    "Claim",
    "Evidence",
    # Breakpoints
    "BreakpointManager",
    "Breakpoint",
    "HumanGuidance",
    # Graph-based Debates
    "DebateGraph",
    "DebateNode",
    "BranchPolicy",
    "BranchReason",
    "MergeStrategy",
    "ConvergenceScorer",
    "GraphReplayBuilder",
    "GraphDebateOrchestrator",
    "NodeType",
    # Scenario Matrix
    "ScenarioMatrix",
    "Scenario",
    "ScenarioType",
    "ScenarioResult",
    "MatrixResult",
    "MatrixDebateRunner",
    "OutcomeCategory",
    # Memory
    "CritiqueStore",
    "SemanticRetriever",
    "MemoryStream",
    "Memory",
    "RetrievedMemory",
    # Consensus Memory
    "ConsensusMemory",
    "ConsensusRecord",
    "ConsensusStrength",
    "DissentRecord",
    "DissentType",
    "DissentRetriever",
    # Evolution
    "PromptEvolver",
    "EvolutionStrategy",
    # Personas
    "Persona",
    "PersonaManager",
    # Persona Laboratory
    "PersonaLaboratory",
    "PersonaExperiment",
    "EmergentTrait",
    "TraitTransfer",
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
    # Reasoning
    "ClaimsKernel",
    "TypedClaim",
    "TypedEvidence",
    "ClaimType",
    # Provenance
    "ProvenanceManager",
    "ProvenanceChain",
    "ProvenanceRecord",
    "CitationGraph",
    "SourceType",
    # Modes
    "RedTeamMode",
    "RedTeamResult",
    "Attack",
    "AttackType",
    # Tools
    "CodeReader",
    "CodeWriter",
    "SelfImprover",
    "CodeProposal",
    # Routing
    "AgentSelector",
    "AgentProfile",
    "TaskRequirements",
    "TeamComposition",
    # Templates
    "DebateTemplate",
    "TemplateType",
    "CODE_REVIEW_TEMPLATE",
    "DESIGN_DOC_TEMPLATE",
    "INCIDENT_RESPONSE_TEMPLATE",
    "RESEARCH_SYNTHESIS_TEMPLATE",
    "get_template",
    "list_templates",
    # Verification
    "VerificationProof",
    "ProofType",
    "ProofStatus",
    "VerificationResult",
    "ProofExecutor",
    "ClaimVerifier",
    "VerificationReport",
    "ProofBuilder",
    # Formal Verification (stub interface)
    "FormalVerificationBackend",
    "FormalVerificationManager",
    "FormalProofStatus",
    "FormalLanguage",
]
