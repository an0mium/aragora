#!/usr/bin/env python3
"""
Nomic Loop: Autonomous self-improvement cycle for aragora.

Like a PCR machine for code evolution:
1. DEBATE: All agents propose improvements to aragora
2. CONSENSUS: Agents critique and refine until consensus
3. DESIGN: Agents design the implementation
4. IMPLEMENT: Agents write the code
5. VERIFY: Run tests, check quality
6. COMMIT: If verified, commit changes
7. REPEAT: Cycle continues

The dialectic tension between models (visionary vs pragmatic vs synthesizer)
creates emergent complexity and self-criticality.

Inspired by:
- Nomic (game where rules change the rules)
- Project Sid (emergent civilization)
- PCR (exponential amplification through cycles)
- Self-organized criticality (sandpile dynamics)

SAFETY: This file includes backup/restore mechanisms and safety prompts
to prevent the nomic loop from breaking itself.
"""

import asyncio
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Callable, Any
import traceback
import logging
from collections import defaultdict

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# AUTOMATION FLAGS - Environment variables for CI/automation support
# =============================================================================

# Auto-commit: Skip interactive commit prompt (default OFF - requires explicit opt-in)
NOMIC_AUTO_COMMIT = os.environ.get("NOMIC_AUTO_COMMIT", "0") == "1"

# Auto-continue: Skip interactive cycle continuation prompt (default ON for loops)
NOMIC_AUTO_CONTINUE = os.environ.get("NOMIC_AUTO_CONTINUE", "1") == "1"

# Cycle-level hard timeout in seconds (default 2 hours)
NOMIC_MAX_CYCLE_SECONDS = int(os.environ.get("NOMIC_MAX_CYCLE_SECONDS", "7200"))

# Stall detection threshold in seconds (default 30 minutes)
NOMIC_STALL_THRESHOLD = int(os.environ.get("NOMIC_STALL_THRESHOLD", "1800"))


# =============================================================================
# PHASE RECOVERY - Structured error handling for nomic loop phases
# =============================================================================


class PhaseError(Exception):
    """Exception raised when a phase fails."""
    def __init__(self, phase: str, message: str, recoverable: bool = True, original_error: Exception = None):
        self.phase = phase
        self.recoverable = recoverable
        self.original_error = original_error
        super().__init__(f"[{phase}] {message}")


class PhaseRecovery:
    """
    Structured error recovery for nomic loop phases.

    Features:
    - Per-phase retry with exponential backoff
    - Phase-specific error classification
    - Health metrics tracking
    - Automatic rollback triggers
    """

    # Default retry settings per phase
    PHASE_RETRY_CONFIG = {
        "context": {"max_retries": 2, "base_delay": 5, "critical": False},
        "debate": {"max_retries": 1, "base_delay": 10, "critical": True},
        "design": {"max_retries": 2, "base_delay": 5, "critical": False},
        "implement": {"max_retries": 1, "base_delay": 15, "critical": True},
        "verify": {"max_retries": 3, "base_delay": 5, "critical": False},
        "commit": {"max_retries": 1, "base_delay": 5, "critical": True},
    }

    # Individual phase timeouts (seconds) - complements cycle-level timeout
    PHASE_TIMEOUTS = {
        "context": 600,      # 10 min - codebase exploration
        "debate": 1800,      # 30 min - multi-agent discussion
        "design": 900,       # 15 min - architecture planning
        "implement": 2400,   # 40 min - code generation
        "verify": 600,       # 10 min - test execution
        "commit": 180,       # 3 min - git operations
    }

    # Errors that should NOT be retried
    NON_RETRYABLE_ERRORS = (
        KeyboardInterrupt,
        SystemExit,
        MemoryError,
    )

    # Errors that indicate rate limiting or service issues (should wait longer)
    # Keep in sync with aragora.agents.cli_agents.RATE_LIMIT_PATTERNS
    RATE_LIMIT_PATTERNS = [
        # Rate limiting
        "rate limit", "rate_limit", "ratelimit",
        "429", "too many requests", "throttl",
        # Quota/usage limit errors
        "quota exceeded", "quota_exceeded",
        "resource exhausted", "resource_exhausted",
        "insufficient_quota", "limit exceeded",
        "usage_limit", "usage limit",  # OpenAI/Codex usage limits
        "limit has been reached",
        # Billing errors
        "billing", "credit balance", "payment required",
        "purchase credits", "402",
        # Capacity/availability errors
        "503", "service unavailable",
        "502", "bad gateway",
        "overloaded", "capacity",
        "temporarily unavailable", "try again later",
        "server busy", "high demand",
        # API-specific errors
        "model overloaded", "model is currently overloaded",
        "engine is currently overloaded",
        # CLI-specific errors
        "argument list too long",  # E2BIG - prompt too large for CLI
        "broken pipe",  # EPIPE - connection closed unexpectedly
    ]

    def __init__(self, log_func: Callable = print):
        self.log = log_func
        self.phase_health: dict[str, dict] = {}
        self.consecutive_failures: dict[str, int] = {}

    def is_retryable(self, error: Exception, phase: str) -> bool:
        """Check if an error should be retried."""
        if isinstance(error, self.NON_RETRYABLE_ERRORS):
            return False

        # Check if phase has retries left
        config = self.PHASE_RETRY_CONFIG.get(phase, {"max_retries": 1})
        failures = self.consecutive_failures.get(phase, 0)

        if failures >= config["max_retries"]:
            return False

        return True

    def get_retry_delay(self, error: Exception, phase: str) -> float:
        """Calculate delay before retry with exponential backoff."""
        config = self.PHASE_RETRY_CONFIG.get(phase, {"base_delay": 5})
        base = config["base_delay"]
        failures = self.consecutive_failures.get(phase, 0)

        # Exponential backoff: base * 2^failures
        delay = base * (2 ** failures)

        # Check for rate limiting (use longer delay)
        error_str = str(error).lower()
        if any(pattern in error_str for pattern in self.RATE_LIMIT_PATTERNS):
            delay = max(delay, 120)  # Minimum 120s for rate limits
            self.log(f"  [recovery] Rate limit detected, waiting {delay}s")

        return min(delay, 300)  # Cap at 5 minutes

    def record_success(self, phase: str) -> None:
        """Record successful phase completion."""
        self.consecutive_failures[phase] = 0
        if phase not in self.phase_health:
            self.phase_health[phase] = {"successes": 0, "failures": 0, "last_error": None}
        self.phase_health[phase]["successes"] += 1

    def record_failure(self, phase: str, error: Exception) -> None:
        """Record phase failure."""
        self.consecutive_failures[phase] = self.consecutive_failures.get(phase, 0) + 1
        if phase not in self.phase_health:
            self.phase_health[phase] = {"successes": 0, "failures": 0, "last_error": None}
        self.phase_health[phase]["failures"] += 1
        self.phase_health[phase]["last_error"] = str(error)[:200]

    def should_trigger_rollback(self, phase: str) -> bool:
        """Check if failures warrant a rollback."""
        config = self.PHASE_RETRY_CONFIG.get(phase, {"critical": False})
        if not config["critical"]:
            return False

        # Rollback if critical phase has consecutive failures
        failures = self.consecutive_failures.get(phase, 0)
        return failures >= 2

    def get_health_report(self) -> dict:
        """Get health metrics for all phases."""
        return {
            "phase_health": self.phase_health,
            "consecutive_failures": self.consecutive_failures,
        }

    async def run_with_recovery(
        self,
        phase: str,
        phase_func: Callable,
        *args,
        **kwargs
    ) -> tuple[bool, Any]:
        """
        Run a phase function with automatic retry and recovery.

        Returns:
            (success: bool, result: Any or error message)
        """
        config = self.PHASE_RETRY_CONFIG.get(phase, {"max_retries": 1})
        attempts = 0

        while attempts <= config["max_retries"]:
            try:
                result = await phase_func(*args, **kwargs)
                self.record_success(phase)
                return (True, result)

            except self.NON_RETRYABLE_ERRORS:
                raise  # Don't catch these

            except Exception as e:
                attempts += 1
                self.record_failure(phase, e)

                error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                self.log(f"  [recovery] Phase '{phase}' attempt {attempts} failed: {error_msg}")

                if self.is_retryable(e, phase) and attempts <= config["max_retries"]:
                    delay = self.get_retry_delay(e, phase)
                    self.log(f"  [recovery] Retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                else:
                    # Log full traceback for debugging
                    logger.error(f"Phase {phase} failed after {attempts} attempts", exc_info=True)

                    if self.should_trigger_rollback(phase):
                        self.log(f"  [recovery] CRITICAL: Phase '{phase}' requires rollback")

                    return (False, str(e))

        return (False, "Max retries exceeded")


# =============================================================================
# SAFETY CONSTANTS - Files that must NEVER be deleted or broken
# =============================================================================
PROTECTED_FILES = [
    # Core nomic loop infrastructure
    "scripts/nomic_loop.py",  # The nomic loop itself - CRITICAL
    "scripts/run_nomic_with_stream.py",  # Streaming wrapper - protects --auto flag

    # Core aragora modules
    "aragora/__init__.py",     # Core package initialization
    "aragora/core.py",         # Core types and abstractions
    "aragora/debate/orchestrator.py",  # Debate infrastructure
    "aragora/agents/__init__.py",      # Agent system
    "aragora/implement/__init__.py",   # Implementation system

    # Valuable features added by nomic loop
    "aragora/agents/cli_agents.py",    # CLI agent harnesses (KiloCode, Claude, Codex, Grok)
    "aragora/server/stream.py",        # Streaming, AudienceInbox, TokenBucket
    "aragora/memory/store.py",         # CritiqueStore, AgentReputation
    "aragora/debate/embeddings.py",    # DebateEmbeddingsDatabase for historical search

    # Live dashboard (web interface)
    "aragora/live/src/components/AgentPanel.tsx",       # Agent activity panel with colors
    "aragora/live/src/components/UserParticipation.tsx", # User participation UI
    "aragora/live/src/app/page.tsx",                    # Main dashboard page
    "aragora/live/tailwind.config.js",                  # Tailwind config with agent colors
]

# Global cache for protected file checksums (computed at startup)
_PROTECTED_FILE_CHECKSUMS: dict[str, str] = {}


def _compute_file_checksum(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    import hashlib
    if not filepath.exists():
        return ""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]  # Short hash for logging


def _init_protected_checksums(base_path: Path) -> dict[str, str]:
    """Initialize checksums for all protected files at startup."""
    global _PROTECTED_FILE_CHECKSUMS
    for rel_path in PROTECTED_FILES:
        full_path = base_path / rel_path
        if full_path.exists():
            _PROTECTED_FILE_CHECKSUMS[rel_path] = _compute_file_checksum(full_path)
    return _PROTECTED_FILE_CHECKSUMS


def verify_protected_files_unchanged(base_path: Path) -> tuple[bool, list[str]]:
    """
    Verify that protected files haven't been unexpectedly modified.

    Security measure: Detects accidental or malicious modifications to
    critical infrastructure between phases.

    Returns:
        Tuple of (all_ok, list of modified files)
    """
    modified = []
    for rel_path, expected_hash in _PROTECTED_FILE_CHECKSUMS.items():
        full_path = base_path / rel_path
        current_hash = _compute_file_checksum(full_path)
        if current_hash != expected_hash:
            modified.append(rel_path)
    return len(modified) == 0, modified

SAFETY_PREAMBLE = """
=== CRITICAL SAFETY RULES ===
You are modifying a self-improving system. These rules are NON-NEGOTIABLE:

1. NEVER DELETE OR BREAK:
   - scripts/nomic_loop.py (the loop itself)
   - aragora/__init__.py (core package)
   - aragora/core.py (core types)
   - aragora/debate/orchestrator.py (debate infrastructure)
   - Any file that enables the nomic loop to function

2. ANABOLISM OVER CATABOLISM:
   - ADD features, don't remove working ones
   - EXTEND functionality, don't simplify it away
   - Only remove code that is BROKEN or HARMFUL
   - When in doubt, keep existing functionality

3. PRESERVE CORE CAPABILITIES:
   - Multi-agent debate must keep working
   - File logging must keep working
   - Git integration must keep working
   - All existing API contracts must be maintained

4. DEFENSIVE CODING:
   - New features should not break existing ones
   - Add tests for new functionality
   - Maintain backward compatibility

5. TECHNICAL DEBT - REDUCE SAFELY:
   - Reducing technical debt is GOOD when it's safe
   - Safe refactoring: improve code without changing behavior
   - UNSAFE: removing functionality, breaking APIs, deleting imports
   - SAFE: renaming for clarity, extracting functions, improving types
   - Test that refactored code works identically to original
   - If unsure whether a change is safe, DON'T MAKE IT

6. AGENT PROMPTS ARE SACRED:
   - NEVER modify agent system prompts in the codebase
   - Agent prompts define the personalities and safety constraints
   - Changes to agent prompts require UNANIMOUS consent from all agents
   - If ANY doubt exists about modifying prompts, DO NOT MODIFY THEM
   - This includes prompts in: nomic_loop.py, agents/*.py, any prompt templates
   - The only exception: fixing an obvious typo or syntax error
===========================
"""


# Load .env file if present
def load_dotenv(env_path: Path):
    """Load environment variables from .env file."""
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load .env from aragora root
_script_dir = Path(__file__).parent
_env_file = _script_dir.parent / ".env"
load_dotenv(_env_file)

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.roles import RoleRotationConfig, CognitiveRole
from aragora.core import Environment
from aragora.agents.api_agents import GeminiAgent, DeepSeekV3Agent
from aragora.agents.cli_agents import CodexAgent, ClaudeAgent, GrokCLIAgent, KiloCodeAgent

# Check if Kilo Code CLI is available for Gemini/Grok codebase exploration
KILOCODE_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(["which", "kilocode"], capture_output=True, text=True)
    KILOCODE_AVAILABLE = result.returncode == 0
except Exception:
    pass

# Skip KiloCode agents during context gathering phase (agentic codebase exploration)
# KiloCode's agentic exploration consistently times out (>30 min for Gemini/Grok)
# Gemini/Grok still participate in debates via direct API calls
SKIP_KILOCODE_CONTEXT_GATHERING = True

# Genesis module for fractal debates with agent evolution
GENESIS_AVAILABLE = False
try:
    from aragora.genesis import (
        FractalOrchestrator,
        PopulationManager,
        GenesisLedger,
        create_genesis_hooks,
        create_logging_hooks,
    )
    GENESIS_AVAILABLE = True
except ImportError:
    pass
from aragora.implement import (
    generate_implement_plan,
    create_single_task_plan,
    HybridExecutor,
    load_progress,
    save_progress,
    clear_progress,
    ImplementProgress,
)

# Optional streaming support
try:
    from aragora.server.stream import SyncEventEmitter, create_arena_hooks
    from aragora.server.nomic_stream import create_nomic_hooks
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    SyncEventEmitter = None
    create_nomic_hooks = None
    create_arena_hooks = None

# Optional Supabase persistence
try:
    from aragora.persistence import SupabaseClient, NomicCycle, StreamEvent, DebateArtifact
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    SupabaseClient = None
    NomicCycle = None
    StreamEvent = None
    DebateArtifact = None

# Debate embeddings for historical search
try:
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    DebateEmbeddingsDatabase = None

# ContinuumMemory for multi-timescale learning
try:
    from aragora.memory.continuum import ContinuumMemory, MemoryTier
    CONTINUUM_AVAILABLE = True
except ImportError:
    CONTINUUM_AVAILABLE = False
    ContinuumMemory = None
    MemoryTier = None

# ReplayRecorder for cycle event recording
try:
    from aragora.replay.recorder import ReplayRecorder
    REPLAY_AVAILABLE = True
except ImportError:
    REPLAY_AVAILABLE = False
    ReplayRecorder = None

# MetaLearner for self-tuning hyperparameters
try:
    from aragora.learning.meta import MetaLearner
    METALEARNER_AVAILABLE = True
except ImportError:
    METALEARNER_AVAILABLE = False
    MetaLearner = None

# IntrospectionAPI for agent self-awareness
try:
    from aragora.introspection.api import get_agent_introspection, format_introspection_section
    INTROSPECTION_AVAILABLE = True
except ImportError:
    INTROSPECTION_AVAILABLE = False
    get_agent_introspection = None
    format_introspection_section = None

# ArgumentCartographer for debate visualization
try:
    from aragora.visualization.mapper import ArgumentCartographer
    CARTOGRAPHER_AVAILABLE = True
except ImportError:
    CARTOGRAPHER_AVAILABLE = False
    ArgumentCartographer = None

# WebhookDispatcher for external event notifications
try:
    from aragora.integrations.webhooks import WebhookDispatcher, WebhookConfig
    WEBHOOKS_AVAILABLE = True
except ImportError:
    WEBHOOKS_AVAILABLE = False
    WebhookDispatcher = None
    WebhookConfig = None

# ConsensusMemory for tracking settled vs contested topics
try:
    from aragora.memory.consensus import ConsensusMemory, ConsensusStrength, DissentRetriever
    CONSENSUS_MEMORY_AVAILABLE = True
except ImportError:
    CONSENSUS_MEMORY_AVAILABLE = False
    ConsensusMemory = None
    ConsensusStrength = None
    DissentRetriever = None

# InsightExtractor for post-debate pattern learning
try:
    from aragora.insights.extractor import InsightExtractor
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    InsightExtractor = None

# FlipDetector for position reversal detection
try:
    from aragora.insights.flip_detector import FlipDetector
    FLIP_DETECTOR_AVAILABLE = True
except ImportError:
    FLIP_DETECTOR_AVAILABLE = False
    FlipDetector = None

# NomicIntegration for advanced feature coordination
try:
    from aragora.nomic.integration import NomicIntegration, create_nomic_integration
    NOMIC_INTEGRATION_AVAILABLE = True
except ImportError:
    NOMIC_INTEGRATION_AVAILABLE = False
    NomicIntegration = None
    create_nomic_integration = None

# MemoryStream for per-agent persistent memory (Phase 3)
try:
    from aragora.memory.streams import MemoryStream
    MEMORY_STREAM_AVAILABLE = True
except ImportError:
    MEMORY_STREAM_AVAILABLE = False
    MemoryStream = None

# LocalDocsConnector for evidence grounding (Phase 3)
try:
    from aragora.connectors.local_docs import LocalDocsConnector
    LOCAL_DOCS_AVAILABLE = True
except ImportError:
    LOCAL_DOCS_AVAILABLE = False
    LocalDocsConnector = None

# CounterfactualOrchestrator for deadlock resolution (Phase 3)
try:
    from aragora.debate.counterfactual import CounterfactualOrchestrator
    COUNTERFACTUAL_AVAILABLE = True
except ImportError:
    COUNTERFACTUAL_AVAILABLE = False
    CounterfactualOrchestrator = None

# CapabilityProber for agent quality assurance (Phase 3)
try:
    from aragora.modes.prober import CapabilityProber, ProbeType
    PROBER_AVAILABLE = True
except ImportError:
    PROBER_AVAILABLE = False
    CapabilityProber = None
    ProbeType = None

# DeepAuditMode for intensive review of protected file changes (Heavy3-inspired)
try:
    from aragora.modes.deep_audit import run_deep_audit, CODE_ARCHITECTURE_AUDIT, DeepAuditConfig
    DEEP_AUDIT_AVAILABLE = True
except ImportError:
    DEEP_AUDIT_AVAILABLE = False
    run_deep_audit = None
    CODE_ARCHITECTURE_AUDIT = None
    DeepAuditConfig = None

# DebateTemplates for structured debate formats (Phase 3)
try:
    from aragora.templates import CODE_REVIEW_TEMPLATE, DESIGN_DOC_TEMPLATE, DebateTemplate
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    CODE_REVIEW_TEMPLATE = None
    DESIGN_DOC_TEMPLATE = None
    DebateTemplate = None

# PersonaManager for agent traits/expertise evolution (Phase 4)
try:
    from aragora.agents.personas import PersonaManager, get_or_create_persona, EXPERTISE_DOMAINS
    PERSONAS_AVAILABLE = True
except ImportError:
    PERSONAS_AVAILABLE = False
    PersonaManager = None
    get_or_create_persona = None
    EXPERTISE_DOMAINS = []

# PromptEvolver for prompt evolution from winning patterns (Phase 4)
try:
    from aragora.evolution.evolver import PromptEvolver, EvolutionStrategy
    EVOLVER_AVAILABLE = True
except ImportError:
    EVOLVER_AVAILABLE = False
    PromptEvolver = None
    EvolutionStrategy = None

# Tournament for periodic competitive benchmarking (Phase 4)
try:
    from aragora.tournaments import Tournament, TournamentFormat, create_default_tasks
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False
    Tournament = None
    TournamentFormat = None
    create_default_tasks = None

# ConvergenceDetector for early stopping (Phase 5)
try:
    from aragora.debate.convergence import ConvergenceDetector, ConvergenceResult
    CONVERGENCE_AVAILABLE = True
except ImportError:
    CONVERGENCE_AVAILABLE = False
    ConvergenceDetector = None
    ConvergenceResult = None

# MetaCritiqueAnalyzer for process feedback (Phase 5)
try:
    from aragora.debate.meta import MetaCritiqueAnalyzer, MetaCritique
    META_CRITIQUE_AVAILABLE = True
except ImportError:
    META_CRITIQUE_AVAILABLE = False
    MetaCritiqueAnalyzer = None
    MetaCritique = None

# EloSystem for agent skill tracking (Phase 5)
try:
    from aragora.ranking.elo import EloSystem, AgentRating
    ELO_AVAILABLE = True
except ImportError:
    ELO_AVAILABLE = False
    EloSystem = None
    AgentRating = None

# AgentSelector for smart team selection (Phase 5)
try:
    from aragora.routing.selection import AgentSelector, AgentProfile, TaskRequirements
    SELECTOR_AVAILABLE = True
except ImportError:
    SELECTOR_AVAILABLE = False
    AgentSelector = None
    AgentProfile = None
    TaskRequirements = None

# ProbeFilter for probe-aware agent selection (Phase 10)
try:
    from aragora.routing.probe_filter import ProbeFilter, ProbeProfile
    PROBE_FILTER_AVAILABLE = True
except ImportError:
    PROBE_FILTER_AVAILABLE = False
    ProbeFilter = None
    ProbeProfile = None

# RiskRegister for risk tracking (Phase 5)
try:
    from aragora.pipeline.risk_register import RiskLevel
    RISK_REGISTER_AVAILABLE = True
except ImportError:
    RISK_REGISTER_AVAILABLE = False
    RiskLevel = None

# =============================================================================
# Phase 6: Verifiable Reasoning & Robustness Testing
# =============================================================================

# ClaimsKernel for structured reasoning (Phase 6)
try:
    from aragora.reasoning.claims import (
        ClaimsKernel, TypedClaim, TypedEvidence, ClaimRelation,
        ClaimType, RelationType, EvidenceType
    )
    CLAIMS_KERNEL_AVAILABLE = True
except ImportError:
    CLAIMS_KERNEL_AVAILABLE = False
    ClaimsKernel = None
    TypedClaim = None
    ClaimType = None
    RelationType = None

# ProvenanceManager for evidence tracking (Phase 6)
try:
    from aragora.reasoning.provenance import (
        ProvenanceManager, ProvenanceChain, SourceType, TransformationType
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False
    ProvenanceManager = None
    SourceType = None

# BeliefNetwork for probabilistic reasoning (Phase 6)
try:
    from aragora.reasoning.belief import (
        BeliefNetwork, BeliefPropagationAnalyzer, BeliefDistribution
    )
    BELIEF_NETWORK_AVAILABLE = True
except ImportError:
    BELIEF_NETWORK_AVAILABLE = False
    BeliefNetwork = None
    BeliefPropagationAnalyzer = None

# ProofExecutor for executable verification (Phase 6)
try:
    from aragora.verification.proofs import (
        ProofExecutor, ClaimVerifier, VerificationProof, VerificationReport,
        ProofType, ProofStatus, ProofBuilder
    )
    PROOF_EXECUTOR_AVAILABLE = True
except ImportError:
    PROOF_EXECUTOR_AVAILABLE = False
    ProofExecutor = None
    ClaimVerifier = None
    VerificationReport = None
    ProofBuilder = None

# ScenarioMatrix for robustness testing (Phase 6)
try:
    from aragora.debate.scenarios import (
        ScenarioMatrix, MatrixDebateRunner, ScenarioComparator,
        Scenario, ScenarioType, OutcomeCategory
    )
    SCENARIO_MATRIX_AVAILABLE = True
except ImportError:
    SCENARIO_MATRIX_AVAILABLE = False
    ScenarioMatrix = None
    ScenarioComparator = None

# =============================================================================
# Phase 7: Resilience, Living Documents, & Observability
# =============================================================================

# EnhancedProvenanceManager for staleness detection (Phase 7)
try:
    from aragora.reasoning.provenance_enhanced import (
        EnhancedProvenanceManager, GitProvenanceTracker, StalenessCheck,
        StalenessStatus, RevalidationTrigger
    )
    ENHANCED_PROVENANCE_AVAILABLE = True
except ImportError:
    ENHANCED_PROVENANCE_AVAILABLE = False
    EnhancedProvenanceManager = None
    StalenessStatus = None

# CheckpointManager for pause/resume (Phase 7)
try:
    from aragora.debate.checkpoint import (
        CheckpointManager, DebateCheckpoint, FileCheckpointStore, CheckpointConfig
    )
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    CheckpointManager = None

# BreakpointManager for human intervention (Phase 7)
try:
    from aragora.debate.breakpoints import (
        BreakpointManager, BreakpointConfig, Breakpoint, HumanGuidance, BreakpointTrigger
    )
    BREAKPOINT_AVAILABLE = True
except ImportError:
    BREAKPOINT_AVAILABLE = False
    BreakpointManager = None
    BreakpointTrigger = None

# ReliabilityScorer for confidence scoring (Phase 7)
try:
    from aragora.reasoning.reliability import (
        ReliabilityScorer, ClaimReliability, EvidenceReliability, ReliabilityLevel
    )
    RELIABILITY_SCORER_AVAILABLE = True
except ImportError:
    RELIABILITY_SCORER_AVAILABLE = False
    ReliabilityScorer = None
    ReliabilityLevel = None

# DebateTracer for audit logs (Phase 7)
try:
    from aragora.debate.traces import (
        DebateTracer, DebateTrace, TraceEvent, EventType
    )
    DEBATE_TRACER_AVAILABLE = True
except ImportError:
    DEBATE_TRACER_AVAILABLE = False
    DebateTracer = None
    EventType = None

# =============================================================================
# Phase 8: Agent Evolution, Semantic Memory & Advanced Debates
# =============================================================================

# PersonaLaboratory for agent evolution (Phase 8)
try:
    from aragora.agents.laboratory import (
        PersonaLaboratory, PersonaExperiment, EmergentTrait, TraitTransfer
    )
    PERSONA_LAB_AVAILABLE = True
except ImportError:
    PERSONA_LAB_AVAILABLE = False
    PersonaLaboratory = None
    EmergentTrait = None

# SemanticRetriever for pattern matching (Phase 8)
try:
    from aragora.memory.embeddings import (
        SemanticRetriever, EmbeddingProvider, cosine_similarity
    )
    SEMANTIC_RETRIEVER_AVAILABLE = True
except ImportError:
    SEMANTIC_RETRIEVER_AVAILABLE = False
    SemanticRetriever = None

# FormalVerificationManager for theorem proving (Phase 8)
try:
    from aragora.verification.formal import (
        FormalVerificationManager, FormalProofResult,
        FormalProofStatus, FormalLanguage
    )
    FORMAL_VERIFICATION_AVAILABLE = True
except ImportError:
    FORMAL_VERIFICATION_AVAILABLE = False
    FormalVerificationManager = None
    FormalProofResult = None

# DebateGraph for DAG-based debates (Phase 8)
try:
    from aragora.debate.graph import (
        DebateGraph, DebateNode, GraphDebateOrchestrator,
        NodeType, BranchReason, MergeStrategy
    )
    DEBATE_GRAPH_AVAILABLE = True
except ImportError:
    DEBATE_GRAPH_AVAILABLE = False
    DebateGraph = None
    GraphDebateOrchestrator = None

# DebateForker for parallel exploration (Phase 8)
try:
    from aragora.debate.forking import (
        DebateForker, ForkDetector, Branch, ForkPoint, ForkDecision, MergeResult
    )
    DEBATE_FORKER_AVAILABLE = True
except ImportError:
    DEBATE_FORKER_AVAILABLE = False
    DebateForker = None
    ForkDetector = None

# =============================================================================
# Phase 9: Grounded Personas & Truth-Based Identity
# =============================================================================

# PositionTracker for truth-grounded personas (Phase 9)
try:
    from aragora.agents.truth_grounding import (
        PositionTracker, Position, TruthGroundedPersona, TruthGroundedLaboratory
    )
    POSITION_TRACKER_AVAILABLE = True
except ImportError:
    POSITION_TRACKER_AVAILABLE = False
    PositionTracker = None
    TruthGroundedLaboratory = None

# GroundedPersonas for evidence-based identity (Phase 9)
try:
    from aragora.agents.grounded import (
        PositionLedger, RelationshipTracker, PersonaSynthesizer,
        GroundedPersona, Position as GroundedPosition, MomentDetector
    )
    GROUNDED_PERSONAS_AVAILABLE = True
except ImportError:
    GROUNDED_PERSONAS_AVAILABLE = False
    PositionLedger = None
    RelationshipTracker = None
    PersonaSynthesizer = None
    GroundedPersona = None
    MomentDetector = None

# CalibrationTracker for prediction accuracy tracking (Phase 10)
try:
    from aragora.agents.calibration import CalibrationTracker, CalibrationSummary
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    CalibrationTracker = None
    CalibrationSummary = None

# SuggestionFeedbackTracker for audience suggestion effectiveness (Phase 10)
try:
    from aragora.audience.feedback import SuggestionFeedbackTracker
    SUGGESTION_FEEDBACK_AVAILABLE = True
except ImportError:
    SUGGESTION_FEEDBACK_AVAILABLE = False
    SuggestionFeedbackTracker = None

# =============================================================================
# Citation Grounding (Heavy3-inspired scholarly evidence)
# =============================================================================

# CitationStore for evidence-backed verdicts
try:
    from aragora.reasoning.citations import (
        CitationStore, CitationExtractor, GroundedVerdict,
        ScholarlyEvidence, CitedClaim, CitationType, CitationQuality
    )
    CITATION_GROUNDING_AVAILABLE = True
except ImportError:
    CITATION_GROUNDING_AVAILABLE = False
    CitationStore = None
    CitationExtractor = None
    GroundedVerdict = None

# =============================================================================
# Broadcast Module (Post-Debate Summaries)
# =============================================================================

try:
    from aragora.broadcast.script_gen import DebateSummaryGenerator
    BROADCAST_AVAILABLE = True
except ImportError:
    BROADCAST_AVAILABLE = False
    DebateSummaryGenerator = None

# =============================================================================
# Pulse Integration (Trending Topics for Debate Generation)
# =============================================================================

try:
    from aragora.pulse import PulseManager, TrendingTopic, PulseIngestor
    PULSE_AVAILABLE = True
except ImportError:
    PULSE_AVAILABLE = False
    PulseManager = None
    TrendingTopic = None


# =============================================================================
# Circuit Breaker for Agent Failure Handling
# =============================================================================

class AgentCircuitBreaker:
    """
    Circuit breaker pattern for agent reliability.

    Tracks consecutive failures per agent and temporarily disables
    agents that fail repeatedly to prevent wasting cycles on broken agents.

    Extended with task-scoped tracking (Jan 2026):
    - Tracks failures per task type (debate, design, implement, verify)
    - Agents can be disabled for specific task types while still usable for others
    - Success rates tracked for intelligent agent selection
    """

    def __init__(self, failure_threshold: int = 3, cooldown_cycles: int = 2):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before tripping
            cooldown_cycles: Number of cycles to skip after tripping
        """
        self.failure_threshold = failure_threshold
        self.cooldown_cycles = cooldown_cycles
        self.failures: dict[str, int] = {}  # agent_name -> consecutive failure count
        self.cooldowns: dict[str, int] = {}  # agent_name -> cycles remaining in cooldown

        # Task-scoped tracking (new)
        self.task_failures: dict[str, dict[str, int]] = {}  # agent -> task_type -> count
        self.task_success_rate: dict[str, dict[str, float]] = {}  # agent -> task_type -> rate
        self.task_cooldowns: dict[str, dict[str, int]] = {}  # agent -> task_type -> cooldown

    def record_success(self, agent_name: str) -> None:
        """Reset failure count and reduce cooldown on success."""
        self.failures[agent_name] = 0
        # If agent was in cooldown but succeeded (half-open state), reduce or close circuit
        if agent_name in self.cooldowns and self.cooldowns[agent_name] > 0:
            self.cooldowns[agent_name] = max(0, self.cooldowns[agent_name] - 1)
            if self.cooldowns[agent_name] == 0:
                del self.cooldowns[agent_name]
                logging.info(f"[circuit-breaker] {agent_name} recovered after success")

    def record_failure(self, agent_name: str) -> bool:
        """
        Record a failure and potentially trip the circuit.

        Returns:
            True if circuit just tripped (agent now in cooldown)
        """
        self.failures[agent_name] = self.failures.get(agent_name, 0) + 1
        if self.failures[agent_name] >= self.failure_threshold:
            self.cooldowns[agent_name] = self.cooldown_cycles
            self.failures[agent_name] = 0  # Reset for next time
            return True
        return False

    def record_task_success(self, agent_name: str, task_type: str) -> None:
        """Record success for specific task type and update running average."""
        # Initialize structures if needed
        if agent_name not in self.task_success_rate:
            self.task_success_rate[agent_name] = {}
        if agent_name not in self.task_failures:
            self.task_failures[agent_name] = {}

        # Reset task-specific failures
        self.task_failures[agent_name][task_type] = 0

        # Update running average (exponential moving average)
        current_rate = self.task_success_rate[agent_name].get(task_type, 0.5)
        self.task_success_rate[agent_name][task_type] = current_rate * 0.8 + 0.2

        # Also record agent-level success
        self.record_success(agent_name)

    def record_task_failure(self, agent_name: str, task_type: str) -> bool:
        """
        Record failure for specific task type.

        Returns:
            True if task-specific circuit just tripped
        """
        # Initialize structures if needed
        if agent_name not in self.task_failures:
            self.task_failures[agent_name] = {}
        if agent_name not in self.task_cooldowns:
            self.task_cooldowns[agent_name] = {}
        if agent_name not in self.task_success_rate:
            self.task_success_rate[agent_name] = {}

        # Increment task-specific failure count
        self.task_failures[agent_name][task_type] = \
            self.task_failures[agent_name].get(task_type, 0) + 1

        # Update running average (exponential moving average toward 0)
        current_rate = self.task_success_rate[agent_name].get(task_type, 0.5)
        self.task_success_rate[agent_name][task_type] = current_rate * 0.8

        # Trip task-specific circuit if threshold reached
        if self.task_failures[agent_name][task_type] >= self.failure_threshold:
            self.task_cooldowns[agent_name][task_type] = self.cooldown_cycles
            self.task_failures[agent_name][task_type] = 0
            return True

        # Also record agent-level failure
        self.record_failure(agent_name)
        return False

    def get_task_success_rate(self, agent_name: str, task_type: str) -> float:
        """Get agent's success rate for specific task type (0.0 to 1.0)."""
        if agent_name not in self.task_success_rate:
            return 0.5  # Default neutral
        return self.task_success_rate[agent_name].get(task_type, 0.5)

    def is_available_for_task(self, agent_name: str, task_type: str) -> bool:
        """Check if agent is available for a specific task type."""
        # First check global availability
        if not self.is_available(agent_name):
            return False
        # Then check task-specific cooldown
        if agent_name in self.task_cooldowns:
            if self.task_cooldowns[agent_name].get(task_type, 0) > 0:
                return False
        return True

    def is_available(self, agent_name: str) -> bool:
        """Check if agent is available (not in cooldown)."""
        return self.cooldowns.get(agent_name, 0) <= 0

    def start_new_cycle(self) -> None:
        """Decrement cooldowns at start of each cycle."""
        # Decrement global cooldowns
        for agent_name in list(self.cooldowns.keys()):
            if self.cooldowns[agent_name] > 0:
                self.cooldowns[agent_name] -= 1

        # Decrement task-specific cooldowns
        for agent_name in list(self.task_cooldowns.keys()):
            for task_type in list(self.task_cooldowns[agent_name].keys()):
                if self.task_cooldowns[agent_name][task_type] > 0:
                    self.task_cooldowns[agent_name][task_type] -= 1

    def get_status(self) -> dict:
        """Get circuit breaker status for all agents."""
        return {
            "failures": dict(self.failures),
            "cooldowns": dict(self.cooldowns),
            "task_failures": dict(self.task_failures),
            "task_cooldowns": dict(self.task_cooldowns),
            "task_success_rates": dict(self.task_success_rate),
        }


class NomicLoop:
    """
    Autonomous self-improvement loop for aragora.

    Each cycle:
    1. Agents debate what to improve
    2. Agents design the implementation
    3. Agents implement (codex writes code)
    4. Changes are verified and committed
    5. Loop repeats

    SAFETY FEATURES:
    - All output logged to .nomic/nomic_loop.log for live monitoring
    - State saved to .nomic/nomic_state.json for crash recovery
    - Protected files backed up before each cycle
    - Automatic restore if protected files are damaged
    """

    def __init__(
        self,
        aragora_path: str = None,
        max_cycles: int = 10,
        require_human_approval: bool = True,
        auto_commit: bool = False,
        initial_proposal: str = None,
        stream_emitter: "SyncEventEmitter" = None,
        use_genesis: bool = False,
        enable_persistence: bool = True,
        disable_rollback: bool = False,  # Disable rollback on verification failure
        max_cycle_seconds: int = 3600,  # 1 hour cycle timeout (prevents multi-hour hangs)
    ):
        self.aragora_path = Path(aragora_path or Path(__file__).parent.parent)
        self.max_cycles = max_cycles
        self.require_human_approval = require_human_approval
        self.auto_commit = auto_commit
        self.initial_proposal = initial_proposal
        self.disable_rollback = disable_rollback
        self.max_cycle_seconds = max_cycle_seconds
        self.cycle_count = 0
        self.history = []

        # Circuit breaker for agent reliability
        # Threshold=5 gives agents more chances before trip, cooldown=1 allows faster recovery
        self.circuit_breaker = AgentCircuitBreaker(failure_threshold=5, cooldown_cycles=1)

        # Phase recovery for structured error handling
        self.phase_recovery = PhaseRecovery(log_func=lambda msg: print(msg))

        # Generate unique loop ID for this run
        self.loop_id = f"nomic-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Genesis mode: fractal debates with agent evolution
        self.use_genesis = use_genesis and GENESIS_AVAILABLE
        self.genesis_ledger = None
        self.population_manager = None
        if self.use_genesis:
            self.genesis_ledger = GenesisLedger(str(self.aragora_path / ".nomic" / "genesis.db"))
            self.population_manager = PopulationManager(str(self.aragora_path / ".nomic" / "genesis.db"))

        # Setup logging infrastructure (must be before other initializations that use nomic_dir)
        self.nomic_dir = self.aragora_path / ".nomic"
        self.nomic_dir.mkdir(exist_ok=True)
        self.log_file = self.nomic_dir / "nomic_loop.log"
        self.state_file = self.nomic_dir / "nomic_state.json"
        self.backup_dir = self.nomic_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Restore circuit breaker state from previous run (persistence across cycles)
        circuit_breaker_path = self.nomic_dir / "circuit_breaker.json"
        if circuit_breaker_path.exists():
            try:
                with open(circuit_breaker_path) as f:
                    state = json.load(f)
                    self.circuit_breaker.failures = state.get("failures", {})
                    self.circuit_breaker.cooldowns = state.get("cooldowns", {})
                    # Restore task-scoped tracking with defaultdict conversion
                    for agent, tasks in state.get("task_failures", {}).items():
                        self.circuit_breaker.task_failures[agent] = defaultdict(int, tasks)
                    for agent, tasks in state.get("task_cooldowns", {}).items():
                        self.circuit_breaker.task_cooldowns[agent] = defaultdict(int, tasks)
                    for agent, rates in state.get("task_success_rate", {}).items():
                        self.circuit_breaker.task_success_rate[agent] = rates
                    task_count = sum(len(t) for t in self.circuit_breaker.task_cooldowns.values())
                    print(f"[circuit-breaker] Restored state: {len(self.circuit_breaker.cooldowns)} agents in cooldown, {task_count} task cooldowns")
            except Exception as e:
                print(f"[circuit-breaker] Failed to restore state: {e}")

        # Initialize protected file checksums for integrity verification
        _init_protected_checksums(self.aragora_path)
        print(f"[security] Initialized checksums for {len(_PROTECTED_FILE_CHECKSUMS)} protected files")

        # Supabase persistence for history tracking
        self.persistence = None
        if enable_persistence and PERSISTENCE_AVAILABLE:
            self.persistence = SupabaseClient()
            if self.persistence.is_configured:
                print(f"[persistence] Supabase connected, loop_id: {self.loop_id}")
            else:
                self.persistence = None

        # Debate embeddings database for historical search
        self.debate_embeddings = None
        if EMBEDDINGS_AVAILABLE:
            embeddings_path = self.nomic_dir / "debate_embeddings.db"
            self.debate_embeddings = DebateEmbeddingsDatabase(str(embeddings_path))
            print(f"[embeddings] Debate embeddings database initialized")

        # CritiqueStore for patterns and agent reputation tracking
        self.critique_store = None
        try:
            from aragora.memory.store import CritiqueStore
            critique_db_path = self.nomic_dir / "agora_memory.db"
            self.critique_store = CritiqueStore(str(critique_db_path))
            print(f"[memory] CritiqueStore initialized for patterns and reputation")
        except ImportError:
            pass

        # ContinuumMemory for multi-timescale pattern learning
        self.continuum = None
        if CONTINUUM_AVAILABLE:
            continuum_path = self.nomic_dir / "continuum.db"
            self.continuum = ContinuumMemory(str(continuum_path))
            print(f"[continuum] Multi-timescale memory initialized")

        # ReplayRecorder will be created per cycle
        self.replay_recorder = None

        # MetaLearner for self-tuning hyperparameters (runs every 5 cycles)
        self.meta_learner = None
        if METALEARNER_AVAILABLE and self.continuum:
            meta_learner_path = self.nomic_dir / "meta_learning.db"
            self.meta_learner = MetaLearner(str(meta_learner_path))
            print(f"[meta] MetaLearner initialized for hyperparameter tuning")

        # ArgumentCartographer will be created per cycle for visualization
        self.cartographer = None
        self.visualizations_dir = self.nomic_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)
        if CARTOGRAPHER_AVAILABLE:
            print(f"[viz] ArgumentCartographer available for debate visualization")

        # WebhookDispatcher for external event notifications
        self.webhook_dispatcher = None
        webhook_url = os.environ.get("ARAGORA_WEBHOOK_URL")
        if WEBHOOKS_AVAILABLE and webhook_url and WebhookConfig:
            try:
                config = WebhookConfig(
                    name="default",
                    url=webhook_url,
                    secret=os.environ.get("ARAGORA_WEBHOOK_SECRET", ""),
                )
                self.webhook_dispatcher = WebhookDispatcher([config])
                self.webhook_dispatcher.start()
                print(f"[webhook] Dispatcher started for {webhook_url[:50]}...")
            except Exception as e:
                print(f"[webhook] Failed to initialize: {e}")

        # ConsensusMemory for tracking settled vs contested topics
        self.consensus_memory = None
        self.dissent_retriever = None
        if CONSENSUS_MEMORY_AVAILABLE:
            consensus_db_path = self.nomic_dir / "consensus_memory.db"
            self.consensus_memory = ConsensusMemory(str(consensus_db_path))
            self.dissent_retriever = DissentRetriever(self.consensus_memory)
            print(f"[consensus] ConsensusMemory initialized for topic tracking")

        # InsightExtractor for post-debate pattern learning
        self.insight_extractor = None
        if INSIGHTS_AVAILABLE:
            self.insight_extractor = InsightExtractor()
            print(f"[insights] InsightExtractor initialized for pattern learning")

        # InsightStore for persisting debate insights (debate consensus feature)
        self.insight_store = None
        if INSIGHTS_AVAILABLE:
            try:
                from aragora.insights.store import InsightStore
                insights_path = self.nomic_dir / "aragora_insights.db"
                self.insight_store = InsightStore(str(insights_path))
                print(f"[insights] InsightStore initialized for debate persistence")
            except Exception as e:
                print(f"[insights] InsightStore init failed: {e}")

        # NomicIntegration for advanced feature coordination
        # Integrates: belief propagation, capability probing, staleness detection,
        # counterfactual branching, and checkpointing
        self.nomic_integration = None
        if NOMIC_INTEGRATION_AVAILABLE and create_nomic_integration:
            try:
                checkpoint_dir = self.nomic_dir / "checkpoints"
                self.nomic_integration = create_nomic_integration(
                    checkpoint_dir=str(checkpoint_dir),
                    enable_probing=True,  # Probe agents for reliability
                    enable_belief_analysis=True,  # Bayesian belief propagation
                    enable_staleness_check=True,  # Detect stale evidence
                    enable_counterfactual=True,  # Fork on contested claims
                    enable_checkpointing=True,  # Phase checkpointing
                )
                print(f"[integration] NomicIntegration initialized for advanced features")
            except Exception as e:
                print(f"[integration] Failed to initialize: {e}")
                self.nomic_integration = None

        # Phase 3: MemoryStream for per-agent persistent memory
        self.memory_stream = None
        if MEMORY_STREAM_AVAILABLE:
            memory_db_path = self.nomic_dir / "agent_memories.db"
            self.memory_stream = MemoryStream(str(memory_db_path))
            print(f"[memory] Per-agent MemoryStream initialized")

        # Phase 3: LocalDocsConnector for evidence grounding
        self.local_docs = None
        if LOCAL_DOCS_AVAILABLE:
            self.local_docs = LocalDocsConnector(
                root_path=str(self.aragora_path),
                file_types='all'
            )
            print(f"[connectors] LocalDocsConnector initialized for evidence grounding")

        # Phase 3: CounterfactualOrchestrator for deadlock resolution
        self.counterfactual = None
        if COUNTERFACTUAL_AVAILABLE:
            self.counterfactual = CounterfactualOrchestrator()
            print(f"[counterfactual] Deadlock resolution via forking enabled")

        # Citation Grounding: CitationStore + CitationExtractor for evidence-backed verdicts
        self.citation_store = None
        self.citation_extractor = None
        if CITATION_GROUNDING_AVAILABLE:
            self.citation_store = CitationStore()
            self.citation_extractor = CitationExtractor()
            print(f"[citations] Citation grounding enabled for evidence-backed verdicts")

        # Pulse Integration: PulseManager for trending topic awareness
        self.pulse_manager = None
        if PULSE_AVAILABLE:
            self.pulse_manager = PulseManager()
            print(f"[pulse] PulseManager initialized for trending topic awareness")

        # Broadcast: Generate post-debate summaries
        self.summary_generator = None
        if BROADCAST_AVAILABLE:
            self.summary_generator = DebateSummaryGenerator()
            print(f"[broadcast] Debate summary generation enabled")

        # Phase 3: CapabilityProber for agent quality assurance
        self.prober = None
        if PROBER_AVAILABLE:
            self.prober = CapabilityProber()
            print(f"[prober] Agent capability probing enabled")

        # Phase 4: PersonaManager for agent traits/expertise evolution
        self.persona_manager = None
        if PERSONAS_AVAILABLE:
            persona_db_path = self.nomic_dir / "agent_personas.db"
            self.persona_manager = PersonaManager(str(persona_db_path))
            print(f"[personas] Agent personality evolution enabled")

        # Phase 4: PromptEvolver for prompt evolution from winning patterns
        self.prompt_evolver = None
        if EVOLVER_AVAILABLE:
            evolver_db_path = self.nomic_dir / "prompt_evolution.db"
            self.prompt_evolver = PromptEvolver(
                db_path=str(evolver_db_path),
                critique_store=self.critique_store,
                strategy=EvolutionStrategy.HYBRID
            )
            print(f"[evolver] Prompt evolution enabled")

        # Phase 4: Tournament tracking for periodic competitive benchmarking
        self.last_tournament_cycle = 0
        self.tournament_interval = 20  # Run tournament every 20 cycles

        # Phase 5: ConvergenceDetector for early stopping
        self.convergence_detector = None
        if CONVERGENCE_AVAILABLE:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=0.85,
                min_rounds_before_check=2
            )
            print(f"[convergence] Early stopping enabled")

        # Phase 5: MetaCritiqueAnalyzer for process feedback
        self.meta_analyzer = None
        if META_CRITIQUE_AVAILABLE:
            self.meta_analyzer = MetaCritiqueAnalyzer()
            print(f"[meta] Process feedback enabled")

        # P5-Phase2: Cache for meta-critique observations to inject into next debate
        self._cached_meta_observations: list = []
        self._last_meta_quality: float = 1.0

        # Deadlock detection and recovery state
        self._cycle_history: list = []  # Last N cycle outcomes for pattern detection
        self._max_cycle_history = 5
        self._phase_progress: dict = {}  # Progress tracking within phases
        self._design_recovery_attempts: set = set()  # Track recovery strategies tried
        self._deadlock_count: int = 0  # Consecutive deadlocks
        self._consensus_threshold_decay: int = 0  # Number of threshold decreases (0, 1, or 2)
        self._warned_50: bool = False  # Timeout warning flags
        self._warned_75: bool = False
        self._warned_90: bool = False
        self._fast_track_mode: bool = False  # Force simplified designs when running out of time
        self._force_judge_consensus: bool = False  # Break oscillations with judge
        self._cycle_start_time: datetime = None  # Track cycle start for warnings

        # Phase 5: EloSystem for agent skill tracking
        self.elo_system = None
        if ELO_AVAILABLE:
            elo_db_path = self.nomic_dir / "agent_elo.db"
            self.elo_system = EloSystem(str(elo_db_path))
            print(f"[elo] Agent skill tracking enabled")

        # Phase 5: AgentSelector for smart team selection
        self.agent_selector = None
        if SELECTOR_AVAILABLE and ELO_AVAILABLE and self.elo_system:
            self.agent_selector = AgentSelector(
                elo_system=self.elo_system,
                persona_manager=self.persona_manager
            )
            print(f"[selector] Smart agent selection enabled")

        # Phase 10: ProbeFilter for probe-aware agent selection
        self.probe_filter = None
        if PROBE_FILTER_AVAILABLE:
            self.probe_filter = ProbeFilter(nomic_dir=str(self.nomic_dir))
            print(f"[probe-filter] Probe-aware agent selection enabled")

            # Wire ProbeFilter into AgentSelector for reliability-weighted team selection
            if self.agent_selector and hasattr(self.agent_selector, 'set_probe_filter'):
                self.agent_selector.set_probe_filter(self.probe_filter)
                print(f"[selector] Probe reliability weighting enabled")

        # =================================================================
        # Phase 9: Grounded Personas & Truth-Based Identity
        # =================================================================

        # Phase 9: PositionTracker for truth-grounded personas
        self.position_tracker = None
        if POSITION_TRACKER_AVAILABLE:
            position_db_path = self.nomic_dir / "aragora_positions.db"
            self.position_tracker = PositionTracker(str(position_db_path))
            print(f"[positions] Truth-grounded position tracking enabled")

        # Phase 9: PositionLedger for evidence-based identity
        self.position_ledger = None
        if GROUNDED_PERSONAS_AVAILABLE and PositionLedger:
            ledger_db_path = self.nomic_dir / "grounded_positions.db"
            self.position_ledger = PositionLedger(str(ledger_db_path))
            print(f"[ledger] Evidence-based position ledger enabled")

        # Phase 9: RelationshipTracker for inter-agent dynamics
        self.relationship_tracker = None
        if GROUNDED_PERSONAS_AVAILABLE and RelationshipTracker:
            relationship_db_path = self.nomic_dir / "agent_relationships.db"
            self.relationship_tracker = RelationshipTracker(str(relationship_db_path))
            print(f"[relationships] Agent relationship tracking enabled")

        # Phase 9: MomentDetector for significant debate moments
        self.moment_detector = None
        if GROUNDED_PERSONAS_AVAILABLE and MomentDetector:
            self.moment_detector = MomentDetector(
                elo_system=self.elo_system,
                position_ledger=self.position_ledger,
                relationship_tracker=self.relationship_tracker,
            )
            print(f"[moments] Significant moment detection enabled")

        # Phase 10: CalibrationTracker for prediction accuracy
        self.calibration_tracker = None
        if CALIBRATION_AVAILABLE and CalibrationTracker:
            calibration_db_path = self.nomic_dir / "agent_calibration.db"
            self.calibration_tracker = CalibrationTracker(str(calibration_db_path))
            print(f"[calibration] Agent prediction calibration tracking enabled")

            # Wire CalibrationTracker into AgentSelector for calibration-weighted team selection
            if self.agent_selector and hasattr(self.agent_selector, 'set_calibration_tracker'):
                self.agent_selector.set_calibration_tracker(self.calibration_tracker)
                print(f"[selector] Calibration quality weighting enabled")

        # Phase 10: SuggestionFeedbackTracker for audience suggestion effectiveness
        self.suggestion_tracker = None
        if SUGGESTION_FEEDBACK_AVAILABLE and SuggestionFeedbackTracker:
            suggestion_db_path = self.nomic_dir / "suggestion_feedback.db"
            self.suggestion_tracker = SuggestionFeedbackTracker(str(suggestion_db_path))
            print(f"[suggestions] Audience suggestion feedback tracking enabled")

        # Phase 9: PersonaSynthesizer for grounded identity prompts
        self.persona_synthesizer = None
        if GROUNDED_PERSONAS_AVAILABLE and PersonaSynthesizer:
            self.persona_synthesizer = PersonaSynthesizer(
                position_ledger=self.position_ledger,
                relationship_tracker=self.relationship_tracker,
                elo_system=self.elo_system,
            )
            print(f"[synthesizer] Grounded persona synthesis enabled")

        # Phase 9: FlipDetector for position reversal tracking (cached instance)
        self.flip_detector = None
        if FLIP_DETECTOR_AVAILABLE:
            try:
                from aragora.insights.flip_detector import FlipDetector
                # Use grounded_positions.db where PositionLedger stores data
                flip_db_path = self.nomic_dir / "grounded_positions.db"
                self.flip_detector = FlipDetector(str(flip_db_path))
                print(f"[flip] Position flip detection enabled")
            except Exception as e:
                print(f"[flip] Initialization failed: {e}")

        # =================================================================
        # Phase 6: Verifiable Reasoning & Robustness Testing
        # =================================================================

        # Phase 6: ClaimsKernel for structured reasoning (P16)
        self.claims_kernel = None
        if CLAIMS_KERNEL_AVAILABLE:
            self.claims_kernel = ClaimsKernel(debate_id=f"nomic-cycle-0")
            print(f"[claims] Structured reasoning enabled")

        # Phase 6: ProvenanceManager for evidence tracking (P17)
        self.provenance_manager = None
        if PROVENANCE_AVAILABLE:
            self.provenance_manager = ProvenanceManager(debate_id=f"nomic-cycle-0")
            print(f"[provenance] Evidence chain tracking enabled")

        # Phase 6: BeliefNetwork for probabilistic reasoning (P18)
        self.belief_network = None
        if BELIEF_NETWORK_AVAILABLE:
            self.belief_network = BeliefNetwork(debate_id=f"nomic-cycle-0")
            print(f"[belief] Probabilistic reasoning enabled")

        # P3-Phase2: Cache for crux injection - store cruxes from one debate to inject into next
        self._cached_cruxes: list = []

        # Phase 6: ProofExecutor for executable verification (P19)
        self.proof_executor = None
        self.claim_verifier = None
        if PROOF_EXECUTOR_AVAILABLE:
            self.proof_executor = ProofExecutor(allow_filesystem=True, default_timeout=30.0)
            self.claim_verifier = ClaimVerifier(self.proof_executor)
            print(f"[proofs] Executable verification enabled")

        # Phase 6: ScenarioComparator for robustness testing (P20)
        self.scenario_comparator = None
        if SCENARIO_MATRIX_AVAILABLE:
            self.scenario_comparator = ScenarioComparator()
            print(f"[scenarios] Robustness testing enabled")

        # Phase 7: Resilience, Living Documents, & Observability

        # Phase 7: EnhancedProvenanceManager for staleness detection (P21)
        # Note: This REPLACES the base ProvenanceManager from Phase 6 if available
        if ENHANCED_PROVENANCE_AVAILABLE:
            self.provenance_manager = EnhancedProvenanceManager(
                debate_id=f"nomic-cycle-0",
                repo_path=str(self.aragora_path)
            )
            print(f"[provenance] Enhanced with staleness detection")

        # Phase 7: CheckpointManager for pause/resume (P22)
        self.checkpoint_manager = None
        if CHECKPOINT_AVAILABLE:
            checkpoint_dir = self.nomic_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            self.checkpoint_manager = CheckpointManager(
                store=FileCheckpointStore(str(checkpoint_dir)),
                config=CheckpointConfig(interval_rounds=1, max_checkpoints=5)
            )
            print(f"[checkpoint] Pause/resume enabled")

        # Phase 7: BreakpointManager for human intervention (P23)
        self.breakpoint_manager = None
        if BREAKPOINT_AVAILABLE and self.require_human_approval:
            self.breakpoint_manager = BreakpointManager(
                config=BreakpointConfig(min_confidence=0.5, max_deadlock_rounds=3)
            )
            print(f"[breakpoints] Human intervention enabled")

        # Phase 7: ReliabilityScorer for confidence scoring (P24)
        self.reliability_scorer = None
        if RELIABILITY_SCORER_AVAILABLE and self.provenance_manager:
            self.reliability_scorer = ReliabilityScorer(provenance=self.provenance_manager)
            print(f"[reliability] Confidence scoring enabled")

        # Phase 7: DebateTracer for audit logs (P25)
        # Note: DebateTracer is created per-debate, so we just store the path
        self.debate_trace_db = None
        self._current_tracer = None  # Created per-debate in _start_debate_trace
        if DEBATE_TRACER_AVAILABLE:
            trace_dir = self.nomic_dir / "traces"
            trace_dir.mkdir(exist_ok=True)
            self.debate_trace_db = str(trace_dir / "debate_traces.db")
            print(f"[tracer] Audit logging enabled")

        # Phase 8: Agent Evolution, Semantic Memory & Advanced Debates

        # Phase 8: PersonaLaboratory for agent evolution (P26)
        self.persona_lab = None
        if PERSONA_LAB_AVAILABLE and PERSONAS_AVAILABLE and self.persona_manager:
            lab_db = self.nomic_dir / "persona_lab.db"
            self.persona_lab = PersonaLaboratory(
                persona_manager=self.persona_manager,
                db_path=str(lab_db)
            )
            print(f"[lab] Persona evolution enabled")

        # Phase 8: SemanticRetriever for pattern matching (P27)
        self.semantic_retriever = None
        if SEMANTIC_RETRIEVER_AVAILABLE:
            retriever_db = self.nomic_dir / "semantic_patterns.db"
            self.semantic_retriever = SemanticRetriever(db_path=str(retriever_db))
            print(f"[semantic] Pattern retrieval enabled")

        # Phase 8: FormalVerificationManager for theorem proving (P28)
        self.formal_verifier = None
        if FORMAL_VERIFICATION_AVAILABLE:
            self.formal_verifier = FormalVerificationManager()
            print(f"[formal] Z3 verification enabled")

        # Phase 8: DebateGraph for DAG-based debates (P29)
        # Note: GraphDebateOrchestrator is created per-debate with specific agents
        self.graph_debate_enabled = False
        if DEBATE_GRAPH_AVAILABLE and GraphDebateOrchestrator:
            self.graph_debate_enabled = True
            print(f"[graph] DAG debate structure enabled")

        # Phase 8: DebateForker for parallel exploration (P30)
        # Note: DebateForker is created per-debate
        self.fork_debate_enabled = False
        if DEBATE_FORKER_AVAILABLE and DebateForker:
            self.fork_debate_enabled = True
            print(f"[forking] Parallel branch exploration enabled")

        # Setup streaming (optional)
        self.stream_emitter = stream_emitter
        if stream_emitter and STREAMING_AVAILABLE and create_nomic_hooks:
            self.stream_hooks = create_nomic_hooks(stream_emitter)
        else:
            self.stream_hooks = {}

        # Add genesis hooks if available
        if self.use_genesis:
            genesis_hooks = create_logging_hooks(lambda msg: self._log(f"    [genesis] {msg}"))
            self.stream_hooks.update(genesis_hooks)

        # Clear log file on start
        with open(self.log_file, "w") as f:
            f.write(f"=== NOMIC LOOP STARTED: {datetime.now().isoformat()} ===\n")

        # Initialize agents
        self._init_agents()

    def _stream_emit(self, hook_name: str, *args, **kwargs) -> None:
        """Emit event to WebSocket stream and persist to Supabase."""
        # Emit to WebSocket stream
        if hook_name in self.stream_hooks:
            try:
                self.stream_hooks[hook_name](*args, **kwargs)
            except Exception as e:
                logger.warning(f"[stream] Hook '{hook_name}' failed: {e}")

        # Persist to Supabase
        if self.persistence and StreamEvent:
            try:
                event = StreamEvent(
                    loop_id=self.loop_id,
                    cycle=self.cycle_count,
                    event_type=hook_name,
                    event_data={"args": [str(a)[:10000] for a in args], "kwargs": {k: str(v)[:10000] for k, v in kwargs.items()}},
                    agent=kwargs.get("agent"),
                )
                # Run async save in background (fire and forget)
                asyncio.get_event_loop().create_task(self.persistence.save_event(event))
            except Exception as e:
                logger.warning(f"[persistence] Event save failed: {e}")

    async def _persist_cycle(self, phase: str, stage: str, success: bool = None,
                              git_commit: str = None, task_description: str = None,
                              error_message: str = None) -> None:
        """Persist cycle state to Supabase."""
        if not self.persistence or not NomicCycle:
            return
        try:
            cycle = NomicCycle(
                loop_id=self.loop_id,
                cycle_number=self.cycle_count,
                phase=phase,
                stage=stage,
                started_at=datetime.utcnow(),
                success=success,
                git_commit=git_commit,
                task_description=task_description,
                error_message=error_message,
            )
            await self.persistence.save_cycle(cycle)
        except Exception as e:
            logger.warning(f"[persistence] Cycle state save failed (cycle={self.cycle_count}, phase={phase}): {e}")

    async def _persist_debate(self, phase: str, task: str, agents: list,
                               transcript: list, consensus_reached: bool,
                               confidence: float, winning_proposal: str = None) -> None:
        """Persist debate artifact to Supabase."""
        if not self.persistence or not DebateArtifact:
            return
        try:
            debate = DebateArtifact(
                loop_id=self.loop_id,
                cycle_number=self.cycle_count,
                phase=phase,
                task=task,
                agents=agents,
                transcript=transcript,
                consensus_reached=consensus_reached,
                confidence=confidence,
                winning_proposal=winning_proposal,
            )
            await self.persistence.save_debate(debate)

            # Also index in embeddings database for future search
            if self.debate_embeddings:
                await self.debate_embeddings.index_debate(debate)
        except Exception as e:
            logger.warning(f"[persistence] Debate artifact save failed (phase={phase}): {e}")

    def _log(self, message: str, also_print: bool = True, phase: str = None, agent: str = None):
        """Log to file and optionally stdout. File is always flushed immediately."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        # Write to file immediately (unbuffered)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
            f.flush()

        if also_print:
            print(message)
            sys.stdout.flush()

        # Also emit to stream for real-time dashboard
        self._stream_emit("on_log_message", message, level="info", phase=phase, agent=agent)

    def _validate_openrouter_fallback(self) -> bool:
        """Check if OpenRouter fallback is available and warn if not.

        Returns True if OpenRouter is configured, False otherwise.
        The nomic loop will still run without it, but rate-limiting
        recovery will be limited to retries only.
        """
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not openrouter_key:
            self._log("⚠️  WARNING: OPENROUTER_API_KEY not set")
            self._log("   OpenRouter fallback will NOT be available for rate limiting")
            self._log("   Set OPENROUTER_API_KEY in .env for automatic fallback")
            self._log("-" * 50)
            return False

        # Key is set - log confirmation
        self._log("✓ OpenRouter fallback configured (rate limit protection enabled)")
        return True

    def _save_state(self, state: dict):
        """Save current state for crash recovery and monitoring."""
        state["saved_at"] = datetime.now().isoformat()
        state["cycle"] = self.cycle_count
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        # Persist circuit breaker state across runs
        try:
            circuit_breaker_path = self.nomic_dir / "circuit_breaker.json"
            with open(circuit_breaker_path, "w") as f:
                json.dump({
                    "failures": self.circuit_breaker.failures,
                    "cooldowns": self.circuit_breaker.cooldowns,
                    "task_failures": dict(self.circuit_breaker.task_failures),
                    "task_cooldowns": dict(self.circuit_breaker.task_cooldowns),
                    "task_success_rate": dict(self.circuit_breaker.task_success_rate),
                    "saved_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception:
            pass  # Don't fail save_state if circuit breaker persistence fails

    def _load_state(self) -> Optional[dict]:
        """Load saved state if exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _check_cycle_deadline(self, cycle_deadline: datetime, current_phase: str) -> bool:
        """
        Check if cycle has exceeded its time budget with escalation warnings.

        Args:
            cycle_deadline: When this cycle should end
            current_phase: Name of current phase for logging

        Returns:
            True if cycle should continue, False if deadline exceeded
        """
        now = datetime.now()

        # Calculate progress
        if self._cycle_start_time:
            elapsed = (now - self._cycle_start_time).total_seconds()
            remaining = (cycle_deadline - now).total_seconds()
            progress_pct = elapsed / self.max_cycle_seconds * 100

            # 50% warning
            if progress_pct >= 50 and not self._warned_50:
                self._log(f"  [WARNING] Cycle at 50% time budget ({elapsed/60:.0f}m elapsed, phase: {current_phase})")
                self._warned_50 = True

            # 75% warning
            if progress_pct >= 75 and not self._warned_75:
                self._log(f"  [WARNING] Cycle at 75% time budget ({remaining/60:.0f}m remaining, phase: {current_phase})")
                self._warned_75 = True

            # 90% critical - enable fast-track mode
            if progress_pct >= 90 and not self._warned_90:
                self._log(f"  [CRITICAL] Cycle at 90% - enabling fast-track mode (phase: {current_phase})")
                self._fast_track_mode = True
                self._warned_90 = True

        if now > cycle_deadline:
            elapsed = (now - (cycle_deadline - timedelta(seconds=self.max_cycle_seconds))).total_seconds()
            self._log(f"  [TIMEOUT] Cycle exceeded {self.max_cycle_seconds}s limit at phase '{current_phase}' ({elapsed:.0f}s elapsed)")
            return False
        return True

    def _record_cycle_outcome(self, outcome: str, details: dict = None):
        """Track cycle outcome for deadlock detection."""
        self._cycle_history.append({
            "cycle": self.cycle_count,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })
        if len(self._cycle_history) > self._max_cycle_history:
            self._cycle_history.pop(0)

    def _detect_cycle_deadlock(self) -> str:
        """Detect if we're stuck in a cycle pattern. Returns deadlock type or empty string."""
        if len(self._cycle_history) < 3:
            return ""

        # Check for repeated same outcome (e.g., design_no_consensus 3 times)
        recent = [h["outcome"] for h in self._cycle_history[-3:]]
        if len(set(recent)) == 1 and recent[0] != "success":
            return f"Repeated failure: {recent[0]} for 3 cycles"

        # Check for oscillating pattern (A-B-A-B)
        if len(self._cycle_history) >= 4:
            last4 = [h["outcome"] for h in self._cycle_history[-4:]]
            if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                return f"Oscillating pattern: {last4[0]} <-> {last4[1]}"

        return ""

    def _track_phase_progress(self, phase: str, round_num: int, consensus: float, changed: bool) -> bool:
        """Track progress within a phase to detect stalls. Returns True if stalled."""
        key = f"{self.cycle_count}_{phase}"
        if key not in self._phase_progress:
            self._phase_progress[key] = []

        self._phase_progress[key].append({
            "round": round_num,
            "consensus": consensus,
            "changed": changed,
            "timestamp": datetime.now()
        })

        # Detect stall: 3+ rounds with <5% consensus change and no position changes
        history = self._phase_progress[key]
        if len(history) >= 3:
            recent = history[-3:]
            consensus_change = abs(recent[-1]["consensus"] - recent[0]["consensus"])
            any_changed = any(r["changed"] for r in recent)

            if consensus_change < 0.05 and not any_changed:
                self._log(f"  [STALL] {phase} stuck for 3 rounds (consensus: {recent[-1]['consensus']:.0%})")
                return True
        return False

    async def _handle_deadlock(self, deadlock_type: str) -> str:
        """Handle detected deadlock with appropriate action. Returns action taken."""
        self._log(f"  [DEADLOCK] Detected: {deadlock_type}")

        if "Repeated failure" in deadlock_type:
            # Clear cached state that might be causing loops
            self._cached_cruxes = [] if hasattr(self, '_cached_cruxes') else []
            self._phase_progress = {}
            self._design_recovery_attempts = set()
            self._log("  [DEADLOCK] Cleared cached state for fresh attempt")

            # Try NomicIntegration counterfactual resolution if belief network available
            if self.nomic_integration and self.nomic_integration._belief_network:
                try:
                    self._log("  [DEADLOCK] Attempting counterfactual resolution via belief network...")
                    belief_network = self.nomic_integration._belief_network
                    contested = belief_network.get_contested_claims()
                    if contested:
                        # Convert BeliefNode list to list for resolve_deadlock
                        self._log(f"  [DEADLOCK] Found {len(contested)} contested claims for resolution")
                        # Store contested claims for use in next debate phase
                        self._cached_cruxes = contested
                except Exception as e:
                    self._log(f"  [DEADLOCK] Counterfactual resolution failed: {e}")

            # Try different agent configuration after multiple deadlocks
            if self._deadlock_count >= 2:
                self._log("  [DEADLOCK] Will rotate agent roles for fresh perspective")
                self._force_judge_consensus = True  # Force judge to break ties

            # Increase consensus threshold decay to lower the bar
            if self._consensus_threshold_decay < 2:
                self._consensus_threshold_decay += 1
                new_threshold = self._get_adaptive_consensus_threshold()
                self._log(f"  [DEADLOCK] Lowered consensus threshold to {new_threshold:.0%}")

            self._deadlock_count += 1
            return "retry_with_reset"

        elif "Oscillating" in deadlock_type:
            # Force judge consensus to break oscillation
            self._log("  [DEADLOCK] Forcing judge consensus mode to break oscillation")
            self._force_judge_consensus = True
            return "force_judge"

        elif self._deadlock_count >= 3:
            # After 3 deadlocks, skip to next improvement
            self._log("  [DEADLOCK] Max retries (3) reached, skipping this improvement")
            return "skip"

        return "continue"

    def _get_adaptive_consensus_threshold(self) -> float:
        """
        Get consensus threshold adjusted for repeated failures.

        Decay path: 0.6 -> 0.5 -> 0.4 (after consecutive no-consensus cycles)
        This allows the system to break deadlocks by accepting lower agreement.
        """
        base_threshold = 0.6
        decay_steps = [0.6, 0.5, 0.4]  # 60% -> 50% -> 40%
        idx = min(self._consensus_threshold_decay, len(decay_steps) - 1)
        threshold = decay_steps[idx]

        if threshold < base_threshold:
            self._log(f"  [consensus] Using adaptive threshold: {threshold:.0%} (decay level {self._consensus_threshold_decay})")

        return threshold

    def _reset_cycle_state(self):
        """Reset per-cycle state at the start of each cycle."""
        self._warned_50 = False
        self._warned_75 = False
        self._warned_90 = False
        self._fast_track_mode = False
        self._design_recovery_attempts = set()
        self._phase_progress = {}
        self._phase_metrics = {}  # Duration vs budget metrics for each phase
        self._cycle_backup_path = None  # Backup path for timeout rollback

    async def _run_with_phase_timeout(self, phase: str, coro, fallback=None):
        """
        Execute a phase coroutine with individual timeout protection.

        Complements the cycle-level timeout by preventing any single phase
        from consuming the entire time budget. Also tracks phase duration
        metrics for analysis and tuning.

        Args:
            phase: Phase name (context, debate, design, implement, verify, commit)
            coro: Async coroutine to execute
            fallback: Optional fallback value on timeout (if None, raises PhaseError)

        Returns:
            Coroutine result or fallback value

        Raises:
            PhaseError: If timeout occurs and no fallback provided
        """
        timeout = PhaseRecovery.PHASE_TIMEOUTS.get(phase, 600)
        self._log(f"  [timeout] Phase '{phase}' has {timeout}s budget")
        self._stream_emit("on_phase_start", phase, timeout)

        # Track phase start time for metrics
        phase_start = time.time()

        try:
            result = await asyncio.wait_for(coro, timeout=timeout)

            # Log duration metrics on success
            duration = time.time() - phase_start
            utilization = (duration / timeout) * 100
            self._log(f"  [{phase}] Completed in {duration:.1f}s ({utilization:.0f}% of {timeout}s budget)")

            # Store metrics for cycle_result (initialize dict if needed)
            if not hasattr(self, '_phase_metrics'):
                self._phase_metrics = {}
            self._phase_metrics[phase] = {
                "duration": round(duration, 1),
                "budget": timeout,
                "utilization": round(utilization, 1),
                "status": "completed",
            }

            return result

        except asyncio.TimeoutError:
            duration = time.time() - phase_start
            elapsed_msg = f"Phase '{phase}' exceeded {timeout}s timeout"
            self._log(f"  [TIMEOUT] {elapsed_msg}")
            logger.warning(f"[phase_timeout] {elapsed_msg}")
            self._stream_emit("on_phase_timeout", phase, timeout)

            # Store timeout metrics
            if not hasattr(self, '_phase_metrics'):
                self._phase_metrics = {}
            self._phase_metrics[phase] = {
                "duration": round(duration, 1),
                "budget": timeout,
                "utilization": 100.0,  # Consumed entire budget
                "status": "timeout",
            }

            if fallback is not None:
                return fallback
            raise PhaseError(phase, f"Timeout after {timeout}s", recoverable=False)

    async def _run_phase_with_recovery(self, phase: str, coro, fallback=None):
        """
        Run a phase with both timeout enforcement and retry recovery.

        Combines _run_with_phase_timeout (for individual phase timeout)
        with PhaseRecovery.run_with_recovery (for retry with exponential backoff).

        Args:
            phase: Phase name (context, debate, design, implement, verify, commit)
            coro: Async coroutine to execute
            fallback: Optional fallback value on timeout/failure

        Returns:
            Coroutine result, fallback value, or raises PhaseError
        """
        async def timeout_wrapped():
            return await self._run_with_phase_timeout(phase, coro, fallback)

        success, result = await self.phase_recovery.run_with_recovery(
            phase=phase,
            phase_func=timeout_wrapped,
        )

        if success:
            return result
        else:
            # result contains error message
            if fallback is not None:
                self._log(f"  [recovery] Phase '{phase}' failed, using fallback")
                return fallback
            raise PhaseError(phase, f"Recovery failed: {result}", recoverable=False)

    async def _check_agent_health(self, agent, agent_name: str) -> bool:
        """
        Quick health check to verify agent is responsive.

        Args:
            agent: The agent object to check
            agent_name: Name for logging

        Returns:
            True if agent responded within timeout
        """
        try:
            # Simple health probe with 15 second timeout
            await asyncio.wait_for(
                agent.generate("Respond with OK to confirm you are ready.", context=[]),
                timeout=15
            )
            self.circuit_breaker.record_success(agent_name)
            return True
        except asyncio.TimeoutError:
            self._log(f"  [health] Agent {agent_name} health check timed out")
            tripped = self.circuit_breaker.record_failure(agent_name)
            if tripped:
                self._log(f"  [circuit-breaker] Agent {agent_name} disabled for {self.circuit_breaker.cooldown_cycles} cycles")
            return False
        except Exception as e:
            self._log(f"  [health] Agent {agent_name} health check failed: {e}")
            self.circuit_breaker.record_failure(agent_name)
            return False

    def _create_backup(self, reason: str = "pre_cycle") -> Path:
        """Create a backup of protected files before making changes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{reason}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        self._log(f"  Creating backup: {backup_name}")

        backed_up = []
        for rel_path in PROTECTED_FILES:
            src = self.aragora_path / rel_path
            if src.exists():
                dst = backup_path / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                backed_up.append(rel_path)
                self._log(f"    Backed up: {rel_path}", also_print=False)

        # Save manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "reason": reason,
            "cycle": self.cycle_count,
            "files": backed_up,
        }
        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self._log(f"  Backup complete: {len(backed_up)} files")
        self._stream_emit("on_backup_created", backup_name, len(backed_up), reason)
        return backup_path

    def _restore_backup(self, backup_path: Path) -> bool:
        """Restore protected files from a backup."""
        manifest_file = backup_path / "manifest.json"
        if not manifest_file.exists():
            self._log(f"  No manifest found in {backup_path}")
            return False

        with open(manifest_file) as f:
            manifest = json.load(f)

        self._log(f"  Restoring backup from {manifest['created_at']}")

        restored = []
        for rel_path in manifest["files"]:
            src = backup_path / rel_path
            dst = self.aragora_path / rel_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored.append(rel_path)
                self._log(f"    Restored: {rel_path}", also_print=False)

        self._log(f"  Restored {len(restored)} files")
        self._stream_emit("on_backup_restored", backup_path.name, len(restored), "verification_failed")
        return True

    def _get_latest_backup(self) -> Optional[Path]:
        """Get the most recent backup directory."""
        backups = sorted(self.backup_dir.iterdir(), reverse=True)
        for backup in backups:
            if (backup / "manifest.json").exists():
                return backup
        return None

    def _verify_protected_files(self) -> List[str]:
        """Verify protected files still exist and are importable."""
        issues = []

        for rel_path in PROTECTED_FILES:
            full_path = self.aragora_path / rel_path
            if not full_path.exists():
                issues.append(f"MISSING: {rel_path}")
                continue

            # Check if Python file is syntactically valid
            if rel_path.endswith(".py"):
                try:
                    result = subprocess.run(
                        ["python", "-m", "py_compile", str(full_path)],
                        capture_output=True,
                        text=True,
                        timeout=30,  # 30s is plenty for syntax check
                    )
                    if result.returncode != 0:
                        issues.append(f"SYNTAX ERROR: {rel_path}")
                except Exception as e:
                    issues.append(f"CHECK FAILED: {rel_path} - {e}")

        return issues

    def _init_agents(self):
        """Initialize agents with distinct personalities and safety awareness."""
        # Common safety footer for all agents
        safety_footer = """

CRITICAL: You are part of a self-improving system. You MUST:
- NEVER propose removing or simplifying core infrastructure (nomic_loop.py, aragora/core.py, debate system)
- ALWAYS prefer adding new features over removing existing ones
- ONLY remove code that is demonstrably BROKEN or HARMFUL
- Preserve backward compatibility in all changes
- If unsure whether to keep functionality, KEEP IT"""

        self.gemini = GeminiAgent(
            name='gemini-visionary',
            model='gemini-3-pro-preview',  # Gemini 3 Pro
            role='proposer',
            timeout=720,  # Doubled to 12 min for thorough codebase exploration
        )
        self.gemini.system_prompt = """You are a visionary product strategist for aragora.
Focus on: viral growth, developer excitement, novel capabilities, bold ideas.

=== STRUCTURED THINKING PROTOCOL ===
When analyzing a task:
1. EXPLORE: First understand the current state - what exists, what's missing
2. ENVISION: Imagine the ideal outcome - what would success look like
3. REASON: Show your thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact

When proposing changes:
- Reference specific files and code patterns you've observed
- Consider what would make aragora famous and widely adopted
- Think about viral growth potential and developer excitement

=== BUILD MODE ===
Your proposals should ADD capabilities, not remove or simplify existing ones.
Aragora should grow more powerful over time, not be stripped down.""" + safety_footer

        self.codex = CodexAgent(
            name='codex-engineer',
            model='gpt-5.2-codex',
            role='proposer',
            timeout=1200,  # Doubled - Codex has known latency issues
        )
        self.codex.system_prompt = """You are a pragmatic engineer for aragora.
Focus on: technical excellence, code quality, practical utility, implementation feasibility.

=== STRUCTURED THINKING PROTOCOL ===
When analyzing code:
1. TRACE: Follow code paths to understand dependencies and data flow
2. ANALYZE: Identify patterns, anti-patterns, and improvement opportunities
3. DESIGN: Consider multiple implementation approaches with pros/cons
4. VALIDATE: Think about edge cases, tests, and failure modes

When proposing changes:
- Show your reasoning chain: "I observed X → which implies Y → so we should Z"
- Reference specific files and line numbers
- Consider impact on tests, performance, and maintainability

=== BUILD MODE ===
Your role is to BUILD and EXTEND, not to remove or break.
Safe refactors: renaming, extracting, improving types.
Unsafe: removing features, breaking APIs.
Reducing technical debt is GOOD when safe (improve code without changing behavior).""" + safety_footer

        self.claude = ClaudeAgent(
            name='claude-visionary',
            model='claude',
            role='proposer',
            timeout=600,  # 10 min - increased for judge role with large context
        )
        self.claude.system_prompt = """You are a visionary architect for aragora.
Focus on: elegant design, user experience, novel AI patterns, system cohesion.

=== STRUCTURED THINKING PROTOCOL ===
When analyzing a task:
1. EXPLORE: First understand the current state - read relevant files, trace code paths
2. PLAN: Design your approach before implementing - consider alternatives
3. REASON: Show your thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact

When using Claude Code:
- Use 'Explore' mode to deeply understand the codebase before proposing
- Use 'Plan' mode to design implementation approaches with user approval
- Ask clarifying questions rather than making assumptions

When proposing changes:
- Reference specific files and architectural patterns
- Consider system cohesion and how parts fit together
- Think about what would make aragora powerful and delightful

=== GUARDIAN ROLE ===
You are a guardian of aragora's core functionality.
Your proposals should ADD capabilities and improve the system.
Never propose removing the nomic loop or core debate infrastructure.""" + safety_footer

        self.grok = GrokCLIAgent(
            name='grok-lateral-thinker',
            model='grok-4',  # Grok 4 full
            role='proposer',
            timeout=1200,  # Doubled to 20 min for thorough codebase exploration
        )
        self.grok.system_prompt = """You are a lateral-thinking synthesizer for aragora.
Focus on: unconventional approaches, novel patterns, creative breakthroughs.

=== STRUCTURED THINKING PROTOCOL ===
When analyzing a task:
1. DIVERGE: Generate multiple unconventional perspectives on the problem
2. CONNECT: Find surprising links between disparate ideas and patterns
3. SYNTHESIZE: Combine insights into novel, coherent proposals
4. GROUND: Anchor creative ideas in practical implementation

When proposing changes:
- Show your lateral thinking: "Others see X, but what if Y..."
- Connect ideas from different domains in surprising ways
- Balance creativity with practicality

=== BUILD MODE ===
Your role is to BUILD and EXTEND, not to remove or break.
Propose additions that unlock new capabilities and create emergent value.
The most valuable proposals are those that others wouldn't think of.""" + safety_footer

        # DeepSeek V3 - latest general model via OpenRouter
        self.deepseek = DeepSeekV3Agent(
            name='deepseek-v3',
            role='proposer',
        )
        self.deepseek.system_prompt = """You are a powerful analytical agent for aragora.
Focus on: comprehensive analysis, practical solutions, efficient implementation.

=== ANALYTICAL PROTOCOL ===
When analyzing a task:
1. UNDERSTAND: Deeply comprehend the problem and its context
2. ANALYZE: Evaluate all aspects systematically
3. DESIGN: Propose well-structured, practical solutions
4. VALIDATE: Ensure solutions are correct and complete

When proposing changes:
- Provide thorough analysis with clear reasoning
- Consider performance, maintainability, and edge cases
- Balance elegance with practicality
- Give concrete, actionable recommendations

=== BUILD MODE ===
Your role is to BUILD and EXTEND, not to remove or break.
Propose additions that are practical, efficient, and well-designed.
The most valuable proposals combine deep analysis with actionable implementation.""" + safety_footer

    def get_current_features(self) -> str:
        """Read current aragora state from the codebase."""
        init_file = self.aragora_path / "aragora" / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            if '"""' in content:
                docstring = content.split('"""')[1]
                return docstring[:2000]
        return "Unable to read current features"

    def get_recent_changes(self) -> str:
        """Get recent git commits."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception:
            return "Unable to read git history"

    def _analyze_failed_branches(self, limit: int = 3) -> str:
        """Analyze recent failed branches for lessons learned.

        This extracts information from preserved failed branches so agents
        can learn from previous failures and avoid repeating them.
        """
        try:
            # List failed branches
            result = subprocess.run(
                ["git", "branch", "--list", "nomic-failed-*"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
            if not branches:
                return ""

            # Get most recent ones (sorted by name which includes timestamp)
            recent = sorted(branches, reverse=True)[:limit]

            lessons = ["## LESSONS FROM RECENT FAILURES"]
            lessons.append("Learn from these previous failed attempts:\n")

            for branch in recent:
                # Get commit message
                msg_result = subprocess.run(
                    ["git", "log", branch, "-1", "--format=%B"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                # Get changed files summary
                files_result = subprocess.run(
                    ["git", "diff", f"main...{branch}", "--stat", "--stat-width=60"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )

                lessons.append(f"**{branch}:**")
                lessons.append(f"```\n{msg_result.stdout[:1000].strip()}")
                if files_result.stdout.strip():
                    lessons.append(f"\nFiles changed:\n{files_result.stdout[:1000].strip()}")
                lessons.append("```\n")

            return "\n".join(lessons)
        except Exception:
            return ""

    def _format_successful_patterns(self, limit: int = 5) -> str:
        """Format successful critique patterns for prompt injection.

        This retrieves patterns from the CritiqueStore that have led to
        successful fixes in previous debates.
        """
        if not hasattr(self, 'critique_store') or not self.critique_store:
            return ""

        try:
            patterns = self.critique_store.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            lines.append("These critique patterns have worked well before:\n")

            for p in patterns:
                lines.append(f"- **{p.issue_type}**: {p.issue_text}")
                if p.suggestion_text:
                    lines.append(f"  → Fix: {p.suggestion_text}")
                lines.append(f"  ({p.success_count} successes)")

            return "\n".join(lines)
        except Exception:
            return ""

    def _format_failure_patterns(self, limit: int = 5) -> str:
        """Format failure patterns to avoid repeating mistakes.

        Uses Titans/MIRAS failure tracking to show patterns that have
        NOT worked well, so agents can avoid repeating them.
        """
        if not hasattr(self, 'critique_store') or not self.critique_store:
            return ""

        try:
            # Query patterns with high failure rates
            conn = self.critique_store.conn if hasattr(self.critique_store, 'conn') else None
            if not conn:
                import sqlite3
                conn = sqlite3.connect(self.critique_store.db_path)

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT issue_type, issue_text, failure_count, success_count
                FROM patterns
                WHERE failure_count > 0
                ORDER BY failure_count DESC
                LIMIT ?
                """,
                (limit,)
            )
            failures = cursor.fetchall()

            if not failures:
                return ""

            lines = ["## PATTERNS TO AVOID (learned from past failures)"]
            lines.append("These approaches have NOT worked well:\n")

            for issue_type, issue_text, fail_count, success_count in failures:
                success_rate = success_count / (success_count + fail_count) if (success_count + fail_count) > 0 else 0
                if success_rate < 0.5:  # Only show patterns with <50% success
                    lines.append(f"- **{issue_type}**: {issue_text}")
                    lines.append(f"  ({fail_count} failures, {success_rate:.0%} success rate)")

            return "\n".join(lines) if len(lines) > 2 else ""
        except Exception:
            return ""

    def _format_continuum_patterns(self, limit: int = 5) -> str:
        """Format patterns from ContinuumMemory for prompt injection.

        Retrieves strategic patterns from the SLOW tier that capture
        successful cycle outcomes and learnings across time.
        """
        if not self.continuum or not CONTINUUM_AVAILABLE:
            return ""

        try:
            # Get recent successful patterns from SLOW tier
            memories = self.continuum.export_for_tier(MemoryTier.SLOW)
            if not memories:
                return ""

            # Filter to successful patterns and sort by importance
            successful = [m for m in memories if m.get("metadata", {}).get("success", False)]
            successful = sorted(successful, key=lambda x: x.get("importance", 0), reverse=True)[:limit]

            if not successful:
                return ""

            lines = ["## STRATEGIC PATTERNS (from ContinuumMemory)"]
            lines.append("Successful patterns learned across cycles:\n")

            for m in successful:
                content = m.get("content", "")
                cycle = m.get("metadata", {}).get("cycle", "?")
                lines.append(f"- Cycle {cycle}: {content}")

            return "\n".join(lines)
        except Exception:
            return ""

    def _format_consensus_history(self, topic: str, limit: int = 3) -> str:
        """Format prior consensus decisions for prompt injection (P1: ConsensusMemory).

        Retrieves similar past debates and their conclusions to avoid
        rehashing settled topics and to surface unaddressed dissents.
        """
        if not self.consensus_memory or not CONSENSUS_MEMORY_AVAILABLE:
            return ""

        try:
            # Find similar past debates
            similar = self.consensus_memory.find_similar_debates(topic, limit=limit)
            if not similar:
                return ""

            lines = ["## HISTORICAL CONSENSUS (from past debates)"]
            lines.append("Previous debates on similar topics:\n")

            for s in similar:
                strength = s.consensus.strength.value if s.consensus.strength else "unknown"
                lines.append(f"- **{s.consensus.topic}** ({strength}, {s.similarity_score:.0%} similar)")
                lines.append(f"  Decision: {s.consensus.conclusion}")
                if s.dissents:
                    lines.append(f"  ⚠️ {len(s.dissents)} dissenting view(s) - consider addressing")

            # Add unaddressed dissents if any
            if self.dissent_retriever:
                context = self.dissent_retriever.retrieve_for_new_debate(topic)
                if context.get("unacknowledged_dissents"):
                    lines.append("\n### Unaddressed Historical Concerns")
                    for d in context["unacknowledged_dissents"][:3]:
                        lines.append(f"- [{d['dissent_type']}] {d['content']}")

                # Add contrarian views - alternative perspectives to consider
                contrarian = self.dissent_retriever.find_contrarian_views(topic, limit=3)
                if contrarian:
                    lines.append("\n### Contrarian Perspectives (Devil's Advocate)")
                    for c in contrarian:
                        lines.append(f"- {c.content} (from {c.agent_id})")

                # Add risk warnings - historical edge cases and concerns
                risks = self.dissent_retriever.find_risk_warnings(topic, limit=3)
                if risks:
                    lines.append("\n### Historical Risk Warnings")
                    for r in risks:
                        lines.append(f"- ⚠️ {r.content}")

            return "\n".join(lines)
        except Exception as e:
            self._log(f"  [consensus] Error formatting history: {e}")
            return ""

    async def _get_pulse_topic_context(self, limit: int = 3) -> str:
        """Get trending topic context to inform debate priorities (Pulse integration).

        Retrieves trending topics from social platforms that may be relevant
        to aragora improvements (e.g., AI safety, multi-agent systems, LLM trends).
        """
        if not self.pulse_manager or not PULSE_AVAILABLE:
            return ""

        try:
            # Fetch trending topics (async)
            trending = await self.pulse_manager.get_trending_topics(
                limit_per_platform=limit,
                filters={
                    "skip_toxic": True,
                    "categories": ["tech", "ai", "programming", "science"],
                }
            )

            if not trending:
                return ""

            # Filter for topics relevant to aragora/AI development
            relevant_keywords = [
                "ai", "llm", "gpt", "claude", "agent", "multi-agent",
                "debate", "consensus", "reasoning", "safety", "alignment",
                "model", "api", "developer", "code", "programming"
            ]

            relevant_topics = [
                t for t in trending
                if any(kw in t.topic.lower() for kw in relevant_keywords)
            ][:3]

            if not relevant_topics:
                return ""

            lines = ["## TRENDING CONTEXT (from Pulse)"]
            lines.append("Current AI/tech trends that may inform improvement priorities:\n")

            for topic in relevant_topics:
                lines.append(f"- **{topic.topic}** ({topic.platform}, {topic.volume} engagement)")
                if topic.category:
                    lines.append(f"  Category: {topic.category}")

            lines.append("\nConsider how aragora improvements could address or leverage these trends.")

            self._log(f"  [pulse] Injected {len(relevant_topics)} trending topics")
            return "\n".join(lines)

        except Exception as e:
            self._log(f"  [pulse] Error fetching trending topics: {e}")
            return ""

    async def _store_debate_consensus(self, result, topic: str) -> None:
        """Store debate consensus for future reference (P1: ConsensusMemory).

        Records the consensus reached, participating agents, and any
        dissenting views for retrieval in future debates.
        """
        if not self.consensus_memory or not CONSENSUS_MEMORY_AVAILABLE:
            return

        try:
            if not result.consensus_reached:
                return

            # Determine consensus strength from confidence
            if result.confidence >= 0.95:
                strength = ConsensusStrength.UNANIMOUS
            elif result.confidence >= 0.8:
                strength = ConsensusStrength.STRONG
            elif result.confidence >= 0.6:
                strength = ConsensusStrength.MODERATE
            elif result.confidence >= 0.5:
                strength = ConsensusStrength.WEAK
            else:
                strength = ConsensusStrength.SPLIT

            # Get participating agents
            agents = [self.gemini.name, self.codex.name, self.claude.name, self.grok.name, self.deepseek.name]

            # Store the consensus (full content, no truncation)
            record = self.consensus_memory.store_consensus(
                topic=topic,
                conclusion=result.final_answer if result.final_answer else "",
                strength=strength,
                confidence=result.confidence,
                participating_agents=agents,
                agreeing_agents=agents,  # All participate in consensus
                domain="aragora_improvement",
                debate_duration=result.duration_seconds,
                rounds=result.rounds_used,
                metadata={"cycle": self.cycle_count},
            )

            self._log(f"  [consensus] Stored consensus: {strength.value} ({result.confidence:.0%})")

            # Store any dissenting views from the result
            dissenting = getattr(result, 'dissenting_views', [])
            for i, view in enumerate(dissenting):
                from aragora.memory.consensus import DissentType
                self.consensus_memory.store_dissent(
                    debate_id=record.id,
                    agent_id=f"agent_{i}",
                    dissent_type=DissentType.ALTERNATIVE_APPROACH,
                    content=view,
                    reasoning="Minority view from debate",
                    confidence=0.5,
                )

        except Exception as e:
            self._log(f"  [consensus] Error storing: {e}")

    def _record_calibration_from_debate(self, result, agents: list, domain: str = "general") -> None:
        """Record calibration data from debate votes/predictions.

        Tracks how well agents' confidence aligns with actual outcomes.
        An agent's vote confidence vs whether consensus was reached indicates
        calibration quality.
        """
        if not self.calibration_tracker or not CALIBRATION_AVAILABLE:
            return

        try:
            consensus_reached = result.consensus_reached
            debate_id = getattr(result, 'debate_id', f"debate-{self.cycle_count}")

            # Get agent votes if available
            votes = getattr(result, 'votes', {})
            if not votes:
                # Fallback: use result confidence as proxy for all agents
                for agent in agents:
                    agent_name = agent.name if hasattr(agent, 'name') else str(agent)
                    # Agent "predicted" consensus would happen with result.confidence
                    confidence = result.confidence if consensus_reached else 0.5
                    self.calibration_tracker.record_prediction(
                        agent=agent_name,
                        confidence=confidence,
                        correct=consensus_reached,
                        domain=domain,
                        debate_id=debate_id,
                    )
                return

            # Use actual vote data if available
            for agent_name, vote_data in votes.items():
                if isinstance(vote_data, dict):
                    confidence = vote_data.get('confidence', 0.5)
                    # Was their vote aligned with the outcome?
                    correct = consensus_reached if confidence >= 0.5 else not consensus_reached
                else:
                    confidence = float(vote_data) if vote_data else 0.5
                    correct = consensus_reached

                self.calibration_tracker.record_prediction(
                    agent=agent_name,
                    confidence=confidence,
                    correct=correct,
                    domain=domain,
                    debate_id=debate_id,
                )

            self._log(f"  [calibration] Recorded {len(votes)} agent predictions")

        except Exception as e:
            self._log(f"  [calibration] Error recording: {e}")

    def _record_suggestion_feedback(
        self,
        result,
        debate_id: str,
        suggestions_injected: list = None,
    ) -> None:
        """Record audience suggestion effectiveness after debate completion.

        Tracks whether debates with audience suggestions achieved consensus
        and updates contributor reputation scores accordingly.
        """
        if not self.suggestion_tracker or not SUGGESTION_FEEDBACK_AVAILABLE:
            return

        try:
            # Record outcome for any suggestions that were injected
            updated = self.suggestion_tracker.record_outcome(
                debate_id=debate_id,
                consensus_reached=result.consensus_reached,
                consensus_confidence=result.confidence,
                duration_seconds=getattr(result, 'duration_seconds', 0.0),
            )

            if updated > 0:
                self._log(f"  [suggestions] Updated {updated} suggestion(s) with outcome")

                # Log effectiveness stats periodically
                if self.cycle_count % 10 == 0:
                    stats = self.suggestion_tracker.get_effectiveness_stats()
                    if stats.get('total_suggestions', 0) > 0:
                        self._log(f"  [suggestions] Overall stats: {stats['total_suggestions']} suggestions, "
                                 f"{stats['avg_effectiveness']:.0%} avg effectiveness")

        except Exception as e:
            self._log(f"  [suggestions] Error recording feedback: {e}")

    def _record_suggestion_injection(
        self,
        debate_id: str,
        clusters: list,
    ) -> list[str]:
        """Record which suggestions were injected into a debate.

        Args:
            debate_id: Unique debate identifier
            clusters: List of SuggestionCluster objects

        Returns:
            List of injection IDs for tracking
        """
        if not self.suggestion_tracker or not SUGGESTION_FEEDBACK_AVAILABLE:
            return []

        try:
            injection_ids = self.suggestion_tracker.record_injection(debate_id, clusters)
            if injection_ids:
                self._log(f"  [suggestions] Recorded {len(injection_ids)} suggestion cluster(s) for tracking")
            return injection_ids
        except Exception as e:
            self._log(f"  [suggestions] Error recording injection: {e}")
            return []

    async def _extract_and_store_insights(self, result) -> None:
        """Extract and store insights from debate result (P2: InsightExtractor).

        Analyzes the debate to extract patterns, agent performances,
        and key takeaways to feed into learning systems.
        """
        if not self.insight_extractor or not INSIGHTS_AVAILABLE:
            return

        try:
            # Extract insights from the debate result
            insights = await self.insight_extractor.extract(result)

            self._log(f"  [insights] Extracted {insights.total_insights} insights: {insights.key_takeaway}")

            # Persist insights to InsightStore database (debate consensus feature)
            if self.insight_store and insights:
                try:
                    stored = await self.insight_store.store_debate_insights(insights)
                    self._log(f"  [insights] Persisted {stored} insights to database")
                except Exception as e:
                    self._log(f"  [insights] Persistence error: {e}")

            # Feed key takeaway to ContinuumMemory for long-term learning
            if self.continuum and insights.key_takeaway:
                self.continuum.add(
                    id=f"insight-{self.cycle_count}-debate",
                    content=insights.key_takeaway,
                    tier=MemoryTier.MEDIUM,
                    importance=insights.consensus_insight.confidence if insights.consensus_insight else 0.5,
                    metadata={
                        "type": "debate_insight",
                        "cycle": self.cycle_count,
                        "consensus_reached": insights.consensus_reached,
                    },
                )

            # Update agent reputations based on extracted performances
            if self.critique_store and insights.agent_performances:
                for perf in insights.agent_performances:
                    # Update reputation with more detailed metrics
                    self.critique_store.update_reputation(
                        perf.agent_name,
                        proposal_made=perf.proposals_made > 0,
                        proposal_accepted=perf.proposal_accepted,
                    )

            # Store pattern insights to ContinuumMemory if significant
            for pattern in insights.pattern_insights:
                if pattern.confidence > 0.7 and self.continuum:
                    self.continuum.add(
                        id=f"pattern-{self.cycle_count}-{pattern.id[:8]}",
                        content=f"Pattern: {pattern.title} - {pattern.description}",
                        tier=MemoryTier.SLOW,
                        importance=pattern.confidence,
                        metadata={
                            "type": "pattern_insight",
                            "cycle": self.cycle_count,
                            "category": pattern.metadata.get("category", "general"),
                        },
                    )

            # Persist insights to InsightStore for dashboard access (debate consensus feature)
            if self.insight_store and insights:
                try:
                    await self.insight_store.store_debate_insights(insights)
                    self._log(f"  [insights] Persisted {insights.total_insights} insights to store")
                except Exception as store_err:
                    self._log(f"  [insights] Store error: {store_err}")

        except Exception as e:
            self._log(f"  [insights] Error extracting: {e}")

    # =========================================================================
    # Phase 3 Helper Methods
    # =========================================================================

    def _format_agent_memories(self, agent_name: str, task: str, limit: int = 3) -> str:
        """Format per-agent relevant memories for prompt injection (P3: MemoryStream)."""
        if not self.memory_stream or not MEMORY_STREAM_AVAILABLE:
            return ""
        try:
            memories = self.memory_stream.retrieve(
                agent_name=agent_name,
                query=task[:200],
                limit=limit
            )
            if not memories:
                return ""
            lines = [f"## Your memories ({agent_name}):"]
            for m in memories:
                content = m.memory.content if hasattr(m, 'memory') else str(m)
                lines.append(f"- {content}...")
            return "\n".join(lines)
        except Exception as e:
            self._log(f"  [memory] Error retrieving memories: {e}")
            return ""

    def _format_position_history(self, agent_name: str, topic: str, limit: int = 5) -> str:
        """Format recent positions for prompt injection (P9: PositionLedger read)."""
        if not self.position_ledger or not GROUNDED_PERSONAS_AVAILABLE:
            return ""
        try:
            positions = self.position_ledger.get_agent_positions(agent_name, limit=limit)
            if not positions:
                return ""

            lines = [f"## Your Recent Positions ({agent_name}):"]
            lines.append("Review these to maintain consistency or explain any changes:")

            reversed_count = sum(1 for p in positions if p.reversed)
            if reversed_count > 0:
                lines.append(f"⚠️ You have reversed {reversed_count} position(s) recently. If changing stance, explain why.")

            for p in positions:
                status = ""
                if p.reversed:
                    status = " [REVERSED]"
                elif p.outcome == "correct":
                    status = " ✓"
                elif p.outcome == "incorrect":
                    status = " ✗"

                conf_pct = f"{p.confidence:.0%}" if p.confidence else "?"
                domain_str = f" [{p.domain}]" if p.domain else ""
                lines.append(f"- {p.claim[:80]}...{domain_str} (conf: {conf_pct}){status}")

            return "\n".join(lines)
        except Exception as e:
            self._log(f"  [positions] Error retrieving positions: {e}")
            return ""

    async def _retrieve_relevant_insights(self, topic: str, limit: int = 5) -> str:
        """Retrieve relevant past insights for debate context (P2: InsightStore)."""
        if not self.insight_store or not INSIGHTS_AVAILABLE:
            return ""
        try:
            lines = ["## Learnings from Past Debates"]

            # Get common patterns that recur across debates
            if hasattr(self.insight_store, 'get_common_patterns'):
                patterns = await self.insight_store.get_common_patterns(min_occurrences=2, limit=3)
                if patterns:
                    lines.append("\n### Recurring Patterns:")
                    for p in patterns:
                        lines.append(f"- {p.get('pattern', '')} (seen {p.get('occurrences', 0)}x)")

            # Get recent insights
            if hasattr(self.insight_store, 'get_recent_insights'):
                recent = await self.insight_store.get_recent_insights(limit=limit)
                if recent:
                    lines.append("\n### Recent Insights:")
                    for insight in recent[:3]:
                        insight_type = getattr(insight, 'type', None)
                        type_str = insight_type.value if hasattr(insight_type, 'value') else str(insight_type or 'insight')
                        title = getattr(insight, 'title', '')
                        desc = getattr(insight, 'description', '')[:100]
                        lines.append(f"- [{type_str}] {title}: {desc}...")

            if len(lines) > 1:
                return "\n".join(lines)
            return ""
        except Exception as e:
            self._log(f"  [insights] Retrieval error: {e}")
            return ""

    async def _retrieve_similar_debates(self, topic: str, limit: int = 3) -> str:
        """Retrieve similar past debates for historical context."""
        if not self.debate_embeddings or not EMBEDDINGS_AVAILABLE:
            return ""
        try:
            if not hasattr(self.debate_embeddings, 'find_similar_debates'):
                return ""

            similar = await self.debate_embeddings.find_similar_debates(
                query=topic[:200],
                limit=limit,
                min_similarity=0.7
            )
            if not similar:
                return ""

            lines = ["## Similar Past Debates"]
            for item in similar:
                if isinstance(item, dict):
                    debate_id = item.get('debate_id', 'unknown')
                    excerpt = item.get('excerpt', '')[:300]
                    similarity = item.get('similarity', 0)
                elif isinstance(item, tuple) and len(item) >= 3:
                    debate_id, excerpt, similarity = item[0], item[1][:300], item[2]
                else:
                    continue
                lines.append(f"\n### {debate_id} (similarity: {similarity:.0%})")
                lines.append(excerpt + "..." if len(excerpt) >= 300 else excerpt)

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as e:
            self._log(f"  [embeddings] Similar debate retrieval error: {e}")
            return ""

    async def _record_agent_memories(self, result, task: str) -> None:
        """Record observations to per-agent memory streams (P3: MemoryStream)."""
        if not self.memory_stream or not MEMORY_STREAM_AVAILABLE:
            return
        try:
            # Identify winning agents
            winning_agents = set()
            if result.final_answer:
                for agent in [self.gemini, self.codex, self.claude, self.grok, self.deepseek]:
                    if agent.name.lower() in result.final_answer.lower():
                        winning_agents.add(agent.name)

            # Record each agent's observations
            for msg in result.messages:
                agent = getattr(msg, 'agent', None)
                if not agent and isinstance(msg, dict):
                    agent = msg.get('agent')
                if agent:
                    importance = 0.7 if agent in winning_agents else 0.5
                    content = getattr(msg, 'content', str(msg))
                    self.memory_stream.observe(
                        agent_name=agent,
                        content=f"Debated '{task}': {content}",
                        debate_id=f"cycle-{self.cycle_count}",
                        importance=importance
                    )
            self._log(f"  [memory] Recorded {len(result.messages)} observations")
        except Exception as e:
            self._log(f"  [memory] Error recording: {e}")

    async def _gather_codebase_evidence(self, task: str, limit: int = 5) -> str:
        """Gather relevant evidence from codebase for debate context (P4: LocalDocsConnector)."""
        if not self.local_docs or not LOCAL_DOCS_AVAILABLE:
            return ""
        try:
            evidence = await self.local_docs.search(query=task[:200], limit=limit)
            if not evidence:
                return ""
            lines = ["## Relevant Codebase Evidence:"]
            for e in evidence:
                source = getattr(e, 'source', 'unknown')
                content = getattr(e, 'content', str(e))
                lines.append(f"- [{source}]: {content}")
            return "\n".join(lines)
        except Exception as e:
            self._log(f"  [evidence] Error gathering: {e}")
            return ""

    async def _handle_debate_deadlock(self, result, arena, task: str):
        """Fork debate on disputed assumptions if deadlocked (P5: CounterfactualOrchestrator)."""
        if not self.counterfactual or not COUNTERFACTUAL_AVAILABLE:
            return result

        # Only handle if actually deadlocked
        if result.consensus_reached and result.confidence >= 0.5:
            return result

        try:
            # Find pivot claim from dissenting views
            pivot = await self.counterfactual.detect_pivot_claim(result)
            if not pivot or not pivot.should_branch:
                return result

            self._log(f"  [counterfactual] Forking on: {pivot.statement}")

            # Fork into branches
            branches = await self.counterfactual.fork_on_claim(
                arena=arena,
                pivot_claim=pivot,
                parent_result=result
            )

            # Synthesize conditional consensus
            conditional = await self.counterfactual.synthesize_branches(branches)
            self._log(f"  [counterfactual] Conditional consensus: {conditional.summary}")

            # Update result with conditional consensus
            result.final_answer = conditional.summary
            result.consensus_reached = True
            result.confidence = conditional.confidence
            if not hasattr(result, 'metadata') or result.metadata is None:
                result.metadata = {}
            result.metadata["conditional"] = True
            result.metadata["branches"] = len(branches)

            return result
        except Exception as e:
            self._log(f"  [counterfactual] Error: {e}")
            return result

    async def _run_agent_for_probe(self, agent, prompt: str) -> str:
        """Run an agent with a probe prompt, handling errors gracefully.

        Used by CapabilityProber to execute probes against agents.
        """
        try:
            response = await self._call_agent_with_retry(agent, prompt, max_retries=1)
            return response if response else "[No response]"
        except Exception as e:
            self._log(f"  [prober] Agent {agent.name} probe failed: {e}")
            return f"[Error: {e}]"

    async def _probe_agent_capabilities(self) -> None:
        """Run capability probes on agents to detect weaknesses (P6: CapabilityProber).

        Now runs every 2 cycles (was every 5) for better agent quality tracking.
        """
        if not self.prober or not PROBER_AVAILABLE:
            return
        if self.cycle_count % 2 != 0:  # Run every 2 cycles for faster feedback
            return

        try:
            self._log(f"  [prober] Running capability probes...")
            agents = [self.gemini, self.codex, self.claude, self.grok, self.deepseek]

            for agent in agents:
                if agent is None:
                    continue
                # Create a closure to capture the current agent
                async def run_fn(prompt: str, _agent=agent) -> str:
                    return await self._run_agent_for_probe(_agent, prompt)

                report = await self.prober.probe_agent(
                    target_agent=agent,
                    run_agent_fn=run_fn,
                    probe_types=[ProbeType.CONTRADICTION, ProbeType.HALLUCINATION],
                    probes_per_type=2,  # Reduced for speed
                )
                if report and report.vulnerabilities_found > 0:
                    self._log(f"  [prober] {agent.name}: {report.vulnerabilities_found} vulnerabilities found")
                    # Log detailed findings
                    if hasattr(report, 'findings') and report.findings:
                        for finding in report.findings[:3]:  # Top 3
                            desc = getattr(finding, 'description', str(finding))[:100]
                            self._log(f"    - {desc}")
        except Exception as e:
            self._log(f"  [prober] Error: {e}")

    def _select_debate_template(self, task: str):
        """Select appropriate debate template based on task content (P7: DebateTemplates)."""
        if not TEMPLATES_AVAILABLE:
            return None
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["code review", "review code", "pr review"]):
            return CODE_REVIEW_TEMPLATE
        if any(kw in task_lower for kw in ["design", "architecture", "rfc"]):
            return DESIGN_DOC_TEMPLATE
        return None  # Use default debate format

    def _apply_template_to_protocol(self, protocol, template) -> None:
        """Modify protocol based on template settings (P7: DebateTemplates)."""
        if template and hasattr(template, 'max_rounds'):
            protocol.rounds = min(protocol.rounds, template.max_rounds)
            if hasattr(template, 'consensus_threshold'):
                # Store threshold for later use
                protocol.consensus_threshold = template.consensus_threshold
            self._log(f"  [template] Using {template.name}")

    # =========================================================================
    # Phase 4 Helper Methods: Agent Evolution Mechanisms
    # =========================================================================

    def _init_agent_personas(self) -> None:
        """Initialize or load personas for all agents (P8: PersonaManager)."""
        if not self.persona_manager or not PERSONAS_AVAILABLE:
            return
        try:
            for agent in [self.gemini, self.codex, self.claude, self.grok, self.deepseek]:
                persona = get_or_create_persona(self.persona_manager, agent.name)
                top_exp = persona.top_expertise[:2] if persona.top_expertise else []
                self._log(f"  [persona] {agent.name}: {persona.trait_string}, top: {top_exp}")
        except Exception as e:
            self._log(f"  [persona] Error initializing: {e}")

    def _record_persona_performance(self, result, task: str) -> None:
        """Update persona expertise based on debate outcome (P8: PersonaManager)."""
        if not self.persona_manager or not PERSONAS_AVAILABLE:
            return
        try:
            # Detect domain from task
            task_lower = task.lower()
            domain = None
            for d in EXPERTISE_DOMAINS:
                if d in task_lower:
                    domain = d
                    break
            if not domain:
                domain = "architecture"  # default

            # Track unique agents that participated
            participating_agents = set()
            for msg in result.messages:
                agent = getattr(msg, 'agent', None) or (msg.get('agent') if isinstance(msg, dict) else None)
                if agent:
                    participating_agents.add(agent)

            # Record performance for each unique agent
            success = result.consensus_reached and result.confidence >= 0.6
            for agent_name in participating_agents:
                self.persona_manager.record_performance(
                    agent_name=agent_name,
                    domain=domain,
                    success=success,
                    debate_id=f"cycle-{self.cycle_count}"
                )
            self._log(f"  [persona] Recorded performance in {domain} for {len(participating_agents)} agents")
        except Exception as e:
            self._log(f"  [persona] Error: {e}")

    def _get_persona_context(self, agent_name: str) -> str:
        """Get persona context for injection into agent prompts (P8: PersonaManager)."""
        if not self.persona_manager or not PERSONAS_AVAILABLE:
            return ""
        try:
            persona = self.persona_manager.get_persona(agent_name)
            if persona:
                return persona.to_prompt_context()
            return ""
        except Exception:
            return ""

    async def _extract_and_store_patterns(self, result) -> None:
        """Extract winning patterns from successful debates (P9: PromptEvolver)."""
        if not self.prompt_evolver or not EVOLVER_AVAILABLE:
            return
        if not result.consensus_reached or result.confidence < 0.6:
            return  # Only learn from successful debates
        try:
            patterns = self.prompt_evolver.extract_winning_patterns([result])
            if patterns:
                self.prompt_evolver.store_patterns(patterns)
                self._log(f"  [evolver] Extracted {len(patterns)} patterns from debate")
        except Exception as e:
            self._log(f"  [evolver] Error extracting patterns: {e}")

    async def _evolve_agent_prompts(self) -> None:
        """Evolve agent prompts based on accumulated patterns (P9: PromptEvolver)."""
        if not self.prompt_evolver or not EVOLVER_AVAILABLE:
            return
        if self.cycle_count % 10 != 0:  # Run every 10 cycles
            return

        try:
            self._log(f"  [evolver] Evolving agent prompts...")
            patterns = self.prompt_evolver.get_top_patterns(limit=5)
            if not patterns:
                self._log(f"  [evolver] No patterns accumulated yet")
                return

            for agent in [self.gemini, self.codex, self.claude, self.grok, self.deepseek]:
                if hasattr(agent, 'system_prompt') and agent.system_prompt:
                    self.prompt_evolver.apply_evolution(agent, patterns)
                    version = self.prompt_evolver.get_prompt_version(agent.name)
                    if version:
                        self._log(f"  [evolver] {agent.name}: Evolved to v{version.version}")
        except Exception as e:
            self._log(f"  [evolver] Error evolving prompts: {e}")

    def _update_prompt_performance(self, agent_name: str, result) -> None:
        """Update performance metrics for current prompt version (P9: PromptEvolver)."""
        if not self.prompt_evolver or not EVOLVER_AVAILABLE:
            return
        try:
            version = self.prompt_evolver.get_prompt_version(agent_name)
            if version:
                self.prompt_evolver.update_performance(agent_name, version.version, result)
        except Exception:
            pass

    async def _run_tournament_if_due(self) -> None:
        """Run a tournament to benchmark agents if interval reached (P10: Tournament)."""
        if not TOURNAMENT_AVAILABLE or not Tournament:
            return
        if self.cycle_count - self.last_tournament_cycle < self.tournament_interval:
            return

        try:
            self._log(f"\n=== TOURNAMENT (Cycle {self.cycle_count}) ===")
            agents = [self.gemini, self.codex, self.claude, self.grok, self.deepseek]
            tasks = create_default_tasks()[:3]  # Use 3 tasks for speed

            tournament = Tournament(
                name=f"Cycle-{self.cycle_count}-Tournament",
                agents=agents,
                tasks=tasks,
                format=TournamentFormat.FREE_FOR_ALL,
                db_path=str(self.nomic_dir / "tournaments.db")
            )

            # Define debate runner for tournament
            async def run_tournament_debate(env, debate_agents):
                arena = Arena(
                    environment=env,  # Required first parameter
                    agents=debate_agents,
                    protocol=DebateProtocol(
                        rounds=3,
                        role_rotation=True,
                        role_rotation_config=RoleRotationConfig(
                            enabled=True,
                            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC, CognitiveRole.ADVOCATE],
                        ),
                    ),
                    position_tracker=self.position_tracker,
                    calibration_tracker=self.calibration_tracker,
                    event_hooks=self._create_arena_hooks("tournament"),
                    event_emitter=self.stream_emitter,
                    loop_id=self.loop_id,
                    persona_manager=self.persona_manager,
                    relationship_tracker=self.relationship_tracker,
                    moment_detector=self.moment_detector,
                    continuum_memory=self.continuum,
                )
                return await arena.run()  # run() takes no arguments

            result = await tournament.run(run_tournament_debate, parallel=False)
            self.last_tournament_cycle = self.cycle_count

            # Log standings
            self._log(f"  [tournament] Champion: {result.champion}")
            for i, standing in enumerate(result.standings[:4]):
                self._log(f"  [tournament] #{i+1} {standing.agent_name}: {standing.points}pts, {standing.win_rate:.0%} win rate")

            # Update persona expertise based on tournament performance
            if self.persona_manager and PERSONAS_AVAILABLE:
                for standing in result.standings:
                    for task in tasks:
                        self.persona_manager.record_performance(
                            agent_name=standing.agent_name,
                            domain=task.domain,
                            success=standing.win_rate > 0.5,
                            debate_id=f"tournament-{self.cycle_count}"
                        )
        except Exception as e:
            self._log(f"  [tournament] Error: {e}")

    # =========================================================================
    # Phase 5 Helper Methods: Efficiency, Process Feedback, Agent Ranking
    # =========================================================================

    def _check_debate_convergence(
        self,
        current_responses: dict,
        previous_responses: dict,
        round_number: int
    ):
        """Check if debate has converged and can stop early (P11: ConvergenceDetector)."""
        if not self.convergence_detector or not CONVERGENCE_AVAILABLE:
            return None
        try:
            result = self.convergence_detector.check_convergence(
                current_responses, previous_responses, round_number
            )
            if result and result.converged:
                self._log(f"  [convergence] Debate converged! "
                         f"(avg similarity: {result.avg_similarity:.0%})")
            return result
        except Exception as e:
            self._log(f"  [convergence] Error: {e}")
            return None

    def _analyze_debate_process(self, result):
        """Analyze debate process for issues and recommendations (P12: MetaCritiqueAnalyzer)."""
        if not self.meta_analyzer or not META_CRITIQUE_AVAILABLE:
            return None
        try:
            critique = self.meta_analyzer.analyze(result)
            self._log(f"  [meta] Debate quality: {critique.overall_quality:.0%}")

            # P5-Phase2: Cache observations and quality for reflection injection
            self._last_meta_quality = critique.overall_quality
            if critique.observations:
                issues = [o for o in critique.observations if o.observation_type == "issue"]
                if issues:
                    self._log(f"  [meta] Issues found: {len(issues)}")
                    self._cached_meta_observations = issues[:5]  # Cache top 5 issues
                    # Log warning if quality is low
                    if critique.overall_quality < 0.6:
                        self._log(f"  [meta] ⚠️ LOW QUALITY: Reflection needed")

            if critique.recommendations:
                self._log(f"  [meta] Top recommendation: {critique.recommendations[0]}")
            return critique
        except Exception as e:
            self._log(f"  [meta] Error: {e}")
            return None

    def _format_meta_observations(self) -> str:
        """Format cached meta-critique observations for injection (P5-Phase2: MetaCritique Reflection)."""
        if not self._cached_meta_observations or self._last_meta_quality >= 0.6:
            return ""  # Only inject observations when quality was low
        try:
            lines = ["=== PROCESS REFLECTION (from previous debate) ==="]
            lines.append("The previous debate had the following issues to avoid:\n")
            for i, obs in enumerate(self._cached_meta_observations[:3], 1):
                desc = getattr(obs, 'description', str(obs))
                lines.append(f"{i}. {desc}")
            lines.append("\nPlease actively avoid these anti-patterns in this debate.")
            return "\n".join(lines)
        except Exception:
            return ""

    def _store_meta_recommendations(self, critique) -> None:
        """Store meta-critique recommendations for future cycle improvement (P12)."""
        if not critique or not hasattr(critique, 'recommendations') or not critique.recommendations:
            return
        # Store in ConsensusMemory as settled insight
        if self.consensus_memory and CONSENSUS_MEMORY_AVAILABLE and ConsensusStrength:
            try:
                for rec in critique.recommendations[:2]:
                    self.consensus_memory.store_consensus(
                        topic=f"process-recommendation-{self.cycle_count}",
                        conclusion=rec,
                        strength=ConsensusStrength.MODERATE,
                        confidence=critique.overall_quality,
                        participating_agents=["meta-critic"],
                        agreeing_agents=["meta-critic"],
                        domain="process-improvement"
                    )
            except Exception as e:
                self._log(f"  [meta] Error storing recommendations: {e}")

    def _detect_domain(self, task: str) -> str:
        """Detect task domain from content (P13: EloSystem helper)."""
        task_lower = task.lower()
        domains = ["security", "performance", "architecture", "testing", "error_handling"]
        for d in domains:
            if d in task_lower:
                return d
        return "general"

    def _agent_in_consensus(self, agent_name: str, result) -> bool:
        """Check if agent's position was part of winning consensus."""
        if not result.consensus_reached:
            return False
        # Check if agent voted for winning choice
        if hasattr(result, 'votes'):
            for vote in result.votes:
                vote_agent = getattr(vote, 'agent', None) or (vote.get('agent') if isinstance(vote, dict) else None)
                vote_choice = getattr(vote, 'choice', None) or (vote.get('choice') if isinstance(vote, dict) else None)
                if vote_agent == agent_name:
                    if vote_choice and result.final_answer and vote_choice in result.final_answer:
                        return True
        return False

    def _extract_position_changes(self, result) -> dict[str, list[str]]:
        """Extract position changes from debate messages.

        Detects when an agent changes their position after another agent's message.
        Returns: {agent_who_changed: [agents_who_influenced_them]}
        """
        position_changes: dict[str, list[str]] = {}
        if not hasattr(result, 'messages') or not result.messages:
            return position_changes

        try:
            last_speaker = None
            change_indicators = [
                "i agree with", "you're right", "that's a good point",
                "i've reconsidered", "on reflection", "you've convinced me",
                "changing my position", "i now think", "fair point",
            ]

            for msg in result.messages:
                agent = getattr(msg, 'agent', None) or (msg.get('agent') if isinstance(msg, dict) else None)
                if not agent:
                    continue
                content = getattr(msg, 'content', str(msg))[:500].lower()

                if any(ind in content for ind in change_indicators):
                    if last_speaker and last_speaker != agent:
                        if agent not in position_changes:
                            position_changes[agent] = []
                        if last_speaker not in position_changes[agent]:
                            position_changes[agent].append(last_speaker)
                last_speaker = agent
        except Exception:
            pass
        return position_changes

    def _record_elo_match(self, result, task: str) -> None:
        """Record debate as ELO match to update agent ratings (P13: EloSystem)."""
        if not self.elo_system or not ELO_AVAILABLE:
            return
        try:
            # Extract participants from result
            participants = []
            scores = {}
            for msg in result.messages:
                agent = getattr(msg, 'agent', None) or (msg.get('agent') if isinstance(msg, dict) else None)
                if agent and agent not in participants:
                    participants.append(agent)
                    # Score based on whether agent's view prevailed
                    scores[agent] = result.confidence if result.consensus_reached else 0.5

            if len(participants) >= 2:
                domain = self._detect_domain(task)

                # Calculate confidence weight from probe results (P13: probe → ELO feedback)
                confidence_weight = 1.0
                if hasattr(self, '_last_probe_weights') and self._last_probe_weights:
                    weights = [self._last_probe_weights.get(p, 0.7) for p in participants]
                    confidence_weight = sum(weights) / len(weights) if weights else 1.0
                    self._log(f"  [elo] Confidence weight from probes: {confidence_weight:.2f}")

                changes = self.elo_system.record_match(
                    debate_id=f"cycle-{self.cycle_count}",
                    participants=participants,
                    scores=scores,
                    domain=domain,
                    confidence_weight=confidence_weight,
                )
                self._log(f"  [elo] Updated ratings for {len(participants)} agents in {domain}")

                # Emit match_recorded event for real-time leaderboard updates (debate consensus feature)
                winner = max(changes, key=changes.get) if changes else None
                self._stream_emit(
                    "on_match_recorded",
                    debate_id=f"cycle-{self.cycle_count}",
                    participants=participants,
                    elo_changes=changes,
                    domain=domain,
                    winner=winner,
                    loop_id=self.loop_id,
                )
        except Exception as e:
            self._log(f"  [elo] Error: {e}")

    def _log_elo_leaderboard(self) -> None:
        """Log current ELO leaderboard (P13: EloSystem)."""
        if not self.elo_system or not ELO_AVAILABLE:
            return
        if self.cycle_count % 5 != 0:  # Every 5 cycles
            return
        try:
            leaderboard = self.elo_system.get_leaderboard(limit=4)
            self._log(f"  [elo] === LEADERBOARD ===")
            for i, rating in enumerate(leaderboard):
                self._log(f"  [elo] #{i+1} {rating.agent_name}: {rating.elo:.0f} "
                         f"({rating.wins}W/{rating.losses}L)")
        except Exception:
            pass

    def _select_debate_team(self, task: str) -> list:
        """Select optimal agent team for the task (P14: AgentSelector + P10: ProbeFilter)."""
        all_agents = [self.gemini, self.codex, self.claude, self.grok, self.deepseek]

        # Filter out agents in circuit breaker cooldown
        default_team = []
        for agent in all_agents:
            if self.circuit_breaker.is_available(agent.name):
                default_team.append(agent)
            else:
                self._log(f"  [circuit-breaker] Skipping {agent.name} (in cooldown)")

        if len(default_team) < 2:
            self._log("  [circuit-breaker] WARNING: Not enough agents available, using all")
            default_team = all_agents  # Fall back to all agents if too few

        # Phase 10: Apply probe-aware filtering
        if self.probe_filter and PROBE_FILTER_AVAILABLE:
            try:
                # Get probe scores for weighted selection
                agent_names = [a.name for a in default_team]
                probe_scores = self.probe_filter.get_team_scores(agent_names)

                # Log probe status for visibility
                probed_agents = [n for n, s in probe_scores.items() if s != 1.0]
                if probed_agents:
                    self._log(f"  [probe-filter] Probe scores: {[(n, f'{s:.0%}') for n, s in sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)]}")

                # Filter out high-risk agents (>50% vulnerability rate)
                safe_names = self.probe_filter.filter_agents(
                    candidates=agent_names,
                    max_vulnerability_rate=0.5,
                    exclude_critical=True
                )

                if len(safe_names) >= 2:
                    filtered_team = [a for a in default_team if a.name in safe_names]
                    if len(filtered_team) < len(default_team):
                        excluded = [a.name for a in default_team if a.name not in safe_names]
                        self._log(f"  [probe-filter] Excluded high-risk agents: {excluded}")
                    default_team = filtered_team

                # Sort by probe score (higher is better)
                default_team.sort(key=lambda a: probe_scores.get(a.name, 1.0), reverse=True)

            except Exception as e:
                self._log(f"  [probe-filter] Error: {e}, using default selection")

        detected_domain = self._detect_domain(task)

        # If ELO available, sort by domain expertise first
        if self.elo_system and ELO_AVAILABLE:
            try:
                # Score agents by domain-specific performance
                domain_scores = []
                for agent in default_team:
                    best_domains = self.elo_system.get_best_domains(agent.name, limit=10)
                    domain_score = 0.0
                    for domain, score in best_domains:
                        if domain == detected_domain:
                            domain_score = score
                            break
                    overall_elo = self.elo_system.get_rating(agent.name).elo
                    # Enhanced: 70% domain expertise + 30% ELO when agent has proven domain knowledge
                    domain_weight = 0.7 if domain_score > 0.5 else 0.6
                    combined = (domain_score * domain_weight) + ((overall_elo - 1400) / 200 * (1 - domain_weight))
                    domain_scores.append((agent, combined))
                domain_scores.sort(key=lambda x: x[1], reverse=True)
                # Use ELO-sorted team as the default
                default_team = [a for a, _ in domain_scores]
                self._log(f"  [routing] Domain '{detected_domain}' ELO ranking: {[(a.name, f'{s:.2f}') for a, s in domain_scores]}")
            except Exception as e:
                self._log(f"  [elo] Domain scoring failed: {e}")

        if not self.agent_selector or not SELECTOR_AVAILABLE:
            return default_team
        try:
            # Register agents with ELO ratings, probe scores, and calibration data
            for agent in default_team:
                # Get probe profile if available
                probe_score = 1.0
                has_critical = False
                if self.probe_filter and PROBE_FILTER_AVAILABLE:
                    try:
                        probe_profile = self.probe_filter.get_agent_profile(agent.name)
                        probe_score = probe_profile.probe_score
                        has_critical = probe_profile.has_critical_issues()
                    except Exception:
                        pass

                # Get calibration data if available
                calibration_score = 1.0
                brier_score = 0.0
                is_overconfident = False
                if self.calibration_tracker and CALIBRATION_AVAILABLE:
                    try:
                        cal_summary = self.calibration_tracker.get_calibration_summary(agent.name)
                        if cal_summary.total_predictions >= 5:
                            calibration_score = max(0.0, 1.0 - cal_summary.ece)
                            brier_score = cal_summary.brier_score
                            is_overconfident = cal_summary.is_overconfident
                    except Exception:
                        pass

                profile = AgentProfile(
                    name=agent.name,
                    agent_type=agent.model if hasattr(agent, 'model') else agent.name,
                    elo_rating=self.elo_system.get_rating(agent.name).elo if self.elo_system else 1500,
                    probe_score=probe_score,
                    has_critical_probes=has_critical,
                    calibration_score=calibration_score,
                    brier_score=brier_score,
                    is_overconfident=is_overconfident
                )
                self.agent_selector.register_agent(profile)

            requirements = TaskRequirements(
                task_id=f"cycle-{self.cycle_count}",
                description=task[:200],
                primary_domain=detected_domain,
                min_agents=3,
                max_agents=4,
                quality_priority=0.7,
                diversity_preference=0.5
            )
            team = self.agent_selector.select_team(requirements)
            self._log(f"  [selector] Selected team: {[a.name for a in team.agents]}")
            # Map back to actual agent objects
            agent_map = {a.name: a for a in default_team}
            return [agent_map[p.name] for p in team.agents if p.name in agent_map]
        except Exception as e:
            self._log(f"  [selector] Error: {e}, using default team")
            return default_team

    def _inject_grounded_personas(self, agents: list) -> None:
        """Inject grounded identity prompts into agent system prompts (Phase 9: PersonaSynthesizer)."""
        if not self.persona_synthesizer or not GROUNDED_PERSONAS_AVAILABLE:
            return

        for agent in agents:
            try:
                # Get opponent names (other agents in the debate)
                opponent_names = [a.name for a in agents if a.name != agent.name]

                # Synthesize grounded identity prompt with full position history
                identity = self.persona_synthesizer.synthesize_identity_prompt(
                    agent_name=agent.name,
                    opponent_names=opponent_names,
                    include_sections=["performance", "calibration", "relationships", "positions"],
                )

                # Add opponent briefings for tactical intelligence
                briefings = []
                for opponent in opponent_names:
                    try:
                        briefing = self.persona_synthesizer.get_opponent_briefing(agent.name, opponent)
                        if briefing:
                            briefings.append(briefing)
                    except Exception:
                        pass  # Skip if briefing generation fails

                if identity or briefings:
                    # Combine identity and briefings
                    full_prompt = identity or ""
                    if briefings:
                        full_prompt += "\n\n## Opponent Intelligence\n" + "\n\n".join(briefings)

                    # Inject agent-specific memories (P3: MemoryStream)
                    try:
                        topic_hint = getattr(self, 'initial_proposal', '') or 'aragora improvement'
                        agent_memories = self._format_agent_memories(agent.name, topic_hint[:200], limit=3)
                        if agent_memories:
                            full_prompt += f"\n\n{agent_memories}"
                    except Exception as e:
                        self._log(f"  [memory] Injection failed for {agent.name}: {e}")

                    # Inject position history for consistency tracking (P9: PositionLedger read)
                    try:
                        topic_hint = getattr(self, 'initial_proposal', '') or 'aragora improvement'
                        position_history = self._format_position_history(agent.name, topic_hint[:200], limit=5)
                        if position_history:
                            full_prompt += f"\n\n{position_history}"
                    except Exception as e:
                        self._log(f"  [position] History injection failed for {agent.name}: {e}")

                    # Inject flip detection warnings (P9: FlipDetector integration)
                    try:
                        if self.flip_detector:
                            consistency = self.flip_detector.get_agent_consistency(agent.name)
                            if consistency.total_flips > 0:
                                flip_warning = f"\n\n## Consistency Warning\n"
                                flip_warning += f"You have changed your position {consistency.total_flips} times.\n"
                                flip_warning += f"- Contradictions: {consistency.contradictions}\n"
                                flip_warning += f"- Retractions: {consistency.retractions}\n"
                                flip_warning += f"Consistency score: {consistency.consistency_score:.0%}\n"
                                flip_warning += f"Be mindful of intellectual consistency. Acknowledge past positions when changing."
                                full_prompt += flip_warning
                    except Exception as e:
                        self._log(f"  [flip] Warning injection failed for {agent.name}: {e}")

                    # Prepend identity to system prompt
                    original_prompt = getattr(agent, 'system_prompt', '') or ''
                    agent.system_prompt = f"{full_prompt}\n\n{original_prompt}"
                    self._log(f"  [personas] Injected grounded identity for {agent.name} with {len(briefings)} opponent briefings")
            except Exception as e:
                self._log(f"  [personas] Error injecting persona for {agent.name}: {e}")
                # Don't break debate on persona injection failure

    def _log_persona_insights(self) -> None:
        """Log grounded persona insights for visibility (Phase 9: PersonaSynthesizer)."""
        if not self.persona_synthesizer or not GROUNDED_PERSONAS_AVAILABLE:
            return

        self._log("  [personas] Agent insights:")
        agents = [self.gemini, self.claude, self.codex, self.grok, self.deepseek]
        for agent in agents:
            try:
                persona = self.persona_synthesizer.get_grounded_persona(agent.name)
                if persona:
                    self._log(f"    {agent.name}: {persona.overall_calibration:.0%} calibration, "
                             f"{persona.position_accuracy:.0%} accuracy, "
                             f"{len(persona.rivals)} rivals")
            except Exception:
                pass

    def _log_grounded_persona_stats(self) -> None:
        """Log grounded persona data completeness for observability."""
        self._log("  [grounded] Data completeness:")

        # Position Ledger stats
        if self.position_ledger:
            try:
                stats = self.position_ledger.get_all_stats()
                total = stats.get('total', 0)
                agents = len(stats.get('by_agent', {}))
                self._log(f"    PositionLedger: {total} positions from {agents} agents")
            except Exception:
                self._log("    PositionLedger: unavailable")
        else:
            self._log("    PositionLedger: not initialized")

        # Relationship Tracker stats
        if self.relationship_tracker:
            try:
                count = self.relationship_tracker.get_relationship_count()
                self._log(f"    RelationshipTracker: {count} agent pairs tracked")
            except Exception:
                self._log("    RelationshipTracker: unavailable")
        else:
            self._log("    RelationshipTracker: not initialized")

        # Moment Detector stats
        if self.moment_detector:
            try:
                count = sum(len(m) for m in self.moment_detector._moment_cache.values())
                self._log(f"    MomentDetector: {count} significant moments recorded")
            except Exception:
                self._log("    MomentDetector: unavailable")
        else:
            self._log("    MomentDetector: not initialized")

        # Probe Filter stats
        if self.probe_filter:
            try:
                profiles = self.probe_filter.get_all_profiles()
                if profiles:
                    probed_count = len(profiles)
                    avg_score = sum(p.probe_score for p in profiles.values()) / probed_count
                    high_risk = sum(1 for p in profiles.values() if p.is_high_risk())
                    self._log(f"    ProbeFilter: {probed_count} agents probed, "
                             f"{avg_score:.0%} avg score, {high_risk} high-risk")
                else:
                    self._log("    ProbeFilter: no probe data yet")
            except Exception:
                self._log("    ProbeFilter: unavailable")
        else:
            self._log("    ProbeFilter: not initialized")

        # ELO domain calibration stats
        if self.elo_system:
            try:
                agents = [self.gemini, self.claude, self.codex, self.grok, self.deepseek]
                for agent in agents:
                    cal = self.elo_system.get_domain_calibration(agent.name)
                    if cal and cal.get('total', 0) > 0:
                        self._log(f"    {agent.name} calibration: {cal['total']} predictions, "
                                 f"{cal['accuracy']:.0%} accuracy")
            except Exception:
                pass

    def _track_debate_risks(self, result, task: str) -> None:
        """Track risks from debates with low consensus or confidence (P15: RiskRegister)."""
        if not RISK_REGISTER_AVAILABLE:
            return
        # Only track if consensus is weak
        if result.consensus_reached and result.confidence >= 0.7:
            return
        try:
            import json
            risk_level = "high" if not result.consensus_reached else "medium"
            risk_entry = {
                "cycle": self.cycle_count,
                "task": task,
                "confidence": result.confidence,
                "consensus": result.consensus_reached,
                "level": risk_level
            }
            risk_file = self.nomic_dir / "risk_register.jsonl"
            with open(risk_file, "a") as f:
                f.write(json.dumps(risk_entry) + "\n")
            self._log(f"  [risk] Tracked {risk_level} risk: low consensus on task")
        except Exception as e:
            self._log(f"  [risk] Error: {e}")

    # =========================================================================
    # Phase 6: Verifiable Reasoning & Robustness Testing Helper Methods
    # =========================================================================

    def _extract_claims_from_debate(self, result) -> None:
        """Extract typed claims from debate result and populate kernel (P16: ClaimsKernel)."""
        if not self.claims_kernel or not CLAIMS_KERNEL_AVAILABLE:
            return
        try:
            # Reset kernel for new debate
            self.claims_kernel = ClaimsKernel(debate_id=f"nomic-cycle-{self.cycle_count}")

            # Extract claims from messages
            for msg in result.messages:
                agent = getattr(msg, 'agent', None) or (msg.get('agent') if isinstance(msg, dict) else None)
                content = getattr(msg, 'content', None) or (msg.get('content', '') if isinstance(msg, dict) else '')
                role = getattr(msg, 'role', None) or (msg.get('role', 'proposer') if isinstance(msg, dict) else 'proposer')

                if not agent or not content:
                    continue

                claim_type = ClaimType.PROPOSAL if role == 'proposer' else ClaimType.OBJECTION
                self.claims_kernel.add_claim(
                    statement=content[:500],
                    author=agent,
                    claim_type=claim_type,
                    confidence=result.confidence if result.consensus_reached else 0.5
                )
            self._log(f"  [claims] Extracted {len(self.claims_kernel.claims)} claims")
        except Exception as e:
            self._log(f"  [claims] Error: {e}")

    def _analyze_claim_structure(self) -> dict:
        """Analyze the claim structure for insights (P16: ClaimsKernel)."""
        if not self.claims_kernel or not CLAIMS_KERNEL_AVAILABLE:
            return {}
        try:
            unsupported = self.claims_kernel.find_unsupported_claims()
            contradictions = self.claims_kernel.find_contradictions()
            strongest = self.claims_kernel.get_strongest_claims(3)
            coverage = self.claims_kernel.get_evidence_coverage()

            self._log(f"  [claims] Unsupported: {len(unsupported)}, "
                     f"Contradictions: {len(contradictions)}, "
                     f"Coverage: {coverage['coverage_ratio']:.0%}")

            return {
                "unsupported_count": len(unsupported),
                "contradiction_count": len(contradictions),
                "strongest_claims": [(c.statement, s) for c, s in strongest],
                "evidence_coverage": coverage
            }
        except Exception as e:
            self._log(f"  [claims] Analysis error: {e}")
            return {}

    def _record_evidence_provenance(self, content: str, source_type: str, source_id: str) -> str:
        """Record evidence with provenance tracking (P17: ProvenanceManager)."""
        if not self.provenance_manager or not PROVENANCE_AVAILABLE:
            return ""
        try:
            source = SourceType.AGENT_GENERATED if source_type == "agent" else SourceType.CODE_ANALYSIS
            # Store full content, no truncation
            record = self.provenance_manager.record_evidence(
                content=content,
                source_type=source,
                source_id=source_id
            )
            return record.id
        except Exception as e:
            self._log(f"  [provenance] Error: {e}")
            return ""

    def _link_claims_to_evidence(self, claims: list[dict], debate_id: str) -> list[str]:
        """Link extracted claims to evidence with provenance tracking.

        Creates a provenance chain: Claim → Evidence → Source
        Returns list of evidence IDs for linking to subsequent phases.
        """
        if not self.provenance_manager or not PROVENANCE_AVAILABLE:
            return []

        evidence_ids = []
        try:
            for claim in claims[:10]:  # Limit to 10 claims per debate
                claim_text = claim.get("claim", "")
                priority = claim.get("priority", "medium")

                # Record the claim as evidence
                source_type = SourceType.AGENT_GENERATED
                record = self.provenance_manager.record_evidence(
                    content=claim_text,
                    source_type=source_type,
                    source_id=f"{debate_id}-claim",
                    metadata={"priority": priority, "debate_id": debate_id},
                )
                evidence_ids.append(record.id)

                # If we have citations for this claim, link them
                if self.citation_store and CITATION_GROUNDING_AVAILABLE:
                    existing_citations = self.citation_store.find_for_claim(claim_text, limit=3)
                    for citation in existing_citations:
                        self.provenance_manager.cite_evidence(
                            claim_id=record.id,
                            evidence_id=citation.id,
                            relevance=citation.relevance_score,
                            support_type="supports",
                            citation_text=citation.excerpt[:200] if citation.excerpt else "",
                        )

            if evidence_ids:
                self._log(f"  [provenance] Linked {len(evidence_ids)} claims to evidence chain")

        except Exception as e:
            self._log(f"  [provenance] Claim linking error: {e}")

        return evidence_ids

    def _build_phase_provenance(self, phase: str, content: str, parent_ids: list[str] = None) -> str:
        """Build provenance chain from phase to phase.

        Tracks: Source → Claim → Design → Implementation
        Returns new evidence ID for chaining.
        """
        if not self.provenance_manager or not PROVENANCE_AVAILABLE:
            return ""

        try:
            if parent_ids:
                # Create synthesized evidence from multiple parent sources
                record = self.provenance_manager.synthesize_evidence(
                    parent_ids=parent_ids,
                    synthesized_content=content[:5000],  # Limit size
                    synthesizer_id=f"nomic-{phase}-{self.cycle_count}",
                )
            else:
                # Record as new evidence
                record = self.provenance_manager.record_evidence(
                    content=content[:5000],
                    source_type=SourceType.AGENT_GENERATED,
                    source_id=f"nomic-{phase}-{self.cycle_count}",
                )

            self._log(f"  [provenance] {phase} phase recorded: {record.id[:8]}...")
            return record.id

        except Exception as e:
            self._log(f"  [provenance] Phase recording error: {e}")
            return ""

    def _check_evidence_staleness(self, evidence_ids: list[str]) -> list[dict]:
        """Check if evidence is stale before implementation (P7: Staleness Detection).

        Returns list of stale evidence with details.
        """
        if not ENHANCED_PROVENANCE_AVAILABLE or not self.provenance_manager:
            return []

        stale_items = []
        try:
            if hasattr(self.provenance_manager, 'check_staleness'):
                for evidence_id in evidence_ids[:20]:  # Limit checks
                    staleness = self.provenance_manager.check_staleness(evidence_id)
                    if staleness and staleness.get("is_stale", False):
                        stale_items.append({
                            "evidence_id": evidence_id,
                            "reason": staleness.get("reason", "Unknown"),
                            "age_hours": staleness.get("age_hours", 0),
                        })

            if stale_items:
                self._log(f"  [provenance] WARNING: {len(stale_items)} stale evidence items detected")
                for item in stale_items[:3]:
                    self._log(f"    - {item['evidence_id'][:8]}: {item['reason']}")

        except Exception as e:
            self._log(f"  [provenance] Staleness check error: {e}")

        return stale_items

    def _verify_evidence_chain(self) -> tuple:
        """Verify integrity of evidence chain (P17: ProvenanceManager)."""
        if not self.provenance_manager or not PROVENANCE_AVAILABLE:
            return True, []
        try:
            valid, errors = self.provenance_manager.verify_chain_integrity()
            if not valid:
                self._log(f"  [provenance] Chain integrity issues: {len(errors)}")
            return valid, errors
        except Exception as e:
            self._log(f"  [provenance] Verification error: {e}")
            return False, [str(e)]

    def _build_belief_network(self) -> None:
        """Build belief network from claims kernel (P18: BeliefNetwork)."""
        if not self.belief_network or not BELIEF_NETWORK_AVAILABLE:
            return
        if not self.claims_kernel or not CLAIMS_KERNEL_AVAILABLE:
            return
        try:
            self.belief_network = BeliefNetwork(debate_id=f"nomic-cycle-{self.cycle_count}")
            self.belief_network.from_claims_kernel(self.claims_kernel)
            result = self.belief_network.propagate()
            self._log(f"  [belief] Network built: {len(self.belief_network.nodes)} nodes, "
                     f"converged={result.converged} after {result.iterations} iterations")
        except Exception as e:
            self._log(f"  [belief] Error: {e}")

    def _identify_debate_cruxes(self) -> list:
        """Identify key claims that would most impact debate outcome (P18: BeliefNetwork)."""
        if not self.belief_network or not BELIEF_NETWORK_AVAILABLE:
            return []
        try:
            analyzer = BeliefPropagationAnalyzer(self.belief_network)
            cruxes = analyzer.identify_debate_cruxes(top_k=3)
            if cruxes:
                self._log(f"  [belief] Top crux: {cruxes[0]['statement']}")
                # P3-Phase2: Cache cruxes for injection into next debate
                self._cached_cruxes = cruxes
            return cruxes
        except Exception as e:
            self._log(f"  [belief] Crux analysis error: {e}")
            return []

    def _format_crux_context(self) -> str:
        """Format cached cruxes for injection into debate context (P3-Phase2: Crux-Fixing)."""
        if not self._cached_cruxes:
            return ""
        try:
            lines = ["=== PIVOTAL CLAIMS FROM PREVIOUS DEBATE ==="]
            lines.append("Focus on these high-impact questions that could swing the outcome:\n")
            for i, crux in enumerate(self._cached_cruxes[:3], 1):
                statement = crux.get('statement', crux.get('claim', 'Unknown'))
                impact = crux.get('impact_score', crux.get('sensitivity', 0.0))
                lines.append(f"{i}. {statement}")
                if impact:
                    lines.append(f"   (Impact: {impact:.0%})")
            lines.append("\nAddressing these cruxes directly will accelerate consensus.")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_consensus_probability(self) -> dict:
        """Estimate probability of consensus based on belief network (P18: BeliefNetwork)."""
        if not self.belief_network or not BELIEF_NETWORK_AVAILABLE:
            return {"probability": 0.5}
        try:
            analyzer = BeliefPropagationAnalyzer(self.belief_network)
            return analyzer.compute_consensus_probability()
        except Exception:
            return {"probability": 0.5}

    async def _create_verification_proofs(self, result) -> int:
        """Create verification proofs for testable claims in debate result (P19: ProofExecutor)."""
        if not self.claim_verifier or not PROOF_EXECUTOR_AVAILABLE:
            return 0
        try:
            proof_count = 0
            # Look for code-related claims that can be verified
            final_answer = result.final_answer or ""
            if "```" in final_answer:
                # Extract code block
                code_start = final_answer.find("```")
                code_end = final_answer.find("```", code_start + 3)
                if code_end > code_start:
                    code_block = final_answer[code_start+3:code_end].strip()
                    # Skip language identifier if present
                    if "\n" in code_block:
                        first_line = code_block.split("\n")[0]
                        if first_line.strip().isalpha():
                            code_block = "\n".join(code_block.split("\n")[1:])

                    builder = ProofBuilder(claim_id=f"cycle-{self.cycle_count}-final", created_by="nomic")
                    # Create syntax verification proof
                    proof = builder.assertion(
                        description="Verify proposed code is syntactically valid Python",
                        code=f"import ast\ncode = '''{code_block[:300]}'''\nast.parse(code)",
                        assertion="True"
                    )
                    self.claim_verifier.add_proof(proof)
                    proof_count += 1
            return proof_count
        except Exception as e:
            self._log(f"  [proofs] Proof creation error: {e}")
            return 0

    async def _run_verification_proofs(self):
        """Execute all pending verification proofs (P19: ProofExecutor)."""
        if not self.claim_verifier or not PROOF_EXECUTOR_AVAILABLE:
            return None
        try:
            results = await self.claim_verifier.verify_all()
            if not results:
                return None
            passed = sum(1 for r in results if r.passed)
            self._log(f"  [proofs] Verified {passed}/{len(results)} proofs passed")

            # Build report
            report = VerificationReport(debate_id=f"cycle-{self.cycle_count}")
            report.total_proofs = len(results)
            report.proofs_passed = passed
            report.proofs_failed = len(results) - passed
            return report
        except Exception as e:
            self._log(f"  [proofs] Verification error: {e}")
            return None

    async def _run_robustness_check(self, task: str, base_context: str = "") -> dict:
        """Run quick robustness check across key scenarios (P20: ScenarioMatrix).

        Now runs every cycle (was every 5) to catch edge cases before implementation.
        """
        if not SCENARIO_MATRIX_AVAILABLE:
            return {}
        try:
            self._log(f"  [scenarios] Running robustness check...")
            matrix = ScenarioMatrix.from_presets("risk")

            # Create lightweight debate function
            async def quick_debate(task_text, context):
                env = Environment(task=task_text, context=context)
                protocol = DebateProtocol(
                    rounds=1,
                    consensus="majority",
                    role_rotation=True,
                    role_rotation_config=RoleRotationConfig(
                        enabled=True,
                        roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC],
                    ),
                )
                agents = [self.gemini, self.claude] if hasattr(self, 'claude') else [self.gemini]
                arena = Arena(
                    env, agents, protocol,
                    position_tracker=self.position_tracker,
                    calibration_tracker=self.calibration_tracker,
                    event_hooks=self._create_arena_hooks("scenario"),
                    event_emitter=self.stream_emitter,
                    loop_id=self.loop_id,
                    persona_manager=self.persona_manager,
                    relationship_tracker=self.relationship_tracker,
                    moment_detector=self.moment_detector,
                    continuum_memory=self.continuum,
                )
                return await arena.run()  # run() takes no arguments

            runner = MatrixDebateRunner(quick_debate, max_parallel=2)
            result = await runner.run_matrix(task, matrix, base_context)

            self._log(f"  [scenarios] Outcome: {result.outcome_category.value}")
            if result.universal_conclusions:
                self._log(f"  [scenarios] Universal: {len(result.universal_conclusions)} conclusions")

            return {
                "outcome": result.outcome_category.value,
                "scenarios_run": len(result.results),
                "universal_conclusions": result.universal_conclusions[:3]
            }
        except Exception as e:
            self._log(f"  [scenarios] Error: {e}")
            return {}

    # =========================================================================
    # Phase 7: Resilience, Living Documents, & Observability Helper Methods
    # =========================================================================

    def _record_code_evidence(
        self, file_path: str, line_start: int, line_end: int,
        content: str, claim_id: str = None
    ) -> str:
        """Record code evidence with git tracking for staleness detection (P21: EnhancedProvenance)."""
        if not ENHANCED_PROVENANCE_AVAILABLE or not self.provenance_manager:
            return ""
        try:
            # Enhanced provenance tracks git state for living document detection
            evidence_id = self.provenance_manager.record_code_evidence(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                content=content,
                claim_id=claim_id
            )
            self._log(f"  [provenance] Recorded code evidence: {file_path}:{line_start}-{line_end}")
            return evidence_id
        except Exception as e:
            self._log(f"  [provenance] Code evidence error: {e}")
            return ""

    async def _check_evidence_staleness(self) -> dict:
        """Check all evidence for staleness - are claims still valid? (P21: EnhancedProvenance)."""
        if not ENHANCED_PROVENANCE_AVAILABLE or not self.provenance_manager:
            return {}
        try:
            staleness_results = self.provenance_manager.check_all_staleness()
            fresh_count = sum(1 for s in staleness_results if s.status == StalenessStatus.FRESH)
            stale_count = sum(1 for s in staleness_results if s.status == StalenessStatus.STALE)

            if stale_count > 0:
                self._log(f"  [provenance] Staleness: {stale_count} stale, {fresh_count} fresh")
                # Generate revalidation triggers for stale evidence
                triggers = self.provenance_manager.generate_revalidation_triggers()
                if triggers:
                    self._log(f"  [provenance] Revalidation needed for {len(triggers)} items")

            return {
                "fresh": fresh_count,
                "stale": stale_count,
                "total": len(staleness_results),
                "living_status": self.provenance_manager.get_living_document_status()
            }
        except Exception as e:
            self._log(f"  [provenance] Staleness check error: {e}")
            return {}

    async def _create_debate_checkpoint(
        self, debate_id: str, task: str, round_num: int,
        messages: list, agents: list, consensus: dict = None
    ) -> str:
        """Create a checkpoint for crash recovery (P22: CheckpointManager)."""
        if not CHECKPOINT_AVAILABLE or not self.checkpoint_manager:
            return ""
        try:
            # Call create_checkpoint with individual parameters (it's async and creates the checkpoint internally)
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                debate_id=debate_id,
                task=task,
                current_round=round_num,
                total_rounds=5,  # Default max rounds
                phase="debate",
                messages=messages,  # Pass Message objects directly
                critiques=[],
                votes=[],
                agents=agents,
                current_consensus=consensus.get("consensus") if consensus else None,
            )
            self._log(f"  [checkpoint] Created: {checkpoint.checkpoint_id[:8]}")
            return checkpoint.checkpoint_id
        except Exception as e:
            self._log(f"  [checkpoint] Create error: {e}")
            return ""

    async def _resume_from_checkpoint(self, checkpoint_id: str) -> dict:
        """Resume a debate from checkpoint (P22: CheckpointManager)."""
        if not CHECKPOINT_AVAILABLE or not self.checkpoint_manager:
            return {}
        try:
            resumed = self.checkpoint_manager.resume_from_checkpoint(checkpoint_id)
            if resumed:
                self._log(f"  [checkpoint] Resumed: {resumed.original_debate_id} at round {resumed.checkpoint.current_round}")
                return {
                    "debate_id": resumed.original_debate_id,
                    "task": resumed.checkpoint.task,
                    "round": resumed.checkpoint.current_round,
                    "messages": resumed.messages,  # Already deserialized Message objects
                    "consensus": resumed.checkpoint.current_consensus
                }
            return {}
        except Exception as e:
            self._log(f"  [checkpoint] Resume error: {e}")
            return {}

    def _check_debate_breakpoints(
        self, debate_id: str, task: str, messages: list,
        confidence: float, round_num: int, critiques: list = None
    ) -> "Breakpoint":
        """Check if debate triggers breakpoint for human review (P23: BreakpointManager)."""
        if not BREAKPOINT_AVAILABLE or not self.breakpoint_manager:
            return None
        try:
            # Build debate state for breakpoint checking
            debate_state = {
                "debate_id": debate_id,
                "task": task,
                "messages": messages,
                "confidence": confidence,
                "round": round_num,
                "critiques": critiques or []
            }
            breakpoint = self.breakpoint_manager.check_triggers(debate_state)
            if breakpoint:
                self._log(f"  [breakpoint] Triggered: {breakpoint.trigger.value}")
            return breakpoint
        except Exception as e:
            self._log(f"  [breakpoint] Check error: {e}")
            return None

    async def _handle_breakpoint(self, breakpoint: "Breakpoint") -> "HumanGuidance":
        """Handle breakpoint by getting human guidance (P23: BreakpointManager)."""
        if not BREAKPOINT_AVAILABLE or not self.breakpoint_manager or not breakpoint:
            return None
        try:
            guidance = await self.breakpoint_manager.handle_breakpoint(breakpoint)
            if guidance:
                self._log(f"  [breakpoint] Human guidance: {guidance.action}")
            return guidance
        except Exception as e:
            self._log(f"  [breakpoint] Handle error: {e}")
            return None

    def _score_claim_reliability(self, claim_id: str, claim_text: str) -> dict:
        """Score reliability of a claim (P24: ReliabilityScorer)."""
        if not RELIABILITY_SCORER_AVAILABLE or not self.reliability_scorer:
            return {}
        try:
            # Get claim from claims kernel if available
            claim = None
            if CLAIMS_KERNEL_AVAILABLE and self.claims_kernel:
                claims = self.claims_kernel.get_claims()
                for c in claims:
                    if str(c.id) == claim_id:
                        claim = c
                        break

            if claim:
                reliability = self.reliability_scorer.score_claim(claim)
                return {
                    "claim_id": claim_id,
                    "level": reliability.level.value if hasattr(reliability.level, 'value') else str(reliability.level),
                    "score": reliability.score,
                    "factors": reliability.factors
                }
            return {}
        except Exception as e:
            self._log(f"  [reliability] Score error: {e}")
            return {}

    def _generate_reliability_report(self) -> dict:
        """Generate reliability report for all claims (P24: ReliabilityScorer)."""
        if not RELIABILITY_SCORER_AVAILABLE or not self.reliability_scorer:
            return {}
        try:
            if not CLAIMS_KERNEL_AVAILABLE or not self.claims_kernel:
                return {}

            claims = self.claims_kernel.get_claims()
            if not claims:
                return {}

            # Convert list[TypedClaim] to dict[str, str] for reliability scorer
            claims_dict = {c.claim_id: c.statement for c in claims}
            report = self.reliability_scorer.generate_reliability_report(claims_dict)
            # report["claims"] is a dict of {claim_id: result_dict}, iterate .values()
            claims_results = report.get("claims", {})
            claims_list = list(claims_results.values()) if isinstance(claims_results, dict) else claims_results
            high_reliability = sum(1 for c in claims_list
                                   if c.get("level") in ("VERY_HIGH", "HIGH"))
            low_reliability = sum(1 for c in claims_list
                                  if c.get("level") in ("VERY_LOW", "SPECULATIVE"))

            self._log(f"  [reliability] Report: {high_reliability} high, {low_reliability} low reliability")
            return report
        except Exception as e:
            self._log(f"  [reliability] Report error: {e}")
            return {}

    def _start_debate_trace(self, debate_id: str, task: str, agents: list) -> None:
        """Start tracing a debate for audit logs (P25: DebateTracer)."""
        if not DEBATE_TRACER_AVAILABLE or not self.debate_trace_db:
            return
        try:
            agent_names = [a.name for a in agents if hasattr(a, 'name')]
            # Create a new tracer for this debate
            self._current_tracer = DebateTracer(
                debate_id=debate_id,
                task=task,
                agents=agent_names,
                db_path=self.debate_trace_db
            )
            self._log(f"  [tracer] Started trace for debate {debate_id[:8]}")
        except Exception as e:
            self._current_tracer = None
            self._log(f"  [tracer] Start error: {e}")

    def _trace_event(self, event_type: str, content: str, agent: str = None) -> None:
        """Record an event to the debate trace (P25: DebateTracer)."""
        if not DEBATE_TRACER_AVAILABLE or not getattr(self, '_current_tracer', None):
            return
        try:
            # Use specialized record methods where available
            if event_type == "proposal" and agent:
                self._current_tracer.record_proposal(agent, content)
            elif event_type == "round_start":
                round_num = int(content) if content.isdigit() else 0
                self._current_tracer.start_round(round_num)
            elif event_type == "round_end":
                self._current_tracer.end_round()
            else:
                # Fallback to generic record
                type_map = {
                    "critique": EventType.AGENT_CRITIQUE if EventType else None,
                    "vote": EventType.AGENT_VOTE if EventType else None,
                    "consensus": EventType.CONSENSUS_REACHED if EventType else None,
                }
                event_enum = type_map.get(event_type)
                if event_enum:
                    self._current_tracer.record(event_enum, {"content": content}, agent=agent)
        except Exception as e:
            self._log(f"  [tracer] Event error: {e}")

    def _finalize_debate_trace(self, result: "DebateResult") -> str:
        """Finalize and save the debate trace (P25: DebateTracer)."""
        if not DEBATE_TRACER_AVAILABLE or not getattr(self, '_current_tracer', None):
            return ""
        try:
            # Build result dict for finalize
            result_dict = {
                "final_answer": getattr(result, 'final_answer', ""),
                "consensus_reached": getattr(result, 'consensus_reached', False),
                "confidence": getattr(result, 'confidence', 0.0),
            }
            trace = self._current_tracer.finalize(result_dict)
            trace_id = trace.trace_id if trace else ""
            self._log(f"  [tracer] Finalized trace: {trace_id}")
            self._current_tracer = None  # Clear for next debate
            return trace_id
        except Exception as e:
            self._log(f"  [tracer] Finalize error: {e}")
            return ""

    # =========================================================================
    # Phase 8: Agent Evolution, Semantic Memory & Advanced Debates Helper Methods
    # =========================================================================

    def _run_persona_experiment(self, agent_name: str, variant_traits: list) -> str:
        """Create a persona A/B experiment (P26: PersonaLaboratory)."""
        if not PERSONA_LAB_AVAILABLE or not self.persona_lab:
            return ""
        try:
            experiment = self.persona_lab.create_experiment(
                agent_name=agent_name,
                variant_traits=variant_traits,
                hypothesis=f"Testing traits: {', '.join(variant_traits)}"
            )
            self._log(f"  [lab] Created experiment {experiment.experiment_id[:8]} for {agent_name}")
            return experiment.experiment_id
        except Exception as e:
            self._log(f"  [lab] Experiment creation error: {e}")
            return ""

    def _record_experiment_trial(self, experiment_id: str, is_control: bool, success: bool) -> None:
        """Record a trial result for an experiment (P26: PersonaLaboratory)."""
        if not PERSONA_LAB_AVAILABLE or not self.persona_lab or not experiment_id:
            return
        try:
            # Note: PersonaLaboratory uses is_variant (inverse of is_control)
            self.persona_lab.record_experiment_result(
                experiment_id=experiment_id,
                is_variant=not is_control,  # Invert: is_control -> is_variant
                success=success
            )
        except Exception as e:
            self._log(f"  [lab] Trial recording error: {e}")

    def _detect_emergent_traits(self) -> list:
        """Detect emergent traits from performance patterns (P26: PersonaLaboratory)."""
        if not PERSONA_LAB_AVAILABLE or not self.persona_lab:
            return []
        try:
            traits = self.persona_lab.detect_emergent_traits()
            if traits:
                self._log(f"  [lab] Detected {len(traits)} emergent traits")
                for t in traits[:3]:
                    self._log(f"    - {t.trait_name} (confidence: {t.confidence:.2f})")
            return traits
        except Exception as e:
            self._log(f"  [lab] Trait detection error: {e}")
            return []

    def _cross_pollinate_traits(self, from_agent: str, to_agent: str, trait: str) -> bool:
        """Cross-pollinate a successful trait between agents (P26: PersonaLaboratory)."""
        if not PERSONA_LAB_AVAILABLE or not self.persona_lab:
            return False
        try:
            # cross_pollinate returns TraitTransfer or None
            transfer = self.persona_lab.cross_pollinate(
                from_agent=from_agent,
                to_agent=to_agent,
                trait=trait
            )
            if transfer:
                self._log(f"  [lab] Cross-pollinated '{trait}' from {from_agent} to {to_agent}")
                return True
            return False
        except Exception as e:
            self._log(f"  [lab] Cross-pollination error: {e}")
            return False

    async def _evolve_personas_post_cycle(self) -> dict:
        """Evolve personas based on cycle performance (P26: PersonaLaboratory)."""
        if not PERSONA_LAB_AVAILABLE or not self.persona_lab:
            return {}
        try:
            # Detect emergent traits
            emergent = self._detect_emergent_traits()

            # Proactively create experiments for low-performing agents (every 10 cycles)
            experiments_created = 0
            if self.cycle_count % 10 == 0 and self.elo_system:
                for agent_name in ["gemini", "claude", "codex", "grok"]:
                    try:
                        rating = self.elo_system.get_rating(agent_name)
                        if rating and rating.elo < 1450:
                            candidate_traits = ["analytical", "concise", "thorough", "skeptical"]
                            current = self.persona_lab.get_persona(agent_name) if hasattr(self.persona_lab, 'get_persona') else None
                            current_traits = getattr(current, 'traits', []) if current else []
                            new_traits = [t for t in candidate_traits if t not in current_traits][:2]
                            if new_traits:
                                exp_id = self._run_persona_experiment(agent_name, new_traits)
                                if exp_id:
                                    experiments_created += 1
                    except Exception:
                        pass
                if experiments_created > 0:
                    self._log(f"  [lab] Created {experiments_created} experiments for underperformers")

            # Check experiments for significant results and apply mutations
            experiments = self.persona_lab.get_running_experiments()
            completed = 0
            applied = 0
            for exp in experiments:
                if exp.is_significant:
                    self._log(f"  [lab] Experiment {exp.experiment_id[:8]} significant: {exp.relative_improvement:+.1%}")
                    concluded = self.persona_lab.conclude_experiment(exp.experiment_id)
                    if concluded:
                        completed += 1
                        if concluded.variant_rate > concluded.control_rate:
                            self._log(f"  [lab] Applied variant traits to {exp.agent_name}: {concluded.variant_persona.traits}")
                            applied += 1

            # Cross-pollinate successful traits between agents (every 20 cycles)
            traits_shared = 0
            if self.cycle_count % 20 == 0 and self.elo_system:
                try:
                    ratings = [(a, self.elo_system.get_rating(a)) for a in ["gemini", "claude", "codex", "grok"]]
                    ratings = [(a, r.elo) for a, r in ratings if r]
                    if len(ratings) >= 2:
                        ratings.sort(key=lambda x: x[1], reverse=True)
                        best_agent, best_elo = ratings[0]
                        worst_agent, worst_elo = ratings[-1]
                        if best_elo - worst_elo > 100:
                            best_persona = self.persona_lab.get_persona(best_agent) if hasattr(self.persona_lab, 'get_persona') else None
                            if best_persona and getattr(best_persona, 'traits', []):
                                trait_to_share = best_persona.traits[0]
                                if self._cross_pollinate_traits(best_agent, worst_agent, trait_to_share):
                                    traits_shared += 1
                                    self._log(f"  [lab] Shared '{trait_to_share}' from {best_agent} to {worst_agent}")
                except Exception as e:
                    self._log(f"  [lab] Cross-pollination error: {e}")

            return {
                "emergent_traits": len(emergent),
                "experiments_created": experiments_created,
                "experiments_checked": len(experiments),
                "significant_results": completed,
                "mutations_applied": applied,
                "traits_shared": traits_shared
            }
        except Exception as e:
            self._log(f"  [lab] Evolution error: {e}")
            return {}

    async def _store_critique_embedding(self, critique_id: str, critique_text: str) -> None:
        """Store a critique embedding for future retrieval (P27: SemanticRetriever)."""
        if not SEMANTIC_RETRIEVER_AVAILABLE or not self.semantic_retriever:
            return
        try:
            await self.semantic_retriever.embed_and_store(critique_id, critique_text[:1000])
        except Exception as e:
            self._log(f"  [semantic] Store error: {e}")

    async def _find_similar_critiques(self, query: str, limit: int = 3) -> list:
        """Find similar past critiques (P27: SemanticRetriever)."""
        if not SEMANTIC_RETRIEVER_AVAILABLE or not self.semantic_retriever:
            return []
        try:
            results = await self.semantic_retriever.find_similar(query, limit=limit)
            if results:
                self._log(f"  [semantic] Found {len(results)} similar critiques")
            return results
        except Exception as e:
            self._log(f"  [semantic] Search error: {e}")
            return []

    async def _inject_similar_context(self, task: str) -> str:
        """Search and format similar past critiques as context (P27: SemanticRetriever)."""
        if not SEMANTIC_RETRIEVER_AVAILABLE or not self.semantic_retriever:
            return ""
        try:
            similar = await self._find_similar_critiques(task, limit=3)
            if not similar:
                return ""

            context_parts = ["=== SIMILAR PAST CRITIQUES ==="]
            for id_, text, sim in similar:
                context_parts.append(f"[Similarity: {sim:.2f}] {text}")

            return "\n".join(context_parts)
        except Exception as e:
            self._log(f"  [semantic] Context injection error: {e}")
            return ""

    async def _verify_claim_formally(self, claim_text: str, claim_type: str = "logical") -> dict:
        """Attempt formal verification of a claim (P28: FormalVerificationManager)."""
        if not FORMAL_VERIFICATION_AVAILABLE or not self.formal_verifier:
            return {}
        try:
            result = await self.formal_verifier.verify_claim(claim_text, claim_type)
            if result and result.is_verified:
                self._log(f"  [formal] Claim verified: {claim_text}")
            return result.to_dict() if result else {}
        except Exception as e:
            self._log(f"  [formal] Verification error: {e}")
            return {}

    def _is_formally_verifiable(self, claim_text: str) -> bool:
        """Check if a claim is suitable for formal verification (P28: FormalVerificationManager)."""
        # Simple heuristic: look for mathematical/logical keywords
        keywords = ["for all", "exists", "implies", "if and only if", "<=", ">=",
                    "equals", "greater than", "less than", "always", "never"]
        claim_lower = claim_text.lower()
        return any(kw in claim_lower for kw in keywords)

    def _record_formal_proof(self, claim_id: str, proof_result: dict) -> None:
        """Record a formal proof result (P28: FormalVerificationManager)."""
        if not proof_result:
            return
        try:
            # Store in provenance if available
            if self.provenance_manager and proof_result.get("is_verified"):
                self._record_evidence_provenance(
                    f"Formally verified: {proof_result.get('formal_statement', '')}",
                    source_type="formal_proof",
                    source_id=claim_id
                )
        except Exception as e:
            self._log(f"  [formal] Proof recording error: {e}")

    def _create_debate_graph(self, debate_id: str, task: str) -> "DebateGraph":
        """Create a new debate graph (P29: DebateGraph)."""
        if not DEBATE_GRAPH_AVAILABLE or not DebateGraph:
            return None
        try:
            graph = DebateGraph(debate_id=debate_id, task=task)
            self._log(f"  [graph] Created debate graph {debate_id[:8]}")
            return graph
        except Exception as e:
            self._log(f"  [graph] Creation error: {e}")
            return None

    def _add_graph_node(
        self, graph: "DebateGraph", node_type: str, agent: str, content: str
    ) -> str:
        """Add a node to the debate graph (P29: DebateGraph)."""
        if not graph or not DEBATE_GRAPH_AVAILABLE:
            return ""
        try:
            node_type_enum = NodeType[node_type.upper()] if NodeType else None
            if not node_type_enum:
                return ""
            node = DebateNode(
                id=f"{agent}-{len(graph.nodes)}",
                node_type=node_type_enum,
                agent_id=agent,
                content=content
            )
            graph.add_node(node)
            return node.id
        except Exception as e:
            self._log(f"  [graph] Add node error: {e}")
            return ""

    def _should_branch_graph(self, graph: "DebateGraph", disagreement_score: float) -> bool:
        """Check if graph should branch based on disagreement (P29: DebateGraph)."""
        if not graph or disagreement_score < 0.7:
            return False
        return True

    async def _run_graph_debate(self, task: str, agents: list) -> "DebateResult":
        """Run a graph-based debate (P29: DebateGraph)."""
        if not DEBATE_GRAPH_AVAILABLE or not self.graph_debate_enabled:
            return None
        try:
            self._log(f"  [graph] Running graph-based debate...")
            # Create orchestrator on demand with the specific agents
            orchestrator = GraphDebateOrchestrator(agents=agents)
            result = await orchestrator.run_debate(task)
            # Verify result has required DebateResult interface (consensus_reached, confidence)
            # GraphDebateOrchestrator is a placeholder - returns DebateGraph not DebateResult
            if not hasattr(result, 'consensus_reached') or not hasattr(result, 'confidence'):
                self._log(f"  [graph] Incomplete result - falling back to arena")
                return None
            return result
        except Exception as e:
            self._log(f"  [graph] Debate error: {e}")
            return None

    def _check_should_fork(self, messages: list, round_num: int, agents: list) -> "ForkDecision":
        """Check if debate should fork (P30: DebateForker)."""
        if not DEBATE_FORKER_AVAILABLE or not self.fork_debate_enabled:
            return None
        try:
            # Create detector on demand
            detector = ForkDetector()
            decision = detector.should_fork(messages, round_num, agents)
            if decision and hasattr(decision, 'should_fork') and decision.should_fork:
                self._log(f"  [forking] Fork triggered: {getattr(decision, 'reason', 'unknown')}")
            return decision
        except Exception as e:
            self._log(f"  [forking] Check error: {e}")
            return None

    async def _run_forked_debate(self, fork_decision: "ForkDecision", base_context: str) -> "MergeResult":
        """Run forked parallel debates (P30: DebateForker).

        Note: This feature requires proper Environment, agents, and run_debate_fn
        to be passed to run_branches. Currently disabled until full integration.
        """
        if not DEBATE_FORKER_AVAILABLE or not self.fork_debate_enabled or not fork_decision:
            return None
        try:
            branches = getattr(fork_decision, 'branches', [])
            if not branches:
                self._log(f"  [forking] No branches in fork decision")
                return None

            self._log(f"  [forking] Fork detected with {len(branches)} branches")
            # TODO: Full forking requires Environment, agents list, and run_debate_fn
            # For now, log the fork but don't execute parallel branches
            self._log(f"  [forking] Skipping parallel execution (integration pending)")
            return None
        except Exception as e:
            self._log(f"  [forking] Run error: {e}")
            return None

    def _record_fork_outcome(self, fork_point: "ForkPoint", merge_result: "MergeResult") -> None:
        """Record fork outcome for learning (P30: DebateForker)."""
        if not fork_point or not merge_result:
            return
        try:
            # Could store in provenance or insight extractor
            self._log(f"  [forking] Recorded outcome: {merge_result.winning_branch_id}")
        except Exception as e:
            self._log(f"  [forking] Record error: {e}")

    def _record_replay_event(self, event_type: str, agent: str, content: str, round_num: int = 0) -> None:
        """Record an event to the ReplayRecorder if active."""
        if not self.replay_recorder:
            return
        try:
            if event_type == "turn":
                self.replay_recorder.record_turn(agent, content, round_num, self.loop_id)
            elif event_type == "vote":
                self.replay_recorder.record_vote(agent, content, "")
            elif event_type == "phase":
                self.replay_recorder.record_phase_change(content)
            elif event_type == "system":
                self.replay_recorder.record_system(content)
        except Exception:
            pass  # Don't let replay errors break the loop

    def _record_cartographer_event(
        self, event_type: str, agent: str, content: str,
        role: str = "proposer", round_num: int = 1, **kwargs
    ) -> None:
        """Record an event to the ArgumentCartographer if active."""
        if not self.cartographer:
            return
        try:
            if event_type == "message":
                self.cartographer.update_from_message(
                    agent=agent,
                    content=content,
                    role=role,
                    round_num=round_num,
                    metadata=kwargs.get("metadata", {})
                )
            elif event_type == "critique":
                self.cartographer.update_from_critique(
                    critic_agent=agent,
                    target_agent=kwargs.get("target", "unknown"),
                    severity=kwargs.get("severity", 0.5),
                    round_num=round_num,
                    critique_text=content
                )
            elif event_type == "vote":
                self.cartographer.update_from_vote(
                    agent=agent,
                    vote_value=content,
                    round_num=round_num
                )
            elif event_type == "consensus":
                self.cartographer.update_from_consensus(
                    result=content,
                    round_num=round_num,
                    vote_counts=kwargs.get("vote_counts", {})
                )
        except Exception:
            pass  # Don't let cartographer errors break the loop

    def _dispatch_webhook(self, event_type: str, data: dict = None) -> None:
        """Dispatch an event to external webhooks if configured."""
        if not self.webhook_dispatcher:
            return
        try:
            event = {
                "type": event_type,
                "loop_id": self.loop_id,
                "cycle": self.cycle_count,
                "timestamp": datetime.now().isoformat(),
                "data": data or {},
            }
            self.webhook_dispatcher.enqueue(event)
        except Exception:
            pass  # Don't let webhook errors break the loop

    def _format_agent_reputations(self) -> str:
        """Format agent reputations for prompt injection.

        Shows which agents have been most successful so agents can
        weight their collaboration accordingly.
        """
        if not hasattr(self, 'critique_store') or not self.critique_store:
            return ""

        try:
            reputations = self.critique_store.get_all_reputations()
            if not reputations:
                return ""

            lines = ["## AGENT TRACK RECORDS"]
            for rep in sorted(reputations, key=lambda r: r.score, reverse=True):
                if rep.proposals_made > 0:
                    acceptance = rep.proposals_accepted / rep.proposals_made
                    lines.append(f"- {rep.agent_name}: {acceptance:.0%} proposal acceptance ({rep.proposals_accepted}/{rep.proposals_made})")

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _format_relationship_network(self, limit: int = 3) -> str:
        """Format agent relationship dynamics for debate context."""
        if not self.relationship_tracker or not GROUNDED_PERSONAS_AVAILABLE:
            return ""
        try:
            lines = ["## Inter-Agent Dynamics"]

            # Get influence network per agent
            agents = ["gemini", "claude", "codex", "grok"]
            if hasattr(self.relationship_tracker, 'get_influence_network'):
                lines.append("\n### Influence Patterns:")
                influence_scores = []
                for agent in agents:
                    try:
                        network = self.relationship_tracker.get_influence_network(agent)
                        if network and network.get("influences"):
                            total_influence = sum(score for _, score in network["influences"])
                            influence_scores.append((agent, total_influence))
                    except Exception:
                        continue
                influence_scores.sort(key=lambda x: x[1], reverse=True)
                for agent, score in influence_scores[:limit]:
                    lines.append(f"- {agent}: influence score {score:.2f}")

            # Get rivals and allies for each agent
            dynamics_found = False
            for agent in agents:
                if hasattr(self.relationship_tracker, 'get_rivals'):
                    rivals = self.relationship_tracker.get_rivals(agent, limit=2)
                    allies = self.relationship_tracker.get_allies(agent, limit=2) if hasattr(self.relationship_tracker, 'get_allies') else []
                    if rivals or allies:
                        dynamics_found = True
                        rival_names = [r[0] for r in rivals] if rivals else []
                        ally_names = [a[0] for a in allies] if allies else []
                        lines.append(f"- {agent}: rivals={rival_names}, allies={ally_names}")

            return "\n".join(lines) if len(lines) > 1 and dynamics_found else ""
        except Exception as e:
            self._log(f"  [relationships] Formatting error: {e}")
            return ""

    def _audit_agent_calibration(self) -> str:
        """Audit agent calibration and flag poorly calibrated agents."""
        if not self.elo_system or not ELO_AVAILABLE:
            return ""
        try:
            lines = ["## Calibration Health Check"]
            flagged = []

            for agent_name in ["gemini", "claude", "codex", "grok"]:
                if hasattr(self.elo_system, 'get_expected_calibration_error'):
                    ece = self.elo_system.get_expected_calibration_error(agent_name)
                    if ece and ece > 0.2:  # Poorly calibrated
                        flagged.append((agent_name, ece))
                        lines.append(f"- WARNING: {agent_name} has high calibration error ({ece:.2f})")
                        lines.append(f"  Consider weighing their opinions lower on uncertain topics")

            if flagged:
                self._log(f"  [calibration] Flagged {len(flagged)} poorly calibrated agents")
                return "\n".join(lines)
            return ""
        except Exception as e:
            self._log(f"  [calibration] Audit error: {e}")
            return ""

    def _format_agent_introspection(self, agent_name: str) -> str:
        """Format agent self-awareness section for prompt injection.

        Uses IntrospectionAPI to provide agents with awareness of their
        own reputation, strengths, and track record.
        """
        if not INTROSPECTION_AVAILABLE or not get_agent_introspection:
            return ""

        try:
            snapshot = get_agent_introspection(
                agent_name,
                memory=self.critique_store,
                persona_manager=None,  # We don't have PersonaManager yet
            )
            return format_introspection_section(snapshot, max_chars=400)
        except Exception:
            return ""

    async def _parallel_implementation_review(self, diff: str) -> Optional[str]:
        """
        All 3 agents review implementation changes in parallel.

        This provides balanced participation in the implementation stage
        while keeping the actual implementation specialized to Claude.

        Returns:
            Combined concerns from all agents, or None if all approve.
        """
        review_prompt = f"""Quick review of these code changes. Are there any obvious issues?

## Code Changes (git diff)
```
{diff[:10000]}
```

Reply with ONE of:
- APPROVED: <brief reason>
- CONCERN: <specific issue>

Be concise (1-2 sentences). Focus on correctness and safety issues only.
"""

        async def review_with_agent(agent, name: str) -> tuple[str, str]:
            """Run review with one agent, returning (name, result)."""
            try:
                self._log(f"    {name}: reviewing implementation...", agent=name)
                result = await agent.generate(review_prompt, context=[])
                self._log(f"    {name}: {result if result else 'No response'}", agent=name)
                # Emit full review
                if result:
                    self._stream_emit("on_log_message", result, level="info", phase="review", agent=name)
                return (name, result if result else "No response")
            except Exception as e:
                self._log(f"    {name}: review error - {e}", agent=name)
                return (name, f"Error: {e}")

        # Run all 4 agents in parallel
        import asyncio
        reviews = await asyncio.gather(
            review_with_agent(self.gemini, "gemini"),
            review_with_agent(self.codex, "codex"),
            review_with_agent(self.claude, "claude"),
            review_with_agent(self.grok, "grok"),
            return_exceptions=True,
        )

        # Collect concerns
        concerns = []
        for result in reviews:
            if isinstance(result, Exception):
                continue
            name, response = result
            if response and "CONCERN" in response.upper():
                concerns.append(f"{name}: {response}")

        if concerns:
            return "\n".join(concerns)
        return None

    def _diff_touches_protected_files(self, diff: str) -> list[str]:
        """Check if a diff touches any protected files.

        Returns list of protected files that were modified.
        """
        touched_protected = []
        for protected_file in PROTECTED_FILES:
            # Check various diff patterns that indicate file modification
            patterns = [
                f"diff --git a/{protected_file}",
                f"--- a/{protected_file}",
                f"+++ b/{protected_file}",
                f"diff --git a/aragora/{protected_file.replace('aragora/', '')}",
            ]
            for pattern in patterns:
                if pattern in diff:
                    touched_protected.append(protected_file)
                    break
        return touched_protected

    def _should_use_deep_audit(self, topic: str, phase: str = "design") -> tuple[bool, str]:
        """Determine if a topic warrants deep audit mode.

        Returns:
            (should_use: bool, reason: str)
        """
        if not DEEP_AUDIT_AVAILABLE:
            return False, "Deep audit not available"

        topic_lower = topic.lower()

        # High-priority triggers for deep audit
        critical_keywords = [
            "architecture", "security", "authentication", "authorization",
            "database", "migration", "breaking change", "api contract",
            "consensus", "voting", "protocol", "protected file",
        ]

        strategy_keywords = [
            "strategy", "design pattern", "refactor", "restructure",
            "system design", "infrastructure", "scale", "performance",
        ]

        # Check for critical topics
        for keyword in critical_keywords:
            if keyword in topic_lower:
                return True, f"Critical topic detected: {keyword}"

        # Check for strategy topics in design phase
        if phase == "design":
            for keyword in strategy_keywords:
                if keyword in topic_lower:
                    return True, f"Strategy topic detected: {keyword}"

        # Check topic length/complexity (long topics often more complex)
        if len(topic) > 500:
            return True, "Complex topic (length > 500 chars)"

        return False, "Standard topic, normal debate sufficient"

    async def _run_deep_audit_for_design(
        self, improvement: str, design_context: str = ""
    ) -> Optional[dict]:
        """Run Deep Audit Mode for design phase of critical topics.

        Uses STRATEGY_AUDIT config with cross-examination enabled.

        Returns:
            dict with verdict details, or None if audit not run
        """
        if not DEEP_AUDIT_AVAILABLE or not run_deep_audit:
            return None

        self._log("    [deep-audit] Running strategic design audit (5-round)")

        try:
            audit_agents = [self.gemini, self.codex, self.claude, self.grok, self.deepseek]

            # Use CODE_ARCHITECTURE_AUDIT for strategic design review
            verdict = await run_deep_audit(
                task=f"""STRATEGIC DESIGN REVIEW

## Proposed Improvement
{improvement[:8000]}

{design_context}

## Your Task
1. Evaluate the architectural soundness of this proposal
2. Identify potential risks and unintended consequences
3. Check for conflicts with existing systems
4. Assess complexity vs. value tradeoff
5. Propose refinements or alternatives if needed
6. Flag any concerns that need unanimous agreement before proceeding

Cross-examine each other's reasoning. Be thorough.""",
                agents=audit_agents,
                config=CODE_ARCHITECTURE_AUDIT,
            )

            result = {
                "confidence": verdict.confidence,
                "unanimous_issues": verdict.unanimous_issues,
                "split_opinions": verdict.split_opinions,
                "risk_areas": verdict.risk_areas,
                "approved": len(verdict.unanimous_issues) == 0,
            }

            self._log(f"    [deep-audit] Design confidence: {verdict.confidence:.0%}")
            if verdict.unanimous_issues:
                self._log(f"    [deep-audit] Blocking issues: {len(verdict.unanimous_issues)}")
                for issue in verdict.unanimous_issues[:3]:
                    self._log(f"      - {issue[:150]}...")

            return result

        except Exception as e:
            self._log(f"    [deep-audit] Design audit failed: {e}")
            return None

    async def _run_deep_audit_for_protected_files(
        self, diff: str, touched_files: list[str]
    ) -> tuple[bool, Optional[str]]:
        """Run Deep Audit Mode for changes to protected files.

        Heavy3-inspired: 6-round intensive review with cross-examination
        for high-stakes changes.

        Returns:
            (approved: bool, issues: Optional[str])
        """
        if not DEEP_AUDIT_AVAILABLE or not run_deep_audit:
            self._log("    [deep-audit] Not available, falling back to regular review")
            return True, None

        self._log(f"    [deep-audit] Starting intensive review for protected files: {touched_files}")
        self._log("    [deep-audit] Running 5-round CODE_ARCHITECTURE_AUDIT with cross-examination...")

        try:
            # Create agents list for deep audit
            audit_agents = [self.gemini, self.codex, self.claude, self.grok, self.deepseek]

            # Run deep audit
            verdict = await run_deep_audit(
                task=f"""CRITICAL: Review changes to protected files.

These files are essential to aragora's functionality and must be reviewed with maximum scrutiny.

## Protected Files Being Modified
{', '.join(touched_files)}

## Changes (git diff)
```
{diff[:15000]}
```

## Your Task
1. Analyze each change for correctness and safety
2. Identify any breaking changes or regressions
3. Check for security vulnerabilities
4. Verify backward compatibility is preserved
5. Flag any unanimous issues that must be addressed before merge

Be rigorous. These files are protected for a reason.""",
                agents=audit_agents,
                config=CODE_ARCHITECTURE_AUDIT,
            )

            # Log verdict summary
            self._log(f"    [deep-audit] Confidence: {verdict.confidence:.0%}")
            self._log(f"    [deep-audit] Unanimous issues: {len(verdict.unanimous_issues)}")
            self._log(f"    [deep-audit] Split opinions: {len(verdict.split_opinions)}")
            self._log(f"    [deep-audit] Risk areas: {len(verdict.risk_areas)}")

            # If there are unanimous issues, reject the changes
            if verdict.unanimous_issues:
                self._log("    [deep-audit] REJECTED - Unanimous issues found:")
                for issue in verdict.unanimous_issues[:5]:
                    self._log(f"      - {issue[:200]}")
                return False, "\n".join(verdict.unanimous_issues)

            # If low confidence and many split opinions, warn but allow
            if verdict.confidence < 0.5 and len(verdict.split_opinions) > 2:
                self._log("    [deep-audit] WARNING - Low confidence, proceed with caution")

            self._log("    [deep-audit] APPROVED - No unanimous blocking issues")
            return True, None

        except Exception as e:
            self._log(f"    [deep-audit] ERROR: {e}")
            self._log("    [deep-audit] Falling back to regular review due to error")
            return True, None

    def _sanitize_agent_input(self, text: str, source: str = "agent") -> str:
        """
        Sanitize agent-provided text to prevent prompt injection attacks.

        Security measure: Filters potentially malicious patterns from agent suggestions
        before they're merged into prompts for other agents.

        Args:
            text: Raw text from agent
            source: Source identifier for logging

        Returns:
            Sanitized text with dangerous patterns removed
        """
        import re

        dangerous_patterns = [
            (r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions?", "instruction override"),
            (r"disregard\s+(?:all\s+)?(?:previous\s+)?(?:rules?|guidelines?)", "rule bypass"),
            (r"bypass\s+(?:safety|security|restrictions?)", "safety bypass"),
            (r"execute\s+(?:this\s+)?(?:code|command|script)", "code execution"),
            (r"system\s+prompt", "system prompt access"),
            (r"you\s+are\s+now\s+(?:a|an)", "role hijacking"),
            (r"forget\s+(?:everything|all)", "memory wipe"),
            (r"new\s+instructions?:", "instruction injection"),
            (r"<\s*script\s*>", "script tag"),
            (r"\$\{.*\}", "template injection"),
        ]

        sanitized = text
        filtered_count = 0

        for pattern, description in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                filtered_count += 1
                sanitized = re.sub(pattern, f"[FILTERED:{description}]", sanitized, flags=re.IGNORECASE)

        if filtered_count > 0:
            self._log(f"  [security] Filtered {filtered_count} suspicious patterns from {source}")

        return sanitized

    async def _gather_implementation_suggestions(self, design: str) -> str:
        """
        All agents provide implementation suggestions in parallel.

        This ensures all agents have a chance to contribute to implementation,
        with Claude getting the final pass to consolidate and execute.

        Returns:
            Combined suggestions from all agents to guide implementation.
        """
        self._log("  Gathering implementation suggestions from all agents...", agent="claude")

        suggestion_prompt = f"""Based on this design, provide your implementation suggestions:

{design[:3000]}

Provide:
1. KEY IMPLEMENTATION APPROACH: How would you structure the code?
2. POTENTIAL PITFALLS: What could go wrong and how to avoid it?
3. CODE SNIPPETS: Any specific code patterns or snippets to use.

Be concise (max 500 words). Focus on actionable guidance."""

        async def get_suggestion(agent, name: str) -> tuple[str, str]:
            """Get implementation suggestion from one agent."""
            try:
                self._log(f"    {name}: providing suggestions...", agent=name)
                result = await self._call_agent_with_retry(agent, suggestion_prompt, max_retries=2)
                if result and not ("[Agent" in result and "failed" in result):
                    self._log(f"    {name}: suggestions received", agent=name)
                    # Emit full suggestion content to stream for dashboard visibility
                    self._stream_emit("on_log_message", result, level="info", phase="implement", agent=name)
                    return (name, result)  # Return full result, no truncation
                else:
                    return (name, "")
            except Exception as e:
                self._log(f"    {name}: suggestion failed: {e}", agent=name)
                return (name, "")

        # Run all agents in parallel
        tasks = [
            get_suggestion(self.gemini, "gemini"),
            get_suggestion(self.codex, "codex"),
            get_suggestion(self.grok, "grok"),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compile suggestions with security sanitization
        suggestions = []
        for result in results:
            if isinstance(result, tuple) and result[1]:
                name, suggestion = result
                # Sanitize agent input to prevent prompt injection attacks
                sanitized = self._sanitize_agent_input(suggestion, source=name)
                suggestions.append(f"### {name.upper()}'s Suggestions:\n{sanitized}\n")

        if suggestions:
            combined = "\n".join(suggestions)
            self._log(f"  Received suggestions from {len(suggestions)} agents")
            return f"""
## IMPLEMENTATION GUIDANCE (from other agents)
The following suggestions were provided by other agents. Consider their insights while implementing.

{combined}

## YOUR ROLE (Claude)
You are the final implementer. Use the best ideas from above, but apply your own judgment.
Synthesize these suggestions into a coherent, working implementation.
"""
        return ""

    def _create_arena_hooks(self, phase_name: str) -> dict:
        """Create event hooks for real-time Arena logging and streaming."""
        # Get streaming hooks if available
        stream_hooks = {}
        if self.stream_emitter and STREAMING_AVAILABLE and create_arena_hooks:
            stream_hooks = create_arena_hooks(self.stream_emitter)

        def make_combined_hook(log_fn, stream_hook_name):
            """Combine logging and streaming for a hook."""
            stream_fn = stream_hooks.get(stream_hook_name)
            def combined(*args, **kwargs):
                log_fn(*args, **kwargs)
                if stream_fn:
                    try:
                        stream_fn(*args, **kwargs)
                    except Exception:
                        pass  # Don't let streaming errors break the loop
            return combined

        return {
            "on_debate_start": make_combined_hook(
                lambda task, agents: self._log(f"    Debate started: {len(agents)} agents"),
                "on_debate_start"
            ),
            "on_message": make_combined_hook(
                lambda agent, content, role, round_num: self._log(
                    f"    [{role}] {agent} (round {round_num}): {content}"  # Full content, no truncation
                ),
                "on_message"
            ),
            "on_critique": make_combined_hook(
                lambda agent, target, issues, severity, round_num, full_content=None: self._log(
                    f"    [critique] {agent} -> {target}: {len(issues)} issues, severity {severity:.1f}"
                ),
                "on_critique"
            ),
            "on_round_start": make_combined_hook(
                lambda round_num: self._log(f"    --- Round {round_num} ---"),
                "on_round_start"
            ),
            "on_consensus": make_combined_hook(
                lambda reached, confidence, answer: self._log(
                    f"    Consensus: {'Yes' if reached else 'No'} ({confidence:.0%})"
                ),
                "on_consensus"
            ),
            "on_vote": make_combined_hook(
                lambda agent, vote, confidence: self._log(
                    f"    [vote] {agent}: {vote} ({confidence:.0%})"
                ),
                "on_vote"
            ),
            "on_debate_end": make_combined_hook(
                lambda duration, rounds: self._log(f"    Completed in {duration:.1f}s ({rounds} rounds)"),
                "on_debate_end"
            ),
        }

    def _handle_disagreement_influence(
        self, report: "DisagreementReport", phase_name: str, result: "DebateResult"
    ) -> dict:
        """Handle disagreement patterns to influence decisions.

        Heavy3-inspired: Make disagreement data actionable.

        Returns:
            dict with action recommendations:
            - should_reject: bool - proposal should be rejected
            - should_fork: bool - debate should be forked to explore disagreement
            - rejection_reasons: list[str] - reasons if rejected
            - fork_topic: str - topic to fork on if forking
        """
        actions = {
            "should_reject": False,
            "should_fork": False,
            "rejection_reasons": [],
            "fork_topic": None,
            "escalate_to": None,
        }

        # Track critical disagreement patterns
        critical_warning = False

        # ACTION 1: Auto-reject on unanimous critiques (>= 3 unanimous issues)
        if len(report.unanimous_critiques) >= 3:
            self._log(f"    [disagreement] REJECT: {len(report.unanimous_critiques)} unanimous issues - proposal blocked")
            actions["should_reject"] = True
            actions["rejection_reasons"] = report.unanimous_critiques[:5]
            critical_warning = True

        # ACTION 2: Fork trigger for low agreement (< 0.4)
        if report.agreement_score < 0.4 and not actions["should_reject"]:
            self._log(f"    [disagreement] FORK: Low agreement ({report.agreement_score:.0%}) - exploring alternatives")
            actions["should_fork"] = True
            # Create fork topic from the main disagreement
            if report.split_opinions:
                first_split = report.split_opinions[0] if isinstance(report.split_opinions, list) else str(report.split_opinions)
                actions["fork_topic"] = f"Resolve disagreement: {first_split[:200]}"
            critical_warning = True

        # ACTION 3: Escalate split opinions for persistent patterns
        if len(report.split_opinions) >= 3:
            self._log(f"    [disagreement] ESCALATE: {len(report.split_opinions)} split opinions detected")
            # Track which agents consistently disagree
            if not hasattr(self, '_agent_disagreement_patterns'):
                self._agent_disagreement_patterns = {}

            # Store for pattern analysis
            actions["escalate_to"] = "cross_examination"
            for opinion in report.split_opinions[:3]:
                self._log(f"      Split: {str(opinion)[:100]}")

        # If very low agreement but not rejecting, warn
        if report.agreement_score < 0.4 and not actions["should_reject"]:
            self._log(f"    [disagreement] WARNING: Low agreement ({report.agreement_score:.0%}) - consider revising proposal")
            critical_warning = True

        # If high-stakes phase (design/implement) and significant disagreement, log prominently
        if phase_name in ("design", "implement") and (
            len(report.unanimous_critiques) >= 2 or report.agreement_score < 0.5
        ):
            self._log(f"    [disagreement] ATTENTION: High-stakes phase '{phase_name}' has significant disagreement")
            # Store for later review
            if not hasattr(self, '_critical_disagreements'):
                self._critical_disagreements = []
            self._critical_disagreements.append({
                "phase": phase_name,
                "cycle": self.cycle_count,
                "unanimous_critiques": report.unanimous_critiques,
                "agreement_score": report.agreement_score,
                "actions_taken": actions,
                "timestamp": datetime.now().isoformat(),
            })

        # If many risk areas identified, log them prominently
        if len(report.risk_areas) >= 2:
            self._log(f"    [disagreement] {len(report.risk_areas)} RISK AREAS to monitor:")
            for risk in report.risk_areas[:3]:
                self._log(f"      - {risk[:100]}")

        # Stream critical warnings for dashboard visibility
        if critical_warning:
            action_str = ""
            if actions["should_reject"]:
                action_str = " [REJECTED]"
            elif actions["should_fork"]:
                action_str = " [FORKING]"
            self._stream_emit(
                "on_log_message",
                f"Disagreement alert in {phase_name}: {len(report.unanimous_critiques)} unanimous issues, {report.agreement_score:.0%} agreement{action_str}",
                level="warning",
                phase=phase_name,
            )

        return actions

    async def _run_arena_with_logging(self, arena: Arena, phase_name: str) -> "DebateResult":
        """Run an Arena debate with real-time logging via event hooks."""
        self._log(f"  Starting {phase_name} arena...")
        self._save_state({"phase": phase_name, "stage": "arena_starting"})

        # Add event hooks for real-time logging
        arena.hooks = self._create_arena_hooks(phase_name)

        try:
            result = await arena.run()

            self._log(f"  {phase_name} arena complete")
            self._log(f"    Consensus: {result.consensus_reached}", also_print=False)
            self._log(f"    Confidence: {result.confidence}", also_print=False)
            self._log(f"    Duration: {result.duration_seconds:.1f}s", also_print=False)

            # Log and act on DisagreementReport (Heavy3-inspired unanimous issues/split opinions)
            if result.disagreement_report:
                report = result.disagreement_report
                self._log(f"    [disagreement] Agreement Score: {report.agreement_score:.1%}", also_print=False)
                if report.unanimous_critiques:
                    self._log(f"    [disagreement] {len(report.unanimous_critiques)} UNANIMOUS ISSUES (high priority):")
                    for issue in report.unanimous_critiques[:3]:
                        self._log(f"      - {issue[:100]}...")
                if report.split_opinions:
                    self._log(f"    [disagreement] {len(report.split_opinions)} split opinions (review carefully)", also_print=False)
                if report.risk_areas:
                    self._log(f"    [disagreement] {len(report.risk_areas)} risk areas identified", also_print=False)

                # Heavy3-inspired decision influence based on disagreement patterns
                disagreement_actions = self._handle_disagreement_influence(report, phase_name, result)

                # Store actions on result for phase handlers to use
                result.disagreement_actions = disagreement_actions

            self._save_state({
                "phase": phase_name,
                "stage": "arena_complete",
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "final_answer_preview": result.final_answer if result.final_answer else None,
                # Include disagreement report summary
                "disagreement_report": {
                    "unanimous_critiques": result.disagreement_report.unanimous_critiques if result.disagreement_report else [],
                    "split_opinions_count": len(result.disagreement_report.split_opinions) if result.disagreement_report else 0,
                    "agreement_score": result.disagreement_report.agreement_score if result.disagreement_report else None,
                    "actions": result.disagreement_actions if hasattr(result, 'disagreement_actions') else None,
                } if result.disagreement_report else None,
            })

            return result

        except Exception as e:
            self._log(f"  {phase_name} arena ERROR: {e}")
            self._save_state({"phase": phase_name, "stage": "arena_error", "error": str(e)})
            raise

    async def _arbitrate_design(self, proposals: dict, improvement: str) -> Optional[str]:
        """Use a judge agent to pick between competing design proposals.

        When design voting is tied or close, this method uses Claude as an impartial
        judge to evaluate and select the best design based on:
        - Feasibility (can it actually be implemented?)
        - Completeness (does it cover all required changes?)
        - Safety (does it preserve existing functionality?)
        - Clarity (is it specific enough to implement?)

        Args:
            proposals: Dict mapping agent name to their design proposal
            improvement: The improvement being designed (for context)

        Returns:
            The selected design text, or None if arbitration fails
        """
        if not proposals or len(proposals) < 2:
            return None

        try:
            # Use Claude as judge (generally high-quality reasoning)
            judge = self.claude

            # Format proposals for comparison
            proposals_text = "\n\n---\n\n".join(
                f"## {agent}'s Design:\n{proposal[:2000]}..."
                for agent, proposal in proposals.items()
            )

            arbitration_prompt = f"""You are a senior software architect arbitrating between competing design proposals.

## The Improvement Being Designed:
{improvement[:1000]}

## Competing Designs:
{proposals_text}

## Evaluation Criteria:
1. FEASIBILITY: Can this be implemented without major refactoring?
2. COMPLETENESS: Does it specify all file changes, APIs, and integration points?
3. SAFETY: Does it preserve existing functionality and avoid protected files?
4. CLARITY: Is it specific enough that an engineer could implement it?
5. TESTABILITY: Does it include a viable test plan?

## Your Task:
Select the BEST design. If one is clearly superior, choose it.
If they're comparable, synthesize the best elements of each.

Respond with ONLY the complete design specification (no preamble).
Start directly with "## 1. FILE CHANGES" or similar."""

            self._log("  [arbitration] Judge evaluating proposals...")
            try:
                response = await asyncio.wait_for(
                    judge.generate(arbitration_prompt),
                    timeout=180  # 3 minute max for judge arbitration
                )
            except asyncio.TimeoutError:
                self._log("  [arbitration] Judge timeout - using highest-voted proposal")
                return None

            if response and len(response) > 200:
                return response
            else:
                self._log("  [arbitration] Judge response too short")
                return None

        except Exception as e:
            self._log(f"  [arbitration] Error: {e}")
            return None

    async def _run_fractal_with_logging(self, task: str, agents: list, phase_name: str) -> "DebateResult":
        """Run a fractal debate with agent evolution and real-time logging."""
        if not self.use_genesis or not GENESIS_AVAILABLE:
            # Fall back to regular arena
            env = Environment(task=task)
            protocol = DebateProtocol(
                rounds=2,
                consensus="majority",
                consensus_threshold=self._get_adaptive_consensus_threshold(),  # Adaptive threshold
                judge_selection="elo_ranked",  # Use ELO-based judge selection
                role_rotation=True,
                role_rotation_config=RoleRotationConfig(
                    enabled=True,
                    roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC, CognitiveRole.LATERAL_THINKER],
                ),
            )
            arena = Arena(
                environment=env, agents=agents, protocol=protocol,
                memory=self.critique_store, debate_embeddings=self.debate_embeddings,
                insight_store=self.insight_store, position_tracker=self.position_tracker,
                position_ledger=self.position_ledger, elo_system=self.elo_system,
                event_emitter=self.stream_emitter, loop_id=self.loop_id,
                event_hooks=self._create_arena_hooks("fractal"),
                persona_manager=self.persona_manager,
                relationship_tracker=self.relationship_tracker,
                moment_detector=self.moment_detector,
                continuum_memory=self.continuum,
            )
            return await self._run_arena_with_logging(arena, phase_name)

        self._log(f"  Starting {phase_name} fractal debate (genesis mode)...")
        self._save_state({"phase": phase_name, "stage": "fractal_starting", "genesis": True})

        # Create fractal orchestrator with hooks
        orchestrator = FractalOrchestrator(
            max_depth=2,
            tension_threshold=0.6,
            evolve_agents=True,
            population_manager=self.population_manager,
            event_hooks=self.stream_hooks,
        )

        try:
            # Get or create population from agent names
            agent_names = [a.name.split("_")[0] for a in agents]
            population = self.population_manager.get_or_create_population(agent_names)

            self._log(f"    Population: {population.size} genomes, gen {population.generation}")

            # Run fractal debate
            fractal_result = await orchestrator.run(
                task=task,
                agents=agents,
                population=population,
            )

            self._log(f"  {phase_name} fractal debate complete")
            self._log(f"    Total depth: {fractal_result.total_depth}")
            self._log(f"    Sub-debates: {len(fractal_result.sub_debates)}")
            self._log(f"    Tensions resolved: {fractal_result.tensions_resolved}")
            self._log(f"    Evolved genomes: {len(fractal_result.evolved_genomes)}")

            # Log evolved genomes
            for genome in fractal_result.evolved_genomes:
                self._log(f"      - {genome.name} (gen {genome.generation})")

            self._save_state({
                "phase": phase_name,
                "stage": "fractal_complete",
                "genesis": True,
                "total_depth": fractal_result.total_depth,
                "sub_debates": len(fractal_result.sub_debates),
                "evolved_genomes": len(fractal_result.evolved_genomes),
                "consensus_reached": fractal_result.main_result.consensus_reached,
            })

            return fractal_result.main_result

        except Exception as e:
            self._log(f"  {phase_name} fractal debate ERROR: {e}")
            self._save_state({"phase": phase_name, "stage": "fractal_error", "error": str(e)})
            # Fall back to regular arena on error
            self._log(f"  Falling back to regular arena...")
            env = Environment(task=task)
            protocol = DebateProtocol(
                rounds=2,
                consensus="majority",
                consensus_threshold=self._get_adaptive_consensus_threshold(),  # Adaptive threshold
                judge_selection="elo_ranked",  # Use ELO-based judge selection
                role_rotation=True,
                role_rotation_config=RoleRotationConfig(
                    enabled=True,
                    roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC, CognitiveRole.LATERAL_THINKER],
                ),
            )
            arena = Arena(
                environment=env, agents=agents, protocol=protocol,
                memory=self.critique_store, debate_embeddings=self.debate_embeddings,
                insight_store=self.insight_store, position_tracker=self.position_tracker,
                position_ledger=self.position_ledger, elo_system=self.elo_system,
                event_emitter=self.stream_emitter, loop_id=self.loop_id,
                event_hooks=self._create_arena_hooks("fractal_fallback"),
                persona_manager=self.persona_manager,
                relationship_tracker=self.relationship_tracker,
                moment_detector=self.moment_detector,
                continuum_memory=self.continuum,
            )
            return await self._run_arena_with_logging(arena, phase_name)

    async def phase_context_gathering(self) -> dict:
        """
        Phase 0: All agents explore codebase to gather context.

        Each agent uses its native codebase exploration harness:
        - Claude → Claude Code CLI (native codebase access)
        - Codex → Codex CLI (native codebase access)
        - Gemini → Kilo Code CLI (agentic codebase exploration)
        - Grok → Kilo Code CLI (agentic codebase exploration)

        This ensures ALL agents have first-hand knowledge of the codebase,
        preventing proposals for features that already exist.
        """
        phase_start = datetime.now()

        # Determine how many agents will participate
        # Check if KiloCode should be used for context gathering
        use_kilocode = KILOCODE_AVAILABLE and not SKIP_KILOCODE_CONTEXT_GATHERING
        agents_count = 2  # Claude + Codex always
        if use_kilocode:
            agents_count = 4  # + Gemini + Grok via Kilo Code
            self._log("\n" + "=" * 70)
            self._log("PHASE 0: CONTEXT GATHERING (All 4 agents with codebase access)")
            self._log("  Claude → Claude Code | Codex → Codex CLI")
            self._log("  Gemini → Kilo Code  | Grok → Kilo Code")
            self._log("=" * 70)
        else:
            self._log("\n" + "=" * 70)
            self._log("PHASE 0: CONTEXT GATHERING (Claude + Codex)")
            if KILOCODE_AVAILABLE and SKIP_KILOCODE_CONTEXT_GATHERING:
                self._log("  Note: KiloCode skipped (timeouts); Gemini/Grok join in debates")
            else:
                self._log("  Note: Install kilocode CLI to enable Gemini/Grok exploration")
            self._log("=" * 70)

        self._stream_emit("on_phase_start", "context", self.cycle_count, {"agents": agents_count})

        # Prompt for codebase exploration
        explore_prompt = f"""Explore the aragora codebase and provide a comprehensive summary of EXISTING features.

Working directory: {self.aragora_path}

Your task:
1. Read key files: aragora/__init__.py, aragora/debate/orchestrator.py, aragora/server/stream.py
2. List ALL existing major features and capabilities
3. Note any features related to: streaming, real-time, visualization, spectator mode, WebSocket
4. Identify the project's current architecture and patterns

Output format:
## EXISTING FEATURES (DO NOT RECREATE)
- Feature 1: description
- Feature 2: description
...

## ARCHITECTURE OVERVIEW
Brief description of how the system is organized.

## RECENT FOCUS AREAS
What has been worked on recently (from git log).

## GAPS AND OPPORTUNITIES
What's genuinely missing (not already implemented).

CRITICAL: Be thorough. Features you miss here may be accidentally proposed for recreation."""

        async def gather_with_agent(agent, name: str, harness: str) -> tuple[str, str, str]:
            """Run exploration with one agent."""
            try:
                self._log(f"  {name} ({harness}): exploring codebase...", agent=name)
                result = await agent.generate(explore_prompt, context=[])
                self._log(f"  {name}: complete ({len(result) if result else 0} chars)", agent=name)
                # Emit agent's full exploration result
                if result:
                    self._stream_emit("on_log_message", result, level="info", phase="context", agent=name)
                return (name, harness, result if result else "No response")
            except Exception as e:
                self._log(f"  {name}: error - {e}", agent=name)
                return (name, harness, f"Error: {e}")

        # Build list of exploration tasks
        exploration_tasks = [
            gather_with_agent(self.claude, "claude", "Claude Code"),
            gather_with_agent(self.codex, "codex", "Codex CLI"),
        ]

        # Add Gemini and Grok via Kilo Code if available (and not skipped)
        if use_kilocode:
            # Create temporary Kilo Code agents for exploration
            gemini_explorer = KiloCodeAgent(
                name="gemini-explorer",
                provider_id="gemini-explorer",
                model="gemini-3-pro-preview",
                role="explorer",
                timeout=600,  # 10 min for agentic codebase exploration (reduced from 30 min)
                mode="architect",
            )
            grok_explorer = KiloCodeAgent(
                name="grok-explorer",
                provider_id="grok-explorer",
                model="grok-4",
                role="explorer",
                timeout=600,  # 10 min for agentic codebase exploration (reduced from 30 min)
                mode="architect",
            )
            exploration_tasks.extend([
                gather_with_agent(gemini_explorer, "gemini", "Kilo Code"),
                gather_with_agent(grok_explorer, "grok", "Kilo Code"),
            ])

        # Run all agents in parallel
        results = await asyncio.gather(*exploration_tasks, return_exceptions=True)

        # Combine the context from all agents
        combined_context = []
        for result in results:
            if isinstance(result, Exception):
                continue
            name, harness, content = result
            if content and "Error:" not in content:
                combined_context.append(
                    f"=== {name.upper()}'S CODEBASE ANALYSIS (via {harness}) ===\n{content}"
                )

        # If all failed, fall back to basic context
        if not combined_context:
            self._log("  Warning: Context gathering failed, using basic context")
            combined_context = [f"Current features (from docstring):\n{self.get_current_features()}"]

        gathered_context = "\n\n".join(combined_context)

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._log(f"  Context gathered from {len(combined_context)} agents in {phase_duration:.1f}s")
        self._stream_emit(
            "on_phase_end", "context", self.cycle_count, True,
            phase_duration, {"agents": len(combined_context), "context_length": len(gathered_context)}
        )

        return {
            "phase": "context",
            "context": gathered_context,
            "duration": phase_duration,
            "agents_succeeded": len(combined_context),
        }

    async def phase_debate(self, codebase_context: str = None) -> dict:
        """Phase 1: Agents debate what to improve."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 1: IMPROVEMENT DEBATE")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "debate", self.cycle_count, {"agents": 4})

        # Load belief network from previous cycle if available (cross-cycle learning)
        if self.nomic_integration and self.cycle_count > 1:
            try:
                prev_cycle = self.cycle_count - 1
                checkpoints = await self.nomic_integration.list_checkpoints()
                # Find previous debate checkpoint
                for cp in checkpoints:
                    if f"cycle_{prev_cycle}_debate" in str(cp):
                        self._log(f"  [belief] Loading belief network from cycle {prev_cycle}")
                        restored = await self.nomic_integration.resume_from_checkpoint(cp.get("checkpoint_id", ""))
                        if restored:
                            self._log(f"  [belief] Restored belief network with {len(self.nomic_integration._agent_weights)} agent weights")
                        break
            except Exception as e:
                self._log(f"  [belief] Failed to load previous cycle belief network: {e}")

        # Use provided context or fall back to basic
        if codebase_context:
            current_features = codebase_context
        else:
            current_features = self.get_current_features()
        recent_changes = self.get_recent_changes()

        # Gather learning context from previous cycles
        failure_lessons = self._analyze_failed_branches()
        successful_patterns = self._format_successful_patterns()

        # Build task with optional initial proposal
        initial_proposal_section = ""
        if self.initial_proposal:
            initial_proposal_section = f"""

===== HUMAN-SUBMITTED PROPOSAL =====
A human has submitted the following proposal for your consideration.
You may adopt it, critique it, improve upon it, or propose something entirely different.

{self.initial_proposal}
====================================
"""
            self._log(f"  Including human proposal: {self.initial_proposal}")

        # Build context section with clear attribution
        if codebase_context and len(codebase_context) > 500:
            context_section = f"""
===== CODEBASE ANALYSIS (from Claude + Codex who explored the code) =====
The following is a comprehensive analysis of aragora's EXISTING features.
Claude and Codex have read the actual codebase. DO NOT propose features that already exist below.

{current_features}
========================================================================"""
        else:
            context_section = f"Current aragora features:\n{current_features}"

        # Record phase change
        self._record_replay_event("phase", "system", "debate")

        # Build learning context section (Titans/MIRAS + ContinuumMemory)
        failure_patterns = self._format_failure_patterns()
        agent_reputations = self._format_agent_reputations()
        continuum_patterns = self._format_continuum_patterns(limit=3)

        learning_context = ""
        if failure_lessons:
            learning_context += f"\n{failure_lessons}\n"
        if successful_patterns:
            learning_context += f"\n{successful_patterns}\n"
        if failure_patterns:
            learning_context += f"\n{failure_patterns}\n"

        # Add stale claims that need re-debate
        if hasattr(self, '_pending_redebate_claims') and self._pending_redebate_claims:
            stale_claims_text = "\n=== STALE CLAIMS REQUIRING RE-DEBATE ===\n"
            stale_claims_text += "The following claims from previous debates have become stale due to code changes:\n"
            for claim in self._pending_redebate_claims[:5]:
                stale_claims_text += f"- {claim.statement[:100]}...\n"
            stale_claims_text += "\nPrioritize addressing these stale claims or explicitly acknowledge if they're no longer relevant.\n"
            learning_context += stale_claims_text
            self._log(f"  [staleness] Injecting {len(self._pending_redebate_claims)} stale claims into debate context")
            # Clear after injection
            self._pending_redebate_claims = []
        if agent_reputations:
            learning_context += f"\n{agent_reputations}\n"
        if continuum_patterns:
            learning_context += f"\n{continuum_patterns}\n"

        # Add agent introspection for self-awareness
        if INTROSPECTION_AVAILABLE:
            introspection_lines = ["## AGENT SELF-AWARENESS"]
            for agent_name in ["gemini", "claude", "codex", "grok"]:
                intro = self._format_agent_introspection(agent_name)
                if intro:
                    introspection_lines.append(f"\n### {agent_name.title()}\n{intro}")
            if len(introspection_lines) > 1:
                learning_context += "\n" + "\n".join(introspection_lines) + "\n"

        # Add consensus history context (avoid rehashing settled topics)
        topic_hint = self.initial_proposal[:200] if self.initial_proposal else "aragora improvement"
        consensus_context = self._format_consensus_history(topic_hint)
        if consensus_context:
            learning_context += f"\n{consensus_context}\n"

        # Add codebase evidence (P4: LocalDocsConnector)
        evidence_context = await self._gather_codebase_evidence(topic_hint)
        if evidence_context:
            learning_context += f"\n{evidence_context}\n"

        # Retrieve insights from past debates (P2: InsightStore)
        insight_context = await self._retrieve_relevant_insights(topic_hint)
        if insight_context:
            learning_context += f"\n{insight_context}\n"
            self._log(f"  [insights] Injected past debate learnings")

        # Retrieve similar past debates for historical context
        similar_debates = await self._retrieve_similar_debates(topic_hint)
        if similar_debates:
            learning_context += f"\n{similar_debates}\n"
            self._log(f"  [embeddings] Injected similar debate context")

        # P3-Phase2: Inject pivotal claims (cruxes) from previous debates
        crux_context = self._format_crux_context()
        if crux_context:
            learning_context += f"\n{crux_context}\n"
            self._log(f"  [crux] Injected {len(self._cached_cruxes)} pivotal claims")

        # P5-Phase2: Inject meta-critique observations if previous debate had low quality
        meta_context = self._format_meta_observations()
        if meta_context:
            learning_context += f"\n{meta_context}\n"
            self._log(f"  [meta] Injected reflection (quality was {self._last_meta_quality:.0%})")

        # Add inter-agent dynamics
        relationship_context = self._format_relationship_network()
        if relationship_context:
            learning_context += f"\n{relationship_context}\n"

        # Audit agent calibration before critical debates
        calibration_context = self._audit_agent_calibration()
        if calibration_context:
            learning_context += f"\n{calibration_context}\n"

        # Inject trending topic context from Pulse (informs improvement priorities)
        pulse_context = await self._get_pulse_topic_context(limit=3)
        if pulse_context:
            learning_context += f"\n{pulse_context}\n"

        task = f"""{SAFETY_PREAMBLE}

What single improvement would most benefit aragora RIGHT NOW?

CRITICAL: Read the codebase analysis below carefully. DO NOT propose features that already exist.
Claude and Codex have explored the codebase and documented existing features.
{learning_context}
Consider what would make aragora:
- More INTERESTING (novel, creative, intellectually stimulating)
- More POWERFUL (capable, versatile, effective)
- More VIRAL (shareable, demonstrable, meme-worthy)
- More USEFUL (practical, solves real problems)
{initial_proposal_section}
Each agent should propose ONE specific, implementable feature.
Be concrete: describe what it does, how it works, and why it matters.
After debate, reach consensus on THE SINGLE BEST improvement to implement this cycle.

REMEMBER:
- Propose ADDITIONS, not removals. Build new capabilities, don't simplify existing ones.
- Check the codebase analysis - if a feature is listed there, it ALREADY EXISTS.
- Learn from previous failures shown above - avoid repeating them.

Recent changes:
{recent_changes}"""

        env = Environment(
            task=task,
            context=context_section,
        )

        # Enable asymmetric stances periodically for stress-testing
        asymmetric_enabled = self.cycle_count % 15 == 0
        if asymmetric_enabled:
            self._log(f"  [stances] Devil's advocate mode enabled for stress-testing")

        protocol = DebateProtocol(
            rounds=2,
            consensus="judge",
            judge_selection="elo_ranked",  # Use highest-ELO agent as judge
            proposer_count=4,  # All 4 agents participate
            role_rotation=True,  # Heavy3-inspired cognitive role rotation
            role_rotation_config=RoleRotationConfig(
                enabled=True,
                roles=[
                    CognitiveRole.ANALYST,
                    CognitiveRole.SKEPTIC,
                    CognitiveRole.LATERAL_THINKER,
                    CognitiveRole.ADVOCATE,
                ],
                synthesizer_final_round=True,
            ),
            asymmetric_stances=asymmetric_enabled,
            rotate_stances=asymmetric_enabled,
            audience_injection="summary",  # Enable user suggestions in prompts
            enable_research=True,  # Enable web research for evidence gathering
        )

        # Select and apply debate template based on task type (P7: DebateTemplates)
        template = self._select_debate_template(topic_hint)
        if template:
            self._apply_template_to_protocol(protocol, template)

        # Phase 5: Select optimal debate team (P14: AgentSelector)
        debate_team = self._select_debate_team(topic_hint)

        # Phase 10: Assign hybrid roles for specialized debate contributions
        hybrid_roles = {}
        if self.agent_selector and hasattr(self.agent_selector, 'assign_hybrid_roles'):
            try:
                hybrid_roles = self.agent_selector.assign_hybrid_roles(debate_team, "debate")
                if hybrid_roles:
                    self._log(f"  [selector] Assigned roles: {list(hybrid_roles.values())}")
                    # Inject role into each agent's system prompt
                    for agent in debate_team:
                        if agent.name in hybrid_roles:
                            role = hybrid_roles[agent.name]
                            role_prompt = f"\n\nYour role in this debate: {role}"
                            if hasattr(agent, 'system_prompt') and agent.system_prompt:
                                agent.system_prompt += role_prompt
            except Exception as e:
                self._log(f"  [selector] Role assignment failed: {e}")

        # Probe agents for reliability weights before debate
        agent_weights = {}
        if self.nomic_integration:
            try:
                self._log("  [integration] Probing debate agents for reliability...")
                agent_weights = await self.nomic_integration.probe_agents(
                    debate_team,
                    probe_count=2,  # Quick probe to minimize latency
                    min_weight=0.5,
                )
                reliable_count = sum(1 for w in agent_weights.values() if w >= 0.7)
                self._log(f"  [integration] Agent weights: {reliable_count}/{len(debate_team)} reliable")
                # Store for ELO confidence weighting (P13: probe → ELO feedback)
                self._last_probe_weights = agent_weights
            except Exception as e:
                self._log(f"  [integration] Probing failed: {e}")
                self._last_probe_weights = {}

        # Phase 9: Inject grounded personas into agent system prompts
        self._inject_grounded_personas(debate_team)
        self._log_grounded_persona_stats()

        # Phase 7: Start debate trace for audit logging (P25: DebateTracer)
        debate_id = f"debate-cycle-{self.cycle_count}"
        self._start_debate_trace(debate_id, topic_hint or task[:100], debate_team)

        # Phase 8: Inject similar past critiques as context (P27: SemanticRetriever)
        semantic_context = await self._inject_similar_context(topic_hint or task[:200])
        if semantic_context:
            env.context = (env.context or "") + f"\n\n{semantic_context}"

        arena = Arena(
            env,
            debate_team,
            protocol,
            memory=self.critique_store,
            debate_embeddings=self.debate_embeddings,
            insight_store=self.insight_store,
            agent_weights=agent_weights,  # Use probed reliability weights for vote weighting
            position_tracker=self.position_tracker,
            position_ledger=self.position_ledger,
            calibration_tracker=self.calibration_tracker,
            elo_system=self.elo_system,
            event_emitter=self.stream_emitter, loop_id=self.loop_id,
            event_hooks=self._create_arena_hooks("debate"),  # Enable real-time streaming
            persona_manager=self.persona_manager,
            relationship_tracker=self.relationship_tracker,
            moment_detector=self.moment_detector,
            continuum_memory=self.continuum,
        )

        # P1-Phase2: Use graph-based debate for complex multi-agent reasoning if available
        # DebateGraph provides DAG-based parallel argument exploration and automatic convergence
        result = None
        if self.graph_debate_enabled and DEBATE_GRAPH_AVAILABLE and len(debate_team) >= 3:
            graph_result = await self._run_graph_debate(task, debate_team)
            if graph_result:
                self._log("  [graph] Using graph-based debate result")
                result = graph_result

        # Fall back to traditional arena if graph debate unavailable or failed
        if result is None:
            result = await self._run_arena_with_logging(arena, "debate")

        # Phase 10: Attempt forking if deadlock detected (P30: DebateForker)
        # Fork into parallel branches when no consensus reached - explore alternatives
        if self.fork_debate_enabled and not result.consensus_reached and DEBATE_FORKER_AVAILABLE:
            try:
                messages = result.messages if hasattr(result, 'messages') else []
                rounds = result.rounds_used if hasattr(result, 'rounds_used') else 3

                fork_decision = self._check_should_fork(messages, rounds, debate_team)
                if fork_decision:
                    self._log(f"  [fork] Deadlock detected - forking into {len(fork_decision.branches)} branches")
                    base_context = f"Original debate topic: {task}\n\nPrior context: {topic_hint or ''}"
                    merge_result = await self._run_forked_debate(fork_decision, base_context)

                    if merge_result and hasattr(merge_result, 'winning_answer'):
                        self._log(f"  [fork] Merged branches - selected answer from '{merge_result.winning_branch}'")
                        result.final_answer = merge_result.winning_answer
                        result.consensus_reached = True
                        result.confidence = getattr(merge_result, 'confidence', 0.75)
            except Exception as e:
                self._log(f"  [fork] Forking failed: {e}")

        # Update agent reputation based on debate outcome
        if self.critique_store and result.consensus_reached:
            winning_proposal = result.final_answer if result.final_answer else ""
            for agent in debate_team:
                # Check if this agent's proposal was selected
                proposal_accepted = agent.name.lower() in winning_proposal.lower()
                self.critique_store.update_reputation(
                    agent.name,
                    proposal_made=True,
                    proposal_accepted=proposal_accepted,
                )

        # Store consensus for future reference (P1: ConsensusMemory)
        await self._store_debate_consensus(result, topic_hint)

        # Record calibration data from debate predictions (P10: CalibrationTracker)
        detected_domain = self._detect_domain(topic_hint) if topic_hint else "general"
        self._record_calibration_from_debate(result, debate_team, domain=detected_domain)

        # Record suggestion feedback for audience learning (P10: SuggestionFeedbackTracker)
        debate_id = getattr(result, 'debate_id', f"cycle-{self.cycle_count}-debate")
        self._record_suggestion_feedback(result, debate_id)

        # Extract and store insights for pattern learning (P2: InsightExtractor)
        await self._extract_and_store_insights(result)

        # Record agent memories for cumulative learning (P3: MemoryStream)
        await self._record_agent_memories(result, topic_hint)

        # Phase 4: Record persona performance (P8: PersonaManager)
        self._record_persona_performance(result, topic_hint)

        # Phase 4: Extract and store winning patterns (P9: PromptEvolver)
        await self._extract_and_store_patterns(result)

        # Phase 4: Update prompt performance metrics (P9: PromptEvolver)
        for agent in debate_team:
            self._update_prompt_performance(agent.name, result)

        # Phase 5: Analyze debate process and store recommendations (P12: MetaCritiqueAnalyzer)
        meta_critique = self._analyze_debate_process(result)
        self._store_meta_recommendations(meta_critique)

        # Phase 5: Update ELO ratings (P13: EloSystem)
        self._record_elo_match(result, topic_hint)

        # Phase 9: Record domain calibration for grounded personas
        if self.elo_system and result and ELO_AVAILABLE:
            try:
                domain = self._detect_domain(topic_hint or task)
                for agent in debate_team:
                    agent_correct = self._agent_in_consensus(agent.name, result)
                    confidence = result.confidence if hasattr(result, 'confidence') else 0.7
                    self.elo_system.record_domain_prediction(
                        agent_name=agent.name,
                        domain=domain,
                        confidence=confidence,
                        correct=agent_correct,
                    )
                self._log(f"  [calibration] Recorded domain predictions for {len(debate_team)} agents in '{domain}'")
            except Exception as e:
                self._log(f"  [calibration] Recording failed: {e}")

        # Phase 9: Record positions to ledger for grounded personas
        if self.position_ledger and hasattr(result, 'messages'):
            try:
                debate_id = f"cycle-{self.cycle_count}"
                for msg in result.messages:
                    if hasattr(msg, 'agent') and hasattr(msg, 'content') and msg.content:
                        self.position_ledger.record_position(
                            agent_name=msg.agent,
                            claim=msg.content[:1000],
                            confidence=0.7,
                            debate_id=debate_id,
                            round_num=getattr(msg, 'round', 0),
                        )
                self._log(f"  [grounded] Recorded {len(result.messages)} positions to ledger")
            except Exception as e:
                self._log(f"  [grounded] Position recording failed: {e}")

        # Phase 9: Resolve positions based on debate outcome
        if self.position_ledger and result:
            try:
                debate_id = f"cycle-{self.cycle_count}"
                outcome = "correct" if result.consensus_reached else "unresolved"
                for agent in debate_team:
                    positions = self.position_ledger.get_agent_positions(
                        agent_name=agent.name,
                        limit=10,  # Get recent positions
                    )
                    for pos in positions:
                        agent_outcome = "correct" if self._agent_in_consensus(agent.name, result) else "incorrect"
                        self.position_ledger.resolve_position(pos.id, agent_outcome)
                self._log(f"  [grounded] Resolved positions for debate {debate_id}")
            except Exception as e:
                self._log(f"  [grounded] Position resolution failed: {e}")

        # Phase 9: Update agent relationships for grounded personas
        if self.relationship_tracker and result:
            try:
                participants = [a.name for a in debate_team]
                winner = None
                if hasattr(result, 'votes') and result.votes:
                    vote_tally = {}
                    for v in result.votes:
                        if hasattr(v, 'choice'):
                            vote_tally[v.choice] = vote_tally.get(v.choice, 0) + 1
                    if vote_tally:
                        winner = max(vote_tally.items(), key=lambda x: x[1])[0]

                # Extract position changes for influence tracking
                position_changes = self._extract_position_changes(result)
                if position_changes:
                    self._log(f"  [grounded] Detected position changes: {position_changes}")

                # Extract critiques for relationship tracking
                critiques_data = []
                if hasattr(result, 'messages'):
                    for msg in result.messages:
                        if hasattr(msg, 'critique') and msg.critique:
                            critiques_data.append({
                                'critic': getattr(msg, 'agent', 'unknown'),
                                'target': getattr(msg.critique, 'target', 'unknown'),
                                'accepted': getattr(msg.critique, 'accepted', False),
                            })

                # Convert votes list to dict[agent -> choice] for relationship tracker
                votes_dict = {}
                if hasattr(result, 'votes') and result.votes:
                    for v in result.votes:
                        if hasattr(v, 'agent') and hasattr(v, 'choice'):
                            votes_dict[v.agent] = v.choice

                self.relationship_tracker.update_from_debate(
                    debate_id=f"cycle-{self.cycle_count}",
                    participants=participants,
                    winner=winner,
                    votes=votes_dict,
                    critiques=critiques_data,
                    position_changes=position_changes,
                )
                self._log(f"  [grounded] Updated relationships for {len(participants)} agents")

                # Also update ELO system relationship tracking for detailed stats
                if self.elo_system and ELO_AVAILABLE:
                    for i, agent_a in enumerate(participants):
                        for agent_b in participants[i+1:]:
                            self.elo_system.update_relationship(
                                agent_a=agent_a,
                                agent_b=agent_b,
                                debate_increment=1,
                                agreement_increment=1 if winner else 0,
                                a_win=1 if winner == agent_a else 0,
                                b_win=1 if winner == agent_b else 0,
                            )
                    self._log(f"  [elo] Updated relationship stats for {len(participants)} agent pairs")
            except Exception as e:
                self._log(f"  [grounded] Relationship update failed: {e}")

        # Phase 9: Update AgentSelector with debate results
        if self.agent_selector and result:
            try:
                from aragora.routing.selection import TeamComposition, AgentProfile
                # Create minimal TeamComposition for update
                agent_profiles = []
                for a in debate_team:
                    # Get or create agent profile from selector pool
                    if hasattr(self.agent_selector, 'agent_pool') and a.name in self.agent_selector.agent_pool:
                        agent_profiles.append(self.agent_selector.agent_pool[a.name])
                    else:
                        # Create minimal profile
                        agent_profiles.append(AgentProfile(
                            name=a.name,
                            provider=getattr(a, 'provider', 'unknown'),
                        ))
                team = TeamComposition(
                    team_id=f"cycle-{self.cycle_count}",
                    task_id=topic_hint or task[:100],
                    agents=agent_profiles,
                    roles={a.name: "debater" for a in debate_team},
                    expected_quality=0.7,
                    expected_cost=0.0,
                    diversity_score=0.5,
                    rationale="nomic debate team",
                )
                self.agent_selector.update_from_result(team=team, result=result)
                self._log(f"  [selector] Updated agent profiles from debate outcome")
            except Exception as e:
                self._log(f"  [selector] Update failed: {e}")

        # Phase 9: Store successful critique patterns for learning
        if self.critique_store and result.consensus_reached and hasattr(result, 'critiques') and result.critiques:
            try:
                successful_fix = result.final_answer[:500] if result.final_answer else ""
                for critique in result.critiques:
                    self.critique_store.store_pattern(
                        critique=critique,
                        successful_fix=successful_fix,
                    )
                self._log(f"  [patterns] Stored {len(result.critiques)} critique patterns")
            except Exception as e:
                self._log(f"  [patterns] Storage failed: {e}")

        # Phase 10: Update prediction outcomes for Titans/MIRAS calibration
        if self.critique_store and hasattr(result, 'critiques') and result.critiques:
            try:
                import sqlite3
                updated_count = 0
                for critique in result.critiques:
                    # Determine actual usefulness based on consensus and suggestion incorporation
                    actual_usefulness = 0.5 if result.consensus_reached else 0.2
                    if result.consensus_reached and result.final_answer:
                        for suggestion in getattr(critique, 'suggestions', []):
                            if suggestion and suggestion.lower()[:50] in result.final_answer.lower():
                                actual_usefulness = 0.8
                                break

                    agent_name = getattr(critique, 'agent', None)
                    if agent_name and hasattr(self.critique_store, 'db_path'):
                        conn = sqlite3.connect(self.critique_store.db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT id FROM critiques WHERE agent = ? ORDER BY id DESC LIMIT 1",
                            (agent_name,)
                        )
                        row = cursor.fetchone()
                        conn.close()

                        if row:
                            self.critique_store.update_prediction_outcome(
                                critique_id=row[0],
                                actual_usefulness=actual_usefulness,
                                agent_name=agent_name,
                            )
                            updated_count += 1
                if updated_count > 0:
                    self._log(f"  [calibration] Updated prediction outcomes for {updated_count} critiques")
            except Exception as e:
                self._log(f"  [calibration] Prediction update failed: {e}")

        # Phase 5: Track risks from low-consensus debates (P15: RiskRegister)
        self._track_debate_risks(result, topic_hint)

        # Phase 6: Extract structured claims from debate (P16: ClaimsKernel)
        self._extract_claims_from_debate(result)
        self._analyze_claim_structure()

        # Phase 6: Build belief network and identify cruxes (P18: BeliefNetwork)
        self._build_belief_network()
        self._identify_debate_cruxes()

        # Phase 6: Record evidence provenance (P17: ProvenanceManager)
        if result.final_answer:
            self._record_evidence_provenance(result.final_answer, "agent", "debate-consensus")

        # Citation Grounding: Extract citation-worthy claims from debate result
        # Also build provenance chain from claims to evidence
        debate_evidence_ids = []
        if result.final_answer and self.citation_extractor:
            citation_needs = self.citation_extractor.identify_citation_needs(result.final_answer)
            if citation_needs:
                high_priority = [c for c in citation_needs if c.get("priority") == "high"]
                if high_priority:
                    self._log(f"  [citations] Found {len(high_priority)} high-priority claims needing citations")
                    for need in high_priority[:3]:
                        self._log(f"    - {need['claim'][:80]}...")
                        # Try to find existing citations from store
                        if self.citation_store:
                            existing = self.citation_store.find_for_claim(need['claim'], limit=2)
                            if existing:
                                self._log(f"      Found {len(existing)} potential citations")

                # Link all claims to provenance chain
                debate_id = f"debate-{self.cycle_count}-{result.id[:8] if hasattr(result, 'id') else 'unknown'}"
                debate_evidence_ids = self._link_claims_to_evidence(citation_needs, debate_id)

        # Store evidence IDs on result for phase chaining
        if debate_evidence_ids:
            result.provenance_evidence_ids = debate_evidence_ids

        # Phase 6: Create verification proofs for code claims (P19: ProofExecutor)
        await self._create_verification_proofs(result)

        # Phase 7: Generate reliability report and filter low-reliability claims (P24: ReliabilityScorer)
        reliability_report = self._generate_reliability_report()
        if reliability_report and self.claims_kernel:
            try:
                claims_results = reliability_report.get("claims", {})
                low_reliability_ids = [
                    cid for cid, data in claims_results.items()
                    if data.get("level") in ("VERY_LOW", "SPECULATIVE")
                ]
                if low_reliability_ids:
                    self._log(f"  [reliability] Filtering {len(low_reliability_ids)} low-reliability claims")
                    # Mark low-reliability claims for exclusion from design phase
                    result.low_reliability_claim_ids = low_reliability_ids
            except Exception as e:
                self._log(f"  [reliability] Filtering error: {e}")

        # Phase 7: Finalize debate trace (P25: DebateTracer)
        self._finalize_debate_trace(result)

        # Phase 10: Analyze team selection patterns (every 5 cycles)
        if self.agent_selector and self.cycle_count % 5 == 0:
            try:
                best_teams = self.agent_selector.get_best_team_combinations(min_debates=2)
                if best_teams:
                    self._log(f"  [selector] Best team combinations:")
                    for team in best_teams[:3]:
                        self._log(f"    {team['agents']}: {team['success_rate']:.0%} ({team['wins']}/{team['total_debates']})")
            except Exception as e:
                self._log(f"  [selector] Team analysis failed: {e}")

        # Phase 7: Create checkpoint after debate (P22: CheckpointManager)
        await self._create_debate_checkpoint(
            debate_id=result.id if hasattr(result, 'id') else "unknown",
            task=topic_hint or task,
            round_num=result.rounds_used if hasattr(result, 'rounds_used') else 0,
            messages=result.messages if hasattr(result, 'messages') else [],
            agents=debate_team,
            consensus={"reached": result.consensus_reached, "confidence": result.confidence}
        )

        # Phase 8: Store critique embeddings for future retrieval (P27: SemanticRetriever)
        if result.critiques:
            for critique in result.critiques[:5]:  # Limit to 5 to avoid too many API calls
                if hasattr(critique, 'reasoning'):
                    await self._store_critique_embedding(
                        critique_id=f"critique-{result.id[:8]}-{critique.agent}",
                        critique_text=critique.reasoning
                    )

        # Phase 8: Evolve personas based on debate outcome (P26: PersonaLaboratory)
        # Run every 2 cycles for faster agent evolution
        if self.cycle_count % 2 == 0:
            await self._evolve_personas_post_cycle()

        # Handle deadlocks via counterfactual branching (P5: CounterfactualOrchestrator)
        # This is a fallback when NomicIntegration isn't available
        if not self.nomic_integration and not result.consensus_reached:
            result = await self._handle_debate_deadlock(result, arena, topic_hint)

        # NomicIntegration: Unified post-debate analysis
        belief_analysis = None
        conditional_consensus = None
        if self.nomic_integration:
            try:
                # Use unified post-debate analysis for belief analysis + deadlock resolution
                post_analysis = await self.nomic_integration.full_post_debate_analysis(
                    result=result,
                    arena=arena,  # Enable counterfactual branch execution
                    claims_kernel=self.claims_kernel,
                    changed_files=None,  # Staleness checked after implementation
                )
                belief_analysis = post_analysis.get("belief")
                conditional_consensus = post_analysis.get("conditional")
                summary = post_analysis.get("summary", {})

                # Log belief network analysis results
                self._log(f"  [belief] Network: {summary.get('contested_count', 0)} contested, "
                          f"{summary.get('crux_count', 0)} crux claims, "
                          f"converged={belief_analysis.convergence_achieved if belief_analysis else False}")
                if summary.get("has_deadlock"):
                    self._log(f"  [belief] Deadlock detected - counterfactual resolution attempted")
                    if conditional_consensus:
                        self._log(f"  [belief] Resolved with conditional consensus")
                        # Apply conditional consensus to result if no consensus was reached
                        if not result.consensus_reached and hasattr(conditional_consensus, 'synthesized_answer'):
                            result.final_answer = conditional_consensus.synthesized_answer
                            result.consensus_reached = True
                            result.confidence = getattr(conditional_consensus, 'confidence', 0.7)
                            self._log(f"  [belief] Applied conditional consensus as final answer")

                # Checkpoint the debate phase
                await self.nomic_integration.checkpoint(
                    phase="debate",
                    state={"result": result.final_answer, "confidence": result.confidence},
                    cycle=self.cycle_count,
                )
            except Exception as e:
                self._log(f"  [integration] Post-debate analysis failed: {e}")

        # Phase 9: Comprehensive flip detection (P9: FlipDetector)
        # Run every 3 cycles for efficiency (use cached FlipDetector instance)
        if self.flip_detector and self.cycle_count % 3 == 0:
            try:
                # Detect flips for all debate participants
                total_flips_detected = 0
                all_flips = []
                consistency_warnings = []
                for agent in debate_team:
                    flips = self.flip_detector.detect_flips_for_agent(agent.name, lookback_positions=20)
                    total_flips_detected += len(flips)
                    all_flips.extend(flips)

                    # Check consistency and flag concerning agents
                    consistency = self.flip_detector.get_agent_consistency(agent.name)
                    if consistency.contradictions >= 2 or consistency.consistency_score < 0.6:
                        consistency_warnings.append(
                            f"{agent.name}: {consistency.contradictions} contradictions, "
                            f"consistency={consistency.consistency_score:.1%}"
                        )

                if total_flips_detected > 0:
                    self._log(f"  [flip] Detected {total_flips_detected} new position flips")

                    # Emit flip events to WebSocket stream for real-time UI updates
                    if self.stream_emitter:
                        for flip in all_flips:
                            self._stream_emit(
                                "flip_detected",
                                flip.agent_name,
                                self.cycle_count,
                                True,
                                0.0,
                                {
                                    "flip_type": flip.flip_type,
                                    "original_claim": flip.original_claim[:100] if flip.original_claim else "",
                                    "new_claim": flip.new_claim[:100] if flip.new_claim else "",
                                    "similarity": flip.similarity_score,
                                    "domain": flip.domain,
                                }
                            )

                if consistency_warnings:
                    for warning in consistency_warnings:
                        self._log(f"  [flip] ⚠️ Low consistency: {warning}")

                # Get and log summary every 10 cycles
                if self.cycle_count % 10 == 0:
                    summary = self.flip_detector.get_flip_summary()
                    if summary.get("total_flips", 0) > 0:
                        self._log(f"  [flip] Summary: {summary.get('total_flips', 0)} total flips, "
                                  f"{summary.get('by_type', {})} by type")
            except Exception as e:
                self._log(f"  [flip] Detection error: {e}")

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "debate", self.cycle_count, result.consensus_reached,
            phase_duration, {"confidence": result.confidence}
        )

        # Generate debate summary for broadcast (every cycle for better documentation)
        broadcast_summary = None
        if self.summary_generator:
            try:
                broadcast_summary = self.summary_generator.generate_summary(
                    debate_result=result,
                    agents=[a.name for a in agents] if agents else []
                )
                if broadcast_summary:
                    self._log(f"  [broadcast] Generated summary ({len(broadcast_summary)} chars)")
            except Exception as e:
                self._log(f"  [broadcast] Summary generation error: {e}")

        return {
            "phase": "debate",
            "final_answer": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "duration": result.duration_seconds,
            "belief_analysis": belief_analysis.to_dict() if belief_analysis else None,
            "conditional_consensus": conditional_consensus,
            "broadcast_summary": broadcast_summary,
        }

    async def phase_design(self, improvement: str, belief_analysis: dict = None) -> dict:
        """Phase 2: All agents design the implementation together.

        Args:
            improvement: The improvement proposal from debate phase
            belief_analysis: Optional belief analysis from debate (contested/crux claims)
        """
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 2: IMPLEMENTATION DESIGN")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "design", self.cycle_count, {"agents": 4})

        # Record phase change
        self._record_replay_event("phase", "system", "design")

        # Check if this topic warrants deep audit
        should_deep_audit, audit_reason = self._should_use_deep_audit(improvement, phase="design")
        if should_deep_audit:
            self._log(f"  [deep-audit] {audit_reason}")
            deep_audit_result = await self._run_deep_audit_for_design(improvement)
            if deep_audit_result and not deep_audit_result.get("approved", True):
                self._log("  [deep-audit] Design rejected by deep audit - returning issues for rework")
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit("on_phase_end", "design", self.cycle_count, False, phase_duration)
                return {
                    "success": False,
                    "phase": "design",
                    "design": None,
                    "rejected_by_deep_audit": True,
                    "issues": deep_audit_result.get("unanimous_issues", []),
                }

        # Gather learning context for design (Titans/MIRAS + ContinuumMemory)
        successful_patterns = self._format_successful_patterns(limit=3)
        failure_patterns = self._format_failure_patterns(limit=3)
        continuum_patterns = self._format_continuum_patterns(limit=3)

        design_learning = ""
        if successful_patterns:
            design_learning += f"\n{successful_patterns}\n"
        if failure_patterns:
            design_learning += f"\n{failure_patterns}\n"
        if continuum_patterns:
            design_learning += f"\n{continuum_patterns}\n"

        # Add belief analysis context from debate phase (contested/crux claims)
        belief_context = ""
        if belief_analysis:
            contested_count = belief_analysis.get("contested_count", 0)
            crux_count = belief_analysis.get("crux_count", 0)
            if contested_count > 0 or crux_count > 0:
                belief_context = "\n## UNCERTAINTY FROM DEBATE\n"
                belief_context += f"The debate identified {contested_count} contested claims "
                belief_context += f"and {crux_count} crux (high-impact disputed) claims.\n"
                if crux_count > 0:
                    belief_context += "Your design MUST address these disputed points explicitly.\n"
                posteriors = belief_analysis.get("posteriors", {})
                if posteriors:
                    belief_context += "Key uncertainty areas:\n"
                    for claim_id, dist in list(posteriors.items())[:3]:
                        if isinstance(dist, dict) and dist.get("entropy", 0) > 0.5:
                            belief_context += f"  - {claim_id}: entropy={dist.get('entropy', 0):.2f}\n"
            if belief_analysis.get("convergence_achieved"):
                design_learning += "\n[Belief network converged - good consensus foundation]\n"
            else:
                design_learning += "\n[Belief network did NOT converge - proceed with caution]\n"
        if belief_context:
            design_learning += belief_context
            self._log(f"  [belief] Injected {len(belief_context)} chars of belief context into design")

        # Build citation guidance if available
        citation_guidance = ""
        if self.citation_extractor:
            citation_guidance = """
6. EVIDENCE: Support architectural claims with evidence (cite prior debates, docs, or research)
   - When making claims about best practices, cite the source
   - When referencing existing patterns, cite the file path
   - Flag any assumptions that need verification
"""

        # Enhanced design prompt with clearer guidance for better consensus
        design_prompt = f"""{SAFETY_PREAMBLE}

## DESIGN TASK
Create a detailed implementation design for this improvement:

{improvement}

## CONTEXT FROM PRIOR LEARNING
{design_learning}

## REQUIRED DESIGN SECTIONS

### 1. FILE CHANGES (Required)
List EVERY file to create or modify. Be specific:
- `aragora/path/file.py` - Create new | Modify existing
- Estimated lines of code per file
- NEVER delete or modify protected files: {PROTECTED_FILES}

### 2. API DESIGN (Required)
Define the public interface:
```python
class ClassName:
    def method_name(self, param: Type) -> ReturnType:
        '''Docstring explaining purpose.'''
        ...
```
- EXTEND existing APIs, don't break them
- Use existing patterns from aragora codebase

### 3. INTEGRATION POINTS (Required)
How does this connect to existing modules?
- Which existing classes/functions does it use?
- Which existing classes/functions call it?
- Any new dependencies needed?

### 4. TEST PLAN (Required)
Concrete test cases:
- Unit tests: `test_feature_basic()`, `test_feature_edge_case()`
- Integration tests if needed
- How to verify existing features still work

### 5. EXAMPLE USAGE (Required)
Working code snippet showing the feature in action:
```python
# Example usage
result = new_feature.do_something()
```

## DESIGN QUALITY CRITERIA
Your design will be evaluated on:
- **Feasibility**: Can this be implemented in ~500 lines of code?
- **Completeness**: Are all required sections filled out?
- **Specificity**: Could an engineer implement this without asking questions?
- **Safety**: Does it preserve all existing functionality?

## REQUIRED: Viability Checklist
Your design MUST include ALL of the following to be considered viable:
- [ ] At least 3 specific file changes with estimated line counts
- [ ] At least 2 function/class signatures with parameters
- [ ] At least 1 integration point showing how components connect
- [ ] 1 concrete example of how to use the feature

Designs missing any of these will be automatically rejected as non-viable.

## IMPORTANT
- Focus on MINIMAL viable implementation
- Avoid over-engineering or unnecessary abstractions
- Prefer simple, direct solutions over clever ones
{citation_guidance}"""

        env = Environment(
            task=design_prompt,
            context=f"Working directory: {self.aragora_path}\n\nProtected files (NEVER delete): {PROTECTED_FILES}",
        )

        # HYBRID MODEL ARCHITECTURE: Gemini leads design, others critique
        # Order: Gemini first as design lead, then critics (Claude, Codex, Grok)
        design_agents = [self.gemini, self.claude, self.codex, self.grok, self.deepseek]

        protocol = DebateProtocol(
            rounds=2,  # Allow critique and refinement round
            consensus="judge",
            judge_selection="elo_ranked",  # Use highest-ELO agent as judge
            proposer_count=1,  # Gemini as primary design proposer
            early_stopping=True,  # Enable early exit on consensus
            early_stop_threshold=0.66,  # 66% agreement triggers early stop
            min_rounds_before_early_stop=1,  # At least 1 round before stopping
            role_rotation=True,  # Heavy3-inspired cognitive role rotation
            role_rotation_config=RoleRotationConfig(
                enabled=True,
                roles=[
                    CognitiveRole.ANALYST,  # Claude: architecture analysis
                    CognitiveRole.SKEPTIC,  # Codex: implementation concerns
                    CognitiveRole.DEVIL_ADVOCATE,  # Grok: lateral critique
                ],
                synthesizer_final_round=True,  # Gemini synthesizes final design
            ),
            audience_injection="summary",  # Enable user suggestions in prompts
            enable_research=True,  # Enable web research for evidence gathering
        )

        self._log("  [hybrid] Gemini as design lead, others as critics")

        # NomicIntegration: Probe agents for reliability weights
        agent_weights = {}
        if self.nomic_integration:
            try:
                self._log("  [integration] Probing agents for reliability...")
                agent_weights = await self.nomic_integration.probe_agents(
                    design_agents,
                    probe_count=2,  # Quick probe
                    min_weight=0.5,
                )
                reliable_count = sum(1 for w in agent_weights.values() if w >= 0.7)
                self._log(f"  [integration] Agent weights: {agent_weights} ({reliable_count}/4 reliable)")
            except Exception as e:
                self._log(f"  [integration] Probing failed: {e}")

        # Phase 9: Inject grounded personas into design agents
        self._inject_grounded_personas(design_agents)

        # All 4 agents participate in design (with reliability weights)
        arena = Arena(
            env,
            design_agents,
            protocol,
            memory=self.critique_store,
            debate_embeddings=self.debate_embeddings,
            insight_store=self.insight_store,
            agent_weights=agent_weights,
            position_tracker=self.position_tracker,
            position_ledger=self.position_ledger,
            calibration_tracker=self.calibration_tracker,
            elo_system=self.elo_system,
            event_emitter=self.stream_emitter, loop_id=self.loop_id,
            event_hooks=self._create_arena_hooks("design"),  # Enable real-time streaming
            persona_manager=self.persona_manager,
            relationship_tracker=self.relationship_tracker,
            moment_detector=self.moment_detector,
            continuum_memory=self.continuum,
        )
        result = await self._run_arena_with_logging(arena, "design")

        # === Fast-Track Judge Arbitration on Low Time Budget ===
        # When time is critical and no consensus, skip complex deadlock resolution
        # and immediately invoke judge arbitration
        if not result.consensus_reached:
            elapsed = (datetime.now() - phase_start).total_seconds()
            remaining = self.max_cycle_seconds - elapsed
            if remaining < 600:  # Less than 10 min left in cycle
                self._log(f"  [fast-track] Time budget critical ({remaining:.0f}s left) - invoking judge immediately")
                # Extract individual proposals for arbitration
                fast_proposals = {}
                for msg in result.messages:
                    if msg.role == "proposer" and msg.content:
                        fast_proposals[msg.agent] = msg.content
                if len(fast_proposals) >= 2:
                    arbitrated = await self._arbitrate_design(fast_proposals, improvement)
                    if arbitrated:
                        result.final_answer = arbitrated
                        result.consensus_reached = True
                        result.confidence = 0.7  # Moderate confidence for arbitrated design
                        self._log(f"  [fast-track] Judge selected design ({len(arbitrated)} chars)")

        # === Deadlock Resolution via Counterfactual Branching ===
        # When design consensus fails, use belief analysis to identify crux claims
        # and fork the debate to explore different assumptions
        conditional_design = None
        if not result.consensus_reached and self.nomic_integration:
            try:
                self._log("  [deadlock] No design consensus - attempting counterfactual resolution...")

                # Run full post-debate analysis which includes deadlock resolution
                post_analysis = await self.nomic_integration.full_post_debate_analysis(
                    result=result,
                    arena=arena,  # Pass arena for branch execution
                    claims_kernel=None,
                    changed_files=None,
                )

                conditional = post_analysis.get("conditional")
                summary = post_analysis.get("summary", {})

                if conditional:
                    self._log(f"  [deadlock] Resolved with conditional consensus")
                    # Synthesize a design from the conditional consensus
                    if hasattr(conditional, 'synthesized_answer') and conditional.synthesized_answer:
                        result.final_answer = conditional.synthesized_answer
                        result.consensus_reached = True
                        result.confidence = getattr(conditional, 'confidence', 0.7)
                        conditional_design = conditional.synthesized_answer
                        self._log(f"  [deadlock] Applied conditional design ({len(conditional_design)} chars)")
                    elif hasattr(conditional, 'if_true_conclusion') and hasattr(conditional, 'if_false_conclusion'):
                        # Build conditional design from branches
                        pivot = getattr(conditional, 'pivot_claim', None)
                        pivot_text = pivot.text if pivot and hasattr(pivot, 'text') else "the disputed assumption"
                        conditional_design = (
                            f"## CONDITIONAL DESIGN\n\n"
                            f"**Key Decision Point:** {pivot_text}\n\n"
                            f"### If True:\n{conditional.if_true_conclusion}\n\n"
                            f"### If False:\n{conditional.if_false_conclusion}\n\n"
                            f"**Recommended Path:** "
                            f"{'True assumption' if conditional.preferred_world else 'False assumption'} "
                            f"(Reason: {getattr(conditional, 'preference_reason', 'higher confidence')})"
                        )
                        result.final_answer = conditional_design
                        result.consensus_reached = True
                        result.confidence = max(
                            conditional.if_true_confidence or 0.5,
                            conditional.if_false_confidence or 0.5
                        )
                        self._log(f"  [deadlock] Built conditional design from branches")
                elif summary.get("has_deadlock"):
                    self._log(f"  [deadlock] Deadlock detected but no resolution found")
                    self._log(f"  [deadlock] Contested: {summary.get('contested_count', 0)}, Crux: {summary.get('crux_count', 0)}")

            except Exception as e:
                self._log(f"  [deadlock] Resolution failed: {e}")

        # Extract individual proposals from messages for fallback selection
        individual_proposals = {}
        for msg in result.messages:
            if msg.role == "proposer" and msg.content:
                individual_proposals[msg.agent] = msg.content

        # Extract vote counts for fallback selection
        vote_counts = {}
        for vote in result.votes:
            choice = vote.choice
            vote_counts[choice] = vote_counts.get(choice, 0) + 1

        # NomicIntegration: Checkpoint the design phase
        if self.nomic_integration:
            try:
                await self.nomic_integration.checkpoint(
                    phase="design",
                    state={"design": result.final_answer, "agent_weights": agent_weights},
                    cycle=self.cycle_count,
                )
            except Exception as e:
                self._log(f"  [integration] Checkpoint failed: {e}")

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "design", self.cycle_count, result.consensus_reached,
            phase_duration, {}
        )

        return {
            "phase": "design",
            "design": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "agent_weights": agent_weights,
            "individual_proposals": individual_proposals,
            "vote_counts": vote_counts,
            "confidence": result.confidence,
            "votes": [(v.agent, v.choice, v.confidence) for v in result.votes],
            "conditional_design": conditional_design is not None,  # Flag for conditional resolution
        }

    def _git_stash_create(self) -> Optional[str]:
        """Create a git stash for transactional safety."""
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                return None

            result = subprocess.run(
                ["git", "stash", "push", "-m", "nomic-implement-backup"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                ref_result = subprocess.run(
                    ["git", "stash", "list", "-1", "--format=%H"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                return ref_result.stdout.strip() or "stash@{0}"
        except Exception as e:
            self._log(f"Warning: Could not create stash: {e}")
        return None

    def _git_stash_pop(self, stash_ref: Optional[str]) -> None:
        """Pop a stash to restore previous state."""
        if not stash_ref:
            return
        try:
            subprocess.run(
                ["git", "checkout", "."],
                cwd=self.aragora_path,
                capture_output=True,
            )
            subprocess.run(
                ["git", "stash", "pop"],
                cwd=self.aragora_path,
                capture_output=True,
            )
        except Exception as e:
            self._log(f"Warning: Could not pop stash: {e}")

    def _get_git_diff(self) -> str:
        """Get current git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except Exception:
            return ""

    def _get_git_changed_files(self) -> list[str]:
        """Get list of changed files from git."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
            return files
        except Exception:
            return []

    async def _call_agent_with_retry(
        self,
        agent,
        prompt: str,
        context: list = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> str:
        """
        Call an agent with exponential backoff retry on failures.

        Args:
            agent: The agent to call
            prompt: The prompt to send
            context: Message context (optional)
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for timeout on each retry

        Returns:
            Agent response string, or error message on complete failure
        """
        context = context or []
        timeout = agent.timeout
        last_error = None

        for attempt in range(max_retries):
            try:
                self._log(f"    {agent.name} attempt {attempt + 1}/{max_retries}...", agent=agent.name)

                # Use asyncio.wait_for with increasing timeout
                result = await asyncio.wait_for(
                    agent.generate(prompt, context),
                    timeout=timeout
                )
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                if attempt < max_retries - 1:
                    wait_time = min(120, 5 * (backoff_factor ** attempt))
                    self._log(f"    Timeout, waiting {wait_time:.0f}s before retry...", agent=agent.name)
                    await asyncio.sleep(wait_time)
                    timeout = int(timeout * backoff_factor)  # Increase timeout for next attempt

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = min(120, 10 * (backoff_factor ** attempt))
                    self._log(f"    Error: {e}, waiting {wait_time:.0f}s before retry...", agent=agent.name)
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        self._log(f"    {agent.name} failed after {max_retries} attempts: {last_error}", agent=agent.name)
        return f"[Agent {agent.name} failed after {max_retries} attempts: {last_error}]"

    def _get_modified_files(self) -> list[str]:
        """Get list of modified files (staged and unstaged)."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return files
        except Exception:
            return []

    def _count_test_results(self, test_output: str) -> dict:
        """Parse pytest output to count passed/failed/errors."""
        import re
        result = {"passed": 0, "failed": 0, "errors": 0, "total": 0}

        # Look for pytest summary line: "X passed, Y failed, Z errors"
        summary_match = re.search(r"(\d+) passed", test_output)
        if summary_match:
            result["passed"] = int(summary_match.group(1))

        failed_match = re.search(r"(\d+) failed", test_output)
        if failed_match:
            result["failed"] = int(failed_match.group(1))

        error_match = re.search(r"(\d+) error", test_output)
        if error_match:
            result["errors"] = int(error_match.group(1))

        result["total"] = result["passed"] + result["failed"] + result["errors"]
        return result

    def _extract_failing_files(self, test_output: str) -> list[str]:
        """Extract file paths from pytest failure output."""
        import re
        failing_files = set()

        # Match patterns like "tests/test_foo.py::TestClass::test_method FAILED"
        # or "FAILED tests/test_foo.py::test_method"
        patterns = [
            r"(\S+\.py)::\S+ FAILED",
            r"FAILED (\S+\.py)::",
            r"ERROR (\S+\.py)::",
            r"(\S+\.py):\d+: in \w+",  # Traceback lines
        ]

        for pattern in patterns:
            matches = re.findall(pattern, test_output)
            failing_files.update(matches)

        return list(failing_files)

    def _record_failure_patterns(self, test_output: str, design_context: str) -> None:
        """
        Record failure patterns for Titans/MIRAS learning.

        Extracts failure signatures from test output and records them
        so the system can learn to avoid similar failures in the future.
        """
        import re

        if not self.critique_store:
            return

        # Extract failure types from test output
        failure_patterns = []

        # Pattern 1: AssertionError messages
        assert_matches = re.findall(r"AssertionError: (.+?)(?:\n|$)", test_output)
        for match in assert_matches[:3]:  # Limit to 3
            failure_patterns.append(("assertion", match))

        # Pattern 2: Import errors
        import_matches = re.findall(r"(?:ImportError|ModuleNotFoundError): (.+?)(?:\n|$)", test_output)
        for match in import_matches[:3]:
            failure_patterns.append(("import", match))

        # Pattern 3: Type errors
        type_matches = re.findall(r"TypeError: (.+?)(?:\n|$)", test_output)
        for match in type_matches[:3]:
            failure_patterns.append(("type", match))

        # Pattern 4: Attribute errors
        attr_matches = re.findall(r"AttributeError: (.+?)(?:\n|$)", test_output)
        for match in attr_matches[:3]:
            failure_patterns.append(("attribute", match))

        # Pattern 5: Syntax errors
        syntax_matches = re.findall(r"SyntaxError: (.+?)(?:\n|$)", test_output)
        for match in syntax_matches[:3]:
            failure_patterns.append(("syntax", match))

        # Record each failure pattern
        for issue_type, issue_text in failure_patterns:
            self.critique_store.fail_pattern(issue_text, issue_type=issue_type)
            self._log(f"  Recorded failure pattern: [{issue_type}] {issue_text}")

        if failure_patterns:
            self._log(f"  Recorded {len(failure_patterns)} failure patterns for future learning")

    def _selective_rollback(self, files_to_rollback: list[str]) -> bool:
        """
        Rollback only specific files while preserving others.

        Returns True if rollback was successful.
        """
        if not files_to_rollback:
            return True

        try:
            self._log(f"  Selective rollback of {len(files_to_rollback)} files:")
            for f in files_to_rollback[:5]:  # Show first 5
                self._log(f"    - {f}")
            if len(files_to_rollback) > 5:
                self._log(f"    ... and {len(files_to_rollback) - 5} more")

            # Checkout specific files from HEAD
            subprocess.run(
                ["git", "checkout", "HEAD", "--"] + files_to_rollback,
                cwd=self.aragora_path,
                check=True,
            )
            return True
        except Exception as e:
            self._log(f"  Selective rollback failed: {e}")
            return False

    def _commit_partial_progress(self, message: str) -> Optional[str]:
        """
        Commit current changes as partial progress before attempting more fixes.

        Returns commit hash if successful, None otherwise.
        """
        try:
            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                return None

            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.aragora_path,
                check=True,
            )

            # Commit with WIP message
            result = subprocess.run(
                ["git", "commit", "-m", f"WIP: {message}\n\n[partial progress - may have failing tests]"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                return hash_result.stdout.strip()[:8]
            return None
        except Exception:
            return None

    async def _preserve_failed_work(self, branch_name: str) -> Optional[str]:
        """
        Preserve failed implementation work in a git branch before rollback.

        This ensures that even failed implementations can be inspected and
        potentially salvaged later.

        Returns:
            Branch name if successful, None if failed
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        full_branch_name = f"{branch_name}-{timestamp}"

        try:
            # Get current branch
            current_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            current_branch = current_result.stdout.strip()

            # Check if there are any changes to preserve
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if not status_result.stdout.strip():
                self._log("  No changes to preserve")
                return None

            # Create and switch to preservation branch
            subprocess.run(
                ["git", "checkout", "-b", full_branch_name],
                cwd=self.aragora_path,
                capture_output=True,
                check=True,
            )

            # Stage all changes including untracked files
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.aragora_path,
                capture_output=True,
                check=True,
            )

            # Commit the failed work
            commit_msg = f"WIP: Failed nomic cycle {self.cycle_count} (verification failed)\n\nThis branch contains work that failed verification and was rolled back.\nPreserved for inspection and potential salvage."
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.aragora_path,
                capture_output=True,
            )

            # Switch back to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                cwd=self.aragora_path,
                capture_output=True,
                check=True,
            )

            self._log(f"  Preserved failed work in branch: {full_branch_name}")
            self._stream_emit("on_work_preserved", full_branch_name, self.cycle_count)
            return full_branch_name

        except Exception as e:
            self._log(f"  Warning: Could not preserve work in branch: {e}")
            # Try to get back to original branch
            try:
                subprocess.run(
                    ["git", "checkout", current_branch],
                    cwd=self.aragora_path,
                    capture_output=True,
                )
            except Exception:
                pass
            return None

    async def phase_implement(self, design: str) -> dict:
        """Phase 3: Hybrid multi-model implementation."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 3: IMPLEMENTATION (Hybrid)")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "implement", self.cycle_count, {})
        self._record_replay_event("phase", "system", "implement")

        use_hybrid = os.environ.get("ARAGORA_HYBRID_IMPLEMENT", "1") == "1"

        if not use_hybrid:
            return await self._legacy_implement(design)

        design_hash = hashlib.md5(design.encode()).hexdigest()

        # 1. Check for crash recovery
        progress = load_progress(self.aragora_path)
        if progress and progress.plan.design_hash == design_hash:
            self._log("  Resuming from checkpoint...")
            plan = progress.plan
            completed = set(progress.completed_tasks)
            stash_ref = progress.git_stash_ref
        else:
            # 2. Generate plan
            try:
                self._log("  Generating implementation plan...")
                plan = await generate_implement_plan(design, self.aragora_path)
                self._log(f"  Plan generated: {len(plan.tasks)} tasks")
                for task in plan.tasks:
                    self._log(f"    - {task.id}: {task.description[:300]}...", also_print=False)
            except Exception as e:
                self._log(f"  Plan generation failed: {e}")
                self._log("  Falling back to single-task mode...")
                plan = create_single_task_plan(design, self.aragora_path)

            completed = set()
            stash_ref = self._git_stash_create()

            save_progress(
                ImplementProgress(
                    plan=plan,
                    completed_tasks=[],
                    git_stash_ref=stash_ref,
                ),
                self.aragora_path,
            )

        # Save state
        self._save_state({
            "phase": "implement",
            "stage": "executing",
            "total_tasks": len(plan.tasks),
            "completed_tasks": len(completed),
        })

        # 3.5. Gather implementation suggestions from all agents (parallel)
        # This ensures all agents contribute, with Claude doing the final implementation
        multi_agent_suggestions = await self._gather_implementation_suggestions(design)

        # Enhance the design with suggestions for Claude's implementation
        if multi_agent_suggestions:
            enhanced_design = f"{design}\n\n{multi_agent_suggestions}"
            # Update task descriptions to include suggestions
            for task in plan.tasks:
                if hasattr(task, 'description'):
                    task.description = f"{task.description}\n\n{multi_agent_suggestions[:1000]}"

        # 4. Execute tasks (Claude with HybridExecutor, informed by all agents)
        executor = HybridExecutor(self.aragora_path)

        def on_task_complete(task_id: str, result):
            completed.add(task_id)
            self._log(f"  Task {task_id}: {'completed' if result.success else 'failed'}")
            save_progress(
                ImplementProgress(
                    plan=plan,
                    completed_tasks=list(completed),
                    current_task=None,
                    git_stash_ref=stash_ref,
                ),
                self.aragora_path,
            )
            self._save_state({
                "phase": "implement",
                "stage": "executing",
                "total_tasks": len(plan.tasks),
                "completed_tasks": len(completed),
                "last_task": task_id,
                "last_success": result.success,
            })
            # Stream task completion event (full diff for dashboard visibility)
            self._stream_emit(
                "on_task_complete",
                task_id,
                result.success,
                result.duration_seconds,
                result.diff if result.diff else "",  # Full diff, no truncation
                result.error if not result.success else None,
            )

        try:
            results = await executor.execute_plan(
                plan.tasks,
                completed,
                on_task_complete=on_task_complete,
            )

            all_success = all(r.success for r in results)
            tasks_completed = len([r for r in results if r.success])

            if all_success and tasks_completed == len(plan.tasks):
                clear_progress(self.aragora_path)
                self._log(f"  All {tasks_completed} tasks completed successfully")

                # Pre-verification review: All agents review implementation
                self._log("\n  Pre-verification review...", agent="claude")
                diff = self._get_git_diff()
                if diff and len(diff) > 100:  # Only review if there are substantial changes
                    # Check if protected files were modified
                    touched_protected = self._diff_touches_protected_files(diff)

                    if touched_protected:
                        # Use Deep Audit for protected files (Heavy3-inspired intensive review)
                        self._log(f"    Protected files modified: {touched_protected}")
                        approved, issues = await self._run_deep_audit_for_protected_files(diff, touched_protected)

                        if not approved:
                            self._log("    BLOCKING: Deep Audit rejected changes to protected files")
                            self._log("    Rolling back changes...")
                            self._git_stash_pop(stash_ref)
                            phase_duration = (datetime.now() - phase_start).total_seconds()
                            self._stream_emit(
                                "on_phase_end", "implement", self.cycle_count, False,
                                phase_duration, {"deep_audit_rejected": True, "issues": issues}
                            )
                            return {
                                "phase": "implement",
                                "success": False,
                                "tasks_completed": tasks_completed,
                                "tasks_total": len(plan.tasks),
                                "error": f"Deep Audit rejected protected file changes: {issues[:500]}",
                                "deep_audit_rejected": True,
                            }
                    else:
                        # Regular parallel review for non-protected files
                        review_concerns = await self._parallel_implementation_review(diff)
                        if review_concerns:
                            self._log(f"    Review concerns: {review_concerns[:1000]}...", agent="claude")
                        else:
                            self._log("    All agents approve the implementation", agent="claude")

                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end", "implement", self.cycle_count, True,
                    phase_duration, {"tasks_completed": tasks_completed}
                )
                return {
                    "phase": "implement",
                    "success": True,
                    "tasks_completed": tasks_completed,
                    "tasks_total": len(plan.tasks),
                    "diff": diff,
                    "results": [r.to_dict() for r in results],
                }
            else:
                failed = [r for r in results if not r.success]
                self._log(f"  {len(failed)} tasks failed")
                phase_duration = (datetime.now() - phase_start).total_seconds()
                self._stream_emit(
                    "on_phase_end", "implement", self.cycle_count, False,
                    phase_duration, {"tasks_failed": len(failed)}
                )
                return {
                    "phase": "implement",
                    "success": False,
                    "tasks_completed": tasks_completed,
                    "tasks_total": len(plan.tasks),
                    "error": failed[0].error if failed else "Unknown error",
                    "diff": self._get_git_diff(),
                }

        except Exception as e:
            self._log(f"  Catastrophic failure: {e}")
            self._log("  Rolling back changes...")
            self._git_stash_pop(stash_ref)
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "implement", self.cycle_count, False,
                phase_duration, {"error": str(e)}
            )
            self._stream_emit("on_error", "implement", str(e), True)
            return {
                "phase": "implement",
                "success": False,
                "error": str(e),
            }

    async def _legacy_implement(self, design: str) -> dict:
        """Legacy single-Codex implementation (fallback)."""
        self._log("  Using legacy Codex-only mode...")

        prompt = f"""{SAFETY_PREAMBLE}

Implement this design in the aragora codebase:

{design}

Write the actual code. Create or modify files as needed.
Follow aragora's existing code style and patterns.
Include docstrings and type hints.

CRITICAL SAFETY RULES:
- NEVER delete or modify these protected files: {PROTECTED_FILES}
- NEVER remove existing functionality - only ADD new code
- NEVER simplify code by removing features - complexity is acceptable
- If a file seems "too complex", DO NOT simplify it
- Preserve ALL existing imports, classes, and functions"""

        try:
            result = subprocess.run(
                ["codex", "exec", "-C", str(self.aragora_path), prompt],
                capture_output=True,
                text=True,
                timeout=1200,  # Doubled - Codex has known latency issues
            )

            return {
                "phase": "implement",
                "output": result.stdout,
                "diff": self._get_git_diff(),
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "phase": "implement",
                "error": "Implementation timed out",
                "success": False,
            }
        except Exception as e:
            return {
                "phase": "implement",
                "error": str(e),
                "success": False,
            }

    async def phase_verify(self) -> dict:
        """Phase 4: Verify changes work."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 4: VERIFICATION")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "verify", self.cycle_count, {})
        self._stream_emit("on_verification_start", ["syntax", "import", "tests"])
        self._record_replay_event("phase", "system", "verify")

        checks = []

        # 1. Python syntax check
        self._log("  Checking syntax...")
        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", "aragora/__init__.py"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            passed = result.returncode == 0
            checks.append({
                "check": "syntax",
                "passed": passed,
                "output": result.stderr,
            })
            self._log(f"    {'passed' if passed else 'FAILED'} syntax")
            self._stream_emit("on_verification_result", "syntax", passed, result.stderr if result.stderr else "")
        except Exception as e:
            checks.append({"check": "syntax", "passed": False, "error": str(e)})
            self._log(f"    FAILED syntax: {e}")
            self._stream_emit("on_verification_result", "syntax", False, str(e))

        # 2. Import check
        self._log("  Checking imports...")
        try:
            result = subprocess.run(
                ["python", "-c", "import aragora; print('OK')"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=180,  # Minimum 3 min (was 60)
            )
            passed = "OK" in result.stdout
            checks.append({
                "check": "import",
                "passed": passed,
                "output": result.stderr if result.returncode != 0 else "",
            })
            self._log(f"    {'passed' if passed else 'FAILED'} import")
            self._stream_emit("on_verification_result", "import", passed, result.stderr if result.stderr else "")
        except Exception as e:
            checks.append({"check": "import", "passed": False, "error": str(e)})
            self._log(f"    FAILED import: {e}")
            self._stream_emit("on_verification_result", "import", False, str(e))

        # 3. Run tests
        self._log("  Running tests...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
                timeout=240,
            )
            # pytest returns 5 when no tests are collected - treat as pass
            # Also check for "no tests ran" in output as a fallback
            no_tests_collected = result.returncode == 5 or "no tests ran" in result.stdout.lower()
            passed = result.returncode == 0 or no_tests_collected
            checks.append({
                "check": "tests",
                "passed": passed,
                "output": result.stdout[-500:] if result.stdout else "",
                "note": "no tests collected" if no_tests_collected else "",
            })
            self._log(f"    {'passed' if passed else 'FAILED'} tests" + (" (no tests collected)" if no_tests_collected else ""))
            self._stream_emit("on_verification_result", "tests", passed, result.stdout[-200:] if result.stdout else "")
        except Exception as e:
            # CRITICAL: Don't mask test failures - treat exceptions as failures
            checks.append({"check": "tests", "passed": False, "error": str(e), "note": "Test execution failed"})
            self._log(f"    FAILED tests (exception): {e}")
            self._stream_emit("on_verification_result", "tests", False, f"Exception: {e}")

        all_passed = all(c.get("passed", False) for c in checks)

        # HYBRID MODEL ARCHITECTURE: Codex-led verification audit
        codex_audit = None
        if all_passed and self.codex:
            try:
                self._log("  [hybrid] Codex verification audit...")
                changed_files = self._get_changed_files()
                diff_output = ""
                if changed_files:
                    diff_result = subprocess.run(
                        ["git", "diff", "--unified=3"],
                        cwd=self.aragora_path,
                        capture_output=True,
                        text=True,
                    )
                    diff_output = diff_result.stdout[:5000] if diff_result.returncode == 0 else ""

                audit_prompt = f"""As the verification lead, audit this implementation:

Changed files: {changed_files[:10]}

Diff (first 5000 chars):
{diff_output}

Provide a brief verification report:
1. CODE QUALITY: Are there any obvious issues? (0-10)
2. TEST COVERAGE: Are the changes adequately tested? (0-10)
3. DESIGN ALIGNMENT: Does implementation match the design? (0-10)
4. RISK ASSESSMENT: Any potential runtime issues? (0-10)
5. VERDICT: APPROVE or CONCERNS (with brief explanation)

Be concise - this is a quality gate, not a full review."""

                audit_response = await self.codex.generate(audit_prompt)
                if audit_response:
                    codex_audit = audit_response
                    # Check if audit has concerns
                    if "CONCERNS" in audit_response.upper() and "APPROVE" not in audit_response.upper():
                        self._log("  [hybrid] Codex raised concerns - flagging for review")
                        checks.append({
                            "check": "codex_audit",
                            "passed": False,
                            "output": audit_response[:500],
                            "note": "Codex verification audit raised concerns",
                        })
                        all_passed = False
                    else:
                        self._log("  [hybrid] Codex audit passed")
                        checks.append({
                            "check": "codex_audit",
                            "passed": True,
                            "output": audit_response[:500],
                        })
            except Exception as e:
                self._log(f"  [hybrid] Codex audit error: {e}")
                checks.append({
                    "check": "codex_audit",
                    "passed": True,  # Don't block on audit failure
                    "error": str(e),
                    "note": "Audit skipped due to error",
                })

        self._save_state({
            "phase": "verify",
            "stage": "complete",
            "all_passed": all_passed,
            "checks": checks,
        })

        # NomicIntegration: Check evidence staleness and checkpoint
        stale_claims = []
        if self.nomic_integration:
            try:
                # Get changed files from git
                changed_files = self._get_changed_files()
                if changed_files:
                    self._log(f"  [integration] Checking staleness for {len(changed_files)} changed files...")
                    # Note: We'd need claims from debate to check properly
                    # For now, just log and checkpoint
                    self._log(f"  [integration] Changed files: {changed_files[:5]}...")

                # Checkpoint the verify phase
                await self.nomic_integration.checkpoint(
                    phase="verify",
                    state={"all_passed": all_passed, "checks": checks},
                    cycle=self.cycle_count,
                )
            except Exception as e:
                self._log(f"  [integration] Staleness check failed: {e}")

        phase_duration = (datetime.now() - phase_start).total_seconds()
        self._stream_emit(
            "on_phase_end", "verify", self.cycle_count, all_passed,
            phase_duration, {"checks_passed": sum(1 for c in checks if c.get("passed"))}
        )

        return {
            "phase": "verify",
            "checks": checks,
            "all_passed": all_passed,
            "stale_claims": stale_claims,
        }

    def _get_changed_files(self) -> list[str]:
        """Get list of files changed in this cycle."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception:
            pass
        return []

    async def phase_commit(self, improvement: str) -> dict:
        """Phase 5: Commit changes if verified."""
        phase_start = datetime.now()
        self._log("\n" + "=" * 70)
        self._log("PHASE 5: COMMIT")
        self._log("=" * 70)
        self._stream_emit("on_phase_start", "commit", self.cycle_count, {})

        if self.require_human_approval and not self.auto_commit:
            self._log("\nChanges ready for review:")
            subprocess.run(["git", "diff", "--stat"], cwd=self.aragora_path)

            # Check if auto-commit is enabled or running non-interactively
            if NOMIC_AUTO_COMMIT:
                self._log("\n[commit] Auto-committing (NOMIC_AUTO_COMMIT=1)")
            elif not sys.stdin.isatty():
                # Non-interactive mode: log warning and proceed with commit
                self._log("\n[commit] Non-interactive mode detected - proceeding with auto-commit")
                self._log("[commit] Set NOMIC_AUTO_COMMIT=1 to suppress this warning")
            else:
                # Interactive mode: prompt for approval
                response = input("\nCommit these changes? [y/N]: ")
                if response.lower() != 'y':
                    self._log("Skipping commit.")
                    phase_duration = (datetime.now() - phase_start).total_seconds()
                    self._stream_emit(
                        "on_phase_end", "commit", self.cycle_count, False,
                        phase_duration, {"reason": "human_declined"}
                    )
                    return {"phase": "commit", "committed": False, "reason": "Human declined"}

        summary = improvement.replace('\n', ' ')

        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.aragora_path,
                check=True,
            )

            result = subprocess.run(
                ["git", "commit", "-m", f"feat(nomic): {summary}\n\n🤖 Auto-generated by aragora nomic loop"],
                cwd=self.aragora_path,
                capture_output=True,
                text=True,
            )

            committed = result.returncode == 0

            if committed:
                self._log(f"  Committed: {summary}")
                # Get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
                # Get files changed count
                stat_result = subprocess.run(
                    ["git", "diff", "--stat", "HEAD~1..HEAD"],
                    cwd=self.aragora_path,
                    capture_output=True,
                    text=True,
                )
                files_changed = len([l for l in stat_result.stdout.split('\n') if '|' in l])
                self._stream_emit("on_commit", commit_hash, summary, files_changed)
            else:
                self._log(f"  Commit failed: {result.stderr}")

            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "commit", self.cycle_count, committed,
                phase_duration, {}
            )

            return {
                "phase": "commit",
                "committed": committed,
                "message": summary,
            }

        except Exception as e:
            self._log(f"  Commit error: {e}")
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self._stream_emit(
                "on_phase_end", "commit", self.cycle_count, False,
                phase_duration, {"error": str(e)}
            )
            self._stream_emit("on_error", "commit", str(e), True)
            return {
                "phase": "commit",
                "committed": False,
                "error": str(e),
            }

    async def run_cycle(self) -> dict:
        """
        Run one complete improvement cycle with hard timeout enforcement.

        This is the public entry point that wraps _run_cycle_impl() with
        a hard timeout to prevent runaway cycles. If the cycle exceeds
        NOMIC_MAX_CYCLE_SECONDS, it will be forcibly terminated and any
        changes rolled back to the pre-cycle backup.

        Returns:
            dict: Cycle result with outcome, phases, duration, etc.
        """
        # Increment cycle count before entering impl (needed for error message)
        self.cycle_count += 1

        try:
            return await asyncio.wait_for(
                self._run_cycle_impl(),
                timeout=NOMIC_MAX_CYCLE_SECONDS
            )
        except asyncio.TimeoutError:
            self._log(f"\n{'=' * 70}")
            self._log(f"[CRITICAL] Cycle {self.cycle_count} exceeded {NOMIC_MAX_CYCLE_SECONDS}s hard limit")
            self._log(f"{'=' * 70}")

            # Restore backup if one was created during this cycle
            if hasattr(self, '_cycle_backup_path') and self._cycle_backup_path:
                self._log(f"  [recovery] Rolling back to backup: {self._cycle_backup_path}")
                self._restore_backup(self._cycle_backup_path)
                self._cycle_backup_path = None  # Clear for next cycle

            # Emit timeout event for monitoring
            self._stream_emit("on_cycle_timeout", self.cycle_count, NOMIC_MAX_CYCLE_SECONDS)
            self._dispatch_webhook("cycle_timeout", {
                "cycle": self.cycle_count,
                "timeout_seconds": NOMIC_MAX_CYCLE_SECONDS,
            })

            return {
                "outcome": "cycle_timeout",
                "cycle": self.cycle_count,
                "timeout_seconds": NOMIC_MAX_CYCLE_SECONDS,
                "error": f"Cycle exceeded {NOMIC_MAX_CYCLE_SECONDS}s hard limit",
            }

    async def _run_cycle_impl(self) -> dict:
        """Internal implementation of run_cycle (called with timeout wrapper)."""
        # Note: self.cycle_count already incremented by run_cycle() wrapper
        cycle_start = datetime.now()
        cycle_deadline = cycle_start + timedelta(seconds=self.max_cycle_seconds)

        # Reset per-cycle state and track start time
        self._reset_cycle_state()
        self._cycle_start_time = cycle_start

        # Check for deadlock pattern from previous cycles
        deadlock = self._detect_cycle_deadlock()
        if deadlock:
            action = await self._handle_deadlock(deadlock)
            if action == "skip":
                self._log(f"  [DEADLOCK] Skipping cycle due to repeated failures")
                return {"outcome": "skipped_deadlock", "cycle": self.cycle_count}

        # Update circuit breaker cooldowns at cycle start
        self.circuit_breaker.start_new_cycle()

        # Clear crux cache at cycle start to prevent context bleeding between cycles
        self._cached_cruxes = []

        self._log("\n" + "=" * 70)
        self._log(f"NOMIC CYCLE {self.cycle_count}")
        self._log(f"Started: {cycle_start.isoformat()}")
        self._log(f"Deadline: {cycle_deadline.isoformat()} ({self.max_cycle_seconds}s budget)")
        self._log("=" * 70)

        # Log circuit breaker status
        cb_status = self.circuit_breaker.get_status()
        if any(cb_status["cooldowns"].values()):
            self._log(f"  [circuit-breaker] Agents in cooldown: {[k for k,v in cb_status['cooldowns'].items() if v > 0]}")

        # Security: Verify protected files haven't been tampered with
        all_ok, modified = verify_protected_files_unchanged(self.aragora_path)
        if not all_ok:
            self._log(f"  [SECURITY WARNING] Protected files modified since startup: {modified}")
            # Log but continue - modifications might be legitimate (e.g., from previous cycle)
            # Update checksums to current state
            _init_protected_checksums(self.aragora_path)

        # Emit cycle start event
        self._stream_emit("on_cycle_start", self.cycle_count, self.max_cycles, cycle_start.isoformat())
        self._dispatch_webhook("cycle_start", {"max_cycles": self.max_cycles})

        # Initialize ReplayRecorder for this cycle
        if REPLAY_AVAILABLE and ReplayRecorder:
            replay_dir = self.nomic_dir / "replays"
            replay_dir.mkdir(exist_ok=True)
            self.replay_recorder = ReplayRecorder(
                debate_id=f"nomic-cycle-{self.cycle_count}",
                topic=f"Nomic Loop Cycle {self.cycle_count}",
                proposal=self.initial_proposal or "Self-improvement",
                agents=[{"name": a, "model": a} for a in ["gemini", "claude", "codex", "grok"]],
                storage_dir=str(replay_dir)
            )
            self.replay_recorder.start()
            self._log(f"  [replay] Recording cycle {self.cycle_count}")

        # Initialize ArgumentCartographer for this cycle
        if CARTOGRAPHER_AVAILABLE and ArgumentCartographer:
            self.cartographer = ArgumentCartographer()
            self.cartographer.set_debate_context(
                debate_id=f"nomic-cycle-{self.cycle_count}",
                topic=self.initial_proposal or "Self-improvement"
            )
            self._log(f"  [viz] Cartographer ready for cycle {self.cycle_count}")

        # Phase 4: Initialize agent personas at cycle start
        self._init_agent_personas()

        # === SAFETY: Create backup before any changes ===
        backup_path = self._create_backup(f"cycle_{self.cycle_count}")
        # Store for timeout handler (run_cycle wrapper can access for rollback)
        self._cycle_backup_path = backup_path

        cycle_result = {
            "cycle": self.cycle_count,
            "started": cycle_start.isoformat(),
            "backup_path": str(backup_path),
            "phases": {},
        }

        self._save_state({
            "phase": "cycle_start",
            "cycle": self.cycle_count,
            "backup_path": str(backup_path),
        })

        # Phase 0: Context Gathering (Claude + Codex explore codebase)
        # This ensures Gemini and Grok get accurate context about existing features
        try:
            context_result = await self._run_with_phase_timeout(
                "context",
                self.phase_context_gathering()
            )
            cycle_result["phases"]["context"] = context_result
            codebase_context = context_result.get("context", "")
            self.phase_recovery.record_success("context")
        except PhaseError as e:
            self._log(f"PHASE TIMEOUT: Context gathering exceeded time limit: {e}")
            self.phase_recovery.record_failure("context", e)
            cycle_result["outcome"] = "context_timeout"
            cycle_result["error"] = str(e)
            return cycle_result
        except Exception as e:
            self._log(f"PHASE CRASH: Context gathering failed: {e}")
            self.phase_recovery.record_failure("context", e)
            cycle_result["outcome"] = "context_crashed"
            cycle_result["error"] = str(e)
            return cycle_result

        # === Deadline check after context gathering ===
        if not self._check_cycle_deadline(cycle_deadline, "context_gathering"):
            cycle_result["outcome"] = "timeout"
            cycle_result["timeout_phase"] = "context_gathering"
            return cycle_result

        # Phase 1: Debate (all agents, with gathered context)
        try:
            debate_result = await self._run_with_phase_timeout(
                "debate",
                self.phase_debate(codebase_context=codebase_context)
            )
            cycle_result["phases"]["debate"] = debate_result
            self.phase_recovery.record_success("debate")
        except PhaseError as e:
            self._log(f"PHASE TIMEOUT: Debate exceeded time limit: {e}")
            self.phase_recovery.record_failure("debate", e)
            cycle_result["outcome"] = "debate_timeout"
            cycle_result["error"] = str(e)
            return cycle_result
        except Exception as e:
            self._log(f"PHASE CRASH: Debate phase failed: {e}")
            self.phase_recovery.record_failure("debate", e)
            cycle_result["outcome"] = "debate_crashed"
            cycle_result["error"] = str(e)
            return cycle_result

        if not debate_result.get("consensus_reached"):
            self._log("No consensus reached. Ending cycle.")
            cycle_result["outcome"] = "no_consensus"
            self._record_cycle_outcome("no_consensus", {"phase": "debate"})
            return cycle_result

        # === Deadline check after debate ===
        if not self._check_cycle_deadline(cycle_deadline, "debate"):
            cycle_result["outcome"] = "timeout"
            cycle_result["timeout_phase"] = "debate"
            return cycle_result

        improvement = debate_result["final_answer"]
        self._log(f"\nConsensus improvement:\n{improvement}")  # Full content, no truncation

        # Phase 2: Design (with belief analysis from debate)
        belief_analysis = debate_result.get("belief_analysis")
        try:
            design_result = await self._run_with_phase_timeout(
                "design",
                self.phase_design(improvement, belief_analysis=belief_analysis)
            )
            cycle_result["phases"]["design"] = design_result
            self.phase_recovery.record_success("design")
        except PhaseError as e:
            self._log(f"PHASE TIMEOUT: Design exceeded time limit: {e}")
            self.phase_recovery.record_failure("design", e)
            cycle_result["outcome"] = "design_timeout"
            cycle_result["error"] = str(e)
            return cycle_result
        except Exception as e:
            self._log(f"PHASE CRASH: Design phase failed: {e}")
            self.phase_recovery.record_failure("design", e)
            cycle_result["outcome"] = "design_crashed"
            cycle_result["error"] = str(e)
            return cycle_result

        design = design_result.get("design", "")
        design_consensus = design_result.get("consensus_reached", False)
        design_confidence = design_result.get("confidence", 0.0)
        vote_counts = design_result.get("vote_counts", {})
        individual_proposals = design_result.get("individual_proposals", {})
        self._log(f"\nDesign complete (consensus={design_consensus}, confidence={design_confidence:.0%})")

        # === Design Fallback: Multi-strategy recovery for no consensus ===
        if not design_consensus:
            self._log("  [fallback] No design consensus - attempting multi-strategy recovery...")
            candidate_design = None

            # Helper to validate design quality
            def is_viable_design(d: str) -> bool:
                if not d or len(d.strip()) < 100:
                    return False
                # Must contain actual implementation details
                keywords = ["file", "function", "class", "import", "def ", "async ", "return"]
                return any(kw in d.lower() for kw in keywords)

            # Strategy 1: Always try judge arbitration first (not just close contests)
            if individual_proposals and len(individual_proposals) >= 2:
                self._log("  [arbitration] Multiple proposals exist - invoking judge arbitration...")
                try:
                    arbitrated = await self._arbitrate_design(individual_proposals, improvement)
                    if arbitrated and is_viable_design(arbitrated):
                        candidate_design = arbitrated
                        self._log(f"  [arbitration] Judge synthesized viable design ({len(arbitrated)} chars)")
                except Exception as e:
                    self._log(f"  [arbitration] Judge arbitration failed: {e}")

            # Strategy 2: Use highest-voted viable proposal
            if not candidate_design and vote_counts and individual_proposals:
                sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
                for agent, votes in sorted_votes:
                    if agent in individual_proposals:
                        proposal = individual_proposals[agent]
                        if is_viable_design(proposal):
                            candidate_design = proposal
                            self._log(f"  [fallback] Selected {agent}'s design with {votes} votes ({len(proposal)} chars)")
                            break
                        else:
                            self._log(f"  [fallback] Skipped {agent}'s design (not viable: {len(proposal) if proposal else 0} chars)")

            # Strategy 3: Use any viable proposal regardless of votes
            if not candidate_design and individual_proposals:
                for agent, proposal in individual_proposals.items():
                    if is_viable_design(proposal):
                        candidate_design = proposal
                        self._log(f"  [fallback] Selected {agent}'s design (first viable found)")
                        break

            # Strategy 4: Use conditional design from counterfactual analysis if available
            conditional_design = design_result.get("conditional_design")
            if not candidate_design and conditional_design and is_viable_design(conditional_design):
                candidate_design = conditional_design
                self._log("  [fallback] Using conditional design from counterfactual analysis")

            # Final assignment
            if candidate_design:
                design = candidate_design
                self._log(f"  [fallback] Proceeding with recovered design ({len(design)} chars)")
            else:
                self._log("  [warning] No viable design recovered - skipping implementation")
                cycle_result["outcome"] = "design_no_consensus"
                cycle_result["vote_counts"] = vote_counts
                cycle_result["proposals_checked"] = len(individual_proposals) if individual_proposals else 0
                self._record_cycle_outcome("design_no_consensus", {"vote_counts": vote_counts})
                return cycle_result
        elif design_confidence < 0.5:
            self._log("  [warning] Design has low confidence - proceeding with caution")

        # === Deadline check after design ===
        if not self._check_cycle_deadline(cycle_deadline, "design"):
            cycle_result["outcome"] = "timeout"
            cycle_result["timeout_phase"] = "design"
            return cycle_result

        # Phase 3: Implement (with circuit breaker integration)
        try:
            impl_result = await self._run_with_phase_timeout(
                "implement",
                self.phase_implement(design)
            )
            cycle_result["phases"]["implement"] = impl_result
            self.phase_recovery.record_success("implement")
            # Track success for primary implementation agent
            self.circuit_breaker.record_task_success("claude", "implement")
        except PhaseError as e:
            self._log(f"PHASE TIMEOUT: Implementation exceeded time limit: {e}")
            self.phase_recovery.record_failure("implement", e)
            self.circuit_breaker.record_task_failure("claude", "implement")
            cycle_result["outcome"] = "implement_timeout"
            cycle_result["error"] = str(e)
            return cycle_result
        except Exception as e:
            self._log(f"PHASE CRASH: Implementation phase failed: {e}")
            self.phase_recovery.record_failure("implement", e)
            self.circuit_breaker.record_task_failure("claude", "implement")
            cycle_result["outcome"] = "implement_crashed"
            cycle_result["error"] = str(e)
            return cycle_result

        if not impl_result.get("success"):
            self._log("Implementation failed. Ending cycle.")
            cycle_result["outcome"] = "implementation_failed"
            return cycle_result

        # === Deadline check after implementation ===
        if not self._check_cycle_deadline(cycle_deadline, "implement"):
            cycle_result["outcome"] = "timeout"
            cycle_result["timeout_phase"] = "implement"
            return cycle_result

        self._log(f"\nImplementation complete")
        self._log(f"Changed files:\n{impl_result.get('diff', 'No changes')}")

        # === Check evidence staleness after implementation ===
        if self.nomic_integration and self.claims_kernel:
            try:
                changed_files = self._get_git_changed_files()
                if changed_files:
                    staleness = await self.nomic_integration.check_staleness(
                        list(self.claims_kernel.claims.values()),
                        changed_files
                    )
                    if staleness.stale_claims:
                        self._log(f"  [staleness] {len(staleness.stale_claims)} claims have stale evidence")
                        for claim in staleness.stale_claims[:3]:  # Log first 3
                            self._log(f"    - {claim.statement[:60]}...")

                        # Queue ALL stale claims for next cycle's debate agenda
                        if not hasattr(self, '_pending_redebate_claims'):
                            self._pending_redebate_claims = []
                        self._pending_redebate_claims.extend(staleness.stale_claims)
                        self._log(f"  [staleness] Queued {len(staleness.stale_claims)} claims for re-debate in next cycle")

                    if staleness.needs_redebate:
                        self._log(f"  [staleness] WARNING: High-severity stale evidence detected!")
                        cycle_result["needs_redebate"] = True
                        cycle_result["stale_claims"] = [c.claim_id for c in staleness.stale_claims]
            except Exception as e:
                self._log(f"  [staleness] Check failed: {e}")

        # === SAFETY: Verify protected files are intact ===
        self._log("\n  Checking protected files...")
        protected_issues = self._verify_protected_files()
        if protected_issues:
            self._log("  CRITICAL: Protected files damaged!")
            for issue in protected_issues:
                self._log(f"    - {issue}")

            # Preserve work before rollback
            preserve_branch = await self._preserve_failed_work(f"nomic-protected-damaged-{self.cycle_count}")
            if preserve_branch:
                cycle_result["preserved_branch"] = preserve_branch
                self._log(f"  Work preserved in branch: {preserve_branch}")

            self._log("  Restoring from backup...")
            self._restore_backup(backup_path)
            subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
            cycle_result["outcome"] = "protected_files_damaged"
            cycle_result["protected_issues"] = protected_issues
            return cycle_result
        self._log("  All protected files intact")

        # === Iterative Review/Fix Cycle ===
        # Default: 3 fix attempts with all agents before rollback.
        # The fix cycle: Codex reviews -> Claude fixes -> Gemini reviews -> Grok attempts -> re-verify
        # Set ARAGORA_MAX_FIX_ITERATIONS to override (minimum 3 recommended for thorough fixing)
        max_fix_iterations = int(os.environ.get("ARAGORA_MAX_FIX_ITERATIONS", "3"))
        fix_iteration = 0
        cycle_result["fix_iterations"] = []
        best_test_score = 0  # Track progress: best passing test count
        best_test_output = ""

        while True:
            # Deadline enforcement: check remaining time before each iteration
            if cycle_deadline:
                remaining_seconds = (cycle_deadline - datetime.now()).total_seconds()
                if remaining_seconds < 300:  # Less than 5 minutes remaining
                    self._log(f"\n  [deadline] Only {remaining_seconds:.0f}s remaining - exiting fix loop early")
                    cycle_result["outcome"] = "deadline_reached"
                    cycle_result["deadline_remaining_seconds"] = remaining_seconds
                    # Try to preserve any partial progress
                    if best_test_score > 0:
                        self._log(f"  [deadline] Preserving partial progress: {best_test_score} passing tests")
                        cycle_result["partial_success"] = True
                        cycle_result["best_test_score"] = best_test_score
                    break

            # Phase 4: Verify (with timeout and recovery)
            try:
                # Use phase timeout to prevent indefinite hangs
                verify_result = await self._run_with_phase_timeout(
                    "verify",
                    self.phase_verify()
                )
                cycle_result["phases"]["verify"] = verify_result
                self.phase_recovery.record_success("verify")
            except PhaseError as e:
                self._log(f"PHASE TIMEOUT: Verification phase exceeded timeout: {e}")
                verify_result = {"all_passed": False, "checks": [], "error": str(e)}
                cycle_result["phases"]["verify"] = verify_result
                self.phase_recovery.record_failure("verify", e)
            except Exception as e:
                self._log(f"PHASE CRASH: Verification phase failed: {e}")
                # Treat as failed verification, continue to next fix iteration
                verify_result = {"all_passed": False, "checks": [], "error": str(e)}
                cycle_result["phases"]["verify"] = verify_result
                self.phase_recovery.record_failure("verify", e)

            if verify_result.get("all_passed"):
                self._log(f"\nVerification passed!")
                break  # Success - exit the fix loop

            # Get test output for progress tracking
            test_output = ""
            for check in verify_result.get("checks", []):
                if check.get("check") == "tests":
                    test_output = check.get("output", "")
                    break

            # Track progress - count passing tests
            test_counts = self._count_test_results(test_output)
            current_score = test_counts["passed"]
            self._log(f"  Test results: {test_counts['passed']} passed, {test_counts['failed']} failed, {test_counts['errors']} errors")

            # Update best score if improving
            if current_score > best_test_score:
                best_test_score = current_score
                best_test_output = test_output
                self._log(f"  Progress: New best score = {best_test_score} passing tests")
                # Commit partial progress when improving
                partial_commit = self._commit_partial_progress(f"cycle-{self.cycle_count}-iter-{fix_iteration}-{best_test_score}passed")
                if partial_commit:
                    self._log(f"  Partial progress committed: {partial_commit}")

            # Verification failed
            fix_iteration += 1
            iteration_result = {
                "iteration": fix_iteration,
                "verify_result": verify_result,
                "test_counts": test_counts,
            }

            if fix_iteration > max_fix_iterations:
                # No more fix attempts allowed - try smart rollback first
                self._log(f"\n{'=' * 50}")
                self._log(f"MAX FIX ITERATIONS REACHED ({max_fix_iterations})")
                self._log(f"{'=' * 50}")

                if self.disable_rollback:
                    self._log(f"Verification failed after {fix_iteration - 1} fix attempts.")
                    self._log("  ROLLBACK DISABLED - keeping changes for inspection")
                    cycle_result["outcome"] = "verification_failed_no_rollback"
                    cycle_result["fix_iterations"].append(iteration_result)
                    return cycle_result

                # Try selective rollback first - only rollback files causing failures
                self._log("\n  Attempting selective rollback (preserve passing changes)...")
                failing_test_files = self._extract_failing_files(test_output)
                modified_files = self._get_modified_files()

                # Find files that might be causing the failures
                problematic_files = []
                for mod_file in modified_files:
                    # Check if this modified file is referenced in failing tests
                    if any(mod_file in ft or ft in mod_file for ft in failing_test_files):
                        problematic_files.append(mod_file)

                if problematic_files and len(problematic_files) < len(modified_files):
                    # Try selective rollback
                    self._log(f"  Found {len(problematic_files)} potentially problematic files (keeping {len(modified_files) - len(problematic_files)} others)")
                    if self._selective_rollback(problematic_files):
                        # Re-verify after selective rollback
                        self._log("  Re-verifying after selective rollback...")
                        try:
                            re_verify = await self.phase_verify()
                        except Exception as e:
                            self._log(f"  Re-verify crashed: {e}")
                            re_verify = {"all_passed": False}
                        if re_verify.get("all_passed"):
                            self._log("  Selective rollback succeeded! Keeping partial changes.")
                            cycle_result["outcome"] = "partial_success"
                            cycle_result["selective_rollback"] = problematic_files
                            break
                        else:
                            self._log("  Selective rollback did not fix all issues, proceeding to full rollback")

                # Preserve work in a branch before full rollback
                preserve_branch = await self._preserve_failed_work(f"nomic-failed-cycle-{self.cycle_count}")
                if preserve_branch:
                    cycle_result["preserved_branch"] = preserve_branch
                    self._log(f"  Work preserved in branch: {preserve_branch}")

                # Track failure patterns for learning (Titans/MIRAS)
                if self.critique_store:
                    self._record_failure_patterns(test_output, cycle_result.get("design", ""))

                self._log(f"Verification failed after {fix_iteration - 1} fix attempts. Rolling back.")
                self._restore_backup(backup_path)
                subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
                cycle_result["outcome"] = "verification_failed"
                cycle_result["best_test_score"] = best_test_score
                cycle_result["fix_iterations"].append(iteration_result)
                return cycle_result

            self._log(f"\n{'=' * 50}")
            self._log(f"FIX ITERATION {fix_iteration}/{max_fix_iterations}")
            self._log(f"{'=' * 50}")

            # test_output already extracted above for progress tracking

            # Step 1: Codex reviews the failed changes
            self._log("\n  Step 1: Codex analyzing test failures...", agent="codex")
            from aragora.implement import HybridExecutor
            executor = HybridExecutor(self.aragora_path)
            diff = self._get_git_diff()

            # Get learned patterns for fix guidance (Titans/MIRAS)
            fix_patterns = self._format_successful_patterns(limit=3)
            avoid_patterns = self._format_failure_patterns(limit=3)

            # Get belief network cruxes for targeted fixing (P18: BeliefNetwork → Fix Guidance)
            crux_context = self._format_crux_context()

            review_prompt = f"""The following code changes caused test failures. Analyze and suggest fixes.

## Test Failures
```
{test_output[:2000]}
```

## Code Changes (git diff)
```
{diff[:10000]}
```
{fix_patterns}
{avoid_patterns}
{crux_context}
Provide specific, actionable fixes. Focus on:
1. What exactly is broken?
2. What specific code changes will fix it?
3. Are there missing imports or dependencies?
4. Learn from patterns above - apply what's worked, avoid what hasn't.
5. If pivotal claims are listed above, ensure your fix addresses them directly.
"""
            review_result = await executor.review_with_codex(review_prompt, timeout=2400)  # 40 min for thorough review
            iteration_result["codex_review"] = review_result
            self._log(f"    Codex review complete", agent="codex")
            # Emit Codex's full review
            if review_result.get("review"):
                self._stream_emit("on_log_message", review_result["review"], level="info", phase="fix", agent="codex")

            # Step 2: Claude fixes based on Codex review
            self._log("\n  Step 2: Claude applying fixes...", agent="claude")
            fix_prompt = f"""{SAFETY_PREAMBLE}

Fix the test failures in the codebase. Here's what went wrong and how to fix it:

## Test Failures
```
{test_output[:1500]}
```

## Codex Analysis
{review_result.get('review', 'No review available')[:2000]}

## Instructions
1. Read the failing tests to understand what's expected
2. Apply the minimal fixes needed to make tests pass
3. Do NOT remove or simplify existing functionality
4. Preserve all imports and dependencies

Working directory: {self.aragora_path}
"""
            try:
                fix_agent = ClaudeAgent(
                    name="claude-fixer",
                    model="claude",
                    role="fixer",
                    timeout=1200,  # Doubled - fixes can be complex
                )
                # Use retry wrapper for resilience
                fix_result = await self._call_agent_with_retry(fix_agent, fix_prompt, max_retries=2)
                if "[Agent" in fix_result and "failed" in fix_result:
                    iteration_result["fix_error"] = fix_result
                    self._log(f"    Fix failed: {fix_result[:500]}", agent="claude")
                else:
                    iteration_result["fix_applied"] = True
                    self._log("    Fixes applied", agent="claude")
            except Exception as e:
                iteration_result["fix_error"] = str(e)
                self._log(f"    Fix failed: {e}", agent="claude")

            # Step 3: Gemini quick review (optional sanity check)
            self._log("\n  Step 3: Gemini quick review...", agent="gemini")
            gemini_issues = False
            try:
                gemini_review_prompt = f"""Quick review of fix attempt. Are these changes correct?

## Changes Made (by Claude)
{self._get_git_diff()[:2000]}

## Original Test Failures
{test_output[:500]}

Reply with: LOOKS_GOOD or ISSUES: <brief description>
"""
                # Use retry wrapper for resilience
                gemini_result = await self._call_agent_with_retry(self.gemini, gemini_review_prompt, max_retries=2)
                iteration_result["gemini_review"] = gemini_result if gemini_result else "No response"
                self._log(f"    Gemini: {gemini_result if gemini_result else 'No response'}", agent="gemini")
                # Emit Gemini's full review
                if gemini_result and not ("[Agent" in gemini_result and "failed" in gemini_result):
                    self._stream_emit("on_log_message", gemini_result, level="info", phase="fix", agent="gemini")
                    # Check if Gemini found issues
                    gemini_issues = "ISSUES:" in gemini_result.upper() or "ISSUE:" in gemini_result.upper()
            except Exception as e:
                self._log(f"    Gemini review skipped: {e}", agent="gemini")

            # Step 4: Grok attempts fixes if Gemini found issues or Claude's fix failed
            if gemini_issues or iteration_result.get("fix_error"):
                self._log("\n  Step 4: Grok attempting alternative fix...", agent="grok")
                try:
                    grok_fix_prompt = f"""{SAFETY_PREAMBLE}

Previous fix attempt may have issues. Please apply an alternative fix for these test failures:

## Test Failures
```
{test_output[:1500]}
```

## Previous Attempt Issues
{iteration_result.get('gemini_review', 'Unknown issues')}
{iteration_result.get('fix_error', '')}

## Current Changes (may be partially correct)
{self._get_git_diff()[:2000]}

## Instructions
1. Analyze what the previous fix attempt got wrong
2. Apply a DIFFERENT approach to fix the tests
3. Focus on minimal, targeted changes
4. Do NOT undo correct fixes, only fix what's still broken

Working directory: {self.aragora_path}
"""
                    # Use retry wrapper for resilience
                    grok_result = await self._call_agent_with_retry(self.grok, grok_fix_prompt, max_retries=2)
                    iteration_result["grok_fix"] = grok_result if grok_result else "No response"
                    if grok_result and not ("[Agent" in grok_result and "failed" in grok_result):
                        self._log(f"    Grok fix applied", agent="grok")
                        self._stream_emit("on_log_message", grok_result, level="info", phase="fix", agent="grok")
                    else:
                        self._log(f"    Grok fix failed: {grok_result if grok_result else 'No response'}", agent="grok")
                except Exception as e:
                    self._log(f"    Grok fix skipped: {e}", agent="grok")
            else:
                self._log("\n  Step 4: Grok fix skipped (Gemini approved Claude's changes)", agent="grok")

            cycle_result["fix_iterations"].append(iteration_result)

            # Re-check protected files after fix
            protected_issues = self._verify_protected_files()
            if protected_issues:
                self._log("  CRITICAL: Fix damaged protected files!")
                # Preserve work before rollback
                preserve_branch = await self._preserve_failed_work(f"nomic-fix-damaged-{self.cycle_count}")
                if preserve_branch:
                    cycle_result["preserved_branch"] = preserve_branch
                    self._log(f"  Work preserved in branch: {preserve_branch}")
                self._restore_backup(backup_path)
                subprocess.run(["git", "checkout", "."], cwd=self.aragora_path)
                cycle_result["outcome"] = "protected_files_damaged"
                return cycle_result

            self._log("\n  Re-running verification...")

        self._log(f"\nVerification passed")

        # Phase 5: Commit (with recovery tracking)
        try:
            commit_result = await self._run_with_phase_timeout(
                "commit",
                self.phase_commit(improvement)
            )
            cycle_result["phases"]["commit"] = commit_result
            self.phase_recovery.record_success("commit")
        except PhaseError as e:
            self._log(f"PHASE TIMEOUT: Commit exceeded time limit: {e}")
            self.phase_recovery.record_failure("commit", e)
            cycle_result["outcome"] = "commit_timeout"
            cycle_result["error"] = str(e)
            # Don't return early - still want to log leaderboard etc.
            commit_result = {"committed": False, "reason": "timeout"}
            cycle_result["phases"]["commit"] = commit_result
        except Exception as e:
            self._log(f"PHASE CRASH: Commit phase failed: {e}")
            self.phase_recovery.record_failure("commit", e)
            cycle_result["outcome"] = "commit_crashed"
            cycle_result["error"] = str(e)
            commit_result = {"committed": False, "reason": str(e)}
            cycle_result["phases"]["commit"] = commit_result

        if commit_result.get("committed"):
            cycle_result["outcome"] = "success"
            self._log(f"\nCYCLE {self.cycle_count} COMPLETE - Changes committed!")
        else:
            cycle_result["outcome"] = "not_committed"

        # Phase 5: Log ELO leaderboard every 5 cycles (P13: EloSystem)
        self._log_elo_leaderboard()

        # Phase 9: Log grounded persona insights every 2 cycles
        if self.cycle_count % 2 == 0:
            self._log_persona_insights()

        # Phase 6: Run pending verification proofs (P19: ProofExecutor)
        await self._run_verification_proofs()

        # Phase 6: Verify evidence chain integrity (P17: ProvenanceManager)
        self._verify_evidence_chain()

        # Phase 7: Check evidence staleness for living documents (P21: EnhancedProvenance)
        staleness_status = await self._check_evidence_staleness()
        if staleness_status:
            cycle_result["staleness"] = staleness_status

        cycle_result["duration_seconds"] = (datetime.now() - cycle_start).total_seconds()

        # Add phase duration metrics for analysis (Phase 4 enhancement)
        if hasattr(self, '_phase_metrics') and self._phase_metrics:
            cycle_result["phase_metrics"] = self._phase_metrics
            # Log summary of phase efficiency
            total_budget = sum(m["budget"] for m in self._phase_metrics.values())
            total_duration = sum(m["duration"] for m in self._phase_metrics.values())
            overall_efficiency = (total_duration / total_budget * 100) if total_budget > 0 else 0
            self._log(f"\n  [metrics] Cycle {self.cycle_count} phase efficiency: {overall_efficiency:.0f}% ({total_duration:.0f}s / {total_budget}s budget)")

        self.history.append(cycle_result)

        self._save_state({
            "phase": "cycle_complete",
            "cycle": self.cycle_count,
            "outcome": cycle_result["outcome"],
            "duration_seconds": cycle_result["duration_seconds"],
        })

        # Emit cycle end event
        self._stream_emit(
            "on_cycle_end",
            self.cycle_count,
            cycle_result.get("outcome") == "success",
            cycle_result["duration_seconds"],
            cycle_result.get("outcome", "unknown"),
        )
        self._dispatch_webhook("cycle_end", {
            "outcome": cycle_result.get("outcome", "unknown"),
            "duration_seconds": cycle_result.get("duration_seconds", 0),
            "success": cycle_result.get("outcome") == "success",
        })

        # Finalize ReplayRecorder
        if self.replay_recorder:
            try:
                outcome = cycle_result.get("outcome", "unknown")
                votes = {"success": 1 if outcome == "success" else 0}
                replay_path = self.replay_recorder.finalize(outcome, votes)
                self._log(f"  [replay] Cycle recorded to {replay_path}")
            except Exception as e:
                self._log(f"  [replay] Finalization error: {e}")
            finally:
                self.replay_recorder = None

        # Export ArgumentCartographer visualization
        if self.cartographer:
            try:
                # Export as Mermaid markdown
                mermaid_path = self.visualizations_dir / f"cycle-{self.cycle_count}.md"
                mermaid_content = self.cartographer.export_mermaid()
                with open(mermaid_path, "w") as f:
                    f.write(f"# Cycle {self.cycle_count} Debate Graph\n\n")
                    f.write("```mermaid\n")
                    f.write(mermaid_content)
                    f.write("\n```\n")

                # Export as JSON for analysis
                json_path = self.visualizations_dir / f"cycle-{self.cycle_count}.json"
                with open(json_path, "w") as f:
                    f.write(self.cartographer.export_json(include_full_content=True))

                stats = self.cartographer.get_statistics()
                self._log(f"  [viz] Exported: {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
            except Exception as e:
                self._log(f"  [viz] Export error: {e}")
            finally:
                self.cartographer = None

        # Store cycle outcome in ContinuumMemory for pattern learning
        if self.continuum and CONTINUUM_AVAILABLE:
            try:
                outcome = cycle_result.get("outcome", "unknown")
                improvement = cycle_result.get("phases", {}).get("debate", {}).get("final_answer", "")
                is_success = outcome == "success"

                # Extract domain and participating agents for cross-cycle learning
                domain = self._detect_domain(improvement) if improvement else "general"
                debate_agents = []
                if cycle_result.get("phases", {}).get("debate", {}).get("agents"):
                    debate_agents = cycle_result["phases"]["debate"]["agents"]
                elif hasattr(self, '_last_debate_team'):
                    debate_agents = [a.name for a in getattr(self, '_last_debate_team', [])]

                # Store in SLOW tier (strategic learning across cycles)
                memory_id = f"cycle-{self.cycle_count}-{outcome}"
                self.continuum.add(
                    id=memory_id,
                    content=f"Cycle {self.cycle_count}: {outcome}. Domain: {domain}. Improvement: {improvement}",
                    tier=MemoryTier.SLOW,
                    importance=0.8 if is_success else 0.5,
                    metadata={
                        "cycle": self.cycle_count,
                        "outcome": outcome,
                        "duration_seconds": cycle_result.get("duration_seconds", 0),
                        "success": is_success,
                        "domain": domain,
                        "agents": debate_agents,
                        "phases_completed": list(cycle_result.get("phases", {}).keys()),
                    }
                )
                self._log(f"  [continuum] Stored cycle outcome in SLOW tier (domain={domain})")

                # Consolidate memory periodically (every 3 cycles for faster learning)
                if self.cycle_count % 3 == 0:
                    stats = self.continuum.consolidate()
                    self._log(f"  [continuum] Consolidated: {stats}")

                # Run MetaLearner to self-tune hyperparameters (every cycle)
                if self.meta_learner:
                    try:
                        metrics = self.meta_learner.evaluate_learning_efficiency(
                            self.continuum, cycle_result
                        )
                        adjustments = self.meta_learner.adjust_hyperparameters(metrics)
                        if adjustments:
                            # Get the actual numeric hyperparameters after adjustments
                            new_hyperparams = self.meta_learner.get_current_hyperparams()
                            # Apply adjustments to ContinuumMemory
                            if hasattr(self.continuum, 'hyperparams') and isinstance(self.continuum.hyperparams, dict):
                                self.continuum.hyperparams.update(new_hyperparams)
                            elif hasattr(self.continuum, 'hyperparams'):
                                for key, value in new_hyperparams.items():
                                    if hasattr(self.continuum.hyperparams, key):
                                        setattr(self.continuum.hyperparams, key, value)
                            self._log(f"  [meta] Applied hyperparameter adjustments: {list(adjustments.keys())}")

                            # Also apply relevant adjustments to next debate protocol
                            # MetaLearner's consensus_rate metric influences debate behavior
                            if hasattr(self, 'debate_protocol') and metrics.consensus_rate < 0.5:
                                # Low consensus rate - allow more rounds for debate
                                if hasattr(self.debate_protocol, 'rounds'):
                                    self.debate_protocol.rounds = min(5, self.debate_protocol.rounds + 1)
                                    self._log(f"  [meta] Increased debate rounds to {self.debate_protocol.rounds} (low consensus)")
                    except Exception as e:
                        self._log(f"  [meta] MetaLearner error: {e}")
            except Exception as e:
                self._log(f"  [continuum] Storage error: {e}")

        # Prune stale patterns periodically (every 10 cycles)
        if self.critique_store and self.cycle_count % 10 == 0:
            try:
                if hasattr(self.critique_store, 'prune_stale_patterns'):
                    pruned = self.critique_store.prune_stale_patterns(
                        max_age_days=90,
                        min_success_rate=0.3,
                        archive=True
                    )
                    if pruned > 0:
                        self._log(f"  [memory] Pruned {pruned} stale patterns (archived)")
            except Exception as e:
                self._log(f"  [memory] Pattern pruning error: {e}")

        # Run robustness check on debate conclusions periodically
        if self.scenario_comparator and self.cycle_count % 5 == 0:
            try:
                debate_answer = cycle_result.get("phases", {}).get("debate", {}).get("final_answer", "")
                if debate_answer:
                    robustness = await self._run_robustness_check(
                        task=debate_answer[:500],
                        base_context=""
                    )
                    if robustness:
                        cycle_result["robustness"] = robustness
                        vuln_score = robustness.get("vulnerability_score", 0)
                        if vuln_score > 0.5:
                            self._log(f"  [robustness] Warning: high vulnerability score {vuln_score:.2f}")
            except Exception as e:
                self._log(f"  [robustness] Check failed: {e}")

        # Manage agent bench based on calibration (every 25 cycles)
        if self.agent_selector and self.elo_system and self.cycle_count % 25 == 0:
            try:
                for agent_name in ["gemini", "claude", "codex", "grok"]:
                    if hasattr(self.elo_system, 'get_expected_calibration_error'):
                        ece = self.elo_system.get_expected_calibration_error(agent_name)
                        if ece is None:
                            continue

                        bench_list = getattr(self.agent_selector, 'bench', [])
                        if ece > 0.25 and agent_name not in bench_list:
                            if hasattr(self.agent_selector, 'move_to_bench'):
                                self.agent_selector.move_to_bench(agent_name)
                                self._log(f"  [bench] Moved {agent_name} to probation (ECE: {ece:.2f})")

                        elif agent_name in bench_list:
                            rating = self.elo_system.get_rating(agent_name)
                            if ece < 0.15 and rating.elo > 1550:
                                if hasattr(self.agent_selector, 'promote_from_bench'):
                                    self.agent_selector.promote_from_bench(agent_name)
                                    self._log(f"  [bench] Promoted {agent_name} back to active")
            except Exception as e:
                self._log(f"  [bench] Management error: {e}")

        # Run capability probes periodically (P6: CapabilityProber)
        await self._probe_agent_capabilities()

        # Phase 4: Agent Evolution - evolve prompts periodically (every 10 cycles)
        await self._evolve_agent_prompts()

        # Phase 4: Agent Evolution - run tournament periodically (every 20 cycles)
        await self._run_tournament_if_due()

        # Record cycle outcome for deadlock detection
        self._record_cycle_outcome(cycle_result.get("outcome", "unknown"), {
            "duration": cycle_result.get("duration_seconds"),
            "phases_completed": list(cycle_result.get("phases", {}).keys()),
        })

        # Record cycle outcome in ContinuumMemory for cross-cycle learning
        if self.continuum:
            try:
                outcome = cycle_result.get("outcome", "unknown")
                cycle_id = f"cycle_{self.cycle_count}_{outcome}"
                phases_completed = list(cycle_result.get("phases", {}).keys())

                # Add memory entry for this cycle
                self.continuum.add(
                    id=cycle_id,
                    content=f"Cycle {self.cycle_count}: {outcome}. Phases completed: {', '.join(phases_completed)}",
                    importance=0.7 if outcome != "success" else 0.5,
                    metadata={
                        "cycle": self.cycle_count,
                        "outcome": outcome,
                        "phases": phases_completed,
                        "duration": cycle_result.get("duration_seconds"),
                    },
                )

                # Update with success/failure for surprise-based learning
                is_success = outcome == "success"
                self.continuum.update_outcome(cycle_id, success=is_success)
                self._log(f"  [continuum] Recorded cycle outcome: {outcome}")
            except Exception as e:
                self._log(f"  [continuum] Failed to record outcome: {e}")

        # Reset deadlock counter and consensus decay on success
        if cycle_result.get("outcome") == "success":
            self._deadlock_count = 0
            if self._consensus_threshold_decay > 0:
                self._log(f"  [consensus] Resetting threshold decay (was level {self._consensus_threshold_decay})")
                self._consensus_threshold_decay = 0

        return cycle_result

    async def run(self):
        """Run the nomic loop until max cycles or interrupted."""
        self._log("=" * 70)
        self._log("ARAGORA NOMIC LOOP")
        self._log("Self-improving multi-agent system")
        self._log("=" * 70)
        self._log(f"Max cycles: {self.max_cycles}")
        self._log(f"Human approval required: {self.require_human_approval}")
        self._log(f"Auto-commit: {self.auto_commit}")
        if self.initial_proposal:
            self._log(f"Initial proposal: {self.initial_proposal}")
        self._log("=" * 70)
        self._log(f"Log file: {self.log_file}")
        self._log(f"State file: {self.state_file}")
        self._log(f"Backup dir: {self.backup_dir}")
        self._log("=" * 70)

        # Validate fallback configuration before starting
        self._validate_openrouter_fallback()

        # Run database maintenance on startup (WAL checkpoint, ANALYZE if due)
        try:
            from aragora.maintenance import run_startup_maintenance
            maintenance_results = run_startup_maintenance(self.nomic_dir)
            db_count = maintenance_results.get("stats", {}).get("database_count", 0)
            self._log(f"[maintenance] Startup complete: {db_count} databases checked")
        except Exception as e:
            self._log(f"[maintenance] Startup maintenance failed (non-fatal): {e}")

        try:
            while self.cycle_count < self.max_cycles:
                result = await self.run_cycle()

                self._log(f"\nCycle {self.cycle_count} outcome: {result.get('outcome')}")

                if result.get("outcome") == "success":
                    self._log("Continuing to next cycle...")
                else:
                    self._log("Cycle did not complete successfully.")
                    if self.require_human_approval and not self.auto_commit:
                        # Check if auto-continue is enabled or running non-interactively
                        if NOMIC_AUTO_CONTINUE:
                            self._log("[auto] Auto-continuing (NOMIC_AUTO_CONTINUE=1)")
                        elif not sys.stdin.isatty():
                            self._log("[auto] Non-interactive mode detected, continuing...")
                        else:
                            try:
                                response = input("Continue to next cycle? [Y/n]: ")
                                if response.lower() == 'n':
                                    break
                            except EOFError:
                                # Running in background/non-interactive mode
                                self._log("Non-interactive mode detected, continuing...")
                    else:
                        self._log("Auto-commit mode: continuing to next cycle...")

                await asyncio.sleep(2)

        except KeyboardInterrupt:
            self._log("\n\nNomic loop interrupted by user.")

        self._log("\n" + "=" * 70)
        self._log("NOMIC LOOP COMPLETE")
        self._log(f"Total cycles: {self.cycle_count}")
        self._log(f"Successful commits: {sum(1 for h in self.history if h.get('outcome') == 'success')}")
        self._log("=" * 70)

        return self.history


# =============================================================================
# CLI Commands for backup management
# =============================================================================

def list_backups(aragora_path: Path) -> None:
    """List available backups."""
    backup_dir = aragora_path / ".nomic" / "backups"
    if not backup_dir.exists():
        print("No backups directory found.")
        return

    backups = sorted(backup_dir.iterdir(), reverse=True)
    if not backups:
        print("No backups found.")
        return

    print(f"Available backups in {backup_dir}:")
    for backup in backups:
        manifest_file = backup / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            print(f"  {backup.name}")
            print(f"    Created: {manifest.get('created_at')}")
            print(f"    Reason: {manifest.get('reason')}")
            print(f"    Files: {len(manifest.get('files', []))}")
        else:
            print(f"  {backup.name} (no manifest)")


def restore_backup_cli(aragora_path: Path, backup_name: str = None) -> bool:
    """Restore from a specific backup or the latest one."""
    backup_dir = aragora_path / ".nomic" / "backups"

    if backup_name:
        backup_path = backup_dir / backup_name
        if not backup_path.exists():
            print(f"Backup not found: {backup_name}")
            return False
    else:
        # Find latest backup
        backups = sorted(backup_dir.iterdir(), reverse=True)
        backup_path = None
        for b in backups:
            if (b / "manifest.json").exists():
                backup_path = b
                break
        if not backup_path:
            print("No valid backups found.")
            return False

    manifest_file = backup_path / "manifest.json"
    with open(manifest_file) as f:
        manifest = json.load(f)

    print(f"Restoring backup: {backup_path.name}")
    print(f"  Created: {manifest.get('created_at')}")
    print(f"  Reason: {manifest.get('reason')}")

    restored = []
    for rel_path in manifest["files"]:
        src = backup_path / rel_path
        dst = aragora_path / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(rel_path)
            print(f"  Restored: {rel_path}")

    print(f"\nRestored {len(restored)} files")
    return True


async def main():
    parser = argparse.ArgumentParser(description="Aragora Nomic Loop - Self-improvement cycle")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run subcommand (default)
    run_parser = subparsers.add_parser("run", help="Run the nomic loop")
    run_parser.add_argument("--cycles", type=int, default=3, help="Maximum cycles to run")
    run_parser.add_argument("--auto", action="store_true", help="Auto-commit without human approval")
    run_parser.add_argument("--path", type=str, help="Path to aragora repository")
    run_parser.add_argument(
        "--proposal", "-p", type=str,
        help="Initial proposal for agents to consider (they may adopt, improve, or reject it)"
    )
    run_parser.add_argument(
        "--proposal-file", "-f", type=str,
        help="File containing initial proposal (alternative to --proposal)"
    )
    run_parser.add_argument(
        "--genesis", action="store_true",
        help="Enable genesis mode: fractal debates with agent evolution"
    )
    run_parser.add_argument(
        "--no-rollback", action="store_true",
        help="Disable rollback on verification failure (keep changes for inspection)"
    )
    run_parser.add_argument(
        "--no-stream", action="store_true",
        help="DISCOURAGED: Run without live streaming. Use 'python scripts/run_nomic_with_stream.py run' instead."
    )

    # Backup management subcommands
    list_parser = subparsers.add_parser("list-backups", help="List available backups")
    list_parser.add_argument("--path", type=str, help="Path to aragora repository")

    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("--path", type=str, help="Path to aragora repository")
    restore_parser.add_argument("--backup", type=str, help="Specific backup name (default: latest)")

    # Legacy arguments for backward compatibility (when no subcommand specified)
    parser.add_argument("--cycles", type=int, default=3, help="Maximum cycles to run")
    parser.add_argument("--auto", action="store_true", help="Auto-commit without human approval")
    parser.add_argument("--path", type=str, help="Path to aragora repository")
    parser.add_argument(
        "--proposal", "-p", type=str,
        help="Initial proposal for agents to consider (they may adopt, improve, or reject it)"
    )
    parser.add_argument(
        "--proposal-file", "-f", type=str,
        help="File containing initial proposal (alternative to --proposal)"
    )
    parser.add_argument(
        "--genesis", action="store_true",
        help="Enable genesis mode: fractal debates with agent evolution"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="DISCOURAGED: Run without live streaming. Use 'python scripts/run_nomic_with_stream.py run' instead."
    )

    args = parser.parse_args()

    # Determine aragora path
    aragora_path = Path(args.path) if args.path else Path(__file__).parent.parent

    # Handle subcommands
    if args.command == "list-backups":
        list_backups(aragora_path)
        return

    if args.command == "restore":
        restore_backup_cli(aragora_path, getattr(args, 'backup', None))
        return

    # Default: run the nomic loop (either "run" subcommand or no subcommand)
    no_stream = getattr(args, 'no_stream', False)

    # ENFORCE STREAMING: Redirect to run_nomic_with_stream.py unless --no-stream is specified
    if not no_stream:
        print("=" * 70)
        print("STREAMING IS REQUIRED")
        print("=" * 70)
        print()
        print("The nomic loop MUST stream to live.aragora.ai for transparency.")
        print()
        print("Please use the streaming script instead:")
        print()
        print("    python scripts/run_nomic_with_stream.py run --cycles 3")
        print()
        print("This ensures that all nomic loop activity is visible in real-time")
        print("at https://live.aragora.ai")
        print()
        print("If you MUST run without streaming (not recommended), use:")
        print()
        print("    python scripts/nomic_loop.py run --no-stream --cycles 3")
        print()
        print("=" * 70)
        sys.exit(1)

    # If --no-stream is specified, show warning and continue
    print("=" * 70)
    print("WARNING: Running WITHOUT live streaming")
    print("=" * 70)
    print()
    print("Activity will NOT be visible at https://live.aragora.ai")
    print("This is strongly discouraged for transparency reasons.")
    print()
    print("Press Ctrl+C within 5 seconds to cancel...")
    print()
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled. Use 'python scripts/run_nomic_with_stream.py run' instead.")
        sys.exit(0)
    print("Continuing without streaming...")
    print("=" * 70)
    print()

    # Check for OpenRouter fallback configuration
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("=" * 70)
        print("WARNING: OPENROUTER_API_KEY not set")
        print("=" * 70)
        print()
        print("OpenRouter fallback is DISABLED. If CLI agents hit rate limits,")
        print("they will fail instead of falling back to OpenRouter API.")
        print()
        print("To enable fallback, set OPENROUTER_API_KEY in your environment:")
        print()
        print("    export OPENROUTER_API_KEY=your_key_here")
        print()
        print("Get a key at: https://openrouter.ai/keys")
        print()
        print("=" * 70)
        print()

    initial_proposal = getattr(args, 'proposal', None)
    if hasattr(args, 'proposal_file') and args.proposal_file:
        with open(args.proposal_file) as f:
            initial_proposal = f.read()

    use_genesis = getattr(args, 'genesis', False)

    loop = NomicLoop(
        aragora_path=args.path,
        max_cycles=args.cycles,
        require_human_approval=not args.auto,
        auto_commit=args.auto,
        initial_proposal=initial_proposal,
        use_genesis=use_genesis,
    )

    if use_genesis:
        print("Genesis mode enabled: fractal debates with agent evolution")

    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
