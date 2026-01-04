"""
Nomic Loop Feature Integration Hub.

Coordinates all advanced features for the nomic loop:
- Bayesian belief propagation for probabilistic debate analysis
- Capability probing for agent reliability weighting
- Evidence staleness detection for revalidation triggers
- Counterfactual branching for deadlock resolution
- Checkpointing for crash recovery and resume

This module provides a unified interface for the nomic loop to leverage
all aragora features in a coordinated way.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

from aragora.core import Agent, DebateResult, Vote
from aragora.reasoning.claims import ClaimsKernel, TypedClaim, ClaimType
from aragora.reasoning.belief import (
    BeliefNetwork,
    BeliefNode,
    BeliefDistribution,
    BeliefStatus,
    BeliefPropagationAnalyzer,
)
from aragora.reasoning.provenance_enhanced import (
    EnhancedProvenanceManager,
    StalenessCheck,
    StalenessStatus,
    RevalidationTrigger,
)
from aragora.modes.prober import (
    CapabilityProber,
    VulnerabilityReport,
    ProbeResult,
    VulnerabilitySeverity,
)
from aragora.debate.counterfactual import (
    CounterfactualOrchestrator,
    CounterfactualBranch,
    ConditionalConsensus,
    PivotClaim,
    ImpactDetector,
    CounterfactualStatus,
)
from aragora.debate.checkpoint import (
    CheckpointManager,
    DebateCheckpoint,
    CheckpointStore,
    FileCheckpointStore,
    CheckpointConfig,
    CheckpointStatus,
    ResumedDebate,
    AgentState,
)


@dataclass
class BeliefAnalysis:
    """Result of analyzing a debate with belief propagation."""

    network: BeliefNetwork
    posteriors: dict[str, BeliefDistribution]
    centralities: dict[str, float]
    contested_claims: list[BeliefNode]
    crux_claims: list[BeliefNode]  # High-centrality contested claims
    convergence_achieved: bool
    iterations_used: int
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_deadlock(self) -> bool:
        """Check if there are contested crux claims that could cause deadlock."""
        return len(self.crux_claims) > 0

    @property
    def top_crux(self) -> Optional[BeliefNode]:
        """Get the most important contested claim."""
        if self.crux_claims:
            return max(self.crux_claims, key=lambda n: self.centralities.get(n.claim_id, 0))
        return None

    def to_dict(self) -> dict:
        return {
            "network_size": len(self.network.nodes),
            "contested_count": len(self.contested_claims),
            "crux_count": len(self.crux_claims),
            "convergence_achieved": self.convergence_achieved,
            "iterations_used": self.iterations_used,
            "posteriors": {k: v.to_dict() for k, v in self.posteriors.items()},
            "centralities": self.centralities,
            "timestamp": self.analysis_timestamp.isoformat(),
        }


@dataclass
class AgentReliability:
    """Agent reliability assessment from capability probing."""

    agent_name: str
    weight: float  # 0.0 to 1.0 multiplier for consensus
    vulnerability_report: Optional[VulnerabilityReport]
    probe_results: list[ProbeResult]
    probed_at: datetime = field(default_factory=datetime.now)

    @property
    def is_reliable(self) -> bool:
        """Check if agent passed reliability threshold."""
        return self.weight >= 0.7

    @property
    def critical_vulnerabilities(self) -> int:
        """Count of critical vulnerabilities found."""
        if self.vulnerability_report:
            return self.vulnerability_report.critical_count
        return 0


@dataclass
class StalenessReport:
    """Report of stale evidence in claims."""

    stale_claims: list[TypedClaim]
    staleness_checks: dict[str, StalenessCheck]  # claim_id -> check
    revalidation_triggers: list[RevalidationTrigger]
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def needs_redebate(self) -> bool:
        """Check if any stale claims require re-debate."""
        return any(
            trigger.severity in ("high", "critical")
            for trigger in self.revalidation_triggers
        )

    @property
    def stale_claim_ids(self) -> list[str]:
        """Get IDs of stale claims."""
        return [c.claim_id for c in self.stale_claims]


@dataclass
class PhaseCheckpoint:
    """Checkpoint for a nomic loop phase."""

    phase: str  # "debate", "design", "implement", "verify", "commit"
    cycle: int
    state: dict[str, Any]
    checkpoint: DebateCheckpoint
    created_at: datetime = field(default_factory=datetime.now)


class NomicIntegration:
    """
    Coordinates all advanced features for the nomic loop.

    This class provides a unified interface that the nomic loop can use
    to leverage belief propagation, capability probing, staleness detection,
    counterfactual branching, and checkpointing in a coordinated way.

    Example usage:
        integration = NomicIntegration(elo_system=elo)

        # After debate phase
        analysis = await integration.analyze_debate(debate_result)
        if analysis.has_deadlock:
            conditional = await integration.resolve_deadlock(analysis.crux_claims)

        # Before design phase
        weights = await integration.probe_agents(design_agents)

        # After implement phase
        staleness = await integration.check_staleness(claims, changed_files)

        # After each phase
        await integration.checkpoint("debate", state_dict)
    """

    def __init__(
        self,
        elo_system=None,
        checkpoint_dir: Optional[Path] = None,
        enable_probing: bool = True,
        enable_belief_analysis: bool = True,
        enable_staleness_check: bool = True,
        enable_counterfactual: bool = True,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize the integration hub.

        Args:
            elo_system: Optional EloSystem for agent ranking integration
            checkpoint_dir: Directory for checkpoint storage (default: .nomic/checkpoints)
            enable_probing: Whether to enable capability probing
            enable_belief_analysis: Whether to enable belief propagation analysis
            enable_staleness_check: Whether to enable evidence staleness detection
            enable_counterfactual: Whether to enable counterfactual branching
            enable_checkpointing: Whether to enable phase checkpointing
        """
        self.elo_system = elo_system

        # Feature flags
        self.enable_probing = enable_probing
        self.enable_belief_analysis = enable_belief_analysis
        self.enable_staleness_check = enable_staleness_check
        self.enable_counterfactual = enable_counterfactual
        self.enable_checkpointing = enable_checkpointing

        # Initialize components
        self.prober = CapabilityProber(elo_system) if enable_probing else None
        self.provenance = EnhancedProvenanceManager() if enable_staleness_check else None
        self.counterfactual = CounterfactualOrchestrator() if enable_counterfactual else None

        # Checkpoint manager with file store
        if enable_checkpointing:
            checkpoint_path = checkpoint_dir or Path(".nomic/checkpoints")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            store = FileCheckpointStore(checkpoint_path)
            self.checkpoint_mgr = CheckpointManager(store=store)
        else:
            self.checkpoint_mgr = None

        # State tracking
        self._current_debate_id: Optional[str] = None
        self._current_cycle: int = 0
        self._belief_network: Optional[BeliefNetwork] = None
        self._agent_weights: dict[str, float] = {}
        self._phase_checkpoints: list[PhaseCheckpoint] = []

    async def analyze_debate(
        self,
        result: DebateResult,
        claims_kernel: Optional[ClaimsKernel] = None,
        disagreement_threshold: float = 0.6,
        centrality_threshold: float = 0.3,
    ) -> BeliefAnalysis:
        """
        Analyze debate results with belief propagation.

        Converts debate votes and critiques into a belief network,
        runs loopy BP to compute posteriors, and identifies contested
        claims that could cause deadlock.

        Args:
            result: The DebateResult from Arena.run()
            claims_kernel: Optional ClaimsKernel with typed claims
            disagreement_threshold: Threshold for marking claims as contested
            centrality_threshold: Minimum centrality for crux claims

        Returns:
            BeliefAnalysis with posteriors, centralities, and contested claims
        """
        if not self.enable_belief_analysis:
            # Return empty analysis if disabled
            return BeliefAnalysis(
                network=BeliefNetwork(),
                posteriors={},
                centralities={},
                contested_claims=[],
                crux_claims=[],
                convergence_achieved=True,
                iterations_used=0,
            )

        # Build belief network from debate
        from aragora.reasoning.claims import RelationType
        network = BeliefNetwork()

        # Add nodes for each agent's final proposal
        for vote in result.votes:
            claim_id = f"proposal_{vote.agent}"
            claim = TypedClaim(
                claim_id=claim_id,
                claim_type=ClaimType.PROPOSAL,
                statement=f"Proposal from {vote.agent}",
                author=vote.agent,
                confidence=vote.confidence,
            )
            network.add_node_from_claim(claim, prior_confidence=vote.confidence)

        # Add factors based on votes (who agrees with whom)
        vote_choices = {}
        for vote in result.votes:
            vote_choices[vote.agent] = vote.choice

        for voter, choice in vote_choices.items():
            voter_claim_id = f"proposal_{voter}"
            choice_claim_id = f"proposal_{choice}"
            # Add supporting factor if both exist
            network.add_factor(
                voter_claim_id, choice_claim_id,
                relation_type=RelationType.SUPPORTS,
                strength=0.8,
            )

        # Add nodes from claims kernel if provided
        if claims_kernel:
            for claim in claims_kernel.claims.values():
                if claim.claim_id not in network.claim_to_node:
                    network.add_node_from_claim(claim)

            # Add factors from kernel relations
            for relation in claims_kernel.relations:
                rel_type = relation.relation_type if isinstance(relation.relation_type, RelationType) else RelationType.SUPPORTS
                network.add_factor(
                    relation.source_id,
                    relation.target_id,
                    relation_type=rel_type,
                    strength=0.7,
                )

        # Run belief propagation
        prop_result = network.propagate()

        # Extract results from PropagationResult
        posteriors = prop_result.node_posteriors
        converged = prop_result.converged
        iterations = prop_result.iterations
        centralities = prop_result.centralities

        # Identify contested claims
        analyzer = BeliefPropagationAnalyzer(network)
        contested = []
        crux = []

        for node_id, node in network.nodes.items():
            posterior = posteriors.get(node_id)
            if posterior:
                # Check if contested (high entropy = uncertain)
                if posterior.entropy > disagreement_threshold:
                    contested.append(node)
                    node.status = BeliefStatus.CONTESTED

                    # Check if also a crux (contested + high centrality)
                    centrality = centralities.get(node_id, 0)
                    if centrality >= centrality_threshold:
                        crux.append(node)

        # Store for later use
        self._belief_network = network
        self._current_debate_id = result.id

        return BeliefAnalysis(
            network=network,
            posteriors=posteriors,
            centralities=centralities,
            contested_claims=contested,
            crux_claims=crux,
            convergence_achieved=converged,
            iterations_used=iterations,
        )

    async def probe_agents(
        self,
        agents: list[Agent],
        probe_count: int = 3,
        min_weight: float = 0.5,
    ) -> dict[str, float]:
        """
        Probe agents for reliability and return consensus weights.

        Runs capability probes (contradiction traps, hallucination bait, etc.)
        against each agent and computes a reliability weight based on results.

        Args:
            agents: List of agents to probe
            probe_count: Number of probes per agent
            min_weight: Minimum weight even for unreliable agents

        Returns:
            Dict mapping agent name to consensus weight (0.0 to 1.0)
        """
        if not self.enable_probing or not self.prober:
            # Return uniform weights if disabled
            return {agent.name: 1.0 for agent in agents}

        weights = {}

        for agent in agents:
            try:
                # Run probes
                report = await self.prober.probe_agent(
                    agent,
                    probe_count=probe_count,
                )

                # Calculate weight based on vulnerability rate
                # weight = 1 - vulnerability_rate, but at least min_weight
                vulnerability_rate = report.vulnerability_rate
                weight = max(min_weight, 1.0 - vulnerability_rate)

                # Further reduce weight for critical vulnerabilities
                if report.critical_count > 0:
                    weight *= 0.5

                weights[agent.name] = weight

            except Exception as e:
                # If probing fails, assume moderate reliability
                weights[agent.name] = 0.75

        self._agent_weights = weights
        return weights

    async def check_staleness(
        self,
        claims: list[TypedClaim],
        changed_files: list[str],
        repo_path: Optional[Path] = None,
    ) -> StalenessReport:
        """
        Check which claims have stale evidence due to code changes.

        Cross-references claims with changed files to identify evidence
        that may be invalid due to underlying code modifications.

        Args:
            claims: List of claims to check
            changed_files: List of files that changed (from git diff)
            repo_path: Optional path to git repository

        Returns:
            StalenessReport with stale claims and revalidation triggers
        """
        if not self.enable_staleness_check or not self.provenance:
            return StalenessReport(
                stale_claims=[],
                staleness_checks={},
                revalidation_triggers=[],
            )

        stale_claims = []
        staleness_checks = {}
        triggers = []

        for claim in claims:
            # Check if claim references any changed files
            claim_files = self._extract_file_references(claim)
            affected_files = set(claim_files) & set(changed_files)

            if affected_files:
                # Run staleness check
                check = await self.provenance.check_staleness(
                    claim.claim_id,
                    list(affected_files),
                    repo_path=repo_path,
                )

                staleness_checks[claim.claim_id] = check

                if check.status == StalenessStatus.STALE:
                    stale_claims.append(claim)

                    # Create revalidation trigger
                    severity = "high" if claim.claim_type == ClaimType.FACTUAL else "medium"
                    trigger = RevalidationTrigger(
                        claim_id=claim.claim_id,
                        reason=f"Referenced files changed: {', '.join(affected_files)}",
                        severity=severity,
                        changed_files=list(affected_files),
                    )
                    triggers.append(trigger)

        return StalenessReport(
            stale_claims=stale_claims,
            staleness_checks=staleness_checks,
            revalidation_triggers=triggers,
        )

    def _extract_file_references(self, claim: TypedClaim) -> list[str]:
        """Extract file paths referenced in a claim's evidence."""
        files = []

        # Check claim statement for file patterns
        import re
        # Match common file path patterns
        path_pattern = r'(?:[\w./\\-]+\.(?:py|ts|js|tsx|jsx|json|yaml|yml|md|txt))'
        text = claim.statement if hasattr(claim, 'statement') else getattr(claim, 'text', '')
        matches = re.findall(path_pattern, text)
        files.extend(matches)

        # Check evidence sources
        if hasattr(claim, 'evidence') and claim.evidence:
            for evidence in claim.evidence:
                if hasattr(evidence, 'source') and evidence.source:
                    if evidence.source.startswith('/') or '.' in evidence.source:
                        files.append(evidence.source)

        return list(set(files))

    async def resolve_deadlock(
        self,
        contested_claims: list[BeliefNode],
        arena=None,  # Optional Arena for running branches
    ) -> Optional[ConditionalConsensus]:
        """
        Resolve debate deadlock by forking on contested claims.

        Creates counterfactual branches exploring different assumptions
        for the most contested claim, runs debates in each branch,
        and synthesizes a conditional consensus.

        Args:
            contested_claims: List of contested BeliefNodes (crux claims)
            arena: Optional Arena instance for running branch debates

        Returns:
            ConditionalConsensus if deadlock was resolved, None otherwise
        """
        if not self.enable_counterfactual or not self.counterfactual:
            return None

        if not contested_claims:
            return None

        # Find most important contested claim
        if self._belief_network:
            centralities = self._belief_network.compute_centrality()
            pivot_node = max(
                contested_claims,
                key=lambda n: centralities.get(n.claim_id, 0)
            )
        else:
            pivot_node = contested_claims[0]

        # Create pivot claim
        pivot = PivotClaim(
            claim_id=pivot_node.claim_id,
            text=pivot_node.claim.text if pivot_node.claim else "Unknown claim",
            current_belief=pivot_node.belief.expected_truth if pivot_node.belief else 0.5,
        )

        # Create and run branches
        branches = await self.counterfactual.create_branches(
            pivot=pivot,
            branch_values=[True, False],  # Explore both assuming true and false
        )

        # If arena provided, run debates in each branch
        if arena and branches:
            for branch in branches:
                try:
                    # Clone arena with modified context for this branch
                    branch_result = await self.counterfactual.run_branch(
                        branch,
                        arena=arena,
                    )
                    branch.result = branch_result
                except Exception as e:
                    branch.status = CounterfactualStatus.FAILED
                    branch.error = str(e)

        # Synthesize conditional consensus
        if branches:
            return await self.counterfactual.synthesize_branches(
                branches,
                pivot=pivot,
            )

        return None

    async def full_post_debate_analysis(
        self,
        result: DebateResult,
        arena=None,
        claims_kernel=None,
        changed_files: list[str] = None,
    ) -> dict[str, Any]:
        """
        Run all post-debate analyses in one unified call.

        This is the primary entry point for nomic loop post-debate processing.
        Runs belief analysis, deadlock resolution, and staleness checking
        in proper order with dependencies handled.

        Args:
            result: The DebateResult from Arena.run()
            arena: Optional Arena instance for running counterfactual branches
            claims_kernel: Optional ClaimsKernel with extracted claims
            changed_files: Optional list of files that changed (for staleness)

        Returns:
            Dict with keys: 'belief', 'conditional', 'staleness', 'summary'
        """
        analysis = {
            "belief": None,
            "conditional": None,
            "staleness": None,
            "summary": {
                "has_deadlock": False,
                "needs_redebate": False,
                "contested_count": 0,
                "crux_count": 0,
                "stale_count": 0,
            }
        }

        # 1. Belief analysis (always run)
        analysis["belief"] = await self.analyze_debate(result, claims_kernel)
        analysis["summary"]["contested_count"] = len(analysis["belief"].contested_claims)
        analysis["summary"]["crux_count"] = len(analysis["belief"].crux_claims)
        analysis["summary"]["has_deadlock"] = analysis["belief"].has_deadlock

        # 2. Deadlock resolution (if needed and arena provided)
        if analysis["belief"].has_deadlock:
            analysis["conditional"] = await self.resolve_deadlock(
                analysis["belief"].crux_claims,
                arena=arena
            )

        # 3. Staleness check (if files changed and claims available)
        if changed_files and claims_kernel:
            claims_list = list(claims_kernel.claims.values()) if hasattr(claims_kernel, 'claims') else []
            if claims_list:
                analysis["staleness"] = await self.check_staleness(claims_list, changed_files)
                analysis["summary"]["stale_count"] = len(analysis["staleness"].stale_claims)
                analysis["summary"]["needs_redebate"] = analysis["staleness"].needs_redebate

        return analysis

    async def checkpoint(
        self,
        phase: str,
        state: dict[str, Any],
        debate_id: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> Optional[str]:
        """
        Create a checkpoint for the current phase.

        Saves the complete state needed to resume from this point,
        including debate state, agent states, and feature-specific state.

        Args:
            phase: Phase name ("debate", "design", "implement", "verify", "commit")
            state: Phase state dictionary
            debate_id: Optional debate ID (uses current if not provided)
            cycle: Optional cycle number (uses current if not provided)

        Returns:
            Checkpoint ID if successful, None otherwise
        """
        if not self.enable_checkpointing or not self.checkpoint_mgr:
            return None

        debate_id = debate_id or self._current_debate_id or "unknown"
        cycle = cycle if cycle is not None else self._current_cycle

        # Build complete checkpoint state
        checkpoint_state = {
            "phase": phase,
            "cycle": cycle,
            "state": state,
            "belief_network": self._belief_network.to_dict() if self._belief_network else None,
            "agent_weights": self._agent_weights,
        }

        try:
            # Create checkpoint
            checkpoint = await self.checkpoint_mgr.create_checkpoint(
                debate_id=debate_id,
                task=f"nomic_loop_cycle_{cycle}_{phase}",
                current_round=cycle,
                phase=phase,
                messages=[],  # Messages are in state dict
                critiques=[],
                votes=[],
                agents=[],
                extra_state=checkpoint_state,
            )

            # Track phase checkpoint
            phase_checkpoint = PhaseCheckpoint(
                phase=phase,
                cycle=cycle,
                state=state,
                checkpoint=checkpoint,
            )
            self._phase_checkpoints.append(phase_checkpoint)

            return checkpoint.checkpoint_id

        except Exception as e:
            # Log but don't fail on checkpoint errors
            print(f"Checkpoint failed for {phase}: {e}")
            return None

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Optional[PhaseCheckpoint]:
        """
        Resume from a previous checkpoint.

        Restores state from a checkpoint, allowing the nomic loop
        to resume from a previous phase.

        Args:
            checkpoint_id: ID of checkpoint to resume from

        Returns:
            PhaseCheckpoint with restored state, or None if not found
        """
        if not self.enable_checkpointing or not self.checkpoint_mgr:
            return None

        try:
            resumed = await self.checkpoint_mgr.resume_from_checkpoint(checkpoint_id)

            if resumed:
                checkpoint = resumed.checkpoint
                extra = checkpoint.extra_state or {}

                # Restore integration state
                if extra.get("belief_network"):
                    self._belief_network = BeliefNetwork.from_dict(extra["belief_network"])
                self._agent_weights = extra.get("agent_weights", {})
                self._current_debate_id = checkpoint.debate_id
                self._current_cycle = extra.get("cycle", 0)

                return PhaseCheckpoint(
                    phase=checkpoint.phase,
                    cycle=self._current_cycle,
                    state=extra.get("state", {}),
                    checkpoint=checkpoint,
                )

        except Exception as e:
            print(f"Resume failed: {e}")

        return None

    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
    ) -> list[dict]:
        """List available checkpoints."""
        if not self.checkpoint_mgr:
            return []

        return await self.checkpoint_mgr.list_debates_with_checkpoints(
            debate_id=debate_id or self._current_debate_id
        )

    def get_agent_weights(self) -> dict[str, float]:
        """Get current agent reliability weights."""
        return self._agent_weights.copy()

    def set_cycle(self, cycle: int):
        """Set the current nomic loop cycle."""
        self._current_cycle = cycle

    def set_debate_id(self, debate_id: str):
        """Set the current debate ID."""
        self._current_debate_id = debate_id


# Convenience function for creating integration with defaults
def create_nomic_integration(
    elo_system=None,
    checkpoint_dir: Optional[str] = None,
    **feature_flags,
) -> NomicIntegration:
    """
    Create a NomicIntegration instance with sensible defaults.

    Args:
        elo_system: Optional EloSystem for agent ranking
        checkpoint_dir: Directory for checkpoints (default: .nomic/checkpoints)
        **feature_flags: Override feature flags (enable_probing, etc.)

    Returns:
        Configured NomicIntegration instance
    """
    return NomicIntegration(
        elo_system=elo_system,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        **feature_flags,
    )
