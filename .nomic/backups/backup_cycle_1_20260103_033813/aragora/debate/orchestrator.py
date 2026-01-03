"""
Multi-agent debate orchestrator.

Implements the propose -> critique -> revise loop with configurable
debate protocols and consensus mechanisms.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
from collections import Counter

from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.convergence import (
    ConvergenceDetector,
    ConvergenceResult,
    get_similarity_backend,
)
from aragora.spectate.stream import SpectatorStream
from aragora.audience.suggestions import cluster_suggestions, format_for_prompt

# Lazy import to avoid circular dependencies
InsightExtractor = None
InsightStore = None

def _get_insight_classes():
    """Lazy-load insight classes to avoid circular imports."""
    global InsightExtractor, InsightStore
    if InsightExtractor is None:
        from aragora.insights import InsightExtractor as _IE, InsightStore as _IS
        InsightExtractor = _IE
        InsightStore = _IS
    return InsightExtractor, InsightStore


@dataclass
class DebateProtocol:
    """Configuration for how debates are conducted."""

    topology: Literal["all-to-all", "sparse", "round-robin", "ring", "star", "random-graph"] = "round-robin"
    topology_sparsity: float = 0.5  # fraction of possible critique connections (for sparse/random-graph)
    topology_hub_agent: Optional[str] = None  # for star topology, which agent is the hub
    rounds: int = 3
    consensus: Literal["majority", "unanimous", "judge", "none"] = "majority"
    consensus_threshold: float = 0.6  # fraction needed for majority
    allow_abstain: bool = True
    require_reasoning: bool = True

    # Role assignments
    proposer_count: int = 1  # how many agents propose initially
    critic_count: int = -1  # -1 means all non-proposers critique

    # Judge selection (for consensus="judge" mode)
    judge_selection: Literal["random", "voted", "last"] = "random"

    # Agreement intensity (0-10): Controls how much agents agree vs disagree
    # 0 = strongly disagree 100% of the time (adversarial)
    # 5 = balanced (agree/disagree based on argument quality)
    # 10 = fully incorporate others' opinions (collaborative)
    # Research shows intensity=2 (slight disagreement bias) often improves accuracy
    agreement_intensity: int = 5

    # Early stopping: End debate when agents agree further rounds won't help
    # Based on ai-counsel pattern - can save 40-70% API costs
    early_stopping: bool = True
    early_stop_threshold: float = 0.66  # fraction of agents saying stop to trigger
    min_rounds_before_early_stop: int = 1  # minimum rounds before allowing early exit

    # Asymmetric debate roles: Assign affirmative/negative/neutral stances
    # Forces perspective diversity, prevents premature consensus
    asymmetric_stances: bool = False  # Enable asymmetric stance assignment
    rotate_stances: bool = True  # Rotate stances between rounds

    # Semantic convergence detection
    # Auto-detect consensus without explicit voting
    convergence_detection: bool = True
    convergence_threshold: float = 0.85  # Similarity for convergence
    divergence_threshold: float = 0.40  # Below this is diverging

    # Vote option grouping: Merge semantically similar vote choices
    # Prevents artificial disagreement from wording variations
    vote_grouping: bool = True
    vote_grouping_threshold: float = 0.85  # Similarity to merge options

    # Judge-based termination: Single judge decides when debate is conclusive
    # Different from early_stopping (agent votes) - uses a designated judge
    judge_termination: bool = False
    min_rounds_before_judge_check: int = 2  # Check only after this many rounds

    # Human participation settings
    user_vote_weight: float = 0.5  # Weight of user votes relative to agent votes (0.5 = half weight)

    # Audience suggestion injection
    audience_injection: Literal["off", "summary", "inject"] = "off"

    # Pre-debate web research
    enable_research: bool = False  # Enable web research before debates

    # Agent introspection: Inject agent's own reputation/history into prompts
    # Enables agents to reason about their own track record during debates
    enable_introspection: bool = False


class Arena:
    """
    Orchestrates multi-agent debates.

    The Arena manages the flow of a debate:
    1. Proposers generate initial proposals
    2. Critics critique each proposal
    3. Proposers revise based on critique
    4. Repeat for configured rounds
    5. Consensus mechanism selects final answer
    """

    def __init__(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol = None,
        memory=None,  # CritiqueStore instance
        event_hooks: dict = None,  # Optional hooks for streaming events
        event_emitter=None,  # Optional event emitter for subscribing to user events
        spectator: SpectatorStream = None,  # Optional spectator stream for real-time events
        debate_embeddings=None,  # DebateEmbeddingsDatabase for historical context
        insight_store=None,  # Optional InsightStore for extracting learnings from debates
        persona_manager=None,  # Optional PersonaManager for agent introspection
    ):
        self.env = environment
        self.agents = agents
        self.protocol = protocol or DebateProtocol()
        self.memory = memory
        self.hooks = event_hooks or {}
        self.event_emitter = event_emitter
        self.spectator = spectator or SpectatorStream(enabled=False)
        self.debate_embeddings = debate_embeddings
        self.insight_store = insight_store
        self.persona_manager = persona_manager

        # User participation tracking
        self.user_votes: list[dict] = []  # List of user vote events
        self.user_suggestions: list[dict] = []  # List of user suggestion events

        # Subscribe to user participation events if emitter provided
        if event_emitter:
            from aragora.server.stream import StreamEventType
            event_emitter.subscribe(self._handle_user_event)

        # Assign roles if not already set
        self._assign_roles()

        # Assign initial stances for asymmetric debate
        self._assign_stances(round_num=0)

        # Apply agreement intensity guidance to all agents
        self._apply_agreement_intensity()

        # Initialize convergence detector if enabled
        self.convergence_detector = None
        if self.protocol.convergence_detection:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=self.protocol.convergence_threshold,
                divergence_threshold=self.protocol.divergence_threshold,
                min_rounds_before_check=1,
            )

        # Track responses for convergence detection
        self._previous_round_responses: dict[str, str] = {}

        # Cache for historical context (computed once per debate)
        self._historical_context_cache: str = ""

        # Cache for research context (computed once per debate)
        self._research_context_cache: Optional[str] = None

        # Cache for agent introspection data (computed once per debate)
        self._introspection_cache = None
        if self.protocol.enable_introspection:
            from aragora.introspection import IntrospectionCache
            self._introspection_cache = IntrospectionCache()
            self._introspection_cache.warm(
                agents=self.agents,
                memory=self.memory,
                persona_manager=self.persona_manager,
            )

    def _select_critics_for_proposal(self, proposal_agent: str, all_critics: list[Agent]) -> list[Agent]:
        """Select which critics should critique the given proposal based on topology."""
        if self.protocol.topology == "all-to-all":
            # All critics except self
            return [c for c in all_critics if c.name != proposal_agent]

        elif self.protocol.topology == "round-robin":
            # Simple round-robin: each critic critiques the next one in alphabetical order
            # First, filter out the proposer from critics
            eligible_critics = [c for c in all_critics if c.name != proposal_agent]
            if not eligible_critics:
                return []
            # Sort critics by name for deterministic ordering
            eligible_critics_sorted = sorted(eligible_critics, key=lambda c: c.name)
            # Each proposal gets critiqued by the "next" critic based on hash
            proposal_index = hash(proposal_agent) % len(eligible_critics_sorted)
            return [eligible_critics_sorted[proposal_index]]

        elif self.protocol.topology == "ring":
            # Ring topology: each agent critiques its "neighbors"
            agent_names = sorted([a.name for a in all_critics] + [proposal_agent])
            agent_names.remove(proposal_agent)
            if not agent_names:
                return []
            # Find position of proposal_agent in the ring
            all_names = sorted([a.name for a in self.agents])
            if proposal_agent not in all_names:
                return all_critics  # fallback
            idx = all_names.index(proposal_agent)
            # Critique by left and right neighbors in the ring
            left = all_names[(idx - 1) % len(all_names)]
            right = all_names[(idx + 1) % len(all_names)]
            return [c for c in all_critics if c.name in (left, right)]

        elif self.protocol.topology == "star":
            # Star topology: hub agent critiques everyone, or everyone critiques hub
            if self.protocol.topology_hub_agent:
                hub = self.protocol.topology_hub_agent
            else:
                # Default hub is first agent
                hub = self.agents[0].name
            if proposal_agent == hub:
                # Hub's proposal gets critiqued by all others
                return [c for c in all_critics if c.name != hub]
            else:
                # Others' proposals get critiqued only by hub (if hub is a critic)
                return [c for c in all_critics if c.name == hub]

        elif self.protocol.topology in ("sparse", "random-graph"):
            # Random subset based on sparsity
            available_critics = [c for c in all_critics if c.name != proposal_agent]
            if not available_critics:
                return []
            num_to_select = max(1, int(len(available_critics) * self.protocol.topology_sparsity))
            # Deterministic random based on proposal_agent for reproducibility
            random.seed(hash(proposal_agent))
            selected = random.sample(available_critics, min(num_to_select, len(available_critics)))
            random.seed()  # Reset seed
            return selected

        else:
            # Default to all-to-all
            return [c for c in all_critics if c.name != proposal_agent]

    def _handle_user_event(self, event) -> None:
        """Handle incoming user participation events."""
        from aragora.server.stream import StreamEventType

        if event.type == StreamEventType.USER_VOTE:
            self.user_votes.append(event.data)
        elif event.type == StreamEventType.USER_SUGGESTION:
            self.user_suggestions.append(event.data)

    def _notify_spectator(self, event_type: str, **kwargs):
        """Helper method to emit spectator events."""
        if self.spectator:
            self.spectator.emit(event_type, **kwargs)

    async def _fetch_historical_context(self, task: str, limit: int = 3) -> str:
        """Fetch similar past debates for historical context.

        This enables agents to learn from what worked (or didn't) in similar debates.
        """
        if not self.debate_embeddings:
            return ""

        try:
            results = await self.debate_embeddings.find_similar_debates(
                task, limit=limit, min_similarity=0.6
            )
            if not results:
                return ""

            lines = ["## HISTORICAL CONTEXT (Similar Past Debates)"]
            lines.append("Learn from these previous debates on similar topics:\n")

            for debate_id, excerpt, similarity in results:
                lines.append(f"**[{similarity:.0%} similar]** {excerpt[:400]}...")
                lines.append("")  # blank line between entries

            return "\n".join(lines)
        except Exception:
            return ""

    async def _perform_research(self, task: str) -> str:
        """Perform web research for the debate topic and return formatted context.

        Uses EvidenceCollector with WebConnector to gather relevant information.
        """
        try:
            # Check if web search dependencies are available
            try:
                from aragora.connectors.web import DDGS_AVAILABLE
                if not DDGS_AVAILABLE:
                    return "Web research unavailable: duckduckgo-search package not installed."
            except ImportError:
                return "Web research unavailable: required packages not installed."

            from aragora.evidence.collector import EvidenceCollector
            from aragora.connectors.web import WebConnector

            # Create evidence collector with web connector
            collector = EvidenceCollector()
            web_connector = WebConnector()
            collector.add_connector("web", web_connector)

            # Collect evidence for the task
            evidence_pack = await collector.collect_evidence(task, enabled_connectors=["web"])

            # Format as context string
            if evidence_pack.snippets:
                context = evidence_pack.to_context_string()
                return f"## WEB RESEARCH CONTEXT\n{context}"
            else:
                return "No relevant web research found for this topic."

        except Exception as e:
            print(f"Research failed: {e}")
            return f"Web research failed: {str(e)}"

    def _assign_roles(self):
        """Assign roles to agents based on protocol."""
        # If agents already have roles, respect them
        if all(a.role for a in self.agents):
            return

        # Otherwise assign based on protocol
        proposers_needed = self.protocol.proposer_count
        for i, agent in enumerate(self.agents):
            if i < proposers_needed:
                agent.role = "proposer"
            elif i == len(self.agents) - 1:
                agent.role = "synthesizer"
            else:
                agent.role = "critic"

    def _apply_agreement_intensity(self):
        """Apply agreement intensity guidance to all agents' system prompts.

        This modifies each agent's system_prompt to include guidance on how
        much to agree vs disagree with other agents, based on the protocol's
        agreement_intensity setting.
        """
        guidance = self._get_agreement_intensity_guidance()

        for agent in self.agents:
            if agent.system_prompt:
                agent.system_prompt = f"{agent.system_prompt}\n\n{guidance}"
            else:
                agent.system_prompt = guidance

    def _assign_stances(self, round_num: int = 0):
        """Assign debate stances to agents for asymmetric debate.

        Stances: "affirmative" (defend), "negative" (challenge), "neutral" (evaluate)
        If rotate_stances is True, stances rotate each round.
        """
        if not self.protocol.asymmetric_stances:
            return

        stances = ["affirmative", "negative", "neutral"]
        n_agents = len(self.agents)

        for i, agent in enumerate(self.agents):
            # Rotate stance based on round number if enabled
            if self.protocol.rotate_stances:
                stance_idx = (i + round_num) % len(stances)
            else:
                stance_idx = i % len(stances)

            agent.stance = stances[stance_idx]

    def _get_stance_guidance(self, agent) -> str:
        """Generate prompt guidance based on agent's debate stance."""
        if not self.protocol.asymmetric_stances:
            return ""

        if agent.stance == "affirmative":
            return """DEBATE STANCE: AFFIRMATIVE
You are assigned to DEFEND and SUPPORT proposals. Your role is to:
- Find strengths and merits in arguments
- Build upon existing ideas
- Advocate for the proposal's value
- Counter criticisms constructively
Even if you personally disagree, argue the affirmative position."""

        elif agent.stance == "negative":
            return """DEBATE STANCE: NEGATIVE
You are assigned to CHALLENGE and CRITIQUE proposals. Your role is to:
- Identify weaknesses, flaws, and risks
- Play devil's advocate
- Raise objections and counterarguments
- Stress-test the proposal
Even if you personally agree, argue the negative position."""

        else:  # neutral
            return """DEBATE STANCE: NEUTRAL
You are assigned to EVALUATE FAIRLY. Your role is to:
- Weigh arguments from both sides impartially
- Identify the strongest and weakest points
- Seek balanced synthesis
- Judge on merit, not position"""

    async def run(self) -> DebateResult:
        """Run the full debate and return results."""
        start_time = time.time()

        # Fetch historical context once at debate start (for institutional memory)
        if self.debate_embeddings:
            try:
                self._historical_context_cache = await self._fetch_historical_context(
                    self.env.task, limit=2  # Limit to 2 to avoid prompt bloat
                )
            except Exception:
                self._historical_context_cache = ""

        # Pre-debate research phase
        if self.protocol.enable_research:
            try:
                research_context = await self._perform_research(self.env.task)
                if research_context:
                    # Add research to environment context
                    if self.env.context:
                        self.env.context += "\n\n" + research_context
                    else:
                        self.env.context = research_context
            except Exception as e:
                print(f"Research phase failed: {e}")
                # Continue without research - don't break the debate

        result = DebateResult(
            task=self.env.task,
            messages=[],
            critiques=[],
            votes=[],
            dissenting_views=[],
        )

        proposals: dict[str, str] = {}
        context: list[Message] = []

        # === ROUND 0: Initial Proposals ===
        proposers = [a for a in self.agents if a.role == "proposer"]
        if not proposers:
            proposers = [self.agents[0]]  # Default to first agent

        print(f"\n{'='*60}")
        print(f"DEBATE: {self.env.task[:80]}...")
        print(f"Agents: {', '.join(a.name for a in self.agents)}")
        print(f"Rounds: {self.protocol.rounds}")
        print(f"Agreement intensity: {self.protocol.agreement_intensity}/10")
        print(f"{'='*60}\n")

        # Emit debate start event
        if "on_debate_start" in self.hooks:
            self.hooks["on_debate_start"](self.env.task, [a.name for a in self.agents])

        # Notify spectator of debate start
        self._notify_spectator("debate_start", details=f"Task: {self.env.task[:50]}...", agent="system")

        # Generate initial proposals (stream output as each completes)
        print("Round 0: Initial Proposals")
        print("-" * 40)

        # Create tasks with agent reference for streaming output
        async def generate_proposal(agent):
            """Generate proposal and return (agent, result_or_error)."""
            prompt = self._build_proposal_prompt(agent)
            print(f"  {agent.name}: generating...", flush=True)
            try:
                result = await self._generate_with_agent(agent, prompt, context)
                return (agent, result)
            except Exception as e:
                return (agent, e)

        # Use asyncio.as_completed to stream output as each agent finishes
        tasks = [asyncio.create_task(generate_proposal(agent)) for agent in proposers]

        for completed_task in asyncio.as_completed(tasks):
            agent, result_or_error = await completed_task

            if isinstance(result_or_error, Exception):
                print(f"  {agent.name}: ERROR - {result_or_error}")
                proposals[agent.name] = f"[Error generating proposal: {result_or_error}]"
            else:
                proposals[agent.name] = result_or_error
                print(f"  {agent.name}: {result_or_error}")  # Full content

                # Notify spectator of proposal
                self._notify_spectator("propose", agent=agent.name, details=f"Initial proposal ({len(result_or_error)} chars)", metric=len(result_or_error))

            msg = Message(
                role="proposer",
                agent=agent.name,
                content=proposals[agent.name],
                round=0,
            )
            context.append(msg)
            result.messages.append(msg)

            # Emit message event
            if "on_message" in self.hooks:
                self.hooks["on_message"](
                    agent=agent.name,
                    content=proposals[agent.name],
                    role="proposer",
                    round_num=0,
                )

        # === DEBATE ROUNDS ===
        for round_num in range(1, self.protocol.rounds + 1):
            print(f"\nRound {round_num}: Critique & Revise")
            print("-" * 40)

            # Notify spectator of round start
            self._notify_spectator("round", details=f"Starting Round {round_num}", agent="system")

            # Rotate stances if asymmetric debate is enabled
            if self.protocol.asymmetric_stances and self.protocol.rotate_stances:
                self._assign_stances(round_num)
                stances_str = ", ".join(f"{a.name}:{a.stance}" for a in self.agents)
                print(f"  Stances: {stances_str}")

            # Emit round start event
            if "on_round_start" in self.hooks:
                self.hooks["on_round_start"](round_num)

            # Get critics - when all agents are proposers, they all critique each other
            critics = [a for a in self.agents if a.role in ("critic", "synthesizer")]
            if not critics:
                # When no dedicated critics exist, all agents critique each other
                # The loop below already skips self-critique via "if critic.name != proposal_agent"
                critics = list(self.agents)

            # === Critique Phase (stream output as each critique completes) ===
            async def generate_critique(critic, proposal_agent, proposal):
                """Generate critique and return (critic, proposal_agent, result_or_error)."""
                print(f"  {critic.name} -> {proposal_agent}: critiquing...", flush=True)
                try:
                    crit_result = await self._critique_with_agent(critic, proposal, self.env.task, context)
                    return (critic, proposal_agent, crit_result)
                except Exception as e:
                    return (critic, proposal_agent, e)

            # Create critique tasks based on topology
            critique_tasks = []
            for proposal_agent, proposal in proposals.items():
                selected_critics = self._select_critics_for_proposal(proposal_agent, critics)
                for critic in selected_critics:
                    critique_tasks.append(
                        asyncio.create_task(generate_critique(critic, proposal_agent, proposal))
                    )

            # Stream output as each critique completes
            for completed_task in asyncio.as_completed(critique_tasks):
                critic, proposal_agent, crit_result = await completed_task

                if isinstance(crit_result, Exception):
                    print(f"  {critic.name} -> {proposal_agent}: ERROR - {crit_result}")
                else:
                    result.critiques.append(crit_result)
                    print(
                        f"  {critic.name} -> {proposal_agent}: "
                        f"{len(crit_result.issues)} issues, "
                        f"severity {crit_result.severity:.1f}"
                    )

                    # Notify spectator of critique
                    self._notify_spectator("critique", agent=critic.name, details=f"Critiqued {proposal_agent}: {len(crit_result.issues)} issues", metric=crit_result.severity)

                    # Get full critique content
                    critique_content = crit_result.to_prompt()

                    # Emit critique event with full content
                    if "on_critique" in self.hooks:
                        self.hooks["on_critique"](
                            agent=critic.name,
                            target=proposal_agent,
                            issues=crit_result.issues,
                            severity=crit_result.severity,
                            round_num=round_num,
                            full_content=critique_content,
                        )

                    # Also emit as a message for the activity feed
                    if "on_message" in self.hooks:
                        self.hooks["on_message"](
                            agent=critic.name,
                            content=critique_content,
                            role="critic",
                            round_num=round_num,
                        )

                    # Add critique to context
                    msg = Message(
                        role="critic",
                        agent=critic.name,
                        content=critique_content,
                        round=round_num,
                    )
                    context.append(msg)
                    result.messages.append(msg)

            # === Revision Phase ===
            # Get critiques for each proposer and let them revise
            for agent in proposers:
                agent_critiques = [
                    c for c in result.critiques if c.target_agent == "proposal"  # simplified
                ]

                if agent_critiques:
                    revision_prompt = self._build_revision_prompt(
                        agent, proposals[agent.name], agent_critiques[-len(critics) :]
                    )
                    try:
                        revised = await self._generate_with_agent(agent, revision_prompt, context)
                        proposals[agent.name] = revised
                        print(f"  {agent.name} revised: {revised}")  # Full content

                        # Notify spectator of revision
                        self._notify_spectator("propose", agent=agent.name, details=f"Revised proposal ({len(revised)} chars)", metric=len(revised))

                        msg = Message(
                            role="proposer",
                            agent=agent.name,
                            content=revised,
                            round=round_num,
                        )
                        context.append(msg)
                        result.messages.append(msg)

                        # Emit message event for revision
                        if "on_message" in self.hooks:
                            self.hooks["on_message"](
                                agent=agent.name,
                                content=revised,
                                role="proposer",
                                round_num=round_num,
                            )
                    except Exception as e:
                        print(f"  {agent.name} revision ERROR: {e}")

            result.rounds_used = round_num

            # === Convergence Detection ===
            # Track responses for convergence comparison
            current_responses = {agent: proposals[agent] for agent in proposals}

            if self.convergence_detector and self._previous_round_responses:
                convergence = self.convergence_detector.check_convergence(
                    current_responses, self._previous_round_responses, round_num
                )
                if convergence:
                    result.convergence_status = convergence.status
                    result.convergence_similarity = convergence.avg_similarity
                    result.per_agent_similarity = convergence.per_agent_similarity

                    print(f"  Convergence: {convergence.status} ({convergence.avg_similarity:.0%} avg)")

                    # Notify spectator of convergence
                    self._notify_spectator("convergence", details=f"{convergence.status}", metric=convergence.avg_similarity)

                    # Emit convergence event
                    if "on_convergence_check" in self.hooks:
                        self.hooks["on_convergence_check"](
                            status=convergence.status,
                            similarity=convergence.avg_similarity,
                            per_agent=convergence.per_agent_similarity,
                            round_num=round_num,
                        )

                    # Stop early if converged
                    if convergence.converged:
                        print(f"  Debate converged at round {round_num}")
                        self._previous_round_responses = current_responses
                        break

            self._previous_round_responses = current_responses

            # === Termination Checks ===
            if round_num < self.protocol.rounds:  # Only check if not last round
                # Judge-based termination (single judge decision)
                should_continue, reason = await self._check_judge_termination(
                    round_num, proposals, context
                )
                if not should_continue:
                    break  # Exit debate loop, proceed to consensus

                # Early stopping (agent votes)
                should_continue = await self._check_early_stopping(
                    round_num, proposals, context
                )
                if not should_continue:
                    break  # Exit debate loop, proceed to consensus

        # === CONSENSUS PHASE ===
        print(f"\nConsensus Phase ({self.protocol.consensus})")
        print("-" * 40)

        if self.protocol.consensus == "none":
            # No consensus - just return all proposals
            result.final_answer = "\n\n---\n\n".join(
                f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
            )
            result.consensus_reached = False
            result.confidence = 0.5

        elif self.protocol.consensus == "majority":
            # All agents vote (stream output as each vote completes)
            async def cast_vote(agent):
                """Cast vote and return (agent, result_or_error)."""
                print(f"  {agent.name}: voting...", flush=True)
                try:
                    vote_result = await self._vote_with_agent(agent, proposals, self.env.task)
                    return (agent, vote_result)
                except Exception as e:
                    return (agent, e)

            vote_tasks = [asyncio.create_task(cast_vote(agent)) for agent in self.agents]

            for completed_task in asyncio.as_completed(vote_tasks):
                agent, vote_result = await completed_task

                if isinstance(vote_result, Exception):
                    print(f"  {agent.name}: ERROR voting - {vote_result}")
                else:
                    result.votes.append(vote_result)
                    print(f"  {agent.name} votes: {vote_result.choice} ({vote_result.confidence:.0%})")

                    # Notify spectator of vote
                    self._notify_spectator("vote", agent=agent.name, details=f"Voted for {vote_result.choice}", metric=vote_result.confidence)

                    # Emit vote event
                    if "on_vote" in self.hooks:
                        self.hooks["on_vote"](agent.name, vote_result.choice, vote_result.confidence)

            # Group similar vote options before counting
            vote_groups = self._group_similar_votes(result.votes)

            # Create mapping from variant -> canonical
            choice_mapping: dict[str, str] = {}
            for canonical, variants in vote_groups.items():
                for variant in variants:
                    choice_mapping[variant] = canonical

            if vote_groups:
                print(f"  Vote grouping merged: {vote_groups}")

            # Count votes using canonical choices with reputation weighting
            vote_counts = Counter()
            total_weighted_votes = 0.0

            for v in result.votes:
                if not isinstance(v, Exception):
                    canonical = choice_mapping.get(v.choice, v.choice)

                    # Get reputation-based vote weight (0.5-1.5 range)
                    vote_weight = 1.0
                    if self.memory and hasattr(self.memory, 'get_vote_weight'):
                        vote_weight = self.memory.get_vote_weight(v.agent)

                    vote_counts[canonical] += vote_weight
                    total_weighted_votes += vote_weight

            # Include user votes with configurable weight
            user_vote_weight = getattr(self.protocol, 'user_vote_weight', 0.5)  # Default: users count as half an agent
            for user_vote in self.user_votes:
                choice = user_vote.get("choice", "")
                if choice:
                    canonical = choice_mapping.get(choice, choice)
                    vote_counts[canonical] += user_vote_weight
                    total_weighted_votes += user_vote_weight
                    print(f"  User {user_vote.get('user_id', 'anonymous')} votes: {choice}")

            total_votes = total_weighted_votes

            if vote_counts:
                winner, count = vote_counts.most_common(1)[0]
                result.final_answer = proposals.get(winner, list(proposals.values())[0])
                result.consensus_reached = count / total_votes >= self.protocol.consensus_threshold
                result.confidence = count / total_votes

                # Calculate consensus variance and strength
                if len(vote_counts) > 1:
                    counts = list(vote_counts.values())
                    mean = sum(counts) / len(counts)
                    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
                    result.consensus_variance = variance

                    if variance < 1:
                        result.consensus_strength = "strong"
                    elif variance < 2:
                        result.consensus_strength = "medium"
                    else:
                        result.consensus_strength = "weak"

                    print(f"  Consensus strength: {result.consensus_strength} (variance: {variance:.2f})")
                else:
                    result.consensus_strength = "unanimous"
                    result.consensus_variance = 0.0

                # Track dissenting views (full content)
                for agent, prop in proposals.items():
                    if agent != winner:
                        result.dissenting_views.append(f"[{agent}]: {prop}")

                print(f"\n  Winner: {winner} ({count}/{len(self.agents)} votes)")

                # Notify spectator of consensus
                self._notify_spectator("consensus", details=f"Majority vote: {winner}", metric=result.confidence)
            else:
                result.final_answer = list(proposals.values())[0]
                result.consensus_reached = False
                result.confidence = 0.0

        elif self.protocol.consensus == "unanimous":
            # Unanimous mode: ALL agents must agree for consensus
            # Uses same voting mechanism as majority, but requires 100% agreement
            async def cast_vote(agent):
                """Cast vote and return (agent, result_or_error)."""
                print(f"  {agent.name}: voting (unanimous mode)...", flush=True)
                try:
                    vote_result = await self._vote_with_agent(agent, proposals, self.env.task)
                    return (agent, vote_result)
                except Exception as e:
                    return (agent, e)

            vote_tasks = [asyncio.create_task(cast_vote(agent)) for agent in self.agents]
            voting_errors = 0  # Track voting failures for unanimity calculation

            for completed_task in asyncio.as_completed(vote_tasks):
                agent, vote_result = await completed_task

                if isinstance(vote_result, Exception):
                    print(f"  {agent.name}: ERROR voting - {vote_result}")
                    voting_errors += 1  # Count as failed vote (breaks unanimity)
                else:
                    result.votes.append(vote_result)
                    print(f"  {agent.name} votes: {vote_result.choice} ({vote_result.confidence:.0%})")

                    # Notify spectator of vote
                    self._notify_spectator("vote", agent=agent.name, details=f"Voted for {vote_result.choice}", metric=vote_result.confidence)

                    # Emit vote event
                    if "on_vote" in self.hooks:
                        self.hooks["on_vote"](agent.name, vote_result.choice, vote_result.confidence)

            # Group similar votes to handle minor wording differences
            vote_groups = self._group_similar_votes(result.votes)
            choice_mapping: dict[str, str] = {}
            for canonical, variants in vote_groups.items():
                for variant in variants:
                    choice_mapping[variant] = canonical

            # Count votes (no reputation weighting for unanimous - all votes equal)
            vote_counts = Counter()
            for v in result.votes:
                if not isinstance(v, Exception):
                    canonical = choice_mapping.get(v.choice, v.choice)
                    vote_counts[canonical] += 1

            # Include user votes if configured
            user_vote_weight = getattr(self.protocol, 'user_vote_weight', 0.0)  # Default: users don't affect unanimity
            if user_vote_weight > 0:
                for user_vote in self.user_votes:
                    choice = user_vote.get("choice", "")
                    if choice:
                        canonical = choice_mapping.get(choice, choice)
                        vote_counts[canonical] += 1
                        print(f"  User {user_vote.get('user_id', 'anonymous')} votes: {choice}")

            # Check for unanimous agreement
            # Include voting errors in total - they count as dissent in unanimous mode
            total_voters = len(result.votes) + voting_errors + (len(self.user_votes) if user_vote_weight > 0 else 0)

            if vote_counts and total_voters > 0:
                winner, count = vote_counts.most_common(1)[0]
                unanimity_ratio = count / total_voters

                # Unanimous requires 100% agreement - no exceptions
                unanimous_threshold = 1.0

                if unanimity_ratio >= unanimous_threshold:
                    result.final_answer = proposals.get(winner, list(proposals.values())[0])
                    result.consensus_reached = True
                    result.confidence = unanimity_ratio
                    result.consensus_strength = "unanimous"
                    result.consensus_variance = 0.0
                    print(f"\n  UNANIMOUS: {winner} ({count}/{total_voters} votes, {unanimity_ratio:.0%})")

                    # Notify spectator of unanimous consensus
                    self._notify_spectator("consensus", details=f"Unanimous: {winner}", metric=result.confidence)
                else:
                    # Not unanimous - no consensus
                    result.final_answer = f"[No unanimous consensus reached]\n\nProposals:\n" + "\n\n---\n\n".join(
                        f"[{agent}] ({vote_counts.get(choice_mapping.get(agent, agent), 0)} votes):\n{prop}"
                        for agent, prop in proposals.items()
                    )
                    result.consensus_reached = False
                    result.confidence = unanimity_ratio
                    result.consensus_strength = "none"

                    # Track all views as dissenting since no consensus
                    for agent, prop in proposals.items():
                        result.dissenting_views.append(f"[{agent}]: {prop}")

                    print(f"\n  NO UNANIMOUS CONSENSUS: Best was {winner} with {unanimity_ratio:.0%} ({count}/{total_voters})")
                    self._notify_spectator("consensus", details=f"No unanimity: {winner} got {unanimity_ratio:.0%}", metric=unanimity_ratio)
            else:
                result.final_answer = list(proposals.values())[0]
                result.consensus_reached = False
                result.confidence = 0.0

        elif self.protocol.consensus == "judge":
            # Select judge based on protocol setting (random, voted, or last)
            judge = await self._select_judge(proposals, context)
            print(f"  Judge selected: {judge.name} (via {self.protocol.judge_selection})")

            # Notify spectator of judge selection
            self._notify_spectator("judge", agent=judge.name, details=f"Selected as judge via {self.protocol.judge_selection}")

            # Emit judge selection event
            if "on_judge_selected" in self.hooks:
                self.hooks["on_judge_selected"](judge.name, self.protocol.judge_selection)

            judge_prompt = self._build_judge_prompt(proposals, self.env.task, result.critiques)
            try:
                synthesis = await self._generate_with_agent(judge, judge_prompt, context)
                result.final_answer = synthesis
                result.consensus_reached = True
                result.confidence = 0.8
                print(f"  Judge ({judge.name}): {synthesis}")  # Full content

                # Notify spectator of judge synthesis
                self._notify_spectator("consensus", agent=judge.name, details=f"Judge synthesis ({len(synthesis)} chars)", metric=0.8)

                # Emit judge's synthesis as a message for the activity feed
                if "on_message" in self.hooks:
                    self.hooks["on_message"](
                        agent=judge.name,
                        content=synthesis,
                        role="judge",
                        round_num=self.protocol.rounds + 1,  # After all debate rounds
                    )
            except Exception as e:
                print(f"  Judge ERROR: {e}")
                result.final_answer = list(proposals.values())[0]
                result.consensus_reached = False

        # === Store successful patterns ===
        if self.memory and result.consensus_reached:
            for critique in result.critiques:
                if critique.severity < 0.5:  # Low severity = successful pattern
                    self.memory.store_pattern(critique, result.final_answer)

        result.duration_seconds = time.time() - start_time

        # Emit consensus event
        if "on_consensus" in self.hooks:
            self.hooks["on_consensus"](
                reached=result.consensus_reached,
                confidence=result.confidence,
                answer=result.final_answer,
            )

        # Emit debate end event
        if "on_debate_end" in self.hooks:
            self.hooks["on_debate_end"](
                duration=result.duration_seconds,
                rounds=result.rounds_used,
            )

        # Notify spectator of debate end
        self._notify_spectator("debate_end", details=f"Complete in {result.duration_seconds:.1f}s", metric=result.confidence)

        # === Extract and store insights ===
        if self.insight_store:
            try:
                IE, _ = _get_insight_classes()
                extractor = IE()
                insights = extractor.extract(result, [a.name for a in self.agents])
                stored_count = await self.insight_store.store_debate_insights(insights)
                if stored_count > 0:
                    print(f"  Extracted {insights.total_insights} insights ({stored_count} stored)")
            except Exception as e:
                print(f"  Insight extraction failed: {e}")

        print(f"\n{'='*60}")
        print(f"DEBATE COMPLETE in {result.duration_seconds:.1f}s")
        print(f"Consensus: {'Yes' if result.consensus_reached else 'No'} ({result.confidence:.0%})")
        print(f"{'='*60}\n")

        return result

    async def _generate_with_agent(
        self, agent: Agent, prompt: str, context: list[Message]
    ) -> str:
        """Generate response with an agent, handling errors."""
        return await agent.generate(prompt, context)

    async def _critique_with_agent(
        self, agent: Agent, proposal: str, task: str, context: list[Message]
    ) -> Critique:
        """Get critique from an agent."""
        return await agent.critique(proposal, task, context)

    async def _vote_with_agent(
        self, agent: Agent, proposals: dict[str, str], task: str
    ) -> Vote:
        """Get vote from an agent."""
        return await agent.vote(proposals, task)

    def _group_similar_votes(self, votes: list[Vote]) -> dict[str, list[str]]:
        """
        Group semantically similar vote choices.

        This prevents artificial disagreement when agents vote for the
        same thing using different wording (e.g., "Vector DB" vs "Use vector database").

        Returns:
            Dict mapping canonical choice -> list of original choices that map to it
        """
        if not self.protocol.vote_grouping or not votes:
            return {}

        # Get similarity backend
        backend = get_similarity_backend("auto")

        # Extract unique choices
        choices = list(set(v.choice for v in votes if v.choice))
        if len(choices) < 2:
            return {}

        # Build groups using union-find approach
        groups: dict[str, list[str]] = {}  # canonical -> [choices]
        assigned: dict[str, str] = {}  # choice -> canonical

        for choice in choices:
            if choice in assigned:
                continue

            # Start a new group with this choice as canonical
            groups[choice] = [choice]
            assigned[choice] = choice

            # Check other unassigned choices for similarity
            for other in choices:
                if other in assigned or other == choice:
                    continue

                similarity = backend.compute_similarity(choice, other)
                if similarity >= self.protocol.vote_grouping_threshold:
                    groups[choice].append(other)
                    assigned[other] = choice

        # Only return groups with multiple members (merges occurred)
        return {k: v for k, v in groups.items() if len(v) > 1}

    async def _check_judge_termination(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> tuple[bool, str]:
        """
        Have a judge evaluate if the debate is conclusive.

        Returns:
            Tuple of (should_continue: bool, reason: str)
        """
        if not self.protocol.judge_termination:
            return True, ""

        if round_num < self.protocol.min_rounds_before_judge_check:
            return True, ""

        # Select a judge (use existing method)
        judge = await self._select_judge(proposals, context)

        prompt = f"""You are evaluating whether this multi-agent debate has reached a conclusive state.

Task: {self.env.task[:300]}

After {round_num} rounds of debate, the proposals are:
{chr(10).join(f"- {agent}: {prop[:200]}..." for agent, prop in proposals.items())}

Evaluate:
1. Have the key issues been thoroughly discussed?
2. Are there major unresolved disagreements that more debate could resolve?
3. Would additional rounds likely produce meaningful improvements?

Respond with:
CONCLUSIVE: <yes/no>
REASON: <brief explanation>"""

        try:
            response = await self._generate_with_agent(judge, prompt, context[-5:])
            lines = response.strip().split('\n')

            conclusive = False
            reason = ""

            for line in lines:
                if line.upper().startswith("CONCLUSIVE:"):
                    val = line.split(":", 1)[1].strip().lower()
                    conclusive = val in ("yes", "true", "1")
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            if conclusive:
                print(f"  Judge ({judge.name}) says debate is conclusive: {reason[:100]}")
                # Emit event
                if "on_judge_termination" in self.hooks:
                    self.hooks["on_judge_termination"](judge.name, reason)
                return False, reason

        except Exception as e:
            print(f"  Judge termination check failed: {e}")

        return True, ""

    async def _check_early_stopping(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> bool:
        """Check if agents want to stop debate early.

        Returns True if debate should continue, False if it should stop.
        """
        if not self.protocol.early_stopping:
            return True  # Continue

        if round_num < self.protocol.min_rounds_before_early_stop:
            return True  # Continue - haven't met minimum rounds

        # Ask each agent if they think more debate would help
        prompt = f"""After {round_num} round(s) of debate on this task:
Task: {self.env.task[:200]}

Current proposals have been critiqued and revised. Do you think additional debate
rounds would significantly improve the answer quality?

Respond with only: CONTINUE or STOP
- CONTINUE: More debate rounds would help refine the answer
- STOP: The proposals are mature enough, further debate is unlikely to help"""

        stop_votes = 0
        total_votes = 0

        tasks = [self._generate_with_agent(agent, prompt, context[-5:]) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                continue
            total_votes += 1
            response = result.strip().upper()
            if "STOP" in response and "CONTINUE" not in response:
                stop_votes += 1

        if total_votes == 0:
            return True  # Continue if voting failed

        stop_ratio = stop_votes / total_votes
        should_stop = stop_ratio >= self.protocol.early_stop_threshold

        if should_stop:
            print(f"\n  Early stopping: {stop_votes}/{total_votes} agents voted to stop")
            # Emit early stop event
            if "on_early_stop" in self.hooks:
                self.hooks["on_early_stop"](round_num, stop_votes, total_votes)

        return not should_stop  # Return True to continue, False to stop

    async def _select_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Select judge based on protocol.judge_selection setting."""
        if self.protocol.judge_selection == "last":
            # Legacy behavior - use synthesizer or last agent
            synthesizers = [a for a in self.agents if a.role == "synthesizer"]
            return synthesizers[0] if synthesizers else self.agents[-1]

        elif self.protocol.judge_selection == "random":
            # Random selection from all agents
            return random.choice(self.agents)

        elif self.protocol.judge_selection == "voted":
            # Agents vote on who should judge
            return await self._vote_for_judge(proposals, context)

        # Default fallback
        return random.choice(self.agents)

    async def _vote_for_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Have agents vote on who should be the judge."""
        vote_counts: dict[str, int] = {}

        for agent in self.agents:
            # Each agent votes for who should judge (can't vote for self)
            other_agents = [a for a in self.agents if a.name != agent.name]
            prompt = self._build_judge_vote_prompt(other_agents, proposals)

            try:
                response = await agent.generate(prompt, context)
                # Parse vote from response - look for agent names
                for other in other_agents:
                    if other.name.lower() in response.lower():
                        vote_counts[other.name] = vote_counts.get(other.name, 0) + 1
                        break
            except Exception:
                pass  # Skip failed votes

        # Select agent with most votes, random tiebreaker
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [name for name, count in vote_counts.items() if count == max_votes]
            winner_name = random.choice(candidates)
            return next(a for a in self.agents if a.name == winner_name)

        # Fallback to random if voting fails
        return random.choice(self.agents)

    def _build_judge_vote_prompt(self, candidates: list[Agent], proposals: dict[str, str]) -> str:
        """Build prompt for voting on who should judge."""
        candidate_names = ", ".join(a.name for a in candidates)
        proposals_summary = "\n".join(
            f"- {name}: {prop[:300]}..." for name, prop in proposals.items()
        )

        return f"""Based on the proposals in this debate, vote for which agent should synthesize the final answer.

Candidates: {candidate_names}

Proposals summary:
{proposals_summary}

Consider: Which agent showed the most balanced, thorough, and fair reasoning?
Vote by stating ONLY the agent's name. You cannot vote for yourself."""

    def _get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Agreement intensity (0-10) affects how agents approach disagreements:
        - Low (0-3): Adversarial - strongly challenge others' positions
        - Medium (4-6): Balanced - judge arguments on merit
        - High (7-10): Collaborative - seek common ground and synthesis
        """
        intensity = self.protocol.agreement_intensity

        if intensity <= 1:
            return """IMPORTANT: You strongly disagree with other agents. Challenge every assumption,
find flaws in every argument, and maintain your original position unless presented
with irrefutable evidence. Be adversarial but constructive."""
        elif intensity <= 3:
            return """IMPORTANT: Approach others' arguments with healthy skepticism. Be critical of
proposals and require strong evidence before changing your position. Point out
weaknesses even if you partially agree."""
        elif intensity <= 6:
            return """Evaluate arguments on their merits. Agree when others make valid points,
disagree when you see genuine flaws. Let the quality of reasoning guide your response."""
        elif intensity <= 8:
            return """Look for common ground with other agents. Acknowledge valid points in others'
arguments and try to build on them. Seek synthesis where possible while maintaining
your own reasoned perspective."""
        else:  # 9-10
            return """Actively seek to incorporate other agents' perspectives. Find value in all
proposals and work toward collaborative synthesis. Prioritize finding agreement
and building on others' ideas."""

    def _format_successful_patterns(self, limit: int = 3) -> str:
        """Format successful critique patterns for prompt injection."""
        if not self.memory:
            return ""
        try:
            patterns = self.memory.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            for p in patterns:
                issue_preview = p.issue_text[:100] + "..." if len(p.issue_text) > 100 else p.issue_text
                fix_preview = p.suggestion_text[:80] + "..." if len(p.suggestion_text) > 80 else p.suggestion_text
                lines.append(f"- **{p.issue_type}**: {issue_preview}")
                if fix_preview:
                    lines.append(f"  Fix: {fix_preview} ({p.success_count} successes)")
            return "\n".join(lines)
        except Exception:
            return ""

    def _build_proposal_prompt(self, agent: Agent) -> str:
        """Build the initial proposal prompt."""
        context_str = f"\n\nContext: {self.env.context}" if self.env.context else ""
        stance_str = self._get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        # Include historical context if available (capped at 800 chars to prevent bloat)
        historical_section = ""
        if self._historical_context_cache:
            historical = self._historical_context_cache[:800]
            historical_section = f"\n\n{historical}"

        # Include successful patterns from past debates
        patterns_section = ""
        patterns = self._format_successful_patterns(limit=3)
        if patterns:
            patterns_section = f"\n\n{patterns}"

        # Inject audience suggestions if enabled
        audience_section = ""
        if (
            self.protocol.audience_injection in ("summary", "inject")
            and self.user_suggestions
        ):
            clusters = cluster_suggestions(self.user_suggestions)
            audience_section = format_for_prompt(clusters)
            if audience_section:
                audience_section = f"\n\n{audience_section}"
            
            # Emit stream event for dashboard
            if self.spectator and clusters:
                self._notify_spectator(
                    "audience_summary",
                    details=f"{sum(c.count for c in clusters)} suggestions in {len(clusters)} clusters",
                    metric=len(clusters),
                )

        # Inject agent introspection if enabled
        introspection_section = ""
        if self._introspection_cache:
            snapshot = self._introspection_cache.get(agent.name)
            if snapshot:
                introspection_section = f"\n\n{snapshot.to_prompt_section()}"

        return f"""You are acting as a {agent.role} in a multi-agent debate.{stance_section}
{historical_section}{patterns_section}{audience_section}{introspection_section}
Task: {self.env.task}{context_str}

Please provide your best proposal to address this task. Be thorough and specific.
Your proposal will be critiqued by other agents, so anticipate potential objections."""

    def _build_revision_prompt(
        self, agent: Agent, original: str, critiques: list[Critique]
    ) -> str:
        """Build the revision prompt including critiques."""
        critiques_str = "\n\n".join(c.to_prompt() for c in critiques)
        intensity_guidance = self._get_agreement_intensity_guidance()
        stance_str = self._get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        # Include successful patterns that may help address critiques
        patterns_section = ""
        patterns = self._format_successful_patterns(limit=2)
        if patterns:
            patterns_section = f"\n\n{patterns}"

        # Inject audience suggestions if enabled
        audience_section = ""
        if (
            self.protocol.audience_injection in ("summary", "inject")
            and self.user_suggestions
        ):
            clusters = cluster_suggestions(self.user_suggestions)
            audience_section = format_for_prompt(clusters)
            if audience_section:
                audience_section = f"\n\n{audience_section}"

        # Inject agent introspection if enabled
        introspection_section = ""
        if self._introspection_cache:
            snapshot = self._introspection_cache.get(agent.name)
            if snapshot:
                introspection_section = f"\n\n{snapshot.to_prompt_section()}"

        return f"""You are revising your proposal based on critiques from other agents.

{intensity_guidance}{stance_section}{patterns_section}{audience_section}{introspection_section}

Original Task: {self.env.task}

Your Original Proposal:
{original}

Critiques Received:
{critiques_str}

Please provide a revised proposal that addresses the valid critiques.
Explain what you changed and why. If you disagree with a critique, explain your reasoning."""

    def _build_judge_prompt(
        self, proposals: dict[str, str], task: str, critiques: list[Critique]
    ) -> str:
        """Build the judge/synthesizer prompt."""
        proposals_str = "\n\n---\n\n".join(
            f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
        )
        critiques_str = "\n".join(
            f"- {c.agent}: {', '.join(c.issues[:2])}" for c in critiques[:5]
        )

        return f"""You are the synthesizer/judge in a multi-agent debate.

Task: {task}

Proposals:
{proposals_str}

Key Critiques:
{critiques_str}

Synthesize the best elements of all proposals into a final answer.
Address the most important critiques raised. Explain your synthesis."""
