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


@dataclass
class DebateProtocol:
    """Configuration for how debates are conducted."""

    topology: Literal["all-to-all", "sparse", "round-robin"] = "round-robin"
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
    ):
        self.env = environment
        self.agents = agents
        self.protocol = protocol or DebateProtocol()
        self.memory = memory
        self.hooks = event_hooks or {}

        # Assign roles if not already set
        self._assign_roles()

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

    async def run(self) -> DebateResult:
        """Run the full debate and return results."""
        start_time = time.time()

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
        print(f"{'='*60}\n")

        # Emit debate start event
        if "on_debate_start" in self.hooks:
            self.hooks["on_debate_start"](self.env.task, [a.name for a in self.agents])

        # Generate initial proposals
        print("Round 0: Initial Proposals")
        print("-" * 40)

        proposal_tasks = []
        for agent in proposers:
            prompt = self._build_proposal_prompt(agent)
            proposal_tasks.append(self._generate_with_agent(agent, prompt, context))

        proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)

        for agent, result_or_error in zip(proposers, proposal_results):
            if isinstance(result_or_error, Exception):
                print(f"  {agent.name}: ERROR - {result_or_error}")
                proposals[agent.name] = f"[Error generating proposal: {result_or_error}]"
            else:
                proposals[agent.name] = result_or_error
                print(f"  {agent.name}: {result_or_error}")  # Full content

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

            # Emit round start event
            if "on_round_start" in self.hooks:
                self.hooks["on_round_start"](round_num)

            # Get critics - when all agents are proposers, they all critique each other
            critics = [a for a in self.agents if a.role in ("critic", "synthesizer")]
            if not critics:
                # When no dedicated critics exist, all agents critique each other
                # The loop below already skips self-critique via "if critic.name != proposal_agent"
                critics = list(self.agents)

            # === Critique Phase ===
            for proposal_agent, proposal in proposals.items():
                critique_tasks = []
                for critic in critics:
                    if critic.name != proposal_agent:  # Don't critique yourself
                        critique_tasks.append(
                            self._critique_with_agent(critic, proposal, self.env.task, context)
                        )

                if critique_tasks:
                    critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)

                    for critic, crit_result in zip(
                        [c for c in critics if c.name != proposal_agent], critique_results
                    ):
                        if isinstance(crit_result, Exception):
                            print(f"  {critic.name} -> {proposal_agent}: ERROR - {crit_result}")
                        else:
                            result.critiques.append(crit_result)
                            print(
                                f"  {critic.name} -> {proposal_agent}: "
                                f"{len(crit_result.issues)} issues, "
                                f"severity {crit_result.severity:.1f}"
                            )

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
            # All agents vote
            vote_tasks = [
                self._vote_with_agent(agent, proposals, self.env.task) for agent in self.agents
            ]
            votes = await asyncio.gather(*vote_tasks, return_exceptions=True)

            for agent, vote_result in zip(self.agents, votes):
                if isinstance(vote_result, Exception):
                    print(f"  {agent.name}: ERROR voting - {vote_result}")
                else:
                    result.votes.append(vote_result)
                    print(f"  {agent.name} votes: {vote_result.choice} ({vote_result.confidence:.0%})")

                    # Emit vote event
                    if "on_vote" in self.hooks:
                        self.hooks["on_vote"](agent.name, vote_result.choice, vote_result.confidence)

            # Count votes
            vote_counts = Counter(v.choice for v in result.votes if not isinstance(v, Exception))
            if vote_counts:
                winner, count = vote_counts.most_common(1)[0]
                result.final_answer = proposals.get(winner, list(proposals.values())[0])
                result.consensus_reached = count / len(self.agents) >= self.protocol.consensus_threshold
                result.confidence = count / len(self.agents)

                # Track dissenting views (full content)
                for agent, prop in proposals.items():
                    if agent != winner:
                        result.dissenting_views.append(f"[{agent}]: {prop}")

                print(f"\n  Winner: {winner} ({count}/{len(self.agents)} votes)")
            else:
                result.final_answer = list(proposals.values())[0]
                result.consensus_reached = False
                result.confidence = 0.0

        elif self.protocol.consensus == "judge":
            # Select judge based on protocol setting (random, voted, or last)
            judge = await self._select_judge(proposals, context)
            print(f"  Judge selected: {judge.name} (via {self.protocol.judge_selection})")

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

    def _build_proposal_prompt(self, agent: Agent) -> str:
        """Build the initial proposal prompt."""
        context_str = f"\n\nContext: {self.env.context}" if self.env.context else ""

        return f"""You are acting as a {agent.role} in a multi-agent debate.

Task: {self.env.task}{context_str}

Please provide your best proposal to address this task. Be thorough and specific.
Your proposal will be critiqued by other agents, so anticipate potential objections."""

    def _build_revision_prompt(
        self, agent: Agent, original: str, critiques: list[Critique]
    ) -> str:
        """Build the revision prompt including critiques."""
        critiques_str = "\n\n".join(c.to_prompt() for c in critiques)

        return f"""You are revising your proposal based on critiques from other agents.

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
