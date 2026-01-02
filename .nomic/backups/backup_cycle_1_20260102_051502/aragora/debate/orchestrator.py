"""
Multi-agent debate orchestrator.

Implements the propose -> critique -> revise loop with configurable
debate protocols and consensus mechanisms.
"""

import asyncio
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

            # Get critics
            critics = [a for a in self.agents if a.role in ("critic", "synthesizer")]
            if not critics:
                critics = [a for a in self.agents if a not in proposers]

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

                    # Emit vote event for streaming
                    if "on_vote" in self.hooks:
                        self.hooks["on_vote"](agent.name, vote_result.choice, vote_result.confidence)

            # Count votes
            vote_counts = Counter(v.choice for v in result.votes if not isinstance(v, Exception))
            if vote_counts:
                winner, count = vote_counts.most_common(1)[0]
                result.final_answer = proposals.get(winner, list(proposals.values())[0])
                result.consensus_reached = count / len(self.agents) >= self.protocol.consensus_threshold
                result.confidence = count / len(self.agents)

                # Track dissenting views (full content for streaming)
                for agent, prop in proposals.items():
                    if agent != winner:
                        result.dissenting_views.append(f"[{agent}]: {prop}")

                print(f"\n  Winner: {winner} ({count}/{len(self.agents)} votes)")
            else:
                result.final_answer = list(proposals.values())[0]
                result.consensus_reached = False
                result.confidence = 0.0

        elif self.protocol.consensus == "judge":
            # Use synthesizer as judge
            synthesizers = [a for a in self.agents if a.role == "synthesizer"]
            judge = synthesizers[0] if synthesizers else self.agents[-1]

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
