"""
Prompt assembly mixin for PromptBuilder.

Provides the build_* methods that assemble final prompt strings
from context sections: proposal, revision, judge, and vote prompts.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from aragora.debate.context_budgeter import ContextSection

if TYPE_CHECKING:
    from aragora.core import Agent, Critique

logger = logging.getLogger(__name__)


class PromptAssemblyMixin:
    """Mixin providing prompt assembly methods."""

    # These attributes/methods are defined in the main PromptBuilder class or other mixins
    protocol: Any
    env: Any
    _rlm_context: Any
    _rlm_adapter: Any
    _enable_rlm_hints: bool
    _historical_context_cache: str
    dissent_retriever: Any
    _context_budgeter: Any

    # Methods from PromptContextMixin (available via MRO)
    get_stance_guidance: Any
    get_agreement_intensity_guidance: Any
    get_role_context: Any
    get_persona_context: Any
    get_flip_context: Any
    get_round_phase_context: Any
    get_rlm_abstract: Any
    get_rlm_context_hint: Any
    get_continuum_context: Any
    get_supermemory_context: Any
    get_language_constraint: Any
    format_successful_patterns: Any
    format_evidence_for_prompt: Any
    format_trending_for_prompt: Any
    get_elo_context: Any
    _inject_belief_context: Any
    _inject_calibration_context: Any
    _estimate_tokens: Any
    _apply_context_budget: Any

    def build_proposal_prompt(
        self,
        agent: "Agent",
        audience_section: str = "",
        all_agents: list["Agent"] | None = None,
    ) -> str:
        """Build the initial proposal prompt.

        Args:
            agent: The agent to build the prompt for
            audience_section: Optional pre-formatted audience suggestions section
            all_agents: Optional list of all agents for ELO context injection
        """
        env_context = f"Context: {self.env.context}" if self.env.context else ""

        # Add research status indicator
        research_status = ""
        if not self.env.context or "No research context" in str(self.env.context):
            research_status = "\n\n[RESEARCH STATUS: No external research was performed. Base your response on your training knowledge and clearly state any limitations or uncertainties about specific entities, websites, or current events.]"
        elif "EVIDENCE CONTEXT" in str(self.env.context):
            research_status = "\n\n[RESEARCH STATUS: Research context has been provided above. Use this information in your response and cite it where applicable.]"

        stance_str = self.get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        role_section = self.get_role_context(agent)
        if role_section:
            role_section = f"\n\n{role_section}"

        persona_section = ""
        persona_context = self.get_persona_context(agent)
        if persona_context:
            persona_section = f"\n\n{persona_context}"

        flip_section = ""
        flip_context = self.get_flip_context(agent)
        if flip_context:
            flip_section = f"\n\n{flip_context}"

        # Prefer RLM abstract for semantic compression over simple truncation
        historical_section = ""
        if self._rlm_context:
            rlm_abstract = self.get_rlm_abstract(max_chars=800)
            if rlm_abstract:
                historical_section = f"## Prior Context (Compressed)\n{rlm_abstract}"
                rlm_hint = self.get_rlm_context_hint()
                if rlm_hint:
                    historical_section += f"\n{rlm_hint}"
        elif self._historical_context_cache:
            historical = self._historical_context_cache[:800]
            historical_section = f"{historical}"

        continuum_section = ""
        continuum_context = self.get_continuum_context()
        if continuum_context:
            continuum_section = f"{continuum_context}"

        supermemory_section = ""
        supermemory_context = self.get_supermemory_context()
        if supermemory_context:
            supermemory_section = f"{supermemory_context}"

        belief_section = ""
        belief_context = self._inject_belief_context(limit=3)
        if belief_context:
            belief_section = f"{belief_context}"

        # Historical dissents and minority views
        dissent_section = ""
        if self.dissent_retriever:
            try:
                dissent_context = self.dissent_retriever.get_debate_preparation_context(
                    topic=self.env.task
                )
                if dissent_context:
                    if self._rlm_adapter and len(dissent_context) > 600:
                        formatted = self._rlm_adapter.format_for_prompt(
                            content=dissent_context,
                            max_chars=600,
                            content_type="dissent",
                            include_hint=self._enable_rlm_hints,
                        )
                        dissent_section = f"## Historical Minority Views\n{formatted}"
                    else:
                        dissent_section = f"## Historical Minority Views\n{dissent_context[:600]}"
            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"Dissent retrieval error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected dissent retrieval error: {e}")

        patterns_section = ""
        patterns = self.format_successful_patterns(limit=3)
        if patterns:
            patterns_section = f"{patterns}"

        calibration_section = ""
        calibration_context = self._inject_calibration_context(agent)
        if calibration_context:
            calibration_section = f"{calibration_context}"

        elo_section = ""
        if all_agents:
            elo_context = self.get_elo_context(agent, all_agents)
            if elo_context:
                elo_section = f"{elo_context}"

        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=5)
        if evidence_context:
            evidence_section = f"{evidence_context}"

        trending_section = ""
        trending_context = self.format_trending_for_prompt(max_topics=3)
        if trending_context:
            trending_section = f"{trending_context}"

        if audience_section:
            audience_section = f"{audience_section}"

        sections = [
            ContextSection("historical", historical_section.strip()),
            ContextSection("continuum", continuum_section.strip()),
            ContextSection("supermemory", supermemory_section.strip()),
            ContextSection("belief", belief_section.strip()),
            ContextSection("dissent", dissent_section.strip()),
            ContextSection("patterns", patterns_section.strip()),
            ContextSection("calibration", calibration_section.strip()),
            ContextSection("elo", elo_section.strip()),
            ContextSection("evidence", evidence_section.strip()),
            ContextSection("trending", trending_section.strip()),
            ContextSection("audience", audience_section.strip()),
        ]

        context_block, context_str = self._apply_context_budget(
            env_context=env_context,
            sections=sections,
        )

        return f"""You are acting as a {agent.role} in a multi-agent debate (decision stress-test).{stance_section}{role_section}{persona_section}{flip_section}
{context_block}
Task: {self.env.task}{context_str}{research_status}

IMPORTANT: If this task mentions a specific website, company, product, or current topic, you MUST:
1. State what you know vs what you would need to research
2. If research context was provided above, use it. If not, acknowledge the limitation.
3. Do NOT make up facts or speculate about specific entities you don't have verified information about.

Please provide your best proposal to address this task. Be thorough and specific.
Your proposal will be critiqued by other agents, so anticipate potential objections.{self.get_language_constraint()}"""

    def build_revision_prompt(
        self,
        agent: "Agent",
        original: str,
        critiques: list["Critique"],
        audience_section: str = "",
        all_agents: list["Agent"] | None = None,
        round_number: int = 0,
    ) -> str:
        """Build the revision prompt including critiques."""
        critiques_str = "\n\n".join(c.to_prompt() for c in critiques)
        intensity_guidance = self.get_agreement_intensity_guidance()
        stance_str = self.get_stance_guidance(agent)
        stance_section = f"\n\n{stance_str}" if stance_str else ""

        round_phase_section = ""
        if round_number > 0:
            round_phase_context = self.get_round_phase_context(round_number)
            if round_phase_context:
                round_phase_section = f"\n\n{round_phase_context}"

        role_section = self.get_role_context(agent)
        if role_section:
            role_section = f"\n\n{role_section}"

        persona_section = ""
        persona_context = self.get_persona_context(agent)
        if persona_context:
            persona_section = f"\n\n{persona_context}"

        flip_section = ""
        flip_context = self.get_flip_context(agent)
        if flip_context:
            flip_section = f"\n\n{flip_context}"

        patterns_section = ""
        patterns = self.format_successful_patterns(limit=2)
        if patterns:
            patterns_section = patterns

        belief_section = ""
        belief_context = self._inject_belief_context(limit=2)
        if belief_context:
            belief_section = belief_context

        calibration_section = ""
        calibration_context = self._inject_calibration_context(agent)
        if calibration_context:
            calibration_section = calibration_context

        elo_section = ""
        if all_agents:
            elo_context = self.get_elo_context(agent, all_agents)
            if elo_context:
                elo_section = elo_context

        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=3)
        if evidence_context:
            evidence_section = evidence_context

        trending_section = ""
        trending_context = self.format_trending_for_prompt(max_topics=2)
        if trending_context:
            trending_section = trending_context

        sections = [
            ContextSection("patterns", patterns_section.strip()),
            ContextSection("belief", belief_section.strip()),
            ContextSection("calibration", calibration_section.strip()),
            ContextSection("elo", elo_section.strip()),
            ContextSection("evidence", evidence_section.strip()),
            ContextSection("trending", trending_section.strip()),
            ContextSection("audience", audience_section.strip()),
        ]
        context_block, _ = self._apply_context_budget(env_context="", sections=sections)

        return f"""You are revising your proposal based on critiques from other agents.{round_phase_section}{role_section}{persona_section}{flip_section}

{intensity_guidance}{stance_section}{context_block}

Original Task: {self.env.task}

Your Original Proposal:
{original}

Critiques Received:
{critiques_str}

Please provide a revised proposal that addresses the valid critiques.
Use evidence citations [EVID-N] to support strengthened claims.
Explain what you changed and why. If you disagree with a critique, explain your reasoning.{self.get_language_constraint()}"""

    def build_judge_prompt(
        self,
        proposals: dict[str, str],
        task: str,
        critiques: list["Critique"],
    ) -> str:
        """Build the judge/synthesizer prompt."""
        proposals_str = "\n\n---\n\n".join(
            f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
        )
        critiques_str = "\n".join(f"- {c.agent}: {', '.join(c.issues[:2])}" for c in critiques[:5])

        evidence_section = ""
        evidence_context = self.format_evidence_for_prompt(max_snippets=5)
        if evidence_context:
            evidence_section = evidence_context

        context_block, _ = self._apply_context_budget(
            env_context="",
            sections=[ContextSection("evidence", evidence_section.strip())],
        )

        return f"""You are the synthesizer/judge in a multi-agent debate (decision stress-test).

Task: {task}
{context_block}
Proposals:
{proposals_str}

Key Critiques:
{critiques_str}

Synthesize the best elements of all proposals into a final answer.
Reference evidence [EVID-N] to support key claims in your synthesis.
Address the most important critiques raised. Explain your synthesis."""

    def build_judge_vote_prompt(self, candidates: list["Agent"], proposals: dict[str, str]) -> str:
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
