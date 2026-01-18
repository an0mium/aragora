"""
Dialectic Pattern - Thesis-antithesis-synthesis pattern.

The Dialectic pattern implements the classic philosophical method where:
1. A thesis (initial position) is proposed
2. An antithesis (opposing view) challenges it
3. A synthesis resolves the tension into a higher understanding

This is ideal for:
- Policy analysis
- Strategic decision making
- Exploring complex trade-offs
- Devil's advocate scenarios

Structure:
    [Input] -> [Thesis Agent] -> [Antithesis Agent] -> [Synthesis Agent] -> [Output]

Configuration:
    - thesis_agent: Agent to propose initial position
    - antithesis_agent: Agent to challenge/oppose
    - synthesis_agent: Agent to integrate and resolve
    - thesis_prompt: How to frame the initial position
    - antithesis_prompt: How to challenge
    - synthesis_prompt: How to integrate
"""

from __future__ import annotations

from typing import List, Optional

from aragora.workflow.types import (
    WorkflowDefinition,
    Position,
    NodeCategory,
    WorkflowCategory,
)
from aragora.workflow.patterns.base import WorkflowPattern, PatternType


class DialecticPattern(WorkflowPattern):
    """
    Thesis-antithesis-synthesis pattern.

    Three-stage dialectical reasoning for exploring complex topics
    with opposing viewpoints and resolution.

    Example:
        workflow = DialecticPattern.create(
            name="AI Ethics Analysis",
            thesis_agent="claude",
            antithesis_agent="gpt4",
            synthesis_agent="claude",
            task="Should AI systems be given legal personhood?",
        )
    """

    pattern_type = PatternType.DIALECTIC

    def __init__(
        self,
        name: str,
        agents: Optional[List[str]] = None,
        task: str = "",
        thesis_agent: Optional[str] = None,
        antithesis_agent: Optional[str] = None,
        synthesis_agent: Optional[str] = None,
        thesis_prompt: str = "",
        antithesis_prompt: str = "",
        synthesis_prompt: str = "",
        thesis_stance: str = "supportive",  # supportive, critical, neutral
        include_meta_analysis: bool = True,
        timeout_per_step: float = 120.0,
        **kwargs,
    ):
        super().__init__(name, agents, task, **kwargs)

        # Assign agents (use provided or defaults)
        agents = agents or ["claude", "gpt4", "claude"]
        self.thesis_agent = thesis_agent or agents[0]
        self.antithesis_agent = antithesis_agent or (agents[1] if len(agents) > 1 else agents[0])
        self.synthesis_agent = synthesis_agent or (agents[2] if len(agents) > 2 else agents[0])

        self.thesis_prompt = thesis_prompt
        self.antithesis_prompt = antithesis_prompt
        self.synthesis_prompt = synthesis_prompt
        self.thesis_stance = thesis_stance
        self.include_meta_analysis = include_meta_analysis
        self.timeout_per_step = timeout_per_step

    def create_workflow(self) -> WorkflowDefinition:
        """Create a dialectic workflow definition."""
        workflow_id = self._generate_id("dial")
        steps = []
        transitions = []

        # Positions for visual layout
        thesis_x = 100
        antithesis_x = 400
        synthesis_x = 700
        meta_x = 1000
        y_pos = 200

        # Step 1: Thesis
        thesis_prompt = self.thesis_prompt or self._build_thesis_prompt()
        thesis_step = self._create_agent_step(
            step_id="thesis",
            name="Thesis",
            agent_type=self.thesis_agent,
            prompt=thesis_prompt,
            position=Position(x=thesis_x, y=y_pos),
            timeout=self.timeout_per_step,
        )
        thesis_step.config["system_prompt"] = self._get_thesis_system_prompt()
        steps.append(thesis_step)

        # Step 2: Antithesis
        antithesis_prompt = self.antithesis_prompt or self._build_antithesis_prompt()
        antithesis_step = self._create_agent_step(
            step_id="antithesis",
            name="Antithesis",
            agent_type=self.antithesis_agent,
            prompt=antithesis_prompt,
            position=Position(x=antithesis_x, y=y_pos),
            timeout=self.timeout_per_step,
        )
        antithesis_step.config["system_prompt"] = (
            "You are a critical thinker who identifies flaws, counterarguments, "
            "and alternative perspectives. Challenge assumptions and expose weaknesses."
        )
        steps.append(antithesis_step)

        # Step 3: Synthesis
        synthesis_prompt = self.synthesis_prompt or self._build_synthesis_prompt()
        synthesis_step = self._create_agent_step(
            step_id="synthesis",
            name="Synthesis",
            agent_type=self.synthesis_agent,
            prompt=synthesis_prompt,
            position=Position(x=synthesis_x, y=y_pos),
            timeout=self.timeout_per_step,
        )
        synthesis_step.config["system_prompt"] = (
            "You are a philosophical synthesizer who integrates opposing viewpoints. "
            "Find higher truths that transcend the original positions."
        )
        steps.append(synthesis_step)

        # Set up transitions
        thesis_step.next_steps = ["antithesis"]
        antithesis_step.next_steps = ["synthesis"]

        transitions.extend([
            self._create_transition("thesis", "antithesis"),
            self._create_transition("antithesis", "synthesis"),
        ])

        # Optional: Meta-analysis step
        if self.include_meta_analysis:
            meta_step = self._create_task_step(
                step_id="meta_analysis",
                name="Meta-Analysis",
                task_type="transform",
                config={
                    "transform": self._build_meta_transform(),
                    "output_format": "json",
                },
                position=Position(x=meta_x, y=y_pos),
                category=NodeCategory.TASK,
            )
            steps.append(meta_step)

            synthesis_step.next_steps = ["meta_analysis"]
            transitions.append(self._create_transition("synthesis", "meta_analysis"))

        return WorkflowDefinition(
            id=workflow_id,
            name=self.name,
            description="Dialectic pattern: thesis-antithesis-synthesis",
            steps=steps,
            transitions=transitions,
            entry_step="thesis",
            category=self.config.get("category", WorkflowCategory.GENERAL),
            tags=["dialectic", "debate", "synthesis"] + self.config.get("tags", []),
            metadata={
                "pattern": "dialectic",
                "thesis_agent": self.thesis_agent,
                "antithesis_agent": self.antithesis_agent,
                "synthesis_agent": self.synthesis_agent,
                "thesis_stance": self.thesis_stance,
            },
        )

    def _get_thesis_system_prompt(self) -> str:
        """Get system prompt based on thesis stance."""
        stances = {
            "supportive": "You are an advocate who builds the strongest possible case in favor of a position.",
            "critical": "You are a critic who identifies problems and concerns with a topic.",
            "neutral": "You are an analyst who presents a balanced initial assessment.",
        }
        return stances.get(self.thesis_stance, stances["supportive"])

    def _build_thesis_prompt(self) -> str:
        """Build the thesis prompt."""
        stance_instruction = {
            "supportive": "Present the strongest arguments IN FAVOR of this position.",
            "critical": "Present the main CONCERNS and PROBLEMS with this topic.",
            "neutral": "Present a BALANCED initial assessment of this topic.",
        }.get(self.thesis_stance, "Present your initial position.")

        return f"""Analyze this topic and present your thesis.

Topic: {{task}}

{stance_instruction}

Structure your response:
1. THESIS STATEMENT: Your main claim or position
2. KEY ARGUMENTS: Supporting points (3-5)
3. EVIDENCE: Facts, logic, or examples
4. IMPLICATIONS: What follows if this thesis is correct

Present your thesis:"""

    def _build_antithesis_prompt(self) -> str:
        """Build the antithesis prompt."""
        return """Challenge the thesis with a compelling antithesis.

Topic: {task}

THESIS:
{step.thesis}

Your task is to present the strongest possible COUNTER-POSITION:

1. ANTITHESIS STATEMENT: Your opposing claim
2. CRITIQUES: Identify flaws in the thesis (logical, factual, ethical)
3. COUNTER-ARGUMENTS: Present opposing evidence and reasoning
4. ALTERNATIVE VIEW: What the thesis misses or gets wrong

Present your antithesis:"""

    def _build_synthesis_prompt(self) -> str:
        """Build the synthesis prompt."""
        return """Synthesize the thesis and antithesis into a higher understanding.

Topic: {task}

THESIS:
{step.thesis}

ANTITHESIS:
{step.antithesis}

Your task is to SYNTHESIZE these opposing views:

1. SYNTHESIS STATEMENT: A position that transcends and includes both views
2. RESOLUTION: How the apparent contradiction is resolved
3. PRESERVED INSIGHTS: What's valid from each position
4. TRANSCENDED LIMITATIONS: What both positions missed
5. PRACTICAL IMPLICATIONS: What this synthesis means for action/decision

Present your synthesis:"""

    def _build_meta_transform(self) -> str:
        """Build meta-analysis transformation expression."""
        return """{
    "topic": inputs.get("task", ""),
    "thesis_summary": outputs.get("thesis", {}).get("response", "")[:500],
    "antithesis_summary": outputs.get("antithesis", {}).get("response", "")[:500],
    "synthesis_summary": outputs.get("synthesis", {}).get("response", "")[:500],
    "dialectic_complete": True,
    "agents_used": {
        "thesis": "%s",
        "antithesis": "%s",
        "synthesis": "%s"
    }
}""" % (self.thesis_agent, self.antithesis_agent, self.synthesis_agent)
