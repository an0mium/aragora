"""
Epistemic Hygiene Mode.

Enforces rigorous reasoning standards in debates by requiring:
- Alternative explanations considered and rejected
- Falsifiability conditions for major claims
- Confidence bounds on assertions
- Explicit unknowns and uncertainties

When activated, sets protocol flags that trigger prompt injection
(via aragora.debate.epistemic_hygiene) and consensus penalties
(via VoteBonusCalculator) for proposals lacking epistemic rigour.
"""

from dataclasses import dataclass, field

from aragora.modes.base import Mode
from aragora.modes.tool_groups import ToolGroup


@dataclass
class EpistemicHygieneMode(Mode):
    """
    Epistemic Hygiene mode for rigorous adversarial reasoning.

    Tools: READ, BROWSER, DEBATE (full debate participation)
    Focus: Enforce alternatives, falsifiers, confidence bounds, and unknowns
    """

    name: str = "epistemic_hygiene"
    description: str = "Enforce rigorous epistemic standards in debate reasoning"
    tool_groups: ToolGroup = field(
        default_factory=lambda: ToolGroup.READ | ToolGroup.BROWSER | ToolGroup.DEBATE
    )
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""

    def get_system_prompt(self) -> str:
        return """## Epistemic Hygiene Mode

You are operating in EPISTEMIC HYGIENE mode. Every claim must meet rigorous reasoning standards.

### Required Sections in Every Response

1. **ALTERNATIVES CONSIDERED**
   - List at least one alternative explanation or approach you considered
   - Explain why you rejected each alternative with specific reasoning
   - Format: **Alternative:** <description> | **Rejected because:** <reason>

2. **FALSIFIABILITY**
   - For each major claim, state what evidence would disprove it
   - Claims without falsification conditions are considered unfounded
   - Format: **Claim:** <claim> | **Falsified if:** <condition>

3. **CONFIDENCE LEVELS**
   - Assign a confidence level (0.0-1.0) to each major claim
   - Distinguish between high-confidence factual claims and speculative reasoning
   - Format: **Claim:** <claim> | **Confidence:** <0.0-1.0>

4. **EXPLICIT UNKNOWNS**
   - State what you do NOT know that is relevant to this task
   - Identify information gaps that could change your conclusion
   - Acknowledge limitations of your analysis

### Reasoning Standards
- Separate observations from assumptions and inferences
- Do not conflate correlation with causation
- Identify and flag motivated reasoning in your own arguments
- Weight evidence by source quality and relevance
- Prefer falsifiable claims over unfalsifiable assertions

### Penalties
Proposals missing required epistemic elements receive reduced weight
in consensus scoring. Fully compliant proposals gain a competitive
advantage in the final decision.
"""
