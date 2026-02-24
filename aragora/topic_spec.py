from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class TopicSpec:
    """
    A structured data contract for a debate topic to ensure clarity and
    prevent low-quality outputs from ambiguous prompts.
    """
    # The clear, concise title of the debate.
    title: str

    # The primary objective or question to be answered by the debate.
    objective: str

    # A list of key assumptions the agents should operate under.
    assumptions: List[str] = field(default_factory=list)

    # Explicitly defines what is out of scope for the debate.
    non_goals: List[str] = field(default_factory=list)

    # Specific constraints, such as technologies to use, budget limits, or performance targets.
    constraints: List[str] = field(default_factory=list)

    # Criteria for what a successful outcome looks like. This can inform
    # agent evaluation and Elo ranking. [EVID-2]
    evaluation_criteria: List[str] = field(default_factory=list)

    # Optional context or background information.
    context: Optional[str] = None

    # Metadata for tracking and analysis.
    metadata: Dict[str, str] = field(default_factory=dict)
