"""
Defines the core data structure for a validated and executable task.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class TaskBriefV1:
    """
    A structured and versioned representation of a task to be executed by the system.
    This artifact is the primary output of the Topic Resolution pipeline.
    
    Attributes:
        schema_version: The version of the TaskBrief schema.
        goal: A clear, concise statement of the desired outcome.
        constraints: A list of specific limitations or rules that must be followed.
        success_criteria: A list of measurable conditions that define task completion.
        confidence: A score (0.0-1.0) representing the system's confidence in this interpretation.
        provenance: Metadata tracking the origin and derivation of the brief.
        assumptions: A list of assumptions made during the interpretation of the input.
        requires_user_confirmation: A flag indicating if the task requires explicit user
                                    approval before execution of any side effects.
    """
    schema_version: str = "1.0"
    goal: str
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    confidence: float = 1.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    requires_user_confirmation: bool = False