"""
Decision Playbooks — pre-built end-to-end decision workflows.

A playbook composes a deliberation template, vertical weight profile,
compliance artifacts, agent criteria, output format, and approval gates
into a single runnable package.
"""

from .models import Playbook, PlaybookStep, ApprovalGate
from .registry import PlaybookRegistry, get_playbook_registry

__all__ = [
    "Playbook",
    "PlaybookStep",
    "ApprovalGate",
    "PlaybookRegistry",
    "get_playbook_registry",
]
