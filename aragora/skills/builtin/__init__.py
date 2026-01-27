"""
Built-in skills for Aragora.

These skills provide core functionality:
- web_search: Search the web for information
- code_execution: Execute code in a sandboxed environment
- knowledge_query: Query the Knowledge Mound
- evidence_fetch: Collect evidence from various sources
"""

from typing import List

from ..base import Skill


def register_skills() -> List[Skill]:
    """
    Register all built-in skills.

    Called by the SkillLoader when loading builtin skills.
    """
    skills: List[Skill] = []

    # Try to import each builtin skill
    try:
        from .web_search import WebSearchSkill

        skills.append(WebSearchSkill())
    except ImportError:
        pass

    try:
        from .code_execution import CodeExecutionSkill

        skills.append(CodeExecutionSkill())
    except ImportError:
        pass

    try:
        from .knowledge_query import KnowledgeQuerySkill

        skills.append(KnowledgeQuerySkill())
    except ImportError:
        pass

    try:
        from .evidence_fetch import EvidenceFetchSkill

        skills.append(EvidenceFetchSkill())
    except ImportError:
        pass

    return skills
