"""
Built-in skills for Aragora.

These skills provide core functionality:
- web_search: Search the web for information
- code_execution: Execute code in a sandboxed environment
- knowledge_query: Query the Knowledge Mound
- evidence_fetch: Collect evidence from various sources
- summarization: Summarize long text content
- calculation: Perform safe mathematical calculations
- data_extraction: Extract structured data from text
- fact_check: Verify claims against knowledge base
- file_analysis: Analyze file contents
- translation: Translate text between languages
- pr_reviewer: Autonomous PR code review via multi-agent debate
"""

from ..base import Skill


def register_skills() -> list[Skill]:
    """
    Register all built-in skills.

    Called by the SkillLoader when loading builtin skills.
    """
    skills: list[Skill] = []

    # Original skills
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

    # New skills
    try:
        from .summarization import SummarizationSkill

        skills.append(SummarizationSkill())
    except ImportError:
        pass

    try:
        from .calculation import CalculationSkill

        skills.append(CalculationSkill())
    except ImportError:
        pass

    try:
        from .data_extraction import DataExtractionSkill

        skills.append(DataExtractionSkill())
    except ImportError:
        pass

    try:
        from .fact_check import FactCheckSkill

        skills.append(FactCheckSkill())
    except ImportError:
        pass

    try:
        from .file_analysis import FileAnalysisSkill

        skills.append(FileAnalysisSkill())
    except ImportError:
        pass

    try:
        from .translation import TranslationSkill

        skills.append(TranslationSkill())
    except ImportError:
        pass

    try:
        from .pr_reviewer import PRReviewerSkill

        skills.append(PRReviewerSkill())
    except ImportError:
        pass

    return skills
