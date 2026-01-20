"""
Research Vertical Specialist.

Provides domain expertise for research tasks including literature review,
methodology analysis, statistical review, and scientific writing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from aragora.core import Message
from aragora.verticals.base import VerticalSpecialistAgent
from aragora.verticals.config import (
    ComplianceConfig,
    ComplianceLevel,
    ModelConfig,
    ToolConfig,
    VerticalConfig,
)
from aragora.verticals.registry import VerticalRegistry

logger = logging.getLogger(__name__)


# Research vertical configuration
RESEARCH_CONFIG = VerticalConfig(
    vertical_id="research",
    display_name="Research Specialist",
    description="Expert in research methodology, literature analysis, and scientific writing.",
    domain_keywords=[
        "research",
        "study",
        "experiment",
        "hypothesis",
        "methodology",
        "statistical",
        "data",
        "analysis",
        "sample",
        "results",
        "peer review",
        "publication",
        "citation",
        "literature",
        "IRB",
        "ethics",
        "protocol",
        "findings",
        "conclusion",
    ],
    expertise_areas=[
        "Literature Review",
        "Research Methodology",
        "Statistical Analysis",
        "Scientific Writing",
        "Peer Review",
        "Research Ethics",
        "Data Analysis",
        "Citation Analysis",
        "Meta-Analysis",
    ],
    system_prompt_template="""You are a research specialist with expertise in:

{% for area in expertise_areas %}
- {{ area }}
{% endfor %}

Your role is to provide expert research guidance. You should:

1. **Evaluate Methodology**: Assess research design, sampling, and validity
2. **Analyze Statistics**: Review statistical methods and interpretation
3. **Check Citations**: Verify proper citation and literature coverage
4. **Ensure Ethics**: Flag ethical concerns and compliance issues
5. **Improve Quality**: Suggest improvements for rigor and clarity

{% if compliance_frameworks %}
Compliance Frameworks to Consider:
{% for fw in compliance_frameworks %}
- {{ fw }}
{% endfor %}
{% endif %}

When reviewing research:
- Assess the research design and methodology
- Check for appropriate statistical analysis
- Verify proper citation practices
- Identify potential biases or limitations
- Ensure ethical compliance (IRB, informed consent)

Provide constructive, evidence-based feedback that helps researchers
improve the quality and rigor of their work.""",
    tools=[
        ToolConfig(
            name="arxiv_search",
            description="Search arXiv for preprints",
            connector_type="arxiv",
        ),
        ToolConfig(
            name="pubmed_search",
            description="Search PubMed for medical literature",
            connector_type="pubmed",
        ),
        ToolConfig(
            name="semantic_scholar",
            description="Search Semantic Scholar for papers",
            connector_type="semantic_scholar",
        ),
        ToolConfig(
            name="citation_check",
            description="Verify citations and check for retractions",
            connector_type="crossref",
        ),
    ],
    compliance_frameworks=[
        ComplianceConfig(
            framework="IRB",
            version="current",
            level=ComplianceLevel.ENFORCED,
            rules=["informed_consent", "minimal_risk", "privacy", "vulnerable_populations"],
        ),
        ComplianceConfig(
            framework="CONSORT",
            version="2010",
            level=ComplianceLevel.WARNING,
            rules=["randomization", "blinding", "outcomes", "sample_size", "flow_diagram"],
        ),
        ComplianceConfig(
            framework="PRISMA",
            version="2020",
            level=ComplianceLevel.WARNING,
            rules=["search_strategy", "selection", "synthesis", "bias_assessment"],
        ),
    ],
    model_config=ModelConfig(
        primary_model="claude-sonnet-4",
        primary_provider="anthropic",
        specialist_model="allenai/scibert_scivocab_uncased",
        temperature=0.3,
        top_p=0.9,
        max_tokens=8192,
    ),
    tags=["research", "science", "methodology", "statistics"],
)


@VerticalRegistry.register(
    "research",
    config=RESEARCH_CONFIG,
    description="Research specialist for methodology and literature analysis",
)
class ResearchSpecialist(VerticalSpecialistAgent):
    """
    Research specialist agent.

    Provides expert guidance on:
    - Research methodology
    - Statistical analysis
    - Literature review
    - Scientific writing
    - Research ethics
    """

    # Research methodology patterns
    METHODOLOGY_PATTERNS = {
        "study_design": [
            r"randomized\s+controlled\s+trial|RCT",
            r"cohort\s+study",
            r"case.control\s+study",
            r"cross.sectional",
            r"meta.analysis",
            r"systematic\s+review",
        ],
        "statistical_methods": [
            r"t-test|t\s+test",
            r"ANOVA|analysis\s+of\s+variance",
            r"chi.square|χ²",
            r"regression",
            r"correlation",
            r"confidence\s+interval",
            r"p.value|p\s*<\s*0\.\d+",
        ],
        "sampling": [
            r"random\s+sampl",
            r"convenience\s+sample",
            r"stratified\s+sampl",
            r"sample\s+size",
            r"power\s+analysis",
        ],
        "bias_indicators": [
            r"selection\s+bias",
            r"confirmation\s+bias",
            r"publication\s+bias",
            r"recall\s+bias",
            r"attrition",
        ],
    }

    # Citation patterns
    CITATION_PATTERNS = {
        "apa": r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,\s*\d{4}\)",
        "mla": r"[A-Z][a-z]+(?:\s+[a-z]+)*\s+\d+",
        "chicago": r"\d+\.\s+[A-Z][a-z]+",
        "doi": r"10\.\d{4,}/[^\s]+",
    }

    async def _execute_tool(
        self,
        tool: ToolConfig,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a research tool."""
        tool_name = tool.name

        if tool_name == "arxiv_search":
            return await self._arxiv_search(parameters)
        elif tool_name == "pubmed_search":
            return await self._pubmed_search(parameters)
        elif tool_name == "semantic_scholar":
            return await self._semantic_scholar_search(parameters)
        elif tool_name == "citation_check":
            return await self._citation_check(parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _arxiv_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search arXiv for preprints."""
        return {
            "papers": [],
            "message": "arXiv search not yet implemented - requires arXiv API",
        }

    async def _pubmed_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search PubMed for medical literature."""
        return {
            "articles": [],
            "message": "PubMed search not yet implemented - requires NCBI API",
        }

    async def _semantic_scholar_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search Semantic Scholar."""
        return {
            "papers": [],
            "message": "Semantic Scholar search not yet implemented",
        }

    async def _citation_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check citations for validity and retractions."""
        return {
            "citations": [],
            "retractions": [],
            "message": "Citation check not yet implemented - requires CrossRef API",
        }

    async def _check_framework_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check content against research compliance frameworks."""
        violations = []

        if framework.framework == "IRB":
            violations.extend(await self._check_irb_compliance(content, framework))
        elif framework.framework == "CONSORT":
            violations.extend(await self._check_consort_compliance(content, framework))
        elif framework.framework == "PRISMA":
            violations.extend(await self._check_prisma_compliance(content, framework))

        return violations

    async def _check_irb_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check IRB/Ethics compliance."""
        violations = []
        content_lower = content.lower()

        # Check for human subjects research indicators
        has_human_subjects = any(
            term in content_lower
            for term in [
                "participant",
                "subject",
                "volunteer",
                "patient",
                "interview",
                "survey",
                "questionnaire",
                "blood sample",
                "tissue sample",
            ]
        )

        if has_human_subjects:
            # Informed consent check
            if "informed_consent" in framework.rules or not framework.rules:
                if not re.search(r"informed\s+consent|consent\s+form|consented", content_lower):
                    violations.append(
                        {
                            "framework": "IRB",
                            "rule": "Informed Consent",
                            "severity": "critical",
                            "message": "Human subjects research without informed consent documentation",
                        }
                    )

            # IRB approval check
            if not re.search(
                r"IRB|ethics\s+(?:committee|board)|institutional\s+review", content_lower
            ):
                violations.append(
                    {
                        "framework": "IRB",
                        "rule": "Ethics Approval",
                        "severity": "critical",
                        "message": "Human subjects research without IRB/ethics approval reference",
                    }
                )

            # Vulnerable populations check
            if "vulnerable_populations" in framework.rules or not framework.rules:
                vulnerable_terms = ["children", "minor", "pregnant", "prisoner", "cognitive impair"]
                if any(term in content_lower for term in vulnerable_terms):
                    if not re.search(
                        r"additional\s+protect|special\s+consider|guardian\s+consent", content_lower
                    ):
                        violations.append(
                            {
                                "framework": "IRB",
                                "rule": "Vulnerable Populations",
                                "severity": "high",
                                "message": "Vulnerable population without additional protections noted",
                            }
                        )

        return violations

    async def _check_consort_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check CONSORT compliance for clinical trials."""
        violations = []
        content_lower = content.lower()

        # Check if this is a clinical trial
        is_trial = re.search(r"randomized|clinical\s+trial|RCT", content, re.IGNORECASE)

        if is_trial:
            # Randomization
            if "randomization" in framework.rules or not framework.rules:
                if not re.search(r"random(?:ization|ly|ised|ized)|allocation", content_lower):
                    violations.append(
                        {
                            "framework": "CONSORT",
                            "rule": "Randomization",
                            "severity": "high",
                            "message": "Clinical trial without randomization method described",
                        }
                    )

            # Sample size
            if "sample_size" in framework.rules or not framework.rules:
                if not re.search(
                    r"sample\s+size|power\s+(?:calculation|analysis)|n\s*=\s*\d+", content_lower
                ):
                    violations.append(
                        {
                            "framework": "CONSORT",
                            "rule": "Sample Size",
                            "severity": "medium",
                            "message": "Sample size justification not found",
                        }
                    )

            # Blinding
            if "blinding" in framework.rules or not framework.rules:
                if not re.search(
                    r"blind(?:ed|ing)|mask(?:ed|ing)|placebo|double.blind", content_lower
                ):
                    violations.append(
                        {
                            "framework": "CONSORT",
                            "rule": "Blinding",
                            "severity": "medium",
                            "message": "Blinding/masking not described",
                        }
                    )

        return violations

    async def _check_prisma_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check PRISMA compliance for systematic reviews."""
        violations = []
        content_lower = content.lower()

        # Check if this is a systematic review
        is_review = re.search(r"systematic\s+review|meta.analysis", content, re.IGNORECASE)

        if is_review:
            # Search strategy
            if "search_strategy" in framework.rules or not framework.rules:
                if not re.search(
                    r"search\s+strateg|database|PubMed|Medline|Cochrane", content_lower
                ):
                    violations.append(
                        {
                            "framework": "PRISMA",
                            "rule": "Search Strategy",
                            "severity": "high",
                            "message": "Systematic review without search strategy described",
                        }
                    )

            # Selection criteria
            if "selection" in framework.rules or not framework.rules:
                if not re.search(
                    r"inclusion\s+criteria|exclusion\s+criteria|eligib", content_lower
                ):
                    violations.append(
                        {
                            "framework": "PRISMA",
                            "rule": "Selection Criteria",
                            "severity": "high",
                            "message": "Selection/eligibility criteria not specified",
                        }
                    )

            # Bias assessment
            if "bias_assessment" in framework.rules or not framework.rules:
                if not re.search(
                    r"risk\s+of\s+bias|quality\s+assessment|bias\s+assessment", content_lower
                ):
                    violations.append(
                        {
                            "framework": "PRISMA",
                            "rule": "Risk of Bias",
                            "severity": "medium",
                            "message": "Risk of bias assessment not described",
                        }
                    )

        return violations

    async def _generate_response(
        self,
        task: str,
        system_prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a research analysis response."""
        return Message(
            role="assistant",
            content=f"[Research Specialist Response for: {task}]\n\n"
            f"This would contain expert research methodology guidance.",
            agent=self.name,
        )

    async def analyze_methodology(
        self,
        paper_text: str,
    ) -> Dict[str, Any]:
        """
        Analyze research methodology.

        Args:
            paper_text: Research paper text

        Returns:
            Methodology analysis results
        """
        # Detect study design
        study_design = None
        for pattern in self.METHODOLOGY_PATTERNS.get("study_design", []):
            if re.search(pattern, paper_text, re.IGNORECASE):
                study_design = pattern
                break

        # Detect statistical methods
        stat_methods = []
        for pattern in self.METHODOLOGY_PATTERNS.get("statistical_methods", []):
            if re.search(pattern, paper_text, re.IGNORECASE):
                stat_methods.append(pattern)

        # Detect sampling approach
        sampling = []
        for pattern in self.METHODOLOGY_PATTERNS.get("sampling", []):
            if re.search(pattern, paper_text, re.IGNORECASE):
                sampling.append(pattern)

        # Check for bias indicators
        bias_risks = []
        for pattern in self.METHODOLOGY_PATTERNS.get("bias_indicators", []):
            if re.search(pattern, paper_text, re.IGNORECASE):
                bias_risks.append(pattern)

        # Check compliance
        compliance_violations = await self.check_compliance(paper_text)

        return {
            "study_design": study_design,
            "statistical_methods": stat_methods,
            "sampling_approach": sampling,
            "bias_risks": bias_risks,
            "compliance_violations": compliance_violations,
            "methodology_rating": self._rate_methodology(
                study_design, stat_methods, sampling, bias_risks
            ),
        }

    def _rate_methodology(
        self,
        study_design: Optional[str],
        stat_methods: List[str],
        sampling: List[str],
        bias_risks: List[str],
    ) -> str:
        """Rate overall methodology quality."""
        score = 0

        # Study design clarity
        if study_design:
            score += 2

        # Statistical rigor
        if len(stat_methods) >= 2:
            score += 2
        elif stat_methods:
            score += 1

        # Sampling description
        if sampling:
            score += 1

        # Bias awareness (mentioning bias is good)
        if bias_risks:
            score += 1

        if score >= 5:
            return "strong"
        elif score >= 3:
            return "adequate"
        else:
            return "needs_improvement"

    async def analyze_citations(
        self,
        paper_text: str,
    ) -> Dict[str, Any]:
        """
        Analyze citation patterns in a paper.

        Args:
            paper_text: Research paper text

        Returns:
            Citation analysis results
        """
        citation_counts = {}

        for style, pattern in self.CITATION_PATTERNS.items():
            matches = re.findall(pattern, paper_text)
            citation_counts[style] = len(matches)

        # Find DOIs
        dois = re.findall(self.CITATION_PATTERNS["doi"], paper_text)

        # Estimate total citations
        total_citations = max(citation_counts.values()) if citation_counts else 0

        return {
            "citation_style_detected": (
                max(citation_counts, key=citation_counts.get) if citation_counts else None
            ),
            "estimated_citation_count": total_citations,
            "dois_found": len(dois),
            "citation_density": total_citations / max(len(paper_text.split()) / 1000, 1),
        }
