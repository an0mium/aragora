"""
Question classification and persona assignment for debates.

Uses Claude to analyze debate questions and recommend appropriate
agent personas from the available pool for more focused discussions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from aragora.agents.personas import DEFAULT_PERSONAS, EXPERTISE_DOMAINS

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Question categories with associated expertise domains
QUESTION_CATEGORIES = {
    "technical": {
        "description": "Software, technology, engineering questions",
        "keywords": ["code", "software", "api", "database", "architecture", "performance", "bug", "deploy"],
        "domains": ["architecture", "performance", "api_design", "database", "security", "testing"],
    },
    "legal": {
        "description": "Law, regulations, compliance, lawsuits",
        "keywords": ["law", "legal", "lawsuit", "court", "regulation", "compliance", "rights", "contract"],
        "domains": ["sox_compliance", "pci_dss", "hipaa", "gdpr", "finra", "fda_21_cfr"],
    },
    "security": {
        "description": "Cybersecurity, privacy, data protection",
        "keywords": ["security", "privacy", "encryption", "breach", "hack", "vulnerability", "authentication"],
        "domains": ["security", "encryption", "access_control", "data_privacy"],
    },
    "scientific": {
        "description": "Science, research, methodology questions",
        "keywords": ["research", "study", "science", "experiment", "data", "evidence", "hypothesis"],
        "domains": ["testing", "documentation", "error_handling"],
    },
    "political": {
        "description": "Politics, policy, government, elections",
        "keywords": ["policy", "government", "election", "political", "congress", "president", "vote"],
        "domains": ["fisma", "nist_800_53", "audit_trails"],
    },
    "ethical": {
        "description": "Ethics, morality, philosophy questions",
        "keywords": ["ethics", "moral", "right", "wrong", "should", "philosophy", "values"],
        "domains": ["data_privacy", "access_control"],
    },
    "financial": {
        "description": "Finance, economics, markets, money",
        "keywords": ["finance", "economic", "market", "money", "investment", "bank", "stock", "trade"],
        "domains": ["sox_compliance", "finra", "audit_trails"],
    },
    "healthcare": {
        "description": "Health, medicine, medical questions",
        "keywords": ["health", "medical", "patient", "hospital", "treatment", "disease", "doctor"],
        "domains": ["hipaa", "fda_21_cfr", "data_privacy"],
    },
}

# Map personas to agent providers
PERSONA_TO_AGENT = {
    # Technical personas
    "claude": "anthropic-api",
    "codex": "openai-api",
    "gemini": "gemini",
    "grok": "grok",
    "qwen": "openrouter:qwen/qwen3-235b-a22b",
    "qwen-max": "openrouter:qwen/qwen-max",
    "yi": "openrouter:01-ai/yi-large",
    "deepseek": "openrouter:deepseek/deepseek-chat-v3",
    "deepseek-r1": "openrouter:deepseek/deepseek-r1",
    "kimi": "openrouter:moonshotai/kimi-k2",
    # Compliance personas
    "sox": "anthropic-api",
    "pci_dss": "anthropic-api",
    "hipaa": "anthropic-api",
    "gdpr": "anthropic-api",
    "finra": "anthropic-api",
    "fda_21_cfr": "anthropic-api",
    "fisma": "anthropic-api",
    "ccpa": "anthropic-api",
    "iso_27001": "anthropic-api",
    # Specialist personas
    "security_engineer": "openai-api",
    "performance_engineer": "openai-api",
    "data_architect": "anthropic-api",
    "devops_engineer": "gemini",
    "accessibility": "gemini",
    # Philosophical personas
    "philosopher": "anthropic-api",
    "humanist": "openai-api",
    "existentialist": "anthropic-api",
}


@dataclass
class QuestionClassification:
    """Result of question classification."""

    category: str
    complexity: str  # "simple", "moderate", "complex"
    requires_expertise: list[str] = field(default_factory=list)
    recommended_personas: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


class QuestionClassifier:
    """Classifies questions and assigns appropriate debate personas."""

    def __init__(self, client: Optional["anthropic.Anthropic"] = None):
        """Initialize the classifier.

        Args:
            client: Optional Anthropic client. If not provided, will be
                    created when needed using ANTHROPIC_API_KEY.
        """
        self._client = client

    @property
    def client(self) -> "anthropic.Anthropic":
        """Get or create the Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def classify_simple(self, question: str) -> QuestionClassification:
        """Quick classification using keyword matching.

        Use this for fast classification without API calls.
        """
        question_lower = question.lower()
        category_scores: dict[str, int] = {}

        # Score each category by keyword matches
        for category, info in QUESTION_CATEGORIES.items():
            score = sum(1 for kw in info["keywords"] if kw in question_lower)
            if score > 0:
                category_scores[category] = score

        # Determine primary category
        if category_scores:
            category = max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            category = "technical"  # Default fallback

        # Determine complexity based on question length and structure
        word_count = len(question.split())
        if word_count < 20:
            complexity = "simple"
        elif word_count < 50:
            complexity = "moderate"
        else:
            complexity = "complex"

        # Get relevant expertise domains
        domains = QUESTION_CATEGORIES.get(category, {}).get("domains", [])

        # Recommend personas based on domains
        personas = self._select_personas_for_domains(domains, complexity)

        return QuestionClassification(
            category=category,
            complexity=complexity,
            requires_expertise=domains[:3],
            recommended_personas=personas,
            confidence=0.6,
            reasoning=f"Keyword matching: {category_scores}",
        )

    def classify(self, question: str) -> QuestionClassification:
        """Full classification using Claude for analysis.

        This provides more accurate persona recommendations but requires
        an API call.
        """
        # Get list of available personas
        available_personas = list(DEFAULT_PERSONAS.keys())

        prompt = f"""Analyze this debate question and recommend the best agent personas.

Question: {question}

Available personas: {json.dumps(available_personas)}

Available expertise domains: {json.dumps(EXPERTISE_DOMAINS)}

Respond in JSON format:
{{
    "category": "one of: technical, legal, security, scientific, political, ethical, financial, healthcare, general",
    "complexity": "one of: simple, moderate, complex",
    "requires_expertise": ["list of 2-4 relevant expertise domains"],
    "recommended_personas": ["list of 3-5 persona names from the available list"],
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation of your choices"
}}

Guidelines:
- For legal/regulatory questions, include compliance personas (sox, hipaa, gdpr, etc.)
- For technical questions, mix different AI providers for diverse perspectives
- For ethical/philosophical questions, include philosopher, humanist, or existentialist
- Always include at least one contrarian/skeptic persona for balance
- Prefer personas whose expertise aligns with the question domain"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content = response.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content)

            # Validate personas exist
            valid_personas = [p for p in result.get("recommended_personas", []) if p in available_personas]
            if len(valid_personas) < 2:
                # Fallback to simple classification
                return self.classify_simple(question)

            return QuestionClassification(
                category=result.get("category", "general"),
                complexity=result.get("complexity", "moderate"),
                requires_expertise=result.get("requires_expertise", [])[:4],
                recommended_personas=valid_personas[:5],
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"Classification failed, using simple fallback: {e}")
            return self.classify_simple(question)

    def _select_personas_for_domains(
        self, domains: list[str], complexity: str
    ) -> list[str]:
        """Select personas that have expertise in given domains."""
        persona_scores: dict[str, float] = {}

        for persona_name, persona in DEFAULT_PERSONAS.items():
            score = 0.0
            for domain in domains:
                score += persona.expertise.get(domain, 0.0)
            if score > 0:
                persona_scores[persona_name] = score

        # Sort by score and select top personas
        sorted_personas = sorted(persona_scores.items(), key=lambda x: x[1], reverse=True)

        # Select based on complexity
        if complexity == "simple":
            count = 3
        elif complexity == "moderate":
            count = 4
        else:
            count = 5

        selected = [p[0] for p in sorted_personas[:count]]

        # Ensure diversity: add a contrarian if not present
        if "grok" not in selected and "contrarian" not in selected:
            if "grok" in DEFAULT_PERSONAS:
                selected = selected[:count - 1] + ["grok"]

        return selected

    def get_agent_string(self, classification: QuestionClassification) -> str:
        """Convert persona recommendations to agent string for debate config.

        Returns comma-separated agent identifiers like:
        "anthropic-api:claude,openai-api:codex,gemini:gemini"
        """
        agents = []
        seen_providers = set()

        for persona in classification.recommended_personas:
            provider = PERSONA_TO_AGENT.get(persona, "anthropic-api")

            # Avoid duplicate providers for diversity
            base_provider = provider.split(":")[0]
            if base_provider in seen_providers and len(agents) >= 2:
                continue
            seen_providers.add(base_provider)

            # Format: "provider:persona" or just "provider" if persona matches
            if ":" in provider:
                agents.append(f"{provider}:{persona}")
            else:
                agents.append(f"{provider}:{persona}")

        return ",".join(agents)


def classify_and_assign_agents(question: str, use_llm: bool = True) -> tuple[str, QuestionClassification]:
    """Convenience function to classify question and get agent string.

    Args:
        question: The debate question
        use_llm: Whether to use Claude for classification (slower but more accurate)

    Returns:
        Tuple of (agent_string, classification)
    """
    classifier = QuestionClassifier()

    if use_llm:
        classification = classifier.classify(question)
    else:
        classification = classifier.classify_simple(question)

    agent_string = classifier.get_agent_string(classification)

    logger.info(
        f"Question classified: category={classification.category}, "
        f"complexity={classification.complexity}, "
        f"personas={classification.recommended_personas}"
    )

    return agent_string, classification
