"""
Essay Workflow - Complete pipeline for synthesizing essays from conversations.

This module provides the full workflow for:
1. Loading ChatGPT/Claude conversation exports
2. Extracting intellectual claims and positions
3. Clustering claims by topic
4. Running multi-agent debate to stress-test positions
5. Finding scholarly attribution for claims
6. Weaving seed essays with extracted ideas
7. Generating final essay output with strengthened arguments

Usage:
    from aragora.pipelines.essay_workflow import EssayWorkflow

    workflow = EssayWorkflow()

    # Load conversation exports
    await workflow.load_exports("/path/to/exports")

    # Set seed essay (optional starting point)
    await workflow.set_seed_essay("path/to/seed_essay.md")

    # Run the full pipeline
    result = await workflow.run(
        title="AI, Evolution, and the Myth of Final States",
        debate_rounds=3,
        find_attribution=True,
    )

    # Export final essay
    workflow.export_essay("output/final_essay.md")
"""

from __future__ import annotations

__all__ = [
    "EssayWorkflow",
    "WorkflowConfig",
    "WorkflowResult",
    "DebateResult",
    "SeedEssay",
]

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from aragora.connectors.conversation_ingestor import (
    ConversationIngestorConnector,
    ClaimExtraction,
    Conversation,
    ConversationExport,
)
from aragora.pipelines.essay_synthesis import (
    EssaySynthesisPipeline,
    SynthesisConfig,
    TopicCluster,
    AttributedClaim,
    EssayOutline,
    EssaySection,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for the essay workflow."""

    # Claim extraction
    min_claim_length: int = 50
    max_claims_per_conversation: int = 50
    claim_confidence_threshold: float = 0.4

    # Topic clustering
    min_cluster_size: int = 2
    max_clusters: int = 30
    similarity_threshold: float = 0.2

    # Debate settings (defaults match aragora's 9-round, 8-agent standard)
    # Models verified as of Feb 2026:
    # - GPT-5.2 (latest OpenAI, 40% faster as of Feb 3, 2026)
    # - Claude Opus 4.5 (top writing/reasoning)
    # - Grok 4 (with Grok 4 Heavy for complex tasks)
    # - Gemini 3 Pro (1501 Elo on LMArena)
    # - Mistral Large 3 (675B params, Apache 2.0)
    # - DeepSeek V3/R1 (R2 not yet released)
    enable_debate: bool = True
    debate_rounds: int = 9
    debate_agents: list[str] = field(
        default_factory=lambda: [
            "anthropic-api",  # Claude Opus 4.5 - top writing quality & reasoning
            "openai-api",     # GPT-5.2 - strong synthesis, general reasoning
            "grok",           # Grok 4 - contrarian perspectives, real-time search
            "gemini",         # Gemini 3 Pro - 1501 Elo, strong multimodal
            "mistral",        # Mistral Large 3 - 675B params, analytical
            "deepseek",       # DeepSeek V3/R1 - technical rigor, cost-efficient
            "qwen",           # Qwen 2.5 - multilingual, long context
            "kimi",           # Kimi K2 - strong on Chinese sources
        ]
    )
    claims_to_debate: int = 10  # Top N claims to stress-test

    # Attribution
    enable_attribution: bool = True
    max_sources_per_claim: int = 5
    attribution_connectors: list[str] = field(default_factory=lambda: ["arxiv", "semantic_scholar"])

    # Essay generation
    target_word_count: int = 50000
    include_counterarguments: bool = True
    include_debate_summaries: bool = True
    voice_style: str = "analytical"  # analytical, conversational, academic

    # Seed filtering
    enable_seed_filtering: bool = True
    seed_min_score: float = 0.05
    max_claims: int | None = 300

    # Output
    output_format: Literal["markdown", "json", "html"] = "markdown"


@dataclass
class SeedEssay:
    """A seed essay to weave into the synthesis."""

    title: str
    content: str
    sections: list[dict[str, str]] = field(default_factory=list)
    key_claims: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    keywords: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_markdown(cls, path: str | Path) -> "SeedEssay":
        """Load seed essay from markdown file."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")

        # Extract title from first H1
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else path.stem

        # Extract sections
        sections = []
        section_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        matches = list(section_pattern.finditer(content))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            sections.append({
                "level": level,
                "heading": heading,
                "content": section_content[:2000],  # Truncate for analysis
            })

        # Extract key claims using patterns
        key_claims = []
        claim_patterns = [
            r"(?:The|This|My) (?:core|key|main|central) (?:argument|claim|thesis|position) is[:\s]+(.{50,300})",
            r"I argue that\s+(.{50,300})",
            r"The future (?:will|is likely to)\s+(.{50,200})",
        ]
        for pattern in claim_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                key_claims.append(match.group(1).strip())

        # Extract themes from headings
        themes = [s["heading"] for s in sections if s["level"] <= 2]

        # Extract keywords for claim scoring
        keywords = cls._extract_keywords(content)

        return cls(
            title=title,
            content=content,
            sections=sections,
            key_claims=key_claims,
            themes=themes,
            keywords=keywords,
            metadata={"source_file": str(path)},
        )

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract significant keywords from text for claim scoring."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "that", "this", "these", "those", "it", "its", "i", "you", "we", "they",
            "what", "which", "who", "how", "when", "where", "why", "think", "believe",
            "about", "more", "some", "any", "just", "only", "also", "very", "really",
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        return {w for w in words if w not in stopwords}


@dataclass
class DebateResult:
    """Result from debating a claim."""

    original_claim: str
    strengthened_claim: str | None = None
    counterarguments: list[str] = field(default_factory=list)
    rebuttals: list[str] = field(default_factory=list)
    consensus_reached: bool = False
    debate_summary: str = ""
    agent_positions: dict[str, str] = field(default_factory=dict)
    confidence_change: float = 0.0

    def to_dict(self) -> dict:
        return {
            "original_claim": self.original_claim,
            "strengthened_claim": self.strengthened_claim,
            "counterarguments": self.counterarguments,
            "rebuttals": self.rebuttals,
            "consensus_reached": self.consensus_reached,
            "debate_summary": self.debate_summary,
            "agent_positions": self.agent_positions,
            "confidence_change": self.confidence_change,
        }


@dataclass
class WorkflowResult:
    """Complete result from the essay workflow."""

    title: str
    thesis: str
    outline: EssayOutline
    claims: list[ClaimExtraction]
    clusters: list[TopicCluster]
    attributed_claims: list[AttributedClaim]
    debate_results: list[DebateResult]
    steelman_claims: dict[str, str]
    seed_essay: SeedEssay | None
    final_essay: str
    statistics: dict
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "thesis": self.thesis,
            "outline": self.outline.to_dict(),
            "claim_count": len(self.claims),
            "cluster_count": len(self.clusters),
            "attributed_count": len(self.attributed_claims),
            "debate_count": len(self.debate_results),
            "steelman_count": len(self.steelman_claims),
            "steelman_claims": self.steelman_claims,
            "has_seed_essay": self.seed_essay is not None,
            "statistics": self.statistics,
            "generated_at": self.generated_at.isoformat(),
        }


class EssayWorkflow:
    """
    Complete workflow for synthesizing essays from conversation exports.

    This class orchestrates the full pipeline:
    1. Load and parse conversation exports
    2. Extract intellectual claims
    3. Cluster by topic
    4. Run multi-agent debates to stress-test positions
    5. Find scholarly attribution
    6. Weave seed essay with extracted ideas
    7. Generate final essay
    """

    def __init__(
        self,
        config: WorkflowConfig | None = None,
    ):
        self.config = config or WorkflowConfig()

        # Create synthesis config from workflow config
        synthesis_config = SynthesisConfig(
            min_claim_length=self.config.min_claim_length,
            max_claims_per_conversation=self.config.max_claims_per_conversation,
            claim_confidence_threshold=self.config.claim_confidence_threshold,
            min_cluster_size=self.config.min_cluster_size,
            max_clusters=self.config.max_clusters,
            similarity_threshold=self.config.similarity_threshold,
            target_word_count=self.config.target_word_count,
            include_counterarguments=self.config.include_counterarguments,
        )

        self.pipeline = EssaySynthesisPipeline(config=synthesis_config)
        # Use the pipeline's ingestor to avoid split state
        self.ingestor = self.pipeline.ingestor

        # State
        self._exports: list[ConversationExport] = []
        self._seed_essay: SeedEssay | None = None
        self._claims: list[ClaimExtraction] = []
        self._all_claims: list[ClaimExtraction] = []
        self._clusters: list[TopicCluster] = []
        self._attributed_claims: list[AttributedClaim] = []
        self._debate_results: list[DebateResult] = []
        self._steelman_claims: dict[str, str] = {}
        self._seed_scores_preview: list[dict] = []
        self._arena = None  # Lazy-loaded

    # =========================================================================
    # Loading
    # =========================================================================

    async def load_exports(
        self,
        path: str | Path,
        recursive: bool = True,
    ) -> list[ConversationExport]:
        """
        Load conversation exports from file or directory.

        Args:
            path: Path to export file or directory
            recursive: Search subdirectories for exports

        Returns:
            List of loaded exports
        """
        path = Path(path)

        if path.is_file():
            export = self.ingestor.load_export(path)
            self._exports.append(export)
            return [export]

        elif path.is_dir():
            exports = []
            pattern = "**/*.json" if recursive else "*.json"

            for file_path in path.glob(pattern):
                try:
                    export = self.ingestor.load_export(file_path)
                    exports.append(export)
                    self._exports.append(export)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            return exports

        else:
            raise FileNotFoundError(f"Path not found: {path}")

    async def set_seed_essay(
        self,
        path: str | Path | None = None,
        content: str | None = None,
        title: str = "Seed Essay",
    ) -> SeedEssay | None:
        """
        Set a seed essay to weave into the synthesis.

        Args:
            path: Path to markdown file
            content: Raw content (alternative to path)
            title: Essay title (used if content provided directly)

        Returns:
            Parsed SeedEssay
        """
        if path:
            self._seed_essay = SeedEssay.from_markdown(path)
        elif content:
            self._seed_essay = SeedEssay(
                title=title,
                content=content,
                keywords=SeedEssay._extract_keywords(content),
            )
        else:
            self._seed_essay = None

        return self._seed_essay

    # =========================================================================
    # Extraction and Clustering
    # =========================================================================

    async def extract_claims(
        self,
        custom_patterns: list[str] | None = None,
    ) -> list[ClaimExtraction]:
        """
        Extract claims from all loaded conversations.

        Also extracts claims from seed essay if set.
        """
        # Extract from conversations via pipeline
        self._claims = self.pipeline.extract_all_claims(custom_patterns)

        # Add claims from seed essay
        if self._seed_essay:
            for claim_text in self._seed_essay.key_claims:
                self._claims.append(
                    ClaimExtraction(
                        claim=claim_text,
                        context="From seed essay",
                        conversation_id="seed_essay",
                        confidence=0.9,  # High confidence for explicit claims
                        claim_type="assertion",
                    )
                )

        self._all_claims = list(self._claims)

        # Apply seed filtering if enabled
        if self._seed_essay and self.config.enable_seed_filtering:
            self._claims, self._seed_scores_preview = self.filter_claims_by_seed(
                claims=self._claims,
                min_score=self.config.seed_min_score,
                max_claims=self.config.max_claims,
            )

        logger.info(f"Extracted {len(self._claims)} total claims (filtered)")
        return self._claims

    async def cluster_claims(self) -> list[TopicCluster]:
        """Cluster extracted claims by topic."""
        if not self._claims:
            await self.extract_claims()

        self._clusters = self.pipeline.cluster_claims(self._claims)

        # If we have a seed essay, try to align clusters with seed themes
        if self._seed_essay and self._seed_essay.themes:
            self._align_clusters_with_seed()

        return self._clusters

    def _align_clusters_with_seed(self) -> None:
        """Align clusters with seed essay themes."""
        if not self._seed_essay:
            return

        seed_themes = set(t.lower() for t in self._seed_essay.themes)

        for cluster in self._clusters:
            # Check if cluster name matches any seed theme
            for theme in seed_themes:
                if theme in cluster.name.lower() or cluster.name.lower() in theme:
                    cluster.metadata["aligned_with_seed"] = True
                    cluster.metadata["seed_theme"] = theme
                    break

    # =========================================================================
    # Debate
    # =========================================================================

    async def debate_claims(
        self,
        claims: list[ClaimExtraction] | None = None,
        max_claims: int | None = None,
    ) -> list[DebateResult]:
        """
        Run multi-agent debate on top claims to stress-test them.

        Args:
            claims: Claims to debate (uses top claims from clusters if not provided)
            max_claims: Maximum claims to debate

        Returns:
            List of DebateResults with strengthened claims and counterarguments
        """
        if not self.config.enable_debate:
            logger.info("Debate disabled in config, skipping")
            return []

        max_claims = max_claims or self.config.claims_to_debate

        if not self._clusters:
            await self.cluster_claims()

        # Select claims to debate
        if claims is None:
            # Get representative claims from top clusters
            claims = []
            for cluster in sorted(self._clusters, key=lambda c: -c.coherence_score):
                if cluster.representative_claim:
                    claims.append(cluster.representative_claim)
                if len(claims) >= max_claims:
                    break

        claims = claims[:max_claims]

        # Run debates
        self._debate_results = []
        for claim in claims:
            result = await self._debate_single_claim(claim)
            self._debate_results.append(result)

        logger.info(f"Completed {len(self._debate_results)} debates")
        return self._debate_results

    async def _debate_single_claim(self, claim: ClaimExtraction) -> DebateResult:
        """Run debate on a single claim."""
        # Attempt to run real multi-agent debate via Arena.
        try:
            from aragora import Arena, DebateProtocol, Environment, create_agent
            from aragora.agents import list_available_agents

            available = set(list_available_agents().keys())
            agent_specs = [a for a in self.config.debate_agents if a in available]

            agents = []
            roles = ["proposer", "critic", "synthesizer"]
            for idx, agent_type in enumerate(agent_specs):
                try:
                    role = roles[min(idx, len(roles) - 1)]
                    agent = create_agent(agent_type, name=f"{agent_type}_{idx}", role=role)
                    agents.append(agent)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping agent %s: %s", agent_type, exc)

            if len(agents) >= 2:
                env = Environment(
                    task=f"Stress-test and strengthen this claim:\n\n{claim.claim}",
                    context=f"Context: {claim.context}",
                    roles=[a.role for a in agents],
                    max_rounds=self.config.debate_rounds,
                )
                protocol = DebateProtocol(rounds=self.config.debate_rounds, consensus="majority")
                arena = Arena(environment=env, agents=agents, protocol=protocol)
                result = await arena.run()

                # Extract counterarguments from critiques
                counterarguments = []
                for critique in getattr(result, "critiques", []) or []:
                    counterarguments.extend(critique.issues or [])
                    if critique.reasoning:
                        counterarguments.append(critique.reasoning)
                counterarguments = [c for c in counterarguments if c]

                strengthened = getattr(result, "final_answer", None)
                consensus = bool(getattr(result, "consensus_reached", False))

                return DebateResult(
                    original_claim=claim.claim,
                    strengthened_claim=strengthened,
                    counterarguments=counterarguments,
                    rebuttals=[],
                    consensus_reached=consensus,
                    debate_summary="Arena debate completed",
                    agent_positions=getattr(result, "proposals", {}) or {},
                    confidence_change=0.1 if consensus else -0.05,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Arena debate failed; falling back to heuristic critique: %s", exc)

        # Fallback: heuristic counterarguments
        counterarguments = self._generate_counterarguments(claim)
        rebuttals = self._generate_rebuttals(claim, counterarguments)
        strengthened = self._strengthen_claim(claim, counterarguments, rebuttals)

        return DebateResult(
            original_claim=claim.claim,
            strengthened_claim=strengthened,
            counterarguments=counterarguments,
            rebuttals=rebuttals,
            consensus_reached=True,
            debate_summary=f"Claim stress-tested against {len(counterarguments)} counterarguments (heuristic)",
            confidence_change=0.1 if strengthened else -0.1,
        )

    def _generate_counterarguments(self, claim: ClaimExtraction) -> list[str]:
        """Generate potential counterarguments for a claim."""
        # Pattern-based counterargument generation
        counterarguments = []

        # Check for absolute claims
        if re.search(r"\b(always|never|all|none|every)\b", claim.claim, re.I):
            counterarguments.append(
                "This claim uses absolute language; edge cases may exist that challenge its universality."
            )

        # Check for causal claims
        if re.search(r"\b(causes?|leads? to|results? in)\b", claim.claim, re.I):
            counterarguments.append(
                "Correlation does not imply causation; alternative explanations may exist."
            )

        # Check for predictive claims
        if re.search(r"\b(will|inevitable|certain|bound to)\b", claim.claim, re.I):
            counterarguments.append(
                "Predictions about complex systems face fundamental uncertainty limits."
            )

        # Add generic counterargument
        if not counterarguments:
            counterarguments.append(
                "Alternative frameworks may provide equally valid interpretations of this phenomenon."
            )

        return counterarguments

    def _generate_rebuttals(
        self,
        claim: ClaimExtraction,
        counterarguments: list[str],
    ) -> list[str]:
        """Generate rebuttals to counterarguments."""
        rebuttals = []

        for counter in counterarguments:
            if "absolute" in counter.lower():
                rebuttals.append(
                    "While edge cases exist, the claim describes the dominant pattern "
                    "observable across most relevant cases."
                )
            elif "correlation" in counter.lower():
                rebuttals.append(
                    "The causal mechanism is supported by theoretical frameworks "
                    "and converging lines of evidence from multiple domains."
                )
            elif "prediction" in counter.lower():
                rebuttals.append(
                    "The claim acknowledges uncertainty via confidence intervals; "
                    "it describes probability distributions, not certainties."
                )
            else:
                rebuttals.append(
                    "The claim can be refined to accommodate this concern "
                    "without undermining its core insight."
                )

        return rebuttals

    def _strengthen_claim(
        self,
        claim: ClaimExtraction,
        counterarguments: list[str],
        rebuttals: list[str],
    ) -> str:
        """Strengthen claim based on debate."""
        strengthened = claim.claim

        # Add qualifiers if absolute language detected
        if re.search(r"\b(always|never|all|none|every)\b", claim.claim, re.I):
            strengthened = re.sub(r"\balways\b", "typically", strengthened, flags=re.I)
            strengthened = re.sub(r"\bnever\b", "rarely", strengthened, flags=re.I)
            strengthened = re.sub(r"\ball\b", "most", strengthened, flags=re.I)

        return strengthened

    # =========================================================================
    # Attribution
    # =========================================================================

    async def find_attribution(
        self,
        claims: list[ClaimExtraction] | None = None,
    ) -> list[AttributedClaim]:
        """
        Find scholarly attribution for claims.

        Uses configured connectors to search for supporting evidence.
        """
        if not self.config.enable_attribution:
            logger.info("Attribution disabled in config, skipping")
            return []

        claims = claims or self._claims

        # Ensure connectors are present
        if not self.pipeline.connectors:
            try:
                from aragora.connectors import ArXivConnector, SemanticScholarConnector, CrossRefConnector

                self.pipeline.connectors = {
                    "arxiv": ArXivConnector(),
                    "semantic_scholar": SemanticScholarConnector(),
                    "crossref": CrossRefConnector(),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Attribution connectors unavailable: %s", exc)
                self.pipeline.connectors = {}

        # Use pipeline's attribution finding
        self._attributed_claims = await self.pipeline.find_attribution(
            claims=claims,
            connectors=self.config.attribution_connectors,
        )

        return self._attributed_claims

    # =========================================================================
    # Steelman Pass
    # =========================================================================

    def _steelman_claim(self, claim: ClaimExtraction) -> str:
        """Generate a stronger, more defensible version of a claim."""
        text = claim.claim.strip()
        text = re.sub(r"\balways\b", "typically", text, flags=re.I)
        text = re.sub(r"\bnever\b", "rarely", text, flags=re.I)
        text = re.sub(r"\ball\b", "most", text, flags=re.I)
        text = re.sub(r"\bnone\b", "few", text, flags=re.I)

        if self.config.voice_style == "academic":
            return (
                "A more defensible framing is that "
                f"{text}, contingent on specific institutional, temporal, and incentive structures."
            )
        if self.config.voice_style == "conversational":
            return (
                "A stronger version is: "
                f"{text}, especially under the conditions we actually observe in practice."
            )
        return (
            "A stronger version is that "
            f"{text}, with scope limits and explicit boundary conditions."
        )

    def steelman_claims(self, claims: list[ClaimExtraction] | None = None) -> dict[str, str]:
        """Steelman claims using debate results where available."""
        claims = claims or self._claims
        steelman_map: dict[str, str] = {}

        debate_map = {d.original_claim: d for d in self._debate_results}

        for claim in claims:
            if claim.claim in debate_map and debate_map[claim.claim].strengthened_claim:
                steelman_map[claim.claim] = debate_map[claim.claim].strengthened_claim  # type: ignore[assignment]
            else:
                steelman_map[claim.claim] = self._steelman_claim(claim)

        self._steelman_claims = steelman_map
        return steelman_map

    # =========================================================================
    # Seed-Based Scoring
    # =========================================================================

    def score_claim_by_seed(self, claim: ClaimExtraction) -> float:
        """Score a claim's relevance to the seed essay."""
        if not self._seed_essay or not self._seed_essay.keywords:
            return 0.0
        claim_keywords = SeedEssay._extract_keywords(claim.claim)
        if not claim_keywords:
            return 0.0
        return len(claim_keywords & self._seed_essay.keywords) / max(len(self._seed_essay.keywords), 1)

    def filter_claims_by_seed(
        self,
        claims: list[ClaimExtraction] | None = None,
        min_score: float = 0.05,
        max_claims: int | None = None,
    ) -> tuple[list[ClaimExtraction], list[dict]]:
        """
        Filter and rank claims by seed relevance.

        Args:
            claims: Claims to filter (uses self._claims if None)
            min_score: Minimum relevance score
            max_claims: Maximum claims to keep

        Returns:
            Tuple of (filtered_claims, score_preview)
        """
        claims = claims or self._claims
        if not self._seed_essay:
            return claims, []

        scored = [(c, self.score_claim_by_seed(c)) for c in claims]
        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [c for c, score in scored if score >= min_score]
        if not filtered:
            # Fall back to top N if nothing meets threshold
            filtered = [c for c, _ in scored[: max_claims or len(scored)]]

        if max_claims is not None:
            filtered = filtered[:max_claims]

        preview = [{"claim": c.claim[:100], "score": round(score, 4)} for c, score in scored[:50]]
        return filtered, preview

    # =========================================================================
    # Thread Generation
    # =========================================================================

    def generate_thread_skeleton(
        self,
        outline: EssayOutline,
        max_posts: int = 15,
    ) -> str:
        """
        Generate an X/Twitter thread skeleton from the essay outline.

        Args:
            outline: Essay outline to convert
            max_posts: Maximum number of posts in thread

        Returns:
            Markdown-formatted thread skeleton
        """
        lines = []
        lines.append(f"1/ {outline.title}")
        lines.append("")
        lines.append(f"2/ {outline.thesis[:250]}...")
        lines.append("")

        idx = 3
        for section in outline.sections[:max_posts - 3]:
            # Use section title, truncate if needed
            title = section.title[:200]
            lines.append(f"{idx}/ {title}")
            lines.append("")
            idx += 1
            if idx > max_posts - 1:
                break

        # Closing
        lines.append(f"{idx}/ The future isn't safe or doomed. It's uneven, turbulent, and ongoing.")

        return "\n".join(lines)

    # =========================================================================
    # Essay Generation
    # =========================================================================

    async def generate_outline(
        self,
        title: str,
        thesis: str | None = None,
    ) -> EssayOutline:
        """Generate essay outline from clusters."""
        if not self._clusters:
            await self.cluster_claims()

        # Generate thesis if not provided
        if not thesis and self._seed_essay:
            # Use seed essay structure to inform thesis
            thesis = self._generate_thesis_from_seed()

        outline = await self.pipeline.generate_outline(
            title=title,
            thesis=thesis,
            clusters=self._clusters,
        )

        # Enhance with debate results
        if self.config.include_debate_summaries and self._debate_results:
            self._enhance_outline_with_debates(outline)

        return outline

    def _generate_thesis_from_seed(self) -> str:
        """Generate thesis informed by seed essay."""
        if not self._seed_essay:
            return ""

        # Use seed essay's key claims
        if self._seed_essay.key_claims:
            return self._seed_essay.key_claims[0]

        # Extract from content
        thesis_patterns = [
            r"(?:This essay|I) argue[s]? that\s+(.{100,500})",
            r"The (?:central|main|core) (?:thesis|argument|claim) is[:\s]+(.{100,500})",
        ]

        for pattern in thesis_patterns:
            match = re.search(pattern, self._seed_essay.content, re.I)
            if match:
                return match.group(1).strip()

        return f"This essay explores themes from {self._seed_essay.title}."

    def _enhance_outline_with_debates(self, outline: EssayOutline) -> None:
        """Enhance outline with debate results."""
        # Add counterarguments section if not present
        has_counterargs = any("counter" in s.title.lower() for s in outline.sections)

        if not has_counterargs and self._debate_results:
            # Collect all counterarguments
            all_counterargs = []
            for result in self._debate_results:
                all_counterargs.extend(result.counterarguments)

            if all_counterargs:
                counterarg_section = EssaySection(
                    id="section_counterarguments",
                    title="Addressing Counterarguments",
                    level=1,
                    content="",
                    word_count=self.config.target_word_count // 10,
                    claims_referenced=[r.original_claim[:50] for r in self._debate_results],
                    sources_cited=[],
                )

                # Add subsections for each type of counterargument
                for i, result in enumerate(self._debate_results[:5]):
                    subsection = EssaySection(
                        id=f"section_counterarg_{i}",
                        title=f"On: {result.original_claim[:50]}...",
                        level=2,
                        content="",
                        word_count=1000,
                        claims_referenced=[result.original_claim[:50]],
                        sources_cited=[],
                    )
                    counterarg_section.subsections.append(subsection)

                outline.sections.append(counterarg_section)

    async def generate_essay(
        self,
        outline: EssayOutline,
    ) -> str:
        """
        Generate the final essay text.

        This creates a structured markdown document from the outline,
        weaving in claims, attribution, and debate results.
        """
        sections = []

        # Title
        sections.append(f"# {outline.title}\n")

        # Thesis
        sections.append(f"## Abstract\n\n{outline.thesis}\n")

        # Generate each section
        for section in outline.sections:
            section_text = self._generate_section_text(section)
            sections.append(section_text)

        # Bibliography
        if outline.bibliography:
            sections.append("\n## References\n")
            for i, source in enumerate(outline.bibliography[:50], 1):
                sections.append(f"{i}. {source.title}\n")

        return "\n".join(sections)

    def _generate_section_text(self, section: EssaySection, depth: int = 0) -> str:
        """Generate text for a section."""
        lines = []

        # Heading
        heading_level = "#" * (section.level + 1)
        lines.append(f"\n{heading_level} {section.title}\n")

        # Find relevant claims for this section
        relevant_claims = [
            c for c in self._claims
            if any(ref[:30] in c.claim[:30] for ref in section.claims_referenced)
        ]

        # Find relevant debate results
        relevant_debates = [
            d for d in self._debate_results
            if any(ref[:30] in d.original_claim[:30] for ref in section.claims_referenced)
        ]

        # Generate placeholder content
        if relevant_claims:
            lines.append("\n*Key claims in this section:*\n")
            for claim in relevant_claims[:5]:
                lines.append(f"- {claim.claim[:200]}...")
                if claim.claim in self._steelman_claims:
                    lines.append(f"  - Steelman: {self._steelman_claims[claim.claim][:200]}...")
                if claim.claim in self._attributed_claims:
                    attr = self._attributed_claims[claim.claim]
                    if attr.sources:
                        lines.append(f"  - Supported by: {attr.sources[0].title}")

        if relevant_debates and self.config.include_debate_summaries:
            lines.append("\n*Counterarguments addressed:*\n")
            for debate in relevant_debates[:3]:
                for counter in debate.counterarguments[:2]:
                    lines.append(f"- {counter}")

        # Placeholder for actual content
        lines.append(f"\n[TODO: Generate {section.word_count} words of content]\n")

        # Subsections
        for subsection in section.subsections:
            lines.append(self._generate_section_text(subsection, depth + 1))

        return "\n".join(lines)

    # =========================================================================
    # Main Workflow
    # =========================================================================

    async def run(
        self,
        title: str,
        thesis: str | None = None,
        debate_rounds: int | None = None,
        find_attribution: bool | None = None,
    ) -> WorkflowResult:
        """
        Run the complete essay workflow.

        Args:
            title: Essay title
            thesis: Optional thesis statement
            debate_rounds: Override config debate rounds
            find_attribution: Override config attribution setting

        Returns:
            WorkflowResult with complete essay data
        """
        if debate_rounds is not None:
            self.config.debate_rounds = debate_rounds
        if find_attribution is not None:
            self.config.enable_attribution = find_attribution

        logger.info(f"Starting essay workflow: {title}")

        # Step 1: Extract claims
        logger.info("Step 1/6: Extracting claims...")
        await self.extract_claims()

        # Step 2: Cluster claims
        logger.info("Step 2/6: Clustering claims by topic...")
        await self.cluster_claims()

        # Step 3: Run debates
        logger.info("Step 3/6: Running multi-agent debates...")
        await self.debate_claims()

        # Step 3.5: Steelman claims
        self.steelman_claims()

        # Step 4: Find attribution
        logger.info("Step 4/6: Finding scholarly attribution...")
        await self.find_attribution()

        # Step 5: Generate outline
        logger.info("Step 5/6: Generating essay outline...")
        outline = await self.generate_outline(title, thesis)

        # Step 6: Generate essay
        logger.info("Step 6/6: Generating essay text...")
        essay_text = await self.generate_essay(outline)

        # Compile result
        result = WorkflowResult(
            title=title,
            thesis=thesis or outline.thesis,
            outline=outline,
            claims=self._claims,
            clusters=self._clusters,
            attributed_claims=self._attributed_claims,
            debate_results=self._debate_results,
            steelman_claims=self._steelman_claims,
            seed_essay=self._seed_essay,
            final_essay=essay_text,
            statistics=self._compile_statistics(),
        )

        logger.info(f"Workflow complete! Generated {len(essay_text)} character essay")
        return result

    def _compile_statistics(self) -> dict:
        """Compile workflow statistics."""
        return {
            "exports_loaded": len(self._exports),
            "total_conversations": sum(e.conversation_count for e in self._exports),
            "total_words_analyzed": sum(e.total_words for e in self._exports),
            "claims_extracted": len(self._claims),
            "clusters_created": len(self._clusters),
            "claims_attributed": len(self._attributed_claims),
            "debates_conducted": len(self._debate_results),
            "steelman_claims": len(self._steelman_claims),
            "has_seed_essay": self._seed_essay is not None,
            "counterarguments_addressed": sum(
                len(d.counterarguments) for d in self._debate_results
            ),
            "seed_filtered": self.config.enable_seed_filtering and self._seed_essay is not None,
        }

    # =========================================================================
    # Export
    # =========================================================================

    def export_essay(
        self,
        path: str | Path,
        result: WorkflowResult | None = None,
    ) -> None:
        """Export the final essay to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if result is None:
            raise ValueError("No result to export. Run workflow first.")

        if self.config.output_format == "markdown":
            path.write_text(result.final_essay, encoding="utf-8")

        elif self.config.output_format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

        elif self.config.output_format == "html":
            # Simple markdown to HTML conversion
            html = f"<html><head><title>{result.title}</title></head><body>"
            html += result.final_essay.replace("\n", "<br>")
            html += "</body></html>"
            path.write_text(html, encoding="utf-8")

        logger.info(f"Exported essay to {path}")

    def export_synthesis_package(
        self,
        path: str | Path,
        outline: EssayOutline | None = None,
    ) -> None:
        """Export complete synthesis package for LLM processing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get seed scores preview
        _, seed_scores = self.filter_claims_by_seed(max_claims=50)

        # Prepare seed essay for serialization (convert set to list)
        seed_data = None
        if self._seed_essay:
            seed_data = {
                "title": self._seed_essay.title,
                "content": self._seed_essay.content[:5000],  # Truncate for package
                "sections": self._seed_essay.sections,
                "key_claims": self._seed_essay.key_claims,
                "themes": self._seed_essay.themes,
                "keywords": list(self._seed_essay.keywords)[:100],  # Top keywords
            }

        package = {
            "seed_essay": seed_data,
            "seed_scores": seed_scores,
            "claims": [c.to_dict() for c in self._claims],
            "clusters": [c.to_dict() for c in self._clusters],
            "debate_results": [d.to_dict() for d in self._debate_results],
            "steelman_claims": self._steelman_claims,
            "statistics": self._compile_statistics(),
            "config": {
                "target_word_count": self.config.target_word_count,
                "voice_style": self.config.voice_style,
                "include_counterarguments": self.config.include_counterarguments,
            },
        }

        # Add thread skeleton if outline available
        if outline:
            package["thread_skeleton"] = self.generate_thread_skeleton(outline)
            package["outline"] = outline.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(package, f, indent=2, default=str)

        logger.info(f"Exported synthesis package to {path}")


# Convenience function
def create_workflow(
    config: WorkflowConfig | None = None,
    **kwargs,
) -> EssayWorkflow:
    """Create an essay workflow with optional configuration."""
    return EssayWorkflow(config=config)
