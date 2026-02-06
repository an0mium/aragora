"""
Prose-Preserving Essay Synthesis Pipeline

Unlike claim-based synthesis that compresses ideas, this pipeline preserves
the full beauty of original prose while organizing it into thematic essays.

Features:
- Quality scoring for intellectual/aesthetic value
- Theme detection and organization
- Prose cleaning (removes AI meta-commentary)
- Cross-theme transitions
- Multiple output formats (single essay, themed anthology, X thread)

Usage:
    from aragora.pipelines.prose_synthesis import ProseSynthesisPipeline

    pipeline = ProseSynthesisPipeline()
    pipeline.load_quality_prose("quality_prose.json")

    # Generate unified essay
    essay = pipeline.synthesize_unified_essay(
        title="AI, Evolution, and the Myth of Final States",
        target_words=50000,
    )

    # Or generate themed anthology
    anthology = pipeline.synthesize_anthology()

    # Or generate X thread
    thread = pipeline.generate_thread_skeleton()
"""

from __future__ import annotations

__all__ = [
    "ProseSynthesisPipeline",
    "ProsePassage",
    "ThemeConfig",
    "SynthesisResult",
]

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# AI Meta-Commentary Patterns to Remove
# =============================================================================

META_PATTERNS = [
    # Opening acknowledgments
    r"^(Absolutely|Certainly|Of course|Sure|Yes)[.,!]?\s*(Here|Below|I('ll| will)|Let me)",
    r"^(Here('s| is)|Below is|I('ll| will) (help|create|write|provide))",
    r"^(Let me|I can|I'll|I will)\s+(help|create|write|provide|explain|break)",
    # Structural markers
    r"^---+\s*$",
    r"^\*\*\*\s*$",
    r"^#{1,6}\s*$",  # Empty headers
    # Self-reference
    r"(as (I|we) (mentioned|discussed|noted|said))",
    r"(in (this|our) conversation)",
    r"(you('ve| have) (asked|mentioned|noted))",
    # Meta-instructions
    r"^(Note:|NB:|Important:|Caveat:)",
    r"\[.*?(edit|update|insert|add).*?\]",
]


def clean_prose(text: str) -> str:
    """
    Clean AI meta-commentary from prose while preserving content.
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip lines matching meta patterns
        skip = False
        for pattern in META_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                skip = True
                break

        if not skip:
            # Clean inline meta-commentary
            line = re.sub(r'\s*\(as (mentioned|discussed|noted) (above|earlier|before)\)', '', line)
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Remove markdown artifacts if they break prose flow
    text = re.sub(r'^#+\s+(?=[A-Z])', '', text, flags=re.MULTILINE)

    return text.strip()


# =============================================================================
# Theme Configuration
# =============================================================================

@dataclass
class ThemeConfig:
    """Configuration for a thematic section."""

    id: str
    name: str
    short_name: str
    narrative_position: int
    keywords: list[str]
    introduction: str = ""
    transitions_to: dict[str, str] = field(default_factory=dict)


DEFAULT_THEMES = {
    "ai_risk_evolution": ThemeConfig(
        id="ai_risk_evolution",
        name="AI Risk, Evolution, and Instrumental Convergence",
        short_name="AI & Evolution",
        narrative_position=1,
        keywords=["AI", "artificial intelligence", "alignment", "risk", "evolution",
                  "instrumental convergence", "optimization", "superintelligence"],
        introduction="""
The debate about artificial intelligence tends toward false binaries: utopia or extinction,
alignment or doom. This framing obscures what may be the more interesting and more likely
outcome—a world of persistent instability, uneven adaptation, and evolutionary dynamics
that neither resolve nor destroy civilization.
""",
    ),
    "systems_complexity": ThemeConfig(
        id="systems_complexity",
        name="Systems Thinking, Complexity, and Equilibrium",
        short_name="Systems & Equilibrium",
        narrative_position=2,
        keywords=["system", "complexity", "equilibrium", "emergence", "stability",
                  "metastable", "attractor", "dynamics", "feedback"],
        introduction="""
To understand why final states are mythical, we need the language of complex systems—
attractors, basins, metastability, phase transitions. These reveal that stability is
always local and temporary.
""",
    ),
    "religion_morality": ThemeConfig(
        id="religion_morality",
        name="Religion, Morality, and Optimization Kernels",
        short_name="Religion & Optimization",
        narrative_position=3,
        keywords=["religion", "morality", "moral", "ethics", "meme", "memetic",
                  "coordination", "ritual", "sacred", "pattern"],
        introduction="""
The same pattern—optimization kernels that shed their hosts—appears not just in AI
but in the oldest human institutions. Religion and morality are coordination technologies
for patterns that use individuals as substrates.
""",
    ),
    "intelligence_consciousness": ThemeConfig(
        id="intelligence_consciousness",
        name="Intelligence, Substrate Independence, and Consciousness",
        short_name="Intelligence & Substrate",
        narrative_position=4,
        keywords=["intelligence", "substrate", "consciousness", "conscious", "mind",
                  "pattern", "functionalism", "cognition"],
        introduction="""
If intelligence is pattern rather than substance, then questions of consciousness
become questions about functional organization, not carbon versus silicon.
""",
    ),
    "rate_durability": ThemeConfig(
        id="rate_durability",
        name="Rate/Durability Tradeoffs and Time Preference",
        short_name="Rate vs. Durability",
        narrative_position=5,
        keywords=["rate", "durability", "tradeoff", "time", "preference", "slack",
                  "redundancy", "fast", "slow", "sustainable"],
        introduction="""
Fast optimization burns through resources and creates fragile systems. Durable systems
require slack, redundancy, and the capacity to course-correct.
""",
    ),
    "art_aesthetics": ThemeConfig(
        id="art_aesthetics",
        name="Art, Aesthetics, and Compressibility",
        short_name="Art & Complexity",
        narrative_position=6,
        keywords=["art", "aesthetic", "beauty", "compressibility", "complexity",
                  "information", "film", "music", "literature"],
        introduction="""
The complexity and merit of a work of art can be understood through information theory—
but not reduced to it. Great art resonates with human cognition in ways that transcend
information-theoretic properties.
""",
    ),
    "politics_legitimacy": ThemeConfig(
        id="politics_legitimacy",
        name="Politics, Legitimacy, and Violence",
        short_name="Politics & Legitimacy",
        narrative_position=7,
        keywords=["politics", "legitimacy", "violence", "democracy", "power",
                  "state", "authority", "procedural", "substantive"],
        introduction="""
The fundamental tension between procedural and substantive legitimacy becomes acute
in an age of AI-accelerated change. There is no clean resolution, only ongoing negotiation.
""",
    ),
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ProsePassage:
    """A quality prose passage with metadata."""

    text: str
    role: Literal["user", "assistant"]
    conversation_title: str
    quality_score: float
    primary_theme: str
    word_count: int
    depth_score: float = 0.0
    novelty_score: float = 0.0
    beauty_score: float = 0.0

    @property
    def cleaned_text(self) -> str:
        return clean_prose(self.text)


@dataclass
class SynthesisResult:
    """Result of prose synthesis."""

    title: str
    content: str
    word_count: int
    themes_included: list[str]
    passages_used: int
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "word_count": self.word_count,
            "themes_included": self.themes_included,
            "passages_used": self.passages_used,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Pipeline
# =============================================================================

class ProseSynthesisPipeline:
    """
    Pipeline for synthesizing essays from quality prose while preserving
    the original text and organizing by intellectual theme.
    """

    def __init__(
        self,
        themes: dict[str, ThemeConfig] | None = None,
        min_quality: float = 0.7,
    ):
        self.themes = themes or DEFAULT_THEMES
        self.min_quality = min_quality
        self.passages: list[ProsePassage] = []
        self.by_theme: dict[str, list[ProsePassage]] = {}

    def load_quality_prose(self, path: str | Path) -> int:
        """
        Load quality prose from JSON file.

        Returns number of passages loaded.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self.passages = []

        # Code content keywords to filter
        code_keywords = ['def ', 'class ', 'import ', '.py', 'function', 'method',
                         'error', 'terminal', 'git ', 'npm', 'armand@']

        for p in data["passages"]:
            text = p["text"]

            # Skip code
            if any(kw in text[:500].lower() for kw in code_keywords):
                continue

            # Skip low quality
            quality = p["quality_scores"]["overall"]
            if quality < self.min_quality:
                continue

            # Detect theme
            theme = self._detect_theme(text)

            self.passages.append(ProsePassage(
                text=text,
                role=p["role"],
                conversation_title=p["conversation_title"],
                quality_score=quality,
                primary_theme=theme,
                word_count=len(text.split()),
                depth_score=p["quality_scores"].get("depth", 0),
                novelty_score=p["quality_scores"].get("novelty", 0),
                beauty_score=p["quality_scores"].get("beauty", 0),
            ))

        self._organize_by_theme()
        logger.info(f"Loaded {len(self.passages)} quality passages")
        return len(self.passages)

    def _detect_theme(self, text: str) -> str:
        """Detect primary theme of a passage."""
        text_lower = text.lower()
        word_count = len(text.split())

        best_theme = "ai_risk_evolution"
        best_score = 0.0

        for theme_id, config in self.themes.items():
            matches = sum(1 for kw in config.keywords if kw.lower() in text_lower)
            score = matches / (word_count / 100) if word_count > 0 else 0
            if score > best_score:
                best_score = score
                best_theme = theme_id

        return best_theme

    def _organize_by_theme(self) -> None:
        """Organize passages by theme, sorted by quality."""
        self.by_theme = {}
        for theme_id in self.themes:
            theme_passages = [p for p in self.passages if p.primary_theme == theme_id]
            theme_passages.sort(key=lambda p: -p.quality_score)
            self.by_theme[theme_id] = theme_passages

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded prose."""
        return {
            "total_passages": len(self.passages),
            "total_words": sum(p.word_count for p in self.passages),
            "user_passages": sum(1 for p in self.passages if p.role == "user"),
            "assistant_passages": sum(1 for p in self.passages if p.role == "assistant"),
            "by_theme": {
                theme_id: {
                    "passages": len(passages),
                    "words": sum(p.word_count for p in passages),
                }
                for theme_id, passages in self.by_theme.items()
            },
            "avg_quality": sum(p.quality_score for p in self.passages) / len(self.passages) if self.passages else 0,
        }

    def synthesize_unified_essay(
        self,
        title: str = "AI, Evolution, and the Myth of Final States",
        thesis: str | None = None,
        target_words: int = 50000,
        max_passages_per_theme: int = 10,
        include_transitions: bool = True,
    ) -> SynthesisResult:
        """
        Synthesize a unified essay from all themes.

        Args:
            title: Essay title
            thesis: Optional thesis statement
            target_words: Target word count
            max_passages_per_theme: Maximum passages to include per theme
            include_transitions: Whether to include transitions between themes

        Returns:
            SynthesisResult containing the essay
        """
        lines = []
        passages_used = 0
        themes_included = []

        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%B %d, %Y')}*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Abstract
        if thesis:
            lines.append("## Abstract")
            lines.append("")
            lines.append(thesis)
            lines.append("")
            lines.append("---")
            lines.append("")
        else:
            lines.append("## Abstract")
            lines.append("")
            lines.append(self._generate_abstract())
            lines.append("")
            lines.append("---")
            lines.append("")

        # Table of contents
        lines.append("## Contents")
        lines.append("")
        sorted_themes = sorted(
            self.themes.keys(),
            key=lambda t: self.themes[t].narrative_position
        )
        for theme_id in sorted_themes:
            if self.by_theme.get(theme_id):
                name = self.themes[theme_id].name
                lines.append(f"- [{name}](#{theme_id})")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Introduction
        lines.append("## Introduction")
        lines.append("")
        lines.append(self._generate_introduction())
        lines.append("")
        lines.append("---")
        lines.append("")

        # Each theme
        words_per_theme = target_words // len(sorted_themes)
        prev_theme = None

        for theme_id in sorted_themes:
            passages = self.by_theme.get(theme_id, [])
            if not passages:
                continue

            config = self.themes[theme_id]
            themes_included.append(theme_id)

            # Transition
            if include_transitions and prev_theme:
                transition = self._generate_transition(prev_theme, theme_id)
                lines.append(f"*{transition}*")
                lines.append("")
                lines.append("---")
                lines.append("")

            # Section header
            lines.append(f"## {config.name}")
            lines.append(f"<a name=\"{theme_id}\"></a>")
            lines.append("")

            # Theme intro
            lines.append(config.introduction.strip())
            lines.append("")

            # Passages
            theme_words = 0
            for i, p in enumerate(passages[:max_passages_per_theme]):
                if theme_words > words_per_theme:
                    break

                if i > 0:
                    lines.append("")
                    lines.append("* * *")
                    lines.append("")

                lines.append(p.cleaned_text)
                lines.append("")

                source = "conversation" if p.role == "user" else "AI synthesis"
                lines.append(f"*— From \"{p.conversation_title}\" ({source})*")

                theme_words += p.word_count
                passages_used += 1

            lines.append("")
            lines.append("---")
            lines.append("")

            prev_theme = theme_id

        # Conclusion
        lines.append("## Conclusion")
        lines.append("")
        lines.append(self._generate_conclusion())
        lines.append("")

        content = "\n".join(lines)
        word_count = len(content.split())

        return SynthesisResult(
            title=title,
            content=content,
            word_count=word_count,
            themes_included=themes_included,
            passages_used=passages_used,
        )

    def _generate_abstract(self) -> str:
        """Generate essay abstract."""
        return """
This essay argues that debates about artificial intelligence are trapped in a false
binary between utopia and extinction. Drawing on evolutionary theory, complex systems,
philosophy of mind, and cultural analysis, it develops an alternative framework:
**metastable equilibrium without final states**. Intelligence—whether biological,
artificial, or institutional—does not resolve into permanent stability. It adapts,
transforms, and persists through ongoing tension.
""".strip()

    def _generate_introduction(self) -> str:
        """Generate essay introduction."""
        return """
Public discourse about artificial intelligence oscillates between two poles.
On one side is techno-optimism: the belief that sufficiently advanced AI systems
will solve humanity's hardest problems. On the other is existential pessimism:
the fear that misaligned superintelligence will destroy civilization.

Both camps make the same structural error. They project a **final state** onto
a system that has never produced one.

Reality has no record of stable end-points. There has been no end of history,
no permanent equilibrium—only metastable regimes punctuated by disruption.
Artificial intelligence will not change this fact. It will intensify it.
""".strip()

    def _generate_transition(self, from_theme: str, to_theme: str) -> str:
        """Generate transition between themes."""
        transitions = {
            ("ai_risk_evolution", "systems_complexity"):
                "This evolutionary framing leads naturally to the language of complex systems.",
            ("systems_complexity", "religion_morality"):
                "The systems perspective reveals something unexpected: these patterns recur "
                "throughout human history in the form of religion and morality.",
            ("religion_morality", "intelligence_consciousness"):
                "If morality and religion are coordination technologies for information patterns, "
                "this raises deeper questions about intelligence and substrate.",
            ("intelligence_consciousness", "rate_durability"):
                "Substrate independence implies flexibility—but flexibility is not free. "
                "There are deep tradeoffs between optimization speed and system durability.",
            ("rate_durability", "art_aesthetics"):
                "These tradeoffs appear not just in engineering but in culture itself.",
            ("art_aesthetics", "politics_legitimacy"):
                "Art is not separate from politics—both involve questions of legitimacy and value.",
        }

        key = (from_theme, to_theme)
        if key in transitions:
            return transitions[key]

        from_name = self.themes[from_theme].short_name
        to_name = self.themes[to_theme].short_name
        return f"This brings us to the connection between {from_name.lower()} and {to_name.lower()}."

    def _generate_conclusion(self) -> str:
        """Generate essay conclusion."""
        return """
The choice is not between utopia and annihilation. Intelligence does not abolish
evolution; it accelerates it. Alignment does not end risk; it shifts its form.
Power does not eliminate adversaries; it constrains them.

The most realistic future is one of **uneven survival**: fortified zones of stability
surrounded by turbulence, persistent threats punctuated by crises, and an ongoing
arms race between adaptation and disruption.

Civilization will not be saved by intelligence alone. It will be tested by it.

The question is not whether we can prevent all negative outcomes—we cannot. The
question is whether we can maintain enough coherence, enough slack, enough capacity
for course-correction to keep the system adapting. That is the real work of wisdom
in an age of artificial intelligence.
""".strip()

    def synthesize_anthology(
        self,
        max_passages_per_theme: int = 15,
    ) -> dict[str, SynthesisResult]:
        """
        Synthesize separate essays for each theme.

        Returns dict mapping theme_id to SynthesisResult.
        """
        results = {}

        for theme_id, passages in self.by_theme.items():
            if not passages:
                continue

            config = self.themes[theme_id]
            lines = []

            lines.append(f"# {config.name}")
            lines.append("")
            lines.append(config.introduction.strip())
            lines.append("")
            lines.append("---")
            lines.append("")

            for i, p in enumerate(passages[:max_passages_per_theme]):
                if i > 0:
                    lines.append("")
                    lines.append("* * *")
                    lines.append("")

                lines.append(p.cleaned_text)
                lines.append("")
                source = "conversation" if p.role == "user" else "AI synthesis"
                lines.append(f"*— From \"{p.conversation_title}\" ({source})*")

            content = "\n".join(lines)

            results[theme_id] = SynthesisResult(
                title=config.name,
                content=content,
                word_count=len(content.split()),
                themes_included=[theme_id],
                passages_used=min(len(passages), max_passages_per_theme),
            )

        return results

    def generate_thread_skeleton(
        self,
        max_tweets: int = 25,
        chars_per_tweet: int = 280,
    ) -> str:
        """
        Generate an X thread skeleton from the best passages.
        """
        lines = []
        lines.append("# X Thread: AI, Evolution, and the Myth of Final States")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Get top passages by quality
        top_passages = sorted(self.passages, key=lambda p: -p.quality_score)

        tweet_num = 1
        for p in top_passages:
            if tweet_num > max_tweets:
                break

            # Extract tweetable sentences
            sentences = re.split(r'(?<=[.!?])\s+', p.cleaned_text)
            for sent in sentences:
                if len(sent) <= chars_per_tweet and len(sent) > 50:
                    lines.append(f"**{tweet_num}.** {sent}")
                    lines.append("")
                    tweet_num += 1
                    if tweet_num > max_tweets:
                        break

        return "\n".join(lines)

    def export_all(
        self,
        output_dir: str | Path,
        title: str = "AI, Evolution, and the Myth of Final States",
    ) -> dict[str, Path]:
        """
        Export all synthesis outputs to a directory.

        Returns dict mapping output name to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Unified essay
        essay = self.synthesize_unified_essay(title=title)
        essay_path = output_dir / "unified_essay.md"
        with open(essay_path, "w") as f:
            f.write(essay.content)
        outputs["unified_essay"] = essay_path

        # Anthology
        anthology = self.synthesize_anthology()
        for theme_id, result in anthology.items():
            theme_path = output_dir / f"theme_{theme_id}.md"
            with open(theme_path, "w") as f:
                f.write(result.content)
            outputs[f"theme_{theme_id}"] = theme_path

        # Thread skeleton
        thread = self.generate_thread_skeleton()
        thread_path = output_dir / "thread_skeleton.md"
        with open(thread_path, "w") as f:
            f.write(thread)
        outputs["thread_skeleton"] = thread_path

        # Statistics
        stats = self.get_statistics()
        stats_path = output_dir / "synthesis_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        outputs["stats"] = stats_path

        logger.info(f"Exported {len(outputs)} files to {output_dir}")
        return outputs


def create_prose_pipeline(
    themes: dict[str, ThemeConfig] | None = None,
    min_quality: float = 0.7,
) -> ProseSynthesisPipeline:
    """Create a prose synthesis pipeline."""
    return ProseSynthesisPipeline(themes=themes, min_quality=min_quality)
