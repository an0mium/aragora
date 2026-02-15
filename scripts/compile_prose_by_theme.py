#!/usr/bin/env python3
"""
Compile Quality Prose by Intellectual Theme

Takes the extracted quality prose and organizes it into thematic sections,
preserving the full prose rather than compressing into claims.

Themes identified from the seed conversations:
1. AI Risk / Instrumental Convergence / Evolution
2. Systems Thinking / Complexity / Equilibrium
3. Religion / Morality / Optimization Kernels
4. Art / Aesthetics / Compressibility
5. Politics / Legitimacy / Violence
6. Intelligence / Substrate Independence / Consciousness

Output: A compilation document suitable for essay synthesis.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Theme Detection
# =============================================================================

THEMES = {
    "ai_risk_evolution": {
        "name": "AI Risk, Evolution, and Instrumental Convergence",
        "keywords": [
            r"\b(AI|artificial intelligence|machine learning|superintelligence)\b",
            r"\b(alignment|misalignment|safety|risk|doom)\b",
            r"\b(instrumental convergence|terminal goals|optimization)\b",
            r"\b(evolution|evolutionary|selection pressure|adaptive)\b",
            r"\b(Bostrom|Yudkowsky|MIRI|Anthropic|OpenAI)\b",
        ],
        "weight": 1.5,
    },
    "systems_complexity": {
        "name": "Systems Thinking, Complexity, and Equilibrium",
        "keywords": [
            r"\b(system|systems thinking|complex systems|complexity)\b",
            r"\b(equilibrium|stability|instability|metastable)\b",
            r"\b(emergence|emergent|self-organizing)\b",
            r"\b(feedback|nonlinear|attractor|basin)\b",
            r"\b(resilience|robustness|fragility|antifragile)\b",
        ],
        "weight": 1.3,
    },
    "religion_morality": {
        "name": "Religion, Morality, and Optimization Kernels",
        "keywords": [
            r"\b(religion|religious|spiritual|sacred)\b",
            r"\b(morality|moral|ethics|ethical|values)\b",
            r"\b(meme|memetic|cultural evolution)\b",
            r"\b(ritual|myth|narrative|meaning)\b",
            r"\b(Crustafarian|coordination|cooperation)\b",
        ],
        "weight": 1.4,
    },
    "art_aesthetics": {
        "name": "Art, Aesthetics, and Compressibility",
        "keywords": [
            r"\b(art|artistic|aesthetic|beauty|beautiful)\b",
            r"\b(compressibility|complexity|information|entropy)\b",
            r"\b(film|cinema|movie|play|theater|theatre)\b",
            r"\b(music|symphony|opera|classical)\b",
            r"\b(Farhadi|Beckett|Ionesco|Cronenberg)\b",
        ],
        "weight": 1.2,
    },
    "politics_legitimacy": {
        "name": "Politics, Legitimacy, and Violence",
        "keywords": [
            r"\b(politics|political|polarization|partisan)\b",
            r"\b(legitimacy|authority|power|violence)\b",
            r"\b(democracy|democratic|procedural|substantive)\b",
            r"\b(state|government|institution)\b",
            r"\b(Rittenhouse|ICE|protest|enforcement)\b",
        ],
        "weight": 1.1,
    },
    "intelligence_consciousness": {
        "name": "Intelligence, Substrate Independence, and Consciousness",
        "keywords": [
            r"\b(intelligence|intelligent|cognition|cognitive)\b",
            r"\b(substrate|substrate-independent|functionalism)\b",
            r"\b(consciousness|conscious|sentience|awareness)\b",
            r"\b(mind|mental|phenomenal|qualia)\b",
            r"\b(pattern|information|computation)\b",
        ],
        "weight": 1.2,
    },
    "rate_durability": {
        "name": "Rate/Durability Tradeoffs and Time Preference",
        "keywords": [
            r"\b(rate|speed|fast|slow|growth)\b",
            r"\b(durability|durable|sustainable|long-term)\b",
            r"\b(tradeoff|trade-off|balance)\b",
            r"\b(time preference|discount|patience)\b",
            r"\b(slack|redundancy|resilience)\b",
        ],
        "weight": 1.3,
    },
}


def detect_themes(text: str) -> list[tuple[str, float]]:
    """
    Detect themes in a passage and return (theme_id, score) pairs.
    """
    text_lower = text.lower()
    word_count = len(text.split())
    if word_count < 50:
        return []

    scores = []
    for theme_id, theme in THEMES.items():
        match_count = 0
        for pattern in theme["keywords"]:
            matches = len(re.findall(pattern, text, re.I))
            match_count += matches

        if match_count > 0:
            # Normalize by word count and apply weight
            normalized = (match_count / (word_count / 100)) * theme["weight"]
            scores.append((theme_id, normalized))

    # Sort by score descending
    scores.sort(key=lambda x: -x[1])
    return scores


# =============================================================================
# Compilation
# =============================================================================


@dataclass
class ThemedPassage:
    """A passage with its theme classification."""

    text: str
    role: str
    conversation_title: str
    quality_score: float
    primary_theme: str
    theme_score: float
    all_themes: list[tuple[str, float]]


def compile_by_theme(
    quality_prose_path: Path,
    min_quality: float = 0.8,
    min_theme_score: float = 0.3,
) -> dict[str, list[ThemedPassage]]:
    """
    Compile quality prose organized by theme.
    """
    with open(quality_prose_path) as f:
        data = json.load(f)

    passages = data["passages"]

    # Filter out code content
    code_keywords = [
        "def ",
        "class ",
        "import ",
        ".py",
        "function",
        "method",
        "error",
        "terminal",
        "git ",
        "npm",
        "armand@",
    ]

    themed_passages: dict[str, list[ThemedPassage]] = defaultdict(list)

    for p in passages:
        # Skip low quality
        quality = p["quality_scores"]["overall"]
        if quality < min_quality:
            continue

        # Skip code
        text = p["text"]
        if any(kw in text[:500].lower() for kw in code_keywords):
            continue

        # Detect themes
        themes = detect_themes(text)
        if not themes or themes[0][1] < min_theme_score:
            continue

        primary_theme, theme_score = themes[0]

        themed_passage = ThemedPassage(
            text=text,
            role=p["role"],
            conversation_title=p["conversation_title"],
            quality_score=quality,
            primary_theme=primary_theme,
            theme_score=theme_score,
            all_themes=themes,
        )

        themed_passages[primary_theme].append(themed_passage)

    # Sort each theme's passages by quality
    for theme_id in themed_passages:
        themed_passages[theme_id].sort(key=lambda p: -p.quality_score)

    return dict(themed_passages)


def generate_compilation_document(
    themed_passages: dict[str, list[ThemedPassage]],
    output_path: Path,
    max_passages_per_theme: int = 10,
):
    """
    Generate a markdown compilation document organized by theme.
    """
    lines = []
    lines.append("# Intellectual Prose Compilation")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append(
        "This compilation preserves the full prose from conversations, organized by intellectual theme."
    )
    lines.append("")

    # Table of contents
    lines.append("## Contents")
    lines.append("")
    for theme_id in sorted(themed_passages.keys(), key=lambda t: -len(themed_passages[t])):
        theme_name = THEMES[theme_id]["name"]
        count = len(themed_passages[theme_id])
        lines.append(f"- [{theme_name}](#{theme_id}) ({count} passages)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Statistics
    total_passages = sum(len(p) for p in themed_passages.values())
    total_words = sum(
        len(p.text.split()) for passages in themed_passages.values() for p in passages
    )
    user_count = sum(
        1 for passages in themed_passages.values() for p in passages if p.role == "user"
    )
    ai_count = total_passages - user_count

    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- **Total passages:** {total_passages}")
    lines.append(f"- **Total words:** {total_words:,}")
    lines.append(f"- **User passages:** {user_count}")
    lines.append(f"- **AI passages:** {ai_count}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Each theme section
    for theme_id in sorted(themed_passages.keys(), key=lambda t: -len(themed_passages[t])):
        theme_name = THEMES[theme_id]["name"]
        passages = themed_passages[theme_id][:max_passages_per_theme]

        lines.append(f"## {theme_name}")
        lines.append(f'<a name="{theme_id}"></a>')
        lines.append("")
        lines.append(f"*{len(passages)} passages*")
        lines.append("")

        for i, p in enumerate(passages, 1):
            role_label = "**You:**" if p.role == "user" else "**AI:**"
            lines.append(f"### {i}. {p.conversation_title}")
            lines.append("")
            lines.append(f"*Quality: {p.quality_score:.2f} | {role_label}*")
            lines.append("")
            lines.append(p.text)
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return total_passages, total_words


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compile prose by intellectual theme")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Quality prose JSON")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output markdown")
    parser.add_argument("--min-quality", type=float, default=0.8, help="Min quality score")
    parser.add_argument("--max-per-theme", type=int, default=10, help="Max passages per theme")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    themed = compile_by_theme(args.input, min_quality=args.min_quality)

    print("\nThemes found:")
    for theme_id, passages in sorted(themed.items(), key=lambda x: -len(x[1])):
        theme_name = THEMES[theme_id]["name"]
        print(f"  {theme_name}: {len(passages)} passages")

    total, words = generate_compilation_document(
        themed, args.output, max_passages_per_theme=args.max_per_theme
    )

    print(f"\nCompilation written to: {args.output}")
    print(f"Total: {total} passages, {words:,} words")


if __name__ == "__main__":
    main()
