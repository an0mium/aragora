#!/usr/bin/env python3
"""
Extract Quality Prose from Conversations

Instead of compressing conversations into claims, this extracts and scores
substantial prose passages for intellectual/aesthetic quality.

Scores passages on:
- Intellectual depth (conceptual density, abstraction level)
- Novelty/surprise (unexpected connections, original framings)
- Beauty (rhetorical craft, metaphor, rhythm)
- Edification (insight that changes how you see things)

Filters OUT:
- Tutoring/homework help (chemistry, biology, physics problems)
- Short transactional exchanges
- Technical debugging sessions

Preserves:
- Extended philosophical discussions
- Original framings and metaphors
- Surprising intellectual connections
- Beautiful prose from both user AND AI

Usage:
    python scripts/extract_quality_prose.py \
        --input exports/consolidated_conversations.json \
        --output output/quality_prose.json \
        --min-quality 0.6 \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Tutoring/Homework Detection
# =============================================================================

TUTORING_PATTERNS = [
    # Chemistry
    r"\b(pH|pKa|molarity|titration|oxidation|reduction|electrochemical|galvanic)\b",
    r"\b(acid|base|buffer|equilibrium constant|E-cell|half-reaction)\b",
    r"\b(electrons?|protons?|neutrons?|isotopes?|ionic|covalent)\b",
    r"\b(moles?|grams?|liters?|concentration|dilution)\b",
    # Biology
    r"\b(mitosis|meiosis|chromosomes?|alleles?|genotype|phenotype)\b",
    r"\b(DNA|RNA|transcription|translation|ribosome|amino acids?)\b",
    r"\b(cells?|membrane|cytoplasm|nucleus|organelles?)\b",
    r"\b(dominant|recessive|heterozygous|homozygous|Punnett)\b",
    r"\b(surfactant|alveoli|lungs?|respiration)\b",
    # Physics
    r"\b(velocity|acceleration|momentum|kinetic energy|potential energy)\b",
    r"\b(Newton's|force|mass|friction|gravity|projectile)\b",
    r"\b(circuits?|voltage|current|resistance|Ohm's)\b",
    r"\b(wavelength|frequency|amplitude|interference)\b",
    # Math homework patterns
    r"\b(solve for|calculate|find the value|what is the answer)\b",
    r"\b(derivative|integral|limit|polynomial|quadratic)\b",
    # Homework indicators
    r"\b(homework|assignment|lab report|experiment|procedure)\b",
    r"\b(step by step|show your work|explain how to)\b",
    r"what (is|are) the (units?|formula|equation)",
]

TUTORING_TITLE_PATTERNS = [
    r"(chemistry|biology|physics|biochem|organic chem)",
    r"(homework|lab|assignment|quiz|exam|test prep)",
    r"(amino acid|protein|enzyme|metabolism)",
    r"(electrolytic|galvanic|electrochemistry)",
    r"(genetics|heredity|Mendelian)",
]


def is_tutoring_content(text: str, title: str = "") -> tuple[bool, float]:
    """
    Detect if content is tutoring/homework help.

    Returns (is_tutoring, confidence)
    """
    text_lower = text.lower()
    title_lower = title.lower()

    # Check title first
    title_matches = sum(1 for p in TUTORING_TITLE_PATTERNS if re.search(p, title_lower, re.I))
    if title_matches >= 1:
        return True, 0.9

    # Count pattern matches in text
    matches = 0
    for pattern in TUTORING_PATTERNS:
        if re.search(pattern, text_lower, re.I):
            matches += 1

    # Threshold based on density
    word_count = len(text.split())
    if word_count < 100:
        threshold = 2
    elif word_count < 500:
        threshold = 4
    else:
        threshold = 6

    if matches >= threshold:
        confidence = min(0.5 + (matches - threshold) * 0.1, 0.95)
        return True, confidence

    return False, 0.0


# =============================================================================
# Quality Scoring
# =============================================================================

# Markers of intellectual depth
DEPTH_MARKERS = [
    # Conceptual vocabulary
    (r"\b(epistem|ontolog|phenomen|hermeneutic|dialectic)\w*\b", 2.0),
    (r"\b(emergence|convergence|equilibrium|attractor|dynamics)\b", 1.5),
    (r"\b(substrate|instantiat|implement|manifest|embod)\w*\b", 1.3),
    (r"\b(optimization|instrumental|terminal|convergent)\b", 1.5),
    (r"\b(selection pressure|evolutionary|adaptive|fitness)\b", 1.4),
    (r"\b(system|structure|pattern|process|mechanism)\b", 1.0),
    # Abstract reasoning
    (r"\b(therefore|thus|hence|consequently|implies)\b", 1.2),
    (r"\b(paradox|contradiction|tension|tradeoff|dilemma)\b", 1.4),
    (r"\b(necessary|sufficient|contingent|possible|impossible)\b", 1.3),
    # Meta-level thinking
    (r"\b(meta-|second-order|recursive|self-referential)\b", 1.5),
    (r"\b(framework|paradigm|worldview|lens|perspective)\b", 1.2),
]

# Markers of novelty/surprise
NOVELTY_MARKERS = [
    # Unexpected connections
    (r"like .{10,50} (but|except|unlike)", 1.5),
    (r"(reminds me of|echoes|parallels|mirrors)", 1.3),
    (r"(surprisingly|counterintuitively|paradoxically)", 1.8),
    # Original framings
    (r"(what if|imagine|suppose|consider)", 1.2),
    (r"(reframe|reconceptualize|rethink|reimagine)", 1.5),
    (r"here's (the|a) (key|crucial|interesting|subtle)", 1.4),
    # Insight markers
    (r"(the (real|deeper|underlying|hidden) (issue|point|question))", 1.6),
    (r"(this (suggests|implies|reveals|exposes))", 1.3),
    (r"(the (interesting|surprising|subtle) thing is)", 1.5),
]

# Markers of rhetorical beauty
BEAUTY_MARKERS = [
    # Metaphor and imagery
    (r"(like a|as if|imagine a|picture a)", 1.3),
    (r"(dance|weave|thread|tapestry|mosaic)", 1.4),
    (r"(crystallize|illuminate|resonate|echo)", 1.3),
    # Rhythm and structure
    (r"(not .{5,30}, but .{5,30})", 1.5),  # Antithesis
    (r"(\w+), (\w+), (and|or) (\w+)", 1.2),  # Tricolon
    (r"(the more .{5,30}, the more .{5,30})", 1.4),  # Parallelism
    # Memorable phrasing
    (r"(in other words|put differently|to put it another way)", 1.1),
    (r"(at (its|the) (core|heart|root))", 1.2),
]

# Markers of edifying value
EDIFICATION_MARKERS = [
    # Perspective shifts
    (r"(changes how|shifts (the|your)|reframes)", 1.6),
    (r"(once you see|when you realize|if you notice)", 1.4),
    (r"(the (key|crucial|essential) insight)", 1.5),
    # Practical wisdom
    (r"(this (means|implies) that)", 1.2),
    (r"(the (lesson|takeaway|implication))", 1.3),
    (r"(in practice|practically speaking|concretely)", 1.2),
    # Synthesis
    (r"(brings together|synthesizes|unifies|integrates)", 1.5),
    (r"(the (common|shared|underlying) (thread|pattern))", 1.4),
]


def score_passage_quality(text: str) -> dict[str, float]:
    """
    Score a passage on multiple quality dimensions.

    Returns dict with scores for depth, novelty, beauty, edification.
    """
    word_count = len(text.split())
    if word_count < 50:
        return {"depth": 0, "novelty": 0, "beauty": 0, "edification": 0, "overall": 0}

    def count_markers(markers: list[tuple[str, float]]) -> float:
        total = 0.0
        for pattern, weight in markers:
            matches = len(re.findall(pattern, text, re.I))
            total += matches * weight
        # Normalize by word count (per 100 words)
        return total / (word_count / 100)

    depth = count_markers(DEPTH_MARKERS)
    novelty = count_markers(NOVELTY_MARKERS)
    beauty = count_markers(BEAUTY_MARKERS)
    edification = count_markers(EDIFICATION_MARKERS)

    # Length bonus for substantial passages
    length_bonus = min(word_count / 500, 1.5)

    # Calculate overall score (weighted)
    overall = (depth * 0.3 + novelty * 0.25 + beauty * 0.2 + edification * 0.25) * length_bonus

    return {
        "depth": round(depth, 3),
        "novelty": round(novelty, 3),
        "beauty": round(beauty, 3),
        "edification": round(edification, 3),
        "overall": round(overall, 3),
        "word_count": word_count,
    }


# =============================================================================
# Prose Extraction
# =============================================================================


@dataclass
class ProsePassage:
    """A substantial prose passage with quality scores."""

    text: str
    role: Literal["user", "assistant"]
    conversation_id: str
    conversation_title: str
    timestamp: str | None
    quality_scores: dict[str, float]
    context_before: str = ""
    context_after: str = ""

    @property
    def overall_quality(self) -> float:
        return self.quality_scores.get("overall", 0)

    @property
    def word_count(self) -> int:
        return self.quality_scores.get("word_count", len(self.text.split()))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "role": self.role,
            "conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "timestamp": self.timestamp,
            "quality_scores": self.quality_scores,
            "context_before": self.context_before[:500] if self.context_before else "",
            "context_after": self.context_after[:500] if self.context_after else "",
            "word_count": self.word_count,
        }


def extract_messages_from_conversation(conv: dict) -> list[dict]:
    """Extract messages from ChatGPT or Claude format."""
    messages = []

    # ChatGPT format (mapping tree)
    if "mapping" in conv:
        nodes = []
        for node_id, node in conv["mapping"].items():
            msg = node.get("message")
            if msg and msg.get("content", {}).get("parts"):
                role = msg.get("author", {}).get("role", "unknown")
                if role in ("user", "assistant"):
                    parts = msg.get("content", {}).get("parts", [])
                    content = "\n".join(str(p) for p in parts if p)
                    timestamp = msg.get("create_time")
                    nodes.append(
                        {
                            "role": role,
                            "content": content,
                            "timestamp": datetime.fromtimestamp(timestamp).isoformat()
                            if timestamp
                            else None,
                            "sort_key": timestamp or 0,
                        }
                    )
        nodes.sort(key=lambda x: x["sort_key"])
        messages = nodes

    # Claude format (chat_messages)
    elif "chat_messages" in conv:
        for msg in conv["chat_messages"]:
            role = msg.get("sender", msg.get("role", "unknown"))
            if role in ("human", "user"):
                role = "user"
            elif role in ("assistant", "claude"):
                role = "assistant"
            else:
                continue

            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)

            timestamp = msg.get("created_at")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                }
            )

    return messages


def extract_quality_prose(
    conversations: list[dict],
    min_quality: float = 0.5,
    min_words: int = 100,
    include_user: bool = True,
    include_assistant: bool = True,
    verbose: bool = False,
) -> list[ProsePassage]:
    """
    Extract high-quality prose passages from conversations.

    Filters out tutoring content and scores for intellectual quality.
    """
    passages = []
    stats = {
        "total_conversations": len(conversations),
        "filtered_tutoring": 0,
        "total_messages_scanned": 0,
        "passages_extracted": 0,
    }

    for conv in conversations:
        conv_id = conv.get("id", conv.get("uuid", conv.get("conversation_id", "unknown")))
        title = conv.get("title", conv.get("name", "Untitled"))

        # Check if this is a tutoring conversation
        messages = extract_messages_from_conversation(conv)
        all_text = " ".join(m["content"] for m in messages)

        is_tutoring, confidence = is_tutoring_content(all_text, title)
        if is_tutoring and confidence > 0.7:
            stats["filtered_tutoring"] += 1
            if verbose:
                print(f"  Filtered (tutoring): {title[:50]}...")
            continue

        # Extract quality passages from each message
        for i, msg in enumerate(messages):
            stats["total_messages_scanned"] += 1

            role = msg["role"]
            content = msg["content"]

            # Skip based on role filter
            if role == "user" and not include_user:
                continue
            if role == "assistant" and not include_assistant:
                continue

            # Skip short messages
            if len(content.split()) < min_words:
                continue

            # Score quality
            scores = score_passage_quality(content)

            if scores["overall"] >= min_quality:
                # Get context
                context_before = messages[i - 1]["content"] if i > 0 else ""
                context_after = messages[i + 1]["content"] if i < len(messages) - 1 else ""

                passage = ProsePassage(
                    text=content,
                    role=role,
                    conversation_id=conv_id,
                    conversation_title=title,
                    timestamp=msg.get("timestamp"),
                    quality_scores=scores,
                    context_before=context_before,
                    context_after=context_after,
                )
                passages.append(passage)
                stats["passages_extracted"] += 1

    # Sort by quality
    passages.sort(key=lambda p: -p.overall_quality)

    if verbose:
        print(f"\nExtraction Statistics:")
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Filtered as tutoring: {stats['filtered_tutoring']}")
        print(f"  Messages scanned: {stats['total_messages_scanned']}")
        print(f"  Quality passages: {stats['passages_extracted']}")

    return passages


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality prose from conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input conversations JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output quality prose JSON file",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score (default: 0.5)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=100,
        help="Minimum words per passage (default: 100)",
    )
    parser.add_argument(
        "--user-only",
        action="store_true",
        help="Only extract user messages",
    )
    parser.add_argument(
        "--assistant-only",
        action="store_true",
        help="Only extract assistant messages",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Only output top N passages by quality",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load conversations
    print(f"Loading: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    else:
        conversations = [data]

    print(f"Loaded {len(conversations)} conversations")

    # Extract quality prose
    include_user = not args.assistant_only
    include_assistant = not args.user_only

    passages = extract_quality_prose(
        conversations,
        min_quality=args.min_quality,
        min_words=args.min_words,
        include_user=include_user,
        include_assistant=include_assistant,
        verbose=args.verbose,
    )

    # Apply top-n filter
    if args.top_n:
        passages = passages[: args.top_n]

    # Calculate statistics
    total_words = sum(p.word_count for p in passages)
    user_passages = [p for p in passages if p.role == "user"]
    assistant_passages = [p for p in passages if p.role == "assistant"]

    # Output
    output_data = {
        "metadata": {
            "source": str(args.input),
            "extracted_at": datetime.now().isoformat(),
            "min_quality": args.min_quality,
            "min_words": args.min_words,
            "total_passages": len(passages),
            "total_words": total_words,
            "user_passages": len(user_passages),
            "assistant_passages": len(assistant_passages),
            "avg_quality": sum(p.overall_quality for p in passages) / len(passages)
            if passages
            else 0,
        },
        "passages": [p.to_dict() for p in passages],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("QUALITY PROSE EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nTotal passages: {len(passages)}")
    print(f"  User passages: {len(user_passages)}")
    print(f"  Assistant passages: {len(assistant_passages)}")
    print(f"Total words: {total_words:,}")
    print(f"Avg quality score: {output_data['metadata']['avg_quality']:.3f}")

    print(f"\nTop 10 passages by quality:")
    for i, p in enumerate(passages[:10], 1):
        preview = p.text[:80].replace("\n", " ") + "..."
        print(f"  {i}. [{p.role}] (q={p.overall_quality:.2f}) {preview}")

    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
