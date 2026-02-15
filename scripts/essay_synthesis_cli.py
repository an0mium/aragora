#!/usr/bin/env python3
"""
CLI for synthesizing long-form essays from ChatGPT/Claude exports.

Note: This is a lightweight pipeline CLI. For the full workflow with
multi-agent debate, seed scoring, and richer exports, prefer:
    scripts/synthesize_essay.py

This script wires together:
1. Conversation ingestion
2. Claim extraction (optional counterarguments)
3. Seed-based filtering and clustering
4. Outline generation
5. Export package for human/LLM synthesis

Example:
    python scripts/essay_synthesis_cli.py \
        --exports ~/Downloads/chat_exports \
        --seed-file ./notes/seed_essay.md \
        --title "AI, Evolution, and the Myth of Final States" \
        --output ./output/essay_synthesis
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

from aragora.pipelines.essay_synthesis import EssaySynthesisPipeline, SynthesisConfig


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _extract_keywords(text: str) -> set[str]:
    # Mirror pipeline keyword extraction (keep local for CLI independence)
    import re

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "think",
        "believe",
        "about",
        "more",
        "some",
        "any",
        "just",
        "only",
        "also",
        "very",
        "really",
    }
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    return {w for w in words if w not in stopwords}


def _score_claim(claim_text: str, seed_keywords: set[str]) -> float:
    if not seed_keywords:
        return 0.0
    claim_keywords = _extract_keywords(claim_text)
    if not claim_keywords:
        return 0.0
    return len(claim_keywords & seed_keywords) / max(len(seed_keywords), 1)


def _select_claims_by_seed(
    claims,
    seed_text: str,
    min_score: float = 0.05,
    max_claims: int | None = None,
):
    seed_keywords = _extract_keywords(seed_text)
    if not seed_keywords:
        return claims, []

    scored = [(c, _score_claim(c.claim, seed_keywords)) for c in claims]
    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = [c for c, score in scored if score >= min_score]
    if not filtered:
        filtered = [c for c, _ in scored[: max_claims or len(scored)]]

    if max_claims is not None:
        filtered = filtered[:max_claims]

    return filtered, [{"claim": c.claim, "score": score} for c, score in scored[:50]]


def _render_outline_markdown(outline, claims_by_prefix: dict[str, str]) -> str:
    lines = [
        f"# {outline.title}",
        "",
        f"**Thesis:** {outline.thesis}",
        "",
        f"**Target word count:** {outline.target_word_count:,}",
        "",
        "## Sections",
        "",
    ]

    for section in outline.sections:
        lines.append(f"### {section.title}")
        if section.claims_referenced:
            lines.append("")
            lines.append("Key claims:")
            for prefix in section.claims_referenced[:6]:
                claim = claims_by_prefix.get(prefix, prefix)
                lines.append(f"- {claim}")
        if section.subsections:
            lines.append("")
            lines.append("Subsections:")
            for sub in section.subsections:
                lines.append(f"- {sub.title}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_thread_skeleton(outline) -> str:
    # Simple X-thread skeleton (1 tweet per top section)
    lines = []
    lines.append(f"1/ {outline.title}")
    lines.append(f"2/ Thesis: {outline.thesis}")

    idx = 3
    for section in outline.sections[:10]:
        lines.append(f"{idx}/ {section.title}")
        idx += 1

    lines.append(f"{idx}/ The future isn’t safe or doomed. It’s uneven, turbulent, and ongoing.")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize essays from ChatGPT/Claude exports.")
    parser.add_argument("--exports", required=True, help="Path to export JSON file or directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: output/essay_synthesis/<timestamp>)",
    )
    parser.add_argument("--seed-text", default=None, help="Seed idea text (string)")
    parser.add_argument("--seed-file", default=None, help="Path to seed idea file (markdown/text)")
    parser.add_argument("--title", default="Synthesized Essay", help="Essay title")
    parser.add_argument("--thesis", default=None, help="Optional thesis statement override")
    parser.add_argument("--target-words", type=int, default=50000, help="Target essay word count")
    parser.add_argument("--min-claim-length", type=int, default=50, help="Minimum claim length")
    parser.add_argument(
        "--max-claims", type=int, default=300, help="Max claims to keep after filtering"
    )
    parser.add_argument(
        "--seed-min-score", type=float, default=0.05, help="Minimum seed relevance score"
    )
    parser.add_argument(
        "--include-counterarguments", action="store_true", help="Include assistant counterarguments"
    )
    parser.add_argument(
        "--with-attribution", action="store_true", help="Attempt scholarly attribution"
    )

    args = parser.parse_args()

    export_path = Path(args.exports).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else None
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / "essay_synthesis" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_text = args.seed_text or ""
    if args.seed_file:
        seed_text = (seed_text + "\n" + _read_text_file(Path(args.seed_file))).strip()

    config = SynthesisConfig(
        target_word_count=args.target_words,
        min_claim_length=args.min_claim_length,
        include_counterarguments=args.include_counterarguments,
    )

    pipeline = EssaySynthesisPipeline(config=config)
    pipeline.load_conversations(export_path)

    claims = pipeline.extract_all_claims()

    seed_scores_preview = []
    if seed_text:
        claims, seed_scores_preview = _select_claims_by_seed(
            claims,
            seed_text,
            min_score=args.seed_min_score,
            max_claims=args.max_claims,
        )
        pipeline._claims = claims  # update pipeline state for clustering
    elif args.max_claims:
        claims = claims[: args.max_claims]
        pipeline._claims = claims

    clusters = pipeline.cluster_claims(claims)

    # Optional attribution (off by default to avoid network/API requirements)
    if args.with_attribution:
        try:
            from aragora.connectors import (
                ArXivConnector,
                SemanticScholarConnector,
                CrossRefConnector,
            )

            pipeline.connectors = {
                "arxiv": ArXivConnector(),
                "semantic_scholar": SemanticScholarConnector(),
                "crossref": CrossRefConnector(),
            }
        except Exception:
            pipeline.connectors = {}

        if pipeline.connectors:
            import asyncio

            asyncio.run(pipeline.find_attribution(claims))

    import asyncio

    outline = asyncio.run(
        pipeline.generate_outline(
            title=args.title,
            thesis=args.thesis,
            clusters=clusters,
        )
    )

    export_payload = pipeline.export_for_synthesis(outline)

    # Build claim prefix lookup for markdown rendering
    claims_by_prefix = {c.claim[:50]: c.claim for c in claims}

    # Write outputs
    _write_json(output_dir / "essay_package.json", export_payload)
    _write_json(output_dir / "outline.json", outline.to_dict())
    _write_json(output_dir / "claims.json", {"claims": [asdict(c) for c in claims]})
    _write_json(output_dir / "clusters.json", {"clusters": [c.to_dict() for c in clusters]})

    (output_dir / "outline.md").write_text(
        _render_outline_markdown(outline, claims_by_prefix), encoding="utf-8"
    )
    (output_dir / "thread_skeleton.md").write_text(
        _render_thread_skeleton(outline), encoding="utf-8"
    )

    if seed_text:
        (output_dir / "seed.txt").write_text(seed_text, encoding="utf-8")
        _write_json(output_dir / "seed_scores.json", {"top_scores": seed_scores_preview})

    stats = pipeline.get_statistics()
    _write_json(output_dir / "stats.json", stats)

    print(f"Essay synthesis package written to: {output_dir}")
    print(f"Conversations: {stats.get('total_conversations')}")
    print(f"Claims: {stats.get('claims_extracted')}")
    print(f"Clusters: {stats.get('clusters_created')}")
    print(f"Outline sections: {outline.section_count}")


if __name__ == "__main__":
    main()
