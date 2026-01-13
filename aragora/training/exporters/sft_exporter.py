"""
SFT (Supervised Fine-Tuning) exporter for CritiqueStore data.

Exports successful debate patterns, critiques, and winning responses
for instruction-response style training.

Data Sources:
- High-confidence debate winners
- Successful critique patterns (success_rate >= threshold)
- Domain expert responses
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from aragora.config import resolve_db_path
from aragora.memory.store import CritiqueStore
from aragora.training.exporters.base import BaseExporter

logger = logging.getLogger(__name__)


@dataclass
class SFTExportConfig:
    """Configuration for SFT export."""

    min_confidence: float = 0.7
    min_success_rate: float = 0.6
    min_pattern_count: int = 2
    include_critiques: bool = True
    include_patterns: bool = True
    include_debates: bool = True
    max_response_length: int = 4096


class SFTExporter(BaseExporter):
    """
    Export training data for Supervised Fine-Tuning.

    Extracts instruction-response pairs from:
    1. Winning debate responses (task -> answer)
    2. Successful critique patterns (issue -> suggestion)
    3. Expert domain responses

    Example:
        exporter = SFTExporter()
        data = exporter.export(min_confidence=0.8, limit=1000)

        # Export to file
        exporter.export_to_file("training_data.jsonl", limit=5000)
    """

    exporter_type = "sft"

    def __init__(self, db_path: str = "agora_memory.db"):
        self.db_path = resolve_db_path(db_path)
        self._store: CritiqueStore | None = None

    @property
    def store(self) -> CritiqueStore:
        """Lazy-load CritiqueStore."""
        if self._store is None:
            self._store = CritiqueStore(self.db_path)
        return self._store

    def export(
        self,
        min_confidence: float = 0.7,
        min_success_rate: float = 0.6,
        limit: int = 1000,
        offset: int = 0,
        include_critiques: bool = True,
        include_patterns: bool = True,
        include_debates: bool = True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Export SFT training data.

        Args:
            min_confidence: Minimum debate confidence to include
            min_success_rate: Minimum pattern success rate
            limit: Maximum records to export
            offset: Starting offset for pagination
            include_critiques: Include critique-based training data
            include_patterns: Include successful patterns
            include_debates: Include winning debate responses

        Returns:
            List of training records: {"instruction": ..., "response": ...}
        """
        records: list[dict[str, Any]] = []

        if include_debates:
            debate_records = self._export_debates(
                min_confidence=min_confidence,
                limit=limit,
                offset=offset,
            )
            records.extend(debate_records)

        if include_patterns:
            pattern_records = self._export_patterns(
                min_success_rate=min_success_rate,
                limit=max(0, limit - len(records)),
            )
            records.extend(pattern_records)

        if include_critiques:
            critique_records = self._export_critiques(
                min_confidence=min_confidence,
                limit=max(0, limit - len(records)),
            )
            records.extend(critique_records)

        logger.info(
            "Exported %d SFT records (debates=%d, patterns=%d, critiques=%d)",
            len(records),
            len([r for r in records if r.get("metadata", {}).get("source") == "debate"]),
            len([r for r in records if r.get("metadata", {}).get("source") == "pattern"]),
            len([r for r in records if r.get("metadata", {}).get("source") == "critique"]),
        )

        return records[:limit]

    def _export_debates(
        self,
        min_confidence: float,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Export winning debate responses."""
        records = []

        with self.store.connection() as conn:
            cursor = conn.cursor()

            # Get high-confidence consensus debates
            cursor.execute(
                """
                SELECT id, task, final_answer, confidence, rounds_used
                FROM debates
                WHERE consensus_reached = 1
                  AND confidence >= ?
                  AND final_answer IS NOT NULL
                  AND LENGTH(final_answer) > 50
                ORDER BY confidence DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (min_confidence, limit, offset),
            )

            for row in cursor.fetchall():
                debate_id, task, answer, confidence, rounds = row

                # Create instruction-response pair
                instruction = self._format_debate_instruction(task)
                response = answer.strip()

                if len(response) > 50:  # Skip empty/trivial responses
                    records.append(
                        {
                            "instruction": instruction,
                            "response": response,
                            "metadata": {
                                "source": "debate",
                                "debate_id": debate_id,
                                "confidence": confidence,
                                "rounds_used": rounds,
                            },
                        }
                    )

        return records

    def _export_patterns(
        self,
        min_success_rate: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Export successful critique patterns."""
        records = []

        patterns = self.store.retrieve_patterns(
            min_success=2,
            limit=limit,
        )

        for pattern in patterns:
            if pattern.success_rate < min_success_rate:
                continue

            # Create instruction-response pair for critique patterns
            instruction = self._format_pattern_instruction(
                pattern.issue_type,
                pattern.issue_text,
            )
            response = pattern.suggestion_text or f"Address the issue: {pattern.issue_text}"

            if len(response) > 20:
                records.append(
                    {
                        "instruction": instruction,
                        "response": response,
                        "metadata": {
                            "source": "pattern",
                            "pattern_id": pattern.id,
                            "issue_type": pattern.issue_type,
                            "success_rate": pattern.success_rate,
                            "success_count": pattern.success_count,
                        },
                    }
                )

        return records

    def _export_critiques(
        self,
        min_confidence: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Export critiques from high-confidence debates."""
        records = []

        with self.store.connection() as conn:
            cursor = conn.cursor()

            # Get critiques from successful debates
            cursor.execute(
                """
                SELECT c.agent, c.target_agent, c.issues, c.suggestions,
                       c.severity, c.reasoning, d.task, d.confidence
                FROM critiques c
                JOIN debates d ON c.debate_id = d.id
                WHERE d.consensus_reached = 1
                  AND d.confidence >= ?
                  AND c.led_to_improvement = 1
                ORDER BY d.confidence DESC
                LIMIT ?
                """,
                (min_confidence, limit),
            )

            for row in cursor.fetchall():
                (
                    agent,
                    target,
                    issues_json,
                    suggestions_json,
                    severity,
                    reasoning,
                    task,
                    confidence,
                ) = row

                try:
                    issues = json.loads(issues_json) if issues_json else []
                    suggestions = json.loads(suggestions_json) if suggestions_json else []
                except json.JSONDecodeError:
                    continue

                if not issues or not suggestions:
                    continue

                # Create instruction-response pair
                instruction = self._format_critique_instruction(task, issues)
                response = self._format_critique_response(suggestions, reasoning)

                if len(response) > 50:
                    records.append(
                        {
                            "instruction": instruction,
                            "response": response,
                            "metadata": {
                                "source": "critique",
                                "agent": agent,
                                "target_agent": target,
                                "severity": severity,
                                "confidence": confidence,
                            },
                        }
                    )

        return records

    def _format_debate_instruction(self, task: str) -> str:
        """Format a debate task as an instruction."""
        return f"Analyze the following task and provide a well-reasoned response that addresses all aspects:\n\n{task}"

    def _format_pattern_instruction(self, issue_type: str, issue_text: str) -> str:
        """Format a pattern as an instruction."""
        return (
            f"You are reviewing code or a proposal. The following {issue_type} issue "
            f"has been identified:\n\n{issue_text}\n\n"
            "Provide a concrete suggestion to address this issue."
        )

    def _format_critique_instruction(self, task: str, issues: list[str]) -> str:
        """Format a critique as an instruction."""
        issues_text = "\n".join(f"- {issue}" for issue in issues[:5])
        return (
            f"Given the following debate task:\n\n{task}\n\n"
            f"The following issues have been identified:\n{issues_text}\n\n"
            "Provide suggestions to address these issues."
        )

    def _format_critique_response(
        self,
        suggestions: list[str],
        reasoning: str | None,
    ) -> str:
        """Format critique suggestions as a response."""
        response_parts = []

        if reasoning:
            response_parts.append(reasoning)

        if suggestions:
            suggestions_text = "\n".join(f"- {s}" for s in suggestions[:5])
            response_parts.append(f"\nSuggestions:\n{suggestions_text}")

        return "\n".join(response_parts)


# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export SFT training data")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--min-success-rate", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", "-o", default="sft_training_data.jsonl")
    parser.add_argument("--db-path", default="agora_memory.db")
    args = parser.parse_args()

    exporter = SFTExporter(db_path=args.db_path)
    metadata = exporter.export_to_file(
        args.output,
        min_confidence=args.min_confidence,
        min_success_rate=args.min_success_rate,
        limit=args.limit,
    )

    print(f"Exported {metadata.total_records} records to {args.output}")
