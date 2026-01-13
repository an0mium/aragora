"""
Debate-to-training data exporter for real-time training data collection.

This module provides infrastructure for automatically exporting training
data from completed debates. It's designed to be hooked into the Arena
orchestrator for continuous self-improvement via Tinker.

Features:
- Real-time export after debate completion
- SFT export for winning responses
- DPO export for preference pairs (winner vs losers)
- Configurable confidence thresholds
- Automatic path management
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DebateTrainingConfig:
    """Configuration for debate training export."""

    # Output settings
    output_dir: str = "data/training"
    sft_file: str = "sft_debates.jsonl"
    dpo_file: str = "dpo_debates.jsonl"

    # Quality thresholds
    min_confidence: float = 0.75
    min_rounds: int = 2
    require_consensus: bool = True

    # Export toggles
    export_sft: bool = True
    export_dpo: bool = True

    # Metadata
    include_agent_names: bool = True
    include_debate_id: bool = True
    include_domain: bool = True


class DebateTrainingExporter:
    """
    Export training data from completed debates.

    This class is designed to be instantiated once and reused across
    multiple debates. It maintains append-only logs of training data.

    Example:
        exporter = DebateTrainingExporter()

        # After each debate completes
        exporter.export_debate(result, debate_id="abc123", domain="tech")

        # Get statistics
        stats = exporter.get_stats()
        print(f"Exported {stats['sft_count']} SFT records")
    """

    def __init__(self, config: Optional[DebateTrainingConfig] = None):
        self.config = config or DebateTrainingConfig()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self._sft_count = 0
        self._dpo_count = 0
        self._skipped_count = 0

    def export_debate(
        self,
        result: Any,
        debate_id: str = "",
        domain: str = "",
        agents: Optional[list] = None,
    ) -> dict[str, int]:
        """
        Export training data from a completed debate.

        Args:
            result: DebateResult from Arena.run()
            debate_id: Unique debate identifier
            domain: Domain classification (e.g., "tech", "ethics")
            agents: List of agents that participated

        Returns:
            Dict with counts: {"sft": N, "dpo": M, "skipped": K}
        """
        exported = {"sft": 0, "dpo": 0, "skipped": 0}

        # Check quality thresholds
        if not self._meets_quality_threshold(result):
            self._skipped_count += 1
            exported["skipped"] = 1
            logger.debug(
                "debate_export_skipped debate_id=%s reason=quality_threshold",
                debate_id[:8] if debate_id else "unknown",
            )
            return exported

        # Export SFT data (winning response)
        if self.config.export_sft and hasattr(result, "final_answer") and result.final_answer:
            sft_record = self._create_sft_record(result, debate_id, domain, agents)
            if sft_record:
                self._write_record(self.config.sft_file, sft_record)
                exported["sft"] = 1
                self._sft_count += 1

        # Export DPO data (preference pairs)
        if self.config.export_dpo and hasattr(result, "messages") and len(result.messages) >= 2:
            dpo_records = self._create_dpo_records(result, debate_id, domain, agents)
            for record in dpo_records:
                self._write_record(self.config.dpo_file, record)
                exported["dpo"] += 1
                self._dpo_count += 1

        logger.info(
            "debate_export_complete debate_id=%s sft=%d dpo=%d",
            debate_id[:8] if debate_id else "unknown",
            exported["sft"],
            exported["dpo"],
        )

        return exported

    def _meets_quality_threshold(self, result: Any) -> bool:
        """Check if debate result meets quality thresholds for export."""
        # Check confidence
        confidence = getattr(result, "confidence", 0.0)
        if confidence < self.config.min_confidence:
            return False

        # Check rounds
        rounds_used = getattr(result, "rounds_used", 0)
        if rounds_used < self.config.min_rounds:
            return False

        # Check consensus (if required)
        if self.config.require_consensus:
            consensus = getattr(result, "consensus_reached", False)
            if not consensus:
                return False

        # Check for actual content
        final_answer = getattr(result, "final_answer", "")
        if not final_answer or len(final_answer.strip()) < 50:
            return False

        return True

    def _create_sft_record(
        self,
        result: Any,
        debate_id: str,
        domain: str,
        agents: Optional[list],
    ) -> Optional[dict[str, Any]]:
        """Create an SFT training record from debate result."""
        task = getattr(result, "task", "")
        final_answer = getattr(result, "final_answer", "")

        if not task or not final_answer:
            return None

        # Format instruction
        instruction = (
            f"Analyze the following task and provide a well-reasoned, "
            f"comprehensive response:\n\n{task}"
        )

        # Build metadata
        metadata = {
            "source": "debate_auto_export",
            "exported_at": datetime.now().isoformat(),
            "confidence": getattr(result, "confidence", 0.0),
            "rounds_used": getattr(result, "rounds_used", 0),
        }

        if self.config.include_debate_id and debate_id:
            metadata["debate_id"] = debate_id

        if self.config.include_domain and domain:
            metadata["domain"] = domain

        if self.config.include_agent_names and agents:
            metadata["agents"] = [getattr(a, "name", str(a)) for a in agents]

        return {
            "instruction": instruction,
            "response": final_answer.strip(),
            "metadata": metadata,
        }

    def _create_dpo_records(
        self,
        result: Any,
        debate_id: str,
        domain: str,
        agents: Optional[list],
    ) -> list[dict[str, Any]]:
        """Create DPO preference records from debate result.

        Pairs the winning response against other proposals that
        didn't become the final answer.
        """
        records: list[dict[str, Any]] = []

        task = getattr(result, "task", "")
        final_answer = getattr(result, "final_answer", "")
        messages = getattr(result, "messages", [])

        if not task or not final_answer or len(messages) < 2:
            return records

        # Format prompt
        prompt = (
            f"You are participating in a multi-agent debate. "
            f"The task is:\n\n{task}\n\n"
            f"Provide your best response."
        )

        # Find alternative proposals (not the winning answer)
        seen_responses = set()
        seen_responses.add(final_answer.strip().lower()[:200])  # Normalize for comparison

        for msg in messages:
            content = ""
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                content = msg.content

            if not content or len(content.strip()) < 50:
                continue

            normalized = content.strip().lower()[:200]
            if normalized in seen_responses:
                continue
            seen_responses.add(normalized)

            # Build metadata
            metadata = {
                "source": "debate_dpo_export",
                "exported_at": datetime.now().isoformat(),
            }

            if self.config.include_debate_id and debate_id:
                metadata["debate_id"] = debate_id

            if self.config.include_domain and domain:
                metadata["domain"] = domain

            # Create preference pair: winner (chosen) vs this response (rejected)
            records.append(
                {
                    "prompt": prompt,
                    "chosen": final_answer.strip(),
                    "rejected": content.strip(),
                    "metadata": metadata,
                }
            )

        return records

    def _write_record(self, filename: str, record: dict[str, Any]) -> None:
        """Append a record to the specified JSONL file."""
        filepath = self._output_dir / filename

        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_stats(self) -> dict[str, Any]:
        """Get export statistics."""
        return {
            "sft_count": self._sft_count,
            "dpo_count": self._dpo_count,
            "skipped_count": self._skipped_count,
            "output_dir": str(self._output_dir),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._sft_count = 0
        self._dpo_count = 0
        self._skipped_count = 0


# Convenience function for one-off exports
def export_debate_to_training(
    result: Any,
    debate_id: str = "",
    domain: str = "",
    output_dir: str = "data/training",
    min_confidence: float = 0.75,
) -> dict[str, int]:
    """
    Export a single debate result to training data.

    Convenience function for one-off exports without maintaining
    a persistent exporter instance.

    Args:
        result: DebateResult from Arena.run()
        debate_id: Unique debate identifier
        domain: Domain classification
        output_dir: Directory for training data output
        min_confidence: Minimum confidence threshold

    Returns:
        Dict with export counts
    """
    config = DebateTrainingConfig(
        output_dir=output_dir,
        min_confidence=min_confidence,
    )
    exporter = DebateTrainingExporter(config)
    return exporter.export_debate(result, debate_id=debate_id, domain=domain)
