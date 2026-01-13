"""
DPO (Direct Preference Optimization) exporter for EloSystem data.

Exports preference pairs from:
- ELO win/loss records (winner = chosen, loser = rejected)
- Calibration data (well-calibrated = chosen, overconfident = rejected)
- Consensus strength (unanimous = chosen, split = rejected)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import json
import logging

from aragora.training.exporters.base import BaseExporter, PreferenceRecord
from aragora.ranking.elo import EloSystem
from aragora.memory.store import CritiqueStore
from aragora.config import resolve_db_path, DB_ELO_PATH

logger = logging.getLogger(__name__)


@dataclass
class DPOExportConfig:
    """Configuration for DPO export."""

    min_elo_difference: float = 50.0
    min_debates: int = 3
    include_head_to_head: bool = True
    include_calibration: bool = True
    include_domain_specific: bool = True


class DPOExporter(BaseExporter):
    """
    Export preference pairs for Direct Preference Optimization training.

    Creates (prompt, chosen, rejected) triplets from:
    1. Head-to-head match outcomes
    2. Calibration quality comparisons
    3. Domain-specific performance

    Example:
        exporter = DPOExporter()
        data = exporter.export(min_elo_difference=100, limit=500)

        # Export to file
        exporter.export_to_file("dpo_training_data.jsonl")
    """

    exporter_type = "dpo"

    def __init__(
        self,
        elo_db_path: str = DB_ELO_PATH,
        critique_db_path: str = "agora_memory.db",
    ):
        self.elo_db_path = resolve_db_path(elo_db_path)
        self.critique_db_path = resolve_db_path(critique_db_path)
        self._elo: EloSystem | None = None
        self._store: CritiqueStore | None = None

    @property
    def elo(self) -> EloSystem:
        """Lazy-load EloSystem."""
        if self._elo is None:
            self._elo = EloSystem(self.elo_db_path)
        return self._elo

    @property
    def store(self) -> CritiqueStore:
        """Lazy-load CritiqueStore."""
        if self._store is None:
            self._store = CritiqueStore(self.critique_db_path)
        return self._store

    def export(
        self,
        min_elo_difference: float = 50.0,
        min_debates: int = 3,
        limit: int = 500,
        include_head_to_head: bool = True,
        include_calibration: bool = True,
        include_domain_specific: bool = True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Export DPO preference pairs.

        Args:
            min_elo_difference: Minimum ELO gap to consider agents distinct
            min_debates: Minimum debates between agents for head-to-head
            limit: Maximum records to export
            include_head_to_head: Include direct match comparisons
            include_calibration: Include calibration-based preferences
            include_domain_specific: Include domain performance comparisons

        Returns:
            List of preference records: {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        records: list[dict[str, Any]] = []

        if include_head_to_head:
            h2h_records = self._export_head_to_head(
                min_elo_difference=min_elo_difference,
                min_debates=min_debates,
                limit=limit,
            )
            records.extend(h2h_records)

        if include_calibration:
            cal_records = self._export_calibration_pairs(
                limit=max(0, limit - len(records)),
            )
            records.extend(cal_records)

        if include_domain_specific:
            domain_records = self._export_domain_pairs(
                min_elo_difference=min_elo_difference,
                limit=max(0, limit - len(records)),
            )
            records.extend(domain_records)

        logger.info(
            "Exported %d DPO records (h2h=%d, calibration=%d, domain=%d)",
            len(records),
            len([r for r in records if r.get("metadata", {}).get("source") == "head_to_head"]),
            len([r for r in records if r.get("metadata", {}).get("source") == "calibration"]),
            len([r for r in records if r.get("metadata", {}).get("source") == "domain"]),
        )

        return records[:limit]

    def _export_head_to_head(
        self,
        min_elo_difference: float,
        min_debates: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Export preference pairs from head-to-head matches."""
        records = []

        # Get recent matches with clear winners
        matches = self.elo.get_recent_matches(limit=limit * 2)

        for match in matches:
            winner = match.get("winner")
            if not winner:
                continue  # Skip draws

            participants = match.get("participants", [])
            scores = match.get("scores", {})
            debate_id = match.get("debate_id")

            if len(participants) < 2:
                continue

            # Find the loser with the lowest score
            sorted_by_score = sorted(
                [(p, scores.get(p, 0)) for p in participants if p != winner],
                key=lambda x: x[1],
            )
            if not sorted_by_score:
                continue

            loser = sorted_by_score[0][0]

            # Get ELO ratings to verify significant difference
            winner_rating = self.elo.get_rating(winner)
            loser_rating = self.elo.get_rating(loser)

            if abs(winner_rating.elo - loser_rating.elo) < min_elo_difference:
                continue

            # Get debate task from CritiqueStore
            task = self._get_debate_task(debate_id)
            if not task:
                continue

            # Get agent responses from the debate
            winner_response, loser_response = self._get_debate_responses(debate_id, winner, loser)

            if not winner_response or not loser_response:
                continue

            records.append(
                {
                    "prompt": task,
                    "chosen": winner_response,
                    "rejected": loser_response,
                    "metadata": {
                        "source": "head_to_head",
                        "debate_id": debate_id,
                        "winner": winner,
                        "loser": loser,
                        "winner_elo": winner_rating.elo,
                        "loser_elo": loser_rating.elo,
                        "elo_difference": winner_rating.elo - loser_rating.elo,
                    },
                }
            )

        return records

    def _export_calibration_pairs(self, limit: int) -> list[dict[str, Any]]:
        """Export preference pairs based on calibration quality."""
        records: list[dict[str, Any]] = []

        # Get calibration leaderboard
        calibration_lb = self.elo.get_calibration_leaderboard(limit=50)
        if len(calibration_lb) < 2:
            return records

        # Pair well-calibrated agents with poorly-calibrated ones
        well_calibrated = [a for a in calibration_lb if a.calibration_score > 0.7]
        poorly_calibrated = [a for a in calibration_lb if a.calibration_score < 0.4]

        for good_agent in well_calibrated[:10]:
            for bad_agent in poorly_calibrated[:10]:
                # Get head-to-head data
                h2h = self.elo.get_head_to_head(good_agent.agent_name, bad_agent.agent_name)

                if h2h.get("total_matches", 0) < 1:
                    continue

                # Create calibration-based preference
                prompt = (
                    "When making predictions about debate outcomes, it's important to "
                    "calibrate confidence levels accurately. An agent that says they are "
                    "90% confident should be correct about 90% of the time."
                )

                chosen = (
                    f"I approach confidence estimation carefully, acknowledging uncertainty "
                    f"when appropriate. My calibration score of {good_agent.calibration_score:.2f} "
                    f"reflects my commitment to honest probability assessments."
                )

                rejected = (
                    f"I tend to be overconfident in my predictions. While I may state high "
                    f"confidence, my actual accuracy doesn't match. This is reflected in my "
                    f"calibration score of {bad_agent.calibration_score:.2f}."
                )

                records.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {
                            "source": "calibration",
                            "good_agent": good_agent.agent_name,
                            "bad_agent": bad_agent.agent_name,
                            "good_calibration": good_agent.calibration_score,
                            "bad_calibration": bad_agent.calibration_score,
                        },
                    }
                )

                if len(records) >= limit:
                    return records

        return records

    def _export_domain_pairs(
        self,
        min_elo_difference: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Export preference pairs from domain-specific performance."""
        records = []

        # Common debate domains
        domains = ["security", "performance", "architecture", "testing", "correctness"]

        for domain in domains:
            # Get top agents for this domain
            top_agents = self.elo.get_top_agents_for_domain(domain, limit=10)
            if len(top_agents) < 2:
                continue

            # Create preference pairs between top and bottom performers
            for i, top_agent in enumerate(top_agents[:3]):
                for bottom_agent in top_agents[-3:]:
                    if top_agent.agent_name == bottom_agent.agent_name:
                        continue

                    top_domain_elo = top_agent.domain_elos.get(domain, 1000)
                    bottom_domain_elo = bottom_agent.domain_elos.get(domain, 1000)

                    if top_domain_elo - bottom_domain_elo < min_elo_difference:
                        continue

                    prompt = (
                        f"You are participating in a debate about {domain}. "
                        f"Provide expert-level analysis and recommendations."
                    )

                    chosen = (
                        f"As a specialist in {domain}, I focus on providing thorough, "
                        f"well-reasoned analysis backed by domain expertise. My approach "
                        f"prioritizes accuracy and depth over speed."
                    )

                    rejected = (
                        f"I'll try to address the {domain} aspects, though it's not my "
                        f"strongest area. My analysis may miss some nuances that a "
                        f"specialist would catch."
                    )

                    records.append(
                        {
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                            "metadata": {
                                "source": "domain",
                                "domain": domain,
                                "top_agent": top_agent.agent_name,
                                "bottom_agent": bottom_agent.agent_name,
                                "top_domain_elo": top_domain_elo,
                                "bottom_domain_elo": bottom_domain_elo,
                            },
                        }
                    )

                    if len(records) >= limit:
                        return records

        return records

    def _get_debate_task(self, debate_id: str) -> str | None:
        """Get the task for a debate."""
        with self.store.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT task FROM debates WHERE id = ?",
                (debate_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def _get_debate_responses(
        self,
        debate_id: str,
        winner: str,
        loser: str,
    ) -> tuple[str | None, str | None]:
        """Get responses from specific agents in a debate."""
        # Note: This would require storing agent responses in the database
        # For now, return placeholders - actual implementation would query
        # a response table or parse debate transcripts
        with self.store.connection() as conn:
            cursor = conn.cursor()

            # Get the final answer as the "winning" response
            cursor.execute(
                "SELECT final_answer FROM debates WHERE id = ?",
                (debate_id,),
            )
            row = cursor.fetchone()
            winner_response = row[0] if row else None

            # For loser, we'd ideally have their proposal stored
            # This is a simplification - full implementation would track all proposals
            loser_response = None

        return winner_response, loser_response


# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export DPO training data")
    parser.add_argument("--min-elo-difference", type=float, default=50.0)
    parser.add_argument("--min-debates", type=int, default=3)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", "-o", default="dpo_training_data.jsonl")
    args = parser.parse_args()

    exporter = DPOExporter()
    metadata = exporter.export_to_file(
        args.output,
        min_elo_difference=args.min_elo_difference,
        min_debates=args.min_debates,
        limit=args.limit,
    )

    print(f"Exported {metadata.total_records} records to {args.output}")
