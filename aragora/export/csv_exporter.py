"""
CSV exporter for debate data analysis.

Exports debate artifacts to CSV format for analysis with spreadsheet tools
or data science pipelines (pandas, R, etc.).

Provides multiple export formats:
- messages: All debate messages with metadata
- critiques: All critiques with severity and outcomes
- votes: Vote breakdown for consensus analysis
- summary: High-level debate statistics
"""

__all__ = [
    "CSVExporter",
    "export_debate_to_csv",
    "export_multiple_debates",
]

import csv
from io import StringIO
from pathlib import Path
from typing import Optional

from aragora.export.artifact import DebateArtifact


class CSVExporter:
    """
    Export debate artifacts to CSV format.

    Supports multiple table formats for different analysis needs.
    """

    def __init__(self, artifact: DebateArtifact):
        self.artifact = artifact

    def export_messages(self, output_path: Optional[Path] = None) -> str:
        """
        Export all messages as CSV.

        Columns: round, agent, role, content, timestamp
        """
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["debate_id", "round", "agent", "role", "content", "timestamp"])

        # Extract messages from trace
        if self.artifact.trace_data and "events" in self.artifact.trace_data:
            for event in self.artifact.trace_data["events"]:
                if event.get("event_type") == "message":
                    writer.writerow(
                        [
                            self.artifact.debate_id,
                            event.get("round", 0),
                            event.get("agent", ""),
                            event.get("role", ""),
                            event.get("content", ""),
                            event.get("timestamp", ""),
                        ]
                    )

        content = output.getvalue()
        if output_path:
            output_path.write_text(content)
        return content

    def export_critiques(self, output_path: Optional[Path] = None) -> str:
        """
        Export all critiques as CSV.

        Columns: round, critic, target, severity, issues, accepted
        """
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "debate_id",
                "round",
                "critic",
                "target",
                "severity",
                "issue_count",
                "issues",
                "accepted",
            ]
        )

        # Extract critiques from trace
        if self.artifact.trace_data and "events" in self.artifact.trace_data:
            for event in self.artifact.trace_data["events"]:
                if event.get("event_type") == "critique":
                    issues = event.get("issues", [])
                    writer.writerow(
                        [
                            self.artifact.debate_id,
                            event.get("round", 0),
                            event.get("agent", ""),
                            event.get("target", ""),
                            event.get("severity", 0),
                            len(issues),
                            "; ".join(issues) if isinstance(issues, list) else str(issues),
                            event.get("accepted", ""),
                        ]
                    )

        content = output.getvalue()
        if output_path:
            output_path.write_text(content)
        return content

    def export_votes(self, output_path: Optional[Path] = None) -> str:
        """
        Export vote breakdown as CSV.

        Columns: agent, voted_for, agreed_with_consensus
        """
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            ["debate_id", "agent", "agreed_with_consensus", "final_answer", "confidence"]
        )

        if self.artifact.consensus_proof:
            cp = self.artifact.consensus_proof
            for agent, agreed in cp.vote_breakdown.items():
                writer.writerow(
                    [
                        self.artifact.debate_id,
                        agent,
                        agreed,
                        cp.final_answer[:100],
                        cp.confidence,
                    ]
                )

        content = output.getvalue()
        if output_path:
            output_path.write_text(content)
        return content

    def export_summary(self, output_path: Optional[Path] = None) -> str:
        """
        Export high-level summary as CSV.

        One row per debate with key statistics.
        """
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "debate_id",
                "artifact_id",
                "task",
                "agents",
                "rounds",
                "messages",
                "critiques",
                "consensus_reached",
                "confidence",
                "duration_seconds",
                "created_at",
            ]
        )

        cp = self.artifact.consensus_proof
        writer.writerow(
            [
                self.artifact.debate_id,
                self.artifact.artifact_id,
                self.artifact.task[:200],
                ";".join(self.artifact.agents),
                self.artifact.rounds,
                self.artifact.message_count,
                self.artifact.critique_count,
                cp.reached if cp else "",
                cp.confidence if cp else "",
                self.artifact.duration_seconds,
                self.artifact.created_at,
            ]
        )

        content = output.getvalue()
        if output_path:
            output_path.write_text(content)
        return content

    def export_verifications(self, output_path: Optional[Path] = None) -> str:
        """
        Export verification results as CSV.

        Columns: claim_id, claim_text, status, method, duration_ms
        """
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "debate_id",
                "claim_id",
                "claim_text",
                "status",
                "method",
                "duration_ms",
                "has_proof",
                "has_counterexample",
            ]
        )

        for v in self.artifact.verification_results:
            writer.writerow(
                [
                    self.artifact.debate_id,
                    v.claim_id,
                    v.claim_text[:200],
                    v.status,
                    v.method,
                    v.duration_ms,
                    bool(v.proof_trace),
                    bool(v.counterexample),
                ]
            )

        content = output.getvalue()
        if output_path:
            output_path.write_text(content)
        return content

    def export_all(self, output_dir: Path) -> dict[str, Path]:
        """
        Export all CSV formats to a directory.

        Returns dict of type -> output path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        messages_path = output_dir / f"{self.artifact.artifact_id}_messages.csv"
        self.export_messages(messages_path)
        outputs["messages"] = messages_path

        critiques_path = output_dir / f"{self.artifact.artifact_id}_critiques.csv"
        self.export_critiques(critiques_path)
        outputs["critiques"] = critiques_path

        votes_path = output_dir / f"{self.artifact.artifact_id}_votes.csv"
        self.export_votes(votes_path)
        outputs["votes"] = votes_path

        summary_path = output_dir / f"{self.artifact.artifact_id}_summary.csv"
        self.export_summary(summary_path)
        outputs["summary"] = summary_path

        if self.artifact.verification_results:
            verifications_path = output_dir / f"{self.artifact.artifact_id}_verifications.csv"
            self.export_verifications(verifications_path)
            outputs["verifications"] = verifications_path

        return outputs


def export_debate_to_csv(
    artifact: DebateArtifact,
    output_path: Path,
    table: str = "summary",
) -> str:
    """
    Convenience function to export a debate to CSV format.

    Args:
        artifact: The debate artifact to export
        output_path: Where to save the CSV file
        table: "messages", "critiques", "votes", "summary", or "verifications"

    Returns:
        The CSV content as a string
    """
    exporter = CSVExporter(artifact)

    if table == "messages":
        return exporter.export_messages(output_path)
    elif table == "critiques":
        return exporter.export_critiques(output_path)
    elif table == "votes":
        return exporter.export_votes(output_path)
    elif table == "summary":
        return exporter.export_summary(output_path)
    elif table == "verifications":
        return exporter.export_verifications(output_path)
    else:
        raise ValueError(
            f"Unknown table: {table}. Use 'messages', 'critiques', 'votes', 'summary', or 'verifications'"
        )


def export_multiple_debates(
    artifacts: list[DebateArtifact],
    output_path: Path,
    table: str = "summary",
) -> str:
    """
    Export multiple debates to a single CSV file.

    Useful for batch analysis across many debates.
    """
    output = StringIO()

    # Get header from first artifact
    first_exporter = CSVExporter(artifacts[0])

    if table == "summary":
        first_content = first_exporter.export_summary()
    elif table == "messages":
        first_content = first_exporter.export_messages()
    elif table == "critiques":
        first_content = first_exporter.export_critiques()
    elif table == "votes":
        first_content = first_exporter.export_votes()
    else:
        raise ValueError(f"Unknown table: {table}")

    output.write(first_content)

    # Append remaining artifacts (skip header)
    for artifact in artifacts[1:]:
        exporter = CSVExporter(artifact)
        if table == "summary":
            content = exporter.export_summary()
        elif table == "messages":
            content = exporter.export_messages()
        elif table == "critiques":
            content = exporter.export_critiques()
        elif table == "votes":
            content = exporter.export_votes()

        # Skip header line
        lines = content.split("\n")
        if len(lines) > 1:
            output.write("\n".join(lines[1:]))

    content = output.getvalue()
    output_path.write_text(content)
    return content
