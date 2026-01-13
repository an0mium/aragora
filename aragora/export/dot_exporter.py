"""
DOT/GraphViz exporter for debate visualization.

Exports debate graphs in DOT format for visualization with GraphViz tools.
Supports different visualization modes:
- flow: Shows message flow between agents
- critiques: Shows critique relationships
- consensus: Highlights consensus-building path
"""

from pathlib import Path
from typing import Optional
from aragora.export.artifact import DebateArtifact


def escape_label(text: str, max_len: int = 50) -> str:
    """Escape text for DOT labels."""
    # Truncate and escape special characters
    if len(text) > max_len:
        text = text[:max_len] + "..."
    # Escape quotes and newlines
    text = text.replace('"', '\\"').replace("\n", "\\n")
    return text


class DOTExporter:
    """
    Export debate artifacts to DOT/GraphViz format.

    Supports multiple visualization modes for different analysis needs.
    """

    def __init__(self, artifact: DebateArtifact):
        self.artifact = artifact

    def export_flow(self, output_path: Optional[Path] = None) -> str:
        """
        Export message flow graph.

        Shows how messages flow between agents across rounds.
        """
        lines = [
            "digraph debate_flow {",
            "    rankdir=TB;",
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial", fontsize=10];',
            "",
            "    // Agent color scheme",
        ]

        # Define colors for each agent
        colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5", "#E0F7FA"]
        for i, agent in enumerate(self.artifact.agents):
            color = colors[i % len(colors)]
            safe_name = agent.replace("-", "_").replace(".", "_")
            lines.append(f"    subgraph cluster_{safe_name} {{")
            lines.append(f'        label="{agent}";')
            lines.append(f"        style=filled;")
            lines.append(f'        color="{color}";')
            lines.append("    }")

        lines.append("")
        lines.append("    // Message nodes")

        # Extract messages from trace data
        if self.artifact.trace_data and "events" in self.artifact.trace_data:
            msg_count = 0
            prev_node = None
            for event in self.artifact.trace_data["events"]:
                if event.get("event_type") == "message":
                    msg_count += 1
                    node_id = f"msg_{msg_count}"
                    agent = event.get("agent", "unknown")
                    content = event.get("content", "")[:100]
                    round_num = event.get("round", 0)

                    label = escape_label(f"R{round_num}: {content}")
                    lines.append(f'    {node_id} [label="{label}"];')

                    if prev_node:
                        lines.append(f"    {prev_node} -> {node_id};")
                    prev_node = node_id

        # Add consensus result
        if self.artifact.consensus_proof:
            cp = self.artifact.consensus_proof
            status = "Consensus" if cp.reached else "No Consensus"
            confidence = f"{cp.confidence:.0%}"
            lines.append("")
            lines.append("    // Consensus result")
            lines.append(
                f'    consensus [label="{status}\\n{confidence}", shape=ellipse, style=filled, color="#C8E6C9"];'
            )
            if prev_node:
                lines.append(f"    {prev_node} -> consensus [style=dashed];")

        lines.append("}")

        dot_content = "\n".join(lines)

        if output_path:
            output_path.write_text(dot_content)

        return dot_content

    def export_critiques(self, output_path: Optional[Path] = None) -> str:
        """
        Export critique relationship graph.

        Shows who critiqued whom and with what severity.
        """
        lines = [
            "digraph critique_graph {",
            "    rankdir=LR;",
            '    node [shape=ellipse, style=filled, fontname="Arial"];',
            '    edge [fontname="Arial"];',
            "",
        ]

        # Create agent nodes
        for agent in self.artifact.agents:
            safe_name = agent.replace("-", "_").replace(".", "_")
            lines.append(f'    {safe_name} [label="{agent}", fillcolor="#E3F2FD"];')

        lines.append("")
        lines.append("    // Critique edges")

        # Extract critiques from trace data
        if self.artifact.trace_data and "events" in self.artifact.trace_data:
            critique_edges = {}  # (from, to) -> count
            for event in self.artifact.trace_data["events"]:
                if event.get("event_type") == "critique":
                    critic = event.get("agent", "unknown")
                    target = event.get("target", "unknown")
                    severity = event.get("severity", 0.5)

                    key = (critic, target)
                    if key not in critique_edges:
                        critique_edges[key] = {"count": 0, "total_severity": 0}
                    critique_edges[key]["count"] += 1
                    critique_edges[key]["total_severity"] += severity

            # Add critique edges
            for (critic, target), data in critique_edges.items():
                safe_critic = critic.replace("-", "_").replace(".", "_")
                safe_target = target.replace("-", "_").replace(".", "_")
                avg_severity = data["total_severity"] / data["count"]
                count = data["count"]

                # Color by severity (red = high, green = low)
                if avg_severity > 0.7:
                    color = "#F44336"  # Red
                elif avg_severity > 0.4:
                    color = "#FF9800"  # Orange
                else:
                    color = "#4CAF50"  # Green

                weight = min(5, count)
                lines.append(
                    f'    {safe_critic} -> {safe_target} [label="{count}x ({avg_severity:.1f})", color="{color}", penwidth={weight}];'
                )

        lines.append("}")

        dot_content = "\n".join(lines)

        if output_path:
            output_path.write_text(dot_content)

        return dot_content

    def export_consensus(self, output_path: Optional[Path] = None) -> str:
        """
        Export consensus-building graph.

        Shows the path to consensus with vote breakdown.
        """
        lines = [
            "digraph consensus_path {",
            "    rankdir=TB;",
            '    node [fontname="Arial"];',
            '    edge [fontname="Arial"];',
            "",
            "    // Task",
            f'    task [label="{escape_label(self.artifact.task)}", shape=box, style=filled, fillcolor="#BBDEFB"];',
            "",
            "    // Agent votes",
        ]

        if self.artifact.consensus_proof:
            cp = self.artifact.consensus_proof
            for agent, agreed in cp.vote_breakdown.items():
                safe_name = agent.replace("-", "_").replace(".", "_")
                color = "#C8E6C9" if agreed else "#FFCDD2"
                vote = "Agreed" if agreed else "Disagreed"
                lines.append(
                    f'    {safe_name} [label="{agent}\\n{vote}", shape=box, style=filled, fillcolor="{color}"];'
                )
                lines.append(f"    task -> {safe_name};")

            # Final answer
            lines.append("")
            lines.append("    // Final answer")
            status = "CONSENSUS" if cp.reached else "NO CONSENSUS"
            answer = escape_label(cp.final_answer, 80)
            lines.append(
                f'    final [label="{status}\\n{cp.confidence:.0%}\\n{answer}", shape=box, style="rounded,filled", fillcolor="#FFF9C4"];'
            )

            for agent in cp.vote_breakdown.keys():
                safe_name = agent.replace("-", "_").replace(".", "_")
                lines.append(f"    {safe_name} -> final [style=dashed];")

        lines.append("}")

        dot_content = "\n".join(lines)

        if output_path:
            output_path.write_text(dot_content)

        return dot_content

    def export_all(self, output_dir: Path) -> dict[str, Path]:
        """
        Export all visualization types to a directory.

        Returns dict of type -> output path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        flow_path = output_dir / f"{self.artifact.artifact_id}_flow.dot"
        self.export_flow(flow_path)
        outputs["flow"] = flow_path

        critiques_path = output_dir / f"{self.artifact.artifact_id}_critiques.dot"
        self.export_critiques(critiques_path)
        outputs["critiques"] = critiques_path

        consensus_path = output_dir / f"{self.artifact.artifact_id}_consensus.dot"
        self.export_consensus(consensus_path)
        outputs["consensus"] = consensus_path

        return outputs


def export_debate_to_dot(
    artifact: DebateArtifact,
    output_path: Path,
    mode: str = "flow",
) -> str:
    """
    Convenience function to export a debate to DOT format.

    Args:
        artifact: The debate artifact to export
        output_path: Where to save the DOT file
        mode: "flow", "critiques", or "consensus"

    Returns:
        The DOT content as a string
    """
    exporter = DOTExporter(artifact)

    if mode == "flow":
        return exporter.export_flow(output_path)
    elif mode == "critiques":
        return exporter.export_critiques(output_path)
    elif mode == "consensus":
        return exporter.export_consensus(output_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'flow', 'critiques', or 'consensus'")
