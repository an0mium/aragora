"""
Replay Theater: Interactive HTML visualizations for aragora debates.

Generates self-contained HTML files with timelines and verdict cards.
"""

import html as html_module
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from aragora.core import DebateResult, Message, Vote

# Optional imports for trace support
try:
    from aragora.debate.traces import DebateTrace, EventType

    HAS_TRACE_SUPPORT = True
except ImportError:
    HAS_TRACE_SUPPORT = False
    DebateTrace: Optional[Type[Any]] = None
    EventType: Optional[Type[Any]] = None


@dataclass
class ReplayScene:
    """A single scene (round) in the debate replay."""

    round_number: int
    timestamp: datetime
    messages: List[Message] = field(default_factory=list)
    consensus_indicators: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization with HTML escaping."""
        return {
            "round_number": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "messages": [
                {
                    "role": html_module.escape(str(getattr(msg, "role", "unknown"))),
                    "agent": html_module.escape(str(getattr(msg, "agent", "unknown"))),
                    "content": html_module.escape(str(getattr(msg, "content", ""))),
                    "timestamp": getattr(msg, "timestamp", datetime.now()).isoformat(),
                    "round": getattr(msg, "round", 0),
                }
                for msg in self.messages
            ],
            "consensus_indicators": self.consensus_indicators,
        }


@dataclass
class ReplayArtifact:
    """Complete debate data for HTML generation."""

    debate_id: str
    task: str
    scenes: List[ReplayScene] = field(default_factory=list)
    verdict: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for JSON embedding in HTML."""
        return {
            "debate_id": self.debate_id,
            "task": self.task,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "verdict": self.verdict,
            "metadata": self.metadata,
        }


class ReplayGenerator:
    """Generates HTML replay artifacts from debate results."""

    def __init__(self):
        self.html_template = self._get_html_template()

    def generate(self, debate_result: DebateResult, trace: Optional["DebateTrace"] = None) -> str:
        """Generate HTML replay from a DebateResult.

        Args:
            debate_result: The DebateResult from orchestrator
            trace: Optional DebateTrace for accurate consensus markers

        Returns:
            Complete HTML document as string
        """
        artifact = self._create_artifact(debate_result, trace)
        return self._render_html(artifact)

    def _create_artifact(
        self, debate_result: DebateResult, trace: Optional["DebateTrace"] = None
    ) -> ReplayArtifact:
        """Transform DebateResult into ReplayArtifact."""
        # Group messages by round
        scenes = self._extract_scenes(debate_result.messages, trace)

        # Create verdict card data
        verdict = self._create_verdict_card(debate_result)

        # Metadata
        metadata = {
            "duration_seconds": debate_result.duration_seconds,
            "rounds_used": debate_result.rounds_used,
            "consensus_reached": debate_result.consensus_reached,
            "confidence": debate_result.confidence,
            "convergence_status": debate_result.convergence_status,
            "consensus_strength": debate_result.consensus_strength,
            "generated_at": datetime.now().isoformat(),
        }

        return ReplayArtifact(
            debate_id=debate_result.id,
            task=debate_result.task,
            scenes=scenes,
            verdict=verdict,
            metadata=metadata,
        )

    def _extract_scenes(
        self, messages: List[Message], trace: Optional["DebateTrace"] = None
    ) -> List[ReplayScene]:
        """Extract scenes (rounds) from messages with consensus indicators."""
        scenes: List[ReplayScene] = []
        round_groups: Dict[int, List[Message]] = {}

        # Group messages by round
        for msg in messages:
            if msg.round not in round_groups:
                round_groups[msg.round] = []
            round_groups[msg.round].append(msg)

        # Build consensus event map from trace (if available)
        consensus_events = {}
        if trace and HAS_TRACE_SUPPORT:
            for event in trace.events:
                if getattr(event, "type", None) == EventType.CONSENSUS_CHECK:
                    consensus_events[getattr(event, "round_num", 0)] = getattr(event, "data", {})

        # Create scenes
        for round_num in sorted(round_groups.keys()):
            msgs = round_groups[round_num]
            # Use timestamp of first message in round
            timestamp = msgs[0].timestamp if msgs else datetime.now()

            # Determine consensus indicator for this scene (default: not reached)
            consensus_indicators = {"reached": False, "source": "default"}
            if round_num in consensus_events:
                # Use actual trace data
                event_data = consensus_events[round_num]
                consensus_indicators = {
                    "reached": event_data.get("reached", False),
                    "confidence": event_data.get("confidence", 0),
                    "source": "trace",
                    "description": event_data.get("description", "Consensus check"),
                }
            elif (
                msgs
                and round_num == max(round_groups.keys())
                and getattr(msgs[0], "role", "") == "synthesizer"
            ):
                # Fallback: mark final round if consensus_reached in debate result
                # This will be overridden by verdict logic, but provides basic indication
                consensus_indicators = {
                    "reached": True,
                    "source": "fallback",
                    "description": "Final round (potential consensus)",
                }

            scene = ReplayScene(
                round_number=round_num,
                timestamp=timestamp,
                messages=msgs,
                consensus_indicators=consensus_indicators,
            )
            scenes.append(scene)

        return scenes

    def _create_verdict_card(self, debate_result: DebateResult) -> Dict[str, Any]:
        """Create verdict card data from debate result with proper tie handling."""
        votes = getattr(debate_result, "votes", []) or []
        consensus = getattr(debate_result, "consensus_reached", False)

        # Build vote breakdown
        vote_counts: Dict[str, List[Vote]] = {}
        for v in votes:
            choice = str(v.choice)
            vote_counts.setdefault(choice, []).append(v)

        vote_breakdown = []
        for choice, choice_votes in vote_counts.items():
            avg_conf = sum(v.confidence for v in choice_votes) / len(choice_votes)
            vote_breakdown.append(
                {
                    "choice": choice,
                    "count": len(choice_votes),
                    "avg_confidence": round(avg_conf, 2),
                }
            )

        # Determine winner with tie handling
        winner_label = "No winner"
        winner = None

        if consensus and vote_breakdown:
            # Sort by count descending
            sorted_votes = sorted(
                vote_breakdown, key=lambda x: int(str(x.get("count", 0) or 0)), reverse=True
            )

            if len(sorted_votes) >= 2 and sorted_votes[0]["count"] == sorted_votes[1]["count"]:
                winner_label = "Tie"
            else:
                winner = str(sorted_votes[0].get("choice", ""))
                winner_label = winner

        # Build evidence list (HTML-escaped for security)
        evidence = []
        if debate_result.winning_patterns:
            evidence.extend([html_module.escape(str(p)) for p in debate_result.winning_patterns])
        if debate_result.critiques:
            # Add key critique insights (limited and escaped)
            for critique in debate_result.critiques[:3]:  # Top 3
                evidence.append(
                    html_module.escape(
                        f"Critique from {critique.agent}: {critique.reasoning[:100]}..."
                    )
                )

        return {
            "final_answer": html_module.escape(
                str(getattr(debate_result, "final_answer", "") or "")
            ),
            "confidence": getattr(debate_result, "confidence", 0),
            "consensus_reached": consensus,
            "winner": winner,
            "winner_label": winner_label,
            "evidence": evidence[:5],  # Limit to 5 items
            "rounds_used": getattr(debate_result, "rounds_used", 0),
            "duration_seconds": getattr(debate_result, "duration_seconds", 0),
            "dissenting_views": getattr(debate_result, "dissenting_views", []),
            "vote_breakdown": vote_breakdown,
            "convergence_status": getattr(debate_result, "convergence_status", None),
        }

    def _render_html(self, artifact: ReplayArtifact) -> str:
        """Render HTML using the template with security measures."""
        # Safe JSON embedding: escape </script> to prevent tag termination
        data_json = json.dumps(artifact.to_dict(), indent=2)
        safe_json = data_json.replace("</script>", "</\\script>")

        html = self.html_template.replace("{{DATA}}", safe_json)
        debate_id_escaped = (
            html_module.escape(str(artifact.debate_id)[:8]) if artifact.debate_id else "unknown"
        )
        html = html.replace("{{DEBATE_ID}}", debate_id_escaped)
        return html

    def _get_html_template(self) -> str:
        """Load the HTML template from the templates directory.

        Returns the template content with {{DATA}} and {{DEBATE_ID}} placeholders.
        Falls back to a minimal inline template if file cannot be loaded.
        """
        template_path = Path(__file__).parent / "templates" / "replay.html"
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, IOError) as e:
            # Fallback: return minimal template if file not found
            import logging

            logging.getLogger(__name__).warning(f"Could not load replay template: {e}")
            return """<!DOCTYPE html>
<html><head><title>Replay - {{DEBATE_ID}}</title></head>
<body><h1>Debate Replay</h1><pre id="data">{{DATA}}</pre>
<script>document.getElementById('data').textContent = JSON.stringify({{DATA}}, null, 2);</script>
</body></html>"""
