"""
Debate export formatting utilities.

Provides CSV and HTML export formatters for debates.
Extracted from handlers/debates.py for better modularity.
"""

import csv
import html
import io
from dataclasses import dataclass
from typing import Any, Optional

# Type alias for csv.writer return type (csv._writer is internal)
CSVWriter = Any


@dataclass
class ExportResult:
    """Result of export formatting."""

    content: bytes
    content_type: str
    filename: str


def format_debate_csv(debate: dict, table: str = "summary") -> ExportResult:
    """Format debate as CSV for the specified table type.

    Args:
        debate: Debate data dictionary
        table: Table type ("messages", "critiques", "votes", "summary")

    Returns:
        ExportResult with CSV content
    """
    valid_tables = {"messages", "critiques", "votes", "summary"}
    if table not in valid_tables:
        table = "summary"

    output = io.StringIO()
    writer = csv.writer(output)

    if table == "messages":
        _write_messages_csv(writer, debate)
    elif table == "critiques":
        _write_critiques_csv(writer, debate)
    elif table == "votes":
        _write_votes_csv(writer, debate)
    else:
        _write_summary_csv(writer, debate)

    csv_content = output.getvalue()
    debate_id = debate.get("slug", debate.get("id", "export"))

    return ExportResult(
        content=csv_content.encode("utf-8"),
        content_type="text/csv; charset=utf-8",
        filename=f"debate-{debate_id}-{table}.csv",
    )


def _write_messages_csv(writer: CSVWriter, debate: dict) -> None:
    """Write messages timeline to CSV."""
    writer.writerow(["round", "agent", "role", "content", "timestamp"])
    for msg in debate.get("messages", []):
        writer.writerow(
            [
                msg.get("round", ""),
                msg.get("agent", ""),
                msg.get("role", ""),
                msg.get("content", "")[:1000],  # Truncate for CSV
                msg.get("timestamp", ""),
            ]
        )


def _write_critiques_csv(writer: CSVWriter, debate: dict) -> None:
    """Write critiques to CSV."""
    writer.writerow(["round", "critic", "target", "severity", "summary", "timestamp"])
    for critique in debate.get("critiques", []):
        writer.writerow(
            [
                critique.get("round", ""),
                critique.get("critic", ""),
                critique.get("target", ""),
                critique.get("severity", ""),
                critique.get("summary", "")[:500],
                critique.get("timestamp", ""),
            ]
        )


def _write_votes_csv(writer: CSVWriter, debate: dict) -> None:
    """Write votes to CSV."""
    writer.writerow(["round", "voter", "choice", "reason", "timestamp"])
    for vote in debate.get("votes", []):
        writer.writerow(
            [
                vote.get("round", ""),
                vote.get("voter", ""),
                vote.get("choice", ""),
                vote.get("reason", "")[:500],
                vote.get("timestamp", ""),
            ]
        )


def _write_summary_csv(writer: CSVWriter, debate: dict) -> None:
    """Write summary statistics to CSV."""
    writer.writerow(["field", "value"])
    writer.writerow(["debate_id", debate.get("slug", debate.get("id", ""))])
    writer.writerow(["topic", debate.get("topic", "")])
    writer.writerow(["started_at", debate.get("started_at", "")])
    writer.writerow(["ended_at", debate.get("ended_at", "")])
    writer.writerow(["rounds_used", debate.get("rounds_used", 0)])
    writer.writerow(["consensus_reached", debate.get("consensus_reached", False)])
    writer.writerow(["final_answer", debate.get("final_answer", "")[:1000]])
    writer.writerow(["message_count", len(debate.get("messages", []))])
    writer.writerow(["critique_count", len(debate.get("critiques", []))])
    writer.writerow(["vote_count", len(debate.get("votes", []))])


def format_debate_html(debate: dict) -> ExportResult:
    """Format debate as standalone HTML page.

    Args:
        debate: Debate data dictionary

    Returns:
        ExportResult with HTML content
    """
    debate_id = debate.get("slug", debate.get("id", "export"))
    topic = html.escape(debate.get("topic", "Untitled Debate"))
    messages = debate.get("messages", [])
    critiques = debate.get("critiques", [])
    consensus = debate.get("consensus_reached", False)
    final_answer = html.escape(debate.get("final_answer", "")[:2000])

    # Build message timeline HTML
    messages_html = _build_messages_html(messages)

    html_content = _build_html_template(
        topic=topic,
        messages=messages,
        critiques=critiques,
        consensus=consensus,
        final_answer=final_answer,
        messages_html=messages_html,
        debate=debate,
    )

    return ExportResult(
        content=html_content.encode("utf-8"),
        content_type="text/html; charset=utf-8",
        filename=f"debate-{debate_id}.html",
    )


def _build_messages_html(messages: list, limit: int = 50) -> str:
    """Build HTML for message timeline."""
    html_parts = []
    for msg in messages[:limit]:
        agent = html.escape(msg.get("agent", "unknown"))
        content = html.escape(msg.get("content", "")[:500])
        role = msg.get("role", "speaker")
        round_num = msg.get("round", 0)
        html_parts.append(
            f"""
        <div class="message {role}">
            <div class="message-header">
                <span class="agent">{agent}</span>
                <span class="round">Round {round_num}</span>
            </div>
            <div class="message-content">{content}</div>
        </div>"""
        )
    return "".join(html_parts)


def _build_html_template(
    topic: str,
    messages: list,
    critiques: list,
    consensus: bool,
    final_answer: str,
    messages_html: str,
    debate: dict,
) -> str:
    """Build complete HTML template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aragora Debate: {topic}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #16213e;
            padding: 15px 25px;
            border-radius: 8px;
            border: 1px solid #0f3460;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .timeline {{
            margin-top: 20px;
        }}
        .message {{
            background: #16213e;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .message.critic {{
            border-left-color: #FF5722;
        }}
        .message.judge {{
            border-left-color: #2196F3;
        }}
        .message-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        .agent {{
            font-weight: bold;
            color: #4CAF50;
        }}
        .round {{
            color: #888;
            font-size: 12px;
        }}
        .message-content {{
            line-height: 1.5;
            white-space: pre-wrap;
        }}
        .consensus {{
            background: #1b4d3e;
            border: 2px solid #4CAF50;
            padding: 20px;
            margin-top: 20px;
            border-radius: 8px;
        }}
        .consensus h2 {{
            color: #4CAF50;
            margin-top: 0;
        }}
        .no-consensus {{
            background: #4d1b1b;
            border-color: #FF5722;
        }}
        .no-consensus h2 {{
            color: #FF5722;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{topic}</h1>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(messages)}</div>
                <div class="stat-label">Messages</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(critiques)}</div>
                <div class="stat-label">Critiques</div>
            </div>
            <div class="stat">
                <div class="stat-value">{debate.get("rounds_used", 0)}</div>
                <div class="stat-label">Rounds</div>
            </div>
            <div class="stat">
                <div class="stat-value">{"Yes" if consensus else "No"}</div>
                <div class="stat-label">Consensus</div>
            </div>
        </div>

        <div class="timeline">
            <h2>Debate Timeline</h2>
            {messages_html if messages_html else "<p>No messages recorded.</p>"}
        </div>

        <div class="consensus {"" if consensus else "no-consensus"}">
            <h2>{"Final Consensus" if consensus else "No Consensus Reached"}</h2>
            <p>{final_answer if final_answer else "No final answer recorded."}</p>
        </div>

        <p style="color: #666; text-align: center; margin-top: 40px;">
            Exported from Aragora
        </p>
    </div>
</body>
</html>"""
