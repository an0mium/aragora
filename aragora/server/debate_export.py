"""
Debate export formatting utilities.

Provides CSV, HTML, TXT, and Markdown export formatters for debates.
Extracted from handlers/debates.py for better modularity.
"""

import csv
import html
import io
from dataclasses import dataclass
from typing import Any

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
        html_parts.append(f"""
        <div class="message {role}">
            <div class="message-header">
                <span class="agent">{agent}</span>
                <span class="round">Round {round_num}</span>
            </div>
            <div class="message-content">{content}</div>
        </div>""")
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


def format_debate_txt(debate: dict) -> ExportResult:
    """Format debate as plain text transcript.

    Args:
        debate: Debate data dictionary

    Returns:
        ExportResult with plain text content
    """
    debate_id = debate.get("slug", debate.get("id", "export"))
    topic = debate.get("topic", "Untitled Debate")
    messages = debate.get("messages", [])
    consensus = debate.get("consensus_reached", False)
    final_answer = debate.get("final_answer", "")
    synthesis = debate.get("synthesis", "")

    lines = []
    lines.append("=" * 70)
    lines.append("ARAGORA DEBATE TRANSCRIPT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Topic: {topic}")
    lines.append(f"Debate ID: {debate_id}")
    lines.append(f"Started: {debate.get('started_at', 'N/A')}")
    lines.append(f"Ended: {debate.get('ended_at', 'N/A')}")
    lines.append(f"Rounds: {debate.get('rounds_used', 0)}")
    lines.append(f"Consensus Reached: {'Yes' if consensus else 'No'}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("DEBATE TIMELINE")
    lines.append("-" * 70)
    lines.append("")

    current_round = -1
    for msg in messages:
        round_num = msg.get("round", 0)
        if round_num != current_round:
            current_round = round_num
            lines.append("")
            lines.append(f"--- Round {current_round} ---")
            lines.append("")

        agent = msg.get("agent", "unknown")
        role = msg.get("role", "speaker")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        lines.append(f"[{agent.upper()}] ({role})")
        if timestamp:
            lines.append(f"Time: {timestamp}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    # Add critiques section if present
    critiques = debate.get("critiques", [])
    if critiques:
        lines.append("")
        lines.append("-" * 70)
        lines.append("CRITIQUES")
        lines.append("-" * 70)
        lines.append("")
        for critique in critiques:
            critic = critique.get("critic", "unknown")
            target = critique.get("target", "unknown")
            severity = critique.get("severity", 0)
            summary = critique.get("summary", "")
            lines.append(f"{critic} -> {target} (severity: {severity})")
            lines.append(summary)
            lines.append("")

    # Add synthesis/conclusion
    lines.append("")
    lines.append("=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)
    lines.append("")

    if synthesis:
        lines.append("SYNTHESIS:")
        lines.append(synthesis)
        lines.append("")

    if final_answer:
        lines.append("FINAL ANSWER:")
        lines.append(final_answer)
    elif not synthesis:
        lines.append("No conclusion reached.")

    lines.append("")
    lines.append("-" * 70)
    lines.append("Exported from Aragora")
    lines.append("-" * 70)

    content = "\n".join(lines)

    return ExportResult(
        content=content.encode("utf-8"),
        content_type="text/plain; charset=utf-8",
        filename=f"debate-{debate_id}.txt",
    )


def format_debate_md(debate: dict) -> ExportResult:
    """Format debate as Markdown transcript.

    Args:
        debate: Debate data dictionary

    Returns:
        ExportResult with Markdown content
    """
    debate_id = debate.get("slug", debate.get("id", "export"))
    topic = debate.get("topic", "Untitled Debate")
    messages = debate.get("messages", [])
    consensus = debate.get("consensus_reached", False)
    final_answer = debate.get("final_answer", "")
    synthesis = debate.get("synthesis", "")

    lines = []
    lines.append(f"# Aragora Debate: {topic}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Debate ID:** `{debate_id}`")
    lines.append(f"- **Started:** {debate.get('started_at', 'N/A')}")
    lines.append(f"- **Ended:** {debate.get('ended_at', 'N/A')}")
    lines.append(f"- **Rounds:** {debate.get('rounds_used', 0)}")
    lines.append(f"- **Consensus Reached:** {'Yes' if consensus else 'No'}")
    lines.append(f"- **Confidence:** {debate.get('confidence', 0):.1%}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Debate Timeline")
    lines.append("")

    current_round = -1
    for msg in messages:
        round_num = msg.get("round", 0)
        if round_num != current_round:
            current_round = round_num
            lines.append(f"### Round {current_round}")
            lines.append("")

        agent = msg.get("agent", "unknown")
        role = msg.get("role", "speaker")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        # Format agent name with role badge
        role_emoji = {"speaker": "💬", "critic": "🔍", "judge": "⚖️"}.get(role, "💬")
        lines.append(f"#### {role_emoji} {agent}")
        if timestamp:
            lines.append(f"*{timestamp}*")
        lines.append("")
        lines.append(content)
        lines.append("")

    # Add critiques section if present
    critiques = debate.get("critiques", [])
    if critiques:
        lines.append("---")
        lines.append("")
        lines.append("## Critiques")
        lines.append("")
        for critique in critiques:
            critic = critique.get("critic", "unknown")
            target = critique.get("target", "unknown")
            severity = critique.get("severity", 0)
            summary = critique.get("summary", "")
            severity_bar = "🔴" if severity > 0.7 else "🟡" if severity > 0.4 else "🟢"
            lines.append(f"### {critic} → {target} {severity_bar}")
            lines.append(f"*Severity: {severity:.1%}*")
            lines.append("")
            lines.append(summary)
            lines.append("")

    # Add synthesis/conclusion
    lines.append("---")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")

    if synthesis:
        lines.append("### Synthesis")
        lines.append("")
        lines.append(synthesis)
        lines.append("")

    if final_answer:
        lines.append("### Final Answer")
        lines.append("")
        if consensus:
            lines.append("> ✅ **Consensus Reached**")
        else:
            lines.append("> ⚠️ **No Consensus**")
        lines.append("")
        lines.append(final_answer)
    elif not synthesis:
        lines.append("*No conclusion reached.*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Exported from [Aragora](https://aragora.ai)*")

    content = "\n".join(lines)

    return ExportResult(
        content=content.encode("utf-8"),
        content_type="text/markdown; charset=utf-8",
        filename=f"debate-{debate_id}.md",
    )


def format_debate_latex(debate: dict) -> ExportResult:
    """Format debate as LaTeX document.

    Creates a properly formatted LaTeX document suitable for academic papers
    or formal documentation. Uses standard article class with custom styling.

    Args:
        debate: Debate data dictionary

    Returns:
        ExportResult with LaTeX content
    """
    debate_id = debate.get("slug", debate.get("id", "export"))
    topic = _latex_escape(debate.get("topic", "Untitled Debate"))
    messages = debate.get("messages", [])
    critiques = debate.get("critiques", [])
    consensus = debate.get("consensus_reached", False)
    final_answer = _latex_escape(debate.get("final_answer", ""))
    synthesis = _latex_escape(debate.get("synthesis", ""))

    lines = []

    # Document preamble
    lines.append(r"\documentclass[11pt,a4paper]{article}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\usepackage{geometry}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{titlesec}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{fancyhdr}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append("")
    lines.append(r"\geometry{margin=1in}")
    lines.append("")
    lines.append(r"% Custom colors")
    lines.append(r"\definecolor{aragora}{RGB}{76, 175, 80}")
    lines.append(r"\definecolor{critic}{RGB}{255, 87, 34}")
    lines.append(r"\definecolor{judge}{RGB}{33, 150, 243}")
    lines.append(r"\definecolor{darkbg}{RGB}{26, 26, 46}")
    lines.append("")
    lines.append(r"% Hyperref setup")
    lines.append(r"\hypersetup{colorlinks=true,linkcolor=aragora,urlcolor=aragora}")
    lines.append("")
    lines.append(r"% Custom quote environment for agent messages")
    lines.append(r"\newenvironment{agentmsg}[2]{%")
    lines.append(r"  \par\vspace{0.5em}")
    lines.append(r"  \noindent\textbf{\textcolor{#1}{#2}}\par")
    lines.append(r"  \begin{quote}")
    lines.append(r"}{%")
    lines.append(r"  \end{quote}")
    lines.append(r"  \vspace{0.5em}")
    lines.append(r"}")
    lines.append("")
    lines.append(r"\title{Aragora Debate Transcript\\[0.5em]\large " + topic + r"}")
    lines.append(r"\author{Multi-Agent AI Debate System}")
    lines.append(r"\date{" + _latex_escape(debate.get("started_at", "")) + r"}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\maketitle")
    lines.append("")

    # Metadata section
    lines.append(r"\section*{Debate Metadata}")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Field} & \textbf{Value} \\")
    lines.append(r"\midrule")
    lines.append(f"Debate ID & \\texttt{{{_latex_escape(debate_id)}}} \\\\")
    lines.append(f"Started & {_latex_escape(debate.get('started_at', 'N/A'))} \\\\")
    lines.append(f"Ended & {_latex_escape(debate.get('ended_at', 'N/A'))} \\\\")
    lines.append(f"Rounds & {debate.get('rounds_used', 0)} \\\\")
    lines.append(f"Messages & {len(messages)} \\\\")
    lines.append(f"Critiques & {len(critiques)} \\\\")
    lines.append(f"Consensus & {'Yes' if consensus else 'No'} \\\\")
    confidence = debate.get("confidence", 0)
    lines.append(f"Confidence & {confidence:.1%} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    # Timeline section
    lines.append(r"\section{Debate Timeline}")
    lines.append("")

    current_round = -1
    for msg in messages:
        round_num = msg.get("round", 0)
        if round_num != current_round:
            current_round = round_num
            lines.append(f"\\subsection{{Round {current_round}}}")
            lines.append("")

        agent = _latex_escape(msg.get("agent", "unknown"))
        role = msg.get("role", "speaker")
        content = _latex_escape(msg.get("content", ""))

        # Color based on role
        color = {"speaker": "aragora", "critic": "critic", "judge": "judge"}.get(role, "black")
        role_label = {"speaker": "Proposer", "critic": "Critic", "judge": "Judge"}.get(
            role, role.title()
        )

        lines.append(f"\\begin{{agentmsg}}{{{color}}}{{{agent} ({role_label})}}")
        lines.append(content)
        lines.append(r"\end{agentmsg}")
        lines.append("")

    # Critiques section
    if critiques:
        lines.append(r"\section{Critiques}")
        lines.append("")
        for critique in critiques:
            critic = _latex_escape(critique.get("critic", "unknown"))
            target = _latex_escape(critique.get("target", "unknown"))
            severity = critique.get("severity", 0)
            summary = _latex_escape(critique.get("summary", ""))

            severity_text = "High" if severity > 0.7 else "Medium" if severity > 0.4 else "Low"
            lines.append(
                f"\\paragraph{{{critic} $\\rightarrow$ {target} (Severity: {severity_text})}}"
            )
            lines.append(summary)
            lines.append("")

    # Conclusion section
    lines.append(r"\section{Conclusion}")
    lines.append("")

    if synthesis:
        lines.append(r"\subsection{Synthesis}")
        lines.append(synthesis)
        lines.append("")

    if final_answer:
        lines.append(r"\subsection{Final Answer}")
        if consensus:
            lines.append(r"\textbf{\textcolor{aragora}{$\checkmark$ Consensus Reached}}")
        else:
            lines.append(r"\textbf{\textcolor{critic}{$\times$ No Consensus}}")
        lines.append("")
        lines.append(final_answer)
    elif not synthesis:
        lines.append(r"\emph{No conclusion reached.}")

    lines.append("")
    lines.append(r"\vfill")
    lines.append(r"\begin{center}")
    lines.append(
        r"\small\textit{Exported from \href{https://aragora.ai}{Aragora} -- Multi-Agent AI Debate System}"
    )
    lines.append(r"\end{center}")
    lines.append("")
    lines.append(r"\end{document}")

    content = "\n".join(lines)

    return ExportResult(
        content=content.encode("utf-8"),
        content_type="application/x-latex; charset=utf-8",
        filename=f"debate-{debate_id}.tex",
    )


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in text.

    Args:
        text: Raw text to escape

    Returns:
        Text with LaTeX special characters escaped
    """
    if not text:
        return ""

    # Order matters: backslash must be first
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("<", r"\textless{}"),
        (">", r"\textgreater{}"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text
